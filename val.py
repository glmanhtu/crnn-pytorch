from __future__ import print_function
from __future__ import division

import argparse
import csv
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
import os
import utils
import dataset

import models.crnn as net
import params
from augmentation import GridDistortion
from error_rates import cer, jaccard_similarity
from imgaug import augmenters as iaa
import imgaug as ia

parser = argparse.ArgumentParser()
parser.add_argument('-val', '--valroot', required=True, help='path to val dataset')
parser.add_argument('-out', '--outfile', required=True, help='path to csv output file')
args = parser.parse_args()

if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

# ensure everytime the random is the same
random.seed(params.manualSeed)
np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)

cudnn.benchmark = True


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.35, iaa.GaussianBlur(sigma=(0, 1.5))),
            iaa.Sometimes(0.35,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.05)),
                                     iaa.CoarseDropout(0, size_percent=0.05)])),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")

# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""
def data_loader():

    # val
    transform = torchvision.transforms.Compose([dataset.resizeNormalize((params.imgW, params.imgH))])
    val_dataset = dataset.lmdbDataset(root=args.valroot, transform=transform)
    assert val_dataset
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    
    return val_loader

val_loader = data_loader()

# -----------------------------------------------
"""
In this block
    Net init
    Weight init
    Load pretrained model
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def net_init():
    nclass = len(params.alphabet) + 1
    crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
    crnn.apply(weights_init)
    if params.pretrained != '':
        print('loading pretrained model from %s' % params.pretrained)
        if params.multi_gpu:
            crnn = torch.nn.DataParallel(crnn)
        std = torch.load(params.pretrained)

        # # Remove the last FC layer
        # std.popitem(last=True)
        # std.popitem(last=True)
        crnn.load_state_dict(std, strict=False)
    
    return crnn

crnn = net_init()
print(crnn)

# -----------------------------------------------
"""
In this block
    Init some utils defined in utils.py
"""
# Compute average for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.averager()

# Convert between str and label.
converter = utils.strLabelConverter(params.alphabet)

# -----------------------------------------------
"""
In this block
    criterion define
"""
criterion = CTCLoss()

# -----------------------------------------------
"""
In this block
    Init some tensor
    Put tensor and net on cuda
    NOTE:
        image, text, length is used by both val and train
        becaues train and val will never use it at the same time.
"""
image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

    crnn = crnn.cuda()
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(params.ngpu))

image = Variable(image)
text = Variable(text)
length = Variable(length)

# -----------------------------------------------
"""
In this block
    Setup optimizer
"""
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

# -----------------------------------------------
"""
In this block
    Dealwith lossnan
    NOTE:
        I use different way to dealwith loss nan according to the torch version. 
"""
if params.dealwith_lossnan:
    if torch.__version__ >= '1.1.0':
        """
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.
        Pytorch add this param after v1.1.0 
        """
        criterion = CTCLoss(zero_infinity = True)
    else:
        """
        only when
            torch.__version__ < '1.1.0'
        we use this way to change the inf to zero
        """
        crnn.register_backward_hook(crnn.backward_hook)

# -----------------------------------------------

def val(net, criterion):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    similarity = 0
    distances = 0
    count = 0.0
    loss_avg = utils.averager() # The blobal loss_avg is used by train

    max_iter = len(val_loader)
    all_predicts = []
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for i in cpu_texts:
            cpu_texts_decode.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, cpu_texts_decode):
            if pred == target:
                n_correct += 1
            simr = jaccard_similarity([x for x in pred], [x for x in target])
            distance = cer(pred, target)
            all_predicts.append({'pred': pred, 'actual': target, 'similarity': simr, 'distant': distance})
            similarity += simr
            distances += distance
            count += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / count
    similarity = similarity / count
    distance = distances / count
    print('Val loss: %f, accuracy: %f, similarity: %f, distance: %f' % (loss_avg.val(), accuracy, similarity, distance))
    return accuracy, all_predicts


if __name__ == "__main__":
    best_acc = 0
    acc, predicts = val(crnn, criterion)
    with open(args.outfile, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=predicts[0].keys())
        writer.writeheader()
        writer.writerows(predicts)
