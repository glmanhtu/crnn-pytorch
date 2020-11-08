import alphabets

# about data and net
import lvt_alphabets

alphabet = lvt_alphabets.alphabet
keep_ratio = False # whether to keep ratio for image resize
manualSeed = 1234 # reproduce experiemnt
random_sample = True # whether to sample the dataset with random sampler
imgH = 48 # the height of the input image to network
imgW = 432 # the width of the input image to network
nh = 256 # size of the lstm hidden state
nc = 3
pretrained = '' # path to pretrained model (to continue training)
expr_dir = 'expr' # where to store samples and models
dealwith_lossnan = False # whether to replace all nan/inf in gradients to zero

# hardware
cuda = True # enables cuda
multi_gpu = False # whether to use multi gpu
ngpu = 1 # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 10 # number of data loading workers

# training process
displayInterval = 50 # interval to be print the train loss
valInterval = 50 # interval to val the model loss and accuray
saveInterval = 50 # interval to save model
n_val_disp = 10 # number of samples to display when val the model

# finetune
nepoch = 1500 # number of epochs to train for
batchSize = 64 # input batch size
lr = 0.0001 # learning rate for Critic, not used by adadealta
beta1 = 0.5 # beta1 for adam. default=0.5
adam = True # whether to use adam (default is rmsprop)
adadelta = False # whether to use adadelta (default is rmsprop)
