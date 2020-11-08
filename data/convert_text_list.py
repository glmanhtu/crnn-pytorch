import os
import sys

with open('char_std_5990.txt') as fd:
    cvt_lines = fd.readlines()

cvt_dict = {}
for i, line in enumerate(cvt_lines):
    key = i + 1
    value = line.strip()
    cvt_dict[key] = value

if __name__ == "__main__":
    cvt_fpath = sys.argv[1]
    out_fpath = sys.argv[2]
    base_dir = os.path.dirname(cvt_fpath)
    result = []

    with open(cvt_fpath) as fd:
        lines = fd.readlines()

    for line in lines:
        line_split = line.strip().split()
        img_path = line_split[0]
        img_full_path = os.path.join(base_dir, 'images', img_path)
        assert os.path.isfile(img_full_path)
        result.append(img_full_path)
        label = ''
        for i in line_split[1:]:
            label += cvt_dict[int(i)]
        result.append(label)
    with open(out_fpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
