from __future__ import print_function
import csv
import glob
import os

from shutil import copy
from PIL import Image

path_to_images = 'D:/datasets/re_id/MiniMarket1501'


# # Resize images
#
# for i, image_file in enumerate(all_images):
#     im = Image.open(image_file)
#     im = im.resize((84, 84), resample=Image.LANCZOS)
#     im.save(image_file)
#     if i % 500 == 0:
#         print(i)

# Put in correct directory
for datatype in ['train', 'query', 'test']:
    # os.system('mkdir ' + datatype)
    cur_dir = os.path.join(path_to_images, datatype)
    os.makedirs(cur_dir, exist_ok=True)
    with open(os.path.join(path_to_images, datatype + '.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        last_label = ''
        for i, row in enumerate(reader):
            if i == 0:  # skip the headers
                continue
            label = row[1]
            image_name = row[0]
            if label != last_label:
                save_dir = os.path.join(path_to_images, datatype + '_cvt', label)
                # os.system('mkdir ' + cur_dir)
                os.makedirs(save_dir, exist_ok=True)
                last_label = label

            # os.system('cp images/' + image_name + ' ' + cur_dir)
            # print((os.path.join(cur_dir, image_name), os.path.join(save_dir, image_name)))
            copy(os.path.join(cur_dir, image_name), os.path.join(save_dir, image_name))
