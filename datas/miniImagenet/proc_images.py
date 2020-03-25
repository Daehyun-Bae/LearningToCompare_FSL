"""
code copied from https://github.com/cbfinn/maml/blob/master/data/miniImagenet/proc_images.py
Script for converting from csv file datafiles to a directory for each image (which is how it is loaded by MAML code)

Acquire miniImagenet from Ravi & Larochelle '17, along with the train, val, and test csv files. Put the
csv files in the miniImagenet directory and put the images in the directory 'miniImagenet/images/'.
Then run this script from the miniImagenet directory:
    cd data/miniImagenet/
    python proc_images.py
"""

from __future__ import print_function
import csv
import glob
import os

from shutil import copy
from PIL import Image

path_to_images = 'mini_imagenet_convert/'

all_images = glob.glob(path_to_images + '*')

# Resize images

for i, image_file in enumerate(all_images):
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
    if i % 500 == 0:
        print(i)

# Put in correct directory
for datatype in ['train', 'val', 'test']:
    # os.system('mkdir ' + datatype)
    os.makedirs(datatype, exist_ok=True)
    with open(datatype + '_cvt.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        last_label = ''
        for i, row in enumerate(reader):
            if i == 0:  # skip the headers
                continue
            label = row[1]
            image_name = row[0]
            if label != last_label:
                cur_dir = datatype + '/' + label + '/'
                # os.system('mkdir ' + cur_dir)
                os.makedirs(cur_dir, exist_ok=True)
                last_label = label
            # os.system('cp images/' + image_name + ' ' + cur_dir)
            copy(os.path.join(path_to_images, image_name), os.path.join(cur_dir, image_name))
