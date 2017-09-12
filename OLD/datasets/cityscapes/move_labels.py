import os
from os.path import join

import PIL.Image as pimg
import numpy as np
from tqdm import trange

import data_utils

data_dir = '/home/kivan/datasets/Cityscapes/orig/gtFine'
save_dir = '/home/kivan/datasets/Cityscapes/2048x1024/labels'


def prepare_data(name):
  root_dir = join(data_dir, name)
  cities = next(os.walk(root_dir))[1]
  for city in cities:
    print(city)
    city_dir = join(root_dir, city)
    image_list = next(os.walk(city_dir))[2]
    image_list = [x for x in image_list if x.find('labelIds') >= 0]
    city_save_dir = join(save_dir, name, city)
    os.makedirs(city_save_dir, exist_ok=True)
    for i in trange(len(image_list)):
      img = np.array(pimg.open(join(city_dir, image_list[i])))
      img, _ = data_utils.convert_ids(img, ignore_id=19)
      img_prefix = image_list[i][:-20]

      save_path = join(city_save_dir, img_prefix + '.png')
      img = pimg.fromarray(img)
      img.save(save_path)

prepare_data('train')
prepare_data('val')
