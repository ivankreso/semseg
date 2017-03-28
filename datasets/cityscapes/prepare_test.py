import os
from os.path import join

import numpy as np
from tqdm import trange
import skimage as ski
import skimage.data, skimage.transform

data_dir = '/home/kivan/datasets/Cityscapes/orig/test'
labels_dir = '/home/kivan/datasets/Cityscapes/orig/gtFine/test'
#save_dir = '/home/kivan/datasets/Cityscapes/masked/croped/test'
save_dir = '/home/kivan/datasets/Cityscapes/masked/mean/full/test'


rgb_mean =  [75, 85, 75]
cities = next(os.walk(data_dir))[1]
for city in cities:
  city_dir = join(data_dir, city)
  image_list = next(os.walk(city_dir))[2]
  print(city)
  os.makedirs(join(save_dir, city), exist_ok=True)
  for i in trange(len(image_list)):
    img = ski.data.load(join(city_dir, image_list[i]))
    img_prefix = image_list[i][:-16]
    mask_path = join(labels_dir, city, img_prefix + '_gtFine_labelIds.png')
    mask_img = ski.data.load(mask_path)
    img[mask_img==1] = 0
    img[mask_img==2] = 0
    img[mask_img==3] = 0

    img[mask_img==1] = rgb_mean
    height = img.shape[0]
    img[height-5:,...] = rgb_mean
    #img[mask_img==2] = rgb_mean
    ##img[mask_img==3] = rgb_mean

    #img = np.ascontiguousarray(img[:896,...])
    save_path = join(save_dir, city, image_list[i])
    ski.io.imsave(save_path, img)
