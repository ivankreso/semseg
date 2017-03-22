import os
from os.path import join

import numpy as np
from tqdm import trange
import skimage as ski
import skimage.data, skimage.transform

#def merge_and_save(img_name, main_dir, hood_dir, save_dir):
#  main_img = ski.data.load(join(main_dir, img_name))
#  full_img = ski.data.load(join(hood_dir, img_name))
#  cy = 896
#  full_img[:cy,...] = main_img
#  save_path = join(save_dir, img_name)
#  ski.io.imsave(save_path, full_img)

def merge_and_save(img_name, main_dir, hood_dir, save_dir):
  main_img = ski.data.load(join(main_dir, img_name))
  h = 1024
  w = 2048
  shape = list(main_img.shape)
  shape[0] = h
  shape[1] = w
  full_img = np.zeros(shape, dtype=main_img.dtype)
  full_img.fill(7)
  cy = 896
  full_img[:cy,...] = main_img
  save_path = join(save_dir, img_name)
  ski.io.imsave(save_path, full_img)


main_dir = '/home/kivan/datasets/results/out/cityscapes/main'
hood_dir = '/home/kivan/datasets/results/out/cityscapes/hood'
save_dir = '/home/kivan/datasets/results/out/cityscapes/submit'


os.makedirs(join(save_dir, 'color'), exist_ok=True)
os.makedirs(join(save_dir, 'labels'), exist_ok=True)
filelist = next(os.walk(join(main_dir, 'labels')))[2]
for i in trange(len(filelist)):
  merge_and_save(filelist[i], join(main_dir, 'color'),
                 join(hood_dir, 'color'), join(save_dir, 'color'))
  merge_and_save(filelist[i], join(main_dir, 'labels'),
                 join(hood_dir, 'labels'), join(save_dir, 'labels'))
