import sys
sys.path.append('../..')

import os
import numpy as np
import tensorflow as tf
#from pgmagick import Image
import skimage as ski
import skimage.data, skimage.transform
from tqdm import trange
from cityscapes_info import class_info, class_color_map
from datasets.dataset_helper import convert_colors_to_indices

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
flags = tf.app.flags
flags.DEFINE_string('data_dir',
    '/home/kivan/datasets/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/', 'Dataset dir')
flags.DEFINE_string('gt_dir',
    '/home/kivan/datasets/Cityscapes/gtFine_trainvaltest/gtFine_19/', 'Dataset dir')
flags.DEFINE_string('save_dir',
    '/home/kivan/datasets/Cityscapes/ppm/', '')

FLAGS = flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def prepare_dataset(name):
  print('Preparing ' + name)
  root_dir = FLAGS.data_dir + name + '/'
  gt_dir = FLAGS.gt_dir + name + '/'
  cities = next(os.walk(root_dir))[1]
  for city in cities:
    rgb_save_dir = FLAGS.save_dir + '/rgb/' + name + '/' + city + '/'
    gt_save_dir = FLAGS.save_dir + '/gt/' + name + '/' + city + '/'
    os.makedirs(rgb_save_dir, exist_ok=True)
    os.makedirs(gt_save_dir, exist_ok=True)
    print(city)
    img_list = next(os.walk(root_dir + city))[2]
    #for img_name in img_list:
    for i in trange(len(img_list)):
      img_name = img_list[i]
      img_prefix = img_name[:-16]
      rgb_path = root_dir + city + '/' + img_name
      rgb = ski.data.load(rgb_path)
      gt_path = gt_dir + city + '/' + img_prefix + '_gtFine_color.png'
      gt_img = ski.data.load(gt_path)
      ski.io.imsave(rgb_save_dir + img_prefix + '.ppm', rgb)
      ski.io.imsave(gt_save_dir + img_prefix + '.ppm', gt_img)


def main(argv):
  prepare_dataset('train')
  prepare_dataset('val')


if __name__ == '__main__':
  tf.app.run()
