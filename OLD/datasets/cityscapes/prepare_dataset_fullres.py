import sys
import os
import pickle
from os.path import join

import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.data
from tqdm import trange
#import cv2

import data_utils

FLAGS = tf.app.flags.FLAGS
IMG_MEAN = [75, 85, 75]

# Basic model parameters.
flags = tf.app.flags
flags.DEFINE_string('data_dir',
    '/home/kivan/datasets/Cityscapes/2048x1024/', 'Dataset dir')
    #'/home/kivan/datasets/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/', 'Dataset dir')
tf.app.flags.DEFINE_string('gt_dir',
    '/home/kivan/datasets/Cityscapes/orig/gtFine/', '')
flags.DEFINE_integer('img_width', 2048, '')
flags.DEFINE_integer('img_height', 1024, '')
#flags.DEFINE_integer('rf_half_size', 100, '')
flags.DEFINE_integer('rf_half_size', 128, '')
flags.DEFINE_string('save_dir',
    '/home/kivan/datasets/Cityscapes/tensorflow/' + str(FLAGS.img_width) +
    'x' + str(FLAGS.img_height) + '_nohood/', '')
tf.app.flags.DEFINE_integer('num_classes', 19, '')

#flags.DEFINE_integer('cx_start', 0, '')
#flags.DEFINE_integer('cx_end', 2048, '')
#flags.DEFINE_integer('cy_start', 30, '')
#flags.DEFINE_integer('cy_end', 894, '')
flags.DEFINE_integer('cy_start', 0, '')
flags.DEFINE_integer('cy_end', 896, '')
FLAGS = flags.FLAGS

def crop_data_split(img, left_end, right_start, clear_overlap=False, fill_val=None):
  img_left = np.ascontiguousarray(img[:,:left_end,...])
  img_right = np.ascontiguousarray(img[:,right_start:,...])
  if clear_overlap:
    half_width = FLAGS.img_width // 2
    img_left[:,half_width:] = fill_val
    img_right[:,:FLAGS.rf_half_size] = fill_val
  return [img_left, img_right]


def crop_data(img):
  img = np.ascontiguousarray(img[FLAGS.cy_start:FLAGS.cy_end,...])
  return [img]

#def crop_data(img):
#  return [img]


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(img, label_map, class_hist, depth_img,
                    num_labels, img_name, save_dir):
  height = img.shape[0]
  width = img.shape[1]
  channels = img.shape[2]

  filename = join(save_dir + img_name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  img_str = img.tostring()
  labels_str = label_map.tostring()
  class_hist_str = class_hist.tostring()
  depth_raw = depth_img.tostring()
  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'channels': _int64_feature(channels),
      'num_labels': _int64_feature(int(num_labels)),
      'img_name': _bytes_feature(img_name.encode()),
      'image': _bytes_feature(img_str),
      'class_hist': _bytes_feature(class_hist_str),
      'labels': _bytes_feature(labels_str),
      'depth': _bytes_feature(depth_raw)
      }))
  writer.write(example.SerializeToString())
  writer.close()


def prepare_dataset(name):
  print('Preparing ' + name)
  root_dir = join(FLAGS.data_dir, 'rgb', name)
  depth_dir = join(FLAGS.data_dir, 'depth', name)
  gt_dir = join(FLAGS.gt_dir, name)
  cities = next(os.walk(root_dir))[1]
  save_dir = FLAGS.save_dir + name + '/'
  gt_save_dir = join(FLAGS.save_dir, 'GT', name)
  print('Save dir = ', save_dir)
  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(gt_save_dir, exist_ok=True)
  os.makedirs(join(gt_save_dir, 'label'), exist_ok=True)
  os.makedirs(join(gt_save_dir, 'instance'), exist_ok=True)
  #print('Writing', filename)
  half_width = int(FLAGS.img_width / 2)
  left_x_end = int(half_width + FLAGS.rf_half_size)
  right_x_start = int(half_width - FLAGS.rf_half_size)
  #mean_sum = np.zeros(3, dtype=np.float64)
  #std_sum = np.zeros(3, dtype=np.float64)
  mean_sum = np.zeros(1, dtype=np.float64)
  std_sum = np.zeros(1, dtype=np.float64)
  img_cnt = 0
  for city in cities:
    print(city)
    img_list = next(os.walk(join(root_dir, city)))[2]
    #for img_name in img_list:
    for i in trange(len(img_list)):
      img_cnt += 1
      img_name = img_list[i]
      img_prefix = img_name[:-4]
      img_path = join(root_dir, city, img_name)
      img = ski.data.load(img_path)
      #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      depth_path = join(depth_dir, city, img_prefix + '_leftImg8bit.png')
      depth = ski.data.load(depth_path)
      depth = np.round(depth / 256.0).astype(np.uint8)

      # compute mean
      #mean_sum += rgb.mean((0,1))
      #std_sum += rgb.std((0,1))
      #mean_sum += depth.mean()
      #std_sum += depth.std()
      #print('mean = ', mean_sum / img_cnt)
      #print('std = ', std_sum / img_cnt)
      #continue

      gt_path = join(gt_dir, city, img_prefix + '_gtFine_labelIds.png')
      #print(gt_path)
      orig_gt_img = ski.data.load(gt_path)
      #full_gt_img = np.ascontiguousarray(full_gt_img[cy_start:cy_end,cx_start:cx_end])
      instance_gt_path = join(gt_dir, city, img_prefix + '_gtFine_instanceIds.png')
      instance_gt_img = ski.data.load(instance_gt_path)
      #instance_gt_img = np.ascontiguousarray(instance_gt_img[cy_start:cy_end,cx_start:cx_end])
      gt_img, car_mask = data_utils.convert_ids(orig_gt_img)

      #img[car_mask] = 0
      img[car_mask] = IMG_MEAN

      #ski.io.imsave(join('/home/kivan/datasets/Cityscapes/tensorflow/tmp/',
      #              img_prefix + '.png'), img)
      gt_img = gt_img.astype(np.int8)
      gt_img[gt_img == -1] = FLAGS.num_classes

      #weights, num_labels = data_utils.get_class_weights(gt_img)


      #orig_gt_crops = crop_data(orig_gt_img, left_x_end, right_x_start)
      #instance_gt_crops = crop_data(instance_gt_img, left_x_end, right_x_start)
      #rgb_crops = crop_data(rgb, left_x_end, right_x_start)
      #gt_crops = crop_data(gt_img, left_x_end, right_x_start,
      #                     clear_overlap=True, fill_val=-1)
      #weights_crops = crop_data(weights, left_x_end, right_x_start,
      #                          clear_overlap=True, fill_val=0)
      orig_gt_crops = crop_data(orig_gt_img)
      instance_gt_crops = crop_data(instance_gt_img)
      img_crops = crop_data(img)
      depth_crops = crop_data(depth)
      gt_crops = crop_data(gt_img)
      #weights_crops = crop_data(weights)

      for i in range(len(img_crops)):
        class_hist, num_labels = data_utils.get_class_hist(gt_crops[i], FLAGS.num_classes)
        img_name = img_prefix + '_' + str(i)
        create_tfrecord(img_crops[i], gt_crops[i], class_hist,
                        depth_crops[i], num_labels, img_name, save_dir)
        if name == 'val':
          ski.io.imsave(join(gt_save_dir, 'label', img_name + '.png'),
                        orig_gt_crops[i])
          ski.io.imsave(join(gt_save_dir, 'instance', img_name + '.png'),
                        instance_gt_crops[i])


def main(argv):
  prepare_dataset('val')
  prepare_dataset('train')


if __name__ == '__main__':
  tf.app.run()
