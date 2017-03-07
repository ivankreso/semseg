import sys
import os
import pickle
from os.path import join

import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.data
from tqdm import trange
import cv2

import data_utils

FLAGS = tf.app.flags.FLAGS


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
    'x' + str(FLAGS.img_height) + '_full/', '')

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

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord(rgb, label_map, weight_map, num_labels, img_name, save_dir):
  rows = rgb.shape[0]
  cols = rgb.shape[1]
  depth = rgb.shape[2]

  filename = os.path.join(save_dir + img_name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  rgb_raw = rgb.tostring()
  labels_raw = label_map.tostring()
  weights_str = weight_map.tostring()
  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(rows),
      'width': _int64_feature(cols),
      'depth': _int64_feature(depth),
      'num_labels': _int64_feature(int(num_labels)),
      'img_name': _bytes_feature(img_name.encode()),
      'rgb': _bytes_feature(rgb_raw),
      'label_weights': _bytes_feature(weights_str),
      'labels': _bytes_feature(labels_raw)}))
      #'disparity': _bytes_feature(disp_raw),
  writer.write(example.SerializeToString())
  writer.close()


def prepare_dataset(name):
  #rgb_means = [123.68, 116.779, 103.939]
  print('Preparing ' + name)
  root_dir = join(FLAGS.data_dir, 'rgb', name)
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
  img_cnt = 0
  for city in cities:
    print(city)
    img_list = next(os.walk(join(root_dir, city)))[2]
    #for img_name in img_list:
    for i in trange(len(img_list)):
      img_cnt += 1
      img_name = img_list[i]
      img_prefix = img_name[:-4]
      rgb_path = join(root_dir, city, img_name)
      #rgb = ski.data.load(rgb_path)
      rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)

      # compute mean
      #mean_sum += rgb.mean((0,1))
      #std_sum += rgb.std((0,1))
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
      gt_img = data_utils.convert_ids(orig_gt_img)

      gt_img = gt_img.astype(np.int8)
      weights, num_labels = data_utils.get_class_weights(gt_img)


      #orig_gt_crops = crop_data(orig_gt_img, left_x_end, right_x_start)
      #instance_gt_crops = crop_data(instance_gt_img, left_x_end, right_x_start)
      #rgb_crops = crop_data(rgb, left_x_end, right_x_start)
      #gt_crops = crop_data(gt_img, left_x_end, right_x_start,
      #                     clear_overlap=True, fill_val=-1)
      #weights_crops = crop_data(weights, left_x_end, right_x_start,
      #                          clear_overlap=True, fill_val=0)
      orig_gt_crops = crop_data(orig_gt_img)
      instance_gt_crops = crop_data(instance_gt_img)
      rgb_crops = crop_data(rgb)
      gt_crops = crop_data(gt_img)
      weights_crops = crop_data(weights)

      for i in range(len(rgb_crops)):
        img_name = img_prefix + '_' + str(i)
        create_tfrecord(rgb_crops[i], gt_crops[i], weights_crops[i],
                        num_labels, img_name, save_dir)
        ski.io.imsave(join(gt_save_dir, 'label', img_name + '.png'),
                      orig_gt_crops[i])
        ski.io.imsave(join(gt_save_dir, 'instance', img_name + '.png'),
                      instance_gt_crops[i])


def main(argv):
  prepare_dataset('val')
  prepare_dataset('train')


if __name__ == '__main__':
  tf.app.run()
