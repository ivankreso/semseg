import sys
sys.path.append('../..')

import os
import pickle
import numpy as np
import tensorflow as tf
#from pgmagick import Image
import skimage as ski
import skimage.data
from tqdm import trange

FLAGS = tf.app.flags.FLAGS

VGG_MEAN = [123.68, 116.779, 103.939]

# Basic model parameters.
flags = tf.app.flags
flags.DEFINE_string('data_dir',
    '/home/kivan/datasets/Cityscapes/2048x1024/', 'Dataset dir')
    #'/home/kivan/datasets/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/', 'Dataset dir')
flags.DEFINE_integer('img_width', 2048, '')
flags.DEFINE_integer('img_height', 1024, '')
flags.DEFINE_integer('rf_half_size', 100, '')
flags.DEFINE_string('save_dir',
    '/home/kivan/datasets/Cityscapes/tensorflow/' + str(FLAGS.img_width) +
    'x' + str(FLAGS.img_height) + '/', '')

#flags.DEFINE_integer('cx_start', 0, '')
#flags.DEFINE_integer('cx_end', 2048, '')
#flags.DEFINE_integer('cy_start', 30, '')
#flags.DEFINE_integer('cy_end', 894, '')
FLAGS = flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord(rgb, label_map, weight_map, num_labels, img_name, save_dir):
  #for c in range(3):
  #  rgb[:,:,c] -= rgb[:,:,c].mean()
  #  rgb[:,:,c] /= rgb[:,:,c].std()
  for c in range(3):
    rgb[:,:,c] -= VGG_MEAN[c]

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
  root_dir = FLAGS.data_dir + '/rgb/' + name + '/'
  gt_dir = FLAGS.data_dir + '/gt_data/' + name + '/'
  cities = next(os.walk(root_dir))[1]
  save_dir = FLAGS.save_dir + name + '/'
  print('Save dir = ', save_dir)
  os.makedirs(save_dir, exist_ok=True)
  #print('Writing', filename)
  half_width = int(FLAGS.img_width / 2)
  left_x_end = int(half_width + FLAGS.rf_half_size)
  right_x_start = int(half_width - FLAGS.rf_half_size)
  for city in cities:
    print(city)
    img_list = next(os.walk(root_dir + city))[2]
    #for img_name in img_list:
    for i in trange(len(img_list)):
      img_name = img_list[i]
      img_prefix = img_name[:-4]
      rgb_path = root_dir + city + '/' + img_name
      rgb = ski.data.load(rgb_path).astype(np.float32)
      #gt_path = gt_dir + city + '/' + img_name
      #gt_rgb = ski.data.load(gt_path).astype(np.uint8)
      gt_path = gt_dir + city + '/' + img_prefix + '.pickle'
      with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
      #gt_rgb = gt_rgb.astype(np.uint8)
      gt_map = gt_data[0]
      gt_weights = gt_data[1]
      num_labels = gt_data[2]
      assert(num_labels == (gt_map < 255).sum())

      rgb_left = np.ascontiguousarray(rgb[:,:left_x_end,:])
      rgb_right = np.ascontiguousarray(rgb[:,right_x_start:,:])
      gt_left = np.ascontiguousarray(gt_map[:,:left_x_end])
      gt_right = np.ascontiguousarray(gt_map[:,right_x_start:])
      gt_left[:,half_width:] = -1
      gt_right[:,:FLAGS.rf_half_size] = -1
      weights_left = np.ascontiguousarray(gt_weights[:,:left_x_end])
      weights_right = np.ascontiguousarray(gt_weights[:,right_x_start:])
      left_num_labels = (gt_left >= 0).sum()
      right_num_labels = (gt_right >= 0).sum()
      create_tfrecord(rgb_left, gt_left, weights_left, left_num_labels, img_prefix + '_left',
          save_dir)
      create_tfrecord(rgb_right, gt_right, weights_right, right_num_labels, img_prefix + '_right',
          save_dir)
      #print('left = ', rgb_left.shape)
      #print('right = ', rgb_right.shape)

      #create_tfrecord(rgb, gt_map, gt_weights, num_labels, img_prefix,
      #    save_dir)



def main(argv):
  prepare_dataset('train')
  prepare_dataset('val')


if __name__ == '__main__':
  tf.app.run()
