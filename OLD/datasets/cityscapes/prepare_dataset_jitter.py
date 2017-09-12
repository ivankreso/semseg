import os
from os.path import join

import pickle
import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.data
import skimage.transform
from tqdm import trange

import data_utils

np.set_printoptions(linewidth=250)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir',
    '/home/kivan/datasets/Cityscapes/tensorflow/jitter', '')
tf.app.flags.DEFINE_integer('num_classes', 19, '')


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(img, label_map, class_hist, num_labels, img_name, save_dir):
  height = img.shape[0]
  width = img.shape[1]
  channels = img.shape[2]

  filename = join(save_dir, img_name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  img_str = img.tostring()
  labels_str = label_map.tostring()
  class_hist_str = class_hist.tostring()
  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'channels': _int64_feature(channels),
      'num_labels': _int64_feature(int(num_labels)),
      'img_name': _bytes_feature(img_name.encode()),
      'image': _bytes_feature(img_str),
      'class_hist': _bytes_feature(class_hist_str),
      'labels': _bytes_feature(labels_str),
      }))
  writer.write(example.SerializeToString())
  writer.close()


def prepare_dataset_jitter():
  name = 'jitter'
  #classes = ['fence', 'wall', 'motorcycle']
  root_dir = '/home/kivan/datasets/Cityscapes/crops/crops170314classres'
  save_dir = FLAGS.save_dir
  print('Save dir = ', save_dir)
  os.makedirs(save_dir, exist_ok=True)
  file_list = next(os.walk(root_dir))[2]
  img_list = []
  for img_name in file_list:
    if img_name.find('labelIds') < 0:
      img_list.append(img_name)
  for i in trange(len(img_list)):
    img_name = img_list[i]
    #print(img_name)
    if img_name.find('ppm') < 0:
      continue
    #for cname in classes:
    #  if img_name.find(cname) >= 0:
    #    break
    #else:
    #  continue

    #print(img_name)
    rgb_path = join(root_dir, img_name)
    rgb = ski.data.load(rgb_path)
    rgb = rgb.astype(np.uint8)
    img_prefix = img_name[:-4]
    gt_path = join(root_dir, img_prefix + '_gtFine_labelIds.png')
    #print(gt_path)
    full_gt_img = ski.data.load(gt_path)
    gt_img, _ = data_utils.convert_ids(full_gt_img, has_hood=False)
    gt_img = gt_img.astype(np.int8)
    #print((gt_img == -1).sum())
    class_hist, num_labels = data_utils.get_class_hist(gt_img, FLAGS.num_classes)
    create_tfrecord(rgb, gt_img, class_hist, num_labels, img_prefix, save_dir)


def main(argv):
  prepare_dataset_jitter()
  #prepare_dataset('train')
  #prepare_dataset('val')


if __name__ == '__main__':
  tf.app.run()
