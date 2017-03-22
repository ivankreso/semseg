import os
from os.path import join

import pickle
import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.data
import skimage.transform
#import cv2
from tqdm import trange

import data_utils

IMG_MEAN = [75, 85, 75]
np.set_printoptions(linewidth=250)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir',
    '/home/kivan/datasets/Cityscapes/2048x1024/', 'Dataset dir')
tf.app.flags.DEFINE_string('gt_dir',
    '/home/kivan/datasets/Cityscapes/orig/gtFine/', '')
tf.app.flags.DEFINE_integer('num_classes', 19, '')

#tf.app.flags.DEFINE_integer('img_width', 320, '')
#tf.app.flags.DEFINE_integer('img_height', 144, '')
#tf.app.flags.DEFINE_integer('img_width', 1024, '')
#tf.app.flags.DEFINE_integer('img_height', 448, '')
# leave out the car hood
tf.app.flags.DEFINE_integer('cx_start', 0, '')
tf.app.flags.DEFINE_integer('cx_end', 2048, '')
#tf.app.flags.DEFINE_integer('cy_start', 0, '')
#tf.app.flags.DEFINE_integer('cy_end', 900, '')
#tf.app.flags.DEFINE_integer('img_width', 640, '')
#tf.app.flags.DEFINE_integer('img_height', 288, '')

tf.app.flags.DEFINE_integer('cy_start', 30, '')
tf.app.flags.DEFINE_integer('cy_end', 900, '')
tf.app.flags.DEFINE_integer('img_width', 768, '')
tf.app.flags.DEFINE_integer('img_height', 320, '')

#tf.app.flags.DEFINE_integer('img_width', 1024, '')
#tf.app.flags.DEFINE_integer('img_height', 448, '')

#tf.app.flags.DEFINE_integer('img_width', 640, '')
#tf.app.flags.DEFINE_integer('img_height', 272, '')
#tf.app.flags.DEFINE_integer('img_width', 384, '')
#tf.app.flags.DEFINE_integer('img_height', 164, '')
tf.app.flags.DEFINE_boolean('downsample', True, '')
#tf.app.flags.DEFINE_integer('img_width', 1600, '')
#tf.app.flags.DEFINE_integer('img_height', 680, '')

# depth
#tf.app.flags.DEFINE_integer('img_width', 640, '')
#tf.app.flags.DEFINE_integer('cx_start', 120, '')
#tf.app.flags.DEFINE_integer('img_height', 298, '')
#tf.app.flags.DEFINE_integer('img_width', 480, '')
##tf.app.flags.DEFINE_integer('img_height', 208, '')
#tf.app.flags.DEFINE_integer('img_height', 224, '')

tf.app.flags.DEFINE_string('save_dir',
    '/home/kivan/datasets/Cityscapes/tensorflow/' +
    '{}x{}'.format(FLAGS.img_width, FLAGS.img_height) + '/', '')
    #'{}x{}'.format(FLAGS.img_width, FLAGS.img_height) + '_rgbd/', '')

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
  height = FLAGS.img_height
  width = FLAGS.img_width
  root_dir = FLAGS.data_dir + '/rgb/' + name + '/'
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
  cx_start = FLAGS.cx_start
  cx_end = FLAGS.cx_end
  cy_start = FLAGS.cy_start
  cy_end = FLAGS.cy_end
  img_cnt = 0
  depth_sum = np.zeros((FLAGS.img_height, FLAGS.img_width))
  for city in cities:
    print(city)
    img_list = next(os.walk(root_dir + city))[2]
    for i in trange(len(img_list)):
      img_cnt += 1
      img_name = img_list[i]
      img_prefix = img_name[:-4]
      rgb_path = root_dir + city + '/' + img_name
      rgb = ski.data.load(rgb_path)
      #rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
      rgb = np.ascontiguousarray(rgb[cy_start:cy_end,cx_start:cx_end,:])
      #rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
      rgb = ski.transform.resize(rgb, (height, width), preserve_range=True, order=3)
      rgb = rgb.astype(np.uint8)
      #depth_img = None
      depth_path = join(depth_dir, city, img_name[:-4] + '_leftImg8bit.png')
      depth_img = ski.data.load(depth_path)
      #depth_img = cv2.imread(rgb_path)
      depth_img = np.ascontiguousarray(depth_img[cy_start:cy_end,cx_start:cx_end])
      #depth_img = cv2.resize(depth_img, (width, height), interpolation=cv2.INTER_NEAREST)
      depth_img = ski.transform.resize(depth_img, (FLAGS.img_height, FLAGS.img_width),
                                       order=0, preserve_range=True)
      depth_img = np.round(depth_img / 256.0).astype(np.uint8)
      #depth_sum += depth_img
      #print((depth_sum / img_cnt).mean((0,1)))

      gt_path = join(gt_dir, city, img_name[:-4] + '_gtFine_labelIds.png')
      #print(gt_path)
      full_gt_img = ski.data.load(gt_path)
      full_gt_img = np.ascontiguousarray(full_gt_img[cy_start:cy_end,cx_start:cx_end])
      if FLAGS.downsample:
        full_gt_img = ski.transform.resize(full_gt_img, (FLAGS.img_height, FLAGS.img_width),
                                           order=0, preserve_range=True).astype(np.uint8)
      gt_img, car_mask = data_utils.convert_ids(full_gt_img)
      #rgb[car_mask] = 0
      rgb[car_mask] = IMG_MEAN
      #print(gt_img[40:60,100:110])
      #gt_weights = gt_data[1]
      gt_img = gt_img.astype(np.int8)
      gt_img[gt_img == -1] = FLAGS.num_classes
      #gt_weights, num_labels = data_utils.get_class_weights(gt_img)
      class_hist, num_labels = data_utils.get_class_hist(gt_img, FLAGS.num_classes)

      # Just to test correct casting in numpy/skimage - this must be the same
      #gt_ids_test = ski.util.img_as_ubyte(gt_ids_test).astype(np.int8)
      #assert (gt_ids != gt_ids_test).sum() == 0

      if name == 'val':
        instance_gt_path = join(gt_dir, city, img_name[:-4] + '_gtFine_instanceIds.png')
        instance_gt_img = ski.data.load(instance_gt_path)
        instance_gt_img = np.ascontiguousarray(instance_gt_img[cy_start:cy_end,cx_start:cx_end])
        if FLAGS.downsample:
          instance_gt_img = ski.transform.resize(
              instance_gt_img, (FLAGS.img_height, FLAGS.img_width),
              order=0, preserve_range=True).astype(np.uint16)
        ski.io.imsave(join(gt_save_dir, 'label', img_name[:-4]+'.png'), full_gt_img)
        ski.io.imsave(join(gt_save_dir, 'instance', img_name[:-4]+'.png'), instance_gt_img)

      create_tfrecord(rgb, gt_img, class_hist, depth_img,
                      num_labels, img_prefix, save_dir)


def main(argv):
  prepare_dataset('val')
  prepare_dataset('train')


if __name__ == '__main__':
  tf.app.run()
