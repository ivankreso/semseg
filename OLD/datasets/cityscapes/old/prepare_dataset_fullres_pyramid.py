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
import datasets.dataset_helper as dataset_helper
import models.model_helper as model_helper

FLAGS = tf.app.flags.FLAGS


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
    'x' + str(FLAGS.img_height) + '_pyramid/', '')

#flags.DEFINE_integer('cx_start', 0, '')
#flags.DEFINE_integer('cx_end', 2048, '')
#flags.DEFINE_integer('cy_start', 30, '')
#flags.DEFINE_integer('cy_end', 894, '')
FLAGS = flags.FLAGS

def _select_scale(scale_factors, target_sf):
  scale = len(scale_factors) - 1
  for k in range(len(scale_factors)):
    sf = scale_factors[k]
    if target_sf <= sf:
      if k == 0:
        scale = k
        break
      else:
        prev_sf = scale_factors[k-1]
        dist1 = abs(target_sf - prev_sf)
        dist2 = abs(target_sf - sf)
        if dist1 < dist2:
          scale = k - 1
        else:
          scale = k
        break
  return scale 

def _get_depth_routing(baseline, metric_scales, scale_factors, rf_size):
  depth_routing = []
  downscale_factors = [1 / x for x in scale_factors]
  for i, s in enumerate(metric_scales):
    depth_routing += [[]]
    for d in range(130):
      sf = (d * s / baseline) / rf_size
      scale_level = _select_scale(downscale_factors, sf)
      depth_routing[i] += [scale_level]
      #print(depth_routing)
  return depth_routing


IMG_WIDTH = 1124
IMG_HEIGHT = 1024
#scale_factors = [1.0, 0.65, 0.3]
#scale_factors = [1.2, 0.9, 0.6, 0.3]
#scale_factors = [1.2, 1.0, 0.8, 0.6, 0.4]
#scale_factors = [1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
#APROX_SCALE_FACTORS = [1.0, 0.8, 0.6, 0.4, 0.2]
#APROX_SCALE_FACTORS = [1.0, 0.7, 0.5, 0.35, 0.25]
# s = 1.4
APROX_SCALE_FACTORS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15]
#NET_SUBSAMPLING = 8
NET_SUBSAMPLING = 16
#METRIC_SCALES = [1, 4, 7]
#METRIC_SCALES = [1, 2]
METRIC_SCALES = [1, 2, 4]
RF_SIZE = 186
BASELINE = 0.21
IMG_SIZES, SCALE_FACTORS = model_helper.get_multiscale_resolutions(
    IMG_WIDTH, IMG_HEIGHT, APROX_SCALE_FACTORS)
EMBED_SIZES = []
for i, e in enumerate(IMG_SIZES):
  EMBED_SIZES += [[int(e[0] / NET_SUBSAMPLING), int(e[1] / NET_SUBSAMPLING)]]
DEPTH_ROUTING = _get_depth_routing(BASELINE, METRIC_SCALES, SCALE_FACTORS, RF_SIZE)
print('Scale factors = ', SCALE_FACTORS)
print('Image resolutions = ', IMG_SIZES)
print('Embedding resolutions = ', EMBED_SIZES)


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord(rgb, depth_img, label_map, weight_map, num_labels,
                    img_name, save_dir, debug_dir):
  rows = rgb.shape[0]
  cols = rgb.shape[1]
  depth = rgb.shape[2]

  routing_data = dataset_helper.get_scale_selection_routing(
      depth_img, NET_SUBSAMPLING, DEPTH_ROUTING, EMBED_SIZES, img_name, debug_dir)

  filename = os.path.join(save_dir + img_name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  rgb_raw = rgb.tostring()
  routing_data_raw = routing_data.tostring()
  labels_raw = label_map.tostring()
  weights_str = weight_map.tostring()
  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(rows),
      'width': _int64_feature(cols),
      'depth': _int64_feature(depth),
      'num_labels': _int64_feature(int(num_labels)),
      'img_name': _bytes_feature(img_name.encode()),
      'rgb': _bytes_feature(rgb_raw),
      'routing_data': _bytes_feature(routing_data_raw),
      'label_weights': _bytes_feature(weights_str),
      'labels': _bytes_feature(labels_raw)}))
      #'disparity': _bytes_feature(disp_raw),
  writer.write(example.SerializeToString())
  writer.close()


def prepare_dataset(name):
  #rgb_means = [123.68, 116.779, 103.939]
  print('Preparing ' + name)
  root_dir = FLAGS.data_dir + '/rgb/' + name + '/'
  depth_dir = os.path.join(FLAGS.data_dir, 'depth', name)
  gt_dir = FLAGS.data_dir + '/gt_data/' + name + '/'
  cities = next(os.walk(root_dir))[1]
  save_dir = FLAGS.save_dir + name + '/'
  print('Save dir = ', save_dir)
  os.makedirs(save_dir, exist_ok=True)
  #print('Writing', filename)
  half_width = int(FLAGS.img_width / 2)
  left_x_end = int(half_width + FLAGS.rf_half_size)
  right_x_start = int(half_width - FLAGS.rf_half_size)

  debug_dir = os.path.join(FLAGS.save_dir, 'debug', name)
  os.makedirs(debug_dir, exist_ok=True)
  for city in cities:
    print(city)
    img_list = next(os.walk(root_dir + city))[2]
    #for img_name in img_list:
    for i in trange(len(img_list)):
      img_name = img_list[i]
      img_prefix = img_name[:-4]
      rgb_path = root_dir + city + '/' + img_name
      rgb = ski.data.load(rgb_path)
      depth_path = os.path.join(depth_dir, city, img_prefix + '_leftImg8bit.png')
      depth = ski.data.load(depth_path)
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

      depth = depth.astype(np.float32) / 256
      #print(depth.min(), depth.max())
      rgb_left = np.ascontiguousarray(rgb[:,:left_x_end,:])
      rgb_right = np.ascontiguousarray(rgb[:,right_x_start:,:])
      depth_left = np.ascontiguousarray(depth[:,:left_x_end])
      depth_right = np.ascontiguousarray(depth[:,right_x_start:])
      gt_left = np.ascontiguousarray(gt_map[:,:left_x_end])
      gt_right = np.ascontiguousarray(gt_map[:,right_x_start:])
      gt_left[:,half_width:] = -1
      gt_right[:,:FLAGS.rf_half_size] = -1
      weights_left = np.ascontiguousarray(gt_weights[:,:left_x_end])
      weights_right = np.ascontiguousarray(gt_weights[:,right_x_start:])
      left_num_labels = (gt_left >= 0).sum()
      right_num_labels = (gt_right >= 0).sum()
      create_tfrecord(rgb_left, depth_left, gt_left, weights_left, left_num_labels,
          img_prefix + '_left', save_dir, debug_dir)
      create_tfrecord(rgb_right, depth_right, gt_right, weights_right, right_num_labels,
          img_prefix + '_right', save_dir, debug_dir)
      #print('left = ', rgb_left.shape)
      #print('right = ', rgb_right.shape)

      #create_tfrecord(rgb, gt_map, gt_weights, num_labels, img_prefix,
      #    save_dir)



def main(argv):
  prepare_dataset('val')
  prepare_dataset('train')


if __name__ == '__main__':
  tf.app.run()
