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
tf.app.flags.DEFINE_integer('img_width', 1024, '')
tf.app.flags.DEFINE_integer('img_height', 448, '')
# leave out the car hood
#tf.app.flags.DEFINE_integer('cx_start', 0, '')
tf.app.flags.DEFINE_integer('cx_start', 120, '')
tf.app.flags.DEFINE_integer('cx_end', 2048, '')
#tf.app.flags.DEFINE_integer('cy_start', 0, '')
#tf.app.flags.DEFINE_integer('cy_end', 900, '')
#tf.app.flags.DEFINE_integer('img_width', 640, '')
#tf.app.flags.DEFINE_integer('img_height', 288, '')

tf.app.flags.DEFINE_integer('cy_start', 30, '')
tf.app.flags.DEFINE_integer('cy_end', 900, '')
#tf.app.flags.DEFINE_integer('img_width', 768, '')
#tf.app.flags.DEFINE_integer('img_height', 320, '')

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
    '{}x{}'.format(FLAGS.img_width, FLAGS.img_height) + '_pyramid_last3/', '')
    #'{}x{}'.format(FLAGS.img_width, FLAGS.img_height) + '_pyramid/', '')

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(img, label_map, class_hist, depth_img, mux_indices,
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
  mux_indices_raw = mux_indices.tostring()
  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'channels': _int64_feature(channels),
      'num_labels': _int64_feature(int(num_labels)),
      'img_name': _bytes_feature(img_name.encode()),
      'image': _bytes_feature(img_str),
      'mux_indices': _bytes_feature(mux_indices_raw),
      'class_hist': _bytes_feature(class_hist_str),
      'labels': _bytes_feature(labels_str),
      'depth': _bytes_feature(depth_raw)
      }))
  writer.write(example.SerializeToString())
  writer.close()

def _select_level(pyramid_rfs, target_rf):
  level = len(pyramid_rfs) - 1
  for k in range(len(pyramid_rfs)):
    curr_rf = pyramid_rfs[k]
    if target_rf <= curr_rf:
      if k == 0:
        level = k
        break
      else:
        prev_rf = pyramid_rfs[k-1]
        dist1 = abs(target_rf - prev_rf)
        dist2 = abs(target_rf - curr_rf)
        if dist1 < dist2:
          level = k - 1
        else:
          level = k
        break
  return level

#METRIC_SCALES = [1, 2, 4]
#RF_SIZE = 186
#def _get_depth_routing(baseline, metric_scales, scale_factors, rf_size):
#def _get_depth_routing(baseline, metric_scales, pyramid_data):
def _get_depth_routing(baseline, metric_scales, pyramid_rfs):
  depth_routing = []
  #downscale_factors = [1 / x for x in scale_factors]
  for i, s in enumerate(metric_scales):
    depth_routing.append([])
    for d in range(130):
      #sf = (d * s / baseline) / rf_size
      target_rf = d * s / baseline
      skip_level = _select_level(pyramid_rfs, target_rf)
      print(d, ' - ', s, ' -- ', target_rf, ' -> ', skip_level)
      depth_routing[i].append(skip_level)
      #print(depth_routing)
  return depth_routing


def get_multiplexer_indices(depth, depth_routing, embed_sizes, filename, debug_save_dir):
  color_coding = [[0,0,0], [128,64,128], [244,35,232], [70,70,70], [102,102,156], [190,153,153],
                  [153,153,153], [250,170,30], [220,220,0]]
  height = depth.shape[0]
  width = depth.shape[1]
  #height = height // net_subsampling
  #width = width // net_subsampling
  #depth = ski.transform.resize(depth, (height, width), preserve_range=True, order=3)
  #print(depth.shape)
  #print(depth.min())
  #print(depth.max())
  #for i in range(129):
  #  depth_map[i] = 0
  num_scales = len(depth_routing)
  #routing_data = np.zeros((height, width, num_scales), dtype=np.uint8)
  routing_data = np.zeros((height, width, num_scales), dtype=np.int32)
  debug_img = np.ndarray((height, width, 3), dtype=np.uint8)

  level_offsets = [0]
    #level_offset = level_offsets[level]
  #level_offset = 0
  for level_res in embed_sizes[:-1]:
    level_offsets.append(level_offsets[-1] + (level_res[0] * level_res[1]))
  #print(level_offsets)
  for s, routing in enumerate(depth_routing):
    for y in range(height):
      for x in range(width):
        d = int(round(depth[y,x]))
        #routing_data[s = depth_routing[s][d]
        #routing_data[r,c,s] = routing[d]

        # TODO
        #level = routing[d]
        level = len(level_offsets) - 1 - s

        level_offset = level_offsets[level]
        #level_offset = 0
        level_height = embed_sizes[level][0]
        level_width = embed_sizes[level][1]
        #px = int(round((x / width) * level_width))
        #py = int(round((y / height) * level_height))
        px = int((x / width) * level_width)
        py = int((y / height) * level_height)
        routing_data[y,x,s] = level_offset + (py * level_width + px)
        #debug_img[y,x] = color_coding[level]
        #print(level, py, px)
    #ski.io.imsave(os.path.join(debug_save_dir, filename + '_' + str(s) + '.png'), debug_img)
  return routing_data


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


  # - 128x  550x550
  # - 64x   400x400
  # - 32x   200x200
  # - 16x   100x100
  # - 8x    60x60
  # - 4x    20x20
  #input_size = [1024,448]
  pyramid_sizes = [[112,256], [56,128], [28,64], [14,32], [7,16], [4,8]]
  pyramid_rfs = [20, 60, 100, 200, 400, 550]
  baseline = 0.21
  metric_scales = [1, 3, 7]
  #metric_scales = [1, 4, 7]
  depth_routing = _get_depth_routing(baseline, metric_scales, pyramid_rfs)
  for city in cities:
    print(city)
    img_list = next(os.walk(root_dir + city))[2]
    for i in trange(len(img_list)):
      img_cnt += 1
      img_name = img_list[i]
      img_prefix = img_name[:-4]
      rgb_path = root_dir + city + '/' + img_name
      rgb = ski.data.load(rgb_path)
      orig_height = rgb.shape[0]
      #rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
      rgb = np.ascontiguousarray(rgb[cy_start:cy_end,cx_start:cx_end,:])
      #rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
      #rgb = ski.transform.resize(rgb, (height, width), preserve_range=True, order=3)
      rgb = ski.transform.resize(rgb, (height, width), preserve_range=True, order=2)
      rgb = rgb.astype(np.uint8)
      #depth_img = None
      depth_path = join(depth_dir, city, img_name[:-4] + '_leftImg8bit.png')
      depth_img = ski.data.load(depth_path)
      #depth_img = cv2.imread(rgb_path)
      depth_img_full = np.ascontiguousarray(depth_img[cy_start:cy_end,cx_start:cx_end])
      #depth_img = cv2.resize(depth_img, (width, height), interpolation=cv2.INTER_NEAREST)
      #depth_img = ski.transform.resize(depth_img, (FLAGS.img_height, FLAGS.img_width),
      #                                 order=0, preserve_range=True)
      depth_img_input = ski.transform.resize(depth_img_full, (FLAGS.img_height, FLAGS.img_width),
                                             order=2, preserve_range=True)

      downsample_factor = 2048 / FLAGS.img_width
      depth_img_input = np.round(depth_img_input / (256.0 * downsample_factor)).astype(np.uint8)

      target_sub = 4
      depth_img = ski.transform.resize(depth_img_full,
                                       (FLAGS.img_height // target_sub,
                                        FLAGS.img_width // target_sub),
                                        #order=1, preserve_range=True)
                                        order=2, preserve_range=True)
                                        #order=0, preserve_range=True)
      depth_img = np.round(depth_img / (256.0 * downsample_factor))
      #depth_img /= downsample_factor
      #print(depth_img.mean())
      #print(depth_img.min(), depth_img.max())
      depth_img = depth_img.astype(np.uint8)
      #depth_sum += depth_img
      #print((depth_sum / img_cnt).mean((0,1)))

      gt_path = join(gt_dir, city, img_name[:-4] + '_gtFine_labelIds.png')
      #print(gt_path)
      full_gt_img = ski.data.load(gt_path)
      full_gt_img = np.ascontiguousarray(full_gt_img[cy_start:cy_end,cx_start:cx_end])
      if FLAGS.downsample:
        full_gt_img = ski.transform.resize(full_gt_img, (FLAGS.img_height, FLAGS.img_width),
                                           order=0, preserve_range=True).astype(np.uint8)
      if cy_end < orig_height:
        has_hood = False
      gt_img, car_mask = data_utils.convert_ids(full_gt_img, has_hood)
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

      #indices = get_multiplexer_indices(depth_img, input_sizes, subsample_rfs, pyramid_rfs)
      debug_dir = '/home/kivan/datasets/tmp/debug/multiplexer/'
      indices = get_multiplexer_indices(depth_img, depth_routing, pyramid_sizes,
                                        img_prefix, debug_dir)
      create_tfrecord(rgb, gt_img, class_hist, depth_img_input, indices,
                      num_labels, img_prefix, save_dir)


def main(argv):
  prepare_dataset('train')
  prepare_dataset('val')


if __name__ == '__main__':
  tf.app.run()
