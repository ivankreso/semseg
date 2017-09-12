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
    '/home/kivan/datasets/Cityscapes/ppm/rgb/', 'Dataset dir')
    #'/home/kivan/datasets/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/', 'Dataset dir')
flags.DEFINE_string('gt_dir',
    '/home/kivan/datasets/Cityscapes/ppm/gt/', 'Dataset dir')
    #'/home/kivan/datasets/Cityscapes/gtFine_trainvaltest/gtFine_19/', 'Dataset dir')
flags.DEFINE_integer('img_width', 1024, '')
flags.DEFINE_integer('img_height', 432, '')
flags.DEFINE_string('save_dir',
    '/home/kivan/datasets/Cityscapes/tf/' + str(FLAGS.img_width) +
    'x' + str(FLAGS.img_height) + '/', '')

flags.DEFINE_integer('cx_start', 0, '')
flags.DEFINE_integer('cx_end', 2048, '')
flags.DEFINE_integer('cy_start', 30, '')
flags.DEFINE_integer('cy_end', 894, '')
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
  #rgb_means = [123.68, 116.779, 103.939]
  print('Preparing ' + name)
  root_dir = FLAGS.data_dir + name + '/'
  gt_dir = FLAGS.gt_dir + name + '/'
  cities = next(os.walk(root_dir))[1]
  save_dir = FLAGS.save_dir + name + '/'
  print('Save dir = ', save_dir)
  os.makedirs(save_dir, exist_ok=True)
  #print('Writing', filename)
  cx_start = FLAGS.cx_start
  cx_end = FLAGS.cx_end
  cy_start = FLAGS.cy_start
  cy_end = FLAGS.cy_end
  for city in cities:
    print(city)
    img_list = next(os.walk(root_dir + city))[2]
    #for img_name in img_list:
    rgb_save_dir = FLAGS.save_dir + '/img/rgb/' + name + '/' + city + '/'
    gt_save_dir = FLAGS.save_dir + '/img/gt/' + name + '/' + city + '/'
    os.makedirs(rgb_save_dir, exist_ok=True)
    os.makedirs(gt_save_dir, exist_ok=True)
    for i in trange(len(img_list)):
      img_name = img_list[i]
      rgb_path = root_dir + city + '/' + img_name
      rgb = ski.data.load(rgb_path)
      rgb = rgb[cy_start:cy_end,cx_start:cx_end,:]
      rgb = ski.transform.resize(rgb, (FLAGS.img_height, FLAGS.img_width), order=3)
      ski.io.imsave(rgb_save_dir + img_name, rgb)

      rgb = rgb.astype(np.float32)
      #for c in range(3):
      #  rgb[:,:,c] -= rgb[:,:,c].mean()
      #  rgb[:,:,c] /= rgb[:,:,c].std()
      #  #rgb = (rgb - rgb.mean()) / rgb.std()
      for c in range(3):
        rgb[:,:,c] -= rgb_means[c]
      gt_path = gt_dir + city + '/' + img_name
      gt_rgb = ski.data.load(gt_path)
      #dump_nparray(array, filename)
      gt_rgb = gt_rgb[cy_start:cy_end,cx_start:cx_end,:]
      gt_rgb = ski.transform.resize(gt_rgb, (FLAGS.img_height, FLAGS.img_width),
                                    order=0, preserve_range=True)
      gt_rgb = gt_rgb.astype(np.uint8)
      #print(gt_rgb)
      ski.io.imsave(gt_save_dir + img_name, gt_rgb)
      #gt_rgb = ski.util.img_as_ubyte(gt_rgb)
      labels, label_weights, num_labels = convert_colors_to_indices(gt_rgb, class_color_map)
      rows = rgb.shape[0]
      cols = rgb.shape[1]
      depth = rgb.shape[2]

      img_prefix = img_name[:-4]
      filename = os.path.join(save_dir + img_prefix + '.tfrecords')
      writer = tf.python_io.TFRecordWriter(filename)
      rgb_raw = rgb.tostring()
      labels_raw = labels.tostring()
      weights_str = label_weights.tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(rows),
          'width': _int64_feature(cols),
          'depth': _int64_feature(depth),
          'num_labels': _int64_feature(num_labels),
          'img_name': _bytes_feature(img_prefix.encode()),
          'rgb_norm': _bytes_feature(rgb_raw),
          'class_weights': _bytes_feature(weights_str),
          'labels': _bytes_feature(labels_raw)}))
      writer.write(example.SerializeToString())
      writer.close()


def main(argv):
  crop_width = FLAGS.cx_end - FLAGS.cx_start
  crop_height = FLAGS.cy_end - FLAGS.cy_start
  print('Crop ratio = ', crop_width / crop_height)
  print('Resize ratio = ', FLAGS.img_width / FLAGS.img_height)
  prepare_dataset('train')
  prepare_dataset('val')


if __name__ == '__main__':
  tf.app.run()
