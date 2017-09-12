import sys
sys.path.append('../..')

import os
import pickle
import numpy as np
import tensorflow as tf
#from pgmagick import Image
import skimage as ski
import skimage.data, skimage.transform
from tqdm import trange
from kitti_info import class_info, class_color_map
from datasets.dataset_helper import convert_colors_to_indices, convert_colors_to_indices_slow

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
flags = tf.app.flags
flags.DEFINE_string('gt_dir', '/home/kivan/datasets/KITTI/semantic_segmentation/',
                    'Dataset dir')
flags.DEFINE_string('save_dir', FLAGS.gt_dir + '/gt_data/', '')
tf.app.flags.DEFINE_integer('width', 1216, '')
tf.app.flags.DEFINE_integer('height', 384, '')

FLAGS = flags.FLAGS


def prepare_dataset(name):
  #rgb_means = [123.68, 116.779, 103.939]
  print('Preparing ' + name)
  gt_dir = FLAGS.gt_dir + name + '/labels/'
  img_list = next(os.walk(gt_dir))[2]
  #print(img_list)
  save_dir = FLAGS.save_dir + name + '/'
  print('Save dir = ', save_dir)
  os.makedirs(save_dir, exist_ok=True)
  for i in trange(len(img_list)):
    img_name = img_list[i]
    gt_path = gt_dir + '/' + img_name
    img = ski.data.load(gt_path)
    img = ski.transform.resize(img, (FLAGS.height, FLAGS.width), order=0, preserve_range=True)
    img = img.astype(np.uint8)
    #print(gt_rgb)
    #gt_rgb = ski.util.img_as_ubyte(gt_rgb)
    label_map, label_weights, num_labels, class_hist, _ = \
        convert_colors_to_indices(img, class_color_map, 1000)

    pickle_filepath = save_dir + img_name[:-4] + '.pickle'
    with open(pickle_filepath, 'wb') as f:
      pickle.dump([label_map, label_weights, num_labels, class_hist], f)


def main(argv):
  prepare_dataset('valid')
  prepare_dataset('train')


if __name__ == '__main__':
  tf.app.run()
