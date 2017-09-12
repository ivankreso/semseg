import os
import pickle
import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.data
from tqdm import trange
from cityscapes import CityscapesDataset
from datasets.dataset_helper import convert_colors_to_indices, convert_colors_to_indices_slow


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('gt_dir',
    '/home/kivan/datasets/Cityscapes/2048x1024/gt/', 'Dataset dir')
tf.app.flags.DEFINE_string('save_dir', FLAGS.gt_dir + '/../gt_data/', '')


def prepare_dataset(name):
  print('Preparing ' + name)
  gt_dir = FLAGS.gt_dir + name + '/'
  cities = next(os.walk(gt_dir))[1]
  save_dir = FLAGS.save_dir + name + '/'
  print('Save dir = ', save_dir)
  os.makedirs(save_dir, exist_ok=True)
  for city in cities:
    print(city)
    img_list = next(os.walk(gt_dir + city))[2]
    gt_data_save_dir = FLAGS.save_dir + name + '/' + city + '/'
    os.makedirs(gt_data_save_dir, exist_ok=True)
    for i in trange(len(img_list)):
      img_name = img_list[i]
      gt_path = gt_dir + city + '/' + img_name
      gt_rgb = ski.data.load(gt_path).astype(np.uint8)
      #gt_rgb = ski.util.img_as_ubyte(gt_rgb)
      label_map, label_weights, num_labels, class_hist, class_weights = \
          convert_colors_to_indices(gt_rgb, CityscapesDataset.CLASS_COLOR_MAP, 1000)

      pickle_filepath = gt_data_save_dir + img_name[:-4] + '.pickle'
      with open(pickle_filepath, 'wb') as f:
        pickle.dump([label_map, label_weights, num_labels, class_hist, class_weights], f)


def main(argv):
  prepare_dataset('train')
  prepare_dataset('val')


if __name__ == '__main__':
  tf.app.run()
