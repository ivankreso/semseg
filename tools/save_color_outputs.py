import os
import sys
import time
from datetime import datetime
from shutil import copyfile
import importlib.util
from os.path import join

import numpy as np
import tensorflow as tf
from tqdm import trange
import PIL.Image as pimg
import skimage as ski
import skimage.data
import skimage.transform

import helper
import eval_helper
#from datasets.voc2012.dataset import Dataset
from datasets.cityscapes.cityscapes import CityscapesDataset as Dataset

np.set_printoptions(linewidth=250)

#DATA_DIR = '/home/kivan/datasets/voc2012_aug/data/'
#split = 'val'

#DATA_DIR = '/home/kivan/datasets/VOC2012/test_data'
seq_name = 'stuttgart_00'
#seq_name = 'stuttgart_01'
#seq_name = 'stuttgart_02'
#data_dir = '/home/kivan/datasets/Cityscapes/orig/leftImg8bit/demoVideo/stuttgart_00'
data_dir = join('/home/kivan/datasets/Cityscapes/orig/leftImg8bit/demoVideo', seq_name)
#model_dir = '/home/kivan/datasets/results/tmp/cityscapes/1_6_10-23-47'
model_dir = '/home/kivan/datasets/results/tmp/cityscapes/1_6_14-55-59'
tf.app.flags.DEFINE_string('model_dir', model_dir, '')
FLAGS = tf.app.flags.FLAGS


helper.import_module('config', os.path.join(FLAGS.model_dir, 'config.py'))


def forward_pass(model, save_dir):
  #img_dir = join(data_dir, 'JPEGImages')
  #file_path = join(data_dir, 'ImageSets', 'Segmentation', 'test.txt')
  #fp = open(file_path)
  #file_list = [line.strip() for line in fp]

  file_list = next(os.walk(data_dir))[2]
  file_list = sorted(file_list)

  save_dir_rgb = join(save_dir, seq_name)
  tf.gfile.MakeDirs(save_dir_rgb)
  #sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
  config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
  #config.gpu_options.per_process_gpu_memory_fraction = 0.5 # don't hog all vRAM
  #config.operation_timeout_in_ms = 5000   # terminate on long hangs
  #config.operation_timeout_in_ms = 15000   # terminate on long hangs
  sess = tf.Session(config=config)
  # Get images and labels.
  #run_ops = model.inference()

  batch_shape = (1, None, None, 3)
  image_tf = tf.placeholder(tf.float32, shape=batch_shape)
  logits, _ = model.inference(image_tf, constant_shape=False)

  #sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  latest = os.path.join(FLAGS.model_dir, 'model.ckpt')
  restorer = tf.train.Saver(tf.global_variables())
  restorer.restore(sess, latest)

  img_names = []
  cy_start = 30
  cy_end = 900

  width = 1024
  height = 448
  print(save_dir_rgb)
  for i in trange(len(file_list)):
    img_path = join(data_dir, file_list[i])
    print(img_path)
    image = np.array(pimg.open(img_path))
    image = np.ascontiguousarray(image[cy_start:cy_end,:,:])
    image = ski.transform.resize(image, (height, width), preserve_range=True, order=3)

    image = image[np.newaxis,...]
    logits_val = sess.run(logits, feed_dict={image_tf:image})
    #pred_labels = logits_val[0].argmax(2).astype(np.int32)
    pred_labels = logits_val[0].argmax(2).astype(np.uint8)
    #save_path = os.path.join(save_dir_rgb, file_list[i])
    #pred_rgb = eval_helper.draw_output(pred_labels, Dataset.class_info, save_path)
    pred_rgb = eval_helper.draw_output(pred_labels, Dataset.class_info)
    merged = np.concatenate((image[0], pred_rgb), axis=0).astype(np.uint8)
    #print(merged.shape, merged.dtype)
    merged_img = pimg.fromarray(merged)
    save_path = os.path.join(save_dir_rgb, '%06d.png' % i)
    merged_img.save(save_path)
    #pred_img.save(join(save_dir_submit, file_list[i] + '.png'))
    #pred_img.save(join(save_dir_submit, file_list[i] + '.png'))

    ##gt_labels = gt_labels.astype(np.int32, copy=False)
    #cylib.collect_confusion_matrix(net_labels.reshape(-1), gt_labels.reshape(-1), conf_mat)
    #gt_labels = gt_labels.reshape(net_labels.shape)
    #pred_labels = np.copy(net_labels)
    #net_labels[net_labels == gt_labels] = -1
    #net_labels[gt_labels == -1] = -1
    #num_mistakes = (net_labels >= 0).sum()
    #img_prefix = '%07d_'%num_mistakes + img_prefix

    #error_save_path = os.path.join(save_dir, str(loss_val) + img_prefix + '_errors.png')
    #filename =  img_prefix + '_' + str(loss_val) + '_error.png'
    #error_save_path = os.path.join(save_dir, filename)
    #eval_helper.draw_output(net_labels, CityscapesDataset.CLASS_INFO, error_save_path)
    #print(q_size)
  #print(conf_mat)
  #img_names = [[x,y] for (y,x) in sorted(zip(loss_vals, img_names))]
  #sorted_data = [x for x in sorted(zip(loss_vals, img_names), reverse=True)]
  #print(img_names)
  #for i, elem in enumerate(sorted_data):
  #  print('Xent loss = ', elem[0])
  #  ski.io.imshow(os.path.join(save_dir, elem[1] + '_errors.png'))
  #  ski.io.show()

  #print('')
  #pixel_acc, iou_acc, recall, precision, _ = eval_helper.compute_errors(
  #    conf_mat, 'Validation', CityscapesDataset.CLASS_INFO, verbose=True)
  sess.close()


def main(argv=None):  # pylint: disable=unused-argument
  model = helper.import_module('model', os.path.join(FLAGS.model_dir, 'model.py'))

  if not tf.gfile.Exists(FLAGS.model_dir):
    raise ValueError('Net dir not found: ' + FLAGS.model_dir)
  save_dir = os.path.join(FLAGS.model_dir, 'evaluation', 'test')
  tf.gfile.MakeDirs(save_dir)

  forward_pass(model, save_dir)


if __name__ == '__main__':
  tf.app.run()

