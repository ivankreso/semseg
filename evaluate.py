import os
import sys
import time
from datetime import datetime
from shutil import copyfile
import importlib.util

import numpy as np
import tensorflow as tf
from tqdm import trange
import skimage as ski
import skimage.io

np.set_printoptions(linewidth=250)

import libs.cylib as cylib
import helper
import eval_helper
import train_helper
from datasets.cityscapes.cityscapes import CityscapesDataset
#import datasets.flip_reader as reader
#import datasets.reader_pyramid as reader
import datasets.reader as reader


tf.app.flags.DEFINE_string('model_dir', '', """Path to experiment dir.""")
FLAGS = tf.app.flags.FLAGS


helper.import_module('config', os.path.join(FLAGS.model_dir, 'config.py'))


def evaluate(model, dataset, save_dir):
  tf.gfile.MakeDirs(save_dir)
  with tf.Graph().as_default():
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5 # don't hog all vRAM
    #config.operation_timeout_in_ms = 5000   # terminate on long hangs
    #config.operation_timeout_in_ms = 15000   # terminate on long hangs
    sess = tf.Session(config=config)
    # Get images and labels.
    image, labels, weights, num_labels, img_name = \
        reader.inputs(dataset, num_epochs=FLAGS.max_epochs)
    image = model.normalize_input(image)

    # Build a Graph that computes the logits predictions from the inference model.
    # Calculate loss.
    #with tf.variable_scope("model"):
    logits, loss, draw_data = model.build(image, labels, weights, num_labels, is_training=False)


    #sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    #latest = tf.train.latest_checkpoint(FLAGS.model_dir)
    latest = os.path.join(FLAGS.model_dir, 'model.ckpt')
    print(latest)
    #variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
    restorer = tf.train.Saver(tf.all_variables())
    #restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, latest)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #tf.train.start_queue_runners(sess=sess)

    conf_mat = np.ascontiguousarray(
        np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64))
    loss_avg = 0
    loss_vals = []
    img_names = []
    for i in trange(dataset.num_examples()):
      out_logits, gt_labels, loss_val, img_prefix = sess.run([logits, labels, loss, img_name])
      img_prefix = img_prefix[0].decode("utf-8")
      loss_avg += loss_val
      loss_vals += [loss_val]
      img_names += [img_prefix]
      #net_labels = out_logits[0].argmax(2).astype(np.int32, copy=False)
      net_labels = out_logits[0].argmax(2).astype(np.int32)
      #gt_labels = gt_labels.astype(np.int32, copy=False)
      cylib.collect_confusion_matrix(net_labels.reshape(-1), gt_labels.reshape(-1), conf_mat)
      save_path = os.path.join(save_dir, img_prefix + '.png')
      #error_save_path = os.path.join(save_dir, str(loss_val) + img_prefix + '_errors.png')
      error_save_path = os.path.join(save_dir, img_prefix + '_' + str(loss_val) + '_error.png')
      gt_labels = gt_labels.reshape(net_labels.shape)
      eval_helper.draw_output(net_labels, CityscapesDataset.CLASS_INFO, save_path)
      net_labels[net_labels == gt_labels] = -1
      net_labels[gt_labels == -1] = -1
      eval_helper.draw_output(net_labels, CityscapesDataset.CLASS_INFO, error_save_path)
      #print(q_size)
    #print(conf_mat)
    #img_names = [[x,y] for (y,x) in sorted(zip(loss_vals, img_names))]
    sorted_data = [x for x in sorted(zip(loss_vals, img_names), reverse=True)]
    print(img_names)
    for i, elem in enumerate(sorted_data):
      print('Xent loss = ', elem[0])
      ski.io.imshow(os.path.join(save_dir, elem[1] + '_errors.png'))
      ski.io.show()

    print('')
    pixel_acc, iou_acc, recall, precision, _ = eval_helper.compute_errors(
        conf_mat, 'Validation', CityscapesDataset.CLASS_INFO, verbose=True)

    coord.request_stop()
    coord.join(threads)
    sess.close()

  return loss_avg / dataset.num_examples(), pixel_acc, iou_acc, recall, precision


def main(argv=None):  # pylint: disable=unused-argument
  model = helper.import_module('model', os.path.join(FLAGS.model_dir, 'model.py'))

  if not tf.gfile.Exists(FLAGS.model_dir):
    raise ValueError('Net dir not found: ' + FLAGS.model_dir)
  save_dir = os.path.join(FLAGS.model_dir, 'evaluation')
  tf.gfile.MakeDirs(save_dir)

  train_dataset = CityscapesDataset(FLAGS.dataset_dir, 'train')
  valid_dataset = CityscapesDataset(FLAGS.dataset_dir, 'val')
  #train(model, train_dataset, valid_dataset)
  evaluate(model, valid_dataset, os.path.join(save_dir, 'validation'))
  #evaluate(model, train_dataset, os.path.join(save_dir, 'train'))


if __name__ == '__main__':
  tf.app.run()

