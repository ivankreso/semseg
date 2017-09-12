import tensorflow as tf
import argparse
import os, re
import numpy as np
import skimage as ski
import skimage.data
import skimage.transform
import cv2

import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

import losses
import eval_helper
#import datasets.reader_rgbd_depth as reader
import datasets.reader as reader

FLAGS = tf.app.flags.FLAGS

MODEL_DEPTH = 50
#MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]
MEAN_BGR = [75.08929598, 85.01498926, 75.2051479]
#MEAN_BGR = [103.939, 116.779, 123.68]


def evaluate(name, sess, epoch_num, run_ops, dataset, data):
  #TODO iIOU
  loss_val, accuracy, iou, recall, precision = eval_helper.evaluate_segmentation(
      sess, epoch_num, run_ops, dataset.num_examples())
  if iou > data['best_iou'][0]:
    data['best_iou'] = [iou, epoch_num]
  data['iou'] += [iou]
  data['acc'] += [accuracy]
  data['loss'] += [loss_val]

def plot_results(train_data, valid_data):
  eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats'),
                                     train_data, valid_data)

def print_results(data):
  print('Best validation IOU = %.2f (epoch %d)' % tuple(data['best_iou']))

def init_eval_data():
  train_data = {}
  valid_data = {}
  train_data['lr'] = []
  train_data['loss'] = []
  train_data['iou'] = []
  train_data['acc'] = []
  train_data['best_iou'] = [0, 0]
  valid_data['best_iou'] = [0, 0]
  valid_data['loss'] = []
  valid_data['iou'] = []
  valid_data['acc'] = []
  return train_data, valid_data


def normalize_input(img):
  return img - MEAN_BGR
  #"""Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
  #with tf.name_scope('input'), tf.device('/cpu:0'):
  #  #rgb -= MEAN_RGB
  #  red, green, blue = tf.split(3, 3, rgb)
  #  bgr = tf.concat(3, [blue, green, red])
  #  #bgr -= MEAN_BGR
  #  return bgr

bn_params = {
  # Decay for the moving averages.
  #'decay': 0.999,
  'decay': 0.9,
  'center': True,
  'scale': True,
  # epsilon to prevent 0s in variance.
  #'epsilon': 0.001,
  'epsilon': 1e-5,
  # None to force the updates
  'updates_collections': None,
  'is_training': True
}

def layer(net, num_filters, name, is_training):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    net = layers.convolution2d(net, num_filters, kernel_size=3)
    #if is_training: 
      #net = tf.nn.dropout(net, keep_prob=0.8)
  return net

def dense_block(net, size, r, name, is_training):
  with tf.variable_scope(name):
    outputs = []
    for i in range(size):
      if i < size - 1:
        x = net
        net = layer(net, r, 'layer'+str(i), is_training)
        outputs += [net]
        net = tf.concat(3, [x, net])
      else:
        net = layer(net, r, 'layer'+str(i), is_training)
        outputs += [net]
    net = tf.concat(3, outputs)
  return net

def downsample(net, name, is_training):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net)
    net = tf.nn.relu(net)
    num_filters = net.get_shape().as_list()[3]
    net = layers.convolution2d(net, num_filters, kernel_size=1)
    #if is_training:
    #  net = tf.nn.dropout(net, keep_prob=0.8)
    net = layers.max_pool2d(net, 2, stride=2, padding='SAME')
  return net

def upsample(net, name):
  with tf.variable_scope(name):
    num_filters = net.get_shape().as_list()[3]
    net = tf.contrib.layers.convolution2d_transpose(net, num_filters, kernel_size=3, stride=2)
    return net

def _build(image, is_training):
  bn_params['is_training'] = is_training
  weight_decay = 1e-4
  #init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
  init_func = layers.variance_scaling_initializer()

  cfg = {
    #5: [4,5,7,10,12,15],
    5: [2,3,4,5,6,8],
    #5: [3,3,3,3,3,3],
    #5: [3,3,3],
    #5: [2,2],
  }
  block_sizes = cfg[5]
  r = 16
  #r = 12
  
  with arg_scope([layers.convolution2d, layers.convolution2d_transpose],
      stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.convolution2d(image, 48, 3, scope='conv0')
    block_outputs = []
    for i, size in enumerate(block_sizes):
      print(i, size)
      x = net
      net = dense_block(net, size, r, 'block'+str(i), is_training)
      net = tf.concat(3, [x, net])
      block_outputs += [net]
      print(net)
      if i < len(block_sizes) - 1:
        net = downsample(net, 'block'+str(i)+'_downsample', is_training)
  logits_mid = layers.convolution2d(net, FLAGS.num_classes, 1, activation_fn=None,
      biases_initializer=tf.zeros_initializer, scope='logits_middle')
  logits_mid = tf.image.resize_bilinear(logits_mid, [FLAGS.img_height, FLAGS.img_width],
                                    name='resize_logits_middle')
  return logits_mid, None

  #  #net = tf.nn.relu(net)
  #  #num_filters = net.get_shape().as_list()[3]
  #  #net = layers.convolution2d(net, num_filters, kernel_size=1)

  #  for i, size in reversed(list(enumerate(block_sizes[:-1]))):
  #    print(i, size)
  #    net = upsample(net, 'block'+str(i)+'_back_upsample')
  #    print(block_outputs[i])
  #    net = tf.concat(3, [block_outputs[i], net])
  #    print(net)
  #    net = dense_block(net, size, r, 'block'+str(i)+'_back', is_training)
  #    print(net)
  #  logits = layers.convolution2d(net, FLAGS.num_classes, 1,
  #      biases_initializer=tf.zeros_initializer, scope='logits')
  #  #logits = tf.image.resize_bilinear(logits, [FLAGS.img_height, FLAGS.img_width],
  #  #                                  name='resize_logits')
  #return logits, None


def build(dataset, is_training, reuse=False):
  # Get images and labels.
  x, labels, weights, depth, img_names = reader.inputs(dataset, is_training=is_training, num_epochs=FLAGS.max_epochs)
  x = normalize_input(x)

  if reuse:
    tf.get_variable_scope().reuse_variables()

  #logits = _build(x, is_training)
  logits, logits_mid = _build(x, is_training)
  total_loss = loss(logits, logits_mid, labels, weights, is_training)

  #all_vars = tf.contrib.framework.get_variables()
  #for v in all_vars:
  #  print(v.name)
  if is_training:
    #init_op, init_feed = create_init_op(resnet_param)
    return [total_loss], None, None
  else:
    return [total_loss, logits, labels, img_names]


def loss(logits, logits_mid, labels, weights, is_training=True):
#def loss(logits, labels, weights, is_training=True):
  xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights)
  #xent_loss += losses.weighted_cross_entropy_loss(logits_mid, labels, weights)
  #xent_loss /= 2

  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels)
  #xent_loss += losses.weighted_cross_entropy_loss(logits_mid, labels)

  #loss_tf = tf.contrib.losses.softmax_cross_entropy()
  #loss_val = losses.weighted_hinge_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss_symmetric(logits, labels, weights, num_labels)
  #all_losses = [depth_loss, xent_loss]
  all_losses = [xent_loss]

  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)

  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss

def minimize(opts, loss, global_step):
  grads = opts[0].compute_gradients(loss)
  train_op = opts[0].apply_gradients(grads, global_step=global_step)
  return train_op

def train_step(sess, run_ops):
  return sess.run(run_ops)

def num_batches(dataset):
  return reader.num_examples(dataset)

def num_examples(dataset):
  return reader.num_examples(dataset)
