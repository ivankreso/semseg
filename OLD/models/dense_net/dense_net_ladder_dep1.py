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
import datasets.reader_rgbd as reader
#import datasets.reader as reader

FLAGS = tf.app.flags.FLAGS

#MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]
MEAN_BGR = [75.08929598, 85.01498926, 75.2051479]
#MEAN_BGR = [103.939, 116.779, 123.68]

weight_decay = 1e-4
#init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
init_func = layers.variance_scaling_initializer()

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


def normalize_input(img, depth):
  return img - MEAN_BGR, depth - 33.0
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

def build_refinement_module(top_layer, skip_data):
  skip_layer = skip_data[0]
  size_bottom = skip_data[1]
  skip_name = skip_data[2]

  top_height = top_layer.get_shape()[1].value
  top_width = top_layer.get_shape()[2].value
  skip_height = skip_layer.get_shape()[1].value
  skip_width = skip_layer.get_shape()[2].value
  size_top = top_layer.get_shape()[3].value

  if top_height != skip_height or top_width != skip_width:
    assert(2*top_height == skip_height)
    top_layer = tf.image.resize_bilinear(top_layer, [skip_height, skip_width],
                                         name=skip_name + '_refine_upsample')

  with arg_scope([layers.convolution2d],
    padding='SAME', activation_fn=tf.nn.relu,
    normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
    weights_initializer=init_func,
    weights_regularizer=layers.l2_regularizer(weight_decay)):
    print(skip_name, top_layer, skip_layer)
    skip_layer = tf.nn.relu(skip_layer)
    skip_layer = layers.convolution2d(skip_layer, size_top, kernel_size=3,
                                      scope=skip_name+'_refine_prep')
    net = tf.concat([top_layer, skip_layer], 3)
    net = layers.convolution2d(net, size_bottom, kernel_size=3,
                               scope=skip_name+'_refine_fuse')
  return net

def BNReluConv(net, num_filters, name, is_training, k=3, rate=1):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    net = layers.conv2d(net, num_filters, kernel_size=k, rate=rate)
  return net

def layer(net, num_filters, name, is_training, k=3):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    net = layers.conv2d(net, 4*num_filters, kernel_size=1)
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    net = layers.conv2d(net, num_filters, kernel_size=k)
    #if is_training: 
      #net = tf.nn.dropout(net, keep_prob=0.8)
  return net

def dense_block(net, size, k, name, is_training):
  with tf.variable_scope(name):
    outputs = []
    for i in range(size):
      x = net
      net = layer(net, k, 'layer'+str(i), is_training)
      net = tf.concat([x, net], 3)
  print(net)
  return net

def transition(net, compression, name):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net)
    net = tf.nn.relu(net)
    num_filters = net.get_shape().as_list()[3]
    num_filters = int(round(num_filters*compression))
    net = layers.convolution2d(net, num_filters, kernel_size=1)
    net = layers.max_pool2d(net, 2, stride=2, padding='SAME')
    #net = layers.avg_pool2d(net, 2, stride=2, padding='SAME')
  print(net)
  return net

def _build(image, depth, is_training):
  bn_params['is_training'] = is_training
  block_sizes = [6,12,24,16]
  k = 32
  compression = 0.5
  
  with arg_scope([layers.convolution2d, layers.convolution2d_transpose],
      stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.convolution2d(image, 2*k, 7, stride=2, scope='conv0')
    #net = layers.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool0')
    #net = layers.max_pool2d(net, 2, stride=2, padding='SAME', scope='pool0')
    #net = layers.convolution2d(image, 2*k, 3, scope='conv0')
    #net = dense_block(net, 2, k, 'block0', is_training)
    #net = transition(net, compression, 'pool0')

    depth = tf.image.resize_nearest_neighbor(depth, tf.shape(net)[1:3],
              name='resize_depth')
    net = tf.concat([depth, net], 3)
    skip_layers = []
    km = 256
    net = dense_block(net, block_sizes[0], k, 'block1', is_training)
    skip_layers.append([net, km//2, 'block1'])
    net = transition(net, compression, 'pool1')
    net = dense_block(net, block_sizes[1], k, 'block2', is_training)
    skip_layers.append([net, km//2, 'block2'])
    net = transition(net, compression, 'pool2')
    net = dense_block(net, block_sizes[2], k, 'block3', is_training)
    skip_layers.append([net, km, 'block3'])
    net = transition(net, compression, 'pool3')
    net = dense_block(net, block_sizes[3], k, 'block4', is_training)
    skip_layers.append([net, km, 'block4'])
    net = transition(net, compression, 'pool4')
    print(net)

  with tf.variable_scope('head'):
    #net = layers.conv2d(net, 512, kernel_size=5, rate=2, scope='head_conv1')
    net = BNReluConv(net, 512, 'conv1', is_training, k=5, rate=2)
  for skip_layer in reversed(skip_layers):
    net = build_refinement_module(net, skip_layer)
    print(net)

  logits = layers.conv2d(net, FLAGS.num_classes, 1, activation_fn=None, scope='logits')
  logits = tf.image.resize_bilinear(logits, [FLAGS.img_height, FLAGS.img_width],
                                    name='resize_logits')
  return logits


def build(dataset, is_training, reuse=False):
  with tf.variable_scope('', reuse=reuse):
    x, labels, weights, depth, img_names = reader.inputs(dataset, is_training=is_training, num_epochs=FLAGS.max_epochs)
    x, depth = normalize_input(x, depth)

    logits = _build(x, depth, is_training)
    total_loss = _loss(logits, labels, weights, is_training)

    if is_training:
      return [total_loss], None, None
    else:
      return [total_loss, logits, labels, img_names]


def _loss(logits, labels, weights, is_training=True):
  xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=10)
  all_losses = [xent_loss]

  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)

  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss

def minimize(loss, global_step, num_batches):
  decay_steps = int(num_batches * FLAGS.num_epochs_per_decay)
  # Decay the learning rate exponentially based on the number of steps.
  global lr
  lr = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step, decay_steps,
                                  FLAGS.learning_rate_decay_factor, staircase=True)
  tf.summary.scalar('learning_rate', lr)
  print('Using optimizer: Adam')
  opt = tf.train.AdamOptimizer(lr)
  grads = opt.compute_gradients(loss)
  all_vars = tf.contrib.framework.get_variables()
  #for v in all_vars:
  #  print(v.name)

  train_op = opt.apply_gradients(grads, global_step=global_step)
  return train_op

def train_step(sess, run_ops):
  return sess.run(run_ops)

def num_batches(dataset):
  return reader.num_examples(dataset)

def num_examples(dataset):
  return reader.num_examples(dataset)

