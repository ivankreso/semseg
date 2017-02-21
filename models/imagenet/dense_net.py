import time
import tensorflow as tf
import argparse
import os, re
import numpy as np
import h5py
import skimage as ski
import skimage.data
import skimage.transform
import cv2

import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

FLAGS = tf.app.flags.FLAGS

MEAN_BGR = [103.939, 116.779, 123.68]

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


def normalize_input(rgb):
  return rgb - MEAN_BGR

def BNReluConv(net, num_filters, name, is_training, k=3):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    net = layers.conv2d(net, num_filters, kernel_size=k)
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

def build(image, is_training=False):
  bn_params['is_training'] = is_training
  weight_decay = 1e-4
  #init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
  init_func = layers.variance_scaling_initializer()

  block_sizes = [6,12,24,16]
  k = 32
  compression = 0.5
  
  with arg_scope([layers.convolution2d, layers.convolution2d_transpose],
      stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.variable_scope('conv0'):
      net = layers.convolution2d(image, 2*k, 7, stride=2, scope='conv0')
      net = tf.contrib.layers.batch_norm(net, **bn_params)
      net = tf.nn.relu(net)

    net = layers.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool0')
    net = dense_block(net, block_sizes[0], k, 'block1', is_training)
    net = transition(net, compression, 'trainsition1')
    net = dense_block(net, block_sizes[1], k, 'block2', is_training)
    net = transition(net, compression, 'trainsition2')
    net = dense_block(net, block_sizes[2], k, 'block3', is_training)
    net = transition(net, compression, 'transition3')
    net = dense_block(net, block_sizes[3], k, 'block4', is_training)
    print(net)

  with tf.variable_scope('head'):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    in_k = net.get_shape().as_list()[-2]
    net = layers.avg_pool2d(net, kernel_size=in_k, scope='global_avg_pool')
    net = layers.flatten(net, scope='flatten')
    logits = layers.fully_connected(net, 1000, activation_fn=None, scope='fc1000')
    return logits


def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

if __name__ == '__main__':
  img_size = 224
  image = tf.placeholder(tf.float32, [None, img_size, img_size, 3], 'input')
  labels = tf.placeholder(tf.int32, [None], 'label')
  logits = build(image, is_training=False)
  all_vars = tf.contrib.framework.get_variables()
  for v in all_vars:
    print(v.name)
    print(v)
  #init_op, init_feed = create_init_op(resnet_param)

  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  sess.run(tf.initialize_local_variables())
  #sess.run(init_op, feed_dict=init_feed)

  batch_size = 100

  data_path = '/home/kivan/datasets/imagenet/ILSVRC2015/numpy/val_data.hdf5'
  h5f = h5py.File(data_path, 'r')
  data_x = h5f['data_x'][()]
  print(data_x.shape)
  data_y = h5f['data_y'][()]
  h5f.close()

  N = data_x.shape[0]
  assert N % batch_size == 0
  num_batches = N // batch_size

  top5_error = tf.nn.in_top_k(logits, labels, 5)
  top5_wrong = 0
  cnt_wrong = 0
  for i in range(num_batches):
    offset = i * batch_size
    batch_x = data_x[offset:offset+batch_size, ...]
    batch_y = data_y[offset:offset+batch_size, ...]
    start_time = time.time()
    logits_val, top5 = sess.run([logits, top5_error], feed_dict={image:batch_x, labels:batch_y})
    duration = time.time() - start_time
    num_examples_per_step = batch_size
    examples_per_sec = num_examples_per_step / duration
    sec_per_batch = float(duration)

    top5_wrong += (top5==0).sum()
    yp = logits_val.argmax(1).astype(np.int32)
    cnt_wrong += (yp != batch_y).sum()
    if i % 10 == 0:
      print('[%d / %d] top1error = %.2f - top5error = %.2f (%.1f examples/sec; %.3f sec/batch)' % (i, num_batches,
            cnt_wrong / ((i+1)*batch_size) * 100, top5_wrong / ((i+1)*batch_size) * 100,
            examples_per_sec, sec_per_batch))
  print(cnt_wrong / N)
  print(top5_wrong / N)
