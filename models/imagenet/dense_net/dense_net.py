import time
import pickle
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

# RGB
DATA_MEAN = [123.68, 116.779, 103.939]
DATA_STD = [70.59564226, 68.52497082, 71.41913876]
#MEAN_BGR = [103.939, 116.779, 123.68]
#DATA_STD = [71.41913876, 68.52497082, 70.59564226]

model_depth = 121
init_dir = '/home/kivan/datasets/pretrained/dense_net/'

weight_decay = 1e-4
#init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
init_func = layers.variance_scaling_initializer()

block_sizes = [6,12,24,16]
k = 32
compression = 0.5

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


def normalize_input(img):
  return (img - DATA_MEAN) / DATA_STD

def BNReluConv(net, num_filters, name, is_training, k=3):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    net = layers.conv2d(net, num_filters, kernel_size=k)
  return net

def layer(net, num_filters, name, is_training, k=3):
  with tf.variable_scope(name):
    with tf.variable_scope('bottleneck'):
      net = tf.contrib.layers.batch_norm(net, **bn_params)
      net = tf.nn.relu(net)
      net = layers.conv2d(net, 4*num_filters, kernel_size=1)
    with tf.variable_scope('conv'):
      net = tf.contrib.layers.batch_norm(net, **bn_params)
      net = tf.nn.relu(net)
      net = layers.conv2d(net, num_filters, kernel_size=k)
    #if is_training: 
      #net = tf.nn.dropout(net, keep_prob=0.8)
  return net

def dense_block(net, size, k, name, is_training):
  with tf.variable_scope(name):
    for i in range(size):
      x = net
      net = layer(net, k, 'layer'+str(i), is_training)
      net = tf.concat([x, net], 3)
  print(net)
  return net

def transition(net, compression, name):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    num_filters = net.get_shape().as_list()[3]
    num_filters = int(round(num_filters*compression))
    net = layers.convolution2d(net, num_filters, kernel_size=1)
    net = layers.avg_pool2d(net, 2, stride=2, padding='SAME')
  print(net)
  return net

def build(image, is_training=False):
  bn_params['is_training'] = is_training
  image = normalize_input(image)
  
  with arg_scope([layers.convolution2d, layers.convolution2d_transpose],
      stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.variable_scope('conv0'):
      net = layers.convolution2d(image, 2*k, 7, stride=2)
      net = tf.contrib.layers.batch_norm(net, **bn_params)
      net = tf.nn.relu(net)

    net = layers.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool0')
    net = dense_block(net, block_sizes[0], k, 'block0', is_training)
    net = transition(net, compression, 'block0/transition')
    net = dense_block(net, block_sizes[1], k, 'block1', is_training)
    net = transition(net, compression, 'block1/transition')
    net = dense_block(net, block_sizes[2], k, 'block2', is_training)
    net = transition(net, compression, 'block2/transition')
    net = dense_block(net, block_sizes[3], k, 'block3', is_training)
    print(net)

  with tf.variable_scope('head'):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    in_k = net.get_shape().as_list()[-2]
    net = layers.avg_pool2d(net, kernel_size=in_k, scope='global_avg_pool')
    net = layers.flatten(net, scope='flatten')
    logits = layers.fully_connected(net, 1000, activation_fn=None, scope='fc1000')
    prob = tf.nn.softmax(logits)
    return logits, prob


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
  logits, prob = build(image, is_training=False)
  #init_op, init_feed = create_init_op(resnet_param)
  init_path = init_dir + 'dense_net_' + str(model_depth) + '.pickle'
  #init_map = np.load(init_path)
  with open(init_path, 'rb') as f:
    init_map = pickle.load(f)
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)

  all_vars = tf.contrib.framework.get_variables()
  for v in all_vars:
    #print(v.name)
    if v.name in init_map:
      pass
      #del init_map[v.name]
    else:
      print(v.name)
      raise 'Not in'
  #print('Dont exist: ', list(init_map.keys()))


  sess = tf.Session()
  #sess.run(tf.initialize_all_variables())
  #sess.run(tf.initialize_local_variables())
  sess.run(init_op, feed_dict=init_feed)

  batch_size = 100

  data_path = '/home/kivan/datasets/imagenet/ILSVRC2015/numpy/val_data.hdf5'
  h5f = h5py.File(data_path, 'r')
  data_x = h5f['data_x'][()]
  print(data_x.shape)
  data_y = h5f['data_y'][()]
  h5f.close()
  y_pred = np.zeros((data_y.shape[0]), dtype=np.int32)
  #data_std = data_x.std((0,1,2), dtype=np.float64)
  #print(data_std)
  b_tmp = data_x[...,0].copy()
  data_x[...,0] = data_x[...,2]
  data_x[...,2] = b_tmp

  N = data_x.shape[0]
  assert N % batch_size == 0
  num_batches = N // batch_size

  top5_error = tf.nn.in_top_k(logits, labels, 5)
  top5_wrong = 0
  cnt_wrong = 0
  #num_batches=100
  for i in range(num_batches):
    offset = i * batch_size
    batch_x = data_x[offset:offset+batch_size, ...]
    batch_y = data_y[offset:offset+batch_size, ...]
    #print(batch_y)
    start_time = time.time()
    #batch_x = batch_x.astype(np.float32)
    #batch_x -= MEAN_BGR
    #batch_x /= DATA_STD
    #batch_x -= MEAN_RGB
    #batch_x /= DATA_STD_RGB
    #print(batch_x.mean((0,1,2)))
    #print(batch_x.std((0,1,2)))
    logits_val, prob_val, top5 = sess.run([logits, prob, top5_error],
        feed_dict={image:batch_x, labels:batch_y})
    #print(np.max(prob_val, 1))
    duration = time.time() - start_time
    num_examples_per_step = batch_size
    examples_per_sec = num_examples_per_step / duration
    sec_per_batch = float(duration)

    top5_wrong += (top5==0).sum()
    yp = logits_val.argmax(1).astype(np.int32)
    y_pred[offset:offset+batch_size] = yp
    cnt_wrong += (yp != batch_y).sum()
    if i % 10 == 0:
      print('[%d / %d] top1error = %.2f - top5error = %.2f (%.1f examples/sec; %.3f sec/batch)' % (i, num_batches,
            cnt_wrong / ((i+1)*batch_size) * 100, top5_wrong / ((i+1)*batch_size) * 100,
            examples_per_sec, sec_per_batch))
  print(cnt_wrong / N)
  print(top5_wrong / N)

#data_y = data_y[:num_batches*batch_size]
#for i in range(100):
#  mask = data_y == i
#  print(y_pred[mask])
