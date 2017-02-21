import os, re
import pickle
import tensorflow as tf
import numpy as np
import cv2

import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

import losses
import eval_helper
#import datasets.reader_rgbd_depth as reader
import datasets.reader_rgbd as reader
#import datasets.reader as reader

FLAGS = tf.app.flags.FLAGS

# RGB
DATA_MEAN =  [75.2051479, 85.01498926, 75.08929598]
DATA_STD =  [46.89434904, 47.63335775, 46.47197535]
#DATA_STD = [103.939, 116.779, 123.68]
#DATA_MEAN = [103.939, 116.779, 123.68]

#MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]
#MEAN_BGR = [75.08929598, 85.01498926, 75.2051479]

model_depth = 121
init_dir = '/home/kivan/datasets/pretrained/dense_net/'

weight_decay = 1e-4
#init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
init_func = layers.variance_scaling_initializer()

block_sizes = [6,12,24,16]
k = 32
compression = 0.5


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


def normalize_input(bgr, depth):
  with tf.name_scope('input'), tf.device('/cpu:0'):
    blue, green, red = tf.split(bgr, 3, 3)
    img = tf.concat([red, green, blue], 3)
    return (img - DATA_MEAN) / DATA_STD, depth - 33

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
    #print(top_height, skip_height)
    #assert(2*top_height == skip_height)
    top_layer = tf.image.resize_bilinear(top_layer, [skip_height, skip_width],
                                         name=skip_name + '_refine_upsample')

  with arg_scope([layers.convolution2d],
    padding='SAME', activation_fn=tf.nn.relu,
    normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
    weights_initializer=init_func,
    weights_regularizer=layers.l2_regularizer(weight_decay)):
    print(skip_name, top_layer, skip_layer)
    skip_layer = tf.nn.relu(skip_layer)

    depth = skip_data[3]
    depth = tf.image.resize_bilinear(depth, tf.shape(skip_layer)[1:3],
                                     name=skip_name+'_resize_depth')
    skip_layer = tf.concat([skip_layer, depth], 3)

    skip_layer = layers.convolution2d(skip_layer, size_top, kernel_size=3,
                                      scope=skip_name+'_refine_prep')
    net = tf.concat([top_layer, skip_layer], 3)
    net = layers.convolution2d(net, size_bottom, kernel_size=3,
                               scope=skip_name+'_refine_fuse')
  return net

def BNReluConv(net, num_filters, name, k=3, rate=1):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    net = layers.conv2d(net, num_filters, kernel_size=k, rate=rate)
  return net

def layer(net, num_filters, name, is_training):
  with tf.variable_scope(name):
    #with tf.variable_scope('bottleneck'):
    #  net = tf.contrib.layers.batch_norm(net, **bn_params)
    #  net = tf.nn.relu(net)
    #  net = layers.conv2d(net, 4*num_filters, kernel_size=1)
    #with tf.variable_scope('conv'):
    #  net = tf.contrib.layers.batch_norm(net, **bn_params)
    #  net = tf.nn.relu(net)
    #  net = layers.conv2d(net, num_filters, kernel_size=k)
    net = BNReluConv(net, 4*num_filters, 'bottleneck', k=1)
    net = BNReluConv(net, num_filters, 'conv', k=3)
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
    #net = layers.avg_pool2d(net, 2, stride=2, padding='SAME')
    net = layers.max_pool2d(net, 2, stride=2, padding='SAME')
  print(net)
  return net

def _build(image, depth, is_training=False):
  bn_params['is_training'] = is_training
  with arg_scope([layers.convolution2d, layers.convolution2d_transpose],
      stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.variable_scope('conv0'):
      net = layers.convolution2d(image, 2*k, 7, stride=2)
      net = tf.contrib.layers.batch_norm(net, **bn_params)
      net = tf.nn.relu(net)

    #skip_layers = []
    #km = 256
    ##net = layers.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool0')
    #net = dense_block(net, block_sizes[0], k, 'block0', is_training)
    #skip_layers.append([net, km//2, 'block0'])
    #net = transition(net, compression, 'block0/transition')
    #net = dense_block(net, block_sizes[1], k, 'block1', is_training)
    #skip_layers.append([net, km//2, 'block1'])
    #net = transition(net, compression, 'block1/transition')
    #net = dense_block(net, block_sizes[2], k, 'block2', is_training)
    #skip_layers.append([net, km, 'block2'])
    #net = transition(net, compression, 'block2/transition')
    #net = dense_block(net, block_sizes[3], k, 'block3', is_training)
    #skip_layers.append([net, km, 'block3'])
    #net = transition(net, compression, 'block3/transition')
    #print(net)
    skip_layers = []
    km = 256
    #net = layers.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool0')
    net = dense_block(net, block_sizes[0], k, 'block0', is_training)
    skip_layers.append([net, km//2, 'block0', depth])
    net = transition(net, compression, 'block0/transition')
    net = dense_block(net, block_sizes[1], k, 'block1', is_training)
    skip_layers.append([net, km//2, 'block1', depth])
    net = transition(net, compression, 'block1/transition')
    net = dense_block(net, block_sizes[2], k, 'block2', is_training)
    skip_layers.append([net, km, 'block2', depth])
    net = transition(net, compression, 'block2/transition')
    net = dense_block(net, block_sizes[3], k, 'block3', is_training)
    skip_layers.append([net, km, 'block3', depth])
    net = transition(net, compression, 'block3/transition')
    print(net)

  #with tf.variable_scope('head'):
  #  net = tf.contrib.layers.batch_norm(net, **bn_params)
  #  net = tf.nn.relu(net)
  #  in_k = net.get_shape().as_list()[-2]
  #  net = layers.avg_pool2d(net, kernel_size=in_k, scope='global_avg_pool')
  #  net = layers.flatten(net, scope='flatten')
  #  logits = layers.fully_connected(net, 1000, activation_fn=None, scope='fc1000')
  #  prob = tf.nn.softmax(logits)
  #  return logits, prob
  with tf.variable_scope('head'):
    #net = layers.conv2d(net, 512, kernel_size=5, rate=2, scope='head_conv1')
    #net = BNReluConv(net, 512, 'conv1', k=5, rate=2)
    net = BNReluConv(net, 512, 'conv1', k=5)
  for skip_layer in reversed(skip_layers):
    net = build_refinement_module(net, skip_layer)
    print(net)

  logits = layers.conv2d(net, FLAGS.num_classes, 1, activation_fn=None, scope='logits')
  logits = tf.image.resize_bilinear(logits, [FLAGS.img_height, FLAGS.img_width],
                                    name='resize_logits')
  return logits

def create_init_op(params):
  variables = tf.contrib.framework.get_variables()
  init_map = {}
  for var in variables:
    name = var.name
    if name in params:
      #print(name, ' --> found init')
      init_map[var.name] = params[name]
      del params[name]
    #else:
    #  print(name, ' --> init not found!')
  print('Unused: ', list(params.keys()))
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
  return init_op, init_feed

def build(dataset, is_training, reuse=False):
  with tf.variable_scope('', reuse=reuse):
    x, labels, weights, depth, img_names = \
      reader.inputs(dataset, is_training=is_training, num_epochs=FLAGS.max_epochs)
    x, depth = normalize_input(x, depth)

    logits = _build(x, depth, is_training)
    total_loss = _loss(logits, labels, weights, is_training)

    init_path = init_dir + 'dense_net_' + str(model_depth) + '.pickle'
    with open(init_path, 'rb') as f:
      init_map = pickle.load(f)
    init_op, init_feed = create_init_op(init_map)
    #init_op, init_feed = None, None
    if is_training:
      return [total_loss], init_op, init_feed
    else:
      return [total_loss, logits, labels, img_names]


def _loss(logits, labels, weights, is_training=True):
  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=10)
  xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=100)
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

