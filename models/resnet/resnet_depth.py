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
import datasets.reader_rgbd_depth as reader

FLAGS = tf.app.flags.FLAGS

#from tensorpack import *
#from tensorpack.utils import logger
#from tensorpack.utils.stat import RatioCounter
#from tensorpack.tfutils.symbolic_functions import *
#from tensorpack.tfutils.summary import *
#from tensorpack.dataflow.dataset import ILSVRCMeta

MODEL_DEPTH = 50
#MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]
MEAN_BGR = [75.08929598, 85.01498926, 75.2051479]
#MEAN_BGR = [103.939, 116.779, 123.68]


def evaluate(sess, epoch_num, run_ops, dataset):
  eval_helper.evaluate_depth_prediction(sess, epoch_num, run_ops, num_examples(dataset))

def normalize_input(img):
  return img - MEAN_BGR
  #"""Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
  #with tf.name_scope('input'), tf.device('/cpu:0'):
  #  #rgb -= MEAN_RGB
  #  red, green, blue = tf.split(3, 3, rgb)
  #  bgr = tf.concat(3, [blue, green, red])
  #  #bgr -= MEAN_BGR
  #  return bgr

def _build(image, is_training):
  #def BatchNorm(x, use_local_stat=None, decay=0.9, epsilon=1e-5):
  weight_decay = 1e-4
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
    'is_training': is_training,
  }
  #init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
  init_func = layers.variance_scaling_initializer()

  def shortcut(l, n_in, n_out, stride):
    if n_in != n_out:
      return layers.convolution2d(l, n_out, kernel_size=1, stride=stride,
                                  activation_fn=None, scope='convshortcut')
      #l = Conv2D('convshortcut', l, n_out, 1, stride=stride)
      #return BatchNorm('bnshortcut', l)
    else:
      return l

  def bottleneck(l, ch_out, stride, preact):
    ch_in = l.get_shape().as_list()[-1]
    if preact == 'both_preact':
      l = tf.nn.relu(l, name='preact-relu')
    bottom_in = l
    with arg_scope([layers.convolution2d],
      stride=1, padding='SAME', activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=init_func,
      #weights_initializer=layers.variance_scaling_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):

      l = layers.convolution2d(l, ch_out, kernel_size=1, stride=stride, scope='conv1')
      l = layers.convolution2d(l, ch_out, kernel_size=3, scope='conv2')
      l = layers.convolution2d(l, ch_out * 4, kernel_size=1, activation_fn=None, scope='conv3')
      return l + shortcut(bottom_in, ch_in, ch_out * 4, stride)

  def layer(l, layername, features, count, stride, first=False):
    with tf.variable_scope(layername):
      with tf.variable_scope('block0'):
        l = bottleneck(l, features, stride, 'no_preact' if first else 'both_preact')
      for i in range(1, count):
        with tf.variable_scope('block{}'.format(i)):
          l = bottleneck(l, features, 1, 'both_preact')
      return l

  cfg = {
      50: ([3,4,6,3]),
      101: ([3,4,23,3]),
      152: ([3,8,36,3])
  }
  defs = cfg[MODEL_DEPTH]
  
  #image = tf.pad(image, [[0,0],[3,3],[3,3],[0,0]])
  #l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
  #l = layers.convolution2d(image, 64, 7, stride=2, padding='VALID',
  l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
      activation_fn=tf.nn.relu, weights_initializer=init_func,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_regularizer=layers.l2_regularizer(weight_decay), scope='conv0')
  #l = layers.max_pool2d(l, 3, stride=2, padding='SAME', scope='pool0')
  l = layers.max_pool2d(l, 3, stride=1, padding='SAME', scope='pool0')
  l = layer(l, 'group0', 64, defs[0], 1, first=True)
  l = layer(l, 'group1', 128, defs[1], 2)
  l = layer(l, 'group2', 256, defs[2], 2)
  l = layer(l, 'group3', 512, defs[3], 2)
  #l = layer(l, 'group3', 512, defs[3], 1)
  l = tf.nn.relu(l)
  #in_k = l.get_shape().as_list()[-2]
  #print(l.get_shape().as_list())
  #print(l)
  #l = layers.avg_pool2d(l, kernel_size=in_k, scope='global_avg_pool')
  #l = layers.flatten(l, scope='flatten')
  #logits = layers.fully_connected(l, FLAGS.num_classes, activation_fn=None, scope='fc1000')

  with arg_scope([layers.convolution2d],
      stride=1, padding='SAME', activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=init_func,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
      l = layers.convolution2d(l, 512, kernel_size=1, scope='conv1') # faster
      #l = layers.convolution2d(l, 512, kernel_size=5, rate=4, scope='conv2')
      l = layers.convolution2d(l, 512, kernel_size=5, scope='conv2')
      l = layers.convolution2d(l, 512, kernel_size=3, scope='conv3')
  logits = layers.convolution2d(l, 1, 1, padding='SAME',
      activation_fn=None, weights_initializer=init_func,
      #normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      #weights_regularizer=layers.l2_regularizer(weight_decay), scope='logits')
      weights_regularizer=None, scope='logits')

  logits = tf.image.resize_bilinear(logits, [FLAGS.img_height, FLAGS.img_width],
                                    name='resize_score')
  return logits
  

  #with argscope(Conv2D, nl=tf.identity, use_bias=False,
  #              W_init=variance_scaling_initializer(mode='FAN_OUT')):
  #  # tensorflow with padding=SAME will by default pad [2,3] here.
  #  # but caffe conv with stride will pad [3,3]
  #  image = tf.pad(image, [[0,0],[3,3],[3,3],[0,0]])

  #  fc1000 = (LinearWrap(image)
  #      .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU, padding='VALID')
  #      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
  #      .apply(layer, 'group0', 64, defs[0], 1, first=True)
  #      .apply(layer, 'group1', 128, defs[1], 2)
  #      .apply(layer, 'group2', 256, defs[2], 2)
  #      .apply(layer, 'group3', 512, defs[3], 2)())
  #      #.tf.nn.relu()
  #      #.GlobalAvgPooling('gap')
  #      #.FullyConnected('fc1000', 1000, nl=tf.identity)())
  ##prob = tf.nn.softmax(fc1000, name='prob')
  ##nr_wrong = prediction_incorrect(fc1000, label, name='wrong-top1')
  ##nr_wrong = prediction_incorrect(fc1000, label, 5, name='wrong-top5')


def name_conversion(caffe_layer_name, prefix=''):
  """ Convert a caffe parameter name to a tensorflow parameter name as
      defined in the above model """
  # beginning & end mapping
  NAME_MAP = {
      'bn_conv1/beta': 'conv0/BatchNorm/beta:0',
      'bn_conv1/gamma': 'conv0/BatchNorm/gamma:0',
      'bn_conv1/mean/EMA': 'conv0/BatchNorm/moving_mean:0',
      'bn_conv1/variance/EMA': 'conv0/BatchNorm/moving_variance:0',
      'conv1/W': 'conv0/weights:0', 'conv1/b': 'conv0/biases:0',
      'fc1000/W': 'fc1000/weights:0', 'fc1000/b': 'fc1000/biases:0'}
  if caffe_layer_name in NAME_MAP:
    return prefix + NAME_MAP[caffe_layer_name]

  s = re.search('([a-z]+)([0-9]+)([a-z]+)_', caffe_layer_name)
  if s is None:
    s = re.search('([a-z]+)([0-9]+)([a-z]+)([0-9]+)_', caffe_layer_name)
    layer_block_part1 = s.group(3)
    layer_block_part2 = s.group(4)
    assert layer_block_part1 in ['a', 'b']
    layer_block = 0 if layer_block_part1 == 'a' else int(layer_block_part2)
  else:
    layer_block = ord(s.group(3)) - ord('a')
  layer_type = s.group(1)
  layer_group = s.group(2)

  layer_branch = int(re.search('_branch([0-9])', caffe_layer_name).group(1))
  assert layer_branch in [1, 2]
  if layer_branch == 2:
    layer_id = re.search('_branch[0-9]([a-z])/', caffe_layer_name).group(1)
    layer_id = ord(layer_id) - ord('a') + 1

  TYPE_DICT = {'res':'conv', 'bn':'BatchNorm'}
  name_map = {'/W': '/weights:0', '/b': '/biases:0', '/beta': '/beta:0',
              '/gamma': '/gamma:0', '/mean/EMA': '/moving_mean:0',
              '/variance/EMA': '/moving_variance:0'}

  tf_name = caffe_layer_name[caffe_layer_name.index('/'):]
  #print(tf_name)
  if tf_name in name_map:
    tf_name = name_map[tf_name]
  #print(layer_type)
  #if layer_type != 'bn':
  if layer_type == 'res':
    layer_type = TYPE_DICT[layer_type] + (str(layer_id) if layer_branch == 2 else 'shortcut')
  elif layer_branch == 2:
    layer_type = 'conv' + str(layer_id) + '/' + TYPE_DICT[layer_type]
  elif layer_branch == 1:
    layer_type = 'convshortcut/' + TYPE_DICT[layer_type]
  tf_name = 'group{}/block{}/{}'.format(int(layer_group) - 2,
      layer_block, layer_type) + tf_name
  return prefix + tf_name


def create_init_op(params):
  variables = tf.contrib.framework.get_variables()
  init_map = {}
  for var in variables:
    #name_split = var.name.split('/')
    #if len(name_split) != 3:
    #  continue
    #name = name_split[1] + '/' + name_split[2][:-2]
    name = var.name
    if name in params:
      #print(var.name, ' --> found init')
      init_map[var.name] = params[name]
      del params[name]
    else:
      print(var.name, ' --> init not found!')
      #raise 1
  print(list(params.keys()))
  #print(params['conv0/biases:0'].sum())
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
  return init_op, init_feed


def build(dataset, is_training, reuse=False):
  # Get images and labels.
  x, y, x_names = reader.inputs(dataset, shuffle=is_training, num_epochs=FLAGS.max_epochs)
  x = normalize_input(x)

  if reuse:
    tf.get_variable_scope().reuse_variables()
  if is_training:
    MODEL_PATH ='/home/kivan/datasets/pretrained/resnet/ResNet'+str(MODEL_DEPTH)+'.npy'
    param = np.load(MODEL_PATH, encoding='latin1').item()
    resnet_param = {}
    for k, v in param.items():
      try:
        newname = name_conversion(k)
      except:
        logger.error("Exception when processing caffe layer {}".format(k))
        raise
      #print("Name Transform: " + k + ' --> ' + newname)
      resnet_param[newname] = v
      #print(v.shape)

  y_pred = _build(x, is_training)
  total_loss = loss(y_pred, y, is_training)

  #all_vars = tf.contrib.framework.get_variables()
  #for v in all_vars:
  #  print(v.name)
  if is_training:
    init_op, init_feed = create_init_op(resnet_param)
    return [total_loss], init_op, init_feed
  else:
    return [total_loss, y_pred, y, x, x_names]


def loss(yp, yt, is_training=True):
  loss_val = losses.mse(yp, yt)
  #loss_tf = tf.contrib.losses.softmax_cross_entropy()
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, num_labels)
  #loss_val = losses.weighted_hinge_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss_symmetric(logits, labels, weights, num_labels)
  all_losses = [loss_val]

  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)

  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss


def num_examples(dataset):
  return reader.num_examples(dataset)
