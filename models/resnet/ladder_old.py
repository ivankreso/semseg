import os
import tensorflow as tf
import argparse
import numpy as np
import cv2


import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

import models.resnet.resnet_utils as resnet_utils
import eval_helper
import losses
import datasets.reader as reader
#import datasets.flip_reader as reader
#import datasets.reader_jitter as reader

FLAGS = tf.app.flags.FLAGS

HEAD_PREFIX = 'head'

MODEL_DEPTH = 50
#MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]
MEAN_BGR = [75.08929598, 85.01498926, 75.2051479]
#MEAN_BGR = [103.939, 116.779, 123.68]
#init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
init_func = layers.variance_scaling_initializer()
weight_decay = 1e-4
#apply_jitter = True
apply_jitter = False


def evaluate(name, sess, epoch_num, run_ops, dataset, data):
  #TODO iIOU
  loss_val, accuracy, iou, recall, precision = eval_helper.evaluate_segmentation(
      sess, epoch_num, run_ops, dataset.num_examples(), get_feed_dict=get_valid_feed)
  if iou > data['best_iou'][0]:
    data['best_iou'] = [iou, epoch_num]
  data['iou'] += [iou]
  data['acc'] += [accuracy]
  data['loss'] += [loss_val]

def plot_results(train_data, valid_data):
  eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats'),
                                     train_data, valid_data)
  #eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats')), train_data)


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
  #  red, green, blue = tf.split(3, 3, rgb)
  #  bgr = tf.concat(3, [blue, green, red])
  #  #bgr -= MEAN_BGR
  #  return bgr

def upsample(net, name):
  with tf.variable_scope(name):
    num_filters = net.get_shape().as_list()[3]
    net = tf.contrib.layers.convolution2d_transpose(net, num_filters, kernel_size=3, stride=2)
    return net

PADDINGS = [[0, 0], [0, 0]]
CROPS = [[0, 0], [0, 0]]

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

  #print('size_top = ', top_height, top_width, size_top)

  with arg_scope([layers.convolution2d],
    padding='SAME', activation_fn=tf.nn.relu,
    normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
    weights_initializer=init_func,
    weights_regularizer=layers.l2_regularizer(weight_decay)):
    print(skip_name, top_layer, skip_layer)
    skip_layer = tf.nn.relu(skip_layer)
    skip_layer = layers.convolution2d(skip_layer, size_top, kernel_size=3,
                                      scope=skip_name+'_refine_prep')
  #skip_layer = convolve(skip_layer, size_top, 3, skip_name + '_refine_prep')
    net = tf.concat(3, [top_layer, skip_layer])
  #print(net)
    net = layers.convolution2d(net, size_bottom, kernel_size=3,
                               scope=skip_name+'_refine_fuse')
    #wAAAAS BUUUUUUUG
    #net = layers.convolution2d(net, size_top, kernel_size=3,
    #net = convolve(net, size_bottom, 3, skip_name + '_refine_fuse')
  return net


def _build(image, is_training):
  #def BatchNorm(x, use_local_stat=None, decay=0.9, epsilon=1e-5):
  global bn_params
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

  def shortcut(l, n_in, n_out, stride):
    if n_in != n_out:
      return layers.convolution2d(l, n_out, kernel_size=1, stride=stride,
                                  activation_fn=None, scope='convshortcut')
      #l = Conv2D('convshortcut', l, n_out, 1, stride=stride)
      #return BatchNorm('bnshortcut', l)
    else:
      return l

  def bottleneck(l, ch_out, stride, preact, rate=1):
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

      if rate > 1:
        l = tf.space_to_batch(l, paddings=PADDINGS, block_size=rate)
      l = layers.convolution2d(l, ch_out, kernel_size=3, scope='conv2')
      l = layers.convolution2d(l, ch_out * 4, kernel_size=1, activation_fn=None, scope='conv3')
      if rate > 1:
        l = tf.batch_to_space(l, crops=CROPS, block_size=rate)
      return l + shortcut(bottom_in, ch_in, ch_out * 4, stride)

  def layer(l, layername, features, count, stride, rate=1, first=False):
    with tf.variable_scope(layername):
      with tf.variable_scope('block0'):
        l = bottleneck(l, features, stride, 'no_preact' if first else 'both_preact', rate)
      if rate > 1:
        l = tf.space_to_batch(l, paddings=PADDINGS, block_size=rate)
      for i in range(1, count):
        with tf.variable_scope('block{}'.format(i)):
          l = bottleneck(l, features, 1, 'both_preact')
      if rate > 1:
        assert stride == 1
        l = tf.batch_to_space(l, crops=CROPS, block_size=rate)
      return l

  cfg = {
      50: ([3,4,6,3]),
      101: ([3,4,23,3]),
      152: ([3,8,36,3])
  }
  defs = cfg[MODEL_DEPTH]
  
  skip_layers = []
  km = 128
  #image = tf.pad(image, [[0,0],[3,3],[3,3],[0,0]])
  #l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
  #l = layers.convolution2d(image, 64, 7, stride=2, padding='VALID',
  l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
      activation_fn=tf.nn.relu, weights_initializer=init_func,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_regularizer=layers.l2_regularizer(weight_decay), scope='conv0')
  l = layers.max_pool2d(l, 3, stride=2, padding='SAME', scope='pool0')
  #l = layers.max_pool2d(l, 3, stride=1, padding='SAME', scope='pool0')
  l = layer(l, 'group0', 64, defs[0], 1, first=True)
  #skip_layers += [[l, km/2, 'group0']]
  l = layer(l, 'group1', 128, defs[1], 2)
  #skip_layers += [l]
  skip_layers += [[l, km, 'group1']]

  bsz = 2
  paddings, crops = tf.required_space_to_batch_paddings(tf.shape(l)[1:3], [bsz, bsz])
  l = tf.space_to_batch(l, paddings=paddings, block_size=bsz)
  l = layer(l, 'group2', 256, defs[2], 1)
  l = tf.batch_to_space(l, crops=crops, block_size=bsz)
  #l = layer(l, 'group2', 256, defs[2], 2)
  #skip_layers += [[l, km, 'group2']]

  bsz = 4
  paddings, crops = tf.required_space_to_batch_paddings(tf.shape(l)[1:3], [bsz, bsz])
  l = tf.space_to_batch(l, paddings=paddings, block_size=bsz)
  l = layer(l, 'group3', 512, defs[3], 1)
  l = tf.batch_to_space(l, crops=crops, block_size=bsz)
  #l = layer(l, 'group3', 512, defs[3], 2)

  l = tf.nn.relu(l)
  print('resnet:', l)

  with tf.variable_scope('head'):
    with arg_scope([layers.convolution2d],
        stride=1, padding='SAME', activation_fn=tf.nn.relu,
        normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
        weights_initializer=init_func,
        weights_regularizer=layers.l2_regularizer(weight_decay)):

      #l = pyramid_pooling(l, 'pyramid_pooling')
      #l = layers.convolution2d(l, 512, kernel_size=3, scope='conv1') # faster
      l = layers.convolution2d(l, 1024, kernel_size=1, scope='conv1') # faster
      l = layers.convolution2d(l, 512, kernel_size=5, rate=2, scope='conv2')
      #l = tf.Print(l, [tf.shape(image)], message='IMG SHAPE = ')
      #l = layers.convolution2d(l, 512, kernel_size=5, rate=8, scope='conv2')
      #l = layers.convolution2d(l, 1024, kernel_size=1, scope='conv2')
      #l = layers.convolution2d(l, 1024, kernel_size=1, activation_fn=None, scope='conv2')

      #for skip_layer in reversed(skip_layers):
      #  l = build_refinement_module(l, skip_layer)

      logits = layers.convolution2d(l, FLAGS.num_classes, 1, padding='SAME',
          activation_fn=None, weights_initializer=init_func, normalizer_fn=None,
          scope='logits')
          #weights_regularizer=None, scope='logits')
    input_shape = tf.shape(image)
    print(input_shape)
    #global resize_height, resize_width
    height = input_shape[1]
    width = input_shape[2]
    #height = tf.Print(height, [height, width], message='SHAPE = ')
    #logits = tf.image.resize_bilinear(logits, [FLAGS.img_height, FLAGS.img_width],
    #logits = tf.image.resize_bilinear(logits, [resize_height, resize_width],
    logits = tf.image.resize_bilinear(logits, [height, width],
                                      name='resize_logits')
  return logits
  


def jitter(image, labels, weights):
  global random_flip_tf, resize_width, resize_height
  random_flip_tf = tf.placeholder(tf.bool, shape=(), name='random_flip')
  resize_width = tf.placeholder(tf.int32, shape=(), name='resize_width')
  resize_height = tf.placeholder(tf.int32, shape=(), name='resize_height')
  
  image_split = tf.unstack(image, axis=0)
  weights_split = tf.unstack(weights, axis=0)
  labels_split = tf.unstack(labels, axis=0)
  out_img = []
  out_weights = []
  out_labels = []
  for i in range(FLAGS.batch_size):
    out_img.append(tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(image_split[i]),
                       lambda: image_split[i]))
    #print(cond_op)
    #image_split[i] = tf.assign(image_split[i], cond_op)
    #image[i] = tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(image[i]),
    #cond_op = tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(image[i]),
                       #lambda: tf.identity(image[i]))
    #image[i] = tf.assign(image[i], cond_op)
    out_labels.append(tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(labels[i]),
                      lambda: labels[i]))
    out_weights.append(tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(weights[i]),
                       lambda: weights[i]))
  image = tf.stack(out_img, axis=0)
  weights = tf.stack(out_weights, axis=0)
  labels = tf.stack(out_labels, axis=0)

  image = tf.image.resize_bicubic(image, [resize_height, resize_width])
  labels = tf.image.resize_nearest_neighbor(labels, [resize_height, resize_width])
  weights = tf.image.resize_nearest_neighbor(weights, [resize_height, resize_width])
  return image, labels, weights

def train_step(sess, run_ops):
  if apply_jitter:
    feed_dict = _get_train_feed()
    vals = sess.run(run_ops, feed_dict=feed_dict)
  else:
    vals = sess.run(run_ops)
  return vals

def _get_train_feed():
  global random_flip_tf, resize_width, resize_height
  random_flip = int(np.random.choice(2, 1))
  #resize_scale = np.random.uniform(0.5, 2)
  #resize_scale = np.random.uniform(0.4, 1.5)
  #resize_scale = np.random.uniform(0.5, 1.2)
  resize_scale = np.random.uniform(0.6, 1.4)
  width = np.int32(int(round(FLAGS.img_width * resize_scale)))
  height = np.int32(int(round(FLAGS.img_height * resize_scale)))
  feed_dict = {random_flip_tf:random_flip, resize_width:width, resize_height:height}
  return feed_dict

def get_valid_feed():
  if not apply_jitter:
    return {}
  global random_flip_tf, resize_width, resize_height
  feed_dict = {random_flip_tf:0, resize_width:0, resize_height:0}
  return feed_dict

def build(dataset, is_training, reuse=False):
  # Get images and labels.
  image, labels, weights, _, img_names = reader.inputs(
      dataset, is_training=is_training, num_epochs=FLAGS.max_epochs)
  if is_training and apply_jitter:
    image, labels, weights = jitter(image, labels, weights)
  image = normalize_input(image)

  if reuse:
    tf.get_variable_scope().reuse_variables()
  if is_training:
    MODEL_PATH ='/home/kivan/datasets/pretrained/resnet/ResNet'+str(MODEL_DEPTH)+'.npy'
    param = np.load(MODEL_PATH, encoding='latin1').item()
    resnet_param = {}
    for k, v in param.items():
      try:
        newname = resnet_utils.name_conversion(k)
      except:
        logger.error("Exception when processing caffe layer {}".format(k))
        raise
      #print("Name Transform: " + k + ' --> ' + newname)
      resnet_param[newname] = v
      #print(v.shape)

  logits = _build(image, is_training)
  total_loss = loss(logits, labels, weights, is_training)

  #all_vars = tf.contrib.framework.get_variables()
  #for v in all_vars:
  #  print(v.name)
  if is_training:
    init_op, init_feed = resnet_utils.create_init_op(resnet_param)
    #return [total_loss], init_op, init_feed
    return [total_loss], None, None
  else:
    return [total_loss, logits, labels, img_names]


def loss(logits, labels, weights, is_training=True):
  # TODO
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=10)
  loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=100)
  #loss_val = losses.flip_xent_loss(logits, labels, weights, max_weight=10)
  #loss_tf = tf.contrib.losses.softmax_cross_entropy()
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights)
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=1)
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

def minimize(loss, global_step):
  # Calculate the learning rate schedule.
  num_batches_per_epoch = num_batches(train_dataset)
  decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay) * 2
  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step, decay_steps,
                                  FLAGS.learning_rate_decay_factor, staircase=True)
  lr2 = tf.train.exponential_decay(10*FLAGS.initial_learning_rate, global_step, decay_steps,
                                  FLAGS.learning_rate_decay_factor, staircase=True)
  tf.summary.scalar('learning_rate', lr)

  print('Using optimizer:', FLAGS.optimizer)
  opts = []
  if FLAGS.optimizer == 'Adam':
    opts += [tf.train.AdamOptimizer(lr)]
    opts += [tf.train.AdamOptimizer(lr2)]
  elif FLAGS.optimizer == 'Momentum':
    opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
    #opt = tf.train.GradientDescentOptimizer(lr)
  elif FLAGS.optimizer == 'RMSprop':
    opt = tf.train.RMSPropOptimizer(lr)
  else:
    raise ValueError()

  return resnet_utils.minimize(opts, loss, global_step, HEAD_PREFIX)

def num_batches(dataset):
  return reader.num_batches(dataset)
