import os
import tensorflow as tf
import argparse
import numpy as np
import cv2


import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

import libs.cylib as cylib
import models.resnet.resnet_utils as resnet_utils
import train_helper
import eval_helper
import losses
import datasets.reader as reader
from datasets.cityscapes.cityscapes import CityscapesDataset
#import datasets.flip_reader as reader
#import datasets.reader_jitter as reader

FLAGS = tf.app.flags.FLAGS

HEAD_PREFIX = 'head'

train_step_iter = 0
#imagenet_init = True
imagenet_init = False
MODEL_DEPTH = 50
#MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]
MEAN_BGR = [75.08929598, 85.01498926, 75.2051479]
#MEAN_BGR = [103.939, 116.779, 123.68]
#init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
init_func = layers.variance_scaling_initializer()
weight_decay = 1e-4
apply_jitter = True

context_size = 512
compression = 0.5

fused_batch_norm = True
data_format = 'NHWC'
maps_dim = 3
height_dim = 1


def evaluate(name, sess, epoch_num, run_ops, dataset, data):
  loss_val, accuracy, iou, recall, precision = eval_helper.evaluate_segmentation(
      sess, epoch_num, run_ops, dataset.num_examples() // FLAGS.batch_size_valid)
  is_best = False
  if iou > data['best_iou'][0]:
    is_best = True
    data['best_iou'] = [iou, epoch_num]
  data['iou'] += [iou]
  data['acc'] += [accuracy]
  data['loss'] += [loss_val]
  return is_best


def start_epoch(train_data):
  global train_loss_arr, train_conf_mat
  train_conf_mat = np.ascontiguousarray(
      np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64))
  train_loss_arr = []
  train_data['lr'].append(lr.eval())


def end_epoch(train_data):
  pixacc, iou, _, _, _ = eval_helper.compute_errors(
      train_conf_mat, 'Train', CityscapesDataset.CLASS_INFO)
  is_best = False
  if len(train_data['iou']) and iou > max(train_data['iou']):
    is_best = True
  train_data['iou'].append(iou)
  train_data['acc'].append(pixacc)
  train_loss_val = np.mean(train_loss_arr)
  train_data['loss'].append(train_loss_val)
  return is_best


def update_stats(ret_val):
  global train_loss_arr
  loss_val = ret_val[0]
  yp = ret_val[1]
  yt = ret_val[2]
  train_loss_arr.append(loss_val)
  yp = yp.argmax(3).astype(np.int32)
  cylib.collect_confusion_matrix(yp.reshape(-1), yt.reshape(-1), train_conf_mat)


def plot_results(train_data, valid_data):
  eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats'),
                                     train_data, valid_data)


def print_results(train_data, valid_data):
  print('\nBest train IOU = %.2f' % max(train_data['iou']))
  print('Best validation IOU = %.2f (epoch %d)\n' % tuple(valid_data['best_iou']))


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
  FIX FOR RGB
  return img - MEAN_BGR, depth - 33


def resize_tensor(net, shape, name):
  if data_format == 'NCHW':
    net = tf.transpose(net, perm=[0,2,3,1])
  net = tf.image.resize_bilinear(net, shape, name=name)
  if data_format == 'NCHW':
    net = tf.transpose(net, perm=[0,3,1,2])
  return net


def refine(net, skip_data):
  print(skip_data)
  skip_net = skip_data[0]
  num_layers = skip_data[1]
  growth = skip_data[2]
  block_name = skip_data[3]
  depth = skip_data[4]

  #size_top = top_layer.get_shape()[maps_dim].value
  #skip_width = skip_layer.get_shape()[2].value
  #if top_height != skip_height or top_width != skip_width:
    #print(top_height, skip_height)
    #assert(2*top_height == skip_height)
  
  #TODO try convolution2d_transpose
  #up_shape = tf.shape(skip_net)[height_dim:height_dim+2]
  with tf.variable_scope(block_name):
    up_shape = skip_net.get_shape().as_list()[height_dim:height_dim+2]
    net = resize_tensor(net, up_shape, name='upsample')
    print('\nup = ', net)
    print('skip = ', skip_net)
    return dense_block_upsample(net, skip_net, depth, num_layers, growth, 'dense_block')


def ConvBNRelu(net, num_filters, name, k=3):
  with arg_scope([layers.conv2d],
      stride=1, padding='SAME', activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=init_func,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.conv2d(net, num_filters, kernel_size=k, scope=name)
    return net

up_sizes = [256,256,512,512]
growth_up = 32
#up_sizes = [128,256,512,512]
def dense_block_upsample(net, skip_net, depth, size, growth, name):
  with tf.variable_scope(name):
    net = tf.concat([net, skip_net], maps_dim)
    #new_size = net.get_shape().as_list()[height_dim:height_dim+2]
    #depth = resize_tensor(depth, new_size, 'resize_depth')
    #net = tf.concat([net, skip_net, depth], maps_dim)
    num_filters = net.get_shape().as_list()[maps_dim]
    num_filters = int(round(num_filters*compression))
    #num_filters = int(round(num_filters*compression/2))
    #num_filters = int(round(num_filters*0.3))

    # TODO try 3 vs 1 -> 3 not helping
    net = ConvBNRelu(net, num_filters, 'bottleneck', k=1)
    #net = BNReluConv(net, num_filters, 'bottleneck', k=3)
    #net = tf.concat([net, depth], maps_dim)
    #net = BNReluConv(net, num_filters, 'bottleneck', k=3)
    print('after bottleneck = ', net)
    net = ConvBNRelu(net, size, 'layer')
  return net


def _build(image, depth, is_training=False):
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
    'fused': fused_batch_norm,
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
      101: ([3,4,23,3]),
      152: ([3,8,36,3])
  }
  defs = [3,4,6,3]
  
  skip_layers = []
  km = 256
  #image = tf.pad(image, [[0,0],[3,3],[3,3],[0,0]])
  #l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
  #l = layers.convolution2d(image, 64, 7, stride=2, padding='VALID',
  l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
      activation_fn=tf.nn.relu, weights_initializer=init_func,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_regularizer=layers.l2_regularizer(weight_decay), scope='conv0')
  l = layers.max_pool2d(l, 2, stride=2, padding='SAME', scope='pool0')
  #l = layers.max_pool2d(l, 3, stride=1, padding='SAME', scope='pool0')
  l = layer(l, 'group0', 64, defs[0], 1, first=True)
  #skip_layers.append([tf.nn.relu(l), km//2, 'group0', depth])
  skip_layers.append([tf.nn.relu(l), up_sizes[0], growth_up, 'block0_refine', depth])
  l = layer(l, 'group1', 128, defs[1], 2)
  #skip_layers.append([tf.nn.relu(l), km//2, 'group1', depth])
  skip_layers.append([tf.nn.relu(l), up_sizes[1], growth_up, 'block1_refine', depth])

  #bsz = 2
  #paddings, crops = tf.required_space_to_batch_paddings(tf.shape(l)[1:3], [bsz, bsz])
  #l = tf.space_to_batch(l, paddings=paddings, block_size=bsz)
  #l = layer(l, 'group2', 256, defs[2], 1)
  #l = tf.batch_to_space(l, crops=crops, block_size=bsz)
  l = layer(l, 'group2', 256, defs[2], 2)
  #skip_layers.append([tf.nn.relu(l), km, 'group2', depth])
  skip_layers.append([tf.nn.relu(l), up_sizes[2], growth_up, 'block2_refine', depth])

  #bsz = 4
  #paddings, crops = tf.required_space_to_batch_paddings(tf.shape(l)[1:3], [bsz, bsz])
  #l = tf.space_to_batch(l, paddings=paddings, block_size=bsz)
  #l = layer(l, 'group3', 512, defs[3], 1)
  #l = tf.batch_to_space(l, crops=crops, block_size=bsz)
  l = layer(l, 'group3', 512, defs[3], 2)
  #skip_layers.append([tf.nn.relu(l), km, 'group3', depth])
  l = tf.nn.relu(l)
  print('resnet:', l)

  with tf.variable_scope('head'):
    with arg_scope([layers.conv2d],
        stride=1, padding='SAME', activation_fn=tf.nn.relu,
        normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
        weights_initializer=init_func,
        weights_regularizer=layers.l2_regularizer(weight_decay)):
      l = layers.conv2d(l, 1024, kernel_size=1, scope='conv1')
      l = layers.conv2d(l, context_size, kernel_size=5, scope='conv2')
      logits_mid = l
      #final_h = l.get_shape().as_list()[1]
      #if final_h >= 10:
      #  skip_layers.append([l, km, 'head', depth])
      #  l = layers.max_pool2d(l, 2, stride=2, padding='SAME')
      #if final_h >= 7:
      #  l = layers.conv2d(l, context_size, kernel_size=7, scope='conv2')
      #elif final_h >= 5:
      #  l = layers.conv2d(l, context_size, kernel_size=5, scope='conv2')
      #else:
      #  l = layers.conv2d(l, context_size, kernel_size=3, scope='conv2')

    print('Before upsampling: ', l)
    for skip_layer in reversed(skip_layers):
      l = refine(l, skip_layer)
      print(l)

    logits_mid = layers.conv2d(logits_mid, FLAGS.num_classes, 1, weights_initializer=init_func,
                           activation_fn=None, scope='logits_mid')
    logits = layers.conv2d(l, FLAGS.num_classes, 1, weights_initializer=init_func,
                           activation_fn=None, scope='logits')
    input_shape = tf.shape(image)[1:3]
    logits_mid = tf.image.resize_bilinear(logits_mid, input_shape, name='resize_logits_mid')
    logits = tf.image.resize_bilinear(logits, input_shape, name='resize_logits')
    return logits, logits_mid


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



def jitter(image, labels, depth):
  with tf.name_scope('jitter'), tf.device('/cpu:0'):
    print('\nJittering enabled')
    global random_flip_tf, resize_width, resize_height
    #random_flip_tf = tf.placeholder(tf.bool, shape=(), name='random_flip')
    random_flip_tf = tf.placeholder(tf.bool, shape=(FLAGS.batch_size), name='random_flip')
    resize_width = tf.placeholder(tf.int32, shape=(), name='resize_width')
    resize_height = tf.placeholder(tf.int32, shape=(), name='resize_height')
    
    #image_split = tf.unstack(image, axis=0)
    #depth_split = tf.unstack(depth, axis=0)
    #weights_split = tf.unstack(weights, axis=0)
    #labels_split = tf.unstack(labels, axis=0)
    out_img = []
    out_depth = []
    #out_weights = []
    out_labels = []
    #image = tf.Print(image, [image[0]], message='img1 = ', summarize=10)
    for i in range(FLAGS.batch_size):
      out_img.append(tf.cond(random_flip_tf[i],
        lambda: tf.image.flip_left_right(image[i]), lambda: image[i]))
        #lambda: tf.image.flip_left_right(image_split[i]),
        #lambda: image_split[i]))
      out_depth.append(tf.cond(random_flip_tf[i],
        lambda: tf.image.flip_left_right(depth[i]),
        lambda: depth[i]))
      #print(cond_op)
      #image_split[i] = tf.assign(image_split[i], cond_op)
      #image[i] = tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(image[i]),
      #cond_op = tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(image[i]),
                         #lambda: tf.identity(image[i]))
      #image[i] = tf.assign(image[i], cond_op)
      print(labels)
      out_labels.append(tf.cond(random_flip_tf[i], lambda: tf.image.flip_left_right(labels[i]),
                        lambda: labels[i]))
      #out_weights.append(tf.cond(random_flip_tf[i], lambda: tf.image.flip_left_right(weights[i]),
      #                   lambda: weights[i]))
    image = tf.stack(out_img, axis=0)
    depth = tf.stack(out_depth, axis=0)
    #weights = tf.stack(out_weights, axis=0)
    labels = tf.stack(out_labels, axis=0)
    #image = tf.Print(image, [random_flip_tf], message='random_flip_tf = ', summarize=10)
    #image = tf.Print(image, [image[0]], message='img = ', summarize=10)

    # TODO
    #image = tf.image.resize_bicubic(image, [resize_height, resize_width])
    #depth = tf.image.resize_bilinear(depth, [resize_height, resize_width])
    #labels = tf.image.resize_nearest_neighbor(labels, [resize_height, resize_width])
    #weights = tf.image.resize_nearest_neighbor(weights, [resize_height, resize_width])
    #return image, labels, weights, depth
    return image, labels, depth


def _get_train_feed():
  global random_flip_tf, resize_width, resize_height
  #random_flip = int(np.random.choice(2, 1))
  random_flip = np.random.choice(2, FLAGS.batch_size).astype(np.bool)
  #resize_scale = np.random.uniform(0.5, 2)
  #resize_scale = np.random.uniform(0.4, 1.5)
  #resize_scale = np.random.uniform(0.5, 1.2)
  min_resize = 0.7
  max_resize = 1.3
  if train_step_iter == 0:
    resize_scale = max_resize
  else:
    resize_scale = np.random.uniform(min_resize, max_resize)
  width = np.int32(int(round(FLAGS.img_width * resize_scale)))
  height = np.int32(int(round(FLAGS.img_height * resize_scale)))
  feed_dict = {random_flip_tf:random_flip, resize_width:width, resize_height:height}
  return feed_dict



def build(dataset, is_training, reuse=False):
  with tf.variable_scope('', reuse=reuse):
    #x, labels, weights, depth, img_names = \
    x, labels, num_labels, class_hist, depth, img_names = \
        reader.inputs(dataset, is_training=is_training, num_epochs=FLAGS.max_epochs)
    if is_training and apply_jitter:
      x, labels, depth = jitter(x, labels, depth)
    x, depth = normalize_input(x, depth)

    #logits = _build(x, depth, is_training)
    #total_loss = _loss(logits, labels, weights, is_training)
    logits, mid_logits = _build(x, depth, is_training)
    #total_loss = _multiloss(logits, mid_logits, labels, weights, is_training)
    total_loss = _multiloss(logits, mid_logits, labels, num_labels, class_hist, is_training)

    if is_training and imagenet_init:
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
      init_op, init_feed = resnet_utils.create_init_op(resnet_param)
    else:
      init_op, init_feed = None, None

    run_ops = [total_loss, logits, labels, img_names]
    if is_training:
      return run_ops, init_op, init_feed
    else:
      return run_ops

def inference(image, constant_shape=True):
  x = normalize_input(image)
  logits, mid_logits = _build(x, is_training=False)
  return logits, mid_logits


#def _multiloss(logits, mid_logits, labels, weights, is_training=True):
def _multiloss(logits, mid_logits, labels, num_labels, class_hist, is_training):
  #gpu = '/gpu:1'
  gpu = '/gpu:0'
  with tf.device(gpu):
    max_weight = FLAGS.max_weight
    #max_weight = 10
    #max_weight = 50
    loss1 = losses.weighted_cross_entropy_loss(
        logits, labels, num_labels, class_hist, max_weight=max_weight)
    loss2 = losses.weighted_cross_entropy_loss(
        mid_logits, labels, num_labels, class_hist, max_weight=max_weight)
    #loss1 = losses.weighted_cross_entropy_loss(logits, labels, weights,
    #    max_weight=max_weight)
    #loss2 = losses.weighted_cross_entropy_loss(mid_logits, labels, weights,
    #    max_weight=max_weight)
    #wgt = 0.4
    #xent_loss = loss1 + wgt * loss2
    wgt = 0.3 # best
    #wgt = 0.2
    #wgt = 0.4
    xent_loss = (1-wgt)*loss1 + wgt*loss2

  all_losses = [xent_loss]
  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)
  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss


def minimize(loss, global_step, num_batches):
  # Calculate the learning rate schedule.
  decay_steps = int(num_batches * FLAGS.num_epochs_per_decay)
  # Decay the learning rate exponentially based on the number of steps.
  global lr
  #base_lr = 1e-2 # for sgd
  base_lr = FLAGS.initial_learning_rate
  #stairs = True
  stairs = FLAGS.staircase
  fine_lr_div = FLAGS.fine_lr_div
  #fine_lr_div = 10
  print('fine_lr = base_lr / ', fine_lr_div)
  #lr_fine = tf.train.exponential_decay(base_lr / 10, global_step, decay_steps,
  #lr_fine = tf.train.exponential_decay(base_lr / 20, global_step, decay_steps,

  lr_fine = tf.train.exponential_decay(base_lr / fine_lr_div, global_step, decay_steps,
                                  FLAGS.learning_rate_decay_factor, staircase=stairs)
  lr = tf.train.exponential_decay(base_lr, global_step, decay_steps,
                                  FLAGS.learning_rate_decay_factor, staircase=stairs)

  ## TODO
  #base_lr = 1e-3
  #end_lr = 1e-5
  #decay_steps = num_batches * 20
  #lr_fine = tf.train.polynomial_decay(base_lr / fine_lr_div, global_step,
  #                                    decay_steps, end_lr, power=1)
  #lr = tf.train.polynomial_decay(base_lr, global_step, decay_steps, end_lr, power=1)
  ##lr = tf.Print(lr, [lr], message='lr = ', summarize=10)

  tf.summary.scalar('learning_rate', lr)
  # adam works much better here!
  if imagenet_init:
    opts = [tf.train.AdamOptimizer(lr_fine), tf.train.AdamOptimizer(lr)]
    # TODO
    #eps = 1e-5
    #opts = [tf.train.AdamOptimizer(lr_fine, epsilon=eps),
    #        tf.train.AdamOptimizer(lr, epsilon=eps)]
    return train_helper.minimize_fine_tune(opts, loss, global_step, 'head')
  else:
    opt = tf.train.AdamOptimizer(lr)
    return train_helper.minimize(opt, loss, global_step)
  #opts = [tf.train.RMSPropOptimizer(lr_fine, momentum=0.9, centered=True),
  #        tf.train.RMSPropOptimizer(lr, momentum=0.9, centered=True)]
  #opts = [tf.train.MomentumOptimizer(lr_fine, 0.9), tf.train.MomentumOptimizer(lr, 0.9)]



def train_step(sess, run_ops):
  global train_step_iter
  if apply_jitter:
    feed_dict = _get_train_feed()
    vals = sess.run(run_ops, feed_dict=feed_dict)
  else:
    vals = sess.run(run_ops)
  train_step_iter += 1
  return vals


def num_batches(dataset):
  return dataset.num_examples() // FLAGS.batch_size
  #return 1
