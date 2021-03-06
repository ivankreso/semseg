import os, re
import pickle
import tensorflow as tf
import numpy as np
import cv2

import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

import libs.cylib as cylib
import train_helper
import losses
import eval_helper
#import datasets.reader_rgb as reader
import datasets.reader as reader
from datasets.cityscapes.cityscapes import CityscapesDataset

FLAGS = tf.app.flags.FLAGS

# RGB
DATA_MEAN =  [75.2051479, 85.01498926, 75.08929598]
DATA_STD =  [46.89434904, 47.63335775, 46.47197535]
# TODO NORMALIZE DEPTH STD
DEPTH_MEAN =  37.79630544
DEPTH_STD = 29.21617326

#DATA_STD = [103.939, 116.779, 123.68]
#DATA_MEAN = [103.939, 116.779, 123.68]

#MEAN_BGR = [75.08929598, 85.01498926, 75.2051479]

model_depth = 121
imagenet_init = True
#imagenet_init = False
init_dir = '/home/kivan/datasets/pretrained/dense_net/'
apply_jitter = True
#apply_jitter = False
pool_func = layers.avg_pool2d
#pool_func = layers.max_pool2d

train_step_iter = 0

weight_decay = 1e-4
#init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
init_func = layers.variance_scaling_initializer()

block_sizes = [6,12,24,16]
#block_sizes = [6,12,24,16,8]
context_size = 512
#imagenet_init = False
#block_sizes = [3,5,6,6,6]
#context_size = 256
growth = 32
compression = 0.5
growth_up = 32
#up_sizes = [4,4,8,8]
growth_up = 64
up_sizes = [4,4,6,6]
#up_sizes = [2,2,4,4]
#up_sizes = [3,3,4,4]


km = 256
#km = 512
# works the same as 256
#km = 128
#km = 512

use_dropout = False
keep_prob = 0.8

fused_batch_norm = True
#data_format = 'NCHW'
#maps_dim = 1
#height_dim = 2
data_format = 'NHWC'
maps_dim = 3
height_dim = 1


bn_params = {
  # Decay for the moving averages.
  'decay': 0.9,
  'center': True,
  'scale': True,
  # epsilon to prevent 0s in variance.
  'epsilon': 1e-5,
  # None to force the updates
  'updates_collections': None,
  # TODO
  'fused': fused_batch_norm,
  'data_format': data_format,
  'is_training': True
}


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
  train_data['iou'].append(iou)
  train_data['acc'].append(pixacc)
  train_loss_val = np.mean(train_loss_arr)
  train_data['loss'].append(train_loss_val)


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


def normalize_input(bgr, depth):
  with tf.name_scope('input'), tf.device('/cpu:0'):
    if data_format == 'NCHW':
      bgr = tf.transpose(bgr, perm=[0,3,1,2])
      depth = tf.transpose(depth, perm=[0,3,1,2])
    blue, green, red = tf.split(bgr, 3, axis=maps_dim)
    #print(blue, green)
    #img = tf.concat([red, green, blue], 3)
    img = tf.concat([red, green, blue], maps_dim)
    if data_format == 'NCHW':
      mean = tf.constant(DATA_MEAN, dtype=tf.float32, shape=[1,3,1,1])
      std = tf.constant(DATA_STD, dtype=tf.float32, shape=[1,3,1,1])
    else:
      mean = DATA_MEAN
      std = DATA_STD
    #return (img - DATA_MEAN) / DATA_STD, depth - 33
    img = (img - mean) / std
    depth = (depth - DEPTH_MEAN) / DEPTH_STD
    return img, depth


def resize_tensor(net, shape, name):
  if data_format == 'NCHW':
    net = tf.transpose(net, perm=[0,2,3,1])
  net = tf.image.resize_bilinear(net, shape, name=name)
  if data_format == 'NCHW':
    net = tf.transpose(net, perm=[0,3,1,2])
  return net

def image_size(net):
  return net.get_shape().as_list()[height_dim:height_dim+2]

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


def BNReluConv(net, num_filters, name, k=3, rate=1, first=False, concat=None):
  with arg_scope([layers.conv2d],
      data_format=data_format, stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.variable_scope(name):
      # TODO check this
      relu = None
      if not first:
        net = tf.contrib.layers.batch_norm(net, **bn_params)
        net = tf.nn.relu(net)
        relu = net
      if concat is not None:
        net = tf.concat([net, concat], maps_dim)
        print('c ', net)
      net = layers.conv2d(net, num_filters, kernel_size=k, rate=rate)
    return net


def layer(net, num_filters, name, is_training, first):
  with tf.variable_scope(name):
    net = BNReluConv(net, 4*num_filters, 'bottleneck', k=1, first=first)
    net = BNReluConv(net, num_filters, 'conv', k=3)
    #if use_dropout and is_training: 
    #  net = tf.nn.dropout(net, keep_prob=keep_prob)
  return net


def dense_block(net, size, growth, name, is_training=False, first=False, split=False):
  with tf.variable_scope(name):
    for i in range(size):
      x = net
      #net, first_relu = layer(net, k, 'layer'+str(i), is_training, first=first)
      net = layer(net, growth, 'layer'+str(i), is_training, first=first)
      net = tf.concat([x, net], maps_dim)
      if first:
        first = False
      if split and i == (size // 2) - 1:
        split_out = net
        print('Split shape = ', net)
        net = pool_func(net, 2, stride=2, padding='SAME', data_format=data_format)
  print('Dense block out: ', net)
  if split == True:
    return net, split_out
  return net


def dense_block_multigpu(net, size, growth, name, is_training=False, first=False, split=False):
  with tf.variable_scope(name):
    for i in range(size):
      if i < size//2:
        gpu = '/gpu:0'
      else:
        gpu = '/gpu:1'
      with tf.device(gpu):
        x = net
        #net, first_relu = layer(net, k, 'layer'+str(i), is_training, first=first)
        net = layer(net, growth, 'layer'+str(i), is_training, first=first)
        net = tf.concat([x, net], maps_dim)
        if first:
          first = False
        if split and i == (size // 2) - 1:
          split_out = net
          print('Split shape = ', net)
          net = pool_func(net, 2, stride=2, padding='SAME', data_format=data_format)
  print('Dense block out: ', net)
  if split == True:
    return net, split_out
  return net


#up_sizes = [256,256,512,512]
#up_sizes = [196,256,384,512]
#up_sizes = [256,256,384,512]
#up_sizes = [128,128,256,256]
#up_sizes = [128,128,256,512] # 2gpus
#up_sizes = [128,256,384,512] # 2gpus
#up_sizes = [128,256,384,512] # 2gpus
#up_sizes = [128,128,256,384,512] # 2gpus
up_sizes = [256,256,512,512]
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
    net = BNReluConv(net, num_filters, 'bottleneck', k=1)
    #net = BNReluConv(net, num_filters, 'bottleneck', k=3)
    #net = tf.concat([net, depth], maps_dim)
    #net = BNReluConv(net, num_filters, 'bottleneck', k=3)
    print('after bottleneck = ', net)
    net = BNReluConv(net, size, 'layer')
  return net
  #return dense_block(net, size, growth, name)


def dense_block_context(net):
  print('Dense context')
  with tf.variable_scope('block_context'):
    outputs = []
    size = 8
    #size = 4
    #size = 6
    for i in range(size):
      x = net
      net = BNReluConv(net, 64, 'layer'+str(i))
      #net = BNReluConv(net, 128, 'layer'+str(i))
      outputs.append(net)
      if i < size - 1:
        net = tf.concat([x, net], maps_dim)
    net = tf.concat(outputs, maps_dim)
  return net

def transition(net, compression, name, stride=2):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    num_filters = net.get_shape().as_list()[maps_dim]
    num_filters = int(round(num_filters*compression))
    net = layers.conv2d(net, num_filters, kernel_size=1)
    skip_layer = net
    # avg works little better on small res
    net = pool_func(net, 2, stride=stride, data_format=data_format, padding='SAME')
  print('Transition: ', net)
  return net, skip_layer


def _build(image, depth, is_training=False):
  #image = tf.Print(image, [tf.shape(image)], message='img_shape = ', summarize=10)
  bn_params['is_training'] = is_training
  with arg_scope([layers.conv2d],
      data_format=data_format, stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.variable_scope('conv0'):
      net = layers.conv2d(image, 2*growth, 7, stride=2)
      #net = layers.conv2d(image, 2*growth, 7, stride=1)
      # TODO
      net = tf.contrib.layers.batch_norm(net, **bn_params)
      net = tf.nn.relu(net)

    #net = layers.max_pool2d(net, 2, stride=2, padding='SAME',
    #                        data_format=data_format, scope='pool0')
    net = layers.max_pool2d(net, 2, stride=2, padding='SAME',
                            data_format=data_format, scope='pool0')

    #depth = resize_tensor(depth, tf.shape(net)[height_dim:height_dim+2],
    #                      name='resize_depth')
    #net = tf.concat([net, depth], maps_dim)
    skip_layers = []

    # no diff with double BN from orig densenet, first=True
    net = dense_block(net, block_sizes[0], growth, 'block0', is_training, first=True)
    #net, skip = dense_block(net, block_sizes[0], growth, 'block0', is_training,
    #    first=True, split=True)
    #skip_layers.append([skip, 256, growth_up, 'block0_mid_refine', depth])
    #skip_layers.append([net, up_sizes[0], growth_up, 'block0_refine', depth])
    net, skip = transition(net, compression, 'block0/transition')
    #skip_layers.append([skip, up_sizes[0], growth_up, 'block0_refine', depth])

    net = dense_block(net, block_sizes[1], growth, 'block1', is_training)
    #skip_layers.append([net, up_sizes[1], growth_up, 'block1_refine', depth])
    #net, skip = dense_block(net, block_sizes[1], k, 'block1', is_training, split=True)
    net, _ = transition(net, compression, 'block1/transition', stride=1)
    #skip_layers.append([skip, up_sizes[1], growth_up, 'block1_refine', depth])

    bsz = 2
    paddings, crops = tf.required_space_to_batch_paddings(image_size(net), [bsz, bsz])
    net = tf.space_to_batch(net, paddings=paddings, block_size=bsz)
    # works the same with split, not 100%
    #net, skip = dense_block(net, block_sizes[2], growth, 'block2', is_training, split=True)
    #skip_layers.append([skip, up_sizes[2], growth_up, 'block2_mid_refine', depth])
    net = dense_block(net, block_sizes[2], growth, 'block2', is_training)
    #skip_layers.append([net, up_sizes[3], growth_up, 'block2_refine', depth])
    net, _ = transition(net, compression, 'block2/transition', stride=1)
    net = tf.batch_to_space(net, crops=crops, block_size=bsz)
    mid_logits = net

    bsz = 4
    paddings, crops = tf.required_space_to_batch_paddings(image_size(net), [bsz, bsz])
    net = tf.space_to_batch(net, paddings=paddings, block_size=bsz)
    #skip_layers.append([skip, up_sizes[3], growth_up, 'block2_refine', depth])
    net = dense_block(net, block_sizes[3], growth, 'block3', is_training)
    #net, skip = dense_block(net, block_sizes[3], k, 'block3', is_training, split=True)
    #skip_layers.append([skip, km, 'block3', depth])

    with tf.variable_scope('head'):
      net = dense_block_context(net)
      #print('5x5')
      #net = BNReluConv(net, context_size, 'context_conv', k=5)
      #print('7x7')
      #net = BNReluConv(net, context_size, 'context_conv', k=7)
      #print('3 x 3x3')
      #net = BNReluConv(net, context_size, 'context_conv1', k=3)
      #net = BNReluConv(net, context_size, 'context_conv2', k=3)
      #net = BNReluConv(net, context_size, 'context_conv3', k=3)
      #net = pyramid_pooling(net)
      net = tf.batch_to_space(net, crops=crops, block_size=bsz)

  print('Final shape: ', net)
  with tf.variable_scope('head'):
    with tf.variable_scope('logits'):
      net = tf.nn.relu(layers.batch_norm(net, **bn_params))
      logits = layers.conv2d(net, FLAGS.num_classes, 1, activation_fn=None,
                             data_format=data_format)

    with tf.variable_scope('mid_logits'):
      # dont forget bn and relu here
      mid_logits = tf.nn.relu(layers.batch_norm(mid_logits, **bn_params))
      mid_logits = layers.conv2d(mid_logits, FLAGS.num_classes, 1, activation_fn=None,
                                 data_format=data_format)

    if data_format == 'NCHW':
      logits = tf.transpose(logits, perm=[0,2,3,1])
      mid_logits = tf.transpose(mid_logits, perm=[0,2,3,1])
    input_shape = tf.shape(image)[height_dim:height_dim+2]
    logits = tf.image.resize_bilinear(logits, input_shape, name='resize_logits')
    mid_logits = tf.image.resize_bilinear(mid_logits, input_shape, name='resize_mid_logits')
    #if data_format == 'NCHW':
    #  top_layer = tf.transpose(top_layer, perm=[0,3,1,2])
    return logits, mid_logits


def _build_2gpu_new(image, depth, is_training=False):
  #image = tf.Print(image, [tf.shape(image)], message='img_shape = ', summarize=10)
  bn_params['is_training'] = is_training
  with arg_scope([layers.conv2d],
      data_format=data_format, stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    gpu1 = '/gpu:0'
    gpu2 = '/gpu:1'
    #gpu3 = '/gpu:2'
    with tf.device('/gpu:0'):
      with tf.variable_scope('conv0'):
        net = layers.conv2d(image, 2*growth, 7, stride=2)
        #net = layers.conv2d(image, 2*growth, 7, stride=1)
        # TODO
        net = tf.contrib.layers.batch_norm(net, **bn_params)
        net = tf.nn.relu(net)

      #net = layers.max_pool2d(net, 2, stride=2, padding='SAME',
      #                        data_format=data_format, scope='pool0')
      net = layers.max_pool2d(net, 2, stride=2, padding='SAME',
                              data_format=data_format, scope='pool0')

      #depth = resize_tensor(depth, tf.shape(net)[height_dim:height_dim+2],
      #                      name='resize_depth')
      #net = tf.concat([net, depth], maps_dim)
      skip_layers = []

      # no diff with double BN from orig densenet, first=True
      net = dense_block(net, block_sizes[0], growth, 'block0', is_training, first=True)
      #net, skip = dense_block(net, block_sizes[0], growth, 'block0', is_training,
      #    first=True, split=True)
      #skip_layers.append([skip, 256, growth_up, 'block0_mid_refine', depth])
      skip_layers.append([net, up_sizes[0], growth_up, 'block0_refine', depth])
      net, skip = transition(net, compression, 'block0/transition')
      #skip_layers.append([skip, up_sizes[0], growth_up, 'block0_refine', depth])

      #net = dense_block(net, block_sizes[1], growth, 'block1', is_training)
      net = dense_block_multigpu(net, block_sizes[1], growth, 'block1', is_training)
      skip_layers.append([net, up_sizes[1], growth_up, 'block1_refine', depth])

    with tf.device(gpu2):
      #net, skip = dense_block(net, block_sizes[1], k, 'block1', is_training, split=True)
      net, skip = transition(net, compression, 'block1/transition')
      #skip_layers.append([skip, up_sizes[1], growth_up, 'block1_refine', depth])

      # works the same with split, not 100%
      net, skip = dense_block(net, block_sizes[2], growth, 'block2', is_training, split=True)
      skip_layers.append([skip, up_sizes[2], growth_up, 'block2_mid_refine', depth])
      #net = dense_block(net, block_sizes[2], growth, 'block2', is_training)
      skip_layers.append([net, up_sizes[3], growth_up, 'block2_refine', depth])

      net, skip = transition(net, compression, 'block2/transition')
      #skip_layers.append([skip, up_sizes[3], growth_up, 'block2_refine', depth])
      net = dense_block(net, block_sizes[3], growth, 'block3', is_training)
      #net, skip = dense_block(net, block_sizes[3], growth, 'block3', is_training, split=True)
      #skip_layers.append([skip, up_sizes[3], growth_up, 'block3_refine', depth])

      with tf.variable_scope('head'):
        net = dense_block_context(net)
        #print('5x5')
        #net = BNReluConv(net, context_size, 'context_conv', k=5)
        #print('7x7')
        #net = BNReluConv(net, context_size, 'context_conv', k=7)
        #print('3 x 3x3')
        #net = BNReluConv(net, context_size, 'context_conv1', k=3)
        #net = BNReluConv(net, context_size, 'context_conv2', k=3)
        #net = BNReluConv(net, context_size, 'context_conv3', k=3)
        #net = pyramid_pooling(net)
        print('Before upsampling: ', net)
        mid_logits = net

        for skip_layer in reversed(skip_layers):
          net = refine(net, skip_layer)
          print('after upsampling = ', net)

  with tf.device(gpu2), tf.variable_scope('head'):
    with tf.variable_scope('logits'):
      net = tf.nn.relu(layers.batch_norm(net, **bn_params))
      logits = layers.conv2d(net, FLAGS.num_classes, 1, activation_fn=None,
                             data_format=data_format)

    with tf.variable_scope('mid_logits'):
      # dont forget bn and relu here
      mid_logits = tf.nn.relu(layers.batch_norm(mid_logits, **bn_params))
      mid_logits = layers.conv2d(mid_logits, FLAGS.num_classes, 1, activation_fn=None,
                                 data_format=data_format)

    if data_format == 'NCHW':
      logits = tf.transpose(logits, perm=[0,2,3,1])
      mid_logits = tf.transpose(mid_logits, perm=[0,2,3,1])
    input_shape = tf.shape(image)[height_dim:height_dim+2]
    logits = tf.image.resize_bilinear(logits, input_shape, name='resize_logits')
    mid_logits = tf.image.resize_bilinear(mid_logits, input_shape, name='resize_mid_logits')
    #if data_format == 'NCHW':
    #  top_layer = tf.transpose(top_layer, perm=[0,3,1,2])
    return logits, mid_logits


def create_init_op(params):
  variables = tf.contrib.framework.get_variables()
  init_map = {}
  # clear head vars from imagenet
  remove_keys = []
  for key in params.keys():
    if 'head/' in key:
      print('delete ', key)
      remove_keys.append(key)
  for key in remove_keys:
    del params[key]

  for var in variables:
    name = var.name
    if name in params:
      #print(name, ' --> found init')
      #print(var)
      #print(params[name].shape)
      init_map[var.name] = params[name]
      del params[name]
    #else:
    #  print(name, ' --> init not found!')
  print('Unused: ', list(params.keys()))
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
  return init_op, init_feed


def jitter(image, labels, weights, depth):
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
    out_weights = []
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
      out_weights.append(tf.cond(random_flip_tf[i], lambda: tf.image.flip_left_right(weights[i]),
                         lambda: weights[i]))
    image = tf.stack(out_img, axis=0)
    depth = tf.stack(out_depth, axis=0)
    weights = tf.stack(out_weights, axis=0)
    labels = tf.stack(out_labels, axis=0)
    #image = tf.Print(image, [random_flip_tf], message='random_flip_tf = ', summarize=10)
    #image = tf.Print(image, [image[0]], message='img = ', summarize=10)

    # TODO
    #image = tf.image.resize_bicubic(image, [resize_height, resize_width])
    #depth = tf.image.resize_bilinear(depth, [resize_height, resize_width])
    #labels = tf.image.resize_nearest_neighbor(labels, [resize_height, resize_width])
    #weights = tf.image.resize_nearest_neighbor(weights, [resize_height, resize_width])
    return image, labels, weights, depth


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
    x, labels, weights, depth, img_names = \
      reader.inputs(dataset, is_training=is_training, num_epochs=FLAGS.max_epochs)
    if is_training and apply_jitter:
      x, labels, weights, depth = jitter(x, labels, weights, depth)
    x, depth = normalize_input(x, depth)

    #logits = _build(x, depth, is_training)
    #total_loss = _loss(logits, labels, weights, is_training)
    logits, mid_logits = _build(x, depth, is_training)
    total_loss = _multiloss(logits, mid_logits, labels, weights, is_training)

    if is_training and imagenet_init:
      init_path = init_dir + 'dense_net_' + str(model_depth) + '.pickle'
      with open(init_path, 'rb') as f:
        init_map = pickle.load(f)
      init_op, init_feed = create_init_op(init_map)
    else:
      init_op, init_feed = None, None
    run_ops = [total_loss, logits, labels, img_names]
    if is_training:
      return run_ops, init_op, init_feed
    else:
      return run_ops

def _multiloss(logits, mid_logits, labels, weights, is_training=True):
  max_weight = 10
  #max_weight = 20
  loss1 = losses.weighted_cross_entropy_loss(logits, labels, weights,
      max_weight=max_weight)
  loss2 = losses.weighted_cross_entropy_loss(mid_logits, labels, weights,
      max_weight=max_weight)
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

def _loss(logits, labels, weights, is_training=True):
  #TODO
  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=1)
  xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=10)
  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=20)
  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=50)
  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=100)
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
  stairs = True
  #stairs = False
  fine_lr_div = 5
  print('fine_lr = base_lr / ', fine_lr_div)
  #lr_fine = tf.train.exponential_decay(base_lr / 10, global_step, decay_steps,
  #lr_fine = tf.train.exponential_decay(base_lr / 20, global_step, decay_steps,
  lr_fine = tf.train.exponential_decay(base_lr / fine_lr_div, global_step, decay_steps,
                                  FLAGS.learning_rate_decay_factor, staircase=stairs)
  lr = tf.train.exponential_decay(base_lr, global_step, decay_steps,
                                  FLAGS.learning_rate_decay_factor, staircase=stairs)
  tf.summary.scalar('learning_rate', lr)
  # adam works much better here!
  if imagenet_init:
    opts = [tf.train.AdamOptimizer(lr_fine), tf.train.AdamOptimizer(lr)]
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
