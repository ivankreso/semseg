import os, re
import pickle
import tensorflow as tf
import numpy as np
#import cv2
from os.path import join

import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope
#import skimage as ski
#import skimage.io

import libs.cylib as cylib
import models.resnet.resnet_utils as resnet_utils
import train_helper
import losses
import eval_helper
#import datasets.reader_rgb as reader
import datasets.reader as reader
from datasets.cityscapes.cityscapes import CityscapesDataset as Dataset


FLAGS = tf.app.flags.FLAGS
dataset_dir = os.path.join('/home/kivan/datasets/Cityscapes/tensorflow/',
                           '{}x{}'.format(FLAGS.img_width, FLAGS.img_height))
#tf.app.flags.DEFINE_string('dataset_dir', DATASET_DIR, '')
print('Dataset dir: ' + dataset_dir)

# RGB
data_mean =  [75.2051479, 85.01498926, 75.08929598]
data_std =  [46.89434904, 47.63335775, 46.47197535]
# TODO NORMALIZE DEPTH STD
depth_mean = 37.79630544
depth_std = 29.21617326

if FLAGS.no_valid:
  train_dataset = Dataset(dataset_dir, ['train', 'val'])
else:
  train_dataset = Dataset(dataset_dir, ['train'])
  valid_dataset = Dataset(dataset_dir, ['val'])

print('Num training examples = ', train_dataset.num_examples())

#model_depth = 121
#block_sizes = [6,12,24,16]
#model_depth = 169
#block_sizes = [6,12,32,32]

imagenet_init = True
#imagenet_init = False
init_dir = '/home/kivan/datasets/pretrained/dense_net/'
apply_jitter = True
#apply_jitter = False
jitter_scale = False
#jitter_scale = True
pool_func = layers.avg_pool2d
#pool_func = layers.max_pool2d
known_shape = True

train_step_iter = 0

weight_decay = 1e-4
#weight_decay = 4e-5
#weight_decay = 2e-4
#init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
init_func = layers.variance_scaling_initializer()

model_depth = 50

use_dropout = False
keep_prob = 0.8

# must be false if BN is frozen
fused_batch_norm = True
#fused_batch_norm = False

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
  'fused': fused_batch_norm,
  'data_format': data_format,
  'is_training': True
}


def evaluate(name, sess, epoch_num, run_ops, data):
  loss_val, accuracy, iou, recall, precision = eval_helper.evaluate_segmentation_voc2012(
      sess, epoch_num, run_ops, valid_dataset)
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
      train_conf_mat, 'Train', train_dataset.class_info)
  is_best = False
  if len(train_data['iou']) > 0 and iou > max(train_data['iou']):
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


def normalize_input(img):
  with tf.name_scope('input'):
    r, g, b = tf.split(img, 3, axis=maps_dim)
    img = tf.concat([b, g, r], maps_dim)
    return img - data_mean


def resize_tensor(net, shape, name):
  if data_format == 'NCHW':
    net = tf.transpose(net, perm=[0,2,3,1])
  net = tf.image.resize_bilinear(net, shape, name=name)
  if data_format == 'NCHW':
    net = tf.transpose(net, perm=[0,3,1,2])
  return net


def refine(net, skip_data):
  skip_net = skip_data[0]
  num_layers = skip_data[1]
  block_name = skip_data[2]

  #size_top = top_layer.get_shape()[maps_dim].value
  #skip_width = skip_layer.get_shape()[2].value
  #if top_height != skip_height or top_width != skip_width:
    #print(top_height, skip_height)
    #assert(2*top_height == skip_height)
  
  #TODO try convolution2d_transpose
  #up_shape = tf.shape(skip_net)[height_dim:height_dim+2]
  with tf.variable_scope(block_name):
    if known_shape:
      up_shape = skip_net.get_shape().as_list()[height_dim:height_dim+2]
    else:
      up_shape = tf.shape(skip_net)[height_dim:height_dim+2]
    shape_info = net.get_shape().as_list()
    print(net)
    net = resize_tensor(net, up_shape, name='upsample')
    print(net)
    if not known_shape:
      print(shape_info)
      shape_info[height_dim] = None
      shape_info[height_dim+1] = None
      net.set_shape(shape_info)
    print('\nup = ', net)
    print('skip = ', skip_net)
    #print(skip_data)
    return refine_layer(net, skip_net, num_layers, 'dense_block')

up_sizes = [128,128,128,128,128]
def refine_layer(net, skip_net, size, name):
  with tf.variable_scope(name):
    # TODO
    num_filters = net.get_shape().as_list()[maps_dim]
    #skip_net = BNReluConv(skip_net, num_filters, 'bottleneck', k=1)
    net = layers.conv2d(net, num_filters, kernel_size=1, scope='bottleneck')
    net = tf.concat([net, skip_net], maps_dim)
    #net = net + skip_net
    #net = BNReluConv(net, num_filters, 'bottleneck', k=3)
    print('after concat = ', net)
    net = layers.conv2d(net, size, kernel_size=3, scope='layer')
    #net = BNReluConv(net, size, 'layer')
  return net


def _build(image, is_training=False):
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
  skip_layers.append([tf.nn.relu(l), up_sizes[0], 'block0_refine'])
  l = layer(l, 'group1', 128, defs[1], 2)

  #bsz = 2
  #paddings, crops = tf.required_space_to_batch_paddings(tf.shape(l)[1:3], [bsz, bsz])
  #l = tf.space_to_batch(l, paddings=paddings, block_size=bsz)
  #l = layer(l, 'group2', 256, defs[2], 1)
  #l = tf.batch_to_space(l, crops=crops, block_size=bsz)
  skip_layers.append([tf.nn.relu(l), up_sizes[1], 'block1_refine'])
  l = layer(l, 'group2', 256, defs[2], 2)

  #bsz = 4
  #paddings, crops = tf.required_space_to_batch_paddings(tf.shape(l)[1:3], [bsz, bsz])
  #l = tf.space_to_batch(l, paddings=paddings, block_size=bsz)
  #l = layer(l, 'group3', 512, defs[3], 1)
  #net = tf.batch_to_space(l, crops=crops, block_size=bsz)
  skip_layers.append([tf.nn.relu(l), up_sizes[2], 'block2_refine'])
  net = layer(l, 'group3', 512, defs[3], 2)
  #skip_layers.append([tf.nn.relu(l), km, 'group3', depth])
  print('resnet:', net)

  with tf.variable_scope('head'):
    with arg_scope([layers.conv2d],
        stride=1, padding='SAME', activation_fn=tf.nn.relu,
        normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
        weights_initializer=init_func,
        weights_regularizer=layers.l2_regularizer(weight_decay)):

      net = tf.nn.relu(net)
      #bsz = 2
      #paddings, crops = tf.required_space_to_batch_paddings(image_size(net), [bsz, bsz])
      #net = tf.space_to_batch(net, paddings=paddings, block_size=bsz)
      #net = layer(net, 'group4', 512, defs[3], 1)
      #net = tf.nn.relu(net)
      #l = layers.conv2d(l, 1024, kernel_size=1, scope='conv1')
      net = layers.conv2d(net, 512, kernel_size=1, scope='bottleneck')
      net = layers.conv2d(net, 256, kernel_size=3, rate=2, scope='context')
      #net = tf.batch_to_space(net, crops=crops, block_size=bsz)
      #l = layers.conv2d(l, context_size, kernel_size=3, rate=4, scope='conv2')
      #l = _pyramid_pooling(l, size=4)
      #l = _pyramid_pooling(l, size=2)

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

    #logits_mid = None
    logits_mid = net
    print('Before upsampling: ', net)
    for skip_layer in reversed(skip_layers):
      net = refine(net, skip_layer)

    logits = layers.conv2d(net, FLAGS.num_classes, 1, weights_initializer=init_func,
                           activation_fn=None, scope='logits')
    input_shape = tf.shape(image)[1:3]

    if logits_mid is not None:
      logits_mid = layers.conv2d(logits_mid, FLAGS.num_classes, 1, weights_initializer=init_func,
                                 activation_fn=None, scope='logits_mid')
      logits_mid = tf.image.resize_bilinear(logits_mid, input_shape, name='resize_logits_mid')
      aux_logits = [logits_mid]
    else:
      aux_logits = []
    logits = tf.image.resize_bilinear(logits, input_shape, name='resize_logits')
    return logits, aux_logits


def _pyramid_pooling(net, size=3):
  print('Pyramid context pooling')
  with tf.variable_scope('pyramid_context_pooling'):
    if known_shape:
      shape = net.get_shape().as_list()
    else:
      shape = tf.shape(net)
    print('shape = ', shape)
    up_size = shape[height_dim:height_dim+2]
    shape_info = net.get_shape().as_list()
    num_maps = net.get_shape().as_list()[maps_dim]
    #grid_size = [6, 3, 2, 1]
    pool_dim = int(round(num_maps / size))
    concat_lst = [net]
    for i in range(size):
      #pool = layers.avg_pool2d(net, kernel_size=[kh, kw], stride=[kh, kw], padding='SAME')
      #pool = layers.avg_pool2d(net, kernel_size=[kh, kh], stride=[kh, kh], padding='SAME')
      print('before pool = ', net)
      net = layers.avg_pool2d(net, 2, 2, padding='SAME', data_format=data_format)
      print(net)
      pool = BNReluConv(net, pool_dim, k=1, name='bottleneck'+str(i))
      #pool = tf.image.resize_bilinear(pool, [height, width], name='resize_score')
      pool = resize_tensor(pool, up_size, name='upsample_level_'+str(i))
      concat_lst.append(pool)
    net = tf.concat(concat_lst, maps_dim)
    print('Pyramid pooling out: ', net)
    #net = BNReluConv(net, 512, k=3, name='bottleneck_out')
    net = BNReluConv(net, 256, k=3, name='bottleneck_out')
    return net


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


#def jitter(image, labels, weights):
def jitter(image, labels):
  with tf.name_scope('jitter'), tf.device('/cpu:0'):
    print('\nJittering enabled')
    global random_flip_tf, resize_width, resize_height
    #random_flip_tf = tf.placeholder(tf.bool, shape=(), name='random_flip')
    random_flip_tf = tf.placeholder(tf.bool, shape=(FLAGS.batch_size), name='random_flip')
    resize_width = tf.placeholder(tf.int32, shape=(), name='resize_width')
    resize_height = tf.placeholder(tf.int32, shape=(), name='resize_height')
    
    #image_split = tf.unstack(image, axis=0)
    #labels_split = tf.unstack(labels, axis=0)
    #weights_split = tf.unstack(weights, axis=0)
    out_img = []
    #out_weights = []
    out_labels = []
    for i in range(FLAGS.batch_size):
      out_img.append(tf.cond(random_flip_tf[i],
        lambda: tf.image.flip_left_right(image[i]),
        lambda: image[i]))
      out_labels.append(tf.cond(random_flip_tf[i],
        lambda: tf.image.flip_left_right(labels[i]),
        lambda: labels[i]))
      #out_weights.append(tf.cond(random_flip_tf[i],
      #  lambda: tf.image.flip_left_right(weights[i]),
      #  lambda: weights[i]))
    image = tf.stack(out_img, axis=0)
    labels = tf.stack(out_labels, axis=0)
    #weights = tf.stack(out_weights, axis=0)

    if jitter_scale:
      global known_shape
      known_shape = False
      image = tf.image.resize_bicubic(image, [resize_height, resize_width])
      #image = tf.image.resize_bilinear(image, [resize_height, resize_width])
      image = tf.round(image)
      image = tf.minimum(255.0, image)
      image = tf.maximum(0.0, image)
      labels = tf.image.resize_nearest_neighbor(labels, [resize_height, resize_width])
      # TODO is this safe for zero wgts?
      #weights = tf.image.resize_nearest_neighbor(weights, [resize_height, resize_width])
    #return image, labels, weights
    return image, labels


def _get_train_feed():
  global random_flip_tf, resize_width, resize_height
  #random_flip = int(np.random.choice(2, 1))
  random_flip = np.random.choice(2, FLAGS.batch_size).astype(np.bool)
  #resize_scale = np.random.uniform(0.5, 2)
  #resize_scale = np.random.uniform(0.4, 1.5)
  #resize_scale = np.random.uniform(0.5, 1.2)
  #min_resize = 0.7
  #max_resize = 1.3
  min_resize = 0.8
  max_resize = 1.2
  #min_resize = 0.9
  #max_resize = 1.1
  #max_resize = 1
  if train_step_iter == 0:
    resize_scale = max_resize
  else:
    resize_scale = np.random.uniform(min_resize, max_resize)
  width = np.int32(int(round(FLAGS.img_width * resize_scale)))
  height = np.int32(int(round(FLAGS.img_height * resize_scale)))
  feed_dict = {random_flip_tf:random_flip, resize_width:width, resize_height:height}
  return feed_dict


def build(mode):
  if mode == 'train':
    is_training = True
    reuse = False
    dataset = train_dataset
  elif mode == 'validation':
    is_training = False
    reuse = True
    dataset = valid_dataset

  with tf.variable_scope('', reuse=reuse):
    x, labels, num_labels, class_hist, depth, img_names = \
      reader.inputs(dataset, is_training=is_training)
      #reader.inputs(dataset, is_training=is_training, num_epochs=FLAGS.max_epochs)

    if is_training and apply_jitter:
      x, labels = jitter(x, labels)
    image = x
    x = normalize_input(x)

    #logits = _build(x, depth, is_training)
    #total_loss = _loss(logits, labels, weights, is_training)
    #logits, mid_logits = _build(x, is_training)
    logits, aux_logits = _build(x, is_training)
    total_loss = _multiloss(logits, aux_logits, labels, class_hist, num_labels, is_training)

    if is_training and imagenet_init:
      model_path ='/home/kivan/datasets/pretrained/resnet/ResNet'+str(model_depth)+'.npy'
      param = np.load(model_path, encoding='latin1').item()
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

    train_run_ops = [total_loss, logits, labels, img_names]
    #train_run_ops = [total_loss, logits, labels, img_names, image]
    val_run_ops = [total_loss, logits, labels, img_names]
    if is_training:
      return train_run_ops, init_op, init_feed
    else:
      return val_run_ops


def inference(image, labels=None, constant_shape=True, is_training=False):
  global known_shape
  known_shape = constant_shape
  x = normalize_input(image)
  logits, aux_logits = _build(x, is_training=is_training)
  if labels:
    main_wgt = 0.7
    xent_loss = main_wgt * losses.weighted_cross_entropy_loss(logits, labels)
    xent_loss = (1-main_wgt) * losses.weighted_cross_entropy_loss(aux_logits, labels)
    return logits, aux_logits, xent_loss
  return logits, aux_logits


def _multiloss(logits, aux_logits, labels, num_labels, class_hist, is_training):
  max_weight = FLAGS.max_weight
  xent_loss = 0
  #main_wgt = 0.6
  if len(aux_logits) > 0:
    main_wgt = 0.7
    aux_wgt = (1 - main_wgt) / len(aux_logits)
  else:
    main_wgt = 1.0
    aux_wgt = 0
  xent_loss = main_wgt * losses.weighted_cross_entropy_loss(
      logits, labels, class_hist, max_weight=max_weight)
  for i, l in enumerate(aux_logits):
    print('loss' + str(i), ' --> ' , l)
    xent_loss += aux_wgt * losses.weighted_cross_entropy_loss(
      l, labels, class_hist, max_weight=max_weight)

  all_losses = [xent_loss]
  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)
  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)
  return total_loss


def _dualloss(logits, mid_logits, labels, class_hist, num_labels, is_training=True):
  #loss1 = losses.cross_entropy_loss(logits, labels, weights, num_labels)
  #loss2 = losses.cross_entropy_loss(mid_logits, labels, weights, num_labels)
  #max_weight = 10
  max_weight = 1
  loss1 = losses.weighted_cross_entropy_loss(logits, labels, class_hist,
                                             max_weight=max_weight)
  loss2 = losses.weighted_cross_entropy_loss(mid_logits, labels, class_hist,
                                             max_weight=max_weight)
  #loss1 = losses.weighted_cross_entropy_loss_dense(logits, labels, weights, num_labels,
  #    max_weight=max_weight)
  #loss2 = losses.weighted_cross_entropy_loss_dense(mid_logits, labels, weights, num_labels,
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
  #decay_steps = int(num_batches * FLAGS.num_epochs_per_decay)
  # Decay the learning rate exponentially based on the number of steps.
  global lr
  #base_lr = 1e-2 # for sgd
  base_lr = FLAGS.initial_learning_rate
  #TODO
  #fine_lr_div = 5
  fine_lr_div = 10
  #fine_lr_div = 7
  print('LR = ', base_lr)
  print('fine_lr = LR / ', fine_lr_div)
  #lr_fine = tf.train.exponential_decay(base_lr / 10, global_step, decay_steps,
  #lr_fine = tf.train.exponential_decay(base_lr / 20, global_step, decay_steps,

  #decay_steps = int(num_batches * 30)
  #decay_steps = num_batches * FLAGS.max_epochs
  decay_steps = FLAGS.num_iters
  lr_fine = tf.train.polynomial_decay(base_lr / fine_lr_div, global_step, decay_steps,
                                      end_learning_rate=0, power=FLAGS.decay_power)
  lr = tf.train.polynomial_decay(base_lr, global_step, decay_steps,
                                 end_learning_rate=0, power=FLAGS.decay_power)
  #lr = tf.Print(lr, [lr], message='lr = ', summarize=10)

  #stairs = True
  #lr_fine = tf.train.exponential_decay(base_lr / fine_lr_div, global_step, decay_steps,
  #                                FLAGS.learning_rate_decay_factor, staircase=stairs)
  #lr = tf.train.exponential_decay(base_lr, global_step, decay_steps,
  #                                FLAGS.learning_rate_decay_factor, staircase=stairs)
  tf.summary.scalar('learning_rate', lr)
  # adam works much better here!
  if imagenet_init:
    if FLAGS.optimizer == 'adam':
      print('\nOptimizer = ADAM\n')
      opts = [tf.train.AdamOptimizer(lr_fine), tf.train.AdamOptimizer(lr)]
    elif FLAGS.optimizer == 'momentum':
      print('\nOptimizer = SGD + momentum\n')
      opts = [tf.train.MomentumOptimizer(lr_fine, 0.9), tf.train.MomentumOptimizer(lr, 0.9)]
    else:
      raise ValueError('unknown optimizer')
    return train_helper.minimize_fine_tune(opts, loss, global_step, 'head')
  else:
    opt = tf.train.AdamOptimizer(lr)
    #opt = tf.train.MomentumOptimizer(lr, 0.9)
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
  #img = vals[-3]
  #print(img.shape)
  ##print(img.mean())
  #for i in range(img.shape[0]):
  #  rgb = img[i]
  #  print(rgb.min())
  #  print(rgb.max())
  #  ski.io.imsave(join('/home/kivan/datasets/results/tmp/debug', str(i)+'.png'),
  #                rgb.astype(np.uint8))
  return vals


def num_batches():
  return train_dataset.num_examples() // FLAGS.batch_size


def image_size(net):
  return net.get_shape().as_list()[height_dim:height_dim+2]

def _build_dilated(image, is_training=False):
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

    net = layers.max_pool2d(net, 2, stride=2, padding='SAME',
                            data_format=data_format, scope='pool0')

    skip_layers = []

    # no diff with double BN from orig densenet, first=True
    net = dense_block(net, block_sizes[0], growth, 'block0', is_training, first=True)
    #net, skip = dense_block(net, block_sizes[0], growth, 'block0', is_training,
    #    first=True, split=True)
    #skip_layers.append([skip, 256, growth_up, 'block0_mid_refine', depth])
    #skip_layers.append([skip, up_sizes[0], growth_up, 'block0_mid_refine'])
    skip_layers.append([net, up_sizes[0], growth_up, 'block0_refine'])
    net, _ = transition(net, compression, 'block0/transition')
    #skip_layers.append([skip, up_sizes[0], growth_up, 'block0_refine'])

    #net, skip = dense_block(net, block_sizes[1], growth, 'block1', is_training, split=True)
    #skip_layers.append([skip, up_sizes[1], growth_up, 'block1_mid_refine'])
    net = dense_block(net, block_sizes[1], growth, 'block1', is_training)
    skip_layers.append([net, up_sizes[1], growth_up, 'block1_refine'])
    net, _ = transition(net, compression, 'block1/transition')
    #skip_layers.append([skip, up_sizes[1], growth_up, 'block1_refine'])

    # works the same with split, not 100%
    #context_pool_num = 3
    #context_pool_num = 4
    context_pool_num = 5
    #net, skip = dense_block(net, block_sizes[2], growth, 'block2', is_training, split=True)
    #skip_layers.append([skip, up_sizes[2], growth_up, 'block2_mid_refine'])
    net = dense_block(net, block_sizes[2], growth, 'block2', is_training)
    #skip_layers.append([net, up_sizes[3], growth_up, 'block2_refine'])
    #skip_layers.append([net, up_sizes[2], growth_up, 'block2_refine'])
    net, _ = transition(net, compression, 'block2/transition', stride=1)

    bsz = 2
    paddings, crops = tf.required_space_to_batch_paddings(image_size(net), [bsz, bsz])
    net = tf.space_to_batch(net, paddings=paddings, block_size=bsz)
    net = dense_block(net, block_sizes[3], growth, 'block3', is_training)
    net = tf.batch_to_space(net, crops=crops, block_size=bsz)
    print('before context = ', net)

    with tf.variable_scope('head'):
      net = BNReluConv(net, 512, 'bottleneck', k=1)
      net = _pyramid_pooling(net, size=context_pool_num)
      #net = BNReluConv(net, context_size, 'context_conv', k=3)

      print('Before upsampling: ', net)

      all_logits = [net]
      for skip_layer in reversed(skip_layers):
        net = refine(net, skip_layer)
        all_logits.append(net)
        print('after upsampling = ', net)

      all_logits = [all_logits[0], all_logits[-1]]
      #all_logits = [all_logits[1], all_logits[-1]]
      #all_logits = [all_logits[2], all_logits[-1]]

  with tf.variable_scope('head'):
    for i, logits in enumerate(all_logits):
      with tf.variable_scope('logits_'+str(i)):
      # FIX
      #net = tf.nn.relu(layers.batch_norm(net, **bn_params))
      #logits = layers.conv2d(net, FLAGS.num_classes, 1, activation_fn=None,
      #                       data_format=data_format)
        logits = layers.conv2d(tf.nn.relu(logits), FLAGS.num_classes, 1,
                               activation_fn=None, data_format=data_format)

        if data_format == 'NCHW':
          logits = tf.transpose(logits, perm=[0,2,3,1])
        input_shape = tf.shape(image)[height_dim:height_dim+2]
        logits = tf.image.resize_bilinear(logits, input_shape, name='resize_logits')
        all_logits[i] = logits
    logits = all_logits.pop()
    return logits, all_logits

#def _loss(logits, labels, weights, is_training=True):
#  #TODO
#  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=1)
#  xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=10)
#  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=20)
#  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=50)
#  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=100)
#  all_losses = [xent_loss]
#
#  # get losses + regularization
#  total_loss = losses.total_loss_sum(all_losses)
#
#  if is_training:
#    loss_averages_op = losses.add_loss_summaries(total_loss)
#    with tf.control_dependencies([loss_averages_op]):
#      total_loss = tf.identity(total_loss)
#
#  return total_loss

