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

from tensorpack import *
#from tensorpack.utils import logger
#from tensorpack.utils.stat import RatioCounter
#from tensorpack.tfutils.symbolic_functions import *
#from tensorpack.tfutils.summary import *
#from tensorpack.dataflow.dataset import ILSVRCMeta

MODEL_DEPTH = None
#MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]
MEAN_BGR = [103.939, 116.779, 123.68]
DATASET = 'IMAGENET'

SAVE_DIR = '/home/kivan/source/out/imagenet/'

def normalize_input(rgb):
  return rgb - MEAN_BGR
  #"""Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
  #with tf.name_scope('input'), tf.device('/cpu:0'):
  #  #rgb -= MEAN_RGB
  #  red, green, blue = tf.split(3, 3, rgb)
  #  bgr = tf.concat(3, [blue, green, red])
  #  #bgr -= MEAN_BGR
  #  return bgr

def build_orig(image, labels, is_training):
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
  init_func = layers.variance_scaling_initializer(mode='FAN_OUT')

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
      l = layers.convolution2d(l, ch_out * 4, kernel_size=1, activation_fn=None,
                               scope='conv3')
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
  
  image = normalize_input(image)
  #image = tf.pad(image, [[0,0],[3,3],[3,3],[0,0]])
  #l = layers.convolution2d(image, 64, 7, stride=2, padding='VALID',
  l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
      activation_fn=tf.nn.relu, weights_initializer=init_func,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_regularizer=layers.l2_regularizer(weight_decay), scope='conv0')
  l = layers.max_pool2d(l, 3, stride=2, padding='SAME', scope='pool0')
  l = layer(l, 'group0', 64, defs[0], 1, first=True)
  l = layer(l, 'group1', 128, defs[1], 2)
  l = layer(l, 'group2', 256, defs[2], 2)
  l = layer(l, 'group3', 512, defs[3], 2)
  l = tf.nn.relu(l)
  upsample = l
  in_k = l.get_shape().as_list()[-2]
  #print(l.get_shape().as_list())
  #print(l)
  l = layers.avg_pool2d(l, kernel_size=in_k, scope='global_avg_pool')
  l = layers.flatten(l, scope='flatten')
  logits = layers.fully_connected(l, 1000, activation_fn=None, scope='fc1000')


  xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  loss = tf.reduce_mean(xent_loss)
  regularization_losses = tf.contrib.losses.get_regularization_losses()
  total_loss = tf.add_n([loss] + regularization_losses, name='total_loss')

  return total_loss, logits
  

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
    height, width = net.get_shape().as_list()[1:3]
    net = tf.image.resize_bilinear(net, [2*height, 2*width])
    #num_filters = net.get_shape().as_list()[3]
    #net = tf.contrib.layers.convolution2d_transpose(net, num_filters, kernel_size=3, stride=2)
    return net

def build(image, labels, is_training):
  #def BatchNorm(x, use_local_stat=None, decay=0.9, epsilon=1e-5):
  weight_decay = 1e-4
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
  init_func = layers.variance_scaling_initializer(mode='FAN_OUT')

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
      l = layers.convolution2d(l, ch_out * 4, kernel_size=1, activation_fn=None,
                               scope='conv3')
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
  
  image = normalize_input(image)

  block_outputs = []
  with tf.variable_scope(''):
    l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
        activation_fn=tf.nn.relu, weights_initializer=init_func,
        normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
        weights_regularizer=layers.l2_regularizer(weight_decay), scope='conv0')
    l = layers.max_pool2d(l, 3, stride=2, padding='SAME', scope='pool0')
    l = layer(l, 'group0', 64, defs[0], 1, first=True)
    block_outputs += [l]
    l = layer(l, 'group1', 128, defs[1], 2)
    block_outputs += [l]
    l = layer(l, 'group2', 256, defs[2], 2)
    block_outputs += [l]
    l = layer(l, 'group3', 512, defs[3], 2)
    l = tf.nn.relu(l)

  #block_sizes = [3,4,5]
  block_sizes = [3,6,8]
  r = 16
  net = l
  for i, size in reversed(list(enumerate(block_sizes))):
    print(i, size)
    net = upsample(net, 'block'+str(i)+'_back_upsample')
    print(block_outputs[i])
    net = tf.concat(3, [block_outputs[i], net])
    print(net)
    net = dense_block(net, size, r, 'block'+str(i)+'_back', is_training)
    print(net)
    #net = tf.Print(net, [tf.reduce_sum(net)], message=str(i)+' out = ')

  mask = layers.convolution2d(net, 1, 1, biases_initializer=tf.zeros_initializer,
                              activation_fn=None, scope='mask')
  #mask = tf.nn.relu(mask)
  #mask = tf.minimum(tf.nn.relu(mask), 1)
  #mask = tf.sigmoid(mask)
  mask = tf.nn.relu(tf.tanh(mask))
  mask = tf.Print(mask, [tf.reduce_mean(mask)], message='mask sum = ')
  #reg_scale = 1e-5
  #reg_scale = 5e-6
  #reg_scale = 2e-6
  reg_scale = 1e-6
  #reg_scale = 4e-7
  #reg_scale = 1e-3
  #reg_scale = 1e-1
  #reg_scale = 1e-6
  #reg_scale = 1e-4
  # works!
  #reg_scale = 5e-6
  mask_regularizer = layers.l1_regularizer(reg_scale)
  print(mask_regularizer)
  #mask_regularizer = layers.l2_regularizer(reg_scale)
  reg_loss = mask_regularizer(mask)
  reg_loss += 5 * tf.maximum(0.0, 0.3 - tf.reduce_mean(mask))
  #reg_loss = tf.reduce_mean(mask_regularizer(mask))
  #l1_loss = 0

  height, width = image.get_shape().as_list()[1:3]
  #mask = tf.image.resize_bilinear(mask, [height, width])
  mask = tf.image.resize_nearest_neighbor(mask, [height, width])

  image = tf.mul(image, mask)

  with tf.variable_scope('', reuse=True):
    l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
        activation_fn=tf.nn.relu, weights_initializer=init_func,
        normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
        weights_regularizer=layers.l2_regularizer(weight_decay), scope='conv0')
    l = layers.max_pool2d(l, 3, stride=2, padding='SAME', scope='pool0')
    l = layer(l, 'group0', 64, defs[0], 1, first=True)
    l = layer(l, 'group1', 128, defs[1], 2)
    l = layer(l, 'group2', 256, defs[2], 2)
    l = layer(l, 'group3', 512, defs[3], 2)
    l = tf.nn.relu(l)

  in_k = l.get_shape().as_list()[-2]
  l = layers.avg_pool2d(l, kernel_size=in_k, scope='global_avg_pool')
  l = layers.flatten(l, scope='flatten')
  logits = layers.fully_connected(l, 1000, activation_fn=None, scope='fc1000')


  xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  loss = tf.reduce_mean(xent_loss)
  regularization_losses = tf.contrib.losses.get_regularization_losses()
  total_loss = tf.add_n([loss, reg_loss] + regularization_losses, name='total_loss')

  return total_loss, logits, mask

def name_conversion(caffe_layer_name):
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
    return NAME_MAP[caffe_layer_name]

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
    layer_type = TYPE_DICT[layer_type] + (str(layer_id)
        if layer_branch == 2 else 'shortcut')
  elif layer_branch == 2:
    layer_type = 'conv' + str(layer_id) + '/' + TYPE_DICT[layer_type]
  elif layer_branch == 1:
    layer_type = 'convshortcut/' + TYPE_DICT[layer_type]
  tf_name = 'group{}/block{}/{}'.format(int(layer_group) - 2,
      layer_block, layer_type) + tf_name
  return tf_name


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

def draw_image(img, mean, std, step, path):
  img = img.astype(np.float32)
  #img *= std
  #img += mean
  #img = img.reshape(img.shape[:-1]).astype(np.uint8)
  if DATASET == 'MNIST':
    img = img.reshape(img.shape[:-1])
  elif DATASET == 'CIFAR' or DATASET == 'IMAGENET':
    img = img.astype(np.uint8)
  #ski.io.imshow(img)
  #ski.io.show()
  ski.io.imsave(path+'/'+ '%05d'%step + '_img.png', img)

def draw_mask(step, img, save_dir):
  #img = mask / mask.max()
  img = img.reshape(img.shape[:-1])
  ski.io.imsave(save_dir + '/'+ '%05d'%step +'_mask.png', img)

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

if __name__ == '__main__':
  #parser = argparse.ArgumentParser()
  #parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
  #parser.add_argument('--load', required=True,
  #                    help='.npy model file generated by tensorpack.utils.loadcaffe')
  #parser.add_argument('-d', '--depth', help='resnet depth', required=True, type=int, choices=[50, 101, 152])
  #parser.add_argument('--input', help='an input image')
  #parser.add_argument('--eval', help='ILSVRC dir to run validation on')
  #args = parser.parse_args()
  #assert args.input or args.eval, "Choose either input or eval!"

  MODEL_DEPTH = 50
  MODEL_PATH ='/home/kivan/datasets/pretrained/resnet/ResNet'+str(MODEL_DEPTH)+'.npy'
  data_path = '/home/kivan/datasets/imagenet/ILSVRC2015/numpy/val_data.hdf5'

  #data = np.load(data_path)
  #data_x = data[0]
  #data_y = data[1]
  h5f = h5py.File(data_path, 'r')
  data_x = h5f['data_x'][()]
  print(data_x.shape)
  data_y = h5f['data_y'][()]
  h5f.close()

  #from tensorpack.utils.loadcaffe import get_caffe_pb
  #caffepb = get_caffe_pb()
  #obj = caffepb.BlobProto()
  #mean_file = '/home/kivan/datasets/imagenet/ILSVRC2015/caffe/imagenet_mean.binaryproto'
  #with open(mean_file, 'rb') as f:
  #  obj.ParseFromString(f.read())
  #data_mean = np.array(obj.data).reshape((3, 256, 256)).astype('float32')
  #data_mean = np.transpose(data_mean, [1,2,0])
  #if img_size != data_mean.shape[0]:
  #  data_mean = cv2.resize(data_mean, (img_size, img_size))
  #  #data_mean = ski.transform.resize(data_mean, (img_size, img_size),
  #  #                                 preserve_range=True, order=3)

  #data_x = data_x.astype(np.float32)
  #print(data_x.mean((0,1,2), dtype=np.float64))
  #data_x -= data_mean
  #data_mean = np.array([], dtype=np.float32)
  #data_mean = data_x.mean(0)
  #print(data_x.std((0,1,2)))

  param = np.load(MODEL_PATH, encoding='latin1').item()
  resnet_param = {}
  for k, v in param.items():
    try:
      newname = name_conversion(k)
    except:
      logger.error("Exception when processing caffe layer {}".format(k))
      raise
    #logger.info("Name Transform: " + k + ' --> ' + newname)
    resnet_param[newname] = v
    #print(v.shape)

  img_size = 224
  #batch_size = 100
  batch_size = 40
  learning_rate = 1e-4
  num_epochs_per_decay = 8
  lr_decay_factor = 0.5
  max_epochs = 50

  image = tf.placeholder(tf.float32, [None, img_size, img_size, 3], 'input')
  labels = tf.placeholder(tf.int32, [None], 'label')
  #loss, logits = build(image, labels, is_training=False)
  loss, logits, mask = build(image, labels, is_training=True)
  #all_vars = tf.contrib.framework.get_variables()
  #for v in all_vars:
  #  print(v.name)
  init_op, init_feed = create_init_op(resnet_param)

  N = data_x.shape[0]
  assert N % batch_size == 0
  num_batches = N // batch_size
  decay_steps = num_batches * num_epochs_per_decay

  # ADAM, exponential decay
  global_step = tf.Variable(0, trainable=False) 
  #self.adam_rate = tf.train.exponential_decay(
  #  1e-4, self.adam_step, 1, 0.9999)
  lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, lr_decay_factor)

  trainer = tf.train.AdamOptimizer(lr)
  #train_op = trainer.minimize(loss, global_step=global_step) 
  grads_and_vars = trainer.compute_gradients(loss)
  train_op = trainer.apply_gradients(grads_and_vars, global_step=global_step)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(init_op, feed_dict=init_feed)

  top5_error = tf.nn.in_top_k(logits, labels, 5)
  for epoch_num in range(max_epochs):
    data_x, data_y = shuffle_data(data_x, data_y)
    top5_wrong = 0
    cnt_wrong = 0
    for i in range(num_batches):
      offset = i * batch_size
      batch_x = data_x[offset:offset+batch_size, ...]
      batch_y = data_y[offset:offset+batch_size, ...]
      start_time = time.time()
      _, loss_val, logits_val, mask_val, top5 = sess.run([train_op, loss, logits, mask, top5_error],
      #logits_val, top5 = sess.run([logits, top5_error],
          feed_dict={image:batch_x, labels:batch_y})
      duration = time.time() - start_time
      num_examples_per_step = batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)

      if i % 5 == 0:
        draw_image(batch_x[0], MEAN_BGR, 1, i, SAVE_DIR)
        draw_mask(i, mask_val[0], SAVE_DIR)

      top5_wrong += (top5==0).sum()
      yp = logits_val.argmax(1).astype(np.int32)
      cnt_wrong += (yp != batch_y).sum()
      if i % 10 == 0:
        print('[%d] epoch: [%d / %d] loss = %.3f - top1error = %.2f - top5error = %.2f' \
              ' (%.1f examples/sec; %.3f sec/batch)' % (epoch_num, i, num_batches,
              loss_val,
              cnt_wrong / ((i+1)*batch_size) * 100,
              top5_wrong / ((i+1)*batch_size) * 100,
              examples_per_sec, sec_per_batch))
    print(cnt_wrong / N)
    print(top5_wrong / N)
