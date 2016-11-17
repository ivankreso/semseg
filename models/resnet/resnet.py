import tensorflow as tf
import argparse
import os, re
import numpy as np

import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

from tensorpack import *
from tensorpack.utils import logger
from tensorpack.utils.stat import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow.dataset import ILSVRCMeta

MODEL_DEPTH = None


def build(image, labels, weights, is_training=True):
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
  
  l = layers.convolution2d(image, 64, 7, stride=2, padding='SAME',
      activation_fn=tf.nn.relu, weights_initializer=init_func,
      #normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      biases_initializer=tf.zeros_initializer,
      weights_regularizer=layers.l2_regularizer(weight_decay), scope='conv0')
  l = layers.batch_norm(l, **bn_params, scope='conv0/BatchNorm')
  l = layers.max_pool2d(l, 2, 2, padding='SAME', scope='pool0')
  l = layer(l, 'group0', 64, defs[0], 1, first=True)
  l = layer(l, 'group1', 128, defs[1], 2)
  l = layer(l, 'group2', 256, defs[2], 2)
  l = layer(l, 'group3', 512, defs[3], 2)
  l = tf.nn.relu(l)
  in_k = l.get_shape().as_list()[-2]
  #print(l.get_shape().as_list())
  l = layers.avg_pool2d(l, kernel_size=in_k, scope='global_avg_pool')
  l = layers.flatten(l, scope='flatten')
  logits = layers.fully_connected(l, 1000, activation_fn=None, scope='fc1000')
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


def init_params(params, data_dir):
  session_init = ParamRestore(params),
  pass
    #ds = dataset.ILSVRC12(data_dir, 'val', shuffle=False, dir_structure='train')
    #ds = AugmentImageComponent(ds, get_inference_augmentor())
    #ds = BatchData(ds, 128, remainder=True)
    #pred_config = PredictConfig(
    #    model=Model(),
    #    session_init=ParamRestore(params),
    #    input_names=['input', 'label'],
    #    output_names=['wrong-top1', 'wrong-top5']
    #)
    #pred = SimpleDatasetPredictor(pred_config, ds)
    #acc1, acc5 = RatioCounter(), RatioCounter()
    #for o in pred.get_result():
    #    batch_size = o[0].shape[0]
    #    acc1.feed(o[0].sum(), batch_size)
    #    acc5.feed(o[1].sum(), batch_size)
    #print("Top1 Error: {}".format(acc1.ratio))
    #print("Top5 Error: {}".format(acc5.ratio))

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
  print(tf_name)
  if tf_name == '/b':
    print(tf_name)
  if tf_name in name_map:
    tf_name = name_map[tf_name]
  print(layer_type)
  #if layer_type != 'bn':
  if layer_type == 'res':
    layer_type = TYPE_DICT[layer_type] + (str(layer_id) if layer_branch == 2 else 'shortcut')
  elif layer_branch == 2:
    layer_type = 'conv' + str(layer_id) + '/' + TYPE_DICT[layer_type]
  elif layer_branch == 1:
    layer_type = 'convshortcut/' + TYPE_DICT[layer_type]
  tf_name = 'group{}/block{}/{}'.format(int(layer_group) - 2, layer_block, layer_type) + tf_name
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
      print(var.name, ' --> init from ', name)
      init_map[var.name] = params[name]
    else:
      print(var.name, ' --> random init')
      raise 1
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
  return init_op, init_feed


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

  param = np.load(MODEL_PATH, encoding='latin1').item()
  resnet_param = {}
  for k, v in param.items():
    try:
      newname = name_conversion(k)
    except:
      logger.error("Exception when processing caffe layer {}".format(k))
      raise
    logger.info("Name Transform: " + k + ' --> ' + newname)
    resnet_param[newname] = v
    #print(v.shape)

  image = tf.placeholder(tf.float32, [None, 224, 224, 3], 'input')
  labels = tf.placeholder(tf.int32, [None], 'label')
  logits = build(image, labels, None)
  #all_vars = tf.contrib.framework.get_variables()
  #for v in all_vars:
  #  print(v.name)
  init_op, init_feed = create_init_op(resnet_param)

  sess = tf.session()


  #eval_on_ILSVRC12(resnet_param, args.eval)
