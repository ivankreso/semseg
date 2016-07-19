import numpy as np
import tensorflow as tf
import slim
from slim import ops
from slim import scopes
import np_helper

FLAGS = tf.app.flags.FLAGS

def convolve(inputs, num_maps, k, name, init_layers=None, activation=tf.nn.relu,
             dilation=None, stride=1):
  if init_layers != None:
    init_map = {'weights':init_layers[name + '/weights'],
                'biases':init_layers[name + '/biases']}
  else:
    init_map = None
  return ops.conv2d(inputs, num_maps, [k, k], scope=name, init=init_map, activation=activation,
                    seed=FLAGS.seed, dilation=dilation, stride=stride)


def _read_conv_params(in_dir, name):
  weights = np_helper.load_nparray(in_dir + name + '_weights.bin', np.float32)
  biases = np_helper.load_nparray(in_dir + name + '_biases.bin', np.float32)
  return weights, biases


def read_vgg_init(in_dir):
  names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
  layers = {}
  for name in names:
    weights, biases = _read_conv_params(in_dir, name)
    layers[name + '/weights'] = weights
    layers[name + '/biases'] = biases

  # transform fc6 parameters to conv6_1 parameters
  weights, biases = _read_conv_params(in_dir, 'fc6')
  weights = weights.reshape((7, 7, 512, 4096))
  layers['conv6_1' + '/weights'] = weights
  layers['conv6_1' + '/biases'] = biases
  names.append('conv6_1')
  return layers, names


def get_multiscale_resolutions(width, height, scale_factors):
  def align_for_pooling(num_pixels):
    sub_factor = 32
    res = num_pixels % sub_factor
    if res >= (sub_factor // 2):
      res = -(sub_factor - res)
    num_pixels = num_pixels - res
    return num_pixels

  sizes = []
  aspect_ratio = width / height
  for _, s in enumerate(scale_factors):
    new_w = round(width * s)
    new_w = align_for_pooling(new_w)
    new_h = round(new_w / aspect_ratio)
    new_h = align_for_pooling(new_h)
    #table.insert(sizes, {new_w, new_h})
    sizes += [[new_w, new_h]]
    print('s={} --> {}x{}'.format(s, new_w, new_h))
  true_scale_factors = []
  first_w = sizes[0][0]
  for _, e in enumerate(sizes):
    true_scale_factors += [e[0] / first_w]
  return sizes, true_scale_factors


#def init_vgg(vgg_layers, layer_names, var_map):
#  for name in layer_names:
#    print('Init: ' + name)
#    var_map[name + '/weights:0'].assign(vgg_layers[name + '/weights'])
#    var_map[name + '/biases:0'].assign(vgg_layers[name + '/biases'])
