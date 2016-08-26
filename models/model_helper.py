import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
#import slim
#from slim import ops
#from slim import scopes
import np_helper

FLAGS = tf.app.flags.FLAGS

def convolve(inputs, num_outputs, k, name, init_layers):
  if init_layers is not None:
    weight_init = init_layers[name + '/weights']
    bias_init = init_layers[name + '/biases']
  else:
    raise ValueError("No init!")
    weight_init = bias_init = None
  net = layers.convolution2d(inputs, num_outputs, k, weights_initializer=weight_init,
                             biases_initializer=bias_init, scope=name)
  return net


#def convolve_slim(inputs, num_maps, k, name, init_layers=None, activation=tf.nn.relu,
#             dilation=None, stride=1):
#  if init_layers != None:
#    init_map = {'weights':init_layers[name + '/weights'],
#                'biases':init_layers[name + '/biases']}
#  else:
#    init_map = None
#  return ops.conv2d(inputs, num_maps, [k, k], scope=name, init=init_map, activation=activation,
#                    seed=FLAGS.seed, dilation=dilation, stride=stride)


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
  skip_layer = convolve(skip_layer, size_top, 3, skip_name + '_refine_prep')
  net = tf.concat(3, [top_layer, skip_layer])
  #print(net)
  net = convolve(net, size_bottom, 3, skip_name + '_refine_fuse')
  return net

def build_refinement_module_old(top_layer, skip_data):
  skip_layer = skip_data[0]
  size_bottom = skip_data[1]
  skip_name = skip_data[2]
  top_height = top_layer.get_shape()[1].value
  top_width = top_layer.get_shape()[2].value
  size_top = top_layer.get_shape()[3].value
  #print('size_top = ', top_height, top_width, size_top)
  skip_layer = convolve(skip_layer, size_top, 3, skip_name + '_refine_prep')
  net = tf.concat(3, [top_layer, skip_layer])
  #print(net)
  net = convolve(net, size_bottom, 3, skip_name + '_refine_fuse')
  net = tf.image.resize_bilinear(net, [2*top_height, 2*top_width],
                                 name=skip_name + '_refine_upsample')
  return net


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
