import tensorflow as tf
from inception.slim import slim
from inception.slim import ops
from inception.slim import scopes
import numpy as np
import np_helper
import losses

FLAGS = tf.app.flags.FLAGS

def Convolve(inputs, num_maps, k, name, init_layers):
  init_map = None
  if init_layers != None:
    init_map = {'weights':init_layers[name + '/weights'],
                'biases':init_layers[name + '/biases']}
  return slim.ops.conv2d(inputs, num_maps, [k, k], scope=name, init=init_map)


def read_conv_params(in_dir, name):
  weights = np_helper.load_nparray(in_dir + name + '_weights.bin', np.float32)
  biases = np_helper.load_nparray(in_dir + name + '_biases.bin', np.float32)
  return weights, biases


def read_vgg_init(in_dir):
  names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
  layers = {}
  for name in names:
    weights, biases = read_conv_params(in_dir, name)
    layers[name + '/weights'] = weights
    layers[name + '/biases'] = biases
  return layers, names

#def init_vgg(vgg_layers, layer_names, var_map):
#  for name in layer_names:
#    print('Init: ' + name)
#    var_map[name + '/weights:0'].assign(vgg_layers[name + '/weights'])
#    var_map[name + '/biases:0'].assign(vgg_layers[name + '/biases'])

def inference(inputs, is_training=True):
  vgg_layers, vgg_layer_names = read_vgg_init('/home/kivan/datasets/pretrained/vgg16/')
  conv1_sz = 64
  conv2_sz = 128
  conv3_sz = 256
  conv4_sz = 512
  conv5_sz = 512
  k = 3
  bn_params = {
      # Decay for the moving averages.
      'decay': 0.999,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      'center': False,
      'scale': False,
  }
  #with scopes.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.0005):
  #with scopes.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.005):
  with scopes.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      #net = slim.ops.repeat_op(1, inputs, slim.ops.conv2d, 64, [3, 3], scope='conv1')
      net = Convolve(inputs, conv1_sz, k, 'conv1_1', vgg_layers)
      net = Convolve(net, conv1_sz, k, 'conv1_2', vgg_layers)
      net = ops.max_pool(net, [2, 2], scope='pool1')
      net = Convolve(net, conv2_sz, k, 'conv2_1', vgg_layers)
      net = Convolve(net, conv2_sz, k, 'conv2_2', vgg_layers)
      net = ops.max_pool(net, [2, 2], scope='pool2')
      net = Convolve(net, conv3_sz, k, 'conv3_1', vgg_layers)
      net = Convolve(net, conv3_sz, k, 'conv3_2', vgg_layers)
      conv3_3 = Convolve(net, conv3_sz, k, 'conv3_3', vgg_layers)
      net = ops.max_pool(conv3_3, [2, 2], scope='pool3')
      net = Convolve(net, conv4_sz, k, 'conv4_1', vgg_layers)
      net = Convolve(net, conv4_sz, k, 'conv4_2', vgg_layers)
      net = Convolve(net, conv4_sz, k, 'conv4_3', vgg_layers)
      net = ops.max_pool(net, [2, 2], scope='pool4')
      net = Convolve(net, conv5_sz, k, 'conv5_1', vgg_layers)
      net = Convolve(net, conv5_sz, k, 'conv5_2', vgg_layers)
      conv5_3 = Convolve(net, conv5_sz, k, 'conv5_3', vgg_layers)
      #pool5 = ops.max_pool(conv5_3, [2, 2], scope='pool5')

      #conv3_shape = conv3_3.get_shape()
      #resize_shape = [conv3_shape[1].value, conv3_shape[2].value]
      #up_conv5_3 = tf.image.resize_nearest_neighbor(conv5_3, resize_shape)
      ##up_conv5_3 = tf.image.resize_nearest_neighbor(conv5_3, [108, 256])
      #concat = tf.concat(3, [conv3_3, up_conv5_3])
      concat = pool5
      #net = slim.ops.max_pool(net, [2, 2], scope='pool5')
      with scopes.arg_scope([ops.conv2d, ops.fc], batch_norm_params=bn_params):
        net = ops.conv2d(concat, 1024, [1, 1], scope='fc6')
        net = ops.conv2d(net, 1024, [1, 1], scope='fc7')
      #net = ops.conv2d(net, 1024, [1, 1], scope='fc6')
      #net = ops.conv2d(net, 1024, [1, 1], scope='fc7')
      net = ops.conv2d(net, FLAGS.num_classes, [1, 1], activation=None, scope='score')
      #net = slim.ops.flatten(net, scope='flatten5')
      #net = slim.ops.fc(net, 4096, scope='fc6')
      #net = slim.ops.dropout(net, 0.5, scope='dropout6')
      #net = slim.ops.fc(net, 4096, scope='fc7')
      #net = slim.ops.dropout(net, 0.5, scope='dropout7')
      #net = slim.ops.fc(net, 1000, activation=None, scope='fc8')
      logits_up = tf.image.resize_bilinear(net, [FLAGS.img_height, FLAGS.img_width],
                                           name='resize_scores')

  #var_list = tf.all_variables()
  #var_list = slim.variables.get_variables()
  #var_map = {}
  #for v in var_list:
  #  print(v.name)
  #  var_map[v.name] = v
  #var_map['conv1_1/weights:0'].assign(vgg_layers['conv1_1'][0])
  #init_vgg(vgg_layers, vgg_layer_names, var_map)
  #init_op = tf.initialize_variables([var_map['fc6/weights:0'], var_map['fc6/biases:0'],
  #  var_map['fc7/weights:0'], var_map['fc7/biases:0'], var_map['score/weights:0'],
  #  var_map['score/biases:0'], var_map['global_step:0']])

  #print(var_list[1].name)
  #var_list[1].assign(vgg_layers['conv1_1'])
  return logits_up


def loss(logits, labels, weights, num_labels, is_training=True):
  loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, num_labels)
  all_losses = [loss_val]
  #all_losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)

  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)

  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss
  
  #return losses.weighted_hinge_loss(logits, labels, weights, num_labels)
  #return losses.cross_entropy_loss(logits, labels, num_labels)

