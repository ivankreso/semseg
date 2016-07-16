import tensorflow as tf
from slim import ops
from slim import scopes
import numpy as np
import np_helper
import losses
from models.model_helper import convolve, read_vgg_init

FLAGS = tf.app.flags.FLAGS


def normalize_input(image):
  vgg_mean = tf.constant([123.68, 116.779, 103.939])
  image -= vgg_mean
  return image


def build_refinement_module(top_layer, skip_data):
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


#def build_model(inputs, is_training=True):
def build(inputs, labels, weights, num_labels, is_training=True):
  if is_training:
    vgg_layers, vgg_layer_names = read_vgg_init(FLAGS.vgg_init_dir)
  else:
    vgg_layers = None
    vgg_layer_names = None
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
  # best so far = 0.0005
  with scopes.arg_scope([ops.conv2d, ops.fc], stddev=0.01, weight_decay=0.0005):
  #with scopes.arg_scope([ops.conv2d, ops.fc], stddev=0.01, weight_decay=0.005):
  #with scopes.arg_scope([ops.conv2d, ops.fc], stddev=0.01):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      #net = slim.ops.repeat_op(1, inputs, slim.ops.conv2d, 64, [3, 3], scope='conv1')
      skip_connections = []
      km = 128
      net = convolve(inputs, conv1_sz, k, 'conv1_1', vgg_layers)
      net = convolve(net, conv1_sz, k, 'conv1_2', vgg_layers)
      net = ops.max_pool(net, [2, 2], scope='pool1')
      net = convolve(net, conv2_sz, k, 'conv2_1', vgg_layers)
      net = convolve(net, conv2_sz, k, 'conv2_2', vgg_layers)
      skip_connections += [[net, km/4, 'conv2_2']]
      net = ops.max_pool(net, [2, 2], scope='pool2')
      net = convolve(net, conv3_sz, k, 'conv3_1', vgg_layers)
      net = convolve(net, conv3_sz, k, 'conv3_2', vgg_layers)
      net = convolve(net, conv3_sz, k, 'conv3_3', vgg_layers)
      #skip_connections += [[net, ]]
      skip_connections += [[net, km/2, 'conv3_3']]
      net = ops.max_pool(net, [2, 2], scope='pool3')
      net = convolve(net, conv4_sz, k, 'conv4_1', vgg_layers)
      net = convolve(net, conv4_sz, k, 'conv4_2', vgg_layers)
      net = convolve(net, conv4_sz, k, 'conv4_3', vgg_layers)
      skip_connections += [[net, km, 'conv4_3']]
      net = ops.max_pool(net, [2, 2], scope='pool4')
      net = convolve(net, conv5_sz, k, 'conv5_1', vgg_layers)
      net = convolve(net, conv5_sz, k, 'conv5_2', vgg_layers)
      net = convolve(net, conv5_sz, k, 'conv5_3', vgg_layers)
      skip_connections += [[net, km, 'conv5_3']]

      with scopes.arg_scope([ops.conv2d, ops.fc], batch_norm_params=bn_params):
      #with scopes.arg_scope([ops.conv2d, ops.fc], batch_norm_params=bn_params, weight_decay=0.0005):
        #pad = [[0, 0], [0, 0]]
        ##net = tf.space_to_batch(net, paddings=pad, block_size=4)
        #net = tf.space_to_batch(net, paddings=pad, block_size=2)
        #net = convolve(net, 1024, 7, 'conv6_1')
        ##net = convolve(net, 512, 5, 'conv6_1')
        #net = tf.batch_to_space(net, crops=pad, block_size=2)

        net = convolve(net, 1024, 7, 'conv6_1')
        #net = convolve(net, 1024, 5, 'conv6_1')
        net = convolve(net, 512, 3, 'conv6_2')
        #net = convolve(net, 512, 3, 'conv6_3')
        net = convolve(net, 512, 1, 'ladder_head')
        ladder_head = net

        #net = convolve(net, 1024, 1, 'conv6_1')
        #net = convolve(net, 512, 1, 'conv6_2')
        #net = convolve(net, 512, 1, 'conv6_3')
        #net = convolve(net, 512, 1, 'fc7')

        #net = ops.conv2d(net, 1024, [1, 1], scope='conv5_4')
        #net = ops.conv2d(net, 512, [1, 1], scope='conv6_1')
        #net = ops.conv2d(net, 512, [1, 1], scope='conv6_2')

        for skip_layer in reversed(skip_connections):
          net = build_refinement_module(net, skip_layer)

      out_size = [FLAGS.img_height, FLAGS.img_width]
      #net = convolve(net, 64, 3, 'conv7_1') - worse
      net = convolve(net, FLAGS.num_classes, 1, 'score', activation=None)
      bottom_logits = tf.image.resize_bilinear(net, out_size, name='resize_score')

      top_score = convolve(ladder_head, FLAGS.num_classes, 1, 'top_score', activation=None)
      top_logits = tf.image.resize_bilinear(top_score, out_size, name='resize_top_score')
      loss_val = loss([top_logits, bottom_logits], labels, weights, num_labels, is_training)
  return bottom_logits, loss_val


def loss(logits, labels, weights, num_labels, is_training=True):
  loss_top = losses.weighted_cross_entropy_loss(logits[0], labels, weights, num_labels)
  loss_bottom = losses.weighted_cross_entropy_loss(logits[1], labels, weights, num_labels)
  #loss_val = losses.multiclass_hinge_loss(logits, labels, weights, num_labels)
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, None, num_labels)
  #loss_val = losses.weighted_hinge_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss_symmetric(logits, labels, weights, num_labels)
  all_losses = [loss_top, loss_bottom]
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

