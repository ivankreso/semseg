import tensorflow as tf
from slim import ops
from slim import scopes
import numpy as np
import np_helper
import losses
from models.model_helper import convolve, read_vgg_init

FLAGS = tf.app.flags.FLAGS


def inference(inputs, is_training=True):
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
  #with scopes.arg_scope([ops.conv2d, ops.fc], stddev=0.01, weight_decay=0.05):
  #with scopes.arg_scope([ops.conv2d, ops.fc], stddev=0.01):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      #net = slim.ops.repeat_op(1, inputs, slim.ops.conv2d, 64, [3, 3], scope='conv1')
      net = convolve(inputs, conv1_sz, k, 'conv1_1', vgg_layers)
      net = convolve(net, conv1_sz, k, 'conv1_2', vgg_layers)
      net = ops.max_pool(net, [2, 2], scope='pool1')
      net = convolve(net, conv2_sz, k, 'conv2_1', vgg_layers)
      net = convolve(net, conv2_sz, k, 'conv2_2', vgg_layers)
      net = ops.max_pool(net, [2, 2], scope='pool2')
      net = convolve(net, conv3_sz, k, 'conv3_1', vgg_layers)
      net = convolve(net, conv3_sz, k, 'conv3_2', vgg_layers)
      conv3_3 = convolve(net, conv3_sz, k, 'conv3_3', vgg_layers)
      pool3 = ops.max_pool(conv3_3, [2, 2], scope='pool3')
      net = convolve(pool3, conv4_sz, k, 'conv4_1', vgg_layers)
      net = convolve(net, conv4_sz, k, 'conv4_2', vgg_layers)
      net = convolve(net, conv4_sz, k, 'conv4_3', vgg_layers)

      zero_pad = [[0, 0], [0, 0]]


      #net = ops.max_pool(net, [2, 2], scope='pool4')
      #pad = ...  # padding so that the input dims are multiples of rate
      #net = space_to_batch(net, paddings=pad, block_size=rate)
      #net = conv2d(net, filters1, strides=[1, 1, 1, 1], padding="SAME")
      #net = conv2d(net, filters2, strides=[1, 1, 1, 1], padding="SAME")
      #...
      #net = conv2d(net, filtersK, strides=[1, 1, 1, 1], padding="SAME")
      #net = batch_to_space(net, crops=pad, block_size=rate)

      net = tf.space_to_batch(net, paddings=zero_pad, block_size=2)
      #net = ops.conv2d(net, conv5_sz, [k, k], scope='conv5_1')
      net = convolve(net, conv5_sz, k, 'conv5_1', vgg_layers)
      net = convolve(net, conv5_sz, k, 'conv5_2', vgg_layers)
      net = convolve(net, conv5_sz, k, 'conv5_3', vgg_layers)
      net = tf.batch_to_space(net, crops=zero_pad, block_size=2)

      curr_scope = tf.get_variable_scope()
      with tf.variable_scope(curr_scope, reuse=True):
        net2 = tf.space_to_batch(pool3, paddings=zero_pad, block_size=2)
        net2 = convolve(net2, conv4_sz, k, 'conv4_1', vgg_layers)
        net2 = convolve(net2, conv4_sz, k, 'conv4_2', vgg_layers)
        net2 = convolve(net2, conv4_sz, k, 'conv4_3', vgg_layers)
        net2 = tf.batch_to_space(net2, crops=zero_pad, block_size=2)
        net2 = tf.space_to_batch(net2, paddings=zero_pad, block_size=4)
        net2 = convolve(net2, conv5_sz, k, 'conv5_1', vgg_layers)
        net2 = convolve(net2, conv5_sz, k, 'conv5_2', vgg_layers)
        net2 = convolve(net2, conv5_sz, k, 'conv5_3', vgg_layers)
        net2 = tf.batch_to_space(net2, crops=zero_pad, block_size=4)


      #concat = tf.concat(3, [net, net2])

      # TODO do we need this
      #net = ops.max_pool(net, [2, 2], stride=1, scope='pool5')
      #net = ops.max_pool(net, [2, 2], stride=1, padding='SAME', scope='pool5')

      #conv3_shape = conv3_3.get_shape()
      #resize_shape = [conv3_shape[1].value, conv3_shape[2].value]
      #up_conv5_3 = tf.image.resize_nearest_neighbor(conv5_3, resize_shape)
      ##up_conv5_3 = tf.image.resize_nearest_neighbor(conv5_3, [108, 256])
      #concat = tf.concat(3, [conv3_3, up_conv5_3])
      #net = slim.ops.max_pool(net, [2, 2], scope='pool5')

      #net = convolve(net, 4096, 7, 'conv6_1', vgg_layers)
      with scopes.arg_scope([ops.conv2d, ops.fc], batch_norm_params=bn_params):
      #with scopes.arg_scope([ops.conv2d, ops.fc], batch_norm_params=bn_params, weight_decay=0.0005):
        #net = convolve(net, 1024, 3, 'conv6_1')
        #net = tf.nn.atrous_conv2d(net, [7, 7, 512, 1024], rate=4, padding="SAME", name='conv6_1')
        net = tf.space_to_batch(net, paddings=zero_pad, block_size=4)
        #net = convolve(net, 1024, 7, 'conv6_1')
        net = convolve(net, 512, 7, 'conv6_1')
        net = tf.batch_to_space(net, crops=zero_pad, block_size=4)
        curr_scope = tf.get_variable_scope()
        with tf.variable_scope(curr_scope, reuse=True):
          print(net2.get_shape())
          pad = [[0, 0], [2, 2]]
          net2 = tf.space_to_batch(net2, paddings=pad, block_size=8)
          #net2 = convolve(net2, 1024, 7, 'conv6_1')
          net2 = convolve(net2, 512, 7, 'conv6_1')
          net2 = tf.batch_to_space(net2, crops=pad, block_size=8)

        print(net.get_shape())
        print(net2.get_shape())
        concat = tf.concat(3, [net, net2])

        #net = convolve(net, 1024, 7, 'conv6_1')
        net = convolve(concat, 512, 3, 'conv6_2')
        net = convolve(net, 512, 3, 'conv6_3')
        net = convolve(net, 512, 1, 'fc7')

        #net = ops.conv2d(net, 1024, [1, 1], scope='conv5_4')
        #net = ops.conv2d(net, 512, [1, 1], scope='conv6_1')
        #net = ops.conv2d(net, 512, [1, 1], scope='conv6_2')
      net = convolve(net, FLAGS.num_classes, 1, 'score', activation=None)
      #net = slim.ops.flatten(net, scope='flatten5')
      #net = slim.ops.fc(net, 4096, scope='fc6')
      #net = slim.ops.dropout(net, 0.5, scope='dropout6')
      #net = slim.ops.fc(net, 4096, scope='fc7')
      #net = slim.ops.dropout(net, 0.5, scope='dropout7')
      #net = slim.ops.fc(net, 1000, activation=None, scope='fc8')
      logits_up = tf.image.resize_bilinear(net, [FLAGS.img_height, FLAGS.img_width],
                                           name='resize_scores')
  return logits_up


def loss(logits, labels, weights, num_labels, is_training=True):
  loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, num_labels)
  #loss_val = losses.weighted_hinge_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss_symmetric(logits, labels, weights, num_labels)
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

