import tensorflow as tf
from slim import ops
from slim import scopes
import numpy as np
import np_helper
import losses
import models.model_helper as model_helper
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

  def build_shared_net(rgb_data):
    net = convolve(rgb_data, conv1_sz, k, 'conv1_1', vgg_layers)
    net = convolve(net, conv1_sz, k, 'conv1_2', vgg_layers)
    net = ops.max_pool(net, [2, 2], scope='pool1')
    net = convolve(net, conv2_sz, k, 'conv2_1', vgg_layers)
    net = convolve(net, conv2_sz, k, 'conv2_2', vgg_layers)
    net = ops.max_pool(net, [2, 2], scope='pool2')
    net = convolve(net, conv3_sz, k, 'conv3_1', vgg_layers)
    net = convolve(net, conv3_sz, k, 'conv3_2', vgg_layers)
    conv3_3 = convolve(net, conv3_sz, k, 'conv3_3', vgg_layers)
    net = ops.max_pool(conv3_3, [2, 2], scope='pool3')
    net = convolve(net, conv4_sz, k, 'conv4_1', vgg_layers)
    net = convolve(net, conv4_sz, k, 'conv4_2', vgg_layers)
    net = convolve(net, conv4_sz, k, 'conv4_3', vgg_layers)
    net = ops.max_pool(net, [2, 2], scope='pool4')
    net = convolve(net, conv5_sz, k, 'conv5_1', vgg_layers)
    net = convolve(net, conv5_sz, k, 'conv5_2', vgg_layers)
    net = convolve(net, conv5_sz, k, 'conv5_3', vgg_layers)
    #pad = [[0, 0], [0, 0]]
    #net = tf.space_to_batch(net, paddings=pad, block_size=2)
    #net = convolve(net, conv5_sz, k, 'conv5_1', vgg_layers)
    #net = convolve(net, conv5_sz, k, 'conv5_2', vgg_layers)
    #net = convolve(net, conv5_sz, k, 'conv5_3', vgg_layers)
    #net = tf.batch_to_space(net, crops=pad, block_size=2)
    return net

  # best so far = 0.0005
  with scopes.arg_scope([ops.conv2d, ops.fc], stddev=0.01, weight_decay=0.0005):
  #with scopes.arg_scope([ops.conv2d, ops.fc], stddev=0.01, weight_decay=0.005):
  #with scopes.arg_scope([ops.conv2d, ops.fc], stddev=0.01, weight_decay=0.05):
  #with scopes.arg_scope([ops.conv2d, ops.fc], stddev=0.01):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      #scale_factors = [1.2, 0.8, 0.4]
      #scale_factors = [1.0, 0.65, 0.3]
      scale_factors = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15]
      IMG_WIDTH = 1124
      IMG_HEIGHT = 1024
      assert FLAGS.img_width == IMG_WIDTH
      #scale_factors = [1.2, 0.9, 0.6, 0.3]
      #scale_factors = [1.2, 1.0, 0.8, 0.6, 0.4]
      #scale_factors = [1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
      #scale_factors = [1.0, 0.8, 0.6, 0.4, 0.2]
      resolutions, _ = model_helper.get_multiscale_resolutions(
          FLAGS.img_width, FLAGS.img_height, scale_factors)
      print(resolutions)
      with tf.variable_scope('shared') as scope:
        nets = []
        for i, res in enumerate(resolutions):
          #input_i = tf.image.resize_bilinear(inputs, res, name='resize_level' + str(i))
          #input_i = tf.image.resize_nearest_neighbor(inputs, res, name='resize_level' + str(i))
          input_i = tf.image.resize_bicubic(inputs[0], res, name='resize_level' + str(i))
          net = build_shared_net(input_i)
          print(net.get_shape())
          code_size = net.get_shape()[3].value
          print('code_sz=', code_size)
          nets += [tf.reshape(net, shape=[-1, code_size])]
          #print(nets)
          scope.reuse_variables()
        #for i, net in enumerate(nets):
        net = tf.concat(0, nets)
        print(net)
        net = tf.gather(net, inputs[1])
        print(net)
        batch_size = net.get_shape()[0].value
        height = net.get_shape()[1].value
        width = net.get_shape()[2].value
        num_scales = net.get_shape()[3].value
        code_size = net.get_shape()[4].value
        #net = tf.reshape(net, shape=[1, 64, 70, 3 * 512])
        net = tf.reshape(net, shape=[batch_size, height, width, num_scales * code_size])
        print(net)

        #level0_shape = nets[0].get_shape()
        #level0_shape = [level0_shape[1].value, level0_shape[2].value]
        #print(level0_shape)
        #for i in range(1, len(nets)):
        #  nets[i] = tf.image.resize_nearest_neighbor(
        #      nets[i], level0_shape, name='upsample_level' + str(i))


        #net_sum = nets[2]
        #for i in range(3, len(nets)):
        #  net_sum += nets[i]
        #net = tf.concat(3, [nets[0], nets[1], net_sum])

        #net = tf.concat(3, [nets[0], nets[1], nets[2]])
        #net = tf.concat(3, [nets[3], nets[4]])

      #conv3_shape = conv3_3.get_shape()
      #resize_shape = [conv3_shape[1].value, conv3_shape[2].value]
      #up_conv5_3 = tf.image.resize_nearest_neighbor(conv5_3, resize_shape)
      ##up_conv5_3 = tf.image.resize_nearest_neighbor(conv5_3, [108, 256])
      #concat = tf.concat(3, [conv3_3, up_conv5_3])
      #net = slim.ops.max_pool(net, [2, 2], scope='pool5')

      #net = convolve(net, 4096, 7, 'conv6_1', vgg_layers)
      #with scopes.arg_scope([ops.conv2d, ops.fc], batch_norm_params=bn_params):
      with scopes.arg_scope([ops.conv2d, ops.fc], batch_norm_params=bn_params,
                             weight_decay=0.0005):
        #net = convolve(net, 512, 1, 'conv6_2')
        #net = convolve(net, 512, 1, 'conv6_3')
        #net = convolve(net, 1024, 7, 'conv6_1')
        net = convolve(net, 1024, 5, 'conv6_1')
        #net = convolve(net, 1024, 3, 'conv6_1')
        net = convolve(net, 512, 3, 'conv6_2')
        #net = convolve(net, 512, 3, 'conv6_3')
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
  #loss_val = losses.cross_entropy_loss(logits, labels, num_labels)
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

