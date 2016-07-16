import tensorflow as tf
import slim
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

#def inference(inputs, is_training=True):
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
      net = convolve(inputs, conv1_sz, k, 'conv1_1', vgg_layers)
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

      with scopes.arg_scope([ops.conv2d, ops.fc], batch_norm_params=bn_params):
      #with scopes.arg_scope([ops.conv2d, ops.fc], batch_norm_params=bn_params, weight_decay=0.0005):
        pad = [[0, 0], [0, 0]]
        ##net = tf.space_to_batch(net, paddings=pad, block_size=4)
        #net = tf.space_to_batch(net, paddings=pad, block_size=2)
        #net = convolve(net, 1024, 7, 'conv6_1')
        ##net = convolve(net, 512, 5, 'conv6_1')
        #net = tf.batch_to_space(net, crops=pad, block_size=2)

        rate = 2
        net = tf.space_to_batch(net, paddings=pad, block_size=rate)
        net = convolve(net, 512, 5, 'conv6_1')
        net = convolve(net, 512, 3, 'conv6_2')
        #x = convolve(x, 512, 1, 'conv6_3')
        net = tf.batch_to_space(net, crops=pad, block_size=rate)
        ##net = convolve(net, 1024, 7, 'conv6_1')
        #net = convolve(net, 1024, 3, 'conv6_1')
        #net = convolve(net, 512, 3, 'conv6_2')
        ##net = convolve(net, 512, 3, 'conv6_3')

        #net = convolve(net, 1024, 1, 'conv6_1')
        #net = convolve(net, 512, 1, 'conv6_2')
        #net = convolve(net, 512, 1, 'conv6_3')
        #net = convolve(net, 512, 1, 'fc7')

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
      loss_val = loss(logits_up, labels, weights, num_labels, is_training)
  return logits_up, loss_val


def loss(logits, labels, weights, num_labels, is_training=True):
  loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, num_labels)
  #loss_val = losses.multiclass_hinge_loss(logits, labels, weights, num_labels)
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, None, num_labels)
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

