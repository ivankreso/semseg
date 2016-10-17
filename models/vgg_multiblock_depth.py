import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import np_helper
import losses
from models.model_helper import convolve, read_vgg_init
#import datasets.reader as reader
import datasets.reader_multiscale as reader
import eval_helper

FLAGS = tf.app.flags.FLAGS

MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]
VGG_INIT, VGG_LAYER_NAMES = read_vgg_init(FLAGS.vgg_init_dir)

def get_reader():
  return reader

def normalize_input(rgb):
  """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
  with tf.name_scope('input'), tf.device('/cpu:0'):
    rgb -= MEAN_RGB
    red, green, blue = tf.split(3, 3, rgb)
    bgr = tf.concat(3, [blue, green, red])
    return bgr

#def inference(inputs, is_training=True):
def build(inputs, labels, weights, num_labels, is_training=True):
  bn_params = {
      # Decay for the moving averages.
      'decay': 0.999,
      'center': True,
      'scale': True,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # None to force the updates
      'updates_collections': None,
      'is_training': is_training,
  }
  weight_decay = 5e-4
  # to big weight_decay = 5e-3
  num_classes = FLAGS.num_classes
  with tf.contrib.framework.arg_scope([layers.convolution2d],
      stride=1, padding='SAME', activation_fn=tf.nn.relu,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = convolve(inputs, 64, 3, 'conv1_1', VGG_INIT)
    net = convolve(net, 64, 3, 'conv1_2', VGG_INIT)
    net = layers.max_pool2d(net, 2, 2, scope='pool1')
    net = convolve(net, 128, 3, 'conv2_1', VGG_INIT)
    net = convolve(net, 128, 3, 'conv2_2', VGG_INIT)
    net = layers.max_pool2d(net, 2, 2, scope='pool2')
    net = convolve(net, 256, 3, 'conv3_1', VGG_INIT)
    net = convolve(net, 256, 3, 'conv3_2', VGG_INIT)
    net = convolve(net, 256, 3, 'conv3_3', VGG_INIT)
    net = layers.max_pool2d(net, 2, 2, scope='pool3')
    net = convolve(net, 512, 3, 'conv4_1', VGG_INIT)
    net = convolve(net, 512, 3, 'conv4_2', VGG_INIT)
    net = convolve(net, 512, 3, 'conv4_3', VGG_INIT)
    net = layers.max_pool2d(net, 2, 2, scope='pool4')
    # put BN here!
    with tf.contrib.framework.arg_scope([layers.convolution2d],
        normalizer_fn=layers.batch_norm, normalizer_params=bn_params):
      net = convolve(net, 512, 3, 'conv5_1', VGG_INIT)
      net = convolve(net, 512, 3, 'conv5_2', VGG_INIT)
      net = convolve(net, 512, 3, 'conv5_3', VGG_INIT)
      #net = layers.convolution2d(vgg_out, 512, 7, scope='conv6_1')
      #net = layers.convolution2d(net, 512, 3, scope='conv6_2')

      #print(net)
      #r = 4
      #if FLAGS.img_width <= 720:
      #  r = 2
      #rates = [1, 2, 4, 6, 8, 10]

      #pool_rates = [2, 4, 8, 16]
      #pool_rates = [0]
      pool_rates = [0, 2, 2, 2]
      conv_kernels = [5, 5, 3, 3]

      vgg_levels = [net]
      for i in range(3):
        print(net)
        net = layers.max_pool2d(net, 2, 2, padding='SAME', scope='pool'+str(5+i))
        vgg_levels += [net]

      num_scales = 4
      scale_levels = []
      for s in range(num_scales):
        for i, level in enumerate(vgg_levels):
          net = layers.convolution2d(level, 512, conv_kernels[i],
              scope='conv6_'+str(s)+'_'+str(i+1)+'_1')
          net = layers.convolution2d(net, 512, 3, scope='conv6_'+str(s)+'_'+str(i+1)+'_2')
          net = layers.convolution2d(net, 512, 3, scope='conv6_'+str(s)+'_'+str(i+1)+'_3')
          print(net)
          if i == 0:
            top_layer = net
            top_height = net.get_shape()[1].value
            top_width = net.get_shape()[2].value
          else:
            net = tf.image.resize_bilinear(net, [top_height, top_width],
                                           name='conv6_upsample_'+str(s)+'_'+str(i+1))
            #net = tf.image.resize_nearest_neighbor(net, [top_height, top_width],
            #                               name='conv6_upsample_'+str(s)+'_'+str(i+1))
            top_layer = tf.concat(3, [top_layer, net])

        net = layers.convolution2d(top_layer, 1024, 3, scope='conv7_'+str(s)+'_1')
        #net = layers.convolution2d(top_layer, 512, 3, scope='conv7_2')
        net = layers.convolution2d(net, 1024, 1, scope='conv7_'+str(s)+'_3')
        scale_levels += [net]
        #for i, head in enumerate(levels):
        #  levels[i] = layers.convolution2d(head, 512, 1, scope='conv7_'+str(i+1))
        ##  #levels[i] = layers.convolution2d(head, 512, 3, scope='conv7_'+str(i+1))


    #loss_val = 0
    #for i, head in enumerate(levels):
    #  net = layers.convolution2d(head, num_classes, 1, activation_fn=None,
    #                             scope='scores_'+str(i+1))
    #  levels[i] = tf.image.resize_bilinear(net, [FLAGS.img_height, FLAGS.img_width],
    #                                       name='resize_score_'+str(i+1))
    #logits = levels[0]
    #for i in range(len(levels)-1):
    #  logits += levels[i+1]
    #loss_val += loss(logits, labels, weights, num_labels, is_training)
    #return logits, loss_val, [logits] + levels

    loss_val = 0
    multi_logits = []
    for s in range(num_scales):
      net = layers.convolution2d(scale_levels[s], num_classes, 1, activation_fn=None,
                                 scope='scores_'+str(s))
      logits = tf.image.resize_bilinear(net, [FLAGS.img_height, FLAGS.img_width],
                                        name='resize_score_'+str(s))
      print('labels = ', labels[0,s])
      loss_val += loss(logits, labels[0,s], weights, num_labels, is_training)
      multi_logits += [logits]
      if s == 0:
        logits_concat = logits
      else:
        logits_concat = tf.concat(0, [logits_concat, logits])
  #return logits, loss_val, multi_logits
  return logits_concat, loss_val, multi_logits
  #return multi_logits, loss_val, multi_logits


def draw_output(draw_data, class_info, save_prefix):
  for i, logits in enumerate(draw_data):
    label_img = logits[0].argmax(2).astype(np.int32)
    eval_helper.draw_output(label_img, class_info,
                            save_prefix + '_lvl_' + str(i) + '.png')


def loss(logits, labels, weights, num_labels, is_training=True):
  loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, num_labels)
  #loss_val = losses.multiclass_hinge_loss(logits, labels, weights)
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

