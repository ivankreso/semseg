#import skimage.io  # bug. need to import this before tensorflow
#import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.training import moving_averages

from models.config import Config
from models.model_helper import convolve
import losses
from slim import scopes
from slim import ops

FLAGS = tf.app.flags.FLAGS

#MOVING_AVERAGE_DECAY = 0.9997
MOVING_AVERAGE_DECAY = 0.999
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
#CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_DECAY = 0.0005
CONV_WEIGHT_STDDEV = 0.1
#FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_DECAY = 0.0005
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
VARIABLES_TO_RESTORE = 'resnet_variables_to_restore'
#UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
#IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]
#MEAN_RGB = [123.151630838, 115.902882574, 103.062623801]
# cityscapes mean
MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]


BN_PARAMS = {
  # Decay for the moving averages.
  'decay': BN_DECAY,
  # epsilon to prevent 0s in variance.
  'epsilon': BN_EPSILON,
  #'center': False,
  #'scale': False,
  'center': True,
  #'scale': True
  'scale': False
}

activation = tf.nn.relu


def variables_to_restore():
  return tf.get_collection(RESNET_VARIABLES)


def normalize_input(rgb):
  """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
  with tf.name_scope('input'), tf.device('/cpu:0'):
    rgb -= MEAN_RGB
    red, green, blue = tf.split(3, 3, rgb)
    bgr = tf.concat(3, [blue, green, red])
    return bgr


#def normalize_input(rgb):
#  """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
#  with tf.name_scope('input'), tf.device('/cpu:0'):
#    red, green, blue = tf.split(3, 3, rgb)
#    bgr = tf.concat(3, [blue, green, red])
#    #bgr -= IMAGENET_MEAN_BGR
#    bgr = tf.div(bgr, 127.5)
#    bgr = tf.sub(bgr, 1.0)
#    return bgr

def build(inputs, labels, weights, num_labels, is_training=True):
  logits = inference(inputs, is_training)
  loss_val = loss(logits, labels, weights, num_labels, is_training)
  return logits, loss_val


def inference(x, is_training,
              num_classes=19,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=True):
  if FLAGS.num_layers == 50:
    num_blocks=[3, 4, 6, 3]
  elif FLAGS.num_layers == 101:
    num_blocks=[3, 4, 23, 3]
  else:
    raise ValueError()

  c = Config()
  c['bottleneck'] = bottleneck
  #c['is_training'] = tf.convert_to_tensor(is_training,
  #                                        dtype='bool',
  #                                        name='is_training')
  c['is_training'] = is_training
  c['ksize'] = 3
  c['stride'] = 1
  c['use_bias'] = use_bias
  c['fc_units_out'] = num_classes
  c['num_blocks'] = num_blocks
  c['stack_stride'] = 2

  with tf.variable_scope('scale1'):
    c['conv_filters_out'] = 64
    c['ksize'] = 7
    c['stride'] = 2
    x = conv(x, c)
    x = bn(x, c)
    x = activation(x)
    # TODO added one more conv
    #with scopes.arg_scope([ops.conv2d], stddev=CONV_WEIGHT_STDDEV, is_training=is_training,
    #                      weight_decay=CONV_WEIGHT_DECAY, batch_norm_params=BN_PARAMS):
    #    x = convolve(x, 64, 7, 'conv2', stride=2)

  with tf.variable_scope('scale2'):
    # TODO k was 3
    x = _max_pool(x, ksize=3, stride=2)
    #x = _max_pool(x, ksize=2, stride=2)
    c['num_blocks'] = num_blocks[0]
    c['stack_stride'] = 1
    c['block_filters_internal'] = 64
    x = stack(x, c)

  with tf.variable_scope('scale3'):
    c['num_blocks'] = num_blocks[1]
    c['block_filters_internal'] = 128
    assert c['stack_stride'] == 2
    x = stack(x, c)

  with tf.variable_scope('scale4'):
    c['num_blocks'] = num_blocks[2]
    c['block_filters_internal'] = 256
    x = stack(x, c)

  with tf.variable_scope('scale5'):
    #TODO s was 2
    c['stack_stride'] = 1
    c['num_blocks'] = num_blocks[3]
    c['block_filters_internal'] = 512
    x = stack(x, c)

  # post-net
  #x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

  with tf.variable_scope('score'):
    with scopes.arg_scope([ops.conv2d], stddev=CONV_WEIGHT_STDDEV, is_training=is_training,
                          weight_decay=CONV_WEIGHT_DECAY):
      with scopes.arg_scope([ops.conv2d], batch_norm_params=BN_PARAMS):
        r = 4
        #x = convolve(x, 1024, 5, 'conv6_1', dilation=r)
        #x = convolve(x, 512, 3, 'conv6_2')
        fix this BN is not the same
        x = convolve(x, 1024, 1, 'conv6_0')
        x = convolve(x, 512, 7, 'conv6_1', dilation=r)
        x = convolve(x, 512, 3, 'conv6_2')

      x = convolve(x, num_classes, 1, 'score', activation=None)
      x = tf.image.resize_bilinear(x, [FLAGS.img_height, FLAGS.img_width],
                                    name='resize_score')

  return x


def loss(logits, labels, weights, num_labels, is_training=True):
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, num_labels)
  loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, num_labels,
                                                max_weight=100)
  #loss_val = losses.multiclass_hinge_loss(logits, labels, weights, num_labels)
  all_losses = [loss_val]

  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)

  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss

#def loss(logits, labels):
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
#    cross_entropy_mean = tf.reduce_mean(cross_entropy)
#
#    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#
#    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
#    tf.scalar_summary('loss', loss_)
#
#    return loss_


def stack(x, c):
  for n in range(c['num_blocks']):
    s = c['stack_stride'] if n == 0 else 1
    c['block_stride'] = s
    with tf.variable_scope('block%d' % (n + 1)):
      x = block(x, c)
  return x


def block(x, c):
  filters_in = x.get_shape()[-1]
  # Note: filters_out isn't how many filters are outputed.
  # That is the case when bottleneck=False but when bottleneck is
  # True, filters_internal*4 filters are outputted. filters_internal is how many filters
  # the 3x3 convs output internally.
  m = 4 if c['bottleneck'] else 1
  filters_out = m * c['block_filters_internal']
  shortcut = x  # branch 1
  c['conv_filters_out'] = c['block_filters_internal']

  if c['bottleneck']:
    with tf.variable_scope('a'):
      c['ksize'] = 1
      c['stride'] = c['block_stride']
      x = conv(x, c)
      x = bn(x, c)
      x = activation(x)

    with tf.variable_scope('b'):
      x = conv(x, c)
      x = bn(x, c)
      x = activation(x)

    with tf.variable_scope('c'):
      c['conv_filters_out'] = filters_out
      c['ksize'] = 1
      assert c['stride'] == 1
      x = conv(x, c)
      x = bn(x, c)
  else:
    with tf.variable_scope('A'):
      c['stride'] = c['block_stride']
      assert c['ksize'] == 3
      x = conv(x, c)
      x = bn(x, c)
      x = activation(x)

    with tf.variable_scope('B'):
      c['conv_filters_out'] = filters_out
      assert c['ksize'] == 3
      assert c['stride'] == 1
      x = conv(x, c)
      x = bn(x, c)

  with tf.variable_scope('shortcut'):
    if filters_out != filters_in or c['block_stride'] != 1:
      c['ksize'] = 1
      c['stride'] = c['block_stride']
      c['conv_filters_out'] = filters_out
      shortcut = conv(shortcut, c)
      shortcut = bn(shortcut, c)

  return activation(x + shortcut)


def bn(x, c):
  x_shape = x.get_shape()
  params_shape = x_shape[-1:]

  if c['use_bias']:
    bias = _get_variable('bias', params_shape,
                         initializer=tf.zeros_initializer)
    return x + bias


  axis = list(range(len(x_shape) - 1))

  beta = _get_variable('beta',
                       params_shape,
                       initializer=tf.zeros_initializer)
  gamma = _get_variable('gamma',
                        params_shape,
                        initializer=tf.ones_initializer)

  moving_mean = _get_variable('moving_mean',
                              params_shape,
                              initializer=tf.zeros_initializer,
                              trainable=False)
  moving_variance = _get_variable('moving_variance',
                                  params_shape,
                                  initializer=tf.ones_initializer,
                                  trainable=False)

  # These ops will only be preformed when training.
  if c['is_training']:
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
      moving_variance, variance, BN_DECAY)
    #print(params_shape)
    #update_moving_mean = moving_mean.assign(mean)
    #update_moving_variance = moving_variance.assign(variance)
    #tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    #tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    #if moving_mean.name == 'scale1/moving_mean:0':
    #  mean = tf.Print(mean, [mean, moving_mean], message='mean = ' + moving_mean.name)
    #variance = tf.Print(variance, [variance, moving_variance], message='variance = ' + moving_variance.name)
    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
      x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
      #x = tf.Print(x, [mean, moving_mean], message='mean = ' + moving_mean.name)
  else:
    mean = moving_mean
    variance = moving_variance
    if moving_mean.name == 'scale1/moving_mean:0':
      mean, variance = tf.nn.moments(x, axis)

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x = tf.Print(x, [mean], message='mean = ' + moving_mean.name)

  #x.set_shape(inputs.get_shape()) ??
  return x


def fc(x, c):
  num_units_in = x.get_shape()[1]
  num_units_out = c['fc_units_out']
  weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)

  weights = _get_variable('weights',
                          shape=[num_units_in, num_units_out],
                          initializer=weights_initializer,
                          weight_decay=FC_WEIGHT_STDDEV)
  biases = _get_variable('biases',
                          shape=[num_units_out],
                          initializer=tf.zeros_initializer)
  x = tf.nn.xw_plus_b(x, weights, biases)
  return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True,
                  restore=True):
  "A little wrapper around tf.get_variable to do weight decay and add to"
  "resnet collection"
  if weight_decay > 0:
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    regularizer = None
  if restore:
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES, VARIABLES_TO_RESTORE]
  else:
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
  return tf.get_variable(name,
                          shape=shape,
                          initializer=initializer,
                          dtype=dtype,
                          regularizer=regularizer,
                          collections=collections,
                          trainable=trainable)


def conv(x, c):
  ksize = c['ksize']
  stride = c['stride']
  filters_out = c['conv_filters_out']

  filters_in = x.get_shape()[-1]
  shape = [ksize, ksize, filters_in, filters_out]
  initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
  weights = _get_variable('weights',
                          shape=shape,
                          dtype='float',
                          initializer=initializer,
                          weight_decay=CONV_WEIGHT_DECAY)
  return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
  return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                        strides=[1, stride, stride, 1],
                        padding='SAME')


'''
#aspp_rates = [None, 6, 12]
#aspp_rates = [None, 2, 4]
#aspp_rates = [2]
#aspp_branches = []
with tf.variable_scope('classifier') as scope:
  ##x = convolve(x, 512, 7, 'conv6_1')
  ## 5x5 much faster then 7x7
  #pad = [[0, 0], [0, 0]]
  ## set rate=8 with 3x3
  #rate = 2
  #x = tf.space_to_batch(x, paddings=pad, block_size=rate)
  #x = convolve(x, 512, 5, 'conv6_1')
  #x = convolve(x, 512, 3, 'conv6_2')
  ##x = convolve(x, 512, 1, 'conv6_3')
  #x = tf.batch_to_space(x, crops=pad, block_size=rate)
  #for i, r in enumerate(aspp_rates):
  #  #aspp_branches += [convolve(x, 512, 5, 'conv6_1_branch_' + str(i), dilation=r)]
  #  #aspp_branches += [convolve(x, 512, 5, 'conv6_1', dilation=r)]
  #  #with scopes.arg_scope([ops.conv2d], batch_norm_params=BN_PARAMS):
  #  #  conv6 = convolve(x, 512, 5, 'conv6_1', dilation=r)
  #  #  conv6 = convolve(conv6, 512, 1, 'conv6_2')
  #  #aspp_branches += [convolve(conv6, num_classes, 1, 'score', activation=None)]
  #    #conv6 = convolve(x, 512, 5, 'conv6_1'+str(i), dilation=r)
  #    #conv6 = convolve(x, 1024, 5, 'conv6_1'+str(i), dilation=r)
  #    #conv6 = convolve(conv6, 512, 1, 'conv6_2'+str(i))
  #    #conv6 = convolve(x, 512, 5, 'conv6_2'+str(i))
  #  aspp_branches += [convolve(conv6, num_classes, 1, 'score'+str(i), activation=None)]
  #TODO try without reuse
  #scope.reuse_variables()

            #conv6 = convolve(x, 512, 5, 'conv6_1_branch_'+str(i), dilation=r)
            #aspp_branches += [convolve(conv6, num_classes, 1, 'score_branch'+str(i), activation=None)]

            #aspp_branches += [convolve(x, 512, 3, 'conv6_1_branch_' + str(i), dilation=r)]
            #x = tf.space_to_batch(x, paddings=pad, block_size=rate)
            #x = convolve(x, 512, 5, 'conv6_1')
            #x = convolve(x, 512, 3, 'conv6_2')
            ##x = convolve(x, 512, 1, 'conv6_3')
            #x = tf.batch_to_space(x, crops=pad, block_size=rate)
            #if len(aspp_branches) > 1:
            #  x = tf.add_n(aspp_branches)
            '''
