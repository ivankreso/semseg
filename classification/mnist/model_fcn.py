import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope
import losses


def total_loss_sum(losses):
  # Assemble all of the losses for the current tower only.
  # Calculate the total loss for the current tower.
  regularization_losses = tf.contrib.losses.get_regularization_losses()
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  return total_loss


def loss(logits, labels, num_classes):
  xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  loss = tf.reduce_mean(xent_loss)
  #loss = losses.multiclass_hinge_loss(logits, labels, num_classes)
  total_loss = total_loss_sum([loss])
  return total_loss



def layer(net, num_filters, name, is_training):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    net = layers.convolution2d(net, num_filters, kernel_size=3)
    #if is_training: 
      #net = tf.nn.dropout(net, keep_prob=0.8)
  return net

def dense_block(net, size, r, name, is_training):
  with tf.variable_scope(name):
    outputs = []
    for i in range(size):
      if i < size - 1:
        x = net
        net = layer(net, r, 'layer'+str(i), is_training)
        outputs += [net]
        net = tf.concat(3, [x, net])
      else:
        net = layer(net, r, 'layer'+str(i), is_training)
        outputs += [net]
    net = tf.concat(3, outputs)
  return net

def downsample(net, name, is_training):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net)
    net = tf.nn.relu(net)
    num_filters = net.get_shape().as_list()[3]
    net = layers.convolution2d(net, num_filters, kernel_size=1)
    #if is_training:
    #  net = tf.nn.dropout(net, keep_prob=0.8)
    net = layers.max_pool2d(net, 2, stride=2, padding='SAME')
  return net

def upsample(net, name):
  with tf.variable_scope(name):
    num_filters = net.get_shape().as_list()[3]
    net = tf.contrib.layers.convolution2d_transpose(net, num_filters, kernel_size=3, stride=2)
    return net

def _build(image, num_classes, is_training):
  global bn_params
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
  weight_decay = 1e-4
  #init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
  init_func = layers.variance_scaling_initializer()

  cfg = {
    2: [3,4,5],
  }
  block_sizes = cfg[2]
  r = 16
  
  with arg_scope([layers.convolution2d, layers.convolution2d_transpose],
      stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    block_outputs = []
    with tf.variable_scope('classifier'):
      net = layers.convolution2d(image, 48, 3, scope='conv0')
      for i, size in enumerate(block_sizes):
        print(i, size)
        x = net
        net = dense_block(net, size, r, 'block'+str(i), is_training)
        net = tf.concat(3, [x, net])
        block_outputs += [net]
        if i < len(block_sizes) - 1:
          net = downsample(net, 'block'+str(i)+'_downsample', is_training)
        print(net)
      #logits_mid = layers.convolution2d(net, FLAGS.num_classes, 1,
      #    biases_initializer=tf.zeros_initializer, scope='logits_middle')
      #logits_mid = tf.image.resize_bilinear(logits_mid, [FLAGS.img_height, FLAGS.img_width],
      #                                  name='resize_logits_middle')

      #net = tf.nn.relu(net)
      #num_filters = net.get_shape().as_list()[3]
      #net = layers.convolution2d(net, num_filters, kernel_size=1)

    for i, size in reversed(list(enumerate(block_sizes[:-1]))):
      print(i, size)
      net = upsample(net, 'block'+str(i)+'_back_upsample')
      print(block_outputs[i])
      net = tf.concat(3, [block_outputs[i], net])
      print(net)
      net = dense_block(net, size, r, 'block'+str(i)+'_back', is_training)
      print(net)

    mask = layers.convolution2d(net, 1, 1, biases_initializer=tf.zeros_initializer,
        scope='mask')
    mask = tf.nn.relu(mask)
    l1_scale = 1e-3
    #l1_scale = 1e-6
    l1_regularizer = layers.l1_regularizer(l1_scale)
    l1_loss = l1_regularizer(mask)
    #l1_loss = 0
    image = tf.mul(image, mask)

    #tf.get_variable_scope().reuse_variables()
    with tf.variable_scope('classifier', reuse=True):
      net = layers.convolution2d(image, 48, 3, scope='conv0')
      for i, size in enumerate(block_sizes):
        print(i, size)
        x = net
        net = dense_block(net, size, r, 'block'+str(i), is_training)
        net = tf.concat(3, [x, net])
        if i < len(block_sizes) - 1:
          net = downsample(net, 'block'+str(i)+'_downsample', is_training)
        print(net)
    #logits = layers.convolution2d(net, num_classes, 1, biases_initializer=tf.zeros_initializer,
    #    scope='logits')

  with tf.contrib.framework.arg_scope([layers.fully_connected],
      activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=layers.variance_scaling_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.flatten(net, scope='flatten')
    #net = layers.fully_connected(net, 512, scope='fc3')
    #net = layers.fully_connected(net, 256, scope='fc4')
    net = layers.fully_connected(net, 256, scope='fc3')
    net = layers.fully_connected(net, 128, scope='fc4')
  logits = layers.fully_connected(net, num_classes, activation_fn=None,
      weights_regularizer=layers.l2_regularizer(weight_decay), scope='logits')

  return logits, mask, l1_loss


def build(x, labels, num_classes, is_training, reuse=False):
  if reuse:
    tf.get_variable_scope().reuse_variables()

  #logits = _build(x, is_training)
  logits, mask, reg_loss = _build(x, num_classes, is_training)
  total_loss = loss(logits, labels, is_training) + reg_loss

  #all_vars = tf.contrib.framework.get_variables()
  #for v in all_vars:
  #  print(v.name)
  return [total_loss, logits, mask]


def build_old(inputs, labels, num_classes, is_training):
  weight_decay = 1e-4
  # to big weight_decay = 5e-3
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
  conv1sz = 32
  conv2sz = 64
  #conv1sz = 32
  #conv2sz = 64
  with tf.contrib.framework.arg_scope([layers.convolution2d],
      kernel_size=3, stride=1, padding='SAME', rate=1, activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      #normalizer_fn=None,
      weights_initializer=layers.variance_scaling_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.convolution2d(inputs, conv1sz, scope='conv1_1')
    net = layers.convolution2d(net, conv1sz, scope='conv1_2')
    #net = layers.convolution2d(inputs, conv1sz, scope='conv1_1')
    net = layers.max_pool2d(net, 2, 2, padding='SAME', scope='pool1')
    net = layers.convolution2d(net, conv2sz, scope='conv2_1')
    net = layers.convolution2d(net, conv2sz, scope='conv2_2')
    #net = layers.convolution2d(net, conv2sz, kernel_size=5, scope='conv2_2')
    net = layers.max_pool2d(net, 2, 2, padding='SAME', scope='pool2')
    #net = layers.convolution2d(net, conv2sz, padding='VALID', scope='conv3_1')
    #net = layers.convolution2d(net, conv2sz, kernel_size=1, scope='conv3_2')
    #net = layers.convolution2d(net, num_classes, kernel_size=1, scope='conv3_3')
    mask = layers.convolution2d(net, conv2sz, scope='conv1_mask')
    mask = layers.convolution2d(mask, 1, scope='conv2_mask')
    #mask = tf.sigmoid(mask)
    mask = tf.nn.relu(mask)
    print(mask)
    #l1_scale = 1e-4
    l1_scale = 1e-3
    #l1_scale = 1e-9
    l1_regularizer = layers.l1_regularizer(l1_scale)
    l1_loss = l1_regularizer(mask)
    #l1_loss = 0
    #mask = tf.Print(mask, [tf.reduce_sum(mask)], message='MASK=')

    ##net = layers.convolution2d(net, num_classes, kernel_size=1, normalizer_fn=None,
    ##                           activation_fn=None, scope='conv3_3',)
    #print(net.get_shape())
    net = tf.mul(net, mask)
    net = tf.contrib.layers.avg_pool2d(net, kernel_size=7, scope='avg_pool')
    print(net)
    #logits = tf.reshape(logits, [-1, num_classes])
    #print(logits.get_shape())

  with tf.contrib.framework.arg_scope([layers.fully_connected],
      activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      #normalizer_fn=None,
      weights_initializer=layers.variance_scaling_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.flatten(net, scope='flatten')
    #net = layers.fully_connected(net, 512, scope='fc3')
    #net = layers.fully_connected(net, 256, scope='fc4')
    net = layers.fully_connected(net, 256, scope='fc3')
    net = layers.fully_connected(net, 128, scope='fc4')
  logits = layers.fully_connected(net, num_classes, activation_fn=None,
      weights_regularizer=layers.l2_regularizer(weight_decay), scope='logits')
  loss = build_loss(logits, labels, num_classes)
  loss += l1_loss
  return [loss, logits, mask]
