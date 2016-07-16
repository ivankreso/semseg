import tensorflow as tf
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

def inference(inputs):
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.0005):
    net = slim.ops.repeat_op(1, inputs, slim.ops.conv2d, 64, [3, 3], scope='conv1')
    net = slim.ops.max_pool(net, [2, 2], scope='pool1')
    net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 128, [3, 3], scope='conv2')
    net = slim.ops.max_pool(net, [2, 2], scope='pool2')
    net = slim.ops.repeat_op(2, net, slim.ops.conv2d, 256, [3, 3], scope='conv3')
    net = slim.ops.max_pool(net, [2, 2], scope='pool3')
    net = slim.ops.repeat_op(2, net, slim.ops.conv2d, 512, [3, 3], scope='conv4')
    net = slim.ops.max_pool(net, [2, 2], scope='pool4')
    net = slim.ops.repeat_op(2, net, slim.ops.conv2d, 512, [3, 3], scope='conv5')
    #net = slim.ops.max_pool(net, [2, 2], scope='pool5')
    net = slim.ops.conv2d(net, 1024, [1, 1], scope='fc6')
    net = slim.ops.conv2d(net, 1024, [1, 1], scope='fc7')
    net = slim.ops.conv2d(net, 19, [1, 1], scope='score')
    #net = slim.ops.flatten(net, scope='flatten5')
    #net = slim.ops.fc(net, 4096, scope='fc6')
    #net = slim.ops.dropout(net, 0.5, scope='dropout6')
    #net = slim.ops.fc(net, 4096, scope='fc7')
    #net = slim.ops.dropout(net, 0.5, scope='dropout7')
    #net = slim.ops.fc(net, 1000, activation=None, scope='fc8')
  return net


def loss(logits, labels):
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
  logits_1s = tf.image.resize_bilinear(logits, [FLAGS.img_height, FLAGS.img_width])
  logits_1d = tf.reshape(logits_1s, [num_examples, FLAGS.num_classes])
  return slim.losses.cross_entropy_loss(logits_1d, one_hot_labels)
