import sys

import tensorflow as tf
import numpy as np
sys.path.append('/home/kivan/source/forks/tf/models/inception/')
from inception.slim import slim

def vgg16(inputs):
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

import skimage as ski
import skimage.data, skimage.transform
img = ski.data.load('/home/kivan/datasets/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png')
batch_size = 1
height = 512
width = 1024
channels = 3
num_classes = 19
num_examples = batch_size * height * width
img = ski.transform.resize(img, (height, width))
img = (img - img.mean()) / img.std()
batch = img.reshape(1, height, width, 3)
batch_y = np.zeros((batch_size * height * width))
#batch_y[:] = -1

data = tf.placeholder(tf.float32, shape=(batch_size, height, width, channels))
#labels = tf.placeholder(tf.float32, shape=(batch_size, height/16, width/16, 19))
indices = tf.placeholder(tf.int32, shape=(batch_size * height * width))
labels = tf.one_hot(tf.to_int64(indices), num_classes, 1, 0)
logits_16s = vgg16(data)
logits_2d = tf.image.resize_bilinear(logits_16s, [height, width])
logits = tf.reshape(logits_2d, [num_examples, num_classes])
loss = slim.losses.cross_entropy_loss(logits, labels)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
  #logits = sess.run(net, feed_dict = {data: batch})
  sess.run(init_op)
  #loss_val = sess.run(net, feed_dict = {data: batch})
  #loss_val = sess.run([labels, logits], feed_dict = {data: batch, indices: batch_y})
  #loss_val = sess.run(logits, feed_dict = {data: batch})
  loss_val = sess.run(loss, feed_dict = {data: batch, indices: batch_y})

#sess = tf.InteractiveSession()
#loss.eval()


#sess = tf.InteractiveSession()
#a = tf.constant(5.0)
#b = tf.constant(6.0)
#c = a * b
## We can just use 'c.eval()' without passing 'sess'
#print(c.eval())
#sess.close()
