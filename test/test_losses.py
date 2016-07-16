import tensorflow as tf
import numpy as np
import skimage as ski
import skimage.data, skimage.transform
#import models.vgg_16s_baseline as model
import losses

#img = ski.data.load('/home/kivan/datasets/Cityscapes/ppm/rgb/')
batch_size = 1
#height = 512
#width = 1024
height = 64
width = 64
num_classes = 19

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('img_width', width, '')
tf.app.flags.DEFINE_integer('img_height', height, '')
tf.app.flags.DEFINE_integer('batch_size', batch_size, '')
tf.app.flags.DEFINE_integer('num_classes', num_classes, '')

num_examples = batch_size * height * width
np_logits = np.random.randn(batch_size, height, width, num_classes).astype(np.float32)
num_pixels = batch_size * height * width
np_weights = np.abs(np.random.randn(num_pixels) * 100) + 1
np_weights = np_weights.astype(np.float32)
np_labels = np.ones(num_pixels, dtype=np.float32)
np_weights_unit = np.ones(num_pixels, dtype=np.float32)
np_weights_zero = np.zeros(num_pixels, dtype=np.float32)
np_weights_scale = np.ones(num_pixels, dtype=np.float32) * 100

logits = tf.placeholder(tf.float32, shape=(batch_size, height, width, num_classes))
labels = tf.placeholder(tf.int32, shape=(num_pixels))
weights = tf.placeholder(tf.float32, shape=(num_pixels))
loss1 = losses.cross_entropy_loss(logits, labels, weights, num_pixels)
loss2 = losses.weighted_cross_entropy_loss(logits, labels, weights, num_pixels)

#loss = slim.losses.cross_entropy_loss(logits, labels)
#loss = slim.losses.cross_entropy_loss(logits, labels)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
  print(sess.run(loss1, feed_dict = {logits: np_logits, labels: np_labels, weights: np_weights}))
  print(sess.run(loss2, feed_dict = {logits: np_logits, labels: np_labels, weights: np_weights_unit}))
  print(sess.run(loss2, feed_dict = {logits: np_logits, labels: np_labels, weights: np_weights_zero}))
  print(sess.run(loss2, feed_dict = {logits: np_logits, labels: np_labels, weights: np_weights_scale}))
  print(sess.run(loss2, feed_dict = {logits: np_logits, labels: np_labels, weights: np_weights}))
