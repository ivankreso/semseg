import tensorflow as tf
import numpy as np
import skimage as ski
import skimage.data, skimage.transform
import models.vgg_16s as model

#img = ski.data.load('/home/kivan/datasets/Cityscapes/ppm/rgb/')
batch_size = 1
#height = 512
#width = 1024
height = 64
width = 64
channels = 3
num_classes = 19
num_examples = batch_size * height * width
#img = ski.transform.resize(img, (height, width))
#img = (img - img.mean()) / img.std()
#batch = img.reshape(1, height, width, 3)
batch = np.random.randn(1, height, width, 3)
batch_y = np.zeros((batch_size * height * width))
#batch_y[:] = -1

data = tf.placeholder(tf.float32, shape=(batch_size, height, width, channels))
#labels = tf.placeholder(tf.float32, shape=(batch_size, height/16, width/16, 19))
indices = tf.placeholder(tf.int32, shape=(batch_size * height * width))
labels = tf.one_hot(tf.to_int64(indices), num_classes, 1, 0)
logits_16s = model.inference(data)
logits_2d = tf.image.resize_bilinear(logits_16s, [height, width])
logits = tf.reshape(logits_2d, [num_examples, num_classes])
log_softmax = tf.log(tf.nn.softmax(logits))
xent = -tf.mul(tf.to_float(labels), log_softmax)

#loss = slim.losses.cross_entropy_loss(logits, labels)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
  #logits = sess.run(net, feed_dict = {data: batch})
  sess.run(init_op)
  #loss_val = sess.run(net, feed_dict = {data: batch})
  #loss_val = sess.run([labels, logits], feed_dict = {data: batch, indices: batch_y})
  #loss_val = sess.run(logits, feed_dict = {data: batch})
  print(sess.run(xent, feed_dict = {data: batch, indices: batch_y}))
  #loss_val = sess.run(loss, feed_dict = {data: batch, indices: batch_y})
