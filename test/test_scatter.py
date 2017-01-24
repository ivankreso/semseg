import numpy as np
import tensorflow as tf

#indices = tf.constant([[4], [3], [1], [7]], tf.int64)
#updates = tf.constant([9.1, 10.2, 11, 12])
#shape = tf.constant([8], tf.int64)
#print(indices)
#print(updates)
#print(shape)
#scatter = tf.scatter_nd(indices, updates, shape)
#oshape = tf.constant([2,2,2], tf.int32)
#scatter = tf.reshape(scatter, oshape)
#with tf.Session() as sess:
#  print(sess.run(scatter))


img_shape = (2,4,4,1)
img = tf.placeholder(tf.float32, img_shape)
ksize = [1,2,2,1]
stride = [1,2,2,1]
pool, argmax = tf.nn.max_pool_with_argmax(img, ksize, stride, padding='SAME', name='pool')

img_np = np.zeros(img_shape)
img_np[0,0,0] = 1
img_np[0,2,2] = 2
img_np[1,3,3] = 3

with tf.Session() as sess:
  ops = [pool, argmax]
  feed = {img:img_np}
  out = sess.run(ops, feed_dict=feed)
  print(img_np)
  print(out[0])
  print(out[1])
