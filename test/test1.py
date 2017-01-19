import tensorflow as tf

indices = tf.constant([[4], [3], [1], [7]], tf.int64)
updates = tf.constant([9.1, 10.2, 11, 12])
shape = tf.constant([8], tf.int64)
print(indices)
print(updates)
print(shape)
scatter = tf.scatter_nd(indices, updates, shape)
oshape = tf.constant([2,2,2], tf.int32)
scatter = tf.reshape(scatter, oshape)
with tf.Session() as sess:
  print(sess.run(scatter))

