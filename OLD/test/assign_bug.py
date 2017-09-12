import tensorflow as tf

sess = tf.Session()
a = tf.get_variable('a', [], initializer=tf.constant_initializer(1))
b = tf.get_variable('b', [], initializer=tf.constant_initializer(1))
sess.run(tf.initialize_all_variables())
assign_a = a.assign(tf.constant(10.0))
with tf.control_dependencies([assign_a]):
  b = tf.Print(b, [b, a], message='b, a = ')
print(sess.run(b))
