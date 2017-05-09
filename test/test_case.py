import tensorflow as tf

sess = tf.Session()

#x = tf.constant(0)
#y = tf.constant(1)
#z = tf.constant(2)
#def f1(): return tf.constant(17)
#def f2(): return tf.constant(23)
#def f3(): return tf.constant(-1)
#r = tf.case({tf.less(x, y): f1, tf.greater(x, z): f2},
#         default=f3, exclusive=True)
#print(sess.run(r))
#raise 1


idx = tf.placeholder(tf.int32, shape=(), name='data_stack_index')
#idx = tf.constant(1)
pred_pairs = {}
#funcs = []
size = 10
for i in range(size):
  #def f(): return tf.constant(i)
  #funcs.append(f)
  pred_pairs[tf.equal(idx, tf.constant(i))] = lambda i=i: tf.constant(i)
  #pred_pairs[tf.equal(idx, tf.constant(i))] = funcs[-1]
  print(tf.constant(i))
print(pred_pairs.keys())

case = tf.case(pred_pairs, default=lambda: tf.constant(-1), exclusive=True)
case = tf.Print(case, [idx], message='idx = ', summarize=10)
for i in range(size):
  print(i, ' -- ', sess.run(case, feed_dict={idx:i}))
  #print(i, ' -- ', sess.run(case))
