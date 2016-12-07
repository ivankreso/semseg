import tensorflow as tf

def multiclass_hinge_loss(logits, labels, num_classes):
  print('loss: Hinge loss')
  partitions = tf.one_hot(tf.to_int64(labels), num_classes, 1, 0, dtype=tf.int32)
  #print(partitions)
  #one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
  #one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])
  #partitions = tf.to_int32(one_hot_labels)

  num_partitions = 2
  scores, score_yt = tf.dynamic_partition(logits, partitions, num_partitions)
  #scores = tf.reshape(scores, [num_examples, num_classes - 1])
  #score_yt = tf.reshape(score_yt, [num_examples,  1])
  scores = tf.reshape(scores, [-1, num_classes - 1])
  score_yt = tf.reshape(score_yt, [-1,  1])
  #print(scores)
  #print(score_yt)

  margin = 1.0
  #hinge_loss = tf.maximum(0.0, scores - score_yt + margin)
  hinge_loss = tf.square(tf.maximum(0.0, scores - score_yt + margin))
  hinge_loss = tf.reduce_sum(hinge_loss, 1)

  #total_loss = tf.reduce_sum(tf.mul(weights, hinge_loss))
  #total_loss = tf.div(total_loss, tf.to_float(num_examples), name='value')
  total_loss = tf.reduce_mean(hinge_loss)

  #tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
  #tf.nn.l2_loss(t, name=None)
  return total_loss
