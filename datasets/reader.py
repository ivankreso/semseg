import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def _read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'num_labels': tf.FixedLenFeature([], tf.int64),
          'img_name': tf.FixedLenFeature([], tf.string),
          'rgb': tf.FixedLenFeature([], tf.string),
          'label_weights': tf.FixedLenFeature([], tf.string),
          'labels': tf.FixedLenFeature([], tf.string),
      })

  #labels = tf.decode_raw(features['labels'], tf.int32)
  labels = tf.to_int32(tf.decode_raw(features['labels'], tf.int8, name='decode_labels'))
  #image = tf.decode_raw(features['rgb'], tf.float32, name='decode_image')
  image = tf.to_float(tf.decode_raw(features['rgb'], tf.uint8, name='decode_image'))
  weights = tf.decode_raw(features['label_weights'], tf.float32, name='decode_weights')
  num_labels = features['num_labels']
  img_name = features['img_name']
  #width = features['width']
  #depth = features['depth']
  #image.set_shape([mnist.IMAGE_PIXELS])
  #image.set_shape([height, width, depth])
  #image.set_shape([FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
  #labels.set_shape([num_pixels])

  image = tf.reshape(image, shape=[FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
  num_pixels = FLAGS.img_height * FLAGS.img_width
  labels = tf.reshape(labels, shape=[num_pixels])
  weights = tf.reshape(weights, shape=[num_pixels])

  #image = tf.Print(image, [img_name, image[100,100,:]], message="P1: ")
  #image = tf.Print(image, [img_name, image[100,101,:]], message="P2: ")
  #image = tf.Print(image, [img_name, image[100,100,:]], message="P1_N: ")
  #image = tf.Print(image, [img_name, image[100,101,:]], message="P2_N: ")

  return image, labels, weights, num_labels, img_name


def num_examples(dataset):
  return int(dataset.num_examples() / FLAGS.batch_size)


def inputs(dataset, shuffle=True, num_epochs=None):
  """Reads input data num_epochs times.

  Args:
    dataset:
    num_epochs: Number of times to read the input data, or 0/None to train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
    * labels is an int32 tensor with shape [batch_size] with the true label
  """

  with tf.name_scope('input'), tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer(dataset.get_filenames(), num_epochs=num_epochs,
        shuffle=shuffle, seed=FLAGS.seed, capacity=dataset.num_examples())

    #filename_queue_size = tf.Print(filename_queue.size(), [filename_queue.size()])
    #with tf.control_dependencies([filename_queue_size]):
    image, labels, weights, num_labels, img_name = _read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # Run this in two threads to avoid being a bottleneck.
    image, labels, weights, num_labels, img_name = tf.train.batch(
        [image, labels, weights, num_labels, img_name], batch_size=FLAGS.batch_size, num_threads=2,
        capacity=64)

    return image, labels, weights, num_labels, img_name


def inputs_single_epoch(sess, dataset, shuffle=True):
  batch_size = FLAGS.batch_size
  
  with tf.name_scope('input'), tf.device('/cpu:0'):
  #with tf.name_scope('input'):
    if shuffle:
      filename_queue = tf.RandomShuffleQueue(capacity=dataset.num_examples(),
                                             min_after_dequeue=0, dtypes=tf.string)
      #filename_queue = tf.FIFOQueue(capacity=dataset.num_examples(), dtypes=tf.string)
    else:
      filename_queue = tf.FIFOQueue(capacity=dataset.num_examples(), dtypes=tf.string)
    enqueue_placeholder = tf.placeholder(dtype=tf.string)
    enqueue_op = filename_queue.enqueue(enqueue_placeholder)
    dataset.enqueue(sess, enqueue_op, enqueue_placeholder)
    sess.run(filename_queue.close())

    image, labels, weights, num_labels, img_name = _read_and_decode(filename_queue)

    image, labels, weights, num_labels, img_name = tf.train.batch(
        [image, labels, weights, num_labels, img_name], batch_size=batch_size, num_threads=2,
        capacity=64)

    return image, labels, weights, num_labels, img_name

#def inputs_for_inference(sess, dataset):
#  batch_size = FLAGS.batch_size
#  
#  with tf.name_scope('input'), tf.device('/cpu:0'):
#  #with tf.name_scope('input'):
#    filename_queue = tf.FIFOQueue(capacity=dataset.num_examples(), dtypes=tf.string)
#    enqueue_placeholder = tf.placeholder(dtype=tf.string)
#    enqueue_op = filename_queue.enqueue(enqueue_placeholder)
#    for f in dataset.get_filenames():
#      sess.run([enqueue_op], feed_dict={enqueue_placeholder: f})
#    sess.run(filename_queue.close())
#
#    image, labels, weights, num_labels, img_name = read_and_decode(filename_queue)
#
#    images, labels, weights, num_labels, img_name = tf.train.batch(
#        [image, labels, weights, num_labels, img_name], batch_size=batch_size, num_threads=2,
#        capacity=64)
#
#    num_pixels = FLAGS.img_height * FLAGS.img_width
#    labels = tf.reshape(labels, shape=[num_pixels])
#    weights = tf.reshape(weights, shape=[num_pixels])
#    num_labels = tf.reshape(num_labels, shape=[])
#
#    return images, labels, weights, num_labels, img_name, filename_queue

