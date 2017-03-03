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
          #'num_labels': tf.FixedLenFeature([], tf.int64),
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
  img_name = features['img_name']
  #width = features['width']
  #depth = features['depth']
  #image.set_shape([mnist.IMAGE_PIXELS])
  #image.set_shape([height, width, depth])
  #image.set_shape([FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
  #labels.set_shape([num_pixels])

  image = tf.reshape(image, shape=[FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
  #num_pixels = FLAGS.img_height * FLAGS.img_width
  #labels = tf.reshape(labels, shape=[num_pixels])
  labels = tf.reshape(labels, shape=[FLAGS.img_height, FLAGS.img_width, 1])
  #weights = tf.reshape(weights, shape=[num_pixels])
  weights = tf.reshape(weights, shape=[FLAGS.img_height, FLAGS.img_width, 1])

  #image = tf.Print(image, [img_name, image[100,100,:]], message="P1: ")
  #image = tf.Print(image, [img_name, image[100,101,:]], message="P2: ")
  #image = tf.Print(image, [img_name, image[100,100,:]], message="P1_N: ")
  #image = tf.Print(image, [img_name, image[100,101,:]], message="P2_N: ")

  return image, labels, weights, img_name


def num_examples(dataset):
  return int(dataset.num_examples() // FLAGS.batch_size)


def inputs(dataset, is_training=False, num_epochs=None):
  """Reads input data num_epochs times.

  Args:
    dataset:
    num_epochs: Number of times to read the input data, or 0/None to train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
    * labels is an int32 tensor with shape [batch_size] with the true label
  """
  shuffle = is_training
  if is_training:
    batch_size = FLAGS.batch_size
  else:
    batch_size = 1

  with tf.name_scope('input'), tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer(dataset.get_filenames(), num_epochs=num_epochs,
        shuffle=shuffle, seed=FLAGS.seed, capacity=dataset.num_examples())

    #filename_queue_size = tf.Print(filename_queue.size(), [filename_queue.size()])
    #with tf.control_dependencies([filename_queue_size]):
    image, labels, weights, img_name = _read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # Run this in two threads to avoid being a bottleneck.
    image, labels, weights, img_name = tf.train.batch(
        [image, labels, weights, img_name], batch_size=batch_size, num_threads=2,
        capacity=64)

    return image, labels, weights, labels, img_name

