import os
import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def _read_and_decode(filename_queue, size):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'channels': tf.FixedLenFeature([], tf.int64),
          'num_labels': tf.FixedLenFeature([], tf.int64),
          'img_name': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string),
          'class_hist': tf.FixedLenFeature([], tf.string),
          'labels': tf.FixedLenFeature([], tf.string),
          #'depth': tf.FixedLenFeature([], tf.string),
      })

  #height = features['height']
  #width = features['width']
  #channels = features['channels']
  #assert FLAGS.img_height == height
  #assert FLAGS.img_width == width
  #assert FLAGS.img_depth == channels
  img_name = features['img_name']
  num_labels = features['num_labels']
  labels = tf.to_int32(tf.decode_raw(features['labels'], tf.int8, name='decode_labels'))
  #depth = tf.to_float(tf.decode_raw(features['depth'], tf.uint8, name='decode_depth'))
  image = tf.to_float(tf.decode_raw(features['image'], tf.uint8, name='decode_image'))
  class_hist = tf.decode_raw(features['class_hist'], tf.int32, name='decode_weights')

  height = size[0]
  width = size[1]
  #image = tf.reshape(image, shape=[FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
  image = tf.reshape(image, shape=[height, width, 3])
  #depth = tf.reshape(depth, shape=[FLAGS.img_height, FLAGS.img_width, 1])
  #num_pixels = FLAGS.img_height * FLAGS.img_width
  #labels = tf.reshape(labels, shape=[num_pixels])
  labels = tf.reshape(labels, shape=[height, width, 1])
  #weights = tf.reshape(weights, shape=[num_pixels])
  #class_hist = tf.reshape(class_hist, shape=[FLAGS.img_height, FLAGS.img_width, 1])
  class_hist = tf.reshape(class_hist, shape=[FLAGS.num_classes])
  #image = tf.Print(image, [img_name, image[100,100,:]], message="P1: ")

  return image, labels, num_labels, class_hist, img_name


def num_examples(dataset):
  return int(dataset.num_examples() // FLAGS.batch_size)


def _create_fetcher(filenames, name, size, num_epochs, shuffle, batch_size):
  with tf.name_scope(name), tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs,
        #shuffle=shuffle, seed=FLAGS.seed, capacity=dataset.num_examples())
        shuffle=shuffle, capacity=len(filenames))

    #filename_queue_size = tf.Print(filename_queue.size(), [filename_queue.size()])
    #with tf.control_dependencies([filename_queue_size]):
    #image, labels, weights, depth, img_name = _read_and_decode(filename_queue)
    image, labels, num_labels, class_hist, img_name = _read_and_decode(filename_queue, size)

    # Shuffle the examples and collect them into batch_size batches.
    # Run this in two threads to avoid being a bottleneck.
    image, labels, num_labels, class_hist, img_name = tf.train.batch(
        [image, labels, num_labels, class_hist, img_name],
        batch_size=batch_size, num_threads=1, capacity=64)

    #image = tf.Print(image, [img_name, tf.shape(image)], message=name, summarize=20)
    return [image, labels, num_labels, class_hist, img_name]


def inputs(filenames, size, is_training=False, num_epochs=None, stack_idx=None):
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
    batch_size = FLAGS.batch_size_valid
    #assert dataset.num_examples() % batch_size == 0
  return _create_fetcher(filenames, 'input', size, num_epochs,
                         shuffle, batch_size)
