import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def _read_and_decode(filename_queue, is_training):
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

  assert FLAGS.batch_size <= 2
  #labels = tf.decode_raw(features['labels'], tf.int32)
  labels = tf.to_int32(tf.decode_raw(features['labels'], tf.int8, name='decode_labels'))
  #image = tf.decode_raw(features['rgb'], tf.float32, name='decode_image')
  image = tf.to_float(tf.decode_raw(features['rgb'], tf.uint8, name='decode_image'))
  weights = tf.decode_raw(features['label_weights'], tf.float32, name='decode_weights')
  num_labels = features['num_labels']
  img_name = features['img_name']
  #width = features['width']
  #depth = features['depth']
  #image.set_shape([height, width, depth])
  #image.set_shape([FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
  #labels.set_shape([num_pixels])

  #labels = tf.reshape(labels, shape=[num_pixels])
  #weights = tf.reshape(weights, shape=[num_pixels])

  num_pixels = FLAGS.img_height * FLAGS.img_width
  image = tf.reshape(image, shape=[FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
  image_flip = tf.image.flip_left_right(image)
  image = tf.reshape(image, shape=[1, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
  image_flip = tf.reshape(image_flip, shape=[1, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
  image = tf.concat(0, [image, image_flip])

  labels = tf.reshape(labels, shape=[FLAGS.img_height, FLAGS.img_width, 1])
  labels_flip = tf.image.flip_left_right(labels)
  labels = tf.reshape(labels, shape=[1, FLAGS.img_height, FLAGS.img_width, 1])
  labels_flip = tf.reshape(labels_flip, shape=[1, FLAGS.img_height, FLAGS.img_width, 1])
  labels = tf.concat(0, [labels, labels_flip])

  weights = tf.reshape(weights, shape=[FLAGS.img_height, FLAGS.img_width, 1])
  weights_flip = tf.image.flip_left_right(weights)
  weights = tf.reshape(weights, shape=[1, FLAGS.img_height, FLAGS.img_width, 1])
  weights_flip = tf.reshape(weights_flip, shape=[1, FLAGS.img_height, FLAGS.img_width, 1])
  weights = tf.concat(0, [weights, weights_flip])
  #labels_flip = tf.reshape(labels_flip, shape=[num_pixels])
  #weights = tf.reshape(weights, shape=[num_pixels])
  #weights_flip = tf.reshape(weights_flip, shape=[num_pixels])
  num_labels = tf.reshape(num_labels, shape=[1])
  num_labels = tf.concat(0, [num_labels, num_labels])

  img_name = tf.reshape(img_name, shape=[1])
  img_name = tf.concat(0, [img_name, img_name])

  return image, labels, weights, num_labels, img_name


def num_examples(dataset):
  return int(dataset.num_examples())
  #return int(2 * dataset.num_examples() / FLAGS.batch_size)


def inputs(dataset, is_training=False, num_epochs=None):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  shuffle = is_training
  if is_training:
    batch_size = FLAGS.batch_size
  else:
    #batch_size = 1
    batch_size = FLAGS.batch_size

  with tf.name_scope('input'), tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer(dataset.get_filenames(),
        num_epochs=num_epochs, shuffle=shuffle, capacity=dataset.num_examples())
    
    #filename_queue_size = tf.Print(filename_queue.size(), [filename_queue.size()])
    #with tf.control_dependencies([filename_queue_size]):
    image, labels, weights, num_labels, img_name = _read_and_decode(filename_queue,
        is_training)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue)
    # Run this in two threads to avoid being a bottleneck.
    image, labels, weights, num_labels, img_name = tf.train.batch(
        [image, labels, weights, num_labels, img_name], batch_size=batch_size, num_threads=2,
        #[image, labels, weights, num_labels, img_name], batch_size=2, num_threads=2,
        enqueue_many=True, capacity=64)
        #enqueue_many=True, capacity=64)

    return image, labels, weights, num_labels, img_name

