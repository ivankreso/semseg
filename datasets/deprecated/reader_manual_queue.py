
import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          #'image_raw': tf.FixedLenFeature([], tf.string),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'num_labels': tf.FixedLenFeature([], tf.int64),
          'img_name': tf.FixedLenFeature([], tf.string),
          'class_weights': tf.FixedLenFeature([], tf.string),
          'rgb_norm': tf.FixedLenFeature([], tf.string),
          'labels': tf.FixedLenFeature([], tf.string),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['rgb_norm'], tf.float32)
  labels = tf.decode_raw(features['labels'], tf.int32)
  weights = tf.decode_raw(features['class_weights'], tf.float32)
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

  return image, labels, weights, num_labels, img_name


#def inputs(dataset, batch_size, num_epochs, train):
def inputs(filename_queue, num_epochs=False, train=True):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
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
  batch_size = FLAGS.batch_size
  
  if not num_epochs: num_epochs = None

  with tf.name_scope('input'):
    #filename_queue = tf.train.string_input_producer(dataset.get_filenames(), num_epochs=num_epochs)
    #filename_queue = tf.RandomShuffleQueue(dataset.size(), 0, tf.string)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, labels, weights, num_labels, img_name = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    # TODO images, sparse_labels = tf.train.batch(
    #images, labels, weights, num_labels = tf.train.shuffle_batch(
    #    [image, labels, weights, num_labels], batch_size=batch_size, num_threads=2,
    #    capacity=300,
    #    # Ensures a minimum amount of shuffling of examples.
    #    min_after_dequeue=200)

    images, labels, weights, num_labels, img_name = tf.train.batch(
        [image, labels, weights, num_labels, img_name],
        batch_size=batch_size, num_threads=2,
        capacity=50)
                   
#tensor_list, batch_size, num_threads=1, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, shared_name=None, name=None)

    num_pixels = FLAGS.img_height * FLAGS.img_width
    labels = tf.reshape(labels, shape=[num_pixels])
    weights = tf.reshape(weights, shape=[num_pixels])
    num_labels = tf.reshape(num_labels, shape=[])

    return images, labels, weights, num_labels, img_name



#def parse_example_proto(example_serialized):
#  """Parses an Example proto containing a training example of an image.
#
#  The output of the build_image_data.py image preprocessing script is a dataset
#  containing serialized Example protocol buffers. Each Example proto contains
#  the following fields:
#
#    image/height: 462
#    image/width: 581
#    image/colorspace: 'RGB'
#    image/channels: 3
#    image/class/label: 615
#    image/class/synset: 'n03623198'
#    image/class/text: 'knee pad'
#    image/object/bbox/xmin: 0.1
#    image/object/bbox/xmax: 0.9
#    image/object/bbox/ymin: 0.2
#    image/object/bbox/ymax: 0.6
#    image/object/bbox/label: 615
#    image/format: 'JPEG'
#    image/filename: 'ILSVRC2012_val_00041207.JPEG'
#    image/encoded: <JPEG encoded string>
#
#  Args:
#    example_serialized: scalar Tensor tf.string containing a serialized
#      Example protocol buffer.
#
#  Returns:
#    image_buffer: Tensor tf.string containing the contents of a JPEG file.
#    label: Tensor tf.int32 containing the label.
#    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
#      where each coordinate is [0, 1) and the coordinates are arranged as
#      [ymin, xmin, ymax, xmax].
#    text: Tensor tf.string containing the human-readable label.
#  """
#  # Dense features in Example proto.
#  feature_map = {
#      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
#        default_value=''),
#      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
#        default_value=-1),
#      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
#        default_value=''),
#      }
#  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
#  # Sparse features in Example proto.
#  feature_map.update(
#      {k: sparse_float32 for k in ['image/object/bbox/xmin',
#        'image/object/bbox/ymin',
#        'image/object/bbox/xmax',
#        'image/object/bbox/ymax']})
#
#      features = tf.parse_single_example(example_serialized, feature_map)
#  label = tf.cast(features['image/class/label'], dtype=tf.int32)
#
#  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
#  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
#  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
#  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
#
#  # Note that we impose an ordering of (y, x) just to make life difficult.
#  bbox = tf.concat(0, [ymin, xmin, ymax, xmax])
#
#  # Force the variable number of bounding boxes into the shape
#  # [1, num_boxes, coords].
#  bbox = tf.expand_dims(bbox, 0)
#  bbox = tf.transpose(bbox, [0, 2, 1])
#
#  return features['image/encoded'], label, bbox, features['image/class/text']
#
#
#def batch_inputs(dataset, batch_size, num_preprocess_threads=None):
#  """Contruct batches of training or evaluation examples from the image dataset.
#
#  Args:
#    dataset: instance of Dataset class specifying the dataset.
#      See dataset.py for details.
#    batch_size: integer
#    train: boolean
#    num_preprocess_threads: integer, total number of preprocessing threads
#
#  Returns:
#    images: 4-D float Tensor of a batch of images
#    labels: 1-D integer Tensor of [batch_size].
#
#  Raises:
#    ValueError: if data is not found
#  """
#  with tf.name_scope('batch_processing'):
#    data_files = dataset.data_files()
#    if data_files is None:
#      raise ValueError('No data files found for this dataset')
#    filename_queue = tf.train.string_input_producer(data_files, capacity=16)
#
#    if num_preprocess_threads is None:
#      num_preprocess_threads = FLAGS.num_preprocess_threads
#
#    if num_preprocess_threads % 4:
#      raise ValueError('Please make num_preprocess_threads a multiple '
#          'of 4 (%d % 4 != 0).', num_preprocess_threads)
#      # Create a subgraph with its own reader (but sharing the
#    # filename_queue) for each preprocessing thread.
#    images_and_labels = []
#    for thread_id in range(num_preprocess_threads):
#      reader = dataset.reader()
#      _, example_serialized = reader.read(filename_queue)
#
#      # Parse a serialized Example proto to extract the image and metadata.
#      image_buffer, label_index, bbox, _ = parse_example_proto(
#          example_serialized)
#      image = image_preprocessing(image_buffer, bbox, train, thread_id)
#      images_and_labels.append([image, label_index])
#
#    # Approximate number of examples per shard.
#    examples_per_shard = 1024
#    # Size the random shuffle queue to balance between good global
#    # mixing (more examples) and memory use (fewer examples).
#    # 1 image uses 299*299*3*4 bytes = 1MB
#    # The default input_queue_memory_factor is 16 implying a shuffling queue
#    # size: examples_per_shard * 16 * 1MB = 17.6GB
#    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
#
#    # Create a queue that produces the examples in batches after shuffling.
#    if train:
#      images, label_index_batch = tf.train.shuffle_batch_join(
#          images_and_labels,
#          batch_size=batch_size,
#          capacity=min_queue_examples + 3 * batch_size,
#          min_after_dequeue=min_queue_examples)
#    else:
#      images, label_index_batch = tf.train.batch_join(
#          images_and_labels,
#          batch_size=batch_size,
#          capacity=min_queue_examples + 3 * batch_size)
#
#      # Reshape images into these desired dimensions.
#    height = FLAGS.image_size
#    width = FLAGS.image_size
#    depth = 3
#
#    images = tf.cast(images, tf.float32)
#    images = tf.reshape(images, shape=[batch_size, height, width, depth])
#
#    # Display the training images in the visualizer.
#    tf.image_summary('images', images)
#
#    return images, tf.reshape(label_index_batch, [batch_size])
#
#def inputs(self):
#  # Force all input processing onto CPU in order to reserve the GPU for
#  # the forward inference and back-propagation.
#  with tf.device('/cpu:0'):
#    images, labels = batch_inputs(dataset, batch_size, train=False,
#        num_preprocess_threads=num_preprocess_threads)

