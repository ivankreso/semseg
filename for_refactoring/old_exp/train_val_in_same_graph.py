import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from shutil import copyfile
import importlib.util
import subprocess

from datasets.cityscapes.cityscapes import CityscapesDataset
import datasets.reader as reader


def get_time_string():
  time = datetime.now()
  name = str(time.day) + '_' + str(time.month) + '_%02d' % time.hour + '-%02d' % time.minute + \
         '-%02d' % time.second + '/'
  return name


g_model_path = './models/vgg_16s.py'
spec = importlib.util.spec_from_file_location("model", g_model_path)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
g_save_dir = '/home/kivan/source/results/semseg/tf/nets/' + get_time_string()
tf.gfile.MakeDirs(g_save_dir)
copyfile(g_model_path, g_save_dir + 'model.py')
print('\ntensorboard --logdir=' + g_save_dir + '\n')

#subprocess.Popen(['tensorboard', '--logdir=' + g_save_dir])
#subprocess.Popen('tensorboard --logdir=' + g_save_dir + ' > /dev/null', shell=True)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', g_save_dir,
    #'/home/kivan/source/deep-learning/semantic_segmentation/tensorflow/results/',
    """Directory where to write event logs """
    """and checkpoint.""")
tf.app.flags.DEFINE_integer('img_width', 1024, '')
tf.app.flags.DEFINE_integer('img_height', 432, '')
tf.app.flags.DEFINE_integer('img_depth', 3, '')

tf.app.flags.DEFINE_string('dataset_dir',
    '/home/kivan/datasets/Cityscapes/tensorflow/records/1024x432/', '')
#tf.app.flags.DEFINE_integer('max_steps', 100000,
tf.app.flags.DEFINE_integer('max_epochs', 50,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_integer('num_classes', 19, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('is_training', True, '')

#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-5,
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4,
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3,
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4,
                          """Initial learning rate.""")
#tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
tf.app.flags.DEFINE_float('num_epochs_per_decay', 4.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, '')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])


  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train(train_data, val_data):
  with tf.Graph().as_default():
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (train_data.num_examples() /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    #opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
    #                                momentum=RMSPROP_MOMENTUM,
    #                                epsilon=RMSPROP_EPSILON)

    # Get images and labels.
    filename_queue = tf.RandomShuffleQueue(capacity=max(train_data.num_examples(),
                                           val_data.num_examples()),
                                           min_after_dequeue=0, dtypes=tf.string)
    enqueue_placeholder = tf.placeholder(dtype=tf.string)
    enqueue_op = filename_queue.enqueue(enqueue_placeholder)
    #dequeue_op = filename_queue.dequeue()

    image, labels, weights, num_labels, img_name = reader.inputs(filename_queue)

    input_shape = shape=(FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth)
    #label_shape = shape=(FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width)
    label_shape = shape=(FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width)
    input_p = tf.placeholder(tf.float32, input_shape, 'input_placeholder')
    labels_p = tf.placeholder(tf.int32, label_shape)
    #labels_p = tf.placeholder(tf.int32, ())
    weights_p = tf.placeholder(tf.float32)
    num_labels_p = tf.placeholder(tf.int64)
    train_phase = tf.placeholder(tf.bool)
    #weights_p = tf.placeholder(tf.float32, (None))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(input_p)
    #logits = model.inference(image)

    # Calculate loss.
    loss = model.loss(logits, labels_p, weights_p, num_labels_p)
    #loss = model.loss(logits, labels, weights, num_labels)
    #loss_val = model.loss(logits_val, val_labels, weights, num_labels)
    #loss_val = model.loss(logits_val, labels, weights, num_labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    #train_op = model.train(loss, global_step)
      # Variables that affect learning rate.

    # Add a summary to track the learning rate.
    tf.scalar_summary('learning_rate', lr)
    #tf.scalar_summary('learning_rate', tf.mul(lr, tf.constant(1 / FLAGS.initial_learning_rate)))

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      #opt = tf.train.GradientDescentOptimizer(lr)
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    #tf.train.start_queue_runners(sess=sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #graph_def = sess.graph.as_graph_def(add_shapes=True)
    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

    num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
    for epoch in range(FLAGS.max_epochs):
      #_, loss_value = sess.run([train_op, loss])
      train_data.enqueue(sess, enqueue_op, enqueue_placeholder)
      #for i in range(1000):
      #  print(sess.run([filename_queue.size()])[0])
      #print(train_data.num_examples())
      for step in range(train_data.num_examples()):
        start_time = time.time()
        #_, loss_value, clr = sess.run([train_op, loss, lr])
        print(sess.run([filename_queue.size()])[0])
        _, loss_value, clr, filename = sess.run([train_op, loss, lr, img_name])
        print(filename)
        loss_value, filename = sess.run([loss_val, img_name])
        print(step)
        print(filename)
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print('lr = ', clr)
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

        if step % 100 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)

      #val_data.enqueue(sess, enqueue_op, enqueue_placeholder)
      #for step in range(val_data.num_examples()):
      #  start_time = time.time()
      #  #_, loss_value, clr = sess.run([train_op, loss, lr])
      #  _, loss_value, clr = sess.run(loss)
      #  print(sess.run([filename_queue.size()])[0])
      #  duration = time.time() - start_time
      #for step in range(val_data.size()):

      ## Save the model checkpoint periodically.
      #if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
      #  checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      #  saver.save(sess, checkpoint_path, global_step=step)

  coord.request_stop()
  coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  #if tf.gfile.Exists(FLAGS.train_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
  #tf.gfile.MakeDirs(FLAGS.train_dir)
  print('Experiment dir: ' + FLAGS.train_dir)
  print('Dataset dir: ' + FLAGS.dataset_dir)
  train_data = CityscapesDataset(FLAGS.dataset_dir, 'train')
  val_data = CityscapesDataset(FLAGS.dataset_dir, 'val')
  train(train_data, val_data)


if __name__ == '__main__':
  tf.app.run()

#data = tf.placeholder(tf.float32, shape=(batch_size, height, width, channels))
##labels = tf.placeholder(tf.float32, shape=(batch_size, height/16, width/16, 19))
#indices = tf.placeholder(tf.int32, shape=(batch_size * height * width))
#labels = tf.one_hot(tf.to_int64(indices), num_classes, 1, 0)
#logits_16s = vgg16(data)
#logits_2d = tf.image.resize_bilinear(logits_16s, [height, width])
#logits = tf.reshape(logits_2d, [num_examples, num_classes])
#loss = slim.losses.cross_entropy_loss(logits, labels)
#
#init_op = tf.initialize_all_variables()
#with tf.Session() as sess:
#  #logits = sess.run(net, feed_dict = {data: batch})
#  sess.run(init_op)
#  #loss_val = sess.run(net, feed_dict = {data: batch})
#  #loss_val = sess.run([labels, logits], feed_dict = {data: batch, indices: batch_y})
#  #loss_val = sess.run(logits, feed_dict = {data: batch})
#  loss_val = sess.run(loss, feed_dict = {data: batch, indices: batch_y})
