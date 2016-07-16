import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tqdm import trange
from inception.slim import slim
from shutil import copyfile
import importlib.util
import subprocess

import libs.cylib as cylib
import eval_helper
from train_helper import *
from datasets.cityscapes.cityscapes import CityscapesDataset
from datasets.cityscapes.cityscapes_info import class_info, class_color_map
import datasets.reader as reader


#g_model_path = './models/vgg_16s.py'
g_model_path = './models/vgg_16s_baseline.py'
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
#                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_epochs', 20, 'Number of epochs to run.')
tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_integer('num_classes', 19, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
#tf.app.flags.DEFINE_boolean('is_training', True, '')

#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-5,
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4,
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3,
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 3.0,
#tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
#tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.2,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, '')


def evaluate_epoch(dataset, checkpoint_path):
  with tf.Graph().as_default(), tf.Session() as sess:
    image, labels, _, _, img_name = reader.inputs_single_epoch(sess, dataset, shuffle=False)
    # Restores from checkpoint
    logits = model.inference(image, is_training=False)

    #saver = tf.train.Saver()
    variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_path)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    data_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #pixel_acc, iou_acc, recall, precision, num_pixels = evaluate_model(sess, dataset, logits, labels)
    conf_mat = np.ascontiguousarray(np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64))
    for i in trange(dataset.num_examples()):
      out_logits, yt = sess.run([logits, labels])
      y = out_logits[0].argmax(2).astype(np.int32, copy=False)
      yt = yt.astype(np.int32, copy=False)
      cylib.collect_confusion_matrix(y.reshape(-1), yt, conf_mat)
      #print(q_size)
    pixel_acc, iou_acc, recall, precision, num_pixels = eval_helper.compute_errors(conf_mat,
                                                        'Validation', class_info, verbose=True)

    coord.request_stop()
    # TODO
    coord.join(data_threads)
  return pixel_acc, iou_acc, recall, precision

def train_epoch(dataset, resume_path, epoch_num):
  with tf.Graph().as_default():
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess = tf.Session()
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (dataset.num_examples() /
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
    #image, labels, weights, num_labels, img_name, filename_queue, enqueue_op, \
    image, labels, weights, num_labels, img_name = reader.inputs_single_epoch(sess, dataset)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(image)

    # Calculate loss.
    loss = model.loss(logits, labels, weights, num_labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    #train_op = model.train(loss, global_step)
      # Variables that affect learning rate.

    # Add a summary to track the learning rate.
    tf.scalar_summary('learning_rate', lr)
    #tf.scalar_summary('learning_rate', tf.mul(lr, tf.constant(1 / FLAGS.initial_learning_rate)))

    # Generate moving averages of all losses and associated summaries.
    #loss_averages_op = add_loss_summaries(loss)

    # Compute gradients.
    #with tf.control_dependencies([loss_averages_op]):
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
    #saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.max_epochs)
    saver = tf.train.Saver(tf.all_variables())
    if resume_path == None:
      # Build an initialization operation to run below.
      init = tf.initialize_all_variables()
      sess.run(init)
    else:
      assert tf.gfile.Exists(resume_path)
      saver.restore(sess, resume_path)
      #variables_to_restore = tf.get_collection(
      #    slim.variables.VARIABLES_TO_RESTORE)
      #restorer = tf.train.Saver(variables_to_restore)
      #restorer.restore(sess, resume_path)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Start running operations on the Graph.
    #dataset.enqueue(sess, enqueue_op, enqueue_placeholder)
    #filename_queue.close()

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #graph_def = sess.graph.as_graph_def(add_shapes=True)
    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

    num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
    ex_start_time = time.time()
    conf_mat = np.ascontiguousarray(np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64))
    for step in range(dataset.num_examples()):
      start_time = time.time()
      #_, loss_value = sess.run([train_op, loss])
      if step % 100 == 0:
        _, loss_value, scores, yt, clr, filename, global_step_val, summary_str = sess.run(
            [train_op, loss, logits, labels, lr, img_name, global_step, summary_op])
        summary_writer.add_summary(summary_str, global_step_val)
      else:
        _, loss_value, scores, yt, clr, filename, global_step_val = sess.run(
            [train_op, loss, logits, labels, lr, img_name, global_step])
      #loss_val, out_logits, yt = sess.run([loss, logits, labels])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      y = scores[0].argmax(2).astype(np.int32, copy=False)
      cylib.collect_confusion_matrix(y.reshape(-1), yt, conf_mat)

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = '%s: epoch %d, step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
        #print('lr = ', clr)
        print(format_str % (get_expired_time(ex_start_time), epoch_num, step, loss_value,
                            examples_per_sec, sec_per_batch))

    # Save the model checkpoint periodically.
    eval_helper.compute_errors(conf_mat, 'train', class_info)
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=epoch_num)

    coord.request_stop()
    #print('join')
    # TODO
    coord.join(threads)
    sess.close()

  return checkpoint_path + '-' + str(epoch_num)


def train(train_dataset, val_dataset):
  iou_acc_data = []
  pixel_acc_data = []
  checkpoint_path = None
  for epoch_num in range(1, FLAGS.max_epochs):
    checkpoint_path = train_epoch(train_dataset, checkpoint_path, epoch_num)
    pixel_acc, iou, recall, precision = evaluate_epoch(val_dataset, checkpoint_path)
    #time.sleep(1)
    iou_acc_data += [iou]
    pixel_acc_data += [pixel_acc]
    #iou_acc_data += [epoch_num]
    #pixel_acc_data += [epoch_num]
    #if epoch_num > 1:
    #  eval_helper.plot_accuracy(iou_fig, iou_acc_data)
    #  eval_helper.plot_accuracy(pixel_fig, pixel_acc_data)
  input('Press ENTER to exit...')


def main(argv=None):  # pylint: disable=unused-argument
  #if tf.gfile.Exists(FLAGS.train_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
  #tf.gfile.MakeDirs(FLAGS.train_dir)
  print('Experiment dir: ' + FLAGS.train_dir)
  print('Dataset dir: ' + FLAGS.dataset_dir)
  train_dataset = CityscapesDataset(FLAGS.dataset_dir, 'train')
  val_dataset = CityscapesDataset(FLAGS.dataset_dir, 'val')
  train(train_dataset, val_dataset)


if __name__ == '__main__':
  tf.app.run()

