import os
import sys
import time
from shutil import copyfile
#import subprocess

import numpy as np
import tensorflow as tf
#from tqdm import trange

import libs.cylib as cylib
import helper
import eval_helper
import train_helper
from datasets.cityscapes.cityscapes import CityscapesDataset
#import datasets.flip_reader as reader
#import datasets.reader_pyramid as reader
import datasets.reader as reader

np.set_printoptions(linewidth=250)

tf.app.flags.DEFINE_string('config_path', '', """Path to experiment config.""")
FLAGS = tf.app.flags.FLAGS

helper.import_module('config', FLAGS.config_path)
print(FLAGS.config_path)


def evaluate(sess, epoch_num, logits, loss, labels, img_name, dataset):
  """ Trains the network
    Args:
      sess: TF session
      logits: network logits
  """
  print('\nValidation performance:')
  conf_mat = np.ascontiguousarray(
      np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64))
  loss_avg = 0
  for step in range(dataset.num_examples()):
    start_time = time.time()
    out_logits, gt_labels, loss_val, img_prefix = sess.run(
      [logits, labels, loss, img_name])
    duration = time.time() - start_time
    loss_avg += loss_val
    #net_labels = out_logits[0].argmax(2).astype(np.int32, copy=False)
    net_labels = out_logits[0].argmax(2).astype(np.int32)
    #gt_labels = gt_labels.astype(np.int32, copy=False)
    cylib.collect_confusion_matrix(net_labels.reshape(-1),
                                   gt_labels.reshape(-1), conf_mat)

    if step % 10 == 0:
      num_examples_per_step = FLAGS.batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)
      format_str = 'epoch %d, step %d / %d, loss = %.2f \
        (%.1f examples/sec; %.3f sec/batch)'
      #print('lr = ', clr)
      print(format_str % (epoch_num, step, reader.num_examples(dataset), loss_val,
                          examples_per_sec, sec_per_batch))
    if FLAGS.draw_predictions and step % 100 == 0:
      img_prefix = img_prefix[0].decode("utf-8")
      save_path = FLAGS.debug_dir + '/val/' + '%03d_' % epoch_num + img_prefix + '.png'
      eval_helper.draw_output(net_labels, CityscapesDataset.CLASS_INFO, save_path)
    #print(q_size)
  #print(conf_mat)
  print('')
  pixel_acc, iou_acc, recall, precision, _ = eval_helper.compute_errors(
      conf_mat, 'Validation', CityscapesDataset.CLASS_INFO, verbose=True)
  return loss_avg / dataset.num_examples(), pixel_acc, iou_acc, recall, precision


def train(model, train_dataset, valid_dataset):
  """ Trains the network
    Args:
      model: module containing model architecture
      train_dataset: training data object
      valid_dataset: validation data object
  """
  #with tf.Graph().as_default(), tf.device('/gpu:0'):
  with tf.Graph().as_default():
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5 # don't hog all vRAM
    #config.operation_timeout_in_ms = 5000   # terminate on long hangs
    #config.operation_timeout_in_ms = 15000   # terminate on long hangs
    sess = tf.Session(config=config)
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (train_dataset.num_examples() / FLAGS.batch_size)
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
    image, labels, weights, num_labels, img_name = \
        reader.inputs(train_dataset, num_epochs=FLAGS.max_epochs)
    image_valid, labels_valid, weights_valid, num_labels_valid, img_name_valid = \
        reader.inputs(valid_dataset, shuffle=False, num_epochs=FLAGS.max_epochs)
    image = model.normalize_input(image)
    image_valid = model.normalize_input(image_valid)

    # Build a Graph that computes the logits predictions from the inference model.
    # Calculate loss.
    #with tf.variable_scope("model"):
    logits, loss = model.build(image, labels, weights, num_labels, is_training=True)
    #loss = model.loss(logits, labels, weights, num_labels)
    #with tf.variable_scope("model", reuse=True):
    with tf.variable_scope('', reuse=True):
      logits_valid, loss_valid = model.build(image_valid, labels_valid, weights_valid,
          num_labels_valid, is_training=False)
      #loss_valid = model.loss(logits_valid, labels_valid, weights_valid,
      #                        num_labels_valid, is_training=False)
    #logits_valid, loss_valid = logits, loss


    # Add a summary to track the learning rate.
    tf.scalar_summary('learning_rate', lr)
    #tf.scalar_summary('learning_rate', tf.mul(lr, tf.constant(1 / FLAGS.initial_learning_rate)))

    #with tf.control_dependencies([loss_averages_op]):
    opt = None
    if FLAGS.optimizer == 'Adam':
      opt = tf.train.AdamOptimizer(lr)
    elif FLAGS.optimizer == 'Momentum':
      opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
      #opt = tf.train.GradientDescentOptimizer(lr)
    else:
      raise ValueError()
    grads = opt.compute_gradients(loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.histogram_summary(var.op.name, var)
    # Add histograms for gradients.
    grad_tensors = []
    for grad, var in grads:
      grad_tensors += [grad]
      #print(var)
      if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)
    #grad = grads[-2][0]
    #print(grad)

    # Track the moving averages of all trainable variables.
    #variable_averages = tf.train.ExponentialMovingAverage(
    #    FLAGS.moving_average_decay, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # with slim's BN
    #batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
    #batchnorm_updates_op = tf.group(*batchnorm_updates)
    #train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)
    #with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    with tf.control_dependencies([apply_gradient_op]):
      train_op = tf.no_op(name='train')

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.max_epochs)
    #saver = tf.train.Saver(tf.all_variables())

    # Build an initialization operation to run below.
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    if len(FLAGS.resume_path) > 0:
      print('\nRestoring params from:', FLAGS.resume_path)
      assert tf.gfile.Exists(FLAGS.resume_path)
      resnet_restore = tf.train.Saver(model.variables_to_restore())
      resnet_restore.restore(sess, FLAGS.resume_path)
      #latest = tf.train.latest_checkpoint(FLAGS.train_dir)
      #saver.restore(sess, '')
      #variables_to_restore = tf.get_collection(
      #    slim.variables.VARIABLES_TO_RESTORE)
      #restorer = tf.train.Saver(variables_to_restore)
      #restorer.restore(sess, resume_path)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Start the queue runners.
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

    variable_map = train_helper.get_variable_map()
    # take the train loss moving average
    loss_avg_train = variable_map['total_loss/avg:0']
    train_iou_data = []
    train_pixacc_data = []
    train_loss_data = []
    valid_iou_data = []
    valid_pixacc_data = []
    valid_loss_data = []
    ex_start_time = time.time()
    for epoch_num in range(1, FLAGS.max_epochs + 1):
      print('\ntensorboard --logdir=' + FLAGS.train_dir + '\n')
      conf_mat = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64)
      conf_mat = np.ascontiguousarray(conf_mat)
      #for step in range(reader.num_examples(train_dataset)):
      num_batches = 1488
      #num_batches = train_dataset.num_examples()
      for step in range(num_batches):
        start_time = time.time()
        run_ops = [train_op, loss, logits, labels, lr, img_name, global_step]
        if step % 100 == 0:
          run_ops += [summary_op, loss_avg_train]
          ret_val = sess.run(run_ops)
          (_, loss_val, scores, yt, clr, img_prefix, \
              global_step_val, summary_str, loss_avg_train_val) = ret_val
          summary_writer.add_summary(summary_str, global_step_val)
        else:
          #run_ops += [grad_tensors]
          ret_val = sess.run(run_ops)
          (_, loss_val, scores, yt, clr, img_prefix, global_step_val) = ret_val
          #(_, loss_val, scores, yt, clr, img_prefix, global_step_val, grads_val) = ret_val
          #train_helper.print_grad_stats(grads_val, grad_tensors)
        duration = time.time() - start_time

        #print(ret_val[5])
        #print('loss = ', ret_val[1])
        #print('logits min/max/mean = ', ret_val[2].min(), ret_val[2].max(), ret_val[2].mean())
        #print(ret_val[3].sum())

        assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

        # estimate training accuracy on the last 30% of the epoch
        if step > int(0.7 * num_batches):
          label_map = scores[0].argmax(2).astype(np.int32)
          cylib.collect_confusion_matrix(label_map.reshape(-1),
                                         yt.reshape(-1), conf_mat)

        img_prefix = img_prefix[0].decode("utf-8")

        #print(scores[0,100:102,300:302,:])
        if FLAGS.draw_predictions and step % 100 == 0:
          save_path = os.path.join(FLAGS.debug_dir, 'train',
                                   '%05d_%03d_' % (step, epoch_num) +
                                   img_prefix + '.png')
          #print(save_path)
          label_map = scores[0].argmax(2).astype(np.int32)
          eval_helper.draw_output(label_map, CityscapesDataset.CLASS_INFO, save_path)

        if step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = '%s: epoch %d, step %d / %d, loss = %.2f \
            (%.1f examples/sec; %.3f sec/batch)'
          #print('lr = ', clr)
          print(format_str % (train_helper.get_expired_time(ex_start_time), epoch_num,
                              step, reader.num_examples(train_dataset), loss_val,
                              examples_per_sec, sec_per_batch))

      # TODO add moving average
      train_pixacc, train_iou, _, _, _ = eval_helper.compute_errors(conf_mat, 'Train',
          CityscapesDataset.CLASS_INFO)

      valid_loss, valid_pixacc, valid_iou, valid_recall, valid_precision = evaluate(
        sess, epoch_num, logits_valid, loss_valid,
        labels_valid, img_name_valid, valid_dataset)
      train_iou_data += [train_iou]
      train_pixacc_data += [train_pixacc]
      valid_iou_data += [valid_iou]
      valid_pixacc_data += [valid_pixacc]
      train_loss_data += [loss_avg_train_val]
      #train_loss_data += [val_loss]
      valid_loss_data += [valid_loss]
      #print_best_result()
      if epoch_num > 1:
        #eval_helper.plot_accuracy(iou_fig, iou_acc_data)
        #eval_helper.plot_accuracy(pixel_fig, pixel_acc_data)
        print('Best IoU = ', max(valid_iou_data))
        eval_helper.plot_training_progress(
            os.path.join(FLAGS.train_dir, 'stats'), [train_loss_data, valid_loss_data],
            [train_iou_data, valid_iou_data], [train_pixacc_data, valid_pixacc_data])

      # Save the model checkpoint periodically.
      if FLAGS.save_net:
        if valid_iou >= max(valid_iou_data):
          print('Saving model...')
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          #saver.save(sess, checkpoint_path, global_step=epoch_num)
          saver.save(sess, checkpoint_path)

    #coord.request_stop()
    #coord.join(threads)
    sess.close()


def main(argv=None):  # pylint: disable=unused-argument
  model = helper.import_module('model', FLAGS.model_path)

  if tf.gfile.Exists(FLAGS.train_dir):
    raise ValueError('Train dir exists: ' + FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  stats_dir = os.path.join(FLAGS.train_dir, 'stats')
  tf.gfile.MakeDirs(stats_dir)
  tf.gfile.MakeDirs(FLAGS.debug_dir + '/train/')
  tf.gfile.MakeDirs(FLAGS.debug_dir + '/val/')
  f = open(os.path.join(stats_dir, 'log.txt'), 'w')
  sys.stdout = train_helper.Logger(sys.stdout, f)

  copyfile(FLAGS.model_path, os.path.join(FLAGS.train_dir, 'model.py'))
  copyfile(FLAGS.config_path, os.path.join(FLAGS.train_dir, 'config.py'))
  #subprocess.Popen(['tensorboard', '--logdir=' + g_save_dir])
  #subprocess.Popen('tensorboard --logdir=' + g_save_dir + ' > /dev/null', shell=True)

  print('Experiment dir: ' + FLAGS.train_dir)
  print('Dataset dir: ' + FLAGS.dataset_dir)
  train_dataset = CityscapesDataset(FLAGS.dataset_dir, 'train')
  valid_dataset = CityscapesDataset(FLAGS.dataset_dir, 'val')
  train(model, train_dataset, valid_dataset)


if __name__ == '__main__':
  tf.app.run()

