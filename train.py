import os
import sys
import time
from shutil import copyfile

import numpy as np
import tensorflow as tf

import libs.cylib as cylib
import helper
import train_helper
from datasets.cityscapes.cityscapes import CityscapesDataset

np.set_printoptions(linewidth=250)

tf.app.flags.DEFINE_string('config_path', '', """Path to experiment config.""")
FLAGS = tf.app.flags.FLAGS

helper.import_module('config', FLAGS.config_path)
print(FLAGS.config_path)


def train(model, train_dataset, valid_dataset):
  """ Trains the network
    Args:
      model: module containing model architecture
      train_dataset: training data object
      valid_dataset: validation data object
  """
  with tf.Graph().as_default():
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    #config.operation_timeout_in_ms = 5000   # terminate on long hangs
    sess = tf.Session(config=config)
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (model.num_examples(train_dataset) / FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step, decay_steps,
                                    FLAGS.learning_rate_decay_factor, staircase=True)

    # Build a Graph that computes the logits predictions from the inference model.
    train_ops, init_op, init_feed = model.build(train_dataset, is_training=True)
    valid_ops = model.build(valid_dataset, is_training=False, reuse=True)
    loss = train_ops[0]

    # Add a summary to track the learning rate.
    tf.scalar_summary('learning_rate', lr)

    print('Using optimizer:', FLAGS.optimizer)
    if FLAGS.optimizer == 'Adam':
      opt = tf.train.AdamOptimizer(lr)
    elif FLAGS.optimizer == 'Momentum':
      opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
      #opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'RMSprop':
      opt = tf.train.RMSPropOptimizer(lr)
    else:
      raise ValueError()

    train_op = model.minimize(opt, loss, global_step)

    #grads = opt.compute_gradients(loss)
    #train_op = opt.apply_gradients(grads, global_step=global_step)



    # Add histograms for trainable variables.
    #for var in tf.trainable_variables():
    #  tf.histogram_summary(var.op.name, var)
    ## Add histograms for gradients.
    #grad_tensors = []
    #for grad, var in grads:
    #  grad_tensors += [grad]
    #  #print(var)
    #  if grad is not None:
    #    tf.histogram_summary(var.op.name + '/gradients', grad)
    ##grad = grads[-2][0]
    ##print(grad)

    # Track the moving averages of all trainable variables.
    #variable_averages = tf.train.ExponentialMovingAverage(
    #    FLAGS.moving_average_decay, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())


    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_epochs)
    #saver = tf.train.Saver(tf.all_variables())

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    if init_op != None:
      sess.run(init_op, feed_dict=init_feed)

    if len(FLAGS.resume_path) > 0:
      print('\nRestoring params from:', FLAGS.resume_path)
      assert tf.gfile.Exists(FLAGS.resume_path)
      resnet_restore = tf.train.Saver(model.variables_to_restore())
      resnet_restore.restore(sess, FLAGS.resume_path)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

    init_vars = train_helper.get_variables(sess)
    train_helper.print_variable_diff(sess, init_vars)
    #variable_map = train_helper.get_variable_map()
    num_params = train_helper.get_num_params()
    print('Number of parameters = ', num_params)
    # take the train loss moving average
    #loss_avg_train = variable_map['total_loss/avg:0']
    train_data, valid_data = model.init_eval_data()
    ex_start_time = time.time()
    for epoch_num in range(1, FLAGS.max_epochs + 1):
      print('\ntensorboard --logdir=' + FLAGS.train_dir + '\n')
      train_data['lr'] += [lr.eval(session=sess)]
      num_batches = model.num_examples(train_dataset) // FLAGS.num_validations_per_epoch
      for step in range(num_batches):
      #for step in range(100):
        start_time = time.time()
        run_ops = train_ops + [train_op, global_step]
        #run_ops = [train_op, loss, logits, labels, draw_data, img_name, global_step]
        if step % 300 == 0:
          #run_ops += [summary_op, loss_avg_train]
          run_ops += [summary_op]
          ret_val = sess.run(run_ops)
          loss_val = ret_val[0]
          summary_str = ret_val[-1]
          global_step_val = ret_val[-2]
          summary_writer.add_summary(summary_str, global_step_val)
        else:
          ret_val = sess.run(run_ops)
          loss_val = ret_val[0]
          #train_helper.print_grad_stats(grads_val, grad_tensors)
          #run_metadata = tf.RunMetadata()
          #ret_val = sess.run(run_ops,
          #            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
          #            run_metadata=run_metadata)
          #(_, loss_val, scores, yt, draw_data_val, img_prefix, global_step_val) = ret_val
          #if step > 10:
          #  trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          #  trace_file = open('timeline.ctf.json', 'w')
          #  trace_file.write(trace.generate_chrome_trace_format())
          #  raise 1
        duration = time.time() - start_time

        assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

        ## estimate training accuracy on the last 30% of the epoch
        #if step > int(0.7 * num_batches):
        #  #label_map = scores[0].argmax(2).astype(np.int32)
        #  label_map = scores.argmax(3).astype(np.int32)
        #  #print(label_map.shape)
        #  #print(yt.shape)
        #  cylib.collect_confusion_matrix(label_map.reshape(-1),
        #                                 yt.reshape(-1), conf_mat)
        #img_prefix = img_prefix[0].decode("utf-8")

        #if FLAGS.draw_predictions and step % 50 == 0:
        #  model.draw_prediction('train', epoch_num, step, ret_val)

        if step % 20 == 0:
          examples_per_sec = FLAGS.batch_size / duration
          sec_per_batch = float(duration)

          format_str = '%s: epoch %d, step %d / %d, loss = %.2f \
            (%.1f examples/sec; %.3f sec/batch)'
          #print('lr = ', clr)
          print(format_str % (train_helper.get_expired_time(ex_start_time), epoch_num,
                              step, model.num_examples(train_dataset), loss_val,
                              examples_per_sec, sec_per_batch))
      train_helper.print_variable_diff(sess, init_vars)
      model.evaluate('valid', sess, epoch_num, valid_ops, valid_dataset, valid_data)
      model.print_results(valid_data)
      #model.plot_results(train_data, valid_data)

      #train_pixacc, train_iou, _, _, _ = eval_helper.compute_errors(conf_mat, 'Train',
      #    CityscapesDataset.CLASS_INFO)

      #valid_loss, valid_pixacc, valid_iou, valid_recall, valid_precision = evaluate(
      #  sess, epoch_num, logits_valid, loss_valid,
      #  labels_valid, img_name_valid, valid_dataset, reader)

      ##print_best_result()
      #if valid_iou >= max_valid_iou:
      #  max_valid_iou = valid_iou
      #if epoch_num > 1:
      #  print('Best IoU = ', max_valid_iou)
      #  eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats'), plot_data)

      ## Save the best model checkpoint
      #if FLAGS.save_net:
      #  if valid_iou >= max_valid_iou:
      #    print('Saving model...')
      #    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      #    #saver.save(sess, checkpoint_path, global_step=epoch_num)
      #    saver.save(sess, checkpoint_path)

    coord.request_stop()
    coord.join(threads)
    sess.close()


def main(argv=None):  # pylint: disable=unused-argument
  model = helper.import_module('model', FLAGS.model_path)

  if tf.gfile.Exists(FLAGS.train_dir):
    raise ValueError('Train dir exists: ' + FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  stats_dir = os.path.join(FLAGS.train_dir, 'stats')
  tf.gfile.MakeDirs(stats_dir)
  tf.gfile.MakeDirs(FLAGS.debug_dir + '/train/')
  tf.gfile.MakeDirs(FLAGS.debug_dir + '/valid/')
  f = open(os.path.join(stats_dir, 'log.txt'), 'w')
  sys.stdout = train_helper.Logger(sys.stdout, f)

  copyfile(FLAGS.model_path, os.path.join(FLAGS.train_dir, 'model.py'))
  copyfile(FLAGS.config_path, os.path.join(FLAGS.train_dir, 'config.py'))

  print('Experiment dir: ' + FLAGS.train_dir)
  print('Dataset dir: ' + FLAGS.dataset_dir)
  train_dataset = CityscapesDataset(FLAGS.dataset_dir, 'train')
  valid_dataset = CityscapesDataset(FLAGS.dataset_dir, 'val')
  train(model, train_dataset, valid_dataset)


if __name__ == '__main__':
  tf.app.run()

