import os
from os.path import join
import sys
import time
from shutil import copyfile

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import helper
import train_helper
from datasets.cityscapes.cityscapes import CityscapesDataset

np.set_printoptions(linewidth=250)

tf.app.flags.DEFINE_string('config_path', '', """Path to experiment config.""")
FLAGS = tf.app.flags.FLAGS

helper.import_module('config', FLAGS.config_path)
print(FLAGS.config_path)


def train_epoch(epoch_num, model, train_dataset, train_size,
                valid_dataset, valid_size, restore_path=None):
  config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
  #config.operation_timeout_in_ms = 5000   # terminate on long hangs
  #sess = tf.Session(config=config)
  with tf.Session(config=config) as sess:
    # Build a Graph that computes the logits predictions from the inference model.
    train_ops, init_op, init_feed = model.build(train_dataset, train_size, is_training=True)
    num_params = train_helper.get_num_params()
    vars_to_restore = tf.contrib.framework.get_variables_to_restore()

    if valid_dataset != None:
      valid_ops = model.build(valid_dataset, valid_size, is_training=False, reuse=True)
    loss = train_ops[0]

    num_batches = model.num_batches(train_dataset)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                  trainable=False)
    train_op = model.minimize(loss, global_step, num_batches)

    print('Total number of parameters = ', num_params)

    #grads = opt.compute_gradients(loss)
    #train_op = opt.apply_gradients(grads, global_step=global_step)

    # Create a saver.
    #saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_epochs)
    #saver = tf.train.Saver(sess, max_to_keep=FLAGS.max_epochs)
    saver = tf.train.Saver()
    #saver = tf.train.Saver(tf.all_variables())

    sess.run(tf.global_variables_initializer())
    if restore_path is not None:
      print(f'\nRestoring params from: {FLAGS.resume_path}\n')
      #print(tf.train.latest_checkpoint(FLAGS.resume_path))
      #assert tf.gfile.Exists(FLAGS.resume_path)
      #resnet_restore = tf.train.Saver(vars_to_restore)
      #resnet_restore.restore(sess, restore_path)
      saver.restore(sess, restore_path)
    elif init_op != None:
      print('\nInitializing from pretrained weights...')
      sess.run(init_op, feed_dict=init_feed)
    else:
      print('All params are using random init')
    sess.run(tf.local_variables_initializer())

    # Build the summary operation based on the TF collection of Summaries.
    #summary_op = tf.merge_all_summaries()
    summary_op = tf.summary.merge_all()

    #ops = sess.graph.get_operations()
    #for op in ops:
    #  print(op.name, op.values())

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

    #init_vars = train_helper.get_variables(sess)
    #train_helper.print_variable_diff(sess, init_vars)

    #variable_map = train_helper.get_variable_map()
    # take the train loss moving average
    #loss_avg_train = variable_map['total_loss/avg:0']
    train_loss_val = 0
    train_data, valid_data = model.get_eval_data()
    ex_start_time = time.time()
    print('\nnvim ' + FLAGS.train_dir + 'model.py')
    print('tensorboard --logdir=' + FLAGS.train_dir + '\n')
    #num_batches = model.num_batches(train_dataset) // FLAGS.num_validations_per_epoch
    #num_batches = model.num_batches(train_dataset)
    model.start_epoch(train_data)
    #for step in range(40):
    #for step in range(0):
    for step in range(num_batches):
      start_time = time.time()
      run_ops = train_ops + [train_op, global_step]
      #run_ops = [train_op, loss, logits, labels, draw_data, img_name, global_step]
      if False:
      #if step == 0:
        run_ops.append(summary_op)
        ret_val = model.train_step(sess, run_ops)
        loss_val = ret_val[0]
        summary_str = ret_val[-1]
        global_step_val = ret_val[-2]
        summary_writer.add_summary(summary_str, global_step_val)
      else:
        #ret_val = sess.run(run_ops, feed_dict=feed_dict)
        ret_val = model.train_step(sess, run_ops)
        loss_val = ret_val[0]
        #if step % 100 == 0:
        #  model.evaluate_output(ret_val, step)
        #train_helper.print_grad_stats(grads_val, grad_tensors)
        #if step > 20:
        #  run_metadata = tf.RunMetadata()
        #  ret_val = sess.run(run_ops, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #                     run_metadata=run_metadata)
        #  trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        #  trace_file = open('timeline.ctf.json', 'w')
        #  trace_file.write(trace.generate_chrome_trace_format())
        #  raise 1
      duration = time.time() - start_time

      assert not np.isnan(loss_val), 'Model diverged with loss = NaN'
      # estimate training accuracy on the last 40% of the epoch
      #if step > int(0.5 * num_batches):
      model.update_stats(ret_val)

      #img_prefix = img_prefix[0].decode("utf-8")

      #if FLAGS.draw_predictions and step % 50 == 0:
      #  model.draw_prediction('train', epoch_num, step, ret_val)

      #if step % 50 == 0:
      #if step % 1 == 0:
      if step % 30 == 0:
        examples_per_sec = FLAGS.batch_size / duration
        sec_per_batch = float(duration)

        format_str = '%s: epoch %d, step %d / %d, loss = %.2f \
          (%.1f examples/sec; %.3f sec/batch)'
        #print('lr = ', clr)
        print(format_str % (train_helper.get_expired_time(ex_start_time), epoch_num,
                            step, model.num_batches(train_dataset), loss_val,
                            examples_per_sec, sec_per_batch))
    is_best = model.end_epoch(train_data)
    #train_helper.print_variable_diff(sess, init_vars)
    if valid_dataset != None:
      is_best = model.evaluate('valid', sess, epoch_num, valid_ops, len(valid_dataset), valid_data)
    model.print_results(train_data, valid_data)

    #valid_loss, valid_pixacc, valid_iou, valid_recall, valid_precision = evaluate(
    #  sess, epoch_num, logits_valid, loss_valid,
    #  labels_valid, img_name_valid, valid_dataset, reader)

    model.plot_results(train_data, valid_data)
    #  eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats'), plot_data)

    # Save the best model checkpoint
    #if FLAGS.save_net and is_best:
    #  print('Saving model...')
    #  checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    #  #saver.save(sess, checkpoint_path, global_step=epoch_num)
    #  saver.save(sess, checkpoint_path)
    print('Saving model...')
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    #saver.save(sess, checkpoint_path, global_step=epoch_num)
    save_path = saver.save(sess, checkpoint_path)

    coord.request_stop()
    coord.join(threads)
    #sess.close()
    return save_path, num_batches


def shuffle(files):
  perm = np.random.permutation(len(files))
  shuffled = []
  for i in range(perm.shape[0]):
    shuffled.append(files[perm[i]])
  return shuffled


def get_dir_files(dir_path):
  files = next(os.walk(subset_dir))[2]


def filter_filenames(filenames, strings):
  filtered = []
  for f in filenames:
    for s in strings:
      if f.find(s) >= 0:
        break
    else:
      continue
    filtered.append(f)
  return filtered


def _create_jitter_filenames():
  #jitter_data_dir = '/home/kivan/datasets/Cityscapes/crops/crops170314classres'
  jitter_data_dir = '/home/kivan/datasets/Cityscapes/tensorflow/jitter'
  filenames = next(os.walk(jitter_data_dir))[2]
  #classes = ['wall', 'truck', 'bus', 'terrain']
  #filenames = filter_filenames(filenames, classes)
  filenames = [os.path.join(jitter_data_dir, f) for f in filenames]
  all_filenames = []
  #sizes = [[128,128], [256,256], [512,512], [1024,1024]]
  #sizes = [[128,128], [256,256], [512,512]]
  #batch_sizes = [24, 16, 8]
  sizes = [[512,512]]
  batch_sizes = [8]
  for s in sizes:
    s = f'{s[0]}x{s[1]}'
    arr = [f for f in filenames if f.find(s) >= 0]
    all_filenames.append(arr)
    print(all_filenames)
  return all_filenames, sizes, batch_sizes


def train(model, train_dataset, valid_dataset):
  train_files = train_dataset.get_filenames()
  valid_files = valid_dataset.get_filenames()
  train_queues = [train_files]
  sizes = [[FLAGS.img_height, FLAGS.img_width]]
  batch_sizes = [FLAGS.batch_size]
  jitter_lists, jitter_sizes, jitter_batch_sizes = _create_jitter_filenames()
  train_queues.extend(jitter_lists)
  sizes.extend(jitter_sizes)
  batch_sizes.extend(jitter_batch_sizes)

  save_path = None
  #save_path = '/home/kivan/datasets/results/cityscapes/75.6_5_4_23-38-05/model.ckpt'
  iter_num = 0
  for epoch_num in range(1, FLAGS.max_epochs + 1):
    shuffled_valid = None
    for i in range(len(train_queues)):
      print('Training size = ', sizes[i])
      shuffled_train = shuffle(train_queues[i])
      FLAGS.batch_size = batch_sizes[i]
      if i == len(train_queues)-1:
        shuffled_valid = shuffle(valid_files)
      #print(len(shuffled_train))
      #continue
      save_path, num_iters = train_epoch(epoch_num, model, shuffled_train, sizes[i],
                              shuffled_valid, sizes[0], restore_path=save_path)
      iter_num += num_iters
      print('ITER_NUM = ', iter_num)
      tf.reset_default_graph()
    if iter_num > FLAGS.num_iters:
      break


def main(argv=None):  # pylint: disable=unused-argument
  model = helper.import_module('model', FLAGS.model_path)

  if tf.gfile.Exists(FLAGS.train_dir):
    raise ValueError('Train dir exists: ' + FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  stats_dir = join(FLAGS.train_dir, 'stats')
  tf.gfile.MakeDirs(stats_dir)
  tf.gfile.MakeDirs(join(FLAGS.debug_dir, 'train'))
  tf.gfile.MakeDirs(join(FLAGS.debug_dir, 'valid'))
  tf.gfile.MakeDirs(join(FLAGS.train_dir, 'results'))
  f = open(join(stats_dir, 'log.txt'), 'w')
  sys.stdout = train_helper.Logger(sys.stdout, f)

  copyfile(FLAGS.model_path, os.path.join(FLAGS.train_dir, 'model.py'))
  copyfile(FLAGS.config_path, os.path.join(FLAGS.train_dir, 'config.py'))

  print('Experiment dir: ' + FLAGS.train_dir)
  print('Dataset dir: ' + FLAGS.dataset_dir)
  train_dataset = CityscapesDataset(FLAGS.dataset_dir, ['train'])
  valid_dataset = CityscapesDataset(FLAGS.dataset_dir, ['val'])
  train(model, train_dataset, valid_dataset)
  #train_dataset = CityscapesDataset(FLAGS.dataset_dir, ['train', 'val'])
  #train(model, train_dataset, None)


if __name__ == '__main__':
  tf.app.run()

