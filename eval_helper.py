import os
import time
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io
import cv2
import libs.cylib as cylib

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

from datasets.cityscapes.cityscapes import CityscapesDataset

def evaluate_segmentation(sess, epoch_num, run_ops, num_examples):
  print('\nValidation performance:')
  conf_mat = np.ascontiguousarray(
      np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64))
  loss_avg = 0
  for step in range(num_examples):
    start_time = time.time()
    loss_val, logits, labels, img_prefix = sess.run(run_ops)
    duration = time.time() - start_time
    loss_avg += loss_val
    #net_labels = out_logits[0].argmax(2).astype(np.int32, copy=False)
    #net_labels = logits[0].argmax(2).astype(np.int32)
    net_labels = logits.argmax(3).astype(np.int32)
    #gt_labels = gt_labels.astype(np.int32, copy=False)
    cylib.collect_confusion_matrix(net_labels.reshape(-1),
                                   labels.reshape(-1), conf_mat)

    if step % 10 == 0:
      num_examples_per_step = FLAGS.batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)
      format_str = 'epoch %d, step %d / %d, loss = %.2f \
        (%.1f examples/sec; %.3f sec/batch)'
      #print('lr = ', clr)
      print(format_str % (epoch_num, step, num_examples, loss_val,
                          examples_per_sec, sec_per_batch))
    if FLAGS.draw_predictions and step % 100 == 0:
      img_prefix = img_prefix[0].decode("utf-8")
      save_path = FLAGS.debug_dir + '/val/' + '%03d_' % epoch_num + img_prefix + '.png'
      draw_output(net_labels, CityscapesDataset.CLASS_INFO, save_path)
  #print(conf_mat)
  print('')
  pixel_acc, iou_acc, recall, precision, _ = compute_errors(
      conf_mat, 'Validation', CityscapesDataset.CLASS_INFO, verbose=True)
  return loss_avg / num_examples, pixel_acc, iou_acc, recall, precision

def evaluate_depth_prediction(name, sess, epoch_num, run_ops, num_examples):
  print('\nValidation performance:')
  loss_avg = 0
  N = num_examples
  for step in range(N):
    start_time = time.time()
    loss_val, yp, yt, x, img_names = sess.run(run_ops)
    duration = time.time() - start_time
    loss_avg += loss_val
    if step % 20 == 0:
      num_examples_per_step = FLAGS.batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)
      format_str = 'epoch %d, step %d / %d, loss = %.2f \
        (%.1f examples/sec; %.3f sec/batch)'
      print(format_str % (epoch_num, step, N, loss_val,
                          examples_per_sec, sec_per_batch))
    #print(yp)
    if FLAGS.draw_predictions and step % 10 == 0:
      draw_depth_prediction(name, epoch_num, step, yp, yt, img_names)

    #  eval_helper.draw_output(net_labels, CityscapesDataset.CLASS_INFO, save_path)
  loss_avg /= N
  print('Loss = ', loss_avg)
  return loss_avg


def draw_depth_prediction(name, epoch_num, step, yp, yt, img_names):
  yp[yp<0] = 0
  yp[yp>255] = 255
  yp = yp.astype(np.uint8)
  yt = yt.reshape(yp.shape).astype(np.uint8)
  for i in range(len(img_names)):
    img_prefix = img_names[i].decode("utf-8")
    #save_path = FLAGS.debug_dir + '/val/' + '%03d_' % epoch_num + img_prefix + '.png'
    save_path = os.path.join(FLAGS.debug_dir, name,
        '%03d_%06d_' % (epoch_num, step) + img_prefix + '_pred.png')
    cv2.imwrite(save_path, yp[i])
    save_path = os.path.join(FLAGS.debug_dir, name,
        '%03d_%06d_' % (epoch_num, step) + img_prefix + '_gt.png')
    cv2.imwrite(save_path, yt[i])


def draw_output(y, class_colors, save_path):
  width = y.shape[1]
  height = y.shape[0]
  y_rgb = np.zeros((height, width, 3), dtype=np.uint8)
  for cid in range(len(class_colors)):
    cpos = np.repeat((y == cid).reshape((height, width, 1)), 3, axis=2)
    cnum = cpos.sum() // 3
    y_rgb[cpos] = np.array(class_colors[cid][:3] * cnum, dtype=np.uint8)
    #pixels = y_rgb[[np.repeat(np.equal(y, cid).reshape((height, width, 1)), 3, axis=2)]
    #if pixels.size > 0:
    #  #pixels.reshape((-1, 3))[:,:] = class_colors[cid][:3]
    #  #pixels.resize((int(pixels.size/3), 3))
    #  print(np.array(class_colors[cid][:3] * (pixels.size // 3), dtype=np.uint8))
    #  pixels = np.array(class_colors[cid][:3] * (pixels.size // 3), dtype=np.uint8)
    #y_rgb[np.repeat(np.equal(y, cid).reshape((height, width, 1)), 3, axis=2)].reshape((-1, 3)) = \
    #    class_colors[cid][:3]
  ski.io.imsave(save_path, y_rgb)


def draw_prediction_slow(y, colors, path):
  width = y.shape[1]
  height = y.shape[0]
  col = np.zeros(3)
  yimg = np.empty((height, width, 3), dtype=np.uint8)
  for i in range(height):
    for j in range(width):
      cid = y[i,j]
      for k in range(3):
        yimg[i,j,k] = colors[cid][k]
      #img[i,j,:] = col
  #print(yimg.shape)
  #yimg = ski.transform.resize(yimg, (height*16, width*16), order=0, preserve_range=True).astype(np.uint8)
  ski.io.imsave(path, yimg)


def collect_confusion_matrix(y, yt, conf_mat):
  for i in range(y.size):
    l = y[i]
    lt = yt[i]
    if lt >= 0:
      conf_mat[l,lt] += 1


def compute_errors(conf_mat, name, class_info, verbose=True):
  num_correct = conf_mat.trace()
  num_classes = conf_mat.shape[0]
  total_size = conf_mat.sum()
  avg_pixel_acc = num_correct / total_size * 100.0
  TPFN = conf_mat.sum(0)
  TPFP = conf_mat.sum(1)
  FN = TPFN - conf_mat.diagonal()
  FP = TPFP - conf_mat.diagonal()
  class_iou = np.zeros(num_classes)
  class_recall = np.zeros(num_classes)
  class_precision = np.zeros(num_classes)
  if verbose:
    print(name + ' errors:')
  for i in range(num_classes):
    TP = conf_mat[i,i]
    class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
    if TPFN[i] > 0:
      class_recall[i] = (TP / TPFN[i]) * 100.0
    else:
      class_recall[i] = 0
    if TPFP[i] > 0:
      class_precision[i] = (TP / TPFP[i]) * 100.0
    else:
      class_precision[i] = 0

    class_name = class_info[i][3]
    if verbose:
      print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))
  avg_class_iou = class_iou.mean()
  avg_class_recall = class_recall.mean()
  avg_class_precision = class_precision.mean()
  if verbose:
    print(name + ' IoU mean class accuracy - TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
    print(name + ' mean class recall - TP / (TP+FN) = %.2f %%' % avg_class_recall)
    print(name + ' mean class precision - TP / (TP+FP) = %.2f %%' % avg_class_precision)
    print(name + ' pixel accuracy = %.2f %%' % avg_pixel_acc)
  return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size

def plot_training_progress(save_dir, plot_data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 6
  title_size = 10
  train_color = 'm'
  val_color = 'c'

  train_loss = plot_data['train_loss']
  valid_loss = plot_data['valid_loss']
  train_iou = plot_data['train_iou']
  valid_iou = plot_data['valid_iou']
  train_acc = plot_data['train_acc']
  valid_acc = plot_data['valid_acc']
  lr = plot_data['lr']
  x_data = np.linspace(1, len(train_loss), len(train_loss))
  ax1.set_title('cross entropy loss', fontsize=title_size)
  ax1.plot(x_data, train_loss, marker='o', color=train_color, linewidth=linewidth, linestyle='-', \
      label='train')
  ax1.plot(x_data, valid_loss, marker='o', color=val_color, linewidth=linewidth, linestyle='-',
      label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('IoU accuracy')
  ax2.plot(x_data, train_iou, marker='o', color=train_color, linewidth=linewidth, linestyle='-',
      label='train')
  ax2.plot(x_data, valid_iou, marker='o', color=val_color, linewidth=linewidth, linestyle='-',
      label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('pixel accuracy')
  ax3.plot(x_data, train_acc, marker='o', color=train_color, linewidth=linewidth, linestyle='-',
      label='train')
  ax3.plot(x_data, valid_acc, marker='o', color=val_color, linewidth=linewidth, linestyle='-',
      label='validation')
  ax3.legend(loc='upper left', fontsize=legend_size)

  ax4.set_title('learning rate')
  ax4.plot(x_data, lr, marker='o', color=train_color, linewidth=linewidth,
           linestyle='-', label='learning rate')
  ax4.legend(loc='upper right', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.pdf')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


def map_cityscapes_to_kitti(y, id_map):
  y_kitti = np.zeros(y.shape, dtype=y.dtype)
  for i in range(len(id_map)):
    #print(i , ' --> ', id_map[i])
    #print(np.equal(y, i).sum(), '\n')
    #print('sum')
    #print(y[np.equal(y, i)].sum())
    y_kitti[np.equal(y, i)] = id_map[i]
    #print(np.equal(y, id_map[i]).sum())
    #y[np.equal(y, i)] = 2
    #print(y[np.equal(y, i)].sum())
  return y_kitti


#def plot_accuracy(fig, data):
#  x_data = np.linspace(1, len(data), len(data))
#  plt.figure(fig.number)
#  plt.clf()
#  plt.plot(x_data, data, 'b-')
#  plt.savefig(str(fig.number) + '_plot.pdf', bbox_inches='tight')
