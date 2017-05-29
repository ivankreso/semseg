import os
import sys
import time
from datetime import datetime
from shutil import copyfile
import importlib.util
from os.path import join

import numpy as np
import tensorflow as tf
from tqdm import trange
import PIL.Image as pimg

import helper
import eval_helper
from datasets.voc2012.dataset import Dataset

np.set_printoptions(linewidth=250)

num_classes = 21
img_name = '2011_001620'
img_dir = '/home/kivan/datasets/VOC2012/JPEGImages/'
label_dir = '/home/kivan/datasets/voc2012_aug/data/'
#split = 'val'

#DATA_DIR = '/home/kivan/datasets/VOC2012/test_data'

tf.app.flags.DEFINE_string('model_dir',
    '/home/kivan/datasets/results/voc2012/iou77_24_5_22-39-18', '')
FLAGS = tf.app.flags.FLAGS


helper.import_module('config', os.path.join(FLAGS.model_dir, 'config.py'))


def forward_pass(model, save_dir):
  #file_path = join(DATA_DIR, 'ImageSets', 'Segmentation', 'test.txt')
  #fp = open(file_path)
  #file_list = [line.strip() for line in fp]

  save_dir_rgb = join(save_dir, 'rgb')
  tf.gfile.MakeDirs(save_dir_rgb)
  save_dir_pred = join(save_dir, 'pred')
  tf.gfile.MakeDirs(save_dir_pred)
  #sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
  config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
  #config.gpu_options.per_process_gpu_memory_fraction = 0.5 # don't hog all vRAM
  #config.operation_timeout_in_ms = 5000   # terminate on long hangs
  #config.operation_timeout_in_ms = 15000   # terminate on long hangs
  sess = tf.Session(config=config)
  # Get images and labels.
  #run_ops = model.inference()

  batch_shape = (1, None, None, 3)
  image_tf = tf.placeholder(tf.float32, shape=batch_shape)
  labels_tf = tf.placeholder(tf.int32, shape=(1, None, None, 1))
  logits, aux_logits, loss = model.inference(image_tf, labels_tf, constant_shape=False,
      is_training=False)
      #is_training=True)
  img_grads = tf.gradients(loss, image_tf)

  #sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  latest = os.path.join(FLAGS.model_dir, 'model.ckpt')
  restorer = tf.train.Saver(tf.global_variables())
  restorer.restore(sess, latest)

  img_path = join(img_dir, img_name + '.jpg')
  label_path = join(label_dir, img_name + '.png')
  image = np.array(pimg.open(img_path)).astype(np.float32)
  image = image[np.newaxis,...]
  labels = np.array(pimg.open(label_path)).astype(np.int8)
  labels[labels==-1] = num_classes
  #labels[labels==-1] = 20
  labels = labels[np.newaxis,...,np.newaxis]

  while True:
    loss_val, logits_val, img_grads_val = sess.run([loss, logits, img_grads],
        feed_dict={image_tf:image, labels_tf:labels})
    print('loss = ', loss_val)
    img_grads_val = img_grads_val[0]
    print('grad norm = ', np.linalg.norm(img_grads_val))
    pred_labels = logits_val[0].argmax(2).astype(np.int32)
    save_path = os.path.join(save_dir_pred, img_name + '.png')
    eval_helper.draw_output(pred_labels, Dataset.class_info, save_path)

    labels_2d = labels[0,:,:,0]
    pred_labels[pred_labels != labels_2d] = -1
    #pred_labels[labels == num_classes] = -1
    num_correct = (pred_labels >= 0).sum()
    num_labels = np.sum(labels_2d != num_classes)
    image += 1e4 * img_grads_val
    #image += np.sign(img_grads_val)
    save_img = np.minimum(255, np.round(image[0]))
    save_img = np.maximum(0, save_img)
    save_img = save_img.astype(np.uint8)
    pil_img = pimg.fromarray(save_img)
    pil_img.save(join(save_dir_rgb, img_name + '.png'))
  #  pred_img = pimg.fromarray(pred_labels)
  #  pred_img.save(join(save_dir_submit, file_list[i] + '.png'))

    print('pixel accuracy = ', num_correct / num_labels * 100)
    print('press key...')
    input()

  #  #pred_labels = logits_val[0].argmax(2).astype(np.int32)
  #  pred_labels = logits_val[0].argmax(2).astype(np.uint8)
  #  save_path = os.path.join(save_dir_rgb, file_list[i] + '.png')
  #  eval_helper.draw_output(pred_labels, Dataset.class_info, save_path)
  #  pred_img = pimg.fromarray(pred_labels)
  #  pred_img.save(join(save_dir_submit, file_list[i] + '.png'))

    ##gt_labels = gt_labels.astype(np.int32, copy=False)
    #cylib.collect_confusion_matrix(net_labels.reshape(-1), gt_labels.reshape(-1), conf_mat)
    #gt_labels = gt_labels.reshape(net_labels.shape)
    #pred_labels = np.copy(net_labels)
    #net_labels[net_labels == gt_labels] = -1
    #net_labels[gt_labels == -1] = -1
    #num_mistakes = (net_labels >= 0).sum()
    #img_prefix = '%07d_'%num_mistakes + img_prefix

    #error_save_path = os.path.join(save_dir, str(loss_val) + img_prefix + '_errors.png')
    #filename =  img_prefix + '_' + str(loss_val) + '_error.png'
    #error_save_path = os.path.join(save_dir, filename)
    #eval_helper.draw_output(net_labels, CityscapesDataset.CLASS_INFO, error_save_path)
    #print(q_size)
  #print(conf_mat)
  #img_names = [[x,y] for (y,x) in sorted(zip(loss_vals, img_names))]
  #sorted_data = [x for x in sorted(zip(loss_vals, img_names), reverse=True)]
  #print(img_names)
  #for i, elem in enumerate(sorted_data):
  #  print('Xent loss = ', elem[0])
  #  ski.io.imshow(os.path.join(save_dir, elem[1] + '_errors.png'))
  #  ski.io.show()

  #print('')
  #pixel_acc, iou_acc, recall, precision, _ = eval_helper.compute_errors(
  #    conf_mat, 'Validation', CityscapesDataset.CLASS_INFO, verbose=True)
  sess.close()


def main(argv=None):  # pylint: disable=unused-argument
  model = helper.import_module('model', os.path.join(FLAGS.model_dir, 'model.py'))

  if not tf.gfile.Exists(FLAGS.model_dir):
    raise ValueError('Net dir not found: ' + FLAGS.model_dir)
  save_dir = os.path.join(FLAGS.model_dir, 'evaluation', 'adversarial')
  tf.gfile.MakeDirs(save_dir)

  forward_pass(model, save_dir)


if __name__ == '__main__':
  tf.app.run()

