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

num_classes = 19
#split = 'val'

#DATA_DIR = '/home/kivan/datasets/VOC2012/test_data'

tf.app.flags.DEFINE_string('model_dir',
    #'/home/kivan/datasets/results/tmp/cityscapes/14_6_14-11-57/', '')
    '/home/kivan/datasets/results/tmp/cityscapes/26_6_12-09-41/', '')
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

  #batch_shape = (1, None, None, 3)
  #batch_shape = (1, 1024, 2048, 3)
  batch_shape = (1, 448, 1024, 3)
  image_pl = tf.placeholder(tf.float32, shape=batch_shape)
  labels_pl = tf.placeholder(tf.int32, shape=(1, None, None, 1))

  logits, aux_logits, embed = model.inference(image_pl, is_training=False)
      #is_training=True)

  embed_shape = embed.get_shape().as_list()
  print('embed shape = ', embed_shape)
  grad_embed_pl = tf.placeholder(tf.float32, shape=embed_shape)
  grad_embed_np = np.zeros(embed_shape, dtype=np.float32)
  image_np = np.zeros(batch_shape, dtype=np.float32)
  #grad_embed_np[:,:,128,256] = 1
  grad_embed_np[:,:,embed_shape[2]//2-1,embed_shape[3]//2-1] = 1

  #sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  latest = os.path.join(FLAGS.model_dir, 'model.ckpt')
  restorer = tf.train.Saver(tf.global_variables())
  restorer.restore(sess, latest)

  #img_path = join(img_dir, img_name + '.jpg')
  #label_path = join(label_dir, img_name + '.png')
  #image = np.array(pimg.open(img_path)).astype(np.float32)
  #image = image[np.newaxis,...]
  #labels = np.array(pimg.open(label_path)).astype(np.int8)
  #labels[labels==-1] = num_classes
  ##labels[labels==-1] = 20
  #labels = labels[np.newaxis,...,np.newaxis]

  grad_img = tf.gradients(embed, image_pl, grad_embed_pl)
  grad_img_val = sess.run(grad_img,
      feed_dict={grad_embed_pl: grad_embed_np, image_pl: image_np})
  grad = grad_img_val[0][0]
  print(grad.shape)
  grad = np.abs(grad)
  gmin = grad.min()
  gmax = grad.max()
  scale = gmax - gmin
  grad = (grad - gmin) / scale
  print(grad.min())
  print(grad.max())
  save_img = (grad*255).astype(np.uint8)
  pil_img = pimg.fromarray(save_img)
  save_dir = '/home/kivan/datasets/tmp/out/'
  pil_img.save(join(save_dir, 'test.png'))
  raise 1

  while True:
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

