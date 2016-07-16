import os
import pickle

import tensorflow as tf
import numpy as np
import skimage as ski
import skimage.data, skimage.transform
#import vgg_16s_baseline as model
import importlib.util
from tqdm import trange
import skimage as ski
import skimage.data, skimage.transform

import losses
import eval_helper
import libs.cylib as cylib

#from datasets.cityscapes.cityscapes import CityscapesDataset as Dataset
import datasets.kitti.kitti_info as Dataset
import datasets.reader as reader

#tf.app.flags.DEFINE_integer('img_width', 1024, '')
#tf.app.flags.DEFINE_integer('img_height', 432, '')

tf.app.flags.DEFINE_integer('img_width', 1216, '')
tf.app.flags.DEFINE_integer('img_height', 384, '')
#tf.app.flags.DEFINE_integer('img_width', 1920, '')
#tf.app.flags.DEFINE_integer('img_height', 608, '')
DATA_NAME = 'valid'
#DATA_NAME = 'train'
DATA_DIR = '/home/kivan/datasets/KITTI/semantic_segmentation/' + DATA_NAME + '/data/rgb/'
SAVE_DIR = '/home/kivan/datasets/KITTI/output/kitti_semseg/'

#tf.app.flags.DEFINE_integer('img_width', 2048, '')
#tf.app.flags.DEFINE_integer('img_height', 1536, '')
#DATA_DIR = '/home/kivan/datasets/play/'
#SAVE_DIR = os.path.join(DATA_DIR, 'inference')

tf.app.flags.DEFINE_integer('img_depth', 3, '')
tf.app.flags.DEFINE_integer('batch_size', 1, '')

tf.app.flags.DEFINE_string('dataset_dir', DATA_DIR, '')
    #'/home/kivan/datasets/KITTI/semantic_segmentation/train/data/rgb/', '')
    #'/home/kivan/datasets/KITTI/sequences_color/07/image_2/', '')
tf.app.flags.DEFINE_integer('num_classes', 19, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('is_training', False, '')
tf.app.flags.DEFINE_string('save_dir', SAVE_DIR, '')
tf.app.flags.DEFINE_string('gt_dir',
  '/home/kivan/datasets/KITTI/semantic_segmentation/gt_data/' + DATA_NAME + '/', '')
  #'/home/kivan/datasets/KITTI/semantic_segmentation/gt_data/train/', '')

FLAGS = tf.app.flags.FLAGS

#NET_DIR = '/home/kivan/source/results/semseg/tf/nets/8_5_10-40-27/'
#NET_DIR = '/home/kivan/source/results/semseg/tf/saved/dilated_full66_30_5_15-19-36/'
NET_DIR = '/home/kivan/source/results/semseg/tf/nets/7_6_15-08-57/'
#NET_DIR = '/home/kivan/source/results/semseg/tf/saved/61.5_640_3_6_10-09-12/'
MODEL_PATH = NET_DIR + 'model.py'

VGG_MEAN = [123.68, 116.779, 103.939]



#def evaluate_model(sess, dataset, logits, labels, img_name):
#  conf_mat = np.ascontiguousarray(np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64))
#  print(dataset.num_examples())
#  for i in trange(dataset.num_examples()):
#    #_, loss_value = sess.run([train_op, loss])
#    out_logits, yt, filename = sess.run([logits, labels, img_name])
#    y = out_logits[0].argmax(2).astype(np.int32, copy=False)
#    draw_prediction(y, class_info,  FLAGS.save_dir + filename[0].decode("utf-8") + '.png')
#    yt = yt.astype(np.int32, copy=False)
#    cylib.collect_confusion_matrix(y.reshape(-1), yt, conf_mat)
#    #print(q_size)
#  return compute_errors(conf_mat, 'Validation', class_info, verbose=False)


def save_predictions(sess, image, logits, softmax):
  width = FLAGS.img_width
  height = FLAGS.img_height
  img_dir = FLAGS.dataset_dir
  image_list = next(os.walk(img_dir))[2]
  for i in trange(len(image_list)):
    img = ski.data.load(img_dir + image_list[i])
    img = ski.transform.resize(img, (height, width), preserve_range=True, order=3)
    img = img.astype(np.float32)
    for c in range(3):
      img[:,:,c] -= VGG_MEAN[c]
    img_data = img.reshape(1, height, width, 3)
    out_logits, out_softmax = sess.run([logits, softmax], feed_dict={image : img_data})
    y = out_logits[0].argmax(2).astype(np.int32)
    p = np.amax(out_softmax, axis=2)
    print((p > 0.4).sum() / p.size)
    #print(p)
    eval_helper.draw_output(y, Dataset.CLASS_INFO, os.path.join(FLAGS.save_dir, image_list[i]))
    save_path = os.path.join(FLAGS.save_dir, 'softmax_' + image_list[i])
    ski.io.imsave(save_path, p)


def eval_and_save_predictions(sess, image, logits, softmax):
  width = FLAGS.img_width
  height = FLAGS.img_height
  img_dir = FLAGS.dataset_dir
  image_list = next(os.walk(img_dir))[2]
  num_classes = 11
  conf_mat = np.ascontiguousarray(np.zeros((num_classes, num_classes), dtype=np.uint64))
  for i in trange(len(image_list)):
    img = ski.data.load(img_dir + image_list[i])
    img = ski.transform.resize(img, (height, width), preserve_range=True, order=3)
    img = img.astype(np.float32)
    for c in range(3):
      img[:,:,c] -= VGG_MEAN[c]
    img_data = img.reshape(1, height, width, 3)
    out_logits = sess.run(logits, feed_dict={image : img_data})
    y = out_logits[0].argmax(2).astype(np.int32)

    y = eval_helper.map_cityscapes_to_kitti(y, Dataset.CITYSCAPES_TO_KITTI_MAP)
    eval_helper.draw_output(y, Dataset.CLASS_INFO, os.path.join(FLAGS.save_dir + image_list[i]))
    with open(FLAGS.gt_dir + image_list[i][:-4] + '.pickle', 'rb') as f:
      data_list = pickle.load(f)
    yt = data_list[0].astype(np.int32)
    #assert(yt.max() < 11 && yt.min() >= -1)
    #assert(yt.max() < 11 && yt.min() >= -1)

    cylib.collect_confusion_matrix(y.reshape(-1), yt.reshape(-1), conf_mat)
    #eval_helper.collect_confusion_matrix(y.reshape(-1), yt.reshape(-1), conf_mat)
    eval_helper.compute_errors(conf_mat, 'Validation', Dataset.CLASS_INFO)


def main(argv=None):
  os.makedirs(FLAGS.save_dir, exist_ok=True)
  spec = importlib.util.spec_from_file_location("model", MODEL_PATH)
  model = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(model)
  #dataset = CityscapesDataset(FLAGS.dataset_dir, 'val')
  model_checkpoint_path = NET_DIR + 'model.ckpt'
  with tf.Graph().as_default():
    sess = tf.Session()

    #image, labels, weights, num_labels, img_name = \
    #    reader.inputs(dataset)
    batch_shape = (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth)
    image = tf.placeholder(tf.float32, shape=batch_shape)

    #image_val, labels_val, weights_val, num_labels_val, img_name_val = \
    #    reader.inputs(dataset, shuffle=False)   
    # Restores from checkpoint
    with tf.Session() as sess:
      #logits = model.inference(image)
      with tf.variable_scope("model"):                                                                 
        logits = model.inference(image, is_training=False)
        softmax = losses.softmax(logits)
        #loss = model.loss(logits, labels, weights, num_labels)                                         
      #with tf.variable_scope("model", reuse=True):                                                     
      #  logits_val = model.inference(image_val, is_training=False)                                     
      #  loss_val = model.loss(logits_val, labels_val, weights_val, num_labels_val)

      saver = tf.train.Saver()
      saver.restore(sess, model_checkpoint_path)
      #save_predictions(sess, image, logits, softmax)
      eval_and_save_predictions(sess, image, logits, softmax)

      ## Start the queue runners.
      #coord = tf.train.Coordinator()
      #data_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      #pixel_acc, iou_acc, recall, precision, num_pixels = evaluate_model(sess, dataset, logits, labels, img_name)
      #pixel_acc_lst += [pixel_acc]
      #iou_acc_lst += [iou_acc]
      #print(iou_acc)

      #tf.reset_default_graph() - bad
      #push_notebook()
      #coord.request_stop()
      #coord.join(data_threads)


if __name__ == '__main__':
  tf.app.run()
