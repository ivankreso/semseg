import os
import pickle
from os.path import join

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
import cv2

from datasets.cityscapes.cityscapes import CityscapesDataset as Dataset
#import datasets.kitti.kitti_info as Dataset
import datasets.reader as reader

tf.app.flags.DEFINE_integer('img_width', 1242, '')
tf.app.flags.DEFINE_integer('img_height', 375, '')
#tf.app.flags.DEFINE_integer('img_height', 896, '')

#DATA_DIR = '/home/kivan/datasets/Cityscapes/orig/test'
#DATA_DIR = '/home/kivan/datasets/Cityscapes/orig/test_masked'
#DATA_DIR = '/home/kivan/datasets/Cityscapes/masked/test'
#DATA_DIR = '/home/kivan/datasets/Cityscapes/masked_1/test'

#DATA_DIR = '/home/kivan/datasets/Cityscapes/masked/black/full/test'
#SAVE_DIR = '/home/kivan/datasets/results/out/cityscapes/hood/'
#DATA_DIR = '/home/kivan/datasets/KITTI/training/image_2'
#SAVE_DIR = '/home/kivan/datasets/KITTI/out_softmax/train_left'
DATA_DIR = '/home/kivan/datasets/KITTI/training/image_3'
SAVE_DIR = '/home/kivan/datasets/KITTI/out_softmax/train_right'
#DATA_DIR = '/home/kivan/datasets/KITTI/testing/image_3'
#SAVE_DIR = '/home/kivan/datasets/KITTI/out/test_right'

#DATA_DIR = '/home/kivan/datasets/Cityscapes/masked/black/croped/test'
#SAVE_DIR = '/home/kivan/datasets/results/out/cityscapes/main/'

#NET_DIR = '/home/kivan/datasets/results/iccv/03_7_3_17-57-58/'
NET_DIR = '/home/kivan/datasets/results/iccv/cityscapes/full_best_16_3_16-25-14/'
#NET_DIR = '/home/kivan/datasets/results/tmp/cityscapes/22_3_10-50-51/'
#NET_DIR = '/home/kivan/datasets/results/tmp/cityscapes/28_3_07-01-21/'
MODEL_PATH = NET_DIR + 'model.py'

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
#tf.app.flags.DEFINE_string('gt_dir',
#  '/home/kivan/datasets/KITTI/semantic_segmentation/gt_data/' + DATA_NAME + '/', '')
#  #'/home/kivan/datasets/KITTI/semantic_segmentation/gt_data/train/', '')

FLAGS = tf.app.flags.FLAGS

#NET_DIR = '/home/kivan/source/results/semseg/tf/nets/8_5_10-40-27/'
#NET_DIR = '/home/kivan/source/results/semseg/tf/saved/dilated_full66_30_5_15-19-36/'
#NET_DIR = '/home/kivan/source/results/semseg/tf/nets/7_6_15-08-57/'
#NET_DIR = '/home/kivan/source/results/semseg/tf/saved/61.5_640_3_6_10-09-12/'

def map_to_submit_ids(img):
  img_ids = np.zeros_like(img, dtype=np.uint8)
  for i, cid in enumerate(Dataset.train_ids):
    img_ids[img==i] = cid
  return img_ids


def save_predictions(sess, image, logits, codes, softmax, depth):
  #width = FLAGS.img_width
  #height = FLAGS.img_height
  img_dir = FLAGS.dataset_dir
  image_list = next(os.walk(img_dir))[2]
  for i in trange(len(image_list)):
    #print(image_list[i])
    #img = ski.data.load(img_dir + image_list[i])
    #img = ski.transform.resize(img, (height, width), preserve_range=True, order=3)
    #img = img.astype(np.float32)
    #for c in range(3):
    #  img[:,:,c] -= VGG_MEAN[c]
    #img = cv2.imread(join(city_dir, image_list[i]), cv2.IMREAD_COLOR)
    img = ski.data.load(join(img_dir, image_list[i]))
    print(img.shape)
    #depth_data = ski.data.load(join(depth_dir, image_list[i])).astype(np.float32)
    #depth_data /= 256.0
    #assert img.shape[0] == height and img.shape[1] == width
    height = img.shape[0]
    width = img.shape[1]
    img_data = img.reshape(1, height, width, 3)
    #depth_data = depth_data.reshape(1, height, width, 1)
    #out_logits, out_softmax = sess.run([logits, softmax], feed_dict={image : img_data})
    #feed_dict={image:img_data, depth:depth_data}
    feed_dict={image:img_data}
    out_logits, out_softmax, out_code = sess.run([logits, softmax, codes], feed_dict=feed_dict)
    y = out_logits[0].argmax(2).astype(np.int32)
    #p = np.amax(out_softmax, axis=2)
    #print('Over 90% = ', (p > 0.9).sum() / p.size)
    #print(p)
    out_code = out_code[0]
    print(out_code.shape)
    out_code = np.ascontiguousarray(out_code.transpose([1,2,0]))
    print(out_code.shape)
    #eval_helper.draw_output(y, Dataset.CLASS_INFO, os.path.join(FLAGS.save_dir, 'color', image_list[i]))
    y_submit = map_to_submit_ids(y)
    save_path = join(FLAGS.save_dir, 'labels', image_list[i])
    #ski.io.imsave(save_path, y_submit)
    save_path = join(FLAGS.save_dir, 'embedding', image_list[i][:-3]+'npy')
    print(save_path)
    #np.save(save_path, out_code)
    save_path = join(FLAGS.save_dir, 'softmax', image_list[i][:-3]+'npy')
    np.save(save_path, out_softmax)
    #save_path = os.path.join(FLAGS.save_dir, 'softmax_' + image_list[i])
    #ski.io.imsave(save_path, p)


def main(argv=None):
  os.makedirs(FLAGS.save_dir, exist_ok=True)
  os.makedirs(join(FLAGS.save_dir, 'color'), exist_ok=True)
  os.makedirs(join(FLAGS.save_dir, 'labels'), exist_ok=True)
  os.makedirs(join(FLAGS.save_dir, 'embedding'), exist_ok=True)
  os.makedirs(join(FLAGS.save_dir, 'softmax'), exist_ok=True)
  print(MODEL_PATH)
  spec = importlib.util.spec_from_file_location("model", MODEL_PATH)
  model = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(model)
  #dataset = CityscapesDataset(FLAGS.dataset_dir, 'val')
  model_checkpoint_path = NET_DIR + 'model.ckpt'
  sess = tf.Session()

  #image, labels, weights, num_labels, img_name = \
  #    reader.inputs(dataset)
  #batch_shape = (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth)
  batch_shape = (FLAGS.batch_size, None, None, FLAGS.img_depth)
  #depth_shape = (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 1)
  image = tf.placeholder(tf.float32, shape=batch_shape)
  #depth = tf.placeholder(tf.float32, shape=depth_shape)
  depth = None
  #logits, mid_logits = model.inference(image, depth)
  logits, mid_logits, codes = model.inference(image)
  #softmax = losses.softmax(logits)
  softmax = tf.nn.softmax(logits)
  saver = tf.train.Saver()
  saver.restore(sess, model_checkpoint_path)
  save_predictions(sess, image, logits, codes, softmax, depth)

    # Restores from checkpoint
      #loss = model.loss(logits, labels, weights, num_labels)                                         
    #with tf.variable_scope("model", reuse=True):                                                     
    #  logits_val = model.inference(image_val, is_training=False)                                     
    #  loss_val = model.loss(logits_val, labels_val, weights_val, num_labels_val)

    #eval_and_save_predictions(sess, image, logits, softmax)

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


#def eval_and_save_predictions(sess, image, logits, softmax):
#  width = FLAGS.img_width
#  height = FLAGS.img_height
#  img_dir = FLAGS.dataset_dir
#  image_list = next(os.walk(img_dir))[2]
#  num_classes = 11
#  conf_mat = np.ascontiguousarray(np.zeros((num_classes, num_classes), dtype=np.uint64))
#  for i in trange(len(image_list)):
#    img = ski.data.load(img_dir + image_list[i])
#    img = ski.transform.resize(img, (height, width), preserve_range=True, order=3)
#    img = img.astype(np.float32)
#    for c in range(3):
#      img[:,:,c] -= VGG_MEAN[c]
#    img_data = img.reshape(1, height, width, 3)
#    out_logits = sess.run(logits, feed_dict={image : img_data})
#    y = out_logits[0].argmax(2).astype(np.int32)
#
#    y = eval_helper.map_cityscapes_to_kitti(y, Dataset.CITYSCAPES_TO_KITTI_MAP)
#    eval_helper.draw_output(y, Dataset.CLASS_INFO, os.path.join(FLAGS.save_dir, image_list[i]))
#    with open(FLAGS.gt_dir + image_list[i][:-4] + '.pickle', 'rb') as f:
#      data_list = pickle.load(f)
#    yt = data_list[0].astype(np.int32)
#    #assert(yt.max() < 11 && yt.min() >= -1)
#    #assert(yt.max() < 11 && yt.min() >= -1)
#
#    cylib.collect_confusion_matrix(y.reshape(-1), yt.reshape(-1), conf_mat)
#    #eval_helper.collect_confusion_matrix(y.reshape(-1), yt.reshape(-1), conf_mat)
#    eval_helper.compute_errors(conf_mat, 'Validation', Dataset.CLASS_INFO)



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

