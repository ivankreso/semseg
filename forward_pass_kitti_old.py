import os

import tensorflow as tf
import numpy as np
import skimage as ski
import skimage.data, skimage.transform
#import vgg_16s_baseline as model
import importlib.util
from tqdm import trange
import skimage as ski
import skimage.data, skimage.transform

from eval_helper import *
import libs.cylib as cylib

from datasets.cityscapes.cityscapes import CityscapesDataset
from datasets.cityscapes.cityscapes_info import class_info, class_color_map
import datasets.reader as reader

#tf.app.flags.DEFINE_integer('img_width', 1024, '')
#tf.app.flags.DEFINE_integer('img_height', 432, '')
tf.app.flags.DEFINE_integer('img_width', 1232, '')
tf.app.flags.DEFINE_integer('img_height', 384, '')
tf.app.flags.DEFINE_integer('img_depth', 3, '')
tf.app.flags.DEFINE_integer('batch_size', 1, '')

tf.app.flags.DEFINE_string('dataset_dir',
    '/home/kivan/datasets/KITTI/semantic_segmentation/train/data/rgb/', '')
    #'/home/kivan/datasets/KITTI/sequences_color/07/image_2/', '')
tf.app.flags.DEFINE_integer('num_classes', 19, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('is_training', False, '')
tf.app.flags.DEFINE_string('save_dir', '/home/kivan/datasets/KITTI/output/kitti_semseg/', '')

FLAGS = tf.app.flags.FLAGS

#NET_DIR = '/home/kivan/source/results/semseg/tf/nets/8_5_10-40-27/'
NET_DIR = '/home/kivan/source/results/semseg/tf/trained/100_9_5_14-35-05/'
MODEL_PATH = NET_DIR + 'model.py'

def draw_prediction(y, colors, path):
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


def save_predictions(sess, image, logits):
  width = FLAGS.img_width
  height = FLAGS.img_height
  img_dir = FLAGS.dataset_dir
  image_list = next(os.walk(img_dir))[2]
  for i in trange(len(image_list)):
    img = ski.data.load(img_dir + image_list[i])
    img = ski.transform.resize(img, (height, width))
    img = img.astype(np.float32)
    #img = (img - img.mean()) / img.std()
    for c in range(3):
      img[:,:,c] -= img[:,:,c].mean()
      img[:,:,c] /= img[:,:,c].std()
    img_data = img.reshape(1, height, width, 3)
    out_logits = sess.run(logits, feed_dict={image : img_data})
    id_img = out_logits[0].argmax(2).astype(np.int32, copy=False)
    draw_prediction(id_img, class_info, FLAGS.save_dir + image_list[i])


def main(argv=None):
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
        logits = model.inference(image)                                                                
        #loss = model.loss(logits, labels, weights, num_labels)                                         
      #with tf.variable_scope("model", reuse=True):                                                     
      #  logits_val = model.inference(image_val, is_training=False)                                     
      #  loss_val = model.loss(logits_val, labels_val, weights_val, num_labels_val)

      saver = tf.train.Saver()
      saver.restore(sess, model_checkpoint_path)
      save_predictions(sess, image, logits)

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
