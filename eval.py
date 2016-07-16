import tensorflow as tf
import numpy as np
import skimage as ski
import skimage.data, skimage.transform
import models.vgg_16s as model
import importlib.util
from tqdm import trange
from eval_helper import *
import libs.cylib as cylib

from datasets.cityscapes.cityscapes import CityscapesDataset
from datasets.cityscapes.cityscapes_info import class_info, class_color_map
import datasets.reader as reader

FLAGS = tf.app.flags.FLAGS

#g_net_dir = '/home/kivan/source/results/semseg/tf/nets/23_3_13:39:43/'
#g_net_dir = '/home/kivan/source/results/semseg/tf/nets/14_4_22-21-28/'
g_net_dir = '/home/kivan/source/results/semseg/tf/nets/17_4_17-04-30/'
g_model_path = g_net_dir + 'model.py'
spec = importlib.util.spec_from_file_location("model", g_model_path)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def loss(logits, labels):
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
  logits_1s = tf.image.resize_bilinear(logits, [FLAGS.img_height, FLAGS.img_width])
  logits_1d = tf.reshape(logits_1s, [num_examples, FLAGS.num_classes])
  return slim.losses.cross_entropy_loss(logits_1d, one_hot_labels)


def main(argv=None):  # pylint: disable=unused-argument
  dataset = CityscapesDataset(FLAGS.dataset_dir, 'val')
  ckpt = tf.train.get_checkpoint_state(g_net_dir)
  sess = tf.Session()
  if ckpt and ckpt.model_checkpoint_path:
    image, labels, _, _, img_name = reader.inputs(dataset)
    # Restores from checkpoint
    #with tf.Session() as sess:
    logits_sub = model.inference(image)
    logits = tf.image.resize_bilinear(logits_sub, [FLAGS.img_height, FLAGS.img_width])

    # TODO
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    saver = tf.train.Saver()
    print(ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/cifar10_train/model.ckpt-0,
    # extract global_step from it.
    #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
  else:
    print('No checkpoint file found')
    raise

  conf_mat = np.ascontiguousarray(np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64))
  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)
  for i in trange(dataset.num_examples()):
    #_, loss_value = sess.run([train_op, loss])
    out_logits, yt = sess.run([logits, labels])
    y = out_logits[0].argmax(2).astype(np.int32, copy=False)
    yt = yt.astype(np.int32, copy=False)
    #cylib.collect_confusion_matrix(y.reshape(-1), yt, conf_mat)
    collect_confusion_matrix(y.reshape(-1), yt, conf_mat)
    compute_errors(conf_mat, 'validation', class_info)
  compute_errors(conf_mat, 'validation', class_info)


if __name__ == '__main__':
  tf.app.run()
