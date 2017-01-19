import pdb
import tensorflow as tf
import argparse
import os, re
import numpy as np
import skimage as ski
import skimage.data
import skimage.transform
import cv2 as cv
from tqdm import trange

import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

import eval_helper
from models.model_helper import read_vgg_init
import losses
#import datasets.reader as reader
#import datasets.flip_reader as reader
import datasets.reader_jitter as reader


from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
  return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                               grad,
                                               op.outputs[1],
                                               op.get_attr("ksize"),
                                               op.get_attr("strides"),
                                               padding=op.get_attr("padding"))

FLAGS = tf.app.flags.FLAGS
HEAD_PREFIX = 'head'

MODEL_DEPTH = 50
#MEAN_RGB = [75.2051479, 85.01498926, 75.08929598]
MEAN_BGR = [75.08929598, 85.01498926, 75.2051479]
#MEAN_BGR = [103.939, 116.779, 123.68]
weight_decay = 1e-4
# to big weight_decay = 5e-3
init_func = layers.variance_scaling_initializer()



def evaluate(name, sess, epoch_num, run_ops, dataset, data):
  num_examples = dataset.num_examples()
  for step in trange(num_examples):
    out = sess.run(run_ops)
    if step % 50 == 0:
      evaluate_output(out, step, 'valid')
  #loss_val, accuracy, iou, recall, precision = eval_helper.evaluate_segmentation(
  #    sess, epoch_num, run_ops, dataset.num_examples(), get_feed_dict=get_valid_feed)
  #if iou > data['best_iou'][0]:
  #  data['best_iou'] = [iou, epoch_num]
  #data['iou'] += [iou]
  #data['acc'] += [accuracy]
  #data['loss'] += [loss_val]

def plot_results(train_data, valid_data):
  pass
  #eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats'),
  #                                   train_data, valid_data)
  #eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats')), train_data)


def print_results(data):
  pass
  #print('Best validation IOU = %.2f (epoch %d)' % tuple(data['best_iou']))

def evaluate_output(out, step, name='train'):
  x = out[1][0]
  x_rec = out[2][0]
  for c in range(3):
    x[:,:,c] += MEAN_BGR[c]
    x_rec[:,:,c] += MEAN_BGR[c]
  x = np.round(x)
  x_rec = np.round(x_rec)
  #print(x_rec.shape)
  #print('x = ', x.min(), x.max())
  #print('rec = ', x_rec.min(), x_rec.max())
  x_rec[x_rec<0] = 0
  x_rec[x_rec>255] = 255
  x_rec = x_rec.astype(np.uint8)
  x = x.astype(np.uint8)
  #save_dir = FLAGS.debug_dir, 'train'
  save_dir = '/home/kivan/source/results/semseg/tf/tmp/' + name
  path = os.path.join(save_dir, str(step)+'_img_rec.png')
  cv.imwrite(path, x_rec)
  path = os.path.join(save_dir, str(step)+'_img_raw.png')
  cv.imwrite(path, x)
  

def init_eval_data():
  train_data = {}
  valid_data = {}
  train_data['lr'] = []
  train_data['loss'] = []
  train_data['iou'] = []
  train_data['acc'] = []
  train_data['best_iou'] = [0, 0]
  valid_data['best_iou'] = [0, 0]
  valid_data['loss'] = []
  valid_data['iou'] = []
  valid_data['acc'] = []
  return train_data, valid_data

def normalize_input(img):
  return img - MEAN_BGR
  #"""Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
  #with tf.name_scope('input'), tf.device('/cpu:0'):
  #  red, green, blue = tf.split(3, 3, rgb)
  #  bgr = tf.concat(3, [blue, green, red])
  #  #bgr -= MEAN_BGR
  #  return bgr


def total_loss_sum(losses):
  # Assemble all of the losses for the current tower only.
  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  return total_loss


def create_init_op(vgg_layers):
  variables = tf.contrib.framework.get_variables()
  init_map = {}
  for var in variables:
    name_split = var.name.split('/')
    if len(name_split) != 2:
      continue
    name = name_split[0] + '/' + name_split[1][:-2]
    if name in vgg_layers:
      print(var.name, ' --> init from ', name)
      init_map[var.name] = vgg_layers[name]
    else:
      print(var.name, ' --> random init')
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
  return init_op, init_feed


def unravel_argmax(argmax, shape):
  output_list = [argmax // (shape[2]*shape[3]),
                 argmax % (shape[2]*shape[3]) // shape[3]]
  return tf.pack(output_list)

def unpool_layer_2(bottom, argmax):
  bottom_shape = tf.shape(bottom)
  top_shape = [bottom_shape[0], bottom_shape[1]*2, bottom_shape[2]*2, bottom_shape[3]]

  batch_size = top_shape[0]
  height = top_shape[1]
  width = top_shape[2]
  channels = top_shape[3]

  argmax_shape = tf.to_int64([batch_size, height, width, channels])
  argmax = unravel_argmax(argmax, argmax_shape)

  t1 = tf.to_int64(tf.range(channels))
  t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
  t1 = tf.reshape(t1, [-1, channels])
  t1 = tf.transpose(t1, perm=[1, 0])
  t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
  t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

  t2 = tf.to_int64(tf.range(batch_size))
  t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
  t2 = tf.reshape(t2, [-1, batch_size])
  t2 = tf.transpose(t2, perm=[1, 0])
  t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

  t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

  t = tf.concat(4, [t2, t3, t1])
  indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

  x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
  values = tf.reshape(x1, [-1])

  delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
  return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))


def unpool_layer(net, argmax_data):
  indices, top_shape = argmax_data
  #bottom_shape = tf.shape(net, out_type=tf.int64)
  #top_shape = [bottom_shape[0], bottom_shape[1]*2, bottom_shape[2]*2, bottom_shape[3]]
  #output_size = 4 * tf.size(net, out_type=tf.int64)
  output_size = tf.reduce_prod(top_shape)
  output_size = tf.reshape(output_size, [-1])
  #print('size = ', output_size)
  #top_shape = tf.to_int32(tf.stack(top_shape))

  net = tf.reshape(net, [-1])
  indices = tf.reshape(indices, [-1, 1])
  #print(indices)
  #print(net)
  #print(output_size)
  #top = tf.scatter_nd(indices, bottom, top_shape)
  #net = tf.Print(net, [tf.shape(net)], message='U4_scat_start = ', summarize=10)
  net = tf.scatter_nd(indices, net, output_size)
  #net = tf.Print(net, [tf.shape(net)], message='U4_scat_end = ', summarize=10)
  net = tf.reshape(net, tf.to_int32(top_shape))
  #net = tf.Print(net, [tf.shape(net)], message='U4_reshape_end = ', summarize=10)
  return net


def decode(net, argmax, bn_params):
  with tf.contrib.framework.arg_scope([layers.convolution2d],
      kernel_size=3, stride=1, padding='SAME', rate=1, activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      #normalizer_fn=None, normalizer_params=None,
      weights_initializer=layers.variance_scaling_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    stride = [1,1,1,1]
    loss_data = []

    #net = unpool_layer(net, argmax[-1])
    #net.set_shape([None,None,None,512])
    #net = layers.convolution2d(net, 512, scope='dec_conv5_3')
    #net = layers.convolution2d(net, 512, scope='dec_conv5_2')
    #net = layers.convolution2d(net, 512, scope='dec_conv5_1')
    #loss_data.append(net)
    #net = unpool_layer(net, argmax[-2])
    #net.set_shape([None,None,None,512])
    #net = layers.convolution2d(net, 512, scope='dec_conv4_3')
    #net = layers.convolution2d(net, 512, scope='dec_conv4_2')
    #net = layers.convolution2d(net, 256, scope='dec_conv4_1')
    #loss_data.append(net)
    #net = unpool_layer(net, argmax[-3])
    #net.set_shape([None,None,None,256])
    #net = layers.convolution2d(net, 256, scope='dec_conv3_3')
    #net = layers.convolution2d(net, 256, scope='dec_conv3_2')
    #net = layers.convolution2d(net, 128, scope='dec_conv3_1')
    #loss_data.append(net)
    net = unpool_layer(net, argmax[1])
    net.set_shape([None,None,None,128])
    net = layers.convolution2d(net, 128, scope='dec_conv2_2')
    #net = layers.convolution2d(net, 64, scope='dec_conv2_2')
    net = layers.convolution2d(net, 64, scope='dec_conv2_1')
    loss_data.append(net)
    net = unpool_layer(net, argmax[0])
    net.set_shape([None,None,None,64])
    net = layers.convolution2d(net, 64, scope='dec_conv1_2')
    #net = layers.convolution2d(net, 32, scope='dec_conv1_2')
    net = layers.convolution2d(net, 3, activation_fn=None, normalizer_fn=None,
                               weights_regularizer=None, scope='dec_conv1_1')
    loss_data.append(net)
    return loss_data


def l2_loss(layer1, layer2):
  l2_loss = tf.nn.l2_loss(layer1 - layer2)
  return l2_loss / tf.to_float(tf.size(layer1))
  #return l2_loss / layer1.get_shape().num_elements()

def _build(x, is_training):
  bn_params = {
      # Decay for the moving averages.
      #'decay': 0.999,
      'decay': 0.9,
      'center': True,
      'scale': True,
      # epsilon to prevent 0s in variance.
      #'epsilon': 0.001,
      'epsilon': 1e-5,
      # None to force the updates
      'updates_collections': None,
      'is_training': is_training,
  }
  with tf.contrib.framework.arg_scope([layers.convolution2d],
      kernel_size=3, stride=1, padding='SAME', rate=1, activation_fn=tf.nn.relu,
      #normalizer_fn=None,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=None,
      #weights_initializer=layers.variance_scaling_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):

    argmax_data = []
    enc_data = [x]
    ksize = [1,2,2,1]
    stride = [1,2,2,1]
    net = layers.convolution2d(x, 64, scope='conv1_1')
    net = layers.convolution2d(net, 64, scope='conv1_2')
    #net = layers.max_pool2d(net, 2, 2, scope='pool1')
    up_shape = tf.shape(net, out_type=tf.int64)
    net, argmax = tf.nn.max_pool_with_argmax(net, ksize, stride,
                   padding='SAME', name='pool1')
    argmax_data.append([argmax, up_shape])
    enc_data.append(net)
    net = layers.convolution2d(net, 128, scope='conv2_1')
    net = layers.convolution2d(net, 128, scope='conv2_2')
    #net = layers.max_pool2d(net, 2, 2, scope='pool2')
    up_shape = tf.shape(net, out_type=tf.int64)
    net, argmax = tf.nn.max_pool_with_argmax(net, ksize, stride,
                  padding='SAME', name='pool2')
    argmax_data.append([argmax, up_shape])
    enc_data.append(net)
    #net = layers.convolution2d(net, 256, scope='conv3_1')
    #net = layers.convolution2d(net, 256, scope='conv3_2')
    #net = layers.convolution2d(net, 256, scope='conv3_3')
    ##net = layers.max_pool2d(net, 2, 2, scope='pool3')
    #up_shape = tf.shape(net, out_type=tf.int64)
    #net, argmax = tf.nn.max_pool_with_argmax(net, ksize, stride,
    #              padding='SAME', name='pool3')
    #argmax_data.append([argmax, up_shape])
    #enc_data.append(net)
    #net = layers.convolution2d(net, 512, scope='conv4_1')
    #net = layers.convolution2d(net, 512, scope='conv4_2')
    #net = layers.convolution2d(net, 512, scope='conv4_3')
    ##net = layers.max_pool2d(net, 2, 2, scope='pool4')
    #up_shape = tf.shape(net, out_type=tf.int64)
    #net, argmax = tf.nn.max_pool_with_argmax(net, ksize, stride,
    #              padding='SAME', name='pool4')
    #argmax_data.append([argmax, up_shape])
    #enc_data.append(net)
    #net = layers.convolution2d(net, 512, scope='conv5_1')
    #net = layers.convolution2d(net, 512, scope='conv5_2')
    #net = layers.convolution2d(net, 512, scope='conv5_3')
    ##net = layers.max_pool2d(net, 2, 2, scope='pool5')
    #up_shape = tf.shape(net, out_type=tf.int64)
    #net, argmax = tf.nn.max_pool_with_argmax(net, ksize, stride,
    #              padding='SAME', name='pool5')
    #argmax_data.append([argmax, up_shape])


  with tf.variable_scope(HEAD_PREFIX):
    dec_data = decode(net, argmax_data, bn_params)
    unsup_weight = 1
    unsup_loss = tf.to_float(0)
    #l2_mid_wgt = 0.2 batch norm!
    l2_mid_wgt = 0.0
    l2_wgt = 1.0
    weights = [l2_wgt]
    weights += [l2_mid_wgt] * 4
    #for i, enc in enumerate(enc_data):
    #  unsup_loss += weights[i] * l2_loss(enc, dec_data[-1-i])
    unsup_loss += l2_wgt * l2_loss(enc_data[0], dec_data[-1])
    logits = net

    #with arg_scope([layers.convolution2d],
    #      padding='SAME', activation_fn=tf.nn.relu,
    #      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
    #      weights_initializer=init_func,
    #      weights_regularizer=layers.l2_regularizer(weight_decay)):
    #  net = layers.convolution2d(net, 512, kernel_size=5, rate=2, scope='conv1')
    #  net = layers.convolution2d(net, 512, kernel_size=1, scope='conv2') # faster
    #  #l = tf.Print(l, [tf.shape(image)], message='IMG SHAPE = ')
    #  logits = layers.convolution2d(net, FLAGS.num_classes, kernel_size=1,
    #        activation_fn=None, weights_initializer=init_func, normalizer_fn=None,
    #        scope='logits')
    #  print(input_shape)
    #  #global resize_height, resize_width
    #  #height = tf.Print(height, [height, width], message='SHAPE = ')
    #  #logits = tf.image.resize_bilinear(logits, [FLAGS.img_height, FLAGS.img_width],
    #  #logits = tf.image.resize_bilinear(logits, [resize_height, resize_width],
    input_shape = tf.shape(x)
    height = input_shape[1]
    width = input_shape[2]
    logits = tf.image.resize_bilinear(logits, [height, width],
                                      name='resize_logits')
    return logits, unsup_loss, dec_data[-1]


def jitter(image, labels, weights):
  global random_flip_tf, resize_width, resize_height
  random_flip_tf = tf.placeholder(tf.bool, shape=(), name='random_flip')
  resize_width = tf.placeholder(tf.int32, shape=(), name='resize_width')
  resize_height = tf.placeholder(tf.int32, shape=(), name='resize_height')
  image_split = tf.unstack(image, axis=0)
  weights_split = tf.unstack(weights, axis=0)
  labels_split = tf.unstack(labels, axis=0)
  out_img = []
  out_weights = []
  out_labels = []
  for i in range(FLAGS.batch_size):
    out_img.append(tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(image_split[i]),
                       lambda: image_split[i]))
    #print(cond_op)
    #image_split[i] = tf.assign(image_split[i], cond_op)
    #image[i] = tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(image[i]),
    #cond_op = tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(image[i]),
                       #lambda: tf.identity(image[i]))
    #image[i] = tf.assign(image[i], cond_op)
    out_labels.append(tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(labels[i]),
                      lambda: labels[i]))
    out_weights.append(tf.cond(random_flip_tf, lambda: tf.image.flip_left_right(weights[i]),
                       lambda: weights[i]))
  image = tf.stack(out_img, axis=0)
  weights = tf.stack(out_weights, axis=0)
  labels = tf.stack(out_labels, axis=0)

  ##image = tf.image.resize_bicubic(image, [resize_height, resize_width])
  #image = tf.image.resize_bilinear(image, [resize_height, resize_width])
  #labels = tf.image.resize_nearest_neighbor(labels, [resize_height, resize_width])
  #weights = tf.image.resize_nearest_neighbor(weights, [resize_height, resize_width])
  return image, labels, weights


def get_train_feed():
  global random_flip_tf, resize_width, resize_height
  random_flip = int(np.random.choice(2, 1))
  #resize_scale = np.random.uniform(0.5, 2)
  #resize_scale = np.random.uniform(0.4, 1.5)
  resize_scale = np.random.uniform(0.5, 1)
  #resize_scale = np.random.uniform(0.6, 1.4)
  width = np.int32(int(round(FLAGS.img_width * resize_scale)))
  height = np.int32(int(round(FLAGS.img_height * resize_scale)))
  feed_dict = {random_flip_tf:random_flip, resize_width:width, resize_height:height}
  return feed_dict

def get_valid_feed():
  global random_flip_tf, resize_width, resize_height
  feed_dict = {random_flip_tf:0, resize_width:0, resize_height:0}
  return feed_dict

def build(dataset, is_training, reuse=False):
  # Get images and labels.
  image, labels, weights, _, img_names = reader.inputs(
      dataset, is_training=is_training, num_epochs=FLAGS.max_epochs)

  #image = tf.Print(image, [tf.reduce_min(image), tf.reduce_max(image)], message='img min = ', summarize=10)
  if is_training:
    image, labels, weights = jitter(image, labels, weights)
  image = normalize_input(image)

  if reuse:
    tf.get_variable_scope().reuse_variables()

  logits, unsup_loss, rec_img = _build(image, is_training)
  total_loss = loss(logits, labels, weights, is_training)
   
  total_loss += unsup_loss

  #all_vars = tf.contrib.framework.get_variables()
  #for v in all_vars:
  #  print(v.name)
  if is_training:
    vgg_layers, vgg_layer_names = read_vgg_init(FLAGS.vgg_init_dir)
    init_op, init_feed = create_init_op(vgg_layers)
    #return [total_loss], init_op, init_feed
    return [total_loss, image, rec_img], init_op, init_feed
  else:
    return [total_loss, image, rec_img, img_names]

def minimize(opts, loss, global_step):
  all_vars = tf.trainable_variables()
  grads = tf.gradients(loss, all_vars)
  resnet_grads_and_vars = []
  head_grads_and_vars = []
  for i, v in enumerate(all_vars):
    if v.name[:4] == HEAD_PREFIX:
      print(v.name, ' --> lr*10')
      head_grads_and_vars += [(grads[i], v)]
    else:
      resnet_grads_and_vars += [(grads[i], v)]
  train_op1 = opts[0].apply_gradients(resnet_grads_and_vars, global_step=global_step)
  train_op2 = opts[1].apply_gradients(head_grads_and_vars, global_step=global_step)
  return tf.group(train_op1, train_op2)

#def minimize(opt, loss, global_step):
#  #resnet_vars = tf.trainable_variables()
#  all_vars = tf.trainable_variables()
#  resnet_vars = []
#  head_vars = []
#  for v in all_vars:
#    if v.name[:4] == 'head':
#      print(v.name)
#      head_vars += [v]
#    else:
#      resnet_vars += [v]
#  grads_and_vars = opt.compute_gradients(loss, resnet_vars + head_vars)
#  resnet_gv = grads_and_vars[:len(resnet_vars)]
#  head_gv = grads_and_vars[len(resnet_vars):]
#  lr_mul = 10
#  #lr_mul = 1
#  print(head_gv[0])
#  #head_gv = [[g*lr_mul, v] for g,v in head_gv]
#  print(head_gv[0])
#  #  ygrad, _ = grads_and_vars[1]
#  train_op = opt.apply_gradients(resnet_gv + head_gv, global_step=global_step)
#  return train_op


def loss(logits, labels, weights, is_training=True):
  # TODO
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=10)
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=100)
  loss_val = 0.0
  #loss_val = losses.flip_xent_loss(logits, labels, weights, max_weight=10)
  #loss_tf = tf.contrib.losses.softmax_cross_entropy()
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights)
  #loss_val = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=1)
  #loss_val = losses.weighted_hinge_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss(logits, labels, weights, num_labels)
  #loss_val = losses.flip_xent_loss_symmetric(logits, labels, weights, num_labels)
  all_losses = [loss_val]

  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)

  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss

def num_batches(dataset):
  return reader.num_examples(dataset)


