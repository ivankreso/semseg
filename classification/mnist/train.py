import os
import pickle
import time
import math
from tqdm import trange

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io

import eval_helper
#import model1 as model
#import model2 as model
import model_fcn as model

#np.random.seed(100) 
np.random.seed(int(time.time() * 1e6) % 2**31)

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

## MNIST
#DATASET = 'MNIST'
#DATA_DIR = '/home/kivan/datasets/MNIST/'
#SAVE_DIR = '/home/kivan/source/out/mnist/'
#img_height = 28
#img_width = 28
#num_channels = 1
#num_classes = 10
##batch_size = 50
#batch_size = 100
#num_epochs = 60
##num_epochs_per_decay = 5
##learning_rate = 1e-4
##learning_rate = 1e-2
##learning_rate = 1e-1
#learning_rate = 1e-3
#momentum = 0.9
#num_epochs_per_decay = 8
#lr_decay_factor = 0.5
##dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
#dataset = input_data.read_data_sets(DATA_DIR)
#train_x = dataset.train.images
#train_x = train_x.reshape([-1, 28, 28, 1])
#train_y = dataset.train.labels
#valid_x = dataset.validation.images
#valid_x = valid_x.reshape([-1, 28, 28, 1])
#valid_y = dataset.validation.labels
#test_x = dataset.test.images
#test_x = test_x.reshape([-1, 28, 28, 1])
#test_y = dataset.test.labels
#train_mean = train_x.mean()
#train_x -= train_mean
#valid_x -= train_mean
#test_x -= train_mean


# CIFAR
DATASET = 'CIFAR'
DATA_DIR = '/home/kivan/datasets/CIFAR-10/'
SAVE_DIR = '/home/kivan/source/out/cifar10/'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10
#batch_size = 50
batch_size = 100
num_epochs = 60
#num_epochs_per_decay = 5
#learning_rate = 1e-4
learning_rate = 1e-2
#learning_rate = 1e-1
#learning_rate = 1e-3
num_epochs_per_decay = 8
lr_decay_factor = 0.5

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)


train_mean = train_x.mean((0,1,2))
train_std = train_x.std((0,1,2))
valid_x = test_x
valid_y = test_y
#train_x = (train_x - data_mean)
#valid_x = (valid_x - data_mean)
train_x = (train_x - train_mean) / train_std
valid_x = (valid_x - train_mean) / train_std

print(train_x.shape)
print(valid_x.shape)
print(test_x.shape)




def check_grads(tf_vars, grads):
  for i, g in enumerate(grads):
    g = g.flatten()
    print(i, ' --> ', np.linalg.norm(np.dot(g.T, g)))
    #print(tf_vars[i].name, ' --> ', np.linalg.norm(g))


def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[3]
  num_channels = w.shape[2]
  k = w.shape[0]
  assert w.shape[0] == w.shape[1]
  w = w.reshape(k, k, num_channels, num_filters)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  if img.shape[2] == 1:
    img = img.reshape([img.shape[0], -1])
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  ski.io.imsave(os.path.join(save_dir, filename), img)



def fill_confusion_matrix(data_y, pred_y, conf_mat):
  data_size = data_y.shape[0]
  for i in range(data_size):
    conf_mat[data_y[i], pred_y[i]] += 1
   
def evaluate(run_ops, data_x, data_y):
  batch_size = 100
  num_examples = data_x.shape[0]
  assert(num_examples % batch_size == 0)
  num_batches = num_examples // batch_size
  conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
  loss_avg = 0
  for step in trange(num_batches):
    offset = step * batch_size 
    batch_x = data_x[offset:(offset + batch_size), ...]
    batch_y = data_y[offset:(offset + batch_size)]
    feed_dict = {node_x: batch_x, node_y: batch_y}
    #start_time = time.time()
    #run_ops = [loss, logits]
    loss, logits, mask = sess.run(run_ops, feed_dict=feed_dict)
    loss_avg += loss
    pred_y = np.argmax(logits, axis=1).astype(dtype=np.int32)
    fill_confusion_matrix(batch_y, pred_y, conf_mat)
    #duration = time.time() - start_time
    #if (step+1) % 50 == 0:
    #  sec_per_batch = float(duration)
    #  format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
    #  print(format_str % (epoch_num, step+1, num_batches, loss_val, sec_per_batch))
  #print(conf_mat)
  avg_accuracy = conf_mat.diagonal().sum() / conf_mat.sum()
  print("Class avg accuracy = ", avg_accuracy)
  tp_fp = conf_mat.sum(0)
  tp_fn = conf_mat.sum(1)
  class_precision = conf_mat.diagonal() / tp_fp
  class_recall = conf_mat.diagonal() / tp_fn
  avg_prec = class_precision.mean()
  avg_recall = class_recall.mean()
  #print('Precision:\n', class_precision, avg_prec, end='\n\n')
  #print('Recall:\n', class_recall, avg_recall, end='\n\n')
  loss_avg /= num_batches
  return loss_avg, avg_accuracy


def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def draw_image(img, mean, std, step, path):
  img *= std
  img += mean
  #img = img.reshape(img.shape[:-1]).astype(np.uint8)
  if DATASET == 'MNIST':
    img = img.reshape(img.shape[:-1])
  elif DATASET == 'CIFAR':
    img = img.astype(np.uint8)
  #ski.io.imshow(img)
  #ski.io.show()
  ski.io.imsave(path+'/'+ '%05d'%step + '_img.png', img)

def draw_mask(step, mask, save_dir):
  img = mask / mask.max()
  img = img.reshape(img.shape[:-1])
  ski.io.imsave(save_dir + '/'+ '%05d'%step +'_mask.png', img)




node_x = tf.placeholder(tf.float32, [None, img_height, img_width, num_channels])
#node_y = tf.placeholder(tf.float32, [None, num_classes])
node_y = tf.placeholder(tf.int32, [None])

#logits, loss = model.build_model(node_x, node_y, num_classes, is_training=True)

with tf.variable_scope('model'):
  train_ops = model.build(node_x, node_y, num_classes, is_training=True)
with tf.variable_scope('model', reuse=True):
  valid_ops = model.build(node_x, node_y, num_classes, is_training=False, reuse=True)
  #logits_eval, loss_eval = model.build(data_node, labels_node, is_training=False)

train_size = train_x.shape[0]
assert(train_size % batch_size == 0)
num_batches = train_size // batch_size
num_batches_per_epoch = train_size // batch_size
decay_steps = num_batches_per_epoch * num_epochs_per_decay

# ADAM, exponential decay
global_step = tf.Variable(0, trainable=False) 
#self.adam_rate = tf.train.exponential_decay(
#  1e-4, self.adam_step, 1, 0.9999)
# ADAM, vanilla
#learning_rate = tf.train.exponential_decay(1e-2, self.adam_step, 1, 0.9999)
#lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, lr_decay_factor, staircase=True)
lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, lr_decay_factor)
#trainer = tf.train.MomentumOptimizer(lr, momentum)
#trainer = tf.train.GradientDescentOptimizer(lr)
trainer = tf.train.AdamOptimizer(lr)
#train_op = trainer.minimize(loss, global_step=global_step) 
loss = train_ops[0]
grads_and_vars = trainer.compute_gradients(loss)
train_op = trainer.apply_gradients(grads_and_vars, global_step=global_step)
grads = [e[0] for e in grads_and_vars]

sess = tf.Session()
sess.run(tf.initialize_all_variables())

#variables = tf.contrib.framework.get_variables('model/conv1_1/weights:0')
#conv1w = variables[0]

#conv1_weights = conv1w.eval(session=sess)
#draw_conv_filters(0, 0, conv1_weights, SAVE_DIR)

#all_vars = tf.contrib.framework.get_variables()
#for v in all_vars:
#  print(v.name)
#grads = tf.gradients(loss, all_vars)
#print(grads)

best_acc = 0
plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []
for epoch_num in range(1, num_epochs + 1):
  train_x, train_y = shuffle_data(train_x, train_y)
  #indices = np.arange(train_size)
  #np.random.shuffle(indices)
  #train_x = np.ascontiguousarray(train_x[indices])
  #train_y = np.ascontiguousarray(train_y[indices])
  for step in range(num_batches):
  #for step in range(2):
    offset = step * batch_size 
    batch_x = train_x[offset:(offset + batch_size), ...]
    batch_y = train_y[offset:(offset + batch_size)]
    feed_dict = {node_x: batch_x, node_y: batch_y}
    start_time = time.time()
    #run_ops = train_ops + [train_op, global_step, lr, conv1w, grads]
    run_ops = train_ops + [train_op]
    ret_val = sess.run(run_ops, feed_dict=feed_dict)
    #loss_val, logits_val, mask_val, _, _, lr_val, conv1_weights, grad_vals = ret_val
    loss_val, logits_val, mask_val, _ = ret_val
    #print((mask_val < 0.1).sum())
    #print(mask_val)
    duration = time.time() - start_time
    if (step) % 30 == 0:
      #check_grads(grads, grad_vals)
      draw_image(batch_x[0], train_mean, train_std, step, SAVE_DIR)
      draw_mask(step, mask_val[0], SAVE_DIR)
      sec_per_batch = float(duration)
      format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
      print(format_str % (epoch_num, step, num_batches-1, loss_val, sec_per_batch))
    #if (step+1) % 500 == 0:
    #  draw_conv_filters(epoch_num, step, conv1_weights, SAVE_DIR)

  print('Train error:')
  train_loss, train_acc = evaluate(train_ops, train_x, train_y)
  print('Validation error:')
  valid_loss, valid_acc = evaluate(valid_ops, valid_x, valid_y)
  best_acc = max(best_acc, valid_acc)
  print('Best validation accuracy:', best_acc)
  plot_data['train_loss'] += [train_loss]
  plot_data['valid_loss'] += [valid_loss]
  plot_data['train_acc'] += [train_acc]
  plot_data['valid_acc'] += [valid_acc]
  plot_data['lr'] += [lr.eval(session=sess)]
  #eval_helper.plot_training_progress(SAVE_DIR, plot_data)
