from datetime import datetime
import time
import tensorflow as tf


class Logger(object):
  def __init__(self, *files):
    self.files = files
  def write(self, obj):
    for f in self.files:
      f.write(obj)
      f.flush() # If you want the output to be visible immediately
  def flush(self) :
    for f in self.files:
      f.flush()


def get_variable_map():
  var_list = tf.all_variables()
  var_map = {}
  for var in var_list:
    var_map[var.name] = var
    #print(var.name)
  return var_map


def get_time_string():
  time = datetime.now()
  name = str(time.day) + '_' + str(time.month) + '_%02d' % time.hour + '-%02d' % time.minute + \
         '-%02d' % time.second + '/'
  return name


def get_time():
  time = datetime.now()
  return '%02d' % time.hour + ':%02d' % time.minute + ':%02d' % time.second


def get_expired_time(start_time):
  curr_time = time.time()
  delta = curr_time - start_time
  hour = int(delta / 3600)
  delta -= hour * 3600
  minute = int(delta / 60)
  delta -= minute * 60
  seconds = delta
  return '%02d' % hour + ':%02d' % minute + ':%02d' % seconds

def put_kernels_on_grid(kernel, grid_Y, grid_X, pad=1):
  '''Visualize conv. features as an image (mostly for the 1st layer).
  Place kernel into a grid, with some paddings between adjacent filters.

  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                         User is responsible of how to break into two multiples.
    pad:               number of black pixels around each filter (between them)
  
  Return:
    Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
  '''
  # pad X and Y
  x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + pad
  X = kernel.get_shape()[1] + pad

  # put NumKernels to the 1st dimension
  x2 = tf.transpose(x1, (3, 0, 1, 2))
  # organize grid on Y axis
  x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 3]))
  
  # switch X and Y axes
  x4 = tf.transpose(x3, (0, 2, 1, 3))
  # organize grid on X axis
  x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 3]))
  
  # back to normal order (not combining with the next step for clarity)
  x6 = tf.transpose(x5, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x7 = tf.transpose(x6, (3, 0, 1, 2))

  # scale to [0, 1]
  x_min = tf.reduce_min(x7)
  x_max = tf.reduce_max(x7)
  x8 = (x7 - x_min) / (x_max - x_min)

  return x8


