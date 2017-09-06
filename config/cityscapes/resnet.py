import os
import tensorflow as tf
import train_helper

MODEL_PATH = './models/cityscapes/resnet.py'
#MODEL_PATH = './models/cityscapes/dense_net_multiplexer.py'
#MODEL_PATH = './models/cityscapes/dense_net_concat_all.py'
#MODEL_PATH = './models/cityscapes/dense_net_jitter.py'
#MODEL_PATH = './models/cityscapes/dense_net_fix_bn.py'
#MODEL_PATH = './models/cityscapes/dense_net_dilated.py'

#IMG_WIDTH, IMG_HEIGHT = 2048, 1024
IMG_WIDTH, IMG_HEIGHT = 1024, 448
#IMG_WIDTH, IMG_HEIGHT = 768, 320

SAVE_DIR = os.path.join('/home/kivan/datasets/results/tmp/cityscapes',
                        train_helper.get_time_string())

#DATASET_DIR = '/home/kivan/datasets/Cityscapes/tensorflow/2048x1024/'


#~/datasets/Cityscapes/tensorflow/1024x448_pyramid

#IMG_WIDTH, IMG_HEIGHT = 2048, 896
#DATASET_DIR = '/home/kivan/datasets/Cityscapes/tensorflow/2048x1024_nohood'

#tf.app.flags.DEFINE_string('optimizer', 'Adam', '')
# best
#tf.app.flags.DEFINE_float('initial_learning_rate', 4e-4, '')


#tf.app.flags.DEFINE_float('initial_learning_rate', 5e-4, '')
#tf.app.flags.DEFINE_float('initial_learning_rate', 8e-4, '')
# 68.30
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, '')

#tf.app.flags.DEFINE_float('fine_lr_div', 5, '')
#tf.app.flags.DEFINE_float('fine_lr_div', 7, '')
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 5, '')
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 4, '')
#fix

# SGD
# 1e-3 to small
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-2, '')
#tf.app.flags.DEFINE_float('initial_learning_rate', 4e-2, '')
#tf.app.flags.DEFINE_float('fine_lr_div', 10, '')
# 1.5% bolje
#tf.app.flags.DEFINE_float('fine_lr_div', 5, '')
#tf.app.flags.DEFINE_integer('max_epochs', 8, 'Number of epochs to run.')
#tf.app.flags.DEFINE_integer('max_epochs', 40, 'Number of epochs to run.')

# 30 1% better then 20 on adam


# SPP has 90K

#tf.app.flags.DEFINE_integer('num_iters', 30000, '')
tf.app.flags.DEFINE_integer('num_iters', 20000, '')
#tf.app.flags.DEFINE_integer('num_iters', 60000, '')
tf.app.flags.DEFINE_integer('max_num_epochs', 100, 'Number of epochs to run.')

###tf.app.flags.DEFINE_integer('num_iters', 20000, '')
tf.app.flags.DEFINE_string('optimizer', 'adam', '')
tf.app.flags.DEFINE_float('decay_power', 1.5, '')
#tf.app.flags.DEFINE_float('decay_power', 1.4, '')
tf.app.flags.DEFINE_float('initial_learning_rate', 5e-4, '')

#tf.app.flags.DEFINE_integer('max_epochs', 40, 'Number of epochs to run.')
#tf.app.flags.DEFINE_string('optimizer', 'momentum', '')
#tf.app.flags.DEFINE_float('decay_power', 0.9, '')
#tf.app.flags.DEFINE_float('initial_learning_rate', 3e-2, '')

tf.app.flags.DEFINE_integer('max_weight', 1, '')

#tf.app.flags.DEFINE_string('optimizer', 'Momentum', '')
##tf.app.flags.DEFINE_float('initial_learning_rate', 2e-4,
## 1e-2 is to big
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, """Initial learning rate.""")
#tf.app.flags.DEFINE_float('momentum', 0.9, '')
#tf.app.flags.DEFINE_float('num_epochs_per_decay', 3.0,
#                          """Epochs after which learning rate decays.""")

## 69.45
##tf.app.flags.DEFINE_float('initial_learning_rate', 7e-4, '')
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, '')
##tf.app.flags.DEFINE_float('initial_learning_rate', 3e-4, '')
## adam
##tf.app.flags.DEFINE_float('initial_learning_rate', 4e-4, '')
##tf.app.flags.DEFINE_integer('num_epochs_per_decay', 1, '')
##tf.app.flags.DEFINE_integer('num_epochs_per_decay', 6, '')
##tf.app.flags.DEFINE_integer('num_epochs_per_decay', 8, '')
##tf.app.flags.DEFINE_boolean('staircase', True, '.')
## 10 0.5% better then 1
##tf.app.flags.DEFINE_integer('max_weight', 10, '')

#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, '')
#tf.app.flags.DEFINE_float('fine_lr_div', 10, '') bad
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 4, '')
#tf.app.flags.DEFINE_boolean('staircase', False, '.')

# 1e-3 to big?
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, '')
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 4, '')
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 3, '')
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, '')
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 4, '')
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 7, '')
#tf.app.flags.DEFINE_float('initial_learning_rate', 3e-4, '')
# TODO better 4?
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 13, '')

# 1-4
#tf.app.flags.DEFINE_integer('batch_size', 1, '')
#tf.app.flags.DEFINE_integer('batch_size', 2, '')
tf.app.flags.DEFINE_integer('batch_size', 4, '')
tf.app.flags.DEFINE_integer('batch_size_valid', 2, '')
tf.app.flags.DEFINE_integer('num_validations_per_epoch', 1, '')


#tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
#tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.2,
#                          """Learning rate decay factor.""")
#tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, '')

tf.app.flags.DEFINE_string('resume_path', '', '')
#tf.app.flags.DEFINE_string('resume_path',
#    '/home/kivan/datasets/results/iccv/05_11_3_08-57-48/model.ckpt', '')

#povecaj_lr za w=1
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', SAVE_DIR, \
    """Directory where to write event logs and checkpoint.""")

tf.app.flags.DEFINE_integer('img_width', IMG_WIDTH, '')
tf.app.flags.DEFINE_integer('img_height', IMG_HEIGHT, '')
tf.app.flags.DEFINE_integer('img_depth', 3, '')

tf.app.flags.DEFINE_string('model_path', MODEL_PATH, '')
tf.app.flags.DEFINE_string('debug_dir', os.path.join(SAVE_DIR, 'debug'), '')
tf.app.flags.DEFINE_integer('num_classes', 19, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_boolean('draw_predictions', False, 'Whether to draw.')
tf.app.flags.DEFINE_boolean('save_net', True, 'Whether to save.')
tf.app.flags.DEFINE_boolean('no_valid', False, '')


tf.app.flags.DEFINE_integer('seed', 66478, '')

