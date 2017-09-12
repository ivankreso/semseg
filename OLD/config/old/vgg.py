import os
import tensorflow as tf
import train_helper

#MODEL_PATH = './models/vgg.py'
MODEL_PATH = './models/all_dilated.py'
#MODEL_PATH = './models/ladder_net.py'
#MODEL_PATH = './models/vgg_multiblock.py'
#MODEL_PATH = './models/vgg_multiblock_depth.py'

#MODEL_PATH = './models/six_blocks.py'
#MODEL_PATH = './models/dilated_multiscale.py'
#MODEL_PATH = './models/baseline.py'
#MODEL_PATH = './models/dilated_conv.py'
#MODEL_PATH = './models/multiscale_pyramid.py'
#MODEL_PATH = './models/resnet.py'
SAVE_DIR = os.path.join('/home/kivan/source/results/semseg/tf/nets', train_helper.get_time_string())

#IMG_WIDTH = 1152
#IMG_HEIGHT = 1024
#DATASET_DIR = '/home/kivan/datasets/Cityscapes/tensorflow/2048x1024/'

IMG_WIDTH = 640
IMG_HEIGHT = 288
#IMG_WIDTH = 1024
#IMG_HEIGHT = 448
DATASET_DIR = os.path.join('/home/kivan/datasets/Cityscapes/tensorflow/',
                           '{}x{}'.format(IMG_WIDTH, IMG_HEIGHT))
#DATASET_DIR = os.path.join('/home/kivan/datasets/Cityscapes/tensorflow/',
#                           '{}x{}_multiscale/'.format(IMG_WIDTH, IMG_HEIGHT))


tf.app.flags.DEFINE_string('optimizer', 'Adam', '')
# 1e-4 best, 1e-3 is too big
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4, '')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 2, '')

#tf.app.flags.DEFINE_string('optimizer', 'Momentum', '')
##tf.app.flags.DEFINE_float('initial_learning_rate', 2e-4,
## 1e-2 is to big
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, """Initial learning rate.""")
#tf.app.flags.DEFINE_float('momentum', 0.9, '')
#tf.app.flags.DEFINE_float('num_epochs_per_decay', 3.0,
#                          """Epochs after which learning rate decays.""")

tf.app.flags.DEFINE_integer('num_validations_per_epoch', 2, '')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
#tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.2,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, '')

#povecaj_lr za w=1
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', SAVE_DIR, \
    """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('resume_path', '', '')
tf.app.flags.DEFINE_integer('img_width', IMG_WIDTH, '')
tf.app.flags.DEFINE_integer('img_height', IMG_HEIGHT, '')
tf.app.flags.DEFINE_integer('img_depth', 3, '')
tf.app.flags.DEFINE_integer('net_subsampling', 16, '')

tf.app.flags.DEFINE_string('model_path', MODEL_PATH, '')
tf.app.flags.DEFINE_string('dataset_dir', DATASET_DIR, '')
tf.app.flags.DEFINE_string('debug_dir', os.path.join(SAVE_DIR, 'debug'), '')
tf.app.flags.DEFINE_string('vgg_init_dir', '/home/kivan/datasets/pretrained/vgg16/', '')
#tf.app.flags.DEFINE_integer('max_steps', 100000,
#                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_epochs', 30, 'Number of epochs to run.')
tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_integer('num_classes', 19, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_boolean('draw_predictions', True, 'Whether to draw.')
tf.app.flags.DEFINE_boolean('save_net', True, 'Whether to save.')

tf.app.flags.DEFINE_integer('seed', 66478, '')

