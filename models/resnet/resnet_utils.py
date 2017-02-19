import os, re
import tensorflow as tf

def minimize(opts, loss, global_step, prefix):
  all_vars = tf.trainable_variables()
  grads = tf.gradients(loss, all_vars)
  resnet_grads_and_vars = []
  head_grads_and_vars = []
  for i, v in enumerate(all_vars):
    if v.name[:4] == prefix:
      print(v.name, ' --> lr*10')
      head_grads_and_vars += [(grads[i], v)]
    else:
      resnet_grads_and_vars += [(grads[i], v)]
  train_op1 = opts[0].apply_gradients(resnet_grads_and_vars, global_step=global_step)
  train_op2 = opts[1].apply_gradients(head_grads_and_vars)
  return tf.group(train_op1, train_op2)


def create_init_op(params):
  variables = tf.contrib.framework.get_variables()
  init_map = {}
  for var in variables:
    #name_split = var.name.split('/')
    #if len(name_split) != 3:
    #  continue
    #name = name_split[1] + '/' + name_split[2][:-2]
    name = var.name
    if name in params:
      #print(var.name, ' --> found init')
      init_map[var.name] = params[name]
      del params[name]
    else:
      print(var.name, ' --> init not found!')
      #raise 1
  print(list(params.keys()))
  #print(params['conv0/biases:0'].sum())
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
  return init_op, init_feed

def name_conversion(caffe_layer_name, prefix=''):
  """ Convert a caffe parameter name to a tensorflow parameter name as
      defined in the above model """
  # beginning & end mapping
  NAME_MAP = {
      'bn_conv1/beta': 'conv0/BatchNorm/beta:0',
      'bn_conv1/gamma': 'conv0/BatchNorm/gamma:0',
      'bn_conv1/mean/EMA': 'conv0/BatchNorm/moving_mean:0',
      'bn_conv1/variance/EMA': 'conv0/BatchNorm/moving_variance:0',
      'conv1/W': 'conv0/weights:0', 'conv1/b': 'conv0/biases:0',
      'fc1000/W': 'fc1000/weights:0', 'fc1000/b': 'fc1000/biases:0'}
  if caffe_layer_name in NAME_MAP:
    return prefix + NAME_MAP[caffe_layer_name]

  s = re.search('([a-z]+)([0-9]+)([a-z]+)_', caffe_layer_name)
  if s is None:
    s = re.search('([a-z]+)([0-9]+)([a-z]+)([0-9]+)_', caffe_layer_name)
    layer_block_part1 = s.group(3)
    layer_block_part2 = s.group(4)
    assert layer_block_part1 in ['a', 'b']
    layer_block = 0 if layer_block_part1 == 'a' else int(layer_block_part2)
  else:
    layer_block = ord(s.group(3)) - ord('a')
  layer_type = s.group(1)
  layer_group = s.group(2)

  layer_branch = int(re.search('_branch([0-9])', caffe_layer_name).group(1))
  assert layer_branch in [1, 2]
  if layer_branch == 2:
    layer_id = re.search('_branch[0-9]([a-z])/', caffe_layer_name).group(1)
    layer_id = ord(layer_id) - ord('a') + 1

  TYPE_DICT = {'res':'conv', 'bn':'BatchNorm'}
  name_map = {'/W': '/weights:0', '/b': '/biases:0', '/beta': '/beta:0',
              '/gamma': '/gamma:0', '/mean/EMA': '/moving_mean:0',
              '/variance/EMA': '/moving_variance:0'}

  tf_name = caffe_layer_name[caffe_layer_name.index('/'):]
  #print(tf_name)
  if tf_name in name_map:
    tf_name = name_map[tf_name]
  #print(layer_type)
  #if layer_type != 'bn':
  if layer_type == 'res':
    layer_type = TYPE_DICT[layer_type] + (str(layer_id)
        if layer_branch == 2 else 'shortcut')
  elif layer_branch == 2:
    layer_type = 'conv' + str(layer_id) + '/' + TYPE_DICT[layer_type]
  elif layer_branch == 1:
    layer_type = 'convshortcut/' + TYPE_DICT[layer_type]
  tf_name = 'group{}/block{}/{}'.format(int(layer_group) - 2,
      layer_block, layer_type) + tf_name
  return prefix + tf_name


