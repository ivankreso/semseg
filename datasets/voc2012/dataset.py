import os
from os.path import join

class Dataset(object):
  class_info = [[128,64,128,   'background'],
                [244,35,232,   'aeroplane'],
                [70,70,70,     'bicycle'],
                [102,102,156,  'wall'],
                [190,153,153,  'fence'],
                [153,153,153,  'pole'],
                [250,170,30,   'traffic light'],
                [220,220,0,    'traffic sign'],
                [107,142,35,   'vegetation'],
                [152,251,152,  'terrain'],
                [70,130,180,   'sky'],
                [220,20,60,    'person'],
                [255,0,0,      'rider'],
                [0,0,142,      'car'],
                [0,0,70,       'truck'],
                [0,60,100,     'bus'],
                [0,80,100,     'train'],
                [0,0,230,      'motorcycle'],
                [0,0,230,      'motorcycle'],
                [0,0,230,      'motorcycle'],
                [119,11,32,    'bicycle']]

  def __init__(self, data_dir, subset):
    self.subset = subset
    filepath = join('/home/kivan/datasets/VOC2012/ImageSets/Segmentation', subset + '.txt')
    fp = open(filepath)
    filenames = [line.strip() + '.tfrecords' for line in fp.readlines()]
    self.data_dir = data_dir
    self.filenames = [os.path.join(self.data_dir, f) for f in filenames]

  def num_classes(self):
    return self.num_classes

  def num_examples(self):
    return len(self.filenames)

  def get_filenames(self):
    return self.filenames

