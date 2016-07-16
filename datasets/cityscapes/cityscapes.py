import os
from datasets.dataset import Dataset

def _create_class_data(class_info):
  CLASS_COLOR_MAP = {}
  # black is ignore
  CLASS_COLOR_MAP[(0,0,0)] = [255, 'ignore']
  for i, item in enumerate(class_info):
    r = item[0]
    g = item[1]
    b = item[2]
    key = (r, g, b)
    CLASS_COLOR_MAP[key] = [i, item[3]]
  return CLASS_COLOR_MAP

class CityscapesDataset(Dataset):
  CLASS_INFO = [[128,64,128,   'road'],
                [244,35,232,   'sidewalk'],
                [70,70,70,     'building'],
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
                [119,11,32,    'bicycle']]

  CLASS_COLOR_MAP = _create_class_data(CLASS_INFO)
  NUM_CLASSES = 19

  #@staticmethod
  #def class_info():
  #  return CLASS_INFO

  def __init__(self, data_dir, subset):
    super(CityscapesDataset, self).__init__(data_dir, subset)
    self.num_classes = CityscapesDataset.NUM_CLASSES


