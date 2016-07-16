CLASS_INFO = [[128,128,128,  'sky'],
              [128,0,0,      'building'],
              [128,64,128,   'road'],
              [0,0,192,      'sidewalk'],
              [64,64,128,    'fence'],
              [128,128,0,    'vegetation'],
              [192,192,128,  'pole'],
              [64,0,128,     'car'], 
              [192,128,128,  'sign'],
              [64,64,0,      'pedestrian'],
              [0,128,192,    'cyclist']]

class_color_map = {}
# black is ignore
class_color_map[(0,0,0)] = [-1, 'ignore']
for i in range(len(CLASS_INFO)):
  r = CLASS_INFO[i][0]
  g = CLASS_INFO[i][1]
  b = CLASS_INFO[i][2]
  key = (r, g, b)
  class_color_map[key] = [i, CLASS_INFO[i][3]]


CITYSCAPES_TO_KITTI_MAP = [2, 3, 1, 1, 4, 6, 6, 8, 5, 5, 0, 9, 10, 7, 7, 7, 7, 10, 10]
