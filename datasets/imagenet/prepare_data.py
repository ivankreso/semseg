import os
import xml.etree.ElementTree as ET

from tqdm import trange
import numpy as np
import skimage as ski
import skimage.data
import skimage.transform
import skimage.color


DATA_DIR = '/home/kivan/datasets/imagenet/ILSVRC2015/'
class_info_path = os.path.join(DATA_DIR, 'devkit/data/map_clsloc.txt')

img_dir = os.path.join(DATA_DIR, 'Data/CLS-LOC/val')
labels_dir = os.path.join(DATA_DIR, 'Annotations/CLS-LOC/val/')

img_list = next(os.walk(img_dir))[2]

img_size = 224

data_x = np.ndarray((0, img_size, img_size, 3), dtype=np.uint8)
data_y = []
data_imgs = []
#for i in range(1, 6):
#  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
#  train_y += subset['labels']
#train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
#train_y = np.array(train_y, dtype=np.int32)
#np.save(train_x, )


for i in trange(len(img_list)):
  img_name = img_list[i]
  #img_prefix = img_name[:-4]
  rgb_path = os.path.join(img_dir, img_name)
  #print(rgb_path)
  rgb = ski.data.load(rgb_path)
  if len(rgb.shape) == 2:
    rgb = ski.color.gray2rgb(rgb)
  height = rgb.shape[0]
  width = rgb.shape[1]
  if height > width:
    new_w = img_size
    new_h = int(round(img_size * (height/width)))
  else:
    new_h = img_size
    new_w = int(round(img_size * (width/height)))
  height, width = new_h, new_w
  rgb = ski.transform.resize(rgb, (new_h, new_w), preserve_range=True, order=3)
  if height > width:
    cy_start = int(round((height / 2) - (img_size / 2)))
    cy_end = cy_start + img_size
    rgb = np.ascontiguousarray(rgb[cy_start:cy_end,:,:])
  else:
    cx_start = int(round((width / 2) - (img_size / 2)))
    cx_end = cx_start + img_size
    rgb = np.ascontiguousarray(rgb[:,cx_start:cx_end,:])
  rgb = rgb.astype(np.uint8)

  data_x = np.vstack((data_x, rgb.reshape([-1] + list(rgb.shape))))
  data_imgs += [img_name]

  label_path = os.path.join(labels_dir, img_name[:-4]+'xml')
  tree = ET.parse(label_path)
  root = tree.getroot()
  print(root[5][0].text)
  #for child in root:
    #print(child.tag, child.attrib)
  #print(root)

#np.save(train_x)
