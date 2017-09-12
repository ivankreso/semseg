import os
import xml.etree.ElementTree as ET

from tqdm import trange
import numpy as np
import h5py
import skimage as ski
import skimage.data
import skimage.transform
import skimage.color


DATA_DIR = '/home/kivan/datasets/imagenet/ILSVRC2015/'
save_path = os.path.join(DATA_DIR, 'numpy', 'val_data.hdf5')
#class_info_path = os.path.join(DATA_DIR, 'devkit/data/map_clsloc.txt')
#fp = open(class_info_path, 'r')
#class_info = []
#class_id_map = {}
#for line in fp:
#  lst = line.strip().split(' ')
#  class_id_map[lst[0]] = int(lst[1]) - 1
#  class_info += [lst[0], int(lst[1]), lst[2]]
imglist_fp = open('/home/kivan/datasets/imagenet/ILSVRC2015/caffe/val.txt', 'r')
#fp = open(class_info_path, 'r')
img_list = []
img_class = []
#data_y = np.ndarray((0, 1), dtype=np.int32)
for line in imglist_fp:
  lst = line.strip().split(' ')
  #print(lst)
  img_list += [lst[0]]
  img_class += [int(lst[1])]
  #data_y = np.vstack((data_y, int(lst[1])))
N = len(img_list)
data_y = np.zeros((N), dtype=np.int32)
for i, c in enumerate(img_class):
  data_y[i] = c
#data_y = data_y.reshape((data_y.shape[0]))

img_dir = os.path.join(DATA_DIR, 'Data/CLS-LOC/val')
labels_dir = os.path.join(DATA_DIR, 'Annotations/CLS-LOC/val/')

#img_list = next(os.walk(img_dir))[2]

resize_size = 256
crop_size = 224

N = len(img_list)
#N = 2000
#data_x = np.ndarray((0, img_size, img_size, 3), dtype=np.uint8)
#data_x = np.ndarray((len(img_list), img_size, img_size, 3), dtype=np.uint8)
data_x = np.zeros((N, crop_size, crop_size, 3), dtype=np.uint8)
#data_x = np.zeros((100, img_size, img_size, 3), dtype=np.uint8)
#data_y = np.zeros((100), dtype=np.int32)
#data_y = []
#data_imgs = []

import cv2
for i in trange(N):
  img_name = img_list[i]
  #img_prefix = img_name[:-4]
  rgb_path = os.path.join(img_dir, img_name)
  #print(rgb_path)
  #rgb = ski.data.load(rgb_path)
  #if len(rgb.shape) == 2:
  #  rgb = ski.color.gray2rgb(rgb)
  rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
  height = rgb.shape[0]
  width = rgb.shape[1]
  if height > width:
    new_w = resize_size
    new_h = int(round(resize_size * (height/width)))
  else:
    new_h = resize_size
    new_w = int(round(resize_size * (width/height)))
  height, width = new_h, new_w

  rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
  # slow :(
  #rgb = ski.transform.resize(rgb, (new_h, new_w), preserve_range=True, order=3)
  assert resize_size > crop_size
  ys = int((height - crop_size) * 0.5)
  xs = int((width - crop_size) * 0.5)
  rgb = np.ascontiguousarray(rgb[ys:ys+crop_size,xs:xs+crop_size,:])

  #if height > width:
  #  cy_start = int(round((height / 2) - (img_size / 2)))
  #  cy_end = cy_start + img_size
  #  rgb = np.ascontiguousarray(rgb[cy_start:cy_end,:,:])
  #else:
  #  cx_start = int(round((width / 2) - (img_size / 2)))
  #  cx_end = cx_start + img_size
  #  rgb = np.ascontiguousarray(rgb[:,cx_start:cx_end,:])
  #rgb = rgb.astype(np.uint8)

  #data_x = np.vstack((data_x, rgb.reshape([-1] + list(rgb.shape))))
  #data_x[i,...] = rgb.reshape([-1] + list(rgb.shape))
  data_x[i] = rgb
  #data_imgs += [img_name]

  #label_path = os.path.join(labels_dir, img_name[:-4]+'xml')
  #tree = ET.parse(label_path)
  #root = tree.getroot()
  #class_id = root[5][0].text
  #data_y += [class_id_map[class_id]]
  #data_y = np.vstack((data_y, class_id_map[class_id]))
  #print(rgb_path, ' - ', class_id_map[class_id])

h5f = h5py.File(save_path, 'w')
h5f.create_dataset('data_x', data=data_x)
h5f.create_dataset('data_y', data=data_y)
#dt = h5py.special_dtype(vlen=bytes)
dt = h5py.special_dtype(vlen=str)
img_list = np.array(img_list, dtype=object)
h5f.create_dataset('img_names', data=img_list, dtype=dt)
h5f.close()

#data = [data_x, data_y, data_imgs, class_info, class_id_map]
#data = [data_x, data_y, img_list]
#data = [data_x, data_y]
#np.save(save_path, data)
#np.save(save_path, data_x)

