import numpy as np
from cityscapes import CityscapesDataset


def convert_ids(img):
  img_train = np.zeros_like(img)
  img_train.fill(255)
  for i, cid in enumerate(CityscapesDataset.train_ids):
    img_train[img==cid] = i
  return img_train

def get_class_weights(gt_img, num_classes=19, max_wgt=100):
  height = gt_img.shape[0]
  width = gt_img.shape[1]
  weights = np.zeros((height, width), dtype=np.float32)
  num_labels = (gt_img >= 0).sum()
  for i in range(num_classes):
    mask = gt_img == i
    class_cnt = mask.sum()
    if class_cnt > 0:
      wgt = min(max_wgt, 1.0 / (class_cnt / num_labels))
      weights[mask] = wgt
      #print(i, wgt)
  return weights, num_labels
