import os
import numpy as np
import libs.cylib as cylib
import skimage as ski
import skimage.transform


def get_class_hist(gt_img, num_classes):
  hist = np.zeros(num_classes, dtype=np.int32)
  #hist = np.ones(num_classes, dtype=np.int32)
  #num_labels = (gt_img >= 0).sum()
  for i in range(num_classes):
    mask = gt_img == i
    hist[i] += mask.sum()
  num_labels = (gt_img < num_classes).sum()
  return hist, num_labels


def convert_colors_to_indices(rgb_img, color_map, max_wgt):
  height = rgb_img.shape[0]
  width = rgb_img.shape[1]
  #labels = np.empty((height, width), dtype=np.int32)
  labels = np.ascontiguousarray(np.empty((height, width), dtype=np.uint8))
  rgb_img = np.ascontiguousarray(rgb_img)
  num_classes = len(color_map) - 1
  #class_hist = np.zeros(num_classes)
  #class_hist = np.ones(num_classes, dtype=np.int64)
  class_hist = np.ascontiguousarray(np.zeros(num_classes, dtype=np.uint64))

  map_size = len(color_map)
  color_map_data = np.ascontiguousarray(np.zeros((map_size, 4), dtype=np.int32))
  #id_map_data = np.ascontiguousarray(np.zeros(map_size, dtype=np.int8))
  i = 0
  for key, value in color_map.items():
    for j in range(3):
      color_map_data[i,j] = key[j]
    color_map_data[i,3] = value[0]
    #id_map_data[i] = value[0]
    i += 1
  #print(color_map_data)
  #print(id_map_data)
  class_weights = np.ascontiguousarray(np.zeros(num_classes, dtype=np.float32))
  weights = np.ascontiguousarray(np.zeros((height, width), dtype=np.float32))
  cylib.convert_colors_to_ids(color_map_data, rgb_img, labels, class_hist, max_wgt,
                              class_weights, weights)
  num_labels = class_hist.sum()
  return labels, weights, int(num_labels), class_hist, class_weights


#def _get_level_location(loc, x, y, embedding_sizes):
#  px = int(round((x / out_width) * embedding_sizes[loc][0]))
#  py = int(round((y / out_height) * embedding_sizes[loc][1]))
#  if px < 1:
#    px = 1
#  elif py < 1:
#    py = 1
#  return px, py


def get_scale_selection_routing(depth, net_subsampling, depth_routing, embed_sizes,
                                filename, debug_save_dir):

  color_coding = [[0,0,0], [128,64,128], [244,35,232], [70,70,70], [102,102,156], [190,153,153],
                  [153,153,153], [250,170,30], [220,220,0]]
  height = depth.shape[0]
  width = depth.shape[1]
  height = height // net_subsampling
  width = width // net_subsampling
  #depth = ski.transform.resize(depth, (height, width), preserve_range=True, order=3)
  #print(depth.shape)
  #print(depth.min())
  #print(depth.max())
  #for i in range(129):
  #  depth_map[i] = 0
  num_scales = len(depth_routing)
  #routing_data = np.zeros((height, width, num_scales), dtype=np.uint8)
  routing_data = np.zeros((height, width, num_scales), dtype=np.int32)
  debug_img = np.ndarray((height, width, 3), dtype=np.uint8)

  level_offsets = [0]
    #level_offset = level_offsets[level]
  #level_offset = 0
  for level_res in embed_sizes[:-1]:
    level_offsets += [level_offsets[-1] + (level_res[0] * level_res[1])]
  #print(level_offsets)
  for s, routing in enumerate(depth_routing):
    for y in range(height):
      for x in range(width):
        d = int(round(depth[y,x]))
        #routing_data[s = depth_routing[s][d]
        #routing_data[r,c,s] = routing[d]
        level = routing[d]
        level_offset = level_offsets[level]
        #level_offset = 0
        level_width = embed_sizes[level][0]
        level_height = embed_sizes[level][1]
        px = int(round((x / width) * level_width))
        py = int(round((y / height) * level_height))
        routing_data[y,x,s] = level_offset + (py * level_width + px)
        debug_img[y,x] = color_coding[level]
        #print(level, py, px)
    ski.io.imsave(os.path.join(debug_save_dir, filename + '_' + str(s) + '.png'), debug_img)
  return routing_data


  #height = depth_img:size(1)
  #width = depth_img:size(2)
  #debug_img = {}
  #if debug_save_dir ~= nil then
  #  for i = 1, #scales do
  #    table.insert(debug_img, torch.ByteTensor(3, height, width):fill(0))
  #  end
  #end

  #local scale_routing = {}
  #for i = 1, #scales do
  #  table.insert(scale_routing, torch.IntTensor(height, width, 3):fill(0))
  #end
  #for i = 1, height do
  #  for j = 1, width do
  #    local d = depth_img[{i,j}]
  #    --print(d)
  #    for k = 1, #scales do
  #      local sf = (d * scales[k] / baseline) / rf_size
  #      --print(sf)
  #      local pyr_pos = GetPyramidPosition(sf)
  #      --print('pos = ', pyr_pos)
  #      --print(j,i)
  #      local x, y = GetLocationInPyramid(pyr_pos, j, i)
  #      --label_pyr[pyr_pos][{y,x}] = label_img[{i,j}]
  #      --print(pyr_pos, x,y)
  #      scale_routing[k][{i,j,1}] = pyr_pos
  #      scale_routing[k][{i,j,2}] = y
  #      scale_routing[k][{i,j,3}] = x
  #      if debug_save_dir ~= nil then
  #        for c = 1, 3 do
  #          debug_img[k][{c,i,j}] = color_coding[pyr_pos][c]
  #        end
  #      end
  #    end
  #  end
  #end
  #if debug_save_dir ~= nil then
  #  for i = 1, #scales do
  #    image.save(debug_save_dir .. filename:sub(1,-5) .. '_' .. i .. '.png', debug_img[i])
  #  end
  #end
  #return scale_routing



def convert_colors_to_indices_slow(rgb_img, color_map):
  height = rgb_img.shape[0]
  width = rgb_img.shape[1]
  labels = np.empty((height, width), dtype=np.int32)
  num_classes = len(color_map) - 1
  #class_hist = np.zeros(num_classes)
  #class_hist = np.ones(num_classes, dtype=np.int64)
  class_hist = np.zeros(num_classes, dtype=np.uint64)
  for i in range(height):
    for j in range(width):
      key = tuple(rgb_img[i,j])
      #print(key)
      class_id = color_map[key][0]
      labels[i,j] = class_id
      if class_id >= 0:
        class_hist[class_id] += 1
      #if key not in color_map:
      #  raise 1
  #print(labels.sum())
  num_labels = class_hist.sum()
  #weights = 1.0 / (class_hist / num_labels)
  weights = np.zeros(num_classes, dtype=np.float32)
  # TODO
  max_wgt = 1000
  #max_wgt = 100
  for i in range(num_classes):
    if class_hist[i] > 0:
      weights[i] = min(max_wgt, 1.0 / (class_hist[i] / num_labels))
    else:
      weights[i] = 0
  #print(weights)
  label_weights = np.zeros((height, width), dtype=np.float32)
  for i in range(height):
    for j in range(width):
      cidx = labels[i,j]
      if cidx >= 0:
        label_weights[i,j] = weights[cidx]
  #print(label_weights[200,500:800])
  return labels, label_weights, int(num_labels), class_hist
  #return labels.reshape(height * width), label_weights.reshape(height * width), num_labels
