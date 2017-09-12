from os import listdir
from os.path import isfile, join
import pickle
import numpy as np

def read_tensor(path):
  fp = open(path, "rb")
  num_dims = int.from_bytes(fp.read(4), byteorder='little')
  #assert(num_dims == 3)
  dims = []
  for i in range(num_dims):
    dims.append(int.from_bytes(fp.read(8), byteorder='little'))
  # print(dims)
  # print(num_dims)
  tensor = np.ndarray(dims)
  # return repr
  data = fp.read()
  tensor = np.frombuffer(data, dtype=np.dtype('float32'))
  #tensor = tensor.reshape(dims[0], dims[1], dims[2])
  tensor = tensor.reshape(dims)
  return tensor


#model = 121
model = 169
root_path = '/home/kivan/datasets/pretrained/dense_net/'
dir_path = root_path + 'torch/' + str(model)
filelist = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

init_map = {}
for filename in filelist:
  t = read_tensor(join(dir_path, filename))
  print(filename)
  varname = filename.replace('-', '/') + ':0'
  print(varname)
  print(t.shape)
  init_map[varname] = t

#np.save(join(root_path, 'dense_net_'+str(model)+'.npy'), init_map)
save_path = join(root_path, 'dense_net_'+str(model)+'.pickle')
with open(save_path, 'wb') as f:
  pickle.dump(init_map, f)
