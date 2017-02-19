import numpy as np

def read_tensor(path):
  fp = open(path, "rb")
  num_dims = int.from_bytes(fp.read(4), byteorder='little')
  assert(num_dims == 3)
  dims = [0] * num_dims
  for i in range(num_dims):
    dims[i] = int.from_bytes(fp.read(8), byteorder='little')
  # print(dims)
  # print(num_dims)
  tensor = np.ndarray(dims)
  # return repr
  data = fp.read()
  tensor = np.frombuffer(data, dtype=np.dtype('float32'))
  tensor = tensor.reshape(dims[0], dims[1], dims[2])
  return tensor


