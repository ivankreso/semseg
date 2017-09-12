import time

dim1 = 3
dim2 = 130
num_iters = int(1e4)

import numpy as np
data = np.random.randn(dim1, dim2)

data_lst = []
for i in range(dim1):
  data_lst += [[]]
  for j in range(dim2):
    data_lst[i] += [i * j / 2]
print(len(data_lst), len(data_lst[0]))

start = time.time()
for n in range(num_iters):
  for i in range(dim1):
    for j in range(dim2):
      val = data[i,j]
end = time.time()
print('Numpy = ', end - start)

start = time.time()
for n in range(num_iters):
  for i in range(dim1):
    for j in range(dim2):
      val = data_lst[i][j]
end = time.time()
print('List = ', end - start)
