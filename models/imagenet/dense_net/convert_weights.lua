require 'nn'
require 'cudnn'

function write_tensor(tensor, path)
  print(path)
  --tensor = tensor:float()
  local file = torch.DiskFile(path..'', 'w'):binary()
  --print("Tensor size = ", tensor:nElement())
  local size = tensor:size()
  --print(size:size())
  file:writeInt(size:size())
  file:writeLong(size)
  --local data = tensor:storage()
  local t = tensor
  local data = t:storage().new(t:storage(), t:storageOffset(), t:nElement())
  print(size)
  assert(file:writeFloat(data) == t:nElement())
  file:close()
end

function dump_fc(layer, prefix)
  local weights = layer.weight
  weights = weights:transpose(1,2):contiguous()
  local path = prefix .. 'weights'
  write_tensor(weights, path)
  if layer.bias ~= nil then
    local path = prefix .. 'biases'
    write_tensor(layer.bias, path)
  end
end

function dump_conv(layer, prefix)
  local weights = layer.weight
  weights = weights:transpose(1,3):transpose(2,4):transpose(3,4):contiguous()
  local path = prefix .. 'Conv-weights'
  write_tensor(weights, path)
  if layer.bias ~= nil then
    print('Conv with bias')
    local path = prefix .. 'Conv-biases'
    write_tensor(layer.bias, path)
  end
end

function dump_batch_norm(layer, prefix)
  local mean = layer.running_mean
  local path = prefix .. 'BatchNorm-moving_mean'
  write_tensor(mean, path)
  local variance = layer.running_var
  local path = prefix .. 'BatchNorm-moving_variance'
  write_tensor(variance, path)
  local gamma = layer.weight
  local path = prefix .. 'BatchNorm-gamma'
  write_tensor(gamma, path)
  local beta = layer.bias
  local path = prefix .. 'BatchNorm-beta'
  write_tensor(beta, path)
end

depth = 121
path = '/home/kivan/datasets/pretrained/dense_net/torch/densenet-'..depth..'.t7'
save_dir = '/home/kivan/datasets/pretrained/dense_net/torch/' .. depth .. '/'
net = torch.load(path)
net = net:float()
print(net)

if depth == 121 then
  block_sizes = {6, 12, 24, 16}
end

conv0 = net:get(1)
dump_conv(conv0, save_dir..'conv0-')
bn0 = net:get(2)
dump_batch_norm(bn0, save_dir..'conv0-')
local cidx = 5
for i = 1,#block_sizes do
  local block = 'block'..(i-1)
  local bs = block_sizes[i]
  for j = 1,bs do
    local prefix = save_dir..block..'-layer'..(j-1)..'-'
    local l = net:get(cidx):get(2)
    dump_batch_norm(l:get(1), prefix..'bottleneck-')
    dump_conv(l:get(3), prefix..'bottleneck-')
    dump_batch_norm(l:get(4), prefix..'conv-')
    dump_conv(l:get(6), prefix..'conv-')
    cidx = cidx + 1
  end
  if i < #block_sizes then
    local prefix = save_dir..block..'-transition-'
    dump_batch_norm(net:get(cidx), prefix)
    dump_conv(net:get(cidx+2), prefix)
    cidx = cidx + 4
  end
end
print(cidx)
local prefix = save_dir..'head-'
dump_batch_norm(net:get(cidx), prefix)
cidx = cidx + 4
dump_fc(net:get(cidx), prefix..'fc1000-')

