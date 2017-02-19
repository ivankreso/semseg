from torch.utils.serialization import load_lua

path = '/home/kivan/datasets/pretrained/dense_net/densenet-121.t7'
x = load_lua(path, unknown_classes=True)
print(x.get(1))
