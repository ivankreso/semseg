import subprocess

#img_name_pred = '0438839_frankfurt_000001_012738.png'
#img_name_pred = '0301615_lindau_000001_000019.png'
#img_name_pred = '0111584_lindau_000034_000019.png'
img_name_pred = '0252568_lindau_000054_000019.png'

img_prefix = img_name_pred[8:-4]
city = img_prefix.split('_')[0]

out_dir = '/home/kivan/tmp/'

cmd = 'gm convert /home/kivan/datasets/results/iccv2/cityscapes_75.75_25_7_14-03-35/evaluation/validation/' + \
      img_name_pred + ' -resize 1024x512 ' + out_dir + img_prefix + '_pred.png'
subprocess.run(cmd, shell=True)

cmd = 'gm convert /home/kivan/datasets/Cityscapes/2048x1024/rgb/val/' + city + '/' + img_prefix + '.ppm' + \
      ' -resize 1024x512 ' + out_dir + img_prefix + '_raw.png'
subprocess.run(cmd, shell=True)

cmd = 'gm convert /home/kivan/datasets/Cityscapes/2048x1024/gt_rgb/val/' + city + '/' + img_prefix + '.ppm' + \
      ' -resize 1024x512 ' + out_dir + img_prefix + '_gt.png'
subprocess.run(cmd, shell=True)
