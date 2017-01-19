import os
import time
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_progress(save_dir, train_data, valid_data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))
  linewidth = 2
  legend_size = 6
  title_size = 10
  train_color = 'm'
  val_color = 'c'

  train_loss = train_data['loss']
  valid_loss = valid_data['loss']
  #train_iou = plot_data['train_iou']
  valid_iou = valid_data['iou']
  #train_acc = plot_data['train_acc']
  valid_acc = valid_data['acc']
  lr = train_data['lr']
  x_data = np.linspace(1, len(train_loss), len(train_loss))
  ax1.set_title('cross entropy loss', fontsize=title_size)
  ax1.plot(x_data, train_loss, marker='o', color=train_color, linewidth=linewidth,
      linestyle='-', label='train')
  ax1.plot(x_data, valid_loss, marker='o', color=val_color, linewidth=linewidth, linestyle='-',
      label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('IoU accuracy')
  #ax2.plot(x_data, train_iou, marker='o', color=train_color, linewidth=linewidth,
  #linestyle='-', label='train')
  ax2.plot(x_data, valid_iou, marker='o', color=val_color, linewidth=linewidth,
      linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('pixel accuracy')
  #ax3.plot(x_data, train_acc, marker='o', color=train_color, linewidth=linewidth,
  #linestyle='-', label='train')
  ax3.plot(x_data, valid_acc, marker='o', color=val_color, linewidth=linewidth,
      linestyle='-', label='validation')
  ax3.legend(loc='upper left', fontsize=legend_size)

  ax4.set_title('learning rate')
  ax4.plot(x_data, lr, marker='o', color=train_color, linewidth=linewidth,
           linestyle='-', label='learning rate')
  ax4.legend(loc='upper right', fontsize=legend_size)

  #save_path = os.path.join(save_dir, 'training_plot.pdf')
  #print('Plotting in: ', save_path)
  #plt.savefig(save_path)
  plt.show()

save_dir = ''
train_data = {}
valid_data = {}
train_data['lr'] = [3, 2, 1]
train_data['loss'] = [4, 3, 1]
valid_data['loss'] = [6, 4, 2]
valid_data['iou'] = [4, 3, 1]
valid_data['acc'] = [4, 3, 1]
plot_training_progress(save_dir, train_data, valid_data)
