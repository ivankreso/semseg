#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


graph_data = np.load('gcpr_graphs_data.npy')

# 4 grafa
color_4096_range = np.arange(0, 6)
color_1024_range = np.arange(6, 12)
sift_4096_range = np.arange(12, 18)
sift_1024_range = np.arange(18, 24)

# 3 stvari na grafu
speedups_id = 0
dot_id = 1
dot_n_id = 2

# npr. za 1. graf podaci su
speedups_color_4096 = np.array(graph_data[speedups_id])[color_4096_range]
dot_color_4096 = np.array(graph_data[dot_id])[color_4096_range]
dotn_color_4096 = np.array(graph_data[dot_n_id])[color_4096_range]

speedups_color_4096 = speedups_color_4096[::-1]
dot_color_4096 = dot_color_4096[::-1]
dotn_color_4096 = dotn_color_4096[::-1]

speedups_color_1024 = np.array(graph_data[speedups_id])[color_1024_range]
dot_color_1024 = np.array(graph_data[dot_id])[color_1024_range]
dotn_color_1024 = np.array(graph_data[dot_n_id])[color_1024_range]

speedups_color_1024 = speedups_color_1024[::-1]
dot_color_1024 = dot_color_1024[::-1]
dotn_color_1024 = dotn_color_1024[::-1]

speedups_sift_4096 = np.array(graph_data[speedups_id])[sift_4096_range]
dot_sift_4096 = np.array(graph_data[dot_id])[sift_4096_range]
dotn_sift_4096 = np.array(graph_data[dot_n_id])[sift_4096_range]

speedups_sift_4096 = speedups_sift_4096[::-1]
dot_sift_4096 = dot_sift_4096[::-1]
dotn_sift_4096 = dotn_sift_4096[::-1]

speedups_sift_1024 = np.array(graph_data[speedups_id])[sift_1024_range]
dot_sift_1024 = np.array(graph_data[dot_id])[sift_1024_range]
dotn_sift_1024 = np.array(graph_data[dot_n_id])[sift_1024_range]

speedups_sift_1024 = speedups_sift_1024[::-1]
dot_sift_1024 = dot_sift_1024[::-1]
dotn_sift_1024 = dotn_sift_1024[::-1]

line_thickness = 2
dot_color='m'
speed_color='c'

fig, ((ax1, ax3), (ax4, ax5)) = plt.subplots(2, 2)

for tl in ax1.get_yticklabels():
    tl.set_color(dot_color)
for tl in ax3.get_yticklabels():
    tl.set_color(dot_color)
for tl in ax4.get_yticklabels():
    tl.set_color(dot_color)
for tl in ax5.get_yticklabels():
    tl.set_color(dot_color)

ax3.set_yticks(np.arange(0, 1.1, 0.1))
ax3.set_xlim([-0.5, 5.5])
ax3.set_ylim([0, 1.1])
ax3.set_xlabel('k_t')
ax3.set_ylabel('dot_product', color=dot_color)
ax3.set_title('color_4096')
ax3.set_xticklabels(['2', '4', '8', '16', '32', '64'])
ax3.plot(color_4096_range, dot_color_4096[:, 0], marker='o', color=dot_color, linewidth=line_thickness, linestyle='--')
ax3.plot(color_4096_range, dotn_color_4096[:, 0], marker='o', color=dot_color, linewidth=line_thickness)

ax2 = ax3.twinx()
ax2.set_xticks(np.arange(0, 7, 1))
ax2.set_yticks(np.arange(0, 41, 5))
ax2.set_xlim([-0.5, 5.5])
ax2.set_ylabel('speedup', color=speed_color)
for tl in ax2.get_yticklabels():
    tl.set_color(speed_color)
ax2.set_ylim([0, 41])
ax2.set_xticklabels(['2', '4', '8', '16', '32', '64'])
ax2.plot(color_4096_range, speedups_color_4096[:, 0], marker='o', color=speed_color, linewidth=line_thickness)

plt.grid()

ax1.set_title('color_1024')
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax1.set_xlim([-0.5, 5.5])
ax1.set_ylim([0, 1.1])
ax1.set_xlabel('k_t')
ax1.set_ylabel('dot_product', color=dot_color)
ax1.set_xticklabels(['2', '4', '8', '16', '32', '64'])
ax1.plot(color_4096_range, dot_color_1024[:, 0], marker='o', color=dot_color, linewidth=line_thickness, linestyle='--')
ax1.plot(color_4096_range, dotn_color_1024[:, 0], marker='o', color=dot_color, linewidth=line_thickness)

ax2 = ax1.twinx()
ax2.set_xticks(np.arange(0, 7, 1))
ax2.set_yticks(np.arange(0, 41, 5))
ax2.set_xlim([-0.5, 5.5])
ax2.set_ylabel('speedup', color=speed_color)
for tl in ax2.get_yticklabels():
    tl.set_color(speed_color)
ax2.set_ylim([0, 41])
ax2.set_xticklabels(['2', '4', '8', '16', '32', '64'])
ax2.plot(color_4096_range, speedups_color_1024[:, 0], marker='o', color=speed_color, linewidth=line_thickness)

plt.grid()

ax5.set_title('sift_4096')
ax5.set_yticks(np.arange(0, 1.1, 0.1))
ax5.set_xlim([-0.5, 5.5])
ax5.set_ylim([0, 1.1])
ax5.set_xlabel('k_t')
ax5.set_ylabel('dot_product', color=dot_color)
ax5.set_xticklabels(['2', '4', '8', '16', '32', '64'])
ax5.plot(color_4096_range, dot_sift_4096[:, 0], marker='o', color=dot_color, linewidth=line_thickness, linestyle='--')
ax5.plot(color_4096_range, dotn_sift_4096[:, 0], marker='o', color=dot_color, linewidth=line_thickness)

ax2 = ax5.twinx()
ax2.set_xticks(np.arange(0, 7, 1))
ax2.set_yticks(np.arange(0, 41, 5))
ax2.set_xlim([-0.5, 5.5])
ax2.set_ylabel('speedup', color=speed_color)
for tl in ax2.get_yticklabels():
    tl.set_color(speed_color)
ax2.set_ylim([0, 41])
ax2.set_xticklabels(['2', '4', '8', '16', '32', '64'])
ax2.plot(color_4096_range, speedups_sift_4096[:, 0], marker='o', color=speed_color, linewidth=line_thickness)

plt.grid()

ax4.set_title('sift_1024')
ax4.set_yticks(np.arange(0, 1.1, 0.1))
ax4.set_xlim([-0.5, 5.5])
ax4.set_ylim([0, 1.1])
ax4.set_xlabel('k_t')
ax4.set_ylabel('dot_product', color=dot_color)
ax4.set_xticklabels(['2', '4', '8', '16', '32', '64'])
ax4.plot(color_4096_range, dot_sift_1024[:, 0], marker='o', color=dot_color, linewidth=line_thickness, linestyle='--')
ax4.plot(color_4096_range, dotn_sift_1024[:, 0], marker='o', color=dot_color, linewidth=line_thickness)

ax2 = ax4.twinx()
ax2.set_xticks(np.arange(0, 7, 1))
ax2.set_yticks(np.arange(0, 41, 5))
ax2.set_xlim([-0.5, 5.5])
ax2.set_ylabel('speedup', color=speed_color)
for tl in ax2.get_yticklabels():
    tl.set_color(speed_color)
ax2.set_ylim([0, 41])
ax2.set_xticklabels(['2', '4', '8', '16', '32', '64'])
ax2.plot(color_4096_range, speedups_sift_1024[:, 0], marker='o', color=speed_color, linewidth=line_thickness)

plt.tight_layout(pad=1, w_pad=1.0, h_pad=1.5)
plt.grid()
plt.show()
