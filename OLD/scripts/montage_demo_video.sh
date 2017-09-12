#!/bin/bash
##!/bin/zsh

input_raw="/home/kivan/datasets/KITTI/sequences_color/07/image_2/"
resized_raw="/home/kivan/datasets/KITTI/output/raw/07/"
input_semseg="/home/kivan/datasets/KITTI/output/semseg_xent/"
out_folder="/home/kivan/datasets/KITTI/output/montage/"

mkdir -p $resized_raw
mkdir -p $out_folder

for i in {0..1100}
do
   #echo i
   printf -v filename "%06d.png" $i
   #printf -v filename_right "%06d.jpg" $i
   path_input_raw=$input_raw$filename
   path_semseg=$input_semseg$filename
   path_raw=$resized_raw$filename
   echo $path_raw
   printf -v out_img $out_folder"%06d.png" $i
   convert $path_input_raw -resize 1232x384 $path_raw
   montage -mode concatenate -tile 1x $path_raw $path_semseg $out_img
done
