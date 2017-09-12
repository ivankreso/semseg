#!/bin/bash
##!/bin/zsh

seq_name="stuttgart_02"
input_raw="/home/kivan/datasets/Cityscapes/orig/leftImg8bit/demoVideo/"$seq_name"/"
resized_raw="/home/kivan/datasets/Cityscapes/demo/raw/"
input_semseg="/home/kivan/datasets/results/tmp/cityscapes/1_6_14-55-59/evaluation/test/"$seq_name"/"
out_folder="/home/kivan/datasets/Cityscapes/demo/montage/"

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
