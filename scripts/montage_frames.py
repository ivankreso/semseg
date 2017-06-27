from os.path import join
import sub

seq_name="stuttgart_02"
input_raw = join("/home/kivan/datasets/Cityscapes/orig/leftImg8bit/demoVideo/", seq_name)
resized_raw="/home/kivan/datasets/Cityscapes/demo/raw/"
input_semseg="/home/kivan/datasets/results/tmp/cityscapes/1_6_14-55-59/evaluation/test/"$seq_name"/"
out_folder="/home/kivan/datasets/Cityscapes/demo/montage/"

   printf -v out_img $out_folder"%06d.png" $i
   convert $path_input_raw -resize 1232x384 $path_raw
   montage -mode concatenate -tile 1x $path_raw $path_semseg $out_img
subprocess.run([cmd, gt_dir, save_dir])
