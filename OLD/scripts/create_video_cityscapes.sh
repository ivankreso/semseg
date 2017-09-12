seq_name="stuttgart_00"
#seq_name="stuttgart_01"
#seq_name="stuttgart_02"
input_dir="/home/kivan/datasets/results/tmp/cityscapes/1_6_14-55-59/evaluation/test/"$seq_name"/"
save_path="/home/kivan/datasets/Cityscapes/demo/"$seq_name".mkv"

#ffmpeg -f image2 -r 10 -i $input_dir%06d.png -c:v libx264 -crf 18 -r 10 $save_path
ffmpeg -f image2 -framerate 20 -i $input_dir%06d.png -c:v libx264 -crf 18 $save_path
#ffmpeg -f image2 -r 10 -i /home/kivan/datasets/KITTI/output/montage/%06d.png -c:v libvpx-vp9 -crf 10 -b:v 0 -c:a libvorbis -r 10 $save_path
#ffmpeg -f image2 -i /home/kivan/datasets/KITTI/output/montage/%06d.png -c:v libvpx-vp9 -crf 10 -b:v 0 -c:a libvorbis $save_path
#ffmpeg -f image2 -framerate 10 -i /home/kivan/datasets/KITTI/output/montage/%06d.png -c:v libvpx-vp9 -crf 10 -b:v 0 $save_path
#ffmpeg -f image2 -i /home/kivan/datasets/KITTI/output/montage/%06d.png -c:v libvpx-vp9 -crf 10 -b:v 0 $save_path

