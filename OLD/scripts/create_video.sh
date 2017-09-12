
#input_dir="/home/kivan/datasets/KITTI/output/montage/"
#save_path="/home/kivan/datasets/KITTI/output/demo.mp4"
save_path="/home/kivan/datasets/KITTI/output/demo.webm"

#ffmpeg -f image2 -r 10 -i /home/kivan/datasets/KITTI/output/montage/%06d.png -c:v libx264 -crf 18 -r 10 $save_path
#ffmpeg -f image2 -r 10 -i /home/kivan/datasets/KITTI/output/montage/%06d.png -c:v libvpx-vp9 -crf 10 -b:v 0 -c:a libvorbis -r 10 $save_path
#ffmpeg -f image2 -i /home/kivan/datasets/KITTI/output/montage/%06d.png -c:v libvpx-vp9 -crf 10 -b:v 0 -c:a libvorbis $save_path
#ffmpeg -f image2 -framerate 10 -i /home/kivan/datasets/KITTI/output/montage/%06d.png -c:v libvpx-vp9 -crf 10 -b:v 0 $save_path
ffmpeg -f image2 -i /home/kivan/datasets/KITTI/output/montage/%06d.png -c:v libvpx-vp9 -crf 10 -b:v 0 $save_path

