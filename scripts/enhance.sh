#!/bin/bash

<<'COMMIT'
Teeth enhance Infer bash version1.0 by yyl
Usage: bash scripts/enhance.sh <video_path> <save_path>

    example: bash scripts/enhance.sh /path/to/*.mp4 /path/to/save_path

COMMIT

CRTDIR=$(pwd)

video_path=$1
save_path=$2
default_save_path=$CRTDIR/sample/test
if [ -z "$2" ]; then
    save_path=$default_save_path
else
    save_path=$2
fi

video_base=$(basename "$video_path")
video_ext="${video_base%.*}"

echo "++++++ Infer teeth +++++"
echo "python enhance.py --video_path=$video_path --save_path=$save_path"
python enhance.py \
    --video_path $video_path \
    --save_path $save_path \
    --save_restored_img false