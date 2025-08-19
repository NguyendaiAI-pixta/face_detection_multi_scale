#!/bin/bash

# Convert WiderFace validation annotations to YOLO format
echo "Converting WiderFace validation labels to YOLO format..."

python /home/dainguyenvan/project/ARV/auto-footage/yolov7-face/convert_widerface_to_yolo.py \
    --annotation_file /mnt/md0/projects/nguyendai-footage/wider_face_split/wider_face_val_bbx_gt.txt \
    --images_dir /mnt/md0/projects/nguyendai-footage/WIDER_val \
    --output_dir /mnt/md0/projects/nguyendai-footage/WIDER_val/labels

echo "Validation label conversion completed!"
echo "Checking some validation labels..."
find /mnt/md0/projects/nguyendai-footage/WIDER_val/labels -name "*.txt" | head -3 | xargs -I {} sh -c 'echo "File: {}" && head -2 "{}" && echo ""'
