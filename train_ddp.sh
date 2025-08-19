#!/bin/bash

# YOLOv7-Face Training Script
# Multi-GPU DDP training on 3x RTX 3090

export CUDA_VISIBLE_DEVICES=0,1,2

python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=12345 \
    train.py \
    --weights '' \
    --cfg cfg/yolov7-face.yaml \
    --data data/widerface.yaml \
    --hyp data/hyp.scratch.p6.yaml \
    --epochs 3 \
    --image-weights \
    --multi-scale \
    --batch-size 192 \
    --img-size 960 960 \
    --device 0,1,2 \
    --workers 20 \
    --project runs/train \
    --name test_face_detection_ddp \
    --kpt-label 0
