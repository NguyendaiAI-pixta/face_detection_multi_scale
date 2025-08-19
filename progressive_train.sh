#!/bin/bash
# Progressive training strategy for multi-dataset face detection
# Stage-based training for optimal convergence

echo "=== YOLOv7-Face Progressive Training Pipeline ==="

# Stage 1: Base training on WiderFace (warm-up)
echo "Stage 1: Base training on WiderFace..."
python train.py \
    --data data/widerface.yaml \
    --weights yolov7-w6-face.pt \
    --cfg cfg/yolov7-face.yaml \
    --img-size 640 640 \
    --batch-size 32 \
    --epochs 50 \
    --name stage1_base \
    --hyp data/hyp.scratch.p6.yaml \
    --workers 8

# Stage 2: Fine-tune with enhanced dataset
echo "Stage 2: Fine-tuning with enhanced dataset..."
python train.py \
    --data data/enhanced_face.yaml \
    --weights runs/train/stage1_base/weights/best.pt \
    --cfg cfg/yolov7-face.yaml \
    --img-size 640 640 \
    --batch-size 24 \
    --epochs 100 \
    --name stage2_enhanced \
    --hyp data/hyp.finetune.yaml \
    --workers 8

# Stage 3: High-resolution fine-tuning
echo "Stage 3: High-resolution fine-tuning..."
python train.py \
    --data data/enhanced_face.yaml \
    --weights runs/train/stage2_enhanced/weights/best.pt \
    --cfg cfg/yolov7-face.yaml \
    --img-size 960 960 \
    --batch-size 16 \
    --epochs 50 \
    --name stage3_hires \
    --hyp data/hyp.finetune.yaml \
    --workers 8

# Stage 4: Domain-specific fine-tuning (if you have specific use case)
echo "Stage 4: Domain-specific fine-tuning..."
python train.py \
    --data data/domain_specific.yaml \
    --weights runs/train/stage3_hires/weights/best.pt \
    --cfg cfg/yolov7-face.yaml \
    --img-size 640 640 \
    --batch-size 32 \
    --epochs 30 \
    --name stage4_domain \
    --hyp data/hyp.finetune.yaml \
    --workers 8

echo "=== Training Complete! ==="
echo "Best models saved in runs/train/stage*_*/weights/best.pt"
