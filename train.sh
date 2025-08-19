
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py \
    --weights 'yolov7-w6-face.pt' \
    --cfg cfg/yolov7-face.yaml \
    --data data/enhanced_face.yaml \
    --hyp data/hyp.scratch.p6.yaml \
    --epochs 4 \
    --batch-size 16 \
    --img-size 640 640 \
    --device '' \
    --workers 4 \
    --project runs/train \
    --name test_face_detection_gp2_combined_data \
    --kpt-label 0