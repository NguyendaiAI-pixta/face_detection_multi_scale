#!/usr/bin/env python3
"""
Phân tích và visualize kết quả từ bộ dữ liệu blur vừa tạo
- Vẽ bounding box lên 1 ảnh blur bất kỳ
- Đếm tổng số face mới của từng mức easy/medium/hard
- So sánh với số face gốc WiderFace
"""
import cv2
import json
from pathlib import Path
import random

def draw_bbox_on_image(image_path, label_path, output_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Không load được ảnh: {image_path}")
        return False
    h, w = img.shape[:2]
    # Đọc label YOLO
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, x_center, y_center, width, height = map(float, parts[:5])
                x_center_px = x_center * w
                y_center_px = y_center * h
                width_px = width * w
                height_px = height * h
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imwrite(str(output_path), img)
    print(f"✅ Đã vẽ bbox lên ảnh: {output_path}")
    return True

def analyze_blur_dataset(metadata_path, widerface_labels_dir):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    results = metadata['results']
    # Đếm số face mới theo từng mức
    stats = {'easy': 0, 'medium': 0, 'hard': 0, 'total': 0}
    for r in results:
        stats[r['difficulty']] += r['num_faces']
        stats['total'] += r['num_faces']
    print("\n===== Thống kê số face mới tạo ra =====")
    print("| Difficulty | Total Faces |")
    print("|------------|------------|")
    for k in ['easy','medium','hard']:
        print(f"| {k:10s} | {stats[k]:10d} |")
    print(f"| {'TOTAL':10s} | {stats['total']:10d} |")
    # Đếm số face gốc WiderFace
    widerface_total = 0
    for label_file in Path(widerface_labels_dir).rglob('*.txt'):
        with open(label_file, 'r') as f:
            widerface_total += sum(1 for _ in f)
    print("\n===== So sánh với WiderFace gốc =====")
    print(f"Số face gốc WiderFace: {widerface_total}")
    print(f"Số face mới (blur): {stats['total']}")
    print(f"Tỉ lệ tăng: {stats['total']/widerface_total:.2f}x")
    return stats, widerface_total

def demo_visualize_one_image(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    results = metadata['results']
    # Chọn ngẫu nhiên 1 ảnh blur
    sample = random.choice(results)
    image_path = sample['output_image']
    label_path = sample['output_label']
    output_path = str(Path(image_path).parent / f"vis_{Path(image_path).name}")
    draw_bbox_on_image(image_path, label_path, output_path)
    print(f"Ảnh demo: {output_path} (Difficulty: {sample['difficulty']}, Faces: {sample['num_faces']})")

if __name__ == "__main__":
    # Đường dẫn metadata và WiderFace labels
    metadata_path = "/mnt/md0/projects/nguyendai-footage/blur_dataset/dataset_metadata.json"
    widerface_labels_dir = "/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/WIDER_train/labels"
    # Phân tích số lượng face
    analyze_blur_dataset(metadata_path, widerface_labels_dir)
    # Vẽ bbox lên 1 ảnh blur bất kỳ
    demo_visualize_one_image(metadata_path)
