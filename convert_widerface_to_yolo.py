#!/usr/bin/env python3
"""
Convert WiderFace annotation format to YOLO format
WiderFace format: x y w h blur expression illumination invalid occlusion pose
YOLO format: class_id x_center y_center width height (all normalized 0-1)
"""
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_widerface_to_yolo(annotation_file, images_dir, output_labels_dir):
    """
    Convert WiderFace annotations to YOLO format
    
    Args:
        annotation_file: Path to WiderFace annotation file (e.g., wider_face_train_bbx_gt.txt)
        images_dir: Path to images directory 
        output_labels_dir: Path to output labels directory
    """
    
    # Create output directory
    Path(output_labels_dir).mkdir(parents=True, exist_ok=True)
    
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    total_images = 0
    processed_images = 0
    
    # Count total images first
    for line in lines:
        line = line.strip()
        if line.endswith('.jpg'):
            total_images += 1
    
    print(f"Found {total_images} images to process")
    
    # Progress bar
    pbar = tqdm(total=total_images, desc="Converting annotations")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.endswith('.jpg'):
            img_path = line
            img_name = os.path.basename(img_path)
            img_name_no_ext = os.path.splitext(img_name)[0]
            
            # Get full image path
            # Check if images are in images/ subfolder
            full_img_path = os.path.join(images_dir, "images", img_path)
            if not os.path.exists(full_img_path):
                # Try without images/ subfolder
                full_img_path = os.path.join(images_dir, img_path)
            
            # Check if image exists
            if not os.path.exists(full_img_path):
                print(f"Warning: Image not found: {full_img_path}")
                i += 2  # Skip num_faces line
                continue
                
            # Get image dimensions
            try:
                img = cv2.imread(full_img_path)
                if img is None:
                    print(f"Warning: Cannot read image: {full_img_path}")
                    i += 2
                    continue
                    
                img_height, img_width = img.shape[:2]
            except Exception as e:
                print(f"Error reading image {full_img_path}: {e}")
                i += 2
                continue
            
            i += 1  # Move to num_faces line
            if i >= len(lines):
                break
                
            try:
                num_faces = int(lines[i].strip())
            except ValueError:
                print(f"Error parsing number of faces for {img_path}")
                i += 1
                continue
                
            i += 1  # Move to first face annotation
            
            # Create label file path
            label_subdir = os.path.dirname(img_path)
            label_dir = os.path.join(output_labels_dir, label_subdir)
            Path(label_dir).mkdir(parents=True, exist_ok=True)
            
            label_file = os.path.join(label_dir, img_name_no_ext + '.txt')
            
            yolo_annotations = []
            
            # Process each face
            for face_idx in range(num_faces):
                if i >= len(lines):
                    break
                    
                try:
                    face_data = lines[i].strip().split()
                    
                    if len(face_data) < 4:
                        print(f"Warning: Invalid face data for {img_path}: {face_data}")
                        i += 1
                        continue
                    
                    # Parse WiderFace format: x y w h blur expression illumination invalid occlusion pose
                    x = float(face_data[0])  # top-left x
                    y = float(face_data[1])  # top-left y  
                    w = float(face_data[2])  # width
                    h = float(face_data[3])  # height
                    
                    # Skip invalid faces (if available)
                    if len(face_data) >= 8:
                        invalid = int(face_data[7])
                        if invalid == 1:  # Skip invalid faces
                            i += 1
                            continue
                    
                    # Skip faces that are too small or invalid
                    if w <= 0 or h <= 0 or x < 0 or y < 0:
                        i += 1
                        continue
                    
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    norm_width = w / img_width
                    norm_height = h / img_height
                    
                    # Ensure values are within [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    norm_width = max(0, min(1, norm_width))
                    norm_height = max(0, min(1, norm_height))
                    
                    # YOLO format: class_id x_center y_center width height
                    yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                    
                except Exception as e:
                    print(f"Error processing face {face_idx} for {img_path}: {e}")
                
                i += 1
            
            # Write YOLO annotations to file
            with open(label_file, 'w') as label_f:
                for annotation in yolo_annotations:
                    label_f.write(annotation + '\n')
            
            processed_images += 1
            pbar.update(1)
        else:
            i += 1
    
    pbar.close()
    print(f"Conversion complete! Processed {processed_images}/{total_images} images")
    print(f"Labels saved to: {output_labels_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert WiderFace annotations to YOLO format')
    parser.add_argument('--annotation_file', required=True, help='Path to WiderFace annotation file')
    parser.add_argument('--images_dir', required=True, help='Path to images directory')
    parser.add_argument('--output_dir', required=True, help='Path to output labels directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return
    
    convert_widerface_to_yolo(args.annotation_file, args.images_dir, args.output_dir)


if __name__ == "__main__":
    main()
'''
python /home/dainguyenvan/project/ARV/auto-footage/yolov7-face/convert_widerface_to_yolo.py 
--annotation_file /mnt/md0/projects/nguyendai-footage/wider_face_split/wider_face_train_bbx_gt.txt 
--images_dir /mnt/md0/projects/nguyendai-footage/WIDER_train 
--output_dir /mnt/md0/projects/nguyendai-footage/WIDER_train/labels

python /home/dainguyenvan/project/ARV/auto-footage/yolov7-face/convert_widerface_to_yolo.py --annotation_file /mnt/md0/projects/nguyendai-footage/wider_face_split/wider_face_val_bbx_gt.txt --images_dir /mnt/md0/projects/nguyendai-footage/WIDER_val --output_dir /mnt/md0/projects/nguyendai-footage/WIDER_val/labels
'''
