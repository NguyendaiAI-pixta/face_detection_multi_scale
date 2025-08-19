#!/usr/bin/env python3

import os
import glob

print("=== Debugging Label Path Issue ===")

# Check current structure
val_images_path = "/mnt/md0/projects/nguyendai-footage/WIDER_val/images"
val_labels_path = "/mnt/md0/projects/nguyendai-footage/WIDER_val/labels"

print(f"Images path: {val_images_path}")
print(f"Labels path: {val_labels_path}")

# Get sample image and check corresponding label
sample_images = glob.glob(f"{val_images_path}/**/*.jpg", recursive=True)[:5]
print(f"\nSample images ({len(sample_images)}):")

for img_path in sample_images:
    print(f"\nImage: {img_path}")
    
    # Convert image path to expected label path
    rel_path = os.path.relpath(img_path, val_images_path)
    label_path = os.path.join(val_labels_path, rel_path).replace('.jpg', '.txt')
    print(f"Expected label: {label_path}")
    print(f"Label exists: {os.path.exists(label_path)}")
    
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                print(f"Label content: {content[:50]}..." if len(content) > 50 else f"Label content: {content}")
        except Exception as e:
            print(f"Error reading label: {e}")

# Check total counts
all_images = glob.glob(f"{val_images_path}/**/*.jpg", recursive=True)
all_labels = glob.glob(f"{val_labels_path}/**/*.txt", recursive=True)

print(f"\n=== Summary ===")
print(f"Total images: {len(all_images)}")
print(f"Total labels: {len(all_labels)}")

# Check if images and labels have same structure
print(f"\n=== Structure Check ===")
image_dirs = set()
label_dirs = set()

for img in all_images[:10]:
    rel_dir = os.path.dirname(os.path.relpath(img, val_images_path))
    image_dirs.add(rel_dir)

for lbl in all_labels[:10]:
    rel_dir = os.path.dirname(os.path.relpath(lbl, val_labels_path))
    label_dirs.add(rel_dir)

print(f"Sample image directories: {sorted(list(image_dirs))[:5]}")
print(f"Sample label directories: {sorted(list(label_dirs))[:5]}")
print(f"Directories match: {image_dirs == label_dirs}")
