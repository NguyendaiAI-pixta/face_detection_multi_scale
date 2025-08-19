#!/usr/bin/env python3

import os
import glob

print("=== Validation Dataset Debug ===")

# Check validation path structure
val_path = "/mnt/md0/projects/nguyendai-footage/WIDER_val/"
print(f"Validation path: {val_path}")
print(f"Path exists: {os.path.exists(val_path)}")

if os.path.exists(val_path):
    print(f"Contents of validation directory:")
    for item in os.listdir(val_path):
        item_path = os.path.join(val_path, item)
        print(f"  {item} ({'dir' if os.path.isdir(item_path) else 'file'})")

# Check for images
images_patterns = [
    f"{val_path}/**/*.jpg",
    f"{val_path}/**/*.png", 
    f"{val_path}/**/*.jpeg"
]

total_images = 0
for pattern in images_patterns:
    images = glob.glob(pattern, recursive=True)
    total_images += len(images)
    print(f"Images matching {pattern}: {len(images)}")

print(f"Total validation images: {total_images}")

# Check for labels
labels_path = f"{val_path}/labels"
print(f"\nLabels directory: {labels_path}")
print(f"Labels directory exists: {os.path.exists(labels_path)}")

if os.path.exists(labels_path):
    label_files = glob.glob(f"{labels_path}/**/*.txt", recursive=True)
    print(f"Total label files: {len(label_files)}")
    
    if label_files:
        print("Sample label files:")
        for i, label_file in enumerate(label_files[:5]):
            print(f"  {label_file}")
            try:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        print(f"    Content: {content.split(chr(10))[0]}")  # First line
                    else:
                        print(f"    Content: EMPTY FILE")
            except Exception as e:
                print(f"    Error reading: {e}")
else:
    print("Labels directory does not exist!")
