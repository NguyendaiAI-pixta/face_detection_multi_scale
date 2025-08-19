#!/usr/bin/env python3

import os
import glob

print("=== Checking WIDER_val Structure ===")

val_path = "/mnt/md0/projects/nguyendai-footage/WIDER_val"
print(f"Validation path: {val_path}")

# Check immediate contents
print(f"\nImmediate contents of {val_path}:")
if os.path.exists(val_path):
    items = os.listdir(val_path)
    for item in sorted(items):
        item_path = os.path.join(val_path, item)
        if os.path.isdir(item_path):
            # Count images in subdirectory
            images = glob.glob(f"{item_path}/**/*.jpg", recursive=True)
            print(f"  📁 {item}/ ({len(images)} images)")
        else:
            print(f"  📄 {item}")

# Check if images are in root or subdirectories
root_images = glob.glob(f"{val_path}/*.jpg")
all_images = glob.glob(f"{val_path}/**/*.jpg", recursive=True)

print(f"\nImage distribution:")
print(f"  Images in root: {len(root_images)}")
print(f"  Images in subdirs: {len(all_images) - len(root_images)}")
print(f"  Total images: {len(all_images)}")

if all_images:
    print(f"\nSample image paths:")
    for img in all_images[:5]:
        print(f"  {img}")

# Check labels structure
labels_path = f"{val_path}/labels"
if os.path.exists(labels_path):
    print(f"\nLabels directory structure:")
    label_items = os.listdir(labels_path)
    for item in sorted(label_items)[:5]:  # Show first 5
        item_path = os.path.join(labels_path, item)
        if os.path.isdir(item_path):
            labels = glob.glob(f"{item_path}/*.txt")
            print(f"  📁 {item}/ ({len(labels)} labels)")

# Check if YOLOv7 expects images and labels in images/ folder
images_dir = f"{val_path}/images"
print(f"\nChecking for images/ subdirectory:")
print(f"  {images_dir} exists: {os.path.exists(images_dir)}")

if os.path.exists(images_dir):
    imgs_in_images = glob.glob(f"{images_dir}/**/*.jpg", recursive=True)
    print(f"  Images in images/ folder: {len(imgs_in_images)}")
