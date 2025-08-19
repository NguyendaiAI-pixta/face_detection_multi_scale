#!/usr/bin/env python3

import sys
sys.path.append('/home/dainguyenvan/project/ARV/auto-footage/yolov7-face')

from utils.datasets import create_dataloader
import yaml

# Load config
with open('/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/data/widerface.yaml') as f:
    data_dict = yaml.safe_load(f)

print("=== Testing Dataset Loading ===")
print(f"Train path: {data_dict['train']}")
print(f"Val path: {data_dict['val']}")

class MockOpt:
    def __init__(self):
        self.single_cls = False
        self.cache_images = False
        self.rect = True
        self.image_weights = False
        self.quad = False
        self.world_size = 1
        self.workers = 1

opt = MockOpt()

print("\n=== Testing Training Dataset ===")
try:
    train_loader, train_dataset = create_dataloader(
        data_dict['train'], 640, 32, 32, opt,
        hyp={}, augment=False, cache=False, rect=True, rank=-1,
        world_size=1, workers=1, prefix='train: ', kpt_label=0
    )
    print(f"✓ Training dataset loaded successfully")
    print(f"  Images: {len(train_dataset.imgs)}")
    print(f"  Labels: {len(train_dataset.labels)}")
    if len(train_dataset.labels) > 0:
        import numpy as np
        all_labels = np.concatenate(train_dataset.labels, 0)
        print(f"  Total annotations: {len(all_labels)}")
        print(f"  Sample labels: {all_labels[:3] if len(all_labels) > 3 else all_labels}")
except Exception as e:
    print(f"✗ Training dataset failed: {e}")

print("\n=== Testing Validation Dataset ===")
try:
    val_loader, val_dataset = create_dataloader(
        data_dict['val'], 640, 32, 32, opt,
        hyp={}, augment=False, cache=False, rect=True, rank=-1,
        world_size=1, workers=1, prefix='val: ', kpt_label=0
    )
    print(f"✓ Validation dataset loaded successfully")
    print(f"  Images: {len(val_dataset.imgs)}")
    print(f"  Labels: {len(val_dataset.labels)}")
    if len(val_dataset.labels) > 0:
        import numpy as np
        all_labels = np.concatenate(val_dataset.labels, 0)
        print(f"  Total annotations: {len(all_labels)}")
        print(f"  Sample labels: {all_labels[:3] if len(all_labels) > 3 else all_labels}")
    else:
        print("  ✗ No validation labels loaded!")
        print(f"  Image paths sample: {val_dataset.imgs[:3] if hasattr(val_dataset, 'imgs') else 'No imgs attr'}")
        print(f"  Label paths sample: {val_dataset.label_files[:3] if hasattr(val_dataset, 'label_files') else 'No label_files attr'}")
        
except Exception as e:
    print(f"✗ Validation dataset failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Manual Label Check ===")
import os
import glob
val_labels = glob.glob('/mnt/md0/projects/nguyendai-footage/WIDER_val/labels/**/*.txt', recursive=True)
print(f"Found {len(val_labels)} label files manually")
if val_labels:
    print(f"Sample label files: {val_labels[:3]}")
    with open(val_labels[0], 'r') as f:
        print(f"Sample content: {f.readline().strip()}")
