#!/usr/bin/env python3
"""
Visualize WiderFace dataset with bounding boxes
python visualize_widerface.py --data_dir /mnt/md0/projects/nguy            'occluded_faces': 0,  # occlusion >= 1
            'heavy_occluded_faces': 0,  # occlusion == 2 (màu tím)
        }
        
        for img_path, faces in annotations.items():ai-footage --split train --num_images 5
tất cả : python visualize_widerface.py --data_dir ./data/widerface --split val --num_images 0
các face hơp lệ : python visualize_widerface.py --data_dir ./data/widerface --split val --filter_invalid
thống kê dataset: python visualize_widerface.py --data_dir ./data/widerface --stats_only
visualize 1 ảnh: python visualize_widerface.py --single_image ./path/to/image.jpg --data_dir ./data/widerface
python visualize_widerface.py --single_image /mnt/md0/projects/nguyendai-footage/WIDER_val/images/0--Parade/0_Parade_marchingband_1_849.jpg --data_dir /mnt/md0/projects/nguyendai-footage
phát hiện face màu tím(bị che khuất): python visualize_widerface.py --data_dir /mnt/md0/projects/nguyendai-footage --detect_occluded --splits train val
"""

import argparse
import os
import cv2
import random
from pathlib import Path
import numpy as np


def load_annotations(annot_file):
    """Load WiderFace annotation file"""
    annotations = {}
    
    with open(annot_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
                
            line = line.strip()
            if not line.endswith('.jpg'):
                continue
                
            image_path = line
            annotations[image_path] = []
            
            # Read number of faces
            num_faces = int(f.readline().strip())
            
            if num_faces == 0:
                # Skip the line with zeros
                f.readline()
            else:
                for _ in range(num_faces):
                    face_line = f.readline().strip()
                    if face_line:
                        parts = list(map(int, face_line.split()))
                        if len(parts) >= 4:
                            x, y, w, h = parts[:4]
                            invalid = parts[7] if len(parts) > 7 else 0
                            blur = parts[4] if len(parts) > 4 else 0
                            occlusion = parts[8] if len(parts) > 8 else 0
                            
                            annotations[image_path].append({
                                'bbox': [x, y, w, h],
                                'invalid': invalid,
                                'blur': blur,
                                'occlusion': occlusion
                            })
    
    return annotations


def get_dataset_statistics(data_dir):
    """Get comprehensive statistics for all splits"""
    data_dir = Path(data_dir)
    
    total_stats = {
        'total_images': 0,
        'total_faces': 0,
        'valid_faces': 0,
        'invalid_faces': 0,
        'small_faces': 0,
        'medium_faces': 0,
        'large_faces': 0,
        'blurry_faces': 0,
        'occluded_faces': 0,
        'heavy_occluded_faces': 0,  # occlusion == 2 (màu tím)
    }
    
    splits_stats = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\n📊 Analyzing {split.upper()} split...")
        
        if split == 'test':
            # Test set only has file list, no bounding boxes
            image_dir = data_dir / f"WIDER_{split}" / "images"
            if image_dir.exists():
                test_images = []
                for root, dirs, files in os.walk(image_dir):
                    for file in files:
                        if file.lower().endswith('.jpg'):
                            test_images.append(file)
                
                splits_stats[split] = {
                    'total_images': len(test_images),
                    'total_faces': 'N/A (no annotations)',
                    'valid_faces': 'N/A',
                    'invalid_faces': 'N/A',
                    'small_faces': 'N/A',
                    'medium_faces': 'N/A', 
                    'large_faces': 'N/A',
                    'avg_faces_per_img': 'N/A'
                }
                total_stats['total_images'] += len(test_images)
                print(f"   Images: {len(test_images)}")
                print(f"   Faces: N/A (test set has no annotations)")
            else:
                print(f"   ❌ {split} directory not found")
                splits_stats[split] = None
            continue
        
        # For train/val splits with annotations
        annot_file = data_dir / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
        image_dir = data_dir / f"WIDER_{split}" / "images"
        
        if not annot_file.exists():
            print(f"   ❌ Annotation file not found: {annot_file}")
            splits_stats[split] = None
            continue
            
        if not image_dir.exists():
            print(f"   ❌ Image directory not found: {image_dir}")
            splits_stats[split] = None
            continue
        
        annotations = load_annotations(annot_file)
        
        split_stats = {
            'total_images': len(annotations),
            'total_faces': 0,
            'valid_faces': 0,
            'invalid_faces': 0,
            'small_faces': 0,   # < 32x32
            'medium_faces': 0,  # 32x32 to 96x96
            'large_faces': 0,   # > 96x96
            'blurry_faces': 0,  # blur >= 1
            'occluded_faces': 0,  # occlusion >= 1
            'heavy_occluded_faces': 0,  # occlusion == 2 (màu tím)
        }
        
        for img_path, faces in annotations.items():
            split_stats['total_faces'] += len(faces)
            
            for face in faces:
                w, h = face['bbox'][2], face['bbox'][3]
                area = w * h
                invalid = face.get('invalid', 0)
                blur = face.get('blur', 0)
                occlusion = face.get('occlusion', 0)
                
                if invalid == 0:
                    split_stats['valid_faces'] += 1
                else:
                    split_stats['invalid_faces'] += 1
                
                # Size categories
                if area < 32 * 32:
                    split_stats['small_faces'] += 1
                elif area < 96 * 96:
                    split_stats['medium_faces'] += 1
                else:
                    split_stats['large_faces'] += 1
                
                # Quality categories
                if blur >= 1:
                    split_stats['blurry_faces'] += 1
                if occlusion >= 1:
                    split_stats['occluded_faces'] += 1
                if occlusion == 2:  # Heavy occlusion (màu tím)
                    split_stats['heavy_occluded_faces'] += 1
        
        split_stats['avg_faces_per_img'] = split_stats['total_faces'] / split_stats['total_images'] if split_stats['total_images'] > 0 else 0
        splits_stats[split] = split_stats
        
        # Update total stats
        if split != 'test':  # Don't count test images twice
            total_stats['total_images'] += split_stats['total_images']
            total_stats['total_faces'] += split_stats['total_faces']
            total_stats['valid_faces'] += split_stats['valid_faces']
            total_stats['invalid_faces'] += split_stats['invalid_faces']
            total_stats['small_faces'] += split_stats['small_faces']
            total_stats['medium_faces'] += split_stats['medium_faces']
            total_stats['large_faces'] += split_stats['large_faces']
            total_stats['blurry_faces'] += split_stats['blurry_faces']
            total_stats['occluded_faces'] += split_stats['occluded_faces']
            total_stats['heavy_occluded_faces'] += split_stats['heavy_occluded_faces']
        
        # Print split statistics
        print(f"   Images: {split_stats['total_images']:,}")
        print(f"   Total faces: {split_stats['total_faces']:,}")
        print(f"   Valid faces: {split_stats['valid_faces']:,} ({split_stats['valid_faces']/split_stats['total_faces']*100:.1f}%)")
        print(f"   Invalid faces: {split_stats['invalid_faces']:,} ({split_stats['invalid_faces']/split_stats['total_faces']*100:.1f}%)")
        print(f"   Avg faces/image: {split_stats['avg_faces_per_img']:.1f}")
    
    return total_stats, splits_stats


def visualize_single_image(image_path, data_dir, output_dir='visualize_demo'):
    """Visualize a single image with its annotations"""
    image_path = Path(image_path)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🖼️  Visualizing single image: {image_path}")
    
    # Try to find the image in dataset structure
    found_annotation = False
    faces = []
    
    for split in ['train', 'val']:
        annot_file = data_dir / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
        if not annot_file.exists():
            continue
            
        annotations = load_annotations(annot_file)
        
        # Check if image exists in annotations
        for img_relative_path, img_faces in annotations.items():
            img_name = Path(img_relative_path).name
            if img_name == image_path.name:
                faces = img_faces
                found_annotation = True
                print(f"✅ Found annotation in {split} split: {len(faces)} faces")
                break
        
        if found_annotation:
            break
    
    if not found_annotation:
        print(f"⚠️  No annotation found for {image_path.name}, will show image without bounding boxes")
    
    # Load and visualize image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    if faces:
        image_with_boxes = draw_bounding_boxes(image.copy(), faces, show_labels=True)
        
        # Add image info
        img_info = f"Image: {image_path.name} | Faces: {len(faces)} | Size: {image.shape[1]}x{image.shape[0]}"
        cv2.putText(image_with_boxes, img_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image_with_boxes, img_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    else:
        image_with_boxes = image.copy()
        # Add "No annotation" info
        no_annot_info = f"Image: {image_path.name} | No annotations found | Size: {image.shape[1]}x{image.shape[0]}"
        cv2.putText(image_with_boxes, no_annot_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image_with_boxes, no_annot_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    
    # Save result
    output_name = f"single_{image_path.stem}_visualized.jpg"
    output_path = output_dir / output_name
    cv2.imwrite(str(output_path), image_with_boxes)
    
    print(f"✅ Visualization saved: {output_path}")
    return output_path


def detect_heavy_occluded_faces(data_dir, splits=['train', 'val'], output_dir='occluded_faces_analysis', num_samples=50):
    """Detect and visualize images with heavy occluded faces (màu tím)"""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    visualize_dir = output_dir / 'visualized_images'
    visualize_dir.mkdir(exist_ok=True)
    
    results = {
        'summary': {
            'total_images': 0,
            'images_with_heavy_occluded': 0,
            'total_faces': 0,
            'heavy_occluded_faces': 0
        },
        'splits': {},
        'sample_images': []
    }
    
    print(f"\n🔍 Detecting Heavy Occluded Faces (Màu Tím - occlusion=2)")
    print(f"📂 Data directory: {data_dir}")
    print(f"📊 Splits: {splits}")
    
    for split in splits:
        print(f"\n📊 Analyzing {split.upper()} split...")
        
        annot_file = data_dir / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
        image_dir = data_dir / f"WIDER_{split}" / "images"
        
        if not annot_file.exists():
            print(f"❌ Annotation file not found: {annot_file}")
            continue
            
        if not image_dir.exists():
            print(f"❌ Image directory not found: {image_dir}")
            continue
        
        annotations = load_annotations(annot_file)
        
        split_results = {
            'total_images': len(annotations),
            'images_with_heavy_occluded': 0,
            'total_faces': 0,
            'heavy_occluded_faces': 0,
            'heavy_occluded_images': [],
            'sample_visualized': []
        }
        
        # Find images with heavy occluded faces
        print("🔍 Finding heavy occluded faces...")
        from tqdm import tqdm
        
        for img_path, faces in tqdm(annotations.items(), desc=f"Processing {split}"):
            split_results['total_faces'] += len(faces)
            
            # Check for heavy occluded faces
            heavy_occluded_in_image = []
            for i, face in enumerate(faces):
                occlusion = face.get('occlusion', 0)
                if occlusion == 2:  # Heavy occlusion
                    split_results['heavy_occluded_faces'] += 1
                    heavy_occluded_in_image.append({
                        'face_id': i,
                        'bbox': face['bbox'],
                        'invalid': face.get('invalid', 0),
                        'blur': face.get('blur', 0),
                        'pose': face.get('pose', 0)
                    })
            
            if heavy_occluded_in_image:
                split_results['images_with_heavy_occluded'] += 1
                
                image_info = {
                    'split': split,
                    'relative_path': img_path,
                    'full_path': str(image_dir / img_path),
                    'exists': (image_dir / img_path).exists(),
                    'total_faces': len(faces),
                    'heavy_occluded_faces': heavy_occluded_in_image,
                    'event_category': img_path.split('/')[0].split('--')[1] if '--' in img_path else 'unknown'
                }
                
                split_results['heavy_occluded_images'].append(image_info)
        
        # Sample some images for visualization
        if split_results['heavy_occluded_images']:
            # Sort by number of heavy occluded faces (descending)
            sorted_images = sorted(
                split_results['heavy_occluded_images'], 
                key=lambda x: len(x['heavy_occluded_faces']), 
                reverse=True
            )
            
            # Take samples for visualization
            num_samples_split = min(num_samples // len(splits), len(sorted_images))
            sample_images = sorted_images[:num_samples_split]
            
            print(f"🖼️  Visualizing {len(sample_images)} sample images...")
            
            for i, img_info in enumerate(sample_images, 1):
                if not img_info['exists']:
                    continue
                    
                # Load and visualize image
                image_path = Path(img_info['full_path'])
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                # Get all faces for this image
                faces = annotations[img_info['relative_path']]
                
                # Draw bounding boxes (will show heavy occluded as magenta)
                image_with_boxes = draw_bounding_boxes(image.copy(), faces, show_labels=True)
                
                # Highlight heavy occluded faces with extra thick border
                for face in faces:
                    if face.get('occlusion', 0) == 2:
                        x, y, w, h = face['bbox']
                        # Extra thick magenta border for heavy occluded
                        cv2.rectangle(image_with_boxes, (x-2, y-2), (x + w + 2, y + h + 2), (255, 0, 255), 4)
                
                # Add info text
                info_text = (
                    f"Image: {img_info['relative_path']} | "
                    f"Total: {img_info['total_faces']} faces | "
                    f"Heavy Occluded: {len(img_info['heavy_occluded_faces'])} faces | "
                    f"Event: {img_info['event_category']}"
                )
                
                # Multi-line info if too long
                if len(info_text) > 100:
                    lines = [
                        f"Image: {Path(img_info['relative_path']).name}",
                        f"Total: {img_info['total_faces']} faces | Heavy Occluded: {len(img_info['heavy_occluded_faces'])} | Event: {img_info['event_category']}"
                    ]
                else:
                    lines = [info_text]
                
                y_offset = 30
                for line in lines:
                    cv2.putText(image_with_boxes, line, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(image_with_boxes, line, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    y_offset += 25
                
                # Save visualization
                output_name = f"{split}_{i:03d}_occluded_{Path(img_info['relative_path']).stem}.jpg"
                output_path = visualize_dir / output_name
                cv2.imwrite(str(output_path), image_with_boxes)
                
                split_results['sample_visualized'].append({
                    'original_path': img_info['relative_path'],
                    'visualized_path': str(output_path),
                    'heavy_occluded_count': len(img_info['heavy_occluded_faces'])
                })
                
                print(f"   ✅ {i}/{len(sample_images)}: {img_info['relative_path']} -> {output_name}")
            
            results['sample_images'].extend(split_results['sample_visualized'])
        
        # Update totals
        results['summary']['total_images'] += split_results['total_images']
        results['summary']['images_with_heavy_occluded'] += split_results['images_with_heavy_occluded']
        results['summary']['total_faces'] += split_results['total_faces']
        results['summary']['heavy_occluded_faces'] += split_results['heavy_occluded_faces']
        
        results['splits'][split] = split_results
        
        # Print split summary
        print(f"\n📊 {split.upper()} SUMMARY:")
        print(f"   Total images: {split_results['total_images']:,}")
        print(f"   Images with heavy occluded faces: {split_results['images_with_heavy_occluded']:,} ({split_results['images_with_heavy_occluded']/split_results['total_images']*100:.2f}%)")
        print(f"   Total faces: {split_results['total_faces']:,}")
        print(f"   Heavy occluded faces (màu tím): {split_results['heavy_occluded_faces']:,} ({split_results['heavy_occluded_faces']/split_results['total_faces']*100:.2f}%)")
        print(f"   Sample images visualized: {len(split_results['sample_visualized'])}")
    
    # Save detailed results
    import json
    results_file = output_dir / 'heavy_occluded_analysis.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save summary report
    summary_file = output_dir / 'heavy_occluded_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Heavy Occluded Faces Analysis (Màu Tím - occlusion=2)\n\n")
        
        f.write("## OVERALL SUMMARY\n")
        f.write(f"Total images: {results['summary']['total_images']:,}\n")
        f.write(f"Images with heavy occluded faces: {results['summary']['images_with_heavy_occluded']:,} ({results['summary']['images_with_heavy_occluded']/results['summary']['total_images']*100:.2f}%)\n")
        f.write(f"Total faces: {results['summary']['total_faces']:,}\n")
        f.write(f"Heavy occluded faces: {results['summary']['heavy_occluded_faces']:,} ({results['summary']['heavy_occluded_faces']/results['summary']['total_faces']*100:.2f}%)\n\n")
        
        for split, split_data in results['splits'].items():
            f.write(f"## {split.upper()} SPLIT\n")
            f.write(f"Total images: {split_data['total_images']:,}\n")
            f.write(f"Images with heavy occluded: {split_data['images_with_heavy_occluded']:,} ({split_data['images_with_heavy_occluded']/split_data['total_images']*100:.2f}%)\n")
            f.write(f"Total faces: {split_data['total_faces']:,}\n")
            f.write(f"Heavy occluded faces: {split_data['heavy_occluded_faces']:,} ({split_data['heavy_occluded_faces']/split_data['total_faces']*100:.2f}%)\n")
            f.write(f"Sample images: {len(split_data['sample_visualized'])}\n\n")
    
    # Print final summary
    print(f"\n" + "="*70)
    print(f"📊 HEAVY OCCLUDED FACES ANALYSIS SUMMARY")
    print(f"="*70)
    print(f"Total images analyzed: {results['summary']['total_images']:,}")
    print(f"Images with heavy occluded faces: {results['summary']['images_with_heavy_occluded']:,} ({results['summary']['images_with_heavy_occluded']/results['summary']['total_images']*100:.2f}%)")
    print(f"Total faces: {results['summary']['total_faces']:,}")
    print(f"🟣 Heavy occluded faces (màu tím): {results['summary']['heavy_occluded_faces']:,} ({results['summary']['heavy_occluded_faces']/results['summary']['total_faces']*100:.2f}%)")
    print(f"Sample images visualized: {len(results['sample_images'])}")
    
    print(f"\n💾 Results saved to:")
    print(f"   📄 {results_file}")
    print(f"   📄 {summary_file}")
    print(f"   📂 {visualize_dir} (sample visualizations)")
    
    return results


def draw_bounding_boxes(image, faces, show_labels=True):
    """Draw bounding boxes on image"""
    colors = {
        'valid': (0, 255, 0),      # Green for valid faces
        'invalid': (0, 0, 255),    # Red for invalid faces
        'blur': (255, 165, 0),     # Orange for blurry faces
        'occluded': (255, 0, 255)  # Magenta for occluded faces
    }
    
    for i, face in enumerate(faces):
        x, y, w, h = face['bbox']
        invalid = face.get('invalid', 0)
        blur = face.get('blur', 0)
        occlusion = face.get('occlusion', 0)
        
        # Choose color based on face properties
        if invalid == 1:
            color = colors['invalid']
            status = 'Invalid'
        elif blur == 2:  # Heavy blur
            color = colors['blur']
            status = 'Blurry'
        elif occlusion == 2:  # Heavy occlusion
            color = colors['occluded']
            status = 'Occluded'
        else:
            color = colors['valid']
            status = 'Valid'
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        if show_labels:
            # Draw label
            label = f"Face{i+1}: {status} ({w}x{h})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(image, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            
            # Text
            cv2.putText(image, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def visualize_dataset(data_dir, split='train', num_images=10, output_dir='visualize_demo', 
                     show_labels=True, filter_invalid=False, min_size=0):
    """Visualize WiderFace dataset"""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Paths
    if split == 'test':
        image_dir = data_dir / f"WIDER_{split}" / "images"
        annot_file = data_dir / "wider_face_split" / "wider_face_test_filelist.txt"
        print("⚠️ Test set doesn't have bounding box annotations, only file list")
        return
    else:
        image_dir = data_dir / f"WIDER_{split}" / "images"
        annot_file = data_dir / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    
    # Check paths
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    if not annot_file.exists():
        print(f"❌ Annotation file not found: {annot_file}")
        return
    
    print(f"📂 Loading annotations from: {annot_file}")
    annotations = load_annotations(annot_file)
    print(f"📊 Found {len(annotations)} images with annotations")
    
    # Filter and sample images
    valid_images = []
    for img_path, faces in annotations.items():
        if not faces and filter_invalid:
            continue
            
        # Filter by minimum size
        if min_size > 0:
            valid_faces = [f for f in faces if f['bbox'][2] * f['bbox'][3] >= min_size]
            if not valid_faces:
                continue
        
        full_img_path = image_dir / img_path
        if full_img_path.exists():
            valid_images.append(img_path)
    
    print(f"📊 Found {len(valid_images)} valid images")
    
    # Sample images
    if num_images > 0:
        sample_images = random.sample(valid_images, min(num_images, len(valid_images)))
    else:
        sample_images = valid_images
    
    print(f"🎯 Visualizing {len(sample_images)} images...")
    
    # Process images
    stats = {'total_faces': 0, 'valid_faces': 0, 'invalid_faces': 0}
    
    for i, img_path in enumerate(sample_images, 1):
        full_img_path = image_dir / img_path
        faces = annotations[img_path]
        
        # Load image
        image = cv2.imread(str(full_img_path))
        if image is None:
            print(f"⚠️ Cannot load image: {img_path}")
            continue
        
        # Filter faces if needed
        if filter_invalid:
            faces = [f for f in faces if f.get('invalid', 0) == 0]
        
        if min_size > 0:
            faces = [f for f in faces if f['bbox'][2] * f['bbox'][3] >= min_size]
        
        # Update stats
        stats['total_faces'] += len(faces)
        stats['valid_faces'] += sum(1 for f in faces if f.get('invalid', 0) == 0)
        stats['invalid_faces'] += sum(1 for f in faces if f.get('invalid', 0) == 1)
        
        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes(image.copy(), faces, show_labels)
        
        # Add image info
        img_info = f"Image: {img_path} | Faces: {len(faces)} | Size: {image.shape[1]}x{image.shape[0]}"
        cv2.putText(image_with_boxes, img_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image_with_boxes, img_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Save image
        output_name = f"{split}_{i:03d}_{Path(img_path).stem}.jpg"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), image_with_boxes)
        
        print(f"✅ {i}/{len(sample_images)}: {img_path} -> {output_name} ({len(faces)} faces)")
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"   Total images processed: {len(sample_images)}")
    print(f"   Total faces: {stats['total_faces']}")
    print(f"   Valid faces: {stats['valid_faces']}")
    print(f"   Invalid faces: {stats['invalid_faces']}")
    print(f"   Average faces per image: {stats['total_faces'] / len(sample_images):.1f}")
    
    print(f"\n🎉 Visualization completed!")
    print(f"📂 Results saved to: {output_dir.absolute()}")
    print(f"\n🎨 Legend:")
    print(f"   🟢 Green: Valid faces")
    print(f"   🔴 Red: Invalid faces")
    print(f"   🟠 Orange: Heavy blur")
    print(f"   🟣 Magenta: Heavy occlusion")


def main():
    parser = argparse.ArgumentParser(description='Visualize WiderFace dataset')
    parser.add_argument('--data_dir', type=str, default='./data/widerface',
                       help='Directory containing WiderFace data')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='val',
                       help='Dataset split to visualize (default: val)')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of images to visualize (0 for all, default: 10)')
    parser.add_argument('--output_dir', type=str, default='visualize_demo',
                       help='Output directory for visualization results')
    parser.add_argument('--hide_labels', action='store_true',
                       help='Hide face labels on bounding boxes')
    parser.add_argument('--filter_invalid', action='store_true',
                       help='Filter out invalid faces')
    parser.add_argument('--min_size', type=int, default=0,
                       help='Minimum face area (width*height) to include')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for image sampling')
    
    # New arguments
    parser.add_argument('--stats_only', action='store_true',
                       help='Only show dataset statistics, do not save images')
    parser.add_argument('--single_image', type=str,
                       help='Path to a single image to visualize')
    parser.add_argument('--detect_occluded', action='store_true',
                       help='Detect and analyze heavy occluded faces (màu tím)')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], 
                       default=['train', 'val'],
                       help='Dataset splits to analyze (for --detect_occluded)')
    parser.add_argument('--occluded_samples', type=int, default=50,
                       help='Number of sample images to visualize for occluded analysis')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print(f"🎯 WiderFace Dataset Visualizer")
    
    # Handle single image visualization
    if args.single_image:
        print(f"🖼️  Single image mode")
        visualize_single_image(args.single_image, args.data_dir, args.output_dir)
        return
    
    # Handle heavy occluded detection
    if args.detect_occluded:
        print(f"🟣 Heavy occluded faces detection mode")
        detect_heavy_occluded_faces(
            data_dir=args.data_dir,
            splits=args.splits,
            output_dir='occluded_faces_analysis',
            num_samples=args.occluded_samples
        )
        return
    
    # Handle statistics only mode
    if args.stats_only:
        print(f"📊 Statistics only mode")
        print(f"📂 Data directory: {args.data_dir}")
        
        total_stats, splits_stats = get_dataset_statistics(args.data_dir)
        
        # Print comprehensive statistics
        print(f"\n" + "="*60)
        print(f"📊 WIDERFACE DATASET COMPREHENSIVE STATISTICS")
        print(f"="*60)
        
        print(f"\n🎯 OVERALL SUMMARY:")
        print(f"   Total images (train+val): {total_stats['total_images']:,}")
        if splits_stats.get('test'):
            test_images = splits_stats['test']['total_images']
            print(f"   Total images (including test): {total_stats['total_images'] + test_images:,}")
        print(f"   Total faces: {total_stats['total_faces']:,}")
        print(f"   ✅ Usable for training: {total_stats['valid_faces']:,} ({total_stats['valid_faces']/total_stats['total_faces']*100:.1f}%)")
        print(f"   ❌ Invalid faces: {total_stats['invalid_faces']:,} ({total_stats['invalid_faces']/total_stats['total_faces']*100:.1f}%)")
        
        print(f"\n📏 FACE SIZES:")
        print(f"   Small faces (<32x32): {total_stats['small_faces']:,} ({total_stats['small_faces']/total_stats['total_faces']*100:.1f}%)")
        print(f"   Medium faces (32-96): {total_stats['medium_faces']:,} ({total_stats['medium_faces']/total_stats['total_faces']*100:.1f}%)")
        print(f"   Large faces (>96x96): {total_stats['large_faces']:,} ({total_stats['large_faces']/total_stats['total_faces']*100:.1f}%)")
        
        print(f"\n🌫️  QUALITY ISSUES:")
        print(f"   Blurry faces: {total_stats['blurry_faces']:,} ({total_stats['blurry_faces']/total_stats['total_faces']*100:.1f}%)")
        print(f"   Occluded faces: {total_stats['occluded_faces']:,} ({total_stats['occluded_faces']/total_stats['total_faces']*100:.1f}%)")
        print(f"   🟣 Heavy occluded faces (màu tím): {total_stats['heavy_occluded_faces']:,} ({total_stats['heavy_occluded_faces']/total_stats['total_faces']*100:.1f}%)")
        
        print(f"\n📋 DETAILED SPLIT BREAKDOWN:")
        for split in ['train', 'val', 'test']:
            if splits_stats.get(split):
                stats = splits_stats[split]
                print(f"\n   {split.upper()} SET:")
                print(f"      Images: {stats['total_images']:,}")
                if isinstance(stats['total_faces'], int):
                    print(f"      Total faces: {stats['total_faces']:,}")
                    print(f"      Valid faces: {stats['valid_faces']:,}")
                    print(f"      Invalid faces: {stats['invalid_faces']:,}")
                    print(f"      Average faces/image: {stats['avg_faces_per_img']:.1f}")
                else:
                    print(f"      Faces: {stats['total_faces']}")
        
        print(f"\n✨ TRAINING RECOMMENDATIONS:")
        valid_ratio = total_stats['valid_faces'] / total_stats['total_faces'] * 100
        if valid_ratio > 90:
            print(f"   🟢 Excellent! {valid_ratio:.1f}% of faces are valid for training")
        elif valid_ratio > 80:
            print(f"   🟡 Good! {valid_ratio:.1f}% of faces are valid for training")
        else:
            print(f"   🔴 Consider filtering: only {valid_ratio:.1f}% of faces are valid")
        
        small_ratio = total_stats['small_faces'] / total_stats['total_faces'] * 100
        if small_ratio > 30:
            print(f"   ⚠️  {small_ratio:.1f}% are small faces - consider min_size filtering for better training")
        
        return
    
    # Regular visualization mode
    print(f"🖼️  Visualization mode")
    print(f"📂 Data directory: {args.data_dir}")
    print(f"📊 Split: {args.split}")
    print(f"🖼️  Number of images: {args.num_images if args.num_images > 0 else 'ALL'}")
    print(f"💾 Output directory: {args.output_dir}")
    
    visualize_dataset(
        data_dir=args.data_dir,
        split=args.split,
        num_images=args.num_images,
        output_dir=args.output_dir,
        show_labels=not args.hide_labels,
        filter_invalid=args.filter_invalid,
        min_size=args.min_size
    )


if __name__ == "__main__":
    main()
