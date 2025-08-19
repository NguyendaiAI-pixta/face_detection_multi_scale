#!/usr/bin/env python3
"""
Find invalid faces and artwork/cartoon images in WiderFace dataset
Usage: python find_invalid_faces.py --data_dir ./data/widerface --output_dir ./invalid_analysis
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict


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
                        if len(parts) >= 10:
                            x, y, w, h = parts[:4]
                            blur, expression, illumination, invalid, occlusion, pose = parts[4:10]
                            
                            annotations[image_path].append({
                                'bbox': [x, y, w, h],
                                'blur': blur,
                                'expression': expression,
                                'illumination': illumination,
                                'invalid': invalid,
                                'occlusion': occlusion,
                                'pose': pose
                            })
    
    return annotations


def detect_artwork_features(image_path):
    """Detect if image is artwork/cartoon based on visual features"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None, "Cannot load image"
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        features = {}
        
        # 1. Color saturation analysis (cartoons tend to have higher saturation)
        saturation = hsv[:, :, 1]
        features['avg_saturation'] = np.mean(saturation)
        features['high_saturation_ratio'] = np.sum(saturation > 150) / saturation.size
        
        # 2. Edge analysis (cartoons have cleaner, sharper edges)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 3. Color palette analysis (cartoons use fewer distinct colors)
        # Quantize image to reduce colors
        h, w = image.shape[:2]
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Calculate color variance
        unique_labels, counts = np.unique(labels, return_counts=True)
        features['color_clusters'] = len(unique_labels)
        features['dominant_color_ratio'] = np.max(counts) / len(labels)
        
        # 4. Texture analysis
        # Calculate local binary patterns or gradient analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features['avg_gradient'] = np.mean(gradient_magnitude)
        
        # 5. Brightness and contrast
        features['avg_brightness'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        
        # Simple heuristic classification
        artwork_score = 0
        
        # High saturation suggests artwork
        if features['high_saturation_ratio'] > 0.3:
            artwork_score += 2
        elif features['high_saturation_ratio'] > 0.2:
            artwork_score += 1
            
        # High dominant color ratio suggests limited palette
        if features['dominant_color_ratio'] > 0.4:
            artwork_score += 2
        elif features['dominant_color_ratio'] > 0.3:
            artwork_score += 1
            
        # High edge density with low gradient variance suggests clean lines
        if features['edge_density'] > 0.1 and features['brightness_std'] < 50:
            artwork_score += 1
            
        # Very high average saturation
        if features['avg_saturation'] > 120:
            artwork_score += 1
        
        features['artwork_score'] = artwork_score
        features['is_likely_artwork'] = artwork_score >= 3
        
        return features, None
        
    except Exception as e:
        return None, str(e)


def analyze_event_categories(annotations):
    """Analyze WiderFace event categories that might contain artwork"""
    # Known categories that often contain artwork/cartoons
    artwork_categories = [
        'Handshaking',  # Often includes illustrations
        'Ceremony',     # May include artistic elements
        'Picnic',       # Sometimes cartoon-style images
        'Parade',       # May include artistic floats/decorations
        'Press_Conference',  # Sometimes includes drawings/charts
        'Movie',        # Film scenes, potentially animated
        'TV_Show',      # TV content, potentially animated
        'Family_Gathering',  # Personal photos, sometimes artwork
        'Festival'      # May include artistic performances
    ]
    
    event_stats = defaultdict(lambda: {
        'total_images': 0,
        'images_with_invalid': 0,
        'total_faces': 0,
        'invalid_faces': 0,
        'sample_images': []
    })
    
    for img_path, faces in annotations.items():
        # Extract event category from path
        event = img_path.split('/')[0].split('--')[1] if '--' in img_path else 'unknown'
        
        event_stats[event]['total_images'] += 1
        event_stats[event]['total_faces'] += len(faces)
        
        has_invalid = False
        for face in faces:
            if face.get('invalid', 0) == 1:
                event_stats[event]['invalid_faces'] += 1
                has_invalid = True
        
        if has_invalid:
            event_stats[event]['images_with_invalid'] += 1
            if len(event_stats[event]['sample_images']) < 5:
                event_stats[event]['sample_images'].append(img_path)
    
    return dict(event_stats), artwork_categories


def find_invalid_faces(data_dir, splits=['train', 'val'], output_dir='invalid_analysis', analyze_artwork=False):
    """Find all images with invalid faces and optionally detect artwork"""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'summary': {},
        'splits': {},
        'invalid_images': [],
        'artwork_analysis': {} if analyze_artwork else None
    }
    
    total_stats = {
        'total_images': 0,
        'images_with_invalid': 0,
        'total_faces': 0,
        'invalid_faces': 0
    }
    
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
        print(f"📂 Loaded {len(annotations)} images")
        
        # Analyze event categories
        event_stats, artwork_categories = analyze_event_categories(annotations)
        
        split_results = {
            'total_images': len(annotations),
            'images_with_invalid': 0,
            'total_faces': 0,
            'invalid_faces': 0,
            'invalid_images': [],
            'event_analysis': event_stats,
            'artwork_candidates': []
        }
        
        # Find images with invalid faces
        print("🔍 Finding invalid faces...")
        for img_path, faces in tqdm(annotations.items(), desc="Processing images"):
            full_img_path = image_dir / img_path
            
            split_results['total_faces'] += len(faces)
            has_invalid = False
            invalid_faces_in_image = []
            
            for i, face in enumerate(faces):
                if face.get('invalid', 0) == 1:
                    split_results['invalid_faces'] += 1
                    has_invalid = True
                    invalid_faces_in_image.append({
                        'face_id': i,
                        'bbox': face['bbox'],
                        'blur': face['blur'],
                        'occlusion': face['occlusion'],
                        'pose': face['pose']
                    })
            
            if has_invalid:
                split_results['images_with_invalid'] += 1
                
                image_info = {
                    'split': split,
                    'relative_path': img_path,
                    'full_path': str(full_img_path),
                    'exists': full_img_path.exists(),
                    'total_faces': len(faces),
                    'invalid_faces': invalid_faces_in_image,
                    'event_category': img_path.split('/')[0].split('--')[1] if '--' in img_path else 'unknown'
                }
                
                split_results['invalid_images'].append(image_info)
                results['invalid_images'].append(image_info)
        
        # Artwork analysis
        if analyze_artwork:
            print("🎨 Analyzing artwork candidates...")
            artwork_analysis = {
                'by_event': {},
                'detected_artwork': []
            }
            
            # Check categories likely to contain artwork
            for event in artwork_categories:
                if event in event_stats:
                    artwork_analysis['by_event'][event] = event_stats[event]
                    
                    # Sample some images from this category for analysis
                    sample_images = event_stats[event]['sample_images'][:3]
                    for img_path in sample_images:
                        full_img_path = image_dir / img_path
                        if full_img_path.exists():
                            features, error = detect_artwork_features(full_img_path)
                            if features and features['is_likely_artwork']:
                                artwork_analysis['detected_artwork'].append({
                                    'path': img_path,
                                    'full_path': str(full_img_path),
                                    'event': event,
                                    'artwork_score': features['artwork_score'],
                                    'features': features
                                })
                                split_results['artwork_candidates'].append({
                                    'path': img_path,
                                    'event': event,
                                    'artwork_score': features['artwork_score']
                                })
            
            results['artwork_analysis'][split] = artwork_analysis
        
        # Update totals
        total_stats['total_images'] += split_results['total_images']
        total_stats['images_with_invalid'] += split_results['images_with_invalid']
        total_stats['total_faces'] += split_results['total_faces']
        total_stats['invalid_faces'] += split_results['invalid_faces']
        
        results['splits'][split] = split_results
        
        # Print split summary
        print(f"\n📊 {split.upper()} SUMMARY:")
        print(f"   Total images: {split_results['total_images']:,}")
        print(f"   Images with invalid faces: {split_results['images_with_invalid']:,}")
        print(f"   Total faces: {split_results['total_faces']:,}")
        print(f"   Invalid faces: {split_results['invalid_faces']:,}")
        if analyze_artwork:
            print(f"   Artwork candidates: {len(split_results['artwork_candidates'])}")
    
    results['summary'] = total_stats
    
    # Save results
    output_file = output_dir / 'invalid_faces_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save invalid images list
    invalid_list_file = output_dir / 'invalid_images_list.txt'
    with open(invalid_list_file, 'w') as f:
        f.write("# Images with invalid faces (invalid=1)\n")
        f.write(f"# Total: {len(results['invalid_images'])} images\n\n")
        for img_info in results['invalid_images']:
            f.write(f"{img_info['full_path']}\n")
    
    # Save by event category
    event_summary_file = output_dir / 'events_with_invalid_faces.txt'
    with open(event_summary_file, 'w') as f:
        f.write("# Event categories with invalid faces\n\n")
        
        for split in splits:
            if split in results['splits']:
                f.write(f"\n=== {split.upper()} SPLIT ===\n")
                event_stats = results['splits'][split]['event_analysis']
                
                # Sort by invalid face ratio
                sorted_events = sorted(
                    [(event, stats) for event, stats in event_stats.items() if stats['invalid_faces'] > 0],
                    key=lambda x: x[1]['invalid_faces'] / x[1]['total_faces'] if x[1]['total_faces'] > 0 else 0,
                    reverse=True
                )
                
                for event, stats in sorted_events:
                    invalid_ratio = stats['invalid_faces'] / stats['total_faces'] * 100 if stats['total_faces'] > 0 else 0
                    f.write(f"\n{event}:\n")
                    f.write(f"  Images: {stats['total_images']}, Invalid images: {stats['images_with_invalid']}\n")
                    f.write(f"  Faces: {stats['total_faces']}, Invalid faces: {stats['invalid_faces']} ({invalid_ratio:.1f}%)\n")
                    if stats['sample_images']:
                        f.write(f"  Sample images:\n")
                        for img in stats['sample_images'][:3]:
                            f.write(f"    {img}\n")
    
    # Print final summary
    print(f"\n" + "="*60)
    print(f"📊 OVERALL SUMMARY")
    print(f"="*60)
    print(f"Total images: {total_stats['total_images']:,}")
    print(f"Images with invalid faces: {total_stats['images_with_invalid']:,} ({total_stats['images_with_invalid']/total_stats['total_images']*100:.1f}%)")
    print(f"Total faces: {total_stats['total_faces']:,}")
    print(f"Invalid faces: {total_stats['invalid_faces']:,} ({total_stats['invalid_faces']/total_stats['total_faces']*100:.1f}%)")
    
    print(f"\n💾 Results saved to:")
    print(f"   📄 {output_file}")
    print(f"   📄 {invalid_list_file}")
    print(f"   📄 {event_summary_file}")
    
    if analyze_artwork and results['artwork_analysis']:
        artwork_file = output_dir / 'artwork_candidates.json'
        artwork_summary = {
            'summary': {
                'total_candidates': sum(len(split_data['detected_artwork']) 
                                      for split_data in results['artwork_analysis'].values()),
                'by_split': {split: len(split_data['detected_artwork']) 
                           for split, split_data in results['artwork_analysis'].items()}
            },
            'candidates': results['artwork_analysis']
        }
        
        with open(artwork_file, 'w') as f:
            json.dump(artwork_summary, f, indent=2, ensure_ascii=False)
        
        print(f"   🎨 {artwork_file}")
        print(f"\n🎨 Artwork Analysis:")
        for split, split_data in results['artwork_analysis'].items():
            print(f"   {split}: {len(split_data['detected_artwork'])} artwork candidates")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Find invalid faces and artwork in WiderFace')
    parser.add_argument('--data_dir', type=str, default='./data/widerface',
                       help='Directory containing WiderFace data')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], 
                       default=['train', 'val'],
                       help='Dataset splits to analyze')
    parser.add_argument('--output_dir', type=str, default='./invalid_analysis',
                       help='Output directory for results')
    parser.add_argument('--analyze_artwork', action='store_true',
                       help='Also analyze potential artwork/cartoon images (slower)')
    parser.add_argument('--events_only', action='store_true',
                       help='Only analyze event categories, skip individual image analysis')
    
    args = parser.parse_args()
    
    print(f"🔍 WiderFace Invalid Faces & Artwork Analysis")
    print(f"📂 Data directory: {args.data_dir}")
    print(f"📊 Splits: {args.splits}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"🎨 Artwork analysis: {'Enabled' if args.analyze_artwork else 'Disabled'}")
    
    results = find_invalid_faces(
        data_dir=args.data_dir,
        splits=args.splits,
        output_dir=args.output_dir,
        analyze_artwork=args.analyze_artwork
    )
    
    print(f"\n✅ Analysis completed!")


if __name__ == "__main__":
    main()
