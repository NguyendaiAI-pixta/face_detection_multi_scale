#!/usr/bin/env python3
"""
WiderFace Blur Dataset Generator for Robust Face Detection Training
Tạo bộ dữ liệu blur với phân cấp độ khó và chất lượng face phù hợp
"""

import cv2
import numpy as np
import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

class WiderFaceBlurDatasetGenerator:
    def __init__(self, wider_path):
        """Initialize with WiderFace dataset path"""
        self.wider_path = Path(wider_path)
        self.images_dir = self.wider_path / "images"
        self.labels_dir = self.wider_path / "labels"
        
        # Blur configurations từ nhẹ đến nặng
        self.blur_levels = {
            'light': [
                {'type': 'gaussian', 'strength': 3, 'label': 'Gaussian_Light'},
                {'type': 'motion', 'strength': 5, 'label': 'Motion_Light'},
                {'type': 'radial', 'strength': 2, 'label': 'Radial_Light'}
            ],
            'medium': [
                {'type': 'gaussian', 'strength': 7, 'label': 'Gaussian_Medium'},
                {'type': 'motion', 'strength': 12, 'label': 'Motion_Medium'},
                {'type': 'radial', 'strength': 4, 'label': 'Radial_Medium'}
            ],
            'heavy': [
                {'type': 'gaussian', 'strength': 12, 'label': 'Gaussian_Heavy'},
                {'type': 'motion', 'strength': 19, 'label': 'Motion_Heavy'},
                {'type': 'radial', 'strength': 6, 'label': 'Radial_Heavy'}
            ]
        }
        
        # WiderFace difficulty categories
        self.easy_categories = [
            "22--Picnic", "20--Family_Group", "50--Celebration_Or_Party",
            "21--Festival", "11--Meeting", "49--Greeting"
        ]
        
        self.medium_categories = [
            "12--Group", "13--Interview", "19--Couple", "29--Students_Schoolkids",
            "7--Cheering", "18--Concerts", "28--Sports_Fan"
        ]
        
        self.hard_categories = [
            "3--Riot", "5--Car_Accident", "14--Traffic", "61--Street_Battle",
            "53--Raid", "54--Rescue", "2--Demonstration"
        ]
        
    def get_face_quality_stats(self, label_path, img_width, img_height):
        """Phân tích chất lượng faces trong ảnh"""
        faces = self.load_yolo_labels(label_path, img_width, img_height)
        
        quality_faces = []
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Chỉ lấy faces có kích thước >= 32x32
            if width >= 32 and height >= 32:
                quality_faces.append({
                    'bbox': face['bbox'],
                    'width': width,
                    'height': height,
                    'area': area,
                    'size_category': self.classify_face_size(width, height)
                })
        
        return quality_faces
    
    def classify_face_size(self, width, height):
        """Phân loại kích thước face"""
        min_dim = min(width, height)
        if min_dim >= 96:
            return 'large'
        elif min_dim >= 64:
            return 'medium'
        else:  # >= 32
            return 'small'
    
    def sample_images_by_difficulty(self, target_counts):
        """Sample ảnh theo độ khó với tỷ lệ 30% easy, 50% medium, 20% hard"""
        sampled_images = {
            'easy': [],
            'medium': [], 
            'hard': []
        }
        
        # Sample easy cases (30%)
        for category in self.easy_categories:
            cat_dir = self.images_dir / category
            if cat_dir.exists():
                images = list(cat_dir.glob("*.jpg"))
                sampled = self.filter_quality_images(images, category)
                sampled_images['easy'].extend(sampled[:target_counts['easy']//len(self.easy_categories)])
        
        # Sample medium cases (50%)
        for category in self.medium_categories:
            cat_dir = self.images_dir / category
            if cat_dir.exists():
                images = list(cat_dir.glob("*.jpg"))
                sampled = self.filter_quality_images(images, category)
                sampled_images['medium'].extend(sampled[:target_counts['medium']//len(self.medium_categories)])
        
        # Sample hard cases (20%)
        for category in self.hard_categories:
            cat_dir = self.images_dir / category
            if cat_dir.exists():
                images = list(cat_dir.glob("*.jpg"))
                sampled = self.filter_quality_images(images, category)
                sampled_images['hard'].extend(sampled[:target_counts['hard']//len(self.hard_categories)])
        
        return sampled_images
    
    def filter_quality_images(self, images, category):
        """Lọc ảnh có quality faces >= 32x32"""
        quality_images = []
        
        for img_path in images:
            label_path = self.labels_dir / category / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            try:
                # Load ảnh để lấy dimensions
                temp_img = cv2.imread(str(img_path))
                if temp_img is None:
                    continue
                    
                h, w = temp_img.shape[:2]
                quality_faces = self.get_face_quality_stats(label_path, w, h)
                
                # Chỉ lấy ảnh có ít nhất 1 face chất lượng
                if len(quality_faces) >= 1:
                    quality_images.append({
                        'image_path': img_path,
                        'label_path': label_path,
                        'category': category,
                        'num_quality_faces': len(quality_faces),
                        'quality_faces': quality_faces
                    })
                    
            except Exception as e:
                continue
        
    
    def create_blur_dataset(self, total_images=1000, output_base_dir="/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/blur_dataset"):
        """Tạo bộ dữ liệu blur với phân phối: 30% easy, 50% medium, 20% hard"""
        
        # Tính toán số lượng ảnh cho từng độ khó
        target_counts = {
            'easy': int(total_images * 0.3),    # 30%
            'medium': int(total_images * 0.5),  # 50%
            'hard': int(total_images * 0.2)     # 20%
        }
        
        print(f"🎯 Target Distribution:")
        print(f"   • Easy cases: {target_counts['easy']} images")
        print(f"   • Medium cases: {target_counts['medium']} images") 
        print(f"   • Hard cases: {target_counts['hard']} images")
        print(f"   • Total: {sum(target_counts.values())} images")
        
        # Sample ảnh theo độ khó
        sampled_images = self.sample_images_by_difficulty(target_counts)
        
        # Tạo output directories
        output_dir = Path(output_base_dir)
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Thống kê processing
        stats = defaultdict(int)
        all_results = []
        
        # Process từng difficulty level
        for difficulty, images in sampled_images.items():
            print(f"\n📊 Processing {difficulty.upper()} cases: {len(images)} images")
            
            for i, img_data in enumerate(images):
                print(f"   Processing [{i+1}/{len(images)}]: {img_data['image_path'].name}")
                
                # Tạo blur variants cho mỗi ảnh
                blur_variants = self.generate_blur_variants(
                    img_data, output_dir, difficulty
                )
                
                stats[f'{difficulty}_processed'] += 1
                stats['total_variants'] += len(blur_variants)
                all_results.extend(blur_variants)
        
        # Lưu metadata
        metadata = {
            'dataset_info': {
                'total_original_images': sum(len(images) for images in sampled_images.values()),
                'total_blur_variants': stats['total_variants'],
                'distribution': target_counts,
                'blur_levels': ['light', 'medium', 'heavy'],
                'blur_types': ['gaussian', 'motion', 'radial']
            },
            'processing_stats': dict(stats),
            'blur_configurations': self.blur_levels,
            'results': all_results
        }
        
        metadata_path = output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎉 Dataset Generation Completed!")
        print(f"📊 Final Statistics:")
        print(f"   • Original images processed: {metadata['dataset_info']['total_original_images']}")
        print(f"   • Total blur variants created: {metadata['dataset_info']['total_blur_variants']}")
        print(f"   • Output directory: {output_dir}")
        print(f"   • Metadata saved: {metadata_path}")
        
        return metadata
    
    def generate_blur_variants(self, img_data, output_dir, difficulty):
        """Tạo các blur variants cho 1 ảnh"""
        image_path = img_data['image_path']
        label_path = img_data['label_path']
        category = img_data['category']
        
        # Load ảnh gốc
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        variants = []
        base_name = f"{difficulty}_{category}_{image_path.stem}"
        
        # Tạo blur cho mỗi level (light, medium, heavy)
        for level_name, level_configs in self.blur_levels.items():
            # Chọn ngẫu nhiên 1 blur type từ level này
            blur_config = random.choice(level_configs)
            
            # Apply blur
            blurred_image = self.apply_blur_effects(
                image, blur_config['type'], blur_config['strength']
            )
            
            # Tạo tên file output
            output_name = f"{base_name}_{blur_config['label']}"
            
            # Save ảnh
            img_output_path = output_dir / "images" / f"{output_name}.jpg"
            success = cv2.imwrite(str(img_output_path), blurred_image)
            
            if success:
                # Copy label file
                label_output_path = output_dir / "labels" / f"{output_name}.txt"
                shutil.copy2(label_path, label_output_path)
                
                variants.append({
                    'original_image': str(image_path),
                    'output_image': str(img_output_path),
                    'output_label': str(label_output_path),
                    'difficulty': difficulty,
                    'category': category,
                    'blur_type': blur_config['type'],
                    'blur_strength': blur_config['strength'],
                    'blur_level': level_name,
                    'num_faces': img_data['num_quality_faces']
                })
        
        return variants

def generate_blur_dataset():
    """Tạo bộ dữ liệu blur cho training"""
    
    print("🚀 WiderFace Blur Dataset Generator")
    print("=" * 45)
    
    # Cấu hình
    wider_path = "/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/WIDER_train"
    output_dir = "/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/blur_dataset"
    total_images = 500  # Số ảnh gốc để tạo blur (sẽ tạo ra ~1500 blur variants)
    
    # Khởi tạo generator
    generator = WiderFaceBlurDatasetGenerator(wider_path)
    
    print(f"📊 Configuration:")
    print(f"   • Source dataset: {wider_path}")
    print(f"   • Output directory: {output_dir}")
    print(f"   • Target original images: {total_images}")
    print(f"   • Expected blur variants: ~{total_images * 3} (3 blur levels per image)")
    
    print(f"\n🎛️  Blur Configurations:")
    for level_name, configs in generator.blur_levels.items():
        print(f"   • {level_name.upper()}:")
        for config in configs:
            print(f"     - {config['label']}: strength={config['strength']}")
    
    # Tạo dataset
    try:
        metadata = generator.create_blur_dataset(
            total_images=total_images,
            output_base_dir=output_dir
        )
        
        print(f"\n✅ Success! Created blur dataset for robust face detection training")
        return metadata
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return None

def demo_blur_visualization():
    """Demo visualization một vài ảnh để kiểm tra chất lượng"""
    
    print("\n🎨 Creating Blur Visualization Samples")
    print("=" * 40)
    
    wider_path = "/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/WIDER_train"
    output_dir = "/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/blur_demo_samples"
    
    generator = WiderFaceBlurDatasetGenerator(wider_path)
    
    # Sample một vài ảnh để demo
    sample_counts = {'easy': 2, 'medium': 2, 'hard': 1}
    sampled_images = generator.sample_images_by_difficulty(sample_counts)
    
    results = []
    
    for difficulty, images in sampled_images.items():
        for img_data in images:
            result_path = generator.create_blur_comparison_grid(
                img_data['image_path'], 
                img_data['label_path'],
                output_dir,
                difficulty
            )
            
            if result_path:
                results.append(result_path)
    
    print(f"✅ Created {len(results)} visualization grids in: {output_dir}")
    return results
    
    def load_yolo_labels(self, label_path, img_width, img_height):
        """Load YOLO format labels and convert to pixel coordinates"""
        faces = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x_center, y_center, width, height = map(float, parts[:5])
                    
                    # Convert normalized to pixel coordinates
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    width_px = width * img_width
                    height_px = height * img_height
                    
                    # Convert to x1, y1, x2, y2
                    x1 = int(x_center_px - width_px / 2)
                    y1 = int(y_center_px - height_px / 2)
                    x2 = int(x_center_px + width_px / 2)
                    y2 = int(y_center_px + height_px / 2)
                    
                    faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': int(cls),
                        'confidence': 1.0  # Ground truth
                    })
        
        return faces
    
    def apply_blur_effects(self, image, blur_type='motion', strength=15):
        """Apply different blur effects to entire image"""
        if blur_type == 'motion':
            # Motion blur
            kernel_size = int(strength)
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel_motion_blur = kernel_motion_blur / kernel_size
            return cv2.filter2D(image, -1, kernel_motion_blur)
            
        elif blur_type == 'gaussian':
            # Gaussian blur
            kernel_size = int(strength) * 2 + 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), strength/3)
            
        elif blur_type == 'radial':
            # Radial blur (zoom blur)
            h, w = image.shape[:2]
            center_x, center_y = w//2, h//2
            
            result = np.zeros_like(image)
            num_layers = int(strength)
            
            for i in range(num_layers):
                scale = 1.0 + (i * 0.02)
                M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
                layer = cv2.warpAffine(image, M, (w, h))
                result = cv2.addWeighted(result, i/(i+1), layer, 1/(i+1), 0)
                
            return result.astype(np.uint8)
            
        else:  # default gaussian
            return cv2.GaussianBlur(image, (15, 15), strength/5)
    
    def draw_face_boxes(self, image, faces, color=(0, 255, 0), thickness=2):
        """Draw face bounding boxes on image"""
        result = image.copy()
        
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face['bbox']
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw face number
            cv2.putText(result, f"Face {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            
            # Draw confidence if available
            if 'confidence' in face:
                conf_text = f"{face['confidence']:.2f}"
                cv2.putText(result, conf_text, (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def create_blur_comparison(self, image_path, label_path, output_dir):
        """Create comparison of different blur effects"""
        # Load image and labels
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Cannot load image: {image_path}")
            return None
            
        h, w = image.shape[:2]
        faces = self.load_yolo_labels(label_path, w, h)
        
        print(f"📊 Found {len(faces)} faces in {image_path.name}")
        
        # Blur configurations
        blur_configs = [
            {'type': 'original', 'strength': 0, 'color': (0, 255, 0), 'label': 'Original'},
            {'type': 'gaussian', 'strength': 12, 'color': (255, 0, 0), 'label': 'Gaussian Blur'},
            {'type': 'motion', 'strength': 19, 'color': (0, 0, 255), 'label': 'Motion Blur'},
            {'type': 'radial', 'strength': 3, 'color': (255, 255, 0), 'label': 'Radial Blur'}
        ]
        
        # Create grid
        grid_imgs = []
        for config in blur_configs:
            if config['type'] == 'original':
                blurred = image.copy()
            else:
                blurred = self.apply_blur_effects(image, config['type'], config['strength'])
            
            # Draw bounding boxes
            result = self.draw_face_boxes(blurred, faces, config['color'], 2)
            
            # Add label
            cv2.putText(result, config['label'], (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, config['color'], 2)
            
            # Add face count
            cv2.putText(result, f"Faces: {len(faces)}", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            grid_imgs.append(result)
        
        # Create 2x2 grid
        top_row = np.hstack([grid_imgs[0], grid_imgs[1]])
        bottom_row = np.hstack([grid_imgs[2], grid_imgs[3]])
        final_grid = np.vstack([top_row, bottom_row])
        
        # Save result
        output_path = Path(output_dir) / f"blur_comparison_{image_path.stem}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(output_path), final_grid)
        if success:
            print(f"✅ Saved comparison: {output_path}")
            return str(output_path)
        else:
            print(f"❌ Failed to save: {output_path}")
            return None

def demo_full_blur():
    """Demo làm mờ toàn bộ ảnh với bounding box từ WiderFace"""
    
    print("🌫️  WiderFace Full Image Blur Demo")
    print("=" * 40)
    
    # Cấu hình
    wider_path = "/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/WIDER_train"
    output_dir = "/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/blur_demo_output_new_4"
    
    # Khởi tạo demo
    demo = WiderFaceBlurDemo(wider_path)
    
    print(f"📊 Found {len(demo.demo_images)} sample images")
    
    results = []
    
    for i, sample in enumerate(demo.demo_images[:5]):  # Process 5 samples
        image_path = sample['image_path']
        label_path = sample['label_path']
        category = sample['category']
        
        print(f"\n🖼️  Processing [{i+1}/5]: {image_path.name}")
        print(f"📁 Category: {category}")
        
        try:
            result_path = demo.create_blur_comparison(
                image_path, label_path, output_dir
            )
            
            if result_path:
                results.append({
                    'source_image': str(image_path),
                    'category': category,
                    'output_path': result_path,
                    'num_faces': len(demo.load_yolo_labels(
                        label_path, 
                        cv2.imread(str(image_path)).shape[1],
                        cv2.imread(str(image_path)).shape[0]
                    ))
                })
                
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
    
    # Save metadata
    if results:
        metadata = {
            'total_processed': len(results),
            'dataset': 'WiderFace',
            'blur_effects': ['Original', 'Gaussian', 'Motion', 'Radial'],
            'results': results
        }
        
        metadata_path = Path(output_dir) / "demo_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📝 Metadata saved: {metadata_path}")
    
    print(f"\n🎉 Demo completed! Generated {len(results)} comparison images")
    print(f"📁 Output directory: {output_dir}")
    
    return results
def print_summary():
    """In summary về các tính năng"""
    
    print(f"\n� Tính Năng Demo:")
    print("=" * 25)
    
    features = [
        "🌫️  Full image blur với WiderFace dataset",
        "📦 Ground truth bounding boxes từ YOLO labels", 
        "🎨 Multiple blur effects (Gaussian, Motion, Radial)",
        "🎛️  Adjustable blur strength parameters",
        "🌈 Color-coded bounding boxes cho mỗi effect",
        "📊 2x2 comparison grid visualization",
        "⚡ Batch processing multiple sample images",
        "📝 Metadata export với face statistics",
        "📁 Organized output structure"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n💡 Training Applications:")
    print("   • Face detection under camera blur conditions")
    print("   • Robust model training với degraded image quality") 
    print("   • Data augmentation cho adverse weather/motion")
    print("   • Evaluation của detection models trong blur scenarios")
    
    print(f"\n📁 WiderFace Categories Used:")
    categories = [
        "22--Picnic (outdoor group scenes)",
        "20--Family_Group (family gatherings)", 
        "12--Group (general group photos)",
        "50--Celebration_Or_Party (party scenes)",
        "11--Meeting (meeting scenarios)"
    ]
    for cat in categories:
        print(f"   • {cat}")

if __name__ == "__main__":
    # Chạy demo
    results = demo_full_blur()
    
    # In summary
    print_summary()
    
    if results:
        print(f"\n� Processing Summary:")
        total_faces = sum(r['num_faces'] for r in results)
        print(f"   • Total images processed: {len(results)}")
        print(f"   • Total faces detected: {total_faces}")
        print(f"   • Average faces per image: {total_faces/len(results):.1f}")
        
        print(f"\n🖼️  Generated Images:")
        for result in results:
            print(f"   • {Path(result['output_path']).name} ({result['num_faces']} faces)")
        
        print(f"\n💡 Next Steps:")
        print("   • Use blurred images for training data augmentation")
        print("   • Test face detection models on these challenging conditions")
        print("   • Experiment with different blur parameters và strengths")
