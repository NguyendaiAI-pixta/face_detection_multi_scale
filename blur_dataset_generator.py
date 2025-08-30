#!/usr/bin/env python3
"""
WiderFace Blur Dataset Generator for Robust Face Detection Training
Tạo bộ dữ liệu blur với phân cấp độ khó và chất lượng face phù hợp cho training
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
    def create_blur_val(self, wider_val_path, output_base_dir, blur_level='medium'):
        """Tạo tập val mờ từ tập val của WiderFace"""
        print(f"\n🔄 Generating blurred val set from WiderFace val...")
        images_dir = Path(wider_val_path) / "images"
        labels_dir = Path(wider_val_path) / "labels"
        output_img_dir = Path(output_base_dir) / "val" / "images"
        output_lbl_dir = Path(output_base_dir) / "val" / "labels"
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_lbl_dir.mkdir(parents=True, exist_ok=True)
        # Chọn blur config
        blur_configs = self.blur_levels.get(blur_level, self.blur_levels['medium'])
        blur_config = random.choice(blur_configs)
        count = 0
        for category_dir in images_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for img_file in category_dir.glob("*.jpg"):
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                # Apply blur
                blurred_image = self.apply_blur_effects(
                    image, blur_config['type'], blur_config['strength']
                )
                # Tạo tên file output
                output_name = f"{category_dir.name}_{img_file.stem}_{blur_config['label']}"
                img_output_path = output_img_dir / f"{output_name}.jpg"
                success = cv2.imwrite(str(img_output_path), blurred_image)
                if success:
                    # Copy label file
                    label_file = labels_dir / category_dir.name / f"{img_file.stem}.txt"
                    if label_file.exists():
                        label_output_path = output_lbl_dir / f"{output_name}.txt"
                        shutil.copy2(label_file, label_output_path)
                        count += 1
        print(f"✅ Blurred val set created: {count} images in {output_img_dir}")
    def __init__(self, wider_path):
        """Initialize với WiderFace dataset path"""
        self.wider_path = Path(wider_path)
        self.images_dir = self.wider_path / "images"
        self.labels_dir = self.wider_path / "labels"
        
        # Blur configurations từ nhẹ đến nặng - điều chỉnh hợp lý cho training
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
        
        # WiderFace difficulty categories - phân loại theo độ khó thực tế
        self.easy_categories = [
            "22--Picnic", "20--Family_Group", "50--Celebration_Or_Party",
            "21--Festival", "11--Meeting", "49--Greeting", "19--Couple"
        ]
        
        self.medium_categories = [
            "12--Group", "13--Interview", "29--Students_Schoolkids",
            "7--Cheering", "18--Concerts", "28--Sports_Fan", "23--Shoppers",
            "52--Photographers", "8--Election_Campain"
        ]
        
        self.hard_categories = [
            "3--Riot", "5--Car_Accident", "14--Traffic", "61--Street_Battle",
            "53--Raid", "54--Rescue", "2--Demonstration", "4--Dancing",
            "24--Soldier_Firing", "34--Baseball"
        ]
    
    def load_yolo_labels(self, label_path, img_width, img_height):
        """Load YOLO format labels và convert sang pixel coordinates"""
        faces = []
        
        if not label_path.exists():
            return faces
            
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
                        'confidence': 1.0
                    })
        
        return faces
    
    def get_face_quality_stats(self, label_path, img_width, img_height):
        """Phân tích chất lượng faces trong ảnh - chỉ lấy faces >= 32x32"""
        faces = self.load_yolo_labels(label_path, img_width, img_height)
        
        quality_faces = []
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Chỉ lấy faces có kích thước >= 32x32 như yêu cầu
            if width >= 32 and height >= 32:
                quality_faces.append({
                    'bbox': face['bbox'],
                    'width': width,
                    'height': height,
                    'area': width * height,
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
        print(f"🔍 Sampling easy cases...")
        for category in self.easy_categories:
            cat_dir = self.images_dir / category
            if cat_dir.exists():
                images = list(cat_dir.glob("*.jpg"))
                random.shuffle(images)  # Shuffle để diversity
                sampled = self.filter_quality_images(images, category)
                needed = target_counts['easy'] // len(self.easy_categories)
                sampled_images['easy'].extend(sampled[:needed])
        
        # Sample medium cases (50%)
        print(f"🔍 Sampling medium cases...")
        for category in self.medium_categories:
            cat_dir = self.images_dir / category
            if cat_dir.exists():
                images = list(cat_dir.glob("*.jpg"))
                random.shuffle(images)
                sampled = self.filter_quality_images(images, category)
                needed = target_counts['medium'] // len(self.medium_categories)
                sampled_images['medium'].extend(sampled[:needed])
        
        # Sample hard cases (20%)
        print(f"🔍 Sampling hard cases...")
        for category in self.hard_categories:
            cat_dir = self.images_dir / category
            if cat_dir.exists():
                images = list(cat_dir.glob("*.jpg"))
                random.shuffle(images)
                sampled = self.filter_quality_images(images, category)
                needed = target_counts['hard'] // len(self.hard_categories)
                sampled_images['hard'].extend(sampled[:needed])
        
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
                
                # Chỉ lấy ảnh có ít nhất 1 face chất lượng >= 32x32
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
        
        # Sort theo số lượng quality faces (ưu tiên ảnh có nhiều faces)
        quality_images.sort(key=lambda x: x['num_quality_faces'], reverse=True)
        return quality_images
    
    def apply_blur_effects(self, image, blur_type='gaussian', strength=5):
        """Apply blur effects với parameters được điều chỉnh hợp lý"""
        if blur_type == 'gaussian':
            # Gaussian blur - kernel size phải lẻ
            kernel_size = int(strength) * 2 + 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), strength/3)
            
        elif blur_type == 'motion':
            # Motion blur - horizontal motion
            kernel_size = int(strength)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel_motion_blur = kernel_motion_blur / kernel_size
            return cv2.filter2D(image, -1, kernel_motion_blur)
            
        elif blur_type == 'radial':
            # Radial blur (zoom blur) - từ center ra ngoài
            h, w = image.shape[:2]
            center_x, center_y = w//2, h//2
            
            result = np.zeros_like(image, dtype=np.float64)
            num_layers = max(int(strength), 2)
            
            for i in range(num_layers):
                scale = 1.0 + (i * 0.015)  # Giảm scale factor để không quá mạnh
                M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
                layer = cv2.warpAffine(image, M, (w, h))
                result = cv2.addWeighted(result, i/(i+1), layer.astype(np.float64), 1/(i+1), 0)
                
            return np.clip(result, 0, 255).astype(np.uint8)
            
        else:
            # Default gaussian
            return cv2.GaussianBlur(image, (15, 15), strength/5)
    
    def create_blur_dataset(self, total_images=500, output_base_dir="/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/blur_dataset"):
        """Tạo bộ dữ liệu blur với phân phối theo ý tưởng của bạn"""
        
        # Tính toán số lượng theo ý tưởng: 30% easy, 50% medium, 20% hard
        target_counts = {
            'easy': int(total_images * 0.3),    # 30%
            'medium': int(total_images * 0.5),  # 50%
            'hard': int(total_images * 0.2)     # 20%
        }
        
        print(f"🎯 Target Distribution (theo ý tưởng của bạn):")
        print(f"   • Easy cases: {target_counts['easy']} images (30%)")
        print(f"   • Medium cases: {target_counts['medium']} images (50%)") 
        print(f"   • Hard cases: {target_counts['hard']} images (20%)")
        print(f"   • Total original: {sum(target_counts.values())} images")
        print(f"   • Expected blur variants: ~{sum(target_counts.values()) * 3} (light, medium, heavy)")
        
        # Sample ảnh theo độ khó
        sampled_images = self.sample_images_by_difficulty(target_counts)
        
        # Verify sampling results
        actual_counts = {k: len(v) for k, v in sampled_images.items()}
        print(f"\n📊 Actual Sampled:")
        for difficulty, count in actual_counts.items():
            print(f"   • {difficulty.capitalize()}: {count} images")
        
        # Tạo output directories
        output_dir = Path(output_base_dir)
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Thống kê processing
        stats = defaultdict(int)
        all_results = []
        
        # Process từng difficulty level
        for difficulty, images in sampled_images.items():
            if not images:
                continue
                
            print(f"\n📊 Processing {difficulty.upper()} cases: {len(images)} images")
            
            for i, img_data in enumerate(images):
                print(f"   [{i+1:3d}/{len(images)}] {img_data['image_path'].name} (Faces: {img_data['num_quality_faces']})")
                
                # Tạo blur variants - phân bổ đều light, medium, heavy
                blur_variants = self.generate_blur_variants(
                    img_data, output_dir, difficulty
                )
                
                stats[f'{difficulty}_processed'] += 1
                stats['total_variants'] += len(blur_variants)
                all_results.extend(blur_variants)
        
        # Lưu metadata
        metadata = {
            'dataset_info': {
                'creation_date': str(Path().cwd()),
                'source_dataset': 'WiderFace',
                'total_original_images': sum(len(images) for images in sampled_images.values()),
                'total_blur_variants': stats['total_variants'],
                'distribution_target': target_counts,
                'distribution_actual': actual_counts,
                'blur_levels': ['light', 'medium', 'heavy'],
                'blur_types': ['gaussian', 'motion', 'radial'],
                'min_face_size': '32x32 pixels'
            },
            'processing_stats': dict(stats),
            'blur_configurations': self.blur_levels,
            'difficulty_categories': {
                'easy': self.easy_categories,
                'medium': self.medium_categories,
                'hard': self.hard_categories
            },
            'results': all_results
        }
        
        metadata_path = output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎉 Blur Dataset Generation Completed!")
        print(f"📊 Final Statistics:")
        print(f"   • Original images processed: {metadata['dataset_info']['total_original_images']}")
        print(f"   • Total blur variants created: {metadata['dataset_info']['total_blur_variants']}")
        print(f"   • Average variants per image: {metadata['dataset_info']['total_blur_variants'] / max(1, metadata['dataset_info']['total_original_images']):.1f}")
        print(f"   • Output directory: {output_dir}")
        print(f"   • Metadata saved: {metadata_path}")
        
        self.print_usage_instructions(output_dir)
        
        return metadata
    
    def generate_blur_variants(self, img_data, output_dir, difficulty):
        """Tạo blur variants theo phân bổ đều light, medium, heavy"""
        image_path = img_data['image_path']
        label_path = img_data['label_path']
        category = img_data['category']
        
        # Load ảnh gốc
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        variants = []
        base_name = f"{difficulty}_{category.replace('--', '_')}_{image_path.stem}"
        
        # Tạo blur cho mỗi level - phân bổ đều như yêu cầu
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
                # Copy label file (giữ nguyên bounding box)
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
    
    def print_usage_instructions(self, output_dir):
        """In hướng dẫn sử dụng dataset cho training"""
        print(f"\n📖 Usage Instructions:")
        print(f"=" * 50)
        print(f"1. 📁 Dataset Structure:")
        print(f"   {output_dir}/")
        print(f"   ├── images/           # Blur images")
        print(f"   ├── labels/           # YOLO format labels") 
        print(f"   └── dataset_metadata.json")
        
        print(f"\n2. 🔗 Integration với Original Dataset:")
        print(f"   • Merge vào WIDER_train bằng symlink hoặc copy")
        print(f"   • Update data/widerface.yaml để include blur data")
        print(f"   • Hoặc tạo riêng config cho combined dataset")
        
        print(f"\n3. 🚀 Training Commands:")
        print(f"   # Option 1: Train trên blur data only") 
        print(f"   python train.py --data blur_dataset.yaml")
        print(f"   ")
        print(f"   # Option 2: Combine với original data")
        print(f"   python train.py --data combined_widerface.yaml")
        
        print(f"\n4. 📊 Expected Benefits:")
        print(f"   • Improved robustness trong adverse conditions")
        print(f"   • Better generalization cho camera blur")
        print(f"   • Enhanced performance trong real-world scenarios")

def main():
    """Main function để chạy dataset generation"""
    
    print("🚀 WiderFace Blur Dataset Generator")
    print("=" * 50)
    print("Tạo dataset blur cho robust face detection training")
    print("Theo ý tưởng: 30% easy, 50% medium, 20% hard cases")
    print("Với face size >= 32x32, blur levels: light/medium/heavy")
    
    # Configuration
    wider_path = "/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/WIDER_train"
    output_dir = "/mnt/md0/projects/nguyendai-footage/blur_dataset"
    total_images = 600  # Số ảnh gốc để tạo blur variants
    
    # Validate input
    if not Path(wider_path).exists():
        print(f"❌ Source dataset not found: {wider_path}")
        return
    
    # Initialize generator
    generator = WiderFaceBlurDatasetGenerator(wider_path)
    
    print(f"\n📊 Configuration:")
    print(f"   • Source: {wider_path}")
    print(f"   • Output: {output_dir}")
    print(f"   • Target images: {total_images}")
    print(f"   • Expected variants: ~{total_images * 3}")
    print(f"   • Min face size: 32x32 pixels")
    
    print(f"\n🎛️  Blur Configurations:")
    for level_name, configs in generator.blur_levels.items():
        print(f"   • {level_name.upper()}:")
        for config in configs:
            print(f"     - {config['type'].capitalize()}: strength={config['strength']}")
    
    # Generate dataset
    try:
        print(f"\n🔄 Starting dataset generation...")
        metadata = generator.create_blur_dataset(
            total_images=total_images,
            output_base_dir=output_dir
        )
        # Tạo tập val mờ từ WiderFace val
        wider_val_path = "/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/WIDER_val"
        generator.create_blur_val(
            wider_val_path=wider_val_path,
            output_base_dir=output_dir,
            blur_level='medium'  # hoặc 'light', 'heavy' nếu muốn
        )
        print(f"\n✅ SUCCESS! Blur dataset created successfully")
        print(f"Ready for training với robust face detection!")
        return metadata
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
