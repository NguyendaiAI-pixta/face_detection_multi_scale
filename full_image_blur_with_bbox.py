#!/usr/bin/env python3
"""
Full Image Blur with Face Bounding Box Overlay
Làm mờ toàn bộ ảnh và vẽ bounding box lên trên
với 3 bộ lọc chính
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random
from typing import Tuple, List, Dict
import math

class FullImageBlurWithBBox:
    """
    Làm mờ toàn bộ ảnh và vẽ bounding box face lên trên
    """
    
    def __init__(self, csv_path: str, images_dir: str):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.bbox_data = self.load_bbox_data()
    
    def load_bbox_data(self):
        """Load dữ liệu bounding box"""
        try:
            df = pd.read_csv(self.csv_path)
            bbox_dict = {}
            
            for _, row in df.iterrows():
                # Format đúng: x, y, width, height
                x = int(row['x_1'])
                y = int(row['y_1'])
                width = int(row['width'])
                height = int(row['height'])
                
                if width > 0 and height > 0:
                    bbox_dict[row['image_id']] = {
                        'x': x, 'y': y, 
                        'width': width, 'height': height
                    }
            
            return bbox_dict
        except Exception as e:
            print(f"Lỗi load CSV: {e}")
            return {}
    
    def create_motion_blur_full(self, image: np.ndarray, kernel_size: int = 15, angle: float = 0) -> np.ndarray:
        """Làm mờ toàn bộ ảnh với motion blur"""
        # Tạo motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Xoay kernel theo góc
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        return cv2.filter2D(image, -1, kernel)
    
    def create_bokeh_blur_full(self, image: np.ndarray, bokeh_strength: float = 2.0) -> np.ndarray:
        """Làm mờ toàn bộ ảnh với bokeh effect"""
        kernel_size = int(bokeh_strength * 8)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Tạo bokeh kernel
        kernel = cv2.getGaussianKernel(kernel_size, bokeh_strength)
        kernel = np.outer(kernel, kernel)
        
        # Tăng cường highlights
        enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        blurred = cv2.filter2D(enhanced, -1, kernel)
        
        return blurred
    
    def create_lens_blur_full(self, image: np.ndarray, blur_strength: float = 2.5) -> np.ndarray:
        """Làm mờ toàn bộ ảnh với lens blur"""
        blurred = cv2.bilateralFilter(image, 15, 35, 35)
        blurred = cv2.GaussianBlur(blurred, (0, 0), blur_strength * 0.8)
        return blurred
    
    def add_noise_effect(self, image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Thêm nhiễu để mô phỏng điều kiện chụp kém"""
        noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    def get_blur_strength_preset(self, blur_type: str, intensity: str = 'medium') -> float:
        """
        Lấy preset cường độ blur
        
        Args:
            blur_type: 'motion', 'bokeh', 'lens'
            intensity: 'light', 'medium', 'strong'
        
        Returns:
            Blur strength value
        """
        presets = {
            'motion': {
                'light': random.randint(8, 12),     # Nhẹ
                'medium': random.randint(15, 20),   # Vừa  
                'strong': random.randint(25, 35)    # Mạnh
            },
            'bokeh': {
                'light': random.uniform(1.0, 1.5),   # f/4.0-f/2.8
                'medium': random.uniform(2.0, 2.5),  # f/2.0-f/1.8
                'strong': random.uniform(3.0, 4.0)   # f/1.4-f/1.0
            },
            'lens': {
                'light': random.uniform(1.5, 2.0),   # f/5.6-f/4.0
                'medium': random.uniform(2.5, 3.0),  # f/2.8-f/2.0
                'strong': random.uniform(3.5, 5.0)   # f/1.8-f/1.0
            }
        }
        
        if blur_type in presets and intensity in presets[blur_type]:
            return presets[blur_type][intensity]
        else:
            print(f"Invalid preset: {blur_type}/{intensity}")
            return 2.0  # Default value
    
    def scale_bbox_to_current_image(self, bbox: Dict, original_size: Tuple, current_size: Tuple) -> Dict:
        """Scale bounding box từ kích thước gốc về kích thước hiện tại"""
        orig_w, orig_h = original_size
        curr_w, curr_h = current_size
        
        scale_x = curr_w / orig_w
        scale_y = curr_h / orig_h
        
        scaled_bbox = {
            'x': max(0, int(bbox['x'] * scale_x)),
            'y': max(0, int(bbox['y'] * scale_y)),
            'width': int(bbox['width'] * scale_x),
            'height': int(bbox['height'] * scale_y)
        }
        
        # Đảm bảo không vượt quá kích thước ảnh
        scaled_bbox['width'] = min(scaled_bbox['width'], curr_w - scaled_bbox['x'])
        scaled_bbox['height'] = min(scaled_bbox['height'], curr_h - scaled_bbox['y'])
        
        return scaled_bbox
    
    def draw_bbox_on_image(self, image: np.ndarray, bbox: Dict, 
                          bbox_color: Tuple = (255, 0, 0), 
                          bbox_thickness: int = 3,
                          add_label: bool = True) -> np.ndarray:
        """Vẽ bounding box lên ảnh"""
        image_with_bbox = image.copy()
        
        # Format đúng: x, y, width, height -> cần chuyển sang (x1, y1), (x2, y2)
        x1 = bbox['x']
        y1 = bbox['y']
        x2 = bbox['x'] + bbox['width']
        y2 = bbox['y'] + bbox['height']
        
        # Vẽ rectangle
        cv2.rectangle(image_with_bbox, 
                     (x1, y1), 
                     (x2, y2), 
                     bbox_color, 
                     bbox_thickness)
        
        if add_label:
            # Vẽ label
            label_text = f"Face ({bbox['width']}x{bbox['height']})"
            
            # Tính vị trí label
            label_x = x1
            label_y = y1 - 10 if y1 > 30 else y1 + 25
            
            # Background cho text
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(image_with_bbox,
                         (label_x, label_y - text_height - 5),
                         (label_x + text_width, label_y + baseline),
                         bbox_color, -1)
            
            # Vẽ text
            cv2.putText(image_with_bbox, label_text,
                       (label_x, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image_with_bbox
    
    def apply_full_blur_with_bbox(self, image_path: str, 
                                 blur_type: str = 'bokeh',
                                 blur_strength: float = None,
                                 blur_intensity: str = None,
                                 add_noise: bool = False,
                                 bbox_color: Tuple = (0, 255, 0),
                                 output_path: str = None) -> Dict:
        """
        Áp dụng blur cho toàn bộ ảnh và vẽ bounding box
        
        Args:
            image_path: Đường dẫn ảnh
            blur_type: Loại blur ['motion', 'bokeh', 'lens']
            blur_strength: Độ mạnh blur (None = random, hoặc dùng blur_intensity)
            blur_intensity: Preset intensity ['light', 'medium', 'strong'] (nếu blur_strength=None)
            add_noise: Có thêm noise không
            bbox_color: Màu bounding box (B, G, R)
            output_path: Đường dẫn lưu kết quả
            
        Returns:
            Dictionary thông tin kết quả
        """
        
        image_id = os.path.basename(image_path)
        
        if image_id not in self.bbox_data:
            print(f"Không tìm thấy bounding box cho {image_id}")
            return None
        
        try:
            # Load ảnh
            image = cv2.imread(image_path)
            if image is None:
                print(f"Không thể load ảnh: {image_path}")
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Lấy bounding box (không cần scale vì đã chính xác)
            bbox = self.bbox_data[image_id]
            
            # Kiểm tra bbox có hợp lệ với kích thước ảnh không
            if (bbox['x'] + bbox['width'] > w or 
                bbox['y'] + bbox['height'] > h or
                bbox['x'] < 0 or bbox['y'] < 0):
                print(f"⚠️  Bbox không hợp lệ cho ảnh {image_id}: {bbox}")
                # Clip bbox về phạm vi hợp lệ
                bbox = {
                    'x': max(0, min(bbox['x'], w-1)),
                    'y': max(0, min(bbox['y'], h-1)),
                    'width': min(bbox['width'], w - bbox['x']),
                    'height': min(bbox['height'], h - bbox['y'])
                }
                print(f"   Đã clip về: {bbox}")
            
            scaled_bbox = bbox  # Không cần scale
            
            # Áp dụng blur cho toàn bộ ảnh
            if blur_strength is None:
                if blur_intensity:
                    # Sử dụng preset intensity
                    blur_strength = self.get_blur_strength_preset(blur_type, blur_intensity)
                else:
                    # Random mặc định
                    if blur_type == 'motion':
                        blur_strength = random.randint(30, 45)
                    elif blur_type in ['bokeh', 'lens']:
                        blur_strength = random.uniform(3.0, 5.0)
            
            if blur_type == 'motion':
                angle = random.uniform(0, 180)
                blurred_image = self.create_motion_blur_full(image_rgb, int(blur_strength), angle)
            elif blur_type == 'bokeh':
                blurred_image = self.create_bokeh_blur_full(image_rgb, blur_strength)
            elif blur_type == 'lens':
                blurred_image = self.create_lens_blur_full(image_rgb, blur_strength)
            else:
                print(f"Blur type không hỗ trợ: {blur_type}")
                print("Các loại hỗ trợ: 'motion', 'bokeh', 'lens'")
                return None
            
            # Thêm noise nếu cần
            if add_noise:
                blurred_image = self.add_noise_effect(blurred_image, random.uniform(0.05, 0.15))
            
            # Vẽ bounding box lên ảnh đã blur
            final_image = self.draw_bbox_on_image(
                blurred_image, bbox, bbox_color, bbox_thickness=3
            )
            
            # Lưu kết quả
            if output_path:
                final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, final_bgr)
            
            return {
                'original_image': image_id,
                'blur_type': blur_type,
                'blur_strength': blur_strength,
                'image_size': (w, h),
                'bbox': bbox,
                'output_path': output_path,
                'blurred_image': final_image,
                'add_noise': add_noise
            }
            
        except Exception as e:
            print(f"Lỗi khi xử lý {image_path}: {e}")
            return None
    
    def create_blur_comparison_grid(self, image_path: str, save_path: str = None) -> None:
        """Tạo grid so sánh các loại blur"""
        image_id = os.path.basename(image_path)
        
        if image_id not in self.bbox_data:
            print(f"Không tìm thấy bounding box cho {image_id}")
            return
        
        # Load ảnh gốc
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        blur_types = ['motion', 'bokeh', 'lens']
        
        # Tạo các phiên bản blur
        results = {}
        
        # Ảnh gốc với bbox
        bbox = self.bbox_data[image_id]
        h, w = original_rgb.shape[:2]
        
        # Kiểm tra bbox có hợp lệ không
        if (bbox['x'] + bbox['width'] > w or 
            bbox['y'] + bbox['height'] > h or
            bbox['x'] < 0 or bbox['y'] < 0):
            # Clip bbox về phạm vi hợp lệ
            bbox = {
                'x': max(0, min(bbox['x'], w-1)),
                'y': max(0, min(bbox['y'], h-1)),
                'width': min(bbox['width'], w - bbox['x']),
                'height': min(bbox['height'], h - bbox['y'])
            }
        
        results['Original'] = self.draw_bbox_on_image(
            original_rgb, bbox, (0, 255, 0)
        )
        
        # Tạo các blur variants
        for blur_type in blur_types:
            result = self.apply_full_blur_with_bbox(
                image_path, 
                blur_type=blur_type,
                bbox_color=(255, 0, 0)  # Red for blur variants
            )
            
            if result:
                results[blur_type.title()] = result['blurred_image']
        
        # Visualize grid (2x2 cho 4 ảnh: Original + 3 blur types)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (method_name, blurred_image) in enumerate(results.items()):
            if idx < len(axes):
                axes[idx].imshow(blurred_image)
                axes[idx].set_title(f'{method_name} + BBox', fontsize=14, weight='bold')
                axes[idx].axis('off')
        
        # Ẩn subplot thừa nếu có
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Full Image Blur with Face BBox - {image_id}', 
                    fontsize=18, weight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Grid comparison saved: {save_path}")
        
        plt.show()
    
    def batch_process_full_blur(self, num_images: int = 10, 
                               output_dir: str = "full_blur_dataset",
                               blur_types: List[str] = None,
                               add_noise_prob: float = 0.3) -> Dict:
        """Xử lý batch nhiều ảnh với full blur + bbox"""
        
        if blur_types is None:
            blur_types = ['motion', 'bokeh', 'lens']
        
        # Lấy ảnh ngẫu nhiên
        available_images = list(self.bbox_data.keys())
        selected_images = random.sample(available_images, min(num_images, len(available_images)))
        
        os.makedirs(output_dir, exist_ok=True)
        
        stats = {'processed': 0, 'total_created': 0, 'failed': 0}
        results = []
        
        for image_id in selected_images:
            image_path = os.path.join(self.images_dir, image_id)
            
            if not os.path.exists(image_path):
                stats['failed'] += 1
                continue
            
            image_results = []
            
            # Tạo multiple blur variants cho mỗi ảnh
            for blur_type in blur_types:
                add_noise = random.random() < add_noise_prob
                
                output_filename = f"{os.path.splitext(image_id)[0]}_{blur_type}_full_blur.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                result = self.apply_full_blur_with_bbox(
                    image_path,
                    blur_type=blur_type,
                    add_noise=add_noise,
                    output_path=output_path
                )
                
                if result:
                    image_results.append(result)
                    stats['total_created'] += 1
            
            if image_results:
                stats['processed'] += 1
                results.extend(image_results)
                print(f"✅ Processed {image_id}: {len(image_results)} variants")
            else:
                stats['failed'] += 1
                print(f"❌ Failed {image_id}")
        
        return {'stats': stats, 'results': results}

# Demo usage
if __name__ == "__main__":
    csv_path = "/Users/nguyenvandai/facedetection/datasets/jessicali9530/celeba-dataset/versions/2/list_bbox_celeba.csv"
    images_dir = "/Users/nguyenvandai/facedetection"
    
    print("=== Full Image Blur with Face BBox Demo ===")
    
    # Khởi tạo tool
    blur_tool = FullImageBlurWithBBox(csv_path, images_dir)
    
    # Test với một ảnh
    test_image = os.path.join(images_dir, "000059.jpg")
    
    if os.path.exists(test_image):
        print(f"\n📸 Demo với ảnh: 000059.jpg")
        
        # Tạo comparison grid
        blur_tool.create_blur_comparison_grid(
            test_image,
            save_path="/Users/nguyenvandai/facedetection/full_blur_comparison.png"
        )
        
        # Tạo một vài samples
        blur_types = ['motion', 'bokeh', 'lens']
        for blur_type in blur_types:
            output_path = f"/Users/nguyenvandai/facedetection/demo_{blur_type}_full_blur.jpg"
            
            result = blur_tool.apply_full_blur_with_bbox(
                test_image,
                blur_type=blur_type,
                add_noise=True,
                output_path=output_path
            )
            
            if result:
                print(f"✅ Created {blur_type} sample: {output_path}")
    
    else:
        print("❌ Test image not found")
