import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time
import glob
import json
import re
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Import từ YOLOv7 Face
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device, time_synchronized

# Import từ MultiScaleFaceDetector
from multi_scale_face_detector import MultiScaleFaceDetector

# Import shared utilities
from utils.preprocess_yolo_predict import (
    normalize_bbox, denormalize_bbox, draw_faces_on_image,
    get_image_paths_from_base, find_images_in_directory,
    load_yolo_model, create_yolo_json_format, save_json_results,
    calculate_face_statistics, print_processing_summary,
    scale_coords_api_approach  # Thêm coordinate scaling function
)



# Global config
MODEL_PATH = "auto_review_yolo_face_module_weight.pt"  # Model weights
JSON_OUTPUT_DIR = "./api_predict_json_results_df_multi_scale_640_3840_new"
MAX_FACES_IMAGES_DIR = "./api_predict_max_faces_images_640_3840_new"
BASE_IMAGE_PATH = "/mnt/md0/projects/auto_review_footage/"  # Đường dẫn gốc đến ảnh

# Create output directories
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
os.makedirs(MAX_FACES_IMAGES_DIR, exist_ok=True)

# Biến global cho cấu hình multiprocessing
SKIP_PROCESSED = False  # Có bỏ qua những item đã xử lý không

# Detection parameters
NUM_GPU = 3  # Số lượng GPU thực tế (0,1,2)
MAX_ITEMS = 24000  # Giới hạn items
CONF_THRES = 0.6  # Confidence threshold
IOU_THRES = 0.3  # NMS threshold
IMG_SIZES = [640, 3840]  # List các kích thước ảnh cho multi-scale detection
NUM_WORKERS = 30  # Số lượng worker cho xử lý đa luồng

# CUDA Environment Setup - Let workers handle GPU assignment
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  # Commented out - workers will set this
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Use PCI bus order
os.environ['OMP_NUM_THREADS'] = '1'  # Optimize for multi-GPU

# normalize_bbox đã được import từ utils.preprocess_yolo_predict


# draw_faces_on_image và get_image_paths_from_base đã được import từ utils.preprocess_yolo_predict


def create_detector(model_path=MODEL_PATH, device='', img_sizes=IMG_SIZES, 
                    conf_thres=CONF_THRES, iou_thres=IOU_THRES):
    """
    Tạo một MultiScaleFaceDetector với GPU selection
    
    Args:
        device: GPU device ('', 'cpu', '0', '1', '2', hoặc '0,1,2')
    """
    # Initialize CUDA in the new process
    if torch.cuda.is_available():
        torch.cuda.init()
    
    # Since init_worker already set CUDA_VISIBLE_DEVICES, we always use device '0'
    # which will map to the actual GPU assigned to this worker
    device_str = '0' if torch.cuda.is_available() else 'cpu'
    
    current_process = multiprocessing.current_process()
    assigned_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
    print(f"🔧 Creating detector on device: {device_str} (Process: {current_process.name}, Assigned GPU: {assigned_gpu})")
    
    # Chọn device - always use '0' since CUDA_VISIBLE_DEVICES is set by init_worker
    device_obj = select_device(device_str)
    
    # Khởi tạo detector
    detector = MultiScaleFaceDetector(
        model_path=model_path,
        device=device_str,
        img_sizes=img_sizes,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        use_api_preprocess=True
    )
    
    return detector


class MultiScaleFaceDataFramePredictor:
    """
    Phát hiện khuôn mặt với Multi-scale TTA và lưu kết quả vào DataFrame
    """
    
    def __init__(self, detector, save_images=True, save_dir='results', base_image_path=None, num_workers=10):
        """
        Khởi tạo Multi-scale Face DataFrame Predictor
        
        Args:
            detector: MultiScaleFaceDetector đã được khởi tạo
            save_images: Có lưu ảnh kết quả hay không
            save_dir: Thư mục lưu kết quả
            base_image_path: Đường dẫn cơ sở cho các ảnh (prefix cho relative paths)
            num_workers: Số lượng worker cho xử lý đa luồng
        """
        self.detector = detector
        self.save_images = save_images
        self.save_dir = save_dir
        self.base_image_path = base_image_path or ""
        self.num_workers = min(num_workers, multiprocessing.cpu_count())
        
        print(f"🧵 Using {self.num_workers} workers for parallel processing")
        
        # Tạo thư mục lưu kết quả nếu chưa tồn tại
        if self.save_images and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"✅ Created directory: {self.save_dir}")
    
    def process_image(self, img_path, save_visualization=True):
        """
        Xử lý một ảnh và trả về thông tin detections dưới dạng DataFrame
        
        Args:
            img_path: Đường dẫn ảnh (relative hoặc absolute path)
            save_visualization: Có lưu ảnh visualization không
            
        Returns:
            df: DataFrame chứa thông tin detections
        """
        # Kết hợp base_image_path nếu img_path không phải là absolute path
        full_img_path = img_path
        if not os.path.isabs(img_path) and self.base_image_path:
            full_img_path = os.path.join(self.base_image_path, img_path)
            
        # Phát hiện khuôn mặt bằng multi-scale detector
        final_detections, img0_shape = self.detector.detect_multi_scale(full_img_path)
        
        # Lưu ảnh kết quả nếu được yêu cầu
        if self.save_images:
            img_name = os.path.basename(full_img_path)
            output_path = os.path.join(self.save_dir, f"detected_{img_name}")
            self.detector.save_detection_result(full_img_path, final_detections, output_path)
            
            # Lưu ảnh visualization nếu được yêu cầu
            if save_visualization:
                vis_path = os.path.join(self.save_dir, f"vis_{img_name.split('.')[0]}.png")
                all_scale_detections, _ = self.detector.visualize_multi_scale_results(full_img_path, vis_path)
        
        # Tạo DataFrame từ detections
        if len(final_detections) > 0:
            # Chuẩn bị dữ liệu cho DataFrame
            data = []
            for i, det in enumerate(final_detections):
                # Kiểm tra kích thước của det
                if len(det) != 7:
                    # Cố gắng extract thông tin cần thiết nếu có thể
                    if len(det) >= 5:
                        x1, y1, x2, y2, conf = det[:5]
                        cls = 0 if len(det) <= 5 else det[5]
                        scale_idx = 0 if len(det) <= 6 else det[6]
                    else:
                        continue  # Bỏ qua detection này nếu không đủ thông tin
                else:
                    x1, y1, x2, y2, conf, cls, scale_idx = det
                
                # Tính thêm các thông tin hữu ích
                width = x2 - x1
                height = y2 - y1
                area = width * height
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                aspect_ratio = width / height if height > 0 else 0
                
                # Xác định scale được sử dụng
                if hasattr(self.detector, 'img_sizes') and int(scale_idx) < len(self.detector.img_sizes):
                    scale_used = self.detector.img_sizes[int(scale_idx)]
                else:
                    scale_used = "unknown"
                
                # Thêm vào data
                data.append({
                    'image_path': img_path,
                    'full_image_path': full_img_path,
                    'file_name': os.path.basename(img_path),
                    'face_id': i,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'width': int(width),
                    'height': int(height),
                    'area': int(area),
                    'center_x': int(center_x),
                    'center_y': int(center_y),
                    'aspect_ratio': aspect_ratio,
                    'confidence': conf,
                    'scale_used': scale_used
                })
                
            # Tạo DataFrame
            df = pd.DataFrame(data)
        else:
            # DataFrame rỗng nếu không có detection
            df = pd.DataFrame(columns=[
                'image_path', 'full_image_path', 'file_name', 'face_id', 
                'x1', 'y1', 'x2', 'y2', 'width', 'height', 'area', 
                'center_x', 'center_y', 'aspect_ratio', 'confidence', 'scale_used'
            ])
        
        return df
    
    def process_directory(self, dir_path, image_formats=None, save_csv=True, save_excel=False):
        """
        Xử lý tất cả ảnh trong thư mục và trả về DataFrame tổng hợp
        
        Args:
            dir_path: Đường dẫn thư mục chứa ảnh
            image_formats: List các định dạng ảnh cần xử lý (None = tất cả)
            save_csv: Có lưu kết quả dưới dạng CSV không
            save_excel: Có lưu kết quả dưới dạng Excel không
            
        Returns:
            df_all: DataFrame chứa thông tin tất cả detections
        """
        # Mặc định xử lý các định dạng ảnh phổ biến
        if image_formats is None:
            image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        # Tìm tất cả file ảnh trong thư mục
        image_paths = []
        for ext in image_formats:
            image_paths.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))
        
        if not image_paths:
            print(f"❌ No images found in {dir_path} with formats {image_formats}")
            return pd.DataFrame()
        
        print(f"🔍 Found {len(image_paths)} images in {dir_path}")
        
        # Xử lý từng ảnh song song sử dụng multi-threading
        all_dfs = []
        
        # Hàm xử lý trong worker thread
        def process_single_image(img_path):
            try:
                df = self.process_image(img_path, save_visualization=False)
                return df
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                return pd.DataFrame()
        
        # Sử dụng ThreadPoolExecutor để xử lý đa luồng
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit các task
            future_to_path = {executor.submit(process_single_image, img_path): img_path 
                             for img_path in image_paths}
            
            # Sử dụng tqdm để hiển thị progress bar
            for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="Processing images"):
                img_path = future_to_path[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    print(f"❌ Error retrieving result for {img_path}: {e}")
        
        # Kết hợp tất cả DataFrames
        if all_dfs:
            df_all = pd.concat(all_dfs, ignore_index=True)
            
            # Lưu kết quả dưới dạng CSV nếu được yêu cầu
            if save_csv:
                csv_path = os.path.join(self.save_dir, "face_detections.csv")
                df_all.to_csv(csv_path, index=False)
                print(f"💾 Saved results to CSV: {csv_path}")
            
            # Lưu kết quả dưới dạng Excel nếu được yêu cầu
            if save_excel:
                excel_path = os.path.join(self.save_dir, "face_detections.xlsx")
                df_all.to_excel(excel_path, index=False)
                print(f"💾 Saved results to Excel: {excel_path}")
            
            return df_all
        else:
            print("❌ No valid detections found in any image")
            return pd.DataFrame()
    
    def analyze_results(self, df):
        """
        Phân tích kết quả từ DataFrame
        
        Args:
            df: DataFrame chứa thông tin detections
            
        Returns:
            analysis: Dict chứa các phân tích về kết quả
        """
        if df.empty:
            return {"error": "No detections to analyze"}
        
        analysis = {}
        
        # Tổng số khuôn mặt phát hiện được
        analysis['total_faces'] = len(df)
        
        # Số ảnh đã xử lý
        analysis['total_images'] = df['image_path'].nunique()
        
        # Số ảnh có ít nhất một khuôn mặt
        images_with_faces = df.groupby('image_path').size().reset_index(name='face_count')
        analysis['images_with_faces'] = len(images_with_faces)
        
        # Số ảnh không có khuôn mặt nào
        analysis['images_without_faces'] = analysis['total_images'] - analysis['images_with_faces']
        
        # Trung bình số khuôn mặt trên mỗi ảnh
        analysis['avg_faces_per_image'] = analysis['total_faces'] / analysis['total_images']
        
        # Thống kê về kích thước khuôn mặt
        analysis['face_area'] = {
            'min': df['area'].min(),
            'max': df['area'].max(),
            'mean': df['area'].mean(),
            'median': df['area'].median()
        }
        
        # Thống kê về confidence
        analysis['confidence'] = {
            'min': df['confidence'].min(),
            'max': df['confidence'].max(),
            'mean': df['confidence'].mean(),
            'median': df['confidence'].median()
        }
        
        # Phân bố theo scales
        if 'scale_used' in df.columns:
            scale_counts = df['scale_used'].value_counts().to_dict()
            analysis['scale_distribution'] = scale_counts
        
        return analysis
    
    def generate_report(self, analysis, output_path=None):
        """
        Tạo báo cáo từ kết quả phân tích
        
        Args:
            analysis: Dict chứa các phân tích về kết quả
            output_path: Đường dẫn lưu báo cáo
            
        Returns:
            report_text: Nội dung báo cáo
        """
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
        
        report = []
        report.append("# Face Detection Report")
        report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## Overall Statistics")
        report.append(f"- Total images processed: {analysis['total_images']}")
        report.append(f"- Total faces detected: {analysis['total_faces']}")
        report.append(f"- Images with at least one face: {analysis['images_with_faces']} ({analysis['images_with_faces']/analysis['total_images']*100:.2f}%)")
        report.append(f"- Images without faces: {analysis['images_without_faces']} ({analysis['images_without_faces']/analysis['total_images']*100:.2f}%)")
        report.append(f"- Average faces per image: {analysis['avg_faces_per_image']:.2f}")
        report.append("")
        
        report.append("## Face Size Statistics")
        report.append(f"- Min area: {analysis['face_area']['min']:.2f} pixels²")
        report.append(f"- Max area: {analysis['face_area']['max']:.2f} pixels²")
        report.append(f"- Mean area: {analysis['face_area']['mean']:.2f} pixels²")
        report.append(f"- Median area: {analysis['face_area']['median']:.2f} pixels²")
        report.append("")
        
        report.append("## Confidence Score Statistics")
        report.append(f"- Min confidence: {analysis['confidence']['min']:.2f}")
        report.append(f"- Max confidence: {analysis['confidence']['max']:.2f}")
        report.append(f"- Mean confidence: {analysis['confidence']['mean']:.2f}")
        report.append(f"- Median confidence: {analysis['confidence']['median']:.2f}")
        report.append("")
        
        if 'scale_distribution' in analysis:
            report.append("## Scale Distribution")
            for scale, count in analysis['scale_distribution'].items():
                report.append(f"- Scale {scale}: {count} faces ({count/analysis['total_faces']*100:.2f}%)")
        
        report_text = "\n".join(report)
        
        # Lưu báo cáo nếu được yêu cầu
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"💾 Saved report to: {output_path}")
        
        return report_text


def process_from_csv(csv_path, base_image_path, predictor, output_csv_path=None, output_excel_path=None, save_report=False):
    """
    Xử lý ảnh từ CSV có chứa đường dẫn ảnh
    
    Args:
        csv_path: Đường dẫn đến file CSV
        base_image_path: Đường dẫn cơ sở cho ảnh
        predictor: MultiScaleFaceDataFramePredictor đã được khởi tạo
        output_csv_path: Đường dẫn lưu kết quả CSV
        output_excel_path: Đường dẫn lưu kết quả Excel
        save_report: Có tạo báo cáo không
    
    Returns:
        df_all: DataFrame chứa thông tin tất cả detections
    """
    # Đọc CSV
    print(f"📊 Reading image paths from CSV: {csv_path}")
    df_input = pd.read_csv(csv_path)
    
    # Kiểm tra xem CSV có cột đường dẫn ảnh không
    image_path_column = None
    for col in df_input.columns:
        if 'path' in col.lower() or 'image' in col.lower() or 'file' in col.lower():
            image_path_column = col
            break
    
    if not image_path_column:
        print(f"❌ No image path column found in CSV. Available columns: {df_input.columns.tolist()}")
        return pd.DataFrame()
    
    # Lấy danh sách đường dẫn ảnh
    image_paths = df_input[image_path_column].tolist()
    print(f"🔍 Found {len(image_paths)} image paths in CSV")
    
    # Xử lý từng ảnh song song
    all_dfs = []
    
    # Hàm xử lý trong worker thread
    def process_single_image(img_path):
        try:
            df = predictor.process_image(img_path, save_visualization=False)
            return df
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            return pd.DataFrame()
    
    # Sử dụng ThreadPoolExecutor để xử lý đa luồng
    with ThreadPoolExecutor(max_workers=predictor.num_workers) as executor:
        # Submit các task
        future_to_path = {executor.submit(process_single_image, img_path): img_path 
                        for img_path in image_paths}
        
        # Sử dụng tqdm để hiển thị progress bar
        for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="Processing images from CSV"):
            img_path = future_to_path[future]
            try:
                df = future.result()
                if not df.empty:
                    # Thêm thông tin từ CSV gốc nếu cần
                    img_info = df_input[df_input[image_path_column] == img_path].iloc[0].to_dict()
                    for key, value in img_info.items():
                        if key != image_path_column and key not in df.columns:
                            df[key] = value
                    all_dfs.append(df)
            except Exception as e:
                print(f"❌ Error retrieving result for {img_path}: {e}")
    
    # Kết hợp tất cả DataFrames
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        
        # Lưu kết quả dưới dạng CSV nếu được yêu cầu
        if output_csv_path:
            df_all.to_csv(output_csv_path, index=False)
            print(f"💾 Saved results to CSV: {output_csv_path}")
        
        # Lưu kết quả dưới dạng Excel nếu được yêu cầu
        if output_excel_path:
            df_all.to_excel(output_excel_path, index=False)
            print(f"💾 Saved results to Excel: {output_excel_path}")
        
        # Phân tích kết quả
        if save_report:
            analysis = predictor.analyze_results(df_all)
            report_path = os.path.join(os.path.dirname(output_csv_path or output_excel_path or ''), "detection_report.md")
            predictor.generate_report(analysis, report_path)
        
        return df_all
    else:
        print("❌ No valid detections found in any image from CSV")
        return pd.DataFrame()


def detect_faces(detector, img_path):
    """
    Detect faces trong ảnh sử dụng Multi-scale detection
    
    Args:
        detector: MultiScaleFaceDetector đã khởi tạo
        img_path: Đường dẫn đến ảnh
    
    Returns:
        faces_data: List các face detections [{'bbox': [x1, y1, x2, y2], 'conf': confidence, 'scale_used': scale}]
    """
    try:
        # Đo thời gian
        start_time = time_synchronized()
        
        # Thực hiện phát hiện khuôn mặt
        final_detections, img0_shape = detector.detect_multi_scale(img_path)
        
        # Tính thời gian
        elapsed = time_synchronized() - start_time
        
        # Chuyển đổi định dạng kết quả
        faces_data = []
        
        for det in final_detections:
            # Kiểm tra số lượng giá trị trong det
            if len(det) >= 5:
                if len(det) == 7:
                    x1, y1, x2, y2, conf, cls, scale_idx = det
                    scale_idx = int(scale_idx)
                    scale_used = detector.img_sizes[scale_idx] if scale_idx < len(detector.img_sizes) else "unknown"
                else:
                    x1, y1, x2, y2, conf = det[:5]
                    cls = det[5] if len(det) > 5 else 0
                    scale_used = "unknown"
                
                faces_data.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'conf': float(conf),
                    'cls': int(cls),
                    'scale_used': scale_used
                })
        
        return faces_data, elapsed
        
    except Exception as e:
        print(f"Lỗi detect faces cho {img_path}: {e}")
        return [], 0


def init_worker():
    """Initialize worker process with proper CUDA context"""
    import torch
    import os
    import multiprocessing
    
    # Get process ID and assign GPU FIRST - before any CUDA operations
    current_process = multiprocessing.current_process()
    if hasattr(current_process, '_identity') and current_process._identity:
        process_id = current_process._identity[0] - 1
        gpu_id = process_id % NUM_GPU
        # Set CUDA_VISIBLE_DEVICES IMMEDIATELY to restrict GPU visibility
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"🔧 Worker {current_process.name} (PID: {process_id}) assigned to GPU {gpu_id}")
        print(f"    CUDA_VISIBLE_DEVICES set to: {gpu_id}")
    else:
        # Fallback for main process or unexpected cases
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print(f"🔧 Worker {current_process.name} using default GPU assignment (GPU 0)")
    
    # Now initialize CUDA after setting the environment
    if torch.cuda.is_available():
        torch.cuda.init()
        # Clear any existing CUDA cache
        torch.cuda.empty_cache()
        print(f"    CUDA initialized, available devices: {torch.cuda.device_count()}")
    else:
        print(f"    CUDA not available in this worker")


def process_item_helper(item_data):
    """
    Wrapper function for process_item that uses global SKIP_PROCESSED parameter
    """
    global SKIP_PROCESSED
    return process_item(item_data, SKIP_PROCESSED)


def process_item(item_data, skip_processed=False):
    """
    Xử lý một item: detect faces cho tất cả ảnh và tạo JSON output
    
    Args:
        item_data: (item_id, tiny_face_module)
        skip_processed: Nếu True, bỏ qua item đã được xử lý (có JSON)
    
    Returns:
        result: (item_id, num_frames, total_faces, total_elapsed)
    """
    item_id, tiny_face_module = item_data
    
    # Kiểm tra xem kết quả JSON đã tồn tại chưa
    json_path = os.path.join(JSON_OUTPUT_DIR, f"{item_id}.json")
    max_faces_image_path = os.path.join(MAX_FACES_IMAGES_DIR, f"{item_id}_max_*.jpg")
    max_faces_images = glob.glob(max_faces_image_path)
    
    if skip_processed and os.path.exists(json_path) and max_faces_images:
        # Lấy thông tin cơ bản từ file JSON đã tồn tại
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Tìm shape để xác định số lượng frames và faces
            for tensor in json_data.get("yolo_face_prediction", []):
                if tensor.get("name") == "yolo-face-bboxes":
                    shape = tensor.get("shape", [0, 0, 0])
                    num_frames = shape[0]
                    max_faces = shape[1]
                    
                    # Tìm tổng số faces (loại bỏ các face padding có bbox [-1,-1,-1,-1])
                    total_faces = 0
                    if "data" in tensor:
                        for frame_data in tensor["data"]:
                            for bbox in frame_data:
                                if bbox[0] > -0.99:  # Không phải padding
                                    total_faces += 1
                    
                    # Lấy thời gian xử lý từ JSON nếu có
                    total_elapsed = 0
                    for tensor_time in json_data.get("yolo_face_prediction", []):
                        if tensor_time.get("name") == "yolo-face-total_time":
                            if "data" in tensor_time and tensor_time["data"]:
                                total_elapsed = tensor_time["data"][0]
                    
                    max_faces_count = max([len([b for b in frame if b[0] > -0.99]) for frame in tensor.get("data", [])])
                    print(f"✅ Đã tồn tại: Item {item_id}: {num_frames} frames, {total_faces} total faces, max {max_faces_count} faces/frame -> {json_path}")
                    return (item_id, num_frames, total_faces, total_elapsed)
            
            print(f"⚠️ File JSON {json_path} không hợp lệ hoặc thiếu thông tin, sẽ xử lý lại.")
        except Exception as e:
            print(f"⚠️ Lỗi đọc file JSON {json_path}: {e}, sẽ xử lý lại.")
    
    try:
        print(f"Đang xử lý item {item_id} với path: {tiny_face_module}")
        
        try:
            # Tạo detector với GPU auto-selection
            detector = create_detector(device='')  # Auto-select based on process
        except Exception as e:
            print(f"Lỗi khởi tạo detector cho item {item_id}: {e}")
            # Fallback - thử với specific GPU
            current_process = multiprocessing.current_process()
            if hasattr(current_process, '_identity') and current_process._identity:
                fallback_gpu = str((current_process._identity[0] - 1) % NUM_GPU)
            else:
                fallback_gpu = '0'
            print(f"Thử lại với GPU {fallback_gpu}...")
            detector = create_detector(device=fallback_gpu)
        
        # Lấy danh sách ảnh cần predict
        image_paths = get_image_paths_from_base(tiny_face_module, BASE_IMAGE_PATH)
        
        if not image_paths:
            print(f"Không tìm thấy ảnh cho item_id {item_id} tại {tiny_face_module}")
            return None
        
        all_frames_data = []
        total_start_time = time.time()
        max_faces_count = 0
        max_faces_frame_data = None
        
        for frame_idx, img_path in enumerate(image_paths):
            try:
                # Detect faces
                faces_data, elapsed = detect_faces(detector, img_path)
                
                # Load image để lưu sau này
                image = Image.open(img_path).convert("RGB")
                img_width, img_height = image.size
                
                # Chuẩn bị dữ liệu cho frame này
                bboxes_data = []
                confidence_data = []
                class_names_data = []
                class_indexes_data = []
                class_groups_data = []
                scale_used_data = []
                
                # Điền dữ liệu cho các face thực tế
                for face_data in faces_data:
                    norm_bbox = normalize_bbox(face_data['bbox'], img_width, img_height)
                    if norm_bbox:
                        bboxes_data.append(norm_bbox)
                        confidence_data.append(face_data['conf'])
                        class_names_data.append("face")
                        class_indexes_data.append(0)
                        class_groups_data.append("face")
                        scale_used_data.append(face_data.get('scale_used', 'unknown'))
                
                frame_data = {
                    "frame_idx": frame_idx,
                    "image_path": img_path,
                    "num_faces": len(faces_data),
                    "bboxes": bboxes_data,
                    "confidence": confidence_data,
                    "class_names": class_names_data,
                    "class_indexes": class_indexes_data,
                    "class_groups": class_groups_data,
                    "scale_used": scale_used_data,
                    "infer_time": elapsed,
                    "faces_data": faces_data,  # Lưu faces để vẽ sau này
                    "image": image   # Lưu ảnh để vẽ sau này
                }
                
                all_frames_data.append(frame_data)
                
                # Theo dõi frame có nhiều face nhất
                if len(faces_data) > max_faces_count:
                    max_faces_count = len(faces_data)
                    max_faces_frame_data = frame_data
                
            except Exception as e:
                print(f"Lỗi xử lý frame {frame_idx} của item {item_id}: {e}")
                continue
        
        total_elapsed = time.time() - total_start_time
        
        if not all_frames_data:
            print(f"Không có frame nào được xử lý thành công cho item_id {item_id}")
            return None
        
        # Tạo JSON theo format yêu cầu cho tất cả frames
        num_frames = len(all_frames_data)
        max_faces_per_frame = max([frame["num_faces"] for frame in all_frames_data]) if all_frames_data else 0
        
        # Chuẩn bị dữ liệu theo shape [num_frames, max_faces, 4]
        all_bboxes = []
        all_confidence = []
        all_class_names = []
        all_class_indexes = []
        all_class_groups = []
        all_scales_used = []
        
        for frame_data in all_frames_data:
            # Pad frame data to max_faces_per_frame
            frame_bboxes = frame_data["bboxes"] + [[-1.0, -1.0, -1.0, -1.0]] * (max_faces_per_frame - frame_data["num_faces"])
            frame_confidence = frame_data["confidence"] + [-1.0] * (max_faces_per_frame - frame_data["num_faces"])
            frame_class_names = frame_data["class_names"] + ["unknown"] * (max_faces_per_frame - frame_data["num_faces"])
            frame_class_indexes = frame_data["class_indexes"] + [-1] * (max_faces_per_frame - frame_data["num_faces"])
            frame_class_groups = frame_data["class_groups"] + ["unknown"] * (max_faces_per_frame - frame_data["num_faces"])
            frame_scales = frame_data["scale_used"] + ["unknown"] * (max_faces_per_frame - frame_data["num_faces"])
            
            all_bboxes.append(frame_bboxes)
            all_confidence.append(frame_confidence)
            all_class_names.append(frame_class_names)
            all_class_indexes.append(frame_class_indexes)
            all_class_groups.append(frame_class_groups)
            all_scales_used.append(frame_scales)
        
        json_data = {
            "yolo_face_prediction": [
                {
                    "name": "yolo-face-bboxes",
                    "datatype": "FP32",
                    "shape": [num_frames, max_faces_per_frame, 4],
                    "data": all_bboxes
                },
                {
                    "name": "yolo-face-confidence",
                    "datatype": "FP32",
                    "shape": [num_frames, max_faces_per_frame],
                    "data": all_confidence
                },
                {
                    "name": "yolo-face-class_names",
                    "datatype": "BYTES",
                    "shape": [num_frames, max_faces_per_frame],
                    "data": all_class_names
                },
                {
                    "name": "yolo-face-class_indexes",
                    "datatype": "INT32",
                    "shape": [num_frames, max_faces_per_frame],
                    "data": all_class_indexes
                },
                {
                    "name": "yolo-face-class_groups",
                    "datatype": "BYTES",
                    "shape": [num_frames, max_faces_per_frame],
                    "data": all_class_groups
                },
                {
                    "name": "yolo-face-scale_used",
                    "datatype": "BYTES",
                    "shape": [num_frames, max_faces_per_frame],
                    "data": all_scales_used
                },
                {
                    "name": "yolo-face-ckpt_version",
                    "datatype": "BYTES",
                    "shape": [num_frames],
                    "data": ["yolo_w6_face_multiscale_v1"] * num_frames
                },
                {
                    "name": "yolo-face-infer_time",
                    "datatype": "FP32",
                    "shape": [num_frames],
                    "data": [frame_data["infer_time"] for frame_data in all_frames_data]
                },
                {
                    "name": "yolo-face-total_time",
                    "datatype": "FP32",
                    "shape": [1],
                    "data": [total_elapsed]
                }
            ]
        }
        
        # Lưu JSON
        json_path = os.path.join(JSON_OUTPUT_DIR, f"{item_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Lưu ảnh có nhiều face nhất với bounding box
        if max_faces_frame_data and max_faces_count > 0:
            try:
                # Tạo copy của ảnh để vẽ
                image_with_boxes = max_faces_frame_data["image"].copy()
                image_with_boxes = draw_faces_on_image(image_with_boxes, max_faces_frame_data["faces_data"])
                
                # Lưu ảnh
                max_faces_image_path = os.path.join(MAX_FACES_IMAGES_DIR, f"{item_id}_max_{max_faces_count}_faces.jpg")
                image_with_boxes.save(max_faces_image_path, "JPEG", quality=95)
                print(f"Saved max faces image: {max_faces_image_path}")
            except Exception as e:
                print(f"Lỗi lưu ảnh max faces cho item {item_id}: {e}")
        
        total_faces = sum([frame["num_faces"] for frame in all_frames_data])
        print(f"Item {item_id}: {num_frames} frames, {total_faces} total faces, max {max_faces_count} faces/frame -> {json_path}")
        
        return (item_id, num_frames, total_faces, total_elapsed)
        
    except Exception as e:
        print(f"Lỗi xử lý item_id {item_id}: {e}")
        return None


def create_new_directories(base_json_dir, base_images_dir):
    """
    Tạo thư mục mới với tên có thêm 'new' ở cuối
    
    Args:
        base_json_dir: Thư mục JSON gốc
        base_images_dir: Thư mục ảnh gốc
        
    Returns:
        new_json_dir, new_images_dir: Đường dẫn thư mục mới
    """
    # Tạo tên thư mục mới
    new_json_dir = base_json_dir.rstrip('/') + '_new'
    new_images_dir = base_images_dir.rstrip('/') + '_new'
    
    # Nếu thư mục _new đã tồn tại, thêm số vào cuối
    counter = 1
    original_json = new_json_dir
    original_images = new_images_dir
    
    while os.path.exists(new_json_dir) or os.path.exists(new_images_dir):
        new_json_dir = f"{original_json}_{counter}"
        new_images_dir = f"{original_images}_{counter}"
        counter += 1
    
    # Tạo thư mục
    os.makedirs(new_json_dir, exist_ok=True)
    os.makedirs(new_images_dir, exist_ok=True)
    
    print(f"✅ Đã tạo thư mục mới:")
    print(f"   JSON: {new_json_dir}")
    print(f"   Images: {new_images_dir}")
    
    return new_json_dir, new_images_dir


def check_current_progress(json_dir, images_dir, items_data):
    """
    Kiểm tra tiến độ hiện tại và hỏi người dùng có muốn tiếp tục không
    
    Args:
        json_dir: Thư mục chứa JSON
        images_dir: Thư mục chứa ảnh
        items_data: Danh sách các items cần xử lý
        
    Returns:
        should_continue: True nếu tiếp tục, False nếu bắt đầu lại
        updated_json_dir: Thư mục JSON (có thể là mới)
        updated_images_dir: Thư mục ảnh (có thể là mới)
    """
    if not os.path.exists(json_dir) and not os.path.exists(images_dir):
        print("📂 Thư mục output chưa tồn tại, sẽ tạo mới...")
        return True, json_dir, images_dir
    
    # Đếm số items đã xử lý thành công
    processed_count = 0
    total_items = len(items_data)
    
    for item_id, _ in items_data:
        json_path = os.path.join(json_dir, f"{item_id}.json")
        max_faces_image_path = os.path.join(images_dir, f"{item_id}_max_*.jpg")
        max_faces_images = glob.glob(max_faces_image_path)
        
        if os.path.exists(json_path) and max_faces_images:
            processed_count += 1
    
    if processed_count == 0:
        print("📊 Chưa có items nào được xử lý trong thư mục hiện tại.")
        return True, json_dir, images_dir
    
    # Hiển thị tiến độ hiện tại
    progress_percent = (processed_count / total_items) * 100
    remaining_items = total_items - processed_count
    
    print("\n" + "="*60)
    print("📊 KIỂM TRA TIẾN ĐỘ HIỆN TẠI")
    print("="*60)
    print(f"📁 Thư mục JSON: {json_dir}")
    print(f"📁 Thư mục Images: {images_dir}")
    print(f"✅ Đã xử lý thành công: {processed_count}/{total_items} items ({progress_percent:.1f}%)")
    print(f"⏳ Còn lại cần xử lý: {remaining_items} items")
    
    if processed_count > 0:
        # Hiển thị một số items đã xử lý gần đây
        processed_items = []
        for item_id, _ in items_data:
            json_path = os.path.join(json_dir, f"{item_id}.json")
            max_faces_image_path = os.path.join(images_dir, f"{item_id}_max_*.jpg")
            max_faces_images = glob.glob(max_faces_image_path)
            
            if os.path.exists(json_path) and max_faces_images:
                try:
                    # Lấy thời gian sửa đổi file
                    json_mtime = os.path.getmtime(json_path)
                    processed_items.append((item_id, json_mtime))
                except:
                    processed_items.append((item_id, 0))
        
        # Sắp xếp theo thời gian mới nhất
        processed_items.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📋 5 items được xử lý gần đây nhất:")
        for i, (item_id, mtime) in enumerate(processed_items[:5]):
            if mtime > 0:
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                print(f"   {i+1}. {item_id} (lúc {time_str})")
            else:
                print(f"   {i+1}. {item_id}")
    
    print("="*60)
    
    # Hỏi người dùng
    while True:
        try:
            user_input = input("\n❓ Bạn muốn tiếp tục xử lý từ chỗ đã dừng? (y/n): ").strip().lower()
            
            if user_input in ['y', 'yes']:
                print("✅ Sẽ tiếp tục xử lý các items còn lại...")
                return True, json_dir, images_dir
            
            elif user_input in ['n', 'no']:
                print("🔄 Sẽ bắt đầu lại với thư mục mới...")
                new_json_dir, new_images_dir = create_new_directories(json_dir, images_dir)
                return False, new_json_dir, new_images_dir
            
            else:
                print("❌ Vui lòng nhập 'y' (có) hoặc 'n' (không)")
                
        except KeyboardInterrupt:
            print("\n\n👋 Chương trình bị dừng bởi người dùng.")
            sys.exit(0)
        except EOFError:
            print("\n\n👋 Chương trình bị dừng bởi người dùng.")
            sys.exit(0)


def main():
    """
    Hàm main để chạy chương trình
    """
    # Khai báo global variables
    global MODEL_PATH, JSON_OUTPUT_DIR, MAX_FACES_IMAGES_DIR, BASE_IMAGE_PATH
    global CONF_THRES, IOU_THRES, IMG_SIZES, NUM_GPU, MAX_ITEMS, SKIP_PROCESSED
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='YOLOv7 Face Multi-Scale Detection with DataFrame JSON Output')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='path to model weights file')
    parser.add_argument('--output-dir', type=str, default=JSON_OUTPUT_DIR, help='directory to save JSON results')
    parser.add_argument('--max-faces-dir', type=str, default=MAX_FACES_IMAGES_DIR, help='directory to save max faces images')
    parser.add_argument('--img-sizes', nargs='+', type=int, default=IMG_SIZES, help='list of image sizes for multi-scale detection')
    parser.add_argument('--conf-thres', type=float, default=CONF_THRES, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=IOU_THRES, help='IoU threshold for NMS')
    parser.add_argument('--max-items', type=int, default=MAX_ITEMS, help='maximum number of items to process')
    parser.add_argument('--num-gpus', type=int, default=NUM_GPU, help='number of GPUs to use')
    parser.add_argument('--base-path', type=str, default=BASE_IMAGE_PATH, help='base path for images')
    parser.add_argument('--csv-file', type=str, 
                       default="/home/dainguyenvan/.clearml/cache/storage_manager/datasets/ds_4da1a9d86df546be86fede62610f7b64/val_data.csv",
                       help='CSV file containing item_id and tiny_face_module')
    parser.add_argument('--skip-processed', action='store_true', 
                      help='Bỏ qua các item đã xử lý (có JSON trong thư mục output)')
    parser.add_argument('--force-continue', action='store_true',
                      help='Tự động tiếp tục mà không hỏi người dùng')
    parser.add_argument('--force-restart', action='store_true', 
                      help='Tự động bắt đầu lại với thư mục mới mà không hỏi người dùng')
    
    args = parser.parse_args()
    
    MODEL_PATH = args.model
    JSON_OUTPUT_DIR = args.output_dir
    MAX_FACES_IMAGES_DIR = args.max_faces_dir
    BASE_IMAGE_PATH = args.base_path
    CONF_THRES = args.conf_thres
    IOU_THRES = args.iou_thres
    IMG_SIZES = args.img_sizes
    NUM_GPU = args.num_gpus
    MAX_ITEMS = args.max_items
    SKIP_PROCESSED = args.skip_processed
    
    print("🚀 YOLOv7 Face Multi-Scale Detection với DataFrame JSON Output")
    print(f"⚙️ Config:")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - Output JSON: {JSON_OUTPUT_DIR}")
    print(f"  - Max Faces Images: {MAX_FACES_IMAGES_DIR}")
    print(f"  - Image Sizes: {IMG_SIZES}")
    print(f"  - Confidence Threshold: {CONF_THRES}")
    print(f"  - IoU Threshold: {IOU_THRES}")
    print(f"  - Base Image Path: {BASE_IMAGE_PATH}")
    print(f"  - GPUs Available: {NUM_GPU} (0,1,2)")
    print(f"  - CUDA Visible Devices: Workers will set individual GPU assignment")
    print(f"  - Skip Processed: {SKIP_PROCESSED}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"  - CUDA Available: ✅ ({torch.cuda.device_count()} devices detected)")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print(f"  - CUDA Available: ❌")
        print(f"  - Warning: CUDA không khả dụng, sẽ sử dụng CPU (rất chậm!)")
    print()
    
    # Load DataFrame
    print(f"📋 Loading DataFrame from: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    
    # Check columns
    required_columns = ['item_id', 'tiny_face_module']
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ Column '{col}' not found in CSV. Available columns: {df.columns.tolist()}")
            return
    
    # Lấy items đầu tiên
    df_subset = df.head(MAX_ITEMS)
    print(f"Sẽ xử lý {len(df_subset)} items")
    
    # Chuẩn bị dữ liệu cho multiprocessing
    items_data = [(row['item_id'], row['tiny_face_module']) for _, row in df_subset.iterrows()]
    
    # Kiểm tra tiến độ hiện tại và hỏi người dùng (trừ khi có force flags)
    if args.force_continue:
        print("🔧 Force continue mode - sẽ tiếp tục xử lý...")
        should_continue = True
        final_json_dir = JSON_OUTPUT_DIR
        final_images_dir = MAX_FACES_IMAGES_DIR
    elif args.force_restart:
        print("🔄 Force restart mode - tạo thư mục mới...")
        should_continue = False
        final_json_dir, final_images_dir = create_new_directories(JSON_OUTPUT_DIR, MAX_FACES_IMAGES_DIR)
    else:
        should_continue, final_json_dir, final_images_dir = check_current_progress(
            JSON_OUTPUT_DIR, MAX_FACES_IMAGES_DIR, items_data)
    
    # Cập nhật thông tin thư mục
    JSON_OUTPUT_DIR = final_json_dir
    MAX_FACES_IMAGES_DIR = final_images_dir
    
    # Tạo output directories nếu chưa tồn tại
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MAX_FACES_IMAGES_DIR, exist_ok=True)
    
    # Nếu bắt đầu lại, tắt skip-processed
    if not should_continue:
        SKIP_PROCESSED = False
        print("🔄 Bắt đầu lại - tắt chế độ skip-processed")
    
    # Giới hạn số process dựa trên số GPU
    n_process = min(cpu_count(), NUM_GPU * 2)  # 2 processes per GPU for better utilization
    if n_process < 1:
        n_process = 1
    
    print(f"🔧 Processing Config:")
    print(f"  - Total Processes: {n_process}")
    print(f"  - Processes per GPU: {n_process // NUM_GPU if NUM_GPU > 0 else n_process}")
    print(f"  - GPU Assignment: Round-robin across GPUs 0,1,2 (per process)")
    print(f"  - Multiprocessing Method: spawn (CUDA compatible)")
    print(f"  - GPU Control: Each worker sets its own CUDA_VISIBLE_DEVICES")
    print(f"Bắt đầu predict cho {len(items_data)} items...")
    print()
    
    # Show initial GPU status
    print("📊 Initial GPU Status:")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id, name, mem_used, mem_total = parts[:4]
                    print(f"    GPU {gpu_id}: {name} - {mem_used}MB/{mem_total}MB used")
        else:
            print("    nvidia-smi not available")
    except Exception as e:
        print(f"    Could not get GPU status: {e}")
    print()
    
    # Chạy multiprocessing với spawn method và worker initialization
    with Pool(processes=n_process, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_item_helper, items_data), total=len(items_data), desc="Processing items"))
    
    # Show final GPU status
    print("\n📊 Final GPU Status:")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id, name, mem_used, mem_total = parts[:4]
                    print(f"    GPU {gpu_id}: {name} - {mem_used}MB/{mem_total}MB used")
        else:
            print("    nvidia-smi not available")
    except Exception as e:
        print(f"    Could not get GPU status: {e}")
    print()
    
    # Thống kê kết quả
    successful_items = 0
    total_frames = 0
    total_faces = 0
    total_time = 0
    
    for result in results:
        if result is not None:
            item_id, num_frames, num_faces, elapsed = result
            successful_items += 1
            total_frames += num_frames
            total_faces += num_faces
            total_time += elapsed
    
    print(f"\n=== KẾT QUẢ ===")
    print(f"Items xử lý thành công: {successful_items}/{len(items_data)}")
    print(f"Tỷ lệ thành công: {successful_items/len(items_data)*100:.1f}%")
    print(f"Tổng số frames: {total_frames}")
    print(f"Tổng số faces phát hiện: {total_faces}")
    print(f"Thời gian xử lý trung bình/item: {total_time/successful_items:.3f}s" if successful_items > 0 else "N/A")
    print(f"File JSON lưu tại: {JSON_OUTPUT_DIR}")
    print(f"File ảnh max faces lưu tại: {MAX_FACES_IMAGES_DIR}")
    
    print(f"\n🎉 ĐÃ HOÀN THÀNH: {successful_items} items thành công!")
    if successful_items < len(items_data):
        failed_items = len(items_data) - successful_items
        print(f"⚠️  {failed_items} items thất bại")
    
    print(f"📊 Tổng cộng đã xử lý: {successful_items}/{len(items_data)} items")


if __name__ == "__main__":
    main()
"""
# Xử lý từ mặc định CSV với model mặc định
python yolov7_face_multi_scale_dataframe_predict.py

# Chỉ định model và CSV
python yolov7_face_multi_scale_dataframe_predict.py --model yolov7-w6-face.pt --csv-file your_data.csv

# Tuỳ chỉnh output và thresholds
python yolov7_face_multi_scale_dataframe_predict.py --output-dir ./json_output --max-faces-dir ./max_faces --conf-thres 0.6 --iou-thres 0.3

# Thay đổi scales cho multi-scale detection
python yolov7_face_multi_scale_dataframe_predict.py --img-sizes 640 1280 1920 3840

# Hạn chế số items xử lý'
python yolov7_face_multi_scale_dataframe_predict.py --max-items 100
# bỏ qua item đã xử lý và xử lý tiếp :python yolov7_face_multi_scale_dataframe_predict.py --force-continue
# xử lý từ đầu: python yolov7_face_multi_scale_dataframe_predict.py --force-restart
"""