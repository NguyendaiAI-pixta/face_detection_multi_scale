import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from torchvision.ops import nms
from typing import List, Tuple, Dict, Any, Optional
# Import từ YOLOv7 Face
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device, time_synchronized 

# Import shared utilities
from utils.preprocess_yolo_predict import (
    load_yolo_model, calculate_face_statistics, 
    print_processing_summary, create_yolo_json_format,
    save_json_results, preprocess_api_approach,
    pad_to_square_top_left, letterbox_api,
    draw_faces_on_image, normalize_bbox, scale_coords_api_approach
)


class MultiScaleFaceDetector:
    """
    Phát hiện khuôn mặt với multi-scale Test Time Augmentation (TTA)
    """

    def __init__(self, model_path, device='', img_sizes=[640, 3840], conf_thres=0.5, iou_thres=0.5, 
                 use_api_preprocess=False):
        """
        Khởi tạo Multi-scale Face Detector
        
        Args:
            model_path: Đường dẫn đến model weights
            device: Device để chạy model (cuda hoặc cpu)
            img_sizes: List các kích thước ảnh để test
            conf_thres: Confidence threshold cho detection
            iou_thres: IoU threshold cho NMS
            use_api_preprocess: Có sử dụng API preprocessing approach không (pad to square + letterbox)
        """
        # Load model using shared utility
        self.model, self.device = load_yolo_model(model_path, device)
        self.half = self.device.type != 'cpu'  # half precision only on CUDA
        
        # Model config
        self.stride = int(self.model.stride.max())
        self.img_sizes = [check_img_size(size, s=self.stride) for size in img_sizes]
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        # Detection thresholds
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Preprocessing config
        self.use_api_preprocess = use_api_preprocess
        
        print(f"✅ Initialized Multi-Scale Face Detector")
        print(f"📏 Scales: {self.img_sizes}")
        print(f"📱 Device: {self.device}")
        print(f"🔍 Confidence threshold: {self.conf_thres}")
        print(f"🔗 IoU threshold: {self.iou_thres}")
        print(f"🛠️ API Preprocessing: {'Enabled' if use_api_preprocess else 'Disabled'}")
        
    def preprocess_image(self, img_path, img_size):
        """
        Tiền xử lý ảnh để phù hợp với input của model
        Hỗ trợ cả standard letterbox và API preprocessing approach
        
        Args:
            img_path: Đường dẫn ảnh
            img_size: Target size để resize và letterbox
            
        Returns:
            img_tensor: Tensor đầu vào cho model
            img0: Ảnh gốc dạng BGR
            img0_shape: Shape của ảnh gốc
        """
        # Load image
        img0 = cv2.imread(img_path)
        if img0 is None:
            raise ValueError(f"Could not read image: {img_path}")
        img0_shape = img0.shape
        
        if self.use_api_preprocess:
            # Sử dụng API preprocessing approach từ shared utility
            img = preprocess_api_approach(img_path, img_size, self.stride)
        else:
            # Sử dụng standard letterbox preprocessing
            img = letterbox(img0, img_size, stride=self.stride)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.ascontiguousarray(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        return img, img0, img0_shape
        
    @torch.no_grad()
    def detect_single_scale(self, img_path, img_size):
        """
        Thực hiện phát hiện khuôn mặt trên một scale cụ thể
        
        Args:
            img_path: Đường dẫn ảnh
            img_size: Kích thước target
            
        Returns:
            detections: List các detection [x1, y1, x2, y2, conf, cls, scale_idx]
            img0_shape: Shape của ảnh gốc
        """
        # Preprocess
        img, img0, img0_shape = self.preprocess_image(img_path, img_size)
        
        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]
        
        # NMS
        try:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
        except:
            # Fallback cho một số model khác structure
            pred = non_max_suppression(pred[..., :6], self.conf_thres, self.iou_thres, classes=None, agnostic=False)
        
        t2 = time_synchronized()
        
        # Process detections
        detections = []
        
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes từ img_size về img0_shape
                det[:, :4] =  scale_coords_api_approach(img.shape[2:], det[:, :4], img0_shape).round()
                
                # Convert to numpy
                det_numpy = det.cpu().numpy()
                
                # Check detection shape - some YOLOv7 models output additional info
                # We only need [x1, y1, x2, y2, conf, cls]
                if det_numpy.shape[1] > 6:
                    # Lấy chỉ 6 giá trị đầu tiên
                    det_numpy = det_numpy[:, :6]
                
                # Thêm scale_idx cho mỗi detection
                scale_idx = np.ones((det_numpy.shape[0], 1)) * self.img_sizes.index(img_size)
                det_numpy = np.hstack((det_numpy, scale_idx))
                
                detections.append(det_numpy)
        
        if detections:
            detections = np.vstack(detections)
        else:
            detections = np.zeros((0, 7))  # [x1, y1, x2, y2, conf, cls, scale_idx]
        
        return detections, img0_shape, t2 - t1
    
    def calculate_scale_weights(self, detections):
        """
        Tính weight cho mỗi detection dựa trên kích thước khuôn mặt và scale tương ứng
        
        Args:
            detections: Numpy array với format [x1, y1, x2, y2, conf, cls, scale_idx]
            
        Returns:
            scale_weights: Numpy array các weights
        """
        if len(detections) == 0:
            return np.array([])
        
        # Tính face size (diện tích bbox)
        face_sizes = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        scale_indices = detections[:, 6].astype(int)
        
        # Khởi tạo weights
        weights = np.ones_like(face_sizes)
        
        # Small faces (< 32x32) -> ưu tiên scale lớn (img_sizes[-1], img_sizes[-2])
        small_mask = face_sizes < 1024  # 32x32
        weights[small_mask & (scale_indices >= len(self.img_sizes) - 2)] *= 1.2
        
        # Medium faces (32x32 - 128x128) -> ưu tiên scale trung bình
        medium_mask = (face_sizes >= 1024) & (face_sizes <= 16384)
        middle_scales = len(self.img_sizes) // 2
        weights[medium_mask & (scale_indices == middle_scales)] *= 1.1
        
        # Large faces (> 128x128) -> ưu tiên scale nhỏ (img_sizes[0], img_sizes[1]) 
        large_mask = face_sizes > 16384  # 128x128
        weights[large_mask & (scale_indices <= 1)] *= 1.2
        
        return weights
    
    def weighted_nms(self, detections, iou_thresh=None):
        """
        Áp dụng weighted NMS cho các detections từ nhiều scale
        
        Args:
            detections: Numpy array với format [x1, y1, x2, y2, conf, cls, scale_idx]
            iou_thresh: IoU threshold cho NMS, nếu None thì dùng self.iou_thres
            
        Returns:
            keep_detections: Detections sau khi áp dụng weighted NMS
        """
        if len(detections) == 0:
            return detections
        
        if iou_thresh is None:
            iou_thresh = self.iou_thres
        
        # Tính scale weights
        scale_weights = self.calculate_scale_weights(detections)
        
        # Apply weights to confidence scores
        detections_copy = detections.copy()
        detections_copy[:, 4] = detections[:, 4] * scale_weights
        
        # Convert to PyTorch tensors
        boxes = torch.from_numpy(detections_copy[:, :4]).float()
        scores = torch.from_numpy(detections_copy[:, 4]).float()
        
        # Áp dụng NMS
        # sử dụng torchvision.ops.nms thay vì tự implement
     
        keep_indices = nms(boxes, scores, iou_thresh)
        
        # Convert indices về numpy và filter detections
        keep_indices = keep_indices.cpu().numpy()
        keep_detections = detections[keep_indices]
        
        return keep_detections
    
    def detect_multi_scale(self, img_path):
        """
        Thực hiện phát hiện khuôn mặt với nhiều scales và merge kết quả
        
        Args:
            img_path: Đường dẫn ảnh
            
        Returns:
            final_detections: Kết quả cuối cùng sau khi merge
            img0_shape: Shape của ảnh gốc
        """
        all_detections = []
        img0_shape = None
        
        print(f"🖼️ Processing image: {img_path}")
        total_time = 0
        
        # Detect trên từng scale
        for i, img_size in enumerate(self.img_sizes):
            print(f"  📏 Scale {i+1}/{len(self.img_sizes)}: {img_size}x{img_size}")
            detections, img0_shape, infer_time = self.detect_single_scale(img_path, img_size)
            total_time += infer_time
            
            if len(detections) > 0:
                all_detections.append(detections)
                print(f"    ✅ Found {len(detections)} faces in {infer_time*1000:.1f}ms")
            else:
                print(f"    ❌ No faces detected in {infer_time*1000:.1f}ms")
        
        # Merge tất cả detections
        if all_detections:
            merged_detections = np.vstack(all_detections)
            
            # Áp dụng weighted NMS
            final_detections = self.weighted_nms(merged_detections)
            
            print(f"🎯 Final result: {len(final_detections)} faces after merging {sum([len(d) for d in all_detections])} detections")
            print(f"⏱️ Total inference time: {total_time*1000:.1f}ms")
            
            # In thống kê chi tiết sử dụng shared utility
            stats = self.get_detection_statistics(final_detections)
            print(f"📊 Detection Statistics: {stats['total_faces']} faces, avg conf: {stats['avg_confidence']:.3f}")
        else:
            final_detections = np.array([])
            print("❌ No faces detected in any scale")
        
        return final_detections, img0_shape
    
    def visualize_multi_scale_results(self, img_path, save_path=None):
        """
        Trực quan hóa kết quả phát hiện khuôn mặt với nhiều scale
        
        Args:
            img_path: Đường dẫn ảnh
            save_path: Đường dẫn lưu kết quả
            
        Returns:
            all_scale_detections: Dict các detections ở từng scale
            final_detections: Kết quả cuối cùng sau khi merge
        """
        # Detect trên từng scale riêng lẻ
        all_scale_detections = {}
        
        print("🔍 Running single scale detections...")
        for img_size in self.img_sizes:
            detections, img0_shape, _ = self.detect_single_scale(img_path, img_size)
            all_scale_detections[img_size] = detections
        
        # Detect với multi-scale
        print("🔍 Running multi-scale detection...")
        final_detections, img0_shape = self.detect_multi_scale(img_path)
        
        # Load ảnh gốc
        img0 = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        
        # Tạo visualization
        n_scales = len(self.img_sizes)
        fig_height = 4 * (n_scales // 2 + 2)  # +2 cho final result và original
        
        fig, axs = plt.subplots(n_scales // 2 + 2, 2, figsize=(16, fig_height))
        axs = axs.flatten()
        
        # Original image
        axs[0].imshow(img_rgb)
        axs[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axs[0].axis('off')
        
        # Scale detections
        colors = plt.cm.rainbow(np.linspace(0, 1, n_scales))
        
        for i, img_size in enumerate(self.img_sizes):
            detections = all_scale_detections[img_size]
            axs[i+1].imshow(img_rgb)
            
            # Debug thông tin về shape của detections
            if len(detections) > 0:
                print(f"DEBUG: detections shape for scale {img_size}: {detections.shape}")
                if detections.shape[1] != 7:
                    print(f"WARNING: Expected 7 values per detection but got {detections.shape[1]}")
                
            for det in detections:
                # Kiểm tra kích thước của mỗi detection
                if len(det) != 7:
                    print(f"WARNING: Incorrect detection length: {len(det)}, values: {det}")
                    # Cố gắng extract thông tin cần thiết nếu có thể
                    if len(det) >= 5:
                        x1, y1, x2, y2, conf = det[:5]
                        cls = 0
                        scale_idx = 0
                    else:
                        continue
                else:
                    x1, y1, x2, y2, conf, cls, scale_idx = det
                    
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      linewidth=2, edgecolor=colors[i], facecolor='none')
                axs[i+1].add_patch(rect)
                axs[i+1].text(x1, y1-10, f"{conf:.2f}", 
                           color=colors[i], fontsize=9, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
            
            axs[i+1].set_title(f'Scale {img_size}x{img_size}: {len(detections)} faces', 
                            fontsize=12, fontweight='bold', color=colors[i])
            axs[i+1].axis('off')
        
        # Final merged result
        final_idx = n_scales + 1
        axs[final_idx].imshow(img_rgb)
        
        # Debug thông tin về shape của final_detections
        if len(final_detections) > 0:
            print(f"DEBUG: final_detections shape: {final_detections.shape}")
            if final_detections.shape[1] != 7:
                print(f"WARNING: Expected 7 values per final detection but got {final_detections.shape[1]}")
        
        for det in final_detections:
            # Kiểm tra kích thước của mỗi detection
            if len(det) != 7:
                print(f"WARNING: Incorrect final detection length: {len(det)}, values: {det}")
                # Cố gắng extract thông tin cần thiết nếu có thể
                if len(det) >= 5:
                    x1, y1, x2, y2, conf = det[:5]
                    cls = 0
                    scale_idx = 0
                else:
                    continue
            else:
                x1, y1, x2, y2, conf, cls, scale_idx = det
                scale_idx = int(scale_idx)
                
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=3, edgecolor='red', facecolor='none')
            axs[final_idx].add_patch(rect)
            
            # Show which scale this detection came from
            scale_size = self.img_sizes[scale_idx]
            axs[final_idx].text(x1, y2+15, f"Scale: {scale_size}", 
                             color='blue', fontsize=8, fontweight='bold')
            
            axs[final_idx].text(x1, y1-10, f"{conf:.2f}", 
                             color='red', fontsize=9, fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        axs[final_idx].set_title(f'Final Merged Result: {len(final_detections)} faces', 
                              fontsize=14, fontweight='bold', color='red')
        axs[final_idx].axis('off')
        
        # Fill remaining axes
        for i in range(n_scales + 2, len(axs)):
            axs[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved visualization to: {save_path}")
        else:
            plt.show()
        
        return all_scale_detections, final_detections
    
    def save_detection_result(self, img_path, detections, output_path=None):
        """
        Lưu ảnh với bounding boxes, tự động điều chỉnh cho API preprocessing
        
        Args:
            img_path: Đường dẫn ảnh gốc
            detections: Các detections [x1, y1, x2, y2, conf, cls, scale_idx]
            output_path: Đường dẫn lưu ảnh kết quả
        """
        if output_path is None:
            base_name = os.path.basename(img_path)
            output_path = f"detection_result_{base_name}"
        
        # Load ảnh gốc
        img_cv = cv2.imread(img_path)
        img_pil = Image.open(img_path).convert('RGB')
        
        # Convert detections thành faces_data format
        faces_data = []
        for det in detections:
            if len(det) >= 5:
                x1, y1, x2, y2, conf = det[:5]
                faces_data.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'conf': float(conf),
                    'cls': int(det[5]) if len(det) > 5 else 0
                })
        
        # Sử dụng enhanced draw_faces_on_image với API preprocessing support
        if self.use_api_preprocess and faces_data:
            # Lấy shape ảnh gốc và img_size lớn nhất (thường là 3840)
            original_shape = img_cv.shape[:2]  # (H, W)
            max_img_size = max(self.img_sizes) if self.img_sizes else 3840
            
            result_img = draw_faces_on_image(
                img_pil, 
                faces_data,
                use_api_preprocess=True,
                original_shape=original_shape,
                img_size=max_img_size
            )
        else:
            # Standard drawing không cần điều chỉnh
            result_img = draw_faces_on_image(img_pil, faces_data)
        
        # Lưu ảnh
        result_img.save(output_path, "JPEG", quality=95)
        print(f"💾 Saved detection result to: {output_path}")
        
        # Fallback: Cũng lưu bằng OpenCV method cho tương thích
        img_cv_result = cv2.imread(img_path)
        for det in detections:
            if len(det) >= 5:
                x1, y1, x2, y2, conf = det[:5]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Vẽ rectangle
                cv2.rectangle(img_cv_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Vẽ confidence score
                conf_text = f"{conf:.2f}"
                cv2.putText(img_cv_result, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Lưu phiên bản OpenCV
        cv_output_path = output_path.replace('.jpg', '_cv.jpg').replace('.jpeg', '_cv.jpeg')
        cv2.imwrite(cv_output_path, img_cv_result)
        print(f"💾 Saved OpenCV version to: {cv_output_path}")
    
    def export_detection_to_json(self, img_path, detections, output_path=None, item_id=None):
        """
        Export detection results sang JSON format sử dụng shared utility
        
        Args:
            img_path: đường dẫn ảnh
            detections: numpy array các detections
            output_path: đường dẫn lưu JSON
            item_id: ID của item (optional)
        """
        if len(detections) == 0:
            print("⚠️ No detections to export")
            return
        
        # Load ảnh để lấy kích thước
        img0 = cv2.imread(img_path)
        if img0 is None:
            print(f"❌ Could not load image: {img_path}")
            return
        
        img_height, img_width = img0.shape[:2]
        
        # Convert detections thành format cho JSON
        faces_data = []
        for i, det in enumerate(detections):
            if len(det) >= 5:
                x1, y1, x2, y2, conf = det[:5]
                
                # Normalize bbox coordinates
                norm_bbox = normalize_bbox([x1, y1, x2, y2], img_width, img_height)
                
                face_data = {
                    'bbox': norm_bbox,
                    'conf': float(conf),
                    'cls': int(det[5]) if len(det) > 5 else 0,
                    'scale_idx': int(det[6]) if len(det) > 6 else 0
                }
                faces_data.append(face_data)
    
    def get_detection_statistics(self, detections):
        """
        Tính thống kê về face detections sử dụng shared utility
        
        Args:
            detections: Numpy array với format [x1, y1, x2, y2, conf, cls, scale_idx]
            
        Returns:
            Dict chứa các thống kê
        """
        if len(detections) == 0:
            return {"total_faces": 0, "avg_confidence": 0.0}
        
        # Convert detections về format mà shared utility expect
        faces_data = []
        for det in detections:
            if len(det) >= 5:
                x1, y1, x2, y2, conf = det[:5]
                faces_data.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': float(conf),
                    'cls': int(det[5]) if len(det) > 5 else 0
                })
        
        # Sử dụng shared utility để tính statistics
        return calculate_face_statistics(faces_data)
    
    def print_detection_summary(self, detections_list, total_time=None):
        """
        In tóm tắt kết quả detection sử dụng shared utility
        
        Args:
            detections_list: List các detections arrays
            total_time: tổng thời gian xử lý (optional)
        """
        stats_list = []
        for detections in detections_list:
            stats = self.get_detection_statistics(detections)
            stats_list.append(stats)
        
        # Sử dụng shared utility để in summary
        print_processing_summary(stats_list, total_time)
    
    def export_to_json(self, img_path, detections, output_path=None, item_id=None):
        """
        Export detection results sang JSON format sử dụng shared utility
        
        Args:
            img_path: đường dẫn ảnh
            detections: numpy array các detections
            output_path: đường dẫn lưu JSON
            item_id: ID của item
        """
        if len(detections) == 0:
            print("⚠️ No detections to export")
            return None
        
        # Convert detections thành faces_data format
        faces_data = []
        for det in detections:
            if len(det) >= 5:
                x1, y1, x2, y2, conf = det[:5]
                faces_data.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': float(conf),
                    'cls': int(det[5]) if len(det) > 5 else 0
                })
        
        # Tạo frame data
        frame_data = {
            "frame_idx": 0,
            "num_faces": len(faces_data),
            "bboxes": [face['bbox'] for face in faces_data],
            "confidence": [face['conf'] for face in faces_data],
            "class_names": ["face"] * len(faces_data),
            "class_indexes": [face['cls'] for face in faces_data],
            "class_groups": ["person"] * len(faces_data),
            "infer_time": 0.0
        }
        
        # Tạo và lưu JSON
        json_data = create_yolo_json_format([frame_data], item_id=item_id)
        if output_path and json_data:
            save_json_results(json_data, output_path, item_id=item_id)
        
        return json_data
    
    def compare_preprocessing_methods(self, img_path, save_comparison=True):
        """
        So sánh kết quả giữa standard letterbox và API preprocessing approach
        
        Args:
            img_path: đường dẫn ảnh test
            save_comparison: có lưu kết quả so sánh không
            
        Returns:
            Dict chứa kết quả so sánh
        """
        print(f"🔍 Comparing preprocessing methods for: {img_path}")
        
        results = {}
        
        # Test với standard preprocessing
        print("📊 Testing Standard Letterbox Preprocessing...")
        self.use_api_preprocess = False
        standard_detections, img0_shape = self.detect_multi_scale(img_path)
        standard_stats = self.get_detection_statistics(standard_detections)
        results['standard'] = {
            'detections': standard_detections,
            'stats': standard_stats,
            'method': 'Standard Letterbox'
        }
        
        # Test với API preprocessing
        print("📊 Testing API Preprocessing Approach...")
        self.use_api_preprocess = True
        api_detections, _ = self.detect_multi_scale(img_path)
        api_stats = self.get_detection_statistics(api_detections)
        results['api'] = {
            'detections': api_detections,
            'stats': api_stats,
            'method': 'API Approach (Pad to Square + Letterbox)'
        }
        
        # So sánh kết quả
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"{'Method':<35} {'Faces':<8} {'Avg Conf':<10} {'Min Conf':<10} {'Max Conf':<10}")
        print("-" * 75)
        
        for method_key, result in results.items():
            stats = result['stats']
            print(f"{result['method']:<35} {stats['total_faces']:<8} {stats['avg_confidence']:<10.3f} "
                  f"{stats['min_confidence']:<10.3f} {stats['max_confidence']:<10.3f}")
        
        # Tính độ chênh lệch
        face_diff = api_stats['total_faces'] - standard_stats['total_faces']
        conf_diff = api_stats['avg_confidence'] - standard_stats['avg_confidence']
        
        print(f"\n🔢 DIFFERENCES:")
        print(f"  Face count difference: {face_diff:+d}")
        print(f"  Confidence difference: {conf_diff:+.3f}")
        
        # Lưu kết quả nếu được yêu cầu
        if save_comparison:
            comparison_file = "preprocessing_comparison.json"
            import json
            with open(comparison_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                for method_key in results:
                    if len(results[method_key]['detections']) > 0:
                        results[method_key]['detections'] = results[method_key]['detections'].tolist()
                    else:
                        results[method_key]['detections'] = []
                
                json.dump(results, f, indent=2)
            print(f"💾 Saved comparison results to: {comparison_file}")
        
        return results


def main():
    """
    Hàm main để test detector với shared utilities và API preprocessing
    """
    # Config
    model_path = "yolov7-w6-face.pt"  # Model weights
    img_path = "91871038_original.jpg"  # Test image
    save_path = "multi_scale_detection_api_preprocess_test.png"  # Visualization output
    device = ''  # Empty for auto-detection (CUDA if available)
    img_sizes = [640, 3840]  # Scales to test
    conf_thres = 0.5  # Confidence threshold
    iou_thres = 0.5  # IoU threshold for NMS
    use_api_preprocess = True  # 🛠️ Config: Sử dụng API preprocessing approach
    
    print("🚀 YOLOv7 Face Multi-Scale Detection with Shared Utilities")
    print(f"🛠️ API Preprocessing: {'Enabled' if use_api_preprocess else 'Disabled'}")
    
    # Tạo detector với API preprocessing config
    detector = MultiScaleFaceDetector(
        model_path=model_path,
        device=device,
        img_sizes=img_sizes,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        use_api_preprocess=use_api_preprocess  # 🛠️ Tham số config preprocessing
    )
    
    # Detect và visualize
    all_scale_detections, final_detections = detector.visualize_multi_scale_results(img_path, save_path)
    
    # Lưu kết quả cuối cùng
    detector.save_detection_result(img_path, final_detections, "final_detection_multi_person.jpg")
    
    # Export sang JSON format sử dụng shared utilities
    json_output = detector.export_to_json(img_path, final_detections, 
                                        output_path="./json_results/", 
                                        item_id="multi_scale_test")
    
    # In thống kê chi tiết sử dụng shared utilities
    stats = detector.get_detection_statistics(final_detections)
    print(f"\n📊 Final Statistics:")
    print(f"  - Total faces: {stats['total_faces']}")
    print(f"  - Average confidence: {stats['avg_confidence']:.3f}")
    print(f"  - Min confidence: {stats['min_confidence']:.3f}")
    print(f"  - Max confidence: {stats['max_confidence']:.3f}")
    
    print("✅ Multi-scale detection with shared utilities completed!")
    
    # 🔄 Demo preprocessing comparison
    print("\n" + "="*60)
    print("🔍 PREPROCESSING COMPARISON DEMO")
    print("="*60)
    
    # So sánh hai phương pháp preprocessing
    comparison_results = detector.compare_preprocessing_methods(img_path)
    
    # Hiển thị kết luận
    standard_count = comparison_results['standard_letterbox']['total_faces']
    api_count = comparison_results['api_approach']['total_faces']
    
    print("\n🎯 COMPARISON SUMMARY:")
    if api_count > standard_count:
        print(f"✨ API Preprocessing approach detected {api_count - standard_count} more faces!")
        print("   → Recommended: Use API preprocessing for better results")
    elif standard_count > api_count:
        print(f"📋 Standard letterbox detected {standard_count - api_count} more faces!")
        print("   → Recommended: Use standard preprocessing for this image")
    else:
        print("⚖️  Both methods detected the same number of faces")
        print("   → Either method can be used")
    
    print(f"\n💾 Comparison results saved to: {comparison_results.get('saved_to', 'comparison_results.json')}")


if __name__ == "__main__":
    main()
