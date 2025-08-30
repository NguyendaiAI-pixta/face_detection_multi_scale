"""
Shared utilities and preprocessing functions for YOLOv7 Face Detection
Contains common functions that can be reused across multiple prediction scripts.
"""

import os
import json
import glob
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import time
from typing import List, Tuple, Dict, Any, Optional, Union

# Import từ YOLOv7 Face
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Import for parent directory access
import sys
from pathlib import Path


def normalize_bbox(bbox, img_width, img_height):
    """
    Chuyển bbox từ pixel về normalized coordinates (0-1)
    
    Args:
        bbox: tuple/list chứa (x1, y1, x2, y2) 
        img_width: chiều rộng ảnh
        img_height: chiều cao ảnh
        
    Returns:
        List normalized coordinates [x1, y1, x2, y2] trong khoảng [0, 1]
    """
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
    else:
        return None
    
    # Normalize về 0-1
    norm_x1 = x1 / img_width
    norm_y1 = y1 / img_height
    norm_x2 = x2 / img_width
    norm_y2 = y2 / img_height
    
    return [norm_x1, norm_y1, norm_x2, norm_y2]


def denormalize_bbox(bbox, img_width, img_height):
    """
    Chuyển bbox từ normalized coordinates (0-1) về pixel coordinates
    
    Args:
        bbox: tuple/list chứa normalized (x1, y1, x2, y2) 
        img_width: chiều rộng ảnh
        img_height: chiều cao ảnh
        
    Returns:
        List pixel coordinates [x1, y1, x2, y2]
    """
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        norm_x1, norm_y1, norm_x2, norm_y2 = bbox
    else:
        return None
    
    # Denormalize về pixel
    x1 = norm_x1 * img_width
    y1 = norm_y1 * img_height
    x2 = norm_x2 * img_width
    y2 = norm_y2 * img_height
    
    return [x1, y1, x2, y2]


def draw_faces_on_image(image, faces_data, bbox_color="red", text_color="red", line_width=3, 
                       use_api_preprocess=True, original_shape=None, img_size=None):
    """
    Vẽ bounding box lên ảnh với hỗ trợ API preprocessing
    
    Args:
        image: PIL Image object
        faces_data: List các dict chứa thông tin face {'bbox': [x1,y1,x2,y2], 'conf': float}
        bbox_color: màu của bounding box
        text_color: màu của text confidence
        line_width: độ dày của đường viền
        use_api_preprocess: có sử dụng API preprocessing không
        original_shape: (H, W) shape của ảnh gốc trước khi preprocessing
        img_size: kích thước model input nếu dùng API preprocessing
        
    Returns:
        PIL Image với bounding boxes được vẽ
    """
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    for face_data in faces_data:
        bbox = face_data.get('bbox', [])
        conf = face_data.get('conf', 0.0)
        
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            
            # Điều chỉnh tọa độ nếu sử dụng API preprocessing
            if use_api_preprocess and original_shape is not None and img_size is not None:
                x1, y1, x2, y2 = adjust_bbox_from_api_preprocess(
                    [x1, y1, x2, y2], original_shape, img_size, image.size
                )
            
            # Vẽ bounding box
            draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=line_width)
            
            # Vẽ confidence score
            conf_text = f"{conf:.2f}"
            draw.text((x1, max(0, y1-20)), conf_text, fill=text_color)
    
    return image_copy


def scale_coords_api_approach(img_input_shape, coords, img0_shape):
    """
    Scale coordinates cho API preprocessing approach (pad to square + letterbox)
    
    Args:
        img_input_shape: (H, W) - Shape của model input (e.g. 640x640)
        coords: Tensor coordinates cần scale
        img0_shape: (H, W, C) - Shape của ảnh gốc
        
    Returns:
        Scaled coordinates về ảnh gốc
    """
    img_h, img_w = img_input_shape
    orig_h, orig_w = img0_shape[:2]
    
    # Debug info
    print(f"DEBUG scale_coords: input_shape={img_input_shape}, orig_shape={img0_shape[:2]}")
    print(f"DEBUG coords before scale: {coords[:2] if len(coords) > 0 else 'empty'}")
    
    # Với API approach: original → pad to square → letterbox to model input
    # Step 1: Tính square size (max của original dimensions)
    square_size = max(orig_h, orig_w)
    
    # Step 2: Tính scale factor từ model input về square size
    scale_factor = square_size / img_h  # Đảo ngược: từ small input về large square
    
    # Step 3: Scale coordinates từ model input về square size
    coords[:, [0, 2]] *= scale_factor  # x coordinates
    coords[:, [1, 3]] *= scale_factor  # y coordinates
    
    # Step 4: Clip về original image bounds (vì pad top-left nên chỉ cần clip)
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, orig_w)  # clip x
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, orig_h)  # clip y
    
    print(f"DEBUG coords after scale: {coords[:2] if len(coords) > 0 else 'empty'}")
    return coords


def adjust_bbox_from_api_preprocess(bbox, original_shape, img_size, current_size):
    """
    Wrapper function để giữ tương thích với API cũ
    Sử dụng scale_coords_api_approach bên trong
    
    Args:
        bbox: [x1, y1, x2, y2] - tọa độ bbox từ model
        original_shape: (H, W) - shape của ảnh gốc
        img_size: kích thước input model (640)
        current_size: (W, H) - kích thước ảnh hiện tại để vẽ
        
    Returns:
        [x1, y1, x2, y2] - tọa độ đã điều chỉnh
    """
    import torch
    
    # Convert bbox to tensor format for scale_coords_api_approach
    coords_tensor = torch.tensor([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=torch.float32)
    
    # Tạo img0_shape format (H, W, C) từ original_shape
    img0_shape = (original_shape[0], original_shape[1], 3)
    
    # Scale coordinates từ model input về ảnh gốc
    scaled_coords = scale_coords_api_approach((img_size, img_size), coords_tensor, img0_shape)
    
    # Extract scaled coordinates
    x1, y1, x2, y2 = scaled_coords[0].tolist()
    
    # Scale về kích thước ảnh hiện tại để vẽ
    orig_h, orig_w = original_shape
    curr_w, curr_h = current_size
    
    scale_w = curr_w / orig_w
    scale_h = curr_h / orig_h
    
    x1 = x1 * scale_w
    y1 = y1 * scale_h
    x2 = x2 * scale_w
    y2 = y2 * scale_h
    
    return [int(x1), int(y1), int(x2), int(y2)]


def get_image_paths_from_base(base_path, base_image_path="/mnt/md0/projects/auto_review_footage/"):
    """
    Từ đường dẫn base như 099/016/118/99016118_original.jpg
    Tìm tất cả ảnh _original_xxx.jpg trong cùng thư mục
    
    Args:
        base_path: đường dẫn relative từ base_image_path
        base_image_path: đường dẫn gốc đến thư mục chứa ảnh
        
    Returns:
        List các đường dẫn ảnh tìm được
    """
    # Nối với đường dẫn gốc
    full_base_path = os.path.join(base_image_path, base_path)
    
    if not os.path.exists(full_base_path):
        return []
    
    # Lấy thư mục và tên file gốc
    dir_path = os.path.dirname(full_base_path)
    base_name = os.path.basename(full_base_path)
    
    # Tách phần trước _original.jpg
    if '_original.jpg' in base_name:
        prefix = base_name.replace('_original.jpg', '')
        # Tìm tất cả file _original_xxx.jpg
        pattern = os.path.join(dir_path, f"{prefix}_original_*.jpg")
        image_paths = sorted(glob.glob(pattern))
        
        # Nếu không tìm thấy, thêm file gốc vào
        if not image_paths:
            image_paths = [full_base_path] if os.path.exists(full_base_path) else []
        
        return image_paths
    
    return [full_base_path] if os.path.exists(full_base_path) else []


def find_images_in_directory(dir_path, image_formats=None, recursive=False):
    """
    Tìm tất cả file ảnh trong thư mục
    
    Args:
        dir_path: đường dẫn thư mục
        image_formats: list các extension ảnh cần tìm (None = mặc định)
        recursive: có tìm kiếm recursive trong sub-folders không
        
    Returns:
        List đường dẫn các file ảnh
    """
    if image_formats is None:
        image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
    
    image_paths = []
    
    if recursive:
        # Tìm kiếm recursive
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in image_formats):
                    image_paths.append(os.path.join(root, file))
    else:
        # Tìm kiếm trong thư mục hiện tại
        for ext in image_formats:
            image_paths.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))
    
    return sorted(image_paths)


def pad_to_square_top_left(img):
    """
    API Framework approach - pad to square (top-left alignment)
    
    Args:
        img: numpy array or PIL Image
        
    Returns:
        numpy array - ảnh đã pad thành hình vuông
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    h, w, c = img.shape
    new_size = max(h, w)
    padded_img = np.zeros((new_size, new_size, c), dtype=img.dtype)
    padded_img[:h, :w, :] = img
    return padded_img


def letterbox_api(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    Custom letterbox implementation for API approach
    Resize and pad image to fit new_shape while maintaining aspect ratio
    
    Args:
        img: input image (numpy array)
        new_shape: target shape (height, width)
        color: padding color (R, G, B)
        auto: minimum rectangle
        scaleFill: stretch
        scaleup: allow scale up
        stride: stride for padding alignment
        
    Returns:
        tuple: (processed_image, ratio, (dw, dh))
    """
    # Current shape [height, width]
    shape = img.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)


def preprocess_api_approach(img_path, img_size, stride):
    """
    API Framework preprocessing approach - EXACT MATCH với arvutils.api
    Step 1: Pad image to square (top-left) 
    Step 2: Letterbox to model input size (sử dụng YOLOv7 letterbox)
    Step 3: Convert to CHW format and make contiguous
    
    Args:
        img_path: đường dẫn ảnh
        img_size: kích thước target 
        stride: stride của model
        
    Returns:
        numpy array - ảnh đã được preprocessing với format CHW và contiguous
    """
    from utils.datasets import letterbox  # Import YOLOv7 letterbox function
    
    # Load image as PIL then convert to numpy (giống API Framework)
    img_pil = Image.open(img_path).convert('RGB')
    img0 = np.array(img_pil)  # RGB format
    
    # Step 1: Pad to square (top-left) - EXACT match với arvutils.api.pad_to_square_top_left
    squared_img = pad_to_square_top_left(img0)
    
    # Step 2: Letterbox using YOLOv7 function (EXACT match với API Framework)
    img = letterbox(squared_img, img_size, stride=stride, auto=False)[0]
    
    # Step 3: Convert to CHW format and make contiguous 
    # EXACT match với API Framework transforms:
    # img = img.transpose(2, 0, 1)  # HWC to CHW (NO BGR to RGB conversion!)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img)
    
    return img


def load_yolo_model(model_path, device=''):
    """
    Load YOLOv7 model với device được chỉ định
    
    Args:
        model_path: đường dẫn đến file weights
        device: device để load model ('' = auto, 'cpu', 'cuda:0', etc.)
        
    Returns:
        tuple: (model, device)
    """
    device = select_device(device)
    model = attempt_load(model_path, map_location=device)
    
    if device.type != 'cpu':
        model.half()
    model.eval()
    
    return model, device


def detect_faces_yolov7_basic(image_path, model, device, img_size=640, conf_thres=0.5, iou_thres=0.45):
    """
    Detect faces bằng YOLOv7-Face model (basic version)
    
    Args:
        image_path: đường dẫn ảnh
        model: YOLOv7 model đã load
        device: torch device
        img_size: kích thước input
        conf_thres: confidence threshold
        iou_thres: IoU threshold cho NMS
        
    Returns:
        List các face detection với format [x1, y1, x2, y2, conf, cls]
    """
    try:
        # Preprocessing - đã trả về CHW format và contiguous
        img = preprocess_api_approach(image_path, img_size, int(model.stride.max()))
        
        # Load original image for scaling coordinates later
        img0 = cv2.imread(image_path)  # BGR
        if img0 is None:
            return []
        
        # Convert to tensor (img đã là CHW format từ preprocess_api_approach)
        img = torch.from_numpy(img).to(device)
        img = img.half() if device.type != 'cpu' else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        
        # Apply NMS
        try:
            # Thử với kpt_label=False trước
            pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, kpt_label=False, nc=1)
        except RuntimeError as e:
            if "Expected size" in str(e) and "dimension" in str(e):
                print(f"NMS error: {e}")
                # Fallback: Thử với kpt_label=True
                try:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, kpt_label=True, nc=1)
                except:
                    # Fallback 2: chỉ lấy phần bbox + conf (6 columns đầu)
                    pred_fixed = pred[..., :6]  # x1, y1, x2, y2, conf, cls
                    pred = non_max_suppression(pred_fixed, conf_thres, iou_thres, agnostic=False, kpt_label=False, nc=1)
            else:
                raise e
        
        faces_data = []
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                # Extract face data
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(x) for x in xyxy]
                    faces_data.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': float(conf),
                        'cls': int(cls) if cls is not None else 0
                    })
        
        return faces_data
        
    except Exception as e:
        print(f"Lỗi detect faces cho {image_path}: {e}")
        return []


def create_yolo_json_format(all_frames_data, item_id=None):
    """
    Tạo JSON theo format chuẩn cho YOLOv7 Face detection
    
    Args:
        all_frames_data: List các dict chứa thông tin frame
        item_id: ID của item (optional)
        
    Returns:
        Dict JSON data theo format yêu cầu
    """
    if not all_frames_data:
        return None
    
    num_frames = len(all_frames_data)
    max_faces_per_frame = max([frame.get("num_faces", 0) for frame in all_frames_data])
    
    # Chuẩn bị dữ liệu theo shape [num_frames, max_faces, 4]
    all_bboxes = []
    all_confidence = []
    all_class_names = []
    all_class_indexes = []
    all_class_groups = []
    
    for frame_data in all_frames_data:
        # Pad frame data to max_faces_per_frame
        num_faces = frame_data.get("num_faces", 0)
        
        frame_bboxes = frame_data.get("bboxes", []) + [[-1.0, -1.0, -1.0, -1.0]] * (max_faces_per_frame - num_faces)
        frame_confidence = frame_data.get("confidence", []) + [-1.0] * (max_faces_per_frame - num_faces)
        frame_class_names = frame_data.get("class_names", []) + ["unknown"] * (max_faces_per_frame - num_faces)
        frame_class_indexes = frame_data.get("class_indexes", []) + [-1] * (max_faces_per_frame - num_faces)
        frame_class_groups = frame_data.get("class_groups", []) + ["unknown"] * (max_faces_per_frame - num_faces)
        
        all_bboxes.append(frame_bboxes)
        all_confidence.append(frame_confidence)
        all_class_names.append(frame_class_names)
        all_class_indexes.append(frame_class_indexes)
        all_class_groups.append(frame_class_groups)
    
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
                "name": "yolo-face-ckpt_version",
                "datatype": "BYTES",
                "shape": [num_frames],
                "data": ["yolo_w6_face_v1"] * num_frames
            },
            {
                "name": "yolo-face-infer_time",
                "datatype": "FP32",
                "shape": [num_frames],
                "data": [frame_data.get("infer_time", 0.0) for frame_data in all_frames_data]
            },
            {
                "name": "yolo-face-total_time",
                "datatype": "FP32",
                "shape": [1],
                "data": [sum(frame_data.get("infer_time", 0.0) for frame_data in all_frames_data)]
            }
        ]
    }
    
    return json_data


def save_json_results(json_data, output_path, item_id=None):
    """
    Lưu kết quả JSON
    
    Args:
        json_data: Dict chứa dữ liệu JSON
        output_path: đường dẫn lưu file hoặc thư mục
        item_id: ID của item (dùng để tạo tên file nếu output_path là thư mục)
    """
    if json_data is None:
        return
    
    # Xác định đường dẫn file cuối cùng
    if os.path.isdir(output_path) or output_path.endswith('/'):
        # output_path là thư mục
        os.makedirs(output_path, exist_ok=True)
        filename = f"{item_id}.json" if item_id else "results.json"
        final_path = os.path.join(output_path, filename)
    else:
        # output_path là file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_path = output_path
    
    # Lưu file JSON
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved JSON results to: {final_path}")


def calculate_face_statistics(faces_data_list):
    """
    Tính toán thống kê về face detection results
    
    Args:
        faces_data_list: List các face detection results
        
    Returns:
        Dict chứa các thống kê
    """
    if not faces_data_list:
        return {"total_faces": 0, "avg_confidence": 0.0}
    
    total_faces = len(faces_data_list)
    confidences = [face.get('conf', 0.0) for face in faces_data_list]
    
    # Tính toán bbox areas
    areas = []
    for face in faces_data_list:
        bbox = face.get('bbox', [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
    
    stats = {
        "total_faces": total_faces,
        "avg_confidence": np.mean(confidences) if confidences else 0.0,
        "min_confidence": min(confidences) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
        "avg_area": np.mean(areas) if areas else 0.0,
        "min_area": min(areas) if areas else 0.0,
        "max_area": max(areas) if areas else 0.0
    }
    
    return stats


def print_processing_summary(stats_list, total_time=None):
    """
    In ra tóm tắt quá trình xử lý
    
    Args:
        stats_list: List các dict thống kê
        total_time: tổng thời gian xử lý (optional)
    """
    if not stats_list:
        print("❌ No processing results to summarize")
        return
    
    total_items = len(stats_list)
    total_faces = sum(stats.get("total_faces", 0) for stats in stats_list)
    avg_confidence = np.mean([stats.get("avg_confidence", 0.0) for stats in stats_list])
    
    print(f"\n=== XỬ LÝ HOÀN THÀNH ===")
    print(f"📊 Tổng số items xử lý: {total_items}")
    print(f"👥 Tổng số faces phát hiện: {total_faces}")
    print(f"📈 Trung bình faces/item: {total_faces/total_items:.1f}")
    print(f"🎯 Confidence trung bình: {avg_confidence:.3f}")
    
    if total_time:
        print(f"⏱️  Tổng thời gian: {total_time:.2f}s")
        print(f"⚡ Thời gian/item: {total_time/total_items:.3f}s")
