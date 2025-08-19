# YOLOv7-Face Detection với DataFrame
# Script này predict faces sử dụng YOLOv7-Face model cho dữ liệu từ DataFrame:
# 
# Tính năng:
# 1. Load DataFrame với columns `item_id` và `tiny_face_module` 
# 2. Từ đường dẫn base `tiny_face_module`, tìm tất cả ảnh `_original_xxx.jpg`
# 3. Predict faces cho tất cả frames của mỗi item_id
# 4. Lưu kết quả JSON theo format chuẩn cho từng item_id
# 5. Sử dụng multiprocessing với GPU để tăng tốc
# 6. Giới hạn 10k items đầu tiên
# 
# Cách sử dụng:
# 1. Sửa phần load DataFrame thật trong main section
# 2. Đảm bảo đường dẫn ảnh trong `tiny_face_module` tồn tại
# 3. Chạy script
# 
# Output:
# - Folder `json_results_df/`: Chứa file JSON cho từng item_id
# - Format JSON tương thích với hệ thống hiện tại

import os
import json
import pandas as pd
import glob
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import re
import time
from multiprocessing import Pool, cpu_count

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size, set_logging
from utils.torch_utils import select_device, time_synchronized

# Config
model_path = "./yolov7-face.pt"
json_output_dir = "./all_json_results_df"
max_faces_images_dir = "./yolov7_all_max_faces_images"
base_image_path = "/mnt/md0/projects/auto_review_footage/"  # Đường dẫn gốc đến ảnh
os.makedirs(json_output_dir, exist_ok=True)
os.makedirs(max_faces_images_dir, exist_ok=True)

NUM_GPU = 2  # Số lượng GPU thực tế
MAX_ITEMS = 24000  # Giới hạn items
CONF_THRES = 0.6  # Confidence threshold
IOU_THRES = 0.3  # NMS threshold - Giảm từ 0.45 xuống 0.3 để loại bỏ nhiều box trùng lặp hơn
IMG_SIZE = 640    # Input image size

# Các giá trị IOU threshold khuyến nghị:
# 0.2-0.3: Loại bỏ nhiều box trùng lặp (tốt cho face detection)
# 0.4-0.5: Cân bằng giữa loại bỏ trùng lặp và giữ lại detection
# 0.6-0.7: Ít loại bỏ box (có thể có nhiều box chồng lên nhau)

print(f"🔧 NMS Config: CONF_THRES={CONF_THRES}, IOU_THRES={IOU_THRES}")

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize và pad ảnh để phù hợp với stride"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down
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

def normalize_bbox(bbox, img_width, img_height):
    """Chuyển bbox từ pixel về normalized coordinates (0-1)"""
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

def draw_faces_on_image(image, faces_data):
    """Vẽ bounding box lên ảnh"""
    draw = ImageDraw.Draw(image)
    
    for face_data in faces_data:
        bbox = face_data['bbox']
        conf = face_data['conf']
        
        x1, y1, x2, y2 = bbox
        
        # Vẽ bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Vẽ confidence score
        conf_text = f"{conf:.2f}"
        draw.text((x1, y1-20), conf_text, fill="red")
    
    return image

def get_image_paths_from_base(base_path):
    """
    Từ đường dẫn base như 099/016/118/99016118_original.jpg
    Tìm tất cả ảnh _original_xxx.jpg trong cùng thư mục
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
        return sorted(glob.glob(pattern))
    
    return []

def detect_faces_yolov7(image_path, model, device, img_size=640, conf_thres=None, iou_thres=None):
    """Detect faces bằng YOLOv7-Face model"""
    # Sử dụng config global nếu không truyền vào
    if conf_thres is None:
        conf_thres = CONF_THRES
    if iou_thres is None:
        iou_thres = IOU_THRES
        
    try:
        # Load image
        img0 = cv2.imread(image_path)  # BGR
        if img0 is None:
            return []
        
        # Letterbox
        img = letterbox(img0, img_size, stride=int(model.stride.max()))[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if device.type != 'cpu' else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        
        # Debug: Kiểm tra shape của prediction
        print(f"Prediction shape: {pred.shape}")
        
        # Apply NMS - sửa lỗi tensor size mismatch
        try:
            # Thử với kpt_label=False trước - sử dụng thresholds được truyền vào
            pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, kpt_label=False, nc=1)
        except RuntimeError as e:
            if "Expected size" in str(e) and "dimension" in str(e):
                print(f"NMS error: {e}")
                print(f"Original pred shape: {pred.shape}")
                # Fallback 1: Thử với kpt_label=True
                try:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, kpt_label=True, nc=1)
                except:
                    # Fallback 2: chỉ lấy phần bbox + conf (6 columns đầu)
                    pred_fixed = pred[..., :6]  # x1, y1, x2, y2, conf, cls
                    print(f"Fixed pred shape: {pred_fixed.shape}")
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
                    x1, y1, x2, y2 = [float(x) for x in xyxy]
                    confidence = float(conf)
                    
                    # Normalize confidence nếu cần
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                    confidence = max(0.0, min(1.0, confidence))
                    
                    faces_data.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': confidence
                    })
        
        return faces_data
        
    except Exception as e:
        print(f"Lỗi detect faces cho {image_path}: {e}")
        return []

def detect_item_images(item_data):
    """Detect faces cho tất cả ảnh của 1 item_id"""
    item_id, tiny_face_module = item_data
    
    try:
        # Khởi tạo model cho mỗi process
        device = select_device('')  # Auto select GPU/CPU
        model = attempt_load(model_path, map_location=device)
        
        if device.type != 'cpu':
            model.half()
        model.eval()
        
        # Lấy danh sách ảnh cần predict
        image_paths = get_image_paths_from_base(tiny_face_module)
        
        if not image_paths:
            print(f"Không tìm thấy ảnh cho item_id {item_id} tại {tiny_face_module}")
            return None
        
        all_frames_data = []
        total_start_time = time.time()
        max_faces_count = 0
        max_faces_frame_data = None
        
        for frame_idx, img_path in enumerate(image_paths):
            try:
                start_time = time.time()
                
                # Detect faces
                faces_data = detect_faces_yolov7(img_path, model, device, IMG_SIZE)
                
                elapsed = time.time() - start_time
                
                # Load image để lưu sau này
                image = Image.open(img_path).convert("RGB")
                img_width, img_height = image.size
                
                # Chuẩn bị dữ liệu cho frame này
                bboxes_data = []
                confidence_data = []
                class_names_data = []
                class_indexes_data = []
                class_groups_data = []
                
                # Điền dữ liệu cho các face thực tế
                for face_data in faces_data:
                    norm_bbox = normalize_bbox(face_data['bbox'], img_width, img_height)
                    if norm_bbox:
                        bboxes_data.append(norm_bbox)
                        confidence_data.append(face_data['conf'])
                        class_names_data.append("face")
                        class_indexes_data.append(0)
                        class_groups_data.append("face")
                
                frame_data = {
                    "frame_idx": frame_idx,
                    "image_path": img_path,
                    "num_faces": len(faces_data),
                    "bboxes": bboxes_data,
                    "confidence": confidence_data,
                    "class_names": class_names_data,
                    "class_indexes": class_indexes_data,
                    "class_groups": class_groups_data,
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
        
        for frame_data in all_frames_data:
            # Pad frame data to max_faces_per_frame
            frame_bboxes = frame_data["bboxes"] + [[-1.0, -1.0, -1.0, -1.0]] * (max_faces_per_frame - frame_data["num_faces"])
            frame_confidence = frame_data["confidence"] + [-1.0] * (max_faces_per_frame - frame_data["num_faces"])
            frame_class_names = frame_data["class_names"] + ["unknown"] * (max_faces_per_frame - frame_data["num_faces"])
            frame_class_indexes = frame_data["class_indexes"] + [-1] * (max_faces_per_frame - frame_data["num_faces"])
            frame_class_groups = frame_data["class_groups"] + ["unknown"] * (max_faces_per_frame - frame_data["num_faces"])
            
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
        json_path = os.path.join(json_output_dir, f"{item_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Lưu ảnh có nhiều face nhất với bounding box
        if max_faces_frame_data and max_faces_count > 0:
            try:
                # Tạo copy của ảnh để vẽ
                image_with_boxes = max_faces_frame_data["image"].copy()
                image_with_boxes = draw_faces_on_image(image_with_boxes, max_faces_frame_data["faces_data"])
                
                # Lưu ảnh
                max_faces_image_path = os.path.join(max_faces_images_dir, f"{item_id}_max_{max_faces_count}_faces.jpg")
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

# Main execution
if __name__ == "__main__":
    # TODO: Load DataFrame với item_id và tiny_face_module columns
    # Ví dụ:
    # df = pd.read_csv("your_dataframe.csv")
    # df = pd.read_parquet("your_dataframe.parquet")
    
    # Để test, tạo DataFrame mẫu
    # Bạn cần thay thế phần này bằng cách load DataFrame thực tế
    print("WARNING: Đang sử dụng DataFrame mẫu. Hãy thay thế bằng DataFrame thực tế của bạn!")
    df = pd.read_csv(
        "/home/dainguyenvan/.clearml/cache/storage_manager/datasets/ds_3d1c894fb7d54154a1d7bfc2d005bebc/val_data.csv"
    )
    
    # Lấy items đầu tiên
    df_subset = df.head(MAX_ITEMS)
    print(f"Sẽ xử lý {len(df_subset)} items")
    
    # Chuẩn bị dữ liệu cho multiprocessing
    items_data = [(row['item_id'], row['tiny_face_module']) for _, row in df_subset.iterrows()]
    
    # Giới hạn số process
    n_process = min(cpu_count(), NUM_GPU)
    if n_process < 1:
        n_process = 1
    
    print(f"Sử dụng {n_process} processes")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {CONF_THRES}")
    print(f"IoU threshold: {IOU_THRES}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Bắt đầu predict cho {len(items_data)} items...")
    
    # Chạy multiprocessing
    with Pool(processes=n_process) as pool:
        results = pool.map(detect_item_images, items_data)
    
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
    print(f"File JSON lưu tại: {json_output_dir}")
    print(f"File ảnh max faces lưu tại: {max_faces_images_dir}")
    
    print(f"\n🎉 ĐÃ HOÀN THÀNH: {successful_items} items thành công!")
    if successful_items < len(items_data):
        failed_items = len(items_data) - successful_items
        print(f"⚠️  {failed_items} items thất bại")
    
    print(f"📊 Tổng cộng đã xử lý: {successful_items}/{len(items_data)} items")
