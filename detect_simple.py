import argparse
import os
import glob
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size, set_logging, increment_path, colorstr
from utils.torch_utils import select_device, time_synchronized
from utils.plots import plot_one_box


def detect():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7-w6-face.pt', help='model.pt path')
    parser.add_argument('--source', type=str, required=True, help='source image or folder')
    parser.add_argument('--img-size', type=int, default=1000, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='runs/detect', help='save results to this directory')
    parser.add_argument('--line-thickness', type=int, default=3, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    
    opt = parser.parse_args()
    print(opt)
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size
    
    if half:
        model.half()  # to FP16
    
    # Configure
    model.eval()
    
    # Get image files
    if os.path.isdir(opt.source):
        img_files = sorted(glob.glob(os.path.join(opt.source, '*.*')))
        img_files = [f for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.dng'))]
    else:
        img_files = [opt.source]
    
    # Create save directory
    save_dir = increment_path(Path(opt.save_dir), exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    
    # Colors for different classes
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(80)]
    
    print(f'Detecting faces in {len(img_files)} images...')
    
    # Inference
    for i, img_path in enumerate(img_files):
        print(f'Processing {i+1}/{len(img_files)}: {img_path}')
        
        # Load image
        img0 = cv2.imread(img_path)  # BGR
        if img0 is None:
            print(f'Error: Could not load image {img_path}')
            continue
            
        # Padded resize
        img = letterbox(img0, imgsz, stride=stride)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=False)
        t2 = time_synchronized()
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = img0.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                n_faces = len(det)
                print(f'Found {n_faces} face{"s" if n_faces != 1 else ""}')
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    print(f"Debug - Raw confidence: {conf}")  # Debug line
                    
                    # Normalize confidence về 0-1 nếu nó > 1
                    if conf > 1.0:
                        conf = conf / 100.0  # Nếu confidence ở dạng 0-100, chia cho 100
                    
                    # Clamp confidence trong khoảng 0-1
                    conf = max(0.0, min(1.0, float(conf)))
                    
                    if not opt.hide_labels or not opt.hide_conf:
                        # Hiển thị confidence dưới dạng phần trăm
                        label = f'{conf*100:.1f}%' if not opt.hide_conf else 'Face'
                    else:
                        label = None
                    
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=opt.line_thickness)
            else:
                print('No faces detected')
            
            # Save results
            save_path = str(save_dir / Path(img_path).name)
            cv2.imwrite(save_path, im0)
            print(f'Saved to {save_path}')
        
        print(f'Inference time: {t2 - t1:.3f}s')
    
    print(f'\nResults saved to {save_dir}')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
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


if __name__ == '__main__':
    detect()
