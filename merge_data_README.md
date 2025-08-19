# Multi-dataset Face Data Merger for YOLOv7-Face

## Mục đích
Script này giúp bạn tự động gộp nhiều bộ dữ liệu nhận diện khuôn mặt (WiderFace, custom, Roboflow, FDDB...) thành một bộ dữ liệu chuẩn YOLOv7-Face để train, đồng thời chia bộ val chung đại diện và giữ nguyên các bộ val riêng từng nguồn.

## Cách hoạt động
1. **Gộp dữ liệu train:**
   - Gộp tất cả ảnh và label từ các thư mục train của WiderFace và các custom dataset vào một tập train chung.
   - Tất cả ảnh/label train sẽ nằm trong `train/images` và `train/labels`.

2. **Gộp dữ liệu val:**
   - Gộp tất cả ảnh và label từ các thư mục val của WiderFace và các custom dataset vào một tập val chung.
   - Tất cả ảnh/label val sẽ nằm trong `val/images` và `val/labels`.

3. **Thống kê dữ liệu:**
   - Script thống kê số lượng ảnh và số face (bounding box) của từng phần (train/val) cho từng dataset trước và sau khi gộp.
   - Thống kê này được lưu ra file JSON (`merge_stats.json`) để tiện phân tích, vẽ biểu đồ sau này.

## Cấu trúc thư mục đầu vào
- Mỗi bộ dữ liệu custom cần có cấu trúc:
  ```
  custom_dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
  ```
- Tên ảnh và tên file label phải giống nhau (img1.jpg <-> img1.txt).

## Đầu ra sau khi merge
- `enhanced_face_dataset/`
  ├── train/images/
  ├── train/labels/
  ├── val/images/
  ├── val/labels/
  └── ...
- File cấu hình YAML (ví dụ: `data/enhanced_face.yaml`) để train YOLOv7-Face.
- File thống kê dữ liệu: `merge_stats.json`

## Hướng dẫn sử dụng
1. Chỉnh sửa file `merge_datasets.py`:
   - Thêm các dòng sau vào hàm `main()` để gộp các bộ dữ liệu mong muốn:
     ```python
     merger.add_widerface("WIDER_train", "WIDER_val", "wider")
     merger.add_custom_dataset("/path/to/custom1/train", prefix="custom1_train")
     merger.add_custom_dataset("/path/to/custom2/train", prefix="custom2_train")
     # ...
     ```
2. Chạy script:
   ```bash
   python merge_datasets.py
   ```
3. Kiểm tra thư mục `enhanced_face_dataset` và file `merge_stats.json` để xem thống kê dữ liệu.
4. Sử dụng file cấu hình YAML để train YOLOv7-Face với dữ liệu đã gộp.

## Lưu ý
- Tất cả dữ liệu train/val đều được gộp chung, không còn phân chia val riêng từng nguồn.
- Thống kê chi tiết từng nguồn vẫn được lưu trong file JSON để tiện phân tích.

---
**Mọi thắc mắc hoặc cần mở rộng chức năng, hãy liên hệ hoặc chỉnh sửa script theo nhu cầu!**
