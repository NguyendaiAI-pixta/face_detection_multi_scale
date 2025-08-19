# WiderFace Blur Dataset Generator

## Mục đích
Tạo bộ dữ liệu khuôn mặt bị mờ (blurred faces) từ dataset WiderFace để huấn luyện các mô hình nhận diện khuôn mặt robust trong điều kiện ảnh bị xoá phông, chuyển động, hoặc chất lượng kém.

## Ý tưởng chính
- **Augmentation thực tế**: Mô phỏng các trường hợp camera xoá phông, chuyển động, hoặc điều kiện ánh sáng kém bằng các thuật toán blur (Gaussian, Motion, Radial).
- **Phân phối độ khó**: Lấy mẫu ảnh từ WiderFace theo tỷ lệ:
  - 30% từ các case dễ (easy)
  - 50% từ các case trung bình (medium)
  - 20% từ các case khó (hard)
- **Lọc chất lượng**: Chỉ chọn các khuôn mặt có kích thước >= 32x32 pixels để đảm bảo dữ liệu có giá trị huấn luyện.
- **Phân bổ mức độ mờ**: Mỗi ảnh gốc sẽ được tạo ra 3 phiên bản blur với mức độ nhẹ, vừa, nặng (light, medium, heavy).
- **Giữ nguyên label**: Bounding box gốc được copy sang ảnh blur để đảm bảo annotation chính xác.

## Các thuật toán blur sử dụng
- **Gaussian Blur**: Mô phỏng xoá phông nhẹ, vừa, nặng.
- **Motion Blur**: Mô phỏng chuyển động ngang của camera.
- **Radial Blur**: Mô phỏng hiệu ứng zoom hoặc rung từ tâm ảnh.

## Quy trình tạo dữ liệu
1. **Phân loại ảnh theo độ khó** dựa trên category WiderFace (easy, medium, hard).
2. **Lọc ảnh có khuôn mặt đủ lớn** (>= 32x32).
3. **Tạo các phiên bản blur** cho mỗi ảnh gốc:
   - Mỗi ảnh sẽ có 3 phiên bản: light, medium, heavy (mỗi loại blur chọn ngẫu nhiên).
4. **Copy label bounding box** sang ảnh blur.
5. **Lưu metadata** về quá trình tạo dữ liệu, phân phối, cấu hình blur, v.v.

## Cấu trúc thư mục output
```
blur_dataset/
├── images/   # Ảnh blur đã tạo
├── labels/   # YOLO labels tương ứng
└── dataset_metadata.json  # Thông tin chi tiết về dataset
```

## Hướng dẫn sử dụng
- **Kết hợp với WiderFace gốc**: Có thể merge blur images vào WIDER_train/images và labels để huấn luyện chung.
- **Huấn luyện riêng**: Tạo file yaml riêng cho blur dataset để train hoặc fine-tune.
- **Tăng robustness**: Sử dụng bộ dữ liệu này để tăng khả năng nhận diện khuôn mặt trong điều kiện thực tế, camera chất lượng thấp, hoặc môi trường phức tạp.

## Lợi ích
- Tăng khả năng generalization cho model.
- Giúp model nhận diện tốt hơn trong các tình huống mờ, xoá phông, chuyển động.
- Phù hợp cho các ứng dụng thực tế như camera giám sát, điện thoại, v.v.

## Cách chạy
```bash
python blur_dataset_generator.py
```
- Có thể chỉnh sửa số lượng ảnh, đường dẫn, hoặc cấu hình blur trong file code.

🎉 Blur Dataset Generation Completed!
📊 Final Statistics:
   • Original images processed: 592
   • Total blur variants created: 1776
   • Average variants per image: 3.0
   • Output directory: /mnt/md0/projects/nguyendai-footage/blur_dataset
   • Metadata saved: /mnt/md0/projects/nguyendai-footage/blur_dataset/dataset_metadata.json

📖 Usage Instructions:
==================================================
1. 📁 Dataset Structure:
   /mnt/md0/projects/nguyendai-footage/blur_dataset/
   ├── images/           # Blur images
   ├── labels/           # YOLO format labels
   └── dataset_metadata.json

2. 🔗 Integration với Original Dataset:
   • Merge vào WIDER_train bằng symlink hoặc copy
   • Update data/widerface.yaml để include blur data
   • Hoặc tạo riêng config cho combined dataset

3. 🚀 Training Commands:
   # Option 1: Train trên blur data only
   python train.py --data blur_dataset.yaml
   
   # Option 2: Combine với original data
   python train.py --data combined_widerface.yaml

4. 📊 Expected Benefits:
   • Improved robustness trong adverse conditions
   • Better generalization cho camera blur
   • Enhanced performance trong real-world scenarios

✅ SUCCESS! Blur dataset created successfully
Ready for training với robust face detection!
