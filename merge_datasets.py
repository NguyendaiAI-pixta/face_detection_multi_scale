#!/usr/bin/env python3
"""
Multi-dataset merger for YOLOv7-Face training
Automatically converts and merges multiple face detection datasets
Đã tối ưu hóa cho 30 core với xử lý đa luồng
"""
import os
import shutil
from pathlib import Path
import json
import yaml
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

'''
- tạo thư mục chứa dữ liệu gộp
- copy ảnh và label từ các nguồn vào thư mục mới (tiền tố prefix) để phân biệt thư mục gốc
- tạo file cấu hình mới chỉ định đường dẫn train/val, số class, tên class
'''
class DatasetMerger:
    def _copy_val_file(self, args):
        """Copy một cặp file val (image + label) - dùng cho multiprocessing, bỏ qua file đã tồn tại"""
        img_file, val_lbl_dir, val_common_img, val_common_lbl = args
        faces_count = 0
        # Copy image nếu chưa tồn tại
        dst_img = val_common_img / img_file.name
        if dst_img.exists():
            return 0, 0
        shutil.copy2(img_file, dst_img)
        # Copy và đếm label nếu chưa tồn tại
        lbl_file = val_lbl_dir / (img_file.stem + ".txt")
        dst_lbl = val_common_lbl / lbl_file.name
        if lbl_file.exists():
            if not dst_lbl.exists():
                shutil.copy2(lbl_file, dst_lbl)
            try:
                with open(lbl_file, "r") as f:
                    faces_count = sum(1 for _ in f)
            except:
                pass
        return 1, faces_count

    def merge_val_sets(self):
        """Thống kê số lượng ảnh và mặt trong val chung (không cần prefix)"""
        print("\n🔄 Thống kê lại val chung không cần prefix...")
        val_img_dir = self.output_dir / "val" / "images"
        val_lbl_dir = self.output_dir / "val" / "labels"
        # Đếm số lượng ảnh
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            jpg_future = executor.submit(list, val_img_dir.glob("*.jpg"))
            png_future = executor.submit(list, val_img_dir.glob("*.png"))
            img_files = jpg_future.result() + png_future.result()
        total_val_imgs = len(img_files)
        # Đếm số lượng mặt
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            txt_future = executor.submit(list, val_lbl_dir.glob("*.txt"))
            txt_files = txt_future.result()
        total_val_faces = count_faces_parallel(txt_files, self.num_workers)
        print(f"✅ Val merged: {total_val_imgs} ảnh, {total_val_faces} faces")
        self.dataset_stats['val_merged'] = {'val_images': total_val_imgs, 'val_faces': total_val_faces}
    def merge_train_sets(self):
        """Gộp tất cả các bộ train thành 1 bộ train chung (đã mặc định) sử dụng đa luồng"""
        print("\n🔄 Thống kê bộ train đã gộp...")
        
        # Thu thập tất cả files song song
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            jpg_future = executor.submit(list, self.train_dir.glob("*.jpg"))
            png_future = executor.submit(list, self.train_dir.glob("*.png"))
            txt_future = executor.submit(list, self.train_labels_dir.glob("*.txt"))
            
            jpg_files = jpg_future.result()
            png_files = png_future.result()
            txt_files = txt_future.result()
            
        train_imgs = len(jpg_files) + len(png_files)
        
        # Đếm faces song song
        print(f"📊 Đếm faces trong {len(txt_files)} files train...")
        train_faces = count_faces_parallel(txt_files, self.num_workers)
        
        self.dataset_stats['train_merged'] = {'train_images': train_imgs, 'train_faces': train_faces}
    def save_stats_json(self, json_path="merge_stats.json"):
        """Lưu thống kê dữ liệu vào file JSON"""
        with open(json_path, "w") as f:
            json.dump(self.dataset_stats, f, indent=2)
        print(f"Thống kê dữ liệu đã lưu vào {json_path}")
    def __init__(self, output_dir="/mnt/data1/arv/data-clean/merged_dataset", num_workers=None):
        self.output_dir = Path(output_dir)
        self.train_dir = self.output_dir / "train" / "images"
        self.train_labels_dir = self.output_dir / "train" / "labels"
        self.val_dir = self.output_dir / "val" / "images"
        self.val_labels_dir = self.output_dir / "val" / "labels"
        # Thống kê từng dataset
        self.dataset_stats = {}
        # Số lượng worker cho multiprocessing, mặc định là số core hiện có
        self.num_workers = num_workers or min(30, multiprocessing.cpu_count())
        print(f"🚀 Sử dụng {self.num_workers} workers cho xử lý song song")
        # Create directories
        for dir_path in [self.train_dir, self.train_labels_dir, self.val_dir, self.val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def add_widerface(self, wider_train_path, wider_val_path, prefix="wider", train_sampling_ratio=1.0, val_sampling_ratio=1.0):
        """Add WiderFace dataset with sampling support"""
        print(f"Adding WiderFace dataset with prefix: {prefix}")
        if train_sampling_ratio < 1.0:
            print(f"🎯 Will sample {train_sampling_ratio*100:.1f}% from training set")
        if val_sampling_ratio < 1.0:
            print(f"🎯 Will sample {val_sampling_ratio*100:.1f}% from validation set")
            
        # Thống kê số ảnh/số face
        def count_faces_labels(labels_dir):
            total_faces = 0
            total_labels = 0
            for label_file in Path(labels_dir).rglob("*.txt"):
                total_labels += 1
                with open(label_file, "r") as f:
                    total_faces += sum(1 for _ in f)
            return total_labels, total_faces
        
        # Copy training images and labels with sampling
        self._copy_sampled_dataset(
            wider_train_path + "/images", 
            wider_train_path + "/labels",
            self.train_dir, 
            self.train_labels_dir, 
            prefix,
            train_sampling_ratio
        )
        
        # Copy validation images and labels with sampling
        self._copy_sampled_dataset(
            wider_val_path + "/images",
            wider_val_path + "/labels", 
            self.val_dir,
            self.val_labels_dir,
            prefix + "_val",
            val_sampling_ratio
        )
        
        # Thống kê sau khi copy
        train_labels, train_faces = count_faces_labels(self.train_labels_dir)
        val_labels, val_faces = count_faces_labels(self.val_labels_dir)
        self.dataset_stats[prefix] = {
            'train_labels': train_labels,
            'train_faces': train_faces,
            'val_labels': val_labels,
            'val_faces': val_faces,
            'train_sampling_ratio': train_sampling_ratio,
            'val_sampling_ratio': val_sampling_ratio
        }
        print(f"  {prefix} - train labels: {train_labels}, train faces: {train_faces}")
        print(f"  {prefix} - val labels: {val_labels}, val faces: {val_faces}")
    
    def add_fddb(self, fddb_path, train_split=0.8, prefix="fddb"):
        """Add FDDB dataset"""
        print(f"Adding FDDB dataset with prefix: {prefix}")
        # Implementation for FDDB format conversion
        pass
    
    def count_faces_in_file(self, args):
        """Đếm faces trong một file (helper function cho multiprocessing)"""
        img_file, labels_path = args
        label_file = labels_path / (img_file.stem + ".txt")
        if label_file.exists():
            try:
                with open(label_file, "r") as f:
                    return 1, sum(1 for _ in f)
            except:
                return 0, 0
        return 0, 0
    
    def count_faces_parallel(self, files, labels_path):
        """Đếm faces trong nhiều files song song"""
        args_list = [(img_file, labels_path) for img_file in files]
        total_files = 0
        total_faces = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for file_count, face_count in executor.map(self.count_faces_in_file, args_list):
                total_files += file_count
                total_faces += face_count
                
        return total_files, total_faces
    
    def add_custom_dataset(self, train_path, val_path=None, dataset_type="yolo", prefix="custom", 
                          train_sampling_ratio=1.0, val_sampling_ratio=1.0):
        """Add custom dataset in YOLO format with sampling support"""
        print(f"Adding custom dataset: {prefix}")
        if train_sampling_ratio < 1.0:
            print(f"🎯 Will sample {train_sampling_ratio*100:.1f}% from training set")
        if val_sampling_ratio < 1.0:
            print(f"🎯 Will sample {val_sampling_ratio*100:.1f}% from validation set")
            
        train_images_path = Path(train_path) / "images"
        train_labels_path = Path(train_path) / "labels"
        if not train_images_path.exists() or not train_labels_path.exists():
            print(f"Error: {train_path} must have 'images' and 'labels' folders")
            return
        
        # Sample and copy training files
        if train_sampling_ratio >= 1.0:
            # No sampling - use original method
            print(f"🔍 Quét files trong {train_path}...")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                jpg_future = executor.submit(list, train_images_path.glob("*.jpg"))
                png_future = executor.submit(list, train_images_path.glob("*.png"))
                train_jpg_files = jpg_future.result()
                train_png_files = png_future.result()
                
            train_files = train_jpg_files + train_png_files
            print(f"📊 Đếm faces trong {len(train_files)} files train ({prefix})...")
            train_imgs, train_faces = self.count_faces_parallel(train_files, train_labels_path)
            
            # Copy training files
            self._copy_files(train_files, train_images_path, train_labels_path, self.train_dir, self.train_labels_dir, prefix)
        else:
            # Use sampling
            print(f"🔍 Sampling from {train_path}...")
            
            # Check structure and sample
            subdirs = [d for d in train_images_path.iterdir() if d.is_dir()]
            
            if subdirs:
                # Stratified sampling for subdirectories
                sampled_subdirs = self._sample_stratified(subdirs, train_sampling_ratio)
                train_files = []
                for subdir_info in sampled_subdirs:
                    train_files.extend(subdir_info['sampled_files'])
            else:
                # Flat directory sampling
                all_files = list(train_images_path.glob("*.jpg")) + list(train_images_path.glob("*.png"))
                train_files = self._sample_flat_directory(all_files, train_sampling_ratio)
            
            print(f"📊 Đếm faces trong {len(train_files)} sampled train files ({prefix})...")
            train_imgs, train_faces = self.count_faces_parallel(train_files, train_labels_path)
            
            # Copy sampled files
            if subdirs:
                # Use structured copying for subdirectories
                args_list = [(subdir_info, train_labels_path, self.train_dir, self.train_labels_dir, prefix) 
                            for subdir_info in sampled_subdirs]
                
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    with tqdm(total=len(sampled_subdirs), desc=f"Copying sampled {prefix} train") as pbar:
                        for _ in executor.map(self._copy_sampled_subdir, args_list):
                            pbar.update(1)
            else:
                # Use flat copying
                self._copy_sampled_files(train_files, train_images_path, train_labels_path, self.train_dir, self.train_labels_dir, prefix)
        
        val_imgs = val_faces = 0
        
        if val_path:
            val_images_path = Path(val_path) / "images"
            val_labels_path = Path(val_path) / "labels"
            if not val_images_path.exists() or not val_labels_path.exists():
                print(f"Error: {val_path} must have 'images' and 'labels' folders")
            else:
                # Sample and copy validation files
                if val_sampling_ratio >= 1.0:
                    # No sampling
                    print(f"🔍 Quét files trong {val_path}...")
                    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        jpg_future = executor.submit(list, val_images_path.glob("*.jpg"))
                        png_future = executor.submit(list, val_images_path.glob("*.png"))
                        val_jpg_files = jpg_future.result()
                        val_png_files = png_future.result()
                        
                    val_files = val_jpg_files + val_png_files
                    print(f"📊 Đếm faces trong {len(val_files)} files val ({prefix})...")
                    val_imgs, val_faces = self.count_faces_parallel(val_files, val_labels_path)
                    
                    # Copy validation files 
                    self._copy_files(val_files, val_images_path, val_labels_path, self.val_dir, self.val_labels_dir, prefix)
                else:
                    # Use sampling for validation
                    print(f"🔍 Sampling from {val_path}...")
                    
                    # Check structure and sample
                    val_subdirs = [d for d in val_images_path.iterdir() if d.is_dir()]
                    
                    if val_subdirs:
                        # Stratified sampling for subdirectories
                        sampled_val_subdirs = self._sample_stratified(val_subdirs, val_sampling_ratio)
                        val_files = []
                        for subdir_info in sampled_val_subdirs:
                            val_files.extend(subdir_info['sampled_files'])
                    else:
                        # Flat directory sampling
                        all_val_files = list(val_images_path.glob("*.jpg")) + list(val_images_path.glob("*.png"))
                        val_files = self._sample_flat_directory(all_val_files, val_sampling_ratio)
                    
                    print(f"📊 Đếm faces trong {len(val_files)} sampled val files ({prefix})...")
                    val_imgs, val_faces = self.count_faces_parallel(val_files, val_labels_path)
                    
                    # Copy sampled validation files
                    if val_subdirs:
                        # Use structured copying for subdirectories
                        args_list = [(subdir_info, val_labels_path, self.val_dir, self.val_labels_dir, prefix) 
                                    for subdir_info in sampled_val_subdirs]
                        
                        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                            with tqdm(total=len(sampled_val_subdirs), desc=f"Copying sampled {prefix} val") as pbar:
                                for _ in executor.map(self._copy_sampled_subdir, args_list):
                                    pbar.update(1)
                    else:
                        # Use flat copying
                        self._copy_sampled_files(val_files, val_images_path, val_labels_path, self.val_dir, self.val_labels_dir, prefix)
        
        self.dataset_stats[prefix] = {
            'train_images': train_imgs,
            'train_faces': train_faces,
            'val_images': val_imgs,
            'val_faces': val_faces,
            'train_sampling_ratio': train_sampling_ratio,
            'val_sampling_ratio': val_sampling_ratio
        }
        print(f"  {prefix} - train images: {train_imgs}, train faces: {train_faces}")
        print(f"  {prefix} - val images: {val_imgs}, val faces: {val_faces}")
    
    def _copy_single_subdir(self, args):
        """Copy một thư mục con (helper function cho multiprocessing), bỏ qua file đã tồn tại"""
        subdir, src_images_path, src_labels_path, dst_images, dst_labels, prefix = args
        if not subdir.is_dir():
            return 0, 0
        dst_subdir = dst_images / f"{prefix}_{subdir.name}"
        dst_subdir.mkdir(exist_ok=True)
        img_count = 0
        label_count = 0
        for img_file in subdir.glob("*.jpg"):
            dst_img = dst_subdir / img_file.name
            if dst_img.exists():
                continue
            shutil.copy2(img_file, dst_img)
            img_count += 1
        label_subdir = src_labels_path / subdir.name
        if label_subdir.exists():
            dst_label_subdir = dst_labels / f"{prefix}_{subdir.name}"
            dst_label_subdir.mkdir(exist_ok=True)
            for label_file in label_subdir.glob("*.txt"):
                dst_label = dst_label_subdir / label_file.name
                if dst_label.exists():
                    continue
                shutil.copy2(label_file, dst_label)
                label_count += 1
        return img_count, label_count
    
    def _copy_dataset(self, src_images, src_labels, dst_images, dst_labels, prefix):
        """Copy dataset with prefix using multiprocessing"""
        src_images_path = Path(src_images)
        src_labels_path = Path(src_labels)
        
        if not src_images_path.exists():
            print(f"Warning: {src_images_path} does not exist")
            return
            
        # Get all subdirectories (WiderFace structure)
        subdirs = [d for d in src_images_path.iterdir() if d.is_dir()]
        total = len(subdirs)
        
        if total == 0:
            return
            
        print(f"🔄 Copying {prefix} dataset với {self.num_workers} processes...")
        
        # Tạo arguments cho multiprocessing
        args_list = [(subdir, src_images_path, src_labels_path, dst_images, dst_labels, prefix) 
                    for subdir in subdirs]
        
        # Sử dụng ProcessPoolExecutor để copy song song
        img_count = 0
        label_count = 0
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=total, desc=f"Copying {prefix}") as pbar:
                for i, (imgs, labels) in enumerate(executor.map(self._copy_single_subdir, args_list)):
                    img_count += imgs
                    label_count += labels
                    pbar.update(1)
                    
        print(f"✅ {prefix}: Đã copy {img_count} ảnh và {label_count} labels")
    
    def _copy_single_file(self, args):
        """Copy một file riêng lẻ (helper function cho multiprocessing), bỏ qua file đã tồn tại"""
        img_file, src_images_path, src_labels_path, dst_images, dst_labels, prefix = args
        dst_img = dst_images / f"{prefix}_{img_file.name}"
        if dst_img.exists():
            return 0, 0
        shutil.copy2(img_file, dst_img)
        label_copied = False
        label_file = src_labels_path / (img_file.stem + ".txt")
        dst_label = dst_labels / f"{prefix}_{img_file.stem}.txt"
        if label_file.exists():
            if not dst_label.exists():
                shutil.copy2(label_file, dst_label)
            label_copied = True
        return 1, 1 if label_copied else 0

    def _copy_files(self, files, src_images_path, src_labels_path, dst_images, dst_labels, prefix):
        """Copy individual files with prefix using multiprocessing"""
        total = len(files)
        if total == 0:
            return
            
        print(f"🔄 Copying {prefix} files với {self.num_workers} processes...")
        
        # Chunk files để xử lý song song hiệu quả hơn
        args_list = [(img_file, src_images_path, src_labels_path, dst_images, dst_labels, prefix) 
                    for img_file in files]
        
        # Sử dụng ProcessPoolExecutor để copy song song
        img_count = 0
        label_count = 0
        
        chunk_size = max(1, min(1000, total // self.num_workers))
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=total, desc=f"Copying {prefix}") as pbar:
                for i, (img, label) in enumerate(executor.map(self._copy_single_file, args_list, chunksize=chunk_size)):
                    img_count += img
                    label_count += label
                    pbar.update(1)
                    
        print(f"✅ {prefix}: Đã copy {img_count} ảnh và {label_count} labels")
    
    def create_config(self, config_path="data/multi_face.yaml"):
        """Create YAML config for merged dataset"""
        config = {
            'train': str(self.train_dir) + "/",
            'val': str(self.val_dir) + "/", 
            'nc': 1,
            'names': ['face']
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Config saved to: {config_path}")
    
    def get_statistics(self):
        """Get dataset statistics sử dụng đa luồng"""
        print(f"🔍 Đang tính toán thống kê song song với {self.num_workers} workers...")
        
        # Liệt kê files song song sử dụng ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            train_jpg_future = executor.submit(list, self.train_dir.rglob("*.jpg"))
            train_png_future = executor.submit(list, self.train_dir.rglob("*.png"))
            train_txt_future = executor.submit(list, self.train_labels_dir.rglob("*.txt"))
            val_jpg_future = executor.submit(list, self.val_dir.rglob("*.jpg"))
            val_png_future = executor.submit(list, self.val_dir.rglob("*.png"))
            val_txt_future = executor.submit(list, self.val_labels_dir.rglob("*.txt"))
            
            train_jpg_files = train_jpg_future.result()
            train_png_files = train_png_future.result()
            train_txt_files = train_txt_future.result()
            val_jpg_files = val_jpg_future.result()
            val_png_files = val_png_future.result()
            val_txt_files = val_txt_future.result()
        
        train_images = len(train_jpg_files) + len(train_png_files)
        train_labels = len(train_txt_files)
        val_images = len(val_jpg_files) + len(val_png_files)
        val_labels = len(val_txt_files)
        
        print("🔢 Đếm số lượng faces song song...")
        train_faces = count_faces_parallel(train_txt_files, self.num_workers)
        val_faces = count_faces_parallel(val_txt_files, self.num_workers)
        total_faces = train_faces + val_faces
        stats = {
            'train_images': train_images,
            'train_labels': train_labels,
            'train_faces': train_faces,
            'val_images': val_images,
            'val_labels': val_labels,
            'val_faces': val_faces,
            'total_images': train_images + val_images,
            'total_faces': total_faces
        }
        print("\n=== Dataset Statistics (Merged) ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("\n=== Statistics by Dataset ===")
        for k, v in self.dataset_stats.items():
            print(f"{k}: {v}")
        # So sánh với WiderFace nếu có
        if 'wider' in self.dataset_stats:
            wider_total_faces = self.dataset_stats['wider'].get('train_faces', 0) + self.dataset_stats['wider'].get('val_faces', 0)
            wider_total_labels = self.dataset_stats['wider'].get('train_labels', 0) + self.dataset_stats['wider'].get('val_labels', 0)
            print(f"\nSo sánh với WiderFace gốc:")
            print(f"  Tổng số face WiderFace: {wider_total_faces}")
            print(f"  Tổng số face sau khi gộp: {total_faces}")
            print(f"  Tỉ lệ tăng: {total_faces/wider_total_faces:.2f}x")
            print(f"  Tổng số ảnh WiderFace: {wider_total_labels}")
            print(f"  Tổng số ảnh sau khi gộp: {train_images + val_images}")
            print(f"  Tỉ lệ tăng: {(train_images + val_images)/wider_total_labels:.2f}x")
        return stats

    def _sample_stratified(self, subdirs, sampling_ratio):
        """
        Phân tầng sampling: lấy một tỷ lệ từ mỗi thư mục con để đảm bảo distribution
        Args:
            subdirs: danh sách các thư mục con
            sampling_ratio: tỷ lệ sampling (0.0 - 1.0)
        """
        import random
        sampled_subdirs = []
        
        for subdir in subdirs:
            # Đếm số file trong thư mục con
            img_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
            total_files = len(img_files)
            
            if total_files == 0:
                continue
                
            # Tính số file cần lấy từ thư mục này
            sample_count = max(1, int(total_files * sampling_ratio))
            
            # Random sampling các file
            sampled_files = random.sample(img_files, min(sample_count, total_files))
            
            if sampled_files:
                sampled_subdirs.append({
                    'subdir': subdir,
                    'sampled_files': sampled_files,
                    'total_files': total_files,
                    'sampled_count': len(sampled_files)
                })
        
        return sampled_subdirs
    
    def _sample_flat_directory(self, img_files, sampling_ratio):
        """
        Random sampling cho thư mục phẳng (không có thư mục con)
        Args:
            img_files: danh sách các file ảnh
            sampling_ratio: tỷ lệ sampling (0.0 - 1.0)
        """
        import random
        total_files = len(img_files)
        
        if total_files == 0:
            return []
            
        # Tính số file cần lấy
        sample_count = max(1, int(total_files * sampling_ratio))
        
        # Random sampling
        sampled_files = random.sample(img_files, min(sample_count, total_files))
        
        return sampled_files
    
    def _copy_sampled_subdir(self, args):
        """Copy sampled files from a subdirectory (helper for multiprocessing)"""
        subdir_info, src_labels_path, dst_images, dst_labels, prefix = args
        subdir = subdir_info['subdir']
        sampled_files = subdir_info['sampled_files']
        
        # Tạo thư mục đích
        dst_subdir = dst_images / f"{prefix}_{subdir.name}"
        dst_subdir.mkdir(exist_ok=True)
        
        # Tạo thư mục label đích
        dst_label_subdir = dst_labels / f"{prefix}_{subdir.name}"
        dst_label_subdir.mkdir(exist_ok=True)
        
        img_count = 0
        label_count = 0
        
        for img_file in sampled_files:
            # Copy image
            dst_img = dst_subdir / img_file.name
            if not dst_img.exists():
                shutil.copy2(img_file, dst_img)
                img_count += 1
            
            # Copy corresponding label
            label_file = src_labels_path / subdir.name / (img_file.stem + ".txt")
            if label_file.exists():
                dst_label = dst_label_subdir / label_file.name
                if not dst_label.exists():
                    shutil.copy2(label_file, dst_label)
                    label_count += 1
        
        return img_count, label_count
    
    def _copy_sampled_files(self, sampled_files, src_images_path, src_labels_path, dst_images, dst_labels, prefix):
        """Copy sampled files for flat directory structure"""
        total = len(sampled_files)
        if total == 0:
            return
            
        print(f"🔄 Copying {total} sampled {prefix} files với {self.num_workers} processes...")
        
        # Tạo arguments cho multiprocessing
        args_list = [(img_file, src_images_path, src_labels_path, dst_images, dst_labels, prefix) 
                    for img_file in sampled_files]
        
        # Sử dụng ProcessPoolExecutor để copy song song
        img_count = 0
        label_count = 0
        
        chunk_size = max(1, min(1000, total // self.num_workers))
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=total, desc=f"Copying sampled {prefix}") as pbar:
                for i, (img, label) in enumerate(executor.map(self._copy_single_file, args_list, chunksize=chunk_size)):
                    img_count += img
                    label_count += label
                    pbar.update(1)
                    
        print(f"✅ {prefix}: Đã copy {img_count} ảnh và {label_count} labels (sampled)")
    
    def _copy_sampled_dataset(self, src_images, src_labels, dst_images, dst_labels, prefix, sampling_ratio=1.0):
        """Copy dataset with sampling support"""
        src_images_path = Path(src_images)
        src_labels_path = Path(src_labels)
        
        if not src_images_path.exists():
            print(f"Warning: {src_images_path} does not exist")
            return
        
        if sampling_ratio >= 1.0:
            # Không sampling, dùng method cũ
            return self._copy_dataset(src_images, src_labels, dst_images, dst_labels, prefix)
        
        print(f"🎯 Sampling {sampling_ratio*100:.1f}% from {prefix} dataset...")
        
        # Check cấu trúc thư mục
        subdirs = [d for d in src_images_path.iterdir() if d.is_dir()]
        
        if subdirs:
            # Có thư mục con -> dùng stratified sampling
            print(f"📁 Detected {len(subdirs)} subdirectories, using stratified sampling")
            sampled_subdirs = self._sample_stratified(subdirs, sampling_ratio)
            
            if not sampled_subdirs:
                print(f"⚠️ No files to sample from {prefix}")
                return
            
            # Copy sampled subdirectories
            args_list = [(subdir_info, src_labels_path, dst_images, dst_labels, prefix) 
                        for subdir_info in sampled_subdirs]
            
            img_count = 0
            label_count = 0
            
            print(f"🔄 Processing {len(sampled_subdirs)} sampled subdirectories...")
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                with tqdm(total=len(sampled_subdirs), desc=f"Copying sampled {prefix}") as pbar:
                    for imgs, labels in executor.map(self._copy_sampled_subdir, args_list):
                        img_count += imgs
                        label_count += labels
                        pbar.update(1)
            
            # Print sampling stats
            total_original = sum(info['total_files'] for info in sampled_subdirs)
            total_sampled = sum(info['sampled_count'] for info in sampled_subdirs)
            
            print(f"✅ {prefix}: Đã copy {img_count} ảnh và {label_count} labels")
            print(f"📊 Sampling stats: {total_sampled}/{total_original} files ({total_sampled/total_original*100:.1f}%)")
            
        else:
            # Thư mục phẳng -> dùng random sampling
            print(f"📄 Flat directory structure, using random sampling")
            img_files = list(src_images_path.glob("*.jpg")) + list(src_images_path.glob("*.png"))
            
            if not img_files:
                print(f"⚠️ No image files found in {prefix}")
                return
            
            sampled_files = self._sample_flat_directory(img_files, sampling_ratio)
            print(f"📊 Sampled {len(sampled_files)}/{len(img_files)} files ({len(sampled_files)/len(img_files)*100:.1f}%)")
            
            # Copy sampled files
            self._copy_sampled_files(sampled_files, src_images_path, src_labels_path, dst_images, dst_labels, prefix)

    # ...existing code...
def count_faces_parallel(label_files, num_workers=8):
    """Đếm số faces trong nhiều file label song song"""
    def count_faces_in_file(file_path):
        try:
            with open(file_path, "r") as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(count_faces_in_file, label_files))
    return sum(results)

def main():
    import time
    start_time = time.time()

    # Đặt số lượng worker (tận dụng 20 cores)
    num_workers = 10
    
    print(f"🚀 Bắt đầu merge dataset với {num_workers} cores...")
    
    # Initialize merger
    merger = DatasetMerger("/mnt/md0/projects/nguyendai-footage/data-clean/enhanced_face_dataset_wider_face_and_closed", num_workers=num_workers)

    # Add WiderFace (base dataset)
    merger.add_widerface(
        "/mnt/md0/projects/nguyendai-footage/WIDER_train", 
        "/mnt/md0/projects/nguyendai-footage/WIDER_val",
        train_sampling_ratio=1, val_sampling_ratio=1,
        prefix="wider"
    )

    # Add custom datasets (example)
    merger.add_custom_dataset("/mnt/md0/projects/nguyendai-footage/roboflow/close_face_new/train", 
                             val_path="/mnt/md0/projects/nguyendai-footage/roboflow/close_face_new/valid", 
                             prefix="closed_face")
    # merger.add_custom_dataset("/mnt/data1/arv/data-clean/blur_dataset_new/train", 
    #                          val_path="/mnt/data1/arv/data-clean/blur_dataset_new/val",train_sampling_ratio=0.8, val_sampling_ratio=0.8, 
    #                          prefix="blurred_face")
    # merger.add_custom_dataset("/mnt/data1/arv/data-clean/lag_train", 
    #                          val_path="/mnt/data1/arv/data-clean/lag_val", 
    #                          prefix="lag_face")

    # merger.add_custom_dataset("/path/to/custom2", val_path="/path/to/custom2_val", prefix="custom2")
    # merger.add_fddb("path/to/fddb", prefix="fddb")

    # Create config file
    merger.create_config("data/enhanced_wider_face_and_closed_not_augmented.yaml")
    # Gộp các bộ val thành 1 bộ val chung
    merger.merge_val_sets()
    # Gộp các bộ train thành 1 bộ train chung (thống kê)
    merger.merge_train_sets()
    # Get statistics
    merger.get_statistics()
    # Lưu thống kê ra file JSON
    merger.save_stats_json("merge_enhanced_wider_face_and_closed_not_augmented.json")

    # In thời gian thực thi
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n✨ Hoàn tất merge dataset trong {minutes} phút {seconds} giây")
    print(f"🔥 Sử dụng {num_workers} cores song song để xử lý")

if __name__ == "__main__":
    # Uncomment to test balanced sampling
    # test_balanced_sampling()
    
    # Production merge (current configuration)
    main()

# def test_balanced_sampling():
#     """Test balanced sampling functionality with example usage"""
#     print("🧪 Testing balanced sampling functionality...")
    
#     merger = DatasetMerger("test_merged_datasets", num_workers=4)
    
#     print("\n📖 Example 1: 50/50 balanced sampling")
#     print("merger.add_widerface(train_path, val_path, train_sampling_ratio=0.5, val_sampling_ratio=0.5)")
#     print("merger.add_custom_dataset(custom_train, custom_val, train_sampling_ratio=0.5, val_sampling_ratio=0.5)")
    
#     print("\n📖 Example 2: Training heavy (80% train, 30% val)")  
#     print("merger.add_widerface(train_path, val_path, train_sampling_ratio=0.8, val_sampling_ratio=0.3)")
    
#     print("\n📖 Example 3: Full dataset + sampled augmentation")
#     print("merger.add_widerface(train_path, val_path)  # 100% base dataset")
#     print("merger.add_custom_dataset(augment_path, prefix='augment', train_sampling_ratio=0.2)  # 20% augmentation")
    
#     print("\n🎯 Sampling Features:")
#     print("  ✅ Stratified sampling for subdirectory datasets (preserves distribution)")
#     print("  ✅ Random sampling for flat directory structures")
#     print("  ✅ Multiprocessing optimization for large datasets")
#     print("  ✅ Statistics tracking with sampling ratios")
#     print("  ✅ Independent train/val sampling ratios")
    
#     # Create example YAML
#     merger.dataset_stats = {
#         'wider': {'train_images': 8000, 'train_faces': 20000, 'val_images': 2000, 'val_faces': 5000, 'train_sampling_ratio': 0.5, 'val_sampling_ratio': 0.5},
#         'custom': {'train_images': 4000, 'train_faces': 8000, 'val_images': 1000, 'val_faces': 2000, 'train_sampling_ratio': 0.5, 'val_sampling_ratio': 0.5}
#     }
    
#     merger.create_yaml("test_balanced.yaml", description="Example: 50%/50% balanced sampling")
#     print(f"\n📄 Created example config: test_balanced.yaml")
#     merger.print_stats()

