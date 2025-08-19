#!/usr/bin/env python3
"""
Multi-dataset merger for YOLOv7-Face training
Automatically converts and merges multiple face detection datasets
"""
import os
import shutil
from pathlib import Path
import json
import yaml
from tqdm import tqdm

'''
- tạo thư mục chứa dữ liệu gộp
- copy ảnh và label từ các nguồn vào thư mục mới (tiền tố prefix) để phân biệt thư mục gốc
- tạo file cấu hình mới chỉ định đường dẫn train/val, số class, tên class
'''
class DatasetMerger:
    def merge_val_sets(self):
        """Gộp tất cả các bộ val thành 1 bộ val chung"""
        print("\nGộp tất cả các bộ val thành 1 bộ val chung...")
        val_common_img = self.output_dir / "val" / "images"
        val_common_lbl = self.output_dir / "val" / "labels"
        val_common_img.mkdir(parents=True, exist_ok=True)
        val_common_lbl.mkdir(parents=True, exist_ok=True)
        prefixes = [k for k in self.dataset_stats.keys() if k.endswith('_val') or k == 'wider_val']
        total_val_imgs = 0
        total_val_faces = 0
        for prefix in prefixes:
            # Tìm các ảnh val của từng nguồn
            val_img_dir = self.val_dir
            val_lbl_dir = self.val_labels_dir
            img_files = list(val_img_dir.glob(f"{prefix}_*.jpg")) + list(val_img_dir.glob(f"{prefix}_*.png"))
            for img_file in img_files:
                dst_img = val_common_img / img_file.name
                shutil.copy2(img_file, dst_img)
                total_val_imgs += 1
                lbl_file = val_lbl_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    dst_lbl = val_common_lbl / lbl_file.name
                    shutil.copy2(lbl_file, dst_lbl)
                    with open(lbl_file, "r") as f:
                        total_val_faces += sum(1 for _ in f)
        print(f"Đã gộp xong val chung: {total_val_imgs} ảnh, {total_val_faces} faces")
        self.dataset_stats['val_merged'] = {'val_images': total_val_imgs, 'val_faces': total_val_faces}
    def merge_train_sets(self):
        """Gộp tất cả các bộ train thành 1 bộ train chung (đã mặc định)"""
        # Do các bộ train đã được copy vào train_dir, chỉ cần thống kê lại
        train_imgs = len(list(self.train_dir.glob("*.jpg"))) + len(list(self.train_dir.glob("*.png")))
        train_faces = 0
        for lbl_file in self.train_labels_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                train_faces += sum(1 for _ in f)
        self.dataset_stats['train_merged'] = {'train_images': train_imgs, 'train_faces': train_faces}
    def save_stats_json(self, json_path="merge_stats.json"):
        """Lưu thống kê dữ liệu vào file JSON"""
        with open(json_path, "w") as f:
            json.dump(self.dataset_stats, f, indent=2)
        print(f"Thống kê dữ liệu đã lưu vào {json_path}")
    def __init__(self, output_dir="/mnt/md0/projects/nguyendai-footage/merged_dataset"):
        self.output_dir = Path(output_dir)
        self.train_dir = self.output_dir / "train" / "images"
        self.train_labels_dir = self.output_dir / "train" / "labels"
        self.val_dir = self.output_dir / "val" / "images"
        self.val_labels_dir = self.output_dir / "val" / "labels"
        # Thống kê từng dataset
        self.dataset_stats = {}
        # Create directories
        for dir_path in [self.train_dir, self.train_labels_dir, self.val_dir, self.val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def add_widerface(self, wider_train_path, wider_val_path, prefix="wider"):
        """Add WiderFace dataset"""
        print(f"Adding WiderFace dataset with prefix: {prefix}")
        # Thống kê số ảnh/số face
        def count_faces_labels(labels_dir):
            total_faces = 0
            total_labels = 0
            for label_file in Path(labels_dir).rglob("*.txt"):
                total_labels += 1
                with open(label_file, "r") as f:
                    total_faces += sum(1 for _ in f)
            return total_labels, total_faces
        # Copy training images and labels
        self._copy_dataset(
            wider_train_path + "/images", 
            wider_train_path + "/labels",
            self.train_dir, 
            self.train_labels_dir, 
            prefix
        )
        # Copy validation images and labels  
        self._copy_dataset(
            wider_val_path + "/images",
            wider_val_path + "/labels", 
            self.val_dir,
            self.val_labels_dir,
            prefix + "_val"
        )
        # Thống kê sau khi copy
        train_labels, train_faces = count_faces_labels(self.train_labels_dir)
        val_labels, val_faces = count_faces_labels(self.val_labels_dir)
        self.dataset_stats[prefix] = {
            'train_labels': train_labels,
            'train_faces': train_faces,
            'val_labels': val_labels,
            'val_faces': val_faces
        }
        print(f"  {prefix} - train labels: {train_labels}, train faces: {train_faces}")
        print(f"  {prefix} - val labels: {val_labels}, val faces: {val_faces}")
    
    def add_fddb(self, fddb_path, train_split=0.8, prefix="fddb"):
        """Add FDDB dataset"""
        print(f"Adding FDDB dataset with prefix: {prefix}")
        # Implementation for FDDB format conversion
        pass
    
    def add_custom_dataset(self, dataset_path, dataset_type="yolo", train_split=0.8, prefix="custom"):
        """Add custom dataset in YOLO format"""
        print(f"Adding custom dataset: {prefix}")
        images_path = Path(dataset_path) / "images"
        labels_path = Path(dataset_path) / "labels"
        if not images_path.exists() or not labels_path.exists():
            print(f"Error: {dataset_path} must have 'images' and 'labels' folders")
            return
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        train_count = int(len(image_files) * train_split)
        train_files = image_files[:train_count]
        val_files = image_files[train_count:]
        # Thống kê số face
        def count_faces(files, labels_path):
            total_faces = 0
            for img_file in files:
                label_file = labels_path / (img_file.stem + ".txt")
                if label_file.exists():
                    with open(label_file, "r") as f:
                        total_faces += sum(1 for _ in f)
            return len(files), total_faces
        train_imgs, train_faces = count_faces(train_files, labels_path)
        val_imgs, val_faces = count_faces(val_files, labels_path)
        self.dataset_stats[prefix] = {
            'train_images': train_imgs,
            'train_faces': train_faces,
            'val_images': val_imgs,
            'val_faces': val_faces
        }
        print(f"  {prefix} - train images: {train_imgs}, train faces: {train_faces}")
        print(f"  {prefix} - val images: {val_imgs}, val faces: {val_faces}")
        # Copy training files
        self._copy_files(train_files, images_path, labels_path, self.train_dir, self.train_labels_dir, prefix)
        # Copy validation files
        self._copy_files(val_files, images_path, labels_path, self.val_dir, self.val_labels_dir, prefix)
    
    def _copy_dataset(self, src_images, src_labels, dst_images, dst_labels, prefix):
        """Copy dataset with prefix"""
        src_images_path = Path(src_images)
        src_labels_path = Path(src_labels)
        
        if not src_images_path.exists():
            print(f"Warning: {src_images_path} does not exist")
            return
            
        # Get all subdirectories (WiderFace structure)
        for subdir in tqdm(src_images_path.iterdir(), desc=f"Copying {prefix}"):
            if subdir.is_dir():
                # Copy images
                dst_subdir = dst_images / f"{prefix}_{subdir.name}"
                dst_subdir.mkdir(exist_ok=True)
                
                for img_file in subdir.glob("*.jpg"):
                    dst_img = dst_subdir / img_file.name
                    shutil.copy2(img_file, dst_img)
                
                # Copy corresponding labels
                label_subdir = src_labels_path / subdir.name
                if label_subdir.exists():
                    dst_label_subdir = dst_labels / f"{prefix}_{subdir.name}"
                    dst_label_subdir.mkdir(exist_ok=True)
                    
                    for label_file in label_subdir.glob("*.txt"):
                        dst_label = dst_label_subdir / label_file.name
                        shutil.copy2(label_file, dst_label)
    
    def _copy_files(self, files, src_images_path, src_labels_path, dst_images, dst_labels, prefix):
        """Copy individual files with prefix"""
        for img_file in tqdm(files, desc=f"Copying {prefix}"):
            # Copy image
            dst_img = dst_images / f"{prefix}_{img_file.name}"
            shutil.copy2(img_file, dst_img)
            
            # Copy corresponding label
            label_file = src_labels_path / (img_file.stem + ".txt")
            if label_file.exists():
                dst_label = dst_labels / f"{prefix}_{img_file.stem}.txt"
                shutil.copy2(label_file, dst_label)
    
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
        """Get dataset statistics"""
        train_images = len(list(self.train_dir.rglob("*.jpg"))) + len(list(self.train_dir.rglob("*.png")))
        train_labels = len(list(self.train_labels_dir.rglob("*.txt")))
        val_images = len(list(self.val_dir.rglob("*.jpg"))) + len(list(self.val_dir.rglob("*.png")))
        val_labels = len(list(self.val_labels_dir.rglob("*.txt")))
        def count_total_faces(labels_dir):
            total_faces = 0
            for label_file in labels_dir.rglob("*.txt"):
                with open(label_file, "r") as f:
                    total_faces += sum(1 for _ in f)
            return total_faces
        train_faces = count_total_faces(self.train_labels_dir)
        val_faces = count_total_faces(self.val_labels_dir)
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

def main():
    # Initialize merger
    merger = DatasetMerger("/mnt/md0/projects/nguyendai-footage/enhanced_face_dataset")
    
    # Add WiderFace (base dataset)
    merger.add_widerface(
        "WIDER_train", 
        "WIDER_val",
        "wider"
    )
    
    # Add custom datasets (example)
    merger.add_custom_dataset("/mnt/md0/projects/nguyendai-footage/roboflow/close_face_new/train", prefix="closed_face")
    merger.add_custom_dataset("/mnt/md0/projects/nguyendai-footage/blur_dataset/train", prefix="blurred_face")
    # merger.add_custom_dataset("/path/to/custom2", prefix="custom2")
    # merger.add_fddb("path/to/fddb", prefix="fddb")
    
    # Create config file
    merger.create_config("data/enhanced_face.yaml")
    # Gộp các bộ val thành 1 bộ val chung
    merger.merge_val_sets()
    # Gộp các bộ train thành 1 bộ train chung (thống kê)
    merger.merge_train_sets()
    # Get statistics
    merger.get_statistics()
    # Lưu thống kê ra file JSON
    merger.save_stats_json("merge_stats.json")

if __name__ == "__main__":
    main()

