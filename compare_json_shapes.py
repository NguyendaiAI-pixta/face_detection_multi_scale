#!/usr/bin/env python3
"""
So sánh các file JSON trong 2 thư mục và kiểm tra giá trị đầu tiên của shape
Tìm các file có cùng tên nhưng shape[0] khác nhau
"""

import os
import json
import argparse
from pathlib import Path
import glob

class JSONShapeComparator:
    """So sánh shape[0] của các file JSON cùng tên trong 2 thư mục"""
    
    def __init__(self, folder1, folder2):
        self.folder1 = folder1
        self.folder2 = folder2
        self.mismatches = []
        self.matches = []
        self.errors = []
        
    def get_shape_first_value(self, json_data):
        """Lấy giá trị đầu tiên của shape từ yolo_face_prediction"""
        try:
            if "yolo_face_prediction" in json_data:
                predictions = json_data["yolo_face_prediction"]
                if isinstance(predictions, list) and len(predictions) > 0:
                    # Tìm item có name là "yolo-face-bboxes"
                    for item in predictions:
                        if isinstance(item, dict) and item.get("name") == "yolo-face-bboxes":
                            shape = item.get("shape", [])
                            if isinstance(shape, list) and len(shape) > 0:
                                return shape[0]
            return None
        except Exception as e:
            return None
    
    def load_json_file(self, file_path):
        """Load và parse JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, None
        except Exception as e:
            return None, str(e)
    
    def compare_files(self):
        """So sánh các file JSON trong 2 thư mục"""
        print(f"🔍 Comparing JSON files between:")
        print(f"   📁 Folder 1: {os.path.abspath(self.folder1)}")
        print(f"   📁 Folder 2: {os.path.abspath(self.folder2)}")
        
        # Lấy danh sách file JSON từ folder 1
        json_files1 = set()
        for ext in ['*.json', '*.JSON']:
            json_files1.update([os.path.basename(f) for f in glob.glob(os.path.join(self.folder1, ext))])
        
        # Lấy danh sách file JSON từ folder 2
        json_files2 = set()
        for ext in ['*.json', '*.JSON']:
            json_files2.update([os.path.basename(f) for f in glob.glob(os.path.join(self.folder2, ext))])
        
        # Tìm file chung
        common_files = json_files1.intersection(json_files2)
        
        if not common_files:
            print("❌ No common JSON files found between the two folders")
            return
        
        print(f"📊 Found {len(common_files)} common JSON files")
        print(f"📄 Files only in folder 1: {len(json_files1 - json_files2)}")
        print(f"📄 Files only in folder 2: {len(json_files2 - json_files1)}")
        print()
        
        # So sánh từng file
        for filename in sorted(common_files):
            file1_path = os.path.join(self.folder1, filename)
            file2_path = os.path.join(self.folder2, filename)
            
            # Load file 1
            data1, error1 = self.load_json_file(file1_path)
            if error1:
                self.errors.append({
                    'file': filename,
                    'error': f"Cannot load file from folder 1: {error1}",
                    'path': file1_path
                })
                continue
            
            # Load file 2
            data2, error2 = self.load_json_file(file2_path)
            if error2:
                self.errors.append({
                    'file': filename,
                    'error': f"Cannot load file from folder 2: {error2}",
                    'path': file2_path
                })
                continue
            
            # Lấy shape[0] values
            shape1 = self.get_shape_first_value(data1)
            shape2 = self.get_shape_first_value(data2)
            
            if shape1 is None or shape2 is None:
                self.errors.append({
                    'file': filename,
                    'error': f"Cannot extract shape[0] value (shape1: {shape1}, shape2: {shape2})",
                    'path1': file1_path,
                    'path2': file2_path
                })
                continue
            
            # So sánh
            if shape1 != shape2:
                self.mismatches.append({
                    'file': filename,
                    'shape1': shape1,
                    'shape2': shape2,
                    'path1': file1_path,
                    'path2': file2_path
                })
                print(f"❌ MISMATCH: {filename}")
                print(f"   📁 Folder 1: shape[0] = {shape1}")
                print(f"   📁 Folder 2: shape[0] = {shape2}")
                print(f"   📂 Path 1: {file1_path}")
                print(f"   📂 Path 2: {file2_path}")
                print()
            else:
                self.matches.append({
                    'file': filename,
                    'shape': shape1,
                    'path1': file1_path,
                    'path2': file2_path
                })
    
    def generate_report(self, output_file=None):
        """Tạo báo cáo chi tiết"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("JSON SHAPE COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"📁 Folder 1: {os.path.abspath(self.folder1)}")
        report_lines.append(f"📁 Folder 2: {os.path.abspath(self.folder2)}")
        report_lines.append("")
        
        # Summary
        total_files = len(self.matches) + len(self.mismatches) + len(self.errors)
        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total files compared: {total_files}")
        report_lines.append(f"✅ Matching files: {len(self.matches)}")
        report_lines.append(f"❌ Mismatching files: {len(self.mismatches)}")
        report_lines.append(f"⚠️ Error files: {len(self.errors)}")
        report_lines.append("")
        
        # Mismatches (most important)
        if self.mismatches:
            report_lines.append("❌ MISMATCHING FILES")
            report_lines.append("-" * 40)
            for mismatch in self.mismatches:
                report_lines.append(f"File: {mismatch['file']}")
                report_lines.append(f"  📁 Folder 1 shape[0]: {mismatch['shape1']}")
                report_lines.append(f"  📁 Folder 2 shape[0]: {mismatch['shape2']}")
                report_lines.append(f"  📂 Path 1: {mismatch['path1']}")
                report_lines.append(f"  📂 Path 2: {mismatch['path2']}")
                report_lines.append("")
        else:
            report_lines.append("✅ No mismatching files found!")
            report_lines.append("")
        
        # Errors
        if self.errors:
            report_lines.append("⚠️ ERROR FILES")
            report_lines.append("-" * 40)
            for error in self.errors:
                report_lines.append(f"File: {error['file']}")
                report_lines.append(f"  Error: {error['error']}")
                if 'path1' in error:
                    report_lines.append(f"  Path 1: {error['path1']}")
                if 'path2' in error:
                    report_lines.append(f"  Path 2: {error['path2']}")
                if 'path' in error:
                    report_lines.append(f"  Path: {error['path']}")
                report_lines.append("")
        
        # Matching files (summary only to avoid too much output)
        if self.matches:
            report_lines.append("✅ MATCHING FILES SUMMARY")
            report_lines.append("-" * 40)
            # Group by shape value
            shape_groups = {}
            for match in self.matches:
                shape = match['shape']
                if shape not in shape_groups:
                    shape_groups[shape] = []
                shape_groups[shape].append(match['file'])
            
            for shape, files in sorted(shape_groups.items()):
                report_lines.append(f"Shape[0] = {shape}: {len(files)} files")
                if len(files) <= 5:
                    for file in files:
                        report_lines.append(f"  - {file}")
                else:
                    for file in files[:3]:
                        report_lines.append(f"  - {file}")
                    report_lines.append(f"  ... and {len(files) - 3} more files")
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Print to console
        print(report_content)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"📄 Report saved to: {os.path.abspath(output_file)}")
            except Exception as e:
                print(f"❌ Cannot save report to {output_file}: {e}")
    
    def print_summary(self):
        """In tóm tắt ngắn gọn"""
        print("\n" + "="*60)
        print("🎯 COMPARISON SUMMARY")
        print("="*60)
        
        total_files = len(self.matches) + len(self.mismatches) + len(self.errors)
        print(f"📊 Total files compared: {total_files}")
        print(f"✅ Matching files: {len(self.matches)}")
        print(f"❌ Mismatching files: {len(self.mismatches)}")
        print(f"⚠️ Error files: {len(self.errors)}")
        
        if self.mismatches:
            print(f"\n🚨 FILES WITH DIFFERENT SHAPE[0] VALUES:")
            for mismatch in self.mismatches:
                print(f"   • {mismatch['file']}: {mismatch['shape1']} vs {mismatch['shape2']}")
        else:
            print(f"\n🎉 All compared files have matching shape[0] values!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compare JSON files shape[0] values between two folders')
    parser.add_argument('--folder1', type=str, required=True,
                      help='Path to first folder containing JSON files')
    parser.add_argument('--folder2', type=str, required=True,
                      help='Path to second folder containing JSON files')
    parser.add_argument('--output', type=str, default=None,
                      help='Output report file path (optional)')
    parser.add_argument('--verbose', action='store_true',
                      help='Show detailed output for matching files')
    
    args = parser.parse_args()
    
    # Validate input folders
    if not os.path.exists(args.folder1):
        print(f"❌ Folder 1 not found: {args.folder1}")
        return
    
    if not os.path.exists(args.folder2):
        print(f"❌ Folder 2 not found: {args.folder2}")
        return
    
    print("🔍 JSON Shape Comparator")
    print("=" * 50)
    
    # Create comparator and run comparison
    comparator = JSONShapeComparator(args.folder1, args.folder2)
    comparator.compare_files()
    
    # Generate report
    if args.verbose or args.output:
        comparator.generate_report(args.output)
    else:
        comparator.print_summary()


if __name__ == "__main__":
    main()

"""
🚀 Cách sử dụng:

# 1. So sánh cơ bản 2 thư mục:
python compare_json_shapes.py \
    --folder1 "/path/to/folder1" \
    --folder2 "/path/to/folder2"

# 2. So sánh và lưu báo cáo chi tiết:
python compare_json_shapes.py \
    --folder1 "/path/to/folder1" \
    --folder2 "/path/to/folder2" \
    --output "comparison_report.txt"

# 3. So sánh với output chi tiết:
python compare_json_shapes.py \
    --folder1 "/path/to/folder1" \
    --folder2 "/path/to/folder2" \
    --verbose

# 4. Ví dụ cụ thể:
python compare_json_shapes.py \
    --folder1 "./json_results_df" \
    --folder2 "./json_results_backup" \
    --output "shape_comparison_report.txt" \
    --verbose

📊 Tool sẽ:
• Tìm tất cả file JSON có cùng tên trong 2 thư mục
• Đọc giá trị shape[0] từ yolo-face-bboxes trong mỗi file
• So sánh và báo cáo những file có shape[0] khác nhau
• Tạo báo cáo chi tiết với đường dẫn đầy đủ
• Hiển thị summary và statistics

🎯 Output sẽ hiển thị:
• Danh sách file MISMATCH (quan trọng nhất)
• Đường dẫn đầy đủ đến từng file
• Giá trị shape[0] của từng thư mục
• Tổng kết số lượng file match/mismatch/error

Example JSON structure expected:
{
  "yolo_face_prediction": [
    {
      "name": "yolo-face-bboxes",
      "shape": [11, 0, 4],  // 11 là giá trị được so sánh
      ...
    }
  ]
}
"""
