#!/usr/bin/env python3
"""
So sánh các phương pháp resize từ dual preprocessing face detection
Hiển thị trực quan sự khác biệt giữa API Framework và Standalone Script preprocessing
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import glob
from pathlib import Path

# Import preprocessing functions từ main file
from utils.datasets import letterbox

class ResizeMethodComparator:
    """So sánh các phương pháp resize preprocessing"""
    
    def __init__(self, target_size=960):
        self.target_size = target_size
        
    def pad_to_square_top_left(self, img):
        """API Framework approach - pad to square (inherited from main code)"""
        if isinstance(img, Image.Image):
            img = np.array(img)
        h, w, c = img.shape
        new_size = max(h, w)
        padded_img = np.zeros((new_size, new_size, c), dtype=img.dtype)
        padded_img[:h, :w, :] = img
        return padded_img
    
    def letterbox_gray_pad(self, img, new_shape=None, color=(114, 114, 114)):
        """Standalone Script approach - letterbox with gray padding (inherited from main code)"""
        if new_shape is None:
            new_shape = (self.target_size, self.target_size)
            
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)
    
    def preprocess_api_approach(self, img_path):
        """API Framework preprocessing workflow"""
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        original_shape = img_array.shape[:2]
        
        # Step 1: Pad to square (top-left)
        squared_img = self.pad_to_square_top_left(img_array)
        
        # Step 2: Letterbox to model input size
        transformed_img = letterbox(squared_img, (self.target_size, self.target_size), stride=32, auto=False)[0]
        
        return {
            'original': img_array,
            'original_shape': original_shape,
            'step1_squared': squared_img,
            'final_resized': transformed_img,
            'method': 'API Framework',
            'steps': [
                f"Original: {original_shape[1]}x{original_shape[0]} (WxH)",
                f"Step 1: Pad to square: {squared_img.shape[1]}x{squared_img.shape[0]}",
                f"Step 2: Letterbox to {self.target_size}x{self.target_size}: {transformed_img.shape[1]}x{transformed_img.shape[0]}"
            ]
        }
    
    def preprocess_script_approach(self, img_path):
        """Standalone Script preprocessing workflow"""
        # Load image
        img0 = cv2.imread(img_path)
        img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        original_shape = img0_rgb.shape[:2]
        
        # Letterbox with gray padding
        img, ratio, pad = self.letterbox_gray_pad(img0_rgb, self.target_size)
        
        return {
            'original': img0_rgb,
            'original_shape': original_shape,
            'final_resized': img,
            'ratio': ratio,
            'pad': pad,
            'method': 'Standalone Script',
            'steps': [
                f"Original: {original_shape[1]}x{original_shape[0]} (WxH)",
                f"Letterbox to {self.target_size}x{self.target_size}: {img.shape[1]}x{img.shape[0]}",
                f"Scale ratio: {ratio[0]:.3f}x{ratio[1]:.3f}",
                f"Padding: {pad[0]:.1f}x{pad[1]:.1f} (left/right x top/bottom)"
            ]
        }
    
    def compare_single_image(self, img_path, save_path=None):
        """So sánh một ảnh với cả 2 phương pháp"""
        print(f"🔍 Processing: {os.path.basename(img_path)}")
        
        try:
            # Process with both methods
            api_result = self.preprocess_api_approach(img_path)
            script_result = self.preprocess_script_approach(img_path)
            
            # Create comparison visualization
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(3, 4, hspace=0.3, wspace=0.2)
            
            # Original image (shared)
            ax_orig = fig.add_subplot(gs[0, 1:3])
            ax_orig.imshow(api_result['original'])
            ax_orig.set_title(f"Original Image\n{api_result['original_shape'][1]}x{api_result['original_shape'][0]} (WxH)", 
                            fontsize=14, fontweight='bold')
            ax_orig.axis('off')
            
            # API Framework results
            ax_api_step1 = fig.add_subplot(gs[1, 0])
            ax_api_step1.imshow(api_result['step1_squared'])
            ax_api_step1.set_title("API: Step 1\nPad to Square", fontsize=12, color='blue')
            ax_api_step1.axis('off')
            
            ax_api_final = fig.add_subplot(gs[1, 1])
            ax_api_final.imshow(api_result['final_resized'])
            ax_api_final.set_title(f"API: Final Result\n{self.target_size}x{self.target_size}", fontsize=12, color='blue')
            ax_api_final.axis('off')
            
            # Script approach results
            ax_script_final = fig.add_subplot(gs[1, 2])
            ax_script_final.imshow(script_result['final_resized'])
            ax_script_final.set_title(f"Script: Final Result\n{self.target_size}x{self.target_size}", fontsize=12, color='green')
            ax_script_final.axis('off')
            
            # Difference visualization
            ax_diff = fig.add_subplot(gs[1, 3])
            # Calculate difference between final results
            api_final_gray = cv2.cvtColor(api_result['final_resized'], cv2.COLOR_RGB2GRAY)
            script_final_gray = cv2.cvtColor(script_result['final_resized'], cv2.COLOR_RGB2GRAY)
            diff = np.abs(api_final_gray.astype(np.float32) - script_final_gray.astype(np.float32))
            ax_diff.imshow(diff, cmap='hot')
            ax_diff.set_title("Difference Map\n(API vs Script)", fontsize=12, color='red')
            ax_diff.axis('off')
            
            # Add text information
            ax_text = fig.add_subplot(gs[2, :])
            ax_text.axis('off')
            
            info_text = f"""
🔵 API Framework Method:
{chr(10).join(['   ' + step for step in api_result['steps']])}

🟢 Standalone Script Method:
{chr(10).join(['   ' + step for step in script_result['steps']])}

📊 Analysis:
   • API method: Pads to square first, then letterbox → maintains aspect ratio after squaring
   • Script method: Direct letterbox with gray padding → maintains original aspect ratio
   • Difference: API method may introduce more padding for non-square images
   • Use case: API for accuracy, Script for speed and aspect ratio preservation
            """
            
            ax_text.text(0.05, 0.95, info_text, transform=ax_text.transAxes, fontsize=11,
                        verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(f"Resize Method Comparison: {os.path.basename(img_path)}", 
                        fontsize=16, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"✅ Saved comparison: {save_path}")
            
            plt.show()
            
            return {
                'api_result': api_result,
                'script_result': script_result,
                'difference_stats': {
                    'mean_diff': np.mean(diff),
                    'max_diff': np.max(diff),
                    'std_diff': np.std(diff)
                }
            }
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            return None
    
    def compare_multiple_images(self, input_folder, output_folder, max_images=5):
        """So sánh nhiều ảnh và lưu kết quả"""
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
            image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
        image_paths = sorted(image_paths)[:max_images]  # Limit number of images
        
        if not image_paths:
            print(f"❌ No images found in {input_folder}")
            return
        
        print(f"🔍 Found {len(image_paths)} images to process")
        results = []
        
        for i, img_path in enumerate(image_paths):
            print(f"\n📷 Processing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            save_path = os.path.join(output_folder, f"comparison_{i+1:02d}_{os.path.basename(img_path)}.png")
            result = self.compare_single_image(img_path, save_path)
            
            if result:
                results.append({
                    'image_path': img_path,
                    'result': result
                })
        
        # Generate summary
        self.generate_summary_report(results, output_folder)
        
        return results
    
    def generate_summary_report(self, results, output_folder):
        """Tạo báo cáo tổng kết"""
        if not results:
            return
        
        report_path = os.path.join(output_folder, "resize_comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESIZE METHOD COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Target size: {self.target_size}x{self.target_size}\n")
            f.write(f"Total images processed: {len(results)}\n\n")
            
            # Statistics
            all_diffs = []
            original_shapes = []
            
            for result_data in results:
                result = result_data['result']
                img_path = result_data['image_path']
                
                f.write(f"📷 {os.path.basename(img_path)}\n")
                f.write(f"   Original shape: {result['api_result']['original_shape']}\n")
                f.write(f"   API steps: {len(result['api_result']['steps'])}\n")
                f.write(f"   Script steps: {len(result['script_result']['steps'])}\n")
                f.write(f"   Mean difference: {result['difference_stats']['mean_diff']:.2f}\n")
                f.write(f"   Max difference: {result['difference_stats']['max_diff']:.2f}\n\n")
                
                all_diffs.append(result['difference_stats']['mean_diff'])
                original_shapes.append(result['api_result']['original_shape'])
            
            # Overall statistics
            f.write("📊 OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average difference between methods: {np.mean(all_diffs):.2f}\n")
            f.write(f"Max difference observed: {max([r['result']['difference_stats']['max_diff'] for r in results]):.2f}\n")
            f.write(f"Min difference observed: {min([r['result']['difference_stats']['mean_diff'] for r in results]):.2f}\n\n")
            
            # Shape analysis
            f.write("📐 INPUT SHAPE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            unique_shapes = list(set([f"{s[1]}x{s[0]}" for s in original_shapes]))
            for shape in unique_shapes:
                count = len([s for s in original_shapes if f"{s[1]}x{s[0]}" == shape])
                f.write(f"{shape}: {count} image(s)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n")
            f.write("🔵 API Framework Method:\n")
            f.write("   • Better for square or near-square images\n")
            f.write("   • More padding for rectangular images\n")
            f.write("   • Good for accuracy-focused applications\n\n")
            f.write("🟢 Standalone Script Method:\n")
            f.write("   • Better aspect ratio preservation\n")
            f.write("   • More efficient for rectangular images\n")
            f.write("   • Good for speed-focused applications\n")
        
        print(f"📄 Report saved: {report_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compare resize methods from dual preprocessing')
    parser.add_argument('--input', type=str, required=True,
                      help='Input image file or folder')
    parser.add_argument('--output', type=str, required=True,
                      help='Output folder for comparisons')
    parser.add_argument('--target-size', type=int, default=960,
                      help='Target size for resize (default: 960)')
    parser.add_argument('--max-images', type=int, default=5,
                      help='Maximum number of images to process (for folders)')
    parser.add_argument('--single-image', action='store_true',
                      help='Process as single image (show interactive plot)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Input not found: {args.input}")
        return
    
    # Create comparator
    comparator = ResizeMethodComparator(target_size=args.target_size)
    
    print(f"🔄 Resize Method Comparison")
    print(f"📐 Target size: {args.target_size}x{args.target_size}")
    print(f"📂 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    
    if os.path.isfile(args.input):
        # Single image
        print(f"🖼️ Processing single image")
        save_path = None if args.single_image else os.path.join(args.output, f"comparison_{os.path.basename(args.input)}.png")
        if save_path:
            os.makedirs(args.output, exist_ok=True)
        
        result = comparator.compare_single_image(args.input, save_path)
        if result:
            print("\n✅ Comparison completed!")
            print(f"📊 Difference stats:")
            print(f"   Mean: {result['difference_stats']['mean_diff']:.2f}")
            print(f"   Max: {result['difference_stats']['max_diff']:.2f}")
            print(f"   Std: {result['difference_stats']['std_diff']:.2f}")
    
    elif os.path.isdir(args.input):
        # Multiple images
        print(f"📁 Processing folder (max {args.max_images} images)")
        results = comparator.compare_multiple_images(args.input, args.output, args.max_images)
        print(f"\n✅ Processed {len(results)} images!")
    
    else:
        print(f"❌ Invalid input: {args.input}")

if __name__ == "__main__":
    main()

"""
🚀 Cách sử dụng:

# 1. So sánh một ảnh cụ thể (hiển thị interactive):
python compare_resize_methods.py \
    --input "/path/to/single_image.jpg" \
    --output "/path/to/output" \
    --target-size 960 \
    --single-image

# 2. So sánh một ảnh và lưu kết quả:
python compare_resize_methods.py \
    --input "/path/to/single_image.jpg" \
    --output "/path/to/output" \
    --target-size 960

# 3. So sánh nhiều ảnh trong folder:
python compare_resize_methods.py \
    --input "/path/to/image_folder" \
    --output "/path/to/output" \
    --target-size 960 \
    --max-images 10

# 4. So sánh với target size khác (ví dụ 640):
python compare_resize_methods.py \
    --input "/path/to/images" \
    --output "/path/to/output" \
    --target-size 640

# 5. Ví dụ cụ thể với folder của bạn:
python compare_resize_methods.py \
    --input "/home/dainguyenvan/project/ARV/auto-footage/analyst_result_from_footage copy/need_to_improve_when_remove_person_2904/selected_frames" \
    --output "./resize_comparison_results" \
    --target-size 960 \
    --max-images 5

📊 Kết quả sẽ bao gồm:
• Visualization cho từng ảnh showing original, intermediate steps, và final results
• Difference map giữa 2 phương pháp
• Text analysis về các steps và thông số
• Summary report với statistics tổng quát
• Recommendations về khi nào nên dùng method nào

🎯 Ứng dụng:
• Hiểu rõ sự khác biệt giữa 2 phương pháp preprocessing
• Chọn method phù hợp cho dataset cụ thể  
• Debug preprocessing pipeline
• Optimize cho different aspect ratios
• Analyze padding và scaling effects

Example với ảnh 1290x1080:
🔵 API Framework: 1290x1080 → 1290x1290 (pad) → 960x960 (letterbox)
🟢 Script: 1290x1080 → 960x960 (direct letterbox với gray padding)
"""
