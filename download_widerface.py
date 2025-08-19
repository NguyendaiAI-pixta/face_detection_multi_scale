#!/usr/bin/env python3
"""
Download WiderFace dataset from Hugging Face
Usage: python download_widerface.py --output_dir ./data/widerface
"""

import argparse
import os
import sys
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm


def download_file(url, local_filename):
    """Download file with progress bar"""
    print(f"Downloading {url}...")
    
    # Stream download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_filename, 'wb') as f, tqdm(
        desc=str(local_filename),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Downloaded: {local_filename}")
    return local_filename


def extract_zip(zip_path, extract_to):
    """Extract zip file with progress bar"""
    print(f"Extracting {zip_path} to {extract_to}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in zip
        file_list = zip_ref.namelist()
        
        # Extract with progress bar
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, extract_to)
    
    print(f"Extraction completed to: {extract_to}")


def main():
    parser = argparse.ArgumentParser(description='Download WiderFace dataset')
    parser.add_argument('--output_dir', type=str, default='./data/widerface', 
                       help='Output directory to save dataset (default: ./data/widerface)')
    parser.add_argument('--keep_zip', action='store_true', 
                       help='Keep downloaded zip files after extraction')
    parser.add_argument('--download_all', action='store_true',
                       help='Download train, val, and test sets (default: only train)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}")
    
    # URLs for WiderFace dataset
    urls = {
        'train': 'https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_train.zip',
        'val': 'https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip',
        'test': 'https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_test.zip',
        'annot': 'https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip'
    }
    
    # Determine which datasets to download
    if args.download_all:
        datasets_to_download = ['train', 'val', 'test', 'annot']
    else:
        datasets_to_download = ['train', 'annot']  # Always need annotations
    
    print(f"Will download: {datasets_to_download}")
    
    for dataset in datasets_to_download:
        url = urls[dataset]
        if dataset == 'annot':
            filename = "wider_face_split.zip"
            extracted_folder = output_dir / "wider_face_split"
        else:
            filename = f"WIDER_{dataset}.zip"
            extracted_folder = output_dir / f"WIDER_{dataset}"
        
        local_path = output_dir / filename
        
        try:
            # Check if folder already exists
            if extracted_folder.exists():
                print(f"📁 Folder already exists, skipping: {extracted_folder}")
                print(f"✅ {dataset.upper()} dataset ready!")
                continue
            
            # Download if not exists
            if not local_path.exists():
                download_file(url, local_path)
            else:
                print(f"File already exists: {local_path}")
            
            # Extract zip file
            extract_zip(local_path, output_dir)
            
            # Remove zip file if not keeping
            if not args.keep_zip:
                print(f"Removing zip file: {local_path}")
                local_path.unlink()
            
            print(f"✅ {dataset.upper()} dataset ready!")
            
        except Exception as e:
            print(f"❌ Error downloading {dataset}: {str(e)}")
            sys.exit(1)
    
    print("\n🎉 Download completed!")
    print(f"Dataset location: {output_dir.absolute()}")
    
    # Show directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(str(output_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")


if __name__ == "__main__":
    main()
'''
python download_widerface.py --output_dir ./data/widerface --download_all
'''