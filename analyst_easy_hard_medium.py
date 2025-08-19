import os
from scipy.io import loadmat

def print_widerface_difficulty_info(gt_dir):
    for diff in ['easy', 'medium', 'hard']:
        mat_path = os.path.join(gt_dir, f'wider_{diff}_val.mat')
        mat = loadmat(mat_path)
        gt_list = mat['gt_list']
        print(f"\nDifficulty: {diff.upper()}")
        print(f"  Số event: {len(gt_list)}")
        total_faces = 0
        for i in range(len(gt_list)):
            for j in range(len(gt_list[i][0])):
                indices = gt_list[i][0][j][0]
                total_faces += len(indices)
        print(f"  Tổng số face {diff}: {total_faces}")

# Ví dụ sử dụng:
gt_dir = '/home/dainguyenvan/project/ARV/auto-footage/yolov7-face/widerface_evaluate/ground_truth'  # Thay bằng đường dẫn thực tế
print_widerface_difficulty_info(gt_dir)
'''
Difficulty: EASY
  Số event: 61
  Tổng số face easy: 7211

Difficulty: MEDIUM
  Số event: 61
  Tổng số face medium: 13319

Difficulty: HARD
  Số event: 61
  Tổng số face hard: 31958
'''