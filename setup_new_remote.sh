#!/bin/bash

# Xóa liên kết đến remote hiện tại
git remote remove origin

# Thêm remote mới (thay YOUR_USERNAME và YOUR_REPO bằng thông tin của bạn)
echo "Vui lòng nhập username GitHub của bạn:"
read github_username
echo "Vui lòng nhập tên repository:"
read repo_name

# Thêm remote mới
git remote add origin https://github.com/$github_username/$repo_name.git

# Thêm tất cả các file đã thay đổi (trừ những file trong .gitignore)
git add .

# Commit các thay đổi
echo "Vui lòng nhập message cho commit:"
read commit_message
git commit -m "$commit_message"

# Push lên remote mới
echo "Đang push lên remote mới..."
git push -u origin main

echo "Hoàn thành! Kiểm tra repository của bạn tại: https://github.com/$github_username/$repo_name"
