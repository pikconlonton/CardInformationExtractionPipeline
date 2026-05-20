import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from paddleocr import PaddleOCR
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import glob

# 1. Khởi tạo mô hình mới
print("[THÔNG TIN] Đang khởi tạo mô hình mới...")
ocr_new = PaddleOCR(
    rec_model_dir=r'C:\OCR\OCR\test\new_inference_model',
    rec_char_dict_path=r'C:\OCR\OCR\test\dict.txt',
    use_angle_cls=False,
    use_gpu=True, 
    lang='vi'
)

img_path = r"C:\Users\duchb\OneDrive\Pictures\Screenshots\Screenshot 2026-05-20 220100.png"
# Danh sách các đường dẫn để thử đọc, theo thứ tự ưu tiên
paths_to_try = [img_path]

# 1. Nếu ảnh mặc định không tồn tại, tìm ảnh mới nhất trong Screenshots
if not os.path.exists(img_path):
    print(f"[CẢNH BÁO] Không tìm thấy ảnh mặc định tại: {img_path}")
    screenshot_dir = r"C:\Users\duchb\OneDrive\Pictures\Screenshots"
    if os.path.exists(screenshot_dir):
        files = glob.glob(os.path.join(screenshot_dir, "*.png")) + glob.glob(os.path.join(screenshot_dir, "*.jpg"))
        if files:
            # Sắp xếp theo thời gian sửa đổi mới nhất
            files.sort(key=os.path.getmtime, reverse=True)
            paths_to_try.append(files[0])
            print(f"[THÔNG TIN] Thêm ảnh mới nhất trong Screenshots vào danh sách thử: {files[0]}")

# 2. Thêm ảnh mẫu dự phòng trong dự án vào danh sách thử
fallback_project_img = os.path.join("check_fonts_preview", "CourierPrime-Regular_digits.png")
paths_to_try.append(fallback_project_img)

image = None
# Thử đọc ảnh từ danh sách
for path in paths_to_try:
    if os.path.exists(path):
        print(f"[THÔNG TIN] Đang thử đọc ảnh: {path}")
        image = cv2.imread(path)
        if image is not None:
            img_path = path
            print(f"[OK] Đọc ảnh thành công: {path}")
            break
        else:
            print(f"[CẢNH BÁO] OpenCV không thể giải mã/đọc file: {path} (có thể do OneDrive offline hoặc file lỗi)")
    else:
        print(f"[CẢNH BÁO] Đường dẫn không tồn tại: {path}")

if image is None:
    print("[LỖI] Không thể đọc được bất kỳ ảnh nào từ danh sách thử nghiệm!")
    exit(1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Hàm phụ trợ trích xuất kết quả OCR
def extract_ocr_result(result):
    if result and len(result) > 0:
        res = result[0]
        if isinstance(res, list) or isinstance(res, tuple):
            if len(res) > 0:
                if isinstance(res[0], list) or isinstance(res[0], tuple):
                    return res[0][0], res[0][1]
                elif len(res) >= 2:
                    return res[0], res[1]
    return "Không nhận diện được", 0.0

# 2. Chạy nhận diện với mô hình mới (det=False)
print("\n[THÔNG TIN] Đang nhận diện bằng mô hình mới...")
result_new = ocr_new.ocr(img_path, det=False, rec=True)
text_new, score_new = extract_ocr_result(result_new)

# 3. Hiển thị kết quả nhận diện
print("\n" + "="*50)
print("--- KẾT QUẢ NHẬN DIỆN MÔ HÌNH MỚI ---")
print(f" Nội dung: {text_new}")
print(f" Độ tin cậy: {score_new:.2f}")
print("="*50)

# 4. Hiển thị ảnh kèm kết quả
plt.figure(figsize=(6, 4))
plt.imshow(image_rgb)
title_str = f"Mô hình mới: {text_new} ({score_new:.2f})"
plt.title(title_str, fontsize=10, color='blue', loc='left')
plt.axis('off')
plt.tight_layout()
plt.show()