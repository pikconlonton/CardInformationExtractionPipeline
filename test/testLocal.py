from paddleocr import PaddleOCR
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 1. Khởi tạo
ocr = PaddleOCR(
    rec_model_dir=r'C:\OCR\test\inference_model',
    rec_char_dict_path=r'C:\OCR\test\vietnamese_dict.txt',
    use_angle_cls=False,
    use_gpu=True, 
    lang='vi'
)

img_path = r"C:\Users\duchb\OneDrive\Pictures\Screenshots\Screenshot 2026-04-18 020637.png"
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Chạy với det=False
result = ocr.ocr(img_path, det=False, rec=True)

print("\n" + "="*50)
print("--- KẾT QUẢ KIỂM TRA ---")

# 3. Xử lý lỗi IndexError bằng cách kiểm tra kỹ cấu trúc result
if result and len(result) > 0:
    # Với det=False, result thường có dạng [[('TEXT', CONF)], ...] hoặc [('TEXT', CONF)]
    # Đoạn này sẽ giúp lấy đúng dữ liệu dù result ở dạng nào
    res = result[0]
    if isinstance(res, list) or isinstance(res, tuple):
        # Nếu res vẫn là list (thường gặp ở bản 2.7), lấy phần tử đầu tiên
        if isinstance(res[0], list) or isinstance(res[0], tuple):
            final_text = res[0][0]
            final_score = res[0][1]
        else:
            final_text = res[0]
            final_score = res[1]
            
        print(f" Nội dung: {final_text}")
        print(f" Độ tin cậy: {final_score:.2f}")
        print("="*50)

        # 4. Hiển thị ảnh nhỏ (4, 3)
        plt.figure(figsize=(4, 3))
        plt.imshow(image_rgb)
        plt.title(f"Kết quả: {final_text}", fontsize=10, color='red')
        plt.axis('off')
        plt.show()
    else:
        print(" Cấu trúc result lạ: ", result)
else:
    print(" Model không đọc được chữ nào!")