import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def generate_font_previews():
    # 1. Cấu hình đường dẫn
    font_dir = Path(r"C:\OCR\OCR\genData\fontVisa")
    output_dir = Path("check_fonts_preview")
    output_dir.mkdir(exist_ok=True)

    # 2. Lấy danh sách font
    font_files = list(font_dir.glob("**/*.ttf")) + list(font_dir.glob("**/*.otf"))
    
    # 3. Các mẫu nội dung để check
    test_cases = [
        ("digits", "0123 4567 8901 2345"),      # Check số thẻ
        ("uppercase", "NGUYEN VAN ADMIN"),     # Check tên chủ thẻ
        ("full_charset", "abcABC 123 @#%&*")   # Check tổng hợp (dễ lòi ô vuông nhất)
    ]

    print(f"Đang kiểm tra {len(font_files)} fonts...")

    for fp in font_files:
        font_name = fp.stem
        for case_name, text in test_cases:
            try:
                # Load font size 40 để nhìn cho rõ
                font = ImageFont.truetype(str(fp), 40)
                
                # Tính toán kích thước ảnh dựa trên chữ
                left, top, right, bottom = font.getbbox(text)
                w, h = right - left + 40, bottom - top + 40
                
                # Vẽ ảnh: Nền trắng, chữ đen
                img = Image.new("RGB", (w, h), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                draw.text((20 - left, 20 - top), text, font=font, fill=(0, 0, 0))
                
                # Lưu ảnh: TênFont_LoaiCheck.png
                fname = f"{font_name}_{case_name}.png"
                img.save(output_dir / fname)
                
            except Exception as e:
                print(f"Lỗi font {font_name} ở case {case_name}: {e}")

    print(f"\n✔ Xong! Bạn vào thư mục '{output_dir}' để xem ảnh nào có ô vuông nhé.")

if __name__ == "__main__":
    generate_font_previews()