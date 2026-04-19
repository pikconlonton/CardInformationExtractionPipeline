import os
import cv2
from ultralytics import YOLO
from pathlib import Path

def run_inference():
    # --- CẤU HÌNH ---
    model_path = 'best3.pt'  # Đường dẫn model của bạn
    input_path = 'YOUR_PATH'  # Thư mục ảnh đầu vào
    output_dir = 'inference_results' # Thư mục lưu kết quả
    
    conf_threshold = 0.25  # Ngưỡng tin cậy
    save_crops = True      # Có cắt ảnh vùng detect hay không?
    # ----------------

    # 1. Load model
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file model tại {model_path}")
        return
    
    model = YOLO(model_path)

    # 2. Chạy dự đoán
    # stream=True giúp tiết kiệm RAM khi xử lý số lượng ảnh lớn
    results = model.predict(
        source=input_path,
        conf=conf_threshold,
        save=True,           # Lưu ảnh đã vẽ bounding box
        save_crop=save_crops, # Lưu ảnh đã cắt vào thư mục riêng
        project=output_dir,  # Thư mục gốc để lưu
        name='exp',          # Tên thư mục con (ví dụ: inference_results/exp)
        exist_ok=True,       # Ghi đè nếu thư mục đã tồn tại
        line_width=2,
        show=False           # Tắt hiển thị để chạy nhanh hơn trên server/folder lớn
    )

    print("-" * 30)
    print(f"Xử lý hoàn tất!")
    
    # 3. Duyệt qua kết quả để in thông báo (Tùy chọn)
    total_boxes = 0
    for i, r in enumerate(results):
        num_boxes = len(r.boxes)
        total_boxes += num_boxes
        # r.path là đường dẫn ảnh đang xử lý
        print(f"Ảnh {i+1}: {os.path.basename(r.path)} -> Tìm thấy {num_boxes} vùng.")

    print("-" * 30)
    print(f"Tổng cộng: Đã xử lý {len(list(results))} ảnh.")
    print(f"Tổng số vùng chữ tìm thấy: {total_boxes}")
    print(f"Kết quả lưu tại: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    run_inference()