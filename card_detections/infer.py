import os
import cv2
from ultralytics import YOLO

# --- CẤU HÌNH ---
MODEL_PATH = 'models/best.pt'
INPUT_DIR = 'input_images'
OUTPUT_DIR = 'output_crops'
CONF_THRESHOLD = 0.5  # Ngưỡng tin cậy

def run_inference():
    # 1. Khởi tạo model
    model = YOLO(MODEL_PATH)
    
    # 2. Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # 3. Duyệt qua các ảnh trong thư mục đầu vào
    valid_formats = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_formats)]
    
    if not image_files:
        print("Không tìm thấy ảnh hợp lệ trong thư mục đầu vào!")
        return

    print(f"Bắt đầu xử lý {len(image_files)} ảnh...")

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        
        # Thực hiện predict
        results = model.predict(source=img_path, conf=CONF_THRESHOLD, save=False)
        
        for i, r in enumerate(results):
            # Lấy ảnh gốc
            original_img = r.orig_img
            
            # Duyệt qua các box phát hiện được
            for j, box in enumerate(r.boxes):
                # Lấy tọa độ (x1, y1, x2, y2)
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                
                # Crop ảnh theo tọa độ box
                cropped_img = original_img[y1:y2, x1:x2]
                
                # Lưu ảnh đã crop
                output_filename = f"crop_{j}_{img_name}"
                save_path = os.path.join(OUTPUT_DIR, output_filename)
                cv2.imwrite(save_path, cropped_img)
                
                print(f"--> Đã lưu: {save_path}")

if __name__ == "__main__":
    run_inference()