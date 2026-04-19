Card Detection & Auto-Crop Inference
Dự án này sử dụng model YOLOv12n để tự động phát hiện và cắt (crop) các loại thẻ từ hình ảnh thực tế. Ảnh sau khi crop sẽ được chuẩn hóa để làm đầu vào cho các model xử lý tiếp theo (ví dụ: OCR nhận diện thông tin trên thẻ).

📋 Hướng dẫn cài đặt
Trước khi chạy, hãy đảm bảo bạn đã cài đặt các thư viện cần thiết:

Bash
pip install ultralytics opencv-python

🚀 Cách chạy Inference
Chuẩn bị Model: Lưu file weight đã train (ví dụ: best.pt) vào thư mục models/.

Chuẩn bị Dữ liệu: Để các ảnh cần xử lý vào thư mục input_images/.

Thực thi:

Bash
python infer.py
📤 Đầu ra (Output)
Ảnh kết quả sẽ được lưu tại thư mục output_crops/.

Tên file sẽ có định dạng: crop_[index]_[tên_file_gốc].jpg.

⚠️ Lưu ý quan trọng cho Model sau
Đầu vào Model sau: Ảnh đầu ra của file này là vùng chứa thẻ đã được căn chỉnh. Nếu model tiếp theo yêu cầu kích thước cố định (ví dụ 640x640), bạn cần thêm bước cv2.resize() trong file infer.py.

Độ tin cậy: Bạn có thể điều chỉnh biến CONF_THRESHOLD trong mã nguồn để lọc bỏ các kết quả nhận diện sai.

Xử lý nhiều thẻ: Nếu trong một ảnh có nhiều thẻ, hệ thống sẽ lưu mỗi thẻ thành một file ảnh riêng biệt.