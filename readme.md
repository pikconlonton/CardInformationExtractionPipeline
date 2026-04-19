Bước 1: Kích hoạt môi trường ảo (Venv)

PowerShell

# Di chuyển vào folder dự án OCR

cd C:\OCR

# Kích hoạt venv

.\venv\Scripts\Activate.ps1
Bước 2: Cài đặt thư viện

PowerShell
pip install -r requirements.txt

Bước 3: Kiểm tra các đường dẫn

Model: C:\OCR\test\inference_model/ phải chứa các file .pdmodel và .pdiparams.

Từ điển: C:\OCR\test\vietnamese_dict.txt (Dùng để map kết quả nhận diện sang tiếng Việt).

testLocal.py img_path = r"C:\Users\duchb\OneDrive\Pictures\Screenshots\Screenshot 2026-04-18 210518.png" (thay đường dẫn ảnh trong testLocal.py bằng ảnh của bạn muốn)

Bước 4: Chạy file
PowerShell
python test\testLocal.py
