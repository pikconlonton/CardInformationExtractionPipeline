import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_full_log(file_path):
    data = []
    # Regex xịn để hốt sạch thông số của Đức
    pattern = re.compile(
        r"epoch: \[(?P<epoch>\d+)/\d+\], global_step: (?P<step>\d+), lr: (?P<lr>[\d\.]+), "
        r"acc: (?P<acc>[\d\.]+), norm_edit_dis: (?P<edit_dis>[\d\.]+), "
        r"CTCLoss: (?P<ctc>[\d\.]+), NRTRLoss: (?P<nrtr>[\d\.]+), loss: (?P<loss>[\d\.]+)"
    )
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                d = match.groupdict()
                data.append({
                    'epoch': int(d['epoch']),
                    'step': int(d['step']),
                    'lr': float(d['lr']),
                    'acc': float(d['acc']),
                    'edit_dis': float(d['edit_dis']),
                    'ctc': float(d['ctc']),
                    'nrtr': float(d['nrtr']),
                    'loss': float(d['loss'])
                })
    return pd.DataFrame(data)

# Đọc file train.txt
df = parse_full_log(r'C:\OCR\src_train\train.txt')

# Tạo khung hình 4 biểu đồ
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Biểu đồ Loss tổng hợp
axes[0, 0].plot(df['step'], df['loss'], label='Total Loss', color='black', alpha=0.3)
axes[0, 0].plot(df['step'], df['ctc'], label='CTC Loss', color='blue')
axes[0, 0].plot(df['step'], df['nrtr'], label='NRTR Loss', color='orange')
axes[0, 0].set_title('Biểu đồ Loss (Total vs CTC vs NRTR)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Độ chính xác & Edit Distance
axes[0, 1].plot(df['step'], df['acc'], label='Accuracy', color='green')
axes[0, 1].plot(df['step'], df['edit_dis'], label='Edit Distance', color='lime', linestyle='--')
axes[0, 1].set_title('Độ chính xác & Khoảng cách chỉnh sửa')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Learning Rate Schedule
axes[1, 0].plot(df['step'], df['lr'], label='Learning Rate', color='purple')
axes[1, 0].set_title('Chiến lược giảm Learning Rate')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Phân bổ Accuracy theo từng Epoch
df.boxplot(column='acc', by='epoch', ax=axes[1, 1])
axes[1, 1].set_title('Phân bổ Accuracy qua 10 Epoch')

plt.suptitle('PHÂN TÍCH QUÁ TRÌNH HUẤN LUYỆN MODEL OCR ', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()