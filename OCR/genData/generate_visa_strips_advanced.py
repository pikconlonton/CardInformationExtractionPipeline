#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Visa Card Text Strip Generator for OCR Training
======================================================
Generates synthetic text images with realistic Visa card effects:
  - Base gradient & Metal texture (Layers 1 & 2)
  - Specular highlight & Edge glow (Layers 3 & 4)
  - Chip noise (Layer 5)
  - Emboss effect on text (Layer 6)
  - Balanced character distribution (inherited from original script)

Usage:
    python genData\generate_visa_strips_advanced.py --num-samples 1000 --output-dir dataset_visa_advanced
"""

import os
import re
import sys
import json
import math
import random
import argparse
import colorsys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ══════════════════════════════════════════════════════════════════════════════
# CHARACTER SET & FONTS (Kế thừa từ file cũ)
# ══════════════════════════════════════════════════════════════════════════════

_CHARSET_RAW = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
)

# Deduplicate, preserve order
_seen = set()
_deduped = []
for _c in _CHARSET_RAW:
    if _c not in _seen:
        _seen.add(_c)
        _deduped.append(_c)

CHARSET      = "".join(_deduped)
CHARSET_LIST = list(CHARSET)
CHARSET_SIZE = len(CHARSET_LIST)

_FONT_SIZES = (22, 26, 30, 34, 38)

def load_fonts(fonts_dir: Path) -> list:
    font_files = []
    if fonts_dir.exists():
        font_files += list(fonts_dir.glob("**/*.ttf"))
        font_files += list(fonts_dir.glob("**/*.otf"))

    font_groups = defaultdict(list)
    for fp in font_files:
        for size in _FONT_SIZES:
            try:
                f = ImageFont.truetype(str(fp), size)
                font_groups[str(fp)].append(f)
            except Exception:
                pass

    if not font_groups:
        print("[WARN] Không tìm thấy font trong thư mục chỉ định, dùng font mặc định.")
        return [[ImageFont.load_default()]]
    return list(font_groups.values())

# ══════════════════════════════════════════════════════════════════════════════
# COLOR & CONTRAST UTILITIES (Kế thừa và nâng cấp)
# ══════════════════════════════════════════════════════════════════════════════

def _linearize(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def relative_luminance(rgb: tuple) -> float:
    r, g, b = rgb[:3]
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)

def contrast_ratio(c1: tuple, c2: tuple) -> float:
    l1, l2 = relative_luminance(c1), relative_luminance(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

FOIL_COLORS = {
    "gold": (212, 175, 55),
    "silver": (192, 192, 192),
    "white": (245, 245, 245),
    "black": (25, 25, 25),
    "blue_foil": (0, 102, 204)
}

def choose_foil_color(bg_avg_rgb: tuple) -> tuple:
    """Chọn ngẫu nhiên một màu foil có độ tương phản hợp lệ."""
    valid_colors = []
    for name, color in FOIL_COLORS.items():
        cr = contrast_ratio(bg_avg_rgb, color)
        if cr >= 3.0: # Chấp nhận độ tương phản từ 3.0 trở lên
            valid_colors.append(color)
            
    if valid_colors:
        return random.choice(valid_colors)
        
    # Nếu không có màu nào đạt chuẩn, lấy màu có tương phản cao nhất
    best_color = FOIL_COLORS["white"]
    best_cr = 0
    for name, color in FOIL_COLORS.items():
        cr = contrast_ratio(bg_avg_rgb, color)
        if cr > best_cr:
            best_cr = cr
            best_color = color
    return best_color

# ══════════════════════════════════════════════════════════════════════════════
# NEW GRAPHICS LAYERS (Xử lý 7 Layer)
# ══════════════════════════════════════════════════════════════════════════════

def create_base_gradient(w: int, h: int) -> np.ndarray:
    """Layer 1: Base gradient (màu nền vàng/bạc/đen...)"""
    # Chọn ngẫu nhiên kiểu nền
    style = random.choice(["gold", "silver", "black", "blue", "custom"])
    
    if style == "gold":
        c1 = np.array([218, 165, 32]) # Goldenrod
        c2 = np.array([255, 215, 0])  # Gold
    elif style == "silver":
        c1 = np.array([169, 169, 169]) # Dark Gray
        c2 = np.array([220, 220, 220]) # Gainsboro
    elif style == "black":
        c1 = np.array([20, 20, 20])
        c2 = np.array([50, 50, 50])
    elif style == "blue":
        c1 = np.array([0, 32, 96])
        c2 = np.array([0, 112, 192])
    else:
        c1 = np.array([random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)])
        c2 = c1 + 50
        
    t = np.linspace(0, 1, w).reshape(1, w, 1)
    arr = np.broadcast_to(c1 * (1 - t) + c2 * t, (h, w, 3)).copy()
    return arr.astype(np.uint8)

def apply_metal_texture(arr: np.ndarray) -> np.ndarray:
    """Layer 2: Metal texture (vân xước)"""
    h, w, _ = arr.shape
    
    # Random mức độ nhiễu (noise scale) từ 0 đến 15
    noise_scale = random.uniform(0, 15)
    noise_small = np.random.normal(0, noise_scale, (h, w // 4, 1))
    
    # Squeeze axis=2 để từ (H, W, 1) về (H, W) giúp PIL xử lý được dạng grayscale ('L')
    tiled_noise = np.tile(noise_small, (1, 4, 1))
    noise_stretched = Image.fromarray(np.squeeze(tiled_noise, axis=2).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    noise_arr = np.array(noise_stretched).astype(float) - 128
    
    # Random hệ số trộn (blend factor) nhẹ hơn từ 0.1 đến 0.5
    blend_factor = random.uniform(0.1, 0.5)
    
    arr = np.clip(arr.astype(float) + noise_arr[..., np.newaxis] * blend_factor, 0, 255)
    return arr.astype(np.uint8)

def apply_specular_highlight(img: Image.Image) -> Image.Image:
    """Layer 3: Specular highlight (vệt sáng chạy chéo)"""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Vẽ một dải đa giác chéo màu trắng mờ
    cw = random.randint(20, 50)
    start_x = random.randint(-w, w)
    
    # Đa giác chéo
    poly = [
        (start_x, 0),
        (start_x + cw, 0),
        (start_x + cw + h, h),
        (start_x + h, h)
    ]
    draw.polygon(poly, fill=(255, 255, 255, random.randint(30, 80)))
    
    # Làm mờ vệt sáng để trông tự nhiên
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=5))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")



def draw_emboss_text(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font, text_color: tuple, is_emboss: bool):
    """Layer 6: Emboss effect trên chữ"""
    if not is_emboss:
        draw.text((x, y), text, font=font, fill=text_color)
        return
        
    # Mô phỏng dập nổi bằng cách vẽ lệch bóng và highlight
    sh_off = 1
    
    # 1. Vẽ bóng tối (Shadow) lệch xuống dưới bên phải
    shadow_color = (0, 0, 0, 180)
    draw.text((x + sh_off, y + sh_off), text, font=font, fill=shadow_color)
    
    # 2. Vẽ viền sáng (Highlight) lệch lên trên bên trái
    highlight_color = (255, 255, 255, 180)
    draw.text((x - sh_off, y - sh_off), text, font=font, fill=highlight_color)
    
    # 3. Vẽ chữ chính đè lên
    draw.text((x, y), text, font=font, fill=text_color)

# ══════════════════════════════════════════════════════════════════════════════
# TEXT GENERATION (Kế thừa từ file cũ)
# ══════════════════════════════════════════════════════════════════════════════

def generate_text(char_counts: dict, min_len: int = 4, max_len: int = 16) -> str:
    length = random.randint(min_len, max_len)
    text = []
    for _ in range(length):
        weights = [1.0 / (char_counts[c] + 1.0) for c in CHARSET_LIST]
        chosen = random.choices(CHARSET_LIST, weights=weights, k=1)[0]
        text.append(chosen)
        char_counts[chosen] += 1
    return "".join(text)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER & LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _get_text_bbox(font, text: str):
    try:
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top, left, top
    except AttributeError:
        w, h = font.getsize(text)
        return w, h, 0, 0

def render_sample(text: str, fonts: list, target_height: int = 64) -> Image.Image:
    font_group = random.choice(fonts)
    font = random.choice(font_group)
    
    text_w, text_h, off_x, off_y = _get_text_bbox(font, text)
    
    pad_x = 20
    width = max(100, text_w + pad_x * 2)
    height = target_height
    
    # Layer 1 & 2: Base + Metal
    bg_arr = create_base_gradient(width, height)
    bg_arr = apply_metal_texture(bg_arr)
    
    avg_rgb = tuple(bg_arr.mean(axis=(0, 1)).astype(int))
    text_color = choose_foil_color(avg_rgb)
    
    img = Image.fromarray(bg_arr)
    
    # Layer 3: Specular Highlight
    img = apply_specular_highlight(img)
    
    
    
    # Chuẩn bị vẽ chữ
    img_rgba = img.convert("RGBA")
    text_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    
    x = pad_x - off_x
    y = (height - text_h) // 2 - off_y
    
    is_emboss = random.random() > 0.5 # 50% dập nổi
    
    # Layer 6: Emboss Text
    draw_emboss_text(draw, x, y, text, font, text_color, is_emboss)
    
    img = Image.alpha_composite(img_rgba, text_layer).convert("RGB")
    
    # Layer 7: Vignette (Shadow tổng thể)
    w, h = img.size
    # Tạo mask vignette bằng numpy để làm tối nhẹ các góc
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    dist = np.sqrt(X**2 + Y**2)
    vignette = np.clip(1 - dist * 0.2, 0, 1) # 0.2 quyết định độ tối ở rìa
    
    arr = np.array(img).astype(float)
    arr = arr * vignette[..., np.newaxis]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    
    return img

def main():
    parser = argparse.ArgumentParser(description="Advanced Visa Card Text Strip Generator")
    parser.add_argument("--output-dir", "-o", default="dataset_visa_advanced", help="Output directory")
    parser.add_argument("--num-samples", "-n", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val ratio")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    fonts_dir = Path(r"C:\OCR\OCR\genData\fontVisa")
    fonts = load_fonts(fonts_dir)
    
    char_counts = defaultdict(int)
    all_labels = []
    
    print(f"[INFO] Generating {args.num_samples} samples...")
    
    iterator = range(args.num_samples)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Generating")
        
    for i in iterator:
        text = generate_text(char_counts)
        img = render_sample(text, fonts)
        
        fname = f"{i:08d}.png"
        img.save(images_dir / fname)
        all_labels.append((f"images/{fname}", text))
        
    # ── Shuffle & split ──────────────────────────────────────
    random.shuffle(all_labels)
    n = len(all_labels)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    splits = {
        "train": all_labels[:n_train],
        "val":   all_labels[n_train : n_train + n_val],
        "test":  all_labels[n_train + n_val :],
    }

    print("\n-- Writing label files -----------------------------------------")
    for split_name, data in splits.items():
        out_file = output_dir / f"{split_name}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for path, text in data:
                f.write(f"{path}\t{text}\n")
        print(f"  {split_name:5s}: {len(data):6,} samples  ->  {out_file}")

    # ── Charset file ─────────────────────────────────────────
    charset_file = output_dir / "charset.txt"
    with open(charset_file, "w", encoding="utf-8") as f:
        f.write(CHARSET)
    print(f"\n  Charset ({CHARSET_SIZE} chars)  ->  {charset_file}")

    # ── Stats file ───────────────────────────────────────────
    stats_file = output_dir / "stats.json"
    stats = {
        "total_samples": n,
        "splits": {name: len(data) for name, data in splits.items()},
        "char_counts": dict(char_counts),
        "charset_size": CHARSET_SIZE,
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f"  Stats file  ->  {stats_file}")
            
    print(f"\n[INFO] Done! Dataset saved to {output_dir}")

if __name__ == "__main__":
    main()
