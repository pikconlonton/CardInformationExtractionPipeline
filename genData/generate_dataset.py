#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese OCR Dataset Generator for PARSeq
============================================
Generates synthetic text images with:
  - Balanced character distribution (inverse-frequency sampling)
  - Multiple background types: solid, gradient, noise, blotchy
  - High contrast text/background (WCAG-based)
  - Multiple font families & sizes
  - Optional train/val/test split

Usage:
    python generate_dataset.py --num-samples 10000 --output-dir dataset
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

# ─── optional progress bar ────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ══════════════════════════════════════════════════════════════════════════════
# CHARACTER SET
# ══════════════════════════════════════════════════════════════════════════════

_CHARSET_RAW = (
    "AÁÀẠẢÃĂẮẰẲẶẴÂẤẦẨẪẬBCDĐ"
    "EÈÉẼẺẸÊẾỀỂỄỆFGHIÍÌỈĨỊJKLMN"
    "OÒÓỎÕỌÔỐỒỔỖỘƠỚỜỠỞỢPQRST"
    "UÙÚỦŨỤƯỪỨỬỮỰVXYỴÝỲỶỸWZ"
    "aáàạảãăằắẳẵặâấầẩẫậbcdđ"
    "eèéẻẽẹêếềểễệfghiíìỉĩịjklmn"
    "oòóỏõọôốồổỗộơớờởỡợpqrst"
    "uùúủũụưừứửữựvyỳýỷỹỵxwz"
    '0123456789-",;?/.@#!%^&*()'
)

# Deduplicate, preserve order
_seen: set = set()
_deduped: list = []
for _c in _CHARSET_RAW:
    if _c not in _seen:
        _seen.add(_c)
        _deduped.append(_c)

CHARSET      = "".join(_deduped)
CHARSET_LIST = list(CHARSET)
CHARSET_SIZE = len(CHARSET_LIST)

print(f"[INFO] Charset size: {CHARSET_SIZE} unique characters")

# ══════════════════════════════════════════════════════════════════════════════
# FONT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

# Common Windows system fonts that support Vietnamese
_WIN_SYSTEM_FONTS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/ariali.ttf",
    "C:/Windows/Fonts/arialbi.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "C:/Windows/Fonts/timesi.ttf",
    "C:/Windows/Fonts/timesbi.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/calibril.ttf",
    "C:/Windows/Fonts/calibrii.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/segoeuii.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/verdanai.ttf",
    "C:/Windows/Fonts/verdanaz.ttf",
    "C:/Windows/Fonts/georgia.ttf",
    "C:/Windows/Fonts/georgiai.ttf",
    "C:/Windows/Fonts/georgiab.ttf",
    "C:/Windows/Fonts/georgiaz.ttf",
    "C:/Windows/Fonts/trebuc.ttf",
    "C:/Windows/Fonts/trebucbd.ttf",
    "C:/Windows/Fonts/trebucbi.ttf",
    "C:/Windows/Fonts/trebucit.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/courbd.ttf",
    "C:/Windows/Fonts/couri.ttf",
]

_FONT_SIZES = (18, 22, 26, 30, 34, 38, 42)

def load_fonts(fonts_dir: Path) -> list:
    """
    Load PIL ImageFont objects từ thư mục fonts (và các thư mục con).
    Không lấy font hệ thống.
    """
    font_files = []
    if fonts_dir.exists():
        font_files += list(fonts_dir.glob("**/*.ttf"))
        font_files += list(fonts_dir.glob("**/*.otf"))
    fonts = []
    for fp in font_files:
        for size in _FONT_SIZES:
            try:
                f = ImageFont.truetype(str(fp), size)
                fonts.append(f)
            except Exception:
                pass
    if not fonts:
        print("[WARN] Không tìm thấy font nào, sẽ dùng font mặc định.")
        fonts = [ImageFont.load_default()]
    else:
        print(f"[INFO] Loaded {len(fonts)} font variants from {len(font_files)} font files")
    return fonts

# def load_fonts(fonts_dir: Path) -> list:
#     """
#     Load PIL ImageFont objects từ thư mục fonts hoặc danh sách font hệ thống Windows.
#     """
#     font_files = []
    
#     # 1. Kiểm tra và lấy font từ thư mục người dùng chỉ định (tham số --fonts-dir)
#     if fonts_dir.exists():
#         font_files += list(fonts_dir.glob("**/*.ttf"))
#         font_files += list(fonts_dir.glob("**/*.otf"))

#     # 2. Nếu thư mục trên trống, tự động nạp font từ danh sách _WIN_SYSTEM_FONTS
#     if not font_files:
#         print("[INFO] Không tìm thấy font trong thư mục chỉ định. Đang nạp font hệ thống Windows...")
#         for path_str in _WIN_SYSTEM_FONTS:
#             path_obj = Path(path_str)
#             if path_obj.exists():
#                 font_files.append(path_obj)

#     fonts = []
#     for fp in font_files:
#         for size in _FONT_SIZES:
#             try:
#                 # Load font với kích thước tương ứng
#                 f = ImageFont.truetype(str(fp), size)
#                 fonts.append(f)
#             except Exception:
#                 pass

#     if not fonts:
#         print("[WARN] Không tìm thấy font nào khả dụng, sẽ dùng font mặc định (không hỗ trợ tiếng Việt tốt).")
#         fonts = [ImageFont.load_default()]
#     else:
#         # Loại bỏ các đường dẫn trùng lặp để đếm chính xác số file
#         unique_files = set(font_files)
#         print(f"[INFO] Đã nạp {len(fonts)} biến thể font từ {len(unique_files)} tệp font khác nhau.")
        
#     return fonts

# ══════════════════════════════════════════════════════════════════════════════
# COLOR UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _linearize(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: tuple) -> float:
    """WCAG 2.1 relative luminance, rgb values in [0, 255]."""
    r, g, b = rgb[:3]
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def contrast_ratio(c1: tuple, c2: tuple) -> float:
    """WCAG 2.1 contrast ratio (1:1 to 21:1)."""
    l1, l2 = relative_luminance(c1), relative_luminance(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def rand_rgb() -> tuple:
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def choose_text_color(bg: tuple, min_cr: float = 4.5, attempts: int = 300) -> tuple:
    """
    Pick a random text color whose contrast ratio with bg >= min_cr.
    Biases toward dark or light depending on background luminance.
    """
    bg_lum = relative_luminance(bg)
    for _ in range(attempts):
        h = random.random()
        s = random.uniform(0.15, 1.0)
        # Bias: dark text on light bg, light text on dark bg
        if bg_lum > 0.35:
            v = random.uniform(0.0, 0.45)
        else:
            v = random.uniform(0.55, 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color = (int(r * 255), int(g * 255), int(b * 255))
        if contrast_ratio(bg, color) >= min_cr:
            return color
    # Guaranteed fallback
    return (15, 15, 15) if bg_lum > 0.35 else (240, 240, 240)


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND GENERATORS (each returns (np.ndarray HxWx3 uint8, avg_rgb tuple))
# ══════════════════════════════════════════════════════════════════════════════

def _avg(arr: np.ndarray) -> tuple:
    """Mean color of an RGB array."""
    m = arr.reshape(-1, 3).mean(axis=0)
    return (int(m[0]), int(m[1]), int(m[2]))


def bg_solid(w: int, h: int):
    color = rand_rgb()
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    return arr, color


def bg_gradient_linear(w: int, h: int):
    c1 = np.array(rand_rgb(), dtype=float)
    c2 = np.array(rand_rgb(), dtype=float)
    direction = random.choice(("h", "v", "d"))

    if direction == "h":
        t = np.linspace(0, 1, w).reshape(1, w, 1)
        arr = np.broadcast_to(c1 * (1 - t) + c2 * t, (h, w, 3)).copy()
    elif direction == "v":
        t = np.linspace(0, 1, h).reshape(h, 1, 1)
        arr = np.broadcast_to(c1 * (1 - t) + c2 * t, (h, w, 3)).copy()
    else:  # diagonal
        tx = np.linspace(0, 1, w).reshape(1, w, 1)
        ty = np.linspace(0, 1, h).reshape(h, 1, 1)
        t  = (tx + ty) / 2.0
        arr = (c1 * (1 - t) + c2 * t)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr, _avg(arr)


def bg_gradient_radial(w: int, h: int):
    c1 = np.array(rand_rgb(), dtype=float)
    c2 = np.array(rand_rgb(), dtype=float)
    cx = random.uniform(0.2, 0.8) * w
    cy = random.uniform(0.2, 0.8) * h

    xs = np.arange(w).reshape(1, w)
    ys = np.arange(h).reshape(h, 1)
    dist = np.sqrt(((xs - cx) / w) ** 2 + ((ys - cy) / h) ** 2)
    t = np.clip(dist / dist.max(), 0, 1).reshape(h, w, 1)
    arr = np.clip(c1 * (1 - t) + c2 * t, 0, 255).astype(np.uint8)
    return arr, _avg(arr)


def bg_noise(w: int, h: int):
    base = np.array(rand_rgb(), dtype=float)
    noise = np.random.normal(0, random.uniform(15, 35), (h, w, 3))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return arr, (int(base[0]), int(base[1]), int(base[2]))


def bg_gradient_noise(w: int, h: int):
    arr, avg = bg_gradient_linear(w, h)
    noise = np.random.normal(0, 15, arr.shape)
    arr = np.clip(arr.astype(float) + noise, 0, 255).astype(np.uint8)
    return arr, avg


def bg_blotchy(w: int, h: int):
    """Random elliptical color patches on a base background."""
    base = np.array(rand_rgb(), dtype=float)
    arr  = np.full((h, w, 3), base, dtype=float)
    ys, xs = np.ogrid[:h, :w]

    for _ in range(random.randint(3, 7)):
        color = np.array(rand_rgb(), dtype=float)
        cx = random.uniform(0, w)
        cy = random.uniform(0, h)
        rx = random.uniform(w * 0.25, w * 1.2)
        ry = random.uniform(h * 0.3, h * 2.0)
        mask = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1
        alpha = random.uniform(0.2, 0.55)
        arr[mask] = arr[mask] * (1 - alpha) + color * alpha

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr, _avg(arr)


def bg_paper(w: int, h: int):
    """Paper-like texture: warm tint + fine noise."""
    hue = random.uniform(0.07, 0.14)           # yellowish
    sat = random.uniform(0.05, 0.25)
    val = random.uniform(0.78, 0.95)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    base = np.array([r * 255, g * 255, b * 255], dtype=float)
    noise = np.random.normal(0, 8, (h, w, 3))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return arr, (int(base[0]), int(base[1]), int(base[2]))


_BG_GENERATORS = [
    bg_solid,
    bg_gradient_linear,
    bg_gradient_radial,
    bg_noise,
    bg_gradient_noise,
    bg_blotchy,
    bg_paper,
]
_BG_WEIGHTS = [0.15, 0.20, 0.10, 0.15, 0.15, 0.15, 0.10]


def generate_background(w: int, h: int):
    gen = random.choices(_BG_GENERATORS, weights=_BG_WEIGHTS, k=1)[0]
    return gen(w, h)


# ══════════════════════════════════════════════════════════════════════════════
# TEXT GENERATION — BALANCED DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_text(char_counts: dict, min_len: int = 3, max_len: int = 35) -> str:
    """
    Produce a random string of length in [min_len, max_len].

    Each character is sampled with probability proportional to
    1 / (global_count + 1), so under-represented characters are
    preferred, naturally flattening the distribution over many
    generated samples.

    char_counts is mutated in place (global counter shared across calls).
    """
    length = random.randint(min_len, max_len)
    text: list[str] = []

    for _ in range(length):
        weights = [1.0 / (char_counts[c] + 1.0) for c in CHARSET_LIST]
        chosen  = random.choices(CHARSET_LIST, weights=weights, k=1)[0]
        text.append(chosen)
        char_counts[chosen] += 1

    s = "".join(text)
    # Chèn dấu cách vào giữa các đoạn liên tiếp 3-6 ký tự không phải dấu cách
    chars = list(s)
    i = 0
    while i < len(chars):
        # Bỏ qua nếu là dấu cách
        if chars[i] == " ":
            i += 1
            continue
        # Xác định đoạn liên tiếp không có dấu cách
        j = i
        while j < len(chars) and chars[j] != " ":
            j += 1
        seg_len = j - i
        if seg_len >= 3:
            # Chia nhỏ đoạn dài thành các đoạn 3-6 ký tự
            k = i
            while k + 3 <= j:
                chunk_len = min(random.randint(3, 6), j - k)
                # Không chèn ở đầu hoặc cuối chuỗi
                if chunk_len > 1 and k > 0 and (k + chunk_len) < len(chars):
                    insert_pos = k + chunk_len // 2
                    chars.insert(insert_pos, " ")
                    j += 1
                    k = insert_pos + 2  # Bỏ qua sau dấu cách vừa chèn
                else:
                    k += chunk_len
        i = j + 1
    s = "".join(chars)
    return s.strip()


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _get_text_bbox(font, text: str):
    """Return (text_w, text_h, offset_x, offset_y) compatible with Pillow ≥9."""
    try:
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top, left, top
    except AttributeError:
        w, h = font.getsize(text)  # type: ignore[attr-defined]
        return w, h, 0, 0


def render_sample(text: str, fonts: list, target_height: int = 64) -> Image.Image:
    """
    Render *text* onto a synthetic background image.

    Visual variations applied randomly:
      - background type (7 styles)
      - font family & size
      - text color (guaranteed high contrast)
      - drop shadow (25 % chance)
      - stroke / outline  (15 % chance)
      - slight Gaussian blur (10 % chance)
    """
    font = random.choice(fonts)

    text_w, text_h, off_x, off_y = _get_text_bbox(font, text)

    # Padding
    pad_x = random.randint(6, 24)
    pad_y = random.randint(4, 14)

    width  = max(32, text_w + pad_x * 2)
    height = target_height  # fixed height; text will be vertically centred

    # Background
    bg_arr, avg_bg = generate_background(width, height)

    # Text colour
    text_color = choose_text_color(avg_bg)

    img  = Image.fromarray(bg_arr, "RGB")
    draw = ImageDraw.Draw(img)

    # Vertical centre
    x = pad_x - off_x
    y = (height - text_h) // 2 - off_y

    # ── Drop shadow ──────────────────────────────────────────
    if random.random() < 0.25:
        sh_off = random.randint(1, 2)
        lum = relative_luminance(text_color)
        shadow = tuple(max(0, c - 90) for c in text_color) if lum > 0.5 \
                 else tuple(min(255, c + 90) for c in text_color)
        draw.text((x + sh_off, y + sh_off), text, font=font, fill=shadow)

    # ── Stroke / outline ─────────────────────────────────────
    if random.random() < 0.15:
        lum = relative_luminance(text_color)
        outline = tuple(max(0, c - 120) for c in text_color) if lum > 0.5 \
                  else tuple(min(255, c + 120) for c in text_color)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline)

    # ── Main text ────────────────────────────────────────────
    draw.text((x, y), text, font=font, fill=text_color)

    # ── Mild Gaussian blur ───────────────────────────────────
    if random.random() < 0.10:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.8)))

    return img


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Vietnamese OCR dataset generator for PARSeq",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output-dir",   "-o", default="dataset",
                   help="Root output directory (default: dataset)")
    p.add_argument("--fonts-dir",    "-f", default="fonts",
                   help="Directory with custom Vietnamese TTF/OTF fonts (default: fonts)")
    p.add_argument("--num-samples",  "-n", type=int, default=250000,
                   help="Total number of images to generate (default: 10000)")
    p.add_argument("--min-len",            type=int, default=3,
                   help="Minimum characters per sample (default: 3)")
    p.add_argument("--max-len",      "-l", type=int, default=30,
                   help="Maximum characters per sample (default: 35)")
    p.add_argument("--img-height",         type=int, default=64,
                   help="Image height in pixels (default: 64)")
    p.add_argument("--train-ratio",        type=float, default=0.8,
                   help="Fraction for training split (default: 0.8)")
    p.add_argument("--val-ratio",          type=float, default=0.1,
                   help="Fraction for validation split (default: 0.1)")
    p.add_argument("--seed",               type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Directories ──────────────────────────────────────────
    output_dir = Path(args.output_dir)
    fonts_dir = Path(r"C:\OCR\genData\fonts")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── Fonts ────────────────────────────────────────────────
    print("\n── Loading fonts ──────────────────────────────────────────────")
    fonts = load_fonts(fonts_dir)

    # ── Generation ───────────────────────────────────────────
    print(f"\n── Generating {args.num_samples:,} samples ────────────────────────────────")

    char_counts: dict = defaultdict(int)
    all_labels: list  = []              # list of (rel_path, text)

    iterator = range(args.num_samples)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Generating", unit="img", dynamic_ncols=True)

    for i in iterator:
        text  = generate_text(char_counts, args.min_len, args.max_len)
        img   = render_sample(text, fonts, target_height=args.img_height)

        fname = f"{i:08d}.png"
        img.save(images_dir / fname, optimize=True)
        all_labels.append((f"images/{fname}", text))

    # ── Shuffle & split ──────────────────────────────────────
    random.shuffle(all_labels)
    n       = len(all_labels)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)

    splits = {
        "train": all_labels[:n_train],
        "val":   all_labels[n_train : n_train + n_val],
        "test":  all_labels[n_train + n_val :],
    }

    print("\n── Writing label files ─────────────────────────────────────────")
    for split_name, data in splits.items():
        out_file = output_dir / f"{split_name}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for path, text in data:
                f.write(f"{path}\t{text}\n")
        print(f"  {split_name:5s}: {len(data):6,} samples  →  {out_file}")

    # ── Charset file ─────────────────────────────────────────
    charset_file = output_dir / "charset.txt"
    with open(charset_file, "w", encoding="utf-8") as f:
        f.write(CHARSET)
    print(f"\n  Charset ({CHARSET_SIZE} chars)  →  {charset_file}")

    # ── Statistics ───────────────────────────────────────────
    counts_sorted = dict(sorted(char_counts.items(), key=lambda kv: kv[1]))
    total_tokens  = sum(char_counts.values())
    min_c  = min(char_counts.values()) if char_counts else 0
    max_c  = max(char_counts.values()) if char_counts else 0
    mean_c = total_tokens / max(len(char_counts), 1)
    cv     = (np.std(list(char_counts.values())) / mean_c * 100) if mean_c else 0

    stats = {
        "num_samples":        n,
        "total_tokens":       total_tokens,
        "charset_size":       CHARSET_SIZE,
        "charset":            CHARSET,
        "distribution_stats": {
            "min_count":  min_c,
            "max_count":  max_c,
            "mean_count": round(mean_c, 2),
            "cv_percent": round(float(cv), 2),   # coefficient of variation (lower = more uniform)
        },
        "char_distribution":  counts_sorted,
    }
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ── Summary ──────────────────────────────────────────────
    print("\n══ Summary ═══════════════════════════════════════════════════════")
    print(f"  Total samples  : {n:,}")
    print(f"  Total tokens   : {total_tokens:,}")
    print(f"  Unique chars   : {len(char_counts)}/{CHARSET_SIZE}")
    print(f"  Count range    : [{min_c}, {max_c}]")
    print(f"  Mean / char    : {mean_c:.1f}")
    print(f"  CV (uniformity): {cv:.1f}%  (lower = more uniform)")
    print(f"  Image height   : {args.img_height}px")
    print(f"  Output dir     : {output_dir.resolve()}")
    print(f"  Stats file     : {stats_file}")
    print("\n✔  Dataset generation complete!\n")


if __name__ == "__main__":
    main()
