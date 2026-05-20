#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automation Script to Generate and Combine OCR Datasets
======================================================
1. Generates 100,000 Vietnamese Italic text samples.
2. Generates 20,000 Visa Card text strip samples (embossed/metallic).
3. Merges and renames images into a single 'images' folder to prevent name collisions.
4. Combines the label files using the new unified image folder.
5. Unifies the character sets into a single dictionary file for PaddleOCR.
6. Cleans up temporary directories.

Usage:
    python genData/generate_and_combine.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def run_command(cmd: list) -> bool:
    print(f"[RUNNING] {' '.join(cmd)}")
    try:
        # Run process and stream output to console
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        rc = process.poll()
        if rc != 0:
            print(f"[ERROR] Command failed with return code {rc}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Failed to run command: {e}")
        return False

def merge_and_move_images(src_file: Path, dest_file: Path, src_img_dir: Path, dest_img_dir: Path, prefix: str):
    """
    Reads lines from src_file.
    For each line, moves the image file to dest_img_dir and renames it with prefix_
    to prevent collisions, then appends the new label path to dest_file.
    """
    if not src_file.exists():
        print(f"[WARN] Source label file {src_file} does not exist. Skipping.")
        return

    with open(src_file, "r", encoding="utf-8") as sf, open(dest_file, "a", encoding="utf-8") as df:
        for line in sf:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                img_path_rel, text = parts
                
                # Extract filename from relative path (e.g. "images/00000000.png" -> "00000000.png")
                filename = Path(img_path_rel).name
                new_filename = f"{prefix}_{filename}"
                
                src_img_path = src_img_dir / filename
                dest_img_path = dest_img_dir / new_filename
                
                # Move image file
                if src_img_path.exists():
                    try:
                        shutil.move(str(src_img_path), str(dest_img_path))
                    except Exception as e:
                        print(f"[WARN] Failed to move {src_img_path} to {dest_img_path}: {e}")
                
                # Write updated label path (e.g. "images/vi_00000000.png")
                df.write(f"images/{new_filename}\t{text}\n")

def main():
    script_dir = Path(__file__).parent.resolve()
    python_exe = sys.executable

    # Define paths
    vi_script = script_dir / "generate_italic_dataset.py"
    visa_script = script_dir / "generate_visa_strips_advanced.py"
    output_dir = script_dir.parent / "dataset_combined"
    dest_img_dir = output_dir / "images"
    
    # Target samples count
    vi_samples = 100000
    visa_samples = 20000

    print("======================================================================")
    print("      STARTING HYBRID OCR DATASET GENERATION & COMBINATOR             ")
    print("======================================================================")
    print(f"Target Directory : {output_dir}")
    print(f"Vietnamese Text  : {vi_samples:,} samples")
    print(f"Visa Card Strips : {visa_samples:,} samples")
    print("----------------------------------------------------------------------")

    # Create target directory structure
    dest_img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Vietnamese Dataset
    vi_out = output_dir / "dataset_vi"
    cmd_vi = [
        python_exe, str(vi_script),
        "--num-samples", str(vi_samples),
        "--output-dir", str(vi_out),
        "--train-ratio", "0.8",
        "--val-ratio", "0.1"
    ]
    if not run_command(cmd_vi):
        sys.exit(1)

    # 2. Generate Visa Dataset
    visa_out = output_dir / "dataset_visa"
    cmd_visa = [
        python_exe, str(visa_script),
        "--num-samples", str(visa_samples),
        "--output-dir", str(visa_out),
        "--train-ratio", "0.8",
        "--val-ratio", "0.1"
    ]
    if not run_command(cmd_visa):
        sys.exit(1)

    # 3. Combine label files and move images
    print("\n----------------------------------------------------------------------")
    print("Merging label files and moving/renaming images...")
    print("----------------------------------------------------------------------")
    
    import random
    splits = ["train", "val", "test"]
    for split in splits:
        combined_file = output_dir / f"{split}.txt"
        with open(combined_file, "w", encoding="utf-8") as f:
            pass # clear contents
            
        # Append from Vietnamese dataset
        merge_and_move_images(
            src_file=vi_out / f"{split}.txt",
            dest_file=combined_file,
            src_img_dir=vi_out / "images",
            dest_img_dir=dest_img_dir,
            prefix="vi"
        )
        # Append from Visa dataset
        merge_and_move_images(
            src_file=visa_out / f"{split}.txt",
            dest_file=combined_file,
            src_img_dir=visa_out / "images",
            dest_img_dir=dest_img_dir,
            prefix="visa"
        )
        
        # Shuffle the combined label file
        try:
            with open(combined_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            lines = [line for line in lines if line.strip()]
            random.seed(42)
            random.shuffle(lines)
            with open(combined_file, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"  Merged & Shuffled {split}.txt  ->  {len(lines):,} total samples")
        except Exception as e:
            print(f"  [WARN] Failed to shuffle {split}.txt: {e}")

    # 4. Generate combined character dictionary
    print("\n----------------------------------------------------------------------")
    print("Generating unified character dictionary for PaddleOCR...")
    print("----------------------------------------------------------------------")
    
    vi_charset_file = vi_out / "charset.txt"
    visa_charset_file = visa_out / "charset.txt"
    
    combined_chars = set()
    
    # Read Vietnamese charset
    if vi_charset_file.exists():
        with open(vi_charset_file, "r", encoding="utf-8") as f:
            combined_chars.update(f.read())
            
    # Read Visa charset
    if visa_charset_file.exists():
        with open(visa_charset_file, "r", encoding="utf-8") as f:
            combined_chars.update(f.read())

    # Filter out whitespace characters if present, as PaddleOCR manages space separately
    combined_chars.discard(" ")
    combined_chars.discard("\t")
    combined_chars.discard("\n")
    combined_chars.discard("\r")
    
    # Sort for deterministic output
    sorted_chars = sorted(list(combined_chars))
    
    # Write flat charset (single-line)
    flat_charset_file = output_dir / "charset_flat.txt"
    with open(flat_charset_file, "w", encoding="utf-8") as f:
        f.write("".join(sorted_chars))
        
    # Write PaddleOCR dictionary (one char per line)
    dict_file = output_dir / "dict.txt"
    with open(dict_file, "w", encoding="utf-8") as f:
        for char in sorted_chars:
            f.write(f"{char}\n")
            
    print(f"  Combined charset size : {len(sorted_chars)} unique characters (excluding spaces)")
    print(f"  PaddleOCR dict file   : {dict_file}")
    print(f"  Flat charset file     : {flat_charset_file}")

    # 5. Clean up temporary directories
    print("\n----------------------------------------------------------------------")
    print("Cleaning up temporary directories...")
    print("----------------------------------------------------------------------")
    try:
        shutil.rmtree(vi_out)
        shutil.rmtree(visa_out)
        print("  Removed temporary directory dataset_vi")
        print("  Removed temporary directory dataset_visa")
    except Exception as e:
        print(f"  [WARN] Failed to clean up: {e}")

    print("\n[SUCCESS] Dataset combining and formatting complete!")
    print(f"To configure PaddleOCR, set rec_char_dict_path to: {dict_file.resolve()}")
    print("======================================================================")

if __name__ == "__main__":
    main()
