"""
Step 1: Download Candlestick Pattern Detector Dataset from Roboflow
Dataset: 13 candlestick pattern classes
Source: https://universe.roboflow.com/anonimo-3nggp/candlestick-pattern-detector
"""

import io
import os
import sys
import zipfile

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from tqdm import tqdm

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ROBOFLOW_API_KEY = "1ZHyIKUwHvEKC0kSKp6e"   # ← replace with your free key from roboflow.com
WORKSPACE        = "anonimo-3nggp"
PROJECT          = "candlestick-pattern-detector"
VERSION          = 1
FORMAT           = "yolov8"
DATA_DIR         = "./data"

# 13 candlestick pattern class names (candlestick-pattern-detector dataset)
CLASS_NAMES = [
    "Hammer",
    "Bearish Marubozu",
    "Bullish Marubozu",
    "Dragonfly Doji",
    "Four Price Doji",
    "Gravestone Doji",
    "Inverted Hammer",
    "Long-Legged Doji",
    "Morning Star",
    "Shooting Star",
    "Tweezer Bottom",
    "Tweezer Top",
    "Hanging Man",
]

# ──────────────────────────────────────────────
# PATTERN METADATA  (bullish / bearish / neutral)
# Used later by pattern_mapper.py
# ──────────────────────────────────────────────
PATTERN_META = {
    "Hammer":            {"type": "reversal",     "bias": "bullish"},
    "Bearish Marubozu":  {"type": "continuation", "bias": "bearish"},
    "Bullish Marubozu":  {"type": "continuation", "bias": "bullish"},
    "Dragonfly Doji":    {"type": "reversal",     "bias": "bullish"},
    "Four Price Doji":   {"type": "reversal",     "bias": "neutral"},
    "Gravestone Doji":   {"type": "reversal",     "bias": "bearish"},
    "Inverted Hammer":   {"type": "reversal",     "bias": "bullish"},
    "Long-Legged Doji":  {"type": "reversal",     "bias": "neutral"},
    "Morning Star":      {"type": "reversal",     "bias": "bullish"},
    "Shooting Star":     {"type": "reversal",     "bias": "bearish"},
    "Tweezer Bottom":    {"type": "reversal",     "bias": "bullish"},
    "Tweezer Top":       {"type": "reversal",     "bias": "bearish"},
    "Hanging Man":       {"type": "reversal",     "bias": "bearish"},
}


def download_dataset():
    """Download dataset via Roboflow REST API, extract zip, return local path."""
    if ROBOFLOW_API_KEY == "YOUR_ROBOFLOW_API_KEY":
        print("\n[!] You need a free Roboflow API key.")
        print("    1. Go to https://roboflow.com and create a free account.")
        print("    2. Copy your API key from https://app.roboflow.com/settings/api")
        print("    3. Replace 'YOUR_ROBOFLOW_API_KEY' at the top of this file.\n")
        sys.exit(1)

    # ── Step 1: Get the download URL from the Roboflow API ──────────────────
    api_url = (
        f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{VERSION}/{FORMAT}"
        f"?api_key={ROBOFLOW_API_KEY}"
    )
    print(f"[*] Fetching export URL from Roboflow API...")
    resp = requests.get(api_url, timeout=30)
    if resp.status_code != 200:
        print(f"[ERROR] API request failed ({resp.status_code}): {resp.text[:300]}")
        sys.exit(1)

    data = resp.json()
    # The download link lives at data['export']['link'] or data['link']
    download_url = (
        data.get("export", {}).get("link")
        or data.get("link")
    )
    if not download_url:
        print(f"[ERROR] Could not find download link in API response.")
        print(f"        Response keys: {list(data.keys())}")
        print(f"        Full response: {data}")
        sys.exit(1)

    print(f"[✓] Got download link.")

    # ── Step 2: Download the zip with a progress bar ─────────────────────────
    print(f"[*] Downloading zip archive...")
    zip_resp = requests.get(download_url, stream=True, timeout=120)
    zip_resp.raise_for_status()
    total = int(zip_resp.headers.get("content-length", 0))

    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for chunk in zip_resp.iter_content(chunk_size=8192):
            buf.write(chunk)
            pbar.update(len(chunk))

    # ── Step 3: Extract into DATA_DIR ────────────────────────────────────────
    print(f"[*] Extracting to {DATA_DIR} ...")
    os.makedirs(DATA_DIR, exist_ok=True)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(DATA_DIR)

    print(f"\n[✓] Dataset extracted to: {os.path.abspath(DATA_DIR)}")
    print(f"    Classes ({len(CLASS_NAMES)}): {CLASS_NAMES} ")
    return DATA_DIR


def verify_dataset(dataset_path):
    """Check that train/valid/test splits exist and count images."""
    splits = ["train", "valid", "test"]
    print("\n[*] Verifying dataset structure...")
    for split in splits:
        img_dir = os.path.join(dataset_path, split, "images")
        lbl_dir = os.path.join(dataset_path, split, "labels")
        if os.path.exists(img_dir):
            n_imgs = len(os.listdir(img_dir))
            n_lbls = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
            print(f"    {split:10s}: {n_imgs} images, {n_lbls} labels")
        else:
            print(f"    {split:10s}: [NOT FOUND] expected at {img_dir}")
    print("[✓] Verification complete.\n")


if __name__ == "__main__":
    path = download_dataset()
    verify_dataset(path)
    print("Next step → run: python 2_train_yolo.py")
