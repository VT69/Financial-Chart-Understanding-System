"""
Step 3: Extract Visual + Numerical Features for Fusion Model Training
- Runs YOLO inference on all dataset images → visual feature vectors
- Downloads BTC-USD OHLCV data → numerical feature vectors
- Aligns them by index and saves as .npy files
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.ohlcv_features import fetch_ohlcv, add_all_features, label_volatility_regimes, get_feature_columns
from utils.pattern_mapper  import PATTERN_KB, PATTERN_VOL_SCORE, CLASS_NAMES_ORDERED

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR        = "./data"
OUTPUT_DIR      = "./data/features"
YOLO_MODEL_PATH = "./best_model_path.txt"   # written by step 2
TICKER          = "BTC-USD"
OHLCV_PERIOD    = "3y"
VISUAL_DIM      = 64    # fixed visual feature vector size
NUM_DIM         = None  # will be computed from OHLCV features

CLASS_NAMES_ORDERED = [
    "Three Inside Up-Down", "Hikkake Pattern", "Advance Block",
    "Three Outside Up-Down", "Upside-Downside Gap Three Methods",
    "Tasuki Gap", "Evening Star", "Rising-Falling Three Methods",
    "Morning Doji Star", "Morning Star", "Three Black Crows",
    "Three Line Strike", "Evening Doji Star", "Tristar Pattern",
    "Up-Down Gap Side-by-side White Lines", "Stick Sandwich",
    "Ladder Bottom", "Unique 3 River", "Three Advancing White Soldiers",
    "Identical Three Crows",
]
N_CLASSES = len(CLASS_NAMES_ORDERED)

BIAS_ENCODING = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


# ──────────────────────────────────────────────
# VISUAL FEATURE EXTRACTION
# ──────────────────────────────────────────────

def yolo_detections_to_vector(results, n_classes: int = 20) -> np.ndarray:
    """
    Convert YOLO detection results for one image into a fixed-length feature vector.

    Vector layout (per class): [max_confidence, mean_confidence, count, vol_score, bias_enc]
    Total dim = n_classes * 5 (padded/truncated to VISUAL_DIM)
    """
    # Initialize: [max_conf, mean_conf, count, vol_score, bias_enc] per class
    vec = np.zeros(n_classes * 5, dtype=np.float32)

    if results is None or len(results) == 0:
        return vec[:VISUAL_DIM] if len(vec) >= VISUAL_DIM else np.pad(vec, (0, VISUAL_DIM - len(vec)))

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return vec[:VISUAL_DIM] if len(vec) >= VISUAL_DIM else np.pad(vec, (0, VISUAL_DIM - len(vec)))

    class_confs = {i: [] for i in range(n_classes)}
    names_map   = results[0].names  # {0: "ClassName", ...}

    for i in range(len(boxes)):
        cls_idx = int(boxes.cls[i].item())
        conf    = float(boxes.conf[i].item())
        class_confs[cls_idx].append(conf)

    for cls_idx, confs in class_confs.items():
        base = cls_idx * 5
        if cls_idx < n_classes:
            cls_name = names_map.get(cls_idx, "")
            if confs:
                vec[base]     = max(confs)
                vec[base + 1] = np.mean(confs)
                vec[base + 2] = len(confs)
            vec[base + 3] = PATTERN_VOL_SCORE.get(cls_name, 0.5)
            kb = PATTERN_KB.get(cls_name, {})
            vec[base + 4] = BIAS_ENCODING.get(kb.get("bias", "neutral"), 0.0)

    # Truncate or pad to VISUAL_DIM
    if len(vec) >= VISUAL_DIM:
        return vec[:VISUAL_DIM]
    return np.pad(vec, (0, VISUAL_DIM - len(vec)))


def extract_visual_features(model_path: str, split: str = "train") -> tuple:
    """Run YOLO on all images in a dataset split and extract feature vectors."""
    from ultralytics import YOLO

    model   = YOLO(model_path)
    img_dir = Path(DATA_DIR) / split / "images"

    if not img_dir.exists():
        print(f"[!] Image dir not found: {img_dir}")
        return np.array([]), []

    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"[*] Extracting visual features from {len(img_paths)} images ({split})...")

    visual_feats = []
    filenames    = []

    for path in tqdm(img_paths):
        results = model.predict(str(path), verbose=False, conf=0.25)
        feat    = yolo_detections_to_vector(results, n_classes=N_CLASSES)
        visual_feats.append(feat)
        filenames.append(path.name)

    return np.array(visual_feats, dtype=np.float32), filenames


# ──────────────────────────────────────────────
# NUMERICAL FEATURE EXTRACTION
# ──────────────────────────────────────────────

def extract_numerical_features(n_samples: int) -> tuple:
    """
    Download OHLCV data and extract n_samples of numerical feature rows.
    Each row corresponds to one trading day.
    Also returns labels (volatility regimes).
    """
    print(f"[*] Downloading {TICKER} OHLCV data...")
    df = fetch_ohlcv(TICKER, period=OHLCV_PERIOD)
    df = add_all_features(df)
    df = label_volatility_regimes(df)

    feat_cols  = [c for c in get_feature_columns() if c in df.columns]
    feats      = df[feat_cols].values.astype(np.float32)
    labels     = df["vol_regime"].values.astype(np.int64)

    print(f"    OHLCV rows available: {len(feats)}")
    print(f"    Feature columns     : {len(feat_cols)}")

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feats  = scaler.fit_transform(feats).astype(np.float32)

    # Save scaler for inference
    import joblib
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "ohlcv_scaler.pkl"))

    # If we have more OHLCV rows than images, sample to match
    if len(feats) > n_samples:
        idx   = np.linspace(0, len(feats) - 1, n_samples, dtype=int)
        feats  = feats[idx]
        labels = labels[idx]
    elif len(feats) < n_samples:
        # Tile to match
        reps   = (n_samples // len(feats)) + 1
        feats  = np.tile(feats, (reps, 1))[:n_samples]
        labels = np.tile(labels, reps)[:n_samples]

    return feats, labels, feat_cols


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read YOLO model path
    if not os.path.exists(YOLO_MODEL_PATH):
        print("[ERROR] best_model_path.txt not found. Run 2_train_yolo.py first.")
        sys.exit(1)
    with open(YOLO_MODEL_PATH) as f:
        model_path = f.read().strip()
    print(f"[*] Using YOLO model: {model_path}")

    all_visual, all_numerical, all_labels = [], [], []

    for split in ["train", "valid", "test"]:
        print(f"\n── Processing split: {split} ──")
        visual_feats, filenames = extract_visual_features(model_path, split)
        n = len(visual_feats)
        if n == 0:
            print(f"    Skipping {split} (no images found)")
            continue

        num_feats, labels, feat_cols = extract_numerical_features(n)

        all_visual.append(visual_feats)
        all_numerical.append(num_feats)
        all_labels.append(labels)

        # Save per-split
        np.save(f"{OUTPUT_DIR}/{split}_visual.npy",    visual_feats)
        np.save(f"{OUTPUT_DIR}/{split}_numerical.npy", num_feats)
        np.save(f"{OUTPUT_DIR}/{split}_labels.npy",    labels)
        print(f"    Saved: {split}_visual.npy ({visual_feats.shape}), "
              f"{split}_numerical.npy ({num_feats.shape}), "
              f"{split}_labels.npy ({labels.shape})")

    # Save metadata
    meta = {
        "visual_dim":    VISUAL_DIM,
        "numerical_dim": len(feat_cols),
        "n_classes":     3,
        "feat_cols":     feat_cols,
        "ticker":        TICKER,
    }
    with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[✓] Feature extraction complete. Metadata saved to {OUTPUT_DIR}/meta.json")


if __name__ == "__main__":
    main()
    print("Next step → run: python 4_train_fusion_model.py")
