# 📈 Financial Chart Understanding System
### Candlestick Pattern Detection + Multimodal OHLCV Fusion

A computer vision pipeline that detects **13 candlestick chart patterns** from financial chart images
using **YOLOv8**, then fuses visual features with numerical OHLCV data for volatility regime prediction.

---

## 🗂 Project Structure

```
CV Project/
├── README.md
├── requirements.txt
├── .gitignore
├── 1_download_dataset.py       # Download candlestick dataset from Roboflow
├── 2_train_yolo.py             # Train YOLOv8 on candlestick patterns
├── 3_extract_features.py       # Extract YOLO features + OHLCV features
├── 4_train_fusion_model.py     # Train multimodal fusion classifier
├── 5_evaluate.py               # Full evaluation + metrics report
├── 6_streamlit_app.py          # Interactive demo app
├── utils/
│   ├── ohlcv_features.py       # OHLCV feature engineering
│   ├── pattern_mapper.py       # Pattern → volatility signal mapping
│   └── visualizer.py           # Chart annotation utilities
├── models/                     # Trained weights saved here (git-ignored)
└── data/                       # Dataset downloaded here (git-ignored)
```

---

## 📦 Dataset

**Source:** [Candlestick Pattern Detector — Roboflow Universe](https://universe.roboflow.com/anonimo-3nggp/candlestick-pattern-detector)

| Split | Images |
|-------|--------|
| Train | 658 |
| Valid | 212 |
| Test  | 103 |
| **Total** | **973** |

**13 Classes:**
`Hammer`, `Bearish Marubozu`, `Bullish Marubozu`, `Dragonfly Doji`, `Four Price Doji`,
`Gravestone Doji`, `Inverted Hammer`, `Long-Legged Doji`, `Shooting Star`,
`Standard Doji`, `Spinning Top`, `High Wave`, `Hanging Man`

> The dataset is **not included in this repo** (git-ignored). Run `1_download_dataset.py` to fetch it automatically.

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/VT69/Financial-Chart-Understanding-System.git
cd Financial-Chart-Understanding-System
```

### 2. Create and activate a conda environment

```bash
conda create -n cv python=3.12 -y
conda activate cv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Pipeline (in order)

### Step 1 — Download Dataset

```bash
python 1_download_dataset.py
```

> **Requires a free Roboflow API key.**
> 1. Create a free account at [roboflow.com](https://roboflow.com)
> 2. Copy your API key from [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api)
> 3. Replace `ROBOFLOW_API_KEY` at the top of `1_download_dataset.py`

Downloads ~973 images and extracts them to `./data/` with the correct `train/valid/test` split structure.

---

### Step 2 — Train YOLOv8

```bash
python 2_train_yolo.py
```

Trains `yolov8s` for 50 epochs on the downloaded dataset. Training outputs are saved to `./runs/`.

| Setting | Value |
|---------|-------|
| Model | YOLOv8s (pretrained ImageNet) |
| Epochs | 50 (early stopping @ patience=10) |
| Image size | 640×640 |
| Batch size | 16 (reduce to 8 if OOM) |
| Device | Auto (GPU if available, else CPU) |

> ⚠️ **Training on CPU takes 2–4 hours.** A CUDA-capable GPU is strongly recommended.

---

### Step 3 — Extract Features

```bash
python 3_extract_features.py
```

Runs the trained YOLOv8 detector on all chart images and extracts:
- YOLO bounding boxes + confidence scores
- OHLCV-derived technical features (via `utils/ohlcv_features.py`)

---

### Step 4 — Train Fusion Model

```bash
python 4_train_fusion_model.py
```

Trains a multimodal fusion classifier that combines visual detection outputs with OHLCV features
to predict volatility regimes.

---

### Step 5 — Evaluate

```bash
python 5_evaluate.py
```

Prints a full metrics report including mAP50, mAP50-95, precision, recall, and fusion accuracy.

---

### Step 6 — Launch Demo App

```bash
streamlit run 6_streamlit_app.py
```

Opens an interactive Streamlit dashboard for live candlestick detection on uploaded chart images.

---

## 📚 References

- Vijayababu & Bennur (2023) — *ComplexCandlestickModel*, Yeshiva University
- Jung-Hua Liu (2025) — *CV for Cryptocurrency Trading*, Medium
- [Candlestick Pattern Detector Dataset](https://universe.roboflow.com/anonimo-3nggp/candlestick-pattern-detector) — Roboflow Universe (CC BY 4.0)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
