# 📈 Financial Chart Understanding System
### Candlestick Pattern Detection + Multimodal OHLCV Fusion

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kandls.streamlit.app/)

A computer vision pipeline that detects **13 candlestick chart patterns** from financial chart images using **YOLOv8**, then fuses visual features with numerical OHLCV data for volatility regime prediction.

---

## 🔗 Live Interactive Dashboard
🔥 **Try the application live here:** [kandls.streamlit.app](https://kandls.streamlit.app/)

The web dashboard features a fully transparent **Explainable AI (XAI)** pipeline that breaks down exactly how the model visualizes the chart, maps bounding box anchors to semantic patterns, fuses it with numerical moving averages, and outputs a final volatility regime.

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
├── models/                     # Trained weights saved here 
└── data/                       # Dataset downloaded here 
```

---

## 📊 Results & Performance

Based on our final benchmark runs combining the YOLOv8 visual module and the OHLCV Deep Neural Fusion module:

### YOLOv8 Object Detection (Pattern Localization)
| Metric | Score |
|--------|-------|
| **mAP50** | **~89.4%** |
| Precision | ~88.2% |
| Recall | ~85.1% |

### Multimodal Fusion (Volatility Classification)
| Regime | F1-Score | Accuracy Contribution |
|--------|----------|-----------------------|
| Low Volatility | 0.92 | High |
| Med Volatility | 0.84 | Medium | 
| High Volatility | 0.89 | High |
| **Overall Accuracy** | **~88.5%** | **Robust** |

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

*(Note for Cloud Deployments: To run headless versions of OpenCV on cloud servers like Streamlit Community Cloud, `opencv-python-headless` is utilized in `requirements.txt` to avoid Debian `libGL` system failures).*

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

---

### Step 2 — Train YOLOv8

```bash
python 2_train_yolo.py
```

Trains `yolov8s` for 150 epochs on the downloaded dataset. Training outputs are saved to `./runs/`.

| Setting | Value |
|---------|-------|
| Model | YOLOv8s (pretrained ImageNet) |
| Epochs | 150 (early stopping @ patience=25) |
| Image size | 640×640 |
| Batch size | 16 (reduce to 8 if OOM) |
| Device | Auto (GPU if available, else CPU) |

> ⚠️ **Need a free GPU? Use Google Colab:**
> If your laptop lacks an Nvidia GPU or crashes, open [Google Colab](https://colab.research.google.com/), select a **T4 GPU**, clone this repository in the notebook, and run steps 1, 2, and 3 there. Then simply download the `best.pt` file!

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

Trains a multimodal fusion classifier that combines visual detection outputs with OHLCV features to predict volatility regimes.

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

Opens an interactive Streamlit dashboard for live candlestick detection and visualizer explanations.

---

## 📚 References

- Vijayababu & Bennur (2023) — *ComplexCandlestickModel*, Yeshiva University
- Jung-Hua Liu (2025) — *CV for Cryptocurrency Trading*, Medium
- [Candlestick Pattern Detector Dataset](https://universe.roboflow.com/anonimo-3nggp/candlestick-pattern-detector) — Roboflow Universe (CC BY 4.0)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
