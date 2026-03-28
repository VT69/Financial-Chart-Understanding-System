# 📈 Financial Chart Understanding System
### Candlestick Pattern Detection + Multimodal OHLCV Fusion

A computer vision project that detects candlestick patterns from chart images using YOLOv8,
then fuses visual features with numerical OHLCV data for volatility regime prediction.

---

## 🗂 Project Structure

```
chart_cv_project/
├── README.md
├── requirements.txt
├── 1_download_dataset.py       # Download Ahihi dataset from Roboflow
├── 2_train_yolo.py             # Train YOLOv8 on candlestick patterns
├── 3_extract_features.py       # Extract YOLO features + OHLCV features
├── 4_train_fusion_model.py     # Train multimodal fusion classifier
├── 5_evaluate.py               # Full evaluation + metrics report
├── 6_streamlit_app.py          # Interactive demo app
├── utils/
│   ├── ohlcv_features.py       # OHLCV feature engineering
│   ├── pattern_mapper.py       # Pattern → volatility signal mapping
│   └── visualizer.py           # Chart annotation utilities
├── models/
│   └── fusion_model.py         # Multimodal fusion architecture (PyTorch)
└── data/                       # Dataset will be downloaded here
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 🚀 Run in Order

```bash
# Step 1: Download dataset (requires free Roboflow account)
python 1_download_dataset.py

# Step 2: Train YOLOv8 baseline
python 2_train_yolo.py

# Step 3: Extract features for fusion model
python 3_extract_features.py

# Step 4: Train multimodal fusion model
python 4_train_fusion_model.py

# Step 5: Evaluate everything
python 5_evaluate.py

# Step 6: Launch Streamlit app
streamlit run 6_streamlit_app.py
```

---

## 📚 References

- Vijayababu & Bennur (2023) — ComplexCandlestickModel, Yeshiva University
- Jung-Hua Liu (2025) — CV for Cryptocurrency Trading, Medium
- Ahihi Dataset — Roboflow Universe (DuoKan, 2023)
- Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics
