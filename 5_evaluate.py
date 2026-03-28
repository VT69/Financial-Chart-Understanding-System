"""
Step 5: Full System Evaluation
Evaluates both:
  1. YOLOv8 detection performance (mAP, per-class AP)
  2. Fusion model classification (accuracy, F1, confusion matrix)
Generates a combined report.
"""

import os
import sys
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from models.fusion_model import FusionModel, FusionDataset

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR      = "./data"
FEATURES_DIR  = "./data/features"
OUTPUT_DIR    = "./outputs"
RUNS_DIR      = "./runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGIME_NAMES  = ["Low Volatility", "Med Volatility", "High Volatility"]
BASELINE_ACC  = 0.9151   # Vijayababu & Bennur 2023


# ──────────────────────────────────────────────
# YOLO EVALUATION
# ──────────────────────────────────────────────

def evaluate_yolo():
    from ultralytics import YOLO

    model_txt = "./best_model_path.txt"
    if not os.path.exists(model_txt):
        print("[!] YOLO model path not found — skipping YOLO evaluation.")
        return None

    with open(model_txt) as f:
        model_path = f.read().strip()

    if not os.path.exists(model_path):
        print(f"[!] YOLO model file not found: {model_path}")
        return None

    # Find YAML
    yamls = list(Path(DATA_DIR).rglob("*patched*.yaml"))
    if not yamls:
        yamls = list(Path(DATA_DIR).rglob("*.yaml"))
    if not yamls:
        print("[!] Dataset YAML not found — skipping YOLO evaluation.")
        return None

    print("[*] Evaluating YOLOv8 on test set...")
    model   = YOLO(model_path)
    metrics = model.val(data=str(yamls[0]), split="test", verbose=False)

    results = {
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }

    print(f"\n{'='*50}")
    print("  YOLOv8 Detection Results (Test Set)")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k:15s}: {v:.4f}")
    print(f"{'='*50}\n")

    return results


# ──────────────────────────────────────────────
# FUSION MODEL EVALUATION
# ──────────────────────────────────────────────

def evaluate_fusion():
    meta_path = os.path.join(FEATURES_DIR, "meta.json")
    model_path = os.path.join(OUTPUT_DIR, "best_fusion_model.pt")

    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        print("[!] Fusion model or features not found — skipping.")
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Load test features
    vpath = os.path.join(FEATURES_DIR, "test_visual.npy")
    npath = os.path.join(FEATURES_DIR, "test_numerical.npy")
    lpath = os.path.join(FEATURES_DIR, "test_labels.npy")

    if not all(os.path.exists(p) for p in [vpath, npath, lpath]):
        print("[!] Test feature files not found — skipping fusion evaluation.")
        return None

    vis_feats = np.load(vpath)
    num_feats = np.load(npath)
    labels    = np.load(lpath)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model    = FusionModel(meta["visual_dim"], meta["numerical_dim"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model    = model.to(device)
    model.eval()

    dataset  = FusionDataset(vis_feats, num_feats, labels)
    loader   = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for vis, num, lbl in loader:
            vis, num = vis.to(device), num.to(device)
            out      = model(vis, num)
            preds    = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.numpy())

    acc   = accuracy_score(all_labels, all_preds)
    f1    = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, target_names=REGIME_NAMES)
    cm    = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*50}")
    print("  Fusion Model Results (Test Set)")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Baseline  : {BASELINE_ACC:.4f}  (Vijayababu & Bennur 2023)")
    delta = acc - BASELINE_ACC
    print(f"  Delta     : {delta:+.4f}  {'▲ IMPROVED' if delta > 0 else '▼ below baseline'}")
    print(f"\n{report}")
    print(f"{'='*50}\n")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Low", "Med", "High"],
        yticklabels=["Low", "Med", "High"],
        ax=ax,
    )
    ax.set_title("Fusion Model — Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fusion_confusion_matrix.png"), dpi=150)
    plt.close()

    return {"accuracy": acc, "f1_macro": f1, "report": report}


# ──────────────────────────────────────────────
# COMPARISON PLOT
# ──────────────────────────────────────────────

def plot_comparison(yolo_results: dict, fusion_results: dict):
    """Bar chart comparing this project vs prior work."""
    methods  = ["VGG16\n(prior)", "ResNet50\n(prior)", "GoogLeNet\n(prior)",
                 "YOLOv8\n(prior)", "ComplexCNN\n(prior)", "Ours\n(Fusion)"]
    # Prior work test accuracies from Vijayababu & Bennur 2023
    accs     = [0.7015, 0.6343, 0.7811, 0.8896, 0.9151, fusion_results.get("accuracy", 0.0)]
    colors   = ["#aec7e8"] * 5 + ["#1f77b4"]

    fig, ax  = plt.subplots(figsize=(10, 5))
    bars     = ax.bar(methods, accs, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=0.9151, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Baseline (ComplexCNN = {0.9151:.4f})")
    ax.set_ylim([0.5, 1.0])
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison: Prior Work vs. Ours (Multimodal Fusion)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Comparison plot saved: {path}")


# ──────────────────────────────────────────────
# SAVE REPORT
# ──────────────────────────────────────────────

def save_report(yolo_results, fusion_results):
    report = {
        "yolo_detection": yolo_results or {},
        "fusion_classification": {
            k: v for k, v in (fusion_results or {}).items() if k != "report"
        },
        "baseline_comparison": {
            "prior_work_best_acc": BASELINE_ACC,
            "our_acc": fusion_results.get("accuracy", None) if fusion_results else None,
            "delta": (fusion_results.get("accuracy", BASELINE_ACC) - BASELINE_ACC)
                      if fusion_results else None,
        },
    }
    path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[✓] Evaluation report saved: {path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Financial Chart CV System — Full Evaluation")
    print("="*60 + "\n")

    yolo_results   = evaluate_yolo()
    fusion_results = evaluate_fusion()

    if fusion_results:
        plot_comparison(yolo_results, fusion_results)

    save_report(yolo_results, fusion_results)

    print("\n[✓] Evaluation complete. Check ./outputs/ for all results.")
    print("Next step → run: streamlit run 6_streamlit_app.py")
