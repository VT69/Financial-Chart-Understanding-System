"""
Step 4: Train the Multimodal Fusion Model
Combines visual (YOLO) + numerical (OHLCV) features
→ Volatility regime classification (Low / Med / High)
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from models.fusion_model import FusionModel, FusionDataset, train_fusion_model

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FEATURES_DIR  = "./data/features"
OUTPUT_DIR    = "./outputs"
EPOCHS        = 40
BATCH_SIZE    = 32
LR            = 1e-3
HIDDEN_DIM    = 128
DROPOUT       = 0.4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE    = "./outputs/best_fusion_model.pt"


def load_features():
    """Load all split features, combine train+valid for training."""
    meta_path = os.path.join(FEATURES_DIR, "meta.json")
    if not os.path.exists(meta_path):
        print("[ERROR] meta.json not found. Run 3_extract_features.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"[*] Feature dimensions: visual={meta['visual_dim']}, numerical={meta['numerical_dim']}")
    print(f"[*] Classes: {meta['n_classes']}")

    # Load splits
    splits = {}
    for split in ["train", "valid", "test"]:
        vpath = os.path.join(FEATURES_DIR, f"{split}_visual.npy")
        npath = os.path.join(FEATURES_DIR, f"{split}_numerical.npy")
        lpath = os.path.join(FEATURES_DIR, f"{split}_labels.npy")
        if os.path.exists(vpath):
            splits[split] = {
                "visual":    np.load(vpath),
                "numerical": np.load(npath),
                "labels":    np.load(lpath),
            }
            print(f"    {split}: {splits[split]['visual'].shape[0]} samples")

    return splits, meta


def plot_training_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#1f77b4")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#ff7f0e")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_acc"], label="Val Accuracy", color="#2ca02c")
    axes[1].axhline(y=0.9151, color="#d62728", linestyle="--",
                    label="Baseline (Vijayababu & Bennur, 2023)")
    axes[1].set_title("Validation Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Multimodal Fusion Model Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Training plot saved: {save_path}")


def evaluate_model(model, test_loader, device):
    """Full evaluation with per-class accuracy."""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for vis, num, labels in test_loader:
            vis, num = vis.to(device), num.to(device)
            logits   = model(vis, num)
            preds    = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n" + "="*60)
    print("  Test Set Evaluation — Multimodal Fusion Model")
    print("="*60)
    report = classification_report(
        all_labels, all_preds,
        target_names=["Low Vol", "Med Vol", "High Vol"],
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Med", "High"],
                yticklabels=["Low", "Med", "High"], ax=ax)
    ax.set_title("Confusion Matrix — Fusion Model")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Confusion matrix saved: {cm_path}")

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return acc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load features
    splits, meta = load_features()
    VISUAL_DIM   = meta["visual_dim"]
    NUM_DIM      = meta["numerical_dim"]

    # Combine train + valid for training
    train_vis = np.vstack([splits["train"]["visual"],    splits.get("valid", splits["train"])["visual"]])
    train_num = np.vstack([splits["train"]["numerical"], splits.get("valid", splits["train"])["numerical"]])
    train_lbl = np.hstack([splits["train"]["labels"],    splits.get("valid", splits["train"])["labels"]])

    # Split off a val set
    tv, vv, tn, vn, tl, vl = train_test_split(
        train_vis, train_num, train_lbl, test_size=0.15, random_state=42, stratify=train_lbl
    )

    train_ds = FusionDataset(tv, tn, tl)
    val_ds   = FusionDataset(vv, vn, vl)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[*] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"[*] Device: {DEVICE}\n")

    # Build model
    model = FusionModel(
        visual_input_dim=VISUAL_DIM,
        numerical_input_dim=NUM_DIM,
        hidden_dim=HIDDEN_DIM,
        n_classes=3,
        dropout=DROPOUT,
    )
    print(f"[*] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n[*] Training fusion model...")
    history = train_fusion_model(
        model, train_dl, val_dl,
        epochs=EPOCHS, lr=LR, device=DEVICE,
        save_path=MODEL_SAVE,
    )

    # Plot history
    plot_training_history(history, os.path.join(OUTPUT_DIR, "training_history.png"))

    # Test evaluation
    if "test" in splits:
        test_ds  = FusionDataset(splits["test"]["visual"], splits["test"]["numerical"], splits["test"]["labels"])
        test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
        test_acc = evaluate_model(model, test_dl, DEVICE)
        print(f"\n[✓] Final Test Accuracy: {test_acc:.4f}")

        # Compare to baseline
        baseline = 0.9151
        print(f"    Baseline (Vijayababu & Bennur 2023): {baseline:.4f}")
        delta = test_acc - baseline
        status = "▲ IMPROVED" if delta > 0 else "▼ below baseline"
        print(f"    Delta: {delta:+.4f}  {status}")

    # Save model metadata
    model_meta = {
        "visual_dim": VISUAL_DIM,
        "numerical_dim": NUM_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_classes": 3,
        "class_names": ["Low Volatility", "Medium Volatility", "High Volatility"],
    }
    with open(os.path.join(OUTPUT_DIR, "fusion_model_meta.json"), "w") as f:
        json.dump(model_meta, f, indent=2)

    print(f"\n[✓] All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
    print("Next step → run: python 5_evaluate.py")
