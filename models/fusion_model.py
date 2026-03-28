"""
models/fusion_model.py
Multimodal Fusion Model: Visual features (from YOLO) + Numerical features (OHLCV)
→ Volatility Regime Classifier (Low / Medium / High)

Architecture:
  [Visual Branch]    YOLO detection features → FC layers
  [Numerical Branch] OHLCV features → FC layers
  [Fusion]           Concatenate → FC → Softmax (3 classes)

This is the core contribution beyond Vijayababu & Bennur (2023)
who only used visual features. We add numerical fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────
# ARCHITECTURE
# ──────────────────────────────────────────────

class VisualBranch(nn.Module):
    """Processes visual features extracted from YOLO detections."""
    def __init__(self, visual_input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class NumericalBranch(nn.Module):
    """Processes OHLCV technical indicator features."""
    def __init__(self, numerical_input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(numerical_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    """
    Multimodal fusion: visual + numerical → volatility regime (3 classes).
    
    Args:
        visual_input_dim   : dimension of visual feature vector
        numerical_input_dim: dimension of OHLCV feature vector
        hidden_dim         : hidden size per branch (default 128)
        n_classes          : number of output classes (default 3: low/med/high vol)
        dropout            : dropout rate in fusion layers
    """
    def __init__(
        self,
        visual_input_dim: int,
        numerical_input_dim: int,
        hidden_dim: int = 128,
        n_classes: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.visual_branch    = VisualBranch(visual_input_dim, hidden_dim)
        self.numerical_branch = NumericalBranch(numerical_input_dim, hidden_dim)

        fusion_input_dim = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, visual_feat, numerical_feat):
        v = self.visual_branch(visual_feat)
        n = self.numerical_branch(numerical_feat)
        fused = torch.cat([v, n], dim=1)
        return self.fusion(fused)

    def predict_proba(self, visual_feat, numerical_feat):
        self.eval()
        with torch.no_grad():
            logits = self.forward(visual_feat, numerical_feat)
            return F.softmax(logits, dim=1)


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────

class FusionDataset(Dataset):
    """
    Dataset of (visual_features, numerical_features, label) triples.
    Features are precomputed and stored as numpy arrays.
    """
    def __init__(self, visual_feats, numerical_feats, labels):
        assert len(visual_feats) == len(numerical_feats) == len(labels)
        self.visual     = torch.tensor(visual_feats,    dtype=torch.float32)
        self.numerical  = torch.tensor(numerical_feats, dtype=torch.float32)
        self.labels     = torch.tensor(labels,          dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.visual[idx], self.numerical[idx], self.labels[idx]


# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────

def train_fusion_model(
    model: FusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 40,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: str = "./best_fusion_model.pt",
) -> dict:
    """
    Train the fusion model.
    Returns dict with training history.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for vis, num, labels in train_loader:
            vis, num, labels = vis.to(device), num.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(vis, num)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for vis, num, labels in val_loader:
                vis, num, labels = vis.to(device), num.to(device), labels.to(device)
                logits  = model(vis, num)
                loss    = criterion(logits, labels)
                val_loss  += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        val_acc   = correct    / total

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(val_acc)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss={avg_train:.4f} | "
                  f"val_loss={avg_val:.4f} | "
                  f"val_acc={val_acc:.4f}")

    print(f"\n[✓] Best val accuracy: {best_val_acc:.4f} — model saved to {save_path}")
    return history


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    VISUAL_DIM  = 64    # e.g. 20 pattern classes * 3 (conf, vol_score, bias_enc) + padding
    NUM_DIM     = 50    # OHLCV feature count

    model = FusionModel(
        visual_input_dim=VISUAL_DIM,
        numerical_input_dim=NUM_DIM,
    )
    print(f"[*] FusionModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy forward pass
    vis = torch.randn(4, VISUAL_DIM)
    num = torch.randn(4, NUM_DIM)
    out = model(vis, num)
    print(f"[*] Output shape: {out.shape}")   # should be (4, 3)
    print("[✓] FusionModel OK")
