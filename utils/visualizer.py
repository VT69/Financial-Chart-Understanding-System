"""
utils/visualizer.py
Chart annotation and visualization utilities.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path


BIAS_COLORS = {
    "bullish":  (0, 200, 0),    # green  (BGR for cv2)
    "bearish":  (0, 0, 220),    # red
    "neutral":  (180, 180, 0),  # yellow
}

BIAS_COLORS_HEX = {
    "bullish": "#00c800",
    "bearish": "#dc0000",
    "neutral": "#b4b400",
}


def draw_detections_on_image(
    image_path: str,
    detections: list,
    pattern_signals: list,
    save_path: str = None,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a chart image.

    Args:
        image_path     : path to input chart image
        detections     : raw YOLO result boxes (list of dicts with bbox, name, conf)
        pattern_signals: PatternSignal objects from pattern_mapper
        save_path      : if provided, saves annotated image here

    Returns:
        annotated image as numpy array (BGR)
    """
    img = cv2.imread(image_path)
    if img is None:
        img = np.array(Image.open(image_path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    signal_map = {s.name: s for s in pattern_signals}

    for det in detections:
        name = det.get("name", "Unknown")
        conf = det.get("confidence", 0.0)
        bbox = det.get("bbox", None)  # [x1, y1, x2, y2] normalized or pixel

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        # Convert normalized coords to pixels if needed
        if all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        sig   = signal_map.get(name)
        bias  = sig.bias if sig else "neutral"
        color = BIAS_COLORS.get(bias, (180, 180, 180))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{name[:20]} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

    if save_path:
        cv2.imwrite(save_path, img)

    return img


def plot_signal_summary(summary: dict, save_path: str = None):
    """
    Plot a donut chart of bullish/bearish/neutral pattern counts.
    """
    counts = [
        summary.get("bullish_count", 0),
        summary.get("bearish_count", 0),
        summary.get("neutral_count", 0),
    ]
    labels = ["Bullish", "Bearish", "Neutral"]
    colors = [BIAS_COLORS_HEX["bullish"], BIAS_COLORS_HEX["bearish"], BIAS_COLORS_HEX["neutral"]]

    # Only plot non-zero slices
    non_zero = [(c, l, col) for c, l, col in zip(counts, labels, colors) if c > 0]
    if not non_zero:
        return

    counts_nz, labels_nz, colors_nz = zip(*non_zero)

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        counts_nz,
        labels=labels_nz,
        colors=colors_nz,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(width=0.5),
    )
    ax.set_title("Pattern Bias Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_volatility_regime_timeline(
    df,
    ticker: str = "",
    save_path: str = None,
):
    """
    Plot price + volatility regime overlay as a color-coded timeline.
    """
    import pandas as pd

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Price
    axes[0].plot(df.index, df["Close"], color="#1f77b4", linewidth=1)
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"{ticker} Price & Volatility Regime" if ticker else "Price & Volatility Regime")
    axes[0].grid(alpha=0.3)

    # Regime background shading
    if "vol_regime" in df.columns:
        regime_colors = {0: "#00c80033", 1: "#b4b40033", 2: "#dc000033"}
        regime_labels = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}
        prev_regime = None
        start_idx = df.index[0]
        for i, (idx, row) in enumerate(df.iterrows()):
            r = int(row["vol_regime"])
            if r != prev_regime:
                if prev_regime is not None:
                    axes[0].axvspan(start_idx, idx, alpha=0.2,
                                    color=list(regime_colors.values())[prev_regime])
                start_idx   = idx
                prev_regime = r
        # Last segment
        if prev_regime is not None:
            axes[0].axvspan(start_idx, df.index[-1], alpha=0.2,
                            color=list(regime_colors.values())[prev_regime])

        # Regime subplot
        axes[1].fill_between(df.index, df["vol_regime"], alpha=0.5, color="#7f7f7f")
        axes[1].set_ylabel("Regime (0/1/2)")
        axes[1].set_yticks([0, 1, 2])
        axes[1].set_yticklabels(["Low", "Med", "High"])
        axes[1].grid(alpha=0.3)

    patches = [
        mpatches.Patch(color="#00c800", alpha=0.5, label="Low Vol"),
        mpatches.Patch(color="#b4b400", alpha=0.5, label="Med Vol"),
        mpatches.Patch(color="#dc0000", alpha=0.5, label="High Vol"),
    ]
    axes[0].legend(handles=patches, loc="upper left", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("[✓] visualizer.py loaded OK — no direct test (requires image input)")
