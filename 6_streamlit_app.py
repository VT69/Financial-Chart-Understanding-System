"""
Step 6: Streamlit Demo App
Upload a candlestick chart image → detect patterns → show volatility signal
"""

import os
import sys
import json
import tempfile
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from utils.pattern_mapper  import map_detections_to_signals, aggregate_signals, PATTERN_KB
from utils.ohlcv_features  import fetch_ohlcv, add_all_features, label_volatility_regimes, get_feature_columns
from utils.visualizer      import draw_detections_on_image, plot_signal_summary

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Financial Chart CV System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric { background: #1a1a2e; border-radius: 8px; padding: 8px; }
    .bullish  { color: #00c853 !important; }
    .bearish  { color: #ff1744 !important; }
    .neutral  { color: #ffd600 !important; }
    h1        { color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CACHED LOADERS
# ──────────────────────────────────────────────

@st.cache_resource
def load_yolo_model():
    try:
        from ultralytics import YOLO
        if os.path.exists("./best_model_path.txt"):
            with open("./best_model_path.txt") as f:
                path = f.read().strip()
            if os.path.exists(path):
                return YOLO(path)
        st.warning("⚠️ Trained YOLO model not found. Using pretrained YOLOv8n as demo.")
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"YOLO load error: {e}")
        return None


@st.cache_resource
def load_fusion_model():
    try:
        meta_path  = "./outputs/fusion_model_meta.json"
        model_path = "./outputs/best_fusion_model.pt"
        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            return None, None
        with open(meta_path) as f:
            meta = json.load(f)
        from models.fusion_model import FusionModel
        model = FusionModel(meta["visual_dim"], meta["numerical_dim"])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model, meta
    except Exception as e:
        return None, None


@st.cache_data(ttl=3600)
def load_ohlcv(ticker, period="6mo"):
    try:
        df = fetch_ohlcv(ticker, period=period)
        df = add_all_features(df)
        df = label_volatility_regimes(df)
        return df
    except:
        return None


# ──────────────────────────────────────────────
# INFERENCE HELPERS
# ──────────────────────────────────────────────

def run_yolo_inference(model, image_path: str, conf: float = 0.25) -> list:
    """Run YOLO and return list of detection dicts."""
    results  = model.predict(image_path, conf=conf, verbose=False)
    names    = results[0].names
    detections = []
    if results[0].boxes is not None:
        for i in range(len(results[0].boxes)):
            box  = results[0].boxes[i]
            cls  = int(box.cls.item())
            detections.append({
                "name":       names.get(cls, f"class_{cls}"),
                "confidence": float(box.conf.item()),
                "bbox":       box.xyxyn[0].tolist(),
                "class_id":   cls,
            })
    return detections


def visual_feats_from_detections(detections: list, visual_dim: int = 64) -> np.ndarray:
    """Convert detections to visual feature vector (same as step 3)."""
    from utils.pattern_mapper import PATTERN_VOL_SCORE, PATTERN_KB
    BIAS_ENC = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    N = 20
    vec = np.zeros(N * 5, dtype=np.float32)
    for det in detections:
        name = det["name"]
        conf = det["confidence"]
        # find class index
        from 1_download_dataset import CLASS_NAMES
        idx = CLASS_NAMES.index(name) if name in CLASS_NAMES else -1
        if idx >= 0:
            base = idx * 5
            vec[base]     = max(vec[base], conf)
            vec[base + 1] += conf
            vec[base + 2] += 1
            vec[base + 3]  = PATTERN_VOL_SCORE.get(name, 0.5)
            kb = PATTERN_KB.get(name, {})
            vec[base + 4]  = BIAS_ENC.get(kb.get("bias", "neutral"), 0.0)
    out = vec[:visual_dim] if len(vec) >= visual_dim else np.pad(vec, (0, visual_dim - len(vec)))
    return out.reshape(1, -1)


def numerical_feats_from_df(df, visual_dim: int = 64) -> np.ndarray:
    """Get latest numerical features from OHLCV dataframe."""
    feat_cols = [c for c in get_feature_columns() if c in df.columns]
    row = df[feat_cols].iloc[-1].values.astype(np.float32)
    return row.reshape(1, -1)


# ──────────────────────────────────────────────
# PRICE CHART (Plotly Candlestick)
# ──────────────────────────────────────────────

def plot_candlestick(df, ticker: str, n_candles: int = 60):
    recent = df.tail(n_candles)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=recent.index,
        open=recent["Open"].squeeze(),
        high=recent["High"].squeeze(),
        low=recent["Low"].squeeze(),
        close=recent["Close"].squeeze(),
        name=ticker,
    ))
    if "vol_regime" in recent.columns:
        colors = {0: "rgba(0,200,0,0.08)", 1: "rgba(180,180,0,0.08)", 2: "rgba(220,0,0,0.08)"}
        for i, (idx, row) in enumerate(recent.iterrows()):
            fig.add_vrect(
                x0=idx, x1=recent.index[min(i + 1, len(recent) - 1)],
                fillcolor=colors.get(int(row.get("vol_regime", 1)), "transparent"),
                line_width=0,
            )
    fig.update_layout(
        title=f"{ticker} — Last {n_candles} Candles",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────

def main():
    st.title("📈 Financial Chart Understanding System")
    st.markdown(
        "**Candlestick Pattern Detection** (YOLOv8) + "
        "**Multimodal Volatility Forecasting** (Visual + OHLCV Fusion)"
    )
    st.divider()

    # ── Sidebar ──
    st.sidebar.header("⚙️ Settings")
    ticker    = st.sidebar.text_input("Ticker Symbol", value="BTC-USD",
                                       help="e.g. BTC-USD, RELIANCE.NS, ^NSEI")
    conf_thr  = st.sidebar.slider("Detection Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    period    = st.sidebar.selectbox("OHLCV Period", ["3mo", "6mo", "1y", "2y"], index=1)
    n_candles = st.sidebar.slider("Candles to display", 30, 120, 60)

    st.sidebar.divider()
    st.sidebar.markdown("**📚 References**")
    st.sidebar.markdown(
        "- Vijayababu & Bennur (2023)\n"
        "- Jung-Hua Liu (2025)\n"
        "- Ahihi Dataset (Roboflow)\n"
        "- YOLOv8 (Ultralytics)"
    )

    # ── Load models ──
    yolo_model          = load_yolo_model()
    fusion_model, fmeta = load_fusion_model()

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["🔍 Chart Analysis", "📊 Live OHLCV", "📖 Pattern Library"])

    # ────────────────────────────────────────
    # TAB 1: Upload & Detect
    # ────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Upload Chart Image")
            uploaded = st.file_uploader(
                "Upload a candlestick chart image (PNG/JPG)",
                type=["png", "jpg", "jpeg"],
            )
            st.caption("💡 Tip: Use clean charts without extra indicators for best results.")

        with col2:
            st.subheader("Detection Results")
            if uploaded is None:
                st.info("👆 Upload a chart image to start analysis.")

        if uploaded and yolo_model:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("🔍 Running pattern detection..."):
                detections = run_yolo_inference(yolo_model, tmp_path, conf=conf_thr)

            # Map to signals
            signals = map_detections_to_signals(detections)
            summary = aggregate_signals(signals)

            # Annotate image
            img_annotated = draw_detections_on_image(tmp_path, detections, signals)

            with col1:
                st.image(
                    cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB),
                    caption="Detected Patterns",
                    use_container_width=True,
                )

            with col2:
                if not detections:
                    st.warning("No patterns detected. Try lowering the confidence threshold.")
                else:
                    # Composite signal
                    bias  = summary["composite_bias"]
                    color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(bias, "⚪")
                    st.metric("Composite Signal",     f"{color} {bias.upper()}")
                    st.metric("Patterns Detected",    summary["n_patterns"])
                    st.metric("Avg Confidence",       f"{summary['avg_confidence']:.2%}")
                    st.metric("Avg Volatility Score", f"{summary['avg_vol_score']:.2f}")

                    st.divider()
                    st.markdown("**Detected Patterns:**")
                    for s in signals:
                        bias_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(s.bias, "⚪")
                        st.markdown(
                            f"{bias_icon} **{s.name}**  \n"
                            f"  Confidence: `{s.confidence:.2%}` | "
                            f"Action: `{s.action}` | "
                            f"Reliability: `{s.reliability:.0%}`"
                        )

            # Fusion prediction
            if fusion_model and fmeta and detections:
                st.divider()
                st.subheader("🧠 Fusion Model — Volatility Regime Prediction")
                with st.spinner("Fetching OHLCV data for fusion..."):
                    df_ohlcv = load_ohlcv(ticker, period)

                if df_ohlcv is not None:
                    vis_feat = visual_feats_from_detections(detections, fmeta["visual_dim"])
                    num_feat = numerical_feats_from_df(df_ohlcv, fmeta["visual_dim"])

                    if num_feat.shape[1] != fmeta["numerical_dim"]:
                        st.warning("Feature dimension mismatch — skipping fusion prediction.")
                    else:
                        vis_t  = torch.tensor(vis_feat,  dtype=torch.float32)
                        num_t  = torch.tensor(num_feat,  dtype=torch.float32)
                        probs  = fusion_model.predict_proba(vis_t, num_t)[0].numpy()
                        regime = int(np.argmax(probs))
                        names  = ["🟢 Low Volatility", "🟡 Medium Volatility", "🔴 High Volatility"]

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Predicted Regime", names[regime])
                        c2.metric("Confidence",       f"{probs[regime]:.1%}")
                        c3.metric("Dominant Pattern", summary.get("dominant_type", "—").capitalize())

                        # Probability bar
                        fig_prob = go.Figure(go.Bar(
                            x=["Low Vol", "Med Vol", "High Vol"],
                            y=probs,
                            marker_color=["#00c853", "#ffd600", "#ff1744"],
                        ))
                        fig_prob.update_layout(
                            title="Regime Probability Distribution",
                            template="plotly_dark", height=250,
                            margin=dict(l=10, r=10, t=40, b=10),
                            yaxis=dict(range=[0, 1]),
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)

    # ────────────────────────────────────────
    # TAB 2: Live OHLCV
    # ────────────────────────────────────────
    with tab2:
        st.subheader(f"Live OHLCV Data — {ticker}")
        with st.spinner(f"Fetching {ticker} data..."):
            df = load_ohlcv(ticker, period)

        if df is not None:
            # Metrics row
            last  = df.iloc[-1]
            prev  = df.iloc[-2]
            delta = float(last["Close"].squeeze()) - float(prev["Close"].squeeze())
            pct   = delta / float(prev["Close"].squeeze()) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Close",     f"${float(last['Close'].squeeze()):,.2f}", f"{delta:+.2f}")
            m2.metric("24h Change", f"{pct:+.2f}%")
            m3.metric("Volume",     f"{float(last['Volume'].squeeze()):,.0f}")

            regime = int(last.get("vol_regime", 1))
            regime_label = ["🟢 Low", "🟡 Medium", "🔴 High"][regime]
            m4.metric("Vol Regime", regime_label)

            st.plotly_chart(plot_candlestick(df, ticker, n_candles), use_container_width=True)

            with st.expander("📋 Raw Feature Table (last 10 rows)"):
                feat_cols = [c for c in get_feature_columns() if c in df.columns]
                st.dataframe(df[feat_cols].tail(10).style.format("{:.4f}"), use_container_width=True)
        else:
            st.error(f"Could not fetch data for {ticker}. Check the ticker symbol.")

    # ────────────────────────────────────────
    # TAB 3: Pattern Library
    # ────────────────────────────────────────
    with tab3:
        st.subheader("📖 Candlestick Pattern Knowledge Base")
        st.markdown(f"Total patterns: **{len(PATTERN_KB)}**")

        filter_bias = st.selectbox("Filter by bias", ["All", "bullish", "bearish", "neutral"])
        filter_type = st.selectbox("Filter by type", ["All", "reversal", "continuation"])

        rows = []
        for name, kb in PATTERN_KB.items():
            if filter_bias != "All" and kb["bias"] != filter_bias:
                continue
            if filter_type != "All" and kb["type"] != filter_type:
                continue
            rows.append({
                "Pattern":      name,
                "Bias":         kb["bias"],
                "Type":         kb["type"],
                "Strength":     "⭐" * kb["strength"],
                "Reliability":  f"{kb['reliability']:.0%}",
                "Action":       kb["action"],
                "Description":  kb["description"],
            })

        if rows:
            st.dataframe(
                rows,
                use_container_width=True,
                column_config={
                    "Reliability": st.column_config.ProgressColumn(
                        min_value=0, max_value=1,
                    ),
                },
            )
        else:
            st.info("No patterns match the selected filters.")


if __name__ == "__main__":
    main()
