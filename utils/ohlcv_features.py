"""
utils/ohlcv_features.py
Feature engineering on OHLCV time-series data.
Used to build the numerical branch of the multimodal fusion model.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


# ──────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────

def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    
    Args:
        ticker   : e.g. "BTC-USD", "RELIANCE.NS", "^NSEI"
        period   : "1y", "2y", "5y", etc.
        interval : "1d", "1h", "15m"
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicator features from OHLCV data.
    Returns a new DataFrame with features appended.
    """
    df = df.copy()
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    # ── Price-based features ──
    df["returns"]        = close.pct_change()
    df["log_returns"]    = np.log(close / close.shift(1))
    df["hl_range"]       = (high - low) / close          # intraday range normalized
    df["oc_range"]       = (close - df["Open"].squeeze()) / close

    # ── Moving averages ──
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"]   = SMAIndicator(close, window=w).sma_indicator()
        df[f"ema_{w}"]   = EMAIndicator(close, window=w).ema_indicator()
        df[f"price_vs_sma_{w}"] = (close - df[f"sma_{w}"]) / df[f"sma_{w}"]

    # ── Momentum ──
    df["rsi_14"]         = RSIIndicator(close, window=14).rsi()
    df["rsi_7"]          = RSIIndicator(close, window=7).rsi()
    stoch                = StochasticOscillator(high, low, close, window=14)
    df["stoch_k"]        = stoch.stoch()
    df["stoch_d"]        = stoch.stoch_signal()

    # ── MACD ──
    macd_obj             = MACD(close)
    df["macd"]           = macd_obj.macd()
    df["macd_signal"]    = macd_obj.macd_signal()
    df["macd_hist"]      = macd_obj.macd_diff()

    # ── Volatility ──
    bb                   = BollingerBands(close, window=20)
    df["bb_upper"]       = bb.bollinger_hband()
    df["bb_lower"]       = bb.bollinger_lband()
    df["bb_width"]       = (df["bb_upper"] - df["bb_lower"]) / close
    df["bb_pct"]         = bb.bollinger_pband()
    df["atr_14"]         = AverageTrueRange(high, low, close, window=14).average_true_range()

    # Rolling volatility (realized vol over different windows)
    for w in [5, 10, 20]:
        df[f"rvol_{w}"]  = df["log_returns"].rolling(w).std() * np.sqrt(252)

    # ── Volume features ──
    df["vol_change"]     = vol.pct_change()
    df["vol_sma_10"]     = vol.rolling(10).mean()
    df["vol_ratio"]      = vol / df["vol_sma_10"]

    # ── Lag features ──
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["returns"].shift(lag)

    df.dropna(inplace=True)
    return df


def get_feature_columns() -> list:
    """Return list of all feature column names (for model input)."""
    cols = ["returns", "log_returns", "hl_range", "oc_range"]
    for w in [5, 10, 20, 50]:
        cols += [f"sma_{w}", f"ema_{w}", f"price_vs_sma_{w}"]
    cols += [
        "rsi_14", "rsi_7", "stoch_k", "stoch_d",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width", "bb_pct", "atr_14",
        "rvol_5", "rvol_10", "rvol_20",
        "vol_change", "vol_sma_10", "vol_ratio",
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    ]
    return cols


# ──────────────────────────────────────────────
# VOLATILITY REGIME LABELING
# ──────────────────────────────────────────────

def label_volatility_regimes(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Label each row with a volatility regime:
        0 = Low volatility
        1 = Medium volatility
        2 = High volatility

    Based on rolling realized volatility percentile.
    This is the target label for the fusion model.
    """
    df = df.copy()
    rvol = df["log_returns"].rolling(window).std() * np.sqrt(252)

    low_q  = rvol.quantile(0.33)
    high_q = rvol.quantile(0.66)

    df["vol_regime"] = rvol.apply(
        lambda x: 0 if x <= low_q else (2 if x >= high_q else 1)
    )
    df.dropna(inplace=True)
    return df


def get_window_features(df: pd.DataFrame, idx: int, lookback: int = 10) -> np.ndarray:
    """
    Extract a flat feature vector for a window of `lookback` rows ending at `idx`.
    Used when aligning OHLCV features with a chart image for one data point.
    """
    feat_cols = get_feature_columns()
    available = [c for c in feat_cols if c in df.columns]
    window = df[available].iloc[max(0, idx - lookback): idx].values
    # Pad if needed
    if window.shape[0] < lookback:
        pad = np.zeros((lookback - window.shape[0], window.shape[1]))
        window = np.vstack([pad, window])
    return window.flatten().astype(np.float32)


if __name__ == "__main__":
    print("[*] Testing OHLCV feature engineering on BTC-USD...")
    df = fetch_ohlcv("BTC-USD", period="1y")
    df = add_all_features(df)
    df = label_volatility_regimes(df)

    print(f"    Shape  : {df.shape}")
    print(f"    Features: {get_feature_columns()[:5]} ... ({len(get_feature_columns())} total)")
    print(f"    Regime distribution:\n{df['vol_regime'].value_counts()}")
    print("[✓] OHLCV features OK")
