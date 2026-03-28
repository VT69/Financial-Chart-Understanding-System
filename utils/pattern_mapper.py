"""
utils/pattern_mapper.py
Maps detected candlestick patterns to trading signals and volatility scores.
This creates the interpretable output layer of the system.
"""

from dataclasses import dataclass
from typing import List, Dict


# ──────────────────────────────────────────────
# PATTERN KNOWLEDGE BASE
# ──────────────────────────────────────────────

PATTERN_KB = {
    "Three Inside Up-Down": {
        "bias": "neutral", "type": "reversal",
        "strength": 2, "reliability": 0.65,
        "description": "3-candle reversal; direction depends on preceding trend.",
        "action": "wait_for_confirmation",
    },
    "Hikkake Pattern": {
        "bias": "neutral", "type": "reversal",
        "strength": 2, "reliability": 0.60,
        "description": "False breakout pattern; often traps traders.",
        "action": "watch",
    },
    "Advance Block": {
        "bias": "bearish", "type": "reversal",
        "strength": 2, "reliability": 0.62,
        "description": "Weakening bullish momentum in uptrend; potential reversal.",
        "action": "reduce_longs",
    },
    "Three Outside Up-Down": {
        "bias": "neutral", "type": "reversal",
        "strength": 3, "reliability": 0.70,
        "description": "Strong reversal signal confirmed by engulfing pattern.",
        "action": "wait_for_confirmation",
    },
    "Upside-Downside Gap Three Methods": {
        "bias": "neutral", "type": "continuation",
        "strength": 2, "reliability": 0.60,
        "description": "Continuation with gap; trend expected to resume.",
        "action": "hold",
    },
    "Tasuki Gap": {
        "bias": "neutral", "type": "continuation",
        "strength": 2, "reliability": 0.63,
        "description": "Partial gap fill followed by trend continuation.",
        "action": "hold",
    },
    "Evening Star": {
        "bias": "bearish", "type": "reversal",
        "strength": 3, "reliability": 0.72,
        "description": "Classic 3-candle bearish reversal at market top.",
        "action": "consider_short",
    },
    "Rising-Falling Three Methods": {
        "bias": "neutral", "type": "continuation",
        "strength": 3, "reliability": 0.68,
        "description": "Consolidation within trend; trend likely to continue.",
        "action": "hold",
    },
    "Morning Doji Star": {
        "bias": "bullish", "type": "reversal",
        "strength": 3, "reliability": 0.73,
        "description": "Strong bullish reversal with doji gap at bottom.",
        "action": "consider_long",
    },
    "Morning Star": {
        "bias": "bullish", "type": "reversal",
        "strength": 3, "reliability": 0.75,
        "description": "Classic 3-candle bullish reversal at market bottom.",
        "action": "consider_long",
    },
    "Three Black Crows": {
        "bias": "bearish", "type": "reversal",
        "strength": 3, "reliability": 0.74,
        "description": "Three consecutive bearish candles; strong downtrend signal.",
        "action": "consider_short",
    },
    "Three Line Strike": {
        "bias": "bullish", "type": "reversal",
        "strength": 2, "reliability": 0.65,
        "description": "4-candle pattern; bullish engulfing after 3 bearish.",
        "action": "consider_long",
    },
    "Evening Doji Star": {
        "bias": "bearish", "type": "reversal",
        "strength": 3, "reliability": 0.71,
        "description": "Bearish reversal with doji gap at top.",
        "action": "consider_short",
    },
    "Tristar Pattern": {
        "bias": "neutral", "type": "reversal",
        "strength": 2, "reliability": 0.58,
        "description": "Three consecutive doji candles; extreme indecision.",
        "action": "watch",
    },
    "Up-Down Gap Side-by-side White Lines": {
        "bias": "bullish", "type": "continuation",
        "strength": 2, "reliability": 0.61,
        "description": "Gap continuation with two parallel bullish candles.",
        "action": "hold_long",
    },
    "Stick Sandwich": {
        "bias": "bearish", "type": "reversal",
        "strength": 2, "reliability": 0.60,
        "description": "Bearish candle sandwiched between two bullish; reversal signal.",
        "action": "reduce_longs",
    },
    "Ladder Bottom": {
        "bias": "bullish", "type": "reversal",
        "strength": 3, "reliability": 0.70,
        "description": "5-candle bullish reversal pattern after downtrend.",
        "action": "consider_long",
    },
    "Unique 3 River": {
        "bias": "bullish", "type": "reversal",
        "strength": 2, "reliability": 0.63,
        "description": "Rare bullish reversal; 3rd candle closes within 1st candle.",
        "action": "consider_long",
    },
    "Three Advancing White Soldiers": {
        "bias": "bullish", "type": "reversal",
        "strength": 3, "reliability": 0.76,
        "description": "Three consecutive bullish candles; strong uptrend signal.",
        "action": "consider_long",
    },
    "Identical Three Crows": {
        "bias": "bearish", "type": "reversal",
        "strength": 3, "reliability": 0.73,
        "description": "Stronger variant of Three Black Crows; bearish confirmation.",
        "action": "consider_short",
    },
}

# Volatility score per pattern (heuristic: complex patterns → higher vol)
PATTERN_VOL_SCORE = {
    "Three Inside Up-Down": 0.5,
    "Hikkake Pattern": 0.7,
    "Advance Block": 0.4,
    "Three Outside Up-Down": 0.6,
    "Upside-Downside Gap Three Methods": 0.5,
    "Tasuki Gap": 0.5,
    "Evening Star": 0.6,
    "Rising-Falling Three Methods": 0.4,
    "Morning Doji Star": 0.7,
    "Morning Star": 0.6,
    "Three Black Crows": 0.8,
    "Three Line Strike": 0.6,
    "Evening Doji Star": 0.7,
    "Tristar Pattern": 0.9,
    "Up-Down Gap Side-by-side White Lines": 0.5,
    "Stick Sandwich": 0.5,
    "Ladder Bottom": 0.6,
    "Unique 3 River": 0.6,
    "Three Advancing White Soldiers": 0.7,
    "Identical Three Crows": 0.8,
}


@dataclass
class PatternSignal:
    name: str
    confidence: float       # YOLO detection confidence
    bias: str               # bullish / bearish / neutral
    pattern_type: str       # reversal / continuation
    strength: int           # 1-3
    reliability: float      # historical reliability score
    action: str             # recommended action
    vol_score: float        # 0-1 volatility contribution score
    description: str


def map_detections_to_signals(detections: List[Dict]) -> List[PatternSignal]:
    """
    Convert raw YOLO detections to structured PatternSignal objects.

    Args:
        detections: list of dicts with keys: 'name', 'confidence'
    
    Returns:
        List of PatternSignal objects
    """
    signals = []
    for det in detections:
        name = det.get("name", "")
        conf = det.get("confidence", 0.0)
        if name in PATTERN_KB:
            kb = PATTERN_KB[name]
            signals.append(PatternSignal(
                name=name,
                confidence=conf,
                bias=kb["bias"],
                pattern_type=kb["type"],
                strength=kb["strength"],
                reliability=kb["reliability"],
                action=kb["action"],
                vol_score=PATTERN_VOL_SCORE.get(name, 0.5),
                description=kb["description"],
            ))
    return signals


def aggregate_signals(signals: List[PatternSignal]) -> Dict:
    """
    Aggregate multiple detected patterns into a single composite signal.
    Returns a summary dict suitable for display or downstream model input.
    """
    if not signals:
        return {
            "composite_bias": "neutral",
            "avg_confidence": 0.0,
            "avg_vol_score": 0.0,
            "dominant_type": "none",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "n_patterns": 0,
        }

    bullish  = sum(1 for s in signals if s.bias == "bullish")
    bearish  = sum(1 for s in signals if s.bias == "bearish")
    neutral  = sum(1 for s in signals if s.bias == "neutral")
    total    = len(signals)

    if bullish > bearish and bullish > neutral:
        composite_bias = "bullish"
    elif bearish > bullish and bearish > neutral:
        composite_bias = "bearish"
    else:
        composite_bias = "neutral"

    reversals     = sum(1 for s in signals if s.pattern_type == "reversal")
    continuations = sum(1 for s in signals if s.pattern_type == "continuation")
    dominant_type = "reversal" if reversals >= continuations else "continuation"

    return {
        "composite_bias":  composite_bias,
        "avg_confidence":  sum(s.confidence for s in signals) / total,
        "avg_vol_score":   sum(s.vol_score for s in signals) / total,
        "dominant_type":   dominant_type,
        "bullish_count":   bullish,
        "bearish_count":   bearish,
        "neutral_count":   neutral,
        "n_patterns":      total,
        "patterns":        [s.name for s in signals],
    }


if __name__ == "__main__":
    # Quick test
    mock_detections = [
        {"name": "Morning Star", "confidence": 0.87},
        {"name": "Three Advancing White Soldiers", "confidence": 0.91},
    ]
    signals = map_detections_to_signals(mock_detections)
    summary = aggregate_signals(signals)
    print("[*] Pattern signals:")
    for s in signals:
        print(f"    {s.name}: {s.bias} | action={s.action} | vol={s.vol_score}")
    print(f"\n[*] Aggregate: {summary}")
