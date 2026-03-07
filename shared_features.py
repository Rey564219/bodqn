from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_feature_cache: dict[Tuple, np.ndarray] = {}


def _to_ohlc_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC columns missing: {missing}")
    out = df[required].copy()
    out[required] = out[required].astype(float)
    return out


def _safe_div(num: pd.Series | np.ndarray | float, den: pd.Series | np.ndarray | float, eps: float = 1e-12):
    return np.asarray(num) / (np.asarray(den) + eps)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / (avg_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def FeatureExtraction(df: pd.DataFrame, use_cache: bool = True) -> np.ndarray:
    ohlc = _to_ohlc_frame(df)
    if len(ohlc) == 0:
        return np.zeros((0, 18), dtype=np.float32)

    close = ohlc["close"]
    open_ = ohlc["open"]
    high = ohlc["high"]
    low = ohlc["low"]

    ret1 = close.pct_change(1).fillna(0.0)
    ret3 = close.pct_change(3).fillna(0.0)
    ret5 = close.pct_change(5).fillna(0.0)

    hl_range = ((high - low) / (close + 1e-12)).fillna(0.0)
    body = ((close - open_) / (open_ + 1e-12)).fillna(0.0)
    upper_wick = ((high - np.maximum(open_, close)) / (close + 1e-12)).fillna(0.0)
    lower_wick = ((np.minimum(open_, close) - low) / (close + 1e-12)).fillna(0.0)

    ema5 = close.ewm(span=5, adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    ema5r = (ema5 / (close + 1e-12) - 1.0).fillna(0.0)
    ema10r = (ema10 / (close + 1e-12) - 1.0).fillna(0.0)
    ema20r = (ema20 / (close + 1e-12) - 1.0).fillna(0.0)
    ema50r = (ema50 / (close + 1e-12) - 1.0).fillna(0.0)

    rsi14 = (_rsi(close, 14) / 100.0).fillna(0.5)
    vol10 = close.pct_change().rolling(10, min_periods=1).std().fillna(0.0)
    vol20 = close.pct_change().rolling(20, min_periods=1).std().fillna(0.0)

    roll_min = close.rolling(20, min_periods=1).min()
    roll_max = close.rolling(20, min_periods=1).max()
    pos20 = ((close - roll_min) / (roll_max - roll_min + 1e-12)).fillna(0.5)

    atr14 = (_atr(ohlc, 14) / (close + 1e-12)).fillna(0.0)
    mom10 = (close / (close.shift(10) + 1e-12) - 1.0).fillna(0.0)

    feat_df = pd.DataFrame(
        {
            "ret1": ret1,
            "ret3": ret3,
            "ret5": ret5,
            "hl_range": hl_range,
            "body": body,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "ema5r": ema5r,
            "ema10r": ema10r,
            "ema20r": ema20r,
            "ema50r": ema50r,
            "rsi14": rsi14,
            "vol10": vol10,
            "vol20": vol20,
            "pos20": pos20,
            "atr14": atr14,
            "mom10": mom10,
            "trend_short": (ema5 / (ema20 + 1e-12) - 1.0).fillna(0.0),
        }
    )

    values = feat_df.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32).to_numpy()
    return values


def _cache_key(ohlc_win_df: pd.DataFrame, extra: Optional[Sequence[float]]) -> Tuple:
    if len(ohlc_win_df) == 0:
        return (0, None, None, tuple(extra) if extra is not None else None)
    last_idx = ohlc_win_df.index[-1]
    last_close = float(ohlc_win_df["close"].iloc[-1]) if "close" in ohlc_win_df.columns else None
    first_close = float(ohlc_win_df["close"].iloc[0]) if "close" in ohlc_win_df.columns else None
    extra_key = tuple(float(x) for x in extra) if extra is not None else None
    return (len(ohlc_win_df), str(last_idx), first_close, last_close, extra_key)


def build_state_vec(
    ohlc_win_df: pd.DataFrame,
    extra: Optional[Sequence[float]] = None,
    use_cache: bool = True,
) -> np.ndarray:
    key = _cache_key(ohlc_win_df, extra)
    if use_cache and key in _feature_cache:
        return _feature_cache[key].copy()

    feats = FeatureExtraction(ohlc_win_df, use_cache=False)
    base = feats[-1] if len(feats) > 0 else np.zeros(18, dtype=np.float32)

    if extra is None:
        ext = np.zeros(2, dtype=np.float32)
    else:
        extra_vals = [float(v) for v in extra]
        if len(extra_vals) < 2:
            extra_vals = extra_vals + [0.0] * (2 - len(extra_vals))
        ext = np.asarray(extra_vals[:2], dtype=np.float32)

    state = np.concatenate([base.astype(np.float32), ext], axis=0).astype(np.float32)
    state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)

    if use_cache:
        _feature_cache[key] = state.copy()
    return state


def build_state_vec_fast(ohlc_win_df: pd.DataFrame, phase: float, sec_range: float) -> np.ndarray:
    return build_state_vec(ohlc_win_df, [phase, sec_range], use_cache=True)


def clear_feature_cache() -> None:
    _feature_cache.clear()


def compute_trend_direction(close_series: Iterable[float] | pd.Series, window: int = 75) -> float:
    if isinstance(close_series, pd.Series):
        s = close_series.astype(float)
    else:
        s = pd.Series(list(close_series), dtype=float)
    if len(s) == 0:
        return 0.0

    tail = s.tail(max(2, int(window)))
    y = tail.to_numpy(dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)

    x_mean = x.mean()
    y_mean = y.mean()
    den = np.sum((x - x_mean) ** 2)
    if den <= 1e-12:
        return 0.0

    slope = np.sum((x - x_mean) * (y - y_mean)) / den
    norm = abs(y_mean) + 1e-12
    return float(slope / norm)
