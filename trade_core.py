from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd


ATR_FAST_PERIOD = 14
ATR_SLOW_PERIOD = 100
VOL_LOW_TH = 0.8
VOL_HIGH_TH = 1.3
MIN_TP_PIPS = 1.5
MIN_SL_PIPS = 2.0
N0_HOLD_MIN = 4
N_MIN_HOLD = 2
N_MAX_HOLD = 6


@dataclass
class ExitParams:
    trade_allowed: bool
    regime: str
    tp_pips: float
    sl_pips: float
    max_hold_min: int
    r: float


@dataclass
class PositionState:
    side: str
    entry_price: float
    tp_price: float
    sl_price: float
    timeout_time: datetime
    entry_time: datetime


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def get_vol_regime(atr_fast: float, atr_slow: float) -> str:
    if not np.isfinite(atr_fast) or not np.isfinite(atr_slow) or atr_slow <= 0:
        return "MID"
    r = atr_fast / atr_slow
    if r < VOL_LOW_TH:
        return "LOW"
    if r < VOL_HIGH_TH:
        return "MID"
    return "HIGH"


def make_exit_params(
    atr_fast_pips: float,
    atr_slow_pips: float,
    spread_pips: float,
    min_tp_pips: float = MIN_TP_PIPS,
    min_sl_pips: float = MIN_SL_PIPS,
    n0: int = N0_HOLD_MIN,
    n_min: int = N_MIN_HOLD,
    n_max: int = N_MAX_HOLD,
) -> ExitParams:
    if not np.isfinite(atr_fast_pips) or not np.isfinite(atr_slow_pips) or atr_slow_pips <= 0:
        return ExitParams(
            trade_allowed=False,
            regime="MID",
            tp_pips=min_tp_pips,
            sl_pips=min_sl_pips,
            max_hold_min=n_max,
            r=0.0,
        )
    r = atr_fast_pips / atr_slow_pips
    regime = get_vol_regime(atr_fast_pips, atr_slow_pips)
    trade_allowed = atr_fast_pips >= spread_pips * 5.0

    if regime == "LOW":
        k_tp, k_sl = 0.6, 0.8
    elif regime == "MID":
        k_tp, k_sl = 0.8, 1.0
    else:
        k_tp, k_sl = 1.0, 1.2

    tp_pips_raw = k_tp * atr_fast_pips
    sl_pips_raw = k_sl * atr_fast_pips
    tp_pips = max(tp_pips_raw, spread_pips * 2.5, min_tp_pips)
    sl_pips = max(sl_pips_raw, spread_pips * 3.0, min_sl_pips)

    n_raw = round(n0 / r) if r > 0 else n_max
    n = int(min(max(n_raw, n_min), n_max))
    return ExitParams(
        trade_allowed=trade_allowed,
        regime=regime,
        tp_pips=float(tp_pips),
        sl_pips=float(sl_pips),
        max_hold_min=n,
        r=float(r),
    )


def decide_entry_two_models(p_long: float, p_short: float, entry_th: float) -> str:
    if p_long >= entry_th and p_long > p_short:
        return "LONG"
    if p_short >= entry_th and p_short > p_long:
        return "SHORT"
    return "HOLD"


def softmax_probs(q_vals) -> np.ndarray:
    q = np.array(q_vals, dtype=float)
    q = q - np.max(q)
    exp_q = np.exp(q)
    return exp_q / np.sum(exp_q)


def build_exit_levels(
    side: str,
    entry_price: float,
    pip_size: float,
    exit_params: ExitParams,
    entry_time: datetime,
) -> PositionState:
    tp_pips = exit_params.tp_pips
    sl_pips = exit_params.sl_pips
    max_hold = exit_params.max_hold_min
    if side == "LONG":
        tp = entry_price + tp_pips * pip_size
        sl = entry_price - sl_pips * pip_size
    else:
        tp = entry_price - tp_pips * pip_size
        sl = entry_price + sl_pips * pip_size
    timeout_time = entry_time + timedelta(minutes=max_hold)
    return PositionState(
        side=side,
        entry_price=entry_price,
        tp_price=tp,
        sl_price=sl,
        timeout_time=timeout_time,
        entry_time=entry_time,
    )


def check_exit(
    position: PositionState,
    ohlc: dict,
    current_time: datetime,
) -> Tuple[bool, Optional[str]]:
    high = float(ohlc.get("high"))
    low = float(ohlc.get("low"))

    if position.side == "LONG":
        hit_tp = high >= position.tp_price
        hit_sl = low <= position.sl_price
    else:
        hit_tp = low <= position.tp_price
        hit_sl = high >= position.sl_price

    if hit_tp and hit_sl:
        return True, "SL"
    if hit_sl:
        return True, "SL"
    if hit_tp:
        return True, "TP"
    if current_time >= position.timeout_time:
        return True, "TIMEOUT"
    return False, None


def calc_risk_amount(balance: float, risk_pct: float) -> float:
    return max(0.0, balance * risk_pct)


def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return np.floor(value / step) * step


def calc_fx_lots_fixed_risk(
    balance: float,
    risk_pct: float,
    sl_pips: float,
    pip_value_per_lot: float,
    min_lot: float,
    lot_step: float,
) -> float:
    if sl_pips <= 0 or pip_value_per_lot <= 0:
        return 0.0
    risk_amount = calc_risk_amount(balance, risk_pct)
    lots = risk_amount / (sl_pips * pip_value_per_lot)
    lots = max(lots, 0.0)
    lots = round_step(lots, lot_step)
    return max(lots, min_lot)


def calc_crypto_qty_fixed_risk(
    balance: float,
    risk_pct: float,
    entry_price: float,
    sl_price: float,
    leverage: float,
    min_qty: float,
    qty_step: float,
) -> float:
    if entry_price <= 0:
        return 0.0
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 0:
        return 0.0
    risk_amount = calc_risk_amount(balance, risk_pct)
    qty_risk = risk_amount / sl_distance
    max_notional = balance * leverage
    qty_leverage = max_notional / entry_price if max_notional > 0 else qty_risk
    qty = min(qty_risk, qty_leverage)
    qty = round_step(qty, qty_step)
    return max(qty, min_qty)


def estimate_net_pnl(
    side: str,
    entry_price: float,
    exit_price: float,
    qty: float,
    fee_rate: float,
    slippage_rate: float = 0.0,
) -> float:
    if side == "LONG":
        gross = (exit_price - entry_price) * qty
    else:
        gross = (entry_price - exit_price) * qty
    fee = abs(entry_price * qty) * fee_rate + abs(exit_price * qty) * fee_rate
    slip = abs(entry_price * qty) * slippage_rate + abs(exit_price * qty) * slippage_rate
    return gross - fee - slip
