from __future__ import annotations

import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from shared_features import build_state_vec_fast
from trade_core import softmax_probs


RegimeLabel = str


@dataclass
class RegimeDecision:
    """Container for 1分足モデルの判断結果."""

    timestamp: datetime
    expires_at: datetime
    regime: RegimeLabel
    p_buy: float
    p_sell: float

    def allows(self, direction: RegimeLabel) -> bool:
        if self.regime == "LONG_ONLY":
            return direction == "LONG"
        if self.regime == "SHORT_ONLY":
            return direction == "SHORT"
        return False

    @property
    def regime_id(self) -> str:
        return f"{self.timestamp.isoformat()}_{self.regime}"


class RegimeDecider:
    """Buy/Sellモデルから許可ゾーンを生成する."""

    def __init__(
        self,
        scaler,
        long_model: torch.nn.Module,
        short_model: torch.nn.Module,
        *,
        required_candles: int = 75,
        th_buy: float = 0.55,
        th_sell: float = 0.55,
        hold_minutes: int = 1,
    ) -> None:
        self.scaler = scaler
        self.long_model = long_model
        self.short_model = short_model
        self.required_candles = required_candles
        self.th_buy = th_buy
        self.th_sell = th_sell
        self.hold_minutes = max(1, hold_minutes)

    def _build_state(self, ohlc_df: pd.DataFrame, phase: float, sec_range: float) -> Optional[np.ndarray]:
        if ohlc_df is None or len(ohlc_df) < self.required_candles:
            return None
        tail = ohlc_df.tail(self.required_candles)
        state_vec = build_state_vec_fast(tail, phase, sec_range)
        if self.scaler is None:
            return state_vec.astype(np.float32)
        scaled = self.scaler.transform([state_vec])[0]
        return scaled.astype(np.float32)

    def _predict_prob(self, model: torch.nn.Module, x: np.ndarray) -> float:
        with torch.no_grad():
            t = torch.from_numpy(x).unsqueeze(0).float()
            logits = model(t).cpu().numpy().reshape(-1)
        probs = softmax_probs(logits)
        return float(probs[1])

    def evaluate(
        self,
        ohlc_df: pd.DataFrame,
        *,
        timestamp: datetime,
        phase: float = 1.0,
        sec_range: Optional[float] = None,
    ) -> Optional[RegimeDecision]:
        if ohlc_df is None or len(ohlc_df) < self.required_candles:
            return None
        latest = ohlc_df.iloc[-1]
        sec_range = sec_range if sec_range is not None else float(latest["high"] - latest["low"])
        state_vec = self._build_state(ohlc_df, phase, sec_range)
        if state_vec is None:
            return None

        p_long = self._predict_prob(self.long_model, state_vec)
        p_short = self._predict_prob(self.short_model, state_vec)

        if p_long >= self.th_buy and p_long >= p_short:
            regime = "LONG_ONLY"
        elif p_short >= self.th_sell and p_short > p_long:
            regime = "SHORT_ONLY"
        else:
            regime = "NO_TRADE"

        expires_at = timestamp + timedelta(minutes=self.hold_minutes)
        return RegimeDecision(
            timestamp=timestamp,
            expires_at=expires_at,
            regime=regime,
            p_buy=p_long,
            p_sell=p_short,
        )


@dataclass
class TriggerSignal:
    direction: RegimeLabel
    price: float
    timestamp: datetime


class MeanReversionTrigger:
    """短期EMAと長期EMAの乖離→回帰をトリガーとする."""

    def __init__(
        self,
        *,
        fast_window: int = 8,
        slow_window: int = 34,
        deviation_pips: float = 0.35,
        buffer_sec: int = 300,
    ) -> None:
        if slow_window <= fast_window:
            raise ValueError("slow_window must be greater than fast_window.")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.deviation_pips = deviation_pips
        self.buffer: Deque[Tuple[datetime, float]] = deque(maxlen=max(buffer_sec // 5, slow_window * 4))
        self.prev_spread: Optional[float] = None
        self.min_spread: Optional[float] = None
        self.max_spread: Optional[float] = None
        self.last_signal_time: Dict[RegimeLabel, datetime] = {}

    def reset(self) -> None:
        self.prev_spread = None
        self.min_spread = None
        self.max_spread = None

    def on_regime_change(self, regime: Optional[RegimeDecision]) -> None:
        if regime is None or regime.regime == "NO_TRADE":
            self.reset()
        elif regime.regime == "LONG_ONLY":
            self.max_spread = None
        elif regime.regime == "SHORT_ONLY":
            self.min_spread = None

    def update_price(self, timestamp: datetime, price: float) -> None:
        self.buffer.append((timestamp, float(price)))

    def _compute_spread(self) -> Optional[float]:
        if len(self.buffer) < self.slow_window:
            return None
        df = pd.DataFrame(self.buffer, columns=["ts", "price"]).set_index("ts")
        ema_fast = df["price"].ewm(span=self.fast_window, adjust=False).mean().iloc[-1]
        ema_slow = df["price"].ewm(span=self.slow_window, adjust=False).mean().iloc[-1]
        return float(ema_fast - ema_slow)

    def check(
        self,
        regime: RegimeDecision,
        *,
        pip_size: float,
        now: datetime,
        cooldown_seconds: float = 0.0,
    ) -> Optional[TriggerSignal]:
        if regime is None or regime.regime == "NO_TRADE":
            return None
        spread = self._compute_spread()
        if spread is None or pip_size <= 0:
            return None

        spread_pips = spread / pip_size
        signal: Optional[RegimeLabel] = None

        if regime.regime == "LONG_ONLY":
            self.min_spread = spread_pips if self.min_spread is None else min(self.min_spread, spread_pips)
            if self.min_spread <= -self.deviation_pips and spread_pips >= 0.0:
                last_sig = self.last_signal_time.get("LONG")
                if not last_sig or (now - last_sig).total_seconds() >= cooldown_seconds:
                    signal = "LONG"
                    self.last_signal_time["LONG"] = now
                self.min_spread = None

        elif regime.regime == "SHORT_ONLY":
            self.max_spread = spread_pips if self.max_spread is None else max(self.max_spread, spread_pips)
            if self.max_spread >= self.deviation_pips and spread_pips <= 0.0:
                last_sig = self.last_signal_time.get("SHORT")
                if not last_sig or (now - last_sig).total_seconds() >= cooldown_seconds:
                    signal = "SHORT"
                    self.last_signal_time["SHORT"] = now
                self.max_spread = None

        self.prev_spread = spread_pips

        if signal:
            price = self.buffer[-1][1]
            return TriggerSignal(direction=signal, price=price, timestamp=now)
        return None


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""


@dataclass
class RiskManager:
    max_positions_per_side: int = 2
    max_entries_per_minute: int = 3
    entry_cooldown_seconds: float = 10.0
    per_regime_loss_limit: float = 0.004  # -0.4%
    daily_loss_limit: float = 0.02  # -2%
    consecutive_loss_limit: int = 3
    max_spread_pips: Optional[float] = None

    entries_this_minute: int = field(init=False, default=0)
    last_minute: Optional[datetime] = field(init=False, default=None)
    last_entry_time: Optional[datetime] = field(init=False, default=None)
    current_regime_id: Optional[str] = field(init=False, default=None)
    regime_loss: float = field(init=False, default=0.0)
    daily_loss: Dict[str, float] = field(init=False, default_factory=lambda: defaultdict(float))
    loss_streak: Dict[RegimeLabel, int] = field(init=False, default_factory=lambda: defaultdict(int))

    def _reset_minute_if_needed(self, current_time: datetime) -> None:
        minute = current_time.replace(second=0, microsecond=0)
        if self.last_minute != minute:
            self.entries_this_minute = 0
            self.last_minute = minute

    def on_regime_change(self, regime_id: Optional[str]) -> None:
        if regime_id != self.current_regime_id:
            self.current_regime_id = regime_id
            self.regime_loss = 0.0
            self.entries_this_minute = 0

    def can_enter(
        self,
        direction: RegimeLabel,
        *,
        current_time: datetime,
        open_positions: Iterable[Dict],
        regime_id: Optional[str],
        spread_pips: Optional[float] = None,
        ignore_cooldown: bool = False,
    ) -> RiskCheckResult:
        self._reset_minute_if_needed(current_time)
        self.on_regime_change(regime_id)

        if self.max_spread_pips is not None and spread_pips is not None:
            if spread_pips > self.max_spread_pips:
                return RiskCheckResult(False, "spread_block")

        if self.last_entry_time and not ignore_cooldown:
            delta = (current_time - self.last_entry_time).total_seconds()
            if delta < self.entry_cooldown_seconds:
                return RiskCheckResult(False, "cooldown")

        if self.entries_this_minute >= self.max_entries_per_minute:
            return RiskCheckResult(False, "per_minute_cap")

        side_positions = [pos for pos in open_positions if pos.get("side") == direction]
        if len(side_positions) >= self.max_positions_per_side:
            return RiskCheckResult(False, "per_side_cap")

        if self.regime_loss <= -abs(self.per_regime_loss_limit):
            return RiskCheckResult(False, "regime_loss_cap")

        day_key = current_time.strftime("%Y-%m-%d")
        if self.daily_loss[day_key] <= -abs(self.daily_loss_limit):
            return RiskCheckResult(False, "daily_loss_cap")

        if self.loss_streak[direction] >= self.consecutive_loss_limit:
            return RiskCheckResult(False, "consecutive_losses")

        return RiskCheckResult(True, "")

    def register_entry(self, current_time: datetime) -> None:
        self._reset_minute_if_needed(current_time)
        self.entries_this_minute += 1
        self.last_entry_time = current_time

    def register_exit(
        self,
        *,
        direction: RegimeLabel,
        pnl_fraction: float,
        exit_time: datetime,
        regime_id: Optional[str],
    ) -> None:
        self.on_regime_change(regime_id)
        self.regime_loss += pnl_fraction
        day_key = exit_time.strftime("%Y-%m-%d")
        self.daily_loss[day_key] += pnl_fraction
        if pnl_fraction < 0:
            self.loss_streak[direction] += 1
        else:
            self.loss_streak[direction] = 0
