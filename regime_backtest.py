from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from regime_executor import MeanReversionTrigger, RegimeDecider, RiskManager
from shared_features import build_state_vec_fast
from trade_core import ATR_FAST_PERIOD, ATR_SLOW_PERIOD, calc_atr, make_exit_params
from train_dqn import QNet, TRADE_PAIRS, simulate_exit


def _resolve_model_artifacts(pair: str):
    model_pair = str(pair).upper()
    if model_pair not in TRADE_PAIRS:
        raise SystemExit(f"Unsupported pair '{pair}'. Choose from {TRADE_PAIRS}")

    model_dir = os.path.join("Models", model_pair)
    model_files = {
        "long": os.path.join(model_dir, "dqn_policy_high.pt"),
        "short": os.path.join(model_dir, "dqn_policy_low.pt"),
    }
    scaler_file = os.path.join(model_dir, "dqn_scaler.pkl")

    missing = [p for p in [*model_files.values(), scaler_file] if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Model artifacts not found for {model_pair}: {missing}")

    return model_files, scaler_file


def _pip_size_for_pair(pair: str) -> float:
    return 0.01 if "JPY" in pair.upper() else 0.0001


def _default_spread(pair: str) -> float:
    return 0.2 if "JPY" in pair.upper() else 0.8


def _load_scaler(path: str):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def _load_model(path: str, input_dim: int) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu")
    model = QNet(input_dim, 2)
    if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
        state = state.get("model_state_dict", state.get("state_dict"))
    model.load_state_dict(state)
    model.eval()
    return model


@dataclass
class TradeResult:
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl_pips: float
    reason: str


def _build_minute_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        parse_dates={"ts": ["date", "time"]},
    )
    df = df.set_index("ts").sort_index()
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df


def _resample_lower(df: pd.DataFrame, seconds: int = 5) -> pd.Series:
    series = df["close"].resample(f"{seconds}S").interpolate("linear")
    return series


def run_backtest(
    minute_df: pd.DataFrame,
    lower_prices: pd.Series,
    scaler,
    long_model: torch.nn.Module,
    short_model: torch.nn.Module,
    *,
    pair: str,
    th_long: float,
    th_short: float,
) -> List[TradeResult]:
    regime_decider = RegimeDecider(
        scaler=scaler,
        long_model=long_model,
        short_model=short_model,
        required_candles=75,
        th_buy=th_long,
        th_sell=th_short,
    )
    trigger = MeanReversionTrigger()
    risk = RiskManager()
    pip_size = _pip_size_for_pair(pair)
    spread = _default_spread(pair)

    trade_results: List[TradeResult] = []
    minute_iter = iter(minute_df.iterrows())
    next_minute = next(minute_iter, None)
    active_regime = None
    last_regime_time: Optional[datetime] = None

    atr_fast = calc_atr(minute_df, ATR_FAST_PERIOD)
    atr_slow = calc_atr(minute_df, ATR_SLOW_PERIOD)

    open_trades: List[dict] = []

    for ts, price in lower_prices.items():
        price = float(price)
        # update regime on minute boundaries
        if next_minute and ts >= next_minute[0]:
            while next_minute and ts >= next_minute[0]:
                bar_time, bar_row = next_minute
                last_regime_time = bar_time
                sec_range = float(bar_row["high"] - bar_row["low"])
                df_slice = minute_df.loc[:bar_time]
                regime = regime_decider.evaluate(
                    df_slice,
                    timestamp=bar_time,
                    phase=1.0,
                    sec_range=sec_range,
                )
                active_regime = regime
                trigger.on_regime_change(active_regime)
                risk.on_regime_change(regime.regime_id if regime else None)
                next_minute = next(minute_iter, None)

        trigger.update_price(ts, price)
        if not active_regime or active_regime.regime == "NO_TRADE":
            continue
        signal = trigger.check(
            active_regime,
            pip_size=pip_size,
            now=ts,
            cooldown_seconds=risk.entry_cooldown_seconds,
        )
        if not signal:
            continue

        risk_check = risk.can_enter(
            signal.direction,
            current_time=ts,
            open_positions=open_trades,
            regime_id=active_regime.regime_id,
            spread_pips=spread,
        )
        if not risk_check.allowed:
            continue

        minute_bar = minute_df.loc[:ts].iloc[-1]
        idx = minute_df.index.get_loc(minute_bar.name)
        atr_f = float(atr_fast.iloc[idx])
        atr_s = float(atr_slow.iloc[idx])
        exit_params = make_exit_params(atr_f / pip_size, atr_s / pip_size, spread)
        future_slice = minute_df.iloc[idx + 1 : idx + 1 + exit_params.max_hold_min]
        exit_price, exit_reason = simulate_exit(
            1 if signal.direction == "LONG" else 2,
            price,
            future_slice,
            exit_params,
            pip_size,
        )
        pnl_pips = (exit_price - price) / pip_size if signal.direction == "LONG" else (price - exit_price) / pip_size
        pnl_pips -= spread
        exit_time = future_slice.index[0] if len(future_slice) else ts + timedelta(minutes=exit_params.max_hold_min)
        trade_results.append(
            TradeResult(
                direction=signal.direction,
                entry_time=ts,
                exit_time=exit_time,
                entry_price=price,
                exit_price=exit_price,
                pnl_pips=float(pnl_pips),
                reason=exit_reason,
            )
        )
        open_trades.append({"side": signal.direction, "entry_time": ts})
        risk.register_entry(ts)
        risk.register_exit(
            direction=signal.direction,
            pnl_fraction=pnl_pips * pip_size / price,
            exit_time=exit_time,
            regime_id=active_regime.regime_id,
        )
        open_trades.pop()

    return trade_results


def summarize(trades: List[TradeResult]):
    gross_profit = sum(max(t.pnl_pips, 0.0) for t in trades)
    gross_loss = sum(-min(t.pnl_pips, 0.0) for t in trades)
    wins = sum(1 for t in trades if t.pnl_pips > 0)
    losses = sum(1 for t in trades if t.pnl_pips < 0)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t.pnl_pips
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = wins / len(trades) * 100 if trades else 0.0
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_drawdown_pips": abs(max_dd),
        "net_pips": equity,
    }


def main():
    parser = argparse.ArgumentParser(description="Regime/Trigger backtest")
    parser.add_argument("--data", default="data/USDJPY_M1.csv")
    parser.add_argument("--pair", default="USDJPY", choices=TRADE_PAIRS)
    parser.add_argument("--th-long", type=float, default=0.55)
    parser.add_argument("--th-short", type=float, default=0.55)
    parser.add_argument("--rows", type=int, default=None, help="Limit to last N rows for quick runs")
    args = parser.parse_args()

    minute_df = _build_minute_df(args.data)
    if args.rows:
        minute_df = minute_df.tail(args.rows)
    lower = _resample_lower(minute_df)

    model_files, scaler_file = _resolve_model_artifacts(args.pair)
    scaler = _load_scaler(scaler_file)
    feature_dim = getattr(scaler, "n_features_in_", None)
    if feature_dim is None:
        raise SystemExit("Scaler missing n_features_in_")

    long_model = _load_model(model_files["long"], feature_dim)
    short_model = _load_model(model_files["short"], feature_dim)

    trades = run_backtest(
        minute_df,
        lower,
        scaler,
        long_model,
        short_model,
        pair=args.pair,
        th_long=args.th_long,
        th_short=args.th_short,
    )
    stats = summarize(trades)
    print(f"Trades: {stats['trades']} | Wins: {stats['wins']} | Losses: {stats['losses']}")
    print(f"Win Rate: {stats['win_rate']:.2f}% | Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Net Pips: {stats['net_pips']:.2f} | Max DD (pips): {stats['max_drawdown_pips']:.2f}")


if __name__ == "__main__":
    main()
