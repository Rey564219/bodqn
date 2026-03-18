from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch

from trade_core import (
    ATR_FAST_PERIOD,
    ATR_SLOW_PERIOD,
    build_exit_levels,
    calc_atr,
    calc_crypto_qty_fixed_risk,
    decide_entry_two_models,
    estimate_net_pnl,
    make_exit_params,
    softmax_probs,
)

try:
    from shared_features import build_state_vec_fast, compute_trend_direction
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "shared_features is required for feature extraction."
    ) from e

from train_dqn import QNet, TRADE_PAIRS


BASE_URL = os.getenv("BINANCE_FAPI_BASE_URL", "https://fapi.binance.com")
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

SYMBOL = os.getenv("BINANCE_SYMBOL", "BTCUSDT")
ENTRY_TH = float(os.getenv("ENTRY_TH", "0.55"))
REQUIRED_CANDLES = int(os.getenv("REQUIRED_CANDLES", "75"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.005"))
LEVERAGE = int(os.getenv("LEVERAGE", "5"))
SPREAD_PIPS = float(os.getenv("SPREAD_PIPS", "0.0"))
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.0004"))
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0001"))
ADD_ON_MIN_ADVERSE_PIPS = float(os.getenv("ADD_ON_MIN_ADVERSE_PIPS", "0.25"))
ADD_ON_REBOUND_CONFIRM_PIPS = float(os.getenv("ADD_ON_REBOUND_CONFIRM_PIPS", "0.12"))
ADD_ON_SIZE_RATIO = float(os.getenv("ADD_ON_SIZE_RATIO", "0.7"))
ADD_ON_MIN_EXTREME_UPDATES = int(os.getenv("ADD_ON_MIN_EXTREME_UPDATES", "2"))


def _normalize_pair_name(symbol: str) -> str:
    normalized = "".join(ch for ch in str(symbol).upper() if ch.isalnum())
    if normalized.endswith("USDT"):
        normalized = normalized[:-1]
    return normalized


def _resolve_model_artifacts(symbol: str):
    model_pair = _normalize_pair_name(symbol)
    if model_pair not in TRADE_PAIRS:
        raise SystemExit(f"Unsupported model pair '{model_pair}' from symbol '{symbol}'. Choose from {TRADE_PAIRS}")

    model_dir = os.path.join("Models", model_pair)
    model_files = {
        "long": os.path.join(model_dir, "dqn_policy_high.pt"),
        "short": os.path.join(model_dir, "dqn_policy_low.pt"),
    }
    scaler_file = os.path.join(model_dir, "dqn_scaler.pkl")

    missing = [p for p in [*model_files.values(), scaler_file] if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Model artifacts not found for {model_pair}: {missing}")

    return model_pair, model_files, scaler_file


class BinanceFuturesClient:
    def __init__(self, base_url: str, api_key: str, api_secret: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")

    def _sign(self, params: dict) -> dict:
        query = urlencode(params, doseq=True)
        signature = hmac.new(self.api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, path: str, params: Optional[dict] = None, signed: bool = False):
        params = params or {}
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            params = self._sign(params)
        query = urlencode(params, doseq=True)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        headers = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        req = Request(url, method=method, headers=headers)
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_exchange_info(self):
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def get_klines(self, symbol: str, interval: str, limit: int):
        return self._request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    def get_balance(self):
        return self._request("GET", "/fapi/v2/balance", signed=True)

    def set_leverage(self, symbol: str, leverage: int):
        return self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    def place_order(self, params: dict):
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_order(self, symbol: str, order_id: int):
        return self._request("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id}, signed=True)


class BinanceFuturesBot:
    def __init__(self):
        self.client = BinanceFuturesClient(BASE_URL, API_KEY, API_SECRET)
        self.model_pair, self.model_files, self.scaler_file = _resolve_model_artifacts(SYMBOL)
        print(f"[INFO] Using model artifacts for pair: {self.model_pair}")

        self.scaler = self._load_scaler(self.scaler_file)
        self.models = self._load_models(self.model_files)
        self.symbol_info = self._load_symbol_info()

        self.open_position = None
        self.tp_order_id = None
        self.sl_order_id = None
        self.last_primary_entry_minute = {}
        self.add_on_plan = None

        self.client.set_leverage(SYMBOL, LEVERAGE)

    def _minute_key(self, dt: datetime) -> datetime:
        return dt.replace(second=0, microsecond=0)

    def _build_add_on_plan(self, direction: str, entry_time: datetime, entry_price: float):
        self.add_on_plan = {
            "minute": self._minute_key(entry_time),
            "direction": direction,
            "entry_price": entry_price,
            "extreme_price": entry_price,
            "extreme_updates": 0,
            "added": False,
        }

    def _update_add_on_state(self, now: datetime, price: float, pip_size: float) -> bool:
        if not self.add_on_plan or self.add_on_plan.get("added"):
            return False
        if self._minute_key(now) != self.add_on_plan["minute"]:
            self.add_on_plan = None
            return False

        direction = self.add_on_plan["direction"]
        if direction == "LONG":
            if price < self.add_on_plan["extreme_price"]:
                self.add_on_plan["extreme_price"] = price
                self.add_on_plan["extreme_updates"] += 1
            adverse = (self.add_on_plan["entry_price"] - self.add_on_plan["extreme_price"]) / pip_size
            rebound = (price - self.add_on_plan["extreme_price"]) / pip_size
        else:
            if price > self.add_on_plan["extreme_price"]:
                self.add_on_plan["extreme_price"] = price
                self.add_on_plan["extreme_updates"] += 1
            adverse = (self.add_on_plan["extreme_price"] - self.add_on_plan["entry_price"]) / pip_size
            rebound = (self.add_on_plan["extreme_price"] - price) / pip_size

        return (
            adverse >= ADD_ON_MIN_ADVERSE_PIPS
            and rebound >= ADD_ON_REBOUND_CONFIRM_PIPS
            and self.add_on_plan["extreme_updates"] >= ADD_ON_MIN_EXTREME_UPDATES
        )

    def _load_scaler(self, scaler_file: str):
        import pickle

        with open(scaler_file, "rb") as f:
            return pickle.load(f)

    def _load_models(self, model_files: dict):
        models = {}
        for key, path in model_files.items():
            qnet = QNet(self.scaler.n_features_in_, 2)
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                qnet.load_state_dict(state["state_dict"])
            else:
                qnet.load_state_dict(state)
            qnet.eval()
            models[key] = qnet
        return models

    def _load_symbol_info(self) -> Tuple[float, float]:
        info = self.client.get_exchange_info()
        for sym in info.get("symbols", []):
            if sym.get("symbol") == SYMBOL:
                tick = float(next(f["tickSize"] for f in sym["filters"] if f["filterType"] == "PRICE_FILTER"))
                step = float(next(f["stepSize"] for f in sym["filters"] if f["filterType"] == "LOT_SIZE"))
                return tick, step
        raise RuntimeError(f"Symbol not found: {SYMBOL}")

    def _get_balance(self) -> float:
        data = self.client.get_balance()
        for row in data:
            if row.get("asset") == "USDT":
                return float(row.get("availableBalance", 0.0))
        return 0.0

    def _fetch_ohlc(self) -> pd.DataFrame:
        raw = self.client.get_klines(SYMBOL, "1m", REQUIRED_CANDLES)
        rows = []
        for item in raw:
            ts = datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc)
            rows.append(
                (ts, float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]))
            )
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        return df.set_index("ts")

    def _place_tp_sl(self, side: str, qty: float, tp_price: float, sl_price: float):
        tp_side = "SELL" if side == "LONG" else "BUY"
        sl_side = tp_side
        tp_params = {
            "symbol": SYMBOL,
            "side": tp_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": self._round_price(tp_price),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
        }
        sl_params = {
            "symbol": SYMBOL,
            "side": sl_side,
            "type": "STOP_MARKET",
            "stopPrice": self._round_price(sl_price),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
        }
        tp_res = self.client.place_order(tp_params)
        sl_res = self.client.place_order(sl_params)
        self.tp_order_id = tp_res.get("orderId")
        self.sl_order_id = sl_res.get("orderId")

    def _round_qty(self, qty: float) -> float:
        _, step = self.symbol_info
        if step <= 0:
            return qty
        return float(np.floor(qty / step) * step)

    def _round_price(self, price: float) -> float:
        tick, _ = self.symbol_info
        if tick <= 0:
            return price
        return float(np.round(price / tick) * tick)

    def _close_market(self, side: str, qty: float):
        close_side = "SELL" if side == "LONG" else "BUY"
        params = {
            "symbol": SYMBOL,
            "side": close_side,
            "type": "MARKET",
            "quantity": self._round_qty(qty),
            "reduceOnly": "true",
        }
        self.client.place_order(params)

    def _cancel_exit_orders(self):
        if self.tp_order_id:
            self.client.cancel_order(SYMBOL, self.tp_order_id)
            self.tp_order_id = None
        if self.sl_order_id:
            self.client.cancel_order(SYMBOL, self.sl_order_id)
            self.sl_order_id = None

    def run(self):
        while True:
            time.sleep(2)
            df = self._fetch_ohlc()
            last_ts = df.index[-1]

            latest = df.iloc[-1]
            entry_time = last_ts.to_pydatetime().replace(tzinfo=None)
            minute_key = self._minute_key(entry_time)

            atr_fast = calc_atr(df, ATR_FAST_PERIOD).iloc[-1]
            atr_slow = calc_atr(df, ATR_SLOW_PERIOD).iloc[-1]
            tick_size, _ = self.symbol_info
            atr_fast_pips = atr_fast / tick_size
            atr_slow_pips = atr_slow / tick_size
            exit_params = make_exit_params(atr_fast_pips, atr_slow_pips, SPREAD_PIPS)

            if self.open_position and entry_time >= self.open_position.timeout_time:
                self._cancel_exit_orders()
                self._close_market(self.open_position.side, self.open_position.qty)
                pnl = estimate_net_pnl(
                    self.open_position.side,
                    self.open_position.entry_price,
                    float(latest["close"]),
                    self.open_position.qty,
                    fee_rate=TAKER_FEE,
                    slippage_rate=SLIPPAGE_RATE,
                )
                print(f"[EXIT] TIMEOUT PnL~ {pnl:.4f}")
                self.open_position = None
                self.add_on_plan = None
                continue

            if self.open_position:
                if self._update_add_on_state(entry_time, float(latest["close"]), tick_size):
                    add_qty = self._round_qty(self.open_position.qty * max(0.05, min(1.0, ADD_ON_SIZE_RATIO)))
                    add_qty = self._round_qty(add_qty)
                    if add_qty > 0:
                        side = "BUY" if self.open_position.side == "LONG" else "SELL"
                        add_params = {
                            "symbol": SYMBOL,
                            "side": side,
                            "type": "MARKET",
                            "quantity": add_qty,
                        }
                        self.client.place_order(add_params)
                        total_qty = self.open_position.qty + add_qty
                        self.open_position.entry_price = (
                            (self.open_position.entry_price * self.open_position.qty) + (float(latest["close"]) * add_qty)
                        ) / total_qty
                        self.open_position.qty = total_qty
                        self.add_on_plan["added"] = True
                        print(f"[ENTRY-ADD] {self.open_position.side} rebound within minute qty={add_qty}")
                continue

            phase = 1.0
            sec_range = float(latest["high"] - latest["low"])
            state_vec = build_state_vec_fast(df.tail(REQUIRED_CANDLES), phase, sec_range)
            state_vec = self.scaler.transform([state_vec])[0].astype(np.float32)

            with torch.no_grad():
                t = torch.from_numpy(state_vec).unsqueeze(0).float()
                q_long = self.models["long"](t).cpu().numpy().reshape(-1)
                q_short = self.models["short"](t).cpu().numpy().reshape(-1)
            p_long = float(softmax_probs(q_long)[1])
            p_short = float(softmax_probs(q_short)[1])

            if not exit_params.trade_allowed:
                continue

            decision = decide_entry_two_models(p_long, p_short, ENTRY_TH)
            if decision == "HOLD":
                continue

            if self.last_primary_entry_minute.get(decision) == minute_key:
                continue

            trend_dir = compute_trend_direction(df["close"], window=REQUIRED_CANDLES)
            if decision == "LONG" and trend_dir < 0:
                continue
            if decision == "SHORT" and trend_dir > 0:
                continue

            position = build_exit_levels(
                decision,
                float(latest["close"]),
                tick_size,
                exit_params,
                entry_time,
            )

            balance = self._get_balance()
            qty = calc_crypto_qty_fixed_risk(
                balance=balance,
                risk_pct=RISK_PCT,
                entry_price=position.entry_price,
                sl_price=position.sl_price,
                leverage=LEVERAGE,
                min_qty=0.001,
                qty_step=self.symbol_info[1],
            )
            qty = self._round_qty(qty)
            if qty <= 0:
                continue

            side = "BUY" if position.side == "LONG" else "SELL"
            entry_params = {
                "symbol": SYMBOL,
                "side": side,
                "type": "MARKET",
                "quantity": qty,
            }
            self.client.place_order(entry_params)
            self._place_tp_sl(position.side, qty, position.tp_price, position.sl_price)
            position.qty = qty
            self.open_position = position
            self.last_primary_entry_minute[position.side] = minute_key
            self._build_add_on_plan(position.side, entry_time, position.entry_price)
            print(
                f"[ENTRY] {position.side} price={position.entry_price:.2f} TP={position.tp_price:.2f} "
                f"SL={position.sl_price:.2f} qty={qty}"
            )


if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        raise SystemExit("Set BINANCE_API_KEY and BINANCE_API_SECRET.")
    bot = BinanceFuturesBot()
    bot.run()
