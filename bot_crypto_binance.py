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
    calibrate_tp_k,
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


BASE_URL = os.getenv("MEXC_FUTURES_BASE_URL", "https://api.mexc.com")
API_KEY = os.getenv("MEXC_API_KEY", os.getenv("BINANCE_API_KEY", ""))
API_SECRET = os.getenv("MEXC_API_SECRET", os.getenv("BINANCE_API_SECRET", ""))

def _to_mexc_symbol(raw_symbol: str) -> str:
    s = str(raw_symbol or "").upper().strip()
    if "_" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}_USDT"
    return s


SYMBOL = _to_mexc_symbol(os.getenv("MEXC_SYMBOL", os.getenv("BINANCE_SYMBOL", "BTC_USDT")))
ENTRY_TH = float(os.getenv("ENTRY_TH", "0.55"))
REQUIRED_CANDLES = int(os.getenv("REQUIRED_CANDLES", "75"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.005"))
LEVERAGE = int(os.getenv("LEVERAGE", "5"))
MEXC_OPEN_TYPE = int(os.getenv("MEXC_OPEN_TYPE", "2"))  # 1: isolated, 2: cross
MEXC_POSITION_MODE = int(os.getenv("MEXC_POSITION_MODE", "2"))  # 1: hedged, 2: one-way
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


class MexcFuturesClient:
    def __init__(self, base_url: str, api_key: str, api_secret: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")

    def _sign(self, timestamp_ms: str, method: str, params: Optional[dict], body: Optional[dict]) -> str:
        if method in ("GET", "DELETE"):
            sign_params = {k: v for k, v in (params or {}).items() if v is not None}
            param_str = urlencode(sorted(sign_params.items()), doseq=True)
        else:
            if isinstance(body, list):
                clean_body = body
            else:
                clean_body = {k: v for k, v in (body or {}).items() if v is not None}
            param_str = json.dumps(clean_body, separators=(",", ":"), ensure_ascii=False)
        target = f"{self.api_key}{timestamp_ms}{param_str}"
        return hmac.new(self.api_secret, target.encode("utf-8"), hashlib.sha256).hexdigest()

    def _request(self, method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None, signed: bool = False):
        params = params or {}
        url = f"{self.base_url}{path}"
        query = urlencode(params, doseq=True)
        if query:
            url = f"{url}?{query}"
        headers = {}
        payload = None
        if method == "POST":
            if isinstance(body, list):
                clean_body = body
            else:
                clean_body = {k: v for k, v in (body or {}).items() if v is not None}
            payload = json.dumps(clean_body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        if signed:
            timestamp_ms = str(int(time.time() * 1000))
            headers["ApiKey"] = self.api_key
            headers["Request-Time"] = timestamp_ms
            headers["Signature"] = self._sign(timestamp_ms, method, params, body)
            headers["Recv-Window"] = "10000"

        req = Request(url, method=method, headers=headers, data=payload)
        with urlopen(req, timeout=10) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        if isinstance(raw, dict) and "success" in raw:
            if not raw.get("success"):
                raise RuntimeError(f"MEXC API error code={raw.get('code')} message={raw.get('message')}")
            return raw.get("data")
        return raw

    def get_exchange_info(self):
        return self._request("GET", "/api/v1/contract/detail", params={"symbol": SYMBOL})

    def get_klines(self, symbol: str, interval: str, limit: int):
        end_ts = int(time.time())
        start_ts = max(0, end_ts - int(max(1, limit) * 60))
        return self._request(
            "GET",
            f"/api/v1/contract/kline/{symbol}",
            params={"interval": interval, "start": start_ts, "end": end_ts},
        )

    def get_balance(self):
        return self._request("GET", "/api/v1/private/account/assets", signed=True)

    def set_leverage(self, symbol: str, leverage: int):
        # MEXC keeps leverage per side; apply both for compatibility.
        self._request(
            "POST",
            "/api/v1/private/position/change_leverage",
            body={"symbol": symbol, "leverage": int(leverage), "openType": MEXC_OPEN_TYPE, "positionType": 1},
            signed=True,
        )
        return self._request(
            "POST",
            "/api/v1/private/position/change_leverage",
            body={"symbol": symbol, "leverage": int(leverage), "openType": MEXC_OPEN_TYPE, "positionType": 2},
            signed=True,
        )

    def place_order(self, params: dict):
        return self._request("POST", "/api/v1/private/order/create", body=params, signed=True)

    def cancel_order(self, symbol: str, order_id: int):
        return self._request("POST", "/api/v1/private/order/cancel", body=[int(order_id)], signed=True)


class MexcFuturesBot:
    def __init__(self):
        self.client = MexcFuturesClient(BASE_URL, API_KEY, API_SECRET)
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
        self.last_price = 0.0

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

    def _load_symbol_info(self) -> Tuple[float, float, float, float]:
        info = self.client.get_exchange_info()
        if isinstance(info, list):
            for sym in info:
                if sym.get("symbol") == SYMBOL:
                    tick = float(sym.get("priceUnit", 0.0) or 0.0)
                    step = float(sym.get("volUnit", 1.0) or 1.0)
                    min_vol = float(sym.get("minVol", 1.0) or 1.0)
                    contract_size = float(sym.get("contractSize", 1.0) or 1.0)
                    return tick, step, min_vol, contract_size
        elif isinstance(info, dict) and info.get("symbol") == SYMBOL:
            tick = float(info.get("priceUnit", 0.0) or 0.0)
            step = float(info.get("volUnit", 1.0) or 1.0)
            min_vol = float(info.get("minVol", 1.0) or 1.0)
            contract_size = float(info.get("contractSize", 1.0) or 1.0)
            return tick, step, min_vol, contract_size
        raise RuntimeError(f"Symbol not found in MEXC contract detail: {SYMBOL}")

    def _get_balance(self) -> float:
        data = self.client.get_balance()
        for row in data:
            if row.get("asset") == "USDT":
                return float(row.get("availableBalance", 0.0))
            if row.get("currency") == "USDT":
                return float(row.get("availableBalance", 0.0))
        return 0.0

    def _fetch_ohlc(self) -> pd.DataFrame:
        raw = self.client.get_klines(SYMBOL, "Min1", REQUIRED_CANDLES)
        times = raw.get("time", [])
        opens = raw.get("open", [])
        highs = raw.get("high", [])
        lows = raw.get("low", [])
        closes = raw.get("close", [])
        vols = raw.get("vol", [])
        rows = []
        for ts_s, o, h, l, c, v in zip(times, opens, highs, lows, closes, vols):
            ts = datetime.fromtimestamp(float(ts_s), tz=timezone.utc)
            rows.append((ts, float(o), float(h), float(l), float(c), float(v)))
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        return df.set_index("ts")

    def _place_tp_sl(self, side: str, qty: float, tp_price: float, sl_price: float):
        # TP/SL is attached to the entry order on MEXC; keep this as a no-op for compatibility.
        return

    def _round_qty(self, qty: float) -> float:
        _, step, min_vol, _ = self.symbol_info
        if step <= 0:
            return max(qty, min_vol)
        rounded = float(np.floor(qty / step) * step)
        return max(rounded, min_vol)

    def _round_price(self, price: float) -> float:
        tick = self.symbol_info[0]
        if tick <= 0:
            return price
        return float(np.round(price / tick) * tick)

    def _qty_base_to_contracts(self, qty_base: float) -> float:
        _, vol_step, min_vol, contract_size = self.symbol_info
        if contract_size <= 0:
            return self._round_qty(qty_base)
        contracts = qty_base / contract_size
        contracts = self._round_qty(contracts)
        return max(contracts, min_vol)

    def _contracts_to_qty_base(self, contracts: float) -> float:
        contract_size = self.symbol_info[3]
        return float(contracts) * float(contract_size)

    def _close_market(self, side: str, qty: float):
        # MEXC side mapping: 2=close short, 4=close long
        close_side = 4 if side == "LONG" else 2
        vol = self._qty_base_to_contracts(qty)
        params = {
            "symbol": SYMBOL,
            "side": close_side,
            "type": 5,
            "openType": MEXC_OPEN_TYPE,
            "positionMode": MEXC_POSITION_MODE,
            "vol": vol,
            "price": self._round_price(self.last_price) if self.last_price > 0 else None,
            "reduceOnly": True,
        }
        self.client.place_order(params)

    def _cancel_exit_orders(self):
        return

    def run(self):
        while True:
            time.sleep(2)
            df = self._fetch_ohlc()
            last_ts = df.index[-1]

            latest = df.iloc[-1]
            self.last_price = float(latest["close"])
            entry_time = last_ts.to_pydatetime().replace(tzinfo=None)
            minute_key = self._minute_key(entry_time)

            atr_fast = calc_atr(df, ATR_FAST_PERIOD).iloc[-1]
            atr_slow = calc_atr(df, ATR_SLOW_PERIOD).iloc[-1]
            tick_size = self.symbol_info[0]
            atr_fast_pips = atr_fast / tick_size
            atr_slow_pips = atr_slow / tick_size
            tp_k = calibrate_tp_k(
                df,
                pip_size=tick_size,
                horizon_min=10,
            )
            exit_params = make_exit_params(
                atr_fast_pips,
                atr_slow_pips,
                SPREAD_PIPS,
                tp_k=tp_k,
            )

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
                        side = 1 if self.open_position.side == "LONG" else 3
                        add_contracts = self._qty_base_to_contracts(add_qty)
                        if add_contracts <= 0:
                            continue
                        add_params = {
                            "symbol": SYMBOL,
                            "side": side,
                            "type": 5,
                            "openType": MEXC_OPEN_TYPE,
                            "positionMode": MEXC_POSITION_MODE,
                            "vol": add_contracts,
                            "price": self._round_price(self.last_price) if self.last_price > 0 else None,
                            "leverage": LEVERAGE,
                        }
                        self.client.place_order(add_params)
                        add_qty_base = self._contracts_to_qty_base(add_contracts)
                        total_qty = self.open_position.qty + add_qty_base
                        self.open_position.entry_price = (
                            (self.open_position.entry_price * self.open_position.qty) + (float(latest["close"]) * add_qty_base)
                        ) / total_qty
                        self.open_position.qty = total_qty
                        self.add_on_plan["added"] = True
                        print(f"[ENTRY-ADD] {self.open_position.side} rebound within minute contracts={add_contracts}")
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
                min_qty=self._contracts_to_qty_base(self.symbol_info[2]),
                qty_step=self._contracts_to_qty_base(self.symbol_info[1]),
            )
            contracts = self._qty_base_to_contracts(qty)
            qty = self._contracts_to_qty_base(contracts)
            if qty <= 0:
                continue

            side = 1 if position.side == "LONG" else 3
            entry_params = {
                "symbol": SYMBOL,
                "side": side,
                "type": 5,
                "openType": MEXC_OPEN_TYPE,
                "positionMode": MEXC_POSITION_MODE,
                "vol": contracts,
                "price": self._round_price(position.entry_price),
                "leverage": LEVERAGE,
                "takeProfitPrice": self._round_price(position.tp_price),
                "stopLossPrice": self._round_price(position.sl_price),
                "profitTrend": 1,
                "lossTrend": 1,
            }
            self.client.place_order(entry_params)
            self._place_tp_sl(position.side, qty, position.tp_price, position.sl_price)
            position.qty = qty
            self.open_position = position
            self.last_primary_entry_minute[position.side] = minute_key
            self._build_add_on_plan(position.side, entry_time, position.entry_price)
            print(
                f"[ENTRY] {position.side} price={position.entry_price:.2f} TP={position.tp_price:.2f} "
                f"SL={position.sl_price:.2f} contracts={contracts}"
            )


if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        raise SystemExit("Set MEXC_API_KEY and MEXC_API_SECRET.")
    bot = MexcFuturesBot()
    bot.run()
