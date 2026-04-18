from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import torch

from trade_core import (
    ATR_FAST_PERIOD,
    ATR_SLOW_PERIOD,
    ExitParams,
    build_exit_levels,
    calibrate_tp_k,
    calc_atr,
    calc_fx_lots_fixed_risk,
    make_exit_params,
    round_step,
)

try:
    from ctrader_open_api import Client, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoHeartbeatEvent
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAAccountAuthReq,
        ProtoOAAccountAuthRes,
        ProtoOAApplicationAuthReq,
        ProtoOAApplicationAuthRes,
        ProtoOAExecutionEvent,
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAGetAccountListByAccessTokenRes,
        ProtoOAGetTrendbarsReq,
        ProtoOAGetTrendbarsRes,
        ProtoOANewOrderReq,
        ProtoOASymbolByIdReq,
        ProtoOASymbolByIdRes,
        ProtoOASymbolsListReq,
        ProtoOASymbolsListRes,
        ProtoOATraderReq,
        ProtoOATraderRes,
        ProtoOAClosePositionReq,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
        ProtoOAOrderType,
        ProtoOATradeSide,
        ProtoOATrendbarPeriod,
    )
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "ctrader-open-api is required. Install with `pip install ctrader-open-api`."
    ) from e

try:
    from shared_features import compute_trend_direction
    from regime_executor import MeanReversionTrigger, RegimeDecision, RegimeDecider, RiskManager
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "shared_features is required for feature extraction."
    ) from e

from train_dqn import QNet, TRADE_PAIRS

ENTRY_TH = float(os.getenv("ENTRY_TH", "0.55"))
REQUIRED_CANDLES = int(os.getenv("REQUIRED_CANDLES", "75"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.005"))
LEVERAGE = float(os.getenv("LEVERAGE", "5"))
SPREAD_PIPS = float(os.getenv("SPREAD_PIPS", "0.2"))

CTRADER_CLIENT_ID = os.getenv("CTRADER_CLIENT_ID", "")
CTRADER_CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET", "")
CTRADER_ACCESS_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN", "")
CTRADER_ACCOUNT_ID = os.getenv("CTRADER_ACCOUNT_ID", "")
CTRADER_HOST_TYPE = os.getenv("CTRADER_HOST_TYPE", "live").lower()
CTRADER_SYMBOL = os.getenv("CTRADER_SYMBOL", "USDJPY")

TH_LONG = float(os.getenv("TH_LONG", str(ENTRY_TH)))
TH_SHORT = float(os.getenv("TH_SHORT", str(ENTRY_TH)))
MAX_POSITIONS_PER_SIDE = int(os.getenv("MAX_POSITIONS_PER_SIDE", "2"))
MAX_ENTRIES_PER_MINUTE = int(os.getenv("MAX_ENTRIES_PER_MINUTE", "3"))
REGIME_LOSS_LIMIT = float(os.getenv("REGIME_LOSS_LIMIT", "0.004"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.02"))
CONSECUTIVE_LOSS_LIMIT = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "3"))
MEAN_REV_FAST = int(os.getenv("MEAN_REV_FAST", "8"))
MEAN_REV_SLOW = int(os.getenv("MEAN_REV_SLOW", "34"))
MEAN_REV_DEV_PIPS = float(os.getenv("MEAN_REV_DEV_PIPS", "0.35"))
LOWER_TRIGGER_COOLDOWN = float(os.getenv("LOWER_TRIGGER_COOLDOWN", "5"))
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


@dataclass
class SymbolInfo:
    symbol_id: int
    digits: int
    pip_position: int
    lot_size: int
    min_volume: int
    step_volume: int

    @property
    def pip_size(self) -> float:
        return 10 ** (-self.pip_position)

    @property
    def pip_value_per_lot(self) -> float:
        return self.lot_size * self.pip_size


class AxioryCTraderBot:
    def __init__(self):
        host = EndPoints.PROTOBUF_LIVE_HOST if CTRADER_HOST_TYPE == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        self.client.setMessageReceivedCallback(self._on_message)

        self.account_id = int(CTRADER_ACCOUNT_ID) if CTRADER_ACCOUNT_ID else None
        self.symbol_info: Optional[SymbolInfo] = None
        self.balance = 0.0
        self.ohlc_df: Optional[pd.DataFrame] = None
        self.last_bar_time: Optional[datetime] = None
        self.last_trendbar_request = 0.0

        self.open_positions: List[PositionState] = []
        self.active_regime: Optional[RegimeDecision] = None
        self.last_regime_timestamp: Optional[datetime] = None
        self.last_primary_entry_minute = {}
        self.add_on_plan: Optional[dict] = None
        self.trigger_engine = MeanReversionTrigger(
            fast_window=MEAN_REV_FAST,
            slow_window=MEAN_REV_SLOW,
            deviation_pips=MEAN_REV_DEV_PIPS,
        )
        self.risk_manager = RiskManager(
            max_positions_per_side=MAX_POSITIONS_PER_SIDE,
            max_entries_per_minute=MAX_ENTRIES_PER_MINUTE,
            entry_cooldown_seconds=LOWER_TRIGGER_COOLDOWN,
            per_regime_loss_limit=REGIME_LOSS_LIMIT,
            daily_loss_limit=DAILY_LOSS_LIMIT,
            consecutive_loss_limit=CONSECUTIVE_LOSS_LIMIT,
            max_spread_pips=SPREAD_PIPS * 2,
        )
        self.regime_decider: Optional[RegimeDecider] = None

        self.model_pair, self.model_files, self.scaler_file = _resolve_model_artifacts(CTRADER_SYMBOL)
        print(f"[INFO] Using model artifacts for pair: {self.model_pair}")

        self.scaler = self._load_scaler(self.scaler_file)
        self.models = self._load_models(self.model_files)
        if self.scaler and self.models:
            self.regime_decider = RegimeDecider(
                scaler=self.scaler,
                long_model=self.models["long"],
                short_model=self.models["short"],
                required_candles=REQUIRED_CANDLES,
                th_buy=TH_LONG,
                th_sell=TH_SHORT,
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

    def start(self):
        self.client.startService()
        self._run_loop()

    def _run_loop(self):
        from twisted.internet import reactor
        reactor.run()

    def _on_connected(self, _client):
        req = ProtoOAApplicationAuthReq()
        req.clientId = CTRADER_CLIENT_ID
        req.clientSecret = CTRADER_CLIENT_SECRET
        self.client.send(req).addErrback(self._on_error)

    def _on_disconnected(self, _client, reason):
        print(f"[WARN] Disconnected: {reason}")

    def _on_error(self, failure):
        print(f"[ERROR] {failure}")

    def _on_message(self, _client, message):
        if message.payloadType == ProtoHeartbeatEvent().payloadType:
            return
        if message.payloadType == ProtoOAApplicationAuthRes().payloadType:
            self._handle_app_auth()
            return
        if message.payloadType == ProtoOAAccountAuthRes().payloadType:
            self._handle_account_auth()
            return
        if message.payloadType == ProtoOAGetAccountListByAccessTokenRes().payloadType:
            self._handle_account_list(message)
            return
        if message.payloadType == ProtoOASymbolsListRes().payloadType:
            self._handle_symbols_list(message)
            return
        if message.payloadType == ProtoOASymbolByIdRes().payloadType:
            self._handle_symbol_details(message)
            return
        if message.payloadType == ProtoOATraderRes().payloadType:
            self._handle_trader(message)
            return
        if message.payloadType == ProtoOAGetTrendbarsRes().payloadType:
            self._handle_trendbars(message)
            return
        if message.payloadType == ProtoOAExecutionEvent().payloadType:
            self._handle_execution_event(message)
            return

    def _handle_app_auth(self):
        if self.account_id is None:
            req = ProtoOAGetAccountListByAccessTokenReq()
            req.accessToken = CTRADER_ACCESS_TOKEN
            self.client.send(req).addErrback(self._on_error)
            return
        self._send_account_auth()

    def _handle_account_list(self, message):
        accounts = list(message.ctidTraderAccount)
        if not accounts:
            print("[ERROR] No accounts available for access token.")
            return
        self.account_id = int(accounts[0].ctidTraderAccountId)
        self._send_account_auth()

    def _send_account_auth(self):
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = int(self.account_id)
        req.accessToken = CTRADER_ACCESS_TOKEN
        self.client.send(req).addErrback(self._on_error)

    def _handle_account_auth(self):
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = int(self.account_id)
        self.client.send(req).addErrback(self._on_error)

        trader_req = ProtoOATraderReq()
        trader_req.ctidTraderAccountId = int(self.account_id)
        self.client.send(trader_req).addErrback(self._on_error)

    def _handle_trader(self, message):
        trader = message.trader
        money_digits = trader.moneyDigits if trader.moneyDigits else 0
        self.balance = float(trader.balance) / (10 ** money_digits)

    def _handle_symbols_list(self, message):
        for sym in message.symbol:
            if sym.symbolName == CTRADER_SYMBOL:
                req = ProtoOASymbolByIdReq()
                req.ctidTraderAccountId = int(self.account_id)
                req.symbolId.append(sym.symbolId)
                self.client.send(req).addErrback(self._on_error)
                return
        print(f"[ERROR] Symbol not found: {CTRADER_SYMBOL}")

    def _handle_symbol_details(self, message):
        if not message.symbol:
            return
        sym = message.symbol[0]
        self.symbol_info = SymbolInfo(
            symbol_id=sym.symbolId,
            digits=sym.digits,
            pip_position=sym.pipPosition,
            lot_size=sym.lotSize,
            min_volume=sym.minVolume,
            step_volume=sym.stepVolume,
        )
        self._request_trendbars()

    def _request_trendbars(self):
        if not self.symbol_info:
            return
        now = time.time()
        if now - self.last_trendbar_request < 5:
            return
        self.last_trendbar_request = now
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = int(self.account_id)
        req.symbolId = int(self.symbol_info.symbol_id)
        req.period = ProtoOATrendbarPeriod.Value("M1")
        req.count = REQUIRED_CANDLES
        self.client.send(req).addErrback(self._on_error)

    def _handle_trendbars(self, message):
        if not self.symbol_info:
            return
        df = self._trendbars_to_df(message.trendbar, self.symbol_info.digits)
        if df.empty:
            return
        self.ohlc_df = df
        last_ts = df.index[-1]
        if self.last_bar_time is None or last_ts > self.last_bar_time:
            self.last_bar_time = last_ts
        self._maybe_trade()
        self._request_trendbars()

    def _minute_key(self, dt: datetime) -> datetime:
        return dt.replace(second=0, microsecond=0)

    def _clear_add_on(self):
        self.add_on_plan = None

    def _update_add_on_state(self, now: datetime, price: float, pip_size: float) -> Optional[str]:
        if not self.add_on_plan:
            return None
        if self.add_on_plan.get("added"):
            return None
        if self._minute_key(now) != self.add_on_plan["minute"]:
            self._clear_add_on()
            return None

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

        if (
            adverse >= ADD_ON_MIN_ADVERSE_PIPS
            and rebound >= ADD_ON_REBOUND_CONFIRM_PIPS
            and self.add_on_plan["extreme_updates"] >= ADD_ON_MIN_EXTREME_UPDATES
        ):
            return direction
        return None

    def _trendbars_to_df(self, trendbars, digits: int) -> pd.DataFrame:
        scale = 10 ** digits
        rows = []
        for tb in trendbars:
            low = tb.low / scale
            open_p = (tb.low + tb.deltaOpen) / scale
            close_p = (tb.low + tb.deltaClose) / scale
            high_p = (tb.low + tb.deltaHigh) / scale
            ts = datetime.fromtimestamp(tb.utcTimestampInMinutes * 60, tz=timezone.utc)
            rows.append((ts, open_p, high_p, low, close_p, tb.volume))
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset=["ts"]).set_index("ts").sort_index()
        return df

    def _maybe_trade(self):
        if self.ohlc_df is None or len(self.ohlc_df) < REQUIRED_CANDLES:
            return
        if not self.symbol_info:
            return

        entry_time = self.ohlc_df.index[-1].to_pydatetime().replace(tzinfo=None)
        latest = self.ohlc_df.iloc[-1]
        close_price = float(latest["close"])
        minute_key = self._minute_key(entry_time)

        atr_fast = calc_atr(self.ohlc_df, ATR_FAST_PERIOD).iloc[-1]
        atr_slow = calc_atr(self.ohlc_df, ATR_SLOW_PERIOD).iloc[-1]
        pip_size = self.symbol_info.pip_size
        atr_fast_pips = atr_fast / pip_size
        atr_slow_pips = atr_slow / pip_size
        tp_k = calibrate_tp_k(
            self.ohlc_df,
            pip_size=pip_size,
            horizon_min=10,
        )
        exit_params = make_exit_params(
            atr_fast_pips,
            atr_slow_pips,
            SPREAD_PIPS,
            tp_k=tp_k,
        )
        if not exit_params.trade_allowed:
            return

        # timeout handling
        for position in list(self.open_positions):
            if entry_time >= position.timeout_time:
                self._close_position_market(position)

        # expire regime window
        if self.active_regime and entry_time >= self.active_regime.expires_at:
            self.active_regime = None
            self.trigger_engine.reset()
            self.risk_manager.on_regime_change(None)
            self._clear_add_on()

        # evaluate regime once per completed bar
        if self.regime_decider and (
            self.last_regime_timestamp is None or entry_time > self.last_regime_timestamp
        ):
            sec_range = float(latest["high"] - latest["low"])
            regime = self.regime_decider.evaluate(
                self.ohlc_df,
                timestamp=entry_time,
                phase=1.0,
                sec_range=sec_range,
            )
            self.active_regime = regime
            self.last_regime_timestamp = entry_time
            self.trigger_engine.on_regime_change(self.active_regime)
            regime_id = regime.regime_id if regime else None
            self.risk_manager.on_regime_change(regime_id)
            if regime:
                print(
                    f"[REGIME] {regime.regime} p_buy={regime.p_buy:.3f} p_sell={regime.p_sell:.3f} valid_until={regime.expires_at}"
                )

        if not self.active_regime or self.active_regime.regime == "NO_TRADE":
            self._clear_add_on()
            return

        add_on_direction = self._update_add_on_state(entry_time, close_price, pip_size)
        if add_on_direction:
            add_on_risk = self.risk_manager.can_enter(
                add_on_direction,
                current_time=entry_time,
                open_positions=self.open_positions,
                regime_id=self.active_regime.regime_id if self.active_regime else None,
                spread_pips=SPREAD_PIPS,
                ignore_cooldown=True,
            )
            if add_on_risk.allowed:
                add_position = build_exit_levels(
                    add_on_direction,
                    close_price,
                    pip_size,
                    exit_params,
                    entry_time,
                )
                add_position.regime_id = self.active_regime.regime_id if self.active_regime else None
                self._open_position(add_position, exit_params, lot_scale=ADD_ON_SIZE_RATIO)
                self.add_on_plan["added"] = True
                print(f"[ENTRY-ADD] {add_on_direction} rebound within minute @ {close_price:.5f}")

        self.trigger_engine.update_price(entry_time, close_price)
        signal = self.trigger_engine.check(
            self.active_regime,
            pip_size=pip_size,
            now=entry_time,
            cooldown_seconds=LOWER_TRIGGER_COOLDOWN,
        )
        if not signal:
            return

        if self.last_primary_entry_minute.get(signal.direction) == minute_key:
            return

        trend_dir = compute_trend_direction(self.ohlc_df["close"], window=REQUIRED_CANDLES)
        if signal.direction == "LONG" and trend_dir < 0:
            return
        if signal.direction == "SHORT" and trend_dir > 0:
            return

        risk = self.risk_manager.can_enter(
            signal.direction,
            current_time=entry_time,
            open_positions=self.open_positions,
            regime_id=self.active_regime.regime_id if self.active_regime else None,
            spread_pips=SPREAD_PIPS,
        )
        if not risk.allowed:
            print(f"[RISK] Blocked {signal.direction}: {risk.reason}")
            return

        position = build_exit_levels(
            signal.direction,
            close_price,
            pip_size,
            exit_params,
            entry_time,
        )
        position.regime_id = self.active_regime.regime_id if self.active_regime else None
        self._open_position(position, exit_params)
        self.last_primary_entry_minute[signal.direction] = minute_key
        self.add_on_plan = {
            "minute": minute_key,
            "direction": signal.direction,
            "entry_price": close_price,
            "extreme_price": close_price,
            "extreme_updates": 0,
            "added": False,
        }

    def _open_position(self, position, exit_params: ExitParams, lot_scale: float = 1.0):
        if not self.symbol_info:
            return
        min_lot = max(0.01, self.symbol_info.min_volume / 100.0)
        lot_step = max(0.01, self.symbol_info.step_volume / 100.0)
        lots = calc_fx_lots_fixed_risk(
            balance=self.balance,
            risk_pct=RISK_PCT,
            sl_pips=exit_params.sl_pips,
            pip_value_per_lot=self.symbol_info.pip_value_per_lot,
            min_lot=min_lot,
            lot_step=lot_step,
        )
        scale = max(0.05, min(1.0, float(lot_scale)))
        lots = max(min_lot, round_step(lots * scale, lot_step))
        volume_units = int(round(lots * 100))
        if volume_units <= 0:
            return

        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = int(self.account_id)
        req.symbolId = int(self.symbol_info.symbol_id)
        req.orderType = ProtoOAOrderType.Value("MARKET")
        req.tradeSide = ProtoOATradeSide.Value("BUY" if position.side == "LONG" else "SELL")
        req.volume = volume_units
        req.stopLoss = position.sl_price
        req.takeProfit = position.tp_price

        self.client.send(req).addErrback(self._on_error)
        position.volume_units = volume_units
        self.open_positions.append(position)
        self.risk_manager.register_entry(position.entry_time)
        print(
            f"[ENTRY] {position.side} price={position.entry_price:.5f} TP={position.tp_price:.5f} "
            f"SL={position.sl_price:.5f} lots={lots:.2f}"
        )

    def _close_position_market(self, position: Optional[PositionState] = None):
        if position is None:
            if not self.open_positions:
                return
            position = self.open_positions[0]
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = int(self.account_id)
        if position.position_id:
            req.positionId = int(position.position_id)
        req.volume = getattr(position, "volume_units", 0)
        self.client.send(req).addErrback(self._on_error)
        print(f"[EXIT] Closing {position.side} by market (timeout)")
        if position in self.open_positions:
            self.open_positions.remove(position)

    def _handle_execution_event(self, message):
        if not self.open_positions:
            return
        position = message.position
        if not position or not position.positionId:
            return
        for local in list(self.open_positions):
            if local.position_id and local.position_id != position.positionId:
                continue
            local.position_id = position.positionId
            if position.positionStatus == 2:  # closed
                print(f"[EXIT] Broker closed position {position.positionId}")
                self.open_positions.remove(local)
            break


if __name__ == "__main__":
    if not CTRADER_CLIENT_ID or not CTRADER_CLIENT_SECRET:
        raise SystemExit("Set CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET.")
    if not CTRADER_ACCESS_TOKEN:
        raise SystemExit("Set CTRADER_ACCESS_TOKEN.")
    bot = AxioryCTraderBot()
    bot.start()
