from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import torch

from trade_core import (
    ATR_FAST_PERIOD,
    ATR_SLOW_PERIOD,
    ExitParams,
    build_exit_levels,
    calc_atr,
    calc_fx_lots_fixed_risk,
    decide_entry_two_models,
    make_exit_params,
    softmax_probs,
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
    from shared_features import build_state_vec_fast, compute_trend_direction
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "shared_features is required for feature extraction."
    ) from e

from train_dqn import QNet


MODEL_FILES = {
    "long": "./Models/dqn_policy_high.pt",
    "short": "./Models/dqn_policy_low.pt",
}
SCALER_FILE = "./Models/dqn_scaler.pkl"

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

        self.open_position = None
        self.pending_exit_params: Optional[ExitParams] = None

        self.scaler = self._load_scaler()
        self.models = self._load_models()

    def _load_scaler(self):
        import pickle

        with open(SCALER_FILE, "rb") as f:
            return pickle.load(f)

    def _load_models(self):
        models = {}
        for key, path in MODEL_FILES.items():
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

        atr_fast = calc_atr(self.ohlc_df, ATR_FAST_PERIOD).iloc[-1]
        atr_slow = calc_atr(self.ohlc_df, ATR_SLOW_PERIOD).iloc[-1]
        atr_fast_pips = atr_fast / self.symbol_info.pip_size
        atr_slow_pips = atr_slow / self.symbol_info.pip_size
        exit_params = make_exit_params(atr_fast_pips, atr_slow_pips, SPREAD_PIPS)
        if not exit_params.trade_allowed:
            return

        if self.open_position and entry_time >= self.open_position.timeout_time:
            self._close_position_market()
            return

        if self.open_position:
            return

        phase = 1.0
        sec_range = float(latest["high"] - latest["low"])
        state_vec = build_state_vec_fast(self.ohlc_df.tail(REQUIRED_CANDLES), phase, sec_range)
        state_vec = self.scaler.transform([state_vec])[0].astype(np.float32)

        with torch.no_grad():
            t = torch.from_numpy(state_vec).unsqueeze(0).float()
            q_long = self.models["long"](t).cpu().numpy().reshape(-1)
            q_short = self.models["short"](t).cpu().numpy().reshape(-1)
        p_long = float(softmax_probs(q_long)[1])
        p_short = float(softmax_probs(q_short)[1])

        decision = decide_entry_two_models(p_long, p_short, ENTRY_TH)
        if decision == "HOLD":
            return

        trend_dir = compute_trend_direction(self.ohlc_df["close"], window=REQUIRED_CANDLES)
        if decision == "LONG" and trend_dir < 0:
            return
        if decision == "SHORT" and trend_dir > 0:
            return

        position = build_exit_levels(
            decision,
            float(latest["close"]),
            self.symbol_info.pip_size,
            exit_params,
            entry_time,
        )
        self._open_position(position, exit_params)

    def _open_position(self, position, exit_params: ExitParams):
        if not self.symbol_info:
            return
        lots = calc_fx_lots_fixed_risk(
            balance=self.balance,
            risk_pct=RISK_PCT,
            sl_pips=exit_params.sl_pips,
            pip_value_per_lot=self.symbol_info.pip_value_per_lot,
            min_lot=max(0.01, self.symbol_info.min_volume / 100.0),
            lot_step=max(0.01, self.symbol_info.step_volume / 100.0),
        )
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
        self.pending_exit_params = exit_params
        position.volume_units = volume_units
        self.open_position = position
        print(
            f"[ENTRY] {position.side} price={position.entry_price:.5f} TP={position.tp_price:.5f} "
            f"SL={position.sl_price:.5f} lots={lots:.2f}"
        )

    def _close_position_market(self):
        if not self.open_position:
            return
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = int(self.account_id)
        if hasattr(self.open_position, "position_id"):
            req.positionId = int(self.open_position.position_id)
        req.volume = getattr(self.open_position, "volume_units", 0)
        self.client.send(req).addErrback(self._on_error)
        print("[EXIT] TIMEOUT - closing position by market")
        self.open_position = None

    def _handle_execution_event(self, message):
        if not self.open_position:
            return
        position = message.position
        if position and position.positionId:
            self.open_position.position_id = position.positionId
            if position.positionStatus == 2:  # POSITION_STATUS_CLOSED
                self.open_position = None


if __name__ == "__main__":
    if not CTRADER_CLIENT_ID or not CTRADER_CLIENT_SECRET:
        raise SystemExit("Set CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET.")
    if not CTRADER_ACCESS_TOKEN:
        raise SystemExit("Set CTRADER_ACCESS_TOKEN.")
    bot = AxioryCTraderBot()
    bot.start()
