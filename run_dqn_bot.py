#!/usr/bin/env python
# coding: utf-8

"""
BeddingSystem_dqn.py
- Playwright + DQN (Torch or pickled) の実行版
- human_click / human_type / try_close_popups / ensure_session を全面適用
- all_ticks -> ohlc_data を明確に生成し current_time はループ開始で設定
- ログに q値 / action を出力
"""

import os
import csv
import time
import random
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

# TA-lib (必要)
import talib as ta

# Playwright
from playwright.sync_api import sync_playwright

# Torch (optional)
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# 設定
# -----------------------
pair = "BTCUSD"
MODEL_PT = f"./Models/{pair}_dqn.pt"
MODEL_PKL = f"./Models/{pair}_dqn.pickle"
TICK_INTERVAL_SECONDS = 0.5
CANDLE_TIMEFRAME = '1min'
REQUIRED_CANDLES = 11
ENTRY_COOLDOWN_SECONDS = 25
LOG_DIR = "./logs"
LOG_PATH = os.path.join(LOG_DIR, f"live_signals_{pair}.csv")
os.makedirs(LOG_DIR, exist_ok=True)

# ログヘッダ
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ts","price","phase",
            "q_hold","q_high","q_low","action_idx","action","entry","reason"
        ])

# DQNの閾値（Holdをスキップする／しきい値）
DQN_Q_MARGIN = 0.0  # Holdとの差でエントリーを抑制したければ正にする

# -----------------------
# FeatureExtraction（既存ロジック準拠）
# -----------------------
def _CalcSMAR(df,periods):
    for period in periods:
        df["SMAR_"+str(period)] = ta.EMA(df["close"], period)/df["close"]
    return df

def _CalcRSIR(df,periods):
    for period in periods:
        df["RSIR_"+str(period)] = ta.RSI(df["close"], period)
        df["RSIR_diff_"+str(period)] = df["RSIR_"+str(period)].diff()
    return df

def _CalcOtherR(df,periods):
    for period in periods:
        df['bb_width'+str(period)] = ta.BBANDS(df['close'], period)[1] - ta.BBANDS(df['close'], period)[0]
        df['bb_width_diff' + str(period)] = df['bb_width'+str(period)].diff()
        df['atr'+str(period)] = ta.ATR(df['high'], df['low'], df['close'], period)
        df['atr_diff' + str(period)] = df['atr'+str(period)].diff()
    # ストキャスティクス (ta.STOCH returns arrays; ensure length match)
    try:
        slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'])
        df['slowk'+str(period)] = slowk
        df['slowk_diff'+str(period)] = df['slowk'+str(period)].diff()
        df['slowd'+str(period)] = slowd
        df['slowd_diff'+str(period)] = df['slowd'+str(period)].diff()
    except Exception:
        pass
    return df

def FeatureExtraction(df):
    # expects df with columns open,high,low,close and DatetimeIndex
    df = df.copy().reset_index(drop=True)
    periods_RSI = [3, 5, 10]
    periods_SMA = [3, 5, 10]
    # Ensure float dtype
    for c in ['open','high','low','close']:
        df[c] = df[c].astype(float)
    # apply indicators
    df = _CalcSMAR(df, periods_SMA)
    df = _CalcRSIR(df, periods_RSI)
    df = _CalcOtherR(df, periods_RSI)
    df["open_r"] = df["open"]/df["close"]
    df["high_r"] = df["high"]/df["close"]
    df["low_r"] = df["low"]/df["close"]
    # drop originals if exist
    cols_to_drop = [c for c in ["open","close","high","low"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    # forward-fill / fillna to avoid NaNs in short windows
    df = df.fillna(method="ffill").fillna(0.0)
    return df

# -----------------------
# human-like 操作関数 (Playwright用)
# -----------------------
def human_click(element, page):
    try:
        box = element.bounding_box()
        if not box:
            element.click(force=True)
            return
        x = box['x'] + box['width']/2 + random.randint(-5,5)
        y = box['y'] + box['height']/2 + random.randint(-5,5)
        steps = random.randint(12, 30)
        page.mouse.move(x, y, steps=steps)
        time.sleep(random.uniform(0.15, 0.45))
        page.mouse.click(x, y, delay=random.randint(40,160))
    except Exception as e:
        try:
            element.click(force=True)
        except Exception:
            print(f"[WARN] human_click失敗: {e}")

def human_type(element, text):
    try:
        element.click()
        for ch in text:
            element.type(ch, delay=random.randint(60, 180))
            if random.random() < 0.06:
                time.sleep(random.uniform(0.2,0.6))
        time.sleep(random.uniform(0.1,0.3))
    except Exception as e:
        print(f"[WARN] human_type失敗: {e} -> fallback fill")
        try:
            element.fill(text)
        except Exception as e2:
            print(f"[ERROR] fallback fill 失敗: {e2}")

def try_close_popups(page):
    try:
        # hide chat iframe
        chat_iframe = page.query_selector("iframe.intercom-with-namespace-vo6dyv")
        if chat_iframe:
            try:
                page.evaluate("iframe => iframe.style.display = 'none'", chat_iframe)
            except Exception:
                pass
        selectors = [
            ".modal-close", ".close", ".modal-header .close", ".modal .btn-close",
            "button[aria-label='Close']", "button[aria-label='閉じる']",
            ".ant-modal-close", ".Toastify__toast button[aria-label='close']"
        ]
        for sel in selectors:
            for btn in page.query_selector_all(sel):
                try:
                    human_click(btn, page)
                except Exception:
                    pass
        overlay_selectors = ".modal-backdrop, .overlay, .ant-modal-wrap, .ant-drawer-mask"
        for ov in page.query_selector_all(overlay_selectors):
            try:
                human_click(ov, page)
            except Exception:
                pass
        # ESC
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] ポップアップ消去処理でエラー: {e}")

def ensure_session(page, email, passward):
    try:
        login_btn = page.query_selector('#btnSubmit')
        if not login_btn:
            return False
        # login form present -> attempt re-login
        print("[INFO] ログインフォーム検出 -> 再ログイン実施")
        inputs = page.query_selector_all('.form-control.lg-input')
        if len(inputs) >= 2:
            human_type(inputs[0], email)
            human_type(inputs[1], passward)
        human_click(login_btn, page)
        try:
            page.wait_for_selector('.strikeWrapper div', timeout=3000)
        except Exception:
            print("[WARN] strikeWrapper待機タイムアウト (セッション復帰遅延)")
        try_close_popups(page)
        return True
    except Exception as e:
        print(f"[WARN] 再ログイン試行でエラー: {e}")
    return False

# -----------------------
# ticks -> OHLC helper
# -----------------------
def ticks_to_ohlc(ticks, timeframe_sec=60, max_bars=200):
    """ticks: list of (datetime, price)"""
    if not ticks:
        return pd.DataFrame(columns=['ts','open','high','low','close'])
    df = pd.DataFrame(ticks, columns=['ts','price'])
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts')
    ohlc = df['price'].resample(f'{timeframe_sec}s').ohlc()
    ohlc = ohlc.dropna().tail(max_bars).reset_index()
    # rename to open/high/low/close
    ohlc = ohlc.rename(columns={'index':'ts'})
    return ohlc[['ts','open','high','low','close']]

# -----------------------
# DQN loader (flexible)
# -----------------------
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

dqn_model = None
dqn_is_torch = False

def infer_feature_dim_for_model():
    # build a dummy OHLC window so FeatureExtraction can compute features
    try:
        N = max(30, REQUIRED_CANDLES + 10)
        idx = pd.date_range(end=datetime.now(), periods=N, freq='T')
        base = np.linspace(1.0, 1.0 + 0.001*N, N)
        dummy = pd.DataFrame({
            'open': base,
            'high': base + 0.0005,
            'low': base - 0.0005,
            'close': base
        }, index=idx)
        feat = FeatureExtraction(dummy)[-1:]
        return feat.shape[1]
    except Exception:
        return 20  # fallback

# try torch .pt first
if os.path.exists(MODEL_PT):
    try:
        ck = torch.load(MODEL_PT, map_location="cpu")
        # if ck is dict with state dict
        if isinstance(ck, dict) and ("model_state_dict" in ck or "state_dict" in ck):
            feat_len = infer_feature_dim_for_model()
            in_dim = feat_len + 2
            qnet = QNet(in_dim, 3)
            st = ck.get("model_state_dict", ck.get("state_dict"))
            qnet.load_state_dict(st)
            qnet.eval()
            dqn_model = qnet
            dqn_is_torch = True
            print("[INFO] DQN (torch state_dict) ロード完了")
        elif isinstance(ck, nn.Module):
            dqn_model = ck
            dqn_is_torch = True
            print("[INFO] DQN (torch module) ロード完了")
        else:
            # maybe saved plain tensor or other - fallback to try pickle path below
            print("[WARN] MODEL_PT 読込はしたが形式不明、pickleを試行します")
    except Exception as e:
        print(f"[WARN] torch load 失敗: {e} - pickleを試行します")

# if not loaded, try pickle
if dqn_model is None and os.path.exists(MODEL_PKL):
    try:
        with open(MODEL_PKL, "rb") as f:
            data = pickle.load(f)
        # if dict with 'model' key (user earlier used this pattern)
        if isinstance(data, dict) and "model" in data:
            dqn_model = data["model"]
            dqn_is_torch = isinstance(dqn_model, nn.Module)
            print("[INFO] DQN (pickled dict) ロード完了")
        else:
            # if whole object is a sklearn-like model (has predict / predict_proba)
            dqn_model = data
            dqn_is_torch = isinstance(dqn_model, nn.Module)
            print("[INFO] DQN (pickled object) ロード完了")
    except Exception as e:
        print(f"[WARN] pickle load 失敗: {e}")

if dqn_model is None:
    print("[WARN] DQNモデルが見つかりません。予測はスキップされます。")

# -----------------------
# ログ関数 (q値とactionを記録)
# -----------------------
def _log_signal(ts, price, phase, q_values, action_idx, action_str, entry, reason):
    try:
        q_hold = q_values[0] if q_values is not None else ""
        q_high = q_values[1] if q_values is not None else ""
        q_low  = q_values[2] if q_values is not None else ""
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ts.isoformat(), price, round(phase,4),
                q_hold, q_high, q_low,
                action_idx if action_idx is not None else "",
                action_str or "",
                int(bool(entry)),
                reason or ""
            ])
    except Exception as e:
        print(f"[WARN] ログ書き込み失敗: {e}")

# -----------------------
# 実行ループ (Playwright)
# -----------------------
url = "https://jp-demo.theoption.com/trading"
email = "miya4444nyan@gmail.com"
passward = "Miya564219"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(
        viewport={"width":1280,"height":800},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    )
    page = context.new_page()
    page.goto(url)

    # 初期待機 & ポップアップ閉じ
    time.sleep(10)
    try_close_popups(page)

    # ログイン (human-like)
    inputs = page.query_selector_all('.form-control.lg-input')
    if len(inputs) >= 2:
        human_type(inputs[0], email)
        human_type(inputs[1], passward)
    login_btn = page.query_selector('#btnSubmit')
    if login_btn:
        human_click(login_btn, page)
    try:
        page.wait_for_selector(".strikeWrapper div", timeout=20000)
    except Exception:
        pass
    try_close_popups(page)

    # ループ準備
    all_ticks = []
    last_entry_time = None
    next_entry_allowed_time = None
    recent_prices = deque(maxlen= int(10 / max(TICK_INTERVAL_SECONDS, 0.001)) + 2)

    while True:
        try:
            # session & popups
            try_close_popups(page)
            ensure_session(page, email, passward)

            current_time = datetime.now()

            # 価格取得
            price_elem = page.query_selector('.strikeWrapper div')
            if not price_elem:
                time.sleep(TICK_INTERVAL_SECONDS)
                continue
            price_str = (price_elem.inner_text() or '').strip()
            if not price_str or price_str in ('-', '—'):
                time.sleep(TICK_INTERVAL_SECONDS)
                continue
            try:
                current_price = float(price_str)
            except Exception:
                # couldn't parse
                time.sleep(TICK_INTERVAL_SECONDS)
                continue

            # ティック蓄積
            all_ticks.append((current_time, current_price))
            recent_prices.append(current_price)

            # OHLC生成
            ohlc_data = ticks_to_ohlc(all_ticks, timeframe_sec=60, max_bars=REQUIRED_CANDLES+20)

            if len(ohlc_data) < REQUIRED_CANDLES:
                # 足りない
                print(f"\r{current_time.strftime('%H:%M:%S')} - OHLC収集中 ({len(ohlc_data)}/{REQUIRED_CANDLES})", end="")
                time.sleep(TICK_INTERVAL_SECONDS)
                continue

            # phase, pseudo-last bar
            phase = 0.0
            try:
                # compute phase using current_time relative to minute
                sec = current_time.second + current_time.microsecond/1e6
                phase = min(1.0, sec/60.0)
            except Exception:
                phase = 0.0

            # FeatureExtraction expects DataFrame with open/high/low/close columns
            fea_ohlc = ohlc_data[['open','high','low','close']].copy()
            feats_df = FeatureExtraction(fea_ohlc)
            # take last row
            feat_row = feats_df.iloc[-1].values.astype(np.float32)
            # second extra: range of last candle
            sec_range = float(fea_ohlc['high'].iloc[-1] - fea_ohlc['low'].iloc[-1])
            feat_vec = np.concatenate([feat_row, np.asarray([phase, sec_range], dtype=np.float32)])

            # Predict via model (if present)
            q_values = None
            action_idx = None
            action_str = None
            entry = False
            reason = ""

            if dqn_model is None:
                reason = "no_model"
                print(f"[{current_time.strftime('%H:%M:%S')}] モデル無し - スキップ")
                _log_signal(current_time, current_price, phase, None, None, "Hold", False, reason)
                time.sleep(TICK_INTERVAL_SECONDS)
                continue

            # Torch model path
            if dqn_is_torch and isinstance(dqn_model, nn.Module):
                with torch.no_grad():
                    t = torch.from_numpy(feat_vec).unsqueeze(0).float()
                    out = dqn_model(t)
                    qv = out.cpu().numpy().reshape(-1)
                # Ensure qv length 3; if model outputs 2 (High/Low), adapt:
                if qv.shape[0] == 2:
                    # map to [hold, high, low] by: hold=0, high=qv[0], low=qv[1]
                    q_values = np.array([0.0, float(qv[0]), float(qv[1])], dtype=float)
                elif qv.shape[0] >= 3:
                    q_values = qv[:3].astype(float)
                else:
                    q_values = np.pad(qv.astype(float), (0,3-qv.shape[0]), 'constant')
                action_idx = int(np.argmax(q_values))
                # map idx -> action: 0=Hold,1=High,2=Low
                action_map = {0:"Hold", 1:"High", 2:"Low"}
                action_str = action_map.get(action_idx, "Hold")
            else:
                # non-torch (sklearn/pickle) model: try predict_proba or predict
                try:
                    # if model has predict_proba and outputs 3 probs:
                    if hasattr(dqn_model, "predict_proba"):
                        proba = dqn_model.predict_proba(pd.DataFrame([feat_vec]))
                        # pick last row
                        if proba.shape[1] >= 3:
                            q_values = proba[0][:3].astype(float)
                        elif proba.shape[1] == 2:
                            # binary classifier -> map to high/low and hold=0
                            q_values = np.array([0.0, float(proba[0,1]), float(1.0-proba[0,1])])
                        else:
                            q_values = np.pad(proba[0].astype(float), (0,3-proba.shape[1]), 'constant')
                    else:
                        # fallback: use predict -> binary label; convert to q-values
                        pred = dqn_model.predict(pd.DataFrame([feat_vec]))
                        label = int(pred[0])
                        # interpret label 1 -> High, 0 -> Low ; create q vector
                        if label == 1:
                            q_values = np.array([0.0, 1.0, 0.0])
                            action_idx = 1
                            action_str = "High"
                        else:
                            q_values = np.array([0.0, 0.0, 1.0])
                            action_idx = 2
                            action_str = "Low"
                except Exception as e:
                    print(f"[WARN] 非Torchモデル推論失敗: {e}")
                    reason = "predict_error"
                    _log_signal(current_time, current_price, phase, None, None, "Hold", False, reason)
                    time.sleep(TICK_INTERVAL_SECONDS)
                    continue
                if action_idx is None:
                    action_idx = int(np.argmax(q_values))
                    action_map = {0:"Hold", 1:"High", 2:"Low"}
                    action_str = action_map.get(action_idx, "Hold")

            # Decide entry: skip Hold
            if action_str == "Hold":
                reason = "hold"
                entry = False
            else:
                # optionally require q advantage over hold
                if (q_values[action_idx] - q_values[0]) >= DQN_Q_MARGIN:
                    # cooldown check
                    if next_entry_allowed_time and current_time < next_entry_allowed_time:
                        reason = "cooldown"
                        entry = False
                    else:
                        # execute entry
                        sel = '.invest-btn-up.button' if action_str == "High" else '.invest-btn-down.button'
                        btn = page.query_selector(sel)
                        if btn:
                            human_click(btn, page)
                            last_entry_time = current_time
                            next_entry_allowed_time = current_time + timedelta(seconds=ENTRY_COOLDOWN_SECONDS)
                            entry = True
                            reason = "entry_executed"
                            print(f"[ENTRY] {action_str} at {current_time.strftime('%H:%M:%S')} price={current_price}")
                        else:
                            reason = "button_not_found"
                            entry = False

            # log
            _log_signal(current_time, current_price, phase, q_values, action_idx, action_str, entry, reason)

            # prune ticks older than e.g. 2 hours to keep memory bounded
            two_hours_ago = current_time - timedelta(hours=2)
            all_ticks = [t for t in all_ticks if t[0] > two_hours_ago]

            time.sleep(TICK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("ユーザー割込みで終了します")
            break
        except Exception as e:
            import traceback
            print(f"[ERROR] 例外: {e}")
            traceback.print_exc()
            # 一時的にticksをクリアしてリカバリ
            all_ticks = []
            time.sleep(TICK_INTERVAL_SECONDS)

# end
