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
ta = None
try:
    import talib as ta
    print("[INFO] TA-lib (talib) loaded successfully")
except ImportError:
    try:
        import ta
        print("[INFO] TA-lib alternative (ta) loaded successfully")
    except ImportError:
        print("[WARN] TA-lib not available. Using basic calculations...")
        # TA-lib関数のモック版を作成
        class MockTA:
            @staticmethod
            def EMA(close, period):
                return close.ewm(span=period).mean()
            
            @staticmethod
            def RSI(close, period):
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            @staticmethod
            def BBANDS(close, period, nbdevup=2, nbdevdn=2, matype=0):
                ma = close.rolling(period).mean()
                std = close.rolling(period).std()
                upper = ma + (std * nbdevup)
                lower = ma - (std * nbdevdn)
                return upper, ma, lower
                
            @staticmethod
            def ATR(high, low, close, period):
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                return tr.rolling(period).mean()
                
            @staticmethod
            def MOM(close, period):
                return close.diff(period)
                
            @staticmethod
            def STOCH(high, low, close, period):
                lowest_low = low.rolling(period).min()
                highest_high = high.rolling(period).max()
                k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
                d_percent = k_percent.rolling(3).mean()
                return k_percent, d_percent
                
            @staticmethod
            def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
                exp1 = close.ewm(span=fastperiod).mean()
                exp2 = close.ewm(span=slowperiod).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=signalperiod).mean()
                histogram = macd - signal
                return macd, signal, histogram
                
            @staticmethod
            def WILLR(high, low, close, period):
                highest_high = high.rolling(period).max()
                lowest_low = low.rolling(period).min()
                wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
                return wr
                
            @staticmethod
            def CCI(high, low, close, period):
                tp = (high + low + close) / 3
                sma = tp.rolling(period).mean()
                mad = tp.rolling(period).apply(lambda x: abs(x - x.mean()).mean())
                cci = (tp - sma) / (0.015 * mad)
                return cci
                
            @staticmethod
            def ROC(close, period=10):
                return ((close - close.shift(period)) / close.shift(period)) * 100
        
        ta = MockTA()

# Playwright
from playwright.sync_api import sync_playwright

# Torch (optional)
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# 設定
# -----------------------
pair = "USDJPY"  # BTCUSDからUSDJPYに変更
MODEL_PT = f"./Models/dqn_policy_{pair}.pt"  # train_dqn.pyの保存形式に合わせる
MODEL_PKL = f"./Models/dqn_scaler_{pair}.pkl"  # スケーラーファイルも追加
TICK_INTERVAL_SECONDS = 0.5
CANDLE_TIMEFRAME = '1min'
REQUIRED_CANDLES = 12  # 11から30に増加（より多くの履歴データ）
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
        ema = ta.EMA(df["close"], period)
        # ゼロ除算を防ぐ
        df["SMAR_"+str(period)] = np.where(df["close"] != 0, ema/df["close"], 1.0)
    return df

def _CalcRSIR(df,periods):
    for period in periods:
        df["RSIR_"+str(period)] = ta.RSI(df["close"], period)
        df["RSIR_diff_"+str(period)] = df["RSIR_"+str(period)].diff()
    return df
def _CalcOtherR(df, periods):
    # 全データで計算するように変更
    for period in periods:
        if len(df) < period:
            continue
            
        # ボリンジャーバンド幅
        try:
            upper, middle, lower = ta.BBANDS(df['close'], period)
            df['bb_width'+str(period)] = upper - lower
            df['bb_width_diff' + str(period)] = df['bb_width'+str(period)].diff()
            
            # ボリンジャーバンドの位置（%B）- ゼロ除算防止
            width = upper - lower
            df['bb_percent'+str(period)] = np.where(width != 0, (df['close'] - lower) / width, 0.5)
            
            # ATR（平均的な値幅）
            df['atr'+str(period)] = ta.ATR(df['high'], df['low'], df['close'], period)
            df['atr_diff' + str(period)] = df['atr'+str(period)].diff()
            
            # 価格変動率 - 異常値防止
            df['price_change'+str(period)] = df['close'].pct_change(period).clip(-1, 1)
            
            # ボラティリティ
            df['volatility'+str(period)] = df['close'].rolling(period).std()
            
            # 新しい特徴量：価格の勢い
            df['momentum'+str(period)] = ta.MOM(df['close'], period)
            df['momentum_norm'+str(period)] = df['momentum'+str(period)] / (df['close'] + 1e-8)
            
            # 価格の位置（高値・安値に対する相対位置）
            high_max = df['high'].rolling(period).max()
            low_min = df['low'].rolling(period).min()
            df['high_pos'+str(period)] = (df['close'] - low_min) / (high_max - low_min + 1e-8)
            
        except Exception as e:
            print(f"[WARNING] Error calculating period {period}: {e}")
            continue
    
    # より軽量な技術指標のみ計算
    try:
        # ストキャスティクス（期間を固定）
        period = 14
        slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'], period)
        df['slowk'] = slowk
        df['slowk_diff'] = df['slowk'].diff()
        df['slowd'] = slowd
        df['slowd_diff'] = df['slowd'].diff()
        
        # MACD
        macd, macdsignal, macdhist = ta.MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        df['macd_hist'] = macdhist
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)  # MACDクロス
        
        # Williams %R
        df['williams_r'] = ta.WILLR(df['high'], df['low'], df['close'])
        
        # CCI (Commodity Channel Index)
        df['cci'] = ta.CCI(df['high'], df['low'], df['close'])
        
        # ROC (Rate of Change)
        df['roc'] = ta.ROC(df['close'])
        
    except Exception as e:
        print(f"[WARNING] Error in additional indicators: {e}")
    
    # 全ての列でNaNと無限大を処理
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].ffill().fillna(0)  # 新しい記法に変更
    
    return df
    
    return df

# 特徴量計算のキャッシュを追加
_feature_cache = {}

def FeatureExtraction(df, use_cache=True):
    if use_cache:
        # DataFrameのハッシュをキーにしてキャッシュ
        df_hash = hash(tuple(df.iloc[-1].values))
        if df_hash in _feature_cache:
            return _feature_cache[df_hash]
    
    df = df.copy()
    periods_RSI = [14, 21]  # 期間を削減（計算量削減）
    periods_SMA = [10, 20]  # 期間を削減（計算量削減）

    df = _CalcSMAR(df, periods_SMA)
    df = _CalcRSIR(df, periods_RSI)
    df = _CalcOtherR(df, periods_RSI)

    # 基本的な価格比率 - ゼロ除算防止
    df["open_r"] = np.where(df["close"] != 0, df["open"]/df["close"], 1.0)
    df["high_r"] = np.where(df["close"] != 0, df["high"]/df["close"], 1.0)
    df["low_r"] = np.where(df["close"] != 0, df["low"]/df["close"], 1.0)
    
    # 追加の特徴量 - ゼロ除算防止
    df["hl_ratio"] = np.where(df["close"] != 0, (df["high"] - df["low"]) / df["close"], 0.0)
    df["oc_ratio"] = np.where(df["close"] != 0, (df["open"] - df["close"]) / df["close"], 0.0)
    
    # 移動平均との関係 - ゼロ除算防止
    for period in [5, 10, 20, 50]:
        if len(df) >= period:
            sma = df["close"].rolling(period).mean()
            ema = df["close"].ewm(span=period).mean()
            df[f"sma_distance_{period}"] = np.where(sma != 0, (df["close"] - sma) / sma, 0.0)
            df[f"ema_distance_{period}"] = np.where(ema != 0, (df["close"] - ema) / ema, 0.0)
            df[f"sma_ema_diff_{period}"] = np.where(ema != 0, (sma - ema) / ema, 0.0)
    
    # 価格と移動平均の交差シグナル
    sma5 = df["close"].rolling(5).mean()
    sma20 = df["close"].rolling(20).mean()
    df["golden_cross"] = np.where(sma5 > sma20, 1, 0)  # ゴールデンクロス
    df["dead_cross"] = np.where(sma5 < sma20, 1, 0)    # デッドクロス
    
    # 前の足との比較 - 異常値クリップ
    df["prev_close_ratio"] = df["close"].pct_change().clip(-1, 1)
    df["prev_volume_ratio"] = df["volume"].pct_change().clip(-10, 10) if "volume" in df.columns else 0
    
    # 複数期間の価格変化率
    for lookback in [2, 3, 5]:
        df[f"price_change_{lookback}"] = df["close"].pct_change(lookback).clip(-1, 1)
    
    # 高値・安値のブレイクアウトシグナル
    for period in [10, 20]:
        df[f"high_breakout_{period}"] = (df["high"] > df["high"].rolling(period).max().shift(1)).astype(int)
        df[f"low_breakout_{period}"] = (df["low"] < df["low"].rolling(period).min().shift(1)).astype(int)
    
    result = df.drop(columns = ["open", "close", "high", "low", "volume"], errors='ignore')
    
    # 異常値処理
    result = result.replace([np.inf, -np.inf], np.nan)  # 無限大をNaNに変換
    result = result.fillna(0)  # NaNを0で埋める
    
    # デバッグ: 特徴量の確認
    print(f"[DEBUG] drop前の列数: {len(df.columns)}")
    print(f"[DEBUG] drop後の列数: {len(result.columns)}")
    print(f"[DEBUG] 最初の10列: {list(result.columns[:10])}")
    print(f"[DEBUG] 最後の10列: {list(result.columns[-10:])}")
    
    # 異常値処理
    result = result.replace([np.inf, -np.inf], np.nan)  # 無限大をNaNに変換
    result = result.fillna(0)  # NaNを0で埋める
    
    # 異常に大きな値をクリップ
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # 99.9%分位点でクリップ
        upper_limit = result[col].quantile(0.999)
        lower_limit = result[col].quantile(0.001)
        result[col] = result[col].clip(lower=lower_limit, upper=upper_limit)
    
    # 最後の行のみを抽出して、正確に62次元に調整
    feature_row = result.iloc[-1:].copy()
    
    # 62次元を確保
    if len(feature_row.columns) > 62:
        # 最初の62列のみを選択
        feature_row = feature_row.iloc[:, :62]
        print(f"[INFO] 特徴量を62次元に削減 (元: {len(result.columns)})")
    elif len(feature_row.columns) < 62:
        # 不足分を0で補完
        missing = 62 - len(feature_row.columns)
        for i in range(missing):
            feature_row[f"pad_{i}"] = 0.0
        print(f"[INFO] 特徴量を62次元に補完 (元: {len(result.columns)})")
    
    print(f"[DEBUG] 最終特徴量数: {len(feature_row.columns)}")
    
    if use_cache:
        _feature_cache[df_hash] = feature_row
        # キャッシュサイズ制限
        if len(_feature_cache) > 10000:
            # 古いキャッシュを削除
            oldest_key = next(iter(_feature_cache))
            del _feature_cache[oldest_key]
    
    return feature_row

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
    """ポップアップ・広告・モーダルを確実に閉じる"""
    try:
        print("[INFO] ポップアップ・広告の閉じ処理を開始...")
        
        # 1. チャットウィジェットを無効化
        try:
            page.evaluate("""
                // Intercomチャットを非表示
                const chatIframes = document.querySelectorAll('iframe[title*="Intercom"], iframe.intercom-with-namespace-vo6dyv');
                chatIframes.forEach(iframe => {
                    iframe.style.display = 'none';
                    iframe.style.visibility = 'hidden';
                });
                
                // チャットコンテナも非表示
                const chatContainers = document.querySelectorAll('#intercom-container, .intercom-namespace');
                chatContainers.forEach(container => {
                    container.style.display = 'none';
                    container.style.visibility = 'hidden';
                });
            """)
            print("[INFO] チャットウィジェット無効化完了")
        except Exception as e:
            print(f"[WARN] チャット無効化失敗: {e}")
        
        # 2. 共通的な閉じるボタンを探して実行
        close_selectors = [
            # 標準的な閉じるボタン
            "button[aria-label='Close']",
            "button[aria-label='閉じる']", 
            "button[title='Close']",
            ".close",
            ".modal-close",
            ".popup-close",
            ".dialog-close",
            
            # 特定のライブラリ
            ".ant-modal-close",
            ".ant-modal-close-x",
            ".ant-drawer-close",
            ".el-dialog__close",
            ".el-message-box__close",
            
            # Toast/通知
            ".Toastify__close-button",
            ".toast-close",
            ".notification-close",
            
            # その他
            "[data-dismiss='modal']",
            "[data-bs-dismiss='modal']",
            ".btn-close"
        ]
        
        closed_count = 0
        for selector in close_selectors:
            elements = page.query_selector_all(selector)
            for element in elements:
                try:
                    if element.is_visible():
                        element.click(force=True)
                        closed_count += 1
                        time.sleep(0.1)
                except Exception:
                    pass
        
        if closed_count > 0:
            print(f"[INFO] {closed_count}個のポップアップを閉じました")
        
        # 3. モーダルオーバーレイを直接クリック
        overlay_selectors = [
            ".modal-backdrop",
            ".overlay", 
            ".ant-modal-wrap",
            ".ant-drawer-mask",
            ".el-overlay",
            ".v-overlay__scrim"
        ]
        
        for selector in overlay_selectors:
            elements = page.query_selector_all(selector)
            for element in elements:
                try:
                    if element.is_visible():
                        element.click(force=True)
                        time.sleep(0.1)
                except Exception:
                    pass
        
        # 4. Escapeキーを押す
        try:
            page.keyboard.press("Escape")
            time.sleep(0.2)
        except Exception:
            pass
        
        # 5. JavaScript実行で強制的にポップアップを削除
        try:
            page.evaluate("""
                // 固定位置の要素（ポップアップの可能性）を削除
                const fixedElements = document.querySelectorAll('*');
                fixedElements.forEach(el => {
                    const style = window.getComputedStyle(el);
                    if (style.position === 'fixed' && 
                        (style.zIndex > 1000 || el.classList.contains('modal') || 
                         el.classList.contains('popup') || el.classList.contains('dialog'))) {
                        el.style.display = 'none';
                    }
                });
                
                // 既知の広告・ポップアップクラスを削除
                const adSelectors = [
                    '.advertisement', '.ad-banner', '.popup', '.modal', 
                    '.overlay', '.lightbox', '.dialog', '.notification'
                ];
                adSelectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => {
                        if (el.style.zIndex > 100) el.style.display = 'none';
                    });
                });
            """)
            print("[INFO] JavaScript強制削除完了")
        except Exception as e:
            print(f"[WARN] JavaScript削除失敗: {e}")
        
        print("[INFO] ポップアップ閉じ処理完了")
        
    except Exception as e:
        print(f"[ERROR] ポップアップ処理でエラー: {e}")

def ensure_session(page, email, passward):
    try:
        login_btn = page.query_selector('#btnSubmit')
        if not login_btn:
            return False
        # login form present -> attempt re-login
        print("[INFO] ログインフォーム検出 -> 再ログイン実施")
        try:
            # メールアドレス入力
            email_input = page.query_selector('input[type="email"]') or page.query_selector('input[name="email"]') or page.query_selector('.form-control.lg-input')
            if email_input:
                email_input.clear()
                email_input.type(email, delay=50)
            
            # パスワード入力  
            password_input = page.query_selector('input[type="password"]') or page.query_selector('input[name="password"]')
            if not password_input:
                inputs = page.query_selector_all('.form-control.lg-input')
                if len(inputs) >= 2:
                    password_input = inputs[1]
            
            if password_input:
                password_input.clear()
                password_input.type(passward, delay=50)
            
            # ログインボタンクリック
            login_btn.click()
            
        except Exception as e:
            print(f"[WARN] Standard login failed, using fallback: {e}")
            # フォールバック: 従来の方法
            inputs = page.query_selector_all('.form-control.lg-input')
            if len(inputs) >= 2:
                inputs[0].fill(email)
                inputs[1].fill(passward)
            login_btn.click()
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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 保存されたモデルの正確な構造に合わせる
        # 入力: 64次元, 隠れ層: 512-512-256, 出力ストリーム: 128
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, 512),  # [512, 64] -> 入力64次元、出力512次元
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, 512),     # [512, 512]
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),     # [256, 512]
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # Value stream: 256 -> 128 -> 1
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),     # [128, 256]
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)        # [1, 128]
        )
        
        # Advantage stream: 256 -> 128 -> 3
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),     # [128, 256]
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)  # [3, 128]
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

dqn_model = None
dqn_is_torch = False
scaler = None  # スケーラーを追加

def infer_feature_dim_for_model():
    # build a dummy OHLC window so FeatureExtraction can compute features
    try:
        print("[DEBUG] Starting feature dimension inference...")
        N = max(50, REQUIRED_CANDLES + 20)  # より多くのサンプル
        idx = pd.date_range(end=datetime.now(), periods=N, freq='T')
        base = np.linspace(1.0, 1.0 + 0.001*N, N)
        dummy = pd.DataFrame({
            'open': base,
            'high': base + 0.0005,
            'low': base - 0.0005,
            'close': base
        }, index=idx)
        print(f"[DEBUG] Created dummy data with shape: {dummy.shape}")
        
        feat = FeatureExtraction(dummy)[-1:]
        print(f"[DEBUG] Feature extraction result shape: {feat.shape}")
        print(f"[INFO] 推定特徴量次元: {feat.shape[1]} + 2 (phase, range) = {feat.shape[1] + 2}")
        return feat.shape[1]
    except Exception as e:
        print(f"[ERROR] 特徴量次元推定失敗: {e}")
        import traceback
        print(f"[DEBUG] Feature inference traceback:\n{traceback.format_exc()}")
        return 50  # より大きなfallback値

# try torch .pt first
if os.path.exists(MODEL_PT):
    try:
        print(f"[DEBUG] Loading model from: {MODEL_PT}")
        ck = torch.load(MODEL_PT, map_location="cpu")
        print(f"[DEBUG] Loaded object type: {type(ck)}")
        
        # train_dqn.pyは直接state_dictを保存している
        # 保存されたモデルは64次元入力用
        in_dim = 64  # 保存されたモデルの入力次元に固定
        print(f"[DEBUG] Creating QNet with in_dim={in_dim}, out_dim=3")
        qnet = QNet(in_dim, 3)
        
        if isinstance(ck, dict) and ("model_state_dict" in ck or "state_dict" in ck):
            # 辞書形式の場合
            print("[DEBUG] Dict with model_state_dict/state_dict detected")
            st = ck.get("model_state_dict", ck.get("state_dict"))
            qnet.load_state_dict(st)
            print("[INFO] DQN (torch wrapped state_dict) ロード完了")
        elif isinstance(ck, dict):
            # 直接state_dictの場合（train_dqn.pyの保存形式）
            print("[DEBUG] Direct state_dict detected")
            print(f"[DEBUG] State dict keys: {list(ck.keys())[:5]}...")  # 最初の5つのキーを表示
            qnet.load_state_dict(ck)
            print("[INFO] DQN (torch direct state_dict) ロード完了")
        elif isinstance(ck, nn.Module):
            # モジュール全体が保存されている場合
            print("[DEBUG] PyTorch module detected")
            qnet = ck
            print("[INFO] DQN (torch module) ロード完了")
        else:
            print(f"[ERROR] MODEL_PT 読込はしたが形式不明: {type(ck)}")
            if hasattr(ck, '__dict__'):
                print(f"[DEBUG] Object attributes: {list(vars(ck).keys())}")
            qnet = None
        
        if qnet is not None:
            qnet.eval()
            dqn_model = qnet
            dqn_is_torch = True
            print(f"[INFO] Model successfully loaded and set to eval mode")
        
    except Exception as e:
        print(f"[ERROR] torch load 失敗: {e}")
        import traceback
        print(f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
        dqn_model = None
else:
    print(f"[ERROR] Model file not found: {MODEL_PT}")

# Load scaler
try:
    with open(MODEL_PKL, "rb") as f:
        scaler = pickle.load(f)
    print("[INFO] Scaler ロード完了")
except Exception as e:
    print(f"[WARN] Scaler load 失敗: {e}")
    scaler = None

if dqn_model is None:
    print("[ERROR] DQNモデルが見つかりません。予測はスキップされます。")
    print(f"[DEBUG] MODEL_PT: {MODEL_PT}")
    print(f"[DEBUG] ファイル存在: {os.path.exists(MODEL_PT)}")
else:
    print(f"[INFO] DQNモデル読み込み成功 - PyTorch: {dqn_is_torch}")

if scaler is None:
    print("[ERROR] スケーラーが見つかりません。特徴量の正規化ができません。")
    print(f"[DEBUG] MODEL_PKL: {MODEL_PKL}")
    print(f"[DEBUG] ファイル存在: {os.path.exists(MODEL_PKL)}")
else:
    print("[INFO] スケーラー読み込み成功")

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

    # ログイン (simple and stable)
    time.sleep(2)
    try:
        # メールアドレス入力
        email_input = page.query_selector('input[type="email"]') or page.query_selector('input[name="email"]') or page.query_selector('.form-control.lg-input')
        if email_input:
            email_input.clear()
            email_input.type(email, delay=100)
            print(f"[INFO] Email entered: {email}")
        
        # パスワード入力  
        password_input = page.query_selector('input[type="password"]') or page.query_selector('input[name="password"]')
        if not password_input:
            inputs = page.query_selector_all('.form-control.lg-input')
            if len(inputs) >= 2:
                password_input = inputs[1]
        
        if password_input:
            password_input.clear()
            password_input.type(passward, delay=100)
            print(f"[INFO] Password entered")
        
        # ログインボタンクリック
        login_btn = page.query_selector('#btnSubmit') or page.query_selector('button[type="submit"]') or page.query_selector('.btn-primary')
        if login_btn:
            login_btn.click()
            print(f"[INFO] Login button clicked")
        
    except Exception as e:
        print(f"[ERROR] Login process failed: {e}")
        # フォールバック: 従来の方法
        inputs = page.query_selector_all('.form-control.lg-input')
        if len(inputs) >= 2:
            inputs[0].fill(email)
            inputs[1].fill(passward)
            login_btn = page.query_selector('#btnSubmit')
            if login_btn:
                login_btn.click()
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
            try:
                ohlc_data = ticks_to_ohlc(all_ticks, timeframe_sec=60, max_bars=REQUIRED_CANDLES+20)
            except Exception as e:
                print(f"[WARN] OHLC生成エラー: {e}")
                time.sleep(TICK_INTERVAL_SECONDS)
                continue

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
            try:
                fea_ohlc = ohlc_data[['open','high','low','close']].copy()
                feats_df = FeatureExtraction(fea_ohlc)
                # take last row - should be 62 dimensions
                feat_row = feats_df.iloc[-1].values.astype(np.float32)
                print(f"[DEBUG] FeatureExtraction output shape: {feat_row.shape}")
                
                # Add phase and sec_range to make 64 dimensions total
                sec_range = float(fea_ohlc['high'].iloc[-1] - fea_ohlc['low'].iloc[-1])
                feat_vec = np.concatenate([feat_row, np.asarray([phase, sec_range], dtype=np.float32)])
                print(f"[DEBUG] Final feature vector shape: {feat_vec.shape}")
                
                # 特徴量の正規化（スケーラーがある場合）
                if scaler is not None:
                    feat_vec = scaler.transform([feat_vec])[0].astype(np.float32)
                    print(f"[DEBUG] Scaled feature vector shape: {feat_vec.shape}")
            except Exception as e:
                print(f"[WARN] 特徴量抽出エラー: {e}")
                time.sleep(TICK_INTERVAL_SECONDS)
                continue

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
                
            if scaler is None:
                reason = "no_scaler"
                print(f"[{current_time.strftime('%H:%M:%S')}] スケーラー無し - スキップ")
                _log_signal(current_time, current_price, phase, None, None, "Hold", False, reason)
                time.sleep(TICK_INTERVAL_SECONDS)
                continue

            # Torch model prediction
            if dqn_is_torch and isinstance(dqn_model, nn.Module):
                try:
                    with torch.no_grad():
                        t = torch.from_numpy(feat_vec).unsqueeze(0).float()
                        out = dqn_model(t)
                        qv = out.cpu().numpy().reshape(-1)
                    
                    # Ensure qv length 3
                    if qv.shape[0] >= 3:
                        q_values = qv[:3].astype(float)
                    else:
                        q_values = np.pad(qv.astype(float), (0,3-qv.shape[0]), 'constant')
                    
                    action_idx = int(np.argmax(q_values))
                    # map idx -> action: 0=Hold,1=High,2=Low
                    action_map = {0:"Hold", 1:"High", 2:"Low"}
                    action_str = action_map.get(action_idx, "Hold")
                    
                except Exception as e:
                    print(f"[WARN] モデル推論失敗: {e}")
                    reason = "predict_error"
                    _log_signal(current_time, current_price, phase, None, None, "Hold", False, reason)
                    time.sleep(TICK_INTERVAL_SECONDS)
                    continue
            else:
                reason = "unsupported_model"
                print(f"[WARN] 非Torchモデルはサポートされていません")
                _log_signal(current_time, current_price, phase, None, None, "Hold", False, reason)
                time.sleep(TICK_INTERVAL_SECONDS)
                continue

            # Decide entry: skip Hold
            if action_str == "Hold":
                reason = "hold"
                entry = False
                print(f"[{current_time.strftime('%H:%M:%S')}] Hold - Q値: Hold={q_values[0]:.3f}, High={q_values[1]:.3f}, Low={q_values[2]:.3f}")
            else:
                # optionally require q advantage over hold
                q_advantage = q_values[action_idx] - q_values[0]
                if q_advantage >= DQN_Q_MARGIN:
                    # cooldown check
                    if next_entry_allowed_time and current_time < next_entry_allowed_time:
                        reason = "cooldown"
                        entry = False
                        print(f"[{current_time.strftime('%H:%M:%S')}] {action_str} - クールダウン中 (残り{(next_entry_allowed_time-current_time).total_seconds():.1f}秒)")
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
                            print(f"[ENTRY] {action_str} at {current_time.strftime('%H:%M:%S')} price={current_price} Q値: {q_values[action_idx]:.3f} (優位性: {q_advantage:.3f})")
                        else:
                            reason = "button_not_found"
                            entry = False
                            print(f"[WARN] {action_str}ボタンが見つかりません")
                else:
                    reason = "insufficient_q_advantage"
                    entry = False
                    print(f"[{current_time.strftime('%H:%M:%S')}] {action_str} - Q値優位性不足 ({q_advantage:.3f} < {DQN_Q_MARGIN})")

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

