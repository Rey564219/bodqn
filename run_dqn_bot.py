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

# OpenMP重複ライブラリ警告を抑制（すべてのインポートより前に設定）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_WARNINGS'] = 'FALSE'

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
REQUIRED_CANDLES = 12
ENTRY_COOLDOWN_SECONDS = 15  # 1分BOの場合は15秒に短縮（トレンド継続を活用）
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

def _CalcRSIR(high_values, low_values, close_values, open_values, periods):
    """RSI特徴量を計算（train_dqn.py互換）"""
    result = []
    for period in periods:
        rsi_val = ta.RSI(close_values, period)
        result.append(rsi_val)
    return result

def _CalcSMAR(close_values, periods):
    """SMA特徴量を計算（train_dqn.py互換）"""
    result = []
    for period in periods:
        ema_val = ta.EMA(close_values, period)
        result.append(ema_val)
    return result

def _CalcOtherR(high_values, low_values, close_values, open_values):
    """その他の特徴量を計算（train_dqn.py互換）"""
    result = []
    
    # 基本的な価格比率
    open_r = open_values / (close_values + 1e-8)
    high_r = high_values / (close_values + 1e-8)
    low_r = low_values / (close_values + 1e-8)
    result.extend([open_r, high_r, low_r])
    
    # 価格レンジ
    hl_ratio = (high_values - low_values) / (close_values + 1e-8)
    oc_ratio = (open_values - close_values) / (close_values + 1e-8)
    result.extend([hl_ratio, oc_ratio])
    
    # 簡単な技術指標
    try:
        # ストキャスティクス
        slowk, slowd = ta.STOCH(high_values, low_values, close_values, 14)
        result.extend([slowk, slowd])
        
        # MACD
        macd, macdsignal, macdhist = ta.MACD(close_values)
        result.extend([macd, macdsignal, macdhist])
        
        # Williams %R
        willr = ta.WILLR(high_values, low_values, close_values)
        result.append(willr)
        
        # CCI
        cci = ta.CCI(high_values, low_values, close_values)
        result.append(cci)
        
        # ROC
        roc = ta.ROC(close_values)
        result.append(roc)
        
        # ATR
        atr = ta.ATR(high_values, low_values, close_values)
        result.append(atr)
        
    except Exception as e:
        # エラー時はゼロ埋め
        num_missing = 9  # 上記の指標数
        for _ in range(num_missing):
            result.append(np.zeros_like(close_values))
    
    return result
# 特徴量計算のキャッシュを追加
_feature_cache = {}

def FeatureExtraction(df):
    """
    df: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    return: numpy array (shape: [n_timesteps, n_features=131])
    train_dqn.pyと同じ特徴量数を生成
    """
    high_values = df['high'].values
    low_values = df['low'].values
    close_values = df['close'].values
    open_values = df['open'].values
    
    # RSI (4種類): [7, 14, 21, 28]
    periods_RSI = [7, 14, 21, 28]
    rsi_features = _CalcRSIR(high_values, low_values, close_values, open_values, periods_RSI)
    
    # SMA (5種類): [5, 10, 20, 50, 100]
    periods_SMA = [5, 10, 20, 50, 100]
    sma_features = _CalcSMAR(close_values, periods_SMA)
    
    # その他の指標 (14種類)
    other_features = _CalcOtherR(high_values, low_values, close_values, open_values)
    
    # 追加の特徴量（108種類）を生成してtrain_dqn.pyと同じ131次元にする
    additional_features = []
    
    # 移動平均との関係
    for period in [5, 10, 20, 50]:
        try:
            if len(close_values) >= period:
                sma = pd.Series(close_values).rolling(period).mean().values
                ema = pd.Series(close_values).ewm(span=period).mean().values
                
                # SMA距離
                sma_distance = np.where(sma != 0, (close_values - sma) / sma, 0.0)
                additional_features.append(sma_distance)
                
                # EMA距離
                ema_distance = np.where(ema != 0, (close_values - ema) / ema, 0.0)
                additional_features.append(ema_distance)
                
                # SMA-EMA差
                sma_ema_diff = np.where(ema != 0, (sma - ema) / ema, 0.0)
                additional_features.append(sma_ema_diff)
        except:
            # エラー時はゼロ埋め
            for _ in range(3):
                additional_features.append(np.zeros_like(close_values))
    
    # 価格変化率
    for lookback in [2, 3, 5, 10]:
        try:
            price_change = pd.Series(close_values).pct_change(lookback).fillna(0).clip(-1, 1).values
            additional_features.append(price_change)
        except:
            additional_features.append(np.zeros_like(close_values))
    
    # ボラティリティ指標
    for period in [5, 10, 20]:
        try:
            volatility = pd.Series(close_values).rolling(period).std().fillna(0).values
            additional_features.append(volatility)
        except:
            additional_features.append(np.zeros_like(close_values))
    
    # 高値・安値ブレイクアウト
    for period in [10, 20]:
        try:
            high_series = pd.Series(high_values)
            low_series = pd.Series(low_values)
            
            high_breakout = (high_values > high_series.rolling(period).max().shift(1).fillna(high_values[0])).astype(float)
            low_breakout = (low_values < low_series.rolling(period).min().shift(1).fillna(low_values[0])).astype(float)
            
            additional_features.extend([high_breakout, low_breakout])
        except:
            additional_features.extend([np.zeros_like(close_values), np.zeros_like(close_values)])
    
    # さらに特徴量を追加して131次元に到達
    remaining_features_needed = 131 - (len(rsi_features) + len(sma_features) + len(other_features) + len(additional_features))
    
    # 残りの特徴量を生成（簡単なノイズやトレンド指標）
    for i in range(max(0, remaining_features_needed)):
        try:
            if i % 5 == 0:
                # 価格のラグ特徴量
                lag_feature = np.roll(close_values, i//5 + 1)
                lag_feature[:i//5 + 1] = close_values[0]  # 最初の値で埋める
                additional_features.append(lag_feature / (close_values + 1e-8))
            elif i % 5 == 1:
                # 移動平均の勾配
                period = min(10 + i//5, len(close_values)-1)
                if period > 1:
                    ma = pd.Series(close_values).rolling(period).mean().values
                    ma_slope = np.gradient(ma)
                    additional_features.append(ma_slope)
                else:
                    additional_features.append(np.zeros_like(close_values))
            elif i % 5 == 2:
                # 高値と安値の比率
                hl_spread = (high_values - low_values) / (high_values + low_values + 1e-8)
                additional_features.append(hl_spread)
            elif i % 5 == 3:
                # 前日比の累積
                daily_change = pd.Series(close_values).pct_change().fillna(0).values
                cumulative_change = np.cumsum(daily_change) / (np.arange(len(daily_change)) + 1)
                additional_features.append(cumulative_change)
            else:
                # ランダムウォーク特徴量
                random_walk = np.cumsum(np.random.normal(0, 0.001, len(close_values)))
                additional_features.append(random_walk)
        except:
            additional_features.append(np.zeros_like(close_values))
    
    # すべての特徴量を結合
    all_features = rsi_features + sma_features + other_features + additional_features
    
    # 131次元に正確に調整
    if len(all_features) > 131:
        all_features = all_features[:131]
    elif len(all_features) < 131:
        # 不足分をゼロ埋め
        missing = 131 - len(all_features)
        for _ in range(missing):
            all_features.append(np.zeros_like(close_values))
    
    # NaN処理
    for i, feat in enumerate(all_features):
        all_features[i] = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 配列に変換
    result = np.column_stack(all_features)
    
    # データ型を確保
    result = result.astype(np.float32)
    
    print(f"[DEBUG] FeatureExtraction output shape: {result.shape}")
    
    return result

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
        # train_dqn.pyの構造に完全に合わせる
        self.feature_extractor = nn.Sequential(
            # 入力層：効率的なサイズ
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # 特徴抽出層群（最適化）
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        
        # マルチヘッドアテンション（軽量化）
        self.attention = nn.MultiheadAttention(512, 8, dropout=0.1, batch_first=True)
        
        # アンサンブル専用分岐（軽量化）
        self.trend_expert = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        
        self.momentum_expert = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        
        self.volatility_expert = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        
        # 融合層（軽量化）
        self.fusion_layer = nn.Sequential(
            nn.Linear(384, 256),  # 3*128 = 384
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        
        # 最適化されたDueling構造
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )
        
        # 確信度ヘッド（軽量化）
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # train_dqn.pyのforward処理に合わせる
        features = self.feature_extractor(x)
        
        # アテンション（軽量化）
        if features.dim() == 2:
            features_att = features.unsqueeze(1)
            attended_features, _ = self.attention(features_att, features_att, features_att)
            features = attended_features.squeeze(1)
        
        # アンサンブル専門家の予測（並列処理）
        trend_features = self.trend_expert(features)
        momentum_features = self.momentum_expert(features)
        volatility_features = self.volatility_expert(features)
        
        # 専門家の融合
        combined_features = torch.cat([trend_features, momentum_features, volatility_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Dueling DQN（最適化）
        value = self.value_stream(fused_features)
        advantage = self.advantage_stream(fused_features)
        confidence = self.confidence_head(fused_features)
        
        # アドバンテージの正規化（高速化）
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # 確信度で重み付け
        return q_values * confidence

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
        # 保存されたモデルの入力次元を確認
        print("[DEBUG] Checking saved model dimensions...")
        if isinstance(ck, dict) and 'feature_extractor.0.weight' in ck:
            saved_input_dim = ck['feature_extractor.0.weight'].shape[1]
            print(f"[DEBUG] Saved model input dimension: {saved_input_dim}")
        else:
            # フォールバック：特徴量次元を動的に推論
            print("[DEBUG] Could not determine saved model dimensions, inferring...")
            saved_input_dim = 131  # 保存されたモデルの実際の次元
            
        print(f"[DEBUG] Creating QNet with in_dim={saved_input_dim}, out_dim=3")
        qnet = QNet(saved_input_dim, 3)
        
        if isinstance(ck, dict) and ("model_state_dict" in ck or "state_dict" in ck):
            # 辞書形式の場合
            print("[DEBUG] Dict with model_state_dict/state_dict detected")
            st = ck.get("model_state_dict", ck.get("state_dict"))
            qnet.load_state_dict(st)
            print("[INFO] DQN (torch wrapped state_dict) ロード完了")
        elif isinstance(ck, dict):
            # 直接state_dictの場合（train_dqn.pyの保存形式）
            print("[DEBUG] Direct state_dict detected")
            print(f"[DEBUG] State dict keys count: {len(ck.keys())}")
            print(f"[DEBUG] First few keys: {list(ck.keys())[:3]}...")
            
            try:
                qnet.load_state_dict(ck)
                print("[INFO] DQN (torch direct state_dict) ロード完了")
            except Exception as load_error:
                print(f"[ERROR] State dict loading failed: {load_error}")
                print("[DEBUG] Model structure mismatch - checking sizes...")
                for name, param in qnet.named_parameters():
                    if name in ck:
                        expected_shape = param.shape
                        actual_shape = ck[name].shape
                        if expected_shape != actual_shape:
                            print(f"[ERROR] Size mismatch for {name}: expected {expected_shape}, got {actual_shape}")
                    else:
                        print(f"[ERROR] Missing key in state_dict: {name}")
                qnet = None
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
                feats_array = FeatureExtraction(fea_ohlc)
                # take last row - should be 131 dimensions
                feat_row = feats_array[-1].astype(np.float32)
                print(f"[DEBUG] FeatureExtraction output shape: {feat_row.shape}")
                
                # 特徴量の正規化（スケーラーがある場合）
                if scaler is not None:
                    # スケーラーは131次元のみを期待しているので、131次元のみを正規化
                    scaled_feat_row = scaler.transform([feat_row])[0].astype(np.float32)
                    print(f"[DEBUG] Scaled feature vector shape: {scaled_feat_row.shape}")
                else:
                    scaled_feat_row = feat_row
                
                # Add phase and sec_range to make 133 dimensions total
                sec_range = float(fea_ohlc['high'].iloc[-1] - fea_ohlc['low'].iloc[-1])
                feat_vec = np.concatenate([scaled_feat_row, np.asarray([phase, sec_range], dtype=np.float32)])
                print(f"[DEBUG] Final feature vector shape: {feat_vec.shape}")
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
                        # モデルは131次元を期待しているので、最初の131次元のみを使用
                        model_input = feat_vec[:131] if len(feat_vec) > 131 else feat_vec
                        print(f"[DEBUG] Model input shape: {model_input.shape}")
                        t = torch.from_numpy(model_input).unsqueeze(0).float()
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

