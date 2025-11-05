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
import threading
import traceback
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

# トレンドフィルター設定（連敗システムは削除）
# すべての連敗ストッパー機能を削除しました

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

# ========================================
# 連敗システム関連（一時的にコメントアウト）
# ========================================
# def analyze_price_slope_and_losses(prices, price_times, loss_history):
#     """価格の傾きと直近の負け履歴を分析する（負けエントリー地点基準版）"""
#     ... (省略)
#
# def apply_slope_and_loss_filter(action_str, q_values, slope_analysis):
#     """価格傾きと負け履歴に基づくシンプルなフィルター（連敗時3分間ブロック機能付き）"""
#     ... (省略)
# ========================================

# -----------------------
# ========================================
# 連敗システム関連ここまで
# ========================================

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
    """ポップアップ・広告・モーダルを確実に閉じる（ログインダイアログは除外）"""
    try:
        print("[INFO] ポップアップ・広告の閉じ処理を開始...")
        
        # ログインダイアログが表示されている場合は処理をスキップ
        try:
            login_btn = page.query_selector('#btnSubmit')
            if login_btn and login_btn.is_visible():
                print("[INFO] ログインダイアログ表示中のため、ポップアップ閉じ処理をスキップ")
                return
        except Exception:
            pass
        
        # 1. チャットウィジェットを完全に削除
        try:
            page.evaluate("""
                // Intercomチャットを完全に削除（非表示ではなく削除）
                const chatIframes = document.querySelectorAll('iframe[title*="Intercom"], iframe.intercom-with-namespace-vo6dyv, iframe[name*="intercom"]');
                chatIframes.forEach(iframe => {
                    iframe.remove();  // DOMから削除
                });
                
                // チャットコンテナも削除
                const chatContainers = document.querySelectorAll('#intercom-container, .intercom-namespace, .intercom-with-namespace-vo6dyv');
                chatContainers.forEach(container => {
                    container.remove();  // DOMから削除
                });
                
                // Intercomスクリプトも無効化
                if (window.Intercom) {
                    try { window.Intercom('shutdown'); } catch(e) {}
                }
            """)
            print("[INFO] チャットウィジェット削除完了")
        except Exception as e:
            print(f"[WARN] チャット削除失敗: {e}")
        
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
        
        # 5. JavaScript実行で強制的にポップアップを削除（ログインフォームは除外）
        try:
            page.evaluate("""
                // ログインボタンが存在するかチェック
                const loginBtn = document.querySelector('#btnSubmit');
                if (loginBtn && loginBtn.offsetParent !== null) {
                    // ログインダイアログ表示中なので削除処理をスキップ
                    console.log('[INFO] ログインダイアログ表示中のため、削除処理スキップ');
                    return;
                }
                
                // 固定位置の要素（ポップアップの可能性）を削除
                const fixedElements = document.querySelectorAll('*');
                fixedElements.forEach(el => {
                    // ログインフォーム関連の要素は除外
                    if (el.id === 'loginForm' || el.closest('#loginForm') || 
                        el.querySelector('#btnSubmit') || el.closest('[class*="login"]')) {
                        return;
                    }
                    
                    const style = window.getComputedStyle(el);
                    if (style.position === 'fixed' && 
                        (style.zIndex > 1000 || el.classList.contains('modal') || 
                         el.classList.contains('popup') || el.classList.contains('dialog'))) {
                        el.style.display = 'none';
                    }
                });
                
                // 既知の広告・ポップアップクラスを削除（ログイン関連は除外）
                const adSelectors = [
                    '.advertisement', '.ad-banner', '.popup:not([class*="login"])', 
                    '.modal:not([class*="login"])', '.overlay:not([class*="login"])', 
                    '.lightbox', '.dialog:not([class*="login"])', '.notification'
                ];
                adSelectors.forEach(selector => {
                    try {
                        document.querySelectorAll(selector).forEach(el => {
                            if (!el.querySelector('#btnSubmit') && !el.closest('[class*="login"]')) {
                                if (el.style.zIndex > 100) el.style.display = 'none';
                            }
                        });
                    } catch(e) {}
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
            # Intercomチャットを削除
            page.evaluate("""
                const intercomContainer = document.querySelector('#intercom-container');
                if (intercomContainer) intercomContainer.remove();
                document.querySelectorAll('iframe[title*="Intercom"]').forEach(iframe => iframe.remove());
            """)
            
            # メールアドレス入力
            email_input = page.query_selector('input[type="email"]') or page.query_selector('input[name="email"]') or page.query_selector('.form-control.lg-input')
            if email_input:
                email_input.fill("")  # clear()の代わりにfill("")を使用
                email_input.type(email, delay=50)
            
            # パスワード入力  
            password_input = page.query_selector('input[type="password"]') or page.query_selector('input[name="password"]')
            if not password_input:
                inputs = page.query_selector_all('.form-control.lg-input')
                if len(inputs) >= 2:
                    password_input = inputs[1]
            
            if password_input:
                password_input.fill("")  # clear()の代わりにfill("")を使用
                password_input.type(passward, delay=50)
            
            # ログインボタンクリック（force=Trueで強制クリック）
            login_btn.click(force=True)
            print("[INFO] ログインボタンクリック完了、ページ遷移を待機...")
            
        except Exception as e:
            print(f"[WARN] Standard login failed, using fallback: {e}")
            # フォールバック: 従来の方法
            inputs = page.query_selector_all('.form-control.lg-input')
            if len(inputs) >= 2:
                inputs[0].fill(email)
                inputs[1].fill(passward)
            login_btn.click()
        
        # ログイン後の待機時間を長めに
        time.sleep(3)
        
        # ログインダイアログが消えるのを待つ
        try:
            print("[INFO] ログインダイアログの消失を待機...")
            page.wait_for_selector('#btnSubmit', state='hidden', timeout=10000)
            print("[INFO] ログインダイアログが閉じました")
        except Exception as e:
            print(f"[WARN] ログインダイアログ消失待機タイムアウト: {e}")
        
        # strikeWrapper待機
        try:
            page.wait_for_selector('.strikeWrapper div', timeout=5000)
            print("[INFO] strikeWrapper検出完了")
        except Exception:
            print("[WARN] strikeWrapper待機タイムアウト (セッション復帰遅延)")
        
        # ポップアップ閉じる（ログイン後の広告など）
        time.sleep(1)
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
# スクレイピング関数（削除）
# -----------------------
# 矢印スクレイピングと連敗ストッパーを全て削除しました

# -----------------------
# 連敗システム関連（全て削除）
# -----------------------
# evaluate_chart_arrows_and_pause, evaluate_recent_outcomes_and_pause,
# manual_add_arrow_result, test_chart_analysis 等の関数を全て削除しました
"""
            
            # 矢印らしい要素を特定
            potential_arrows = []
            for el in page_analysis['allElements']:
                # 実際の矢印の条件：
                # 1. 適度な小さいサイズ（三角矢印は小さなpath要素やpolygon要素）
                # 2. チャート領域内（価格ライン上）
                # 3. text要素やrect要素は除外（これらはUIパーツ）
                if (3 <= el['width'] <= 30 and 3 <= el['height'] <= 30 and 
                    50 <= el['x'] <= 850 and 200 <= el['y'] <= 700 and  # チャート内の価格ライン付近（範囲拡張）
                    el['tagName'].lower() not in ['text', 'rect'] and  # UI要素を除外
                    el['tagName'].lower() in ['path', 'polygon', 'circle']):  # 矢印の可能性がある要素タイプ
                    
                    # 色による判定（より柔軟に）
                    result = None
                    
                    # 色情報を安全に文字列化
                    fill = str(el.get('fill', ''))
                    stroke = str(el.get('stroke', ''))
                    backgroundColor = str(el.get('backgroundColor', ''))
                    color = str(el.get('color', ''))
                    all_colors = f"{fill} {stroke} {backgroundColor} {color}".lower()
                    
                    # 実際の矢印の色パターン（TheOptionの実際の仕様に合わせて修正）
                    
                    # 負け矢印の色パターン（グレー系 + 黒系）
                    loss_patterns = [
                        # グレー系
                        'gray', 'grey', '#666', '#999', '#ccc', '#808080', '#888', '#aaa',
                        'rgb(128', 'rgb(169', 'rgb(192', 'rgb(105', 'rgb(92, 91, 91)',
                        'rgba(128', 'rgba(169', 'rgba(192', 'rgba(105',
                        '92, 91, 91', 'rgb(78, 71, 78)',  # 実際に検出されたグレー色
                        # 黒系（負け矢印として追加）
                        'black', '#000000', '#000', 'rgb(0, 0, 0)', 'rgba(0, 0, 0',
                        'rgb(51, 51, 51)', 'rgb(33, 33, 33)'  # ダークグレー系も負け
                    ]
                    
                    # 緑系の色（勝ち矢印）- 実際に検出された緑色を含む
                    green_win_patterns = [
                        'green', '#00ff00', '#0f0', 'rgb(0, 255, 0)', 'rgb(0, 128, 0)',
                        'rgb(34, 139, 34)', '#228b22', '#006400', '#32cd32',
                        'rgb(13, 159, 27)',  # 実際に検出された緑色（勝ち矢印）
                        # 追加の緑系バリエーション
                        'rgb(0, 255', 'rgb(0, 128', 'rgb(13, 159', 'rgb(34, 139',
                        '#0d9f1b', '#22b14c', '#2ecc71', '#27ae60', '#16a085',
                        'rgba(13, 159', 'rgba(0, 255', 'rgba(0, 128',
                        '13, 159, 27', '0, 255, 0', '0, 128, 0'  # RGB値のみのパターン
                    ]
                    
                    # 赤系の色（勝ち矢印）
                    red_patterns = [
                        'red', '#ff0000', '#f00', 'rgb(255, 0, 0)', 'rgb(220, 20, 60)',
                        'rgb(255,', '#dc143c', '#b22222', '#8b0000',
                        # 追加の赤系バリエーション
                        'rgb(255, 0', 'rgb(220, 20', 'rgb(178, 34',
                        '#e74c3c', '#c0392b', '#e67e22', '#d35400',
                        'rgba(255, 0', 'rgba(220, 20',
                        '255, 0, 0', '220, 20, 60'  # RGB値のみのパターン
                    ]
                    
                    # 白い背景の矢印（矢印の背景部分）
                    white_patterns = [
                        'rgb(255, 255, 255)', '#ffffff', '#fff', 'white',
                        'rgba(255, 255, 255'
                    ]
                    
                    # 青系の色（特殊な矢印または現在位置マーカー）
                    blue_patterns = [
                        'blue', 'rgb(2, 62, 210)', '#0000ff', '#00f'
                    ]
                    
                    # 矢印らしい要素の判定
                    is_arrow_like = False
                    
                    # 色による判定（勝ち判定を優先、黒・グレーのみ負け判定）
                    matched_patterns = []  # マッチした色パターンを記録
                    
                    # 優先順位1: 緑色（勝ち矢印）を先にチェック
                    if any(pattern in all_colors for pattern in green_win_patterns):
                        result = 'win'  
                        is_arrow_like = True
                        matched_patterns = [p for p in green_win_patterns if p in all_colors]
                    # 優先順位2: 赤色（勝ち矢印）
                    elif any(pattern in all_colors for pattern in red_patterns):
                        result = 'win'  
                        is_arrow_like = True
                        matched_patterns = [p for p in red_patterns if p in all_colors]
                    # 優先順位3: 黒・グレー（負け矢印）
                    elif any(pattern in all_colors for pattern in loss_patterns):
                        result = 'loss'
                        is_arrow_like = True
                        matched_patterns = [p for p in loss_patterns if p in all_colors]
                    elif any(pattern in all_colors for pattern in blue_patterns):
                        result = 'unknown'  # 青は特殊マーカー、矢印ではない可能性
                        is_arrow_like = False  # 青いマーカーは除外
                    elif any(pattern in all_colors for pattern in white_patterns):
                        # 白い要素は矢印の背景部分の可能性
                        # path要素で小さければ矢印の一部
                        if el['tagName'].lower() == 'path' and el['width'] <= 15 and el['height'] <= 15:
                            result = 'unknown'  # 色が不明なので判定保留
                            is_arrow_like = True
                        else:
                            is_arrow_like = False
                    
                    # 緊急モード: 全ての検出された矢印を負けとして扱う
                    if EMERGENCY_MODE:
                        # 三角矢印の可能性がある要素（path, polygon, circle）
                        if (el['tagName'].lower() in ['circle', 'path', 'polygon'] and 
                            3 <= el['width'] <= 25 and 3 <= el['height'] <= 25 and
                            not any(pattern in all_colors for pattern in blue_patterns)):
                            result = 'loss'  # 緊急モードでは全て負け
                            is_arrow_like = True
                    
                    # 特定の形状やクラス名で矢印を判定
                    classes_str = str(el.get('classes', ''))
                    if any(keyword in classes_str.lower() for keyword in ['arrow', 'marker', 'indicator', 'signal']):
                        if not EMERGENCY_MODE:
                            is_arrow_like = True
                            if not result:
                                result = 'unknown'
                    
                    # 現在位置マーカーは除外
                    if 'current-coordinate' in classes_str:
                        is_arrow_like = False
                    
                    # circle要素の場合（矢印の背景円）
                    if el['tagName'].lower() == 'circle' and not EMERGENCY_MODE:
                        # 緑・赤・グレーの円は矢印
                        if result in ['win', 'loss']:
                            is_arrow_like = True
                        # 青い円（現在位置マーカーなど）は除外
                        elif any(pattern in all_colors for pattern in blue_patterns):
                            is_arrow_like = False
                        else:
                            is_arrow_like = False
                    
                    # path要素で小さなもの（矢印の形状部分）
                    if (el['tagName'].lower() == 'path' and 
                        5 <= el['width'] <= 20 and 5 <= el['height'] <= 20 and
                        not EMERGENCY_MODE):
                        if result:
                            is_arrow_like = True
                    
                    # 条件に合致した要素のみを候補とする
                    if is_arrow_like and result:
                        potential_arrows.append({
                            'result': result,
                            'x': el['x'],
                            'y': el['y'],
                            'width': el['width'],
                            'height': el['height'],
                            'colors': all_colors,
                            'tagName': el['tagName'],
                            'element': el,
                            'matched_patterns': matched_patterns  # マッチした色パターンを記録
                        })
            
            # 重複要素を除外（同じ位置にあるpath要素とcircle要素）
            unique_arrows = []
            for arrow in potential_arrows:
                # 同じ位置（±5px以内）に既存の矢印がないかチェック
                is_duplicate = False
                for existing in unique_arrows:
                    if (abs(arrow['x'] - existing['x']) <= 5 and 
                        abs(arrow['y'] - existing['y']) <= 5):
                        # 既存の矢印と重複している場合
                        # より確実な判定結果（win/loss）を優先
                        if arrow['result'] in ['win', 'loss'] and existing['result'] == 'unknown':
                            # 新しい矢印の方が確実な結果なので置き換える
                            unique_arrows[unique_arrows.index(existing)] = arrow
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_arrows.append(arrow)
            
            potential_arrows = unique_arrows
            
            # X座標でソート（右から左へ）
            potential_arrows.sort(key=lambda x: x['x'], reverse=True)
            
            # 最新10個の矢印のみを使用（unknownを除外）
            definite_arrows = [arrow for arrow in potential_arrows if arrow['result'] in ['win', 'loss']]
            recent_arrows = definite_arrows[:10]
            
            print(f"\n[🏆 FINAL ARROWS] 検出された確定矢印: {len(recent_arrows)}個")
            
            arrow_results = []
            for idx, arrow in enumerate(recent_arrows):
                arrow_results.append(arrow['result'])
                status_emoji = "💀" if arrow['result'] == 'loss' else "✅"
                print(f"  {status_emoji} #{idx+1}: {arrow['result']}")
            
            print(f"\n[📊 RESULT] 確定矢印: {len(arrow_results)}個")
                
            if len(arrow_results) >= DYNAMIC_MIN_COUNT:
                loss_count = arrow_results.count('loss')
                win_count = arrow_results.count('win')
                print(f"  勝ち: {win_count}個, 負け: {loss_count}個")
            else:
                print(f"  ⚠️ データ不足（最低{DYNAMIC_MIN_COUNT}個必要）")
                return []
            
            return arrow_results
        
        except Exception as js_error:
            print(f"[ERROR] JavaScript解析エラー: {js_error}")
            return []
        
    except Exception as e:
        print(f"[❌ CHART ERROR] チャート矢印解析エラー: {e}")
        return []
        print(traceback.format_exc())
        return []

def scrape_trade_results(page):
    Webページから取引結果をスクレイピング（チャート矢印版）
    Args:
        page: Playwrightのページオブジェクト
    Returns:
        list: [(entry_time, action_str, result, entry_price), ...]
        # チャート上の矢印を解析
        arrow_results = scrape_chart_arrows(page)
        
        # 従来の取引履歴も確認（参考用）
        print(f"\n[📋 BACKUP] 取引履歴パネルも確認...")
        
        return []  # チャート解析結果は別途処理
        
    except Exception as e:
        print(f"[❌ SCRAPE ERROR] 取引結果スクレイピングエラー: {e}")
        import traceback
        print(traceback.format_exc())
        return []

# ========================================
# 連敗システム関連の取引結果チェック（一時的にコメントアウト）
# ========================================
# def check_trade_result(entry_time, action_str, entry_price, loss_history_ref, page):
#     """
#     取引結果をスクレイピングで確認して負け履歴に追加
#     Args:
#         entry_time: エントリー時刻
#         action_str: アクション（'High' or 'Low'）
#         entry_price: エントリー価格
#         loss_history_ref: 負け履歴リストの参照
#         page: Playwrightのページオブジェクト
#     """
#     try:
#         print(f"\n{'='*60}")
#         print(f"[⏰ RESULT CHECK] {entry_time.strftime('%H:%M:%S')}の{action_str}取引結果確認")
#         print(f"[💰 ENTRY] エントリー価格: {entry_price:.3f}")
#         print(f"{'='*60}")
#         
#         # スクレイピングで取引結果を取得
#         results = scrape_trade_results(page)
#         
#         # 結果から該当する取引を探す（時刻とアクションで一致判定）
#         found = False
#         for result_time, result_action, result_status, result_price in results:
#             time_diff = abs((result_time - entry_time).total_seconds())
#             if time_diff < 10 and result_action == action_str:  # 10秒以内の一致
#                 found = True
#                 if result_status == 'loss':
#                     loss_history_ref.append((entry_time, action_str, 'loss', entry_price))
#                     print(f"[❌ RESULT] 負け記録追加: {action_str} @ {entry_price:.3f}")
#                     # append recent outcome and evaluate
#                     try:
#                         recent_trade_outcomes.append('loss')
#                         print(f"[RECENT] recent_trade_outcomes: {list(recent_trade_outcomes)}")
#                         paused, triggered = evaluate_recent_outcomes_and_pause(recent_trade_outcomes)
#                         if triggered:
#                             global trading_paused_until
#                             trading_paused_until = paused
#                     except Exception as e:
#                         print(f"[WARN] recent append failed: {e}")
#                 else:
#                     print(f"[✅ RESULT] 勝ち: {action_str} @ {entry_price:.3f}")
#                     try:
#                         recent_trade_outcomes.append('win')
#                         print(f"[RECENT] recent_trade_outcomes: {list(recent_trade_outcomes)}")
#                     except Exception:
#                         pass
#                 break
#         
#         if not found:
#             print(f"\n[⚠️ WARNING] 自動検出できませんでした")
#             print(f"[📋 手動記録方法]")
#             print(f"  負けた場合:")
#             print(f"    add_loss_to_history(loss_history, '{action_str}', {entry_price:.3f})")
#             print(f"  勝った場合:")
#             print(f"    （何もしなくてOK）")
#             print(f"{'='*60}\n")
#         
#     except Exception as e:
#         print(f"[ERROR] 取引結果確認エラー: {e}")
# 
# def add_loss_to_history(loss_history, action_str, entry_price, entry_time=None):
#     """
#     手動で負け履歴に追加するヘルパー関数
#     Args:
#         loss_history: 負け履歴リスト
#         action_str: 負けたアクション（'High' or 'Low'）
#         entry_price: エントリー価格
#         entry_time: エントリー時刻（Noneの場合は現在時刻）
#     """
#     if entry_time is None:
#         entry_time = datetime.now()
#     
#     loss_history.append((entry_time, action_str, 'loss', entry_price))
#     print(f"\n{'='*60}")
#     print(f"[✍️ MANUAL LOSS] 負け履歴を手動追加しました")
#     print(f"[📅 時刻] {entry_time.strftime('%H:%M:%S')}")
#     print(f"[📊 方向] {action_str}")
#     print(f"[💰 価格] {entry_price:.3f}")
#     print(f"[📈 累計] 直近の負け - High:{sum(1 for _, a, _, _ in loss_history if a == 'High')}回, Low:{sum(1 for _, a, _, _ in loss_history if a == 'Low')}回")
#     print(f"{'='*60}\n")
#     
#     # 古い履歴をクリーンアップ
#     cutoff_time = datetime.now() - timedelta(minutes=LOSS_LOOKBACK_MINUTES * 2)
#     old_count = len(loss_history)
#     loss_history[:] = [loss for loss in loss_history if loss[0] > cutoff_time]
#     cleaned = old_count - len(loss_history)
#     if cleaned > 0:
#         print(f"[🗑️ CLEANUP] 古い履歴{cleaned}件を削除しました")
# 
#     # recent_trade_outcomesに'loss'を追加して評価
#     try:
#         recent_trade_outcomes.append('loss')
#         print(f"[RECENT] recent_trade_outcomes: {list(recent_trade_outcomes)}")
#         paused, triggered = evaluate_recent_outcomes_and_pause(recent_trade_outcomes)
#         if triggered:
#             global trading_paused_until
#             trading_paused_until = paused
#     except Exception as e:
#         print(f"[WARN] recent outcome append failed: {e}")
# ========================================
# 連敗システム関連ここまで
# ========================================

# ========================================
# 連敗システム関連（一時的にコメントアウト）
# ========================================
# def add_last_trade_loss(loss_history, pending_trades):
#     """
#     直近のエントリーを負けとして記録する簡易関数
#     Args:
#         loss_history: 負け履歴リスト
#         pending_trades: 待機中の取引リスト
#     """
#     if pending_trades:
#         # 最後のエントリーを取得
#         last_trade = pending_trades[-1]
#         entry_time, action_str, entry_price = last_trade
#         add_loss_to_history(loss_history, action_str, entry_price, entry_time)
#         # 待機リストから削除
#         pending_trades.remove(last_trade)
#         print(f"[INFO] 待機リストから削除しました")
#     else:
#         print(f"[⚠️ WARNING] 記録する取引がありません")
#         print(f"[INFO] 直接記録する場合:")
#         print(f"  add_loss_to_history(loss_history, 'High', 150.123)  # High負け")
#         print(f"  add_loss_to_history(loss_history, 'Low', 150.123)   # Low負け")
# ========================================
# 連敗システム終わり
# ========================================

def _log_signal(ts, price, phase, q_values, action_idx, action_str, entry, reason, slope_info=None):
    try:
        q_hold = q_values[0] if q_values is not None else ""
        q_high = q_values[1] if q_values is not None else ""
        q_low  = q_values[2] if q_values is not None else ""
        
        # 傾き・負け履歴情報を理由に追加
        if slope_info:
            slope_suffix = f"|slope:{slope_info['price_slope']:.6f}"
            slope_suffix += f"|decline:{slope_info['is_declining']}"
            slope_suffix += f"|high_losses:{slope_info['recent_high_losses']}"
            slope_suffix += f"|low_losses:{slope_info['recent_low_losses']}"
            if slope_info['should_block_high']:
                slope_suffix += "|BLOCK_HIGH"
            elif slope_info['should_block_low']:
                slope_suffix += "|BLOCK_LOW"
            reason = (reason or "") + slope_suffix
        
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
    print("[INFO] サイトを開きました。ポップアップの表示を待機中...")

    # サイトを開いて10秒待機（ポップアップが出現するまで）
    time.sleep(10)
    print("[INFO] 10秒経過。ポップアップを閉じます...")
    
    # ポップアップを閉じる（複数回試行）
    for i in range(3):
        try_close_popups(page)
        time.sleep(1)
        print(f"[INFO] ポップアップ閉じ試行 {i+1}/3 完了")
    
    print("[INFO] ポップアップ処理完了。ログインを開始します...")

    # ログイン前にIntercomチャットを完全に削除
    try:
        page.evaluate("""
            // Intercomチャット関連を完全に削除
            const intercomContainer = document.querySelector('#intercom-container');
            if (intercomContainer) {
                intercomContainer.remove();
            }
            const chatIframes = document.querySelectorAll('iframe[title*="Intercom"]');
            chatIframes.forEach(iframe => iframe.remove());
        """)
        print("[INFO] Intercomチャットを削除しました")
    except Exception as e:
        print(f"[WARN] Intercom削除失敗: {e}")
    
    # ログイン前に少し待機
    time.sleep(2)
    try:
        # メールアドレス入力
        email_input = page.query_selector('input[type="email"]') or page.query_selector('input[name="email"]') or page.query_selector('.form-control.lg-input')
        if email_input:
            email_input.fill("")  # clear()の代わりにfill("")を使用
            email_input.type(email, delay=100)
            print(f"[INFO] Email entered: {email}")
        
        # パスワード入力  
        password_input = page.query_selector('input[type="password"]') or page.query_selector('input[name="password"]')
        if not password_input:
            inputs = page.query_selector_all('.form-control.lg-input')
            if len(inputs) >= 2:
                password_input = inputs[1]
        
        if password_input:
            password_input.fill("")  # clear()の代わりにfill("")を使用
            password_input.type(passward, delay=100)
            print(f"[INFO] Password entered")
        
        # ログインボタンクリック（force=Trueで強制クリック）
        login_btn = page.query_selector('#btnSubmit') or page.query_selector('button[type="submit"]') or page.query_selector('.btn-primary')
        if login_btn:
            login_btn.click(force=True)
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
    
    # ログイン後の待機
    print("[INFO] ログイン処理完了。ページ遷移を待機中...")
    time.sleep(3)
    
    # ログインダイアログが消えるのを待つ
    try:
        page.wait_for_selector('#btnSubmit', state='hidden', timeout=10000)
        print("[INFO] ログインダイアログが閉じました")
    except Exception as e:
        print(f"[WARN] ログインダイアログ消失待機タイムアウト: {e}")
    
    # strikeWrapper待機
    try:
        page.wait_for_selector(".strikeWrapper div", timeout=20000)
        print("[INFO] 取引画面の読み込み完了")
    except Exception:
        print("[WARN] strikeWrapper待機タイムアウト")
    
    # ログイン後のポップアップを閉じる
    time.sleep(1)
    try_close_popups(page)
    print("[INFO] 初期化完了。取引ループを開始します...")

    # ループ準備
    all_ticks = []
    # loss_history = []  # 負け履歴: [(datetime, action_str, result, entry_price), ...] ※連敗システムコメントアウト
    # pending_trades = []  # エントリー待ちの取引: [(entry_time, action_str, entry_price), ...] ※連敗システムコメントアウト
    # recent_trade_outcomes and trading_paused_until are module-level
    last_entry_time = None
    next_entry_allowed_time = None
    recent_prices = deque(maxlen= int(10 / max(TICK_INTERVAL_SECONDS, 0.001)) + 2)
    
    print("\n" + "="*80)
    print("🤖 DQN自動取引BOT - チャート矢印フィルター機能付き")
    print("="*80)
    print("\n📊 負け履歴管理機能の使い方")
    print("-"*80)
    print("\n【方法1】最も簡単 - 直近の取引が負けた場合:")
    print("  >>> add_last_trade_loss(loss_history, pending_trades)")
    print("")
    print("【方法2】個別に記録する場合:")
    print("  High負け:")
    print("    >>> add_loss_to_history(loss_history, 'High', 150.123)")
    print("  Low負け:")
    print("    >>> add_loss_to_history(loss_history, 'Low', 150.456)")
    print("")
    print("【方法3】過去の取引を記録する場合:")
    print("  >>> from datetime import datetime, timedelta")
    print("  >>> past_time = datetime.now() - timedelta(minutes=2)")
    print("")
    print("🔥 すべての連敗ストッパーと矢印スクレイピングを削除しました")
    print("   DQNの判定のみで取引を行います")
    print("="*80 + "\n")

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
                    
                    # Epsilon-Greedy: 探索のため一定確率でランダム行動
                    exploration_rate = 0.15  # 15%の確率でランダム選択
                    if np.random.random() < exploration_rate:
                        action_idx = np.random.choice([0, 1, 2])
                        print(f"[EXPLORATION] ランダム選択 (epsilon={exploration_rate})")
                    else:
                        # Q値から行動を選択（単純にargmax）
                        action_idx = int(np.argmax(q_values))
                    
                    # デバッグ用: 各Q値の詳細表示
                    hold_q = q_values[0]
                    high_q = q_values[1]
                    low_q = q_values[2]
                    
                    # map idx -> action: 0=Hold,1=High,2=Low
                    action_map = {0:"Hold", 1:"High", 2:"Low"}
                    action_str = action_map.get(action_idx, "Hold")
                    
                    # Q値の詳細をログ出力（デバッグ用）
                    print(f"[Q-VALUES] Hold:{q_values[0]:.4f}, High:{q_values[1]:.4f}, Low:{q_values[2]:.4f}")
                    print(f"[ACTION] 選択されたアクション: {action_str} (idx:{action_idx})")


                    
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

            # 連敗ストッパーと矢印スクレイピングを全て削除しました
            # チャート矢印による一時停止システムを廃止

            # ========================================
            # 連敗システムによるフィルター（削除済み）
            # ========================================
            # time_history = [t[0] for t in all_ticks[-TREND_LOOKBACK_PERIODS:]] if len(all_ticks) >= TREND_LOOKBACK_PERIODS else [t[0] for t in all_ticks]
            # slope_analysis = analyze_price_slope_and_losses(price_history, time_history, loss_history)
            # 
            # # 傾きと負け履歴フィルターを適用
            # original_action = action_str
            # action_str, filter_reason = apply_slope_and_loss_filter(action_str, q_values, slope_analysis)
            # 
            # # アクションが変更された場合、action_idxも更新
            # if action_str != original_action:
            #     action_map_reverse = {"Hold": 0, "High": 1, "Low": 2}
            #     action_idx = action_map_reverse.get(action_str, 0)
            # 
            # # 傾き・負け履歴情報をログ出力
            # if TREND_FILTER_ENABLED:
            #     direction = "下降" if slope_analysis['is_declining'] else "上昇/横ばい"
            #     print(f"\n[📈 SLOPE] 傾き方向:{direction}, 傾き値:{slope_analysis['price_slope']:.8f}")
            #     
            #     if slope_analysis['loss_entry_point']:
            #         loss_time, loss_price = slope_analysis['loss_entry_point']
            #         print(f"[📍 SLOPE] 基準点: {loss_time.strftime('%H:%M:%S')} @ {loss_price:.3f} (最初の負けエントリー)")
            #     else:
            #         print(f"[📍 SLOPE] 基準点: 直近{TREND_LOOKBACK_PERIODS}期間の線形回帰")
            #     
            #     print(f"[📊 LOSS] 直近負け - High:{slope_analysis['recent_high_losses']}回, Low:{slope_analysis['recent_low_losses']}回")
            # 
            #     # ブロック状態の詳細表示
            #     if slope_analysis['should_block_high']:
            #         block_until = slope_analysis.get('block_high_until')
            #         if block_until:
            #             remaining = int((block_until - current_time).total_seconds())
            #             if remaining > 0:
            #                 print(f"[🚫 BLOCK] High判定ブロック中 - 残り{remaining}秒（{block_until.strftime('%H:%M:%S')}まで）")
            #             else:
            #                 print(f"[⏰ BLOCK] Highブロック期間終了 - トレンド確認中")
            #         else:
            #             print(f"[🚫 WARNING] High判定ブロック条件検出")
            #     
            #     if slope_analysis['should_block_low']:
            #         block_until = slope_analysis.get('block_low_until')
            #         if block_until:
            #             remaining = int((block_until - current_time).total_seconds())
            #             if remaining > 0:
            #                 print(f"[🚫 BLOCK] Low判定ブロック中 - 残り{remaining}秒（{block_until.strftime('%H:%M:%S')}まで）")
            #             else:
            #                 print(f"[⏰ BLOCK] Lowブロック期間終了 - トレンド確認中")
            #         else:
            #             print(f"[🚫 WARNING] Low判定ブロック条件検出")
            #     
            #     if original_action != action_str:
            #         print(f"[🛡️ FILTER] アクション変更: {original_action} -> {action_str}")
            # ========================================
            # 連敗システム終わり
            # ========================================

            # Decide entry: skip Hold
            if action_str == "Hold":
                # reason = filter_reason or "hold"  # 連敗システムコメントアウト
                reason = "hold"
                entry = False
                print(f"[{current_time.strftime('%H:%M:%S')}] Hold - Q値: Hold={q_values[0]:.3f}, High={q_values[1]:.3f}, Low={q_values[2]:.3f}")
                # if filter_reason:  # 連敗システムコメントアウト
                #     print(f"[{current_time.strftime('%H:%M:%S')}] トレンドフィルターによりHold: {filter_reason}")
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
                            # reason = filter_reason or "entry_executed"  # 連敗システムコメントアウト
                            reason = "entry_executed"
                            print(f"[ENTRY] {action_str} at {current_time.strftime('%H:%M:%S')} price={current_price} Q値: {q_values[action_idx]:.3f} (優位性: {q_advantage:.3f})")
                            # if original_action != action_str:  # 連敗システムコメントアウト
                            #     print(f"[ENTRY] 元の予測:{original_action} -> トレンドフィルター適用後:{action_str}")
                            
                            # 取引を待ちリストに追加（60秒後に結果確認）※連敗システムコメントアウト
                            # pending_trades.append((current_time, action_str, current_price))
                            # print(f"[INFO] 取引を待ちリストに追加（60秒後に結果確認）")
                        else:
                            reason = "button_not_found"
                            entry = False
                            print(f"[WARN] {action_str}ボタンが見つかりません")
                else:
                    reason = "insufficient_q_advantage"
                    entry = False
                    print(f"[{current_time.strftime('%H:%M:%S')}] {action_str} - Q値優位性不足 ({q_advantage:.3f} < {DQN_Q_MARGIN})")

            # log
            # slope_info = slope_analysis if 'slope_analysis' in locals() else None  # 連敗システムコメントアウト
            slope_info = None
            _log_signal(current_time, current_price, phase, q_values, action_idx, action_str, entry, reason, slope_info)

            # ========================================
            # 待機中の取引結果確認（連敗システム）※コメントアウト
            # ========================================
            # 連敗システム関連（一時的にコメントアウト）
            # ========================================
            # 待機中の取引結果を確認（60秒経過したもの）
            # completed_trades = []
            # for trade_time, trade_action, trade_price in pending_trades[:]:
            #     time_elapsed = (current_time - trade_time).total_seconds()
            #     if time_elapsed >= 60:  # 60秒経過（1分BO終了）
            #         print(f"\n[⏰ CHECK] {trade_action}取引の結果確認 (エントリー: {trade_time.strftime('%H:%M:%S')} @ {trade_price:.3f})")
            #         check_trade_result(trade_time, trade_action, trade_price, loss_history, page)
            #         completed_trades.append((trade_time, trade_action, trade_price))
            #     elif time_elapsed >= 50:  # 50秒経過で事前通知
            #         remaining = 60 - int(time_elapsed)
            #         if remaining > 0 and remaining % 5 == 0:  # 5秒ごとに表示
            #             print(f"[⏳ PENDING] {trade_action}取引の結果確認まで{remaining}秒...")
            # 
            # # 確認済みの取引を待ちリストから削除
            # for completed in completed_trades:
            #     if completed in pending_trades:
            #         pending_trades.remove(completed)
            # 
            # # 待機中の取引がある場合は表示（1分ごと）
            # if pending_trades and current_time.second % 60 < TICK_INTERVAL_SECONDS:
            #     print(f"\n[📋 PENDING TRADES] 結果待ち: {len(pending_trades)}件")
            #     for trade_time, trade_action, trade_price in pending_trades:
            #         elapsed = int((current_time - trade_time).total_seconds())
            #         print(f"  - {trade_action} @ {trade_price:.3f} ({trade_time.strftime('%H:%M:%S')}, {elapsed}秒経過)")
            # ========================================
            # 連敗システム終わり
            # ========================================            # 状態を定期的に表示（30秒ごと）
            if current_time.second % 30 < TICK_INTERVAL_SECONDS:
                print(f"\n[📊 STATUS] システム状態")
                
                # チャート矢印による一時停止システムは削除されました
                print(f"  ✅ 連敗ストッパー: 廃止済み")
                
                # recent arrow statusも削除されました
            
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

