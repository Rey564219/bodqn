# train_dqn.py
import os

# PyTorchのDynamo機能を無効化（最優先で設定）
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['TORCH_DISABLE_DYNAMIC_SHAPES'] = '1'
os.environ['PYTORCH_DISABLE_DYNAMO'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import pickle, random, math
import numpy as np
import pandas as pd

import torch, torch.nn as nn, torch.optim as optim

from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import talib as ta
import numpy as np
from functools import lru_cache
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

pair = "BTCUSD"
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
    # 大量データの場合、計算する期間を制限
    if len(df) > 100:
        # 最後の部分のみ計算（効率化）
        calc_df = df.iloc[-min(500, len(df)):].copy()  # 最大500行まで
        
        for period in periods:
            if len(calc_df) < period:
                continue  # 期間が不足している場合はスキップ
                
            # ボリンジャーバンド幅
            try:
                upper, middle, lower = ta.BBANDS(calc_df['close'], period)
                calc_df['bb_width'+str(period)] = upper - lower
                calc_df['bb_width_diff' + str(period)] = calc_df['bb_width'+str(period)].diff()
                
                # ボリンジャーバンドの位置（%B）- ゼロ除算防止
                width = upper - lower
                calc_df['bb_percent'+str(period)] = np.where(width != 0, (calc_df['close'] - lower) / width, 0.5)
                
                # ATR（平均的な値幅）
                calc_df['atr'+str(period)] = ta.ATR(calc_df['high'], calc_df['low'], calc_df['close'], period)
                calc_df['atr_diff' + str(period)] = calc_df['atr'+str(period)].diff()
                
                # 価格変動率 - 異常値防止
                calc_df['price_change'+str(period)] = calc_df['close'].pct_change(period).clip(-1, 1)
                
                # ボラティリティ
                calc_df['volatility'+str(period)] = calc_df['close'].rolling(period).std()
                
                # 新しい特徴量：価格の勢い（軽量版）
                calc_df['momentum'+str(period)] = ta.MOM(calc_df['close'], period)
                calc_df['momentum_norm'+str(period)] = calc_df['momentum'+str(period)] / calc_df['close']
                
                # 価格の位置（高値・安値に対する相対位置）
                high_max = calc_df['high'].rolling(period).max()
                low_min = calc_df['low'].rolling(period).min()
                calc_df['high_pos'+str(period)] = (calc_df['close'] - low_min) / (high_max - low_min + 1e-8)
                
            except Exception as e:
                print(f"[WARNING] Error calculating period {period}: {e}")
                continue
        
        # 元のデータフレームに結果を統合（最後の行のみ）
        for col in calc_df.columns:
            if col not in df.columns:
                df[col] = np.nan
                df.iloc[-1, df.columns.get_loc(col)] = calc_df.iloc[-1][col]
    else:
        # 小さなデータの場合は通常の計算
        for period in periods:
            if len(df) < period:
                continue
                
            # ボリンジャーバンド幅
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
            df['momentum_norm'+str(period)] = df['momentum'+str(period)] / df['close']
            
            # 価格の位置（高値・安値に対する相対位置）
            high_max = df['high'].rolling(period).max()
            low_min = df['low'].rolling(period).min()
            df['high_pos'+str(period)] = (df['close'] - low_min) / (high_max - low_min + 1e-8)
    
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
    
    # 異常に大きな値をクリップ
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # 99.9%分位点でクリップ
        upper_limit = result[col].quantile(0.999)
        lower_limit = result[col].quantile(0.001)
        result[col] = result[col].clip(lower=lower_limit, upper=upper_limit)
    
    if use_cache:
        _feature_cache[df_hash] = result
        # キャッシュサイズ制限
        if len(_feature_cache) > 10000:
            # 古いキャッシュを削除
            oldest_key = next(iter(_feature_cache))
            del _feature_cache[oldest_key]
    
    return result
ACTIONS = 3  # 0:Hold, 1:High, 2:Low

class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # より深く、幅広いネットワークで表現力向上
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # 価値関数とアドバンテージ関数を分離（Dueling DQN）
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)
        )
    
    def forward(self, x): 
        features = self.feature_extractor(x)
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # アドバンテージの正規化
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values

class Replay:
    def __init__(self, cap=100_000): 
        self.buf = deque(maxlen=cap)
        self._cached_arrays = None
        self._cache_valid = False
        # Prioritized Replay用
        self.priorities = deque(maxlen=cap)
        self.alpha = 0.6  # priorityの重み
        self.beta = 0.4   # importance sampling補正
        self.beta_increment = 0.001
        self.epsilon = 1e-6  # ゼロ除算防止
    
    def push(self, *exp, priority=None): 
        self.buf.append(exp)
        # 新しい経験には高い優先度を設定
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(priority)
        self._cache_valid = False
    
    def sample(self, bs):
        if len(self.priorities) == 0:
            # 通常のランダムサンプリング
            indices = np.random.choice(len(self.buf), bs, replace=False)
        else:
            # Prioritized sampling
            priorities = np.array(self.priorities, dtype=np.float32)
            priorities = priorities ** self.alpha
            prob = priorities / priorities.sum()
            
            indices = np.random.choice(len(self.buf), bs, replace=True, p=prob)
            
            # Importance sampling weights
            weights = (len(self.buf) * prob[indices]) ** (-self.beta)
            weights = weights / weights.max()  # 正規化
            
        batch = [self.buf[i] for i in indices]
        
        # numpy配列への変換を最適化
        s, a, r, ns, done = zip(*batch)
        
        result = (np.stack(s, axis=0).astype(np.float32),
                 np.array(a, dtype=np.int64),
                 np.array(r, dtype=np.float32),
                 np.stack(ns, axis=0).astype(np.float32),
                 np.array(done, dtype=np.float32))
        
        if len(self.priorities) > 0:
            self.beta = min(1.0, self.beta + self.beta_increment)
            return result + (indices, weights.astype(np.float32))
        else:
            return result + (indices, np.ones(bs, dtype=np.float32))
    
    def update_priorities(self, indices, td_errors):
        """TD誤差に基づいて優先度を更新"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            self.priorities[idx] = priority
    
    def __len__(self): return len(self.buf)

def build_state_vec(ohlc_win_df, extra=None, use_cache=True):
    feat = FeatureExtraction(ohlc_win_df, use_cache=use_cache)[-1:]
    x = feat.values.astype(np.float32).reshape(-1)  # 明示的にfloat32に変換
    if extra is not None:
        x = np.concatenate([x, np.asarray(extra, dtype=np.float32)])
    return x.astype(np.float32)  # 確実にfloat32にする

def build_state_vec_fast(ohlc_slice, phase, sec_range):
    """高速版の状態ベクトル構築（キャッシュなし）"""
    feat = FeatureExtraction(ohlc_slice, use_cache=False)[-1:]
    x = feat.values.astype(np.float32).reshape(-1)
    x = np.concatenate([x, np.asarray([phase, sec_range], dtype=np.float32)])
    return x.astype(np.float32)

# バッチ処理用の関数を追加
def build_state_batch(ohlc_data_list, extra_list=None):
    """複数の状態ベクトルを一度に処理"""
    states = []
    for i, ohlc_win_df in enumerate(ohlc_data_list):
        extra = extra_list[i] if extra_list else None
        state = build_state_vec(ohlc_win_df, extra, use_cache=True)
        states.append(state)
    return np.stack(states, axis=0).astype(np.float32)  # float32に統一

def compute_reward(entry_action, next_close, entry_price):
    if entry_action == 0:  # Hold
        return 0.0
    
    # 価格差を計算（pips単位での考慮）
    price_diff = abs(next_close - entry_price) / entry_price
    direction_correct = False
    
    # より細かい報酬設計（High/Low偏り修正）
    if entry_action == 1:  # High予測
        direction_correct = next_close > entry_price
        price_change = (next_close - entry_price) / entry_price
    elif entry_action == 2:  # Low予測
        direction_correct = next_close < entry_price
        price_change = (entry_price - next_close) / entry_price
    else:
        return 0.0
    
    # 基本報酬計算（High/Low同等に調整）
    if direction_correct:
        # 成功時：段階的報酬（High/Low同じスケール）
        if price_diff > 0.002:  # 20pips以上の大幅な動き
            base_reward = 2.5  # 3.0→2.5に調整
        elif price_diff > 0.001:  # 10pips以上の中程度の動き
            base_reward = 1.8  # 2.0→1.8に調整
        elif price_diff > 0.0005:  # 5pips以上の小さな動き
            base_reward = 1.2  # 1.0→1.2に調整
        else:  # 5pips未満の微小な動き
            base_reward = 0.6  # 0.5→0.6に調整
        
        # 価格変動に比例したボーナス（High/Low同等）
        momentum_bonus = min(abs(price_change) * 3000, 1.5)  # 5000→3000、2.0→1.5に調整
        return base_reward + momentum_bonus
    else:
        # 失敗時：段階的ペナルティ（High/Low同等）
        if price_diff > 0.002:  # 大幅な逆行
            penalty = -2.0  # -2.5→-2.0に調整
        elif price_diff > 0.001:  # 中程度の逆行
            penalty = -1.3  # -1.5→-1.3に調整
        elif price_diff > 0.0005:  # 小さな逆行
            penalty = -1.0
        else:  # 微小な逆行
            penalty = -0.3
        
        # 逆行の大きさに応じた追加ペナルティ
        reverse_penalty = min(abs(price_change) * 3000, 1.5)
        return penalty - reverse_penalty

def train_dqn(ohlc_df, pair=pair, save_dir="./Models",
              gamma=0.995, lr=1e-4, batch_size=64,  # バッチサイズをさらに削減
              warmup=2000, updates=20000, target_sync=500,  # 学習数を大幅削減
              epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=15000,  # 探索期間短縮
              device='cuda' if torch.cuda.is_available() else 'cpu',
              num_workers=2, max_time_hours=1):  # 実行時間短縮
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Using device: {device}")
    
    # タイムアウト設定
    import time
    start_time = time.time()
    max_time_seconds = max_time_hours * 3600
    
    ohlc_df = ohlc_df[['open','high','low','close']].copy()
    
    # データサイズを制限（無限ループ防止）
    max_data_points = 10000  # 50000 → 10000 に大幅削減
    if len(ohlc_df) > max_data_points:
        print(f"[INFO] Data too large ({len(ohlc_df)} rows), using last {max_data_points} rows")
        ohlc_df = ohlc_df.iloc[-max_data_points:].copy()
    
    if not isinstance(ohlc_df.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex が必要です")

    # 入力次元を確定（ウィンドウサイズを削減）
    window_size = 20  # 30 → 20 に削減（計算量削減）
    probe = FeatureExtraction(ohlc_df.iloc[:window_size+10].copy(), use_cache=False)[-1:]
    in_dim = probe.shape[1] + 2  # +phase, range

    # スケーラ学習用サンプルを並列処理で高速化
    print("[INFO] Preparing scaler samples...")
    sample_indices = list(range(window_size, min(len(ohlc_df)-2, 2000)))
    
    def prepare_sample(i):
        sl = ohlc_df.iloc[i-window_size:i+1].copy()
        phase = (sl.index[-1].second % 60)/60.0
        sec_range = float(sl['high'].iloc[-1] - sl['low'].iloc[-1])
        return build_state_vec(sl, [phase, sec_range], use_cache=False)
    # 並列処理でサンプル準備
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        samples = list(executor.map(prepare_sample, sample_indices))
    
    # float32に統一してスケーラを学習
    samples_array = np.array(samples, dtype=np.float32)
    
    # 異常値チェック
    if np.any(np.isinf(samples_array)) or np.any(np.isnan(samples_array)):
        print("[WARNING] 異常値を検出、クリーニング中...")
        samples_array = np.nan_to_num(samples_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
    # 異常に大きな値をクリップ
    samples_array = np.clip(samples_array, -1e6, 1e6)
    
    print(f"[INFO] Sample array shape: {samples_array.shape}")
    print(f"[INFO] Sample array stats - Min: {samples_array.min():.4f}, Max: {samples_array.max():.4f}")
    
    scaler = StandardScaler().fit(samples_array)

    # GPU対応
    q = QNet(in_dim, ACTIONS).to(device)
    tgt = QNet(in_dim, ACTIONS).to(device)
    tgt.load_state_dict(q.state_dict())
    
    # モードを明確に設定
    q.train()  # 学習モード
    tgt.eval()  # ターゲットは常に評価モード
    
    # 最適化アルゴリズム（手動実装でPyTorchの問題を完全回避）
    print("[INFO] Initializing custom optimizer to avoid PyTorch issues...")
    
    # パラメータを手動で管理
    params = list(q.parameters())
    param_count = sum(p.numel() for p in params)
    print(f"[INFO] Model has {param_count:,} parameters")
    
    # 手動Adam実装用の状態
    adam_state = {
        'step': 0,
        'm': [torch.zeros_like(p) for p in params],  # 1次モーメント
        'v': [torch.zeros_like(p) for p in params],  # 2次モーメント
        'lr': lr,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'weight_decay': 1e-5
    }
    
    def manual_adam_step():
        adam_state['step'] += 1
        bias_correction1 = 1 - adam_state['beta1'] ** adam_state['step']
        bias_correction2 = 1 - adam_state['beta2'] ** adam_state['step']
        
        # コサインアニーリングによる学習率調整
        progress = adam_state['step'] / updates
        current_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        adam_state['lr'] = current_lr
        
        for i, param in enumerate(params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            if adam_state['weight_decay'] != 0:
                grad = grad.add(param.data, alpha=adam_state['weight_decay'])
            
            # 1次および2次モーメントの更新
            adam_state['m'][i].mul_(adam_state['beta1']).add_(grad, alpha=1 - adam_state['beta1'])
            adam_state['v'][i].mul_(adam_state['beta2']).addcmul_(grad, grad, value=1 - adam_state['beta2'])
            
            # バイアス補正
            m_hat = adam_state['m'][i] / bias_correction1
            v_hat = adam_state['v'][i] / bias_correction2
            
            # パラメータ更新
            param.data.addcdiv_(m_hat, v_hat.sqrt().add_(adam_state['eps']), value=-adam_state['lr'])
    
    print("[INFO] Custom Adam optimizer initialized successfully")
    
    mem = Replay()

    # 事前計算でボトルネック解消（簡素化版）
    print("[INFO] Pre-computing states for fast training...")
    
    # 全ての状態を事前計算（サンプル数を制限）
    pre_computed_states = {}
    pre_computed_extras = {}
    
    # 計算範囲を制限（最大1000サンプルまで）
    compute_range = min(1000, len(ohlc_df) - window_size - 1)
    compute_indices = list(range(window_size+1, window_size+1+compute_range))
    
    print(f"[INFO] Computing {len(compute_indices)} states...")
    
    for i in compute_indices:
        try:
            sl = ohlc_df.iloc[i-window_size:i+1].copy()
            phase = (sl.index[-1].second % 60)/60.0
            sec_range = float(sl['high'].iloc[-1] - sl['low'].iloc[-1])
            
            s = build_state_vec_fast(sl, phase, sec_range)
            s = scaler.transform([s])[0].astype(np.float32)
            
            pre_computed_states[i] = s
            pre_computed_extras[i] = (phase, sec_range)
            
        except Exception as e:
            print(f"[WARNING] Error computing state for index {i}: {e}")
            continue
    
    print(f"[INFO] Pre-computed {len(pre_computed_states)} states")
    
    # 事前計算した範囲を保存
    if pre_computed_states:
        min_idx = min(pre_computed_states.keys())
        max_idx = max(pre_computed_states.keys())
        print(f"[INFO] Pre-computed range: {min_idx} to {max_idx}")
    else:
        print("[ERROR] No states were pre-computed!")
        return

    steps, eps = 0, epsilon_start
    # 事前計算した範囲に合わせてidxsを制限
    if pre_computed_states:
        available_indices = list(pre_computed_states.keys())
        idxs = available_indices
        print(f"[INFO] Using {len(idxs)} pre-computed indices")
    else:
        idxs = []
        print("[ERROR] No pre-computed states available!")
        return
    episode = 0
    
    # 学習統計
    loss_history = deque(maxlen=1000)
    reward_history = deque(maxlen=1000)
    
    # エントリー統計
    entry_stats = {'Hold': 0, 'High': 0, 'Low': 0}
    reward_stats = {'Hold': [], 'High': [], 'Low': []}

    print("[INFO] Starting training...")
    
    # バッチ処理での高速化
    batch_process_size = 1000  # 1000ステップずつまとめて処理
    
    # 無限ループ防止のための追加チェック
    max_episodes = 200  # 1000 → 200 に削減
    episode_count = 0
    
    while steps < updates and episode_count < max_episodes:
        # タイムアウトチェック
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time_seconds:
            print(f"[WARNING] Training timed out after {elapsed_time/3600:.2f} hours")
            break
            
        random.shuffle(idxs)
        episode_count += 1
        
        # 進捗表示を追加（頻度を上げる）
        if episode_count % 5 == 0:  # 10 → 5 に変更
            print(f"[PROGRESS] Episode {episode_count}, Steps {steps}/{updates}, Elapsed: {elapsed_time/60:.1f}min")
        
        # バッチ処理で効率化
        for batch_start in range(0, len(idxs), batch_process_size):
            batch_end = min(batch_start + batch_process_size, len(idxs))
            batch_idxs = idxs[batch_start:batch_end]
            
            # バッチで状態とアクションを処理
            batch_states = []
            valid_batch_idxs = []
            
            # 事前計算済みの状態のみを使用
            for i in batch_idxs:
                if i in pre_computed_states:
                    batch_states.append(pre_computed_states[i])
                    valid_batch_idxs.append(i)
            
            if not batch_states:
                print(f"[WARNING] No valid states in batch, skipping...")
                continue
                
            batch_states = np.array(batch_states)
            batch_idxs = valid_batch_idxs
            
            # バッチ推論で高速化
            if random.random() < eps:
                batch_actions = np.random.randint(0, ACTIONS, len(batch_idxs))
            else:
                q.eval()
                with torch.no_grad():
                    states_tensor = torch.from_numpy(batch_states).float().to(device)
                    q_values = q(states_tensor)
                    batch_actions = torch.argmax(q_values, dim=1).cpu().numpy()
                q.train()
            
            # バッチで報酬計算
            experiences = []
            for idx, (i, a) in enumerate(zip(batch_idxs, batch_actions)):
                s = pre_computed_states[i]
                
                # 価格情報の高速取得
                entry_price = float(ohlc_df['close'].iloc[i])
                next_close = float(ohlc_df['close'].iloc[i+1])
                r = compute_reward(a, next_close, entry_price)
                reward_history.append(r)
                
                # 統計記録
                action_names = ['Hold', 'High', 'Low']
                if 0 <= a < len(action_names):
                    entry_stats[action_names[a]] += 1
                    reward_stats[action_names[a]].append(r)
                
                # 次の状態（事前計算済み、またはその場で計算）
                if i+1 in pre_computed_states:
                    ns = pre_computed_states[i+1]
                else:
                    # その場で計算（範囲チェック付き）
                    if i+1 < len(ohlc_df):
                        try:
                            sl_next = ohlc_df.iloc[max(0, i-window_size+1):i+2].copy()
                            if len(sl_next) >= 2:  # 最低限のデータがあるかチェック
                                phase_n = (sl_next.index[-1].second % 60)/60.0
                                sec_range_n = float(sl_next['high'].iloc[-1] - sl_next['low'].iloc[-1])
                                ns = build_state_vec(sl_next, [phase_n, sec_range_n])
                                ns = scaler.transform([ns])[0].astype(np.float32)
                            else:
                                ns = s  # データが不足している場合は現在の状態を使用
                        except Exception as e:
                            print(f"[WARNING] Error computing next state for {i}: {e}")
                            ns = s  # エラーの場合は現在の状態を使用
                    else:
                        ns = s  # 範囲外の場合は現在の状態を使用
                
                experiences.append((s, a, r, ns, 0.0))
            
            # バッチでメモリに追加
            for exp in experiences:
                mem.push(*exp)
                steps += 1
                
                # 無限ループ防止: ステップ数の上限チェック
                if steps >= updates:
                    print(f"[INFO] Reached maximum steps: {steps}")
                    break
                
                # 学習実行（頻度を調整）
                if len(mem) >= warmup and steps % 4 == 0:  # 4ステップごとに学習（より頻繁に学習）
                    # 高速学習ステップ
                    q.train()
                    sample_result = mem.sample(batch_size)
                    if len(sample_result) == 7:  # prioritized sampling
                        S, A, R, NS, DN, indices, weights = sample_result
                        weights = torch.from_numpy(weights).float().to(device)
                    else:  # normal sampling
                        S, A, R, NS, DN, indices, weights = sample_result + (np.ones(batch_size),)
                        weights = torch.ones(batch_size).to(device)
                    
                    # GPU転送
                    S = torch.from_numpy(S).float().to(device)
                    A = torch.from_numpy(A).long().to(device)
                    R = torch.from_numpy(R).float().to(device)
                    NS = torch.from_numpy(NS).float().to(device)
                    
                    # Double DQN
                    q_sa = q(S).gather(1, A.view(-1,1)).squeeze(1)
                    with torch.no_grad():
                        tgt.eval()
                        next_actions = q(NS).max(1)[1]
                        next_q_values = tgt(NS).gather(1, next_actions.view(-1,1)).squeeze(1)
                        tgt_q = R + gamma * next_q_values
                    
                    # 損失計算と更新
                    td_errors = (q_sa - tgt_q).detach().cpu().numpy()
                    loss = F.smooth_l1_loss(q_sa, tgt_q, reduction='none')
                    loss = (loss * weights).mean()
                    loss_history.append(loss.item())
                    
                    mem.update_priorities(indices, td_errors)
                    
                    # 手動optimizer実行
                    for param in q.parameters():
                        if param.grad is not None:
                            param.grad.zero_()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
                    manual_adam_step()
                    
                    if steps % target_sync == 0:
                        tgt.load_state_dict(q.state_dict())
                    
                    eps = epsilon_end + (epsilon_start-epsilon_end) * math.exp(-(steps-warmup)/epsilon_decay)
                
                if steps >= updates:
                    break
            
            if steps >= updates:
                break
        episode += 1
        if episode % 10 == 0:  # 表示頻度を下げて高速化（5→10）
            avg_loss = np.mean(loss_history) if loss_history else 0.0
            avg_reward = np.mean(reward_history) if reward_history else 0.0
            
            # エントリー統計の計算
            total_entries = sum(entry_stats.values())
            hold_pct = (entry_stats['Hold'] / total_entries * 100) if total_entries > 0 else 0
            high_pct = (entry_stats['High'] / total_entries * 100) if total_entries > 0 else 0
            low_pct = (entry_stats['Low'] / total_entries * 100) if total_entries > 0 else 0
            
            # 各アクションの平均報酬
            avg_reward_hold = np.mean(reward_stats['Hold']) if reward_stats['Hold'] else 0.0
            avg_reward_high = np.mean(reward_stats['High']) if reward_stats['High'] else 0.0
            avg_reward_low = np.mean(reward_stats['Low']) if reward_stats['Low'] else 0.0
            
            torch.save(q.state_dict(), os.path.join(save_dir, f"dqn_policy_{pair}.pt"))
            with open(os.path.join(save_dir, f"dqn_scaler_{pair}.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            print(f"[CKPT] Episode={episode}, Steps={steps}, Eps={eps:.3f}, "
                  f"AvgLoss={avg_loss:.4f}, AvgReward={avg_reward:.4f}")
            print(f"[STATS] エントリー回数 - Hold:{entry_stats['Hold']}({hold_pct:.1f}%), "
                  f"High:{entry_stats['High']}({high_pct:.1f}%), Low:{entry_stats['Low']}({low_pct:.1f}%)")
            print(f"[REWARDS] 平均報酬 - Hold:{avg_reward_hold:.3f}, "
                  f"High:{avg_reward_high:.3f}, Low:{avg_reward_low:.3f}")
            
            # メモリ使用量をチェック（デバッグ用）
            if device.startswith('cuda'):
                print(f"[INFO] GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

    # 最終保存
    torch.save(q.state_dict(), os.path.join(save_dir, f"dqn_policy_{pair}.pt"))
    with open(os.path.join(save_dir, f"dqn_scaler_{pair}.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # 最終統計表示
    total_entries = sum(entry_stats.values())
    if total_entries > 0:
        print("\n" + "="*60)
        print("FINAL TRAINING STATISTICS")
        print("="*60)
        print(f"Total Actions: {total_entries}")
        print(f"Hold: {entry_stats['Hold']} ({entry_stats['Hold']/total_entries*100:.1f}%)")
        print(f"High: {entry_stats['High']} ({entry_stats['High']/total_entries*100:.1f}%)")
        print(f"Low:  {entry_stats['Low']} ({entry_stats['Low']/total_entries*100:.1f}%)")
        
        if reward_stats['High']:
            print(f"Average Reward High: {np.mean(reward_stats['High']):.3f}")
        if reward_stats['Low']:
            print(f"Average Reward Low:  {np.mean(reward_stats['Low']):.3f}")
        if reward_stats['Hold']:
            print(f"Average Reward Hold: {np.mean(reward_stats['Hold']):.3f}")
        print("="*60)
    
    # キャッシュクリア
    global _feature_cache
    _feature_cache.clear()
    
    print("[DONE] DQN training completed and saved.")
    evaluate_dqn_model(q, scaler, ohlc_df, device=device, window_size=window_size)
def evaluate_dqn_model(q, scaler, ohlc_df, n_eval=2000, device='cpu', window_size=20):
    """
    学習済みDQNモデルを使ってOHLCデータ上で勝率、損益、最大ドローダウンを測定
    - q: 学習済み QNet
    - scaler: 学習時の StandardScaler
    - ohlc_df: DataFrame (DatetimeIndex 必須, open/high/low/close)
    - n_eval: 評価に使うサンプル数
    - device: 計算デバイス
    """
    # 全体の統計
    correct, total = 0, 0
    
    # 予測別の統計
    high_correct, high_total = 0, 0  # High予測の統計
    low_correct, low_total = 0, 0    # Low予測の統計
    hold_count = 0                   # Hold回数
    
    profit = 0.0
    max_dd, peak = 0.0, 0.0

    idxs = list(range(window_size+1, min(len(ohlc_df)-1, window_size+1+min(1000, n_eval))))  # 評価範囲も制限
    q.eval()
    
    print(f"[INFO] Evaluating model on {len(idxs)} samples...")

    # バッチ評価で高速化
    batch_size = 256
    for batch_start in range(0, len(idxs), batch_size):
        batch_end = min(batch_start + batch_size, len(idxs))
        batch_idxs = idxs[batch_start:batch_end]
        
        # バッチで状態を準備
        states = []
        valid_batch_idxs = []
        
        for i in batch_idxs:
            try:
                if i >= window_size and i < len(ohlc_df):
                    sl = ohlc_df.iloc[max(0, i-window_size):i+1].copy()
                    if len(sl) >= 2:  # 最低限のデータがあるかチェック
                        phase = (sl.index[-1].second % 60)/60.0
                        sec_range = float(sl['high'].iloc[-1] - sl['low'].iloc[-1])
                        s = build_state_vec(sl, [phase, sec_range])
                        s = scaler.transform([s])[0].astype(np.float32)  # float32に変換
                        states.append(s)
                        valid_batch_idxs.append(i)
            except Exception as e:
                print(f"[WARNING] Error preparing state for evaluation index {i}: {e}")
                continue
        
        if not states:
            print(f"[WARNING] No valid states in evaluation batch, skipping...")
            continue
            
        batch_idxs = valid_batch_idxs
        
        # バッチ推論
        states_tensor = torch.from_numpy(np.array(states, dtype=np.float32)).float().to(device)  # 明示的にfloat32
        with torch.no_grad():
            q_values = q(states_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        
        # 報酬計算と勝率判定
        for idx, (i, a) in enumerate(zip(batch_idxs, actions)):
            sl = ohlc_df.iloc[i-20:i+1].copy()
            entry_price = float(sl['close'].iloc[-1])
            next_close = float(ohlc_df['close'].iloc[i+1])
            r = compute_reward(a, next_close, entry_price)

            # アクション別統計
            if a == 0:  # Hold
                hold_count += 1
            elif a == 1:  # High予測
                high_total += 1
                total += 1
                if next_close > entry_price:  # 上昇したか
                    high_correct += 1
                    correct += 1
            elif a == 2:  # Low予測
                low_total += 1
                total += 1
                if next_close < entry_price:  # 下降したか
                    low_correct += 1
                    correct += 1

            # 累積損益計算
            profit += r
        peak = max(peak, profit)
        dd = peak - profit
        max_dd = max(max_dd, dd)

    # 全体の勝率
    acc = correct / total if total > 0 else 0.0
    
    # 予測別の勝率
    high_acc = high_correct / high_total if high_total > 0 else 0.0
    low_acc = low_correct / low_total if low_total > 0 else 0.0
    
    # 結果表示
    print(f"[EVAL] Overall Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"[EVAL] High Prediction Accuracy: {high_acc:.3f} ({high_correct}/{high_total})")
    print(f"[EVAL] Low Prediction Accuracy: {low_acc:.3f} ({low_correct}/{low_total})")
    print(f"[EVAL] Hold Count: {hold_count}")
    print(f"[EVAL] Action Distribution - High: {high_total}, Low: {low_total}, Hold: {hold_count}")
    print(f"[EVAL] Total Profit: {profit:.2f}, Max Drawdown: {max_dd:.2f}")
    
    return acc, profit, max_dd

if __name__ == "__main__":
    # 既存のOHLC DataFrameを読み込み
    import sys
    
    # データファイルのパスを設定
    data_file = f"data/{pair}_M1.csv"
    
    try:
        print(f"[INFO] Loading data from {data_file}...")
        
        # CSVにヘッダーがないので、カラム名を手動で指定
        column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        df = pd.read_csv(data_file, names=column_names)
        
        # 日付と時刻を結合してDatetimeIndexを作成
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
        df = df.set_index('datetime')
        
        # 不要な列を削除
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"[INFO] Data loaded: {len(df)} rows")
        print(f"[INFO] Data range: {df.index[0]} to {df.index[-1]}")
        print(f"[INFO] Columns: {list(df.columns)}")
        print(f"[INFO] Sample data:")
        print(df.head())
        
        # 学習開始
        train_dqn(df, pair=pair)
        print("[INFO] モデル保存完了")
        
    except FileNotFoundError:
        print(f"[ERROR] データファイルが見つかりません: {data_file}")
        print("[INFO] 利用可能なデータファイル:")
        
        # 利用可能なファイルを表示
        import glob
        available_files = glob.glob("data/*_M1.csv")
        for file in available_files:
            print(f"  - {file}")
        
        if available_files:
            print(f"\n[INFO] 代替ファイルを使用しますか？最初のファイルを使用: {available_files[0]}")
            
            column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
            df = pd.read_csv(available_files[0], names=column_names)
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
            df = df.set_index('datetime')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            pair_from_file = available_files[0].split('\\')[-1].split('_')[0]  # Windowsパス区切り対応
            print(f"[INFO] ペア名を {pair_from_file} に変更")
            train_dqn(df, pair=pair_from_file)
            print("[INFO] モデル保存完了")
        else:
            print("[ERROR] M1データファイルが見つかりません")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] データ読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
