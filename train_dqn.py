# train_dqn.py
import os, pickle, random, math
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

pair = "USDJPY"
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
def _CalcOtherR(df,periods):
    for period in periods:
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
        
    # ストキャスティクス
    slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'])
    df['slowk'+str(period)] = slowk
    df['slowk_diff'+str(period)] = df['slowk'+str(period)].diff()
    df['slowd'+str(period)] = slowd
    df['slowd_diff'+str(period)] = df['slowd'+str(period)].diff()
    
    # MACD
    macd, macdsignal, macdhist = ta.MACD(df['close'])
    df['macd'] = macd
    df['macd_signal'] = macdsignal
    df['macd_hist'] = macdhist
    
    # Williams %R
    df['williams_r'] = ta.WILLR(df['high'], df['low'], df['close'])
    
    # CCI (Commodity Channel Index)
    df['cci'] = ta.CCI(df['high'], df['low'], df['close'])
    
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
    periods_RSI = [3, 5, 10, 14]  # 期間を追加
    periods_SMA = [3, 5, 10, 20]  # 期間を追加

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
    for period in [5, 10, 20]:
        if len(df) >= period:
            sma = df["close"].rolling(period).mean()
            df[f"sma_distance_{period}"] = np.where(sma != 0, (df["close"] - sma) / sma, 0.0)
    
    # 前の足との比較 - 異常値クリップ
    df["prev_close_ratio"] = df["close"].pct_change().clip(-1, 1)
    df["prev_volume_ratio"] = df["volume"].pct_change().clip(-10, 10) if "volume" in df.columns else 0
    
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
        # より深いネットワークで表現力向上
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),  # ユニット数増加
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(64, out_dim)
        )
    
    def forward(self, x): 
        return self.net(x)

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
    
    # より細かい報酬設計
    if entry_action == 1:  # High予測
        if next_close > entry_price:
            # 成功時：価格差に応じた報酬（より大きな報酬）
            base_reward = min(price_diff * 10000, 5.0)  # 10000倍でpips相当、最大5.0
            # ボーナス：大きな値動きには追加報酬
            if price_diff > 0.001:  # 10pips以上
                base_reward += 1.0
            return base_reward
        else:
            # 失敗時：より厳しいペナルティ
            penalty = min(price_diff * 10000, 3.0)  # 最大-3.0のペナルティ
            # 逆方向に大きく動いた場合の追加ペナルティ
            if price_diff > 0.001:
                penalty += 1.0
            return -penalty
    
    if entry_action == 2:  # Low予測
        if next_close < entry_price:
            # 成功時：価格差に応じた報酬
            base_reward = min(price_diff * 10000, 5.0)
            if price_diff > 0.001:
                base_reward += 1.0
            return base_reward
        else:
            # 失敗時：厳しいペナルティ
            penalty = min(price_diff * 10000, 3.0)
            if price_diff > 0.001:
                penalty += 1.0
            return -penalty
    
    return 0.0

def train_dqn(ohlc_df, pair=pair, save_dir="./Models",
              gamma=0.99, lr=5e-4, batch_size=256,  # 学習率を下げ、バッチサイズを調整
              warmup=10000, updates=200000, target_sync=1000,  # より多くの学習
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=100000,  # より長い探索
              device='cuda' if torch.cuda.is_available() else 'cpu',
              num_workers=4):
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Using device: {device}")
    
    ohlc_df = ohlc_df[['open','high','low','close']].copy()
    if not isinstance(ohlc_df.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex が必要です")

    # 入力次元を確定
    probe = FeatureExtraction(ohlc_df.iloc[:20].copy(), use_cache=False)[-1:]
    in_dim = probe.shape[1] + 2  # +phase, range

    # スケーラ学習用サンプルを並列処理で高速化
    print("[INFO] Preparing scaler samples...")
    sample_indices = list(range(20, min(len(ohlc_df)-2, 2000)))
    
    def prepare_sample(i):
        sl = ohlc_df.iloc[i-20:i+1].copy()
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
    
    # 最適化アルゴリズムを改善
    opt = optim.AdamW(q.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates)
    
    mem = Replay()

    steps, eps = 0, epsilon_start
    idxs = list(range(21, len(ohlc_df)-1))
    episode = 0
    
    # 学習統計
    loss_history = deque(maxlen=1000)
    reward_history = deque(maxlen=1000)

    print("[INFO] Starting training...")
    
    # 事前計算でボトルネック解消
    print("[INFO] Pre-computing states for fast training...")
    
    # 全ての状態を事前計算
    pre_computed_states = {}
    pre_computed_extras = {}
    
    for i in range(21, len(ohlc_df)-1):
        sl = ohlc_df.iloc[i-20:i+1].copy()
        phase = (sl.index[-1].second % 60)/60.0
        sec_range = float(sl['high'].iloc[-1] - sl['low'].iloc[-1])
        
        s = build_state_vec_fast(sl, phase, sec_range)
        s = scaler.transform([s])[0].astype(np.float32)
        
        pre_computed_states[i] = s
        pre_computed_extras[i] = (phase, sec_range)
    
    print(f"[INFO] Pre-computed {len(pre_computed_states)} states")
    
    # バッチ処理での高速化
    batch_process_size = 1000  # 1000ステップずつまとめて処理
    
    while steps < updates:
        random.shuffle(idxs)
        
        # バッチ処理で効率化
        for batch_start in range(0, len(idxs), batch_process_size):
            batch_end = min(batch_start + batch_process_size, len(idxs))
            batch_idxs = idxs[batch_start:batch_end]
            
            # バッチで状態とアクションを処理
            batch_states = np.array([pre_computed_states[i] for i in batch_idxs])
            
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
                
                # 次の状態（事前計算済み、またはその場で計算）
                if i+1 in pre_computed_states:
                    ns = pre_computed_states[i+1]
                else:
                    sl_next = ohlc_df.iloc[i-19:i+2].copy()
                    phase_n = (sl_next.index[-1].second % 60)/60.0
                    sec_range_n = float(sl_next['high'].iloc[-1] - sl_next['low'].iloc[-1])
                    ns = build_state_vec(sl_next, [phase_n, sec_range_n])
                    ns = scaler.transform([ns])[0].astype(np.float32)
                
                experiences.append((s, a, r, ns, 0.0))
            
            # バッチでメモリに追加
            for exp in experiences:
                mem.push(*exp)
                steps += 1
                
                # 学習実行（頻度を調整）
                if len(mem) >= warmup and steps % 8 == 0:  # 8ステップごとに学習（頻度を下げて高速化）
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
                    
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
                    opt.step()
                    scheduler.step()
                    
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
            
            torch.save(q.state_dict(), os.path.join(save_dir, f"dqn_policy_{pair}.pt"))
            with open(os.path.join(save_dir, f"dqn_scaler_{pair}.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            print(f"[CKPT] Episode={episode}, Steps={steps}, Eps={eps:.3f}, "
                  f"AvgLoss={avg_loss:.4f}, AvgReward={avg_reward:.4f}")
            
            # メモリ使用量をチェック（デバッグ用）
            if device.startswith('cuda'):
                print(f"[INFO] GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

    # 最終保存
    torch.save(q.state_dict(), os.path.join(save_dir, f"dqn_policy_{pair}.pt"))
    with open(os.path.join(save_dir, f"dqn_scaler_{pair}.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # キャッシュクリア
    global _feature_cache
    _feature_cache.clear()
    
    print("[DONE] DQN training completed and saved.")
    evaluate_dqn_model(q, scaler, ohlc_df, device=device)
def evaluate_dqn_model(q, scaler, ohlc_df, n_eval=2000, device='cpu'):
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

    idxs = list(range(21, min(len(ohlc_df)-1, n_eval)))
    q.eval()
    
    print(f"[INFO] Evaluating model on {len(idxs)} samples...")

    # バッチ評価で高速化
    batch_size = 256
    for batch_start in range(0, len(idxs), batch_size):
        batch_end = min(batch_start + batch_size, len(idxs))
        batch_idxs = idxs[batch_start:batch_end]
        
        # バッチで状態を準備
        states = []
        for i in batch_idxs:
            sl = ohlc_df.iloc[i-20:i+1].copy()
            phase = (sl.index[-1].second % 60)/60.0
            sec_range = float(sl['high'].iloc[-1] - sl['low'].iloc[-1])
            s = build_state_vec(sl, [phase, sec_range])
            s = scaler.transform([s])[0].astype(np.float32)  # float32に変換
            states.append(s)
        
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
