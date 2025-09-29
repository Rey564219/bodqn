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

pair = "EURUSD"
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
        # ボリンジャーバンド幅
        df['bb_width'+str(period)] = ta.BBANDS(df['close'], period)[1] - ta.BBANDS(df['close'], period)[0]
        df['bb_width_diff' + str(period)] = df['bb_width'+str(period)].diff()
        # ATR（平均的な値幅）
        df['atr'+str(period)] = ta.ATR(df['high'], df['low'], df['close'], period)
        df['atr_diff' + str(period)] = df['atr'+str(period)].diff()
    # ストキャスティクス
    slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'])
    df['slowk'+str(period)] = slowk
    df['slowk_diff'+str(period)] = df['slowk'+str(period)].diff()
    df['slowd'+str(period)] = slowd
    df['slowd_diff'+str(period)] = df['slowd'+str(period)].diff()
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
    periods_RSI = [3, 5, 10]
    periods_SMA = [3, 5, 10]

    df = _CalcSMAR(df, periods_SMA)
    df = _CalcRSIR(df, periods_RSI)
    df = _CalcOtherR(df, periods_RSI)

    df["open_r"] = df["open"]/df["close"]
    df["high_r"] = df["high"]/df["close"]
    df["low_r"] = df["low"]/df["close"]
    result = df.drop(columns = ["open", "close", "high", "low"])
    
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
        # ネットワークをより効率的に設計
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),  # バッチ正規化で安定性向上
            nn.ReLU(inplace=True),  # インプレース操作でメモリ節約
            nn.Dropout(0.1),  # 過学習防止
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )
    
    def forward(self, x): 
        return self.net(x)

class Replay:
    def __init__(self, cap=100_000): 
        self.buf = deque(maxlen=cap)
        self._cached_arrays = None
        self._cache_valid = False
    
    def push(self, *exp): 
        self.buf.append(exp)
        self._cache_valid = False  # キャッシュを無効化
    
    def sample(self, bs):
        # 高速サンプリング
        indices = np.random.choice(len(self.buf), bs, replace=False)
        batch = [self.buf[i] for i in indices]
        
        # numpy配列への変換を最適化
        s, a, r, ns, done = zip(*batch)
        return (np.stack(s, axis=0).astype(np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.stack(ns, axis=0).astype(np.float32),
                np.array(done, dtype=np.float32))
    
    def __len__(self): return len(self.buf)

def build_state_vec(ohlc_win_df, extra=None, use_cache=True):
    feat = FeatureExtraction(ohlc_win_df, use_cache=use_cache)[-1:]
    x = feat.values.astype(np.float32).reshape(-1)
    if extra is not None:
        x = np.concatenate([x, np.asarray(extra, dtype=np.float32)])
    return x

# バッチ処理用の関数を追加
def build_state_batch(ohlc_data_list, extra_list=None):
    """複数の状態ベクトルを一度に処理"""
    states = []
    for i, ohlc_win_df in enumerate(ohlc_data_list):
        extra = extra_list[i] if extra_list else None
        state = build_state_vec(ohlc_win_df, extra, use_cache=True)
        states.append(state)
    return np.stack(states, axis=0)

def compute_reward(entry_action, next_close, entry_price):
    if entry_action == 0:  # Hold
        return 0.0
    
    # 価格差を計算
    price_diff = abs(next_close - entry_price) / entry_price
    
    if entry_action == 1:  # High予測
        if next_close > entry_price:
            # 狙った通りに動いた場合、価格差に応じた報酬
            return min(price_diff * 100, 2.0)  # 最大2.0の報酬
        else:
            # 狙った通りに動かなかった場合、価格差に応じたペナルティ
            return -min(price_diff * 100, 2.5)  # 最大-2.5のペナルティ
    
    if entry_action == 2:  # Low予測
        if next_close < entry_price:
            # 狙った通りに動いた場合、価格差に応じた報酬
            return min(price_diff * 100, 2.0)  # 最大2.0の報酬
        else:
            # 狙った通りに動かなかった場合、価格差に応じたペナルティ
            return -min(price_diff * 100, 2.5)  # 最大-2.5のペナルティ
    
    return 0.0

def train_dqn(ohlc_df, pair=pair, save_dir="./Models",
              gamma=0.995, lr=1e-3, batch_size=512,
              warmup=5000, updates=150000, target_sync=2000,
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=80000,
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
    
    scaler = StandardScaler().fit(np.array(samples))

    # GPU対応
    q = QNet(in_dim, ACTIONS).to(device)
    tgt = QNet(in_dim, ACTIONS).to(device)
    tgt.load_state_dict(q.state_dict())
    
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
    while steps < updates:
        random.shuffle(idxs)
        
        # バッチ単位でエピソードを処理して効率化
        batch_states, batch_actions, batch_rewards = [], [], []
        batch_next_states = []
        
        for i in idxs:
            sl = ohlc_df.iloc[i-20:i+1].copy()
            phase = (sl.index[-1].second % 60)/60.0
            sec_range = float(sl['high'].iloc[-1] - sl['low'].iloc[-1])
            s = build_state_vec(sl, [phase, sec_range])
            s = scaler.transform([s])[0]

            if random.random() < eps:
                a = random.randrange(ACTIONS)
            else:
                with torch.no_grad():
                    s_tensor = torch.from_numpy(s).unsqueeze(0).to(device)
                    qv = q(s_tensor)
                    a = int(torch.argmax(qv, dim=1).item())

            entry_price = float(sl['close'].iloc[-1])
            next_close = float(ohlc_df['close'].iloc[i+1])
            r = compute_reward(a, next_close, entry_price)
            reward_history.append(r)

            sl_next = ohlc_df.iloc[i-19:i+2].copy()
            phase_n = (sl_next.index[-1].second % 60)/60.0
            sec_range_n = float(sl_next['high'].iloc[-1] - sl_next['low'].iloc[-1])
            ns = build_state_vec(sl_next, [phase_n, sec_range_n])
            ns = scaler.transform([ns])[0]

            mem.push(s, a, r, ns, 0.0)
            steps += 1

            # より頻繁に学習を実行
            if len(mem) >= warmup and steps % 4 == 0:  # 4ステップごとに学習
                S, A, R, NS, DN = mem.sample(batch_size)
                
                # GPU転送
                S = torch.from_numpy(S).to(device)
                A = torch.from_numpy(A).to(device)
                R = torch.from_numpy(R).to(device)
                NS = torch.from_numpy(NS).to(device)
                
                # Double DQN実装
                q_sa = q(S).gather(1, A.view(-1,1)).squeeze(1)
                with torch.no_grad():
                    # Double DQNでオーバーエスティメーションを防ぐ
                    next_actions = q(NS).max(1)[1]
                    next_q_values = tgt(NS).gather(1, next_actions.view(-1,1)).squeeze(1)
                    tgt_q = R + gamma * next_q_values
                
                loss = F.smooth_l1_loss(q_sa, tgt_q)
                loss_history.append(loss.item())
                
                opt.zero_grad()
                loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
                opt.step()
                scheduler.step()

                if steps % target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

                eps = epsilon_end + (epsilon_start-epsilon_end) * math.exp(-(steps-warmup)/epsilon_decay)

            if steps >= updates: break
        episode += 1
        if episode % 5 == 0:
            avg_loss = np.mean(loss_history) if loss_history else 0.0
            avg_reward = np.mean(reward_history) if reward_history else 0.0
            
            torch.save(q.state_dict(), os.path.join(save_dir, f"dqn_policy_{pair}.pt"))
            with open(os.path.join(save_dir, f"dqn_scaler_{pair}.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            print(f"[CKPT] Episode={episode}, Steps={steps}, Eps={eps:.3f}, "
                  f"AvgLoss={avg_loss:.4f}, AvgReward={avg_reward:.4f}")

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
    correct, total = 0, 0
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
            s = scaler.transform([s])[0]
            states.append(s)
        
        # バッチ推論
        states_tensor = torch.from_numpy(np.array(states)).to(device)
        with torch.no_grad():
            q_values = q(states_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        
        # 報酬計算
        for idx, (i, a) in enumerate(zip(batch_idxs, actions)):
            sl = ohlc_df.iloc[i-20:i+1].copy()
            entry_price = float(sl['close'].iloc[-1])
            next_close = float(ohlc_df['close'].iloc[i+1])
            r = compute_reward(a, next_close, entry_price)

            # Holdはスキップ、High/Lowの的中率を評価
            if a in (1, 2):
                total += 1
                if r > 0:
                    correct += 1

            # 累積損益計算
            profit += r
        peak = max(peak, profit)
        dd = peak - profit
        max_dd = max(max_dd, dd)

    acc = correct / total if total > 0 else 0.0
    print(f"[EVAL] Accuracy={acc:.3f} ({correct}/{total})")
    print(f"[EVAL] Total Profit={profit:.2f}, Max Drawdown={max_dd:.2f}")
    
    return acc, profit, max_dd

if __name__ == "__main__":
    # 例: 既存のOHLC DataFrameを読み込み
    df = pd.read_csv(f"{pair}_1min.csv", parse_dates=["close_time"])
    df = df.set_index("close_time")
    train_dqn(df, pair=pair)
    print("[INFO] モデル保存完了")
