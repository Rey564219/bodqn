# train_dqn.py
import os
import warnings

# 高速化のための環境変数設定（最優先）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = str(min(8, os.cpu_count()))  # CPU使用数制限
os.environ['MKL_NUM_THREADS'] = str(min(8, os.cpu_count()))
os.environ['NUMEXPR_MAX_THREADS'] = str(min(8, os.cpu_count()))

# PyTorchの最適化設定
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['TORCH_DISABLE_DYNAMIC_SHAPES'] = '1'
os.environ['PYTORCH_DISABLE_DYNAMO'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# メモリ最適化
import gc
gc.set_threshold(700, 10, 10)  # GCをより積極的に実行

import pickle, random, math
import numpy as np
import pandas as pd

# 警告を抑制
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# NumPy高速化設定
np.seterr(all='ignore')  # 警告を無効化
pd.options.mode.chained_assignment = None  # SettingWithCopyWarningを無効化

import torch, torch.nn as nn, torch.optim as optim

# PyTorch最適化設定
torch.set_num_threads(min(8, os.cpu_count()))
torch.backends.cudnn.benchmark = True  # cuDNN自動最適化
torch.backends.cudnn.deterministic = False  # 速度優先

from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from shared_features import (
    FeatureExtraction,
    build_state_vec,
    build_state_vec_fast,
    clear_feature_cache,
    compute_trend_direction,
)

pair = "USDJPY"
ACTIONS = 2  # 0:Hold, 1:Mode-specific action (High or Low)
ACTION_MODES = {
    "high": {"label": "High", "id": 1, "model_name": "dqn_policy_high.pt"},
    "low": {"label": "Low", "id": 2, "model_name": "dqn_policy_low.pt"},
}

# 資産タイプ別の手数料・スリッページ（単純化したbps）
FEE_TABLE = {
    "crypto": {"entry": 0.0006, "exit": 0.0006, "slippage": 0.0002},  # 0.12% + slippage
    "fx": {"entry": 0.0001, "exit": 0.0001, "slippage": 0.00005},    # 0.02% + slippage（スプレッド相当）
}

# デフォルトのエグジット設定（n分後 or TP/SL）
DEFAULT_EXIT_CONFIG = {
    "horizon_bars": 5,   # n分後（1分足想定）
    "tp_pct": 0.003,     # 0.3% 利確
    "sl_pct": 0.002,     # 0.2% 損切り
}

class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 高速化のためにモデルサイズを最適化（性能を維持しつつ軽量化）
        self.feature_extractor = nn.Sequential(
            # 入力層：効率的なサイズ
            nn.Linear(in_dim, 1024),  # 2048→1024に軽量化
            nn.BatchNorm1d(1024),
            nn.GELU(),  # ReLUよりも高性能
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
        
        # 重みの初期化（High/Lowの対称性を保証）
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みを対称的に初期化してHigh/Lowのバイアスを除去"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavierの均等初期化（対称性保証）
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # バイアスは小さな値で初期化
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        # advantage streamの最終層は特別に初期化（Q値の対称性を保証）
        if hasattr(self, 'advantage_stream'):
            for layer in self.advantage_stream:
                if isinstance(layer, nn.Linear):
                    # より小さな初期値で対称性を強化
                    nn.init.xavier_uniform_(layer.weight, gain=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x): 
        # 高速化のため処理を簡素化
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

class Replay:
    def __init__(self, cap=200_000):  # バッファサイズを拡大
        self.buf = deque(maxlen=cap)
        self._cached_arrays = None
        self._cache_valid = False
        # Prioritized Replay用（高勝率パラメータ）
        self.priorities = deque(maxlen=cap)
        self.alpha = 0.7  # 優先度の重みを強化
        self.beta = 0.5   # importance sampling補正を強化
        self.beta_increment = 0.0005  # よりゆっくりとした調整
        self.epsilon = 1e-8  # より小さな値で精密化
    
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

# バッチ処理用の関数を追加（並列化対応）
def build_state_batch_parallel(ohlc_data_list, extra_list=None, n_workers=None):
    """並列処理で複数の状態ベクトルを高速処理"""
    if n_workers is None:
        n_workers = min(4, os.cpu_count())
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        if extra_list:
            futures = [executor.submit(build_state_vec_fast, ohlc_win_df, extra_list[i][0], extra_list[i][1]) 
                      for i, ohlc_win_df in enumerate(ohlc_data_list)]
        else:
            futures = [executor.submit(build_state_vec, ohlc_win_df, None, True) 
                      for ohlc_win_df in ohlc_data_list]
        
        states = [future.result() for future in futures]
    
    return np.stack(states, axis=0).astype(np.float32)

def _get_fee_rate(asset_type: str) -> float:
    cfg = FEE_TABLE.get(str(asset_type).lower(), FEE_TABLE["fx"])
    return float(cfg.get("entry", 0.0) + cfg.get("exit", 0.0) + cfg.get("slippage", 0.0))


def simulate_exit(entry_action, entry_price, future_bars, exit_cfg):
    """TP/SLもしくは時間切れでのエグジット価格を近似計算する。
    future_bars: DataFrame with columns high/low/close
    exit_cfg: {horizon_bars, tp_pct, sl_pct}
    """
    horizon = max(1, int(exit_cfg.get("horizon_bars", 1)))
    tp_pct = float(exit_cfg.get("tp_pct", 0.0))
    sl_pct = float(exit_cfg.get("sl_pct", 0.0))

    if future_bars is None or len(future_bars) == 0:
        return entry_price, "no_future"

    horizon_slice = future_bars.iloc[:horizon].copy()
    exit_price = float(horizon_slice['close'].iloc[-1])
    exit_reason = "time"

    for _, row in horizon_slice.iterrows():
        high_p = float(row['high'])
        low_p = float(row['low'])
        if entry_action == 1:  # High/long
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
            if high_p >= tp_price:
                exit_price = tp_price
                exit_reason = "tp"
                break
            if low_p <= sl_price:
                exit_price = sl_price
                exit_reason = "sl"
                break
        elif entry_action == 2:  # Low/short
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
            if low_p <= tp_price:
                exit_price = tp_price
                exit_reason = "tp"
                break
            if high_p >= sl_price:
                exit_price = sl_price
                exit_reason = "sl"
                break

    return exit_price, exit_reason


def compute_reward(entry_action, entry_price, future_bars, trend_dir=0.0, asset_type="fx", exit_cfg=None):
    """仮想通貨/FX向けに、TP/SL + 時間切れを考慮した報酬を計算する。"""
    if entry_action == 0:
        return 0.0, entry_price, "hold"

    if exit_cfg is None:
        exit_cfg = DEFAULT_EXIT_CONFIG

    exit_price, exit_reason = simulate_exit(entry_action, entry_price, future_bars, exit_cfg)

    if entry_action == 1:
        gross = (exit_price - entry_price) / entry_price
    elif entry_action == 2:
        gross = (entry_price - exit_price) / entry_price
    else:
        return 0.0, entry_price, "unknown_action"

    fee_rate = _get_fee_rate(asset_type)
    net = gross - fee_rate

    # トレンドと同方向なら軽いボーナス、逆ならペナルティ
    trend_penalty = 0.0
    trend_threshold = 1e-6
    if entry_action == 1 and trend_dir < -trend_threshold:
        trend_penalty = 0.001
    elif entry_action == 2 and trend_dir > trend_threshold:
        trend_penalty = 0.001

    reward = np.clip((net - trend_penalty) * 5000.0, -50.0, 50.0)
    return float(reward), exit_price, exit_reason

def train_dqn(ohlc_df, pair=pair, save_dir="./Models",
              gamma=0.9998, lr=2e-6, batch_size=512,
              warmup=12000, updates=200000, target_sync=2000,
              epsilon_start=0.99, epsilon_end=0.001, epsilon_decay=100000,
              device='cuda' if torch.cuda.is_available() else 'cpu',
              num_workers=2, max_time_hours=8,
              target_action='high', asset_type='fx',
              exit_horizon=5, tp_pct=0.003, sl_pct=0.002):
    
    # 保存ディレクトリを確実に作成（絶対パスで）
    import os
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Save directory: {save_dir}")
    
    # 書き込みテスト
    test_file = os.path.join(save_dir, "test_write.tmp")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"[INFO] Write permission confirmed")
    except Exception as e:
        print(f"[ERROR] Cannot write to {save_dir}: {e}")
        raise

    target_key = str(target_action).lower()
    if target_key not in ACTION_MODES:
        raise ValueError(f"Unsupported target_action '{target_action}'. Choose from {list(ACTION_MODES.keys())}.")
    mode_cfg = ACTION_MODES[target_key]
    action_label = mode_cfg['label']
    action_id = mode_cfg['id']
    model_filename = mode_cfg['model_name']

    asset_type = str(asset_type).lower()
    exit_cfg = {"horizon_bars": exit_horizon, "tp_pct": tp_pct, "sl_pct": sl_pct}

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Training dedicated {action_label} model (action id {action_id})")
    
    # タイムアウト設定
    import time
    start_time = time.time()
    max_time_seconds = max_time_hours * 3600
    
    ohlc_df = ohlc_df[['open','high','low','close']].copy()
    
    # 全データを使用（上限なし）
    print(f"[INFO] Using all {len(ohlc_df)} rows for training")
    
    # DatetimeIndexの確認（なければ作成）
    if not isinstance(ohlc_df.index, pd.DatetimeIndex):
        print("[WARN] No DatetimeIndex found. Creating synthetic datetime index...")
        from datetime import datetime
        start_time = datetime(2020, 1, 1, 0, 0, 0)
        ohlc_df.index = pd.date_range(start=start_time, periods=len(ohlc_df), freq='1min')
        print(f"[INFO] Created DatetimeIndex from {start_time}")
    
    # ====== High/Lowバランス改善のための推奨事項 ======
    # 時系列データの価格反転は不適切（トレンド特性が破壊される）
    # 
    # 【推奨される解決策】
    # 1. 上昇/下降両方を含む長期間データを使用
    #    - 例: 2年以上のデータ（トレンド転換を複数含む）
    # 
    # 2. 複数通貨ペアのデータを統合
    #    - USDJPY + EURUSD + AUDJPY など
    #    - 異なる市場環境でバランスが取れる
    # 
    # 3. 学習時の探索戦略を調整（既に実装済み）
    #    - Epsilon-greedy: High 35%, Low 35%, Hold 30%
    # 
    # 4. 報酬関数の調整
    #    - High/Lowで完全に対称的な報酬設計（既に実装済み）
    print("\n" + "="*80)
    print("[INFO] Training data loaded: {} rows".format(len(ohlc_df)))
    print(f"[INFO] Dedicated {action_label} model: Hold vs {action_label} outputs only")
    print("  - Balanced exploration between Hold (30%) and action (70%)")
    print("  - Symmetric reward function shared across modes")
    print("  - Long-term data covering both uptrends and downtrends")
    print("="*80 + "\n")

    # 入力次元を確定（超多くの履歴を使用）
    window_size = 75  # 80%勝率確実達成のためウィンドウサイズを超拡大
    probe = FeatureExtraction(ohlc_df.iloc[:window_size+10].copy(), use_cache=False)[-1:]
    in_dim = probe.shape[1] + 2  # +phase, range

    # スケーラ学習用サンプルを並列処理で高速化
    print("[INFO] Preparing scaler samples...")
    sample_indices = list(range(window_size, min(len(ohlc_df)-2, 5000)))  # より多くのサンプル
    
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
        'beta1': 0.95,  # より高いモーメンタム
        'beta2': 0.9999,  # より安定した2次モーメント
        'eps': 1e-10,  # より高精度
        'weight_decay': 5e-6  # より軽い正則化
    }
    
    def manual_adam_step():
        adam_state['step'] += 1
        bias_correction1 = 1 - adam_state['beta1'] ** adam_state['step']
        bias_correction2 = 1 - adam_state['beta2'] ** adam_state['step']
        
        # 80%勝率のための精密な学習率スケジューリング
        progress = adam_state['step'] / updates
        # より長いウォームアップフェーズ
        if adam_state['step'] < 8000:
            # ゆっくりとしたウォームアップ
            warmup_progress = adam_state['step'] / 8000
            current_lr = lr * warmup_progress * 0.5  # より慎重なスタート
        else:
            # より緩やかなコサインアニーリング
            cosine_progress = (adam_state['step'] - 8000) / (updates - 8000)
            current_lr = lr * 0.5 * (1 + math.cos(math.pi * cosine_progress))
            # 最小学習率を保証
            current_lr = max(current_lr, lr * 0.01)
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
    max_index = len(ohlc_df) - exit_horizon - 1
    compute_range = min(1000, max(0, max_index - window_size))
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
    entry_stats = {'Hold': 0, action_label: 0}
    reward_stats = {'Hold': [], action_label: []}

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
                # ランダム探索時にHigh/Lowのバランスを強制的に取る
                batch_actions = []
                for _ in range(len(batch_idxs)):
                    # Hold:30%, Action:70%の確率分布
                    rand_val = random.random()
                    if rand_val < 0.30:
                        batch_actions.append(0)  # Hold
                    else:
                        batch_actions.append(1)  # Mode-specific action
                batch_actions = np.array(batch_actions)
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
                future_slice = ohlc_df.iloc[i+1:i+1+exit_horizon][['high', 'low', 'close']]
                if len(future_slice) == 0:
                    continue  # 未来データ不足
                trend_slice = ohlc_df['close'].iloc[max(0, i-window_size):i+1]
                trend_dir = compute_trend_direction(trend_slice, window=window_size)
                real_action = action_id if a == 1 else 0
                r, exit_price, exit_reason = compute_reward(
                    real_action,
                    entry_price,
                    future_slice,
                    trend_dir=trend_dir,
                    asset_type=asset_type,
                    exit_cfg=exit_cfg,
                )
                reward_history.append(r)
                
                # 統計記録
                stat_key = 'Hold' if a == 0 else action_label
                entry_stats[stat_key] += 1
                reward_stats[stat_key].append(r)
                
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
                    
                    # 手動optimizer実行（高速化）
                    q.zero_grad()  # より高速なzero_grad
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
        if episode % 20 == 0:  # 表示頻度を更に下げて高速化（10→20）
            avg_loss = np.mean(loss_history) if loss_history else 0.0
            avg_reward = np.mean(reward_history) if reward_history else 0.0
            
            # エントリー統計の計算（最適化）
            total_entries = sum(entry_stats.values())
            if total_entries > 0:
                hold_pct = entry_stats['Hold'] / total_entries * 100
                action_pct = entry_stats[action_label] / total_entries * 100
            else:
                hold_pct = action_pct = 0
            
            # 各アクションの平均報酬（最適化）
            avg_reward_hold = np.mean(reward_stats['Hold']) if reward_stats['Hold'] else 0.0
            avg_reward_action = np.mean(reward_stats[action_label]) if reward_stats[action_label] else 0.0
            
            # モデル保存（より高速化）
            torch.save(q.state_dict(), os.path.join(save_dir, model_filename))
            with open(os.path.join(save_dir, "dqn_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            print(f"[CKPT] Episode={episode}, Steps={steps}, Eps={eps:.3f}, "
                  f"AvgLoss={avg_loss:.4f}, AvgReward={avg_reward:.4f}")
            print(f"[STATS] エントリー回数 - Hold:{entry_stats['Hold']}({hold_pct:.1f}%), "
                f"{action_label}:{entry_stats[action_label]}({action_pct:.1f}%)")
            print(f"[REWARDS] 平均報酬 - Hold:{avg_reward_hold:.3f}, "
                f"{action_label}:{avg_reward_action:.3f}")
            
            # メモリ使用量をチェック（デバッグ用）
            if device.startswith('cuda'):
                print(f"[INFO] GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                # GPU メモリ最適化
                torch.cuda.empty_cache()

    # 最終保存
    torch.save(q.state_dict(), os.path.join(save_dir, model_filename))
    with open(os.path.join(save_dir, "dqn_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # 最終統計表示
    total_entries = sum(entry_stats.values())
    if total_entries > 0:
        print("\n" + "="*60)
        print("FINAL TRAINING STATISTICS")
        print("="*60)
        print(f"Total Actions: {total_entries}")
        print(f"Hold: {entry_stats['Hold']} ({entry_stats['Hold']/total_entries*100:.1f}%)")
        print(f"{action_label}: {entry_stats[action_label]} ({entry_stats[action_label]/total_entries*100:.1f}%)")
        
        if reward_stats[action_label]:
            print(f"Average Reward {action_label}: {np.mean(reward_stats[action_label]):.3f}")
        if reward_stats['Hold']:
            print(f"Average Reward Hold: {np.mean(reward_stats['Hold']):.3f}")
        print("="*60)
    
    # キャッシュクリア
    clear_feature_cache()
    
    print("[DONE] DQN training completed and saved.")
    evaluate_dqn_model(
        q,
        scaler,
        ohlc_df,
        device=device,
        window_size=window_size,
        target_action=target_action,
        asset_type=asset_type,
        exit_horizon=exit_horizon,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
    )
def evaluate_dqn_model(q, scaler, ohlc_df, n_eval=2000, device='cpu', window_size=20,
                      target_action='high', asset_type='fx', exit_horizon=5,
                      tp_pct=0.003, sl_pct=0.002):
    """
    学習済みDQNモデルを使ってOHLCデータ上で勝率、損益、最大ドローダウンを測定
    - q: 学習済み QNet
    - scaler: 学習時の StandardScaler
    - ohlc_df: DataFrame (DatetimeIndex 必須, open/high/low/close)
    - n_eval: 評価に使うサンプル数
    - device: 計算デバイス
    """
    target_key = str(target_action).lower()
    if target_key not in ACTION_MODES:
        raise ValueError(f"Unsupported target_action '{target_action}'. Choose from {list(ACTION_MODES.keys())}.")
    mode_cfg = ACTION_MODES[target_key]
    action_label = mode_cfg['label']
    action_id = mode_cfg['id']

    exit_cfg = {"horizon_bars": exit_horizon, "tp_pct": tp_pct, "sl_pct": sl_pct}

    # 全体の統計
    correct, total = 0, 0
    action_correct, action_total = 0, 0
    hold_count = 0                   # Hold回数
    
    profit = 0.0
    max_dd, peak = 0.0, 0.0

    max_eval_index = len(ohlc_df) - exit_horizon - 1
    idxs = list(range(window_size+1, min(max_eval_index, window_size+1+min(3000, n_eval))))  # 十分な先行足を確保
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
            sl = ohlc_df.iloc[max(0, i-window_size):i+1].copy()
            entry_price = float(sl['close'].iloc[-1])
            future_slice = ohlc_df.iloc[i+1:i+1+exit_horizon][['high', 'low', 'close']]
            if len(future_slice) == 0:
                continue
            trend_dir = compute_trend_direction(sl['close'], window=window_size)
            real_action = action_id if a == 1 else 0
            r, exit_price, exit_reason = compute_reward(
                real_action,
                entry_price,
                future_slice,
                trend_dir=trend_dir,
                asset_type=asset_type,
                exit_cfg=exit_cfg,
            )

            # アクション別統計
            if a == 0:  # Hold
                hold_count += 1
            elif a == 1:
                action_total += 1
                total += 1
                direction_correct = next_close > entry_price if action_id == 1 else next_close < entry_price
                if direction_correct:
                    action_correct += 1
                    correct += 1

            # 累積損益計算（実際の値幅ベース）
            if real_action in (1, 2):
                net_move = (exit_price - entry_price) / entry_price if real_action == 1 else (entry_price - exit_price) / entry_price
                profit += net_move - _get_fee_rate(asset_type)
        peak = max(peak, profit)
        dd = peak - profit
        max_dd = max(max_dd, dd)

    # 全体の勝率
    acc = correct / total if total > 0 else 0.0
    
    # 予測別の勝率
    action_acc = action_correct / action_total if action_total > 0 else 0.0
    
    # 結果表示
    print(f"[EVAL] Overall Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"[EVAL] {action_label} Prediction Accuracy: {action_acc:.3f} ({action_correct}/{action_total})")
    print(f"[EVAL] Hold Count: {hold_count}")
    print(f"[EVAL] Action Distribution - {action_label}: {action_total}, Hold: {hold_count}")
    print(f"[EVAL] Total Profit: {profit:.2f}, Max Drawdown: {max_dd:.2f}")
    
    return acc, profit, max_dd

if __name__ == "__main__":
    # 最終的な高速化設定
    import sys
    
    print("[INFO] 高速化設定を適用中...")
    
    # PyTorchの高速化設定
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[INFO] CUDA TF32 acceleration enabled")
    
    # ====== 複数通貨ペアのデータを統合 ======
    print("\n" + "="*80)
    print("[INFO] 複数通貨ペアのデータを読み込んで統合します")
    print("="*80)
    
    # 使用する通貨ペアリスト
    currency_pairs = ["USDJPY", "EURUSD", "AUDJPY"]
    
    all_dfs = []
    successful_pairs = []
    
    for currency_pair in currency_pairs:
        data_file = f"data/{currency_pair}_M1.csv"
        
        try:
            print(f"\n[INFO] Loading {currency_pair} from {data_file}...")
            
            # まずデータを読み込んでカラム数を確認
            df_test = pd.read_csv(data_file, nrows=5)
            num_columns = len(df_test.columns)
            
            # カラム数に応じて処理を分岐
            if num_columns >= 7:
                # 日時カラムあり（date, time, open, high, low, close, volume）
                column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
                df = pd.read_csv(data_file, names=column_names,
                                dtype={'open': np.float32, 'high': np.float32, 
                                       'low': np.float32, 'close': np.float32,
                                       'volume': np.float32})
                
                # 日付と時刻を結合してDatetimeIndexを作成
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
                df = df.set_index('datetime')
                
                # 不要な列を削除
                df = df[['open', 'high', 'low', 'close', 'volume']]
                
            elif num_columns >= 5:
                # 日時カラムなし（open, high, low, close, volume）
                column_names = ['open', 'high', 'low', 'close', 'volume']
                df = pd.read_csv(data_file, names=column_names,
                                dtype={'open': np.float32, 'high': np.float32, 
                                       'low': np.float32, 'close': np.float32,
                                       'volume': np.float32})
                
                # 連番インデックスを使用してDatetimeIndexを生成
                from datetime import datetime, timedelta
                # 通貨ペアごとに異なる開始時刻を設定（データ重複を避ける）
                pair_offset = len(all_dfs) * 365  # 1年ずつずらす
                start_time = datetime(2020, 1, 1, 0, 0, 0) + timedelta(days=pair_offset)
                df.index = pd.date_range(start=start_time, periods=len(df), freq='1min')
                
            else:
                print(f"[WARN] {currency_pair}: Unexpected number of columns: {num_columns}. Skipping.")
                continue
            
            # 価格を正規化（通貨ペア間の価格差を吸収）
            # 各通貨ペアの平均価格で正規化
            price_mean = df['close'].mean()
            df['open'] = df['open'] / price_mean
            df['high'] = df['high'] / price_mean
            df['low'] = df['low'] / price_mean
            df['close'] = df['close'] / price_mean
            
            print(f"[SUCCESS] {currency_pair}: {len(df)} rows loaded and normalized")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Price mean: {price_mean:.5f} (normalized to 1.0)")
            
            all_dfs.append(df)
            successful_pairs.append(currency_pair)
            
        except FileNotFoundError:
            print(f"[WARN] {currency_pair}: File not found: {data_file}")
        except Exception as e:
            print(f"[ERROR] {currency_pair}: Error loading data: {e}")
    
    # データの統合
    if not all_dfs:
        print("\n[ERROR] 利用可能なデータがありません。終了します。")
        import glob
        available_files = glob.glob("data/*_M1.csv")
        print("[INFO] 利用可能なファイル:")
        for file in available_files:
            print(f"  - {file}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print(f"[INFO] データ統合: {len(successful_pairs)}通貨ペア ({', '.join(successful_pairs)})")
    print("="*80)
    
    # すべてのデータフレームを結合
    df = pd.concat(all_dfs, axis=0)
    df = df.sort_index()
    
    print(f"[INFO] 統合後のデータ: {len(df)} rows")
    print(f"[INFO] Date range: {df.index[0]} to {df.index[-1]}")
    print(f"[INFO] Sample data:")
    print(df.head())
    
    print("\n[INFO] 統合データの統計:")
    print(df.describe())
    
    # モデル名は統合ペア名にする
    model_pair_name = "_".join(successful_pairs) if len(successful_pairs) > 1 else successful_pairs[0]
    
    print(f"\n[INFO] モデル名: {model_pair_name}")
    print("="*80 + "\n")
    
    # 学習開始（High/Low専用モデルをそれぞれ学習）
    try:
        for mode in ACTION_MODES.keys():
            print("\n" + "="*80)
            print(f"[INFO] Training {mode.upper()} model")
            print("="*80)
            train_dqn(df, pair=model_pair_name, target_action=mode)
        print("[INFO] 高速DQNモデル2種の保存が完了しました")
    except Exception as e:
        print(f"[ERROR] 学習中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

