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

def FeatureExtraction(df):
    df = df.copy()
    periods_RSI = [3, 5, 10]
    periods_SMA = [3, 5, 10]

    df = _CalcSMAR(df, periods_SMA)
    df = _CalcRSIR(df, periods_RSI)
    df = _CalcOtherR(df, periods_RSI)

    df["open_r"] = df["open"]/df["close"]
    df["high_r"] = df["high"]/df["close"]
    df["low_r"] = df["low"]/df["close"]
    df = df.drop(columns = ["open", "close", "high", "low"])
 
    return df
ACTIONS = 3  # 0:Hold, 1:High, 2:Low

class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x): return self.net(x)

class Replay:
    def __init__(self, cap=100_000): self.buf=deque(maxlen=cap)
    def push(self, *exp): self.buf.append(exp)
    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s,a,r,ns,done = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(done, dtype=np.float32))
    def __len__(self): return len(self.buf)

def build_state_vec(ohlc_win_df, extra=None):
    feat = FeatureExtraction(ohlc_win_df)[-1:]
    x = feat.values.astype(np.float32).reshape(-1)
    if extra is not None:
        x = np.concatenate([x, np.asarray(extra, dtype=np.float32)])
    return x

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
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=80000):
    os.makedirs(save_dir, exist_ok=True)

    ohlc_df = ohlc_df[['open','high','low','close']].copy()
    if not isinstance(ohlc_df.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex が必要です")

    # 入力次元を確定
    probe = FeatureExtraction(ohlc_df.iloc[:20].copy())[-1:]
    in_dim = probe.shape[1] + 2  # +phase, range

    # スケーラ学習用サンプル
    samples = []
    for i in range(20, min(len(ohlc_df)-2, 2000)):
        sl = ohlc_df.iloc[i-20:i+1].copy()
        phase = (sl.index[-1].second % 60)/60.0
        sec_range = float(sl['high'].iloc[-1] - sl['low'].iloc[-1])
        samples.append(build_state_vec(sl, [phase, sec_range]))
    scaler = StandardScaler().fit(np.array(samples))

    q = QNet(in_dim, ACTIONS)
    tgt = QNet(in_dim, ACTIONS); tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    mem = Replay()

    steps, eps = 0, epsilon_start
    idxs = list(range(21, len(ohlc_df)-1))
    episode = 0

    while steps < updates:
        random.shuffle(idxs)
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
                    qv = q(torch.from_numpy(s).unsqueeze(0))
                    a = int(torch.argmax(qv, dim=1).item())

            entry_price = float(sl['close'].iloc[-1])
            next_close = float(ohlc_df['close'].iloc[i+1])
            r = compute_reward(a, next_close, entry_price)

            sl_next = ohlc_df.iloc[i-19:i+2].copy()
            phase_n = (sl_next.index[-1].second % 60)/60.0
            sec_range_n = float(sl_next['high'].iloc[-1] - sl_next['low'].iloc[-1])
            ns = build_state_vec(sl_next, [phase_n, sec_range_n])
            ns = scaler.transform([ns])[0]

            mem.push(s, a, r, ns, 0.0)
            steps += 1

            if len(mem) >= warmup:
                S,A,R,NS,DN = mem.sample(batch_size)
                S = torch.from_numpy(S); A = torch.from_numpy(A)
                R = torch.from_numpy(R); NS = torch.from_numpy(NS)
                q_sa = q(S).gather(1, A.view(-1,1)).squeeze(1)
                with torch.no_grad():
                    next_max = tgt(NS).max(1)[0]
                    tgt_q = R + gamma * next_max
                loss = nn.functional.smooth_l1_loss(q_sa, tgt_q)
                opt.zero_grad(); loss.backward(); opt.step()

                if steps % target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

                eps = epsilon_end + (epsilon_start-epsilon_end) * math.exp(-(steps-warmup)/epsilon_decay)

            if steps >= updates: break
        episode += 1
        if episode % 5 == 0:
            torch.save(q.state_dict(), os.path.join(save_dir, f"dqn_policy_{pair}.pt"))
            with open(os.path.join(save_dir, f"dqn_scaler_{pair}.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            print(f"[CKPT] steps={steps}, eps={eps:.3f}")

    torch.save(q.state_dict(), os.path.join(save_dir, f"dqn_policy_{pair}.pt"))
    with open(os.path.join(save_dir, f"dqn_scaler_{pair}.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("[DONE] DQN saved.")
    evaluate_dqn_model(q, scaler, ohlc_df)
def evaluate_dqn_model(q, scaler, ohlc_df, n_eval=2000):
    """
    学習済みDQNモデルを使ってOHLCデータ上で勝率、損益、最大ドローダウンを測定
    - q: 学習済み QNet
    - scaler: 学習時の StandardScaler
    - ohlc_df: DataFrame (DatetimeIndex 必須, open/high/low/close)
    - n_eval: 評価に使うサンプル数
    """
    correct, total = 0, 0
    profit = 0.0
    max_dd, peak = 0.0, 0.0

    idxs = list(range(21, min(len(ohlc_df)-1, n_eval)))
    q.eval()

    for i in idxs:
        sl = ohlc_df.iloc[i-20:i+1].copy()
        phase = (sl.index[-1].second % 60)/60.0
        sec_range = float(sl['high'].iloc[-1] - sl['low'].iloc[-1])
        s = build_state_vec(sl, [phase, sec_range])
        s = scaler.transform([s])[0]

        with torch.no_grad():
            q_values = q(torch.from_numpy(s).unsqueeze(0))
            a = int(torch.argmax(q_values, dim=1).item())

        entry_price = float(sl['close'].iloc[-1])
        next_close = float(ohlc_df['close'].iloc[i+1])
        r = compute_reward(a, next_close, entry_price)

        # Holdはスキップ、High/Lowの的中率を評価
        if a in (1,2):
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
