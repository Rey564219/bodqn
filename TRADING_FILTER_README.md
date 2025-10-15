# 取引フィルター機能説明

## 📋 実装した機能

### 1. 連敗検出と自動ブロック機能 🚫

特定方向（HighまたはLow）で連続して負けた場合、その方向のエントリーを **3分間自動的にブロック** します。

#### 設定パラメータ
- `CONSECUTIVE_LOSS_THRESHOLD = 3`: 連続負け回数の閾値（3回連続で負けたらブロック）
- `ENTRY_BLOCK_DURATION_SECONDS = 180`: ブロック時間（3分=180秒）
- `LOSS_LOOKBACK_MINUTES = 5`: 直近何分間の負け履歴を確認するか

#### 動作の流れ
1. **連敗検出**: 直近5分間でHigh負けが3回連続 → Highエントリーを3分間ブロック
2. **ブロック中**: 残り時間を表示しながらHold判定に変更
3. **3分経過後**: トレンドの状況を確認
   - トレンドが上昇に転換 → ブロック解除（High再開）
   - まだ下降傾向 → ブロック継続

### 2. 取引結果の自動スクレイピング 🔍

エントリー後、60秒後に自動的に取引結果を確認し、負け履歴に追加します。

#### 動作
- エントリー実行時に `pending_trades` リストに追加
- 60秒経過後、Webページから取引結果をスクレイピング
- 負けた場合は自動的に `loss_history` に記録

### 3. 手動での負け履歴追加機能 ✍️

自動スクレイピングがうまく動かない場合、手動で負け履歴を追加できます。

#### 使用方法
プログラム実行中のコンソールで以下のコマンドを実行：

```python
# High負けを記録
add_loss_to_history(loss_history, 'High', 150.123)

# Low負けを記録
add_loss_to_history(loss_history, 'Low', 150.456)

# 過去の時刻を指定して記録
from datetime import datetime, timedelta
past_time = datetime.now() - timedelta(minutes=2)
add_loss_to_history(loss_history, 'High', 150.789, past_time)
```

### 4. リアルタイムブロック状態表示 📊

ログ出力で現在のブロック状態を常に確認できます：

```
[📈 SLOPE] 傾き方向:下降, 傾き値:-0.00015234
[📍 SLOPE] 基準点: 14:35:21 @ 150.123 (最初の負けエントリー)
[📊 LOSS] 直近負け - High:3回, Low:0回
[🚫 BLOCK] High判定ブロック中 - 残り145秒（14:38:21まで）
```

## 🎯 Low判定が出ない問題の修正

### 原因
元のコードでは以下の厳しい条件がありました：
```python
should_block_low = (
    not is_declining and          # 上昇傾向
    normalized_slope > -PRICE_SLOPE_THRESHOLD and  # 明確な上昇
    recent_low_losses >= CONSECUTIVE_LOSS_THRESHOLD
)
```

### 修正内容
Low判定のブロック条件を **連敗のみ** に簡素化しました：
```python
should_block_low = (
    recent_low_losses >= CONSECUTIVE_LOSS_THRESHOLD
)
```

これにより、Low判定も公平に出力されるようになります。

## 📝 ログの見方

### Q値の確認
```
[Q-VALUES] Hold:0.1234, High:0.5678, Low:0.4321
[ACTION] 選択されたアクション: High (idx:1)
```

### フィルター適用
```
[🛡️ FILTER] アクション変更: High -> Hold
```

### エントリー実行
```
[ENTRY] High at 14:35:21 price=150.123 Q値: 0.5678 (優位性: 0.4444)
[INFO] 取引を待ちリストに追加（60秒後に結果確認）
```

### 取引結果確認
```
[⏰ CHECK] High取引の結果確認 (エントリー: 14:35:21 @ 150.123)
[RESULT] 負け記録追加: High @ 150.123
```

## 🔧 カスタマイズ

### ブロック時間を変更
```python
ENTRY_BLOCK_DURATION_SECONDS = 180  # 3分 → 好みの秒数に変更
```

### 連敗閾値を変更
```python
CONSECUTIVE_LOSS_THRESHOLD = 3  # 3回 → 好みの回数に変更
```

### 負け履歴の参照期間を変更
```python
LOSS_LOOKBACK_MINUTES = 5  # 5分 → 好みの分数に変更
```

## ⚠️ 注意事項

1. **自動スクレイピング**: サイトの構造が変わった場合、`scrape_trade_results()` 関数を調整する必要があります。

2. **手動記録の推奨**: 自動スクレイピングが不安定な場合は、手動で負け履歴を記録することをお勧めします。

3. **ブロック解除**: 3分経過後、トレンドが変わっていない場合はブロックが継続されます。

4. **ログファイル**: すべての判定結果は `logs/live_signals_USDJPY.csv` に記録されます。

## 🚀 使い方

1. プログラムを起動
2. 取引が自動実行される
3. 負けた場合は自動的に記録される（または手動で記録）
4. 3回連続で負けた方向のエントリーが3分間ブロックされる
5. 3分後、トレンドが変われば自動的にブロック解除

すべて自動で動作しますが、必要に応じて手動で負け履歴を管理できます。
