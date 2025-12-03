import os
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# --- TA-Lib fallback handling -------------------------------------------------
try:
    import talib as ta  # type: ignore
except ImportError:  # pragma: no cover - fallback runtime path
    try:
        import ta  # type: ignore
    except ImportError:
        class MockTA:
            @staticmethod
            def EMA(close, period):
                return pd.Series(close).ewm(span=period).mean().values

            @staticmethod
            def RSI(close, period):
                close_series = pd.Series(close)
                delta = close_series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss.replace(0, np.nan))
                rsi = 100 - (100 / (1 + rs))
                return rsi.fillna(50).values

            @staticmethod
            def BBANDS(close, period, nbdevup=2, nbdevdn=2, matype=0):
                series = pd.Series(close)
                ma = series.rolling(period).mean()
                std = series.rolling(period).std().fillna(0)
                upper = ma + (std * nbdevup)
                lower = ma - (std * nbdevdn)
                return upper.values, ma.values, lower.values

            @staticmethod
            def ATR(high, low, close, period):
                high_series = pd.Series(high)
                low_series = pd.Series(low)
                close_series = pd.Series(close)
                tr1 = high_series - low_series
                tr2 = (high_series - close_series.shift()).abs()
                tr3 = (low_series - close_series.shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                return tr.rolling(period).mean().fillna(0).values

            @staticmethod
            def MOM(close, period):
                return pd.Series(close).diff(period).fillna(0).values

            @staticmethod
            def STOCH(high, low, close, period):
                high_series = pd.Series(high)
                low_series = pd.Series(low)
                close_series = pd.Series(close)
                lowest_low = low_series.rolling(period).min()
                highest_high = high_series.rolling(period).max()
                k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low + 1e-8))
                d_percent = k_percent.rolling(3).mean()
                return k_percent.fillna(0).values, d_percent.fillna(0).values

            @staticmethod
            def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
                series = pd.Series(close)
                exp1 = series.ewm(span=fastperiod).mean()
                exp2 = series.ewm(span=slowperiod).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=signalperiod).mean()
                histogram = macd - signal
                return macd.values, signal.values, histogram.values

            @staticmethod
            def WILLR(high, low, close, period=14):
                high_series = pd.Series(high)
                low_series = pd.Series(low)
                close_series = pd.Series(close)
                highest_high = high_series.rolling(period).max()
                lowest_low = low_series.rolling(period).min()
                wr = -100 * ((highest_high - close_series) / (highest_high - lowest_low + 1e-8))
                return wr.fillna(0).values

            @staticmethod
            def CCI(high, low, close, period=20):
                high_series = pd.Series(high)
                low_series = pd.Series(low)
                close_series = pd.Series(close)
                tp = (high_series + low_series + close_series) / 3
                sma = tp.rolling(period).mean()
                mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
                cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
                return cci.fillna(0).values

            @staticmethod
            def ROC(close, period=10):
                series = pd.Series(close)
                return ((series - series.shift(period)) / series.shift(period + 1e-8) * 100).fillna(0).values

        ta = MockTA()


_feature_cache = {}


def _CalcSMAR(df, periods):
    sma_features = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for period in periods:
            ema = ta.EMA(df["close"], period)
            close_vals = df["close"].values
            safe_mask = (close_vals != 0) & np.isfinite(close_vals)
            ema_vals = np.asarray(ema)
            sma_features[f"SMAR_{period}"] = np.where(
                safe_mask,
                np.divide(ema_vals, close_vals, out=np.ones_like(close_vals), where=safe_mask),
                1.0,
            )
    if sma_features:
        sma_df = pd.DataFrame(sma_features, index=df.index)
        df = pd.concat([df, sma_df], axis=1)
    return df


def _CalcRSIR(df, periods):
    rsi_features = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for period in periods:
            rsi_val = ta.RSI(df["close"], period)
            rsi_features[f"RSIR_{period}"] = rsi_val
            rsi_features[f"RSIR_diff_{period}"] = pd.Series(rsi_val, index=df.index).diff()
    if rsi_features:
        rsi_df = pd.DataFrame(rsi_features, index=df.index)
        df = pd.concat([df, rsi_df], axis=1)
    return df


def _CalcOtherR(df, periods):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other_features = {}
        for period in periods:
            if len(df) < period:
                continue
            try:
                upper, _, lower = ta.BBANDS(df['close'], period)
                upper = np.asarray(upper)
                lower = np.asarray(lower)
                bb_width = upper - lower
                other_features[f'bb_width{period}'] = bb_width
                other_features[f'bb_width_diff{period}'] = pd.Series(bb_width, index=df.index).diff()
                width = upper - lower
                safe_width = np.where((width != 0) & np.isfinite(width), width, 1.0)
                other_features[f'bb_percent{period}'] = np.where(safe_width != 1.0, (df['close'] - lower) / safe_width, 0.5)
                atr_val = ta.ATR(df['high'], df['low'], df['close'], period)
                other_features[f'atr{period}'] = atr_val
                other_features[f'atr_diff{period}'] = pd.Series(atr_val, index=df.index).diff()
                other_features[f'price_change{period}'] = df['close'].pct_change(period).clip(-1, 1)
                other_features[f'volatility{period}'] = df['close'].rolling(period).std()
                momentum_val = ta.MOM(df['close'], period)
                close_vals = df['close'].values
                momentum_vals = np.asarray(momentum_val)
                safe_close = np.where((close_vals != 0) & np.isfinite(close_vals), close_vals, 1.0)
                other_features[f'momentum{period}'] = momentum_vals
                other_features[f'momentum_norm{period}'] = np.divide(
                    momentum_vals,
                    safe_close,
                    out=np.zeros_like(momentum_vals),
                    where=safe_close != 1.0,
                )
                high_max = df['high'].rolling(period).max()
                low_min = df['low'].rolling(period).min()
                range_diff = high_max - low_min + 1e-8
                other_features[f'high_pos{period}'] = (df['close'] - low_min) / range_diff
            except Exception:
                continue
        if other_features:
            other_df = pd.DataFrame(other_features, index=df.index)
            df = pd.concat([df, other_df], axis=1)
    extra_features = {}
    try:
        slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'], 14)
        extra_features['slowk'] = slowk
        extra_features['slowk_diff'] = pd.Series(slowk, index=df.index).diff()
        extra_features['slowd'] = slowd
        extra_features['slowd_diff'] = pd.Series(slowd, index=df.index).diff()
        macd, macdsignal, macdhist = ta.MACD(df['close'])
        extra_features['macd'] = macd
        extra_features['macd_signal'] = macdsignal
        extra_features['macd_hist'] = macdhist
        extra_features['macd_cross'] = np.where(np.asarray(macd) > np.asarray(macdsignal), 1, -1)
        extra_features['williams_r'] = ta.WILLR(df['high'], df['low'], df['close'])
        extra_features['cci'] = ta.CCI(df['high'], df['low'], df['close'])
        extra_features['roc'] = ta.ROC(df['close'])
    except Exception:
        pass
    if extra_features:
        extra_df = pd.DataFrame(extra_features, index=df.index)
        df = pd.concat([df, extra_df], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].ffill().fillna(0)
    return df.copy()


def FeatureExtraction(df, use_cache=True):
    if use_cache:
        df_hash = hash((tuple(df.iloc[-1].values), len(df)))
        if df_hash in _feature_cache:
            return _feature_cache[df_hash]
    df = df.copy()
    periods_RSI = [7, 14, 21, 28]
    periods_SMA = [5, 10, 20, 50, 100]
    with ThreadPoolExecutor(max_workers=min(4,  os.cpu_count() or 1)) as executor:
        future_sma = executor.submit(_CalcSMAR, df, periods_SMA)
        future_rsi = executor.submit(_CalcRSIR, df, periods_RSI)
        future_other = executor.submit(_CalcOtherR, df, periods_RSI)
        df = future_sma.result()
        df = future_rsi.result()
        df = future_other.result()
    close = df["close"].values
    open_vals = df["open"].values
    high = df["high"].values
    low = df["low"].values
    basic_features = {}
    close_nonzero = (close != 0) & np.isfinite(close)
    basic_features["open_r"] = np.where(close_nonzero, np.divide(open_vals, close, out=np.ones_like(close), where=close_nonzero), 1.0)
    basic_features["high_r"] = np.where(close_nonzero, np.divide(high, close, out=np.ones_like(close), where=close_nonzero), 1.0)
    basic_features["low_r"] = np.where(close_nonzero, np.divide(low, close, out=np.ones_like(close), where=close_nonzero), 1.0)
    hl_diff = high - low
    oc_diff = open_vals - close
    hl_nonzero = (hl_diff != 0) & np.isfinite(hl_diff)
    basic_features["hl_ratio"] = np.where(close_nonzero, np.divide(hl_diff, close, out=np.zeros_like(close), where=close_nonzero), 0.0)
    basic_features["oc_ratio"] = np.where(close_nonzero, np.divide(oc_diff, close, out=np.zeros_like(close), where=close_nonzero), 0.0)
    basic_features["body_ratio"] = np.where(hl_nonzero, np.divide(np.abs(oc_diff), hl_diff, out=np.zeros_like(hl_diff), where=hl_nonzero), 0.0)
    max_oc = np.maximum(open_vals, close)
    min_oc = np.minimum(open_vals, close)
    basic_features["upper_shadow"] = np.where(hl_nonzero, np.divide(high - max_oc, hl_diff, out=np.zeros_like(hl_diff), where=hl_nonzero), 0.0)
    basic_features["lower_shadow"] = np.where(hl_nonzero, np.divide(min_oc - low, hl_diff, out=np.zeros_like(hl_diff), where=hl_nonzero), 0.0)
    basic_df = pd.DataFrame(basic_features, index=df.index)
    df = pd.concat([df, basic_df], axis=1)
    momentum_features = {}
    for period in [3, 5, 10, 20]:
        if len(df) >= period:
            momentum_features[f"momentum_{period}"] = df["close"].pct_change(period).fillna(0)
            momentum_features[f"volatility_{period}"] = df["close"].rolling(period).std().fillna(0)
            momentum_features[f"volume_price_trend_{period}"] = (df["close"].diff() * df.get("volume", 1)).rolling(period).sum().fillna(0)
    if momentum_features:
        momentum_df = pd.DataFrame(momentum_features, index=df.index)
        df = pd.concat([df, momentum_df], axis=1)
    ma_features = {}
    for period in [3, 5, 10, 20, 50, 100]:
        if len(df) >= period:
            sma = df["close"].rolling(period).mean()
            ema = df["close"].ewm(span=period).mean()
            wma = df["close"].rolling(period).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
            ma_features[f"sma_distance_{period}"] = np.where(sma != 0, (df["close"] - sma) / sma, 0.0)
            ma_features[f"ema_distance_{period}"] = np.where(ema != 0, (df["close"] - ema) / ema, 0.0)
            ma_features[f"wma_distance_{period}"] = np.where(wma != 0, (df["close"] - wma) / wma, 0.0)
            ma_features[f"sma_ema_diff_{period}"] = np.where(ema != 0, (sma - ema) / ema, 0.0)
            ma_features[f"sma_slope_{period}"] = sma.diff().fillna(0)
            ma_features[f"ema_slope_{period}"] = ema.diff().fillna(0)
    if ma_features:
        ma_df = pd.DataFrame(ma_features, index=df.index)
        df = pd.concat([df, ma_df], axis=1)
    cross_features = {}
    for fast, slow in [(5, 20), (10, 50), (20, 100)]:
        if len(df) >= slow:
            sma_fast = df["close"].rolling(fast).mean()
            sma_slow = df["close"].rolling(slow).mean()
            cross_features[f"golden_cross_{fast}_{slow}"] = np.where(sma_fast > sma_slow, 1, 0)
            cross_features[f"death_cross_{fast}_{slow}"] = np.where(sma_fast < sma_slow, 1, 0)
            cross_features[f"ma_convergence_{fast}_{slow}"] = np.where(sma_slow != 0, (sma_fast - sma_slow) / sma_slow, 0.0)
    if cross_features:
        cross_df = pd.DataFrame(cross_features, index=df.index)
        df = pd.concat([df, cross_df], axis=1)
    rsi_features = {}
    for period in periods_RSI:
        rsi_col = f"RSIR_{period}"
        if rsi_col in df.columns:
            rsi_features[f"rsi_overbought_{period}"] = np.where(df[rsi_col] > 70, 1, 0)
            rsi_features[f"rsi_oversold_{period}"] = np.where(df[rsi_col] < 30, 1, 0)
            rsi_features[f"rsi_neutral_{period}"] = np.where((df[rsi_col] >= 40) & (df[rsi_col] <= 60), 1, 0)
            rsi_features[f"rsi_momentum_{period}"] = df[rsi_col].diff().fillna(0)
    if rsi_features:
        rsi_df = pd.DataFrame(rsi_features, index=df.index)
        df = pd.concat([df, rsi_df], axis=1)
    bb_features = {}
    for period in [10, 20]:
        if len(df) >= period:
            sma = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            bb_position = np.where(std != 0, (df["close"] - sma) / (2 * std), 0.0)
            bb_width = np.where(sma != 0, (upper_band - lower_band) / sma, 0.0)
            bb_features[f"bb_position_{period}"] = bb_position
            bb_features[f"bb_width_{period}"] = bb_width
            bb_features[f"bb_squeeze_{period}"] = np.where(pd.Series(bb_width).rolling(10).mean() > bb_width, 1, 0)
    if bb_features:
        bb_df = pd.DataFrame(bb_features, index=df.index)
        df = pd.concat([df, bb_df], axis=1)
    pattern_features = {}
    pattern_features["pin_bar_bull"] = np.where((df["lower_shadow"] > 0.6) & (df["body_ratio"] < 0.3) & (df["upper_shadow"] < 0.3), 1, 0)
    pattern_features["pin_bar_bear"] = np.where((df["upper_shadow"] > 0.6) & (df["body_ratio"] < 0.3) & (df["lower_shadow"] < 0.3), 1, 0)
    pattern_features["doji"] = np.where(df["body_ratio"] < 0.1, 1, 0)
    bullish_candle = np.where(df["close"] > df["open"], 1, 0)
    bearish_candle = np.where(df["close"] < df["open"], 1, 0)
    pattern_features["bullish_candle"] = bullish_candle
    pattern_features["bearish_candle"] = bearish_candle
    pattern_features["consecutive_bull"] = pd.Series(bullish_candle, index=df.index).rolling(3).sum()
    pattern_features["consecutive_bear"] = pd.Series(bearish_candle, index=df.index).rolling(3).sum()
    pattern_features["prev_close_ratio"] = df["close"].pct_change().clip(-1, 1)
    pattern_features["prev_volume_ratio"] = df["volume"].pct_change().clip(-10, 10) if "volume" in df.columns else 0
    pattern_df = pd.DataFrame(pattern_features, index=df.index)
    df = pd.concat([df, pattern_df], axis=1)
    new_columns = {}
    for lookback in [2, 3, 5]:
        new_columns[f"price_change_{lookback}"] = df["close"].pct_change(lookback).clip(-1, 1)
    for period in [10, 20]:
        new_columns[f"high_breakout_{period}"] = (df["high"] > df["high"].rolling(period).max().shift(1)).astype(int)
        new_columns[f"low_breakout_{period}"] = (df["low"] < df["low"].rolling(period).min().shift(1)).astype(int)
    new_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    result = df.drop(columns=["open", "close", "high", "low", "volume"], errors='ignore')
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0)
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        upper_limit = result[col].quantile(0.999)
        lower_limit = result[col].quantile(0.001)
        result[col] = result[col].clip(lower=lower_limit, upper=upper_limit)
    if use_cache:
        _feature_cache[df_hash] = result.copy()
        if len(_feature_cache) > 5000:
            keys_to_remove = list(_feature_cache.keys())[:1000]
            for key in keys_to_remove:
                _feature_cache.pop(key, None)
    return result.copy()


def build_state_vec(ohlc_win_df, extra=None, use_cache=True):
    feat = FeatureExtraction(ohlc_win_df, use_cache=use_cache)[-1:]
    x = feat.values.astype(np.float32).reshape(-1)
    if extra is not None:
        x = np.concatenate([x, np.asarray(extra, dtype=np.float32)])
    return x.astype(np.float32)


def build_state_vec_fast(ohlc_slice, phase, sec_range):
    feat = FeatureExtraction(ohlc_slice, use_cache=False)
    x = feat.iloc[-1].values.astype(np.float32)
    extra_features = np.array([phase, sec_range], dtype=np.float32)
    return np.concatenate([x, extra_features])


def clear_feature_cache():
    _feature_cache.clear()


__all__ = [
    "FeatureExtraction",
    "build_state_vec",
    "build_state_vec_fast",
    "clear_feature_cache",
]
