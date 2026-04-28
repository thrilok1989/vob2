import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import streamlit as st

class PivotIndicator:
    """Higher Timeframe Pivot Support/Resistance Indicator"""

    @staticmethod
    def pivot_high(series, left, right):
        max_values = series.rolling(window=left+right+1, center=True).max()
        return series == max_values

    @staticmethod
    def pivot_low(series, left, right):
        min_values = series.rolling(window=left+right+1, center=True).min()
        return series == min_values

    @staticmethod
    def resample_ohlc(df, tf):
        rule_map = {
            "3": "3min",
            "5": "5min",
            "10": "10min",
            "15": "15min",
            "60": "60min",
            "D": "1D",
            "W": "1W"
        }
        rule = rule_map.get(tf, tf)
        if df.empty or 'datetime' not in df.columns:
            return pd.DataFrame()
        df_temp = df.copy()
        df_temp.set_index('datetime', inplace=True)
        try:
            resampled = df_temp.resample(rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()
            return resampled
        except Exception as e:
            st.warning(f"Error resampling data for timeframe {tf}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_pivots(df, tf="D", length=5):
        df_htf = PivotIndicator.resample_ohlc(df, tf)
        if df_htf.empty or len(df_htf) < length * 2 + 1:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        highs = df_htf['high']
        lows = df_htf['low']
        ph_mask = PivotIndicator.pivot_high(highs, length, length)
        pl_mask = PivotIndicator.pivot_low(lows, length, length)
        pivot_highs = highs[ph_mask].dropna()
        pivot_lows = lows[pl_mask].dropna()
        return pivot_highs, pivot_lows

    @staticmethod
    def get_all_pivots(df, pivot_settings):
        configs = [
            ("3", 3, "#00ff88", "3M", pivot_settings.get('show_3m', True)),
            ("5", 4, "#ff9900", "5M", pivot_settings.get('show_5m', True)),
            ("10", 4, "#ff44ff", "10M", pivot_settings.get('show_10m', True)),
            ("15", 4, "#4444ff", "15M", pivot_settings.get('show_15m', True)),
            ("60", 5, "#ff0000", "1H", pivot_settings.get('show_1h', True)),
        ]
        all_pivots = []
        for tf, length, color, label, enabled in configs:
            if not enabled:
                continue
            try:
                ph, pl = PivotIndicator.get_pivots(df, tf, length)
                for timestamp, value in ph.items():
                    all_pivots.append({
                        'type': 'high',
                        'timeframe': label,
                        'timestamp': timestamp,
                        'value': value,
                        'color': color
                    })
                for timestamp, value in pl.items():
                    all_pivots.append({
                        'type': 'low',
                        'timeframe': label,
                        'timestamp': timestamp,
                        'value': value,
                        'color': color
                    })
            except Exception as e:
                st.warning(f"Error calculating pivots for {tf}: {str(e)}")
                continue
        return all_pivots

class VolumeOrderBlocks:
    def __init__(self, sensitivity=5):
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.max_blocks = 15

    @staticmethod
    def calculate_ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_atr(df, period=200):
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def detect_blocks(self, df):
        if df.empty or len(df) < self.length2 + 10:
            return {'bullish': [], 'bearish': []}
        df = df.copy().reset_index(drop=True)
        ema_fast = self.calculate_ema(df['close'], self.length1)
        ema_slow = self.calculate_ema(df['close'], self.length2)
        atr = self.calculate_atr(df)
        max_atr = atr.rolling(window=200, min_periods=1).max()
        atr_threshold = max_atr * 2
        overlap_threshold = max_atr * 3
        cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        bullish_blocks = []
        bearish_blocks = []
        for idx in df[cross_up].index:
            if idx < self.length2:
                continue
            lookback_start = max(0, idx - self.length2)
            lookback_df = df.loc[lookback_start:idx]
            lowest_idx = lookback_df['low'].idxmin()
            lowest = df.loc[lowest_idx, 'low']
            vol = df.loc[lowest_idx:idx, 'volume'].sum()
            upper = min(df.loc[lowest_idx, 'open'], df.loc[lowest_idx, 'close'])
            if idx < len(atr_threshold) and not pd.isna(atr_threshold.iloc[idx]):
                min_size = atr_threshold.iloc[idx] * 0.5
                if (upper - lowest) < min_size:
                    upper = lowest + min_size
            mid = (upper + lowest) / 2
            bullish_blocks.append({
                'index': lowest_idx,
                'datetime': df.loc[lowest_idx, 'datetime'] if 'datetime' in df.columns else None,
                'upper': upper,
                'lower': lowest,
                'mid': mid,
                'volume': vol,
                'type': 'bullish'
            })
        for idx in df[cross_down].index:
            if idx < self.length2:
                continue
            lookback_start = max(0, idx - self.length2)
            lookback_df = df.loc[lookback_start:idx]
            highest_idx = lookback_df['high'].idxmax()
            highest = df.loc[highest_idx, 'high']
            vol = df.loc[highest_idx:idx, 'volume'].sum()
            lower = max(df.loc[highest_idx, 'open'], df.loc[highest_idx, 'close'])
            if idx < len(atr_threshold) and not pd.isna(atr_threshold.iloc[idx]):
                min_size = atr_threshold.iloc[idx] * 0.5
                if (highest - lower) < min_size:
                    lower = highest - min_size
            mid = (highest + lower) / 2
            bearish_blocks.append({
                'index': highest_idx,
                'datetime': df.loc[highest_idx, 'datetime'] if 'datetime' in df.columns else None,
                'upper': highest,
                'lower': lower,
                'mid': mid,
                'volume': vol,
                'type': 'bearish'
            })
        current_close = df['close'].iloc[-1]
        bullish_blocks = [b for b in bullish_blocks if current_close >= b['lower']]
        bearish_blocks = [b for b in bearish_blocks if current_close <= b['upper']]
        bullish_blocks = self._remove_overlaps(bullish_blocks, overlap_threshold.iloc[-1] if len(overlap_threshold) > 0 else 50)
        bearish_blocks = self._remove_overlaps(bearish_blocks, overlap_threshold.iloc[-1] if len(overlap_threshold) > 0 else 50)
        bullish_blocks = bullish_blocks[-self.max_blocks:]
        bearish_blocks = bearish_blocks[-self.max_blocks:]
        total_bull_vol = sum(b['volume'] for b in bullish_blocks) if bullish_blocks else 1
        total_bear_vol = sum(b['volume'] for b in bearish_blocks) if bearish_blocks else 1
        for blocks, total in [(bullish_blocks, total_bull_vol), (bearish_blocks, total_bear_vol)]:
            for b in blocks:
                b['volume_pct'] = (b['volume'] / total * 100) if total > 0 else 0
        return {'bullish': bullish_blocks, 'bearish': bearish_blocks}

    def _remove_overlaps(self, blocks, threshold):
        if len(blocks) < 2:
            return blocks
        blocks = sorted(blocks, key=lambda x: x['mid'])
        filtered = []
        for block in blocks:
            overlap = False
            for existing in filtered:
                if abs(block['mid'] - existing['mid']) < threshold:
                    if block['volume'] > existing['volume']:
                        filtered.remove(existing)
                        filtered.append(block)
                    overlap = True
                    break
            if not overlap:
                filtered.append(block)
        return filtered

    @staticmethod
    def format_volume(vol):
        if vol >= 1_000_000:
            return f"{vol/1_000_000:.1f}M"
        elif vol >= 1_000:
            return f"{vol/1_000:.0f}K"
        else:
            return str(int(vol))

    def get_sr_levels(self, df):
        blocks = self.detect_blocks(df)
        sr_levels = []
        for btype, label in [('bullish', '🟢 VOB Support'), ('bearish', '🔴 VOB Resistance')]:
            for block in blocks[btype]:
                sr_levels.append({
                    'Type': label, 'Level': f"₹{block['mid']:.0f}",
                    'Source': f"Vol: {self.format_volume(block['volume'])} ({block['volume_pct']:.1f}%)",
                    'Strength': 'VOB Zone', 'Signal': f"Range: ₹{block['lower']:.0f} - ₹{block['upper']:.0f}",
                    'upper': block['upper'], 'lower': block['lower'], 'mid': block['mid'],
                    'volume': block['volume'], 'volume_pct': block['volume_pct']
                })
        return sr_levels, blocks

class TriplePOC:
    def __init__(self, period1=10, period2=25, period3=70, bins=25):
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.bins = bins

    def calculate_poc(self, df, period):
        if df.empty or len(df) < period:
            return None
        recent_df = df.tail(period).copy()
        H = recent_df['high'].max()
        L = recent_df['low'].min()
        if H == L:
            return {
                'poc': H,
                'upper_poc': H,
                'lower_poc': L,
                'volume': 0,
                'high': H,
                'low': L
            }
        step = (H - L) / self.bins
        vol_bins = [0.0] * self.bins
        level_mids = []
        for k in range(self.bins):
            l = L + k * step
            mid = l + step / 2
            level_mids.append(mid)
        for _, row in recent_df.iterrows():
            c = row['close']
            v = row.get('volume', 1)
            for k in range(len(level_mids)):
                mid = level_mids[k]
                if abs(c - mid) <= step:
                    vol_bins[k] += v
        max_vol_idx = vol_bins.index(max(vol_bins))
        poc = level_mids[max_vol_idx]
        max_volume = vol_bins[max_vol_idx]
        upper_poc = poc + step * 2
        lower_poc = poc - step * 2
        return {
            'poc': round(poc, 2),
            'upper_poc': round(upper_poc, 2),
            'lower_poc': round(lower_poc, 2),
            'volume': max_volume,
            'high': H,
            'low': L,
            'step': step
        }

    def calculate_all_pocs(self, df):
        poc1 = self.calculate_poc(df, self.period1)
        poc2 = self.calculate_poc(df, self.period2)
        poc3 = self.calculate_poc(df, self.period3)
        return {
            'poc1': poc1,
            'poc2': poc2,
            'poc3': poc3,
            'periods': {
                'poc1': self.period1,
                'poc2': self.period2,
                'poc3': self.period3
            }
        }

    def get_price_position(self, current_price, poc_data):
        if poc_data is None:
            return 'unknown'
        if current_price > poc_data['upper_poc']:
            return 'above'
        elif current_price < poc_data['lower_poc']:
            return 'below'
        else:
            return 'inside'

def compute_vpfr(df, n_bars, n_rows=24, va_pct=70):
    """
    Volume Profile Fixed Range — Python port of Pine Script VPFR.
    Distributes each candle's volume across the price bins it spans (by range overlap),
    finds the POC (max-volume bin), then expands outward to capture va_pct% of volume
    for VAH and VAL.
    Returns dict: {poc, vah, val} or None if insufficient data.
    """
    if df is None or df.empty or len(df) < 3:
        return None
    recent = df.tail(n_bars)
    top = recent['high'].max()
    bot = recent['low'].min()
    if top == bot:
        return {'poc': round(top, 2), 'vah': round(top, 2), 'val': round(bot, 2)}
    step = (top - bot) / n_rows
    bins_lo = [bot + i * step for i in range(n_rows)]
    bins_hi = [bot + (i + 1) * step for i in range(n_rows)]
    vol_bins = [0.0] * n_rows
    for _, row in recent.iterrows():
        h, l = row['high'], row['low']
        v = float(row.get('volume') or 1)
        c_range = h - l
        if c_range <= 0:
            continue
        for i in range(n_rows):
            overlap = min(h, bins_hi[i]) - max(l, bins_lo[i])
            if overlap > 0:
                vol_bins[i] += v * (overlap / c_range)
    poc_idx = vol_bins.index(max(vol_bins))
    poc = (bins_lo[poc_idx] + bins_hi[poc_idx]) / 2
    total = sum(vol_bins)
    target = total * va_pct / 100
    cum = vol_bins[poc_idx]
    lo_i, hi_i = poc_idx, poc_idx
    while cum < target:
        can_lo = lo_i - 1 >= 0
        can_hi = hi_i + 1 < n_rows
        if not can_lo and not can_hi:
            break
        v_lo = vol_bins[lo_i - 1] if can_lo else -1
        v_hi = vol_bins[hi_i + 1] if can_hi else -1
        if v_hi >= v_lo:
            hi_i += 1
            cum += vol_bins[hi_i]
        else:
            lo_i -= 1
            cum += vol_bins[lo_i]
    return {'poc': round(poc, 2), 'vah': round(bins_hi[hi_i], 2), 'val': round(bins_lo[lo_i], 2)}


class FutureSwing:
    def __init__(self, swing_length=30, projection_offset=10, history_samples=5, calc_type='Average'):
        self.swing_length = swing_length
        self.projection_offset = projection_offset
        self.history_samples = history_samples
        self.calc_type = calc_type

    def detect_swings(self, df):
        if df.empty or len(df) < self.swing_length + 1:
            return None
        df = df.copy().reset_index(drop=True)
        swing_highs = []
        swing_lows = []
        df['rolling_high'] = df['high'].rolling(window=self.swing_length, min_periods=1).max()
        df['rolling_low'] = df['low'].rolling(window=self.swing_length, min_periods=1).min()
        for i in range(self.swing_length, len(df) - 1):
            if df.loc[i, 'high'] == df.loc[i, 'rolling_high']:
                if i + 1 < len(df) and df.loc[i + 1, 'high'] < df.loc[i, 'rolling_high']:
                    swing_highs.append({
                        'index': i,
                        'value': df.loc[i, 'high'],
                        'datetime': df.loc[i, 'datetime'] if 'datetime' in df.columns else None
                    })
            if df.loc[i, 'low'] == df.loc[i, 'rolling_low']:
                if i + 1 < len(df) and df.loc[i + 1, 'low'] > df.loc[i, 'rolling_low']:
                    swing_lows.append({
                        'index': i,
                        'value': df.loc[i, 'low'],
                        'datetime': df.loc[i, 'datetime'] if 'datetime' in df.columns else None
                    })
        last_high_idx = swing_highs[-1]['index'] if swing_highs else 0
        last_low_idx = swing_lows[-1]['index'] if swing_lows else 0
        direction = 'bearish' if last_high_idx > last_low_idx else 'bullish'
        return {
            'swing_highs': swing_highs[-self.history_samples:] if swing_highs else [],
            'swing_lows': swing_lows[-self.history_samples:] if swing_lows else [],
            'direction': direction,
            'last_swing_high': swing_highs[-1] if swing_highs else None,
            'last_swing_low': swing_lows[-1] if swing_lows else None
        }

    def calculate_swing_percentages(self, swing_data):
        if swing_data is None:
            return []
        swing_highs = swing_data['swing_highs']
        swing_lows = swing_data['swing_lows']
        if not swing_highs or not swing_lows:
            return []
        percentages = []
        all_swings = []
        for sh in swing_highs:
            all_swings.append({'type': 'high', **sh})
        for sl in swing_lows:
            all_swings.append({'type': 'low', **sl})
        all_swings.sort(key=lambda x: x['index'])
        for i in range(1, len(all_swings)):
            prev = all_swings[i - 1]
            curr = all_swings[i]
            if prev['type'] == 'low' and curr['type'] == 'high':
                pct = (curr['value'] - prev['value']) / prev['value'] * 100
                percentages.append(pct)
            elif prev['type'] == 'high' and curr['type'] == 'low':
                pct = (curr['value'] - prev['value']) / prev['value'] * 100
                percentages.append(pct)
        return percentages[-self.history_samples:]

    def project_future_swing(self, swing_data, percentages):
        if not percentages or swing_data is None:
            return None
        abs_percentages = [abs(p) for p in percentages]
        if self.calc_type == 'Average':
            swing_val = sum(abs_percentages) / len(abs_percentages)
        elif self.calc_type == 'Median':
            sorted_pct = sorted(abs_percentages)
            mid = len(sorted_pct) // 2
            swing_val = sorted_pct[mid] if len(sorted_pct) % 2 == 1 else (sorted_pct[mid-1] + sorted_pct[mid]) / 2
        else:
            from collections import Counter
            rounded = [round(p, 1) for p in abs_percentages]
            counter = Counter(rounded)
            swing_val = counter.most_common(1)[0][0]
        direction = swing_data['direction']
        last_high = swing_data['last_swing_high']
        last_low = swing_data['last_swing_low']
        if direction == 'bearish' and last_high:
            target = last_high['value'] - (last_high['value'] * (swing_val / 100))
            return {
                'direction': 'bearish',
                'from_value': last_high['value'],
                'target': round(target, 2),
                'swing_pct': round(swing_val, 2),
                'sign': '-'
            }
        elif direction == 'bullish' and last_low:
            target = last_low['value'] + (last_low['value'] * (swing_val / 100))
            return {
                'direction': 'bullish',
                'from_value': last_low['value'],
                'target': round(target, 2),
                'swing_pct': round(swing_val, 2),
                'sign': '+'
            }
        return None

    def calculate_volume_delta(self, df, swing_data):
        if df.empty or swing_data is None:
            return {'buy_volume': 0, 'sell_volume': 0, 'delta': 0, 'total': 0}
        last_high = swing_data['last_swing_high']
        last_low = swing_data['last_swing_low']
        if not last_high or not last_low:
            return {'buy_volume': 0, 'sell_volume': 0, 'delta': 0, 'total': 0}
        start_idx = min(last_high['index'], last_low['index'])
        df = df.copy().reset_index(drop=True)
        recent_df = df.iloc[start_idx:]
        buy_volume = 0
        sell_volume = 0
        for _, row in recent_df.iterrows():
            v = row.get('volume', 0)
            if row['close'] > row['open']:
                buy_volume += v
            else:
                sell_volume += v
        return {
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'delta': buy_volume - sell_volume,
            'total': buy_volume + sell_volume
        }

    def analyze(self, df):
        swing_data = self.detect_swings(df)
        if swing_data is None:
            return None
        percentages = self.calculate_swing_percentages(swing_data)
        projection = self.project_future_swing(swing_data, percentages)
        volume_delta = self.calculate_volume_delta(df, swing_data)
        return {
            'swings': swing_data,
            'percentages': percentages,
            'projection': projection,
            'volume': volume_delta,
            'settings': {
                'swing_length': self.swing_length,
                'history_samples': self.history_samples,
                'calc_type': self.calc_type
            }
        }

class ReversalDetector:
    @staticmethod
    def calculate_vwap(df):
        if df.empty or 'volume' not in df.columns:
            return pd.Series(dtype=float)
        df = df.copy()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        df['cumulative_tp_vol'] = df['tp_volume'].cumsum()
        df['cumulative_vol'] = df['volume'].cumsum()
        df['vwap'] = df['cumulative_tp_vol'] / df['cumulative_vol']
        return df['vwap']

    @staticmethod
    def detect_higher_low(df, lookback=5):
        if len(df) < lookback + 1:
            return False, None, None
        recent = df.tail(lookback + 1)
        lows = recent['low'].values
        prev_min_idx = lows[:-1].argmin()
        prev_min = lows[prev_min_idx]
        current_low = lows[-1]
        if len(lows) >= 3:
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    swing_low = lows[i]
                    if current_low > swing_low:
                        return True, swing_low, current_low
        return current_low > prev_min, prev_min, current_low

    @staticmethod
    def detect_no_new_low(df, lookback=10):
        if len(df) < lookback:
            return False, None
        recent = df.tail(lookback)
        lows = recent['low'].values
        min_idx = lows.argmin()
        selling_exhausted = min_idx < lookback - 2
        return selling_exhausted, lows.min()

    @staticmethod
    def detect_strong_bullish_candle(df, threshold=0.5):
        if len(df) < 2:
            return False, {}
        current = df.iloc[-1]
        previous = df.iloc[-2]
        is_green = current['close'] > current['open']
        body = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        body_ratio = body / total_range if total_range > 0 else 0
        strong_body = body_ratio >= threshold
        closes_above_prev_high = current['close'] > previous['high']
        is_strong = is_green and strong_body and closes_above_prev_high
        details = {
            'is_green': is_green,
            'body_ratio': round(body_ratio, 2),
            'strong_body': strong_body,
            'closes_above_prev_high': closes_above_prev_high,
            'current_close': current['close'],
            'prev_high': previous['high']
        }
        return is_strong, details

    @staticmethod
    def detect_volume_confirmation(df, lookback=5):
        if len(df) < lookback:
            return False, "Insufficient Data", {}
        current = df.iloc[-1]
        avg_volume = df.tail(lookback)['volume'].mean()
        is_up_candle = current['close'] > current['open']
        current_volume = current['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        if is_up_candle:
            if volume_ratio >= 1.2:
                signal = "Strong Buying"
                confirmed = True
            elif volume_ratio >= 0.8:
                signal = "Normal Buying"
                confirmed = True
            else:
                signal = "Weak/Fake Bounce"
                confirmed = False
        else:
            signal = "Down Candle"
            confirmed = False
        details = {
            'current_volume': current_volume,
            'avg_volume': round(avg_volume, 0),
            'volume_ratio': round(volume_ratio, 2),
            'is_up_candle': is_up_candle
        }
        return confirmed, signal, details

    @staticmethod
    def check_vwap_position(df):
        if len(df) < 2:
            return False, None, None
        vwap = ReversalDetector.calculate_vwap(df)
        if vwap.empty:
            return False, None, None
        current_price = df.iloc[-1]['close']
        current_vwap = vwap.iloc[-1]
        above_vwap = current_price > current_vwap
        return above_vwap, current_price, current_vwap

    @staticmethod
    def detect_support_respect(df, pivot_lows, proximity_pct=0.3):
        if len(df) < 3 or not pivot_lows:
            return False, None, None
        current_low = df.iloc[-1]['low']
        recent_low = df.tail(5)['low'].min()
        nearest_support = None
        min_distance = float('inf')
        for support in pivot_lows:
            distance = abs(recent_low - support)
            pct_distance = (distance / support) * 100 if support > 0 else float('inf')
            if pct_distance < min_distance and pct_distance <= proximity_pct:
                min_distance = pct_distance
                nearest_support = support
        if nearest_support:
            bounced = df.iloc[-1]['close'] > recent_low
            return bounced, nearest_support, recent_low
        return False, None, recent_low

    @staticmethod
    def calculate_reversal_score(df, pivot_lows=None, lookback=10):
        signals = {}
        score = 0
        no_new_low, swing_low = ReversalDetector.detect_no_new_low(df, lookback)
        signals['Selling_Exhausted'] = "Yes ✅" if no_new_low else "No ❌"
        if no_new_low:
            score += 1
        higher_low, prev_low, curr_low = ReversalDetector.detect_higher_low(df, lookback // 2)
        signals['Higher_Low'] = "Yes ✅" if higher_low else "No ❌"
        if higher_low:
            score += 1.5
        strong_candle, candle_details = ReversalDetector.detect_strong_bullish_candle(df)
        signals['Strong_Bullish_Candle'] = "Yes ✅" if strong_candle else "No ❌"
        if strong_candle:
            score += 1.5
        vol_confirmed, vol_signal, vol_details = ReversalDetector.detect_volume_confirmation(df)
        signals['Volume_Signal'] = vol_signal
        if vol_confirmed:
            score += 1
        elif vol_signal == "Weak/Fake Bounce":
            score -= 0.5
        above_vwap, price, vwap = ReversalDetector.check_vwap_position(df)
        signals['Above_VWAP'] = "Yes ✅" if above_vwap else "No ❌"
        if above_vwap:
            score += 1
        if pivot_lows:
            support_held, support_level, low = ReversalDetector.detect_support_respect(df, pivot_lows)
            signals['Support_Respected'] = "Yes ✅" if support_held else "No ❌"
            if support_held:
                score += 1
                signals['Support_Level'] = support_level
        signals['Reversal_Score'] = round(score, 1)
        if score >= 4:
            verdict = "🟢 STRONG BUY SIGNAL"
            entry_type = "Safe CE Entry"
        elif score >= 2.5:
            verdict = "🟡 MODERATE BUY SIGNAL"
            entry_type = "Wait for Confirmation"
        elif score >= 1:
            verdict = "⚪ WEAK SIGNAL"
            entry_type = "No Entry"
        elif score <= -2:
            verdict = "🔴 BEARISH - AVOID CE"
            entry_type = "Consider PE"
        else:
            verdict = "⚪ NEUTRAL"
            entry_type = "No Trade"
        signals['Verdict'] = verdict
        signals['Entry_Type'] = entry_type
        if len(df) > 0:
            signals['Current_Price'] = df.iloc[-1]['close']
            signals['Day_Low'] = df['low'].min()
            signals['Day_High'] = df['high'].max()
            if vwap:
                signals['VWAP'] = round(vwap, 2)
        return score, signals, verdict

    @staticmethod
    def get_entry_rules(signals, score):
        rules = []
        if signals.get('Strong_Bullish_Candle') == "Yes ✅":
            if signals.get('Higher_Low') != "Yes ✅":
                rules.append("⚠️ First green candle - Wait for higher low confirmation")
            else:
                rules.append("✅ Structure confirmed - Entry possible")
        vol_signal = signals.get('Volume_Signal', '')
        if 'Weak' in vol_signal or 'Fake' in vol_signal:
            rules.append("⚠️ Low volume - Possible fake bounce")
        elif 'Strong' in vol_signal:
            rules.append("✅ Strong volume - Real buying detected")
        if signals.get('Above_VWAP') == "Yes ✅":
            rules.append("✅ Price above VWAP - Bullish bias")
        else:
            rules.append("⚠️ Price below VWAP - Wait for VWAP reclaim")
        if score >= 4:
            rules.append("🎯 ENTRY: Buy CE at current level")
            rules.append(f"🛑 SL: Below higher low ({signals.get('Day_Low', 'N/A')})")
            rules.append("🎯 Target: Previous high / Nearest resistance")
        elif score >= 2.5:
            rules.append("⏳ WAIT: Confirmation pending")
            rules.append("📋 Checklist: Higher Low + Strong Candle + Volume")
        else:
            rules.append("❌ NO ENTRY: Conditions not met")
        return rules

    @staticmethod
    def detect_lower_high(df, lookback=5):
        if len(df) < lookback + 1:
            return False, None, None
        recent = df.tail(lookback + 1)
        highs = recent['high'].values
        prev_max_idx = highs[:-1].argmax()
        prev_max = highs[prev_max_idx]
        current_high = highs[-1]
        if len(highs) >= 3:
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    swing_high = highs[i]
                    if current_high < swing_high:
                        return True, swing_high, current_high
        return current_high < prev_max, prev_max, current_high

    @staticmethod
    def detect_no_new_high(df, lookback=10):
        if len(df) < lookback:
            return False, None
        recent = df.tail(lookback)
        highs = recent['high'].values
        max_idx = highs.argmax()
        buying_exhausted = max_idx < lookback - 2
        return buying_exhausted, highs.max()

    @staticmethod
    def detect_strong_bearish_candle(df, threshold=0.5):
        if len(df) < 2:
            return False, {}
        current = df.iloc[-1]
        previous = df.iloc[-2]
        is_red = current['close'] < current['open']
        body = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        body_ratio = body / total_range if total_range > 0 else 0
        strong_body = body_ratio >= threshold
        closes_below_prev_low = current['close'] < previous['low']
        is_strong = is_red and strong_body and closes_below_prev_low
        details = {
            'is_red': is_red,
            'body_ratio': round(body_ratio, 2),
            'strong_body': strong_body,
            'closes_below_prev_low': closes_below_prev_low,
            'current_close': current['close'],
            'prev_low': previous['low']
        }
        return is_strong, details

    @staticmethod
    def calculate_bearish_reversal_score(df, pivot_highs=None, lookback=10):
        signals = {}
        score = 0
        no_new_high, swing_high = ReversalDetector.detect_no_new_high(df, lookback)
        signals['Buying_Exhausted'] = "Yes ✅" if no_new_high else "No ❌"
        if no_new_high:
            score -= 1
        lower_high, prev_high, curr_high = ReversalDetector.detect_lower_high(df, lookback // 2)
        signals['Lower_High'] = "Yes ✅" if lower_high else "No ❌"
        if lower_high:
            score -= 1.5
        strong_candle, candle_details = ReversalDetector.detect_strong_bearish_candle(df)
        signals['Strong_Bearish_Candle'] = "Yes ✅" if strong_candle else "No ❌"
        if strong_candle:
            score -= 1.5
        vol_confirmed, vol_signal, vol_details = ReversalDetector.detect_volume_confirmation(df)
        current = df.iloc[-1]
        is_down = current['close'] < current['open']
        if is_down and vol_details.get('volume_ratio', 0) >= 1.2:
            signals['Volume_Signal'] = "Strong Selling"
            score -= 1
        elif is_down and vol_details.get('volume_ratio', 0) >= 0.8:
            signals['Volume_Signal'] = "Normal Selling"
            score -= 0.5
        else:
            signals['Volume_Signal'] = vol_signal
        above_vwap, price, vwap = ReversalDetector.check_vwap_position(df)
        signals['Below_VWAP'] = "Yes ✅" if not above_vwap else "No ❌"
        if not above_vwap:
            score -= 1
        if pivot_highs:
            recent_high = df.tail(5)['high'].max()
            nearest_resistance = None
            for resistance in pivot_highs:
                pct_distance = abs(recent_high - resistance) / resistance * 100 if resistance > 0 else float('inf')
                if pct_distance <= 0.3:
                    nearest_resistance = resistance
                    break
            if nearest_resistance:
                rejected = df.iloc[-1]['close'] < recent_high
                signals['Resistance_Rejected'] = "Yes ✅" if rejected else "No ❌"
                if rejected:
                    score -= 1
                    signals['Resistance_Level'] = nearest_resistance
            else:
                signals['Resistance_Rejected'] = "N/A"
        else:
            signals['Resistance_Rejected'] = "N/A"
        signals['Bearish_Score'] = round(score, 1)
        if score <= -4:
            verdict = "🔴 STRONG SELL SIGNAL"
            entry_type = "Safe PE Entry"
        elif score <= -2.5:
            verdict = "🟠 MODERATE SELL SIGNAL"
            entry_type = "Wait for Confirmation"
        elif score <= -1:
            verdict = "⚪ WEAK BEARISH"
            entry_type = "No Entry"
        else:
            verdict = "⚪ NEUTRAL"
            entry_type = "No Trade"
        signals['Bearish_Verdict'] = verdict
        signals['Bearish_Entry_Type'] = entry_type
        if len(df) > 0:
            signals['Current_Price'] = df.iloc[-1]['close']
            signals['Day_High'] = df['high'].max()
            if vwap:
                signals['VWAP'] = round(vwap, 2)
        return score, signals, verdict

def calculate_max_pain(df_options, spot_price):
    if df_options.empty:
        return None, None

    strikes = df_options['Strike'].unique()
    pain_data = []

    for strike in strikes:
        ce_pain = 0
        pe_pain = 0
        for _, row in df_options.iterrows():
            k = row['Strike']
            ce_oi = row.get('openInterest_CE', 0) or 0
            pe_oi = row.get('openInterest_PE', 0) or 0
            # CE is ITM when expiry (strike) > option strike k
            if strike > k:
                ce_pain += (strike - k) * ce_oi
            # PE is ITM when expiry (strike) < option strike k
            if strike < k:
                pe_pain += (k - strike) * pe_oi
        total_pain = ce_pain + pe_pain
        pain_data.append({
            'Strike': strike,
            'CE_Pain': ce_pain,
            'PE_Pain': pe_pain,
            'Total_Pain': total_pain
        })

    pain_df = pd.DataFrame(pain_data)

    if pain_df.empty:
        return None, None

    # Max pain = strike where total ITM payout is minimum (MM pay least)
    max_pain_idx = pain_df['Total_Pain'].idxmin()
    max_pain_strike = pain_df.loc[max_pain_idx, 'Strike']

    return max_pain_strike, pain_df

def calculate_dealer_gex(df_summary, spot_price, contract_multiplier=25):
    if df_summary is None or df_summary.empty:
        return None

    try:
        gex_data = []
        for _, row in df_summary.iterrows():
            strike = row.get('Strike', 0)
            gamma_ce = row.get('Gamma_CE', 0) or 0
            gamma_pe = row.get('Gamma_PE', 0) or 0
            oi_ce = row.get('openInterest_CE', 0) or 0
            oi_pe = row.get('openInterest_PE', 0) or 0
            call_gex = -1 * gamma_ce * oi_ce * contract_multiplier * spot_price / 100000
            put_gex = gamma_pe * oi_pe * contract_multiplier * spot_price / 100000
            net_gex = call_gex + put_gex
            gex_data.append({
                'Strike': strike,
                'Call_GEX': round(call_gex, 2),
                'Put_GEX': round(put_gex, 2),
                'Net_GEX': round(net_gex, 2),
                'Zone': row.get('Zone', '-')
            })
        gex_df = pd.DataFrame(gex_data)
        total_gex = gex_df['Net_GEX'].sum()
        gex_df_sorted = gex_df.sort_values('Strike')
        gamma_flip_level = None
        gamma_flip_direction = None
        for i in range(len(gex_df_sorted) - 1):
            current_gex = gex_df_sorted.iloc[i]['Net_GEX']
            next_gex = gex_df_sorted.iloc[i + 1]['Net_GEX']
            current_strike = gex_df_sorted.iloc[i]['Strike']
            next_strike = gex_df_sorted.iloc[i + 1]['Strike']
            if current_gex * next_gex < 0:
                gamma_flip_level = current_strike + (next_strike - current_strike) * abs(current_gex) / (abs(current_gex) + abs(next_gex))
                gamma_flip_direction = "Positive above" if current_gex < 0 else "Negative above"
                break
        if total_gex > 50:
            gex_interpretation = "STRONG PIN - Dealers long gamma, price likely to revert/chop"
            gex_signal = "Pin/Chop"
            gex_color = "#00ff88"
        elif total_gex > 0:
            gex_interpretation = "MILD PIN - Slight mean reversion tendency"
            gex_signal = "Range"
            gex_color = "#90EE90"
        elif total_gex > -50:
            gex_interpretation = "MILD TREND - Slight directional bias possible"
            gex_signal = "Trending"
            gex_color = "#FFD700"
        else:
            gex_interpretation = "STRONG TREND - Dealers short gamma, violent moves possible"
            gex_signal = "Breakout"
            gex_color = "#ff4444"
        max_positive_idx = gex_df['Net_GEX'].idxmax()
        max_negative_idx = gex_df['Net_GEX'].idxmin()
        gex_magnet = gex_df.loc[max_positive_idx, 'Strike'] if gex_df.loc[max_positive_idx, 'Net_GEX'] > 0 else None
        gex_repeller = gex_df.loc[max_negative_idx, 'Strike'] if gex_df.loc[max_negative_idx, 'Net_GEX'] < 0 else None
        return {
            'gex_df': gex_df,
            'total_gex': round(total_gex, 2),
            'gamma_flip_level': round(gamma_flip_level, 2) if gamma_flip_level else None,
            'gamma_flip_direction': gamma_flip_direction,
            'gex_interpretation': gex_interpretation,
            'gex_signal': gex_signal,
            'gex_color': gex_color,
            'gex_magnet': gex_magnet,
            'gex_repeller': gex_repeller,
            'spot_vs_flip': "Above Gamma Flip" if gamma_flip_level and spot_price > gamma_flip_level else "Below Gamma Flip" if gamma_flip_level else "N/A"
        }

    except Exception as e:
        return None


def check_gex_alert(gex_data, df_summary, underlying_price):
    if gex_data is None or 'gex_history' not in st.session_state:
        return

    try:
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        gex_entry = {
            'time': current_time,
            'total_gex': gex_data['total_gex'],
            'gamma_flip': gex_data['gamma_flip_level'],
            'spot': underlying_price,
            'signal': gex_data['gex_signal']
        }
        should_add = True
        if st.session_state.gex_history:
            last_entry = st.session_state.gex_history[-1]
            time_diff = (current_time - last_entry['time']).total_seconds()
            if time_diff < 30:
                should_add = False
        if should_add:
            st.session_state.gex_history.append(gex_entry)
            if len(st.session_state.gex_history) > 100:
                st.session_state.gex_history = st.session_state.gex_history[-100:]
        if len(st.session_state.gex_history) < 2:
            return
        prev_entry = st.session_state.gex_history[-2]
        delta_gex = gex_data['total_gex'] - prev_entry['total_gex']
        gex_pct_change = abs(delta_gex / prev_entry['total_gex'] * 100) if prev_entry['total_gex'] != 0 else 0
        alert_triggered = False
        alert_type = None
        alert_message = None
        if prev_entry['total_gex'] * gex_data['total_gex'] < 0:
            alert_triggered = True
            alert_type = "GEX SIGN FLIP"
            flip_direction = "Positive → Negative" if prev_entry['total_gex'] > 0 else "Negative → Positive"
            alert_message = f"""
🔄 <b>GEX SIGN FLIP ALERT</b> 🔄

📊 <b>Gamma Exposure Flipped:</b> {flip_direction}
📍 <b>Spot Price:</b> ₹{underlying_price:.2f}

<b>Previous GEX:</b> {prev_entry['total_gex']:.2f}L ({prev_entry['signal']})
<b>Current GEX:</b> {gex_data['total_gex']:.2f}L ({gex_data['gex_signal']})
<b>ΔGEX:</b> {delta_gex:+.2f}L

<b>🎯 Market Implication:</b>
{gex_data['gex_interpretation']}

⚡ <b>ACTION:</b> {'Expect acceleration/trend moves!' if gex_data['total_gex'] < 0 else 'Expect mean reversion/pin!'}

🕐 Time: {current_time.strftime('%H:%M:%S IST')}
"""

        elif gex_pct_change > 30:
            alert_triggered = True
            alert_type = "LARGE ΔGEX"
            alert_message = f"""
⚡ <b>LARGE ΔGEX ALERT</b> ⚡

📊 <b>Gamma Exposure Changed Significantly!</b>
📍 <b>Spot Price:</b> ₹{underlying_price:.2f}

<b>Previous GEX:</b> {prev_entry['total_gex']:.2f}L
<b>Current GEX:</b> {gex_data['total_gex']:.2f}L
<b>ΔGEX:</b> {delta_gex:+.2f}L ({gex_pct_change:.1f}%)

<b>🎯 Market Regime:</b> {gex_data['gex_signal']}
{gex_data['gex_interpretation']}

🕐 Time: {current_time.strftime('%H:%M:%S IST')}
"""

        elif gex_data['gamma_flip_level'] and prev_entry.get('gamma_flip'):
            prev_above_flip = prev_entry['spot'] > prev_entry['gamma_flip']
            curr_above_flip = underlying_price > gex_data['gamma_flip_level']
            if prev_above_flip != curr_above_flip:
                alert_triggered = True
                alert_type = "GAMMA FLIP CROSSED"
                cross_direction = "Crossed ABOVE" if curr_above_flip else "Crossed BELOW"
                alert_message = f"""
🎯 <b>GAMMA FLIP LEVEL CROSSED</b> 🎯

📍 <b>Spot Price:</b> ₹{underlying_price:.2f}
📊 <b>Gamma Flip Level:</b> ₹{gex_data['gamma_flip_level']:.2f}
🔀 <b>Direction:</b> {cross_direction}

<b>Current GEX:</b> {gex_data['total_gex']:.2f}L ({gex_data['gex_signal']})

<b>🎯 Implication:</b>
{'Above flip = More pinning/mean reversion' if curr_above_flip else 'Below flip = More trending/acceleration'}

🕐 Time: {current_time.strftime('%H:%M:%S IST')}
"""

        if alert_triggered:
            alert_key = f"{alert_type}_{current_time.strftime('%Y%m%d_%H%M')}"
            if st.session_state.last_gex_alert != alert_key:
                st.session_state.last_gex_alert = alert_key

    except Exception as e:
        pass


def calculate_pcr_gex_confluence(pcr_value, gex_data, zone='ATM'):
    if gex_data is None:
        return "⚪ N/A", "No GEX Data", 0

    net_gex = gex_data.get('total_gex', 0)
    gex_signal = gex_data.get('gex_signal', 'Unknown')

    if pcr_value > 1.2:
        pcr_signal = "Bullish"
    elif pcr_value < 0.7:
        pcr_signal = "Bearish"
    else:
        pcr_signal = "Neutral"

    gex_negative = net_gex < -10
    gex_positive = net_gex > 10

    if pcr_signal == "Bullish" and gex_negative:
        return "🟢🔥 STRONG BULL", "Bullish + Breakout", 3

    elif pcr_signal == "Bearish" and gex_positive:
        return "🔴🔥 STRONG BEAR", "Bearish + Pin", 3

    elif pcr_signal == "Bullish" and gex_positive:
        return "🟢📍 BULL RANGE", "Bullish + Chop", 2

    elif pcr_signal == "Bearish" and gex_negative:
        return "🔴⚡ BEAR TREND", "Bearish + Accel", 2

    elif pcr_signal == "Bullish":
        return "🟢 BULLISH", "Bullish PCR", 1

    elif pcr_signal == "Bearish":
        return "🔴 BEARISH", "Bearish PCR", 1

    else:
        return "⚪ NEUTRAL", "Mixed Signals", 0


def calculate_exact_time_to_expiry(expiry_date_str):
    try:
        expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").replace(hour=15, minute=30)
        expiry_date = expiry_date.replace(tzinfo=pytz.timezone('Asia/Kolkata'))
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        time_diff = expiry_date - now
        total_seconds = time_diff.total_seconds()
        total_days = total_seconds / (24 * 3600)
        years = total_days / 365.25
        return max(years, 1/365.25)
    except:
        return 1/365.25

def get_iv_fallback(df, strike_price):
    try:
        nearby_strikes = df[abs(df['strikePrice'] - strike_price) <= 100]
        if not nearby_strikes.empty:
            iv_ce_avg = nearby_strikes['impliedVolatility_CE'].mean()
            iv_pe_avg = nearby_strikes['impliedVolatility_PE'].mean()
            if pd.isna(iv_ce_avg):
                iv_ce_avg = df['impliedVolatility_CE'].mean()
            if pd.isna(iv_pe_avg):
                iv_pe_avg = df['impliedVolatility_PE'].mean()
            return iv_ce_avg or 15, iv_pe_avg or 15
        else:
            return 15, 15
    except:
        return 15, 15


def calculate_greeks(option_type, S, K, T, r, sigma):
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'CE' else (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
        return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)
    except:
        return 0, 0, 0, 0, 0


def final_verdict(score):
    if score >= 4:
        return "Strong Bullish"
    elif score >= 2:
        return "Bullish"
    elif score <= -4:
        return "Strong Bearish"
    elif score <= -2:
        return "Bearish"
    else:
        return "Neutral"

def delta_volume_bias(price, volume, chg_oi):
    if price > 0 and volume > 0 and chg_oi > 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi > 0:
        return "Bearish"
    elif price > 0 and volume > 0 and chg_oi < 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi < 0:
        return "Bearish"
    else:
        return "Neutral"


def calculate_bid_ask_pressure(call_bid_qty, call_ask_qty, put_bid_qty, put_ask_qty):
    pressure = (call_bid_qty - call_ask_qty) + (put_ask_qty - put_bid_qty)
    if pressure > 500:
        bias = "Bullish"
    elif pressure < -500:
        bias = "Bearish"
    else:
        bias = "Neutral"
    return pressure, bias

weights = {
    "LTP_Bias": 1,
    "OI_Bias": 2,
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "Delta_Bias": 1,
    "Gamma_Bias": 1,
    "Theta_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "AskBid_Bias": 1,
    "IV_Bias": 1,
    "DVP_Bias": 1,
    "PressureBias": 1,
}


def determine_level(row):
    ce_oi = row.get('openInterest_CE', 0)
    pe_oi = row.get('openInterest_PE', 0)
    if pe_oi > 1.12 * ce_oi:
        return "Support"
    elif ce_oi > 1.12 * pe_oi:
        return "Resistance"
    else:
        return "Neutral"


_CG = 'background-color: #90EE90; color: black'
_CR = 'background-color: #FFB6C1; color: black'
_CY = 'background-color: #FFFFE0; color: black'
_CDG = 'background-color: #228B22; color: white'
_CDR = 'background-color: #DC143C; color: white'
_CF = 'background-color: #F5F5F5; color: black'

def color_pressure(val):
    return _CG if val > 500 else (_CR if val < -500 else _CY)

def color_pcr(val):
    return _CG if val > 1.2 else (_CR if val < 0.7 else _CY)

def color_bias(val):
    return _CG if val == "Bullish" else (_CR if val == "Bearish" else _CY)

def color_verdict(val):
    v = str(val)
    if "Strong Bullish" in v: return _CDG
    if "Bullish" in v: return _CG
    if "Strong Bearish" in v: return _CDR
    if "Bearish" in v: return _CR
    return _CY

def color_entry(val):
    v = str(val)
    return _CG if "Bull" in v else (_CR if "Bear" in v else _CF)

def color_fakereal(val):
    v = str(val)
    if "Real Up" in v: return _CDG
    if "Fake Up" in v: return 'background-color: #98FB98; color: black'
    if "Real Down" in v: return _CDR
    if "Fake Down" in v: return 'background-color: #FFC0CB; color: black'
    return _CF

def color_score(val):
    try:
        s = float(val)
        if s >= 4: return _CDG
        if s >= 2: return _CG
        if s <= -4: return _CDR
        if s <= -2: return _CR
        return _CY
    except: return ''

def highlight_atm_row(row):
    return [''] * len(row)


def process_candle_data(data, interval):
    if not data or 'open' not in data:
        return pd.DataFrame()

    df = pd.DataFrame({
        'timestamp': data['timestamp'],
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume']
    })

    ist = pytz.timezone('Asia/Kolkata')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist)

    return df


def calculate_pcr_sr_level(pcr, reference_strike):
    """PCR-based Support/Resistance level.
    PCR ≤ 0.7 → Resistance offset below/above strike.
    0.8–1.7   → Neutral.
    PCR ≥ 1.8 → Support offset below/above strike.
    """
    pcr = round(pcr, 2)
    if pcr <= 0.7:
        # Map PCR → offset using linear interpolation between anchor points
        anchors = [(0.3, -20), (0.4, -10), (0.5, 0), (0.6, 10), (0.7, 20)]
        if pcr <= 0.3:
            offset = -20
        elif pcr >= 0.7:
            offset = 20
        else:
            offset = -20
            for i in range(len(anchors) - 1):
                p0, o0 = anchors[i]; p1, o1 = anchors[i + 1]
                if p0 <= pcr <= p1:
                    offset = o0 + (pcr - p0) / (p1 - p0) * (o1 - o0)
                    break
        level = reference_strike + offset
        interpretation = (
            f"Resistance at ₹{level:.0f} (below ATM — strong cap)"  if offset < 0 else
            f"Resistance at ₹{level:.0f} (above ATM — moderate cap)" if offset > 0 else
            f"Resistance at ATM ₹{level:.0f}"
        )
        return {'type': 'Resistance 🔴', 'level': level, 'offset': offset, 'interpretation': interpretation}

    elif pcr <= 1.7:  # 0.71–1.7 all neutral
        return {'type': 'Neutral ⚪', 'level': reference_strike, 'offset': 0,
                'interpretation': f"Neutral — no clear S/R offset (PCR {pcr:.2f})"}

    else:  # pcr >= 1.8
        anchors = [(1.8, -20), (2.0, -10), (2.5, 0), (3.0, 10), (3.5, 20)]
        if pcr <= 1.8:
            offset = -20
        elif pcr >= 3.5:
            offset = 20
        else:
            offset = -20
            for i in range(len(anchors) - 1):
                p0, o0 = anchors[i]; p1, o1 = anchors[i + 1]
                if p0 <= pcr <= p1:
                    offset = o0 + (pcr - p0) / (p1 - p0) * (o1 - o0)
                    break
        level = reference_strike + offset
        interpretation = (
            f"Support at ₹{level:.0f} (below ATM — strong floor)" if offset < 0 else
            f"Support at ₹{level:.0f} (above ATM — elevated floor)" if offset > 0 else
            f"Support at ATM ₹{level:.0f}"
        )
        return {'type': 'Support 🟢', 'level': level, 'offset': offset, 'interpretation': interpretation}



def detect_candle_patterns(df, lookback=5):
    """Detect candlestick patterns from last few candles using Nifty price action chart."""
    if df is None or len(df) < lookback:
        return {'pattern': 'Insufficient Data', 'direction': 'Neutral', 'details': {}, 'candles': []}
    recent = df.tail(lookback).copy()
    last = recent.iloc[-1]
    prev = recent.iloc[-2] if len(recent) >= 2 else None
    prev2 = recent.iloc[-3] if len(recent) >= 3 else None

    # Analyze each of the last candles
    candle_list = []
    for idx in range(len(recent)):
        c = recent.iloc[idx]
        c_body = abs(c['close'] - c['open'])
        c_range = c['high'] - c['low']
        c_body_ratio = c_body / c_range if c_range > 0 else 0
        c_green = c['close'] > c['open']
        c_upper = c['high'] - max(c['close'], c['open'])
        c_lower = min(c['close'], c['open']) - c['low']
        c_prev = recent.iloc[idx - 1] if idx > 0 else None
        c_prev2 = recent.iloc[idx - 2] if idx > 1 else None

        c_pattern = 'Normal'
        # Check multi-candle patterns FIRST (higher significance)
        # 3-candle patterns
        if c_prev is not None and c_prev2 is not None:
            p_body = abs(c_prev['close'] - c_prev['open'])
            p_green = c_prev['close'] > c_prev['open']
            p_range = c_prev['high'] - c_prev['low']
            p_body_ratio = p_body / p_range if p_range > 0 else 0
            p2_body = abs(c_prev2['close'] - c_prev2['open'])
            p2_green = c_prev2['close'] > c_prev2['open']
            p2_range = c_prev2['high'] - c_prev2['low']
            if not p2_green and (p2_body / p2_range > 0.5 if p2_range > 0 else False) and p_body_ratio < 0.3 and c_green and c_body_ratio > 0.5 and c['close'] > (c_prev2['open'] + c_prev2['close']) / 2:
                c_pattern = 'Morning Star'
            elif p2_green and (p2_body / p2_range > 0.5 if p2_range > 0 else False) and p_body_ratio < 0.3 and not c_green and c_body_ratio > 0.5 and c['close'] < (c_prev2['open'] + c_prev2['close']) / 2:
                c_pattern = 'Evening Star'
            elif p2_green and p_green and c_green and c_prev['close'] > c_prev2['close'] and c['close'] > c_prev['close'] and (p2_body / p2_range > 0.5 if p2_range > 0 else False) and p_body_ratio > 0.5 and c_body_ratio > 0.5:
                c_pattern = 'Three White Soldiers'
            elif not p2_green and not p_green and not c_green and c_prev['close'] < c_prev2['close'] and c['close'] < c_prev['close'] and (p2_body / p2_range > 0.5 if p2_range > 0 else False) and p_body_ratio > 0.5 and c_body_ratio > 0.5:
                c_pattern = 'Three Black Crows'
        # 2-candle patterns
        if c_pattern == 'Normal' and c_prev is not None:
            p_body = abs(c_prev['close'] - c_prev['open'])
            p_green = c_prev['close'] > c_prev['open']
            p_range = c_prev['high'] - c_prev['low']
            if c_green and not p_green and c_body > p_body and c['close'] > c_prev['open'] and c['open'] < c_prev['close']:
                c_pattern = 'Bullish Engulfing'
            elif not c_green and p_green and c_body > p_body and c['close'] < c_prev['open'] and c['open'] > c_prev['close']:
                c_pattern = 'Bearish Engulfing'
            elif c_body < p_body * 0.6 and not p_green and c_green and min(c['open'], c['close']) > min(c_prev['open'], c_prev['close']) and max(c['open'], c['close']) < max(c_prev['open'], c_prev['close']):
                c_pattern = 'Bullish Harami'
            elif c_body < p_body * 0.6 and p_green and not c_green and min(c['open'], c['close']) > min(c_prev['open'], c_prev['close']) and max(c['open'], c['close']) < max(c_prev['open'], c_prev['close']):
                c_pattern = 'Bearish Harami'
            elif c_green and not p_green and c['open'] < c_prev['low'] and c['close'] > (c_prev['open'] + c_prev['close']) / 2 and c['close'] < c_prev['open']:
                c_pattern = 'Piercing Line'
            elif not c_green and p_green and c['open'] > c_prev['high'] and c['close'] < (c_prev['open'] + c_prev['close']) / 2 and c['close'] > c_prev['open']:
                c_pattern = 'Dark Cloud Cover'
            elif c_green and not p_green and abs(c['low'] - c_prev['low']) / max(c_range, 0.01) < 0.05:
                c_pattern = 'Tweezer Bottom'
            elif not c_green and p_green and abs(c['high'] - c_prev['high']) / max(p_range, 0.01) < 0.05:
                c_pattern = 'Tweezer Top'
        # 1-candle patterns (lowest priority)
        if c_pattern == 'Normal':
            if c_lower > c_body * 2 and c_upper < c_body * 0.5 and c_body_ratio < 0.4:
                c_pattern = 'Hammer'
            elif c_upper > c_body * 2 and c_lower < c_body * 0.5 and c_body_ratio < 0.4:
                c_pattern = 'Shooting Star' if not c_green else 'Inverted Hammer'
            elif c_body_ratio >= 0.95 and c_range > 0:
                c_pattern = 'Bull Marubozu' if c_green else 'Bear Marubozu'
            elif c_body_ratio < 0.1 and c_range > 0:
                c_pattern = 'Doji'
            elif c_body_ratio < 0.35 and c_upper > c_body and c_lower > c_body and c_range > 0:
                c_pattern = 'Spinning Top'

        candle_list.append({
            'open': round(c['open'], 2), 'high': round(c['high'], 2),
            'low': round(c['low'], 2), 'close': round(c['close'], 2),
            'type': 'Bull' if c_green else 'Bear',
            'pattern': c_pattern,
            'body_ratio': round(c_body_ratio, 2),
            'volume': int(c.get('volume', 0)),
            'time': c.get('datetime', '').strftime('%H:%M') if hasattr(c.get('datetime', ''), 'strftime') else str(c.get('datetime', '')),
        })

    # Overall pattern from last candle
    body = abs(last['close'] - last['open'])
    total_range = last['high'] - last['low']
    body_ratio = body / total_range if total_range > 0 else 0
    is_green = last['close'] > last['open']
    upper_wick = last['high'] - max(last['close'], last['open'])
    lower_wick = min(last['close'], last['open']) - last['low']

    pattern = 'No Pattern'
    direction = 'Neutral'

    # Check multi-candle patterns FIRST (higher significance)
    # 3-candle patterns
    if prev is not None and prev2 is not None:
        prev_body = abs(prev['close'] - prev['open'])
        prev_green = prev['close'] > prev['open']
        prev_range = prev['high'] - prev['low']
        prev_body_ratio = prev_body / prev_range if prev_range > 0 else 0
        p2_body = abs(prev2['close'] - prev2['open'])
        p2_green = prev2['close'] > prev2['open']
        p2_range = prev2['high'] - prev2['low']
        if not p2_green and (p2_body / p2_range > 0.5 if p2_range > 0 else False) and prev_body_ratio < 0.3 and is_green and body_ratio > 0.5:
            if last['close'] > (prev2['open'] + prev2['close']) / 2:
                pattern, direction = 'Morning Star', 'Bullish'
        if pattern == 'No Pattern' and p2_green and (p2_body / p2_range > 0.5 if p2_range > 0 else False) and prev_body_ratio < 0.3 and not is_green and body_ratio > 0.5:
            if last['close'] < (prev2['open'] + prev2['close']) / 2:
                pattern, direction = 'Evening Star', 'Bearish'
        if pattern == 'No Pattern' and p2_green and prev_green and is_green and prev['close'] > prev2['close'] and last['close'] > prev['close']:
            if (p2_body / p2_range > 0.5 if p2_range > 0 else False) and prev_body_ratio > 0.5 and body_ratio > 0.5:
                pattern, direction = 'Three White Soldiers', 'Bullish'
        if pattern == 'No Pattern' and not p2_green and not prev_green and not is_green and prev['close'] < prev2['close'] and last['close'] < prev['close']:
            if (p2_body / p2_range > 0.5 if p2_range > 0 else False) and prev_body_ratio > 0.5 and body_ratio > 0.5:
                pattern, direction = 'Three Black Crows', 'Bearish'

    # 2-candle patterns
    if pattern == 'No Pattern' and prev is not None:
        prev_body = abs(prev['close'] - prev['open'])
        prev_green = prev['close'] > prev['open']
        prev_range = prev['high'] - prev['low']
        if is_green and not prev_green and body > prev_body and last['close'] > prev['open'] and last['open'] < prev['close']:
            pattern, direction = 'Bullish Engulfing', 'Bullish'
        elif not is_green and prev_green and body > prev_body and last['close'] < prev['open'] and last['open'] > prev['close']:
            pattern, direction = 'Bearish Engulfing', 'Bearish'
        elif body < prev_body * 0.6 and not prev_green and is_green and min(last['open'], last['close']) > min(prev['open'], prev['close']) and max(last['open'], last['close']) < max(prev['open'], prev['close']):
            pattern, direction = 'Bullish Harami', 'Bullish'
        elif body < prev_body * 0.6 and prev_green and not is_green and min(last['open'], last['close']) > min(prev['open'], prev['close']) and max(last['open'], last['close']) < max(prev['open'], prev['close']):
            pattern, direction = 'Bearish Harami', 'Bearish'
        elif is_green and not prev_green and last['open'] < prev['low'] and last['close'] > (prev['open'] + prev['close']) / 2 and last['close'] < prev['open']:
            pattern, direction = 'Piercing Line', 'Bullish'
        elif not is_green and prev_green and last['open'] > prev['high'] and last['close'] < (prev['open'] + prev['close']) / 2 and last['close'] > prev['open']:
            pattern, direction = 'Dark Cloud Cover', 'Bearish'
        elif is_green and not prev_green and abs(last['low'] - prev['low']) / max(total_range, 0.01) < 0.05:
            pattern, direction = 'Tweezer Bottom', 'Bullish'
        elif not is_green and prev_green and abs(last['high'] - prev['high']) / max(prev_range, 0.01) < 0.05:
            pattern, direction = 'Tweezer Top', 'Bearish'

    # 1-candle patterns (lowest priority)
    if pattern == 'No Pattern':
        if lower_wick > body * 2 and upper_wick < body * 0.5 and body_ratio < 0.4:
            pattern, direction = 'Hammer', 'Bullish'
        elif upper_wick > body * 2 and lower_wick < body * 0.5 and body_ratio < 0.4 and is_green:
            pattern, direction = 'Inverted Hammer', 'Bullish'
        elif upper_wick > body * 2 and lower_wick < body * 0.5 and body_ratio < 0.4 and not is_green:
            pattern, direction = 'Shooting Star', 'Bearish'
        elif body_ratio >= 0.95 and total_range > 0:
            pattern = 'Bull Marubozu' if is_green else 'Bear Marubozu'
            direction = 'Bullish' if is_green else 'Bearish'
        elif body_ratio < 0.1 and total_range > 0:
            pattern, direction = 'Doji', 'Indecision'
        elif body_ratio < 0.35 and upper_wick > body and lower_wick > body and total_range > 0:
            pattern, direction = 'Spinning Top', 'Indecision'

    if pattern == 'No Pattern' and body_ratio >= 0.6:
        pattern = 'Strong Green Candle' if is_green else 'Strong Red Candle'
        direction = 'Bullish' if is_green else 'Bearish'

    # Count bull/bear candles in last 5
    bull_count = sum(1 for c in candle_list if c['type'] == 'Bull')
    bear_count = sum(1 for c in candle_list if c['type'] == 'Bear')

    return {
        'pattern': pattern, 'direction': direction,
        'candles': candle_list,
        'bull_count': bull_count, 'bear_count': bear_count,
        'details': {
            'body_ratio': round(body_ratio, 2), 'is_green': is_green,
            'close': last['close'], 'open': last['open'],
            'high': last['high'], 'low': last['low'],
        }
    }

def detect_order_blocks(df, lookback=20):
    """LuxAlgo Order Block Detector (Python port).

    Bullish OB: volume pivot high during upswing → demand zone (support)
    Bearish OB: volume pivot high during downswing → supply zone (resistance)
    Mitigation: OB is removed when price wicks through the zone bottom/top.
    Returns up to 3 active (non-mitigated) OBs of each type, most recent first.
    """
    length = 5
    ext_last = 3
    if df is None or len(df) < length * 2 + 2:
        return {'bullish_obs': [], 'bearish_obs': [],
                'bullish_ob': None, 'bearish_ob': None}

    df2 = df.reset_index(drop=True).copy()
    n = len(df2)
    h  = df2['high'].values.astype(float)
    lo = df2['low'].values.astype(float)
    c  = df2['close'].values.astype(float)
    v  = df2['volume'].values.astype(float)
    hl2 = (h + lo) / 2.0

    # ── Oscillator: tracks whether we're in an upswing (os=1) or downswing (os=0) ──
    # os=0 when high[length] bars ago > highest(high, length) now → was higher → downswing
    # os=1 when low[length] bars ago  < lowest(low,  length) now → was lower  → upswing
    os = np.zeros(n, dtype=int)
    for i in range(length, n):
        win_h = h[i - length + 1: i + 1]
        win_l = lo[i - length + 1: i + 1]
        if len(win_h) == 0:
            os[i] = os[i - 1]
            continue
        upper = np.max(win_h)
        lower = np.min(win_l)
        if h[i - length] > upper:
            os[i] = 0
        elif lo[i - length] < lower:
            os[i] = 1
        else:
            os[i] = os[i - 1] if i > 0 else 0

    # ── Volume pivot high: volume[pivot_idx] is max in window [pivot_idx-length .. pivot_idx+length] ──
    bull_obs, bear_obs = [], []
    for i in range(length * 2, n):
        pivot_idx = i - length
        v_win = v[max(0, pivot_idx - length): min(n, pivot_idx + length + 1)]
        if len(v_win) == 0 or v[pivot_idx] == 0:
            continue
        if v[pivot_idx] < np.max(v_win):
            continue
        # Volume pivot confirmed — check direction at bar i
        bar_time = df2.iloc[pivot_idx].get('datetime', pivot_idx)
        if os[i] == 1:     # upswing → bullish OB (demand zone below)
            bull_obs.append({'low': lo[pivot_idx], 'high': hl2[pivot_idx],
                             'avg': (lo[pivot_idx] + hl2[pivot_idx]) / 2,
                             'bar_idx': pivot_idx, 'time': bar_time, 'type': 'bullish'})
        elif os[i] == 0:   # downswing → bearish OB (supply zone above)
            bear_obs.append({'low': hl2[pivot_idx], 'high': h[pivot_idx],
                             'avg': (hl2[pivot_idx] + h[pivot_idx]) / 2,
                             'bar_idx': pivot_idx, 'time': bar_time, 'type': 'bearish'})

    # ── Mitigation: remove OBs where price has already wicked through ──
    target_bull = np.min(lo[-length:]) if len(lo) >= length else lo[-1]   # lowest wick
    target_bear = np.max(h[-length:])  if len(h)  >= length else h[-1]    # highest wick
    active_bull = [ob for ob in bull_obs if target_bull >= ob['low']]     # not yet broken below
    active_bear = [ob for ob in bear_obs if target_bear <= ob['high']]    # not yet broken above

    # Keep most recent ext_last
    active_bull = sorted(active_bull, key=lambda x: x['bar_idx'], reverse=True)[:ext_last]
    active_bear = sorted(active_bear, key=lambda x: x['bar_idx'], reverse=True)[:ext_last]

    # Backward-compat: expose closest single OB as bullish_ob / bearish_ob
    closest_bull = active_bull[0] if active_bull else None
    closest_bear = active_bear[0] if active_bear else None

    return {
        'bullish_obs':  active_bull,
        'bearish_obs':  active_bear,
        'bullish_ob':   closest_bull,
        'bearish_ob':   closest_bear,
    }

def detect_volume_spike(df, lookback=5):
    """Check if current candle has volume spike vs recent average."""
    if df is None or len(df) < lookback + 1:
        return {'spike': False, 'ratio': 0, 'label': 'Insufficient Data'}
    current_vol = df.iloc[-1]['volume']
    avg_vol = df.tail(lookback + 1).iloc[:-1]['volume'].mean()
    ratio = current_vol / avg_vol if avg_vol > 0 else 0
    if ratio >= 2.0:
        label = 'HIGH (Spike)'
    elif ratio >= 1.3:
        label = 'Above Avg'
    else:
        label = 'Normal'
    return {'spike': ratio >= 1.5, 'ratio': round(ratio, 2), 'label': label}

def get_candle_location(price, support_levels, resistance_levels, gex_data, ob_data):
    """Determine where the current candle is relative to key levels."""
    locations = []
    # Check near support
    for s in support_levels:
        if abs(price - s) / price * 100 < 0.15:
            locations.append(f"Near Support ₹{s:.0f}")
    # Check near resistance
    for r in resistance_levels:
        if abs(price - r) / price * 100 < 0.15:
            locations.append(f"Near Resistance ₹{r:.0f}")
    # Check GEX levels
    if gex_data:
        if gex_data.get('gex_magnet') and abs(price - gex_data['gex_magnet']) / price * 100 < 0.2:
            locations.append(f"Near GEX Magnet ₹{gex_data['gex_magnet']:.0f}")
        if gex_data.get('gex_repeller') and abs(price - gex_data['gex_repeller']) / price * 100 < 0.2:
            locations.append(f"Near GEX Repeller ₹{gex_data['gex_repeller']:.0f}")
        if gex_data.get('gamma_flip_level') and abs(price - gex_data['gamma_flip_level']) / price * 100 < 0.15:
            locations.append(f"Near Gamma Flip ₹{gex_data['gamma_flip_level']:.0f}")
    # Check order blocks
    if ob_data:
        if ob_data.get('bullish_ob') and ob_data['bullish_ob']['low'] <= price <= ob_data['bullish_ob']['high']:
            locations.append("Inside Bullish OB")
        if ob_data.get('bearish_ob') and ob_data['bearish_ob']['low'] <= price <= ob_data['bearish_ob']['high']:
            locations.append("Inside Bearish OB")
    return locations if locations else ["Middle (No key level)"]

def calculate_vidya(df, length=10, momentum=20, band_distance=2.0):
    """Calculate VIDYA indicator with trend detection (ported from Pine Script)."""
    if df is None or df.empty or len(df) < momentum + 15:
        return {'trend': 'Unknown', 'cross_up': False, 'cross_down': False,
                'buy_vol': 0, 'sell_vol': 0, 'delta_pct': 0, 'smoothed_last': 0}
    src = df['close'].values.astype(float)
    opens = df['open'].values.astype(float)
    n = len(src)
    alpha = 2 / (length + 1)
    v = np.zeros(n)
    v[0] = src[0]
    for i in range(1, n):
        start = max(0, i - momentum + 1)
        changes = np.diff(src[start:i+1])
        if len(changes) == 0:
            v[i] = v[i-1]
            continue
        pos_sum = float(np.sum(changes[changes >= 0]))
        neg_sum = float(np.sum(-changes[changes < 0]))
        total = pos_sum + neg_sum
        abs_cmo = abs(100 * (pos_sum - neg_sum) / total) if total > 0 else 0
        v[i] = alpha * abs_cmo / 100 * src[i] + (1 - alpha * abs_cmo / 100) * v[i-1]
    vidya_smooth = pd.Series(v).rolling(15, min_periods=1).mean().values
    prev_close = np.roll(src, 1)
    prev_close[0] = src[0]
    tr = np.maximum(df['high'].values.astype(float) - df['low'].values.astype(float),
                    np.maximum(np.abs(df['high'].values.astype(float) - prev_close),
                              np.abs(df['low'].values.astype(float) - prev_close)))
    atr = pd.Series(tr).rolling(200, min_periods=1).mean().values
    upper = vidya_smooth + atr * band_distance
    lower = vidya_smooth - atr * band_distance
    is_up = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if src[i] > upper[i]:
            is_up[i] = True
        elif src[i] < lower[i]:
            is_up[i] = False
        else:
            is_up[i] = is_up[i-1]
    smoothed = np.where(is_up, lower, upper)
    cross_up = bool(not is_up[-2] and is_up[-1]) if n > 1 else False
    cross_down = bool(is_up[-2] and not is_up[-1]) if n > 1 else False
    # Delta volume since last trend cross
    buy_vol, sell_vol = 0.0, 0.0
    vol = df['volume'].values.astype(float)
    last_cross = 0
    for i in range(n - 1, 0, -1):
        if is_up[i] != is_up[i - 1]:
            last_cross = i
            break
    for i in range(last_cross, n):
        if src[i] > opens[i]:
            buy_vol += vol[i]
        elif src[i] < opens[i]:
            sell_vol += vol[i]
    avg = (buy_vol + sell_vol) / 2 if (buy_vol + sell_vol) > 0 else 1
    delta_pct = (buy_vol - sell_vol) / avg * 100
    return {
        'trend': 'Bullish' if is_up[-1] else 'Bearish',
        'smoothed_last': round(float(smoothed[-1]), 2),
        'cross_up': cross_up, 'cross_down': cross_down,
        'buy_vol': buy_vol, 'sell_vol': sell_vol,
        'delta_pct': round(delta_pct, 1),
    }

def calculate_htf_sr(df):
    """Calculate Higher Timeframe Support/Resistance from price action pivots.
    Resamples 1-min data to 15m, 1h, 4h and finds pivot highs/lows."""
    if df is None or df.empty or len(df) < 60:
        return {'levels': [], 'support': [], 'resistance': []}
    levels = []
    timeframes = [('15m', '15min', 4), ('1h', '1h', 5), ('4h', '4h', 5)]
    for tf_label, resample_rule, pivot_len in timeframes:
        try:
            ohlcv = df.set_index('datetime').resample(resample_rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            if len(ohlcv) < pivot_len * 2 + 1:
                continue
            highs = ohlcv['high'].values.astype(float)
            lows = ohlcv['low'].values.astype(float)
            for i in range(pivot_len, len(highs) - pivot_len):
                if all(highs[i] >= highs[i - j] and highs[i] >= highs[i + j] for j in range(1, pivot_len + 1)):
                    levels.append({'type': 'Resistance', 'level': float(highs[i]), 'tf': tf_label})
            for i in range(pivot_len, len(lows) - pivot_len):
                if all(lows[i] <= lows[i - j] and lows[i] <= lows[i + j] for j in range(1, pivot_len + 1)):
                    levels.append({'type': 'Support', 'level': float(lows[i]), 'tf': tf_label})
        except Exception:
            continue
    # Deduplicate nearby levels (within 0.05%)
    if levels:
        levels.sort(key=lambda x: x['level'])
        filtered = [levels[0]]
        tf_rank = {'4h': 3, '1h': 2, '15m': 1}
        for lvl in levels[1:]:
            if abs(lvl['level'] - filtered[-1]['level']) / max(filtered[-1]['level'], 1) * 100 > 0.05:
                filtered.append(lvl)
            elif tf_rank.get(lvl['tf'], 0) > tf_rank.get(filtered[-1]['tf'], 0):
                filtered[-1] = lvl
        levels = filtered
    support = [l for l in levels if l['type'] == 'Support']
    resistance = [l for l in levels if l['type'] == 'Resistance']
    return {'levels': levels, 'support': support, 'resistance': resistance}

def calculate_delta_volume(df):
    """Calculate delta volume (buy vs sell) per bar and cumulative."""
    if df is None or df.empty:
        return None
    rdf = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    rdf['is_green'] = rdf['close'] > rdf['open']
    rdf['buy_vol'] = np.where(rdf['is_green'], rdf['volume'], 0)
    rdf['sell_vol'] = np.where(~rdf['is_green'], rdf['volume'], 0)
    rdf['delta'] = rdf['buy_vol'] - rdf['sell_vol']
    rdf['cum_delta'] = rdf['delta'].cumsum()
    rdf['delta_ma'] = rdf['delta'].rolling(10, min_periods=1).mean()
    delta_std = rdf['delta'].rolling(20, min_periods=5).std().fillna(0)
    rdf['spike_up'] = rdf['delta'] > rdf['delta_ma'] + 1.5 * delta_std
    rdf['spike_down'] = rdf['delta'] < rdf['delta_ma'] - 1.5 * delta_std
    return rdf

def detect_hvp(df, left_bars=15, right_bars=15, vol_filter=2.0):
    """Detect High Volume Pivot points."""
    if df is None or df.empty or len(df) < left_bars + right_bars + 1:
        return {'bullish_hvp': [], 'bearish_hvp': []}
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    volumes = df['volume'].values.astype(float)
    times = df['datetime'].tolist()
    bullish_hvp, bearish_hvp = [], []
    for i in range(left_bars, len(df) - right_bars):
        vol_sum = np.sum(volumes[max(0, i - left_bars):i + right_bars + 1])
        vol_avg = np.mean(volumes[max(0, i - 50):i + 1]) * (left_bars * 2) if i >= 5 else vol_sum
        is_high_vol = vol_sum > vol_avg * vol_filter
        if not is_high_vol:
            continue
        is_ph = all(highs[i] >= highs[i - j] and highs[i] >= highs[i + j] for j in range(1, min(left_bars, right_bars) + 1))
        is_pl = all(lows[i] <= lows[i - j] and lows[i] <= lows[i + j] for j in range(1, min(left_bars, right_bars) + 1))
        if is_ph:
            bearish_hvp.append({'price': float(highs[i]), 'time': times[i], 'volume': float(vol_sum)})
        if is_pl:
            bullish_hvp.append({'price': float(lows[i]), 'time': times[i], 'volume': float(vol_sum)})
    return {'bullish_hvp': bullish_hvp[-5:], 'bearish_hvp': bearish_hvp[-5:]}

def detect_ltp_trap(df, delta_length=10, delta_thresh=1.5):
    """Detect LTP Trap signals (VWAP + delta based)."""
    if df is None or df.empty or len(df) < delta_length + 5:
        return {'buy_trap': False, 'sell_trap': False, 'vwap': 0, 'delta_ma': 0, 'price_vs_vwap': 'N/A'}
    tp = (df['high'] + df['low'] + df['close']) / 3
    cum_tp_vol = (tp * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum()
    vwap = float((cum_tp_vol / cum_vol).iloc[-1])
    delta = df['close'] - df['open']
    delta_ma = float(delta.ewm(span=delta_length, adjust=False).mean().iloc[-1])
    last_close, last_open = float(df['close'].iloc[-1]), float(df['open'].iloc[-1])
    buy_trap = last_close < last_open and last_close > vwap and delta_ma > delta_thresh
    sell_trap = last_close > last_open and last_close < vwap and delta_ma < -delta_thresh
    return {
        'buy_trap': buy_trap, 'sell_trap': sell_trap,
        'vwap': round(vwap, 2), 'delta_ma': round(delta_ma, 4),
        'price_vs_vwap': 'Above' if last_close > vwap else 'Below',
    }


