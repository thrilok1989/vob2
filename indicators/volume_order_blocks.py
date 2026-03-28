import pandas as pd


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
        for btype, label in [('bullish', '\U0001f7e2 VOB Support'), ('bearish', '\U0001f534 VOB Resistance')]:
            for block in blocks[btype]:
                sr_levels.append({
                    'Type': label, 'Level': f"\u20b9{block['mid']:.0f}",
                    'Source': f"Vol: {self.format_volume(block['volume'])} ({block['volume_pct']:.1f}%)",
                    'Strength': 'VOB Zone', 'Signal': f"Range: \u20b9{block['lower']:.0f} - \u20b9{block['upper']:.0f}",
                    'upper': block['upper'], 'lower': block['lower'], 'mid': block['mid'],
                    'volume': block['volume'], 'volume_pct': block['volume_pct']
                })
        return sr_levels, blocks
