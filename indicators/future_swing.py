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
