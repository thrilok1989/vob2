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
