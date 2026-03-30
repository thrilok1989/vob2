"""
Volume Delta Candles - Python port of LuxAlgo Volume Delta Candles
Calculates intrabar volume delta (buying vs selling pressure)
using bar polarity and buying/selling pressure estimation.
"""
import numpy as np
import pandas as pd


def calculate_volume_delta(df):
    """
    Calculate volume delta for each candle using buying/selling pressure estimation.

    Since we don't have tick-level or LTF data, we estimate:
    - Buy volume = volume * (close - low) / (high - low)  (buying pressure proportion)
    - Sell volume = volume * (high - close) / (high - low) (selling pressure proportion)

    Args:
        df: DataFrame with open, high, low, close, volume columns

    Returns:
        DataFrame with delta columns added, plus summary dict
    """
    if df.empty or len(df) < 2:
        return None

    # Ensure required columns exist
    required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            return None

    result = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()

    h = result['high'].values.astype(float)
    l = result['low'].values.astype(float)
    o = result['open'].values.astype(float)
    c = result['close'].values.astype(float)
    v = result['volume'].values.astype(float)

    # If all volumes are 0 (e.g. cached data), use bar range as synthetic volume
    if np.sum(v) == 0:
        v = np.abs(h - l) + 1
        result['volume'] = v

    bar_range = h - l
    bar_range = np.where(bar_range == 0, 1, bar_range)  # avoid division by zero

    # Buying pressure proportion: how much of bar range is close-to-low
    buy_ratio = (c - l) / bar_range
    sell_ratio = (h - c) / bar_range

    buy_vol = v * buy_ratio
    sell_vol = v * sell_ratio

    delta = buy_vol - sell_vol
    delta_pct = np.where(v > 0, delta / v * 100, 0)

    # Cumulative delta
    cum_delta = np.cumsum(delta)

    # Max volume price tracking (approximate)
    max_vol_price = np.where(buy_vol > sell_vol, c, o)

    result['buy_volume'] = np.round(buy_vol, 0).astype(int)
    result['sell_volume'] = np.round(sell_vol, 0).astype(int)
    result['delta'] = np.round(delta, 0).astype(int)
    result['delta_pct'] = np.round(delta_pct, 2)
    result['cum_delta'] = np.round(cum_delta, 0).astype(int)
    result['max_vol_price'] = np.round(max_vol_price, 2)
    result['is_positive_delta'] = delta > 0
    result['bar_type'] = np.where(c > o, 'bullish', np.where(c < o, 'bearish', 'neutral'))

    # Delta divergence: bullish candle but negative delta (or vice versa)
    result['divergence'] = (
        ((c > o) & (delta < 0)) |  # bullish candle, negative delta
        ((c < o) & (delta > 0))    # bearish candle, positive delta
    )

    # Summary stats
    total_buy = int(buy_vol.sum())
    total_sell = int(sell_vol.sum())
    total_delta = total_buy - total_sell
    avg_delta_pct = float(np.mean(delta_pct))

    # Streak tracking
    pos_streak = 0
    neg_streak = 0
    current_streak = 0
    for d in delta:
        if d > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            pos_streak = max(pos_streak, current_streak)
        elif d < 0:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            neg_streak = max(neg_streak, abs(current_streak))
        else:
            current_streak = 0

    summary = {
        'total_buy_volume': total_buy,
        'total_sell_volume': total_sell,
        'total_delta': total_delta,
        'delta_ratio': round(total_buy / total_sell, 2) if total_sell > 0 else 0,
        'avg_delta_pct': round(avg_delta_pct, 2),
        'positive_bars': int((delta > 0).sum()),
        'negative_bars': int((delta < 0).sum()),
        'divergence_bars': int(result['divergence'].sum()),
        'max_positive_streak': pos_streak,
        'max_negative_streak': neg_streak,
        'bias': 'Bullish' if total_delta > 0 else 'Bearish' if total_delta < 0 else 'Neutral',
        'cum_delta_last': int(cum_delta[-1]) if len(cum_delta) > 0 else 0,
    }

    return {
        'df': result,
        'summary': summary,
    }
