"""
Money Flow Profile - Python port of LuxAlgo Money Flow Profile
Calculates volume/money flow distribution across price levels (rows/bins)
with sentiment breakdown (bullish vs bearish volume per bin).
"""
import numpy as np
import pandas as pd


def calculate_money_flow_profile(df, num_rows=25, source='Volume', sentiment_method='Bar Polarity'):
    """
    Calculate money flow profile from OHLCV data.

    Args:
        df: DataFrame with open, high, low, close, volume columns
        num_rows: Number of price bins (10-100)
        source: 'Volume' or 'Money Flow'
        sentiment_method: 'Bar Polarity' or 'Buying/Selling Pressure'

    Returns:
        dict with profile data, POC, sentiment, and summary table
    """
    if df.empty or len(df) < 5:
        return None

    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            return None

    price_low = df['low'].min()
    price_high = df['high'].max()
    step = (price_high - price_low) / num_rows

    if step <= 0:
        return None

    # Arrays for total volume, bullish volume per bin
    total_vol = np.zeros(num_rows)
    bull_vol = np.zeros(num_rows)

    opens = df['open'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)
    volumes = df['volume'].values.astype(float)

    # If all volumes are 0 (e.g. cached data), use bar range as synthetic volume
    if np.sum(volumes) == 0:
        volumes = np.abs(highs - lows) + 1

    for i in range(len(df)):
        h, l, o, c, v = highs[i], lows[i], opens[i], closes[i], volumes[i]
        if v <= 0 or h == l:
            continue

        # Determine if bar is bullish
        if sentiment_method == 'Bar Polarity':
            is_bull = c > o
        else:
            is_bull = (c - l) > (h - c)

        bar_range = h - l

        for row in range(num_rows):
            bin_low = price_low + row * step
            bin_high = bin_low + step

            if h >= bin_low and l < bin_high:
                # Proportional volume allocation
                if l >= bin_low and h > bin_high:
                    portion = (bin_high - l) / bar_range
                elif h <= bin_high and l < bin_low:
                    portion = (h - bin_low) / bar_range
                elif l >= bin_low and h <= bin_high:
                    portion = 1.0
                else:
                    portion = step / bar_range

                portion = min(portion, 1.0)

                if source == 'Money Flow':
                    mid_price = price_low + (row + 0.5) * step
                    total_vol[row] += v * portion * mid_price
                    if is_bull:
                        bull_vol[row] += v * portion * mid_price
                else:
                    total_vol[row] += v * portion
                    if is_bull:
                        bull_vol[row] += v * portion

    total_sum = total_vol.sum()
    if total_sum == 0:
        return None

    max_vol = total_vol.max()
    poc_idx = np.argmax(total_vol)
    poc_price = price_low + (poc_idx + 0.5) * step

    # Sentiment delta per bin: 2 * bull - total (positive = bullish dominated)
    bear_vol = total_vol - bull_vol
    sentiment_delta = bull_vol - bear_vol
    sentiment_abs = np.abs(sentiment_delta)
    max_sentiment = sentiment_abs.max() if sentiment_abs.max() > 0 else 1

    # Classify nodes
    HIGH_THRESHOLD = 0.53
    LOW_THRESHOLD = 0.37

    rows_data = []
    for row in range(num_rows):
        bin_low = price_low + row * step
        bin_high = bin_low + step
        bin_mid = price_low + (row + 0.5) * step
        vol = total_vol[row]
        ratio = vol / max_vol if max_vol > 0 else 0
        pct = (vol / total_sum * 100) if total_sum > 0 else 0

        if ratio > HIGH_THRESHOLD:
            node_type = 'High'
        elif ratio < LOW_THRESHOLD:
            node_type = 'Low'
        else:
            node_type = 'Average'

        bull_v = bull_vol[row]
        bear_v = bear_vol[row]
        delta = sentiment_delta[row]
        sentiment = 'Bullish' if delta > 0 else 'Bearish' if delta < 0 else 'Neutral'
        sentiment_strength = abs(delta) / vol * 100 if vol > 0 else 0

        rows_data.append({
            'bin_low': round(bin_low, 2),
            'bin_high': round(bin_high, 2),
            'price_level': round(bin_mid, 2),
            'total_volume': round(vol, 2),
            'bull_volume': round(bull_v, 2),
            'bear_volume': round(bear_v, 2),
            'delta': round(delta, 2),
            'volume_pct': round(pct, 2),
            'ratio': round(ratio, 4),
            'node_type': node_type,
            'sentiment': sentiment,
            'sentiment_strength': round(sentiment_strength, 1),
            'is_poc': row == poc_idx,
        })

    # Highest sentiment zone
    max_sent_idx = np.argmax(sentiment_abs)
    highest_sentiment_price = price_low + (max_sent_idx + 0.5) * step
    highest_sentiment_direction = 'Bullish' if sentiment_delta[max_sent_idx] > 0 else 'Bearish'

    # Value area (consolidation zone) - rows above threshold
    va_threshold = 0.25
    value_area_rows = [r for r in rows_data if r['ratio'] > va_threshold]
    va_high = max(r['bin_high'] for r in value_area_rows) if value_area_rows else price_high
    va_low = min(r['bin_low'] for r in value_area_rows) if value_area_rows else price_low

    return {
        'rows': rows_data,
        'poc_price': round(poc_price, 2),
        'poc_volume': round(max_vol, 2),
        'price_high': round(price_high, 2),
        'price_low': round(price_low, 2),
        'total_volume': round(total_sum, 2),
        'num_rows': num_rows,
        'step': round(step, 2),
        'value_area_high': round(va_high, 2),
        'value_area_low': round(va_low, 2),
        'highest_sentiment_price': round(highest_sentiment_price, 2),
        'highest_sentiment_direction': highest_sentiment_direction,
        'source': source,
        'num_bars': len(df),
    }
