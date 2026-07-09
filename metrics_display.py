import streamlit as st
import pandas as pd
import pytz
from datetime import datetime

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

def display_metrics(ltp_data, df, db, symbol="NIFTY50"):
    if not (ltp_data and 'data' in ltp_data and not df.empty):
        return
    current_price = None
    for exchange, data in ltp_data['data'].items():
        for sid, pd_ in data.items():
            current_price = pd_.get('last_price', 0)
            break
    if current_price is None:
        current_price = df['close'].iloc[-1]
    if df.empty or len(df) < 2:
        return
    prev_close = df['close'].iloc[-2]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100
    day_high, day_low, day_open = df['high'].max(), df['low'].min(), df['open'].iloc[0]
    volume = df['volume'].sum()
    avg_price = df['close'].mean()
    db.save_market_analytics(symbol, {
        'day_high': float(day_high), 'day_low': float(day_low), 'day_open': float(day_open),
        'day_close': float(current_price), 'total_volume': int(volume),
        'avg_price': float(avg_price), 'price_change': float(change), 'price_change_pct': float(change_pct)
    })
    sign = "+" if change >= 0 else ""
    color = "price-up" if change >= 0 else "price-down"
    metrics = [
        ("Current Price", f'<h2 class="{color}">₹{current_price:,.2f}</h2>'),
        ("Change", f'<h3 class="{color}">{sign}{change:.2f} ({sign}{change_pct:.2f}%)</h3>'),
        ("Day High", f'<h3>₹{day_high:,.2f}</h3>'),
        ("Day Low", f'<h3>₹{day_low:,.2f}</h3>'),
        ("Volume", f'<h3>{volume:,}</h3>'),
    ]
    for col, (label, val) in zip(st.columns(5), metrics):
        col.markdown(f'<div class="metric-container"><h4>{label}</h4>{val}</div>', unsafe_allow_html=True)
