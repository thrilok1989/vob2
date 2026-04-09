import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import time
from datetime import datetime, timedelta
import json
import hashlib
import numpy as np
import math
from scipy.stats import norm
from pytz import timezone
import io
import os
from db.supabase_client import SupabaseDB
from indicators.money_flow_profile import calculate_money_flow_profile
from indicators.volume_delta import calculate_volume_delta

st.set_page_config(
    page_title="Nifty Trading & Options Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Only auto-refresh during market hours (8:30 AM - 3:45 PM IST, weekdays)
_ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
_is_market_open = (
    _ist_now.weekday() < 5 and
    _ist_now.replace(hour=8, minute=30, second=0, microsecond=0) <= _ist_now <= _ist_now.replace(hour=15, minute=45, second=0, microsecond=0)
)
if _is_market_open:
    st_autorefresh(interval=30000, key="datarefresh")

st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stSelectbox > div > div > select {
        background-color:
        color: white;
    }
    .metric-container {
        background-color:
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .price-up {
        color:
    }
    .price-down {
        color:
    }
</style>
""", unsafe_allow_html=True)

try:
    DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "") or st.secrets.get("dhan", {}).get("client_id", "")
    DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "") or st.secrets.get("dhan", {}).get("access_token", "")
    supabase_url = st.secrets.get("supabase", {}).get("url", "")
    supabase_key = st.secrets.get("supabase", {}).get("anon_key", "")
    try:
        TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "") or getattr(st.secrets, "TELEGRAM_BOT_TOKEN", "")
        TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "") or getattr(st.secrets, "TELEGRAM_CHAT_ID", "")
        if isinstance(TELEGRAM_CHAT_ID, (int, float)):
            TELEGRAM_CHAT_ID = str(int(TELEGRAM_CHAT_ID))
    except:
        TELEGRAM_BOT_TOKEN = TELEGRAM_CHAT_ID = ""
except Exception:
    DHAN_CLIENT_ID = DHAN_ACCESS_TOKEN = supabase_url = supabase_key = TELEGRAM_BOT_TOKEN = TELEGRAM_CHAT_ID = ""

NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"

@st.cache_data(ttl=300)
def cached_pivot_calculation(df_json, pivot_settings):
    df = pd.read_json(df_json)
    return PivotIndicator.get_all_pivots(df, pivot_settings)

@st.cache_data(ttl=60)
def cached_iv_average(option_data_json):
    df = pd.read_json(option_data_json)
    iv_ce_avg = df['impliedVolatility_CE'].mean()
    iv_pe_avg = df['impliedVolatility_PE'].mean()
    return iv_ce_avg, iv_pe_avg

def send_telegram_message_sync(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    # Only send during market hours (8:30 AM - 3:45 PM IST, weekdays)
    _now = datetime.now(pytz.timezone('Asia/Kolkata'))
    if _now.weekday() >= 5 or not (_now.replace(hour=8, minute=30, second=0, microsecond=0) <= _now <= _now.replace(hour=15, minute=45, second=0, microsecond=0)):
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Telegram error: {response.status_code}")
    except Exception as e:
        st.error(f"Telegram notification error: {e}")

def test_telegram_connection():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Credentials not configured"

    try:
        test_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            return True, "✅ Telegram bot is active and connected"
        else:
            return False, f"❌ Telegram API error: {response.status_code}"

    except Exception as e:
        return False, f"❌ Telegram connection failed: {str(e)}"


class DhanAPI:
    def __init__(self, access_token, client_id):
        self.access_token = access_token.strip() if access_token else ""
        self.client_id = client_id.strip() if client_id else ""
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id
        }

    def get_intraday_data(self, security_id="13", exchange_segment="IDX_I", instrument="INDEX", interval="1", days_back=1):
        url = f"{self.base_url}/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": False,
            "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    def get_ltp_data(self, security_id="13", exchange_segment="IDX_I"):
        url = f"{self.base_url}/marketfeed/ltp"
        payload = {
            exchange_segment: [int(security_id)]
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"LTP API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching LTP: {str(e)}")
            return None

@st.cache_data(ttl=300)
def get_dhan_expiry_list_cached(underlying_scrip: int, underlying_seg: str):
    return get_dhan_expiry_list(underlying_scrip, underlying_seg)

def _dhan_post(url, payload, max_retries=4):
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    delays = [2, 4, 8, 16]
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 429:
                if attempt < max_retries:
                    d = delays[min(attempt, 3)]
                    st.warning(f"⏳ Rate limited by Dhan API. Retrying in {d}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(d)
                    continue
                st.error("❌ Rate limit exceeded after multiple retries. Please wait a moment and refresh.")
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries and "429" in str(e):
                time.sleep(delays[min(attempt, 3)])
                continue
            st.error(f"Request error: {e}")
            return None
    return None

def get_dhan_option_chain(underlying_scrip: int, underlying_seg: str, expiry: str, max_retries: int = 4):
    return _dhan_post("https://api.dhan.co/v2/optionchain",
                      {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg, "Expiry": expiry},
                      max_retries)

def get_dhan_expiry_list(underlying_scrip: int, underlying_seg: str, max_retries: int = 4):
    return _dhan_post("https://api.dhan.co/v2/optionchain/expirylist",
                      {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg},
                      max_retries)

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
            if strike < k:
                ce_pain += (k - strike) * ce_oi
            if strike > k:
                pe_pain += (strike - k) * pe_oi
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

    max_pain_idx = pain_df['Total_Pain'].idxmax()
    max_pain_strike = pain_df.loc[max_pain_idx, 'Strike']

    return max_pain_strike, pain_df

def check_trading_signals(df, pivot_settings, option_data, current_price, pivot_proximity=5):
    if df.empty or option_data is None or len(option_data) == 0 or not current_price:
        return

    try:
        df_json = df.to_json()
        pivots = cached_pivot_calculation(df_json, pivot_settings)
    except:
        pivots = PivotIndicator.get_all_pivots(df, pivot_settings)

    pivot_lows = [p['value'] for p in pivots if p['type'] == 'low']
    reversal_score, reversal_signals, reversal_verdict = ReversalDetector.calculate_reversal_score(df, pivot_lows)

    near_pivot = False
    pivot_level = None

    for pivot in pivots:
        if pivot['timeframe'] in ['3M', '5M', '10M', '15M']:
            price_diff = current_price - pivot['value']
            if abs(price_diff) <= pivot_proximity:
                near_pivot = True
                pivot_level = pivot
                break

    if near_pivot and len(option_data) > 0:
        atm_data = option_data[option_data['Zone'] == 'ATM']
        if not atm_data.empty:
            row = atm_data.iloc[0]
            bullish_conditions = {
                'Support Level': row.get('Level') == 'Support',
                'ChgOI Bias': row.get('ChgOI_Bias') == 'Bullish',
                'Volume Bias': row.get('Volume_Bias') == 'Bullish',
                'AskQty Bias': row.get('AskQty_Bias') == 'Bullish',
                'BidQty Bias': row.get('BidQty_Bias') == 'Bullish',
                'Pressure Bias': row.get('PressureBias') == 'Bullish',
                'Delta Exposure': row.get('DeltaExp') == 'Bullish',
                'Gamma Exposure': row.get('GammaExp') == 'Bullish'
            }
            bearish_conditions = {
                'Resistance Level': row.get('Level') == 'Resistance',
                'ChgOI Bias': row.get('ChgOI_Bias') == 'Bearish',
                'Volume Bias': row.get('Volume_Bias') == 'Bearish',
                'AskQty Bias': row.get('AskQty_Bias') == 'Bearish',
                'BidQty Bias': row.get('BidQty_Bias') == 'Bearish',
                'Pressure Bias': row.get('PressureBias') == 'Bearish',
                'Delta Exposure': row.get('DeltaExp') == 'Bearish',
                'Gamma Exposure': row.get('GammaExp') == 'Bearish'
            }
            atm_strike = row['Strike']
            stop_loss_percent = 20
            ce_chg_oi = row.get('changeinOpenInterest_CE', 0)
            pe_chg_oi = row.get('changeinOpenInterest_PE', 0)
            bullish_oi_confirm = pe_chg_oi > 1.5 * ce_chg_oi
            bearish_oi_confirm = ce_chg_oi > 1.5 * pe_chg_oi
            reversal_text = f"""
🔄 <b>REVERSAL DETECTOR:</b>
• Score: {reversal_signals.get('Reversal_Score', 0)}/6
• Selling Exhausted: {reversal_signals.get('Selling_Exhausted', 'N/A')}
• Higher Low: {reversal_signals.get('Higher_Low', 'N/A')}
• Strong Candle: {reversal_signals.get('Strong_Bullish_Candle', 'N/A')}
• Volume: {reversal_signals.get('Volume_Signal', 'N/A')}
• Above VWAP: {reversal_signals.get('Above_VWAP', 'N/A')}
• {reversal_verdict}"""

            price_diff = current_price - pivot_level['value']
            near = abs(price_diff) <= pivot_proximity
            time_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')
            pv_tf, pv_val = pivot_level['timeframe'], pivot_level['value']
            for is_bull, conds, oi_confirm in [
                (True, bullish_conditions, bullish_oi_confirm),
                (False, bearish_conditions, bearish_oi_confirm)
            ]:
                if not (all(conds.values()) and near):
                    continue
                emoji = "🚨" if is_bull else "🔴"
                opt = "CE" if is_bull else "PE"
                direction = "CALL" if is_bull else "PUT"
                mark = "✅" if is_bull else "🔴"
                trigger_type = "🔥 OI Dominance Trigger" if oi_confirm else "📊 Normal Bias Trigger"
                conds_text = "\n".join([f"{mark} {k}" for k, v in conds.items() if v])
                oi_text = f"PE ChgOI {pe_chg_oi:,} vs CE ChgOI {ce_chg_oi:,}" if is_bull else f"CE ChgOI {ce_chg_oi:,} vs PE ChgOI {pe_chg_oi:,}"
                message = f"""{emoji} <b>NIFTY {direction} SIGNAL ALERT</b> {emoji}
📍 <b>Spot:</b> ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} Pivot by {price_diff:+.2f}pts)
📌 <b>Near Pivot:</b> {pv_tf} at ₹{pv_val:.2f}
🎯 <b>ATM Strike:</b> {atm_strike}
<b>{mark} {'BULLISH' if is_bull else 'BEARISH'} CONDITIONS MET:</b>
{conds_text}
⚡ <b>{trigger_type}</b>
⚡ <b>OI:</b> {oi_text}
{reversal_text}
📋 <b>REVIEW:</b> {atm_strike} {opt} | SL: {stop_loss_percent}% | Manual verification required
🕐 Time: {time_str}"""
                try:
                    send_telegram_message_sync(message)
                    st.success(f"{'🟢' if is_bull else '🔴'} {'Bullish' if is_bull else 'Bearish'} signal notification sent!")
                except Exception as e:
                    st.error(f"Failed to send notification: {e}")
                break

def check_atm_verdict_alert(df_summary, underlying_price):
    """ATM verdict alert — display only, no Telegram."""
    if df_summary is None or len(df_summary) == 0 or not underlying_price:
        return

    atm_data = df_summary[df_summary['Zone'] == 'ATM']
    if atm_data.empty:
        return

    row = atm_data.iloc[0]
    verdict = row.get('Verdict', 'Neutral')
    atm_strike = row.get('Strike', 0)
    bias_score = row.get('BiasScore', 0)

    if verdict not in ['Strong Bullish', 'Strong Bearish']:
        return

    alert_key = f"atm_verdict_{atm_strike}_{verdict}"
    if 'last_atm_verdict_alert' not in st.session_state:
        st.session_state.last_atm_verdict_alert = None

    if st.session_state.last_atm_verdict_alert == alert_key:
        return

    _g = row.get
    ce_oi, pe_oi = _g('openInterest_CE', 0), _g('openInterest_PE', 0)
    ce_chg_oi, pe_chg_oi = _g('changeinOpenInterest_CE', 0), _g('changeinOpenInterest_PE', 0)
    is_bull = verdict == 'Strong Bullish'
    emoji = "🟢🟢🟢" if is_bull else "🔴🔴🔴"
    direction = "BULLISH" if is_bull else "BEARISH"
    opt = "CE" if is_bull else "PE"
    message = f"""{emoji} <b>ATM STRIKE STRONG {direction} ALERT</b> {emoji}
📍 <b>Spot Price:</b> ₹{underlying_price:.2f}
🎯 <b>ATM Strike:</b> {atm_strike}
📊 <b>Verdict:</b> {verdict} (Score: {bias_score})
<b>📈 BIAS BREAKDOWN:</b>
• OI: {_g('OI_Bias','N/A')} • ChgOI: {_g('ChgOI_Bias','N/A')} • Volume: {_g('Volume_Bias','N/A')}
• Delta Exp: {_g('DeltaExp','N/A')} • Gamma Exp: {_g('GammaExp','N/A')} • Pressure: {_g('PressureBias','N/A')}
<b>📊 OI DATA:</b>
• CE OI: {ce_oi/100000:.1f}L | PE OI: {pe_oi/100000:.1f}L
• CE ΔOI: {ce_chg_oi/1000:.1f}K | PE ΔOI: {pe_chg_oi/1000:.1f}K
<b>⚡ SIGNALS:</b>
• Operator Entry: {_g('Operator_Entry','N/A')} • Scalp/Momentum: {_g('Scalp_Moment','N/A')}
📋 <b>SUGGESTED REVIEW:</b> {atm_strike} {opt} | Manual verification required
🕐 Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}"""
    st.session_state.last_atm_verdict_alert = alert_key

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

def create_candlestick_chart(df, title, interval, show_pivots=True, pivot_settings=None, vob_blocks=None, poc_data=None, swing_data=None, money_flow_data=None, volume_delta_data=None):
    if df.empty:
        return go.Figure()

    # Use 3 rows if volume delta is available (price, delta overlay, volume)
    has_delta = volume_delta_data is not None and volume_delta_data.get('df') is not None
    fig = make_subplots(
        rows=3 if has_delta else 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.2, 0.25] if has_delta else [0.7, 0.3],
        specs=[[{"secondary_y": False}]] * (3 if has_delta else 2),
        subplot_titles=(['', 'Volume Delta', 'Volume'] if has_delta else ['', 'Volume'])
    )

    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Nifty 50',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        ),
        row=1, col=1
    )

    if show_pivots and len(df) > 50:
        try:
            df_json = df.to_json()
            pivots = cached_pivot_calculation(df_json, pivot_settings or {})
            timeframes = {}
            for pivot in pivots:
                tf = pivot['timeframe']
                if tf not in timeframes:
                    timeframes[tf] = {'highs': [], 'lows': [], 'color': pivot['color']}
                if pivot['type'] == 'high':
                    timeframes[tf]['highs'].append((pivot['timestamp'], pivot['value']))
                else:
                    timeframes[tf]['lows'].append((pivot['timestamp'], pivot['value']))
            x_start = df['datetime'].min()
            x_end = df['datetime'].max()
            for tf, data in timeframes.items():
                color = data['color']
                for label, points in [('R', data['highs']), ('S', data['lows'])]:
                    for _, val in points[-3:]:
                        fig.add_shape(type="line", x0=x_start, x1=x_end, y0=val, y1=val,
                                      line=dict(color=color, width=1, dash="dash"), row=1, col=1)
                        fig.add_annotation(x=x_end, y=val, text=f"{tf} {label} {val:.1f}",
                                           showarrow=False, font=dict(color=color, size=10), xanchor="left", row=1, col=1)
        except Exception as e:
            st.warning(f"Error adding pivot levels: {str(e)}")

    try:
        vwap = ReversalDetector.calculate_vwap(df)
        if not vwap.empty:
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=vwap,
                    mode='lines',
                    name='VWAP',
                    line=dict(color='#FFD700', width=2, dash='dot'),
                    opacity=0.8
                ),
                row=1, col=1
            )
    except Exception as e:
        pass

    if vob_blocks:
        try:
            x_start = df['datetime'].min()
            x_end = df['datetime'].max()
            for btype, color, fill, arrow in [
                ('bullish', '#26ba9f', 'rgba(38, 186, 159, 0.15)', '↑'),
                ('bearish', '#6626ba', 'rgba(102, 38, 186, 0.15)', '↓')
            ]:
                for block in vob_blocks.get(btype, [])[-5:]:
                    fig.add_shape(type="rect", x0=x_start, x1=x_end, y0=block['lower'], y1=block['upper'],
                                  fillcolor=fill, line=dict(color=color, width=2), row=1, col=1)
                    fig.add_shape(type="line", x0=x_start, x1=x_end, y0=block['mid'], y1=block['mid'],
                                  line=dict(color=color, width=1, dash="dash"), row=1, col=1)
                    vol_text = VolumeOrderBlocks.format_volume(block['volume'])
                    fig.add_annotation(x=x_end, y=block['mid'], text=f"VOB{arrow} {vol_text} ({block['volume_pct']:.0f}%)",
                                       showarrow=False, font=dict(color=color, size=9), xanchor="left", row=1, col=1)
        except Exception as e:
            pass

    if poc_data:
        try:
            x_start = df['datetime'].min()
            x_end = df['datetime'].max()
            for poc_key, color in [('poc1', '#e91e63'), ('poc2', '#2196f3'), ('poc3', '#4caf50')]:
                poc = poc_data.get(poc_key)
                if poc and poc.get('poc'):
                    period = poc_data.get('periods', {}).get(poc_key, '')
                    fig.add_shape(type="line", x0=x_start, x1=x_end, y0=poc['poc'], y1=poc['poc'],
                                  line=dict(color=color, width=2), row=1, col=1)
                    fig.add_annotation(x=x_end, y=poc['poc'], text=f"POC{poc_key[-1]} ({period}): ₹{poc['poc']:.0f}",
                                       showarrow=False, font=dict(color=color, size=10), xanchor="left", row=1, col=1)
        except Exception as e:
            pass

    if swing_data and swing_data.get('swings'):
        try:
            swings = swing_data['swings']
            projection = swing_data.get('projection')
            x_end = df['datetime'].max()
            atr_est = (df['high'].max() - df['low'].min()) * 0.02
            for key, color, fill, label, y0_off, y1_off in [
                ('last_swing_high', '#eb7514', 'rgba(235, 117, 20, 0.2)', 'H', 0, atr_est),
                ('last_swing_low', '#15dd7c', 'rgba(21, 221, 124, 0.2)', 'L', -atr_est, 0)
            ]:
                pt = swings.get(key)
                if pt and pt.get('value'):
                    x0 = df['datetime'].iloc[pt['index']] if pt['index'] < len(df) else x_end
                    fig.add_shape(type="rect", x0=x0, x1=x_end, y0=pt['value']+y0_off, y1=pt['value']+y1_off,
                                  fillcolor=fill, line=dict(color=color, width=1), row=1, col=1)
                    fig.add_annotation(x=x_end, y=pt['value'], text=f"Swing {label}: ₹{pt['value']:.0f}",
                                       showarrow=False, font=dict(color=color, size=9), xanchor="left", row=1, col=1)
            if projection:
                from_value = projection['from_value']
                target = projection['target']
                direction = projection['direction']
                proj_color = "#15dd7c" if direction == 'bullish' else "#eb7514"
                fig.add_shape(
                    type="line",
                    x0=x_end,
                    x1=x_end,
                    y0=from_value,
                    y1=target,
                    line=dict(color=proj_color, width=2, dash="dash"),
                    row=1, col=1
                )
                fig.add_annotation(
                    x=x_end,
                    y=target,
                    text=f"Target: {projection['sign']}{projection['swing_pct']:.1f}% → ₹{target:.0f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=proj_color,
                    font=dict(color=proj_color, size=10),
                    xanchor="left",
                    row=1, col=1
                )
        except Exception as e:
            pass

    # ── Money Flow Profile overlay on price chart ──
    if money_flow_data and money_flow_data.get('rows'):
        try:
            mf_rows = money_flow_data['rows']
            poc_price = money_flow_data['poc_price']
            va_high = money_flow_data['value_area_high']
            va_low = money_flow_data['value_area_low']
            x_start = df['datetime'].min()
            x_end = df['datetime'].max()

            # Draw horizontal bars as shapes on the right side of the chart
            max_vol = max(r['total_volume'] for r in mf_rows) if mf_rows else 1
            for row in mf_rows:
                ratio = row['total_volume'] / max_vol if max_vol > 0 else 0
                if ratio < 0.05:
                    continue
                # Color by node type
                if row['node_type'] == 'High':
                    fill_color = 'rgba(255, 235, 59, 0.25)'
                elif row['node_type'] == 'Low':
                    fill_color = 'rgba(242, 54, 69, 0.15)'
                else:
                    fill_color = 'rgba(41, 98, 255, 0.18)'

                fig.add_shape(type="rect", x0=x_start, x1=x_end,
                              y0=row['bin_low'], y1=row['bin_high'],
                              fillcolor=fill_color, line=dict(width=0),
                              layer='below', row=1, col=1)

            # POC line
            fig.add_shape(type="line", x0=x_start, x1=x_end,
                          y0=poc_price, y1=poc_price,
                          line=dict(color='#ffeb3b', width=2, dash='solid'),
                          row=1, col=1)
            fig.add_annotation(x=x_end, y=poc_price,
                               text=f"MF-POC ₹{poc_price:.0f}",
                               showarrow=False, font=dict(color='#ffeb3b', size=10),
                               xanchor="left", row=1, col=1)

            # Value area boundaries
            fig.add_shape(type="line", x0=x_start, x1=x_end,
                          y0=va_high, y1=va_high,
                          line=dict(color='#2962ff', width=1, dash='dot'),
                          row=1, col=1)
            fig.add_shape(type="line", x0=x_start, x1=x_end,
                          y0=va_low, y1=va_low,
                          line=dict(color='#2962ff', width=1, dash='dot'),
                          row=1, col=1)
            fig.add_annotation(x=x_start, y=va_high,
                               text=f"VA-H ₹{va_high:.0f}",
                               showarrow=False, font=dict(color='#2962ff', size=9),
                               xanchor="right", row=1, col=1)
            fig.add_annotation(x=x_start, y=va_low,
                               text=f"VA-L ₹{va_low:.0f}",
                               showarrow=False, font=dict(color='#2962ff', size=9),
                               xanchor="right", row=1, col=1)
        except Exception:
            pass

    # ── Volume Delta Candles (row 2) ──
    vol_row = 3 if has_delta else 2
    if has_delta:
        try:
            delta_df = volume_delta_data['df']
            delta_colors = ['#089981' if d > 0 else '#f23645' for d in delta_df['delta'].values]
            # Divergence: bullish candle but negative delta (or vice versa) - use dimmed opposite color
            for i in range(len(delta_df)):
                row_d = delta_df.iloc[i]
                if row_d['divergence']:
                    delta_colors[i] = '#f2364580' if row_d['bar_type'] == 'bullish' else '#08998180'

            fig.add_trace(
                go.Bar(
                    x=delta_df['datetime'],
                    y=delta_df['delta'],
                    name='Volume Delta',
                    marker_color=delta_colors,
                    opacity=0.85,
                    showlegend=False,
                    hovertemplate='Delta: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
            fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, row=2, col=1)

            # Cumulative delta line overlay
            fig.add_trace(
                go.Scatter(
                    x=delta_df['datetime'],
                    y=delta_df['cum_delta'],
                    mode='lines',
                    name='Cum Delta',
                    line=dict(color='#FFD700', width=1.5),
                    opacity=0.7,
                    yaxis='y4' if has_delta else 'y3',
                    showlegend=False,
                    hovertemplate='Cum Δ: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1, secondary_y=False
            )

            # Max volume price dots on the price chart
            fig.add_trace(
                go.Scatter(
                    x=delta_df['datetime'],
                    y=delta_df['max_vol_price'],
                    mode='markers',
                    name='Max Vol Price',
                    marker=dict(size=3, color='white', opacity=0.4),
                    showlegend=False,
                    hovertemplate='Max Vol: ₹%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        except Exception:
            pass

    volume_colors = ['#00ff88' if close >= open else '#ff4444'
                    for close, open in zip(df['close'], df['open'])]

    fig.add_trace(
        go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7,
            showlegend=False
        ),
        row=vol_row, col=1
    )

    chart_height = 850 if has_delta else 700
    fig.update_layout(
        title=title,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=chart_height,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color='white'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e'
    )

    _gc = dict(showgrid=True, gridwidth=1, gridcolor='#333333')
    fig.update_xaxes(title_text="Time (IST)", type='date', row=vol_row, col=1, **_gc)
    fig.update_xaxes(type='date', row=1, col=1, **_gc)
    fig.update_yaxes(title_text="Price (₹)", side='left', row=1, col=1, **_gc)
    fig.update_yaxes(title_text="Volume", side='left', row=vol_row, col=1, **_gc)
    if has_delta:
        fig.update_xaxes(type='date', row=2, col=1, **_gc)
        fig.update_yaxes(title_text="Delta", side='left', row=2, col=1, **_gc)

    return fig

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

def validate_credentials(access_token, client_id):
    issues = []

    clean_token = access_token.strip() if access_token else ""
    clean_client_id = client_id.strip() if client_id else ""

    if not clean_token:
        issues.append("Access token is empty")
    elif len(clean_token) < 50:
        issues.append("Access token seems too short")
    elif clean_token != access_token:
        issues.append("Access token has leading/trailing whitespace")

    if not clean_client_id:
        issues.append("Client ID is empty")
    elif len(clean_client_id) < 5:
        issues.append("Client ID seems too short")
    elif clean_client_id != client_id:
        issues.append("Client ID has leading/trailing whitespace")

    if any(ord(c) < 32 or ord(c) > 126 for c in clean_token):
        issues.append("Access token contains invalid characters")

    if any(ord(c) < 32 or ord(c) > 126 for c in clean_client_id):
        issues.append("Client ID contains invalid characters")

    return clean_token, clean_client_id, issues

def get_user_id():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]
    return st.session_state.user_id

def create_csv_download(df_summary):
    output = io.StringIO()
    df_summary.to_csv(output, index=False)
    return output.getvalue()

def analyze_option_chain(selected_expiry=None, pivot_data=None, vob_data=None):
    now = datetime.now(timezone("Asia/Kolkata"))

    expiry_data = get_dhan_expiry_list_cached(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
    if not expiry_data or 'data' not in expiry_data:
        st.error("Failed to get expiry list from Dhan API")
        return None

    expiry_dates = expiry_data['data']
    if not expiry_dates:
        st.error("No expiry dates available")
        return None

    expiry = selected_expiry if selected_expiry else expiry_dates[0]

    option_chain_data = get_dhan_option_chain(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, expiry)
    if not option_chain_data or 'data' not in option_chain_data:
        st.error("Failed to get option chain from Dhan API")
        return {'underlying': None, 'df_summary': None, 'expiry_dates': expiry_dates, 'expiry': None, 'sr_data': [], 'max_pain_strike': None, 'styled_df': None, 'df_display': None, 'display_cols': [], 'bias_cols': [], 'total_ce_change': 0, 'total_pe_change': 0}

    data = option_chain_data['data']
    underlying = data['last_price']

    oc_data = data['oc']
    calls, puts = [], []
    for strike, strike_data in oc_data.items():
        if 'ce' in strike_data:
            ce_data = strike_data['ce']
            ce_data['strikePrice'] = float(strike)
            calls.append(ce_data)
        if 'pe' in strike_data:
            pe_data = strike_data['pe']
            pe_data['strikePrice'] = float(strike)
            puts.append(pe_data)

    df_ce = pd.DataFrame(calls)
    df_pe = pd.DataFrame(puts)
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice', ascending=False)

    column_mapping = {
        'last_price': 'lastPrice',
        'oi': 'openInterest',
        'previous_oi': 'previousOpenInterest',
        'top_ask_quantity': 'askQty',
        'top_bid_quantity': 'bidQty',
        'volume': 'totalTradedVolume',
        'iv': 'impliedVolatility'
    }
    for old_col, new_col in column_mapping.items():
        if f"{old_col}_CE" in df.columns:
            df.rename(columns={f"{old_col}_CE": f"{new_col}_CE"}, inplace=True)
        if f"{old_col}_PE" in df.columns:
            df.rename(columns={f"{old_col}_PE": f"{new_col}_PE"}, inplace=True)

    df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
    df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']

    T = calculate_exact_time_to_expiry(expiry)
    r = 0.06

    for idx, row in df.iterrows():
        strike = row['strikePrice']
        iv_ce = row.get('impliedVolatility_CE')
        iv_pe = row.get('impliedVolatility_PE')
        if pd.isna(iv_ce) or iv_ce == 0:
            iv_ce, _ = get_iv_fallback(df, strike)
        if pd.isna(iv_pe) or iv_pe == 0:
            _, iv_pe = get_iv_fallback(df, strike)
        iv_ce = iv_ce or 15
        iv_pe = iv_pe or 15
        greeks_ce = calculate_greeks('CE', underlying, strike, T, r, iv_ce / 100)
        greeks_pe = calculate_greeks('PE', underlying, strike, T, r, iv_pe / 100)
        df.at[idx, 'Delta_CE'], df.at[idx, 'Gamma_CE'], df.at[idx, 'Vega_CE'], df.at[idx, 'Theta_CE'], df.at[idx, 'Rho_CE'] = greeks_ce
        df.at[idx, 'Delta_PE'], df.at[idx, 'Gamma_PE'], df.at[idx, 'Vega_PE'], df.at[idx, 'Theta_PE'], df.at[idx, 'Rho_PE'] = greeks_pe

    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))

    atm_plus_minus_2 = df[abs(df['strikePrice'] - atm_strike) <= 100]
    df = atm_plus_minus_2.copy()

    df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
    df['Level'] = df.apply(determine_level, axis=1)

    total_ce_change = df['changeinOpenInterest_CE'].sum() / 100000
    total_pe_change = df['changeinOpenInterest_PE'].sum() / 100000

    bias_results = []
    for _, row in df.iterrows():
        bid_ask_pressure, pressure_bias = calculate_bid_ask_pressure(
            row.get('bidQty_CE', 0), row.get('askQty_CE', 0),
            row.get('bidQty_PE', 0), row.get('askQty_PE', 0)
        )
        score = 0
        row_data = {
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "Level": row['Level'],
        }
        _g = row.get
        row_data["LTP_Bias"] = "Bullish" if _g('lastPrice_CE', 0) > _g('lastPrice_PE', 0) else "Bearish"
        row_data["OI_Bias"] = "Bearish" if _g('openInterest_CE', 0) > _g('openInterest_PE', 0) else "Bullish"
        row_data["ChgOI_Bias"] = "Bearish" if _g('changeinOpenInterest_CE', 0) > _g('changeinOpenInterest_PE', 0) else "Bullish"
        row_data["Volume_Bias"] = "Bullish" if _g('totalTradedVolume_CE', 0) > _g('totalTradedVolume_PE', 0) else "Bearish"
        row_data["Delta_Bias"] = "Bullish" if _g('Delta_CE', 0) > abs(_g('Delta_PE', 0)) else "Bearish"
        row_data["Gamma_Bias"] = "Bullish" if _g('Gamma_CE', 0) > _g('Gamma_PE', 0) else "Bearish"
        row_data["Theta_Bias"] = "Bullish" if _g('Theta_CE', 0) < _g('Theta_PE', 0) else "Bearish"
        row_data["AskQty_Bias"] = "Bullish" if _g('askQty_PE', 0) > _g('askQty_CE', 0) else "Bearish"
        row_data["BidQty_Bias"] = "Bearish" if _g('bidQty_PE', 0) > _g('bidQty_CE', 0) else "Bullish"
        row_data["AskBid_Bias"] = "Bullish" if _g('bidQty_CE', 0) > _g('askQty_CE', 0) else "Bearish"
        row_data["IV_Bias"] = "Bullish" if _g('impliedVolatility_CE', 0) > _g('impliedVolatility_PE', 0) else "Bearish"
        delta_exp_ce = _g('Delta_CE', 0) * _g('openInterest_CE', 0)
        delta_exp_pe = _g('Delta_PE', 0) * _g('openInterest_PE', 0)
        gamma_exp_ce = _g('Gamma_CE', 0) * _g('openInterest_CE', 0)
        gamma_exp_pe = _g('Gamma_PE', 0) * _g('openInterest_PE', 0)
        row_data["DeltaExp"] = "Bullish" if delta_exp_ce > abs(delta_exp_pe) else "Bearish"
        row_data["GammaExp"] = "Bullish" if gamma_exp_ce > gamma_exp_pe else "Bearish"
        row_data["DVP_Bias"] = delta_volume_bias(
            row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
            row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
            row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
        )
        row_data["BidAskPressure"] = bid_ask_pressure
        row_data["PressureBias"] = pressure_bias
        for k in row_data:
            if "_Bias" in k or k in ["DeltaExp", "GammaExp"]:
                bias_val = row_data[k]
                if bias_val == "Bullish":
                    score += 1
                elif bias_val == "Bearish":
                    score -= 1
        row_data["BiasScore"] = score
        row_data["Verdict"] = final_verdict(score)
        if row_data['OI_Bias'] == "Bullish" and row_data['ChgOI_Bias'] == "Bullish":
            row_data["Operator_Entry"] = "Entry Bull"
        elif row_data['OI_Bias'] == "Bearish" and row_data['ChgOI_Bias'] == "Bearish":
            row_data["Operator_Entry"] = "Entry Bear"
        else:
            row_data["Operator_Entry"] = "No Entry"
        if score >= 4:
            row_data["Scalp_Moment"] = "Scalp Bull"
        elif score >= 2:
            row_data["Scalp_Moment"] = "Moment Bull"
        elif score <= -4:
            row_data["Scalp_Moment"] = "Scalp Bear"
        elif score <= -2:
            row_data["Scalp_Moment"] = "Moment Bear"
        else:
            row_data["Scalp_Moment"] = "No Signal"
        if score >= 4:
            row_data["FakeReal"] = "Real Up"
        elif 1 <= score < 4:
            row_data["FakeReal"] = "Fake Up"
        elif score <= -4:
            row_data["FakeReal"] = "Real Down"
        elif -4 < score <= -1:
            row_data["FakeReal"] = "Fake Down"
        else:
            row_data["FakeReal"] = "No Move"
        chg_oi_ce = row.get('changeinOpenInterest_CE', 0)
        chg_oi_pe = row.get('changeinOpenInterest_PE', 0)
        oi_ce = row.get('openInterest_CE', 0)
        oi_pe = row.get('openInterest_PE', 0)
        chg_oi_cmp = '>' if chg_oi_ce > chg_oi_pe else ('<' if chg_oi_ce < chg_oi_pe else '≈')
        row_data["ChgOI_Cmp"] = f"{int(chg_oi_ce/1000)}K {chg_oi_cmp} {int(chg_oi_pe/1000)}K"
        oi_cmp = '>' if oi_ce > oi_pe else ('<' if oi_ce < oi_pe else '≈')
        row_data["OI_Cmp"] = f"{round(oi_ce/1e6, 2)}M {oi_cmp} {round(oi_pe/1e6, 2)}M"
        bias_results.append(row_data)

    df_summary = pd.DataFrame(bias_results)

    merge_cols = ['strikePrice', 'openInterest_CE', 'openInterest_PE', 'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                  'lastPrice_CE', 'lastPrice_PE', 'totalTradedVolume_CE', 'totalTradedVolume_PE',
                  'Delta_CE', 'Delta_PE', 'Gamma_CE', 'Gamma_PE', 'Vega_CE', 'Vega_PE', 'Theta_CE', 'Theta_PE',
                  'impliedVolatility_CE', 'impliedVolatility_PE', 'bidQty_CE', 'bidQty_PE', 'askQty_CE', 'askQty_PE']
    merge_cols = [col for col in merge_cols if col in df.columns]

    df_summary = pd.merge(
        df_summary,
        df[merge_cols],
        left_on='Strike', right_on='strikePrice', how='left'
    )

    if 'openInterest_CE' in df_summary.columns and 'openInterest_PE' in df_summary.columns:
        df_summary['PCR'] = df_summary['openInterest_PE'] / df_summary['openInterest_CE']
        df_summary['PCR'] = np.where(df_summary['openInterest_CE'] == 0, 0, df_summary['PCR'])
        df_summary['PCR'] = df_summary['PCR'].round(2)
        df_summary['PCR_Signal'] = np.where(
            df_summary['PCR'] > 1.2, "Bullish",
            np.where(df_summary['PCR'] < 0.7, "Bearish", "Neutral")
        )
    else:
        df_summary['PCR'] = 0
        df_summary['PCR_Signal'] = "N/A"

    if 'Gamma_CE' in df_summary.columns and 'openInterest_CE' in df_summary.columns:
        df_summary['GammaExp_CE'] = df_summary['Gamma_CE'] * df_summary['openInterest_CE']
        df_summary['GammaExp_PE'] = df_summary['Gamma_PE'] * df_summary['openInterest_PE']
        df_summary['GammaExp_Net'] = df_summary['GammaExp_CE'] - df_summary['GammaExp_PE']
        max_gamma_ce_strike = df_summary.loc[df_summary['GammaExp_CE'].idxmax(), 'Strike'] if not df_summary['GammaExp_CE'].isna().all() else None
        max_gamma_pe_strike = df_summary.loc[df_summary['GammaExp_PE'].idxmax(), 'Strike'] if not df_summary['GammaExp_PE'].isna().all() else None
        df_summary['Gamma_SR'] = df_summary['Strike'].apply(
            lambda x: '🔴 Γ-Resist' if x == max_gamma_ce_strike else ('🟢 Γ-Support' if x == max_gamma_pe_strike else '-')
        )
    else:
        df_summary['Gamma_SR'] = '-'

    if 'Delta_CE' in df_summary.columns and 'openInterest_CE' in df_summary.columns:
        df_summary['DeltaExp_CE'] = df_summary['Delta_CE'] * df_summary['openInterest_CE']
        df_summary['DeltaExp_PE'] = abs(df_summary['Delta_PE'] * df_summary['openInterest_PE'])
        df_summary['DeltaExp_Net'] = df_summary['DeltaExp_CE'] - df_summary['DeltaExp_PE']
        max_delta_ce_strike = df_summary.loc[df_summary['DeltaExp_CE'].idxmax(), 'Strike'] if not df_summary['DeltaExp_CE'].isna().all() else None
        max_delta_pe_strike = df_summary.loc[df_summary['DeltaExp_PE'].idxmax(), 'Strike'] if not df_summary['DeltaExp_PE'].isna().all() else None
        df_summary['Delta_SR'] = df_summary['Strike'].apply(
            lambda x: '🔴 Δ-Resist' if x == max_delta_ce_strike else ('🟢 Δ-Support' if x == max_delta_pe_strike else '-')
        )
    else:
        df_summary['Delta_SR'] = '-'

    if 'bidQty_CE' in df_summary.columns and 'askQty_CE' in df_summary.columns:
        df_summary['Depth_CE'] = df_summary['bidQty_CE'] + df_summary['askQty_CE']
        df_summary['Depth_PE'] = df_summary['bidQty_PE'] + df_summary['askQty_PE']
        max_bid_pe_strike = df_summary.loc[df_summary['bidQty_PE'].idxmax(), 'Strike'] if not df_summary['bidQty_PE'].isna().all() else None
        max_ask_ce_strike = df_summary.loc[df_summary['askQty_CE'].idxmax(), 'Strike'] if not df_summary['askQty_CE'].isna().all() else None
        df_summary['Depth_SR'] = df_summary['Strike'].apply(
            lambda x: '🔴 Depth-R' if x == max_ask_ce_strike else ('🟢 Depth-S' if x == max_bid_pe_strike else '-')
        )
    else:
        df_summary['Depth_SR'] = '-'

    if 'openInterest_CE' in df_summary.columns and 'openInterest_PE' in df_summary.columns:
        max_oi_ce_strike = df_summary.loc[df_summary['openInterest_CE'].idxmax(), 'Strike'] if not df_summary['openInterest_CE'].isna().all() else None
        max_oi_pe_strike = df_summary.loc[df_summary['openInterest_PE'].idxmax(), 'Strike'] if not df_summary['openInterest_PE'].isna().all() else None
        oi_ce_sorted = df_summary.nlargest(2, 'openInterest_CE')['Strike'].tolist()
        oi_pe_sorted = df_summary.nlargest(2, 'openInterest_PE')['Strike'].tolist()
        def get_oi_wall(strike):
            labels = []
            if strike == max_oi_ce_strike:
                labels.append('🔴 OI-Wall-R1')
            elif strike in oi_ce_sorted:
                labels.append('🟠 OI-Wall-R2')
            if strike == max_oi_pe_strike:
                labels.append('🟢 OI-Wall-S1')
            elif strike in oi_pe_sorted:
                labels.append('🟡 OI-Wall-S2')
            return ' | '.join(labels) if labels else '-'
        df_summary['OI_Wall'] = df_summary['Strike'].apply(get_oi_wall)
    else:
        df_summary['OI_Wall'] = '-'

    if 'changeinOpenInterest_CE' in df_summary.columns and 'changeinOpenInterest_PE' in df_summary.columns:
        max_chgoi_ce_idx = df_summary['changeinOpenInterest_CE'].idxmax()
        max_chgoi_pe_idx = df_summary['changeinOpenInterest_PE'].idxmax()
        max_chgoi_ce_strike = df_summary.loc[max_chgoi_ce_idx, 'Strike'] if df_summary.loc[max_chgoi_ce_idx, 'changeinOpenInterest_CE'] > 0 else None
        max_chgoi_pe_strike = df_summary.loc[max_chgoi_pe_idx, 'Strike'] if df_summary.loc[max_chgoi_pe_idx, 'changeinOpenInterest_PE'] > 0 else None
        min_chgoi_ce_idx = df_summary['changeinOpenInterest_CE'].idxmin()
        min_chgoi_pe_idx = df_summary['changeinOpenInterest_PE'].idxmin()
        unwind_ce_strike = df_summary.loc[min_chgoi_ce_idx, 'Strike'] if df_summary.loc[min_chgoi_ce_idx, 'changeinOpenInterest_CE'] < 0 else None
        unwind_pe_strike = df_summary.loc[min_chgoi_pe_idx, 'Strike'] if df_summary.loc[min_chgoi_pe_idx, 'changeinOpenInterest_PE'] < 0 else None
        def get_chgoi_wall(strike):
            labels = []
            if strike == max_chgoi_ce_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_CE'].values[0]
                labels.append(f'🔴 CE+{int(chgoi_val/1000)}K')
            if strike == max_chgoi_pe_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_PE'].values[0]
                labels.append(f'🟢 PE+{int(chgoi_val/1000)}K')
            if strike == unwind_ce_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_CE'].values[0]
                labels.append(f'⚪ CE{int(chgoi_val/1000)}K')
            if strike == unwind_pe_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_PE'].values[0]
                labels.append(f'⚪ PE{int(chgoi_val/1000)}K')
            return ' | '.join(labels) if labels else '-'
        df_summary['ChgOI_Wall'] = df_summary['Strike'].apply(get_chgoi_wall)
    else:
        df_summary['ChgOI_Wall'] = '-'

    max_pain_strike, pain_df = calculate_max_pain(df_summary, underlying)
    if max_pain_strike:
        df_summary['Max_Pain'] = df_summary['Strike'].apply(
            lambda x: '🎯 MAX PAIN' if x == max_pain_strike else '-'
        )
    else:
        df_summary['Max_Pain'] = '-'

    display_cols = ['Strike', 'PCR', 'Verdict', 'ChgOI_Bias', 'Volume_Bias', 'Max_Pain',
                    'Gamma_SR', 'Delta_SR', 'Depth_SR', 'OI_Wall', 'ChgOI_Wall',
                    'Delta_Bias', 'Gamma_Bias', 'Theta_Bias', 'AskQty_Bias', 'BidQty_Bias', 'IV_Bias',
                    'DeltaExp', 'GammaExp', 'DVP_Bias', 'PressureBias', 'BidAskPressure',
                    'BiasScore', 'Operator_Entry', 'Scalp_Moment', 'FakeReal',
                    'ChgOI_Cmp', 'OI_Cmp', 'LTP_Bias', 'PCR_Signal', 'Zone', 'OI_Bias']

    display_cols = [col for col in display_cols if col in df_summary.columns]
    df_display = df_summary[display_cols].copy()

    bias_cols = [col for col in display_cols if '_Bias' in col or col in ['DeltaExp', 'GammaExp', 'PCR_Signal']]

    styled_df = df_display.style\
        .map(color_bias, subset=bias_cols)\
        .map(color_pcr, subset=['PCR'] if 'PCR' in display_cols else [])\
        .map(color_pressure, subset=['BidAskPressure'] if 'BidAskPressure' in display_cols else [])\
        .map(color_verdict, subset=['Verdict'] if 'Verdict' in display_cols else [])\
        .map(color_entry, subset=['Operator_Entry'] if 'Operator_Entry' in display_cols else [])\
        .map(color_fakereal, subset=['FakeReal'] if 'FakeReal' in display_cols else [])\
        .map(color_score, subset=['BiasScore'] if 'BiasScore' in display_cols else [])\
        .apply(highlight_atm_row, axis=1)

    sr_data = []

    if max_pain_strike:
        sr_data.append({
            'Type': '🎯 Max Pain',
            'Level': f"₹{max_pain_strike:.0f}",
            'Source': 'Options OI',
            'Strength': 'High',
            'Signal': 'Price magnet at expiry'
        })

    _sr_pairs = [
        ('openInterest_PE', '🟢 OI Wall Support', lambda v: f"PE OI: {v/100000:.1f}L", 'High', 'Strong support - PE writers defending', False),
        ('openInterest_CE', '🔴 OI Wall Resistance', lambda v: f"CE OI: {v/100000:.1f}L", 'High', 'Strong resistance - CE writers defending', False),
        ('GammaExp_PE', '🟢 Gamma Support', lambda v: 'Gamma Exposure PE', 'Medium', 'Dealers hedge here - price sticky', False),
        ('GammaExp_CE', '🔴 Gamma Resistance', lambda v: 'Gamma Exposure CE', 'Medium', 'Dealers hedge here - price sticky', False),
        ('DeltaExp_PE', '🟢 Delta Support', lambda v: 'Delta Exposure PE', 'Medium', 'Directional bias support', False),
        ('DeltaExp_CE', '🔴 Delta Resistance', lambda v: 'Delta Exposure CE', 'Medium', 'Directional bias resistance', False),
        ('changeinOpenInterest_PE', '🟢 Fresh PE Buildup', lambda v: f"ChgOI: +{v/1000:.0f}K", 'Fresh', 'New support forming today', True),
        ('changeinOpenInterest_CE', '🔴 Fresh CE Buildup', lambda v: f"ChgOI: +{v/1000:.0f}K", 'Fresh', 'New resistance forming today', True),
        ('bidQty_PE', '🟢 Depth Support', lambda v: 'Max PE Bid Qty', 'Real-time', 'Buyers actively defending', False),
        ('askQty_CE', '🔴 Depth Resistance', lambda v: 'Max CE Ask Qty', 'Real-time', 'Sellers actively defending', False),
    ]
    for col, sr_type, src_fn, strength, signal, check_positive in _sr_pairs:
        if col in df_summary.columns:
            idx = df_summary[col].idxmax()
            val = df_summary.loc[idx, col]
            if check_positive and val <= 0:
                continue
            sr_data.append({'Type': sr_type, 'Level': f"₹{df_summary.loc[idx, 'Strike']:.0f}",
                           'Source': src_fn(val), 'Strength': strength, 'Signal': signal})

    if pivot_data:
        tf_pivots = {}
        for pivot in pivot_data:
            tf = pivot['timeframe']
            if tf not in tf_pivots:
                tf_pivots[tf] = {'highs': [], 'lows': []}
            if pivot['type'] == 'high':
                tf_pivots[tf]['highs'].append(pivot['value'])
            else:
                tf_pivots[tf]['lows'].append(pivot['value'])
        for tf, src, strength, s_sig, r_sig in [
            ('5M', '5-Min Timeframe', 'Intraday', 'Short-term support level', 'Short-term resistance level'),
            ('15M', '15-Min Timeframe', 'Swing', 'Key intraday support', 'Key intraday resistance'),
            ('1H', '1-Hour Timeframe', 'Major', 'Strong hourly support - watch closely', 'Strong hourly resistance - watch closely')
        ]:
            if tf in tf_pivots:
                if tf_pivots[tf]['lows']:
                    sr_data.append({'Type': f'🟢 {tf} Pivot Support', 'Level': f"₹{max(tf_pivots[tf]['lows']):.0f}",
                                    'Source': src, 'Strength': strength, 'Signal': s_sig})
                if tf_pivots[tf]['highs']:
                    sr_data.append({'Type': f'🔴 {tf} Pivot Resistance', 'Level': f"₹{min(tf_pivots[tf]['highs']):.0f}",
                                    'Source': src, 'Strength': strength, 'Signal': r_sig})

    vob_blocks = None
    if vob_data:
        vob_sr_levels = vob_data.get('sr_levels', [])
        vob_blocks = vob_data.get('blocks', None)
        for vob_level in vob_sr_levels:
            sr_data.append({k: vob_level[k] for k in ['Type', 'Level', 'Source', 'Strength', 'Signal']})

    return {
        'underlying': underlying,
        'df_summary': df_summary,
        'expiry_dates': expiry_dates,
        'expiry': expiry,
        'sr_data': sr_data,
        'max_pain_strike': max_pain_strike,
        'styled_df': styled_df,
        'df_display': df_display,
        'display_cols': display_cols,
        'bias_cols': bias_cols,
        'total_ce_change': total_ce_change,
        'total_pe_change': total_pe_change,
        'vob_blocks': vob_blocks
    }

def display_analytics_dashboard(db, symbol="NIFTY50"):
    st.subheader("Market Analytics Dashboard")

    analytics_df = db.get_market_analytics(symbol, days_back=30)

    if not analytics_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=pd.to_datetime(analytics_df['date']),
                y=analytics_df['day_close'],
                mode='lines+markers',
                name='Close Price',
                line=dict(color='#00ff88', width=2)
            ))
            fig_price.update_layout(
                title="30-Day Price Trend",
                template='plotly_dark',
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_price, use_container_width=True)
        with col2:
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=pd.to_datetime(analytics_df['date']),
                y=analytics_df['total_volume'],
                name='Volume',
                marker_color='#4444ff'
            ))
            fig_volume.update_layout(
                title="30-Day Volume Trend",
                template='plotly_dark',
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        st.subheader("30-Day Summary")
        metrics = [
            ("Average Price", f"₹{analytics_df['day_close'].mean():,.2f}"),
            ("Volatility (σ)", f"{analytics_df['price_change_pct'].std():.2f}%"),
            ("Max Daily Gain", f"{analytics_df['price_change_pct'].max():.2f}%"),
            ("Max Daily Loss", f"{analytics_df['price_change_pct'].min():.2f}%")
        ]
        for col, (label, val) in zip(st.columns(4), metrics):
            col.metric(label, val)

def analyze_strike_activity(df_summary, underlying_price):
    """Comprehensive option chain analysis: Call/Put activity, support/resistance, market bias."""
    if df_summary is None or df_summary.empty or 'Zone' not in df_summary.columns:
        return None

    atm_idx = df_summary[df_summary['Zone'] == 'ATM'].index
    if len(atm_idx) == 0:
        return None
    atm_pos = df_summary.index.get_loc(atm_idx[0])
    start_idx = max(0, atm_pos - 5)
    end_idx = min(len(df_summary), atm_pos + 6)
    near_atm = df_summary.iloc[start_idx:end_idx].copy()

    # Compute median OI for "high OI" threshold
    median_ce_oi = near_atm['openInterest_CE'].median() if 'openInterest_CE' in near_atm.columns else 0
    median_pe_oi = near_atm['openInterest_PE'].median() if 'openInterest_PE' in near_atm.columns else 0

    strike_analysis = []
    for _, row in near_atm.iterrows():
        strike = row['Strike']
        zone = row.get('Zone', '')

        ce_oi = row.get('openInterest_CE', 0) or 0
        pe_oi = row.get('openInterest_PE', 0) or 0
        ce_chg_oi = row.get('changeinOpenInterest_CE', 0) or 0
        pe_chg_oi = row.get('changeinOpenInterest_PE', 0) or 0
        ce_ltp = row.get('lastPrice_CE', 0) or 0
        pe_ltp = row.get('lastPrice_PE', 0) or 0
        ce_vol = row.get('totalTradedVolume_CE', 0) or 0
        pe_vol = row.get('totalTradedVolume_PE', 0) or 0
        ce_iv = row.get('impliedVolatility_CE', 0) or 0
        pe_iv = row.get('impliedVolatility_PE', 0) or 0

        # --- CALL SIDE ---
        ce_high_oi = ce_oi > median_ce_oi * 1.2
        ce_oi_rising = ce_chg_oi > 0
        ce_oi_falling = ce_chg_oi < 0
        price_below_strike = underlying_price < strike

        # Call Writer Activity
        call_writing = ce_high_oi and ce_oi_rising
        call_capping = call_writing and price_below_strike

        # Call Buyer Activity
        ce_long_buildup = ce_oi_rising and ce_ltp > 0 and ce_vol > 0
        ce_short_covering = ce_oi_falling and ce_ltp > 0

        # Call Classification
        if ce_high_oi and ce_oi_rising and price_below_strike:
            call_class = "Strong Resistance"
            call_strength = "Strong"
        elif ce_oi_falling and not price_below_strike:
            call_class = "Breakout Zone"
            call_strength = "Breaking"
        elif ce_high_oi and not ce_oi_rising:
            call_class = "Weak Resistance"
            call_strength = "Weak"
        elif ce_oi_rising:
            call_class = "Moderate Resistance"
            call_strength = "Moderate"
        else:
            call_class = "Weak Resistance"
            call_strength = "Weak"

        # Call activity label
        if ce_short_covering:
            call_activity = "Short Covering"
        elif call_writing:
            call_activity = "Writing (Resistance)"
        elif ce_long_buildup:
            call_activity = "Long Build-up"
        else:
            call_activity = "Low Activity"

        # --- PUT SIDE ---
        pe_high_oi = pe_oi > median_pe_oi * 1.2
        pe_oi_rising = pe_chg_oi > 0
        pe_oi_falling = pe_chg_oi < 0
        price_above_strike = underlying_price > strike

        # Put Writer Activity
        put_writing = pe_high_oi and pe_oi_rising
        put_support = put_writing and price_above_strike

        # Put Buyer Activity
        pe_long_buildup = pe_oi_rising and pe_ltp > 0 and pe_vol > 0
        pe_long_unwinding = pe_oi_falling and pe_ltp > 0

        # Put Classification
        if pe_high_oi and pe_oi_rising and price_above_strike:
            put_class = "Strong Support"
            put_strength = "Strong"
        elif pe_oi_falling and not price_above_strike:
            put_class = "Breakdown Zone"
            put_strength = "Breaking"
        elif pe_high_oi and not pe_oi_rising:
            put_class = "Weak Support"
            put_strength = "Weak"
        elif pe_oi_rising:
            put_class = "Moderate Support"
            put_strength = "Moderate"
        else:
            put_class = "Weak Support"
            put_strength = "Weak"

        # Put activity label
        if pe_long_unwinding:
            put_activity = "Long Unwinding"
        elif put_writing:
            put_activity = "Writing (Support)"
        elif pe_long_buildup:
            put_activity = "Long Build-up (Bearish)"
        else:
            put_activity = "Low Activity"

        # Trapped writers detection
        call_trapped = ce_high_oi and ce_oi_falling and not price_below_strike  # price went above, writers trapped
        put_trapped = pe_high_oi and pe_oi_falling and price_below_strike  # price went below, writers trapped

        strike_analysis.append({
            'Strike': strike, 'Zone': zone,
            'CE_OI': ce_oi, 'CE_ChgOI': ce_chg_oi, 'CE_LTP': ce_ltp, 'CE_Vol': ce_vol, 'CE_IV': ce_iv,
            'PE_OI': pe_oi, 'PE_ChgOI': pe_chg_oi, 'PE_LTP': pe_ltp, 'PE_Vol': pe_vol, 'PE_IV': pe_iv,
            'Call_Class': call_class, 'Call_Strength': call_strength, 'Call_Activity': call_activity,
            'Put_Class': put_class, 'Put_Strength': put_strength, 'Put_Activity': put_activity,
            'Call_Trapped': call_trapped, 'Put_Trapped': put_trapped,
        })

    analysis_df = pd.DataFrame(strike_analysis)

    # Top 3 Resistance (Call side) - sort by CE OI descending, prefer strong
    strength_order = {'Strong': 0, 'Moderate': 1, 'Weak': 2, 'Breaking': 3}
    res_df = analysis_df[analysis_df['Strike'] >= underlying_price].copy()
    res_df['_sort'] = res_df['Call_Strength'].map(strength_order).fillna(3)
    res_df = res_df.sort_values(['_sort', 'CE_OI'], ascending=[True, False]).head(3)
    top_resistance = res_df[['Strike', 'Call_Class', 'Call_Strength', 'CE_OI', 'CE_ChgOI', 'Call_Activity']].copy()

    # Top 3 Support (Put side) - sort by PE OI descending, prefer strong
    sup_df = analysis_df[analysis_df['Strike'] <= underlying_price].copy()
    sup_df['_sort'] = sup_df['Put_Strength'].map(strength_order).fillna(3)
    sup_df = sup_df.sort_values(['_sort', 'PE_OI'], ascending=[True, False]).head(3)
    top_support = sup_df[['Strike', 'Put_Class', 'Put_Strength', 'PE_OI', 'PE_ChgOI', 'Put_Activity']].copy()

    # Trapped writers
    trapped_call_writers = analysis_df[analysis_df['Call_Trapped']][['Strike', 'CE_OI', 'CE_ChgOI']].copy()
    trapped_put_writers = analysis_df[analysis_df['Put_Trapped']][['Strike', 'PE_OI', 'PE_ChgOI']].copy()

    # Breakout / Breakdown zones
    breakout_zones = analysis_df[analysis_df['Call_Class'] == 'Breakout Zone'][['Strike', 'CE_OI', 'CE_ChgOI']].copy()
    breakdown_zones = analysis_df[analysis_df['Put_Class'] == 'Breakdown Zone'][['Strike', 'PE_OI', 'PE_ChgOI']].copy()

    # Market Interpretation
    total_ce_chg = analysis_df['CE_ChgOI'].sum()
    total_pe_chg = analysis_df['PE_ChgOI'].sum()
    strong_res_count = len(analysis_df[analysis_df['Call_Class'] == 'Strong Resistance'])
    strong_sup_count = len(analysis_df[analysis_df['Put_Class'] == 'Strong Support'])
    breakout_count = len(breakout_zones)
    breakdown_count = len(breakdown_zones)
    call_unwinding = total_ce_chg < 0
    put_unwinding = total_pe_chg < 0
    call_building = total_ce_chg > 0
    put_building = total_pe_chg > 0

    # Confidence scoring
    confidence = 50
    bias_signals = []

    if call_unwinding and put_building:
        market_bias = "Bullish"
        confidence += 20
        bias_signals.append("Call OI unwinding + Put writing increasing")
    elif call_building and put_unwinding:
        market_bias = "Bearish"
        confidence += 20
        bias_signals.append("Call writing increasing + Put OI unwinding")
    elif strong_res_count > 0 and strong_sup_count > 0:
        market_bias = "Sideways"
        confidence += 10
        bias_signals.append("Strong resistance above + Strong support below")
    elif breakout_count > 0 and strong_sup_count > 0:
        market_bias = "Bullish (Breakout)"
        confidence += 25
        bias_signals.append("Call writers trapped + Put support strong")
    elif breakdown_count > 0 and strong_res_count > 0:
        market_bias = "Bearish (Breakdown)"
        confidence += 25
        bias_signals.append("Put writers trapped + Call resistance strong")
    elif put_building and call_building:
        if total_pe_chg > total_ce_chg:
            market_bias = "Mildly Bullish"
            confidence += 10
            bias_signals.append("Both building but PE OI > CE OI change")
        else:
            market_bias = "Mildly Bearish"
            confidence += 10
            bias_signals.append("Both building but CE OI > PE OI change")
    else:
        market_bias = "Neutral"
        bias_signals.append("No clear directional signal")

    if len(trapped_call_writers) > 0:
        confidence += 10
        bias_signals.append(f"{len(trapped_call_writers)} trapped call writer(s)")
    if len(trapped_put_writers) > 0:
        confidence += 10
        bias_signals.append(f"{len(trapped_put_writers)} trapped put writer(s)")
    if strong_sup_count >= 2:
        confidence += 5
        bias_signals.append(f"{strong_sup_count} strong support levels")
    if strong_res_count >= 2:
        confidence += 5
        bias_signals.append(f"{strong_res_count} strong resistance levels")

    confidence = min(confidence, 95)

    return {
        'analysis_df': analysis_df,
        'top_resistance': top_resistance,
        'top_support': top_support,
        'trapped_call_writers': trapped_call_writers,
        'trapped_put_writers': trapped_put_writers,
        'breakout_zones': breakout_zones,
        'breakdown_zones': breakdown_zones,
        'market_bias': market_bias,
        'confidence': confidence,
        'bias_signals': bias_signals,
        'total_ce_chg': total_ce_chg,
        'total_pe_chg': total_pe_chg,
    }

def send_option_chain_signal(sa_result, underlying_price):
    """Send comprehensive option chain analysis signal to Telegram."""
    if sa_result is None:
        return
    bias = sa_result['market_bias']
    conf = sa_result['confidence']
    analysis_df = sa_result['analysis_df']

    time_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')

    # Active signals detection
    active_signals = []
    # Call capping
    capping = analysis_df[(analysis_df['Call_Class'] == 'Strong Resistance') & (analysis_df['Call_Activity'] == 'Writing (Resistance)')]
    for _, r in capping.iterrows():
        active_signals.append(f"🟥 CALL CAPPING ACTIVE at ₹{r['Strike']:.0f} (OI: {r['CE_OI']/100000:.1f}L, ChgOI: +{r['CE_ChgOI']/1000:.0f}K)")
    # Put support
    support = analysis_df[(analysis_df['Put_Class'] == 'Strong Support') & (analysis_df['Put_Activity'] == 'Writing (Support)')]
    for _, r in support.iterrows():
        active_signals.append(f"🟩 PUT SUPPORT STRONG at ₹{r['Strike']:.0f} (OI: {r['PE_OI']/100000:.1f}L, ChgOI: +{r['PE_ChgOI']/1000:.0f}K)")
    # Breakout
    breakout = analysis_df[analysis_df['Call_Class'] == 'Breakout Zone']
    for _, r in breakout.iterrows():
        active_signals.append(f"🚀 BREAKOUT LOADING at ₹{r['Strike']:.0f} (CE OI unwinding: {r['CE_ChgOI']/1000:.0f}K)")
    # Breakdown
    breakdown = analysis_df[analysis_df['Put_Class'] == 'Breakdown Zone']
    for _, r in breakdown.iterrows():
        active_signals.append(f"💥 BREAKDOWN LOADING at ₹{r['Strike']:.0f} (PE OI unwinding: {r['PE_ChgOI']/1000:.0f}K)")
    # Trapped writers
    trapped_ce = analysis_df[analysis_df['Call_Trapped']]
    for _, r in trapped_ce.iterrows():
        active_signals.append(f"⚠️ TRAPPED CALL WRITERS at ₹{r['Strike']:.0f}")
    trapped_pe = analysis_df[analysis_df['Put_Trapped']]
    for _, r in trapped_pe.iterrows():
        active_signals.append(f"⚠️ TRAPPED PUT WRITERS at ₹{r['Strike']:.0f}")

    # Market condition
    if 'Bullish' in bias:
        market_emoji = "🟢"
        condition = "BULLISH"
    elif 'Bearish' in bias:
        market_emoji = "🔴"
        condition = "BEARISH"
    else:
        market_emoji = "🟡"
        condition = "SIDEWAYS"

    # Top resistance
    res_lines = []
    for _, r in sa_result['top_resistance'].iterrows():
        res_lines.append(f"  🟥 ₹{r['Strike']:.0f} | {r['Call_Strength']} | OI: {r['CE_OI']/100000:.1f}L | {r['Call_Activity']}")
    # Top support
    sup_lines = []
    for _, r in sa_result['top_support'].iterrows():
        sup_lines.append(f"  🟩 ₹{r['Strike']:.0f} | {r['Put_Strength']} | OI: {r['PE_OI']/100000:.1f}L | {r['Put_Activity']}")

    # Breakout/Breakdown levels
    breakout_lvl = f"₹{breakout.iloc[0]['Strike']:.0f}" if not breakout.empty else "None"
    breakdown_lvl = f"₹{breakdown.iloc[0]['Strike']:.0f}" if not breakdown.empty else "None"

    signals_text = "\n".join(active_signals) if active_signals else "  No active signals"
    res_text = "\n".join(res_lines) if res_lines else "  None identified"
    sup_text = "\n".join(sup_lines) if sup_lines else "  None identified"

    message = f"""📊 <b>OPTION CHAIN DEEP ANALYSIS</b> 📊
🕐 {time_str} | Spot: ₹{underlying_price:.2f}

━━━ {market_emoji} MARKET CONDITION: <b>{condition}</b> ━━━
📈 Confidence: <b>{conf}%</b>
{'▓' * (conf // 5)}{'░' * (20 - conf // 5)}

<b>🟥 CALL CAPPING / RESISTANCE:</b>
{res_text}

<b>🟩 PUT SUPPORT LEVELS:</b>
{sup_text}

<b>⚡ ACTIVE SIGNALS:</b>
{signals_text}

<b>🚀 Breakout Level:</b> {breakout_lvl}
<b>💥 Breakdown Level:</b> {breakdown_lvl}

<b>📋 BIAS REASONING:</b>
{"".join([f"• {s}" + chr(10) for s in sa_result['bias_signals']])}
⚠️ <i>Auto-generated. Manual verification required.</i>"""

    try:
        send_telegram_message_sync(message)
    except Exception:
        pass

def detect_candle_patterns(df, lookback=5):
    """Detect candlestick patterns from last few candles using Nifty price action chart."""
    if df is None or len(df) < lookback:
        return {'pattern': 'Insufficient Data', 'direction': 'Neutral', 'details': {}, 'candles': []}
    recent = df.tail(lookback).copy()
    last = recent.iloc[-1]
    prev = recent.iloc[-2] if len(recent) >= 2 else None

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

        c_pattern = 'Normal'
        if c_lower > c_body * 2 and c_upper < c_body * 0.5 and c_body_ratio < 0.4:
            c_pattern = 'Hammer'
        elif c_upper > c_body * 2 and c_lower < c_body * 0.5 and c_body_ratio < 0.4:
            c_pattern = 'Shooting Star' if not c_green else 'Inverted Hammer'
        elif c_body_ratio < 0.1 and c_range > 0:
            c_pattern = 'Doji'
        elif c_prev is not None:
            p_body = abs(c_prev['close'] - c_prev['open'])
            p_green = c_prev['close'] > c_prev['open']
            if c_green and not p_green and c_body > p_body and c['close'] > c_prev['open'] and c['open'] < c_prev['close']:
                c_pattern = 'Bullish Engulfing'
            elif not c_green and p_green and c_body > p_body and c['close'] < c_prev['open'] and c['open'] > c_prev['close']:
                c_pattern = 'Bearish Engulfing'
            elif c_body_ratio >= 0.6:
                c_pattern = 'Strong Green' if c_green else 'Strong Red'
        if c_pattern == 'Normal' and c_body_ratio >= 0.6:
            c_pattern = 'Strong Green' if c_green else 'Strong Red'

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

    if lower_wick > body * 2 and upper_wick < body * 0.5 and body_ratio < 0.4:
        pattern = 'Hammer'
        direction = 'Bullish'
    elif upper_wick > body * 2 and lower_wick < body * 0.5 and body_ratio < 0.4:
        pattern = 'Shooting Star' if not is_green else 'Inverted Hammer'
        direction = 'Bearish' if not is_green else 'Bullish'
    elif body_ratio < 0.1 and total_range > 0:
        pattern = 'Doji'
        direction = 'Indecision'
    elif prev is not None:
        prev_body = abs(prev['close'] - prev['open'])
        prev_green = prev['close'] > prev['open']
        if is_green and not prev_green and body > prev_body and last['close'] > prev['open'] and last['open'] < prev['close']:
            pattern = 'Bullish Engulfing'
            direction = 'Bullish'
        elif not is_green and prev_green and body > prev_body and last['close'] < prev['open'] and last['open'] > prev['close']:
            pattern = 'Bearish Engulfing'
            direction = 'Bearish'
        elif body_ratio >= 0.6:
            pattern = 'Strong Green Candle' if is_green else 'Strong Red Candle'
            direction = 'Bullish' if is_green else 'Bearish'

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
    """Detect bullish and bearish order blocks."""
    if df is None or len(df) < lookback:
        return {'bullish_ob': None, 'bearish_ob': None}
    recent = df.tail(lookback).copy()
    bullish_ob = None
    bearish_ob = None

    for i in range(1, len(recent) - 2):
        curr = recent.iloc[i]
        nxt = recent.iloc[i + 1]
        nxt2 = recent.iloc[i + 2] if i + 2 < len(recent) else None
        # Bullish OB: last red candle before strong up move
        is_red = curr['close'] < curr['open']
        if is_red and nxt['close'] > nxt['open']:
            up_move = nxt['close'] - curr['low']
            avg_range = recent['high'].mean() - recent['low'].mean()
            if up_move > avg_range * 1.5:
                bullish_ob = {'low': curr['low'], 'high': curr['high'],
                              'time': recent.index[i] if hasattr(recent.index[i], 'strftime') else i}
        # Bearish OB: last green candle before strong down move
        is_green = curr['close'] > curr['open']
        if is_green and nxt['close'] < nxt['open']:
            down_move = curr['high'] - nxt['close']
            avg_range = recent['high'].mean() - recent['low'].mean()
            if down_move > avg_range * 1.5:
                bearish_ob = {'low': curr['low'], 'high': curr['high'],
                              'time': recent.index[i] if hasattr(recent.index[i], 'strftime') else i}

    return {'bullish_ob': bullish_ob, 'bearish_ob': bearish_ob}

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

def fetch_alignment_data(api):
    """Fetch candle data for SENSEX, BANKNIFTY, NIFTY IT, RELIANCE, ICICI, VIX.
    Detect candle patterns, compute 10m/1h/4h sentiment for each."""
    # Dhan security IDs
    tickers = [
        ('SENSEX', '51', 'BSE_EQ', 'INDEX'),
        ('BANKNIFTY', '25', 'IDX_I', 'INDEX'),
        ('NIFTY IT', '30', 'IDX_I', 'INDEX'),
        ('RELIANCE', '2885', 'NSE_EQ', 'EQUITY'),
        ('ICICIBANK', '4963', 'NSE_EQ', 'EQUITY'),
        ('INDIA VIX', '26', 'IDX_I', 'INDEX'),
    ]
    alignment = {}
    for name, sec_id, seg, inst in tickers:
        try:
            # Fetch 1-min candle data (last 1 day)
            data = api.get_intraday_data(security_id=sec_id, exchange_segment=seg, instrument=inst, interval="1", days_back=1)
            if data and 'open' in data:
                adf = process_candle_data(data, "1")
                if adf.empty:
                    alignment[name] = {'ltp': 0, 'trend': 'Unknown', 'candle_pattern': 'N/A',
                                       'candle_dir': 'N/A', 'candles': [],
                                       'sentiment_10m': 'N/A', 'sentiment_1h': 'N/A', 'sentiment_4h': 'N/A'}
                    continue

                ltp = adf.iloc[-1]['close']
                # Candle pattern detection
                cp = detect_candle_patterns(adf, lookback=5)

                # Multi-timeframe sentiment
                def calc_sentiment(sub_df):
                    if sub_df is None or len(sub_df) < 2:
                        return 'N/A', 0
                    first_close = sub_df.iloc[0]['close']
                    last_close = sub_df.iloc[-1]['close']
                    pct = ((last_close - first_close) / first_close) * 100 if first_close > 0 else 0
                    green_count = sum(1 for _, r in sub_df.iterrows() if r['close'] > r['open'])
                    bear_count = len(sub_df) - green_count
                    if pct > 0.1 and green_count > bear_count:
                        return 'Bullish', round(pct, 2)
                    elif pct < -0.1 and bear_count > green_count:
                        return 'Bearish', round(pct, 2)
                    else:
                        return 'Neutral', round(pct, 2)

                # Last 10 minutes
                s_10m, pct_10m = calc_sentiment(adf.tail(10))
                # Last 60 minutes (1 hour)
                s_1h, pct_1h = calc_sentiment(adf.tail(60))
                # Last 240 minutes (4 hours)
                s_4h, pct_4h = calc_sentiment(adf.tail(240))

                # Overall trend from LTP
                prev_ltp = adf.iloc[-2]['close'] if len(adf) >= 2 else ltp
                trend = 'Bullish' if ltp > prev_ltp else 'Bearish' if ltp < prev_ltp else 'Flat'

                # % change series from day open for line chart
                day_open = adf.iloc[0]['open']
                pct_series_time = adf['datetime'].tolist()
                pct_series_vals = [((c - day_open) / day_open) * 100 if day_open > 0 else 0 for c in adf['close'].tolist()]

                alignment[name] = {
                    'ltp': ltp, 'trend': trend,
                    'candle_pattern': cp['pattern'], 'candle_dir': cp['direction'],
                    'candles': cp.get('candles', []),
                    'bull_count': cp.get('bull_count', 0), 'bear_count': cp.get('bear_count', 0),
                    'sentiment_10m': s_10m, 'pct_10m': pct_10m,
                    'sentiment_1h': s_1h, 'pct_1h': pct_1h,
                    'sentiment_4h': s_4h, 'pct_4h': pct_4h,
                    'day_high': adf['high'].max(), 'day_low': adf['low'].min(),
                    'open': adf.iloc[0]['open'],
                    'pct_series_time': pct_series_time,
                    'pct_series_vals': pct_series_vals,
                }
            else:
                alignment[name] = {'ltp': 0, 'trend': 'Unknown', 'candle_pattern': 'N/A',
                                   'candle_dir': 'N/A', 'candles': [],
                                   'sentiment_10m': 'N/A', 'sentiment_1h': 'N/A', 'sentiment_4h': 'N/A'}
        except Exception:
            alignment[name] = {'ltp': 0, 'trend': 'Unknown', 'candle_pattern': 'N/A',
                               'candle_dir': 'N/A', 'candles': [],
                               'sentiment_10m': 'N/A', 'sentiment_1h': 'N/A', 'sentiment_4h': 'N/A'}
    return alignment

def fetch_vix_data(api):
    """Fetch India VIX data via Dhan API."""
    try:
        resp = api.get_ltp_data("26", "IDX_I")  # India VIX security ID=26
        if resp and 'data' in resp:
            data = resp['data']
            if isinstance(data, dict):
                for seg_key, items in data.items():
                    if items:
                        item = items[0] if isinstance(items, list) else items
                        return {'vix': item.get('last_price', 0)}
    except Exception:
        pass
    return {'vix': 0}

def generate_master_signal(df, sa_result, gex_data, confluence_data, underlying_price, api):
    """Generate comprehensive trading signal using all available data."""
    if df is None or df.empty or sa_result is None:
        return None

    # 1. Candle Pattern (last 5 candles)
    candle = detect_candle_patterns(df, lookback=5)

    # 2. Order Blocks
    ob = detect_order_blocks(df, lookback=20)

    # 3. Volume Analysis
    vol = detect_volume_spike(df, lookback=5)

    # 4. Key levels from existing app data
    support_levels = [float(r['Strike']) for _, r in sa_result['top_support'].iterrows()] if not sa_result['top_support'].empty else []
    resistance_levels = [float(r['Strike']) for _, r in sa_result['top_resistance'].iterrows()] if not sa_result['top_resistance'].empty else []

    # 5. Candle location
    location = get_candle_location(underlying_price, support_levels, resistance_levels, gex_data, ob)

    # 6. VIX (with session state caching)
    vix_data = {'vix': 0, 'direction': 'Unknown'}
    try:
        if 'vix_history' not in st.session_state:
            st.session_state.vix_history = []
        vix_resp = fetch_vix_data(api)
        vix_val = vix_resp.get('vix', 0)
        if vix_val > 0:
            st.session_state.vix_history.append(vix_val)
            if len(st.session_state.vix_history) > 50:
                st.session_state.vix_history = st.session_state.vix_history[-50:]
            if len(st.session_state.vix_history) >= 2:
                vix_dir = 'Rising' if st.session_state.vix_history[-1] > st.session_state.vix_history[0] else 'Falling'
            else:
                vix_dir = 'Unknown'
            vix_data = {'vix': vix_val, 'direction': vix_dir}
    except Exception:
        pass

    # 7. Index & Stock Alignment (with caching - fetch every 120s to reduce API calls)
    alignment = {}
    try:
        if 'alignment_data' not in st.session_state:
            st.session_state.alignment_data = {}
            st.session_state.alignment_last_fetch = None
        ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
        should_fetch = st.session_state.alignment_last_fetch is None or \
            (ist_now - st.session_state.alignment_last_fetch).total_seconds() > 120
        if should_fetch:
            alignment = fetch_alignment_data(api)
            if alignment:
                st.session_state.alignment_data = alignment
                st.session_state.alignment_last_fetch = ist_now
        alignment = st.session_state.alignment_data

        # Also add NIFTY's own sentiment from the main df
        if df is not None and not df.empty:
            nifty_cp = detect_candle_patterns(df, lookback=5)
            def _nifty_sentiment(sub):
                if sub is None or len(sub) < 2:
                    return 'N/A', 0
                fc, lc = sub.iloc[0]['close'], sub.iloc[-1]['close']
                pct = ((lc - fc) / fc) * 100 if fc > 0 else 0
                gc = sum(1 for _, r in sub.iterrows() if r['close'] > r['open'])
                if pct > 0.1 and gc > len(sub) // 2:
                    return 'Bullish', round(pct, 2)
                elif pct < -0.1 and gc <= len(sub) // 2:
                    return 'Bearish', round(pct, 2)
                return 'Neutral', round(pct, 2)
            s10, p10 = _nifty_sentiment(df.tail(10))
            s1h, p1h = _nifty_sentiment(df.tail(60))
            s4h, p4h = _nifty_sentiment(df.tail(240))
            nifty_day_open = df.iloc[0]['open']
            nifty_pct_time = df['datetime'].tolist()
            nifty_pct_vals = [((c - nifty_day_open) / nifty_day_open) * 100 if nifty_day_open > 0 else 0 for c in df['close'].tolist()]
            alignment['NIFTY 50'] = {
                'ltp': df.iloc[-1]['close'], 'trend': nifty_cp['direction'],
                'candle_pattern': nifty_cp['pattern'], 'candle_dir': nifty_cp['direction'],
                'candles': nifty_cp.get('candles', []),
                'bull_count': nifty_cp.get('bull_count', 0), 'bear_count': nifty_cp.get('bear_count', 0),
                'sentiment_10m': s10, 'pct_10m': p10,
                'sentiment_1h': s1h, 'pct_1h': p1h,
                'sentiment_4h': s4h, 'pct_4h': p4h,
                'day_high': df['high'].max(), 'day_low': df['low'].min(),
                'open': nifty_day_open,
                'pct_series_time': nifty_pct_time,
                'pct_series_vals': nifty_pct_vals,
            }

        # VIX direction from alignment data
        vix_align = alignment.get('INDIA VIX', {})
        if vix_align.get('ltp', 0) > 0 and vix_data.get('vix', 0) == 0:
            vix_data['vix'] = vix_align['ltp']
            vix_data['direction'] = vix_align.get('sentiment_10m', 'Unknown')
            if vix_align.get('sentiment_10m') == 'Bullish':
                vix_data['direction'] = 'Rising'
            elif vix_align.get('sentiment_10m') == 'Bearish':
                vix_data['direction'] = 'Falling'
    except Exception:
        pass

    # === EXISTING APP DATA ===
    market_bias = sa_result.get('market_bias', 'Neutral')
    app_confidence = sa_result.get('confidence', 50)

    # GEX data
    net_gex = gex_data.get('total_gex', 0) if gex_data else 0
    gamma_flip = gex_data.get('gamma_flip_level') if gex_data else None
    gex_magnet = gex_data.get('gex_magnet') if gex_data else None
    gex_repeller = gex_data.get('gex_repeller') if gex_data else None
    gex_signal = gex_data.get('gex_signal', '') if gex_data else ''
    gex_interpretation = gex_data.get('gex_interpretation', '') if gex_data else ''
    market_mode = 'TREND' if net_gex < 0 else 'RANGE'

    # ATM GEX
    atm_gex = 0
    if gex_data and 'gex_df' in gex_data:
        gex_df = gex_data['gex_df']
        atm_rows = gex_df[gex_df['Zone'] == 'ATM'] if 'Zone' in gex_df.columns else pd.DataFrame()
        if not atm_rows.empty:
            atm_gex = atm_rows.iloc[0].get('Net_GEX', 0)

    above_gamma_flip = underlying_price > gamma_flip if gamma_flip else None

    # PCR × GEX Confluence
    pcr_gex_badge = confluence_data[0] if confluence_data else ''
    pcr_gex_signal = confluence_data[1] if confluence_data and len(confluence_data) > 1 else ''

    # === CONFLUENCE SCORING ===
    score = 0
    reasons = []

    # 1. Candle pattern
    if candle['direction'] == 'Bullish':
        score += 1
        reasons.append(f"Bullish {candle['pattern']}")
    elif candle['direction'] == 'Bearish':
        score -= 1
        reasons.append(f"Bearish {candle['pattern']}")

    # 2. Volume spike
    if vol['spike']:
        score += 1 if candle['direction'] == 'Bullish' else -1 if candle['direction'] == 'Bearish' else 0
        reasons.append(f"Volume Spike ({vol['ratio']}x)")

    # 3. Order Block
    near_bullish_ob = any('Bullish OB' in loc for loc in location)
    near_bearish_ob = any('Bearish OB' in loc for loc in location)
    if near_bullish_ob:
        score += 1
        reasons.append("At Bullish Order Block")
    elif near_bearish_ob:
        score -= 1
        reasons.append("At Bearish Order Block")

    # 4. GEX confirmation
    if net_gex < 0 and candle['direction'] == 'Bullish' and above_gamma_flip:
        score += 1
        reasons.append("GEX Trend + Above Gamma Flip")
    elif net_gex < 0 and candle['direction'] == 'Bearish' and not above_gamma_flip:
        score -= 1
        reasons.append("GEX Trend + Below Gamma Flip")
    elif net_gex > 0:
        reasons.append("GEX Range Mode (pinning)")

    # 5. PCR / Option chain confirmation
    if 'Bullish' in market_bias:
        score += 1
        reasons.append(f"Option Chain: {market_bias}")
    elif 'Bearish' in market_bias:
        score -= 1
        reasons.append(f"Option Chain: {market_bias}")

    # 6. Alignment (use 10m sentiment for immediate direction)
    non_vix = {k: v for k, v in alignment.items() if 'VIX' not in k}
    bullish_align = sum(1 for v in non_vix.values() if v.get('sentiment_10m') == 'Bullish')
    bearish_align = sum(1 for v in non_vix.values() if v.get('sentiment_10m') == 'Bearish')
    total_align = len(non_vix) if non_vix else 1
    if bullish_align > bearish_align and bullish_align >= 2:
        score += 1
        reasons.append(f"Alignment Bullish ({bullish_align}/{total_align})")
    elif bearish_align > bullish_align and bearish_align >= 2:
        score -= 1
        reasons.append(f"Alignment Bearish ({bearish_align}/{total_align})")

    # 7. VIX
    vix_dir = vix_data.get('direction', 'Unknown')
    if vix_dir == 'Falling' and score > 0:
        score += 1
        reasons.append("VIX Falling (Aligned)")
    elif vix_dir == 'Rising' and score < 0:
        score += -1  # more bearish
        reasons.append("VIX Rising (Aligned)")
    elif vix_dir == 'Rising' and score > 0:
        reasons.append("VIX Rising (Opposite)")
    elif vix_dir == 'Falling' and score < 0:
        reasons.append("VIX Falling (Opposite)")

    # === DETERMINE SIGNAL ===
    abs_score = abs(score)
    near_support = any('Support' in loc for loc in location)
    near_resistance = any('Resistance' in loc for loc in location)
    breakout_zones = not sa_result['breakout_zones'].empty
    breakdown_zones = not sa_result['breakdown_zones'].empty

    if breakout_zones and vol['spike'] and candle['direction'] == 'Bullish' and net_gex < 0:
        signal = '🚀 BREAKOUT'
        trade_type = 'STRONG BUY'
    elif breakdown_zones and vol['spike'] and candle['direction'] == 'Bearish' and net_gex < 0:
        signal = '💥 BREAKDOWN'
        trade_type = 'STRONG SELL'
    elif score >= 5:
        signal = '🟩 PUT SUPPORT' if near_support else '🟢 STRONG BUY'
        trade_type = 'STRONG BUY'
    elif score >= 3:
        signal = '🟩 PUT SUPPORT' if near_support else '🟢 BUY'
        trade_type = 'STRONG BUY' if near_support else 'SCALP BUY'
    elif score <= -5:
        signal = '🟥 CALL CAPPING' if near_resistance else '🔴 STRONG SELL'
        trade_type = 'STRONG SELL'
    elif score <= -3:
        signal = '🟥 CALL CAPPING' if near_resistance else '🔴 SELL'
        trade_type = 'STRONG SELL' if near_resistance else 'SCALP SELL'
    elif net_gex > 0 and abs_score < 3:
        signal = '⚖️ RANGE'
        trade_type = 'RANGE'
    else:
        signal = '⚪ NO TRADE'
        trade_type = 'NO TRADE'

    # Confidence
    confidence = min(95, max(10, 30 + abs_score * 10))
    if vol['spike']:
        confidence += 5
    if vix_dir in ['Falling', 'Rising'] and 'Aligned' in str(reasons):
        confidence += 5
    confidence = min(95, confidence)

    # Strength label
    if abs_score >= 6:
        strength = 'STRONG'
    elif abs_score >= 4:
        strength = 'MODERATE'
    else:
        strength = 'NO TRADE' if abs_score < 3 else 'WEAK'

    return {
        'candle': candle,
        'volume': vol,
        'order_blocks': ob,
        'location': location,
        'vix': vix_data,
        'alignment': alignment,
        'gex': {
            'net_gex': net_gex, 'atm_gex': atm_gex,
            'gamma_flip': gamma_flip, 'magnet': gex_magnet,
            'repeller': gex_repeller, 'market_mode': market_mode,
            'signal': gex_signal, 'interpretation': gex_interpretation,
            'above_flip': above_gamma_flip,
        },
        'pcr_gex': {'badge': pcr_gex_badge, 'signal': pcr_gex_signal},
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'signal': signal,
        'trade_type': trade_type,
        'score': score,
        'abs_score': abs_score,
        'strength': strength,
        'confidence': confidence,
        'reasons': reasons,
        'market_bias': market_bias,
    }

def send_master_signal_telegram(result, underlying_price):
    """Send master signal to Telegram."""
    if result is None:
        return
    time_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')

    # Alignment text with candle pattern and sentiment
    align_lines = []
    display_order = ['NIFTY 50', 'SENSEX', 'BANKNIFTY', 'NIFTY IT', 'RELIANCE', 'ICICIBANK', 'INDIA VIX']
    for name in display_order:
        data = result.get('alignment', {}).get(name)
        if data is None:
            continue
        s10 = data.get('sentiment_10m', 'N/A')
        s1h = data.get('sentiment_1h', 'N/A')
        pat = data.get('candle_pattern', 'N/A')
        ltp = data.get('ltp', 0)
        emoji = '🟢' if s10 == 'Bullish' else '🔴' if s10 == 'Bearish' else '⚪'
        ltp_str = f"₹{ltp:.0f}" if ltp > 0 else 'N/A'
        align_lines.append(f"  {emoji} {name}: {ltp_str} | {pat} | 10m:{s10} | 1h:{s1h}")
    align_text = "\n".join(align_lines) if align_lines else "  Data unavailable"

    # Location
    loc_text = ", ".join(result['location'])

    # Reasons
    reason_text = "\n".join([f"  ✔ {r}" for r in result['reasons']])

    gex = result['gex']
    vix = result['vix']

    # Signal color
    if 'BUY' in result['trade_type'] or 'BREAKOUT' in result['signal']:
        signal_emoji = "🟢"
    elif 'SELL' in result['trade_type'] or 'BREAKDOWN' in result['signal']:
        signal_emoji = "🔴"
    else:
        signal_emoji = "🟡"

    res_text = ", ".join([f"₹{r:.0f}" for r in result['resistance_levels'][:3]]) if result['resistance_levels'] else "None"
    sup_text = ", ".join([f"₹{s:.0f}" for s in result['support_levels'][:3]]) if result['support_levels'] else "None"

    message = f"""{signal_emoji} <b>MASTER TRADING SIGNAL</b> {signal_emoji}
🕐 {time_str} | Spot: ₹{underlying_price:.2f}

━━━ SIGNAL: <b>{result['signal']}</b> ━━━
📊 Trade: <b>{result['trade_type']}</b>
🎯 Confluence: <b>{result['abs_score']}/7 ({result['strength']})</b>
📈 Confidence: <b>{result['confidence']}%</b>

<b>🕯 Candle:</b> {result['candle']['pattern']} ({result['candle']['direction']})
<b>📍 Location:</b> {loc_text}
<b>📊 Volume:</b> {result['volume']['label']} ({result['volume']['ratio']}x)

<b>🟥 Resistance:</b> {res_text}
<b>🟩 Support:</b> {sup_text}

<b>🔮 GEX:</b>
  Net: {gex['net_gex']:+.1f}L | ATM: {gex['atm_gex']:+.1f}L
  Gamma Flip: {'₹' + str(int(gex['gamma_flip'])) if gex['gamma_flip'] else 'N/A'} ({'Above' if gex['above_flip'] else 'Below' if gex['above_flip'] is not None else 'N/A'})
  Magnet: {'₹' + str(int(gex['magnet'])) if gex['magnet'] else 'N/A'}
  Repeller: {'₹' + str(int(gex['repeller'])) if gex['repeller'] else 'N/A'}
  Mode: {gex['market_mode']}

<b>📊 PCR×GEX:</b> {result['pcr_gex']['badge']}

<b>🟢 Order Blocks:</b>
  Bullish OB: {'₹' + str(int(result['order_blocks']['bullish_ob']['low'])) + '-' + str(int(result['order_blocks']['bullish_ob']['high'])) if result['order_blocks'].get('bullish_ob') else 'None'}
  Bearish OB: {'₹' + str(int(result['order_blocks']['bearish_ob']['low'])) + '-' + str(int(result['order_blocks']['bearish_ob']['high'])) if result['order_blocks'].get('bearish_ob') else 'None'}

<b>🌍 Alignment:</b>
{align_text}

<b>📉 VIX:</b> {vix.get('vix', 'N/A')} ({vix.get('direction', 'Unknown')})

<b>📋 CONFLUENCE FACTORS:</b>
{reason_text}

⚠️ <i>Auto-generated signal. Manual verification required.</i>"""

    try:
        send_telegram_message_sync(message)
    except Exception:
        pass

def main():
    st.title("📈 Nifty Trading & Options Analyzer")

    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    market_open = current_time.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=15, minute=45, second=0, microsecond=0)

    is_market_hours = market_open <= current_time <= market_close
    is_weekday = current_time.weekday() < 5

    if not (is_market_hours and is_weekday):
        st.warning("⏰ **Market is Closed**")
        if not is_weekday:
            st.info(f"📅 Today is {current_time.strftime('%A')}. Markets are closed on weekends.")
        else:
            st.info(f"""
            🕐 **Current Time:** {current_time.strftime('%H:%M:%S IST')}
            📊 **Market Hours:** 8:30 AM - 3:45 PM IST (Monday to Friday)
            The app will be fully functional during market hours.
            """)
        st.caption("You can still view cached data if available.")

    for key, default in [('pcr_history', []), ('pcr_last_valid_data', None),
                          ('gex_history', []), ('gex_last_valid_data', None),
                          ('last_gex_alert', None), ('gex_current_strikes', []),
                          ('oi_history', []), ('oi_last_valid_data', None),
                          ('oi_current_strikes', []),
                          ('chgoi_history', []), ('chgoi_last_valid_data', None),
                          ('chgoi_current_strikes', [])]:
        if key not in st.session_state:
            st.session_state[key] = default

    try:
        if not supabase_url or not supabase_key:
            st.error("Please configure your Supabase credentials in Streamlit secrets")
            st.info("""
            Add the following to your Streamlit secrets:
            ```
            [supabase]
            url = "your_supabase_url"
            anon_key = "your_supabase_anon_key"
            ```
            """)
            return
        db = SupabaseDB(supabase_url, supabase_key)
        db.sync_pending()
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return

    # Load PCR/GEX history from Supabase if session_state is empty (e.g. after page refresh)
    if not st.session_state.pcr_history:
        try:
            pcr_db_df = db.get_pcr_history()
            if not pcr_db_df.empty and 'timestamp' in pcr_db_df.columns:
                pcr_db_df['timestamp'] = pd.to_datetime(pcr_db_df['timestamp'])
                grouped = pcr_db_df.groupby('timestamp')
                for ts, group in grouped:
                    entry = {'time': ts.to_pydatetime()}
                    for _, row in group.iterrows():
                        entry[str(int(row['strike_price']))] = float(row['pcr_value'])
                    st.session_state.pcr_history.append(entry)
                st.session_state.pcr_history = st.session_state.pcr_history[-200:]
        except Exception:
            pass

    if not st.session_state.gex_history:
        try:
            gex_db_df = db.get_gex_history()
            if not gex_db_df.empty and 'timestamp' in gex_db_df.columns:
                gex_db_df['timestamp'] = pd.to_datetime(gex_db_df['timestamp'])
                grouped = gex_db_df.groupby('timestamp')
                for ts, group in grouped:
                    entry = {'time': ts.to_pydatetime(), 'total_gex': float(group['total_gex'].iloc[0])}
                    for _, row in group.iterrows():
                        entry[str(int(row['strike_price']))] = float(row['net_gex'])
                    st.session_state.gex_history.append(entry)
                st.session_state.gex_history = st.session_state.gex_history[-200:]
        except Exception:
            pass

    try:
        if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
            st.error("Please configure your Dhan API credentials in Streamlit secrets")
            st.info("""
            Add the following to your Streamlit secrets:
            ```
            DHAN_CLIENT_ID = "your_client_id"
            DHAN_ACCESS_TOKEN = "your_access_token"
            TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
            TELEGRAM_CHAT_ID = "your_telegram_chat_id"
            ```
            """)
            return
        access_token, client_id, issues = validate_credentials(DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID)
        if issues:
            st.error("Issues found with API credentials:")
            for issue in issues:
                st.error(f"• {issue}")
            st.info("The app will try to use the cleaned values automatically.")
        st.sidebar.success("API credentials processed")
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            st.sidebar.success("Telegram notifications enabled")
        else:
            st.sidebar.warning("Telegram notifications disabled - configure bot token and chat ID")

    except Exception as e:
        st.error(f"Credential validation error: {str(e)}")
        return

    user_id = get_user_id()
    user_prefs = db.get_user_preferences(user_id)

    st.sidebar.header("Configuration")

    timeframes = {
        "1 min": "1",
        "3 min": "3",
        "5 min": "5",
        "10 min": "10",
        "15 min": "15"
    }

    default_timeframe = next((k for k, v in timeframes.items() if v == user_prefs['timeframe']), "5 min")
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        list(timeframes.keys()),
        index=list(timeframes.keys()).index(default_timeframe)
    )

    interval = timeframes[selected_timeframe]

    st.sidebar.header("📊 Pivot Indicator Settings")

    show_pivots = st.sidebar.checkbox("Show Pivot Levels", value=True, help="Display Higher Timeframe Support/Resistance levels")

    if show_pivots:
        st.sidebar.subheader("Toggle Individual Pivot Levels")
        if 'pivot_settings' not in user_prefs:
            user_prefs['pivot_settings'] = {
                'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True
            }
        show_3m = st.sidebar.checkbox("3 Minute Pivots", value=user_prefs['pivot_settings'].get('show_3m', True), help="🟢 Green lines")
        show_5m = st.sidebar.checkbox("5 Minute Pivots", value=user_prefs['pivot_settings'].get('show_5m', True), help="🟠 Orange lines")
        show_10m = st.sidebar.checkbox("10 Minute Pivots", value=user_prefs['pivot_settings'].get('show_10m', True), help="🟣 Pink lines")
        show_15m = st.sidebar.checkbox("15 Minute Pivots", value=user_prefs['pivot_settings'].get('show_15m', True), help="🔵 Blue lines")
        pivot_settings = {
            'show_3m': show_3m,
            'show_5m': show_5m,
            'show_10m': show_10m,
            'show_15m': show_15m
        }
        st.sidebar.info("""
        **Pivot Levels Legend:**
        🟢 3M (Green) - 3-minute timeframe
        🟠 5M (Orange) - 5-minute timeframe
        🟣 10M (Pink) - 10-minute timeframe
        🔵 15M (Blue) - 15-minute timeframe
        S = Support, R = Resistance
        """)
    else:
        pivot_settings = {
            'show_3m': False, 'show_5m': False, 'show_10m': False, 'show_15m': False
        }

    st.sidebar.header("🔔 Trading Signals")
    enable_signals = st.sidebar.checkbox("Enable Telegram Signals", value=True, help="Send notifications when conditions are met")

    pivot_proximity = st.sidebar.slider(
        "Pivot Proximity (± Points)",
        min_value=1,
        max_value=20,
        value=user_prefs.get('pivot_proximity', 5),
        help="Distance from pivot levels to trigger signals (both above and below)"
    )

    if enable_signals:
        st.sidebar.info(f"Signals sent when:\n• Price within ±{pivot_proximity}pts of pivot\n• All option bias aligned\n• ATM at support/resistance")

    st.sidebar.header("📅 Options Settings")

    expiry_data = get_dhan_expiry_list_cached(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
    expiry_dates = []
    if expiry_data and 'data' in expiry_data:
        expiry_dates = expiry_data['data']

    selected_expiry = None
    if expiry_dates:
        expiry_options = [f"{exp} ({'Weekly' if i < 4 else 'Monthly'})" for i, exp in enumerate(expiry_dates)]
        selected_expiry_idx = st.sidebar.selectbox(
            "Select Expiry",
            range(len(expiry_options)),
            format_func=lambda x: expiry_options[x]
        )
        selected_expiry = expiry_dates[selected_expiry_idx]

    auto_refresh = st.sidebar.checkbox("Auto Refresh (2 min)", value=user_prefs['auto_refresh'])

    days_back = st.sidebar.slider("Days of Historical Data", 1, 5, user_prefs['days_back'])

    use_cache = st.sidebar.checkbox("Use Cached Data", value=True, help="Use database cache for faster loading")

    st.sidebar.header("🗑️ Database Management")
    cleanup_days = st.sidebar.selectbox("Clear History Older Than", [7, 14, 30], index=0)

    if st.sidebar.button("🗑 Clear History"):
        deleted_count = db.clear_old_candles(cleanup_days)
        st.sidebar.success(f"Deleted {deleted_count} old records")

    st.sidebar.header("🔧 Connection Test")

    if st.sidebar.button("Test Telegram Connection"):
        success, message = test_telegram_connection()
        if success:
            st.sidebar.success(message)
            test_msg = "🔔 Nifty Analyzer - Test message successful! ✅"
            send_telegram_message_sync(test_msg)
            st.sidebar.success("Test message sent to Telegram!")
        else:
            st.sidebar.error(message)

    if st.sidebar.button("💾 Save Preferences"):
        db.save_user_preferences(user_id, interval, auto_refresh, days_back, pivot_settings, pivot_proximity)
        st.sidebar.success("Preferences saved!")

    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    show_analytics = st.sidebar.checkbox("Show Analytics Dashboard", value=False)

    st.sidebar.subheader("🔧 Debug Info")
    st.sidebar.write(f"Telegram Bot Token: {'✅ Set' if TELEGRAM_BOT_TOKEN else '❌ Missing'}")
    st.sidebar.write(f"Telegram Chat ID: {'✅ Set' if TELEGRAM_CHAT_ID else '❌ Missing'}")
    st.sidebar.write(f"Token length: {len(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else 0}")
    st.sidebar.write(f"Chat ID: {TELEGRAM_CHAT_ID}")

    api = DhanAPI(access_token, client_id)

    col1, col2 = st.columns([2, 1])

    pivots = None
    vob_data = None

    with col1:
        st.header("📈 Trading Chart")
        df = pd.DataFrame()
        current_price = None
        if use_cache:
            df = db.get_candles("NIFTY50", "IDX_I", interval, hours_back=days_back*24)
        need_fetch = not use_cache or df.empty or (datetime.now(pytz.UTC) - df['datetime'].max().tz_convert(pytz.UTC)).total_seconds() > 300
        if need_fetch:
            with st.spinner("Fetching data from API..."):
                data = api.get_intraday_data(security_id="13", exchange_segment="IDX_I", instrument="INDEX", interval=interval, days_back=days_back)
                if data:
                    df = process_candle_data(data, interval)
                    db.upsert_candles("NIFTY50", "IDX_I", interval, df)
        ltp_data = api.get_ltp_data("13", "IDX_I")
        if ltp_data and 'data' in ltp_data:
            for exchange, data in ltp_data['data'].items():
                for security_id, price_data in data.items():
                    current_price = price_data.get('last_price', 0)
                    break
        if current_price is not None and current_price > 0:
            try:
                db.upsert_spot_data(current_price, security_id='13', exchange_segment='IDX_I')
            except Exception:
                pass
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        if not df.empty:
            display_metrics(ltp_data, df, db)
        vob_blocks_for_chart = None
        if not df.empty and len(df) > 30:
            try:
                vob_detector = VolumeOrderBlocks(sensitivity=5)
                _, vob_blocks_for_chart = vob_detector.get_sr_levels(df)
            except Exception:
                vob_blocks_for_chart = None
        poc_data_for_chart = None
        if not df.empty and len(df) > 100:
            try:
                poc_calculator = TriplePOC(period1=10, period2=25, period3=70)
                poc_data_for_chart = poc_calculator.calculate_all_pocs(df)
            except Exception:
                poc_data_for_chart = None
        swing_data_for_chart = None
        if not df.empty and len(df) > 50:
            try:
                swing_calculator = FutureSwing(swing_length=30, history_samples=5, calc_type='Average')
                swing_data_for_chart = swing_calculator.analyze(df)
            except Exception:
                swing_data_for_chart = None
        # Persist detected patterns (VOB, POC, Swing) to Supabase
        try:
            patterns_to_store = []
            if vob_blocks_for_chart:
                for block_type in ['bullish', 'bearish']:
                    for block in vob_blocks_for_chart.get(block_type, []):
                        patterns_to_store.append({
                            'pattern_type': f'VOB_{block_type.upper()}',
                            'timeframe': interval,
                            'direction': block_type,
                            'price_level': block.get('mid'),
                            'upper_bound': block.get('upper'),
                            'lower_bound': block.get('lower'),
                            'score': block.get('volume_pct'),
                            'metadata': {'volume': block.get('volume', 0), 'datetime': str(block.get('datetime', ''))}
                        })
            if poc_data_for_chart:
                for poc_key in ['poc1', 'poc2', 'poc3']:
                    poc = poc_data_for_chart.get(poc_key)
                    if poc:
                        period = poc_data_for_chart.get('periods', {}).get(poc_key, '')
                        patterns_to_store.append({
                            'pattern_type': f'POC_{poc_key.upper()}',
                            'timeframe': str(period),
                            'direction': 'neutral',
                            'price_level': poc.get('poc'),
                            'upper_bound': poc.get('upper_poc'),
                            'lower_bound': poc.get('lower_poc'),
                            'score': None,
                            'metadata': {'high': poc.get('high', 0), 'low': poc.get('low', 0), 'volume': poc.get('volume', 0)}
                        })
            if swing_data_for_chart:
                swings = swing_data_for_chart.get('swings', {})
                projection = swing_data_for_chart.get('projection')
                if projection:
                    patterns_to_store.append({
                        'pattern_type': 'SWING_PROJECTION',
                        'timeframe': interval,
                        'direction': projection.get('direction', 'unknown'),
                        'price_level': projection.get('target'),
                        'upper_bound': projection.get('from_value'),
                        'lower_bound': None,
                        'score': projection.get('swing_pct'),
                        'metadata': {'volume_delta': swing_data_for_chart.get('volume', {}).get('delta', 0)}
                    })
                last_high = swings.get('last_swing_high')
                last_low = swings.get('last_swing_low')
                if last_high:
                    patterns_to_store.append({
                        'pattern_type': 'SWING_HIGH',
                        'timeframe': interval,
                        'direction': 'bearish',
                        'price_level': last_high.get('value'),
                        'upper_bound': None, 'lower_bound': None, 'score': None,
                        'metadata': {'index': last_high.get('index', 0)}
                    })
                if last_low:
                    patterns_to_store.append({
                        'pattern_type': 'SWING_LOW',
                        'timeframe': interval,
                        'direction': 'bullish',
                        'price_level': last_low.get('value'),
                        'upper_bound': None, 'lower_bound': None, 'score': None,
                        'metadata': {'index': last_low.get('index', 0)}
                    })
            if patterns_to_store:
                db.upsert_detected_patterns(patterns_to_store)
        except Exception:
            pass
        # ── Compute Money Flow Profile & Volume Delta ──
        money_flow_data = None
        volume_delta_data = None
        if not df.empty and len(df) > 5:
            try:
                money_flow_data = calculate_money_flow_profile(df, num_rows=25, source='Volume')
            except Exception as e:
                st.caption(f"⚠️ MF Profile error: {str(e)[:80]}")
                money_flow_data = None
            try:
                volume_delta_data = calculate_volume_delta(df)
            except Exception as e:
                st.caption(f"⚠️ Volume Delta error: {str(e)[:80]}")
                volume_delta_data = None
            # Store Money Flow Profile POC + Volume Delta summary as patterns
            try:
                mf_vd_patterns = []
                if money_flow_data:
                    mf_vd_patterns.append({
                        'pattern_type': 'MONEY_FLOW_POC',
                        'timeframe': interval,
                        'direction': money_flow_data.get('highest_sentiment_direction', 'neutral').lower(),
                        'price_level': money_flow_data.get('poc_price'),
                        'upper_bound': money_flow_data.get('value_area_high'),
                        'lower_bound': money_flow_data.get('value_area_low'),
                        'score': None,
                        'metadata': {
                            'total_volume': money_flow_data.get('total_volume', 0),
                            'num_rows': money_flow_data.get('num_rows', 25),
                            'sentiment_price': money_flow_data.get('highest_sentiment_price', 0),
                            'source': money_flow_data.get('source', 'Volume'),
                        }
                    })
                if volume_delta_data and volume_delta_data.get('summary'):
                    vds = volume_delta_data['summary']
                    mf_vd_patterns.append({
                        'pattern_type': 'VOLUME_DELTA',
                        'timeframe': interval,
                        'direction': vds.get('bias', 'neutral').lower(),
                        'price_level': current_price,
                        'upper_bound': None,
                        'lower_bound': None,
                        'score': vds.get('delta_ratio'),
                        'metadata': {
                            'total_delta': vds.get('total_delta', 0),
                            'buy_volume': vds.get('total_buy_volume', 0),
                            'sell_volume': vds.get('total_sell_volume', 0),
                            'divergence_bars': vds.get('divergence_bars', 0),
                            'cum_delta': vds.get('cum_delta_last', 0),
                        }
                    })
                if mf_vd_patterns:
                    db.upsert_detected_patterns(mf_vd_patterns)
            except Exception:
                pass
        if not df.empty:
            fig = create_candlestick_chart(
                df,
                f"Nifty 50 - {selected_timeframe} Chart {'with Pivot Levels' if show_pivots else ''}",
                interval,
                show_pivots=show_pivots,
                pivot_settings=pivot_settings,
                vob_blocks=vob_blocks_for_chart,
                poc_data=poc_data_for_chart,
                swing_data=swing_data_for_chart,
                money_flow_data=money_flow_data,
                volume_delta_data=volume_delta_data
            )
            st.plotly_chart(fig, use_container_width=True)
            infos = [
                f"📊 Data Points: {len(df)}",
                f"🕐 Latest: {df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S IST')}",
                f"📡 Source: {'Database Cache' if use_cache else 'Live API'}",
                f"📈 Pivots: {'✅ Enabled' if show_pivots else '❌ Disabled'}"
            ]
            for c, txt in zip(st.columns(4), infos):
                c.info(txt)
            if show_pivots and len(df) > 50:
                st.markdown("""
                **Pivot Levels Legend:**
                - 🟢 **3M Levels**: 3-minute timeframe support/resistance
                - 🟠 **5M Levels**: 5-minute timeframe swing points
                - 🟣 **10M Levels**: 10-minute support/resistance zones
                - 🔵 **15M Levels**: 15-minute major support/resistance levels
                - 🟡 **VWAP**: Volume Weighted Average Price (dotted line)
                - 🟩 **VOB↑ (Teal)**: Bullish Volume Order Blocks (Support zones)
                - 🟪 **VOB↓ (Purple)**: Bearish Volume Order Blocks (Resistance zones)
                *R = Resistance (Price ceiling), S = Support (Price floor)*
                *VOB zones show volume-backed order flow areas with % distribution*
                """)
            # ── Money Flow Profile & Volume Delta Tables ──
            st.markdown("---")
            mf_tab, vd_tab = st.tabs(["💰 Money Flow Profile", "📊 Volume Delta Candles"])
            with mf_tab:
                if money_flow_data and money_flow_data.get('rows'):
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("MF-POC", f"₹{money_flow_data['poc_price']:.0f}")
                    mc2.metric("Value Area", f"₹{money_flow_data['value_area_low']:.0f} - ₹{money_flow_data['value_area_high']:.0f}")
                    mc3.metric("Top Sentiment", f"{money_flow_data['highest_sentiment_direction']} @ ₹{money_flow_data['highest_sentiment_price']:.0f}")
                    mc4.metric("Bars", f"{money_flow_data['num_bars']}")
                    mf_df = pd.DataFrame(money_flow_data['rows'])
                    mf_display = mf_df[['price_level', 'total_volume', 'bull_volume', 'bear_volume', 'delta', 'volume_pct', 'node_type', 'sentiment', 'sentiment_strength']].copy()
                    mf_display.columns = ['Price', 'Total Vol', 'Buy Vol', 'Sell Vol', 'Delta', 'Vol %', 'Node', 'Sentiment', 'Strength %']
                    mf_display['Price'] = mf_display['Price'].apply(lambda x: f"₹{x:.0f}")
                    mf_display['Total Vol'] = mf_display['Total Vol'].apply(lambda x: f"{x:,.0f}")
                    mf_display['Buy Vol'] = mf_display['Buy Vol'].apply(lambda x: f"{x:,.0f}")
                    mf_display['Sell Vol'] = mf_display['Sell Vol'].apply(lambda x: f"{x:,.0f}")
                    mf_display['Delta'] = mf_display['Delta'].apply(lambda x: f"{x:+,.0f}")
                    mf_display = mf_display.reset_index(drop=True)
                    def _mf_row_style(row):
                        styles = [''] * len(row)
                        try:
                            node = row.get('Node', '')
                            sent = row.get('Sentiment', '')
                            node_idx = list(row.index).index('Node')
                            sent_idx = list(row.index).index('Sentiment')
                            if node == 'High': styles[node_idx] = 'background-color: #ffeb3b40; color: white'
                            elif node == 'Low': styles[node_idx] = 'background-color: #f2364540; color: white'
                            else: styles[node_idx] = 'background-color: #2962ff30; color: white'
                            if sent == 'Bullish': styles[sent_idx] = 'background-color: #26a69a40; color: white'
                            elif sent == 'Bearish': styles[sent_idx] = 'background-color: #ef535040; color: white'
                        except: pass
                        return styles
                    styled_mf = mf_display.style.apply(_mf_row_style, axis=1)
                    st.dataframe(styled_mf, use_container_width=True, hide_index=True)
                    st.markdown("""
                    **Money Flow Profile:**
                    - 🟡 **High Traded Nodes**: Consolidation/value areas - price tends to spend time here
                    - 🔵 **Average Nodes**: Normal trading activity levels
                    - 🔴 **Low Traded Nodes**: Supply/demand zones, liquidity gaps - price moves fast through these
                    - **MF-POC**: Price level with highest traded volume (Point of Control)
                    - **VA-H/VA-L**: Value Area boundaries (consolidation zone)
                    """)
                else:
                    st.info(f"Money Flow Profile: No data yet. df rows={len(df) if not df.empty else 0}, money_flow_data={'computed' if money_flow_data else 'None'}")

            with vd_tab:
                if volume_delta_data and volume_delta_data.get('summary'):
                    vd_summary = volume_delta_data['summary']
                    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
                    bias_color = "normal" if vd_summary['bias'] == 'Bullish' else "inverse"
                    vc1.metric("Total Delta", f"{vd_summary['total_delta']:+,}", delta=vd_summary['bias'], delta_color=bias_color)
                    vc2.metric("Buy Volume", f"{vd_summary['total_buy_volume']:,}")
                    vc3.metric("Sell Volume", f"{vd_summary['total_sell_volume']:,}")
                    vc4.metric("Delta Ratio", f"{vd_summary['delta_ratio']:.2f}")
                    vc5.metric("Divergences", f"{vd_summary['divergence_bars']}")
                    vd_df = volume_delta_data['df'].copy()
                    vd_display = vd_df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 'delta', 'delta_pct', 'cum_delta', 'divergence']].tail(50).copy()
                    vd_display.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Buy Vol', 'Sell Vol', 'Delta', 'Delta %', 'Cum Delta', 'Divergence']
                    vd_display['Time'] = pd.to_datetime(vd_display['Time']).dt.strftime('%H:%M')
                    vd_display = vd_display.reset_index(drop=True)
                    def _vd_style(row):
                        styles = [''] * len(row)
                        try:
                            d = float(row['Delta'])
                            color = '#08998130' if d > 0 else '#f2364530' if d < 0 else ''
                            if color:
                                for i, col in enumerate(row.index):
                                    if col in ['Delta', 'Cum Delta']:
                                        styles[i] = f'background-color: {color}; color: white'
                            if row['Divergence']:
                                div_idx = list(row.index).index('Divergence')
                                styles[div_idx] = 'background-color: #FFD70040; color: white'
                        except: pass
                        return styles
                    styled_vd = vd_display.style.apply(_vd_style, axis=1)
                    st.dataframe(styled_vd, use_container_width=True, hide_index=True)
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown(f"""
                        **Volume Delta Summary:**
                        - +Delta Streak: **{vd_summary['max_positive_streak']}** bars
                        - -Delta Streak: **{vd_summary['max_negative_streak']}** bars
                        - +Delta Bars: **{vd_summary['positive_bars']}** | -Delta Bars: **{vd_summary['negative_bars']}**
                        """)
                    with dc2:
                        st.markdown("""
                        **Interpretation:**
                        - **+Delta**: Buyers dominating (price should rise)
                        - **-Delta**: Sellers dominating (price should fall)
                        - **Divergence**: Candle direction opposes delta (potential reversal)
                        - **Cum Delta**: Running total - trend of buying/selling pressure
                        """)
                else:
                    st.info(f"Volume Delta: No data yet. df rows={len(df) if not df.empty else 0}, volume_delta_data={'computed' if volume_delta_data else 'None'}")

            st.markdown("---")
            st.markdown("## 🔄 Intraday Reversal Detector")
            try:
                pivot_lows = []
                pivot_highs = []
                if show_pivots and len(df) > 50:
                    df_json = df.to_json()
                    pivots = cached_pivot_calculation(df_json, pivot_settings or {})
                    pivot_lows = [p['value'] for p in pivots if p['type'] == 'low']
                    pivot_highs = [p['value'] for p in pivots if p['type'] == 'high']
                if len(df) > 30:
                    try:
                        vob_detector = VolumeOrderBlocks(sensitivity=5)
                        vob_sr_levels, vob_blocks = vob_detector.get_sr_levels(df)
                        vob_data = {
                            'sr_levels': vob_sr_levels,
                            'blocks': vob_blocks
                        }
                    except Exception as e:
                        st.warning(f"VOB calculation error: {str(e)}")
                        vob_data = None
                bull_score, bull_signals, bull_verdict = ReversalDetector.calculate_reversal_score(df, pivot_lows)
                bear_score, bear_signals, bear_verdict = ReversalDetector.calculate_bearish_reversal_score(df, pivot_highs)
                # Persist reversal patterns to Supabase
                try:
                    reversal_patterns = []
                    if bull_score >= 3:
                        reversal_patterns.append({
                            'pattern_type': 'REVERSAL_BULLISH',
                            'timeframe': interval,
                            'direction': 'bullish',
                            'price_level': bull_signals.get('Current_Price'),
                            'upper_bound': bull_signals.get('Day_High'),
                            'lower_bound': bull_signals.get('Day_Low'),
                            'score': bull_score,
                            'metadata': {
                                'verdict': bull_verdict,
                                'vwap': bull_signals.get('VWAP'),
                                'selling_exhausted': bull_signals.get('Selling_Exhausted', ''),
                                'higher_low': bull_signals.get('Higher_Low', ''),
                                'volume_signal': bull_signals.get('Volume_Signal', ''),
                                'support_level': bull_signals.get('Support_Level'),
                            }
                        })
                    if bear_score <= -3:
                        reversal_patterns.append({
                            'pattern_type': 'REVERSAL_BEARISH',
                            'timeframe': interval,
                            'direction': 'bearish',
                            'price_level': bear_signals.get('Current_Price'),
                            'upper_bound': bear_signals.get('Day_High'),
                            'lower_bound': bear_signals.get('Day_Low'),
                            'score': abs(bear_score),
                            'metadata': {
                                'verdict': bear_verdict,
                                'vwap': bear_signals.get('VWAP'),
                                'buying_exhausted': bear_signals.get('Buying_Exhausted', ''),
                                'lower_high': bear_signals.get('Lower_High', ''),
                                'volume_signal': bear_signals.get('Volume_Signal', ''),
                            }
                        })
                    if reversal_patterns:
                        db.upsert_detected_patterns(reversal_patterns)
                except Exception:
                    pass
                col_bull, col_bear = st.columns(2)
                for col, title, verdict, sigs, score_key, items in [
                    (col_bull, "### 🟢 Bullish Reversal", bull_verdict, bull_signals, 'Reversal_Score',
                     [('Selling Exhausted', 'Selling_Exhausted'), ('Higher Low', 'Higher_Low'),
                      ('Strong Bullish Candle', 'Strong_Bullish_Candle'), ('Volume', 'Volume_Signal'), ('Above VWAP', 'Above_VWAP')]),
                    (col_bear, "### 🔴 Bearish Reversal", bear_verdict, bear_signals, 'Bearish_Score',
                     [('Buying Exhausted', 'Buying_Exhausted'), ('Lower High', 'Lower_High'),
                      ('Strong Bearish Candle', 'Strong_Bearish_Candle'), ('Volume', 'Volume_Signal'), ('Below VWAP', 'Below_VWAP')])
                ]:
                    with col:
                        st.markdown(title)
                        if "STRONG" in verdict:
                            (st.success if "BUY" in verdict else st.error)(f"**{verdict}**")
                        elif "MODERATE" in verdict:
                            st.warning(f"**{verdict}**")
                        else:
                            st.info(f"**{verdict}**")
                        st.markdown(f"**Score: {sigs.get(score_key, 0)}/6**")
                        for label, key in items:
                            st.markdown(f"- {label}: {sigs.get(key, 'N/A')}")
                if bull_signals.get('VWAP'):
                    st.info(f"📊 **VWAP:** ₹{bull_signals.get('VWAP')} | **Day High:** ₹{bull_signals.get('Day_High', 'N/A')} | **Day Low:** ₹{bull_signals.get('Day_Low', 'N/A')}")
                with st.expander("📋 Entry Rules & Recommendations"):
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown("**🟢 Bullish Entry Rules:**")
                        entry_rules = ReversalDetector.get_entry_rules(bull_signals, bull_score)
                        for rule in entry_rules:
                            st.markdown(f"- {rule}")
                    with col_r2:
                        st.markdown("**🔴 Bearish Entry Rules:**")
                        if bear_score <= -4:
                            st.markdown("- 🎯 ENTRY: Buy PE at current level")
                            st.markdown(f"- 🛑 SL: Above recent high ({bear_signals.get('Day_High', 'N/A')})")
                            st.markdown("- 🎯 Target: Previous low / Nearest support")
                        elif bear_score <= -2.5:
                            st.markdown("- ⏳ WAIT: Confirmation pending")
                            st.markdown("- 📋 Checklist: Lower High + Strong Bearish Candle + Volume")
                        else:
                            st.markdown("- ❌ NO ENTRY: Bearish conditions not met")
                    st.markdown("---")
                    st.markdown("**🧠 Trading Psychology:**")
                    st.markdown("> *Missing a trade is 100x better than entering a wrong trade.*")
                    st.markdown("- Trade only after structure forms")
                    st.markdown("- No emotional entries")
                    st.markdown("- Fixed SL, fixed target")
                    st.markdown("- If trade missed → day closed")
            except Exception as e:
                st.warning(f"Reversal analysis unavailable: {str(e)}")
            st.markdown("---")
            st.markdown("## 📊 Triple POC + Future Swing Analysis")
            if poc_data_for_chart:
                st.markdown("### 🎯 Triple Point of Control (POC)")
                poc_table_data = []
                current_price_for_poc = df['close'].iloc[-1] if not df.empty else 0
                for poc_key, period_key in [('poc1', 'poc1'), ('poc2', 'poc2'), ('poc3', 'poc3')]:
                    poc = poc_data_for_chart.get(poc_key)
                    period = poc_data_for_chart.get('periods', {}).get(period_key, '')
                    if poc:
                        if current_price_for_poc > poc.get('upper_poc', 0):
                            position = "🟢 Above"
                            signal = "Bullish"
                        elif current_price_for_poc < poc.get('lower_poc', 0):
                            position = "🔴 Below"
                            signal = "Bearish"
                        else:
                            position = "🟡 Inside"
                            signal = "Neutral"
                        poc_table_data.append({
                            'POC': f"POC {poc_key[-1]} ({period})",
                            'Value': f"₹{poc.get('poc', 0):.2f}",
                            'Upper': f"₹{poc.get('upper_poc', 0):.2f}",
                            'Lower': f"₹{poc.get('lower_poc', 0):.2f}",
                            'Range': f"₹{poc.get('high', 0):.0f} - ₹{poc.get('low', 0):.0f}",
                            'Position': position,
                            'Signal': signal
                        })
                if poc_table_data:
                    poc_df = pd.DataFrame(poc_table_data)
                    def style_poc_signal(val):
                        if val == 'Bullish':
                            return 'background-color: #00ff8840; color: white'
                        elif val == 'Bearish':
                            return 'background-color: #ff444440; color: white'
                        else:
                            return 'background-color: #FFD70040; color: white'
                    styled_poc = poc_df.style.map(style_poc_signal, subset=['Signal'])
                    st.dataframe(styled_poc, use_container_width=True, hide_index=True)
                    st.markdown("""
                    **POC Interpretation:**
                    - **POC 1 (25)**: Short-term volume profile - intraday support/resistance
                    - **POC 2 (40)**: Medium-term volume profile - swing trading levels
                    - **POC 3 (100)**: Long-term volume profile - major support/resistance
                    - **Above POC**: Bullish bias - POC acts as support
                    - **Below POC**: Bearish bias - POC acts as resistance
                    - **Inside POC**: Neutral - price consolidating at high-volume zone
                    """)
            if swing_data_for_chart:
                st.markdown("### 🔄 Future Swing Projection")
                swings = swing_data_for_chart.get('swings', {})
                projection = swing_data_for_chart.get('projection')
                volume = swing_data_for_chart.get('volume', {})
                percentages = swing_data_for_chart.get('percentages', [])
                def _swing_card(color, title, h2, sub=""):
                    p = f'<p style="color: white; margin: 0;">{sub}</p>' if sub else ""
                    return f'<div style="background-color: {color}20; padding: 15px; border-radius: 10px; border: 2px solid {color};"><h4 style="color: {color}; margin: 0;">{title}</h4><h2 style="color: {color}; margin: 5px 0;">{h2}</h2>{p}</div>'
                direction = swings.get('direction', 'Unknown')
                dir_c = "#15dd7c" if direction == 'bullish' else "#eb7514"
                delta = volume.get('delta', 0)
                dc = "#15dd7c" if delta > 0 else "#eb7514"
                sc1, sc2, sc3 = st.columns(3)
                sc1.markdown(_swing_card(dir_c, "Current Direction", f"{'🟢' if direction=='bullish' else '🔴'} {direction.upper()}"), unsafe_allow_html=True)
                if projection:
                    tc = "#15dd7c" if projection['direction'] == 'bullish' else "#eb7514"
                    sc2.markdown(_swing_card(tc, "Projected Target", f"₹{projection['target']:.0f}", f"{projection['sign']}{projection['swing_pct']:.1f}%"), unsafe_allow_html=True)
                else:
                    sc2.info("Projection not available")
                sc3.markdown(_swing_card(dc, "Volume Delta", f"{'🟢' if delta>0 else '🔴'} {delta:+,.0f}", f"Buy: {volume.get('buy_volume',0):,.0f} | Sell: {volume.get('sell_volume',0):,.0f}"), unsafe_allow_html=True)
                if percentages:
                    st.markdown("### 📈 Historical Swing Percentages")
                    swing_pct_data = []
                    for i, pct in enumerate(percentages):
                        swing_pct_data.append({
                            'Swing': f"Swing {i+1}",
                            'Percentage': f"{pct:+.2f}%",
                            'Type': '🟢 Bullish' if pct > 0 else '🔴 Bearish'
                        })
                    avg_pct = sum(abs(p) for p in percentages) / len(percentages) if percentages else 0
                    swing_pct_data.append({
                        'Swing': '📊 Average',
                        'Percentage': f"{avg_pct:.2f}%",
                        'Type': 'Used for projection'
                    })
                    swing_pct_df = pd.DataFrame(swing_pct_data)
                    st.dataframe(swing_pct_df, use_container_width=True, hide_index=True)
                st.markdown("### 📍 Swing Levels")
                swing_levels_data = []
                last_high = swings.get('last_swing_high')
                last_low = swings.get('last_swing_low')
                if last_high:
                    swing_levels_data.append({
                        'Type': '🔴 Swing High',
                        'Value': f"₹{last_high['value']:.2f}",
                        'Index': last_high['index']
                    })
                if last_low:
                    swing_levels_data.append({
                        'Type': '🟢 Swing Low',
                        'Value': f"₹{last_low['value']:.2f}",
                        'Index': last_low['index']
                    })
                if swing_levels_data:
                    swing_levels_df = pd.DataFrame(swing_levels_data)
                    st.dataframe(swing_levels_df, use_container_width=True, hide_index=True)
                st.markdown("""
                **Swing Analysis Interpretation:**
                - **Swing High**: Resistance level where price reversed down
                - **Swing Low**: Support level where price reversed up
                - **Volume Delta**: Positive = more buying, Negative = more selling
                - **Projected Target**: Based on average of historical swing percentages
                """)
        else:
            st.error("No data available. Please check your API credentials and try again.")

    with col2:
        option_data = analyze_option_chain(selected_expiry, pivots, vob_data)
        if option_data and option_data.get('underlying'):
            underlying_price = option_data['underlying']
            df_summary = option_data['df_summary']
            expiry = option_data.get('expiry', selected_expiry)
            atm_strike = df_summary.loc[df_summary['Zone'] == 'ATM', 'Strike'].values[0] if df_summary is not None and 'Zone' in df_summary.columns and not df_summary[df_summary['Zone'] == 'ATM'].empty else underlying_price
            # Store option chain, ATM strike data, and orderbook to Supabase
            if df_summary is not None and len(df_summary) > 0 and expiry:
                try:
                    db.upsert_option_chain(df_summary, expiry, underlying_price, atm_strike)
                    db.upsert_atm_strike_data(df_summary, expiry, underlying_price, atm_strike)
                    # Store orderbook depth data
                    orderbook_entries = []
                    for _, row in df_summary.iterrows():
                        orderbook_entries.append({
                            'strike': float(row['Strike']),
                            'bid_qty_ce': int(row.get('bidQty_CE', 0) or 0),
                            'ask_qty_ce': int(row.get('askQty_CE', 0) or 0),
                            'bid_qty_pe': int(row.get('bidQty_PE', 0) or 0),
                            'ask_qty_pe': int(row.get('askQty_PE', 0) or 0),
                            'pressure': float(row.get('BidAskPressure', 0) or 0),
                            'bias': str(row.get('PressureBias', '') or '')
                        })
                    if orderbook_entries:
                        db.upsert_orderbook(orderbook_entries, expiry)
                except Exception:
                    pass
            if enable_signals and not df.empty and df_summary is not None and len(df_summary) > 0:
                check_trading_signals(df, pivot_settings, df_summary, underlying_price, pivot_proximity)
            if df_summary is not None and len(df_summary) > 0:
                check_atm_verdict_alert(df_summary, underlying_price)
        else:
            option_data = None

    if option_data and option_data.get('underlying'):
        st.markdown("---")
        st.header("📊 Options Chain Analysis")
        st.markdown("## Open Interest Change (in Lakhs)")
        oi_col1, oi_col2 = st.columns(2)
        with oi_col1:
            st.metric("CALL ΔOI", f"{option_data['total_ce_change']:+.1f}L", delta_color="inverse")
        with oi_col2:
            st.metric("PUT ΔOI", f"{option_data['total_pe_change']:+.1f}L", delta_color="normal")
        st.markdown("## Option Chain Bias Summary")
        if option_data.get('styled_df') is not None:
            st.dataframe(option_data['styled_df'], use_container_width=True)
        st.markdown("---")
        st.markdown("## 📈 HTF Support & Resistance Levels")
        sr_data = option_data.get('sr_data', [])
        max_pain_strike = option_data.get('max_pain_strike')
        # Store max pain to Supabase
        if max_pain_strike and option_data.get('expiry'):
            try:
                db.upsert_max_pain(option_data['expiry'], max_pain_strike, option_data.get('underlying'))
            except Exception:
                pass
        if sr_data:
            support_data = [d for d in sr_data if '🟢' in d['Type'] or '🎯' in d['Type']]
            resistance_data = [d for d in sr_data if '🔴' in d['Type']]
            for col, title, data in zip(st.columns(2),
                ["### 🟢 SUPPORT LEVELS", "### 🔴 RESISTANCE LEVELS"],
                [support_data, resistance_data]):
                with col:
                    st.markdown(title)
                    if data:
                        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                    else:
                        st.info("No levels identified")
            if max_pain_strike:
                st.info(f"🎯 **Max Pain Level:** ₹{max_pain_strike:.0f} - Price magnet at expiry")
        st.markdown("---")
        st.markdown("## 🔍 Option Chain Deep Analysis (ATM ± 5)")
        try:
            sa_df_summary = option_data.get('df_summary')
            sa_underlying = option_data.get('underlying')
            if sa_df_summary is not None and sa_underlying:
                sa_result = analyze_strike_activity(sa_df_summary, sa_underlying)
                if sa_result:
                    # Send Telegram signal (with cooldown to avoid spam)
                    if 'last_chain_signal_time' not in st.session_state:
                        st.session_state.last_chain_signal_time = None
                    ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                    should_send = st.session_state.last_chain_signal_time is None or \
                        (ist_now - st.session_state.last_chain_signal_time).total_seconds() > 300
                    if should_send:
                        try:
                            send_option_chain_signal(sa_result, sa_underlying)
                            st.session_state.last_chain_signal_time = ist_now
                        except Exception:
                            pass
                    st.session_state._sa_result = sa_result
                    analysis_df = sa_result['analysis_df']
                    # Market Bias Banner
                    bias = sa_result['market_bias']
                    conf = sa_result['confidence']
                    if 'Bullish' in bias:
                        bias_color = '#00ff88'
                    elif 'Bearish' in bias:
                        bias_color = '#ff4444'
                    else:
                        bias_color = '#FFD700'
                    st.markdown(f"""
                    <div style="background:{bias_color}20;padding:20px;border-radius:12px;border:2px solid {bias_color};text-align:center;margin-bottom:15px;">
                        <h2 style="color:{bias_color};margin:0;">Market Bias: {bias}</h2>
                        <h3 style="color:white;margin:5px 0;">Confidence: {conf}%</h3>
                        <div style="background:#33333380;border-radius:10px;height:12px;margin:8px auto;max-width:400px;">
                            <div style="background:{bias_color};border-radius:10px;height:12px;width:{conf}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Bias signals
                    for sig in sa_result['bias_signals']:
                        st.caption(f"• {sig}")
                    # Top Resistance & Support
                    sa_col1, sa_col2 = st.columns(2)
                    with sa_col1:
                        st.markdown("### 🔴 Top 3 Resistance Levels (Call Side)")
                        if not sa_result['top_resistance'].empty:
                            res_display = sa_result['top_resistance'].copy()
                            res_display['Strike'] = res_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                            res_display['CE_OI'] = (res_display['CE_OI'] / 100000).round(2).astype(str) + 'L'
                            res_display['CE_ChgOI'] = (res_display['CE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            res_display.columns = ['Strike', 'Classification', 'Strength', 'OI', 'ChgOI', 'Activity']
                            def style_res(row):
                                s = row['Strength']
                                if s == 'Strong':
                                    return ['background-color:#ff444440;color:white'] * len(row)
                                elif s == 'Breaking':
                                    return ['background-color:#FFD70040;color:white'] * len(row)
                                return [''] * len(row)
                            styled_res = res_display.style.apply(style_res, axis=1)
                            st.dataframe(styled_res, use_container_width=True, hide_index=True)
                        else:
                            st.info("No resistance levels found")
                    with sa_col2:
                        st.markdown("### 🟢 Top 3 Support Levels (Put Side)")
                        if not sa_result['top_support'].empty:
                            sup_display = sa_result['top_support'].copy()
                            sup_display['Strike'] = sup_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                            sup_display['PE_OI'] = (sup_display['PE_OI'] / 100000).round(2).astype(str) + 'L'
                            sup_display['PE_ChgOI'] = (sup_display['PE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            sup_display.columns = ['Strike', 'Classification', 'Strength', 'OI', 'ChgOI', 'Activity']
                            def style_sup(row):
                                s = row['Strength']
                                if s == 'Strong':
                                    return ['background-color:#00ff8840;color:white'] * len(row)
                                elif s == 'Breaking':
                                    return ['background-color:#FFD70040;color:white'] * len(row)
                                return [''] * len(row)
                            styled_sup = sup_display.style.apply(style_sup, axis=1)
                            st.dataframe(styled_sup, use_container_width=True, hide_index=True)
                        else:
                            st.info("No support levels found")
                    # Call Capping & Put Support Zones
                    st.markdown("### 📊 Strike-wise Call & Put Classification")
                    class_display = analysis_df[['Strike', 'Zone', 'Call_Class', 'Call_Activity', 'Put_Class', 'Put_Activity']].copy()
                    class_display['Strike'] = class_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                    def style_class(row):
                        styles = [''] * len(row)
                        call_idx = class_display.columns.get_loc('Call_Class')
                        put_idx = class_display.columns.get_loc('Put_Class')
                        if 'Strong' in str(row.iloc[call_idx]):
                            styles[call_idx] = 'background-color:#ff444440;color:white;font-weight:bold'
                        elif 'Breakout' in str(row.iloc[call_idx]):
                            styles[call_idx] = 'background-color:#FFD70040;color:white;font-weight:bold'
                        if 'Strong' in str(row.iloc[put_idx]):
                            styles[put_idx] = 'background-color:#00ff8840;color:white;font-weight:bold'
                        elif 'Breakdown' in str(row.iloc[put_idx]):
                            styles[put_idx] = 'background-color:#FFD70040;color:white;font-weight:bold'
                        return styles
                    styled_class = class_display.style.apply(style_class, axis=1)
                    st.dataframe(styled_class, use_container_width=True, hide_index=True)
                    # Trapped Writers & Breakout/Breakdown
                    trap_col1, trap_col2 = st.columns(2)
                    with trap_col1:
                        st.markdown("### ⚠️ Trapped Call Writers")
                        if not sa_result['trapped_call_writers'].empty:
                            tc = sa_result['trapped_call_writers'].copy()
                            tc['Strike'] = tc['Strike'].apply(lambda x: f"₹{x:.0f}")
                            tc['CE_OI'] = (tc['CE_OI'] / 100000).round(2).astype(str) + 'L'
                            tc['CE_ChgOI'] = (tc['CE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            tc.columns = ['Strike', 'OI', 'ChgOI']
                            st.dataframe(tc, use_container_width=True, hide_index=True)
                            st.caption("Price above strike + OI falling = writers buying back (bullish)")
                        else:
                            st.success("No trapped call writers")
                    with trap_col2:
                        st.markdown("### ⚠️ Trapped Put Writers")
                        if not sa_result['trapped_put_writers'].empty:
                            tp = sa_result['trapped_put_writers'].copy()
                            tp['Strike'] = tp['Strike'].apply(lambda x: f"₹{x:.0f}")
                            tp['PE_OI'] = (tp['PE_OI'] / 100000).round(2).astype(str) + 'L'
                            tp['PE_ChgOI'] = (tp['PE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            tp.columns = ['Strike', 'OI', 'ChgOI']
                            st.dataframe(tp, use_container_width=True, hide_index=True)
                            st.caption("Price below strike + OI falling = writers buying back (bearish)")
                        else:
                            st.success("No trapped put writers")
                    bk_col1, bk_col2 = st.columns(2)
                    with bk_col1:
                        st.markdown("### 🚀 Breakout Zones")
                        if not sa_result['breakout_zones'].empty:
                            bz = sa_result['breakout_zones'].copy()
                            bz['Strike'] = bz['Strike'].apply(lambda x: f"₹{x:.0f}")
                            bz['CE_OI'] = (bz['CE_OI'] / 100000).round(2).astype(str) + 'L'
                            bz['CE_ChgOI'] = (bz['CE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            bz.columns = ['Strike', 'OI', 'ChgOI']
                            st.dataframe(bz, use_container_width=True, hide_index=True)
                            st.caption("Call OI falling + price rising = resistance breaking")
                        else:
                            st.info("No breakout zones detected")
                    with bk_col2:
                        st.markdown("### 💥 Breakdown Zones")
                        if not sa_result['breakdown_zones'].empty:
                            bd = sa_result['breakdown_zones'].copy()
                            bd['Strike'] = bd['Strike'].apply(lambda x: f"₹{x:.0f}")
                            bd['PE_OI'] = (bd['PE_OI'] / 100000).round(2).astype(str) + 'L'
                            bd['PE_ChgOI'] = (bd['PE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            bd.columns = ['Strike', 'OI', 'ChgOI']
                            st.dataframe(bd, use_container_width=True, hide_index=True)
                            st.caption("Put OI falling + price falling = support breaking")
                        else:
                            st.info("No breakdown zones detected")
                else:
                    st.warning("Unable to analyze option chain. Check data.")
        except Exception as e:
            st.warning(f"Deep analysis unavailable: {str(e)}")
        st.markdown("---")
        st.markdown("## 📊 PCR Analysis - Time Series (ATM ± 2)")
        def create_pcr_chart(history_df, col_name, color, title_prefix):
            """Helper to create individual PCR chart - col_name is now just strike price"""
            if col_name and col_name in history_df.columns:
                strike_val = col_name
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df['time'],
                    y=history_df[col_name],
                    mode='lines+markers',
                    name=f'₹{strike_val}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    fill='tozeroy',
                    fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}'
                ))
                fig.add_hline(y=1.0, line_dash="dash", line_color="white", line_width=1)
                fig.add_hline(y=1.2, line_dash="dot", line_color="#00ff88", line_width=1)
                fig.add_hline(y=0.7, line_dash="dot", line_color="#ff4444", line_width=1)
                current_pcr = history_df[col_name].iloc[-1] if len(history_df) > 0 else 0
                fig.update_layout(
                    title=f"{title_prefix}<br>₹{strike_val}<br>PCR: {current_pcr:.2f}",
                    template='plotly_dark',
                    height=280,
                    showlegend=False,
                    margin=dict(l=10, r=10, t=70, b=30),
                    xaxis=dict(tickformat='%H:%M', title=''),
                    yaxis=dict(title='PCR'),
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e'
                )
                return fig, current_pcr
            return None, 0
        pcr_data_available = False
        pcr_df = None
        df_summary = option_data.get('df_summary') if option_data else None
        if df_summary is not None and 'Zone' in df_summary.columns and 'PCR' in df_summary.columns:
            try:
                atm_idx = df_summary[df_summary['Zone'] == 'ATM'].index
                if len(atm_idx) > 0:
                    atm_pos = df_summary.index.get_loc(atm_idx[0])
                    start_idx = max(0, atm_pos - 2)
                    end_idx = min(len(df_summary), atm_pos + 3)
                    pcr_extract_cols = ['Strike', 'Zone', 'PCR', 'PCR_Signal',
                                                                   'openInterest_CE', 'openInterest_PE']
                    for _chg_col in ['changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                                      'lastPrice_CE', 'lastPrice_PE']:
                        if _chg_col in df_summary.columns:
                            pcr_extract_cols.append(_chg_col)
                    pcr_df = df_summary.iloc[start_idx:end_idx][pcr_extract_cols].copy()
                    if not pcr_df.empty:
                        pcr_data_available = True
                        st.session_state.pcr_last_valid_data = pcr_df.copy()
                        ist = pytz.timezone('Asia/Kolkata')
                        current_time = datetime.now(ist)
                        pcr_entry = {'time': current_time}
                        for _, row in pcr_df.iterrows():
                            strike_label = str(int(row['Strike']))
                            pcr_entry[strike_label] = row['PCR']
                        current_strikes = pcr_df['Strike'].tolist()
                        st.session_state.pcr_current_strikes = [int(s) for s in current_strikes]
                        should_add = True
                        if st.session_state.pcr_history:
                            last_entry = st.session_state.pcr_history[-1]
                            time_diff = (current_time - last_entry['time']).total_seconds()
                            if time_diff < 30:
                                should_add = False
                        if should_add:
                            st.session_state.pcr_history.append(pcr_entry)
                            if len(st.session_state.pcr_history) > 200:
                                st.session_state.pcr_history = st.session_state.pcr_history[-200:]
                            # Persist PCR to Supabase
                            try:
                                expiry_for_pcr = option_data.get('expiry', selected_expiry) if option_data else selected_expiry
                                pcr_records = []
                                for _, row in pcr_df.iterrows():
                                    pcr_records.append({
                                        'timestamp': current_time,
                                        'expiry': expiry_for_pcr,
                                        'strike': float(row['Strike']),
                                        'atm_strike': float(atm_strike) if 'atm_strike' in dir() else 0,
                                        'pcr': float(row['PCR']) if pd.notna(row['PCR']) else 0,
                                        'oi_ce': int(row.get('openInterest_CE', 0) or 0),
                                        'oi_pe': int(row.get('openInterest_PE', 0) or 0),
                                    })
                                if pcr_records:
                                    db.upsert_pcr_history(pcr_records)
                            except Exception:
                                pass
                            # Collect OI history for ATM ± 2 strikes
                            oi_entry = {'time': current_time}
                            chgoi_entry = {'time': current_time}
                            for _, row in pcr_df.iterrows():
                                strike_label = str(int(row['Strike']))
                                oi_entry[f'{strike_label}_CE'] = int(row.get('openInterest_CE', 0) or 0)
                                oi_entry[f'{strike_label}_PE'] = int(row.get('openInterest_PE', 0) or 0)
                                oi_entry[f'{strike_label}_CE_LTP'] = float(row.get('lastPrice_CE', 0) or 0)
                                oi_entry[f'{strike_label}_PE_LTP'] = float(row.get('lastPrice_PE', 0) or 0)
                                chgoi_entry[f'{strike_label}_CE'] = int(row.get('changeinOpenInterest_CE', 0) or 0)
                                chgoi_entry[f'{strike_label}_PE'] = int(row.get('changeinOpenInterest_PE', 0) or 0)
                            st.session_state.oi_history.append(oi_entry)
                            st.session_state.oi_last_valid_data = pcr_df.copy()
                            st.session_state.oi_current_strikes = [int(s) for s in current_strikes]
                            if len(st.session_state.oi_history) > 200:
                                st.session_state.oi_history = st.session_state.oi_history[-200:]
                            st.session_state.chgoi_history.append(chgoi_entry)
                            st.session_state.chgoi_last_valid_data = pcr_df.copy()
                            st.session_state.chgoi_current_strikes = [int(s) for s in current_strikes]
                            if len(st.session_state.chgoi_history) > 200:
                                st.session_state.chgoi_history = st.session_state.chgoi_history[-200:]
            except Exception as e:
                st.caption(f"⚠️ Current fetch issue: {str(e)[:50]}...")
        if len(st.session_state.pcr_history) > 0:
            try:
                history_df = pd.DataFrame(st.session_state.pcr_history)
                current_strikes = getattr(st.session_state, 'pcr_current_strikes', [])
                if not current_strikes and st.session_state.pcr_last_valid_data is not None:
                    current_strikes = [int(s) for s in st.session_state.pcr_last_valid_data['Strike'].tolist()]
                current_strikes = sorted(current_strikes)
                pcr_col1, pcr_col2, pcr_col3, pcr_col4, pcr_col5 = st.columns(5)
                def display_pcr_with_signal(container, fig, pcr_val):
                    if fig:
                        container.plotly_chart(fig, use_container_width=True)
                        if pcr_val > 1.2:
                            container.success("Bullish")
                        elif pcr_val < 0.7:
                            container.error("Bearish")
                        else:
                            container.warning("Neutral")
                zone_info = {}
                zone_df = pcr_df if pcr_df is not None else st.session_state.pcr_last_valid_data
                if zone_df is not None:
                    for _, row in zone_df.iterrows():
                        zone_info[int(row['Strike'])] = row['Zone']
                position_labels = ['🟣 ITM-2', '🟣 ITM-1', '🟡 ATM', '🔵 OTM+1', '🔵 OTM+2']
                position_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']
                columns = [pcr_col1, pcr_col2, pcr_col3, pcr_col4, pcr_col5]
                for i, col in enumerate(columns):
                    with col:
                        if i < len(current_strikes):
                            strike = current_strikes[i]
                            strike_col = str(strike)
                            zone = zone_info.get(strike, position_labels[i].split()[-1])
                            if strike_col in history_df.columns:
                                fig, pcr_val = create_pcr_chart(history_df, strike_col, position_colors[i], f'{position_labels[i]}')
                                display_pcr_with_signal(st, fig, pcr_val)
                            else:
                                st.info(f"₹{strike} - Building history...")
                        else:
                            st.info(f"{position_labels[i]} N/A")
                st.markdown("### Current PCR Values")
                display_df = pcr_df if pcr_df is not None else st.session_state.pcr_last_valid_data
                if display_df is not None:
                    pcr_display = display_df[['Strike', 'Zone', 'PCR', 'PCR_Signal']].copy()
                    pcr_display['CE OI (L)'] = (display_df['openInterest_CE'] / 100000).round(2)
                    pcr_display['PE OI (L)'] = (display_df['openInterest_PE'] / 100000).round(2)
                    st.dataframe(pcr_display, use_container_width=True, hide_index=True)
                col_info1, col_info2 = st.columns([3, 1])
                with col_info1:
                    status = "🟢 Live" if pcr_data_available else "🟡 Using cached history"
                    st.caption(f"{status} | 📈 {len(st.session_state.pcr_history)} data points | History preserved on refresh failures")
                with col_info2:
                    if st.button("🗑️ Clear History"):
                        st.session_state.pcr_history = []
                        st.session_state.pcr_last_valid_data = None
                        st.rerun()
            except Exception as e:
                st.warning(f"Error displaying PCR charts: {str(e)}")
        else:
            st.info("📊 PCR history will build up as the app refreshes. Please wait for data collection...")
        st.markdown("---")
        st.markdown("## 📊 OI Analysis - Time Series (ATM ± 2)")
        if len(st.session_state.oi_history) > 0:
            try:
                oi_history_df = pd.DataFrame(st.session_state.oi_history)
                oi_strikes = st.session_state.oi_current_strikes or []
                if not oi_strikes and st.session_state.oi_last_valid_data is not None:
                    oi_strikes = [int(s) for s in st.session_state.oi_last_valid_data['Strike'].tolist()]
                oi_strikes = sorted(oi_strikes)
                zone_df_oi = pcr_df if pcr_df is not None else st.session_state.oi_last_valid_data
                oi_zone_info = {}
                if zone_df_oi is not None:
                    for _, row in zone_df_oi.iterrows():
                        oi_zone_info[int(row['Strike'])] = row['Zone']
                oi_position_labels = ['ITM-2', 'ITM-1', 'ATM', 'OTM+1', 'OTM+2']
                oi_colors_ce = ['#ff6666', '#ff9999', '#ffcc00', '#66bbff', '#3399ff']
                oi_colors_pe = ['#cc44cc', '#aa66aa', '#ff8800', '#44dd88', '#22bb66']
                if len(oi_strikes) >= 3:
                    oi_col1, oi_col2 = st.columns(2)
                    # Combined OI chart - all strikes CE vs PE
                    with oi_col1:
                        fig_ce = go.Figure()
                        for i, strike in enumerate(oi_strikes):
                            ce_col = f'{strike}_CE'
                            if ce_col in oi_history_df.columns:
                                label = oi_position_labels[i] if i < len(oi_position_labels) else f'Strike {i}'
                                fig_ce.add_trace(go.Scatter(
                                    x=oi_history_df['time'],
                                    y=oi_history_df[ce_col] / 100000,
                                    mode='lines+markers',
                                    name=f'₹{strike} ({label})',
                                    line=dict(width=2),
                                    marker=dict(size=3),
                                ))
                        fig_ce.update_layout(
                            title='Call OI (ATM ± 2)',
                            template='plotly_dark',
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=30),
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='OI (Lakhs)'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_ce, use_container_width=True)
                    with oi_col2:
                        fig_pe = go.Figure()
                        for i, strike in enumerate(oi_strikes):
                            pe_col = f'{strike}_PE'
                            if pe_col in oi_history_df.columns:
                                label = oi_position_labels[i] if i < len(oi_position_labels) else f'Strike {i}'
                                fig_pe.add_trace(go.Scatter(
                                    x=oi_history_df['time'],
                                    y=oi_history_df[pe_col] / 100000,
                                    mode='lines+markers',
                                    name=f'₹{strike} ({label})',
                                    line=dict(width=2),
                                    marker=dict(size=3),
                                ))
                        fig_pe.update_layout(
                            title='Put OI (ATM ± 2)',
                            template='plotly_dark',
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=30),
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='OI (Lakhs)'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_pe, use_container_width=True)
                    # Per-strike CE vs PE comparison charts
                    st.markdown("### Per-Strike Call vs Put OI")
                    oi_strike_cols = st.columns(min(len(oi_strikes), 5))
                    for i, strike in enumerate(oi_strikes):
                        if i >= len(oi_strike_cols):
                            break
                        ce_col = f'{strike}_CE'
                        pe_col = f'{strike}_PE'
                        with oi_strike_cols[i]:
                            label = oi_position_labels[i] if i < len(oi_position_labels) else f'Strike {i}'
                            fig_strike = go.Figure()
                            if ce_col in oi_history_df.columns:
                                fig_strike.add_trace(go.Scatter(
                                    x=oi_history_df['time'],
                                    y=oi_history_df[ce_col] / 100000,
                                    mode='lines+markers',
                                    name='Call OI',
                                    line=dict(color='#ff4444', width=2),
                                    marker=dict(size=3),
                                ))
                            if pe_col in oi_history_df.columns:
                                fig_strike.add_trace(go.Scatter(
                                    x=oi_history_df['time'],
                                    y=oi_history_df[pe_col] / 100000,
                                    mode='lines+markers',
                                    name='Put OI',
                                    line=dict(color='#00cc66', width=2),
                                    marker=dict(size=3),
                                ))
                            current_ce = oi_history_df[ce_col].iloc[-1] / 100000 if ce_col in oi_history_df.columns and len(oi_history_df) > 0 else 0
                            current_pe = oi_history_df[pe_col].iloc[-1] / 100000 if pe_col in oi_history_df.columns and len(oi_history_df) > 0 else 0
                            fig_strike.update_layout(
                                title=f'{label}<br>₹{strike}<br>CE: {current_ce:.1f}L | PE: {current_pe:.1f}L',
                                template='plotly_dark',
                                height=280,
                                showlegend=True,
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=9)),
                                margin=dict(l=10, r=10, t=80, b=30),
                                xaxis=dict(tickformat='%H:%M', title=''),
                                yaxis=dict(title='OI (L)'),
                                plot_bgcolor='#1e1e1e',
                                paper_bgcolor='#1e1e1e'
                            )
                            st.plotly_chart(fig_strike, use_container_width=True)
                            # Analyze OI trend over time for support/resistance strength
                            ce_series = oi_history_df[ce_col] if ce_col in oi_history_df.columns else pd.Series([0])
                            pe_series = oi_history_df[pe_col] if pe_col in oi_history_df.columns else pd.Series([0])
                            if len(ce_series) >= 3:
                                ce_first, ce_last = ce_series.iloc[0], ce_series.iloc[-1]
                                pe_first, pe_last = pe_series.iloc[0], pe_series.iloc[-1]
                                ce_change = ce_last - ce_first
                                pe_change = pe_last - pe_first
                                ce_pct = (ce_change / ce_first * 100) if ce_first > 0 else 0
                                pe_pct = (pe_change / pe_first * 100) if pe_first > 0 else 0
                                # Support signal (PE OI trend)
                                if pe_change > 0:
                                    st.success(f"Support Building (PE +{pe_pct:.1f}%)")
                                elif pe_change < 0:
                                    st.error(f"Support Weakening (PE {pe_pct:.1f}%)")
                                else:
                                    st.info("Support Flat")
                                # Resistance signal (CE OI trend)
                                if ce_change > 0:
                                    st.error(f"Resistance Building (CE +{ce_pct:.1f}%)")
                                elif ce_change < 0:
                                    st.success(f"Resistance Weakening (CE {ce_pct:.1f}%)")
                                else:
                                    st.info("Resistance Flat")
                                # Overall verdict: compare CE vs PE OI difference
                                oi_diff = (current_pe - current_ce) * 100000  # back to absolute
                                oi_diff_lakhs = abs(current_pe - current_ce)
                                pe_ce_ratio = current_pe / current_ce if current_ce > 0 else 0
                                if current_pe > current_ce:
                                    if pe_ce_ratio >= 2.0:
                                        st.markdown(f'<div style="background:#00ff8840;padding:8px;border-radius:8px;border-left:4px solid #00ff88;"><b style="color:#00ff88;">STRONG SUPPORT</b> | PE {current_pe:.1f}L vs CE {current_ce:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {pe_ce_ratio:.1f}x</div>', unsafe_allow_html=True)
                                    elif pe_ce_ratio >= 1.3:
                                        st.markdown(f'<div style="background:#00cc6640;padding:8px;border-radius:8px;border-left:4px solid #00cc66;"><b style="color:#00cc66;">MODERATE SUPPORT</b> | PE {current_pe:.1f}L vs CE {current_ce:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {pe_ce_ratio:.1f}x</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div style="background:#88aa4440;padding:8px;border-radius:8px;border-left:4px solid #88aa44;"><b style="color:#88aa44;">WEAK SUPPORT</b> | PE {current_pe:.1f}L vs CE {current_ce:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {pe_ce_ratio:.1f}x</div>', unsafe_allow_html=True)
                                elif current_ce > current_pe:
                                    ce_pe_ratio = current_ce / current_pe if current_pe > 0 else 0
                                    if ce_pe_ratio >= 2.0:
                                        st.markdown(f'<div style="background:#ff444440;padding:8px;border-radius:8px;border-left:4px solid #ff4444;"><b style="color:#ff4444;">STRONG RESISTANCE</b> | CE {current_ce:.1f}L vs PE {current_pe:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {ce_pe_ratio:.1f}x</div>', unsafe_allow_html=True)
                                    elif ce_pe_ratio >= 1.3:
                                        st.markdown(f'<div style="background:#cc444440;padding:8px;border-radius:8px;border-left:4px solid #cc4444;"><b style="color:#cc4444;">MODERATE RESISTANCE</b> | CE {current_ce:.1f}L vs PE {current_pe:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {ce_pe_ratio:.1f}x</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div style="background:#aa664440;padding:8px;border-radius:8px;border-left:4px solid #aa6644;"><b style="color:#aa6644;">WEAK RESISTANCE</b> | CE {current_ce:.1f}L vs PE {current_pe:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {ce_pe_ratio:.1f}x</div>', unsafe_allow_html=True)
                                else:
                                    st.warning("Balanced (CE = PE)")
                                # Long/Short Building/Covering analysis based on OI + LTP trend
                                ce_ltp_col = f'{strike}_CE_LTP'
                                pe_ltp_col = f'{strike}_PE_LTP'
                                ce_has_ltp = ce_ltp_col in oi_history_df.columns and len(oi_history_df) >= 3
                                pe_has_ltp = pe_ltp_col in oi_history_df.columns and len(oi_history_df) >= 3
                                if ce_has_ltp:
                                    ce_ltp_first = oi_history_df[ce_ltp_col].iloc[0]
                                    ce_ltp_last = oi_history_df[ce_ltp_col].iloc[-1]
                                    ce_ltp_change = ce_ltp_last - ce_ltp_first
                                    ce_ltp_pct = (ce_ltp_change / ce_ltp_first * 100) if ce_ltp_first > 0 else 0
                                    # CE: OI↑+Price↑=Long Building, OI↑+Price↓=Short Building, OI↓+Price↑=Short Covering, OI↓+Price↓=Long Unwinding
                                    if ce_change > 0 and ce_ltp_change > 0:
                                        st.markdown(f'<div style="background:#ff880040;padding:6px;border-radius:6px;border-left:3px solid #ff8800;font-size:13px;"><b style="color:#ff8800;">CE: LONG BUILDING</b> | OI +{ce_pct:.1f}% | LTP +{ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif ce_change > 0 and ce_ltp_change <= 0:
                                        st.markdown(f'<div style="background:#ff444440;padding:6px;border-radius:6px;border-left:3px solid #ff4444;font-size:13px;"><b style="color:#ff4444;">CE: SHORT BUILDING</b> | OI +{ce_pct:.1f}% | LTP {ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif ce_change < 0 and ce_ltp_change > 0:
                                        st.markdown(f'<div style="background:#00ff8840;padding:6px;border-radius:6px;border-left:3px solid #00ff88;font-size:13px;"><b style="color:#00ff88;">CE: SHORT COVERING</b> | OI {ce_pct:.1f}% | LTP +{ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif ce_change < 0 and ce_ltp_change <= 0:
                                        st.markdown(f'<div style="background:#88888840;padding:6px;border-radius:6px;border-left:3px solid #888888;font-size:13px;"><b style="color:#888888;">CE: LONG UNWINDING</b> | OI {ce_pct:.1f}% | LTP {ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                if pe_has_ltp:
                                    pe_ltp_first = oi_history_df[pe_ltp_col].iloc[0]
                                    pe_ltp_last = oi_history_df[pe_ltp_col].iloc[-1]
                                    pe_ltp_change = pe_ltp_last - pe_ltp_first
                                    pe_ltp_pct = (pe_ltp_change / pe_ltp_first * 100) if pe_ltp_first > 0 else 0
                                    # PE: OI↑+Price↑=Long Building, OI↑+Price↓=Short Building, OI↓+Price↓=Short Covering, OI↓+Price↑=Long Unwinding
                                    if pe_change > 0 and pe_ltp_change > 0:
                                        st.markdown(f'<div style="background:#ff444440;padding:6px;border-radius:6px;border-left:3px solid #ff4444;font-size:13px;"><b style="color:#ff4444;">PE: LONG BUILDING</b> | OI +{pe_pct:.1f}% | LTP +{pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif pe_change > 0 and pe_ltp_change <= 0:
                                        st.markdown(f'<div style="background:#00ff8840;padding:6px;border-radius:6px;border-left:3px solid #00ff88;font-size:13px;"><b style="color:#00ff88;">PE: SHORT BUILDING</b> | OI +{pe_pct:.1f}% | LTP {pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif pe_change < 0 and pe_ltp_change < 0:
                                        st.markdown(f'<div style="background:#00cc6640;padding:6px;border-radius:6px;border-left:3px solid #00cc66;font-size:13px;"><b style="color:#00cc66;">PE: SHORT COVERING</b> | OI {pe_pct:.1f}% | LTP {pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif pe_change < 0 and pe_ltp_change >= 0:
                                        st.markdown(f'<div style="background:#88888840;padding:6px;border-radius:6px;border-left:3px solid #888888;font-size:13px;"><b style="color:#888888;">PE: LONG UNWINDING</b> | OI {pe_pct:.1f}% | LTP +{pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                            else:
                                oi_diff_lakhs = abs(current_pe - current_ce)
                                if current_pe > current_ce:
                                    st.success(f"Support > Resistance (Diff: {oi_diff_lakhs:.1f}L)")
                                elif current_ce > current_pe:
                                    st.error(f"Resistance > Support (Diff: {oi_diff_lakhs:.1f}L)")
                                else:
                                    st.warning("Neutral")
                else:
                    st.info("Waiting for ATM ± 2 strike data...")
                oi_info1, oi_info2 = st.columns([3, 1])
                with oi_info1:
                    st.caption(f"📈 {len(st.session_state.oi_history)} data points | OI values in Lakhs")
                with oi_info2:
                    if st.button("🗑️ Clear OI History"):
                        st.session_state.oi_history = []
                        st.session_state.oi_last_valid_data = None
                        st.rerun()
            except Exception as e:
                st.warning(f"Error displaying OI charts: {str(e)}")
        else:
            st.info("📊 OI history will build up as the app refreshes. Please wait for data collection...")
        st.markdown("---")
        st.markdown("## 📊 Change in OI Analysis - Time Series (ATM ± 2)")
        if len(st.session_state.chgoi_history) > 0:
            try:
                chgoi_history_df = pd.DataFrame(st.session_state.chgoi_history)
                chgoi_strikes = st.session_state.chgoi_current_strikes or []
                if not chgoi_strikes and st.session_state.chgoi_last_valid_data is not None:
                    chgoi_strikes = [int(s) for s in st.session_state.chgoi_last_valid_data['Strike'].tolist()]
                chgoi_strikes = sorted(chgoi_strikes)
                zone_df_chgoi = pcr_df if pcr_df is not None else st.session_state.chgoi_last_valid_data
                chgoi_zone_info = {}
                if zone_df_chgoi is not None:
                    for _, row in zone_df_chgoi.iterrows():
                        chgoi_zone_info[int(row['Strike'])] = row['Zone']
                chgoi_position_labels = ['ITM-2', 'ITM-1', 'ATM', 'OTM+1', 'OTM+2']
                if len(chgoi_strikes) >= 3:
                    chgoi_col1, chgoi_col2 = st.columns(2)
                    with chgoi_col1:
                        fig_chg_ce = go.Figure()
                        for i, strike in enumerate(chgoi_strikes):
                            ce_col = f'{strike}_CE'
                            if ce_col in chgoi_history_df.columns:
                                label = chgoi_position_labels[i] if i < len(chgoi_position_labels) else f'Strike {i}'
                                fig_chg_ce.add_trace(go.Scatter(
                                    x=chgoi_history_df['time'],
                                    y=chgoi_history_df[ce_col] / 1000,
                                    mode='lines+markers',
                                    name=f'₹{strike} ({label})',
                                    line=dict(width=2),
                                    marker=dict(size=3),
                                ))
                        fig_chg_ce.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
                        fig_chg_ce.update_layout(
                            title='Change in Call OI (ATM ± 2)',
                            template='plotly_dark',
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=30),
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='Chg OI (K)'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_chg_ce, use_container_width=True)
                    with chgoi_col2:
                        fig_chg_pe = go.Figure()
                        for i, strike in enumerate(chgoi_strikes):
                            pe_col = f'{strike}_PE'
                            if pe_col in chgoi_history_df.columns:
                                label = chgoi_position_labels[i] if i < len(chgoi_position_labels) else f'Strike {i}'
                                fig_chg_pe.add_trace(go.Scatter(
                                    x=chgoi_history_df['time'],
                                    y=chgoi_history_df[pe_col] / 1000,
                                    mode='lines+markers',
                                    name=f'₹{strike} ({label})',
                                    line=dict(width=2),
                                    marker=dict(size=3),
                                ))
                        fig_chg_pe.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
                        fig_chg_pe.update_layout(
                            title='Change in Put OI (ATM ± 2)',
                            template='plotly_dark',
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=30),
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='Chg OI (K)'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_chg_pe, use_container_width=True)
                    # Per-strike Change OI comparison charts
                    st.markdown("### Per-Strike Change in Call vs Put OI")
                    chgoi_strike_cols = st.columns(min(len(chgoi_strikes), 5))
                    for i, strike in enumerate(chgoi_strikes):
                        if i >= len(chgoi_strike_cols):
                            break
                        ce_col = f'{strike}_CE'
                        pe_col = f'{strike}_PE'
                        with chgoi_strike_cols[i]:
                            label = chgoi_position_labels[i] if i < len(chgoi_position_labels) else f'Strike {i}'
                            fig_chg_strike = go.Figure()
                            if ce_col in chgoi_history_df.columns:
                                fig_chg_strike.add_trace(go.Scatter(
                                    x=chgoi_history_df['time'],
                                    y=chgoi_history_df[ce_col] / 1000,
                                    mode='lines+markers',
                                    name='Call ChgOI',
                                    line=dict(color='#ff4444', width=2),
                                    marker=dict(size=3),
                                ))
                            if pe_col in chgoi_history_df.columns:
                                fig_chg_strike.add_trace(go.Scatter(
                                    x=chgoi_history_df['time'],
                                    y=chgoi_history_df[pe_col] / 1000,
                                    mode='lines+markers',
                                    name='Put ChgOI',
                                    line=dict(color='#00cc66', width=2),
                                    marker=dict(size=3),
                                ))
                            fig_chg_strike.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.5)
                            current_chg_ce = chgoi_history_df[ce_col].iloc[-1] / 1000 if ce_col in chgoi_history_df.columns and len(chgoi_history_df) > 0 else 0
                            current_chg_pe = chgoi_history_df[pe_col].iloc[-1] / 1000 if pe_col in chgoi_history_df.columns and len(chgoi_history_df) > 0 else 0
                            fig_chg_strike.update_layout(
                                title=f'{label}<br>₹{strike}<br>CE: {current_chg_ce:+.1f}K | PE: {current_chg_pe:+.1f}K',
                                template='plotly_dark',
                                height=280,
                                showlegend=True,
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=9)),
                                margin=dict(l=10, r=10, t=80, b=30),
                                xaxis=dict(tickformat='%H:%M', title=''),
                                yaxis=dict(title='Chg OI (K)'),
                                plot_bgcolor='#1e1e1e',
                                paper_bgcolor='#1e1e1e'
                            )
                            st.plotly_chart(fig_chg_strike, use_container_width=True)
                            if current_chg_ce > 0 and current_chg_pe <= 0:
                                st.error("CE Buildup (Bearish)")
                            elif current_chg_pe > 0 and current_chg_ce <= 0:
                                st.success("PE Buildup (Bullish)")
                            elif current_chg_ce > 0 and current_chg_pe > 0:
                                st.warning("Both Building")
                            elif current_chg_ce < 0 and current_chg_pe < 0:
                                st.warning("Both Unwinding")
                            else:
                                st.info("Neutral")
                else:
                    st.info("Waiting for ATM ± 2 strike data...")
                chgoi_info1, chgoi_info2 = st.columns([3, 1])
                with chgoi_info1:
                    st.caption(f"📈 {len(st.session_state.chgoi_history)} data points | Change in OI values in Thousands")
                with chgoi_info2:
                    if st.button("🗑️ Clear ChgOI History"):
                        st.session_state.chgoi_history = []
                        st.session_state.chgoi_last_valid_data = None
                        st.rerun()
            except Exception as e:
                st.warning(f"Error displaying Change OI charts: {str(e)}")
        else:
            st.info("📊 Change in OI history will build up as the app refreshes. Please wait for data collection...")
        st.markdown("---")
        st.markdown("## 📊 Gamma Exposure (GEX) Analysis - Dealer Hedging Flow")
        try:
            df_summary = option_data.get('df_summary')
            underlying_price = option_data.get('underlying')
            if df_summary is not None and underlying_price:
                gex_data = calculate_dealer_gex(df_summary, underlying_price)
                if gex_data:
                    st.session_state._gex_data = gex_data
                    gex_df = gex_data['gex_df']
                    st.session_state.gex_last_valid_data = gex_data
                    check_gex_alert(gex_data, df_summary, underlying_price)
                    def _gex_card(c, title, value, sub, color=None):
                        if value is not None and color:
                            c.markdown(f'<div style="background-color: {color}20; padding: 15px; border-radius: 10px; border: 2px solid {color};"><h4 style="color: {color}; margin: 0;">{title}</h4><h2 style="color: {color}; margin: 5px 0;">{value}</h2><p style="color: white; margin: 0; font-size: 12px;">{sub}</p></div>', unsafe_allow_html=True)
                        else:
                            c.markdown(f'<div style="background-color: #33333380; padding: 15px; border-radius: 10px; border: 2px solid #666;"><h4 style="color: #999; margin: 0;">{title}</h4><h2 style="color: #999; margin: 5px 0;">N/A</h2><p style="color: #666; margin: 0; font-size: 12px;">{sub}</p></div>', unsafe_allow_html=True)
                    gc1, gc2, gc3, gc4 = st.columns(4)
                    _gex_card(gc1, "Net GEX", f"{gex_data['total_gex']:+.2f}L", gex_data['gex_signal'], gex_data['gex_color'])
                    if gex_data['gamma_flip_level']:
                        fc = "#00ff88" if underlying_price > gex_data['gamma_flip_level'] else "#ff4444"
                        _gex_card(gc2, "Gamma Flip", f"₹{gex_data['gamma_flip_level']:.0f}", gex_data['spot_vs_flip'], fc)
                    else:
                        _gex_card(gc2, "Gamma Flip", None, "No flip detected")
                    _gex_card(gc3, "GEX Magnet", f"₹{gex_data['gex_magnet']:.0f}" if gex_data['gex_magnet'] else None,
                              "Price attracted here" if gex_data['gex_magnet'] else "No magnet", "#00ff88" if gex_data['gex_magnet'] else None)
                    _gex_card(gc4, "GEX Repeller", f"₹{gex_data['gex_repeller']:.0f}" if gex_data['gex_repeller'] else None,
                              "Price accelerates here" if gex_data['gex_repeller'] else "No repeller", "#ff4444" if gex_data['gex_repeller'] else None)
                    st.markdown(f"""
                    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 4px solid {gex_data['gex_color']}; margin: 10px 0;">
                        <b style="color: {gex_data['gex_color']};">Market Regime:</b> {gex_data['gex_interpretation']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("### 🎯 PCR × GEX Confluence")
                    atm_data = df_summary[df_summary['Zone'] == 'ATM']
                    if not atm_data.empty:
                        atm_pcr = atm_data.iloc[0].get('PCR', 1.0)
                        confluence_badge, confluence_signal, confluence_strength = calculate_pcr_gex_confluence(atm_pcr, gex_data)
                        st.session_state._confluence = (confluence_badge, confluence_signal, confluence_strength)
                        conf_col1, conf_col2 = st.columns([1, 3])
                        with conf_col1:
                            if "BULL" in confluence_badge:
                                badge_color = "#00ff88"
                            elif "BEAR" in confluence_badge:
                                badge_color = "#ff4444"
                            else:
                                badge_color = "#FFD700"
                            st.markdown(f"""
                            <div style="background-color: {badge_color}30; padding: 20px; border-radius: 15px; border: 3px solid {badge_color}; text-align: center;">
                                <h2 style="color: {badge_color}; margin: 0; font-size: 24px;">{confluence_badge}</h2>
                                <p style="color: white; margin: 5px 0; font-size: 14px;">{confluence_signal}</p>
                                <p style="color: #888; margin: 0; font-size: 12px;">Strength: {'★' * confluence_strength}{'☆' * (3 - confluence_strength)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with conf_col2:
                            st.markdown("""
                            **Confluence Matrix:**
                            - 🟢🔥 **STRONG BULL**: Bullish PCR + Negative GEX = Violent upside potential
                            - 🔴🔥 **STRONG BEAR**: Bearish PCR + Positive GEX = Strong rejection/pin down
                            - 🟢📍 **BULL RANGE**: Bullish PCR + Positive GEX = Support with chop
                            - 🔴⚡ **BEAR TREND**: Bearish PCR + Negative GEX = Downside acceleration
                            """)
                    st.markdown("### 📊 Net GEX by Strike (Dealer Hedging Pressure)")
                    fig_gex = go.Figure()
                    colors = ['#00ff88' if x >= 0 else '#ff4444' for x in gex_df['Net_GEX']]
                    fig_gex.add_trace(go.Bar(
                        x=gex_df['Strike'],
                        y=gex_df['Net_GEX'],
                        marker_color=colors,
                        name='Net GEX',
                        text=[f"{x:.1f}L" for x in gex_df['Net_GEX']],
                        textposition='outside',
                        textfont=dict(size=10)
                    ))
                    fig_gex.add_hline(y=0, line_dash="dash", line_color="white", line_width=2)
                    if gex_data['gamma_flip_level']:
                        fig_gex.add_vline(
                            x=gex_data['gamma_flip_level'],
                            line_dash="dot",
                            line_color="#FFD700",
                            line_width=2,
                            annotation_text=f"Gamma Flip: ₹{gex_data['gamma_flip_level']:.0f}",
                            annotation_position="top"
                        )
                    fig_gex.add_vline(
                        x=underlying_price,
                        line_dash="solid",
                        line_color="#00aaff",
                        line_width=3,
                        annotation_text=f"Spot: ₹{underlying_price:.0f}",
                        annotation_position="bottom"
                    )
                    fig_gex.update_layout(
                        title=f"Net GEX by Strike | Total: {gex_data['total_gex']:+.2f}L",
                        template='plotly_dark',
                        height=400,
                        showlegend=False,
                        xaxis_title="Strike Price",
                        yaxis_title="Net GEX (Lakhs)",
                        plot_bgcolor='#1e1e1e',
                        paper_bgcolor='#1e1e1e',
                        margin=dict(l=50, r=50, t=60, b=50)
                    )
                    st.plotly_chart(fig_gex, use_container_width=True)
                    with st.expander("📋 GEX Breakdown by Strike"):
                        gex_display = gex_df.copy()
                        gex_display['Strike'] = gex_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                        def color_gex(val):
                            try:
                                v = float(val)
                                if v > 10:
                                    return 'background-color: #00ff8840; color: white'
                                elif v > 0:
                                    return 'background-color: #00ff8820; color: white'
                                elif v < -10:
                                    return 'background-color: #ff444440; color: white'
                                elif v < 0:
                                    return 'background-color: #ff444420; color: white'
                                else:
                                    return ''
                            except:
                                return ''
                        styled_gex = gex_display.style.map(color_gex, subset=['Call_GEX', 'Put_GEX', 'Net_GEX'])
                        st.dataframe(styled_gex, use_container_width=True, hide_index=True)
                        st.markdown("""
                        **GEX Interpretation:**
                        - **Positive Net GEX (Green)**: Dealers LONG gamma → Price tends to PIN/REVERT
                        - **Negative Net GEX (Red)**: Dealers SHORT gamma → Price tends to ACCELERATE
                        - **GEX Magnet**: Strike with highest positive GEX (price attracted)
                        - **GEX Repeller**: Strike with most negative GEX (price accelerates away)
                        - **Gamma Flip**: Level where dealers switch from long to short gamma
                        """)
                    st.markdown("### 📈 GEX Time Series (ATM ± 2 Strikes)")
                    ist = pytz.timezone('Asia/Kolkata')
                    current_time = datetime.now(ist)
                    gex_entry = {'time': current_time, 'total_gex': gex_data['total_gex']}
                    for _, row in gex_df.iterrows():
                        strike_label = str(int(row['Strike']))
                        gex_entry[strike_label] = row['Net_GEX']
                        gex_entry[f'{strike_label}_Call'] = row.get('Call_GEX', 0)
                        gex_entry[f'{strike_label}_Put'] = row.get('Put_GEX', 0)
                    current_gex_strikes = [int(row['Strike']) for _, row in gex_df.iterrows()]
                    st.session_state.gex_current_strikes = sorted(current_gex_strikes)
                    should_add_gex = True
                    if st.session_state.gex_history:
                        last_gex_entry = st.session_state.gex_history[-1]
                        time_diff = (current_time - last_gex_entry['time']).total_seconds()
                        if time_diff < 30:
                            should_add_gex = False
                    if should_add_gex:
                        st.session_state.gex_history.append(gex_entry)
                        if len(st.session_state.gex_history) > 200:
                            st.session_state.gex_history = st.session_state.gex_history[-200:]
                        # Persist GEX to Supabase
                        try:
                            expiry_for_gex = option_data.get('expiry', selected_expiry) if option_data else selected_expiry
                            gex_records = []
                            for _, row in gex_df.iterrows():
                                gex_records.append({
                                    'timestamp': current_time,
                                    'expiry': expiry_for_gex,
                                    'strike': float(row['Strike']),
                                    'atm_strike': float(atm_strike) if 'atm_strike' in dir() else 0,
                                    'total_gex': float(gex_data['total_gex']),
                                    'call_gex': float(row.get('Call_GEX', 0)),
                                    'put_gex': float(row.get('Put_GEX', 0)),
                                    'net_gex': float(row.get('Net_GEX', 0)),
                                    'gamma_flip': gex_data.get('gamma_flip_level'),
                                    'signal': gex_data.get('gex_signal', ''),
                                    'spot': float(underlying_price),
                                })
                            if gex_records:
                                db.upsert_gex_history(gex_records)
                        except Exception:
                            pass
                    def create_gex_chart(history_df, col_name, color, title_prefix):
                        """Helper to create individual GEX chart per strike"""
                        if col_name and col_name in history_df.columns:
                            strike_val = col_name
                            gex_values = history_df[col_name].dropna()
                            max_abs = 20
                            if len(gex_values) > 0:
                                max_abs = max(abs(gex_values.max()), abs(gex_values.min()), 15)
                            y_range = [-max_abs * 1.1, max_abs * 1.1]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=history_df['time'],
                                y=history_df[col_name],
                                mode='lines+markers',
                                name=f'₹{strike_val}',
                                line=dict(color=color, width=2),
                                marker=dict(size=4),
                                fill='tozeroy',
                                fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}'
                            ))
                            fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=2)
                            fig.add_hline(y=10, line_dash="dot", line_color="#00ff88", line_width=1,
                                         annotation_text="+10", annotation_position="right")
                            fig.add_hline(y=-10, line_dash="dot", line_color="#ff4444", line_width=1,
                                         annotation_text="-10", annotation_position="right")
                            current_gex = history_df[col_name].iloc[-1] if len(history_df) > 0 else 0
                            fig.update_layout(
                                title=f"{title_prefix}<br>₹{strike_val}<br>GEX: {current_gex:+.1f}L",
                                template='plotly_dark',
                                height=300,
                                showlegend=False,
                                margin=dict(l=10, r=10, t=70, b=30),
                                xaxis=dict(tickformat='%H:%M', title=''),
                                yaxis=dict(
                                    title='GEX (L)',
                                    range=y_range,
                                    zeroline=True,
                                    zerolinecolor='white',
                                    zerolinewidth=2,
                                    tickmode='array',
                                    tickvals=[-20, -10, 0, 10, 20] if max_abs <= 25 else None
                                ),
                                plot_bgcolor='#1e1e1e',
                                paper_bgcolor='#1e1e1e'
                            )
                            return fig, current_gex
                        return None, 0
                    try:
                        current_strikes = sorted([int(row['Strike']) for _, row in gex_df.iterrows()])
                        st.session_state.gex_current_strikes = current_strikes
                        current_gex_values = {}
                        for _, row in gex_df.iterrows():
                            strike = int(row['Strike'])
                            current_gex_values[strike] = row['Net_GEX']
                        gex_col1, gex_col2, gex_col3, gex_col4, gex_col5 = st.columns(5)
                        def display_gex_with_signal(container, fig, gex_val):
                            if fig:
                                container.plotly_chart(fig, use_container_width=True)
                                if gex_val > 10:
                                    container.success("📍 Pin Zone")
                                elif gex_val < -10:
                                    container.error("⚡ Accel Zone")
                                else:
                                    container.warning("➡️ Neutral")
                        position_labels = ['🟣 ITM-2', '🟣 ITM-1', '🟡 ATM', '🔵 OTM+1', '🔵 OTM+2']
                        position_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']
                        columns = [gex_col1, gex_col2, gex_col3, gex_col4, gex_col5]
                        has_history = len(st.session_state.gex_history) > 0
                        gex_history_df = pd.DataFrame(st.session_state.gex_history) if has_history else None
                        for i, col in enumerate(columns):
                            with col:
                                if i < len(current_strikes):
                                    strike = current_strikes[i]
                                    strike_col = str(strike)
                                    current_gex = current_gex_values.get(strike, 0)
                                    if has_history and gex_history_df is not None and strike_col in gex_history_df.columns:
                                        fig, gex_val = create_gex_chart(gex_history_df, strike_col, position_colors[i], f'{position_labels[i]}')
                                        display_gex_with_signal(st, fig, gex_val)
                                    else:
                                        fig = go.Figure()
                                        now = datetime.now(pytz.timezone('Asia/Kolkata'))
                                        fig.add_trace(go.Scatter(
                                            x=[now],
                                            y=[current_gex],
                                            mode='markers+text',
                                            marker=dict(size=20, color=position_colors[i], symbol='diamond'),
                                            text=[f'{current_gex:+.1f}L'],
                                            textposition='top center',
                                            textfont=dict(size=12, color='white'),
                                            name=f'₹{strike}'
                                        ))
                                        max_abs = max(abs(current_gex), 15)
                                        y_range = [-max_abs * 1.3, max_abs * 1.3]
                                        fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=2)
                                        fig.add_hline(y=10, line_dash="dot", line_color="#00ff88", line_width=1,
                                                     annotation_text="+10", annotation_position="right")
                                        fig.add_hline(y=-10, line_dash="dot", line_color="#ff4444", line_width=1,
                                                     annotation_text="-10", annotation_position="right")
                                        fig.update_layout(
                                            title=f"{position_labels[i]}<br>₹{strike}<br>GEX: {current_gex:+.1f}L",
                                            template='plotly_dark',
                                            height=300,
                                            showlegend=False,
                                            margin=dict(l=10, r=10, t=70, b=30),
                                            xaxis=dict(tickformat='%H:%M', title='', showticklabels=True),
                                            yaxis=dict(
                                                title='GEX (L)',
                                                range=y_range,
                                                zeroline=True,
                                                zerolinecolor='white',
                                                zerolinewidth=2,
                                                tickmode='array',
                                                tickvals=[-20, -10, 0, 10, 20] if max_abs <= 25 else None
                                            ),
                                            plot_bgcolor='#1e1e1e',
                                            paper_bgcolor='#1e1e1e'
                                        )
                                        display_gex_with_signal(st, fig, current_gex)
                                else:
                                    st.info(f"{position_labels[i]} N/A")
                        # Combined Call GEX vs Put GEX time series
                        st.markdown("### 📊 Call GEX vs Put GEX - Time Series (ATM ± 2)")
                        if has_history and gex_history_df is not None:
                            gex_pos_labels = ['ITM-2', 'ITM-1', 'ATM', 'OTM+1', 'OTM+2']
                            gex_ts_col1, gex_ts_col2 = st.columns(2)
                            with gex_ts_col1:
                                fig_call_gex = go.Figure()
                                for idx, strike in enumerate(current_strikes):
                                    call_col = f'{strike}_Call'
                                    if call_col in gex_history_df.columns:
                                        lbl = gex_pos_labels[idx] if idx < len(gex_pos_labels) else f'Strike {idx}'
                                        fig_call_gex.add_trace(go.Scatter(
                                            x=gex_history_df['time'],
                                            y=gex_history_df[call_col],
                                            mode='lines+markers',
                                            name=f'₹{strike} ({lbl})',
                                            line=dict(width=2),
                                            marker=dict(size=3),
                                        ))
                                fig_call_gex.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
                                fig_call_gex.update_layout(
                                    title='Call GEX by Strike (ATM ± 2)',
                                    template='plotly_dark',
                                    height=350,
                                    margin=dict(l=10, r=10, t=50, b=30),
                                    xaxis=dict(tickformat='%H:%M', title='Time'),
                                    yaxis=dict(title='Call GEX (Lakhs)'),
                                    plot_bgcolor='#1e1e1e',
                                    paper_bgcolor='#1e1e1e',
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                                )
                                st.plotly_chart(fig_call_gex, use_container_width=True)
                            with gex_ts_col2:
                                fig_put_gex = go.Figure()
                                for idx, strike in enumerate(current_strikes):
                                    put_col = f'{strike}_Put'
                                    if put_col in gex_history_df.columns:
                                        lbl = gex_pos_labels[idx] if idx < len(gex_pos_labels) else f'Strike {idx}'
                                        fig_put_gex.add_trace(go.Scatter(
                                            x=gex_history_df['time'],
                                            y=gex_history_df[put_col],
                                            mode='lines+markers',
                                            name=f'₹{strike} ({lbl})',
                                            line=dict(width=2),
                                            marker=dict(size=3),
                                        ))
                                fig_put_gex.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
                                fig_put_gex.update_layout(
                                    title='Put GEX by Strike (ATM ± 2)',
                                    template='plotly_dark',
                                    height=350,
                                    margin=dict(l=10, r=10, t=50, b=30),
                                    xaxis=dict(tickformat='%H:%M', title='Time'),
                                    yaxis=dict(title='Put GEX (Lakhs)'),
                                    plot_bgcolor='#1e1e1e',
                                    paper_bgcolor='#1e1e1e',
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                                )
                                st.plotly_chart(fig_put_gex, use_container_width=True)
                            # Per-strike Call GEX vs Put GEX comparison
                            st.markdown("### Per-Strike Call GEX vs Put GEX")
                            gex_cmp_cols = st.columns(min(len(current_strikes), 5))
                            for idx, strike in enumerate(current_strikes):
                                if idx >= len(gex_cmp_cols):
                                    break
                                call_col = f'{strike}_Call'
                                put_col = f'{strike}_Put'
                                with gex_cmp_cols[idx]:
                                    lbl = gex_pos_labels[idx] if idx < len(gex_pos_labels) else f'Strike {idx}'
                                    fig_gex_cmp = go.Figure()
                                    if call_col in gex_history_df.columns:
                                        fig_gex_cmp.add_trace(go.Scatter(
                                            x=gex_history_df['time'],
                                            y=gex_history_df[call_col],
                                            mode='lines+markers',
                                            name='Call GEX',
                                            line=dict(color='#ff4444', width=2),
                                            marker=dict(size=3),
                                        ))
                                    if put_col in gex_history_df.columns:
                                        fig_gex_cmp.add_trace(go.Scatter(
                                            x=gex_history_df['time'],
                                            y=gex_history_df[put_col],
                                            mode='lines+markers',
                                            name='Put GEX',
                                            line=dict(color='#00cc66', width=2),
                                            marker=dict(size=3),
                                        ))
                                    fig_gex_cmp.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.5)
                                    cur_call_gex = gex_history_df[call_col].iloc[-1] if call_col in gex_history_df.columns and len(gex_history_df) > 0 else 0
                                    cur_put_gex = gex_history_df[put_col].iloc[-1] if put_col in gex_history_df.columns and len(gex_history_df) > 0 else 0
                                    fig_gex_cmp.update_layout(
                                        title=f'{lbl}<br>₹{strike}<br>CE: {cur_call_gex:+.1f}L | PE: {cur_put_gex:+.1f}L',
                                        template='plotly_dark',
                                        height=280,
                                        showlegend=True,
                                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=9)),
                                        margin=dict(l=10, r=10, t=80, b=30),
                                        xaxis=dict(tickformat='%H:%M', title=''),
                                        yaxis=dict(title='GEX (L)'),
                                        plot_bgcolor='#1e1e1e',
                                        paper_bgcolor='#1e1e1e'
                                    )
                                    st.plotly_chart(fig_gex_cmp, use_container_width=True)
                                    net = cur_call_gex + cur_put_gex
                                    if net > 5:
                                        st.success("Pin Zone")
                                    elif net < -5:
                                        st.error("Accel Zone")
                                    else:
                                        st.warning("Neutral")
                        st.markdown("### Current GEX Values")
                        gex_display = gex_df[['Strike', 'Zone', 'Call_GEX', 'Put_GEX', 'Net_GEX']].copy()
                        gex_display['Strike'] = gex_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                        styled_gex_table = gex_display.style.map(color_gex, subset=['Call_GEX', 'Put_GEX', 'Net_GEX'])
                        st.dataframe(styled_gex_table, use_container_width=True, hide_index=True)
                        gex_info1, gex_info2 = st.columns([3, 1])
                        with gex_info1:
                            history_status = f"📈 {len(st.session_state.gex_history)} data points" if has_history else "⏳ Building history..."
                            st.caption(f"🟢 Live | {history_status} | GEX > 10 = Pin Zone | GEX < -10 = Acceleration Zone")
                        with gex_info2:
                            if st.button("🗑️ Clear GEX History"):
                                st.session_state.gex_history = []
                                st.session_state.gex_last_valid_data = None
                                st.rerun()
                    except Exception as e:
                        st.warning(f"Error displaying GEX charts: {str(e)}")
                else:
                    st.warning("Unable to calculate GEX. Check option chain data.")
        except Exception as e:
            st.warning(f"GEX analysis unavailable: {str(e)}")
        with st.expander("📊 Detailed Greeks & Raw Values"):
            df_summary = option_data['df_summary']
            detail_cols = ['Strike', 'Zone', 'lastPrice_CE', 'lastPrice_PE',
                           'openInterest_CE', 'openInterest_PE', 'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                           'totalTradedVolume_CE', 'totalTradedVolume_PE',
                           'Delta_CE', 'Delta_PE', 'Gamma_CE', 'Gamma_PE', 'Vega_CE', 'Vega_PE', 'Theta_CE', 'Theta_PE',
                           'impliedVolatility_CE', 'impliedVolatility_PE',
                           'bidQty_CE', 'bidQty_PE', 'askQty_CE', 'askQty_PE']
            detail_cols = [col for col in detail_cols if col in df_summary.columns]
            if detail_cols:
                st.dataframe(df_summary[detail_cols].style.apply(highlight_atm_row, axis=1), use_container_width=True)
        csv_data = create_csv_download(option_data['df_summary'])
        st.download_button(
            label="📥 Download Summary as CSV",
            data=csv_data,
            file_name=f"nifty_options_summary_{option_data['expiry']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    # === MASTER TRADING SIGNAL ===
    if option_data and option_data.get('underlying') and not df.empty:
        st.markdown("---")
        st.markdown("## 🎯 Master Trading Signal")
        try:
            _sa = getattr(st.session_state, '_sa_result', None)
            _gex = getattr(st.session_state, '_gex_data', None)
            _conf = getattr(st.session_state, '_confluence', None)
            if _sa:
                master = generate_master_signal(df, _sa, _gex, _conf, option_data['underlying'], api)
                if master:
                    # Send Telegram (with cooldown)
                    if 'last_master_signal_time' not in st.session_state:
                        st.session_state.last_master_signal_time = None
                    _ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                    _should_send = st.session_state.last_master_signal_time is None or \
                        (_ist_now - st.session_state.last_master_signal_time).total_seconds() > 300
                    if _should_send:
                        try:
                            send_master_signal_telegram(master, option_data['underlying'])
                            st.session_state.last_master_signal_time = _ist_now
                        except Exception:
                            pass

                    # Signal Banner
                    if 'BUY' in master['trade_type'] or 'BREAKOUT' in master['signal']:
                        sig_color = '#00ff88'
                    elif 'SELL' in master['trade_type'] or 'BREAKDOWN' in master['signal']:
                        sig_color = '#ff4444'
                    else:
                        sig_color = '#FFD700'
                    st.markdown(f"""
                    <div style="background:{sig_color}20;padding:25px;border-radius:15px;border:3px solid {sig_color};text-align:center;margin-bottom:15px;">
                        <h1 style="color:{sig_color};margin:0;font-size:2.5em;">{master['signal']}</h1>
                        <h2 style="color:white;margin:5px 0;">{master['trade_type']}</h2>
                        <h3 style="color:{sig_color};margin:5px 0;">Confluence: {master['abs_score']}/7 ({master['strength']}) | Confidence: {master['confidence']}%</h3>
                        <div style="background:#33333380;border-radius:10px;height:14px;margin:10px auto;max-width:500px;">
                            <div style="background:{sig_color};border-radius:10px;height:14px;width:{master['confidence']}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Details in columns
                    ms_col1, ms_col2, ms_col3 = st.columns(3)
                    with ms_col1:
                        st.markdown("#### 🕯 Candle Pattern (Nifty Price Action)")
                        cd = master['candle']
                        d = cd['details']
                        pat_color = '#00ff88' if cd['direction'] == 'Bullish' else '#ff4444' if cd['direction'] == 'Bearish' else '#FFD700'
                        st.markdown(f"""
                        <div style="background:{pat_color}20;padding:10px;border-radius:8px;border-left:4px solid {pat_color};margin-bottom:8px;">
                            <b style="color:{pat_color};font-size:16px;">{cd['pattern']}</b> <span style="color:white;">({cd['direction']})</span><br>
                            <span style="color:#aaa;font-size:13px;">O: ₹{d['open']:.2f} | H: ₹{d['high']:.2f} | L: ₹{d['low']:.2f} | C: ₹{d['close']:.2f}</span><br>
                            <span style="color:#aaa;font-size:12px;">Body: {d['body_ratio']*100:.0f}% | {'🟢 Bull' if d['is_green'] else '🔴 Bear'}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        # Last 5 candles breakdown
                        st.markdown("**Last 5 Candles:**")
                        candle_rows = []
                        for ci, cn in enumerate(cd.get('candles', [])):
                            candle_rows.append({
                                '#': ci + 1,
                                'Time': cn['time'],
                                'Type': f"{'🟢' if cn['type'] == 'Bull' else '🔴'} {cn['type']}",
                                'Pattern': cn['pattern'],
                                'O': f"₹{cn['open']:.0f}",
                                'H': f"₹{cn['high']:.0f}",
                                'L': f"₹{cn['low']:.0f}",
                                'C': f"₹{cn['close']:.0f}",
                                'Vol': f"{cn['volume']:,}" if cn['volume'] else '-',
                            })
                        if candle_rows:
                            st.dataframe(pd.DataFrame(candle_rows), use_container_width=True, hide_index=True, height=210)
                        st.caption(f"Bull: {cd.get('bull_count', 0)} | Bear: {cd.get('bear_count', 0)} out of last 5")

                        st.markdown(f"**📍 Location:** {', '.join(master['location'])}")
                        vol_color = '#00ff88' if master['volume']['spike'] else '#888888'
                        st.markdown(f"**📊 Volume:** <span style='color:{vol_color}'>{master['volume']['label']} ({master['volume']['ratio']}x)</span>", unsafe_allow_html=True)
                        st.markdown("#### 🟢 Order Blocks")
                        ob = master['order_blocks']
                        if ob.get('bullish_ob'):
                            st.success(f"Bullish OB: ₹{ob['bullish_ob']['low']:.0f} - ₹{ob['bullish_ob']['high']:.0f}")
                        else:
                            st.info("No Bullish OB detected")
                        if ob.get('bearish_ob'):
                            st.error(f"Bearish OB: ₹{ob['bearish_ob']['low']:.0f} - ₹{ob['bearish_ob']['high']:.0f}")
                        else:
                            st.info("No Bearish OB detected")

                    with ms_col2:
                        st.markdown("#### 🔮 GEX (from app)")
                        gex = master['gex']
                        gex_items = [
                            ("Net GEX", f"{gex['net_gex']:+.1f}L"),
                            ("ATM GEX", f"{gex['atm_gex']:+.1f}L"),
                            ("Gamma Flip", f"₹{gex['gamma_flip']:.0f}" if gex['gamma_flip'] else "N/A"),
                            ("Magnet", f"₹{gex['magnet']:.0f}" if gex['magnet'] else "N/A"),
                            ("Repeller", f"₹{gex['repeller']:.0f}" if gex['repeller'] else "N/A"),
                            ("Market Mode", gex['market_mode']),
                        ]
                        for label, val in gex_items:
                            st.markdown(f"**{label}:** {val}")
                        if gex['above_flip'] is not None:
                            if gex['above_flip']:
                                st.success("Above Gamma Flip (Bullish)")
                            else:
                                st.error("Below Gamma Flip (Bearish)")
                        st.markdown(f"#### 📊 PCR × GEX")
                        st.markdown(f"**{master['pcr_gex']['badge']}**")

                        st.markdown("#### 🟥 Resistance / 🟩 Support")
                        for r in master['resistance_levels'][:3]:
                            st.markdown(f"🟥 ₹{r:.0f}")
                        for s in master['support_levels'][:3]:
                            st.markdown(f"🟩 ₹{s:.0f}")

                    with ms_col3:
                        st.markdown("#### 📉 VIX")
                        vix = master['vix']
                        vix_val = vix.get('vix', 0)
                        vix_dir = vix.get('direction', 'Unknown')
                        if vix_val > 0:
                            vix_aligned = ('Aligned' in str(master['reasons']))
                            if vix_dir == 'Falling':
                                st.success(f"VIX: {vix_val:.2f} ({vix_dir}) {'- Aligned' if vix_aligned else '- Opposite'}")
                            elif vix_dir == 'Rising':
                                st.error(f"VIX: {vix_val:.2f} ({vix_dir}) {'- Aligned' if vix_aligned else '- Opposite'}")
                            else:
                                st.info(f"VIX: {vix_val:.2f} ({vix_dir})")
                        else:
                            st.info("VIX data loading...")

                        st.markdown("#### 📋 Confluence Factors")
                        for reason in master['reasons']:
                            st.markdown(f"✔ {reason}")

                    # === FULL ALIGNMENT TABLE (below the 3-col section) ===
                    st.markdown("---")
                    st.markdown("## 🌍 Index & Stock Alignment - Candle Patterns & Sentiment")
                    align_data = master.get('alignment', {})
                    if align_data:
                        # Candle Pattern Table
                        st.markdown("### 🕯 Candle Patterns Across Indices")
                        pattern_rows = []
                        # Define display order
                        display_order = ['NIFTY 50', 'SENSEX', 'BANKNIFTY', 'NIFTY IT', 'RELIANCE', 'ICICIBANK', 'INDIA VIX']
                        for name in display_order:
                            ad = align_data.get(name)
                            if ad is None:
                                continue
                            pat_emoji = '🟢' if ad.get('candle_dir') == 'Bullish' else '🔴' if ad.get('candle_dir') == 'Bearish' else '🟡'
                            pattern_rows.append({
                                'Index': name,
                                'LTP': f"₹{ad['ltp']:.2f}" if ad.get('ltp', 0) > 0 else 'N/A',
                                'Candle': f"{pat_emoji} {ad.get('candle_pattern', 'N/A')}",
                                'Direction': ad.get('candle_dir', 'N/A'),
                                'Bull/Bear (5)': f"{ad.get('bull_count', '-')}/{ad.get('bear_count', '-')}",
                                'Day H': f"₹{ad['day_high']:.0f}" if ad.get('day_high') else '-',
                                'Day L': f"₹{ad['day_low']:.0f}" if ad.get('day_low') else '-',
                            })
                        if pattern_rows:
                            pat_df = pd.DataFrame(pattern_rows)
                            def _style_pat(row):
                                d = row['Direction']
                                if d == 'Bullish':
                                    return ['background-color:#00ff8815;color:white'] * len(row)
                                elif d == 'Bearish':
                                    return ['background-color:#ff444415;color:white'] * len(row)
                                return [''] * len(row)
                            st.dataframe(pat_df.style.apply(_style_pat, axis=1), use_container_width=True, hide_index=True)

                        # Multi-timeframe Sentiment Table
                        st.markdown("### 📊 Multi-Timeframe Sentiment (Price Action)")
                        sent_rows = []
                        for name in display_order:
                            ad = align_data.get(name)
                            if ad is None:
                                continue
                            def _sent_emoji(s):
                                return '🟢' if s == 'Bullish' else '🔴' if s == 'Bearish' else '🟡'
                            s10 = ad.get('sentiment_10m', 'N/A')
                            s1h = ad.get('sentiment_1h', 'N/A')
                            s4h = ad.get('sentiment_4h', 'N/A')
                            sent_rows.append({
                                'Index': name,
                                'LTP': f"₹{ad['ltp']:.2f}" if ad.get('ltp', 0) > 0 else 'N/A',
                                '10 Min': f"{_sent_emoji(s10)} {s10} ({ad.get('pct_10m', 0):+.2f}%)" if s10 != 'N/A' else 'N/A',
                                '1 Hour': f"{_sent_emoji(s1h)} {s1h} ({ad.get('pct_1h', 0):+.2f}%)" if s1h != 'N/A' else 'N/A',
                                '4 Hours': f"{_sent_emoji(s4h)} {s4h} ({ad.get('pct_4h', 0):+.2f}%)" if s4h != 'N/A' else 'N/A',
                                'Trend': ad.get('trend', 'N/A'),
                            })
                        if sent_rows:
                            sent_df = pd.DataFrame(sent_rows)
                            def _style_sent(row):
                                t = row['Trend']
                                if t == 'Bullish':
                                    return ['background-color:#00ff8815;color:white'] * len(row)
                                elif t == 'Bearish':
                                    return ['background-color:#ff444415;color:white'] * len(row)
                                return [''] * len(row)
                            st.dataframe(sent_df.style.apply(_style_sent, axis=1), use_container_width=True, hide_index=True)

                        # Summary
                        non_vix_align = {k: v for k, v in align_data.items() if 'VIX' not in k}
                        bull_10m = sum(1 for v in non_vix_align.values() if v.get('sentiment_10m') == 'Bullish')
                        bear_10m = sum(1 for v in non_vix_align.values() if v.get('sentiment_10m') == 'Bearish')
                        bull_1h = sum(1 for v in non_vix_align.values() if v.get('sentiment_1h') == 'Bullish')
                        bear_1h = sum(1 for v in non_vix_align.values() if v.get('sentiment_1h') == 'Bearish')
                        total_idx = len(non_vix_align)
                        sum_col1, sum_col2, sum_col3 = st.columns(3)
                        with sum_col1:
                            if bull_10m > bear_10m:
                                st.success(f"10m: Bullish ({bull_10m}/{total_idx})")
                            elif bear_10m > bull_10m:
                                st.error(f"10m: Bearish ({bear_10m}/{total_idx})")
                            else:
                                st.warning(f"10m: Mixed ({bull_10m}B/{bear_10m}R)")
                        with sum_col2:
                            if bull_1h > bear_1h:
                                st.success(f"1h: Bullish ({bull_1h}/{total_idx})")
                            elif bear_1h > bull_1h:
                                st.error(f"1h: Bearish ({bear_1h}/{total_idx})")
                            else:
                                st.warning(f"1h: Mixed ({bull_1h}B/{bear_1h}R)")
                        with sum_col3:
                            bull_4h = sum(1 for v in non_vix_align.values() if v.get('sentiment_4h') == 'Bullish')
                            bear_4h = sum(1 for v in non_vix_align.values() if v.get('sentiment_4h') == 'Bearish')
                            if bull_4h > bear_4h:
                                st.success(f"4h: Bullish ({bull_4h}/{total_idx})")
                            elif bear_4h > bull_4h:
                                st.error(f"4h: Bearish ({bear_4h}/{total_idx})")
                            else:
                                st.warning(f"4h: Mixed ({bull_4h}B/{bear_4h}R)")
                        # === % Change from Open - Line Chart ===
                        st.markdown("### 📈 Price Action - % Change from Open (Today)")
                        fig_pct = go.Figure()
                        line_colors = {
                            'NIFTY 50': '#FFD700',
                            'SENSEX': '#FF6B6B',
                            'BANKNIFTY': '#00BFFF',
                            'NIFTY IT': '#FF69B4',
                            'RELIANCE': '#00FF7F',
                            'ICICIBANK': '#FFA500',
                            'INDIA VIX': '#FF4444',
                        }
                        for name in display_order:
                            ad = align_data.get(name)
                            if ad is None:
                                continue
                            pct_time = ad.get('pct_series_time', [])
                            pct_vals = ad.get('pct_series_vals', [])
                            if pct_time and pct_vals:
                                line_width = 3 if name == 'NIFTY 50' else 2
                                dash = 'dot' if name == 'INDIA VIX' else None
                                fig_pct.add_trace(go.Scatter(
                                    x=pct_time, y=pct_vals,
                                    mode='lines',
                                    name=name,
                                    line=dict(color=line_colors.get(name, '#888'), width=line_width, dash=dash),
                                ))
                        fig_pct.add_hline(y=0, line_dash="solid", line_color="white", line_width=1.5)
                        fig_pct.update_layout(
                            title='All Indices & Stocks - % Change from Day Open',
                            template='plotly_dark',
                            height=450,
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='% Change', zeroline=True, zerolinecolor='white', zerolinewidth=2,
                                       ticksuffix='%'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig_pct, use_container_width=True)
                    else:
                        st.info("Alignment data loading... will appear on next refresh.")
                else:
                    st.info("Generating master signal... waiting for data.")
            else:
                st.info("Master signal requires deep analysis data. Please wait...")
        except Exception as e:
            st.warning(f"Master signal unavailable: {str(e)}")

    if show_analytics:
        st.markdown("---")
        display_analytics_dashboard(db)

    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    st.sidebar.info(f"Last Updated: {current_time}")

if __name__ == "__main__":
    main()
