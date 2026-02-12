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
from supabase import create_client, Client
import hashlib
import numpy as np
import math
from scipy.stats import norm
from pytz import timezone
import io
import os

# Page configuration - ADD THIS AT THE VERY TOP
st.set_page_config(
    page_title="Nifty Trading & Options Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 80 seconds - MOVE THIS RIGHT AFTER PAGE CONFIG
st_autorefresh(interval=30000, key="datarefresh")

# Custom CSS for TradingView-like appearance + ATM highlighting
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stSelectbox > div > div > select {
        background-color: #1e1e1e;
        color: white;
    }
    .metric-container {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .price-up {
        color: #00ff88;
    }
    .price-down {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)

# === API Configuration ===
try:
    DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
    DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
    
    if not DHAN_CLIENT_ID:
        DHAN_CLIENT_ID = st.secrets.get("dhan", {}).get("client_id", "")
    if not DHAN_ACCESS_TOKEN:
        DHAN_ACCESS_TOKEN = st.secrets.get("dhan", {}).get("access_token", "")
        
    supabase_url = st.secrets.get("supabase", {}).get("url", "")
    supabase_key = st.secrets.get("supabase", {}).get("anon_key", "")
    
    # Enhanced Telegram Configuration with better error handling
    try:
        # Try multiple ways to get the credentials
        TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
        
        # If still empty, try alternative approaches
        if not TELEGRAM_BOT_TOKEN:
            try:
                TELEGRAM_BOT_TOKEN = st.secrets.TELEGRAM_BOT_TOKEN
            except:
                pass
        
        if not TELEGRAM_CHAT_ID:
            try:
                TELEGRAM_CHAT_ID = st.secrets.TELEGRAM_CHAT_ID
            except:
                pass
        
        # Convert chat ID to string if it's numeric
        if TELEGRAM_CHAT_ID and isinstance(TELEGRAM_CHAT_ID, (int, float)):
            TELEGRAM_CHAT_ID = str(int(TELEGRAM_CHAT_ID))
            
    except Exception as e:
        st.error(f"Telegram config error: {e}")
        TELEGRAM_BOT_TOKEN = ""
        TELEGRAM_CHAT_ID = ""
    
except Exception:
    DHAN_CLIENT_ID = ""
    DHAN_ACCESS_TOKEN = ""
    supabase_url = ""
    supabase_key = ""
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"

# Cached functions for performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_pivot_calculation(df_json, pivot_settings):
    """Cache pivot calculations to improve performance"""
    df = pd.read_json(df_json)
    return PivotIndicator.get_all_pivots(df, pivot_settings)

@st.cache_data(ttl=60)  # Cache for 1 minute
def cached_iv_average(option_data_json):
    """Cache IV average calculation"""
    df = pd.read_json(option_data_json)
    iv_ce_avg = df['impliedVolatility_CE'].mean()
    iv_pe_avg = df['impliedVolatility_PE'].mean()
    return iv_ce_avg, iv_pe_avg

# Telegram Functions
def send_telegram_message_sync(message):
    """Send message to Telegram synchronously"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
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
    """Test if Telegram is properly configured"""
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

class SupabaseDB:
    def __init__(self, url, key):
        self.client: Client = create_client(url, key)
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            self.client.table('candle_data').select('id').limit(1).execute()
        except:
            st.info("Database tables may need to be created. Please run the SQL setup first.")
    
    def save_candle_data(self, symbol, exchange, timeframe, df):
        """Save candle data to Supabase"""
        if df.empty:
            return
        
        try:
            records = []
            for _, row in df.iterrows():
                record = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'timestamp': int(row['timestamp']),
                    'datetime': row['datetime'].isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume'])
                }
                records.append(record)
            
            self.client.table('candle_data').upsert(
                records, 
                on_conflict="symbol,exchange,timeframe,timestamp"
            ).execute()
            
        except Exception as e:
            if "23505" not in str(e) and "duplicate key" not in str(e).lower():
                st.error(f"Error saving candle data: {str(e)}")
    
    def get_candle_data(self, symbol, exchange, timeframe, hours_back=24):
        """Retrieve candle data from Supabase"""
        try:
            cutoff_time = datetime.now(pytz.UTC) - timedelta(hours=hours_back)
            
            result = self.client.table('candle_data')\
                .select('*')\
                .eq('symbol', symbol)\
                .eq('exchange', exchange)\
                .eq('timeframe', timeframe)\
                .gte('datetime', cutoff_time.isoformat())\
                .order('timestamp', desc=False)\
                .execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error retrieving candle data: {str(e)}")
            return pd.DataFrame()
    
    def clear_old_candle_data(self, days_old=7):
        """Clear candle data older than specified days"""
        try:
            cutoff_date = datetime.now(pytz.UTC) - timedelta(days=days_old)
            
            result = self.client.table('candle_data')\
                .delete()\
                .lt('datetime', cutoff_date.isoformat())\
                .execute()
            
            return len(result.data) if result.data else 0
        except Exception as e:
            st.error(f"Error clearing old data: {str(e)}")
            return 0
    
    def save_user_preferences(self, user_id, timeframe, auto_refresh, days_back, pivot_settings, pivot_proximity=5):
        """Save user preferences"""
        try:
            data = {
                'user_id': user_id,
                'timeframe': timeframe,
                'auto_refresh': auto_refresh,
                'days_back': days_back,
                'pivot_settings': json.dumps(pivot_settings),
                'pivot_proximity': pivot_proximity,
                'updated_at': datetime.now(pytz.UTC).isoformat()
            }
            
            self.client.table('user_preferences').upsert(
                data, 
                on_conflict="user_id"
            ).execute()
            
        except Exception as e:
            if "23505" not in str(e) and "duplicate key" not in str(e).lower():
                st.error(f"Error saving preferences: {str(e)}")
    
    def get_user_preferences(self, user_id):
        """Get user preferences"""
        try:
            result = self.client.table('user_preferences')\
                .select('*')\
                .eq('user_id', user_id)\
                .execute()
            
            if result.data:
                prefs = result.data[0]
                if 'pivot_settings' in prefs and prefs['pivot_settings']:
                    prefs['pivot_settings'] = json.loads(prefs['pivot_settings'])
                else:
                    prefs['pivot_settings'] = {
                        'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True
                    }
                return prefs
            else:
                return {
                    'timeframe': '1',
                    'auto_refresh': True,
                    'days_back': 1,
                    'pivot_proximity': 5,
                    'pivot_settings': {
                        'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True
                    }
                }
                
        except Exception as e:
            st.error(f"Error retrieving preferences: {str(e)}")
            return {
                'timeframe': '1',
                'auto_refresh': True,
                'days_back': 1,
                'pivot_proximity': 5,
                'pivot_settings': {
                    'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True
                }
            }
    
    def save_market_analytics(self, symbol, analytics_data):
        """Save daily market analytics"""
        try:
            today = datetime.now(pytz.timezone('Asia/Kolkata')).date()
            
            data = {
                'symbol': symbol,
                'date': today.isoformat(),
                'day_high': analytics_data['day_high'],
                'day_low': analytics_data['day_low'],
                'day_open': analytics_data['day_open'],
                'day_close': analytics_data['day_close'],
                'total_volume': analytics_data['total_volume'],
                'avg_price': analytics_data['avg_price'],
                'price_change': analytics_data['price_change'],
                'price_change_pct': analytics_data['price_change_pct']
            }
            
            self.client.table('market_analytics').upsert(
                data, 
                on_conflict="symbol,date"
            ).execute()
            
        except Exception as e:
            if "23505" not in str(e) and "duplicate key" not in str(e).lower():
                st.error(f"Error saving analytics: {str(e)}")
    
    def get_market_analytics(self, symbol, days_back=30):
        """Get historical market analytics"""
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days_back)
            
            result = self.client.table('market_analytics')\
                .select('*')\
                .eq('symbol', symbol)\
                .gte('date', cutoff_date.isoformat())\
                .order('date', desc=False)\
                .execute()
            
            if result.data:
                return pd.DataFrame(result.data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error retrieving analytics: {str(e)}")
            return pd.DataFrame()

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
        """Get intraday historical data"""
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
        """Get Last Traded Price"""
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

@st.cache_data(ttl=300)  # Cache expiry list for 5 minutes
def get_dhan_expiry_list_cached(underlying_scrip: int, underlying_seg: str):
    return get_dhan_expiry_list(underlying_scrip, underlying_seg)

def get_dhan_option_chain(underlying_scrip: int, underlying_seg: str, expiry: str, max_retries: int = 4):
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    url = "https://api.dhan.co/v2/optionchain"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    payload = {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg,
        "Expiry": expiry
    }

    # Retry logic with exponential backoff for rate limiting (429 errors)
    import time
    retry_delays = [2, 4, 8, 16]  # Exponential backoff: 2s, 4s, 8s, 16s

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload)

            # Handle rate limiting (429) with retry
            if response.status_code == 429:
                if attempt < max_retries:
                    delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                    st.warning(f"⏳ Rate limited by Dhan API. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    st.error("❌ Rate limit exceeded after multiple retries. Please wait a moment and refresh.")
                    return None

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries and "429" in str(e):
                delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                st.warning(f"⏳ Rate limited. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            st.error(f"Error fetching Dhan option chain: {e}")
            return None

    return None

def get_dhan_expiry_list(underlying_scrip: int, underlying_seg: str, max_retries: int = 4):
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    payload = {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg
    }

    # Retry logic with exponential backoff for rate limiting (429 errors)
    import time
    retry_delays = [2, 4, 8, 16]  # Exponential backoff: 2s, 4s, 8s, 16s

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload)

            # Handle rate limiting (429) with retry
            if response.status_code == 429:
                if attempt < max_retries:
                    delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                    st.warning(f"⏳ Rate limited by Dhan API. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    st.error("❌ Rate limit exceeded after multiple retries. Please wait a moment and refresh.")
                    return None

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries and "429" in str(e):
                delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                st.warning(f"⏳ Rate limited. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            st.error(f"Error fetching Dhan expiry list: {e}")
            return None

    return None

class PivotIndicator:
    """Higher Timeframe Pivot Support/Resistance Indicator"""
    
    @staticmethod
    def pivot_high(series, left, right):
        """Detect pivot highs"""
        max_values = series.rolling(window=left+right+1, center=True).max()
        return series == max_values
    
    @staticmethod
    def pivot_low(series, left, right):
        """Detect pivot lows"""
        min_values = series.rolling(window=left+right+1, center=True).min()
        return series == min_values
    
    @staticmethod
    def resample_ohlc(df, tf):
        """Resample OHLC data to higher timeframes"""
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
        """Calculate pivot highs and lows for a given timeframe"""
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
        """Get pivots for all configured timeframes"""
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
    """
    Volume Order Blocks Indicator - Converted from Pine Script [BigBeluga]

    Detects bullish and bearish order blocks based on EMA crossovers with volume analysis.
    - Bullish VOB: EMA cross up -> finds lowest point as support zone
    - Bearish VOB: EMA cross down -> finds highest point as resistance zone
    - Tracks volume collected and percentage distribution for each block
    """

    def __init__(self, sensitivity=5):
        """
        Initialize VOB detector

        Args:
            sensitivity: Detection sensitivity (default 5, maps to length1 in Pine Script)
        """
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.max_blocks = 15  # Maximum blocks to track

    @staticmethod
    def calculate_ema(series, period):
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_atr(df, period=200):
        """Calculate Average True Range"""
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
        """
        Detect Volume Order Blocks from OHLCV data

        Args:
            df: DataFrame with columns ['datetime', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            dict with 'bullish' and 'bearish' lists of order blocks
        """
        if df.empty or len(df) < self.length2 + 10:
            return {'bullish': [], 'bearish': []}

        df = df.copy().reset_index(drop=True)

        # Calculate EMAs
        ema_fast = self.calculate_ema(df['close'], self.length1)
        ema_slow = self.calculate_ema(df['close'], self.length2)

        # Calculate ATR for minimum zone size
        atr = self.calculate_atr(df)
        max_atr = atr.rolling(window=200, min_periods=1).max()
        atr_threshold = max_atr * 2  # atr1 in Pine Script
        overlap_threshold = max_atr * 3  # atr in Pine Script for overlap check

        # Detect crossovers
        cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

        bullish_blocks = []
        bearish_blocks = []

        # Process bullish crossovers (support zones)
        for idx in df[cross_up].index:
            if idx < self.length2:
                continue

            # Find lowest low in lookback period
            lookback_start = max(0, idx - self.length2)
            lookback_df = df.loc[lookback_start:idx]

            lowest_idx = lookback_df['low'].idxmin()
            lowest = df.loc[lowest_idx, 'low']

            # Calculate volume from lowest to crossover
            vol = df.loc[lowest_idx:idx, 'volume'].sum()

            # Get upper bound (min of open/close at the lowest candle)
            upper = min(df.loc[lowest_idx, 'open'], df.loc[lowest_idx, 'close'])

            # Ensure minimum zone size
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

        # Process bearish crossovers (resistance zones)
        for idx in df[cross_down].index:
            if idx < self.length2:
                continue

            # Find highest high in lookback period
            lookback_start = max(0, idx - self.length2)
            lookback_df = df.loc[lookback_start:idx]

            highest_idx = lookback_df['high'].idxmax()
            highest = df.loc[highest_idx, 'high']

            # Calculate volume from highest to crossover
            vol = df.loc[highest_idx:idx, 'volume'].sum()

            # Get lower bound (max of open/close at the highest candle)
            lower = max(df.loc[highest_idx, 'open'], df.loc[highest_idx, 'close'])

            # Ensure minimum zone size
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

        # Remove overlapping blocks and broken blocks
        current_close = df['close'].iloc[-1]

        # Filter bullish blocks - remove if price closed below lower
        bullish_blocks = [b for b in bullish_blocks if current_close >= b['lower']]

        # Filter bearish blocks - remove if price closed above upper
        bearish_blocks = [b for b in bearish_blocks if current_close <= b['upper']]

        # Remove overlapping blocks (keep the one with more volume)
        bullish_blocks = self._remove_overlaps(bullish_blocks, overlap_threshold.iloc[-1] if len(overlap_threshold) > 0 else 50)
        bearish_blocks = self._remove_overlaps(bearish_blocks, overlap_threshold.iloc[-1] if len(overlap_threshold) > 0 else 50)

        # Keep only most recent blocks
        bullish_blocks = bullish_blocks[-self.max_blocks:]
        bearish_blocks = bearish_blocks[-self.max_blocks:]

        # Calculate volume percentages
        total_bull_vol = sum(b['volume'] for b in bullish_blocks) if bullish_blocks else 1
        total_bear_vol = sum(b['volume'] for b in bearish_blocks) if bearish_blocks else 1

        for b in bullish_blocks:
            b['volume_pct'] = (b['volume'] / total_bull_vol * 100) if total_bull_vol > 0 else 0

        for b in bearish_blocks:
            b['volume_pct'] = (b['volume'] / total_bear_vol * 100) if total_bear_vol > 0 else 0

        return {'bullish': bullish_blocks, 'bearish': bearish_blocks}

    def _remove_overlaps(self, blocks, threshold):
        """Remove overlapping blocks, keeping the one with higher volume"""
        if len(blocks) < 2:
            return blocks

        # Sort by mid price
        blocks = sorted(blocks, key=lambda x: x['mid'])

        filtered = []
        for block in blocks:
            overlap = False
            for existing in filtered:
                if abs(block['mid'] - existing['mid']) < threshold:
                    # Overlap detected - keep the one with more volume
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
        """Format volume for display (e.g., 1.5M, 500K)"""
        if vol >= 1_000_000:
            return f"{vol/1_000_000:.1f}M"
        elif vol >= 1_000:
            return f"{vol/1_000:.0f}K"
        else:
            return str(int(vol))

    def get_sr_levels(self, df):
        """
        Get support/resistance levels from VOB for integration with S/R table

        Returns:
            list of dicts compatible with sr_data format
        """
        blocks = self.detect_blocks(df)
        sr_levels = []

        # Add bullish blocks as support levels
        for i, block in enumerate(blocks['bullish']):
            sr_levels.append({
                'Type': '🟢 VOB Support',
                'Level': f"₹{block['mid']:.0f}",
                'Source': f"Vol: {self.format_volume(block['volume'])} ({block['volume_pct']:.1f}%)",
                'Strength': 'VOB Zone',
                'Signal': f"Range: ₹{block['lower']:.0f} - ₹{block['upper']:.0f}",
                'upper': block['upper'],
                'lower': block['lower'],
                'mid': block['mid'],
                'volume': block['volume'],
                'volume_pct': block['volume_pct']
            })

        # Add bearish blocks as resistance levels
        for i, block in enumerate(blocks['bearish']):
            sr_levels.append({
                'Type': '🔴 VOB Resistance',
                'Level': f"₹{block['mid']:.0f}",
                'Source': f"Vol: {self.format_volume(block['volume'])} ({block['volume_pct']:.1f}%)",
                'Strength': 'VOB Zone',
                'Signal': f"Range: ₹{block['lower']:.0f} - ₹{block['upper']:.0f}",
                'upper': block['upper'],
                'lower': block['lower'],
                'mid': block['mid'],
                'volume': block['volume'],
                'volume_pct': block['volume_pct']
            })

        return sr_levels, blocks


class TriplePOC:
    """
    Triple Point of Control (POC) Indicator - Converted from Pine Script [BigBeluga]

    Calculates POC (price level with highest volume) for 3 different periods:
    - POC 1: Short-term (default 10 periods)
    - POC 2: Medium-term (default 25 periods)
    - POC 3: Long-term (default 70 periods)

    POC is computed as a rolling time series (updated every 15 bars in Pine,
    here computed at every bar) and rendered as steplines on the chart.
    """

    def __init__(self, period1=10, period2=25, period3=70, bins=25):
        """
        Initialize Triple POC calculator.

        Args:
            period1: Lookback for POC 1 (short-term)
            period2: Lookback for POC 2 (medium-term)
            period3: Lookback for POC 3 (long-term)
            bins: Number of price bins for volume distribution
        """
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.bins = bins

    def _calculate_poc_series(self, df, period):
        """
        Calculate POC as a rolling time series across all bars.
        Mirrors Pine Script logic: at each bar, look back 'period' bars,
        build volume profile, find POC (max volume level).

        Returns:
            dict with 'poc', 'upper_poc', 'lower_poc' as pandas Series
        """
        n = len(df)
        poc_vals = np.full(n, np.nan)
        upper_vals = np.full(n, np.nan)
        lower_vals = np.full(n, np.nan)

        closes = df['close'].values.astype(float)
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        volumes = df['volume'].values.astype(float) if 'volume' in df.columns else np.ones(n)

        # Pine recalculates every 15 bars (bar_index % 15 == 0).
        # We use a dynamic interval: min(15, period//2) for shorter periods
        recalc_interval = min(15, max(3, period // 3))
        last_poc = np.nan
        last_upper = np.nan
        last_lower = np.nan

        for i in range(period, n):
            if (i - period) % recalc_interval == 0:
                start = i - period
                end = i + 1

                H = highs[start:end].max()
                L = lows[start:end].min()

                if H == L:
                    last_poc = H
                    last_upper = H
                    last_lower = L
                else:
                    step = (H - L) / self.bins
                    vol_bins = np.zeros(self.bins)
                    level_mids = np.zeros(self.bins)

                    for k in range(self.bins):
                        level_mids[k] = L + k * step + step / 2

                    for j in range(start, end):
                        c = closes[j]
                        v = volumes[j]
                        for k in range(self.bins):
                            if abs(c - level_mids[k]) <= step:
                                vol_bins[k] += v

                    max_idx = vol_bins.argmax()
                    last_poc = level_mids[max_idx]
                    last_upper = last_poc + step * 2
                    last_lower = last_poc - step * 2

            poc_vals[i] = last_poc
            upper_vals[i] = last_upper
            lower_vals[i] = last_lower

        return {
            'poc': pd.Series(poc_vals, index=df.index),
            'upper_poc': pd.Series(upper_vals, index=df.index),
            'lower_poc': pd.Series(lower_vals, index=df.index),
        }

    def calculate_poc(self, df, period):
        """
        Calculate single (latest) POC for a given period.
        Used for signal generation and table display.
        """
        if df.empty or len(df) < period:
            return None

        recent_df = df.tail(period).copy()

        H = recent_df['high'].max()
        L = recent_df['low'].min()

        if H == L:
            return {
                'poc': H, 'upper_poc': H, 'lower_poc': L,
                'volume': 0, 'high': H, 'low': L
            }

        step = (H - L) / self.bins
        vol_bins = [0.0] * self.bins
        level_mids = []

        for k in range(self.bins):
            level_mids.append(L + k * step + step / 2)

        for _, row in recent_df.iterrows():
            c = row['close']
            v = row.get('volume', 1)
            for k in range(len(level_mids)):
                if abs(c - level_mids[k]) <= step:
                    vol_bins[k] += v

        max_vol_idx = vol_bins.index(max(vol_bins))
        poc = level_mids[max_vol_idx]

        return {
            'poc': round(poc, 2),
            'upper_poc': round(poc + step * 2, 2),
            'lower_poc': round(poc - step * 2, 2),
            'volume': vol_bins[max_vol_idx],
            'high': H, 'low': L, 'step': step
        }

    def calculate_all_pocs(self, df):
        """
        Calculate all three POCs — both time series (for chart) and latest values (for signals/tables).
        """
        poc1_series = self._calculate_poc_series(df, self.period1)
        poc2_series = self._calculate_poc_series(df, self.period2)
        poc3_series = self._calculate_poc_series(df, self.period3)

        # Latest single values for tables/signals
        poc1 = self.calculate_poc(df, self.period1)
        poc2 = self.calculate_poc(df, self.period2)
        poc3 = self.calculate_poc(df, self.period3)

        return {
            'poc1': poc1,
            'poc2': poc2,
            'poc3': poc3,
            'poc1_series': poc1_series,
            'poc2_series': poc2_series,
            'poc3_series': poc3_series,
            'periods': {
                'poc1': self.period1,
                'poc2': self.period2,
                'poc3': self.period3
            }
        }

    def get_price_position(self, current_price, poc_data):
        """
        Determine price position relative to POC channels.

        Returns:
            'above' if price above POC channel
            'below' if price below POC channel
            'inside' if price inside POC channel
        """
        if poc_data is None:
            return 'unknown'

        if current_price > poc_data['upper_poc']:
            return 'above'
        elif current_price < poc_data['lower_poc']:
            return 'below'
        else:
            return 'inside'


class RSIVolatilitySuppression:
    """
    RSI Volatility Suppression Zones - Converted from Pine Script [BigBeluga]

    Detects zones where RSI volatility is suppressed (low), indicating
    consolidation periods that often precede breakouts.

    When price breaks out of a suppression zone:
    - Upward breakout (low crosses above zone top) → Bullish signal
    - Downward breakout (high crosses below zone bottom) → Bearish signal
    """

    def __init__(self, rsi_length=14, vol_length=5, bins_size=150, zone_threshold=10, extended_threshold=50):
        self.rsi_length = rsi_length
        self.vol_length = vol_length
        self.bins_size = bins_size
        self.zone_threshold = zone_threshold
        self.extended_threshold = extended_threshold

    @staticmethod
    def _hma(series, period):
        """Hull Moving Average"""
        if len(series) < period:
            return series.copy()
        half_period = max(int(period / 2), 1)
        sqrt_period = max(int(np.sqrt(period)), 1)

        wma1 = series.rolling(window=half_period, min_periods=1).apply(
            lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
        )
        wma2 = series.rolling(window=period, min_periods=1).apply(
            lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
        )
        diff = 2 * wma1 - wma2
        hma = diff.rolling(window=sqrt_period, min_periods=1).apply(
            lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
        )
        return hma

    @staticmethod
    def _rsi(series, period):
        """Standard RSI calculation"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_rsi_volatility(self, rsi_series):
        """Calculate historical volatility of RSI (normalized)"""
        log_returns = np.log(rsi_series / rsi_series.shift(1))
        hv = 100 * log_returns.rolling(window=5, min_periods=1).std()
        hv_std = hv.rolling(window=200, min_periods=20).std()
        hv_normalized = hv / hv_std.replace(0, np.nan)
        return hv_normalized.fillna(0)

    def analyze(self, df):
        """
        Analyze RSI volatility suppression zones.

        Returns:
            dict with 'zones' (list of zone dicts), 'rsi' (smoothed RSI series),
            'rsi_volatility' (normalized RSI volatility), 'current_signal'
        """
        if df.empty or len(df) < self.rsi_length + self.vol_length + 10:
            return None

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        hl2 = (high + low) / 2

        # RSI smoothed with HMA
        raw_rsi = self._rsi(close, self.rsi_length)
        rsi = self._hma(raw_rsi, self.vol_length)

        # Average bar range (size)
        bar_range = high - low
        size = bar_range.rolling(window=self.bins_size, min_periods=20).mean()

        # SMA of hl2 for zone center
        sma = hl2.rolling(window=5, min_periods=1).mean()

        # RSI volatility
        rsi_volatility = self._calculate_rsi_volatility(rsi)

        # Build suppression zones bar-by-bar
        zones = []
        count_volatility = 0
        active_zone = None

        for i in range(len(df)):
            rv = rsi_volatility.iloc[i] if not np.isnan(rsi_volatility.iloc[i]) else 0
            curr_low = low.iloc[i]
            curr_high = high.iloc[i]
            curr_sma = sma.iloc[i] if not np.isnan(sma.iloc[i]) else hl2.iloc[i]
            curr_size = size.iloc[i] if not np.isnan(size.iloc[i]) else bar_range.iloc[i]

            prev_rv = rsi_volatility.iloc[i-1] if i > 0 and not np.isnan(rsi_volatility.iloc[i-1]) else 0

            # Count consecutive low-volatility bars
            if rv <= 2:
                count_volatility += 1
            # Reset on crossover above 2
            if rv > 2 and prev_rv <= 2:
                count_volatility = 0

            # Create new zone when count crosses threshold
            prev_count = count_volatility - 1 if rv <= 2 else 0
            zone_top = curr_sma + curr_size * 2
            zone_bottom = curr_sma - curr_size * 2

            if prev_count < self.zone_threshold and count_volatility >= self.zone_threshold and count_volatility > 0:
                active_zone = {
                    'start_idx': max(0, i - self.zone_threshold),
                    'start_time': df['datetime'].iloc[max(0, i - self.zone_threshold)] if 'datetime' in df.columns else None,
                    'end_idx': i,
                    'end_time': df['datetime'].iloc[i] if 'datetime' in df.columns else None,
                    'top': zone_top,
                    'bottom': zone_bottom,
                    'breakout': None,
                    'breakout_idx': None,
                    'breakout_time': None,
                }

            # Extend active zone
            if active_zone and active_zone['breakout'] is None:
                active_zone['end_idx'] = i
                if 'datetime' in df.columns:
                    active_zone['end_time'] = df['datetime'].iloc[i]

            # Check breakout up: low crosses above zone top
            if active_zone and active_zone['breakout'] is None:
                if i > 0 and curr_low > active_zone['top'] and low.iloc[i-1] <= active_zone['top']:
                    active_zone['breakout'] = 'bullish'
                    active_zone['breakout_idx'] = i
                    if 'datetime' in df.columns:
                        active_zone['breakout_time'] = df['datetime'].iloc[i]
                    zones.append(active_zone.copy())
                    active_zone = None

            # Check breakout down: high crosses below zone bottom
            if active_zone and active_zone['breakout'] is None:
                if i > 0 and curr_high < active_zone['bottom'] and high.iloc[i-1] >= active_zone['bottom']:
                    active_zone['breakout'] = 'bearish'
                    active_zone['breakout_idx'] = i
                    if 'datetime' in df.columns:
                        active_zone['breakout_time'] = df['datetime'].iloc[i]
                    zones.append(active_zone.copy())
                    active_zone = None

            # Extended zone: after 50 bars of low vol with no active zone
            if active_zone is None and prev_count < self.extended_threshold and count_volatility >= self.extended_threshold:
                active_zone = {
                    'start_idx': max(0, i - self.zone_threshold),
                    'start_time': df['datetime'].iloc[max(0, i - self.zone_threshold)] if 'datetime' in df.columns else None,
                    'end_idx': i,
                    'end_time': df['datetime'].iloc[i] if 'datetime' in df.columns else None,
                    'top': zone_top,
                    'bottom': zone_bottom,
                    'breakout': None,
                    'breakout_idx': None,
                    'breakout_time': None,
                }

        # Add still-active zone (no breakout yet)
        if active_zone and active_zone['breakout'] is None:
            active_zone['breakout'] = 'pending'
            zones.append(active_zone.copy())

        # Current signal
        current_signal = 'No Zone'
        if zones:
            last_zone = zones[-1]
            if last_zone['breakout'] == 'pending':
                current_signal = 'In Suppression Zone'
            elif last_zone['breakout'] == 'bullish':
                current_signal = 'Bullish Breakout'
            elif last_zone['breakout'] == 'bearish':
                current_signal = 'Bearish Breakout'

        return {
            'zones': zones,
            'rsi': rsi,
            'rsi_volatility': rsi_volatility,
            'current_signal': current_signal,
            'count_volatility': count_volatility,
        }


class UltimateRSI:
    """
    Ultimate RSI Indicator - Converted from Pine Script [LuxAlgo]

    An augmented RSI that uses highest/lowest range to detect momentum shifts
    more accurately than standard RSI. Includes signal line and OB/OS zones.

    Key differences from standard RSI:
    - Uses highest-lowest range as the diff when range expands/contracts
    - Smoothed with selectable MA type (RMA default)
    - Signal line (EMA of RSI) for crossover detection
    """

    def __init__(self, length=14, smo_type='RMA', signal_length=14, signal_type='EMA',
                 ob_value=80, os_value=20):
        self.length = length
        self.smo_type = smo_type
        self.signal_length = signal_length
        self.signal_type = signal_type
        self.ob_value = ob_value
        self.os_value = os_value

    @staticmethod
    def _ma(series, length, ma_type):
        """Moving average with selectable type"""
        if ma_type == 'EMA':
            return series.ewm(span=length, adjust=False).mean()
        elif ma_type == 'SMA':
            return series.rolling(window=length, min_periods=1).mean()
        elif ma_type == 'RMA':
            return series.ewm(alpha=1/length, adjust=False).mean()
        elif ma_type == 'TMA':
            sma1 = series.rolling(window=length, min_periods=1).mean()
            return sma1.rolling(window=length, min_periods=1).mean()
        return series

    def calculate(self, df):
        """
        Calculate Ultimate RSI time series.

        Returns:
            dict with 'arsi' (Series), 'signal' (Series), 'ob', 'os',
            'latest_arsi', 'latest_signal', 'zone', 'cross_signal'
        """
        if df.empty or len(df) < self.length + 5:
            return None

        src = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)

        # Augmented RSI calculation (Pine Script logic)
        upper = high.rolling(window=self.length, min_periods=1).max()
        lower = low.rolling(window=self.length, min_periods=1).min()
        r = upper - lower

        d = src.diff()  # src - src[1]

        # diff = upper > upper[1] ? r : lower < lower[1] ? -r : d
        upper_expanded = upper > upper.shift(1)
        lower_expanded = lower < lower.shift(1)

        diff = pd.Series(np.where(
            upper_expanded, r,
            np.where(lower_expanded, -r, d)
        ), index=df.index, dtype=float)

        # num = ma(diff, length); den = ma(abs(diff), length)
        num = self._ma(diff, self.length, self.smo_type)
        den = self._ma(diff.abs(), self.length, self.smo_type)

        # arsi = num/den * 50 + 50
        arsi = (num / den.replace(0, np.nan) * 50 + 50).fillna(50)

        # Signal line
        signal = self._ma(arsi, self.signal_length, self.signal_type)

        # Latest values
        latest_arsi = arsi.iloc[-1]
        latest_signal = signal.iloc[-1]

        # Zone determination
        if latest_arsi > self.ob_value:
            zone = 'Overbought'
        elif latest_arsi < self.os_value:
            zone = 'Oversold'
        else:
            zone = 'Neutral'

        # Crossover signals
        prev_arsi = arsi.iloc[-2] if len(arsi) > 1 else 50
        prev_signal = signal.iloc[-2] if len(signal) > 1 else 50

        cross_signal = 'None'
        if prev_arsi <= prev_signal and latest_arsi > latest_signal:
            cross_signal = 'Bullish Cross'
        elif prev_arsi >= prev_signal and latest_arsi < latest_signal:
            cross_signal = 'Bearish Cross'

        # Momentum direction
        if latest_arsi > 50 and latest_arsi > latest_signal:
            momentum = 'Bullish'
        elif latest_arsi < 50 and latest_arsi < latest_signal:
            momentum = 'Bearish'
        else:
            momentum = 'Neutral'

        return {
            'arsi': arsi,
            'signal': signal,
            'ob': self.ob_value,
            'os': self.os_value,
            'latest_arsi': round(latest_arsi, 2),
            'latest_signal': round(latest_signal, 2),
            'zone': zone,
            'cross_signal': cross_signal,
            'momentum': momentum,
        }


class FutureSwing:
    """
    Future Swing Projection Indicator - Converted from Pine Script [BigBeluga]

    Detects swing highs and lows, then projects future swing targets
    based on historical swing percentages.

    Key concepts:
    - Swing High: Highest point before price reverses down
    - Swing Low: Lowest point before price reverses up
    - Future projection: Based on average/median/mode of historical swings
    """

    def __init__(self, swing_length=30, projection_offset=10, history_samples=5, calc_type='Average'):
        """
        Initialize Future Swing calculator.

        Args:
            swing_length: Bars to detect swing highs/lows
            projection_offset: How far to project into future
            history_samples: Number of historical swings to use
            calc_type: 'Average', 'Median', or 'Mode'
        """
        self.swing_length = swing_length
        self.projection_offset = projection_offset
        self.history_samples = history_samples
        self.calc_type = calc_type

    def detect_swings(self, df):
        """
        Detect swing highs and lows in the data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            dict with swing_highs, swing_lows, and current direction
        """
        if df.empty or len(df) < self.swing_length + 1:
            return None

        df = df.copy().reset_index(drop=True)

        swing_highs = []
        swing_lows = []

        # Calculate rolling high and low
        df['rolling_high'] = df['high'].rolling(window=self.swing_length, min_periods=1).max()
        df['rolling_low'] = df['low'].rolling(window=self.swing_length, min_periods=1).min()

        # Detect swing points
        for i in range(self.swing_length, len(df) - 1):
            # Swing High: current high equals rolling high and next bar starts moving down
            if df.loc[i, 'high'] == df.loc[i, 'rolling_high']:
                # Check if this is a confirmed swing high (price moved away)
                if i + 1 < len(df) and df.loc[i + 1, 'high'] < df.loc[i, 'rolling_high']:
                    swing_highs.append({
                        'index': i,
                        'value': df.loc[i, 'high'],
                        'datetime': df.loc[i, 'datetime'] if 'datetime' in df.columns else None
                    })

            # Swing Low: current low equals rolling low and next bar starts moving up
            if df.loc[i, 'low'] == df.loc[i, 'rolling_low']:
                # Check if this is a confirmed swing low (price moved away)
                if i + 1 < len(df) and df.loc[i + 1, 'low'] > df.loc[i, 'rolling_low']:
                    swing_lows.append({
                        'index': i,
                        'value': df.loc[i, 'low'],
                        'datetime': df.loc[i, 'datetime'] if 'datetime' in df.columns else None
                    })

        # Determine current direction
        last_high_idx = swing_highs[-1]['index'] if swing_highs else 0
        last_low_idx = swing_lows[-1]['index'] if swing_lows else 0

        # Direction: True = bearish (last swing was high), False = bullish (last swing was low)
        direction = 'bearish' if last_high_idx > last_low_idx else 'bullish'

        return {
            'swing_highs': swing_highs[-self.history_samples:] if swing_highs else [],
            'swing_lows': swing_lows[-self.history_samples:] if swing_lows else [],
            'direction': direction,
            'last_swing_high': swing_highs[-1] if swing_highs else None,
            'last_swing_low': swing_lows[-1] if swing_lows else None
        }

    def calculate_swing_percentages(self, swing_data):
        """
        Calculate percentage moves between swing highs and lows.

        Returns:
            list of swing percentages
        """
        if swing_data is None:
            return []

        swing_highs = swing_data['swing_highs']
        swing_lows = swing_data['swing_lows']

        if not swing_highs or not swing_lows:
            return []

        percentages = []

        # Combine and sort all swings by index
        all_swings = []
        for sh in swing_highs:
            all_swings.append({'type': 'high', **sh})
        for sl in swing_lows:
            all_swings.append({'type': 'low', **sl})

        all_swings.sort(key=lambda x: x['index'])

        # Calculate percentage moves between consecutive swings
        for i in range(1, len(all_swings)):
            prev = all_swings[i - 1]
            curr = all_swings[i]

            if prev['type'] == 'low' and curr['type'] == 'high':
                # Bullish swing: low to high
                pct = (curr['value'] - prev['value']) / prev['value'] * 100
                percentages.append(pct)
            elif prev['type'] == 'high' and curr['type'] == 'low':
                # Bearish swing: high to low
                pct = (curr['value'] - prev['value']) / prev['value'] * 100
                percentages.append(pct)

        return percentages[-self.history_samples:]

    def project_future_swing(self, swing_data, percentages):
        """
        Project future swing target based on historical percentages.

        Returns:
            dict with projected target and calculation details
        """
        if not percentages or swing_data is None:
            return None

        abs_percentages = [abs(p) for p in percentages]

        # Calculate swing value based on method
        if self.calc_type == 'Average':
            swing_val = sum(abs_percentages) / len(abs_percentages)
        elif self.calc_type == 'Median':
            sorted_pct = sorted(abs_percentages)
            mid = len(sorted_pct) // 2
            swing_val = sorted_pct[mid] if len(sorted_pct) % 2 == 1 else (sorted_pct[mid-1] + sorted_pct[mid]) / 2
        else:  # Mode
            from collections import Counter
            rounded = [round(p, 1) for p in abs_percentages]
            counter = Counter(rounded)
            swing_val = counter.most_common(1)[0][0]

        direction = swing_data['direction']
        last_high = swing_data['last_swing_high']
        last_low = swing_data['last_swing_low']

        if direction == 'bearish' and last_high:
            # Project downward from last high
            target = last_high['value'] - (last_high['value'] * (swing_val / 100))
            return {
                'direction': 'bearish',
                'from_value': last_high['value'],
                'target': round(target, 2),
                'swing_pct': round(swing_val, 2),
                'sign': '-'
            }
        elif direction == 'bullish' and last_low:
            # Project upward from last low
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
        """
        Calculate buy/sell volume delta for current swing leg.

        Returns:
            dict with buy_volume, sell_volume, delta
        """
        if df.empty or swing_data is None:
            return {'buy_volume': 0, 'sell_volume': 0, 'delta': 0, 'total': 0}

        last_high = swing_data['last_swing_high']
        last_low = swing_data['last_swing_low']

        if not last_high or not last_low:
            return {'buy_volume': 0, 'sell_volume': 0, 'delta': 0, 'total': 0}

        # Get starting index for volume calculation
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
        """
        Perform complete swing analysis.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Complete analysis dict
        """
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


class VWAPIndicator:
    """
    Volume Weighted Average Price (VWAP) Indicator with Standard Deviation / Percentage Bands.

    Converted from TradingView Pine Script v6.

    Features:
    - Session-anchored VWAP calculation using HLC/3 (typical price)
    - Configurable anchor periods (Session default for intraday)
    - Up to 3 configurable bands (Standard Deviation or Percentage mode)
    - Hides on daily/weekly/monthly timeframes if configured
    """

    def __init__(self, src='hlc3', anchor='Session', calc_mode='Standard Deviation',
                 show_band_1=True, band_mult_1=1.0,
                 show_band_2=False, band_mult_2=2.0,
                 show_band_3=False, band_mult_3=3.0):
        self.src = src
        self.anchor = anchor
        self.calc_mode = calc_mode
        self.show_band_1 = show_band_1
        self.band_mult_1 = band_mult_1
        self.show_band_2 = show_band_2
        self.band_mult_2 = band_mult_2
        self.show_band_3 = show_band_3
        self.band_mult_3 = band_mult_3

    @staticmethod
    def _get_source(df, src='hlc3'):
        """Calculate source price series"""
        if src == 'hlc3':
            return (df['high'] + df['low'] + df['close']) / 3
        elif src == 'close':
            return df['close']
        elif src == 'open':
            return df['open']
        elif src == 'hl2':
            return (df['high'] + df['low']) / 2
        elif src == 'ohlc4':
            return (df['open'] + df['high'] + df['low'] + df['close']) / 4
        return (df['high'] + df['low'] + df['close']) / 3

    @staticmethod
    def _detect_new_period(df, anchor='Session'):
        """Detect anchor period boundaries (new session/day start)"""
        if df.empty or 'datetime' not in df.columns:
            return pd.Series([False] * len(df), index=df.index)

        dt = df['datetime']
        new_period = pd.Series([False] * len(df), index=df.index)
        new_period.iloc[0] = True  # First bar is always a new period

        if anchor == 'Session':
            # New session = new trading day
            dates = dt.dt.date
            new_period = dates != dates.shift(1)
            new_period.iloc[0] = True
        elif anchor == 'Week':
            weeks = dt.dt.isocalendar().week.astype(int)
            new_period = weeks != weeks.shift(1)
            new_period.iloc[0] = True
        elif anchor == 'Month':
            months = dt.dt.month
            new_period = months != months.shift(1)
            new_period.iloc[0] = True
        elif anchor == 'Quarter':
            quarters = dt.dt.quarter
            new_period = quarters != quarters.shift(1)
            new_period.iloc[0] = True
        elif anchor == 'Year':
            years = dt.dt.year
            new_period = years != years.shift(1)
            new_period.iloc[0] = True

        return new_period

    def calculate(self, df):
        """
        Calculate VWAP with bands.

        Returns dict with:
        - vwap: VWAP series
        - upper_band_1/2/3: Upper band series
        - lower_band_1/2/3: Lower band series
        - latest values for tabular display
        """
        if df.empty or 'volume' not in df.columns:
            return None

        df = df.copy()
        src = self._get_source(df, self.src)
        vol = df['volume'].values
        new_period = self._detect_new_period(df, self.anchor).values

        n = len(df)
        vwap = np.full(n, np.nan)
        upper_1 = np.full(n, np.nan)
        lower_1 = np.full(n, np.nan)
        upper_2 = np.full(n, np.nan)
        lower_2 = np.full(n, np.nan)
        upper_3 = np.full(n, np.nan)
        lower_3 = np.full(n, np.nan)

        cum_vol = 0.0
        cum_tp_vol = 0.0
        cum_tp2_vol = 0.0  # For standard deviation calculation

        src_values = src.values

        for i in range(n):
            if new_period[i]:
                cum_vol = 0.0
                cum_tp_vol = 0.0
                cum_tp2_vol = 0.0

            v = vol[i]
            s = src_values[i]

            if np.isnan(v) or np.isnan(s) or v == 0:
                if i > 0:
                    vwap[i] = vwap[i-1]
                continue

            cum_vol += v
            cum_tp_vol += s * v
            cum_tp2_vol += s * s * v

            if cum_vol > 0:
                vwap_val = cum_tp_vol / cum_vol
                vwap[i] = vwap_val

                # Standard deviation: sqrt(E[X^2] - E[X]^2)
                variance = (cum_tp2_vol / cum_vol) - (vwap_val * vwap_val)
                stdev = np.sqrt(max(variance, 0))

                if self.calc_mode == 'Standard Deviation':
                    band_basis = stdev
                else:  # Percentage
                    band_basis = vwap_val * 0.01

                if self.show_band_1:
                    upper_1[i] = vwap_val + band_basis * self.band_mult_1
                    lower_1[i] = vwap_val - band_basis * self.band_mult_1
                if self.show_band_2:
                    upper_2[i] = vwap_val + band_basis * self.band_mult_2
                    lower_2[i] = vwap_val - band_basis * self.band_mult_2
                if self.show_band_3:
                    upper_3[i] = vwap_val + band_basis * self.band_mult_3
                    lower_3[i] = vwap_val - band_basis * self.band_mult_3

        # Build result
        vwap_series = pd.Series(vwap, index=df.index)
        result = {
            'vwap': vwap_series,
            'upper_band_1': pd.Series(upper_1, index=df.index) if self.show_band_1 else None,
            'lower_band_1': pd.Series(lower_1, index=df.index) if self.show_band_1 else None,
            'upper_band_2': pd.Series(upper_2, index=df.index) if self.show_band_2 else None,
            'lower_band_2': pd.Series(lower_2, index=df.index) if self.show_band_2 else None,
            'upper_band_3': pd.Series(upper_3, index=df.index) if self.show_band_3 else None,
            'lower_band_3': pd.Series(lower_3, index=df.index) if self.show_band_3 else None,
            'show_band_1': self.show_band_1,
            'show_band_2': self.show_band_2,
            'show_band_3': self.show_band_3,
            'band_mult_1': self.band_mult_1,
            'band_mult_2': self.band_mult_2,
            'band_mult_3': self.band_mult_3,
            'calc_mode': self.calc_mode,
            'anchor': self.anchor,
        }

        # Latest values for tabular display
        last_valid = vwap_series.last_valid_index()
        if last_valid is not None:
            idx = last_valid
            result['latest_vwap'] = round(vwap_series[idx], 2)
            result['latest_upper_1'] = round(upper_1[df.index.get_loc(idx)], 2) if self.show_band_1 else None
            result['latest_lower_1'] = round(lower_1[df.index.get_loc(idx)], 2) if self.show_band_1 else None
            result['latest_upper_2'] = round(upper_2[df.index.get_loc(idx)], 2) if self.show_band_2 else None
            result['latest_lower_2'] = round(lower_2[df.index.get_loc(idx)], 2) if self.show_band_2 else None
            result['latest_upper_3'] = round(upper_3[df.index.get_loc(idx)], 2) if self.show_band_3 else None
            result['latest_lower_3'] = round(lower_3[df.index.get_loc(idx)], 2) if self.show_band_3 else None

            # Price position relative to VWAP
            if not df.empty:
                current_price = df['close'].iloc[-1]
                result['current_price'] = round(current_price, 2)
                result['price_vs_vwap'] = round(current_price - result['latest_vwap'], 2)
                result['price_vs_vwap_pct'] = round(((current_price / result['latest_vwap']) - 1) * 100, 3)

                # Determine band position
                if self.show_band_1 and result['latest_upper_1'] is not None:
                    if current_price > result['latest_upper_1']:
                        result['band_position'] = 'Above Band 1'
                    elif current_price < result['latest_lower_1']:
                        result['band_position'] = 'Below Band 1'
                    else:
                        result['band_position'] = 'Inside Band 1'

                    if self.show_band_2 and result['latest_upper_2'] is not None:
                        if current_price > result['latest_upper_2']:
                            result['band_position'] = 'Above Band 2'
                        elif current_price < result['latest_lower_2']:
                            result['band_position'] = 'Below Band 2'

                    if self.show_band_3 and result['latest_upper_3'] is not None:
                        if current_price > result['latest_upper_3']:
                            result['band_position'] = 'Above Band 3'
                        elif current_price < result['latest_lower_3']:
                            result['band_position'] = 'Below Band 3'
                else:
                    if current_price > result['latest_vwap']:
                        result['band_position'] = 'Above VWAP'
                    else:
                        result['band_position'] = 'Below VWAP'

        return result


class ReversalDetector:
    """
    Intraday Reversal Detection System based on Price Action Theory.

    Implements the following conditions:
    A) Selling pressure exhaustion (no new lows)
    B) Institutional buying (volume clues)
    C) Short covering (sudden strong recovery)
    D) Support level respected

    Entry Rules:
    1) Don't chase first green candle
    2) Structure: No new low + Higher low + Strong bullish candle
    3) Volume must support the move
    4) Safe CE entry: Price above VWAP or breaks recent high
    """

    @staticmethod
    def calculate_vwap(df):
        """Calculate VWAP (Volume Weighted Average Price)"""
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
        """
        Detect Higher Low formation.
        Returns True if current low is higher than previous swing low.
        """
        if len(df) < lookback + 1:
            return False, None, None

        recent = df.tail(lookback + 1)
        lows = recent['low'].values

        # Find the minimum in the lookback period (excluding last candle)
        prev_min_idx = lows[:-1].argmin()
        prev_min = lows[prev_min_idx]

        # Check if current low is higher than previous minimum
        current_low = lows[-1]

        # Also check if we have a swing low pattern (V shape)
        if len(lows) >= 3:
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    swing_low = lows[i]
                    if current_low > swing_low:
                        return True, swing_low, current_low

        return current_low > prev_min, prev_min, current_low

    @staticmethod
    def detect_no_new_low(df, lookback=10):
        """
        Detect if price has stopped making new lows.
        Returns True if no new low made in recent candles.
        """
        if len(df) < lookback:
            return False, None

        recent = df.tail(lookback)
        lows = recent['low'].values

        # Find where the lowest low occurred
        min_idx = lows.argmin()

        # If minimum is not in the last 2 candles, selling pressure may be exhausted
        selling_exhausted = min_idx < lookback - 2

        return selling_exhausted, lows.min()

    @staticmethod
    def detect_strong_bullish_candle(df, threshold=0.5):
        """
        Detect strong bullish candle:
        - Close > Open (green candle)
        - Body > threshold * total range
        - Close above previous candle's high
        """
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
        """
        Check if current up candle has supporting volume.
        - Up candle + volume > average = Real buying
        - Up candle + volume < average = Fake bounce
        """
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
        """
        Check if price is above VWAP.
        Price above VWAP = Bullish bias
        """
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
        """
        Check if price is respecting a support level.
        Returns True if price bounced from near a support level.
        """
        if len(df) < 3 or not pivot_lows:
            return False, None, None

        current_low = df.iloc[-1]['low']
        recent_low = df.tail(5)['low'].min()

        # Find nearest support
        nearest_support = None
        min_distance = float('inf')

        for support in pivot_lows:
            distance = abs(recent_low - support)
            pct_distance = (distance / support) * 100 if support > 0 else float('inf')

            if pct_distance < min_distance and pct_distance <= proximity_pct:
                min_distance = pct_distance
                nearest_support = support

        if nearest_support:
            # Check if price bounced (current close > recent low)
            bounced = df.iloc[-1]['close'] > recent_low
            return bounced, nearest_support, recent_low

        return False, None, recent_low

    @staticmethod
    def calculate_reversal_score(df, pivot_lows=None, lookback=10):
        """
        Calculate comprehensive reversal score based on all conditions.

        Returns:
        - score: -5 to +5 (positive = bullish reversal, negative = bearish)
        - signals: dict with individual signal details
        - verdict: Overall recommendation
        """
        signals = {}
        score = 0

        # 1. Check selling pressure exhaustion (no new low)
        no_new_low, swing_low = ReversalDetector.detect_no_new_low(df, lookback)
        signals['Selling_Exhausted'] = "Yes ✅" if no_new_low else "No ❌"
        if no_new_low:
            score += 1

        # 2. Check higher low formation
        higher_low, prev_low, curr_low = ReversalDetector.detect_higher_low(df, lookback // 2)
        signals['Higher_Low'] = "Yes ✅" if higher_low else "No ❌"
        if higher_low:
            score += 1.5

        # 3. Check strong bullish candle
        strong_candle, candle_details = ReversalDetector.detect_strong_bullish_candle(df)
        signals['Strong_Bullish_Candle'] = "Yes ✅" if strong_candle else "No ❌"
        if strong_candle:
            score += 1.5

        # 4. Check volume confirmation
        vol_confirmed, vol_signal, vol_details = ReversalDetector.detect_volume_confirmation(df)
        signals['Volume_Signal'] = vol_signal
        if vol_confirmed:
            score += 1
        elif vol_signal == "Weak/Fake Bounce":
            score -= 0.5

        # 5. Check VWAP position
        above_vwap, price, vwap = ReversalDetector.check_vwap_position(df)
        signals['Above_VWAP'] = "Yes ✅" if above_vwap else "No ❌"
        if above_vwap:
            score += 1

        # 6. Check support respect (if pivot lows provided)
        if pivot_lows:
            support_held, support_level, low = ReversalDetector.detect_support_respect(df, pivot_lows)
            signals['Support_Respected'] = "Yes ✅" if support_held else "No ❌"
            if support_held:
                score += 1
                signals['Support_Level'] = support_level

        # Calculate entry signal
        signals['Reversal_Score'] = round(score, 1)

        # Determine verdict
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

        # Add price context
        if len(df) > 0:
            signals['Current_Price'] = df.iloc[-1]['close']
            signals['Day_Low'] = df['low'].min()
            signals['Day_High'] = df['high'].max()
            if vwap:
                signals['VWAP'] = round(vwap, 2)

        return score, signals, verdict

    @staticmethod
    def get_entry_rules(signals, score):
        """
        Generate specific entry rules based on current conditions.
        """
        rules = []

        # Rule 1: Don't chase first green candle
        if signals.get('Strong_Bullish_Candle') == "Yes ✅":
            if signals.get('Higher_Low') != "Yes ✅":
                rules.append("⚠️ First green candle - Wait for higher low confirmation")
            else:
                rules.append("✅ Structure confirmed - Entry possible")

        # Rule 2: Volume check
        vol_signal = signals.get('Volume_Signal', '')
        if 'Weak' in vol_signal or 'Fake' in vol_signal:
            rules.append("⚠️ Low volume - Possible fake bounce")
        elif 'Strong' in vol_signal:
            rules.append("✅ Strong volume - Real buying detected")

        # Rule 3: VWAP position
        if signals.get('Above_VWAP') == "Yes ✅":
            rules.append("✅ Price above VWAP - Bullish bias")
        else:
            rules.append("⚠️ Price below VWAP - Wait for VWAP reclaim")

        # Rule 4: Entry recommendation
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
        """
        Detect Lower High formation (bearish pattern).
        Returns True if current high is lower than previous swing high.
        """
        if len(df) < lookback + 1:
            return False, None, None

        recent = df.tail(lookback + 1)
        highs = recent['high'].values

        # Find the maximum in the lookback period (excluding last candle)
        prev_max_idx = highs[:-1].argmax()
        prev_max = highs[prev_max_idx]

        current_high = highs[-1]

        # Check if we have a swing high pattern (inverted V shape)
        if len(highs) >= 3:
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    swing_high = highs[i]
                    if current_high < swing_high:
                        return True, swing_high, current_high

        return current_high < prev_max, prev_max, current_high

    @staticmethod
    def detect_no_new_high(df, lookback=10):
        """
        Detect if price has stopped making new highs (buying exhausted).
        Returns True if no new high made in recent candles.
        """
        if len(df) < lookback:
            return False, None

        recent = df.tail(lookback)
        highs = recent['high'].values

        # Find where the highest high occurred
        max_idx = highs.argmax()

        # If maximum is not in the last 2 candles, buying pressure may be exhausted
        buying_exhausted = max_idx < lookback - 2

        return buying_exhausted, highs.max()

    @staticmethod
    def detect_strong_bearish_candle(df, threshold=0.5):
        """
        Detect strong bearish candle:
        - Close < Open (red candle)
        - Body > threshold * total range
        - Close below previous candle's low
        """
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
        """
        Calculate comprehensive BEARISH reversal score based on all conditions.

        Returns:
        - score: -6 to 0 (more negative = stronger bearish reversal)
        - signals: dict with individual signal details
        - verdict: Overall recommendation
        """
        signals = {}
        score = 0

        # 1. Check buying pressure exhaustion (no new high)
        no_new_high, swing_high = ReversalDetector.detect_no_new_high(df, lookback)
        signals['Buying_Exhausted'] = "Yes ✅" if no_new_high else "No ❌"
        if no_new_high:
            score -= 1

        # 2. Check lower high formation
        lower_high, prev_high, curr_high = ReversalDetector.detect_lower_high(df, lookback // 2)
        signals['Lower_High'] = "Yes ✅" if lower_high else "No ❌"
        if lower_high:
            score -= 1.5

        # 3. Check strong bearish candle
        strong_candle, candle_details = ReversalDetector.detect_strong_bearish_candle(df)
        signals['Strong_Bearish_Candle'] = "Yes ✅" if strong_candle else "No ❌"
        if strong_candle:
            score -= 1.5

        # 4. Check volume confirmation (selling volume)
        vol_confirmed, vol_signal, vol_details = ReversalDetector.detect_volume_confirmation(df)
        # For bearish, we want down candle with volume
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

        # 5. Check VWAP position (below VWAP = bearish)
        above_vwap, price, vwap = ReversalDetector.check_vwap_position(df)
        signals['Below_VWAP'] = "Yes ✅" if not above_vwap else "No ❌"
        if not above_vwap:
            score -= 1

        # 6. Check resistance respect (if pivot highs provided)
        if pivot_highs:
            # Similar logic to support but for resistance
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

        # Determine verdict
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

        # Add price context
        if len(df) > 0:
            signals['Current_Price'] = df.iloc[-1]['close']
            signals['Day_High'] = df['high'].max()
            if vwap:
                signals['VWAP'] = round(vwap, 2)

        return score, signals, verdict


def calculate_max_pain(df_options, spot_price):
    """
    Calculate Max Pain - the strike price where option buyers lose the most money.

    Max Pain = Strike where total value of options expiring worthless is maximum

    For each strike:
    - CE pain = Sum of (strike - K) * OI for all strikes K < strike
    - PE pain = Sum of (K - strike) * OI for all strikes K > strike
    - Total pain = CE pain + PE pain

    Returns: max_pain_strike, pain_data
    """
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

            # CE buyers lose if spot < strike
            if strike < k:
                ce_pain += (k - strike) * ce_oi

            # PE buyers lose if spot > strike
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

    # Max pain is where total pain is MAXIMUM (option buyers lose most)
    max_pain_idx = pain_df['Total_Pain'].idxmax()
    max_pain_strike = pain_df.loc[max_pain_idx, 'Strike']

    return max_pain_strike, pain_df


def check_confluence_entry_signal(df, pivot_settings, df_summary, current_price, pivot_proximity,
                                   poc_data=None, rsi_sz_data=None, gex_data=None, ultimate_rsi_data=None):
    """
    Unified Confluence Entry Signal — sends ONE Telegram alert only when ALL conditions align:

    1. ATM Bias: Verdict is Strong Bullish or Strong Bearish (BiasScore >= 6 or <= -6)
    2. PCR + GEX: Confluence strength >= 2
    3. POC Alignment: Price position consistent with direction (above for bull, below for bear)
    4. RSI Suppression Zone: Recent breakout in same direction (or active zone = pending entry)
    5. Near Pivot Level: Price within proximity of HTF pivot S/R
    6. Ultimate RSI: Momentum and zone aligned with direction
    """
    if df.empty or df_summary is None or len(df_summary) == 0 or not current_price:
        return

    # Dedup: avoid sending same alert twice
    if 'last_confluence_alert' not in st.session_state:
        st.session_state.last_confluence_alert = None

    try:
        # --- 1. ATM Bias Verdict ---
        atm_data = df_summary[df_summary['Zone'] == 'ATM']
        if atm_data.empty:
            return
        row = atm_data.iloc[0]
        verdict = row.get('Verdict', 'Neutral')
        bias_score = row.get('BiasScore', 0)
        atm_strike = row.get('Strike', 0)

        if verdict == 'Strong Bullish':
            direction = 'bullish'
        elif verdict == 'Strong Bearish':
            direction = 'bearish'
        else:
            return  # No strong verdict → no alert

        # --- 2. PCR + GEX Confluence ---
        atm_pcr = row.get('PCR', 1.0)
        confluence_badge, confluence_signal, confluence_strength = calculate_pcr_gex_confluence(atm_pcr, gex_data)
        if confluence_strength < 2:
            return  # Weak confluence → no alert

        # Check confluence direction matches verdict
        if direction == 'bullish' and 'BULL' not in confluence_badge:
            return
        if direction == 'bearish' and 'BEAR' not in confluence_badge:
            return

        # --- 3. POC Alignment (spot above POC = bull, below POC = bear) ---
        poc_aligned = False
        poc_detail = "N/A"
        if poc_data:
            above_count = 0
            below_count = 0
            total = 0
            for poc_key in ['poc1', 'poc2', 'poc3']:
                poc = poc_data.get(poc_key)
                if poc and poc.get('poc'):
                    total += 1
                    if current_price > poc['poc']:
                        above_count += 1
                    else:
                        below_count += 1

            if direction == 'bullish' and above_count >= 2:
                poc_aligned = True
                poc_detail = f"Above {above_count}/{total} POCs (Bull)"
            elif direction == 'bearish' and below_count >= 2:
                poc_aligned = True
                poc_detail = f"Below {below_count}/{total} POCs (Bear)"
        else:
            poc_aligned = True  # Skip if no POC data available
            poc_detail = "POC data N/A"

        if not poc_aligned:
            return

        # --- 4. RSI Suppression Zone ---
        rsi_sz_signal = "N/A"
        rsi_sz_aligned = False
        if rsi_sz_data and rsi_sz_data.get('zones'):
            last_zone = rsi_sz_data['zones'][-1]
            breakout = last_zone.get('breakout', 'pending')
            if direction == 'bullish' and breakout == 'bullish':
                rsi_sz_aligned = True
                rsi_sz_signal = "Bullish Breakout"
            elif direction == 'bearish' and breakout == 'bearish':
                rsi_sz_aligned = True
                rsi_sz_signal = "Bearish Breakout"
            elif breakout == 'pending':
                rsi_sz_aligned = True  # Active zone = compression, accept it
                rsi_sz_signal = "In Suppression (pending breakout)"
        else:
            rsi_sz_aligned = True  # Skip if no data
            rsi_sz_signal = "RSI SZ data N/A"

        if not rsi_sz_aligned:
            return

        # --- 5. Near Pivot Level ---
        try:
            df_json = df.to_json()
            pivots = cached_pivot_calculation(df_json, pivot_settings)
        except Exception:
            pivots = PivotIndicator.get_all_pivots(df, pivot_settings)

        near_pivot = False
        pivot_level = None
        for pivot in pivots:
            if pivot['timeframe'] in ['3M', '5M', '10M', '15M']:
                if abs(current_price - pivot['value']) <= pivot_proximity:
                    near_pivot = True
                    pivot_level = pivot
                    break

        if not near_pivot:
            return

        # --- 6. Ultimate RSI ---
        ursi_detail = "N/A"
        ursi_aligned = False
        if ultimate_rsi_data:
            ursi_momentum = ultimate_rsi_data.get('momentum', 'Neutral')
            ursi_zone = ultimate_rsi_data.get('zone', 'Neutral')
            ursi_val = ultimate_rsi_data.get('latest_arsi', 50)
            ursi_sig_val = ultimate_rsi_data.get('latest_signal', 50)
            ursi_cross = ultimate_rsi_data.get('cross_signal', 'None')

            if direction == 'bullish' and ursi_momentum == 'Bullish':
                ursi_aligned = True
                ursi_detail = f"Bullish ({ursi_val:.0f} > Sig {ursi_sig_val:.0f})"
            elif direction == 'bearish' and ursi_momentum == 'Bearish':
                ursi_aligned = True
                ursi_detail = f"Bearish ({ursi_val:.0f} < Sig {ursi_sig_val:.0f})"
            elif ursi_cross == 'Bullish Cross' and direction == 'bullish':
                ursi_aligned = True
                ursi_detail = f"Bullish Cross ({ursi_val:.0f})"
            elif ursi_cross == 'Bearish Cross' and direction == 'bearish':
                ursi_aligned = True
                ursi_detail = f"Bearish Cross ({ursi_val:.0f})"
        else:
            ursi_aligned = True  # Skip if no data
            ursi_detail = "URSI data N/A"

        if not ursi_aligned:
            return

        # ===== ALL CONDITIONS MET — BUILD AND SEND ALERT =====
        ist = pytz.timezone('Asia/Kolkata')
        now_str = datetime.now(ist).strftime('%H:%M:%S IST')
        alert_key = f"confluence_{direction}_{atm_strike}_{datetime.now(ist).strftime('%Y%m%d_%H%M')}"

        if st.session_state.last_confluence_alert == alert_key:
            return  # Already sent this minute

        # Gather all details
        price_diff = current_price - pivot_level['value']
        oi_bias = row.get('OI_Bias', 'N/A')
        chgoi_bias = row.get('ChgOI_Bias', 'N/A')
        volume_bias = row.get('Volume_Bias', 'N/A')
        delta_exp = row.get('DeltaExp', 'N/A')
        gamma_exp = row.get('GammaExp', 'N/A')
        pressure_bias = row.get('PressureBias', 'N/A')
        operator_entry = row.get('Operator_Entry', 'N/A')
        ce_chg_oi = row.get('changeinOpenInterest_CE', 0)
        pe_chg_oi = row.get('changeinOpenInterest_PE', 0)

        net_gex = gex_data.get('total_gex', 0) if gex_data else 0
        gex_signal_text = gex_data.get('gex_signal', 'N/A') if gex_data else 'N/A'
        gex_magnet = gex_data.get('gex_magnet', 'N/A') if gex_data else 'N/A'

        if direction == 'bullish':
            emoji = "🟢🔥"
            dir_label = "BULLISH"
            option_type = "CE"
        else:
            emoji = "🔴🔥"
            dir_label = "BEARISH"
            option_type = "PE"

        message = f"""
{emoji} <b>CONFLUENCE ENTRY ALERT — {dir_label}</b> {emoji}

📍 <b>Spot:</b> ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} Pivot by {price_diff:+.1f} pts)
📌 <b>Pivot:</b> {pivot_level['timeframe']} at ₹{pivot_level['value']:.2f}
🎯 <b>ATM Strike:</b> {atm_strike} {option_type}

<b>✅ ALL 6 CONDITIONS MET:</b>
1️⃣ ATM Verdict: {verdict} (Score: {bias_score})
2️⃣ PCR×GEX: {confluence_badge} ({confluence_signal}) ★{confluence_strength}
3️⃣ POC: {poc_detail}
4️⃣ RSI SZ: {rsi_sz_signal}
5️⃣ Pivot: {pivot_level['timeframe']} within {pivot_proximity} pts
6️⃣ URSI: {ursi_detail}

<b>📊 ATM BIAS:</b>
• OI: {oi_bias} | ChgOI: {chgoi_bias} | Vol: {volume_bias}
• Delta: {delta_exp} | Gamma: {gamma_exp} | Pressure: {pressure_bias}
• Operator: {operator_entry}

<b>📈 OI DATA:</b>
• CE ΔOI: {ce_chg_oi/1000:.1f}K | PE ΔOI: {pe_chg_oi/1000:.1f}K | PCR: {atm_pcr:.2f}

<b>🎯 GEX:</b>
• Net: {net_gex:.2f}L | Regime: {gex_signal_text} | Magnet: {gex_magnet}

🕐 {now_str}
"""
        send_telegram_message_sync(message)
        st.session_state.last_confluence_alert = alert_key
        if direction == 'bullish':
            st.success(f"🟢🔥 Confluence BULLISH entry alert sent! Strike {atm_strike} CE")
        else:
            st.success(f"🔴🔥 Confluence BEARISH entry alert sent! Strike {atm_strike} PE")

    except Exception as e:
        pass  # Silently fail to avoid disrupting the app


def check_vwap_poc_alignment_signal(df, current_price, vwap_data=None, poc_data=None, poc_proximity=3.0,
                                     option_data=None, gex_data=None, require_gex_move=False):
    """
    VWAP + POC Alignment Signal — sends Telegram alert when:

    CALL Signal: Spot ABOVE VWAP AND POC 1,2,3 all aligned BELOW spot (within proximity of each other)
    PUT Signal:  Spot BELOW VWAP AND POC 1,2,3 all aligned ABOVE spot (within proximity of each other)

    POC alignment = all 3 POC values are within `poc_proximity` points of each other.

    Enhanced with:
    - ATM strike + Bull/Bear verdict from option chain bias engine
    - GEX regime (Pin/Chop, Range, Trending, Breakout) with big-move detection
    - Optional: require GEX Trending/Breakout for signal (require_gex_move=True)
    """
    if df.empty or not current_price:
        return
    if not vwap_data or vwap_data.get('latest_vwap') is None:
        return
    if not poc_data:
        return

    # Dedup: avoid sending same alert twice
    if 'last_vwap_poc_alert' not in st.session_state:
        st.session_state.last_vwap_poc_alert = None

    try:
        vwap_val = vwap_data['latest_vwap']

        # Collect POC values
        poc_values = []
        poc_labels = []
        for poc_key in ['poc1', 'poc2', 'poc3']:
            poc = poc_data.get(poc_key)
            if poc and poc.get('poc') is not None:
                poc_values.append(poc['poc'])
                period = poc_data.get('periods', {}).get(poc_key, poc_key)
                poc_labels.append(f"POC{poc_key[-1]}({period})")

        # Need all 3 POCs
        if len(poc_values) < 3:
            return

        # Check POC alignment: all 3 must be within poc_proximity of each other
        poc_max = max(poc_values)
        poc_min = min(poc_values)
        poc_spread = poc_max - poc_min

        if poc_spread > poc_proximity:
            return  # POCs not aligned closely enough

        poc_avg = sum(poc_values) / len(poc_values)

        # Determine signal direction
        spot_above_vwap = current_price > vwap_val
        spot_below_vwap = current_price < vwap_val
        all_poc_below_spot = all(p < current_price for p in poc_values)
        all_poc_above_spot = all(p > current_price for p in poc_values)

        direction = None
        if spot_above_vwap and all_poc_below_spot:
            direction = 'bullish'
        elif spot_below_vwap and all_poc_above_spot:
            direction = 'bearish'
        else:
            return  # Conditions not met

        # --- ATM Strike + Bull/Bear Verdict ---
        atm_strike = None
        atm_verdict = "N/A"
        atm_bias_score = 0
        atm_oi_bias = "N/A"
        atm_chgoi_bias = "N/A"
        atm_operator = "N/A"
        atm_option_type = "CE" if direction == 'bullish' else "PE"

        if option_data and option_data.get('df_summary') is not None:
            df_summary = option_data['df_summary']
            atm_rows = df_summary[df_summary['Zone'] == 'ATM']
            if not atm_rows.empty:
                atm_row = atm_rows.iloc[0]
                atm_strike = atm_row.get('Strike', None)
                atm_verdict = atm_row.get('Verdict', 'N/A')
                atm_bias_score = atm_row.get('BiasScore', 0)
                atm_oi_bias = atm_row.get('OI_Bias', 'N/A')
                atm_chgoi_bias = atm_row.get('ChgOI_Bias', 'N/A')
                atm_operator = atm_row.get('Operator_Entry', 'N/A')

        # --- GEX Regime ---
        gex_signal = "N/A"
        gex_total = 0
        gex_magnet = "N/A"
        gex_repeller = "N/A"
        gex_flip = "N/A"
        gex_big_move = False  # True when GEX says Trending or Breakout

        if gex_data:
            gex_signal = gex_data.get('gex_signal', 'N/A')
            gex_total = gex_data.get('total_gex', 0)
            gex_magnet = gex_data.get('gex_magnet', 'N/A')
            gex_repeller = gex_data.get('gex_repeller', 'N/A')
            gex_flip_level = gex_data.get('gamma_flip_level', None)
            gex_flip = f"₹{gex_flip_level:,.0f}" if gex_flip_level else "N/A"
            gex_big_move = gex_signal in ('Trending', 'Breakout')

        # If require_gex_move is enabled, skip signal when GEX is pinning/ranging
        if require_gex_move and not gex_big_move:
            return  # GEX says chop/pin, not a big-move environment

        # Dedup check (per-minute, per-direction)
        ist = pytz.timezone('Asia/Kolkata')
        now_str = datetime.now(ist).strftime('%H:%M:%S IST')
        alert_key = f"vwap_poc_{direction}_{datetime.now(ist).strftime('%Y%m%d_%H%M')}"

        if st.session_state.last_vwap_poc_alert == alert_key:
            return  # Already sent this minute

        # Build message
        vwap_diff = current_price - vwap_val
        poc_diff = current_price - poc_avg

        if direction == 'bullish':
            emoji = "🟢📊"
            dir_label = "CALL"
            action = "BUY CE"
            bias_text = "BULLISH"
        else:
            emoji = "🔴📊"
            dir_label = "PUT"
            action = "BUY PE"
            bias_text = "BEARISH"

        poc_details = "\n".join([
            f"  • {label}: ₹{val:,.2f} ({current_price - val:+.1f} pts from spot)"
            for label, val in zip(poc_labels, poc_values)
        ])

        # Band info if available
        band_info = ""
        if vwap_data.get('show_band_1') and vwap_data.get('latest_upper_1') is not None:
            band_info += f"\n  • Band 1: ₹{vwap_data['latest_lower_1']:,.2f} — ₹{vwap_data['latest_upper_1']:,.2f}"
        if vwap_data.get('show_band_2') and vwap_data.get('latest_upper_2') is not None:
            band_info += f"\n  • Band 2: ₹{vwap_data['latest_lower_2']:,.2f} — ₹{vwap_data['latest_upper_2']:,.2f}"
        if vwap_data.get('show_band_3') and vwap_data.get('latest_upper_3') is not None:
            band_info += f"\n  • Band 3: ₹{vwap_data['latest_lower_3']:,.2f} — ₹{vwap_data['latest_upper_3']:,.2f}"

        band_position = vwap_data.get('band_position', 'N/A')

        # ATM section
        atm_section = ""
        if atm_strike:
            verdict_emoji = "🟢" if "Bullish" in atm_verdict else "🔴" if "Bearish" in atm_verdict else "⚪"
            atm_section = f"""
<b>🎯 ATM STRIKE:</b> {atm_strike} {atm_option_type}
  • Verdict: {verdict_emoji} {atm_verdict} (Score: {atm_bias_score:+.1f})
  • OI: {atm_oi_bias} | ChgOI: {atm_chgoi_bias} | Operator: {atm_operator}"""

        # GEX section
        gex_section = ""
        if gex_data:
            gex_emoji = "⚡" if gex_big_move else "📍"
            move_label = "BIG MOVE EXPECTED" if gex_big_move else "RANGE/PIN MODE"
            gex_section = f"""
<b>{gex_emoji} GEX REGIME:</b> {gex_signal} — {move_label}
  • Net GEX: {gex_total:+.2f}L | Flip: {gex_flip}
  • Magnet: {gex_magnet if gex_magnet != 'N/A' and gex_magnet else 'N/A'} | Repeller: {gex_repeller if gex_repeller != 'N/A' and gex_repeller else 'N/A'}"""

        # Condition count
        cond_num = 3
        conditions_text = f"""1️⃣ Spot {'ABOVE' if spot_above_vwap else 'BELOW'} VWAP ({'Bullish' if spot_above_vwap else 'Bearish'} Bias)
2️⃣ All 3 POCs aligned within {poc_spread:.1f} pts (< {poc_proximity:.0f} pt threshold)
3️⃣ All POCs {'BELOW' if all_poc_below_spot else 'ABOVE'} spot price ({bias_text} confirmation)"""

        if atm_strike and atm_verdict != 'N/A':
            verdict_match = ('Bullish' in atm_verdict and direction == 'bullish') or ('Bearish' in atm_verdict and direction == 'bearish')
            cond_num += 1
            conditions_text += f"\n{cond_num}\u20e3 ATM {atm_verdict} {'✅ CONFIRMS' if verdict_match else '⚠️ DIVERGENT'}"

        if gex_data:
            cond_num += 1
            conditions_text += f"\n{cond_num}\u20e3 GEX: {gex_signal} ({'⚡ BIG MOVE' if gex_big_move else '📍 Range/Pin'})"

        message = f"""{emoji} <b>VWAP + POC ALIGNMENT — {dir_label} SIGNAL</b> {emoji}

📍 <b>Spot:</b> ₹{current_price:,.2f}
📈 <b>VWAP:</b> ₹{vwap_val:,.2f} (Spot {'ABOVE' if vwap_diff > 0 else 'BELOW'} by {abs(vwap_diff):.1f} pts)
📊 <b>Band Position:</b> {band_position}

<b>🎯 POC Alignment (spread: {poc_spread:.1f} pts):</b>
{poc_details}
  • POC Average: ₹{poc_avg:,.2f} (All 3 POCs {'BELOW' if all_poc_below_spot else 'ABOVE'} spot)
{atm_section}
{gex_section}

<b>✅ CONDITIONS MET:</b>
{conditions_text}{f'''

<b>📏 VWAP Bands:</b>{band_info}''' if band_info else ''}

<b>💡 Action:</b> {action}
🕐 {now_str}
"""
        send_telegram_message_sync(message)
        st.session_state.last_vwap_poc_alert = alert_key

        if direction == 'bullish':
            st.success(f"🟢📊 VWAP+POC CALL signal sent! Spot above VWAP, all 3 POCs aligned below (spread: {poc_spread:.1f} pts)" +
                       (f" | ATM {atm_verdict}" if atm_strike else "") +
                       (f" | GEX: {gex_signal}" if gex_data else ""))
        else:
            st.success(f"🔴📊 VWAP+POC PUT signal sent! Spot below VWAP, all 3 POCs aligned above (spread: {poc_spread:.1f} pts)" +
                       (f" | ATM {atm_verdict}" if atm_strike else "") +
                       (f" | GEX: {gex_signal}" if gex_data else ""))

    except Exception:
        pass  # Silently fail to avoid disrupting the app


def calculate_dealer_gex(df_summary, spot_price, contract_multiplier=25):
    """
    Calculate Net Gamma Exposure (GEX) from dealer's perspective.

    Dealers are SHORT options (they sell to retail) so their gamma exposure is INVERTED.
    - Dealer SHORT Call = NEGATIVE gamma exposure (dealers sell into rallies)
    - Dealer SHORT Put = POSITIVE gamma exposure (dealers buy into selloffs)

    Net GEX = (CE_Gamma × CE_OI × -1) + (PE_Gamma × PE_OI × 1)
            = -GammaExp_CE + GammaExp_PE

    Positive Net GEX = Dealers long gamma = Price tends to PIN/REVERT (chop day)
    Negative Net GEX = Dealers short gamma = Price tends to ACCELERATE (trend day)

    Returns: dict with gex_data, gamma_flip_level, total_gex, gex_interpretation
    """
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

            # Dealer's perspective: Short options
            # When dealer is short call, price rise = dealer sells (negative gamma effect)
            # When dealer is short put, price drop = dealer buys (positive gamma effect)
            call_gex = -1 * gamma_ce * oi_ce * contract_multiplier * spot_price / 100000  # in Lakhs
            put_gex = gamma_pe * oi_pe * contract_multiplier * spot_price / 100000  # in Lakhs
            net_gex = call_gex + put_gex

            gex_data.append({
                'Strike': strike,
                'Call_GEX': round(call_gex, 2),
                'Put_GEX': round(put_gex, 2),
                'Net_GEX': round(net_gex, 2),
                'Zone': row.get('Zone', '-')
            })

        gex_df = pd.DataFrame(gex_data)

        # Calculate total Net GEX
        total_gex = gex_df['Net_GEX'].sum()

        # Find Gamma Flip Level (where Net GEX crosses zero)
        # Sort by strike price
        gex_df_sorted = gex_df.sort_values('Strike')
        gamma_flip_level = None
        gamma_flip_direction = None

        for i in range(len(gex_df_sorted) - 1):
            current_gex = gex_df_sorted.iloc[i]['Net_GEX']
            next_gex = gex_df_sorted.iloc[i + 1]['Net_GEX']
            current_strike = gex_df_sorted.iloc[i]['Strike']
            next_strike = gex_df_sorted.iloc[i + 1]['Strike']

            # Check for sign change
            if current_gex * next_gex < 0:
                # Linear interpolation to find exact flip level
                gamma_flip_level = current_strike + (next_strike - current_strike) * abs(current_gex) / (abs(current_gex) + abs(next_gex))
                gamma_flip_direction = "Positive above" if current_gex < 0 else "Negative above"
                break

        # GEX Interpretation
        if total_gex > 50:
            gex_interpretation = "STRONG PIN - Dealers long gamma, price likely to revert/chop"
            gex_signal = "Pin/Chop"
            gex_color = "#00ff88"  # Green
        elif total_gex > 0:
            gex_interpretation = "MILD PIN - Slight mean reversion tendency"
            gex_signal = "Range"
            gex_color = "#90EE90"  # Light green
        elif total_gex > -50:
            gex_interpretation = "MILD TREND - Slight directional bias possible"
            gex_signal = "Trending"
            gex_color = "#FFD700"  # Yellow
        else:
            gex_interpretation = "STRONG TREND - Dealers short gamma, violent moves possible"
            gex_signal = "Breakout"
            gex_color = "#ff4444"  # Red

        # Find max positive and negative GEX strikes (magnets and repellers)
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



def calculate_pcr_gex_confluence(pcr_value, gex_data, zone='ATM'):
    """
    Calculate PCR × GEX Confluence Badge.

    Combines positioning bias (PCR) with dealer hedging pressure (GEX) for stronger signals.

    Confluence Matrix:
    - Bullish PCR (>1.2) + Negative GEX = STRONG BULLISH (upside acceleration)
    - Bearish PCR (<0.7) + Positive GEX = STRONG BEARISH (downside acceleration)
    - Bullish PCR + Positive GEX = BULLISH RANGE (support with chop)
    - Bearish PCR + Negative GEX = BEARISH RANGE (resistance with chop)
    - Neutral = Mixed signals

    Returns: confluence_badge, confluence_signal, confluence_strength
    """
    if gex_data is None:
        return "⚪ N/A", "No GEX Data", 0

    net_gex = gex_data.get('total_gex', 0)
    gex_signal = gex_data.get('gex_signal', 'Unknown')

    # PCR interpretation
    if pcr_value > 1.2:
        pcr_signal = "Bullish"
    elif pcr_value < 0.7:
        pcr_signal = "Bearish"
    else:
        pcr_signal = "Neutral"

    # GEX interpretation for confluence
    gex_negative = net_gex < -10  # Strong negative gamma
    gex_positive = net_gex > 10   # Strong positive gamma

    # Confluence logic
    if pcr_signal == "Bullish" and gex_negative:
        # Best bullish setup: Put heavy + dealers short gamma = violent upside
        return "🟢🔥 STRONG BULL", "Bullish + Breakout", 3

    elif pcr_signal == "Bearish" and gex_positive:
        # Best bearish setup: Call heavy + dealers long gamma = strong pin/rejection
        return "🔴🔥 STRONG BEAR", "Bearish + Pin", 3

    elif pcr_signal == "Bullish" and gex_positive:
        # Bullish bias but pinning action
        return "🟢📍 BULL RANGE", "Bullish + Chop", 2

    elif pcr_signal == "Bearish" and gex_negative:
        # Bearish bias with acceleration risk
        return "🔴⚡ BEAR TREND", "Bearish + Accel", 2

    elif pcr_signal == "Bullish":
        return "🟢 BULLISH", "Bullish PCR", 1

    elif pcr_signal == "Bearish":
        return "🔴 BEARISH", "Bearish PCR", 1

    else:
        return "⚪ NEUTRAL", "Mixed Signals", 0


def calculate_exact_time_to_expiry(expiry_date_str):
    """Calculate exact time to expiry in years (days + hours)"""
    try:
        expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").replace(hour=15, minute=30)
        expiry_date = expiry_date.replace(tzinfo=pytz.timezone('Asia/Kolkata'))
        
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        time_diff = expiry_date - now
        
        # Convert to years (more precise calculation)
        total_seconds = time_diff.total_seconds()
        total_days = total_seconds / (24 * 3600)
        years = total_days / 365.25
        
        return max(years, 1/365.25)  # Minimum 1 day
    except:
        return 1/365.25

def get_iv_fallback(df, strike_price):
    """Get IV fallback using nearest strike average instead of fixed value"""
    try:
        # Find strikes within ±100 points of current strike
        nearby_strikes = df[abs(df['strikePrice'] - strike_price) <= 100]
        
        if not nearby_strikes.empty:
            iv_ce_avg = nearby_strikes['impliedVolatility_CE'].mean()
            iv_pe_avg = nearby_strikes['impliedVolatility_PE'].mean()
            
            # Fill NaN with overall average
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
    if score >= 6:
        return "Strong Bullish"
    elif score >= 3:
        return "Bullish"
    elif score <= -6:
        return "Strong Bearish"
    elif score <= -3:
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

def color_pressure(val):
    if val > 500:
        return 'background-color: #90EE90; color: black'
    elif val < -500:
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

def color_pcr(val):
    if val > 1.2:
        return 'background-color: #90EE90; color: black'
    elif val < 0.7:
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

def color_bias(val):
    """Color bias cells: Green for Bullish, Red for Bearish, Yellow for Neutral"""
    if val == "Bullish":
        return 'background-color: #90EE90; color: black'
    elif val == "Bearish":
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

def color_verdict(val):
    """Color verdict cells based on strength"""
    if "Strong Bullish" in str(val):
        return 'background-color: #228B22; color: white'
    elif "Bullish" in str(val):
        return 'background-color: #90EE90; color: black'
    elif "Strong Bearish" in str(val):
        return 'background-color: #DC143C; color: white'
    elif "Bearish" in str(val):
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

def color_entry(val):
    """Color operator entry signals"""
    if "Bull" in str(val):
        return 'background-color: #90EE90; color: black'
    elif "Bear" in str(val):
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #F5F5F5; color: black'

def color_fakereal(val):
    """Color fake/real move signals"""
    if "Real Up" in str(val):
        return 'background-color: #228B22; color: white'
    elif "Fake Up" in str(val):
        return 'background-color: #98FB98; color: black'
    elif "Real Down" in str(val):
        return 'background-color: #DC143C; color: white'
    elif "Fake Down" in str(val):
        return 'background-color: #FFC0CB; color: black'
    else:
        return 'background-color: #F5F5F5; color: black'

def color_score(val):
    """Color score based on value"""
    try:
        score = float(val)
        if score >= 6:
            return 'background-color: #228B22; color: white'
        elif score >= 3:
            return 'background-color: #90EE90; color: black'
        elif score <= -6:
            return 'background-color: #DC143C; color: white'
        elif score <= -3:
            return 'background-color: #FFB6C1; color: black'
        else:
            return 'background-color: #FFFFE0; color: black'
    except:
        return ''

def highlight_atm_row(row):
    """Highlight ATM row in the dataframe - disabled"""
    # ATM yellow highlight removed per user request
    return [''] * len(row)

def process_candle_data(data, interval):
    """Process API response into DataFrame"""
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

def create_candlestick_chart(df, title, interval, show_pivots=True, pivot_settings=None, vob_blocks=None, poc_data=None, swing_data=None, rsi_sz_data=None, ultimate_rsi_data=None, vwap_data=None):
    """Create TradingView-style candlestick chart with optional pivot levels, VOB zones, POC lines, Swing data, RSI Suppression Zones, Ultimate RSI, and VWAP with bands"""
    if df.empty:
        return go.Figure()

    has_ursi = ultimate_rsi_data is not None and ultimate_rsi_data.get('arsi') is not None

    fig = make_subplots(
        rows=3 if has_ursi else 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.2, 0.25] if has_ursi else [0.7, 0.3],
        subplot_titles=(None, None, "Ultimate RSI") if has_ursi else None
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
            # Use cached pivot calculation for performance
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
                
                for timestamp, high_value in data['highs'][-3:]:
                    fig.add_shape(
                        type="line",
                        x0=x_start, x1=x_end,
                        y0=high_value, y1=high_value,
                        line=dict(color=color, width=1, dash="dash"),
                        row=1, col=1
                    )
                    
                    fig.add_annotation(
                        x=x_end,
                        y=high_value,
                        text=f"{tf} R {high_value:.1f}",
                        showarrow=False,
                        font=dict(color=color, size=10),
                        xanchor="left",
                        row=1, col=1
                    )
                
                for timestamp, low_value in data['lows'][-3:]:
                    fig.add_shape(
                        type="line", 
                        x0=x_start, x1=x_end,
                        y0=low_value, y1=low_value,
                        line=dict(color=color, width=1, dash="dash"),
                        row=1, col=1
                    )
                    
                    fig.add_annotation(
                        x=x_end,
                        y=low_value,
                        text=f"{tf} S {low_value:.1f}",
                        showarrow=False,
                        font=dict(color=color, size=10),
                        xanchor="left",
                        row=1, col=1
                    )
        
        except Exception as e:
            st.warning(f"Error adding pivot levels: {str(e)}")

    # Add VWAP line with bands
    try:
        if vwap_data and vwap_data.get('vwap') is not None:
            vwap_series = vwap_data['vwap']
            valid_mask = vwap_series.notna()
            if valid_mask.any():
                dt = df.loc[valid_mask, 'datetime']
                vwap_vals = vwap_series[valid_mask]

                # Main VWAP line
                fig.add_trace(
                    go.Scatter(
                        x=dt, y=vwap_vals,
                        mode='lines',
                        name='VWAP',
                        line=dict(color='#2962FF', width=2),
                        opacity=0.9,
                        hovertemplate='VWAP: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )

                # Band 1 (green)
                if vwap_data.get('show_band_1') and vwap_data.get('upper_band_1') is not None:
                    ub1 = vwap_data['upper_band_1'][valid_mask]
                    lb1 = vwap_data['lower_band_1'][valid_mask]
                    fig.add_trace(
                        go.Scatter(
                            x=dt, y=ub1, mode='lines',
                            name=f"Upper Band #1 ({vwap_data['band_mult_1']})",
                            line=dict(color='#00C853', width=1),
                            opacity=0.7,
                            hovertemplate='UB1: %{y:.2f}<extra></extra>'
                        ), row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=dt, y=lb1, mode='lines',
                            name=f"Lower Band #1 ({vwap_data['band_mult_1']})",
                            line=dict(color='#00C853', width=1),
                            opacity=0.7,
                            fill='tonexty',
                            fillcolor='rgba(0, 200, 83, 0.05)',
                            hovertemplate='LB1: %{y:.2f}<extra></extra>'
                        ), row=1, col=1
                    )

                # Band 2 (olive)
                if vwap_data.get('show_band_2') and vwap_data.get('upper_band_2') is not None:
                    ub2 = vwap_data['upper_band_2'][valid_mask]
                    lb2 = vwap_data['lower_band_2'][valid_mask]
                    fig.add_trace(
                        go.Scatter(
                            x=dt, y=ub2, mode='lines',
                            name=f"Upper Band #2 ({vwap_data['band_mult_2']})",
                            line=dict(color='#808000', width=1),
                            opacity=0.7,
                            hovertemplate='UB2: %{y:.2f}<extra></extra>'
                        ), row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=dt, y=lb2, mode='lines',
                            name=f"Lower Band #2 ({vwap_data['band_mult_2']})",
                            line=dict(color='#808000', width=1),
                            opacity=0.7,
                            fill='tonexty',
                            fillcolor='rgba(128, 128, 0, 0.05)',
                            hovertemplate='LB2: %{y:.2f}<extra></extra>'
                        ), row=1, col=1
                    )

                # Band 3 (teal)
                if vwap_data.get('show_band_3') and vwap_data.get('upper_band_3') is not None:
                    ub3 = vwap_data['upper_band_3'][valid_mask]
                    lb3 = vwap_data['lower_band_3'][valid_mask]
                    fig.add_trace(
                        go.Scatter(
                            x=dt, y=ub3, mode='lines',
                            name=f"Upper Band #3 ({vwap_data['band_mult_3']})",
                            line=dict(color='#008080', width=1),
                            opacity=0.7,
                            hovertemplate='UB3: %{y:.2f}<extra></extra>'
                        ), row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=dt, y=lb3, mode='lines',
                            name=f"Lower Band #3 ({vwap_data['band_mult_3']})",
                            line=dict(color='#008080', width=1),
                            opacity=0.7,
                            fill='tonexty',
                            fillcolor='rgba(0, 128, 128, 0.05)',
                            hovertemplate='LB3: %{y:.2f}<extra></extra>'
                        ), row=1, col=1
                    )
        else:
            # Fallback: simple VWAP from ReversalDetector
            vwap = ReversalDetector.calculate_vwap(df)
            if not vwap.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df['datetime'], y=vwap,
                        mode='lines', name='VWAP',
                        line=dict(color='#2962FF', width=2),
                        opacity=0.8
                    ), row=1, col=1
                )
    except Exception as e:
        pass  # VWAP calculation failed, skip it

    # Add Volume Order Blocks (VOB) zones
    if vob_blocks:
        try:
            x_start = df['datetime'].min()
            x_end = df['datetime'].max()

            # Draw bullish VOB zones (support - green)
            for block in vob_blocks.get('bullish', [])[-5:]:  # Last 5 bullish zones
                fig.add_shape(
                    type="rect",
                    x0=x_start, x1=x_end,
                    y0=block['lower'], y1=block['upper'],
                    fillcolor="rgba(38, 186, 159, 0.15)",  # Teal with transparency
                    line=dict(color="#26ba9f", width=2),
                    row=1, col=1
                )
                # Add midline
                fig.add_shape(
                    type="line",
                    x0=x_start, x1=x_end,
                    y0=block['mid'], y1=block['mid'],
                    line=dict(color="#26ba9f", width=1, dash="dash"),
                    row=1, col=1
                )
                # Add volume annotation
                vol_text = VolumeOrderBlocks.format_volume(block['volume'])
                fig.add_annotation(
                    x=x_end,
                    y=block['mid'],
                    text=f"VOB↑ {vol_text} ({block['volume_pct']:.0f}%)",
                    showarrow=False,
                    font=dict(color="#26ba9f", size=9),
                    xanchor="left",
                    row=1, col=1
                )

            # Draw bearish VOB zones (resistance - purple)
            for block in vob_blocks.get('bearish', [])[-5:]:  # Last 5 bearish zones
                fig.add_shape(
                    type="rect",
                    x0=x_start, x1=x_end,
                    y0=block['lower'], y1=block['upper'],
                    fillcolor="rgba(102, 38, 186, 0.15)",  # Purple with transparency
                    line=dict(color="#6626ba", width=2),
                    row=1, col=1
                )
                # Add midline
                fig.add_shape(
                    type="line",
                    x0=x_start, x1=x_end,
                    y0=block['mid'], y1=block['mid'],
                    line=dict(color="#6626ba", width=1, dash="dash"),
                    row=1, col=1
                )
                # Add volume annotation
                vol_text = VolumeOrderBlocks.format_volume(block['volume'])
                fig.add_annotation(
                    x=x_end,
                    y=block['mid'],
                    text=f"VOB↓ {vol_text} ({block['volume_pct']:.0f}%)",
                    showarrow=False,
                    font=dict(color="#6626ba", size=9),
                    xanchor="left",
                    row=1, col=1
                )
        except Exception as e:
            pass  # VOB drawing failed, skip it

    # Add Triple POC steplines if provided
    if poc_data:
        try:
            poc_colors = {
                'poc1': '#e91e63',  # Pink - Short-term
                'poc2': '#2196f3',  # Blue - Medium-term
                'poc3': '#4caf50'   # Green - Long-term
            }

            for poc_key in ['poc1', 'poc2', 'poc3']:
                series_key = f'{poc_key}_series'
                series_data = poc_data.get(series_key)
                poc_latest = poc_data.get(poc_key)

                if series_data is not None:
                    color = poc_colors[poc_key]
                    period = poc_data.get('periods', {}).get(poc_key, '')

                    poc_s = series_data['poc']
                    upper_s = series_data['upper_poc']
                    lower_s = series_data['lower_poc']

                    # Filter out NaN values and align with datetime
                    valid_mask = poc_s.notna()
                    if valid_mask.any():
                        dt = df.loc[valid_mask, 'datetime']
                        poc_vals = poc_s[valid_mask]
                        upper_vals = upper_s[valid_mask]
                        lower_vals = lower_s[valid_mask]

                        # Main POC stepline
                        fig.add_trace(
                            go.Scatter(
                                x=dt, y=poc_vals,
                                mode='lines',
                                name=f'POC {poc_key[-1]} ({period})',
                                line=dict(color=color, width=2, shape='hv'),
                                showlegend=True,
                                hovertemplate=f'POC{poc_key[-1]}: ₹%{{y:.2f}}<extra></extra>'
                            ),
                            row=1, col=1
                        )

                        # Upper POC channel (translucent)
                        fig.add_trace(
                            go.Scatter(
                                x=dt, y=upper_vals,
                                mode='lines',
                                name=f'Upper POC {poc_key[-1]}',
                                line=dict(color=color, width=1, shape='hv', dash='dot'),
                                opacity=0.3,
                                showlegend=False,
                                hoverinfo='skip'
                            ),
                            row=1, col=1
                        )

                        # Lower POC channel (translucent) with fill to upper
                        # Convert hex to rgba for fill
                        hex_c = color.lstrip('#')
                        r, g, b = int(hex_c[:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
                        fill_rgba = f'rgba({r},{g},{b},0.08)'

                        fig.add_trace(
                            go.Scatter(
                                x=dt, y=lower_vals,
                                mode='lines',
                                name=f'Lower POC {poc_key[-1]}',
                                line=dict(color=color, width=1, shape='hv', dash='dot'),
                                opacity=0.3,
                                showlegend=False,
                                fill='tonexty',
                                fillcolor=fill_rgba,
                                hoverinfo='skip'
                            ),
                            row=1, col=1
                        )

                        # Label at the end
                        last_poc_val = poc_vals.iloc[-1]
                        last_dt = dt.iloc[-1]
                        fig.add_annotation(
                            x=last_dt, y=last_poc_val,
                            text=f"POC{poc_key[-1]} ({period}): ₹{last_poc_val:.0f}",
                            showarrow=False,
                            font=dict(color=color, size=10),
                            xanchor="left",
                            row=1, col=1
                        )

        except Exception as e:
            pass  # POC drawing failed, skip it

    # Add Future Swing data if provided
    if swing_data and swing_data.get('swings'):
        try:
            swings = swing_data['swings']
            projection = swing_data.get('projection')

            x_end = df['datetime'].max()

            # Draw swing high zone
            last_high = swings.get('last_swing_high')
            if last_high and last_high.get('value'):
                atr_estimate = (df['high'].max() - df['low'].min()) * 0.02  # Rough ATR estimate

                fig.add_shape(
                    type="rect",
                    x0=df['datetime'].iloc[last_high['index']] if last_high['index'] < len(df) else x_end,
                    x1=x_end,
                    y0=last_high['value'],
                    y1=last_high['value'] + atr_estimate,
                    fillcolor="rgba(235, 117, 20, 0.2)",
                    line=dict(color="#eb7514", width=1),
                    row=1, col=1
                )

                fig.add_annotation(
                    x=x_end,
                    y=last_high['value'],
                    text=f"Swing H: ₹{last_high['value']:.0f}",
                    showarrow=False,
                    font=dict(color="#eb7514", size=9),
                    xanchor="left",
                    row=1, col=1
                )

            # Draw swing low zone
            last_low = swings.get('last_swing_low')
            if last_low and last_low.get('value'):
                atr_estimate = (df['high'].max() - df['low'].min()) * 0.02

                fig.add_shape(
                    type="rect",
                    x0=df['datetime'].iloc[last_low['index']] if last_low['index'] < len(df) else x_end,
                    x1=x_end,
                    y0=last_low['value'] - atr_estimate,
                    y1=last_low['value'],
                    fillcolor="rgba(21, 221, 124, 0.2)",
                    line=dict(color="#15dd7c", width=1),
                    row=1, col=1
                )

                fig.add_annotation(
                    x=x_end,
                    y=last_low['value'],
                    text=f"Swing L: ₹{last_low['value']:.0f}",
                    showarrow=False,
                    font=dict(color="#15dd7c", size=9),
                    xanchor="left",
                    row=1, col=1
                )

            # Draw future projection line
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
            pass  # Swing drawing failed, skip it

    # ===== RSI VOLATILITY SUPPRESSION ZONES =====
    if rsi_sz_data and rsi_sz_data.get('zones'):
        try:
            for zone in rsi_sz_data['zones']:
                if zone.get('start_time') is None or zone.get('end_time') is None:
                    continue

                x_start = zone['start_time']
                x_end = zone['end_time']
                y_top = zone['top']
                y_bottom = zone['bottom']
                breakout = zone.get('breakout', 'pending')

                # Colors based on breakout direction
                if breakout == 'bullish':
                    fill_color = 'rgba(0, 187, 212, 0.15)'
                    border_color = 'rgba(0, 187, 212, 0.4)'
                    symbol_text = "▲"
                elif breakout == 'bearish':
                    fill_color = 'rgba(155, 39, 176, 0.15)'
                    border_color = 'rgba(155, 39, 176, 0.4)'
                    symbol_text = "▼"
                else:
                    fill_color = 'rgba(128, 128, 128, 0.1)'
                    border_color = 'rgba(128, 128, 128, 0.3)'
                    symbol_text = "∿"

                # Draw zone rectangle using shapes
                fig.add_shape(
                    type="rect",
                    x0=x_start, x1=x_end,
                    y0=y_bottom, y1=y_top,
                    fillcolor=fill_color,
                    line=dict(color=border_color, width=1),
                    row=1, col=1
                )

                # Add label
                fig.add_annotation(
                    x=x_end,
                    y=y_top,
                    text=f"SZ {symbol_text}",
                    showarrow=False,
                    font=dict(color=border_color.replace('0.4', '1').replace('0.3', '1'), size=9),
                    xanchor="right",
                    yanchor="bottom",
                    row=1, col=1
                )
        except Exception:
            pass  # RSI SZ drawing failed, skip it

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
        row=2, col=1
    )

    # ===== ULTIMATE RSI SUBPLOT (Row 3) =====
    if has_ursi:
        try:
            arsi = ultimate_rsi_data['arsi']
            sig = ultimate_rsi_data['signal']
            ob = ultimate_rsi_data['ob']
            os_val = ultimate_rsi_data['os']

            valid_mask = arsi.notna()
            if valid_mask.any():
                dt = df.loc[valid_mask, 'datetime']
                arsi_vals = arsi[valid_mask]
                sig_vals = sig[valid_mask]

                # Color RSI by zone: green if OB, red if OS, white otherwise
                arsi_colors = ['#089981' if v > ob else '#f23645' if v < os_val else '#c0c0c0'
                               for v in arsi_vals]

                # RSI line (colored segments via markers)
                fig.add_trace(
                    go.Scatter(
                        x=dt, y=arsi_vals,
                        mode='lines',
                        name='Ultimate RSI',
                        line=dict(color='#c0c0c0', width=1.5),
                        showlegend=False,
                        hovertemplate='URSI: %{y:.1f}<extra></extra>'
                    ),
                    row=3, col=1
                )

                # Signal line
                fig.add_trace(
                    go.Scatter(
                        x=dt, y=sig_vals,
                        mode='lines',
                        name='Signal',
                        line=dict(color='#ff5d00', width=1, dash='dot'),
                        showlegend=False,
                        hovertemplate='Signal: %{y:.1f}<extra></extra>'
                    ),
                    row=3, col=1
                )

                # OB fill (RSI above OB)
                arsi_ob = arsi_vals.where(arsi_vals > ob, ob)
                fig.add_trace(
                    go.Scatter(x=dt, y=[ob]*len(dt), mode='lines',
                               line=dict(color='rgba(0,0,0,0)', width=0),
                               showlegend=False, hoverinfo='skip'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=dt, y=arsi_ob, mode='lines',
                               line=dict(color='rgba(0,0,0,0)', width=0),
                               fill='tonexty', fillcolor='rgba(8,153,129,0.25)',
                               showlegend=False, hoverinfo='skip'),
                    row=3, col=1
                )

                # OS fill (RSI below OS)
                arsi_os = arsi_vals.where(arsi_vals < os_val, os_val)
                fig.add_trace(
                    go.Scatter(x=dt, y=arsi_os, mode='lines',
                               line=dict(color='rgba(0,0,0,0)', width=0),
                               showlegend=False, hoverinfo='skip'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=dt, y=[os_val]*len(dt), mode='lines',
                               line=dict(color='rgba(0,0,0,0)', width=0),
                               fill='tonexty', fillcolor='rgba(242,54,69,0.25)',
                               showlegend=False, hoverinfo='skip'),
                    row=3, col=1
                )

                # Horizontal reference lines
                for level, label in [(ob, 'OB'), (50, 'Mid'), (os_val, 'OS')]:
                    fig.add_hline(y=level, line_dash="dot", line_color="#555555",
                                  line_width=1, row=3, col=1,
                                  annotation_text=label, annotation_position="right")

                fig.update_yaxes(title_text="URSI", range=[0, 100],
                                 showgrid=True, gridwidth=1, gridcolor='#333333',
                                 row=3, col=1)
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333',
                                 type='date', row=3, col=1)
        except Exception:
            pass  # Ultimate RSI drawing failed, skip it

    fig.update_layout(
        title=title,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=850 if has_ursi else 700,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color='white'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e'
    )
    
    fig.update_xaxes(
        title_text="Time (IST)",
        showgrid=True,
        gridwidth=1,
        gridcolor='#333333',
        type='date',
        row=2, col=1
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#333333',
        type='date',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Price (₹)",
        showgrid=True,
        gridwidth=1,
        gridcolor='#333333',
        side='left',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Volume",
        showgrid=True,
        gridwidth=1,
        gridcolor='#333333',
        side='left',
        row=2, col=1
    )
    
    return fig

def display_metrics(ltp_data, df, db, symbol="NIFTY50"):
    """Display price metrics and save analytics"""
    if ltp_data and 'data' in ltp_data and not df.empty:
        current_price = None
        for exchange, data in ltp_data['data'].items():
            for security_id, price_data in data.items():
                current_price = price_data.get('last_price', 0)
                break
        
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
        if not df.empty and len(df) > 1:
            prev_close = df['close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            day_high = df['high'].max()
            day_low = df['low'].min()
            day_open = df['open'].iloc[0]
            volume = df['volume'].sum()
            avg_price = df['close'].mean()
            
            analytics_data = {
                'day_high': float(day_high),
                'day_low': float(day_low),
                'day_open': float(day_open),
                'day_close': float(current_price),
                'total_volume': int(volume),
                'avg_price': float(avg_price),
                'price_change': float(change),
                'price_change_pct': float(change_pct)
            }
            db.save_market_analytics(symbol, analytics_data)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                color = "price-up" if change >= 0 else "price-down"
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Current Price</h4>
                    <h2 class="{color}">₹{current_price:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                color = "price-up" if change >= 0 else "price-down"
                sign = "+" if change >= 0 else ""
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Change</h4>
                    <h3 class="{color}">{sign}{change:.2f} ({sign}{change_pct:.2f}%)</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Day High</h4>
                    <h3>₹{day_high:,.2f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Day Low</h4>
                    <h3>₹{day_low:,.2f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Volume</h4>
                    <h3>{volume:,}</h3>
                </div>
                """, unsafe_allow_html=True)

def validate_credentials(access_token, client_id):
    """Validate and clean API credentials"""
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
    """Generate a simple user ID based on session"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]
    return st.session_state.user_id

def create_csv_download(df_summary):
    """Create CSV download for option chain summary"""
    output = io.StringIO()
    df_summary.to_csv(output, index=False)
    return output.getvalue()


def analyze_option_chain(selected_expiry=None, pivot_data=None, vob_data=None):
    """Enhanced options chain analysis with expiry selection, HTF pivot data, and VOB data"""
    now = datetime.now(timezone("Asia/Kolkata"))
    
    # Get expiry list - use cached version for performance
    expiry_data = get_dhan_expiry_list_cached(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
    if not expiry_data or 'data' not in expiry_data:
        st.error("Failed to get expiry list from Dhan API")
        return None

    expiry_dates = expiry_data['data']
    if not expiry_dates:
        st.error("No expiry dates available")
        return None

    # Use selected expiry or default to first
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
    # Sort descending: OTM (higher strikes) at top, ITM (lower strikes) at bottom
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

    # Enhanced Greeks calculation with exact time-to-expiry
    T = calculate_exact_time_to_expiry(expiry)
    r = 0.06
    
    for idx, row in df.iterrows():
        strike = row['strikePrice']
        
        # Enhanced IV fallback using nearest strike average
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
    
    # Limit to ATM ± 2 strikes for faster UI (performance optimization)
    atm_plus_minus_2 = df[abs(df['strikePrice'] - atm_strike) <= 100]  # Assuming 50 point strikes
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

        # ===== Core Bias Calculations =====
        # LTP Bias: Higher CE LTP = Bullish (CE demand higher)
        row_data["LTP_Bias"] = "Bullish" if row.get('lastPrice_CE', 0) > row.get('lastPrice_PE', 0) else "Bearish"

        # OI Bias: Higher CE OI = Bearish (more call writing = resistance)
        row_data["OI_Bias"] = "Bearish" if row.get('openInterest_CE', 0) > row.get('openInterest_PE', 0) else "Bullish"

        # ChgOI Bias: Higher CE ChgOI = Bearish (new call writing)
        row_data["ChgOI_Bias"] = "Bearish" if row.get('changeinOpenInterest_CE', 0) > row.get('changeinOpenInterest_PE', 0) else "Bullish"

        # Volume Bias: Higher CE Volume = Bullish (more CE buying activity)
        row_data["Volume_Bias"] = "Bullish" if row.get('totalTradedVolume_CE', 0) > row.get('totalTradedVolume_PE', 0) else "Bearish"

        # ===== Greeks-Based Bias =====
        # Delta Bias: Higher CE Delta = Bullish
        row_data["Delta_Bias"] = "Bullish" if row.get('Delta_CE', 0) > abs(row.get('Delta_PE', 0)) else "Bearish"

        # Gamma Bias: Higher CE Gamma = Bullish
        row_data["Gamma_Bias"] = "Bullish" if row.get('Gamma_CE', 0) > row.get('Gamma_PE', 0) else "Bearish"

        # Theta Bias: Lower CE Theta (more negative) = Bullish
        row_data["Theta_Bias"] = "Bullish" if row.get('Theta_CE', 0) < row.get('Theta_PE', 0) else "Bearish"

        # ===== Order Flow Bias =====
        # AskQty Bias: Higher PE Ask = Bullish (more PE sellers)
        row_data["AskQty_Bias"] = "Bullish" if row.get('askQty_PE', 0) > row.get('askQty_CE', 0) else "Bearish"

        # BidQty Bias: Higher PE Bid = Bearish (more PE buyers)
        row_data["BidQty_Bias"] = "Bearish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bullish"

        # AskBid Bias (CE side): More bids than asks on CE = Bullish
        row_data["AskBid_Bias"] = "Bullish" if row.get('bidQty_CE', 0) > row.get('askQty_CE', 0) else "Bearish"

        # ===== Volatility Bias =====
        # IV Bias: Higher CE IV = Bullish (more CE demand driving IV up)
        row_data["IV_Bias"] = "Bullish" if row.get('impliedVolatility_CE', 0) > row.get('impliedVolatility_PE', 0) else "Bearish"

        # ===== Exposure Calculations =====
        delta_exp_ce = row.get('Delta_CE', 0) * row.get('openInterest_CE', 0)
        delta_exp_pe = row.get('Delta_PE', 0) * row.get('openInterest_PE', 0)
        gamma_exp_ce = row.get('Gamma_CE', 0) * row.get('openInterest_CE', 0)
        gamma_exp_pe = row.get('Gamma_PE', 0) * row.get('openInterest_PE', 0)

        row_data["DeltaExp"] = "Bullish" if delta_exp_ce > abs(delta_exp_pe) else "Bearish"
        row_data["GammaExp"] = "Bullish" if gamma_exp_ce > gamma_exp_pe else "Bearish"

        # ===== DVP (Delta-Volume-Price) Bias =====
        row_data["DVP_Bias"] = delta_volume_bias(
            row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
            row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
            row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
        )

        # ===== Bid-Ask Pressure =====
        row_data["BidAskPressure"] = bid_ask_pressure
        row_data["PressureBias"] = pressure_bias

        # ===== Weighted Score Calculation (Bias Engine v2) =====
        # Tier 1 (2.0): Institutional-grade signals
        # Tier 2 (1.5): Strong directional signals
        # Tier 3 (1.0): Medium reliability signals
        # Tier 4 (0.5): Weak/noisy signals
        # Theta_Bias excluded — not directionally reliable
        bias_weights = {
            'ChgOI_Bias': 2.0,  'DeltaExp': 2.0,  'GammaExp': 2.0,
            'OI_Bias': 1.5,     'PressureBias': 1.5, 'DVP_Bias': 1.5,
            'Volume_Bias': 1.0, 'LTP_Bias': 1.0,  'Delta_Bias': 1.0,
            'AskQty_Bias': 0.5, 'BidQty_Bias': 0.5, 'AskBid_Bias': 0.5,
            'Gamma_Bias': 0.5,  'IV_Bias': 0.5,
        }
        for k, weight in bias_weights.items():
            bias_val = row_data.get(k)
            if bias_val == "Bullish":
                score += weight
            elif bias_val == "Bearish":
                score -= weight

        row_data["BiasScore"] = score
        row_data["Verdict"] = final_verdict(score)

        # ===== Trading Signals =====
        # Operator Entry: Both OI and ChgOI aligned
        if row_data['OI_Bias'] == "Bullish" and row_data['ChgOI_Bias'] == "Bullish":
            row_data["Operator_Entry"] = "Entry Bull"
        elif row_data['OI_Bias'] == "Bearish" and row_data['ChgOI_Bias'] == "Bearish":
            row_data["Operator_Entry"] = "Entry Bear"
        else:
            row_data["Operator_Entry"] = "No Entry"

        # Scalp/Momentum: Based on score strength
        if score >= 6:
            row_data["Scalp_Moment"] = "Scalp Bull"
        elif score >= 3:
            row_data["Scalp_Moment"] = "Moment Bull"
        elif score <= -6:
            row_data["Scalp_Moment"] = "Scalp Bear"
        elif score <= -3:
            row_data["Scalp_Moment"] = "Moment Bear"
        else:
            row_data["Scalp_Moment"] = "No Signal"

        # FakeReal: Distinguish real moves from fake
        if score >= 6:
            row_data["FakeReal"] = "Real Up"
        elif 1.5 <= score < 6:
            row_data["FakeReal"] = "Fake Up"
        elif score <= -6:
            row_data["FakeReal"] = "Real Down"
        elif -6 < score <= -1.5:
            row_data["FakeReal"] = "Fake Down"
        else:
            row_data["FakeReal"] = "No Move"

        # ===== Comparison Strings for Display =====
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

    # Define columns to merge, filtering to only those that exist in df
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

    # Calculate PCR if the required columns exist
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

    # ===== GAMMA EXPOSURE SUPPORT/RESISTANCE =====
    # Calculate Gamma Exposure for each strike
    if 'Gamma_CE' in df_summary.columns and 'openInterest_CE' in df_summary.columns:
        df_summary['GammaExp_CE'] = df_summary['Gamma_CE'] * df_summary['openInterest_CE']
        df_summary['GammaExp_PE'] = df_summary['Gamma_PE'] * df_summary['openInterest_PE']
        df_summary['GammaExp_Net'] = df_summary['GammaExp_CE'] - df_summary['GammaExp_PE']

        # Max Gamma CE = Resistance, Max Gamma PE = Support
        max_gamma_ce_strike = df_summary.loc[df_summary['GammaExp_CE'].idxmax(), 'Strike'] if not df_summary['GammaExp_CE'].isna().all() else None
        max_gamma_pe_strike = df_summary.loc[df_summary['GammaExp_PE'].idxmax(), 'Strike'] if not df_summary['GammaExp_PE'].isna().all() else None

        df_summary['Gamma_SR'] = df_summary['Strike'].apply(
            lambda x: '🔴 Γ-Resist' if x == max_gamma_ce_strike else ('🟢 Γ-Support' if x == max_gamma_pe_strike else '-')
        )
    else:
        df_summary['Gamma_SR'] = '-'

    # ===== DELTA EXPOSURE SUPPORT/RESISTANCE =====
    if 'Delta_CE' in df_summary.columns and 'openInterest_CE' in df_summary.columns:
        df_summary['DeltaExp_CE'] = df_summary['Delta_CE'] * df_summary['openInterest_CE']
        df_summary['DeltaExp_PE'] = abs(df_summary['Delta_PE'] * df_summary['openInterest_PE'])
        df_summary['DeltaExp_Net'] = df_summary['DeltaExp_CE'] - df_summary['DeltaExp_PE']

        # Max Delta CE = Resistance, Max Delta PE = Support
        max_delta_ce_strike = df_summary.loc[df_summary['DeltaExp_CE'].idxmax(), 'Strike'] if not df_summary['DeltaExp_CE'].isna().all() else None
        max_delta_pe_strike = df_summary.loc[df_summary['DeltaExp_PE'].idxmax(), 'Strike'] if not df_summary['DeltaExp_PE'].isna().all() else None

        df_summary['Delta_SR'] = df_summary['Strike'].apply(
            lambda x: '🔴 Δ-Resist' if x == max_delta_ce_strike else ('🟢 Δ-Support' if x == max_delta_pe_strike else '-')
        )
    else:
        df_summary['Delta_SR'] = '-'

    # ===== MARKET DEPTH SUPPORT/RESISTANCE =====
    # Based on Bid/Ask quantities - High Bid = Support, High Ask = Resistance
    if 'bidQty_CE' in df_summary.columns and 'askQty_CE' in df_summary.columns:
        df_summary['Depth_CE'] = df_summary['bidQty_CE'] + df_summary['askQty_CE']
        df_summary['Depth_PE'] = df_summary['bidQty_PE'] + df_summary['askQty_PE']

        # Max bid on PE side = Support (buyers defending), Max ask on CE side = Resistance (sellers defending)
        max_bid_pe_strike = df_summary.loc[df_summary['bidQty_PE'].idxmax(), 'Strike'] if not df_summary['bidQty_PE'].isna().all() else None
        max_ask_ce_strike = df_summary.loc[df_summary['askQty_CE'].idxmax(), 'Strike'] if not df_summary['askQty_CE'].isna().all() else None

        df_summary['Depth_SR'] = df_summary['Strike'].apply(
            lambda x: '🔴 Depth-R' if x == max_ask_ce_strike else ('🟢 Depth-S' if x == max_bid_pe_strike else '-')
        )
    else:
        df_summary['Depth_SR'] = '-'

    # ===== OI WALL SUPPORT/RESISTANCE =====
    # Highest OI CE = Resistance Wall, Highest OI PE = Support Wall
    if 'openInterest_CE' in df_summary.columns and 'openInterest_PE' in df_summary.columns:
        max_oi_ce_strike = df_summary.loc[df_summary['openInterest_CE'].idxmax(), 'Strike'] if not df_summary['openInterest_CE'].isna().all() else None
        max_oi_pe_strike = df_summary.loc[df_summary['openInterest_PE'].idxmax(), 'Strike'] if not df_summary['openInterest_PE'].isna().all() else None

        # Also find 2nd highest for additional levels
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

    # ===== CHANGE IN OI WALL (FRESH BUILDUP/UNWINDING) =====
    # Positive ChgOI = Fresh Buildup, Negative ChgOI = Unwinding
    if 'changeinOpenInterest_CE' in df_summary.columns and 'changeinOpenInterest_PE' in df_summary.columns:
        # Find max positive ChgOI (fresh buildup)
        max_chgoi_ce_idx = df_summary['changeinOpenInterest_CE'].idxmax()
        max_chgoi_pe_idx = df_summary['changeinOpenInterest_PE'].idxmax()
        max_chgoi_ce_strike = df_summary.loc[max_chgoi_ce_idx, 'Strike'] if df_summary.loc[max_chgoi_ce_idx, 'changeinOpenInterest_CE'] > 0 else None
        max_chgoi_pe_strike = df_summary.loc[max_chgoi_pe_idx, 'Strike'] if df_summary.loc[max_chgoi_pe_idx, 'changeinOpenInterest_PE'] > 0 else None

        # Find max negative ChgOI (unwinding)
        min_chgoi_ce_idx = df_summary['changeinOpenInterest_CE'].idxmin()
        min_chgoi_pe_idx = df_summary['changeinOpenInterest_PE'].idxmin()
        unwind_ce_strike = df_summary.loc[min_chgoi_ce_idx, 'Strike'] if df_summary.loc[min_chgoi_ce_idx, 'changeinOpenInterest_CE'] < 0 else None
        unwind_pe_strike = df_summary.loc[min_chgoi_pe_idx, 'Strike'] if df_summary.loc[min_chgoi_pe_idx, 'changeinOpenInterest_PE'] < 0 else None

        def get_chgoi_wall(strike):
            labels = []
            # Fresh CE buildup = Resistance forming
            if strike == max_chgoi_ce_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_CE'].values[0]
                labels.append(f'🔴 CE+{int(chgoi_val/1000)}K')
            # Fresh PE buildup = Support forming
            if strike == max_chgoi_pe_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_PE'].values[0]
                labels.append(f'🟢 PE+{int(chgoi_val/1000)}K')
            # CE unwinding = Resistance weakening
            if strike == unwind_ce_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_CE'].values[0]
                labels.append(f'⚪ CE{int(chgoi_val/1000)}K')
            # PE unwinding = Support weakening
            if strike == unwind_pe_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_PE'].values[0]
                labels.append(f'⚪ PE{int(chgoi_val/1000)}K')
            return ' | '.join(labels) if labels else '-'

        df_summary['ChgOI_Wall'] = df_summary['Strike'].apply(get_chgoi_wall)
    else:
        df_summary['ChgOI_Wall'] = '-'

    # ===== MAX PAIN CALCULATION =====
    max_pain_strike, pain_df = calculate_max_pain(df_summary, underlying)
    if max_pain_strike:
        df_summary['Max_Pain'] = df_summary['Strike'].apply(
            lambda x: '🎯 MAX PAIN' if x == max_pain_strike else '-'
        )
    else:
        df_summary['Max_Pain'] = '-'

    # Define columns for display (ordered as per user preference for entry judgment)
    display_cols = ['Strike', 'PCR', 'Verdict', 'ChgOI_Bias', 'Volume_Bias', 'Max_Pain',
                    # Support/Resistance Levels
                    'Gamma_SR', 'Delta_SR', 'Depth_SR', 'OI_Wall', 'ChgOI_Wall',
                    # Bias columns for analysis
                    'Delta_Bias', 'Gamma_Bias', 'Theta_Bias', 'AskQty_Bias', 'BidQty_Bias', 'IV_Bias',
                    'DeltaExp', 'GammaExp', 'DVP_Bias', 'PressureBias', 'BidAskPressure',
                    # Decision columns
                    'BiasScore', 'Operator_Entry', 'Scalp_Moment', 'FakeReal',
                    'ChgOI_Cmp', 'OI_Cmp', 'LTP_Bias', 'PCR_Signal', 'Zone', 'OI_Bias']

    # Filter to only existing columns
    display_cols = [col for col in display_cols if col in df_summary.columns]
    df_display = df_summary[display_cols].copy()

    # Define bias columns for styling
    bias_cols = [col for col in display_cols if '_Bias' in col or col in ['DeltaExp', 'GammaExp', 'PCR_Signal']]

    # Enhanced styling with ATM highlighting and bias coloring
    styled_df = df_display.style\
        .applymap(color_bias, subset=bias_cols)\
        .applymap(color_pcr, subset=['PCR'] if 'PCR' in display_cols else [])\
        .applymap(color_pressure, subset=['BidAskPressure'] if 'BidAskPressure' in display_cols else [])\
        .applymap(color_verdict, subset=['Verdict'] if 'Verdict' in display_cols else [])\
        .applymap(color_entry, subset=['Operator_Entry'] if 'Operator_Entry' in display_cols else [])\
        .applymap(color_fakereal, subset=['FakeReal'] if 'FakeReal' in display_cols else [])\
        .applymap(color_score, subset=['BiasScore'] if 'BiasScore' in display_cols else [])\
        .apply(highlight_atm_row, axis=1)

    # ===== HTF SUPPORT/RESISTANCE DATA COLLECTION =====
    # Collect all S/R data
    sr_data = []

    # Max Pain
    if max_pain_strike:
        sr_data.append({
            'Type': '🎯 Max Pain',
            'Level': f"₹{max_pain_strike:.0f}",
            'Source': 'Options OI',
            'Strength': 'High',
            'Signal': 'Price magnet at expiry'
        })

    # OI Wall Support (Max PE OI)
    if 'openInterest_PE' in df_summary.columns:
        max_pe_oi_idx = df_summary['openInterest_PE'].idxmax()
        max_pe_oi_strike = df_summary.loc[max_pe_oi_idx, 'Strike']
        max_pe_oi_val = df_summary.loc[max_pe_oi_idx, 'openInterest_PE']
        sr_data.append({
            'Type': '🟢 OI Wall Support',
            'Level': f"₹{max_pe_oi_strike:.0f}",
            'Source': f"PE OI: {max_pe_oi_val/100000:.1f}L",
            'Strength': 'High',
            'Signal': 'Strong support - PE writers defending'
        })

    # OI Wall Resistance (Max CE OI)
    if 'openInterest_CE' in df_summary.columns:
        max_ce_oi_idx = df_summary['openInterest_CE'].idxmax()
        max_ce_oi_strike = df_summary.loc[max_ce_oi_idx, 'Strike']
        max_ce_oi_val = df_summary.loc[max_ce_oi_idx, 'openInterest_CE']
        sr_data.append({
            'Type': '🔴 OI Wall Resistance',
            'Level': f"₹{max_ce_oi_strike:.0f}",
            'Source': f"CE OI: {max_ce_oi_val/100000:.1f}L",
            'Strength': 'High',
            'Signal': 'Strong resistance - CE writers defending'
        })

    # Gamma Exposure Support
    if 'GammaExp_PE' in df_summary.columns:
        max_gamma_pe_idx = df_summary['GammaExp_PE'].idxmax()
        max_gamma_pe_strike = df_summary.loc[max_gamma_pe_idx, 'Strike']
        sr_data.append({
            'Type': '🟢 Gamma Support',
            'Level': f"₹{max_gamma_pe_strike:.0f}",
            'Source': 'Gamma Exposure PE',
            'Strength': 'Medium',
            'Signal': 'Dealers hedge here - price sticky'
        })

    # Gamma Exposure Resistance
    if 'GammaExp_CE' in df_summary.columns:
        max_gamma_ce_idx = df_summary['GammaExp_CE'].idxmax()
        max_gamma_ce_strike = df_summary.loc[max_gamma_ce_idx, 'Strike']
        sr_data.append({
            'Type': '🔴 Gamma Resistance',
            'Level': f"₹{max_gamma_ce_strike:.0f}",
            'Source': 'Gamma Exposure CE',
            'Strength': 'Medium',
            'Signal': 'Dealers hedge here - price sticky'
        })

    # Delta Exposure Support
    if 'DeltaExp_PE' in df_summary.columns:
        max_delta_pe_idx = df_summary['DeltaExp_PE'].idxmax()
        max_delta_pe_strike = df_summary.loc[max_delta_pe_idx, 'Strike']
        sr_data.append({
            'Type': '🟢 Delta Support',
            'Level': f"₹{max_delta_pe_strike:.0f}",
            'Source': 'Delta Exposure PE',
            'Strength': 'Medium',
            'Signal': 'Directional bias support'
        })

    # Delta Exposure Resistance
    if 'DeltaExp_CE' in df_summary.columns:
        max_delta_ce_idx = df_summary['DeltaExp_CE'].idxmax()
        max_delta_ce_strike = df_summary.loc[max_delta_ce_idx, 'Strike']
        sr_data.append({
            'Type': '🔴 Delta Resistance',
            'Level': f"₹{max_delta_ce_strike:.0f}",
            'Source': 'Delta Exposure CE',
            'Strength': 'Medium',
            'Signal': 'Directional bias resistance'
        })

    # ChgOI Fresh Buildup Support
    if 'changeinOpenInterest_PE' in df_summary.columns:
        max_chgoi_pe_idx = df_summary['changeinOpenInterest_PE'].idxmax()
        if df_summary.loc[max_chgoi_pe_idx, 'changeinOpenInterest_PE'] > 0:
            fresh_pe_strike = df_summary.loc[max_chgoi_pe_idx, 'Strike']
            fresh_pe_val = df_summary.loc[max_chgoi_pe_idx, 'changeinOpenInterest_PE']
            sr_data.append({
                'Type': '🟢 Fresh PE Buildup',
                'Level': f"₹{fresh_pe_strike:.0f}",
                'Source': f"ChgOI: +{fresh_pe_val/1000:.0f}K",
                'Strength': 'Fresh',
                'Signal': 'New support forming today'
            })

    # ChgOI Fresh Buildup Resistance
    if 'changeinOpenInterest_CE' in df_summary.columns:
        max_chgoi_ce_idx = df_summary['changeinOpenInterest_CE'].idxmax()
        if df_summary.loc[max_chgoi_ce_idx, 'changeinOpenInterest_CE'] > 0:
            fresh_ce_strike = df_summary.loc[max_chgoi_ce_idx, 'Strike']
            fresh_ce_val = df_summary.loc[max_chgoi_ce_idx, 'changeinOpenInterest_CE']
            sr_data.append({
                'Type': '🔴 Fresh CE Buildup',
                'Level': f"₹{fresh_ce_strike:.0f}",
                'Source': f"ChgOI: +{fresh_ce_val/1000:.0f}K",
                'Strength': 'Fresh',
                'Signal': 'New resistance forming today'
            })

    # Market Depth Support
    if 'bidQty_PE' in df_summary.columns:
        max_bid_pe_idx = df_summary['bidQty_PE'].idxmax()
        max_bid_pe_strike = df_summary.loc[max_bid_pe_idx, 'Strike']
        sr_data.append({
            'Type': '🟢 Depth Support',
            'Level': f"₹{max_bid_pe_strike:.0f}",
            'Source': 'Max PE Bid Qty',
            'Strength': 'Real-time',
            'Signal': 'Buyers actively defending'
        })

    # Market Depth Resistance
    if 'askQty_CE' in df_summary.columns:
        max_ask_ce_idx = df_summary['askQty_CE'].idxmax()
        max_ask_ce_strike = df_summary.loc[max_ask_ce_idx, 'Strike']
        sr_data.append({
            'Type': '🔴 Depth Resistance',
            'Level': f"₹{max_ask_ce_strike:.0f}",
            'Source': 'Max CE Ask Qty',
            'Strength': 'Real-time',
            'Signal': 'Sellers actively defending'
        })

    # ===== HTF PIVOT-BASED SUPPORT/RESISTANCE (5M, 15M, 1H) =====
    if pivot_data:
        # Group pivots by timeframe
        tf_pivots = {}
        for pivot in pivot_data:
            tf = pivot['timeframe']
            if tf not in tf_pivots:
                tf_pivots[tf] = {'highs': [], 'lows': []}
            if pivot['type'] == 'high':
                tf_pivots[tf]['highs'].append(pivot['value'])
            else:
                tf_pivots[tf]['lows'].append(pivot['value'])

        # Add 5M pivots
        if '5M' in tf_pivots:
            if tf_pivots['5M']['lows']:
                latest_5m_support = max(tf_pivots['5M']['lows'])  # Nearest support
                sr_data.append({
                    'Type': '🟢 5M Pivot Support',
                    'Level': f"₹{latest_5m_support:.0f}",
                    'Source': '5-Min Timeframe',
                    'Strength': 'Intraday',
                    'Signal': 'Short-term support level'
                })
            if tf_pivots['5M']['highs']:
                latest_5m_resist = min(tf_pivots['5M']['highs'])  # Nearest resistance
                sr_data.append({
                    'Type': '🔴 5M Pivot Resistance',
                    'Level': f"₹{latest_5m_resist:.0f}",
                    'Source': '5-Min Timeframe',
                    'Strength': 'Intraday',
                    'Signal': 'Short-term resistance level'
                })

        # Add 15M pivots
        if '15M' in tf_pivots:
            if tf_pivots['15M']['lows']:
                latest_15m_support = max(tf_pivots['15M']['lows'])
                sr_data.append({
                    'Type': '🟢 15M Pivot Support',
                    'Level': f"₹{latest_15m_support:.0f}",
                    'Source': '15-Min Timeframe',
                    'Strength': 'Swing',
                    'Signal': 'Key intraday support'
                })
            if tf_pivots['15M']['highs']:
                latest_15m_resist = min(tf_pivots['15M']['highs'])
                sr_data.append({
                    'Type': '🔴 15M Pivot Resistance',
                    'Level': f"₹{latest_15m_resist:.0f}",
                    'Source': '15-Min Timeframe',
                    'Strength': 'Swing',
                    'Signal': 'Key intraday resistance'
                })

        # Add 1H pivots
        if '1H' in tf_pivots:
            if tf_pivots['1H']['lows']:
                latest_1h_support = max(tf_pivots['1H']['lows'])
                sr_data.append({
                    'Type': '🟢 1H Pivot Support',
                    'Level': f"₹{latest_1h_support:.0f}",
                    'Source': '1-Hour Timeframe',
                    'Strength': 'Major',
                    'Signal': 'Strong hourly support - watch closely'
                })
            if tf_pivots['1H']['highs']:
                latest_1h_resist = min(tf_pivots['1H']['highs'])
                sr_data.append({
                    'Type': '🔴 1H Pivot Resistance',
                    'Level': f"₹{latest_1h_resist:.0f}",
                    'Source': '1-Hour Timeframe',
                    'Strength': 'Major',
                    'Signal': 'Strong hourly resistance - watch closely'
                })

    # ===== VOLUME ORDER BLOCKS (VOB) SUPPORT/RESISTANCE =====
    vob_blocks = None
    if vob_data:
        vob_sr_levels = vob_data.get('sr_levels', [])
        vob_blocks = vob_data.get('blocks', None)

        # Add VOB levels to sr_data
        for vob_level in vob_sr_levels:
            sr_data.append({
                'Type': vob_level['Type'],
                'Level': vob_level['Level'],
                'Source': vob_level['Source'],
                'Strength': vob_level['Strength'],
                'Signal': vob_level['Signal']
            })

    # Return all data for external display
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
    """Display analytics dashboard"""
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
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = analytics_df['day_close'].mean()
            st.metric("Average Price", f"₹{avg_price:,.2f}")
        
        with col2:
            volatility = analytics_df['price_change_pct'].std()
            st.metric("Volatility (σ)", f"{volatility:.2f}%")
        
        with col3:
            max_gain = analytics_df['price_change_pct'].max()
            st.metric("Max Daily Gain", f"{max_gain:.2f}%")
        
        with col4:
            max_loss = analytics_df['price_change_pct'].min()
            st.metric("Max Daily Loss", f"{max_loss:.2f}%")

def main():
    st.title("📈 Nifty Trading & Options Analyzer")

    # Check if within market hours (8:30 AM to 3:45 PM IST)
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    market_open = current_time.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=15, minute=45, second=0, microsecond=0)

    is_market_hours = market_open <= current_time <= market_close
    is_weekday = current_time.weekday() < 5  # Monday = 0, Friday = 4

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
        # Don't return - allow viewing cached data, but skip live fetching

    # Initialize session state for PCR history (EARLY - before any potential failures)
    # This ensures history persists even when API fetches fail
    if 'pcr_history' not in st.session_state:
        st.session_state.pcr_history = []
    if 'pcr_last_valid_data' not in st.session_state:
        st.session_state.pcr_last_valid_data = None

    # Initialize session state for GEX (Gamma Exposure) tracking
    if 'gex_history' not in st.session_state:
        st.session_state.gex_history = []  # Per-strike GEX history like PCR
    if 'gex_last_valid_data' not in st.session_state:
        st.session_state.gex_last_valid_data = None
    if 'last_gex_alert' not in st.session_state:
        st.session_state.last_gex_alert = None
    if 'gex_current_strikes' not in st.session_state:
        st.session_state.gex_current_strikes = []

    # Initialize Supabase
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
        db.create_tables()
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return
    
    # Initialize API credentials
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
        
        # Check Telegram configuration
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            st.sidebar.success("Telegram notifications enabled")
        else:
            st.sidebar.warning("Telegram notifications disabled - configure bot token and chat ID")
        
    except Exception as e:
        st.error(f"Credential validation error: {str(e)}")
        return
    
    # Get user ID and preferences
    user_id = get_user_id()
    user_prefs = db.get_user_preferences(user_id)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Timeframe selection
    timeframes = {
        "1 min": "1",
        "3 min": "3", 
        "5 min": "5",
        "10 min": "10",
        "15 min": "15"
    }
    
    default_timeframe = next((k for k, v in timeframes.items() if v == user_prefs['timeframe']), "1 min")
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        list(timeframes.keys()),
        index=list(timeframes.keys()).index(default_timeframe)
    )
    
    interval = timeframes[selected_timeframe]
    
    # Pivot indicator controls
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
    
    # VWAP Indicator Settings
    st.sidebar.header("📈 VWAP Settings")
    show_vwap = st.sidebar.checkbox("Show VWAP", value=True, help="Volume Weighted Average Price with bands")

    if show_vwap:
        vwap_anchor = st.sidebar.selectbox(
            "Anchor Period",
            ["Session", "Week", "Month", "Quarter", "Year"],
            index=0,
            help="Period to anchor/reset VWAP calculation"
        )
        vwap_src = st.sidebar.selectbox(
            "Source",
            ["hlc3", "close", "hl2", "ohlc4", "open"],
            index=0,
            help="Price source for VWAP calculation"
        )
        vwap_calc_mode = st.sidebar.selectbox(
            "Bands Calculation Mode",
            ["Standard Deviation", "Percentage"],
            index=0,
            help="Standard Deviation or Percentage-based bands. Percentage mode: multiplier 1 = 1%"
        )

        st.sidebar.subheader("Bands Settings")
        vwap_show_band1 = st.sidebar.checkbox("Show Band #1", value=True, help="First standard deviation band (green)")
        vwap_band_mult1 = st.sidebar.number_input("Band #1 Multiplier", min_value=0.0, value=1.0, step=0.5, format="%.1f") if vwap_show_band1 else 1.0
        vwap_show_band2 = st.sidebar.checkbox("Show Band #2", value=False, help="Second standard deviation band (olive)")
        vwap_band_mult2 = st.sidebar.number_input("Band #2 Multiplier", min_value=0.0, value=2.0, step=0.5, format="%.1f") if vwap_show_band2 else 2.0
        vwap_show_band3 = st.sidebar.checkbox("Show Band #3", value=False, help="Third standard deviation band (teal)")
        vwap_band_mult3 = st.sidebar.number_input("Band #3 Multiplier", min_value=0.0, value=3.0, step=0.5, format="%.1f") if vwap_show_band3 else 3.0

        vwap_settings = {
            'anchor': vwap_anchor,
            'src': vwap_src,
            'calc_mode': vwap_calc_mode,
            'show_band_1': vwap_show_band1,
            'band_mult_1': vwap_band_mult1,
            'show_band_2': vwap_show_band2,
            'band_mult_2': vwap_band_mult2,
            'show_band_3': vwap_show_band3,
            'band_mult_3': vwap_band_mult3,
        }
    else:
        vwap_settings = None

    # Trading signal settings
    st.sidebar.header("🔔 Trading Signals")
    enable_signals = st.sidebar.checkbox("Enable Telegram Signals", value=True, help="Send notifications when conditions are met")
    
    # Configurable pivot proximity with both positive and negative values
    pivot_proximity = st.sidebar.slider(
        "Pivot Proximity (± Points)", 
        min_value=1, 
        max_value=20, 
        value=user_prefs.get('pivot_proximity', 5),
        help="Distance from pivot levels to trigger signals (both above and below)"
    )
    
    # VWAP + POC Alignment Signal
    enable_vwap_poc_signal = st.sidebar.checkbox("Enable VWAP+POC Signal", value=True, help="Send CALL/PUT signal when VWAP position + POC 1,2,3 alignment confirms direction")
    poc_alignment_proximity = st.sidebar.slider(
        "POC Alignment Proximity (pts)",
        min_value=1, max_value=10, value=3,
        help="Max spread between POC 1,2,3 values to consider them 'aligned'"
    ) if enable_vwap_poc_signal else 3

    require_gex_big_move = st.sidebar.checkbox(
        "Only signal when GEX = Big Move",
        value=False,
        help="Only fire VWAP+POC signal when GEX regime is Trending/Breakout (dealers short gamma = violent moves expected)"
    ) if enable_vwap_poc_signal else False

    if enable_signals:
        gex_filter_text = "\n• GEX: Only on Trending/Breakout" if require_gex_big_move else "\n• GEX: All regimes (+ ATM Bull/Bear)"
        st.sidebar.info(f"Signals sent when:\n• Price within ±{pivot_proximity}pts of pivot\n• All option bias aligned\n• ATM at support/resistance\n• VWAP+POC: POCs within ±{poc_alignment_proximity}pts{gex_filter_text}")

    # Options expiry selection
    st.sidebar.header("📅 Options Settings")
    
    # Get available expiry dates
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
    
    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh (2 min)", value=user_prefs['auto_refresh'])
    
    # Days back for data
    days_back = st.sidebar.slider("Days of Historical Data", 1, 5, user_prefs['days_back'])
    
    # Data source preference
    use_cache = st.sidebar.checkbox("Use Cached Data", value=True, help="Use database cache for faster loading")
    
    # Database management
    st.sidebar.header("🗑️ Database Management")
    cleanup_days = st.sidebar.selectbox("Clear History Older Than", [7, 14, 30], index=0)
    
    if st.sidebar.button("🗑 Clear History"):
        deleted_count = db.clear_old_candle_data(cleanup_days)
        st.sidebar.success(f"Deleted {deleted_count} old records")
    
    # Connection Test Section
    st.sidebar.header("🔧 Connection Test")
    
    if st.sidebar.button("Test Telegram Connection"):
        success, message = test_telegram_connection()
        if success:
            st.sidebar.success(message)
            
            # Send a test message
            test_msg = "🔔 Nifty Analyzer - Test message successful! ✅"
            send_telegram_message_sync(test_msg)
            st.sidebar.success("Test message sent to Telegram!")
        else:
            st.sidebar.error(message)
    
    # Save preferences
    if st.sidebar.button("💾 Save Preferences"):
        db.save_user_preferences(user_id, interval, auto_refresh, days_back, pivot_settings, pivot_proximity)
        st.sidebar.success("Preferences saved!")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Show analytics dashboard
    show_analytics = st.sidebar.checkbox("Show Analytics Dashboard", value=False)
    
    # Debug info
    st.sidebar.subheader("🔧 Debug Info")
    st.sidebar.write(f"Telegram Bot Token: {'✅ Set' if TELEGRAM_BOT_TOKEN else '❌ Missing'}")
    st.sidebar.write(f"Telegram Chat ID: {'✅ Set' if TELEGRAM_CHAT_ID else '❌ Missing'}")
    st.sidebar.write(f"Token length: {len(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else 0}")
    st.sidebar.write(f"Chat ID: {TELEGRAM_CHAT_ID}")
    
    # Initialize API
    api = DhanAPI(access_token, client_id)
    
    # Main layout - Trading chart and Options analysis side by side
    col1, col2 = st.columns([2, 1])

    # Initialize pivots variable (will be populated in col1, used in col2)
    pivots = None
    # Initialize VOB data (will be populated in col1, used in col2 for S/R tables)
    vob_data = None

    with col1:
        st.header("📈 Trading Chart")
        
        # Data fetching strategy
        df = pd.DataFrame()
        current_price = None
        
        if use_cache:
            df = db.get_candle_data("NIFTY50", "IDX_I", interval, hours_back=days_back*24)
            
            if df.empty or (datetime.now(pytz.UTC) - df['datetime'].max().tz_convert(pytz.UTC)).total_seconds() > 300:
                with st.spinner("Fetching latest data from API..."):
                    data = api.get_intraday_data(
                        security_id="13",
                        exchange_segment="IDX_I", 
                        instrument="INDEX",
                        interval=interval,
                        days_back=days_back
                    )
                    
                    if data:
                        df = process_candle_data(data, interval)
                        db.save_candle_data("NIFTY50", "IDX_I", interval, df)
        else:
            with st.spinner("Fetching fresh data from API..."):
                data = api.get_intraday_data(
                    security_id="13",
                    exchange_segment="IDX_I", 
                    instrument="INDEX",
                    interval=interval,
                    days_back=days_back
                )
                
                if data:
                    df = process_candle_data(data, interval)
                    db.save_candle_data("NIFTY50", "IDX_I", interval, df)
        
        # Get LTP data and current price
        ltp_data = api.get_ltp_data("13", "IDX_I")
        if ltp_data and 'data' in ltp_data:
            for exchange, data in ltp_data['data'].items():
                for security_id, price_data in data.items():
                    current_price = price_data.get('last_price', 0)
                    break
        
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
        # Display metrics
        if not df.empty:
            display_metrics(ltp_data, df, db)

        # Calculate Volume Order Blocks (VOB) early for chart display
        vob_blocks_for_chart = None
        if not df.empty and len(df) > 30:
            try:
                vob_detector = VolumeOrderBlocks(sensitivity=5)
                _, vob_blocks_for_chart = vob_detector.get_sr_levels(df)
            except Exception:
                vob_blocks_for_chart = None

        # Calculate Triple POC for chart display
        poc_data_for_chart = None
        if not df.empty and len(df) > 100:
            try:
                poc_calculator = TriplePOC(period1=10, period2=25, period3=70)
                poc_data_for_chart = poc_calculator.calculate_all_pocs(df)
            except Exception:
                poc_data_for_chart = None

        # Calculate Future Swing for chart display
        swing_data_for_chart = None
        if not df.empty and len(df) > 50:
            try:
                swing_calculator = FutureSwing(swing_length=30, history_samples=5, calc_type='Average')
                swing_data_for_chart = swing_calculator.analyze(df)
            except Exception:
                swing_data_for_chart = None

        # Calculate RSI Volatility Suppression Zones for chart display
        rsi_sz_data_for_chart = None
        if not df.empty and len(df) > 30:
            try:
                rsi_sz_calculator = RSIVolatilitySuppression(rsi_length=14, vol_length=5)
                rsi_sz_data_for_chart = rsi_sz_calculator.analyze(df)
            except Exception:
                rsi_sz_data_for_chart = None

        # Calculate Ultimate RSI for chart display
        ultimate_rsi_data_for_chart = None
        if not df.empty and len(df) > 20:
            try:
                ursi_calculator = UltimateRSI(length=7, smo_type='RMA', signal_length=14, signal_type='EMA', ob_value=70, os_value=40)
                ultimate_rsi_data_for_chart = ursi_calculator.calculate(df)
            except Exception:
                ultimate_rsi_data_for_chart = None

        # Calculate VWAP with bands for chart display
        vwap_data_for_chart = None
        if not df.empty and len(df) > 5 and vwap_settings is not None:
            try:
                vwap_calculator = VWAPIndicator(
                    src=vwap_settings['src'],
                    anchor=vwap_settings['anchor'],
                    calc_mode=vwap_settings['calc_mode'],
                    show_band_1=vwap_settings['show_band_1'],
                    band_mult_1=vwap_settings['band_mult_1'],
                    show_band_2=vwap_settings['show_band_2'],
                    band_mult_2=vwap_settings['band_mult_2'],
                    show_band_3=vwap_settings['show_band_3'],
                    band_mult_3=vwap_settings['band_mult_3'],
                )
                vwap_data_for_chart = vwap_calculator.calculate(df)
            except Exception:
                vwap_data_for_chart = None

        # Create and display chart
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
                rsi_sz_data=rsi_sz_data_for_chart,
                ultimate_rsi_data=ultimate_rsi_data_for_chart,
                vwap_data=vwap_data_for_chart
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data info
            col1_info, col2_info, col3_info, col4_info = st.columns(4)
            with col1_info:
                st.info(f"📊 Data Points: {len(df)}")
            with col2_info:
                latest_time = df['datetime'].max().strftime("%Y-%m-%d %H:%M:%S IST")
                st.info(f"🕐 Latest: {latest_time}")
            with col3_info:
                data_source = "Database Cache" if use_cache else "Live API"
                st.info(f"📡 Source: {data_source}")
            with col4_info:
                pivot_status = "✅ Enabled" if show_pivots else "❌ Disabled"
                st.info(f"📈 Pivots: {pivot_status}")
            
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

            # Reversal Detector Analysis
            st.markdown("---")
            st.markdown("## 🔄 Intraday Reversal Detector")

            try:
                # Get pivot lows and highs for S/R detection
                pivot_lows = []
                pivot_highs = []
                if show_pivots and len(df) > 50:
                    df_json = df.to_json()
                    pivots = cached_pivot_calculation(df_json, pivot_settings or {})
                    pivot_lows = [p['value'] for p in pivots if p['type'] == 'low']
                    pivot_highs = [p['value'] for p in pivots if p['type'] == 'high']

                # Calculate Volume Order Blocks (VOB) for S/R integration
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

                # Calculate BULLISH reversal score
                bull_score, bull_signals, bull_verdict = ReversalDetector.calculate_reversal_score(df, pivot_lows)

                # Calculate BEARISH reversal score
                bear_score, bear_signals, bear_verdict = ReversalDetector.calculate_bearish_reversal_score(df, pivot_highs)

                # Display both verdicts side by side
                col_bull, col_bear = st.columns(2)

                with col_bull:
                    st.markdown("### 🟢 Bullish Reversal")
                    if "STRONG BUY" in bull_verdict:
                        st.success(f"**{bull_verdict}**")
                    elif "MODERATE BUY" in bull_verdict:
                        st.warning(f"**{bull_verdict}**")
                    else:
                        st.info(f"**{bull_verdict}**")

                    st.markdown(f"**Score: {bull_signals.get('Reversal_Score', 0)}/6**")
                    st.markdown(f"- Selling Exhausted: {bull_signals.get('Selling_Exhausted', 'N/A')}")
                    st.markdown(f"- Higher Low: {bull_signals.get('Higher_Low', 'N/A')}")
                    st.markdown(f"- Strong Bullish Candle: {bull_signals.get('Strong_Bullish_Candle', 'N/A')}")
                    st.markdown(f"- Volume: {bull_signals.get('Volume_Signal', 'N/A')}")
                    st.markdown(f"- Above VWAP: {bull_signals.get('Above_VWAP', 'N/A')}")

                with col_bear:
                    st.markdown("### 🔴 Bearish Reversal")
                    if "STRONG SELL" in bear_verdict:
                        st.error(f"**{bear_verdict}**")
                    elif "MODERATE SELL" in bear_verdict:
                        st.warning(f"**{bear_verdict}**")
                    else:
                        st.info(f"**{bear_verdict}**")

                    st.markdown(f"**Score: {bear_signals.get('Bearish_Score', 0)}/6**")
                    st.markdown(f"- Buying Exhausted: {bear_signals.get('Buying_Exhausted', 'N/A')}")
                    st.markdown(f"- Lower High: {bear_signals.get('Lower_High', 'N/A')}")
                    st.markdown(f"- Strong Bearish Candle: {bear_signals.get('Strong_Bearish_Candle', 'N/A')}")
                    st.markdown(f"- Volume: {bear_signals.get('Volume_Signal', 'N/A')}")
                    st.markdown(f"- Below VWAP: {bear_signals.get('Below_VWAP', 'N/A')}")

                # VWAP display
                if bull_signals.get('VWAP'):
                    st.info(f"📊 **VWAP:** ₹{bull_signals.get('VWAP')} | **Day High:** ₹{bull_signals.get('Day_High', 'N/A')} | **Day Low:** ₹{bull_signals.get('Day_Low', 'N/A')}")

                # Entry Rules Expander
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

            # ===== VWAP INDICATOR TABLE =====
            if vwap_data_for_chart and vwap_data_for_chart.get('latest_vwap') is not None:
                st.markdown("---")
                st.markdown("## 📈 VWAP Indicator (Volume Weighted Average Price)")

                try:
                    vd = vwap_data_for_chart

                    # Summary metrics row
                    vwap_col1, vwap_col2, vwap_col3 = st.columns(3)

                    with vwap_col1:
                        vwap_val = vd['latest_vwap']
                        price_diff = vd.get('price_vs_vwap', 0)
                        diff_color = "#15dd7c" if price_diff >= 0 else "#eb7514"
                        diff_icon = "Above" if price_diff >= 0 else "Below"
                        st.markdown(f"""
                        <div style="background-color: {diff_color}20; padding: 15px; border-radius: 10px; border: 2px solid {diff_color};">
                            <h4 style="color: {diff_color}; margin: 0;">VWAP</h4>
                            <h2 style="color: {diff_color}; margin: 5px 0;">₹{vwap_val:,.2f}</h2>
                            <p style="color: white; margin: 0;">Anchor: {vd.get('anchor', 'Session')} | Mode: {vd.get('calc_mode', 'Std Dev')}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with vwap_col2:
                        price_pct = vd.get('price_vs_vwap_pct', 0)
                        pct_color = "#15dd7c" if price_pct >= 0 else "#eb7514"
                        pct_sign = "+" if price_pct >= 0 else ""
                        st.markdown(f"""
                        <div style="background-color: {pct_color}20; padding: 15px; border-radius: 10px; border: 2px solid {pct_color};">
                            <h4 style="color: {pct_color}; margin: 0;">Price vs VWAP</h4>
                            <h2 style="color: {pct_color}; margin: 5px 0;">{diff_icon} ({pct_sign}{price_pct:.3f}%)</h2>
                            <p style="color: white; margin: 0;">Diff: {pct_sign}{price_diff:.2f} pts</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with vwap_col3:
                        band_pos = vd.get('band_position', 'N/A')
                        if 'Above' in band_pos:
                            bp_color = "#15dd7c"
                            bp_icon = "🟢"
                        elif 'Below' in band_pos:
                            bp_color = "#eb7514"
                            bp_icon = "🔴"
                        else:
                            bp_color = "#FFD700"
                            bp_icon = "🟡"
                        st.markdown(f"""
                        <div style="background-color: {bp_color}20; padding: 15px; border-radius: 10px; border: 2px solid {bp_color};">
                            <h4 style="color: {bp_color}; margin: 0;">Band Position</h4>
                            <h2 style="color: {bp_color}; margin: 5px 0;">{bp_icon} {band_pos}</h2>
                            <p style="color: white; margin: 0;">Price: ₹{vd.get('current_price', 0):,.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Bands detail table
                    vwap_table_data = []
                    vwap_table_data.append({
                        'Level': 'VWAP',
                        'Value': f"₹{vd['latest_vwap']:,.2f}",
                        'Upper': '-',
                        'Lower': '-',
                        'Multiplier': '-',
                    })

                    if vd.get('show_band_1') and vd.get('latest_upper_1') is not None:
                        vwap_table_data.append({
                            'Level': 'Band #1',
                            'Value': f"₹{vd['latest_vwap']:,.2f}",
                            'Upper': f"₹{vd['latest_upper_1']:,.2f}",
                            'Lower': f"₹{vd['latest_lower_1']:,.2f}",
                            'Multiplier': f"{vd['band_mult_1']}x",
                        })

                    if vd.get('show_band_2') and vd.get('latest_upper_2') is not None:
                        vwap_table_data.append({
                            'Level': 'Band #2',
                            'Value': f"₹{vd['latest_vwap']:,.2f}",
                            'Upper': f"₹{vd['latest_upper_2']:,.2f}",
                            'Lower': f"₹{vd['latest_lower_2']:,.2f}",
                            'Multiplier': f"{vd['band_mult_2']}x",
                        })

                    if vd.get('show_band_3') and vd.get('latest_upper_3') is not None:
                        vwap_table_data.append({
                            'Level': 'Band #3',
                            'Value': f"₹{vd['latest_vwap']:,.2f}",
                            'Upper': f"₹{vd['latest_upper_3']:,.2f}",
                            'Lower': f"₹{vd['latest_lower_3']:,.2f}",
                            'Multiplier': f"{vd['band_mult_3']}x",
                        })

                    vwap_df = pd.DataFrame(vwap_table_data)

                    def style_vwap_level(val):
                        if val == 'VWAP':
                            return 'background-color: #2962FF40; color: white; font-weight: bold'
                        elif 'Band #1' in str(val):
                            return 'background-color: #00C85340; color: white'
                        elif 'Band #2' in str(val):
                            return 'background-color: #80800040; color: white'
                        elif 'Band #3' in str(val):
                            return 'background-color: #00808040; color: white'
                        return ''

                    styled_vwap = vwap_df.style.applymap(style_vwap_level, subset=['Level'])
                    st.dataframe(styled_vwap, use_container_width=True, hide_index=True)

                    st.markdown("""
                    **VWAP Interpretation:**
                    - **VWAP** (Blue line): Fair value price based on volume - institutional benchmark
                    - **Above VWAP**: Bullish bias - buyers in control, VWAP acts as support
                    - **Below VWAP**: Bearish bias - sellers in control, VWAP acts as resistance
                    - **Band #1** (Green): 1 std dev - normal trading range, mean reversion zone
                    - **Band #2** (Olive): 2 std dev - extended move, potential reversal area
                    - **Band #3** (Teal): 3 std dev - extreme move, high probability reversal
                    """)

                except Exception as e:
                    st.warning(f"VWAP table display error: {str(e)}")

            # ===== TRIPLE POC + FUTURE SWING ANALYSIS =====
            st.markdown("---")
            st.markdown("## 📊 Triple POC + Future Swing Analysis")

            # Triple POC Table
            if poc_data_for_chart:
                st.markdown("### 🎯 Triple Point of Control (POC)")

                poc_table_data = []
                current_price_for_poc = df['close'].iloc[-1] if not df.empty else 0

                for poc_key, period_key in [('poc1', 'poc1'), ('poc2', 'poc2'), ('poc3', 'poc3')]:
                    poc = poc_data_for_chart.get(poc_key)
                    period = poc_data_for_chart.get('periods', {}).get(period_key, '')

                    if poc:
                        # Determine position relative to POC line
                        # Above POC = Bull, Below POC = Bear
                        if current_price_for_poc > poc.get('poc', 0):
                            position = "🟢 Above"
                            signal = "Bullish"
                        else:
                            position = "🔴 Below"
                            signal = "Bearish"

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

                    # Style the table
                    def style_poc_signal(val):
                        if val == 'Bullish':
                            return 'background-color: #00ff8840; color: white'
                        elif val == 'Bearish':
                            return 'background-color: #ff444440; color: white'
                        else:
                            return 'background-color: #FFD70040; color: white'

                    styled_poc = poc_df.style.applymap(style_poc_signal, subset=['Signal'])
                    st.dataframe(styled_poc, use_container_width=True, hide_index=True)

                    st.markdown("""
                    **POC Interpretation:**
                    - **POC 1 (10)**: Short-term volume profile - intraday support/resistance
                    - **POC 2 (25)**: Medium-term volume profile - swing trading levels
                    - **POC 3 (70)**: Long-term volume profile - major support/resistance
                    - **Above POC**: Bullish — market is bull, POC acts as support
                    - **Below POC**: Bearish — market is bear, POC acts as resistance
                    """)

            # Future Swing Table
            if swing_data_for_chart:
                st.markdown("### 🔄 Future Swing Projection")

                swings = swing_data_for_chart.get('swings', {})
                projection = swing_data_for_chart.get('projection')
                volume = swing_data_for_chart.get('volume', {})
                percentages = swing_data_for_chart.get('percentages', [])

                # Swing Summary
                swing_col1, swing_col2, swing_col3 = st.columns(3)

                with swing_col1:
                    direction = swings.get('direction', 'Unknown')
                    dir_color = "#15dd7c" if direction == 'bullish' else "#eb7514"
                    dir_icon = "🟢" if direction == 'bullish' else "🔴"
                    st.markdown(f"""
                    <div style="background-color: {dir_color}20; padding: 15px; border-radius: 10px; border: 2px solid {dir_color};">
                        <h4 style="color: {dir_color}; margin: 0;">Current Direction</h4>
                        <h2 style="color: {dir_color}; margin: 5px 0;">{dir_icon} {direction.upper()}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with swing_col2:
                    if projection:
                        target_color = "#15dd7c" if projection['direction'] == 'bullish' else "#eb7514"
                        st.markdown(f"""
                        <div style="background-color: {target_color}20; padding: 15px; border-radius: 10px; border: 2px solid {target_color};">
                            <h4 style="color: {target_color}; margin: 0;">Projected Target</h4>
                            <h2 style="color: {target_color}; margin: 5px 0;">₹{projection['target']:.0f}</h2>
                            <p style="color: white; margin: 0;">{projection['sign']}{projection['swing_pct']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Projection not available")

                with swing_col3:
                    delta = volume.get('delta', 0)
                    delta_color = "#15dd7c" if delta > 0 else "#eb7514"
                    delta_icon = "🟢" if delta > 0 else "🔴"
                    st.markdown(f"""
                    <div style="background-color: {delta_color}20; padding: 15px; border-radius: 10px; border: 2px solid {delta_color};">
                        <h4 style="color: {delta_color}; margin: 0;">Volume Delta</h4>
                        <h2 style="color: {delta_color}; margin: 5px 0;">{delta_icon} {delta:+,.0f}</h2>
                        <p style="color: white; margin: 0;">Buy: {volume.get('buy_volume', 0):,.0f} | Sell: {volume.get('sell_volume', 0):,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Swing Percentages Table
                if percentages:
                    st.markdown("### 📈 Historical Swing Percentages")

                    swing_pct_data = []
                    for i, pct in enumerate(percentages):
                        swing_pct_data.append({
                            'Swing': f"Swing {i+1}",
                            'Percentage': f"{pct:+.2f}%",
                            'Type': '🟢 Bullish' if pct > 0 else '🔴 Bearish'
                        })

                    # Add average
                    avg_pct = sum(abs(p) for p in percentages) / len(percentages) if percentages else 0
                    swing_pct_data.append({
                        'Swing': '📊 Average',
                        'Percentage': f"{avg_pct:.2f}%",
                        'Type': 'Used for projection'
                    })

                    swing_pct_df = pd.DataFrame(swing_pct_data)
                    st.dataframe(swing_pct_df, use_container_width=True, hide_index=True)

                # Swing Levels Table
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

            # ===== RSI VOLATILITY SUPPRESSION ZONES =====
            if rsi_sz_data_for_chart and rsi_sz_data_for_chart.get('zones'):
                st.markdown("---")
                st.markdown("## ∿ RSI Volatility Suppression Zones")

                current_signal = rsi_sz_data_for_chart.get('current_signal', 'No Zone')
                count_vol = rsi_sz_data_for_chart.get('count_volatility', 0)

                # Signal summary
                sz_col1, sz_col2, sz_col3 = st.columns(3)
                with sz_col1:
                    signal_color = "normal" if current_signal == 'Bullish Breakout' else ("inverse" if current_signal == 'Bearish Breakout' else "off")
                    st.metric("Current Signal", current_signal, delta=current_signal if current_signal != 'No Zone' else None, delta_color=signal_color)
                with sz_col2:
                    st.metric("Low Vol Bar Count", count_vol)
                with sz_col3:
                    total_zones = len(rsi_sz_data_for_chart['zones'])
                    bullish_count = sum(1 for z in rsi_sz_data_for_chart['zones'] if z['breakout'] == 'bullish')
                    bearish_count = sum(1 for z in rsi_sz_data_for_chart['zones'] if z['breakout'] == 'bearish')
                    st.metric("Zones Detected", f"{total_zones} (▲{bullish_count} / ▼{bearish_count})")

                # Zone table
                zone_table_data = []
                for idx, zone in enumerate(reversed(rsi_sz_data_for_chart['zones'][-10:]), 1):
                    breakout = zone.get('breakout', 'pending')
                    if breakout == 'bullish':
                        signal = '▲ Bullish'
                    elif breakout == 'bearish':
                        signal = '▼ Bearish'
                    else:
                        signal = '∿ Pending'

                    zone_table_data.append({
                        '#': idx,
                        'Zone Top': f"₹{zone['top']:.2f}",
                        'Zone Bottom': f"₹{zone['bottom']:.2f}",
                        'Range': f"₹{zone['top'] - zone['bottom']:.2f}",
                        'Bars': zone['end_idx'] - zone['start_idx'],
                        'Breakout': signal,
                    })

                if zone_table_data:
                    sz_df = pd.DataFrame(zone_table_data)

                    def style_sz_signal(val):
                        if '▲' in str(val):
                            return 'background-color: #00bbd440; color: white'
                        elif '▼' in str(val):
                            return 'background-color: #9b27b040; color: white'
                        return 'background-color: #80808040; color: white'

                    styled_sz = sz_df.style.applymap(style_sz_signal, subset=['Breakout'])
                    st.dataframe(styled_sz, use_container_width=True, hide_index=True)

                st.markdown("""
                **RSI Suppression Zone Interpretation:**
                - **Suppression Zone (∿)**: RSI volatility is low — price is consolidating
                - **Bullish Breakout (▲)**: Price broke above zone — momentum shifting up
                - **Bearish Breakout (▼)**: Price broke below zone — momentum shifting down
                - Longer suppression zones often lead to stronger breakouts
                - Use with POC and Swing levels for confluence-based entries
                """)

            # ===== ULTIMATE RSI [LuxAlgo] =====
            if ultimate_rsi_data_for_chart:
                st.markdown("---")
                st.markdown("## 📈 Ultimate RSI [LuxAlgo]")

                ursi_val = ultimate_rsi_data_for_chart.get('latest_arsi', 50)
                ursi_sig = ultimate_rsi_data_for_chart.get('latest_signal', 50)
                ursi_zone = ultimate_rsi_data_for_chart.get('zone', 'Neutral')
                ursi_cross = ultimate_rsi_data_for_chart.get('cross_signal', 'None')
                ursi_momentum = ultimate_rsi_data_for_chart.get('momentum', 'Neutral')

                ursi_col1, ursi_col2, ursi_col3, ursi_col4 = st.columns(4)
                with ursi_col1:
                    delta_color = "normal" if ursi_momentum == 'Bullish' else ("inverse" if ursi_momentum == 'Bearish' else "off")
                    st.metric("URSI Value", f"{ursi_val:.1f}", delta=ursi_momentum, delta_color=delta_color)
                with ursi_col2:
                    st.metric("Signal Line", f"{ursi_sig:.1f}")
                with ursi_col3:
                    zone_icon = "🟢" if ursi_zone == 'Overbought' else ("🔴" if ursi_zone == 'Oversold' else "⚪")
                    st.metric("Zone", f"{zone_icon} {ursi_zone}")
                with ursi_col4:
                    cross_icon = "🔼" if 'Bullish' in ursi_cross else ("🔽" if 'Bearish' in ursi_cross else "➖")
                    st.metric("Cross Signal", f"{cross_icon} {ursi_cross}")

                st.markdown("""
                **Ultimate RSI Interpretation:**
                - **Above 70 (OB)**: Overbought — expect bearish reversal
                - **Below 40 (OS)**: Oversold — expect bullish bounce
                - **URSI > Signal + Above 50**: Bullish momentum confirmed
                - **URSI < Signal + Below 50**: Bearish momentum confirmed
                - **Bullish/Bearish Cross**: URSI crossing signal line = momentum shift
                """)

        else:
            st.error("No data available. Please check your API credentials and try again.")
    
    with col2:
        st.header("📊 Options Analysis")

        # Options chain analysis with expiry selection (pass pivot data and VOB data for HTF S/R table)
        option_data = analyze_option_chain(selected_expiry, pivots, vob_data)

        if option_data and option_data.get('underlying'):
            underlying_price = option_data['underlying']
            df_summary = option_data['df_summary']
            st.info(f"**NIFTY SPOT:** {underlying_price:.2f}")

        else:
            option_data = None

    # ===== OPTIONS CHAIN AND HTF S/R TABLES BELOW CHART =====
    if option_data and option_data.get('underlying'):
        st.markdown("---")
        st.header("📊 Options Chain Analysis")

        # OI Change metrics
        st.markdown("## Open Interest Change (in Lakhs)")
        oi_col1, oi_col2 = st.columns(2)
        with oi_col1:
            st.metric("CALL ΔOI", f"{option_data['total_ce_change']:+.1f}L", delta_color="inverse")
        with oi_col2:
            st.metric("PUT ΔOI", f"{option_data['total_pe_change']:+.1f}L", delta_color="normal")

        # Option Chain Bias Summary Table
        st.markdown("## Option Chain Bias Summary")
        if option_data.get('styled_df') is not None:
            st.dataframe(option_data['styled_df'], use_container_width=True)

        # ===== HTF SUPPORT & RESISTANCE TABLES (SPLIT) =====
        st.markdown("---")
        st.markdown("## 📈 HTF Support & Resistance Levels")

        sr_data = option_data.get('sr_data', [])
        max_pain_strike = option_data.get('max_pain_strike')

        if sr_data:
            # Split into Support and Resistance
            support_data = [d for d in sr_data if '🟢' in d['Type'] or '🎯' in d['Type']]
            resistance_data = [d for d in sr_data if '🔴' in d['Type']]

            sr_col1, sr_col2 = st.columns(2)

            with sr_col1:
                st.markdown("### 🟢 SUPPORT LEVELS")
                if support_data:
                    support_df = pd.DataFrame(support_data)
                    st.dataframe(support_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No support levels identified")

            with sr_col2:
                st.markdown("### 🔴 RESISTANCE LEVELS")
                if resistance_data:
                    resistance_df = pd.DataFrame(resistance_data)
                    st.dataframe(resistance_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No resistance levels identified")

            # Max Pain summary
            if max_pain_strike:
                st.info(f"🎯 **Max Pain Level:** ₹{max_pain_strike:.0f} - Price magnet at expiry")

        # ===== PCR TIME-SERIES GRAPHS FOR ATM ± 2 STRIKES (SIDE BY SIDE) =====
        st.markdown("---")
        st.markdown("## 📊 PCR Analysis - Time Series (ATM ± 2)")

        # Helper function to create PCR chart (defined outside try block for reuse)
        def create_pcr_chart(history_df, col_name, color, title_prefix):
            """Helper to create individual PCR chart - col_name is now just strike price"""
            if col_name and col_name in history_df.columns:
                strike_val = col_name  # Column name is now just the strike price

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

                # Reference lines
                fig.add_hline(y=1.0, line_dash="dash", line_color="white", line_width=1)
                fig.add_hline(y=1.2, line_dash="dot", line_color="#00ff88", line_width=1)
                fig.add_hline(y=0.7, line_dash="dot", line_color="#ff4444", line_width=1)

                # Get current PCR value
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

        # Try to get new data and add to history
        pcr_data_available = False
        pcr_df = None

        df_summary = option_data.get('df_summary') if option_data else None
        if df_summary is not None and 'Zone' in df_summary.columns and 'PCR' in df_summary.columns:
            try:
                # Find ATM index
                atm_idx = df_summary[df_summary['Zone'] == 'ATM'].index
                if len(atm_idx) > 0:
                    atm_pos = df_summary.index.get_loc(atm_idx[0])

                    # Get ATM ± 2 strikes (5 strikes total)
                    start_idx = max(0, atm_pos - 2)
                    end_idx = min(len(df_summary), atm_pos + 3)

                    pcr_df = df_summary.iloc[start_idx:end_idx][['Strike', 'Zone', 'PCR', 'PCR_Signal',
                                                                   'openInterest_CE', 'openInterest_PE']].copy()

                    if not pcr_df.empty:
                        pcr_data_available = True
                        # Save as last valid data
                        st.session_state.pcr_last_valid_data = pcr_df.copy()

                        # Get current time
                        ist = pytz.timezone('Asia/Kolkata')
                        current_time = datetime.now(ist)

                        # Add current PCR data to history - store by STRIKE PRICE ONLY (not zone)
                        # This preserves history even when strikes change zone (OTM->ATM->ITM)
                        pcr_entry = {'time': current_time}
                        for _, row in pcr_df.iterrows():
                            strike_label = str(int(row['Strike']))  # Store by strike only
                            pcr_entry[strike_label] = row['PCR']

                        # Store current ATM ±2 strike positions for display
                        current_strikes = pcr_df['Strike'].tolist()
                        st.session_state.pcr_current_strikes = [int(s) for s in current_strikes]

                        # Check if we should add new entry (avoid duplicates within 30 seconds)
                        should_add = True
                        if st.session_state.pcr_history:
                            last_entry = st.session_state.pcr_history[-1]
                            time_diff = (current_time - last_entry['time']).total_seconds()
                            if time_diff < 30:
                                should_add = False

                        if should_add:
                            st.session_state.pcr_history.append(pcr_entry)
                            # Keep only last 200 entries (longer history)
                            if len(st.session_state.pcr_history) > 200:
                                st.session_state.pcr_history = st.session_state.pcr_history[-200:]

            except Exception as e:
                st.caption(f"⚠️ Current fetch issue: {str(e)[:50]}...")

        # ALWAYS try to display the graph if we have history (even if current fetch failed)
        if len(st.session_state.pcr_history) > 0:
            try:
                history_df = pd.DataFrame(st.session_state.pcr_history)

                # Get current ATM ±2 strikes (stored by strike price only)
                current_strikes = getattr(st.session_state, 'pcr_current_strikes', [])

                # If no current strikes available, try to get from last valid data
                if not current_strikes and st.session_state.pcr_last_valid_data is not None:
                    current_strikes = [int(s) for s in st.session_state.pcr_last_valid_data['Strike'].tolist()]

                # Sort strikes (ascending: ITM-2, ITM-1, ATM, OTM+1, OTM+2)
                current_strikes = sorted(current_strikes)

                # Create 5 columns for side-by-side display (ITM-2, ITM-1, ATM, OTM+1, OTM+2)
                pcr_col1, pcr_col2, pcr_col3, pcr_col4, pcr_col5 = st.columns(5)

                # Helper to display chart with signal
                def display_pcr_with_signal(container, fig, pcr_val):
                    if fig:
                        container.plotly_chart(fig, use_container_width=True)
                        if pcr_val > 1.2:
                            container.success("Bullish")
                        elif pcr_val < 0.7:
                            container.error("Bearish")
                        else:
                            container.warning("Neutral")

                # Get current zone info from pcr_df or last valid data
                zone_info = {}
                zone_df = pcr_df if pcr_df is not None else st.session_state.pcr_last_valid_data
                if zone_df is not None:
                    for _, row in zone_df.iterrows():
                        zone_info[int(row['Strike'])] = row['Zone']

                # Display 5 strikes: position 0=ITM-2, 1=ITM-1, 2=ATM, 3=OTM+1, 4=OTM+2
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

                # Show current PCR data table (use last valid if current not available)
                st.markdown("### Current PCR Values")
                display_df = pcr_df if pcr_df is not None else st.session_state.pcr_last_valid_data
                if display_df is not None:
                    pcr_display = display_df[['Strike', 'Zone', 'PCR', 'PCR_Signal']].copy()
                    pcr_display['CE OI (L)'] = (display_df['openInterest_CE'] / 100000).round(2)
                    pcr_display['PE OI (L)'] = (display_df['openInterest_PE'] / 100000).round(2)
                    st.dataframe(pcr_display, use_container_width=True, hide_index=True)

                # Show status and clear button
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

        # ===== GEX (GAMMA EXPOSURE) ANALYSIS SECTION =====
        st.markdown("---")
        st.markdown("## 📊 Gamma Exposure (GEX) Analysis - Dealer Hedging Flow")

        gex_data = None
        try:
            df_summary = option_data.get('df_summary')
            underlying_price = option_data.get('underlying')

            if df_summary is not None and underlying_price:
                # Calculate GEX
                gex_data = calculate_dealer_gex(df_summary, underlying_price)

                if gex_data:
                    gex_df = gex_data['gex_df']

                    # Save last valid GEX data
                    st.session_state.gex_last_valid_data = gex_data

                    # ===== GEX Summary Cards =====
                    gex_col1, gex_col2, gex_col3, gex_col4 = st.columns(4)

                    with gex_col1:
                        gex_color = gex_data['gex_color']
                        st.markdown(f"""
                        <div style="background-color: {gex_color}20; padding: 15px; border-radius: 10px; border: 2px solid {gex_color};">
                            <h4 style="color: {gex_color}; margin: 0;">Net GEX</h4>
                            <h2 style="color: {gex_color}; margin: 5px 0;">{gex_data['total_gex']:+.2f}L</h2>
                            <p style="color: white; margin: 0; font-size: 12px;">{gex_data['gex_signal']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with gex_col2:
                        if gex_data['gamma_flip_level']:
                            flip_color = "#00ff88" if underlying_price > gex_data['gamma_flip_level'] else "#ff4444"
                            st.markdown(f"""
                            <div style="background-color: {flip_color}20; padding: 15px; border-radius: 10px; border: 2px solid {flip_color};">
                                <h4 style="color: {flip_color}; margin: 0;">Gamma Flip</h4>
                                <h2 style="color: {flip_color}; margin: 5px 0;">₹{gex_data['gamma_flip_level']:.0f}</h2>
                                <p style="color: white; margin: 0; font-size: 12px;">{gex_data['spot_vs_flip']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background-color: #33333380; padding: 15px; border-radius: 10px; border: 2px solid #666;">
                                <h4 style="color: #999; margin: 0;">Gamma Flip</h4>
                                <h2 style="color: #999; margin: 5px 0;">N/A</h2>
                                <p style="color: #666; margin: 0; font-size: 12px;">No flip detected</p>
                            </div>
                            """, unsafe_allow_html=True)

                    with gex_col3:
                        if gex_data['gex_magnet']:
                            st.markdown(f"""
                            <div style="background-color: #00ff8820; padding: 15px; border-radius: 10px; border: 2px solid #00ff88;">
                                <h4 style="color: #00ff88; margin: 0;">GEX Magnet</h4>
                                <h2 style="color: #00ff88; margin: 5px 0;">₹{gex_data['gex_magnet']:.0f}</h2>
                                <p style="color: white; margin: 0; font-size: 12px;">Price attracted here</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background-color: #33333380; padding: 15px; border-radius: 10px; border: 2px solid #666;">
                                <h4 style="color: #999; margin: 0;">GEX Magnet</h4>
                                <h2 style="color: #999; margin: 5px 0;">N/A</h2>
                                <p style="color: #666; margin: 0; font-size: 12px;">No magnet</p>
                            </div>
                            """, unsafe_allow_html=True)

                    with gex_col4:
                        if gex_data['gex_repeller']:
                            st.markdown(f"""
                            <div style="background-color: #ff444420; padding: 15px; border-radius: 10px; border: 2px solid #ff4444;">
                                <h4 style="color: #ff4444; margin: 0;">GEX Repeller</h4>
                                <h2 style="color: #ff4444; margin: 5px 0;">₹{gex_data['gex_repeller']:.0f}</h2>
                                <p style="color: white; margin: 0; font-size: 12px;">Price accelerates here</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background-color: #33333380; padding: 15px; border-radius: 10px; border: 2px solid #666;">
                                <h4 style="color: #999; margin: 0;">GEX Repeller</h4>
                                <h2 style="color: #999; margin: 5px 0;">N/A</h2>
                                <p style="color: #666; margin: 0; font-size: 12px;">No repeller</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # ===== GEX Interpretation Box =====
                    st.markdown(f"""
                    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 4px solid {gex_data['gex_color']}; margin: 10px 0;">
                        <b style="color: {gex_data['gex_color']};">Market Regime:</b> {gex_data['gex_interpretation']}
                    </div>
                    """, unsafe_allow_html=True)

                    # ===== PCR × GEX Confluence Badge =====
                    st.markdown("### 🎯 PCR × GEX Confluence")

                    # Get ATM PCR for confluence
                    atm_data = df_summary[df_summary['Zone'] == 'ATM']
                    if not atm_data.empty:
                        atm_pcr = atm_data.iloc[0].get('PCR', 1.0)
                        confluence_badge, confluence_signal, confluence_strength = calculate_pcr_gex_confluence(atm_pcr, gex_data)

                        conf_col1, conf_col2 = st.columns([1, 3])
                        with conf_col1:
                            # Color based on signal
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

                    # ===== Net GEX Histogram =====
                    st.markdown("### 📊 Net GEX by Strike (Dealer Hedging Pressure)")

                    fig_gex = go.Figure()

                    # Add bars for Net GEX
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

                    # Add zero line
                    fig_gex.add_hline(y=0, line_dash="dash", line_color="white", line_width=2)

                    # Add gamma flip line if exists
                    if gex_data['gamma_flip_level']:
                        fig_gex.add_vline(
                            x=gex_data['gamma_flip_level'],
                            line_dash="dot",
                            line_color="#FFD700",
                            line_width=2,
                            annotation_text=f"Gamma Flip: ₹{gex_data['gamma_flip_level']:.0f}",
                            annotation_position="top"
                        )

                    # Add spot price line
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

                    # ===== GEX Breakdown Table =====
                    with st.expander("📋 GEX Breakdown by Strike"):
                        gex_display = gex_df.copy()
                        gex_display['Strike'] = gex_display['Strike'].apply(lambda x: f"₹{x:.0f}")

                        # Style the dataframe
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

                        styled_gex = gex_display.style.applymap(color_gex, subset=['Call_GEX', 'Put_GEX', 'Net_GEX'])
                        st.dataframe(styled_gex, use_container_width=True, hide_index=True)

                        st.markdown("""
                        **GEX Interpretation:**
                        - **Positive Net GEX (Green)**: Dealers LONG gamma → Price tends to PIN/REVERT
                        - **Negative Net GEX (Red)**: Dealers SHORT gamma → Price tends to ACCELERATE
                        - **GEX Magnet**: Strike with highest positive GEX (price attracted)
                        - **GEX Repeller**: Strike with most negative GEX (price accelerates away)
                        - **Gamma Flip**: Level where dealers switch from long to short gamma
                        """)

                    # ===== GEX TIME-SERIES PER STRIKE (Like PCR - 5 columns) =====
                    st.markdown("### 📈 GEX Time Series (ATM ± 2 Strikes)")

                    # Store GEX per strike in history (like PCR)
                    ist = pytz.timezone('Asia/Kolkata')
                    current_time = datetime.now(ist)

                    # Build GEX entry with per-strike values
                    gex_entry = {'time': current_time, 'total_gex': gex_data['total_gex']}
                    for _, row in gex_df.iterrows():
                        strike_label = str(int(row['Strike']))
                        gex_entry[strike_label] = row['Net_GEX']

                    # Store current strikes for display
                    current_gex_strikes = [int(row['Strike']) for _, row in gex_df.iterrows()]
                    st.session_state.gex_current_strikes = sorted(current_gex_strikes)

                    # Check if we should add new entry (avoid duplicates within 30 seconds)
                    should_add_gex = True
                    if st.session_state.gex_history:
                        last_gex_entry = st.session_state.gex_history[-1]
                        time_diff = (current_time - last_gex_entry['time']).total_seconds()
                        if time_diff < 30:
                            should_add_gex = False

                    if should_add_gex:
                        st.session_state.gex_history.append(gex_entry)
                        # Keep only last 200 entries
                        if len(st.session_state.gex_history) > 200:
                            st.session_state.gex_history = st.session_state.gex_history[-200:]

                    # Helper function to create individual GEX chart (like PCR)
                    def create_gex_chart(history_df, col_name, color, title_prefix):
                        """Helper to create individual GEX chart per strike"""
                        if col_name and col_name in history_df.columns:
                            strike_val = col_name

                            # Get data for this strike
                            gex_values = history_df[col_name].dropna()

                            # Calculate symmetric y-axis range (0 in middle)
                            max_abs = 20  # Default
                            if len(gex_values) > 0:
                                max_abs = max(abs(gex_values.max()), abs(gex_values.min()), 15)  # Min range of ±15
                            y_range = [-max_abs * 1.1, max_abs * 1.1]  # Add 10% padding

                            fig = go.Figure()

                            # Split into positive and negative for different colors
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

                            # Reference line at zero (critical - in the MIDDLE)
                            fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=2)
                            # Positive threshold (pinning zone) - ABOVE zero
                            fig.add_hline(y=10, line_dash="dot", line_color="#00ff88", line_width=1,
                                         annotation_text="+10", annotation_position="right")
                            # Negative threshold (acceleration zone) - BELOW zero
                            fig.add_hline(y=-10, line_dash="dot", line_color="#ff4444", line_width=1,
                                         annotation_text="-10", annotation_position="right")

                            # Get current GEX value
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
                                    range=y_range,  # Symmetric range with 0 in middle
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

                    # Display GEX charts - show immediately even with 1 data point
                    try:
                        # Get current strikes from gex_df
                        current_strikes = sorted([int(row['Strike']) for _, row in gex_df.iterrows()])
                        st.session_state.gex_current_strikes = current_strikes

                        # Get current GEX values for immediate display
                        current_gex_values = {}
                        for _, row in gex_df.iterrows():
                            strike = int(row['Strike'])
                            current_gex_values[strike] = row['Net_GEX']

                        # Create 5 columns for side-by-side display
                        gex_col1, gex_col2, gex_col3, gex_col4, gex_col5 = st.columns(5)

                        # Helper to display chart with signal
                        def display_gex_with_signal(container, fig, gex_val):
                            if fig:
                                container.plotly_chart(fig, use_container_width=True)
                                if gex_val > 10:
                                    container.success("📍 Pin Zone")
                                elif gex_val < -10:
                                    container.error("⚡ Accel Zone")
                                else:
                                    container.warning("➡️ Neutral")

                        # Display 5 strikes: ITM-2, ITM-1, ATM, OTM+1, OTM+2
                        position_labels = ['🟣 ITM-2', '🟣 ITM-1', '🟡 ATM', '🔵 OTM+1', '🔵 OTM+2']
                        position_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']
                        columns = [gex_col1, gex_col2, gex_col3, gex_col4, gex_col5]

                        # Check if we have history
                        has_history = len(st.session_state.gex_history) > 0
                        gex_history_df = pd.DataFrame(st.session_state.gex_history) if has_history else None

                        for i, col in enumerate(columns):
                            with col:
                                if i < len(current_strikes):
                                    strike = current_strikes[i]
                                    strike_col = str(strike)
                                    current_gex = current_gex_values.get(strike, 0)

                                    # Check if we have history for this strike
                                    if has_history and gex_history_df is not None and strike_col in gex_history_df.columns:
                                        fig, gex_val = create_gex_chart(gex_history_df, strike_col, position_colors[i], f'{position_labels[i]}')
                                        display_gex_with_signal(st, fig, gex_val)
                                    else:
                                        # Show current value as single point chart (immediate display)
                                        fig = go.Figure()

                                        # Add single point at current time
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

                                        # Reference lines - symmetric around zero
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

                        # Show current GEX data table
                        st.markdown("### Current GEX Values")
                        gex_display = gex_df[['Strike', 'Zone', 'Call_GEX', 'Put_GEX', 'Net_GEX']].copy()
                        gex_display['Strike'] = gex_display['Strike'].apply(lambda x: f"₹{x:.0f}")

                        # Color coding for table
                        def style_gex_val(val):
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
                                return ''
                            except:
                                return ''

                        styled_gex_table = gex_display.style.applymap(style_gex_val, subset=['Call_GEX', 'Put_GEX', 'Net_GEX'])
                        st.dataframe(styled_gex_table, use_container_width=True, hide_index=True)

                        # Show status and clear button
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

        # Expandable section for detailed Greeks and raw values
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

        # Add download button for CSV
        csv_data = create_csv_download(option_data['df_summary'])
        st.download_button(
            label="📥 Download Summary as CSV",
            data=csv_data,
            file_name=f"nifty_options_summary_{option_data['expiry']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    # ===== UNIFIED CONFLUENCE ENTRY ALERT =====
    if enable_signals and option_data and option_data.get('underlying') and not df.empty:
        try:
            _df_summary = option_data.get('df_summary')
            _underlying = option_data.get('underlying')
            if _df_summary is not None and _underlying:
                check_confluence_entry_signal(
                    df=df,
                    pivot_settings=pivot_settings,
                    df_summary=_df_summary,
                    current_price=_underlying,
                    pivot_proximity=pivot_proximity,
                    poc_data=poc_data_for_chart,
                    rsi_sz_data=rsi_sz_data_for_chart,
                    gex_data=gex_data,
                    ultimate_rsi_data=ultimate_rsi_data_for_chart,
                )
        except Exception:
            pass

    # ===== VWAP + POC ALIGNMENT SIGNAL =====
    if enable_vwap_poc_signal and not df.empty:
        try:
            _spot = None
            if option_data and option_data.get('underlying'):
                _spot = option_data['underlying']
            elif not df.empty:
                _spot = df['close'].iloc[-1]

            if _spot:
                check_vwap_poc_alignment_signal(
                    df=df,
                    current_price=_spot,
                    vwap_data=vwap_data_for_chart,
                    poc_data=poc_data_for_chart,
                    poc_proximity=poc_alignment_proximity,
                    option_data=option_data,
                    gex_data=gex_data,
                    require_gex_move=require_gex_big_move,
                )
        except Exception:
            pass

    # Analytics dashboard below
    if show_analytics:
        st.markdown("---")
        display_analytics_dashboard(db)
    
    # Show current time
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    st.sidebar.info(f"Last Updated: {current_time}")

if __name__ == "__main__":
    main()
