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
st_autorefresh(interval=80000, key="datarefresh")

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
    .atm-row {
        background-color: #FFD700 !important;
        font-weight: bold !important;
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
                    'timeframe': '5',
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
                'timeframe': '5', 
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

def get_dhan_option_chain(underlying_scrip: int, underlying_seg: str, expiry: str):
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
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan option chain: {e}")
        return None

def get_dhan_expiry_list(underlying_scrip: int, underlying_seg: str):
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
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan expiry list: {e}")
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


def check_trading_signals(df, pivot_settings, option_data, current_price, pivot_proximity=5):
    """Trading signal detection with Normal Bias OR OI Dominance (both require full ATM bias alignment)."""
    if df.empty or option_data is None or len(option_data) == 0 or not current_price:
        return

    try:
        df_json = df.to_json()
        pivots = cached_pivot_calculation(df_json, pivot_settings)
    except:
        pivots = PivotIndicator.get_all_pivots(df, pivot_settings)

    # Calculate Reversal Detector signals
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
            
            # Bias checks (including DeltaExp and GammaExp)
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
            
            # Change in OI
            ce_chg_oi = row.get('changeinOpenInterest_CE', 0)
            pe_chg_oi = row.get('changeinOpenInterest_PE', 0)

            # OI dominance logic (flipped as per your request)
            bullish_oi_confirm = pe_chg_oi > 1.5 * ce_chg_oi   # Bullish if Put ChgOI dominates
            bearish_oi_confirm = ce_chg_oi > 1.5 * pe_chg_oi   # Bearish if Call ChgOI dominates

            # === Bullish Call Signal (spot near pivot, above OR below) ===
            if (
                (all(bullish_conditions.values()) and abs(current_price - pivot_level['value']) <= pivot_proximity)
                or (bullish_oi_confirm and all(bullish_conditions.values()) and abs(current_price - pivot_level['value']) <= pivot_proximity)
            ):
                trigger_type = "📊 Normal Bias Trigger" if not bullish_oi_confirm else "🔥 OI Dominance Trigger"
                conditions_text = "\n".join([f"✅ {k}" for k, v in bullish_conditions.items() if v])
                price_diff = current_price - pivot_level['value']
                
                # Build reversal analysis text
                reversal_text = f"""
🔄 <b>REVERSAL DETECTOR:</b>
• Score: {reversal_signals.get('Reversal_Score', 0)}/6
• Selling Exhausted: {reversal_signals.get('Selling_Exhausted', 'N/A')}
• Higher Low: {reversal_signals.get('Higher_Low', 'N/A')}
• Strong Candle: {reversal_signals.get('Strong_Bullish_Candle', 'N/A')}
• Volume: {reversal_signals.get('Volume_Signal', 'N/A')}
• Above VWAP: {reversal_signals.get('Above_VWAP', 'N/A')}
• {reversal_verdict}"""

                message = f"""
🚨 <b>NIFTY CALL SIGNAL ALERT</b> 🚨

📍 <b>Spot Price:</b> ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} Pivot by {price_diff:+.2f} points)
📌 <b>Near Pivot:</b> {pivot_level['timeframe']} Level at ₹{pivot_level['value']:.2f}
🎯 <b>ATM Strike:</b> {atm_strike}

<b>✅ BULLISH CONDITIONS MET:</b>
{conditions_text}

⚡ <b>{trigger_type}</b>
⚡ <b>OI:</b> PE ChgOI {pe_chg_oi:,} vs CE ChgOI {ce_chg_oi:,}
{reversal_text}

📋 <b>SUGGESTED REVIEW:</b>
• Strike: {atm_strike} CE
• Stop Loss: {stop_loss_percent}%
• Manual verification required

🕐 Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
                try:
                    send_telegram_message_sync(message)
                    st.success("🟢 Bullish signal notification sent!")
                except Exception as e:
                    st.error(f"Failed to send notification: {e}")
            
            # === Bearish Put Signal (spot near pivot, above OR below) ===
            elif (
                (all(bearish_conditions.values()) and abs(current_price - pivot_level['value']) <= pivot_proximity)
                or (bearish_oi_confirm and all(bearish_conditions.values()) and abs(current_price - pivot_level['value']) <= pivot_proximity)
            ):
                trigger_type = "📊 Normal Bias Trigger" if not bearish_oi_confirm else "🔥 OI Dominance Trigger"
                conditions_text = "\n".join([f"🔴 {k}" for k, v in bearish_conditions.items() if v])
                price_diff = current_price - pivot_level['value']
                
                # Build reversal analysis text for bearish
                reversal_text = f"""
🔄 <b>REVERSAL DETECTOR:</b>
• Score: {reversal_signals.get('Reversal_Score', 0)}/6
• Selling Exhausted: {reversal_signals.get('Selling_Exhausted', 'N/A')}
• Higher Low: {reversal_signals.get('Higher_Low', 'N/A')}
• Strong Candle: {reversal_signals.get('Strong_Bullish_Candle', 'N/A')}
• Volume: {reversal_signals.get('Volume_Signal', 'N/A')}
• Above VWAP: {reversal_signals.get('Above_VWAP', 'N/A')}
• {reversal_verdict}"""

                message = f"""
🔴 <b>NIFTY PUT SIGNAL ALERT</b> 🔴

📍 <b>Spot Price:</b> ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} Pivot by {price_diff:+.2f} points)
📌 <b>Near Pivot:</b> {pivot_level['timeframe']} Level at ₹{pivot_level['value']:.2f}
🎯 <b>ATM Strike:</b> {atm_strike}

<b>🔴 BEARISH CONDITIONS MET:</b>
{conditions_text}

⚡ <b>{trigger_type}</b>
⚡ <b>OI:</b> CE ChgOI {ce_chg_oi:,} vs PE ChgOI {pe_chg_oi:,}
{reversal_text}

📋 <b>SUGGESTED REVIEW:</b>
• Strike: {atm_strike} PE
• Stop Loss: {stop_loss_percent}%
• Manual verification required

🕐 Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
                try:
                    send_telegram_message_sync(message)
                    st.success("🔴 Bearish signal notification sent!")
                except Exception as e:
                    st.error(f"Failed to send notification: {e}")


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
        if score >= 4:
            return 'background-color: #228B22; color: white'
        elif score >= 2:
            return 'background-color: #90EE90; color: black'
        elif score <= -4:
            return 'background-color: #DC143C; color: white'
        elif score <= -2:
            return 'background-color: #FFB6C1; color: black'
        else:
            return 'background-color: #FFFFE0; color: black'
    except:
        return ''

def highlight_atm_row(row):
    """Highlight ATM row in the dataframe"""
    if row['Zone'] == 'ATM':
        return ['background-color: #FFD700; font-weight: bold'] * len(row)
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

def create_candlestick_chart(df, title, interval, show_pivots=True, pivot_settings=None):
    """Create TradingView-style candlestick chart with optional pivot levels"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
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

    # Add VWAP line
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
        pass  # VWAP calculation failed, skip it

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
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=700,
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


def analyze_option_chain(selected_expiry=None, pivot_data=None):
    """Enhanced options chain analysis with expiry selection and HTF pivot data"""
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
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

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

        # ===== Score Calculation =====
        # Count all bias columns for scoring
        for k in row_data:
            if "_Bias" in k or k in ["DeltaExp", "GammaExp"]:
                bias_val = row_data[k]
                if bias_val == "Bullish":
                    score += 1
                elif bias_val == "Bearish":
                    score -= 1

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

        # FakeReal: Distinguish real moves from fake
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

    # Define columns for display (select key columns to avoid clutter)
    display_cols = ['Strike', 'Zone', 'Level', 'Max_Pain',
                    # Support/Resistance Levels
                    'Gamma_SR', 'Delta_SR', 'Depth_SR', 'OI_Wall', 'ChgOI_Wall',
                    # Bias columns
                    'LTP_Bias', 'OI_Bias', 'ChgOI_Bias', 'Volume_Bias',
                    'Delta_Bias', 'Gamma_Bias', 'AskQty_Bias', 'BidQty_Bias', 'AskBid_Bias', 'IV_Bias',
                    'DeltaExp', 'GammaExp', 'DVP_Bias', 'PressureBias', 'BidAskPressure',
                    'BiasScore', 'Verdict', 'Operator_Entry', 'Scalp_Moment', 'FakeReal',
                    'ChgOI_Cmp', 'OI_Cmp', 'PCR', 'PCR_Signal']

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
        'total_pe_change': total_pe_change
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
    
    default_timeframe = next((k for k, v in timeframes.items() if v == user_prefs['timeframe']), "5 min")
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
    
    if enable_signals:
        st.sidebar.info(f"Signals sent when:\n• Price within ±{pivot_proximity}pts of pivot\n• All option bias aligned\n• ATM at support/resistance")
    
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
        
        # Create and display chart
        if not df.empty:
            fig = create_candlestick_chart(
                df, 
                f"Nifty 50 - {selected_timeframe} Chart {'with Pivot Levels' if show_pivots else ''}", 
                interval,
                show_pivots=show_pivots,
                pivot_settings=pivot_settings
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

                *R = Resistance (Price ceiling), S = Support (Price floor)*
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

        else:
            st.error("No data available. Please check your API credentials and try again.")
    
    with col2:
        st.header("📊 Options Analysis")

        # Options chain analysis with expiry selection (pass pivot data for HTF S/R table)
        option_data = analyze_option_chain(selected_expiry, pivots)

        if option_data and option_data.get('underlying'):
            underlying_price = option_data['underlying']
            df_summary = option_data['df_summary']
            st.info(f"**NIFTY SPOT:** {underlying_price:.2f}")

            # Check for trading signals if enabled
            if enable_signals and not df.empty and df_summary is not None and len(df_summary) > 0:
                check_trading_signals(df, pivot_settings, df_summary, underlying_price, pivot_proximity)
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
