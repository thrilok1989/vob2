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
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Nifty Trading & ICC Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 80 seconds
st_autorefresh(interval=80000, key="datarefresh")

# ============================================================================
# CUSTOM CSS
# ============================================================================
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
    .icc-bullish {
        color: #00ff88;
        font-weight: bold;
    }
    .icc-bearish {
        color: #ff4444;
        font-weight: bold;
    }
    .icc-neutral {
        color: #888888;
        font-weight: bold;
    }
    .phase-box {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================
try:
    DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
    DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
    
    if not DHAN_CLIENT_ID:
        DHAN_CLIENT_ID = st.secrets.get("dhan", {}).get("client_id", "")
    if not DHAN_ACCESS_TOKEN:
        DHAN_ACCESS_TOKEN = st.secrets.get("dhan", {}).get("access_token", "")
        
    supabase_url = st.secrets.get("supabase", {}).get("url", "")
    supabase_key = st.secrets.get("supabase", {}).get("anon_key", "")
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
    
    if TELEGRAM_CHAT_ID and isinstance(TELEGRAM_CHAT_ID, (int, float)):
        TELEGRAM_CHAT_ID = str(int(TELEGRAM_CHAT_ID))
        
except Exception as e:
    st.error(f"Config error: {e}")
    DHAN_CLIENT_ID = ""
    DHAN_ACCESS_TOKEN = ""
    supabase_url = ""
    supabase_key = ""
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"

# ============================================================================
# CACHED FUNCTIONS
# ============================================================================
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

@st.cache_data(ttl=300)
def get_dhan_expiry_list_cached(underlying_scrip: int, underlying_seg: str):
    return get_dhan_expiry_list(underlying_scrip, underlying_seg)

# ============================================================================
# TELEGRAM FUNCTIONS
# ============================================================================
def send_telegram_message_sync(message):
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

# ============================================================================
# DATABASE CLASS
# ============================================================================
class SupabaseDB:
    def __init__(self, url, key):
        self.client: Client = create_client(url, key)
    
    def create_tables(self):
        try:
            self.client.table('candle_data').select('id').limit(1).execute()
        except:
            st.info("Database tables may need to be created.")
    
    def save_candle_data(self, symbol, exchange, timeframe, df):
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

# ============================================================================
# DHAN API CLASS
# ============================================================================
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

# ============================================================================
# ICC MARKET STRUCTURE CLASS (COMPLETE CONVERSION)
# ============================================================================
class ICCMarketStructure:
    def __init__(self, profile='Intraday', use_manual_settings=False, 
                 manual_swing_length=7, manual_consolidation_bars=20):
        
        self.profile = profile
        self.use_manual_settings = use_manual_settings
        self.manual_swing_length = manual_swing_length
        self.manual_consolidation_bars = manual_consolidation_bars
        
        # State variables
        self.last_swing_high = None
        self.last_swing_low = None
        self.prev_swing_high = None
        self.prev_swing_low = None
        self.prev_bullish_indication = None
        self.prev_bearish_indication = None
        self.last_high_bar = None
        self.last_low_bar = None
        self.bars_since_last_structure = 0
        self.bars_in_correction = 0
        self.bars_in_continuation = 0
        self.correction_range_high = None
        self.correction_range_low = None
        self.market_structure = 'no_setup'
        self.no_setup_label = 'No Setup'
        self.last_trend_dir = None
        self.indication_detected = False
        self.indication_level = None
        self.indication_type = None
        self.correction_phase = False
        self.last_entry_type = None
        self.cont_active = False
        self.last_cont_entry_bar = None
        self.last_entry_price = None
        self.last_stop_loss = None
        self.last_tp1_price = None
        self.last_tp2_price = None
        self.support_levels = []
        self.resistance_levels = []
        self.support_start_bars = []
        self.resistance_start_bars = []
        self.MAX_HIST_ZONES = 10
        self.prev_profile = profile
    
    def get_profiled_swing_length(self, current_tf_minutes):
        if self.profile == 'Entry':
            if current_tf_minutes <= 5: return 4
            elif current_tf_minutes <= 15: return 5
            elif current_tf_minutes <= 30: return 6
            elif current_tf_minutes <= 60: return 8
            elif current_tf_minutes <= 240: return 10
            else: return 12
        elif self.profile == 'Scalping':
            if current_tf_minutes <= 5: return 4
            elif current_tf_minutes <= 15: return 5
            elif current_tf_minutes <= 30: return 8
            elif current_tf_minutes <= 60: return 5
            elif current_tf_minutes <= 240: return 3
            else: return 2
        elif self.profile == 'Swing':
            if current_tf_minutes <= 5: return 4
            elif current_tf_minutes <= 15: return 5
            elif current_tf_minutes <= 30: return 17
            elif current_tf_minutes <= 60: return 10
            elif current_tf_minutes <= 240: return 7
            else: return 4
        else:  # Intraday
            if current_tf_minutes <= 5: return 4
            elif current_tf_minutes <= 15: return 5
            elif current_tf_minutes <= 30: return 12
            elif current_tf_minutes <= 60: return 7
            elif current_tf_minutes <= 240: return 5
            else: return 3
    
    def get_profiled_consolidation_bars(self, current_tf_minutes):
        if self.profile == 'Entry':
            if current_tf_minutes <= 5: return 40
            elif current_tf_minutes <= 15: return 26
            elif current_tf_minutes <= 30: return 20
            elif current_tf_minutes <= 60: return 15
            elif current_tf_minutes <= 240: return 12
            else: return 6
        elif self.profile == 'Scalping':
            if current_tf_minutes <= 5: return 35
            elif current_tf_minutes <= 15: return 21
            elif current_tf_minutes <= 30: return 13
            elif current_tf_minutes <= 60: return 8
            elif current_tf_minutes <= 240: return 6
            else: return 3
        elif self.profile == 'Swing':
            if current_tf_minutes <= 5: return 70
            elif current_tf_minutes <= 15: return 42
            elif current_tf_minutes <= 30: return 25
            elif current_tf_minutes <= 60: return 17
            elif current_tf_minutes <= 240: return 11
            else: return 7
        else:  # Intraday
            if current_tf_minutes <= 5: return 50
            elif current_tf_minutes <= 15: return 30
            elif current_tf_minutes <= 30: return 18
            elif current_tf_minutes <= 60: return 12
            elif current_tf_minutes <= 240: return 8
            else: return 5
    
    def get_profiled_wiggle_pct(self):
        if self.profile == 'Entry': return 0.40
        elif self.profile == 'Scalping': return 0.30
        elif self.profile == 'Intraday': return 0.20
        elif self.profile == 'Swing': return 0.10
        else: return 0.20
    
    def detect_swing_points(self, df, swing_length):
        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs = []
        swing_lows = []
        swing_high_indices = []
        swing_low_indices = []
        
        for i in range(swing_length, len(df) - swing_length):
            is_swing_high = True
            for j in range(1, swing_length + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(highs[i])
                swing_high_indices.append(i)
            
            is_swing_low = True
            for j in range(1, swing_length + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(lows[i])
                swing_low_indices.append(i)
        
        return swing_highs, swing_lows, swing_high_indices, swing_low_indices
    
    def calculate_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if len(atr) > 0 else 0
    
    def get_zone_buffer(self, price, atr_percent):
        base_buffer = price * 0.0003
        return base_buffer * (1 + atr_percent * 10)
    
    def calculate_consolidation(self, df, swing_length, consolidation_bars, wiggle_pct):
        if len(df) < consolidation_bars:
            return False, None, None
        
        recent_highs = df['high'].rolling(window=swing_length*2+1).max()
        recent_lows = df['low'].rolling(window=swing_length*2+1).min()
        
        current_price = df['close'].iloc[-1]
        current_high = recent_highs.iloc[-1]
        current_low = recent_lows.iloc[-1]
        
        if pd.isna(current_high) or pd.isna(current_low):
            return False, None, None
        
        upper_band = current_high * (1 + wiggle_pct / 100.0)
        lower_band = current_low * (1 - wiggle_pct / 100.0)
        
        price_in_range = current_price <= upper_band and current_price >= lower_band
        
        in_range_count = 0
        for i in range(min(consolidation_bars, len(df))):
            idx = -1 - i
            price = df['close'].iloc[idx]
            if price <= upper_band and price >= lower_band:
                in_range_count += 1
        
        is_consolidating = price_in_range and in_range_count >= consolidation_bars
        
        if is_consolidating:
            return True, current_high, current_low
        else:
            return False, None, None
    
    def update_structure(self, df, current_tf_minutes=15):
        if self.profile != self.prev_profile:
            self.correction_phase = False
            self.correction_range_high = None
            self.correction_range_low = None
            self.bars_in_correction = 0
            self.bars_in_continuation = 0
            self.indication_detected = False
            self.market_structure = 'no_setup'
            self.prev_profile = self.profile
        
        if self.use_manual_settings:
            swing_length = self.manual_swing_length
            consolidation_bars = self.manual_consolidation_bars
            wiggle_pct = 0.2
        else:
            swing_length = self.get_profiled_swing_length(current_tf_minutes)
            consolidation_bars = self.get_profiled_consolidation_bars(current_tf_minutes)
            wiggle_pct = self.get_profiled_wiggle_pct()
        
        swing_highs, swing_lows, high_indices, low_indices = self.detect_swing_points(df, swing_length)
        
        if swing_highs:
            self.prev_swing_high = self.last_swing_high
            self.last_swing_high = swing_highs[-1]
            self.last_high_bar = high_indices[-1] if high_indices else None
            self.bars_since_last_structure = 0
        else:
            self.bars_since_last_structure += 1
        
        if swing_lows:
            self.prev_swing_low = self.last_swing_low
            self.last_swing_low = swing_lows[-1]
            self.last_low_bar = low_indices[-1] if low_indices else None
            self.bars_since_last_structure = 0
        
        current_price = df['close'].iloc[-1]
        
        if (self.prev_swing_high is not None and self.last_swing_high is not None and 
            self.last_swing_high > self.prev_swing_high):
            self.prev_bullish_indication = self.last_swing_high
            self.indication_detected = True
            self.indication_level = self.last_swing_high
            self.indication_type = 'bullish'
            self.correction_phase = False
            self.market_structure = 'bullish_indication'
            self.bars_in_correction = 0
            self.correction_range_high = None
            self.correction_range_low = None
            self.last_entry_type = None
            self.last_trend_dir = 'bullish'
            self.no_setup_label = 'No Setup'
        
        if (self.prev_swing_low is not None and self.last_swing_low is not None and 
            self.last_swing_low < self.prev_swing_low):
            self.prev_bearish_indication = self.last_swing_low
            self.indication_detected = True
            self.indication_level = self.last_swing_low
            self.indication_type = 'bearish'
            self.correction_phase = False
            self.market_structure = 'bearish_indication'
            self.bars_in_correction = 0
            self.correction_range_high = None
            self.correction_range_low = None
            self.last_entry_type = None
            self.last_trend_dir = 'bearish'
            self.no_setup_label = 'No Setup'
        
        if self.indication_detected and not self.correction_phase:
            if self.indication_type == 'bullish' and current_price < self.indication_level:
                self.correction_phase = True
                self.market_structure = 'bullish_correction'
                self.bars_in_correction = 0
                self.correction_range_high = self.last_swing_high
                self.correction_range_low = self.last_swing_low
            elif self.indication_type == 'bearish' and current_price > self.indication_level:
                self.correction_phase = True
                self.market_structure = 'bearish_correction'
                self.bars_in_correction = 0
                self.correction_range_high = self.last_swing_high
                self.correction_range_low = self.last_swing_low
        
        if self.correction_phase:
            if self.last_swing_high is not None and self.last_swing_low is not None:
                self.correction_range_high = self.last_swing_high
                self.correction_range_low = self.last_swing_low
            self.bars_in_correction += 1
            self.bars_in_continuation = 0
        elif self.cont_active:
            self.bars_in_continuation += 1
            self.bars_in_correction = 0
        else:
            self.bars_in_correction = 0
            self.bars_in_continuation = 0
        
        is_consolidating, consol_high, consol_low = self.calculate_consolidation(
            df, swing_length, consolidation_bars, wiggle_pct
        )
        
        if is_consolidating and not self.indication_detected:
            self.market_structure = 'consolidation'
            self.indication_detected = False
            self.correction_phase = False
            self.indication_level = None
            self.indication_type = None
            self.last_entry_type = None
        
        atr_value = self.calculate_atr(df)
        atr_percent = atr_value / current_price if current_price > 0 else 0
        
        if swing_highs:
            self.update_resistance_zone(swing_highs[-1], high_indices[-1] if high_indices else len(df)-1, atr_percent)
        
        if swing_lows:
            self.update_support_zone(swing_lows[-1], low_indices[-1] if low_indices else len(df)-1, atr_percent)
        
        self.detect_entry_signals(df)
        
        return self.get_structure_summary()
    
    def update_resistance_zone(self, level, bar_index, atr_percent):
        buffer = self.get_zone_buffer(level, atr_percent)
        
        if len(self.resistance_levels) >= self.MAX_HIST_ZONES:
            self.resistance_levels.pop(0)
            self.resistance_start_bars.pop(0)
        
        self.resistance_levels.append(level)
        self.resistance_start_bars.append(bar_index)
    
    def update_support_zone(self, level, bar_index, atr_percent):
        buffer = self.get_zone_buffer(level, atr_percent)
        
        if len(self.support_levels) >= self.MAX_HIST_ZONES:
            self.support_levels.pop(0)
            self.support_start_bars.pop(0)
        
        self.support_levels.append(level)
        self.support_start_bars.append(bar_index)
    
    def detect_entry_signals(self, df):
        if not self.indication_detected or not self.correction_phase:
            return
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        
        if self.indication_type == 'bullish' and current_price > self.indication_level and prev_price <= self.indication_level:
            self.last_entry_type = 'bullish_traditional'
        elif self.indication_type == 'bearish' and current_price < self.indication_level and prev_price >= self.indication_level:
            self.last_entry_type = 'bearish_traditional'
        
        if (self.indication_type == 'bullish' and self.correction_range_high is not None and 
            current_price > self.correction_range_high and prev_price <= self.correction_range_high):
            self.last_entry_type = 'bullish_breakout'
        elif (self.indication_type == 'bearish' and self.correction_range_low is not None and 
              current_price < self.correction_range_low and prev_price >= self.correction_range_low):
            self.last_entry_type = 'bearish_breakout'
    
    def calculate_tp_sl(self, entry_price, atr_value, sl_atr_mult=1.0, tp1_r_mult=1.0, tp2_r_mult=2.0, 
                       use_structure_stop=True, structure_sl_factor=1.0, max_structure_atr_mult=2.0):
        
        if use_structure_stop and self.last_entry_type is not None:
            if 'bullish' in self.last_entry_type and self.last_swing_low is not None:
                raw_dist = abs(entry_price - self.last_swing_low)
                struct_dist = raw_dist * structure_sl_factor
                max_dist = atr_value * max_structure_atr_mult
                final_dist = min(struct_dist, max_dist)
                sl = entry_price - final_dist
            elif 'bearish' in self.last_entry_type and self.last_swing_high is not None:
                raw_dist = abs(self.last_swing_high - entry_price)
                struct_dist = raw_dist * structure_sl_factor
                max_dist = atr_value * max_structure_atr_mult
                final_dist = min(struct_dist, max_dist)
                sl = entry_price + final_dist
            else:
                sl = entry_price - atr_value * sl_atr_mult if 'bullish' in self.last_entry_type else entry_price + atr_value * sl_atr_mult
        else:
            sl = entry_price - atr_value * sl_atr_mult if 'bullish' in self.last_entry_type else entry_price + atr_value * sl_atr_mult
        
        risk = abs(entry_price - sl)
        
        if 'bullish' in self.last_entry_type:
            tp1 = entry_price + risk * tp1_r_mult
            tp2 = entry_price + risk * tp2_r_mult
        else:
            tp1 = entry_price - risk * tp1_r_mult
            tp2 = entry_price - risk * tp2_r_mult
        
        return sl, tp1, tp2
    
    def get_structure_summary(self):
        return {
            'market_structure': self.market_structure,
            'indication_type': self.indication_type,
            'indication_level': self.indication_level,
            'last_swing_high': self.last_swing_high,
            'last_swing_low': self.last_swing_low,
            'correction_phase': self.correction_phase,
            'correction_range_high': self.correction_range_high,
            'correction_range_low': self.correction_range_low,
            'bars_in_correction': self.bars_in_correction,
            'bars_in_continuation': self.bars_in_continuation,
            'last_entry_type': self.last_entry_type,
            'last_trend_dir': self.last_trend_dir,
            'no_setup_label': self.no_setup_label,
            'support_levels': self.support_levels[-5:] if self.support_levels else [],
            'resistance_levels': self.resistance_levels[-5:] if self.resistance_levels else [],
        }
    
    def get_phase_color(self):
        if 'bullish' in self.market_structure:
            return '#00ff88'
        elif 'bearish' in self.market_structure:
            return '#ff4444'
        elif self.market_structure == 'consolidation':
            return '#888888'
        else:
            return '#666666'

# ============================================================================
# ICC VISUALIZER
# ============================================================================
class ICCVisualizer:
    def __init__(self):
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#888888',
            'support': '#00ff88',
            'resistance': '#ff4444',
            'consolidation': '#4444ff',
            'entry': '#ffff00'
        }
    
    def create_icc_chart(self, df, icc_structure):
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price with ICC Structure', 'Volume')
        )
        
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=1, col=1
        )
        
        for level in icc_structure.get('support_levels', []):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color=self.colors['support'],
                opacity=0.5,
                row=1, col=1,
                annotation_text=f"S: {level:.2f}"
            )
        
        for level in icc_structure.get('resistance_levels', []):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color=self.colors['resistance'],
                opacity=0.5,
                row=1, col=1,
                annotation_text=f"R: {level:.2f}"
            )
        
        if icc_structure.get('indication_level'):
            fig.add_hline(
                y=icc_structure['indication_level'],
                line_dash="dot",
                line_color=self.colors[icc_structure.get('indication_type', 'neutral')],
                opacity=0.8,
                row=1, col=1,
                annotation_text=f"IND: {icc_structure['indication_level']:.2f}"
            )
        
        if (icc_structure.get('correction_range_high') and 
            icc_structure.get('correction_range_low')):
            
            fig.add_hrect(
                y0=icc_structure['correction_range_low'],
                y1=icc_structure['correction_range_high'],
                fillcolor=self.colors['consolidation'],
                opacity=0.2,
                line_width=0,
                row=1, col=1,
                annotation_text=f"Correction Range"
            )
        
        volume_colors = [self.colors['bullish'] if close >= open else self.colors['bearish'] 
                        for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"ICC Market Structure: {icc_structure.get('market_structure', 'No Setup')}",
            template='plotly_dark',
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig

# ============================================================================
# PIVOT INDICATOR CLASS (Original)
# ============================================================================
class PivotIndicator:
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
            st.warning(f"Error resampling data: {str(e)}")
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
                st.warning(f"Error calculating pivots: {str(e)}")
                continue
        
        return all_pivots

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def validate_credentials(access_token, client_id):
    issues = []
    
    clean_token = access_token.strip() if access_token else ""
    clean_client_id = client_id.strip() if client_id else ""
    
    if not clean_token:
        issues.append("Access token is empty")
    elif len(clean_token) < 50:
        issues.append("Access token seems too short")
    
    if not clean_client_id:
        issues.append("Client ID is empty")
    elif len(clean_client_id) < 5:
        issues.append("Client ID seems too short")
    
    return clean_token, clean_client_id, issues

def get_user_id():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]
    return st.session_state.user_id

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

def highlight_atm_row(row):
    if row['Zone'] == 'ATM':
        return ['background-color: #FFD700; font-weight: bold'] * len(row)
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

def create_candlestick_chart(df, title, interval, show_pivots=True, pivot_settings=None):
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

# ============================================================================
# ICC DISPLAY FUNCTIONS
# ============================================================================
def display_icc_analysis(df, timeframe_minutes=15):
    st.header("📊 ICC Market Structure Analysis")
    
    st.sidebar.header("🔄 ICC Market Structure Settings")
    
    profile = st.sidebar.selectbox(
        "ICC Profile",
        ['Entry', 'Scalping', 'Intraday', 'Swing'],
        index=2,
        help="Entry - lower timeframes. Scalping - faster alerts. Intraday - default. Swing - slower structure."
    )
    
    use_manual_settings = st.sidebar.checkbox("Use Manual Settings", value=False, key="icc_manual")
    
    manual_swing_length = 7
    manual_consolidation_bars = 20
    
    if use_manual_settings:
        manual_swing_length = st.sidebar.slider("Manual Swing Length", 3, 50, 7, key="icc_swing")
        manual_consolidation_bars = st.sidebar.slider("Manual Consolidation Bars", 3, 100, 20, key="icc_consol")
    
    show_icc_zones = st.sidebar.checkbox("Show ICC Zones", value=True, key="icc_zones")
    show_icc_signals = st.sidebar.checkbox("Show ICC Signals", value=True, key="icc_signals")
    
    icc = ICCMarketStructure(
        profile=profile,
        use_manual_settings=use_manual_settings,
        manual_swing_length=manual_swing_length,
        manual_consolidation_bars=manual_consolidation_bars
    )
    
    visualizer = ICCVisualizer()
    
    if not df.empty:
        icc_summary = icc.update_structure(df, timeframe_minutes)
        current_price = df['close'].iloc[-1]
        
        # ICC Dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        phase_color = icc.get_phase_color()
        
        with col1:
            st.markdown(f"""
            <div class="phase-box" style="border-left-color: {phase_color};">
                <h4 style="margin: 0; color: #ffffff;">Market Phase</h4>
                <h2 style="margin: 5px 0; color: {phase_color};">{icc_summary.get('market_structure', 'No Setup').replace('_', ' ').title()}</h2>
                <p style="margin: 0; color: #aaaaaa;">{icc_summary.get('no_setup_label', '')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ind_type = icc_summary.get('indication_type', 'None')
            ind_color = visualizer.colors.get(ind_type, '#888888')
            st.markdown(f"""
            <div class="phase-box" style="border-left-color: {ind_color};">
                <h4 style="margin: 0; color: #ffffff;">Indication</h4>
                <h3 style="margin: 5px 0; color: {ind_color};">{ind_type.title()}</h3>
                <p style="margin: 0; color: #aaaaaa;">Level: {icc_summary.get('indication_level', 'None'):.2f if icc_summary.get('indication_level') else 'None'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            trend_color = visualizer.colors.get(icc_summary.get('last_trend_dir', 'neutral'), '#888888')
            st.markdown(f"""
            <div class="phase-box" style="border-left-color: {trend_color};">
                <h4 style="margin: 0; color: #ffffff;">Trend Direction</h4>
                <h3 style="margin: 5px 0; color: {trend_color};">{icc_summary.get('last_trend_dir', 'Neutral').title()}</h3>
                <p style="margin: 0; color: #aaaaaa;">Correction: {icc_summary.get('bars_in_correction', 0)} bars</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            entry_type = icc_summary.get('last_entry_type', 'None')
            entry_color = visualizer.colors['entry'] if entry_type != 'None' else '#888888'
            st.markdown(f"""
            <div class="phase-box" style="border-left-color: {entry_color};">
                <h4 style="margin: 0; color: #ffffff;">Last Signal</h4>
                <h3 style="margin: 5px 0; color: {entry_color};">{entry_type.replace('_', ' ').title()}</h3>
                <p style="margin: 0; color: #aaaaaa;">Price: {current_price:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ICC Chart
        if show_icc_zones:
            fig_icc = visualizer.create_icc_chart(df, icc_summary)
            st.plotly_chart(fig_icc, use_container_width=True)
        
        # Detailed Analysis
        with st.expander("📋 ICC Detailed Analysis", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Market Structure**")
                st.info(f"""
                - Phase: {icc_summary['market_structure']}
                - Trend: {icc_summary['last_trend_dir'] or 'Neutral'}
                - Correction: {icc_summary['bars_in_correction']} bars
                - Continuation: {icc_summary['bars_in_continuation']} bars
                """)
            
            with col2:
                st.markdown("**Support & Resistance**")
                support_text = "\n".join([f"• {s:.2f}" for s in icc_summary['support_levels'][-3:]]) if icc_summary['support_levels'] else "None"
                resistance_text = "\n".join([f"• {r:.2f}" for r in icc_summary['resistance_levels'][-3:]]) if icc_summary['resistance_levels'] else "None"
                st.info(f"""
                **Support:**\n{support_text}
                **Resistance:**\n{resistance_text}
                """)
            
            with col3:
                st.markdown("**Indication & Signals**")
                st.info(f"""
                - Type: {icc_summary['indication_type'] or 'None'}
                - Level: {icc_summary['indication_level'] or 'None'}
                - Last Signal: {icc_summary['last_entry_type'] or 'None'}
                - Correction Range: {icc_summary['correction_range_low'] or 'None'} - {icc_summary['correction_range_high'] or 'None'}
                """)
        
        # ICC Trading Signals
        if show_icc_signals and icc_summary['last_entry_type']:
            st.subheader("🎯 ICC Trading Signals")
            
            signal_type = icc_summary['last_entry_type']
            signal_color = '#00ff88' if 'bullish' in signal_type else '#ff4444'
            
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 2px solid {signal_color};">
                <h3 style="color: {signal_color}; margin: 0;">Signal: {signal_type.replace('_', ' ').upper()}</h3>
                <p>Price: {current_price:.2f} | Timeframe: {timeframe_minutes}m</p>
                <p>Market Phase: {icc_summary['market_structure'].replace('_', ' ').title()}</p>
                <p>Trend: {icc_summary['last_trend_dir'] or 'Neutral'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'entry' in icc_summary['last_entry_type']:
                atr_value = icc.calculate_atr(df)
                sl, tp1, tp2 = icc.calculate_tp_sl(
                    current_price, atr_value,
                    sl_atr_mult=1.0,
                    tp1_r_mult=1.0,
                    tp2_r_mult=2.0
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stop Loss", f"{sl:.2f}", delta=f"{(sl - current_price):+.2f}")
                with col2:
                    st.metric("TP 1", f"{tp1:.2f}", delta=f"{(tp1 - current_price):+.2f}")
                with col3:
                    st.metric("TP 2", f"{tp2:.2f}", delta=f"{(tp2 - current_price):+.2f}")
    else:
        st.warning("No data available for ICC analysis")

# ============================================================================
# DHAN API FUNCTIONS
# ============================================================================
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

# ============================================================================
# OPTION CHAIN ANALYSIS
# ============================================================================
def analyze_option_chain(selected_expiry=None):
    now = datetime.now(timezone("Asia/Kolkata"))
    
    expiry_data = get_dhan_expiry_list_cached(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
    if not expiry_data or 'data' not in expiry_data:
        st.error("Failed to get expiry list from Dhan API")
        return None, None, []
    
    expiry_dates = expiry_data['data']
    if not expiry_dates:
        st.error("No expiry dates available")
        return None, None, []
    
    expiry = selected_expiry if selected_expiry else expiry_dates[0]

    option_chain_data = get_dhan_option_chain(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, expiry)
    if not option_chain_data or 'data' not in option_chain_data:
        st.error("Failed to get option chain from Dhan API")
        return None, None, expiry_dates
    
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
        'volume': 'totalTradedVolume'
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
    
    st.markdown("## Open Interest Change (in Lakhs)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CALL ΔOI", f"{total_ce_change:+.1f}L", delta_color="inverse")
    with col2:
        st.metric("PUT ΔOI", f"{total_pe_change:+.1f}L", delta_color="normal")

    weights = {
        "ChgOI_Bias": 2,
        "Volume_Bias": 1,
        "AskQty_Bias": 1,
        "BidQty_Bias": 1,
        "DVP_Bias": 1,
        "PressureBias": 1,
    }
    
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
            "ChgOI_Bias": "Bullish" if row.get('changeinOpenInterest_CE', 0) < row.get('changeinOpenInterest_PE', 0) else "Bearish",
            "Volume_Bias": "Bullish" if row.get('totalTradedVolume_CE', 0) < row.get('totalTradedVolume_PE', 0) else "Bearish",
            "AskQty_Bias": "Bullish" if row.get('askQty_PE', 0) > row.get('askQty_CE', 0) else "Bearish",
            "BidQty_Bias": "Bearish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bullish",
            "DVP_Bias": delta_volume_bias(
                row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
                row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
                row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
            ),
            "BidAskPressure": bid_ask_pressure,
            "PressureBias": pressure_bias
        }
        for k in row_data:
            if "_Bias" in k:
                bias = row_data[k]
                score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)
        row_data["BiasScore"] = score
        row_data["Verdict"] = final_verdict(score)
        bias_results.append(row_data)

    df_summary = pd.DataFrame(bias_results)
    df_summary = pd.merge(
        df_summary,
        df[['strikePrice', 'openInterest_CE', 'openInterest_PE']],
        left_on='Strike', right_on='strikePrice', how='left'
    )

    df_summary['PCR'] = df_summary['openInterest_PE'] / df_summary['openInterest_CE']
    df_summary['PCR'] = np.where(df_summary['openInterest_CE'] == 0, 0, df_summary['PCR'])
    df_summary['PCR'] = df_summary['PCR'].round(2)
    df_summary['PCR_Signal'] = np.where(
        df_summary['PCR'] > 1.2, "Bullish",
        np.where(df_summary['PCR'] < 0.7, "Bearish", "Neutral")
    )

    st.markdown("## Option Chain Bias Summary")
    
    styled_df = df_summary.style\
        .applymap(color_pcr, subset=['PCR'])\
        .applymap(color_pressure, subset=['BidAskPressure'])\
        .apply(highlight_atm_row, axis=1)
    
    st.dataframe(styled_df, use_container_width=True)
    
    csv_data = df_summary.to_csv(index=False)
    st.download_button(
        label="📥 Download Summary as CSV",
        data=csv_data,
        file_name=f"nifty_options_summary_{expiry}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

    return underlying, df_summary, expiry_dates

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    st.title("📈 Nifty Trading & ICC Analyzer")
    
    # Initialize Supabase
    try:
        if not supabase_url or not supabase_key:
            st.error("Please configure your Supabase credentials in Streamlit secrets")
            st.info("""
            Add to .streamlit/secrets.toml:
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
            Add to .streamlit/secrets.toml:
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
        
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            st.sidebar.success("Telegram notifications enabled")
        else:
            st.sidebar.warning("Telegram notifications disabled")
        
    except Exception as e:
        st.error(f"Credential validation error: {str(e)}")
        return
    
    # Get user preferences
    user_id = get_user_id()
    user_prefs = db.get_user_preferences(user_id)
    
    # ============================================================================
    # SIDEBAR CONFIGURATION
    # ============================================================================
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
    show_pivots = st.sidebar.checkbox("Show Pivot Levels", value=True)
    
    if show_pivots:
        st.sidebar.subheader("Toggle Individual Pivot Levels")
        
        if 'pivot_settings' not in user_prefs:
            user_prefs['pivot_settings'] = {
                'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True
            }
        
        show_3m = st.sidebar.checkbox("3 Minute Pivots", value=user_prefs['pivot_settings'].get('show_3m', True))
        show_5m = st.sidebar.checkbox("5 Minute Pivots", value=user_prefs['pivot_settings'].get('show_5m', True))
        show_10m = st.sidebar.checkbox("10 Minute Pivots", value=user_prefs['pivot_settings'].get('show_10m', True))
        show_15m = st.sidebar.checkbox("15 Minute Pivots", value=user_prefs['pivot_settings'].get('show_15m', True))
        
        pivot_settings = {
            'show_3m': show_3m,
            'show_5m': show_5m,
            'show_10m': show_10m,
            'show_15m': show_15m
        }
    else:
        pivot_settings = {
            'show_3m': False, 'show_5m': False, 'show_10m': False, 'show_15m': False
        }
    
    # Trading signal settings
    st.sidebar.header("🔔 Trading Signals")
    enable_signals = st.sidebar.checkbox("Enable Telegram Signals", value=True)
    pivot_proximity = st.sidebar.slider(
        "Pivot Proximity (± Points)", 
        min_value=1, 
        max_value=20, 
        value=user_prefs.get('pivot_proximity', 5)
    )
    
    # Options expiry selection
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
    
    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh (2 min)", value=user_prefs['auto_refresh'])
    days_back = st.sidebar.slider("Days of Historical Data", 1, 5, user_prefs['days_back'])
    use_cache = st.sidebar.checkbox("Use Cached Data", value=True)
    
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
            test_msg = "🔔 Nifty Analyzer - Test message successful! ✅"
            send_telegram_message_sync(test_msg)
            st.sidebar.success("Test message sent!")
        else:
            st.sidebar.error(message)
    
    # Save preferences
    if st.sidebar.button("💾 Save Preferences"):
        db.save_user_preferences(user_id, interval, auto_refresh, days_back, pivot_settings, pivot_proximity)
        st.sidebar.success("Preferences saved!")
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Show debug info
    st.sidebar.subheader("🔧 Debug Info")
    st.sidebar.write(f"Telegram Bot Token: {'✅ Set' if TELEGRAM_BOT_TOKEN else '❌ Missing'}")
    st.sidebar.write(f"Telegram Chat ID: {'✅ Set' if TELEGRAM_CHAT_ID else '❌ Missing'}")
    
    # ============================================================================
    # MAIN CONTENT
    # ============================================================================
    api = DhanAPI(access_token, client_id)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Trading Chart")
        
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
        
        ltp_data = api.get_ltp_data("13", "IDX_I")
        if ltp_data and 'data' in ltp_data:
            for exchange, data in ltp_data['data'].items():
                for security_id, price_data in data.items():
                    current_price = price_data.get('last_price', 0)
                    break
        
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
        if not df.empty:
            display_metrics(ltp_data, df, db)
        
        if not df.empty:
            fig = create_candlestick_chart(
                df, 
                f"Nifty 50 - {selected_timeframe} Chart {'with Pivot Levels' if show_pivots else ''}", 
                interval,
                show_pivots=show_pivots,
                pivot_settings=pivot_settings
            )
            st.plotly_chart(fig, use_container_width=True)
            
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
        else:
            st.error("No data available. Please check your API credentials and try again.")
    
    with col2:
        st.header("📊 Options Analysis")
        underlying_price, df_summary, available_expiries = analyze_option_chain(selected_expiry)
        
        if underlying_price:
            st.info(f"**NIFTY SPOT:** {underlying_price:.2f}")
    
    # ============================================================================
    # ICC ANALYSIS SECTION
    # ============================================================================
    if not df.empty:
        timeframe_minutes_map = {'1': 1, '3': 3, '5': 5, '10': 10, '15': 15}
        timeframe_minutes = timeframe_minutes_map.get(interval, 5)
        display_icc_analysis(df, timeframe_minutes)
    
    # Show current time
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    st.sidebar.info(f"Last Updated: {current_time}")

# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    main()
