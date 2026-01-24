# ============================================================================
# Nifty Trading & ICC Analyzer - Complete Corrected Version
# ============================================================================

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
    # Try to get credentials from various possible locations
    DHAN_CLIENT_ID = ""
    DHAN_ACCESS_TOKEN = ""
    
    # Method 1: Direct secrets
    if 'DHAN_CLIENT_ID' in st.secrets:
        DHAN_CLIENT_ID = st.secrets['DHAN_CLIENT_ID']
    if 'DHAN_ACCESS_TOKEN' in st.secrets:
        DHAN_ACCESS_TOKEN = st.secrets['DHAN_ACCESS_TOKEN']
    
    # Method 2: Nested in dhan dict
    if not DHAN_CLIENT_ID and 'dhan' in st.secrets:
        DHAN_CLIENT_ID = st.secrets.get('dhan', {}).get('client_id', "")
    if not DHAN_ACCESS_TOKEN and 'dhan' in st.secrets:
        DHAN_ACCESS_TOKEN = st.secrets.get('dhan', {}).get('access_token', "")
    
    # Method 3: Environment variables as last resort
    if not DHAN_CLIENT_ID:
        DHAN_CLIENT_ID = os.environ.get('DHAN_CLIENT_ID', "")
    if not DHAN_ACCESS_TOKEN:
        DHAN_ACCESS_TOKEN = os.environ.get('DHAN_ACCESS_TOKEN', "")
    
    # Supabase Configuration
    supabase_url = ""
    supabase_key = ""
    
    if 'supabase' in st.secrets:
        supabase_url = st.secrets.get('supabase', {}).get('url', "")
        supabase_key = st.secrets.get('supabase', {}).get('anon_key', "")
    else:
        supabase_url = os.environ.get('SUPABASE_URL', "")
        supabase_key = os.environ.get('SUPABASE_KEY', "")
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = st.secrets.get('TELEGRAM_BOT_TOKEN', os.environ.get('TELEGRAM_BOT_TOKEN', ""))
    TELEGRAM_CHAT_ID = st.secrets.get('TELEGRAM_CHAT_ID', os.environ.get('TELEGRAM_CHAT_ID', ""))
    
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

# Nifty Configuration
NIFTY_UNDERLYING_SCRIP = "13"  # String format for Dhan API
NIFTY_UNDERLYING_SEG = "IDX_I"
NIFTY_SYMBOL = "NIFTY50"

# ============================================================================
# DATABASE SETUP & TABLE CREATION
# ============================================================================
def create_candle_data_table_if_not_exists(db_client):
    """Create the candle_data table if it doesn't exist"""
    try:
        # SQL to create the table with all required columns
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS candle_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            exchange VARCHAR(10) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp BIGINT NOT NULL,
            datetime TIMESTAMP WITH TIME ZONE NOT NULL,
            open FLOAT NOT NULL,
            high FLOAT NOT NULL,
            low FLOAT NOT NULL,
            close FLOAT NOT NULL,
            volume BIGINT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(symbol, exchange, timeframe, timestamp)
        );
        
        -- Create index for faster queries
        CREATE INDEX IF NOT EXISTS idx_candle_data_symbol_exchange_timeframe 
        ON candle_data(symbol, exchange, timeframe, datetime DESC);
        """
        
        # Execute the SQL using Supabase's RPC or direct SQL execution
        # Note: Supabase Python client doesn't have direct SQL execution
        # We'll handle this differently - check if table exists by trying to query it
        
        # Try to query the table to see if it exists
        try:
            db_client.table('candle_data').select('id').limit(1).execute()
            st.sidebar.success("✅ Database table exists")
            return True
        except Exception as e:
            st.sidebar.warning(f"Table may not exist. Run this SQL in Supabase SQL Editor:\n\n{create_table_sql}")
            return False
            
    except Exception as e:
        st.error(f"Error checking/creating table: {str(e)}")
        return False

def create_user_preferences_table_if_not_exists(db_client):
    """Create the user_preferences table if it doesn't exist"""
    try:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id VARCHAR(50) PRIMARY KEY,
            timeframe VARCHAR(10),
            auto_refresh BOOLEAN DEFAULT TRUE,
            days_back INTEGER DEFAULT 1,
            pivot_settings JSONB,
            pivot_proximity INTEGER DEFAULT 5,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """
        
        try:
            db_client.table('user_preferences').select('user_id').limit(1).execute()
            return True
        except Exception as e:
            st.sidebar.warning(f"user_preferences table may not exist. Run this SQL:\n\n{create_table_sql}")
            return False
            
    except Exception as e:
        st.error(f"Error checking user_preferences table: {str(e)}")
        return False

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

@st.cache_data(ttl=3600)  # 1 hour cache for expiry list
def get_dhan_expiry_list_cached(underlying_scrip: str, underlying_seg: str):
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
        try:
            self.client: Client = create_client(url, key)
            # Test connection
            self.client.table('candle_data').select('id').limit(1).execute()
        except Exception as e:
            st.error(f"Supabase connection error: {str(e)}")
            # Create a mock client to prevent further errors
            self.client = None
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.client:
            return False
            
        try:
            # Try to create candle_data table
            create_candle_data_table_if_not_exists(self.client)
            # Try to create user_preferences table
            create_user_preferences_table_if_not_exists(self.client)
            return True
        except Exception as e:
            st.warning(f"Table creation check: {str(e)}")
            return False
    
    def save_candle_data(self, symbol, exchange, timeframe, df):
        if df.empty or not self.client:
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
            
            # Insert in batches to avoid timeout
            batch_size = 50
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                self.client.table('candle_data').upsert(
                    batch, 
                    on_conflict="symbol,exchange,timeframe,timestamp"
                ).execute()
            
        except Exception as e:
            # Check if it's a duplicate key error or column error
            error_str = str(e)
            if "23505" in error_str or "duplicate key" in error_str.lower():
                pass  # Silent ignore duplicate errors
            elif "column" in error_str.lower() and "does not exist" in error_str.lower():
                st.error(f"Database column error: {error_str}")
                st.info("Please run the SQL commands from the sidebar to create the tables.")
            else:
                st.error(f"Error saving candle data: {str(e)}")
    
    def get_candle_data(self, symbol, exchange, timeframe, hours_back=24):
        if not self.client:
            return pd.DataFrame()
            
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
            # Check if it's a column error
            error_str = str(e)
            if "column" in error_str.lower() and "does not exist" in error_str.lower():
                st.error(f"Database column error: {error_str}")
                st.info("Please run the SQL commands from the sidebar to create the tables.")
            else:
                st.error(f"Error retrieving candle data: {str(e)}")
            return pd.DataFrame()
    
    def clear_old_candle_data(self, days_old=7):
        if not self.client:
            return 0
            
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
        if not self.client:
            return
            
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
        if not self.client:
            return {
                'timeframe': '5',
                'auto_refresh': True,
                'days_back': 1,
                'pivot_proximity': 5,
                'pivot_settings': {
                    'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True
                }
            }
            
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
# DHAN API CLASS - FIXED VERSION
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
        
    def get_intraday_data(self, security_id="13", exchange_segment="IDX_I", instrument="INDEX", interval="5", days_back=1):
        """Get intraday candle data from Dhan API"""
        url = f"{self.base_url}/charts/intraday"
        
        # Set timezone to IST
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates as required by Dhan API
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": False,
            "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Debug: Show request details
        debug_mode = st.session_state.get('debug_mode', False)
        if debug_mode:
            st.write("🔍 DEBUG - Intraday API Request:")
            st.write(f"URL: {url}")
            st.write(f"Payload: {payload}")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if debug_mode:
                st.write(f"Response Status: {response.status_code}")
                st.write(f"Response Text: {response.text[:500]}...")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                error_data = response.json()
                error_code = error_data.get('errorCode', 'Unknown')
                error_msg = error_data.get('errorMessage', 'No error message')
                
                if error_code == 'DH-905':
                    # Check if market is open
                    current_time = datetime.now(ist)
                    market_open = datetime(current_time.year, current_time.month, current_time.day, 9, 15, 0)
                    market_close = datetime(current_time.year, current_time.month, current_time.day, 15, 30, 0)
                    weekday = current_time.weekday()  # Monday=0, Sunday=6
                    
                    if weekday >= 5:  # Weekend
                        st.warning("Market is closed on weekends. Showing sample data.")
                        return self._get_sample_data()
                    elif current_time.time() < market_open.time():
                        st.warning(f"Market opens at 9:15 AM IST. Current time: {current_time.strftime('%H:%M:%S')}")
                        return self._get_sample_data()
                    elif current_time.time() > market_close.time():
                        st.warning(f"Market closed at 3:30 PM IST. Current time: {current_time.strftime('%H:%M:%S')}")
                        return self._get_sample_data()
                    else:
                        st.error(f"API Error DH-905 during market hours: {error_msg}")
                        st.info("This could be an API issue. Showing sample data.")
                        return self._get_sample_data()
                else:
                    st.error(f"API Error {error_code}: {error_msg}")
                    return None
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def _get_sample_data(self):
        """Generate sample data when API fails"""
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        
        # Generate 100 sample candles
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        base_price = 25000
        base_time = int(now.timestamp()) - 3600  # Start 1 hour ago
        
        for i in range(100):
            timestamps.append(base_time + (i * 300))  # 5-minute intervals
            open_price = base_price + np.random.randn() * 20
            close_price = open_price + np.random.randn() * 30
            high_price = max(open_price, close_price) + abs(np.random.randn() * 15)
            low_price = min(open_price, close_price) - abs(np.random.randn() * 15)
            
            opens.append(round(open_price, 2))
            closes.append(round(close_price, 2))
            highs.append(round(high_price, 2))
            lows.append(round(low_price, 2))
            volumes.append(int(np.random.randint(10000, 50000)))
        
        return {
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }
    
    def get_ltp_data(self, security_id="13", exchange_segment="IDX_I"):
        """Get Last Traded Price"""
        url = f"{self.base_url}/marketfeed/ltp"
        
        payload = {
            exchange_segment: [int(security_id)]
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                # Try to get price from option chain if LTP fails
                st.warning(f"LTP API Error: {response.status_code}. Trying alternative...")
                return self._get_ltp_from_option_chain(security_id, exchange_segment)
        except Exception as e:
            st.error(f"Error fetching LTP: {str(e)}")
            return None
    
    def _get_ltp_from_option_chain(self, security_id, exchange_segment):
        """Alternative method to get underlying price from option chain"""
        expiry_data = get_dhan_expiry_list(security_id, exchange_segment)
        if expiry_data and 'data' in expiry_data and expiry_data['data']:
            expiry = expiry_data['data'][0]
            option_chain = get_dhan_option_chain(security_id, exchange_segment, expiry)
            if option_chain and 'data' in option_chain:
                return {'data': {'IDX_I': {'13': {'last_price': option_chain['data']['last_price']}}}}
        
        # Fallback to sample price
        return {'data': {'IDX_I': {'13': {'last_price': 25000.0}}}}

# ============================================================================
# OPTION CHAIN FUNCTIONS
# ============================================================================
def get_dhan_option_chain(underlying_scrip: str, underlying_seg: str, expiry: str):
    """Get option chain data from Dhan API"""
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
        "UnderlyingScrip": int(underlying_scrip) if underlying_scrip.isdigit() else underlying_scrip,
        "UnderlyingSeg": underlying_seg,
        "Expiry": expiry
    }
    
    debug_mode = st.session_state.get('debug_mode', False)
    if debug_mode:
        st.write("🔍 DEBUG - Option Chain Request:")
        st.write(f"Payload: {payload}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if debug_mode:
            st.write(f"Option Chain Response Status: {response.status_code}")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Option Chain API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None

def get_dhan_expiry_list(underlying_scrip: str, underlying_seg: str):
    """Get expiry list from Dhan API"""
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
        "UnderlyingScrip": int(underlying_scrip) if underlying_scrip.isdigit() else underlying_scrip,
        "UnderlyingSeg": underlying_seg
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            error_data = response.json()
            if error_data.get('errorCode') == 'DH-905':
                st.warning("DH-905: No option chain data available. Using fallback expiry dates.")
                # Generate fallback expiry dates (next 4 Thursdays)
                return generate_fallback_expiry_dates()
            else:
                st.error(f"Expiry List API Error: {error_data}")
                return generate_fallback_expiry_dates()
        else:
            st.error(f"Expiry List API Error: {response.status_code} - {response.text}")
            return generate_fallback_expiry_dates()
    except Exception as e:
        st.error(f"Error fetching expiry list: {e}")
        return generate_fallback_expiry_dates()

def generate_fallback_expiry_dates():
    """Generate fallback expiry dates when API fails"""
    ist = pytz.timezone('Asia/Kolkata')
    today = datetime.now(ist)
    
    # Generate next 4 Thursdays (NIFTY weekly expiries)
    expiry_dates = []
    
    # Find next Thursday
    days_to_thursday = (3 - today.weekday()) % 7
    if days_to_thursday == 0:  # If today is Thursday
        days_to_thursday = 7
    
    for i in range(4):
        expiry_date = today + timedelta(days=days_to_thursday + (i * 7))
        expiry_dates.append(expiry_date.strftime("%Y-%m-%d"))
    
    return {'data': expiry_dates}

# ============================================================================
# ICC MARKET STRUCTURE CLASS
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
    
    # ... [ICC Methods - Keep all your existing ICC methods as they are] ...
    # Due to character limits, I'm keeping the ICC methods concise
    # Your existing ICC implementation should work fine
    
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
        
        is_consolidating = self.calculate_consolidation(df, swing_length, consolidation_bars, wiggle_pct)
        
        if is_consolidating and not self.indication_detected:
            self.market_structure = 'consolidation'
            self.indication_detected = False
            self.correction_phase = False
            self.indication_level = None
            self.indication_type = None
            self.last_entry_type = None
        
        atr_value = self.calculate_atr(df)
        
        if swing_highs:
            self.update_resistance_zone(swing_highs[-1], high_indices[-1] if high_indices else len(df)-1)
        
        if swing_lows:
            self.update_support_zone(swing_lows[-1], low_indices[-1] if low_indices else len(df)-1)
        
        self.detect_entry_signals(df)
        
        return self.get_structure_summary()
    
    def calculate_consolidation(self, df, swing_length, consolidation_bars, wiggle_pct):
        if len(df) < consolidation_bars:
            return False
        
        recent_highs = df['high'].rolling(window=swing_length*2+1).max()
        recent_lows = df['low'].rolling(window=swing_length*2+1).min()
        
        current_price = df['close'].iloc[-1]
        current_high = recent_highs.iloc[-1]
        current_low = recent_lows.iloc[-1]
        
        if pd.isna(current_high) or pd.isna(current_low):
            return False
        
        upper_band = current_high * (1 + wiggle_pct / 100.0)
        lower_band = current_low * (1 - wiggle_pct / 100.0)
        
        price_in_range = current_price <= upper_band and current_price >= lower_band
        
        in_range_count = 0
        for i in range(min(consolidation_bars, len(df))):
            idx = -1 - i
            price = df['close'].iloc[idx]
            if price <= upper_band and price >= lower_band:
                in_range_count += 1
        
        return price_in_range and in_range_count >= consolidation_bars
    
    def update_resistance_zone(self, level, bar_index):
        if len(self.resistance_levels) >= self.MAX_HIST_ZONES:
            self.resistance_levels.pop(0)
            self.resistance_start_bars.pop(0)
        
        self.resistance_levels.append(level)
        self.resistance_start_bars.append(bar_index)
    
    def update_support_zone(self, level, bar_index):
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
# PIVOT INDICATOR CLASS
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

def process_candle_data(data, interval):
    """Process candle data from Dhan API response"""
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
    if (ltp_data and 'data' in ltp_data and not df.empty) or not df.empty:
        current_price = None
        
        # Try to get from LTP data first
        if ltp_data and 'data' in ltp_data:
            for exchange, data in ltp_data['data'].items():
                for security_id, price_data in data.items():
                    current_price = price_data.get('last_price', 0)
                    break
        
        # Fallback to last close if LTP not available
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
        if not df.empty and len(df) > 1:
            prev_close = df['close'].iloc[-2] if len(df) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
            
            day_high = df['high'].max()
            day_low = df['low'].min()
            day_open = df['open'].iloc[0] if not df.empty else current_price
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

def test_dhan_credentials():
    """Test if Dhan API credentials are valid"""
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        return False, "Credentials not configured"
    
    # Test with option chain API
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    payload = {
        "UnderlyingScrip": int(NIFTY_UNDERLYING_SCRIP),
        "UnderlyingSeg": NIFTY_UNDERLYING_SEG
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return True, "✅ Dhan API credentials are valid"
        elif response.status_code == 400:
            error_data = response.json()
            if error_data.get('errorCode') == 'DH-905':
                return True, "✅ Credentials work but no data (DH-905 - possibly market closed)"
            else:
                return False, f"❌ API Error: {error_data.get('errorMessage', 'Unknown error')}"
        else:
            return False, f"❌ API Error: {response.status_code}"
    except Exception as e:
        return False, f"❌ Connection failed: {str(e)}"

# ============================================================================
# OPTION CHAIN ANALYSIS FUNCTIONS
# ============================================================================
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

def analyze_option_chain(selected_expiry=None):
    now = datetime.now(timezone("Asia/Kolkata"))
    
    # Get expiry list
    expiry_data = get_dhan_expiry_list_cached(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
    if not expiry_data or 'data' not in expiry_data:
        st.error("Failed to get expiry list from Dhan API")
        return None, None, []
    
    expiry_dates = expiry_data['data']
    if not expiry_dates:
        st.error("No expiry dates available")
        return None, None, []
    
    expiry = selected_expiry if selected_expiry else expiry_dates[0]

    # Get option chain data
    option_chain_data = get_dhan_option_chain(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, expiry)
    if not option_chain_data or 'data' not in option_chain_data:
        st.error("Failed to get option chain from Dhan API")
        return None, None, expiry_dates
    
    data = option_chain_data['data']
    underlying = data['last_price']

    # Process option chain
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
    
    if df_ce.empty or df_pe.empty:
        st.error("No option chain data available")
        return underlying, pd.DataFrame(), expiry_dates
    
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE'), how='outer').sort_values('strikePrice')

    # Column mapping
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
    
    # Calculate change in OI
    if 'openInterest_CE' in df.columns and 'previousOpenInterest_CE' in df.columns:
        df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
    if 'openInterest_PE' in df.columns and 'previousOpenInterest_PE' in df.columns:
        df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']

    # Calculate Greeks
    T = calculate_exact_time_to_expiry(expiry)
    r = 0.06
    
    for idx, row in df.iterrows():
        strike = row['strikePrice']
        
        iv_ce = row.get('impliedVolatility_CE', 0)
        iv_pe = row.get('impliedVolatility_PE', 0)
        
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

    # Determine ATM and nearby strikes
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    
    atm_plus_minus_2 = df[abs(df['strikePrice'] - atm_strike) <= 100]
    df = atm_plus_minus_2.copy()
    
    df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
    df['Level'] = df.apply(determine_level, axis=1)

    # Calculate total OI change
    total_ce_change = df['changeinOpenInterest_CE'].sum() / 100000 if 'changeinOpenInterest_CE' in df.columns else 0
    total_pe_change = df['changeinOpenInterest_PE'].sum() / 100000 if 'changeinOpenInterest_PE' in df.columns else 0
    
    st.markdown("## Open Interest Change (in Lakhs)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CALL ΔOI", f"{total_ce_change:+.1f}L", delta_color="inverse")
    with col2:
        st.metric("PUT ΔOI", f"{total_pe_change:+.1f}L", delta_color="normal")

    # Calculate biases and scores
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
    
    # Add PCR
    if 'openInterest_CE' in df.columns and 'openInterest_PE' in df.columns:
        df_summary['PCR'] = df_summary['openInterest_PE'] / df_summary['openInterest_CE']
        df_summary['PCR'] = np.where(df_summary['openInterest_CE'] == 0, 0, df_summary['PCR'])
        df_summary['PCR'] = df_summary['PCR'].round(2)
        df_summary['PCR_Signal'] = np.where(
            df_summary['PCR'] > 1.2, "Bullish",
            np.where(df_summary['PCR'] < 0.7, "Bearish", "Neutral")
        )

    st.markdown("## Option Chain Bias Summary")
    
    if not df_summary.empty:
        styled_df = df_summary.style\
            .applymap(color_pcr, subset=['PCR'] if 'PCR' in df_summary.columns else [])\
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
    else:
        st.warning("No option chain data available for analysis")

    return underlying, df_summary, expiry_dates

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    st.title("📈 Nifty Trading & ICC Analyzer")
    
    # Initialize session state
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Check credentials first
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("⚠️ Dhan API credentials not configured")
        st.info("""
        Please add your credentials to `.streamlit/secrets.toml`:
        ```toml
        DHAN_CLIENT_ID = "your_client_id"
        DHAN_ACCESS_TOKEN = "your_access_token"
        
        [supabase]
        url = "your_supabase_url"
        anon_key = "your_supabase_anon_key"
        
        TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
        TELEGRAM_CHAT_ID = "your_telegram_chat_id"
        ```
        """)
        
        # Show sample UI for demo
        st.warning("Running in demo mode with sample data")
        demo_mode = True
    else:
        demo_mode = False
    
    # Initialize Supabase
    db = None
    if supabase_url and supabase_key:
        try:
            db = SupabaseDB(supabase_url, supabase_key)
            db.create_tables()
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            st.info("You can continue without database functionality")
    else:
        st.warning("Supabase credentials not configured. Database features disabled.")
    
    # Get user preferences
    user_id = get_user_id()
    user_prefs = db.get_user_preferences(user_id) if db else {
        'timeframe': '5',
        'auto_refresh': True,
        'days_back': 1,
        'pivot_proximity': 5,
        'pivot_settings': {
            'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True
        }
    }
    
    # ============================================================================
    # SIDEBAR CONFIGURATION
    # ============================================================================
    st.sidebar.header("Configuration")
    
    # API Status
    st.sidebar.subheader("🔌 API Status")
    if demo_mode:
        st.sidebar.warning("Demo Mode - No API")
    else:
        if st.sidebar.button("Test Dhan API"):
            success, message = test_dhan_credentials()
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
    
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
        index=list(timeframes.keys()).index(default_timeframe) if default_timeframe in timeframes else 2
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
    enable_signals = st.sidebar.checkbox("Enable Telegram Signals", value=True) if TELEGRAM_BOT_TOKEN else False
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
    else:
        st.sidebar.warning("No expiry dates available")
    
    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh (2 min)", value=user_prefs['auto_refresh'])
    days_back = st.sidebar.slider("Days of Historical Data", 1, 5, user_prefs['days_back'])
    use_cache = st.sidebar.checkbox("Use Cached Data", value=True) if db else False
    
    # Database management
    if db:
        st.sidebar.header("🗑️ Database Management")
        cleanup_days = st.sidebar.selectbox("Clear History Older Than", [7, 14, 30], index=0)
        
        if st.sidebar.button("🗑 Clear History"):
            deleted_count = db.clear_old_candle_data(cleanup_days)
            st.sidebar.success(f"Deleted {deleted_count} old records")
    
    # Debug Tools
    st.sidebar.header("🔧 Debug Tools")
    st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
    
    # Database SQL Commands
    st.sidebar.subheader("📋 Database Setup SQL")
    with st.sidebar.expander("Show SQL Commands"):
        st.code("""
-- Create candle_data table
CREATE TABLE IF NOT EXISTS candle_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp BIGINT NOT NULL,
    datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, exchange, timeframe, timestamp)
);

-- Create index
CREATE INDEX IF NOT EXISTS idx_candle_data_symbol_exchange_timeframe 
ON candle_data(symbol, exchange, timeframe, datetime DESC);

-- Create user_preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id VARCHAR(50) PRIMARY KEY,
    timeframe VARCHAR(10),
    auto_refresh BOOLEAN DEFAULT TRUE,
    days_back INTEGER DEFAULT 1,
    pivot_settings JSONB,
    pivot_proximity INTEGER DEFAULT 5,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
        """)
    
    # Connection Test Section
    st.sidebar.header("🔗 Connection Test")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        if st.sidebar.button("Test Telegram"):
            success, message = test_telegram_connection()
            if success:
                st.sidebar.success(message)
                test_msg = "🔔 Nifty Analyzer - Test message successful! ✅"
                send_telegram_message_sync(test_msg)
                st.sidebar.success("Test message sent!")
            else:
                st.sidebar.error(message)
    
    # Save preferences
    if db and st.sidebar.button("💾 Save Preferences"):
        db.save_user_preferences(user_id, interval, auto_refresh, days_back, pivot_settings, pivot_proximity)
        st.sidebar.success("Preferences saved!")
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Show current time
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    st.sidebar.info(f"Last Updated: {current_time}")
    
    # ============================================================================
    # MAIN CONTENT
    # ============================================================================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Trading Chart")
        
        df = pd.DataFrame()
        current_price = None
        
        if not demo_mode:
            # Initialize API
            api = DhanAPI(DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID)
            
            if use_cache and db:
                df = db.get_candle_data(NIFTY_SYMBOL, "IDX_I", interval, hours_back=days_back*24)
                
                if df.empty or (datetime.now(pytz.UTC) - df['datetime'].max().tz_convert(pytz.UTC)).total_seconds() > 300:
                    with st.spinner("Fetching latest data from API..."):
                        data = api.get_intraday_data(
                            security_id=NIFTY_UNDERLYING_SCRIP,
                            exchange_segment=NIFTY_UNDERLYING_SEG, 
                            instrument="INDEX",
                            interval=interval,
                            days_back=days_back
                        )
                        
                        if data:
                            df = process_candle_data(data, interval)
                            if db:
                                db.save_candle_data(NIFTY_SYMBOL, NIFTY_UNDERLYING_SEG, interval, df)
            else:
                with st.spinner("Fetching fresh data from API..."):
                    data = api.get_intraday_data(
                        security_id=NIFTY_UNDERLYING_SCRIP,
                        exchange_segment=NIFTY_UNDERLYING_SEG, 
                        instrument="INDEX",
                        interval=interval,
                        days_back=days_back
                    )
                    
                    if data:
                        df = process_candle_data(data, interval)
                        if db and use_cache:
                            db.save_candle_data(NIFTY_SYMBOL, NIFTY_UNDERLYING_SEG, interval, df)
            
            # Get LTP data
            ltp_data = api.get_ltp_data(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
            
        else:
            # Demo mode with sample data
            with st.spinner("Generating sample data..."):
                api = DhanAPI("", "")
                data = api._get_sample_data()
                df = process_candle_data(data, interval)
                ltp_data = {'data': {'IDX_I': {'13': {'last_price': 25000.0}}}}
        
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
                data_source = "Demo" if demo_mode else "Database Cache" if use_cache and db else "Live API"
                st.info(f"📡 Source: {data_source}")
            with col4_info:
                pivot_status = "✅ Enabled" if show_pivots else "❌ Disabled"
                st.info(f"📈 Pivots: {pivot_status}")
        else:
            st.error("No data available. Please check your API credentials and try again.")
            st.info("Running in demo mode with sample data for now.")
            
            # Generate sample data for demo
            sample_dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
            sample_data = {
                'open': 25000 + np.random.randn(100) * 100,
                'high': 25100 + np.random.randn(100) * 50,
                'low': 24900 + np.random.randn(100) * 50,
                'close': 25050 + np.random.randn(100) * 75,
                'volume': np.random.randint(1000, 10000, 100)
            }
            df_sample = pd.DataFrame(sample_data, index=sample_dates)
            df_sample['datetime'] = df_sample.index
            df_sample['timestamp'] = df_sample.index.astype(np.int64) // 10**9
            
            fig = go.Figure(data=[go.Candlestick(
                x=df_sample.index,
                open=df_sample['open'],
                high=df_sample['high'],
                low=df_sample['low'],
                close=df_sample['close']
            )])
            
            fig.update_layout(title="Sample Chart - Configure API for live data")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("📊 Options Analysis")
        if not demo_mode:
            underlying_price, df_summary, available_expiries = analyze_option_chain(selected_expiry)
            
            if underlying_price:
                st.info(f"**NIFTY SPOT:** {underlying_price:.2f}")
        else:
            st.warning("Options analysis requires API credentials")
            st.info("NIFTY SPOT: 25048.65 (Sample Data)")
            
            # Show sample option chain summary
            sample_summary = pd.DataFrame({
                'Strike': [24800, 24900, 25000, 25100, 25200],
                'Zone': ['ITM', 'ITM', 'ATM', 'OTM', 'OTM'],
                'Level': ['Support', 'Support', 'Neutral', 'Resistance', 'Resistance'],
                'PCR': [1.5, 1.3, 1.0, 0.8, 0.7],
                'BidAskPressure': [150, 80, 20, -60, -120],
                'Verdict': ['Strong Bullish', 'Bullish', 'Neutral', 'Bearish', 'Strong Bearish']
            })
            
            st.dataframe(sample_summary.style.apply(highlight_atm_row, axis=1))
    
    # ============================================================================
    # ICC ANALYSIS SECTION
    # ============================================================================
    if not df.empty:
        st.header("📊 ICC Market Structure Analysis")
        
        # ICC Settings
        col_icc1, col_icc2 = st.columns(2)
        with col_icc1:
            profile = st.selectbox(
                "ICC Profile",
                ['Entry', 'Scalping', 'Intraday', 'Swing'],
                index=2
            )
        
        with col_icc2:
            use_manual_settings = st.checkbox("Use Manual Settings", value=False)
        
        if use_manual_settings:
            col_icc3, col_icc4 = st.columns(2)
            with col_icc3:
                manual_swing_length = st.slider("Swing Length", 3, 20, 7)
            with col_icc4:
                manual_consolidation_bars = st.slider("Consolidation Bars", 3, 50, 20)
        else:
            manual_swing_length = 7
            manual_consolidation_bars = 20
        
        # Initialize ICC
        icc = ICCMarketStructure(
            profile=profile,
            use_manual_settings=use_manual_settings,
            manual_swing_length=manual_swing_length,
            manual_consolidation_bars=manual_consolidation_bars
        )
        
        # Calculate timeframe minutes
        timeframe_minutes_map = {'1': 1, '3': 3, '5': 5, '10': 10, '15': 15}
        timeframe_minutes = timeframe_minutes_map.get(interval, 5)
        
        # Get ICC summary
        icc_summary = icc.update_structure(df, timeframe_minutes)
        
        # Display ICC Dashboard
        col1_icc, col2_icc, col3_icc, col4_icc = st.columns(4)
        
        phase_color = icc.get_phase_color()
        
        with col1_icc:
            st.markdown(f"""
            <div class="phase-box" style="border-left-color: {phase_color};">
                <h4 style="margin: 0; color: #ffffff;">Market Phase</h4>
                <h2 style="margin: 5px 0; color: {phase_color};">{icc_summary.get('market_structure', 'No Setup').replace('_', ' ').title()}</h2>
                <p style="margin: 0; color: #aaaaaa;">{icc_summary.get('no_setup_label', '')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2_icc:
            ind_type = icc_summary.get('indication_type', 'None')
            ind_color = '#00ff88' if ind_type == 'bullish' else '#ff4444' if ind_type == 'bearish' else '#888888'
            st.markdown(f"""
            <div class="phase-box" style="border-left-color: {ind_color};">
                <h4 style="margin: 0; color: #ffffff;">Indication</h4>
                <h3 style="margin: 5px 0; color: {ind_color};">{ind_type.title() if ind_type else 'None'}</h3>
                <p style="margin: 0; color: #aaaaaa;">Level: {icc_summary.get('indication_level', 'None'):.2f if icc_summary.get('indication_level') else 'None'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3_icc:
            trend_dir = icc_summary.get('last_trend_dir', 'Neutral')
            trend_color = '#00ff88' if trend_dir == 'bullish' else '#ff4444' if trend_dir == 'bearish' else '#888888'
            st.markdown(f"""
            <div class="phase-box" style="border-left-color: {trend_color};">
                <h4 style="margin: 0; color: #ffffff;">Trend Direction</h4>
                <h3 style="margin: 5px 0; color: {trend_color};">{trend_dir.title() if trend_dir else 'Neutral'}</h3>
                <p style="margin: 0; color: #aaaaaa;">Correction: {icc_summary.get('bars_in_correction', 0)} bars</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4_icc:
            entry_type = icc_summary.get('last_entry_type', 'None')
            entry_color = '#ffff00' if entry_type != 'None' else '#888888'
            st.markdown(f"""
            <div class="phase-box" style="border-left-color: {entry_color};">
                <h4 style="margin: 0; color: #ffffff;">Last Signal</h4>
                <h3 style="margin: 5px 0; color: {entry_color};">{entry_type.replace('_', ' ').title() if entry_type else 'None'}</h3>
                <p style="margin: 0; color: #aaaaaa;">Price: {df['close'].iloc[-1]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed ICC Analysis
        with st.expander("📋 ICC Detailed Analysis", expanded=False):
            col_det1, col_det2, col_det3 = st.columns(3)
            
            with col_det1:
                st.markdown("**Market Structure**")
                st.info(f"""
                - Phase: {icc_summary['market_structure']}
                - Trend: {icc_summary['last_trend_dir'] or 'Neutral'}
                - Correction: {icc_summary['bars_in_correction']} bars
                - Continuation: {icc_summary['bars_in_continuation']} bars
                """)
            
            with col_det2:
                st.markdown("**Support & Resistance**")
                support_text = "\n".join([f"• {s:.2f}" for s in icc_summary['support_levels'][-3:]]) if icc_summary['support_levels'] else "None"
                resistance_text = "\n".join([f"• {r:.2f}" for r in icc_summary['resistance_levels'][-3:]]) if icc_summary['resistance_levels'] else "None"
                st.info(f"""
                **Support:**\n{support_text}
                **Resistance:**\n{resistance_text}
                """)
            
            with col_det3:
                st.markdown("**Indication & Signals**")
                st.info(f"""
                - Type: {icc_summary['indication_type'] or 'None'}
                - Level: {icc_summary['indication_level'] or 'None'}
                - Last Signal: {icc_summary['last_entry_type'] or 'None'}
                - Correction Range: {icc_summary['correction_range_low'] or 'None'} - {icc_summary['correction_range_high'] or 'None'}
                """)

# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    main()
