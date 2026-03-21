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
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

# Page configuration - ADD THIS AT THE VERY TOP
st.set_page_config(
    page_title="Nifty Trading & Options Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 80 seconds - MOVE THIS RIGHT AFTER PAGE CONFIG
# ── Auto-refresh only during market hours (08:30–16:00 IST, Mon–Fri) ────────
_ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
_mkt_open  = _ist_now.replace(hour=8,  minute=30, second=0, microsecond=0)
_mkt_close = _ist_now.replace(hour=16, minute=0,  second=0, microsecond=0)
if _ist_now.weekday() < 5 and _mkt_open <= _ist_now <= _mkt_close:
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
    /* ── Prevent page dimming / dark overlay on auto-refresh ── */
    [data-testid="stApp"] {
        opacity: 1 !important;
        transition: none !important;
    }
    /* Hide the thin loading bar at the top */
    [data-testid="stDecoration"] {
        display: none !important;
    }
    /* Keep all content fully visible during reload */
    .main .block-container {
        opacity: 1 !important;
    }
    div[data-stale="true"] {
        opacity: 1 !important;
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

SENSEX_SCRIP_ID = "51"        # BSE Sensex security ID in Dhan API
SENSEX_EXCHANGE_SEG = "IDX_I"  # All indices (NSE + BSE) share IDX_I segment in Dhan API

# ── BankNifty Dashboard: yfinance symbol map ──────────────────────────
BN_DASH_TICKERS = [
    {"name": "BANKNIFTY", "yf": "^NSEBANK",    "weight": 35.0, "inverse": False},
    {"name": "RELIANCE",  "yf": "RELIANCE.NS",  "weight": 25.0, "inverse": False},
    {"name": "NIFTY AUTO","yf": "^CNXAUTO",     "weight": 10.0, "inverse": False},
    {"name": "NIFTY IT",  "yf": "^CNXIT",       "weight":  8.0, "inverse": False},
    {"name": "HDFCBANK",  "yf": "HDFCBANK.NS",  "weight":  8.0, "inverse": False},
    {"name": "ICICIBANK", "yf": "ICICIBANK.NS",  "weight":  7.0, "inverse": False},
    {"name": "KOTAKBANK", "yf": "KOTAKBANK.NS",  "weight":  4.0, "inverse": False},
    {"name": "SBIN",      "yf": "SBIN.NS",       "weight":  3.0, "inverse": False},
    {"name": "SENSEX",    "yf": "^BSESN",        "weight":  0.0, "inverse": False},
]
BN_MACRO_TICKERS = [
    {"name": "INDIA VIX", "yf": "^INDIAVIX", "inverse": True},
    {"name": "USD/INR",   "yf": "USDINR=X",  "inverse": True},
    {"name": "CRUDE OIL", "yf": "CL=F",      "inverse": True},
]

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

def send_telegram_photo_sync(image_bytes, caption=""):
    """Send a photo (PNG bytes) to Telegram via sendPhoto"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        response = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("chart.png", image_bytes, "image/png")},
            timeout=30,
        )
        return response.json() if response.status_code == 200 else None
    except Exception:
        pass

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
    
    _DB_SCHEMA_SQL = """
-- Run this once in your Supabase SQL editor to create the required table:
CREATE TABLE IF NOT EXISTS candle_data (
    id          bigserial PRIMARY KEY,
    symbol      text        NOT NULL,
    exchange    text        NOT NULL,
    timeframe   text        NOT NULL,
    timestamp   bigint      NOT NULL,
    datetime    timestamptz NOT NULL,
    open        double precision,
    high        double precision,
    low         double precision,
    close       double precision,
    volume      bigint,
    UNIQUE (symbol, exchange, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_candle_data_lookup
    ON candle_data (symbol, exchange, timeframe, datetime DESC);
"""

    def save_candle_data(self, symbol, exchange, timeframe, df):
        """Save candle data to Supabase — silently skips on schema errors."""
        if df.empty:
            return
        try:
            records = []
            for _, row in df.iterrows():
                records.append({
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'timestamp': int(row['timestamp']),
                    'datetime': row['datetime'].isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']),
                })
            self.client.table('candle_data').upsert(
                records,
                on_conflict="symbol,exchange,timeframe,timestamp"
            ).execute()
        except Exception as e:
            err = str(e)
            if "23505" in err or "duplicate key" in err.lower():
                return   # ignore duplicate inserts
            if "does not exist" in err or "42703" in err or "42P01" in err:
                # Schema mismatch — show SQL once and skip silently
                if not st.session_state.get('_db_schema_warn_shown'):
                    st.session_state['_db_schema_warn_shown'] = True
                    st.warning(
                        "⚠️ **Supabase table schema mismatch** — candle caching disabled.\n\n"
                        "Run this SQL in your Supabase SQL editor, then reload:\n"
                        f"```sql{self._DB_SCHEMA_SQL}```"
                    )
            # All other errors: silently skip (don't block the chart)

    def get_candle_data(self, symbol, exchange, timeframe, hours_back=24):
        """Retrieve candle data from Supabase — returns empty df on schema errors."""
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
            return pd.DataFrame()
        except Exception as e:
            err = str(e)
            if "does not exist" in err or "42703" in err or "42P01" in err:
                if not st.session_state.get('_db_schema_warn_shown'):
                    st.session_state['_db_schema_warn_shown'] = True
                    st.warning(
                        "⚠️ **Supabase `candle_data` table missing or wrong schema** — "
                        "cache disabled, fetching live from API.\n\n"
                        "Fix: run this SQL in Supabase → SQL Editor:\n"
                        f"```sql{self._DB_SCHEMA_SQL}```"
                    )
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

    def save_spike_history(self, record):
        """Save options spike snapshot; keep last 300 records"""
        try:
            self.client.table('spike_history').insert(record).execute()
            # Trim to 300 records
            try:
                all_rows = self.client.table('spike_history').select('id').order('id', desc=False).execute()
                if all_rows.data and len(all_rows.data) > 300:
                    oldest_ids = [r['id'] for r in all_rows.data[:len(all_rows.data) - 300]]
                    self.client.table('spike_history').delete().in_('id', oldest_ids).execute()
            except Exception:
                pass
        except Exception as e:
            if "23505" not in str(e) and "duplicate key" not in str(e).lower():
                pass  # Silently ignore spike save errors

    def save_expiry_spike_history(self, record):
        """Save expiry spike snapshot; keep last 300 records"""
        try:
            self.client.table('expiry_spike_history').insert(record).execute()
            try:
                all_rows = self.client.table('expiry_spike_history').select('id').order('id', desc=False).execute()
                if all_rows.data and len(all_rows.data) > 300:
                    oldest_ids = [r['id'] for r in all_rows.data[:len(all_rows.data) - 300]]
                    self.client.table('expiry_spike_history').delete().in_('id', oldest_ids).execute()
            except Exception:
                pass
        except Exception as e:
            pass

    def save_gamma_sequence_history(self, record):
        """Save gamma sequence snapshot; keep last 300 records"""
        try:
            self.client.table('gamma_sequence_history').insert(record).execute()
            try:
                all_rows = self.client.table('gamma_sequence_history').select('id').order('id', desc=False).execute()
                if all_rows.data and len(all_rows.data) > 300:
                    oldest_ids = [r['id'] for r in all_rows.data[:len(all_rows.data) - 300]]
                    self.client.table('gamma_sequence_history').delete().in_('id', oldest_ids).execute()
            except Exception:
                pass
        except Exception as e:
            pass


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

    def _handle_api_error(self, status_code, body=""):
        """Show a clean, actionable error for known Dhan API status codes."""
        if status_code == 403:
            if not st.session_state.get('_dhan_403_shown'):
                st.session_state['_dhan_403_shown'] = True
                st.error(
                    "🔐 **Dhan API — Access Denied (403)**\n\n"
                    "Your access token has **expired or is invalid**.\n\n"
                    "**Fix:**\n"
                    "1. Log in to [Dhan Developer Console](https://developer.dhan.co)\n"
                    "2. Generate a new Access Token\n"
                    "3. Update `DHAN_ACCESS_TOKEN` in your Streamlit secrets (`.streamlit/secrets.toml`)\n"
                    "4. Reload the app"
                )
        elif status_code == 401:
            st.error("🔐 **Dhan API — Unauthorised (401):** Invalid client ID or token. Check your credentials.")
        elif status_code == 429:
            st.warning("⏳ **Dhan API — Rate Limited (429):** Too many requests. Please wait a moment and refresh.")
        elif status_code >= 500:
            st.warning(f"🌐 **Dhan API — Server Error ({status_code}):** Dhan servers are temporarily unavailable. Try again shortly.")
        else:
            st.error(f"**Dhan API Error {status_code}** — please check your credentials and network.")
        
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
                self._handle_api_error(response.status_code, response.text)
                return None
        except Exception as e:
            st.error(f"Network error fetching chart data: {str(e)}")
            return None

    def get_intraday_data_range(self, security_id="13", exchange_segment="IDX_I", instrument="INDEX", interval="1", from_date=None, to_date=None):
        """Get intraday historical data for an explicit date range (used for backtesting)."""
        url = f"{self.base_url}/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        if to_date is None:
            to_date = datetime.now(ist)
        if from_date is None:
            from_date = to_date - timedelta(days=1)
        _fmt = lambda d: d.strftime("%Y-%m-%d %H:%M:%S") if hasattr(d, 'strftime') else str(d)
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": False,
            "fromDate": _fmt(from_date),
            "toDate": _fmt(to_date),
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                self._handle_api_error(response.status_code, response.text)
                return None
        except Exception as e:
            st.error(f"Network error fetching range data: {str(e)}")
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
                self._handle_api_error(response.status_code, response.text)
                return None
        except Exception as e:
            st.error(f"Network error fetching LTP: {str(e)}")
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

    1. ATM Bias: Verdict is Strong Bullish or Strong Bearish (BiasScore >= 4 or <= -4)
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


def calculate_gamma_sequence(df_summary, spot_price, contract_multiplier=25):
    """
    Calculate Gamma Sequence - shows gamma exposure progression across strikes.

    For each strike, computes:
    - Cumulative gamma from lowest strike to current (running sum)
    - Gamma profile shape (where gamma concentrates)
    - Gamma acceleration (rate of change of cumulative gamma)

    This helps identify zones where dealer hedging pressure intensifies or weakens.

    Returns: dict with gamma_seq_df, gamma_profile, peak_gamma_strike, gamma_acceleration
    """
    if df_summary is None or df_summary.empty:
        return None

    try:
        seq_data = []
        for _, row in df_summary.iterrows():
            strike = row.get('Strike', 0)
            gamma_ce = row.get('Gamma_CE', 0) or 0
            gamma_pe = row.get('Gamma_PE', 0) or 0
            oi_ce = row.get('openInterest_CE', 0) or 0
            oi_pe = row.get('openInterest_PE', 0) or 0

            # Raw gamma exposure per strike (in Lakhs)
            ce_gamma_exp = gamma_ce * oi_ce * contract_multiplier * spot_price / 100000
            pe_gamma_exp = gamma_pe * oi_pe * contract_multiplier * spot_price / 100000
            total_gamma = ce_gamma_exp + pe_gamma_exp
            net_gamma = pe_gamma_exp - ce_gamma_exp  # Dealer perspective

            seq_data.append({
                'Strike': strike,
                'CE_Gamma_Exp': round(ce_gamma_exp, 2),
                'PE_Gamma_Exp': round(pe_gamma_exp, 2),
                'Total_Gamma': round(total_gamma, 2),
                'Net_Gamma': round(net_gamma, 2),
                'Zone': row.get('Zone', '-')
            })

        gamma_seq_df = pd.DataFrame(seq_data).sort_values('Strike').reset_index(drop=True)

        # Cumulative gamma from lowest strike upward
        gamma_seq_df['Cumul_CE_Gamma'] = gamma_seq_df['CE_Gamma_Exp'].cumsum().round(2)
        gamma_seq_df['Cumul_PE_Gamma'] = gamma_seq_df['PE_Gamma_Exp'].cumsum().round(2)
        gamma_seq_df['Cumul_Net_Gamma'] = gamma_seq_df['Net_Gamma'].cumsum().round(2)

        # Gamma acceleration (diff of cumulative = per-strike contribution rate)
        gamma_seq_df['Gamma_Accel'] = gamma_seq_df['Cumul_Net_Gamma'].diff().fillna(0).round(2)

        # Peak gamma strike (max total gamma exposure)
        peak_idx = gamma_seq_df['Total_Gamma'].idxmax()
        peak_gamma_strike = gamma_seq_df.loc[peak_idx, 'Strike']

        # Gamma profile: where does gamma concentrate relative to spot?
        above_spot = gamma_seq_df[gamma_seq_df['Strike'] > spot_price]['Total_Gamma'].sum()
        below_spot = gamma_seq_df[gamma_seq_df['Strike'] < spot_price]['Total_Gamma'].sum()
        total_gamma_all = gamma_seq_df['Total_Gamma'].sum()

        if total_gamma_all > 0:
            above_pct = (above_spot / total_gamma_all) * 100
            below_pct = (below_spot / total_gamma_all) * 100
        else:
            above_pct = below_pct = 50

        if above_pct > 60:
            gamma_profile = "Gamma Heavy Above (Resistance Dominant)"
            profile_color = "#ff4444"
        elif below_pct > 60:
            gamma_profile = "Gamma Heavy Below (Support Dominant)"
            profile_color = "#00ff88"
        else:
            gamma_profile = "Gamma Balanced"
            profile_color = "#FFD700"

        return {
            'gamma_seq_df': gamma_seq_df,
            'peak_gamma_strike': peak_gamma_strike,
            'gamma_profile': gamma_profile,
            'profile_color': profile_color,
            'above_pct': round(above_pct, 1),
            'below_pct': round(below_pct, 1),
            'total_gamma': round(total_gamma_all, 2)
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

def _detect_chart_candle_types(df):
    """Detect basic candle patterns for each row in df. Returns list of dicts with time, pattern, direction, price."""
    if df.empty or len(df) < 2:
        return []
    patterns = []
    bodies = (df['close'] - df['open']).abs()
    ranges = (df['high'] - df['low'])
    _WINDOW = 15  # rolling window for adaptive thresholds

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        po, ph, pl, pc = prev['open'], prev['high'], prev['low'], prev['close']
        body = abs(c - o)
        candle_range = h - l if h != l else 0.001
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        is_bull = c > o
        is_bear = c < o
        ts = row['datetime']
        price = c

        # Adaptive rolling averages — only look at recent candles
        _win_start = max(0, i - _WINDOW)
        avg_body = bodies.iloc[_win_start:i].mean() or bodies.mean() or 1
        avg_range = ranges.iloc[_win_start:i].mean() or ranges.mean() or 1

        detected = None
        direction = None

        # Doji
        if body < avg_range * 0.1:
            detected = 'Doji'
            direction = 'NEUTRAL'
        # Hammer (bullish reversal)
        elif lower_wick > body * 2 and upper_wick < body * 0.5 and is_bull:
            detected = 'Hammer'
            direction = 'BUY'
        # Inverted Hammer / Shooting Star
        elif upper_wick > body * 2 and lower_wick < body * 0.5 and is_bear:
            detected = 'Shooting Star'
            direction = 'SELL'
        # Bullish Engulfing
        elif (pc < po and is_bull and c > po and o < pc and body > avg_body):
            detected = 'Bull Engulfing'
            direction = 'BUY'
        # Bearish Engulfing
        elif (pc > po and is_bear and c < po and o > pc and body > avg_body):
            detected = 'Bear Engulfing'
            direction = 'SELL'
        # Strong Bullish (Marubozu-like)
        elif is_bull and body > avg_body * 1.5 and upper_wick < body * 0.2 and lower_wick < body * 0.2:
            detected = 'Marubozu ↑'
            direction = 'BUY'
        # Strong Bearish (Marubozu-like)
        elif is_bear and body > avg_body * 1.5 and upper_wick < body * 0.2 and lower_wick < body * 0.2:
            detected = 'Marubozu ↓'
            direction = 'SELL'
        # Long Upper Wick Rejection (bearish)
        elif upper_wick > candle_range * 0.55 and upper_wick > body * 1.3:
            detected = 'Upper Wick Rej'
            direction = 'SELL'
        # Long Lower Wick Rejection (bullish)
        elif lower_wick > candle_range * 0.55 and lower_wick > body * 1.3:
            detected = 'Lower Wick Rej'
            direction = 'BUY'

        if detected:
            patterns.append({
                'time': ts,
                'pattern': detected,
                'direction': direction,
                'price': price,
                'high': h,
                'low': l,
            })
    return patterns


class GeometricPatternDetector:
    """
    Detects geometric and reversal chart patterns on NIFTY50 price data.
    Patterns: Double Bottom/Top, H&S, Inv H&S, Triangles, Falling Wedge,
              Flag, Range Breakout, Channel, Trendline Breakout.
    Each detected pattern returns entry, SL, target, RR, confidence score,
    draw instructions for chart overlay, and sentiment classification.
    Scans the RECENT window (last LOOKBACK bars) so patterns are discovered
    as soon as they form, not only at the exact breakout candle.
    """

    TOLERANCE = 0.025      # 2.5% price similarity for double tops/bottoms
    MIN_PATTERN_BARS = 3   # minimum bars between pivots
    LOOKBACK = 60          # bars to scan for recent patterns

    # ── internal helpers ──────────────────────────────────────────────────
    @staticmethod
    def _find_pivots(df, order=2):
        """
        Return indices of swing highs (using df['high']) and
        swing lows (using df['low']) within a rolling ±order window.
        Uses actual OHLC highs/lows, not closes.
        """
        highs_arr = df['high'].values
        lows_arr  = df['low'].values
        n = len(highs_arr)
        highs, lows = [], []
        for i in range(order, n - order):
            window_h = highs_arr[i - order: i + order + 1]
            window_l = lows_arr[i - order: i + order + 1]
            if highs_arr[i] == window_h.max():
                highs.append(i)
            if lows_arr[i] == window_l.min():
                lows.append(i)
        return highs, lows

    @staticmethod
    def _confidence_score(df, breakout_idx, signal):
        """Score 0-8 → Low/Moderate/High/Strong/Institutional Setup."""
        score = 0
        closes  = df['close'].values
        opens   = df['open'].values
        highs_a = df['high'].values
        lows_a  = df['low'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        n = len(closes)
        idx = min(breakout_idx, n - 1)

        # 1. Volume spike at breakout
        if volumes is not None and idx > 5:
            avg_vol = volumes[max(0, idx - 10):idx].mean()
            if avg_vol > 0 and volumes[idx] > avg_vol * 1.4:
                score += 2

        # 2. Breakout candle momentum
        if idx > 0:
            body = abs(closes[idx] - opens[idx])
            avg_body = np.mean(np.abs(
                closes[max(0, idx - 10):idx] - opens[max(0, idx - 10):idx]
            )) if idx > 0 else 1
            if avg_body > 0 and body > avg_body * 1.2:
                score += 1

        # 3. Strong close direction
        rng = highs_a[idx] - lows_a[idx] if highs_a[idx] != lows_a[idx] else 0.001
        close_pos = (closes[idx] - lows_a[idx]) / rng
        if signal == 'BUY'  and close_pos > 0.55:
            score += 1
        elif signal == 'SELL' and close_pos < 0.45:
            score += 1

        # 4. Recent trend alignment (last 10 bars slope)
        if n >= 12:
            slope = np.polyfit(range(10), closes[n - 10:], 1)[0]
            price_unit = closes[-1] / 100  # normalise by ~1%
            if signal == 'BUY'  and slope > price_unit * 0.05:
                score += 1
            elif signal == 'SELL' and slope < -price_unit * 0.05:
                score += 1

        # 5. RSI proxy: 14-bar momentum
        if n >= 16:
            gains = np.maximum(np.diff(closes[n - 15:]), 0)
            losses = np.abs(np.minimum(np.diff(closes[n - 15:]), 0))
            avg_g = gains.mean() if gains.mean() > 0 else 0.001
            avg_l = losses.mean() if losses.mean() > 0 else 0.001
            rsi = 100 - (100 / (1 + avg_g / avg_l))
            if signal == 'BUY'  and 40 < rsi < 75:
                score += 2
            elif signal == 'SELL' and 25 < rsi < 60:
                score += 2

        labels = [
            'Low', 'Low', 'Moderate', 'Moderate',
            'High', 'High', 'Strong', 'Strong', 'Institutional Setup'
        ]
        return labels[min(score, 8)]

    @staticmethod
    def _win_loss(move_pct, signal):
        if signal == 'BUY':
            return 'WIN' if move_pct > 0 else 'LOSS'
        return 'WIN' if move_pct < 0 else 'LOSS'

    def _make_result(self, df, bo_idx, pat, pat_type, sentiment, signal,
                     entry, sl, target, draw_lines, sr_zones):
        """Build a standardised pattern result dict."""
        n = len(df)
        bo_idx = min(bo_idx, n - 1)
        rr = abs(target - entry) / abs(entry - sl) if abs(entry - sl) > 0.001 else 0
        future_idx = min(bo_idx + 5, n - 1)
        move_pct = (df['close'].iloc[future_idx] - entry) / entry * 100 if entry else 0
        return {
            'pattern': pat, 'pattern_type': pat_type,
            'sentiment': sentiment, 'signal': signal,
            'time': df['datetime'].iloc[bo_idx],
            'entry': round(entry, 2),
            'stoploss': round(sl, 2),
            'target': round(target, 2),
            'rr': round(rr, 2),
            'move_pct': round(move_pct, 2),
            'confidence': self._confidence_score(df, bo_idx, signal),
            'highlight_idx': bo_idx,
            'draw_lines': draw_lines,
            'sr_zones': sr_zones,
        }

    # ── pattern detectors ─────────────────────────────────────────────────
    def _detect_double_bottom(self, df):
        """
        Scans a sliding window. At each candidate right-bottom, look back
        for a matching left-bottom and a neckline that was subsequently
        broken. Works on intraday data of any interval.
        """
        results = []
        highs_arr = df['high'].values
        lows_arr  = df['low'].values
        closes    = df['close'].values
        n = len(closes)

        _, lows = self._find_pivots(df, order=2)

        for j in range(1, len(lows)):
            b2 = lows[j]
            # Scan for a matching b1 in the lookback window
            for k in range(j - 1, max(j - 8, -1), -1):
                b1 = lows[k]
                if b2 - b1 < self.MIN_PATTERN_BARS:
                    continue
                p1 = lows_arr[b1]
                p2 = lows_arr[b2]
                if abs(p1 - p2) / max(p1, p2) > self.TOLERANCE:
                    continue
                # Neckline = max close between the two bottoms
                neckline = closes[b1:b2 + 1].max()
                # Find first bar after b2 that closes above neckline
                for bo_idx in range(b2 + 1, min(b2 + 8, n)):
                    if closes[bo_idx] > neckline:
                        ph = neckline - min(p1, p2)
                        entry = neckline
                        sl    = min(lows_arr[b1], lows_arr[b2]) * 0.998
                        tgt   = entry + ph
                        results.append(self._make_result(
                            df, bo_idx,
                            'Double Bottom', 'Reversal', 'Bullish Reversal', 'BUY',
                            entry, sl, tgt,
                            draw_lines=[
                                (df['datetime'].iloc[b1], p1, df['datetime'].iloc[b2], p2, '#00ff88'),
                                (df['datetime'].iloc[b1], neckline, df['datetime'].iloc[bo_idx], neckline, '#FFD700'),
                            ],
                            sr_zones=[(sl, sl * 1.004, 'rgba(0,255,136,0.15)')],
                        ))
                        break
                break  # only use the nearest matching b1
        return results

    def _detect_double_top(self, df):
        results = []
        highs_arr = df['high'].values
        lows_arr  = df['low'].values
        closes    = df['close'].values
        n = len(closes)

        highs, _ = self._find_pivots(df, order=2)

        for j in range(1, len(highs)):
            t2 = highs[j]
            for k in range(j - 1, max(j - 8, -1), -1):
                t1 = highs[k]
                if t2 - t1 < self.MIN_PATTERN_BARS:
                    continue
                p1 = highs_arr[t1]
                p2 = highs_arr[t2]
                if abs(p1 - p2) / max(p1, p2) > self.TOLERANCE:
                    continue
                neckline = closes[t1:t2 + 1].min()
                for bo_idx in range(t2 + 1, min(t2 + 8, n)):
                    if closes[bo_idx] < neckline:
                        ph = max(p1, p2) - neckline
                        entry = neckline
                        sl    = max(highs_arr[t1], highs_arr[t2]) * 1.002
                        tgt   = entry - ph
                        results.append(self._make_result(
                            df, bo_idx,
                            'Double Top', 'Reversal', 'Bearish Reversal', 'SELL',
                            entry, sl, tgt,
                            draw_lines=[
                                (df['datetime'].iloc[t1], p1, df['datetime'].iloc[t2], p2, '#ff4444'),
                                (df['datetime'].iloc[t1], neckline, df['datetime'].iloc[bo_idx], neckline, '#FFD700'),
                            ],
                            sr_zones=[(sl * 0.996, sl, 'rgba(255,68,68,0.15)')],
                        ))
                        break
                break
        return results

    def _detect_head_shoulders(self, df):
        results = []
        highs_arr = df['high'].values
        lows_arr  = df['low'].values
        closes    = df['close'].values
        n = len(closes)

        highs, _ = self._find_pivots(df, order=2)
        if len(highs) < 3:
            return results

        for i in range(len(highs) - 2):
            ls, head, rs = highs[i], highs[i + 1], highs[i + 2]
            if head - ls < self.MIN_PATTERN_BARS or rs - head < self.MIN_PATTERN_BARS:
                continue
            p_ls   = highs_arr[ls]
            p_head = highs_arr[head]
            p_rs   = highs_arr[rs]
            if p_head <= max(p_ls, p_rs):
                continue
            if abs(p_ls - p_rs) / p_head > 0.05:
                continue
            t1_low   = lows_arr[ls:head + 1].min()
            t2_low   = lows_arr[head:rs + 1].min()
            neckline = (t1_low + t2_low) / 2
            for bo_idx in range(rs + 1, min(rs + 8, n)):
                if closes[bo_idx] < neckline:
                    ph    = p_head - neckline
                    entry = neckline
                    sl    = p_rs * 1.003
                    tgt   = neckline - ph
                    results.append(self._make_result(
                        df, bo_idx,
                        'Head & Shoulders', 'Reversal', 'Bearish Reversal', 'SELL',
                        entry, sl, tgt,
                        draw_lines=[
                            (df['datetime'].iloc[ls],   p_ls,   df['datetime'].iloc[head], p_head, '#ff4444'),
                            (df['datetime'].iloc[head], p_head, df['datetime'].iloc[rs],   p_rs,   '#ff4444'),
                            (df['datetime'].iloc[ls],   neckline, df['datetime'].iloc[bo_idx], neckline, '#FFD700'),
                        ],
                        sr_zones=[],
                    ))
                    break
        return results

    def _detect_inv_head_shoulders(self, df):
        results = []
        highs_arr = df['high'].values
        lows_arr  = df['low'].values
        closes    = df['close'].values
        n = len(closes)

        _, lows = self._find_pivots(df, order=2)
        if len(lows) < 3:
            return results

        for i in range(len(lows) - 2):
            ls, head, rs = lows[i], lows[i + 1], lows[i + 2]
            if head - ls < self.MIN_PATTERN_BARS or rs - head < self.MIN_PATTERN_BARS:
                continue
            p_ls   = lows_arr[ls]
            p_head = lows_arr[head]
            p_rs   = lows_arr[rs]
            if p_head >= min(p_ls, p_rs):
                continue
            if abs(p_ls - p_rs) / max(p_head, 1) > 0.05:
                continue
            t1_high  = highs_arr[ls:head + 1].max()
            t2_high  = highs_arr[head:rs + 1].max()
            neckline = (t1_high + t2_high) / 2
            for bo_idx in range(rs + 1, min(rs + 8, n)):
                if closes[bo_idx] > neckline:
                    ph    = neckline - p_head
                    entry = neckline
                    sl    = p_rs * 0.997
                    tgt   = neckline + ph
                    results.append(self._make_result(
                        df, bo_idx,
                        'Inv Head & Shoulders', 'Reversal', 'Bullish Reversal', 'BUY',
                        entry, sl, tgt,
                        draw_lines=[
                            (df['datetime'].iloc[ls],   p_ls,   df['datetime'].iloc[head], p_head, '#00ff88'),
                            (df['datetime'].iloc[head], p_head, df['datetime'].iloc[rs],   p_rs,   '#00ff88'),
                            (df['datetime'].iloc[ls],   neckline, df['datetime'].iloc[bo_idx], neckline, '#FFD700'),
                        ],
                        sr_zones=[],
                    ))
                    break
        return results

    def _scan_trendline_patterns(self, df, window):
        """
        Helper: fit resistance and support trendlines over `window` bars
        ending at each bar i from window+1 to n. Returns a list of
        (i, h_slope, h_int, l_slope, l_int, upper_at_i, lower_at_i).
        """
        highs_arr = df['high'].values
        lows_arr  = df['low'].values
        n = len(highs_arr)
        out = []
        xs = np.arange(window)
        for i in range(window, n):
            seg_h = highs_arr[i - window: i]
            seg_l = lows_arr[i - window: i]
            h_slope, h_int = np.polyfit(xs, seg_h, 1)
            l_slope, l_int = np.polyfit(xs, seg_l, 1)
            upper = h_slope * (window - 1) + h_int
            lower = l_slope * (window - 1) + l_int
            out.append((i, h_slope, h_int, l_slope, l_int, upper, lower))
        return out

    def _detect_triangles(self, df):
        results = []
        closes    = df['close'].values
        highs_arr = df['high'].values
        lows_arr  = df['low'].values
        n = len(closes)
        window = min(20, n - 2)
        if n < window + 2:
            return results

        for (i, h_slope, h_int, l_slope, l_int, upper, lower) in self._scan_trendline_patterns(df, window):
            entry  = closes[i]
            sl_buy = lower
            sl_sell = upper
            ph     = upper - lower
            if ph <= 0:
                continue
            price_tol = closes[i] * 0.001  # 0.1% normalised slope threshold

            # Ascending triangle: flat resistance (h_slope ≈ 0), rising support
            if abs(h_slope) < price_tol and l_slope > price_tol * 0.3:
                if entry > upper:
                    results.append(self._make_result(
                        df, i, 'Ascending Triangle', 'Breakout', 'Bullish Breakout', 'BUY',
                        entry, sl_buy, upper + ph,
                        draw_lines=[
                            (df['datetime'].iloc[i - window], h_int, df['datetime'].iloc[i], upper, '#FFD700'),
                            (df['datetime'].iloc[i - window], l_int, df['datetime'].iloc[i], lower, '#FFD700'),
                        ],
                        sr_zones=[],
                    ))

            # Descending triangle: flat support (l_slope ≈ 0), falling resistance
            elif abs(l_slope) < price_tol and h_slope < -price_tol * 0.3:
                if entry < lower:
                    results.append(self._make_result(
                        df, i, 'Descending Triangle', 'Breakout', 'Bearish Breakdown', 'SELL',
                        entry, sl_sell, lower - ph,
                        draw_lines=[
                            (df['datetime'].iloc[i - window], h_int, df['datetime'].iloc[i], upper, '#FFD700'),
                            (df['datetime'].iloc[i - window], l_int, df['datetime'].iloc[i], lower, '#FFD700'),
                        ],
                        sr_zones=[],
                    ))

            # Symmetrical triangle: both slopes converging
            elif h_slope < -price_tol * 0.3 and l_slope > price_tol * 0.3:
                if entry > upper:
                    results.append(self._make_result(
                        df, i, 'Symmetrical Triangle', 'Breakout', 'Bullish Breakout', 'BUY',
                        entry, sl_buy, upper + ph,
                        draw_lines=[
                            (df['datetime'].iloc[i - window], h_int, df['datetime'].iloc[i], upper, '#FFD700'),
                            (df['datetime'].iloc[i - window], l_int, df['datetime'].iloc[i], lower, '#FFD700'),
                        ],
                        sr_zones=[],
                    ))
                elif entry < lower:
                    results.append(self._make_result(
                        df, i, 'Symmetrical Triangle', 'Breakout', 'Bearish Breakdown', 'SELL',
                        entry, sl_sell, lower - ph,
                        draw_lines=[
                            (df['datetime'].iloc[i - window], h_int, df['datetime'].iloc[i], upper, '#FFD700'),
                            (df['datetime'].iloc[i - window], l_int, df['datetime'].iloc[i], lower, '#FFD700'),
                        ],
                        sr_zones=[],
                    ))
        return results

    def _detect_falling_wedge(self, df):
        results = []
        closes = df['close'].values
        n = len(closes)
        window = min(18, n - 2)
        if n < window + 2:
            return results

        for (i, h_slope, h_int, l_slope, l_int, upper, lower) in self._scan_trendline_patterns(df, window):
            # Both slopes must be negative; resistance falling faster (steeper) than support
            if h_slope < 0 and l_slope < 0 and h_slope < l_slope - 1e-6:
                entry = closes[i]
                if entry > upper:
                    ph = upper - lower
                    sl = lower * 0.998
                    tgt = entry + ph * 2
                    results.append(self._make_result(
                        df, i, 'Falling Wedge', 'Reversal', 'Bullish', 'BUY',
                        entry, sl, tgt,
                        draw_lines=[
                            (df['datetime'].iloc[i - window], h_int, df['datetime'].iloc[i], upper, '#00ff88'),
                            (df['datetime'].iloc[i - window], l_int, df['datetime'].iloc[i], lower, '#00ff88'),
                        ],
                        sr_zones=[],
                    ))
        return results

    def _detect_flag(self, df):
        results = []
        closes    = df['close'].values
        highs_arr = df['high'].values
        lows_arr  = df['low'].values
        n = len(closes)
        if n < 18:
            return results

        pole_len  = 8
        flag_len  = 5

        for i in range(pole_len + flag_len, n):
            pole_start = i - pole_len - flag_len
            pole_end   = i - flag_len
            flag_start = pole_end
            flag_end   = i   # inclusive (breakout bar)

            pole_move = closes[pole_end] - closes[pole_start]
            pole_pct  = pole_move / closes[pole_start] if closes[pole_start] > 0 else 0

            flag_closes = closes[flag_start: flag_end]
            if len(flag_closes) < 2:
                continue
            flag_slope, _ = np.polyfit(range(len(flag_closes)), flag_closes, 1)

            # Bull flag
            if pole_pct > 0.008 and flag_slope < 0:
                flag_top    = highs_arr[flag_start:flag_end].max()
                flag_bottom = lows_arr[flag_start:flag_end].min()
                if closes[i] > flag_top:
                    entry = closes[i]
                    sl    = flag_bottom * 0.998
                    tgt   = entry + abs(pole_move)
                    results.append(self._make_result(
                        df, i, 'Bull Flag', 'Continuation', 'Bullish Continuation', 'BUY',
                        entry, sl, tgt,
                        draw_lines=[
                            (df['datetime'].iloc[pole_start], closes[pole_start],
                             df['datetime'].iloc[pole_end],   closes[pole_end], '#00ff88'),
                        ],
                        sr_zones=[(sl, flag_top * 1.001, 'rgba(0,255,136,0.1)')],
                    ))

            # Bear flag
            elif pole_pct < -0.008 and flag_slope > 0:
                flag_top    = highs_arr[flag_start:flag_end].max()
                flag_bottom = lows_arr[flag_start:flag_end].min()
                if closes[i] < flag_bottom:
                    entry = closes[i]
                    sl    = flag_top * 1.002
                    tgt   = entry - abs(pole_move)
                    results.append(self._make_result(
                        df, i, 'Bear Flag', 'Continuation', 'Bearish Continuation', 'SELL',
                        entry, sl, tgt,
                        draw_lines=[
                            (df['datetime'].iloc[pole_start], closes[pole_start],
                             df['datetime'].iloc[pole_end],   closes[pole_end], '#ff4444'),
                        ],
                        sr_zones=[(flag_bottom * 0.999, sl, 'rgba(255,68,68,0.1)')],
                    ))
        return results

    def _detect_range_breakout(self, df):
        results = []
        closes    = df['close'].values
        highs_arr = df['high'].values
        lows_arr  = df['low'].values
        n = len(closes)
        range_window = min(20, n - 2)
        if n < range_window + 2:
            return results

        for i in range(range_window + 1, n):
            seg_h    = highs_arr[i - range_window: i]
            seg_l    = lows_arr[i - range_window: i]
            resistance = seg_h.max()
            support    = seg_l.min()
            rng = resistance - support
            # Range must be meaningful: at least 0.2% of price
            if rng / max(support, 1) < 0.002:
                continue
            # The window must be somewhat range-bound: std of closes < 0.4 * rng
            if closes[i - range_window:i].std() > rng * 0.5:
                continue

            entry = closes[i]
            if entry > resistance * 1.001:
                sl  = support
                tgt = resistance + rng
                results.append(self._make_result(
                    df, i, 'Range Breakout', 'Breakout', 'Bullish Momentum', 'BUY',
                    entry, sl, tgt,
                    draw_lines=[
                        (df['datetime'].iloc[i - range_window], resistance,
                         df['datetime'].iloc[i], resistance, '#FFD700'),
                        (df['datetime'].iloc[i - range_window], support,
                         df['datetime'].iloc[i], support, '#FFD700'),
                    ],
                    sr_zones=[(support, resistance, 'rgba(255,215,0,0.08)')],
                ))
            elif entry < support * 0.999:
                sl  = resistance
                tgt = support - rng
                results.append(self._make_result(
                    df, i, 'Range Breakdown', 'Breakout', 'Bearish Momentum', 'SELL',
                    entry, sl, tgt,
                    draw_lines=[
                        (df['datetime'].iloc[i - range_window], resistance,
                         df['datetime'].iloc[i], resistance, '#FFD700'),
                        (df['datetime'].iloc[i - range_window], support,
                         df['datetime'].iloc[i], support, '#FFD700'),
                    ],
                    sr_zones=[(support, resistance, 'rgba(255,215,0,0.08)')],
                ))
        return results

    def _detect_channel(self, df):
        results = []
        closes = df['close'].values
        n = len(closes)
        window = min(20, n - 2)
        if n < window + 2:
            return results

        for (i, h_slope, h_int, l_slope, l_int, upper, lower) in self._scan_trendline_patterns(df, window):
            if lower >= upper:
                continue
            # Parallel channel: slopes similar within 40% relative difference
            denom = abs(h_slope) + abs(l_slope) + 1e-9
            if abs(h_slope - l_slope) / denom > 0.40:
                continue
            ph = upper - lower
            entry = closes[i]

            if entry < lower * 0.999:
                sl  = upper
                tgt = lower - ph
                results.append(self._make_result(
                    df, i, 'Channel Breakdown', 'Breakout', 'Bearish Momentum', 'SELL',
                    entry, sl, tgt,
                    draw_lines=[
                        (df['datetime'].iloc[i - window], h_int, df['datetime'].iloc[i], upper, '#ff4444'),
                        (df['datetime'].iloc[i - window], l_int, df['datetime'].iloc[i], lower, '#ff4444'),
                    ],
                    sr_zones=[],
                ))
            elif entry > upper * 1.001:
                sl  = lower
                tgt = upper + ph
                results.append(self._make_result(
                    df, i, 'Channel Breakout', 'Breakout', 'Bullish Momentum', 'BUY',
                    entry, sl, tgt,
                    draw_lines=[
                        (df['datetime'].iloc[i - window], h_int, df['datetime'].iloc[i], upper, '#00ff88'),
                        (df['datetime'].iloc[i - window], l_int, df['datetime'].iloc[i], lower, '#00ff88'),
                    ],
                    sr_zones=[],
                ))
        return results

    def _detect_trendline_breakout(self, df):
        results = []
        closes = df['close'].values
        n = len(closes)
        window = min(15, n - 2)
        if n < window + 2:
            return results

        for (i, h_slope, h_int, l_slope, l_int, upper, lower) in self._scan_trendline_patterns(df, window):
            entry = closes[i]
            # Downtrend trendline breakout: resistance was falling, price breaks above
            if h_slope < 0 and entry > upper * 1.001:
                sl  = lower
                tgt = entry + (entry - sl) * 1.5
                results.append(self._make_result(
                    df, i, 'Trendline Breakout Up', 'Breakout', 'Bullish Trend Change', 'BUY',
                    entry, sl, tgt,
                    draw_lines=[
                        (df['datetime'].iloc[i - window], h_int, df['datetime'].iloc[i], upper, '#00ff88'),
                    ],
                    sr_zones=[],
                ))
            # Uptrend trendline breakdown: support was rising, price breaks below
            elif l_slope > 0 and entry < lower * 0.999:
                sl  = upper
                tgt = entry - (sl - entry) * 1.5
                results.append(self._make_result(
                    df, i, 'Trendline Breakdown', 'Breakout', 'Bearish Trend Change', 'SELL',
                    entry, sl, tgt,
                    draw_lines=[
                        (df['datetime'].iloc[i - window], l_int, df['datetime'].iloc[i], lower, '#ff4444'),
                    ],
                    sr_zones=[],
                ))
        return results

    # ── public API ────────────────────────────────────────────────────────
    def detect_all(self, df):
        """Run all pattern detectors. Returns merged list of pattern dicts."""
        if df is None or df.empty or len(df) < 15:
            return []
        results = []
        for detector in [
            self._detect_double_bottom,
            self._detect_double_top,
            self._detect_head_shoulders,
            self._detect_inv_head_shoulders,
            self._detect_triangles,
            self._detect_falling_wedge,
            self._detect_flag,
            self._detect_range_breakout,
            self._detect_channel,
            self._detect_trendline_breakout,
        ]:
            try:
                results.extend(detector(df))
            except Exception:
                pass
        # Sort by time and deduplicate by pattern name (keep most recent)
        seen = {}
        for r in sorted(results, key=lambda x: x['time']):
            seen[r['pattern']] = r
        return list(seen.values())

    def backtest_scan(self, df, step=5):
        """
        Walk forward scan: at each step, call detect_all on df[:i].
        Returns list of pattern occurrences with win/loss evaluation.
        """
        results = []
        n = len(df)
        for i in range(30, n, step):
            sub_df = df.iloc[:i].copy()
            patterns = self.detect_all(sub_df)
            for p in patterns:
                # Look ahead to evaluate outcome
                future_close = df['close'].iloc[min(i + 5, n - 1)]
                actual_move = (future_close - p['entry']) / p['entry'] * 100 if p['entry'] > 0 else 0
                result = 'WIN' if (
                    (p['signal'] == 'BUY' and future_close > p['entry']) or
                    (p['signal'] == 'SELL' and future_close < p['entry'])
                ) else 'LOSS'
                p2 = dict(p)
                p2['actual_move_pct'] = round(actual_move, 2)
                p2['result'] = result
                results.append(p2)
        return results


def create_candlestick_chart(df, title, interval, show_pivots=True, pivot_settings=None, vob_blocks=None, poc_data=None, swing_data=None, rsi_sz_data=None, ultimate_rsi_data=None, candle_patterns=None, geo_patterns=None):
    """Create TradingView-style candlestick chart with optional pivot levels, VOB zones, POC lines, Swing data, RSI Suppression Zones, and Ultimate RSI"""
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

    # ── Candle Pattern Markers ────────────────────────────────────────────
    if candle_patterns:
        try:
            _buy_x, _buy_y, _buy_txt = [], [], []
            _sell_x, _sell_y, _sell_txt = [], [], []
            _neutral_x, _neutral_y, _neutral_txt = [], [], []
            _offset = (df['high'].max() - df['low'].min()) * 0.005

            for _cp in candle_patterns:
                _t = _cp['time']
                _lbl = _cp['pattern']
                if _cp['direction'] == 'BUY':
                    _buy_x.append(_t)
                    _buy_y.append(_cp['low'] - _offset)
                    _buy_txt.append(_lbl)
                elif _cp['direction'] == 'SELL':
                    _sell_x.append(_t)
                    _sell_y.append(_cp['high'] + _offset)
                    _sell_txt.append(_lbl)
                else:
                    _neutral_x.append(_t)
                    _neutral_y.append(_cp['high'] + _offset)
                    _neutral_txt.append(_lbl)

            if _buy_x:
                fig.add_trace(go.Scatter(
                    x=_buy_x, y=_buy_y, mode='markers+text',
                    marker=dict(symbol='triangle-up', size=12, color='#00ff88'),
                    text=_buy_txt, textposition='bottom center',
                    textfont=dict(color='#00ff88', size=9),
                    name='Bullish Pattern', hovertemplate='%{text}<br>%{x}<extra></extra>',
                    showlegend=True
                ), row=1, col=1)

            if _sell_x:
                fig.add_trace(go.Scatter(
                    x=_sell_x, y=_sell_y, mode='markers+text',
                    marker=dict(symbol='triangle-down', size=12, color='#ff4444'),
                    text=_sell_txt, textposition='top center',
                    textfont=dict(color='#ff4444', size=9),
                    name='Bearish Pattern', hovertemplate='%{text}<br>%{x}<extra></extra>',
                    showlegend=True
                ), row=1, col=1)

            if _neutral_x:
                fig.add_trace(go.Scatter(
                    x=_neutral_x, y=_neutral_y, mode='markers+text',
                    marker=dict(symbol='diamond', size=9, color='#FFD700'),
                    text=_neutral_txt, textposition='top center',
                    textfont=dict(color='#FFD700', size=9),
                    name='Neutral Pattern', hovertemplate='%{text}<br>%{x}<extra></extra>',
                    showlegend=True
                ), row=1, col=1)
        except Exception:
            pass  # Candle pattern markers failed, skip

    # ── Geometric / Reversal pattern overlays ─────────────────────────────
    if geo_patterns:
        try:
            for gp in geo_patterns:
                sig = gp.get('signal', 'BUY')
                pat_color = '#00ff88' if sig == 'BUY' else ('#ff4444' if sig == 'SELL' else '#FFD700')

                # Draw trendlines / necklines for this pattern
                for (x0, y0, x1, y1, line_color) in gp.get('draw_lines', []):
                    fig.add_shape(
                        type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                        line=dict(color=line_color, width=2, dash='dash'),
                        row=1, col=1
                    )

                # Draw S/R zones
                for (low_z, high_z, fill_color) in gp.get('sr_zones', []):
                    fig.add_shape(
                        type='rect',
                        x0=df['datetime'].min(), x1=df['datetime'].max(),
                        y0=low_z, y1=high_z,
                        fillcolor=fill_color,
                        line=dict(width=0),
                        row=1, col=1
                    )

                # Highlight breakout candle
                hi_idx = gp.get('highlight_idx')
                if hi_idx is not None and 0 <= hi_idx < len(df):
                    hi_time = df['datetime'].iloc[hi_idx]
                    hi_price = df['close'].iloc[hi_idx]
                    marker_symbol = 'triangle-up' if sig == 'BUY' else ('triangle-down' if sig == 'SELL' else 'diamond')
                    text_pos = 'bottom center' if sig == 'BUY' else 'top center'
                    fig.add_trace(go.Scatter(
                        x=[hi_time], y=[hi_price],
                        mode='markers+text',
                        marker=dict(symbol=marker_symbol, size=14, color=pat_color,
                                    line=dict(color='white', width=1)),
                        text=[gp.get('pattern', '')[:10]],
                        textposition=text_pos,
                        textfont=dict(color=pat_color, size=8),
                        name=gp.get('pattern', 'Pattern'),
                        hovertemplate=(
                            f"{gp.get('pattern','')} | {gp.get('sentiment','')}<br>"
                            f"Signal: {sig}<br>"
                            f"Entry: ₹{gp.get('entry',0):.1f}<br>"
                            f"SL: ₹{gp.get('stoploss',0):.1f}<br>"
                            f"Target: ₹{gp.get('target',0):.1f}<br>"
                            f"RR: {gp.get('rr',0):.2f}<br>"
                            f"Confidence: {gp.get('confidence','')}<extra></extra>"
                        ),
                        showlegend=True
                    ), row=1, col=1)

                    # Draw entry, SL, target horizontal lines
                    x_end = df['datetime'].max()
                    x_start = hi_time
                    for level, lbl, lcolor in [
                        (gp.get('entry', 0), 'Entry', pat_color),
                        (gp.get('stoploss', 0), 'SL', '#ff8800'),
                        (gp.get('target', 0), 'Target', '#00ccff'),
                    ]:
                        if level:
                            fig.add_shape(
                                type='line', x0=x_start, y0=level, x1=x_end, y1=level,
                                line=dict(color=lcolor, width=1, dash='dot'),
                                row=1, col=1
                            )
                            fig.add_annotation(
                                x=x_end, y=level,
                                text=f"{lbl} {level:.0f}",
                                showarrow=False,
                                font=dict(color=lcolor, size=9),
                                xanchor='left',
                                row=1, col=1
                            )
        except Exception:
            pass  # geo pattern drawing failed, skip

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

def plot_depth_levels(df_summary, underlying_price=None):
    """Diverging horizontal bar chart: top-3 PE bid qty (support) vs top-3 CE ask qty (resistance)."""
    has_bid = 'bidQty_PE' in df_summary.columns and not df_summary['bidQty_PE'].isna().all()
    has_ask = 'askQty_CE' in df_summary.columns and not df_summary['askQty_CE'].isna().all()
    if not has_bid and not has_ask:
        return None

    rows = []
    if has_bid:
        top3_sup = df_summary.nlargest(3, 'bidQty_PE')[['Strike', 'bidQty_PE']].copy()
        top3_sup = top3_sup.sort_values('Strike', ascending=False).reset_index(drop=True)
        for i, r in top3_sup.iterrows():
            rows.append({'price': r['Strike'], 'qty': r['bidQty_PE'], 'label': f'S{i+1}', 'side': 'support'})

    if has_ask:
        top3_res = df_summary.nlargest(3, 'askQty_CE')[['Strike', 'askQty_CE']].copy()
        top3_res = top3_res.sort_values('Strike').reset_index(drop=True)
        for i, r in top3_res.iterrows():
            rows.append({'price': r['Strike'], 'qty': r['askQty_CE'], 'label': f'R{i+1}', 'side': 'resistance'})

    if not rows:
        return None

    rows.sort(key=lambda x: x['price'])

    fig = go.Figure()
    for row in rows:
        is_sup = row['side'] == 'support'
        x_val = -row['qty'] if is_sup else row['qty']
        color = '#00cc66' if is_sup else '#ff4444'
        y_label = f"{row['label']}: ₹{row['price']:,.0f}"
        fig.add_trace(go.Bar(
            x=[x_val],
            y=[y_label],
            orientation='h',
            marker_color=color,
            showlegend=False,
            text=[f"{row['qty']:,.0f}"],
            textposition='outside',
            hovertemplate=f"{row['label']} ₹{row['price']:,.0f}<br>Qty: {row['qty']:,.0f}<extra></extra>",
        ))

    max_qty = max(r['qty'] for r in rows)
    fig.update_layout(
        title=dict(text="Key Levels from Order Book Depth", font=dict(size=16)),
        template='plotly_dark',
        height=320,
        barmode='overlay',
        xaxis=dict(
            title='← Support (PE Bid) &nbsp;&nbsp;|&nbsp;&nbsp; Resistance (CE Ask) →',
            range=[-max_qty * 1.35, max_qty * 1.35],
            zeroline=True,
            zerolinecolor='#FFD700',
            zerolinewidth=2,
            tickformat=',.0f',
            tickvals=[-max_qty, -max_qty // 2, 0, max_qty // 2, max_qty],
            ticktext=[f"{max_qty:,.0f}", f"{max_qty//2:,.0f}", "0",
                      f"{max_qty//2:,.0f}", f"{max_qty:,.0f}"],
        ),
        yaxis=dict(autorange=True, tickfont=dict(size=12)),
        margin=dict(l=10, r=80, t=50, b=50),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        annotations=[dict(
            x=0.02, y=1.08, xref='paper', yref='paper',
            text="🟢 Support Levels (Largest bid quantities) &nbsp;&nbsp; 🔴 Resistance Levels (Largest ask quantities)",
            showarrow=False, font=dict(color='white', size=11)
        )],
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


# ============================================================
# MARKET ACCELERATION ENGINES
# ============================================================

def _safe(val, default=0.0):
    """Return float or default when val is None/NaN."""
    try:
        v = float(val)
        return default if (v != v) else v  # NaN check
    except Exception:
        return default


def calculate_options_spike_score(df_summary, atm_strike, spike_snapshots):
    """
    Options Spike Detector.
    Returns dict with spike_score (0-100), direction, signal_label, conditions_met, individual scores.
    spike_snapshots: list of past df_summary snapshots (last 10).
    """
    result = {
        'spike_score': 0, 'direction': 'Neutral', 'signal': 'Normal Flow',
        'conditions_met': 0, 'vol_score': 0, 'oi_score': 0,
        'straddle_score': 0, 'iv_score': 0, 'pressure_score': 0,
        'volume_spike': False, 'oi_spike': False, 'straddle_spike': False,
        'iv_spike': False, 'pressure_spike': False, 'delta_spike': False,
    }
    try:
        atm_row = df_summary[df_summary['Strike'] == atm_strike]
        if atm_row.empty:
            return result
        atm = atm_row.iloc[0]

        cur_vol_ce = _safe(atm.get('totalTradedVolume_CE', 0))
        cur_vol_pe = _safe(atm.get('totalTradedVolume_PE', 0))
        cur_vol = cur_vol_ce + cur_vol_pe

        cur_chgoi_ce = _safe(atm.get('changeinOpenInterest_CE', 0))
        cur_chgoi_pe = _safe(atm.get('changeinOpenInterest_PE', 0))
        cur_chgoi = abs(cur_chgoi_ce) + abs(cur_chgoi_pe)

        cur_atm_straddle = _safe(atm.get('lastPrice_CE', 0)) + _safe(atm.get('lastPrice_PE', 0))
        cur_iv_ce = _safe(atm.get('impliedVolatility_CE', 15))
        cur_iv_pe = _safe(atm.get('impliedVolatility_PE', 15))
        cur_iv = (cur_iv_ce + cur_iv_pe) / 2.0

        cur_bid_ce = _safe(atm.get('bidQty_CE', 0))
        cur_ask_ce = _safe(atm.get('askQty_CE', 0))
        cur_bid_pe = _safe(atm.get('bidQty_PE', 0))
        cur_ask_pe = _safe(atm.get('askQty_PE', 0))
        denom = cur_bid_ce + cur_ask_ce + cur_bid_pe + cur_ask_pe
        cur_pressure = (cur_bid_pe - cur_ask_ce) / denom if denom > 0 else 0.0

        cur_delta_ce = _safe(atm.get('Delta_CE', 0.5))
        cur_delta_pe = _safe(atm.get('Delta_PE', -0.5))

        # Historical averages from last 10 snapshots
        avg_vol = cur_vol; avg_chgoi = cur_chgoi
        avg_straddle = cur_atm_straddle; avg_iv = cur_iv; avg_pressure = cur_pressure
        if spike_snapshots:
            vols, chgois, straddles, ivs, pressures = [], [], [], [], []
            for snap in spike_snapshots[-10:]:
                atm_snap = snap[snap['Strike'] == atm_strike] if 'Strike' in snap.columns else pd.DataFrame()
                if atm_snap.empty:
                    continue
                s = atm_snap.iloc[0]
                vols.append(_safe(s.get('totalTradedVolume_CE', 0)) + _safe(s.get('totalTradedVolume_PE', 0)))
                chgois.append(abs(_safe(s.get('changeinOpenInterest_CE', 0))) + abs(_safe(s.get('changeinOpenInterest_PE', 0))))
                straddles.append(_safe(s.get('lastPrice_CE', 0)) + _safe(s.get('lastPrice_PE', 0)))
                iv_avg = (_safe(s.get('impliedVolatility_CE', 15)) + _safe(s.get('impliedVolatility_PE', 15))) / 2.0
                ivs.append(iv_avg)
                bd = (_safe(s.get('bidQty_CE', 0)) + _safe(s.get('askQty_CE', 0)) +
                      _safe(s.get('bidQty_PE', 0)) + _safe(s.get('askQty_PE', 0)))
                pressures.append((_safe(s.get('bidQty_PE', 0)) - _safe(s.get('askQty_CE', 0))) / bd if bd > 0 else 0)
            if vols:
                avg_vol = sum(vols) / len(vols)
            if chgois:
                avg_chgoi = sum(chgois) / len(chgois)
            if straddles and straddles[-1] > 0:
                avg_straddle = straddles[-1]
                prev_straddle = straddles[0] if straddles[0] > 0 else straddles[-1]
            if ivs and ivs[-1] > 0:
                avg_iv = ivs[-1]; prev_iv = ivs[0] if ivs[0] > 0 else ivs[-1]
            if pressures:
                avg_pressure = sum(pressures) / len(pressures)

        # Condition checks
        conditions = 0
        # 1. Volume > 2x avg
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_spike = vol_ratio >= 2.0
        result['volume_spike'] = vol_spike
        if vol_spike:
            conditions += 1

        # 2. OI Change > 1.5x avg
        oi_ratio = cur_chgoi / avg_chgoi if avg_chgoi > 0 else 1.0
        oi_spike = oi_ratio >= 1.5
        result['oi_spike'] = oi_spike
        if oi_spike:
            conditions += 1

        # 3. Straddle increase > 1%
        straddle_change_pct = 0.0
        if spike_snapshots and len(spike_snapshots) >= 2:
            snap_prev = spike_snapshots[-2]
            prev_row = snap_prev[snap_prev['Strike'] == atm_strike] if 'Strike' in snap_prev.columns else pd.DataFrame()
            if not prev_row.empty:
                prev_str = _safe(prev_row.iloc[0].get('lastPrice_CE', 0)) + _safe(prev_row.iloc[0].get('lastPrice_PE', 0))
                straddle_change_pct = (cur_atm_straddle - prev_str) / prev_str * 100 if prev_str > 0 else 0
        straddle_spike = straddle_change_pct > 1.0
        result['straddle_spike'] = straddle_spike
        if straddle_spike:
            conditions += 1

        # 4. IV increase > 0.5%
        iv_change = 0.0
        if spike_snapshots and len(spike_snapshots) >= 2:
            snap_prev = spike_snapshots[-2]
            prev_row = snap_prev[snap_prev['Strike'] == atm_strike] if 'Strike' in snap_prev.columns else pd.DataFrame()
            if not prev_row.empty:
                prev_iv = (_safe(prev_row.iloc[0].get('impliedVolatility_CE', 15)) +
                           _safe(prev_row.iloc[0].get('impliedVolatility_PE', 15))) / 2.0
                iv_change = cur_iv - prev_iv
        iv_spike = iv_change > 0.5
        result['iv_spike'] = iv_spike
        if iv_spike:
            conditions += 1

        # 5. Pressure jump > 0.12
        pressure_jump = abs(cur_pressure - avg_pressure) > 0.12
        result['pressure_spike'] = pressure_jump
        if pressure_jump:
            conditions += 1

        # 6. Delta shift (net delta moved significantly)
        delta_shift = abs(cur_delta_ce + cur_delta_pe) > 0.08
        result['delta_spike'] = delta_shift
        if delta_shift:
            conditions += 1

        result['conditions_met'] = conditions

        # Component scores (0-20 each, sum max 100)
        vol_score    = min(20, int((min(vol_ratio, 4) / 4) * 20))
        oi_score     = min(20, int((min(oi_ratio, 3) / 3) * 20))
        straddle_score = min(20, int(min(abs(straddle_change_pct) / 3, 1) * 20))
        iv_score     = min(20, int(min(abs(iv_change) / 2, 1) * 20))
        pressure_score = min(20, int(min(abs(cur_pressure - avg_pressure) / 0.3, 1) * 20))

        spike_score = vol_score + oi_score + straddle_score + iv_score + pressure_score
        spike_score = max(0, min(100, spike_score))

        # Gate by conditions met (< 3 conditions = cap score at 30)
        if conditions < 3:
            spike_score = min(spike_score, 30)

        # Direction detection
        ce_pressure_rising = cur_bid_ce > cur_ask_ce
        pe_pressure_rising = cur_bid_pe > cur_ask_pe
        delta_rising = cur_delta_ce + cur_delta_pe > 0
        ce_vol_dominant = cur_vol_ce > cur_vol_pe

        if ce_pressure_rising and delta_rising and ce_vol_dominant:
            direction = 'Bullish'
        elif pe_pressure_rising and not delta_rising and not ce_vol_dominant:
            direction = 'Bearish'
        else:
            direction = 'Neutral'

        # Signal label
        if spike_score >= 80:
            signal = 'Institutional Spike'
        elif spike_score >= 60:
            signal = 'Smart Money Activity'
        elif spike_score >= 30:
            signal = 'Activity Increasing'
        else:
            signal = 'Normal Flow'

        result.update({
            'spike_score': spike_score, 'direction': direction, 'signal': signal,
            'vol_score': vol_score, 'oi_score': oi_score, 'straddle_score': straddle_score,
            'iv_score': iv_score, 'pressure_score': pressure_score,
            'vol_ratio': round(vol_ratio, 2), 'oi_ratio': round(oi_ratio, 2),
            'straddle_change_pct': round(straddle_change_pct, 2), 'iv_change': round(iv_change, 2),
            'pressure': round(cur_pressure, 3),
        })
    except Exception as e:
        pass
    return result


def calculate_expiry_spike_score(df_summary, atm_strike, expiry_date_str, expiry_history):
    """
    Expiry Spike Detector — activates when DTE ≤ 2.
    Returns dict with expiry_spike_score, market_type, signals, short_cover, long_unwind.
    expiry_history: list of past df_summary snapshots.
    """
    result = {
        'active': False, 'dte': 999, 'expiry_spike_score': 0,
        'signal': 'Normal Expiry Movement', 'straddle_move': 0.0,
        'gamma_shift': 0.0, 'oi_shift': 0.0, 'iv_change': 0.0, 'pressure_shift': 0.0,
        'short_cover': False, 'long_unwind': False,
    }
    try:
        # Calculate DTE
        expiry_dt = datetime.strptime(expiry_date_str, "%Y-%m-%d")
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        dte = (expiry_dt.date() - now.date()).days
        result['dte'] = dte

        if dte > 2:
            return result
        result['active'] = True

        atm_row = df_summary[df_summary['Strike'] == atm_strike]
        if atm_row.empty:
            return result
        atm = atm_row.iloc[0]

        cur_straddle = _safe(atm.get('lastPrice_CE', 0)) + _safe(atm.get('lastPrice_PE', 0))
        cur_gamma_ce = _safe(atm.get('Gamma_CE', 0))
        cur_gamma_pe = _safe(atm.get('Gamma_PE', 0))
        cur_oi_ce = _safe(atm.get('openInterest_CE', 0))
        cur_oi_pe = _safe(atm.get('openInterest_PE', 0))
        cur_chgoi_ce = _safe(atm.get('changeinOpenInterest_CE', 0))
        cur_chgoi_pe = _safe(atm.get('changeinOpenInterest_PE', 0))
        cur_iv_ce = _safe(atm.get('impliedVolatility_CE', 15))
        cur_iv_pe = _safe(atm.get('impliedVolatility_PE', 15))
        cur_bid_pe = _safe(atm.get('bidQty_PE', 0))
        cur_ask_ce = _safe(atm.get('askQty_CE', 0))

        straddle_move = 0.0; gamma_shift = 0.0; oi_shift = 0.0
        iv_change = 0.0; pressure_shift = 0.0

        if expiry_history and len(expiry_history) >= 2:
            prev_snap = expiry_history[-2]
            prev_atm = prev_snap[prev_snap['Strike'] == atm_strike] if 'Strike' in prev_snap.columns else pd.DataFrame()
            if not prev_atm.empty:
                p = prev_atm.iloc[0]
                prev_straddle = _safe(p.get('lastPrice_CE', 0)) + _safe(p.get('lastPrice_PE', 0))
                straddle_move = (cur_straddle - prev_straddle) / prev_straddle * 100 if prev_straddle > 0 else 0

                prev_gamma = _safe(p.get('Gamma_CE', 0)) + _safe(p.get('Gamma_PE', 0))
                cur_gamma = cur_gamma_ce + cur_gamma_pe
                gamma_shift = abs(cur_gamma - prev_gamma) * 100

                prev_oi = _safe(p.get('openInterest_CE', 0)) + _safe(p.get('openInterest_PE', 0))
                cur_oi = cur_oi_ce + cur_oi_pe
                oi_shift = abs(cur_oi - prev_oi) / max(prev_oi, 1) * 100

                prev_iv = (_safe(p.get('impliedVolatility_CE', 15)) + _safe(p.get('impliedVolatility_PE', 15))) / 2
                cur_iv = (cur_iv_ce + cur_iv_pe) / 2
                iv_change = abs(cur_iv - prev_iv)

                prev_bid_pe = _safe(p.get('bidQty_PE', 0))
                prev_ask_ce = _safe(p.get('askQty_CE', 0))
                pressure_shift = abs((cur_bid_pe - cur_ask_ce) - (prev_bid_pe - prev_ask_ce)) / max(cur_straddle, 1)

        # Expiry spike score formula
        expiry_spike_score = (
            0.30 * min(abs(straddle_move) * 10, 100) +
            0.25 * min(gamma_shift * 5, 100) +
            0.20 * min(oi_shift * 2, 100) +
            0.15 * min(iv_change * 10, 100) +
            0.10 * min(pressure_shift * 20, 100)
        )
        expiry_spike_score = max(0, min(100, expiry_spike_score))

        if expiry_spike_score >= 85:
            signal = 'Expiry Explosion'
        elif expiry_spike_score >= 70:
            signal = 'Expiry Breakout Setup'
        elif expiry_spike_score >= 40:
            signal = 'Expiry Build-Up'
        else:
            signal = 'Normal Expiry Movement'

        # Short covering: CE OI decreasing, price rising, IV falling
        short_cover = (cur_chgoi_ce < 0 and iv_change < 0)
        # Long unwinding: PE OI decreasing
        long_unwind = (cur_chgoi_pe < 0)

        result.update({
            'expiry_spike_score': round(expiry_spike_score, 1),
            'signal': signal, 'straddle_move': round(straddle_move, 2),
            'gamma_shift': round(gamma_shift, 4), 'oi_shift': round(oi_shift, 2),
            'iv_change': round(iv_change, 2), 'pressure_shift': round(pressure_shift, 4),
            'short_cover': short_cover, 'long_unwind': long_unwind,
        })
    except Exception:
        pass
    return result


def analyze_gamma_sequence_mae(df_summary, atm_strike, gamma_seq_history):
    """
    Gamma Sequence Analyzer + Gamma Trap Detector (Market Acceleration Engine).
    Returns dict with pattern, acceleration, trap, dealer_signal.
    gamma_seq_history: list of per-snapshot dicts [{ATM-2: gamma, ATM-1: gamma, ATM: gamma, ...}].
    """
    result = {
        'pattern': 'Unknown', 'direction': 'Neutral',
        'acceleration': False, 'dealer_signal': 'Normal',
        'bull_trap': False, 'bear_trap': False, 'trap_signal': '',
        'gamma_values': {}, 'gamma_ramp': 'None',
    }
    try:
        strikes = sorted(df_summary['Strike'].unique())
        atm_idx = None
        for i, s in enumerate(strikes):
            if s == atm_strike:
                atm_idx = i
                break
        if atm_idx is None:
            return result

        pos_keys = ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']
        pos_offsets = [-2, -1, 0, 1, 2]
        gamma_vals = {}
        for pk, offset in zip(pos_keys, pos_offsets):
            idx = atm_idx + offset
            if 0 <= idx < len(strikes):
                row = df_summary[df_summary['Strike'] == strikes[idx]]
                if not row.empty:
                    gce = _safe(row.iloc[0].get('Gamma_CE', 0))
                    gpe = _safe(row.iloc[0].get('Gamma_PE', 0))
                    gamma_vals[pk] = gce + gpe
            else:
                gamma_vals[pk] = 0.0

        result['gamma_values'] = gamma_vals

        # Gamma Ramp Up: gamma increases from lower to higher strikes
        g_vals = [gamma_vals.get(k, 0) for k in pos_keys]
        ramp_up_count = sum(1 for i in range(len(g_vals) - 1) if g_vals[i+1] > g_vals[i])
        ramp_dn_count = sum(1 for i in range(len(g_vals) - 1) if g_vals[i+1] < g_vals[i])

        if ramp_up_count >= 3:
            pattern = 'Gamma Ramp Up'
            direction = 'Bullish'
        elif ramp_dn_count >= 3:
            pattern = 'Gamma Ramp Down'
            direction = 'Bearish'
        else:
            pattern = 'Gamma Flat'
            direction = 'Neutral'

        result['pattern'] = pattern
        result['direction'] = direction
        result['gamma_ramp'] = pattern

        # Gamma Acceleration: gamma ATM increased quickly over last 2 updates
        acceleration = False
        if gamma_seq_history and len(gamma_seq_history) >= 2:
            prev_gamma_atm = _safe(gamma_seq_history[-2].get('ATM', 0))
            cur_gamma_atm = gamma_vals.get('ATM', 0)
            if prev_gamma_atm > 0 and cur_gamma_atm > prev_gamma_atm * 1.15:
                acceleration = True
        result['acceleration'] = acceleration
        result['dealer_signal'] = 'Dealer Hedge Acceleration' if acceleration else 'Normal Hedging'

        # Gamma Trap Detector
        atm_row = df_summary[df_summary['Strike'] == atm_strike]
        bull_trap = False; bear_trap = False; trap_signal = ''
        if not atm_row.empty:
            a = atm_row.iloc[0]
            net_gamma = _safe(a.get('GammaExp_Net', 0))
            chgoi_ce = _safe(a.get('changeinOpenInterest_CE', 0))
            chgoi_pe = _safe(a.get('changeinOpenInterest_PE', 0))
            # Bull trap: gamma positive, CE OI increasing but price not rising (no upward ramp)
            if net_gamma > 0 and chgoi_ce > 0 and direction != 'Bullish':
                bull_trap = True
                trap_signal = 'CALL TRAP'
            # Bear trap: gamma negative, PE OI increasing but price not falling
            if net_gamma < 0 and chgoi_pe > 0 and direction != 'Bearish':
                bear_trap = True
                trap_signal = 'PUT TRAP' if not bull_trap else trap_signal

        result['bull_trap'] = bull_trap
        result['bear_trap'] = bear_trap
        result['trap_signal'] = trap_signal

    except Exception:
        pass
    return result


def calculate_expiry_day_intelligence(df_summary, atm_strike, underlying, expiry_date_str, expiry_history):
    """
    Expiry Day Intelligence Engine — full expiry analysis.
    Returns comprehensive dict; active only when DTE ≤ 1.
    """
    result = {
        'active': False, 'dte': 999, 'market_type': 'N/A',
        'atm_straddle': 0.0, 'straddle_roc': 0.0,
        'max_pain': None, 'max_pain_signal': 'N/A',
        'gamma_flip': None, 'highest_gamma_strike': None,
        'breakout_level': None, 'breakdown_level': None,
        'entry_support': None, 'entry_resistance': None,
        'expiry_score': 0.0, 'expiry_signal': 'Normal Expiry',
        'oi_shift_signal': '', 'expiry_trap': '',
    }
    try:
        expiry_dt = datetime.strptime(expiry_date_str, "%Y-%m-%d")
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        dte = (expiry_dt.date() - now.date()).days
        result['dte'] = dte

        if dte > 1:
            return result
        result['active'] = True

        atm_row = df_summary[df_summary['Strike'] == atm_strike]
        if atm_row.empty:
            return result
        atm = atm_row.iloc[0]

        cur_straddle = _safe(atm.get('lastPrice_CE', 0)) + _safe(atm.get('lastPrice_PE', 0))
        result['atm_straddle'] = round(cur_straddle, 2)

        # Straddle Rate of Change
        straddle_roc = 0.0
        if expiry_history and len(expiry_history) >= 2:
            prev_snap = expiry_history[-2]
            prev_atm = prev_snap[prev_snap['Strike'] == atm_strike] if 'Strike' in prev_snap.columns else pd.DataFrame()
            if not prev_atm.empty:
                prev_str = _safe(prev_atm.iloc[0].get('lastPrice_CE', 0)) + _safe(prev_atm.iloc[0].get('lastPrice_PE', 0))
                straddle_roc = (cur_straddle - prev_str) / prev_str * 100 if prev_str > 0 else 0
        result['straddle_roc'] = round(straddle_roc, 2)

        # Market type classification
        cur_iv = (_safe(atm.get('impliedVolatility_CE', 15)) + _safe(atm.get('impliedVolatility_PE', 15))) / 2
        price_dist = abs(underlying - atm_strike)
        if price_dist <= 25 and abs(straddle_roc) < 0.5:
            market_type = 'PINNING EXPIRY'
        elif abs(straddle_roc) > 1.0 or price_dist > 100:
            market_type = 'TRENDING EXPIRY'
        else:
            market_type = 'STRIKE ROTATION'
        result['market_type'] = market_type

        # Max Pain (strike with min total pain to writers)
        max_pain_strike = None
        min_pain = float('inf')
        if 'openInterest_CE' in df_summary.columns and 'openInterest_PE' in df_summary.columns:
            for _, row in df_summary.iterrows():
                s = row['Strike']
                pain = 0
                for _, r2 in df_summary.iterrows():
                    s2 = r2['Strike']
                    if s < s2:
                        pain += _safe(r2.get('openInterest_CE', 0)) * (s2 - s)
                    if s > s2:
                        pain += _safe(r2.get('openInterest_PE', 0)) * (s - s2)
                if pain < min_pain:
                    min_pain = pain
                    max_pain_strike = s
        result['max_pain'] = max_pain_strike
        if max_pain_strike:
            if underlying > max_pain_strike:
                result['max_pain_signal'] = 'Magnet Move (Price above Max Pain, expect pull-down)'
            elif underlying < max_pain_strike:
                result['max_pain_signal'] = 'Magnet Move (Price below Max Pain, expect pull-up)'
            else:
                result['max_pain_signal'] = 'At Max Pain'

        # Highest Gamma Strike and Gamma Flip
        if 'GammaExp_Net' in df_summary.columns:
            idx_max_g = df_summary['GammaExp_Net'].abs().idxmax()
            result['highest_gamma_strike'] = df_summary.loc[idx_max_g, 'Strike']
            # Gamma flip = strike where GammaExp_Net changes sign
            sorted_df = df_summary.sort_values('Strike')
            prev_sign = None
            for _, row in sorted_df.iterrows():
                sign = 1 if _safe(row.get('GammaExp_Net', 0)) >= 0 else -1
                if prev_sign is not None and sign != prev_sign:
                    result['gamma_flip'] = row['Strike']
                    break
                prev_sign = sign

        # OI shift signals
        chgoi_ce = _safe(atm.get('changeinOpenInterest_CE', 0))
        chgoi_pe = _safe(atm.get('changeinOpenInterest_PE', 0))
        if chgoi_ce < 0 and straddle_roc > 0:
            oi_shift_signal = 'SHORT COVERING'
        elif chgoi_pe < 0:
            oi_shift_signal = 'LONG UNWINDING'
        elif chgoi_ce > 0:
            oi_shift_signal = 'Fresh Call Writing (Resistance forming)'
        elif chgoi_pe > 0:
            oi_shift_signal = 'Fresh Put Writing (Support forming)'
        else:
            oi_shift_signal = 'Neutral'
        result['oi_shift_signal'] = oi_shift_signal

        # Entry levels
        max_pe_oi_strike = df_summary.loc[df_summary['openInterest_PE'].idxmax(), 'Strike'] if 'openInterest_PE' in df_summary.columns else underlying
        max_ce_oi_strike = df_summary.loc[df_summary['openInterest_CE'].idxmax(), 'Strike'] if 'openInterest_CE' in df_summary.columns else underlying
        gamma_support = result['gamma_flip'] or atm_strike
        gamma_resist = result['highest_gamma_strike'] or atm_strike

        entry_support = round((max_pe_oi_strike + gamma_support) / 2, 0)
        entry_resistance = round((max_ce_oi_strike + gamma_resist) / 2, 0)
        result['entry_support'] = entry_support
        result['entry_resistance'] = entry_resistance

        # Breakout / Breakdown levels
        atm_p1 = df_summary[df_summary['Strike'] > atm_strike]['Strike'].min() if not df_summary[df_summary['Strike'] > atm_strike].empty else atm_strike + 50
        atm_m1 = df_summary[df_summary['Strike'] < atm_strike]['Strike'].max() if not df_summary[df_summary['Strike'] < atm_strike].empty else atm_strike - 50
        breakout = round((atm_p1 + (gamma_resist if gamma_resist else atm_strike + 50)) / 2, 0)
        breakdown = round((atm_m1 + (gamma_support if gamma_support else atm_strike - 50)) / 2, 0)
        result['breakout_level'] = breakout
        result['breakdown_level'] = breakdown

        # Expiry trap detection
        if chgoi_ce > 0 and straddle_roc < 0:
            expiry_trap = 'CALL WRITER TRAP'
        elif chgoi_pe > 0 and straddle_roc < 0:
            expiry_trap = 'PUT WRITER TRAP'
        else:
            expiry_trap = ''
        result['expiry_trap'] = expiry_trap

        # Expiry confidence score
        score = (
            min(abs(straddle_roc) * 20, 40) +
            min(abs(chgoi_ce + chgoi_pe) / 100000, 20) +
            (10 if result['gamma_flip'] else 0) +
            min(cur_iv / 3, 20) +
            (10 if abs(cur_iv - 15) > 5 else 0)
        )
        expiry_score = max(0, min(100, score))
        result['expiry_score'] = round(expiry_score, 1)

        if expiry_score >= 85:
            result['expiry_signal'] = 'EXPIRY MOVE STARTED'
        elif expiry_score >= 70:
            result['expiry_signal'] = 'Breakout Setup'
        elif expiry_score >= 40:
            result['expiry_signal'] = 'Build-up'
        else:
            result['expiry_signal'] = 'Normal Expiry'

    except Exception:
        pass
    return result


def get_combined_acceleration_signal(sentiment_verdict, spike_result, gamma_result):
    """Combine Sentiment Engine + Spike Engine + Gamma Sequence into final signal."""
    spike_score = spike_result.get('spike_score', 0)
    gamma_pattern = gamma_result.get('pattern', 'Unknown')
    spike_direction = spike_result.get('direction', 'Neutral')
    gamma_direction = gamma_result.get('direction', 'Neutral')

    sentiment_bull = 'bullish' in str(sentiment_verdict).lower()
    sentiment_bear = 'bearish' in str(sentiment_verdict).lower()

    if sentiment_bull and spike_score >= 70 and gamma_pattern == 'Gamma Ramp Up':
        return 'HIGH PROBABILITY BREAKOUT', '#00ff88', 'INSTITUTIONAL BUYING DETECTED'
    elif sentiment_bear and spike_score >= 70 and gamma_pattern == 'Gamma Ramp Down':
        return 'HIGH PROBABILITY BREAKDOWN', '#ff4444', 'INSTITUTIONAL SELLING DETECTED'
    elif spike_score >= 60 and spike_direction == 'Bullish':
        return 'BULLISH ACCELERATION', '#44ff88', 'Smart Money Bullish Activity'
    elif spike_score >= 60 and spike_direction == 'Bearish':
        return 'BEARISH ACCELERATION', '#ff6644', 'Smart Money Bearish Activity'
    elif gamma_pattern == 'Gamma Ramp Up':
        return 'GAMMA RAMP UP', '#aaffdd', 'Dealer Repositioning Bullish'
    elif gamma_pattern == 'Gamma Ramp Down':
        return 'GAMMA RAMP DOWN', '#ffaaaa', 'Dealer Repositioning Bearish'
    else:
        return 'MONITORING', '#aaaaaa', 'Normal Market Flow'


def analyze_option_chain(selected_expiry=None, pivot_data=None, vob_data=None, live_spot_price=None):
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
    # Use live spot price from LTP API if available, as option chain's last_price can be stale
    underlying = live_spot_price if live_spot_price and live_spot_price > 0 else data['last_price']

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
        'iv': 'impliedVolatility',
        'security_id': 'scrp_cd'  # Dhan v2 API returns security_id for each option contract
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


def render_smart_money_master_analysis(option_data, current_price):
    """
    Complete Master Prompt – Option Chain + Market Sentiment Analysis (14-step framework).
    Covers: OI build/close, support/resistance strength, ATM±2 snapshot,
    strike migration, smart money, sentiment, trap & breakout detection.
    """
    if option_data is None:
        return
    df_summary = option_data.get('df_summary')
    if df_summary is None or df_summary.empty:
        return

    underlying   = option_data.get('underlying', current_price or 0)
    atm_strike   = option_data.get('atm_strike')
    max_pain     = option_data.get('max_pain_strike')
    total_ce_chg = option_data.get('total_ce_change', 0)   # lakhs
    total_pe_chg = option_data.get('total_pe_change', 0)   # lakhs

    if atm_strike is None:
        return

    def gv(row, col, default=0.0):
        try:
            v = row[col]
            return float(v) if pd.notna(v) else default
        except Exception:
            return default

    LOT = 100000  # raw OI → lakhs divisor

    # Sort & locate ATM
    ds = df_summary.sort_values('Strike').reset_index(drop=True)
    atm_matches = ds[ds['Strike'] == atm_strike].index.tolist()
    ai = atm_matches[0] if atm_matches else int((ds['Strike'] - atm_strike).abs().idxmin())

    def safe_row(i):
        return ds.iloc[i] if 0 <= i < len(ds) else None

    r_m2 = safe_row(ai - 2)
    r_m1 = safe_row(ai - 1)
    r_0  = ds.iloc[ai]
    r_p1 = safe_row(ai + 1)
    r_p2 = safe_row(ai + 2)

    # ── STEP 1-2: Totals & PCR ──────────────────────────────────────────────
    total_ce_oi = ds['openInterest_CE'].sum() if 'openInterest_CE' in ds.columns else 0
    total_pe_oi = ds['openInterest_PE'].sum() if 'openInterest_PE' in ds.columns else 0
    pcr_overall = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 1.0

    max_ce_row = ds.loc[ds['openInterest_CE'].idxmax()] if 'openInterest_CE' in ds.columns else None
    max_pe_row = ds.loc[ds['openInterest_PE'].idxmax()] if 'openInterest_PE' in ds.columns else None
    max_ce_strike_val = gv(max_ce_row, 'Strike') if max_ce_row is not None else None
    max_pe_strike_val = gv(max_pe_row, 'Strike') if max_pe_row is not None else None

    # ── STEP 3: OI Activity ─────────────────────────────────────────────────
    _price_up = underlying >= st.session_state.get('_oi_prev_underlying', underlying)
    total_chg  = total_ce_chg + total_pe_chg

    if _price_up and total_pe_chg > 0 and total_pe_chg >= abs(total_ce_chg):
        oi_activity, oi_detail = "🟢 Long Build-up",   "Price ↑ + Put OI ↑ — bulls adding floor"
    elif not _price_up and total_ce_chg > 0 and total_ce_chg >= abs(total_pe_chg):
        oi_activity, oi_detail = "🔴 Short Build-up",  "Price ↓ + Call OI ↑ — bears adding resistance"
    elif _price_up and total_ce_chg < 0:
        oi_activity, oi_detail = "🟡 Short Covering",  "Price ↑ + Call OI ↓ — shorts being covered"
    elif not _price_up and total_pe_chg < 0:
        oi_activity, oi_detail = "🟠 Long Unwinding",  "Price ↓ + Put OI ↓ — longs being unwound"
    else:
        oi_activity, oi_detail = "⚪ Mixed / Neutral",  "No clear directional OI bias"

    # ── STEP 4-5: Support & Resistance Strength ─────────────────────────────
    pe_chg_atm = gv(r_0,  'changeinOpenInterest_PE')
    ce_chg_atm = gv(r_0,  'changeinOpenInterest_CE')
    pe_chg_m1  = gv(r_m1, 'changeinOpenInterest_PE') if r_m1 is not None else 0
    pe_chg_m2  = gv(r_m2, 'changeinOpenInterest_PE') if r_m2 is not None else 0
    ce_chg_p1  = gv(r_p1, 'changeinOpenInterest_CE') if r_p1 is not None else 0
    ce_chg_p2  = gv(r_p2, 'changeinOpenInterest_CE') if r_p2 is not None else 0
    pe_oi_atm  = gv(r_0,  'openInterest_PE')
    ce_oi_atm  = gv(r_0,  'openInterest_CE')
    pe_oi_m1   = gv(r_m1, 'openInterest_PE') if r_m1 is not None else 0
    ce_oi_p1   = gv(r_p1, 'openInterest_CE') if r_p1 is not None else 0

    sup_score = sum([total_pe_chg > 0, total_pe_chg > abs(total_ce_chg),
                     pe_chg_atm > 0, pe_chg_m1 > 0, pe_chg_m2 > 0, pe_oi_atm > pe_oi_m1])
    res_score = sum([total_ce_chg > 0, total_ce_chg > abs(total_pe_chg),
                     ce_chg_atm > 0, ce_chg_p1 > 0, ce_chg_p2 > 0, ce_oi_atm > ce_oi_p1])

    _sup_labels = ["❌ Very Weak", "⚠️ Weak", "🟡 Moderate", "✅ Strong", "🔒 Very Strong", "💪 Dominant"]
    _res_labels = ["❌ Very Weak", "⚠️ Weak", "🟡 Moderate", "🔴 Strong", "🚧 Very Strong", "🚨 Dominant"]
    support_strength    = _sup_labels[min(sup_score, 5)]
    resistance_strength = _res_labels[min(res_score, 5)]

    # ── STEP 7-8: Strike Migration ───────────────────────────────────────────
    if ce_chg_atm < 0 and (ce_chg_p1 > 0 or ce_chg_p2 > 0):
        resist_migration = "⬆️ Shifting Higher (Bullish)"
    elif ce_chg_atm > 0 and ce_chg_p1 < 0:
        resist_migration = "⬇️ Shifting Lower (Bearish)"
    else:
        resist_migration = "↔️ Stable"

    if pe_chg_atm < 0 and (pe_chg_m1 > 0 or pe_chg_m2 > 0):
        support_migration = "⬇️ Shifting Lower (Bearish)"
    elif pe_chg_atm > 0 and pe_chg_m1 < 0:
        support_migration = "⬆️ Shifting Higher (Bullish)"
    else:
        support_migration = "↔️ Stable"

    # ── STEP 9: Smart Money ──────────────────────────────────────────────────
    smart_signals = []
    for side, pos_lbl, neg_lbl in [
        ('CE', '🔴 Institutional Call writing', '🟡 Institutional Call covering'),
        ('PE', '🟢 Institutional Put writing',  '🟠 Institutional Put covering'),
    ]:
        col = f'changeinOpenInterest_{side}'
        if col in ds.columns:
            avg_abs = ds[col].abs().mean()
            spikes  = ds[ds[col].abs() > avg_abs * 3]
            for _, srow in spikes.head(3).iterrows():
                lbl = pos_lbl if srow[col] > 0 else neg_lbl
                smart_signals.append(f"{lbl} at ₹{int(srow['Strike'])}")
    if not smart_signals:
        smart_signals = ["✅ No major institutional spikes detected"]

    # ── STEP 10: Sentiment Score ─────────────────────────────────────────────
    ss = 0
    ss += 2 if pcr_overall > 1.2 else (1 if pcr_overall > 0.9 else (-2 if pcr_overall < 0.7 else -1))
    ss += min(sup_score - 2, 2) - min(res_score - 2, 2)
    if "Long Build"  in oi_activity: ss += 1
    if "Short Build" in oi_activity: ss -= 1
    if "Short Cover" in oi_activity: ss += 1
    if "Long Unwind" in oi_activity: ss -= 1
    if "Higher" in resist_migration: ss += 1
    if "Lower"  in support_migration: ss -= 1

    if   ss >= 6:   sentiment, s_clr = "🟢🔥 Strong Bullish", "#00ff88"
    elif ss >= 4:   sentiment, s_clr = "🟢 Bullish",          "#00cc66"
    elif ss >= 2:   sentiment, s_clr = "🟡 Mildly Bullish",   "#aacc00"
    elif ss == 1:   sentiment, s_clr = "🟡 Slight Bullish",   "#cccc00"
    elif ss == 0:   sentiment, s_clr = "⚪ Neutral / Range",   "#aaaaaa"
    elif ss == -1:  sentiment, s_clr = "🟠 Slight Bearish",   "#cc9900"
    elif ss >= -3:  sentiment, s_clr = "🟠 Mildly Bearish",   "#cc6600"
    elif ss >= -5:  sentiment, s_clr = "🔴 Bearish",          "#cc3333"
    else:           sentiment, s_clr = "🔴🔥 Strong Bearish", "#ff2222"

    # ── STEP 11: Trap Detection ──────────────────────────────────────────────
    trap_signals, trap_pts = [], 0
    if total_pe_chg > 0 and total_ce_chg > total_pe_chg:
        trap_signals.append("⚠️ Bull Trap: PUT OI building but CALL OI growth stronger"); trap_pts += 2
    if total_ce_chg > 0 and total_pe_chg > total_ce_chg:
        trap_signals.append("⚠️ Bear Trap: CALL OI building but PUT OI growth stronger"); trap_pts += 2
    avg_oi = (total_ce_oi + total_pe_oi) / max(len(ds), 1)
    if (ce_oi_atm + pe_oi_atm) > avg_oi * 2:
        trap_signals.append("📍 Gamma Trap: High OI concentration at ATM — price likely pinned"); trap_pts += 2
    if ce_chg_atm < 0 and pe_chg_atm < 0:
        trap_signals.append("⚡ Fake Breakout Risk: Both CE & PE OI unwinding at ATM"); trap_pts += 1
    if not trap_signals:
        trap_signals = ["✅ No trap patterns detected"]
    trap_prob = "High" if trap_pts >= 4 else ("Medium" if trap_pts >= 2 else "Low")

    # ── STEP 12: Breakout Detection ──────────────────────────────────────────
    bo_signals, bo_pts = [], 0
    if ce_chg_atm < 0: bo_signals.append("📈 ATM Call OI unwinding — resistance weakening"); bo_pts += 1
    if pe_chg_atm < 0: bo_signals.append("📉 ATM Put OI unwinding — support weakening");    bo_pts += 1
    if "Higher" in resist_migration: bo_signals.append("⬆️ Resistance shifting higher — bullish breakout setup"); bo_pts += 2
    if "Lower"  in support_migration: bo_signals.append("⬇️ Support shifting lower — bearish breakdown setup");    bo_pts += 2
    if total_ce_oi + total_pe_oi > 0:
        raw_total = (total_ce_oi + total_pe_oi) / LOT
        if raw_total > 0 and abs(total_chg) / raw_total > 0.05:
            bo_signals.append("💥 Large total OI movement (>5%) — breakout energy building"); bo_pts += 1
    if not bo_signals:
        bo_signals = ["↔️ No breakout signals — range-bound expected"]
    bo_prob = "High" if bo_pts >= 4 else ("Medium" if bo_pts >= 2 else "Low")
    bo_dir  = ("⬆️ Upside"   if "Higher" in resist_migration and total_pe_chg > total_ce_chg else
               "⬇️ Downside" if "Lower"  in support_migration and total_ce_chg > total_pe_chg else
               "↔️ No clear direction")

    # ══════════════════════════════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🧠 Smart Money & Market Sentiment Analysis")

    # Row 1 — Sentiment · OI Activity · PCR
    m1, m2, m3 = st.columns(3)
    oi_clr  = ("#00ff88" if "Long Build"  in oi_activity else
               "#ff4444" if "Short Build" in oi_activity else
               "#FFD700" if "Cover"       in oi_activity else "#ff8800")
    pcr_clr = "#00cc66" if pcr_overall > 1.1 else ("#ff4444" if pcr_overall < 0.8 else "#FFD700")
    pcr_lbl = "Bullish" if pcr_overall > 1.1 else ("Bearish" if pcr_overall < 0.8 else "Neutral")

    def _card(col_widget, title, value, subtitle, clr):
        with col_widget:
            st.markdown(
                f"<div style='background:#111827;padding:12px;border-radius:8px;"
                f"border-left:4px solid {clr};margin-bottom:8px'>"
                f"<span style='font-size:0.75em;color:#888;text-transform:uppercase'>{title}</span><br>"
                f"<span style='font-size:1.05em;color:{clr};font-weight:bold'>{value}</span><br>"
                f"<span style='font-size:0.8em;color:#666'>{subtitle}</span></div>",
                unsafe_allow_html=True
            )

    _card(m1, "MARKET SENTIMENT",  sentiment,   f"Score: {ss:+d}", s_clr)
    _card(m2, "OI ACTIVITY",       oi_activity, oi_detail,         oi_clr)
    _card(m3, "PCR (OVERALL)",     f"{pcr_overall:.2f} · {pcr_lbl}", "", pcr_clr)

    st.markdown("")

    # Row 2 — Support & Resistance Strength
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("#### 🟢 Support Strength")
        st.progress(min(sup_score / 6, 1.0), text=support_strength)
        st.caption(f"Total PE ΔOI: {total_pe_chg:+.1f}L | ATM: {pe_chg_atm/LOT*100:+.1f}L | "
                   f"ATM-1: {pe_chg_m1/LOT*100:+.1f}L | ATM-2: {pe_chg_m2/LOT*100:+.1f}L")
    with sc2:
        st.markdown("#### 🔴 Resistance Strength")
        st.progress(min(res_score / 6, 1.0), text=resistance_strength)
        st.caption(f"Total CE ΔOI: {total_ce_chg:+.1f}L | ATM: {ce_chg_atm/LOT*100:+.1f}L | "
                   f"ATM+1: {ce_chg_p1/LOT*100:+.1f}L | ATM+2: {ce_chg_p2/LOT*100:+.1f}L")

    # ATM ±2 OI Snapshot
    st.markdown("#### 📊 ATM ±2 Strike OI Snapshot")
    snap_rows = []
    for lbl, row in [("ATM-2", r_m2), ("ATM-1", r_m1), ("ATM ★", r_0), ("ATM+1", r_p1), ("ATM+2", r_p2)]:
        if row is None:
            continue
        snap_rows.append({
            "Zone":       lbl,
            "Strike":     int(gv(row, 'Strike')),
            "CE OI(L)":   f"{gv(row,'openInterest_CE')/LOT:.2f}",
            "CE ΔOI(L)":  f"{gv(row,'changeinOpenInterest_CE')/LOT:+.2f}",
            "PE OI(L)":   f"{gv(row,'openInterest_PE')/LOT:.2f}",
            "PE ΔOI(L)":  f"{gv(row,'changeinOpenInterest_PE')/LOT:+.2f}",
            "PCR":        f"{gv(row,'PCR',1.0):.2f}" if 'PCR' in ds.columns else "—",
        })
    if snap_rows:
        st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)

    # Strike Migration + Smart Money
    mig1, mig2 = st.columns(2)
    with mig1:
        st.markdown("#### 🔄 Strike Migration")
        r_clr2 = "#00cc66" if "Higher" in resist_migration else ("#ff4444" if "Lower" in resist_migration else "#888")
        s_clr2 = "#ff4444" if "Lower"  in support_migration else ("#00cc66" if "Higher" in support_migration else "#888")
        st.markdown(f"<b style='color:{r_clr2}'>Resistance:</b> {resist_migration}<br>"
                    f"<b style='color:{s_clr2}'>Support:</b> {support_migration}", unsafe_allow_html=True)
    with mig2:
        st.markdown("#### 🏦 Smart Money Signals")
        for sig in smart_signals[:5]:
            st.markdown(f"- {sig}")

    # Trap + Breakout
    tc1, tc2 = st.columns(2)
    trap_clr = "#ff4444" if trap_prob == "High" else ("#FFD700" if trap_prob == "Medium" else "#00cc66")
    bo_clr   = "#00ff88" if bo_prob == "High"   else ("#FFD700" if bo_prob == "Medium"   else "#888")
    with tc1:
        st.markdown("#### 🪤 Trap Detection")
        st.markdown(f"**Probability:** <span style='color:{trap_clr};font-size:1.1em'>{trap_prob}</span>",
                    unsafe_allow_html=True)
        for sig in trap_signals:
            st.markdown(f"- {sig}")
    with tc2:
        st.markdown("#### 🚀 Breakout Detection")
        st.markdown(f"**Probability:** <span style='color:{bo_clr};font-size:1.1em'>{bo_prob}</span>"
                    f" &nbsp; {bo_dir}", unsafe_allow_html=True)
        for sig in bo_signals[:4]:
            st.markdown(f"- {sig}")

    # Final Summary Table
    st.markdown("---")
    st.markdown("### 📋 Final Analysis Summary")
    fa1, fa2 = st.columns(2)
    with fa1:
        st.markdown(f"""
| Metric | Value |
|---|---|
| **Sentiment** | {sentiment} |
| **OI Activity** | {oi_activity} |
| **Support Strength** | {support_strength} |
| **Resistance Strength** | {resistance_strength} |
| **PCR Overall** | {pcr_overall:.2f} ({pcr_lbl}) |
""")
    with fa2:
        st.markdown(f"""
| Metric | Value |
|---|---|
| **Resistance Migration** | {resist_migration} |
| **Support Migration** | {support_migration} |
| **Trap Probability** | {trap_prob} |
| **Breakout Probability** | {bo_prob} ({bo_dir}) |
| **Max Pain** | {'₹' + str(int(max_pain)) if max_pain else 'N/A'} |
""")

    # Key Levels
    kl1, kl2, kl3 = st.columns(3)
    with kl1:
        if max_pe_strike_val:
            st.success(f"🟢 Key Support: ₹{int(max_pe_strike_val)}")
    with kl2:
        if max_pain:
            st.info(f"🎯 Max Pain: ₹{int(max_pain)}")
    with kl3:
        if max_ce_strike_val:
            st.error(f"🔴 Key Resistance: ₹{int(max_ce_strike_val)}")

    if trap_prob == "High" or bo_prob == "High":
        st.warning(
            "⚠️ **Risk Warning:** "
            + ("High trap probability — avoid chasing moves. " if trap_prob == "High" else "")
            + ("High breakout energy — use strict stop-losses." if bo_prob == "High" else "")
        )


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

def fetch_index_metrics(api, security_id, exchange_segment, interval="1", days_back=1):
    """Fetch intraday OHLCV + LTP for an index and return a metrics dict."""
    try:
        data = api.get_intraday_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument="INDEX",
            interval=interval,
            days_back=days_back
        )
        df = process_candle_data(data, interval) if data else pd.DataFrame()
        ltp_resp = api.get_ltp_data(security_id, exchange_segment)
        current = None
        if ltp_resp and 'data' in ltp_resp:
            for _exc, _d in ltp_resp['data'].items():
                for _sid, _pd in _d.items():
                    current = _pd.get('last_price')
                    break
        if current is None and not df.empty:
            current = float(df['close'].iloc[-1])
        if df.empty or current is None:
            return None
        prev_close = float(df['close'].iloc[-2]) if len(df) > 1 else float(df['close'].iloc[0])
        change = current - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0
        return {
            'current': current,
            'change': change,
            'change_pct': change_pct,
            'day_high': float(df['high'].max()),
            'day_low': float(df['low'].min()),
            'day_open': float(df['open'].iloc[0]),
            'volume': int(df['volume'].sum()),
        }
    except Exception:
        return None


def show_market_overview(api, interval="1", days_back=1):
    """Render NIFTY 50 and SENSEX metrics in a tabbed table at the top of the page."""
    st.markdown("### Market Overview")

    tab_nifty, tab_sensex = st.tabs(["📈 NIFTY 50", "📊 SENSEX"])

    def _render_metrics(m, label):
        if m is None:
            st.warning(f"Could not fetch {label} data.")
            return
        sign = "+" if m['change'] >= 0 else ""
        arrow = "▲" if m['change'] >= 0 else "▼"
        chg_color = "#00cc66" if m['change'] >= 0 else "#ff4444"
        cols = st.columns(5)
        cols[0].metric("Current Price", f"₹{m['current']:,.2f}")
        cols[1].metric(
            "Change",
            f"{sign}{m['change']:,.2f}",
            delta=f"{sign}{m['change_pct']:.2f}%"
        )
        cols[2].metric("Day High", f"₹{m['day_high']:,.2f}")
        cols[3].metric("Day Low",  f"₹{m['day_low']:,.2f}")
        cols[4].metric("Day Open", f"₹{m['day_open']:,.2f}")

    with tab_nifty:
        nifty_m = fetch_index_metrics(api, "13", "IDX_I", interval, days_back)
        _render_metrics(nifty_m, "NIFTY 50")

    with tab_sensex:
        sensex_m = fetch_index_metrics(api, SENSEX_SCRIP_ID, SENSEX_EXCHANGE_SEG, interval, days_back)
        _render_metrics(sensex_m, "SENSEX")

    st.markdown("---")


# =====================================================================
# BANK NIFTY DASHBOARD — Helper Calculations
# =====================================================================

def _bn_calc_rsi(series: pd.Series, period: int = 14) -> float:
    """Wilder RSI – returns latest value."""
    if len(series) < period + 1:
        return float("nan")
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1])


def _bn_calc_ema(series: pd.Series, period: int = 20) -> float:
    if len(series) < period:
        return float("nan")
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def _bn_calc_vwap(df: pd.DataFrame) -> float:
    """Session VWAP (cumulative from first bar)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, np.nan)
    vwap = (tp * vol).cumsum() / vol.cumsum()
    return float(vwap.iloc[-1])


def _bn_calc_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 2.0):
    """Returns (supertrend_value, direction) where direction=-1 up-trend, 1 down-trend."""
    if len(df) < period + 1:
        return float("nan"), 0
    high = df["High"].values
    low  = df["Low"].values
    close = df["Close"].values
    # ATR
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(abs(high[1:] - close[:-1]),
                    abs(low[1:]  - close[:-1])))
    atr = np.zeros(len(close))
    atr[period] = tr[:period].mean()
    for i in range(period + 1, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
    hl2 = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    supertrend = np.zeros(len(close))
    direction  = np.zeros(len(close), dtype=int)
    direction[period] = 1
    for i in range(period + 1, len(close)):
        if close[i - 1] > supertrend[i - 1]:
            lower[i] = max(lower[i], lower[i - 1])
        if close[i - 1] < supertrend[i - 1]:
            upper[i] = min(upper[i], upper[i - 1])
        if close[i] > upper[i - 1]:
            direction[i] = -1
        elif close[i] < lower[i - 1]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
        supertrend[i] = lower[i] if direction[i] == -1 else upper[i]
    return float(supertrend[-1]), int(direction[-1])


def _bn_calc_adx_dmi(df: pd.DataFrame, period: int = 13, smooth: int = 8):
    """Returns (adx, di_plus, di_minus) latest values."""
    if len(df) < period + smooth + 5:
        return float("nan"), float("nan"), float("nan")
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n = len(close)
    dm_plus  = np.zeros(n)
    dm_minus = np.zeros(n)
    tr_arr   = np.zeros(n)
    for i in range(1, n):
        up   = high[i]  - high[i-1]
        down = low[i-1] - low[i]
        dm_plus[i]  = up   if up > down and up > 0   else 0
        dm_minus[i] = down if down > up and down > 0 else 0
        tr_arr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i]  - close[i-1]))
    def _wilder(arr, p):
        out = np.zeros(n)
        out[p] = arr[1:p+1].sum()
        for i in range(p + 1, n):
            out[i] = out[i-1] - out[i-1] / p + arr[i]
        return out
    atr_w  = _wilder(tr_arr,   period)
    dmp_w  = _wilder(dm_plus,  period)
    dmm_w  = _wilder(dm_minus, period)
    dip = np.where(atr_w > 0, 100 * dmp_w / atr_w, 0)
    dim = np.where(atr_w > 0, 100 * dmm_w / atr_w, 0)
    dx  = np.where((dip + dim) > 0, 100 * abs(dip - dim) / (dip + dim), 0)
    adx = np.zeros(n)
    adx[period * 2] = dx[period:period*2+1].mean()
    for i in range(period * 2 + 1, n):
        adx[i] = (adx[i-1] * (smooth - 1) + dx[i]) / smooth
    return float(adx[-1]), float(dip[-1]), float(dim[-1])


@st.cache_data(ttl=60, show_spinner=False)
def _bn_fetch_yf_data(tf1: str = "15m", tf2: str = "60m"):
    """Fetch daily + TF1 + TF2 OHLCV for all BN dashboard symbols via yfinance."""
    if not _YF_AVAILABLE:
        return None
    all_tickers = [t["yf"] for t in BN_DASH_TICKERS] + [t["yf"] for t in BN_MACRO_TICKERS]
    try:
        daily = yf.download(all_tickers, period="5d", interval="1d",
                            group_by="ticker", auto_adjust=True, progress=False, threads=True)
        intra15 = yf.download(all_tickers, period="2d", interval="15m",
                              group_by="ticker", auto_adjust=True, progress=False, threads=True)
        intra60 = yf.download(all_tickers, period="5d", interval="60m",
                              group_by="ticker", auto_adjust=True, progress=False, threads=True)
        return {"daily": daily, "tf1": intra15, "tf2": intra60}
    except Exception:
        return None


def _bn_extract_ticker(data_dict, yf_ticker, key="daily"):
    """Extract single-ticker DataFrame from multi-ticker yfinance download."""
    raw = data_dict.get(key)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            df = raw[yf_ticker].dropna(how="all")
        except KeyError:
            return pd.DataFrame()
    else:
        df = raw.dropna(how="all")
    return df


def _bn_ticker_metrics(data, yf_ticker, tf1_key="tf1", tf2_key="tf2"):
    """Compute LTP, daily %chg, TF1 %chg, TF2 %chg for one ticker."""
    daily = _bn_extract_ticker(data, yf_ticker, "daily")
    tf1   = _bn_extract_ticker(data, yf_ticker, tf1_key)
    tf2   = _bn_extract_ticker(data, yf_ticker, tf2_key)
    if daily.empty or len(daily) < 2:
        return None
    ltp        = float(daily["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2])
    chng       = ltp - prev_close
    pct_d      = (chng / prev_close * 100) if prev_close else 0.0
    # TF1: % change from last completed bar's close
    pct_tf1 = 0.0
    if not tf1.empty and len(tf1) >= 2:
        ref = float(tf1["Close"].iloc[-2])
        pct_tf1 = (ltp - ref) / ref * 100 if ref else 0.0
    # TF2: % change from last completed bar's close
    pct_tf2 = 0.0
    if not tf2.empty and len(tf2) >= 2:
        ref = float(tf2["Close"].iloc[-2])
        pct_tf2 = (ltp - ref) / ref * 100 if ref else 0.0
    return {"ltp": ltp, "chng": chng, "pct_d": pct_d,
            "pct_tf1": pct_tf1, "pct_tf2": pct_tf2}


def _bn_indicator_metrics(data, yf_ticker, ema_period=20, rsi_period=14, tf_keys=("daily", "tf1", "tf2")):
    """Compute VWAP, EMA, SuperTrend, RSI, ADX, DI+, DI- for 3 timeframes."""
    results = {}
    for key in tf_keys:
        df = _bn_extract_ticker(data, yf_ticker, key)
        if df.empty or len(df) < 20:
            results[key] = None
            continue
        close = df["Close"]
        ltp   = float(close.iloc[-1])
        vwap  = _bn_calc_vwap(df)
        ema   = _bn_calc_ema(close, ema_period)
        rsi   = _bn_calc_rsi(close, rsi_period)
        st_val, st_dir = _bn_calc_supertrend(df)
        adx, dip, dim  = _bn_calc_adx_dmi(df)
        results[key] = {
            "ltp": ltp, "vwap": vwap, "ema": ema,
            "rsi": rsi, "st": st_val, "st_dir": st_dir,
            "adx": adx, "dip": dip, "dim": dim,
        }
    return results


def _bn_color_cell(val, is_positive: bool, inverse: bool = False) -> str:
    """Return CSS background color string."""
    bull = "#1a472a"   # dark green
    bear = "#7f1d1d"   # dark red
    neut = "#374151"   # gray
    if is_positive:
        return bull if not inverse else bear
    return bear if not inverse else bull


def _bn_trend_label(pct: float, inverse: bool = False) -> str:
    if abs(pct) < 0.05:
        return "🟡"
    return ("🔴" if inverse else "🟢") if pct > 0 else ("🟢" if inverse else "🔴")


def show_bn_dashboard(nifty_df: pd.DataFrame = None, interval: str = "1"):
    """Full Bank Nifty Dashboard — Pine Script converted to Python/Streamlit."""
    st.markdown("### 📊 Bank Nifty Multi-Ticker Dashboard")

    if not _YF_AVAILABLE:
        st.warning("yfinance not available. Install it to use the BankNifty Dashboard.")
        return

    with st.spinner("Fetching market data…"):
        data = _bn_fetch_yf_data(tf1="15m", tf2="60m")

    if data is None:
        st.error("Could not fetch market data. Check your network connection.")
        return

    # ── Build Ticker Table ─────────────────────────────────────────────
    rows = []
    sent_score = 0.0
    for t in BN_DASH_TICKERS:
        m = _bn_ticker_metrics(data, t["yf"])
        if m is None:
            rows.append({
                "Sentiment": "—", "Symbol": t["name"],
                "LTP": "—", "Chng": "—", "%Chng": "—",
                "TF1 15m%": "—", "TF2 60m%": "—",
                "Weight": f"{int(t['weight'])}%",
                "_pct_d": 0, "_pct_tf1": 0, "_pct_tf2": 0, "_inverse": False,
            })
            continue
        sent_score += m["pct_d"] * t["weight"]
        rows.append({
            "Sentiment": _bn_trend_label(m["pct_d"]),
            "Symbol":    t["name"],
            "LTP":       f"{m['ltp']:,.2f}",
            "Chng":      f"{m['chng']:+.2f}",
            "%Chng":     f"{m['pct_d']:+.2f}%",
            "TF1 15m%":  f"{m['pct_tf1']:+.2f}%",
            "TF2 60m%":  f"{m['pct_tf2']:+.2f}%",
            "Weight":    f"{int(t['weight'])}%",
            "_pct_d": m["pct_d"], "_pct_tf1": m["pct_tf1"],
            "_pct_tf2": m["pct_tf2"], "_inverse": False,
        })
    sent_score /= 100.0

    # Macro penalty
    macro_penalty = 0.0
    macro_rows = []
    macro_weights = [0.5, 0.4, 0.3]
    for i, t in enumerate(BN_MACRO_TICKERS):
        m = _bn_ticker_metrics(data, t["yf"])
        if m is None:
            macro_rows.append({
                "Sentiment": "—", "Symbol": t["name"],
                "LTP": "—", "Chng": "—", "%Chng": "—",
                "TF1 15m%": "—", "TF2 60m%": "—",
                "Weight": "Inverse",
                "_pct_d": 0, "_pct_tf1": 0, "_pct_tf2": 0, "_inverse": True,
            })
            continue
        macro_penalty += m["pct_d"] * macro_weights[i]
        macro_rows.append({
            "Sentiment": _bn_trend_label(m["pct_d"], inverse=True),
            "Symbol":    t["name"],
            "LTP":       f"{m['ltp']:,.4f}" if t["yf"] == "USDINR=X" else f"{m['ltp']:,.2f}",
            "Chng":      f"{m['chng']:+.4f}" if t["yf"] == "USDINR=X" else f"{m['chng']:+.2f}",
            "%Chng":     f"{m['pct_d']:+.2f}%",
            "TF1 15m%":  f"{m['pct_tf1']:+.2f}%",
            "TF2 60m%":  f"{m['pct_tf2']:+.2f}%",
            "Weight":    "Inverse",
            "_pct_d": m["pct_d"], "_pct_tf1": m["pct_tf1"],
            "_pct_tf2": m["pct_tf2"], "_inverse": True,
        })

    final_score = sent_score - macro_penalty * 10
    if final_score > 3.0:
        sent_text, sent_bg = "🟢 BULLISH", "#14532d"
    elif final_score < -3.0:
        sent_text, sent_bg = "🔴 BEARISH", "#7f1d1d"
    else:
        sent_text, sent_bg = "🟡 NEUTRAL", "#374151"

    # Overall sentiment header row (injected at top)
    header_row = {
        "Sentiment": sent_text, "Symbol": "NIFTY SCORE",
        "LTP": f"{final_score:.2f}", "Chng": "", "%Chng": "",
        "TF1 15m%": "", "TF2 60m%": "", "Weight": "100%",
        "_pct_d": final_score, "_pct_tf1": 0, "_pct_tf2": 0, "_inverse": False,
    }
    display_cols = ["Sentiment", "Symbol", "LTP", "Chng", "%Chng", "TF1 15m%", "TF2 60m%", "Weight"]

    all_rows = [header_row] + rows

    def _style_ticker_table(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for i, row in df.iterrows():
            inv = row.get("_inverse", False)
            for col, pct_key in [("%Chng", "_pct_d"), ("TF1 15m%", "_pct_tf1"), ("TF2 60m%", "_pct_tf2")]:
                val = row.get(pct_key, 0)
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    fval = 0.0
                pos = fval >= 0
                bg = _bn_color_cell(fval, pos, inv)
                styles.at[i, col] = f"background-color: {bg}; color: white"
            styles.at[i, "Sentiment"] = "background-color: #1f2937; color: white; font-size:1.1em"
            styles.at[i, "Symbol"] = "background-color: #111827; color: white; font-weight: bold"
        styles.at[0, "LTP"] = f"background-color: {sent_bg}; color: white; font-weight: bold"
        styles.at[0, "Sentiment"] = f"background-color: {sent_bg}; color: white; font-weight: bold"
        return styles

    df_display = pd.DataFrame(all_rows)
    df_styled = df_display[display_cols + ["_pct_d", "_pct_tf1", "_pct_tf2", "_inverse"]].copy()
    df_final = df_styled.style.apply(lambda _: _style_ticker_table(df_styled), axis=None)\
                               .hide(axis="index")\
                               .hide(["_pct_d", "_pct_tf1", "_pct_tf2", "_inverse"], axis="columns")

    st.dataframe(df_final, use_container_width=True, height=420)

    # ── Macro Section ──────────────────────────────────────────────────
    st.markdown("#### ⚠ Macro Signals (Inverse — Up = Bearish for Market)")
    macro_display = pd.DataFrame(macro_rows)
    if not macro_display.empty:
        macro_styled = macro_display[display_cols + ["_pct_d", "_pct_tf1", "_pct_tf2", "_inverse"]].copy()

        def _style_macro(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for i, row in df.iterrows():
                for col, pct_key in [("%Chng", "_pct_d"), ("TF1 15m%", "_pct_tf1"), ("TF2 60m%", "_pct_tf2")]:
                    val = row.get(pct_key, 0)
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        fval = 0.0
                    bg = _bn_color_cell(fval, fval >= 0, inverse=True)
                    styles.at[i, col] = f"background-color: {bg}; color: white"
                styles.at[i, "Sentiment"] = "background-color: #1f2937; color: #fbbf24"
                styles.at[i, "Symbol"] = "background-color: #111827; color: #fbbf24; font-weight: bold"
                styles.at[i, "Weight"] = "background-color: #111827; color: #fbbf24"
            return styles

        st.dataframe(
            macro_styled.style.apply(lambda _: _style_macro(macro_styled), axis=None)
                               .hide(axis="index")
                               .hide(["_pct_d", "_pct_tf1", "_pct_tf2", "_inverse"], axis="columns"),
            use_container_width=True, height=160
        )

    # ── Indicator Dashboard (NIFTY 50 across 3 TFs) ───────────────────
    st.markdown("#### 📐 Indicator Dashboard — NIFTY 50 (Current | 15m | 60m)")
    nifty_yf = "^NSEI"
    ind = _bn_indicator_metrics(data, nifty_yf, tf_keys=("daily", "tf1", "tf2"))

    def _ind_val(key, field, fmt=".1f"):
        m = ind.get(key)
        if m is None or m.get(field) is None:
            return "—"
        v = m[field]
        if isinstance(v, float) and np.isnan(v):
            return "—"
        return f"{v:{fmt}}"

    def _ind_dir(key, close_field, ref_field):
        m = ind.get(key)
        if m is None:
            return "—"
        ltp = m.get("ltp")
        ref = m.get(ref_field)
        if ltp is None or ref is None:
            return "—"
        return "UP" if ltp > ref else "Down"

    def _st_dir_text(key):
        m = ind.get(key)
        if m is None:
            return "—"
        ltp, st_val, st_dir = m.get("ltp"), m.get("st"), m.get("st_dir")
        if ltp is None or st_val is None:
            return "—"
        return "UP" if st_dir == -1 else "Down"

    def _adx_text(key):
        m = ind.get(key)
        if m is None:
            return "—"
        adx, dip, dim = m.get("adx", 0), m.get("dip", 0), m.get("dim", 0)
        if isinstance(adx, float) and np.isnan(adx):
            return "—"
        if adx >= 25 and dip > dim:
            return "Strong UP"
        elif adx >= 25 and dip <= dim:
            return "Strong Down"
        elif adx >= 20 and dip > dim:
            return "UP"
        elif adx >= 20 and dip <= dim:
            return "Down"
        return "Neutral"

    ind_rows = [
        {"Indicator": "VWAP",
         "Value (Curr)":  _ind_val("daily", "vwap", ",.0f"),
         "Trend (Curr)":  _ind_dir("daily", "ltp", "vwap"),
         "TF1 15m":       _ind_val("tf1",   "vwap", ",.0f"),
         "TF2 60m":       _ind_val("tf2",   "vwap", ",.0f"),
         "_up_curr":  (ind.get("daily") or {}).get("ltp", 0) > (ind.get("daily") or {}).get("vwap", 0),
         "_up_tf1":   (ind.get("tf1")   or {}).get("ltp", 0) > (ind.get("tf1")   or {}).get("vwap", 0),
         "_up_tf2":   (ind.get("tf2")   or {}).get("ltp", 0) > (ind.get("tf2")   or {}).get("vwap", 0),
         "_field": "vwap"},
        {"Indicator": "EMA 20",
         "Value (Curr)":  _ind_val("daily", "ema", ",.0f"),
         "Trend (Curr)":  _ind_dir("daily", "ltp", "ema"),
         "TF1 15m":       _ind_val("tf1",   "ema", ",.0f"),
         "TF2 60m":       _ind_val("tf2",   "ema", ",.0f"),
         "_up_curr":  (ind.get("daily") or {}).get("ltp", 0) > (ind.get("daily") or {}).get("ema", 0),
         "_up_tf1":   (ind.get("tf1")   or {}).get("ltp", 0) > (ind.get("tf1")   or {}).get("ema", 0),
         "_up_tf2":   (ind.get("tf2")   or {}).get("ltp", 0) > (ind.get("tf2")   or {}).get("ema", 0),
         "_field": "ema"},
        {"Indicator": "SuperTrend",
         "Value (Curr)":  _ind_val("daily", "st", ",.0f"),
         "Trend (Curr)":  _st_dir_text("daily"),
         "TF1 15m":       _ind_val("tf1",   "st", ",.0f"),
         "TF2 60m":       _ind_val("tf2",   "st", ",.0f"),
         "_up_curr": (ind.get("daily") or {}).get("st_dir", 1) == -1,
         "_up_tf1":  (ind.get("tf1")   or {}).get("st_dir", 1) == -1,
         "_up_tf2":  (ind.get("tf2")   or {}).get("st_dir", 1) == -1,
         "_field": "st"},
        {"Indicator": "RSI (14)",
         "Value (Curr)":  _ind_val("daily", "rsi", ".1f"),
         "Trend (Curr)":  "UP" if float(_ind_val("daily", "rsi", ".1f").replace("—","50")) > 50 else "Down",
         "TF1 15m":       _ind_val("tf1",   "rsi", ".1f"),
         "TF2 60m":       _ind_val("tf2",   "rsi", ".1f"),
         "_up_curr": float(_ind_val("daily", "rsi", ".1f").replace("—","50")) > 50,
         "_up_tf1":  float(_ind_val("tf1",   "rsi", ".1f").replace("—","50")) > 50,
         "_up_tf2":  float(_ind_val("tf2",   "rsi", ".1f").replace("—","50")) > 50,
         "_field": "rsi"},
        {"Indicator": "ADX (13)",
         "Value (Curr)":  _ind_val("daily", "adx", ".1f"),
         "Trend (Curr)":  _adx_text("daily"),
         "TF1 15m":       _ind_val("tf1",   "adx", ".1f"),
         "TF2 60m":       _ind_val("tf2",   "adx", ".1f"),
         "_up_curr": "UP" in _adx_text("daily"),
         "_up_tf1":  "UP" in _adx_text("tf1"),
         "_up_tf2":  "UP" in _adx_text("tf2"),
         "_field": "adx"},
        {"Indicator": "DI+",
         "Value (Curr)":  _ind_val("daily", "dip", ".1f"),
         "Trend (Curr)":  "UP" if float(_ind_val("daily","dip",".1f").replace("—","0")) >= 25 else "-",
         "TF1 15m":       _ind_val("tf1",   "dip", ".1f"),
         "TF2 60m":       _ind_val("tf2",   "dip", ".1f"),
         "_up_curr": float(_ind_val("daily","dip",".1f").replace("—","0")) >= 25,
         "_up_tf1":  float(_ind_val("tf1",  "dip",".1f").replace("—","0")) >= 25,
         "_up_tf2":  float(_ind_val("tf2",  "dip",".1f").replace("—","0")) >= 25,
         "_field": "dip"},
        {"Indicator": "DI-",
         "Value (Curr)":  _ind_val("daily", "dim", ".1f"),
         "Trend (Curr)":  "Down" if float(_ind_val("daily","dim",".1f").replace("—","0")) >= 25 else "-",
         "TF1 15m":       _ind_val("tf1",   "dim", ".1f"),
         "TF2 60m":       _ind_val("tf2",   "dim", ".1f"),
         "_up_curr": float(_ind_val("daily","dim",".1f").replace("—","0")) < 25,
         "_up_tf1":  float(_ind_val("tf1",  "dim",".1f").replace("—","0")) < 25,
         "_up_tf2":  float(_ind_val("tf2",  "dim",".1f").replace("—","0")) < 25,
         "_field": "dim"},
    ]

    ind_cols = ["Indicator", "Value (Curr)", "Trend (Curr)", "TF1 15m", "TF2 60m"]
    df_ind = pd.DataFrame(ind_rows)

    def _style_ind_table(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for i, row in df.iterrows():
            styles.at[i, "Indicator"] = "background-color: #111827; color: white; font-weight: bold"
            for col, up_key in [("Value (Curr)", "_up_curr"), ("Trend (Curr)", "_up_curr"),
                                 ("TF1 15m", "_up_tf1"), ("TF2 60m", "_up_tf2")]:
                up = bool(row.get(up_key, False))
                styles.at[i, col] = f"background-color: {'#1a472a' if up else '#7f1d1d'}; color: white"
        return styles

    st.dataframe(
        df_ind[ind_cols + ["_up_curr", "_up_tf1", "_up_tf2"]].style
              .apply(lambda _: _style_ind_table(df_ind[ind_cols + ["_up_curr","_up_tf1","_up_tf2"]]), axis=None)
              .hide(axis="index")
              .hide(["_up_curr", "_up_tf1", "_up_tf2"], axis="columns"),
        use_container_width=True, height=310
    )


# =====================================================================
# FUTURES MARKET ANALYSIS ENGINE
# =====================================================================

def _fae_calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def _fae_calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _fae_calc_atr(df, period=14):
    hi, lo, cl = df['high'], df['low'], df['close']
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _fae_supertrend(df, period=10, mult=3.0):
    atr = _fae_calc_atr(df, period)
    mid = (df['high'] + df['low']) / 2
    upper = mid + mult * atr
    lower = mid - mult * atr
    st = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)
    for i in range(1, len(df)):
        prev_upper = upper.iloc[i-1]
        prev_lower = lower.iloc[i-1]
        curr_close = df['close'].iloc[i]
        upper.iloc[i] = min(upper.iloc[i], prev_upper) if curr_close > prev_upper else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], prev_lower) if curr_close < prev_lower else lower.iloc[i]
        if pd.isna(st.iloc[i-1]):
            trend.iloc[i] = 1
        elif st.iloc[i-1] == prev_upper:
            trend.iloc[i] = -1 if curr_close < upper.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if curr_close > lower.iloc[i] else -1
        st.iloc[i] = lower.iloc[i] if trend.iloc[i] == 1 else upper.iloc[i]
    return st, trend

def _fae_adx(df, period=14):
    hi, lo, cl = df['high'], df['low'], df['close']
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    dm_plus = hi.diff().clip(lower=0)
    dm_minus = (-lo.diff()).clip(lower=0)
    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    atr14 = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr14.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx, di_plus, di_minus

def show_futures_analysis_engine(df: pd.DataFrame, option_data: dict, current_price: float):
    """12-Module Futures Market Analysis Engine for NIFTY."""
    st.markdown("### 🔮 Futures Market Analysis Engine — NIFTY")

    if df is None or df.empty or len(df) < 20:
        st.warning("Insufficient OHLCV data for Futures Analysis.")
        return

    # Normalise columns
    df2 = df.copy()
    df2.columns = [c.lower() for c in df2.columns]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df2.columns:
            st.warning(f"Missing column: {col}")
            return
    df2 = df2.dropna(subset=['open', 'high', 'low', 'close']).copy()

    close = df2['close']
    volume = df2['volume'].fillna(0)

    # ── MODULE 1: Trend Detection ──────────────────────────────────────
    ema9   = _fae_calc_ema(close, 9)
    ema21  = _fae_calc_ema(close, 21)
    ema50  = _fae_calc_ema(close, 50)
    ema200 = _fae_calc_ema(close, 200)
    _, st_trend = _fae_supertrend(df2, 10, 3.0)
    adx_val, di_plus, di_minus = _fae_adx(df2, 14)

    curr_ema9   = ema9.iloc[-1]
    curr_ema21  = ema21.iloc[-1]
    curr_ema50  = ema50.iloc[-1]
    curr_ema200 = ema200.iloc[-1]
    curr_adx    = adx_val.iloc[-1]
    curr_di_p   = di_plus.iloc[-1]
    curr_di_m   = di_minus.iloc[-1]
    curr_st     = st_trend.iloc[-1] if not pd.isna(st_trend.iloc[-1]) else 0

    price_above_200 = current_price > curr_ema200
    price_above_50  = current_price > curr_ema50
    ema_bull = curr_ema9 > curr_ema21 > curr_ema50
    adx_strong = curr_adx > 25
    trend_dir = "BULLISH" if (ema_bull and curr_st >= 0) else ("BEARISH" if (not ema_bull and curr_st <= 0) else "SIDEWAYS")

    # ── MODULE 2: OI Analysis ─────────────────────────────────────────
    oi_data   = option_data.get('df_summary', pd.DataFrame())
    pcr_oi    = option_data.get('pcr', None)
    max_pain  = option_data.get('max_pain', None)
    if isinstance(oi_data, pd.DataFrame) and not oi_data.empty:
        call_oi = oi_data.get('call_oi', pd.Series(dtype=float)).sum() if 'call_oi' in oi_data.columns else 0
        put_oi  = oi_data.get('put_oi',  pd.Series(dtype=float)).sum() if 'put_oi'  in oi_data.columns else 0
    else:
        call_oi, put_oi = 0, 0
    pcr_num = pcr_oi if isinstance(pcr_oi, (int, float)) else 0
    oi_sentiment = "BULLISH" if pcr_num > 1.2 else ("BEARISH" if pcr_num < 0.8 else "NEUTRAL")

    # ── MODULE 3: Volume & Institutional Activity ──────────────────────
    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    curr_vol   = volume.iloc[-1]
    vol_ratio  = (curr_vol / avg_vol_20) if avg_vol_20 > 0 else 0
    vol_label  = "HIGH" if vol_ratio > 1.5 else ("LOW" if vol_ratio < 0.7 else "NORMAL")
    price_chg  = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100 if len(close) > 1 else 0
    if vol_ratio > 1.5:
        inst_signal = "INSTITUTIONAL BUYING" if price_chg > 0 else "INSTITUTIONAL SELLING"
    else:
        inst_signal = "RETAIL ACTIVITY"

    # ── MODULE 4: Basis / Premium ──────────────────────────────────────
    risk_free = 0.065
    days_to_expiry = option_data.get('days_to_expiry', 7) or 7
    theoretical_fut = current_price * (1 + risk_free * days_to_expiry / 365)
    basis_pts = theoretical_fut - current_price
    basis_pct = (basis_pts / current_price) * 100
    basis_signal = "FAIR" if abs(basis_pct) < 0.3 else ("CONTANGO" if basis_pct > 0 else "BACKWARDATION")

    # ── MODULE 5: Options Confirmation ────────────────────────────────
    iv_skew  = option_data.get('iv_skew', None)
    atm_iv   = option_data.get('atm_iv',  None)
    put_call_iv = "CALL SKEW (BULLISH)" if (isinstance(iv_skew, (int, float)) and iv_skew < 0) \
                  else ("PUT SKEW (BEARISH)" if (isinstance(iv_skew, (int, float)) and iv_skew > 0) else "NEUTRAL")
    atm_iv_str = f"{atm_iv:.1f}%" if isinstance(atm_iv, (int, float)) else "N/A"

    # ── MODULE 6: Expiry Behaviour ────────────────────────────────────
    dte = int(days_to_expiry)
    if dte <= 1:
        expiry_phase = "EXPIRY DAY — High Gamma Risk"
    elif dte <= 3:
        expiry_phase = "NEAR EXPIRY — Pinning possible"
    elif dte <= 7:
        expiry_phase = "LAST WEEK — Vol crush risk"
    else:
        expiry_phase = "MID CYCLE — Normal trading"
    max_pain_str = f"{max_pain:,.0f}" if isinstance(max_pain, (int, float)) and max_pain else "N/A"

    # ── MODULE 7: Trade Setup ─────────────────────────────────────────
    bulls = sum([
        trend_dir == "BULLISH",
        oi_sentiment == "BULLISH",
        vol_ratio > 1.0 and price_chg > 0,
        curr_st > 0,
        price_above_50,
        curr_di_p > curr_di_m,
    ])
    bears = sum([
        trend_dir == "BEARISH",
        oi_sentiment == "BEARISH",
        vol_ratio > 1.0 and price_chg < 0,
        curr_st < 0,
        not price_above_50,
        curr_di_m > curr_di_p,
    ])
    conf_score = round((bulls / 6) * 100)
    if conf_score >= 70:
        setup = "STRONG BUY"
    elif conf_score >= 55:
        setup = "BUY"
    elif conf_score <= 30:
        setup = "STRONG SELL"
    elif conf_score <= 45:
        setup = "SELL"
    else:
        setup = "NEUTRAL / WAIT"

    # ── MODULE 9: Market Day Classification ──────────────────────────
    atr_val  = _fae_calc_atr(df2, 14).iloc[-1]
    day_range = df2['high'].iloc[-1] - df2['low'].iloc[-1]
    range_ratio = day_range / atr_val if atr_val > 0 else 1.0
    if range_ratio > 1.5:
        day_type = "TRENDING DAY"
    elif range_ratio < 0.5:
        day_type = "INSIDE / TIGHT"
    else:
        day_type = "RANGE-BOUND DAY"

    # ── MODULE 10: Visual / Support-Resistance ────────────────────────
    recent_high = df2['high'].rolling(20).max().iloc[-1]
    recent_low  = df2['low'].rolling(20).min().iloc[-1]
    pivot       = (recent_high + recent_low + close.iloc[-1]) / 3
    r1 = 2 * pivot - recent_low
    s1 = 2 * pivot - recent_high

    # ── MODULE 12: Alerts ─────────────────────────────────────────────
    alerts = []
    if adx_strong and curr_st > 0 and ema_bull:
        alerts.append("🟢 Strong Bullish Trend — EMA aligned + SuperTrend up + ADX>25")
    if adx_strong and curr_st < 0 and not ema_bull:
        alerts.append("🔴 Strong Bearish Trend — EMA aligned + SuperTrend down + ADX>25")
    if vol_ratio > 2.0:
        alerts.append(f"⚡ Volume Spike: {vol_ratio:.1f}x average — Watch for breakout")
    if dte <= 1:
        alerts.append("⏰ Expiry Today — Avoid naked options, high gamma")
    if pcr_num < 0.7:
        alerts.append("🔴 Very Low PCR — Heavy call writing / Bearish OI buildup")
    if pcr_num > 1.5:
        alerts.append("🟢 High PCR — Heavy put writing / Bullish OI support")

    # ── MODULE 8: Dashboard Table ─────────────────────────────────────
    st.markdown("#### 📋 Futures Analysis Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Trend Direction", trend_dir)
        st.metric("ADX (Trend Strength)", f"{curr_adx:.1f}", delta="Strong" if adx_strong else "Weak")
        st.metric("SuperTrend", "BULLISH" if curr_st > 0 else "BEARISH")
        st.metric("EMA Stack", "BULLISH" if ema_bull else "BEARISH/MIXED")
    with col2:
        st.metric("PCR (OI)", f"{pcr_num:.2f}", delta=oi_sentiment)
        st.metric("Max Pain", max_pain_str)
        st.metric("Basis Signal", basis_signal, delta=f"{basis_pts:.1f} pts")
        st.metric("ATM IV", atm_iv_str)
    with col3:
        st.metric("Volume Ratio", f"{vol_ratio:.2f}x", delta=vol_label)
        st.metric("Day Type", day_type)
        st.metric("DTE", str(dte))
        st.metric("Trade Setup", setup, delta=f"{conf_score}% confidence")

    # Summary table
    summary_rows = [
        ("MODULE 1 — Trend",        trend_dir,          f"EMA9={curr_ema9:.0f} EMA50={curr_ema50:.0f} ADX={curr_adx:.1f}"),
        ("MODULE 2 — OI",           oi_sentiment,       f"PCR={pcr_num:.2f} MaxPain={max_pain_str}"),
        ("MODULE 3 — Volume",       vol_label,          f"Ratio={vol_ratio:.2f}x | {inst_signal}"),
        ("MODULE 4 — Basis",        basis_signal,       f"{basis_pts:.1f} pts ({basis_pct:.3f}%) | DTE={dte}"),
        ("MODULE 5 — IV Skew",      put_call_iv,        f"ATM IV={atm_iv_str}"),
        ("MODULE 6 — Expiry",       expiry_phase,       f"DTE={dte} | MaxPain={max_pain_str}"),
        ("MODULE 7 — Trade Setup",  setup,              f"Confidence={conf_score}% ({bulls}/6 bulls)"),
        ("MODULE 9 — Day Type",     day_type,           f"Range={day_range:.0f} ATR={atr_val:.0f} Ratio={range_ratio:.2f}"),
        ("MODULE 10 — S/R Levels",  f"Pvt={pivot:.0f}", f"R1={r1:.0f} | S1={s1:.0f}"),
    ]

    def _fae_color(val, col_idx):
        bull_kw = {"BULLISH", "BUY", "HIGH", "STRONG BUY", "INSTITUTIONAL BUYING", "CALL SKEW (BULLISH)"}
        bear_kw = {"BEARISH", "SELL", "STRONG SELL", "INSTITUTIONAL SELLING", "PUT SKEW (BEARISH)"}
        if col_idx == 1:
            if any(k in str(val).upper() for k in [b.upper() for b in bull_kw]):
                return "background-color:#1a3a1a; color:#00e676"
            if any(k in str(val).upper() for k in [b.upper() for b in bear_kw]):
                return "background-color:#3a1a1a; color:#ff5252"
        return ""

    sum_df = pd.DataFrame(summary_rows, columns=["Module", "Signal", "Details"])

    def _style_fae(row):
        styles = ["", "", ""]
        bull_kw = ["BULLISH", "BUY", "HIGH", "INSTITUTIONAL BUYING", "CALL SKEW"]
        bear_kw = ["BEARISH", "SELL", "INSTITUTIONAL SELLING", "PUT SKEW"]
        sig = str(row["Signal"]).upper()
        if any(k in sig for k in bull_kw):
            styles[1] = "background-color:#1a3a1a; color:#00e676; font-weight:bold"
        elif any(k in sig for k in bear_kw):
            styles[1] = "background-color:#3a1a1a; color:#ff5252; font-weight:bold"
        else:
            styles[1] = "background-color:#2a2a1a; color:#ffeb3b; font-weight:bold"
        return styles

    st.dataframe(
        sum_df.style.apply(_style_fae, axis=1),
        use_container_width=True, hide_index=True
    )

    # Alerts
    if alerts:
        st.markdown("#### 🚨 Active Alerts")
        for a in alerts:
            st.info(a)
    else:
        st.success("✅ No critical alerts — Market conditions normal")

    # ── MODULE 11: Mini Backtest ──────────────────────────────────────
    with st.expander("📈 EMA Crossover Backtest (last 50 bars)", expanded=False):
        trades = []
        in_trade = False
        entry_price = 0
        for i in range(21, min(len(df2), 52)):
            e9  = ema9.iloc[i]
            e21 = ema21.iloc[i]
            pe9 = ema9.iloc[i-1]
            pe21= ema21.iloc[i-1]
            c   = close.iloc[i]
            if not in_trade and e9 > e21 and pe9 <= pe21:
                in_trade = True; entry_price = c
            elif in_trade and e9 < e21 and pe9 >= pe21:
                pnl = c - entry_price
                trades.append({"Entry": entry_price, "Exit": c, "PnL": round(pnl, 2),
                                "Result": "WIN" if pnl > 0 else "LOSS"})
                in_trade = False
        if trades:
            bt_df = pd.DataFrame(trades)
            wins  = (bt_df["PnL"] > 0).sum()
            total = len(bt_df)
            win_rate = wins / total * 100
            total_pnl = bt_df["PnL"].sum()
            st.markdown(f"**Trades:** {total} | **Win Rate:** {win_rate:.0f}% | **Total PnL:** {total_pnl:+.1f} pts")

            def _bt_style(row):
                c = "background-color:#1a3a1a; color:#00e676" if row["Result"] == "WIN" \
                    else "background-color:#3a1a1a; color:#ff5252"
                return [c] * len(row)
            st.dataframe(bt_df.style.apply(_bt_style, axis=1), use_container_width=True, hide_index=True)
        else:
            st.info("No completed EMA crossover trades in the last 50 bars.")


# =====================================================================
# FII / DII ACTIVITY ANALYSIS ENGINE
# =====================================================================

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

@st.cache_data(ttl=1800)
def _fii_fetch_cash_data():
    """Fetch FII/DII daily cash segment data from NSE."""
    try:
        import requests, json
        sess = requests.Session()
        sess.headers.update(_NSE_HEADERS)
        # Warm up session with main page to get cookies
        sess.get("https://www.nseindia.com", timeout=10)
        resp = sess.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            return data
    except Exception:
        pass
    return None

@st.cache_data(ttl=1800)
def _fii_fetch_futures_data():
    """Fetch participant-wise futures OI from NSE."""
    try:
        import requests
        from datetime import datetime
        sess = requests.Session()
        sess.headers.update(_NSE_HEADERS)
        sess.get("https://www.nseindia.com", timeout=10)
        today_str = datetime.now().strftime("%d-%b-%Y")
        url = (
            "https://www.nseindia.com/api/reports?"
            "archives=%5B%7B%22name%22%3A%22F%26O%20-%20Participant%20wise%20OI%20(Contracts)%22%2C"
            "%22type%22%3A%22archives%22%2C%22category%22%3A%22derivatives%22%2C"
            "%22section%22%3A%22equity%22%7D%5D"
            f"&date={today_str}&type=archives&category=derivatives&section=equity"
        )
        resp = sess.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def _fii_parse_cash(raw):
    """Parse NSE fiidiiTradeReact response into a clean DataFrame."""
    if not raw or not isinstance(raw, list):
        return pd.DataFrame()
    rows = []
    for item in raw[:15]:  # Last 15 trading days
        try:
            rows.append({
                "Date":           item.get("date", ""),
                "FII Buy":        float(str(item.get("fiiBuy",  "0")).replace(",", "") or 0),
                "FII Sell":       float(str(item.get("fiiSell", "0")).replace(",", "") or 0),
                "DII Buy":        float(str(item.get("diiBuy",  "0")).replace(",", "") or 0),
                "DII Sell":       float(str(item.get("diiSell", "0")).replace(",", "") or 0),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["FII Net"] = df["FII Buy"] - df["FII Sell"]
    df["DII Net"] = df["DII Buy"] - df["DII Sell"]
    return df

def _fii_parse_futures(raw):
    """Parse participant-wise OI CSV/JSON into FII futures long/short."""
    result = {"fii_long": 0, "fii_short": 0,
              "dii_long": 0, "dii_short": 0,
              "pro_long": 0, "pro_short": 0,
              "client_long": 0, "client_short": 0}
    if not raw:
        return result
    try:
        # NSE returns a redirect to CSV; parse if list of dicts
        if isinstance(raw, list):
            for row in raw:
                cat = str(row.get("clientType", row.get("category", ""))).upper()
                long_  = float(str(row.get("futIndexLong",  row.get("long",  "0"))).replace(",","") or 0)
                short_ = float(str(row.get("futIndexShort", row.get("short", "0"))).replace(",","") or 0)
                if "FII" in cat or "FOREIGN" in cat:
                    result["fii_long"]  += long_
                    result["fii_short"] += short_
                elif "DII" in cat or "MUTUAL" in cat or "INSURANCE" in cat:
                    result["dii_long"]  += long_
                    result["dii_short"] += short_
                elif "PRO" in cat:
                    result["pro_long"]  += long_
                    result["pro_short"] += short_
                elif "CLIENT" in cat or "RETAIL" in cat:
                    result["client_long"]  += long_
                    result["client_short"] += short_
    except Exception:
        pass
    return result

def _fii_sentiment(fii_net, dii_net):
    if fii_net > 500:
        return "STRONG BULLISH"
    elif fii_net > 0:
        return "BULLISH"
    elif fii_net < -500:
        return "STRONG BEARISH"
    elif fii_net < 0 and dii_net > 0:
        return "DOMESTIC SUPPORT"
    else:
        return "BEARISH"

def _fii_futures_signal(fii_long, fii_short, prev_long=None, prev_short=None):
    """Detect futures position change pattern."""
    net = fii_long - fii_short
    if prev_long is None:
        return "LONG BUILDUP" if net > 0 else "SHORT BUILDUP"
    prev_net = prev_long - prev_short
    long_chg  = fii_long  - prev_long
    short_chg = fii_short - prev_short
    if long_chg > 0 and short_chg <= 0:
        return "LONG BUILDUP"
    elif long_chg < 0 and short_chg < 0:
        return "LONG UNWINDING"
    elif short_chg > 0 and long_chg <= 0:
        return "SHORT BUILDUP"
    elif short_chg < 0 and long_chg >= 0:
        return "SHORT COVERING"
    return "NEUTRAL"

def _fii_impact_score(fii_net, dii_net, oi_chg_pct, price_chg_pct):
    """Institutional Market Impact Score 0–100."""
    score = 50
    # FII contribution (±30)
    if fii_net > 2000:   score += 30
    elif fii_net > 500:  score += 15
    elif fii_net > 0:    score += 5
    elif fii_net < -2000: score -= 30
    elif fii_net < -500:  score -= 15
    elif fii_net < 0:     score -= 5
    # DII contribution (±10)
    if dii_net > 500:  score += 10
    elif dii_net > 0:  score += 5
    elif dii_net < -500: score -= 10
    elif dii_net < 0:    score -= 5
    # OI contribution (±5)
    if oi_chg_pct > 2:  score += 5
    elif oi_chg_pct < -2: score -= 5
    # Price direction (±5)
    if price_chg_pct > 0.5:  score += 5
    elif price_chg_pct < -0.5: score -= 5
    return max(0, min(100, int(score)))

def show_fii_dii_analysis(df: pd.DataFrame = None, option_data: dict = None,
                          current_price: float = None):
    """9-Module FII & DII Activity Analysis Engine."""
    st.markdown("### 🏦 FII & DII Activity Analysis Engine")

    # ── MODULE 1: Data Collection ─────────────────────────────────────
    with st.spinner("Fetching institutional activity data from NSE..."):
        cash_raw    = _fii_fetch_cash_data()
        futures_raw = _fii_fetch_futures_data()

    cash_df  = _fii_parse_cash(cash_raw)
    fut_data = _fii_parse_futures(futures_raw)

    data_ok = not cash_df.empty

    # Fallback demo notice
    if not data_ok:
        st.info(
            "ℹ️ Live NSE data unavailable (network/session restriction on cloud). "
            "Showing last-available / demo values. Deploy locally for live data."
        )
        # Seed with plausible demo rows so all modules render
        import random
        random.seed(42)
        demo_rows = []
        from datetime import datetime, timedelta
        base = datetime.now()
        for i in range(10):
            day = base - timedelta(days=i+1)
            fb  = round(random.uniform(3000, 12000), 2)
            fs  = round(random.uniform(3000, 12000), 2)
            db  = round(random.uniform(2000,  8000), 2)
            ds  = round(random.uniform(2000,  8000), 2)
            demo_rows.append({
                "Date": day.strftime("%d-%b-%Y"),
                "FII Buy": fb, "FII Sell": fs,
                "DII Buy": db, "DII Sell": ds,
                "FII Net": round(fb - fs, 2),
                "DII Net": round(db - ds, 2),
            })
        cash_df = pd.DataFrame(demo_rows)
        data_ok = True

    # ── MODULE 2: Net Flow Calculation ────────────────────────────────
    today_fii_net = cash_df["FII Net"].iloc[0] if data_ok else 0
    today_dii_net = cash_df["DII Net"].iloc[0] if data_ok else 0
    fii_3d = cash_df["FII Net"].iloc[:3].sum() if data_ok else 0
    dii_3d = cash_df["DII Net"].iloc[:3].sum() if data_ok else 0
    fii_5d = cash_df["FII Net"].iloc[:5].sum() if data_ok else 0
    dii_5d = cash_df["DII Net"].iloc[:5].sum() if data_ok else 0

    # ── MODULE 3: Institutional Sentiment ────────────────────────────
    sentiment    = _fii_sentiment(today_fii_net, today_dii_net)
    fii_trend    = "NET BUYER" if fii_5d > 0 else "NET SELLER"
    dii_trend    = "NET BUYER" if dii_5d > 0 else "NET SELLER"

    # ── MODULE 4: Futures Position Analysis ──────────────────────────
    fii_long  = fut_data["fii_long"]
    fii_short = fut_data["fii_short"]
    fii_fut_net = fii_long - fii_short
    fut_signal = _fii_futures_signal(fii_long, fii_short)

    # ── MODULE 5: Smart Money Signals ────────────────────────────────
    smart_alerts = []
    if today_fii_net > 2000:
        smart_alerts.append(("🟢", "LARGE FII BUYING",
                              f"FII net inflow ₹{today_fii_net:,.0f} Cr today"))
    elif today_fii_net < -2000:
        smart_alerts.append(("🔴", "LARGE FII SELLING",
                              f"FII net outflow ₹{abs(today_fii_net):,.0f} Cr today"))
    if fut_signal == "SHORT COVERING":
        smart_alerts.append(("🟡", "FII SHORT COVERING",
                              "FII reducing short positions → bullish signal"))
    if fut_signal == "LONG BUILDUP":
        smart_alerts.append(("🟢", "FII LONG BUILDUP",
                              "FII adding long futures → bullish confirmation"))
    if fut_signal == "SHORT BUILDUP":
        smart_alerts.append(("🔴", "FII SHORT BUILDUP",
                              "FII adding short futures → bearish positioning"))
    if fii_5d < -5000:
        smart_alerts.append(("🔴", "SUSTAINED FII OUTFLOW",
                              f"FII sold ₹{abs(fii_5d):,.0f} Cr in last 5 days"))
    if fii_5d > 5000:
        smart_alerts.append(("🟢", "SUSTAINED FII INFLOW",
                              f"FII bought ₹{fii_5d:,.0f} Cr in last 5 days"))
    if today_fii_net < 0 and today_dii_net > abs(today_fii_net) * 0.8:
        smart_alerts.append(("🔵", "DII ABSORBING FII SELLING",
                              "Domestic institutions defending market levels"))

    # ── MODULE 6: Market Impact Score ────────────────────────────────
    oi_chg_pct    = 0.0
    price_chg_pct = 0.0
    if option_data:
        pcr = option_data.get('pcr', 1.0) or 1.0
        oi_chg_pct = (pcr - 1.0) * 10  # proxy
    if df is not None and not df.empty and len(df) > 1:
        cl = df.iloc[:, 3] if df.shape[1] > 3 else df.iloc[:, 0]
        try:
            price_chg_pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100
        except Exception:
            pass

    impact_score = _fii_impact_score(today_fii_net, today_dii_net,
                                     oi_chg_pct, price_chg_pct)
    if impact_score >= 70:
        impact_label = "STRONG BULLISH"
        impact_color = "#00e676"
    elif impact_score >= 55:
        impact_label = "BULLISH"
        impact_color = "#69f0ae"
    elif impact_score <= 30:
        impact_label = "STRONG BEARISH"
        impact_color = "#ff5252"
    elif impact_score <= 45:
        impact_label = "BEARISH"
        impact_color = "#ff8a80"
    else:
        impact_label = "NEUTRAL"
        impact_color = "#ffeb3b"

    # ── MODULE 7: Dashboard Table ─────────────────────────────────────
    st.markdown("#### 📊 Today's Institutional Activity")

    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        delta_color = "normal" if today_fii_net >= 0 else "inverse"
        st.metric("FII Net Flow (₹ Cr)", f"{today_fii_net:+,.0f}",
                  delta=f"5D: {fii_5d:+,.0f}", delta_color=delta_color)
    with kc2:
        delta_color = "normal" if today_dii_net >= 0 else "inverse"
        st.metric("DII Net Flow (₹ Cr)", f"{today_dii_net:+,.0f}",
                  delta=f"5D: {dii_5d:+,.0f}", delta_color=delta_color)
    with kc3:
        st.metric("FII Futures Signal", fut_signal,
                  delta=f"Net: {fii_fut_net:+,.0f} contracts" if fii_fut_net else None)
    with kc4:
        st.metric("Impact Score", f"{impact_score}/100", delta=impact_label)

    # Full history table
    st.markdown("#### 📋 FII / DII Cash Flow — Last 15 Days")
    hist_df = cash_df.copy()
    hist_df["FII Buy"]  = hist_df["FII Buy"].apply(lambda x: f"₹{x:,.0f}")
    hist_df["FII Sell"] = hist_df["FII Sell"].apply(lambda x: f"₹{x:,.0f}")
    hist_df["DII Buy"]  = hist_df["DII Buy"].apply(lambda x: f"₹{x:,.0f}")
    hist_df["DII Sell"] = hist_df["DII Sell"].apply(lambda x: f"₹{x:,.0f}")
    hist_df["FII Net"]  = cash_df["FII Net"].apply(lambda x: f"₹{x:+,.0f}")
    hist_df["DII Net"]  = cash_df["DII Net"].apply(lambda x: f"₹{x:+,.0f}")
    hist_df["Sentiment"] = [
        _fii_sentiment(fn, dn)
        for fn, dn in zip(cash_df["FII Net"], cash_df["DII Net"])
    ]

    _SENT_COL = {
        "STRONG BULLISH":   "background-color:#1a3a1a; color:#00e676; font-weight:bold",
        "BULLISH":          "background-color:#1a3a1a; color:#69f0ae; font-weight:bold",
        "DOMESTIC SUPPORT": "background-color:#1a2a3a; color:#40c4ff; font-weight:bold",
        "BEARISH":          "background-color:#3a1a1a; color:#ff8a80; font-weight:bold",
        "STRONG BEARISH":   "background-color:#3a1a1a; color:#ff5252; font-weight:bold",
    }

    def _hist_style(row):
        styles = [""] * len(row)
        col_names = list(row.index)
        sent = str(row.get("Sentiment", ""))
        for i, col in enumerate(col_names):
            if col == "Sentiment":
                styles[i] = _SENT_COL.get(sent, "")
            elif col == "FII Net":
                try:
                    v = float(str(row[col]).replace("₹","").replace(",","").replace("+",""))
                    styles[i] = "color:#00e676" if v >= 0 else "color:#ff5252"
                except Exception:
                    pass
            elif col == "DII Net":
                try:
                    v = float(str(row[col]).replace("₹","").replace(",","").replace("+",""))
                    styles[i] = "color:#00e676" if v >= 0 else "color:#ff5252"
                except Exception:
                    pass
        return styles

    st.dataframe(
        hist_df.style.apply(_hist_style, axis=1),
        use_container_width=True, hide_index=True
    )

    # Futures positions table
    if any(v > 0 for v in [fii_long, fii_short, fut_data["dii_long"], fut_data["pro_long"]]):
        st.markdown("#### 📋 Participant-wise Futures OI")
        fut_rows = [
            {"Participant": "FII / Foreign",
             "Long (Contracts)":   f"{fii_long:,.0f}",
             "Short (Contracts)":  f"{fii_short:,.0f}",
             "Net Position":       f"{fii_fut_net:+,.0f}",
             "Signal":             fut_signal},
            {"Participant": "DII / Domestic",
             "Long (Contracts)":   f"{fut_data['dii_long']:,.0f}",
             "Short (Contracts)":  f"{fut_data['dii_short']:,.0f}",
             "Net Position":       f"{fut_data['dii_long']-fut_data['dii_short']:+,.0f}",
             "Signal":             _fii_futures_signal(fut_data['dii_long'], fut_data['dii_short'])},
            {"Participant": "Pro / Proprietary",
             "Long (Contracts)":   f"{fut_data['pro_long']:,.0f}",
             "Short (Contracts)":  f"{fut_data['pro_short']:,.0f}",
             "Net Position":       f"{fut_data['pro_long']-fut_data['pro_short']:+,.0f}",
             "Signal":             _fii_futures_signal(fut_data['pro_long'], fut_data['pro_short'])},
            {"Participant": "Retail / Client",
             "Long (Contracts)":   f"{fut_data['client_long']:,.0f}",
             "Short (Contracts)":  f"{fut_data['client_short']:,.0f}",
             "Net Position":       f"{fut_data['client_long']-fut_data['client_short']:+,.0f}",
             "Signal":             _fii_futures_signal(fut_data['client_long'], fut_data['client_short'])},
        ]
        fut_df = pd.DataFrame(fut_rows)

        _FUT_SIG_COLORS = {
            "LONG BUILDUP":   "background-color:#1a3a1a; color:#00e676; font-weight:bold",
            "SHORT COVERING": "background-color:#1a3a2a; color:#69f0ae; font-weight:bold",
            "SHORT BUILDUP":  "background-color:#3a1a1a; color:#ff5252; font-weight:bold",
            "LONG UNWINDING": "background-color:#3a2a1a; color:#ff8a80; font-weight:bold",
            "NEUTRAL":        "color:#bdbdbd",
        }

        def _fut_style(row):
            styles = [""] * len(row)
            col_names = list(row.index)
            for i, col in enumerate(col_names):
                if col == "Signal":
                    styles[i] = _FUT_SIG_COLORS.get(str(row[col]), "")
                elif col == "Net Position":
                    try:
                        v = float(str(row[col]).replace(",","").replace("+",""))
                        styles[i] = "color:#00e676" if v >= 0 else "color:#ff5252"
                    except Exception:
                        pass
            return styles

        st.dataframe(
            fut_df.style.apply(_fut_style, axis=1),
            use_container_width=True, hide_index=True
        )

    # ── MODULE 3 summary row ──────────────────────────────────────────
    st.markdown("#### 🧭 Sentiment Summary")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1: st.metric("Overall Sentiment", sentiment)
    with sc2: st.metric("FII 5-Day Trend", fii_trend)
    with sc3: st.metric("DII 5-Day Trend", dii_trend)
    with sc4: st.metric("Futures Signal", fut_signal)
    with sc5:
        st.markdown(
            f"<div style='text-align:center; padding:8px; "
            f"background-color:#1e1e1e; border-radius:8px;'>"
            f"<span style='font-size:12px; color:#aaa;'>Impact Score</span><br>"
            f"<span style='font-size:28px; font-weight:bold; color:{impact_color};'>"
            f"{impact_score}</span>"
            f"<br><span style='font-size:11px; color:{impact_color};'>{impact_label}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ── MODULE 5: Smart Money Alerts ──────────────────────────────────
    st.markdown("#### 🚨 Smart Money Signals")
    if smart_alerts:
        for icon, title, desc in smart_alerts:
            st.info(f"{icon} **{title}** — {desc}")
    else:
        st.success("✅ No extreme institutional signals detected — normal activity")

    # ── MODULE 8: Visual Indicators — rolling bar chart ───────────────
    with st.expander("📈 FII/DII Net Flow Chart (Last 15 Days)", expanded=False):
        try:
            import plotly.graph_objects as go
            chart_df = cash_df.iloc[::-1].reset_index(drop=True)  # oldest first
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=chart_df["Date"], y=chart_df["FII Net"],
                name="FII Net",
                marker_color=[("#00e676" if v >= 0 else "#ff5252") for v in chart_df["FII Net"]],
            ))
            fig.add_trace(go.Bar(
                x=chart_df["Date"], y=chart_df["DII Net"],
                name="DII Net",
                marker_color=[("#40c4ff" if v >= 0 else "#ff9800") for v in chart_df["DII Net"]],
            ))
            fig.update_layout(
                title="FII vs DII Net Cash Flow (₹ Crore)",
                barmode="group",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e0e0e0"),
                legend=dict(bgcolor="#1e1e1e"),
                height=350,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#555")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            # Fallback: simple table
            chart_df2 = cash_df[["Date","FII Net","DII Net"]].iloc[::-1].reset_index(drop=True)
            st.dataframe(chart_df2, use_container_width=True, hide_index=True)

    # ── MODULE 9: Backtest ────────────────────────────────────────────
    with st.expander("📊 FII Signal Backtest — Historical Win Rate", expanded=False):
        if len(cash_df) >= 5:
            bt_rows = []
            for i in range(min(len(cash_df)-1, 14)):
                fii_n  = cash_df["FII Net"].iloc[i]
                dii_n  = cash_df["DII Net"].iloc[i]
                nxt_fn = cash_df["FII Net"].iloc[i+1] if i+1 < len(cash_df) else 0
                signal = "BUY" if fii_n > 0 else "SELL"
                # outcome: if BUY and next FII also positive → WIN
                outcome = "WIN" if (signal == "BUY" and nxt_fn > 0) or \
                                   (signal == "SELL" and nxt_fn < 0) else "LOSS"
                bt_rows.append({
                    "Date":       cash_df["Date"].iloc[i],
                    "FII Net":    f"₹{fii_n:+,.0f}",
                    "DII Net":    f"₹{dii_n:+,.0f}",
                    "Signal":     signal,
                    "Next Day FII": f"₹{nxt_fn:+,.0f}",
                    "Outcome":    outcome,
                })
            bt_df = pd.DataFrame(bt_rows)
            wins     = (bt_df["Outcome"] == "WIN").sum()
            total_bt = len(bt_df)
            win_rate = wins / total_bt * 100 if total_bt > 0 else 0

            st.markdown(
                f"**FII Follow-Through Signal** | "
                f"Trades: **{total_bt}** | "
                f"Wins: **{wins}** | "
                f"Win Rate: **{win_rate:.0f}%**"
            )

            def _bt_style(row):
                c = ("background-color:#1a3a1a; color:#00e676"
                     if row["Outcome"] == "WIN"
                     else "background-color:#3a1a1a; color:#ff5252")
                return [c] * len(row)
            st.dataframe(
                bt_df.style.apply(_bt_style, axis=1),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("Need at least 5 days of data for backtest.")


# =====================================================================
# SECTOR ROTATION ANALYSIS ENGINE
# =====================================================================

_SECTOR_TICKERS = [
    {"name": "NIFTY BANK",    "yf": "^NSEBANK",  "sector": "Banking"},
    {"name": "NIFTY IT",      "yf": "^CNXIT",    "sector": "IT"},
    {"name": "NIFTY AUTO",    "yf": "^CNXAUTO",  "sector": "Auto"},
    {"name": "NIFTY FMCG",    "yf": "^CNXFMCG",  "sector": "FMCG"},
    {"name": "NIFTY PHARMA",  "yf": "^CNXPHARMA","sector": "Pharma"},
    {"name": "NIFTY METAL",   "yf": "^CNXMETAL", "sector": "Metal"},
    {"name": "NIFTY ENERGY",  "yf": "^CNXENERGY","sector": "Energy"},
    {"name": "NIFTY REALTY",  "yf": "^CNXREALTY","sector": "Realty"},
    {"name": "NIFTY PSU BANK","yf": "^CNXPSUBANK","sector": "PSU Bank"},
]

@st.cache_data(ttl=300)
def _sre_fetch_all():
    """Fetch 3-month daily data for all sector indices via yfinance."""
    if not _YF_AVAILABLE:
        return {}
    import yfinance as yf
    results = {}
    tickers = [t["yf"] for t in _SECTOR_TICKERS]
    try:
        raw = yf.download(tickers, period="3mo", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False)
        for t_info in _SECTOR_TICKERS:
            sym = t_info["yf"]
            try:
                if len(tickers) == 1:
                    df_sym = raw.copy()
                else:
                    df_sym = raw[sym].copy() if sym in raw.columns.get_level_values(0) else pd.DataFrame()
                df_sym.columns = [c.lower() for c in df_sym.columns]
                df_sym = df_sym.dropna(subset=["close"])
                results[sym] = df_sym
            except Exception:
                results[sym] = pd.DataFrame()
    except Exception:
        pass
    return results

def _sre_momentum(close_series, period):
    if len(close_series) < period + 1:
        return np.nan
    return (close_series.iloc[-1] / close_series.iloc[-period] - 1) * 100

def _sre_rs_ratio(sector_close, bench_close):
    """Relative strength ratio of sector vs benchmark."""
    s = sector_close.reindex(bench_close.index, method='ffill').dropna()
    b = bench_close.reindex(s.index).dropna()
    idx = s.index.intersection(b.index)
    if len(idx) < 2:
        return np.nan
    return (s.loc[idx].iloc[-1] / b.loc[idx].iloc[-1]) * 100

def _sre_rotation_phase(rs_ratio_vals):
    """
    JdK RS-Ratio + RS-Momentum simplified:
    Leading (RS>100, RS rising), Weakening (RS>100, RS falling),
    Lagging (RS<100, RS falling), Improving (RS<100, RS rising)
    """
    if len(rs_ratio_vals) < 2 or any(np.isnan(v) for v in rs_ratio_vals[-2:]):
        return "UNKNOWN"
    curr, prev = rs_ratio_vals[-1], rs_ratio_vals[-2]
    rising = curr > prev
    if curr >= 100 and rising:
        return "LEADING"
    elif curr >= 100 and not rising:
        return "WEAKENING"
    elif curr < 100 and not rising:
        return "LAGGING"
    else:
        return "IMPROVING"

def show_sector_rotation_engine():
    """14-Module Sector Rotation Analysis Engine."""
    st.markdown("### 🔄 Sector Rotation Analysis Engine — NSE India")

    if not _YF_AVAILABLE:
        st.warning("yfinance not available. Install it (requirements.txt) to enable this engine.")
        return

    with st.spinner("Fetching sector data..."):
        sector_data = _sre_fetch_all()

    if not sector_data:
        st.error("Could not fetch sector data. Check internet connectivity.")
        return

    # Get NIFTY50 as benchmark
    import yfinance as yf
    try:
        bench_raw = yf.download("^NSEI", period="3mo", interval="1d",
                                auto_adjust=True, progress=False)
        bench_raw.columns = [c.lower() for c in bench_raw.columns]
        bench_close = bench_raw['close'].dropna()
    except Exception:
        bench_close = pd.Series(dtype=float)

    rows = []
    rs_history = {}

    for t_info in _SECTOR_TICKERS:
        sym  = t_info["yf"]
        name = t_info["name"]
        df_s = sector_data.get(sym, pd.DataFrame())

        if df_s.empty or 'close' not in df_s.columns or len(df_s) < 5:
            rows.append({
                "Sector": name, "LTP": "N/A", "1D%": "N/A", "1W%": "N/A",
                "1M%": "N/A", "3M%": "N/A", "RS vs NIFTY": "N/A",
                "Phase": "UNKNOWN", "Signal": "N/A", "Strength": "N/A"
            })
            continue

        cl = df_s['close']
        ltp = cl.iloc[-1]
        chg_1d = _sre_momentum(cl, 1)
        chg_1w = _sre_momentum(cl, 5)
        chg_1m = _sre_momentum(cl, 21)
        chg_3m = _sre_momentum(cl, 63)

        # Relative strength
        rs_val = _sre_rs_ratio(cl, bench_close) if not bench_close.empty else np.nan

        # Build rolling RS for phase detection (last 5 daily values)
        rs_series = []
        if not bench_close.empty:
            for i in range(min(5, len(cl))):
                idx_offset = -(i+1)
                s_slice = cl.iloc[:len(cl)+idx_offset] if idx_offset < 0 else cl
                b_slice = bench_close.iloc[:len(bench_close)+idx_offset] if idx_offset < 0 else bench_close
                rs_series.insert(0, _sre_rs_ratio(s_slice, b_slice))
        rs_history[sym] = rs_series

        phase = _sre_rotation_phase(rs_series) if rs_series else "UNKNOWN"

        # Signal based on phase + momentum
        if phase == "LEADING" and chg_1w > 0:
            signal = "OVERWEIGHT"
        elif phase == "WEAKENING":
            signal = "REDUCE"
        elif phase == "LAGGING":
            signal = "UNDERWEIGHT"
        elif phase == "IMPROVING" and chg_1d > 0:
            signal = "ACCUMULATE"
        else:
            signal = "NEUTRAL"

        # Strength score (0-100)
        score_parts = [
            min(max(chg_1d * 10 + 50, 0), 100) if not np.isnan(chg_1d) else 50,
            min(max(chg_1w * 5  + 50, 0), 100) if not np.isnan(chg_1w) else 50,
            min(max(chg_1m * 2  + 50, 0), 100) if not np.isnan(chg_1m) else 50,
        ]
        strength = int(np.mean(score_parts))

        rows.append({
            "Sector":      name,
            "LTP":         f"{ltp:,.0f}",
            "1D%":         f"{chg_1d:+.2f}%" if not np.isnan(chg_1d) else "N/A",
            "1W%":         f"{chg_1w:+.2f}%" if not np.isnan(chg_1w) else "N/A",
            "1M%":         f"{chg_1m:+.2f}%" if not np.isnan(chg_1m) else "N/A",
            "3M%":         f"{chg_3m:+.2f}%" if not np.isnan(chg_3m) else "N/A",
            "RS vs NIFTY": f"{rs_val:.1f}" if not np.isnan(rs_val) else "N/A",
            "Phase":       phase,
            "Signal":      signal,
            "Strength":    strength,
            "_chg1d_raw":  chg_1d if not np.isnan(chg_1d) else 0,
            "_chg1w_raw":  chg_1w if not np.isnan(chg_1w) else 0,
            "_phase_raw":  phase,
            "_signal_raw": signal,
        })

    if not rows:
        st.warning("No sector data available.")
        return

    full_df = pd.DataFrame(rows)
    display_cols = ["Sector","LTP","1D%","1W%","1M%","3M%","RS vs NIFTY","Phase","Signal","Strength"]
    disp_df = full_df[display_cols].copy()

    # ── MODULE 2: Heatmap-style styling ──────────────────────────────
    def _sre_style_row(row):
        styles = [""] * len(row)
        phase  = str(row.get("Phase", ""))
        signal = str(row.get("Signal", ""))
        phase_colors = {
            "LEADING":   ("background-color:#1a3a1a", "color:#00e676"),
            "IMPROVING": ("background-color:#1a2a3a", "color:#40c4ff"),
            "WEAKENING": ("background-color:#2a2a1a", "color:#ffeb3b"),
            "LAGGING":   ("background-color:#3a1a1a", "color:#ff5252"),
        }
        sig_colors = {
            "OVERWEIGHT":  "color:#00e676; font-weight:bold",
            "ACCUMULATE":  "color:#40c4ff; font-weight:bold",
            "NEUTRAL":     "color:#bdbdbd",
            "REDUCE":      "color:#ffeb3b; font-weight:bold",
            "UNDERWEIGHT": "color:#ff5252; font-weight:bold",
        }
        col_names = list(row.index)
        for i, col in enumerate(col_names):
            if col == "Phase":
                bg, fg = phase_colors.get(phase, ("", ""))
                styles[i] = f"{bg}; {fg}"
            elif col == "Signal":
                styles[i] = sig_colors.get(signal, "")
            elif col in ("1D%", "1W%", "1M%", "3M%"):
                try:
                    val = float(str(row[col]).replace("%","").replace("+",""))
                    if val > 1:
                        styles[i] = "color:#00e676"
                    elif val < -1:
                        styles[i] = "color:#ff5252"
                    else:
                        styles[i] = "color:#bdbdbd"
                except Exception:
                    pass
        return styles

    # ── MODULE 10: Dashboard Table ────────────────────────────────────
    st.markdown("#### 📊 Sector Rotation Table")
    st.dataframe(
        disp_df.style.apply(_sre_style_row, axis=1),
        use_container_width=True, hide_index=True
    )

    # ── MODULE 3: Money Flow Heatmap ──────────────────────────────────
    st.markdown("#### 💹 Money Flow Heatmap (1W Performance)")
    heat_data = []
    for r in rows:
        try:
            pct = float(str(r["1W%"]).replace("%","").replace("+",""))
        except Exception:
            pct = 0
        heat_data.append({"Sector": r["Sector"], "1W%": pct})
    heat_df = pd.DataFrame(heat_data).sort_values("1W%", ascending=False)

    def _heat_color(val):
        if val > 2:    return "background-color:#00600a; color:white"
        elif val > 0:  return "background-color:#1b5e20; color:#e8f5e9"
        elif val > -2: return "background-color:#b71c1c; color:#ffcdd2"
        else:          return "background-color:#7f0000; color:white"

    st.dataframe(
        heat_df.style.applymap(_heat_color, subset=["1W%"]),
        use_container_width=True, hide_index=True
    )

    # ── MODULE 4: Relative Strength Rankings ─────────────────────────
    st.markdown("#### 🏆 Relative Strength Rankings (vs NIFTY 50)")
    rs_rows = []
    for r in rows:
        rs_val_str = r.get("RS vs NIFTY", "N/A")
        try:
            rs_num = float(rs_val_str)
        except Exception:
            rs_num = np.nan
        phase = r.get("Phase", "UNKNOWN")
        rs_rows.append({
            "Sector": r["Sector"],
            "RS Ratio": rs_val_str,
            "Phase": phase,
            "Rank Signal": "OUTPERFORM" if (not np.isnan(rs_num) and rs_num >= 100) else "UNDERPERFORM"
        })
    rs_df = pd.DataFrame(rs_rows)

    def _rs_style(row):
        if row["Rank Signal"] == "OUTPERFORM":
            return [""] + ["background-color:#1a3a1a; color:#00e676"] * (len(row)-1)
        return [""] + ["background-color:#3a1a1a; color:#ff5252"] * (len(row)-1)

    st.dataframe(rs_df.style.apply(_rs_style, axis=1), use_container_width=True, hide_index=True)

    # ── MODULE 5: Rotation Signals Summary ───────────────────────────
    leading   = [r["Sector"] for r in rows if r["Phase"] == "LEADING"]
    improving = [r["Sector"] for r in rows if r["Phase"] == "IMPROVING"]
    weakening = [r["Sector"] for r in rows if r["Phase"] == "WEAKENING"]
    lagging   = [r["Sector"] for r in rows if r["Phase"] == "LAGGING"]

    st.markdown("#### 🔄 Rotation Phase Summary")
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.success(f"**LEADING** ({len(leading)})\n\n" + "\n".join(f"• {s}" for s in leading) if leading else "None")
    with rc2:
        st.info(f"**IMPROVING** ({len(improving)})\n\n" + "\n".join(f"• {s}" for s in improving) if improving else "None")
    with rc3:
        st.warning(f"**WEAKENING** ({len(weakening)})\n\n" + "\n".join(f"• {s}" for s in weakening) if weakening else "None")
    with rc4:
        st.error(f"**LAGGING** ({len(lagging)})\n\n" + "\n".join(f"• {s}" for s in lagging) if lagging else "None")

    # ── MODULE 6: Intraday Tracker (1D momentum rankings) ────────────
    with st.expander("⚡ Intraday Sector Momentum Tracker", expanded=False):
        intra_rows = sorted(rows, key=lambda x: x["_chg1d_raw"], reverse=True)
        intra_df = pd.DataFrame([
            {"Rank": i+1, "Sector": r["Sector"], "1D Change": r["1D%"], "Signal": r["Signal"]}
            for i, r in enumerate(intra_rows)
        ])

        def _intra_style(row):
            try:
                val = float(str(row["1D Change"]).replace("%","").replace("+",""))
                c = "color:#00e676" if val > 0 else "color:#ff5252"
            except Exception:
                c = ""
            return ["", "", c, ""]
        st.dataframe(intra_df.style.apply(_intra_style, axis=1), use_container_width=True, hide_index=True)

    # ── MODULE 7: Alerts ──────────────────────────────────────────────
    st.markdown("#### 🚨 Sector Rotation Alerts")
    sector_alerts = []
    for r in rows:
        if r["Phase"] == "IMPROVING" and r["_chg1w_raw"] > 2:
            sector_alerts.append(f"🟢 **{r['Sector']}** entering IMPROVING phase with strong 1W gain ({r['1W%']})")
        if r["Phase"] == "WEAKENING" and r["_chg1w_raw"] < -1:
            sector_alerts.append(f"🟡 **{r['Sector']}** WEAKENING — consider reducing exposure ({r['1W%']})")
        if r["Phase"] == "LAGGING" and r["_chg1d_raw"] < -1.5:
            sector_alerts.append(f"🔴 **{r['Sector']}** LAGGING with today's drop ({r['1D%']}) — avoid")
        if r["Phase"] == "LEADING" and r["_chg1w_raw"] > 3:
            sector_alerts.append(f"⚡ **{r['Sector']}** LEADING strongly ({r['1W%']} in 1W) — momentum play")

    if sector_alerts:
        for al in sector_alerts:
            st.info(al)
    else:
        st.success("✅ No critical sector rotation alerts at this time.")


# =====================================================================
# CANDLESTICK INTELLIGENCE ENGINE — Helper Functions
# =====================================================================

def _cie_detect_swing_sr(df, lookback=200, cluster_pct=0.003):
    """Detect swing high/low S/R levels and cluster nearby ones."""
    if df is None or len(df) < 10:
        return [], []
    recent = df.tail(lookback).copy().reset_index(drop=True)
    n = len(recent)
    raw_sup, raw_res = [], []
    for i in range(2, n - 2):
        lo = recent['low'].iloc[i]
        hi = recent['high'].iloc[i]
        if (lo < recent['low'].iloc[i-1] and lo < recent['low'].iloc[i-2] and
                lo < recent['low'].iloc[i+1] and lo < recent['low'].iloc[i+2]):
            raw_sup.append(lo)
        if (hi > recent['high'].iloc[i-1] and hi > recent['high'].iloc[i-2] and
                hi > recent['high'].iloc[i+1] and hi > recent['high'].iloc[i+2]):
            raw_res.append(hi)

    def _cluster(levels):
        if not levels:
            return []
        levels = sorted(levels)
        clusters, group = [], [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - group[-1]) / (group[-1] + 1e-6) <= cluster_pct:
                group.append(lvl)
            else:
                clusters.append(float(np.mean(group)))
                group = [lvl]
        clusters.append(float(np.mean(group)))
        return clusters

    return _cluster(raw_sup), _cluster(raw_res)


def _cie_find_nearest_sr(price, supports, resistances, thr=0.002):
    """Return (level, type, dist_pct) of nearest S/R within threshold."""
    best_dist, best_lvl, best_type = float('inf'), None, None
    for lvl in supports:
        d = abs(price - lvl) / (lvl + 1e-6)
        if d < best_dist and d <= thr:
            best_dist, best_lvl, best_type = d, lvl, 'support'
    for lvl in resistances:
        d = abs(price - lvl) / (lvl + 1e-6)
        if d < best_dist and d <= thr:
            best_dist, best_lvl, best_type = d, lvl, 'resistance'
    return best_lvl, best_type, best_dist


def _cie_detect_patterns(df, supports, resistances):
    """Detect reversal and continuation patterns near S/R levels."""
    signals = []
    if df is None or len(df) < 3:
        return signals

    n = len(df)
    avg_body = (df['close'] - df['open']).abs().rolling(20, min_periods=3).mean().fillna(50)
    avg_vol = df['volume'].rolling(20, min_periods=3).mean().fillna(1)
    thr = 0.002  # 0.2% proximity threshold

    for i in range(2, n):
        c  = df.iloc[i]
        p1 = df.iloc[i - 1]
        candle_range = c['high'] - c['low']
        if candle_range < 1e-6:
            continue
        body        = abs(c['close'] - c['open'])
        upper_wick  = c['high'] - max(c['close'], c['open'])
        lower_wick  = min(c['close'], c['open']) - c['low']
        is_green    = c['close'] >= c['open']
        is_red      = c['close'] < c['open']
        avg_b       = max(avg_body.iloc[i], 1)
        vol_spike   = bool(c['volume'] > avg_vol.iloc[i] * 1.5)
        ts          = c['datetime'] if 'datetime' in df.columns else i

        # S/R proximity — check low side for support, high side for resistance
        probe_sup = c['low'] if is_red else min(c['close'], c['open'])
        probe_res = c['high'] if is_green else max(c['close'], c['open'])
        lvl_s, typ_s, dist_s = _cie_find_nearest_sr(probe_sup, supports, resistances, thr)
        lvl_r, typ_r, dist_r = _cie_find_nearest_sr(probe_res, supports, resistances, thr)
        near_sup = typ_s == 'support'
        near_res = typ_r == 'resistance'

        # ── BULLISH REVERSALS AT SUPPORT ────────────────────────────────
        if near_sup:
            # 1. Hammer
            if lower_wick > 2 * max(body, 1) and upper_wick <= body * 0.5 and lower_wick > candle_range * 0.5:
                signals.append({
                    'index': i, 'time': ts, 'pattern': 'Hammer', 'direction': 'BUY',
                    'category': 'Reversal', 'price': c['close'], 'level': lvl_s,
                    'level_type': 'Support', 'dist_pct': dist_s, 'vol_spike': vol_spike,
                    'candle_strength': min(28, int(lower_wick / candle_range * 28)),
                })
            # 2. Long Lower Wick Rejection
            if lower_wick > candle_range * 0.60:
                signals.append({
                    'index': i, 'time': ts, 'pattern': 'Long Lower Wick Rejection',
                    'direction': 'BUY', 'category': 'Reversal', 'price': c['close'],
                    'level': lvl_s, 'level_type': 'Support', 'dist_pct': dist_s,
                    'vol_spike': vol_spike,
                    'candle_strength': min(28, int(lower_wick / candle_range * 28)),
                })
            # 3. Bullish Engulfing
            if (p1['close'] < p1['open'] and is_green and
                    min(c['open'], c['close']) <= min(p1['open'], p1['close']) and
                    max(c['open'], c['close']) >= max(p1['open'], p1['close'])):
                signals.append({
                    'index': i, 'time': ts, 'pattern': 'Bullish Engulfing',
                    'direction': 'BUY', 'category': 'Reversal', 'price': c['close'],
                    'level': lvl_s, 'level_type': 'Support', 'dist_pct': dist_s,
                    'vol_spike': vol_spike,
                    'candle_strength': min(28, int(body / avg_b * 14)),
                })
            # 4. Morning Star (3-candle)
            if i >= 2:
                c0, c1, c2 = df.iloc[i-2], df.iloc[i-1], c
                b0 = abs(c0['close'] - c0['open'])
                b1 = abs(c1['close'] - c1['open'])
                mid0 = (c0['open'] + c0['close']) / 2
                if (c0['close'] < c0['open'] and b0 > avg_b * 0.8 and b1 < avg_b * 0.4 and
                        c2['close'] > c2['open'] and c2['close'] > mid0):
                    signals.append({
                        'index': i, 'time': ts, 'pattern': 'Morning Star',
                        'direction': 'BUY', 'category': 'Reversal', 'price': c2['close'],
                        'level': lvl_s, 'level_type': 'Support', 'dist_pct': dist_s,
                        'vol_spike': vol_spike, 'candle_strength': 25,
                    })
            # 5. Bullish Marubozu (continuation)
            if (is_green and body > avg_b * 1.5 and upper_wick < body * 0.1 and lower_wick < body * 0.1):
                signals.append({
                    'index': i, 'time': ts, 'pattern': 'Bullish Marubozu',
                    'direction': 'BUY', 'category': 'Continuation', 'price': c['close'],
                    'level': lvl_s, 'level_type': 'Support', 'dist_pct': dist_s,
                    'vol_spike': vol_spike,
                    'candle_strength': min(28, int(body / avg_b * 12)),
                })
            # 6. Inside Bar Breakout (prev candle range small, curr breaks above)
            p1_rng = p1['high'] - p1['low']
            if (p1_rng < avg_b * 0.8 and c['high'] > p1['high'] and is_green):
                signals.append({
                    'index': i, 'time': ts, 'pattern': 'Inside Bar Breakout',
                    'direction': 'BUY', 'category': 'Continuation', 'price': c['close'],
                    'level': lvl_s, 'level_type': 'Support', 'dist_pct': dist_s,
                    'vol_spike': vol_spike, 'candle_strength': 18,
                })

        # ── BEARISH REVERSALS AT RESISTANCE ─────────────────────────────
        if near_res:
            # 7. Shooting Star
            if upper_wick > 2 * max(body, 1) and lower_wick <= body * 0.5 and upper_wick > candle_range * 0.5:
                signals.append({
                    'index': i, 'time': ts, 'pattern': 'Shooting Star',
                    'direction': 'SELL', 'category': 'Reversal', 'price': c['close'],
                    'level': lvl_r, 'level_type': 'Resistance', 'dist_pct': dist_r,
                    'vol_spike': vol_spike,
                    'candle_strength': min(28, int(upper_wick / candle_range * 28)),
                })
            # 8. Upper Wick Rejection
            if upper_wick > candle_range * 0.60:
                signals.append({
                    'index': i, 'time': ts, 'pattern': 'Upper Wick Rejection',
                    'direction': 'SELL', 'category': 'Reversal', 'price': c['close'],
                    'level': lvl_r, 'level_type': 'Resistance', 'dist_pct': dist_r,
                    'vol_spike': vol_spike,
                    'candle_strength': min(28, int(upper_wick / candle_range * 28)),
                })
            # 9. Bearish Engulfing
            if (p1['close'] > p1['open'] and is_red and
                    min(c['open'], c['close']) <= min(p1['open'], p1['close']) and
                    max(c['open'], c['close']) >= max(p1['open'], p1['close'])):
                signals.append({
                    'index': i, 'time': ts, 'pattern': 'Bearish Engulfing',
                    'direction': 'SELL', 'category': 'Reversal', 'price': c['close'],
                    'level': lvl_r, 'level_type': 'Resistance', 'dist_pct': dist_r,
                    'vol_spike': vol_spike,
                    'candle_strength': min(28, int(body / avg_b * 14)),
                })
            # 10. Evening Star (3-candle)
            if i >= 2:
                c0, c1, c2 = df.iloc[i-2], df.iloc[i-1], c
                b0 = abs(c0['close'] - c0['open'])
                b1 = abs(c1['close'] - c1['open'])
                mid0 = (c0['open'] + c0['close']) / 2
                if (c0['close'] > c0['open'] and b0 > avg_b * 0.8 and b1 < avg_b * 0.4 and
                        c2['close'] < c2['open'] and c2['close'] < mid0):
                    signals.append({
                        'index': i, 'time': ts, 'pattern': 'Evening Star',
                        'direction': 'SELL', 'category': 'Reversal', 'price': c2['close'],
                        'level': lvl_r, 'level_type': 'Resistance', 'dist_pct': dist_r,
                        'vol_spike': vol_spike, 'candle_strength': 25,
                    })
            # 11. Bearish Marubozu (continuation)
            if (is_red and body > avg_b * 1.5 and upper_wick < body * 0.1 and lower_wick < body * 0.1):
                signals.append({
                    'index': i, 'time': ts, 'pattern': 'Bearish Marubozu',
                    'direction': 'SELL', 'category': 'Continuation', 'price': c['close'],
                    'level': lvl_r, 'level_type': 'Resistance', 'dist_pct': dist_r,
                    'vol_spike': vol_spike,
                    'candle_strength': min(28, int(body / avg_b * 12)),
                })

    # 12. Lower Low Momentum (last 5 candles making lower highs & lows)
    if n >= 6:
        last6 = df.tail(6)
        if (all(last6['high'].iloc[j] < last6['high'].iloc[j-1] for j in range(1, 6)) and
                all(last6['low'].iloc[j] < last6['low'].iloc[j-1] for j in range(1, 6))):
            cl = df.iloc[-1]
            lvl_r2, _, dr2 = _cie_find_nearest_sr(cl['high'], supports, resistances, 0.005)
            signals.append({
                'index': n - 1,
                'time': cl['datetime'] if 'datetime' in df.columns else n - 1,
                'pattern': 'Lower Low Momentum', 'direction': 'SELL',
                'category': 'Continuation', 'price': cl['close'],
                'level': lvl_r2, 'level_type': 'Resistance', 'dist_pct': dr2,
                'vol_spike': False, 'candle_strength': 20,
            })

    # Return only signals from last 4 candles
    cutoff = max(0, n - 4)
    return [s for s in signals if s.get('index', 0) >= cutoff]


def _cie_options_confirmation(signal, df_summary, straddle_history, underlying_price):
    """
    Apply options flow filters and volatility filter.
    Returns (confirmed: bool, details: dict).
    """
    if df_summary is None or underlying_price is None:
        return False, {'volatility_filter': 'No data'}
    try:
        atm_strike = min(df_summary['Strike'].tolist(), key=lambda x: abs(x - underlying_price))
        atm_rows = df_summary[df_summary['Strike'] == atm_strike]
        details = {}
        opt_score = 0

        # Straddle ROC (volatility filter)
        straddle_roc = 0.0
        if len(straddle_history) >= 2:
            s_last = straddle_history[-1].get('straddle', 0) if isinstance(straddle_history[-1], dict) else 0
            s_prev = straddle_history[-2].get('straddle', 0) if isinstance(straddle_history[-2], dict) else 0
            if s_prev > 0:
                straddle_roc = abs((s_last - s_prev) / s_prev * 100)
        details['straddle_roc'] = round(straddle_roc, 2)
        details['straddle_expanding'] = straddle_roc >= 0.5

        if straddle_roc < 0.5:
            details['volatility_filter'] = 'BLOCKED'
            return False, details
        details['volatility_filter'] = 'PASS'
        opt_score += 10

        if not atm_rows.empty:
            row = atm_rows.iloc[0]
            pcr    = float(row.get('PCR', 1.0) or 1.0)
            ce_chg = float(row.get('changeinOpenInterest_CE', 0) or 0)
            pe_chg = float(row.get('changeinOpenInterest_PE', 0) or 0)
            d_ce   = float(row.get('Delta_CE', 0) or 0)
            d_pe   = float(row.get('Delta_PE', 0) or 0)
            gamma  = row.get('Gamma_SR', '-')
            details.update({
                'pcr': round(pcr, 2),
                'ce_chg_oi': round(ce_chg, 0),
                'pe_chg_oi': round(pe_chg, 0),
                'net_delta': round(d_ce + d_pe, 4),
                'gamma_sr': str(gamma),
            })

            if signal['direction'] == 'BUY':
                if pcr > 1.0:
                    opt_score += 15
                    details['pcr_conf'] = f'Bullish PCR {pcr:.2f} ✅'
                if pe_chg > 0:
                    opt_score += 15
                    details['oi_conf'] = 'Put Writing ✅'
                elif ce_chg < 0:
                    opt_score += 8
                    details['oi_conf'] = 'Call Unwinding ✅'
                if 'Support' in str(gamma):
                    opt_score += 10
                    details['gamma_conf'] = 'Gamma Support ✅'
                else:
                    details['gamma_conf'] = str(gamma) or '-'
                if (d_ce + d_pe) > 0:
                    opt_score += 10
                    details['delta_conf'] = 'Delta Positive ✅'
            else:  # SELL
                if pcr < 1.0:
                    opt_score += 15
                    details['pcr_conf'] = f'Bearish PCR {pcr:.2f} ✅'
                if ce_chg > 0:
                    opt_score += 15
                    details['oi_conf'] = 'Call Writing ✅'
                elif pe_chg < 0:
                    opt_score += 8
                    details['oi_conf'] = 'Put Unwinding ✅'
                if 'Resist' in str(gamma):
                    opt_score += 10
                    details['gamma_conf'] = 'Gamma Resistance ✅'
                else:
                    details['gamma_conf'] = str(gamma) or '-'
                if (d_ce + d_pe) < 0:
                    opt_score += 10
                    details['delta_conf'] = 'Delta Negative ✅'

        details['options_score'] = opt_score
        confirmed = opt_score >= 20
        details['confirmed'] = confirmed
        return confirmed, details
    except Exception:
        return False, {'volatility_filter': 'Error'}


def _cie_confidence_score(signal, opt_details, opt_confirmed):
    """
    Score 0-100:
      Candle strength     0-28
      Distance to S/R     0-20
      OI + options score  0-15
      Gamma               0-15
      Straddle expansion  0-10
      Volume spike        0-10
      Institutional bonus 0-10 (vol spike + confirmed + strong candle)
    """
    score = 0
    score += min(28, signal.get('candle_strength', 10))

    dist_pct = signal.get('dist_pct') or 0.002
    score += max(0, min(20, 20 - int(dist_pct / 0.0001)))

    score += min(15, int(opt_details.get('options_score', 0) * 0.25))

    gamma_conf = str(opt_details.get('gamma_conf', ''))
    if '✅' in gamma_conf:
        score += 15
    elif gamma_conf and gamma_conf != '-':
        score += 4

    if opt_details.get('straddle_expanding'):
        score += 10

    if signal.get('vol_spike'):
        score += 10

    if signal.get('vol_spike') and opt_confirmed and signal.get('candle_strength', 0) >= 20:
        score += 10  # institutional bonus

    return min(100, score)


def run_candlestick_intelligence_engine(df, option_data, straddle_history, underlying_price, is_expiry=False):
    """
    Main CIE entry point. Returns list of scored, filtered signal dicts.
    Each signal contains: pattern, direction, category, price, level,
    level_type, confidence, signal_strength, strength_color, options_details.
    """
    if df is None or len(df) < 5 or underlying_price is None:
        return []

    df_summary = option_data.get('df_summary') if option_data else None
    sr_data    = option_data.get('sr_data', []) if option_data else []
    vob_blocks = option_data.get('vob_blocks', {}) if option_data else {}

    # Build S/R from swing detection
    supports, resistances = _cie_detect_swing_sr(df)

    # Augment with option_data S/R
    for sr in (sr_data or []):
        v = sr.get('value', 0)
        if v > 0:
            (supports if sr.get('type') == 'low' else resistances).append(float(v))

    if isinstance(vob_blocks, dict):
        for b in vob_blocks.get('bullish', []):
            if b.get('mid', 0) > 0:
                supports.append(float(b['mid']))
        for b in vob_blocks.get('bearish', []):
            if b.get('mid', 0) > 0:
                resistances.append(float(b['mid']))

    # Re-cluster merged levels
    def _recluster(lvls, pct=0.003):
        if not lvls:
            return []
        lvls = sorted(set([round(l, 1) for l in lvls if l > 0]))
        clusters, grp = [], [lvls[0]]
        for l in lvls[1:]:
            if abs(l - grp[-1]) / (grp[-1] + 1e-6) <= pct:
                grp.append(l)
            else:
                clusters.append(float(np.mean(grp)))
                grp = [l]
        clusters.append(float(np.mean(grp)))
        return clusters

    supports    = _recluster(supports)
    resistances = _recluster(resistances)

    raw_signals = _cie_detect_patterns(df, supports, resistances)
    final_signals = []

    for sig in raw_signals:
        opt_confirmed, opt_details = _cie_options_confirmation(
            sig, df_summary, straddle_history, underlying_price)

        # Volatility filter — skip unless expiry day
        if opt_details.get('volatility_filter') == 'BLOCKED' and not is_expiry:
            continue

        confidence = _cie_confidence_score(sig, opt_details, opt_confirmed)
        if confidence < 40:
            continue

        sig['confidence']       = confidence
        sig['options_confirmed'] = opt_confirmed
        sig['options_details']  = opt_details

        if confidence >= 85:
            sig['signal_strength'] = 'INSTITUTIONAL'
            sig['strength_color']  = '#ff6600'
        elif confidence >= 70:
            sig['signal_strength'] = 'STRONG'
            sig['strength_color']  = '#ffaa00'
        else:
            sig['signal_strength'] = 'NORMAL'
            sig['strength_color']  = '#00aaff'

        final_signals.append(sig)

    # Deduplicate by (pattern, direction), keep highest confidence
    seen, deduped = set(), []
    for s in sorted(final_signals, key=lambda x: -x['confidence']):
        key = (s['pattern'], s['direction'])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC PATTERN ANALYSIS — UI RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def render_geo_pattern_analysis(df, df_full, date_label='Today'):
    """
    Render the full Pattern Analysis panel:
      1. Real-time pattern alerts
      2. Pattern detection results table
      3. Backtest chart + summary
    """
    if df is None or df.empty:
        return

    detector = GeometricPatternDetector()

    # ── Live patterns on today's / selected-date df ────────────────────
    live_patterns = detector.detect_all(df)

    st.markdown("---")
    st.markdown("## 📐 Geometric & Reversal Pattern Analysis")
    st.caption(f"Detecting 10 pattern types on NIFTY50 — {date_label}")

    # ── Pattern Alerts — single unified table ─────────────────────────
    st.markdown("### 🔔 Pattern Alerts")
    if live_patterns:
        _alert_rows = []
        for p in sorted(live_patterns, key=lambda x: x['time'], reverse=True):
            sig = p['signal']
            icon = '🟢' if sig == 'BUY' else ('🔴' if sig == 'SELL' else '🟡')
            sig_lbl = f"{icon} {sig}"
            ts = p['time']
            time_str = ts.strftime('%H:%M') if hasattr(ts, 'strftime') else str(ts)
            _alert_rows.append({
                'Time': time_str,
                'Pattern': f"{icon} {p['pattern']}",
                'Sentiment': p['sentiment'],
                'Signal': sig_lbl,
                'Breakout (₹)': f"₹{p['entry']:.0f}",
                'SL (₹)': f"₹{p['stoploss']:.0f}",
                'Target (₹)': f"₹{p['target']:.0f}",
                'RR': f"{p['rr']:.2f}",
                'Confidence': p['confidence'],
                'Move %': f"{p['move_pct']:+.2f}%",
            })
        st.dataframe(pd.DataFrame(_alert_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No geometric patterns detected on current price data. Patterns will appear when confirmed breakouts occur.")

    # ── Pattern Heatmap (frequency by pattern name) ───────────────────
    st.markdown("### 🗺️ Pattern Heatmap")
    if df_full is not None and not df_full.empty and len(df_full) >= 30:
        bt_results = detector.backtest_scan(df_full, step=10)
        if bt_results:
            from collections import Counter
            pat_counts = Counter(r['pattern'] for r in bt_results)
            pat_wins   = {}
            pat_total  = {}
            for r in bt_results:
                k = r['pattern']
                pat_total[k] = pat_total.get(k, 0) + 1
                if r.get('result') == 'WIN':
                    pat_wins[k] = pat_wins.get(k, 0) + 1

            heatmap_rows = []
            for pat, cnt in sorted(pat_counts.items(), key=lambda x: -x[1]):
                wins = pat_wins.get(pat, 0)
                total = pat_total.get(pat, 1)
                wr = wins / total * 100
                heatmap_rows.append({
                    'Pattern': pat,
                    'Occurrences': cnt,
                    'Win Rate %': f"{wr:.1f}%",
                    'Wins': wins,
                    'Losses': total - wins,
                })
            st.dataframe(pd.DataFrame(heatmap_rows), use_container_width=True, hide_index=True)

            # Most frequent pattern indicator
            top_pat = max(pat_counts, key=pat_counts.get)
            best_wr_pat = max(pat_wins, key=lambda k: pat_wins[k] / pat_total.get(k, 1)) if pat_wins else top_pat
            worst_wr_pat = min(pat_total, key=lambda k: pat_wins.get(k, 0) / pat_total[k]) if pat_total else top_pat

            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                st.metric("Most Frequent Pattern", top_pat, f"{pat_counts[top_pat]} times")
            with col_h2:
                best_wr = pat_wins.get(best_wr_pat, 0) / pat_total.get(best_wr_pat, 1) * 100
                st.metric("Highest Win Rate", best_wr_pat, f"{best_wr:.1f}%")
            with col_h3:
                worst_wr = pat_wins.get(worst_wr_pat, 0) / pat_total.get(worst_wr_pat, 1) * 100
                st.metric("Lowest Win Rate", worst_wr_pat, f"{worst_wr:.1f}%")

            # ── Backtest Summary Panel ─────────────────────────────────
            st.markdown("### 📋 Backtest Summary")
            total_detected = len(bt_results)
            total_wins = sum(1 for r in bt_results if r.get('result') == 'WIN')
            overall_wr = total_wins / total_detected * 100 if total_detected > 0 else 0
            avg_move = np.mean([abs(r.get('actual_move_pct', 0)) for r in bt_results]) if bt_results else 0
            buy_results = [r for r in bt_results if r['signal'] == 'BUY']
            sell_results = [r for r in bt_results if r['signal'] == 'SELL']

            bs_col1, bs_col2, bs_col3, bs_col4 = st.columns(4)
            with bs_col1:
                st.metric("Total Patterns Detected", total_detected)
            with bs_col2:
                st.metric("Overall Win Rate", f"{overall_wr:.1f}%",
                          delta=f"{total_wins}W / {total_detected - total_wins}L")
            with bs_col3:
                st.metric("Avg Move After Breakout", f"{avg_move:.2f}%")
            with bs_col4:
                buy_wr = sum(1 for r in buy_results if r.get('result') == 'WIN') / max(len(buy_results), 1) * 100
                st.metric("BUY Signal Win Rate", f"{buy_wr:.1f}%", f"{len(buy_results)} signals")

            # Backtest table (last 30 entries)
            st.markdown("#### Recent Backtest Results (last 30)")
            bt_table_rows = []
            for r in bt_results[-30:]:
                ts = r['time']
                time_str = ts.strftime('%d-%b %H:%M') if hasattr(ts, 'strftime') else str(ts)
                sig = r['signal']
                sig_lbl = '🟢 BUY' if sig == 'BUY' else '🔴 SELL'
                result = r.get('result', '')
                res_lbl = '✅ WIN' if result == 'WIN' else '❌ LOSS'
                bt_table_rows.append({
                    'Time': time_str,
                    'Pattern': r['pattern'],
                    'Sentiment': r['sentiment'],
                    'Signal': sig_lbl,
                    'Entry (₹)': f"{r['entry']:.1f}",
                    'SL (₹)': f"{r['stoploss']:.1f}",
                    'Target (₹)': f"{r['target']:.1f}",
                    'RR': f"{r['rr']:.2f}",
                    'Actual Move %': f"{r.get('actual_move_pct', 0):+.2f}%",
                    'Result': res_lbl,
                    'Confidence': r['confidence'],
                })
            if bt_table_rows:
                st.dataframe(pd.DataFrame(bt_table_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough historical data to run backtest scan.")
    else:
        st.info("Load more data (increase 'Days Back') to enable backtest scan.")


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

    # Initialize session state for PCR of Total Change in OI history
    if 'pcr_chgoi_history' not in st.session_state:
        st.session_state.pcr_chgoi_history = []
    if 'pcr_chgoi_last_valid' not in st.session_state:
        st.session_state.pcr_chgoi_last_valid = None

    # Initialize session state for per-strike ChgOI PCR history (for comparison view)
    if 'pcr_chgoi_strike_history' not in st.session_state:
        st.session_state.pcr_chgoi_strike_history = []
    if 'pcr_chgoi_strike_last_valid' not in st.session_state:
        st.session_state.pcr_chgoi_strike_last_valid = None
    if 'pcr_chgoi_strike_current_strikes' not in st.session_state:
        st.session_state.pcr_chgoi_strike_current_strikes = []
    if 'pcr_telegram_last_sent' not in st.session_state:
        st.session_state.pcr_telegram_last_sent = None

    # Initialize session state for Composite Direction Signal history
    if 'composite_signal_history' not in st.session_state:
        st.session_state.composite_signal_history = []
    if 'composite_signal_last_valid' not in st.session_state:
        st.session_state.composite_signal_last_valid = None

    # Initialize session state for Total GEX time-series history
    if 'total_gex_history' not in st.session_state:
        st.session_state.total_gex_history = []
    if 'total_gex_last_valid' not in st.session_state:
        st.session_state.total_gex_last_valid = None

    # Initialize session state for Order Book Depth key levels history
    if 'depth_history' not in st.session_state:
        st.session_state.depth_history = []

    # Initialize session state for Volume PCR per-strike history (ATM±2)
    if 'vol_pcr_history' not in st.session_state:
        st.session_state.vol_pcr_history = []
    if 'vol_pcr_current_strikes' not in st.session_state:
        st.session_state.vol_pcr_current_strikes = []

    # Initialize session state for ATM Straddle history
    if 'straddle_history' not in st.session_state:
        st.session_state.straddle_history = []

    # Initialize session state for Delta & Gamma per-strike history
    if 'delta_gamma_history' not in st.session_state:
        st.session_state.delta_gamma_history = []
    if 'delta_gamma_last_valid' not in st.session_state:
        st.session_state.delta_gamma_last_valid = None

    # Initialize session state for IV Skew & Pressure history
    if 'iv_skew_history' not in st.session_state:
        st.session_state.iv_skew_history = []
    if 'iv_skew_last_valid' not in st.session_state:
        st.session_state.iv_skew_last_valid = None
    if 'pressure_history' not in st.session_state:
        st.session_state.pressure_history = []
    if 'pressure_last_valid' not in st.session_state:
        st.session_state.pressure_last_valid = None

    # Initialize session state for Pro Trader Dashboard
    if 'pro_trader_history' not in st.session_state:
        st.session_state.pro_trader_history = []
    if 'pro_smart_signal_last' not in st.session_state:
        st.session_state.pro_smart_signal_last = None
    # Initialize session state for Unified Sentiment Engine
    if 'sentiment_history' not in st.session_state:
        st.session_state.sentiment_history = []
    if 'itm_last_alert' not in st.session_state:
        st.session_state.itm_last_alert = (None, None)

    # ===== NEW ENGINE HISTORIES =====
    if 'spike_history' not in st.session_state:
        st.session_state.spike_history = []          # Options Spike Detector snapshots
    if 'expiry_spike_history' not in st.session_state:
        st.session_state.expiry_spike_history = []   # Expiry Spike Detector snapshots
    if 'gamma_seq_history' not in st.session_state:
        st.session_state.gamma_seq_history = []      # Gamma Sequence snapshots
    if 'expiry_intel_history' not in st.session_state:
        st.session_state.expiry_intel_history = []   # Expiry Day Intelligence snapshots
    if 'last_spike_alert' not in st.session_state:
        st.session_state.last_spike_alert = None
    if 'last_gamma_alert' not in st.session_state:
        st.session_state.last_gamma_alert = None
    if 'last_expiry_spike_alert' not in st.session_state:
        st.session_state.last_expiry_spike_alert = None
    # Candlestick Intelligence Engine
    if 'cie_signal_history' not in st.session_state:
        st.session_state.cie_signal_history = []   # list of signal dicts with timestamp
    if 'cie_last_alert' not in st.session_state:
        st.session_state.cie_last_alert = {}        # pattern -> last alert datetime

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

    # ── Backtesting Section ────────────────────────────────────────────────
    st.sidebar.header("🔍 Backtesting")
    backtest_mode = st.sidebar.checkbox(
        "Enable Backtest Mode", value=False,
        help="Select any past trading day (up to 50 days back) to replay the chart with HTF S/R, VOB, and candle patterns"
    )
    backtest_date = None
    if backtest_mode:
        _min_bt = (datetime.now() - timedelta(days=50)).date()
        _max_bt = (datetime.now() - timedelta(days=1)).date()
        backtest_date = st.sidebar.date_input(
            "Select Backtest Date",
            value=_max_bt,
            min_value=_min_bt,
            max_value=_max_bt,
            help="Pick any past trading day up to 50 days back"
        )
        if backtest_date.weekday() >= 5:
            st.sidebar.warning(f"⚠️ {backtest_date.strftime('%A %d %b %Y')} is a weekend — markets are closed. Please pick a weekday.")
            backtest_date = None
        else:
            st.sidebar.success(f"📅 Backtesting: {backtest_date.strftime('%d %b %Y (%A)')}")
            st.sidebar.info(
                "Chart will show:\n"
                "• 5-min candles for selected date\n"
                "• 🟢 HTF Support Levels\n"
                "• 🔴 HTF Resistance Levels\n"
                "• 🟩 VOB Bullish zones\n"
                "• 🟪 VOB Bearish zones\n"
                "• 🕯️ Candle Patterns"
            )
    
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

    # ===== MARKET OVERVIEW (NIFTY + SENSEX) =====
    show_market_overview(api, interval=timeframes.get(selected_timeframe, "1"), days_back=days_back)

    # ===== BANK NIFTY MULTI-TICKER DASHBOARD =====
    with st.expander("📊 Bank Nifty Dashboard — Multi-Ticker & Indicator Table", expanded=False):
        show_bn_dashboard(interval=timeframes.get(selected_timeframe, "1"))

    # Main layout - Trading chart and Options analysis side by side
    col1, col2 = st.columns([2, 1])

    # Initialize pivots variable (will be populated in col1, used in col2)
    pivots = None
    # Initialize VOB data (will be populated in col1, used in col2 for S/R tables)
    vob_data = None

    with col1:
        st.header("📈 Trading Chart")

        # ── Backtest banner ────────────────────────────────────────────────
        if backtest_mode and backtest_date:
            st.info(
                f"🔍 **Backtest Mode** — Viewing **{backtest_date.strftime('%d %b %Y (%A)')}**  \n"
                "HTF Support/Resistance & VOB are computed from the 30 days of data preceding the selected date."
            )

        # Data fetching strategy
        df = pd.DataFrame()
        current_price = None

        if backtest_mode and backtest_date:
            # ── Backtesting: fetch 30 days of context before selected date ──
            _bt_ist = pytz.timezone('Asia/Kolkata')
            _bt_to = datetime.combine(backtest_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=_bt_ist)
            _bt_from = datetime.combine(backtest_date - timedelta(days=30), datetime.min.time()).replace(tzinfo=_bt_ist)
            with st.spinner(f"Fetching historical data for {backtest_date.strftime('%d %b %Y')}..."):
                _bt_raw = api.get_intraday_data_range(
                    security_id="13",
                    exchange_segment="IDX_I",
                    instrument="INDEX",
                    interval=interval,
                    from_date=_bt_from,
                    to_date=_bt_to,
                )
                if _bt_raw:
                    df = process_candle_data(_bt_raw, interval)
        elif use_cache:
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

        # Calculate Volume Order Blocks (VOB) from full df (HTF context)
        vob_blocks_for_chart = None
        if not df.empty and len(df) > 30:
            try:
                vob_detector = VolumeOrderBlocks(sensitivity=5)
                _, vob_blocks_for_chart = vob_detector.get_sr_levels(df)
            except Exception:
                vob_blocks_for_chart = None

        # Create and display chart
        if not df.empty:
            # ── Determine the target date (today or backtest date) ────────
            _ist_tz = pytz.timezone('Asia/Kolkata')
            if backtest_mode and backtest_date:
                _target_date = backtest_date
                _date_label = backtest_date.strftime('%d %b %Y')
                _chart_label = f"Backtesting: {_date_label}"
            else:
                _target_date = datetime.now(_ist_tz).date()
                _date_label = 'Today'
                _chart_label = 'Today'

            # ── Filter df to the target date ──────────────────────────────
            _df_today = df.copy()
            try:
                _df_today = df[pd.to_datetime(df['datetime']).dt.tz_convert('Asia/Kolkata').dt.date == _target_date].copy()
            except Exception:
                _df_today = df[df['datetime'].apply(
                    lambda x: x.date() if hasattr(x, 'date') else x
                ) == _target_date].copy()
            if _df_today.empty:
                _df_today = df.copy()  # fallback
            _df_today = _df_today.reset_index(drop=True)

            # ── Compute time-series indicators on the display df ──────────
            # Must run AFTER filtering so indices align with _df_today
            poc_data_for_chart = None
            if len(_df_today) > 30:
                try:
                    poc_calculator = TriplePOC(period1=10, period2=25, period3=70)
                    poc_data_for_chart = poc_calculator.calculate_all_pocs(_df_today)
                except Exception:
                    poc_data_for_chart = None

            swing_data_for_chart = None
            if len(_df_today) > 30:
                try:
                    swing_calculator = FutureSwing(swing_length=30, history_samples=5, calc_type='Average')
                    swing_data_for_chart = swing_calculator.analyze(_df_today)
                except Exception:
                    swing_data_for_chart = None

            rsi_sz_data_for_chart = None
            if len(_df_today) > 20:
                try:
                    rsi_sz_calculator = RSIVolatilitySuppression(rsi_length=14, vol_length=5)
                    rsi_sz_data_for_chart = rsi_sz_calculator.analyze(_df_today)
                except Exception:
                    rsi_sz_data_for_chart = None

            ultimate_rsi_data_for_chart = None
            if len(_df_today) > 14:
                try:
                    ursi_calculator = UltimateRSI(length=7, smo_type='RMA', signal_length=14, signal_type='EMA', ob_value=70, os_value=40)
                    ultimate_rsi_data_for_chart = ursi_calculator.calculate(_df_today)
                except Exception:
                    ultimate_rsi_data_for_chart = None

            # ── Detect candle patterns for chart markers ──────────────────
            _chart_candle_markers = _detect_chart_candle_types(_df_today)

            # ── Detect geometric / reversal patterns ──────────────────────
            _geo_detector = GeometricPatternDetector()
            _geo_patterns_live = []
            try:
                _geo_patterns_live = _geo_detector.detect_all(_df_today)
            except Exception:
                _geo_patterns_live = []

            fig = create_candlestick_chart(
                _df_today,
                f"Nifty 50 - {selected_timeframe} Chart ({_chart_label}) {'with Pivot Levels' if show_pivots else ''}",
                interval,
                show_pivots=show_pivots,
                pivot_settings=pivot_settings,
                vob_blocks=vob_blocks_for_chart,
                poc_data=poc_data_for_chart,
                swing_data=swing_data_for_chart,
                rsi_sz_data=rsi_sz_data_for_chart,
                ultimate_rsi_data=ultimate_rsi_data_for_chart,
                candle_patterns=_chart_candle_markers,
                geo_patterns=_geo_patterns_live
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

            # ── Pre-compute HTF pivot + VOB levels for candle pattern enrichment ──
            _cp_htf_supports    = []
            _cp_htf_resistances = []
            _cp_vob_supports    = []
            _cp_vob_resistances = []
            if len(df) > 50:
                try:
                    _cp_pivots_raw = cached_pivot_calculation(df.to_json(), pivot_settings or {})
                    _cp_htf_supports    = sorted([p['value'] for p in _cp_pivots_raw if p['type'] == 'low'], reverse=True)
                    _cp_htf_resistances = sorted([p['value'] for p in _cp_pivots_raw if p['type'] == 'high'])
                except Exception:
                    pass
            if len(df) > 30:
                try:
                    _cp_vob_det = VolumeOrderBlocks(sensitivity=5)
                    _cp_vob_sr, _cp_vob_blks = _cp_vob_det.get_sr_levels(df)
                    _cp_vob_supports    = sorted(_cp_vob_sr.get('support', []), reverse=True)
                    _cp_vob_resistances = sorted(_cp_vob_sr.get('resistance', []))
                except Exception:
                    pass

            def _nearest_level(price, levels):
                """Return (nearest_level, distance_pts, distance_pct) or (None,None,None)."""
                if not levels:
                    return None, None, None
                closest = min(levels, key=lambda v: abs(v - price))
                dist_pts = abs(price - closest)
                dist_pct = dist_pts / price * 100 if price else 0
                return closest, dist_pts, dist_pct

            def _nearest_vob(price, vob_levels):
                """Return nearest VOB level as string, or '—'."""
                if not vob_levels:
                    return '—'
                closest = min(vob_levels, key=lambda v: abs(v - price))
                dist_pct = abs(price - closest) / price * 100 if price else 0
                return f"₹{closest:.0f} ({dist_pct:.2f}% away)"

            # ── Candle Types Table ────────────────────────────────────────
            if _chart_candle_markers:
                st.markdown(f"### 🕯️ Candle Patterns Detected — {_date_label}")
                _ctype_rows = []
                for _cp in reversed(_chart_candle_markers):
                    _ts = _cp['time']
                    _time_str = _ts.strftime('%H:%M') if hasattr(_ts, 'strftime') else str(_ts)
                    _dir = _cp['direction']
                    _dir_lbl = '🟢 BUY' if _dir == 'BUY' else ('🔴 SELL' if _dir == 'SELL' else '🟡 NEUTRAL')
                    _price = _cp['price']

                    # For BUY → show nearest HTF support + distance + VOB support
                    # For SELL → show nearest HTF resistance + distance + VOB resistance
                    if _dir == 'BUY':
                        _htf_level, _dist_pts, _dist_pct = _nearest_level(_price, _cp_htf_supports)
                        _htf_label = (f"₹{_htf_level:.0f} ({_dist_pct:.2f}% below)"
                                      if _htf_level else '—')
                        _vob_label = _nearest_vob(_price, _cp_vob_supports)
                        _htf_col   = 'Nearest HTF Support'
                        _vob_col   = 'Nearest VOB Support'
                    elif _dir == 'SELL':
                        _htf_level, _dist_pts, _dist_pct = _nearest_level(_price, _cp_htf_resistances)
                        _htf_label = (f"₹{_htf_level:.0f} ({_dist_pct:.2f}% above)"
                                      if _htf_level else '—')
                        _vob_label = _nearest_vob(_price, _cp_vob_resistances)
                        _htf_col   = 'Nearest HTF Resistance'
                        _vob_col   = 'Nearest VOB Resistance'
                    else:
                        # NEUTRAL — show both nearest support and resistance
                        _sup_lvl, _sup_pts, _sup_pct = _nearest_level(_price, _cp_htf_supports)
                        _res_lvl, _res_pts, _res_pct = _nearest_level(_price, _cp_htf_resistances)
                        _htf_label = (f"S ₹{_sup_lvl:.0f} / R ₹{_res_lvl:.0f}"
                                      if (_sup_lvl and _res_lvl) else '—')
                        _vob_label = '—'
                        _htf_col   = 'HTF S/R Context'
                        _vob_col   = 'VOB'

                    _ctype_rows.append({
                        'Time': _time_str,
                        'Pattern': _cp['pattern'],
                        'Signal': _dir_lbl,
                        'Price (₹)': f"{_price:.1f}",
                        'High (₹)': f"{_cp['high']:.1f}",
                        'Low (₹)': f"{_cp['low']:.1f}",
                        'Nearest HTF Pivot': _htf_label,
                        'Nearest VOB': _vob_label,
                    })
                st.dataframe(pd.DataFrame(_ctype_rows), use_container_width=True, hide_index=True)

            # ── Geometric & Reversal Pattern Analysis ──────────────────────
            render_geo_pattern_analysis(_df_today, df, date_label=_date_label)

            # ── HTF Support & Resistance + VOB Summary ────────────────────
            st.markdown("---")
            _sr_label = f"HTF Support & Resistance Levels — {_date_label}" if (backtest_mode and backtest_date) else "HTF Support & Resistance Levels"
            st.markdown(f"## 📊 {_sr_label}")

            _htf_pivots_raw = []
            if show_pivots and len(df) > 50:
                try:
                    _htf_df_json = df.to_json()
                    _htf_pivots_raw = cached_pivot_calculation(_htf_df_json, pivot_settings or {})
                except Exception:
                    _htf_pivots_raw = []

            _htf_supports = sorted([p['value'] for p in _htf_pivots_raw if p['type'] == 'low'], reverse=True)
            _htf_resistances = sorted([p['value'] for p in _htf_pivots_raw if p['type'] == 'high'])

            _vob_supports = []
            _vob_resistances = []
            if len(df) > 30:
                try:
                    _vob_det = VolumeOrderBlocks(sensitivity=5)
                    _vob_sr, _ = _vob_det.get_sr_levels(df)
                    _vob_supports = sorted([v for v in _vob_sr.get('support', [])], reverse=True)
                    _vob_resistances = sorted([v for v in _vob_sr.get('resistance', [])])
                except Exception:
                    pass

            _col_sup, _col_res = st.columns(2)
            with _col_sup:
                st.markdown("### 🟢 Support Levels")
                _sup_rows = []
                for _v in _htf_supports[:10]:
                    _sup_rows.append({'Type': 'HTF Pivot', 'Level (₹)': f"{_v:.1f}"})
                for _v in _vob_supports[:5]:
                    _sup_rows.append({'Type': 'VOB↑', 'Level (₹)': f"{_v:.1f}"})
                if _sup_rows:
                    st.dataframe(pd.DataFrame(_sup_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No support levels detected")

            with _col_res:
                st.markdown("### 🔴 Resistance Levels")
                _res_rows = []
                for _v in _htf_resistances[:10]:
                    _res_rows.append({'Type': 'HTF Pivot', 'Level (₹)': f"{_v:.1f}"})
                for _v in _vob_resistances[:5]:
                    _res_rows.append({'Type': 'VOB↓', 'Level (₹)': f"{_v:.1f}"})
                if _res_rows:
                    st.dataframe(pd.DataFrame(_res_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No resistance levels detected")

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

        else:
            st.error("No data available. Please check your API credentials and try again.")
    
    with col2:
        st.header("📊 Options Analysis")

        # Options chain analysis with expiry selection (pass pivot data and VOB data for HTF S/R table)
        # Pass live spot price from LTP API to avoid stale option chain last_price for ATM calculation
        option_data = analyze_option_chain(selected_expiry, pivots, vob_data, live_spot_price=current_price)

        if option_data and option_data.get('underlying'):
            underlying_price = option_data['underlying']
            df_summary = option_data['df_summary']
            st.info(f"**NIFTY SPOT:** {underlying_price:.2f}")

        else:
            option_data = None

    # ===== OPTIONS CHAIN AND HTF S/R TABLES BELOW CHART =====
    if option_data and option_data.get('underlying'):
        st.markdown("---")
        # Anchor for sidebar navigation
        st.markdown('<a name="smart-money-panel"></a>', unsafe_allow_html=True)
        st.header("📊 Options Chain Analysis")
        st.info(
            "⬇️ **Smart Money & Market Sentiment Analysis** panel is directly below — scroll down to see "
            "OI activity, sentiment score, trap & breakout detection, support/resistance strength, and key levels."
        )

        # ── 🧠 SMART MONEY PANEL — shown first so it's always visible ────────
        render_smart_money_master_analysis(option_data, current_price)

        # OI Change metrics
        st.markdown("## Open Interest Change (in Lakhs)")
        oi_col1, oi_col2 = st.columns(2)
        with oi_col1:
            st.metric("CALL ΔOI", f"{option_data['total_ce_change']:+.1f}L", delta_color="inverse")
        with oi_col2:
            st.metric("PUT ΔOI", f"{option_data['total_pe_change']:+.1f}L", delta_color="normal")

        # ===== STRIKE PRICE vs LTP TABLE (ATM ± 5) =====
        st.markdown("---")
        st.markdown("## Strike Price vs LTP (ATM ± 5)")

        df_summary_ltp = option_data.get('df_summary')
        atm_strike_ltp = option_data.get('atm_strike')

        if (df_summary_ltp is not None
                and 'lastPrice_CE' in df_summary_ltp.columns
                and 'lastPrice_PE' in df_summary_ltp.columns):

            NIFTY_LOT_SIZE = 65
            _expiry = option_data.get('expiry', '')

            # Collect scrp_cd columns if available (for order placement)
            ltp_extra = [c for c in ('scrp_cd_CE', 'scrp_cd_PE') if c in df_summary_ltp.columns]
            ltp_tbl = df_summary_ltp[['Strike', 'lastPrice_CE', 'lastPrice_PE', 'Zone'] + ltp_extra].copy()
            ltp_tbl = ltp_tbl.sort_values('Strike', ascending=False).reset_index(drop=True)

            has_scrp = 'scrp_cd_CE' in ltp_tbl.columns and 'scrp_cd_PE' in ltp_tbl.columns
            if not has_scrp:
                st.info("Security IDs (scrp_cd) not available in option chain data — Buy buttons disabled.")

            # Build expiry suffix for trading symbol e.g. "25FEB27"
            try:
                from datetime import datetime as _dt
                _exp_sfx = _dt.strptime(_expiry, "%Y-%m-%d").strftime('%y%b%d').upper()
            except Exception:
                _exp_sfx = ''

            # Header row
            hdr = st.columns([2, 1.5, 1, 2, 1, 2, 1.5])
            for col_h, label, color in zip(
                hdr,
                ["CE LTP", "CE Val (x65)", "BUY CE", "STRIKE", "BUY PE", "PE LTP", "PE Val (x65)"],
                ["#00cc66", "#00cc66", "#aaa", "#FFD700", "#aaa", "#ff6b6b", "#ff6b6b"]
            ):
                col_h.markdown(f"<b style='color:{color}'>{label}</b>", unsafe_allow_html=True)

            st.markdown("<hr style='margin:4px 0; border-color:#333'>", unsafe_allow_html=True)

            for _, row in ltp_tbl.iterrows():
                strike = int(row['Strike'])
                ce_ltp = float(row['lastPrice_CE'])
                pe_ltp = float(row['lastPrice_PE'])
                ce_val = ce_ltp * NIFTY_LOT_SIZE
                pe_val = pe_ltp * NIFTY_LOT_SIZE
                is_atm = row['Zone'] == 'ATM'
                scrp_ce = str(row['scrp_cd_CE']) if has_scrp else ''
                scrp_pe = str(row['scrp_cd_PE']) if has_scrp else ''
                ts_ce = f"NIFTY{_exp_sfx}{strike}CE" if _exp_sfx else f"NIFTY{strike}CE"
                ts_pe = f"NIFTY{_exp_sfx}{strike}PE" if _exp_sfx else f"NIFTY{strike}PE"

                ce_color = "#00ff88" if is_atm else "#00cc66"
                pe_color = "#ff8888" if is_atm else "#ff6b6b"
                fw = "bold" if is_atm else "normal"
                strike_style = "color:#FFD700; font-weight:bold; font-size:1.1em" if is_atm else "color:#c0c0c0"

                r = st.columns([2, 1.5, 1, 2, 1, 2, 1.5])
                r[0].markdown(f"<span style='color:{ce_color}; font-weight:{fw}'>{ce_ltp:.2f}</span>", unsafe_allow_html=True)
                r[1].markdown(f"<span style='color:{ce_color}'>₹{ce_val:,.0f}</span>", unsafe_allow_html=True)
                buy_ce = r[2].button("BUY", key=f"buy_ce_{strike}", type="primary",
                                     disabled=not (has_scrp and scrp_ce))
                r[3].markdown(f"<div style='text-align:center'><span style='{strike_style}'>{strike}</span></div>",
                               unsafe_allow_html=True)
                buy_pe = r[4].button("BUY", key=f"buy_pe_{strike}", type="primary",
                                     disabled=not (has_scrp and scrp_pe))
                r[5].markdown(f"<span style='color:{pe_color}; font-weight:{fw}'>{pe_ltp:.2f}</span>", unsafe_allow_html=True)
                r[6].markdown(f"<span style='color:{pe_color}'>₹{pe_val:,.0f}</span>", unsafe_allow_html=True)

                if buy_ce:
                    ok, res = api.place_order(scrp_ce, ts_ce)
                    if ok:
                        st.success(f"CE BUY order placed for {ts_ce} | Order ID: {res.get('orderId', res)}")
                    else:
                        st.error(f"CE BUY failed for {ts_ce}: {res}")

                if buy_pe:
                    ok, res = api.place_order(scrp_pe, ts_pe)
                    if ok:
                        st.success(f"PE BUY order placed for {ts_pe} | Order ID: {res.get('orderId', res)}")
                    else:
                        st.error(f"PE BUY failed for {ts_pe}: {res}")

        # ===== ATM ±3 STRIKE DATA TABLE =====
        st.markdown("---")
        st.markdown("## 📋 ATM ±3 Strike Data — Full Tabulation (7 Strikes)")
        try:
            _t5_src = option_data.get('df_summary') if option_data else None
            if _t5_src is not None:
                _t5_atm_idx = _t5_src[_t5_src['Zone'] == 'ATM'].index
                if len(_t5_atm_idx) > 0:
                    _t5_atm_pos = _t5_src.index.get_loc(_t5_atm_idx[0])
                    _t5_start   = max(0, _t5_atm_pos - 3)
                    _t5_end     = min(len(_t5_src), _t5_atm_pos + 4)
                    _t5         = _t5_src.iloc[_t5_start:_t5_end].copy().reset_index(drop=True)

                    _t5_atm_strike  = _t5[_t5['Zone'] == 'ATM']['Strike'].values[0]
                    _t5_stk_list    = _t5['Strike'].tolist()
                    _t5_atm_i       = _t5_stk_list.index(_t5_atm_strike)
                    def _t5_zone(i):
                        d = i - _t5_atm_i
                        return '🟡 ATM' if d == 0 else (f'🟣 ITM{d}' if d < 0 else f'🔵 OTM+{d}')

                    # Underlying direction for OI type
                    _t5_und = option_data.get('underlying', 0) or 0
                    if '_oi_prev_underlying' not in st.session_state:
                        st.session_state['_oi_prev_underlying'] = _t5_und
                    _t5_und_up = _t5_und >= st.session_state['_oi_prev_underlying']
                    st.session_state['_oi_prev_underlying'] = _t5_und

                    def _t5_oi_type(chgoi, price_up):
                        oi_up = (chgoi or 0) > 0
                        if price_up and oi_up:     return "🟢 Long Build-up"
                        if not price_up and oi_up: return "🔴 Short Build-up"
                        if price_up and not oi_up: return "🟡 Short Covering"
                        return "🟠 Long Unwinding"

                    _t5_tbl = pd.DataFrame()
                    _t5_tbl['Strike']   = _t5['Strike']
                    _t5_tbl['Zone']     = [_t5_zone(i) for i in range(len(_t5))]

                    # LTP + Straddle
                    if 'lastPrice_CE' in _t5.columns:
                        _t5_tbl['CE LTP']   = _t5['lastPrice_CE'].round(2)
                        _t5_tbl['PE LTP']   = _t5['lastPrice_PE'].round(2)
                        _t5_tbl['Straddle'] = (_t5['lastPrice_CE'] + _t5['lastPrice_PE']).round(2)

                    # PCR (OI)
                    if 'PCR' in _t5.columns:
                        _t5_tbl['PCR OI']   = _t5['PCR']
                        _t5_tbl['OI Sig']   = _t5['PCR'].apply(
                            lambda v: '🟢 Bull' if v > 1.2 else ('🔴 Bear' if v < 0.7 else '🟡 Ntrl'))

                    # PCR (ChgOI)
                    if 'changeinOpenInterest_CE' in _t5.columns:
                        _ce_chg = _t5['changeinOpenInterest_CE'].replace(0, np.nan)
                        _t5_tbl['PCR ΔOI']  = (_t5['changeinOpenInterest_PE'] / _ce_chg).round(3)
                        _t5_tbl['ΔOI Sig']  = _t5_tbl['PCR ΔOI'].apply(
                            lambda v: '🟢 Bull' if (pd.notna(v) and v > 1.2) else (
                                      '🔴 Bear' if (pd.notna(v) and v < 0.7) else '🟡 Ntrl'))

                    # Vol PCR
                    if 'totalTradedVolume_CE' in _t5.columns:
                        _ce_vol = _t5['totalTradedVolume_CE'].replace(0, np.nan)
                        _t5_tbl['Vol PCR']  = (_t5['totalTradedVolume_PE'] / _ce_vol).round(3)

                    # OI in Lakhs
                    if 'openInterest_CE' in _t5.columns:
                        _t5_tbl['CE OI(L)'] = (_t5['openInterest_CE'] / 100000).round(2)
                        _t5_tbl['PE OI(L)'] = (_t5['openInterest_PE'] / 100000).round(2)

                    # ΔOI + OI Type (separately for CE and PE)
                    if 'changeinOpenInterest_CE' in _t5.columns:
                        _t5_tbl['CE ΔOI']    = _t5['changeinOpenInterest_CE'].fillna(0).astype(int)
                        _t5_tbl['CE OI Type'] = _t5['changeinOpenInterest_CE'].apply(
                            lambda v: _t5_oi_type(v, _t5_und_up))
                        _t5_tbl['PE ΔOI']    = _t5['changeinOpenInterest_PE'].fillna(0).astype(int)
                        _t5_tbl['PE OI Type'] = _t5['changeinOpenInterest_PE'].apply(
                            lambda v: _t5_oi_type(v, _t5_und_up))

                    # IV
                    if 'impliedVolatility_CE' in _t5.columns:
                        _t5_tbl['IV CE']    = _t5['impliedVolatility_CE'].round(2)
                        _t5_tbl['IV PE']    = _t5['impliedVolatility_PE'].round(2)

                    # GEX Net (Lakhs)
                    if 'GammaExp_Net' in _t5.columns:
                        _t5_tbl['GEX(L)']   = (_t5['GammaExp_Net'] / 100000).round(2)

                    # Bias + Verdict
                    if 'BiasScore' in _t5.columns:
                        _t5_tbl['Bias%']    = _t5['BiasScore'].round(1)
                    if 'Verdict' in _t5.columns:
                        _t5_tbl['Verdict']  = _t5['Verdict']

                    # Row styling
                    def _t5_style(row):
                        if row['Zone'] == '🟡 ATM':
                            return ['background-color: rgba(255,200,0,0.18)'] * len(row)
                        pcr = row.get('PCR OI', 1.0)
                        if pd.notna(pcr) and pcr > 1.2:
                            return ['background-color: rgba(0,255,136,0.07)'] * len(row)
                        if pd.notna(pcr) and pcr < 0.7:
                            return ['background-color: rgba(255,68,68,0.07)'] * len(row)
                        return [''] * len(row)

                    st.dataframe(
                        _t5_tbl.style.apply(_t5_style, axis=1),
                        use_container_width=True,
                        hide_index=True,
                    )
                    _t5_und_dir = "↑" if _t5_und_up else "↓"
                    st.caption(
                        f"📍 ATM ₹{_t5_atm_strike} · ATM±3 ({len(_t5_tbl)} strikes) · "
                        f"Underlying {_t5_und_dir} ₹{_t5_und:.0f} · "
                        f"🟢 Long Build-up  🔴 Short Build-up  🟡 Short Covering  🟠 Long Unwinding"
                    )
                else:
                    st.info("ATM strike not identified in current data.")
            else:
                st.info("Option data not available yet — wait for first refresh.")
        except Exception as _t5_exc:
            st.warning(f"ATM ±3 table error: {str(_t5_exc)[:120]}")

        # Option Chain Bias Summary Table
        st.markdown("---")
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

        # ===== KEY LEVELS FROM ORDER BOOK DEPTH CHART (time-series) =====
        st.markdown("---")
        _depth_df_summary = option_data.get('df_summary')
        if _depth_df_summary is not None:
            _has_bid = 'bidQty_PE' in _depth_df_summary.columns and not _depth_df_summary['bidQty_PE'].isna().all()
            _has_ask = 'askQty_CE' in _depth_df_summary.columns and not _depth_df_summary['askQty_CE'].isna().all()
            if _has_bid or _has_ask:
                _ist = pytz.timezone('Asia/Kolkata')
                _depth_now = datetime.now(_ist)
                _depth_entry = {'time': _depth_now}
                if _has_bid:
                    _top3_sup = _depth_df_summary.nlargest(3, 'bidQty_PE')[['Strike', 'bidQty_PE']].copy()
                    _top3_sup = _top3_sup.sort_values('Strike', ascending=False).reset_index(drop=True)
                    for _i, _r in _top3_sup.iterrows():
                        _depth_entry[f'S{_i+1}_qty'] = _r['bidQty_PE']
                        _depth_entry[f'S{_i+1}_price'] = _r['Strike']
                if _has_ask:
                    _top3_res = _depth_df_summary.nlargest(3, 'askQty_CE')[['Strike', 'askQty_CE']].copy()
                    _top3_res = _top3_res.sort_values('Strike').reset_index(drop=True)
                    for _i, _r in _top3_res.iterrows():
                        _depth_entry[f'R{_i+1}_qty'] = _r['askQty_CE']
                        _depth_entry[f'R{_i+1}_price'] = _r['Strike']
                _should_add_depth = True
                if st.session_state.depth_history:
                    _last_depth = st.session_state.depth_history[-1]
                    if (_depth_now - _last_depth['time']).total_seconds() < 30:
                        _should_add_depth = False
                if _should_add_depth:
                    st.session_state.depth_history.append(_depth_entry)
                    if len(st.session_state.depth_history) > 200:
                        st.session_state.depth_history = st.session_state.depth_history[-200:]

        st.markdown("### 📊 Key Levels from Order Book Depth")
        if st.session_state.depth_history:
            _depth_hist_df = pd.DataFrame(st.session_state.depth_history)
            _support_colors = ['#00cc66', '#00aa55', '#008844']
            _resist_colors  = ['#ff4444', '#dd3333', '#bb2222']
            _depth_levels = (
                [('S', i+1, _support_colors[i], 'Support', 'bidQty_PE') for i in range(3)] +
                [('R', i+1, _resist_colors[i],  'Resistance', 'askQty_CE') for i in range(3)]
            )
            _depth_cols = st.columns(6)
            for _col_idx, (_col, (_side, _n, _clr, _label, _)) in enumerate(zip(_depth_cols, _depth_levels)):
                with _col:
                    _qty_col   = f'{_side}{_n}_qty'
                    _price_col = f'{_side}{_n}_price'
                    if _qty_col not in _depth_hist_df.columns:
                        st.info(f"{_side}{_n} N/A")
                        continue
                    _cur_qty = _depth_hist_df[_qty_col].iloc[-1]
                    _cur_price = (
                        _depth_hist_df[_price_col].iloc[-1]
                        if _price_col in _depth_hist_df.columns else None
                    )
                    _price_str = f'₹{_cur_price:,.0f}' if _cur_price is not None else ''
                    _rgb = tuple(int(_clr.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
                    _fill = f'rgba({_rgb[0]},{_rgb[1]},{_rgb[2]},0.15)'
                    _qty_vals = _depth_hist_df[_qty_col].dropna()
                    _max_qty = _qty_vals.max() if len(_qty_vals) > 0 else 1
                    _fig_d = go.Figure()
                    _fig_d.add_trace(go.Scatter(
                        x=_depth_hist_df['time'],
                        y=_depth_hist_df[_qty_col],
                        mode='lines+markers',
                        name=_label,
                        line=dict(color=_clr, width=2),
                        marker=dict(size=3),
                        fill='tozeroy',
                        fillcolor=_fill,
                        hovertemplate=f'{_side}{_n} {_label}<br>Qty: %{{y:,.0f}}<br>Time: %{{x|%H:%M}}<extra></extra>',
                    ))
                    _fig_d.update_layout(
                        title=dict(
                            text=f"{'🟢' if _side=='S' else '🔴'} {_side}{_n} {_label}<br>{_price_str}<br>Qty: {_cur_qty:,.0f}",
                            font=dict(size=11)
                        ),
                        template='plotly_dark',
                        height=300,
                        showlegend=False,
                        margin=dict(l=5, r=10, t=70, b=30),
                        xaxis=dict(tickformat='%H:%M', title='', tickfont=dict(size=8)),
                        yaxis=dict(
                            title='Bid Qty' if _side == 'S' else 'Ask Qty',
                            tickformat=',.0f',
                            range=[0, _max_qty * 1.2],
                            title_font=dict(color=_clr, size=9),
                            tickfont=dict(size=8),
                        ),
                        plot_bgcolor='#1e1e1e',
                        paper_bgcolor='#1e1e1e',
                    )
                    st.plotly_chart(_fig_d, use_container_width=True)
                    _caption = f"{'PE Bid' if _side=='S' else 'CE Ask'} {_cur_qty:,.0f}"
                    if _cur_price is not None:
                        _caption += f" @ {_price_str}"
                    st.caption(_caption)
        else:
            depth_fig = plot_depth_levels(
                option_data.get('df_summary'),
                option_data.get('underlying')
            )
            if depth_fig is not None:
                st.plotly_chart(depth_fig, use_container_width=True)

        # ===== PRE-COMPUTE GEX + TRACK HISTORY (before comparison view) =====
        _gex_pre_summary = option_data.get('df_summary')
        _gex_pre_underlying = option_data.get('underlying')
        gex_data_pre = None
        if _gex_pre_summary is not None and _gex_pre_underlying:
            try:
                gex_data_pre = calculate_dealer_gex(_gex_pre_summary, _gex_pre_underlying)
                if gex_data_pre:
                    st.session_state.gex_last_valid_data = gex_data_pre
                    _gex_df_pre = gex_data_pre['gex_df']
                    _gex_ist = pytz.timezone('Asia/Kolkata')
                    _gex_now = datetime.now(_gex_ist)
                    _gex_entry = {'time': _gex_now, 'total_gex': gex_data_pre['total_gex']}
                    for _, _gr in _gex_df_pre.iterrows():
                        _gex_entry[str(int(_gr['Strike']))] = _gr['Net_GEX']
                    st.session_state.gex_current_strikes = sorted(
                        [int(_gr['Strike']) for _, _gr in _gex_df_pre.iterrows()])
                    _should_gex = True
                    if st.session_state.gex_history:
                        if (_gex_now - st.session_state.gex_history[-1]['time']).total_seconds() < 30:
                            _should_gex = False
                    if _should_gex:
                        st.session_state.gex_history.append(_gex_entry)
                        if len(st.session_state.gex_history) > 200:
                            st.session_state.gex_history = st.session_state.gex_history[-200:]
            except Exception:
                pass

        # ===== COMPOSITE SCORE & VERDICT — TOP DISPLAY =====
        st.markdown("---")
        st.markdown("## 🧭 Composite Direction Signal — PCR × ΔOI × GEX")
        _cs_last   = st.session_state.get('composite_signal_last_valid')
        _cs_hist   = st.session_state.get('composite_signal_history', [])

        if _cs_last:
            _cs_verdict      = _cs_last['verdict']
            _cs_icon         = _cs_last['verdict_icon']
            _cs_color        = _cs_last['verdict_color']
            _cs_desc         = _cs_last['verdict_desc']
            _cs_score        = _cs_last['total_score']
            _cs_pct          = _cs_last['score_pct']
            _cs_gex          = _cs_last['total_net_gex']
            _cs_pcr          = _cs_last['avg_pcr']
            _cs_chgoi        = _cs_last['avg_chgoi']
            _cs_max          = 14.0
            _cs_gex_trending = _cs_gex < -10
            _cs_gex_pinning  = _cs_gex > 10
            _cs_gex_lbl      = 'Trending' if _cs_gex_trending else 'Pinning' if _cs_gex_pinning else 'Neutral'

            # Main verdict card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {_cs_color}15, {_cs_color}30);
                        padding: 22px; border-radius: 15px; border: 3px solid {_cs_color};
                        text-align: center; margin-bottom: 16px;">
                <h1 style="color: {_cs_color}; margin: 0; font-size: 44px;">{_cs_icon} {_cs_verdict}</h1>
                <p style="color: #cccccc; margin: 8px 0 0 0; font-size: 15px;">{_cs_desc}</p>
                <p style="color: {_cs_color}; margin: 5px 0 0 0; font-size: 13px;">
                    Composite Score: {_cs_score:+.1f} / {_cs_max:.0f} ({_cs_pct:+.0f}%) &nbsp;|&nbsp;
                    Net GEX: {_cs_gex:.1f}L ({_cs_gex_lbl})
                </p>
            </div>""", unsafe_allow_html=True)

            # Three metric cards
            _cm1, _cm2, _cm3 = st.columns(3)
            with _cm1:
                _c = "#00ff88" if _cs_pcr > 1.2 else "#ff4444" if _cs_pcr < 0.7 else "#FFD700"
                st.markdown(f"""
                <div style="background:{_c}20;padding:12px;border-radius:10px;border:2px solid {_c};text-align:center;">
                <h4 style="color:{_c};margin:0;">Avg PCR (OI)</h4>
                <h2 style="color:{_c};margin:5px 0;">{_cs_pcr:.2f}</h2>
                <p style="color:white;margin:0;font-size:12px;">{'Bullish' if _cs_pcr > 1.2 else 'Bearish' if _cs_pcr < 0.7 else 'Neutral'}</p>
                </div>""", unsafe_allow_html=True)
            with _cm2:
                _c = "#00ff88" if _cs_chgoi > 1.2 else "#ff4444" if _cs_chgoi < 0.7 else "#FFD700"
                st.markdown(f"""
                <div style="background:{_c}20;padding:12px;border-radius:10px;border:2px solid {_c};text-align:center;">
                <h4 style="color:{_c};margin:0;">Avg PCR (ΔOI)</h4>
                <h2 style="color:{_c};margin:5px 0;">{_cs_chgoi:.2f}</h2>
                <p style="color:white;margin:0;font-size:12px;">{'Bullish' if _cs_chgoi > 1.2 else 'Bearish' if _cs_chgoi < 0.7 else 'Neutral'}</p>
                </div>""", unsafe_allow_html=True)
            with _cm3:
                _c = "#00ff88" if _cs_gex > 10 else "#ff4444" if _cs_gex < -10 else "#FFD700"
                st.markdown(f"""
                <div style="background:{_c}20;padding:12px;border-radius:10px;border:2px solid {_c};text-align:center;">
                <h4 style="color:{_c};margin:0;">Total GEX (ATM±2)</h4>
                <h2 style="color:{_c};margin:5px 0;">{_cs_gex:.1f}L</h2>
                <p style="color:white;margin:0;font-size:12px;">{'Pin/Chop' if _cs_gex > 10 else 'Trend/Accel' if _cs_gex < -10 else 'Neutral'}</p>
                </div>""", unsafe_allow_html=True)

            # Time-series chart if history available
            if len(_cs_hist) >= 2:
                _cs_df = pd.DataFrame(_cs_hist)
                _cs_marker_colors = [
                    '#00ff88' if r.get('verdict_numeric', 0) >= 2 else
                    '#90EE90' if r.get('verdict_numeric', 0) == 1 else
                    '#ff4444' if r.get('verdict_numeric', 0) <= -2 else
                    '#FFB6C1' if r.get('verdict_numeric', 0) == -1 else '#FFD700'
                    for _, r in _cs_df.iterrows()
                ]
                _fig_cs = go.Figure()
                _fig_cs.add_trace(go.Scatter(
                    x=_cs_df['time'], y=_cs_df['score_pct'],
                    mode='lines+markers', name='Score %',
                    line=dict(color='#00aaff', width=3),
                    marker=dict(size=8, color=_cs_marker_colors),
                    fill='tozeroy', fillcolor='rgba(0,170,255,0.08)'
                ))
                _y_max = max(abs(_cs_df['score_pct'].max()), abs(_cs_df['score_pct'].min()), 30) * 1.2
                _fig_cs.add_hline(y=0, line_dash="dash", line_color="white", line_width=1.5,
                                  annotation_text="Neutral (0%)", annotation_position="right")
                _fig_cs.add_hline(y=15, line_dash="dot", line_color="#00ff88", line_width=1,
                                  annotation_text="Bullish Zone", annotation_position="right")
                _fig_cs.add_hline(y=-15, line_dash="dot", line_color="#ff4444", line_width=1,
                                  annotation_text="Bearish Zone", annotation_position="right")
                _fig_cs.add_hrect(y0=15, y1=_y_max, fillcolor="rgba(0,255,136,0.06)", line_width=0)
                _fig_cs.add_hrect(y0=-_y_max, y1=-15, fillcolor="rgba(255,68,68,0.06)", line_width=0)
                # Sparse verdict labels
                _step = max(1, len(_cs_df) // 10)
                for _ci, (_cii, _crow) in enumerate(_cs_df.iterrows()):
                    if _ci % _step == 0:
                        _fig_cs.add_annotation(
                            x=_crow['time'], y=_crow['score_pct'],
                            text=_crow['verdict'], showarrow=False,
                            yshift=14, font=dict(size=8, color='white'),
                            bgcolor='rgba(0,0,0,0.5)', borderpad=2
                        )
                # Companion: PCR + GEX sub-charts inline
                _cc1, _cc2 = st.columns(2)
                with _cc1:
                    _fig_cs.update_layout(
                        title=f"Composite Score Time-Series | Now: {_cs_pct:+.0f}% ({_cs_verdict})",
                        template='plotly_dark', height=360,
                        showlegend=False,
                        xaxis=dict(tickformat='%H:%M', title='Time'),
                        yaxis=dict(title='Score %', zeroline=True, zerolinecolor='white'),
                        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                        margin=dict(l=50, r=60, t=55, b=40)
                    )
                    st.plotly_chart(_fig_cs, use_container_width=True)
                with _cc2:
                    _fig_cs2 = go.Figure()
                    if 'avg_pcr' in _cs_df.columns:
                        _fig_cs2.add_trace(go.Scatter(
                            x=_cs_df['time'], y=_cs_df['avg_pcr'],
                            mode='lines+markers', name='PCR OI',
                            line=dict(color='#00aaff', width=2), marker=dict(size=4)
                        ))
                    if 'avg_chgoi' in _cs_df.columns:
                        _fig_cs2.add_trace(go.Scatter(
                            x=_cs_df['time'], y=_cs_df['avg_chgoi'],
                            mode='lines+markers', name='PCR ΔOI',
                            line=dict(color='#ff44ff', width=2), marker=dict(size=4)
                        ))
                    if 'total_gex' in _cs_df.columns:
                        _fig_cs2.add_trace(go.Scatter(
                            x=_cs_df['time'], y=_cs_df['total_gex'],
                            mode='lines', name='GEX (L)',
                            line=dict(color='#FFD700', width=2, dash='dot'),
                            yaxis='y2'
                        ))
                    _fig_cs2.add_hline(y=1.2, line_dash="dot", line_color="#00ff88", line_width=1)
                    _fig_cs2.add_hline(y=0.7, line_dash="dot", line_color="#ff4444", line_width=1)
                    _fig_cs2.add_hline(y=1.0, line_dash="dash", line_color="white", line_width=1)
                    _pcr_all = _cs_df['avg_pcr'].dropna().tolist() + _cs_df['avg_chgoi'].dropna().tolist() + [0.7, 1.2]
                    _fig_cs2.update_layout(
                        title="PCR OI · PCR ΔOI · GEX",
                        template='plotly_dark', height=360,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
                        xaxis=dict(tickformat='%H:%M', title=''),
                        yaxis=dict(title='PCR', range=[max(0, min(_pcr_all)*0.9), max(_pcr_all)*1.1]),
                        yaxis2=dict(title='GEX (L)', overlaying='y', side='right',
                                    showgrid=False, zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'),
                        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                        margin=dict(l=40, r=55, t=55, b=40)
                    )
                    st.plotly_chart(_fig_cs2, use_container_width=True)

            st.caption(f"📊 {len(_cs_hist)} data points · updates every ~30s · "
                       f"Full analysis with per-strike breakdown further below ↓")
        else:
            st.info("🕐 Composite Score builds up after first refresh — data will appear here on the next cycle.")

        # ===== UNIFIED OPTIONS FLOW SENTIMENT ENGINE =====
        st.markdown("---")
        st.markdown("## 🧭 Unified Options Flow Sentiment Engine")
        try:
            _pro_df    = option_data.get('df_summary') if option_data else None
            _pro_spot  = option_data.get('underlying') if option_data else None

            # Ensure optional columns exist
            if _pro_df is not None:
                for _c in ['impliedVolatility_CE', 'impliedVolatility_PE',
                           'bidQty_CE', 'askQty_CE', 'bidQty_PE', 'askQty_PE',
                           'Delta_CE', 'Delta_PE', 'Gamma_CE', 'Gamma_PE',
                           'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                           'totalTradedVolume_CE', 'totalTradedVolume_PE']:
                    if _c not in _pro_df.columns:
                        _pro_df[_c] = 0.0

            if _pro_df is not None and _pro_spot and 'Zone' in _pro_df.columns:
                _pro_atm_idx = _pro_df[_pro_df['Zone'] == 'ATM'].index
                if len(_pro_atm_idx) > 0:
                    _pro_atm_pos = _pro_df.index.get_loc(_pro_atm_idx[0])
                    _pro_start   = max(0, _pro_atm_pos - 2)
                    _pro_end     = min(len(_pro_df), _pro_atm_pos + 3)
                    _pro_slice   = _pro_df.iloc[_pro_start:_pro_end].copy()
                    _pro_atm_val = float(_pro_df[_pro_df['Zone'] == 'ATM']['Strike'].values[0])
                    _pro_strikes = sorted(_pro_slice['Strike'].unique())
                    _pro_step    = int(_pro_strikes[1] - _pro_strikes[0]) if len(_pro_strikes) >= 2 else 50

                    def _pro_lbl(s):
                        d = int(round((s - _pro_atm_val) / _pro_step))
                        if d == 0: return "ATM"
                        return f"ATM+{d}" if d > 0 else f"ATM{d}"

                    _pro_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                    _pro_lot = 25

                    # ── 1. Straddle per strike ──────────────────────────────────
                    _pro_straddles = {}
                    for _, _r in _pro_slice.iterrows():
                        _lbl = _pro_lbl(_r['Strike'])
                        _pro_straddles[_lbl] = round(
                            float(_r.get('lastPrice_CE', 0) or 0) +
                            float(_r.get('lastPrice_PE', 0) or 0), 2)
                    _pro_atm_straddle = _pro_straddles.get('ATM', 0)

                    # ── 2. IV per strike ───────────────────────────────────────
                    _pro_iv_ce, _pro_iv_pe = {}, {}
                    for _, _r in _pro_slice.iterrows():
                        _lbl = _pro_lbl(_r['Strike'])
                        _pro_iv_ce[_lbl] = float(_r.get('impliedVolatility_CE', 0) or 0)
                        _pro_iv_pe[_lbl] = float(_r.get('impliedVolatility_PE', 0) or 0)
                    _pro_avg_iv_ce    = sum(_pro_iv_ce.values()) / max(len(_pro_iv_ce), 1)
                    _pro_avg_iv_pe    = sum(_pro_iv_pe.values()) / max(len(_pro_iv_pe), 1)
                    _pro_market_iv_sk = round(_pro_avg_iv_pe - _pro_avg_iv_ce, 2)

                    # ── 3. PCR (OI, Volume, ΔOI) ──────────────────────────────
                    _pro_ce_oi    = _pro_slice['openInterest_CE'].fillna(0).sum()
                    _pro_pe_oi    = _pro_slice['openInterest_PE'].fillna(0).sum()
                    _pro_pcr_oi   = round(_pro_pe_oi / (_pro_ce_oi + 1e-6), 3)
                    _pro_ce_vol   = _pro_slice['totalTradedVolume_CE'].fillna(0).sum()
                    _pro_pe_vol   = _pro_slice['totalTradedVolume_PE'].fillna(0).sum()
                    _pro_pcr_vol  = round(_pro_pe_vol / (_pro_ce_vol + 1e-6), 3)
                    _pro_ce_chg   = _pro_slice['changeinOpenInterest_CE'].fillna(0).sum()
                    _pro_pe_chg   = _pro_slice['changeinOpenInterest_PE'].fillna(0).sum()
                    _pro_pcr_chg  = round(abs(_pro_pe_chg) / (abs(_pro_ce_chg) + 1e-6), 3)

                    # ── 4. Bid/Ask Pressure ────────────────────────────────────
                    _pro_cp_list, _pro_pp_list = [], []
                    for _, _r in _pro_slice.iterrows():
                        _bc = float(_r.get('bidQty_CE', 0) or 0)
                        _ac = float(_r.get('askQty_CE', 0) or 0)
                        _bp = float(_r.get('bidQty_PE', 0) or 0)
                        _ap = float(_r.get('askQty_PE', 0) or 0)
                        _pro_cp_list.append(_bc / (_bc + _ac + 1e-6))
                        _pro_pp_list.append(_bp / (_bp + _ap + 1e-6))
                    _pro_call_pres = round(sum(_pro_cp_list) / max(len(_pro_cp_list), 1), 4)
                    _pro_put_pres  = round(sum(_pro_pp_list) / max(len(_pro_pp_list), 1), 4)

                    # ── 5. Delta Exposure ──────────────────────────────────────
                    _pro_call_dexp = sum(
                        float(_r.get('Delta_CE', 0) or 0) *
                        float(_r.get('openInterest_CE', 0) or 0) * _pro_lot
                        for _, _r in _pro_slice.iterrows())
                    _pro_put_dexp  = sum(
                        float(_r.get('Delta_PE', 0) or 0) *
                        float(_r.get('openInterest_PE', 0) or 0) * _pro_lot
                        for _, _r in _pro_slice.iterrows())
                    _pro_net_delta = round(_pro_call_dexp + _pro_put_dexp, 0)

                    # ── 6. Net Gamma Exposure (Lakhs) ──────────────────────────
                    _pro_net_gex = round(sum(
                        (float(_r.get('Gamma_CE', 0) or 0) * float(_r.get('openInterest_CE', 0) or 0) -
                         float(_r.get('Gamma_PE', 0) or 0) * float(_r.get('openInterest_PE', 0) or 0))
                        * _pro_lot * _pro_spot
                        for _, _r in _pro_slice.iterrows()) / 100000, 2)

                    # ── Store in history ───────────────────────────────────────
                    _pro_entry = {
                        'time': _pro_now, 'spot': _pro_spot, 'atm_strike': _pro_atm_val,
                        'straddle_atm': _pro_atm_straddle,
                        **{f'straddle_{k}': v for k, v in _pro_straddles.items()},
                        'avg_iv_ce': round(_pro_avg_iv_ce, 2),
                        'avg_iv_pe': round(_pro_avg_iv_pe, 2),
                        'iv_skew': _pro_market_iv_sk,
                        'pcr_oi': _pro_pcr_oi, 'pcr_vol': _pro_pcr_vol, 'pcr_chgoi': _pro_pcr_chg,
                        'call_pressure': _pro_call_pres, 'put_pressure': _pro_put_pres,
                        'net_delta': _pro_net_delta, 'net_gex': _pro_net_gex,
                    }
                    _pro_should_add = (
                        not st.session_state.pro_trader_history or
                        (_pro_now - st.session_state.pro_trader_history[-1]['time']).total_seconds() >= 28
                    )
                    if _pro_should_add:
                        st.session_state.pro_trader_history.append(_pro_entry)
                        if len(st.session_state.pro_trader_history) > 200:
                            st.session_state.pro_trader_history = st.session_state.pro_trader_history[-200:]

                    # ── 7a. Composite Direction Signal (PCR × ΔOI × GEX) ────────
                    _comp_position_labels = ['ITM-2', 'ITM-1', 'ATM', 'OTM+1', 'OTM+2']
                    _comp_weights = [1.0, 1.5, 2.0, 1.5, 1.0]
                    _comp_strike_scores, _comp_strike_details, _comp_per_strike_scores = [], [], {}
                    _comp_data_available = False
                    _verdict = _verdict_icon = _verdict_color = _verdict_desc = None
                    _comp_total_score = _comp_score_pct = 0.0
                    _comp_total_gex = _comp_avg_pcr = _comp_avg_chgoi = 0.0
                    _comp_gex_trending = _comp_gex_pinning = False
                    _comp_max_possible = 14.0
                    _verdict_numeric = 0
                    _slice_has_comp = ('PCR' in _pro_slice.columns and
                                       'changeinOpenInterest_CE' in _pro_slice.columns and
                                       'Gamma_CE' in _pro_slice.columns)
                    if _slice_has_comp:
                        _sc = _pro_slice.copy()
                        _sc['_PCR_ChgOI'] = _sc.apply(
                            lambda r: abs(float(r['changeinOpenInterest_PE']) /
                                          float(r['changeinOpenInterest_CE']))
                            if float(r['changeinOpenInterest_CE']) != 0 else 0.0, axis=1)
                        _sc['_Net_GEX'] = (
                            -1 * _sc['Gamma_CE'].fillna(0) * _sc['openInterest_CE'].fillna(0) *
                            _pro_lot * _pro_spot / 100000 +
                            _sc['Gamma_PE'].fillna(0) * _sc['openInterest_PE'].fillna(0) *
                            _pro_lot * _pro_spot / 100000)
                        _comp_total_gex  = _sc['_Net_GEX'].sum()
                        _comp_avg_pcr    = float(_sc['PCR'].mean()) if 'PCR' in _sc.columns else 1.0
                        _comp_avg_chgoi  = float(_sc['_PCR_ChgOI'].mean())
                        _comp_gex_trending = _comp_total_gex < -10
                        _comp_gex_pinning  = _comp_total_gex > 10
                        for _ci, (_, _cr) in enumerate(_sc.iterrows()):
                            _pv = float(_cr.get('PCR', 1.0) or 1.0)
                            _cv = float(_cr['_PCR_ChgOI'])
                            _gv = float(_cr['_Net_GEX'])
                            _ps = 1 if _pv > 1.2 else (-1 if _pv < 0.7 else 0)
                            _cs = 1 if _cv > 1.2 else (-1 if _cv < 0.7 else 0)
                            _ds = _ps + _cs
                            _cw = _comp_weights[_ci] if _ci < len(_comp_weights) else 1.0
                            _comp_strike_scores.append(_ds * _cw)
                            _slbl2 = str(int(_cr['Strike']))
                            _comp_per_strike_scores[_slbl2] = round(_ds * _cw, 2)
                            _clbl = _comp_position_labels[_ci] if _ci < len(_comp_position_labels) else f"S{_ci}"
                            _comp_strike_details.append({
                                'Position': _clbl, 'Strike': int(_cr['Strike']),
                                'PCR (OI)': round(_pv, 2),
                                'PCR Signal': 'Bull' if _ps > 0 else ('Bear' if _ps < 0 else 'Neut'),
                                'PCR (ΔOI)': round(_cv, 2),
                                'ΔOI Signal': 'Bull' if _cs > 0 else ('Bear' if _cs < 0 else 'Neut'),
                                'Net GEX': round(_gv, 2),
                                'GEX Signal': 'Pin' if _gv > 10 else ('Accel' if _gv < -10 else 'Neut'),
                                'Score': round(_ds * _cw, 1)
                            })
                        _comp_total_score = sum(_comp_strike_scores)
                        _comp_score_pct   = (_comp_total_score / _comp_max_possible) * 100
                        if abs(_comp_score_pct) < 15:
                            if _comp_gex_pinning:
                                _verdict, _verdict_icon, _verdict_color = "SIDEWAYS", "↔️", "#FFD700"
                                _verdict_desc = "Mixed signals + Positive GEX = Range-bound / Choppy"
                                _verdict_numeric = 0
                            else:
                                _verdict, _verdict_icon, _verdict_color = "NEUTRAL", "⚪", "#888888"
                                _verdict_desc = "No clear directional bias from ATM±2 strikes"
                                _verdict_numeric = 0
                        elif _comp_score_pct > 0:
                            if _comp_gex_trending:
                                _verdict, _verdict_icon, _verdict_color = "STRONG UP", "🟢🔥", "#00ff88"
                                _verdict_desc = "Bullish PCR + Fresh put writing + Negative GEX = Breakout UP"
                                _verdict_numeric = 3
                            elif _comp_gex_pinning:
                                _verdict, _verdict_icon, _verdict_color = "UP (CAPPED)", "🟢📍", "#90EE90"
                                _verdict_desc = "Bullish bias but positive GEX may cap upside momentum"
                                _verdict_numeric = 1
                            else:
                                _verdict, _verdict_icon, _verdict_color = "UP", "🟢", "#00ff88"
                                _verdict_desc = "Bullish PCR + Put writing activity across ATM±2 strikes"
                                _verdict_numeric = 2
                        else:
                            if _comp_gex_trending:
                                _verdict, _verdict_icon, _verdict_color = "STRONG DOWN", "🔴🔥", "#ff4444"
                                _verdict_desc = "Bearish PCR + Fresh call writing + Negative GEX = Breakdown"
                                _verdict_numeric = -3
                            elif _comp_gex_pinning:
                                _verdict, _verdict_icon, _verdict_color = "DOWN (SUPPORTED)", "🔴📍", "#FFB6C1"
                                _verdict_desc = "Bearish PCR + Positive GEX may provide support"
                                _verdict_numeric = -1
                            else:
                                _verdict, _verdict_icon, _verdict_color = "DOWN", "🔴", "#ff4444"
                                _verdict_desc = "Bearish PCR + Call writing activity across ATM±2 strikes"
                                _verdict_numeric = -2
                        _comp_data_available = True
                        st.session_state.composite_signal_last_valid = {
                            'verdict': _verdict, 'verdict_icon': _verdict_icon,
                            'verdict_color': _verdict_color, 'verdict_desc': _verdict_desc,
                            'total_score': _comp_total_score, 'score_pct': _comp_score_pct,
                            'total_net_gex': _comp_total_gex, 'avg_pcr': _comp_avg_pcr,
                            'avg_chgoi': _comp_avg_chgoi, 'strike_details': _comp_strike_details,
                            'per_strike_scores': _comp_per_strike_scores
                        }
                        _comp_should_add = (
                            not st.session_state.composite_signal_history or
                            (_pro_now - st.session_state.composite_signal_history[-1]['time']).total_seconds() >= 30
                        )
                        if _comp_should_add:
                            _comp_hist_entry = {
                                'time': _pro_now, 'score': round(_comp_total_score, 2),
                                'score_pct': round(_comp_score_pct, 1), 'verdict': _verdict,
                                'verdict_numeric': _verdict_numeric,
                                'avg_pcr': round(_comp_avg_pcr, 3),
                                'avg_chgoi': round(_comp_avg_chgoi, 3),
                                'total_gex': round(_comp_total_gex, 2),
                            }
                            for _ck, _cv2 in _comp_per_strike_scores.items():
                                _comp_hist_entry[f'score_{_ck}'] = _cv2
                            st.session_state.composite_signal_history.append(_comp_hist_entry)
                            if len(st.session_state.composite_signal_history) > 200:
                                st.session_state.composite_signal_history = st.session_state.composite_signal_history[-200:]
                    # Fall back to cached composite verdict
                    if not _comp_data_available:
                        _lv = st.session_state.composite_signal_last_valid
                        if _lv:
                            _verdict, _verdict_icon = _lv['verdict'], _lv['verdict_icon']
                            _verdict_color = _lv['verdict_color']
                            _verdict_desc  = _lv['verdict_desc']
                            _comp_total_score = _lv['total_score']
                            _comp_score_pct   = _lv['score_pct']
                            _comp_total_gex   = _lv['total_net_gex']
                            _comp_avg_pcr     = _lv['avg_pcr']
                            _comp_avg_chgoi   = _lv['avg_chgoi']
                            _comp_strike_details = _lv.get('strike_details', [])
                            _comp_gex_trending   = _comp_total_gex < -10
                            _comp_gex_pinning    = _comp_total_gex > 10

                    # ── 7b. Weighted PCR per strike (ATM±2) ─────────────────────
                    _wts_map = {'ATM-2': 1.0, 'ATM-1': 1.5, 'ATM': 2.0, 'ATM+1': 1.5, 'ATM+2': 1.0}
                    _wpcr_oi = _wpcr_chgoi = _wpcr_vol = _wtotal = 0.0
                    for _, _wr in _pro_slice.iterrows():
                        _wlbl = _pro_lbl(float(_wr['Strike']))
                        _ww = _wts_map.get(_wlbl, 1.0)
                        _wtotal += _ww
                        _wce_oi = float(_wr.get('openInterest_CE', 0) or 0)
                        _wpe_oi = float(_wr.get('openInterest_PE', 0) or 0)
                        _wce_ch = abs(float(_wr.get('changeinOpenInterest_CE', 0) or 0))
                        _wpe_ch = abs(float(_wr.get('changeinOpenInterest_PE', 0) or 0))
                        _wce_vl = float(_wr.get('totalTradedVolume_CE', 0) or 0)
                        _wpe_vl = float(_wr.get('totalTradedVolume_PE', 0) or 0)
                        _wpcr_oi    += (_wpe_oi / (_wce_oi + 1e-6)) * _ww
                        _wpcr_chgoi += (_wpe_ch / (_wce_ch + 1e-6)) * _ww
                        _wpcr_vol   += (_wpe_vl / (_wce_vl + 1e-6)) * _ww
                    if _wtotal > 0:
                        _wpcr_oi /= _wtotal
                        _wpcr_chgoi /= _wtotal
                        _wpcr_vol /= _wtotal

                    # ── 7c. Unified Sentiment Score (0–100) ─────────────────────
                    _pro_h_tmp = st.session_state.pro_trader_history
                    _pro_h_df_tmp = pd.DataFrame(_pro_h_tmp) if len(_pro_h_tmp) >= 2 else None
                    # PCR OI score (±20)
                    _ss_pcr = (20 if _wpcr_oi > 1.2 else
                               -20 if _wpcr_oi < 0.7 else
                               int((_wpcr_oi - 0.95) / 0.25 * 20))
                    # ΔOI trend score (±20)
                    _ss_chgoi = 0
                    if _pro_h_df_tmp is not None and 'pcr_chgoi' in _pro_h_df_tmp.columns and len(_pro_h_df_tmp) >= 3:
                        _chgoi_tr = _pro_h_df_tmp['pcr_chgoi'].iloc[-1] - _pro_h_df_tmp['pcr_chgoi'].iloc[-3]
                        _ss_chgoi = (20 if _chgoi_tr > 0.05 else
                                     -20 if _chgoi_tr < -0.05 else int(_chgoi_tr * 200))
                    elif _wpcr_chgoi > 1.2:
                        _ss_chgoi = 20
                    elif _wpcr_chgoi < 0.7:
                        _ss_chgoi = -20
                    # GEX conviction score (±20) — amplifies PCR direction
                    _gex_mag = abs(_pro_net_gex)
                    _ss_gex_raw = (20 if _gex_mag > 50 else 14 if _gex_mag > 20 else
                                   10 if _gex_mag > 10 else 5 if _gex_mag > 5 else 0)
                    _ss_gex_dir = 1 if _ss_pcr >= 0 else -1
                    if _pro_net_gex < -10:
                        _ss_gex = _ss_gex_raw * _ss_gex_dir
                    elif _pro_net_gex < 0:
                        _ss_gex = int(_ss_gex_raw * 0.5) * _ss_gex_dir
                    elif _pro_net_gex > 10:
                        _ss_gex = -5
                    else:
                        _ss_gex = 0
                    # Straddle momentum score (±20)
                    _ss_straddle = 0
                    if _pro_h_df_tmp is not None and 'straddle_atm' in _pro_h_df_tmp.columns and len(_pro_h_df_tmp) >= 3:
                        _st_roc2 = (_pro_h_df_tmp['straddle_atm'].iloc[-1] -
                                    _pro_h_df_tmp['straddle_atm'].iloc[-3]) / (
                            abs(_pro_h_df_tmp['straddle_atm'].iloc[-3]) + 1e-6) * 100
                        _ss_straddle = (20 if _st_roc2 > 2 else 10 if _st_roc2 > 0.5 else
                                        -10 if _st_roc2 < -2 else 0)
                    # Pressure imbalance score (±20)
                    _pres_diff = _pro_call_pres - _pro_put_pres
                    _ss_pressure = (20 if _pres_diff > 0.10 else
                                    -20 if _pres_diff < -0.10 else int(_pres_diff * 100))
                    # Combine: raw range −20 to +20, map to 0–100
                    _sentiment_raw = (0.25 * _ss_pcr + 0.20 * _ss_chgoi + 0.20 * _ss_gex +
                                      0.20 * _ss_straddle + 0.15 * _ss_pressure)
                    _sentiment_score = int(max(0, min(100, (_sentiment_raw + 20) * 2.5)))

                    if _sentiment_score >= 80:
                        _sent_verdict = "STRONG BULLISH TREND"; _sent_icon = "🚀"; _sent_color = "#00ff88"
                    elif _sentiment_score >= 60:
                        _sent_verdict = "BULLISH BIAS"; _sent_icon = "🟢"; _sent_color = "#00C853"
                    elif _sentiment_score >= 40:
                        _sent_verdict = "SIDEWAYS MARKET"; _sent_icon = "↔️"; _sent_color = "#FFD700"
                    elif _sentiment_score >= 20:
                        _sent_verdict = "BEARISH BIAS"; _sent_icon = "🔴"; _sent_color = "#FF5252"
                    else:
                        _sent_verdict = "STRONG BEARISH TREND"; _sent_icon = "🔥"; _sent_color = "#FF1744"

                    _market_bias = ("BULLISH" if _sentiment_score > 65 else
                                    "BEARISH" if _sentiment_score < 35 else "SIDEWAYS")

                    # ── 7d. Final Institutional Signal ──────────────────────────
                    if _verdict == "STRONG UP" and _sentiment_score > 70:
                        _final_signal = "🔥 INSTITUTIONAL BULLISH FLOW"
                        _final_color  = "#00ff88"
                        _final_desc   = "Options flow + PCR × GEX aligned — Smart money buying"
                    elif _verdict == "STRONG DOWN" and _sentiment_score < 30:
                        _final_signal = "⚡ INSTITUTIONAL BEARISH FLOW"
                        _final_color  = "#FF5252"
                        _final_desc   = "Options flow + PCR × GEX aligned — Smart money selling"
                    elif _verdict in ("STRONG UP", "UP") and _sentiment_score < 40:
                        _final_signal = "⚠️ FLOW DIVERGENCE — BULL TRAP POSSIBLE"
                        _final_color  = "#FF9800"
                        _final_desc   = "Composite bullish but sentiment weak — caution"
                    elif _verdict in ("STRONG DOWN", "DOWN") and _sentiment_score > 60:
                        _final_signal = "⚠️ FLOW DIVERGENCE — BEAR TRAP POSSIBLE"
                        _final_color  = "#FF9800"
                        _final_desc   = "Composite bearish but sentiment strong — caution"
                    elif _sentiment_score >= 60:
                        _final_signal = f"{_sent_icon} {_sent_verdict}"
                        _final_color  = _sent_color
                        _final_desc   = f"Sentiment Score: {_sentiment_score}/100"
                    elif _sentiment_score <= 40:
                        _final_signal = f"{_sent_icon} {_sent_verdict}"
                        _final_color  = _sent_color
                        _final_desc   = f"Sentiment Score: {_sentiment_score}/100"
                    elif _verdict:
                        _final_signal = f"{_verdict_icon} {_verdict}"
                        _final_color  = _verdict_color
                        _final_desc   = _verdict_desc or ""
                    else:
                        _final_signal = "⚪ NEUTRAL"
                        _final_color  = "#888888"
                        _final_desc   = "Insufficient data for signal"

                    # ── Store sentiment history ──────────────────────────────────
                    _sent_should_add = (
                        not st.session_state.sentiment_history or
                        (_pro_now - st.session_state.sentiment_history[-1]['time']).total_seconds() >= 28
                    )
                    if _sent_should_add:
                        st.session_state.sentiment_history.append({
                            'time': _pro_now,
                            'sentiment_score': _sentiment_score,
                            'comp_score_pct': round(_comp_score_pct, 1),
                            'total_gex': round(_comp_total_gex, 2),
                            'verdict': _final_signal,
                        })
                        if len(st.session_state.sentiment_history) > 200:
                            st.session_state.sentiment_history = st.session_state.sentiment_history[-200:]

                    # ── Telegram alerts at 70/30 threshold crossings ─────────────
                    if enable_signals and len(st.session_state.sentiment_history) >= 2:
                        _prev_sent = st.session_state.sentiment_history[-2]['sentiment_score']
                        _sent_alert_msg = None
                        if _prev_sent < 70 <= _sentiment_score:
                            _sent_alert_msg = (f"🚀 <b>BULLISH FLOW INCREASING</b>\n"
                                               f"Spot: ₹{_pro_spot:.0f}\n"
                                               f"Sentiment Score: {_sentiment_score}/100\n"
                                               f"Breakout probability high")
                        elif _prev_sent > 30 >= _sentiment_score:
                            _sent_alert_msg = (f"🔥 <b>BEARISH FLOW INCREASING</b>\n"
                                               f"Spot: ₹{_pro_spot:.0f}\n"
                                               f"Sentiment Score: {_sentiment_score}/100\n"
                                               f"Breakdown probability high")
                        if _sent_alert_msg:
                            send_telegram_message_sync(_sent_alert_msg)

                    # ── 7. Breakout Probability Score (0–100) ──────────────────
                    _pro_h    = st.session_state.pro_trader_history
                    _pro_h_df = pd.DataFrame(_pro_h) if len(_pro_h) >= 2 else None

                    def _rate_of_change(series, window=5):
                        if series is None or len(series) < 2: return 0.0
                        s = series.tail(window)
                        return (s.iloc[-1] - s.iloc[0]) / (abs(s.iloc[0]) + 1e-6) * 100

                    # Straddle momentum score (0–20)
                    _s1 = 10
                    if _pro_h_df is not None and 'straddle_atm' in _pro_h_df.columns:
                        _roc = _rate_of_change(_pro_h_df['straddle_atm'])
                        _s1 = int(min(20, max(0, _roc * 2 + 10)))

                    # IV expansion score (0–20)
                    _s2 = 10
                    if _pro_h_df is not None and 'avg_iv_ce' in _pro_h_df.columns:
                        _roc_ce = _rate_of_change(_pro_h_df['avg_iv_ce'])
                        _roc_pe = _rate_of_change(_pro_h_df['avg_iv_pe'])
                        _s2 = int(min(20, max(0, (_roc_ce + _roc_pe) + 10)))

                    # Gamma shift score — negative GEX → trending (0–20)
                    _s3 = (20 if _pro_net_gex < -50 else 16 if _pro_net_gex < -20 else
                           12 if _pro_net_gex < -10 else 10 if _pro_net_gex < 0 else
                           8  if _pro_net_gex < 10  else 6  if _pro_net_gex < 20  else 4)

                    # Volume spike score (0–20)
                    _s4 = 10
                    if _pro_h_df is not None and 'pcr_vol' in _pro_h_df.columns and len(_pro_h_df) >= 3:
                        _pv = _pro_h_df['pcr_vol']
                        _dev = abs(_pv.iloc[-1] - _pv.mean()) / (_pv.std() + 1e-6)
                        _s4 = int(min(20, max(0, _dev * 5 + 8)))

                    # Pressure imbalance score (0–20)
                    _s5 = int(min(20, max(0, abs(_pro_call_pres - _pro_put_pres) * 80 + 5)))

                    _pro_score = min(100, _s1 + _s2 + _s3 + _s4 + _s5)
                    if st.session_state.pro_trader_history:
                        st.session_state.pro_trader_history[-1]['breakout_score'] = _pro_score

                    _pro_mode_color = "#FF5252" if _pro_net_gex < 0 else "#00C853"
                    _pro_market_mode = "⚡ TRENDING" if _pro_net_gex < 0 else "📌 RANGE"
                    _bs_color = ("#FF5252" if _pro_score >= 70 else
                                 "#FFD740" if _pro_score >= 40 else "#00BCD4")
                    _bs_label = ("🔥 High Prob Breakout" if _pro_score >= 70 else
                                 "⚡ Possible Move"       if _pro_score >= 40 else "📌 Range")

                    # ── 8. Smart Signal Engine ─────────────────────────────────
                    _pro_alert = None
                    _pro_alert_color = "#888"
                    if _pro_h_df is not None and len(_pro_h_df) >= 3:
                        _st_rising  = _pro_h_df['straddle_atm'].iloc[-1] > _pro_h_df['straddle_atm'].iloc[-3]
                        _gex_dec    = _pro_h_df['net_gex'].iloc[-1] < _pro_h_df['net_gex'].iloc[-3]
                        _nd_rising  = _pro_h_df['net_delta'].iloc[-1] > _pro_h_df['net_delta'].iloc[-3]
                        if _st_rising and _pro_call_pres > _pro_put_pres and _gex_dec and _nd_rising:
                            _pro_alert = "🚀 BULLISH BREAKOUT PROBABLE"
                            _pro_alert_color = "#00C853"
                        elif _st_rising and _pro_put_pres > _pro_call_pres and _gex_dec and not _nd_rising:
                            _pro_alert = "🔥 BEARISH BREAKDOWN PROBABLE"
                            _pro_alert_color = "#FF5252"
                        if _pro_alert and enable_signals:
                            _last_sig = st.session_state.pro_smart_signal_last
                            if (_last_sig is None or
                                    (_pro_now - _last_sig[0]).total_seconds() > 300 or
                                    _last_sig[1] != _pro_alert):
                                send_telegram_message_sync(
                                    f"<b>VOB2 Pro Alert</b>\n{_pro_alert}\n"
                                    f"Spot: {_pro_spot:.0f} | ATM: ₹{int(_pro_atm_val)}\n"
                                    f"Straddle: {_pro_atm_straddle:.0f} | Score: {_pro_score}/100\n"
                                    f"GEX: {_pro_net_gex:+.1f}L | Δ: {_pro_net_delta:+.0f}"
                                )
                                st.session_state.pro_smart_signal_last = (_pro_now, _pro_alert)

                    # ── Supabase write ─────────────────────────────────────────
                    if _pro_should_add:
                        try:
                            db.client.table('pro_trader_metrics').upsert({
                                'timestamp':      _pro_now.isoformat(),
                                'spot':           float(_pro_spot),
                                'atm_strike':     int(_pro_atm_val),
                                'straddle_atm':   float(_pro_atm_straddle),
                                'iv_skew':        float(_pro_market_iv_sk),
                                'pcr_oi':         float(_pro_pcr_oi),
                                'pcr_vol':        float(_pro_pcr_vol),
                                'pcr_chgoi':      float(_pro_pcr_chg),
                                'call_pressure':  float(_pro_call_pres),
                                'put_pressure':   float(_pro_put_pres),
                                'net_delta':      float(_pro_net_delta),
                                'net_gex':        float(_pro_net_gex),
                                'breakout_score': int(_pro_score),
                            }, on_conflict='timestamp').execute()
                        except Exception:
                            pass  # Supabase table may not exist yet; fail silently

                    # ══════════════════════════════════════════════════════════
                    # UNIFIED SENTIMENT TOP PANEL
                    # ══════════════════════════════════════════════════════════
                    _prev_atm_st = (_pro_h_df['straddle_atm'].iloc[-2]
                                    if _pro_h_df is not None and len(_pro_h_df) >= 2
                                    else _pro_atm_straddle)
                    _st_chg = _pro_atm_straddle - _prev_atm_st
                    _st_clr = "#FF5252" if _st_chg > 5 else ("#00C853" if _st_chg < -5 else "#FFD740")

                    # Row 1: Large FinalSignal verdict (left) + 4 metric cards (right)
                    _uf1, _uf2 = st.columns([3, 2])
                    with _uf1:
                        _vc_txt = _verdict_color if _verdict_color else "#888888"
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,{_final_color}10,{_final_color}25);
                                    padding:20px;border-radius:12px;border:3px solid {_final_color};
                                    text-align:center;min-height:130px;">
                            <div style="color:#888;font-size:10px;letter-spacing:2px;">
                                UNIFIED OPTIONS FLOW SENTIMENT</div>
                            <div style="color:{_final_color};font-size:30px;font-weight:bold;margin:6px 0;
                                        line-height:1.2;">{_final_signal}</div>
                            <div style="color:#ccc;font-size:12px;">{_final_desc}</div>
                            <div style="color:{_sent_color};font-size:11px;margin-top:6px;">
                                Sentiment <b>{_sentiment_score}/100</b> — {_sent_verdict} &nbsp;|&nbsp;
                                Direction <b>{_verdict_icon if _verdict_icon else "⚪"} {_verdict if _verdict else "NEUTRAL"}</b>
                                ({_comp_score_pct:+.0f}%)
                            </div>
                        </div>""", unsafe_allow_html=True)
                    with _uf2:
                        _um1, _um2 = st.columns(2)
                        with _um1:
                            st.markdown(f"""
                            <div style="background:#1e1e1e;padding:10px;border-radius:8px;border:1px solid #444;margin-bottom:5px;">
                            <div style="color:#aaa;font-size:10px;">SPOT / ATM</div>
                            <div style="font-size:18px;font-weight:bold;color:#FFD740;">{_pro_spot:,.0f}</div>
                            <div style="font-size:10px;color:#888;">ATM: ₹{int(_pro_atm_val)}</div>
                            </div>""", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background:#1e1e1e;padding:10px;border-radius:8px;border:2px solid {_pro_mode_color};">
                            <div style="color:#aaa;font-size:10px;">MARKET MODE</div>
                            <div style="font-size:14px;font-weight:bold;color:{_pro_mode_color};">{_pro_market_mode}</div>
                            <div style="font-size:10px;color:#888;">GEX {_pro_net_gex:+.1f}L</div>
                            </div>""", unsafe_allow_html=True)
                        with _um2:
                            st.markdown(f"""
                            <div style="background:#1e1e1e;padding:10px;border-radius:8px;border:2px solid {_bs_color};margin-bottom:5px;">
                            <div style="color:#aaa;font-size:10px;">BREAKOUT SCORE</div>
                            <div style="font-size:18px;font-weight:bold;color:{_bs_color};">{_pro_score}/100</div>
                            <div style="font-size:10px;color:{_bs_color};">{_bs_label}</div>
                            </div>""", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background:#1e1e1e;padding:10px;border-radius:8px;border:1px solid {_st_clr};">
                            <div style="color:#aaa;font-size:10px;">ATM STRADDLE</div>
                            <div style="font-size:18px;font-weight:bold;color:{_st_clr};">₹{_pro_atm_straddle:.0f}</div>
                            <div style="font-size:10px;color:#888;">Chg {_st_chg:+.1f}</div>
                            </div>""", unsafe_allow_html=True)

                    # Row 2: 4 composite metric cards
                    _cm1, _cm2, _cm3, _cm4 = st.columns(4)
                    with _cm1:
                        _pclr = "#00ff88" if _comp_avg_pcr > 1.2 else "#ff4444" if _comp_avg_pcr < 0.7 else "#FFD700"
                        st.markdown(f"""
                        <div style="background:{_pclr}12;padding:9px;border-radius:7px;border:1px solid {_pclr};
                                    text-align:center;margin-top:5px;">
                        <div style="color:{_pclr};font-size:10px;font-weight:bold;">AVG PCR (OI)</div>
                        <div style="color:{_pclr};font-size:18px;font-weight:bold;">{_comp_avg_pcr:.2f}</div>
                        <div style="color:#ccc;font-size:10px;">{'Bull' if _comp_avg_pcr > 1.2 else 'Bear' if _comp_avg_pcr < 0.7 else 'Neut'}</div>
                        </div>""", unsafe_allow_html=True)
                    with _cm2:
                        _cclr = "#00ff88" if _comp_avg_chgoi > 1.2 else "#ff4444" if _comp_avg_chgoi < 0.7 else "#FFD700"
                        st.markdown(f"""
                        <div style="background:{_cclr}12;padding:9px;border-radius:7px;border:1px solid {_cclr};
                                    text-align:center;margin-top:5px;">
                        <div style="color:{_cclr};font-size:10px;font-weight:bold;">AVG PCR (ΔOI)</div>
                        <div style="color:{_cclr};font-size:18px;font-weight:bold;">{_comp_avg_chgoi:.2f}</div>
                        <div style="color:#ccc;font-size:10px;">{'Bull' if _comp_avg_chgoi > 1.2 else 'Bear' if _comp_avg_chgoi < 0.7 else 'Neut'}</div>
                        </div>""", unsafe_allow_html=True)
                    with _cm3:
                        _gclr = "#00ff88" if _comp_total_gex > 10 else "#ff4444" if _comp_total_gex < -10 else "#FFD700"
                        st.markdown(f"""
                        <div style="background:{_gclr}12;padding:9px;border-radius:7px;border:1px solid {_gclr};
                                    text-align:center;margin-top:5px;">
                        <div style="color:{_gclr};font-size:10px;font-weight:bold;">TOTAL GEX</div>
                        <div style="color:{_gclr};font-size:18px;font-weight:bold;">{_comp_total_gex:.1f}L</div>
                        <div style="color:#ccc;font-size:10px;">{'Pin' if _comp_total_gex > 10 else 'Accel' if _comp_total_gex < -10 else 'Neut'}</div>
                        </div>""", unsafe_allow_html=True)
                    with _cm4:
                        st.markdown(f"""
                        <div style="background:{_sent_color}12;padding:9px;border-radius:7px;border:1px solid {_sent_color};
                                    text-align:center;margin-top:5px;">
                        <div style="color:{_sent_color};font-size:10px;font-weight:bold;">SENTIMENT</div>
                        <div style="color:{_sent_color};font-size:18px;font-weight:bold;">{_sentiment_score}/100</div>
                        <div style="color:#ccc;font-size:10px;">{_market_bias}</div>
                        </div>""", unsafe_allow_html=True)

                    # Smart signal + divergence alert banner
                    _alert_txt = _pro_alert or ""
                    _alert_clr = _pro_alert_color
                    if not _pro_alert and "DIVERGENCE" in _final_signal:
                        _alert_txt = _final_signal
                        _alert_clr = _final_color
                    if _alert_txt:
                        st.markdown(f"""
                        <div style="background:{_alert_clr}22;border:2px solid {_alert_clr};
                             border-radius:8px;padding:10px;margin:8px 0;text-align:center;">
                        <span style="font-size:15px;font-weight:bold;color:{_alert_clr};">{_alert_txt}</span>
                        <span style="font-size:12px;color:#ccc;margin-left:12px;">
                            Score: {_pro_score}/100 · Sentiment: {_sentiment_score}/100</span>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    if _pro_h_df is not None and len(_pro_h_df) >= 2:
                        # ══════════════════════════════════════════════════════
                        # MIDDLE CHARTS  (2 × 2)
                        # ══════════════════════════════════════════════════════
                        _mc1, _mc2 = st.columns(2)
                        _pos_keys   = ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']
                        _pos_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']

                        with _mc1:
                            # Straddle Engine Chart
                            _fig_st = go.Figure()
                            for _si, _sk in enumerate(_pos_keys):
                                _col = f'straddle_{_sk}'
                                if _col in _pro_h_df.columns:
                                    _cv = _pro_h_df[_col].iloc[-1]
                                    _fig_st.add_trace(go.Scatter(
                                        x=_pro_h_df['time'], y=_pro_h_df[_col],
                                        mode='lines', name=_sk,
                                        line=dict(color=_pos_colors[_si],
                                                  width=2.5 if _sk == 'ATM' else 1.5)
                                    ))
                                    _fig_st.add_trace(go.Scatter(
                                        x=[_pro_h_df['time'].iloc[-1]], y=[_cv],
                                        mode='markers+text', text=[f'{_cv:.0f}'],
                                        textposition='top right',
                                        textfont=dict(size=8, color=_pos_colors[_si]),
                                        marker=dict(size=6, color=_pos_colors[_si]),
                                        showlegend=False, hoverinfo='skip'
                                    ))
                            _fig_st.update_layout(
                                title=f'Straddle Engine — ATM: ₹{_pro_atm_straddle:.0f}',
                                height=280, template='plotly_dark',
                                margin=dict(l=40, r=20, t=45, b=30),
                                xaxis=dict(tickformat='%H:%M', gridcolor='#333'),
                                yaxis=dict(gridcolor='#333', title='₹ Straddle'),
                                showlegend=True,
                                legend=dict(orientation='h', y=-0.35, font=dict(size=9)),
                                paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='#ccc'),
                            )
                            st.plotly_chart(_fig_st, use_container_width=True)

                        with _mc2:
                            # IV Skew Engine Chart
                            _fig_iv2 = go.Figure()
                            for _col_k, _col_n, _col_c, _tp in [
                                ('avg_iv_ce', 'Call IV avg', '#00C853', 'top right'),
                                ('avg_iv_pe', 'Put IV avg',  '#FF5252', 'bottom right'),
                                ('iv_skew',   'IV Skew (PE−CE)', '#FFD740', 'top left'),
                            ]:
                                if _col_k in _pro_h_df.columns:
                                    _cv = _pro_h_df[_col_k].iloc[-1]
                                    _fig_iv2.add_trace(go.Scatter(
                                        x=_pro_h_df['time'], y=_pro_h_df[_col_k],
                                        mode='lines', name=_col_n,
                                        line=dict(color=_col_c, width=2,
                                                  dash='dot' if _col_k == 'iv_skew' else 'solid')
                                    ))
                                    _fig_iv2.add_trace(go.Scatter(
                                        x=[_pro_h_df['time'].iloc[-1]], y=[_cv],
                                        mode='markers+text',
                                        text=[f'{_cv:+.1f}' if _col_k == 'iv_skew' else f'{_cv:.1f}%'],
                                        textposition=_tp, textfont=dict(size=8, color=_col_c),
                                        marker=dict(size=7, color=_col_c),
                                        showlegend=False, hoverinfo='skip'
                                    ))
                            _fig_iv2.add_hline(y=0, line_color='#555', line_width=1)
                            _fig_iv2.update_layout(
                                title=f'IV Skew Engine — PE−CE: {_pro_market_iv_sk:+.1f}',
                                height=280, template='plotly_dark',
                                margin=dict(l=40, r=20, t=45, b=30),
                                xaxis=dict(tickformat='%H:%M', gridcolor='#333'),
                                yaxis=dict(gridcolor='#333', title='IV %'),
                                showlegend=True,
                                legend=dict(orientation='h', y=-0.35, font=dict(size=9)),
                                paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='#ccc'),
                            )
                            st.plotly_chart(_fig_iv2, use_container_width=True)

                        _mc3, _mc4 = st.columns(2)

                        with _mc3:
                            # PCR Trend Chart
                            _fig_pcr2 = go.Figure()
                            for _pk, _pn, _pc, _pd in [
                                ('pcr_oi',    'PCR OI',     '#00ccff', 'solid'),
                                ('pcr_vol',   'PCR Volume', '#ffaa00', 'dash'),
                                ('pcr_chgoi', 'PCR ΔOI',   '#ff44ff', 'dot'),
                            ]:
                                if _pk in _pro_h_df.columns:
                                    _cv = _pro_h_df[_pk].iloc[-1]
                                    _fig_pcr2.add_trace(go.Scatter(
                                        x=_pro_h_df['time'], y=_pro_h_df[_pk],
                                        mode='lines', name=_pn,
                                        line=dict(color=_pc, width=2, dash=_pd)
                                    ))
                                    _fig_pcr2.add_trace(go.Scatter(
                                        x=[_pro_h_df['time'].iloc[-1]], y=[_cv],
                                        mode='markers+text', text=[f'{_cv:.2f}'],
                                        textposition='top right',
                                        textfont=dict(size=8, color=_pc),
                                        marker=dict(size=7, color=_pc),
                                        showlegend=False, hoverinfo='skip'
                                    ))
                            _fig_pcr2.add_hline(y=1.2, line_dash='dot',
                                                line_color='rgba(0,255,136,0.5)',
                                                annotation_text='Bull 1.2',
                                                annotation_position='right',
                                                annotation_font_size=8)
                            _fig_pcr2.add_hline(y=0.7, line_dash='dot',
                                                line_color='rgba(255,68,68,0.5)',
                                                annotation_text='Bear 0.7',
                                                annotation_position='right',
                                                annotation_font_size=8)
                            _fig_pcr2.add_hline(y=1.0, line_color='rgba(255,255,255,0.2)', line_width=1)
                            _fig_pcr2.update_layout(
                                title=f'PCR Engine — OI:{_pro_pcr_oi:.2f} Vol:{_pro_pcr_vol:.2f} ΔOI:{_pro_pcr_chg:.2f}',
                                height=280, template='plotly_dark',
                                margin=dict(l=40, r=40, t=45, b=30),
                                xaxis=dict(tickformat='%H:%M', gridcolor='#333'),
                                yaxis=dict(gridcolor='#333', title='PCR'),
                                showlegend=True,
                                legend=dict(orientation='h', y=-0.35, font=dict(size=9)),
                                paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='#ccc'),
                            )
                            st.plotly_chart(_fig_pcr2, use_container_width=True)

                        with _mc4:
                            # Gamma Pressure Chart
                            _fig_gam = go.Figure()
                            if 'net_gex' in _pro_h_df.columns:
                                _gv = _pro_h_df['net_gex']
                                _g_cur = _gv.iloc[-1]
                                _fig_gam.add_trace(go.Scatter(
                                    x=_pro_h_df['time'], y=_gv,
                                    mode='lines+markers', name='Net GEX (L)',
                                    line=dict(color='#FFD740', width=2),
                                    marker=dict(size=4,
                                                color=['#00C853' if v >= 0 else '#FF5252' for v in _gv]),
                                    fill='tozeroy', fillcolor='rgba(255,215,0,0.08)'
                                ))
                                _fig_gam.add_hline(y=0, line_color='white', line_width=1.5)
                                _fig_gam.add_trace(go.Scatter(
                                    x=[_pro_h_df['time'].iloc[-1]], y=[_g_cur],
                                    mode='markers+text', text=[f'{_g_cur:+.1f}L'],
                                    textposition='top right',
                                    textfont=dict(size=9, color='#FFD740'),
                                    marker=dict(size=9, color='#FFD740', symbol='circle'),
                                    showlegend=False, hoverinfo='skip'
                                ))
                            _fig_gam.update_layout(
                                title=f'Gamma Pressure — {_pro_market_mode} (GEX: {_pro_net_gex:+.1f}L)',
                                height=280, template='plotly_dark',
                                margin=dict(l=40, r=20, t=45, b=30),
                                xaxis=dict(tickformat='%H:%M', gridcolor='#333'),
                                yaxis=dict(gridcolor='#333', title='GEX (L)',
                                           zeroline=True, zerolinecolor='white'),
                                showlegend=False,
                                paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='#ccc'),
                            )
                            st.plotly_chart(_fig_gam, use_container_width=True)

                        # ══════════════════════════════════════════════════════
                        # BOTTOM CHARTS  (3 columns)
                        # ══════════════════════════════════════════════════════
                        _bc1, _bc2, _bc3 = st.columns(3)

                        with _bc1:
                            # Bid/Ask Pressure Chart
                            _fig_pres2 = go.Figure()
                            for _pk2, _pn2, _pc2, _tp2 in [
                                ('call_pressure', 'Call Pressure', '#00C853', 'top right'),
                                ('put_pressure',  'Put Pressure',  '#FF5252', 'bottom right'),
                            ]:
                                if _pk2 in _pro_h_df.columns:
                                    _cv2 = _pro_h_df[_pk2].iloc[-1]
                                    _fig_pres2.add_trace(go.Scatter(
                                        x=_pro_h_df['time'], y=_pro_h_df[_pk2],
                                        mode='lines', name=_pn2,
                                        line=dict(color=_pc2, width=2)
                                    ))
                                    _fig_pres2.add_trace(go.Scatter(
                                        x=[_pro_h_df['time'].iloc[-1]], y=[_cv2],
                                        mode='markers+text', text=[f'{_cv2:.3f}'],
                                        textposition=_tp2,
                                        textfont=dict(size=8, color=_pc2),
                                        marker=dict(size=7, color=_pc2),
                                        showlegend=False, hoverinfo='skip'
                                    ))
                            _fig_pres2.add_hline(y=0.5, line_color='#555', line_width=1)
                            _fig_pres2.update_layout(
                                title=f'Bid/Ask Pressure — C:{_pro_call_pres:.3f} P:{_pro_put_pres:.3f}',
                                height=250, template='plotly_dark',
                                margin=dict(l=30, r=20, t=45, b=30),
                                xaxis=dict(tickformat='%H:%M', gridcolor='#333'),
                                yaxis=dict(gridcolor='#333', range=[0, 1], title='Pressure'),
                                showlegend=True,
                                legend=dict(orientation='h', y=-0.4, font=dict(size=9)),
                                paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='#ccc'),
                            )
                            st.plotly_chart(_fig_pres2, use_container_width=True)
                            _p_sig = ("🚀 Call Dominating" if _pro_call_pres > _pro_put_pres + 0.1
                                      else "🔥 Put Dominating" if _pro_put_pres > _pro_call_pres + 0.1
                                      else "⚖️ Balanced")
                            st.caption(_p_sig)

                        with _bc2:
                            # Net Delta Shift Chart
                            _fig_nd = go.Figure()
                            if 'net_delta' in _pro_h_df.columns:
                                _nd_vals = _pro_h_df['net_delta']
                                _nd_cur  = _nd_vals.iloc[-1]
                                _fig_nd.add_trace(go.Scatter(
                                    x=_pro_h_df['time'], y=_nd_vals,
                                    mode='lines+markers', name='Net Delta',
                                    line=dict(color='#00BCD4', width=2),
                                    marker=dict(
                                        size=4,
                                        color=['#00C853' if v >= 0 else '#FF5252' for v in _nd_vals]),
                                    fill='tozeroy', fillcolor='rgba(0,188,212,0.08)'
                                ))
                                _fig_nd.add_hline(y=0, line_color='white', line_width=1)
                                _fig_nd.add_trace(go.Scatter(
                                    x=[_pro_h_df['time'].iloc[-1]], y=[_nd_cur],
                                    mode='markers+text', text=[f'{_nd_cur:+.0f}'],
                                    textposition='top right',
                                    textfont=dict(size=9, color='#00BCD4'),
                                    marker=dict(size=9, color='#00BCD4', symbol='circle'),
                                    showlegend=False, hoverinfo='skip'
                                ))
                            _fig_nd.update_layout(
                                title=f'Net Delta Shift — {_pro_net_delta:+,.0f}',
                                height=250, template='plotly_dark',
                                margin=dict(l=40, r=20, t=45, b=30),
                                xaxis=dict(tickformat='%H:%M', gridcolor='#333'),
                                yaxis=dict(gridcolor='#333', title='Net Δ', zeroline=True,
                                           zerolinecolor='white'),
                                showlegend=False,
                                paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='#ccc'),
                            )
                            st.plotly_chart(_fig_nd, use_container_width=True)
                            _nd_sig = ("📈 Bullish Positioning" if _pro_net_delta > 0
                                       else "📉 Bearish Positioning")
                            st.caption(_nd_sig)

                        with _bc3:
                            # Breakout Probability Bar Chart
                            _fig_bs = go.Figure()
                            if 'breakout_score' in _pro_h_df.columns:
                                _bs_s = _pro_h_df['breakout_score'].dropna()
                                _bs_t = _pro_h_df['time'].loc[_bs_s.index]
                                _fig_bs.add_trace(go.Bar(
                                    x=_bs_t, y=_bs_s,
                                    marker_color=['#FF5252' if v >= 70
                                                  else '#FFD740' if v >= 40
                                                  else '#00BCD4' for v in _bs_s],
                                    name='Breakout Score'
                                ))
                                _fig_bs.add_hline(y=70, line_dash='dot',
                                                  line_color='rgba(255,82,82,0.7)',
                                                  annotation_text='Breakout 70',
                                                  annotation_position='right',
                                                  annotation_font_size=8)
                                _fig_bs.add_hline(y=40, line_dash='dot',
                                                  line_color='rgba(255,215,0,0.7)',
                                                  annotation_text='Watch 40',
                                                  annotation_position='right',
                                                  annotation_font_size=8)
                            _fig_bs.update_layout(
                                title=f'Breakout Probability — {_pro_score}/100',
                                height=250, template='plotly_dark',
                                margin=dict(l=30, r=50, t=45, b=30),
                                xaxis=dict(tickformat='%H:%M', gridcolor='#333'),
                                yaxis=dict(gridcolor='#333', range=[0, 105], title='Score'),
                                showlegend=False,
                                paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='#ccc'),
                            )
                            st.plotly_chart(_fig_bs, use_container_width=True)
                            st.caption(f"Straddle {_s1}/20 · IV {_s2}/20 · Gamma {_s3}/20 · "
                                       f"Volume {_s4}/20 · Pressure {_s5}/20")

                    # ── Sentiment Over Time Chart ───────────────────────────────
                    if st.session_state.sentiment_history:
                        _sent_h_df = pd.DataFrame(st.session_state.sentiment_history)
                        _fig_sent = go.Figure()
                        _sent_mkr = []
                        for _, _shr in _sent_h_df.iterrows():
                            _sv = _shr.get('sentiment_score', 50)
                            _sent_mkr.append(
                                '#00ff88' if _sv >= 70 else '#FFD700' if _sv >= 40 else '#FF5252')
                        _fig_sent.add_trace(go.Scatter(
                            x=_sent_h_df['time'], y=_sent_h_df['sentiment_score'],
                            mode='lines+markers', name='Sentiment Score',
                            line=dict(color='#00aaff', width=2.5),
                            marker=dict(size=6, color=_sent_mkr),
                            fill='tozeroy', fillcolor='rgba(0,170,255,0.07)'))
                        if 'comp_score_pct' in _sent_h_df.columns:
                            _fig_sent.add_trace(go.Scatter(
                                x=_sent_h_df['time'],
                                y=_sent_h_df['comp_score_pct'].apply(lambda x: (x + 100) / 2),
                                mode='lines', name='Direction Score (scaled 0-100)',
                                line=dict(color='#ff44ff', width=1.5, dash='dash')))
                        if 'total_gex' in _sent_h_df.columns:
                            _fig_sent.add_trace(go.Scatter(
                                x=_sent_h_df['time'],
                                y=_sent_h_df['total_gex'].apply(
                                    lambda x: max(0, min(100, (x + 100) / 2))),
                                mode='lines', name='GEX (scaled)',
                                line=dict(color='#FFD740', width=1.5, dash='dot')))
                        _fig_sent.add_hline(y=70, line_dash='dot',
                            line_color='rgba(0,255,136,0.5)', line_width=1,
                            annotation_text='Bullish 70', annotation_position='right',
                            annotation_font_size=8)
                        _fig_sent.add_hline(y=30, line_dash='dot',
                            line_color='rgba(255,68,68,0.5)', line_width=1,
                            annotation_text='Bearish 30', annotation_position='right',
                            annotation_font_size=8)
                        _fig_sent.add_hrect(y0=70, y1=105,
                            fillcolor='rgba(0,255,136,0.05)', line_width=0)
                        _fig_sent.add_hrect(y0=0, y1=30,
                            fillcolor='rgba(255,68,68,0.05)', line_width=0)
                        _fig_sent.update_layout(
                            title=f'Options Flow Sentiment — Current: {_sentiment_score}/100 ({_sent_verdict})',
                            height=320, template='plotly_dark',
                            xaxis=dict(tickformat='%H:%M', gridcolor='#333'),
                            yaxis=dict(title='Score (0–100)', range=[0, 105], gridcolor='#333'),
                            legend=dict(orientation='h', y=-0.3, font=dict(size=9)),
                            margin=dict(l=40, r=65, t=45, b=30),
                            paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='#ccc'))
                        st.plotly_chart(_fig_sent, use_container_width=True)

                    # ── Per-Strike Breakdown ─────────────────────────────────────
                    if _comp_strike_details:
                        with st.expander("📋 Per-Strike Direction Breakdown"):
                            st.dataframe(pd.DataFrame(_comp_strike_details),
                                         use_container_width=True, hide_index=True)

                    # Controls
                    _pro_cl, _pro_cm, _pro_cr = st.columns([3, 1, 1])
                    with _pro_cl:
                        _comp_status = "🟢 Live" if _comp_data_available else "🟡 Cached"
                        st.caption(f"📊 Pro: {len(st.session_state.pro_trader_history)} pts · "
                                   f"Sentiment: {len(st.session_state.sentiment_history)} pts · "
                                   f"Composite: {len(st.session_state.composite_signal_history)} pts · "
                                   f"{_comp_status} · Refreshes every 30s")
                    with _pro_cm:
                        if st.button("🗑️ Clear Pro"):
                            st.session_state.pro_trader_history = []
                            st.session_state.pro_smart_signal_last = None
                            st.session_state.sentiment_history = []
                            st.rerun()
                    with _pro_cr:
                        if st.button("🗑️ Clear Composite"):
                            st.session_state.composite_signal_history = []
                            st.session_state.composite_signal_last_valid = None
                            st.rerun()
                else:
                    st.info("ATM strike not identified — Pro Dashboard unavailable.")
            else:
                st.info("Option chain data required for Pro Trader Dashboard.")
        except Exception as _pro_e:
            st.warning(f"Pro Dashboard error: {str(_pro_e)}")

        # ===== INSTITUTIONAL TRADE MAP ENGINE =====
        st.markdown("---")
        st.markdown("## 🗺️ Institutional Trade Map")
        try:
            _itm_df   = option_data.get('df_summary') if option_data else None
            _itm_spot = option_data.get('underlying') if option_data else None
            _itm_lot  = 25

            # Latest sentiment score from session state
            _itm_sent_score = 50
            _itm_market_bias = "SIDEWAYS"
            if st.session_state.sentiment_history:
                _itm_sent_score = st.session_state.sentiment_history[-1]['sentiment_score']
                _itm_market_bias = ("BULLISH" if _itm_sent_score > 65 else
                                    "BEARISH" if _itm_sent_score < 35 else "SIDEWAYS")
            _itm_final_signal = (st.session_state.sentiment_history[-1].get('verdict', '⚪ NEUTRAL')
                                 if st.session_state.sentiment_history else '⚪ NEUTRAL')

            if _itm_df is not None and _itm_spot and 'Zone' in _itm_df.columns:
                _itm_atm_idx = _itm_df[_itm_df['Zone'] == 'ATM'].index
                if len(_itm_atm_idx) > 0:
                    _itm_atm_pos = _itm_df.index.get_loc(_itm_atm_idx[0])
                    # ATM±5 for OI walls
                    _itm_s5 = max(0, _itm_atm_pos - 5)
                    _itm_e5 = min(len(_itm_df), _itm_atm_pos + 6)
                    _itm_slice5 = _itm_df.iloc[_itm_s5:_itm_e5].copy()
                    _itm_atm_val = float(_itm_df[_itm_df['Zone'] == 'ATM']['Strike'].values[0])
                    _itm_stk_list = sorted(_itm_slice5['Strike'].unique())
                    _itm_step = int(_itm_stk_list[1] - _itm_stk_list[0]) if len(_itm_stk_list) >= 2 else 50
                    for _c in ['openInterest_CE', 'openInterest_PE', 'Gamma_CE', 'Gamma_PE',
                               'changeinOpenInterest_CE', 'changeinOpenInterest_PE']:
                        if _c not in _itm_slice5.columns:
                            _itm_slice5[_c] = 0.0

                    # ── Put OI Wall & Call OI Wall ──────────────────────────────
                    _pe_oi_s = _itm_slice5.set_index('Strike')['openInterest_PE'].fillna(0)
                    _ce_oi_s = _itm_slice5.set_index('Strike')['openInterest_CE'].fillna(0)
                    _put_oi_wall = float(_pe_oi_s.idxmax()) if len(_pe_oi_s) > 0 else _itm_atm_val - _itm_step * 2
                    _call_oi_wall = float(_ce_oi_s.idxmax()) if len(_ce_oi_s) > 0 else _itm_atm_val + _itm_step * 2

                    # ── Gamma Support / Resistance / Flip ──────────────────────
                    _itm_slice5['_gex'] = (
                        _itm_slice5['Gamma_PE'].fillna(0) * _itm_slice5['openInterest_PE'].fillna(0) *
                        _itm_lot * _itm_spot / 100000 -
                        _itm_slice5['Gamma_CE'].fillna(0) * _itm_slice5['openInterest_CE'].fillna(0) *
                        _itm_lot * _itm_spot / 100000)
                    _below_atm = _itm_slice5[_itm_slice5['Strike'] < _itm_atm_val]
                    _above_atm = _itm_slice5[_itm_slice5['Strike'] > _itm_atm_val]
                    _gamma_support = (float(_below_atm.loc[_below_atm['_gex'].idxmax(), 'Strike'])
                                      if len(_below_atm) > 0 else _itm_atm_val - _itm_step)
                    _gamma_resist  = (float(_above_atm.loc[_above_atm['_gex'].idxmin(), 'Strike'])
                                      if len(_above_atm) > 0 else _itm_atm_val + _itm_step)
                    # Gamma flip: first sign change in GEX
                    _gamma_flip = _itm_atm_val
                    _gex_arr = _itm_slice5['_gex'].values
                    for _gi in range(len(_gex_arr) - 1):
                        if (_gex_arr[_gi] * _gex_arr[_gi + 1]) < 0:
                            _gamma_flip = (_itm_slice5['Strike'].iloc[_gi] +
                                           _itm_slice5['Strike'].iloc[_gi + 1]) / 2
                            break

                    # ── VOB Support & Resistance ────────────────────────────────
                    _vob_support = _itm_atm_val - _itm_step * 3
                    _vob_resist  = _itm_atm_val + _itm_step * 3
                    if 'sr_levels' in vob_data:
                        _vs = [l.get('mid', l.get('lower', 0)) for l in vob_data['sr_levels']
                               if ('Support' in l.get('Type', '') or '🟢' in l.get('Type', ''))
                               and l.get('mid', l.get('lower', 0)) > 0
                               and l.get('mid', l.get('lower', 0)) < _itm_spot]
                        _vr = [l.get('mid', l.get('upper', 0)) for l in vob_data['sr_levels']
                               if ('Resistance' in l.get('Type', '') or '🔴' in l.get('Type', ''))
                               and l.get('mid', l.get('upper', 0)) > _itm_spot]
                        if _vs: _vob_support = max(_vs)
                        if _vr: _vob_resist  = min(_vr)

                    # ── Triple POC ──────────────────────────────────────────────
                    _poc_val = _itm_spot
                    if poc_data_for_chart:
                        _pvs = []
                        for _pk in ['poc1', 'poc2', 'poc3']:
                            _pd2 = poc_data_for_chart.get(_pk, {})
                            _pv2 = _pd2.get('poc', 0) if isinstance(_pd2, dict) else 0
                            if isinstance(_pv2, (int, float)) and _pv2 > 0:
                                _pvs.append(float(_pv2))
                        if _pvs: _poc_val = sum(_pvs) / len(_pvs)

                    # ── HTF Pivot Support & Resistance ──────────────────────────
                    _piv_support = _itm_atm_val - _itm_step * 4
                    _piv_resist  = _itm_atm_val + _itm_step * 4
                    if pivots:
                        _pl = [p['value'] for p in pivots
                               if p.get('type') == 'low' and p['value'] < _itm_spot]
                        _ph = [p['value'] for p in pivots
                               if p.get('type') == 'high' and p['value'] > _itm_spot]
                        if _pl: _piv_support = max(_pl)
                        if _ph: _piv_resist  = min(_ph)

                    # ── STEP 2: True Support ────────────────────────────────────
                    _true_support = (0.30 * _put_oi_wall + 0.25 * _gamma_support +
                                     0.20 * _vob_support + 0.15 * _poc_val + 0.10 * _piv_support)
                    _true_support = round(_true_support / _itm_step) * _itm_step

                    # ── STEP 3: True Resistance ─────────────────────────────────
                    _true_resist = (0.30 * _call_oi_wall + 0.25 * _gamma_resist +
                                    0.20 * _vob_resist + 0.15 * _poc_val + 0.10 * _piv_resist)
                    _true_resist = round(_true_resist / _itm_step) * _itm_step

                    # ── STEP 4: Entry Zones ─────────────────────────────────────
                    _entry_support = round(
                        (_gamma_support + _put_oi_wall + _vob_support) / 3 / _itm_step) * _itm_step
                    _entry_resist  = round(
                        (_gamma_resist + _call_oi_wall + _vob_resist) / 3 / _itm_step) * _itm_step

                    # ── STEP 5: Exit Targets ────────────────────────────────────
                    _above_spot = [l for l in [_call_oi_wall, _gamma_resist, _piv_resist, _poc_val]
                                   if l > _itm_spot]
                    _below_spot = [l for l in [_put_oi_wall, _gamma_support, _piv_support, _poc_val]
                                   if l < _itm_spot]
                    _exit_resist  = (round(min(_above_spot) / _itm_step) * _itm_step
                                     if _above_spot else _true_resist)
                    _exit_support = (round(max(_below_spot) / _itm_step) * _itm_step
                                     if _below_spot else _true_support)

                    # ── STEP 6: Breakout Level ──────────────────────────────────
                    _breakout_lvl = round(
                        (_itm_atm_val + _itm_step + _gamma_flip + _call_oi_wall) / 3 / _itm_step) * _itm_step

                    # ── STEP 7: Breakdown Level ─────────────────────────────────
                    _breakdown_lvl = round(
                        (_itm_atm_val - _itm_step + _gamma_flip + _put_oi_wall) / 3 / _itm_step) * _itm_step

                    # ── STEP 8: Stop Loss Engine ────────────────────────────────
                    _sl_call_cands = [l for l in [_gamma_support, _put_oi_wall, _piv_support]
                                      if l < _entry_support]
                    _sl_call = (round(max(_sl_call_cands) / _itm_step) * _itm_step
                                if _sl_call_cands else _entry_support - _itm_step * 2)
                    _sl_put_cands = [l for l in [_gamma_resist, _call_oi_wall, _piv_resist]
                                     if l > _entry_resist]
                    _sl_put = (round(min(_sl_put_cands) / _itm_step) * _itm_step
                               if _sl_put_cands else _entry_resist + _itm_step * 2)

                    # ── STEP 9: Active Trade Setup ──────────────────────────────
                    if _itm_market_bias == "BULLISH":
                        _itm_entry   = _entry_support
                        _itm_target  = _exit_resist
                        _itm_sl      = _sl_call
                        _itm_dir_clr = "#00ff88"
                        _itm_dir_lbl = "🚀 BULLISH"
                    elif _itm_market_bias == "BEARISH":
                        _itm_entry   = _entry_resist
                        _itm_target  = _exit_support
                        _itm_sl      = _sl_put
                        _itm_dir_clr = "#FF5252"
                        _itm_dir_lbl = "🔥 BEARISH"
                    else:
                        _itm_entry   = _itm_atm_val
                        _itm_target  = _itm_atm_val
                        _itm_sl      = _itm_atm_val
                        _itm_dir_clr = "#FFD700"
                        _itm_dir_lbl = "↔️ SIDEWAYS"

                    # ── STEP 10: Confidence Score ───────────────────────────────
                    _conf_pts = 0
                    _itm_net_gex = 0.0
                    _itm_net_delta = 0.0
                    _itm_call_pres = _itm_put_pres = 0.5
                    _itm_wpcr_oi = 1.0
                    # Pull from session state if available
                    if st.session_state.pro_trader_history:
                        _last_pro = st.session_state.pro_trader_history[-1]
                        _itm_net_gex   = _last_pro.get('net_gex', 0)
                        _itm_net_delta = _last_pro.get('net_delta', 0)
                        _itm_call_pres = _last_pro.get('call_pressure', 0.5)
                        _itm_put_pres  = _last_pro.get('put_pressure', 0.5)
                        _itm_wpcr_oi   = _last_pro.get('pcr_oi', 1.0)
                    # PCR aligned
                    if ((_itm_market_bias == "BULLISH" and _itm_wpcr_oi > 1.1) or
                            (_itm_market_bias == "BEARISH" and _itm_wpcr_oi < 0.9)):
                        _conf_pts += 1
                    # GEX trending (negative = conviction)
                    if _itm_net_gex < -5:
                        _conf_pts += 1
                    # Delta aligned
                    if ((_itm_market_bias == "BULLISH" and _itm_net_delta > 0) or
                            (_itm_market_bias == "BEARISH" and _itm_net_delta < 0)):
                        _conf_pts += 1
                    # Pressure aligned
                    if ((_itm_market_bias == "BULLISH" and _itm_call_pres > _itm_put_pres) or
                            (_itm_market_bias == "BEARISH" and _itm_put_pres > _itm_call_pres)):
                        _conf_pts += 1
                    # Sentiment extreme
                    if _itm_sent_score > 65 or _itm_sent_score < 35:
                        _conf_pts += 1
                    _confidence = ("HIGH" if _conf_pts >= 4 else
                                   "MEDIUM" if _conf_pts >= 2 else "LOW")
                    _conf_clr   = ("#00ff88" if _confidence == "HIGH" else
                                   "#FFD700"  if _confidence == "MEDIUM" else "#FF5252")

                    # ── STEP 11: Telegram Alert ─────────────────────────────────
                    if _confidence == "HIGH" and _itm_market_bias != "SIDEWAYS" and enable_signals:
                        _itm_key = f"{_itm_market_bias}_{int(_itm_entry)}_{int(_itm_target)}"
                        _last_itm = st.session_state.itm_last_alert
                        _itm_now2 = datetime.now(pytz.timezone('Asia/Kolkata'))
                        _itm_ok   = (_last_itm[0] != _itm_key or
                                     (_last_itm[1] is not None and
                                      (_itm_now2 - _last_itm[1]).total_seconds() > 600))
                        if _itm_ok:
                            send_telegram_message_sync(
                                f"<b>🗺️ INSTITUTIONAL TRADE SETUP</b>\n"
                                f"Spot: ₹{_itm_spot:.0f}\n"
                                f"Direction: {_itm_dir_lbl}\n"
                                f"Entry: {_itm_entry:.0f} | Target: {_itm_target:.0f} | SL: {_itm_sl:.0f}\n"
                                f"Support: {_true_support:.0f} | Resistance: {_true_resist:.0f}\n"
                                f"Breakout: {_breakout_lvl:.0f} | Breakdown: {_breakdown_lvl:.0f}\n"
                                f"Confidence: {_confidence} | Sentiment: {_itm_sent_score}/100\n"
                                f"Flow: {_itm_final_signal}"
                            )
                            st.session_state.itm_last_alert = (_itm_key, _itm_now2)

                    # ── STEP 9 Display: Trade Map Panel ────────────────────────
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{_itm_dir_clr}08,{_itm_dir_clr}18);
                                padding:18px;border-radius:12px;border:2px solid {_itm_dir_clr};
                                margin-bottom:14px;">
                        <div style="color:#888;font-size:10px;letter-spacing:2px;">
                            INSTITUTIONAL TRADE MAP — ATM±5 LIQUIDITY ANALYSIS</div>
                        <div style="color:{_itm_dir_clr};font-size:28px;font-weight:bold;margin:6px 0;">
                            {_itm_dir_lbl} &nbsp;·&nbsp; Confidence: <span style="border:1px solid {_conf_clr};
                            color:{_conf_clr};padding:2px 10px;border-radius:5px;font-size:16px;">{_confidence}</span>
                        </div>
                        <div style="color:#aaa;font-size:12px;">
                            Spot: <b style="color:#FFD740">{_itm_spot:,.0f}</b> &nbsp;|&nbsp;
                            ATM: <b style="color:#FFD740">₹{int(_itm_atm_val)}</b> &nbsp;|&nbsp;
                            Sentiment: <b style="color:{_conf_clr}">{_itm_sent_score}/100</b> &nbsp;|&nbsp;
                            {_itm_final_signal}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # Level grid (3 cols × 3 rows)
                    _lg1, _lg2, _lg3 = st.columns(3)
                    def _itm_card(label, value, color, note=""):
                        return (f'<div style="background:{color}12;padding:11px;border-radius:8px;'
                                f'border:1px solid {color};text-align:center;margin:3px 0;">'
                                f'<div style="color:{color};font-size:10px;font-weight:bold;">{label}</div>'
                                f'<div style="color:{color};font-size:20px;font-weight:bold;">{value:.0f}</div>'
                                f'<div style="color:#aaa;font-size:10px;">{note}</div></div>')

                    with _lg1:
                        st.markdown(_itm_card("PUT OI WALL",     _put_oi_wall,   "#00ff88", "Max PE OI"),
                                    unsafe_allow_html=True)
                        st.markdown(_itm_card("TRUE SUPPORT",    _true_support,  "#00C853", "Weighted avg"),
                                    unsafe_allow_html=True)
                        st.markdown(_itm_card("ENTRY SUPPORT",   _entry_support, "#00aaff", "GEX+OI+VOB"),
                                    unsafe_allow_html=True)
                    with _lg2:
                        st.markdown(_itm_card("TARGET",          _itm_target,    _itm_dir_clr, "Exit zone"),
                                    unsafe_allow_html=True)
                        st.markdown(_itm_card("ENTRY",           _itm_entry,     _itm_dir_clr, "Trade entry"),
                                    unsafe_allow_html=True)
                        st.markdown(_itm_card("STOP LOSS",       _itm_sl,        "#FF5252", "Max risk"),
                                    unsafe_allow_html=True)
                    with _lg3:
                        st.markdown(_itm_card("CALL OI WALL",    _call_oi_wall,  "#FF5252", "Max CE OI"),
                                    unsafe_allow_html=True)
                        st.markdown(_itm_card("TRUE RESISTANCE", _true_resist,   "#FF1744", "Weighted avg"),
                                    unsafe_allow_html=True)
                        st.markdown(_itm_card("ENTRY RESIST",    _entry_resist,  "#ff8844", "GEX+OI+VOB"),
                                    unsafe_allow_html=True)

                    # Breakout / Breakdown / Gamma levels
                    _bb1, _bb2, _bb3, _bb4 = st.columns(4)
                    with _bb1:
                        st.markdown(_itm_card("BREAKOUT ABOVE", _breakout_lvl,  "#00ff88", "ATM+1+GEX+Call OI"),
                                    unsafe_allow_html=True)
                    with _bb2:
                        st.markdown(_itm_card("BREAKDOWN BELOW", _breakdown_lvl, "#FF5252", "ATM-1+GEX+Put OI"),
                                    unsafe_allow_html=True)
                    with _bb3:
                        st.markdown(_itm_card("GAMMA FLIP",     _gamma_flip,    "#FFD700", "GEX zero crossing"),
                                    unsafe_allow_html=True)
                    with _bb4:
                        st.markdown(_itm_card("POC (avg)",      _poc_val,       "#00BCD4", "Triple POC avg"),
                                    unsafe_allow_html=True)

                    # ── STEP 12: Institutional Levels Chart ────────────────────
                    if st.session_state.sentiment_history:
                        _itm_chart_times = [h['time'] for h in st.session_state.sentiment_history]
                        _fig_itm = go.Figure()
                        for _lv2, _ln, _lc, _ld in [
                            (_true_support,  "Support",        "#00C853", "dash"),
                            (_true_resist,   "Resistance",     "#FF1744", "dash"),
                            (_itm_entry,     "Entry Zone",     _itm_dir_clr, "dot"),
                            (_itm_target,    "Target",         "#00BCD4", "dot"),
                            (_breakout_lvl,  "Breakout Level", "#00ff88", "dashdot"),
                            (_breakdown_lvl, "Breakdown Level","#FF5252", "dashdot"),
                            (_itm_sl,        "Stop Loss",      "#ff8800", "longdash"),
                        ]:
                            _fig_itm.add_hline(
                                y=_lv2, line_dash=_ld, line_color=_lc, line_width=1.5,
                                annotation_text=f"{_ln}: {_lv2:.0f}",
                                annotation_position="right",
                                annotation_font_size=9,
                                annotation_font_color=_lc)
                        _fig_itm.add_trace(go.Scatter(
                            x=_itm_chart_times,
                            y=[_itm_spot] * len(_itm_chart_times),
                            mode='lines', name='Spot Price',
                            line=dict(color='#FFD740', width=2)))
                        _fig_itm.update_layout(
                            title=f"Institutional Trade Map — Spot: {_itm_spot:,.0f}",
                            height=350, template='plotly_dark',
                            xaxis=dict(tickformat='%H:%M', gridcolor='#333'),
                            yaxis=dict(title='Price', gridcolor='#333'),
                            legend=dict(orientation='h', y=-0.3, font=dict(size=9)),
                            margin=dict(l=40, r=120, t=45, b=30),
                            paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='#ccc'))
                        st.plotly_chart(_fig_itm, use_container_width=True)

                    # Source breakdown
                    with st.expander("📖 Level Sources"):
                        _src_data = [
                            ["Put OI Wall",     f"₹{_put_oi_wall:.0f}",   "Max PE open interest strike (ATM±5)"],
                            ["Call OI Wall",    f"₹{_call_oi_wall:.0f}",  "Max CE open interest strike (ATM±5)"],
                            ["Gamma Support",   f"₹{_gamma_support:.0f}", "Highest positive GEX below ATM"],
                            ["Gamma Resistance",f"₹{_gamma_resist:.0f}",  "Lowest negative GEX above ATM"],
                            ["Gamma Flip",      f"₹{_gamma_flip:.0f}",    "GEX sign change level"],
                            ["VOB Support",     f"₹{_vob_support:.0f}",   "Volume Order Block nearest below spot"],
                            ["VOB Resistance",  f"₹{_vob_resist:.0f}",    "Volume Order Block nearest above spot"],
                            ["POC (avg)",       f"₹{_poc_val:.0f}",       "Average of Short/Mid/Long POC"],
                            ["Pivot Support",   f"₹{_piv_support:.0f}",   "Nearest HTF pivot low below spot"],
                            ["Pivot Resistance",f"₹{_piv_resist:.0f}",    "Nearest HTF pivot high above spot"],
                        ]
                        st.dataframe(pd.DataFrame(_src_data,
                            columns=['Level', 'Value', 'Source']),
                            use_container_width=True, hide_index=True)
                else:
                    st.info("ATM strike not identified in option chain.")
            else:
                st.info("Option chain data required for Institutional Trade Map.")
        except Exception as _itm_e:
            st.warning(f"Institutional Trade Map error: {str(_itm_e)}")

        # ===== ATM ±2 STRIKE COMPARISON: PCR / ChgOI PCR / GEX =====
        st.markdown("---")
        st.markdown("## 📊 ATM ±2 Strike Comparison — PCR · ChgOI PCR")

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

                # Dynamic Y range: data + reference thresholds always in view
                _pcr_raw = history_df[col_name].dropna().tolist()
                _pcr_all = _pcr_raw + [0.7, 1.2]
                _pcr_ymin = max(0.0, min(_pcr_all) * 0.9)
                _pcr_ymax = max(_pcr_all) * 1.1

                fig.update_layout(
                    title=f"{title_prefix}<br>₹{strike_val}<br>PCR: {current_pcr:.2f}",
                    template='plotly_dark',
                    height=280,
                    showlegend=False,
                    margin=dict(l=10, r=10, t=70, b=30),
                    xaxis=dict(tickformat='%H:%M', title=''),
                    yaxis=dict(title='PCR', range=[_pcr_ymin, _pcr_ymax]),
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e'
                )
                return fig, current_pcr
            return None, 0

        # Helper function for GEX charts (defined here so it's available in comparison view)
        def create_gex_chart(history_df, col_name, color, title_prefix):
            """Create individual GEX time-series chart per strike."""
            if col_name and col_name in history_df.columns:
                strike_val = col_name
                gex_values = history_df[col_name].dropna()
                max_abs = max(abs(gex_values.max()), abs(gex_values.min()), 15) if len(gex_values) > 0 else 20
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
                    template='plotly_dark', height=280, showlegend=False,
                    margin=dict(l=10, r=10, t=70, b=30),
                    xaxis=dict(tickformat='%H:%M', title=''),
                    yaxis=dict(title='GEX (L)', range=y_range, zeroline=True,
                               zerolinecolor='white', zerolinewidth=2,
                               tickmode='array',
                               tickvals=[-20, -10, 0, 10, 20] if max_abs <= 25 else None),
                    plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e'
                )
                return fig, current_gex
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

                    chgoi_avail = [c for c in ['changeinOpenInterest_CE', 'changeinOpenInterest_PE']
                                   if c in df_summary.columns]
                    pcr_df = df_summary.iloc[start_idx:end_idx][
                        ['Strike', 'Zone', 'PCR', 'PCR_Signal',
                         'openInterest_CE', 'openInterest_PE'] + chgoi_avail
                    ].copy()

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
                        chgoi_entry = {'time': current_time}
                        for _, row in pcr_df.iterrows():
                            strike_label = str(int(row['Strike']))  # Store by strike only
                            pcr_entry[strike_label] = row['PCR']
                            # Per-strike ChgOI PCR
                            if 'changeinOpenInterest_CE' in row and 'changeinOpenInterest_PE' in row:
                                ce_chg = row['changeinOpenInterest_CE']
                                pe_chg = row['changeinOpenInterest_PE']
                                chgoi_entry[strike_label] = round(abs(pe_chg / ce_chg), 3) if ce_chg != 0 else 0.0

                        # Store current ATM ±2 strike positions for display
                        current_strikes = pcr_df['Strike'].tolist()
                        st.session_state.pcr_current_strikes = [int(s) for s in current_strikes]
                        st.session_state.pcr_chgoi_strike_current_strikes = [int(s) for s in current_strikes]

                        # Check if we should add new entry (avoid duplicates within 30 seconds)
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
                            # Also store per-strike ChgOI PCR if data was available
                            if len(chgoi_entry) > 1:
                                st.session_state.pcr_chgoi_strike_history.append(chgoi_entry)
                                if len(st.session_state.pcr_chgoi_strike_history) > 200:
                                    st.session_state.pcr_chgoi_strike_history = st.session_state.pcr_chgoi_strike_history[-200:]

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

                position_labels = ['🟣 ITM-2', '🟣 ITM-1', '🟡 ATM', '🔵 OTM+1', '🔵 OTM+2']
                position_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']

                # ── PCR OI + ChgOI PCR per strike ──
                chgoi_hist = st.session_state.pcr_chgoi_strike_history
                chgoi_history_df = pd.DataFrame(chgoi_hist) if chgoi_hist else None

                combined_cols = st.columns(5)
                for i, col in enumerate(combined_cols):
                    with col:
                        if i >= len(current_strikes):
                            st.info(f"{position_labels[i]} N/A")
                            continue

                        strike = current_strikes[i]
                        strike_col = str(strike)

                        fig = go.Figure()

                        # ── PCR OI (solid blue) ──
                        if strike_col in history_df.columns:
                            fig.add_trace(go.Scatter(
                                x=history_df['time'],
                                y=history_df[strike_col],
                                mode='lines+markers',
                                name='PCR OI',
                                line=dict(color='#00ccff', width=2),
                                marker=dict(size=3),
                            ))

                        # ── ChgOI PCR (dashed orange) ──
                        if chgoi_history_df is not None and strike_col in chgoi_history_df.columns:
                            fig.add_trace(go.Scatter(
                                x=chgoi_history_df['time'],
                                y=chgoi_history_df[strike_col],
                                mode='lines+markers',
                                name='PCR ChgOI',
                                line=dict(color='#ffaa00', width=2, dash='dash'),
                                marker=dict(size=3),
                            ))

                        # PCR reference lines
                        fig.add_hline(y=1.2, line_dash="dot", line_color="rgba(0,255,136,0.533)", line_width=1,
                                      annotation_text="Bull 1.2", annotation_position="right",
                                      annotation_font_size=8, annotation_font_color="#00ff88")
                        fig.add_hline(y=0.7, line_dash="dot", line_color="rgba(255,68,68,0.533)", line_width=1,
                                      annotation_text="Bear 0.7", annotation_position="right",
                                      annotation_font_size=8, annotation_font_color="#ff4444")

                        # Dynamic Y range: include both OI PCR and ChgOI PCR values + thresholds
                        _cmp_vals = []
                        if strike_col in history_df.columns:
                            _cmp_vals += history_df[strike_col].dropna().tolist()
                        if chgoi_history_df is not None and strike_col in chgoi_history_df.columns:
                            _cmp_vals += chgoi_history_df[strike_col].dropna().tolist()
                        _cmp_vals += [0.7, 1.2]
                        _cmp_ymin = max(0.0, min(_cmp_vals) * 0.9)
                        _cmp_ymax = max(_cmp_vals) * 1.1

                        # ── Current value markers (present value shown in graph) ──
                        if strike_col in history_df.columns and len(history_df) > 0:
                            _pcr_cur = history_df[strike_col].iloc[-1]
                            fig.add_trace(go.Scatter(
                                x=[history_df['time'].iloc[-1]], y=[_pcr_cur],
                                mode='markers+text', text=[f'{_pcr_cur:.2f}'],
                                textposition='top right', textfont=dict(size=9, color='#00ccff'),
                                marker=dict(size=9, color='#00ccff', symbol='circle'),
                                showlegend=False, hoverinfo='skip',
                            ))
                        if chgoi_history_df is not None and strike_col in chgoi_history_df.columns and len(chgoi_history_df) > 0:
                            _chgoi_cur = chgoi_history_df[strike_col].iloc[-1]
                            fig.add_trace(go.Scatter(
                                x=[chgoi_history_df['time'].iloc[-1]], y=[_chgoi_cur],
                                mode='markers+text', text=[f'{_chgoi_cur:.2f}'],
                                textposition='bottom right', textfont=dict(size=9, color='#ffaa00'),
                                marker=dict(size=9, color='#ffaa00', symbol='diamond'),
                                showlegend=False, hoverinfo='skip',
                            ))

                        fig.update_layout(
                            title=dict(text=f"{position_labels[i]}<br>₹{strike}", font=dict(size=11)),
                            template='plotly_dark',
                            height=300,
                            showlegend=True,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                        xanchor='center', x=0.5, font=dict(size=8)),
                            margin=dict(l=5, r=10, t=70, b=30),
                            xaxis=dict(tickformat='%H:%M', title='', tickfont=dict(size=8)),
                            yaxis=dict(title='PCR', range=[_cmp_ymin, _cmp_ymax]),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Signal summary below chart
                        cur_pcr = history_df[strike_col].iloc[-1] if (
                            strike_col in history_df.columns and len(history_df) > 0) else 1.0
                        pcr_sig = "🟢 Bull" if cur_pcr > 1.2 else ("🔴 Bear" if cur_pcr < 0.7 else "🟡 Ntrl")
                        st.caption(f"PCR {cur_pcr:.2f} {pcr_sig}")

                # ── Send PCR chart to Telegram every 5 minutes ──
                _now = datetime.now()
                _last_sent = st.session_state.pcr_telegram_last_sent
                _should_send = (
                    _last_sent is None
                    or (_now - _last_sent).total_seconds() >= 300
                )
                if _should_send and current_strikes:
                    try:
                        from plotly.subplots import make_subplots as _make_subplots
                        _n = len(current_strikes)
                        _tg_fig = _make_subplots(
                            rows=1, cols=_n,
                            subplot_titles=[
                                f"{position_labels[i]}<br>₹{current_strikes[i]}"
                                for i in range(_n)
                            ],
                        )
                        for _i, _strike in enumerate(current_strikes):
                            _sc = str(_strike)
                            _col_idx = _i + 1
                            if _sc in history_df.columns:
                                _tg_fig.add_trace(go.Scatter(
                                    x=history_df['time'],
                                    y=history_df[_sc],
                                    mode='lines',
                                    name='PCR OI',
                                    line=dict(color='#00ccff', width=2),
                                    showlegend=(_i == 0),
                                ), row=1, col=_col_idx)
                            if chgoi_history_df is not None and _sc in chgoi_history_df.columns:
                                _tg_fig.add_trace(go.Scatter(
                                    x=chgoi_history_df['time'],
                                    y=chgoi_history_df[_sc],
                                    mode='lines',
                                    name='ChgOI PCR',
                                    line=dict(color='#ffaa00', width=2, dash='dash'),
                                    showlegend=(_i == 0),
                                ), row=1, col=_col_idx)
                            _tg_fig.add_hline(y=1.2, line_dash="dot", line_color="rgba(0,255,136,0.5)",
                                              line_width=1, row=1, col=_col_idx)
                            _tg_fig.add_hline(y=0.7, line_dash="dot", line_color="rgba(255,68,68,0.5)",
                                              line_width=1, row=1, col=_col_idx)
                        _tg_fig.update_layout(
                            title="ATM ±2 Strike Comparison — PCR · ChgOI PCR",
                            template='plotly_dark',
                            height=350,
                            width=1400,
                            paper_bgcolor='#1e1e1e',
                            plot_bgcolor='#1e1e1e',
                            margin=dict(l=10, r=10, t=80, b=40),
                            legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='center', x=0.5),
                        )
                        _tg_fig.update_xaxes(tickformat='%H:%M', tickfont=dict(size=8))
                        _tg_fig.update_yaxes(title_text='PCR', tickfont=dict(size=8))
                        _img_bytes = _tg_fig.to_image(format="png", scale=2)
                        # Build caption with current PCR values
                        _caption_parts = [f"<b>ATM ±2 PCR · ChgOI PCR</b>  {_now.strftime('%H:%M')}  Spot: ₹{underlying_price:.0f}"]
                        for _i, _strike in enumerate(current_strikes):
                            _sc = str(_strike)
                            _pcr_v = history_df[_sc].iloc[-1] if _sc in history_df.columns and len(history_df) > 0 else None
                            _chg_v = (chgoi_history_df[_sc].iloc[-1]
                                      if chgoi_history_df is not None and _sc in chgoi_history_df.columns and len(chgoi_history_df) > 0
                                      else None)
                            _lbl = position_labels[_i].split(" ", 1)[-1]  # strip emoji
                            _sig = "🟢" if (_pcr_v and _pcr_v > 1.2) else ("🔴" if (_pcr_v and _pcr_v < 0.7) else "🟡")
                            _pcr_str = f"{_pcr_v:.2f}" if _pcr_v is not None else "N/A"
                            _chg_str = f"{_chg_v:.2f}" if _chg_v is not None else "N/A"
                            _caption_parts.append(f"{_sig} ₹{_strike} ({_lbl}): OI={_pcr_str} | ChgOI={_chg_str}")
                        send_telegram_photo_sync(_img_bytes, "\n".join(_caption_parts))
                        st.session_state.pcr_telegram_last_sent = _now
                    except Exception as _tg_err:
                        pass  # silently skip if chart export fails

                # ── ATM ±2 Strike GEX (separate row) ──
                st.markdown("---")
                st.markdown("### 📊 ATM ±2 Strike GEX")
                has_gex_hist = len(st.session_state.gex_history) > 0
                gex_hist_df = pd.DataFrame(st.session_state.gex_history) if has_gex_hist else None
                gex_current_strikes = sorted(getattr(st.session_state, 'gex_current_strikes', current_strikes))

                cur_gex_vals = {}
                if gex_data_pre and 'gex_df' in gex_data_pre:
                    for _, gr in gex_data_pre['gex_df'].iterrows():
                        cur_gex_vals[int(gr['Strike'])] = gr['Net_GEX']

                gex_cols = st.columns(5)
                for i, col in enumerate(gex_cols):
                    with col:
                        if i >= len(current_strikes):
                            st.info(f"{position_labels[i]} N/A")
                            continue
                        strike = current_strikes[i]
                        strike_col = str(strike)
                        clr = position_colors[i]
                        _rgb = tuple(int(clr.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
                        _fill = f'rgba({_rgb[0]},{_rgb[1]},{_rgb[2]},0.15)'
                        fig_gex = go.Figure()
                        cur_gex = cur_gex_vals.get(strike, 0)
                        if has_gex_hist and gex_hist_df is not None and strike_col in gex_hist_df.columns:
                            gex_series = gex_hist_df[strike_col]
                            cur_gex = gex_series.iloc[-1]
                            gex_vals = gex_series.dropna()
                            max_abs = max(abs(gex_vals.max()), abs(gex_vals.min()), 15) if len(gex_vals) > 0 else 20
                            fig_gex.add_trace(go.Scatter(
                                x=gex_hist_df['time'],
                                y=gex_series,
                                mode='lines+markers',
                                name='GEX (L)',
                                line=dict(color=clr, width=2),
                                marker=dict(size=3),
                                fill='tozeroy',
                                fillcolor=_fill,
                            ))
                            fig_gex.update_layout(yaxis=dict(
                                range=[-max_abs * 1.1, max_abs * 1.1],
                                zeroline=True, zerolinecolor='white', zerolinewidth=2,
                            ))
                        elif strike in cur_gex_vals:
                            fig_gex.add_trace(go.Scatter(
                                x=[datetime.now(pytz.timezone('Asia/Kolkata'))],
                                y=[cur_gex],
                                mode='markers+text',
                                name='GEX (L)',
                                marker=dict(size=14, color=clr, symbol='diamond'),
                                text=[f'{cur_gex:+.1f}L'],
                                textposition='top center',
                                textfont=dict(size=10, color='white'),
                            ))
                        fig_gex.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.4)", line_width=1)
                        fig_gex.add_hline(y=10, line_dash="dot", line_color="rgba(0,255,136,0.4)", line_width=1,
                                          annotation_text="+10", annotation_position="right",
                                          annotation_font_size=8, annotation_font_color="#00ff88")
                        fig_gex.add_hline(y=-10, line_dash="dot", line_color="rgba(255,68,68,0.4)", line_width=1,
                                          annotation_text="-10", annotation_position="right",
                                          annotation_font_size=8, annotation_font_color="#ff4444")
                        # Current value marker (present value shown in graph)
                        if has_gex_hist and gex_hist_df is not None and strike_col in gex_hist_df.columns and len(gex_hist_df) > 0:
                            fig_gex.add_trace(go.Scatter(
                                x=[gex_hist_df['time'].iloc[-1]], y=[cur_gex],
                                mode='markers+text', text=[f'{cur_gex:+.1f}L'],
                                textposition='top right', textfont=dict(size=9, color=clr),
                                marker=dict(size=9, color=clr, symbol='circle'),
                                showlegend=False, hoverinfo='skip',
                            ))
                        fig_gex.update_layout(
                            title=dict(text=f"{position_labels[i]}<br>₹{strike}<br>GEX: {cur_gex:+.1f}L",
                                       font=dict(size=11)),
                            template='plotly_dark',
                            height=300,
                            showlegend=False,
                            margin=dict(l=5, r=10, t=70, b=30),
                            xaxis=dict(tickformat='%H:%M', title='', tickfont=dict(size=8)),
                            yaxis=dict(title='GEX (L)',
                                       title_font=dict(color=clr, size=9),
                                       tickfont=dict(size=8)),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                        )
                        st.plotly_chart(fig_gex, use_container_width=True)
                        gex_sig = "📍 Pin" if cur_gex > 10 else ("⚡ Accel" if cur_gex < -10 else "➡️ Ntrl")
                        st.caption(f"GEX {cur_gex:+.1f}L {gex_sig}")

                # ── Status & controls ──
                col_info1, col_info2 = st.columns([3, 1])
                with col_info1:
                    status = "🟢 Live" if pcr_data_available else "🟡 Using cached history"
                    st.caption(f"{status} | 📈 {len(st.session_state.pcr_history)} PCR pts · "
                               f"{len(chgoi_hist)} ChgOI pts")
                with col_info2:
                    if st.button("🗑️ Clear History"):
                        st.session_state.pcr_history = []
                        st.session_state.pcr_last_valid_data = None
                        st.session_state.pcr_chgoi_strike_history = []
                        st.session_state.gex_history = []
                        st.rerun()

            except Exception as e:
                st.warning(f"Error displaying comparison charts: {str(e)}")
        else:
            st.info("📊 History will build up as the app refreshes. Please wait for data collection…")

        # ===== VOLUME PCR + STRADDLE + COMBINED SIGNAL (ATM ± 2) =====
        st.markdown("---")
        st.markdown("## 📊 Volume PCR · Straddle · Combined Signal (ATM ± 2)")

        # ── Collect Volume PCR + Straddle data ──
        _vp_df_src = option_data.get('df_summary') if option_data else None
        _vp_data_ok = False
        if (_vp_df_src is not None and 'Zone' in _vp_df_src.columns
                and 'totalTradedVolume_CE' in _vp_df_src.columns
                and 'totalTradedVolume_PE' in _vp_df_src.columns
                and 'lastPrice_CE' in _vp_df_src.columns
                and 'lastPrice_PE' in _vp_df_src.columns):
            try:
                _vp_atm = _vp_df_src[_vp_df_src['Zone'] == 'ATM'].index
                if len(_vp_atm) > 0:
                    _vp_atm_pos = _vp_df_src.index.get_loc(_vp_atm[0])
                    _vp_start = max(0, _vp_atm_pos - 2)
                    _vp_end   = min(len(_vp_df_src), _vp_atm_pos + 3)
                    _vp_slice = _vp_df_src.iloc[_vp_start:_vp_end].copy()

                    _ist_vp = pytz.timezone('Asia/Kolkata')
                    _vp_now = datetime.now(_ist_vp)

                    # Per-strike Volume PCR entry
                    _vp_entry = {'time': _vp_now}
                    for _, _vrow in _vp_slice.iterrows():
                        _sk = str(int(_vrow['Strike']))
                        _ce_vol = _vrow.get('totalTradedVolume_CE', 0) or 0
                        _pe_vol = _vrow.get('totalTradedVolume_PE', 0) or 0
                        _vp_entry[_sk] = round(_pe_vol / _ce_vol, 3) if _ce_vol > 0 else 0.0
                    st.session_state.vol_pcr_current_strikes = sorted([int(_r['Strike']) for _, _r in _vp_slice.iterrows()])

                    # Straddle entry for all ATM±2 strikes (CE LTP + PE LTP per strike)
                    _st_entry = {'time': _vp_now}
                    for _, _srow in _vp_slice.iterrows():
                        _sk_int = int(_srow['Strike'])
                        _ce_ltp = float(_srow.get('lastPrice_CE', 0) or 0)
                        _pe_ltp = float(_srow.get('lastPrice_PE', 0) or 0)
                        _st_entry[f'straddle_{_sk_int}'] = round(_ce_ltp + _pe_ltp, 2)
                        _st_entry[f'ce_{_sk_int}']       = round(_ce_ltp, 2)
                        _st_entry[f'pe_{_sk_int}']       = round(_pe_ltp, 2)
                    # Keep ATM straddle as 'straddle' for Combined Signal
                    _atm_row = _vp_slice[_vp_slice['Zone'] == 'ATM']
                    if len(_atm_row) > 0:
                        _atm_sk = int(_atm_row.iloc[0]['Strike'])
                        _st_entry['straddle']   = _st_entry.get(f'straddle_{_atm_sk}', 0)
                        _st_entry['atm_strike'] = _atm_sk

                    # Avoid duplicates within 30 seconds
                    _vp_add = True
                    if st.session_state.vol_pcr_history:
                        if (_vp_now - st.session_state.vol_pcr_history[-1]['time']).total_seconds() < 30:
                            _vp_add = False
                    if _vp_add:
                        st.session_state.vol_pcr_history.append(_vp_entry)
                        if len(st.session_state.vol_pcr_history) > 200:
                            st.session_state.vol_pcr_history = st.session_state.vol_pcr_history[-200:]
                        if len(_st_entry) > 1:
                            st.session_state.straddle_history.append(_st_entry)
                            if len(st.session_state.straddle_history) > 200:
                                st.session_state.straddle_history = st.session_state.straddle_history[-200:]
                    _vp_data_ok = True
            except Exception as _vp_exc:
                st.caption(f"⚠️ Vol PCR fetch: {str(_vp_exc)[:60]}")

        # ── Panel 1: Volume PCR per strike ──
        st.markdown("### 📈 Volume PCR per Strike (ATM ± 2)")
        _vp_hist = st.session_state.vol_pcr_history
        _vp_strikes = sorted(getattr(st.session_state, 'vol_pcr_current_strikes', []))
        _pos_labels = ['🟣 ITM-2', '🟣 ITM-1', '🟡 ATM', '🔵 OTM+1', '🔵 OTM+2']
        _pos_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']

        if _vp_hist:
            _vp_hist_df = pd.DataFrame(_vp_hist)
            _vp_cols = st.columns(5)
            for _vi, _vcol in enumerate(_vp_cols):
                with _vcol:
                    if _vi >= len(_vp_strikes):
                        st.info(f"{_pos_labels[_vi]} N/A")
                        continue
                    _vstrike = _vp_strikes[_vi]
                    _vscol = str(_vstrike)
                    _vclr = _pos_colors[_vi]
                    _vrgb = tuple(int(_vclr.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
                    _vfill = f'rgba({_vrgb[0]},{_vrgb[1]},{_vrgb[2]},0.12)'
                    if _vscol not in _vp_hist_df.columns:
                        st.info(f"₹{_vstrike} — building…")
                        continue
                    _vcur = _vp_hist_df[_vscol].iloc[-1]
                    _vfig = go.Figure()
                    _vfig.add_trace(go.Scatter(
                        x=_vp_hist_df['time'], y=_vp_hist_df[_vscol],
                        mode='lines+markers', name='Vol PCR',
                        line=dict(color=_vclr, width=2),
                        marker=dict(size=3),
                        fill='tozeroy', fillcolor=_vfill,
                        hovertemplate='Vol PCR: %{y:.2f}<br>%{x|%H:%M}<extra></extra>',
                    ))
                    _vfig.add_hline(y=1.2, line_dash="dot", line_color="rgba(0,255,136,0.5)", line_width=1,
                                    annotation_text="Bull 1.2", annotation_position="right",
                                    annotation_font_size=8, annotation_font_color="#00ff88")
                    _vfig.add_hline(y=0.7, line_dash="dot", line_color="rgba(255,68,68,0.5)", line_width=1,
                                    annotation_text="Bear 0.7", annotation_position="right",
                                    annotation_font_size=8, annotation_font_color="#ff4444")
                    _vfig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
                    # Dynamic Y range: data + thresholds always visible
                    _vp_raw = _vp_hist_df[_vscol].dropna().tolist()
                    _vp_all = _vp_raw + [0.7, 1.2]
                    _vp_ymin = max(0.0, min(_vp_all) * 0.9)
                    _vp_ymax = max(_vp_all) * 1.1
                    _vfig.update_layout(
                        title=dict(text=f"{_pos_labels[_vi]}<br>₹{_vstrike}<br>Vol PCR: {_vcur:.2f}", font=dict(size=11)),
                        template='plotly_dark', height=280, showlegend=False,
                        margin=dict(l=5, r=10, t=70, b=30),
                        xaxis=dict(tickformat='%H:%M', title=''),
                        yaxis=dict(title='Vol PCR', range=[_vp_ymin, _vp_ymax]),
                        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                    )
                    st.plotly_chart(_vfig, use_container_width=True)
                    _vsig = "🟢 Bull" if _vcur > 1.2 else ("🔴 Bear" if _vcur < 0.7 else "🟡 Ntrl")
                    st.caption(f"Vol PCR {_vcur:.2f} {_vsig}")
        else:
            st.info("📊 Volume PCR history building… wait for a few refreshes.")

        # ── Panel 2: Straddle per strike (ATM±2) ──
        st.markdown("---")
        st.markdown("### 🎯 Straddle (CE LTP + PE LTP) — Movement Intensity (ATM ± 2)")
        _st_hist = st.session_state.straddle_history
        _st_strikes = sorted(getattr(st.session_state, 'vol_pcr_current_strikes', []))
        _st_color_map = {'rising': '#ff9944', 'falling': '#44aaff', 'flat': '#888888'}
        if _st_hist and _st_strikes:
            _st_df = pd.DataFrame(_st_hist)
            _st_cols = st.columns(5)
            for _si, _scol in enumerate(_st_cols):
                with _scol:
                    if _si >= len(_st_strikes):
                        st.info(f"{_pos_labels[_si]} N/A")
                        continue
                    _sstrike = _st_strikes[_si]
                    _skey = f'straddle_{_sstrike}'
                    _cekey = f'ce_{_sstrike}'
                    _pekey = f'pe_{_sstrike}'
                    if _skey not in _st_df.columns:
                        st.info(f"₹{_sstrike} — building…")
                        continue
                    _st_vals = _st_df[_skey].dropna().tolist()
                    _st_cur  = _st_vals[-1] if _st_vals else 0
                    if len(_st_vals) >= 3:
                        _st_delta = _st_vals[-1] - _st_vals[-3]
                        _st_dir = "rising" if _st_delta > 0.5 else ("falling" if _st_delta < -0.5 else "flat")
                    else:
                        _st_delta, _st_dir = 0, "flat"
                    _st_clr = _st_color_map[_st_dir]
                    _st_rgb = tuple(int(_st_clr.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
                    _st_icon = "📈" if _st_dir == "rising" else ("📉" if _st_dir == "falling" else "➡️")

                    _sfig = go.Figure()
                    _sfig.add_trace(go.Scatter(
                        x=_st_df['time'], y=_st_df[_skey],
                        mode='lines+markers', name='Straddle',
                        line=dict(color=_st_clr, width=2), marker=dict(size=3),
                        fill='tozeroy', fillcolor=f'rgba({_st_rgb[0]},{_st_rgb[1]},{_st_rgb[2]},0.15)',
                        hovertemplate='Straddle: ₹%{y:.2f}<br>%{x|%H:%M}<extra></extra>',
                    ))
                    if _cekey in _st_df.columns:
                        _sfig.add_trace(go.Scatter(
                            x=_st_df['time'], y=_st_df[_cekey],
                            mode='lines', name='CE',
                            line=dict(color='#ff4444', width=1, dash='dot'),
                            hovertemplate='CE: ₹%{y:.2f}<br>%{x|%H:%M}<extra></extra>',
                        ))
                    if _pekey in _st_df.columns:
                        _sfig.add_trace(go.Scatter(
                            x=_st_df['time'], y=_st_df[_pekey],
                            mode='lines', name='PE',
                            line=dict(color='#00cc66', width=1, dash='dot'),
                            hovertemplate='PE: ₹%{y:.2f}<br>%{x|%H:%M}<extra></extra>',
                        ))
                    # Dynamic Y range: all 3 traces (Straddle, CE, PE) combined
                    _sy_vals = _st_df[_skey].dropna().tolist()
                    if _cekey in _st_df.columns:
                        _sy_vals += _st_df[_cekey].dropna().tolist()
                    if _pekey in _st_df.columns:
                        _sy_vals += _st_df[_pekey].dropna().tolist()
                    _sy_min = max(0.0, min(_sy_vals) * 0.9) if _sy_vals else 0
                    _sy_max = max(_sy_vals) * 1.1 if _sy_vals else 10
                    _sfig.update_layout(
                        title=dict(text=f"{_pos_labels[_si]}<br>₹{_sstrike}<br>{_st_icon} ₹{_st_cur:.1f} (Δ{_st_delta:+.1f})",
                                   font=dict(size=11)),
                        template='plotly_dark', height=300,
                        showlegend=True,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                    xanchor='center', x=0.5, font=dict(size=8)),
                        margin=dict(l=5, r=10, t=75, b=30),
                        xaxis=dict(tickformat='%H:%M', title=''),
                        yaxis=dict(title='Premium (₹)', range=[_sy_min, _sy_max]),
                        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                    )
                    st.plotly_chart(_sfig, use_container_width=True)
                    _move_lbl = "💥 Explosive" if _st_dir == "rising" else ("🐌 Grinding" if _st_dir == "falling" else "⬛ Range")
                    st.caption(f"₹{_st_cur:.1f} → {_move_lbl}")
        else:
            st.info("📊 Straddle history building… wait for a few refreshes.")

        # ── Panel 3: Combined Trend / Strength / Move Type Signal ──
        st.markdown("---")
        st.markdown("### 🧠 Combined Signal — Trend · Strength · Move Type · Confidence")
        try:
            _cs_oi_hist   = st.session_state.pcr_history
            _cs_vol_hist  = st.session_state.vol_pcr_history
            _cs_gex_hist  = st.session_state.gex_history
            _cs_st_hist   = st.session_state.straddle_history
            _cs_strikes   = sorted(getattr(st.session_state, 'vol_pcr_current_strikes', []))

            _cs_ready = (_cs_oi_hist and _cs_vol_hist and len(_cs_strikes) > 0)
            if _cs_ready:
                _cs_oi_df  = pd.DataFrame(_cs_oi_hist)
                _cs_vol_df = pd.DataFrame(_cs_vol_hist)

                # ── Avg OI PCR + Vol PCR across ATM±2 ──
                _cs_oi_vals  = [_cs_oi_df[str(s)].iloc[-1]  for s in _cs_strikes if str(s) in _cs_oi_df.columns]
                _cs_vol_vals = [_cs_vol_df[str(s)].iloc[-1] for s in _cs_strikes if str(s) in _cs_vol_df.columns]
                _avg_oi_pcr  = sum(_cs_oi_vals)  / len(_cs_oi_vals)  if _cs_oi_vals  else 1.0
                _avg_vol_pcr = sum(_cs_vol_vals) / len(_cs_vol_vals) if _cs_vol_vals else 1.0
                _oi_sig  = "bull" if _avg_oi_pcr  > 1.2 else ("bear" if _avg_oi_pcr  < 0.7 else "neut")
                _vol_sig = "bull" if _avg_vol_pcr > 1.2 else ("bear" if _avg_vol_pcr < 0.7 else "neut")

                # ── Trap Detector ──
                if _oi_sig == "bull" and _vol_sig == "bear":
                    _trap = "🚨 BULL TRAP"
                    _trap_clr = "#ff6600"
                    _trap_sub = "OI bull but Vol bear — smart money exiting"
                elif _oi_sig == "bear" and _vol_sig == "bull":
                    _trap = "🚨 BEAR TRAP"
                    _trap_clr = "#ffcc00"
                    _trap_sub = "OI bear but Vol bull — absorption / reversal watch"
                else:
                    _trap = "✅ Clean Signal"
                    _trap_clr = "#44cc88"
                    _trap_sub = "OI PCR & Vol PCR aligned"

                # ── Trend ──
                if _oi_sig == "bull" and _vol_sig == "bull":
                    _trend = "🟢 Strong Bull"
                    _trend_clr = "#00ff88"
                elif _oi_sig == "bear" and _vol_sig == "bear":
                    _trend = "🔴 Strong Bear"
                    _trend_clr = "#ff4444"
                elif _oi_sig == "bull":
                    _trend = "🟢 Bull (positioning)"
                    _trend_clr = "#00cc88"
                elif _oi_sig == "bear":
                    _trend = "🔴 Bear (positioning)"
                    _trend_clr = "#dd4444"
                else:
                    _trend = "⚪ Neutral"
                    _trend_clr = "#888888"

                # ── Strength from GEX ──
                _cs_gex_df = pd.DataFrame(_cs_gex_hist) if _cs_gex_hist else None
                _cs_gex_cols = [str(s) for s in _cs_strikes if _cs_gex_df is not None and str(s) in _cs_gex_df.columns]
                _total_gex = sum(_cs_gex_df[c].iloc[-1] for c in _cs_gex_cols) if _cs_gex_cols and _cs_gex_df is not None else 0
                if _total_gex < -10:
                    _strength = "⚡ Strong (Neg GEX)"
                    _strength_clr = "#ff9900"
                    _gex_mode = "trending"
                elif _total_gex > 10:
                    _strength = "📍 Capped (Pos GEX)"
                    _strength_clr = "#aaaaaa"
                    _gex_mode = "pinning"
                else:
                    _strength = "➡️ Normal"
                    _strength_clr = "#cccccc"
                    _gex_mode = "neutral"

                # ── Move Type: avg straddle delta across ALL 5 strikes ──
                _cs_st_df = pd.DataFrame(_cs_st_hist) if _cs_st_hist else None
                _st_deltas = []
                _strike_dirs = {}    # per-strike direction for pattern detection
                _strike_cur  = {}    # per-strike current straddle value
                if _cs_st_df is not None:
                    for _s in _cs_strikes:
                        _sk = f'straddle_{_s}'
                        if _sk in _cs_st_df.columns:
                            _sv = _cs_st_df[_sk].dropna().tolist()
                            _strike_cur[_s] = _sv[-1] if _sv else 0
                            if len(_sv) >= 3:
                                _d = _sv[-1] - _sv[-3]
                                _st_deltas.append(_d)
                                _strike_dirs[_s] = "rising" if _d > 0.5 else ("falling" if _d < -0.5 else "flat")
                            else:
                                _strike_dirs[_s] = "flat"

                _avg_st_delta = sum(_st_deltas) / len(_st_deltas) if _st_deltas else 0
                _cs_st_dir = "rising" if _avg_st_delta > 0.5 else ("falling" if _avg_st_delta < -0.5 else "flat")
                _move_map = {'rising': "💥 Explosive", 'falling': "🐌 Grinding", 'flat': "⬛ Range"}
                _move_type_s = _move_map[_cs_st_dir]
                _move_clr = '#ff9944' if _cs_st_dir == 'rising' else ('#44aaff' if _cs_st_dir == 'falling' else '#888888')

                # ── Straddle Pattern (case 1-4) ──
                _n_rising  = sum(1 for d in _strike_dirs.values() if d == "rising")
                _n_falling = sum(1 for d in _strike_dirs.values() if d == "falling")
                _itm_s = _cs_strikes[:2]
                _otm_s = _cs_strikes[3:]
                _atm_s = _cs_strikes[2] if len(_cs_strikes) > 2 else None
                _itm_rising = all(_strike_dirs.get(s, "flat") == "rising" for s in _itm_s)
                _otm_flat   = all(_strike_dirs.get(s, "flat") in ("flat", "falling") for s in _otm_s)
                _atm_rising = _strike_dirs.get(_atm_s, "flat") == "rising" if _atm_s else False
                _atm_only   = _atm_rising and all(_strike_dirs.get(s, "flat") != "rising" for s in _cs_strikes if s != _atm_s)

                if _n_rising >= 4:
                    _pattern = "🔥 All Rising"
                    _pattern_desc = "BIG MOVE COMING"
                    _pattern_clr = "#ff4444"
                elif _atm_only:
                    _pattern = "⚠️ ATM Only Rising"
                    _pattern_desc = "Fake / noise — wait"
                    _pattern_clr = "#ff9900"
                elif _itm_rising and _otm_flat:
                    _pattern = "📐 ITM ↑ / OTM Flat"
                    _pattern_desc = "Directional move starting"
                    _pattern_clr = "#00ccff"
                elif _n_falling >= 4:
                    _pattern = "💤 All Falling"
                    _pattern_desc = "Sideways / premium decay"
                    _pattern_clr = "#44aaff"
                else:
                    _pattern = "🔀 Mixed"
                    _pattern_desc = f"{_n_rising}↑ {_n_falling}↓"
                    _pattern_clr = "#cccccc"

                # ── Straddle Spread: OTM avg − ITM avg ──
                _itm_cur = [_strike_cur.get(s, 0) for s in _itm_s if s in _strike_cur]
                _otm_cur = [_strike_cur.get(s, 0) for s in _otm_s if s in _strike_cur]
                _itm_avg = sum(_itm_cur) / len(_itm_cur) if _itm_cur else 0
                _otm_avg = sum(_otm_cur) / len(_otm_cur) if _otm_cur else 0
                _spread  = _otm_avg - _itm_avg
                if _spread > 5:
                    _spread_sig = "⬆️ Upside Expansion"
                    _spread_clr = "#00ff88"
                elif _spread < -5:
                    _spread_sig = "⬇️ Downside Pressure"
                    _spread_clr = "#ff4444"
                else:
                    _spread_sig = "↔️ Balanced"
                    _spread_clr = "#888888"

                # ── Direction Confidence Score ──
                _conf = 0
                # PCR alignment (max 3)
                if _oi_sig == _vol_sig and _oi_sig != "neut":
                    _conf += 3
                elif _oi_sig != "neut" and _vol_sig != "neut":
                    _conf += 1
                # GEX alignment with trend (max 2)
                if (("bull" in _oi_sig or "bull" in _vol_sig) and _gex_mode == "trending") or \
                   (("bear" in _oi_sig or "bear" in _vol_sig) and _gex_mode == "trending"):
                    _conf += 2
                elif _gex_mode == "pinning":
                    _conf += 0
                else:
                    _conf += 1
                # Straddle confirmation (max 2)
                if (_cs_st_dir == "rising" and "bull" in _oi_sig) or (_cs_st_dir == "falling" and "bear" in _oi_sig):
                    _conf += 2
                elif _cs_st_dir == "flat":
                    _conf += 0
                # Spread confirmation (max 1)
                if (_spread > 5 and "bull" in _oi_sig) or (_spread < -5 and "bear" in _oi_sig):
                    _conf += 1
                # No trap bonus (max 1)
                if _trap.startswith("✅"):
                    _conf += 1
                _conf_pct = min(int((_conf / 9) * 100), 100)
                _conf_clr = "#00ff88" if _conf_pct >= 65 else ("#ff9900" if _conf_pct >= 35 else "#ff4444")

                # ── Verdict ──
                if "Trap" in _trap:
                    _interp = "⚠️ CAUTION / TRAP"
                    _interp_clr = "#ff6600"
                elif "Bull" in _trend and "Explosive" in _move_type_s and _gex_mode == "trending":
                    _interp = "🚀 BREAKOUT UP"
                    _interp_clr = "#00ff88"
                elif "Bear" in _trend and "Explosive" in _move_type_s and _gex_mode == "trending":
                    _interp = "💥 BREAKDOWN"
                    _interp_clr = "#ff4444"
                elif "Bull" in _trend and "Grinding" in _move_type_s:
                    _interp = "📈 SLOW GRIND UP"
                    _interp_clr = "#88ff88"
                elif "Bear" in _trend and "Grinding" in _move_type_s:
                    _interp = "📉 SLOW GRIND DOWN"
                    _interp_clr = "#ff8888"
                elif _gex_mode == "pinning" or _cs_st_dir == "flat":
                    _interp = "⬛ RANGE BOUND"
                    _interp_clr = "#888888"
                else:
                    _interp = "➡️ WAIT / UNCLEAR"
                    _interp_clr = "#aaaaaa"

                # ── Row 1: 5 signal cards ──
                _sig_c1, _sig_c2, _sig_c3, _sig_c4, _sig_c5 = st.columns(5)
                def _card(clr, label, value, sub):
                    return f"""<div style="background:#1e1e1e;border:1.5px solid {clr};border-radius:8px;padding:14px;text-align:center;min-height:110px">
                        <div style="color:#888;font-size:10px;margin-bottom:4px;letter-spacing:1px">{label}</div>
                        <div style="color:{clr};font-size:15px;font-weight:bold;line-height:1.3">{value}</div>
                        <div style="color:#999;font-size:10px;margin-top:6px">{sub}</div>
                    </div>"""
                with _sig_c1:
                    st.markdown(_card(_trap_clr, "🚨 TRAP DETECTOR", _trap, _trap_sub), unsafe_allow_html=True)
                with _sig_c2:
                    st.markdown(_card(_trend_clr, "📊 TREND", _trend, f"OI {_avg_oi_pcr:.2f} · Vol {_avg_vol_pcr:.2f}"), unsafe_allow_html=True)
                with _sig_c3:
                    st.markdown(_card(_strength_clr, "⚙️ STRENGTH", _strength, f"GEX {_total_gex:+.1f}L"), unsafe_allow_html=True)
                with _sig_c4:
                    st.markdown(_card(_move_clr, "🎯 MOVE TYPE", _move_type_s, f"5-strike avg Δ {_avg_st_delta:+.1f}"), unsafe_allow_html=True)
                with _sig_c5:
                    st.markdown(_card(_interp_clr, "🏆 VERDICT", _interp, "OI+Vol+GEX+Straddle"), unsafe_allow_html=True)

                # ── Row 2: Straddle Pattern | Spread | Confidence ──
                st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
                _row2_c1, _row2_c2, _row2_c3 = st.columns(3)
                with _row2_c1:
                    st.markdown(_card(_pattern_clr, "🧩 STRADDLE PATTERN", _pattern, _pattern_desc), unsafe_allow_html=True)
                with _row2_c2:
                    st.markdown(_card(_spread_clr, "📐 STRADDLE SPREAD", f"₹{_spread:+.1f}",
                                      f"{_spread_sig} | OTM {_otm_avg:.1f} vs ITM {_itm_avg:.1f}"), unsafe_allow_html=True)
                with _row2_c3:
                    _bar_filled = int(_conf_pct / 10)
                    _bar = "█" * _bar_filled + "░" * (10 - _bar_filled)
                    st.markdown(_card(_conf_clr, "🎯 CONFIDENCE", f"{_conf_pct}%", f"{_bar}"), unsafe_allow_html=True)

            else:
                st.info("📊 Combined signal will appear once history builds for Volume PCR. Wait a few refreshes.")
        except Exception as _cs_exc:
            st.warning(f"Combined signal error: {str(_cs_exc)[:80]}")

        # ── Clear buttons ──
        _vp_c1, _vp_c2 = st.columns([3, 1])
        with _vp_c1:
            _vp_status = "🟢 Live" if _vp_data_ok else "🟡 Cached"
            st.caption(f"{_vp_status} | Vol PCR: {len(_vp_hist)} pts · Straddle: {len(_st_hist)} pts")
        with _vp_c2:
            if st.button("🗑️ Clear Vol/Straddle History"):
                st.session_state.vol_pcr_history = []
                st.session_state.straddle_history = []
                st.rerun()

        # ===== GEX (GAMMA EXPOSURE) ANALYSIS SECTION =====
        st.markdown("---")
        st.markdown("## 📊 Gamma Exposure (GEX) Analysis - Dealer Hedging Flow")

        gex_data = gex_data_pre  # already computed above; avoids duplicate API/calculation
        try:
            underlying_price = option_data.get('underlying')

            if gex_data:
                # nesting preserved from original structure
                if True:
                    gex_df = gex_data['gex_df']

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

                    # ===== GEX PER-STRIKE TABLE (time-series charts shown in comparison view above) =====
                    try:
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

                        st.caption("GEX > 10 = Pin Zone | GEX < -10 = Acceleration Zone | "
                                   "Time-series charts shown in comparison view above")

                    except Exception as e:
                        st.warning(f"Error displaying GEX charts: {str(e)}")

                else:
                    st.warning("Unable to calculate GEX. Check option chain data.")

        except Exception as e:
            st.warning(f"GEX analysis unavailable: {str(e)}")

        # ===== PCR OF TOTAL CHANGE IN OI - TIME SERIES GRAPH =====
        st.markdown("---")
        st.markdown("## 📊 PCR of Total Change in OI - Time Series")

        try:
            df_summary_pcr = option_data.get('df_summary') if option_data else None
            if df_summary_pcr is not None and 'changeinOpenInterest_CE' in df_summary_pcr.columns and 'changeinOpenInterest_PE' in df_summary_pcr.columns:
                total_ce_chgoi = df_summary_pcr['changeinOpenInterest_CE'].sum()
                total_pe_chgoi = df_summary_pcr['changeinOpenInterest_PE'].sum()

                # PCR = Total PE ChgOI / Total CE ChgOI
                if total_ce_chgoi != 0:
                    pcr_chgoi = abs(total_pe_chgoi / total_ce_chgoi)
                else:
                    pcr_chgoi = 0

                pcr_chgoi = round(pcr_chgoi, 3)

                # Cache in session state
                ist = pytz.timezone('Asia/Kolkata')
                current_time = datetime.now(ist)

                st.session_state.pcr_chgoi_last_valid = {
                    'pcr': pcr_chgoi,
                    'ce_chgoi': total_ce_chgoi,
                    'pe_chgoi': total_pe_chgoi,
                    'time': current_time
                }

                # Add to history (avoid duplicates within 30 seconds)
                should_add = True
                if st.session_state.pcr_chgoi_history:
                    last_entry = st.session_state.pcr_chgoi_history[-1]
                    time_diff = (current_time - last_entry['time']).total_seconds()
                    if time_diff < 30:
                        should_add = False

                if should_add:
                    st.session_state.pcr_chgoi_history.append({
                        'time': current_time,
                        'pcr': pcr_chgoi,
                        'ce_chgoi': total_ce_chgoi,
                        'pe_chgoi': total_pe_chgoi
                    })
                    if len(st.session_state.pcr_chgoi_history) > 200:
                        st.session_state.pcr_chgoi_history = st.session_state.pcr_chgoi_history[-200:]

            # Display graph from cached history
            if len(st.session_state.pcr_chgoi_history) > 0:
                pcr_chgoi_df = pd.DataFrame(st.session_state.pcr_chgoi_history)

                # Summary cards
                latest = st.session_state.pcr_chgoi_last_valid or {}
                curr_pcr = latest.get('pcr', 0)
                curr_ce = latest.get('ce_chgoi', 0)
                curr_pe = latest.get('pe_chgoi', 0)

                pcr_card1, pcr_card2, pcr_card3 = st.columns(3)
                with pcr_card1:
                    pcr_color = "#00ff88" if curr_pcr > 1.2 else "#ff4444" if curr_pcr < 0.7 else "#FFD700"
                    pcr_label = "Bullish" if curr_pcr > 1.2 else "Bearish" if curr_pcr < 0.7 else "Neutral"
                    st.markdown(f"""
                    <div style="background-color: {pcr_color}20; padding: 15px; border-radius: 10px; border: 2px solid {pcr_color};">
                        <h4 style="color: {pcr_color}; margin: 0;">PCR (ΔOI)</h4>
                        <h2 style="color: {pcr_color}; margin: 5px 0;">{curr_pcr:.3f}</h2>
                        <p style="color: white; margin: 0; font-size: 12px;">{pcr_label}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with pcr_card2:
                    ce_color = "#ff4444" if curr_ce > 0 else "#00ff88"
                    st.markdown(f"""
                    <div style="background-color: {ce_color}20; padding: 15px; border-radius: 10px; border: 2px solid {ce_color};">
                        <h4 style="color: {ce_color}; margin: 0;">Total CE ΔOI</h4>
                        <h2 style="color: {ce_color}; margin: 5px 0;">{curr_ce/100000:+.2f}L</h2>
                        <p style="color: white; margin: 0; font-size: 12px;">{'Call Writing ↑' if curr_ce > 0 else 'Call Unwinding ↓'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with pcr_card3:
                    pe_color = "#00ff88" if curr_pe > 0 else "#ff4444"
                    st.markdown(f"""
                    <div style="background-color: {pe_color}20; padding: 15px; border-radius: 10px; border: 2px solid {pe_color};">
                        <h4 style="color: {pe_color}; margin: 0;">Total PE ΔOI</h4>
                        <h2 style="color: {pe_color}; margin: 5px 0;">{curr_pe/100000:+.2f}L</h2>
                        <p style="color: white; margin: 0; font-size: 12px;">{'Put Writing ↑' if curr_pe > 0 else 'Put Unwinding ↓'}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # PCR of ChgOI Time-Series Chart
                fig_pcr_chgoi = go.Figure()

                # Color the line based on PCR zones
                fig_pcr_chgoi.add_trace(go.Scatter(
                    x=pcr_chgoi_df['time'],
                    y=pcr_chgoi_df['pcr'],
                    mode='lines+markers',
                    name='PCR (ΔOI)',
                    line=dict(color='#00aaff', width=3),
                    marker=dict(size=6, color=[
                        '#00ff88' if v > 1.2 else '#ff4444' if v < 0.7 else '#FFD700'
                        for v in pcr_chgoi_df['pcr']
                    ]),
                    fill='tozeroy',
                    fillcolor='rgba(0, 170, 255, 0.1)'
                ))

                # Reference zones
                fig_pcr_chgoi.add_hline(y=1.0, line_dash="dash", line_color="white", line_width=1,
                                        annotation_text="1.0 (Neutral)", annotation_position="right")
                fig_pcr_chgoi.add_hline(y=1.2, line_dash="dot", line_color="#00ff88", line_width=1,
                                        annotation_text="1.2 (Bullish)", annotation_position="right")
                fig_pcr_chgoi.add_hline(y=0.7, line_dash="dot", line_color="#ff4444", line_width=1,
                                        annotation_text="0.7 (Bearish)", annotation_position="right")

                # Dynamic Y range: data + reference thresholds always in view
                _ov_raw = pcr_chgoi_df['pcr'].dropna().tolist()
                _ov_all = _ov_raw + [0.7, 1.0, 1.2]
                _ov_ymin = max(0.0, min(_ov_all) * 0.9)
                _ov_ymax = max(_ov_all) * 1.1

                # Add green/red shading zones
                fig_pcr_chgoi.add_hrect(y0=1.2, y1=_ov_ymax, fillcolor="rgba(0,255,136,0.06)", line_width=0)
                fig_pcr_chgoi.add_hrect(y0=_ov_ymin, y1=0.7, fillcolor="rgba(255,68,68,0.06)", line_width=0)

                fig_pcr_chgoi.update_layout(
                    title=f"PCR of Total Change in OI | Current: {curr_pcr:.3f} ({pcr_label})",
                    template='plotly_dark',
                    height=400,
                    showlegend=False,
                    xaxis=dict(tickformat='%H:%M', title='Time'),
                    yaxis=dict(title='PCR (Total PE ΔOI / Total CE ΔOI)', range=[_ov_ymin, _ov_ymax]),
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    margin=dict(l=50, r=50, t=60, b=50)
                )

                st.plotly_chart(fig_pcr_chgoi, use_container_width=True)

                # Status bar
                pcr_info1, pcr_info2 = st.columns([3, 1])
                with pcr_info1:
                    st.caption(f"🟢 Live | 📈 {len(st.session_state.pcr_chgoi_history)} data points | PCR > 1.2 = Bullish | PCR < 0.7 = Bearish")
                with pcr_info2:
                    if st.button("🗑️ Clear PCR ΔOI History"):
                        st.session_state.pcr_chgoi_history = []
                        st.session_state.pcr_chgoi_last_valid = None
                        st.rerun()
            else:
                st.info("📊 PCR (ΔOI) history will build up as the app refreshes. Please wait for data collection...")

        except Exception as e:
            st.warning(f"PCR of ΔOI analysis unavailable: {str(e)}")

        # ===== PCR OF CHANGE IN OI - TIME SERIES PER ATM ± 2 STRIKES =====
        st.markdown("---")
        st.markdown("## 📊 PCR of Change in OI - Time Series (ATM ± 2)")

        # Helper function to create per-strike ChgOI PCR chart
        def create_pcr_chgoi_strike_chart(history_df, col_name, color, title_prefix):
            """Helper to create individual PCR of Change in OI chart per strike"""
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

                # Reference lines
                fig.add_hline(y=1.0, line_dash="dash", line_color="white", line_width=1)
                fig.add_hline(y=1.2, line_dash="dot", line_color="#00ff88", line_width=1)
                fig.add_hline(y=0.7, line_dash="dot", line_color="#ff4444", line_width=1)

                # Get current value
                current_val = history_df[col_name].iloc[-1] if len(history_df) > 0 else 0

                # Dynamic Y range: data + reference thresholds always in view
                _chgoi_raw = history_df[col_name].dropna().tolist()
                _chgoi_all = _chgoi_raw + [0.7, 1.2]
                _chgoi_ymin = max(0.0, min(_chgoi_all) * 0.9)
                _chgoi_ymax = max(_chgoi_all) * 1.1

                fig.update_layout(
                    title=f"{title_prefix}<br>₹{strike_val}<br>PCR(ΔOI): {current_val:.2f}",
                    template='plotly_dark',
                    height=280,
                    showlegend=False,
                    margin=dict(l=10, r=10, t=70, b=30),
                    xaxis=dict(tickformat='%H:%M', title=''),
                    yaxis=dict(title='PCR (ΔOI)', range=[_chgoi_ymin, _chgoi_ymax]),
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e'
                )
                return fig, current_val
            return None, 0

        # Try to get new data and add to history
        pcr_chgoi_strike_available = False
        pcr_chgoi_strike_df = None

        df_summary_chgoi = option_data.get('df_summary') if option_data else None
        if df_summary_chgoi is not None and 'Zone' in df_summary_chgoi.columns and 'changeinOpenInterest_CE' in df_summary_chgoi.columns and 'changeinOpenInterest_PE' in df_summary_chgoi.columns:
            try:
                # Find ATM index
                atm_idx_chgoi = df_summary_chgoi[df_summary_chgoi['Zone'] == 'ATM'].index
                if len(atm_idx_chgoi) > 0:
                    atm_pos_chgoi = df_summary_chgoi.index.get_loc(atm_idx_chgoi[0])

                    # Get ATM ± 2 strikes (5 strikes total)
                    start_idx_chgoi = max(0, atm_pos_chgoi - 2)
                    end_idx_chgoi = min(len(df_summary_chgoi), atm_pos_chgoi + 3)

                    pcr_chgoi_strike_df = df_summary_chgoi.iloc[start_idx_chgoi:end_idx_chgoi][['Strike', 'Zone',
                                                                   'changeinOpenInterest_CE', 'changeinOpenInterest_PE']].copy()

                    # Calculate PCR of Change in OI per strike
                    pcr_chgoi_strike_df['PCR_ChgOI'] = pcr_chgoi_strike_df.apply(
                        lambda row: abs(row['changeinOpenInterest_PE'] / row['changeinOpenInterest_CE'])
                        if row['changeinOpenInterest_CE'] != 0 else 0, axis=1
                    )
                    pcr_chgoi_strike_df['PCR_ChgOI'] = pcr_chgoi_strike_df['PCR_ChgOI'].round(3)
                    pcr_chgoi_strike_df['PCR_ChgOI_Signal'] = np.where(
                        pcr_chgoi_strike_df['PCR_ChgOI'] > 1.2, "Bullish",
                        np.where(pcr_chgoi_strike_df['PCR_ChgOI'] < 0.7, "Bearish", "Neutral")
                    )

                    if not pcr_chgoi_strike_df.empty:
                        pcr_chgoi_strike_available = True
                        st.session_state.pcr_chgoi_strike_last_valid = pcr_chgoi_strike_df.copy()

                        ist = pytz.timezone('Asia/Kolkata')
                        current_time = datetime.now(ist)

                        # Build history entry keyed by strike price
                        chgoi_entry = {'time': current_time}
                        for _, row in pcr_chgoi_strike_df.iterrows():
                            strike_label = str(int(row['Strike']))
                            chgoi_entry[strike_label] = row['PCR_ChgOI']

                        # Store current strikes
                        current_chgoi_strikes = pcr_chgoi_strike_df['Strike'].tolist()
                        st.session_state.pcr_chgoi_strike_current_strikes = [int(s) for s in current_chgoi_strikes]

                        # Deduplicate within 30 seconds
                        should_add = True
                        if st.session_state.pcr_chgoi_strike_history:
                            last_entry = st.session_state.pcr_chgoi_strike_history[-1]
                            time_diff = (current_time - last_entry['time']).total_seconds()
                            if time_diff < 30:
                                should_add = False

                        if should_add:
                            st.session_state.pcr_chgoi_strike_history.append(chgoi_entry)
                            if len(st.session_state.pcr_chgoi_strike_history) > 200:
                                st.session_state.pcr_chgoi_strike_history = st.session_state.pcr_chgoi_strike_history[-200:]

            except Exception as e:
                st.caption(f"⚠️ Current fetch issue: {str(e)[:50]}...")

        # ALWAYS try to display graph if we have history
        if len(st.session_state.pcr_chgoi_strike_history) > 0:
            try:
                chgoi_history_df = pd.DataFrame(st.session_state.pcr_chgoi_strike_history)

                current_chgoi_strikes = getattr(st.session_state, 'pcr_chgoi_strike_current_strikes', [])

                if not current_chgoi_strikes and st.session_state.pcr_chgoi_strike_last_valid is not None:
                    current_chgoi_strikes = [int(s) for s in st.session_state.pcr_chgoi_strike_last_valid['Strike'].tolist()]

                current_chgoi_strikes = sorted(current_chgoi_strikes)

                # 5 columns for side-by-side display
                chgoi_col1, chgoi_col2, chgoi_col3, chgoi_col4, chgoi_col5 = st.columns(5)

                def display_pcr_chgoi_with_signal(container, fig, pcr_val):
                    if fig:
                        container.plotly_chart(fig, use_container_width=True)
                        if pcr_val > 1.2:
                            container.success("Bullish")
                        elif pcr_val < 0.7:
                            container.error("Bearish")
                        else:
                            container.warning("Neutral")

                # Get zone info
                chgoi_zone_info = {}
                chgoi_zone_df = pcr_chgoi_strike_df if pcr_chgoi_strike_df is not None else st.session_state.pcr_chgoi_strike_last_valid
                if chgoi_zone_df is not None:
                    for _, row in chgoi_zone_df.iterrows():
                        chgoi_zone_info[int(row['Strike'])] = row['Zone']

                position_labels = ['🟣 ITM-2', '🟣 ITM-1', '🟡 ATM', '🔵 OTM+1', '🔵 OTM+2']
                position_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']
                chgoi_columns = [chgoi_col1, chgoi_col2, chgoi_col3, chgoi_col4, chgoi_col5]

                for i, col in enumerate(chgoi_columns):
                    with col:
                        if i < len(current_chgoi_strikes):
                            strike = current_chgoi_strikes[i]
                            strike_col = str(strike)
                            zone = chgoi_zone_info.get(strike, position_labels[i].split()[-1])

                            if strike_col in chgoi_history_df.columns:
                                fig, pcr_val = create_pcr_chgoi_strike_chart(chgoi_history_df, strike_col, position_colors[i], f'{position_labels[i]}')
                                display_pcr_chgoi_with_signal(st, fig, pcr_val)
                            else:
                                st.info(f"₹{strike} - Building history...")
                        else:
                            st.info(f"{position_labels[i]} N/A")

                # Show current data table
                st.markdown("### Current PCR of Change in OI Values")
                display_chgoi_df = pcr_chgoi_strike_df if pcr_chgoi_strike_df is not None else st.session_state.pcr_chgoi_strike_last_valid
                if display_chgoi_df is not None:
                    chgoi_display = display_chgoi_df[['Strike', 'Zone', 'PCR_ChgOI', 'PCR_ChgOI_Signal']].copy()
                    chgoi_display['CE ΔOI (L)'] = (display_chgoi_df['changeinOpenInterest_CE'] / 100000).round(2)
                    chgoi_display['PE ΔOI (L)'] = (display_chgoi_df['changeinOpenInterest_PE'] / 100000).round(2)
                    chgoi_display.rename(columns={'PCR_ChgOI': 'PCR (ΔOI)', 'PCR_ChgOI_Signal': 'Signal'}, inplace=True)
                    st.dataframe(chgoi_display, use_container_width=True, hide_index=True)

                # Status bar and clear button
                chgoi_info1, chgoi_info2 = st.columns([3, 1])
                with chgoi_info1:
                    status = "🟢 Live" if pcr_chgoi_strike_available else "🟡 Using cached history"
                    st.caption(f"{status} | 📈 {len(st.session_state.pcr_chgoi_strike_history)} data points | History preserved on refresh failures")
                with chgoi_info2:
                    if st.button("🗑️ Clear ΔOI Strike History"):
                        st.session_state.pcr_chgoi_strike_history = []
                        st.session_state.pcr_chgoi_strike_last_valid = None
                        st.rerun()

            except Exception as e:
                st.warning(f"Error displaying PCR ΔOI strike charts: {str(e)}")
        else:
            st.info("📊 PCR of ΔOI per strike history will build up as the app refreshes. Please wait for data collection...")

        # ===== COMPOSITE DIRECTION SIGNAL — now merged into Unified Sentiment Engine above =====
        # Computation runs inside Unified Options Flow Sentiment Engine; session state is shared.
        # Displaying the verdict & per-strike charts here for historical time-series view.
        st.markdown("---")
        st.markdown("## 🧭 Composite Direction Signal — Time Series (ATM ± 2)")

        try:
            # Computation runs inside Unified Options Flow Sentiment Engine (above).
            # This section shows the composite time-series history from session state.
            last_valid = st.session_state.composite_signal_last_valid
            if last_valid:
                _cv2_verdict = last_valid['verdict']
                _cv2_icon    = last_valid['verdict_icon']
                _cv2_color   = last_valid['verdict_color']
                _cv2_desc    = last_valid['verdict_desc']
                _cv2_score   = last_valid['score_pct']
                _cv2_gex     = last_valid['total_net_gex']
                _cv2_pcr     = last_valid['avg_pcr']
                _cv2_chgoi   = last_valid['avg_chgoi']
                _cv2_max     = 14.0
                _cv2_gtrend  = _cv2_gex < -10
                _cv2_gpin    = _cv2_gex > 10
                strike_details = last_valid.get('strike_details', [])

                # Verdict card
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{_cv2_color}15,{_cv2_color}30);
                            padding:20px;border-radius:12px;border:3px solid {_cv2_color};
                            text-align:center;margin-bottom:14px;">
                    <h1 style="color:{_cv2_color};margin:0;font-size:42px;">{_cv2_icon} {_cv2_verdict}</h1>
                    <p style="color:#ccc;margin:8px 0 0 0;font-size:15px;">{_cv2_desc}</p>
                    <p style="color:{_cv2_color};margin:5px 0 0 0;font-size:13px;">
                        Score: {_cv2_score:+.0f}% | GEX: {_cv2_gex:.1f}L
                        ({'Trending' if _cv2_gtrend else 'Pinning' if _cv2_gpin else 'Neutral'})
                    </p>
                </div>""", unsafe_allow_html=True)

                # 3 metric cards
                _cm1x, _cm2x, _cm3x = st.columns(3)
                with _cm1x:
                    _pc2 = "#00ff88" if _cv2_pcr > 1.2 else "#ff4444" if _cv2_pcr < 0.7 else "#FFD700"
                    st.markdown(f'''<div style="background:{_pc2}20;padding:12px;border-radius:8px;
                        border:2px solid {_pc2};text-align:center;">
                        <h4 style="color:{_pc2};margin:0;">Avg PCR (OI)</h4>
                        <h2 style="color:{_pc2};margin:5px 0;">{_cv2_pcr:.2f}</h2>
                        <p style="color:white;margin:0;font-size:12px;">
                            {'Bullish' if _cv2_pcr > 1.2 else 'Bearish' if _cv2_pcr < 0.7 else 'Neutral'}
                        </p></div>''', unsafe_allow_html=True)
                with _cm2x:
                    _cc2 = "#00ff88" if _cv2_chgoi > 1.2 else "#ff4444" if _cv2_chgoi < 0.7 else "#FFD700"
                    st.markdown(f'''<div style="background:{_cc2}20;padding:12px;border-radius:8px;
                        border:2px solid {_cc2};text-align:center;">
                        <h4 style="color:{_cc2};margin:0;">Avg PCR (ΔOI)</h4>
                        <h2 style="color:{_cc2};margin:5px 0;">{_cv2_chgoi:.2f}</h2>
                        <p style="color:white;margin:0;font-size:12px;">
                            {'Bullish' if _cv2_chgoi > 1.2 else 'Bearish' if _cv2_chgoi < 0.7 else 'Neutral'}
                        </p></div>''', unsafe_allow_html=True)
                with _cm3x:
                    _gc2 = "#00ff88" if _cv2_gex > 10 else "#ff4444" if _cv2_gex < -10 else "#FFD700"
                    st.markdown(f'''<div style="background:{_gc2}20;padding:12px;border-radius:8px;
                        border:2px solid {_gc2};text-align:center;">
                        <h4 style="color:{_gc2};margin:0;">Total GEX (ATM±2)</h4>
                        <h2 style="color:{_gc2};margin:5px 0;">{_cv2_gex:.1f}L</h2>
                        <p style="color:white;margin:0;font-size:12px;">
                            {'Pin/Chop' if _cv2_gex > 10 else 'Trend/Accel' if _cv2_gex < -10 else 'Neutral'}
                        </p></div>''', unsafe_allow_html=True)

            # Time-series charts from shared session state
            if len(st.session_state.composite_signal_history) > 0:
                comp_hist_df = pd.DataFrame(st.session_state.composite_signal_history)

                # Chart 1: Composite Score over time
                fig_score = go.Figure()
                _mkr_colors2 = []
                for _, _hrow2 in comp_hist_df.iterrows():
                    _vn2 = _hrow2.get('verdict_numeric', 0)
                    _mkr_colors2.append(
                        '#00ff88' if _vn2 >= 2 else '#90EE90' if _vn2 == 1 else
                        '#ff4444' if _vn2 <= -2 else '#FFB6C1' if _vn2 == -1 else '#FFD700')
                fig_score.add_trace(go.Scatter(
                    x=comp_hist_df['time'], y=comp_hist_df['score_pct'],
                    mode='lines+markers', name='Score %',
                    line=dict(color='#00aaff', width=3),
                    marker=dict(size=8, color=_mkr_colors2),
                    fill='tozeroy', fillcolor='rgba(0,170,255,0.08)'))
                fig_score.add_hline(y=0, line_dash='dash', line_color='white', line_width=1.5,
                    annotation_text='Neutral (0%)', annotation_position='right')
                fig_score.add_hline(y=15, line_dash='dot', line_color='#00ff88', line_width=1,
                    annotation_text='Bullish Zone', annotation_position='right')
                fig_score.add_hline(y=-15, line_dash='dot', line_color='#ff4444', line_width=1,
                    annotation_text='Bearish Zone', annotation_position='right')
                _ymax2 = max(abs(comp_hist_df['score_pct'].max()), abs(comp_hist_df['score_pct'].min()), 30) * 1.2
                fig_score.add_hrect(y0=15, y1=_ymax2, fillcolor='rgba(0,255,136,0.06)', line_width=0)
                fig_score.add_hrect(y0=-_ymax2, y1=-15, fillcolor='rgba(255,68,68,0.06)', line_width=0)
                _cv2_s_now = last_valid['score_pct'] if last_valid else 0
                _cv2_v_now = last_valid['verdict']    if last_valid else 'NEUTRAL'
                fig_score.update_layout(
                    title=f"Composite Direction Score | Current: {_cv2_s_now:+.0f}% ({_cv2_v_now})",
                    template='plotly_dark', height=380, showlegend=False,
                    xaxis=dict(tickformat='%H:%M', title='Time'),
                    yaxis=dict(title='Score %', zeroline=True, zerolinecolor='white'),
                    plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                    margin=dict(l=50, r=50, t=60, b=50))
                st.plotly_chart(fig_score, use_container_width=True)

                # Charts: PCR components + GEX (2 col)
                ind_col1, ind_col2 = st.columns(2)
                with ind_col1:
                    fig_pcr_ts = go.Figure()
                    fig_pcr_ts.add_trace(go.Scatter(
                        x=comp_hist_df['time'], y=comp_hist_df['avg_pcr'],
                        mode='lines+markers', name='Avg PCR (OI)',
                        line=dict(color='#00aaff', width=2), marker=dict(size=4)))
                    fig_pcr_ts.add_trace(go.Scatter(
                        x=comp_hist_df['time'], y=comp_hist_df['avg_chgoi'],
                        mode='lines+markers', name='Avg PCR (ΔOI)',
                        line=dict(color='#ff44ff', width=2), marker=dict(size=4)))
                    fig_pcr_ts.add_hline(y=1.2, line_dash='dot', line_color='#00ff88', line_width=1)
                    fig_pcr_ts.add_hline(y=1.0, line_dash='dash', line_color='white', line_width=1)
                    fig_pcr_ts.add_hline(y=0.7, line_dash='dot', line_color='#ff4444', line_width=1)
                    _all_pcr = (comp_hist_df['avg_pcr'].dropna().tolist() +
                                comp_hist_df['avg_chgoi'].dropna().tolist() + [0.7, 1.0, 1.2])
                    fig_pcr_ts.update_layout(
                        title='Avg PCR (OI) vs Avg PCR (ΔOI)',
                        template='plotly_dark', height=300, showlegend=True,
                        legend=dict(orientation="h", y=-0.3, font=dict(size=9)),
                        xaxis=dict(tickformat='%H:%M'),
                        yaxis=dict(title='PCR',
                                   range=[max(0, min(_all_pcr)*0.9), max(_all_pcr)*1.1]),
                        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                        margin=dict(l=40, r=10, t=50, b=30))
                    st.plotly_chart(fig_pcr_ts, use_container_width=True)
                with ind_col2:
                    fig_gex_ts = go.Figure()
                    _gex_c2 = ['#00ff88' if g > 10 else '#ff4444' if g < -10 else '#FFD700'
                               for g in comp_hist_df['total_gex']]
                    fig_gex_ts.add_trace(go.Scatter(
                        x=comp_hist_df['time'], y=comp_hist_df['total_gex'],
                        mode='lines+markers', name='Total GEX',
                        line=dict(color='#FFD700', width=2),
                        marker=dict(size=5, color=_gex_c2),
                        fill='tozeroy', fillcolor='rgba(255,215,0,0.08)'))
                    fig_gex_ts.add_hline(y=0, line_dash='dash', line_color='white', line_width=1)
                    fig_gex_ts.add_hline(y=10, line_dash='dot', line_color='#00ff88', line_width=1,
                        annotation_text='Pin Zone', annotation_position='right')
                    fig_gex_ts.add_hline(y=-10, line_dash='dot', line_color='#ff4444', line_width=1,
                        annotation_text='Accel Zone', annotation_position='right')
                    fig_gex_ts.update_layout(
                        title='Total GEX (ATM±2) Over Time',
                        template='plotly_dark', height=300, showlegend=False,
                        xaxis=dict(tickformat='%H:%M'),
                        yaxis=dict(title='GEX (Lakhs)', zeroline=True, zerolinecolor='white'),
                        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                        margin=dict(l=40, r=50, t=50, b=30))
                    st.plotly_chart(fig_gex_ts, use_container_width=True)

                # Per-strike score time series
                score_cols = [c for c in comp_hist_df.columns if c.startswith('score_')]
                if score_cols:
                    fig_strike_ts = go.Figure()
                    _cs_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']
                    _plbls = ['ITM-2', 'ITM-1', 'ATM', 'OTM+1', 'OTM+2']
                    for _idx_s, _sc2 in enumerate(sorted(score_cols)):
                        _sn = _sc2.replace('score_', '')
                        _slbl3 = _plbls[_idx_s] if _idx_s < len(_plbls) else _sn
                        fig_strike_ts.add_trace(go.Scatter(
                            x=comp_hist_df['time'], y=comp_hist_df[_sc2],
                            mode='lines+markers',
                            name=f'{_slbl3} (₹{_sn})',
                            line=dict(color=_cs_colors[_idx_s % len(_cs_colors)], width=2),
                            marker=dict(size=3)))
                    fig_strike_ts.add_hline(y=0, line_dash='dash', line_color='white', line_width=1)
                    fig_strike_ts.update_layout(
                        title='Per-Strike Weighted Score Over Time',
                        template='plotly_dark', height=320, showlegend=True,
                        legend=dict(orientation='h', y=-0.3, font=dict(size=9)),
                        xaxis=dict(tickformat='%H:%M'),
                        yaxis=dict(title='Weighted Score', zeroline=True, zerolinecolor='white'),
                        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                        margin=dict(l=50, r=20, t=50, b=40))
                    st.plotly_chart(fig_strike_ts, use_container_width=True)

                # Per-strike breakdown table
                if last_valid and last_valid.get('strike_details'):
                    st.markdown("### Per-Strike Breakdown")
                    st.dataframe(pd.DataFrame(last_valid['strike_details']),
                                 use_container_width=True, hide_index=True)

                # Status / clear
                comp_info1, comp_info2 = st.columns([3, 1])
                with comp_info1:
                    st.caption(f"📈 {len(st.session_state.composite_signal_history)} pts · "
                               f"Computed inside Unified Sentiment Engine · Updates every ~30s")
                with comp_info2:
                    if st.button("🗑️ Clear Composite History", key="clr_comp_ts"):
                        st.session_state.composite_signal_history = []
                        st.session_state.composite_signal_last_valid = None
                        st.rerun()

            elif not st.session_state.composite_signal_last_valid:
                st.info("📊 Composite signal history will build up as the app refreshes.")

        except Exception as e:
            st.warning(f"Composite direction signal unavailable: {str(e)}")

        # ===== TOTAL GEX TIME-SERIES GRAPH =====
        st.markdown("---")
        st.markdown("## 📊 Total GEX (Gamma Exposure) - Time Series")

        try:
            df_summary_gex = option_data.get('df_summary') if option_data else None
            underlying_price_gex = option_data.get('underlying') if option_data else None

            if df_summary_gex is not None and underlying_price_gex:
                gex_calc = calculate_dealer_gex(df_summary_gex, underlying_price_gex)
                if gex_calc:
                    total_gex_val = gex_calc['total_gex']
                    gex_signal_val = gex_calc['gex_signal']
                    gex_color_val = gex_calc['gex_color']
                    gex_interp_val = gex_calc['gex_interpretation']
                    flip_level = gex_calc.get('gamma_flip_level')

                    # Cache in session state
                    ist = pytz.timezone('Asia/Kolkata')
                    current_time = datetime.now(ist)

                    st.session_state.total_gex_last_valid = {
                        'total_gex': total_gex_val,
                        'signal': gex_signal_val,
                        'color': gex_color_val,
                        'interpretation': gex_interp_val,
                        'flip_level': flip_level,
                        'time': current_time
                    }

                    # Add to history
                    should_add = True
                    if st.session_state.total_gex_history:
                        last_entry = st.session_state.total_gex_history[-1]
                        time_diff = (current_time - last_entry['time']).total_seconds()
                        if time_diff < 30:
                            should_add = False

                    if should_add:
                        st.session_state.total_gex_history.append({
                            'time': current_time,
                            'total_gex': total_gex_val,
                            'signal': gex_signal_val,
                            'flip_level': flip_level
                        })
                        if len(st.session_state.total_gex_history) > 200:
                            st.session_state.total_gex_history = st.session_state.total_gex_history[-200:]

            # Display graph from cached history
            if len(st.session_state.total_gex_history) > 0:
                gex_ts_df = pd.DataFrame(st.session_state.total_gex_history)

                latest_gex = st.session_state.total_gex_last_valid or {}
                curr_gex = latest_gex.get('total_gex', 0)
                curr_signal = latest_gex.get('signal', 'N/A')
                curr_gex_color = latest_gex.get('color', '#FFD700')
                curr_interp = latest_gex.get('interpretation', 'N/A')

                # Total GEX Time-Series Chart
                fig_total_gex = go.Figure()

                # Color markers based on positive/negative
                marker_colors = ['#00ff88' if v >= 0 else '#ff4444' for v in gex_ts_df['total_gex']]

                fig_total_gex.add_trace(go.Scatter(
                    x=gex_ts_df['time'],
                    y=gex_ts_df['total_gex'],
                    mode='lines+markers',
                    name='Total GEX',
                    line=dict(color=curr_gex_color, width=3),
                    marker=dict(size=6, color=marker_colors),
                    fill='tozeroy',
                    fillcolor='rgba(0,255,136,0.08)' if curr_gex >= 0 else 'rgba(255,68,68,0.08)'
                ))

                # Zero reference line (critical flip boundary)
                fig_total_gex.add_hline(y=0, line_dash="solid", line_color="white", line_width=2,
                                        annotation_text="0 (Gamma Flip)", annotation_position="right")

                # Threshold lines
                fig_total_gex.add_hline(y=50, line_dash="dot", line_color="#00ff88", line_width=1,
                                        annotation_text="+50 (Strong Pin)", annotation_position="right")
                fig_total_gex.add_hline(y=-50, line_dash="dot", line_color="#ff4444", line_width=1,
                                        annotation_text="-50 (Strong Trend)", annotation_position="right")

                # Shading zones
                y_max_gex = max(abs(gex_ts_df['total_gex'].max()), abs(gex_ts_df['total_gex'].min()), 60) * 1.2
                fig_total_gex.add_hrect(y0=50, y1=y_max_gex, fillcolor="rgba(0,255,136,0.06)", line_width=0)
                fig_total_gex.add_hrect(y0=-y_max_gex, y1=-50, fillcolor="rgba(255,68,68,0.06)", line_width=0)

                fig_total_gex.update_layout(
                    title=f"Total Net GEX | Current: {curr_gex:+.2f}L ({curr_signal})",
                    template='plotly_dark',
                    height=400,
                    showlegend=False,
                    xaxis=dict(tickformat='%H:%M', title='Time'),
                    yaxis=dict(
                        title='Total Net GEX (Lakhs)',
                        zeroline=True,
                        zerolinecolor='white',
                        zerolinewidth=2
                    ),
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    margin=dict(l=50, r=50, t=60, b=50)
                )

                st.plotly_chart(fig_total_gex, use_container_width=True)

                # Interpretation box
                st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 4px solid {curr_gex_color}; margin: 10px 0;">
                    <b style="color: {curr_gex_color};">Current Regime:</b> {curr_interp}
                </div>
                """, unsafe_allow_html=True)

                # Status bar
                gex_ts_info1, gex_ts_info2 = st.columns([3, 1])
                with gex_ts_info1:
                    st.caption(f"🟢 Live | 📈 {len(st.session_state.total_gex_history)} data points | +GEX = Pin/Chop | -GEX = Trend/Breakout")
                with gex_ts_info2:
                    if st.button("🗑️ Clear Total GEX History"):
                        st.session_state.total_gex_history = []
                        st.session_state.total_gex_last_valid = None
                        st.rerun()
            else:
                st.info("📊 Total GEX history will build up as the app refreshes. Please wait for data collection...")

        except Exception as e:
            st.warning(f"Total GEX time-series unavailable: {str(e)}")

        # ===== GAMMA SEQUENCE ANALYSIS =====
        st.markdown("---")
        st.markdown("## 📊 Gamma Sequence Analysis")

        try:
            df_summary_gs = option_data.get('df_summary') if option_data else None
            underlying_price_gs = option_data.get('underlying') if option_data else None

            if df_summary_gs is not None and underlying_price_gs:
                gamma_seq = calculate_gamma_sequence(df_summary_gs, underlying_price_gs)

                if gamma_seq:
                    gs_df = gamma_seq['gamma_seq_df']

                    # Summary cards
                    gs_col1, gs_col2, gs_col3 = st.columns(3)
                    with gs_col1:
                        pcolor = gamma_seq['profile_color']
                        st.markdown(f"""
                        <div style="background-color: {pcolor}20; padding: 15px; border-radius: 10px; border: 2px solid {pcolor};">
                            <h4 style="color: {pcolor}; margin: 0;">Gamma Profile</h4>
                            <h2 style="color: {pcolor}; margin: 5px 0; font-size: 16px;">{gamma_seq['gamma_profile']}</h2>
                            <p style="color: white; margin: 0; font-size: 12px;">Above: {gamma_seq['above_pct']}% | Below: {gamma_seq['below_pct']}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with gs_col2:
                        st.markdown(f"""
                        <div style="background-color: #FFD70020; padding: 15px; border-radius: 10px; border: 2px solid #FFD700;">
                            <h4 style="color: #FFD700; margin: 0;">Peak Gamma</h4>
                            <h2 style="color: #FFD700; margin: 5px 0;">₹{gamma_seq['peak_gamma_strike']:.0f}</h2>
                            <p style="color: white; margin: 0; font-size: 12px;">Highest total gamma exposure</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with gs_col3:
                        st.markdown(f"""
                        <div style="background-color: #00aaff20; padding: 15px; border-radius: 10px; border: 2px solid #00aaff;">
                            <h4 style="color: #00aaff; margin: 0;">Total Gamma</h4>
                            <h2 style="color: #00aaff; margin: 5px 0;">{gamma_seq['total_gamma']:.2f}L</h2>
                            <p style="color: white; margin: 0; font-size: 12px;">Sum of all strike gamma</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Gamma Sequence Chart - Stacked bar (CE vs PE gamma per strike)
                    fig_gs = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=('CE vs PE Gamma Exposure by Strike', 'Cumulative Net Gamma Sequence'),
                        row_heights=[0.5, 0.5]
                    )

                    # Top: CE vs PE gamma bars
                    fig_gs.add_trace(go.Bar(
                        x=gs_df['Strike'],
                        y=gs_df['CE_Gamma_Exp'],
                        name='CE Gamma',
                        marker_color='#ff4444',
                        text=[f"{v:.1f}" for v in gs_df['CE_Gamma_Exp']],
                        textposition='outside',
                        textfont=dict(size=9)
                    ), row=1, col=1)

                    fig_gs.add_trace(go.Bar(
                        x=gs_df['Strike'],
                        y=gs_df['PE_Gamma_Exp'],
                        name='PE Gamma',
                        marker_color='#00ff88',
                        text=[f"{v:.1f}" for v in gs_df['PE_Gamma_Exp']],
                        textposition='outside',
                        textfont=dict(size=9)
                    ), row=1, col=1)

                    # Bottom: Cumulative net gamma line
                    fig_gs.add_trace(go.Scatter(
                        x=gs_df['Strike'],
                        y=gs_df['Cumul_Net_Gamma'],
                        mode='lines+markers',
                        name='Cumul Net Gamma',
                        line=dict(color='#00aaff', width=3),
                        marker=dict(size=8, color=[
                            '#00ff88' if v >= 0 else '#ff4444' for v in gs_df['Cumul_Net_Gamma']
                        ]),
                        fill='tozeroy',
                        fillcolor='rgba(0, 170, 255, 0.1)'
                    ), row=2, col=1)

                    fig_gs.add_hline(y=0, line_dash="dash", line_color="white", line_width=1, row=2, col=1)

                    # Spot price vertical line on both subplots
                    fig_gs.add_vline(
                        x=underlying_price_gs,
                        line_dash="solid",
                        line_color="#FFD700",
                        line_width=2,
                        annotation_text=f"Spot: ₹{underlying_price_gs:.0f}",
                        annotation_position="top",
                        row=1, col=1
                    )
                    fig_gs.add_vline(
                        x=underlying_price_gs,
                        line_dash="solid",
                        line_color="#FFD700",
                        line_width=2,
                        row=2, col=1
                    )

                    fig_gs.update_layout(
                        template='plotly_dark',
                        height=600,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        plot_bgcolor='#1e1e1e',
                        paper_bgcolor='#1e1e1e',
                        margin=dict(l=50, r=50, t=80, b=50),
                        barmode='group'
                    )

                    fig_gs.update_yaxes(title_text="Gamma Exp (L)", row=1, col=1)
                    fig_gs.update_yaxes(title_text="Cumul Net Gamma (L)", row=2, col=1)
                    fig_gs.update_xaxes(title_text="Strike Price", row=2, col=1)

                    st.plotly_chart(fig_gs, use_container_width=True)

                    # Gamma sequence table
                    with st.expander("📋 Gamma Sequence Data"):
                        gs_display = gs_df[['Strike', 'Zone', 'CE_Gamma_Exp', 'PE_Gamma_Exp', 'Net_Gamma',
                                            'Cumul_Net_Gamma', 'Gamma_Accel']].copy()
                        gs_display['Strike'] = gs_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                        st.dataframe(gs_display, use_container_width=True, hide_index=True)

                        st.markdown("""
                        **Gamma Sequence Interpretation:**
                        - **CE Gamma (Red)**: Call gamma exposure per strike - higher = stronger resistance
                        - **PE Gamma (Green)**: Put gamma exposure per strike - higher = stronger support
                        - **Cumul Net Gamma**: Running sum of net gamma from lowest strike upward
                        - **Gamma Accel**: Rate of change in cumulative gamma - spikes show gamma walls
                        - **Peak Gamma**: Strike with highest total gamma - strongest hedging activity
                        """)
                else:
                    st.warning("Unable to calculate gamma sequence. Check option chain data.")
            else:
                st.info("Gamma sequence requires option chain data.")

        except Exception as e:
            st.warning(f"Gamma sequence unavailable: {str(e)}")

        # ===== IV SKEW & OPTIONS PRESSURE SIGNALS =====
        st.markdown("---")
        st.markdown("## 📡 IV Skew & Options Pressure Signals (ATM ± 2)")

        try:
            _ivp_df = option_data.get('df_summary') if option_data else None
            _ivp_underlying = option_data.get('underlying') if option_data else None

            # Ensure optional bid/ask columns exist (may be absent if Dhan didn't return them)
            if _ivp_df is not None:
                for _col in ['impliedVolatility_CE', 'impliedVolatility_PE',
                             'bidQty_CE', 'bidQty_PE', 'askQty_CE', 'askQty_PE']:
                    if _col not in _ivp_df.columns:
                        _ivp_df[_col] = 0.0

            if (_ivp_df is not None and _ivp_underlying and 'Zone' in _ivp_df.columns):

                # --- ATM ± 2 slice ---
                _ivp_atm_idx = _ivp_df[_ivp_df['Zone'] == 'ATM'].index
                if len(_ivp_atm_idx) > 0:
                    _ivp_atm_pos = _ivp_df.index.get_loc(_ivp_atm_idx[0])
                    _ivp_start = max(0, _ivp_atm_pos - 2)
                    _ivp_end = min(len(_ivp_df), _ivp_atm_pos + 3)
                    _ivp_slice = _ivp_df.iloc[_ivp_start:_ivp_end].copy()

                    # --- Strike label function for IV section ---
                    _ivp_strikes_sorted = sorted(_ivp_slice['Strike'].unique())
                    _ivp_step = int(_ivp_strikes_sorted[1] - _ivp_strikes_sorted[0]) if len(_ivp_strikes_sorted) >= 2 else 50
                    _ivp_atm_val = float(_ivp_df[_ivp_df['Zone'] == 'ATM']['Strike'].values[0])
                    def _ivp_label(s):
                        diff = int(round((s - _ivp_atm_val) / _ivp_step))
                        if diff == 0:  return "ATM"
                        if diff > 0:   return f"ATM+{diff}"
                        return f"ATM{diff}"

                    # --- Per-strike IV (CE and PE) ---
                    _ivp_per_iv_ce = {}
                    _ivp_per_iv_pe = {}
                    for _, _ivr in _ivp_slice.iterrows():
                        _ivs = _ivr['Strike']
                        _ivlbl = _ivp_label(_ivs)
                        _ivp_per_iv_ce[_ivlbl] = float(_ivr.get('impliedVolatility_CE', 0) or 0)
                        _ivp_per_iv_pe[_ivlbl] = float(_ivr.get('impliedVolatility_PE', 0) or 0)

                    # --- IV SKEW ---
                    _iv_ce_vals = _ivp_slice['impliedVolatility_CE'].fillna(0).tolist()
                    _iv_pe_vals = _ivp_slice['impliedVolatility_PE'].fillna(0).tolist()
                    _avg_iv_ce = sum(_iv_ce_vals) / len(_iv_ce_vals) if _iv_ce_vals else 0
                    _avg_iv_pe = sum(_iv_pe_vals) / len(_iv_pe_vals) if _iv_pe_vals else 0
                    _iv_skew = _avg_iv_pe / (_avg_iv_ce + 1e-6)

                    if _iv_skew < 0.90:
                        _iv_signal = "🟢 Bullish Expectation"
                        _iv_signal_color = "#00C853"
                    elif _iv_skew > 1.10:
                        _iv_signal = "🔴 Bearish Expectation"
                        _iv_signal_color = "#FF5252"
                    else:
                        _iv_signal = "🟡 Neutral"
                        _iv_signal_color = "#FFD740"

                    # IV Skew momentum (vs last saved)
                    _prev_iv_skew = st.session_state.iv_skew_history[-1]['iv_skew'] if st.session_state.iv_skew_history else _iv_skew
                    _iv_skew_change = _iv_skew - _prev_iv_skew
                    if _iv_skew_change < -0.02:
                        _iv_momentum = "⬆️ Bullish Building"
                    elif _iv_skew_change > 0.02:
                        _iv_momentum = "⬇️ Bearish Building"
                    else:
                        _iv_momentum = "➡️ Stable"

                    # --- CALL & PUT PRESSURE ---
                    _call_pres = {}
                    _put_pres = {}
                    _ivp_per_net_pres = {}  # per-strike net pressure
                    for _, _row in _ivp_slice.iterrows():
                        _s = _row['Strike']
                        _slbl = _ivp_label(_s)
                        _bc = float(_row.get('bidQty_CE', 0) or 0)
                        _ac = float(_row.get('askQty_CE', 0) or 0)
                        _bp = float(_row.get('bidQty_PE', 0) or 0)
                        _ap = float(_row.get('askQty_PE', 0) or 0)
                        _cp = (_bc - _ac) / (_bc + _ac + 1e-6)
                        _pp = (_bp - _ap) / (_bp + _ap + 1e-6)
                        _call_pres[_s] = _cp
                        _put_pres[_s]  = _pp
                        _ivp_per_net_pres[_slbl] = round(_cp - _pp, 4)

                    _avg_call_pres = sum(_call_pres.values()) / len(_call_pres) if _call_pres else 0
                    _avg_put_pres  = sum(_put_pres.values())  / len(_put_pres)  if _put_pres  else 0
                    _net_pressure  = _avg_call_pres - _avg_put_pres

                    if _net_pressure > 0.15:
                        _net_signal = "🟢 Bullish Pressure"
                        _net_color = "#00C853"
                    elif _net_pressure < -0.15:
                        _net_signal = "🔴 Bearish Pressure"
                        _net_color = "#FF5252"
                    else:
                        _net_signal = "🟡 Neutral"
                        _net_color = "#FFD740"

                    if _avg_call_pres > 0.2 and _avg_put_pres < -0.1:
                        _pressure_signal = "🚀 Strong Bullish"
                    elif _avg_put_pres > 0.2 and _avg_call_pres < -0.1:
                        _pressure_signal = "🔥 Strong Bearish"
                    elif _avg_call_pres > 0.2 and _avg_put_pres > 0.2:
                        _pressure_signal = "⚠️ High Volatility"
                    else:
                        _pressure_signal = "⏳ Sideways"

                    # --- COMBINED FINAL SIGNAL ---
                    if _iv_skew < 0.90 and _net_pressure > 0.15:
                        _final_signal = "🚀 STRONG BUY (Confirmed Breakout)"
                        _final_color = "#00C853"
                    elif _iv_skew > 1.10 and _net_pressure < -0.15:
                        _final_signal = "🔥 STRONG SELL (Confirmed Breakdown)"
                        _final_color = "#FF5252"
                    elif _iv_skew < 0.90 and _net_pressure < 0:
                        _final_signal = "⚠️ Bull Trap"
                        _final_color = "#FF9800"
                    elif _iv_skew > 1.10 and _net_pressure > 0:
                        _final_signal = "⚠️ Bear Trap"
                        _final_color = "#FF9800"
                    else:
                        _final_signal = "⏳ No Clear Edge"
                        _final_color = "#888888"

                    # --- PRESSURE SPIKE (Entry Timing) ---
                    _prev_net_pres = st.session_state.pressure_history[-1]['net_pressure'] if st.session_state.pressure_history else _net_pressure
                    _pressure_change = _net_pressure - _prev_net_pres
                    _spike_alert = "⚡ Sudden Aggression — Entry Signal!" if abs(_pressure_change) > 0.15 else ""

                    # --- Save to history ---
                    _ivp_now = datetime.now(pytz.timezone('Asia/Kolkata'))

                    _should_append_iv = (
                        not st.session_state.iv_skew_history or
                        (_ivp_now - st.session_state.iv_skew_history[-1]['time']).total_seconds() >= 30
                    )
                    if _should_append_iv:
                        st.session_state.iv_skew_history.append({
                            'time': _ivp_now,
                            'iv_skew': round(_iv_skew, 4),
                            'avg_iv_ce': round(_avg_iv_ce, 2),
                            'avg_iv_pe': round(_avg_iv_pe, 2),
                            **{f"iv_ce_{k}": round(v, 2) for k, v in _ivp_per_iv_ce.items()},
                            **{f"iv_pe_{k}": round(v, 2) for k, v in _ivp_per_iv_pe.items()},
                            **{f"pres_{k}": v for k, v in _ivp_per_net_pres.items()},
                        })
                        if len(st.session_state.iv_skew_history) > 200:
                            st.session_state.iv_skew_history = st.session_state.iv_skew_history[-200:]

                        st.session_state.pressure_history.append({
                            'time': _ivp_now,
                            'call_pressure': round(_avg_call_pres, 4),
                            'put_pressure': round(_avg_put_pres, 4),
                            'net_pressure': round(_net_pressure, 4),
                        })
                        if len(st.session_state.pressure_history) > 200:
                            st.session_state.pressure_history = st.session_state.pressure_history[-200:]

                    # ======== UI OUTPUT ========
                    _ivp_c1, _ivp_c2, _ivp_c3 = st.columns(3)
                    with _ivp_c1:
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:1px solid {_iv_signal_color};">
                        <div style="color:#aaa;font-size:12px;">IV SKEW (PE/CE avg)</div>
                        <div style="font-size:22px;font-weight:bold;color:{_iv_signal_color};">{_iv_skew:.3f}</div>
                        <div style="font-size:13px;margin-top:4px;">{_iv_signal}</div>
                        <div style="font-size:12px;color:#aaa;margin-top:2px;">Momentum: {_iv_momentum}</div>
                        <div style="font-size:11px;color:#888;margin-top:4px;">
                            CE avg IV: {_avg_iv_ce:.1f}% &nbsp;|&nbsp; PE avg IV: {_avg_iv_pe:.1f}%
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with _ivp_c2:
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:1px solid {_net_color};">
                        <div style="color:#aaa;font-size:12px;">PRESSURE (ATM ± 2)</div>
                        <div style="font-size:13px;margin-top:4px;">
                            Call: <b style="color:#00C853;">{_avg_call_pres:+.3f}</b> &nbsp;
                            Put: <b style="color:#FF5252;">{_avg_put_pres:+.3f}</b>
                        </div>
                        <div style="font-size:18px;font-weight:bold;color:{_net_color};margin-top:6px;">
                            Net: {_net_pressure:+.3f}
                        </div>
                        <div style="font-size:13px;margin-top:4px;">{_net_signal}</div>
                        <div style="font-size:12px;color:#aaa;margin-top:2px;">{_pressure_signal}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with _ivp_c3:
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:2px solid {_final_color};">
                        <div style="color:#aaa;font-size:12px;">COMBINED SIGNAL</div>
                        <div style="font-size:16px;font-weight:bold;color:{_final_color};margin-top:6px;">{_final_signal}</div>
                        {"<div style='font-size:13px;color:#FFD740;margin-top:6px;'>" + _spike_alert + "</div>" if _spike_alert else ""}
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ---- IV Time-Series Chart ----
                    if len(st.session_state.iv_skew_history) >= 2:
                        _iv_hist_df = pd.DataFrame(st.session_state.iv_skew_history)
                        _fig_iv = go.Figure()
                        _fig_iv.add_trace(go.Scatter(
                            x=_iv_hist_df['time'], y=_iv_hist_df['iv_skew'],
                            mode='lines+markers', name='IV Skew (PE/CE)',
                            line=dict(color='#FFD740', width=2),
                            marker=dict(size=4)
                        ))
                        _fig_iv.add_hline(y=0.90, line_dash='dash', line_color='#00C853',
                                          annotation_text='Bullish (0.90)', annotation_position='bottom right')
                        _fig_iv.add_hline(y=1.10, line_dash='dash', line_color='#FF5252',
                                          annotation_text='Bearish (1.10)', annotation_position='top right')
                        # Current value marker (present value shown in graph)
                        _iv_cur = _iv_hist_df['iv_skew'].iloc[-1]
                        _fig_iv.add_trace(go.Scatter(
                            x=[_iv_hist_df['time'].iloc[-1]], y=[_iv_cur],
                            mode='markers+text', text=[f'{_iv_cur:.3f}'],
                            textposition='top right', textfont=dict(size=10, color='#FFD740'),
                            marker=dict(size=10, color='#FFD740', symbol='circle'),
                            showlegend=False, hoverinfo='skip',
                        ))
                        _fig_iv.update_layout(
                            title=f'IV Skew Over Time (ATM ± 2) — Now: {_iv_cur:.3f}',
                            height=250, margin=dict(l=40, r=20, t=40, b=30),
                            paper_bgcolor='#111', plot_bgcolor='#111',
                            font=dict(color='#ccc'),
                            xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'),
                            showlegend=True, legend=dict(orientation='h', y=-0.3)
                        )
                        st.plotly_chart(_fig_iv, use_container_width=True)

                    # ---- Pressure Time-Series Chart ----
                    if len(st.session_state.pressure_history) >= 2:
                        _pr_hist_df = pd.DataFrame(st.session_state.pressure_history)
                        _fig_pr = go.Figure()
                        _fig_pr.add_trace(go.Scatter(
                            x=_pr_hist_df['time'], y=_pr_hist_df['call_pressure'],
                            mode='lines', name='Call Pressure',
                            line=dict(color='#00C853', width=2)
                        ))
                        _fig_pr.add_trace(go.Scatter(
                            x=_pr_hist_df['time'], y=_pr_hist_df['put_pressure'],
                            mode='lines', name='Put Pressure',
                            line=dict(color='#FF5252', width=2)
                        ))
                        _fig_pr.add_trace(go.Scatter(
                            x=_pr_hist_df['time'], y=_pr_hist_df['net_pressure'],
                            mode='lines+markers', name='Net Pressure',
                            line=dict(color='#FFD740', width=2, dash='dot'),
                            marker=dict(size=4)
                        ))
                        _fig_pr.add_hline(y=0.15, line_dash='dash', line_color='#00C853',
                                          annotation_text='+0.15 Bull', annotation_position='bottom right')
                        _fig_pr.add_hline(y=-0.15, line_dash='dash', line_color='#FF5252',
                                          annotation_text='-0.15 Bear', annotation_position='top right')
                        _fig_pr.add_hline(y=0, line_color='#555', line_width=1)
                        # Current value markers (present values shown in graph)
                        _pr_cur_call = _pr_hist_df['call_pressure'].iloc[-1]
                        _pr_cur_put  = _pr_hist_df['put_pressure'].iloc[-1]
                        _pr_cur_net  = _pr_hist_df['net_pressure'].iloc[-1]
                        _fig_pr.add_trace(go.Scatter(
                            x=[_pr_hist_df['time'].iloc[-1]], y=[_pr_cur_net],
                            mode='markers+text', text=[f'Net:{_pr_cur_net:+.3f}'],
                            textposition='top right', textfont=dict(size=9, color='#FFD740'),
                            marker=dict(size=9, color='#FFD740', symbol='circle'),
                            showlegend=False, hoverinfo='skip',
                        ))
                        _fig_pr.update_layout(
                            title=f'Call / Put / Net Pressure Over Time (ATM ± 2) — Net: {_pr_cur_net:+.3f}',
                            height=250, margin=dict(l=40, r=20, t=40, b=30),
                            paper_bgcolor='#111', plot_bgcolor='#111',
                            font=dict(color='#ccc'),
                            xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'),
                            showlegend=True, legend=dict(orientation='h', y=-0.3)
                        )
                        st.plotly_chart(_fig_pr, use_container_width=True)

                    # ---- ATM ±2 Strike Comparison — CE IV · PE IV (5-column per-strike) ----
                    if len(st.session_state.iv_skew_history) >= 2:
                        _ivh_df = pd.DataFrame(st.session_state.iv_skew_history)
                        st.markdown("### 📊 ATM ±2 Strike Comparison — CE IV · PE IV")
                        _ivp_pos_labels = ['🟣 ITM-2', '🟣 ITM-1', '🟡 ATM', '🔵 OTM+1', '🔵 OTM+2']
                        _ivp_pos_keys   = ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']
                        _ivp_pos_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']
                        _ivp_5cols = st.columns(5)
                        for _ivc_i, _ivc_col in enumerate(_ivp_5cols):
                            with _ivc_col:
                                _ivlk = _ivp_pos_keys[_ivc_i]
                                _ck = f"iv_ce_{_ivlk}"
                                _pk = f"iv_pe_{_ivlk}"
                                _prk = f"pres_{_ivlk}"
                                _ivc_clr = _ivp_pos_colors[_ivc_i]
                                if _ck not in _ivh_df.columns and _pk not in _ivh_df.columns:
                                    st.info(f"{_ivp_pos_labels[_ivc_i]} N/A")
                                    continue
                                _fig_ivs = go.Figure()
                                # CE IV (solid cyan)
                                if _ck in _ivh_df.columns:
                                    _fig_ivs.add_trace(go.Scatter(
                                        x=_ivh_df['time'], y=_ivh_df[_ck],
                                        mode='lines+markers', name='CE IV%',
                                        line=dict(color='#00ccff', width=2),
                                        marker=dict(size=3),
                                    ))
                                # PE IV (dashed orange)
                                if _pk in _ivh_df.columns:
                                    _fig_ivs.add_trace(go.Scatter(
                                        x=_ivh_df['time'], y=_ivh_df[_pk],
                                        mode='lines+markers', name='PE IV%',
                                        line=dict(color='#ffaa00', width=2, dash='dash'),
                                        marker=dict(size=3),
                                    ))
                                # Current value markers
                                _iv_ce_cur = _ivh_df[_ck].iloc[-1] if _ck in _ivh_df.columns else None
                                _iv_pe_cur = _ivh_df[_pk].iloc[-1] if _pk in _ivh_df.columns else None
                                if _iv_ce_cur is not None:
                                    _fig_ivs.add_trace(go.Scatter(
                                        x=[_ivh_df['time'].iloc[-1]], y=[_iv_ce_cur],
                                        mode='markers+text', text=[f'{_iv_ce_cur:.1f}%'],
                                        textposition='top right', textfont=dict(size=8, color='#00ccff'),
                                        marker=dict(size=8, color='#00ccff', symbol='circle'),
                                        showlegend=False, hoverinfo='skip',
                                    ))
                                if _iv_pe_cur is not None:
                                    _fig_ivs.add_trace(go.Scatter(
                                        x=[_ivh_df['time'].iloc[-1]], y=[_iv_pe_cur],
                                        mode='markers+text', text=[f'{_iv_pe_cur:.1f}%'],
                                        textposition='bottom right', textfont=dict(size=8, color='#ffaa00'),
                                        marker=dict(size=8, color='#ffaa00', symbol='diamond'),
                                        showlegend=False, hoverinfo='skip',
                                    ))
                                # Determine IV skew signal for this strike
                                _ivs_skew = (_iv_pe_cur / (_iv_ce_cur + 1e-6)) if (_iv_ce_cur and _iv_pe_cur) else 1.0
                                _ivs_title = f"{_ivp_pos_labels[_ivc_i]}<br>{_ivlk}"
                                if _iv_ce_cur is not None:
                                    _ivs_title += f"<br>CE:{_iv_ce_cur:.1f}% PE:{_iv_pe_cur:.1f}%" if _iv_pe_cur else f"<br>CE:{_iv_ce_cur:.1f}%"
                                _fig_ivs.update_layout(
                                    title=dict(text=_ivs_title, font=dict(size=10)),
                                    template='plotly_dark', height=300,
                                    showlegend=True,
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                                xanchor='center', x=0.5, font=dict(size=8)),
                                    margin=dict(l=5, r=10, t=80, b=30),
                                    xaxis=dict(tickformat='%H:%M', title='', tickfont=dict(size=8)),
                                    yaxis=dict(title='IV%', tickfont=dict(size=8)),
                                    plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                                )
                                st.plotly_chart(_fig_ivs, use_container_width=True)
                                # Net pressure caption
                                _pres_cur = _ivh_df[_prk].iloc[-1] if (_prk in _ivh_df.columns and len(_ivh_df) > 0) else None
                                _pres_sig = "🟢Bull" if (_pres_cur and _pres_cur > 0.15) else ("🔴Bear" if (_pres_cur and _pres_cur < -0.15) else "🟡Ntrl")
                                _pres_str = f'{_pres_cur:+.3f}' if _pres_cur is not None else '—'
                                _iv_skew_str = f'{_ivs_skew:.3f}' if _iv_ce_cur else '—'
                                st.caption(f"IV Skew:{_iv_skew_str} · Pres:{_pres_str} {_pres_sig}")

                    _ivp_col_l, _ivp_col_r = st.columns([3, 1])
                    with _ivp_col_l:
                        st.caption(f"📡 IV pts: {len(st.session_state.iv_skew_history)} · Pressure pts: {len(st.session_state.pressure_history)}")
                    with _ivp_col_r:
                        if st.button("🗑️ Clear IV/Pressure History"):
                            st.session_state.iv_skew_history = []
                            st.session_state.pressure_history = []
                            st.rerun()
                else:
                    st.info("ATM strike not identified — IV Skew & Pressure unavailable.")
            else:
                st.info("Option chain data required for IV Skew & Pressure signals.")

        except Exception as _ivp_e:
            st.warning(f"IV Skew & Pressure unavailable: {str(_ivp_e)}")

        # ===== AUTO ENTRY ENGINE =====
        st.markdown("---")
        st.markdown("## 🎯 Auto Entry Engine — CE / PE Strike + SL + Target")

        try:
            _ae_df = option_data.get('df_summary') if option_data else None
            _ae_underlying = option_data.get('underlying') if option_data else None

            # Ensure optional columns exist
            if _ae_df is not None:
                for _col in ['impliedVolatility_CE', 'impliedVolatility_PE',
                             'bidQty_CE', 'bidQty_PE', 'askQty_CE', 'askQty_PE']:
                    if _col not in _ae_df.columns:
                        _ae_df[_col] = 0.0

            if (_ae_df is not None and _ae_underlying and 'Zone' in _ae_df.columns):

                # Re-compute signals (lightweight — uses same logic as above)
                _ae_atm_idx = _ae_df[_ae_df['Zone'] == 'ATM'].index
                if len(_ae_atm_idx) > 0:
                    _ae_atm_pos = _ae_df.index.get_loc(_ae_atm_idx[0])
                    _ae_start = max(0, _ae_atm_pos - 2)
                    _ae_end = min(len(_ae_df), _ae_atm_pos + 3)
                    _ae_slice = _ae_df.iloc[_ae_start:_ae_end].copy()

                    # IV Skew
                    _ae_iv_ce = _ae_slice['impliedVolatility_CE'].fillna(0).mean()
                    _ae_iv_pe = _ae_slice['impliedVolatility_PE'].fillna(0).mean() if 'impliedVolatility_PE' in _ae_slice.columns else 0
                    _ae_iv_skew = _ae_iv_pe / (_ae_iv_ce + 1e-6)

                    # Net Pressure
                    _ae_cp_list, _ae_pp_list = [], []
                    for _, _aer in _ae_slice.iterrows():
                        _bc = float(_aer.get('bidQty_CE', 0) or 0)
                        _ac = float(_aer.get('askQty_CE', 0) or 0)
                        _bp = float(_aer.get('bidQty_PE', 0) or 0)
                        _ap = float(_aer.get('askQty_PE', 0) or 0)
                        _ae_cp_list.append((_bc - _ac) / (_bc + _ac + 1e-6))
                        _ae_pp_list.append((_bp - _ap) / (_bp + _ap + 1e-6))
                    _ae_avg_cp = sum(_ae_cp_list) / len(_ae_cp_list) if _ae_cp_list else 0
                    _ae_avg_pp = sum(_ae_pp_list) / len(_ae_pp_list) if _ae_pp_list else 0
                    _ae_net_pres = _ae_avg_cp - _ae_avg_pp

                    # Pressure spike (vs history)
                    _ae_prev_np = st.session_state.pressure_history[-2]['net_pressure'] if len(st.session_state.pressure_history) >= 2 else _ae_net_pres
                    _ae_pres_change = _ae_net_pres - _ae_prev_np

                    # PCR OI (ATM)
                    _ae_atm_row = _ae_df[_ae_df['Zone'] == 'ATM']
                    _ae_pcr = float(_ae_atm_row['PCR'].values[0]) if (len(_ae_atm_row) > 0 and 'PCR' in _ae_atm_row.columns) else 1.0

                    # Total Net GEX (from existing gex_data if available)
                    _ae_net_gex = 0.0
                    try:
                        if gex_data and 'net_gex' in gex_data:
                            _ae_net_gex = float(gex_data['net_gex'])
                        elif gex_data and 'gex_df' in gex_data:
                            _ae_gdf = gex_data['gex_df']
                            if 'Net_GEX' in _ae_gdf.columns:
                                _ae_net_gex = float(_ae_gdf['Net_GEX'].sum())
                    except Exception:
                        pass

                    # ATM strike value
                    _ae_atm_strike = float(_ae_atm_row['Strike'].values[0]) if len(_ae_atm_row) > 0 else round(_ae_underlying / 50) * 50

                    # Determine strike step (50 or 100 points)
                    _ae_strikes_sorted = sorted(_ae_slice['Strike'].unique())
                    _ae_step = int(_ae_strikes_sorted[1] - _ae_strikes_sorted[0]) if len(_ae_strikes_sorted) >= 2 else 50

                    # ---- ENTRY LOGIC ----
                    _ae_signal = "NO TRADE"
                    _ae_strike = _ae_atm_strike
                    _ae_sl = None
                    _ae_target = None
                    _ae_trade_type = "Low Edge"
                    _ae_option_type = ""

                    # Safety filter: avoid trades when market is pinned
                    if _ae_net_gex > 10:
                        _ae_signal = "NO TRADE"
                        _ae_trade_type = "Market Capped (GEX > 10L)"
                    elif _ae_iv_skew < 0.90 and _ae_net_pres > 0.15 and _ae_net_gex < 0:
                        _ae_signal = "BUY CE"
                        _ae_option_type = "CE"
                        if _ae_net_pres > 0.30:
                            _ae_strike = _ae_atm_strike + _ae_step
                            _ae_trade_type = "Breakout"
                        elif _ae_net_pres > 0.50:
                            _ae_strike = _ae_atm_strike + 2 * _ae_step
                            _ae_trade_type = "Strong Breakout"
                        else:
                            _ae_strike = _ae_atm_strike
                            _ae_trade_type = "Scalp"
                        _ae_sl = _ae_underlying - 20
                        _ae_target = _ae_underlying + 40

                    elif _ae_iv_skew > 1.10 and _ae_net_pres < -0.15 and _ae_net_gex < 0:
                        _ae_signal = "BUY PE"
                        _ae_option_type = "PE"
                        if _ae_net_pres < -0.30:
                            _ae_strike = _ae_atm_strike - _ae_step
                            _ae_trade_type = "Breakdown"
                        elif _ae_net_pres < -0.50:
                            _ae_strike = _ae_atm_strike - 2 * _ae_step
                            _ae_trade_type = "Strong Breakdown"
                        else:
                            _ae_strike = _ae_atm_strike
                            _ae_trade_type = "Scalp"
                        _ae_sl = _ae_underlying + 20
                        _ae_target = _ae_underlying - 40

                    elif _ae_iv_skew < 0.90 and _ae_net_pres < 0:
                        _ae_signal = "BUY PE"
                        _ae_option_type = "PE"
                        _ae_strike = _ae_atm_strike
                        _ae_trade_type = "Bull Trap Reversal"
                        _ae_sl = _ae_underlying + 15
                        _ae_target = _ae_underlying - 30

                    elif _ae_iv_skew > 1.10 and _ae_net_pres > 0:
                        _ae_signal = "BUY CE"
                        _ae_option_type = "CE"
                        _ae_strike = _ae_atm_strike
                        _ae_trade_type = "Bear Trap Reversal"
                        _ae_sl = _ae_underlying - 15
                        _ae_target = _ae_underlying + 30

                    _ae_timing = "ENTER NOW ⚡" if abs(_ae_pres_change) > 0.15 else "WAIT ⏳"

                    # Signal color
                    if "BUY CE" in _ae_signal:
                        _ae_sig_color = "#00C853"
                    elif "BUY PE" in _ae_signal:
                        _ae_sig_color = "#FF5252"
                    else:
                        _ae_sig_color = "#888888"

                    # ---- UI ----
                    _ae_c1, _ae_c2, _ae_c3, _ae_c4 = st.columns(4)
                    with _ae_c1:
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:2px solid {_ae_sig_color};">
                        <div style="color:#aaa;font-size:11px;">SIGNAL</div>
                        <div style="font-size:20px;font-weight:bold;color:{_ae_sig_color};">{_ae_signal}</div>
                        <div style="font-size:12px;color:#aaa;margin-top:4px;">{_ae_trade_type}</div>
                        <div style="font-size:13px;margin-top:4px;color:#FFD740;">{_ae_timing}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with _ae_c2:
                        _ae_strike_label = f"₹{int(_ae_strike)}" if _ae_signal != "NO TRADE" else "—"
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:1px solid #444;">
                        <div style="color:#aaa;font-size:11px;">STRIKE</div>
                        <div style="font-size:22px;font-weight:bold;color:#fff;">{_ae_strike_label}</div>
                        <div style="font-size:12px;color:#aaa;margin-top:4px;">{_ae_option_type or '—'}</div>
                        <div style="font-size:11px;color:#777;margin-top:2px;">ATM: ₹{int(_ae_atm_strike)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with _ae_c3:
                        _ae_sl_label = f"₹{_ae_sl:.0f}" if _ae_sl else "—"
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:1px solid #FF5252;">
                        <div style="color:#aaa;font-size:11px;">STOP LOSS (Spot)</div>
                        <div style="font-size:20px;font-weight:bold;color:#FF5252;">{_ae_sl_label}</div>
                        <div style="font-size:11px;color:#777;margin-top:4px;">Spot: ₹{_ae_underlying:.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with _ae_c4:
                        _ae_tgt_label = f"₹{_ae_target:.0f}" if _ae_target else "—"
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:1px solid #00C853;">
                        <div style="color:#aaa;font-size:11px;">TARGET (Spot)</div>
                        <div style="font-size:20px;font-weight:bold;color:#00C853;">{_ae_tgt_label}</div>
                        <div style="font-size:11px;color:#777;margin-top:4px;">
                            {"R:R = 1:2" if _ae_sl and _ae_target else ""}
                        </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ---- Inputs summary row ----
                    st.markdown("<br>", unsafe_allow_html=True)
                    _ae_meta_cols = st.columns(5)
                    _ae_meta_labels = [
                        ("IV Skew", f"{_ae_iv_skew:.3f}"),
                        ("Net Pressure", f"{_ae_net_pres:+.3f}"),
                        ("PCR (OI)", f"{_ae_pcr:.2f}"),
                        ("Net GEX", f"{_ae_net_gex:+.1f}L"),
                        ("Pressure Δ", f"{_ae_pres_change:+.3f}"),
                    ]
                    for _col, (_lbl, _val) in zip(_ae_meta_cols, _ae_meta_labels):
                        with _col:
                            st.metric(_lbl, _val)

                    with st.expander("📖 Auto Entry Engine — How It Works"):
                        st.markdown("""
                        | Condition | Signal |
                        |-----------|--------|
                        | IV Skew < 0.90 AND Net Pressure > 0.15 AND GEX < 0 | **BUY CE** (Breakout/Scalp) |
                        | IV Skew > 1.10 AND Net Pressure < -0.15 AND GEX < 0 | **BUY PE** (Breakdown/Scalp) |
                        | IV Skew < 0.90 AND Net Pressure < 0 | **BUY PE** (Bull Trap Reversal) |
                        | IV Skew > 1.10 AND Net Pressure > 0 | **BUY CE** (Bear Trap Reversal) |
                        | GEX > 10L | **NO TRADE** (Market Capped) |

                        **Strike Selection:**
                        - Net Pressure > 0.50 → ATM+2 (Strong Breakout)
                        - Net Pressure > 0.30 → ATM+1 (Breakout)
                        - Otherwise → ATM (Scalp)

                        **SL/Target based on spot points:**
                        - CE trade: SL = Spot − 20, Target = Spot + 40
                        - PE trade: SL = Spot + 20, Target = Spot − 40
                        - Trap reversals: Tighter (±15 SL / ∓30 Target)

                        **Entry Timing:** Pressure spike > 0.15 in last interval → ENTER NOW ⚡

                        > ⚠️ These are directional signals for educational use.
                        > Always apply your own risk management and position sizing.
                        """)
                else:
                    st.info("ATM strike not identified — Auto Entry Engine unavailable.")
            else:
                st.info("Option chain data required for Auto Entry Engine.")

        except Exception as _ae_e:
            st.warning(f"Auto Entry Engine unavailable: {str(_ae_e)}")

        # ===== DELTA & GAMMA ENGINE (PER STRIKE + OVERALL) =====
        st.markdown("---")
        st.markdown("## ⚡ Delta & Gamma Engine — Per Strike + Overall (ATM ± 2)")

        try:
            _dg_df = option_data.get('df_summary') if option_data else None
            _dg_underlying = option_data.get('underlying') if option_data else None

            _dg_required = ['Zone', 'Strike', 'Delta_CE', 'Delta_PE', 'Gamma_CE', 'Gamma_PE',
                            'openInterest_CE', 'openInterest_PE']

            if (_dg_df is not None and _dg_underlying and
                    all(c in _dg_df.columns for c in _dg_required)):

                # --- ATM ± 2 slice ---
                _dg_atm_idx = _dg_df[_dg_df['Zone'] == 'ATM'].index
                if len(_dg_atm_idx) > 0:
                    _dg_atm_pos = _dg_df.index.get_loc(_dg_atm_idx[0])
                    _dg_start = max(0, _dg_atm_pos - 2)
                    _dg_end   = min(len(_dg_df), _dg_atm_pos + 3)
                    _dg_slice = _dg_df.iloc[_dg_start:_dg_end].copy()

                    _dg_strikes_sorted = sorted(_dg_slice['Strike'].unique())
                    _dg_step = int(_dg_strikes_sorted[1] - _dg_strikes_sorted[0]) if len(_dg_strikes_sorted) >= 2 else 50
                    _dg_atm_val = float(_dg_df[_dg_df['Zone'] == 'ATM']['Strike'].values[0])

                    # Strike position labels
                    def _dg_label(s):
                        diff = int(round((s - _dg_atm_val) / _dg_step))
                        if diff == 0:   return "ATM"
                        if diff > 0:    return f"ATM+{diff}"
                        return f"ATM{diff}"

                    # --- PER STRIKE delta & gamma ---
                    _dg_delta_per = {}
                    _dg_gamma_per = {}
                    _contract_mult = 25

                    for _, _r in _dg_slice.iterrows():
                        _s = float(_r['Strike'])
                        _lbl = _dg_label(_s)
                        _d_ce = float(_r.get('Delta_CE', 0) or 0)
                        _d_pe = float(_r.get('Delta_PE', 0) or 0)
                        _g_ce = float(_r.get('Gamma_CE', 0) or 0)
                        _g_pe = float(_r.get('Gamma_PE', 0) or 0)
                        _oi_ce = float(_r.get('openInterest_CE', 0) or 0)
                        _oi_pe = float(_r.get('openInterest_PE', 0) or 0)

                        _dg_delta_per[_lbl] = round(_d_ce + _d_pe, 4)
                        # Gamma per strike in Lakhs (PE builds − CE pins)
                        _dg_gamma_per[_lbl] = round(
                            (_g_pe * _oi_pe - _g_ce * _oi_ce) * _contract_mult * _dg_underlying / 100000, 2
                        )

                    # --- OVERALL NET ---
                    _all_d_ce = [float(_r.get('Delta_CE', 0) or 0) for _, _r in _dg_slice.iterrows()]
                    _all_d_pe = [float(_r.get('Delta_PE', 0) or 0) for _, _r in _dg_slice.iterrows()]
                    _avg_d_ce = sum(_all_d_ce) / len(_all_d_ce) if _all_d_ce else 0
                    _avg_d_pe = sum(_all_d_pe) / len(_all_d_pe) if _all_d_pe else 0
                    _net_delta = round(_avg_d_ce + _avg_d_pe, 4)
                    _net_gamma = round(sum(_dg_gamma_per.values()), 2)

                    # --- HISTORY (time-series) ---
                    _dg_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                    _dg_should_append = (
                        not st.session_state.delta_gamma_history or
                        (_dg_now - st.session_state.delta_gamma_history[-1]['time']).total_seconds() >= 30
                    )
                    if _dg_should_append:
                        st.session_state.delta_gamma_history.append({
                            'time': _dg_now,
                            'net_delta': _net_delta,
                            'net_gamma': _net_gamma,
                            **{f"delta_{k}": v for k, v in _dg_delta_per.items()},
                            **{f"gamma_{k}": v for k, v in _dg_gamma_per.items()},
                        })
                        if len(st.session_state.delta_gamma_history) > 200:
                            st.session_state.delta_gamma_history = st.session_state.delta_gamma_history[-200:]

                    # --- CHANGE vs PREVIOUS ---
                    _prev_dg = st.session_state.delta_gamma_history
                    _prev_nd = _prev_dg[-2]['net_delta'] if len(_prev_dg) >= 2 else _net_delta
                    _prev_ng = _prev_dg[-2]['net_gamma'] if len(_prev_dg) >= 2 else _net_gamma
                    _delta_change = round(_net_delta - _prev_nd, 4)
                    _gamma_change = round(_net_gamma - _prev_ng, 2)

                    # --- HOT STRIKE (highest abs gamma) ---
                    _hot_lbl = max(_dg_gamma_per, key=lambda k: abs(_dg_gamma_per[k])) if _dg_gamma_per else "ATM"
                    _hot_val = _dg_gamma_per.get(_hot_lbl, 0)

                    if _hot_lbl in ("ATM+1", "ATM+2") and _hot_val > 0:
                        _hot_signal = "🚀 Bullish Build Above ATM"
                        _hot_color  = "#00C853"
                    elif _hot_lbl in ("ATM-1", "ATM-2") and _hot_val < 0:
                        _hot_signal = "🔥 Bearish Build Below ATM"
                        _hot_color  = "#FF5252"
                    elif _hot_lbl == "ATM":
                        _hot_signal = "📌 Pinned at ATM"
                        _hot_color  = "#FFD740"
                    else:
                        _hot_signal = "➡️ Neutral"
                        _hot_color  = "#888888"

                    # --- MAIN SIGNAL ---
                    if _net_delta > 0.15 and _gamma_change > 0:
                        _dg_main_signal = "🚀 Bullish Momentum"
                        _dg_main_color  = "#00C853"
                    elif _net_delta < -0.15 and _gamma_change > 0:
                        _dg_main_signal = "🔥 Bearish Momentum"
                        _dg_main_color  = "#FF5252"
                    elif _net_delta > 0.05:
                        _dg_main_signal = "🟡 Mild Bullish"
                        _dg_main_color  = "#FFD740"
                    elif _net_delta < -0.05:
                        _dg_main_signal = "🟡 Mild Bearish"
                        _dg_main_color  = "#FFD740"
                    else:
                        _dg_main_signal = "⏳ No Clear Trend"
                        _dg_main_color  = "#888888"

                    # --- EARLY ENTRY ENGINE ---
                    if _net_delta > 0 and _gamma_change > 0 and _hot_lbl in ("ATM+1", "ATM+2"):
                        _dg_entry = "🚀 ENTER CE — Early Breakout"
                        _dg_entry_color = "#00C853"
                    elif _net_delta < 0 and _gamma_change > 0 and _hot_lbl in ("ATM-1", "ATM-2"):
                        _dg_entry = "🔥 ENTER PE — Early Breakdown"
                        _dg_entry_color = "#FF5252"
                    elif _gamma_change > 0:
                        _dg_entry = "⚡ Gamma Rising — Watch for breakout"
                        _dg_entry_color = "#FFD740"
                    else:
                        _dg_entry = "WAIT ⏳"
                        _dg_entry_color = "#888888"

                    # ======== UI OUTPUT ========
                    _dg_c1, _dg_c2, _dg_c3, _dg_c4 = st.columns(4)
                    with _dg_c1:
                        _nd_color = "#00C853" if _net_delta > 0.05 else ("#FF5252" if _net_delta < -0.05 else "#FFD740")
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:1px solid {_nd_color};">
                        <div style="color:#aaa;font-size:11px;">NET DELTA</div>
                        <div style="font-size:22px;font-weight:bold;color:{_nd_color};">{_net_delta:+.4f}</div>
                        <div style="font-size:12px;color:#aaa;margin-top:4px;">Δ change: {_delta_change:+.4f}</div>
                        <div style="font-size:11px;color:#777;">avg CE Δ: {_avg_d_ce:+.3f} &nbsp; PE Δ: {_avg_d_pe:+.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with _dg_c2:
                        _ng_color = "#00C853" if _net_gamma > 5 else ("#FF5252" if _net_gamma < -5 else "#FFD740")
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:1px solid {_ng_color};">
                        <div style="color:#aaa;font-size:11px;">NET GAMMA (Lakhs)</div>
                        <div style="font-size:22px;font-weight:bold;color:{_ng_color};">{_net_gamma:+.2f}L</div>
                        <div style="font-size:12px;color:#aaa;margin-top:4px;">Δ change: {_gamma_change:+.2f}L</div>
                        <div style="font-size:11px;color:#777;">{"⬆️ Gamma expanding" if _gamma_change > 0 else "⬇️ Gamma contracting"}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with _dg_c3:
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:1px solid {_hot_color};">
                        <div style="color:#aaa;font-size:11px;">HOT STRIKE</div>
                        <div style="font-size:20px;font-weight:bold;color:{_hot_color};">{_hot_lbl}</div>
                        <div style="font-size:12px;margin-top:4px;">{_hot_signal}</div>
                        <div style="font-size:11px;color:#777;margin-top:2px;">Gamma: {_hot_val:+.2f}L</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with _dg_c4:
                        st.markdown(f"""
                        <div style="background:#1e1e1e;padding:14px;border-radius:10px;border:2px solid {_dg_main_color};">
                        <div style="color:#aaa;font-size:11px;">SIGNAL &amp; ENTRY</div>
                        <div style="font-size:14px;font-weight:bold;color:{_dg_main_color};margin-top:4px;">{_dg_main_signal}</div>
                        <div style="font-size:13px;color:{_dg_entry_color};margin-top:6px;">{_dg_entry}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ---- Per-Strike Current Snapshot ----
                    _dg_snap_rows = []
                    for _lk in sorted(_dg_delta_per.keys(), key=lambda x: (
                        -99 if x == 'ATM-2' else -1 if x == 'ATM-1' else 0 if x == 'ATM' else 1 if x == 'ATM+1' else 2
                    )):
                        _dg_snap_rows.append({
                            'Strike': _lk,
                            'Net Delta': f"{_dg_delta_per[_lk]:+.4f}",
                            'Net Gamma (L)': f"{_dg_gamma_per.get(_lk, 0):+.2f}",
                            'Hot': '🔥' if _lk == _hot_lbl else ''
                        })
                    if _dg_snap_rows:
                        _dg_snap_df = pd.DataFrame(_dg_snap_rows)
                        st.dataframe(_dg_snap_df, use_container_width=True, hide_index=True)

                    # ---- Overall Delta & Gamma Time-Series ----
                    if len(st.session_state.delta_gamma_history) >= 2:
                        _dgh = pd.DataFrame(st.session_state.delta_gamma_history)
                        _fig_dg = go.Figure()
                        _fig_dg.add_trace(go.Scatter(
                            x=_dgh['time'], y=_dgh['net_delta'],
                            mode='lines+markers', name='Net Delta',
                            line=dict(color='#00C853', width=2),
                            marker=dict(size=4), yaxis='y1'
                        ))
                        _fig_dg.add_trace(go.Scatter(
                            x=_dgh['time'], y=_dgh['net_gamma'],
                            mode='lines+markers', name='Net Gamma (L)',
                            line=dict(color='#FFD740', width=2, dash='dot'),
                            marker=dict(size=4), yaxis='y2'
                        ))
                        _fig_dg.add_hline(y=0.15, line_dash='dash', line_color='#00C853',
                                          annotation_text='Bull Δ 0.15', annotation_position='bottom right',
                                          yref='y1')
                        _fig_dg.add_hline(y=-0.15, line_dash='dash', line_color='#FF5252',
                                          annotation_text='Bear Δ -0.15', annotation_position='top right',
                                          yref='y1')
                        # Current value markers (present values shown in graph)
                        _cur_nd = _dgh['net_delta'].iloc[-1]
                        _cur_ng = _dgh['net_gamma'].iloc[-1]
                        _fig_dg.add_trace(go.Scatter(
                            x=[_dgh['time'].iloc[-1]], y=[_cur_nd],
                            mode='markers+text', text=[f'{_cur_nd:+.4f}'],
                            textposition='top right', textfont=dict(size=9, color='#00C853'),
                            marker=dict(size=9, color='#00C853', symbol='circle'),
                            showlegend=False, hoverinfo='skip', yaxis='y1'
                        ))
                        _fig_dg.add_trace(go.Scatter(
                            x=[_dgh['time'].iloc[-1]], y=[_cur_ng],
                            mode='markers+text', text=[f'{_cur_ng:+.2f}L'],
                            textposition='bottom right', textfont=dict(size=9, color='#FFD740'),
                            marker=dict(size=9, color='#FFD740', symbol='diamond'),
                            showlegend=False, hoverinfo='skip', yaxis='y2'
                        ))
                        _fig_dg.update_layout(
                            title=f'Net Delta & Net Gamma Over Time (ATM ± 2) — Δ:{_cur_nd:+.4f} Γ:{_cur_ng:+.2f}L',
                            height=280, margin=dict(l=40, r=60, t=40, b=30),
                            paper_bgcolor='#111', plot_bgcolor='#111',
                            font=dict(color='#ccc'),
                            xaxis=dict(gridcolor='#333'),
                            yaxis=dict(title=dict(text='Net Delta', font=dict(color='#00C853')), gridcolor='#333'),
                            yaxis2=dict(title=dict(text='Net Gamma (L)', font=dict(color='#FFD740')),
                                        overlaying='y', side='right', showgrid=False),
                            showlegend=True, legend=dict(orientation='h', y=-0.3)
                        )
                        st.plotly_chart(_fig_dg, use_container_width=True)

                        # ---- ATM ±2 Strike Comparison — Delta · Gamma (5-column per-strike) ----
                        st.markdown("### 📊 ATM ±2 Strike Comparison — Delta · Gamma")
                        _dg_pos_labels = ['🟣 ITM-2', '🟣 ITM-1', '🟡 ATM', '🔵 OTM+1', '🔵 OTM+2']
                        _dg_pos_keys   = ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']
                        _dg_pos_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']
                        _dg_ps_cols = st.columns(5)
                        for _dg_ci, _dg_col in enumerate(_dg_ps_cols):
                            with _dg_col:
                                _lk = _dg_pos_keys[_dg_ci]
                                _dk = f"delta_{_lk}"
                                _gk = f"gamma_{_lk}"
                                _clr = _dg_pos_colors[_dg_ci]
                                if _dk not in _dgh.columns and _gk not in _dgh.columns:
                                    st.info(f"{_dg_pos_labels[_dg_ci]} N/A")
                                    continue
                                _fig_dg_ps = go.Figure()
                                # Delta (solid, left y-axis)
                                if _dk in _dgh.columns:
                                    _fig_dg_ps.add_trace(go.Scatter(
                                        x=_dgh['time'], y=_dgh[_dk],
                                        mode='lines+markers', name='Delta',
                                        line=dict(color='#00C853', width=2),
                                        marker=dict(size=3), yaxis='y1'
                                    ))
                                # Gamma (dashed, right y-axis)
                                if _gk in _dgh.columns:
                                    _fig_dg_ps.add_trace(go.Scatter(
                                        x=_dgh['time'], y=_dgh[_gk],
                                        mode='lines+markers', name='Gamma (L)',
                                        line=dict(color='#FFD740', width=2, dash='dash'),
                                        marker=dict(size=3), yaxis='y2'
                                    ))
                                _fig_dg_ps.add_hline(y=0, line_color='rgba(255,255,255,0.3)', line_width=1, yref='y1')
                                # Current value markers
                                _ps_cur_d = _dgh[_dk].iloc[-1] if _dk in _dgh.columns and len(_dgh) > 0 else None
                                _ps_cur_g = _dgh[_gk].iloc[-1] if _gk in _dgh.columns and len(_dgh) > 0 else None
                                if _ps_cur_d is not None:
                                    _fig_dg_ps.add_trace(go.Scatter(
                                        x=[_dgh['time'].iloc[-1]], y=[_ps_cur_d],
                                        mode='markers+text', text=[f'{_ps_cur_d:+.4f}'],
                                        textposition='top right', textfont=dict(size=8, color='#00C853'),
                                        marker=dict(size=8, color='#00C853', symbol='circle'),
                                        showlegend=False, hoverinfo='skip', yaxis='y1'
                                    ))
                                if _ps_cur_g is not None:
                                    _fig_dg_ps.add_trace(go.Scatter(
                                        x=[_dgh['time'].iloc[-1]], y=[_ps_cur_g],
                                        mode='markers+text', text=[f'{_ps_cur_g:+.2f}L'],
                                        textposition='bottom right', textfont=dict(size=8, color='#FFD740'),
                                        marker=dict(size=8, color='#FFD740', symbol='diamond'),
                                        showlegend=False, hoverinfo='skip', yaxis='y2'
                                    ))
                                _hot_border = '2px solid #ff4444' if _lk == _hot_lbl else '1px solid #333'
                                _title_txt = f"{_dg_pos_labels[_dg_ci]}<br>{_lk}"
                                if _ps_cur_d is not None:
                                    _title_txt += f"<br>Δ:{_ps_cur_d:+.4f}"
                                if _ps_cur_g is not None:
                                    _title_txt += f" Γ:{_ps_cur_g:+.2f}L"
                                _fig_dg_ps.update_layout(
                                    title=dict(text=_title_txt, font=dict(size=10)),
                                    template='plotly_dark', height=300,
                                    showlegend=True,
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                                xanchor='center', x=0.5, font=dict(size=8)),
                                    margin=dict(l=5, r=35, t=80, b=30),
                                    xaxis=dict(tickformat='%H:%M', title='', tickfont=dict(size=8)),
                                    yaxis=dict(title='Delta', title_font=dict(color='#00C853', size=9),
                                               tickfont=dict(size=8), gridcolor='#333'),
                                    yaxis2=dict(title='Gamma (L)', title_font=dict(color='#FFD740', size=9),
                                                overlaying='y', side='right', showgrid=False,
                                                tickfont=dict(size=8)),
                                    plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                                )
                                st.plotly_chart(_fig_dg_ps, use_container_width=True)
                                _sig_d = "🟢" if _ps_cur_d and _ps_cur_d > 0.05 else ("🔴" if _ps_cur_d and _ps_cur_d < -0.05 else "🟡")
                                _sig_g = "📍Pin" if _ps_cur_g and _ps_cur_g > 5 else ("⚡Acc" if _ps_cur_g and _ps_cur_g < -5 else "➡️Ntrl")
                                _ps_d_str = f'{_ps_cur_d:+.4f}' if _ps_cur_d is not None else '—'
                                _ps_g_str = f'{_ps_cur_g:+.2f}L' if _ps_cur_g is not None else '—'
                                st.caption(f"{_sig_d} Δ:{_ps_d_str} · {_sig_g} Γ:{_ps_g_str}")

                    _dg_col_l, _dg_col_r = st.columns([3, 1])
                    with _dg_col_l:
                        st.caption(f"⚡ Delta/Gamma pts: {len(st.session_state.delta_gamma_history)}")
                    with _dg_col_r:
                        if st.button("🗑️ Clear Delta/Gamma History"):
                            st.session_state.delta_gamma_history = []
                            st.rerun()
                else:
                    st.info("ATM strike not identified — Delta & Gamma Engine unavailable.")
            else:
                st.info("Option chain data with Greeks required for Delta & Gamma Engine.")

        except Exception as _dg_e:
            st.warning(f"Delta & Gamma Engine unavailable: {str(_dg_e)}")

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

    # ===== MARKET ACCELERATION ENGINE =====
    st.markdown("---")
    with st.expander("🚀 Market Acceleration Engine — Spike · Gamma · Expiry Intelligence", expanded=False):
        try:
            _mae_data = option_data
            _mae_underlying = _mae_data.get('underlying') if _mae_data else None
            _mae_df = _mae_data.get('df_summary') if _mae_data else None
            _mae_expiry = _mae_data.get('expiry') if _mae_data else None

            if _mae_df is not None and _mae_underlying and _mae_expiry:
                _mae_atm = min(_mae_df['Strike'], key=lambda x: abs(x - _mae_underlying))
                _mae_now_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')

                # ---- Store snapshot in history (limit to 50 per session) ----
                _snap = _mae_df.copy()
                st.session_state.spike_history.append(_snap)
                if len(st.session_state.spike_history) > 50:
                    st.session_state.spike_history = st.session_state.spike_history[-50:]

                st.session_state.expiry_spike_history.append(_snap)
                if len(st.session_state.expiry_spike_history) > 50:
                    st.session_state.expiry_spike_history = st.session_state.expiry_spike_history[-50:]

                st.session_state.expiry_intel_history.append(_snap)
                if len(st.session_state.expiry_intel_history) > 50:
                    st.session_state.expiry_intel_history = st.session_state.expiry_intel_history[-50:]

                # ---- Run engines ----
                _spike = calculate_options_spike_score(_mae_df, _mae_atm, st.session_state.spike_history)
                _expiry_spike = calculate_expiry_spike_score(_mae_df, _mae_atm, _mae_expiry, st.session_state.expiry_spike_history)
                _gamma = analyze_gamma_sequence_mae(_mae_df, _mae_atm, st.session_state.gamma_seq_history)
                _expiry_intel = calculate_expiry_day_intelligence(_mae_df, _mae_atm, _mae_underlying, _mae_expiry, st.session_state.expiry_intel_history)

                # ---- Store gamma sequence snapshot ----
                _gamma_snap = dict(_gamma['gamma_values'])
                _gamma_snap['time'] = _mae_now_str
                st.session_state.gamma_seq_history.append(_gamma_snap)
                if len(st.session_state.gamma_seq_history) > 50:
                    st.session_state.gamma_seq_history = st.session_state.gamma_seq_history[-50:]

                # ---- Get sentiment verdict from session state ----
                _sent_verdict = 'Neutral'
                if st.session_state.get('sentiment_history'):
                    _last_sent = st.session_state.sentiment_history[-1]
                    _sent_verdict = _last_sent.get('verdict', 'Neutral') if isinstance(_last_sent, dict) else 'Neutral'

                _combined_signal, _combined_color, _combined_label = get_combined_acceleration_signal(
                    _sent_verdict, _spike, _gamma
                )

                # ---- Store spike history snapshots in Supabase (background, non-blocking) ----
                try:
                    _spike_record = {
                        'timestamp': datetime.now(pytz.UTC).isoformat(),
                        'atm_strike': float(_mae_atm),
                        'spike_score': float(_spike['spike_score']),
                        'direction': _spike['direction'],
                        'signal': _spike['signal'],
                        'conditions_met': int(_spike['conditions_met']),
                    }
                    db.save_spike_history(_spike_record)

                    if _expiry_spike['active']:
                        _expiry_spike_record = {
                            'timestamp': datetime.now(pytz.UTC).isoformat(),
                            'atm_strike': float(_mae_atm),
                            'dte': int(_expiry_spike['dte']),
                            'expiry_spike_score': float(_expiry_spike['expiry_spike_score']),
                            'signal': _expiry_spike['signal'],
                            'short_cover': bool(_expiry_spike['short_cover']),
                            'long_unwind': bool(_expiry_spike['long_unwind']),
                        }
                        db.save_expiry_spike_history(_expiry_spike_record)

                    _gamma_seq_record = {
                        'timestamp': datetime.now(pytz.UTC).isoformat(),
                        'atm_strike': float(_mae_atm),
                        'pattern': _gamma['pattern'],
                        'direction': _gamma['direction'],
                        'acceleration': bool(_gamma['acceleration']),
                        'bull_trap': bool(_gamma['bull_trap']),
                        'bear_trap': bool(_gamma['bear_trap']),
                    }
                    db.save_gamma_sequence_history(_gamma_seq_record)
                except Exception:
                    pass

                # ---- TELEGRAM ALERTS ----
                _ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                _alert_minute = _ist_now.strftime('%Y-%m-%d %H:%M')

                if _spike['spike_score'] >= 75 and st.session_state.last_spike_alert != _alert_minute:
                    _spike_msg = (
                        f"🚀 <b>INSTITUTIONAL SPIKE DETECTED</b>\n\n"
                        f"Spot: ₹{_mae_underlying:.0f}\n"
                        f"Spike Score: <b>{_spike['spike_score']}</b>\n"
                        f"Signal: {_spike['signal']}\n"
                        f"Direction: {_spike['direction']}\n"
                        f"Conditions Met: {_spike['conditions_met']}/6\n"
                        f"ATM Strike: {_mae_atm}\n"
                        f"Gamma Pattern: {_gamma['pattern']}\n"
                        f"Combined: {_combined_signal}\n"
                        f"Time: {_mae_now_str} IST"
                    )
                    send_telegram_message_sync(_spike_msg)
                    st.session_state.last_spike_alert = _alert_minute

                if _gamma['pattern'] in ('Gamma Ramp Up', 'Gamma Ramp Down') and st.session_state.last_gamma_alert != _alert_minute:
                    _gamma_msg = (
                        f"📊 <b>GAMMA SEQUENCE SIGNAL</b>\n\n"
                        f"Spot: ₹{_mae_underlying:.0f}\n"
                        f"Pattern: <b>{_gamma['pattern']}</b>\n"
                        f"Direction: {_gamma['direction']}\n"
                        f"Dealer Signal: {_gamma['dealer_signal']}\n"
                        f"{'⚡ Dealer Hedge Acceleration!' if _gamma['acceleration'] else ''}\n"
                        f"{'⚠️ Trap: ' + _gamma['trap_signal'] if _gamma['trap_signal'] else ''}\n"
                        f"Spike Score: {_spike['spike_score']}\n"
                        f"Combined: {_combined_signal}\n"
                        f"Time: {_mae_now_str} IST"
                    )
                    send_telegram_message_sync(_gamma_msg)
                    st.session_state.last_gamma_alert = _alert_minute

                if _expiry_spike['active'] and _expiry_spike['expiry_spike_score'] >= 80 and st.session_state.last_expiry_spike_alert != _alert_minute:
                    _expiry_msg = (
                        f"⚡ <b>EXPIRY MOVE DETECTED</b>\n\n"
                        f"Spot: ₹{_mae_underlying:.0f}\n"
                        f"Market Type: {_expiry_intel.get('market_type', 'N/A')}\n"
                        f"Expiry Spike Score: <b>{_expiry_spike['expiry_spike_score']}</b>\n"
                        f"Signal: {_expiry_spike['signal']}\n"
                        f"{'SHORT COVERING DETECTED' if _expiry_spike['short_cover'] else ''}\n"
                        f"{'LONG UNWINDING DETECTED' if _expiry_spike['long_unwind'] else ''}\n"
                        f"Breakout Level: {_expiry_intel.get('breakout_level', 'N/A')}\n"
                        f"Breakdown Level: {_expiry_intel.get('breakdown_level', 'N/A')}\n"
                        f"Expiry Score: {_expiry_intel.get('expiry_score', 0)}\n"
                        f"Confidence: HIGH\nTime: {_mae_now_str} IST"
                    )
                    send_telegram_message_sync(_expiry_msg)
                    st.session_state.last_expiry_spike_alert = _alert_minute

                # ---- DASHBOARD DISPLAY ----
                # Top summary row
                _mae_c1, _mae_c2, _mae_c3, _mae_c4 = st.columns(4)
                _spike_color = '#ff4444' if _spike['spike_score'] >= 80 else ('#ffaa00' if _spike['spike_score'] >= 60 else ('#44aaff' if _spike['spike_score'] >= 30 else '#888888'))
                _gamma_color = '#00ff88' if _gamma['direction'] == 'Bullish' else ('#ff4444' if _gamma['direction'] == 'Bearish' else '#888888')
                _expiry_color = '#ff6600' if _expiry_spike['active'] and _expiry_spike['expiry_spike_score'] >= 70 else '#aaaaaa'

                with _mae_c1:
                    st.markdown(f"""<div style='background:#1e1e1e;padding:12px;border-radius:8px;border-left:4px solid {_spike_color}'>
                    <div style='color:#aaa;font-size:11px'>SPIKE SCORE</div>
                    <div style='font-size:28px;font-weight:bold;color:{_spike_color}'>{_spike['spike_score']}</div>
                    <div style='color:#ccc;font-size:12px'>{_spike['signal']}</div>
                    <div style='color:#999;font-size:11px'>{_spike['direction']} · {_spike['conditions_met']}/6 cond</div>
                    </div>""", unsafe_allow_html=True)

                with _mae_c2:
                    _exp_score_display = f"{_expiry_spike['expiry_spike_score']}" if _expiry_spike['active'] else "N/A (DTE>{_expiry_spike['dte']})"
                    _exp_signal = _expiry_spike['signal'] if _expiry_spike['active'] else f"DTE {_expiry_spike['dte']}d"
                    st.markdown(f"""<div style='background:#1e1e1e;padding:12px;border-radius:8px;border-left:4px solid {_expiry_color}'>
                    <div style='color:#aaa;font-size:11px'>EXPIRY SPIKE</div>
                    <div style='font-size:28px;font-weight:bold;color:{_expiry_color}'>{_exp_score_display}</div>
                    <div style='color:#ccc;font-size:12px'>{_exp_signal}</div>
                    <div style='color:#999;font-size:11px'>DTE: {_expiry_spike['dte']} day(s)</div>
                    </div>""", unsafe_allow_html=True)

                with _mae_c3:
                    _accel_icon = '⚡' if _gamma['acceleration'] else ''
                    st.markdown(f"""<div style='background:#1e1e1e;padding:12px;border-radius:8px;border-left:4px solid {_gamma_color}'>
                    <div style='color:#aaa;font-size:11px'>GAMMA SEQUENCE</div>
                    <div style='font-size:18px;font-weight:bold;color:{_gamma_color}'>{_gamma['pattern']} {_accel_icon}</div>
                    <div style='color:#ccc;font-size:12px'>{_gamma['dealer_signal']}</div>
                    <div style='color:#ff4444;font-size:11px'>{_gamma['trap_signal'] if _gamma['trap_signal'] else ''}</div>
                    </div>""", unsafe_allow_html=True)

                with _mae_c4:
                    st.markdown(f"""<div style='background:#1e1e1e;padding:12px;border-radius:8px;border-left:4px solid {_combined_color}'>
                    <div style='color:#aaa;font-size:11px'>FINAL SIGNAL</div>
                    <div style='font-size:16px;font-weight:bold;color:{_combined_color}'>{_combined_signal}</div>
                    <div style='color:#ccc;font-size:12px'>{_combined_label}</div>
                    <div style='color:#999;font-size:11px'>Sentiment: {_sent_verdict}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")

                # ---- Spike Score component breakdown ----
                st.markdown("### Spike Score Components")
                _comp_cols = st.columns(5)
                _comp_labels = ['Volume', 'OI Change', 'Straddle', 'IV', 'Pressure']
                _comp_scores = [_spike['vol_score'], _spike['oi_score'], _spike['straddle_score'], _spike['iv_score'], _spike['pressure_score']]
                _comp_flags = [_spike['volume_spike'], _spike['oi_spike'], _spike['straddle_spike'], _spike['iv_spike'], _spike['pressure_spike']]
                for _ci, (_col, _lbl, _sc, _fl) in enumerate(zip(_comp_cols, _comp_labels, _comp_scores, _comp_flags)):
                    with _col:
                        _c = '#00ff88' if _fl else '#888888'
                        st.markdown(f"""<div style='text-align:center;background:#222;padding:8px;border-radius:6px;border:1px solid {_c}'>
                        <div style='color:#aaa;font-size:10px'>{_lbl}</div>
                        <div style='font-size:20px;font-weight:bold;color:{_c}'>{_sc}/20</div>
                        <div style='font-size:10px;color:{_c}'>{'SPIKE' if _fl else 'Normal'}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("")

                # ---- Gamma values per strike ----
                st.markdown("### Gamma Map — ATM ±2 Strikes")
                _gv = _gamma['gamma_values']
                _gmap_keys = ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']
                _gmap_cols = st.columns(5)
                _max_gv = max(_gv.values()) if _gv else 1
                for _gi, (_gcol, _gk) in enumerate(zip(_gmap_cols, _gmap_keys)):
                    with _gcol:
                        _gval = _gv.get(_gk, 0)
                        _gbar = int(_gval / _max_gv * 100) if _max_gv > 0 else 0
                        _gc = '#ffaa00' if _gk == 'ATM' else '#00aaff'
                        st.markdown(f"""<div style='text-align:center;background:#222;padding:8px;border-radius:6px'>
                        <div style='color:#aaa;font-size:10px'>{_gk}</div>
                        <div style='font-size:18px;font-weight:bold;color:{_gc}'>{_gval:.4f}</div>
                        <div style='background:#333;border-radius:3px;height:6px;margin-top:4px'>
                          <div style='background:{_gc};width:{_gbar}%;height:6px;border-radius:3px'></div>
                        </div>
                        </div>""", unsafe_allow_html=True)

                # ---- Expiry Day Intelligence Panel (visible when DTE ≤ 1) ----
                if _expiry_intel['active']:
                    st.markdown("---")
                    st.markdown("### ⚡ Expiry Control Center")
                    _ei = _expiry_intel
                    _eic1, _eic2, _eic3, _eic4 = st.columns(4)
                    with _eic1:
                        st.metric("Market Type", _ei['market_type'])
                        st.metric("ATM Straddle", f"₹{_ei['atm_straddle']:.0f}")
                        st.metric("Straddle ROC", f"{_ei['straddle_roc']:+.2f}%")
                    with _eic2:
                        st.metric("Max Pain", f"₹{_ei['max_pain']:.0f}" if _ei['max_pain'] else "—")
                        st.metric("Gamma Flip", f"₹{_ei['gamma_flip']:.0f}" if _ei['gamma_flip'] else "—")
                        st.metric("Highest Gamma Strike", f"₹{_ei['highest_gamma_strike']:.0f}" if _ei['highest_gamma_strike'] else "—")
                    with _eic3:
                        st.metric("Entry Support", f"₹{_ei['entry_support']:.0f}" if _ei['entry_support'] else "—")
                        st.metric("Entry Resistance", f"₹{_ei['entry_resistance']:.0f}" if _ei['entry_resistance'] else "—")
                        st.metric("Expiry Score", f"{_ei['expiry_score']:.1f}")
                    with _eic4:
                        st.metric("Breakout Level", f"₹{_ei['breakout_level']:.0f}" if _ei['breakout_level'] else "—")
                        st.metric("Breakdown Level", f"₹{_ei['breakdown_level']:.0f}" if _ei['breakdown_level'] else "—")
                        _es_color = '#ff4444' if _ei['expiry_score'] >= 85 else ('#ffaa00' if _ei['expiry_score'] >= 70 else '#aaaaaa')
                        st.markdown(f"<span style='color:{_es_color};font-weight:bold'>Signal: {_ei['expiry_signal']}</span>", unsafe_allow_html=True)

                    _oi_col, _trap_col = st.columns(2)
                    with _oi_col:
                        if _ei['oi_shift_signal']:
                            st.info(f"OI Shift: **{_ei['oi_shift_signal']}**")
                        if _ei['max_pain_signal']:
                            st.caption(f"Max Pain: {_ei['max_pain_signal']}")
                    with _trap_col:
                        if _ei['expiry_trap']:
                            st.warning(f"TRAP DETECTED: **{_ei['expiry_trap']}**")

                # ---- Charts ----
                st.markdown("---")
                st.markdown("### Charts")
                _chart_c1, _chart_c2 = st.columns(2)

                with _chart_c1:
                    # Spike Score History
                    if len(st.session_state.spike_history) >= 2:
                        _sh_times = []
                        _sh_scores = []
                        _sh_interval_count = 0
                        for _snap_i, _snap_df in enumerate(st.session_state.spike_history):
                            _sh_interval_count += 1
                            _sh_times.append(f"T-{len(st.session_state.spike_history) - _snap_i}")
                            _snap_atm_r = _snap_df[_snap_df['Strike'] == _mae_atm] if 'Strike' in _snap_df.columns else pd.DataFrame()
                            # Recompute spike for chart (simplified: vol ratio only for speed)
                            _sh_scores.append(0)

                        # Build proper time-stamped history from gamma_seq_history (which has timestamps)
                        _gsh = st.session_state.gamma_seq_history
                        _gsh_times = [s.get('time', f'T-{i}') for i, s in enumerate(_gsh)]
                        _gsh_atm = [_safe(s.get('ATM', 0)) for s in _gsh]

                        _fig_gs = go.Figure()
                        _fig_gs.add_trace(go.Scatter(
                            x=list(range(len(_gsh_atm))), y=_gsh_atm,
                            mode='lines+markers', name='ATM Gamma',
                            line=dict(color='#ffaa00', width=2),
                            marker=dict(size=4),
                            text=_gsh_times, hovertemplate='%{text}<br>ATM Gamma: %{y:.4f}'
                        ))
                        _fig_gs.update_layout(
                            title='ATM Gamma Over Time',
                            template='plotly_dark', height=260,
                            margin=dict(l=40, r=20, t=40, b=30),
                            xaxis=dict(title='Snapshot', showticklabels=False),
                            yaxis=dict(title='Gamma'),
                            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                        )
                        st.plotly_chart(_fig_gs, use_container_width=True)
                    else:
                        st.info("Building Spike/Gamma history...")

                with _chart_c2:
                    # Gamma Sequence Map — 5 strikes over time (last snapshot)
                    _gv_keys = ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']
                    _gv_vals = [_gv.get(k, 0) for k in _gv_keys]
                    _gmap_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']
                    _fig_gmap = go.Figure()
                    _fig_gmap.add_trace(go.Bar(
                        x=_gv_keys, y=_gv_vals,
                        marker_color=_gmap_colors,
                        name='Gamma by Strike',
                        text=[f'{v:.4f}' for v in _gv_vals],
                        textposition='outside'
                    ))
                    _fig_gmap.update_layout(
                        title=f'Gamma Sequence Map — Pattern: {_gamma["pattern"]}',
                        template='plotly_dark', height=260,
                        margin=dict(l=40, r=20, t=50, b=30),
                        xaxis=dict(title='Strike Position'),
                        yaxis=dict(title='Gamma Value'),
                        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                    )
                    st.plotly_chart(_fig_gmap, use_container_width=True)

                # Expiry Spike Score chart (3rd chart, full width if active)
                if _expiry_spike['active'] and len(st.session_state.expiry_spike_history) >= 2:
                    _esh_scores = []
                    _esh_times = [f'T-{i}' for i in range(len(st.session_state.gamma_seq_history))]
                    _gsh_data = st.session_state.gamma_seq_history
                    for _si, _snap_df in enumerate(st.session_state.expiry_spike_history):
                        _esh_snap = calculate_expiry_spike_score(_snap_df, _mae_atm, _mae_expiry, st.session_state.expiry_spike_history[:_si])
                        _esh_scores.append(_esh_snap['expiry_spike_score'])

                    _fig_esh = go.Figure()
                    _fig_esh.add_trace(go.Scatter(
                        x=list(range(len(_esh_scores))), y=_esh_scores,
                        mode='lines+markers+text', name='Expiry Spike Score',
                        line=dict(color='#ff6600', width=2),
                        marker=dict(size=4),
                        fill='tozeroy', fillcolor='rgba(255,102,0,0.1)'
                    ))
                    _fig_esh.add_hline(y=80, line_dash='dot', line_color='#ff4444', annotation_text='Explosion 80')
                    _fig_esh.add_hline(y=70, line_dash='dot', line_color='#ffaa00', annotation_text='Breakout 70')
                    _fig_esh.add_hline(y=40, line_dash='dot', line_color='#aaaaaa', annotation_text='Build-up 40')
                    _fig_esh.update_layout(
                        title='Expiry Spike Score History',
                        template='plotly_dark', height=260,
                        margin=dict(l=40, r=80, t=40, b=30),
                        xaxis=dict(title='Snapshot', showticklabels=False),
                        yaxis=dict(title='Expiry Spike Score', range=[0, 110]),
                        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                    )
                    st.plotly_chart(_fig_esh, use_container_width=True)

                # Gamma Sequence over time (multi-strike)
                if len(st.session_state.gamma_seq_history) >= 3:
                    _gsh_all = st.session_state.gamma_seq_history
                    _fig_gsh = go.Figure()
                    _gsh_colors_map = {'ATM-2': '#ff44ff', 'ATM-1': '#cc44cc', 'ATM': '#ffaa00', 'ATM+1': '#00aaff', 'ATM+2': '#0088dd'}
                    for _gsk in _gv_keys:
                        _gsk_vals = [_safe(s.get(_gsk, 0)) for s in _gsh_all]
                        _fig_gsh.add_trace(go.Scatter(
                            x=list(range(len(_gsk_vals))), y=_gsk_vals,
                            mode='lines', name=_gsk,
                            line=dict(color=_gsh_colors_map.get(_gsk, '#aaa'), width=2)
                        ))
                    _fig_gsh.update_layout(
                        title='Gamma Sequence Over Time (ATM ±2)',
                        template='plotly_dark', height=280,
                        margin=dict(l=40, r=20, t=40, b=30),
                        xaxis=dict(title='Snapshot', showticklabels=False),
                        yaxis=dict(title='Gamma'),
                        showlegend=True, legend=dict(orientation='h', y=-0.3),
                        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                    )
                    st.plotly_chart(_fig_gsh, use_container_width=True)

                # Clear button
                _mae_btn_col1, _mae_btn_col2 = st.columns([4, 1])
                with _mae_btn_col2:
                    if st.button("🗑️ Clear MAE History", key="clr_mae"):
                        st.session_state.spike_history = []
                        st.session_state.expiry_spike_history = []
                        st.session_state.gamma_seq_history = []
                        st.session_state.expiry_intel_history = []
                        st.rerun()

            else:
                st.info("Option chain data required for Market Acceleration Engine.")
        except Exception as _mae_e:
            st.warning(f"Market Acceleration Engine unavailable: {str(_mae_e)}")

    # ===== CANDLESTICK INTELLIGENCE ENGINE =====
    st.markdown("---")
    with st.expander("🕯️ Candlestick Intelligence Engine — Reversal & Continuation Signals", expanded=False):
        try:
            if option_data and option_data.get('underlying') and not df.empty:
                _cie_underlying = option_data['underlying']
                _cie_df_summary = option_data.get('df_summary')
                _cie_straddle_hist = st.session_state.straddle_history

                # Determine expiry day (DTE ≤ 2)
                try:
                    _cie_expiry_str = option_data.get('expiry', '')
                    _cie_expiry_dt  = datetime.strptime(_cie_expiry_str, "%Y-%m-%d") if _cie_expiry_str else None
                    _cie_dte = (_cie_expiry_dt.date() - datetime.now(pytz.timezone('Asia/Kolkata')).date()).days if _cie_expiry_dt else 999
                    _cie_is_expiry = _cie_dte <= 2
                except Exception:
                    _cie_is_expiry = False

                # Run engine
                _cie_signals = run_candlestick_intelligence_engine(
                    df, option_data, _cie_straddle_hist, _cie_underlying, is_expiry=_cie_is_expiry
                )

                # Build S/R for chart display
                _cie_sup, _cie_res = _cie_detect_swing_sr(df)

                # ── Header metrics ───────────────────────────────────────────
                _cie_hc1, _cie_hc2, _cie_hc3, _cie_hc4 = st.columns(4)
                _total_sigs   = len(_cie_signals)
                _buy_sigs     = sum(1 for s in _cie_signals if s['direction'] == 'BUY')
                _sell_sigs    = sum(1 for s in _cie_signals if s['direction'] == 'SELL')
                _inst_sigs    = sum(1 for s in _cie_signals if s.get('signal_strength') == 'INSTITUTIONAL')
                with _cie_hc1:
                    st.markdown(f"""<div style='background:#1e1e1e;padding:12px;border-radius:8px;border-left:4px solid #00aaff'>
                    <div style='color:#aaa;font-size:11px'>ACTIVE SIGNALS</div>
                    <div style='font-size:28px;font-weight:bold;color:#00aaff'>{_total_sigs}</div>
                    <div style='color:#ccc;font-size:12px'>Patterns Detected</div>
                    </div>""", unsafe_allow_html=True)
                with _cie_hc2:
                    st.markdown(f"""<div style='background:#1e1e1e;padding:12px;border-radius:8px;border-left:4px solid #00ff88'>
                    <div style='color:#aaa;font-size:11px'>BUY SIGNALS</div>
                    <div style='font-size:28px;font-weight:bold;color:#00ff88'>{_buy_sigs}</div>
                    <div style='color:#ccc;font-size:12px'>Bullish Patterns</div>
                    </div>""", unsafe_allow_html=True)
                with _cie_hc3:
                    st.markdown(f"""<div style='background:#1e1e1e;padding:12px;border-radius:8px;border-left:4px solid #ff4444'>
                    <div style='color:#aaa;font-size:11px'>SELL SIGNALS</div>
                    <div style='font-size:28px;font-weight:bold;color:#ff4444'>{_sell_sigs}</div>
                    <div style='color:#ccc;font-size:12px'>Bearish Patterns</div>
                    </div>""", unsafe_allow_html=True)
                with _cie_hc4:
                    _inst_clr = '#ff6600' if _inst_sigs > 0 else '#888'
                    st.markdown(f"""<div style='background:#1e1e1e;padding:12px;border-radius:8px;border-left:4px solid {_inst_clr}'>
                    <div style='color:#aaa;font-size:11px'>INSTITUTIONAL</div>
                    <div style='font-size:28px;font-weight:bold;color:{_inst_clr}'>{_inst_sigs}</div>
                    <div style='color:#ccc;font-size:12px'>Score ≥ 85</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")
                if _cie_is_expiry:
                    st.warning("⚡ **EXPIRY DAY MODE** — Volatility filter relaxed. Prioritising wick rejections, engulfing & marubozu patterns.")

                # ── Tabs ─────────────────────────────────────────────────────
                _cie_tab1, _cie_tab2, _cie_tab3, _cie_tab4 = st.tabs(
                    ["📋 Signals", "📈 Price + S/R Chart", "🌊 Options Flow Charts", "📜 Signal History"]
                )

                # ── TAB 1: Signals ────────────────────────────────────────
                with _cie_tab1:
                    if not _cie_signals:
                        st.info("No qualifying signals at this time. Patterns require:\n"
                                "• Near S/R level (within 0.2%)\n"
                                "• Straddle expanding (ROC ≥ 0.5%)\n"
                                "• Options flow confirmation\n"
                                "• Confidence ≥ 40")
                        # ── Today's Signal History ────────────────────────────
                        _today_ist = datetime.now(pytz.timezone('Asia/Kolkata')).date()
                        _today_hist = [
                            _h for _h in st.session_state.cie_signal_history
                            if hasattr(_h.get('time'), 'date') and _h['time'].date() == _today_ist
                        ]
                        if _today_hist:
                            st.markdown("#### 📜 Today's Signal History")
                            _today_rows = []
                            for _h in reversed(_today_hist):
                                _dir_sym = '🟢 BUY' if _h.get('direction') == 'BUY' else '🔴 SELL'
                                _conf_val = _h.get('confidence', 0)
                                _today_rows.append({
                                    'Time': _h['time'].strftime('%H:%M:%S'),
                                    'Pattern': _h.get('pattern', ''),
                                    'Direction': _dir_sym,
                                    'Spot': f"₹{_h.get('spot', 0):.0f}",
                                    'Price': f"₹{_h.get('price', 0):.0f}",
                                    'Level': f"₹{_h.get('level', 0):.0f}" if _h.get('level') else 'N/A',
                                    'Type': _h.get('level_type', ''),
                                    'Confidence': _conf_val,
                                    'Strength': _h.get('strength', ''),
                                })
                            st.dataframe(pd.DataFrame(_today_rows), use_container_width=True, hide_index=True)
                        else:
                            st.caption("No signals recorded today yet.")
                    else:
                        _cie_now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
                        for _sig in _cie_signals:
                            _dir      = _sig['direction']
                            _pat      = _sig['pattern']
                            _cat      = _sig.get('category', '')
                            _price    = _sig['price']
                            _level    = _sig.get('level')
                            _lvl_type = _sig.get('level_type', '')
                            _conf     = _sig['confidence']
                            _strength = _sig['signal_strength']
                            _sclr     = _sig['strength_color']
                            _opt      = _sig.get('options_details', {})
                            _vol_sp   = '⚡ Vol Spike' if _sig.get('vol_spike') else ''
                            _dir_emoji = '🟢' if _dir == 'BUY' else '🔴'
                            _dir_clr   = '#00ff88' if _dir == 'BUY' else '#ff4444'
                            _level_str = f"₹{_level:.0f}" if _level else 'N/A'
                            _dist_str  = f"{(_sig.get('dist_pct') or 0)*100:.2f}%"
                            _pcr_str   = f"PCR {_opt.get('pcr', 'N/A')}" if 'pcr' in _opt else ''
                            _gamma_str = _opt.get('gamma_conf', _opt.get('gamma_sr', ''))
                            _oi_str    = _opt.get('oi_conf', '')
                            _delta_str = _opt.get('delta_conf', '')
                            _strad_str = f"Straddle ROC {_opt.get('straddle_roc','N/A')}%" if 'straddle_roc' in _opt else ''
                            _conf_bar_w = int(_conf)
                            _conf_clr  = '#ff6600' if _conf >= 85 else ('#ffaa00' if _conf >= 70 else '#00aaff')

                            st.markdown(f"""
<div style='background:#1a1a2e;border:1px solid {_dir_clr};border-radius:10px;padding:16px;margin-bottom:12px'>
<div style='display:flex;justify-content:space-between;align-items:center'>
  <span style='font-size:20px;font-weight:bold;color:{_dir_clr}'>{_dir_emoji} {_dir} — {_pat}</span>
  <span style='background:{_sclr};color:#000;font-size:11px;font-weight:bold;padding:4px 10px;border-radius:12px'>{_strength} {_vol_sp}</span>
</div>
<div style='color:#aaa;font-size:12px;margin-top:4px'>{_cat} · {_lvl_type} · Spot: ₹{_cie_underlying:.0f}</div>
<div style='margin:10px 0 4px;display:flex;gap:16px;flex-wrap:wrap'>
  <span style='color:#fff'>Price: <b>₹{_price:.0f}</b></span>
  <span style='color:#aaa'>{_lvl_type}: <b style="color:{_dir_clr}">{_level_str}</b></span>
  <span style='color:#aaa'>Distance: {_dist_str}</span>
</div>
<div style='margin:6px 0;font-size:12px;color:#ccc'>
  {_pcr_str} &nbsp;|&nbsp; {_gamma_str} &nbsp;|&nbsp; {_oi_str} &nbsp;|&nbsp; {_delta_str} &nbsp;|&nbsp; {_strad_str}
</div>
<div style='margin-top:8px'>
  <div style='font-size:11px;color:#aaa;margin-bottom:2px'>Confidence Score</div>
  <div style='background:#333;border-radius:4px;height:10px;width:100%'>
    <div style='background:{_conf_clr};height:10px;width:{_conf_bar_w}%;border-radius:4px'></div>
  </div>
  <div style='color:{_conf_clr};font-weight:bold;font-size:14px;margin-top:2px'>{_conf}/100 — {_strength}</div>
</div>
</div>""", unsafe_allow_html=True)

                            # Telegram alert (throttled per pattern, 5 min cooldown)
                            if enable_signals:
                                _cie_last_ts = st.session_state.cie_last_alert.get(_pat)
                                _cie_should_alert = (
                                    _cie_last_ts is None or
                                    (_cie_now_ist - _cie_last_ts).total_seconds() > 300
                                )
                                if _cie_should_alert and _conf >= 70:
                                    _cie_tg_msg = (
                                        f"{_dir_emoji} <b>CIE {_dir} SIGNAL — {_pat}</b>\n\n"
                                        f"Spot: ₹{_cie_underlying:.0f}\n"
                                        f"Pattern: {_pat} ({_cat})\n"
                                        f"Price: ₹{_price:.0f}\n"
                                        f"{_lvl_type}: {_level_str} (dist {_dist_str})\n"
                                        f"{_pcr_str}\n"
                                        f"{_gamma_str}\n"
                                        f"{_oi_str}\n"
                                        f"Straddle: {'Expanding ✅' if _opt.get('straddle_expanding') else 'Flat'}\n"
                                        f"Confidence: {_conf}/100 — {_strength}\n"
                                        f"Time: {_cie_now_ist.strftime('%H:%M:%S')} IST"
                                    )
                                    send_telegram_message_sync(_cie_tg_msg)
                                    st.session_state.cie_last_alert[_pat] = _cie_now_ist
                                    # Store in history
                                    st.session_state.cie_signal_history.append({
                                        'time': _cie_now_ist,
                                        'pattern': _pat,
                                        'direction': _dir,
                                        'price': _price,
                                        'level': _level,
                                        'level_type': _lvl_type,
                                        'confidence': _conf,
                                        'strength': _strength,
                                        'spot': _cie_underlying,
                                    })
                                    if len(st.session_state.cie_signal_history) > 100:
                                        st.session_state.cie_signal_history = st.session_state.cie_signal_history[-100:]

                # ── TAB 2: Price + S/R Chart ──────────────────────────────
                with _cie_tab2:
                    try:
                        _cie_chart_df = df.tail(80).copy().reset_index(drop=True)
                        _cie_fig = go.Figure()

                        # Candlestick
                        _cie_fig.add_trace(go.Candlestick(
                            x=_cie_chart_df['datetime'],
                            open=_cie_chart_df['open'],
                            high=_cie_chart_df['high'],
                            low=_cie_chart_df['low'],
                            close=_cie_chart_df['close'],
                            name='NIFTY 5m',
                            increasing_line_color='#00ff88',
                            decreasing_line_color='#ff4444',
                        ))

                        # Support levels
                        for _sl in _cie_sup[-6:]:
                            _cie_fig.add_hline(y=_sl, line_dash='dash', line_color='rgba(0,255,136,0.4)',
                                               line_width=1, annotation_text=f'S {_sl:.0f}',
                                               annotation_font_color='#00ff88', annotation_font_size=9)
                        # Resistance levels
                        for _rl in _cie_res[-6:]:
                            _cie_fig.add_hline(y=_rl, line_dash='dash', line_color='rgba(255,68,68,0.4)',
                                               line_width=1, annotation_text=f'R {_rl:.0f}',
                                               annotation_font_color='#ff4444', annotation_font_size=9)

                        # Highlight signal candles
                        for _sig in _cie_signals:
                            _sig_idx = _sig.get('index', -1)
                            _chart_offset = len(df) - len(_cie_chart_df)
                            _local_idx = _sig_idx - _chart_offset
                            if 0 <= _local_idx < len(_cie_chart_df):
                                _sc = _cie_chart_df.iloc[_local_idx]
                                _sig_clr = 'rgba(0,255,136,0.25)' if _sig['direction'] == 'BUY' else 'rgba(255,68,68,0.25)'
                                _cie_fig.add_vrect(
                                    x0=_sc['datetime'], x1=_sc['datetime'],
                                    fillcolor=_sig_clr, opacity=0.6, line_width=3,
                                    line_color=_sig['strength_color'],
                                    annotation_text=_sig['pattern'][:12],
                                    annotation_position="top left",
                                    annotation_font_size=9,
                                    annotation_font_color=_sig['strength_color'],
                                )

                        # Spot price line
                        _cie_fig.add_hline(y=_cie_underlying, line_dash='dot', line_color='#ffffff',
                                           line_width=1.5, annotation_text=f'Spot {_cie_underlying:.0f}',
                                           annotation_font_color='#fff', annotation_font_size=10)

                        _cie_fig.update_layout(
                            title='NIFTY 5-Min — Candlestick Intelligence Engine S/R Map',
                            template='plotly_dark', height=480,
                            xaxis_rangeslider_visible=False,
                            margin=dict(l=10, r=80, t=50, b=30),
                            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                            xaxis=dict(type='category', tickangle=-45, nticks=15,
                                       tickfont=dict(size=8)),
                        )
                        st.plotly_chart(_cie_fig, use_container_width=True)

                        # Support / Resistance level table
                        st.markdown("**Detected S/R Levels**")
                        _cie_sr_rows = (
                            [{'Level': f'₹{r:.0f}', 'Type': '🔴 Resistance', 'Zone': 'R'} for r in sorted(_cie_res, reverse=True)[-6:]] +
                            [{'Level': f'₹{s:.0f}', 'Type': '🟢 Support', 'Zone': 'S'} for s in sorted(_cie_sup, reverse=True)[:6]]
                        )
                        if _cie_sr_rows:
                            st.dataframe(pd.DataFrame(_cie_sr_rows), use_container_width=True, hide_index=True)

                        # ── Detected Reversal Candle Table ─────────────────
                        st.markdown("**Reversal & Continuation Candles Detected**")
                        if _cie_signals:
                            _cie_candle_rows = []
                            for _sig in _cie_signals:
                                _ts = _sig.get('time')
                                if hasattr(_ts, 'strftime'):
                                    _ts_str = _ts.strftime('%H:%M')
                                elif hasattr(_ts, 'to_pydatetime'):
                                    _ts_str = _ts.to_pydatetime().strftime('%H:%M')
                                else:
                                    # Fallback: derive from df index
                                    _si = _sig.get('index', -1)
                                    if 0 <= _si < len(df) and 'datetime' in df.columns:
                                        _ts_raw = df['datetime'].iloc[_si]
                                        _ts_str = _ts_raw.strftime('%H:%M') if hasattr(_ts_raw, 'strftime') else str(_ts_raw)[:5]
                                    else:
                                        _ts_str = 'N/A'
                                _cie_candle_rows.append({
                                    'Time': _ts_str,
                                    'Pattern': _sig['pattern'],
                                    'Signal': _sig['direction'],
                                    'Category': _sig.get('category', ''),
                                    'Price': f"₹{_sig['price']:.0f}",
                                    'Level Type': _sig.get('level_type', ''),
                                    'Level': f"₹{_sig['level']:.0f}" if _sig.get('level') else 'N/A',
                                    'Dist %': f"{(_sig.get('dist_pct') or 0)*100:.2f}%",
                                    'Vol Spike': '⚡ Yes' if _sig.get('vol_spike') else 'No',
                                    'Confidence': f"{_sig['confidence']}/100",
                                    'Strength': _sig.get('signal_strength', ''),
                                })
                            _cie_candle_df = pd.DataFrame(_cie_candle_rows)
                            # Color rows by direction
                            def _cie_color_row(row):
                                clr = 'background-color: rgba(0,255,136,0.08)' if row['Signal'] == 'BUY' else 'background-color: rgba(255,68,68,0.08)'
                                return [clr] * len(row)
                            st.dataframe(
                                _cie_candle_df.style.apply(_cie_color_row, axis=1),
                                use_container_width=True, hide_index=True
                            )
                        else:
                            st.info("No patterns detected near S/R levels.")
                    except Exception as _cie_chart_err:
                        st.warning(f"Chart error: {str(_cie_chart_err)[:80]}")

                # ── TAB 3: Options Flow Charts ────────────────────────────
                with _cie_tab3:
                    _cie_flow_c1, _cie_flow_c2 = st.columns(2)

                    # IV Skew history
                    with _cie_flow_c1:
                        _cie_iv_hist = st.session_state.get('iv_skew_history', [])
                        if len(_cie_iv_hist) >= 3:
                            _cie_iv_df = pd.DataFrame(_cie_iv_hist)
                            _cie_iv_fig = go.Figure()
                            if 'iv_skew' in _cie_iv_df.columns:
                                _cie_iv_fig.add_trace(go.Scatter(
                                    x=list(range(len(_cie_iv_df))),
                                    y=_cie_iv_df['iv_skew'],
                                    mode='lines', name='IV Skew (PE/CE)',
                                    line=dict(color='#ff9900', width=2),
                                ))
                            _cie_iv_fig.add_hline(y=1.10, line_dash='dot', line_color='#ff4444',
                                                  annotation_text='Bearish 1.10')
                            _cie_iv_fig.add_hline(y=0.90, line_dash='dot', line_color='#00ff88',
                                                  annotation_text='Bullish 0.90')
                            _cie_iv_fig.update_layout(
                                title='IV Skew (PE IV / CE IV)',
                                template='plotly_dark', height=260,
                                margin=dict(l=30, r=30, t=40, b=20),
                                plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                                xaxis=dict(title='Snapshot', showticklabels=False),
                                yaxis=dict(title='Skew'),
                            )
                            st.plotly_chart(_cie_iv_fig, use_container_width=True)
                        else:
                            st.info("IV Skew history building…")

                    # Bid vs Ask Pressure
                    with _cie_flow_c2:
                        _cie_pres_hist = st.session_state.get('pressure_history', [])
                        if len(_cie_pres_hist) >= 3:
                            _cie_pres_df = pd.DataFrame(_cie_pres_hist)
                            _cie_pres_fig = go.Figure()
                            if 'net_pressure' in _cie_pres_df.columns:
                                _cie_pres_clrs = ['#00ff88' if v > 0 else '#ff4444'
                                                  for v in _cie_pres_df['net_pressure']]
                                _cie_pres_fig.add_trace(go.Bar(
                                    x=list(range(len(_cie_pres_df))),
                                    y=_cie_pres_df['net_pressure'],
                                    marker_color=_cie_pres_clrs,
                                    name='Net Bid/Ask Pressure',
                                ))
                            _cie_pres_fig.add_hline(y=0.15, line_dash='dot', line_color='#00ff88',
                                                    annotation_text='Bull 0.15')
                            _cie_pres_fig.add_hline(y=-0.15, line_dash='dot', line_color='#ff4444',
                                                    annotation_text='Bear -0.15')
                            _cie_pres_fig.update_layout(
                                title='Net Bid/Ask Pressure',
                                template='plotly_dark', height=260,
                                margin=dict(l=30, r=30, t=40, b=20),
                                plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                                xaxis=dict(title='Snapshot', showticklabels=False),
                                yaxis=dict(title='Pressure'),
                            )
                            st.plotly_chart(_cie_pres_fig, use_container_width=True)
                        else:
                            st.info("Pressure history building…")

                    # Straddle movement
                    _cie_strad_hist = st.session_state.get('straddle_history', [])
                    if len(_cie_strad_hist) >= 3:
                        _cie_strad_df = pd.DataFrame(_cie_strad_hist)
                        _cie_strad_fig = go.Figure()
                        if 'straddle' in _cie_strad_df.columns:
                            _cie_strad_fig.add_trace(go.Scatter(
                                x=list(range(len(_cie_strad_df))),
                                y=_cie_strad_df['straddle'],
                                mode='lines+markers', name='ATM Straddle',
                                line=dict(color='#ff9900', width=2),
                                marker=dict(size=3),
                            ))
                        _cie_strad_fig.update_layout(
                            title='ATM Straddle Movement',
                            template='plotly_dark', height=240,
                            margin=dict(l=30, r=30, t=40, b=20),
                            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                            xaxis=dict(title='Snapshot', showticklabels=False),
                            yaxis=dict(title='Straddle ₹'),
                        )
                        st.plotly_chart(_cie_strad_fig, use_container_width=True)

                    # Gamma Exposure history
                    _cie_gex_hist = st.session_state.get('total_gex_history', [])
                    if len(_cie_gex_hist) >= 3:
                        _cie_gex_df = pd.DataFrame(_cie_gex_hist)
                        _cie_gex_fig = go.Figure()
                        if 'total_gex' in _cie_gex_df.columns:
                            _cie_gex_clrs = ['#00ff88' if v >= 0 else '#ff4444'
                                             for v in _cie_gex_df['total_gex']]
                            _cie_gex_fig.add_trace(go.Bar(
                                x=list(range(len(_cie_gex_df))),
                                y=_cie_gex_df['total_gex'],
                                marker_color=_cie_gex_clrs,
                                name='Net GEX (L)',
                            ))
                        _cie_gex_fig.update_layout(
                            title='Gamma Exposure (GEX) Over Time',
                            template='plotly_dark', height=240,
                            margin=dict(l=30, r=30, t=40, b=20),
                            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                            xaxis=dict(title='Snapshot', showticklabels=False),
                            yaxis=dict(title='GEX (L)'),
                        )
                        st.plotly_chart(_cie_gex_fig, use_container_width=True)

                # ── TAB 4: Signal History ─────────────────────────────────
                with _cie_tab4:
                    _cie_hist = st.session_state.cie_signal_history
                    if not _cie_hist:
                        st.info("No signals have been alerted yet this session.")
                    else:
                        _cie_hist_rows = []
                        for _h in reversed(_cie_hist[-50:]):
                            _cie_hist_rows.append({
                                'Time': _h['time'].strftime('%H:%M:%S') if hasattr(_h.get('time'), 'strftime') else str(_h.get('time', '')),
                                'Pattern': _h.get('pattern', ''),
                                'Direction': _h.get('direction', ''),
                                'Price': f"₹{_h.get('price', 0):.0f}",
                                'Spot': f"₹{_h.get('spot', 0):.0f}",
                                'Level': f"₹{_h.get('level', 0):.0f}" if _h.get('level') else 'N/A',
                                'Type': _h.get('level_type', ''),
                                'Confidence': _h.get('confidence', 0),
                                'Strength': _h.get('strength', ''),
                            })
                        st.dataframe(pd.DataFrame(_cie_hist_rows), use_container_width=True, hide_index=True)

                    _cie_bcol1, _cie_bcol2 = st.columns([4, 1])
                    with _cie_bcol2:
                        if st.button("🗑️ Clear CIE History", key="clr_cie"):
                            st.session_state.cie_signal_history = []
                            st.session_state.cie_last_alert = {}
                            st.rerun()

            else:
                st.info("Option chain data required for Candlestick Intelligence Engine.")
        except Exception as _cie_err:
            st.warning(f"Candlestick Intelligence Engine error: {str(_cie_err)}")

    # ===== FUTURES MARKET ANALYSIS ENGINE =====
    if option_data and option_data.get('underlying') and not df.empty:
        try:
            with st.expander("🔮 Futures Market Analysis Engine", expanded=False):
                show_futures_analysis_engine(df, option_data, current_price)
        except Exception as _fae_err:
            st.warning(f"Futures Analysis Engine error: {str(_fae_err)}")

    # ===== SECTOR ROTATION ANALYSIS ENGINE =====
    with st.expander("🔄 Sector Rotation Analysis Engine", expanded=False):
        try:
            show_sector_rotation_engine()
        except Exception as _sre_err:
            st.warning(f"Sector Rotation Engine error: {str(_sre_err)}")

    # ===== FII / DII ACTIVITY ANALYSIS ENGINE =====
    with st.expander("🏦 FII & DII Activity Analysis Engine", expanded=False):
        try:
            show_fii_dii_analysis(
                df=df if not df.empty else None,
                option_data=option_data,
                current_price=current_price,
            )
        except Exception as _fii_err:
            st.warning(f"FII/DII Analysis Engine error: {str(_fii_err)}")

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
