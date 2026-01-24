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
from scipy import stats
import plotly.express as px
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="Nifty Trading & Options Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 30 seconds
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
    .atm-row {
        background-color: #FFD700 !important;
        font-weight: bold !important;
    }
    .seller-positive {
        background-color: #004d00 !important;
        color: white !important;
        font-weight: bold !important;
    }
    .seller-caution {
        background-color: #4d4d00 !important;
        color: white !important;
        font-weight: bold !important;
    }
    .seller-negative {
        background-color: #660000 !important;
        color: white !important;
        font-weight: bold !important;
    }
    .depth-support {
        background-color: rgba(0, 100, 0, 0.3) !important;
        border-left: 3px solid green !important;
    }
    .depth-resistance {
        background-color: rgba(100, 0, 0, 0.3) !important;
        border-left: 3px solid red !important;
    }
    .depth-neutral {
        background-color: rgba(100, 100, 0, 0.3) !important;
        border-left: 3px solid yellow !important;
    }
    .market-insights {
        background-color: #1a1a2e;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #4d4dff;
    }
    .insight-header {
        color: #4d4dff;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .insight-point {
        margin: 8px 0;
        padding-left: 10px;
        border-left: 2px solid #333;
    }
    .trap-indicator {
        background-color: #2a2a4d;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #4d4dff;
    }
    .api-warning {
        background-color: #660000;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
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
        
    # Enhanced Telegram Configuration
    try:
        TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
        
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
        
        if TELEGRAM_CHAT_ID and isinstance(TELEGRAM_CHAT_ID, (int, float)):
            TELEGRAM_CHAT_ID = str(int(TELEGRAM_CHAT_ID))
            
    except Exception as e:
        st.error(f"Telegram config error: {e}")
        TELEGRAM_BOT_TOKEN = ""
        TELEGRAM_CHAT_ID = ""
    
except Exception:
    DHAN_CLIENT_ID = ""
    DHAN_ACCESS_TOKEN = ""
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"

# Add refresh counter to session state
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0

# Add cache for API calls to prevent rate limiting
if 'api_cache' not in st.session_state:
    st.session_state.api_cache = {}
if 'last_api_call' not in st.session_state:
    st.session_state.last_api_call = {}

# Rate limiting configuration
API_COOLDOWN = 2  # seconds between API calls

def rate_limit_check(api_name):
    """Check if we need to wait before making another API call"""
    now = time.time()
    if api_name in st.session_state.last_api_call:
        time_since_last = now - st.session_state.last_api_call[api_name]
        if time_since_last < API_COOLDOWN:
            time.sleep(API_COOLDOWN - time_since_last)
    st.session_state.last_api_call[api_name] = time.time()

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
        
    def test_connection(self):
        """Test if Dhan API credentials are valid"""
        if not self.access_token or not self.client_id:
            return False, "Credentials not configured"
        
        try:
            # Try to get LTP data as a simple test
            ltp_data = self.get_ltp_data("13", "IDX_I")
            if ltp_data and 'data' in ltp_data:
                return True, "✅ Dhan API connection successful"
            elif ltp_data and 'error' in str(ltp_data).lower():
                return False, "❌ API Error: Invalid credentials"
            else:
                return False, "❌ No response from API"
        except Exception as e:
            return False, f"❌ Connection error: {str(e)}"
        
    def get_intraday_data(self, security_id="13", exchange_segment="IDX_I", instrument="INDEX", interval="1", days_back=1):
        """Get intraday historical data"""
        # Check cache first
        cache_key = f"intraday_{security_id}_{interval}_{days_back}"
        if cache_key in st.session_state.api_cache:
            cached_data = st.session_state.api_cache[cache_key]
            if time.time() - cached_data['timestamp'] < 60:  # Cache for 60 seconds
                return cached_data['data']
        
        rate_limit_check("intraday")
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
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Cache the result
                st.session_state.api_cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
                return data
            elif response.status_code == 401:
                st.error("❌ API Error 401: Invalid or expired credentials. Please check your Dhan API credentials.")
                return None
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def get_ltp_data(self, security_id="13", exchange_segment="IDX_I"):
        """Get Last Traded Price"""
        # Check cache first
        cache_key = f"ltp_{security_id}"
        if cache_key in st.session_state.api_cache:
            cached_data = st.session_state.api_cache[cache_key]
            if time.time() - cached_data['timestamp'] < 10:  # Cache for 10 seconds
                return cached_data['data']
        
        rate_limit_check("ltp")
        url = f"{self.base_url}/marketfeed/ltp"
        
        payload = {
            exchange_segment: [int(security_id)]
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Cache the result
                st.session_state.api_cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
                return data
            elif response.status_code == 401:
                return None
            else:
                return None
        except Exception as e:
            return None

    def get_option_chain(self, underlying_scrip, underlying_seg, expiry):
        """Get real option chain data"""
        # Check cache first
        cache_key = f"optionchain_{underlying_scrip}_{expiry}"
        if cache_key in st.session_state.api_cache:
            cached_data = st.session_state.api_cache[cache_key]
            if time.time() - cached_data['timestamp'] < 60:  # Cache for 60 seconds
                return cached_data['data']
        
        rate_limit_check("optionchain")
        url = f"{self.base_url}/optionchain"
        
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg,
            "Expiry": expiry
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                # Cache the result
                st.session_state.api_cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
                return data
            elif response.status_code == 401:
                st.error("❌ API Error 401: Invalid credentials for option chain")
                return None
            else:
                st.error(f"Option Chain API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching option chain: {str(e)}")
            return None

    def get_expiry_list(self, underlying_scrip, underlying_seg):
        """Get expiry list"""
        # Check cache first
        cache_key = f"expirylist_{underlying_scrip}"
        if cache_key in st.session_state.api_cache:
            cached_data = st.session_state.api_cache[cache_key]
            if time.time() - cached_data['timestamp'] < 300:  # Cache for 5 minutes
                return cached_data['data']
        
        rate_limit_check("expirylist")
        url = f"{self.base_url}/optionchain/expirylist"
        
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Cache the result
                st.session_state.api_cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
                return data
            elif response.status_code == 401:
                return None
            else:
                return None
        except Exception as e:
            return None

def calculate_exact_time_to_expiry(expiry_date_str):
    """Calculate exact time to expiry in years (days + hours)"""
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

def process_option_chain_data(api_data, expiry):
    """Process raw option chain API data into structured DataFrame"""
    if not api_data or 'data' not in api_data:
        return None, pd.DataFrame()
    
    data = api_data['data']
    underlying_price = data.get('last_price', 0)
    
    if 'oc' not in data:
        return underlying_price, pd.DataFrame()
    
    oc_data = data['oc']
    calls, puts = [], []
    
    for strike, strike_data in oc_data.items():
        if 'ce' in strike_data:
            ce_data = strike_data['ce']
            ce_data['strikePrice'] = float(strike)
            ce_data['optionType'] = 'CE'
            calls.append(ce_data)
        
        if 'pe' in strike_data:
            pe_data = strike_data['pe']
            pe_data['strikePrice'] = float(strike)
            pe_data['optionType'] = 'PE'
            puts.append(pe_data)
    
    if not calls and not puts:
        return underlying_price, pd.DataFrame()
    
    # Create DataFrames
    df_ce = pd.DataFrame(calls) if calls else pd.DataFrame()
    df_pe = pd.DataFrame(puts) if puts else pd.DataFrame()
    
    # Standardize column names
    def standardize_columns(df, suffix):
        column_mapping = {
            'last_price': f'lastPrice_{suffix}',
            'oi': f'openInterest_{suffix}',
            'previous_oi': f'previousOpenInterest_{suffix}',
            'implied_volatility': f'impliedVolatility_{suffix}',
            'top_ask_quantity': f'askQty_{suffix}',
            'top_bid_quantity': f'bidQty_{suffix}',
            'volume': f'totalTradedVolume_{suffix}',
            'changein_oi': f'changeinOpenInterest_{suffix}'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Ensure all required columns exist
        required_cols = [
            f'openInterest_{suffix}',
            f'impliedVolatility_{suffix}',
            f'lastPrice_{suffix}',
            f'askQty_{suffix}',
            f'bidQty_{suffix}',
            f'totalTradedVolume_{suffix}',
            f'changeinOpenInterest_{suffix}'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        
        return df
    
    if not df_ce.empty:
        df_ce = standardize_columns(df_ce, 'CE')
    
    if not df_pe.empty:
        df_pe = standardize_columns(df_pe, 'PE')
    
    # Merge call and put data
    if not df_ce.empty and not df_pe.empty:
        df = pd.merge(df_ce, df_pe, on='strikePrice', how='outer').sort_values('strikePrice')
    elif not df_ce.empty:
        df = df_ce.copy()
        # Add missing PE columns
        pe_cols = [col for col in required_cols if '_PE' in col]
        for col in pe_cols:
            df[col] = 0
    elif not df_pe.empty:
        df = df_pe.copy()
        # Add missing CE columns
        ce_cols = [col for col in required_cols if '_CE' in col]
        for col in ce_cols:
            df[col] = 0
    else:
        return underlying_price, pd.DataFrame()
    
    # Fill NaN values
    numeric_cols = [col for col in df.columns if col not in ['strikePrice', 'optionType_CE', 'optionType_PE']]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Calculate Greeks
    T = calculate_exact_time_to_expiry(expiry)
    r = 0.06
    
    # Ensure IV columns exist
    if 'impliedVolatility_CE' not in df.columns:
        df['impliedVolatility_CE'] = 15
    if 'impliedVolatility_PE' not in df.columns:
        df['impliedVolatility_PE'] = 15
    
    for idx, row in df.iterrows():
        strike = row['strikePrice']
        
        iv_ce = row.get('impliedVolatility_CE', 15)
        iv_pe = row.get('impliedVolatility_PE', 15)
        
        # Convert IV from percentage to decimal
        iv_ce = iv_ce / 100 if iv_ce > 1 else iv_ce
        iv_pe = iv_pe / 100 if iv_pe > 1 else iv_pe
        
        greeks_ce = calculate_greeks('CE', underlying_price, strike, T, r, iv_ce)
        greeks_pe = calculate_greeks('PE', underlying_price, strike, T, r, iv_pe)
        
        df.at[idx, 'Delta_CE'], df.at[idx, 'Gamma_CE'], df.at[idx, 'Vega_CE'], df.at[idx, 'Theta_CE'], df.at[idx, 'Rho_CE'] = greeks_ce
        df.at[idx, 'Delta_PE'], df.at[idx, 'Gamma_PE'], df.at[idx, 'Vega_PE'], df.at[idx, 'Theta_PE'], df.at[idx, 'Rho_PE'] = greeks_pe
    
    # Identify ATM strike
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying_price))
    
    # Filter to ATM ± 2 strikes for better display
    atm_plus_minus_2 = df[abs(df['strikePrice'] - atm_strike) <= 100]
    if not atm_plus_minus_2.empty:
        df = atm_plus_minus_2.copy()
    
    # Add zone classification
    df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying_price else 'OTM')
    
    # Add level classification based on OI
    def determine_level(row):
        ce_oi = row.get('openInterest_CE', 0)
        pe_oi = row.get('openInterest_PE', 0)
        if pe_oi > 1.12 * ce_oi:
            return "Support"
        elif ce_oi > 1.12 * pe_oi:
            return "Resistance"
        else:
            return "Neutral"
    
    df['Level'] = df.apply(determine_level, axis=1)
    
    return underlying_price, df

def calculate_bid_ask_pressure(call_bid_qty, call_ask_qty, put_bid_qty, put_ask_qty):
    pressure = (call_bid_qty - call_ask_qty) + (put_ask_qty - put_bid_qty)
    if pressure > 500:
        bias = "Bullish"
    elif pressure < -500:
        bias = "Bearish"
    else:
        bias = "Neutral"
    return pressure, bias

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

weights = {
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "DVP_Bias": 1,
    "PressureBias": 1,
}

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

def color_pcr(val):
    if val > 1.2:
        return 'background-color: #90EE90; color: black'
    elif val < 0.7:
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

def color_pressure(val):
    if val > 500:
        return 'background-color: #90EE90; color: black'
    elif val < -500:
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

def highlight_atm_row(row):
    """Highlight ATM row in the dataframe"""
    if row['Zone'] == 'ATM':
        return ['background-color: #FFD700; font-weight: bold'] * len(row)
    return [''] * len(row)

def create_option_chain_summary(df):
    """Create summary DataFrame from option chain data"""
    if df.empty:
        return pd.DataFrame()
    
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
            "PressureBias": pressure_bias,
            "openInterest_CE": row.get('openInterest_CE', 0),
            "openInterest_PE": row.get('openInterest_PE', 0),
            "changeinOpenInterest_CE": row.get('changeinOpenInterest_CE', 0),
            "changeinOpenInterest_PE": row.get('changeinOpenInterest_PE', 0),
            "lastPrice_CE": row.get('lastPrice_CE', 0),
            "lastPrice_PE": row.get('lastPrice_PE', 0)
        }
        
        for k in row_data:
            if "_Bias" in k:
                bias = row_data[k]
                score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)
        
        row_data["BiasScore"] = score
        row_data["Verdict"] = final_verdict(score)
        bias_results.append(row_data)
    
    df_summary = pd.DataFrame(bias_results)
    
    # Calculate PCR
    df_summary['PCR'] = df_summary['openInterest_PE'] / df_summary['openInterest_CE']
    df_summary['PCR'] = np.where(df_summary['openInterest_CE'] == 0, 0, df_summary['PCR'])
    df_summary['PCR'] = df_summary['PCR'].round(2)
    
    # PCR Signal
    df_summary['PCR_Signal'] = np.where(
        df_summary['PCR'] > 1.2, "BULLISH_SENTIMENT",
        np.where(df_summary['PCR'] < 0.7, "BEARISH_SENTIMENT", "NEUTRAL")
    )
    
    return df_summary

def create_candlestick_chart(df, title, show_pivots=False, pivot_settings=None):
    """Create TradingView-style candlestick chart"""
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

def display_metrics(df, ltp_data=None):
    """Display price metrics"""
    if not df.empty:
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
        
        if len(df) > 1:
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
                    <h3>{volume:,.0f}</h3>
                </div>
                """, unsafe_allow_html=True)

def display_market_insights(df_summary, current_price):
    """Display advanced market insights"""
    st.markdown("---")
    st.subheader("🧠 Advanced Market Insights")
    
    # Market Insight Panel
    st.markdown("""
    <div class="market-insights">
        <div class="insight-header">🎯 REAL Option Chain Logic (This matters)</div>
        <div class="insight-point">📊 <b>OI is useless without "WHY":</b> Instead of "CE OI high → resistance", ask "Is this OI defensive, aggressive, or trapped?"</div>
        <div class="insight-point">⚡ <b>Change in OI > Total OI:</b> Total OI is history. Change in OI is intent. Sudden +OI in ATM in 5–15 mins = active positioning</div>
        <div class="insight-point">🎯 <b>ATM is the battlefield:</b> Most people stare at far OTM like fools. Reality: ATM options control intraday direction</div>
        <div class="insight-point">💰 <b>Premium behavior > OI:</b> CE OI ↑ but CE premium ↑ → Writer in trouble (short covering coming)</div>
        <div class="insight-point">🔄 <b>Volatility tells the truth:</b> IV rising while price flat → Big move loading. IV falling while price moving → Move is ending</div>
    </div>
    """, unsafe_allow_html=True)
    
    if df_summary is not None and not df_summary.empty and current_price:
        atm_data = df_summary[df_summary['Zone'] == 'ATM']
        
        if not atm_data.empty:
            row = atm_data.iloc[0]
            
            # Display key ATM insights
            st.markdown("### 🎯 ATM Zone Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pcr = row.get('PCR', 1)
                pcr_color = "green" if pcr > 1.2 else "red" if pcr < 0.7 else "yellow"
                st.metric("ATM PCR", f"{pcr:.2f}", 
                         delta="Bullish" if pcr > 1.2 else "Bearish" if pcr < 0.7 else "Neutral")
            
            with col2:
                ce_oi = row.get('openInterest_CE', 0)
                pe_oi = row.get('openInterest_PE', 0)
                oi_ratio = ce_oi / pe_oi if pe_oi > 0 else 99
                st.metric("CE/PE OI Ratio", f"{oi_ratio:.2f}")
            
            with col3:
                ce_chg_oi = row.get('changeinOpenInterest_CE', 0)
                pe_chg_oi = row.get('changeinOpenInterest_PE', 0)
                trend = "Bullish Flow" if pe_chg_oi > ce_chg_oi else "Bearish Flow" if ce_chg_oi > pe_chg_oi else "Neutral"
                st.metric("ChgOI Trend", trend)
            
            # Quick assessment
            st.markdown("### 📊 Quick Assessment")
            
            if pcr > 1.2:
                st.success(f"✅ **PCR {pcr:.2f} > 1.2:** Market sentiment is **BULLISH** (more puts being bought for protection)")
            elif pcr < 0.7:
                st.warning(f"⚠️ **PCR {pcr:.2f} < 0.7:** Market sentiment is **BEARISH** (more calls being bought for speculation)")
            else:
                st.info(f"📊 **PCR {pcr:.2f} (0.7-1.2):** Market sentiment is **NEUTRAL**")
            
            if abs(oi_ratio - 1) > 0.3:
                if oi_ratio > 1.3:
                    st.warning(f"⚠️ **High CE/PE OI Ratio ({oi_ratio:.2f}): Resistance building** at current levels")
                else:
                    st.success(f"✅ **Low CE/PE OI Ratio ({oi_ratio:.2f}): Support building** at current levels")

def main():
    st.title("📈 Nifty Trading & Options Analyzer")
    
    # Initialize API credentials
    try:
        if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
            st.error("""
            ⚠️ **Dhan API credentials not configured**
            
            Please add your Dhan API credentials to Streamlit secrets:
            
            ```
            [secrets]
            DHAN_CLIENT_ID = "your_client_id"
            DHAN_ACCESS_TOKEN = "your_access_token"
            ```
            
            Without valid credentials, the app cannot fetch real market data.
            """)
            st.stop()
        
        # Validate credentials
        api = DhanAPI(DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID)
        success, message = api.test_connection()
        
        if not success:
            st.error(f"❌ {message}")
            st.info("Please check your Dhan API credentials and ensure they are valid and not expired.")
            st.stop()
        else:
            st.sidebar.success(message)
        
    except Exception as e:
        st.error(f"Credential validation error: {str(e)}")
        st.stop()
    
    # Telegram configuration
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        st.sidebar.success("Telegram notifications enabled")
    else:
        st.sidebar.warning("Telegram notifications disabled - configure bot token and chat ID")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Timeframe selection
    timeframes = {
        "1 min": "1",
        "3 min": "3", 
        "5 min": "5",
        "10 min": "10",
        "15 min": "15"
    }
    
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        list(timeframes.keys()),
        index=2  # Default to 5 min
    )
    
    interval = timeframes[selected_timeframe]
    
    # Options expiry selection
    st.sidebar.header("📅 Options Settings")
    
    # Get real expiry list from API
    expiry_data = api.get_expiry_list(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
    expiry_dates = []
    
    if expiry_data and 'data' in expiry_data:
        expiry_dates = expiry_data['data']
    else:
        st.sidebar.error("Could not fetch expiry dates from API")
        # Use fallback dates
        today = datetime.now()
        expiry_dates = [
            (today + timedelta(days=7)).strftime("%Y-%m-%d"),
            (today + timedelta(days=14)).strftime("%Y-%m-%d"),
            (today + timedelta(days=21)).strftime("%Y-%m-%d"),
            (today + timedelta(days=28)).strftime("%Y-%m-%d")
        ]
    
    selected_expiry = None
    if expiry_dates:
        expiry_options = [f"{exp} ({'Weekly' if i < 4 else 'Monthly'})" for i, exp in enumerate(expiry_dates)]
        selected_expiry_idx = st.sidebar.selectbox(
            "Select Expiry",
            range(len(expiry_options)),
            format_func=lambda x: expiry_options[x]
        )
        selected_expiry = expiry_dates[selected_expiry_idx]
    
    # Days back for data
    days_back = st.sidebar.slider("Days of Historical Data", 1, 5, 1)
    
    # Show market insights
    show_insights = st.sidebar.checkbox("Show Market Insights", value=True)
    
    # Connection Test
    st.sidebar.header("🔧 Connection Test")
    
    if st.sidebar.button("Test Telegram Connection"):
        success, message = test_telegram_connection()
        if success:
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.session_state.refresh_counter += 1
        st.session_state.api_cache = {}
        st.rerun()
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Trading Chart")
        
        # Fetch real intraday data
        with st.spinner("Fetching real-time chart data from Dhan API..."):
            data = api.get_intraday_data(
                security_id="13",
                exchange_segment="IDX_I", 
                instrument="INDEX",
                interval=interval,
                days_back=days_back
            )
            
            if data:
                df = process_candle_data(data, interval)
                
                if not df.empty:
                    # Display metrics
                    display_metrics(df)
                    
                    # Create and display chart
                    fig = create_candlestick_chart(
                        df, 
                        f"Nifty 50 - {selected_timeframe} Chart (Real Data)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data info
                    col1_info, col2_info = st.columns(2)
                    with col1_info:
                        st.info(f"📊 Data Points: {len(df)}")
                    with col2_info:
                        latest_time = df['datetime'].max().strftime("%Y-%m-%d %H:%M:%S IST")
                        st.info(f"🕐 Latest: {latest_time}")
                else:
                    st.error("Could not process chart data")
            else:
                st.error("Failed to fetch chart data from Dhan API")
    
    with col2:
        st.header("📊 Options Analysis")
        
        if selected_expiry:
            with st.spinner(f"Fetching real option chain data for expiry {selected_expiry}..."):
                # Fetch real option chain data
                option_data = api.get_option_chain(
                    NIFTY_UNDERLYING_SCRIP,
                    NIFTY_UNDERLYING_SEG,
                    selected_expiry
                )
                
                if option_data:
                    underlying_price, df_processed = process_option_chain_data(option_data, selected_expiry)
                    
                    if not df_processed.empty:
                        st.info(f"**NIFTY SPOT:** ₹{underlying_price:,.2f}")
                        
                        # Create summary
                        df_summary = create_option_chain_summary(df_processed)
                        
                        # Display OI changes
                        total_ce_change = df_summary['changeinOpenInterest_CE'].sum() / 100000
                        total_pe_change = df_summary['changeinOpenInterest_PE'].sum() / 100000
                        
                        st.markdown("## Open Interest Change (in Lakhs)")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("CALL ΔOI", f"{total_ce_change:+.1f}L", delta_color="inverse")
                        with col2:
                            st.metric("PUT ΔOI", f"{total_pe_change:+.1f}L", delta_color="normal")
                        
                        # Display option chain summary
                        st.markdown("## Option Chain Bias Summary")
                        
                        styled_df = df_summary.style\
                            .applymap(color_pcr, subset=['PCR'])\
                            .applymap(color_pressure, subset=['BidAskPressure'])\
                            .apply(highlight_atm_row, axis=1)
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Download button
                        csv_data = create_csv_download(df_summary)
                        st.download_button(
                            label="📥 Download Summary as CSV",
                            data=csv_data,
                            file_name=f"nifty_options_summary_{selected_expiry}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Could not process option chain data")
                else:
                    st.error("Failed to fetch option chain data from Dhan API")
        else:
            st.warning("Please select an expiry date")
    
    # Market Insights
    if show_insights and 'df_summary' in locals() and 'underlying_price' in locals():
        display_market_insights(df_summary, underlying_price)
    
    # Show current time
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    st.sidebar.info(f"Last Updated: {current_time}")
    
    # Increment refresh counter
    st.session_state.refresh_counter += 1

if __name__ == "__main__":
    main()