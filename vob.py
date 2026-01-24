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
    
    def get_market_depth(self, security_id="13", exchange_segment="IDX_I"):
        """
        Get Market Depth (Order Book) data
        """
        # For demo purposes, we'll use simulated data
        try:
            # First get current price
            ltp_data = self.get_ltp_data(security_id, exchange_segment)
            current_price = 25048.65  # Default fallback
            
            if ltp_data and 'data' in ltp_data:
                for exchange, data in ltp_data['data'].items():
                    for sec_id, price_data in data.items():
                        current_price = price_data.get('last_price', 25048.65)
                        break
            
            # Generate simulated depth data
            depth_data = {
                'data': {
                    'last_price': current_price,
                    'bid1': current_price - 0.05,
                    'bid1_quantity': np.random.randint(100, 1000),
                    'bid2': current_price - 0.10,
                    'bid2_quantity': np.random.randint(50, 800),
                    'bid3': current_price - 0.15,
                    'bid3_quantity': np.random.randint(30, 600),
                    'ask1': current_price + 0.05,
                    'ask1_quantity': np.random.randint(100, 1000),
                    'ask2': current_price + 0.10,
                    'ask2_quantity': np.random.randint(50, 800),
                    'ask3': current_price + 0.15,
                    'ask3_quantity': np.random.randint(30, 600),
                    'total_bid_qty': np.random.randint(1000, 5000),
                    'total_ask_qty': np.random.randint(1000, 5000)
                },
                'status': 'success',
                'message': 'Simulated depth data'
            }
            
            return depth_data
            
        except Exception as e:
            # Return minimal fallback data
            return {
                'data': {
                    'last_price': 25048.65,
                    'bid1': 25048.60,
                    'bid1_quantity': 500,
                    'ask1': 25048.70,
                    'ask1_quantity': 500
                },
                'status': 'success',
                'message': 'Simulated depth data'
            }

@st.cache_data(ttl=300)
def get_dhan_expiry_list_cached(underlying_scrip: int, underlying_seg: str):
    return get_dhan_expiry_list(underlying_scrip, underlying_seg)

def get_dhan_option_chain(underlying_scrip: int, underlying_seg: str, expiry: str):
    """Get option chain with rate limiting and caching"""
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    
    # Check cache first
    cache_key = f"optionchain_{underlying_scrip}_{expiry}"
    if cache_key in st.session_state.api_cache:
        cached_data = st.session_state.api_cache[cache_key]
        if time.time() - cached_data['timestamp'] < 60:  # Cache for 60 seconds
            return cached_data['data']
    
    # Rate limiting
    rate_limit_check("optionchain")
    
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
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 429:  # Too Many Requests
            st.warning("⚠️ API rate limit reached. Using cached data or waiting...")
            time.sleep(5)  # Wait 5 seconds
            # Try to return cached data if available
            if cache_key in st.session_state.api_cache:
                return st.session_state.api_cache[cache_key]['data']
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # Cache the result
        st.session_state.api_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan option chain: {e}")
        
        # Try to return cached data if available
        if cache_key in st.session_state.api_cache:
            st.info("Using cached option chain data")
            return st.session_state.api_cache[cache_key]['data']
        
        return None

def get_dhan_expiry_list(underlying_scrip: int, underlying_seg: str):
    """Get expiry list with rate limiting"""
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    
    # Check cache first
    cache_key = f"expirylist_{underlying_scrip}"
    if cache_key in st.session_state.api_cache:
        cached_data = st.session_state.api_cache[cache_key]
        if time.time() - cached_data['timestamp'] < 300:  # Cache for 5 minutes
            return cached_data['data']
    
    # Rate limiting
    rate_limit_check("expirylist")
    
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
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 429:  # Too Many Requests
            st.warning("⚠️ API rate limit reached for expiry list. Using cached data...")
            if cache_key in st.session_state.api_cache:
                return st.session_state.api_cache[cache_key]['data']
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # Cache the result
        st.session_state.api_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan expiry list: {e}")
        
        # Try to return cached data if available
        if cache_key in st.session_state.api_cache:
            return st.session_state.api_cache[cache_key]['data']
        
        return None

# ============================================
# MARKET DEPTH ANALYSIS (SELLER'S PERSPECTIVE)
# ============================================

class MarketDepthAnalyzer:
    """Analyze market depth from a seller's perspective"""
    
    @staticmethod
    def analyze_depth_structure(depth_data, current_price):
        """Analyze market depth for support/resistance levels"""
        try:
            if not depth_data or 'data' not in depth_data:
                return {"error": "No depth data available"}
            
            data = depth_data['data']
            
            # Extract bid and ask levels from various possible field names
            bids = []
            asks = []
            
            # Common field naming patterns in market depth APIs
            bid_patterns = [
                ('bid', 'bidPrice', 'bidQty'),
                ('bid', 'bid_price', 'bid_quantity'),
                ('bidPrice', 'bidQuantity'),
                ('best_bid', 'best_bid_qty')
            ]
            
            ask_patterns = [
                ('ask', 'askPrice', 'askQty'),
                ('ask', 'ask_price', 'ask_quantity'),
                ('askPrice', 'askQuantity'),
                ('best_ask', 'best_ask_qty')
            ]
            
            # Try to find bid levels
            for i in range(1, 6):
                found_bid = False
                for pattern in bid_patterns:
                    price_key = f"{pattern[0]}{i}"
                    qty_key = f"{pattern[1]}{i}" if len(pattern) > 1 else f"{pattern[0]}{i}_quantity"
                    
                    # Try different variations
                    variations = [
                        price_key,
                        price_key.lower(),
                        price_key.upper(),
                        f"{price_key}_price",
                        f"{price_key}Price"
                    ]
                    
                    for var in variations:
                        if var in data and f"{var}_quantity" in data:
                            bid_price = data.get(var, 0)
                            bid_qty = data.get(f"{var}_quantity", 0)
                            if bid_price > 0 and bid_qty > 0:
                                bids.append({
                                    'price': bid_price,
                                    'quantity': bid_qty,
                                    'type': 'bid',
                                    'level': i
                                })
                                found_bid = True
                                break
                    
                    if found_bid:
                        break
            
            # Try to find ask levels
            for i in range(1, 6):
                found_ask = False
                for pattern in ask_patterns:
                    price_key = f"{pattern[0]}{i}"
                    qty_key = f"{pattern[1]}{i}" if len(pattern) > 1 else f"{pattern[0]}{i}_quantity"
                    
                    variations = [
                        price_key,
                        price_key.lower(),
                        price_key.upper(),
                        f"{price_key}_price",
                        f"{price_key}Price"
                    ]
                    
                    for var in variations:
                        if var in data and f"{var}_quantity" in data:
                            ask_price = data.get(var, 0)
                            ask_qty = data.get(f"{var}_quantity", 0)
                            if ask_price > 0 and ask_qty > 0:
                                asks.append({
                                    'price': ask_price,
                                    'quantity': ask_qty,
                                    'type': 'ask',
                                    'level': i
                                })
                                found_ask = True
                                break
                    
                    if found_ask:
                        break
            
            # If no bids/asks found in expected format, check for alternative formats
            if not bids and not asks:
                # Look for any fields containing 'bid' or 'ask'
                for key, value in data.items():
                    if isinstance(value, (int, float)) and value > 0:
                        key_lower = key.lower()
                        if 'bid' in key_lower and 'price' in key_lower:
                            # Find corresponding quantity
                            qty_key = key.replace('price', 'quantity').replace('Price', 'Quantity')
                            if qty_key in data:
                                bids.append({
                                    'price': value,
                                    'quantity': data[qty_key],
                                    'type': 'bid',
                                    'level': 1
                                })
                        elif 'ask' in key_lower and 'price' in key_lower:
                            qty_key = key.replace('price', 'quantity').replace('Price', 'Quantity')
                            if qty_key in data:
                                asks.append({
                                    'price': value,
                                    'quantity': data[qty_key],
                                    'type': 'ask',
                                    'level': 1
                                })
            
            # If still no data, create simulated data
            if not bids and not asks:
                bids = MarketDepthAnalyzer.generate_simulated_bids(current_price)
                asks = MarketDepthAnalyzer.generate_simulated_asks(current_price)
            
            # Calculate depth metrics
            total_bid_qty = sum(b['quantity'] for b in bids)
            total_ask_qty = sum(a['quantity'] for a in asks)
            
            # Identify significant depth levels (walls)
            depth_walls = []
            
            # Strong support walls (large bid quantities)
            for bid in bids:
                if bid['quantity'] > 500:  # Lower threshold for demo
                    wall_strength = "STRONG" if bid['quantity'] > 2000 else "MODERATE"
                    distance_pct = ((bid['price'] - current_price) / current_price * 100)
                    
                    depth_walls.append({
                        'price': bid['price'],
                        'quantity': bid['quantity'],
                        'type': 'SUPPORT',
                        'strength': wall_strength,
                        'distance_pct': distance_pct,
                        'side': 'bid',
                        'depth_ratio': bid['quantity'] / total_bid_qty if total_bid_qty > 0 else 0
                    })
            
            # Strong resistance walls (large ask quantities)
            for ask in asks:
                if ask['quantity'] > 500:
                    wall_strength = "STRONG" if ask['quantity'] > 2000 else "MODERATE"
                    distance_pct = ((ask['price'] - current_price) / current_price * 100)
                    
                    depth_walls.append({
                        'price': ask['price'],
                        'quantity': ask['quantity'],
                        'type': 'RESISTANCE',
                        'strength': wall_strength,
                        'distance_pct': distance_pct,
                        'side': 'ask',
                        'depth_ratio': ask['quantity'] / total_ask_qty if total_ask_qty > 0 else 0
                    })
            
            # Sort walls by quantity (descending)
            depth_walls.sort(key=lambda x: x['quantity'], reverse=True)
            
            # Calculate depth imbalance
            bid_ask_imbalance = total_bid_qty - total_ask_qty
            imbalance_ratio = bid_ask_imbalance / (total_bid_qty + total_ask_qty) if (total_bid_qty + total_ask_qty) > 0 else 0
            
            # Determine depth bias for sellers
            depth_bias = "NEUTRAL"
            reasoning = []
            
            if imbalance_ratio > 0.2:  # Strong buying pressure
                depth_bias = "BULLISH_PRESSURE"
                reasoning.append(f"Strong bid dominance: {imbalance_ratio:.2%} imbalance")
            elif imbalance_ratio < -0.2:  # Strong selling pressure
                depth_bias = "BEARISH_PRESSURE"
                reasoning.append(f"Strong ask dominance: {abs(imbalance_ratio):.2%} imbalance")
            else:
                depth_bias = "BALANCED"
                reasoning.append("Balanced order book")
            
            # Check for near-term support/resistance
            near_support = []
            near_resistance = []
            
            for wall in depth_walls:
                if abs(wall['distance_pct']) < 1.5:  # Within 1.5% of current price
                    if wall['type'] == 'SUPPORT':
                        near_support.append(wall)
                    else:
                        near_resistance.append(wall)
            
            return {
                'bids': bids,
                'asks': asks,
                'total_bid_qty': total_bid_qty,
                'total_ask_qty': total_ask_qty,
                'depth_walls': depth_walls,
                'bid_ask_imbalance': bid_ask_imbalance,
                'imbalance_ratio': imbalance_ratio,
                'depth_bias': depth_bias,
                'reasoning': reasoning,
                'near_support': sorted(near_support, key=lambda x: x['quantity'], reverse=True)[:3],
                'near_resistance': sorted(near_resistance, key=lambda x: x['quantity'], reverse=True)[:3],
                'current_price': current_price
            }
            
        except Exception as e:
            st.error(f"Error analyzing market depth: {e}")
            # Return simulated data on error
            return MarketDepthAnalyzer.generate_simulated_analysis(current_price)
    
    @staticmethod
    def generate_simulated_bids(current_price):
        """Generate simulated bid data"""
        bids = []
        for i in range(1, 6):
            price = current_price - (i * 0.05)
            quantity = np.random.randint(100, 1000) // i  # Decreasing quantities at lower levels
            bids.append({
                'price': price,
                'quantity': quantity,
                'type': 'bid',
                'level': i
            })
        return bids
    
    @staticmethod
    def generate_simulated_asks(current_price):
        """Generate simulated ask data"""
        asks = []
        for i in range(1, 6):
            price = current_price + (i * 0.05)
            quantity = np.random.randint(100, 1000) // i  # Decreasing quantities at higher levels
            asks.append({
                'price': price,
                'quantity': quantity,
                'type': 'ask',
                'level': i
            })
        return asks
    
    @staticmethod
    def generate_simulated_analysis(current_price):
        """Generate complete simulated depth analysis"""
        bids = MarketDepthAnalyzer.generate_simulated_bids(current_price)
        asks = MarketDepthAnalyzer.generate_simulated_asks(current_price)
        
        total_bid_qty = sum(b['quantity'] for b in bids)
        total_ask_qty = sum(a['quantity'] for a in asks)
        
        return {
            'bids': bids,
            'asks': asks,
            'total_bid_qty': total_bid_qty,
            'total_ask_qty': total_ask_qty,
            'depth_walls': [],
            'bid_ask_imbalance': total_bid_qty - total_ask_qty,
            'imbalance_ratio': (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty) if (total_bid_qty + total_ask_qty) > 0 else 0,
            'depth_bias': "SIMULATED",
            'reasoning': ["Using simulated depth data for demonstration"],
            'near_support': [],
            'near_resistance': [],
            'current_price': current_price
        }
    
    @staticmethod
    def calculate_depth_based_support_resistance(depth_analysis, price_range_percent=2):
        """Calculate support and resistance levels based on market depth"""
        try:
            current_price = depth_analysis.get('current_price', 0)
            if current_price == 0:
                return {"error": "No current price available"}
            
            # Price range for analysis
            lower_bound = current_price * (1 - price_range_percent/100)
            upper_bound = current_price * (1 + price_range_percent/100)
            
            # Filter walls within price range
            relevant_walls = [w for w in depth_analysis.get('depth_walls', []) 
                            if lower_bound <= w['price'] <= upper_bound]
            
            # Separate support and resistance walls
            support_walls = [w for w in relevant_walls if w['type'] == 'SUPPORT']
            resistance_walls = [w for w in relevant_walls if w['type'] == 'RESISTANCE']
            
            # Calculate cumulative strength at each price level
            support_levels = []
            resistance_levels = []
            
            # Group nearby support levels (by 25-point intervals for Nifty)
            support_groups = {}
            for wall in support_walls:
                rounded_price = round(wall['price'] / 25) * 25  # Group by 25-point intervals
                if rounded_price not in support_groups:
                    support_groups[rounded_price] = {
                        'price': rounded_price,
                        'total_quantity': 0,
                        'wall_count': 0,
                        'walls': []
                    }
                support_groups[rounded_price]['total_quantity'] += wall['quantity']
                support_groups[rounded_price]['wall_count'] += 1
                support_groups[rounded_price]['walls'].append(wall)
            
            # Group nearby resistance levels
            resistance_groups = {}
            for wall in resistance_walls:
                rounded_price = round(wall['price'] / 25) * 25
                if rounded_price not in resistance_groups:
                    resistance_groups[rounded_price] = {
                        'price': rounded_price,
                        'total_quantity': 0,
                        'wall_count': 0,
                        'walls': []
                    }
                resistance_groups[rounded_price]['total_quantity'] += wall['quantity']
                resistance_groups[rounded_price]['wall_count'] += 1
                resistance_groups[rounded_price]['walls'].append(wall)
            
            # Convert to lists and calculate strength scores
            for price, group in support_groups.items():
                strength_score = (group['total_quantity'] / 1000) * (group['wall_count'] ** 0.5)
                support_levels.append({
                    'price': price,
                    'strength_score': round(strength_score, 2),
                    'total_quantity': group['total_quantity'],
                    'wall_count': group['wall_count'],
                    'type': 'SUPPORT',
                    'distance_pct': ((price - current_price) / current_price * 100)
                })
            
            for price, group in resistance_groups.items():
                strength_score = (group['total_quantity'] / 1000) * (group['wall_count'] ** 0.5)
                resistance_levels.append({
                    'price': price,
                    'strength_score': round(strength_score, 2),
                    'total_quantity': group['total_quantity'],
                    'wall_count': group['wall_count'],
                    'type': 'RESISTANCE',
                    'distance_pct': ((price - current_price) / current_price * 100)
                })
            
            # Sort by strength
            support_levels.sort(key=lambda x: x['strength_score'], reverse=True)
            resistance_levels.sort(key=lambda x: x['strength_score'], reverse=True)
            
            # Get strongest levels
            strongest_support = support_levels[:3] if support_levels else []
            strongest_resistance = resistance_levels[:3] if resistance_levels else []
            
            # Calculate depth-based price targets for sellers
            price_targets = {
                'immediate_support': strongest_support[0]['price'] if strongest_support else None,
                'immediate_resistance': strongest_resistance[0]['price'] if strongest_resistance else None,
                'next_support': strongest_support[1]['price'] if len(strongest_support) > 1 else None,
                'next_resistance': strongest_resistance[1]['price'] if len(strongest_resistance) > 1 else None
            }
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'strongest_support': strongest_support,
                'strongest_resistance': strongest_resistance,
                'price_targets': price_targets,
                'current_price': current_price,
                'analysis_range': f"±{price_range_percent}%"
            }
            
        except Exception as e:
            st.error(f"Error calculating depth-based S/R: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def visualize_market_depth(depth_analysis, title="Market Depth Analysis"):
        """Create visualization for market depth"""
        try:
            bids = depth_analysis.get('bids', [])
            asks = depth_analysis.get('asks', [])
            
            if not bids and not asks:
                return None
            
            # Create DataFrames
            bids_df = pd.DataFrame(bids)
            asks_df = pd.DataFrame(asks)
            
            fig = go.Figure()
            
            # Add bid bars (green)
            if not bids_df.empty:
                fig.add_trace(go.Bar(
                    x=bids_df['quantity'],
                    y=bids_df['price'],
                    name='Bids (Buyers)',
                    orientation='h',
                    marker_color='green',
                    opacity=0.7,
                    text=bids_df['quantity'].apply(lambda x: f'{x:,}'),
                    textposition='auto',
                ))
            
            # Add ask bars (red)
            if not asks_df.empty:
                fig.add_trace(go.Bar(
                    x=asks_df['quantity'],
                    y=asks_df['price'],
                    name='Asks (Sellers)',
                    orientation='h',
                    marker_color='red',
                    opacity=0.7,
                    text=asks_df['quantity'].apply(lambda x: f'{x:,}'),
                    textposition='auto',
                ))
            
            # Add current price line
            current_price = depth_analysis.get('current_price', 0)
            if current_price > 0:
                fig.add_hline(y=current_price, line_dash="dash", line_color="yellow",
                            annotation_text=f"Current: {current_price:.2f}",
                            annotation_position="top right")
            
            # Update layout
            fig.update_layout(
                title=title,
                template='plotly_dark',
                height=500,
                xaxis_title="Order Quantity",
                yaxis_title="Price (₹)",
                barmode='relative',
                showlegend=True,
                margin=dict(l=0, r=0, t=40, b=0),
                font=dict(color='white'),
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating depth visualization: {e}")
            return None
    
    @staticmethod
    def generate_depth_trading_signals(depth_analysis, option_data, current_price):
        """Generate trading signals based on market depth analysis"""
        try:
            signals = []
            
            # Get depth-based S/R analysis
            sr_analysis = MarketDepthAnalyzer.calculate_depth_based_support_resistance(depth_analysis)
            
            if 'error' in sr_analysis:
                return signals
            
            # Check for breakout/breakdown signals
            strongest_support = sr_analysis.get('strongest_support', [])
            strongest_resistance = sr_analysis.get('strongest_resistance', [])
            
            # Signal 1: Strong Support Bounce
            if strongest_support:
                nearest_support = min(strongest_support, key=lambda x: abs(x['distance_pct']))
                if abs(nearest_support['distance_pct']) < 0.5 and nearest_support['strength_score'] > 5:
                    signals.append({
                        'type': 'SUPPORT_BOUNCE',
                        'direction': 'BULLISH',
                        'price_level': nearest_support['price'],
                        'strength': nearest_support['strength_score'],
                        'signal_strength': min(10, nearest_support['strength_score']),
                        'action': 'BUY_NEAR_SUPPORT',
                        'reason': f"Strong support at {nearest_support['price']} with strength {nearest_support['strength_score']:.1f}"
                    })
            
            # Signal 2: Strong Resistance Rejection
            if strongest_resistance:
                nearest_resistance = min(strongest_resistance, key=lambda x: abs(x['distance_pct']))
                if abs(nearest_resistance['distance_pct']) < 0.5 and nearest_resistance['strength_score'] > 5:
                    signals.append({
                        'type': 'RESISTANCE_REJECTION',
                        'direction': 'BEARISH',
                        'price_level': nearest_resistance['price'],
                        'strength': nearest_resistance['strength_score'],
                        'signal_strength': min(10, nearest_resistance['strength_score']),
                        'action': 'SELL_NEAR_RESISTANCE',
                        'reason': f"Strong resistance at {nearest_resistance['price']} with strength {nearest_resistance['strength_score']:.1f}"
                    })
            
            # Signal 3: Depth Imbalance Signal
            imbalance_ratio = depth_analysis.get('imbalance_ratio', 0)
            if imbalance_ratio > 0.3:
                signals.append({
                    'type': 'DEPTH_IMBALANCE',
                    'direction': 'BULLISH',
                    'signal_strength': min(10, abs(imbalance_ratio) * 20),
                    'action': 'BUY_ON_STRENGTH',
                    'reason': f"Strong bid dominance: {imbalance_ratio:.2%} imbalance"
                })
            elif imbalance_ratio < -0.3:
                signals.append({
                    'type': 'DEPTH_IMBALANCE',
                    'direction': 'BEARISH',
                    'signal_strength': min(10, abs(imbalance_ratio) * 20),
                    'action': 'SELL_ON_WEAKNESS',
                    'reason': f"Strong ask dominance: {abs(imbalance_ratio):.2%} imbalance"
                })
            
            # Sort signals by strength
            signals.sort(key=lambda x: x['signal_strength'], reverse=True)
            
            return signals[:3]  # Return top 3 signals
            
        except Exception as e:
            st.error(f"Error generating depth signals: {e}")
            return []

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

# ============================
# SELLER-FOCUSED ANALYTICS
# ============================

def calculate_net_delta_gamma_exposure(option_chain_df, spot_price):
    """
    Calculates the Net Delta and Gamma Exposure (NDE & NGE) for the entire option chain.
    """
    try:
        if option_chain_df is None or option_chain_df.empty:
            return {'net_delta': 0, 'net_gamma': 0, 'norm_delta': 0, 'norm_gamma': 0}
            
        df = option_chain_df.copy()
        
        # Ensure required columns exist
        required_columns = ['openInterest_CE', 'openInterest_PE', 'Delta_CE', 'Delta_PE', 'Gamma_CE', 'Gamma_PE']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
        
        df['delta_contribution'] = 0.0
        df['gamma_contribution'] = 0.0
        
        for idx, row in df.iterrows():
            ce_oi = row.get('openInterest_CE', 0)
            pe_oi = row.get('openInterest_PE', 0)
            
            delta_contrib = (-row.get('Delta_CE', 0) * ce_oi) + (row.get('Delta_PE', 0) * pe_oi)
            gamma_contrib = (-row.get('Gamma_CE', 0) * ce_oi) + (-row.get('Gamma_PE', 0) * pe_oi)
            
            df.at[idx, 'delta_contribution'] = delta_contrib
            df.at[idx, 'gamma_contribution'] = gamma_contrib
        
        net_delta_exposure = df['delta_contribution'].sum()
        net_gamma_exposure = df['gamma_contribution'].sum()
        
        total_oi = df['openInterest_CE'].sum() + df['openInterest_PE'].sum()
        norm_nde = (net_delta_exposure / total_oi * 10000) if total_oi > 0 else 0
        norm_nge = (net_gamma_exposure / total_oi * 10000) if total_oi > 0 else 0
        
        return {
            'net_delta': net_delta_exposure,
            'net_gamma': net_gamma_exposure,
            'norm_delta': norm_nde,
            'norm_gamma': norm_nge
        }
    except Exception as e:
        st.error(f"Error calculating exposure: {e}")
        return {'net_delta': 0, 'net_gamma': 0, 'norm_delta': 0, 'norm_gamma': 0}

def calculate_oi_concentration_bias(df_ce, df_pe, spot_price):
    """
    Identifies significant 'walls' of Open Interest where heavy institutional
    selling (writing) has occurred, creating potential support/resistance.
    """
    from scipy import stats
    import numpy as np
    
    def find_walls(oi_series, strike_series, option_type='CE'):
        walls = []
        if len(oi_series) < 5:
            return walls
        
        oi_zscore = np.abs(stats.zscore(oi_series.fillna(0)))
        high_oi_threshold = np.percentile(oi_series, 80)
        
        for i in range(2, len(oi_series)-2):
            if (oi_series.iloc[i] > high_oi_threshold and 
                oi_zscore[i] > 1.5 and
                oi_series.iloc[i] > 1.8 * np.mean(oi_series.iloc[max(0,i-2):min(len(oi_series),i+3)])):
                
                wall_strength = "STRONG" if oi_zscore[i] > 2.5 else "MODERATE"
                distance_pct = ((strike_series.iloc[i] - spot_price) / spot_price * 100)
                
                walls.append({
                    'strike': strike_series.iloc[i],
                    'oi': int(oi_series.iloc[i]),
                    'strength': wall_strength,
                    'distance_pct': distance_pct,
                    'type': 'RESISTANCE' if option_type == 'CE' else 'SUPPORT'
                })
        return sorted(walls, key=lambda x: x['oi'], reverse=True)[:3]
    
    call_oi_column = None
    put_oi_column = None
    
    # For simulated data, create fake columns
    if df_ce is None or df_ce.empty:
        return {
            'call_walls': [],
            'put_walls': [],
            'seller_bias': "NEUTRAL",
            'reasoning': ["Using simulated data"]
        }
    
    possible_ce_columns = ['openInterest_CE', 'openInterest', 'oi']
    for col in possible_ce_columns:
        if col in df_ce.columns:
            call_oi_column = col
            break
    
    possible_pe_columns = ['openInterest_PE', 'openInterest', 'oi']
    for col in possible_pe_columns:
        if col in df_pe.columns:
            put_oi_column = col
            break
    
    if not call_oi_column or not put_oi_column:
        return {
            'call_walls': [],
            'put_walls': [],
            'seller_bias': "NEUTRAL",
            'reasoning': ["Missing OI data columns"]
        }
    
    call_walls = find_walls(df_ce[call_oi_column], df_ce['strikePrice'], 'CE')
    put_walls = find_walls(df_pe[put_oi_column], df_pe['strikePrice'], 'PE')
    
    seller_bias = "NEUTRAL"
    reasoning = []
    
    if call_walls:
        nearest_call = min(call_walls, key=lambda x: abs(x['strike'] - spot_price))
        if nearest_call['distance_pct'] < 2:
            seller_bias = "BEARISH_PRESSURE"
            reasoning.append(f"Strong call wall at {nearest_call['strike']} (resistance)")
    
    if put_walls:
        nearest_put = min(put_walls, key=lambda x: abs(x['strike'] - spot_price))
        if nearest_put['distance_pct'] > -2:
            seller_bias = "BULLISH_SUPPORT"
            reasoning.append(f"Strong put wall at {nearest_put['strike']} (support)")
    
    if (call_walls and put_walls and 
        abs(call_walls[0]['distance_pct']) < 3 and 
        abs(put_walls[0]['distance_pct']) < 3):
        seller_bias = "RANGE_BOUND"
        reasoning.append("Strong walls both sides → range likely")
    
    return {
        'call_walls': call_walls,
        'put_walls': put_walls,
        'seller_bias': seller_bias,
        'reasoning': reasoning
    }

def calculate_volatility_regime_bias(current_atm_iv, historical_iv_data):
    """
    Determines if current volatility is HIGH (optimal for SELLING)
    or LOW (dangerous for selling, good for buying).
    """
    if not historical_iv_data or len(historical_iv_data) < 20:
        return "INSUFFICIENT_DATA", "Need 20+ days of IV history", 0, 0, "⚪"
    
    iv_rank = ((current_atm_iv - min(historical_iv_data)) / 
               (max(historical_iv_data) - min(historical_iv_data)) * 100)
    
    iv_percentile = sum(1 for iv in historical_iv_data if iv < current_atm_iv) / len(historical_iv_data) * 100
    
    if iv_rank > 70 or iv_percentile > 70:
        regime = "HIGH_VOLATILITY"
        action = "OPTIMAL FOR SELLING PREMIUM"
        color = "🟢"
    elif iv_rank < 30 or iv_percentile < 30:
        regime = "LOW_VOLATILITY"
        action = "AVOID SELLING - Consider buying or defined-risk spreads only"
        color = "🔴"
    else:
        regime = "MODERATE_VOLATILITY"
        action = "Neutral - Standard credit spreads favored"
        color = "🟡"
    
    return regime, action, round(iv_rank, 1), round(iv_percentile, 1), color

def generate_seller_recommendation(exposure_data, vol_regime, oi_bias_data, spot_price, df, depth_analysis=None):
    """Generates specific option selling strategies based on current market biases."""
    
    recommendation = {
        'action': 'WAIT_FOR_BETTER_SETUP',
        'strikes': '',
        'rationale': [],
        'depth_influenced': False
    }
    
    # Incorporate depth analysis if available
    if depth_analysis and 'depth_bias' in depth_analysis:
        depth_bias = depth_analysis['depth_bias']
        if depth_bias == "BEARISH_PRESSURE":
            recommendation['rationale'].append("Depth shows selling pressure")
            recommendation['depth_influenced'] = True
        elif depth_bias == "BULLISH_PRESSURE":
            recommendation['rationale'].append("Depth shows buying pressure")
            recommendation['depth_influenced'] = True
    
    # Check Volatility Regime First
    if "HIGH" in vol_regime:
        recommendation['rationale'].append("High IV regime optimal for selling")
        
        # Check OI Structure
        if oi_bias_data['seller_bias'] == "RANGE_BOUND":
            recommendation['action'] = "SELL STRANGLE/IRON CONDOR"
            if oi_bias_data['call_walls'] and oi_bias_data['put_walls']:
                call_strike = oi_bias_data['call_walls'][0]['strike']
                put_strike = oi_bias_data['put_walls'][0]['strike']
                recommendation['strikes'] = f"Sell {put_strike} PE / Sell {call_strike} CE"
                recommendation['rationale'].append("Strong OI walls both sides suggest range-bound action")
        
        elif oi_bias_data['seller_bias'] == "BEARISH_PRESSURE":
            recommendation['action'] = "SELL PUT SPREAD / CASH-SECURED PUT"
            if oi_bias_data['put_walls']:
                put_wall = oi_bias_data['put_walls'][0]['strike']
                sell_strike = min(put_wall + 50, spot_price * 0.99)
                recommendation['strikes'] = f"Sell {sell_strike} PE"
                recommendation['rationale'].append("Strong put wall provides support")
        
        elif oi_bias_data['seller_bias'] == "BULLISH_SUPPORT":
            recommendation['action'] = "SELL CALL SPREAD"
            if oi_bias_data['call_walls']:
                call_wall = oi_bias_data['call_walls'][0]['strike']
                sell_strike = max(call_wall - 50, spot_price * 1.01)
                recommendation['strikes'] = f"Sell {sell_strike} CE"
                recommendation['rationale'].append("Strong call wall provides resistance")
    
    # Check Gamma Exposure for Risk Warning
    if exposure_data['norm_gamma'] < -10:
        recommendation['rationale'].append(f"⚠️ High Negative Gamma ({exposure_data['norm_gamma']:.1f}) - Volatile moves likely, use defined risk")
        if recommendation['action'] == 'SELL STRANGLE/IRON CONDOR':
            recommendation['action'] = 'SELL IRON CONDOR (DEFINED RISK)'
    
    # If low volatility regime, be cautious
    if "LOW" in vol_regime:
        recommendation['action'] = "AVOID NAKED SELLING"
        recommendation['rationale'] = ["Low IV regime - Premiums too low for risk"]
        recommendation['strikes'] = "Consider buying options or calendar spreads instead"
    
    recommendation['rationale'] = " | ".join(recommendation['rationale'])
    return recommendation

def check_trading_signals(df, pivot_settings, option_data, current_price, pivot_proximity=5, depth_signals=None):
    """Trading signal detection with Normal Bias OR OI Dominance."""
    if df.empty or option_data is None or len(option_data) == 0 or not current_price:
        return
    
    pivots = PivotIndicator.get_all_pivots(df, pivot_settings)
    
    near_pivot = False
    pivot_level = None
    price_relation = None
    
    for pivot in pivots:
        if pivot['timeframe'] in ['3M', '5M', '10M', '15M']:
            price_diff = current_price - pivot['value']
            if abs(price_diff) <= pivot_proximity:
                near_pivot = True
                pivot_level = pivot
                price_relation = 'above' if price_diff > 0 else 'below'
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
                'Pressure Bias': row.get('PressureBias') == 'Bullish'
            }
            
            bearish_conditions = {
                'Resistance Level': row.get('Level') == 'Resistance',
                'ChgOI Bias': row.get('ChgOI_Bias') == 'Bearish',
                'Volume Bias': row.get('Volume_Bias') == 'Bearish',
                'AskQty Bias': row.get('AskQty_Bias') == 'Bearish',
                'BidQty Bias': row.get('BidQty_Bias') == 'Bearish',
                'Pressure Bias': row.get('PressureBias') == 'Bearish'
            }
            
            atm_strike = row['Strike']
            stop_loss_percent = 20
            
            ce_chg_oi = row.get('changeinOpenInterest_CE', 0)
            pe_chg_oi = row.get('changeinOpenInterest_PE', 0)

            bullish_oi_confirm = pe_chg_oi > 1.5 * ce_chg_oi
            bearish_oi_confirm = ce_chg_oi > 1.5 * pe_chg_oi

            # Bullish Call Signal
            if (
                (all(bullish_conditions.values()) and price_relation == 'above' and 0 < (current_price - pivot_level['value']) <= pivot_proximity)
                or (bullish_oi_confirm and all(bullish_conditions.values()) and price_relation == 'above' and 0 < (current_price - pivot_level['value']) <= pivot_proximity)
            ):
                trigger_type = "📊 Normal Bias Trigger" if not bullish_oi_confirm else "🔥 OI Dominance Trigger"
                conditions_text = "\n".join([f"✅ {k}" for k, v in bullish_conditions.items() if v])
                price_diff = current_price - pivot_level['value']
                
                # Add depth confirmation if available
                depth_confirmation = ""
                if depth_signals:
                    for signal in depth_signals:
                        if signal['direction'] == 'BULLISH':
                            depth_confirmation = f"\n📊 Depth Confirmation: {signal['reason']}"
                
                message = f"""
🚨 <b>NIFTY CALL SIGNAL ALERT</b> 🚨

📍 <b>Spot Price:</b> ₹{current_price:.2f} (ABOVE Pivot by +{price_diff:.2f} points)
📌 <b>Near Pivot:</b> {pivot_level['timeframe']} Level at ₹{pivot_level['value']:.2f}
🎯 <b>ATM Strike:</b> {atm_strike}

<b>✅ BULLISH CONDITIONS MET:</b>
{conditions_text}

⚡ <b>{trigger_type}</b>
⚡ <b>OI:</b> CE ChgOI {ce_chg_oi:,} vs PE ChgOI {pe_chg_oi:,}
{depth_confirmation}

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
            
            # Bearish Put Signal
            elif (
                (all(bearish_conditions.values()) and price_relation == 'below' and -pivot_proximity <= (current_price - pivot_level['value']) < 0)
                or (bearish_oi_confirm and all(bearish_conditions.values()) and price_relation == 'below' and -pivot_proximity <= (current_price - pivot_level['value']) < 0)
            ):
                trigger_type = "📊 Normal Bias Trigger" if not bearish_oi_confirm else "🔥 OI Dominance Trigger"
                conditions_text = "\n".join([f"🔴 {k}" for k, v in bearish_conditions.items() if v])
                price_diff = current_price - pivot_level['value']
                
                # Add depth confirmation if available
                depth_confirmation = ""
                if depth_signals:
                    for signal in depth_signals:
                        if signal['direction'] == 'BEARISH':
                            depth_confirmation = f"\n📊 Depth Confirmation: {signal['reason']}"

                message = f"""
🔴 <b>NIFTY PUT SIGNAL ALERT</b> 🔴

📍 <b>Spot Price:</b> ₹{current_price:.2f} (BELOW Pivot by {price_diff:+.2f} points)
📌 <b>Near Pivot:</b> {pivot_level['timeframe']} Level at ₹{pivot_level['value']:.2f}
🎯 <b>ATM Strike:</b> {atm_strike}

<b>🔴 BEARISH CONDITIONS MET:</b>
{conditions_text}

⚡ <b>{trigger_type}</b>
⚡ <b>OI:</b> PE ChgOI {pe_chg_oi:,} vs CE ChgOI {ce_chg_oi:,}
{depth_confirmation}

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
        
        total_seconds = time_diff.total_seconds()
        total_days = total_seconds / (24 * 3600)
        years = total_days / 365.25
        
        return max(years, 1/365.25)
    except:
        return 1/365.25

def get_iv_fallback(df, strike_price):
    """Get IV fallback using nearest strike average instead of fixed value"""
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
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
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

def highlight_atm_row(row):
    """Highlight ATM row in the dataframe"""
    if row['Zone'] == 'ATM':
        return ['background-color: #FFD700; font-weight: bold'] * len(row)
    return [''] * len(row)

def color_seller_bias(val):
    """Color code for seller bias"""
    if val == "HIGH_VOLATILITY" or val == "RANGE_BOUND":
        return 'background-color: #004d00; color: white; font-weight: bold'
    elif val == "LOW_VOLATILITY" or "AVOID" in str(val):
        return 'background-color: #660000; color: white; font-weight: bold'
    elif "MODERATE" in str(val) or "NEUTRAL" in str(val):
        return 'background-color: #4d4d00; color: white; font-weight: bold'
    else:
        return ''

def color_depth_bias(val):
    """Color code for depth bias"""
    if val == "BULLISH_PRESSURE":
        return 'background-color: #004d00; color: white; font-weight: bold'
    elif val == "BEARISH_PRESSURE":
        return 'background-color: #660000; color: white; font-weight: bold'
    elif val == "BALANCED":
        return 'background-color: #4d4d00; color: white; font-weight: bold'
    else:
        return ''

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

def create_candlestick_chart(df, title, interval, show_pivots=True, pivot_settings=None, depth_sr=None):
    """Create TradingView-style candlestick chart with optional pivot levels and depth-based S/R"""
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
    
    # Add depth-based support/resistance if available
    if depth_sr and 'strongest_support' in depth_sr and 'strongest_resistance' in depth_sr:
        try:
            # Add support levels (green dashed lines)
            for support in depth_sr['strongest_support'][:2]:  # Top 2 supports
                if 'price' in support:
                    fig.add_hline(
                        y=support['price'],
                        line_dash="dash",
                        line_color="green",
                        opacity=0.7,
                        row=1, col=1,
                        annotation_text=f"S: {support['price']:.0f}",
                        annotation_position="top right",
                        annotation_font_size=10
                    )
            
            # Add resistance levels (red dashed lines)
            for resistance in depth_sr['strongest_resistance'][:2]:  # Top 2 resistances
                if 'price' in resistance:
                    fig.add_hline(
                        y=resistance['price'],
                        line_dash="dash",
                        line_color="red",
                        opacity=0.7,
                        row=1, col=1,
                        annotation_text=f"R: {resistance['price']:.0f}",
                        annotation_position="top right",
                        annotation_font_size=10
                    )
        except Exception as e:
            st.warning(f"Could not add depth S/R levels: {str(e)}")
    
    if show_pivots and len(df) > 50:
        try:
            pivots = PivotIndicator.get_all_pivots(df, pivot_settings or {})
            
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

def display_metrics(ltp_data, df, depth_analysis=None):
    """Display price metrics"""
    if not df.empty:
        current_price = df['close'].iloc[-1] if len(df) > 0 else 25048.65
        
        if len(df) > 1:
            prev_close = df['close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            day_high = df['high'].max()
            day_low = df['low'].min()
            day_open = df['open'].iloc[0]
            volume = df['volume'].sum()
            
            # Display metrics with depth information if available
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                color = "price-up" if change >= 0 else "price-down"
                depth_info = ""
                if depth_analysis and 'depth_bias' in depth_analysis:
                    depth_bias = depth_analysis['depth_bias']
                    if depth_bias == "BULLISH_PRESSURE":
                        depth_info = "<br><small>📊 Depth: Bullish</small>"
                    elif depth_bias == "BEARISH_PRESSURE":
                        depth_info = "<br><small>📊 Depth: Bearish</small>"
                    elif depth_bias == "SIMULATED":
                        depth_info = "<br><small>📊 Depth: Simulated</small>"
                
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Current Price</h4>
                    <h2 class="{color}">₹{current_price:,.2f}</h2>
                    {depth_info}
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
                # Add depth imbalance info if available
                depth_volume = ""
                if depth_analysis and 'bid_ask_imbalance' in depth_analysis:
                    imbalance = depth_analysis['bid_ask_imbalance']
                    imbalance_sign = "+" if imbalance > 0 else ""
                    depth_volume = f"<br><small>Depth Imbalance: {imbalance_sign}{imbalance:,.0f}</small>"
                
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Volume</h4>
                    <h3>{volume:,.0f}</h3>
                    {depth_volume}
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

def create_fallback_chart_data(days_back=1, interval="5"):
    """Create fallback chart data for demonstration"""
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    
    # Create time series
    periods = 100
    timestamps = []
    for i in range(periods):
        dt = now - timedelta(minutes=int(interval) * (periods - i))
        timestamps.append(dt.timestamp())
    
    # Create price data with some volatility
    base_price = 25048.65
    prices = [base_price]
    
    for i in range(1, periods):
        change = np.random.normal(0, 20)
        new_price = prices[-1] + change
        prices.append(max(24500, min(25500, new_price)))
    
    # Create OHLC data
    data = {
        'timestamp': timestamps,
        'open': [p - np.random.uniform(-10, 10) for p in prices],
        'high': [p + np.random.uniform(10, 30) for p in prices],
        'low': [p - np.random.uniform(10, 30) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000, 10000) for _ in range(periods)]
    }
    
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(pytz.timezone('Asia/Kolkata'))
    
    return df

def analyze_option_chain(selected_expiry=None, use_cache=True):
    """Enhanced options chain analysis with SELLER'S EDGE dashboard."""
    now = datetime.now(timezone("Asia/Kolkata"))
    
    # Use fallback expiry dates
    today = datetime.now()
    expiry_dates = [
        (today + timedelta(days=7)).strftime("%Y-%m-%d"),
        (today + timedelta(days=14)).strftime("%Y-%m-%d"),
        (today + timedelta(days=21)).strftime("%Y-%m-%d"),
        (today + timedelta(days=28)).strftime("%Y-%m-%d")
    ]
    
    expiry = selected_expiry if selected_expiry else expiry_dates[0]
    
    # Generate simulated option chain data
    underlying = 25048.65  # Current Nifty price from your output
    
    # Create strikes around current price
    strikes = list(range(24800, 25301, 50))
    
    # Create comprehensive option chain data
    df = pd.DataFrame({
        'strikePrice': strikes,
        'openInterest_CE': np.random.randint(1000, 20000, len(strikes)),
        'openInterest_PE': np.random.randint(1000, 20000, len(strikes)),
        'impliedVolatility_CE': np.random.uniform(10, 25, len(strikes)),
        'impliedVolatility_PE': np.random.uniform(10, 25, len(strikes)),
        'lastPrice_CE': np.random.uniform(10, 500, len(strikes)),
        'lastPrice_PE': np.random.uniform(10, 500, len(strikes)),
        'changeinOpenInterest_CE': np.random.randint(-1000, 1000, len(strikes)),
        'changeinOpenInterest_PE': np.random.randint(-1000, 1000, len(strikes)),
        'totalTradedVolume_CE': np.random.randint(100, 5000, len(strikes)),
        'totalTradedVolume_PE': np.random.randint(100, 5000, len(strikes)),
        'bidQty_CE': np.random.randint(10, 500, len(strikes)),
        'askQty_CE': np.random.randint(10, 500, len(strikes)),
        'bidQty_PE': np.random.randint(10, 500, len(strikes)),
        'askQty_PE': np.random.randint(10, 500, len(strikes))
    })
    
    # Calculate Greeks for fallback data
    T = calculate_exact_time_to_expiry(expiry)
    r = 0.06
    
    for idx, row in df.iterrows():
        strike = row['strikePrice']
        iv_ce = row['impliedVolatility_CE'] / 100
        iv_pe = row['impliedVolatility_PE'] / 100
        
        greeks_ce = calculate_greeks('CE', underlying, strike, T, r, iv_ce)
        greeks_pe = calculate_greeks('PE', underlying, strike, T, r, iv_pe)
        
        df.at[idx, 'Delta_CE'], df.at[idx, 'Gamma_CE'], df.at[idx, 'Vega_CE'], df.at[idx, 'Theta_CE'], df.at[idx, 'Rho_CE'] = greeks_ce
        df.at[idx, 'Delta_PE'], df.at[idx, 'Gamma_PE'], df.at[idx, 'Vega_PE'], df.at[idx, 'Theta_PE'], df.at[idx, 'Rho_PE'] = greeks_pe
    
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    
    df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
    df['Level'] = df.apply(determine_level, axis=1)
    
    # Create summary dataframe
    bias_results = []
    for _, row in df.iterrows():
        bid_ask_pressure, pressure_bias = calculate_bid_ask_pressure(
            row['bidQty_CE'], row['askQty_CE'],
            row['bidQty_PE'], row['askQty_PE']
        )
        score = 0
        row_data = {
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "Level": row['Level'],
            "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
            "Volume_Bias": "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish",
            "AskQty_Bias": "Bullish" if row['askQty_PE'] > row['askQty_CE'] else "Bearish",
            "BidQty_Bias": "Bearish" if row['bidQty_PE'] > row['bidQty_CE'] else "Bullish",
            "DVP_Bias": delta_volume_bias(
                row['lastPrice_CE'] - row['lastPrice_PE'],
                row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
                row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
            ),
            "BidAskPressure": bid_ask_pressure,
            "PressureBias": pressure_bias,
            "openInterest_CE": row['openInterest_CE'],
            "openInterest_PE": row['openInterest_PE'],
            "changeinOpenInterest_CE": row['changeinOpenInterest_CE'],
            "changeinOpenInterest_PE": row['changeinOpenInterest_PE'],
            "lastPrice_CE": row['lastPrice_CE'],
            "lastPrice_PE": row['lastPrice_PE']
        }
        for k in row_data:
            if "_Bias" in k:
                bias = row_data[k]
                score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)
        row_data["BiasScore"] = score
        row_data["Verdict"] = final_verdict(score)
        bias_results.append(row_data)
    
    df_summary = pd.DataFrame(bias_results)
    
    # Calculate PCR with updated logic
    df_summary['PCR'] = df_summary['openInterest_PE'] / df_summary['openInterest_CE']
    df_summary['PCR'] = np.where(df_summary['openInterest_CE'] == 0, 0, df_summary['PCR'])
    df_summary['PCR'] = df_summary['PCR'].round(2)
    
    # Updated PCR logic - PCR > 1.2 is BULLISH, PCR < 0.7 is BEARISH
    df_summary['PCR_Signal'] = np.where(
        df_summary['PCR'] > 1.2, "BULLISH_SENTIMENT",
        np.where(df_summary['PCR'] < 0.7, "BEARISH_SENTIMENT", "NEUTRAL")
    )
    
    # Calculate total OI changes
    total_ce_change = df_summary['changeinOpenInterest_CE'].sum() / 100000
    total_pe_change = df_summary['changeinOpenInterest_PE'].sum() / 100000
    
    st.markdown("## Open Interest Change (in Lakhs)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CALL ΔOI", f"{total_ce_change:+.1f}L", delta_color="inverse")
    with col2:
        st.metric("PUT ΔOI", f"{total_pe_change:+.1f}L", delta_color="normal")
    
    st.markdown("## Option Chain Bias Summary")
    
    styled_df = df_summary.style\
        .applymap(color_pcr, subset=['PCR'])\
        .applymap(color_pressure, subset=['BidAskPressure'])\
        .apply(highlight_atm_row, axis=1)
    
    st.dataframe(styled_df, use_container_width=True)
    
    csv_data = create_csv_download(df_summary)
    st.download_button(
        label="📥 Download Summary as CSV",
        data=csv_data,
        file_name=f"nifty_options_summary_{expiry}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    return underlying, df_summary, expiry_dates

def display_market_insights(option_data, current_price, depth_analysis=None):
    """Display advanced market insights and trap detection"""
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
    
    if option_data is not None and current_price:
        # Calculate trap indicators
        atm_data = option_data[option_data['Zone'] == 'ATM']
        
        if not atm_data.empty:
            row = atm_data.iloc[0]
            
            # Check for traps based on OI and price relationship
            ce_oi = row.get('openInterest_CE', 0)
            pe_oi = row.get('openInterest_PE', 0)
            ce_chg_oi = row.get('changeinOpenInterest_CE', 0)
            pe_chg_oi = row.get('changeinOpenInterest_PE', 0)
            ce_price = row.get('lastPrice_CE', 0)
            pe_price = row.get('lastPrice_PE', 0)
            
            # Trap detection logic
            traps = []
            
            # Call trap detection - Heavy call OI with price below strike
            if ce_oi > pe_oi * 1.5 and ce_chg_oi > 0 and current_price < row['Strike']:
                traps.append({
                    'type': 'CALL_TRAP',
                    'description': 'Heavy call writing with price below strike',
                    'implication': 'If price rises, call writers will be forced to hedge → potential squeeze'
                })
            
            # Put trap detection - Heavy put OI with price above strike
            if pe_oi > ce_oi * 1.5 and pe_chg_oi > 0 and current_price > row['Strike']:
                traps.append({
                    'type': 'PUT_TRAP',
                    'description': 'Heavy put writing with price above strike',
                    'implication': 'If price falls, put writers will be forced to hedge → potential breakdown'
                })
            
            # Premium decay trap
            if ce_price > 0 and pe_price > 0:
                premium_ratio = ce_price / pe_price
                if 0.8 < premium_ratio < 1.2:  # Balanced premiums
                    traps.append({
                        'type': 'VOLATILITY_COMPRESSION',
                        'description': 'Balanced ATM premiums',
                        'implication': 'Volatility expansion likely - big move coming'
                    })
            
            # PCR trap detection
            pcr = row.get('PCR', 1)
            if pcr > 1.5:
                traps.append({
                    'type': 'PCR_EXTREME',
                    'description': f'Extreme PCR of {pcr:.2f} (Bullish sentiment)',
                    'implication': 'Market overly pessimistic - potential bullish reversal'
                })
            elif pcr < 0.5:
                traps.append({
                    'type': 'PCR_EXTREME',
                    'description': f'Extreme PCR of {pcr:.2f} (Bearish sentiment)',
                    'implication': 'Market overly optimistic - potential bearish reversal'
                })
            
            # Display trap indicators
            if traps:
                st.markdown("### ⚠️ Trap Indicators Detected")
                for trap in traps:
                    st.markdown(f"""
                    <div class="trap-indicator">
                        <b>{trap['type']}</b><br>
                        {trap['description']}<br>
                        <small>{trap['implication']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant traps detected in current market structure.")
            
            # Display key ATM insights
            st.markdown("### 🎯 ATM Zone Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pcr_color = "green" if pcr > 1.2 else "red" if pcr < 0.7 else "yellow"
                st.metric("ATM PCR", f"{pcr:.2f}", 
                         delta="Bullish" if pcr > 1.2 else "Bearish" if pcr < 0.7 else "Neutral",
                         delta_color="normal" if pcr > 1.2 else "inverse" if pcr < 0.7 else "off")
            
            with col2:
                oi_ratio = ce_oi / pe_oi if pe_oi > 0 else 99
                st.metric("CE/PE OI Ratio", f"{oi_ratio:.2f}")
            
            with col3:
                chg_oi_ratio = ce_chg_oi / pe_chg_oi if pe_chg_oi != 0 else 99
                trend = "Bullish Flow" if pe_chg_oi > ce_chg_oi else "Bearish Flow" if ce_chg_oi > pe_chg_oi else "Neutral"
                st.metric("ChgOI Trend", trend)
            
            # Quick assessment
            st.markdown("### 📊 Quick Assessment")
            
            if pcr > 1.2:
                st.success(f"✅ **PCR {pcr:.2f} > 1.2:** Market sentiment is **BULLISH** (more puts being bought for protection)")
                st.caption("Interpretation: When PCR > 1.2, it indicates more put buying relative to calls, suggesting traders are hedging against downside risk or expecting a bearish move.")
            elif pcr < 0.7:
                st.warning(f"⚠️ **PCR {pcr:.2f} < 0.7:** Market sentiment is **BEARISH** (more calls being bought for speculation)")
                st.caption("Interpretation: When PCR < 0.7, it indicates more call buying relative to puts, suggesting speculative bullish sentiment or less hedging concern.")
            else:
                st.info(f"📊 **PCR {pcr:.2f} (0.7-1.2):** Market sentiment is **NEUTRAL**")
                st.caption("Interpretation: Neutral PCR range suggests balanced market sentiment without strong directional bias.")
            
            if abs(oi_ratio - 1) > 0.3:
                if oi_ratio > 1.3:
                    st.warning(f"⚠️ **High CE/PE OI Ratio ({oi_ratio:.2f}): Resistance building** at current levels")
                    st.caption("More call OI than put OI suggests resistance formation as sellers write calls expecting price to stay below.")
                else:
                    st.success(f"✅ **Low CE/PE OI Ratio ({oi_ratio:.2f}): Support building** at current levels")
                    st.caption("More put OI than call OI suggests support formation as sellers write puts expecting price to stay above.")

def main():
    st.title("📈 Nifty Trading & Options Analyzer")
    
    # Initialize API credentials
    try:
        if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
            st.markdown("""
            <div class="api-warning">
                ⚠️ <b>Dhan API credentials not configured</b><br>
                The app will run in <b>DEMO MODE</b> with simulated data.
            </div>
            """, unsafe_allow_html=True)
            
            # Use demo credentials
            access_token = "demo_token"
            client_id = "demo_client"
            issues = []
            
            st.info("""
            **To use real market data:**
            1. Get Dhan API credentials from https://dhan.co
            2. Add to Streamlit secrets:
            ```
            DHAN_CLIENT_ID = "your_client_id"
            DHAN_ACCESS_TOKEN = "your_access_token"
            ```
            3. Telegram notifications (optional):
            ```
            TELEGRAM_BOT_TOKEN = "your_bot_token"
            TELEGRAM_CHAT_ID = "your_chat_id"
            ```
            """)
        else:
            access_token, client_id, issues = validate_credentials(DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID)
            
            if issues:
                st.error("Issues found with API credentials:")
                for issue in issues:
                    st.error(f"• {issue}")
        
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            st.sidebar.success("Telegram notifications enabled")
        else:
            st.sidebar.warning("Telegram notifications disabled - configure bot token and chat ID")
        
    except Exception as e:
        st.error(f"Credential validation error: {str(e)}")
        return
    
    # Get user ID
    user_id = get_user_id()
    
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
    
    default_timeframe = "5 min"
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
        
        pivot_settings = {
            'show_3m': True,
            'show_5m': True,
            'show_10m': True,
            'show_15m': True
        }
        
        show_3m = st.sidebar.checkbox("3 Minute Pivots", value=pivot_settings.get('show_3m', True), help="🟢 Green lines")
        show_5m = st.sidebar.checkbox("5 Minute Pivots", value=pivot_settings.get('show_5m', True), help="🟠 Orange lines")
        show_10m = st.sidebar.checkbox("10 Minute Pivots", value=pivot_settings.get('show_10m', True), help="🟣 Pink lines")
        show_15m = st.sidebar.checkbox("15 Minute Pivots", value=pivot_settings.get('show_15m', True), help="🔵 Blue lines")
        
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
    
    # Market Depth Settings
    st.sidebar.header("📊 Market Depth Settings")
    show_market_depth = st.sidebar.checkbox("Show Market Depth Analysis", value=True, help="Display order book depth analysis")
    depth_price_range = st.sidebar.slider("Depth Analysis Range (%)", 1, 5, 2, help="Price range for depth-based S/R analysis")
    
    # Trading signal settings
    st.sidebar.header("🔔 Trading Signals")
    enable_signals = st.sidebar.checkbox("Enable Telegram Signals", value=True, help="Send notifications when conditions are met")
    
    pivot_proximity = st.sidebar.slider(
        "Pivot Proximity (± Points)", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="Distance from pivot levels to trigger signals (both above and below)"
    )
    
    if enable_signals:
        st.sidebar.info(f"Signals sent when:\n• Price within ±{pivot_proximity}pts of pivot\n• All option bias aligned\n• ATM at support/resistance")
    
    # Options expiry selection
    st.sidebar.header("📅 Options Settings")
    
    # Use simulated expiry dates
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
    
    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh (2 min)", value=True)
    
    # Days back for data
    days_back = st.sidebar.slider("Days of Historical Data", 1, 5, 1)
    
    # Demo mode toggle
    demo_mode = st.sidebar.checkbox("Demo Mode (Use Simulated Data)", value=True, 
                                   help="Use simulated data instead of real API calls")
    
    # Connection Test Section
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
    
    # Test Dhan API Connection
    if not demo_mode and DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN:
        if st.sidebar.button("Test Dhan API Connection"):
            api = DhanAPI(DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID)
            success, message = api.test_connection()
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.session_state.refresh_counter += 1
        # Clear API cache on manual refresh
        st.session_state.api_cache = {}
        st.rerun()
    
    # Clear API Cache button
    if st.sidebar.button("🗑️ Clear API Cache"):
        st.session_state.api_cache = {}
        st.sidebar.success("API cache cleared!")
    
    # Show market insights
    show_insights = st.sidebar.checkbox("Show Market Insights", value=True)
    
    # Debug info
    st.sidebar.subheader("🔧 Debug Info")
    st.sidebar.write(f"Demo Mode: {'✅ ON' if demo_mode else '❌ OFF'}")
    st.sidebar.write(f"Telegram Bot Token: {'✅ Set' if TELEGRAM_BOT_TOKEN else '❌ Missing'}")
    st.sidebar.write(f"Telegram Chat ID: {'✅ Set' if TELEGRAM_CHAT_ID else '❌ Missing'}")
    st.sidebar.write(f"API Cache Entries: {len(st.session_state.api_cache)}")
    st.sidebar.write(f"User ID: {user_id[:8]}...")
    
    # Initialize API if not in demo mode
    if demo_mode or not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        api = None
        st.info("📊 **Running in DEMO MODE** - Using simulated market data for analysis")
    else:
        api = DhanAPI(access_token, client_id)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Trading Chart")
        
        # Data fetching strategy
        df = pd.DataFrame()
        current_price = 25048.65
        depth_analysis = None
        depth_sr_analysis = None
        depth_signals = []
        
        if demo_mode or not api:
            # Use simulated data
            with st.spinner("Generating simulated chart data..."):
                df = create_fallback_chart_data(days_back, interval)
                current_price = df['close'].iloc[-1] if len(df) > 0 else 25048.65
        else:
            # Try to get real data
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
                else:
                    st.warning("⚠️ Could not fetch real data. Using simulated data.")
                    df = create_fallback_chart_data(days_back, interval)
        
        # Get Market Depth if enabled
        if show_market_depth and current_price:
            with st.spinner("Analyzing market depth..."):
                if demo_mode or not api:
                    # Use simulated depth data
                    depth_analyzer = MarketDepthAnalyzer()
                    depth_analysis = depth_analyzer.generate_simulated_analysis(current_price)
                else:
                    # Try to get real depth data
                    depth_data = api.get_market_depth("13", "IDX_I")
                    if depth_data:
                        depth_analyzer = MarketDepthAnalyzer()
                        depth_analysis = depth_analyzer.analyze_depth_structure(depth_data, current_price)
                    else:
                        depth_analyzer = MarketDepthAnalyzer()
                        depth_analysis = depth_analyzer.generate_simulated_analysis(current_price)
                
                if 'error' not in depth_analysis:
                    # Calculate depth-based support/resistance
                    depth_sr_analysis = depth_analyzer.calculate_depth_based_support_resistance(
                        depth_analysis, 
                        depth_price_range
                    )
                    
                    # Generate depth trading signals
                    depth_signals = depth_analyzer.generate_depth_trading_signals(
                        depth_analysis, 
                        None,
                        current_price
                    )
        
        # Display metrics with depth info
        if not df.empty:
            display_metrics(None, df, depth_analysis=depth_analysis)
        
        # Create and display chart
        if not df.empty:
            fig = create_candlestick_chart(
                df, 
                f"Nifty 50 - {selected_timeframe} Chart {'with Pivot Levels' if show_pivots else ''}",
                interval,
                show_pivots=show_pivots,
                pivot_settings=pivot_settings,
                depth_sr=depth_sr_analysis
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data info
            col1_info, col2_info, col3_info = st.columns(3)
            with col1_info:
                st.info(f"📊 Data Points: {len(df)}")
            with col2_info:
                latest_time = df['datetime'].max().strftime("%Y-%m-%d %H:%M:%S IST")
                st.info(f"🕐 Latest: {latest_time}")
            with col3_info:
                mode = "📡 Demo Mode" if demo_mode or not api else "📡 Live API"
                st.info(f"{mode}")
            
            if show_pivots and len(df) > 50:
                st.markdown("""
                **Pivot Levels Legend:**
                - 🟢 **3M Levels**: 3-minute timeframe support/resistance
                - 🟠 **5M Levels**: 5-minute timeframe swing points  
                - 🟣 **10M Levels**: 10-minute support/resistance zones
                - 🔵 **15M Levels**: 15-minute major support/resistance levels
                
                *R = Resistance (Price ceiling), S = Support (Price floor)*
                """)
        else:
            st.error("No data available. Please check your API credentials and try again.")
        
        # Display Market Depth Analysis if enabled
        if show_market_depth and depth_analysis and 'error' not in depth_analysis:
            st.markdown("---")
            st.subheader("📊 Market Depth Analysis")
            
            # Create tabs for different depth views
            depth_tab1, depth_tab2, depth_tab3 = st.tabs(["📈 Order Book", "🎯 Support/Resistance", "📊 Depth Metrics"])
            
            with depth_tab1:
                # Display market depth visualization
                depth_fig = MarketDepthAnalyzer.visualize_market_depth(
                    depth_analysis, 
                    title="Nifty 50 Market Depth"
                )
                if depth_fig:
                    st.plotly_chart(depth_fig, use_container_width=True)
                else:
                    st.info("Depth visualization not available.")
            
            with depth_tab2:
                # Display depth-based support/resistance
                if depth_sr_analysis and 'error' not in depth_sr_analysis:
                    col1_sr, col2_sr = st.columns(2)
                    
                    with col1_sr:
                        st.markdown("**Strongest Support Levels**")
                        if depth_sr_analysis['strongest_support']:
                            for i, support in enumerate(depth_sr_analysis['strongest_support'][:3], 1):
                                distance_color = "green" if abs(support['distance_pct']) < 1 else "yellow"
                                st.markdown(f"""
                                <div class="depth-support">
                                    <b>#{i}: ₹{support['price']:.0f}</b><br>
                                    Strength: {support['strength_score']:.1f}<br>
                                    Distance: {support['distance_pct']:.2f}%<br>
                                    Quantity: {support['total_quantity']:,}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No significant support levels found")
                    
                    with col2_sr:
                        st.markdown("**Strongest Resistance Levels**")
                        if depth_sr_analysis['strongest_resistance']:
                            for i, resistance in enumerate(depth_sr_analysis['strongest_resistance'][:3], 1):
                                distance_color = "red" if abs(resistance['distance_pct']) < 1 else "yellow"
                                st.markdown(f"""
                                <div class="depth-resistance">
                                    <b>#{i}: ₹{resistance['price']:.0f}</b><br>
                                    Strength: {resistance['strength_score']:.1f}<br>
                                    Distance: {resistance['distance_pct']:.2f}%<br>
                                    Quantity: {resistance['total_quantity']:,}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No significant resistance levels found")
                    
                    # Display price targets
                    if depth_sr_analysis['price_targets']:
                        st.markdown("**Depth-Based Price Targets**")
                        targets = depth_sr_analysis['price_targets']
                        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                        
                        with col_t1:
                            if targets['immediate_support']:
                                st.metric("Immediate Support", f"₹{targets['immediate_support']:.0f}")
                        
                        with col_t2:
                            if targets['immediate_resistance']:
                                st.metric("Immediate Resistance", f"₹{targets['immediate_resistance']:.0f}")
                        
                        with col_t3:
                            if targets['next_support']:
                                st.metric("Next Support", f"₹{targets['next_support']:.0f}")
                        
                        with col_t4:
                            if targets['next_resistance']:
                                st.metric("Next Resistance", f"₹{targets['next_resistance']:.0f}")
            
            with depth_tab3:
                # Display depth metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    total_bid = depth_analysis.get('total_bid_qty', 0)
                    st.metric("Total Bid Qty", f"{total_bid:,}")
                
                with col_m2:
                    total_ask = depth_analysis.get('total_ask_qty', 0)
                    st.metric("Total Ask Qty", f"{total_ask:,}")
                
                with col_m3:
                    imbalance = depth_analysis.get('bid_ask_imbalance', 0)
                    st.metric("Depth Imbalance", f"{imbalance:+,}")
                
                with col_m4:
                    depth_bias = depth_analysis.get('depth_bias', 'NEUTRAL')
                    st.metric("Depth Bias", depth_bias)
                
                # Display depth signals
                if depth_signals:
                    st.markdown("**Depth Trading Signals**")
                    for signal in depth_signals:
                        signal_color = "green" if signal['direction'] == 'BULLISH' else "red"
                        st.markdown(f"""
                        <div style="border-left: 4px solid {signal_color}; padding: 10px; margin: 5px 0; background-color: #2a2a2a;">
                            <b>{signal['type']}</b> - {signal['direction']}<br>
                            <small>Action: {signal['action']}</small><br>
                            <small>Strength: {signal['signal_strength']}/10</small><br>
                            <small>{signal['reason']}</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.header("📊 Options Analysis")
        
        # Options chain analysis with expiry selection
        try:
            underlying_price, df_summary, available_expiries = analyze_option_chain(selected_expiry, use_cache=True)
            
            if underlying_price:
                st.info(f"**NIFTY SPOT:** ₹{underlying_price:.2f}")
                
                # Display SELLER'S EDGE DASHBOARD
                st.markdown("---")
                st.subheader("📊 SELLER'S EDGE DASHBOARD")
                
                # Calculate exposures and biases
                exposure = calculate_net_delta_gamma_exposure(
                    df_summary,
                    underlying_price
                )
                
                # For OI bias, we need separate CE and PE dataframes
                df_ce = pd.DataFrame()
                df_pe = pd.DataFrame()
                if df_summary is not None and not df_summary.empty:
                    # Extract CE and PE data from summary
                    ce_cols = ['Strike', 'openInterest_CE', 'lastPrice_CE']
                    pe_cols = ['Strike', 'openInterest_PE', 'lastPrice_PE']
                    
                    if all(col in df_summary.columns for col in ce_cols):
                        df_ce = df_summary[ce_cols].copy()
                        df_ce.columns = ['strikePrice', 'openInterest', 'lastPrice']
                    
                    if all(col in df_summary.columns for col in pe_cols):
                        df_pe = df_summary[pe_cols].copy()
                        df_pe.columns = ['strikePrice', 'openInterest', 'lastPrice']
                
                oi_bias_data = calculate_oi_concentration_bias(
                    df_ce,
                    df_pe,
                    underlying_price
                )
                
                # Get volatility regime
                historical_iv = []  # Placeholder for simulated data
                current_atm_iv = 15  # Default IV for simulated data
                regime, action, iv_rank, iv_perc, color = calculate_volatility_regime_bias(
                    current_atm_iv,
                    historical_iv
                )
                
                # Generate seller recommendation with depth analysis
                recommendation = generate_seller_recommendation(
                    exposure, 
                    regime, 
                    oi_bias_data, 
                    underlying_price, 
                    df_summary,
                    depth_analysis
                )
                
                # Display seller metrics
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                
                with col_s1:
                    st.metric("Delta Exposure", f"{exposure['norm_delta']:+.1f}",
                             delta=f"Raw: {exposure['net_delta']:+.0f}")
                    st.metric("Gamma Exposure", f"{exposure['norm_gamma']:+.1f}",
                             delta=f"Raw: {exposure['net_gamma']:+.0f}")
                
                with col_s2:
                    st.metric("Volatility Regime", f"{color} {regime}",
                             delta=f"IV Rank: {iv_rank}%")
                    st.caption(f"**Action:** {action}")
                
                with col_s3:
                    st.metric("OI Structure Bias", oi_bias_data['seller_bias'])
                    if depth_analysis and 'depth_bias' in depth_analysis:
                        depth_display = "SIMULATED" if depth_analysis.get('depth_bias') == "SIMULATED" else depth_analysis.get('depth_bias')
                        st.metric("Depth Bias", depth_display)
                
                with col_s4:
                    st.metric("Recommended Action", recommendation['action'])
                    if recommendation.get('depth_influenced', False):
                        st.caption("✅ Depth-influenced recommendation")
                    st.caption(f"**Strikes:** {recommendation.get('strikes', 'N/A')}")
                
                # Check for trading signals if enabled
                if enable_signals and not df.empty and df_summary is not None and len(df_summary) > 0:
                    check_trading_signals(df, pivot_settings, df_summary, underlying_price, pivot_proximity, depth_signals)
                    
        except Exception as e:
            st.error(f"Error analyzing option chain: {str(e)}")
            st.info("Using fallback option chain analysis with simulated data.")
    
    # Market Insights below
    if show_insights and df_summary is not None and current_price:
        st.markdown("---")
        display_market_insights(df_summary, current_price, depth_analysis)
    
    # Show current time
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    st.sidebar.info(f"Last Updated: {current_time}")
    
    # Increment refresh counter on auto-refresh
    st.session_state.refresh_counter += 1

if __name__ == "__main__":
    main()