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
from scipy import stats
import plotly.express as px

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
    supabase_url = ""
    supabase_key = ""
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"

# Add refresh counter to session state
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0

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
        except Exception as e:
            st.warning("Database tables may need to be created. Please run the SQL setup first.")
    
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
            if "column" in str(e) and "does not exist" in str(e):
                return pd.DataFrame()
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
            if "relation" in str(e) and "does not exist" in str(e):
                return {
                    'timeframe': '5', 
                    'auto_refresh': True, 
                    'days_back': 1,
                    'pivot_proximity': 5,
                    'pivot_settings': {
                        'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True
                    }
                }
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
            if "relation" in str(e) and "does not exist" in str(e):
                return pd.DataFrame()
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
    
    def get_market_depth(self, security_id="13", exchange_segment="IDX_I"):
        """Get Market Depth (Order Book) data"""
        url = f"{self.base_url}/marketfeed/quotes"
        
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Market Depth API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching market depth: {str(e)}")
            return None

@st.cache_data(ttl=300)
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
            
            # Extract bid and ask levels
            bids = []
            asks = []
            
            # Parse bid levels (buy orders)
            for i in range(1, 6):  # Typically 5 levels
                bid_key = f'bid{i}'
                if bid_key in data:
                    bid_price = data.get(f'{bid_key}Price', 0)
                    bid_qty = data.get(f'{bid_key}Quantity', 0)
                    if bid_price > 0 and bid_qty > 0:
                        bids.append({
                            'price': bid_price,
                            'quantity': bid_qty,
                            'type': 'bid',
                            'level': i
                        })
            
            # Parse ask levels (sell orders)
            for i in range(1, 6):
                ask_key = f'ask{i}'
                if ask_key in data:
                    ask_price = data.get(f'{ask_key}Price', 0)
                    ask_qty = data.get(f'{ask_key}Quantity', 0)
                    if ask_price > 0 and ask_qty > 0:
                        asks.append({
                            'price': ask_price,
                            'quantity': ask_qty,
                            'type': 'ask',
                            'level': i
                        })
            
            # Calculate depth metrics
            total_bid_qty = sum(b['quantity'] for b in bids)
            total_ask_qty = sum(a['quantity'] for a in asks)
            
            # Identify significant depth levels (walls)
            depth_walls = []
            
            # Strong support walls (large bid quantities)
            for bid in bids:
                if bid['quantity'] > 1000:  # Threshold for significant wall
                    wall_strength = "STRONG" if bid['quantity'] > 5000 else "MODERATE"
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
                if ask['quantity'] > 1000:
                    wall_strength = "STRONG" if ask['quantity'] > 5000 else "MODERATE"
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
                if abs(wall['distance_pct']) < 1:  # Within 1% of current price
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
            return {"error": str(e)}
    
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
            
            # Group nearby support levels
            support_groups = {}
            for wall in support_walls:
                rounded_price = round(wall['price'] / 50) * 50  # Group by 50-point intervals
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
                rounded_price = round(wall['price'] / 50) * 50
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
                strength_score = (group['total_quantity'] / 10000) * (group['wall_count'] ** 0.5)
                support_levels.append({
                    'price': price,
                    'strength_score': round(strength_score, 2),
                    'total_quantity': group['total_quantity'],
                    'wall_count': group['wall_count'],
                    'type': 'SUPPORT',
                    'distance_pct': ((price - current_price) / current_price * 100)
                })
            
            for price, group in resistance_groups.items():
                strength_score = (group['total_quantity'] / 10000) * (group['wall_count'] ** 0.5)
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
        df = option_chain_df.copy()
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

def display_metrics(ltp_data, df, db, symbol="NIFTY50", depth_analysis=None):
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
            
            try:
                db.save_market_analytics(symbol, analytics_data)
            except Exception as e:
                st.warning(f"Could not save analytics: {e}")
            
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

def analyze_option_chain(selected_expiry=None):
    """Enhanced options chain analysis with SELLER'S EDGE dashboard."""
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
    
    if df_ce.empty:
        st.error("No call data available")
        return None, None, expiry_dates
    
    def map_dhan_fields(df, suffix):
        """Map Dhan API field names to our expected column names"""
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
    
    df_ce = map_dhan_fields(df_ce, 'CE')
    df_pe = map_dhan_fields(df_pe, 'PE')
    
    df = pd.merge(df_ce, df_pe, on='strikePrice', how='outer').sort_values('strikePrice')
    
    numeric_cols = [col for col in df.columns if col not in ['strikePrice']]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    T = calculate_exact_time_to_expiry(expiry)
    r = 0.06
    
    if 'impliedVolatility_CE' not in df.columns:
        df['impliedVolatility_CE'] = 15
    
    if 'impliedVolatility_PE' not in df.columns:
        df['impliedVolatility_PE'] = 15
    
    if 'openInterest_CE' not in df.columns:
        df['openInterest_CE'] = 0
    
    if 'openInterest_PE' not in df.columns:
        df['openInterest_PE'] = 0
    
    for idx, row in df.iterrows():
        strike = row['strikePrice']
        
        iv_ce = row.get('impliedVolatility_CE', 15)
        iv_pe = row.get('impliedVolatility_PE', 15)
        
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
    if not atm_plus_minus_2.empty:
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
    
    required_summary_cols = ['strikePrice', 'openInterest_CE', 'openInterest_PE']
    available_cols = [col for col in required_summary_cols if col in df.columns]
    
    if available_cols:
        df_summary = pd.merge(
            df_summary,
            df[available_cols],
            left_on='Strike', right_on='strikePrice', how='left'
        )
    else:
        df_summary['openInterest_CE'] = 0
        df_summary['openInterest_PE'] = 0

    df_summary['PCR'] = df_summary['openInterest_PE'] / df_summary['openInterest_CE']
    df_summary['PCR'] = np.where(df_summary['openInterest_CE'] == 0, 0, df_summary['PCR'])
    df_summary['PCR'] = df_summary['PCR'].round(2)
    
    df_summary['PCR_Signal'] = np.where(
        df_summary['PCR'] > 1.2, "BEARISH_SENTIMENT",
        np.where(df_summary['PCR'] < 0.7, "BULLISH_SENTIMENT", "NEUTRAL")
    )

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
        value=user_prefs.get('pivot_proximity', 5),
        help="Distance from pivot levels to trigger signals (both above and below)"
    )
    
    if enable_signals:
        st.sidebar.info(f"Signals sent when:\n• Price within ±{pivot_proximity}pts of pivot\n• All option bias aligned\n• ATM at support/resistance")
    
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
        st.session_state.refresh_counter += 1
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
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Trading Chart")
        
        # Data fetching strategy
        df = pd.DataFrame()
        current_price = None
        depth_analysis = None
        depth_sr_analysis = None
        depth_signals = []
        
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
                        try:
                            db.save_candle_data("NIFTY50", "IDX_I", interval, df)
                        except Exception as e:
                            st.warning(f"Could not save candle data: {e}")
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
                    try:
                        db.save_candle_data("NIFTY50", "IDX_I", interval, df)
                    except Exception as e:
                        st.warning(f"Could not save candle data: {e}")
        
        # Get LTP data and current price
        ltp_data = api.get_ltp_data("13", "IDX_I")
        if ltp_data and 'data' in ltp_data:
            for exchange, data in ltp_data['data'].items():
                for security_id, price_data in data.items():
                    current_price = price_data.get('last_price', 0)
                    break
        
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
        # Get Market Depth if enabled
        if show_market_depth and current_price:
            with st.spinner("Fetching market depth..."):
                depth_data = api.get_market_depth("13", "IDX_I")
                if depth_data:
                    depth_analyzer = MarketDepthAnalyzer()
                    depth_analysis = depth_analyzer.analyze_depth_structure(depth_data, current_price)
                    
                    if 'error' not in depth_analysis:
                        # Calculate depth-based support/resistance
                        depth_sr_analysis = depth_analyzer.calculate_depth_based_support_resistance(
                            depth_analysis, 
                            depth_price_range
                        )
                        
                        # Generate depth trading signals
                        depth_signals = depth_analyzer.generate_depth_trading_signals(
                            depth_analysis, 
                            None,  # Option data can be added later
                            current_price
                        )
        
        # Display metrics with depth info
        if not df.empty:
            display_metrics(ltp_data, df, db, depth_analysis=depth_analysis)
        
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
            underlying_price, df_summary, available_expiries = analyze_option_chain(selected_expiry)
            
            if underlying_price:
                st.info(f"**NIFTY SPOT:** {underlying_price:.2f}")
                
                # Display SELLER'S EDGE DASHBOARD
                st.markdown("---")
                st.subheader("📊 SELLER'S EDGE DASHBOARD")
                
                # Get option chain data for exposure calculation
                option_chain_data = get_dhan_option_chain(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, selected_expiry or available_expiries[0])
                if option_chain_data and 'data' in option_chain_data:
                    data = option_chain_data['data']
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
                    
                    # Calculate exposures and biases
                    exposure = calculate_net_delta_gamma_exposure(
                        pd.merge(df_ce, df_pe, on='strikePrice', how='outer').sort_values('strikePrice'),
                        underlying_price
                    )
                    
                    oi_bias_data = calculate_oi_concentration_bias(df_ce, df_pe, underlying_price)
                    
                    # Get volatility regime
                    historical_iv = []  # Placeholder
                    current_atm_iv = 15  # Default
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
                            st.metric("Depth Bias", depth_analysis['depth_bias'])
                    
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
            st.info("Please check your Dhan API credentials and try again.")
    
    # Analytics dashboard below
    if show_analytics:
        st.markdown("---")
        display_analytics_dashboard(db)
    
    # Show current time
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    st.sidebar.info(f"Last Updated: {current_time}")
    
    # Increment refresh counter on auto-refresh
    st.session_state.refresh_counter += 1

if __name__ == "__main__":
    main()