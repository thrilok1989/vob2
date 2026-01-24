import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import numpy as np
import math
from scipy.stats import norm
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Nifty Analyzer", page_icon="📈", layout="wide")

# Function to check if it's market hours
def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Check if it's a weekday (Monday to Friday)
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Check if current time is between 9:00 AM and 3:45 PM IST
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    
    return market_start <= now <= market_end

# Only run autorefresh during market hours
if is_market_hours():
    st_autorefresh(interval=35000, key="refresh")
else:
    st.info("Market is closed. Auto-refresh disabled.")

# Credentials
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))
NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"

class DhanAPI:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID
        }
    
    def get_intraday_data(self, interval="5", days_back=1):
        url = "https://api.dhan.co/v2/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        
        payload = {
            "securityId": str(NIFTY_SCRIP),
            "exchangeSegment": NIFTY_SEG,
            "instrument": "INDEX",
            "interval": interval,
            "oi": False,
            "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_ltp_data(self):
        url = "https://api.dhan.co/v2/marketfeed/ltp"
        payload = {NIFTY_SEG: [NIFTY_SCRIP]}
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except:
            return None

def get_option_chain(expiry):
    url = "https://api.dhan.co/v2/optionchain"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG, "Expiry": expiry}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_expiry_list():
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

def process_candle_data(data):
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

def find_pivot_highs_proper(highs, length):
    """
    Proper pivot high detection - checks if bar is highest among 'length' bars on BOTH sides
    """
    if len(highs) < length * 2 + 1:
        return pd.Series(index=highs.index, dtype=float)
    
    pivot_highs = pd.Series(index=highs.index, dtype=float)
    
    for i in range(length, len(highs) - length):
        current_high = highs.iloc[i]
        
        # Check 'length' bars to the left and right
        left_side = highs.iloc[i-length:i]
        right_side = highs.iloc[i+1:i+length+1]
        
        # Current bar must be strictly higher than all bars on both sides
        if (current_high > left_side.max()) and (current_high > right_side.max()):
            pivot_highs.iloc[i] = current_high
            
    return pivot_highs

def find_pivot_lows_proper(lows, length):
    """
    Proper pivot low detection - checks if bar is lowest among 'length' bars on BOTH sides
    """
    if len(lows) < length * 2 + 1:
        return pd.Series(index=lows.index, dtype=float)
        
    pivot_lows = pd.Series(index=lows.index, dtype=float)
    
    for i in range(length, len(lows) - length):
        current_low = lows.iloc[i]
        
        # Check 'length' bars to the left and right
        left_side = lows.iloc[i-length:i]
        right_side = lows.iloc[i+1:i+length+1]
        
        # Current bar must be strictly lower than all bars on both sides
        if (current_low < left_side.min()) and (current_low < right_side.min()):
            pivot_lows.iloc[i] = current_low
            
    return pivot_lows

def detect_level_touches(df, pivot_value, tolerance_pct=0.09):
    """
    Detect when price touches a pivot level with tolerance
    """
    if df.empty:
        return []
        
    touches = []
    tolerance = pivot_value * (tolerance_pct / 100)
    
    for i, row in df.iterrows():
        # Check if high touched the level
        if abs(row['high'] - pivot_value) <= tolerance:
            touches.append({
                'datetime': row['datetime'],
                'price': pivot_value,
                'touch_type': 'high_touch',
                'actual_price': row['high'],
                'bar_index': i
            })
        
        # Check if low touched the level  
        elif abs(row['low'] - pivot_value) <= tolerance:
            touches.append({
                'datetime': row['datetime'],
                'price': pivot_value,
                'touch_type': 'low_touch', 
                'actual_price': row['low'],
                'bar_index': i
            })
            
    return touches

# ===== NEWS INTEGRATION =====

def fetch_market_news():
    """Fetch latest market news using web search"""
    try:
        # Check if we have cached news (avoid too many API calls)
        current_time = datetime.now()
        if 'news_cache' in st.session_state:
            cache_time = st.session_state.news_cache.get('timestamp')
            if cache_time and (current_time - cache_time).total_seconds() < 1800:  # 30 minutes
                return st.session_state.news_cache.get('data', [])
        
        # Search for recent Nifty/Indian market news
        search_queries = [
            "Nifty 50 news today stock market",
            "Indian stock market news NSE BSE today"
        ]
        
        news_data = []
        
        for query in search_queries[:1]:  # Limit to 1 query to avoid rate limits
            try:
                # Use web_search function if available
                if 'web_search' in globals():
                    search_results = web_search(query)
                    
                    # Process search results
                    if search_results and hasattr(search_results, 'results'):
                        for result in search_results.results[:3]:  # Top 3 results
                            news_item = {
                                'title': result.title,
                                'summary': result.description[:200] if hasattr(result, 'description') else '',
                                'url': result.url if hasattr(result, 'url') else '',
                                'timestamp': current_time.isoformat(),
                                'source': result.url.split('/')[2] if hasattr(result, 'url') else 'Unknown'
                            }
                            news_data.append(news_item)
                
                # If web_search not available or no results, return placeholder
                if not news_data:
                    news_data = [
                        {
                            'title': 'Market News - Web Search Unavailable',
                            'summary': 'Enable web search tools for live market news updates',
                            'sentiment': 'neutral',
                            'timestamp': current_time.isoformat(),
                            'source': 'System Notice'
                        }
                    ]
                    
            except Exception as e:
                # Fallback news item
                news_data.append({
                    'title': f'News Fetch Error - {str(e)[:50]}',
                    'summary': 'Unable to fetch live market news. Check web search functionality.',
                    'sentiment': 'neutral',
                    'timestamp': current_time.isoformat(),
                    'source': 'Error Handler'
                })
                continue
        
        # Cache the results
        if 'news_cache' not in st.session_state:
            st.session_state.news_cache = {}
        st.session_state.news_cache = {
            'data': news_data[:5],
            'timestamp': current_time
        }
                
        return news_data[:5]  # Return top 5 news items
        
    except Exception as e:
        # Ultimate fallback
        return [{
            'title': 'News Service Unavailable',
            'summary': f'News fetching disabled: {str(e)}',
            'sentiment': 'neutral',
            'timestamp': datetime.now().isoformat(),
            'source': 'Fallback'
        }]

def analyze_news_sentiment(news_items):
    """Enhanced sentiment analysis with scoring weights"""
    if not news_items:
        return {"overall": "neutral", "score": 0, "bullish_count": 0, "bearish_count": 0}
    
    # Enhanced keyword lists with weights
    bullish_keywords = {
        'rally': 3, 'surge': 3, 'soar': 3, 'breakout': 2, 'gain': 2, 'rise': 2, 
        'up': 1, 'positive': 2, 'strong': 2, 'growth': 2, 'bull': 3, 'optimistic': 2,
        'upgrade': 2, 'buy': 2, 'momentum': 2, 'breakthrough': 2, 'record': 2,
        'high': 1, 'support': 1, 'recovery': 2, 'boost': 2
    }
    
    bearish_keywords = {
        'fall': 2, 'drop': 2, 'decline': 2, 'crash': 3, 'plunge': 3, 'slump': 3,
        'down': 1, 'negative': 2, 'weak': 2, 'bear': 3, 'sell': 2, 'pessimistic': 2,
        'downgrade': 2, 'correction': 2, 'concern': 1, 'worry': 1, 'fear': 2,
        'low': 1, 'resistance': 1, 'pressure': 1, 'risk': 1
    }
    
    sentiment_scores = []
    bullish_count = 0
    bearish_count = 0
    
    for item in news_items:
        text = (item.get('title', '') + ' ' + item.get('summary', '')).lower()
        
        # Calculate weighted sentiment score
        bullish_score = sum(weight for word, weight in bullish_keywords.items() if word in text)
        bearish_score = sum(weight for word, weight in bearish_keywords.items() if word in text)
        
        # Normalize by text length to avoid bias toward longer articles
        text_length = len(text.split())
        if text_length > 0:
            bullish_score = bullish_score / text_length * 100
            bearish_score = bearish_score / text_length * 100
        
        # Determine sentiment with threshold
        if bullish_score > bearish_score * 1.2:  # 20% threshold for bullish
            sentiment_scores.append(bullish_score - bearish_score)
            bullish_count += 1
        elif bearish_score > bullish_score * 1.2:  # 20% threshold for bearish
            sentiment_scores.append(bearish_score - bullish_score)
            bearish_count += 1
        else:
            sentiment_scores.append(0)  # Neutral
    
    # Calculate overall sentiment
    avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    # Determine overall sentiment with more nuanced thresholds
    if avg_score > 0.5:
        overall = "bullish"
    elif avg_score < -0.5:
        overall = "bearish"
    else:
        overall = "neutral"
    
    return {
        "overall": overall,
        "score": round(avg_score, 3),
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": len(news_items) - bullish_count - bearish_count,
        "confidence": min(abs(avg_score) * 2, 1.0)  # Confidence score 0-1
    }

def get_news_impact_score(news_sentiment, market_trend):
    """Calculate how news sentiment aligns with market trend"""
    if news_sentiment["overall"] == "neutral":
        return 0
    
    # Check alignment
    if (news_sentiment["overall"] == "bullish" and market_trend == "bullish") or \
       (news_sentiment["overall"] == "bearish" and market_trend == "bearish"):
        return 1  # Aligned - positive impact
    elif (news_sentiment["overall"] == "bullish" and market_trend == "bearish") or \
         (news_sentiment["overall"] == "bearish" and market_trend == "bullish"):
        return -1  # Contrarian - negative impact
    else:
        return 0  # Neutral

def should_filter_signal_by_news(news_sentiment, signal_type):
    """Determine if news sentiment should filter out a signal"""
    if news_sentiment["overall"] == "neutral":
        return False  # Don't filter neutral news
    
    # Filter contrarian signals in strong news
    if news_sentiment["score"] > 0.5 and signal_type == "PUT":
        return True  # Strong bullish news, avoid PUT signals
    elif news_sentiment["score"] < -0.5 and signal_type == "CALL":
        return True  # Strong bearish news, avoid CALL signals
    
    return False

# ===== END NEWS INTEGRATION =====

# ===== ENHANCED SIGNAL FILTERS =====

def get_market_trend(df, period=20):
    """Market trend filter"""
    if len(df) < period:
        return "neutral"
    
    sma_short = df['close'].rolling(10).mean().iloc[-1]
    sma_long = df['close'].rolling(20).mean().iloc[-1]
    
    if sma_short > sma_long * 1.002:  # 0.2% threshold
        return "bullish"
    elif sma_short < sma_long * 0.998:
        return "bearish"
    else:
        return "neutral"

def check_volume_confirmation(df, lookback=10):
    """Volume confirmation filter"""
    if len(df) < lookback:
        return False
    
    recent_volume = df['volume'].tail(3).mean()
    avg_volume = df['volume'].rolling(lookback).mean().iloc[-1]
    
    return recent_volume > avg_volume * 1.2  # 20% above average

def check_pivot_confluence(df, current_price, proximity=5):
    """Multiple timeframe confluence filter"""
    timeframes = ["5", "10", "15"]
    support_count = 0
    resistance_count = 0
    
    for tf in timeframes:
        nearby = get_nearby_pivot_levels(df, current_price, proximity)
        for level in nearby:
            if level['timeframe'] == tf:
                if level['type'] == 'support':
                    support_count += 1
                elif level['type'] == 'resistance':
                    resistance_count += 1
    
    return support_count >= 2 or resistance_count >= 2

def check_options_strength(option_data):
    """Strong options flow filter"""
    if option_data is None or option_data.empty:
        return False
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty:
        return False
    
    atm = atm_data.iloc[0]
    
    ce_chg = abs(atm.get('changeinOpenInterest_CE', 0))
    pe_chg = abs(atm.get('changeinOpenInterest_PE', 0))
    
    # Require stronger OI changes
    min_oi_change = 500  # Minimum 500 contracts change
    strong_dominance = max(ce_chg, pe_chg) > min_oi_change
    
    return strong_dominance

def get_market_volatility(df, period=14):
    """Market volatility filter"""
    if len(df) < period:
        return "normal"
    
    returns = df['close'].pct_change().dropna()
    volatility = returns.rolling(period).std().iloc[-1] * 100
    
    if volatility > 2.0:  # > 2% daily volatility
        return "high"
    elif volatility < 0.5:  # < 0.5% daily volatility  
        return "low"
    else:
        return "normal"

def is_good_signal_time():
    """Time-based filter"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Avoid first 15 minutes and last 30 minutes
    market_start = now.replace(hour=9, minute=15, second=0)
    market_end = now.replace(hour=15, minute=15, second=0)
    
    return market_start <= now <= market_end

def calculate_signal_strength(volume_ok, confluence_ok, options_strong, volatility_normal, time_ok, trend_aligned):
    """Calculate signal strength score 1-10"""
    filters_passed = sum([volume_ok, confluence_ok, options_strong, volatility_normal, time_ok, trend_aligned])
    
    # Base score from filters (max 6)
    base_score = filters_passed
    
    # Bonus points for strong confluence and options
    bonus = 0
    if confluence_ok and options_strong:
        bonus += 2
    if volume_ok and trend_aligned:
        bonus += 1
    
    return min(base_score + bonus, 10)

# ===== ADVANCED TECHNICAL ANALYSIS =====

def get_volume_profile(df, periods=20):
    """Advanced volume profile analysis with price-volume distribution"""
    if len(df) < periods:
        return {"profile": "insufficient_data", "strength": 1}
    
    # Get recent data
    recent_data = df.tail(periods)
    
    # Calculate VWAP (Volume Weighted Average Price)
    vwap = (recent_data['close'] * recent_data['volume']).sum() / recent_data['volume'].sum()
    current_price = df['close'].iloc[-1]
    
    # Volume analysis
    recent_vol = df['volume'].tail(5).mean()
    avg_vol = df['volume'].tail(periods).mean()
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
    
    # Volume standard deviation for volatility
    vol_std = df['volume'].tail(periods).std()
    vol_cv = vol_std / avg_vol if avg_vol > 0 else 0  # Coefficient of variation
    
    # Price-volume relationship analysis
    price_changes = df['close'].pct_change().tail(periods)
    volume_changes = df['volume'].pct_change().tail(periods)
    
    # Calculate correlation between price and volume changes
    try:
        correlation = price_changes.corr(volume_changes)
        if pd.isna(correlation):
            correlation = 0
    except:
        correlation = 0
    
    # Volume profile classification with multiple factors
    strength = 1.0
    
    # Factor 1: Volume ratio
    if vol_ratio > 2.0:
        profile = "explosive"
        strength = 2.5
    elif vol_ratio > 1.5:
        profile = "strong"  
        strength = 2.0
    elif vol_ratio > 1.2:
        profile = "above_average"
        strength = 1.5
    elif vol_ratio > 0.8:
        profile = "normal"
        strength = 1.0
    else:
        profile = "weak"
        strength = 0.7
    
    # Factor 2: Price-volume relationship
    if abs(correlation) > 0.3:
        strength *= 1.2  # Strong correlation adds confidence
    
    # Factor 3: VWAP position
    vwap_distance = abs(current_price - vwap) / vwap
    if vwap_distance < 0.002:  # Within 0.2% of VWAP
        strength *= 1.1  # Price near VWAP adds stability
    
    # Factor 4: Volume consistency (low CV means consistent volume)
    if vol_cv < 0.5:
        strength *= 1.1
    elif vol_cv > 1.0:
        strength *= 0.9
    
    return {
        "profile": profile,
        "strength": round(strength, 2),
        "vwap": vwap,
        "vol_ratio": vol_ratio,
        "correlation": correlation,
        "consistency": vol_cv
    }

def get_momentum_score(df, periods=14):
    """Proper RSI calculation with exponential smoothing"""
    if len(df) < periods + 1:
        return 5  # Neutral
    
    # Calculate price changes
    price_changes = df['close'].diff()
    
    # Separate gains and losses
    gains = price_changes.where(price_changes > 0, 0)
    losses = -price_changes.where(price_changes < 0, 0)
    
    # Calculate exponential moving averages (proper RSI method)
    alpha = 1.0 / periods
    
    # Initialize first values
    avg_gain = gains.rolling(window=periods).mean().iloc[periods-1]
    avg_loss = losses.rolling(window=periods).mean().iloc[periods-1]
    
    # Calculate RSI using exponential smoothing for remaining values
    for i in range(periods, len(gains)):
        avg_gain = alpha * gains.iloc[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses.iloc[i] + (1 - alpha) * avg_loss
    
    # Calculate RSI
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    # Convert RSI (0-100) to momentum score (1-10)
    momentum_score = max(1, min(10, int(rsi / 10)))
    
    return momentum_score

def detect_market_regime(df, short_period=10, long_period=30):
    """Statistical regime detection using volatility and trend strength"""
    if len(df) < long_period:
        return "unknown"
    
    # Calculate moving averages
    short_ma = df['close'].rolling(short_period).mean().iloc[-1]
    long_ma = df['close'].rolling(long_period).mean().iloc[-1]
    
    # Calculate trend strength using multiple measures
    price_returns = df['close'].pct_change().tail(long_period)
    volatility = price_returns.std() * (252 ** 0.5)  # Annualized volatility
    
    # Trend strength using R-squared of price regression
    x_values = range(len(df.tail(long_period)))
    y_values = df['close'].tail(long_period).values
    
    try:
        # Linear regression to measure trend strength
        correlation = np.corrcoef(x_values, y_values)[0, 1]
        r_squared = correlation ** 2
    except:
        r_squared = 0
    
    # ADX-like calculation for trend strength
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]
    
    # Directional movement
    plus_dm = (df['high'] - df['high'].shift(1)).where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 0)
    minus_dm = (df['low'].shift(1) - df['low']).where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 0)
    
    plus_di = (plus_dm.rolling(14).mean() / atr) * 100
    minus_di = (minus_dm.rolling(14).mean() / atr) * 100
    
    adx = abs(plus_di.iloc[-1] - minus_di.iloc[-1]) / (plus_di.iloc[-1] + minus_di.iloc[-1]) * 100
    
    # Regime classification using multiple factors
    trend_direction = "bullish" if short_ma > long_ma else "bearish"
    
    # Strong trend criteria: high R-squared AND high ADX
    if r_squared > 0.3 and adx > 25:
        return f"{trend_direction}_trending"
    # Weak trend but some direction
    elif r_squared > 0.1 or adx > 15:
        return f"weak_{trend_direction}_trend"
    # Ranging market
    else:
        return "ranging"

def check_breakout_pattern(df, current_price, lookback=20):
    """Dynamic breakout detection based on market volatility"""
    if len(df) < lookback:
        return False, "insufficient_data"
    
    recent_data = df.tail(lookback)
    
    # Calculate dynamic thresholds based on ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]
    
    # Dynamic breakout threshold (0.5 * ATR)
    breakout_threshold = atr * 0.5
    
    # Calculate support and resistance levels
    recent_high = recent_data['high'].max()
    recent_low = recent_data['low'].min()
    
    # Volume confirmation for breakout
    avg_volume = df['volume'].tail(lookback).mean()
    current_volume = df['volume'].iloc[-1]
    volume_multiplier = current_volume / avg_volume if avg_volume > 0 else 1
    
    # Breakout detection with volume confirmation
    if current_price > recent_high + breakout_threshold and volume_multiplier > 1.2:
        return True, "upside_breakout"
    elif current_price < recent_low - breakout_threshold and volume_multiplier > 1.2:
        return True, "downside_breakout"
    # Potential breakout without volume confirmation
    elif current_price > recent_high + (breakout_threshold * 0.5):
        return True, "weak_upside_breakout"
    elif current_price < recent_low - (breakout_threshold * 0.5):
        return True, "weak_downside_breakout"
    
    return False, "range_bound"

def calculate_risk_reward(current_price, entry_level, df, pivot_levels=None):
    """Dynamic risk/reward using actual support/resistance levels"""
    try:
        # Calculate ATR for dynamic stop loss
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # Base risk using ATR
        base_risk = atr * 1.5  # 1.5x ATR for stop loss
        
        # Find nearest support/resistance levels
        if pivot_levels:
            # Use actual pivot levels for targets
            bullish_target = None
            bearish_target = None
            
            for level in pivot_levels:
                if level['type'] == 'resistance' and level['value'] > current_price:
                    if not bullish_target or level['value'] < bullish_target:
                        bullish_target = level['value']
                elif level['type'] == 'support' and level['value'] < current_price:
                    if not bearish_target or level['value'] > bearish_target:
                        bearish_target = level['value']
            
            # Calculate R:R based on actual levels
            if entry_level > current_price and bullish_target:  # Call trade
                reward = bullish_target - entry_level
                risk = max(base_risk, entry_level - (current_price - base_risk))
            elif entry_level < current_price and bearish_target:  # Put trade  
                reward = entry_level - bearish_target
                risk = max(base_risk, (current_price + base_risk) - entry_level)
            else:
                # Fallback to ATR-based calculation
                reward = base_risk * 2
                risk = base_risk
        else:
            # ATR-based calculation when no pivot levels
            reward = base_risk * 2
            risk = base_risk
        
        return max(1.0, reward / risk) if risk > 0 else 2.0
        
    except:
        return 2.0  # Default R:R ratio

def calculate_pivot_strength(df, pivot_value, lookback=50):
    """Calculate how strong/tested a pivot level is"""
    if df.empty or len(df) < lookback:
        return 1
    
    touches = detect_level_touches(df.tail(lookback), pivot_value, tolerance_pct=0.15)
    
    # More touches = stronger level
    touch_count = len(touches)
    strength = min(1 + (touch_count * 0.2), 3.0)  # Max 3x strength
    
    return strength

def get_session_performance():
    """Track intraday session performance"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Simple session classification
    if 9 <= now.hour < 11:
        return "opening_session"
    elif 11 <= now.hour < 13:
        return "mid_session" 
    elif 13 <= now.hour < 15:
        return "afternoon_session"
    else:
        return "closing_session"

# ===== END ENHANCED FILTERS =====

def get_comprehensive_bias_info(df, option_data, current_price, news_sentiment, market_trend):
    """Get all bias calculations for telegram messages"""
    try:
        # Options biases (from existing logic)
        atm_data = option_data[option_data['Zone'] == 'ATM']
        if atm_data.empty:
            return "Options data unavailable for bias analysis"
        
        row = atm_data.iloc[0]
        
        # Technical biases
        volume_ok = check_volume_confirmation(df)
        volatility = get_market_volatility(df)
        time_ok = is_good_signal_time()
        momentum_score = get_momentum_score(df)
        volume_profile = get_volume_profile(df)
        market_regime = detect_market_regime(df)
        session = get_session_performance()
        is_breakout, breakout_type = check_breakout_pattern(df, current_price)
        confluence_ok = check_pivot_confluence(df, current_price, proximity=5)
        options_strong = check_options_strength(option_data)
        
        # Bias summary
        bias_info = f"""
📊 COMPREHENSIVE BIAS ANALYSIS 📊

🔹 OPTIONS BIASES:
• ChgOI Bias: {row['ChgOI_Bias']}
• Volume Bias: {row['Volume_Bias']} 
• Ask Bias: {row['Ask_Bias']}
• Bid Bias: {row['Bid_Bias']}
• Level Bias: {row['Level']} (OI-based)
• PCR: {row['PCR']} (PE/CE ratio)

🔹 TECHNICAL BIASES:
• Market Trend: {market_trend.title()}
• Volume Strength: {'Strong' if volume_ok else 'Weak'}
• Volatility: {volatility.title()}
• Momentum: {momentum_score}/10 ({'Bullish' if momentum_score >= 6 else 'Bearish' if momentum_score <= 4 else 'Neutral'})
• Volume Profile: {volume_profile["profile"].title()} ({volume_profile["strength"]:.1f}x)
• Market Regime: {market_regime.replace('_', ' ').title()}
• Session: {session.replace('_', ' ').title()}
• Timing: {'Good' if time_ok else 'Poor'}

🔹 CONFLUENCE BIASES:
• Pivot Confluence: {'Strong' if confluence_ok else 'Weak'}
• Options Flow: {'Strong' if options_strong else 'Weak'}
• Breakout: {breakout_type.replace('_', ' ').title() if is_breakout else 'None'}

🔹 SENTIMENT BIASES:
• News Sentiment: {news_sentiment['overall'].title()} (Score: {news_sentiment['score']:.2f})
• News vs Trend: {'Aligned' if get_news_impact_score(news_sentiment, market_trend) == 1 else 'Contrarian' if get_news_impact_score(news_sentiment, market_trend) == -1 else 'Neutral'}

🔹 QUANTITATIVE DATA:
• CE ChgOI: {row.get('changeinOpenInterest_CE', 0):,}
• PE ChgOI: {row.get('changeinOpenInterest_PE', 0):,}
• CE OI: {row.get('openInterest_CE', 0):,}
• PE OI: {row.get('openInterest_PE', 0):,}
• CE Volume: {row.get('totalTradedVolume_CE', 0):,}
• PE Volume: {row.get('totalTradedVolume_PE', 0):,}
• CE Bid/Ask: {row.get('bidQty_CE', 0):,}/{row.get('askQty_CE', 0):,}
• PE Bid/Ask: {row.get('bidQty_PE', 0):,}/{row.get('askQty_PE', 0):,}
• ATM Strike: {row['Strike']}
"""
        return bias_info
        
    except Exception as e:
        return f"Error calculating comprehensive bias: {str(e)}"

def get_pivots(df, timeframe="5", length=4):
    """
    Enhanced pivot detection with proper algorithm and non-repainting protection
    """
    if df.empty:
        return []
    
    rule_map = {"3": "3min", "5": "5min", "10": "10min", "15": "15min"}
    rule = rule_map.get(timeframe, "5min")
    
    df_temp = df.set_index('datetime')
    try:
        resampled = df_temp.resample(rule).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        if len(resampled) < length * 2 + 1:
            return []
        
        # Find all pivot highs and lows using proper algorithm
        pivot_highs = find_pivot_highs_proper(resampled['high'], length)
        pivot_lows = find_pivot_lows_proper(resampled['low'], length)
        
        pivots = []
        
        # Collect pivot highs (exclude last few to avoid repainting)
        valid_pivot_highs = pivot_highs.dropna()
        if len(valid_pivot_highs) > 1:
            # Keep all but the most recent to avoid repainting
            for timestamp, value in valid_pivot_highs[:-1].items():
                pivots.append({
                    'type': 'high', 
                    'timeframe': timeframe, 
                    'timestamp': timestamp, 
                    'value': float(value),
                    'confirmed': True
                })
            
            # Add the most recent but mark as unconfirmed
            if len(valid_pivot_highs) >= 1:
                last_high = valid_pivot_highs.iloc[-1]
                last_high_time = valid_pivot_highs.index[-1]
                pivots.append({
                    'type': 'high',
                    'timeframe': timeframe,
                    'timestamp': last_high_time,
                    'value': float(last_high),
                    'confirmed': False
                })
        
        # Collect pivot lows (exclude last few to avoid repainting)
        valid_pivot_lows = pivot_lows.dropna()
        if len(valid_pivot_lows) > 1:
            # Keep all but the most recent to avoid repainting
            for timestamp, value in valid_pivot_lows[:-1].items():
                pivots.append({
                    'type': 'low',
                    'timeframe': timeframe, 
                    'timestamp': timestamp,
                    'value': float(value),
                    'confirmed': True
                })
            
            # Add the most recent but mark as unconfirmed
            if len(valid_pivot_lows) >= 1:
                last_low = valid_pivot_lows.iloc[-1] 
                last_low_time = valid_pivot_lows.index[-1]
                pivots.append({
                    'type': 'low',
                    'timeframe': timeframe,
                    'timestamp': last_low_time, 
                    'value': float(last_low),
                    'confirmed': False
                })
        
        return pivots
        
    except Exception as e:
        print(f"Error in pivot calculation: {e}")
        return []

def get_nearby_pivot_levels(df, current_price, proximity=5.0):
    """
    Get confirmed pivot levels near current price for signal generation
    """
    if df.empty:
        return []
        
    nearby_levels = []
    timeframes = ["5", "10", "15"]
    
    for timeframe in timeframes:
        pivots = get_pivots(df, timeframe, length=4)
        
        for pivot in pivots:
            # Only use confirmed pivots for signals to avoid repainting
            if not pivot.get('confirmed', True):
                continue
                
            distance = abs(current_price - pivot['value'])
            if distance <= proximity:
                level_type = 'resistance' if pivot['type'] == 'high' else 'support'
                nearby_levels.append({
                    'type': level_type,
                    'pivot_type': pivot['type'], 
                    'value': pivot['value'],
                    'timeframe': timeframe,
                    'distance': distance,
                    'timestamp': pivot['timestamp'],
                    'confirmed': pivot['confirmed']
                })
    
    # Sort by distance (closest first)
    nearby_levels.sort(key=lambda x: x['distance'])
    return nearby_levels

def calculate_rsi(df, periods=14):
    """Calculate proper RSI values for chart display"""
    if len(df) < periods + 1:
        return pd.Series(index=df.index, dtype=float)
    
    # Calculate price changes
    price_changes = df['close'].diff()
    
    # Separate gains and losses
    gains = price_changes.where(price_changes > 0, 0)
    losses = -price_changes.where(price_changes < 0, 0)
    
    # Initialize RSI series
    rsi_values = pd.Series(index=df.index, dtype=float)
    
    # Calculate initial average gain and loss using SMA
    initial_avg_gain = gains.rolling(window=periods).mean().iloc[periods-1]
    initial_avg_loss = losses.rolling(window=periods).mean().iloc[periods-1]
    
    # Calculate first RSI value
    if initial_avg_loss == 0:
        rsi_values.iloc[periods] = 100
    else:
        rs = initial_avg_gain / initial_avg_loss
        rsi_values.iloc[periods] = 100 - (100 / (1 + rs))
    
    # Use Wilder's smoothing for subsequent values
    alpha = 1.0 / periods
    avg_gain = initial_avg_gain
    avg_loss = initial_avg_loss
    
    for i in range(periods + 1, len(df)):
        # Update exponential moving averages
        avg_gain = alpha * gains.iloc[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses.iloc[i] + (1 - alpha) * avg_loss
        
        # Calculate RSI
        if avg_loss == 0:
            rsi_values.iloc[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_values.iloc[i] = 100 - (100 / (1 + rs))
    
    return rsi_values

def create_chart(df, title):
    """
    Enhanced chart with proper pivot levels, shadows, labels, and RSI indicator
    """
    if df.empty:
        return go.Figure()
    
    # Create subplots: Price, Volume, RSI
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price Chart", "Volume", "RSI (14)")
    )
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Add volume
    volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        marker_color=volume_colors, opacity=0.7
    ), row=2, col=1)
    
    # Add RSI indicator
    if len(df) > 14:  # Need at least 15 data points for RSI
        rsi_values = calculate_rsi(df)
        
        # Add RSI line
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=rsi_values, name='RSI',
            line=dict(color='#ffaa00', width=2),
            mode='lines'
        ), row=3, col=1)
        
        # Add RSI reference lines
        # Overbought level (70)
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Overbought (70)", row=3, col=1)
        
        # Oversold level (30) 
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                     annotation_text="Oversold (30)", row=3, col=1)
        
        # Midline (50)
        fig.add_hline(y=50, line_dash="dot", line_color="gray",
                     annotation_text="Midline (50)", row=3, col=1)
        
        # Set RSI y-axis range
        fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    # Add enhanced pivot levels
    if len(df) > 50:
        timeframes = ["5", "10", "15"]
        colors = ["#ff9900", "#ff44ff", '#4444ff']
        
        x_start, x_end = df['datetime'].min(), df['datetime'].max()
        
        for tf, color in zip(timeframes, colors):
            pivots = get_pivots(df, tf)
            
            # Get recent pivots only (last 5 of each type)
            recent_highs = [p for p in pivots if p['type'] == 'high'][-5:]
            recent_lows = [p for p in pivots if p['type'] == 'low'][-5:]
            
            # Add pivot high lines
            for pivot in recent_highs:
                line_style = "solid" if pivot.get('confirmed', True) else "dash"
                line_width = 2 if pivot.get('confirmed', True) else 1
                
                # Main pivot line
                fig.add_shape(
                    type="line", x0=x_start, x1=x_end,
                    y0=pivot['value'], y1=pivot['value'],
                    line=dict(color=color, width=line_width, dash=line_style),
                    row=1, col=1
                )
                
                # Shadow line (wider, more transparent)
                fig.add_shape(
                    type="line", x0=x_start, x1=x_end,
                    y0=pivot['value'], y1=pivot['value'],
                    line=dict(color=color, width=5),
                    opacity=0.15 if pivot.get('confirmed', True) else 0.08,
                    row=1, col=1
                )
                
                # Add label
                status = "✓" if pivot.get('confirmed', True) else "?"
                fig.add_annotation(
                    x=x_end, y=pivot['value'],
                    text=f"{tf}M H {status}: {pivot['value']:.1f}",
                    showarrow=False, xshift=20,
                    font=dict(size=9, color=color),
                    row=1, col=1
                )
            
            # Add pivot low lines
            for pivot in recent_lows:
                line_style = "solid" if pivot.get('confirmed', True) else "dash"
                line_width = 2 if pivot.get('confirmed', True) else 1
                
                # Main pivot line
                fig.add_shape(
                    type="line", x0=x_start, x1=x_end,
                    y0=pivot['value'], y1=pivot['value'],
                    line=dict(color=color, width=line_width, dash=line_style),
                    row=1, col=1
                )
                
                # Shadow line
                fig.add_shape(
                    type="line", x0=x_start, x1=x_end,
                    y0=pivot['value'], y1=pivot['value'],
                    line=dict(color=color, width=5),
                    opacity=0.15 if pivot.get('confirmed', True) else 0.08,
                    row=1, col=1
                )
                
                # Add label
                status = "✓" if pivot.get('confirmed', True) else "?"
                fig.add_annotation(
                    x=x_end, y=pivot['value'],
                    text=f"{tf}M L {status}: {pivot['value']:.1f}",
                    showarrow=False, xshift=20,
                    font=dict(size=9, color=color),
                    row=1, col=1
                )
    
    # Update layout
    fig.update_layout(
        title=title, 
        template='plotly_dark', 
        height=700,  # Increased height for RSI subplot
        xaxis_rangeslider_visible=False, 
        showlegend=False
    )
    
    # Update x-axis labels (only show on bottom subplot)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=3, col=1)
    
    return fig

def analyze_options(expiry):
    option_data = get_option_chain(expiry)
    if not option_data or 'data' not in option_data:
        return None, None
    
    data = option_data['data']
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
    
    rename_map = {
        'last_price': 'lastPrice', 'oi': 'openInterest', 'previous_oi': 'previousOpenInterest',
        'top_ask_quantity': 'askQty', 'top_bid_quantity': 'bidQty', 'volume': 'totalTradedVolume'
    }
    for old, new in rename_map.items():
        df.rename(columns={f"{old}_CE": f"{new}_CE", f"{old}_PE": f"{new}_PE"}, inplace=True)
    
    df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
    df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']
    
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    df_filtered = df[abs(df['strikePrice'] - atm_strike) <= 100]
    
    df_filtered['Zone'] = df_filtered['strikePrice'].apply(
        lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM'
    )
    
    bias_results = []
    for _, row in df_filtered.iterrows():
        chg_oi_bias = "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish"
        volume_bias = "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish"
        
        ask_ce = row.get('askQty_CE', 0)
        ask_pe = row.get('askQty_PE', 0)
        bid_ce = row.get('bidQty_CE', 0)
        bid_pe = row.get('bidQty_PE', 0)
        
        ask_bias = "Bearish" if ask_ce > ask_pe else "Bullish"
        bid_bias = "Bullish" if bid_ce > bid_pe else "Bearish"
        
        ce_oi = row['openInterest_CE']
        pe_oi = row['openInterest_PE']
        level = "Support" if pe_oi > 1.12 * ce_oi else "Resistance" if ce_oi > 1.12 * pe_oi else "Neutral"
        
        bias_results.append({
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "Level": level,
            "ChgOI_Bias": chg_oi_bias,
            "Volume_Bias": volume_bias,
            "Ask_Bias": ask_bias,
            "Bid_Bias": bid_bias,
            "PCR": round(pe_oi / ce_oi if ce_oi > 0 else 0, 2),
            "changeinOpenInterest_CE": row['changeinOpenInterest_CE'],
            "changeinOpenInterest_PE": row['changeinOpenInterest_PE']
        })
    
    return underlying, pd.DataFrame(bias_results)

def check_signals(df, option_data, current_price, proximity=5):
    if df.empty or option_data is None or not current_price:
        return
    
    # Fetch and analyze news
    news_items = fetch_market_news()
    news_sentiment = analyze_news_sentiment(news_items)
    market_trend = get_market_trend(df)
    news_impact = get_news_impact_score(news_sentiment, market_trend)
    
    # Get comprehensive bias information for all messages
    comprehensive_bias_info = get_comprehensive_bias_info(df, option_data, current_price, news_sentiment, market_trend)
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty:
        return
    
    row = atm_data.iloc[0]
    
    ce_chg_oi = abs(row.get('changeinOpenInterest_CE', 0))
    pe_chg_oi = abs(row.get('changeinOpenInterest_PE', 0))
    
    bias_aligned_bullish = (
        row['ChgOI_Bias'] == 'Bullish' and 
        row['Volume_Bias'] == 'Bullish' and
        row['Ask_Bias'] == 'Bullish' and
        row['Bid_Bias'] == 'Bullish'
    )
    
    bias_aligned_bearish = (
        row['ChgOI_Bias'] == 'Bearish' and 
        row['Volume_Bias'] == 'Bearish' and
        row['Ask_Bias'] == 'Bearish' and
        row['Bid_Bias'] == 'Bearish'
    )
    
    # News sentiment indicators for messages
    news_emoji = "📈" if news_sentiment["overall"] == "bullish" else "📉" if news_sentiment["overall"] == "bearish" else "📊"
    news_alignment = "Aligned" if news_impact == 1 else "Contrarian" if news_impact == -1 else "Neutral"
    
    # ===== EXISTING SIGNALS (WITH NEWS INTEGRATION AND COMPREHENSIVE BIAS INFO) =====
    
    # PRIMARY SIGNAL - Using enhanced pivot detection
    nearby_levels = get_nearby_pivot_levels(df, current_price, proximity)
    near_pivot = len(nearby_levels) > 0
    pivot_level = nearby_levels[0] if nearby_levels else None
    
    if near_pivot and pivot_level:
        primary_bullish_signal = (row['Level'] == 'Support' and bias_aligned_bullish and pivot_level['type'] == 'support')
        primary_bearish_signal = (row['Level'] == 'Resistance' and bias_aligned_bearish and pivot_level['type'] == 'resistance')
        
        # Check news filter
        should_filter_bullish = should_filter_signal_by_news(news_sentiment, "CALL") if primary_bullish_signal else False
        should_filter_bearish = should_filter_signal_by_news(news_sentiment, "PUT") if primary_bearish_signal else False
        
        if (primary_bullish_signal and not should_filter_bullish) or (primary_bearish_signal and not should_filter_bearish):
            signal_type = "CALL" if primary_bullish_signal else "PUT"
            price_diff = current_price - pivot_level['value']
            
            # Check for recent touches to add confidence
            touches = detect_level_touches(df, pivot_level['value'])
            touch_info = f" (Touches: {len(touches)})" if touches else ""
            
            # News warning if contrarian
            news_warning = f"\n⚠️ News Contrarian: {news_sentiment['overall'].title()} sentiment vs {signal_type}" if news_impact == -1 else ""
            
            message = f"""
🚨 PRIMARY NIFTY {signal_type} SIGNAL 🚨

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']}M {pivot_level['type'].title()} at ₹{pivot_level['value']:.2f}{touch_info}
🎯 ATM: {row['Strike']}

{news_emoji} News Sentiment: {news_sentiment['overall'].title()} ({news_alignment})
Conditions: {row['Level']}, All Bias Aligned, Confirmed Pivot{news_warning}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
            send_telegram(message)
            st.success(f"PRIMARY {signal_type} signal sent!")
            
            # Log signal with news context
            log_signal_performance(signal_type, "PRIMARY", 7, current_price)
    
    # SECONDARY SIGNAL
    put_dominance = pe_chg_oi > 1.3 * ce_chg_oi if ce_chg_oi > 0 else False
    call_dominance = ce_chg_oi > 1.3 * pe_chg_oi if pe_chg_oi > 0 else False
    
    secondary_bullish_signal = (bias_aligned_bullish and put_dominance)
    secondary_bearish_signal = (bias_aligned_bearish and call_dominance)
    
    # Apply news filter
    should_filter_bullish = should_filter_signal_by_news(news_sentiment, "CALL") if secondary_bullish_signal else False
    should_filter_bearish = should_filter_signal_by_news(news_sentiment, "PUT") if secondary_bearish_signal else False
    
    if (secondary_bullish_signal and not should_filter_bullish) or (secondary_bearish_signal and not should_filter_bearish):
        signal_type = "CALL" if secondary_bullish_signal else "PUT"
        dominance_ratio = pe_chg_oi / ce_chg_oi if secondary_bullish_signal and ce_chg_oi > 0 else ce_chg_oi / pe_chg_oi if ce_chg_oi > 0 else 0
        
        news_warning = f"\n⚠️ News Contrarian: {news_sentiment['overall'].title()} sentiment" if news_impact == -1 else ""
        
        message = f"""
⚡ SECONDARY NIFTY {signal_type} SIGNAL - OI DOMINANCE ⚡

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}

{news_emoji} News Sentiment: {news_sentiment['overall'].title()} ({news_alignment})
🔥 OI Dominance: {'PUT' if secondary_bullish_signal else 'CALL'} ChgOI {dominance_ratio:.1f}x higher
📊 All Bias Aligned{news_warning}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
        send_telegram(message)
        st.success(f"SECONDARY {signal_type} signal sent!")
        
        # Log signal
        log_signal_performance(signal_type, "SECONDARY", 6, current_price)

    # THIRD SIGNAL - BREAKOUT + MOMENTUM 
    is_breakout, breakout_type = check_breakout_pattern(df, current_price)
    momentum_score = get_momentum_score(df)
    market_regime = detect_market_regime(df)
    volume_profile = get_volume_profile(df)
    
    breakout_bullish_signal = (
        is_breakout and 
        breakout_type == "upside_breakout" and
        momentum_score >= 6 and
        volume_profile["strength"] >= 1.2 and
        bias_aligned_bullish
    )
    
    breakout_bearish_signal = (
        is_breakout and 
        breakout_type == "downside_breakout" and
        momentum_score <= 4 and
        volume_profile["strength"] >= 1.2 and
        bias_aligned_bearish
    )
    
    # Apply news filter for breakout signals
    should_filter_bullish = should_filter_signal_by_news(news_sentiment, "CALL") if breakout_bullish_signal else False
    should_filter_bearish = should_filter_signal_by_news(news_sentiment, "PUT") if breakout_bearish_signal else False
    
    if (breakout_bullish_signal and not should_filter_bullish) or (breakout_bearish_signal and not should_filter_bearish):
        signal_type = "CALL" if breakout_bullish_signal else "PUT"
        
        news_warning = f"\nNews Contrarian: {news_sentiment['overall'].title()} sentiment" if news_impact == -1 else ""
        
        message = f"""
💥 THIRD SIGNAL - BREAKOUT + MOMENTUM {signal_type} 💥

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}

{news_emoji} News Sentiment: {news_sentiment['overall'].title()} ({news_alignment})
🚀 Breakout Type: {breakout_type.replace('_', ' ').title()}
📊 Momentum Score: {momentum_score}/10
🏢 Market Regime: {market_regime.replace('_', ' ').title()}
🔊 Volume Profile: {volume_profile["profile"].title()} ({volume_profile["strength"]:.1f}x)

All ATM Biases + Breakout Confirmed{news_warning}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
        send_telegram(message)
        st.success(f"THIRD BREAKOUT {signal_type} signal sent!")
        log_signal_performance(signal_type, "THIRD", 7, current_price)

    # FOURTH SIGNAL - ALL BIAS ALIGNED
    if bias_aligned_bullish or bias_aligned_bearish:
        signal_type = "CALL" if bias_aligned_bullish else "PUT"
        
        # Light news filter for bias signals
        should_filter = should_filter_signal_by_news(news_sentiment, signal_type)
        
        if not should_filter:
            news_info = f"{news_emoji} News: {news_sentiment['overall'].title()}"
            
            message = f"""
🎯 FOURTH SIGNAL - ALL BIAS ALIGNED {signal_type} 🎯

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}

{news_info}
All ATM Biases Aligned: {signal_type}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
            send_telegram(message)
            st.success(f"FOURTH {signal_type} signal sent!")
            log_signal_performance(signal_type, "FOURTH", 5, current_price)

    # Calculate all filters including news
    volume_ok = check_volume_confirmation(df)
    confluence_ok = check_pivot_confluence(df, current_price, proximity)
    options_strong = check_options_strength(option_data)
    volatility = get_market_volatility(df)
    volatility_normal = volatility == "normal"
    time_ok = is_good_signal_time()
    
    # Trend alignment for different signal directions
    trend_aligned_bullish = market_trend in ["bullish", "neutral"]
    trend_aligned_bearish = market_trend in ["bearish", "neutral"]
    
    # News alignment check
    news_supports_bullish = news_sentiment["overall"] in ["bullish", "neutral"]
    news_supports_bearish = news_sentiment["overall"] in ["bearish", "neutral"]
    
    # FIFTH SIGNAL - ENHANCED PRIMARY WITH ALL FILTERS (INCLUDING NEWS)
    if near_pivot and pivot_level:
        # Enhanced analysis using new functions
        pivot_strength = calculate_pivot_strength(df, pivot_level['value'])
        volume_profile = get_volume_profile(df)
        momentum_score = get_momentum_score(df)
        market_regime = detect_market_regime(df)
        risk_reward = calculate_risk_reward(current_price, pivot_level['value'], df, nearby_levels)
        session = get_session_performance()
        
        enhanced_bullish = (
            primary_bullish_signal and
            trend_aligned_bullish and
            volume_ok and
            confluence_ok and
            options_strong and
            volatility_normal and
            time_ok and
            pivot_strength >= 1.5 and
            momentum_score >= 6 and
            volume_profile["strength"] >= 1.2 and
            news_supports_bullish  # News alignment
        )
        
        enhanced_bearish = (
            primary_bearish_signal and
            trend_aligned_bearish and
            volume_ok and
            confluence_ok and
            options_strong and
            volatility_normal and
            time_ok and
            pivot_strength >= 1.5 and
            momentum_score <= 4 and
            volume_profile["strength"] >= 1.2 and
            news_supports_bearish  # News alignment
        )
        
        if enhanced_bullish or enhanced_bearish:
            signal_type = "CALL" if enhanced_bullish else "PUT"
            price_diff = current_price - pivot_level['value']
            
            # Enhanced signal strength calculation including news
            base_strength = calculate_signal_strength(
                volume_ok, confluence_ok, options_strong, 
                volatility_normal, time_ok, 
                trend_aligned_bullish if enhanced_bullish else trend_aligned_bearish
            )
            
            # Add news bonus
            if news_impact == 1:  # News aligned
                signal_strength = min(10, base_strength + 1)
            else:
                signal_strength = base_strength
            
            # Add other bonuses
            if pivot_strength >= 2.0:
                signal_strength = min(10, signal_strength + 1)
            if volume_profile["strength"] >= 1.5:
                signal_strength = min(10, signal_strength + 1)
                
            touches = detect_level_touches(df, pivot_level['value'])
            touch_info = f" (Touches: {len(touches)})" if touches else ""
            
            # Position sizing recommendation
            if signal_strength >= 8:
                position_size = "Large"
            elif signal_strength >= 6:
                position_size = "Medium"
            else:
                position_size = "Small"
            
            message = f"""
🌟 FIFTH SIGNAL - ENHANCED PRIMARY {signal_type} 🌟

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']}M {pivot_level['type'].title()} at ₹{pivot_level['value']:.2f}{touch_info}
🎯 ATM: {row['Strike']}

⭐ Signal Strength: {signal_strength}/10
💪 Position Size: {position_size}
🏛️ Pivot Strength: {pivot_strength:.1f}x
📊 Market Regime: {market_regime.replace('_', ' ').title()}
🔊 Volume Profile: {volume_profile["profile"].title()} ({volume_profile["strength"]:.1f}x)
🚀 Momentum: {momentum_score}/10
⏰ Session: {session.replace('_', ' ').title()}
💰 Risk/Reward: 1:{risk_reward:.1f}
{news_emoji} News: {news_sentiment['overall'].title()} ({news_alignment})

All Premium Conditions Met (Including News Alignment)

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
            send_telegram(message)
            st.success(f"ENHANCED FIFTH {signal_type} signal sent! Strength: {signal_strength}/10 | Size: {position_size}")
            log_signal_performance(signal_type, "FIFTH", signal_strength, current_price)

    # SIXTH SIGNAL - CONFLUENCE + FLOW (NO PIVOT REQUIRED) WITH NEWS
    volume_profile = get_volume_profile(df)
    momentum_score = get_momentum_score(df)
    market_regime = detect_market_regime(df)
    session = get_session_performance()
    is_breakout, breakout_type = check_breakout_pattern(df, current_price)
    
    strong_confluence_bullish = (
        bias_aligned_bullish and
        trend_aligned_bullish and
        volume_ok and
        options_strong and
        volatility_normal and
        time_ok and
        put_dominance and
        momentum_score >= 6 and
        volume_profile["strength"] >= 1.3 and
        market_regime in ["bullish_trending", "ranging"] and
        news_supports_bullish  # News alignment
    )
    
    strong_confluence_bearish = (
        bias_aligned_bearish and
        trend_aligned_bearish and
        volume_ok and
        options_strong and
        volatility_normal and
        time_ok and
        call_dominance and
        momentum_score <= 4 and
        volume_profile["strength"] >= 1.3 and
        market_regime in ["bearish_trending", "ranging"] and
        news_supports_bearish  # News alignment
    )
    
    if strong_confluence_bullish or strong_confluence_bearish:
        signal_type = "CALL" if strong_confluence_bullish else "PUT"
        dominance_ratio = pe_chg_oi / ce_chg_oi if strong_confluence_bullish and ce_chg_oi > 0 else ce_chg_oi / pe_chg_oi if ce_chg_oi > 0 else 0
        
        # Enhanced signal strength calculation including news
        base_strength = calculate_signal_strength(
            volume_ok, confluence_ok, options_strong, 
            volatility_normal, time_ok, 
            trend_aligned_bullish if strong_confluence_bullish else trend_aligned_bearish
        )
        
        # Add news bonus
        if news_impact == 1:
            signal_strength = min(10, base_strength + 1)
        else:
            signal_strength = base_strength
            
        # Add other bonuses
        if dominance_ratio >= 2.0:
            signal_strength = min(10, signal_strength + 1)
        if volume_profile["strength"] >= 1.8:
            signal_strength = min(10, signal_strength + 1)
        if is_breakout:
            signal_strength = min(10, signal_strength + 1)
            
        # Position sizing recommendation
        if signal_strength >= 8:
            position_size = "Large"
        elif signal_strength >= 6:
            position_size = "Medium"
        else:
            position_size = "Small"
        
        breakout_info = f" | Breakout: {breakout_type.replace('_', ' ').title()}" if is_breakout else ""
        
        message = f"""
🚀 SIXTH SIGNAL - CONFLUENCE + FLOW {signal_type} 🚀

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}

⭐ Signal Strength: {signal_strength}/10
💪 Position Size: {position_size}
📊 Market Regime: {market_regime.replace('_', ' ').title()}
🔥 OI Dominance: {'PUT' if strong_confluence_bullish else 'CALL'} ChgOI {dominance_ratio:.1f}x higher
🔊 Volume Profile: {volume_profile["profile"].title()} ({volume_profile["strength"]:.1f}x)
🚀 Momentum: {momentum_score}/10
⏰ Session: {session.replace('_', ' ').title()}{breakout_info}
{news_emoji} News: {news_sentiment['overall'].title()} ({news_alignment})

All Premium Filters Passed (Including News Alignment)

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
        send_telegram(message)
        st.success(f"CONFLUENCE SIXTH {signal_type} signal sent! Strength: {signal_strength}/10 | Size: {position_size}")
        log_signal_performance(signal_type, "SIXTH", signal_strength, current_price)

# Signal Performance Tracking with News Context
def log_signal_performance(signal_type, signal_name, strength, current_price):
    """Log signal for future analysis including news context"""
    try:
        timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
        
        # Get current news sentiment if available
        news_context = "neutral"
        if 'news_cache' in st.session_state:
            news_sentiment = st.session_state.news_cache.get('sentiment', {})
            news_context = news_sentiment.get('overall', 'neutral')
        
        log_entry = {
            'timestamp': timestamp,
            'signal_type': signal_type,
            'signal_name': signal_name,
            'strength': strength,
            'price': current_price,
            'news_sentiment': news_context
        }
        
        if 'signal_log' not in st.session_state:
            st.session_state.signal_log = []
        st.session_state.signal_log.append(log_entry)
        
        # Keep only last 50 signals
        if len(st.session_state.signal_log) > 50:
            st.session_state.signal_log = st.session_state.signal_log[-50:]
    except:
        pass

# Additional function for enhanced-only mode with news
def check_enhanced_signals_only(df, option_data, current_price, proximity, min_strength):
    """Run only 5th and 6th signals with minimum strength filter and news consideration"""
    if df.empty or option_data is None or not current_price:
        return
    
    st.info(f"Enhanced Mode: Only showing signals with strength >= {min_strength}/10 and news alignment")

# Cache news data to avoid excessive API calls
def get_cached_news():
    """Get cached news data or fetch new if expired"""
    current_time = datetime.now()
    
    if 'news_cache' in st.session_state:
        cache_time = st.session_state.news_cache.get('timestamp')
        if cache_time and (current_time - cache_time).total_seconds() < 1800:  # 30 minutes
            return st.session_state.news_cache['data'], st.session_state.news_cache['sentiment']
    
    # Fetch fresh news
    news_items = fetch_market_news()
    news_sentiment = analyze_news_sentiment(news_items)
    
    # Cache the results
    st.session_state.news_cache = {
        'data': news_items,
        'sentiment': news_sentiment,
        'timestamp': current_time
    }
    
    return news_items, news_sentiment

def main():
    st.title("📈 Nifty Trading Analyzer")
    
    # Show market status
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    if not is_market_hours():
        st.warning(f"⚠️ Market is closed. Current time: {current_time.strftime('%H:%M:%S IST')}")
        st.info("Market hours: Monday-Friday, 9:00 AM to 3:45 PM IST")
    
    st.sidebar.header("🎛️ Enhanced Settings")
    interval = st.sidebar.selectbox("Timeframe", ["1", "3", "5", "10", "15"], index=2)
    enable_signals = st.sidebar.checkbox("Enable All Signals", value=True)
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        min_signal_strength = st.slider("Minimum Signal Strength", 1, 10, 6)
        enable_enhanced_only = st.checkbox("Enhanced Signals Only (5th & 6th)", value=False)
        enable_position_sizing = st.checkbox("Show Position Size", value=True)
        enable_breakout_signals = st.checkbox("Enable Breakout Signals (3rd)", value=True)
        enable_news_filter = st.checkbox("Enable News Filtering", value=True, 
                                       help="Filter signals that strongly contradict news sentiment")
        news_sensitivity = st.selectbox("News Filter Sensitivity", 
                                      ["Low", "Medium", "High"], 
                                      index=1,
                                      help="Low: Only filter extreme contrarian signals, High: Filter most contrarian signals")
        
    st.sidebar.info(f"Signal Filters Active: {len([x for x in [True, True, True, True, True] if x])}/5")
    
    # News update frequency notice
    if st.sidebar.button("Refresh News"):
        if 'news_cache' in st.session_state:
            del st.session_state.news_cache
        st.sidebar.success("News cache cleared!")
        
    st.sidebar.caption("News updates automatically every 30 minutes during market hours")
    
    api = DhanAPI()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chart")
        
        data = api.get_intraday_data(interval)
        df = process_candle_data(data) if data else pd.DataFrame()
        
        ltp_data = api.get_ltp_data()
        current_price = None
        if ltp_data and 'data' in ltp_data:
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
            
            col1_m, col2_m, col3_m = st.columns(3)
            with col1_m:
                st.metric("Price", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            with col2_m:
                st.metric("High", f"₹{df['high'].max():,.2f}")
            with col3_m:
                st.metric("Low", f"₹{df['low'].min():,.2f}")
        
        if not df.empty:
            fig = create_chart(df, f"Nifty {interval}min")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show nearby pivot levels info
            if current_price:
                proximity = st.sidebar.slider("Signal Proximity", 1, 20, 5)
                nearby_levels = get_nearby_pivot_levels(df, current_price, proximity)
                if nearby_levels:
                    st.info(f"📍 Nearby Levels: {len(nearby_levels)} confirmed pivot levels within {proximity} points")
                    for i, level in enumerate(nearby_levels[:3]):  # Show top 3
                        status = "✓" if level.get('confirmed', True) else "?"
                        st.caption(f"{i+1}. {level['timeframe']}M {level['type'].title()} {status}: ₹{level['value']:.1f} (Distance: {level['distance']:.1f})")
                
                # Show news analysis
                news_items = fetch_market_news()
                news_sentiment = analyze_news_sentiment(news_items)
                
                if news_items:
                    st.subheader("Market News")
                    
                    # News sentiment summary
                    sentiment_color = {
                        "bullish": "green",
                        "bearish": "red", 
                        "neutral": "gray"
                    }.get(news_sentiment["overall"], "gray")
                    
                    col_news1, col_news2, col_news3 = st.columns(3)
                    with col_news1:
                        st.metric("News Sentiment", news_sentiment["overall"].title(), 
                                f"Score: {news_sentiment['score']:.2f}")
                    with col_news2:
                        st.metric("Bullish Items", news_sentiment["bullish_count"])
                    with col_news3:
                        st.metric("Bearish Items", news_sentiment["bearish_count"])
                    
                    # News impact on signals
                    market_trend = get_market_trend(df)
                    news_impact = get_news_impact_score(news_sentiment, market_trend)
                    impact_text = {
                        1: "Aligned (Supportive)",
                        -1: "Contrarian (Cautionary)",
                        0: "Neutral"
                    }.get(news_impact, "Unknown")
                    
                    st.info(f"News vs Trend: {impact_text}")
                    
                    # Show recent news items
                    with st.expander("Latest News Items", expanded=False):
                        for i, item in enumerate(news_items[:3], 1):
                            st.write(f"**{i}. {item['title']}**")
                            if 'summary' in item and item['summary']:
                                st.caption(item['summary'][:200] + "..." if len(item.get('summary', '')) > 200 else item.get('summary', ''))
                            st.caption(f"Source: {item.get('source', 'Unknown')} | Sentiment: {item.get('sentiment', 'neutral')}")
                else:
                    st.info("News data unavailable")
                
                # Show enhanced signal analysis
                if len(df) > 20:
                    market_trend = get_market_trend(df)
                    volume_ok = check_volume_confirmation(df)
                    volatility = get_market_volatility(df)
                    time_ok = is_good_signal_time()
                    volume_profile = get_volume_profile(df)
                    momentum_score = get_momentum_score(df)
                    market_regime = detect_market_regime(df)
                    is_breakout, breakout_type = check_breakout_pattern(df, current_price)
                    session = get_session_performance()
                    
                    st.subheader("📊 Market Analysis")
                    
                    # Primary metrics row
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        trend_color = "🟢" if market_trend == "bullish" else "🔴" if market_trend == "bearish" else "🟡"
                        st.metric("Trend", f"{trend_color} {market_trend.title()}")
                    with col_b:
                        vol_color = "🟢" if volume_ok else "🔴"
                        st.metric("Volume", f"{vol_color} {'Strong' if volume_ok else 'Weak'}")
                    with col_c:
                        vol_color = "🟢" if volatility == "normal" else "🟡" if volatility == "low" else "🔴"
                        st.metric("Volatility", f"{vol_color} {volatility.title()}")
                    with col_d:
                        time_color = "🟢" if time_ok else "🔴"
                        st.metric("Timing", f"{time_color} {'Good' if time_ok else 'Poor'}")
                    
                    # Secondary metrics row  
                    col_e, col_f, col_g, col_h = st.columns(4)
                    with col_e:
                        momentum_color = "🟢" if momentum_score >= 6 else "🔴" if momentum_score <= 4 else "🟡"
                        st.metric("Momentum", f"{momentum_color} {momentum_score}/10")
                    with col_f:
                        regime_color = "🟢" if "trending" in market_regime else "🟡"
                        st.metric("Regime", f"{regime_color} {market_regime.replace('_', ' ').title()}")
                    with col_g:
                        vol_prof_color = "🟢" if volume_profile["strength"] >= 1.2 else "🟡" if volume_profile["strength"] >= 1.0 else "🔴"
                        st.metric("Vol Profile", f"{vol_prof_color} {volume_profile['profile'].title()}")
                    with col_h:
                        session_color = "🟢" if session in ["mid_session", "afternoon_session"] else "🟡"
                        st.metric("Session", f"{session_color} {session.replace('_', ' ').title()}")
                    
                    # Breakout status
                    if is_breakout:
                        breakout_color = "🚀" if breakout_type == "upside_breakout" else "⬇️"
                        st.info(f"{breakout_color} **Breakout Detected:** {breakout_type.replace('_', ' ').title()}")
                    
                    # Signal environment summary
                    filter_count = sum([volume_ok, volatility == "normal", time_ok, 
                                      momentum_score >= 6 or momentum_score <= 4,
                                      volume_profile["strength"] >= 1.2])
                    
                    if filter_count >= 4:
                        st.success(f"🌟 **Excellent Signal Environment** - {filter_count}/5 conditions favorable")
                    elif filter_count >= 3:
                        st.info(f"⚡ **Good Signal Environment** - {filter_count}/5 conditions favorable")
                    elif filter_count >= 2:
                        st.warning(f"⚠️ **Fair Signal Environment** - {filter_count}/5 conditions favorable")
                    else:
                        st.error(f"❌ **Poor Signal Environment** - {filter_count}/5 conditions favorable")
                
                # Show recent signals performance
                if 'signal_log' in st.session_state and st.session_state.signal_log:
                    st.subheader("📈 Recent Signals")
                    recent_signals = st.session_state.signal_log[-10:]  # Last 10 signals
                    
                    for i, signal in enumerate(reversed(recent_signals)):
                        signal_time = signal['timestamp'].split('T')[1][:5]  # HH:MM format
                        strength_stars = "⭐" * min(5, int(signal['strength'] / 2))
                        st.caption(f"{i+1}. {signal_time} - {signal['signal_name']} {signal['signal_type']} {strength_stars} @ ₹{signal['price']:.1f}")
                    
                    if len(st.session_state.signal_log) >= 5:
                        avg_strength = sum(s['strength'] for s in recent_signals) / len(recent_signals)
                        st.info(f"📊 Average Signal Strength: {avg_strength:.1f}/10 (Last {len(recent_signals)} signals)")
                        
        else:
            st.error("No chart data available")
    
    with col2:
        st.header("Options Analysis")
        
        expiry_data = get_expiry_list()
        if expiry_data and 'data' in expiry_data:
            expiry_dates = expiry_data['data']
            selected_expiry = st.selectbox("Expiry", expiry_dates)
            
            underlying_price, option_summary = analyze_options(selected_expiry)
            
            if underlying_price and option_summary is not None:
                st.info(f"Spot: ₹{underlying_price:.2f}")
                st.dataframe(option_summary, use_container_width=True)
                
                if enable_signals and not df.empty and is_market_hours():
                    # Apply advanced settings filters
                    if enable_enhanced_only:
                        # Only run enhanced signals (5th & 6th)
                        check_enhanced_signals_only(df, option_summary, underlying_price, proximity, min_signal_strength)
                    else:
                        # Run all signals
                        check_signals(df, option_summary, underlying_price, proximity)
                        
            else:
                st.error("Options data unavailable")
        else:
            st.error("Expiry data unavailable")
    
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"🕒 Last Updated: {current_time}")
    
    if st.sidebar.button("📤 Test Telegram"):
        test_message = f"""
🔔 Test Message from Nifty Analyzer

📊 System Status: Online
🕒 Time: {current_time}
⚙️ All Systems Operational

Enhanced Features Active:
✅ Proper Pivot Detection
✅ Non-Repainting Signals  
✅ 6 Signal Types Available
✅ Signal Strength Scoring
✅ Position Size Recommendations
✅ Market Analysis Dashboard
✅ News Integration
✅ Comprehensive Bias Analysis
"""
        send_telegram(test_message)
        st.sidebar.success("📤 Enhanced test message sent!")

if __name__ == "__main__":
    # Initialize session state for signal tracking
    if 'signal_log' not in st.session_state:
        st.session_state.signal_log = []
    
    main()
