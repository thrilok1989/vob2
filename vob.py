import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import yfinance as yf
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
import math
from scipy.stats import norm
import plotly.express as px
from collections import deque

warnings.filterwarnings('ignore')

# Indian Standard Time (IST)
IST = pytz.timezone('Asia/Kolkata')

# Import Dhan API for Indian indices volume data
try:
    from dhan_data_fetcher import DhanDataFetcher
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False
    print("Warning: Dhan API not available. Volume data may be missing for Indian indices.")

# =============================================
# TELEGRAM NOTIFICATION SYSTEM
# =============================================

class TelegramNotifier:
    """Telegram notification system for bias alerts"""
    
    def __init__(self):
        # Get credentials from Streamlit secrets
        self.bot_token = st.secrets.get("TELEGRAM", {}).get("BOT_TOKEN", "")
        self.chat_id = st.secrets.get("TELEGRAM", {}).get("CHAT_ID", "")
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes cooldown between same type alerts
        
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(self.bot_token and self.chat_id)
        
    def send_message(self, message: str, alert_type: str = "INFO") -> bool:
        """Send message to Telegram"""
        try:
            if not self.is_configured():
                print("Telegram credentials not configured in secrets")
                return False
            
            # Check cooldown
            current_time = time.time()
            if alert_type in self.last_alert_time:
                if current_time - self.last_alert_time[alert_type] < self.alert_cooldown:
                    print(f"Alert {alert_type} in cooldown")
                    return False
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                self.last_alert_time[alert_type] = current_time
                print(f"✅ Telegram alert sent: {alert_type}")
                return True
            else:
                print(f"❌ Telegram send failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False
    
    def send_bias_alert(self, technical_bias: str, options_bias: str, atm_bias: str, overall_bias: str, score: float):
        """Send comprehensive bias alert"""
        # Check if all three components are aligned
        components = [technical_bias, options_bias, atm_bias]
        bullish_count = sum(1 for bias in components if "BULL" in bias.upper())
        bearish_count = sum(1 for bias in components if "BEAR" in bias.upper())
        
        if bullish_count >= 2 or bearish_count >= 2:
            emoji = "🚀" if bullish_count >= 2 else "🔻"
            alert_type = "STRONG_BULL" if bullish_count >= 2 else "STRONG_BEAR"
            
            message = f"""
{emoji} <b>STRONG BIAS ALERT - NIFTY 50</b> {emoji}

📊 <b>Component Analysis:</b>
• Technical Analysis: <b>{technical_bias}</b>
• Options Chain: <b>{options_bias}</b>  
• ATM Detailed: <b>{atm_bias}</b>

🎯 <b>Overall Bias:</b> <code>{overall_bias}</code>
⭐ <b>Confidence Score:</b> <code>{score:.1f}/100</code>

⏰ <b>Time:</b> {datetime.now(IST).strftime('%H:%M:%S')}
            
💡 <b>Market Insight:</b>
{'Bullish momentum detected across multiple timeframes' if bullish_count >= 2 else 'Bearish pressure building across indicators'}
"""
            
            return self.send_message(message, alert_type)
        
        return False


# Initialize Telegram Notifier
telegram_notifier = TelegramNotifier()


class BiasAnalysisPro:
    """
    Comprehensive Bias Analysis matching Pine Script indicator EXACTLY
    Analyzes 13 bias indicators:
    - Fast (8): Volume Delta, HVP, VOB, Order Blocks, RSI, DMI, VIDYA, MFI
    - Medium (2): Close vs VWAP, Price vs VWAP
    - Slow (3): Weighted stocks (Daily, TF1, TF2)
    """

    def __init__(self):
        """Initialize bias analysis with default configuration"""
        self.config = self._default_config()
        self.all_bias_results = []
        self.overall_bias = "NEUTRAL"
        self.overall_score = 0

    def _default_config(self) -> Dict:
        """Default configuration from Pine Script"""
        return {
            # Timeframes
            'tf1': '15m',
            'tf2': '1h',

            # Indicator periods
            'rsi_period': 14,
            'mfi_period': 10,
            'dmi_period': 13,
            'dmi_smoothing': 8,
            'atr_period': 14,

            # Volume
            'volume_roc_length': 14,
            'volume_threshold': 1.2,

            # Volatility
            'volatility_ratio_length': 14,
            'volatility_threshold': 1.5,

            # OBV
            'obv_smoothing': 21,

            # Force Index
            'force_index_length': 13,
            'force_index_smoothing': 2,

            # Price ROC
            'price_roc_length': 12,

            # Market Breadth
            'breadth_threshold': 60,

            # Divergence
            'divergence_lookback': 30,
            'rsi_overbought': 70,
            'rsi_oversold': 30,

            # Choppiness Index
            'ci_length': 14,
            'ci_high_threshold': 61.8,
            'ci_low_threshold': 38.2,

            # Bias parameters
            'bias_strength': 60,
            'divergence_threshold': 60,

            # Adaptive weights
            'normal_fast_weight': 2.0,
            'normal_medium_weight': 3.0,
            'normal_slow_weight': 5.0,
            'reversal_fast_weight': 5.0,
            'reversal_medium_weight': 3.0,
            'reversal_slow_weight': 2.0,

            # Stocks with weights
            'stocks': {
                '^NSEBANK': 10.0,  # BANKNIFTY Index
                'RELIANCE.NS': 9.98,
                'HDFCBANK.NS': 9.67,
                'BHARTIARTL.NS': 9.97,
                'TCS.NS': 8.54,
                'ICICIBANK.NS': 8.01,
                'INFY.NS': 8.55,
                'HINDUNILVR.NS': 1.98,
                'ITC.NS': 2.44,
                'MARUTI.NS': 0.0
            }
        }

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Dhan API (for Indian indices) or Yahoo Finance (for others)
        Note: Yahoo Finance limits intraday data - use 7d max for 5m interval
        """
        # Check if this is an Indian index that needs Dhan API
        indian_indices = {'^NSEI': 'NIFTY', '^BSESN': 'SENSEX', '^NSEBANK': 'BANKNIFTY'}

        if symbol in indian_indices and DHAN_AVAILABLE:
            try:
                # Use Dhan API for Indian indices to get proper volume data
                dhan_instrument = indian_indices[symbol]
                fetcher = DhanDataFetcher()

                # Convert interval to Dhan API format (1, 5, 15, 25, 60)
                interval_map = {'1m': '1', '5m': '5', '15m': '15', '1h': '60'}
                dhan_interval = interval_map.get(interval, '5')

                # Calculate date range for historical data (7 days) - Use IST timezone
                now_ist = datetime.now(IST)
                to_date = now_ist.strftime('%Y-%m-%d %H:%M:%S')
                from_date = (now_ist - timedelta(days=7)).replace(hour=9, minute=15, second=0).strftime('%Y-%m-%d %H:%M:%S')

                # Fetch intraday data with 7 days historical range
                result = fetcher.fetch_intraday_data(dhan_instrument, interval=dhan_interval, from_date=from_date, to_date=to_date)

                if result.get('success') and result.get('data') is not None:
                    df = result['data']

                    # Ensure column names match yfinance format (capitalized)
                    df.columns = [col.capitalize() for col in df.columns]

                    # Set timestamp as index
                    if 'Timestamp' in df.columns:
                        df.set_index('Timestamp', inplace=True)

                    # Ensure volume column exists and has valid data
                    if 'Volume' not in df.columns:
                        df['Volume'] = 0
                    else:
                        # Replace NaN volumes with 0
                        df['Volume'] = df['Volume'].fillna(0)

                    if not df.empty:
                        print(f"✅ Fetched {len(df)} candles for {symbol} from Dhan API with volume data (from {from_date} to {to_date})")
                        return df
                    else:
                        print(f"⚠️  Warning: Empty data from Dhan API for {symbol}, falling back to yfinance")
                else:
                    print(f"Warning: Dhan API failed for {symbol}: {result.get('error')}, falling back to yfinance")
            except Exception as e:
                print(f"Error fetching from Dhan API for {symbol}: {e}, falling back to yfinance")

        # Fallback to Yahoo Finance for non-Indian indices or if Dhan fails
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"Warning: No data for {symbol}")
                return pd.DataFrame()

            # Ensure volume column exists (even if it's zeros for indices)
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            else:
                # Replace NaN volumes with 0
                df['Volume'] = df['Volume'].fillna(0)

            # Warn if volume is all zeros (common for Yahoo Finance indices)
            if df['Volume'].sum() == 0 and symbol in indian_indices:
                print(f"⚠️  Warning: Volume data is zero for {symbol} from Yahoo Finance")

            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # TECHNICAL INDICATORS
    # =========================================================================

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Money Flow Index with NaN/zero handling"""
        # Check if volume data is available
        if df['Volume'].sum() == 0:
            # Return neutral MFI (50) if no volume data
            return pd.Series([50.0] * len(df), index=df.index)

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        # Avoid division by zero
        mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfi_ratio))

        # Fill NaN with neutral value (50)
        mfi = mfi.fillna(50)

        return mfi

    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8):
        """Calculate DMI indicators"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=smoothing).mean()

        return plus_di, minus_di, adx

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP with NaN/zero handling"""
        # Check if volume data is available
        if df['Volume'].sum() == 0:
            # Return typical price as fallback if no volume data
            return (df['High'] + df['Low'] + df['Close']) / 3

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()

        # Avoid division by zero
        cumulative_volume_safe = cumulative_volume.replace(0, np.nan)
        vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume_safe

        # Fill NaN with typical price
        vwap = vwap.fillna(typical_price)

        return vwap

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate_vidya(self, df: pd.DataFrame, length: int = 10, momentum: int = 20, band_distance: float = 2.0):
        """Calculate VIDYA (Variable Index Dynamic Average) matching Pine Script"""
        close = df['Close']

        # Calculate momentum (CMO - Chande Momentum Oscillator)
        m = close.diff()
        p = m.where(m >= 0, 0.0).rolling(window=momentum).sum()
        n = (-m.where(m < 0, 0.0)).rolling(window=momentum).sum()

        # Avoid division by zero
        cmo_denom = p + n
        cmo_denom = cmo_denom.replace(0, np.nan)
        abs_cmo = abs(100 * (p - n) / cmo_denom).fillna(0)

        # Calculate VIDYA
        alpha = 2 / (length + 1)
        vidya = pd.Series(index=close.index, dtype=float)
        vidya.iloc[0] = close.iloc[0]

        for i in range(1, len(close)):
            vidya.iloc[i] = (alpha * abs_cmo.iloc[i] / 100 * close.iloc[i] +
                            (1 - alpha * abs_cmo.iloc[i] / 100) * vidya.iloc[i-1])

        # Smooth VIDYA
        vidya_smoothed = vidya.rolling(window=15).mean()

        # Calculate bands
        atr = self.calculate_atr(df, 200)
        upper_band = vidya_smoothed + atr * band_distance
        lower_band = vidya_smoothed - atr * band_distance

        # Determine trend based on band crossovers
        is_trend_up = close > upper_band
        is_trend_down = close < lower_band

        # Get current state
        vidya_bullish = is_trend_up.iloc[-1] if len(is_trend_up) > 0 else False
        vidya_bearish = is_trend_down.iloc[-1] if len(is_trend_down) > 0 else False

        return vidya_smoothed, vidya_bullish, vidya_bearish

    def calculate_volume_delta(self, df: pd.DataFrame):
        """Calculate Volume Delta (up_vol - down_vol) matching Pine Script"""
        if df['Volume'].sum() == 0:
            return 0, False, False

        # Calculate up and down volume
        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()

        volume_delta = up_vol - down_vol
        volume_bullish = volume_delta > 0
        volume_bearish = volume_delta < 0

        return volume_delta, volume_bullish, volume_bearish

    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, vol_filter: float = 2.0):
        """Calculate High Volume Pivots matching Pine Script
        Returns: (hvp_bullish, hvp_bearish, pivot_high_count, pivot_low_count)
        """
        if df['Volume'].sum() == 0:
            return False, False, 0, 0

        # Calculate pivot highs and lows
        pivot_highs = []
        pivot_lows = []

        for i in range(left_bars, len(df) - right_bars):
            # Check for pivot high
            is_pivot_high = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['High'].iloc[j] >= df['High'].iloc[i]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs.append(i)

            # Check for pivot low
            is_pivot_low = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['Low'].iloc[j] <= df['Low'].iloc[i]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows.append(i)

        # Calculate volume sum and reference
        volume_sum = df['Volume'].rolling(window=left_bars * 2).sum()
        ref_vol = volume_sum.quantile(0.95)
        norm_vol = (volume_sum / ref_vol * 5).fillna(0)

        # Check recent HVP signals
        hvp_bullish = False
        hvp_bearish = False

        if len(pivot_lows) > 0:
            last_pivot_low_idx = pivot_lows[-1]
            if norm_vol.iloc[last_pivot_low_idx] > vol_filter:
                hvp_bullish = True

        if len(pivot_highs) > 0:
            last_pivot_high_idx = pivot_highs[-1]
            if norm_vol.iloc[last_pivot_high_idx] > vol_filter:
                hvp_bearish = True

        return hvp_bullish, hvp_bearish, len(pivot_highs), len(pivot_lows)

    def calculate_vob(self, df: pd.DataFrame, length1: int = 5):
        """Calculate Volume Order Blocks matching Pine Script
        Returns: (vob_bullish, vob_bearish, ema1_value, ema2_value)
        """
        # Calculate EMAs
        length2 = length1 + 13
        ema1 = self.calculate_ema(df['Close'], length1)
        ema2 = self.calculate_ema(df['Close'], length2)

        # Detect crossovers
        cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
        cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])

        # In real implementation, we would check if price touched OB zones
        # For simplicity, using crossover signals
        vob_bullish = cross_up
        vob_bearish = cross_dn

        return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]

    # =========================================================================
    # COMPREHENSIVE BIAS ANALYSIS
    # =========================================================================

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict:
        """
        Analyze all 8 bias indicators:
        Fast (8): Volume Delta, HVP, VOB, Order Blocks, RSI, DMI, VIDYA, MFI
        """

        print(f"Fetching data for {symbol}...")
        # Use 7d period with 5m interval (Yahoo Finance limitation for intraday data)
        df = self.fetch_data(symbol, period='7d', interval='5m')

        if df.empty or len(df) < 100:
            error_msg = f'Insufficient data (fetched {len(df)} candles, need at least 100)'
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }

        current_price = df['Close'].iloc[-1]

        # Initialize bias results list
        bias_results = []
        stock_data = []  # Empty since we removed Weighted Stocks indicators

        # =====================================================================
        # FAST INDICATORS (8 total)
        # =====================================================================

        # 1. VOLUME DELTA
        volume_delta, volume_bullish, volume_bearish = self.calculate_volume_delta(df)

        if volume_bullish:
            vol_delta_bias = "BULLISH"
            vol_delta_score = 100
        elif volume_bearish:
            vol_delta_bias = "BEARISH"
            vol_delta_score = -100
        else:
            vol_delta_bias = "NEUTRAL"
            vol_delta_score = 0

        bias_results.append({
            'indicator': 'Volume Delta',
            'value': f"{volume_delta:.0f}",
            'bias': vol_delta_bias,
            'score': vol_delta_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 2. HVP (High Volume Pivots)
        hvp_bullish, hvp_bearish, pivot_highs, pivot_lows = self.calculate_hvp(df)

        if hvp_bullish:
            hvp_bias = "BULLISH"
            hvp_score = 100
            hvp_value = f"Bull Signal (Lows: {pivot_lows}, Highs: {pivot_highs})"
        elif hvp_bearish:
            hvp_bias = "BEARISH"
            hvp_score = -100
            hvp_value = f"Bear Signal (Highs: {pivot_highs}, Lows: {pivot_lows})"
        else:
            hvp_bias = "NEUTRAL"
            hvp_score = 0
            hvp_value = f"No Signal (Highs: {pivot_highs}, Lows: {pivot_lows})"

        bias_results.append({
            'indicator': 'HVP (High Volume Pivots)',
            'value': hvp_value,
            'bias': hvp_bias,
            'score': hvp_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 3. VOB (Volume Order Blocks)
        vob_bullish, vob_bearish, vob_ema5, vob_ema18 = self.calculate_vob(df)

        if vob_bullish:
            vob_bias = "BULLISH"
            vob_score = 100
            vob_value = f"Bull Cross (EMA5: {vob_ema5:.2f} > EMA18: {vob_ema18:.2f})"
        elif vob_bearish:
            vob_bias = "BEARISH"
            vob_score = -100
            vob_value = f"Bear Cross (EMA5: {vob_ema5:.2f} < EMA18: {vob_ema18:.2f})"
        else:
            vob_bias = "NEUTRAL"
            vob_score = 0
            # Determine if EMA5 is above or below EMA18
            if vob_ema5 > vob_ema18:
                vob_value = f"EMA5: {vob_ema5:.2f} > EMA18: {vob_ema18:.2f} (No Cross)"
            else:
                vob_value = f"EMA5: {vob_ema5:.2f} < EMA18: {vob_ema18:.2f} (No Cross)"

        bias_results.append({
            'indicator': 'VOB (Volume Order Blocks)',
            'value': vob_value,
            'bias': vob_bias,
            'score': vob_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 4. ORDER BLOCKS (EMA Crossover)
        ema5 = self.calculate_ema(df['Close'], 5)
        ema18 = self.calculate_ema(df['Close'], 18)

        # Detect crossovers
        cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
        cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])

        if cross_up:
            ob_bias = "BULLISH"
            ob_score = 100
        elif cross_dn:
            ob_bias = "BEARISH"
            ob_score = -100
        else:
            ob_bias = "NEUTRAL"
            ob_score = 0

        bias_results.append({
            'indicator': 'Order Blocks (EMA 5/18)',
            'value': f"EMA5: {ema5.iloc[-1]:.2f} | EMA18: {ema18.iloc[-1]:.2f}",
            'bias': ob_bias,
            'score': ob_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 5. RSI
        rsi = self.calculate_rsi(df['Close'], self.config['rsi_period'])
        rsi_value = rsi.iloc[-1]

        if rsi_value > 50:
            rsi_bias = "BULLISH"
            rsi_score = 100
        else:
            rsi_bias = "BEARISH"
            rsi_score = -100

        bias_results.append({
            'indicator': 'RSI',
            'value': f"{rsi_value:.2f}",
            'bias': rsi_bias,
            'score': rsi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 6. DMI
        plus_di, minus_di, adx = self.calculate_dmi(df, self.config['dmi_period'], self.config['dmi_smoothing'])
        plus_di_value = plus_di.iloc[-1]
        minus_di_value = minus_di.iloc[-1]
        adx_value = adx.iloc[-1]

        if plus_di_value > minus_di_value:
            dmi_bias = "BULLISH"
            dmi_score = 100
        else:
            dmi_bias = "BEARISH"
            dmi_score = -100

        bias_results.append({
            'indicator': 'DMI',
            'value': f"+DI:{plus_di_value:.1f} -DI:{minus_di_value:.1f}",
            'bias': dmi_bias,
            'score': dmi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 7. VIDYA
        vidya_val, vidya_bullish, vidya_bearish = self.calculate_vidya(df)

        if vidya_bullish:
            vidya_bias = "BULLISH"
            vidya_score = 100
        elif vidya_bearish:
            vidya_bias = "BEARISH"
            vidya_score = -100
        else:
            vidya_bias = "NEUTRAL"
            vidya_score = 0

        bias_results.append({
            'indicator': 'VIDYA',
            'value': f"{vidya_val.iloc[-1]:.2f}" if not vidya_val.empty else "N/A",
            'bias': vidya_bias,
            'score': vidya_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 8. MFI
        mfi = self.calculate_mfi(df, self.config['mfi_period'])
        mfi_value = mfi.iloc[-1]

        if np.isnan(mfi_value):
            mfi_value = 50.0  # Neutral default

        if mfi_value > 50:
            mfi_bias = "BULLISH"
            mfi_score = 100
        else:
            mfi_bias = "BEARISH"
            mfi_score = -100

        bias_results.append({
            'indicator': 'MFI (Money Flow)',
            'value': f"{mfi_value:.2f}",
            'bias': mfi_bias,
            'score': mfi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # =====================================================================
        # CALCULATE OVERALL BIAS (MATCHING PINE SCRIPT LOGIC) - FIXED
        # =====================================================================
        fast_bull = 0
        fast_bear = 0
        fast_total = 0

        medium_bull = 0
        medium_bear = 0
        medium_total = 0

        # FIX 1: Disable slow category completely
        slow_bull = 0
        slow_bear = 0
        slow_total = 0  # Set to zero to avoid division by zero

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for bias in bias_results:
            if 'BULLISH' in bias['bias']:
                bullish_count += 1
                if bias['category'] == 'fast':
                    fast_bull += 1
                elif bias['category'] == 'medium':
                    medium_bull += 1
                # Skip slow category
            elif 'BEARISH' in bias['bias']:
                bearish_count += 1
                if bias['category'] == 'fast':
                    fast_bear += 1
                elif bias['category'] == 'medium':
                    medium_bear += 1
                # Skip slow category
            else:
                neutral_count += 1

            if bias['category'] == 'fast':
                fast_total += 1
            elif bias['category'] == 'medium':
                medium_total += 1
            # Skip slow category counting

        # Calculate percentages - FIXED for slow category
        fast_bull_pct = (fast_bull / fast_total) * 100 if fast_total > 0 else 0
        fast_bear_pct = (fast_bear / fast_total) * 100 if fast_total > 0 else 0

        medium_bull_pct = (medium_bull / medium_total) * 100 if medium_total > 0 else 0
        medium_bear_pct = (medium_bear / medium_total) * 100 if medium_total > 0 else 0

        # FIX 1: Set slow percentages to 0 since we disabled slow indicators
        slow_bull_pct = 0
        slow_bear_pct = 0

        # Adaptive weighting (matching Pine Script)
        # Check for divergence - FIXED for slow category
        divergence_threshold = self.config['divergence_threshold']
        # Since slow_bull_pct is 0, divergence won't trigger incorrectly
        bullish_divergence = slow_bull_pct >= 66 and fast_bear_pct >= divergence_threshold
        bearish_divergence = slow_bear_pct >= 66 and fast_bull_pct >= divergence_threshold
        divergence_detected = bullish_divergence or bearish_divergence

        # Determine mode - FIXED: Always use normal mode since slow indicators disabled
        if divergence_detected and slow_total > 0:  # Only if we had slow indicators
            fast_weight = self.config['reversal_fast_weight']
            medium_weight = self.config['reversal_medium_weight']
            slow_weight = self.config['reversal_slow_weight']
            mode = "REVERSAL"
        else:
            # Use normal weights, ignore slow weight
            fast_weight = self.config['normal_fast_weight']
            medium_weight = self.config['normal_medium_weight']
            slow_weight = 0  # FIX: Set slow weight to 0 since no slow indicators
            mode = "NORMAL"

        # Calculate weighted scores - FIXED: Exclude slow category
        bullish_signals = (fast_bull * fast_weight) + (medium_bull * medium_weight) + (slow_bull * slow_weight)
        bearish_signals = (fast_bear * fast_weight) + (medium_bear * medium_weight) + (slow_bear * slow_weight)
        total_signals = (fast_total * fast_weight) + (medium_total * medium_weight) + (slow_total * slow_weight)

        bullish_bias_pct = (bullish_signals / total_signals) * 100 if total_signals > 0 else 0
        bearish_bias_pct = (bearish_signals / total_signals) * 100 if total_signals > 0 else 0

        # Determine overall bias
        bias_strength = self.config['bias_strength']

        if bullish_bias_pct >= bias_strength:
            overall_bias = "BULLISH"
            overall_score = bullish_bias_pct
            overall_confidence = min(100, bullish_bias_pct)
        elif bearish_bias_pct >= bias_strength:
            overall_bias = "BEARISH"
            overall_score = -bearish_bias_pct
            overall_confidence = min(100, bearish_bias_pct)
        else:
            overall_bias = "NEUTRAL"
            overall_score = 0
            overall_confidence = 100 - max(bullish_bias_pct, bearish_bias_pct)

        return {
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now(IST),
            'bias_results': bias_results,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'overall_confidence': overall_confidence,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_indicators': len(bias_results),
            'stock_data': stock_data,
            'mode': mode,
            'fast_bull_pct': fast_bull_pct,
            'fast_bear_pct': fast_bear_pct,
            'slow_bull_pct': slow_bull_pct,
            'slow_bear_pct': slow_bear_pct,
            'bullish_bias_pct': bullish_bias_pct,
            'bearish_bias_pct': bearish_bias_pct
        }


# =============================================
# VOLUME ORDER BLOCKS (FROM SECOND APP)
# =============================================

class VolumeOrderBlocks:
    """Python implementation of Volume Order Blocks indicator by BigBeluga"""
    
    def __init__(self, sensitivity=5):
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.max_lines_count = 500
        self.bullish_blocks = deque(maxlen=15)
        self.bearish_blocks = deque(maxlen=15)
        self.sent_alerts = set()
        
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, df: pd.DataFrame, period=200) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr * 3
    
    def detect_volume_order_blocks(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect Volume Order Blocks based on the Pine Script logic"""
        if len(df) < self.length2:
            return [], []
        
        ema1 = self.calculate_ema(df['Close'], self.length1)
        ema2 = self.calculate_ema(df['Close'], self.length2)
        
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        
        atr = self.calculate_atr(df)
        atr1 = atr * 2 / 3
        
        bullish_blocks = []
        bearish_blocks = []
        
        for i in range(len(df)):
            if cross_up.iloc[i]:
                lookback_data = df.iloc[max(0, i - self.length2):i+1]
                if len(lookback_data) == 0:
                    continue
                    
                lowest_idx = lookback_data['Low'].idxmin()
                lowest_price = lookback_data.loc[lowest_idx, 'Low']
                
                vol = lookback_data['Volume'].sum()
                
                open_price = lookback_data.loc[lowest_idx, 'Open']
                close_price = lookback_data.loc[lowest_idx, 'Close']
                src = min(open_price, close_price)
                
                if pd.notna(atr.iloc[i]) and (src - lowest_price) < atr1.iloc[i] * 0.5:
                    src = lowest_price + atr1.iloc[i] * 0.5
                
                mid = (src + lowest_price) / 2
                
                bullish_blocks.append({
                    'index': lowest_idx,
                    'upper': src,
                    'lower': lowest_price,
                    'mid': mid,
                    'volume': vol,
                    'type': 'bullish'
                })
                
            elif cross_down.iloc[i]:
                lookback_data = df.iloc[max(0, i - self.length2):i+1]
                if len(lookback_data) == 0:
                    continue
                    
                highest_idx = lookback_data['High'].idxmax()
                highest_price = lookback_data.loc[highest_idx, 'High']
                
                vol = lookback_data['Volume'].sum()
                
                open_price = lookback_data.loc[highest_idx, 'Open']
                close_price = lookback_data.loc[highest_idx, 'Close']
                src = max(open_price, close_price)
                
                if pd.notna(atr.iloc[i]) and (highest_price - src) < atr1.iloc[i] * 0.5:
                    src = highest_price - atr1.iloc[i] * 0.5
                
                mid = (src + highest_price) / 2
                
                bearish_blocks.append({
                    'index': highest_idx,
                    'upper': highest_price,
                    'lower': src,
                    'mid': mid,
                    'volume': vol,
                    'type': 'bearish'
                })
        
        bullish_blocks = self.filter_overlapping_blocks(bullish_blocks, atr.iloc[-1] if len(atr) > 0 else 0)
        bearish_blocks = self.filter_overlapping_blocks(bearish_blocks, atr.iloc[-1] if len(atr) > 0 else 0)
        
        return bullish_blocks, bearish_blocks
    
    def filter_overlapping_blocks(self, blocks: List[Dict[str, Any]], atr_value: float) -> List[Dict[str, Any]]:
        if not blocks:
            return []
        
        filtered_blocks = []
        for block in blocks:
            overlap = False
            for existing_block in filtered_blocks:
                if abs(block['mid'] - existing_block['mid']) < atr_value:
                    overlap = True
                    break
            if not overlap:
                filtered_blocks.append(block)
        
        return filtered_blocks
    
    def check_price_near_blocks(self, current_price: float, blocks: List[Dict[str, Any]], threshold: float = 5) -> List[Dict[str, Any]]:
        nearby_blocks = []
        for block in blocks:
            distance_to_upper = abs(current_price - block['upper'])
            distance_to_lower = abs(current_price - block['lower'])
            distance_to_mid = abs(current_price - block['mid'])
            
            if (distance_to_upper <= threshold or 
                distance_to_lower <= threshold or 
                distance_to_mid <= threshold):
                nearby_blocks.append(block)
        
        return nearby_blocks

# FIX 5: Add plotting function for VOB
def plot_vob(df: pd.DataFrame, bullish_blocks: List[Dict], bearish_blocks: List[Dict]) -> go.Figure:
    """Plot Volume Order Blocks on candlestick chart"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3],
                       subplot_titles=('Price with Volume Order Blocks', 'Volume'))
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'),
                 row=1, col=1)
    
    # Add bullish blocks
    for block in bullish_blocks:
        fig.add_hline(y=block['upper'], line_dash="dash", line_color="green", 
                     annotation_text=f"Bull Block", row=1, col=1)
        fig.add_hline(y=block['lower'], line_dash="dash", line_color="green", row=1, col=1)
        # Fill between lines
        fig.add_shape(type="rect", x0=block['index'], x1=df.index[-1],
                     y0=block['lower'], y1=block['upper'],
                     fillcolor="green", opacity=0.1, line_width=0, row=1, col=1)
    
    # Add bearish blocks
    for block in bearish_blocks:
        fig.add_hline(y=block['upper'], line_dash="dash", line_color="red",
                     annotation_text=f"Bear Block", row=1, col=1)
        fig.add_hline(y=block['lower'], line_dash="dash", line_color="red", row=1, col=1)
        # Fill between lines
        fig.add_shape(type="rect", x0=block['index'], x1=df.index[-1],
                     y0=block['lower'], y1=block['upper'],
                     fillcolor="red", opacity=0.1, line_width=0, row=1, col=1)
    
    # Volume
    colors = ['green' if close >= open_ else 'red' 
              for close, open_ in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
                 row=2, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, 
                     template='plotly_dark',
                     height=600,
                     showlegend=True)
    
    return fig


# =============================================
# GAMMA SEQUENCE ANALYZER
# =============================================

class GammaSequenceAnalyzer:
    """Comprehensive Gamma Sequence Analysis for Institutional Bias Detection"""
    
    def __init__(self):
        self.gamma_levels = {
            'EXTREME_POSITIVE': {'threshold': 10000, 'bias': 'STRONG_BULLISH', 'score': 100},
            'HIGH_POSITIVE': {'threshold': 5000, 'bias': 'BULLISH', 'score': 75},
            'MODERATE_POSITIVE': {'threshold': 1000, 'bias': 'MILD_BULLISH', 'score': 50},
            'NEUTRAL': {'threshold': -1000, 'bias': 'NEUTRAL', 'score': 0},
            'MODERATE_NEGATIVE': {'threshold': -5000, 'bias': 'MILD_BEARISH', 'score': -50},
            'HIGH_NEGATIVE': {'threshold': -10000, 'bias': 'BEARISH', 'score': -75},
            'EXTREME_NEGATIVE': {'threshold': -20000, 'bias': 'STRONG_BEARISH', 'score': -100}
        }
    
    def calculate_gamma_exposure(self, df_chain: pd.DataFrame) -> pd.DataFrame:
        """Calculate Gamma exposure for all strikes"""
        df = df_chain.copy()
        
        # Calculate Gamma exposure
        df['gamma_exposure_ce'] = df['Gamma_CE'] * df['openInterest_CE'] * 100  # Multiply by 100 for contract size
        df['gamma_exposure_pe'] = df['Gamma_PE'] * df['openInterest_PE'] * 100
        df['net_gamma_exposure'] = df['gamma_exposure_ce'] + df['gamma_exposure_pe']
        
        # Calculate Gamma profile
        df['gamma_profile'] = df['net_gamma_exposure'].apply(self._get_gamma_profile)
        
        return df
    
    def _get_gamma_profile(self, gamma_exposure: float) -> str:
        """Get Gamma profile based on exposure level"""
        for level, config in self.gamma_levels.items():
            if gamma_exposure >= config['threshold']:
                return level
        return 'EXTREME_NEGATIVE'
    
    def analyze_gamma_sequence_bias(self, df_chain: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """Comprehensive Gamma sequence bias analysis"""
        try:
            df_with_gamma = self.calculate_gamma_exposure(df_chain)
            
            # Analyze different strike zones
            analysis = {
                'total_gamma_exposure': df_with_gamma['net_gamma_exposure'].sum(),
                'gamma_bias': self._calculate_overall_gamma_bias(df_with_gamma),
                'zones': self._analyze_gamma_zones(df_with_gamma, spot_price),
                'sequence': self._analyze_gamma_sequence(df_with_gamma),
                'walls': self._find_gamma_walls(df_with_gamma),
                'profile': self._get_gamma_profile(df_with_gamma['net_gamma_exposure'].sum())
            }
            
            # Calculate comprehensive Gamma score
            analysis['gamma_score'] = self._calculate_gamma_score(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Error in Gamma sequence analysis: {e}")
            return {'gamma_bias': 'NEUTRAL', 'gamma_score': 0, 'error': str(e)}
    
    def _calculate_overall_gamma_bias(self, df: pd.DataFrame) -> str:
        """Calculate overall Gamma bias"""
        total_gamma = df['net_gamma_exposure'].sum()
        
        if total_gamma > 10000:
            return "STRONG_BULLISH"
        elif total_gamma > 5000:
            return "BULLISH"
        elif total_gamma > 1000:
            return "MILD_BULLISH"
        elif total_gamma < -10000:
            return "STRONG_BEARISH"
        elif total_gamma < -5000:
            return "BEARISH"
        elif total_gamma < -1000:
            return "MILD_BEARISH"
        else:
            return "NEUTRAL"
    
    def _analyze_gamma_zones(self, df: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """Analyze Gamma across different price zones"""
        strike_diff = df['strikePrice'].iloc[1] - df['strikePrice'].iloc[0]
        
        zones = {
            'itm_puts': df[df['strikePrice'] < spot_price - strike_diff].copy(),
            'near_otm_puts': df[(df['strikePrice'] >= spot_price - strike_diff) & (df['strikePrice'] < spot_price)].copy(),
            'atm': df[abs(df['strikePrice'] - spot_price) <= strike_diff].copy(),
            'near_otm_calls': df[(df['strikePrice'] > spot_price) & (df['strikePrice'] <= spot_price + strike_diff)].copy(),
            'otm_calls': df[df['strikePrice'] > spot_price + strike_diff].copy()
        }
        
        zone_analysis = {}
        for zone_name, zone_data in zones.items():
            if not zone_data.empty:
                zone_analysis[zone_name] = {
                    'gamma_exposure': zone_data['net_gamma_exposure'].sum(),
                    'bias': self._calculate_overall_gamma_bias(zone_data),
                    'strike_range': f"{zone_data['strikePrice'].min():.0f}-{zone_data['strikePrice'].max():.0f}"
                }
        
        return zone_analysis
    
    def _analyze_gamma_sequence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Gamma sequence patterns"""
        # Sort by strike price
        df_sorted = df.sort_values('strikePrice')
        
        # Calculate Gamma changes between strikes
        df_sorted['gamma_change'] = df_sorted['net_gamma_exposure'].diff()
        
        # Identify sequences
        positive_sequences = []
        negative_sequences = []
        current_sequence = []
        
        for _, row in df_sorted.iterrows():
            if not current_sequence:
                current_sequence.append(row)
                continue
                
            current_gamma = row['net_gamma_exposure']
            prev_gamma = current_sequence[-1]['net_gamma_exposure']
            
            if (current_gamma >= 0 and prev_gamma >= 0) or (current_gamma < 0 and prev_gamma < 0):
                current_sequence.append(row)
            else:
                if current_sequence:
                    seq_gamma = sum([x['net_gamma_exposure'] for x in current_sequence])
                    if seq_gamma >= 0:
                        positive_sequences.append({
                            'strikes': [x['strikePrice'] for x in current_sequence],
                            'total_gamma': seq_gamma,
                            'length': len(current_sequence)
                        })
                    else:
                        negative_sequences.append({
                            'strikes': [x['strikePrice'] for x in current_sequence],
                            'total_gamma': seq_gamma,
                            'length': len(current_sequence)
                        })
                current_sequence = [row]
        
        return {
            'positive_sequences': positive_sequences,
            'negative_sequences': negative_sequences,
            'longest_positive_seq': max([seq['length'] for seq in positive_sequences]) if positive_sequences else 0,
            'longest_negative_seq': max([seq['length'] for seq in negative_sequences]) if negative_sequences else 0
        }
    
    def _find_gamma_walls(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find significant Gamma walls/resistance levels"""
        # Find local maxima/minima in Gamma exposure
        gamma_walls = []
        
        for i in range(1, len(df) - 1):
            current_gamma = df.iloc[i]['net_gamma_exposure']
            prev_gamma = df.iloc[i-1]['net_gamma_exposure']
            next_gamma = df.iloc[i+1]['net_gamma_exposure']
            
            # Gamma wall (local maximum with high positive Gamma)
            if current_gamma > prev_gamma and current_gamma > next_gamma and current_gamma > 5000:
                gamma_walls.append({
                    'strike': df.iloc[i]['strikePrice'],
                    'gamma_exposure': current_gamma,
                    'type': 'RESISTANCE',
                    'strength': 'STRONG' if current_gamma > 10000 else 'MODERATE'
                })
            
            # Gamma vacuum (local minimum with high negative Gamma)
            elif current_gamma < prev_gamma and current_gamma < next_gamma and current_gamma < -5000:
                gamma_walls.append({
                    'strike': df.iloc[i]['strikePrice'],
                    'gamma_exposure': current_gamma,
                    'type': 'SUPPORT',
                    'strength': 'STRONG' if current_gamma < -10000 else 'MODERATE'
                })
        
        return sorted(gamma_walls, key=lambda x: abs(x['gamma_exposure']), reverse=True)[:5]  # Top 5
    
    def _calculate_gamma_score(self, analysis: Dict) -> float:
        """Calculate comprehensive Gamma score from -100 to 100"""
        base_score = self.gamma_levels.get(analysis['profile'], {}).get('score', 0)
        
        # Adjust score based on sequence analysis
        seq_analysis = analysis.get('sequence', {})
        pos_seqs = len(seq_analysis.get('positive_sequences', []))
        neg_seqs = len(seq_analysis.get('negative_sequences', []))
        
        if pos_seqs > neg_seqs:
            sequence_bonus = 10
        elif neg_seqs > pos_seqs:
            sequence_bonus = -10
        else:
            sequence_bonus = 0
        
        # Adjust score based on Gamma walls
        walls = analysis.get('walls', [])
        resistance_walls = len([w for w in walls if w['type'] == 'RESISTANCE'])
        support_walls = len([w for w in walls if w['type'] == 'SUPPORT'])
        
        if resistance_walls > support_walls:
            walls_penalty = -5
        elif support_walls > resistance_walls:
            walls_penalty = 5
        else:
            walls_penalty = 0
        
        final_score = base_score + sequence_bonus + walls_penalty
        return max(-100, min(100, final_score))


# FIX 3: Add caching for Gamma analysis
@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
def cached_gamma_analysis(_analyzer, df_chain: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
    """Cached Gamma analysis to improve performance"""
    return _analyzer.analyze_gamma_sequence_bias(df_chain, spot_price)


# =============================================
# INSTITUTIONAL OI ADVANCED ANALYZER
# =============================================

class InstitutionalOIAdvanced:
    """Advanced Institutional OI Analysis with Gamma Sequencing"""
    
    def __init__(self):
        self.master_table_rules = {
            'CALL': {
                'Winding_Up_Price_Up': {'bias': 'BEARISH', 'institution_move': 'Selling/Writing', 'confidence': 'HIGH'},
                'Winding_Up_Price_Down': {'bias': 'BEARISH', 'institution_move': 'Sellers Dominating', 'confidence': 'HIGH'},
                'Unwinding_Down_Price_Up': {'bias': 'BULLISH', 'institution_move': 'Short Covering', 'confidence': 'MEDIUM'},
                'Unwinding_Down_Price_Down': {'bias': 'MILD_BEARISH', 'institution_move': 'Longs Exiting', 'confidence': 'LOW'}
            },
            'PUT': {
                'Winding_Up_Price_Down': {'bias': 'BULLISH', 'institution_move': 'Selling/Writing', 'confidence': 'HIGH'},
                'Winding_Up_Price_Up': {'bias': 'BULLISH', 'institution_move': 'Sellers Dominating', 'confidence': 'HIGH'},
                'Unwinding_Down_Price_Down': {'bias': 'BEARISH', 'institution_move': 'Short Covering', 'confidence': 'MEDIUM'},
                'Unwinding_Down_Price_Up': {'bias': 'MILD_BULLISH', 'institution_move': 'Longs Exiting', 'confidence': 'LOW'}
            }
        }
        self.gamma_analyzer = GammaSequenceAnalyzer()
    
    def analyze_institutional_oi_pattern(self, option_type: str, oi_change: float, price_change: float, 
                                       volume: float, iv_change: float, bid_ask_ratio: float) -> Dict:
        """Analyze institutional OI patterns based on master table rules"""
        
        # Determine OI action
        oi_action = "Winding_Up" if oi_change > 0 else "Unwinding_Down"
        
        # Determine price action
        price_action = "Price_Up" if price_change > 0 else "Price_Down"
        
        # Determine pattern key
        pattern_key = f"{oi_action}_{price_action}"
        
        # Get base pattern
        base_pattern = self.master_table_rules.get(option_type, {}).get(pattern_key, {})
        
        if not base_pattern:
            return {'bias': 'NEUTRAL', 'confidence': 'LOW', 'pattern': 'Unknown'}
        
        # Enhance with volume and IV analysis
        volume_signal = "High" if volume > 1000 else "Low"
        iv_signal = "Rising" if iv_change > 0 else "Falling"
        
        # Bid/Ask analysis
        liquidity_signal = "Bid_Heavy" if bid_ask_ratio > 1.2 else "Ask_Heavy" if bid_ask_ratio < 0.8 else "Balanced"
        
        # Adjust confidence based on volume and IV
        confidence = base_pattern['confidence']
        if volume_signal == "High" and abs(iv_change) > 1.0:
            confidence = "VERY_HIGH"
        
        return {
            'option_type': option_type,
            'bias': base_pattern['bias'],
            'institution_move': base_pattern['institution_move'],
            'confidence': confidence,
            'pattern': pattern_key,
            'volume_signal': volume_signal,
            'iv_signal': iv_signal,
            'liquidity_signal': liquidity_signal,
            'oi_change': oi_change,
            'price_change': price_change,
            'volume': volume
        }
    
    def analyze_atm_institutional_footprint(self, df_chain: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """Comprehensive institutional footprint analysis for ATM ±2 strikes"""
        try:
            # FIX 4: Normalize column names first
            df_chain = normalize_chain_columns(df_chain)
            
            # Get ATM ±2 strikes
            strike_diff = df_chain['strikePrice'].iloc[1] - df_chain['strikePrice'].iloc[0] if len(df_chain) > 1 else 50
            atm_range = strike_diff * 2
            atm_strikes = df_chain[abs(df_chain['strikePrice'] - spot_price) <= atm_range].copy()
            
            if atm_strikes.empty:
                return {'overall_bias': 'NEUTRAL', 'score': 0, 'patterns': []}
            
            patterns = []
            total_score = 0
            pattern_count = 0
            
            for _, strike_data in atm_strikes.iterrows():
                strike = strike_data['strikePrice']
                
                # Analyze CALL side
                if pd.notna(strike_data.get('change_oi_ce')):
                    ce_pattern = self.analyze_institutional_oi_pattern(
                        option_type='CALL',
                        oi_change=strike_data.get('change_oi_ce', 0),
                        price_change=strike_data.get('ltp_ce', 0) - strike_data.get('previousClose_CE', strike_data.get('ltp_ce', 0)),
                        volume=strike_data.get('volume_ce', 0),
                        iv_change=strike_data.get('iv_ce', 0) - strike_data.get('previousIV_CE', strike_data.get('iv_ce', 0)),
                        bid_ask_ratio=strike_data.get('bid_ce', 1) / max(1, strike_data.get('ask_ce', 1))
                    )
                    ce_pattern['strike'] = strike
                    patterns.append(ce_pattern)
                    
                    # Convert bias to score
                    bias_score = self._bias_to_score(ce_pattern['bias'], ce_pattern['confidence'])
                    total_score += bias_score
                    pattern_count += 1
                
                # Analyze PUT side
                if pd.notna(strike_data.get('change_oi_pe')):
                    pe_pattern = self.analyze_institutional_oi_pattern(
                        option_type='PUT',
                        oi_change=strike_data.get('change_oi_pe', 0),
                        price_change=strike_data.get('ltp_pe', 0) - strike_data.get('previousClose_PE', strike_data.get('ltp_pe', 0)),
                        volume=strike_data.get('volume_pe', 0),
                        iv_change=strike_data.get('iv_pe', 0) - strike_data.get('previousIV_PE', strike_data.get('iv_pe', 0)),
                        bid_ask_ratio=strike_data.get('bid_pe', 1) / max(1, strike_data.get('ask_pe', 1))
                    )
                    pe_pattern['strike'] = strike
                    patterns.append(pe_pattern)
                    
                    # Convert bias to score
                    bias_score = self._bias_to_score(pe_pattern['bias'], pe_pattern['confidence'])
                    total_score += bias_score
                    pattern_count += 1
            
            # Calculate overall bias
            if pattern_count > 0:
                avg_score = total_score / pattern_count
                if avg_score > 0.2:
                    overall_bias = "BULLISH"
                elif avg_score < -0.2:
                    overall_bias = "BEARISH"
                else:
                    overall_bias = "NEUTRAL"
            else:
                overall_bias = "NEUTRAL"
                avg_score = 0
            
            # Add Gamma sequencing analysis with caching
            gamma_analysis = cached_gamma_analysis(self.gamma_analyzer, df_chain, spot_price)
            
            return {
                'overall_bias': overall_bias,
                'score': avg_score * 100,  # Convert to percentage
                'patterns': patterns,
                'gamma_analysis': gamma_analysis,
                'strikes_analyzed': len(atm_strikes),
                'total_patterns': len(patterns)
            }
            
        except Exception as e:
            print(f"Error in institutional footprint analysis: {e}")
            return {'overall_bias': 'NEUTRAL', 'score': 0, 'patterns': [], 'gamma_analysis': {}}
    
    def _bias_to_score(self, bias: str, confidence: str) -> float:
        """Convert bias and confidence to numerical score"""
        bias_scores = {
            'BULLISH': 1.0,
            'MILD_BULLISH': 0.5,
            'NEUTRAL': 0.0,
            'MILD_BEARISH': -0.5,
            'BEARISH': -1.0
        }
        
        confidence_multipliers = {
            'VERY_HIGH': 1.5,
            'HIGH': 1.2,
            'MEDIUM': 1.0,
            'LOW': 0.7
        }
        
        base_score = bias_scores.get(bias, 0.0)
        multiplier = confidence_multipliers.get(confidence, 1.0)
        
        return base_score * multiplier


# FIX 4: Add column normalization function
def normalize_chain_columns(df_chain: pd.DataFrame) -> pd.DataFrame:
    """Normalize option chain column names to handle different API formats"""
    df = df_chain.copy()
    
    # Define column mapping for different API formats
    column_mapping = {
        # Standardize to our expected column names
        'changeinOpenInterest_CE': 'change_oi_ce',
        'changeinOpenInterest_PE': 'change_oi_pe',
        'openInterest_CE': 'oi_ce', 
        'openInterest_PE': 'oi_pe',
        'impliedVolatility_CE': 'iv_ce',
        'impliedVolatility_PE': 'iv_pe',
        'lastPrice_CE': 'ltp_ce',
        'lastPrice_PE': 'ltp_pe',
        'totalTradedVolume_CE': 'volume_ce',
        'totalTradedVolume_PE': 'volume_pe',
        'bidQty_CE': 'bid_ce',
        'askQty_CE': 'ask_ce',
        'bidQty_PE': 'bid_pe', 
        'askQty_PE': 'ask_pe',
        
        # Alternative column names (common in different APIs)
        'CE_changeinOpenInterest': 'change_oi_ce',
        'PE_changeinOpenInterest': 'change_oi_pe',
        'CE_openInterest': 'oi_ce',
        'PE_openInterest': 'oi_pe',
        'CE_impliedVolatility': 'iv_ce',
        'PE_impliedVolatility': 'iv_pe',
        'CE_lastPrice': 'ltp_ce',
        'PE_lastPrice': 'ltp_pe',
        'CE_totalTradedVolume': 'volume_ce',
        'PE_totalTradedVolume': 'volume_pe',
        'CE_bidQty': 'bid_ce',
        'CE_askQty': 'ask_ce',
        'PE_bidQty': 'bid_pe',
        'PE_askQty': 'ask_pe',
        
        # Very short column names
        'chg_oi_ce': 'change_oi_ce',
        'chg_oi_pe': 'change_oi_pe',
        'oi_ce': 'oi_ce',
        'oi_pe': 'oi_pe',
        'iv_ce': 'iv_ce', 
        'iv_pe': 'iv_pe',
        'ltp_ce': 'ltp_ce',
        'ltp_pe': 'ltp_pe',
        'vol_ce': 'volume_ce',
        'vol_pe': 'volume_pe'
    }
    
    # Rename columns that exist in the dataframe
    existing_columns = set(df.columns)
    rename_dict = {}
    
    for old_col, new_col in column_mapping.items():
        if old_col in existing_columns:
            rename_dict[old_col] = new_col
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Create missing columns with default values
    required_columns = ['change_oi_ce', 'change_oi_pe', 'oi_ce', 'oi_pe', 'iv_ce', 'iv_pe', 
                       'ltp_ce', 'ltp_pe', 'volume_ce', 'volume_pe', 'bid_ce', 'ask_ce', 'bid_pe', 'ask_pe']
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Default to 0 for missing columns
    
    # Calculate Greeks if not present (simplified calculation)
    if 'Gamma_CE' not in df.columns:
        df['Gamma_CE'] = 0.01  # Simplified gamma value
        df['Gamma_PE'] = 0.01
    
    return df


# =============================================
# BREAKOUT & REVERSAL CONFIRMATION ANALYZER
# =============================================

class BreakoutReversalAnalyzer:
    """Institutional Breakout & Reversal Confirmation System"""
    
    def __init__(self):
        self.breakout_threshold = 0.6  # 60% confidence for real breakout
        self.reversal_threshold = 0.7   # 70% confidence for reversal
        
    def analyze_breakout_confirmation(self, df_chain: pd.DataFrame, spot_price: float, 
                                    price_change: float, volume_change: float) -> Dict[str, Any]:
        """
        Comprehensive breakout confirmation analysis
        Returns confidence score 0-100 for breakout validity
        """
        try:
            # FIX 4: Normalize column names first
            df_chain = normalize_chain_columns(df_chain)
            
            # Get ATM ±2 strikes for analysis
            strike_diff = df_chain['strikePrice'].iloc[1] - df_chain['strikePrice'].iloc[0] if len(df_chain) > 1 else 50
            atm_strikes = df_chain[abs(df_chain['strikePrice'] - spot_price) <= strike_diff * 2].copy()
            
            if atm_strikes.empty:
                return {'breakout_confidence': 0, 'direction': 'UNKNOWN', 'signals': []}
            
            # Determine breakout direction
            is_upside_breakout = price_change > 0
            direction = "UP" if is_upside_breakout else "DOWN"
            
            signals = []
            total_score = 0
            max_score = 0
            
            # 1. OI Change Analysis (25 points)
            oi_analysis = self._analyze_oi_pattern(atm_strikes, is_upside_breakout)
            signals.extend(oi_analysis['signals'])
            total_score += oi_analysis['score']
            max_score += 25
            
            # 2. Price vs OI Conflict (20 points)
            conflict_analysis = self._analyze_price_oi_conflict(atm_strikes, is_upside_breakout)
            signals.extend(conflict_analysis['signals'])
            total_score += conflict_analysis['score']
            max_score += 20
            
            # 3. IV Behavior Analysis (15 points)
            iv_analysis = self._analyze_iv_behavior(atm_strikes, is_upside_breakout)
            signals.extend(iv_analysis['signals'])
            total_score += iv_analysis['score']
            max_score += 15
            
            # 4. PCR Trend Analysis (15 points)
            pcr_analysis = self._analyze_pcr_trend(df_chain, is_upside_breakout)
            signals.extend(pcr_analysis['signals'])
            total_score += pcr_analysis['score']
            max_score += 15
            
            # 5. Max Pain Movement (10 points)
            max_pain_analysis = self._analyze_max_pain_movement(df_chain, spot_price, is_upside_breakout)
            signals.extend(max_pain_analysis['signals'])
            total_score += max_pain_analysis['score']
            max_score += 10
            
            # 6. Strike OI Wall Breakdown (15 points)
            wall_analysis = self._analyze_oi_wall_breakdown(atm_strikes, is_upside_breakout)
            signals.extend(wall_analysis['signals'])
            total_score += wall_analysis['score']
            max_score += 15
            
            # Calculate final confidence
            breakout_confidence = (total_score / max_score) * 100 if max_score > 0 else 0
            
            # Determine breakout type
            if breakout_confidence >= 60:
                breakout_type = "REAL_BREAKOUT"
            elif breakout_confidence >= 30:
                breakout_type = "WEAK_BREAKOUT"
            else:
                breakout_type = "FAKE_BREAKOUT"
            
            return {
                'breakout_confidence': breakout_confidence,
                'direction': direction,
                'breakout_type': breakout_type,
                'signals': signals,
                'total_score': total_score,
                'max_score': max_score
            }
            
        except Exception as e:
            print(f"Error in breakout analysis: {e}")
            return {'breakout_confidence': 0, 'direction': 'UNKNOWN', 'signals': []}
    
    def analyze_reversal_confirmation(self, df_chain: pd.DataFrame, spot_price: float,
                                   price_action: Dict) -> Dict[str, Any]:
        """
        Comprehensive reversal confirmation analysis
        Returns confidence score 0-100 for reversal validity
        """
        try:
            # FIX 4: Normalize column names first
            df_chain = normalize_chain_columns(df_chain)
            
            strike_diff = df_chain['strikePrice'].iloc[1] - df_chain['strikePrice'].iloc[0] if len(df_chain) > 1 else 50
            atm_strikes = df_chain[abs(df_chain['strikePrice'] - spot_price) <= strike_diff * 2].copy()
            
            if atm_strikes.empty:
                return {'reversal_confidence': 0, 'direction': 'UNKNOWN', 'signals': []}
            
            # Determine if we're looking for top or bottom reversal
            is_top_reversal = price_action.get('has_upper_wick', False) or price_action.get('is_overbought', False)
            direction = "TOP_REVERSAL" if is_top_reversal else "BOTTOM_REVERSAL"
            
            signals = []
            total_score = 0
            max_score = 0
            
            # 1. OI Divergence Analysis (30 points)
            divergence_analysis = self._analyze_oi_divergence(atm_strikes, is_top_reversal)
            signals.extend(divergence_analysis['signals'])
            total_score += divergence_analysis['score']
            max_score += 30
            
            # 2. IV Crash Detection (20 points)
            iv_crash_analysis = self._analyze_iv_crash(atm_strikes)
            signals.extend(iv_crash_analysis['signals'])
            total_score += iv_crash_analysis['score']
            max_score += 20
            
            # 3. Writer Defense Analysis (25 points)
            defense_analysis = self._analyze_writer_defense(df_chain, spot_price, is_top_reversal)
            signals.extend(defense_analysis['signals'])
            total_score += defense_analysis['score']
            max_score += 25
            
            # 4. Opposite OI Build (15 points)
            opposite_oi_analysis = self._analyze_opposite_oi_build(atm_strikes, is_top_reversal)
            signals.extend(opposite_oi_analysis['signals'])
            total_score += opposite_oi_analysis['score']
            max_score += 15
            
            # 5. PCR Extremes (10 points)
            pcr_extreme_analysis = self._analyze_pcr_extremes(df_chain)
            signals.extend(pcr_extreme_analysis['signals'])
            total_score += pcr_extreme_analysis['score']
            max_score += 10
            
            # Calculate final confidence
            reversal_confidence = (total_score / max_score) * 100 if max_score > 0 else 0
            
            # Determine reversal strength
            if reversal_confidence >= 70:
                reversal_type = "STRONG_REVERSAL"
            elif reversal_confidence >= 50:
                reversal_type = "MODERATE_REVERSAL"
            else:
                reversal_type = "WEAK_REVERSAL"
            
            return {
                'reversal_confidence': reversal_confidence,
                'direction': direction,
                'reversal_type': reversal_type,
                'signals': signals,
                'total_score': total_score,
                'max_score': max_score
            }
            
        except Exception as e:
            print(f"Error in reversal analysis: {e}")
            return {'reversal_confidence': 0, 'direction': 'UNKNOWN', 'signals': []}
    
    def _analyze_oi_pattern(self, atm_strikes: pd.DataFrame, is_upside: bool) -> Dict:
        """Analyze OI patterns for breakout confirmation"""
        signals = []
        score = 0
        
        # Calculate total OI changes
        total_ce_oi_change = atm_strikes['change_oi_ce'].sum()
        total_pe_oi_change = atm_strikes['change_oi_pe'].sum()
        
        if is_upside:
            # Upside breakout: CE OI should decrease, PE OI should increase
            if total_ce_oi_change < 0:
                signals.append("✅ CE OI decreasing (call sellers running)")
                score += 10
            else:
                signals.append("❌ CE OI increasing (call writers active)")
            
            if total_pe_oi_change > 0:
                signals.append("✅ PE OI increasing (put writers entering)")
                score += 15
            else:
                signals.append("❌ PE OI decreasing (no put writing)")
        else:
            # Downside breakout: CE OI should increase, PE OI should decrease
            if total_ce_oi_change > 0:
                signals.append("✅ CE OI increasing (call writers attacking)")
                score += 10
            else:
                signals.append("❌ CE OI decreasing (no call writing)")
            
            if total_pe_oi_change < 0:
                signals.append("✅ PE OI decreasing (put sellers exiting)")
                score += 15
            else:
                signals.append("❌ PE OI increasing (put writers defending)")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_price_oi_conflict(self, atm_strikes: pd.DataFrame, is_upside: bool) -> Dict:
        """Analyze price vs OI conflict"""
        signals = []
        score = 0
        
        # Get dominant strike
        dominant_strike = atm_strikes.loc[atm_strikes['oi_ce'].idxmax()] if is_upside else atm_strikes.loc[atm_strikes['oi_pe'].idxmax()]
        
        if is_upside:
            # Clean upside: Price ↑, CE OI ↓
            if dominant_strike['change_oi_ce'] < 0:
                signals.append("✅ Clean breakout: Price ↑ + CE OI ↓ (short covering)")
                score += 20
            else:
                signals.append("❌ Fake breakout: Price ↑ + CE OI ↑ (sellers building wall)")
        else:
            # Clean downside: Price ↓, PE OI ↓
            if dominant_strike['change_oi_pe'] < 0:
                signals.append("✅ Clean breakdown: Price ↓ + PE OI ↓ (put covering)")
                score += 20
            else:
                signals.append("❌ Fake breakdown: Price ↓ + PE OI ↑ (put writers defending)")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_iv_behavior(self, atm_strikes: pd.DataFrame, is_upside: bool) -> Dict:
        """Analyze IV behavior for breakout confirmation"""
        signals = []
        score = 0
        
        avg_ce_iv = atm_strikes['iv_ce'].mean()
        avg_pe_iv = atm_strikes['iv_pe'].mean()
        
        if is_upside:
            # Upside: CE IV should rise slightly, PE IV stable/fall
            if avg_ce_iv > avg_pe_iv:
                signals.append("✅ CE IV > PE IV (upside momentum)")
                score += 10
            else:
                signals.append("❌ CE IV <= PE IV (weak upside)")
            
            if avg_pe_iv < 20:  # Low PE IV indicates no put buying pressure
                signals.append("✅ Low PE IV (no put hedging)")
                score += 5
        else:
            # Downside: PE IV should rise, CE IV stable/fall
            if avg_pe_iv > avg_ce_iv:
                signals.append("✅ PE IV > CE IV (downside momentum)")
                score += 10
            else:
                signals.append("❌ PE IV <= CE IV (weak downside)")
            
            if avg_ce_iv < 20:  # Low CE IV indicates no call buying pressure
                signals.append("✅ Low CE IV (no call hedging)")
                score += 5
        
        return {'signals': signals, 'score': score}
    
    def _analyze_pcr_trend(self, df_chain: pd.DataFrame, is_upside: bool) -> Dict:
        """Analyze PCR trend for breakout direction"""
        signals = []
        score = 0
        
        total_ce_oi = df_chain['oi_ce'].sum()
        total_pe_oi = df_chain['oi_pe'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        if is_upside:
            if pcr > 0.8:
                signals.append(f"✅ PCR {pcr:.2f} > 0.8 (bullish bias)")
                score += 10
            elif pcr > 0.6:
                signals.append(f"⚠️ PCR {pcr:.2f} neutral")
                score += 5
            else:
                signals.append(f"❌ PCR {pcr:.2f} < 0.6 (bearish bias)")
        else:
            if pcr < 0.7:
                signals.append(f"✅ PCR {pcr:.2f} < 0.7 (bearish bias)")
                score += 10
            elif pcr < 0.9:
                signals.append(f"⚠️ PCR {pcr:.2f} neutral")
                score += 5
            else:
                signals.append(f"❌ PCR {pcr:.2f} > 0.9 (bullish bias)")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_max_pain_movement(self, df_chain: pd.DataFrame, spot_price: float, is_upside: bool) -> Dict:
        """Analyze max pain movement"""
        signals = []
        score = 0
        
        # Simple max pain approximation (you can enhance this)
        max_ce_oi_strike = df_chain.loc[df_chain['oi_ce'].idxmax()]['strikePrice']
        max_pe_oi_strike = df_chain.loc[df_chain['oi_pe'].idxmax()]['strikePrice']
        
        if is_upside:
            if max_pe_oi_strike > spot_price:
                signals.append(f"✅ Max PE OI at ₹{max_pe_oi_strike:.0f} (above spot)")
                score += 10
            else:
                signals.append(f"❌ Max PE OI at ₹{max_pe_oi_strike:.0f} (below spot)")
        else:
            if max_ce_oi_strike < spot_price:
                signals.append(f"✅ Max CE OI at ₹{max_ce_oi_strike:.0f} (below spot)")
                score += 10
            else:
                signals.append(f"❌ Max CE OI at ₹{max_ce_oi_strike:.0f} (above spot)")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_oi_wall_breakdown(self, atm_strikes: pd.DataFrame, is_upside: bool) -> Dict:
        """Analyze OI wall breakdown"""
        signals = []
        score = 0
        
        if is_upside:
            # Upside: CE OI should unwind, PE OI should build
            ce_oi_change = atm_strikes['change_oi_ce'].sum()
            pe_oi_change = atm_strikes['change_oi_pe'].sum()
            
            if ce_oi_change < 0:
                signals.append("✅ CE OI wall breaking (unwinding)")
                score += 10
            if pe_oi_change > 0:
                signals.append("✅ PE OI building (fresh writing)")
                score += 5
        else:
            # Downside: PE OI should unwind, CE OI should build
            ce_oi_change = atm_strikes['change_oi_ce'].sum()
            pe_oi_change = atm_strikes['change_oi_pe'].sum()
            
            if pe_oi_change < 0:
                signals.append("✅ PE OI wall breaking (unwinding)")
                score += 10
            if ce_oi_change > 0:
                signals.append("✅ CE OI building (fresh writing)")
                score += 5
        
        return {'signals': signals, 'score': score}
    
    def _analyze_oi_divergence(self, atm_strikes: pd.DataFrame, is_top_reversal: bool) -> Dict:
        """Analyze OI divergence for reversal detection"""
        signals = []
        score = 0
        
        ce_oi_change = atm_strikes['change_oi_ce'].sum()
        pe_oi_change = atm_strikes['change_oi_pe'].sum()
        
        if is_top_reversal:
            # Top reversal: Price ↑ but CE OI ↑ (sellers loading)
            if ce_oi_change > 0 and pe_oi_change < 0:
                signals.append("✅ OI Divergence: CE OI ↑ + PE OI ↓ (sellers loading)")
                score += 30
            else:
                signals.append("❌ No clear OI divergence pattern")
        else:
            # Bottom reversal: Price ↓ but PE OI ↑ (put writers loading)
            if pe_oi_change > 0 and ce_oi_change < 0:
                signals.append("✅ OI Divergence: PE OI ↑ + CE OI ↓ (put writers loading)")
                score += 30
            else:
                signals.append("❌ No clear OI divergence pattern")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_iv_crash(self, atm_strikes: pd.DataFrame) -> Dict:
        """Analyze IV crash for reversal detection"""
        signals = []
        score = 0
        
        # Check if IV is collapsing (you might want historical comparison)
        avg_iv = (atm_strikes['iv_ce'].mean() + atm_strikes['iv_pe'].mean()) / 2
        
        if avg_iv < 15:
            signals.append(f"✅ IV Crash: Avg IV {avg_iv:.1f}% (smart money exiting)")
            score += 20
        elif avg_iv < 20:
            signals.append(f"⚠️ Moderate IV: {avg_iv:.1f}%")
            score += 10
        else:
            signals.append(f"❌ High IV: {avg_iv:.1f}% (volatility present)")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_writer_defense(self, df_chain: pd.DataFrame, spot_price: float, is_top_reversal: bool) -> Dict:
        """Analyze writer defense at key strikes"""
        signals = []
        score = 0
        
        if is_top_reversal:
            # Find resistance strike with high CE OI
            above_spot = df_chain[df_chain['strikePrice'] > spot_price]
            if not above_spot.empty:
                resistance_strike = above_spot.nlargest(1, 'oi_ce')
                if not resistance_strike.empty:
                    strike = resistance_strike['strikePrice'].values[0]
                    oi = resistance_strike['oi_ce'].values[0]
                    if oi > 1000000:  # 1M+ OI indicates strong resistance
                        signals.append(f"✅ Writer Defense: ₹{strike:.0f} CE OI {oi:,.0f}")
                        score += 25
        else:
            # Find support strike with high PE OI
            below_spot = df_chain[df_chain['strikePrice'] < spot_price]
            if not below_spot.empty:
                support_strike = below_spot.nlargest(1, 'oi_pe')
                if not support_strike.empty:
                    strike = support_strike['strikePrice'].values[0]
                    oi = support_strike['oi_pe'].values[0]
                    if oi > 1000000:  # 1M+ OI indicates strong support
                        signals.append(f"✅ Writer Defense: ₹{strike:.0f} PE OI {oi:,.0f}")
                        score += 25
        
        if score == 0:
            signals.append("❌ No strong writer defense detected")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_opposite_oi_build(self, atm_strikes: pd.DataFrame, is_top_reversal: bool) -> Dict:
        """Analyze opposite OI build for reversal"""
        signals = []
        score = 0
        
        ce_oi_change = atm_strikes['change_oi_ce'].sum()
        pe_oi_change = atm_strikes['change_oi_pe'].sum()
        
        if is_top_reversal:
            # Top reversal: New CE writing happening during uptrend
            if ce_oi_change > 10000:  # Significant CE writing
                signals.append(f"✅ Opposite OI: CE writing {ce_oi_change:,.0f} (reversal signal)")
                score += 15
            else:
                signals.append("❌ No significant opposite OI build")
        else:
            # Bottom reversal: New PE writing happening during downtrend
            if pe_oi_change > 10000:  # Significant PE writing
                signals.append(f"✅ Opposite OI: PE writing {pe_oi_change:,.0f} (reversal signal)")
                score += 15
            else:
                signals.append("❌ No significant opposite OI build")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_pcr_extremes(self, df_chain: pd.DataFrame) -> Dict:
        """Analyze PCR extremes for reversal signals"""
        signals = []
        score = 0
        
        total_ce_oi = df_chain['oi_ce'].sum()
        total_pe_oi = df_chain['oi_pe'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        if pcr > 1.4:
            signals.append(f"✅ PCR Extreme: {pcr:.2f} > 1.4 (overbought, reversal likely)")
            score += 10
        elif pcr < 0.5:
            signals.append(f"✅ PCR Extreme: {pcr:.2f} < 0.5 (oversold, reversal likely)")
            score += 10
        else:
            signals.append(f"⚠️ PCR Normal: {pcr:.2f}")
            score += 5
        
        return {'signals': signals, 'score': score}

    def get_breakout_reversal_score(self, df_chain: pd.DataFrame, spot_price: float, 
                                  price_action: Dict) -> Dict[str, Any]:
        """
        Combined breakout/reversal scoring system (0-100)
        """
        # Analyze breakout first
        price_change = price_action.get('price_change', 0)
        volume_change = price_action.get('volume_change', 0)
        
        breakout_analysis = self.analyze_breakout_confirmation(df_chain, spot_price, price_change, volume_change)
        reversal_analysis = self.analyze_reversal_confirmation(df_chain, spot_price, price_action)
        
        # Determine overall market state
        breakout_confidence = breakout_analysis.get('breakout_confidence', 0)
        reversal_confidence = reversal_analysis.get('reversal_confidence', 0)
        
        if breakout_confidence >= 60 and reversal_confidence < 50:
            market_state = "STRONG_BREAKOUT"
            overall_score = breakout_confidence
        elif reversal_confidence >= 70 and breakout_confidence < 40:
            market_state = "STRONG_REVERSAL"
            overall_score = reversal_confidence
        elif breakout_confidence >= 40 and reversal_confidence >= 50:
            market_state = "CONFLICT_ZONE"
            overall_score = (breakout_confidence + reversal_confidence) / 2
        else:
            market_state = "NEUTRAL_CHOPPY"
            overall_score = max(breakout_confidence, reversal_confidence)
        
        return {
            'overall_score': overall_score,
            'market_state': market_state,
            'breakout_analysis': breakout_analysis,
            'reversal_analysis': reversal_analysis,
            'trading_signal': self._generate_trading_signal(market_state, overall_score)
        }
    
    def _generate_trading_signal(self, market_state: str, score: float) -> Dict[str, Any]:
        """Generate trading signals based on analysis"""
        if market_state == "STRONG_BREAKOUT":
            if score >= 70:
                return {'action': 'STRONG_BUY', 'confidence': 'HIGH', 'message': 'Real breakout confirmed'}
            else:
                return {'action': 'MODERATE_BUY', 'confidence': 'MEDIUM', 'message': 'Breakout likely'}
        
        elif market_state == "STRONG_REVERSAL":
            if score >= 75:
                return {'action': 'STRONG_SELL', 'confidence': 'HIGH', 'message': 'Reversal confirmed'}
            else:
                return {'action': 'MODERATE_SELL', 'confidence': 'MEDIUM', 'message': 'Reversal likely'}
        
        elif market_state == "CONFLICT_ZONE":
            return {'action': 'WAIT', 'confidence': 'LOW', 'message': 'Market in conflict, wait for clarity'}
        
        else:  # NEUTRAL_CHOPPY
            return {'action': 'RANGE_TRADE', 'confidence': 'LOW', 'message': 'Market choppy, trade ranges'}


# =============================================
# NSE OPTIONS ANALYZER (FROM SECOND APP)
# =============================================

class NSEOptionsAnalyzer:
    """Integrated NSE Options Analyzer with complete ATM bias analysis"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.NSE_INSTRUMENTS = {
            'indices': {
                'NIFTY': {'lot_size': 50, 'atm_range': 200, 'zone_size': 100},
                'BANKNIFTY': {'lot_size': 25, 'atm_range': 400, 'zone_size': 200},
                'FINNIFTY': {'lot_size': 40, 'atm_range': 200, 'zone_size': 100},
            },
            'stocks': {
                'RELIANCE': {'lot_size': 250, 'atm_range': 100, 'zone_size': 50},
                'TCS': {'lot_size': 150, 'atm_range': 100, 'zone_size': 50},
            }
        }
        self.last_refresh_time = {}
        self.refresh_interval = 2  # 2 minutes default refresh
        self.cached_bias_data = {}
        self.institutional_analyzer = InstitutionalOIAdvanced()
        self.breakout_analyzer = BreakoutReversalAnalyzer()
        
    def set_refresh_interval(self, minutes: int):
        """Set auto-refresh interval"""
        self.refresh_interval = minutes
    
    def should_refresh_data(self, instrument: str) -> bool:
        """Check if data should be refreshed based on last refresh time"""
        current_time = datetime.now(self.ist)
        
        if instrument not in self.last_refresh_time:
            self.last_refresh_time[instrument] = current_time
            return True
        
        last_refresh = self.last_refresh_time[instrument]
        time_diff = (current_time - last_refresh).total_seconds() / 60
        
        if time_diff >= self.refresh_interval:
            self.last_refresh_time[instrument] = current_time
            return True
        
        return False
        
    def calculate_greeks(self, option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float, float, float, float]:
        """Calculate option Greeks"""
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type == 'CE':
                delta = norm.cdf(d1)
            else:
                delta = -norm.cdf(-d1)
                
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'CE' else (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
            
            return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)
        except:
            return 0, 0, 0, 0, 0

    def fetch_option_chain_data(self, instrument: str) -> Dict[str, Any]:
        """Fetch option chain data from NSE"""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=5)

            url_instrument = instrument.replace(' ', '%20')
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={url_instrument}" if instrument in self.NSE_INSTRUMENTS['indices'] else \
                  f"https://www.nseindia.com/api/option-chain-equities?symbol={url_instrument}"

            response = session.get(url, timeout=10)
            data = response.json()

            records = data['records']['data']
            expiry = data['records']['expiryDates'][0]
            underlying = data['records']['underlyingValue']

            # Calculate totals
            total_ce_oi = sum(item['CE']['openInterest'] for item in records if 'CE' in item)
            total_pe_oi = sum(item['PE']['openInterest'] for item in records if 'PE' in item)
            total_ce_change = sum(item['CE']['changeinOpenInterest'] for item in records if 'CE' in item)
            total_pe_change = sum(item['PE']['changeinOpenInterest'] for item in records if 'PE' in item)

            return {
                'success': True,
                'instrument': instrument,
                'spot': underlying,
                'expiry': expiry,
                'total_ce_oi': total_ce_oi,
                'total_pe_oi': total_pe_oi,
                'total_ce_change': total_ce_change,
                'total_pe_change': total_pe_change,
                'records': records
            }
        except Exception as e:
            return {
                'success': False,
                'instrument': instrument,
                'error': str(e)
            }

    def delta_volume_bias(self, price: float, volume: float, chg_oi: float) -> str:
        """Calculate delta volume bias"""
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

    def final_verdict(self, score: float) -> str:
        """Determine final verdict based on score"""
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

    def determine_level(self, row: pd.Series) -> str:
        """Determine support/resistance level based on OI"""
        ce_oi = row['openInterest_CE']
        pe_oi = row['openInterest_PE']

        # Strong Support condition
        if pe_oi > 1.12 * ce_oi:
            return "Support"
        # Strong Resistance condition
        elif ce_oi > 1.12 * pe_oi:
            return "Resistance"
        # Neutral if none dominant
        else:
            return "Neutral"

    def calculate_max_pain(self, df_full_chain: pd.DataFrame) -> Optional[float]:
        """Calculate Max Pain strike"""
        try:
            strikes = df_full_chain['strikePrice'].unique()
            pain_values = []

            for strike in strikes:
                call_pain = 0
                put_pain = 0

                # Calculate pain for all strikes
                for _, row in df_full_chain.iterrows():
                    row_strike = row['strikePrice']

                    # Call pain: If strike price > current strike, calls are ITM
                    if row_strike < strike:
                        call_pain += (strike - row_strike) * row.get('openInterest_CE', 0)

                    # Put pain: If strike price < current strike, puts are ITM
                    if row_strike > strike:
                        put_pain += (row_strike - strike) * row.get('openInterest_PE', 0)

                total_pain = call_pain + put_pain
                pain_values.append({'strike': strike, 'pain': total_pain})

            # Max pain is the strike with minimum total pain
            max_pain_data = min(pain_values, key=lambda x: x['pain'])
            return max_pain_data['strike']
        except:
            return None

    def calculate_synthetic_future_bias(self, atm_ce_price: float, atm_pe_price: float, atm_strike: float, spot_price: float) -> Tuple[str, float, float]:
        """Calculate Synthetic Future Bias at ATM"""
        try:
            synthetic_future = atm_strike + atm_ce_price - atm_pe_price
            difference = synthetic_future - spot_price

            if difference > 5:  # Threshold can be adjusted
                return "Bullish", synthetic_future, difference
            elif difference < -5:
                return "Bearish", synthetic_future, difference
            else:
                return "Neutral", synthetic_future, difference
        except:
            return "Neutral", 0, 0

    def calculate_atm_buildup_pattern(self, atm_ce_oi: float, atm_pe_oi: float, atm_ce_change: float, atm_pe_change: float) -> str:
        """Determine ATM buildup pattern based on OI changes"""
        try:
            # Classify based on OI changes
            if atm_ce_change > 0 and atm_pe_change > 0:
                if atm_ce_change > atm_pe_change:
                    return "Long Buildup (Bearish)"
                else:
                    return "Short Buildup (Bullish)"
            elif atm_ce_change < 0 and atm_pe_change < 0:
                if abs(atm_ce_change) > abs(atm_pe_change):
                    return "Short Covering (Bullish)"
                else:
                    return "Long Unwinding (Bearish)"
            elif atm_ce_change > 0 and atm_pe_change < 0:
                return "Call Writing (Bearish)"
            elif atm_ce_change < 0 and atm_pe_change > 0:
                return "Put Writing (Bullish)"
            else:
                return "Neutral"
        except:
            return "Neutral"

    def calculate_atm_vega_bias(self, atm_ce_vega: float, atm_pe_vega: float, atm_ce_oi: float, atm_pe_oi: float) -> Tuple[str, float]:
        """Calculate ATM Vega exposure bias"""
        try:
            ce_vega_exposure = atm_ce_vega * atm_ce_oi
            pe_vega_exposure = atm_pe_vega * atm_pe_oi

            total_vega_exposure = ce_vega_exposure + pe_vega_exposure

            if pe_vega_exposure > ce_vega_exposure * 1.1:
                return "Bullish (High Put Vega)", total_vega_exposure
            elif ce_vega_exposure > pe_vega_exposure * 1.1:
                return "Bearish (High Call Vega)", total_vega_exposure
            else:
                return "Neutral", total_vega_exposure
        except:
            return "Neutral", 0

    def find_call_resistance_put_support(self, df_full_chain: pd.DataFrame, spot_price: float) -> Tuple[Optional[float], Optional[float]]:
        """Find key resistance (from Call OI) and support (from Put OI) strikes"""
        try:
            # Find strikes above spot with highest Call OI (Resistance)
            above_spot = df_full_chain[df_full_chain['strikePrice'] > spot_price].copy()
            if not above_spot.empty:
                call_resistance = above_spot.nlargest(1, 'openInterest_CE')['strikePrice'].values[0]
            else:
                call_resistance = None

            # Find strikes below spot with highest Put OI (Support)
            below_spot = df_full_chain[df_full_chain['strikePrice'] < spot_price].copy()
            if not below_spot.empty:
                put_support = below_spot.nlargest(1, 'openInterest_PE')['strikePrice'].values[0]
            else:
                put_support = None

            return call_resistance, put_support
        except:
            return None, None

    def calculate_total_vega_bias(self, df_full_chain: pd.DataFrame) -> Tuple[str, float, float, float]:
        """Calculate total Vega bias across all strikes"""
        try:
            total_ce_vega = (df_full_chain['Vega_CE'] * df_full_chain['openInterest_CE']).sum()
            total_pe_vega = (df_full_chain['Vega_PE'] * df_full_chain['openInterest_PE']).sum()

            total_vega = total_ce_vega + total_pe_vega

            if total_pe_vega > total_ce_vega * 1.1:
                return "Bullish (Put Heavy)", total_vega, total_ce_vega, total_pe_vega
            elif total_ce_vega > total_pe_vega * 1.1:
                return "Bearish (Call Heavy)", total_vega, total_ce_vega, total_pe_vega
            else:
                return "Neutral", total_vega, total_ce_vega, total_pe_vega
        except:
            return "Neutral", 0, 0, 0

    def detect_unusual_activity(self, df_full_chain: pd.DataFrame, spot_price: float) -> List[Dict[str, Any]]:
        """Detect strikes with unusual activity (high volume relative to OI)"""
        try:
            unusual_strikes = []

            for _, row in df_full_chain.iterrows():
                strike = row['strikePrice']

                # Check Call side
                ce_oi = row.get('openInterest_CE', 0)
                ce_volume = row.get('totalTradedVolume_CE', 0)
                if ce_oi > 0 and ce_volume / ce_oi > 0.5:  # Volume > 50% of OI
                    unusual_strikes.append({
                        'strike': strike,
                        'type': 'CE',
                        'volume_oi_ratio': ce_volume / ce_oi if ce_oi > 0 else 0,
                        'volume': ce_volume,
                        'oi': ce_oi
                    })

                # Check Put side
                pe_oi = row.get('openInterest_PE', 0)
                pe_volume = row.get('totalTradedVolume_PE', 0)
                if pe_oi > 0 and pe_volume / pe_oi > 0.5:
                    unusual_strikes.append({
                        'strike': strike,
                        'type': 'PE',
                        'volume_oi_ratio': pe_volume / pe_oi if pe_oi > 0 else 0,
                        'volume': pe_volume,
                        'oi': pe_oi
                    })

            # Sort by volume/OI ratio and return top 5
            unusual_strikes.sort(key=lambda x: x['volume_oi_ratio'], reverse=True)
            return unusual_strikes[:5]
        except:
            return []

    def calculate_overall_buildup_pattern(self, df_full_chain: pd.DataFrame, spot_price: float) -> str:
        """Calculate overall buildup pattern across ITM, ATM, and OTM strikes"""
        try:
            # Separate into ITM, ATM, OTM
            itm_calls = df_full_chain[df_full_chain['strikePrice'] < spot_price].copy()
            otm_calls = df_full_chain[df_full_chain['strikePrice'] > spot_price].copy()
            atm_strikes = df_full_chain[abs(df_full_chain['strikePrice'] - spot_price) <= 50].copy()

            # Calculate OI changes for each zone
            itm_ce_change = itm_calls['changeinOpenInterest_CE'].sum() if not itm_calls.empty else 0
            itm_pe_change = itm_calls['changeinOpenInterest_PE'].sum() if not itm_calls.empty else 0

            otm_ce_change = otm_calls['changeinOpenInterest_CE'].sum() if not otm_calls.empty else 0
            otm_pe_change = otm_calls['changeinOpenInterest_PE'].sum() if not otm_calls.empty else 0

            atm_ce_change = atm_strikes['changeinOpenInterest_CE'].sum() if not atm_strikes.empty else 0
            atm_pe_change = atm_strikes['changeinOpenInterest_PE'].sum() if not atm_strikes.empty else 0

            # Determine pattern
            patterns = []

            if itm_pe_change > 0 and otm_ce_change > 0:
                patterns.append("Protective Strategy (Bullish)")
            elif itm_ce_change > 0 and otm_pe_change > 0:
                patterns.append("Protective Strategy (Bearish)")

            if atm_ce_change > atm_pe_change and abs(atm_ce_change) > 1000:
                patterns.append("Strong Call Writing (Bearish)")
            elif atm_pe_change > atm_ce_change and abs(atm_pe_change) > 1000:
                patterns.append("Strong Put Writing (Bullish)")

            if otm_ce_change > itm_ce_change and otm_ce_change > 1000:
                patterns.append("OTM Call Buying (Bullish)")
            elif otm_pe_change > itm_pe_change and otm_pe_change > 1000:
                patterns.append("OTM Put Buying (Bearish)")

            return " | ".join(patterns) if patterns else "Balanced/Neutral"

        except:
            return "Neutral"

    def analyze_comprehensive_atm_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Comprehensive ATM bias analysis with all metrics"""
        try:
            data = self.fetch_option_chain_data(instrument)
            if not data['success']:
                return None

            records = data['records']
            spot = data['spot']
            expiry = data['expiry']

            # Calculate time to expiry
            today = datetime.now(self.ist)
            expiry_date = self.ist.localize(datetime.strptime(expiry, "%d-%b-%Y"))
            T = max((expiry_date - today).days, 1) / 365
            r = 0.06

            calls, puts = [], []
            for item in records:
                if 'CE' in item and item['CE']['expiryDate'] == expiry:
                    ce = item['CE']
                    if ce['impliedVolatility'] > 0:
                        greeks = self.calculate_greeks('CE', spot, ce['strikePrice'], T, r, ce['impliedVolatility'] / 100)
                        ce.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    calls.append(ce)

                if 'PE' in item and item['PE']['expiryDate'] == expiry:
                    pe = item['PE']
                    if pe['impliedVolatility'] > 0:
                        greeks = self.calculate_greeks('PE', spot, pe['strikePrice'], T, r, pe['impliedVolatility'] / 100)
                        pe.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    puts.append(pe)

            if not calls or not puts:
                return None

            df_ce = pd.DataFrame(calls)
            df_pe = pd.DataFrame(puts)
            df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

            # Find ATM strike
            atm_range = self.NSE_INSTRUMENTS['indices'].get(instrument, {}).get('atm_range', 200)
            atm_strike = min(df['strikePrice'], key=lambda x: abs(x - spot))
            df_atm = df[abs(df['strikePrice'] - atm_strike) <= atm_range]

            if df_atm.empty:
                return None

            # Get ATM row data
            atm_df = df[df['strikePrice'] == atm_strike]
            if not atm_df.empty:
                atm_ce_price = atm_df['lastPrice_CE'].values[0]
                atm_pe_price = atm_df['lastPrice_PE'].values[0]
                atm_ce_oi = atm_df['openInterest_CE'].values[0]
                atm_pe_oi = atm_df['openInterest_PE'].values[0]
                atm_ce_change = atm_df['changeinOpenInterest_CE'].values[0]
                atm_pe_change = atm_df['changeinOpenInterest_PE'].values[0]
                atm_ce_vega = atm_df['Vega_CE'].values[0]
                atm_pe_vega = atm_df['Vega_PE'].values[0]
            else:
                return None

            # Calculate all comprehensive metrics
            synthetic_bias, synthetic_future, synthetic_diff = self.calculate_synthetic_future_bias(
                atm_ce_price, atm_pe_price, atm_strike, spot
            )
            
            atm_buildup = self.calculate_atm_buildup_pattern(
                atm_ce_oi, atm_pe_oi, atm_ce_change, atm_pe_change
            )
            
            atm_vega_bias, atm_vega_exposure = self.calculate_atm_vega_bias(
                atm_ce_vega, atm_pe_vega, atm_ce_oi, atm_pe_oi
            )
            
            max_pain_strike = self.calculate_max_pain(df)
            distance_from_max_pain = spot - max_pain_strike if max_pain_strike else 0
            
            call_resistance, put_support = self.find_call_resistance_put_support(df, spot)
            
            total_vega_bias, total_vega, total_ce_vega_exp, total_pe_vega_exp = self.calculate_total_vega_bias(df)
            
            unusual_activity = self.detect_unusual_activity(df, spot)
            
            overall_buildup = self.calculate_overall_buildup_pattern(df, spot)

            # Calculate detailed ATM bias breakdown
            detailed_atm_bias = self.calculate_detailed_atm_bias(df_atm, atm_strike, spot)

            # Calculate comprehensive bias score
            weights = {
                "oi_bias": 2, "chg_oi_bias": 2, "volume_bias": 1, 
                "iv_bias": 1, "premium_bias": 1, "delta_bias": 1,
                "synthetic_bias": 2, "vega_bias": 1, "max_pain_bias": 1
            }

            total_score = 0
            
            # OI Bias
            oi_bias = "Bullish" if data['total_pe_oi'] > data['total_ce_oi'] else "Bearish"
            total_score += weights["oi_bias"] if oi_bias == "Bullish" else -weights["oi_bias"]
            
            # Change in OI Bias
            chg_oi_bias = "Bullish" if data['total_pe_change'] > data['total_ce_change'] else "Bearish"
            total_score += weights["chg_oi_bias"] if chg_oi_bias == "Bullish" else -weights["chg_oi_bias"]
            
            # Synthetic Bias
            total_score += weights["synthetic_bias"] if synthetic_bias == "Bullish" else -weights["synthetic_bias"] if synthetic_bias == "Bearish" else 0
            
            # Vega Bias
            vega_bias_score = 1 if "Bullish" in atm_vega_bias else -1 if "Bearish" in atm_vega_bias else 0
            total_score += weights["vega_bias"] * vega_bias_score
            
            # Max Pain Bias (if spot above max pain, bullish)
            max_pain_bias = "Bullish" if distance_from_max_pain > 0 else "Bearish" if distance_from_max_pain < 0 else "Neutral"
            total_score += weights["max_pain_bias"] if max_pain_bias == "Bullish" else -weights["max_pain_bias"] if max_pain_bias == "Bearish" else 0

            overall_bias = self.final_verdict(total_score)

            return {
                'instrument': instrument,
                'spot_price': spot,
                'atm_strike': atm_strike,
                'overall_bias': overall_bias,
                'bias_score': total_score,
                'pcr_oi': data['total_pe_oi'] / data['total_ce_oi'] if data['total_ce_oi'] > 0 else 0,
                'pcr_change': abs(data['total_pe_change']) / abs(data['total_ce_change']) if data['total_ce_change'] != 0 else 0,
                'total_ce_oi': data['total_ce_oi'],
                'total_pe_oi': data['total_pe_oi'],
                'total_ce_change': data['total_ce_change'],
                'total_pe_change': data['total_pe_change'],
                'detailed_atm_bias': detailed_atm_bias,
                'comprehensive_metrics': {
                    'synthetic_bias': synthetic_bias,
                    'synthetic_future': synthetic_future,
                    'synthetic_diff': synthetic_diff,
                    'atm_buildup': atm_buildup,
                    'atm_vega_bias': atm_vega_bias,
                    'atm_vega_exposure': atm_vega_exposure,
                    'max_pain_strike': max_pain_strike,
                    'distance_from_max_pain': distance_from_max_pain,
                    'call_resistance': call_resistance,
                    'put_support': put_support,
                    'total_vega_bias': total_vega_bias,
                    'total_vega': total_vega,
                    'unusual_activity_count': len(unusual_activity),
                    'overall_buildup': overall_buildup
                }
            }

        except Exception as e:
            print(f"Error in ATM bias analysis: {e}")
            return None

    def calculate_detailed_atm_bias(self, df_atm: pd.DataFrame, atm_strike: float, spot_price: float) -> Dict[str, Any]:
        """Calculate detailed ATM bias breakdown for all metrics"""
        try:
            detailed_bias = {}
            
            for _, row in df_atm.iterrows():
                if row['strikePrice'] == atm_strike:
                    # Calculate per-strike delta and gamma exposure
                    ce_delta_exp = row['Delta_CE'] * row['openInterest_CE']
                    pe_delta_exp = row['Delta_PE'] * row['openInterest_PE']
                    ce_gamma_exp = row['Gamma_CE'] * row['openInterest_CE']
                    pe_gamma_exp = row['Gamma_PE'] * row['openInterest_PE']

                    net_delta_exp = ce_delta_exp + pe_delta_exp
                    net_gamma_exp = ce_gamma_exp + pe_gamma_exp
                    strike_iv_skew = row['impliedVolatility_PE'] - row['impliedVolatility_CE']

                    delta_exp_bias = "Bullish" if net_delta_exp > 0 else "Bearish" if net_delta_exp < 0 else "Neutral"
                    gamma_exp_bias = "Bullish" if net_gamma_exp > 0 else "Bearish" if net_gamma_exp < 0 else "Neutral"
                    iv_skew_bias = "Bullish" if strike_iv_skew > 0 else "Bearish" if strike_iv_skew < 0 else "Neutral"

                    detailed_bias = {
                        "Strike": row['strikePrice'],
                        "Zone": 'ATM',
                        "Level": self.determine_level(row),
                        "OI_Bias": "Bullish" if row['openInterest_CE'] < row['openInterest_PE'] else "Bearish",
                        "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
                        "Volume_Bias": "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish",
                        "Delta_Bias": "Bullish" if abs(row['Delta_PE']) > abs(row['Delta_CE']) else "Bearish",
                        "Gamma_Bias": "Bullish" if row['Gamma_CE'] < row['Gamma_PE'] else "Bearish",
                        "Premium_Bias": "Bullish" if row['lastPrice_CE'] < row['lastPrice_PE'] else "Bearish",
                        "AskQty_Bias": "Bullish" if row['askQty_PE'] > row['askQty_CE'] else "Bearish",
                        "BidQty_Bias": "Bearish" if row['bidQty_PE'] > row['bidQty_CE'] else "Bullish",
                        "IV_Bias": "Bullish" if row['impliedVolatility_CE'] > row['impliedVolatility_PE'] else "Bearish",
                        "DVP_Bias": self.delta_volume_bias(
                            row['lastPrice_CE'] - row['lastPrice_PE'],
                            row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
                            row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
                        ),
                        "Delta_Exposure_Bias": delta_exp_bias,
                        "Gamma_Exposure_Bias": gamma_exp_bias,
                        "IV_Skew_Bias": iv_skew_bias,
                        # Raw values for display
                        "CE_OI": row['openInterest_CE'],
                        "PE_OI": row['openInterest_PE'],
                        "CE_Change": row['changeinOpenInterest_CE'],
                        "PE_Change": row['changeinOpenInterest_PE'],
                        "CE_Volume": row['totalTradedVolume_CE'],
                        "PE_Volume": row['totalTradedVolume_PE'],
                        "CE_Price": row['lastPrice_CE'],
                        "PE_Price": row['lastPrice_PE'],
                        "CE_IV": row['impliedVolatility_CE'],
                        "PE_IV": row['impliedVolatility_PE'],
                        "Delta_CE": row['Delta_CE'],
                        "Delta_PE": row['Delta_PE'],
                        "Gamma_CE": row['Gamma_CE'],
                        "Gamma_PE": row['Gamma_PE']
                    }
                    break
            
            return detailed_bias
            
        except Exception as e:
            print(f"Error in detailed ATM bias: {e}")
            return {}

    def analyze_comprehensive_institutional_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Enhanced analysis with institutional footprint and Gamma sequencing"""
        try:
            # Get existing comprehensive analysis
            basic_analysis = self.analyze_comprehensive_atm_bias(instrument)
            if not basic_analysis:
                return None
            
            # Fetch fresh chain data for institutional analysis
            data = self.fetch_option_chain_data(instrument)
            if not data['success']:
                return None
            
            records = data['records']
            spot = data['spot']
            expiry = data['expiry']
            
            # Process option chain data
            calls, puts = [], []
            for item in records:
                if 'CE' in item and item['CE']['expiryDate'] == expiry:
                    calls.append(item['CE'])
                if 'PE' in item and item['PE']['expiryDate'] == expiry:
                    puts.append(item['PE'])
            
            if not calls or not puts:
                return None
            
            df_ce = pd.DataFrame(calls)
            df_pe = pd.DataFrame(puts)
            df_chain = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
            
            # FIX 4: Normalize column names
            df_chain = normalize_chain_columns(df_chain)
            
            # Add institutional analysis
            institutional_analysis = self.institutional_analyzer.analyze_atm_institutional_footprint(df_chain, spot)
            
            # Add breakout/reversal analysis
            price_action = {
                'price_change': 0.5,  # Mock data - replace with real price change
                'volume_change': 15,
                'has_upper_wick': False,
                'has_lower_wick': False,
                'is_overbought': False,
                'is_oversold': False
            }
            
            breakout_analysis = self.breakout_analyzer.get_breakout_reversal_score(df_chain, spot, price_action)
            
            # Combine analyses
            enhanced_analysis = basic_analysis.copy()
            enhanced_analysis['institutional_analysis'] = institutional_analysis
            enhanced_analysis['breakout_reversal_analysis'] = breakout_analysis
            
            # Calculate combined bias score
            basic_score = basic_analysis.get('bias_score', 0)
            institutional_score = institutional_analysis.get('score', 0)
            gamma_score = institutional_analysis.get('gamma_analysis', {}).get('gamma_score', 0)
            breakout_score = breakout_analysis.get('overall_score', 0)
            
            # Weighted combined score (25% basic, 30% institutional, 20% gamma, 25% breakout)
            combined_score = (basic_score * 0.25) + (institutional_score * 0.3) + (gamma_score * 0.2) + (breakout_score * 0.25)
            
            # Determine combined bias
            if combined_score >= 2:
                combined_bias = "Strong Bullish"
            elif combined_score >= 0.5:
                combined_bias = "Bullish"
            elif combined_score <= -2:
                combined_bias = "Strong Bearish"
            elif combined_score <= -0.5:
                combined_bias = "Bearish"
            else:
                combined_bias = "Neutral"
            
            enhanced_analysis['combined_bias'] = combined_bias
            enhanced_analysis['combined_score'] = combined_score
            enhanced_analysis['institutional_score'] = institutional_score
            enhanced_analysis['gamma_score'] = gamma_score
            enhanced_analysis['breakout_score'] = breakout_score
            
            return enhanced_analysis
            
        except Exception as e:
            print(f"Error in enhanced institutional analysis: {e}")
            return None

    def get_overall_market_bias(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get comprehensive market bias across all instruments with auto-refresh"""
        instruments = list(self.NSE_INSTRUMENTS['indices'].keys())
        results = []
        
        for instrument in instruments:
            if force_refresh or self.should_refresh_data(instrument):
                try:
                    bias_data = self.analyze_comprehensive_institutional_bias(instrument)
                    if bias_data:
                        results.append(bias_data)
                        # Update cache
                        self.cached_bias_data[instrument] = bias_data
                except Exception as e:
                    print(f"Error fetching {instrument}: {e}")
                    # Use cached data if available
                    if instrument in self.cached_bias_data:
                        results.append(self.cached_bias_data[instrument])
            else:
                # Return cached data if available and not forcing refresh
                if instrument in self.cached_bias_data:
                    results.append(self.cached_bias_data[instrument])
        
        return results


# =============================================
# STREAMLIT APP UI (ENTRY) - ENHANCED
# =============================================
st.set_page_config(page_title="Bias Analysis Pro - Complete Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Bias Analysis Pro — Complete Single-file App")
st.markdown(
    "This Streamlit app wraps the **BiasAnalysisPro** engine (Pine → Python) and shows bias summary, "
    "price action, option chain analysis, and bias tabulation."
)

# Initialize all analyzers
analysis = BiasAnalysisPro()
options_analyzer = NSEOptionsAnalyzer()
vob_indicator = VolumeOrderBlocks(sensitivity=5)
gamma_analyzer = GammaSequenceAnalyzer()
institutional_analyzer = InstitutionalOIAdvanced()
breakout_analyzer = BreakoutReversalAnalyzer()

# Sidebar inputs
st.sidebar.header("Data & Symbol")
symbol_input = st.sidebar.text_input("Symbol (Yahoo/Dhan)", value="^NSEI")
period_input = st.sidebar.selectbox("Period", options=['1d', '5d', '7d', '1mo'], index=2)
interval_input = st.sidebar.selectbox("Interval", options=['1m', '5m', '15m', '1h'], index=1)

# Auto-refresh configuration
st.sidebar.header("Auto-Refresh Settings")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (minutes)", min_value=1, max_value=10, value=1)

# Telegram Configuration
st.sidebar.header("🔔 Telegram Alerts")
if telegram_notifier.is_configured():
    st.sidebar.success("✅ Telegram configured via secrets!")
    telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts", value=True)
else:
    st.sidebar.warning("⚠️ Telegram not configured")
    st.sidebar.info("Add to .streamlit/secrets.toml:")
    st.sidebar.code("""
[TELEGRAM]
BOT_TOKEN = "your_bot_token_here"
CHAT_ID = "your_chat_id_here"
""")
    telegram_enabled = False

# Shared state storage
if 'last_df' not in st.session_state:
    st.session_state['last_df'] = None
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None
if 'last_symbol' not in st.session_state:
    st.session_state['last_symbol'] = None
if 'fetch_time' not in st.session_state:
    st.session_state['fetch_time'] = None
if 'market_bias_data' not in st.session_state:
    st.session_state.market_bias_data = None
if 'last_bias_update' not in st.session_state:
    st.session_state.last_bias_update = None
if 'overall_nifty_bias' not in st.session_state:
    st.session_state.overall_nifty_bias = "NEUTRAL"
if 'overall_nifty_score' not in st.session_state:
    st.session_state.overall_nifty_score = 0
if 'atm_detailed_bias' not in st.session_state:
    st.session_state.atm_detailed_bias = None
if 'vob_blocks' not in st.session_state:  # FIX 5: Store VOB blocks
    st.session_state.vob_blocks = {'bullish': [], 'bearish': []}
if 'last_telegram_alert' not in st.session_state:
    st.session_state.last_telegram_alert = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Function to calculate ATM detailed bias score
def calculate_atm_detailed_bias(detailed_bias_data: Dict) -> Tuple[str, float]:
    """Calculate overall ATM bias from detailed bias metrics"""
    if not detailed_bias_data:
        return "NEUTRAL", 0
    
    bias_scores = []
    bias_weights = []
    
    # Define bias mappings with weights
    bias_mappings = {
        'OI_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 2.0},
        'ChgOI_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 2.0},
        'Volume_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
        'Delta_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
        'Gamma_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.0},
        'Premium_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.0},
        'AskQty_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.0},
        'BidQty_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.0},
        'IV_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
        'DVP_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
        'Delta_Exposure_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 2.0},
        'Gamma_Exposure_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
        'IV_Skew_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5}
    }
    
    # Calculate weighted scores
    total_weight = 0
    total_score = 0
    
    for bias_key, mapping in bias_mappings.items():
        if bias_key in detailed_bias_data:
            bias_value = detailed_bias_data[bias_key]
            if bias_value in mapping:
                score = mapping[bias_value]
                weight = mapping['weight']
                total_score += score * weight
                total_weight += weight
    
    if total_weight == 0:
        return "NEUTRAL", 0
    
    # Normalize score to -100 to 100 range
    normalized_score = (total_score / total_weight) * 100
    
    # Determine bias direction
    if normalized_score > 15:
        bias = "BULLISH"
    elif normalized_score < -15:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"
    
    return bias, normalized_score

# Function to run complete analysis
def run_complete_analysis():
    """Run complete analysis for all tabs"""
    st.session_state['last_symbol'] = symbol_input
    
    # Technical Analysis
    with st.spinner("Fetching data and running technical analysis..."):
        df_fetched = analysis.fetch_data(symbol_input, period=period_input, interval=interval_input)
        st.session_state['last_df'] = df_fetched
        st.session_state['fetch_time'] = datetime.now(IST)

    if df_fetched is None or df_fetched.empty:
        st.error("No data fetched. Check symbol or network.")
        return False

    # Run bias analysis
    with st.spinner("Running full bias analysis..."):
        result = analysis.analyze_all_bias_indicators(symbol_input)
        st.session_state['last_result'] = result

    # FIX 5: Run Volume Order Blocks analysis and store results
    with st.spinner("Detecting Volume Order Blocks..."):
        if st.session_state['last_df'] is not None:
            df = st.session_state['last_df']
            bullish_blocks, bearish_blocks = vob_indicator.detect_volume_order_blocks(df)
            st.session_state.vob_blocks = {
                'bullish': bullish_blocks,
                'bearish': bearish_blocks
            }

    # Run ENHANCED options analysis with institutional footprint
    with st.spinner("Running institutional footprint analysis..."):
        enhanced_bias_data = []
        instruments = list(options_analyzer.NSE_INSTRUMENTS['indices'].keys())
        
        for instrument in instruments:
            try:
                bias_data = options_analyzer.analyze_comprehensive_institutional_bias(instrument)
                if bias_data:
                    enhanced_bias_data.append(bias_data)
            except Exception as e:
                print(f"Error in enhanced analysis for {instrument}: {e}")
                # Fallback to basic analysis
                basic_data = options_analyzer.analyze_comprehensive_atm_bias(instrument)
                if basic_data:
                    enhanced_bias_data.append(basic_data)
        
        st.session_state.market_bias_data = enhanced_bias_data
        st.session_state.last_bias_update = datetime.now(IST)
    
    # Calculate ATM detailed bias
    if st.session_state.market_bias_data:
        for instrument_data in st.session_state.market_bias_data:
            if instrument_data['instrument'] == 'NIFTY' and 'detailed_atm_bias' in instrument_data:
                atm_bias, atm_score = calculate_atm_detailed_bias(instrument_data['detailed_atm_bias'])
                st.session_state.atm_detailed_bias = {
                    'bias': atm_bias,
                    'score': atm_score,
                    'details': instrument_data['detailed_atm_bias']
                }
                break
    
    # Calculate overall Nifty bias
    calculate_overall_nifty_bias()
    
    # Send Telegram alert if conditions met
    send_telegram_alert()
    
    st.session_state.analysis_complete = True
    return True

# Function to calculate overall Nifty bias from all tabs
def calculate_overall_nifty_bias():
    """Calculate overall Nifty bias by combining all analysis methods"""
    bias_scores = []
    bias_weights = []
    
    # 1. Technical Analysis Bias (20% weight)
    if st.session_state['last_result'] and st.session_state['last_result'].get('success'):
        tech_result = st.session_state['last_result']
        tech_bias = tech_result.get('overall_bias', 'NEUTRAL')
        tech_score = tech_result.get('overall_score', 0)
        
        if tech_bias == "BULLISH":
            bias_scores.append(1.0)
        elif tech_bias == "BEARISH":
            bias_scores.append(-1.0)
        else:
            bias_scores.append(0.0)
        bias_weights.append(0.20)
    
    # 2. Enhanced Options Chain Bias (30% weight)
    if st.session_state.market_bias_data:
        for instrument_data in st.session_state.market_bias_data:
            if instrument_data['instrument'] == 'NIFTY':
                # Use combined bias if available
                options_bias = instrument_data.get('combined_bias', instrument_data.get('overall_bias', 'Neutral'))
                options_score = instrument_data.get('combined_score', instrument_data.get('bias_score', 0))
                
                # Normalize score to -1 to 1 range
                normalized_score = max(-1, min(1, options_score / 4))
                
                bias_scores.append(normalized_score)
                bias_weights.append(0.30)
                break
    
    # 3. ATM Detailed Bias (15% weight)
    if st.session_state.atm_detailed_bias:
        atm_bias = st.session_state.atm_detailed_bias['bias']
        atm_score = st.session_state.atm_detailed_bias['score']
        
        if atm_bias == "BULLISH":
            bias_scores.append(1.0)
        elif atm_bias == "BEARISH":
            bias_scores.append(-1.0)
        else:
            bias_scores.append(0.0)
        bias_weights.append(0.15)
    
    # 4. Volume Order Blocks Bias (15% weight)
    if st.session_state.vob_blocks:
        bullish_blocks = st.session_state.vob_blocks['bullish']
        bearish_blocks = st.session_state.vob_blocks['bearish']
        
        vob_bias_score = 0
        if len(bullish_blocks) > len(bearish_blocks):
            vob_bias_score = 1.0
        elif len(bearish_blocks) > len(bullish_blocks):
            vob_bias_score = -1.0
        
        bias_scores.append(vob_bias_score)
        bias_weights.append(0.15)
    
    # 5. Breakout/Reversal Analysis (20% weight) - NEW
    if st.session_state.market_bias_data:
        for instrument_data in st.session_state.market_bias_data:
            if instrument_data['instrument'] == 'NIFTY' and 'breakout_reversal_analysis' in instrument_data:
                breakout_data = instrument_data['breakout_reversal_analysis']
                breakout_score = breakout_data.get('overall_score', 0)
                
                # Normalize breakout score to -1 to 1 range
                normalized_breakout_score = max(-1, min(1, breakout_score / 100))
                
                bias_scores.append(normalized_breakout_score)
                bias_weights.append(0.20)
                break
    
    # Calculate weighted average
    if bias_scores and bias_weights:
        total_weight = sum(bias_weights)
        weighted_score = sum(score * weight for score, weight in zip(bias_scores, bias_weights)) / total_weight
        
        if weighted_score > 0.1:
            overall_bias = "BULLISH"
        elif weighted_score < -0.1:
            overall_bias = "BEARISH"
        else:
            overall_bias = "NEUTRAL"
        
        st.session_state.overall_nifty_bias = overall_bias
        st.session_state.overall_nifty_score = weighted_score * 100

# Function to send Telegram alerts
def send_telegram_alert():
    """Send Telegram alert when all three key components are aligned"""
    if not telegram_enabled:
        return
    
    # Get the three key components
    technical_bias = "NEUTRAL"
    options_bias = "NEUTRAL" 
    atm_bias = "NEUTRAL"
    
    if st.session_state['last_result'] and st.session_state['last_result'].get('success'):
        technical_bias = st.session_state['last_result'].get('overall_bias', 'NEUTRAL')
    
    if st.session_state.market_bias_data:
        for instrument_data in st.session_state.market_bias_data:
            if instrument_data['instrument'] == 'NIFTY':
                options_bias = instrument_data.get('combined_bias', instrument_data.get('overall_bias', 'NEUTRAL'))
                break
    
    if st.session_state.atm_detailed_bias:
        atm_bias = st.session_state.atm_detailed_bias['bias']
    
    # Send alert through Telegram notifier
    telegram_notifier.send_bias_alert(
        technical_bias=technical_bias,
        options_bias=options_bias,
        atm_bias=atm_bias,
        overall_bias=st.session_state.overall_nifty_bias,
        score=st.session_state.overall_nifty_score
    )

# AUTO-RUN ANALYSIS ON STARTUP
if not st.session_state.analysis_complete:
    with st.spinner("🚀 Starting initial analysis..."):
        if run_complete_analysis():
            st.success("✅ Initial analysis complete!")
            st.rerun()

# Refresh button
col1, col2 = st.sidebar.columns([2, 1])
with col1:
    if st.button("🔄 Refresh Analysis", type="primary", use_container_width=True):
        if run_complete_analysis():
            st.sidebar.success("Analysis refreshed!")
            st.rerun()
with col2:
    st.sidebar.metric("Auto-Refresh", "ON" if auto_refresh else "OFF")

# Auto-refresh logic
if auto_refresh:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_refresh).total_seconds() / 60
    
    if time_diff >= refresh_interval:
        with st.spinner("Auto-refreshing analysis..."):
            if run_complete_analysis():
                st.session_state.last_refresh = current_time
                st.rerun()

# Display overall Nifty bias prominently
st.sidebar.markdown("---")
st.sidebar.header("Overall Nifty Bias")
if st.session_state.overall_nifty_bias:
    bias_color = "🟢" if st.session_state.overall_nifty_bias == "BULLISH" else "🔴" if st.session_state.overall_nifty_bias == "BEARISH" else "🟡"
    st.sidebar.metric(
        "NIFTY 50 Bias",
        f"{bias_color} {st.session_state.overall_nifty_bias}",
        f"Score: {st.session_state.overall_nifty_score:.1f}"
    )

# Display last update time
if st.session_state.last_bias_update:
    st.sidebar.caption(f"Last update: {st.session_state.last_bias_update.strftime('%H:%M:%S')} IST")

# Enhanced tabs with selected features
tabs = st.tabs([
    "Overall Bias", "Bias Summary", "Price Action", "Option Chain", "Bias Tabulation"
])

# OVERALL BIAS TAB (NEW)
with tabs[0]:
    st.header("🎯 Overall Nifty Bias Analysis")
    
    if not st.session_state.overall_nifty_bias:
        st.info("No analysis run yet. Analysis runs automatically every minute...")
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Display overall bias with color coding
            if st.session_state.overall_nifty_bias == "BULLISH":
                st.success(f"## 🟢 OVERALL NIFTY BIAS: BULLISH")
                st.metric("Bias Score", f"{st.session_state.overall_nifty_score:.1f}", delta="Bullish")
            elif st.session_state.overall_nifty_bias == "BEARISH":
                st.error(f"## 🔴 OVERALL NIFTY BIAS: BEARISH")
                st.metric("Bias Score", f"{st.session_state.overall_nifty_score:.1f}", delta="Bearish", delta_color="inverse")
            else:
                st.warning(f"## 🟡 OVERALL NIFTY BIAS: NEUTRAL")
                st.metric("Bias Score", f"{st.session_state.overall_nifty_score:.1f}")
        
        st.markdown("---")
        
        # Breakdown of bias components
        st.subheader("Bias Components Breakdown")
        
        components_data = []
        
        # Technical Analysis Component
        if st.session_state['last_result'] and st.session_state['last_result'].get('success'):
            tech_result = st.session_state['last_result']
            components_data.append({
                'Component': 'Technical Analysis',
                'Bias': tech_result.get('overall_bias', 'NEUTRAL'),
                'Score': tech_result.get('overall_score', 0),
                'Weight': '20%',
                'Confidence': f"{tech_result.get('overall_confidence', 0):.1f}%"
            })
        
        # Options Analysis Component
        if st.session_state.market_bias_data:
            for instrument_data in st.session_state.market_bias_data:
                if instrument_data['instrument'] == 'NIFTY':
                    components_data.append({
                        'Component': 'Options Chain Overall',
                        'Bias': instrument_data.get('combined_bias', 'Neutral'),
                        'Score': instrument_data.get('combined_score', 0),
                        'Weight': '30%',
                        'Confidence': 'High' if abs(instrument_data.get('combined_score', 0)) > 2 else 'Medium'
                    })
                    break
        
        # ATM Detailed Bias Component
        if st.session_state.atm_detailed_bias:
            atm_data = st.session_state.atm_detailed_bias
            components_data.append({
                'Component': 'ATM Detailed Bias',
                'Bias': atm_data['bias'],
                'Score': atm_data['score'],
                'Weight': '15%',
                'Confidence': f"{abs(atm_data['score']):.1f}%"
            })
        
        # Volume Analysis Component
        if st.session_state.vob_blocks:
            bullish_blocks = st.session_state.vob_blocks['bullish']
            bearish_blocks = st.session_state.vob_blocks['bearish']
            vob_score = len(bullish_blocks) - len(bearish_blocks)
            vob_bias = "BULLISH" if vob_score > 0 else "BEARISH" if vob_score < 0 else "NEUTRAL"
            
            components_data.append({
                'Component': 'Volume Order Blocks',
                'Bias': vob_bias,
                'Score': vob_score,
                'Weight': '15%',
                'Confidence': f"Blocks: {len(bullish_blocks)}B/{len(bearish_blocks)}S"
            })
        
        # Breakout/Reversal Component - NEW
        if st.session_state.market_bias_data:
            for instrument_data in st.session_state.market_bias_data:
                if instrument_data['instrument'] == 'NIFTY' and 'breakout_reversal_analysis' in instrument_data:
                    breakout_data = instrument_data['breakout_reversal_analysis']
                    breakout_score = breakout_data.get('overall_score', 0)
                    market_state = breakout_data.get('market_state', 'NEUTRAL_CHOPPY')
                    
                    components_data.append({
                        'Component': 'Breakout/Reversal',
                        'Bias': market_state.replace('_', ' ').title(),
                        'Score': breakout_score,
                        'Weight': '20%',
                        'Confidence': f"{breakout_score:.1f}%"
                    })
                    break
        
        if components_data:
            components_df = pd.DataFrame(components_data)
            st.dataframe(components_df, use_container_width=True)
        
        # Trading Recommendation
        st.subheader("📈 Trading Recommendation")
        
        if st.session_state.overall_nifty_bias == "BULLISH":
            st.success("""
            **Recommended Action:** Consider LONG positions
            - Look for buying opportunities on dips
            - Support levels are likely to hold
            - Target resistance levels for profit booking
            """)
        elif st.session_state.overall_nifty_bias == "BEARISH":
            st.error("""
            **Recommended Action:** Consider SHORT positions  
            - Look for selling opportunities on rallies
            - Resistance levels are likely to hold
            - Target support levels for profit booking
            """)
        else:
            st.warning("""
            **Recommended Action:** Wait for clearer direction
            - Market is in consolidation phase
            - Consider range-bound strategies
            - Wait for breakout confirmation
            """)

# BIAS SUMMARY TAB
with tabs[1]:
    st.subheader("Technical Bias Summary")
    if st.session_state['last_result'] is None:
        st.info("No analysis run yet. Analysis runs automatically every minute...")
    else:
        res = st.session_state['last_result']
        if not res.get('success', False):
            st.error(f"Analysis failed: {res.get('error')}")
        else:
            st.markdown(f"**Symbol:** `{res['symbol']}`")
            st.markdown(f"**Timestamp (IST):** {res['timestamp']}")
            st.metric("Current Price", f"{res['current_price']:.2f}")
            st.metric("Technical Bias", res['overall_bias'], delta=f"Confidence: {res['overall_confidence']:.1f}%")
            st.write("Mode:", res.get('mode', 'N/A'))

            # Show bias results table
            bias_table = pd.DataFrame(res['bias_results'])
            # Reorder columns for nicer view
            cols_order = ['indicator', 'value', 'bias', 'score', 'weight', 'category']
            bias_table = bias_table[cols_order]
            bias_table.columns = [c.capitalize() for c in bias_table.columns]
            st.subheader("Indicator-level Biases")
            st.dataframe(bias_table, use_container_width=True)

            # Summary stats
            st.write("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Bullish Indicators", int(res['bullish_count']))
            col2.metric("Bearish Indicators", int(res['bearish_count']))
            col3.metric("Neutral Indicators", int(res['neutral_count']))

            st.write("Weighted Bias Percentages:")
            st.write(f"- Fast Bull %: {res.get('fast_bull_pct', 0):.1f}%")
            st.write(f"- Fast Bear %: {res.get('fast_bear_pct', 0):.1f}%")
            st.write(f"- Slow Bull %: {res.get('slow_bull_pct', 0):.1f}%")
            st.write(f"- Slow Bear %: {res.get('slow_bear_pct', 0):.1f}%")
            st.write(f"- Bullish Bias % (weighted): {res.get('bullish_bias_pct', 0):.1f}%")
            st.write(f"- Bearish Bias % (weighted): {res.get('bearish_bias_pct', 0):.1f}%")

# PRICE ACTION TAB
with tabs[2]:
    st.header("📈 Price Action Analysis")
    
    if st.session_state['last_df'] is None:
        st.info("No data loaded yet. Analysis runs automatically every minute...")
    else:
        df = st.session_state['last_df']
        
        # Create price action chart with volume order blocks
        st.subheader("Price Chart with Volume Order Blocks")
        
        # FIX 5: Use stored VOB blocks
        bullish_blocks = st.session_state.vob_blocks.get('bullish', [])
        bearish_blocks = st.session_state.vob_blocks.get('bearish', [])
        
        # Create the chart using the plotting function
        fig = plot_vob(df, bullish_blocks, bearish_blocks)
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume Order Blocks Summary
        st.subheader("Volume Order Blocks Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bullish Blocks", len(bullish_blocks))
            if bullish_blocks:
                latest_bullish = bullish_blocks[-1]
                st.write(f"Latest Bullish Block:")
                st.write(f"- Upper: ₹{latest_bullish['upper']:.2f}")
                st.write(f"- Lower: ₹{latest_bullish['lower']:.2f}")
                st.write(f"- Volume: {latest_bullish['volume']:,.0f}")
        
        with col2:
            st.metric("Bearish Blocks", len(bearish_blocks))
            if bearish_blocks:
                latest_bearish = bearish_blocks[-1]
                st.write(f"Latest Bearish Block:")
                st.write(f"- Upper: ₹{latest_bearish['upper']:.2f}")
                st.write(f"- Lower: ₹{latest_bearish['lower']:.2f}")
                st.write(f"- Volume: {latest_bearish['volume']:,.0f}")

# OPTION CHAIN TAB - ENHANCED WITH BREAKOUT/REVERSAL ANALYSIS
with tabs[3]:
    st.header("📊 NSE Options Chain Analysis - Institutional Footprint")
    
    if st.session_state.last_bias_update:
        st.write(f"Last update: {st.session_state.last_bias_update.strftime('%H:%M:%S')} IST")
    
    if st.session_state.market_bias_data:
        bias_data = st.session_state.market_bias_data
        
        # Display current market bias for each instrument
        st.subheader("🎯 Current Market Bias Summary")
        cols = st.columns(len(bias_data))
        for idx, instrument_data in enumerate(bias_data):
            with cols[idx]:
                # Use combined bias if available, otherwise fallback to overall_bias
                bias_to_show = instrument_data.get('combined_bias', instrument_data.get('overall_bias', 'Neutral'))
                score_to_show = instrument_data.get('combined_score', instrument_data.get('bias_score', 0))
                
                bias_color = "🟢" if "Bullish" in bias_to_show else "🔴" if "Bearish" in bias_to_show else "🟡"
                st.metric(
                    f"{instrument_data['instrument']}",
                    f"{bias_color} {bias_to_show}",
                    f"Score: {score_to_show:.2f}"
                )
        
        # Enhanced detailed analysis for each instrument
        for instrument_data in bias_data:
            with st.expander(f"🏦 {instrument_data['instrument']} - Institutional Footprint Analysis", expanded=True):
                
                # Basic Information
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Spot Price", f"₹{instrument_data['spot_price']:.2f}")
                with col2:
                    st.metric("ATM Strike", f"₹{instrument_data['atm_strike']:.2f}")
                with col3:
                    st.metric("PCR OI", f"{instrument_data['pcr_oi']:.2f}")
                with col4:
                    st.metric("PCR Δ OI", f"{instrument_data['pcr_change']:.2f}")
                
                # BREAKOUT & REVERSAL ANALYSIS - NEW SECTION
                st.subheader("🔥 Breakout & Reversal Confirmation")
                
                if 'breakout_reversal_analysis' in instrument_data:
                    breakout_data = instrument_data['breakout_reversal_analysis']
                    
                    # Display breakout analysis
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Breakout Score", f"{breakout_data['breakout_analysis'].get('breakout_confidence', 0):.1f}%")
                    with col2:
                        st.metric("Reversal Score", f"{breakout_data['reversal_analysis'].get('reversal_confidence', 0):.1f}%")
                    with col3:
                        st.metric("Overall Score", f"{breakout_data['overall_score']:.1f}")
                    with col4:
                        signal = breakout_data['trading_signal']
                        signal_color = "🟢" if "BUY" in signal['action'] else "🔴" if "SELL" in signal['action'] else "🟡"
                        st.metric("Trading Signal", f"{signal_color} {signal['action']}")
                    
                    # Display market state
                    st.info(f"**Market State:** {breakout_data['market_state']} | **Message:** {signal['message']}")
                    
                    # Show breakout signals
                    if breakout_data['breakout_analysis'].get('signals'):
                        st.subheader("📈 Breakout Signals")
                        for signal in breakout_data['breakout_analysis']['signals']:
                            if "✅" in signal:
                                st.success(signal)
                            elif "❌" in signal:
                                st.error(signal)
                            else:
                                st.warning(signal)
                    
                    # Show reversal signals
                    if breakout_data['reversal_analysis'].get('signals'):
                        st.subheader("🔄 Reversal Signals")
                        for signal in breakout_data['reversal_analysis']['signals']:
                            if "✅" in signal:
                                st.success(signal)
                            elif "❌" in signal:
                                st.error(signal)
                            else:
                                st.warning(signal)
                else:
                    st.warning("Breakout/Reversal analysis not available for this instrument")
                
                # Institutional Analysis Section
                if 'institutional_analysis' in instrument_data:
                    inst_analysis = instrument_data['institutional_analysis']
                    
                    st.subheader("🔍 Institutional OI Patterns (ATM ±2 Strikes)")
                    
                    # Display patterns in a table
                    if inst_analysis.get('patterns'):
                        patterns_df = pd.DataFrame(inst_analysis['patterns'])
                        st.dataframe(patterns_df, use_container_width=True)
                    
                    # Gamma Analysis - COMPREHENSIVE
                    gamma_analysis = inst_analysis.get('gamma_analysis', {})
                    st.subheader("γ Gamma Sequencing Analysis")
                    
                    # Gamma Overview
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Gamma Bias", gamma_analysis.get('gamma_bias', 'NEUTRAL'))
                    with col2:
                        st.metric("Gamma Score", f"{gamma_analysis.get('gamma_score', 0):.1f}")
                    with col3:
                        st.metric("Gamma Profile", gamma_analysis.get('profile', 'Unknown'))
                    with col4:
                        st.metric("Total Gamma Exposure", f"{gamma_analysis.get('total_gamma_exposure', 0):.0f}")
                    
                    # Gamma Zones Analysis
                    if gamma_analysis.get('zones'):
                        st.subheader("🎯 Gamma Zones Analysis")
                        zones_data = []
                        for zone_name, zone_info in gamma_analysis['zones'].items():
                            zones_data.append({
                                'Zone': zone_name.replace('_', ' ').title(),
                                'Gamma Exposure': f"{zone_info['gamma_exposure']:.0f}",
                                'Bias': zone_info['bias'],
                                'Strike Range': zone_info['strike_range']
                            })
                        zones_df = pd.DataFrame(zones_data)
                        st.dataframe(zones_df, use_container_width=True)
                    
                    # Gamma Walls/Resistance Levels
                    if gamma_analysis.get('walls'):
                        st.subheader("🧱 Gamma Walls & Support Levels")
                        walls_data = []
                        for wall in gamma_analysis['walls']:
                            walls_data.append({
                                'Strike': f"₹{wall['strike']:.0f}",
                                'Gamma Exposure': f"{wall['gamma_exposure']:.0f}",
                                'Type': wall['type'],
                                'Strength': wall['strength']
                            })
                        walls_df = pd.DataFrame(walls_data)
                        st.dataframe(walls_df, use_container_width=True)
                    
                    # Gamma Sequences
                    if gamma_analysis.get('sequence'):
                        seq_analysis = gamma_analysis['sequence']
                        st.subheader("📈 Gamma Sequence Patterns")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Positive Sequences", len(seq_analysis.get('positive_sequences', [])))
                        with col2:
                            st.metric("Negative Sequences", len(seq_analysis.get('negative_sequences', [])))
                        with col3:
                            st.metric("Longest Positive", seq_analysis.get('longest_positive_seq', 0))
                        with col4:
                            st.metric("Longest Negative", seq_analysis.get('longest_negative_seq', 0))
                
                # Advanced Metrics
                st.subheader("📈 Advanced Option Metrics")
                comp_metrics = instrument_data.get('comprehensive_metrics', {})
                
                if comp_metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Synthetic Bias", comp_metrics.get('synthetic_bias', 'N/A'))
                    with col2:
                        st.metric("ATM Buildup", comp_metrics.get('atm_buildup', 'N/A'))
                    with col3:
                        st.metric("Max Pain", f"₹{comp_metrics.get('max_pain_strike', 'N/A')}")
                    with col4:
                        st.metric("Vega Bias", comp_metrics.get('total_vega_bias', 'N/A'))
                
                # Key Levels
                st.subheader("🎯 Key Trading Levels")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Call Resistance", f"₹{comp_metrics.get('call_resistance', 'N/A')}")
                with col2:
                    st.metric("Put Support", f"₹{comp_metrics.get('put_support', 'N/A')}")
                with col3:
                    st.metric("Distance from Max Pain", f"{comp_metrics.get('distance_from_max_pain', 0):.1f}")
                
                # Combined Bias Breakdown
                st.subheader("⚖️ Combined Bias Breakdown")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Basic Score", f"{instrument_data.get('bias_score', 0):.2f}")
                with col2:
                    st.metric("Institutional Score", f"{instrument_data.get('institutional_score', 0):.2f}")
                with col3:
                    st.metric("Gamma Score", f"{instrument_data.get('gamma_score', 0):.2f}")
                with col4:
                    st.metric("Breakout Score", f"{instrument_data.get('breakout_score', 0):.2f}")
    
    else:
        st.info("No option chain data available. Analysis runs automatically every minute...")

# BIAS TABULATION TAB - ENHANCED
with tabs[4]:
    st.header("📋 Comprehensive Bias Tabulation")
    
    if not st.session_state.market_bias_data:
        st.info("No option chain data available. Analysis runs automatically every minute...")
    else:
        for instrument_data in st.session_state.market_bias_data:
            with st.expander(f"🎯 {instrument_data['instrument']} - Complete Bias Analysis", expanded=True):
                
                # Basic Information Table
                st.subheader("📊 Basic Information")
                basic_info = pd.DataFrame({
                    'Metric': [
                        'Instrument', 'Spot Price', 'ATM Strike', 'Overall Bias', 
                        'Bias Score', 'PCR OI', 'PCR Change OI'
                    ],
                    'Value': [
                        instrument_data['instrument'],
                        f"₹{instrument_data['spot_price']:.2f}",
                        f"₹{instrument_data['atm_strike']:.2f}",
                        instrument_data['overall_bias'],
                        f"{instrument_data['bias_score']:.2f}",
                        f"{instrument_data['pcr_oi']:.2f}",
                        f"{instrument_data['pcr_change']:.2f}"
                    ]
                })
                st.dataframe(basic_info, use_container_width=True, hide_index=True)
                
                # ATM Detailed Bias Summary
                if 'detailed_atm_bias' in instrument_data and instrument_data['detailed_atm_bias']:
                    st.subheader("🎯 ATM Detailed Bias Analysis")
                    
                    detailed_bias = instrument_data['detailed_atm_bias']
                    
                    # Calculate overall ATM detailed bias
                    atm_bias, atm_score = calculate_atm_detailed_bias(detailed_bias)
                    
                    # Display ATM detailed bias summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        bias_color = "🟢" if atm_bias == "BULLISH" else "🔴" if atm_bias == "BEARISH" else "🟡"
                        st.metric("ATM Detailed Bias", f"{bias_color} {atm_bias}")
                    with col2:
                        st.metric("Bias Score", f"{atm_score:.1f}")
                    with col3:
                        st.metric("Total Metrics", f"{len([k for k in detailed_bias.keys() if 'Bias' in k])}")
                    
                    # Create comprehensive table for detailed bias
                    st.subheader("🔍 Detailed Bias Metrics Breakdown")
                    
                    bias_breakdown = []
                    for key, value in detailed_bias.items():
                        if 'Bias' in key:
                            bias_breakdown.append({
                                'Metric': key.replace('_', ' ').title(),
                                'Value': value,
                                'Score': 1 if value == 'Bullish' else -1 if value == 'Bearish' else 0
                            })
                    
                    if bias_breakdown:
                        breakdown_df = pd.DataFrame(bias_breakdown)
                        
                        # Add color formatting
                        def color_bias(val):
                            if val == 'Bullish':
                                return 'color: green; font-weight: bold'
                            elif val == 'Bearish':
                                return 'color: red; font-weight: bold'
                            else:
                                return 'color: orange; font-weight: bold'
                        
                        styled_df = breakdown_df.style.applymap(color_bias, subset=['Value'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Bias distribution
                        st.subheader("📊 Bias Distribution")
                        bull_count = len([b for b in bias_breakdown if b['Value'] == 'Bullish'])
                        bear_count = len([b for b in bias_breakdown if b['Value'] == 'Bearish'])
                        neutral_count = len([b for b in bias_breakdown if b['Value'] == 'Neutral'])
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Bullish Metrics", bull_count)
                        col2.metric("Bearish Metrics", bear_count)
                        col3.metric("Neutral Metrics", neutral_count)
                        
                        # Create bias distribution chart
                        fig = px.pie(
                            names=['Bullish', 'Bearish', 'Neutral'],
                            values=[bull_count, bear_count, neutral_count],
                            title="ATM Bias Distribution",
                            color=['Bullish', 'Bearish', 'Neutral'],
                            color_discrete_map={'Bullish': '#00ff88', 'Bearish': '#ff4444', 'Neutral': '#ffaa00'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Breakout/Reversal Analysis - NEW SECTION
                if 'breakout_reversal_analysis' in instrument_data:
                    st.subheader("🔥 Breakout & Reversal Analysis")
                    
                    breakout_data = instrument_data['breakout_reversal_analysis']
                    
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Market State", breakout_data['market_state'])
                    with col2:
                        st.metric("Overall Score", f"{breakout_data['overall_score']:.1f}")
                    with col3:
                        signal = breakout_data['trading_signal']
                        st.metric("Trading Signal", signal['action'])
                    with col4:
                        st.metric("Confidence", signal['confidence'])
                    
                    # Display breakout analysis details
                    st.subheader("📈 Breakout Analysis Details")
                    breakout_analysis = breakout_data['breakout_analysis']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Breakout Confidence", f"{breakout_analysis.get('breakout_confidence', 0):.1f}%")
                    with col2:
                        st.metric("Direction", breakout_analysis.get('direction', 'UNKNOWN'))
                    with col3:
                        st.metric("Breakout Type", breakout_analysis.get('breakout_type', 'UNKNOWN'))
                    
                    # Display reversal analysis details
                    st.subheader("🔄 Reversal Analysis Details")
                    reversal_analysis = breakout_data['reversal_analysis']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Reversal Confidence", f"{reversal_analysis.get('reversal_confidence', 0):.1f}%")
                    with col2:
                        st.metric("Direction", reversal_analysis.get('direction', 'UNKNOWN'))
                    with col3:
                        st.metric("Reversal Type", reversal_analysis.get('reversal_type', 'UNKNOWN'))
                
                # Comprehensive Metrics Table
                if 'comprehensive_metrics' in instrument_data and instrument_data['comprehensive_metrics']:
                    st.subheader("🎯 Advanced Option Metrics")
                    comp_metrics = instrument_data['comprehensive_metrics']
                    
                    comp_data = []
                    for key, value in comp_metrics.items():
                        if key not in ['total_vega', 'total_ce_vega_exp', 'total_pe_vega_exp']:
                            comp_data.append([
                                key.replace('_', ' ').title(),
                                str(value) if not isinstance(value, (int, float)) else f"{value:.2f}"
                            ])
                    
                    comp_df = pd.DataFrame(comp_data, columns=['Metric', 'Value'])
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.caption("BiasAnalysisPro — Complete Enhanced Dashboard with Auto-Refresh, Overall Nifty Bias Analysis, and Institutional Breakout/Reversal Detection.")
st.caption("🔔 Telegram alerts sent when Technical Analysis, Options Chain, and ATM Detailed Bias are aligned (Bullish/Bearish)")
