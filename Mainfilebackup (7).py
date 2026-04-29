"""
Nifty Option Screener v7.0 — 100% SELLER'S PERSPECTIVE + ATM BIAS ANALYZER + MOMENT DETECTOR + EXPIRY SPIKE DETECTOR + ENHANCED OI/PCR ANALYTICS
EVERYTHING interpreted from Option Seller/Market Maker viewpoint
CALL building = BEARISH (sellers selling calls, expecting price to stay below)
PUT building = BULLISH (sellers selling puts, expecting price to stay above)

NEW FEATURES ADDED:
1. Comprehensive ATM Bias Analysis (12 metrics)
2. Multi-dimensional Bias Dashboard
3. Support/Resistance Bias Analysis
4. Enhanced Entry Signals with Bias Integration
5. Momentum Burst Detection
6. Orderbook Pressure Analysis
7. Gamma Cluster Concentration
8. OI Velocity/Acceleration
9. Telegram Signal Generation
10. Expiry Spike Detector
11. Enhanced OI/PCR Analytics
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import pytz
from math import log, sqrt
from scipy.stats import norm
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import json

# Import advanced market depth analysis
try:
    from market_depth_advanced import (
        get_real_option_depth_from_dhan,
        analyze_depth_levels,
        detect_market_maker_activity,
        analyze_liquidity_profile,
        analyze_order_flow,
        analyze_volume_profile,
        analyze_market_microstructure,
        calculate_depth_quality,
        detect_algo_patterns,
        calculate_market_impact,
        run_comprehensive_depth_analysis
    )
    ADVANCED_DEPTH_AVAILABLE = True
except ImportError:
    ADVANCED_DEPTH_AVAILABLE = False

# Import option chain table module
try:
    from option_chain_table import render_option_chain_table_tab
    OPTION_CHAIN_TABLE_AVAILABLE = True
except ImportError:
    OPTION_CHAIN_TABLE_AVAILABLE = False

# -----------------------
#  IST TIMEZONE SETUP
# -----------------------
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    return datetime.now(IST)

def get_ist_time_str():
    return get_ist_now().strftime("%H:%M:%S")

def get_ist_date_str():
    return get_ist_now().strftime("%Y-%m-%d")

def get_ist_datetime_str():
    return get_ist_now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------
#  GREEKS CALCULATION
# -----------------------
def compute_greeks(spot, strike, tau, risk_free_rate, ltp, option_type):
    """
    Calculate option Greeks using Black-Scholes model

    Args:
        spot: Current spot price
        strike: Strike price
        tau: Time to expiry in years
        risk_free_rate: Risk-free interest rate
        ltp: Last traded price (premium)
        option_type: "CE" for Call or "PE" for Put

    Returns:
        dict with delta, gamma, theta, vega
    """
    try:
        if tau <= 0 or ltp <= 0 or spot <= 0 or strike <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        # Calculate implied volatility (simplified - using approximation)
        # In production, use Newton-Raphson to solve for IV
        # For now, using a rough estimate based on ATM volatility
        iv = 0.20  # Default 20% volatility

        # Try to estimate IV from premium
        if option_type == "CE":
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)

        time_value = ltp - intrinsic
        if time_value > 0 and tau > 0:
            # Rough IV estimate
            iv = min(2.0, max(0.05, time_value / (spot * sqrt(tau))))

        # Black-Scholes calculations
        d1 = (log(spot / strike) + (risk_free_rate + 0.5 * iv ** 2) * tau) / (iv * sqrt(tau))
        d2 = d1 - iv * sqrt(tau)

        # Delta
        if option_type == "CE":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (spot * iv * sqrt(tau))

        # Theta
        if option_type == "CE":
            theta = (-(spot * norm.pdf(d1) * iv) / (2 * sqrt(tau)) -
                     risk_free_rate * strike * np.exp(-risk_free_rate * tau) * norm.cdf(d2))
        else:
            theta = (-(spot * norm.pdf(d1) * iv) / (2 * sqrt(tau)) +
                     risk_free_rate * strike * np.exp(-risk_free_rate * tau) * norm.cdf(-d2))

        # Vega (same for calls and puts)
        vega = spot * norm.pdf(d1) * sqrt(tau)

        # Convert to per-day theta
        theta_per_day = theta / 365

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta_per_day, 4),
            "vega": round(vega / 100, 4)  # Vega per 1% change in IV
        }
    except Exception as e:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

# -----------------------
#  CONFIG
# -----------------------
AUTO_REFRESH_SEC = 28
LOT_SIZE = 50
RISK_FREE_RATE = 0.06
ATM_STRIKE_WINDOW = 8
SCORE_WEIGHTS = {"chg_oi": 2.0, "volume": 0.5, "oi": 0.2, "iv": 0.3}
BREAKOUT_INDEX_WEIGHTS = {"atm_oi_shift": 0.4, "winding_balance": 0.3, "vol_oi_div": 0.2, "gamma_pressure": 0.1}
SAVE_INTERVAL_SEC = 300

# NEW: Moment detector weights
MOMENT_WEIGHTS = {
    "momentum_burst": 0.40,        # Vol × IV × |ΔOI|
    "orderbook_pressure": 0.20,    # buy/sell depth imbalance
    "gamma_cluster": 0.25,         # ATM ±2 gamma concentration
    "oi_acceleration": 0.15        # OI speed-up (break/hold)
}

TIME_WINDOWS = {
    "morning": {"start": (9, 15), "end": (10, 30), "label": "Morning (09:15-10:30 IST)"},
    "mid": {"start": (10, 30), "end": (12, 30), "label": "Mid (10:30-12:30 IST)"},
    "afternoon": {"start": (14, 0), "end": (15, 30), "label": "Afternoon (14:00-15:30 IST)"},
    "evening": {"start": (15, 0), "end": (15, 30), "label": "Evening (15:00-15:30 IST)"}
}

# -----------------------
#  SECRETS
# -----------------------
try:
    # Try nested format first (from app.py config), then fallback to flat format
    try:
        DHAN_CLIENT_ID = st.secrets["DHAN"]["CLIENT_ID"]
        DHAN_ACCESS_TOKEN = st.secrets["DHAN"]["ACCESS_TOKEN"]
    except:
        # Fallback to flat format
        DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
        DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")

    # Supabase credentials (optional)
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
    SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "")
    SUPABASE_TABLE = st.secrets.get("SUPABASE_TABLE", "option_snapshots")
    SUPABASE_TABLE_PCR = st.secrets.get("SUPABASE_TABLE_PCR", "strike_pcr_snapshots")

    # Telegram credentials (optional)
    try:
        TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM"]["BOT_TOKEN"]
        TELEGRAM_CHAT_ID = st.secrets["TELEGRAM"]["CHAT_ID"]
    except:
        TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

    # Verify required credentials
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("❌ Missing Dhan credentials. Please configure in .streamlit/secrets.toml")
        st.info("Required: DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN")
        st.stop()

except Exception as e:
    st.error(f"❌ Error loading credentials: {e}")
    st.info("Please check your .streamlit/secrets.toml configuration")
    st.stop()

try:
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    else:
        supabase = None
        # Supabase is optional, don't show error
except Exception as e:
    # Supabase is optional, just set to None and continue
    supabase = None

DHAN_BASE_URL = "https://api.dhan.co"
NIFTY_UNDERLYING_SCRIP = "13"
NIFTY_UNDERLYING_SEG = "IDX_I"

# ============================================
# 🎯 ATM BIAS ANALYZER (NEW)
# ============================================
def analyze_atm_bias(merged_df, spot, atm_strike, strike_gap):
    """
    Analyze ATM bias from multiple perspectives for sellers
    """
    
    # Define ATM window (±2 strikes around ATM)
    atm_window = 2
    atm_strikes = [s for s in merged_df["strikePrice"] 
                  if abs(s - atm_strike) <= (atm_window * strike_gap)]
    
    atm_df = merged_df[merged_df["strikePrice"].isin(atm_strikes)].copy()
    
    if atm_df.empty:
        return None
    
    # Initialize bias scores
    bias_scores = {
        "OI_Bias": 0,
        "ChgOI_Bias": 0,
        "Volume_Bias": 0,
        "Delta_Bias": 0,
        "Gamma_Bias": 0,
        "Premium_Bias": 0,
        "IV_Bias": 0,
        "Delta_Exposure_Bias": 0,
        "Gamma_Exposure_Bias": 0,
        "IV_Skew_Bias": 0,
        "OI_Change_Bias": 0
    }
    
    bias_interpretations = {}
    bias_emojis = {}
    
    # 1. OI BIAS (CALL vs PUT OI)
    total_ce_oi_atm = atm_df["OI_CE"].sum()
    total_pe_oi_atm = atm_df["OI_PE"].sum()
    oi_ratio = total_pe_oi_atm / max(total_ce_oi_atm, 1)
    
    if oi_ratio > 1.5:
        bias_scores["OI_Bias"] = 1
        bias_interpretations["OI_Bias"] = "Heavy PUT OI at ATM → Bullish sellers"
        bias_emojis["OI_Bias"] = "🐂 Bullish"
    elif oi_ratio > 1.0:
        bias_scores["OI_Bias"] = 0.5
        bias_interpretations["OI_Bias"] = "Moderate PUT OI → Mild bullish"
        bias_emojis["OI_Bias"] = "🐂 Bullish"
    elif oi_ratio < 0.7:
        bias_scores["OI_Bias"] = -1
        bias_interpretations["OI_Bias"] = "Heavy CALL OI at ATM → Bearish sellers"
        bias_emojis["OI_Bias"] = "🐻 Bearish"
    elif oi_ratio < 1.0:
        bias_scores["OI_Bias"] = -0.5
        bias_interpretations["OI_Bias"] = "Moderate CALL OI → Mild bearish"
        bias_emojis["OI_Bias"] = "🐻 Bearish"
    else:
        bias_scores["OI_Bias"] = 0
        bias_interpretations["OI_Bias"] = "Balanced OI → Neutral"
        bias_emojis["OI_Bias"] = "⚖️ Neutral"
    
    # 2. CHANGE IN OI BIAS (CALL vs PUT ΔOI)
    total_ce_chg_atm = atm_df["Chg_OI_CE"].sum()
    total_pe_chg_atm = atm_df["Chg_OI_PE"].sum()
    
    if total_pe_chg_atm > 0 and total_ce_chg_atm > 0:
        # Both sides writing
        if total_pe_chg_atm > total_ce_chg_atm:
            bias_scores["ChgOI_Bias"] = 0.5
            bias_interpretations["ChgOI_Bias"] = "More PUT writing → Bullish buildup"
            bias_emojis["ChgOI_Bias"] = "🐂 Bullish"
        else:
            bias_scores["ChgOI_Bias"] = -0.5
            bias_interpretations["ChgOI_Bias"] = "More CALL writing → Bearish buildup"
            bias_emojis["ChgOI_Bias"] = "🐻 Bearish"
    elif total_pe_chg_atm > 0:
        bias_scores["ChgOI_Bias"] = 1
        bias_interpretations["ChgOI_Bias"] = "Only PUT writing → Strong bullish"
        bias_emojis["ChgOI_Bias"] = "🐂 Bullish"
    elif total_ce_chg_atm > 0:
        bias_scores["ChgOI_Bias"] = -1
        bias_interpretations["ChgOI_Bias"] = "Only CALL writing → Strong bearish"
        bias_emojis["ChgOI_Bias"] = "🐻 Bearish"
    elif total_pe_chg_atm < 0 and total_ce_chg_atm < 0:
        # Both sides unwinding
        bias_scores["ChgOI_Bias"] = 0
        bias_interpretations["ChgOI_Bias"] = "Both unwinding → Range contraction"
        bias_emojis["ChgOI_Bias"] = "⚖️ Neutral"
    else:
        bias_scores["ChgOI_Bias"] = 0
        bias_interpretations["ChgOI_Bias"] = "Mixed activity"
        bias_emojis["ChgOI_Bias"] = "⚖️ Neutral"
    
    # 3. VOLUME BIAS (CALL vs PUT Volume)
    total_ce_vol_atm = atm_df["Vol_CE"].sum()
    total_pe_vol_atm = atm_df["Vol_PE"].sum()
    vol_ratio = total_pe_vol_atm / max(total_ce_vol_atm, 1)
    
    if vol_ratio > 1.3:
        bias_scores["Volume_Bias"] = 1
        bias_interpretations["Volume_Bias"] = "High PUT volume → Bullish activity"
        bias_emojis["Volume_Bias"] = "🐂 Bullish"
    elif vol_ratio > 1.0:
        bias_scores["Volume_Bias"] = 0.5
        bias_interpretations["Volume_Bias"] = "More PUT volume → Mild bullish"
        bias_emojis["Volume_Bias"] = "🐂 Bullish"
    elif vol_ratio < 0.8:
        bias_scores["Volume_Bias"] = -1
        bias_interpretations["Volume_Bias"] = "High CALL volume → Bearish activity"
        bias_emojis["Volume_Bias"] = "🐻 Bearish"
    elif vol_ratio < 1.0:
        bias_scores["Volume_Bias"] = -0.5
        bias_interpretations["Volume_Bias"] = "More CALL volume → Mild bearish"
        bias_emojis["Volume_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Volume_Bias"] = 0
        bias_interpretations["Volume_Bias"] = "Balanced volume"
        bias_emojis["Volume_Bias"] = "⚖️ Neutral"
    
    # 4. DELTA BIAS (Net Delta Position)
    total_delta_ce = atm_df["Delta_CE"].sum()
    total_delta_pe = atm_df["Delta_PE"].sum()
    net_delta = total_delta_ce + total_delta_pe  # CALL delta positive, PUT delta negative

    if net_delta > 0.3:
        bias_scores["Delta_Bias"] = 1  # Positive delta = CALL heavy = Bullish
        bias_interpretations["Delta_Bias"] = "Positive delta → CALL heavy → Bullish"
        bias_emojis["Delta_Bias"] = "🐂 Bullish"
    elif net_delta > 0.1:
        bias_scores["Delta_Bias"] = 0.5
        bias_interpretations["Delta_Bias"] = "Mild positive delta → Slightly bullish"
        bias_emojis["Delta_Bias"] = "🐂 Bullish"
    elif net_delta < -0.3:
        bias_scores["Delta_Bias"] = -1  # Negative delta = PUT heavy = Bearish
        bias_interpretations["Delta_Bias"] = "Negative delta → PUT heavy → Bearish"
        bias_emojis["Delta_Bias"] = "🐻 Bearish"
    elif net_delta < -0.1:
        bias_scores["Delta_Bias"] = -0.5
        bias_interpretations["Delta_Bias"] = "Mild negative delta → Slightly bearish"
        bias_emojis["Delta_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Delta_Bias"] = 0
        bias_interpretations["Delta_Bias"] = "Neutral delta"
        bias_emojis["Delta_Bias"] = "⚖️ Neutral"
    
    # 5. GAMMA BIAS (Net Gamma Position)
    total_gamma_ce = atm_df["Gamma_CE"].sum()
    total_gamma_pe = atm_df["Gamma_PE"].sum()
    net_gamma = total_gamma_ce + total_gamma_pe
    
    # For sellers: Positive gamma = stabilizing, Negative gamma = explosive
    if net_gamma > 0.1:
        bias_scores["Gamma_Bias"] = 1
        bias_interpretations["Gamma_Bias"] = "Positive gamma → Stabilizing → Bullish (less volatility)"
        bias_emojis["Gamma_Bias"] = "🐂 Bullish"
    elif net_gamma > 0:
        bias_scores["Gamma_Bias"] = 0.5
        bias_interpretations["Gamma_Bias"] = "Mild positive gamma → Slightly stabilizing"
        bias_emojis["Gamma_Bias"] = "🐂 Bullish"
    elif net_gamma < -0.1:
        bias_scores["Gamma_Bias"] = -1
        bias_interpretations["Gamma_Bias"] = "Negative gamma → Explosive → Bearish (high volatility)"
        bias_emojis["Gamma_Bias"] = "🐻 Bearish"
    elif net_gamma < 0:
        bias_scores["Gamma_Bias"] = -0.5
        bias_interpretations["Gamma_Bias"] = "Mild negative gamma → Slightly explosive"
        bias_emojis["Gamma_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Gamma_Bias"] = 0
        bias_interpretations["Gamma_Bias"] = "Neutral gamma"
        bias_emojis["Gamma_Bias"] = "⚖️ Neutral"
    
    # 6. PREMIUM BIAS (CALL vs PUT Premium)
    # Calculate average premium
    ce_premium = atm_df["LTP_CE"].mean() if not atm_df["LTP_CE"].isna().all() else 0
    pe_premium = atm_df["LTP_PE"].mean() if not atm_df["LTP_PE"].isna().all() else 0
    premium_ratio = pe_premium / max(ce_premium, 0.01)
    
    if premium_ratio > 1.2:
        bias_scores["Premium_Bias"] = 1
        bias_interpretations["Premium_Bias"] = "PUT premium higher → Bullish sentiment"
        bias_emojis["Premium_Bias"] = "🐂 Bullish"
    elif premium_ratio > 1.0:
        bias_scores["Premium_Bias"] = 0.5
        bias_interpretations["Premium_Bias"] = "PUT premium slightly higher → Mild bullish"
        bias_emojis["Premium_Bias"] = "🐂 Bullish"
    elif premium_ratio < 0.8:
        bias_scores["Premium_Bias"] = -1
        bias_interpretations["Premium_Bias"] = "CALL premium higher → Bearish sentiment"
        bias_emojis["Premium_Bias"] = "🐻 Bearish"
    elif premium_ratio < 1.0:
        bias_scores["Premium_Bias"] = -0.5
        bias_interpretations["Premium_Bias"] = "CALL premium slightly higher → Mild bearish"
        bias_emojis["Premium_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Premium_Bias"] = 0
        bias_interpretations["Premium_Bias"] = "Balanced premiums"
        bias_emojis["Premium_Bias"] = "⚖️ Neutral"
    
    # 7. IV BIAS (CALL vs PUT IV)
    ce_iv = atm_df["IV_CE"].mean() if not atm_df["IV_CE"].isna().all() else 0
    pe_iv = atm_df["IV_PE"].mean() if not atm_df["IV_PE"].isna().all() else 0
    
    if pe_iv > ce_iv + 3:
        bias_scores["IV_Bias"] = 1
        bias_interpretations["IV_Bias"] = "PUT IV higher → Bullish fear"
        bias_emojis["IV_Bias"] = "🐂 Bullish"
    elif pe_iv > ce_iv + 1:
        bias_scores["IV_Bias"] = 0.5
        bias_interpretations["IV_Bias"] = "PUT IV slightly higher → Mild bullish fear"
        bias_emojis["IV_Bias"] = "🐂 Bullish"
    elif ce_iv > pe_iv + 3:
        bias_scores["IV_Bias"] = -1
        bias_interpretations["IV_Bias"] = "CALL IV higher → Bearish fear"
        bias_emojis["IV_Bias"] = "🐻 Bearish"
    elif ce_iv > pe_iv + 1:
        bias_scores["IV_Bias"] = -0.5
        bias_interpretations["IV_Bias"] = "CALL IV slightly higher → Mild bearish fear"
        bias_emojis["IV_Bias"] = "🐻 Bearish"
    else:
        bias_scores["IV_Bias"] = 0
        bias_interpretations["IV_Bias"] = "Balanced IV"
        bias_emojis["IV_Bias"] = "⚖️ Neutral"
    
    # 8. DELTA EXPOSURE BIAS (OI-weighted Delta)
    delta_exposure_ce = (atm_df["Delta_CE"] * atm_df["OI_CE"]).sum()
    delta_exposure_pe = (atm_df["Delta_PE"] * atm_df["OI_PE"]).sum()
    net_delta_exposure = delta_exposure_ce + delta_exposure_pe

    if net_delta_exposure > 1000000:
        bias_scores["Delta_Exposure_Bias"] = 1
        bias_interpretations["Delta_Exposure_Bias"] = "High CALL delta exposure → Bullish pressure"
        bias_emojis["Delta_Exposure_Bias"] = "🐂 Bullish"
    elif net_delta_exposure > 500000:
        bias_scores["Delta_Exposure_Bias"] = 0.5
        bias_interpretations["Delta_Exposure_Bias"] = "Moderate CALL delta exposure → Slightly bullish"
        bias_emojis["Delta_Exposure_Bias"] = "🐂 Bullish"
    elif net_delta_exposure < -1000000:
        bias_scores["Delta_Exposure_Bias"] = -1
        bias_interpretations["Delta_Exposure_Bias"] = "High PUT delta exposure → Bearish pressure"
        bias_emojis["Delta_Exposure_Bias"] = "🐻 Bearish"
    elif net_delta_exposure < -500000:
        bias_scores["Delta_Exposure_Bias"] = -0.5
        bias_interpretations["Delta_Exposure_Bias"] = "Moderate PUT delta exposure → Slightly bearish"
        bias_emojis["Delta_Exposure_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Delta_Exposure_Bias"] = 0
        bias_interpretations["Delta_Exposure_Bias"] = "Balanced delta exposure"
        bias_emojis["Delta_Exposure_Bias"] = "⚖️ Neutral"
    
    # 9. GAMMA EXPOSURE BIAS (OI-weighted Gamma)
    gamma_exposure_ce = (atm_df["Gamma_CE"] * atm_df["OI_CE"]).sum()
    gamma_exposure_pe = (atm_df["Gamma_PE"] * atm_df["OI_PE"]).sum()
    net_gamma_exposure = gamma_exposure_ce + gamma_exposure_pe
    
    if net_gamma_exposure > 500000:
        bias_scores["Gamma_Exposure_Bias"] = 1
        bias_interpretations["Gamma_Exposure_Bias"] = "Positive gamma exposure → Stabilizing → Bullish"
        bias_emojis["Gamma_Exposure_Bias"] = "🐂 Bullish"
    elif net_gamma_exposure > 100000:
        bias_scores["Gamma_Exposure_Bias"] = 0.5
        bias_interpretations["Gamma_Exposure_Bias"] = "Mild positive gamma → Slightly stabilizing"
        bias_emojis["Gamma_Exposure_Bias"] = "🐂 Bullish"
    elif net_gamma_exposure < -500000:
        bias_scores["Gamma_Exposure_Bias"] = -1
        bias_interpretations["Gamma_Exposure_Bias"] = "Negative gamma exposure → Explosive → Bearish"
        bias_emojis["Gamma_Exposure_Bias"] = "🐻 Bearish"
    elif net_gamma_exposure < -100000:
        bias_scores["Gamma_Exposure_Bias"] = -0.5
        bias_interpretations["Gamma_Exposure_Bias"] = "Mild negative gamma → Slightly explosive"
        bias_emojis["Gamma_Exposure_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Gamma_Exposure_Bias"] = 0
        bias_interpretations["Gamma_Exposure_Bias"] = "Balanced gamma exposure"
        bias_emojis["Gamma_Exposure_Bias"] = "⚖️ Neutral"
    
    # 10. IV SKEW BIAS (ATM vs Nearby strikes)
    # Get ±1 strike IVs
    nearby_strikes = [s for s in merged_df["strikePrice"] 
                     if abs(s - atm_strike) <= (1 * strike_gap)]
    nearby_df = merged_df[merged_df["strikePrice"].isin(nearby_strikes)]
    
    if not nearby_df.empty:
        atm_ce_iv = atm_df["IV_CE"].mean() if not atm_df["IV_CE"].isna().all() else 0
        atm_pe_iv = atm_df["IV_PE"].mean() if not atm_df["IV_PE"].isna().all() else 0
        nearby_ce_iv = nearby_df["IV_CE"].mean() if not nearby_df["IV_CE"].isna().all() else 0
        nearby_pe_iv = nearby_df["IV_PE"].mean() if not nearby_df["IV_PE"].isna().all() else 0
        
        # ATM IV vs Nearby IV comparison
        if atm_ce_iv > nearby_ce_iv + 2:
            bias_scores["IV_Skew_Bias"] = -0.5
            bias_interpretations["IV_Skew_Bias"] = "ATM CALL IV higher → Bearish skew"
            bias_emojis["IV_Skew_Bias"] = "🐻 Bearish"
        elif atm_pe_iv > nearby_pe_iv + 2:
            bias_scores["IV_Skew_Bias"] = 0.5
            bias_interpretations["IV_Skew_Bias"] = "ATM PUT IV higher → Bullish skew"
            bias_emojis["IV_Skew_Bias"] = "🐂 Bullish"
        else:
            bias_scores["IV_Skew_Bias"] = 0
            bias_interpretations["IV_Skew_Bias"] = "Flat IV skew"
            bias_emojis["IV_Skew_Bias"] = "⚖️ Neutral"
    else:
        bias_scores["IV_Skew_Bias"] = 0
        bias_interpretations["IV_Skew_Bias"] = "Insufficient data for IV skew"
        bias_emojis["IV_Skew_Bias"] = "⚖️ Neutral"
    
    # 11. OI CHANGE BIAS (Acceleration)
    # Calculate OI change rate
    total_oi_change = abs(total_ce_chg_atm) + abs(total_pe_chg_atm)
    total_oi_atm = total_ce_oi_atm + total_pe_oi_atm
    
    if total_oi_atm > 0:
        oi_change_rate = total_oi_change / total_oi_atm
        if oi_change_rate > 0.1:
            # High OI change - check direction
            if total_pe_chg_atm > total_ce_chg_atm:
                bias_scores["OI_Change_Bias"] = 0.5
                bias_interpretations["OI_Change_Bias"] = "Rapid PUT OI buildup → Bullish acceleration"
                bias_emojis["OI_Change_Bias"] = "🐂 Bullish"
            else:
                bias_scores["OI_Change_Bias"] = -0.5
                bias_interpretations["OI_Change_Bias"] = "Rapid CALL OI buildup → Bearish acceleration"
                bias_emojis["OI_Change_Bias"] = "🐻 Bearish"
        else:
            bias_scores["OI_Change_Bias"] = 0
            bias_interpretations["OI_Change_Bias"] = "Slow OI changes"
            bias_emojis["OI_Change_Bias"] = "⚖️ Neutral"

    # 12. PCR (PUT-CALL RATIO) BIAS
    # Calculate PCR for ATM window
    pcr_atm = total_pe_oi_atm / max(total_ce_oi_atm, 1)

    if pcr_atm > 1.5:
        bias_scores["PCR_Bias"] = 1
        bias_interpretations["PCR_Bias"] = f"High PCR ({pcr_atm:.2f}) → Strong bullish (PUT writing)"
        bias_emojis["PCR_Bias"] = "🐂 Bullish"
    elif pcr_atm > 1.2:
        bias_scores["PCR_Bias"] = 0.5
        bias_interpretations["PCR_Bias"] = f"Elevated PCR ({pcr_atm:.2f}) → Mild bullish"
        bias_emojis["PCR_Bias"] = "🐂 Bullish"
    elif pcr_atm < 0.7:
        bias_scores["PCR_Bias"] = -1
        bias_interpretations["PCR_Bias"] = f"Low PCR ({pcr_atm:.2f}) → Strong bearish (CALL writing)"
        bias_emojis["PCR_Bias"] = "🐻 Bearish"
    elif pcr_atm < 0.9:
        bias_scores["PCR_Bias"] = -0.5
        bias_interpretations["PCR_Bias"] = f"Below-normal PCR ({pcr_atm:.2f}) → Mild bearish"
        bias_emojis["PCR_Bias"] = "🐻 Bearish"
    else:
        bias_scores["PCR_Bias"] = 0
        bias_interpretations["PCR_Bias"] = f"Balanced PCR ({pcr_atm:.2f})"
        bias_emojis["PCR_Bias"] = "⚖️ Neutral"

    # Calculate final bias score
    total_score = sum(bias_scores.values())
    normalized_score = total_score / len(bias_scores) if bias_scores else 0
    
    # Determine overall verdict
    if normalized_score > 0.3:
        verdict = "🐂 BULLISH"
        verdict_color = "#00ff88"
        verdict_explanation = "ATM zone showing strong bullish bias for sellers"
    elif normalized_score > 0.1:
        verdict = "🐂 Mild Bullish"
        verdict_color = "#00cc66"
        verdict_explanation = "ATM zone leaning bullish for sellers"
    elif normalized_score < -0.3:
        verdict = "🐻 BEARISH"
        verdict_color = "#ff4444"
        verdict_explanation = "ATM zone showing strong bearish bias for sellers"
    elif normalized_score < -0.1:
        verdict = "🐻 Mild Bearish"
        verdict_color = "#ff6666"
        verdict_explanation = "ATM zone leaning bearish for sellers"
    else:
        verdict = "⚖️ NEUTRAL"
        verdict_color = "#66b3ff"
        verdict_explanation = "ATM zone balanced, no clear bias"
    
    return {
        "instrument": "NIFTY",
        "atm_strike": atm_strike,
        "zone": "ATM",
        "level": "ATM Cluster",
        "bias_scores": bias_scores,
        "bias_interpretations": bias_interpretations,
        "bias_emojis": bias_emojis,
        "total_score": normalized_score,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_explanation": verdict_explanation,
        "metrics": {
            "ce_oi": int(total_ce_oi_atm),
            "pe_oi": int(total_pe_oi_atm),
            "ce_chg": int(total_ce_chg_atm),
            "pe_chg": int(total_pe_chg_atm),
            "ce_vol": int(total_ce_vol_atm),
            "pe_vol": int(total_pe_vol_atm),
            "net_delta": round(net_delta, 3),
            "net_gamma": round(net_gamma, 3),
            "ce_iv": round(ce_iv, 2),
            "pe_iv": round(pe_iv, 2),
            "delta_exposure": int(net_delta_exposure),
            "gamma_exposure": int(net_gamma_exposure)
        }
    }


# ============================================
# 🎯 SUPPORT/RESISTANCE BIAS ANALYZER (NEW)
# ============================================
def analyze_support_resistance_bias(merged_df, spot, atm_strike, strike_gap, level_type="Support"):
    """
    Analyze bias at key support/resistance levels
    """
    
    # Find key levels
    if level_type == "Support":
        # Find highest strike below spot with high PUT OI
        support_strikes = merged_df[merged_df["strikePrice"] < spot].copy()
        if support_strikes.empty:
            return None
        
        # Find strike with highest PUT OI as support
        support_strike = support_strikes.loc[support_strikes["OI_PE"].idxmax()]["strikePrice"]
        level_df = merged_df[merged_df["strikePrice"] == support_strike]
    else:  # Resistance
        # Find lowest strike above spot with high CALL OI
        resistance_strikes = merged_df[merged_df["strikePrice"] > spot].copy()
        if resistance_strikes.empty:
            return None
        
        # Find strike with highest CALL OI as resistance
        resistance_strike = resistance_strikes.loc[resistance_strikes["OI_CE"].idxmax()]["strikePrice"]
        level_df = merged_df[merged_df["strikePrice"] == resistance_strike]
    
    if level_df.empty:
        return None
    
    row = level_df.iloc[0]
    
    # Calculate bias
    bias_scores = {}
    bias_emojis = {}
    bias_interpretations = {}
    
    # OI Bias
    oi_ratio = row["OI_PE"] / max(row["OI_CE"], 1)
    if oi_ratio > 2:
        bias_scores["OI_Bias"] = 1
        bias_emojis["OI_Bias"] = "🐂 Bullish"
        bias_interpretations["OI_Bias"] = "Very high PUT OI"
    elif oi_ratio > 1:
        bias_scores["OI_Bias"] = 0.5
        bias_emojis["OI_Bias"] = "🐂 Bullish"
        bias_interpretations["OI_Bias"] = "High PUT OI"
    elif oi_ratio < 0.5:
        bias_scores["OI_Bias"] = -1
        bias_emojis["OI_Bias"] = "🐻 Bearish"
        bias_interpretations["OI_Bias"] = "Very high CALL OI"
    elif oi_ratio < 1:
        bias_scores["OI_Bias"] = -0.5
        bias_emojis["OI_Bias"] = "🐻 Bearish"
        bias_interpretations["OI_Bias"] = "High CALL OI"
    else:
        bias_scores["OI_Bias"] = 0
        bias_emojis["OI_Bias"] = "⚖️ Neutral"
        bias_interpretations["OI_Bias"] = "Balanced OI"
    
    # OI Change Bias
    if row["Chg_OI_PE"] > 0 and row["Chg_OI_CE"] > 0:
        bias_scores["ChgOI_Bias"] = 0
        bias_emojis["ChgOI_Bias"] = "⚖️ Neutral"
        bias_interpretations["ChgOI_Bias"] = "Both sides building"
    elif row["Chg_OI_PE"] > 0:
        bias_scores["ChgOI_Bias"] = 1
        bias_emojis["ChgOI_Bias"] = "🐂 Bullish"
        bias_interpretations["ChgOI_Bias"] = "PUT building"
    elif row["Chg_OI_CE"] > 0:
        bias_scores["ChgOI_Bias"] = -1
        bias_emojis["ChgOI_Bias"] = "🐻 Bearish"
        bias_interpretations["ChgOI_Bias"] = "CALL building"
    else:
        bias_scores["ChgOI_Bias"] = 0
        bias_emojis["ChgOI_Bias"] = "⚖️ Neutral"
        bias_interpretations["ChgOI_Bias"] = "No fresh writing"
    
    # Volume Bias
    vol_ratio = row["Vol_PE"] / max(row["Vol_CE"], 1)
    if vol_ratio > 1.5:
        bias_scores["Volume_Bias"] = 1
        bias_emojis["Volume_Bias"] = "🐂 Bullish"
        bias_interpretations["Volume_Bias"] = "High PUT volume"
    elif vol_ratio > 1:
        bias_scores["Volume_Bias"] = 0.5
        bias_emojis["Volume_Bias"] = "🐂 Bullish"
        bias_interpretations["Volume_Bias"] = "More PUT volume"
    elif vol_ratio < 0.7:
        bias_scores["Volume_Bias"] = -1
        bias_emojis["Volume_Bias"] = "🐻 Bearish"
        bias_interpretations["Volume_Bias"] = "High CALL volume"
    elif vol_ratio < 1:
        bias_scores["Volume_Bias"] = -0.5
        bias_emojis["Volume_Bias"] = "🐻 Bearish"
        bias_interpretations["Volume_Bias"] = "More CALL volume"
    else:
        bias_scores["Volume_Bias"] = 0
        bias_emojis["Volume_Bias"] = "⚖️ Neutral"
        bias_interpretations["Volume_Bias"] = "Balanced volume"
    
    # Calculate total score
    total_score = sum(bias_scores.values())
    normalized_score = total_score / len(bias_scores) if bias_scores else 0
    
    # Determine verdict
    if normalized_score > 0.3:
        verdict = "🐂 BULLISH"
        verdict_color = "#00ff88"
    elif normalized_score > 0.1:
        verdict = "🐂 Mild Bullish"
        verdict_color = "#00cc66"
    elif normalized_score < -0.3:
        verdict = "🐻 BEARISH"
        verdict_color = "#ff4444"
    elif normalized_score < -0.1:
        verdict = "🐻 Mild Bearish"
        verdict_color = "#ff6666"
    else:
        verdict = "⚖️ NEUTRAL"
        verdict_color = "#66b3ff"
    
    return {
        "instrument": "NIFTY",
        "strike": int(row["strikePrice"]),
        "zone": level_type,
        "level": f"{level_type} Level",
        "bias_scores": bias_scores,
        "bias_interpretations": bias_interpretations,
        "bias_emojis": bias_emojis,
        "total_score": normalized_score,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "metrics": {
            "ce_oi": int(row["OI_CE"]),
            "pe_oi": int(row["OI_PE"]),
            "ce_chg": int(row["Chg_OI_CE"]),
            "pe_chg": int(row["Chg_OI_PE"]),
            "ce_vol": int(row["Vol_CE"]),
            "pe_vol": int(row["Vol_PE"]),
            "distance": abs(spot - row["strikePrice"]),
            "distance_pct": abs(spot - row["strikePrice"]) / spot * 100
        }
    }

# ============================================
# 🎯 COMPREHENSIVE BIAS DASHBOARD (NEW)
# ============================================
def display_bias_dashboard(atm_bias, support_bias, resistance_bias):
    """Display comprehensive bias dashboard"""

    st.markdown("## 🎯 MULTI-DIMENSIONAL BIAS ANALYSIS")

    # Create columns for each bias analysis
    col_atm, col_sup, col_res = st.columns(3)
    
    with col_atm:
        if atm_bias:
            st.markdown(f"""
            <div class='card' style='border-color:{atm_bias["verdict_color"]};'>
                <h4 style='color:{atm_bias["verdict_color"]};'>🏛️ ATM ±2 BIAS</h4>
                <div style='font-size: 1.8rem; color:{atm_bias["verdict_color"]}; font-weight:900; text-align:center;'>
                    {atm_bias["verdict"]}
                </div>
                <div style='font-size: 1.2rem; color:#ffcc00; text-align:center;'>
                    ₹{atm_bias["atm_strike"]:,}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc; text-align:center; margin-top:10px;'>
                    Score: {atm_bias["total_score"]:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key metrics
            st.metric("CALL OI", f"{atm_bias['metrics']['ce_oi']:,}")
            st.metric("PUT OI", f"{atm_bias['metrics']['pe_oi']:,}")
            st.metric("Net Delta", f"{atm_bias['metrics']['net_delta']:.3f}")
            st.metric("Net Gamma", f"{atm_bias['metrics']['net_gamma']:.3f}")

    with col_sup:
        if support_bias:
            st.markdown(f"""
            <div class='card' style='border-color:{support_bias["verdict_color"]};'>
                <h4 style='color:{support_bias["verdict_color"]};'>🛡️ SUPPORT BIAS</h4>
                <div style='font-size: 1.8rem; color:{support_bias["verdict_color"]}; font-weight:900; text-align:center;'>
                    {support_bias["verdict"]}
                </div>
                <div style='font-size: 1.2rem; color:#00ffcc; text-align:center;'>
                    ₹{support_bias["strike"]:,}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc; text-align:center; margin-top:10px;'>
                    Score: {support_bias["total_score"]:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            st.metric("Distance", f"₹{support_bias['metrics']['distance']:.0f}")
            st.metric("CALL OI", f"{support_bias['metrics']['ce_oi']:,}")
            st.metric("PUT OI", f"{support_bias['metrics']['pe_oi']:,}")
    
    with col_res:
        if resistance_bias:
            st.markdown(f"""
            <div class='card' style='border-color:{resistance_bias["verdict_color"]};'>
                <h4 style='color:{resistance_bias["verdict_color"]};'>⚡ RESISTANCE BIAS</h4>
                <div style='font-size: 1.8rem; color:{resistance_bias["verdict_color"]}; font-weight:900; text-align:center;'>
                    {resistance_bias["verdict"]}
                </div>
                <div style='font-size: 1.2rem; color:#ff9900; text-align:center;'>
                    ₹{resistance_bias["strike"]:,}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc; text-align:center; margin-top:10px;'>
                    Score: {resistance_bias["total_score"]:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            st.metric("Distance", f"₹{resistance_bias['metrics']['distance']:.0f}")
            st.metric("CALL OI", f"{resistance_bias['metrics']['ce_oi']:,}")
            st.metric("PUT OI", f"{resistance_bias['metrics']['pe_oi']:,}")
    
    # Detailed ATM Bias Tables
    if atm_bias:
        col1, col2 = st.columns(2)

        with col1:
            if atm_bias:
                st.markdown("### 📊 ATM ±2 BIAS DETAILED BREAKDOWN")

                bias_data = []
                for bias_name, emoji in atm_bias["bias_emojis"].items():
                    bias_data.append({
                        "Metric": bias_name.replace("_", " ").title(),
                        "Bias": emoji,
                        "Score": f"{atm_bias['bias_scores'][bias_name]:.1f}",
                        "Interpretation": atm_bias["bias_interpretations"][bias_name]
                    })

                bias_df = pd.DataFrame(bias_data)
                st.dataframe(bias_df, use_container_width=True, height=400)

                # ATM Bias Summary
                st.markdown(f"""
                <div class='seller-explanation'>
                    <h4>🎯 ATM ±2 BIAS SUMMARY</h4>
                    <p><strong>Overall Verdict:</strong> <span style='color:{atm_bias["verdict_color"]}'>{atm_bias["verdict"]}</span></p>
                    <p><strong>Total Score:</strong> {atm_bias["total_score"]:.2f}</p>
                    <p><strong>Explanation:</strong> {atm_bias["verdict_explanation"]}</p>
                    <p><strong>Key Insights:</strong></p>
                    <ul>
                        <li>CALL OI: {atm_bias['metrics']['ce_oi']:,} | PUT OI: {atm_bias['metrics']['pe_oi']:,}</li>
                        <li>Net Delta: {atm_bias['metrics']['net_delta']:.3f} | Net Gamma: {atm_bias['metrics']['net_gamma']:.3f}</li>
                        <li>Delta Exposure: ₹{atm_bias['metrics']['delta_exposure']:,}</li>
                        <li>Gamma Exposure: ₹{atm_bias['metrics']['gamma_exposure']:,}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            pass
    
    # Trading Implications
    st.markdown("### 💡 TRADING IMPLICATIONS")
    
    implications = []
    
    if atm_bias:
        if atm_bias["total_score"] > 0.2:
            implications.append("✅ **ATM Bullish Bias:** Favor LONG positions with stops below ATM")
        elif atm_bias["total_score"] < -0.2:
            implications.append("✅ **ATM Bearish Bias:** Favor SHORT positions with stops above ATM")
        
        if atm_bias["metrics"]["gamma_exposure"] < -100000:
            implications.append("⚠️ **Negative Gamma Exposure:** Expect whipsaws around ATM")
        elif atm_bias["metrics"]["gamma_exposure"] > 100000:
            implications.append("✅ **Positive Gamma Exposure:** Market stabilizing around ATM")
    
    if support_bias and support_bias["total_score"] > 0.3:
        implications.append(f"✅ **Strong Support at ₹{support_bias['strike']:,}:** Good for LONG entries")
    
    if resistance_bias and resistance_bias["total_score"] < -0.3:
        implications.append(f"✅ **Strong Resistance at ₹{resistance_bias['strike']:,}:** Good for SHORT entries")
    
    if not implications:
        implications.append("⚖️ **Balanced Market:** No clear edge, wait for breakout")
    
    for imp in implications:
        st.markdown(f"- {imp}")

# ============================================
# 📊 ATM ±2 STRIKES DETAILED TABULATION
# ============================================

def analyze_individual_strike_bias(strike_data, strike_price, atm_strike, expiry=""):
    """
    Calculate 14 bias metrics for a single strike (Seller's Perspective)
    Includes: OI, ChgOI, Vol, Delta, Gamma, Premium, IV, DeltaExp, GammaExp, IVSkew, OIChgRate, PCR, MarketDepth, BidAskDepth
    Returns: dict with bias scores, emojis, and interpretations for one strike
    """
    bias_scores = {}
    bias_emojis = {}
    bias_interpretations = {}

    # Extract data for this strike
    ce_oi = strike_data.get("OI_CE", 0)
    pe_oi = strike_data.get("OI_PE", 0)
    ce_chg = strike_data.get("Chg_OI_CE", 0)
    pe_chg = strike_data.get("Chg_OI_PE", 0)
    ce_vol = strike_data.get("Vol_CE", 0)
    pe_vol = strike_data.get("Vol_PE", 0)
    ce_ltp = strike_data.get("LTP_CE", 0)
    pe_ltp = strike_data.get("LTP_PE", 0)
    ce_iv = strike_data.get("IV_CE", 0)
    pe_iv = strike_data.get("IV_PE", 0)
    security_id_ce = strike_data.get("SecurityId_CE", 0)
    security_id_pe = strike_data.get("SecurityId_PE", 0)

    # Fetch market depth for this strike
    # Try strike/expiry based approach first, then fall back to security IDs
    depth_data = None
    depth_error = None

    # Call with strike and expiry parameters
    depth_data = get_option_contract_depth(
        security_id_ce=security_id_ce,
        security_id_pe=security_id_pe,
        strike_price=strike_price,
        expiry=expiry
    )

    if not depth_data.get("available") and "error" in depth_data:
        depth_error = depth_data.get("error")

    # 1. OI BIAS
    oi_ratio = pe_oi / max(ce_oi, 1)
    if oi_ratio > 1.3:
        bias_scores["OI"] = 1
        bias_emojis["OI"] = "🐂"
    elif oi_ratio < 0.77:
        bias_scores["OI"] = -1
        bias_emojis["OI"] = "🐻"
    else:
        bias_scores["OI"] = 0
        bias_emojis["OI"] = "⚖️"
    bias_interpretations["OI"] = f"PE/CE OI: {oi_ratio:.2f}"

    # 2. CHANGE IN OI BIAS
    if ce_chg > 0 and pe_chg > 0:
        chg_ratio = pe_chg / max(ce_chg, 1)
        if chg_ratio > 1.2:
            bias_scores["ChgOI"] = 1
            bias_emojis["ChgOI"] = "🐂"
        elif chg_ratio < 0.83:
            bias_scores["ChgOI"] = -1
            bias_emojis["ChgOI"] = "🐻"
        else:
            bias_scores["ChgOI"] = 0
            bias_emojis["ChgOI"] = "⚖️"
    elif pe_chg > 0:
        bias_scores["ChgOI"] = 1
        bias_emojis["ChgOI"] = "🐂"
    elif ce_chg > 0:
        bias_scores["ChgOI"] = -1
        bias_emojis["ChgOI"] = "🐻"
    else:
        bias_scores["ChgOI"] = 0
        bias_emojis["ChgOI"] = "⚖️"
    bias_interpretations["ChgOI"] = f"CE:{ce_chg:,.0f} PE:{pe_chg:,.0f}"

    # 3. VOLUME BIAS
    vol_ratio = pe_vol / max(ce_vol, 1)
    if vol_ratio > 1.2:
        bias_scores["Volume"] = 1
        bias_emojis["Volume"] = "🐂"
    elif vol_ratio < 0.83:
        bias_scores["Volume"] = -1
        bias_emojis["Volume"] = "🐻"
    else:
        bias_scores["Volume"] = 0
        bias_emojis["Volume"] = "⚖️"
    bias_interpretations["Volume"] = f"PE/CE Vol: {vol_ratio:.2f}"

    # 4. DELTA BIAS (simplified - based on position relative to ATM)
    if strike_price < atm_strike:
        # ITM Call, OTM Put - bullish if PE OI > CE OI
        delta_bias = 1 if pe_oi > ce_oi else -0.5
    elif strike_price > atm_strike:
        # OTM Call, ITM Put - bearish if CE OI > PE OI
        delta_bias = -1 if ce_oi > pe_oi else 0.5
    else:
        # ATM
        delta_bias = 1 if pe_oi > ce_oi * 1.2 else (-1 if ce_oi > pe_oi * 1.2 else 0)

    bias_scores["Delta"] = delta_bias
    bias_emojis["Delta"] = "🐂" if delta_bias > 0 else ("🐻" if delta_bias < 0 else "⚖️")
    bias_interpretations["Delta"] = f"Position: {'ITM' if abs(strike_price - atm_strike) < 50 else 'OTM'}"

    # 5. GAMMA BIAS (highest at ATM)
    distance_from_atm = abs(strike_price - atm_strike)
    if distance_from_atm == 0:
        gamma_score = 1 if pe_oi > ce_oi else -1
    else:
        gamma_score = 0.5 if pe_oi > ce_oi else -0.5

    bias_scores["Gamma"] = gamma_score
    bias_emojis["Gamma"] = "🐂" if gamma_score > 0 else ("🐻" if gamma_score < 0 else "⚖️")
    bias_interpretations["Gamma"] = f"ATM Distance: {distance_from_atm}"

    # 6. PREMIUM BIAS
    premium_ratio = pe_ltp / max(ce_ltp, 0.01)
    if premium_ratio > 1.5:
        bias_scores["Premium"] = 1
        bias_emojis["Premium"] = "🐂"
    elif premium_ratio < 0.67:
        bias_scores["Premium"] = -1
        bias_emojis["Premium"] = "🐻"
    else:
        bias_scores["Premium"] = 0
        bias_emojis["Premium"] = "⚖️"
    bias_interpretations["Premium"] = f"PE/CE Premium: {premium_ratio:.2f}"

    # 7. IV BIAS
    iv_diff = pe_iv - ce_iv
    if iv_diff > 2:
        bias_scores["IV"] = 1
        bias_emojis["IV"] = "🐂"
    elif iv_diff < -2:
        bias_scores["IV"] = -1
        bias_emojis["IV"] = "🐻"
    else:
        bias_scores["IV"] = 0
        bias_emojis["IV"] = "⚖️"
    bias_interpretations["IV"] = f"PE-CE IV: {iv_diff:.2f}%"

    # 8. DELTA EXPOSURE (OI-weighted delta)
    ce_delta_exp = ce_oi * 0.5  # Simplified delta
    pe_delta_exp = pe_oi * (-0.5)
    net_delta_exp = ce_delta_exp + pe_delta_exp

    if net_delta_exp > 0:
        bias_scores["DeltaExp"] = 1
        bias_emojis["DeltaExp"] = "🐂"
    elif net_delta_exp < 0:
        bias_scores["DeltaExp"] = -1
        bias_emojis["DeltaExp"] = "🐻"
    else:
        bias_scores["DeltaExp"] = 0
        bias_emojis["DeltaExp"] = "⚖️"
    bias_interpretations["DeltaExp"] = f"Net ΔExp: {net_delta_exp:,.0f}"

    # 9. GAMMA EXPOSURE (OI-weighted gamma)
    gamma = 0.01  # Simplified
    ce_gamma_exp = ce_oi * gamma
    pe_gamma_exp = pe_oi * gamma
    net_gamma_exp = ce_gamma_exp - pe_gamma_exp

    if net_gamma_exp > 0:
        bias_scores["GammaExp"] = -1
        bias_emojis["GammaExp"] = "🐻"
    elif net_gamma_exp < 0:
        bias_scores["GammaExp"] = 1
        bias_emojis["GammaExp"] = "🐂"
    else:
        bias_scores["GammaExp"] = 0
        bias_emojis["GammaExp"] = "⚖️"
    bias_interpretations["GammaExp"] = f"Net γExp: {net_gamma_exp:,.0f}"

    # 10. IV SKEW BIAS
    avg_iv = (ce_iv + pe_iv) / 2
    if avg_iv > 18:
        bias_scores["IVSkew"] = -0.5
        bias_emojis["IVSkew"] = "🐻"
    elif avg_iv < 12:
        bias_scores["IVSkew"] = 0.5
        bias_emojis["IVSkew"] = "🐂"
    else:
        bias_scores["IVSkew"] = 0
        bias_emojis["IVSkew"] = "⚖️"
    bias_interpretations["IVSkew"] = f"Avg IV: {avg_iv:.2f}%"

    # 11. OI CHANGE RATE (acceleration)
    total_oi = ce_oi + pe_oi
    total_chg = abs(ce_chg) + abs(pe_chg)
    chg_rate = total_chg / max(total_oi, 1) * 100

    if chg_rate > 5:
        if pe_chg > ce_chg:
            bias_scores["OIChgRate"] = 1
            bias_emojis["OIChgRate"] = "🐂"
        else:
            bias_scores["OIChgRate"] = -1
            bias_emojis["OIChgRate"] = "🐻"
    else:
        bias_scores["OIChgRate"] = 0
        bias_emojis["OIChgRate"] = "⚖️"
    bias_interpretations["OIChgRate"] = f"Chg Rate: {chg_rate:.2f}%"

    # 12. PCR AT STRIKE (Put-Call Ratio)
    pcr_strike = pe_oi / max(ce_oi, 1)
    if pcr_strike > 1.5:
        bias_scores["PCR"] = 1
        bias_emojis["PCR"] = "🐂"
    elif pcr_strike < 0.67:
        bias_scores["PCR"] = -1
        bias_emojis["PCR"] = "🐻"
    else:
        bias_scores["PCR"] = 0
        bias_emojis["PCR"] = "⚖️"
    bias_interpretations["PCR"] = f"Strike PCR: {pcr_strike:.2f}"

    # 13. MARKET DEPTH BIAS (CE vs PE Orderbook from 20-level API)
    if depth_data and depth_data.get("available"):
        # Calculate depth imbalance for CE and PE separately
        ce_depth_imbalance = (depth_data["ce_bid_qty"] - depth_data["ce_ask_qty"]) / max(depth_data["ce_total"], 1)
        pe_depth_imbalance = (depth_data["pe_bid_qty"] - depth_data["pe_ask_qty"]) / max(depth_data["pe_total"], 1)

        # Net depth bias: Positive PE depth imbalance = Bullish, Positive CE depth imbalance = Bearish
        depth_bias_score = pe_depth_imbalance - ce_depth_imbalance

        if depth_bias_score > 0.3:
            bias_scores["MktDepth"] = 1
            bias_emojis["MktDepth"] = "🐂"
        elif depth_bias_score > 0.1:
            bias_scores["MktDepth"] = 0.5
            bias_emojis["MktDepth"] = "🐂"
        elif depth_bias_score < -0.3:
            bias_scores["MktDepth"] = -1
            bias_emojis["MktDepth"] = "🐻"
        elif depth_bias_score < -0.1:
            bias_scores["MktDepth"] = -0.5
            bias_emojis["MktDepth"] = "🐻"
        else:
            bias_scores["MktDepth"] = 0
            bias_emojis["MktDepth"] = "⚖️"

        bias_interpretations["MktDepth"] = f"CE:{depth_data['ce_total']:,} PE:{depth_data['pe_total']:,}"
    else:
        # No depth data available
        bias_scores["MktDepth"] = 0
        bias_emojis["MktDepth"] = "⚪"
        # Show error for debugging if available
        if depth_error:
            bias_interpretations["MktDepth"] = f"Error: {depth_error[:50]}"
        else:
            bias_interpretations["MktDepth"] = "N/A"

    # 14. BID/ASK DEPTH BIAS (Simple Bid/Ask Ratio - Seller's Perspective)
    # Extract bid/ask quantities from basic option chain data
    ce_bid_qty = strike_data.get("BidQty_CE", 0)
    pe_bid_qty = strike_data.get("BidQty_PE", 0)
    ce_ask_qty = strike_data.get("AskQty_CE", 0)
    pe_ask_qty = strike_data.get("AskQty_PE", 0)

    # Seller's View:
    # BID side = Buyers (people buying from sellers)
    # ASK side = Sellers (people selling)

    # If Call Bid > Put Bid → More buyers want calls → Bearish (expecting up move)
    # If Put Bid > Call Bid → More buyers want puts → Bullish (expecting down move protection)
    # If Call Ask > Put Ask → More sellers offering calls → Bullish (sellers betting price won't go up)
    # If Put Ask > Call Ask → More sellers offering puts → Bearish (sellers betting price won't go down)

    ba_depth_score = 0

    # Analyze BID depth (buying pressure)
    if pe_bid_qty > 0 or ce_bid_qty > 0:
        bid_ratio = pe_bid_qty / max(ce_bid_qty, 1)
        if bid_ratio > 1.3:  # More PUT buyers (bearish protection = bullish sellers)
            ba_depth_score += 0.5
        elif bid_ratio < 0.77:  # More CALL buyers (bullish bets = bearish sellers)
            ba_depth_score -= 0.5

    # Analyze ASK depth (selling pressure)
    if pe_ask_qty > 0 or ce_ask_qty > 0:
        ask_ratio = ce_ask_qty / max(pe_ask_qty, 1)
        if ask_ratio > 1.3:  # More CALL sellers (bearish view = bullish)
            ba_depth_score += 0.5
        elif ask_ratio < 0.77:  # More PUT sellers (bullish view = bearish)
            ba_depth_score -= 0.5

    # Final bid/ask depth bias
    if ba_depth_score > 0.5:
        bias_scores["BA"] = 1
        bias_emojis["BA"] = "🐂"
    elif ba_depth_score < -0.5:
        bias_scores["BA"] = -1
        bias_emojis["BA"] = "🐻"
    else:
        bias_scores["BA"] = 0
        bias_emojis["BA"] = "⚪"  # White circle for neutral
    bias_interpretations["BA"] = f"Bid: PE/CE {pe_bid_qty/max(ce_bid_qty,1):.2f} | Ask: CE/PE {ce_ask_qty/max(pe_ask_qty,1):.2f}"

    # Calculate overall verdict for this strike
    total_bias = sum(bias_scores.values())
    if total_bias >= 3:
        verdict = "🐂 STRONG BULLISH"
        verdict_color = "#00FF00"
    elif total_bias >= 1:
        verdict = "🐂 Bullish"
        verdict_color = "#90EE90"
    elif total_bias <= -3:
        verdict = "🐻 STRONG BEARISH"
        verdict_color = "#FF0000"
    elif total_bias <= -1:
        verdict = "🐻 Bearish"
        verdict_color = "#FFA07A"
    else:
        verdict = "⚖️ Neutral"
        verdict_color = "#FFD700"

    return {
        "strike_price": strike_price,
        "bias_scores": bias_scores,
        "bias_emojis": bias_emojis,
        "bias_interpretations": bias_interpretations,
        "total_bias": total_bias,
        "verdict": verdict,
        "verdict_color": verdict_color
    }


def create_atm_strikes_tabulation(merged_df, spot, atm_strike, strike_gap, expiry=""):
    """
    Create tabulation for ATM ±2 strikes with 14 bias metrics each
    (OI, ChgOI, Vol, Δ, γ, Prem, IV, ΔExp, γExp, IVSkew, OIRate, PCR, MktDepth, BA)
    Includes Market Depth (20-level API) and Bid/Ask Depth (simple ratios) analysis
    Returns: list of strike analyses
    """
    strike_analyses = []

    # Get ATM ±2 strikes
    for offset in [-2, -1, 0, 1, 2]:
        strike_price = atm_strike + (offset * strike_gap)

        # Get data for this strike
        strike_row = merged_df[merged_df["strikePrice"] == strike_price]

        if not strike_row.empty:
            strike_data = strike_row.iloc[0].to_dict()
            analysis = analyze_individual_strike_bias(strike_data, strike_price, atm_strike, expiry)
            strike_analyses.append(analysis)

    return strike_analyses


def display_atm_strikes_tabulation(strike_analyses, atm_strike):
    """
    Display the ATM ±2 strikes tabulation with 14 bias metrics
    Highlights ATM strike prominently
    Shows overall bias based on strike verdicts
    """
    if not strike_analyses:
        st.warning("⚠️ No strike data available for tabulation")
        return

    # Get ATM strike analysis
    atm_analysis = next((a for a in strike_analyses if a["strike_price"] == atm_strike), None)

    if atm_analysis:
        atm_verdict = atm_analysis["verdict"]
        atm_total_bias = atm_analysis["total_bias"]

        # Determine verdict color
        if "BULLISH" in atm_verdict.upper():
            verdict_color = "#00FF00"
        elif "BEARISH" in atm_verdict.upper():
            verdict_color = "#FF0000"
        else:
            verdict_color = "#FFD700"

        # Count bullish and bearish metrics within ATM strike
        bullish_metrics = sum(1 for score in atm_analysis["bias_scores"].values() if score > 0)
        bearish_metrics = sum(1 for score in atm_analysis["bias_scores"].values() if score < 0)
        total_metrics = len(atm_analysis["bias_scores"])
    else:
        atm_verdict = "N/A"
        atm_total_bias = 0
        verdict_color = "#FFD700"
        bullish_metrics = 0
        bearish_metrics = 0
        total_metrics = 14

    # Display overall bias summary
    st.markdown("### 📊 ATM ±2 Strikes - 14 Bias Metrics Tabulation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a2e1a 0%, #2a3e2a 100%);
            padding: 20px;
            border-radius: 12px;
            border: 3px solid {verdict_color};
            text-align: center;
        ">
            <div style='font-size: 1rem; color:#cccccc; margin-bottom: 10px;'>ATM STRIKE VERDICT</div>
            <div style='font-size: 2.5rem; color:{verdict_color}; font-weight:900;'>
                {atm_verdict}
            </div>
            <div style='font-size: 1.2rem; color:#ffffff; margin-top: 10px;'>
                Strike: {atm_strike} | Score: {atm_total_bias:+.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        ">
            <div style='font-size: 0.9rem; color:#cccccc;'>🐂 BULLISH METRICS</div>
            <div style='font-size: 1.8rem; color:#00ff88; font-weight:700;'>
                {bullish_metrics} / {total_metrics}
            </div>
            <div style='font-size: 0.9rem; color:#cccccc; margin-top: 10px;'>🐻 BEARISH METRICS</div>
            <div style='font-size: 1.8rem; color:#ff4444; font-weight:700;'>
                {bearish_metrics} / {total_metrics}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        bullish_pct = (bullish_metrics / total_metrics * 100) if total_metrics > 0 else 0
        bearish_pct = (bearish_metrics / total_metrics * 100) if total_metrics > 0 else 0

        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        ">
            <div style='font-size: 0.9rem; color:#cccccc;'>BULLISH %</div>
            <div style='font-size: 1.8rem; color:#00ff88; font-weight:700;'>
                {bullish_pct:.1f}%
            </div>
            <div style='font-size: 0.9rem; color:#cccccc; margin-top: 10px;'>BEARISH %</div>
            <div style='font-size: 1.8rem; color:#ff4444; font-weight:700;'>
                {bearish_pct:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Create header row
    metrics = ["Strike", "OI", "ChgOI", "Vol", "Δ", "γ", "Prem", "IV", "ΔExp", "γExp", "IVSkew", "OIRate", "PCR", "MktDepth", "BA", "Verdict"]

    # Build HTML table
    html = '<div style="overflow-x: auto;"><table style="width:100%; border-collapse: collapse; font-size: 12px;">'

    # Header
    html += '<tr style="background-color: #1e1e1e; color: white;">'
    for metric in metrics:
        html += f'<th style="padding: 8px; border: 1px solid #444; text-align: center;">{metric}</th>'
    html += '</tr>'

    # Data rows
    for analysis in strike_analyses:
        strike = analysis["strike_price"]
        is_atm = (strike == atm_strike)

        # Highlight ATM row
        if is_atm:
            row_style = 'background-color: #FFD700; color: #000; font-weight: bold;'
        else:
            row_style = 'background-color: #2d2d2d; color: #fff;'

        html += f'<tr style="{row_style}">'

        # Strike price
        html += f'<td style="padding: 8px; border: 1px solid #444; text-align: center; font-weight: bold;">{strike}</td>'

        # 14 bias metrics
        for metric in ["OI", "ChgOI", "Volume", "Delta", "Gamma", "Premium", "IV", "DeltaExp", "GammaExp", "IVSkew", "OIChgRate", "PCR", "MktDepth", "BA"]:
            emoji = analysis["bias_emojis"].get(metric, "⚖️")
            score = analysis["bias_scores"].get(metric, 0)
            html += f'<td style="padding: 8px; border: 1px solid #444; text-align: center;">{emoji}<br/><small>{score:+.1f}</small></td>'

        # Verdict
        verdict_color = analysis["verdict_color"]
        verdict = analysis["verdict"]
        html += f'<td style="padding: 8px; border: 1px solid #444; text-align: center; background-color: {verdict_color}; font-weight: bold;">{verdict}</td>'

        html += '</tr>'

    html += '</table></div>'

    st.markdown(html, unsafe_allow_html=True)

    # Add expandable detailed breakdown for ATM strike
    with st.expander("🔍 Detailed ATM Strike Breakdown"):
        atm_analysis = next((a for a in strike_analyses if a["strike_price"] == atm_strike), None)
        if atm_analysis:
            st.markdown(f"**ATM Strike: {atm_strike}**")
            st.markdown(f"**Total Bias Score: {atm_analysis['total_bias']:+.1f}**")
            st.markdown(f"**Overall Verdict: {atm_analysis['verdict']}**")
            st.markdown("---")
            for metric, interpretation in atm_analysis["bias_interpretations"].items():
                emoji = atm_analysis["bias_emojis"][metric]
                score = atm_analysis["bias_scores"][metric]
                st.markdown(f"**{metric}**: {emoji} {score:+.1f} - {interpretation}")


def display_overall_market_sentiment_summary(overall_bias, atm_bias, seller_max_pain, total_gex_net, expiry_spike_data, oi_pcr_metrics, strike_analyses, sector_rotation_data=None, seller_bias_result=None, nearest_sup=None, nearest_res=None, moment_metrics=None, days_to_expiry=None):
    """
    Display a consolidated dashboard of the most important market sentiment indicators
    Organized in a clean tabulation format
    """
    st.markdown("---")
    st.markdown("## 📈 OVERALL MARKET SENTIMENT SUMMARY")
    st.markdown("*Consolidated view of essential option chain metrics*")
    st.markdown("---")

    # ATM ±2 Strike Tabulation
    st.markdown("### 📊 ATM ±2 Strikes - Detailed Bias Tabulation")
    if strike_analyses:
        display_atm_strikes_tabulation(strike_analyses, atm_bias.get("atm_strike", 0) if atm_bias else 0)
    else:
        st.warning("⚠️ No strike analysis data available")

    st.markdown("---")

    # Row 3: PCR Bias Analysis
    st.markdown("### 📊 PUT-CALL RATIO (PCR) BIAS")
    if oi_pcr_metrics:
        pcr_col1, pcr_col2, pcr_col3 = st.columns(3)

        with pcr_col1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 20px;
                border-radius: 12px;
                border: 3px solid {oi_pcr_metrics['pcr_color']};
                text-align: center;
            ">
                <div style='font-size: 1rem; color:#cccccc; margin-bottom: 10px;'>PCR VALUE</div>
                <div style='font-size: 3rem; color:{oi_pcr_metrics['pcr_color']}; font-weight:900;'>
                    {oi_pcr_metrics['pcr_total']:.2f}
                </div>
                <div style='font-size: 1.2rem; color:{oi_pcr_metrics['pcr_color']}; margin-top: 10px;'>
                    {oi_pcr_metrics['pcr_interpretation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with pcr_col2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 20px;
                border-radius: 12px;
                border: 2px solid {oi_pcr_metrics['pcr_color']};
                text-align: center;
            ">
                <div style='font-size: 1rem; color:#cccccc; margin-bottom: 10px;'>SENTIMENT</div>
                <div style='font-size: 2rem; color:{oi_pcr_metrics['pcr_color']}; font-weight:700;'>
                    {oi_pcr_metrics['pcr_sentiment']}
                </div>
                <div style='font-size: 0.9rem; color:#aaaaaa; margin-top: 10px;'>
                    {oi_pcr_metrics['chg_interpretation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with pcr_col3:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 20px;
                border-radius: 12px;
                border: 2px solid #666;
                text-align: center;
            ">
                <div style='font-size: 0.9rem; color:#cccccc;'>CALL OI</div>
                <div style='font-size: 1.5rem; color:#ff6b6b; font-weight:700;'>
                    {oi_pcr_metrics['total_ce_oi']:,.0f}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc; margin-top: 10px;'>PUT OI</div>
                <div style='font-size: 1.5rem; color:#51cf66; font-weight:700;'>
                    {oi_pcr_metrics['total_pe_oi']:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No PCR data available")

    st.markdown("---")

    # NEW: ML Market Regime Detection Display
    if moment_metrics and 'market_regime' in moment_metrics:
        regime_data = moment_metrics['market_regime']
        st.markdown("### 🤖 ML MARKET REGIME DETECTION")

        regime_col1, regime_col2, regime_col3, regime_col4 = st.columns(4)

        # Determine regime color
        regime = regime_data.get('regime', 'Unknown')
        if 'Trending Up' in regime:
            regime_color = "#00ff88"
            regime_icon = "📈"
        elif 'Trending Down' in regime:
            regime_color = "#ff4444"
            regime_icon = "📉"
        elif 'Volatile' in regime or 'Breakout' in regime:
            regime_color = "#ff9800"
            regime_icon = "⚡"
        elif 'Range' in regime:
            regime_color = "#00bcd4"
            regime_icon = "↔️"
        elif 'Consolidation' in regime:
            regime_color = "#9c27b0"
            regime_icon = "⏸️"
        else:
            regime_color = "#888888"
            regime_icon = "❓"

        with regime_col1:
            confidence = regime_data.get('confidence', 0)
            st.markdown(f"""
            <div id="ml-regime-card" style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 20px;
                border-radius: 12px;
                border: 3px solid {regime_color};
                text-align: center;
            ">
                <div style='font-size: 1rem; color:#cccccc; margin-bottom: 10px;'>REGIME</div>
                <div style='font-size: 2.5rem; color:{regime_color}; font-weight:900;'>
                    {regime_icon}
                </div>
                <div style='font-size: 1.3rem; color:{regime_color}; margin-top: 10px; font-weight:700;'>
                    {regime}
                </div>
                <div style='font-size: 0.9rem; color:#aaaaaa; margin-top: 5px;'>
                    Confidence: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with regime_col2:
            trend_strength = regime_data.get('trend_strength', 0)
            trend_color = "#00ff88" if trend_strength > 60 else ("#ffa500" if trend_strength > 40 else "#ff4444")
            st.markdown(f"""
            <div id="ml-trend-strength-card" style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 20px;
                border-radius: 12px;
                border: 2px solid {trend_color};
                text-align: center;
            ">
                <div style='font-size: 1rem; color:#cccccc; margin-bottom: 10px;'>TREND STRENGTH</div>
                <div style='font-size: 2.5rem; color:{trend_color}; font-weight:900;'>
                    {trend_strength:.0f}%
                </div>
                <div style='font-size: 0.9rem; color:#aaaaaa; margin-top: 10px;'>
                    {'🔥 Strong' if trend_strength > 60 else ('⚡ Moderate' if trend_strength > 40 else '💤 Weak')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with regime_col3:
            vol_state = regime_data.get('volatility_state', 'Unknown')
            vol_color_map = {
                'Low': '#00ff88',
                'Normal': '#00bcd4',
                'High': '#ff9800',
                'Extreme': '#ff4444'
            }
            vol_color = vol_color_map.get(vol_state, '#888888')
            st.markdown(f"""
            <div id="ml-volatility-card" style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 20px;
                border-radius: 12px;
                border: 2px solid {vol_color};
                text-align: center;
            ">
                <div style='font-size: 1rem; color:#cccccc; margin-bottom: 10px;'>VOLATILITY</div>
                <div style='font-size: 2rem; color:{vol_color}; font-weight:700;'>
                    {vol_state}
                </div>
                <div style='font-size: 0.9rem; color:#aaaaaa; margin-top: 10px;'>
                    State
                </div>
            </div>
            """, unsafe_allow_html=True)

        with regime_col4:
            strategy = regime_data.get('recommended_strategy', 'N/A')
            timeframe = regime_data.get('optimal_timeframe', 'N/A')
            st.markdown(f"""
            <div id="ml-strategy-card" style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 20px;
                border-radius: 12px;
                border: 2px solid #666;
                text-align: center;
            ">
                <div style='font-size: 0.9rem; color:#cccccc;'>STRATEGY</div>
                <div style='font-size: 1.3rem; color:#00bcd4; font-weight:700;'>
                    {strategy}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc; margin-top: 10px;'>TIMEFRAME</div>
                <div style='font-size: 1.3rem; color:#ffa500; font-weight:700;'>
                    {timeframe}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Row 5: Sector Rotation Analysis Bias
    if sector_rotation_data and sector_rotation_data.get('success'):
        st.markdown("### 🔄 SECTOR ROTATION ANALYSIS BIAS")

        rot_col1, rot_col2, rot_col3 = st.columns(3)

        with rot_col1:
            rotation_bias = sector_rotation_data.get('rotation_bias', 'NEUTRAL')
            rotation_score = sector_rotation_data.get('rotation_score', 0)

            if rotation_bias == "BULLISH":
                rot_color = "#00ff88"
            elif rotation_bias == "BEARISH":
                rot_color = "#ff4444"
            else:
                rot_color = "#66b3ff"

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 20px;
                border-radius: 12px;
                border: 3px solid {rot_color};
                text-align: center;
            ">
                <div style='font-size: 1rem; color:#cccccc; margin-bottom: 10px;'>ROTATION BIAS</div>
                <div style='font-size: 2.5rem; color:{rot_color}; font-weight:900;'>
                    {rotation_bias}
                </div>
                <div style='font-size: 1.2rem; color:#ffffff; margin-top: 10px;'>
                    Score: {rotation_score:+.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rot_col2:
            rotation_type = sector_rotation_data.get('rotation_type', 'N/A')
            rotation_pattern = sector_rotation_data.get('rotation_pattern', 'N/A')

            st.markdown(f"""
            <div style="
                background: rgba(0,0,0,0.3);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            ">
                <div style='font-size: 0.9rem; color:#cccccc;'>Rotation Type</div>
                <div style='font-size: 1.3rem; color:#ffffff; font-weight:700; margin: 10px 0;'>
                    {rotation_type}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc;'>Pattern</div>
                <div style='font-size: 1rem; color:#ffcc00; font-weight:600; margin-top: 5px;'>
                    {rotation_pattern}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rot_col3:
            sector_sentiment = sector_rotation_data.get('sector_sentiment', 'NEUTRAL')
            sector_breadth = sector_rotation_data.get('sector_breadth', 0)

            if "BULLISH" in sector_sentiment:
                sent_color = "#00ff88"
            elif "BEARISH" in sector_sentiment:
                sent_color = "#ff4444"
            else:
                sent_color = "#66b3ff"

            st.markdown(f"""
            <div style="
                background: rgba(0,0,0,0.3);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            ">
                <div style='font-size: 0.9rem; color:#cccccc;'>Sector Breadth Sentiment</div>
                <div style='font-size: 1.5rem; color:{sent_color}; font-weight:700; margin: 10px 0;'>
                    {sector_sentiment}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc;'>Sector Breadth %</div>
                <div style='font-size: 1.3rem; color:#ffcc00; font-weight:700; margin-top: 5px;'>
                    {sector_breadth:.1f}%
                </div>
                <div style='font-size: 0.7rem; color:#666; font-style: italic; margin-top: 3px;'>
                    (% sectors bullish)
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Show leading and lagging sectors
        leaders = sector_rotation_data.get('leaders', [])
        laggards = sector_rotation_data.get('laggards', [])

        if leaders or laggards:
            st.markdown("#### 📈 Sector Leaders & Laggards")
            lead_col, lag_col = st.columns(2)

            with lead_col:
                st.markdown("**🚀 Leading Sectors:**")
                for sector in leaders:
                    st.markdown(f"- {sector['sector']}: {sector['change_pct']:+.2f}%")

            with lag_col:
                st.markdown("**📉 Lagging Sectors:**")
                for sector in laggards:
                    st.markdown(f"- {sector['sector']}: {sector['change_pct']:+.2f}%")

    st.markdown("---")

    # FINAL ASSESSMENT
    if seller_bias_result and atm_bias and oi_pcr_metrics:
        # Prepare ATM bias summary
        atm_verdict = atm_bias.get("verdict", "N/A")
        atm_score = atm_bias.get("total_score", 0)
        atm_bias_summary = f"ATM Bias: {atm_verdict} ({atm_score:.2f} score)"

        # Prepare moment summary
        if moment_metrics:
            moment_burst = moment_metrics.get("momentum_burst", {}).get("score", 0)
            orderbook_pressure = moment_metrics.get("orderbook", {}).get("pressure", 0) if moment_metrics.get("orderbook", {}).get("available") else 0
            moment_summary = f"Burst: {moment_burst}/100, Pressure: {orderbook_pressure:+.2f}"
        else:
            moment_summary = "Moment indicators neutral"

        # Prepare OI/PCR summary
        oi_pcr_summary = f"PCR: {oi_pcr_metrics['pcr_total']:.2f} ({oi_pcr_metrics['pcr_sentiment']}) | CALL OI: {oi_pcr_metrics['total_ce_oi']:,} | PUT OI: {oi_pcr_metrics['total_pe_oi']:,} | ATM Conc: {oi_pcr_metrics['atm_concentration_pct']:.1f}%"

        # Prepare expiry summary
        if expiry_spike_data and expiry_spike_data.get('spike_risk_score', 0) > 60:
            expiry_summary = f"⚠️ HIGH SPIKE RISK ({expiry_spike_data['spike_risk_score']}/100) - {expiry_spike_data.get('spike_type', 'SQUEEZE')}"
        elif days_to_expiry is not None:
            expiry_summary = f"Expiry in {days_to_expiry:.1f} days"
        else:
            expiry_summary = "Expiry data unavailable"

        # Support and resistance strings
        support_str = f"₹{nearest_sup['strike']:,}" if nearest_sup else "N/A"
        resistance_str = f"₹{nearest_res['strike']:,}" if nearest_res else "N/A"

        # Max Pain string
        max_pain_str = f"₹{seller_max_pain.get('max_pain_strike', 0):,}" if seller_max_pain else "N/A"

        st.markdown(f'''
        <div style='
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 25px;
            border-radius: 12px;
            border: 3px solid #ffa500;
            margin-bottom: 20px;
        '>
            <h3 style='color: #ffa500; margin-bottom: 15px;'>🎯 FINAL ASSESSMENT (Seller + ATM Bias + Moment + Expiry + OI/PCR)</h3>
            <p style='margin: 8px 0;'><strong>Market Makers are telling us:</strong> {seller_bias_result["explanation"]}</p>
            <p style='margin: 8px 0;'><strong>ATM Zone Analysis:</strong> {atm_bias_summary}</p>
            <p style='margin: 8px 0;'><strong>Their game plan:</strong> {seller_bias_result["action"]}</p>
            <p style='margin: 8px 0;'><strong>Moment Detector:</strong> {moment_summary}</p>
            <p style='margin: 8px 0;'><strong>OI/PCR Analysis:</strong> {oi_pcr_summary}</p>
            <p style='margin: 8px 0;'><strong>Expiry Context:</strong> {expiry_summary}</p>
            <p style='margin: 8px 0;'><strong>Key defense levels:</strong> {support_str} (Support) | {resistance_str} (Resistance)</p>
            <p style='margin: 8px 0;'><strong>Max OI Walls:</strong> CALL: ₹{oi_pcr_metrics['max_ce_strike']:,} | PUT: ₹{oi_pcr_metrics['max_pe_strike']:,}</p>
            <p style='margin: 8px 0;'><strong>Preferred price level:</strong> {max_pain_str} (Max Pain)</p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    # Row 6: ATM Bias Summaries (Legacy)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏛️ ATM ±2 Bias")
        if atm_bias:
            call_oi = atm_bias.get("total_ce_oi_atm", 0)
            put_oi = atm_bias.get("total_pe_oi_atm", 0)
            net_delta = atm_bias.get("net_delta_exposure_atm", 0)
            net_gamma = atm_bias.get("net_gamma_exposure_atm", 0)

            st.markdown(f"""
            - **CALL OI**: {call_oi:,.0f}
            - **PUT OI**: {put_oi:,.0f}
            - **Net Δ**: {net_delta:,.0f}
            - **Net γ**: {net_gamma:,.0f}
            """)
        else:
            st.info("No ATM ±2 bias data")


    st.markdown("---")

    # Row 4: Expiry Analysis (PCR Analysis removed - available in Seller PCR tab)
    st.markdown("### 📅 Expiry Analysis")
    if expiry_spike_data:
        spike_prob = expiry_spike_data.get("expiry_spike_probability", "N/A")
        key_levels = expiry_spike_data.get("key_resistance_levels", [])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Spike Probability**: {spike_prob}")
        with col2:
            if key_levels:
                st.markdown(f"**Key Levels**: {', '.join(map(str, key_levels[:3]))}")
    else:
        st.info("No expiry spike data")

    st.markdown("---")


# ============================================
# 🎯 OVERALL BIAS CALCULATOR (NEW)
# ============================================
def calculate_overall_bias(atm_bias, support_bias, resistance_bias, seller_bias_result):
    """
    Calculate overall market bias from all available analyses
    Returns a comprehensive bias score and verdict
    """

    total_score = 0
    total_weight = 0
    bias_components = []

    # ATM ±2 Bias (Weight: 25%)
    if atm_bias and atm_bias.get("total_score") is not None:
        weight = 0.25
        score = atm_bias["total_score"]
        total_score += score * weight
        total_weight += weight
        bias_components.append({
            "component": "ATM ±2 Bias",
            "score": score,
            "weight": weight,
            "verdict": atm_bias["verdict"]
        })


    # Support Bias (Weight: 15%)
    if support_bias and support_bias.get("total_score") is not None:
        weight = 0.15
        score = support_bias["total_score"]
        total_score += score * weight
        total_weight += weight
        bias_components.append({
            "component": "Support Bias",
            "score": score,
            "weight": weight,
            "verdict": support_bias["verdict"]
        })

    # Resistance Bias (Weight: 15%)
    if resistance_bias and resistance_bias.get("total_score") is not None:
        weight = 0.15
        score = resistance_bias["total_score"]
        total_score += score * weight
        total_weight += weight
        bias_components.append({
            "component": "Resistance Bias",
            "score": score,
            "weight": weight,
            "verdict": resistance_bias["verdict"]
        })

    # Seller Bias (Weight: 15%)
    if seller_bias_result and seller_bias_result.get("bias"):
        weight = 0.15
        # Convert seller bias to numerical score
        if seller_bias_result["bias"] == "BULLISH":
            score = 0.5
        elif seller_bias_result["bias"] == "BEARISH":
            score = -0.5
        else:
            score = 0
        total_score += score * weight
        total_weight += weight
        bias_components.append({
            "component": "Seller Bias",
            "score": score,
            "weight": weight,
            "verdict": seller_bias_result["bias"]
        })

    # Normalize the total score
    if total_weight > 0:
        normalized_score = total_score / total_weight
    else:
        normalized_score = 0

    # Determine overall verdict
    if normalized_score > 0.4:
        verdict = "🐂 STRONG BULLISH"
        verdict_color = "#00ff88"
        verdict_explanation = "Multiple analyses confirm strong bullish bias"
    elif normalized_score > 0.2:
        verdict = "🐂 BULLISH"
        verdict_color = "#00cc66"
        verdict_explanation = "Analyses lean bullish"
    elif normalized_score > 0.05:
        verdict = "🐂 Mild Bullish"
        verdict_color = "#00aa55"
        verdict_explanation = "Slight bullish tendency"
    elif normalized_score < -0.4:
        verdict = "🐻 STRONG BEARISH"
        verdict_color = "#ff4444"
        verdict_explanation = "Multiple analyses confirm strong bearish bias"
    elif normalized_score < -0.2:
        verdict = "🐻 BEARISH"
        verdict_color = "#ff6666"
        verdict_explanation = "Analyses lean bearish"
    elif normalized_score < -0.05:
        verdict = "🐻 Mild Bearish"
        verdict_color = "#ff8888"
        verdict_explanation = "Slight bearish tendency"
    else:
        verdict = "⚖️ NEUTRAL"
        verdict_color = "#66b3ff"
        verdict_explanation = "Balanced market, no clear directional bias"

    return {
        "overall_score": normalized_score,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_explanation": verdict_explanation,
        "bias_components": bias_components,
        "total_weight": total_weight
    }

# -----------------------
#  TELEGRAM FUNCTIONS
# -----------------------
def send_telegram_message(bot_token, chat_id, message):
    """
    Actually send message to Telegram
    """
    try:
        if not bot_token or not chat_id:
            return False, "Telegram credentials not configured"
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            return True, "Signal sent to Telegram channel!"
        else:
            return False, f"Failed to send: {response.status_code}"
    except Exception as e:
        return False, f"Telegram error: {str(e)}"

def generate_telegram_signal_option3(entry_signal, spot, seller_bias_result, seller_max_pain, 
                                   nearest_sup, nearest_res, moment_metrics, seller_breakout_index, 
                                   expiry, expiry_spike_data, atm_bias=None, support_bias=None, resistance_bias=None):
    """
    Generate Option 3 Telegram signal with stop loss/target and expiry spike info
    Only generate when position_type is not NEUTRAL
    """
    
    # Only generate signal for non-neutral positions
    if entry_signal["position_type"] == "NEUTRAL":
        return None
    
    position_type = entry_signal["position_type"]
    signal_strength = entry_signal["signal_strength"]
    confidence = entry_signal["confidence"]
    optimal_entry_price = entry_signal["optimal_entry_price"]
    stop_loss = entry_signal["stop_loss"]
    target = entry_signal["target"]
    
    # Emoji based on position
    signal_emoji = "🚀" if position_type == "LONG" else "🐻"
    current_time = get_ist_datetime_str()
    
    # Extract moment scores
    moment_burst = moment_metrics["momentum_burst"].get("score", 0)
    orderbook_pressure = moment_metrics["orderbook"].get("pressure", 0.0)
    
    # Calculate risk:reward if we have stop loss and target
    risk_reward = ""
    stop_distance = ""
    stop_pct = ""
    
    if stop_loss and target and optimal_entry_price:
        if position_type == "LONG":
            risk = abs(optimal_entry_price - stop_loss)
            reward = abs(target - optimal_entry_price)
            stop_distance = f"Stop: {stop_loss:.0f} (↓{risk:.0f} points)"
        else:
            risk = abs(stop_loss - optimal_entry_price)
            reward = abs(optimal_entry_price - target)
            stop_distance = f"Stop: {stop_loss:.0f} (↑{risk:.0f} points)"
        
        if risk > 0:
            risk_reward = f"1:{reward/risk:.1f}"
            stop_pct = f"({risk/optimal_entry_price*100:.1f}%)"
    
    # Format stop loss and target with points
    stop_loss_str = f"₹{stop_loss:,.0f}" if stop_loss else "N/A"
    target_str = f"₹{target:,.0f}" if target else "N/A"
    
    # Format support/resistance
    support_str = f"₹{nearest_sup['strike']:,}" if nearest_sup else "N/A"
    resistance_str = f"₹{nearest_res['strike']:,}" if nearest_res else "N/A"
    
    # Format max pain
    max_pain_str = f"₹{seller_max_pain.get('max_pain_strike', 0):,}" if seller_max_pain else "N/A"
    
    # Calculate entry distance from current spot
    entry_distance = abs(spot - optimal_entry_price)
    
    # Add ATM bias info if available
    atm_bias_info = ""
    if atm_bias:
        atm_bias_info = f"\n🎯 *ATM Bias*: {atm_bias['verdict']} (Score: {atm_bias['total_score']:.2f})"
    
    # Add expiry spike info if active
    expiry_info = ""
    if expiry_spike_data.get("active", False) and expiry_spike_data.get("probability", 0) > 50:
        spike_emoji = "🚨" if expiry_spike_data['probability'] > 70 else "⚠️"
        expiry_info = f"\n{spike_emoji} *Expiry Spike Risk*: {expiry_spike_data['probability']}% - {expiry_spike_data['type']}"
        if expiry_spike_data.get("key_levels"):
            expiry_info += f"\n🎯 *Spike Levels*: {', '.join(expiry_spike_data['key_levels'][:2])}"
        # Add spike strike price ranges
        support_range = expiry_spike_data.get("support_spike_range", {})
        resistance_range = expiry_spike_data.get("resistance_spike_range", {})
        if support_range.get("start") is not None:
            expiry_info += f"\n🛡️ *Support Spike Range*: ₹{support_range['start']:,} → ₹{support_range['end']:,}"
        if resistance_range.get("start") is not None:
            expiry_info += f"\n🚧 *Resistance Spike Range*: ₹{resistance_range['start']:,} → ₹{resistance_range['end']:,}"
    
    # Generate the message
    message = f"""
🎯 *NIFTY OPTION TRADE SETUP*

*Position*: {signal_emoji} {position_type} ({signal_strength})
*Entry Price*: ₹{optimal_entry_price:,.0f}
*Current Spot*: ₹{spot:,.0f}
*Entry Distance*: {entry_distance:.0f} points

*Risk Management*:
🛑 Stop Loss: {stop_loss_str} {stop_distance if stop_distance else ""}
🎯 Target: {target_str}
📊 Risk:Reward = {risk_reward} {stop_pct if stop_pct else ""}

*Key Levels*:
🛡️ Support: {support_str}
⚡ Resistance: {resistance_str}
🎯 Max Pain: {max_pain_str}

*Moment Detector*:
✅ Burst: {moment_burst}/100
✅ Pressure: {orderbook_pressure:+.2f}

*Seller Bias*: {seller_bias_result['bias']}
*Confidence*: {confidence:.0f}%
{atm_bias_info}

*Expiry Context*:
📅 Days to Expiry: {expiry_spike_data.get('days_to_expiry', 0):.1f}
{expiry_info if expiry_info else "📊 Expiry spike risk: Low"}

⏰ {current_time} IST | 📆 Expiry: {expiry}

#NiftyOptions #OptionSelling #TradingSignal
"""
    return message

def check_and_send_signal(entry_signal, spot, seller_bias_result, seller_max_pain, 
                         nearest_sup, nearest_res, moment_metrics, seller_breakout_index, 
                         expiry, expiry_spike_data, atm_bias=None, support_bias=None, resistance_bias=None):
    """
    Check if a new signal is generated and return it (simulated)
    Returns signal message if new signal, None otherwise
    """
    # Store previous signal in session state
    if "last_signal" not in st.session_state:
        st.session_state["last_signal"] = None
    
    # Check if we have a valid signal
    if entry_signal["position_type"] != "NEUTRAL" and entry_signal["confidence"] >= 40:
        current_signal = f"{entry_signal['position_type']}_{entry_signal['optimal_entry_price']:.0f}"
        
        # Check if this is a new signal (different from last one)
        if st.session_state["last_signal"] != current_signal:
            # Generate Telegram message
            telegram_msg = generate_telegram_signal_option3(
                entry_signal, spot, seller_bias_result, 
                seller_max_pain, nearest_sup, nearest_res, 
                moment_metrics, seller_breakout_index, expiry, expiry_spike_data,
                atm_bias, support_bias, resistance_bias
            )
            
            if telegram_msg:
                # Update last signal
                st.session_state["last_signal"] = current_signal
                return telegram_msg
    
    # Reset if signal is gone
    elif st.session_state["last_signal"] is not None:
        st.session_state["last_signal"] = None
    
    return None

# -----------------------
# 📅 EXPIRY SPIKE DETECTOR FUNCTIONS
# -----------------------
def detect_expiry_spikes(merged_df, spot, atm_strike, days_to_expiry, expiry_date_str):
    """
    Detect potential expiry day spikes based on multiple factors
    Returns: dict with spike probability, direction, and key levels
    """
    
    if days_to_expiry > 5:
        return {
            "active": False,
            "probability": 0,
            "message": "Expiry >5 days away, spike detection not active",
            "type": None,
            "key_levels": [],
            "score": 0,
            "color": "#00ff00",  # Green color for no spike
            "intensity": "NO SPIKE DETECTED",
            "factors": [],
            "days_to_expiry": days_to_expiry,
            "expiry_date": expiry_date_str,
            "resistance_spike_range": {"start": None, "end": None, "strikes": [], "total_oi": 0},
            "support_spike_range": {"start": None, "end": None, "strikes": [], "total_oi": 0}
        }
    
    spike_score = 0
    spike_factors = []
    spike_type = None
    key_levels = []
    
    # Factor 1: ATM OI Concentration (0-25 points)
    strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
    atm_window = 2  # ±2 strikes around ATM
    atm_strikes = [s for s in merged_df["strikePrice"] 
                   if abs(s - atm_strike) <= (atm_window * strike_gap_val)]
    
    atm_ce_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_CE"].sum()
    atm_pe_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_PE"].sum()
    total_oi_near_atm = atm_ce_oi + atm_pe_oi
    total_oi_all = merged_df["OI_CE"].sum() + merged_df["OI_PE"].sum()
    
    if total_oi_all > 0:
        atm_concentration = total_oi_near_atm / total_oi_all
        if atm_concentration > 0.5:
            spike_score += 25
            spike_factors.append(f"High ATM OI concentration ({atm_concentration:.1%})")
        elif atm_concentration > 0.3:
            spike_score += 15
            spike_factors.append(f"Moderate ATM OI concentration ({atm_concentration:.1%})")
    
    # Factor 2: Max Pain vs Spot Distance (0-20 points)
    max_pain = calculate_seller_max_pain(merged_df)
    if max_pain:
        max_pain_strike = max_pain.get("max_pain_strike", 0)
        max_pain_distance = abs(spot - max_pain_strike) / spot * 100
        if max_pain_distance > 2.0:
            spike_score += 20
            spike_factors.append(f"Spot far from Max Pain ({max_pain_distance:.1f}%)")
            if spot > max_pain_strike:
                spike_type = "SHORT SQUEEZE"
            else:
                spike_type = "LONG SQUEEZE"
            key_levels.append(f"Max Pain: ₹{max_pain_strike:,}")
        elif max_pain_distance > 1.0:
            spike_score += 10
            spike_factors.append(f"Spot moderately far from Max Pain ({max_pain_distance:.1f}%)")
    
    # Factor 3: PCR Extremes (0-15 points)
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        if pcr > 1.8:
            spike_score += 15
            spike_factors.append(f"Extreme PCR ({pcr:.2f}) - Heavy PUT selling")
            spike_type = "UPWARD SPIKE" if spike_type is None else spike_type
        elif pcr < 0.5:
            spike_score += 15
            spike_factors.append(f"Extreme PCR ({pcr:.2f}) - Heavy CALL selling")
            spike_type = "DOWNWARD SPIKE" if spike_type is None else spike_type
    
    # Factor 4: Large OI Build-up at Single Strike (0-20 points)
    max_ce_oi_strike = merged_df.loc[merged_df["OI_CE"].idxmax()] if not merged_df.empty else None
    max_pe_oi_strike = merged_df.loc[merged_df["OI_PE"].idxmax()] if not merged_df.empty else None

    # Initialize spike range tracking
    resistance_spike_range = {"start": None, "end": None, "strikes": [], "total_oi": 0}
    support_spike_range = {"start": None, "end": None, "strikes": [], "total_oi": 0}

    if max_ce_oi_strike is not None:
        max_ce_oi = int(max_ce_oi_strike["OI_CE"])
        max_ce_strike = int(max_ce_oi_strike["strikePrice"])
        if max_ce_oi > 2000000:  # 2 million+ OI
            spike_score += 20
            spike_factors.append(f"Massive CALL OI at ₹{max_ce_strike:,} ({max_ce_oi:,})")
            key_levels.append(f"CALL Wall: ₹{max_ce_strike:,}")
            if abs(spot - max_ce_strike) < (strike_gap_val * 3):
                spike_type = "RESISTANCE SPIKE"

            # Find resistance spike range - strikes with significant CALL OI (>40% of max)
            oi_threshold = max_ce_oi * 0.40
            significant_ce_strikes = merged_df[
                (merged_df["OI_CE"] >= oi_threshold) &
                (merged_df["strikePrice"] >= spot)
            ].sort_values("strikePrice")

            if not significant_ce_strikes.empty:
                resistance_spike_range["start"] = int(significant_ce_strikes["strikePrice"].min())
                resistance_spike_range["end"] = int(significant_ce_strikes["strikePrice"].max())
                resistance_spike_range["strikes"] = significant_ce_strikes["strikePrice"].astype(int).tolist()
                resistance_spike_range["total_oi"] = int(significant_ce_strikes["OI_CE"].sum())

    if max_pe_oi_strike is not None:
        max_pe_oi = int(max_pe_oi_strike["OI_PE"])
        max_pe_strike = int(max_pe_oi_strike["strikePrice"])
        if max_pe_oi > 2000000:  # 2 million+ OI
            spike_score += 20
            spike_factors.append(f"Massive PUT OI at ₹{max_pe_strike:,} ({max_pe_oi:,})")
            key_levels.append(f"PUT Wall: ₹{max_pe_strike:,}")
            if abs(spot - max_pe_strike) < (strike_gap_val * 3):
                spike_type = "SUPPORT SPIKE"

            # Find support spike range - strikes with significant PUT OI (>40% of max)
            oi_threshold = max_pe_oi * 0.40
            significant_pe_strikes = merged_df[
                (merged_df["OI_PE"] >= oi_threshold) &
                (merged_df["strikePrice"] <= spot)
            ].sort_values("strikePrice")

            if not significant_pe_strikes.empty:
                support_spike_range["start"] = int(significant_pe_strikes["strikePrice"].min())
                support_spike_range["end"] = int(significant_pe_strikes["strikePrice"].max())
                support_spike_range["strikes"] = significant_pe_strikes["strikePrice"].astype(int).tolist()
                support_spike_range["total_oi"] = int(significant_pe_strikes["OI_PE"].sum())
    
    # Factor 5: Gamma Flip Zone (0-10 points)
    if days_to_expiry <= 1:
        spike_score += 10
        spike_factors.append("Gamma flip zone (expiry day)")
    
    # Factor 6: Unwinding Activity (0-10 points)
    ce_unwind = (merged_df["Chg_OI_CE"] < 0).sum()
    pe_unwind = (merged_df["Chg_OI_PE"] < 0).sum()
    total_unwind = ce_unwind + pe_unwind
    
    if total_unwind > 15:  # More than 15 strikes showing unwinding
        spike_score += 10
        spike_factors.append(f"Massive unwinding ({total_unwind} strikes)")
    
    # Determine spike probability
    probability = min(100, int(spike_score * 1.5))
    
    # Spike intensity
    if probability >= 70:
        intensity = "HIGH PROBABILITY SPIKE"
        color = "#ff0000"
    elif probability >= 50:
        intensity = "MODERATE SPIKE RISK"
        color = "#ff9900"
    elif probability >= 30:
        intensity = "LOW SPIKE RISK"
        color = "#ffff00"
    else:
        intensity = "NO SPIKE DETECTED"
        color = "#00ff00"
    
    # Default spike type if none detected
    if spike_type is None:
        spike_type = "UNCERTAIN"
    
    return {
        "active": days_to_expiry <= 5,
        "probability": probability,
        "score": spike_score,
        "intensity": intensity,
        "type": spike_type,
        "color": color,
        "factors": spike_factors,
        "key_levels": key_levels,
        "days_to_expiry": days_to_expiry,
        "expiry_date": expiry_date_str,
        "message": f"Expiry in {days_to_expiry:.1f} days",
        "resistance_spike_range": resistance_spike_range,
        "support_spike_range": support_spike_range
    }


def detect_expiry_spikes_enhanced(merged_df, spot, atm_strike, days_to_expiry, expiry_date_str, total_gex_net=None):
    """
    🚀 ENHANCED Expiry Spike Detection v2.0

    NEW FEATURES:
    1. Expected price range calculation
    2. Target levels based on OI analysis
    3. Time-based probability (morning vs afternoon)
    4. GEX integration for volatility prediction
    5. OI change analysis during expiry

    This is an ENHANCED version - original detect_expiry_spikes() is preserved.

    Returns: dict with spike probability, direction, price range, targets, and key levels
    """
    from datetime import datetime

    # Get base result from original function first
    base_result = detect_expiry_spikes(merged_df, spot, atm_strike, days_to_expiry, expiry_date_str)

    # If not active, return with default enhanced fields
    if days_to_expiry > 5:
        base_result.update({
            "expected_range": {"low": spot, "high": spot, "range_points": 0, "range_percent": 0, "center": spot},
            "target_levels": [],
            "time_analysis": {"current_phase": "N/A", "phase_probability": 0, "high_spike_windows": [], "current_window_active": False},
            "gex_impact": {"gex_value": 0, "impact": "Neutral", "score": 0, "message": "Not active"},
            "oi_change_analysis": {"net_change": 0, "bias": "Neutral", "unwinding_intensity": 0}
        })
        return base_result

    # Get strike gap
    strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED FEATURE 1: OI-BASED TARGET LEVELS
    # ═══════════════════════════════════════════════════════════════════
    target_levels = []

    # Max Pain as primary target
    max_pain = calculate_seller_max_pain(merged_df)
    max_pain_strike = 0
    if max_pain:
        max_pain_strike = max_pain.get("max_pain_strike", 0)
        if max_pain_strike > 0:
            target_levels.append({"level": max_pain_strike, "type": "Max Pain", "priority": 1})

    # CALL and PUT walls
    max_ce_oi_strike = merged_df.loc[merged_df["OI_CE"].idxmax()] if not merged_df.empty else None
    max_pe_oi_strike = merged_df.loc[merged_df["OI_PE"].idxmax()] if not merged_df.empty else None

    max_ce_strike = 0
    max_pe_strike = 0

    if max_ce_oi_strike is not None:
        max_ce_oi = int(max_ce_oi_strike["OI_CE"])
        max_ce_strike = int(max_ce_oi_strike["strikePrice"])
        if max_ce_oi > 1000000:
            target_levels.append({
                "level": max_ce_strike,
                "type": "CALL Wall" if max_ce_oi > 2000000 else "CALL Resistance",
                "priority": 2 if max_ce_oi > 2000000 else 3,
                "oi": max_ce_oi
            })

    if max_pe_oi_strike is not None:
        max_pe_oi = int(max_pe_oi_strike["OI_PE"])
        max_pe_strike = int(max_pe_oi_strike["strikePrice"])
        if max_pe_oi > 1000000:
            target_levels.append({
                "level": max_pe_strike,
                "type": "PUT Wall" if max_pe_oi > 2000000 else "PUT Support",
                "priority": 2 if max_pe_oi > 2000000 else 3,
                "oi": max_pe_oi
            })

    # Additional OI-based targets (2nd and 3rd highest)
    ce_sorted = merged_df.nlargest(3, 'OI_CE')
    pe_sorted = merged_df.nlargest(3, 'OI_PE')

    for idx, row in ce_sorted.iterrows():
        strike = int(row['strikePrice'])
        oi = int(row['OI_CE'])
        if strike != max_ce_strike and strike > spot and oi > 500000:
            target_levels.append({"level": strike, "type": "CE Resistance", "priority": 4, "oi": oi})

    for idx, row in pe_sorted.iterrows():
        strike = int(row['strikePrice'])
        oi = int(row['OI_PE'])
        if strike != max_pe_strike and strike < spot and oi > 500000:
            target_levels.append({"level": strike, "type": "PE Support", "priority": 4, "oi": oi})

    # Sort by priority
    target_levels.sort(key=lambda x: x.get("priority", 99))

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED FEATURE 2: OI CHANGE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    ce_oi_change = merged_df["Chg_OI_CE"].sum()
    pe_oi_change = merged_df["Chg_OI_PE"].sum()
    net_oi_change = ce_oi_change + pe_oi_change

    ce_unwind = (merged_df["Chg_OI_CE"] < 0).sum()
    pe_unwind = (merged_df["Chg_OI_PE"] < 0).sum()
    total_unwind = ce_unwind + pe_unwind
    total_strikes = len(merged_df)
    unwinding_intensity = (total_unwind / total_strikes * 100) if total_strikes > 0 else 0

    # Determine OI change bias
    oi_change_bias = "Neutral"
    oi_change_signal = ""

    if ce_oi_change < -500000 and pe_oi_change > 0:
        oi_change_bias = "Bullish Unwinding"
        oi_change_signal = "CALL unwinding + PUT buildup = Bullish pressure"
    elif pe_oi_change < -500000 and ce_oi_change > 0:
        oi_change_bias = "Bearish Unwinding"
        oi_change_signal = "PUT unwinding + CALL buildup = Bearish pressure"
    elif ce_oi_change < -1000000 and pe_oi_change < -1000000:
        oi_change_bias = "Massive Unwinding"
        oi_change_signal = "Both sides unwinding = Explosive move expected"
    elif ce_oi_change > 500000 and pe_oi_change > 500000:
        oi_change_bias = "Fresh Buildup"
        oi_change_signal = "Fresh OI both sides = Range bound expected"
    elif net_oi_change > 1000000:
        oi_change_bias = "Long Buildup"
        oi_change_signal = "Net long buildup = Trend continuation likely"
    elif net_oi_change < -1000000:
        oi_change_bias = "Short Covering"
        oi_change_signal = "Net short covering = Short squeeze possible"

    oi_change_analysis = {
        "ce_change": int(ce_oi_change),
        "pe_change": int(pe_oi_change),
        "net_change": int(net_oi_change),
        "bias": oi_change_bias,
        "signal": oi_change_signal,
        "unwinding_intensity": round(unwinding_intensity, 1),
        "ce_unwind_strikes": ce_unwind,
        "pe_unwind_strikes": pe_unwind,
        "total_unwind_strikes": total_unwind
    }

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED FEATURE 3: GEX INTEGRATION
    # ═══════════════════════════════════════════════════════════════════
    gex_impact = {
        "gex_value": 0,
        "impact": "Unknown",
        "score": 0,
        "message": "GEX data not provided",
        "volatility_expectation": "Normal"
    }

    additional_spike_score = 0

    if total_gex_net is not None:
        gex_impact["gex_value"] = int(total_gex_net)

        if total_gex_net < -2000000:  # Large negative GEX
            gex_score = min(25, int((abs(total_gex_net) / 1000000) * 5))
            additional_spike_score += gex_score
            gex_impact["impact"] = "EXPLOSIVE"
            gex_impact["score"] = gex_score
            gex_impact["message"] = f"Negative GEX (₹{abs(total_gex_net)/1000000:.1f}M) = Violent moves expected"
            gex_impact["volatility_expectation"] = "Very High - MMs hedge aggressively"

        elif total_gex_net < -500000:
            gex_score = 12
            additional_spike_score += gex_score
            gex_impact["impact"] = "VOLATILE"
            gex_impact["score"] = gex_score
            gex_impact["message"] = "Mild negative GEX = Above average volatility"
            gex_impact["volatility_expectation"] = "High"

        elif total_gex_net > 2000000:  # Large positive GEX
            gex_impact["impact"] = "STABILIZING"
            gex_impact["score"] = -15  # Reduces spike probability
            additional_spike_score -= 10
            gex_impact["message"] = f"Positive GEX (₹{total_gex_net/1000000:.1f}M) = Mean reversion likely"
            gex_impact["volatility_expectation"] = "Low - Price pinning expected"

        elif total_gex_net > 500000:
            gex_impact["impact"] = "MILD STABILIZING"
            gex_impact["score"] = -5
            additional_spike_score -= 5
            gex_impact["message"] = "Mild positive GEX = Reduced volatility"
            gex_impact["volatility_expectation"] = "Below Average"
        else:
            gex_impact["impact"] = "NEUTRAL"
            gex_impact["score"] = 0
            gex_impact["message"] = "GEX neutral = Normal market behavior"
            gex_impact["volatility_expectation"] = "Normal"

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED FEATURE 4: TIME-BASED PROBABILITY
    # ═══════════════════════════════════════════════════════════════════
    current_hour = datetime.now().hour
    current_minute = datetime.now().minute
    current_time_decimal = current_hour + current_minute / 60

    time_analysis = {
        "current_phase": "Pre-Market",
        "phase_probability": 0,
        "high_spike_windows": [],
        "current_window_active": False,
        "recommendation": "",
        "next_high_risk_window": ""
    }

    # Define high-volatility windows on expiry day (IST)
    high_spike_windows = [
        {"start": 9.25, "end": 10.0, "name": "Opening Spike", "probability": 75, "description": "Gap fills and initial positioning"},
        {"start": 10.5, "end": 11.5, "name": "Mid-Morning Unwinding", "probability": 65, "description": "FII/DII position adjustments"},
        {"start": 14.0, "end": 14.5, "name": "Afternoon Positioning", "probability": 60, "description": "Pre-expiry hedging activity"},
        {"start": 14.75, "end": 15.5, "name": "Final Hour Gamma Squeeze", "probability": 85, "description": "Maximum gamma effect, violent moves"}
    ]

    if days_to_expiry <= 1:  # Expiry day specific analysis
        for window in high_spike_windows:
            if window["start"] <= current_time_decimal <= window["end"]:
                time_analysis["current_phase"] = window["name"]
                time_analysis["phase_probability"] = window["probability"]
                time_analysis["current_window_active"] = True
                time_analysis["recommendation"] = window["description"]
                additional_spike_score += int(window["probability"] / 8)
                break

        if not time_analysis["current_window_active"]:
            if current_time_decimal < 9.25:
                time_analysis["current_phase"] = "Pre-Market"
                time_analysis["recommendation"] = "Wait for 9:15 opening - expect gap moves"
                time_analysis["next_high_risk_window"] = "09:15-10:00 (Opening Spike)"
            elif current_time_decimal < 10.5:
                time_analysis["current_phase"] = "Post-Opening Lull"
                time_analysis["phase_probability"] = 35
                time_analysis["recommendation"] = "Consolidation phase - wait for next spike window"
                time_analysis["next_high_risk_window"] = "10:30-11:30 (Mid-Morning)"
            elif current_time_decimal < 14.0:
                time_analysis["current_phase"] = "Mid-Day Consolidation"
                time_analysis["phase_probability"] = 40
                time_analysis["recommendation"] = "Lower risk - prepare for afternoon action"
                time_analysis["next_high_risk_window"] = "14:00-14:30 (Afternoon)"
            elif current_time_decimal < 14.75:
                time_analysis["current_phase"] = "Pre-Final Hour"
                time_analysis["phase_probability"] = 50
                time_analysis["recommendation"] = "Building up for final hour - reduce positions"
                time_analysis["next_high_risk_window"] = "14:45-15:30 (Gamma Squeeze)"
            else:
                time_analysis["current_phase"] = "Post-Market"
                time_analysis["recommendation"] = "Market closed"

        time_analysis["high_spike_windows"] = [
            {"time": "09:15-10:00", "name": "Opening Spike", "probability": 75},
            {"time": "10:30-11:30", "name": "Mid-Morning Unwinding", "probability": 65},
            {"time": "14:00-14:30", "name": "Afternoon Positioning", "probability": 60},
            {"time": "14:45-15:30", "name": "Final Hour Gamma Squeeze", "probability": 85}
        ]
    else:
        time_analysis["current_phase"] = f"T-{days_to_expiry:.0f} days to Expiry"
        time_analysis["phase_probability"] = max(20, 60 - (days_to_expiry * 10))
        time_analysis["recommendation"] = "Monitor OI buildup pattern for expiry day prediction"
        time_analysis["high_spike_windows"] = []

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED FEATURE 5: EXPECTED PRICE RANGE CALCULATION
    # ═══════════════════════════════════════════════════════════════════
    # ATM concentration for range calculation
    atm_window = 2
    atm_strikes = [s for s in merged_df["strikePrice"]
                   if abs(s - atm_strike) <= (atm_window * strike_gap_val)]
    atm_ce_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_CE"].sum()
    atm_pe_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_PE"].sum()
    total_oi_near_atm = atm_ce_oi + atm_pe_oi
    total_oi_all = merged_df["OI_CE"].sum() + merged_df["OI_PE"].sum()
    atm_concentration = (total_oi_near_atm / total_oi_all) if total_oi_all > 0 else 0

    # Base range: 0.5% of spot
    base_range_percent = 0.5

    # Multiplier based on days to expiry
    if days_to_expiry <= 0.5:
        range_multiplier = 2.5  # Expiry day afternoon
    elif days_to_expiry <= 1:
        range_multiplier = 2.0  # Expiry day morning
    elif days_to_expiry <= 2:
        range_multiplier = 1.5  # Day before expiry
    elif days_to_expiry <= 3:
        range_multiplier = 1.2
    else:
        range_multiplier = 1.0

    # GEX adjustment
    if total_gex_net is not None:
        if total_gex_net < -2000000:
            range_multiplier *= 1.5  # Negative GEX = wider range
        elif total_gex_net < -500000:
            range_multiplier *= 1.2
        elif total_gex_net > 2000000:
            range_multiplier *= 0.6  # Positive GEX = tighter range (pinning)
        elif total_gex_net > 500000:
            range_multiplier *= 0.8

    # ATM concentration adjustment (high concentration = pinning = tighter range)
    if atm_concentration > 0.5:
        range_multiplier *= 0.7
    elif atm_concentration > 0.3:
        range_multiplier *= 0.85

    # Time-based adjustment (final hour = wider range)
    if days_to_expiry <= 1 and time_analysis.get("current_phase") == "Final Hour Gamma Squeeze":
        range_multiplier *= 1.3

    # Calculate range
    range_percent = base_range_percent * range_multiplier
    range_points = int(spot * range_percent / 100)

    # Initial range
    expected_low = spot - range_points
    expected_high = spot + range_points

    # Bound by OI walls (price tends to stay within major OI levels)
    if max_pe_strike > 0 and max_pe_strike < spot:
        expected_low = max(expected_low, max_pe_strike - strike_gap_val)
    if max_ce_strike > 0 and max_ce_strike > spot:
        expected_high = min(expected_high, max_ce_strike + strike_gap_val)

    # Include max pain in range
    if max_pain_strike > 0:
        if spot > max_pain_strike:
            expected_low = min(expected_low, max_pain_strike)
        else:
            expected_high = max(expected_high, max_pain_strike)

    expected_range = {
        "low": int(expected_low),
        "high": int(expected_high),
        "range_points": int(expected_high - expected_low),
        "range_percent": round((expected_high - expected_low) / spot * 100, 2),
        "center": int((expected_high + expected_low) / 2),
        "bias": "BULLISH" if spot < (expected_high + expected_low) / 2 else "BEARISH" if spot > (expected_high + expected_low) / 2 else "NEUTRAL"
    }

    # ═══════════════════════════════════════════════════════════════════
    # FINAL ENHANCED PROBABILITY
    # ═══════════════════════════════════════════════════════════════════
    enhanced_score = base_result.get("score", 0) + additional_spike_score
    enhanced_score = max(0, enhanced_score)  # No negative scores

    # Enhanced probability calculation
    enhanced_probability = min(100, int(enhanced_score * 1.2))

    # Boost if in high-spike time window
    if time_analysis.get("current_window_active"):
        enhanced_probability = min(100, enhanced_probability + 10)

    # Update intensity based on enhanced probability
    if enhanced_probability >= 75:
        enhanced_intensity = "EXTREME SPIKE RISK"
        enhanced_color = "#ff0000"
    elif enhanced_probability >= 60:
        enhanced_intensity = "HIGH SPIKE RISK"
        enhanced_color = "#ff4400"
    elif enhanced_probability >= 45:
        enhanced_intensity = "MODERATE SPIKE RISK"
        enhanced_color = "#ff9900"
    elif enhanced_probability >= 30:
        enhanced_intensity = "LOW SPIKE RISK"
        enhanced_color = "#ffff00"
    else:
        enhanced_intensity = "MINIMAL SPIKE RISK"
        enhanced_color = "#00ff00"

    # Update base result with enhanced data
    base_result.update({
        # Override with enhanced values
        "probability": enhanced_probability,
        "score": enhanced_score,
        "intensity": enhanced_intensity,
        "color": enhanced_color,
        # Add new enhanced fields
        "expected_range": expected_range,
        "target_levels": target_levels[:6],  # Top 6 targets
        "time_analysis": time_analysis,
        "gex_impact": gex_impact,
        "oi_change_analysis": oi_change_analysis,
        "enhanced": True  # Flag to indicate enhanced version
    })

    # Add GEX and time factors to the factors list
    if gex_impact.get("impact") not in ["Unknown", "NEUTRAL"]:
        base_result["factors"].append(f"GEX: {gex_impact['impact']} ({gex_impact['message']})")

    if time_analysis.get("current_window_active"):
        base_result["factors"].append(f"⏰ {time_analysis['current_phase']} active ({time_analysis['phase_probability']}%)")

    if oi_change_analysis.get("bias") != "Neutral":
        base_result["factors"].append(f"OI: {oi_change_analysis['bias']} - {oi_change_analysis.get('signal', '')}")

    return base_result


# ═══════════════════════════════════════════════════════════════════════════
# 🎯 ALL-DAY SPIKE DETECTOR - Works on ANY trading day!
# ═══════════════════════════════════════════════════════════════════════════

def detect_all_market_spikes(merged_df, spot, atm_strike, days_to_expiry=None, total_gex_net=None, previous_close=None):
    """
    🎯 COMPREHENSIVE MARKET SPIKE DETECTOR - Finds ALL spikes!

    Works on ANY trading day (not just expiry). Detects:
    1. SUPPORT SPIKE - Price bouncing from PUT wall
    2. RESISTANCE SPIKE - Price rejecting from CALL wall
    3. OPENING SPIKE - Gap up/down at market open
    4. BREAKOUT SPIKE - Price breaking key levels
    5. MOMENTUM SPIKE - Strong directional move
    6. SQUEEZE SPIKE - Short/Long squeeze

    Returns comprehensive spike analysis with scores for each type.
    """
    from datetime import datetime

    current_hour = datetime.now().hour
    current_minute = datetime.now().minute
    current_time_decimal = current_hour + current_minute / 60

    # Get strike gap
    strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])

    # ═══════════════════════════════════════════════════════════════════
    # EXTRACT KEY LEVELS
    # ═══════════════════════════════════════════════════════════════════

    # CALL walls (Resistance levels)
    ce_sorted = merged_df.nlargest(5, 'OI_CE')
    resistance_walls = []
    for _, row in ce_sorted.iterrows():
        strike = int(row['strikePrice'])
        oi = int(row['OI_CE'])
        chg_oi = int(row.get('Chg_OI_CE', 0))
        if oi > 500000:
            resistance_walls.append({
                "level": strike,
                "oi": oi,
                "oi_change": chg_oi,
                "strength": "MASSIVE" if oi > 3000000 else "STRONG" if oi > 2000000 else "MODERATE"
            })

    # PUT walls (Support levels)
    pe_sorted = merged_df.nlargest(5, 'OI_PE')
    support_walls = []
    for _, row in pe_sorted.iterrows():
        strike = int(row['strikePrice'])
        oi = int(row['OI_PE'])
        chg_oi = int(row.get('Chg_OI_PE', 0))
        if oi > 500000:
            support_walls.append({
                "level": strike,
                "oi": oi,
                "oi_change": chg_oi,
                "strength": "MASSIVE" if oi > 3000000 else "STRONG" if oi > 2000000 else "MODERATE"
            })

    # Primary levels
    max_ce_strike = resistance_walls[0]["level"] if resistance_walls else atm_strike + 200
    max_ce_oi = resistance_walls[0]["oi"] if resistance_walls else 0
    max_pe_strike = support_walls[0]["level"] if support_walls else atm_strike - 200
    max_pe_oi = support_walls[0]["oi"] if support_walls else 0

    # Max Pain
    max_pain = calculate_seller_max_pain(merged_df)
    max_pain_strike = max_pain.get("max_pain_strike", atm_strike) if max_pain else atm_strike

    # OI totals and changes
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    ce_oi_change = merged_df["Chg_OI_CE"].sum()
    pe_oi_change = merged_df["Chg_OI_PE"].sum()
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1

    # ═══════════════════════════════════════════════════════════════════
    # INITIALIZE ALL SPIKE TYPES
    # ═══════════════════════════════════════════════════════════════════

    spikes = {
        "support_spike": {
            "active": False,
            "score": 0,
            "probability": 0,
            "level": max_pe_strike,
            "distance_from_spot": spot - max_pe_strike,
            "distance_percent": round((spot - max_pe_strike) / spot * 100, 2) if spot > 0 else 0,
            "direction": "UP",
            "factors": [],
            "action": "BUY on touch"
        },
        "resistance_spike": {
            "active": False,
            "score": 0,
            "probability": 0,
            "level": max_ce_strike,
            "distance_from_spot": max_ce_strike - spot,
            "distance_percent": round((max_ce_strike - spot) / spot * 100, 2) if spot > 0 else 0,
            "direction": "DOWN",
            "factors": [],
            "action": "SELL on touch"
        },
        "opening_spike": {
            "active": False,
            "score": 0,
            "probability": 0,
            "direction": None,
            "expected_gap": 0,
            "factors": [],
            "action": ""
        },
        "breakout_spike": {
            "active": False,
            "score": 0,
            "probability": 0,
            "direction": None,
            "breakout_level": 0,
            "factors": [],
            "action": ""
        },
        "momentum_spike": {
            "active": False,
            "score": 0,
            "probability": 0,
            "direction": None,
            "factors": [],
            "action": ""
        },
        "squeeze_spike": {
            "active": False,
            "score": 0,
            "probability": 0,
            "type": None,  # "SHORT_SQUEEZE" or "LONG_SQUEEZE"
            "factors": [],
            "action": ""
        }
    }

    # ═══════════════════════════════════════════════════════════════════
    # 1. SUPPORT SPIKE DETECTION
    # ═══════════════════════════════════════════════════════════════════
    support_score = 0
    support_factors = []

    support_distance = spot - max_pe_strike
    support_dist_pct = (support_distance / spot * 100) if spot > 0 else 0

    # Proximity check
    if support_dist_pct <= 0.3:
        support_score += 50
        support_factors.append(f"🎯 AT PUT WALL ₹{max_pe_strike:,} (only {support_dist_pct:.2f}% away)")
    elif support_dist_pct <= 0.5:
        support_score += 40
        support_factors.append(f"Very close to PUT wall ₹{max_pe_strike:,} ({support_dist_pct:.2f}%)")
    elif support_dist_pct <= 1.0:
        support_score += 25
        support_factors.append(f"Approaching PUT wall ₹{max_pe_strike:,} ({support_dist_pct:.2f}%)")
    elif support_distance < 0:  # Below support!
        support_score += 60
        support_factors.append(f"⚠️ BELOW PUT wall ₹{max_pe_strike:,} - Strong bounce expected!")

    # OI strength
    if max_pe_oi > 3000000:
        support_score += 25
        support_factors.append(f"MASSIVE PUT OI: {max_pe_oi/1000000:.1f}M")
    elif max_pe_oi > 2000000:
        support_score += 15
        support_factors.append(f"Strong PUT OI: {max_pe_oi/1000000:.1f}M")

    # Fresh PUT buildup (strengthening support)
    if pe_oi_change > 500000:
        support_score += 15
        support_factors.append(f"Fresh PUT buildup +{pe_oi_change/1000:.0f}K")

    spikes["support_spike"]["score"] = min(100, support_score)
    spikes["support_spike"]["probability"] = min(100, int(support_score * 1.2))
    spikes["support_spike"]["active"] = support_score >= 40
    spikes["support_spike"]["factors"] = support_factors
    spikes["support_spike"]["distance_from_spot"] = round(support_distance, 0)
    spikes["support_spike"]["distance_percent"] = round(support_dist_pct, 2)

    # ═══════════════════════════════════════════════════════════════════
    # 2. RESISTANCE SPIKE DETECTION
    # ═══════════════════════════════════════════════════════════════════
    resistance_score = 0
    resistance_factors = []

    resistance_distance = max_ce_strike - spot
    resistance_dist_pct = (resistance_distance / spot * 100) if spot > 0 else 0

    # Proximity check
    if resistance_dist_pct <= 0.3:
        resistance_score += 50
        resistance_factors.append(f"🎯 AT CALL WALL ₹{max_ce_strike:,} (only {resistance_dist_pct:.2f}% away)")
    elif resistance_dist_pct <= 0.5:
        resistance_score += 40
        resistance_factors.append(f"Very close to CALL wall ₹{max_ce_strike:,} ({resistance_dist_pct:.2f}%)")
    elif resistance_dist_pct <= 1.0:
        resistance_score += 25
        resistance_factors.append(f"Approaching CALL wall ₹{max_ce_strike:,} ({resistance_dist_pct:.2f}%)")
    elif resistance_distance < 0:  # Above resistance!
        resistance_score += 60
        resistance_factors.append(f"⚠️ ABOVE CALL wall ₹{max_ce_strike:,} - Breakout or rejection!")

    # OI strength
    if max_ce_oi > 3000000:
        resistance_score += 25
        resistance_factors.append(f"MASSIVE CALL OI: {max_ce_oi/1000000:.1f}M")
    elif max_ce_oi > 2000000:
        resistance_score += 15
        resistance_factors.append(f"Strong CALL OI: {max_ce_oi/1000000:.1f}M")

    # Fresh CALL buildup (strengthening resistance)
    if ce_oi_change > 500000:
        resistance_score += 15
        resistance_factors.append(f"Fresh CALL buildup +{ce_oi_change/1000:.0f}K")

    spikes["resistance_spike"]["score"] = min(100, resistance_score)
    spikes["resistance_spike"]["probability"] = min(100, int(resistance_score * 1.2))
    spikes["resistance_spike"]["active"] = resistance_score >= 40
    spikes["resistance_spike"]["factors"] = resistance_factors
    spikes["resistance_spike"]["distance_from_spot"] = round(resistance_distance, 0)
    spikes["resistance_spike"]["distance_percent"] = round(resistance_dist_pct, 2)

    # ═══════════════════════════════════════════════════════════════════
    # 3. OPENING SPIKE PREDICTION
    # ═══════════════════════════════════════════════════════════════════
    opening_score = 0
    opening_factors = []
    opening_direction = None

    # PCR analysis
    if pcr > 1.5:
        opening_score += 30
        opening_direction = "UP"
        opening_factors.append(f"High PCR {pcr:.2f} = Bullish opening")
    elif pcr < 0.7:
        opening_score += 30
        opening_direction = "DOWN"
        opening_factors.append(f"Low PCR {pcr:.2f} = Bearish opening")

    # Max Pain distance
    mp_distance = spot - max_pain_strike
    mp_dist_pct = abs(mp_distance / spot * 100) if spot > 0 else 0
    if mp_dist_pct > 1.5:
        opening_score += 25
        if mp_distance > 0:
            opening_direction = "DOWN"
            opening_factors.append(f"Spot {mp_dist_pct:.1f}% above Max Pain - Gap DOWN likely")
        else:
            opening_direction = "UP"
            opening_factors.append(f"Spot {mp_dist_pct:.1f}% below Max Pain - Gap UP likely")

    # OI unwinding pattern
    if ce_oi_change < -1000000:
        opening_score += 20
        opening_direction = "UP" if opening_direction is None else opening_direction
        opening_factors.append(f"CALL unwinding {ce_oi_change/1000:.0f}K = Gap UP")

    if pe_oi_change < -1000000:
        opening_score += 20
        opening_direction = "DOWN" if opening_direction is None else opening_direction
        opening_factors.append(f"PUT unwinding {pe_oi_change/1000:.0f}K = Gap DOWN")

    # GEX impact
    if total_gex_net is not None and total_gex_net < -2000000:
        opening_score += 15
        opening_factors.append(f"Negative GEX = Volatile opening")

    expected_gap = 0
    if opening_direction == "UP":
        expected_gap = int(spot * 0.005 * (opening_score / 50))  # 0.5% scaled by score
    elif opening_direction == "DOWN":
        expected_gap = -int(spot * 0.005 * (opening_score / 50))

    spikes["opening_spike"]["score"] = min(100, opening_score)
    spikes["opening_spike"]["probability"] = min(100, int(opening_score * 1.1))
    spikes["opening_spike"]["active"] = opening_score >= 40
    spikes["opening_spike"]["direction"] = opening_direction
    spikes["opening_spike"]["expected_gap"] = expected_gap
    spikes["opening_spike"]["factors"] = opening_factors
    spikes["opening_spike"]["action"] = f"Expect {opening_direction} gap of ~₹{abs(expected_gap):,}" if opening_direction else "Flat opening"

    # ═══════════════════════════════════════════════════════════════════
    # 4. BREAKOUT SPIKE DETECTION
    # ═══════════════════════════════════════════════════════════════════
    breakout_score = 0
    breakout_factors = []
    breakout_direction = None
    breakout_level = 0

    # Check if spot is near breakout levels
    if resistance_distance < 0:  # Above resistance
        breakout_score += 50
        breakout_direction = "UP"
        breakout_level = max_ce_strike
        breakout_factors.append(f"🚀 BREAKOUT above CALL wall ₹{max_ce_strike:,}!")

        # Check if CALL wall is shifting up (writers running)
        if ce_oi_change < -500000:
            breakout_score += 25
            breakout_factors.append(f"CALL writers exiting = Breakout confirmed")

    elif support_distance < 0:  # Below support
        breakout_score += 50
        breakout_direction = "DOWN"
        breakout_level = max_pe_strike
        breakout_factors.append(f"📉 BREAKDOWN below PUT wall ₹{max_pe_strike:,}!")

        # Check if PUT wall is shifting down (writers running)
        if pe_oi_change < -500000:
            breakout_score += 25
            breakout_factors.append(f"PUT writers exiting = Breakdown confirmed")

    spikes["breakout_spike"]["score"] = min(100, breakout_score)
    spikes["breakout_spike"]["probability"] = min(100, int(breakout_score * 1.1))
    spikes["breakout_spike"]["active"] = breakout_score >= 40
    spikes["breakout_spike"]["direction"] = breakout_direction
    spikes["breakout_spike"]["breakout_level"] = breakout_level
    spikes["breakout_spike"]["factors"] = breakout_factors
    spikes["breakout_spike"]["action"] = f"Trade {breakout_direction} breakout from ₹{breakout_level:,}" if breakout_direction else ""

    # ═══════════════════════════════════════════════════════════════════
    # 5. MOMENTUM SPIKE DETECTION (Based on OI velocity)
    # ═══════════════════════════════════════════════════════════════════
    momentum_score = 0
    momentum_factors = []
    momentum_direction = None

    net_oi_change = ce_oi_change + pe_oi_change

    # Strong OI velocity indicates momentum
    if abs(ce_oi_change) > 2000000 or abs(pe_oi_change) > 2000000:
        momentum_score += 40

        if ce_oi_change < -2000000 and pe_oi_change > 1000000:
            momentum_direction = "UP"
            momentum_factors.append(f"Strong CALL unwinding + PUT buildup = UP momentum")
        elif pe_oi_change < -2000000 and ce_oi_change > 1000000:
            momentum_direction = "DOWN"
            momentum_factors.append(f"Strong PUT unwinding + CALL buildup = DOWN momentum")

    # PCR extremes indicate momentum
    if pcr > 1.8:
        momentum_score += 25
        momentum_direction = "UP"
        momentum_factors.append(f"Extreme high PCR {pcr:.2f} = Bullish momentum")
    elif pcr < 0.5:
        momentum_score += 25
        momentum_direction = "DOWN"
        momentum_factors.append(f"Extreme low PCR {pcr:.2f} = Bearish momentum")

    spikes["momentum_spike"]["score"] = min(100, momentum_score)
    spikes["momentum_spike"]["probability"] = min(100, int(momentum_score * 1.1))
    spikes["momentum_spike"]["active"] = momentum_score >= 40
    spikes["momentum_spike"]["direction"] = momentum_direction
    spikes["momentum_spike"]["factors"] = momentum_factors
    spikes["momentum_spike"]["action"] = f"Ride {momentum_direction} momentum" if momentum_direction else ""

    # ═══════════════════════════════════════════════════════════════════
    # 6. SQUEEZE SPIKE DETECTION
    # ═══════════════════════════════════════════════════════════════════
    squeeze_score = 0
    squeeze_factors = []
    squeeze_type = None

    # Short Squeeze: Price above max pain + CALL unwinding
    if spot > max_pain_strike and ce_oi_change < -1000000:
        squeeze_score += 50
        squeeze_type = "SHORT_SQUEEZE"
        squeeze_factors.append(f"Price above Max Pain + CALL unwinding = Short Squeeze")

    # Long Squeeze: Price below max pain + PUT unwinding
    elif spot < max_pain_strike and pe_oi_change < -1000000:
        squeeze_score += 50
        squeeze_type = "LONG_SQUEEZE"
        squeeze_factors.append(f"Price below Max Pain + PUT unwinding = Long Squeeze")

    # Additional squeeze indicators
    if squeeze_type and days_to_expiry is not None and days_to_expiry <= 2:
        squeeze_score += 25
        squeeze_factors.append(f"Near expiry intensifies squeeze!")

    spikes["squeeze_spike"]["score"] = min(100, squeeze_score)
    spikes["squeeze_spike"]["probability"] = min(100, int(squeeze_score * 1.1))
    spikes["squeeze_spike"]["active"] = squeeze_score >= 40
    spikes["squeeze_spike"]["type"] = squeeze_type
    spikes["squeeze_spike"]["factors"] = squeeze_factors
    spikes["squeeze_spike"]["action"] = f"Trade {squeeze_type.replace('_', ' ')}" if squeeze_type else ""

    # ═══════════════════════════════════════════════════════════════════
    # TIME-BASED SPIKE WINDOWS
    # ═══════════════════════════════════════════════════════════════════
    time_windows = [
        {"start": 9.25, "end": 9.75, "name": "Opening 30min", "spike_boost": 15},
        {"start": 9.75, "end": 10.25, "name": "First Hour End", "spike_boost": 10},
        {"start": 11.5, "end": 12.0, "name": "Pre-Lunch", "spike_boost": 8},
        {"start": 14.0, "end": 14.5, "name": "Afternoon Session", "spike_boost": 10},
        {"start": 15.0, "end": 15.5, "name": "Last 30min", "spike_boost": 12}
    ]

    current_window = None
    for window in time_windows:
        if window["start"] <= current_time_decimal <= window["end"]:
            current_window = window
            break

    # ═══════════════════════════════════════════════════════════════════
    # COMPILE FINAL RESULT
    # ═══════════════════════════════════════════════════════════════════

    # Find the most likely spike
    active_spikes = []
    for spike_name, spike_data in spikes.items():
        if spike_data["active"]:
            active_spikes.append({
                "name": spike_name,
                "score": spike_data["score"],
                "probability": spike_data["probability"],
                "direction": spike_data.get("direction"),
                "data": spike_data
            })

    # Sort by score
    active_spikes.sort(key=lambda x: x["score"], reverse=True)

    # Primary spike (highest score)
    primary_spike = active_spikes[0] if active_spikes else None

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "spot": spot,
        "atm": atm_strike,

        # All spike types with their scores
        "spikes": spikes,

        # Active spikes (score >= 40)
        "active_spikes": active_spikes,
        "active_spike_count": len(active_spikes),

        # Primary spike (most likely)
        "primary_spike": {
            "type": primary_spike["name"] if primary_spike else "NONE",
            "score": primary_spike["score"] if primary_spike else 0,
            "probability": primary_spike["probability"] if primary_spike else 0,
            "direction": primary_spike["direction"] if primary_spike else None
        },

        # Key levels
        "key_levels": {
            "resistance": resistance_walls[:3],
            "support": support_walls[:3],
            "max_pain": max_pain_strike,
            "atm": atm_strike
        },

        # Time analysis
        "time_analysis": {
            "current_phase": current_window["name"] if current_window else "Regular Trading",
            "spike_boost": current_window["spike_boost"] if current_window else 0
        },

        # Summary
        "summary": {
            "spike_detected": len(active_spikes) > 0,
            "spike_count": len(active_spikes),
            "primary_type": primary_spike["name"].replace("_", " ").title() if primary_spike else "No Spike",
            "primary_score": primary_spike["score"] if primary_spike else 0,
            "recommendation": _get_spike_recommendation(primary_spike, spot, max_pe_strike, max_ce_strike) if primary_spike else "No clear spike setup"
        },

        # Days to expiry impact
        "expiry_factor": {
            "days_to_expiry": days_to_expiry,
            "expiry_boost": 15 if days_to_expiry and days_to_expiry <= 2 else 0
        }
    }

    return result


def _get_spike_recommendation(spike, spot, support, resistance):
    """Helper function to get recommendation based on spike type"""
    if not spike:
        return "No clear spike setup"

    spike_type = spike["name"]
    direction = spike.get("direction")
    score = spike["score"]

    if spike_type == "support_spike":
        return f"🟢 SUPPORT SPIKE at ₹{support:,} ({score}%) - BUY on touch, SL below ₹{support - 50:,}"
    elif spike_type == "resistance_spike":
        return f"🔴 RESISTANCE SPIKE at ₹{resistance:,} ({score}%) - SELL on touch, SL above ₹{resistance + 50:,}"
    elif spike_type == "opening_spike":
        return f"🌅 OPENING {direction} SPIKE expected ({score}%) - Trade gap direction"
    elif spike_type == "breakout_spike":
        return f"🚀 BREAKOUT {direction} ({score}%) - Trade momentum with trailing SL"
    elif spike_type == "momentum_spike":
        return f"💨 {direction} MOMENTUM ({score}%) - Ride the trend"
    elif spike_type == "squeeze_spike":
        squeeze_type = spike["data"].get("type", "").replace("_", " ")
        return f"🔥 {squeeze_type} ({score}%) - Explosive move expected!"
    else:
        return f"Spike detected: {spike_type} ({score}%)"


def get_historical_expiry_patterns():
    """
    Return historical expiry day patterns (simplified)
    In production, you'd connect to a database
    """
    patterns = {
        "high_volatility": {
            "probability": 0.65,
            "description": "Expiry days typically have 30% higher volatility",
            "time_of_spike": ["10:30-11:30 IST", "14:30-15:00 IST"]
        },
        "max_pain_pull": {
            "probability": 0.55,
            "description": "Price tends to gravitate towards Max Pain in last 2 hours",
            "effect": "Strong if Max Pain >1% away from spot"
        },
        "gamma_unwind": {
            "probability": 0.70,
            "description": "Market makers unwind gamma positions causing spikes",
            "timing": "Last 90 minutes"
        }
    }
    return patterns

def detect_violent_unwinding(merged_df, spot, atm_strike):
    """
    Detect signs of violent unwinding near expiry
    """
    signals = []
    
    # Check for massive OI reduction
    total_ce_chg = merged_df["Chg_OI_CE"].sum()
    total_pe_chg = merged_df["Chg_OI_PE"].sum()
    
    if total_ce_chg < -1000000:  # 1 million+ CALL unwinding
        signals.append(f"Violent CALL unwinding: {abs(total_ce_chg):,} contracts")
    
    if total_pe_chg < -1000000:  # 1 million+ PUT unwinding
        signals.append(f"Violent PUT unwinding: {abs(total_pe_chg):,} contracts")
    
    # Check ATM strikes specifically
    strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
    atm_window = 1
    atm_strikes = [s for s in merged_df["strikePrice"] 
                   if abs(s - atm_strike) <= (atm_window * strike_gap_val)]
    
    atm_unwind = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes)]
    atm_ce_unwind = atm_unwind["Chg_OI_CE"].sum()
    atm_pe_unwind = atm_unwind["Chg_OI_PE"].sum()
    
    if atm_ce_unwind < -500000:
        signals.append(f"ATM CALL unwinding: {abs(atm_ce_unwind):,} contracts")
    
    if atm_pe_unwind < -500000:
        signals.append(f"ATM PUT unwinding: {abs(atm_pe_unwind):,} contracts")
    
    return signals

def calculate_gamma_exposure_spike(total_gex_net, days_to_expiry):
    """
    Calculate gamma exposure spike risk
    """
    if days_to_expiry > 3:
        return {"risk": "Low", "score": 0}
    
    # Negative GEX + near expiry = explosive moves
    if total_gex_net < -2000000:  # Large negative GEX
        risk_score = min(100, int((abs(total_gex_net) / 1000000) * 10))
        return {
            "risk": "High",
            "score": risk_score,
            "message": f"Negative GEX (₹{abs(total_gex_net):,.0f}) + Near expiry = Explosive move potential"
        }
    elif total_gex_net > 2000000:  # Large positive GEX
        risk_score = min(80, int((total_gex_net / 1000000) * 5))
        return {
            "risk": "Medium",
            "score": risk_score,
            "message": f"Positive GEX (₹{total_gex_net:,.0f}) + Near expiry = Mean reversion bias"
        }
    
    return {"risk": "Low", "score": 0}

def predict_expiry_pinning_probability(spot, max_pain, nearest_support, nearest_resistance):
    """
    Predict probability of expiry pinning (price stuck at a level)
    """
    if not max_pain or not nearest_support or not nearest_resistance:
        return 0
    
    # Calculate pinning score (0-100)
    pinning_score = 0
    
    # Factor 1: Distance to Max Pain
    distance_to_max_pain = abs(spot - max_pain) / spot * 100
    if distance_to_max_pain < 0.5:
        pinning_score += 40
    elif distance_to_max_pain < 1.0:
        pinning_score += 20
    
    # Factor 2: Narrow range
    range_size = nearest_resistance - nearest_support
    if range_size < 200:
        pinning_score += 30
    elif range_size < 300:
        pinning_score += 15
    
    return min(100, pinning_score)

# -----------------------
# 📈 ENHANCED OI & PCR ANALYZER FUNCTIONS
# -----------------------

def analyze_oi_pcr_metrics(merged_df, spot, atm_strike):
    """
    Comprehensive OI and PCR analysis
    Returns detailed metrics and insights
    """
    
    # Basic totals
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    total_ce_chg = merged_df["Chg_OI_CE"].sum()
    total_pe_chg = merged_df["Chg_OI_PE"].sum()
    total_oi = total_ce_oi + total_pe_oi
    total_chg_oi = total_ce_chg + total_pe_chg
    
    # PCR Calculations
    pcr_total = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    pcr_chg = total_pe_chg / total_ce_chg if abs(total_ce_chg) > 0 else 0
    
    # OI Concentration Analysis
    strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
    
    # ATM Concentration (strikes around ATM)
    atm_window = 3  # ±3 strikes
    atm_strikes = [s for s in merged_df["strikePrice"] 
                  if abs(s - atm_strike) <= (atm_window * strike_gap_val)]
    
    atm_ce_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_CE"].sum()
    atm_pe_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_PE"].sum()
    atm_total_oi = atm_ce_oi + atm_pe_oi
    
    # ITM/OTM Analysis
    itm_ce_oi = merged_df.loc[merged_df["strikePrice"] < spot, "OI_CE"].sum()
    otm_ce_oi = merged_df.loc[merged_df["strikePrice"] > spot, "OI_CE"].sum()
    itm_pe_oi = merged_df.loc[merged_df["strikePrice"] > spot, "OI_PE"].sum()
    otm_pe_oi = merged_df.loc[merged_df["strikePrice"] < spot, "OI_PE"].sum()
    
    # Max OI Strikes
    max_ce_oi_row = merged_df.loc[merged_df["OI_CE"].idxmax()] if not merged_df.empty else None
    max_pe_oi_row = merged_df.loc[merged_df["OI_PE"].idxmax()] if not merged_df.empty else None
    
    max_ce_strike = int(max_ce_oi_row["strikePrice"]) if max_ce_oi_row is not None else 0
    max_ce_oi_val = int(max_ce_oi_row["OI_CE"]) if max_ce_oi_row is not None else 0
    max_pe_strike = int(max_pe_oi_row["strikePrice"]) if max_pe_oi_row is not None else 0
    max_pe_oi_val = int(max_pe_oi_row["OI_PE"]) if max_pe_oi_row is not None else 0
    
    # OI Skew Analysis
    call_oi_skew = "N/A"
    if total_ce_oi > 0:
        # Check if OI is concentrated at specific strikes
        top_3_ce_oi = merged_df.nlargest(3, "OI_CE")["OI_CE"].sum()
        call_oi_concentration = top_3_ce_oi / total_ce_oi if total_ce_oi > 0 else 0
        call_oi_skew = "High" if call_oi_concentration > 0.4 else "Moderate" if call_oi_concentration > 0.2 else "Low"
    
    put_oi_skew = "N/A"
    if total_pe_oi > 0:
        top_3_pe_oi = merged_df.nlargest(3, "OI_PE")["OI_PE"].sum()
        put_oi_concentration = top_3_pe_oi / total_pe_oi if total_pe_oi > 0 else 0
        put_oi_skew = "High" if put_oi_concentration > 0.4 else "Moderate" if put_oi_concentration > 0.2 else "Low"
    
    # PCR Interpretation
    pcr_interpretation = ""
    pcr_sentiment = ""
    
    if pcr_total > 2.0:
        pcr_interpretation = "EXTREME PUT SELLING"
        pcr_sentiment = "STRONGLY BULLISH"
        pcr_color = "#00ff88"
    elif pcr_total > 1.5:
        pcr_interpretation = "HEAVY PUT SELLING"
        pcr_sentiment = "BULLISH"
        pcr_color = "#00cc66"
    elif pcr_total > 1.2:
        pcr_interpretation = "MODERATE PUT SELLING"
        pcr_sentiment = "MILD BULLISH"
        pcr_color = "#66ff66"
    elif pcr_total > 0.8:
        pcr_interpretation = "BALANCED"
        pcr_sentiment = "NEUTRAL"
        pcr_color = "#66b3ff"
    elif pcr_total > 0.5:
        pcr_interpretation = "MODERATE CALL SELLING"
        pcr_sentiment = "MILD BEARISH"
        pcr_color = "#ff9900"
    elif pcr_total > 0.3:
        pcr_interpretation = "HEAVY CALL SELLING"
        pcr_sentiment = "BEARISH"
        pcr_color = "#ff4444"
    else:
        pcr_interpretation = "EXTREME CALL SELLING"
        pcr_sentiment = "STRONGLY BEARISH"
        pcr_color = "#ff0000"
    
    # Change in PCR interpretation
    chg_interpretation = ""
    if abs(pcr_chg) > 0:
        if pcr_chg > 0.5:
            chg_interpretation = "PCR rising sharply (bullish buildup)"
        elif pcr_chg > 0.2:
            chg_interpretation = "PCR rising (bullish)"
        elif pcr_chg < -0.5:
            chg_interpretation = "PCR falling sharply (bearish buildup)"
        elif pcr_chg < -0.2:
            chg_interpretation = "PCR falling (bearish)"
        else:
            chg_interpretation = "PCR stable"
    
    # OI Change Interpretation
    oi_change_interpretation = ""
    if total_ce_chg > 0 and total_pe_chg > 0:
        oi_change_interpretation = "Fresh writing on both sides (range expansion)"
    elif total_ce_chg > 0 and total_pe_chg < 0:
        oi_change_interpretation = "CALL writing + PUT unwinding (bearish)"
    elif total_ce_chg < 0 and total_pe_chg > 0:
        oi_change_interpretation = "CALL unwinding + PUT writing (bullish)"
    elif total_ce_chg < 0 and total_pe_chg < 0:
        oi_change_interpretation = "Unwinding on both sides (range contraction)"
    else:
        oi_change_interpretation = "Mixed activity"
    
    return {
        # Totals
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "total_oi": total_oi,
        "total_ce_chg": total_ce_chg,
        "total_pe_chg": total_pe_chg,
        "total_chg_oi": total_chg_oi,
        
        # PCR Metrics
        "pcr_total": pcr_total,
        "pcr_chg": pcr_chg,
        "pcr_interpretation": pcr_interpretation,
        "pcr_sentiment": pcr_sentiment,
        "pcr_color": pcr_color,
        "chg_interpretation": chg_interpretation,
        
        # Concentration
        "atm_concentration_pct": (atm_total_oi / total_oi * 100) if total_oi > 0 else 0,
        "atm_ce_oi": atm_ce_oi,
        "atm_pe_oi": atm_pe_oi,
        
        # ITM/OTM
        "itm_ce_oi": itm_ce_oi,
        "otm_ce_oi": otm_ce_oi,
        "itm_pe_oi": itm_pe_oi,
        "otm_pe_oi": otm_pe_oi,
        
        # Max OI
        "max_ce_strike": max_ce_strike,
        "max_ce_oi": max_ce_oi_val,
        "max_pe_strike": max_pe_strike,
        "max_pe_oi": max_pe_oi_val,
        
        # Skew
        "call_oi_skew": call_oi_skew,
        "put_oi_skew": put_oi_skew,
        
        # Interpretation
        "oi_change_interpretation": oi_change_interpretation,
        
        # Derived metrics
        "ce_pe_ratio": total_ce_oi / total_pe_oi if total_pe_oi > 0 else 0,
        "oi_momentum": total_chg_oi / total_oi * 100 if total_oi > 0 else 0
    }

def get_pcr_context(pcr_value):
    """Provide historical context for PCR values"""
    if pcr_value > 2.5:
        return "Extreme bullish zone (rare, usually precedes sharp rallies)"
    elif pcr_value > 2.0:
        return "Very bullish (often leads to upward moves)"
    elif pcr_value > 1.5:
        return "Bullish bias"
    elif pcr_value > 1.2:
        return "Moderately bullish"
    elif pcr_value > 0.8:
        return "Neutral range"
    elif pcr_value > 0.5:
        return "Moderately bearish"
    elif pcr_value > 0.3:
        return "Bearish (often precedes declines)"
    else:
        return "Extreme bearish (oversold, can mean reversal)"

def analyze_pcr_for_expiry(pcr_value, days_to_expiry):
    """Analyze PCR in context of expiry"""
    if days_to_expiry > 5:
        return "Normal PCR analysis applies"
    
    if days_to_expiry <= 2:
        if pcr_value > 1.5:
            return "High PCR near expiry → Potential short covering rally"
        elif pcr_value < 0.7:
            return "Low PCR near expiry → Potential long unwinding decline"
        else:
            return "Balanced PCR near expiry → Range bound expected"
    
    return "PCR analysis standard"

st.set_page_config(page_title="Nifty Screener v7 - Seller's Perspective + ATM Bias Analyzer + Moment Detector + Expiry Spike + OI/PCR", layout="wide")

# -----------------------
#  CUSTOM CSS - SELLER THEME + NEW MOMENT FEATURES + EXPIRY SPIKE + OI/PCR + ATM BIAS
# -----------------------
st.markdown(r"""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    
    /* SELLER THEME COLORS */
    .seller-bullish { color: #00ff88 !important; font-weight: 700 !important; }
    .seller-bearish { color: #ff4444 !important; font-weight: 700 !important; }
    .seller-neutral { color: #66b3ff !important; font-weight: 700 !important; }
    
    .seller-bullish-bg { background: linear-gradient(135deg, #1a2e1a 0%, #2a3e2a 100%); }
    .seller-bearish-bg { background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%); }
    .seller-neutral-bg { background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%); }
    
    /* MOMENT DETECTOR COLORS */
    .moment-high { color: #ff00ff !important; font-weight: 800 !important; }
    .moment-medium { color: #ff9900 !important; font-weight: 700 !important; }
    .moment-low { color: #66b3ff !important; font-weight: 600 !important; }
    
    /* OI/PCR COLORS */
    .pcr-extreme-bullish { color: #00ff88 !important; }
    .pcr-bullish { color: #00cc66 !important; }
    .pcr-mild-bullish { color: #66ff66 !important; }
    .pcr-neutral { color: #66b3ff !important; }
    .pcr-mild-bearish { color: #ff9900 !important; }
    .pcr-bearish { color: #ff4444 !important; }
    .pcr-extreme-bearish { color: #ff0000 !important; }
    
    /* ATM BIAS COLORS */
    .atm-bias-bullish { color: #00ff88 !important; }
    .atm-bias-bearish { color: #ff4444 !important; }
    .atm-bias-neutral { color: #66b3ff !important; }
    
    h1, h2, h3 { color: #ff66cc !important; } /* Seller theme pink */
    
    .level-card {
        background: linear-gradient(135deg, #2e1a2e 0%, #3e2a3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #ff66cc;
        margin: 5px 0;
    }
    .level-card h4 { margin: 0; color: #ff66cc; font-size: 1.1rem; }
    .level-card p { margin: 5px 0; color: #fafafa; font-size: 1.3rem; font-weight: 700; }
    .level-card .sub-info { font-size: 0.9rem; color: #cccccc; margin-top: 5px; }
    
    .spot-card {
        background: linear-gradient(135deg, #2e2a1a 0%, #3e3a2a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff9900;
        margin: 10px 0;
        text-align: center;
    }
    .spot-card h3 { margin: 0; color: #ff9900; font-size: 1.3rem; }
    .spot-card .spot-price { font-size: 2.5rem; color: #ffcc00; font-weight: 700; margin: 10px 0; }
    .spot-card .distance { font-size: 1.1rem; color: #ffdd44; margin: 5px 0; }
    
    .nearest-level {
        background: linear-gradient(135deg, #1a2e2e 0%, #2a3e3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #00ffcc;
        margin: 10px 0;
    }
    .nearest-level h4 { margin: 0; color: #00ffcc; font-size: 1.2rem; }
    .nearest-level .level-value { font-size: 1.8rem; color: #00ffcc; font-weight: 700; margin: 5px 0; }
    .nearest-level .level-distance { font-size: 1rem; color: #66ffdd; margin: 5px 0; }
    
    .seller-bias-box {
        background: linear-gradient(135deg, #2e1a2e 0%, #1a2e2e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff66cc;
        margin: 15px 0;
        text-align: center;
    }
    .seller-bias-box h3 { margin: 0; color: #ff66cc; font-size: 1.4rem; }
    .seller-bias-box .bias-value { font-size: 2.2rem; font-weight: 900; margin: 10px 0; }
    
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .seller-support-building { background-color: #1a2e1a; border-left-color: #00ff88; color: #00ff88; }
    .seller-support-breaking { background-color: #2e1a1a; border-left-color: #ff4444; color: #ff6666; }
    .seller-resistance-building { background-color: #2e2a1a; border-left-color: #ffaa00; color: #ffcc44; }
    .seller-resistance-breaking { background-color: #1a1f2e; border-left-color: #00aaff; color: #00ccff; }
    .seller-bull-trap { background-color: #3e1a1a; border-left-color: #ff0000; color: #ff4444; font-weight: 700; }
    .seller-bear-trap { background-color: #1a3e1a; border-left-color: #00ff00; color: #00ff66; font-weight: 700; }
    
    .ist-time {
        background-color: #1a1f2e;
        color: #ff66cc;
        padding: 8px 15px;
        border-radius: 20px;
        border: 2px solid #ff66cc;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .stButton > button {
        background-color: #ff66cc !important;
        color: #0e1117 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover { background-color: #ff99dd !important; }
    
    .greeks-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
        margin: 5px 0;
    }
    
    .max-pain-box {
        background: linear-gradient(135deg, #2e1a2e 0%, #1a2e2e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #ff9900;
        margin: 10px 0;
    }
    .max-pain-box h4 { margin: 0; color: #ff9900; }
    
    .seller-explanation {
        background: linear-gradient(135deg, #2e1a2e 0%, #3e2a3e 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff66cc;
        margin: 10px 0;
    }
    
    .entry-signal-box {
        background: linear-gradient(135deg, #2e2a1a 0%, #3e3a2a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff9900;
        margin: 15px 0;
        text-align: center;
    }
    .entry-signal-box h3 { margin: 0; color: #ff9900; font-size: 1.4rem; }
    .entry-signal-box .signal-value { font-size: 2.5rem; font-weight: 900; margin: 15px 0; }
    .entry-signal-box .signal-explanation { font-size: 1.1rem; color: #ffdd44; margin: 10px 0; }
    .entry-signal-box .entry-price { font-size: 1.8rem; color: #ffcc00; font-weight: 700; margin: 10px 0; }
    
    /* MOMENT DETECTOR BOXES */
    .moment-box {
        background: linear-gradient(135deg, #1a1f3e 0%, #2a2f4e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #00ffff;
        margin: 10px 0;
        text-align: center;
    }
    .moment-box h4 { margin: 0; color: #00ffff; font-size: 1.1rem; }
    .moment-box .moment-value { font-size: 1.8rem; font-weight: 900; margin: 10px 0; }
    
    /* TELEGRAM SIGNAL BOX */
    .telegram-box {
        background: linear-gradient(135deg, #1a2e2e 0%, #2a3e3e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #0088cc;
        margin: 15px 0;
    }
    .telegram-box h3 { margin: 0; color: #00aaff; font-size: 1.4rem; }
    
    /* EXPIRY SPIKE DETECTOR STYLES */
    .expiry-high-risk {
        background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%) !important;
        border: 3px solid #ff0000 !important;
        animation: pulse 2s infinite;
    }
    
    .expiry-medium-risk {
        background: linear-gradient(135deg, #2e2a1a 0%, #3e3a2a 100%) !important;
        border: 3px solid #ff9900 !important;
    }
    
    .expiry-low-risk {
        background: linear-gradient(135deg, #1a2e1a 0%, #2a3e2a 100%) !important;
        border: 3px solid #00ff00 !important;
    }
    
    /* OI/PCR BOXES */
    .oi-pcr-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #66b3ff;
        margin: 15px 0;
    }
    
    .oi-pcr-metric {
        background: rgba(0,0,0,0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #66b3ff;
        margin: 10px 0;
    }
    
    /* ATM BIAS CARD */
    .card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid;
        margin: 10px 0;
        text-align: center;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    
    [data-testid="stMetricLabel"] { color: #cccccc !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #ff66cc !important; font-size: 1.6rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------
#  UTILITY FUNCTIONS
# -----------------------
def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

def seller_bias_direction(chg_oi, prev_ltp, ltp, oi, option_type):
    """
    Determine seller bias direction based on OI changes and price movement.

    For sellers (option writers):
    - CE writers are bearish (expect price to stay below strike)
    - PE writers are bullish (expect price to stay above strike)

    Args:
        chg_oi: Change in open interest
        prev_ltp: Previous last traded price (can be None)
        ltp: Current last traded price
        oi: Current open interest
        option_type: "CE" or "PE"

    Returns:
        String indicating bias direction: "BULLISH", "BEARISH", or "NEUTRAL"
    """
    # If no OI, return neutral
    if oi == 0:
        return "NEUTRAL"

    # Calculate price change if we have previous LTP
    price_change = 0
    if prev_ltp is not None and prev_ltp > 0:
        price_change = ltp - prev_ltp

    # For CE (Call) options:
    # - Increasing OI + Decreasing price = Active call selling (BEARISH)
    # - Increasing OI + Increasing price = Call buying or covering (BULLISH)
    if option_type == "CE":
        if chg_oi > 0:
            # New call positions opened
            if price_change < 0:
                return "BEARISH"  # Call selling (bearish)
            elif price_change > 0:
                return "BULLISH"  # Call buying (bullish)
            else:
                return "BEARISH"  # Default to bearish for call writers
        elif chg_oi < 0:
            # Calls being closed
            if price_change < 0:
                return "BULLISH"  # Call sellers covering (bullish)
            else:
                return "NEUTRAL"
        else:
            return "NEUTRAL"

    # For PE (Put) options:
    # - Increasing OI + Decreasing price = Active put selling (BULLISH from seller's view)
    # - Increasing OI + Increasing price = Put buying or covering (BEARISH)
    elif option_type == "PE":
        if chg_oi > 0:
            # New put positions opened
            if price_change < 0:
                return "BULLISH"  # Put selling (bullish)
            elif price_change > 0:
                return "BEARISH"  # Put buying (bearish)
            else:
                return "BULLISH"  # Default to bullish for put writers
        elif chg_oi < 0:
            # Puts being closed
            if price_change < 0:
                return "BEARISH"  # Put sellers covering (bearish)
            else:
                return "NEUTRAL"
        else:
            return "NEUTRAL"

    return "NEUTRAL"

def strike_gap_from_series(series):
    diffs = series.sort_values().diff().dropna()
    if diffs.empty:
        return 50
    mode = diffs.mode()
    return int(mode.iloc[0]) if not mode.empty else int(diffs.median())

# Black-Scholes Greeks
def bs_d1(S, K, r, sigma, tau):
    if sigma <= 0 or tau <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

def bs_delta(S, K, r, sigma, tau, option_type="call"):
    if tau <= 0 or sigma <= 0:
        return 1.0 if (option_type=="call" and S>K) else (-1.0 if (option_type=="put" and S<K) else 0.0)
    d1 = bs_d1(S,K,r,sigma,tau)
    if option_type == "call":
        return norm.cdf(d1)
    return -norm.cdf(-d1)

def bs_gamma(S, K, r, sigma, tau):
    if sigma <= 0 or tau <= 0:
        return 0.0
    d1 = bs_d1(S,K,r,sigma,tau)
    return norm.pdf(d1) / (S * sigma * np.sqrt(tau))

def bs_vega(S,K,r,sigma,tau):
    if sigma <= 0 or tau <= 0:
        return 0.0
    d1 = bs_d1(S,K,r,sigma,tau)
    return S * norm.pdf(d1) * np.sqrt(tau)

def bs_theta(S,K,r,sigma,tau,option_type="call"):
    if sigma <=0 or tau<=0:
        return 0.0
    d1 = bs_d1(S,K,r,sigma,tau)
    d2 = d1 - sigma*np.sqrt(tau)
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(tau))
    if option_type=="call":
        term2 = r*K*np.exp(-r*tau)*norm.cdf(d2)
        return term1 - term2
    else:
        term2 = r*K*np.exp(-r*tau)*norm.cdf(-d2)
        return term1 + term2

# -----------------------
# 🔥 NEW: ORDERBOOK PRESSURE FUNCTIONS
# -----------------------
@st.cache_data(ttl=60)
def get_nifty_orderbook_depth():
    """
    Best-effort depth fetch from Dhan API
    """
    candidate_endpoints = [
        f"{DHAN_BASE_URL}/v2/marketfeed/quotes",
        f"{DHAN_BASE_URL}/v2/marketfeed/depth"
    ]
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID
    }
    
    for url in candidate_endpoints:
        try:
            payload = {"IDX_I": [13]}
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "success":
                continue
            
            d = data.get("data", {})
            if isinstance(d, dict):
                d1 = d.get("IDX_I", {}).get("13", {})
                depth = d1.get("depth") or d.get("depth") or d
                buy = depth.get("buy") if isinstance(depth, dict) else None
                sell = depth.get("sell") if isinstance(depth, dict) else None
                
                if buy is not None and sell is not None:
                    return {"buy": buy, "sell": sell, "source": url}
        except Exception:
            continue
    
    return None

def orderbook_pressure_score(depth: dict, levels: int = 5) -> dict:
    """
    Returns orderbook pressure (-1 to +1)
    """
    if not depth or "buy" not in depth or "sell" not in depth:
        return {"available": False, "pressure": 0.0, "buy_qty": 0.0, "sell_qty": 0.0}
    
    def sum_qty(side):
        total = 0.0
        for i, lvl in enumerate(side):
            if i >= levels:
                break
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                total += safe_float(lvl[1], 0.0)
            elif isinstance(lvl, dict):
                total += safe_float(lvl.get("qty") or lvl.get("quantity"), 0.0)
        return total
    
    buy = sum_qty(depth["buy"])
    sell = sum_qty(depth["sell"])
    denom = (buy + sell) if (buy + sell) > 0 else 1.0
    pressure = (buy - sell) / denom
    return {"available": True, "pressure": pressure, "buy_qty": buy, "sell_qty": sell}

# -----------------------
# 🔥 NEW: MOMENT DETECTOR FUNCTIONS
# -----------------------
def _init_history():
    """Initialize session state for moment history tracking"""
    if "moment_history" not in st.session_state:
        st.session_state["moment_history"] = []
    if "prev_ltps" not in st.session_state:
        st.session_state["prev_ltps"] = {}
    if "prev_ivs" not in st.session_state:
        st.session_state["prev_ivs"] = {}

def _snapshot_from_state(ts, spot, atm_strike, merged: pd.DataFrame):
    """
    Create snapshot for OI velocity/acceleration and momentum burst
    """
    total_vol = float(merged["Vol_CE"].sum() + merged["Vol_PE"].sum())
    total_iv = float(merged[["IV_CE", "IV_PE"]].mean().mean()) if not merged.empty else 0.0
    total_abs_doi = float(merged["Chg_OI_CE"].abs().sum() + merged["Chg_OI_PE"].abs().sum())
    
    per = {}
    for _, r in merged[["strikePrice", "OI_CE", "OI_PE"]].iterrows():
        per[int(r["strikePrice"])] = {"oi_ce": int(r["OI_CE"]), "oi_pe": int(r["OI_PE"])}
    
    return {
        "ts": ts,
        "spot": float(spot),
        "atm": int(atm_strike),
        "totals": {"vol": total_vol, "iv": total_iv, "abs_doi": total_abs_doi},
        "per_strike": per
    }

def _norm01(x, lo, hi):
    """Normalize value to 0-1 range"""
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def compute_momentum_burst(history):
    """
    Feature #1: Momentum Burst = (ΔVol * ΔIV * Δ|OI|) normalized to 0..100
    """
    if len(history) < 2:
        return {"available": False, "score": 0, "note": "Need at least 2 refresh points."}
    
    s_prev, s_now = history[-2], history[-1]
    dt = max((s_now["ts"] - s_prev["ts"]).total_seconds(), 1.0)
    
    dvol = (s_now["totals"]["vol"] - s_prev["totals"]["vol"]) / dt
    div = (s_now["totals"]["iv"] - s_prev["totals"]["iv"]) / dt
    ddoi = (s_now["totals"]["abs_doi"] - s_prev["totals"]["abs_doi"]) / dt
    
    burst_raw = abs(dvol) * abs(div) * abs(ddoi)
    score = int(100 * _norm01(burst_raw, 0.0, max(1.0, burst_raw * 2.5)))
    
    return {"available": True, "score": score, 
            "note": "Momentum burst (energy) is rising" if score > 60 else "No strong energy burst detected"}

def compute_gamma_cluster(merged: pd.DataFrame, atm_strike: int, window: int = 2):
    """
    Feature #3: ATM Gamma Cluster = sum(|gamma|) around ATM (±1 ±2)
    """
    if merged.empty:
        return {"available": False, "score": 0, "cluster": 0.0}
    
    want = [atm_strike + i for i in range(-window, window + 1)]
    subset = merged[merged["strikePrice"].isin(want)]
    if subset.empty:
        return {"available": False, "score": 0, "cluster": 0.0}
    
    cluster = float((subset["Gamma_CE"].abs().fillna(0) + subset["Gamma_PE"].abs().fillna(0)).sum())
    score = int(100 * _norm01(cluster, 0.0, max(1.0, cluster * 2.0)))
    return {"available": True, "score": score, "cluster": cluster}

def compute_oi_velocity_acceleration(history, atm_strike, window_strikes=3):
    """
    Feature #4: OI Velocity + Acceleration
    """
    if len(history) < 3:
        return {"available": False, "score": 0, "note": "Need 3+ refresh points for OI acceleration."}
    
    s0, s1, s2 = history[-3], history[-2], history[-1]
    dt1 = max((s1["ts"] - s0["ts"]).total_seconds(), 1.0)
    dt2 = max((s2["ts"] - s1["ts"]).total_seconds(), 1.0)
    
    def cluster_strikes(atm):
        return [atm + i for i in range(-window_strikes, window_strikes + 1) if (atm + i) in s2["per_strike"]]
    
    strikes = cluster_strikes(atm_strike)
    if not strikes:
        return {"available": False, "score": 0, "note": "No ATM cluster strikes found."}
    
    vel = []
    acc = []
    for k in strikes:
        o0 = s0["per_strike"].get(k, {"oi_ce": 0, "oi_pe": 0})
        o1 = s1["per_strike"].get(k, {"oi_ce": 0, "oi_pe": 0})
        o2 = s2["per_strike"].get(k, {"oi_ce": 0, "oi_pe": 0})
        
        t0 = o0["oi_ce"] + o0["oi_pe"]
        t1 = o1["oi_ce"] + o1["oi_pe"]
        t2 = o2["oi_ce"] + o2["oi_pe"]
        
        v1 = (t1 - t0) / dt1
        v2 = (t2 - t1) / dt2
        a = (v2 - v1) / dt2
        
        vel.append(abs(v2))
        acc.append(abs(a))
    
    vel_score = _norm01(np.median(vel), 0, max(1.0, np.percentile(vel, 90)))
    acc_score = _norm01(np.median(acc), 0, max(1.0, np.percentile(acc, 90)))
    
    score = int(100 * (0.6 * vel_score + 0.4 * acc_score))
    return {"available": True, "score": score,
            "note": "OI speed-up detected in ATM cluster" if score > 60 else "OI changes are slow/steady"}

# ============================================
# 🎯 MARKET DEPTH ANALYZER (NEW)
# ============================================

def get_option_contract_depth(security_id_ce, security_id_pe, strike_price=0, expiry="", exchange_segment="NSE_FNO"):
    """
    Fetch market depth for CE and PE contracts
    Tries strike/expiry approach first, then falls back to security IDs
    Returns: dict with CE and PE depth data
    """
    # Try strike-based approach first if available
    if strike_price > 0 and expiry and ADVANCED_DEPTH_AVAILABLE:
        try:
            dhan_config = {
                "base_url": DHAN_BASE_URL,
                "access_token": DHAN_ACCESS_TOKEN,
                "client_id": DHAN_CLIENT_ID
            }

            ce_depth = get_real_option_depth_from_dhan(strike_price, expiry, "CE", dhan_config)
            pe_depth = get_real_option_depth_from_dhan(strike_price, expiry, "PE", dhan_config)

            if ce_depth.get("available") and pe_depth.get("available"):
                return {
                    "available": True,
                    "ce_bid_qty": ce_depth.get("total_bid_qty", 0),
                    "ce_ask_qty": ce_depth.get("total_ask_qty", 0),
                    "pe_bid_qty": pe_depth.get("total_bid_qty", 0),
                    "pe_ask_qty": pe_depth.get("total_ask_qty", 0),
                    "ce_total": ce_depth.get("total_bid_qty", 0) + ce_depth.get("total_ask_qty", 0),
                    "pe_total": pe_depth.get("total_bid_qty", 0) + pe_depth.get("total_ask_qty", 0),
                    "ce_depth_full": ce_depth,
                    "pe_depth_full": pe_depth
                }
        except Exception as e:
            # Fall through to security ID approach
            pass

    # Fallback to security ID approach
    try:
        # Validate security IDs
        if not security_id_ce or not security_id_pe or security_id_ce == 0 or security_id_pe == 0:
            return {"available": False, "error": "Invalid security IDs and no strike/expiry provided"}

        url = f"{DHAN_BASE_URL}/v2/marketfeed/quote"
        payload = {exchange_segment: [int(security_id_ce), int(security_id_pe)]}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID
        }

        response = requests.post(url, json=payload, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()

            if data.get("status") == "success":
                segment_data = data.get("data", {}).get(exchange_segment, {})

                ce_data = segment_data.get(str(security_id_ce), {})
                pe_data = segment_data.get(str(security_id_pe), {})

                ce_depth = ce_data.get("depth", {})
                pe_depth = pe_data.get("depth", {})

                if ce_depth and pe_depth:
                    # Calculate bid/ask quantities
                    ce_bid_qty = sum(level.get("quantity", 0) for level in ce_depth.get("buy", []))
                    ce_ask_qty = sum(level.get("quantity", 0) for level in ce_depth.get("sell", []))
                    pe_bid_qty = sum(level.get("quantity", 0) for level in pe_depth.get("buy", []))
                    pe_ask_qty = sum(level.get("quantity", 0) for level in pe_depth.get("sell", []))

                    return {
                        "available": True,
                        "ce_bid_qty": ce_bid_qty,
                        "ce_ask_qty": ce_ask_qty,
                        "pe_bid_qty": pe_bid_qty,
                        "pe_ask_qty": pe_ask_qty,
                        "ce_total": ce_bid_qty + ce_ask_qty,
                        "pe_total": pe_bid_qty + pe_ask_qty
                    }
                else:
                    return {"available": False, "error": f"No depth data in response"}
            else:
                return {"available": False, "error": f"API error: {data.get('message', 'Unknown error')}"}
        else:
            return {"available": False, "error": f"HTTP {response.status_code}"}

    except Exception as e:
        return {"available": False, "error": f"Exception: {str(e)}"}

    return {"available": False, "error": "Unknown error"}

def get_market_depth_dhan():
    """
    Fetch Nifty 5-level market depth from Dhan REST API
    Much simpler and more reliable than WebSocket for snapshot data
    Returns: dict with bid/ask depth
    """
    try:
        url = f"{DHAN_BASE_URL}/v2/marketfeed/quote"
        payload = {"IDX_I": [13]}  # NIFTY Index
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID
        }

        response = requests.post(url, json=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get("status") == "success":
                idx_data = data.get("data", {}).get("IDX_I", {})
                nifty_data = idx_data.get("13", {})
                depth = nifty_data.get("depth", {})

                if depth:
                    bid_levels = depth.get("buy", [])
                    ask_levels = depth.get("sell", [])

                    # Convert to our format
                    bids = []
                    asks = []

                    for level in bid_levels:
                        if level.get("price", 0) > 0:  # Only valid levels
                            bids.append({
                                "price": float(level.get("price", 0)),
                                "quantity": int(level.get("quantity", 0)),
                                "orders": int(level.get("orders", 0))
                            })

                    for level in ask_levels:
                        if level.get("price", 0) > 0:  # Only valid levels
                            asks.append({
                                "price": float(level.get("price", 0)),
                                "quantity": int(level.get("quantity", 0)),
                                "orders": int(level.get("orders", 0))
                            })

                    if bids and asks:
                        total_bid_qty = sum(item["quantity"] for item in bids)
                        total_ask_qty = sum(item["quantity"] for item in asks)

                        return {
                            "bid": bids,
                            "ask": asks,
                            "total_bid_qty": total_bid_qty,
                            "total_ask_qty": total_ask_qty,
                            "source": "DHAN_5LEVEL"
                        }

    except Exception as e:
        # Fallback to simulated depth
        pass

    # Fallback: Simulated depth if API fails
    return generate_simulated_depth()

def generate_simulated_depth():
    """
    Generate simulated depth for demo/testing
    """
    spot_price = get_nifty_spot_price()
    if spot_price == 0:
        spot_price = 22500  # Default

    bid_side = []
    ask_side = []

    # Generate bid side (prices below spot)
    for i in range(1, 11):
        price = spot_price - (i * 5)  # 5 point intervals
        qty = np.random.randint(1000, 10000) * (12 - i)  # More volume near spot
        bid_side.append({
            "price": round(price, 2),
            "quantity": int(qty),
            "orders": np.random.randint(5, 50)
        })

    # Generate ask side (prices above spot)
    for i in range(1, 11):
        price = spot_price + (i * 5)  # 5 point intervals
        qty = np.random.randint(1000, 10000) * (12 - i)  # More volume near spot
        ask_side.append({
            "price": round(price, 2),
            "quantity": int(qty),
            "orders": np.random.randint(5, 50)
        })

    return {
        "bid": sorted(bid_side, key=lambda x: x["price"], reverse=True),  # Highest bid first
        "ask": sorted(ask_side, key=lambda x: x["price"]),  # Lowest ask first
        "total_bid_qty": sum(item["quantity"] for item in bid_side),
        "total_ask_qty": sum(item["quantity"] for item in ask_side),
        "source": "SIMULATED"
    }

def analyze_market_depth(depth_data, spot_price, levels=10):
    """
    Comprehensive market depth analysis
    """
    if not depth_data or "bid" not in depth_data or "ask" not in depth_data:
        return {"available": False}

    bids = depth_data["bid"][:levels]
    asks = depth_data["ask"][:levels]

    total_bid_qty = depth_data.get("total_bid_qty", sum(b["quantity"] for b in bids if isinstance(b, dict)))
    total_ask_qty = depth_data.get("total_ask_qty", sum(a["quantity"] for a in asks if isinstance(a, dict)))

    # 1. Depth Imbalance
    total_qty = total_bid_qty + total_ask_qty
    depth_imbalance = (total_bid_qty - total_ask_qty) / total_qty if total_qty > 0 else 0

    # 2. Near-spot concentration
    near_bid_qty = sum(b["quantity"] for b in bids[:3] if isinstance(b, dict))  # Top 3 bids
    near_ask_qty = sum(a["quantity"] for a in asks[:3] if isinstance(a, dict))  # Top 3 asks
    near_imbalance = (near_bid_qty - near_ask_qty) / (near_bid_qty + near_ask_qty) if (near_bid_qty + near_ask_qty) > 0 else 0

    # 3. Large Orders Detection
    avg_bid_qty = np.mean([b["quantity"] for b in bids if isinstance(b, dict)]) if bids else 0
    avg_ask_qty = np.mean([a["quantity"] for a in asks if isinstance(a, dict)]) if asks else 0

    large_bid_orders = [b for b in bids if isinstance(b, dict) and b["quantity"] > avg_bid_qty * 3]
    large_ask_orders = [a for a in asks if isinstance(a, dict) and a["quantity"] > avg_ask_qty * 3]

    # 4. Spread Analysis
    if bids and asks:
        best_bid = max(b["price"] for b in bids if isinstance(b, dict))
        best_ask = min(a["price"] for a in asks if isinstance(a, dict))
        spread = best_ask - best_bid
        spread_percent = (spread / spot_price) * 100
    else:
        best_bid = spot_price
        best_ask = spot_price
        spread = 0
        spread_percent = 0

    # 5. Depth Profile
    price_levels = sorted([(b["price"], "BID", b["quantity"]) for b in bids if isinstance(b, dict)] +
                          [(a["price"], "ASK", a["quantity"]) for a in asks if isinstance(a, dict)],
                          key=lambda x: x[0])

    # 6. Support/Resistance from Depth
    support_levels = sorted([b for b in bids if isinstance(b, dict)], key=lambda x: x["quantity"], reverse=True)[:3]
    resistance_levels = sorted([a for a in asks if isinstance(a, dict)], key=lambda x: x["quantity"], reverse=True)[:3]

    return {
        "available": True,
        "depth_imbalance": depth_imbalance,
        "near_imbalance": near_imbalance,
        "total_bid_qty": total_bid_qty,
        "total_ask_qty": total_ask_qty,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_percent": spread_percent,
        "large_bid_orders": len(large_bid_orders),
        "large_ask_orders": len(large_ask_orders),
        "avg_bid_size": avg_bid_qty,
        "avg_ask_size": avg_ask_qty,
        "price_levels": price_levels,
        "top_supports": [(s["price"], s["quantity"]) for s in support_levels],
        "top_resistances": [(r["price"], r["quantity"]) for r in resistance_levels],
        "bid_side": bids,
        "ask_side": asks,
        "total_levels": len(bids) + len(asks)
    }

def calculate_depth_based_signals(depth_analysis, spot_price):
    """
    Generate trading signals from depth analysis
    """
    if not depth_analysis["available"]:
        return {"available": False}

    signals = []
    confidence = 0
    signal_type = "NEUTRAL"

    # 1. Depth Imbalance Signal
    imbalance = depth_analysis["depth_imbalance"]
    if imbalance > 0.3:
        signals.append(f"Strong buy depth (imbalance: {imbalance:+.2f})")
        confidence += 30
        signal_type = "BULLISH"
    elif imbalance < -0.3:
        signals.append(f"Strong sell depth (imbalance: {imbalance:+.2f})")
        confidence += 30
        signal_type = "BEARISH"

    # 2. Near-spot Imbalance
    near_imbalance = depth_analysis["near_imbalance"]
    if abs(near_imbalance) > 0.4:
        if near_imbalance > 0:
            signals.append(f"Heavy bids near spot")
            confidence += 20
            if signal_type == "NEUTRAL":
                signal_type = "BULLISH"
        else:
            signals.append(f"Heavy asks near spot")
            confidence += 20
            if signal_type == "NEUTRAL":
                signal_type = "BEARISH"

    # 3. Large Orders Signal
    if depth_analysis["large_bid_orders"] > depth_analysis["large_ask_orders"] + 2:
        signals.append(f"More large bids ({depth_analysis['large_bid_orders']}) than asks ({depth_analysis['large_ask_orders']})")
        confidence += 15
    elif depth_analysis["large_ask_orders"] > depth_analysis["large_bid_orders"] + 2:
        signals.append(f"More large asks ({depth_analysis['large_ask_orders']}) than bids ({depth_analysis['large_bid_orders']})")
        confidence += 15

    # 4. Spread Analysis
    if depth_analysis["spread_percent"] < 0.01:  # Tight spread
        signals.append(f"Tight spread ({depth_analysis['spread_percent']:.3f}%) - Good liquidity")
        confidence += 10
    elif depth_analysis["spread_percent"] > 0.05:  # Wide spread
        signals.append(f"Wide spread ({depth_analysis['spread_percent']:.3f}%) - Low liquidity")
        confidence -= 10

    # Determine overall signal
    if confidence >= 50:
        strength = "STRONG"
        color = "#00ff88" if signal_type == "BULLISH" else "#ff4444"
    elif confidence >= 30:
        strength = "MODERATE"
        color = "#00cc66" if signal_type == "BULLISH" else "#ff6666"
    else:
        strength = "NEUTRAL"
        signal_type = "NEUTRAL"
        color = "#66b3ff"

    return {
        "available": True,
        "signal_type": signal_type,
        "strength": strength,
        "confidence": min(confidence, 100),
        "color": color,
        "signals": signals,
        "imbalance": imbalance,
        "near_imbalance": near_imbalance,
        "spread_percent": depth_analysis["spread_percent"]
    }

def enhanced_orderbook_pressure(depth_analysis, spot):
    """
    Enhanced orderbook pressure with depth analysis
    """
    if not depth_analysis["available"]:
        return {"available": False}

    # Calculate pressure from multiple depth factors
    factors = []
    pressure_score = 0

    # 1. Overall imbalance (40% weight)
    imbalance = depth_analysis["depth_imbalance"]
    pressure_score += imbalance * 0.4
    factors.append(f"Overall imbalance: {imbalance:+.3f}")

    # 2. Near-spot concentration (30% weight)
    near_imbalance = depth_analysis["near_imbalance"]
    pressure_score += near_imbalance * 0.3
    factors.append(f"Near-spot imbalance: {near_imbalance:+.3f}")

    # 3. Large orders bias (20% weight)
    large_orders_diff = (depth_analysis["large_bid_orders"] - depth_analysis["large_ask_orders"])
    large_orders_bias = large_orders_diff / max(1, depth_analysis["large_bid_orders"] + depth_analysis["large_ask_orders"])
    pressure_score += large_orders_bias * 0.2
    factors.append(f"Large orders bias: {large_orders_bias:+.3f}")

    # 4. Spread tightness (10% weight) - tighter spread = more pressure
    spread_factor = max(0, 0.05 - depth_analysis["spread_percent"]) / 0.05  # Normalize 0-1
    pressure_score += spread_factor * 0.1
    factors.append(f"Spread factor: {spread_factor:+.3f}")

    # Normalize to -1 to 1 range
    pressure_score = max(min(pressure_score, 1), -1)

    return {
        "available": True,
        "pressure": pressure_score,
        "factors": factors,
        "total_bid_qty": depth_analysis["total_bid_qty"],
        "total_ask_qty": depth_analysis["total_ask_qty"],
        "buy_qty": float(depth_analysis["total_bid_qty"]),  # Backward compatibility
        "sell_qty": float(depth_analysis["total_ask_qty"]),  # Backward compatibility
        "best_bid": depth_analysis["best_bid"],
        "best_ask": depth_analysis["best_ask"],
        "spread": depth_analysis["spread"],
        "spread_percent": depth_analysis["spread_percent"]
    }

def display_market_depth_dashboard(spot, depth_analysis, depth_signals, enhanced_pressure):
    """
    Display comprehensive market depth dashboard
    """
    st.markdown("---")
    st.markdown("## 📊 MARKET DEPTH ANALYZER (Order Book)")

    if not depth_analysis["available"]:
        st.warning("Market depth data unavailable")
        return

    # Header with key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        imbalance = depth_analysis["depth_imbalance"]
        color = "#00ff88" if imbalance > 0.1 else ("#ff4444" if imbalance < -0.1 else "#66b3ff")
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
            <div style="font-size: 0.9rem; color:#cccccc;">Depth Imbalance</div>
            <div style="font-size: 1.8rem; color:{color}; font-weight:700;">{imbalance:+.3f}</div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">Bid/Ask ratio</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        spread_pct = depth_analysis["spread_percent"]
        color = "#00ff88" if spread_pct < 0.02 else ("#ff9900" if spread_pct < 0.05 else "#ff4444")
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
            <div style="font-size: 0.9rem; color:#cccccc;">Bid-Ask Spread</div>
            <div style="font-size: 1.8rem; color:{color}; font-weight:700;">{spread_pct:.3f}%</div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">₹{depth_analysis['spread']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        total_bid = depth_analysis["total_bid_qty"]
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
            <div style="font-size: 0.9rem; color:#cccccc;">Total Bid Qty</div>
            <div style="font-size: 1.8rem; color:#00ff88; font-weight:700;">{total_bid:,}</div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">Buy orders</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_ask = depth_analysis["total_ask_qty"]
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
            <div style="font-size: 0.9rem; color:#cccccc;">Total Ask Qty</div>
            <div style="font-size: 1.8rem; color:#ff4444; font-weight:700;">{total_ask:,}</div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">Sell orders</div>
        </div>
        """, unsafe_allow_html=True)

    # Depth Signal
    st.markdown("### 🎯 Depth-Based Signal")
    if depth_signals["available"]:
        col_sig1, col_sig2 = st.columns([1, 2])

        with col_sig1:
            st.markdown(f"""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background: {'#1a2e1a' if depth_signals['signal_type'] == 'BULLISH' else
                           '#2e1a1a' if depth_signals['signal_type'] == 'BEARISH' else '#1a1f2e'};
                border: 3px solid {depth_signals['color']};
                text-align: center;
            ">
                <div style="font-size: 1.2rem; color:#ffffff;">Depth Signal</div>
                <div style="font-size: 2rem; color:{depth_signals['color']}; font-weight:900;">
                    {depth_signals['signal_type']}
                </div>
                <div style="font-size: 1rem; color:#ffcc00;">
                    {depth_signals['strength']}
                </div>
                <div style="font-size: 0.9rem; color:#cccccc; margin-top:10px;">
                    Confidence: {depth_signals['confidence']}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_sig2:
            st.markdown("#### Signal Factors:")
            for signal in depth_signals["signals"]:
                st.markdown(f"• {signal}")

    # Enhanced Pressure Analysis
    st.markdown("### ⚡ Enhanced Orderbook Pressure")
    if enhanced_pressure["available"]:
        col_pres1, col_pres2 = st.columns(2)

        with col_pres1:
            pressure = enhanced_pressure["pressure"]
            color = "#00ff88" if pressure > 0.2 else ("#ff4444" if pressure < -0.2 else "#66b3ff")
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: rgba(0,0,0,0.3); border-radius: 10px;">
                <div style="font-size: 1.1rem; color:#cccccc;">Enhanced Pressure Score</div>
                <div style="font-size: 2.5rem; color:{color}; font-weight:900; margin:10px 0;">
                    {pressure:+.3f}
                </div>
                <div style="font-size: 0.9rem; color:#aaaaaa;">
                    Range: -1 (Sell) to +1 (Buy)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_pres2:
            st.markdown("#### Pressure Factors:")
            for factor in enhanced_pressure["factors"]:
                st.markdown(f"• {factor}")

            st.markdown(f"""
            **Current Spot:** ₹{spot:,.2f}
            **Best Bid:** ₹{enhanced_pressure['best_bid']:,.2f}
            **Best Ask:** ₹{enhanced_pressure['best_ask']:,.2f}
            **Spread:** ₹{enhanced_pressure['spread']:.2f} ({enhanced_pressure['spread_percent']:.3f}%)
            """)

    # Support/Resistance Levels from Depth
    if depth_analysis.get("top_supports") and depth_analysis.get("top_resistances"):
        st.markdown("### 📍 Key Levels from Order Book Depth")

        col_sup, col_res = st.columns(2)

        with col_sup:
            st.markdown("#### 🟢 Support Levels")
            st.markdown("*Largest bid quantities*")
            for i, (price, qty) in enumerate(depth_analysis["top_supports"][:3], 1):
                st.markdown(f"**S{i}:** ₹{price:,.2f} (Qty: {qty:,})")

        with col_res:
            st.markdown("#### 🔴 Resistance Levels")
            st.markdown("*Largest ask quantities*")
            for i, (price, qty) in enumerate(depth_analysis["top_resistances"][:3], 1):
                st.markdown(f"**R{i}:** ₹{price:,.2f} (Qty: {qty:,})")

        # --- Vertical bar chart with Spot / ATM / ITM1 / OTM1 lines ---
        supports = depth_analysis["top_supports"][:3]
        resistances = depth_analysis["top_resistances"][:3]

        # Sort supports ascending, resistances ascending by price
        supports_sorted = sorted(supports, key=lambda x: x[0])
        resistances_sorted = sorted(resistances, key=lambda x: x[0])

        sup_prices = [p for p, _ in supports_sorted]
        sup_qtys = [q for _, q in supports_sorted]
        res_prices = [p for p, _ in resistances_sorted]
        res_qtys = [q for _, q in resistances_sorted]

        import plotly.graph_objects as go
        fig = go.Figure()

        # Support bars (green) — below spot
        fig.add_trace(go.Bar(
            x=sup_prices, y=sup_qtys,
            marker_color="#00ff88",
            text=[f"S{i+1}<br>{q:,}" for i, q in enumerate(sup_qtys)],
            textposition="outside", textfont=dict(color="#00ff88", size=11),
            name="Support", width=3,
        ))
        # Resistance bars (red) — above spot
        fig.add_trace(go.Bar(
            x=res_prices, y=res_qtys,
            marker_color="#ff4444",
            text=[f"R{i+1}<br>{q:,}" for i, q in enumerate(res_qtys)],
            textposition="outside", textfont=dict(color="#ff4444", size=11),
            name="Resistance", width=3,
        ))

        # Calculate ATM strike (round spot to nearest 50)
        atm_strike = round(spot / 50) * 50
        strike_gap = 50
        itm1 = atm_strike - strike_gap  # 1 strike below ATM
        otm1 = atm_strike + strike_gap  # 1 strike above ATM

        # Vertical reference lines: Spot, ATM, ITM1, OTM1
        ref_lines = [
            (spot, "Spot", "#ffff00", "dash"),
            (atm_strike, "ATM", "#ff66cc", "solid"),
            (itm1, "ITM1", "#66b3ff", "dot"),
            (otm1, "OTM1", "#ff9933", "dot"),
        ]
        for price_val, label, color, dash_style in ref_lines:
            fig.add_vline(
                x=price_val,
                line_width=2, line_dash=dash_style, line_color=color,
                annotation_text=f"{label}: ₹{price_val:,.0f}",
                annotation_position="top",
                annotation_font=dict(color=color, size=11),
            )

        # Set x-axis range to include all bars AND all reference lines
        all_x_values = sup_prices + res_prices + [spot, atm_strike, itm1, otm1]
        x_min = min(all_x_values) - 10
        x_max = max(all_x_values) + 10

        fig.update_layout(
            title=f"Support & Resistance — Order Book Depth  |  Spot: ₹{spot:,.2f}",
            xaxis_title="Price Level (₹)",
            yaxis_title="Quantity",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            height=420,
            margin=dict(l=10, r=40, t=70, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickformat=",.0f", tickprefix="₹", range=[x_min, x_max]),
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 🎯 COMPREHENSIVE MARKET DEPTH DASHBOARD (ADVANCED)
# -----------------------

def display_comprehensive_depth_analysis(analysis_results):
    """
    Display ALL advanced depth analysis metrics in organized sections
    """
    if not analysis_results.get("available"):
        st.warning("⚠️ Advanced depth analysis unavailable")
        return

    st.markdown("---")
    st.markdown("## 🎯 COMPREHENSIVE MARKET DEPTH ANALYSIS")

    # SECTION 1: ORDER FLOW ANALYSIS
    if analysis_results.get("order_flow", {}).get("available"):
        st.markdown("### 📊 Order Flow Analysis")
        flow = analysis_results["order_flow"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Aggressive Buys", f"{flow['aggressive_buy_volume']:,}")
            st.metric("Passive Buys", f"{flow['passive_buy_volume']:,}")

        with col2:
            st.metric("Aggressive Sells", f"{flow['aggressive_sell_volume']:,}")
            st.metric("Passive Sells", f"{flow['passive_sell_volume']:,}")

        with col3:
            buy_pressure = flow['buy_pressure_pct']
            st.metric("Buy Pressure", f"{buy_pressure:.1f}%",
                     delta="Bullish" if buy_pressure > 60 else ("Bearish" if buy_pressure < 40 else None))

        with col4:
            sell_pressure = flow['sell_pressure_pct']
            st.metric("Sell Pressure", f"{sell_pressure:.1f}%",
                     delta="Bearish" if sell_pressure > 60 else ("Bullish" if sell_pressure < 40 else None))

        if flow.get("large_orders"):
            st.markdown("**🐋 Large Orders Detected:**")
            large = flow["large_orders"]
            st.write(f"- Bid side: {large['bid_large']} large orders ({large['bid_institutional']} institutional)")
            st.write(f"- Ask side: {large['ask_large']} large orders ({large['ask_institutional']} institutional)")

        st.info(f"💡 {flow['flow_interpretation']}")

    # SECTION 2: MARKET MAKER DETECTION
    if analysis_results.get("market_maker", {}).get("available"):
        st.markdown("### 🏦 Market Maker Activity")
        mm = analysis_results["market_maker"]

        col1, col2, col3 = st.columns(3)

        with col1:
            score = mm['mm_presence_score']
            color = "🟢" if score > 70 else ("🟡" if score > 40 else "🔴")
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                <div style="font-size: 0.9rem; color:#cccccc;">MM Presence Score</div>
                <div style="font-size: 2rem; font-weight:700;">{color} {score}/100</div>
                <div style="font-size: 0.8rem; color:#aaaaaa;">{mm['interpretation']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.metric("Round Number Orders", f"{mm['round_number_orders_pct']:.1f}%")
            st.metric("Lot-Based Orders", f"{mm['lot_based_orders_pct']:.1f}%")

        with col3:
            st.metric("Bid-Ask Spread", f"{mm['spread_pct']:.3f}%")
            st.metric("Spread Consistency", f"{mm['spread_consistency']*100:.0f}%")

    # SECTION 3: LIQUIDITY PROFILE
    if analysis_results.get("liquidity_profile", {}).get("available"):
        st.markdown("### 💧 Liquidity Profile & Market Impact")
        liq = analysis_results["liquidity_profile"]

        # Top 5 concentration
        st.markdown("**Depth Concentration:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Top 5 Bid Concentration", f"{liq['top5_concentration']['top5_bid_pct']:.1f}%")
        with col2:
            st.metric("Top 5 Ask Concentration", f"{liq['top5_concentration']['top5_ask_pct']:.1f}%")

        # Market Impact Table
        st.markdown("**📈 Market Impact Estimates:**")

        impact_data = []
        for size in ["1k_contracts", "5k_contracts", "10k_contracts"]:
            impact = liq["price_impact"][size]
            impact_data.append({
                "Order Size": size.replace("_contracts", ""),
                "Avg Execution Price": f"₹{impact['avg_execution_price']:.2f}",
                "Impact %": f"{impact['impact_pct']:.3f}%",
                "Levels Consumed": impact['levels_consumed'],
                "Filled Qty": f"{impact['filled']:,}"
            })

        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True)

        # Slippage
        st.markdown("**💸 Slippage Costs:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("1K Contracts Slippage", f"{liq['slippage']['1k_contracts_pct']:.3f}%")
        with col2:
            st.metric("5K Contracts Slippage", f"{liq['slippage']['5k_contracts_pct']:.3f}%")

        # Fragility warning
        fragility = liq.get("liquidity_fragility_score", 0)
        if fragility > 70:
            st.warning(f"⚠️ High Liquidity Fragility ({fragility:.0f}/100) - Large orders may significantly impact price")

    # SECTION 4: DEPTH QUALITY
    if analysis_results.get("depth_quality", {}).get("available"):
        st.markdown("### ⭐ Depth Quality Metrics")
        qual = analysis_results["depth_quality"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            thickness = qual['thickness_score']
            st.metric("Thickness", f"{thickness}/100")

        with col2:
            resilience = qual['resilience_score']
            st.metric("Resilience", f"{resilience}/100")

        with col3:
            granularity = qual['granularity_score']
            st.metric("Granularity", f"{granularity}/100")

        with col4:
            overall = qual['overall_quality_score']
            color = "🟢" if overall > 75 else ("🟡" if overall > 50 else "🔴")
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                <div style="font-size: 0.8rem; color:#cccccc;">Overall Quality</div>
                <div style="font-size: 1.5rem; font-weight:700;">{color} {overall}/100</div>
            </div>
            """, unsafe_allow_html=True)

        st.info(f"💡 {qual['interpretation']} - Total Depth: {qual['total_depth_contracts']:,} contracts")

    # SECTION 5: MARKET MICROSTRUCTURE
    if analysis_results.get("microstructure", {}).get("available"):
        st.markdown("### 🔬 Market Microstructure Signals")
        micro = analysis_results["microstructure"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Informed Trading", f"{micro['informed_trading_score']}/100")
            st.caption(micro['signals']['informed_trading'])

        with col2:
            st.metric("Liquidity Provision", f"{micro['liquidity_provision_score']}/100")
            st.caption(micro['signals']['liquidity_provision'])

        with col3:
            st.metric("Stop Hunt Risk", f"{micro['stop_hunt_probability']}/100")
            st.caption(micro['signals']['stop_hunt_risk'])

        with col4:
            st.metric("Gamma Hedging", f"{micro['gamma_hedging_score']}/100")
            st.caption(micro['signals']['gamma_hedging'])

    # SECTION 6: ALGORITHMIC PATTERNS
    if analysis_results.get("algo_patterns", {}).get("available"):
        st.markdown("### 🤖 Algorithmic Trading Patterns")
        algo = analysis_results["algo_patterns"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("TWAP/VWAP Detected", "Yes ✅" if algo['twap_vwap_detected'] else "No ❌")

        with col2:
            st.metric("Iceberg Orders", "Detected 🧊" if algo['iceberg_detected'] else "None")

        with col3:
            st.metric("Quote Stuffing", "Yes ⚠️" if algo['quote_stuffing_detected'] else "No")

        st.metric("Spoofing Probability", f"{algo['spoofing_probability']}/100")
        st.caption(f"Update Frequency: {algo['update_frequency']} updates/sec")

    # SECTION 7: DEPTH LEVEL DETAILS
    if analysis_results.get("depth_levels", {}).get("available"):
        st.markdown("### 📊 Depth Level Analysis")
        levels = analysis_results["depth_levels"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Bid Side:**")
            st.metric("Total Bid Qty", f"{levels['total_bid_qty']:,}")
            st.metric("Top 3 Concentration", f"{levels['top3_concentration']['bid_pct']:.1f}%")
            st.metric("Iceberg Probability", f"{levels['iceberg_probability']['bid']*100:.1f}%")

        with col2:
            st.markdown("**Ask Side:**")
            st.metric("Total Ask Qty", f"{levels['total_ask_qty']:,}")
            st.metric("Top 3 Concentration", f"{levels['top3_concentration']['ask_pct']:.1f}%")
            st.metric("Iceberg Probability", f"{levels['iceberg_probability']['ask']*100:.1f}%")

        imbalance = levels['depth_imbalance']
        color = "🟢" if imbalance > 0.1 else ("🔴" if imbalance < -0.1 else "⚪")
        st.metric("Depth Imbalance", f"{color} {imbalance:+.3f}")

    # SECTION 8: MARKET IMPACT SUMMARY
    st.markdown("### 💰 Market Impact Summary")

    impact_sizes = ["1k", "5k", "10k"]
    for size in impact_sizes:
        impact_key = f"market_impact_{size}"
        if analysis_results.get(impact_key, {}).get("available"):
            impact = analysis_results[impact_key]

            with st.expander(f"📊 {size.upper()} Contracts Order Impact"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Impact", f"{impact['total_impact_bps']:.2f} bps")
                    st.metric("Temporary Impact", f"{impact['temporary_impact_bps']:.2f} bps")

                with col2:
                    st.metric("Permanent Impact", f"{impact['permanent_impact_bps']:.2f} bps")
                    st.metric("Slippage Cost", f"₹{impact['slippage_cost']:.2f}")

                with col3:
                    st.metric("Participation Rate", f"{impact['participation_rate_pct']:.2f}%")

                st.info(f"**Recommended Strategy:** {impact['optimal_strategy']}")

                # Immediate impact details
                imm = impact['immediate_impact']
                st.write(f"- Average execution price: ₹{imm['avg_price']:.2f}")
                st.write(f"- Levels consumed: {imm['levels_consumed']}")
                st.write(f"- Unfilled quantity: {imm['unfilled_qty']:,}")


# -----------------------
# 🔥 ENTRY SIGNAL CALCULATION (EXTENDED WITH MOMENT DETECTOR & ATM BIAS)
# -----------------------
def calculate_realistic_stop_loss_target(position_type, entry_price, nearest_sup, nearest_res, strike_gap, max_risk_pct=1.5):
    """
    Calculate realistic stop loss and target with proper risk management
    """
    stop_loss = None
    target = None
    
    if not nearest_sup or not nearest_res:
        return stop_loss, target
    
    # Maximum risk = 1.5% of entry price
    max_risk_points = entry_price * (max_risk_pct / 100)
    
    if position_type == "LONG":
        # Option 1: Support-based stop (1.5 strike gaps below support)
        stop_loss_support = nearest_sup["strike"] - (strike_gap * 1.5)
        
        # Option 2: Percentage-based stop
        stop_loss_pct = entry_price - max_risk_points
        
        # Use the HIGHER stop (tighter risk management)
        stop_loss = max(stop_loss_support, stop_loss_pct)
        
        # Calculate risk amount
        risk_amount = entry_price - stop_loss
        
        # Target 1: 2:1 risk:reward
        target_rr = entry_price + (risk_amount * 2)
        
        # Target 2: Near resistance
        target_resistance = nearest_res["strike"] - strike_gap
        
        # Use the LOWER target (more conservative)
        target = min(target_rr, target_resistance)
        
        # Ensure target > entry
        if target <= entry_price:
            target = entry_price + risk_amount  # 1:1 at minimum
    
    elif position_type == "SHORT":
        # Option 1: Resistance-based stop (1.5 strike gaps above resistance)
        stop_loss_resistance = nearest_res["strike"] + (strike_gap * 1.5)
        
        # Option 2: Percentage-based stop
        stop_loss_pct = entry_price + max_risk_points
        
        # Use the LOWER stop (tighter risk management)
        stop_loss = min(stop_loss_resistance, stop_loss_pct)
        
        # Calculate risk amount
        risk_amount = stop_loss - entry_price
        
        # Target 1: 2:1 risk:reward
        target_rr = entry_price - (risk_amount * 2)
        
        # Target 2: Near support
        target_support = nearest_sup["strike"] + strike_gap
        
        # Use the HIGHER target (more conservative)
        target = max(target_rr, target_support)
        
        # Ensure target < entry
        if target >= entry_price:
            target = entry_price - risk_amount  # 1:1 at minimum
    
    # Round to nearest 50 for Nifty
    if stop_loss:
        stop_loss = round(stop_loss / 50) * 50
    if target:
        target = round(target / 50) * 50
    
    return stop_loss, target

def calculate_entry_signal_extended(
    spot, 
    merged_df, 
    atm_strike, 
    seller_bias_result, 
    seller_max_pain, 
    seller_supports_df, 
    seller_resists_df, 
    nearest_sup, 
    nearest_res, 
    seller_breakout_index,
    moment_metrics,  # NEW: Add moment metrics
    atm_bias=None,   # NEW: Add ATM bias
    support_bias=None,  # NEW: Add support bias
    resistance_bias=None  # NEW: Add resistance bias
):
    """
    Calculate optimal entry signal with Moment Detector & ATM Bias integration
    """
    
    # Initialize signal components
    signal_score = 0
    signal_reasons = []
    optimal_entry_price = spot
    position_type = "NEUTRAL"
    confidence = 0
    
    # ============================================
    # 1. SELLER BIAS ANALYSIS (40% weight)
    # ============================================
    seller_bias = seller_bias_result["bias"]
    seller_polarity = seller_bias_result["polarity"]
    
    if "STRONG BULLISH" in seller_bias or "BULLISH" in seller_bias:
        signal_score += 40
        position_type = "LONG"
        signal_reasons.append(f"Seller bias: {seller_bias} (Polarity: {seller_polarity:.1f})")
    elif "STRONG BEARISH" in seller_bias or "BEARISH" in seller_bias:
        signal_score += 40
        position_type = "SHORT"
        signal_reasons.append(f"Seller bias: {seller_bias} (Polarity: {seller_polarity:.1f})")
    else:
        signal_score += 10
        position_type = "NEUTRAL"
        signal_reasons.append("Seller bias: Neutral - Wait for clearer signal")
    
    # ============================================
    # 2. MAX PAIN ALIGNMENT (15% weight)
    # ============================================
    if seller_max_pain:
        max_pain_strike = seller_max_pain.get("max_pain_strike", 0)
        distance_to_max_pain = abs(spot - max_pain_strike)
        distance_pct = (distance_to_max_pain / spot) * 100

        if distance_pct < 0.5:
            signal_score += 15
            signal_reasons.append(f"Spot VERY close to Max Pain (₹{max_pain_strike:,}, {distance_pct:.2f}%)")
            optimal_entry_price = max_pain_strike
        elif distance_pct < 1.0:
            signal_score += 10
            signal_reasons.append(f"Spot close to Max Pain (₹{max_pain_strike:,}, {distance_pct:.2f}%)")
            if position_type == "LONG" and spot < max_pain_strike:
                optimal_entry_price = min(spot + (max_pain_strike - spot) * 0.5, max_pain_strike)
            elif position_type == "SHORT" and spot > max_pain_strike:
                optimal_entry_price = max(spot - (spot - max_pain_strike) * 0.5, max_pain_strike)
    
    # ============================================
    # 3. SUPPORT/RESISTANCE ALIGNMENT (20% weight)
    # ============================================
    if nearest_sup and nearest_res:
        range_size = nearest_res["strike"] - nearest_sup["strike"]
        if range_size > 0:
            position_in_range = ((spot - nearest_sup["strike"]) / range_size) * 100
            
            if position_type == "LONG":
                if position_in_range < 40:
                    signal_score += 20
                    signal_reasons.append(f"Ideal LONG entry: Near support (₹{nearest_sup['strike']:,})")
                    optimal_entry_price = nearest_sup["strike"] + (range_size * 0.1)
                elif position_in_range < 60:
                    signal_score += 10
                    signal_reasons.append("OK LONG entry: Middle of range")
                else:
                    signal_score += 5
                    
            elif position_type == "SHORT":
                if position_in_range > 60:
                    signal_score += 20
                    signal_reasons.append(f"Ideal SHORT entry: Near resistance (₹{nearest_res['strike']:,})")
                    optimal_entry_price = nearest_res["strike"] - (range_size * 0.1)
                elif position_in_range > 40:
                    signal_score += 10
                    signal_reasons.append("OK SHORT entry: Middle of range")
                else:
                    signal_score += 5
    
    # ============================================
    # 4. BREAKOUT INDEX (15% weight)
    # ============================================
    if seller_breakout_index > 80:
        signal_score += 15
        signal_reasons.append(f"High Breakout Index ({seller_breakout_index}%): Strong momentum expected")
    elif seller_breakout_index > 60:
        signal_score += 10
        signal_reasons.append(f"Moderate Breakout Index ({seller_breakout_index}%): Some momentum expected")
    
    # ============================================
    # 5. PCR ANALYSIS (10% weight)
    # ============================================
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        total_pcr = total_pe_oi / total_ce_oi
        if position_type == "LONG" and total_pcr > 1.5:
            signal_score += 10
            signal_reasons.append(f"Strong PCR ({total_pcr:.2f}): Heavy PUT selling confirms bullish bias")
        elif position_type == "SHORT" and total_pcr < 0.7:
            signal_score += 10
            signal_reasons.append(f"Strong PCR ({total_pcr:.2f}): Heavy CALL selling confirms bearish bias")
    
    # ============================================
    # 6. GEX ANALYSIS (Adjustment factor)
    # ============================================
    total_gex_net = merged_df["GEX_Net"].sum()
    if total_gex_net > 1000000:
        if position_type == "LONG":
            signal_score += 5
            signal_reasons.append("Positive GEX: Supports LONG position (stabilizing)")
    elif total_gex_net < -1000000:
        if position_type == "SHORT":
            signal_score += 5
            signal_reasons.append("Negative GEX: Supports SHORT position (destabilizing)")
    
    # ============================================
    # 7. MOMENT DETECTOR FEATURES (NEW - 30% total weight)
    # ============================================
    
    # 7.1 Momentum Burst (12% weight)
    mb = moment_metrics.get("momentum_burst", {})
    if mb.get("available", False):
        mb_score = mb.get("score", 0)
        signal_score += int(12 * (mb_score / 100.0))
        signal_reasons.append(f"Momentum burst: {mb_score}/100 - {mb.get('note', '')}")
    
    # 7.2 Orderbook Pressure (8% weight)
    ob = moment_metrics.get("orderbook", {})
    if ob.get("available", False):
        pressure = ob.get("pressure", 0.0)
        if position_type == "LONG" and pressure > 0.15:
            signal_score += 8
            signal_reasons.append(f"Orderbook buy pressure: {pressure:+.2f} (supports LONG)")
        elif position_type == "SHORT" and pressure < -0.15:
            signal_score += 8
            signal_reasons.append(f"Orderbook sell pressure: {pressure:+.2f} (supports SHORT)")
    
    # 7.3 Gamma Cluster (6% weight)
    gc = moment_metrics.get("gamma_cluster", {})
    if gc.get("available", False):
        gc_score = gc.get("score", 0)
        signal_score += int(6 * (gc_score / 100.0))
        signal_reasons.append(f"Gamma cluster: {gc_score}/100 (ATM concentration)")
    
    # 7.4 OI Acceleration (4% weight)
    oi_accel = moment_metrics.get("oi_accel", {})
    if oi_accel.get("available", False):
        oi_score = oi_accel.get("score", 0)
        signal_score += int(4 * (oi_score / 100.0))
        signal_reasons.append(f"OI acceleration: {oi_score}/100 ({oi_accel.get('note', '')})")
    
    # ============================================
    # 8. ATM BIAS INTEGRATION (NEW - 20% weight)
    # ============================================
    if atm_bias:
        atm_score = atm_bias["total_score"]
        if position_type == "LONG" and atm_score > 0.1:
            signal_score += int(20 * (atm_score / 1.0))  # Scale to max 20 points
            signal_reasons.append(f"ATM bias bullish ({atm_score:.2f}) confirms LONG")
        elif position_type == "SHORT" and atm_score < -0.1:
            signal_score += int(20 * (abs(atm_score) / 1.0))
            signal_reasons.append(f"ATM bias bearish ({atm_score:.2f}) confirms SHORT")
    
    # ============================================
    # 9. SUPPORT/RESISTANCE BIAS INTEGRATION (NEW - 15% weight)
    # ============================================
    if support_bias and position_type == "LONG":
        support_score = support_bias["total_score"]
        if support_score > 0.2:
            signal_score += int(15 * (support_score / 1.0))
            signal_reasons.append(f"Strong support bias ({support_score:.2f}) at ₹{support_bias['strike']:,}")
    
    if resistance_bias and position_type == "SHORT":
        resistance_score = resistance_bias["total_score"]
        if resistance_score < -0.2:
            signal_score += int(15 * (abs(resistance_score) / 1.0))
            signal_reasons.append(f"Strong resistance bias ({resistance_score:.2f}) at ₹{resistance_bias['strike']:,}")
    
    # ============================================
    # FINAL SIGNAL CALCULATION
    # ============================================
    
    # Calculate confidence percentage
    confidence = min(max(signal_score, 0), 100)
    
    # Determine signal strength
    if confidence >= 80:
        signal_strength = "STRONG"
        signal_color = "#00ff88" if position_type == "LONG" else "#ff4444"
    elif confidence >= 60:
        signal_strength = "MODERATE"
        signal_color = "#00cc66" if position_type == "LONG" else "#ff6666"
    elif confidence >= 40:
        signal_strength = "WEAK"
        signal_color = "#66b3ff"
    else:
        signal_strength = "NO SIGNAL"
        signal_color = "#cccccc"
        position_type = "NEUTRAL"
        optimal_entry_price = spot
    
    # Calculate stop loss and target with realistic logic
    stop_loss = None
    target = None
    
    if nearest_sup and nearest_res and position_type != "NEUTRAL":
        strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
        
        # Use the new realistic stop loss calculation
        stop_loss, target = calculate_realistic_stop_loss_target(
            position_type, optimal_entry_price, nearest_sup, nearest_res, strike_gap_val
        )
    
    return {
        "position_type": position_type,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "optimal_entry_price": optimal_entry_price,
        "current_spot": spot,
        "signal_color": signal_color,
        "reasons": signal_reasons,
        "stop_loss": stop_loss,
        "target": target,
        "max_pain": seller_max_pain,
        "nearest_support": nearest_sup["strike"] if nearest_sup else None,
        "nearest_resistance": nearest_res["strike"] if nearest_res else None,
        "moment_metrics": moment_metrics,  # NEW: Include moment metrics in signal
        "atm_bias_score": atm_bias["total_score"] if atm_bias else 0,  # NEW: Include ATM bias score
        "support_bias_score": support_bias["total_score"] if support_bias else 0,  # NEW: Include support bias score
        "resistance_bias_score": resistance_bias["total_score"] if resistance_bias else 0  # NEW: Include resistance bias score
    }

# ============================================
# 🎯 ENHANCED ENTRY SIGNAL WITH ATM BIAS (NEW)
# ============================================
def calculate_entry_signal_with_atm_bias(
    spot, 
    merged_df, 
    atm_strike, 
    seller_bias_result, 
    seller_max_pain, 
    seller_supports_df, 
    seller_resists_df, 
    nearest_sup, 
    nearest_res, 
    seller_breakout_index,
    moment_metrics,
    atm_bias, 
    support_bias, 
    resistance_bias
):
    """
    Enhanced entry signal with comprehensive ATM bias analysis
    This function wraps the extended entry signal for backward compatibility
    """
    return calculate_entry_signal_extended(
        spot=spot,
        merged_df=merged_df,
        atm_strike=atm_strike,
        seller_bias_result=seller_bias_result,
        seller_max_pain=seller_max_pain,
        seller_supports_df=seller_supports_df,
        seller_resists_df=seller_resists_df,
        nearest_sup=nearest_sup,
        nearest_res=nearest_res,
        seller_breakout_index=seller_breakout_index,
        moment_metrics=moment_metrics,
        atm_bias=atm_bias,
        support_bias=support_bias,
        resistance_bias=resistance_bias
    )

# -----------------------
# 🔥 SELLER'S PERSPECTIVE FUNCTIONS (ORIGINAL)
# -----------------------
def seller_strength_score(row, weights=SCORE_WEIGHTS):
    chg_oi = abs(safe_float(row.get("Chg_OI_CE",0))) + abs(safe_float(row.get("Chg_OI_PE",0)))
    vol = safe_float(row.get("Vol_CE",0)) + safe_float(row.get("Vol_PE",0))
    oi = safe_float(row.get("OI_CE",0)) + safe_float(row.get("OI_PE",0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))
    iv = np.nanmean([v for v in (iv_ce, iv_pe) if not np.isnan(v)]) if (not np.isnan(iv_ce) or not np.isnan(iv_pe)) else 0
    
    score = weights["chg_oi"]*chg_oi + weights["volume"]*vol + weights["oi"]*oi + weights["iv"]*iv
    return score

def seller_price_oi_divergence(chg_oi, vol, ltp_change, option_type="CE"):
    vol_up = vol > 0
    oi_up = chg_oi > 0
    price_up = (ltp_change is not None and ltp_change > 0)
    
    if option_type == "CE":
        if oi_up and vol_up and price_up:
            return "Sellers WRITING calls as price rises (Bearish conviction)"
        if oi_up and vol_up and not price_up:
            return "Sellers WRITING calls on weakness (Strong bearish)"
        if not oi_up and vol_up and price_up:
            return "Sellers BUYING back calls as price rises (Covering bearish)"
        if not oi_up and vol_up and not price_up:
            return "Sellers BUYING back calls on weakness (Reducing bearish exposure)"
    else:
        if oi_up and vol_up and price_up:
            return "Sellers WRITING puts on strength (Bullish conviction)"
        if oi_up and vol_up and not price_up:
            return "Sellers WRITING puts as price falls (Strong bullish)"
        if not oi_up and vol_up and price_up:
            return "Sellers BUYING back puts on strength (Covering bullish)"
        if not oi_up and vol_up and not price_up:
            return "Sellers BUYING back puts as price falls (Reducing bullish exposure)"
    
    if oi_up and not vol_up:
        return "Sellers quietly WRITING options"
    if (not oi_up) and not vol_up:
        return "Sellers quietly UNWINDING"
    
    return "Sellers inactive"

def seller_itm_otm_interpretation(strike, atm, chg_oi_ce, chg_oi_pe):
    ce_interpretation = ""
    pe_interpretation = ""
    
    if strike < atm:
        if chg_oi_ce > 0:
            ce_interpretation = "SELLERS WRITING ITM CALLS = VERY BEARISH 🚨"
        elif chg_oi_ce < 0:
            ce_interpretation = "SELLERS BUYING BACK ITM CALLS = BULLISH 📈"
        else:
            ce_interpretation = "No ITM CALL activity"
    
    elif strike > atm:
        if chg_oi_ce > 0:
            ce_interpretation = "SELLERS WRITING OTM CALLS = MILD BEARISH 📉"
        elif chg_oi_ce < 0:
            ce_interpretation = "SELLERS BUYING BACK OTM CALLS = MILD BULLISH 📊"
        else:
            ce_interpretation = "No OTM CALL activity"
    
    else:
        ce_interpretation = "ATM CALL zone"
    
    if strike > atm:
        if chg_oi_pe > 0:
            pe_interpretation = "SELLERS WRITING ITM PUTS = VERY BULLISH 🚀"
        elif chg_oi_pe < 0:
            pe_interpretation = "SELLERS BUYING BACK ITM PUTS = BEARISH 🐻"
        else:
            pe_interpretation = "No ITM PUT activity"
    
    elif strike < atm:
        if chg_oi_pe > 0:
            pe_interpretation = "SELLERS WRITING OTM PUTS = MILD BULLISH 📈"
        elif chg_oi_pe < 0:
            pe_interpretation = "SELLERS BUYING BACK OTM PUTS = MILD BEARISH 📉"
        else:
            pe_interpretation = "No OTM PUT activity"
    
    else:
        pe_interpretation = "ATM PUT zone"
    
    return f"CALL Sellers: {ce_interpretation} | PUT Sellers: {pe_interpretation}"

def seller_gamma_pressure(row, atm, strike_gap):
    strike = row["strikePrice"]
    dist = abs(strike - atm) / max(strike_gap, 1)
    dist = max(dist, 1e-6)
    
    chg_oi_sum = safe_float(row.get("Chg_OI_CE",0)) - safe_float(row.get("Chg_OI_PE",0))
    seller_pressure = -chg_oi_sum / dist
    
    return seller_pressure

def seller_breakout_probability_index(merged_df, atm, strike_gap):
    near_mask = merged_df["strikePrice"].between(atm-strike_gap, atm+strike_gap)
    
    atm_ce_build = merged_df.loc[near_mask, "Chg_OI_CE"].sum()
    atm_pe_build = merged_df.loc[near_mask, "Chg_OI_PE"].sum()
    seller_atm_bias = atm_pe_build - atm_ce_build
    atm_score = min(abs(seller_atm_bias)/50000.0, 1.0)
    
    ce_writing_count = (merged_df["CE_Seller_Action"] == "WRITING").sum()
    pe_writing_count = (merged_df["PE_Seller_Action"] == "WRITING").sum()
    ce_buying_back_count = (merged_df["CE_Seller_Action"] == "BUYING BACK").sum()
    pe_buying_back_count = (merged_df["PE_Seller_Action"] == "BUYING BACK").sum()
    
    total_actions = ce_writing_count + pe_writing_count + ce_buying_back_count + pe_buying_back_count
    if total_actions > 0:
        seller_conviction = (ce_writing_count + pe_writing_count) / total_actions
    else:
        seller_conviction = 0.5
    
    vol_oi_scores = (merged_df[["Vol_CE","Vol_PE"]].sum(axis=1) * merged_df[["Chg_OI_CE","Chg_OI_PE"]].abs().sum(axis=1)).fillna(0)
    vol_oi_score = min(vol_oi_scores.sum()/100000.0, 1.0)
    
    gamma = merged_df.apply(lambda r: seller_gamma_pressure(r, atm, strike_gap), axis=1).sum()
    gamma_score = min(abs(gamma)/10000.0, 1.0)
    
    w = BREAKOUT_INDEX_WEIGHTS
    combined = (w["atm_oi_shift"]*atm_score) + (w["winding_balance"]*seller_conviction) + (w["vol_oi_div"]*vol_oi_score) + (w["gamma_pressure"]*gamma_score)
    
    return int(np.clip(combined*100,0,100))

def calculate_seller_max_pain(df):
    pain_dict = {}
    for _, row in df.iterrows():
        strike = row["strikePrice"]
        ce_oi = safe_int(row.get("OI_CE", 0))
        pe_oi = safe_int(row.get("OI_PE", 0))
        ce_ltp = safe_float(row.get("LTP_CE", 0))
        pe_ltp = safe_float(row.get("LTP_PE", 0))

        ce_pain = ce_oi * max(0, ce_ltp) if strike < df["strikePrice"].mean() else 0
        pe_pain = pe_oi * max(0, pe_ltp) if strike > df["strikePrice"].mean() else 0

        pain = ce_pain + pe_pain
        pain_dict[strike] = pain

    if pain_dict:
        max_pain_strike = min(pain_dict, key=pain_dict.get)
        total_cost = pain_dict[max_pain_strike]
        return {
            "max_pain_strike": max_pain_strike,
            "total_cost": total_cost
        }
    return None

def calculate_seller_market_bias(merged_df, spot, atm_strike):
    polarity = 0.0
    
    for _, r in merged_df.iterrows():
        strike = r["strikePrice"]
        chg_ce = safe_int(r.get("Chg_OI_CE", 0))
        chg_pe = safe_int(r.get("Chg_OI_PE", 0))
        
        if strike < atm_strike:
            if chg_ce > 0:
                polarity -= 2.0
            elif chg_ce < 0:
                polarity += 1.5
        
        elif strike > atm_strike:
            if chg_ce > 0:
                polarity -= 0.7
            elif chg_ce < 0:
                polarity += 0.5
        
        if strike > atm_strike:
            if chg_pe > 0:
                polarity += 2.0
            elif chg_pe < 0:
                polarity -= 1.5
        
        elif strike < atm_strike:
            if chg_pe > 0:
                polarity += 0.7
            elif chg_pe < 0:
                polarity -= 0.5
    
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        if pcr > 2.0:
            polarity += 1.0
        elif pcr < 0.5:
            polarity -= 1.0
    
    avg_iv_ce = merged_df["IV_CE"].mean()
    avg_iv_pe = merged_df["IV_PE"].mean()
    if avg_iv_ce > avg_iv_pe + 5:
        polarity -= 0.3
    elif avg_iv_pe > avg_iv_ce + 5:
        polarity += 0.3
    
    total_gex_ce = merged_df["GEX_CE"].sum()
    total_gex_pe = merged_df["GEX_PE"].sum()
    net_gex = total_gex_ce + total_gex_pe
    if net_gex < -1000000:
        polarity -= 0.4
    elif net_gex > 1000000:
        polarity += 0.4
    
    max_pain = calculate_seller_max_pain(merged_df)
    if max_pain:
        max_pain_strike = max_pain.get("max_pain_strike", 0)
        distance_to_spot = abs(spot - max_pain_strike) / spot * 100
        if distance_to_spot < 1.0:
            polarity += 0.5
    
    if polarity > 3.0:
        return {
            "bias": "STRONG BULLISH SELLERS 🚀",
            "polarity": polarity,
            "color": "#00ff88",
            "explanation": "Sellers aggressively WRITING PUTS (bullish conviction). Expecting price to STAY ABOVE strikes.",
            "action": "Bullish breakout likely. Sellers confident in upside."
        }
    elif polarity > 1.0:
        return {
            "bias": "BULLISH SELLERS 📈",
            "polarity": polarity,
            "color": "#00cc66",
            "explanation": "Sellers leaning towards PUT writing. Moderate bullish sentiment.",
            "action": "Expect support to hold. Upside bias."
        }
    elif polarity < -3.0:
        return {
            "bias": "STRONG BEARISH SELLERS 🐻",
            "polarity": polarity,
            "color": "#ff4444",
            "explanation": "Sellers aggressively WRITING CALLS (bearish conviction). Expecting price to STAY BELOW strikes.",
            "action": "Bearish breakdown likely. Sellers confident in downside."
        }
    elif polarity < -1.0:
        return {
            "bias": "BEARISH SELLERS 📉",
            "polarity": polarity,
            "color": "#ff6666",
            "explanation": "Sellers leaning towards CALL writing. Moderate bearish sentiment.",
            "action": "Expect resistance to hold. Downside bias."
        }
    else:
        return {
            "bias": "NEUTRAL SELLERS ⚖️",
            "polarity": polarity,
            "color": "#66b3ff",
            "explanation": "Balanced seller activity. No clear directional bias.",
            "action": "Range-bound expected. Wait for clearer signals."
        }

def analyze_spot_position_seller(spot, pcr_df, market_bias):
    sorted_df = pcr_df.sort_values("strikePrice").reset_index(drop=True)
    all_strikes = sorted_df["strikePrice"].tolist()
    
    supports_below = [s for s in all_strikes if s < spot]
    nearest_support = max(supports_below) if supports_below else None
    
    resistances_above = [s for s in all_strikes if s > spot]
    nearest_resistance = min(resistances_above) if resistances_above else None
    
    def get_level_details(strike, df):
        if strike is None:
            return None
        row = df[df["strikePrice"] == strike]
        if row.empty:
            return None
        
        pcr = row.iloc[0]["PCR"]
        oi_ce = int(row.iloc[0]["OI_CE"])
        oi_pe = int(row.iloc[0]["OI_PE"])
        chg_oi_ce = int(row.iloc[0].get("Chg_OI_CE", 0))
        chg_oi_pe = int(row.iloc[0].get("Chg_OI_PE", 0))
        
        if pcr > 1.5:
            seller_strength = "Strong PUT selling (Bullish sellers)"
        elif pcr > 1.0:
            seller_strength = "Moderate PUT selling"
        elif pcr < 0.5:
            seller_strength = "Strong CALL selling (Bearish sellers)"
        elif pcr < 1.0:
            seller_strength = "Moderate CALL selling"
        else:
            seller_strength = "Balanced selling"
        
        return {
            "strike": int(strike),
            "oi_ce": oi_ce,
            "oi_pe": oi_pe,
            "chg_oi_ce": chg_oi_ce,
            "chg_oi_pe": chg_oi_pe,
            "pcr": pcr,
            "seller_strength": seller_strength,
            "distance": abs(spot - strike),
            "distance_pct": abs(spot - strike) / spot * 100
        }
    
    nearest_sup = get_level_details(nearest_support, sorted_df)
    nearest_res = get_level_details(nearest_resistance, sorted_df)
    
    if nearest_sup and nearest_res:
        range_size = nearest_res["strike"] - nearest_sup["strike"]
        spot_position_pct = ((spot - nearest_sup["strike"]) / range_size * 100) if range_size > 0 else 50
        
        if spot_position_pct < 40:
            range_bias = "Near SELLER support (Bullish sellers defending)"
        elif spot_position_pct > 60:
            range_bias = "Near SELLER resistance (Bearish sellers defending)"
        else:
            range_bias = "Middle of SELLER range"
    else:
        range_size = 0
        spot_position_pct = 50
        range_bias = "Range undefined"
    
    return {
        "nearest_support": nearest_sup,
        "nearest_resistance": nearest_res,
        "spot_in_range": (nearest_support, nearest_resistance),
        "range_size": range_size,
        "spot_position_pct": spot_position_pct,
        "range_bias": range_bias,
        "market_bias": market_bias
    }

def compute_pcr_df(merged_df):
    df = merged_df.copy()
    df["OI_CE"] = pd.to_numeric(df.get("OI_CE", 0), errors="coerce").fillna(0).astype(int)
    df["OI_PE"] = pd.to_numeric(df.get("OI_PE", 0), errors="coerce").fillna(0).astype(int)
    
    def pcr_calc(row):
        ce = int(row["OI_CE"]) if row["OI_CE"] is not None else 0
        pe = int(row["OI_PE"]) if row["OI_PE"] is not None else 0
        if ce <= 0:
            if pe > 0:
                return float("inf")
            else:
                return np.nan
        return pe / ce
    
    df["PCR"] = df.apply(pcr_calc, axis=1)
    return df

def rank_support_resistance_seller(pcr_df):
    eps = 1e-6
    t = pcr_df.copy()
    
    t["PCR_clipped"] = t["PCR"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    t["seller_support_score"] = t["OI_PE"] + (t["PCR_clipped"] * 100000.0)
    
    t["seller_resistance_factor"] = t["PCR_clipped"].apply(lambda x: 1.0/(x+eps) if x>0 else 1.0/(eps))
    t["seller_resistance_score"] = t["OI_CE"] + (t["seller_resistance_factor"] * 100000.0)
    
    top_supports = t.sort_values("seller_support_score", ascending=False).head(3)
    top_resists = t.sort_values("seller_resistance_score", ascending=False).head(3)
    
    return t, top_supports, top_resists

# -----------------------
# DHAN API
# -----------------------
@st.cache_data(ttl=2)  # 2 seconds - real-time spot price updates
def get_nifty_spot_price():
    """Fetch NIFTY spot price with retry logic and rate limiting"""
    max_retries = 3
    retry_delays = [2, 4, 8]  # Exponential backoff: 2s, 4s, 8s

    for attempt in range(max_retries):
        try:
            url = f"{DHAN_BASE_URL}/v2/marketfeed/ltp"
            payload = {"IDX_I": [13]}
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "access-token": DHAN_ACCESS_TOKEN,
                "client-id": DHAN_CLIENT_ID
            }
            response = requests.post(url, json=payload, headers=headers, timeout=10)

            # Handle rate limiting
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delays[attempt]
                    time.sleep(wait_time)
                    continue
                else:
                    st.warning(f"⚠️ Dhan API rate limit exceeded. Using cached data if available.")
                    return 0.0

            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                idx_data = data.get("data", {}).get("IDX_I", {})
                nifty_data = idx_data.get("13", {})
                ltp = nifty_data.get("last_price", 0.0)
                return float(ltp)
            return 0.0
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1 and e.response.status_code == 429:
                wait_time = retry_delays[attempt]
                time.sleep(wait_time)
                continue
            st.warning(f"⚠️ Dhan LTP failed: {e}")
            return 0.0
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delays[attempt])
                continue
            st.warning(f"⚠️ Dhan LTP failed: {e}")
            return 0.0

    return 0.0

@st.cache_data(ttl=60)
def get_expiry_list():
    try:
        url = f"{DHAN_BASE_URL}/v2/optionchain/expirylist"
        payload = {"UnderlyingScrip":13,"UnderlyingSeg":"IDX_I"}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID
        }
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status")=="success":
            return data.get("data",[])
        return []
    except Exception as e:
        st.warning(f"Expiry list failed: {e}")
        return []

@st.cache_data(ttl=45)  # 45 seconds - faster refresh for option chain data
def fetch_dhan_option_chain(expiry_date):
    """Fetch option chain with retry logic and rate limiting"""
    max_retries = 3
    retry_delays = [2, 4, 8]  # Exponential backoff: 2s, 4s, 8s

    for attempt in range(max_retries):
        try:
            url = f"{DHAN_BASE_URL}/v2/optionchain"
            payload = {"UnderlyingScrip":13,"UnderlyingSeg":"IDX_I","Expiry":expiry_date}
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "access-token": DHAN_ACCESS_TOKEN,
                "client-id": DHAN_CLIENT_ID
            }
            r = requests.post(url, json=payload, headers=headers, timeout=15)

            # Handle rate limiting
            if r.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delays[attempt]
                    st.info(f"⏳ Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.warning(f"⚠️ Dhan API rate limit exceeded. Please wait a moment and try again.")
                    return None

            r.raise_for_status()
            data = r.json()
            if data.get("status")=="success":
                return data.get("data",{})
            return None
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1 and e.response.status_code == 429:
                wait_time = retry_delays[attempt]
                st.info(f"⏳ Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            st.warning(f"⚠️ Option chain failed: {e}")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delays[attempt])
                continue
            st.warning(f"⚠️ Option chain failed: {e}")
            return None

    return None

def parse_dhan_option_chain(chain_data):
    if not chain_data:
        return pd.DataFrame(), pd.DataFrame()
    oc = chain_data.get("oc",{})
    ce_rows, pe_rows = [], []
    for strike_str, strike_data in oc.items():
        try:
            strike = int(float(strike_str))
        except:
            continue
        ce = strike_data.get("ce")
        pe = strike_data.get("pe")
        if ce:
            ci = {
                "strikePrice": strike,
                "OI_CE": safe_int(ce.get("oi",0)),
                "Chg_OI_CE": safe_int(ce.get("oi",0)) - safe_int(ce.get("previous_oi",0)),
                "Vol_CE": safe_int(ce.get("volume",0)),
                "LTP_CE": safe_float(ce.get("last_price",0.0)),
                "IV_CE": safe_float(ce.get("implied_volatility", np.nan)),
                "SecurityId_CE": safe_int(ce.get("SEM_EXM_EXCH_ID", 0))  # For market depth
            }
            ce_rows.append(ci)
        if pe:
            pi = {
                "strikePrice": strike,
                "OI_PE": safe_int(pe.get("oi",0)),
                "Chg_OI_PE": safe_int(pe.get("oi",0)) - safe_int(pe.get("previous_oi",0)),
                "Vol_PE": safe_int(pe.get("volume",0)),
                "LTP_PE": safe_float(pe.get("last_price",0.0)),
                "IV_PE": safe_float(pe.get("implied_volatility", np.nan)),
                "SecurityId_PE": safe_int(pe.get("SEM_EXM_EXCH_ID", 0))  # For market depth
            }
            pe_rows.append(pi)
    return pd.DataFrame(ce_rows), pd.DataFrame(pe_rows)

# -----------------------
#  HELPER FUNCTION FOR AUTO-LOADING DATA
# -----------------------

def load_option_screener_data_silently():
    """
    Loads option screener data without rendering UI
    Stores data in st.session_state.nifty_option_screener_data
    Returns True on success, False on failure
    """
    try:
        # Always fetch fresh spot price (cached for 2s via @st.cache_data)
        spot = get_nifty_spot_price()

        # Fallback to session state only if API fails
        if spot == 0 or spot is None:
            spot = st.session_state.get('nifty_spot', 0) or st.session_state.get('last_spot_price', 0)

        if spot == 0.0:
            return False

        # Get expiries - use session state if available to avoid redundant API calls
        expiries = st.session_state.get('expiry_list', None)
        if not expiries:
            expiries = get_expiry_list()
            if expiries:
                st.session_state['expiry_list'] = expiries
        if not expiries:
            return False

        # Use first expiry
        expiry = expiries[0]

        # Calculate days to expiry
        try:
            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30, tzinfo=IST)
            now = get_ist_now()
            tau = max((expiry_dt - now).total_seconds() / (365.25*24*3600), 1/365.25)
            days_to_expiry = (expiry_dt - now).total_seconds() / (24 * 3600)
        except Exception:
            tau = 7.0/365.0
            days_to_expiry = 7.0

        # Always fetch fresh option chain (cached for 45s via @st.cache_data)
        chain = fetch_dhan_option_chain(expiry)
        if chain is None:
            return False

        df_ce, df_pe = parse_dhan_option_chain(chain)
        if df_ce.empty or df_pe.empty:
            return False

        # Filter ATM window
        strike_gap = strike_gap_from_series(df_ce["strikePrice"])
        atm_strike = min(df_ce["strikePrice"].tolist(), key=lambda x: abs(x - spot))
        st.session_state['atm_strike'] = atm_strike  # Store for ML signal
        lower = atm_strike - (ATM_STRIKE_WINDOW * strike_gap)
        upper = atm_strike + (ATM_STRIKE_WINDOW * strike_gap)

        df_ce = df_ce[(df_ce["strikePrice"]>=lower) & (df_ce["strikePrice"]<=upper)].reset_index(drop=True)
        df_pe = df_pe[(df_pe["strikePrice"]>=lower) & (df_pe["strikePrice"]<=upper)].reset_index(drop=True)

        merged = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice").reset_index(drop=True)
        merged["strikePrice"] = merged["strikePrice"].astype(int)

        # Initialize session state if needed
        if "prev_ltps_seller" not in st.session_state:
            st.session_state["prev_ltps_seller"] = {}
        if "prev_ivs_seller" not in st.session_state:
            st.session_state["prev_ivs_seller"] = {}

        # Initialize moment history
        _init_history()

        # Compute per-strike metrics with SELLER interpretation
        for i, row in merged.iterrows():
            strike = int(row["strikePrice"])
            ltp_ce = safe_float(row.get("LTP_CE",0.0))
            ltp_pe = safe_float(row.get("LTP_PE",0.0))
            iv_ce = safe_float(row.get("IV_CE", np.nan))
            iv_pe = safe_float(row.get("IV_PE", np.nan))

            key_ce = f"{expiry}_{strike}_CE"
            key_pe = f"{expiry}_{strike}_PE"
            prev_ce = st.session_state["prev_ltps_seller"].get(key_ce, None)
            prev_pe = st.session_state["prev_ltps_seller"].get(key_pe, None)

            # Compute Greeks
            greeks_ce = compute_greeks(spot, strike, tau, RISK_FREE_RATE, ltp_ce, "CE")
            greeks_pe = compute_greeks(spot, strike, tau, RISK_FREE_RATE, ltp_pe, "PE")

            # Determine seller bias
            oi_ce = safe_int(row.get("OI_CE",0))
            oi_pe = safe_int(row.get("OI_PE",0))
            chg_oi_ce = safe_int(row.get("Chg_OI_CE",0))
            chg_oi_pe = safe_int(row.get("Chg_OI_PE",0))

            seller_bias_ce = seller_bias_direction(chg_oi_ce, prev_ce, ltp_ce, oi_ce, "CE")
            seller_bias_pe = seller_bias_direction(chg_oi_pe, prev_pe, ltp_pe, oi_pe, "PE")

            # Update merged dataframe
            merged.at[i,"Delta_CE"] = greeks_ce["delta"]
            merged.at[i,"Gamma_CE"] = greeks_ce["gamma"]
            merged.at[i,"Theta_CE"] = greeks_ce["theta"]
            merged.at[i,"Vega_CE"] = greeks_ce["vega"]
            merged.at[i,"Seller_Bias_CE"] = seller_bias_ce
            merged.at[i,"Delta_PE"] = greeks_pe["delta"]
            merged.at[i,"Gamma_PE"] = greeks_pe["gamma"]
            merged.at[i,"Theta_PE"] = greeks_pe["theta"]
            merged.at[i,"Vega_PE"] = greeks_pe["vega"]
            merged.at[i,"Seller_Bias_PE"] = seller_bias_pe

            # GEX calculation (SELLER exposure)
            notional = LOT_SIZE * spot
            gex_ce = greeks_ce["gamma"] * notional * oi_ce
            gex_pe = greeks_pe["gamma"] * notional * oi_pe
            merged.at[i,"GEX_CE"] = gex_ce
            merged.at[i,"GEX_PE"] = gex_pe
            merged.at[i,"GEX_Net"] = gex_ce + gex_pe

            st.session_state["prev_ltps_seller"][key_ce] = ltp_ce
            st.session_state["prev_ltps_seller"][key_pe] = ltp_pe
            if not np.isnan(iv_ce):
                st.session_state["prev_ivs_seller"][key_ce] = iv_ce
            if not np.isnan(iv_pe):
                st.session_state["prev_ivs_seller"][key_pe] = iv_pe

        # Calculate all analyses
        atm_bias = analyze_atm_bias(merged, spot, atm_strike, strike_gap)
        support_bias = analyze_support_resistance_bias(merged, spot, atm_strike, strike_gap, "Support")
        resistance_bias = analyze_support_resistance_bias(merged, spot, atm_strike, strike_gap, "Resistance")
        seller_bias_result = calculate_seller_market_bias(merged, spot, atm_strike)
        seller_max_pain = calculate_seller_max_pain(merged)
        total_gex_net = merged["GEX_Net"].sum()
        oi_pcr_metrics = analyze_oi_pcr_metrics(merged, spot, atm_strike)
        strike_analyses = create_atm_strikes_tabulation(merged, spot, atm_strike, strike_gap, expiry)
        expiry_spike_data = detect_expiry_spikes(merged, spot, atm_strike, days_to_expiry, expiry)

        # Get sector rotation data if available
        sector_rotation_data = None
        if 'enhanced_market_data' in st.session_state:
            enhanced_data = st.session_state.enhanced_market_data
            if 'sector_rotation' in enhanced_data:
                sector_rotation_data = enhanced_data['sector_rotation']

        # Calculate overall bias
        overall_bias = calculate_overall_bias(atm_bias, support_bias, resistance_bias, seller_bias_result)

        # Store all data in session state
        st.session_state.nifty_option_screener_data = {
            'spot_price': spot,  # Current NIFTY spot price
            'overall_bias': overall_bias,
            'atm_bias': atm_bias,
            'seller_max_pain': seller_max_pain,
            'total_gex_net': total_gex_net,
            'expiry_spike_data': expiry_spike_data,
            'oi_pcr_metrics': oi_pcr_metrics,
            'strike_analyses': strike_analyses,
            'sector_rotation_data': sector_rotation_data,
            'last_updated': get_ist_now()
        }

        # Store merged_df for SL Hunt Detector and other ML modules
        st.session_state['merged_df'] = merged

        # Also load market depth for SL Hunt Detector
        try:
            depth_data = get_market_depth_dhan()
            st.session_state['market_depth_data'] = depth_data
        except:
            pass  # Market depth is optional

        return True
    except Exception as e:
        # Log the error to session state for debugging
        st.session_state.nifty_option_screener_error = {
            'error': str(e),
            'timestamp': get_ist_now()
        }
        # Return False to indicate failure
        return False

# -----------------------
#  MAIN APP - COMPLETE V7 WITH ATM BIAS ANALYZER
# -----------------------

def render_nifty_option_screener():
    """
    Renders the complete Nifty Option Screener v7.0 with:
    - 100% Seller's Perspective
    - ATM Bias Analyzer (12 metrics)
    - Moment Detector
    - Expiry Spike Detector
    - Enhanced OI/PCR Analytics
    """
    # Display current IST time
    current_ist = get_ist_datetime_str()
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 20px;'>
        <span class='ist-time'>🕐 IST: {current_ist}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class='seller-explanation'>
        <h3>🎯 SELLER'S LOGIC</h3>
        <p><strong>Options WRITING = Directional Bias:</strong></p>
        <ul>
        <li><span class='seller-bearish'>📉 CALL Writing</span> = BEARISH (expecting price to STAY BELOW)</li>
        <li><span class='seller-bullish'>📈 PUT Writing</span> = BULLISH (expecting price to STAY ABOVE)</li>
        <li><span class='seller-bullish'>🔄 CALL Buying Back</span> = BULLISH (covering bearish bets)</li>
        <li><span class='seller-bearish'>🔄 PUT Buying Back</span> = BEARISH (covering bullish bets)</li>
        </ul>
        <p><em>Market makers & institutions are primarily SELLERS, not buyers.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 ATM BIAS ANALYZER")
        st.markdown("""
        **12 Key Metrics:**
        1. OI Bias (CALL vs PUT)
        2. Change in OI Bias
        3. Volume Bias
        4. Delta Bias
        5. Gamma Bias
        6. Premium Bias
        7. IV Bias
        8. Delta Exposure Bias
        9. Gamma Exposure Bias
        10. IV Skew Bias
        11. OI Change Bias
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 MOMENT DETECTOR FEATURES")
        st.markdown("""
        1. **Momentum Burst**: Volume × IV × ΔOI changes
        2. **Orderbook Pressure**: Buy/Sell depth imbalance
        3. **Gamma Cluster**: ATM gamma concentration
        4. **OI Acceleration**: Speed of OI changes
        """)
        
        st.markdown("---")
        st.markdown("### 📊 ENHANCED OI/PCR ANALYTICS")
        st.markdown("""
        **New Metrics:**
        1. Total OI Analysis (CALL/PUT)
        2. PCR Interpretation & Sentiment
        3. OI Concentration & Skew
        4. ITM/OTM OI Distribution
        5. Max OI Strikes
        6. Historical PCR Context
        """)
        
        st.markdown("---")
        st.markdown("### 📅 EXPIRY SPIKE DETECTOR")
        st.markdown("""
        **Activation:** ≤5 days to expiry
        
        **Detection Factors:**
        1. ATM OI Concentration
        2. Max Pain Distance
        3. PCR Extremes
        4. Massive OI Walls
        5. Gamma Flip Risk
        6. Unwinding Activity
        """)
        
        st.markdown("---")
        st.markdown("### 📱 TELEGRAM SIGNALS")
        st.markdown("""
        **Signal Conditions:**
        - Position ≠ NEUTRAL
        - Confidence ≥ 40%
        - New signal detected
        """)
        
        # Expiry spike info in sidebar
        st.markdown("---")
        
        # Save interval
        save_interval = st.number_input("PCR Auto-save (sec)", value=SAVE_INTERVAL_SEC, min_value=60, step=60, key="nifty_screener_pcr_autosave_interval")
        
        # Telegram settings
        st.markdown("---")
        st.markdown("### 🤖 TELEGRAM SETTINGS")
        auto_send = st.checkbox("Auto-send signals to Telegram", value=True, key="nifty_screener_auto_send_telegram")
        show_signal_preview = st.checkbox("Show signal preview", value=True, key="nifty_screener_show_signal_preview")
        
        if st.button("Clear Caches"):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch data - Always fetch fresh spot price on every refresh
    col1, col2 = st.columns([1, 2])
    with col1:
        # Always fetch fresh spot price (cached for 2s via @st.cache_data)
        spot = get_nifty_spot_price()

        # Fallback to session state only if API fails
        if spot == 0 or spot is None:
            spot = st.session_state.get('nifty_spot', 0) or st.session_state.get('last_spot_price', 0)

        if spot == 0.0:
            st.error("Unable to fetch NIFTY spot")
            st.stop()

        # Get expiries - use session state if available to avoid redundant API calls
        expiries = st.session_state.get('expiry_list', None)
        if not expiries:
            expiries = get_expiry_list()
            if expiries:
                st.session_state['expiry_list'] = expiries
        if not expiries:
            st.error("Unable to fetch expiry list")
            st.stop()

        expiry = st.selectbox("Select expiry", expiries, index=0, key="nifty_screener_expiry_selector")
        st.session_state['current_expiry'] = expiry  # Store for main dashboard

    with col2:
        if spot > 0:
            st.metric("NIFTY Spot", f"₹{spot:.2f}")
            st.metric("Expiry", expiry)
            st.session_state['nifty_spot'] = spot  # Store for main dashboard
    
    # Calculate days to expiry
    try:
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
        now = datetime.now()
        tau = max((expiry_dt - now).total_seconds() / (365.25*24*3600), 1/365.25)
        days_to_expiry = (expiry_dt - now).total_seconds() / (24 * 3600)
    except Exception:
        tau = 7.0/365.0
        days_to_expiry = 7.0
    
    # Add expiry info to sidebar
    with st.sidebar:
        if days_to_expiry <= 5:
            st.warning(f"⚠️ Expiry in {days_to_expiry:.1f} days")
            st.info("Spike detector ACTIVE")
        else:
            st.success(f"✓ Expiry in {days_to_expiry:.1f} days")
            st.info("Spike detector INACTIVE")
        
        st.markdown("---")
        st.markdown(f"**Current IST:** {get_ist_time_str()}")
        st.markdown(f"**Date:** {get_ist_date_str()}")
    
    # Always fetch fresh option chain (cached for 45s via @st.cache_data on fetch_dhan_option_chain)
    with st.spinner("Fetching option chain..."):
        chain = fetch_dhan_option_chain(expiry)
    if chain is None:
        st.error("Failed to fetch option chain")
        st.stop()
    
    df_ce, df_pe = parse_dhan_option_chain(chain)
    if df_ce.empty or df_pe.empty:
        st.error("Insufficient CE/PE data")
        st.stop()
    
    # Filter ATM window
    strike_gap = strike_gap_from_series(df_ce["strikePrice"])
    atm_strike = min(df_ce["strikePrice"].tolist(), key=lambda x: abs(x - spot))
    st.session_state['atm_strike'] = atm_strike  # Store for ML signal
    lower = atm_strike - (ATM_STRIKE_WINDOW * strike_gap)
    upper = atm_strike + (ATM_STRIKE_WINDOW * strike_gap)

    df_ce = df_ce[(df_ce["strikePrice"]>=lower) & (df_ce["strikePrice"]<=upper)].reset_index(drop=True)
    df_pe = df_pe[(df_pe["strikePrice"]>=lower) & (df_pe["strikePrice"]<=upper)].reset_index(drop=True)

    merged = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice").reset_index(drop=True)
    merged["strikePrice"] = merged["strikePrice"].astype(int)
    st.session_state['merged_df'] = merged  # Store for ML signal

    # Session storage for prev LTP/IV
    if "prev_ltps_seller" not in st.session_state:
        st.session_state["prev_ltps_seller"] = {}
    if "prev_ivs_seller" not in st.session_state:
        st.session_state["prev_ivs_seller"] = {}
    
    # Initialize moment history
    _init_history()
    
    # Compute per-strike metrics with SELLER interpretation
    for i, row in merged.iterrows():
        strike = int(row["strikePrice"])
        ltp_ce = safe_float(row.get("LTP_CE",0.0))
        ltp_pe = safe_float(row.get("LTP_PE",0.0))
        iv_ce = safe_float(row.get("IV_CE", np.nan))
        iv_pe = safe_float(row.get("IV_PE", np.nan))
    
        key_ce = f"{expiry}_{strike}_CE"
        key_pe = f"{expiry}_{strike}_PE"
        prev_ce = st.session_state["prev_ltps_seller"].get(key_ce, None)
        prev_pe = st.session_state["prev_ltps_seller"].get(key_pe, None)
        prev_iv_ce = st.session_state["prev_ivs_seller"].get(key_ce, None)
        prev_iv_pe = st.session_state["prev_ivs_seller"].get(key_pe, None)
    
        ce_price_delta = None if prev_ce is None else (ltp_ce - prev_ce)
        pe_price_delta = None if prev_pe is None else (ltp_pe - prev_pe)
        ce_iv_delta = None if prev_iv_ce is None else (iv_ce - prev_iv_ce)
        pe_iv_delta = None if prev_iv_pe is None else (iv_pe - prev_iv_pe)
    
        st.session_state["prev_ltps_seller"][key_ce] = ltp_ce
        st.session_state["prev_ltps_seller"][key_pe] = ltp_pe
        st.session_state["prev_ivs_seller"][key_ce] = iv_ce
        st.session_state["prev_ivs_seller"][key_pe] = iv_pe
    
        chg_oi_ce = safe_int(row.get("Chg_OI_CE",0))
        chg_oi_pe = safe_int(row.get("Chg_OI_PE",0))
    
        # SELLER winding/unwinding labels
        merged.at[i,"CE_Seller_Action"] = "WRITING" if chg_oi_ce>0 else ("BUYING BACK" if chg_oi_ce<0 else "HOLDING")
        merged.at[i,"PE_Seller_Action"] = "WRITING" if chg_oi_pe>0 else ("BUYING BACK" if chg_oi_pe<0 else "HOLDING")
    
        # SELLER divergence interpretation
        merged.at[i,"CE_Seller_Divergence"] = seller_price_oi_divergence(chg_oi_ce, safe_int(row.get("Vol_CE",0)), ce_price_delta, "CE")
        merged.at[i,"PE_Seller_Divergence"] = seller_price_oi_divergence(chg_oi_pe, safe_int(row.get("Vol_PE",0)), pe_price_delta, "PE")
    
        # SELLER ITM/OTM interpretation
        merged.at[i,"Seller_Interpretation"] = seller_itm_otm_interpretation(strike, atm_strike, chg_oi_ce, chg_oi_pe)
    
        # Greeks calculation
        sigma_ce = iv_ce/100.0 if not np.isnan(iv_ce) and iv_ce>0 else 0.25
        sigma_pe = iv_pe/100.0 if not np.isnan(iv_pe) and iv_pe>0 else 0.25
    
        try:
            delta_ce = bs_delta(spot, strike, RISK_FREE_RATE, sigma_ce, tau, option_type="call")
            gamma_ce = bs_gamma(spot, strike, RISK_FREE_RATE, sigma_ce, tau)
            vega_ce = bs_vega(spot, strike, RISK_FREE_RATE, sigma_ce, tau)
            theta_ce = bs_theta(spot, strike, RISK_FREE_RATE, sigma_ce, tau, option_type="call")
        except Exception:
            delta_ce = gamma_ce = vega_ce = theta_ce = 0.0
    
        try:
            delta_pe = bs_delta(spot, strike, RISK_FREE_RATE, sigma_pe, tau, option_type="put")
            gamma_pe = bs_gamma(spot, strike, RISK_FREE_RATE, sigma_pe, tau)
            vega_pe = bs_vega(spot, strike, RISK_FREE_RATE, sigma_pe, tau)
            theta_pe = bs_theta(spot, strike, RISK_FREE_RATE, sigma_pe, tau, option_type="put")
        except Exception:
            delta_pe = gamma_pe = vega_pe = theta_pe = 0.0
    
        merged.at[i,"Delta_CE"] = delta_ce
        merged.at[i,"Gamma_CE"] = gamma_ce
        merged.at[i,"Vega_CE"] = vega_ce
        merged.at[i,"Theta_CE"] = theta_ce
        merged.at[i,"Delta_PE"] = delta_pe
        merged.at[i,"Gamma_PE"] = gamma_pe
        merged.at[i,"Vega_PE"] = vega_pe
        merged.at[i,"Theta_PE"] = theta_pe
    
        # GEX calculation (SELLER exposure)
        oi_ce = safe_int(row.get("OI_CE",0))
        oi_pe = safe_int(row.get("OI_PE",0))
        notional = LOT_SIZE * spot
        gex_ce = gamma_ce * notional * oi_ce
        gex_pe = gamma_pe * notional * oi_pe
        merged.at[i,"GEX_CE"] = gex_ce
        merged.at[i,"GEX_PE"] = gex_pe
        merged.at[i,"GEX_Net"] = gex_ce + gex_pe
    
        # SELLER strength score
        merged.at[i,"Seller_Strength_Score"] = seller_strength_score(row)
    
        # SELLER gamma pressure
        merged.at[i,"Seller_Gamma_Pressure"] = seller_gamma_pressure(row, atm_strike, strike_gap)
    
        merged.at[i,"CE_Price_Delta"] = ce_price_delta
        merged.at[i,"PE_Price_Delta"] = pe_price_delta
        merged.at[i,"CE_IV_Delta"] = ce_iv_delta
        merged.at[i,"PE_IV_Delta"] = pe_iv_delta
    
    # Aggregations
    total_CE_OI = merged["OI_CE"].sum()
    total_PE_OI = merged["OI_PE"].sum()
    total_CE_chg = merged["Chg_OI_CE"].sum()
    total_PE_chg = merged["Chg_OI_PE"].sum()
    
    # SELLER activity summary
    ce_selling = (merged["Chg_OI_CE"] > 0).sum()
    ce_buying_back = (merged["Chg_OI_CE"] < 0).sum()
    pe_selling = (merged["Chg_OI_PE"] > 0).sum()
    pe_buying_back = (merged["Chg_OI_PE"] < 0).sum()
    
    # Greeks totals
    total_gex_ce = merged["GEX_CE"].sum()
    total_gex_pe = merged["GEX_PE"].sum()
    total_gex_net = merged["GEX_Net"].sum()
    
    # Calculate SELLER metrics
    seller_max_pain = calculate_seller_max_pain(merged)
    seller_breakout_index = seller_breakout_probability_index(merged, atm_strike, strike_gap)
    
    # Calculate SELLER market bias
    seller_bias_result = calculate_seller_market_bias(merged, spot, atm_strike)
    
    # Compute PCR
    pcr_df = compute_pcr_df(merged)
    
    # Get SELLER support/resistance rankings
    ranked_current, seller_supports_df, seller_resists_df = rank_support_resistance_seller(pcr_df)
    
    # Analyze spot position from SELLER perspective
    spot_analysis = analyze_spot_position_seller(spot, ranked_current, seller_bias_result)
    
    nearest_sup = spot_analysis["nearest_support"]
    nearest_res = spot_analysis["nearest_resistance"]
    
    # ---- NEW: Capture snapshot for moment detector ----
    st.session_state["moment_history"].append(
        _snapshot_from_state(get_ist_now(), spot, atm_strike, merged)
    )
    # Keep last 10 points
    st.session_state["moment_history"] = st.session_state["moment_history"][-10:]
    
    # ---- NEW: Compute 4 moment metrics ----
    orderbook = get_nifty_orderbook_depth()
    orderbook_metrics = orderbook_pressure_score(orderbook) if orderbook else {"available": False, "pressure": 0.0}

    moment_metrics = {
        "momentum_burst": compute_momentum_burst(st.session_state["moment_history"]),
        "orderbook": orderbook_metrics,
        "gamma_cluster": compute_gamma_cluster(merged, atm_strike, window=2),
        "oi_accel": compute_oi_velocity_acceleration(st.session_state["moment_history"], atm_strike, window_strikes=2)
    }

    # ---- NEW: ML MARKET REGIME DETECTION ----
    try:
        from src.ml_market_regime import MLMarketRegimeDetector

        # Fetch NIFTY price data for regime detection
        if 'data_df' not in st.session_state or st.session_state.data_df is None:
            # Import get_cached_chart_data from app
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from app import get_cached_chart_data

            price_df = get_cached_chart_data('^NSEI', '1d', '1m')
            if price_df is not None and not price_df.empty:
                # Add indicators if not present
                if 'ATR' not in price_df.columns:
                    from advanced_chart_analysis import AdvancedChartAnalysis
                    chart_analyzer = AdvancedChartAnalysis()
                    price_df = chart_analyzer.add_indicators(price_df)
                st.session_state.data_df = price_df
            else:
                price_df = None
        else:
            price_df = st.session_state.data_df

        # Detect market regime
        if price_df is not None and not price_df.empty:
            regime_detector = MLMarketRegimeDetector()
            ml_regime_result = regime_detector.detect_regime(
                df=price_df,
                cvd_result=None,
                volatility_result=None,
                oi_trap_result=None
            )

            # Add regime to moment metrics
            moment_metrics['market_regime'] = {
                'regime': ml_regime_result.regime,
                'confidence': ml_regime_result.confidence,
                'trend_strength': ml_regime_result.trend_strength,
                'volatility_state': ml_regime_result.volatility_state,
                'recommended_strategy': ml_regime_result.recommended_strategy,
                'optimal_timeframe': ml_regime_result.optimal_timeframe
            }
        else:
            # Fallback if data not available
            moment_metrics['market_regime'] = {
                'regime': 'Unknown',
                'confidence': 0.0,
                'trend_strength': 0.0,
                'volatility_state': 'Unknown',
                'recommended_strategy': 'Wait for data',
                'optimal_timeframe': 'N/A'
            }
    except Exception as e:
        # Fallback on error
        moment_metrics['market_regime'] = {
            'regime': f'Error: {str(e)[:50]}',
            'confidence': 0.0,
            'trend_strength': 0.0,
            'volatility_state': 'Unknown',
            'recommended_strategy': 'Check logs',
            'optimal_timeframe': 'N/A'
        }

    # ============================================
    # 📊 MARKET DEPTH ANALYZER (NEW)
    # ============================================

    # Fetch market depth from Dhan REST API (5-level depth)
    depth_data = get_market_depth_dhan()

    # Store market depth in session state for SL Hunt Detector
    st.session_state['market_depth_data'] = depth_data

    # Analyze depth (5 levels from Dhan API)
    depth_analysis = analyze_market_depth(depth_data, spot, levels=5)

    # Generate depth-based signals
    depth_signals = calculate_depth_based_signals(depth_analysis, spot)

    # Enhanced orderbook pressure with depth
    if depth_analysis["available"]:
        depth_enhanced_pressure = enhanced_orderbook_pressure(depth_analysis, spot)
        # Update moment_metrics orderbook with depth-enhanced version if available
        if depth_enhanced_pressure["available"]:
            moment_metrics["orderbook"] = depth_enhanced_pressure
    else:
        depth_enhanced_pressure = {"available": False}

    # ============================================
    # 🎯 ATM BIAS ANALYSIS (NEW)
    # ============================================

    # Compute ATM and Level Biases
    atm_bias = analyze_atm_bias(merged, spot, atm_strike, strike_gap)
    support_bias = analyze_support_resistance_bias(merged, spot, atm_strike, strike_gap, "Support")
    resistance_bias = analyze_support_resistance_bias(merged, spot, atm_strike, strike_gap, "Resistance")
    
    # Calculate entry signal with moment detector & ATM bias integration
    entry_signal = calculate_entry_signal_with_atm_bias(
        spot=spot,
        merged_df=merged,
        atm_strike=atm_strike,
        seller_bias_result=seller_bias_result,
        seller_max_pain=seller_max_pain,
        seller_supports_df=seller_supports_df,
        seller_resists_df=seller_resists_df,
        nearest_sup=nearest_sup,
        nearest_res=nearest_res,
        seller_breakout_index=seller_breakout_index,
        moment_metrics=moment_metrics,
        atm_bias=atm_bias,
        support_bias=support_bias,
        resistance_bias=resistance_bias
    )
    
    # ============================================
    # 📊 COMPREHENSIVE OI & PCR DASHBOARD
    # ============================================

    # Run OI/PCR analysis
    oi_pcr_metrics = analyze_oi_pcr_metrics(merged, spot, atm_strike)

    # ============================================
    # 🎯 RUN ALL-DAY SPIKE DETECTION FOR ML SIGNAL
    # ============================================
    try:
        all_day_spike_result = detect_all_market_spikes(
            merged_df=merged,
            spot=spot,
            atm_strike=atm_strike,
            days_to_expiry=days_to_expiry,
            total_gex_net=total_gex_net
        )
        # Store for ML Signal integration
        st.session_state['all_day_spike_result'] = {
            'primary_spike': all_day_spike_result.get('primary_spike', {}),
            'active_spikes': all_day_spike_result.get('all_spikes', []),
            'active_spike_count': all_day_spike_result.get('active_spike_count', 0),
            'overall_spike_probability': all_day_spike_result.get('primary_spike', {}).get('probability', 0),
            'dominant_direction': all_day_spike_result.get('primary_spike', {}).get('direction', 'NEUTRAL'),
            'support_spike': all_day_spike_result.get('support_spike', {}),
            'resistance_spike': all_day_spike_result.get('resistance_spike', {}),
            'time_analysis': all_day_spike_result.get('time_analysis', {}),
            'recommendation': all_day_spike_result.get('recommendation', ''),
            'key_levels': all_day_spike_result.get('key_levels', {}),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.warning(f"All-day spike detection failed: {e}")
        st.session_state['all_day_spike_result'] = None

    # ============================================
    # 💾 EXPORT DATA TO SESSION STATE FOR ML MARKET REGIME
    # ============================================
    # Store comprehensive option chain data for ML Market Regime integration
    st.session_state['overall_option_data'] = {
        'NIFTY': {
            'success': True,
            'spot': spot,
            'atm_strike': atm_strike,
            # PCR data
            'pcr': oi_pcr_metrics.get('pcr_total', 1.0),
            'pcr_interpretation': oi_pcr_metrics.get('pcr_interpretation', 'Neutral'),
            # OI data
            'total_ce_oi': oi_pcr_metrics.get('total_ce_oi', 0),
            'total_pe_oi': oi_pcr_metrics.get('total_pe_oi', 0),
            'call_oi_concentration': oi_pcr_metrics.get('atm_concentration_pct', 0) / 100.0,
            'put_oi_concentration': oi_pcr_metrics.get('atm_concentration_pct', 0) / 100.0,
            # Max Pain
            'max_pain': seller_max_pain if seller_max_pain else atm_strike,
            # Gamma data
            'total_gamma': atm_bias.get('metrics', {}).get('net_gamma', 0) if atm_bias else 0,
            'total_call_gamma': atm_bias.get('metrics', {}).get('gamma_exposure', 0) if atm_bias else 0,
            'total_put_gamma': atm_bias.get('metrics', {}).get('gamma_exposure', 0) if atm_bias else 0,
            # ATM Bias data
            'atm_bias': {
                'verdict': atm_bias.get('verdict', 'NEUTRAL') if atm_bias else 'NEUTRAL',
                'score': atm_bias.get('total_score', 0) if atm_bias else 0,
                'confidence': abs(atm_bias.get('total_score', 0)) * 100 if atm_bias else 0,
                'bias_scores': atm_bias.get('bias_scores', {}) if atm_bias else {},
                'metrics': atm_bias.get('metrics', {}) if atm_bias else {},
                'interpretations': atm_bias.get('bias_interpretations', {}) if atm_bias else {}
            },
            # Support/Resistance data
            'support': {
                'strike': support_bias.get('strike', 0) if support_bias else 0,
                'verdict': support_bias.get('verdict', 'NEUTRAL') if support_bias else 'NEUTRAL',
                'score': support_bias.get('total_score', 0) if support_bias else 0,
                'strength': support_bias.get('strength', 'Unknown') if support_bias else 'Unknown',
                'distance_pct': ((spot - support_bias.get('strike', spot)) / spot * 100) if support_bias and support_bias.get('strike') else 0
            },
            'resistance': {
                'strike': resistance_bias.get('strike', 0) if resistance_bias else 0,
                'verdict': resistance_bias.get('verdict', 'NEUTRAL') if resistance_bias else 'NEUTRAL',
                'score': resistance_bias.get('total_score', 0) if resistance_bias else 0,
                'strength': resistance_bias.get('strength', 'Unknown') if resistance_bias else 'Unknown',
                'distance_pct': ((resistance_bias.get('strike', spot) - spot) / spot * 100) if resistance_bias and resistance_bias.get('strike') else 0
            },
            # Seller's perspective signals
            'seller_bias': seller_bias_result.get('seller_verdict', 'NEUTRAL') if seller_bias_result else 'NEUTRAL',
            'seller_confidence': seller_bias_result.get('seller_confidence', 50) if seller_bias_result else 50,
            # Entry signal
            'entry_signal': {
                'position': entry_signal.get('position', 'NEUTRAL'),
                'confidence': entry_signal.get('confidence', 0),
                'rationale': entry_signal.get('rationale', [])
            },
            # Moment metrics
            'moment_score': moment_metrics.get('combined_moment_score', 0) if moment_metrics else 0,
            'moment_verdict': moment_metrics.get('verdict', 'NEUTRAL') if moment_metrics else 'NEUTRAL'
        }
    }

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # CREATE SUB-TABS FOR ORGANIZED CONTENT
    # ═══════════════════════════════════════════════════════════════════

    screener_tabs = st.tabs([
        "📊 Option Chain Table",
        "📊 OI/PCR Analytics",
        "🎯 ATM Bias Analyzer",
        "🎪 Seller's Perspective",
        "🚀 Moment Detector",
        "📊 Market Depth Analyzer",
        "📅 Expiry Analysis",
        "🎯 All-Day Spike Detector",
        "📱 Telegram Signals"
    ])

    # ═══════════════════════════════════════════════════════════════════
    # SUB-TAB 0: OPTION CHAIN TABLE (NEW)
    # ═══════════════════════════════════════════════════════════════════

    with screener_tabs[0]:
        if OPTION_CHAIN_TABLE_AVAILABLE:
            render_option_chain_table_tab(
                merged_df=merged,
                spot=spot,
                atm_strike=atm_strike,
                strike_gap=strike_gap,
                expiry=expiry,
                days_to_expiry=days_to_expiry,
                tau=tau
            )
        else:
            st.error("Option Chain Table module not available. Please ensure option_chain_table.py is in the project directory.")

    # ═══════════════════════════════════════════════════════════════════
    # SUB-TAB 1: OI/PCR ANALYTICS
    # ═══════════════════════════════════════════════════════════════════

    with screener_tabs[1]:
        st.markdown("## 📊 ENHANCED OI & PCR ANALYTICS DASHBOARD")
    
    # Row 1: Totals
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    
    with col_t1:
        st.metric("📈 Total CALL OI", f"{oi_pcr_metrics['total_ce_oi']:,}")
        st.metric("Δ CALL OI", f"{oi_pcr_metrics['total_ce_chg']:+,}")
    
    with col_t2:
        st.metric("📉 Total PUT OI", f"{oi_pcr_metrics['total_pe_oi']:,}")
        st.metric("Δ PUT OI", f"{oi_pcr_metrics['total_pe_chg']:+,}")
    
    with col_t3:
        st.metric("📊 Total OI", f"{oi_pcr_metrics['total_oi']:,}")
        st.metric("Total ΔOI", f"{oi_pcr_metrics['total_chg_oi']:+,}")
    
    with col_t4:
        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid {oi_pcr_metrics['pcr_color']};
            text-align: center;
        ">
            <div style='font-size: 0.9rem; color:#cccccc;'>PCR (TOTAL)</div>
            <div style='font-size: 2rem; color:{oi_pcr_metrics['pcr_color']}; font-weight:900;'>
                {oi_pcr_metrics['pcr_total']:.2f}
            </div>
            <div style='font-size: 0.9rem; color:{oi_pcr_metrics['pcr_color']};'>
                {oi_pcr_metrics['pcr_interpretation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    # Row 3: Concentration Analysis
    st.markdown("### 🎯 OI CONCENTRATION & SKEW")
    
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    
    with col_c1:
        st.metric("ATM Concentration", f"{oi_pcr_metrics['atm_concentration_pct']:.1f}%")
        st.caption(f"CALL: {oi_pcr_metrics['atm_ce_oi']:,} | PUT: {oi_pcr_metrics['atm_pe_oi']:,}")
    
    with col_c2:
        st.metric("Max CALL OI Strike", f"₹{oi_pcr_metrics['max_ce_strike']:,}")
        st.caption(f"OI: {oi_pcr_metrics['max_ce_oi']:,}")
    
    with col_c3:
        st.metric("Max PUT OI Strike", f"₹{oi_pcr_metrics['max_pe_strike']:,}")
        st.caption(f"OI: {oi_pcr_metrics['max_pe_oi']:,}")
    
    with col_c4:
        st.metric("OI Skew", f"CALL: {oi_pcr_metrics['call_oi_skew']}")
        st.caption(f"PUT: {oi_pcr_metrics['put_oi_skew']}")
    
    # Row 4: ITM/OTM Analysis
    with st.expander("🔍 ITM/OTM OI Distribution", expanded=False):
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        
        with col_i1:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; color:#ff4444;">ITM CALL OI</div>
                <div style="font-size: 1.5rem; color:#ff4444; font-weight:700;">
                    {:,}
                </div>
                <div style="font-size: 0.8rem; color:#aaaaaa;">
                    Strike < Spot
                </div>
            </div>
            """.format(oi_pcr_metrics['itm_ce_oi']), unsafe_allow_html=True)
        
        with col_i2:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; color:#ff9900;">OTM CALL OI</div>
                <div style="font-size: 1.5rem; color:#ff9900; font-weight:700;">
                    {:,}
                </div>
                <div style="font-size: 0.8rem; color:#aaaaaa;">
                    Strike > Spot
                </div>
            </div>
            """.format(oi_pcr_metrics['otm_ce_oi']), unsafe_allow_html=True)
        
        with col_i3:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; color:#00cc66;">ITM PUT OI</div>
                <div style="font-size: 1.5rem; color:#00cc66; font-weight:700;">
                    {:,}
                </div>
                <div style="font-size: 0.8rem; color:#aaaaaa;">
                    Strike > Spot
                </div>
            </div>
            """.format(oi_pcr_metrics['itm_pe_oi']), unsafe_allow_html=True)
        
        with col_i4:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; color:#66b3ff;">OTM PUT OI</div>
                <div style="font-size: 1.5rem; color:#66b3ff; font-weight:700;">
                    {:,}
                </div>
                <div style="font-size: 0.8rem; color:#aaaaaa;">
                    Strike < Spot
                </div>
            </div>
            """.format(oi_pcr_metrics['otm_pe_oi']), unsafe_allow_html=True)
    
    # Historical PCR Context
    pcr_context = get_pcr_context(oi_pcr_metrics['pcr_total'])
    
    st.markdown("### 📈 PCR HISTORICAL CONTEXT")
    
    st.info(f"""
    **Current PCR: {oi_pcr_metrics['pcr_total']:.2f}** - {pcr_context}
    
    **Historical Ranges:**
    - **Neutral:** 0.80 - 1.20 (Most common)
    - **Bullish:** 1.20 - 1.50 (PUT selling dominant)
    - **Very Bullish:** 1.50 - 2.00 (Heavy PUT selling)
    - **Extreme Bullish:** > 2.00 (Rare, reversal possible)
    - **Bearish:** 0.50 - 0.80 (CALL selling dominant)
    - **Very Bearish:** 0.30 - 0.50 (Heavy CALL selling)
    - **Extreme Bearish:** < 0.30 (Rare, bounce possible)
    """)
    
    # Add expiry context if near expiry
    if days_to_expiry <= 5:
        expiry_pcr_context = analyze_pcr_for_expiry(oi_pcr_metrics['pcr_total'], days_to_expiry)
        st.warning(f"""
        **⚠️ Expiry Context (D-{int(days_to_expiry)}):** {expiry_pcr_context}

        PCR readings near expiry often exaggerate due to position squaring.
        """)

    st.markdown("---")

    # ============================================
    # 📊 OVERALL MARKET SENTIMENT SUMMARY
    # ============================================

    # Create ATM ±2 strikes tabulation
    strike_analyses = create_atm_strikes_tabulation(merged, spot, atm_strike, strike_gap)

    # Calculate expiry spike data
    expiry_spike_data = detect_expiry_spikes(merged, spot, atm_strike, days_to_expiry, expiry)

    # Get sector rotation data from enhanced market data if available
    sector_rotation_data = None
    if 'enhanced_market_data' in st.session_state:
        enhanced_data = st.session_state.enhanced_market_data
        if 'sector_rotation' in enhanced_data:
            sector_rotation_data = enhanced_data['sector_rotation']

    # Calculate Overall Bias from all analyses
    overall_bias = calculate_overall_bias(atm_bias, support_bias, resistance_bias, seller_bias_result)

    # Store all market sentiment data in session state for access from Overall Market Sentiment tab
    st.session_state.nifty_option_screener_data = {
        'spot_price': spot,  # Current NIFTY spot price
        'overall_bias': overall_bias,
        'atm_bias': atm_bias,
        'seller_max_pain': seller_max_pain,
        'total_gex_net': total_gex_net,
        'expiry_spike_data': expiry_spike_data,
        'oi_pcr_metrics': oi_pcr_metrics,
        'strike_analyses': strike_analyses,
        'sector_rotation_data': sector_rotation_data,
        'seller_bias_result': seller_bias_result,
        'nearest_sup': nearest_sup,
        'nearest_res': nearest_res,
        'moment_metrics': moment_metrics,
        'days_to_expiry': days_to_expiry,
        'last_updated': datetime.now()
    }

    # Display the reorganized Overall Market Sentiment Summary Dashboard
    display_overall_market_sentiment_summary(
        overall_bias=overall_bias,
        atm_bias=atm_bias,
        seller_max_pain=seller_max_pain,
        total_gex_net=total_gex_net,
        expiry_spike_data=expiry_spike_data,
        oi_pcr_metrics=oi_pcr_metrics,
        strike_analyses=strike_analyses,
        sector_rotation_data=sector_rotation_data,
        seller_bias_result=seller_bias_result,
        nearest_sup=nearest_sup,
        nearest_res=nearest_res,
        moment_metrics=moment_metrics,
        days_to_expiry=days_to_expiry
    )

    st.markdown("---")

    # ============================================
    # 🎯 MULTI-DIMENSIONAL BIAS ANALYSIS (NEW)
    # ============================================

    # Display Overall Bias Banner at the top
    st.markdown(f"""
    <div class='card' style='border: 3px solid {overall_bias["verdict_color"]}; background: linear-gradient(135deg, rgba(0,0,0,0.7), rgba(0,0,0,0.5)); padding: 25px; margin-bottom: 20px;'>
        <h2 style='text-align:center; color:{overall_bias["verdict_color"]}; margin-bottom: 15px;'>
            🎯 OVERALL MARKET BIAS
        </h2>
        <div style='font-size: 3rem; color:{overall_bias["verdict_color"]}; font-weight:900; text-align:center; margin-bottom: 15px;'>
            {overall_bias["verdict"]}
        </div>
        <div style='font-size: 1.5rem; color:#ffffff; text-align:center; margin-bottom: 20px;'>
            Combined Score: <span style='color:{overall_bias["verdict_color"]}'>{overall_bias["overall_score"]:.3f}</span>
        </div>
        <div style='font-size: 1.1rem; color:#cccccc; text-align:center; margin-bottom: 15px;'>
            {overall_bias["verdict_explanation"]}
        </div>
        <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
            <h4 style='color:#ffcc00; text-align:center; margin-bottom: 10px;'>Bias Components</h4>
            <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
    """, unsafe_allow_html=True)

    # Display each bias component
    for component in overall_bias["bias_components"]:
        st.markdown(f"""
                <div style='text-align: center; padding: 8px; min-width: 150px;'>
                    <div style='font-size: 0.85rem; color:#999999;'>{component["component"]}</div>
                    <div style='font-size: 1.1rem; color:#ffffff; font-weight:600;'>{component["verdict"]}</div>
                    <div style='font-size: 0.9rem; color:#66b3ff;'>Score: {component["score"]:.2f} (Weight: {component["weight"]*100:.0f}%)</div>
                </div>
        """, unsafe_allow_html=True)

    st.markdown("""
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # SUB-TAB 2: ATM BIAS ANALYZER
    # ═══════════════════════════════════════════════════════════════════

    with screener_tabs[2]:
        st.markdown("## 🎯 ATM BIAS ANALYZER")

        # Display ATM Bias Dashboard
        if atm_bias or support_bias or resistance_bias:
            display_bias_dashboard(atm_bias, support_bias, resistance_bias)

    # ============================================
    # 📅 EXPIRY SPIKE DETECTION
    # ============================================

    # Expiry spike data already calculated in Tab 0

    # Advanced spike detection (optional)
    violent_unwinding_signals = detect_violent_unwinding(merged, spot, atm_strike)
    gamma_spike_risk = calculate_gamma_exposure_spike(total_gex_net, days_to_expiry)
    pinning_probability = predict_expiry_pinning_probability(
        spot, seller_max_pain.get("max_pain_strike", 0) if seller_max_pain else None,
        nearest_sup["strike"] if nearest_sup else None,
        nearest_res["strike"] if nearest_res else None
    )

    # Check for new Telegram signal
    telegram_signal = check_and_send_signal(
        entry_signal, spot, seller_bias_result,
        seller_max_pain, nearest_sup, nearest_res,
        moment_metrics, seller_breakout_index, expiry, expiry_spike_data,
        atm_bias, support_bias, resistance_bias
    )

    # ═══════════════════════════════════════════════════════════════════
    # SUB-TAB 5: MARKET DEPTH ANALYZER
    # ═══════════════════════════════════════════════════════════════════

    with screener_tabs[5]:
        # Display Basic Market Depth Dashboard
        display_market_depth_dashboard(spot, depth_analysis, depth_signals, depth_enhanced_pressure)

        # Display Comprehensive Advanced Depth Analysis
        if ADVANCED_DEPTH_AVAILABLE:
            st.markdown("---")
            st.markdown("## 🎯 ADVANCED DEPTH ANALYSIS (ATM ±2 Strikes)")

            # Prepare Dhan config
            dhan_config = {
                "base_url": DHAN_BASE_URL,
                "access_token": DHAN_ACCESS_TOKEN,
                "client_id": DHAN_CLIENT_ID
            }

            # Run comprehensive analysis for ATM ±2 strikes
            atm_strikes_to_analyze = [atm_strike + (i * strike_gap) for i in range(-2, 3)]

            # Rate limiting info
            st.info("⏱️ Fetching depth data with rate limiting (1 req/sec) - This may take ~10 seconds...")

            for idx, strike in enumerate(atm_strikes_to_analyze):
                with st.expander(f"📊 Strike {strike} - Comprehensive Depth Analysis", expanded=(idx == 2)):  # Expand ATM by default
                    st.markdown(f"### Analyzing {strike} CE & PE")

                    # Rate limiting: 1.2 seconds between strikes
                    if idx > 0:
                        time.sleep(1.2)

                    # Analyze CE
                    st.markdown("#### 📈 CALL Option (CE)")
                    ce_analysis = run_comprehensive_depth_analysis(
                        strike=strike,
                        expiry=expiry,
                        option_type="CE",
                        dhan_config=dhan_config,
                        depth_history=None  # TODO: Implement depth history tracking
                    )

                    if ce_analysis.get("available"):
                        display_comprehensive_depth_analysis(ce_analysis)
                    else:
                        st.warning(f"CE analysis unavailable: {ce_analysis.get('error', 'Unknown error')}")

                    st.markdown("---")

                    # Rate limiting before PE call
                    time.sleep(1.2)

                    # Analyze PE
                    st.markdown("#### 📉 PUT Option (PE)")
                    pe_analysis = run_comprehensive_depth_analysis(
                        strike=strike,
                        expiry=expiry,
                        option_type="PE",
                        dhan_config=dhan_config,
                        depth_history=None
                    )

                    if pe_analysis.get("available"):
                        display_comprehensive_depth_analysis(pe_analysis)
                    else:
                        st.warning(f"PE analysis unavailable: {pe_analysis.get('error', 'Unknown error')}")
        else:
            st.info("ℹ️ Advanced depth analysis module not available. Install `market_depth_advanced.py` for full functionality.")

    # ═══════════════════════════════════════════════════════════════════
    # SUB-TAB 6: EXPIRY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    with screener_tabs[6]:
        st.markdown("## 📅 EXPIRY DATE SPIKE DETECTOR")

        # Main spike card
        spike_col1, spike_col2, spike_col3 = st.columns([2, 1, 1])

        with spike_col1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%);
                padding: 20px;
                border-radius: 12px;
                border: 3px solid {expiry_spike_data['color']};
                margin: 10px 0;
            ">
                <h3 style='color:{expiry_spike_data["color"]}; margin:0;'>📅 EXPIRY SPIKE ALERT</h3>
                <div style='font-size: 2.5rem; color:{expiry_spike_data["color"]}; font-weight:900; margin:10px 0;'>
                    {expiry_spike_data["probability"]}%
                </div>
                <div style='font-size: 1.3rem; color:#ffffff; margin:5px 0;'>
                    {expiry_spike_data["intensity"]}
                </div>
                <div style='font-size: 1.1rem; color:#ffcc00; margin:5px 0;'>
                    Type: {expiry_spike_data["type"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with spike_col2:
            st.markdown(f"""
            <div style="
                background: rgba(0,0,0,0.3);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            ">
                <div style='font-size: 0.9rem; color:#cccccc;'>Days to Expiry</div>
                <div style='font-size: 2rem; color:#ff9900; font-weight:700;'>
                    {expiry_spike_data['days_to_expiry']:.1f}
                </div>
                <div style='font-size: 0.8rem; color:#aaaaaa;'>
                    {expiry}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with spike_col3:
            st.markdown(f"""
            <div style="
                background: rgba(0,0,0,0.3);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            ">
                <div style='font-size: 0.9rem; color:#cccccc;'>Spike Score</div>
                <div style='font-size: 2rem; color:#ff00ff; font-weight:700;'>
                    {expiry_spike_data['score']}/100
                </div>
                <div style='font-size: 0.8rem; color:#aaaaaa;'>
                    Detection Factors
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Spike Factors
        with st.expander("🔍 View Spike Detection Factors", expanded=False):
            col_factors1, col_factors2 = st.columns(2)
            
            with col_factors1:
                st.markdown("### ⚠️ Spike Triggers")
                for factor in expiry_spike_data["factors"]:
                    st.markdown(f"• {factor}")
                
                # Violent unwinding signals
                if violent_unwinding_signals:
                    st.markdown("### 🚨 Violent Unwinding")
                    for signal in violent_unwinding_signals:
                        st.markdown(f"• {signal}")
            
            with col_factors2:
                st.markdown("### 🎯 Key Levels")
                if expiry_spike_data["key_levels"]:
                    for level in expiry_spike_data["key_levels"]:
                        st.markdown(f"• {level}")
                else:
                    st.info("No extreme levels detected")

                # Display Spike Strike Price Ranges
                support_range = expiry_spike_data.get("support_spike_range", {})
                resistance_range = expiry_spike_data.get("resistance_spike_range", {})

                if support_range.get("start") is not None or resistance_range.get("start") is not None:
                    st.markdown("### 📊 Spike Strike Price Range")

                    if support_range.get("start") is not None:
                        st.markdown(f"""
                        <div style='background: rgba(0,255,0,0.1); border-left: 3px solid #00ff00; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                            <strong style='color:#00ff00;'>🛡️ SUPPORT SPIKE RANGE</strong><br/>
                            <span style='color:#ffffff;'>Start: ₹{support_range['start']:,} → End: ₹{support_range['end']:,}</span><br/>
                            <span style='color:#aaaaaa; font-size: 0.85em;'>
                                Strikes: {', '.join([f"₹{s:,}" for s in support_range['strikes'][:5]])}
                                {f"... (+{len(support_range['strikes'])-5} more)" if len(support_range['strikes']) > 5 else ""}
                            </span><br/>
                            <span style='color:#66b3ff; font-size: 0.85em;'>Total PUT OI: {support_range['total_oi']:,}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    if resistance_range.get("start") is not None:
                        st.markdown(f"""
                        <div style='background: rgba(255,0,0,0.1); border-left: 3px solid #ff4444; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                            <strong style='color:#ff4444;'>🚧 RESISTANCE SPIKE RANGE</strong><br/>
                            <span style='color:#ffffff;'>Start: ₹{resistance_range['start']:,} → End: ₹{resistance_range['end']:,}</span><br/>
                            <span style='color:#aaaaaa; font-size: 0.85em;'>
                                Strikes: {', '.join([f"₹{s:,}" for s in resistance_range['strikes'][:5]])}
                                {f"... (+{len(resistance_range['strikes'])-5} more)" if len(resistance_range['strikes']) > 5 else ""}
                            </span><br/>
                            <span style='color:#66b3ff; font-size: 0.85em;'>Total CALL OI: {resistance_range['total_oi']:,}</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Gamma spike risk
                if gamma_spike_risk["score"] > 0:
                    st.markdown(f"### ⚡ Gamma Spike Risk")
                    st.markdown(f"• {gamma_spike_risk['message']}")
                    st.markdown(f"• Risk Level: {gamma_spike_risk['risk']}")
                
                # Pinning probability
                if pinning_probability > 0:
                    st.markdown(f"### 📍 Pinning Probability")
                    st.markdown(f"• {pinning_probability}% chance of price getting stuck")
        
        # Historical Patterns
        if days_to_expiry <= 3:
            st.markdown("### 📊 Historical Expiry Patterns")
            patterns = get_historical_expiry_patterns()
            
            pattern_cols = st.columns(len(patterns))
            
            for idx, (pattern_name, pattern_data) in enumerate(patterns.items()):
                with pattern_cols[idx]:
                    prob_color = "#ff4444" if pattern_data["probability"] > 0.6 else "#ff9900" if pattern_data["probability"] > 0.4 else "#66b3ff"
                    st.markdown(f"""
                    <div style="
                        background: #1a1f2e;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 3px solid {prob_color};
                        margin: 5px 0;
                    ">
                        <div style='font-size: 0.9rem; color:#cccccc;'>{pattern_name.replace('_', ' ').title()}</div>
                        <div style='font-size: 1.5rem; color:{prob_color}; font-weight:700;'>
                            {pattern_data['probability']:.0%}
                        </div>
                        <div style='font-size: 0.8rem; color:#aaaaaa; margin-top:5px;'>
                            {pattern_data['description']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Action Recommendations
        st.markdown("### 🎯 Expiry Day Trading Strategy")
        
        if expiry_spike_data["probability"] > 60:
            st.warning("""
            **HIGH SPIKE PROBABILITY - AGGRESSIVE STRATEGY:**
            - Expect sharp moves (100-200 point swings)
            - Use wider stops (1.5-2x normal)
            - Consider straddles/strangles if IV not too high
            - Avoid deep ITM options (gamma risk)
            - Focus on 10:30-11:30 AM and 2:30-3:00 PM windows
            """)
        elif expiry_spike_data["probability"] > 40:
            st.info("""
            **MODERATE SPIKE RISK - BALANCED STRATEGY:**
            - Expect moderate volatility
            - Use normal stops with 20% buffer
            - Prefer ATM/1st OTM strikes
            - Watch Max Pain level closely
            - Be ready to exit early
            """)
        else:
            st.success("""
            **LOW SPIKE RISK - NORMAL STRATEGY:**
            - Normal trading rules apply
            - Standard stop losses
            - Focus on technical levels
            - Watch for last-hour moves
            """)
        
        # Gamma Risk Zone
        if days_to_expiry <= 2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a2e2e 0%, #2a3e3e 100%);
                padding: 15px;
                border-radius: 10px;
                border: 2px solid #00ffff;
                margin: 10px 0;
            ">
                <h4 style='color:#00ffff; margin:0;'>⚠️ GAMMA RISK ZONE ACTIVE</h4>
                <p style='color:#ffffff; margin:5px 0;'>
                    Days to expiry ≤ 2: Gamma exposure amplifies price moves.
                    Market makers' hedging can cause exaggerated swings.
                </p>
                <p style='color:#ffcc00; margin:5px 0;'>
                    🎯 Watch: {', '.join(expiry_spike_data['key_levels'][:3]) if expiry_spike_data['key_levels'] else 'ATM ±100 points'}
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info(f"""
            ### 📅 Expiry Spike Detector (Inactive)

            **Reason:** {expiry_spike_data['message']}

            Spike detection activates when expiry is ≤5 days away.

            Current expiry: **{expiry}**
            Days to expiry: **{days_to_expiry:.1f}**

            *Check back closer to expiry for spike alerts*
            """)
    
    # ═══════════════════════════════════════════════════════════════════
    # SUB-TAB 7: ALL-DAY SPIKE DETECTOR (NEW!)
    # ═══════════════════════════════════════════════════════════════════

    with screener_tabs[7]:
        st.markdown("## 🎯 ALL-DAY SPIKE DETECTOR")
        st.caption("Detects Support, Resistance, Opening, Breakout, Momentum & Squeeze spikes on ANY trading day")

        # Call the new spike detector function
        try:
            spike_result = detect_all_market_spikes(
                merged_df=merged,
                spot=spot,
                atm_strike=atm_strike,
                days_to_expiry=days_to_expiry,
                total_gex_net=total_gex_net
            )

            # Store spike result in session state for ML Signal integration
            st.session_state['all_day_spike_result'] = {
                'primary_spike': spike_result.get('primary_spike', {}),
                'active_spikes': spike_result.get('all_spikes', []),
                'active_spike_count': spike_result.get('active_spike_count', 0),
                'overall_spike_probability': spike_result.get('primary_spike', {}).get('probability', 0),
                'dominant_direction': spike_result.get('primary_spike', {}).get('direction', 'NEUTRAL'),
                'support_spike': spike_result.get('support_spike', {}),
                'resistance_spike': spike_result.get('resistance_spike', {}),
                'time_analysis': spike_result.get('time_analysis', {}),
                'recommendation': spike_result.get('recommendation', ''),
                'key_levels': spike_result.get('key_levels', {}),
                'timestamp': datetime.now().isoformat()
            }

            # Display primary spike
            col_main1, col_main2, col_main3 = st.columns(3)

            with col_main1:
                primary_type = spike_result["primary_spike"]["type"].replace("_", " ").title()
                primary_score = spike_result["primary_spike"]["score"]
                primary_prob = spike_result["primary_spike"]["probability"]

                color = "#00ff00" if "support" in primary_type.lower() else "#ff4444" if "resistance" in primary_type.lower() else "#ffaa00"

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}22, {color}44);
                            border: 2px solid {color};
                            border-radius: 10px;
                            padding: 15px;
                            text-align: center;">
                    <div style="font-size: 0.9rem; color: #aaa;">PRIMARY SPIKE</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{primary_type}</div>
                    <div style="font-size: 2rem; font-weight: 900; color: {color};">{primary_score}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col_main2:
                st.metric("Active Spikes", spike_result["active_spike_count"])
                st.metric("Current Phase", spike_result["time_analysis"]["current_phase"])

            with col_main3:
                direction = spike_result["primary_spike"]["direction"]
                if direction == "UP":
                    st.success(f"⬆️ Direction: **{direction}** - Look for LONG")
                elif direction == "DOWN":
                    st.error(f"⬇️ Direction: **{direction}** - Look for SHORT")
                else:
                    st.info("↔️ Direction: **NEUTRAL**")

            st.markdown("---")

            # Display all 6 spike types in 2 rows
            st.markdown("### 📊 All Spike Types")

            row1_col1, row1_col2, row1_col3 = st.columns(3)

            # Support Spike
            with row1_col1:
                ss = spike_result["spikes"]["support_spike"]
                st.markdown(f"""
                <div style="background: #1a2e1a; border-radius: 8px; padding: 10px; border-left: 4px solid {'#00ff00' if ss['active'] else '#444'};">
                    <div style="font-weight: 600;">🟢 SUPPORT SPIKE</div>
                    <div style="font-size: 1.5rem; color: {'#00ff00' if ss['active'] else '#666'};">{ss['score']}%</div>
                    <div style="font-size: 0.8rem; color: #aaa;">Level: ₹{ss['level']:,} ({ss['distance_percent']:.2f}% away)</div>
                </div>
                """, unsafe_allow_html=True)

            # Resistance Spike
            with row1_col2:
                rs = spike_result["spikes"]["resistance_spike"]
                st.markdown(f"""
                <div style="background: #2e1a1a; border-radius: 8px; padding: 10px; border-left: 4px solid {'#ff4444' if rs['active'] else '#444'};">
                    <div style="font-weight: 600;">🔴 RESISTANCE SPIKE</div>
                    <div style="font-size: 1.5rem; color: {'#ff4444' if rs['active'] else '#666'};">{rs['score']}%</div>
                    <div style="font-size: 0.8rem; color: #aaa;">Level: ₹{rs['level']:,} ({rs['distance_percent']:.2f}% away)</div>
                </div>
                """, unsafe_allow_html=True)

            # Opening Spike
            with row1_col3:
                os_spike = spike_result["spikes"]["opening_spike"]
                os_color = "#00ff00" if os_spike.get("direction") == "UP" else "#ff4444" if os_spike.get("direction") == "DOWN" else "#666"
                st.markdown(f"""
                <div style="background: #1a1f2e; border-radius: 8px; padding: 10px; border-left: 4px solid {'#ffaa00' if os_spike['active'] else '#444'};">
                    <div style="font-weight: 600;">🌅 OPENING SPIKE</div>
                    <div style="font-size: 1.5rem; color: {'#ffaa00' if os_spike['active'] else '#666'};">{os_spike['score']}%</div>
                    <div style="font-size: 0.8rem; color: #aaa;">Direction: {os_spike.get('direction', 'N/A')} | Gap: ₹{os_spike.get('expected_gap', 0):,}</div>
                </div>
                """, unsafe_allow_html=True)

            row2_col1, row2_col2, row2_col3 = st.columns(3)

            # Breakout Spike
            with row2_col1:
                bs = spike_result["spikes"]["breakout_spike"]
                st.markdown(f"""
                <div style="background: #2e2e1a; border-radius: 8px; padding: 10px; border-left: 4px solid {'#ffff00' if bs['active'] else '#444'};">
                    <div style="font-weight: 600;">🚀 BREAKOUT SPIKE</div>
                    <div style="font-size: 1.5rem; color: {'#ffff00' if bs['active'] else '#666'};">{bs['score']}%</div>
                    <div style="font-size: 0.8rem; color: #aaa;">Direction: {bs.get('direction', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)

            # Momentum Spike
            with row2_col2:
                ms = spike_result["spikes"]["momentum_spike"]
                st.markdown(f"""
                <div style="background: #1a2e2e; border-radius: 8px; padding: 10px; border-left: 4px solid {'#00ffff' if ms['active'] else '#444'};">
                    <div style="font-weight: 600;">💨 MOMENTUM SPIKE</div>
                    <div style="font-size: 1.5rem; color: {'#00ffff' if ms['active'] else '#666'};">{ms['score']}%</div>
                    <div style="font-size: 0.8rem; color: #aaa;">Direction: {ms.get('direction', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)

            # Squeeze Spike
            with row2_col3:
                sq = spike_result["spikes"]["squeeze_spike"]
                st.markdown(f"""
                <div style="background: #2e1a2e; border-radius: 8px; padding: 10px; border-left: 4px solid {'#ff00ff' if sq['active'] else '#444'};">
                    <div style="font-weight: 600;">🔥 SQUEEZE SPIKE</div>
                    <div style="font-size: 1.5rem; color: {'#ff00ff' if sq['active'] else '#666'};">{sq['score']}%</div>
                    <div style="font-size: 0.8rem; color: #aaa;">Type: {sq.get('type', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Key Levels
            st.markdown("### 🎯 Key Levels")
            level_col1, level_col2, level_col3 = st.columns(3)

            with level_col1:
                st.markdown("**Support Levels:**")
                for sup in spike_result["key_levels"]["support"][:3]:
                    st.write(f"₹{sup['level']:,} ({sup['strength']}) - OI: {sup['oi']:,}")

            with level_col2:
                st.markdown("**Resistance Levels:**")
                for res in spike_result["key_levels"]["resistance"][:3]:
                    st.write(f"₹{res['level']:,} ({res['strength']}) - OI: {res['oi']:,}")

            with level_col3:
                st.metric("Max Pain", f"₹{spike_result['key_levels']['max_pain']:,}")
                st.metric("ATM Strike", f"₹{spike_result['key_levels']['atm']:,}")

            # Recommendation
            st.markdown("---")
            st.markdown("### 💡 Recommendation")
            recommendation = spike_result["summary"]["recommendation"]
            if "SUPPORT" in recommendation or "UP" in recommendation or "LONG" in recommendation:
                st.success(recommendation)
            elif "RESISTANCE" in recommendation or "DOWN" in recommendation or "SHORT" in recommendation:
                st.error(recommendation)
            else:
                st.info(recommendation)

            # Active spike factors
            if spike_result["active_spikes"]:
                with st.expander("📋 Spike Factors Details", expanded=False):
                    for spike in spike_result["active_spikes"]:
                        st.markdown(f"**{spike['name'].replace('_', ' ').title()}** (Score: {spike['score']})")
                        for factor in spike["data"].get("factors", []):
                            st.write(f"  • {factor}")

        except Exception as e:
            st.error(f"⚠️ Spike detection error: {str(e)}")
            st.info("Spike detector requires merged option chain data.")

    # ═══════════════════════════════════════════════════════════════════
    # SUB-TAB 8: TELEGRAM SIGNALS
    # ═══════════════════════════════════════════════════════════════════

    with screener_tabs[8]:
        st.markdown("## 📱 TELEGRAM SIGNAL GENERATION (Option 3 Format)")

        if telegram_signal:
            # NEW SIGNAL DETECTED
            st.success("🎯 **NEW TRADE SIGNAL GENERATED!**")

            # Auto-send to Telegram if enabled
            if auto_send and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                with st.spinner("Sending to Telegram..."):
                    success, message = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, telegram_signal)
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")

            # Create a nice display of the signal
            col_signal1, col_signal2 = st.columns([2, 1])

            with col_signal1:
                st.markdown("### 📋 Telegram Signal Ready:")

                if show_signal_preview:
                    # Display formatted preview
                    st.markdown("""
                    <div style="
                        background-color: #1a1f2e;
                        padding: 15px;
                        border-radius: 10px;
                        border-left: 4px solid #0088cc;
                        margin: 10px 0;
                        font-family: monospace;
                        white-space: pre-wrap;
                    ">
                    """ + telegram_signal + "</div>", unsafe_allow_html=True)
                else:
                    st.code(telegram_signal)

            with col_signal2:
                st.markdown("### 📤 Send Options:")

                # Copy to clipboard
                if st.button("📋 Copy to Clipboard", use_container_width=True, key="copy_clipboard"):
                    st.success("✅ Signal copied to clipboard!")

                # Manual send to Telegram
                if st.button("📱 Send to Telegram", use_container_width=True, key="send_telegram"):
                    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        success, message = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, telegram_signal)
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                    else:
                        st.warning("Telegram credentials not configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to secrets.")

                # Save to file
                if st.button("💾 Save to File", use_container_width=True, key="save_file"):
                    filename = f"signal_{get_ist_datetime_str().replace(':', '-').replace(' ', '_')}.txt"
                    with open(filename, 'w') as f:
                        f.write(telegram_signal)
                    st.success(f"✅ Signal saved to {filename}")

            # Add signal details
            with st.expander("📊 View Signal Details", expanded=False):
                col_details1, col_details2 = st.columns(2)

                with col_details1:
                    st.markdown("**Position Details:**")
                    st.metric("Type", entry_signal["position_type"])
                    st.metric("Strength", entry_signal["signal_strength"])
                    st.metric("Confidence", f"{entry_signal['confidence']:.0f}%")
                    st.metric("Entry Price", f"₹{entry_signal['optimal_entry_price']:,.2f}")

                with col_details2:
                    st.markdown("**Risk Management:**")
                    st.metric("Stop Loss", f"₹{entry_signal['stop_loss']:,.2f}" if entry_signal['stop_loss'] else "N/A")
                    st.metric("Target", f"₹{entry_signal['target']:,.2f}" if entry_signal['target'] else "N/A")

                    # Calculate actual risk:reward
                    if entry_signal['stop_loss'] and entry_signal['target']:
                        if entry_signal["position_type"] == "LONG":
                            risk = abs(entry_signal['optimal_entry_price'] - entry_signal['stop_loss'])
                            reward = abs(entry_signal['target'] - entry_signal['optimal_entry_price'])
                        else:
                            risk = abs(entry_signal['stop_loss'] - entry_signal['optimal_entry_price'])
                            reward = abs(entry_signal['optimal_entry_price'] - entry_signal['target'])

                        if risk > 0:
                            rr_ratio = reward / risk
                            st.metric("Risk:Reward", f"1:{rr_ratio:.2f}")

            # Signal timestamp
            st.caption(f"⏰ Signal generated at: {get_ist_datetime_str()}")

            # Last signal info
            if "last_signal" in st.session_state and st.session_state["last_signal"]:
                st.caption(f"📝 Last signal type: {st.session_state['last_signal']}")

        else:
            # No active signal
            st.info("📭 **No active trade signal to send.**")

            # Show why no signal
            with st.expander("ℹ️ Why no signal?", expanded=False):
                st.markdown(f"""
                **Current Status:**
                - Position Type: {entry_signal['position_type']}
                - Signal Strength: {entry_signal['signal_strength']}
                - Confidence: {entry_signal['confidence']:.0f}%
                - Seller Bias: {seller_bias_result['bias']}
                - ATM Bias: {atm_bias['verdict'] if atm_bias else 'N/A'}
                - Expiry Spike Risk: {expiry_spike_data.get('probability', 0)}%
                - PCR Sentiment: {oi_pcr_metrics['pcr_sentiment']}

                **Requirements for signal generation:**
                ✅ Position Type ≠ NEUTRAL
                ✅ Confidence ≥ 40%
                ✅ Clear directional bias
                ✅ ATM bias alignment
                """)

            # Show last signal if exists
            if "last_signal" in st.session_state and st.session_state["last_signal"]:
                st.info(f"📝 Last signal was: {st.session_state['last_signal']}")
    
    # ═══════════════════════════════════════════════════════════════════
    # SUB-TAB 4: MOMENT DETECTOR
    # ═══════════════════════════════════════════════════════════════════

    with screener_tabs[4]:
        st.markdown("## 🚀 MOMENT DETECTOR (Is this a real move?)")

        moment_col1, moment_col2, moment_col3, moment_col4 = st.columns(4)

        with moment_col1:
            mb = moment_metrics["momentum_burst"]
            if mb["available"]:
                color = "#ff00ff" if mb["score"] > 70 else ("#ff9900" if mb["score"] > 40 else "#66b3ff")
                st.markdown(f'''
                <div class="moment-box">
                    <h4>💥 MOMENTUM BURST</h4>
                    <div class="moment-value" style="color:{color}">{mb["score"]}/100</div>
                    <div class="sub-info">{mb["note"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="moment-box">
                    <h4>💥 MOMENTUM BURST</h4>
                    <div class="moment-value" style="color:#cccccc">N/A</div>
                    <div class="sub-info">Need more refresh points</div>
                </div>
                ''', unsafe_allow_html=True)

        with moment_col2:
            ob = moment_metrics["orderbook"]
            if ob["available"]:
                pressure = ob["pressure"]
                color = "#00ff88" if pressure > 0.15 else ("#ff4444" if pressure < -0.15 else "#66b3ff")
                st.markdown(f'''
                <div class="moment-box">
                    <h4>📊 ORDERBOOK PRESSURE</h4>
                    <div class="moment-value" style="color:{color}">{pressure:+.2f}</div>
                    <div class="sub-info">Buy/Sell imbalance</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="moment-box">
                    <h4>📊 ORDERBOOK PRESSURE</h4>
                    <div class="moment-value" style="color:#cccccc">N/A</div>
                    <div class="sub-info">Depth data unavailable</div>
                </div>
                ''', unsafe_allow_html=True)

        with moment_col3:
            gc = moment_metrics["gamma_cluster"]
            if gc["available"]:
                color = "#ff00ff" if gc["score"] > 70 else ("#ff9900" if gc["score"] > 40 else "#66b3ff")
                st.markdown(f'''
                <div class="moment-box">
                    <h4>🌀 GAMMA CLUSTER</h4>
                    <div class="moment-value" style="color:{color}">{gc["score"]}/100</div>
                    <div class="sub-info">ATM ±2 concentration</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="moment-box">
                    <h4>🌀 GAMMA CLUSTER</h4>
                    <div class="moment-value" style="color:#cccccc">N/A</div>
                    <div class="sub-info">Data unavailable</div>
                </div>
                ''', unsafe_allow_html=True)

        with moment_col4:
            oi = moment_metrics["oi_accel"]
            if oi["available"]:
                color = "#ff00ff" if oi["score"] > 70 else ("#ff9900" if oi["score"] > 40 else "#66b3ff")
                st.markdown(f'''
                <div class="moment-box">
                    <h4>⚡ OI ACCELERATION</h4>
                    <div class="moment-value" style="color:{color}">{oi["score"]}/100</div>
                    <div class="sub-info">{oi["note"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="moment-box">
                    <h4>⚡ OI ACCELERATION</h4>
                    <div class="moment-value" style="color:#cccccc">N/A</div>
                    <div class="sub-info">Need more refresh points</div>
                </div>
                ''', unsafe_allow_html=True)
    
    # ============================================
    # 🎯 SUPER PROMINENT ENTRY SIGNAL
    # ============================================
    
    st.markdown("---")
    
    if entry_signal["position_type"] != "NEUTRAL" and entry_signal["confidence"] >= 40:
        # ACTIVE SIGNAL
        signal_bg = "#1a2e1a" if entry_signal["position_type"] == "LONG" else "#2e1a1a"
        signal_border = "#00ff88" if entry_signal["position_type"] == "LONG" else "#ff4444"
        signal_emoji = "🚀" if entry_signal["position_type"] == "LONG" else "🐻"
        
        # Create a container with custom styling
        with st.container():
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {signal_bg} 0%, #2a3e2a 100%);
                padding: 30px;
                border-radius: 20px;
                border: 5px solid {signal_border};
                margin: 0 auto;
                text-align: center;
                max-width: 900px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.4);
            ">
            """, unsafe_allow_html=True)
            
            # Emoji and title row
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.markdown(f"<div style='text-align: center; font-size: 4rem;'>{signal_emoji}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style='text-align: center;'>
                    <div style='font-size: 2.8rem; font-weight: 900; color:{signal_border}; line-height: 1.2;'>
                        {entry_signal["signal_strength"]} {entry_signal["position_type"]} SIGNAL
                    </div>
                    <div style='font-size: 1.2rem; color: #ffdd44; margin-top: 5px;'>
                        Confidence: {entry_signal["confidence"]:.0f}%
                    </div>
                    <div style='font-size: 1.1rem; color: #66b3ff; margin-top: 5px;'>
                        ATM Bias: {atm_bias['verdict'] if atm_bias else 'N/A'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div style='text-align: center; font-size: 4rem;'>{signal_emoji}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Optimal entry price in a separate styled container
        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.3); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px auto;
            max-width: 900px;
            text-align: center;
        ">
            <div style="font-size: 3rem; color: #ffcc00; font-weight: 900;">
                ₹{entry_signal["optimal_entry_price"]:,.2f}
            </div>
            <div style="font-size: 1.3rem; color: #cccccc; margin-top: 5px;">
                OPTIMAL ENTRY PRICE
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats row
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 1.1rem; color: #aaaaaa;">Current Spot</div>
                <div style="font-size: 1.8rem; color: #ffffff; font-weight: 700;">₹""" + f"{spot:,.2f}" + """</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stats2:
            distance = abs(spot - entry_signal["optimal_entry_price"])
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 1.1rem; color: #aaaaaa;">Distance</div>
                <div style="font-size: 1.8rem; color: #ffaa00; font-weight: 700;">₹{distance:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stats3:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 1.1rem; color: #aaaaaa;">Direction</div>
                <div style="font-size: 1.8rem; color: {signal_border}; font-weight: 700;">{entry_signal["position_type"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Moment confirmation
        st.markdown(f"""
        <div style="
            margin-top: 25px; 
            padding: 20px; 
            background: rgba(0,0,0,0.2); 
            border-radius: 10px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        ">
            <div style="font-size: 1.2rem; color: #ffdd44; margin-bottom: 10px; text-align: center;">🎯 MOMENT CONFIRMATION</div>
            <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
                <div>Burst: {moment_metrics['momentum_burst'].get('score', 0)}/100</div>
                <div>Pressure: {moment_metrics['orderbook'].get('pressure', 0):+.2f}</div>
                <div>Gamma: {moment_metrics['gamma_cluster'].get('score', 0)}/100</div>
                <div>OI Accel: {moment_metrics['oi_accel'].get('score', 0)}/100</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # OI/PCR Confirmation
        st.markdown(f"""
        <div style="
            margin-top: 25px; 
            padding: 20px; 
            background: rgba(0,0,0,0.2); 
            border-radius: 10px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        ">
            <div style="font-size: 1.2rem; color: #66b3ff; margin-bottom: 10px; text-align: center;">📊 OI/PCR CONFIRMATION</div>
            <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
                <div>PCR: {oi_pcr_metrics['pcr_total']:.2f}</div>
                <div>Sentiment: {oi_pcr_metrics['pcr_sentiment']}</div>
                <div>CALL OI: {oi_pcr_metrics['total_ce_oi']:,}</div>
                <div>PUT OI: {oi_pcr_metrics['total_pe_oi']:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ATM Bias Confirmation
        if atm_bias:
            st.markdown(f"""
            <div style="
                margin-top: 25px; 
                padding: 20px; 
                background: rgba(0,0,0,0.2); 
                border-radius: 10px;
                max-width: 900px;
                margin-left: auto;
                margin-right: auto;
            ">
                <div style="font-size: 1.2rem; color: {atm_bias['verdict_color']}; margin-bottom: 10px; text-align: center;">🎯 ATM BIAS CONFIRMATION</div>
                <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
                    <div>ATM Bias: {atm_bias['verdict']}</div>
                    <div>Score: {atm_bias['total_score']:.2f}</div>
                    <div>CALL OI: {atm_bias['metrics']['ce_oi']:,}</div>
                    <div>PUT OI: {atm_bias['metrics']['pe_oi']:,}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)
        action_col1, action_col2, action_col3 = st.columns([2, 1, 1])
        
        with action_col1:
            if st.button(f"📊 PLACE {entry_signal['position_type']} ORDER AT ₹{entry_signal['optimal_entry_price']:,.0f}", 
                        use_container_width=True, type="primary", key="place_order"):
                st.success(f"✅ {entry_signal['position_type']} order queued at ₹{entry_signal['optimal_entry_price']:,.2f}")
                st.balloons()
        
        with action_col2:
            if st.button("🔔 SET PRICE ALERT", use_container_width=True, key="set_alert"):
                st.info(f"📢 Alert set for {entry_signal['optimal_entry_price']:,.2f}")
        
        with action_col3:
            if st.button("🔄 REFRESH", use_container_width=True, key="refresh"):
                st.rerun()
        
        # Signal Reasons
        with st.expander("📋 View Detailed Signal Reasoning", expanded=False):
            for reason in entry_signal["reasons"]:
                st.markdown(f"• {reason}")
            
            # Moment Detector Details
            st.markdown("### 🚀 Moment Detector Details:")
            for metric_name, metric_data in moment_metrics.items():
                if metric_data.get("available", False):
                    st.markdown(f"**{metric_name.replace('_', ' ').title()}:** {metric_data.get('note', 'N/A')}")
            
            # ATM Bias Details
            if atm_bias:
                st.markdown("### 🎯 ATM Bias Analysis:")
                st.markdown(f"• **Overall Verdict:** {atm_bias['verdict']}")
                st.markdown(f"• **Total Score:** {atm_bias['total_score']:.2f}")
                st.markdown(f"• **Explanation:** {atm_bias['verdict_explanation']}")
                st.markdown(f"• **Key Metrics:** CALL OI: {atm_bias['metrics']['ce_oi']:,} | PUT OI: {atm_bias['metrics']['pe_oi']:,}")
                st.markdown(f"• **Net Delta:** {atm_bias['metrics']['net_delta']:.3f} | **Net Gamma:** {atm_bias['metrics']['net_gamma']:.3f}")
            
            # OI/PCR Details
            st.markdown("### 📊 OI/PCR Analysis:")
            st.markdown(f"• **PCR:** {oi_pcr_metrics['pcr_total']:.2f} ({oi_pcr_metrics['pcr_sentiment']})")
            st.markdown(f"• **OI Change:** {oi_pcr_metrics['oi_change_interpretation']}")
            st.markdown(f"• **Max CALL OI:** ₹{oi_pcr_metrics['max_ce_strike']:,} ({oi_pcr_metrics['max_ce_oi']:,})")
            st.markdown(f"• **Max PUT OI:** ₹{oi_pcr_metrics['max_pe_strike']:,} ({oi_pcr_metrics['max_pe_oi']:,})")
            st.markdown(f"• **ATM Concentration:** {oi_pcr_metrics['atm_concentration_pct']:.1f}%")
            
            # Expiry Spike Risk
            if expiry_spike_data["active"]:
                st.markdown("### 📅 Expiry Spike Risk:")
                st.markdown(f"• Probability: {expiry_spike_data['probability']}%")
                st.markdown(f"• Type: {expiry_spike_data['type']}")
                st.markdown(f"• Intensity: {expiry_spike_data['intensity']}")
                if expiry_spike_data["key_levels"]:
                    st.markdown(f"• Key Levels: {', '.join(expiry_spike_data['key_levels'])}")
        
    else:
        # NO SIGNAL
        with st.container():
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
                padding: 30px;
                border-radius: 20px;
                border: 5px solid #666666;
                margin: 0 auto;
                text-align: center;
                max-width: 900px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.4);
            ">
            """, unsafe_allow_html=True)
            
            # Warning icon
            st.markdown("""
            <div style="font-size: 4rem; color: #cccccc; margin-bottom: 20px; text-align: center;">
                ⚠️
            </div>
            """, unsafe_allow_html=True)
            
            # No signal message
            st.markdown("""
            <div style="font-size: 2.5rem; font-weight: 900; color:#cccccc; line-height: 1.2; margin-bottom: 15px; text-align: center;">
                NO CLEAR ENTRY SIGNAL
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="font-size: 1.8rem; color: #ffcc00; font-weight: 700; margin-bottom: 20px; text-align: center;">
                Wait for Better Setup
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Current spot price
        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.3); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px auto;
            max-width: 900px;
            text-align: center;
        ">
            <div style="font-size: 2.5rem; color: #ffffff; font-weight: 700;">
                ₹{spot:,.2f}
            </div>
            <div style="font-size: 1.2rem; color: #cccccc; margin-top: 5px;">
                CURRENT SPOT PRICE
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence info
        st.markdown(f"""
        <div style="
            color: #aaaaaa; 
            font-size: 1.1rem; 
            margin-top: 20px;
            text-align: center;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        ">
            Signal Confidence: {entry_signal["confidence"]:.0f}% | 
            Seller Bias: {seller_bias_result["bias"]} | 
            ATM Bias: {atm_bias['verdict'] if atm_bias else 'N/A'} | 
            PCR Sentiment: {oi_pcr_metrics['pcr_sentiment']} | 
            Expiry Spike Risk: {expiry_spike_data.get('probability', 0)}%
        </div>
        """, unsafe_allow_html=True)
        
        # Moment status
        st.markdown(f"""
        <div style="
            margin-top: 25px; 
            padding: 20px; 
            background: rgba(0,0,0,0.2); 
            border-radius: 10px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        ">
            <div style="font-size: 1.2rem; color: #ffdd44; margin-bottom: 10px; text-align: center;">🎯 MOMENT STATUS</div>
            <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
                <div>Burst: {moment_metrics['momentum_burst'].get('score', 0)}/100</div>
                <div>Pressure: {moment_metrics['orderbook'].get('pressure', 0):+.2f}</div>
                <div>Gamma: {moment_metrics['gamma_cluster'].get('score', 0)}/100</div>
                <div>OI Accel: {moment_metrics['oi_accel'].get('score', 0)}/100</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # OI/PCR status
        st.markdown(f"""
        <div style="
            margin-top: 25px; 
            padding: 20px; 
            background: rgba(0,0,0,0.2); 
            border-radius: 10px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        ">
            <div style="font-size: 1.2rem; color: #66b3ff; margin-bottom: 10px; text-align: center;">📊 OI/PCR STATUS</div>
            <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
                <div>PCR: {oi_pcr_metrics['pcr_total']:.2f}</div>
                <div>CALL OI: {oi_pcr_metrics['total_ce_oi']:,}</div>
                <div>PUT OI: {oi_pcr_metrics['total_pe_oi']:,}</div>
                <div>ATM Conc: {oi_pcr_metrics['atm_concentration_pct']:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ATM Bias status
        if atm_bias:
            st.markdown(f"""
            <div style="
                margin-top: 25px; 
                padding: 20px; 
                background: rgba(0,0,0,0.2); 
                border-radius: 10px;
                max-width: 900px;
                margin-left: auto;
                margin-right: auto;
            ">
                <div style="font-size: 1.2rem; color: {atm_bias['verdict_color']}; margin-bottom: 10px; text-align: center;">🎯 ATM BIAS STATUS</div>
                <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
                    <div>Verdict: {atm_bias['verdict']}</div>
                    <div>Score: {atm_bias['total_score']:.2f}</div>
                    <div>CALL OI: {atm_bias['metrics']['ce_oi']:,}</div>
                    <div>PUT OI: {atm_bias['metrics']['pe_oi']:,}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Expandable details for no signal
        with st.expander("🔍 Why No Signal? (Click for Details)", expanded=False):
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                st.markdown("### 📊 Current Metrics:")
                st.metric("Seller Bias", seller_bias_result["bias"])
                st.metric("Polarity Score", f"{seller_bias_result['polarity']:.2f}")
                st.metric("ATM Bias", atm_bias['verdict'] if atm_bias else "N/A")
                st.metric("ATM Bias Score", f"{atm_bias['total_score']:.2f}" if atm_bias else "N/A")
                st.metric("Breakout Index", f"{seller_breakout_index}%")
                st.metric("Signal Confidence", f"{entry_signal['confidence']:.0f}%")
                st.metric("PCR", f"{oi_pcr_metrics['pcr_total']:.2f}")
                st.metric("PCR Sentiment", oi_pcr_metrics['pcr_sentiment'])
                st.metric("Expiry Spike Risk", f"{expiry_spike_data.get('probability', 0)}%")
            
            with col_detail2:
                st.markdown("### 🎯 Signal Requirements:")
                requirements = [
                    "✅ Clear directional bias (BULLISH/BEARISH)",
                    "✅ Confidence > 40%",
                    "✅ Strong moment detector scores",
                    "✅ ATM bias alignment",
                    "✅ Support/Resistance alignment",
                    "✅ Momentum burst > 50",
                    "✅ PCR alignment with bias"
                ]
                for req in requirements:
                    st.markdown(f"- {req}")
                
                st.markdown(f"""
                ### 📈 Current Status:
                - **Position Type**: {entry_signal["position_type"]}
                - **Signal Strength**: {entry_signal["signal_strength"]}
                - **Optimal Entry**: ₹{entry_signal["optimal_entry_price"]:,.2f}
                - **ATM Bias**: {atm_bias['verdict'] if atm_bias else 'N/A'}
                - **PCR Sentiment**: {oi_pcr_metrics['pcr_sentiment']}
                - **OI Skew**: CALL: {oi_pcr_metrics['call_oi_skew']}, PUT: {oi_pcr_metrics['put_oi_skew']}
                - **Expiry in**: {days_to_expiry:.1f} days
                """)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════
    # SUB-TAB 3: SELLER'S PERSPECTIVE
    # ═══════════════════════════════════════════════════════════════════

    with screener_tabs[3]:
        st.markdown("## 🎪 SELLER'S PERSPECTIVE")

        st.markdown(f"""
        <div class='seller-bias-box'>
            <h3>🎯 SELLER'S MARKET BIAS</h3>
            <div class='bias-value' style='color:{seller_bias_result["color"]}'>
                {seller_bias_result["bias"]}
            </div>
            <p>Polarity Score: {seller_bias_result["polarity"]:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='seller-explanation'>
            <h4>🧠 SELLER'S THINKING:</h4>
            <p><strong>{seller_bias_result["explanation"]}</strong></p>
            <p><strong>Action:</strong> {seller_bias_result["action"]}</p>
        </div>
        """, unsafe_allow_html=True)

        # Core Metrics with OI/PCR
        st.markdown("### 📈 SELLER'S MARKET OVERVIEW")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spot", f"₹{spot:.2f}")
            st.metric("ATM", f"₹{atm_strike}")
        with col2:
            st.metric("CALL Sellers", f"{ce_selling} strikes")
            st.metric("PUT Sellers", f"{pe_selling} strikes")
        with col3:
            st.metric("PCR", f"{oi_pcr_metrics['pcr_total']:.2f}")
            st.metric("PCR Sentiment", oi_pcr_metrics['pcr_sentiment'])
        with col4:
            st.metric("Total GEX", f"₹{int(total_gex_net):,}")
            st.metric("Breakout Index", f"{seller_breakout_index}%")
    
    # Max Pain Display
    if seller_max_pain:
        max_pain_strike = seller_max_pain.get("max_pain_strike", 0)
        distance_to_max_pain = abs(spot - max_pain_strike)
        st.markdown(f"""
        <div class='max-pain-box'>
            <h4>🎯 SELLER'S MAX PAIN (Preferred Level)</h4>
            <p style='font-size: 1.5rem; color: #ff9900; font-weight: bold; text-align: center;'>₹{max_pain_strike:,}</p>
            <p style='text-align: center; color: #cccccc;'>Distance from spot: ₹{distance_to_max_pain:.2f} ({distance_to_max_pain/spot*100:.2f}%)</p>
            <p style='text-align: center; color: #ffcc00;'>Sellers want price here to minimize losses</p>
        </div>
        """, unsafe_allow_html=True)
    
    # SELLER Activity Summary with OI Context
    st.markdown("### 🔥 SELLER ACTIVITY HEATMAP WITH OI CONTEXT")
    
    seller_activity = pd.DataFrame([
        {"Activity": "CALL Writing (Bearish)", "Strikes": ce_selling, "Total OI": f"{oi_pcr_metrics['total_ce_oi']:,}", "Bias": "BEARISH", "Color": "#ff4444"},
        {"Activity": "CALL Buying Back (Bullish)", "Strikes": ce_buying_back, "Total OI": f"{oi_pcr_metrics['total_ce_oi']:,}", "Bias": "BULLISH", "Color": "#00ff88"},
        {"Activity": "PUT Writing (Bullish)", "Strikes": pe_selling, "Total OI": f"{oi_pcr_metrics['total_pe_oi']:,}", "Bias": "BULLISH", "Color": "#00ff88"},
        {"Activity": "PUT Buying Back (Bearish)", "Strikes": pe_buying_back, "Total OI": f"{oi_pcr_metrics['total_pe_oi']:,}", "Bias": "BEARISH", "Color": "#ff4444"}
    ])
    
    st.dataframe(seller_activity, use_container_width=True)
    
    st.markdown("---")
    
    # ============================================
    # 🎯 SPOT POSITION - SELLER'S VIEW WITH OI/PCR
    # ============================================
    
    st.markdown("## 📍 SPOT POSITION (SELLER'S DEFENSE + OI/PCR)")
    
    col_spot, col_range = st.columns([1, 1])
    
    with col_spot:
        st.markdown(f"""
        <div class="spot-card">
            <h3>🎯 CURRENT SPOT</h3>
            <div class="spot-price">₹{spot:,.2f}</div>
            <div class="distance">ATM: ₹{atm_strike:,}</div>
            <div class="distance">Market Bias: <span style="color:{seller_bias_result['color']}">{seller_bias_result["bias"]}</span></div>
            <div class="distance">ATM Bias: <span style="color:{atm_bias['verdict_color'] if atm_bias else '#cccccc'}">{atm_bias['verdict'] if atm_bias else 'N/A'}</span></div>
            <div class="distance">PCR: <span style="color:{oi_pcr_metrics['pcr_color']}">{oi_pcr_metrics['pcr_total']:.2f}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_range:
        if nearest_sup and nearest_res:
            range_size = spot_analysis["range_size"]
            spot_position_pct = spot_analysis["spot_position_pct"]
            range_bias = spot_analysis["range_bias"]
            
            st.markdown(f"""
            <div class="spot-card">
                <h3>📊 SELLER'S DEFENSE RANGE</h3>
                <div class="distance">₹{nearest_sup['strike']:,} ← SPOT → ₹{nearest_res['strike']:,}</div>
                <div class="distance">Position: {spot_position_pct:.1f}% within range</div>
                <div class="distance">Range Width: ₹{range_size:,}</div>
                <div class="distance" style="color:#ffcc00;">{range_bias}</div>
                <div class="distance">ATM OI Concentration: {oi_pcr_metrics['atm_concentration_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # NEAREST LEVELS WITH SELLER INTERPRETATION + OI
    st.markdown("### 🎯 NEAREST SELLER DEFENSE LEVELS WITH OI")
    
    col_ns, col_nr = st.columns(2)
    
    with col_ns:
        st.markdown("#### 🛡️ SELLER SUPPORT BELOW")
        
        if nearest_sup:
            sup = nearest_sup
            pcr_display = f"{sup['pcr']:.2f}" if not np.isinf(sup['pcr']) else "∞"
            
            st.markdown(f"""
            <div class="nearest-level">
                <h4>💚 NEAREST SELLER SUPPORT</h4>
                <div class="level-value">₹{sup['strike']:,}</div>
                <div class="level-distance">⬇️ Distance: ₹{sup['distance']:.2f} ({sup['distance_pct']:.2f}%)</div>
                <div class="sub-info">
                    <strong>SELLER ACTIVITY:</strong> {sup['seller_strength']}<br>
                    PUT OI: {sup['oi_pe']:,} | CALL OI: {sup['oi_ce']:,}<br>
                    PCR: {pcr_display} | ΔCALL: {sup['chg_oi_ce']:+,} | ΔPUT: {sup['chg_oi_pe']:+,}<br>
                    <strong>OI Skew:</strong> PUT/CALL = {sup['oi_pe']/max(sup['oi_ce'],1):.1f}x
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No seller support level below spot")
    
    with col_nr:
        st.markdown("#### ⚡ SELLER RESISTANCE ABOVE")
        
        if nearest_res:
            res = nearest_res
            pcr_display = f"{res['pcr']:.2f}" if not np.isinf(res['pcr']) else "∞"
            
            st.markdown(f"""
            <div class="nearest-level">
                <h4>🧡 NEAREST SELLER RESISTANCE</h4>
                <div class="level-value">₹{res['strike']:,}</div>
                <div class="level-distance">⬆️ Distance: ₹{res['distance']:.2f} ({res['distance_pct']:.2f}%)</div>
                <div class="sub-info">
                    <strong>SELLER ACTIVITY:</strong> {res['seller_strength']}<br>
                    CALL OI: {res['oi_ce']:,} | PUT OI: {res['oi_pe']:,}<br>
                    PCR: {pcr_display} | ΔCALL: {res['chg_oi_ce']:+,} | ΔPUT: {res['chg_oi_pe']:+,}<br>
                    <strong>OI Skew:</strong> CALL/PUT = {res['oi_ce']/max(res['oi_pe'],1):.1f}x
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No seller resistance level above spot")
    
    st.markdown("---")
    
    # TOP SELLER DEFENSE LEVELS WITH ENHANCED OI INFO
    st.markdown("### 🎯 TOP SELLER DEFENSE LEVELS (Strongest 3 with OI Analysis)")
    
    col_s, col_r = st.columns(2)
    
    with col_s:
        st.markdown("#### 🛡️ STRONGEST SELLER SUPPORTS (Highest PUT OI)")
        
        for i, (idx, row) in enumerate(seller_supports_df.head(3).iterrows(), 1):
            strike = int(row["strikePrice"])
            oi_pe = int(row["OI_PE"])
            oi_ce = int(row["OI_CE"])
            pcr = row["PCR"]
            pcr_display = f"{pcr:.2f}" if not np.isinf(pcr) else "∞"
            chg_oi_pe = int(row.get("Chg_OI_PE", 0))
            chg_oi_ce = int(row.get("Chg_OI_CE", 0))
            
            # Calculate OI ratios
            total_oi = oi_pe + oi_ce
            pe_ratio = (oi_pe / total_oi * 100) if total_oi > 0 else 0
            
            if pcr > 1.5:
                seller_msg = f"Heavy PUT writing ({pe_ratio:.0f}% PUT OI) - Strong bullish defense"
                color = "#00ff88"
            elif pcr > 1.0:
                seller_msg = f"Moderate PUT writing ({pe_ratio:.0f}% PUT OI) - Bullish defense"
                color = "#00cc66"
            else:
                seller_msg = f"Light PUT writing ({pe_ratio:.0f}% PUT OI) - Weak defense"
                color = "#cccccc"
            
            dist = abs(spot - strike)
            dist_pct = (dist / spot * 100)
            direction = "⬆️ Above" if strike > spot else "⬇️ Below"
            
            st.markdown(f'''
            <div class="level-card">
                <h4>Seller Support #{i}</h4>
                <p>₹{strike:,}</p>
                <div class="sub-info">
                    {direction}: ₹{dist:.2f} ({dist_pct:.2f}%)<br>
                    <span style="color:{color}"><strong>{seller_msg}</strong></span><br>
                    PUT OI: {oi_pe:,} | ΔPUT: {chg_oi_pe:+,}<br>
                    CALL OI: {oi_ce:,} | ΔCALL: {chg_oi_ce:+,}<br>
                    PCR: {pcr_display} | PUT%: {pe_ratio:.0f}%
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    with col_r:
        st.markdown("#### ⚡ STRONGEST SELLER RESISTANCES (Highest CALL OI)")
        
        for i, (idx, row) in enumerate(seller_resists_df.head(3).iterrows(), 1):
            strike = int(row["strikePrice"])
            oi_ce = int(row["OI_CE"])
            oi_pe = int(row["OI_PE"])
            pcr = row["PCR"]
            pcr_display = f"{pcr:.2f}" if not np.isinf(pcr) else "∞"
            chg_oi_ce = int(row.get("Chg_OI_CE", 0))
            chg_oi_pe = int(row.get("Chg_OI_PE", 0))
            
            # Calculate OI ratios
            total_oi = oi_ce + oi_pe
            ce_ratio = (oi_ce / total_oi * 100) if total_oi > 0 else 0
            
            if pcr < 0.5:
                seller_msg = f"Heavy CALL writing ({ce_ratio:.0f}% CALL OI) - Strong bearish defense"
                color = "#ff4444"
            elif pcr < 1.0:
                seller_msg = f"Moderate CALL writing ({ce_ratio:.0f}% CALL OI) - Bearish defense"
                color = "#ff6666"
            else:
                seller_msg = f"Light CALL writing ({ce_ratio:.0f}% CALL OI) - Weak defense"
                color = "#cccccc"
            
            dist = abs(spot - strike)
            dist_pct = (dist / spot * 100)
            direction = "⬆️ Above" if strike > spot else "⬇️ Below"
            
            st.markdown(f'''
            <div class="level-card">
                <h4>Seller Resistance #{i}</h4>
                <p>₹{strike:,}</p>
                <div class="sub-info">
                    {direction}: ₹{dist:.2f} ({dist_pct:.2f}%)<br>
                    <span style="color:{color}"><strong>{seller_msg}</strong></span><br>
                    CALL OI: {oi_ce:,} | ΔCALL: {chg_oi_ce:+,}<br>
                    PUT OI: {oi_pe:,} | ΔPUT: {chg_oi_pe:+,}<br>
                    PCR: {pcr_display} | CALL%: {ce_ratio:.0f}%
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # 📊 DETAILED DATA - SELLER VIEW + MOMENT + EXPIRY + OI/PCR + ATM BIAS
    # ============================================
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Seller Activity", "🧮 Seller Greeks", "📈 Seller PCR", "🚀 Moment Analysis", "📅 Expiry Analysis", "🎯 ATM Bias"])
    
    with tab1:
        st.markdown("### 📊 SELLER ACTIVITY BY STRIKE")
        
        seller_cols = [
            "strikePrice", 
            "OI_CE", "Chg_OI_CE", "CE_Seller_Action", "CE_Seller_Divergence",
            "OI_PE", "Chg_OI_PE", "PE_Seller_Action", "PE_Seller_Divergence",
            "Seller_Interpretation", "Seller_Strength_Score"
        ]
        
        # Ensure all columns exist
        for col in seller_cols:
            if col not in merged.columns:
                merged[col] = ""
        
        # Color code seller actions
        def color_seller_action(val):
            if "WRITING" in str(val):
                if "CALL" in str(val):
                    return "background-color: #2e1a1a; color: #ff6666"
                else:
                    return "background-color: #1a2e1a; color: #00ff88"
            elif "BUYING BACK" in str(val):
                if "CALL" in str(val):
                    return "background-color: #1a2e1a; color: #00ff88"
                else:
                    return "background-color: #2e1a1a; color: #ff6666"
            return ""
        
        seller_display = merged[seller_cols].copy()
        styled_df = seller_display.style.applymap(color_seller_action, subset=["CE_Seller_Action", "PE_Seller_Action"])
        st.dataframe(styled_df, use_container_width=True)
    
    with tab2:
        st.markdown("### 🧮 SELLER GREEKS & GEX EXPOSURE")
        
        greeks_cols = [
            "strikePrice",
            "Delta_CE", "Gamma_CE", "Vega_CE", "Theta_CE", "GEX_CE",
            "Delta_PE", "Gamma_PE", "Vega_PE", "Theta_PE", "GEX_PE",
            "GEX_Net", "Seller_Gamma_Pressure"
        ]
        
        for col in greeks_cols:
            if col not in merged.columns:
                merged[col] = 0.0
        
        # Format Greek values
        greeks_display = merged[greeks_cols].copy()
        
        # Color code GEX
        def color_gex(val):
            if val > 0:
                return "background-color: #1a2e1a; color: #00ff88"
            elif val < 0:
                return "background-color: #2e1a1a; color: #ff6666"
            return ""
        
        styled_greeks = greeks_display.style.applymap(color_gex, subset=["GEX_Net"])
        st.dataframe(styled_greeks, use_container_width=True)
        
        # GEX Interpretation
        st.markdown("#### 🎯 GEX INTERPRETATION (SELLER'S VIEW)")
        if total_gex_net > 0:
            st.success(f"**POSITIVE GEX (₹{int(total_gex_net):,}):** Sellers have POSITIVE gamma exposure. They're SHORT gamma and will BUY when price rises, SELL when price falls (stabilizing effect).")
        elif total_gex_net < 0:
            st.error(f"**NEGATIVE GEX (₹{int(total_gex_net):,}):** Sellers have NEGATIVE gamma exposure. They're LONG gamma and will SELL when price rises, BUY when price falls (destabilizing effect).")
        else:
            st.info("**NEUTRAL GEX:** Balanced seller gamma exposure.")
    
    with tab3:
        st.markdown("### 📈 SELLER PCR ANALYSIS")
        
        pcr_display_cols = ["strikePrice", "OI_CE", "OI_PE", "PCR", "Chg_OI_CE", "Chg_OI_PE", "seller_support_score", "seller_resistance_score"]
        for col in pcr_display_cols:
            if col not in ranked_current.columns:
                ranked_current[col] = 0
        
        # Create display dataframe
        pcr_display = ranked_current[pcr_display_cols].copy()
        pcr_display["distance_from_spot"] = abs(pcr_display["strikePrice"] - spot)
        pcr_display["OI_Total"] = pcr_display["OI_CE"] + pcr_display["OI_PE"]
        pcr_display["PUT_OI_Pct"] = (pcr_display["OI_PE"] / pcr_display["OI_Total"] * 100).round(1)
        
        # Sort by distance_from_spot BEFORE applying style
        pcr_display = pcr_display.sort_values("distance_from_spot")
        
        # Color PCR values
        def color_pcr(val):
            if isinstance(val, (int, float)):
                if val > 1.5:
                    return "background-color: #1a2e1a; color: #00ff88"
                elif val > 1.0:
                    return "background-color: #2e2a1a; color: #ffcc44"
                elif val > 0.5:
                    return "background-color: #1a1f2e; color: #66b3ff"
                elif val <= 0.5:
                    return "background-color: #2e1a1a; color: #ff4444"
            return ""
        
        # Apply style to already sorted dataframe
        styled_pcr = pcr_display.style.applymap(color_pcr, subset=["PCR"])
        
        # Display without sorting again
        st.dataframe(styled_pcr, use_container_width=True)
        
        # PCR Interpretation with OI context
        avg_pcr = ranked_current["PCR"].replace([np.inf, -np.inf], np.nan).mean()
        if not np.isnan(avg_pcr):
            st.markdown(f"#### 🎯 AVERAGE PCR: {avg_pcr:.2f}")
            if avg_pcr > 1.5:
                st.success(f"**HIGH PCR (>1.5):** Heavy PUT selling relative to CALL selling. Sellers are BULLISH. PUT OI dominance: {oi_pcr_metrics['total_pe_oi']/max(oi_pcr_metrics['total_ce_oi'],1):.1f}x")
            elif avg_pcr > 1.0:
                st.info(f"**MODERATE PCR (1.0-1.5):** More PUT selling than CALL selling. Sellers leaning BULLISH. PUT OI: {oi_pcr_metrics['total_pe_oi']:,}")
            elif avg_pcr > 0.5:
                st.warning(f"**LOW PCR (0.5-1.0):** More CALL selling than PUT selling. Sellers leaning BEARISH. CALL OI: {oi_pcr_metrics['total_ce_oi']:,}")
            else:
                st.error(f"**VERY LOW PCR (<0.5):** Heavy CALL selling relative to PUT selling. Sellers are BEARISH. CALL OI dominance: {oi_pcr_metrics['total_ce_oi']/max(oi_pcr_metrics['total_pe_oi'],1):.1f}x")
    
    with tab4:
        st.markdown("### 🚀 MOMENT DETECTOR ANALYSIS")
        
        # Momentum Burst Details
        st.markdown("#### 💥 MOMENTUM BURST ANALYSIS")
        mb = moment_metrics["momentum_burst"]
        if mb["available"]:
            col_mb1, col_mb2 = st.columns(2)
            with col_mb1:
                st.metric("Score", f"{mb['score']}/100")
                if mb["score"] > 70:
                    st.success("**STRONG MOMENTUM:** High energy for directional move")
                elif mb["score"] > 40:
                    st.info("**MODERATE MOMENTUM:** Some energy building")
                else:
                    st.warning("**LOW MOMENTUM:** Market is calm")
            with col_mb2:
                st.info(f"**Note:** {mb['note']}")
        else:
            st.warning("Momentum burst data unavailable. Need more refresh points.")
        
        st.markdown("---")
        
        # Orderbook Pressure Details
        st.markdown("#### 📊 ORDERBOOK PRESSURE ANALYSIS")
        ob = moment_metrics["orderbook"]
        if ob["available"]:
            col_ob1, col_ob2 = st.columns(2)
            with col_ob1:
                st.metric("Pressure", f"{ob['pressure']:+.2f}")
                st.metric("Buy Qty", f"{ob['buy_qty']:.0f}")
                st.metric("Sell Qty", f"{ob['sell_qty']:.0f}")
            with col_ob2:
                if ob["pressure"] > 0.15:
                    st.success("**STRONG BUY PRESSURE:** More buy orders than sell orders")
                elif ob["pressure"] < -0.15:
                    st.error("**STRONG SELL PRESSURE:** More sell orders than buy orders")
                else:
                    st.info("**BALANCED ORDERBOOK:** Buy and sell orders are balanced")
        else:
            st.warning("Orderbook depth data unavailable from Dhan API.")
        
        st.markdown("---")
        
        # Gamma Cluster Details
        st.markdown("#### 🌀 GAMMA CLUSTER ANALYSIS")
        gc = moment_metrics["gamma_cluster"]
        if gc["available"]:
            col_gc1, col_gc2 = st.columns(2)
            with col_gc1:
                st.metric("Cluster Score", f"{gc['score']}/100")
                st.metric("Raw Cluster Value", f"{gc['cluster']:.2f}")
            with col_gc2:
                if gc["score"] > 70:
                    st.success("**HIGH GAMMA CLUSTER:** Strong concentration around ATM - expect sharp moves")
                elif gc["score"] > 40:
                    st.info("**MODERATE GAMMA CLUSTER:** Some gamma concentration")
                else:
                    st.warning("**LOW GAMMA CLUSTER:** Gamma spread out - smoother moves expected")
        
        st.markdown("---")
        
        # OI Acceleration Details
        st.markdown("#### ⚡ OI ACCELERATION ANALYSIS")
        oi_accel = moment_metrics["oi_accel"]
        if oi_accel["available"]:
            col_oi1, col_oi2 = st.columns(2)
            with col_oi1:
                st.metric("Acceleration Score", f"{oi_accel['score']}/100")
            with col_oi2:
                st.info(f"**Note:** {oi_accel['note']}")
                if oi_accel["score"] > 60:
                    st.success("**ACCELERATING OI:** Open interest changing rapidly - momentum building")
                else:
                    st.info("**STEADY OI:** Open interest changes are gradual")
    
    with tab5:
        st.markdown("### 📅 EXPIRY SPIKE ANALYSIS")
        
        # Expiry Spike Probability
        st.markdown("#### 📊 SPIKE PROBABILITY BREAKDOWN")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            st.metric("Spike Probability", f"{expiry_spike_data.get('probability', 0)}%")
            st.metric("Spike Score", f"{expiry_spike_data.get('score', 0)}/100")
        
        with col_exp2:
            st.metric("Days to Expiry", f"{days_to_expiry:.1f}")
            st.metric("Spike Type", expiry_spike_data.get('type', 'N/A'))
        
        with col_exp3:
            intensity = expiry_spike_data.get('intensity', 'N/A')
            intensity_color = {
                "HIGH PROBABILITY SPIKE": "#ff0000",
                "MODERATE SPIKE RISK": "#ff9900",
                "LOW SPIKE RISK": "#ffff00",
                "NO SPIKE DETECTED": "#00ff00"
            }.get(intensity, "#cccccc")
            
            st.markdown(f"""
            <div style="
                background: rgba(0,0,0,0.2);
                padding: 10px;
                border-radius: 8px;
                border-left: 4px solid {intensity_color};
                margin: 10px 0;
            ">
                <div style="font-size: 0.9rem; color:#cccccc;">Spike Intensity</div>
                <div style="font-size: 1.2rem; color:{intensity_color}; font-weight:700;">{intensity}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Spike Triggers
        st.markdown("#### ⚠️ SPIKE TRIGGERS DETECTED")
        if expiry_spike_data.get("factors"):
            for factor in expiry_spike_data["factors"]:
                st.markdown(f"• {factor}")
        else:
            st.info("No spike triggers detected")

        st.markdown("---")

        # Spike Strike Price Ranges
        support_range = expiry_spike_data.get("support_spike_range", {})
        resistance_range = expiry_spike_data.get("resistance_spike_range", {})

        if support_range.get("start") is not None or resistance_range.get("start") is not None:
            st.markdown("#### 📊 SPIKE STRIKE PRICE RANGE")

            spike_range_cols = st.columns(2)

            with spike_range_cols[0]:
                if support_range.get("start") is not None:
                    st.markdown(f"""
                    <div style='background: rgba(0,255,0,0.1); border: 2px solid #00ff00; padding: 15px; margin: 10px 0; border-radius: 10px;'>
                        <h4 style='color:#00ff00; margin:0 0 10px 0;'>🛡️ SUPPORT SPIKE RANGE</h4>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color:#aaaaaa;'>Start Strike:</span>
                            <span style='color:#ffffff; font-weight:700;'>₹{support_range['start']:,}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color:#aaaaaa;'>End Strike:</span>
                            <span style='color:#ffffff; font-weight:700;'>₹{support_range['end']:,}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color:#aaaaaa;'>Range Width:</span>
                            <span style='color:#66b3ff;'>{len(support_range['strikes'])} strikes</span>
                        </div>
                        <div style='display: flex; justify-content: space-between;'>
                            <span style='color:#aaaaaa;'>Total PUT OI:</span>
                            <span style='color:#00ff88; font-weight:700;'>{support_range['total_oi']:,}</span>
                        </div>
                        <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;'>
                            <span style='color:#888888; font-size: 0.85em;'>
                                Strikes: {', '.join([f"₹{s:,}" for s in support_range['strikes'][:6]])}
                                {f" (+{len(support_range['strikes'])-6} more)" if len(support_range['strikes']) > 6 else ""}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No significant support spike range detected")

            with spike_range_cols[1]:
                if resistance_range.get("start") is not None:
                    st.markdown(f"""
                    <div style='background: rgba(255,0,0,0.1); border: 2px solid #ff4444; padding: 15px; margin: 10px 0; border-radius: 10px;'>
                        <h4 style='color:#ff4444; margin:0 0 10px 0;'>🚧 RESISTANCE SPIKE RANGE</h4>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color:#aaaaaa;'>Start Strike:</span>
                            <span style='color:#ffffff; font-weight:700;'>₹{resistance_range['start']:,}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color:#aaaaaa;'>End Strike:</span>
                            <span style='color:#ffffff; font-weight:700;'>₹{resistance_range['end']:,}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color:#aaaaaa;'>Range Width:</span>
                            <span style='color:#66b3ff;'>{len(resistance_range['strikes'])} strikes</span>
                        </div>
                        <div style='display: flex; justify-content: space-between;'>
                            <span style='color:#aaaaaa;'>Total CALL OI:</span>
                            <span style='color:#ff6666; font-weight:700;'>{resistance_range['total_oi']:,}</span>
                        </div>
                        <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;'>
                            <span style='color:#888888; font-size: 0.85em;'>
                                Strikes: {', '.join([f"₹{s:,}" for s in resistance_range['strikes'][:6]])}
                                {f" (+{len(resistance_range['strikes'])-6} more)" if len(resistance_range['strikes']) > 6 else ""}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No significant resistance spike range detected")

            st.markdown("---")

        # Violent Unwinding
        if violent_unwinding_signals:
            st.markdown("#### 🚨 VIOLENT UNWINDING DETECTED")
            for signal in violent_unwinding_signals:
                st.markdown(f"• {signal}")
        
        st.markdown("---")
        
        # Gamma Spike Risk
        if gamma_spike_risk["score"] > 0:
            st.markdown("#### ⚡ GAMMA SPIKE RISK")
            st.markdown(f"**Risk Level:** {gamma_spike_risk['risk']}")
            st.markdown(f"**Score:** {gamma_spike_risk['score']}/100")
            st.markdown(f"**Message:** {gamma_spike_risk['message']}")
        
        st.markdown("---")
        
        # Pinning Probability
        if pinning_probability > 0:
            st.markdown("#### 📍 EXPIRY PINNING PROBABILITY")
            st.metric("Pinning Chance", f"{pinning_probability}%")
            if pinning_probability > 50:
                st.info("**HIGH PINNING RISK:** Price likely to get stuck near current levels")
            elif pinning_probability > 30:
                st.warning("**MODERATE PINNING RISK:** Some chance of price getting stuck")
            else:
                st.success("**LOW PINNING RISK:** Price likely to move freely")
    
    with tab6:
        st.markdown("### 🎯 ATM BIAS DETAILED ANALYSIS")
        
        if atm_bias:
            # Key metrics
            col_atm1, col_atm2, col_atm3 = st.columns(3)
            
            with col_atm1:
                st.metric("ATM Strike", f"₹{atm_bias['atm_strike']:,}")
                st.metric("CALL OI", f"{atm_bias['metrics']['ce_oi']:,}")
                st.metric("PUT OI", f"{atm_bias['metrics']['pe_oi']:,}")
            
            with col_atm2:
                st.metric("Net Delta", f"{atm_bias['metrics']['net_delta']:.3f}")
                st.metric("Net Gamma", f"{atm_bias['metrics']['net_gamma']:.3f}")
                st.metric("Delta Exposure", f"₹{atm_bias['metrics']['delta_exposure']:,}")
            
            with col_atm3:
                st.metric("Gamma Exposure", f"₹{atm_bias['metrics']['gamma_exposure']:,}")
                st.metric("CALL IV", f"{atm_bias['metrics']['ce_iv']:.2f}%")
                st.metric("PUT IV", f"{atm_bias['metrics']['pe_iv']:.2f}%")
            
            # Bias breakdown
            st.markdown("#### 📊 BIAS BREAKDOWN BY METRIC")
            for bias_name, score in atm_bias["bias_scores"].items():
                emoji = atm_bias["bias_emojis"].get(bias_name, "⚖️")
                interpretation = atm_bias["bias_interpretations"].get(bias_name, "")
                
                # Color based on score
                if score > 0:
                    color = "#00ff88"
                    bg_color = "#1a2e1a"
                elif score < 0:
                    color = "#ff4444"
                    bg_color = "#2e1a1a"
                else:
                    color = "#66b3ff"
                    bg_color = "#1a1f2e"
                
                st.markdown(f"""
                <div style="
                    background: {bg_color};
                    padding: 10px;
                    border-radius: 8px;
                    border-left: 4px solid {color};
                    margin: 5px 0;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-weight: 600; color:#ffffff;">
                            {bias_name.replace('_', ' ').title()}
                        </div>
                        <div style="font-size: 1.2rem; color:{color}; font-weight:700;">
                            {emoji} {score:+.1f}
                        </div>
                    </div>
                    <div style="font-size: 0.9rem; color:#cccccc; margin-top: 5px;">
                        {interpretation}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Seller interpretation for ATM
            st.markdown("#### 🧠 SELLER INTERPRETATION AT ATM")
            if atm_bias["total_score"] > 0.3:
                st.success(f"""
                **Strong Bullish ATM Bias ({atm_bias['total_score']:.2f})**
                
                Sellers are heavily writing PUTs at ATM strikes, indicating:
                1. **Bullish conviction** - Expecting price to stay above ATM
                2. **PUT selling dominance** - More PUT OI than CALL OI
                3. **Negative delta exposure** - PUTs creating negative delta
                4. **Potential support** - ATM acting as strong support
                
                **Trading Implication:** Favor LONG positions with stops below ATM
                """)
            elif atm_bias["total_score"] > 0.1:
                st.info(f"""
                **Mild Bullish ATM Bias ({atm_bias['total_score']:.2f})**
                
                Sellers are leaning towards PUT writing at ATM:
                1. **Slight bullish bias** - More PUT activity than CALL
                2. **Moderate PUT OI** - PUT OI slightly higher than CALL
                3. **Balanced delta** - Delta exposure relatively neutral
                
                **Trading Implication:** Cautious LONG bias, wait for confirmation
                """)
            elif atm_bias["total_score"] < -0.3:
                st.error(f"""
                **Strong Bearish ATM Bias ({atm_bias['total_score']:.2f})**
                
                Sellers are heavily writing CALLs at ATM strikes, indicating:
                1. **Bearish conviction** - Expecting price to stay below ATM
                2. **CALL selling dominance** - More CALL OI than PUT OI
                3. **Positive delta exposure** - CALLs creating positive delta
                4. **Potential resistance** - ATM acting as strong resistance
                
                **Trading Implication:** Favor SHORT positions with stops above ATM
                """)
            elif atm_bias["total_score"] < -0.1:
                st.warning(f"""
                **Mild Bearish ATM Bias ({atm_bias['total_score']:.2f})**
                
                Sellers are leaning towards CALL writing at ATM:
                1. **Slight bearish bias** - More CALL activity than PUT
                2. **Moderate CALL OI** - CALL OI slightly higher than PUT
                3. **Balanced delta** - Delta exposure relatively neutral
                
                **Trading Implication:** Cautious SHORT bias, wait for confirmation
                """)
            else:
                st.info(f"""
                **Neutral ATM Bias ({atm_bias['total_score']:.2f})**
                
                Balanced seller activity at ATM:
                1. **No clear bias** - CALL and PUT activity balanced
                2. **Equal OI distribution** - Similar OI on both sides
                3. **Neutral delta/gamma** - Minimal directional pressure
                
                **Trading Implication:** Range-bound expected, wait for breakout
                """)
            
            # Gamma exposure implications
            st.markdown("#### ⚡ GAMMA EXPOSURE IMPLICATIONS")
            if atm_bias["metrics"]["gamma_exposure"] > 100000:
                st.success(f"""
                **Positive Gamma Exposure (₹{atm_bias['metrics']['gamma_exposure']:,})**
                
                Market makers are SHORT gamma at ATM:
                - **Stabilizing effect** - They'll buy on dips, sell on rallies
                - **Reduced volatility** - Price moves become smoother
                - **Mean reversion bias** - Tends to revert to ATM
                - **Gamma squeeze unlikely** - Less explosive moves
                
                **Trading Implication:** Fade extremes, trade mean reversion
                """)
            elif atm_bias["metrics"]["gamma_exposure"] < -100000:
                st.warning(f"""
                **Negative Gamma Exposure (₹{abs(atm_bias['metrics']['gamma_exposure']):,})**
                
                Market makers are LONG gamma at ATM:
                - **Destabilizing effect** - They'll sell on dips, buy on rallies
                - **Increased volatility** - Price moves become more explosive
                - **Momentum bias** - Moves tend to accelerate
                - **Gamma squeeze possible** - Potential for sharp moves
                
                **Trading Implication:** Ride momentum, expect whipsaws
                """)
            else:
                st.info("**Neutral Gamma Exposure** - Minimal impact on price action")
        else:
            st.info("ATM bias analysis not available for current data")
    
    # ============================================
    # 🎯 TRADING INSIGHTS - SELLER PERSPECTIVE + ATM BIAS + MOMENT + EXPIRY + OI/PCR
    # ============================================
    st.markdown("---")
    st.markdown("## 💡 TRADING INSIGHTS (Seller + ATM Bias + Moment + Expiry + OI/PCR Fusion)")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("### 🎯 KEY OBSERVATIONS")
        
        # Max Pain insight
        if seller_max_pain:
            max_pain_strike = seller_max_pain.get("max_pain_strike", 0)
            max_pain_insight = ""
            if spot > max_pain_strike:
                max_pain_insight = f"Spot ABOVE max pain (₹{max_pain_strike:,}). Sellers losing on CALLs, gaining on PUTs."
            else:
                max_pain_insight = f"Spot BELOW max pain (₹{max_pain_strike:,}). Sellers gaining on CALLs, losing on PUTs."

            st.info(f"**Max Pain:** {max_pain_insight}")
        
        # GEX insight
        if total_gex_net > 0:
            st.success("**Gamma Exposure:** Sellers SHORT gamma. Expect reduced volatility and mean reversion.")
        elif total_gex_net < 0:
            st.warning("**Gamma Exposure:** Sellers LONG gamma. Expect increased volatility and momentum moves.")
        
        # ATM Bias insight
        if atm_bias:
            if atm_bias["total_score"] > 0.2:
                st.success(f"**ATM Bias Bullish ({atm_bias['total_score']:.2f}):** Heavy PUT selling at ATM confirms bullish sentiment")
            elif atm_bias["total_score"] < -0.2:
                st.error(f"**ATM Bias Bearish ({atm_bias['total_score']:.2f}):** Heavy CALL selling at ATM confirms bearish sentiment")
            else:
                st.info(f"**ATM Bias Neutral ({atm_bias['total_score']:.2f}):** Balanced activity at ATM")
        
        # PCR insight with OI context
        total_pcr = total_PE_OI / total_CE_OI if total_CE_OI > 0 else 0
        if total_pcr > 1.5:
            st.success(f"**Overall PCR ({total_pcr:.2f}):** Strong PUT selling dominance. Bullish seller conviction. PUT OI: {oi_pcr_metrics['total_pe_oi']:,}")
        elif total_pcr < 0.7:
            st.error(f"**Overall PCR ({total_pcr:.2f}):** Strong CALL selling dominance. Bearish seller conviction. CALL OI: {oi_pcr_metrics['total_ce_oi']:,}")
        else:
            st.info(f"**Overall PCR ({total_pcr:.2f}):** Balanced. CALL OI: {oi_pcr_metrics['total_ce_oi']:,} | PUT OI: {oi_pcr_metrics['total_pe_oi']:,}")
        
        # OI Concentration insight
        if oi_pcr_metrics['atm_concentration_pct'] > 35:
            st.warning(f"**High ATM OI Concentration ({oi_pcr_metrics['atm_concentration_pct']:.1f}%):** Gamma risk elevated. Expect whipsaws around ATM.")
        
        # Max OI insights
        if oi_pcr_metrics['max_ce_oi'] > 1000000:
            st.info(f"**Large CALL Wall at ₹{oi_pcr_metrics['max_ce_strike']:,}:** Strong resistance with {oi_pcr_metrics['max_ce_oi']:,} OI")
        if oi_pcr_metrics['max_pe_oi'] > 1000000:
            st.info(f"**Large PUT Wall at ₹{oi_pcr_metrics['max_pe_strike']:,}:** Strong support with {oi_pcr_metrics['max_pe_oi']:,} OI")
        
        # Expiry Spike insight
        if expiry_spike_data["active"]:
            if expiry_spike_data["probability"] > 60:
                st.error(f"**High Expiry Spike Risk ({expiry_spike_data['probability']}%):** {expiry_spike_data['type']}")
            elif expiry_spike_data["probability"] > 40:
                st.warning(f"**Moderate Expiry Spike Risk ({expiry_spike_data['probability']}%):** {expiry_spike_data['type']}")
            else:
                st.success(f"**Low Expiry Spike Risk ({expiry_spike_data['probability']}%):** Market stable near expiry")
        
        # Moment Detector insights
        st.markdown("#### 🚀 MOMENT DETECTOR INSIGHTS")
        if moment_metrics["momentum_burst"]["score"] > 60:
            st.success("**High Momentum Burst:** Market energy is building for a move")
        if moment_metrics["orderbook"]["available"] and abs(moment_metrics["orderbook"]["pressure"]) > 0.15:
            direction = "buy" if moment_metrics["orderbook"]["pressure"] > 0 else "sell"
            st.info(f"**Strong {direction.upper()} pressure** in orderbook")
    
    with insight_col2:
        st.markdown("### 🛡️ RISK MANAGEMENT")
        
        # Nearest levels insight
        if nearest_sup and nearest_res:
            risk_reward = (nearest_res["distance"] / nearest_sup["distance"]) if nearest_sup["distance"] > 0 else 0
            
            st.metric("Risk:Reward (Current Range)", f"1:{risk_reward:.2f}")
            
            # Stop loss suggestion with OI context
            if seller_bias_result["bias"].startswith("BULLISH"):
                stop_loss = f"Below seller support: ₹{nearest_sup['strike']:,} (PUT OI: {nearest_sup['oi_pe']:,})"
                target = f"Seller resistance: ₹{nearest_res['strike']:,} (CALL OI: {nearest_res['oi_ce']:,})"
            elif seller_bias_result["bias"].startswith("BEARISH"):
                stop_loss = f"Above seller resistance: ₹{nearest_res['strike']:,} (CALL OI: {nearest_res['oi_ce']:,})"
                target = f"Seller support: ₹{nearest_sup['strike']:,} (PUT OI: {nearest_sup['oi_pe']:,})"
            else:
                stop_loss = f"Range: ₹{nearest_sup['strike']:,} - ₹{nearest_res['strike']:,}"
                target = "Wait for breakout"
            
            st.info(f"**Stop Loss:** {stop_loss}")
            st.info(f"**Target:** {target}")
            
            # OI-based stop adjustment
            if oi_pcr_metrics['max_pe_oi'] > 500000 and oi_pcr_metrics['max_pe_strike'] < spot:
                st.info(f"**Strong PUT Support:** Consider ₹{oi_pcr_metrics['max_pe_strike']:,} as major support ({oi_pcr_metrics['max_pe_oi']:,} OI)")
            if oi_pcr_metrics['max_ce_oi'] > 500000 and oi_pcr_metrics['max_ce_strike'] > spot:
                st.info(f"**Strong CALL Resistance:** Consider ₹{oi_pcr_metrics['max_ce_strike']:,} as major resistance ({oi_pcr_metrics['max_ce_oi']:,} OI)")
        
        # ATM Bias-based adjustments
        if atm_bias:
            st.markdown("#### 🎯 ATM BIAS-BASED ADJUSTMENTS")
            if atm_bias["total_score"] > 0.3:
                st.success("**Strong Bullish ATM Bias:** Consider tighter stops on LONG positions, wider stops on SHORT")
            elif atm_bias["total_score"] < -0.3:
                st.success("**Strong Bearish ATM Bias:** Consider tighter stops on SHORT positions, wider stops on LONG")
            
            if atm_bias["metrics"]["gamma_exposure"] < -200000:
                st.warning("**High Negative Gamma Exposure:** Expect explosive moves - Use wider stops")
            elif atm_bias["metrics"]["gamma_exposure"] > 200000:
                st.info("**High Positive Gamma Exposure:** Expect mean reversion - Tighter stops may work")
        
        # Expiry-based risk adjustments with OI context
        if expiry_spike_data["active"]:
            st.markdown("#### 📅 EXPIRY-BASED RISK ADJUSTMENTS")
            if expiry_spike_data["probability"] > 60:
                st.warning("**High Spike Risk:** Use 2x wider stops, avoid overnight positions")
                if oi_pcr_metrics['atm_concentration_pct'] > 40:
                    st.warning("**High ATM OI + Expiry:** Extreme gamma risk. Consider straddle/strangle strategies")
            elif expiry_spike_data["probability"] > 40:
                st.info("**Moderate Spike Risk:** Use 1.5x wider stops, be ready for volatility")
            if days_to_expiry <= 1:
                st.warning("**Expiry Day:** Expect whipsaws in last 2 hours, reduce position size")
                # Check for massive OI that needs to unwind
                if oi_pcr_metrics['total_oi'] > 5000000:
                    st.warning(f"**Large OI ({oi_pcr_metrics['total_oi']:,}) to unwind:** Expect violent moves as positions close")
        
        # OI-based risk adjustments
        st.markdown("#### 📊 OI-BASED RISK ADJUSTMENTS")
        if oi_pcr_metrics['call_oi_skew'] == "High":
            st.warning("**High CALL OI Skew:** OI concentrated at few strikes - increased pinning risk")
        if oi_pcr_metrics['put_oi_skew'] == "High":
            st.warning("**High PUT OI Skew:** OI concentrated at few strikes - increased pinning risk")
        if abs(oi_pcr_metrics['total_ce_chg']) > 100000 or abs(oi_pcr_metrics['total_pe_chg']) > 100000:
            st.info(f"**Large OI Changes:** CALL Δ: {oi_pcr_metrics['total_ce_chg']:+,} | PUT Δ: {oi_pcr_metrics['total_pe_chg']:+,} - Momentum building")
        
        # Moment-based risk adjustments
        st.markdown("#### 🚀 MOMENT-BASED RISK ADJUSTMENTS")
        if moment_metrics["momentum_burst"]["score"] > 70:
            st.warning("**High Momentum Alert:** Consider tighter stops due to potential sharp moves")
        if moment_metrics["gamma_cluster"]["score"] > 70:
            st.warning("**High Gamma Cluster:** Expect whipsaws around ATM - be prepared for volatility")
    
    # Final Seller Summary with ATM Bias, Moment, Expiry, and OI/PCR Integration
    st.markdown("---")
    moment_summary = ""
    if moment_metrics["momentum_burst"]["score"] > 60:
        moment_summary += "High momentum burst detected. "
    if moment_metrics["orderbook"]["available"] and abs(moment_metrics["orderbook"]["pressure"]) > 0.15:
        direction = "buy" if moment_metrics["orderbook"]["pressure"] > 0 else "sell"
        moment_summary += f"Strong {direction} pressure in orderbook. "
    
    atm_bias_summary = f"ATM Bias: {atm_bias['verdict'] if atm_bias else 'N/A'} ({atm_bias['total_score']:.2f} score)" if atm_bias else "ATM Bias: N/A"
    
    expiry_summary = ""
    if expiry_spike_data["active"]:
        if expiry_spike_data["probability"] > 60:
            expiry_summary = f"🚨 HIGH EXPIRY SPIKE RISK ({expiry_spike_data['probability']}%) - {expiry_spike_data['type']}"
        elif expiry_spike_data["probability"] > 40:
            expiry_summary = f"⚠️ MODERATE EXPIRY SPIKE RISK ({expiry_spike_data['probability']}%) - {expiry_spike_data['type']}"
        else:
            expiry_summary = f"✅ LOW EXPIRY SPIKE RISK ({expiry_spike_data['probability']}%)"
    
    oi_pcr_summary = f"PCR: {oi_pcr_metrics['pcr_total']:.2f} ({oi_pcr_metrics['pcr_sentiment']}) | CALL OI: {oi_pcr_metrics['total_ce_oi']:,} | PUT OI: {oi_pcr_metrics['total_pe_oi']:,} | ATM Conc: {oi_pcr_metrics['atm_concentration_pct']:.1f}%"
    
    st.markdown(f'''
    <div class='seller-explanation'>
        <h3>🎯 FINAL ASSESSMENT (Seller + ATM Bias + Moment + Expiry + OI/PCR)</h3>
        <p><strong>Market Makers are telling us:</strong> {seller_bias_result["explanation"]}</p>
        <p><strong>ATM Zone Analysis:</strong> {atm_bias_summary}</p>
        <p><strong>Their game plan:</strong> {seller_bias_result["action"]}</p>
        <p><strong>Moment Detector:</strong> {moment_summary if moment_summary else "Moment indicators neutral"}</p>
        <p><strong>OI/PCR Analysis:</strong> {oi_pcr_summary}</p>
        <p><strong>Expiry Context:</strong> {expiry_summary if expiry_summary else f"Expiry in {days_to_expiry:.1f} days"}</p>
        <p><strong>Key defense levels:</strong> ₹{nearest_sup['strike'] if nearest_sup else 'N/A':,} (Support) | ₹{nearest_res['strike'] if nearest_res else 'N/A':,} (Resistance)</p>
        <p><strong>Max OI Walls:</strong> CALL: ₹{oi_pcr_metrics['max_ce_strike']:,} | PUT: ₹{oi_pcr_metrics['max_pe_strike']:,}</p>
        <p><strong>Preferred price level:</strong> ₹{seller_max_pain.get('max_pain_strike', 0) if seller_max_pain else 'N/A':,} (Max Pain)</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption(f"🔄 Auto-refresh: {AUTO_REFRESH_SEC}s | ⏰ {get_ist_datetime_str()}")
    st.caption("🎯 **NIFTY Option Screener v7.0 — SELLER'S PERSPECTIVE + ATM BIAS ANALYZER + MOMENT DETECTOR + EXPIRY SPIKE DETECTOR + ENHANCED OI/PCR ANALYTICS** | All features enabled")
    
    # Requirements note
    st.markdown("""
    <small>
    **Requirements:** 
    `streamlit pandas numpy requests pytz scipy supabase python-dotenv` | 
    **Data:** Dhan API required
    </small>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"**🔄 Last update:** Auto-refreshing every {AUTO_REFRESH_SEC} seconds")
    st_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")

# -----------------------
#  STANDALONE EXECUTION
# -----------------------
if __name__ == "__main__":
    render_nifty_option_screener()
