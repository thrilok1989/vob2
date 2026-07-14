import streamlit as st
import pandas as pd
import numpy as np
import math
import json
import hashlib
import pytz
import datetime
from datetime import datetime, timedelta
from pytz import timezone
import plotly.graph_objects as go
from vob_indicators import *
from vob_data import *
from indicators.money_flow_profile import calculate_money_flow_profile
from indicators.volume_delta import calculate_volume_delta
from db.supabase_client import SupabaseDB

def check_trading_signals(df, pivot_settings, option_data, current_price, pivot_proximity=5):
    if df.empty or option_data is None or len(option_data) == 0 or not current_price:
        return

    try:
        df_json = df.to_json()
        pivots = cached_pivot_calculation(df_json, pivot_settings)
    except:
        pivots = PivotIndicator.get_all_pivots(df, pivot_settings)

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
            ce_chg_oi = row.get('changeinOpenInterest_CE', 0)
            pe_chg_oi = row.get('changeinOpenInterest_PE', 0)
            bullish_oi_confirm = pe_chg_oi > 1.5 * ce_chg_oi
            bearish_oi_confirm = ce_chg_oi > 1.5 * pe_chg_oi
            reversal_text = f"""
🔄 <b>REVERSAL DETECTOR:</b>
• Score: {reversal_signals.get('Reversal_Score', 0)}/6
• Selling Exhausted: {reversal_signals.get('Selling_Exhausted', 'N/A')}
• Higher Low: {reversal_signals.get('Higher_Low', 'N/A')}
• Strong Candle: {reversal_signals.get('Strong_Bullish_Candle', 'N/A')}
• Volume: {reversal_signals.get('Volume_Signal', 'N/A')}
• Above VWAP: {reversal_signals.get('Above_VWAP', 'N/A')}
• {reversal_verdict}"""

            price_diff = current_price - pivot_level['value']
            near = abs(price_diff) <= pivot_proximity
            time_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')
            pv_tf, pv_val = pivot_level['timeframe'], pivot_level['value']
            for is_bull, conds, oi_confirm in [
                (True, bullish_conditions, bullish_oi_confirm),
                (False, bearish_conditions, bearish_oi_confirm)
            ]:
                if not (all(conds.values()) and near):
                    continue
                emoji = "🚨" if is_bull else "🔴"
                opt = "CE" if is_bull else "PE"
                direction = "CALL" if is_bull else "PUT"
                mark = "✅" if is_bull else "🔴"
                trigger_type = "🔥 OI Dominance Trigger" if oi_confirm else "📊 Normal Bias Trigger"
                conds_text = "\n".join([f"{mark} {k}" for k, v in conds.items() if v])
                oi_text = f"PE ChgOI {pe_chg_oi:,} vs CE ChgOI {ce_chg_oi:,}" if is_bull else f"CE ChgOI {ce_chg_oi:,} vs PE ChgOI {pe_chg_oi:,}"
                message = f"""{emoji} <b>NIFTY {direction} SIGNAL ALERT</b> {emoji}
📍 <b>Spot:</b> ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} Pivot by {price_diff:+.2f}pts)
📌 <b>Near Pivot:</b> {pv_tf} at ₹{pv_val:.2f}
🎯 <b>ATM Strike:</b> {atm_strike}
<b>{mark} {'BULLISH' if is_bull else 'BEARISH'} CONDITIONS MET:</b>
{conds_text}
⚡ <b>{trigger_type}</b>
⚡ <b>OI:</b> {oi_text}
{reversal_text}
📋 <b>REVIEW:</b> {atm_strike} {opt} | SL: {stop_loss_percent}% | Manual verification required
🕐 Time: {time_str}"""
                # Telegram disabled for reversal signals (noise reduction)
                break

def check_atm_verdict_alert(df_summary, underlying_price):
    """ATM verdict alert — display only, no Telegram."""
    if df_summary is None or len(df_summary) == 0 or not underlying_price:
        return

    atm_data = df_summary[df_summary['Zone'] == 'ATM']
    if atm_data.empty:
        return

    row = atm_data.iloc[0]
    verdict = row.get('Verdict', 'Neutral')
    atm_strike = row.get('Strike', 0)
    bias_score = row.get('BiasScore', 0)

    if verdict not in ['Strong Bullish', 'Strong Bearish']:
        return

    alert_key = f"atm_verdict_{atm_strike}_{verdict}"
    if 'last_atm_verdict_alert' not in st.session_state:
        st.session_state.last_atm_verdict_alert = None

    if st.session_state.last_atm_verdict_alert == alert_key:
        return

    _g = row.get
    ce_oi, pe_oi = _g('openInterest_CE', 0), _g('openInterest_PE', 0)
    ce_chg_oi, pe_chg_oi = _g('changeinOpenInterest_CE', 0), _g('changeinOpenInterest_PE', 0)
    is_bull = verdict == 'Strong Bullish'
    emoji = "🟢🟢🟢" if is_bull else "🔴🔴🔴"
    direction = "BULLISH" if is_bull else "BEARISH"
    opt = "CE" if is_bull else "PE"
    message = f"""{emoji} <b>ATM STRIKE STRONG {direction} ALERT</b> {emoji}
📍 <b>Spot Price:</b> ₹{underlying_price:.2f}
🎯 <b>ATM Strike:</b> {atm_strike}
📊 <b>Verdict:</b> {verdict} (Score: {bias_score})
<b>📈 BIAS BREAKDOWN:</b>
• OI: {_g('OI_Bias','N/A')} • ChgOI: {_g('ChgOI_Bias','N/A')} • Volume: {_g('Volume_Bias','N/A')}
• Delta Exp: {_g('DeltaExp','N/A')} • Gamma Exp: {_g('GammaExp','N/A')} • Pressure: {_g('PressureBias','N/A')}
<b>📊 OI DATA:</b>
• CE OI: {ce_oi/100000:.1f}L | PE OI: {pe_oi/100000:.1f}L
• CE ΔOI: {ce_chg_oi/1000:.1f}K | PE ΔOI: {pe_chg_oi/1000:.1f}K
<b>⚡ SIGNALS:</b>
• Operator Entry: {_g('Operator_Entry','N/A')} • Scalp/Momentum: {_g('Scalp_Moment','N/A')}
📋 <b>SUGGESTED REVIEW:</b> {atm_strike} {opt} | Manual verification required
🕐 Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}"""
    st.session_state.last_atm_verdict_alert = alert_key

def calculate_dealer_gex(df_summary, spot_price, contract_multiplier=25):

def color_pressure(val):
    return _CG if val > 500 else (_CR if val < -500 else _CY)

def color_pcr(val):
    return _CG if val > 1.2 else (_CR if val < 0.7 else _CY)

def color_bias(val):
    return _CG if val == "Bullish" else (_CR if val == "Bearish" else _CY)

def color_verdict(val):
    v = str(val)
    if "Strong Bullish" in v: return _CDG
    if "Bullish" in v: return _CG
    if "Strong Bearish" in v: return _CDR
    if "Bearish" in v: return _CR
    return _CY

def color_entry(val):
    v = str(val)
    return _CG if "Bull" in v else (_CR if "Bear" in v else _CF)

def color_fakereal(val):
    v = str(val)
    if "Real Up" in v: return _CDG
    if "Fake Up" in v: return 'background-color: #98FB98; color: black'
    if "Real Down" in v: return _CDR
    if "Fake Down" in v: return 'background-color: #FFC0CB; color: black'
    return _CF

def color_score(val):
    try:
        s = float(val)
        if s >= 4: return _CDG
        if s >= 2: return _CG
        if s <= -4: return _CDR
        if s <= -2: return _CR
        return _CY
    except: return ''

def highlight_atm_row(row):
    return [''] * len(row)

def process_candle_data(data, interval):

def display_metrics(ltp_data, df, db, symbol="NIFTY50"):
    if not (ltp_data and 'data' in ltp_data and not df.empty):
        return
    current_price = None
    for exchange, data in ltp_data['data'].items():
        for sid, pd_ in data.items():
            current_price = pd_.get('last_price', 0)
            break
    if current_price is None:
        current_price = df['close'].iloc[-1]
    if df.empty or len(df) < 2:
        return
    prev_close = df['close'].iloc[-2]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100
    day_high, day_low, day_open = df['high'].max(), df['low'].min(), df['open'].iloc[0]
    volume = df['volume'].sum()
    avg_price = df['close'].mean()
    db.save_market_analytics(symbol, {
        'day_high': float(day_high), 'day_low': float(day_low), 'day_open': float(day_open),
        'day_close': float(current_price), 'total_volume': int(volume),
        'avg_price': float(avg_price), 'price_change': float(change), 'price_change_pct': float(change_pct)
    })
    sign = "+" if change >= 0 else ""
    color = "price-up" if change >= 0 else "price-down"
    metrics = [
        ("Current Price", f'<h2 class="{color}">₹{current_price:,.2f}</h2>'),
        ("Change", f'<h3 class="{color}">{sign}{change:.2f} ({sign}{change_pct:.2f}%)</h3>'),
        ("Day High", f'<h3>₹{day_high:,.2f}</h3>'),
        ("Day Low", f'<h3>₹{day_low:,.2f}</h3>'),
        ("Volume", f'<h3>{volume:,}</h3>'),
    ]
    for col, (label, val) in zip(st.columns(5), metrics):
        col.markdown(f'<div class="metric-container"><h4>{label}</h4>{val}</div>', unsafe_allow_html=True)


def create_csv_download(df_summary):
    output = io.StringIO()
    df_summary.to_csv(output, index=False)
    return output.getvalue()


def calculate_pcr_sr_level(pcr, reference_strike):
    """PCR-based Support/Resistance level.
    PCR ≤ 0.7 → Resistance offset below/above strike.
    0.8–1.7   → Neutral.
    PCR ≥ 1.8 → Support offset below/above strike.
    """
    pcr = round(pcr, 2)
    if pcr <= 0.7:
        # Map PCR → offset using linear interpolation between anchor points
        anchors = [(0.3, -20), (0.4, -10), (0.5, 0), (0.6, 10), (0.7, 20)]
        if pcr <= 0.3:
            offset = -20
        elif pcr >= 0.7:
            offset = 20
        else:
            offset = -20
            for i in range(len(anchors) - 1):
                p0, o0 = anchors[i]; p1, o1 = anchors[i + 1]
                if p0 <= pcr <= p1:
                    offset = o0 + (pcr - p0) / (p1 - p0) * (o1 - o0)
                    break
        level = reference_strike + offset
        interpretation = (
            f"Resistance at ₹{level:.0f} (below ATM — strong cap)"  if offset < 0 else
            f"Resistance at ₹{level:.0f} (above ATM — moderate cap)" if offset > 0 else
            f"Resistance at ATM ₹{level:.0f}"
        )
        return {'type': 'Resistance 🔴', 'level': level, 'offset': offset, 'interpretation': interpretation}

    elif pcr <= 1.7:  # 0.71–1.7 all neutral
        return {'type': 'Neutral ⚪', 'level': reference_strike, 'offset': 0,
                'interpretation': f"Neutral — no clear S/R offset (PCR {pcr:.2f})"}

    else:  # pcr >= 1.8
        anchors = [(1.8, -20), (2.0, -10), (2.5, 0), (3.0, 10), (3.5, 20)]
        if pcr <= 1.8:
            offset = -20
        elif pcr >= 3.5:
            offset = 20
        else:
            offset = -20
            for i in range(len(anchors) - 1):
                p0, o0 = anchors[i]; p1, o1 = anchors[i + 1]
                if p0 <= pcr <= p1:
                    offset = o0 + (pcr - p0) / (p1 - p0) * (o1 - o0)
                    break
        level = reference_strike + offset
        interpretation = (
            f"Support at ₹{level:.0f} (below ATM — strong floor)" if offset < 0 else
            f"Support at ₹{level:.0f} (above ATM — elevated floor)" if offset > 0 else
            f"Support at ATM ₹{level:.0f}"
        )
        return {'type': 'Support 🟢', 'level': level, 'offset': offset, 'interpretation': interpretation}



def analyze_option_chain(selected_expiry=None, pivot_data=None, vob_data=None):
    now = datetime.now(timezone("Asia/Kolkata"))

    expiry_data = get_dhan_expiry_list_cached(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
    if not expiry_data or 'data' not in expiry_data:
        st.error("Failed to get expiry list from Dhan API")
        return None

    expiry_dates = expiry_data['data']
    if not expiry_dates:
        st.error("No expiry dates available")
        return None

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
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice', ascending=False)

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

    # Save ATM±4 strikes (200 pts) for money flow analysis before narrowing
    df_atm4 = df[abs(df['strikePrice'] - atm_strike) <= 200].copy()
    df_atm4['Zone'] = df_atm4['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
    # Save ATM±5 strikes (250 pts) for unwinding/parallel winding analysis
    df_atm8 = df[abs(df['strikePrice'] - atm_strike) <= 250].copy()
    df_atm8['Zone'] = df_atm8['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
    atm_plus_minus_2 = df[abs(df['strikePrice'] - atm_strike) <= 100]
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
        _g = row.get
        row_data["LTP_Bias"] = "Bullish" if _g('lastPrice_CE', 0) > _g('lastPrice_PE', 0) else "Bearish"
        row_data["OI_Bias"] = "Bearish" if _g('openInterest_CE', 0) > _g('openInterest_PE', 0) else "Bullish"
        row_data["ChgOI_Bias"] = "Bearish" if _g('changeinOpenInterest_CE', 0) > _g('changeinOpenInterest_PE', 0) else "Bullish"
        row_data["Volume_Bias"] = "Bullish" if _g('totalTradedVolume_CE', 0) > _g('totalTradedVolume_PE', 0) else "Bearish"
        row_data["Delta_Bias"] = "Bullish" if _g('Delta_CE', 0) > abs(_g('Delta_PE', 0)) else "Bearish"
        row_data["Gamma_Bias"] = "Bullish" if _g('Gamma_CE', 0) > _g('Gamma_PE', 0) else "Bearish"
        row_data["Theta_Bias"] = "Bullish" if _g('Theta_CE', 0) < _g('Theta_PE', 0) else "Bearish"
        row_data["AskQty_Bias"] = "Bullish" if _g('askQty_PE', 0) > _g('askQty_CE', 0) else "Bearish"
        row_data["BidQty_Bias"] = "Bearish" if _g('bidQty_PE', 0) > _g('bidQty_CE', 0) else "Bullish"
        row_data["AskBid_Bias"] = "Bullish" if _g('bidQty_CE', 0) > _g('askQty_CE', 0) else "Bearish"
        row_data["IV_Bias"] = "Bullish" if _g('impliedVolatility_CE', 0) > _g('impliedVolatility_PE', 0) else "Bearish"
        delta_exp_ce = _g('Delta_CE', 0) * _g('openInterest_CE', 0)
        delta_exp_pe = _g('Delta_PE', 0) * _g('openInterest_PE', 0)
        gamma_exp_ce = _g('Gamma_CE', 0) * _g('openInterest_CE', 0)
        gamma_exp_pe = _g('Gamma_PE', 0) * _g('openInterest_PE', 0)
        row_data["DeltaExp"] = "Bullish" if delta_exp_ce > abs(delta_exp_pe) else "Bearish"
        row_data["GammaExp"] = "Bullish" if gamma_exp_ce > gamma_exp_pe else "Bearish"
        row_data["DVP_Bias"] = delta_volume_bias(
            row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
            row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
            row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
        )
        row_data["BidAskPressure"] = bid_ask_pressure
        row_data["PressureBias"] = pressure_bias
        for k in row_data:
            if "_Bias" in k or k in ["DeltaExp", "GammaExp"]:
                bias_val = row_data[k]
                if bias_val == "Bullish":
                    score += 1
                elif bias_val == "Bearish":
                    score -= 1
        row_data["BiasScore"] = score
        row_data["Verdict"] = final_verdict(score)
        if row_data['OI_Bias'] == "Bullish" and row_data['ChgOI_Bias'] == "Bullish":
            row_data["Operator_Entry"] = "Entry Bull"
        elif row_data['OI_Bias'] == "Bearish" and row_data['ChgOI_Bias'] == "Bearish":
            row_data["Operator_Entry"] = "Entry Bear"
        else:
            row_data["Operator_Entry"] = "No Entry"
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

    if 'Gamma_CE' in df_summary.columns and 'openInterest_CE' in df_summary.columns:
        df_summary['GammaExp_CE'] = df_summary['Gamma_CE'] * df_summary['openInterest_CE']
        df_summary['GammaExp_PE'] = df_summary['Gamma_PE'] * df_summary['openInterest_PE']
        df_summary['GammaExp_Net'] = df_summary['GammaExp_CE'] - df_summary['GammaExp_PE']
        max_gamma_ce_strike = df_summary.loc[df_summary['GammaExp_CE'].idxmax(), 'Strike'] if not df_summary['GammaExp_CE'].isna().all() else None
        max_gamma_pe_strike = df_summary.loc[df_summary['GammaExp_PE'].idxmax(), 'Strike'] if not df_summary['GammaExp_PE'].isna().all() else None
        df_summary['Gamma_SR'] = df_summary['Strike'].apply(
            lambda x: '🔴 Γ-Resist' if x == max_gamma_ce_strike else ('🟢 Γ-Support' if x == max_gamma_pe_strike else '-')
        )
    else:
        df_summary['Gamma_SR'] = '-'

    if 'Delta_CE' in df_summary.columns and 'openInterest_CE' in df_summary.columns:
        df_summary['DeltaExp_CE'] = df_summary['Delta_CE'] * df_summary['openInterest_CE']
        df_summary['DeltaExp_PE'] = abs(df_summary['Delta_PE'] * df_summary['openInterest_PE'])
        df_summary['DeltaExp_Net'] = df_summary['DeltaExp_CE'] - df_summary['DeltaExp_PE']
        max_delta_ce_strike = df_summary.loc[df_summary['DeltaExp_CE'].idxmax(), 'Strike'] if not df_summary['DeltaExp_CE'].isna().all() else None
        max_delta_pe_strike = df_summary.loc[df_summary['DeltaExp_PE'].idxmax(), 'Strike'] if not df_summary['DeltaExp_PE'].isna().all() else None
        df_summary['Delta_SR'] = df_summary['Strike'].apply(
            lambda x: '🔴 Δ-Resist' if x == max_delta_ce_strike else ('🟢 Δ-Support' if x == max_delta_pe_strike else '-')
        )
    else:
        df_summary['Delta_SR'] = '-'

    if 'bidQty_CE' in df_summary.columns and 'askQty_CE' in df_summary.columns:
        df_summary['Depth_CE'] = df_summary['bidQty_CE'] + df_summary['askQty_CE']
        df_summary['Depth_PE'] = df_summary['bidQty_PE'] + df_summary['askQty_PE']
        max_bid_pe_strike = df_summary.loc[df_summary['bidQty_PE'].idxmax(), 'Strike'] if not df_summary['bidQty_PE'].isna().all() else None
        max_ask_ce_strike = df_summary.loc[df_summary['askQty_CE'].idxmax(), 'Strike'] if not df_summary['askQty_CE'].isna().all() else None
        df_summary['Depth_SR'] = df_summary['Strike'].apply(
            lambda x: '🔴 Depth-R' if x == max_ask_ce_strike else ('🟢 Depth-S' if x == max_bid_pe_strike else '-')
        )
    else:
        df_summary['Depth_SR'] = '-'

    if 'openInterest_CE' in df_summary.columns and 'openInterest_PE' in df_summary.columns:
        max_oi_ce_strike = df_summary.loc[df_summary['openInterest_CE'].idxmax(), 'Strike'] if not df_summary['openInterest_CE'].isna().all() else None
        max_oi_pe_strike = df_summary.loc[df_summary['openInterest_PE'].idxmax(), 'Strike'] if not df_summary['openInterest_PE'].isna().all() else None
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

    if 'changeinOpenInterest_CE' in df_summary.columns and 'changeinOpenInterest_PE' in df_summary.columns:
        max_chgoi_ce_idx = df_summary['changeinOpenInterest_CE'].idxmax()
        max_chgoi_pe_idx = df_summary['changeinOpenInterest_PE'].idxmax()
        max_chgoi_ce_strike = df_summary.loc[max_chgoi_ce_idx, 'Strike'] if df_summary.loc[max_chgoi_ce_idx, 'changeinOpenInterest_CE'] > 0 else None
        max_chgoi_pe_strike = df_summary.loc[max_chgoi_pe_idx, 'Strike'] if df_summary.loc[max_chgoi_pe_idx, 'changeinOpenInterest_PE'] > 0 else None
        min_chgoi_ce_idx = df_summary['changeinOpenInterest_CE'].idxmin()
        min_chgoi_pe_idx = df_summary['changeinOpenInterest_PE'].idxmin()
        unwind_ce_strike = df_summary.loc[min_chgoi_ce_idx, 'Strike'] if df_summary.loc[min_chgoi_ce_idx, 'changeinOpenInterest_CE'] < 0 else None
        unwind_pe_strike = df_summary.loc[min_chgoi_pe_idx, 'Strike'] if df_summary.loc[min_chgoi_pe_idx, 'changeinOpenInterest_PE'] < 0 else None
        def get_chgoi_wall(strike):
            labels = []
            if strike == max_chgoi_ce_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_CE'].values[0]
                labels.append(f'🔴 CE+{int(chgoi_val/1000)}K')
            if strike == max_chgoi_pe_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_PE'].values[0]
                labels.append(f'🟢 PE+{int(chgoi_val/1000)}K')
            if strike == unwind_ce_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_CE'].values[0]
                labels.append(f'⚪ CE{int(chgoi_val/1000)}K')
            if strike == unwind_pe_strike:
                chgoi_val = df_summary.loc[df_summary['Strike'] == strike, 'changeinOpenInterest_PE'].values[0]
                labels.append(f'⚪ PE{int(chgoi_val/1000)}K')
            return ' | '.join(labels) if labels else '-'
        df_summary['ChgOI_Wall'] = df_summary['Strike'].apply(get_chgoi_wall)
    else:
        df_summary['ChgOI_Wall'] = '-'

    max_pain_strike, pain_df = calculate_max_pain(df_summary, underlying)
    if max_pain_strike:
        df_summary['Max_Pain'] = df_summary['Strike'].apply(
            lambda x: '🎯 MAX PAIN' if x == max_pain_strike else '-'
        )
    else:
        df_summary['Max_Pain'] = '-'

    display_cols = ['Strike', 'PCR', 'Verdict', 'ChgOI_Bias', 'Volume_Bias', 'Max_Pain',
                    'Gamma_SR', 'Delta_SR', 'Depth_SR', 'OI_Wall', 'ChgOI_Wall',
                    'Delta_Bias', 'Gamma_Bias', 'Theta_Bias', 'AskQty_Bias', 'BidQty_Bias', 'IV_Bias',
                    'DeltaExp', 'GammaExp', 'DVP_Bias', 'PressureBias', 'BidAskPressure',
                    'BiasScore', 'Operator_Entry', 'Scalp_Moment', 'FakeReal',
                    'ChgOI_Cmp', 'OI_Cmp', 'LTP_Bias', 'PCR_Signal', 'Zone', 'OI_Bias']

    display_cols = [col for col in display_cols if col in df_summary.columns]
    df_display = df_summary[display_cols].copy()

    bias_cols = [col for col in display_cols if '_Bias' in col or col in ['DeltaExp', 'GammaExp', 'PCR_Signal']]

    styled_df = df_display.style\
        .map(color_bias, subset=bias_cols)\
        .map(color_pcr, subset=['PCR'] if 'PCR' in display_cols else [])\
        .map(color_pressure, subset=['BidAskPressure'] if 'BidAskPressure' in display_cols else [])\
        .map(color_verdict, subset=['Verdict'] if 'Verdict' in display_cols else [])\
        .map(color_entry, subset=['Operator_Entry'] if 'Operator_Entry' in display_cols else [])\
        .map(color_fakereal, subset=['FakeReal'] if 'FakeReal' in display_cols else [])\
        .map(color_score, subset=['BiasScore'] if 'BiasScore' in display_cols else [])\
        .apply(highlight_atm_row, axis=1)

    sr_data = []

    if max_pain_strike:
        sr_data.append({
            'Type': '🎯 Max Pain',
            'Level': f"₹{max_pain_strike:.0f}",
            'Source': 'Options OI',
            'Strength': 'High',
            'Signal': 'Price magnet at expiry'
        })

    _sr_pairs = [
        ('openInterest_PE', '🟢 OI Wall Support', lambda v: f"PE OI: {v/100000:.1f}L", 'High', 'Strong support - PE writers defending', False),
        ('openInterest_CE', '🔴 OI Wall Resistance', lambda v: f"CE OI: {v/100000:.1f}L", 'High', 'Strong resistance - CE writers defending', False),
        ('GammaExp_PE', '🟢 Gamma Support', lambda v: 'Gamma Exposure PE', 'Medium', 'Dealers hedge here - price sticky', False),
        ('GammaExp_CE', '🔴 Gamma Resistance', lambda v: 'Gamma Exposure CE', 'Medium', 'Dealers hedge here - price sticky', False),
        ('DeltaExp_PE', '🟢 Delta Support', lambda v: 'Delta Exposure PE', 'Medium', 'Directional bias support', False),
        ('DeltaExp_CE', '🔴 Delta Resistance', lambda v: 'Delta Exposure CE', 'Medium', 'Directional bias resistance', False),
        ('changeinOpenInterest_PE', '🟢 Fresh PE Buildup', lambda v: f"ChgOI: +{v/1000:.0f}K", 'Fresh', 'New support forming today', True),
        ('changeinOpenInterest_CE', '🔴 Fresh CE Buildup', lambda v: f"ChgOI: +{v/1000:.0f}K", 'Fresh', 'New resistance forming today', True),
        ('bidQty_PE', '🟢 Depth Support', lambda v: 'Max PE Bid Qty', 'Real-time', 'Buyers actively defending', False),
        ('askQty_CE', '🔴 Depth Resistance', lambda v: 'Max CE Ask Qty', 'Real-time', 'Sellers actively defending', False),
    ]
    for col, sr_type, src_fn, strength, signal, check_positive in _sr_pairs:
        if col in df_summary.columns:
            idx = df_summary[col].idxmax()
            val = df_summary.loc[idx, col]
            if check_positive and val <= 0:
                continue
            sr_data.append({'Type': sr_type, 'Level': f"₹{df_summary.loc[idx, 'Strike']:.0f}",
                           'Source': src_fn(val), 'Strength': strength, 'Signal': signal})

    if pivot_data:
        tf_pivots = {}
        for pivot in pivot_data:
            tf = pivot['timeframe']
            if tf not in tf_pivots:
                tf_pivots[tf] = {'highs': [], 'lows': []}
            if pivot['type'] == 'high':
                tf_pivots[tf]['highs'].append(pivot['value'])
            else:
                tf_pivots[tf]['lows'].append(pivot['value'])
        for tf, src, strength, s_sig, r_sig in [
            ('5M', '5-Min Timeframe', 'Intraday', 'Short-term support level', 'Short-term resistance level'),
            ('15M', '15-Min Timeframe', 'Swing', 'Key intraday support', 'Key intraday resistance'),
            ('1H', '1-Hour Timeframe', 'Major', 'Strong hourly support - watch closely', 'Strong hourly resistance - watch closely')
        ]:
            if tf in tf_pivots:
                if tf_pivots[tf]['lows']:
                    sr_data.append({'Type': f'🟢 {tf} Pivot Support', 'Level': f"₹{max(tf_pivots[tf]['lows']):.0f}",
                                    'Source': src, 'Strength': strength, 'Signal': s_sig})
                if tf_pivots[tf]['highs']:
                    sr_data.append({'Type': f'🔴 {tf} Pivot Resistance', 'Level': f"₹{min(tf_pivots[tf]['highs']):.0f}",
                                    'Source': src, 'Strength': strength, 'Signal': r_sig})

    vob_blocks = None
    if vob_data:
        vob_sr_levels = vob_data.get('sr_levels', [])
        vob_blocks = vob_data.get('blocks', None)
        for vob_level in vob_sr_levels:
            sr_data.append({k: vob_level[k] for k in ['Type', 'Level', 'Source', 'Strength', 'Signal']})

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
        'vob_blocks': vob_blocks,
        'df_atm4': df_atm4,
        'df_atm8': df_atm8,
        'atm_strike': atm_strike,
    }

def display_analytics_dashboard(db, symbol="NIFTY50"):

def display_analytics_dashboard(db, symbol="NIFTY50"):
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
        metrics = [
            ("Average Price", f"₹{analytics_df['day_close'].mean():,.2f}"),
            ("Volatility (σ)", f"{analytics_df['price_change_pct'].std():.2f}%"),
            ("Max Daily Gain", f"{analytics_df['price_change_pct'].max():.2f}%"),
            ("Max Daily Loss", f"{analytics_df['price_change_pct'].min():.2f}%")
        ]
        for col, (label, val) in zip(st.columns(4), metrics):
            col.metric(label, val)

def analyze_strike_activity(df_summary, underlying_price):

def analyze_strike_activity(df_summary, underlying_price):
    """Comprehensive option chain analysis: Call/Put activity, support/resistance, market bias."""
    if df_summary is None or df_summary.empty or 'Zone' not in df_summary.columns:
        return None

    atm_idx = df_summary[df_summary['Zone'] == 'ATM'].index
    if len(atm_idx) == 0:
        return None
    atm_pos = df_summary.index.get_loc(atm_idx[0])
    start_idx = max(0, atm_pos - 5)
    end_idx = min(len(df_summary), atm_pos + 6)
    near_atm = df_summary.iloc[start_idx:end_idx].copy()

    # Compute median OI and Volume for "high OI / high vol" thresholds
    median_ce_oi = near_atm['openInterest_CE'].median() if 'openInterest_CE' in near_atm.columns else 0
    median_pe_oi = near_atm['openInterest_PE'].median() if 'openInterest_PE' in near_atm.columns else 0
    median_ce_vol = near_atm['totalTradedVolume_CE'].median() if 'totalTradedVolume_CE' in near_atm.columns else 0
    median_pe_vol = near_atm['totalTradedVolume_PE'].median() if 'totalTradedVolume_PE' in near_atm.columns else 0

    strike_analysis = []
    for _, row in near_atm.iterrows():
        strike = row['Strike']
        zone = row.get('Zone', '')

        ce_oi = row.get('openInterest_CE', 0) or 0
        pe_oi = row.get('openInterest_PE', 0) or 0
        ce_chg_oi = row.get('changeinOpenInterest_CE', 0) or 0
        pe_chg_oi = row.get('changeinOpenInterest_PE', 0) or 0
        ce_ltp = row.get('lastPrice_CE', 0) or 0
        pe_ltp = row.get('lastPrice_PE', 0) or 0
        ce_vol = row.get('totalTradedVolume_CE', 0) or 0
        pe_vol = row.get('totalTradedVolume_PE', 0) or 0
        ce_iv = row.get('impliedVolatility_CE', 0) or 0
        pe_iv = row.get('impliedVolatility_PE', 0) or 0

        # --- CALL SIDE ---
        ce_high_oi = ce_oi > median_ce_oi * 1.2
        ce_high_vol = ce_vol > median_ce_vol * 1.2
        ce_oi_rising = ce_chg_oi > 0
        ce_oi_falling = ce_chg_oi < 0
        price_below_strike = underlying_price < strike

        # Call Writer Activity
        call_writing = ce_high_oi and ce_oi_rising
        call_capping = call_writing and price_below_strike
        call_capping_confirmed = call_capping and ce_high_vol  # OI + Volume both confirm active writing

        # Call Buyer Activity
        ce_long_buildup = ce_oi_rising and ce_ltp > 0 and ce_vol > 0
        ce_short_covering = ce_oi_falling and ce_ltp > 0

        # Call Classification — High Conviction requires volume confirmation
        if ce_high_oi and ce_oi_rising and price_below_strike and ce_high_vol:
            call_class = "High Conviction Resistance"
            call_strength = "High Conviction"
        elif ce_high_oi and ce_oi_rising and price_below_strike:
            call_class = "Strong Resistance"
            call_strength = "Strong"
        elif ce_oi_falling and not price_below_strike:
            call_class = "Breakout Zone"
            call_strength = "Breaking"
        elif ce_high_oi and not ce_oi_rising:
            call_class = "Weak Resistance"
            call_strength = "Weak"
        elif ce_oi_rising:
            call_class = "Moderate Resistance"
            call_strength = "Moderate"
        else:
            call_class = "Weak Resistance"
            call_strength = "Weak"

        # Call activity label
        if ce_short_covering:
            call_activity = "Short Covering"
        elif call_capping_confirmed:
            call_activity = "Writing (Vol Confirmed)"
        elif call_writing:
            call_activity = "Writing (Resistance)"
        elif ce_long_buildup:
            call_activity = "Long Build-up"
        else:
            call_activity = "Low Activity"

        # --- PUT SIDE ---
        pe_high_oi = pe_oi > median_pe_oi * 1.2
        pe_high_vol = pe_vol > median_pe_vol * 1.2
        pe_oi_rising = pe_chg_oi > 0
        pe_oi_falling = pe_chg_oi < 0
        price_above_strike = underlying_price > strike

        # Put Writer Activity
        put_writing = pe_high_oi and pe_oi_rising
        put_support = put_writing and price_above_strike
        put_support_confirmed = put_support and pe_high_vol  # OI + Volume both confirm active support

        # Put Buyer Activity
        pe_long_buildup = pe_oi_rising and pe_ltp > 0 and pe_vol > 0
        pe_long_unwinding = pe_oi_falling and pe_ltp > 0

        # Put Classification — High Conviction requires volume confirmation
        if pe_high_oi and pe_oi_rising and price_above_strike and pe_high_vol:
            put_class = "High Conviction Support"
            put_strength = "High Conviction"
        elif pe_high_oi and pe_oi_rising and price_above_strike:
            put_class = "Strong Support"
            put_strength = "Strong"
        elif pe_oi_falling and not price_above_strike:
            put_class = "Breakdown Zone"
            put_strength = "Breaking"
        elif pe_high_oi and not pe_oi_rising:
            put_class = "Weak Support"
            put_strength = "Weak"
        elif pe_oi_rising:
            put_class = "Moderate Support"
            put_strength = "Moderate"
        else:
            put_class = "Weak Support"
            put_strength = "Weak"

        # Put activity label
        if pe_long_unwinding:
            put_activity = "Long Unwinding"
        elif put_support_confirmed:
            put_activity = "Writing (Vol Confirmed)"
        elif put_writing:
            put_activity = "Writing (Support)"
        elif pe_long_buildup:
            put_activity = "Long Build-up (Bearish)"
        else:
            put_activity = "Low Activity"

        # Trapped writers detection
        call_trapped = ce_high_oi and ce_oi_falling and not price_below_strike  # price went above, writers trapped
        put_trapped = pe_high_oi and pe_oi_falling and price_below_strike  # price went below, writers trapped

        # Market depth + bid-ask pressure
        ce_bid_qty = row.get('bidQty_CE', 0) or 0
        ce_ask_qty = row.get('askQty_CE', 0) or 0
        pe_bid_qty = row.get('bidQty_PE', 0) or 0
        pe_ask_qty = row.get('askQty_PE', 0) or 0
        bid_ask_pressure = (ce_bid_qty + pe_bid_qty) - (ce_ask_qty + pe_ask_qty)

        strike_analysis.append({
            'Strike': strike, 'Zone': zone,
            'CE_OI': ce_oi, 'CE_ChgOI': ce_chg_oi, 'CE_LTP': ce_ltp, 'CE_Vol': ce_vol, 'CE_IV': ce_iv,
            'CE_Vol_High': ce_high_vol,
            'PE_OI': pe_oi, 'PE_ChgOI': pe_chg_oi, 'PE_LTP': pe_ltp, 'PE_Vol': pe_vol, 'PE_IV': pe_iv,
            'PE_Vol_High': pe_high_vol,
            'bidQty_CE': ce_bid_qty, 'askQty_CE': ce_ask_qty,
            'bidQty_PE': pe_bid_qty, 'askQty_PE': pe_ask_qty,
            'BidAskPressure': bid_ask_pressure,
            'changeinOpenInterest_CE': ce_chg_oi, 'changeinOpenInterest_PE': pe_chg_oi,
            'Call_Class': call_class, 'Call_Strength': call_strength, 'Call_Activity': call_activity,
            'Call_Capping_Confirmed': call_capping_confirmed,
            'Put_Class': put_class, 'Put_Strength': put_strength, 'Put_Activity': put_activity,
            'Put_Support_Confirmed': put_support_confirmed,
            'Call_Trapped': call_trapped, 'Put_Trapped': put_trapped,
        })

    analysis_df = pd.DataFrame(strike_analysis)

    # Top 3 Resistance (Call side) - sort by CE OI descending, prefer strong
    strength_order = {'High Conviction': 0, 'Strong': 1, 'Moderate': 2, 'Weak': 3, 'Breaking': 4}
    res_df = analysis_df[analysis_df['Strike'] >= underlying_price].copy()
    res_df['_sort'] = res_df['Call_Strength'].map(strength_order).fillna(3)
    res_df = res_df.sort_values(['_sort', 'CE_OI'], ascending=[True, False]).head(3)
    top_resistance = res_df[['Strike', 'Call_Class', 'Call_Strength', 'CE_OI', 'CE_ChgOI', 'Call_Activity']].copy()

    # Top 3 Support (Put side) - sort by PE OI descending, prefer strong
    sup_df = analysis_df[analysis_df['Strike'] <= underlying_price].copy()
    sup_df['_sort'] = sup_df['Put_Strength'].map(strength_order).fillna(3)
    sup_df = sup_df.sort_values(['_sort', 'PE_OI'], ascending=[True, False]).head(3)
    top_support = sup_df[['Strike', 'Put_Class', 'Put_Strength', 'PE_OI', 'PE_ChgOI', 'Put_Activity']].copy()

    # Trapped writers
    trapped_call_writers = analysis_df[analysis_df['Call_Trapped']][['Strike', 'CE_OI', 'CE_ChgOI']].copy()
    trapped_put_writers = analysis_df[analysis_df['Put_Trapped']][['Strike', 'PE_OI', 'PE_ChgOI']].copy()

    # Breakout / Breakdown zones
    breakout_zones = analysis_df[analysis_df['Call_Class'] == 'Breakout Zone'][['Strike', 'CE_OI', 'CE_ChgOI']].copy()
    breakdown_zones = analysis_df[analysis_df['Put_Class'] == 'Breakdown Zone'][['Strike', 'PE_OI', 'PE_ChgOI']].copy()

    # Market Interpretation
    total_ce_chg = analysis_df['CE_ChgOI'].sum()
    total_pe_chg = analysis_df['PE_ChgOI'].sum()
    strong_res_count = len(analysis_df[analysis_df['Call_Class'] == 'Strong Resistance'])
    strong_sup_count = len(analysis_df[analysis_df['Put_Class'] == 'Strong Support'])
    breakout_count = len(breakout_zones)
    breakdown_count = len(breakdown_zones)
    call_unwinding = total_ce_chg < 0
    put_unwinding = total_pe_chg < 0
    call_building = total_ce_chg > 0
    put_building = total_pe_chg > 0

    # Confidence scoring
    confidence = 50
    bias_signals = []

    if call_unwinding and put_building:
        market_bias = "Bullish"
        confidence += 20
        bias_signals.append("Call OI unwinding + Put writing increasing")
    elif call_building and put_unwinding:
        market_bias = "Bearish"
        confidence += 20
        bias_signals.append("Call writing increasing + Put OI unwinding")
    elif strong_res_count > 0 and strong_sup_count > 0:
        market_bias = "Sideways"
        confidence += 10
        bias_signals.append("Strong resistance above + Strong support below")
    elif breakout_count > 0 and strong_sup_count > 0:
        market_bias = "Bullish (Breakout)"
        confidence += 25
        bias_signals.append("Call writers trapped + Put support strong")
    elif breakdown_count > 0 and strong_res_count > 0:
        market_bias = "Bearish (Breakdown)"
        confidence += 25
        bias_signals.append("Put writers trapped + Call resistance strong")
    elif put_building and call_building:
        if total_pe_chg > total_ce_chg:
            market_bias = "Mildly Bullish"
            confidence += 10
            bias_signals.append("Both building but PE OI > CE OI change")
        else:
            market_bias = "Mildly Bearish"
            confidence += 10
            bias_signals.append("Both building but CE OI > PE OI change")
    else:
        market_bias = "Neutral"
        bias_signals.append("No clear directional signal")

    if len(trapped_call_writers) > 0:
        confidence += 10
        bias_signals.append(f"{len(trapped_call_writers)} trapped call writer(s)")
    if len(trapped_put_writers) > 0:
        confidence += 10
        bias_signals.append(f"{len(trapped_put_writers)} trapped put writer(s)")
    if strong_sup_count >= 2:
        confidence += 5
        bias_signals.append(f"{strong_sup_count} strong support levels")
    if strong_res_count >= 2:
        confidence += 5
        bias_signals.append(f"{strong_res_count} strong resistance levels")

    confidence = min(confidence, 95)

    return {
        'analysis_df': analysis_df,
        'top_resistance': top_resistance,
        'top_support': top_support,
        'trapped_call_writers': trapped_call_writers,
        'trapped_put_writers': trapped_put_writers,
        'breakout_zones': breakout_zones,
        'breakdown_zones': breakdown_zones,
        'market_bias': market_bias,
        'confidence': confidence,
        'bias_signals': bias_signals,
        'total_ce_chg': total_ce_chg,
        'total_pe_chg': total_pe_chg,
    }

def send_option_chain_signal(sa_result, underlying_price, force=False):

def send_option_chain_signal(sa_result, underlying_price, force=False):
    """Store option chain analysis snapshot to Supabase oc_signal_history table."""
    if sa_result is None:
        return
    bias = sa_result['market_bias']
    conf = sa_result['confidence']
    analysis_df = sa_result['analysis_df']

    now = datetime.now(pytz.timezone('Asia/Kolkata'))

    # Active signals
    active_signals = []
    capping = analysis_df[
        analysis_df['Call_Class'].isin(['High Conviction Resistance', 'Strong Resistance']) &
        analysis_df['Call_Activity'].isin(['Writing (Vol Confirmed)', 'Writing (Resistance)'])
    ]
    for _, r in capping.iterrows():
        vol_tag = "VOL CONFIRMED" if r.get('CE_Vol_High', False) else "Low Vol"
        active_signals.append(f"CALL CAPPING ₹{r['Strike']:.0f} [{vol_tag}] OI:{r['CE_OI']/100000:.1f}L ChgOI:+{r['CE_ChgOI']/1000:.0f}K")
    support = analysis_df[
        analysis_df['Put_Class'].isin(['High Conviction Support', 'Strong Support']) &
        analysis_df['Put_Activity'].isin(['Writing (Vol Confirmed)', 'Writing (Support)'])
    ]
    for _, r in support.iterrows():
        vol_tag = "VOL CONFIRMED" if r.get('PE_Vol_High', False) else "Low Vol"
        active_signals.append(f"PUT CAPPING ₹{r['Strike']:.0f} [{vol_tag}] OI:{r['PE_OI']/100000:.1f}L ChgOI:+{r['PE_ChgOI']/1000:.0f}K")
    breakout = analysis_df[analysis_df['Call_Class'] == 'Breakout Zone']
    for _, r in breakout.iterrows():
        active_signals.append(f"BREAKOUT LOADING ₹{r['Strike']:.0f}")
    breakdown = analysis_df[analysis_df['Put_Class'] == 'Breakdown Zone']
    for _, r in breakdown.iterrows():
        active_signals.append(f"BREAKDOWN LOADING ₹{r['Strike']:.0f}")

    if 'Bullish' in bias:
        condition = "BULLISH"
    elif 'Bearish' in bias:
        condition = "BEARISH"
    else:
        condition = "SIDEWAYS"

    resistance_strikes = [
        {'strike': int(r['Strike']), 'strength': r['Call_Strength'],
         'oi_l': round(r['CE_OI']/100000, 1), 'activity': r['Call_Activity']}
        for _, r in sa_result['top_resistance'].iterrows()
    ]
    support_strikes = [
        {'strike': int(r['Strike']), 'strength': r['Put_Strength'],
         'oi_l': round(r['PE_OI']/100000, 1), 'activity': r['Put_Activity']}
        for _, r in sa_result['top_support'].iterrows()
    ]
    breakout_level = float(breakout.iloc[0]['Strike']) if not breakout.empty else None
    breakdown_level = float(breakdown.iloc[0]['Strike']) if not breakdown.empty else None

    if not force:
        if not active_signals:
            return
        if condition == "SIDEWAYS":
            return
        last_stored = st.session_state.get('_last_oc_store_time')
        if last_stored and (now - last_stored).total_seconds() < 300:
            return

    try:
        db = st.session_state.get('db')
        if db:
            db.upsert_oc_signal({
                'timestamp': now.isoformat(),
                'spot_price': underlying_price,
                'condition': condition,
                'confidence': conf,
                'resistance_strikes': resistance_strikes,
                'support_strikes': support_strikes,
                'active_signals': active_signals,
                'breakout_level': breakout_level,
                'breakdown_level': breakdown_level,
                'bias_reasoning': list(sa_result.get('bias_signals', [])),
            })
            if not force:
                st.session_state._last_oc_store_time = now
    except Exception:
        pass

def detect_candle_patterns(df, lookback=5):

def generate_master_signal(df, sa_result, gex_data, confluence_data, underlying_price, api):
    """Generate comprehensive trading signal using all available data."""
    if df is None or df.empty:
        return None

    # 1. Candle Pattern (last 5 candles on 5-min chart)
    try:
        df_5m = df.set_index('datetime').resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
    except Exception:
        df_5m = df
    candle = detect_candle_patterns(df_5m, lookback=5)
    try:
        st.session_state._df_5m = df_5m
    except Exception:
        pass

    # 2. Order Blocks
    ob = detect_order_blocks(df, lookback=20)

    # 3. Volume Analysis
    vol = detect_volume_spike(df, lookback=5)

    # 4. Key levels from existing app data (empty when option chain not available)
    support_levels = []
    resistance_levels = []
    if sa_result is not None:
        try:
            support_levels = [float(r['Strike']) for _, r in sa_result['top_support'].iterrows()] if not sa_result['top_support'].empty else []
            resistance_levels = [float(r['Strike']) for _, r in sa_result['top_resistance'].iterrows()] if not sa_result['top_resistance'].empty else []
        except Exception:
            pass

    # 5. Candle location
    location = get_candle_location(underlying_price, support_levels, resistance_levels, gex_data, ob)

    # 6. VIX (with session state caching)
    vix_data = {'vix': 0, 'direction': 'Unknown'}
    try:
        if 'vix_history' not in st.session_state:
            st.session_state.vix_history = []
        vix_resp = fetch_vix_data(api)
        vix_val = vix_resp.get('vix', 0)
        if vix_val > 0:
            st.session_state.vix_history.append(vix_val)
            if len(st.session_state.vix_history) > 50:
                st.session_state.vix_history = st.session_state.vix_history[-50:]
            if len(st.session_state.vix_history) >= 2:
                vix_dir = 'Rising' if st.session_state.vix_history[-1] > st.session_state.vix_history[0] else 'Falling'
            else:
                vix_dir = 'Unknown'
            vix_data = {'vix': vix_val, 'direction': vix_dir}
    except Exception:
        pass

    # 7. Index & Stock Alignment (with caching - fetch every 120s to reduce API calls)
    alignment = {}
    try:
        if 'alignment_data' not in st.session_state:
            st.session_state.alignment_data = {}
            st.session_state.alignment_last_fetch = None
        ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
        should_fetch = st.session_state.alignment_last_fetch is None or \
            (ist_now - st.session_state.alignment_last_fetch).total_seconds() > 120
        if should_fetch:
            alignment = fetch_alignment_data(api)
            if alignment:
                st.session_state.alignment_data = alignment
                st.session_state.alignment_last_fetch = ist_now
            # Fetch sector rotation alongside alignment (same cadence)
            try:
                _sr = compute_sector_rotation()
                if _sr:
                    st.session_state._sector_rotation = _sr
            except Exception:
                pass
        alignment = st.session_state.alignment_data

        # Also add NIFTY's own sentiment from the main df (5-min chart)
        if df is not None and not df.empty:
            try:
                nifty_5m = df.set_index('datetime').resample('5min').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna().reset_index()
            except Exception:
                nifty_5m = df
            nifty_cp = detect_candle_patterns(nifty_5m, lookback=5)
            def _nifty_sentiment(sub):
                if sub is None or len(sub) < 2:
                    return 'N/A', 0
                fc, lc = sub.iloc[0]['close'], sub.iloc[-1]['close']
                pct = ((lc - fc) / fc) * 100 if fc > 0 else 0
                gc = sum(1 for _, r in sub.iterrows() if r['close'] > r['open'])
                if pct > 0.1 and gc > len(sub) // 2:
                    return 'Bullish', round(pct, 2)
                elif pct < -0.1 and gc <= len(sub) // 2:
                    return 'Bearish', round(pct, 2)
                return 'Neutral', round(pct, 2)
            s10, p10 = _nifty_sentiment(df.tail(10))
            s1h, p1h = _nifty_sentiment(df.tail(60))
            s4h, p4h = _nifty_sentiment(df.tail(240))
            # Use 1-min pct_series from fetch_alignment_data (NIFTY 50 now included there)
            # so the chart line has the same granularity as all other instruments.
            nifty_align_1m = alignment.get('NIFTY 50', {})
            nifty_pct_time = nifty_align_1m.get('pct_series_time', [])
            nifty_pct_vals = nifty_align_1m.get('pct_series_vals', [])
            if not nifty_pct_time:
                # fallback: compute from main df if 1-min data not available
                today_ist = datetime.now(pytz.timezone('Asia/Kolkata')).date()
                df_today = df[df['datetime'].dt.date == today_ist]
                if df_today.empty:
                    df_today = df
                nifty_day_open = df_today.iloc[0]['open']
                nifty_pct_time = df_today['datetime'].tolist()
                nifty_pct_vals = [((c - nifty_day_open) / nifty_day_open) * 100 if nifty_day_open > 0 else 0 for c in df_today['close'].tolist()]
            s1d, p1d = _nifty_sentiment(df.tail(375))
            s4d, p4d = _nifty_sentiment(df.tail(1500))
            alignment['NIFTY 50'] = {
                'ltp': df.iloc[-1]['close'], 'trend': nifty_cp['direction'],
                'candle_pattern': nifty_cp['pattern'], 'candle_dir': nifty_cp['direction'],
                'candles': nifty_cp.get('candles', []),
                'bull_count': nifty_cp.get('bull_count', 0), 'bear_count': nifty_cp.get('bear_count', 0),
                'sentiment_10m': s10, 'pct_10m': p10,
                'sentiment_1h': s1h, 'pct_1h': p1h,
                'sentiment_4h': s4h, 'pct_4h': p4h,
                'sentiment_1d': s1d, 'pct_1d': p1d,
                'sentiment_4d': s4d, 'pct_4d': p4d,
                'day_high': df['high'].max(), 'day_low': df['low'].min(),
                'open': nifty_align_1m.get('open', df.iloc[0]['open']),
                'pct_series_time': nifty_pct_time,
                'pct_series_vals': nifty_pct_vals,
            }

        # VIX direction from alignment data
        vix_align = alignment.get('INDIA VIX', {})
        if vix_align.get('ltp', 0) > 0 and vix_data.get('vix', 0) == 0:
            vix_data['vix'] = vix_align['ltp']
            vix_data['direction'] = vix_align.get('sentiment_10m', 'Unknown')
            if vix_align.get('sentiment_10m') == 'Bullish':
                vix_data['direction'] = 'Rising'
            elif vix_align.get('sentiment_10m') == 'Bearish':
                vix_data['direction'] = 'Falling'
    except Exception:
        pass

    # === EXISTING APP DATA ===
    _sa = sa_result or {}
    market_bias = _sa.get('market_bias', 'Neutral')
    app_confidence = _sa.get('confidence', 50)

    # GEX data
    net_gex = gex_data.get('total_gex', 0) if gex_data else 0
    gamma_flip = gex_data.get('gamma_flip_level') if gex_data else None
    gex_magnet = gex_data.get('gex_magnet') if gex_data else None
    gex_repeller = gex_data.get('gex_repeller') if gex_data else None
    gex_signal = gex_data.get('gex_signal', '') if gex_data else ''
    gex_interpretation = gex_data.get('gex_interpretation', '') if gex_data else ''
    market_mode = 'TREND' if net_gex < 0 else 'RANGE'

    # ATM GEX
    atm_gex = 0
    if gex_data and 'gex_df' in gex_data:
        gex_df = gex_data['gex_df']
        atm_rows = gex_df[gex_df['Zone'] == 'ATM'] if 'Zone' in gex_df.columns else pd.DataFrame()
        if not atm_rows.empty:
            atm_gex = atm_rows.iloc[0].get('Net_GEX', 0)

    above_gamma_flip = underlying_price > gamma_flip if gamma_flip else None

    # PCR × GEX Confluence
    pcr_gex_badge = confluence_data[0] if confluence_data else ''
    pcr_gex_signal = confluence_data[1] if confluence_data and len(confluence_data) > 1 else ''

    # === CONFLUENCE SCORING ===
    score = 0
    reasons = []

    # 1. Candle pattern
    if candle['direction'] == 'Bullish':
        score += 1
        reasons.append(f"Bullish {candle['pattern']}")
    elif candle['direction'] == 'Bearish':
        score -= 1
        reasons.append(f"Bearish {candle['pattern']}")

    # 2. Volume spike
    if vol['spike']:
        score += 1 if candle['direction'] == 'Bullish' else -1 if candle['direction'] == 'Bearish' else 0
        reasons.append(f"Volume Spike ({vol['ratio']}x)")

    # 3. Order Block
    near_bullish_ob = any('Bullish OB' in loc for loc in location)
    near_bearish_ob = any('Bearish OB' in loc for loc in location)
    if near_bullish_ob:
        score += 1
        reasons.append("At Bullish Order Block")
    elif near_bearish_ob:
        score -= 1
        reasons.append("At Bearish Order Block")

    # 4. GEX confirmation
    if net_gex < 0 and candle['direction'] == 'Bullish' and above_gamma_flip:
        score += 1
        reasons.append("GEX Trend + Above Gamma Flip")
    elif net_gex < 0 and candle['direction'] == 'Bearish' and not above_gamma_flip:
        score -= 1
        reasons.append("GEX Trend + Below Gamma Flip")
    elif net_gex > 0:
        reasons.append("GEX Range Mode (pinning)")

    # 5. PCR / Option chain confirmation
    if 'Bullish' in market_bias:
        score += 1
        reasons.append(f"Option Chain: {market_bias}")
    elif 'Bearish' in market_bias:
        score -= 1
        reasons.append(f"Option Chain: {market_bias}")

    # 6. Alignment (use 10m sentiment for immediate direction)
    non_vix = {k: v for k, v in alignment.items() if 'VIX' not in k}
    bullish_align = sum(1 for v in non_vix.values() if v.get('sentiment_10m') == 'Bullish')
    bearish_align = sum(1 for v in non_vix.values() if v.get('sentiment_10m') == 'Bearish')
    total_align = len(non_vix) if non_vix else 1
    if bullish_align > bearish_align and bullish_align >= 2:
        score += 1
        reasons.append(f"Alignment Bullish ({bullish_align}/{total_align})")
    elif bearish_align > bullish_align and bearish_align >= 2:
        score -= 1
        reasons.append(f"Alignment Bearish ({bearish_align}/{total_align})")

    # 7. VIX
    vix_dir = vix_data.get('direction', 'Unknown')
    if vix_dir == 'Falling' and score > 0:
        score += 1
        reasons.append("VIX Falling (Aligned)")
    elif vix_dir == 'Rising' and score < 0:
        score += -1  # more bearish
        reasons.append("VIX Rising (Aligned)")
    elif vix_dir == 'Rising' and score > 0:
        reasons.append("VIX Rising (Opposite)")
    elif vix_dir == 'Falling' and score < 0:
        reasons.append("VIX Falling (Opposite)")

    # 8. OI Timeline Trend (ATM support/resistance building/breaking from oi_history)
    oi_trend = {'atm_strike': None, 'ce_activity': 'N/A', 'pe_activity': 'N/A',
                'support_status': 'N/A', 'resistance_status': 'N/A', 'signal': 'Neutral',
                'ce_oi_pct': 0, 'ce_ltp_pct': 0, 'pe_oi_pct': 0, 'pe_ltp_pct': 0,
                'ce_chgoi': 0, 'pe_chgoi': 0, 'ce_chgoi_trend': 'N/A', 'pe_chgoi_trend': 'N/A'}
    try:
        oi_hist = getattr(st.session_state, 'oi_history', [])
        chgoi_hist = getattr(st.session_state, 'chgoi_history', [])
        oi_strikes = getattr(st.session_state, 'oi_current_strikes', [])
        if len(oi_hist) >= 3 and oi_strikes:
            sorted_strikes = sorted(oi_strikes)
            atm_idx = len(sorted_strikes) // 2
            atm_s = str(sorted_strikes[atm_idx])
            oi_trend['atm_strike'] = int(atm_s)
            oi_df = pd.DataFrame(oi_hist)

            # CE OI + LTP trend
            ce_oi_col, pe_oi_col = f'{atm_s}_CE', f'{atm_s}_PE'
            ce_ltp_col, pe_ltp_col = f'{atm_s}_CE_LTP', f'{atm_s}_PE_LTP'

            if ce_oi_col in oi_df.columns and ce_ltp_col in oi_df.columns:
                ce_oi_first, ce_oi_last = oi_df[ce_oi_col].iloc[0], oi_df[ce_oi_col].iloc[-1]
                ce_oi_change = ce_oi_last - ce_oi_first
                ce_ltp_first, ce_ltp_last = oi_df[ce_ltp_col].iloc[0], oi_df[ce_ltp_col].iloc[-1]
                ce_ltp_change = ce_ltp_last - ce_ltp_first
                if ce_oi_change > 0 and ce_ltp_change > 0:
                    oi_trend['ce_activity'] = 'Long Building'
                elif ce_oi_change > 0 and ce_ltp_change <= 0:
                    oi_trend['ce_activity'] = 'Short Building'
                elif ce_oi_change < 0 and ce_ltp_change > 0:
                    oi_trend['ce_activity'] = 'Short Covering'
                elif ce_oi_change < 0 and ce_ltp_change <= 0:
                    oi_trend['ce_activity'] = 'Long Unwinding'
                oi_trend['ce_oi_pct'] = round((ce_oi_change / ce_oi_first * 100) if ce_oi_first > 0 else 0, 1)
                oi_trend['ce_ltp_pct'] = round((ce_ltp_change / ce_ltp_first * 100) if ce_ltp_first > 0 else 0, 1)

            if pe_oi_col in oi_df.columns and pe_ltp_col in oi_df.columns:
                pe_oi_first, pe_oi_last = oi_df[pe_oi_col].iloc[0], oi_df[pe_oi_col].iloc[-1]
                pe_oi_change = pe_oi_last - pe_oi_first
                pe_ltp_first, pe_ltp_last = oi_df[pe_ltp_col].iloc[0], oi_df[pe_ltp_col].iloc[-1]
                pe_ltp_change = pe_ltp_last - pe_ltp_first
                if pe_oi_change > 0 and pe_ltp_change > 0:
                    oi_trend['pe_activity'] = 'Long Building'
                elif pe_oi_change > 0 and pe_ltp_change <= 0:
                    oi_trend['pe_activity'] = 'Short Building'
                elif pe_oi_change < 0 and pe_ltp_change < 0:
                    oi_trend['pe_activity'] = 'Short Covering'
                elif pe_oi_change < 0 and pe_ltp_change >= 0:
                    oi_trend['pe_activity'] = 'Long Unwinding'
                oi_trend['pe_oi_pct'] = round((pe_oi_change / pe_oi_first * 100) if pe_oi_first > 0 else 0, 1)
                oi_trend['pe_ltp_pct'] = round((pe_ltp_change / pe_ltp_first * 100) if pe_ltp_first > 0 else 0, 1)

            # Support status (from PE activity): PE Short Building = Support Building
            ce_act = oi_trend['ce_activity']
            pe_act = oi_trend['pe_activity']
            if pe_act == 'Short Building':
                oi_trend['support_status'] = 'Building Strong'
            elif pe_act == 'Long Building':
                oi_trend['support_status'] = 'Building (Bearish bets rising)'
            elif pe_act == 'Short Covering':
                oi_trend['support_status'] = 'Breaking'
            elif pe_act == 'Long Unwinding':
                oi_trend['support_status'] = 'Weakening'

            # Resistance status (from CE activity): CE Short Building = Resistance Building
            if ce_act == 'Short Building':
                oi_trend['resistance_status'] = 'Building Strong'
            elif ce_act == 'Long Building':
                oi_trend['resistance_status'] = 'Building (Bullish bets rising)'
            elif ce_act == 'Short Covering':
                oi_trend['resistance_status'] = 'Breaking'
            elif ce_act == 'Long Unwinding':
                oi_trend['resistance_status'] = 'Weakening'

            # Overall OI trend signal
            if pe_act == 'Short Building' and ce_act in ['Short Covering', 'Long Unwinding']:
                oi_trend['signal'] = 'Bullish'
            elif ce_act == 'Short Building' and pe_act in ['Short Covering', 'Long Unwinding']:
                oi_trend['signal'] = 'Bearish'
            elif pe_act == 'Short Building' and ce_act == 'Short Building':
                oi_trend['signal'] = 'Range'
            elif ce_act == 'Short Covering' and pe_act == 'Short Covering':
                oi_trend['signal'] = 'Volatile'
            elif pe_act == 'Short Building' or ce_act in ['Short Covering', 'Long Unwinding']:
                oi_trend['signal'] = 'Mildly Bullish'
            elif ce_act == 'Short Building' or pe_act in ['Short Covering', 'Long Unwinding']:
                oi_trend['signal'] = 'Mildly Bearish'

            # ChgOI trend
            if len(chgoi_hist) >= 3:
                chgoi_df = pd.DataFrame(chgoi_hist)
                ce_chgoi_col, pe_chgoi_col = f'{atm_s}_CE', f'{atm_s}_PE'
                if ce_chgoi_col in chgoi_df.columns and pe_chgoi_col in chgoi_df.columns:
                    oi_trend['ce_chgoi'] = int(chgoi_df[ce_chgoi_col].iloc[-1])
                    oi_trend['pe_chgoi'] = int(chgoi_df[pe_chgoi_col].iloc[-1])
                    oi_trend['ce_chgoi_trend'] = 'Increasing' if chgoi_df[ce_chgoi_col].iloc[-1] > chgoi_df[ce_chgoi_col].iloc[0] else 'Decreasing'
                    oi_trend['pe_chgoi_trend'] = 'Increasing' if chgoi_df[pe_chgoi_col].iloc[-1] > chgoi_df[pe_chgoi_col].iloc[0] else 'Decreasing'
    except Exception:
        pass

    # OI Trend scoring
    oi_sig = oi_trend.get('signal', 'Neutral')
    if oi_sig == 'Bullish':
        score += 1
        reasons.append(f"OI Trend: Bullish (Sup Building + Res Breaking)")
    elif oi_sig == 'Mildly Bullish':
        score += 1
        reasons.append(f"OI Trend: Mildly Bullish (Sup:{oi_trend['support_status']} | Res:{oi_trend['resistance_status']})")
    elif oi_sig == 'Bearish':
        score -= 1
        reasons.append(f"OI Trend: Bearish (Res Building + Sup Breaking)")
    elif oi_sig == 'Mildly Bearish':
        score -= 1
        reasons.append(f"OI Trend: Mildly Bearish (Sup:{oi_trend['support_status']} | Res:{oi_trend['resistance_status']})")
    elif oi_sig == 'Range':
        reasons.append("OI Trend: Range (Both Support & Resistance Building)")
    elif oi_sig == 'Volatile':
        reasons.append("OI Trend: Volatile (Both Covering)")

    # 8b. Decapping / Depeg (intraday OI reduction at dominant wall)
    _dc = getattr(st.session_state, '_decapping', None)
    _dp = getattr(st.session_state, '_depeg', None)
    if _dc:
        if candle['direction'] == 'Bullish':
            score += 1
            reasons.append(f"DECAPPING ₹{_dc['strike']:.0f} CE OI -{_dc['shed_pct']:.1f}% → ceiling lifting (+1)")
        else:
            reasons.append(f"DECAPPING ₹{_dc['strike']:.0f} CE OI -{_dc['shed_pct']:.1f}% (ceiling weakening)")
    if _dp:
        if candle['direction'] == 'Bearish':
            score -= 1
            reasons.append(f"DEPEG ₹{_dp['strike']:.0f} PE OI -{_dp['shed_pct']:.1f}% → floor dropping (-1)")
        else:
            reasons.append(f"DEPEG ₹{_dp['strike']:.0f} PE OI -{_dp['shed_pct']:.1f}% (floor weakening)")

    # 9. VIDYA Trend (from Pine Script)
    vidya_data = calculate_vidya(df)
    if vidya_data['trend'] == 'Bullish' and candle['direction'] == 'Bullish':
        score += 1
        reasons.append(f"VIDYA Trend: Bullish (Delta: {vidya_data['delta_pct']:+.0f}%)")
    elif vidya_data['trend'] == 'Bearish' and candle['direction'] == 'Bearish':
        score -= 1
        reasons.append(f"VIDYA Trend: Bearish (Delta: {vidya_data['delta_pct']:+.0f}%)")
    elif vidya_data['trend'] != 'Unknown':
        reasons.append(f"VIDYA Trend: {vidya_data['trend']} (Divergence from candle)")
    if vidya_data.get('cross_up'):
        reasons.append("VIDYA: Fresh Bullish Cross ▲")
    elif vidya_data.get('cross_down'):
        reasons.append("VIDYA: Fresh Bearish Cross ▼")

    # 10. VOB + HTF S&R (Price Action Support/Resistance)
    vob_blocks = getattr(st.session_state, '_vob_blocks', None)
    htf_sr = calculate_htf_sr(df)
    hvp_data = detect_hvp(df)
    ltp_trap = detect_ltp_trap(df)
    # VPFR — three timeframes
    vpfr_data = {
        'short':  compute_vpfr(df, 30),
        'medium': compute_vpfr(df, 60),
        'long':   compute_vpfr(df, 180),
    }
    delta_vol_df = calculate_delta_volume(df)

    # VOB proximity check
    near_vob_support = False
    near_vob_resistance = False
    vob_support_levels = []
    vob_resistance_levels = []
    if vob_blocks:
        for b in vob_blocks.get('bullish', []):
            vob_support_levels.append(b)
            if b['lower'] <= underlying_price <= b['upper'] * 1.002:
                near_vob_support = True
        for b in vob_blocks.get('bearish', []):
            vob_resistance_levels.append(b)
            if b['lower'] * 0.998 <= underlying_price <= b['upper']:
                near_vob_resistance = True

    # HTF S&R proximity
    htf_support_near = [l for l in htf_sr.get('support', []) if abs(underlying_price - l['level']) / underlying_price * 100 < 0.2]
    htf_resistance_near = [l for l in htf_sr.get('resistance', []) if abs(underlying_price - l['level']) / underlying_price * 100 < 0.2]

    # Scoring: Price action S&R confirmation
    if near_vob_support and score > 0:
        score += 1
        reasons.append("VOB: At Bullish Volume Zone (Support)")
    elif near_vob_resistance and score < 0:
        score -= 1
        reasons.append("VOB: At Bearish Volume Zone (Resistance)")
    elif htf_support_near and score > 0:
        tfs = ', '.join(set(l['tf'] for l in htf_support_near))
        score += 1
        reasons.append(f"HTF S&R: Near Support ({tfs})")
    elif htf_resistance_near and score < 0:
        tfs = ', '.join(set(l['tf'] for l in htf_resistance_near))
        score -= 1
        reasons.append(f"HTF S&R: Near Resistance ({tfs})")
    elif htf_support_near or htf_resistance_near or near_vob_support or near_vob_resistance:
        parts = []
        if near_vob_support:
            parts.append("VOB Support")
        if near_vob_resistance:
            parts.append("VOB Resistance")
        if htf_support_near:
            parts.append(f"HTF Sup({','.join(set(l['tf'] for l in htf_support_near))})")
        if htf_resistance_near:
            parts.append(f"HTF Res({','.join(set(l['tf'] for l in htf_resistance_near))})")
        reasons.append(f"Price Action S&R: {' | '.join(parts)}")

    # LTP Trap signal
    if ltp_trap.get('buy_trap'):
        reasons.append(f"LTP Trap Buy (VWAP: ₹{ltp_trap['vwap']:.0f})")
    elif ltp_trap.get('sell_trap'):
        reasons.append(f"LTP Trap Sell (VWAP: ₹{ltp_trap['vwap']:.0f})")

    # HVP near price
    for hvp in hvp_data.get('bullish_hvp', []):
        if abs(underlying_price - hvp['price']) / underlying_price * 100 < 0.15:
            reasons.append(f"HVP Support: ₹{hvp['price']:.0f}")
            break
    for hvp in hvp_data.get('bearish_hvp', []):
        if abs(underlying_price - hvp['price']) / underlying_price * 100 < 0.15:
            reasons.append(f"HVP Resistance: ₹{hvp['price']:.0f}")
            break

    # Delta volume trend
    delta_trend = 'Neutral'
    if delta_vol_df is not None and len(delta_vol_df) > 0:
        cum_delta_last = delta_vol_df['cum_delta'].iloc[-1]
        delta_ma_last = delta_vol_df['delta_ma'].iloc[-1]
        if cum_delta_last > 0 and delta_ma_last > 0:
            delta_trend = 'Bullish'
        elif cum_delta_last < 0 and delta_ma_last < 0:
            delta_trend = 'Bearish'

    # === DETERMINE SIGNAL ===
    abs_score = abs(score)
    near_support = any('Support' in loc for loc in location)
    near_resistance = any('Resistance' in loc for loc in location)

    # Boost near_support/near_resistance from direct OC writing detection so
    # PUT CAPPING and CALL CAPPING fire even when price proximity check misses
    _adf = sa_result.get('analysis_df') if sa_result is not None else None
    if _adf is not None:
        if not near_resistance and _adf.get('Call_Capping_Confirmed', pd.Series([False])).any():
            near_resistance = True
        if not near_support and _adf.get('Put_Support_Confirmed', pd.Series([False])).any():
            near_support = True

    breakout_zones = sa_result is not None and not sa_result['breakout_zones'].empty
    breakdown_zones = sa_result is not None and not sa_result['breakdown_zones'].empty

    if breakout_zones and vol['spike'] and candle['direction'] == 'Bullish' and net_gex < 0:
        signal = '🚀 BREAKOUT'
        trade_type = 'STRONG BUY'
    elif breakdown_zones and vol['spike'] and candle['direction'] == 'Bearish' and net_gex < 0:
        signal = '💥 BREAKDOWN'
        trade_type = 'STRONG SELL'
    elif score >= 5:
        _sup_vol = near_support and _adf is not None and \
            _adf.get('Put_Support_Confirmed', pd.Series([False])).any()
        signal = ('🟩 PUT CAPPING 🔥' if _sup_vol else '🟩 PUT CAPPING') if near_support else '🟢 STRONG BUY'
        trade_type = 'STRONG BUY'
    elif score >= 3:
        _sup_vol = near_support and _adf is not None and \
            _adf.get('Put_Support_Confirmed', pd.Series([False])).any()
        signal = ('🟩 PUT CAPPING 🔥' if _sup_vol else '🟩 PUT CAPPING') if near_support else '🟢 BUY'
        trade_type = 'STRONG BUY' if near_support else 'SCALP BUY'
    elif score <= -5:
        _cap_vol = near_resistance and _adf is not None and \
            _adf.get('Call_Capping_Confirmed', pd.Series([False])).any()
        signal = ('🟥 CALL CAPPING 🔥' if _cap_vol else '🟥 CALL CAPPING') if near_resistance else '🔴 STRONG SELL'
        trade_type = 'STRONG SELL'
    elif score <= -3:
        _cap_vol = near_resistance and _adf is not None and \
            _adf.get('Call_Capping_Confirmed', pd.Series([False])).any()
        signal = ('🟥 CALL CAPPING 🔥' if _cap_vol else '🟥 CALL CAPPING') if near_resistance else '🔴 SELL'
        trade_type = 'STRONG SELL' if near_resistance else 'SCALP SELL'
    elif net_gex > 0 and abs_score < 3:
        signal = '⚖️ RANGE'
        trade_type = 'RANGE'
    else:
        signal = '⚪ NO TRADE'
        trade_type = 'NO TRADE'

    # Confidence
    confidence = min(95, max(10, 30 + abs_score * 10))
    if vol['spike']:
        confidence += 5
    if vix_dir in ['Falling', 'Rising'] and 'Aligned' in str(reasons):
        confidence += 5
    confidence = min(95, confidence)

    # Strength label
    if abs_score >= 6:
        strength = 'STRONG'
    elif abs_score >= 4:
        strength = 'MODERATE'
    else:
        strength = 'NO TRADE' if abs_score < 3 else 'WEAK'

    return {
        'candle': candle,
        'volume': vol,
        'order_blocks': ob,
        'location': location,
        'vix': vix_data,
        'alignment': alignment,
        'gex': {
            'net_gex': net_gex, 'atm_gex': atm_gex,
            'gamma_flip': gamma_flip, 'magnet': gex_magnet,
            'repeller': gex_repeller, 'market_mode': market_mode,
            'signal': gex_signal, 'interpretation': gex_interpretation,
            'above_flip': above_gamma_flip,
        },
        'pcr_gex': {'badge': pcr_gex_badge, 'signal': pcr_gex_signal},
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'signal': signal,
        'trade_type': trade_type,
        'score': score,
        'abs_score': abs_score,
        'strength': strength,
        'confidence': confidence,
        'reasons': reasons,
        'market_bias': market_bias,
        'oi_trend': oi_trend,
        'vidya': vidya_data,
        'htf_sr': htf_sr,
        'vob_blocks': vob_blocks,
        'hvp': hvp_data,
        'vpfr': vpfr_data,
        'ltp_trap': ltp_trap,
        'delta_vol_df': delta_vol_df,
        'delta_trend': delta_trend,
        'vob_support_levels': vob_support_levels,
        'vob_resistance_levels': vob_resistance_levels,
    }

def ai_analyze_telegram_message(message, kind="master"):

def compute_unwinding_summary(df_atm8):
    """Summarise CE/PE unwinding + parallel winding from ATM±5 strikes."""
    if df_atm8 is None or len(df_atm8) == 0:
        return None
    ce_unwind, pe_unwind, ce_build, pe_build = [], [], [], []
    parallel_pairs = []
    for _, row in df_atm8.iterrows():
        strike = row.get('strikePrice', 0)
        chg_ce = row.get('changeinOpenInterest_CE', 0) or 0
        chg_pe = row.get('changeinOpenInterest_PE', 0) or 0
        ce_act = 'Unwinding' if chg_ce < -1000 else 'Buildup' if chg_ce > 1000 else 'Flat'
        pe_act = 'Unwinding' if chg_pe < -1000 else 'Buildup' if chg_pe > 1000 else 'Flat'
        if ce_act == 'Unwinding':
            ce_unwind.append({'strike': strike, 'chg': chg_ce})
        elif ce_act == 'Buildup':
            ce_build.append({'strike': strike, 'chg': chg_ce})
        if pe_act == 'Unwinding':
            pe_unwind.append({'strike': strike, 'chg': chg_pe})
        elif pe_act == 'Buildup':
            pe_build.append({'strike': strike, 'chg': chg_pe})
        if ce_act == 'Unwinding' and pe_act == 'Buildup':
            parallel_pairs.append({'strike': strike, 'type': 'CE Unwind + PE Buildup', 'impact': 'Bullish Shift'})
        elif pe_act == 'Unwinding' and ce_act == 'Buildup':
            parallel_pairs.append({'strike': strike, 'type': 'PE Unwind + CE Buildup', 'impact': 'Bearish Shift'})
        elif ce_act == 'Unwinding' and pe_act == 'Unwinding':
            parallel_pairs.append({'strike': strike, 'type': 'Both Unwinding', 'impact': 'Expiry Exit'})
        elif ce_act == 'Buildup' and pe_act == 'Buildup':
            parallel_pairs.append({'strike': strike, 'type': 'Both Buildup', 'impact': 'High Activity'})
    bull_parallel = [p for p in parallel_pairs if p['impact'] == 'Bullish Shift']
    bear_parallel = [p for p in parallel_pairs if p['impact'] == 'Bearish Shift']
    if bull_parallel and len(bull_parallel) >= len(bear_parallel):
        verdict = f"⚡ BULLISH PARALLEL WINDING ({len(bull_parallel)} strikes)"
    elif bear_parallel and len(bear_parallel) > len(bull_parallel):
        verdict = f"⚡ BEARISH PARALLEL WINDING ({len(bear_parallel)} strikes)"
    elif len(ce_unwind) > len(pe_unwind) and len(ce_unwind) >= 2:
        verdict = f"🔄 CE UNWINDING dominant ({len(ce_unwind)} strikes) → Bullish"
    elif len(pe_unwind) > len(ce_unwind) and len(pe_unwind) >= 2:
        verdict = f"🔄 PE UNWINDING dominant ({len(pe_unwind)} strikes) → Bearish"
    else:
        verdict = "Balanced / No strong unwinding"
    def _top(lst, rev=False):
        if not lst:
            return "None"
        sorted_lst = sorted(lst, key=lambda x: -abs(x['chg']) if rev else x['chg'])[:3]
        return ", ".join([f"{int(l['strike'])}({l['chg']/1000:+.0f}K)" for l in sorted_lst])
    return {
        'ce_unwind_count': len(ce_unwind),
        'pe_unwind_count': len(pe_unwind),
        'ce_build_count': len(ce_build),
        'pe_build_count': len(pe_build),
        'parallel_count': len(parallel_pairs),
        'bull_parallel': len(bull_parallel),
        'bear_parallel': len(bear_parallel),
        'ce_unwind_top': _top(ce_unwind),
        'pe_unwind_top': _top(pe_unwind),
        'ce_build_top': _top(ce_build, rev=True),
        'pe_build_top': _top(pe_build, rev=True),
        'verdict': verdict,
    }

def check_pcr_sr_proximity_alert(underlying_price, proximity_pts=25):

def compute_depth_sr(df_summary, underlying_price, n=3):
    """Derive S/R levels from live CE+PE bid/ask order pressure near spot.
    Returns effective spot prices — not raw strikes — by applying a proximity
    offset that reflects where hedging pressure starts to be felt before the strike.
    Resistance: strike - offset (sellers hedge as price approaches from below)
    Support:    strike + offset (buyers defend before price reaches the strike)
    """
    if df_summary is None or df_summary.empty:
        return [], []
    levels = []
    for _, row in df_summary.iterrows():
        try:
            strike = float(row.get('Strike', 0))
            if abs(strike - underlying_price) > 400:
                continue
            ce_bid = float(row.get('bidQty_CE', 0) or 0)
            ce_ask = float(row.get('askQty_CE', 0) or 0)
            pe_bid = float(row.get('bidQty_PE', 0) or 0)
            pe_ask = float(row.get('askQty_PE', 0) or 0)
            res_score = ce_ask + pe_bid   # call sellers + put buyers = resistance
            sup_score = pe_ask + ce_bid   # put sellers + call buyers = support
            # Proximity offset: scales with distance from spot, capped at 15 pts
            dist = abs(strike - underlying_price)
            offset = min(15, int(dist * 0.15))
            # Effective price: where pressure is actually felt in the spot market
            eff_res = strike - offset   # resistance felt below the strike
            eff_sup = strike + offset   # support felt above the strike
            levels.append({'strike': strike, 'res': res_score, 'sup': sup_score,
                           'ce_ask': ce_ask, 'pe_ask': pe_ask,
                           'eff_res': eff_res, 'eff_sup': eff_sup})
        except Exception:
            continue
    resistance = sorted([x for x in levels if x['strike'] >= underlying_price],
                        key=lambda x: x['res'], reverse=True)[:n]
    support = sorted([x for x in levels if x['strike'] <= underlying_price],
                     key=lambda x: x['sup'], reverse=True)[:n]
    return resistance, support


