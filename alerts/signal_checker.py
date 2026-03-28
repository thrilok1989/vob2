import streamlit as st
import pandas as pd
import pytz
from datetime import datetime
from indicators.pivot_indicator import PivotIndicator
from alerts.telegram import send_telegram_message_sync


def check_trading_signals(df, pivot_settings, option_data, current_price, pivot_proximity=5):
    if df.empty or option_data is None or len(option_data) == 0 or not current_price:
        return

    try:
        df_json = df.to_json()
        from vob_minimal import cached_pivot_calculation
        pivots = cached_pivot_calculation(df_json, pivot_settings)
    except:
        pivots = PivotIndicator.get_all_pivots(df, pivot_settings)

    pivot_lows = [p['value'] for p in pivots if p['type'] == 'low']
    from vob_minimal import ReversalDetector
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
                try:
                    send_telegram_message_sync(message)
                    st.success(f"{'🟢' if is_bull else '🔴'} {'Bullish' if is_bull else 'Bearish'} signal notification sent!")
                except Exception as e:
                    st.error(f"Failed to send notification: {e}")
                break


def check_atm_verdict_alert(df_summary, underlying_price):
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
    try:
        send_telegram_message_sync(message)
        st.session_state.last_atm_verdict_alert = alert_key
        st.success(f"{'🟢' if is_bull else '🔴'} ATM Strong {direction.title()} alert sent for strike {atm_strike}!")
    except Exception as e:
        st.error(f"Failed to send ATM verdict alert: {e}")


def check_gex_alert(gex_data, df_summary, underlying_price):
    if gex_data is None or 'gex_history' not in st.session_state:
        return

    try:
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        gex_entry = {
            'time': current_time,
            'total_gex': gex_data['total_gex'],
            'gamma_flip': gex_data['gamma_flip_level'],
            'spot': underlying_price,
            'signal': gex_data['gex_signal']
        }
        should_add = True
        if st.session_state.gex_history:
            last_entry = st.session_state.gex_history[-1]
            time_diff = (current_time - last_entry['time']).total_seconds()
            if time_diff < 30:
                should_add = False
        if should_add:
            st.session_state.gex_history.append(gex_entry)
            if len(st.session_state.gex_history) > 100:
                st.session_state.gex_history = st.session_state.gex_history[-100:]
        if len(st.session_state.gex_history) < 2:
            return
        prev_entry = st.session_state.gex_history[-2]
        delta_gex = gex_data['total_gex'] - prev_entry['total_gex']
        gex_pct_change = abs(delta_gex / prev_entry['total_gex'] * 100) if prev_entry['total_gex'] != 0 else 0
        alert_triggered = False
        alert_type = None
        alert_message = None
        if prev_entry['total_gex'] * gex_data['total_gex'] < 0:
            alert_triggered = True
            alert_type = "GEX SIGN FLIP"
            flip_direction = "Positive → Negative" if prev_entry['total_gex'] > 0 else "Negative → Positive"
            alert_message = f"""
🔄 <b>GEX SIGN FLIP ALERT</b> 🔄

📊 <b>Gamma Exposure Flipped:</b> {flip_direction}
📍 <b>Spot Price:</b> ₹{underlying_price:.2f}

<b>Previous GEX:</b> {prev_entry['total_gex']:.2f}L ({prev_entry['signal']})
<b>Current GEX:</b> {gex_data['total_gex']:.2f}L ({gex_data['gex_signal']})
<b>ΔGEX:</b> {delta_gex:+.2f}L

<b>🎯 Market Implication:</b>
{gex_data['gex_interpretation']}

⚡ <b>ACTION:</b> {'Expect acceleration/trend moves!' if gex_data['total_gex'] < 0 else 'Expect mean reversion/pin!'}

🕐 Time: {current_time.strftime('%H:%M:%S IST')}
"""

        elif gex_pct_change > 30:
            alert_triggered = True
            alert_type = "LARGE ΔGEX"
            alert_message = f"""
⚡ <b>LARGE ΔGEX ALERT</b> ⚡

📊 <b>Gamma Exposure Changed Significantly!</b>
📍 <b>Spot Price:</b> ₹{underlying_price:.2f}

<b>Previous GEX:</b> {prev_entry['total_gex']:.2f}L
<b>Current GEX:</b> {gex_data['total_gex']:.2f}L
<b>ΔGEX:</b> {delta_gex:+.2f}L ({gex_pct_change:.1f}%)

<b>🎯 Market Regime:</b> {gex_data['gex_signal']}
{gex_data['gex_interpretation']}

🕐 Time: {current_time.strftime('%H:%M:%S IST')}
"""

        elif gex_data['gamma_flip_level'] and prev_entry.get('gamma_flip'):
            prev_above_flip = prev_entry['spot'] > prev_entry['gamma_flip']
            curr_above_flip = underlying_price > gex_data['gamma_flip_level']
            if prev_above_flip != curr_above_flip:
                alert_triggered = True
                alert_type = "GAMMA FLIP CROSSED"
                cross_direction = "Crossed ABOVE" if curr_above_flip else "Crossed BELOW"
                alert_message = f"""
🎯 <b>GAMMA FLIP LEVEL CROSSED</b> 🎯

📍 <b>Spot Price:</b> ₹{underlying_price:.2f}
📊 <b>Gamma Flip Level:</b> ₹{gex_data['gamma_flip_level']:.2f}
🔀 <b>Direction:</b> {cross_direction}

<b>Current GEX:</b> {gex_data['total_gex']:.2f}L ({gex_data['gex_signal']})

<b>🎯 Implication:</b>
{'Above flip = More pinning/mean reversion' if curr_above_flip else 'Below flip = More trending/acceleration'}

🕐 Time: {current_time.strftime('%H:%M:%S IST')}
"""

        if alert_triggered:
            alert_key = f"{alert_type}_{current_time.strftime('%Y%m%d_%H%M')}"
            if st.session_state.last_gex_alert != alert_key:
                send_telegram_message_sync(alert_message)
                st.session_state.last_gex_alert = alert_key
                st.success(f"📊 {alert_type} alert sent!")

    except Exception as e:
        pass
