import pandas as pd
import numpy as np
import streamlit as st


def calculate_dealer_gex(df_summary, spot_price, contract_multiplier=25):
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
            call_gex = -1 * gamma_ce * oi_ce * contract_multiplier * spot_price / 100000
            put_gex = gamma_pe * oi_pe * contract_multiplier * spot_price / 100000
            net_gex = call_gex + put_gex
            gex_data.append({
                'Strike': strike,
                'Call_GEX': round(call_gex, 2),
                'Put_GEX': round(put_gex, 2),
                'Net_GEX': round(net_gex, 2),
                'Zone': row.get('Zone', '-')
            })
        gex_df = pd.DataFrame(gex_data)
        total_gex = gex_df['Net_GEX'].sum()
        gex_df_sorted = gex_df.sort_values('Strike')
        gamma_flip_level = None
        gamma_flip_direction = None
        for i in range(len(gex_df_sorted) - 1):
            current_gex = gex_df_sorted.iloc[i]['Net_GEX']
            next_gex = gex_df_sorted.iloc[i + 1]['Net_GEX']
            current_strike = gex_df_sorted.iloc[i]['Strike']
            next_strike = gex_df_sorted.iloc[i + 1]['Strike']
            if current_gex * next_gex < 0:
                gamma_flip_level = current_strike + (next_strike - current_strike) * abs(current_gex) / (abs(current_gex) + abs(next_gex))
                gamma_flip_direction = "Positive above" if current_gex < 0 else "Negative above"
                break
        if total_gex > 50:
            gex_interpretation = "STRONG PIN - Dealers long gamma, price likely to revert/chop"
            gex_signal = "Pin/Chop"
            gex_color = "#00ff88"
        elif total_gex > 0:
            gex_interpretation = "MILD PIN - Slight mean reversion tendency"
            gex_signal = "Range"
            gex_color = "#90EE90"
        elif total_gex > -50:
            gex_interpretation = "MILD TREND - Slight directional bias possible"
            gex_signal = "Trending"
            gex_color = "#FFD700"
        else:
            gex_interpretation = "STRONG TREND - Dealers short gamma, violent moves possible"
            gex_signal = "Breakout"
            gex_color = "#ff4444"
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


def calculate_pcr_gex_confluence(pcr_value, gex_data, zone='ATM'):
    if gex_data is None:
        return "⚪ N/A", "No GEX Data", 0

    net_gex = gex_data.get('total_gex', 0)
    gex_signal = gex_data.get('gex_signal', 'Unknown')

    if pcr_value > 1.2:
        pcr_signal = "Bullish"
    elif pcr_value < 0.7:
        pcr_signal = "Bearish"
    else:
        pcr_signal = "Neutral"

    gex_negative = net_gex < -10
    gex_positive = net_gex > 10

    if pcr_signal == "Bullish" and gex_negative:
        return "🟢🔥 STRONG BULL", "Bullish + Breakout", 3

    elif pcr_signal == "Bearish" and gex_positive:
        return "🔴🔥 STRONG BEAR", "Bearish + Pin", 3

    elif pcr_signal == "Bullish" and gex_positive:
        return "🟢📍 BULL RANGE", "Bullish + Chop", 2

    elif pcr_signal == "Bearish" and gex_negative:
        return "🔴⚡ BEAR TREND", "Bearish + Accel", 2

    elif pcr_signal == "Bullish":
        return "🟢 BULLISH", "Bullish PCR", 1

    elif pcr_signal == "Bearish":
        return "🔴 BEARISH", "Bearish PCR", 1

    else:
        return "⚪ NEUTRAL", "Mixed Signals", 0
