"""Lean extraction of the Market Depth Analyzer from seller_perspective.py.

This module exposes only the Market Depth Analyzer (Order Book) feature
so the main app can render it as a sub-tab without loading the full
seller_perspective.py module on every Streamlit rerun.

All analytical functions below are copied verbatim from
seller_perspective.py. The render_market_depth_tab function at the end
wires them up using data already available in the main analyzer.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import pytz

DHAN_BASE_URL = "https://api.dhan.co"

try:
    from market_depth_advanced import (
        run_comprehensive_depth_analysis,
    )
    ADVANCED_DEPTH_AVAILABLE = True
except ImportError:
    ADVANCED_DEPTH_AVAILABLE = False


def _get_dhan_credentials():
    """Read Dhan credentials from Streamlit secrets, matching vob_minimal's pattern."""
    try:
        cid = st.secrets.get("DHAN_CLIENT_ID", "") or st.secrets.get("dhan", {}).get("client_id", "")
        tok = st.secrets.get("DHAN_ACCESS_TOKEN", "") or st.secrets.get("dhan", {}).get("access_token", "")
        return cid, tok
    except Exception:
        return "", ""


DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN = _get_dhan_credentials()


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
    # Read spot from main analyzer's session_state cache; fall back to a default.
    spot_price = (
        st.session_state.get('_last_underlying')
        or (st.session_state.get('_cached_option_data') or {}).get('underlying')
        or 0
    )
    try:
        spot_price = float(spot_price)
    except (TypeError, ValueError):
        spot_price = 0
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




def render_market_depth_tab(spot, atm_strike, expiry="", strike_gap=50):
    """Render the Market Depth Analyzer tab using Dhan's 5-level depth feed.

    Reuses spot/atm/expiry already available in the main analyzer.
    """
    if not spot:
        st.info("Market depth requires a live spot price; analyzer is still warming up.")
        return

    # Fetch market depth from Dhan REST API (5-level depth)
    depth_data = get_market_depth_dhan()

    # Store market depth in session state for SL Hunt Detector reuse
    st.session_state['market_depth_data'] = depth_data

    # Analyze depth (5 levels from Dhan API)
    depth_analysis = analyze_market_depth(depth_data, spot, levels=5)

    # Generate depth-based signals
    depth_signals = calculate_depth_based_signals(depth_analysis, spot)

    # Enhanced orderbook pressure with depth
    if depth_analysis.get("available"):
        depth_enhanced_pressure = enhanced_orderbook_pressure(depth_analysis, spot)
    else:
        depth_enhanced_pressure = {"available": False}

    display_market_depth_dashboard(spot, depth_analysis, depth_signals, depth_enhanced_pressure)

    # Display Comprehensive Advanced Depth Analysis
    if ADVANCED_DEPTH_AVAILABLE:
        st.markdown("---")
        st.markdown("## 🎯 ADVANCED DEPTH ANALYSIS (ATM ±2 Strikes)")

        cid, tok = _get_dhan_credentials()
        dhan_config = {
            "base_url": DHAN_BASE_URL,
            "access_token": tok,
            "client_id": cid,
        }

        atm_strikes_to_analyze = [atm_strike + (i * strike_gap) for i in range(-2, 3)]

        st.info("⏱️ Fetching depth data with rate limiting (1 req/sec) - This may take ~10 seconds...")

        for idx, strike in enumerate(atm_strikes_to_analyze):
            with st.expander(f"📊 Strike {strike} - Comprehensive Depth Analysis", expanded=(idx == 2)):
                st.markdown(f"### Analyzing {strike} CE & PE")

                if idx > 0:
                    time.sleep(1.2)

                st.markdown("#### 📈 CALL Option (CE)")
                ce_analysis = run_comprehensive_depth_analysis(
                    strike=strike,
                    expiry=expiry,
                    option_type="CE",
                    dhan_config=dhan_config,
                    depth_history=None,
                )

                if ce_analysis.get("available"):
                    display_comprehensive_depth_analysis(ce_analysis)
                else:
                    st.warning(f"CE analysis unavailable: {ce_analysis.get('error', 'Unknown error')}")

                st.markdown("---")

                time.sleep(1.2)

                st.markdown("#### 📉 PUT Option (PE)")
                pe_analysis = run_comprehensive_depth_analysis(
                    strike=strike,
                    expiry=expiry,
                    option_type="PE",
                    dhan_config=dhan_config,
                    depth_history=None,
                )

                if pe_analysis.get("available"):
                    display_comprehensive_depth_analysis(pe_analysis)
                else:
                    st.warning(f"PE analysis unavailable: {pe_analysis.get('error', 'Unknown error')}")
    else:
        st.info("ℹ️ Advanced depth analysis module not available. Install `market_depth_advanced.py` for full functionality.")
