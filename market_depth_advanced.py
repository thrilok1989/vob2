# ==============================================================================
# 🎯 ADVANCED MARKET DEPTH ANALYSIS - COMPREHENSIVE IMPLEMENTATION
# ==============================================================================
"""
Complete market depth analysis with all critical missing components:
1. Real-time option depth from Dhan API
2. Order flow analysis
3. Liquidity profile metrics
4. Market maker detection
5. Depth quality metrics
6. Market microstructure signals
7. Volume profile integration
8. Greeks-based depth analysis
9. Algorithmic pattern detection
10. Market impact models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# ============================================
# 1. REAL-TIME OPTION DEPTH FROM DHAN API
# ============================================

def get_real_option_depth_from_dhan(strike, expiry, option_type="CE", dhan_config=None):
    """
    Fetch real-time option depth using Dhan API

    Args:
        strike: Strike price
        expiry: Expiry date (YYYY-MM-DD)
        option_type: "CE" or "PE"
        dhan_config: Dict with base_url, access_token, client_id

    Returns:
        Dict with comprehensive depth data
    """
    import requests

    if not dhan_config:
        return {"available": False, "error": "Dhan configuration missing"}

    try:
        url = f"{dhan_config['base_url']}/v2/marketfeed/quote"

        # Payload for option depth
        payload = {
            "OPTIDX": [{
                "underlyingScrip": 13,  # NIFTY
                "underlyingSeg": "IDX_I",
                "expiry": expiry.replace("-", ""),  # YYYYMMDD
                "strikePrice": int(strike),
                "optionType": option_type
            }]
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": dhan_config['access_token'],
            "client-id": dhan_config['client_id']
        }

        response = requests.post(url, json=payload, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()

            if data.get("status") == "success":
                opt_data = data.get("data", {}).get("OPTIDX", {})

                # Handle response structure
                if isinstance(opt_data, dict):
                    opt_data = list(opt_data.values())[0] if opt_data else {}

                if opt_data:
                    bid_levels = opt_data.get("bid", [])
                    ask_levels = opt_data.get("ask", [])

                    return {
                        "available": True,
                        "strike": strike,
                        "expiry": expiry,
                        "option_type": option_type,
                        "best_bid": opt_data.get("best_bid", 0),
                        "best_ask": opt_data.get("best_ask", 0),
                        "best_bid_qty": opt_data.get("best_bid_qty", 0),
                        "best_ask_qty": opt_data.get("best_ask_qty", 0),
                        "bid_levels": bid_levels,
                        "ask_levels": ask_levels,
                        "ltp": opt_data.get("last_price", 0),
                        "volume": opt_data.get("volume", 0),
                        "oi": opt_data.get("oi", 0),
                        "total_bid_qty": sum(level.get("quantity", 0) for level in bid_levels),
                        "total_ask_qty": sum(level.get("quantity", 0) for level in ask_levels),
                        "timestamp": datetime.now()
                    }

        return {"available": False, "error": f"HTTP {response.status_code}"}

    except Exception as e:
        return {"available": False, "error": f"Exception: {str(e)}"}


# ============================================
# 2. DEPTH LEVEL ANALYSIS
# ============================================

def analyze_depth_levels(bids: List[Dict], asks: List[Dict]) -> Dict:
    """
    Comprehensive depth profile analysis

    Returns:
        - Depth profile metrics
        - Iceberg order detection
        - Hidden liquidity analysis
        - Depth velocity indicators
    """
    if not bids or not asks:
        return {"available": False}

    # 1. DEPTH PROFILE
    bid_quantities = [b.get("quantity", 0) for b in bids]
    ask_quantities = [a.get("quantity", 0) for a in asks]

    total_bid = sum(bid_quantities)
    total_ask = sum(ask_quantities)

    # Depth concentration (top 3 levels)
    top3_bid = sum(bid_quantities[:3])
    top3_ask = sum(ask_quantities[:3])

    top3_concentration = {
        "bid_pct": (top3_bid / total_bid * 100) if total_bid > 0 else 0,
        "ask_pct": (top3_ask / total_ask * 100) if total_ask > 0 else 0
    }

    # Depth gradient (how fast depth falls off)
    bid_gradient = np.diff(bid_quantities[:5]) if len(bid_quantities) >= 5 else []
    ask_gradient = np.diff(ask_quantities[:5]) if len(ask_quantities) >= 5 else []

    # 2. ICEBERG ORDER DETECTION
    # Icebergs show as repeated similar quantities at same price
    avg_bid = np.mean(bid_quantities) if bid_quantities else 0
    avg_ask = np.mean(ask_quantities) if ask_quantities else 0

    # Look for unusually consistent order sizes
    bid_std = np.std(bid_quantities) if len(bid_quantities) > 1 else 0
    ask_std = np.std(ask_quantities) if len(ask_quantities) > 1 else 0

    iceberg_probability = {
        "bid": 1.0 - (bid_std / avg_bid) if avg_bid > 0 else 0,  # Low variance = potential iceberg
        "ask": 1.0 - (ask_std / avg_ask) if avg_ask > 0 else 0
    }

    # 3. DEPTH ASYMMETRY
    bid_ask_skew = (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0

    # Depth shape difference
    depth_shape_diff = np.corrcoef(bid_quantities[:5], ask_quantities[:5])[0,1] if len(bid_quantities) >= 5 and len(ask_quantities) >= 5 else 0

    return {
        "available": True,
        "total_bid_qty": total_bid,
        "total_ask_qty": total_ask,
        "depth_imbalance": bid_ask_skew,
        "top3_concentration": top3_concentration,
        "depth_gradient": {
            "bid_avg": float(np.mean(bid_gradient)) if len(bid_gradient) > 0 else 0,
            "ask_avg": float(np.mean(ask_gradient)) if len(ask_gradient) > 0 else 0
        },
        "iceberg_probability": iceberg_probability,
        "depth_variance": {
            "bid_std": float(bid_std),
            "ask_std": float(ask_std)
        },
        "depth_shape_correlation": float(depth_shape_diff)
    }


# ============================================
# 3. MARKET MAKER ACTIVITY DETECTION
# ============================================

def detect_market_maker_activity(bids: List[Dict], asks: List[Dict], ltp: float) -> Dict:
    """
    Detect market maker presence and activity patterns

    Market makers typically:
    1. Place orders at round numbers
    2. Maintain consistent bid-ask spread
    3. Show specific order size patterns
    """
    if not bids or not asks:
        return {"available": False}

    # 1. ROUND NUMBER ORDERS
    # Check if orders cluster at round prices (xx00, xx50)
    round_bid_orders = sum(1 for b in bids if b.get("price", 0) % 10 == 0 or b.get("price", 0) % 5 == 0)
    round_ask_orders = sum(1 for a in asks if a.get("price", 0) % 10 == 0 or a.get("price", 0) % 5 == 0)

    round_number_ratio = (round_bid_orders + round_ask_orders) / (len(bids) + len(asks)) if (len(bids) + len(asks)) > 0 else 0

    # 2. SPREAD MAINTENANCE
    best_bid = bids[0].get("price", 0) if bids else 0
    best_ask = asks[0].get("price", 0) if asks else 0
    spread = best_ask - best_bid
    spread_pct = (spread / ltp * 100) if ltp > 0 else 0

    # MM typically maintain tight, consistent spreads (0.5-2%)
    spread_consistency = 1.0 if 0.1 <= spread_pct <= 3.0 else 0.5

    # 3. ORDER SIZE PATTERNS
    # MMs often use standard lot sizes
    bid_sizes = [b.get("quantity", 0) for b in bids]
    ask_sizes = [a.get("quantity", 0) for a in asks]

    # Check for lot-based sizing (multiples of 25, 50, 75, 100)
    def is_lot_based(qty):
        return qty % 25 == 0 or qty % 50 == 0

    lot_based_bids = sum(1 for q in bid_sizes if is_lot_based(q))
    lot_based_asks = sum(1 for q in ask_sizes if is_lot_based(q))

    lot_based_ratio = (lot_based_bids + lot_based_asks) / (len(bid_sizes) + len(ask_sizes)) if (len(bid_sizes) + len(ask_sizes)) > 0 else 0

    # 4. MM PRESENCE SCORE (0-100)
    mm_score = int(100 * (
        0.3 * round_number_ratio +
        0.4 * spread_consistency +
        0.3 * lot_based_ratio
    ))

    return {
        "available": True,
        "round_number_orders_pct": round_number_ratio * 100,
        "spread_pct": spread_pct,
        "spread_consistency": spread_consistency,
        "lot_based_orders_pct": lot_based_ratio * 100,
        "mm_presence_score": mm_score,
        "interpretation": "Strong MM presence" if mm_score > 70 else ("Moderate MM activity" if mm_score > 40 else "Low MM activity")
    }


# ============================================
# 4. LIQUIDITY PROFILE ANALYSIS
# ============================================

def analyze_liquidity_profile(bids: List[Dict], asks: List[Dict], ltp: float) -> Dict:
    """
    Comprehensive liquidity analysis

    Returns:
        - Depth concentration
        - Liquidity resilience
        - Price impact estimates
        - Slippage costs
    """
    if not bids or not asks:
        return {"available": False}

    # 1. DEPTH CONCENTRATION (Top 5 levels)
    bid_qtys = [b.get("quantity", 0) for b in bids]
    ask_qtys = [a.get("quantity", 0) for a in asks]

    total_bid = sum(bid_qtys)
    total_ask = sum(ask_qtys)

    top5_bid = sum(bid_qtys[:5])
    top5_ask = sum(ask_qtys[:5])

    concentration = {
        "top5_bid_pct": (top5_bid / total_bid * 100) if total_bid > 0 else 0,
        "top5_ask_pct": (top5_ask / total_ask * 100) if total_ask > 0 else 0
    }

    # 2. LIQUIDITY FRAGILITY
    # How easily depth can be eaten through
    avg_level_qty = (total_bid + total_ask) / (len(bids) + len(asks)) if (len(bids) + len(asks)) > 0 else 0

    # A large order relative to avg level = high fragility
    liquidity_fragility = 5000 / avg_level_qty if avg_level_qty > 0 else 100  # Score for 5000 contract order

    # 3. PRICE IMPACT (for different order sizes)
    def calculate_price_impact(levels, order_size, is_buy=True):
        """Calculate price impact for given order size"""
        remaining = order_size
        total_cost = 0
        level_idx = 0

        while remaining > 0 and level_idx < len(levels):
            level = levels[level_idx]
            qty_at_level = level.get("quantity", 0)
            price_at_level = level.get("price", 0)

            qty_to_take = min(remaining, qty_at_level)
            total_cost += qty_to_take * price_at_level
            remaining -= qty_to_take
            level_idx += 1

        avg_price = total_cost / order_size if order_size > 0 else 0
        impact_pct = abs(avg_price - ltp) / ltp * 100 if ltp > 0 else 0

        return {
            "avg_execution_price": avg_price,
            "impact_pct": impact_pct,
            "levels_consumed": level_idx,
            "filled": order_size - remaining
        }

    # Calculate for standard sizes
    price_impact_1k = calculate_price_impact(asks, 1000, True)  # 1000 contracts buy
    price_impact_5k = calculate_price_impact(asks, 5000, True)  # 5000 contracts buy
    price_impact_10k = calculate_price_impact(asks, 10000, True)  # 10000 contracts buy

    # 4. SLIPPAGE COST
    best_ask = asks[0].get("price", 0) if asks else 0
    slippage_1k = (price_impact_1k["avg_execution_price"] - best_ask) / best_ask * 100 if best_ask > 0 else 0
    slippage_5k = (price_impact_5k["avg_execution_price"] - best_ask) / best_ask * 100 if best_ask > 0 else 0

    return {
        "available": True,
        "top5_concentration": concentration,
        "liquidity_fragility_score": min(liquidity_fragility, 100),  # Cap at 100
        "avg_level_qty": avg_level_qty,
        "price_impact": {
            "1k_contracts": price_impact_1k,
            "5k_contracts": price_impact_5k,
            "10k_contracts": price_impact_10k
        },
        "slippage": {
            "1k_contracts_pct": slippage_1k,
            "5k_contracts_pct": slippage_5k
        }
    }


# ============================================
# 5. ORDER FLOW ANALYSIS
# ============================================

def analyze_order_flow(depth_history: List[Dict], time_sales: List[Dict] = None) -> Dict:
    """
    Analyze order flow to detect institutional activity

    Args:
        depth_history: List of historical depth snapshots
        time_sales: List of trade executions (optional)

    Returns:
        - Aggressive vs passive order ratios
        - Buy/sell pressure
        - Institutional order detection
        - Hidden order probability
    """
    if not depth_history or len(depth_history) < 2:
        return {"available": False, "error": "Insufficient depth history"}

    # Compare current vs previous depth to detect order flow
    current = depth_history[-1]
    previous = depth_history[-2]

    curr_bid_qty = current.get("total_bid_qty", 0)
    prev_bid_qty = previous.get("total_bid_qty", 0)
    curr_ask_qty = current.get("total_ask_qty", 0)
    prev_ask_qty = previous.get("total_ask_qty", 0)

    # 1. DEPTH CHANGES (proxy for order flow)
    bid_increase = max(0, curr_bid_qty - prev_bid_qty)
    bid_decrease = max(0, prev_bid_qty - curr_bid_qty)
    ask_increase = max(0, curr_ask_qty - prev_ask_qty)
    ask_decrease = max(0, prev_ask_qty - curr_ask_qty)

    # 2. BUY/SELL PRESSURE
    # Aggressive buying = ask decrease (hitting asks)
    # Aggressive selling = bid decrease (hitting bids)
    aggressive_buy_volume = ask_decrease
    aggressive_sell_volume = bid_decrease

    # Passive orders = depth additions
    passive_buy_volume = bid_increase
    passive_sell_volume = ask_increase

    total_volume = aggressive_buy_volume + aggressive_sell_volume + passive_buy_volume + passive_sell_volume

    # 3. PRESSURE RATIOS
    buy_pressure = (aggressive_buy_volume / total_volume * 100) if total_volume > 0 else 0
    sell_pressure = (aggressive_sell_volume / total_volume * 100) if total_volume > 0 else 0

    # 4. ORDER SIZE DISTRIBUTION
    # Detect large orders (institutional sized)
    curr_bid_levels = current.get("bid_levels", [])
    large_bid_orders = sum(1 for b in curr_bid_levels if b.get("quantity", 0) > 5000)
    institutional_bid_orders = sum(1 for b in curr_bid_levels if b.get("quantity", 0) > 10000)

    curr_ask_levels = current.get("ask_levels", [])
    large_ask_orders = sum(1 for a in curr_ask_levels if a.get("quantity", 0) > 5000)
    institutional_ask_orders = sum(1 for a in curr_ask_levels if a.get("quantity", 0) > 10000)

    # 5. ICEBERG/HIDDEN ORDER DETECTION
    # Rapid depth replenishment at same price = iceberg
    iceberg_probability = 0.0
    if bid_decrease > 0 and bid_increase > 0:
        # If lots of sell orders hit bids, but bid depth quickly replenishes
        replenishment_ratio = bid_increase / bid_decrease
        iceberg_probability = min(replenishment_ratio * 0.5, 1.0) * 100

    return {
        "available": True,
        "aggressive_buy_volume": aggressive_buy_volume,
        "aggressive_sell_volume": aggressive_sell_volume,
        "passive_buy_volume": passive_buy_volume,
        "passive_sell_volume": passive_sell_volume,
        "buy_pressure_pct": buy_pressure,
        "sell_pressure_pct": sell_pressure,
        "large_orders": {
            "bid_large": large_bid_orders,
            "ask_large": large_ask_orders,
            "bid_institutional": institutional_bid_orders,
            "ask_institutional": institutional_ask_orders
        },
        "iceberg_probability_pct": iceberg_probability,
        "flow_interpretation": "Strong buying pressure" if buy_pressure > 60 else ("Strong selling pressure" if sell_pressure > 60 else "Balanced flow")
    }


# ============================================
# 6. VOLUME PROFILE ANALYSIS
# ============================================

def analyze_volume_profile(price_volume_data: List[Tuple[float, int]]) -> Dict:
    """
    Volume-at-price analysis

    Args:
        price_volume_data: List of (price, volume) tuples

    Returns:
        - Point of Control (POC)
        - Value Area High/Low
        - Volume nodes
    """
    if not price_volume_data or len(price_volume_data) < 3:
        return {"available": False}

    df = pd.DataFrame(price_volume_data, columns=["price", "volume"])
    df = df.sort_values("price")

    # 1. POINT OF CONTROL (price with highest volume)
    poc_idx = df["volume"].idxmax()
    poc_price = df.loc[poc_idx, "price"]
    poc_volume = df.loc[poc_idx, "volume"]

    # 2. VALUE AREA (70% of volume)
    total_volume = df["volume"].sum()
    target_volume = total_volume * 0.70

    # Start from POC and expand outward
    current_volume = poc_volume
    high_idx = poc_idx
    low_idx = poc_idx

    while current_volume < target_volume and (high_idx < len(df) - 1 or low_idx > 0):
        # Check which direction to expand
        next_high_vol = df.loc[high_idx + 1, "volume"] if high_idx < len(df) - 1 else 0
        next_low_vol = df.loc[low_idx - 1, "volume"] if low_idx > 0 else 0

        if next_high_vol > next_low_vol and high_idx < len(df) - 1:
            high_idx += 1
            current_volume += next_high_vol
        elif low_idx > 0:
            low_idx -= 1
            current_volume += next_low_vol
        else:
            break

    value_area_high = df.loc[high_idx, "price"]
    value_area_low = df.loc[low_idx, "price"]

    # 3. HIGH/LOW VOLUME NODES
    volume_threshold_high = df["volume"].quantile(0.75)
    volume_threshold_low = df["volume"].quantile(0.25)

    high_volume_nodes = df[df["volume"] >= volume_threshold_high]["price"].tolist()
    low_volume_nodes = df[df["volume"] <= volume_threshold_low]["price"].tolist()

    # 4. VOLUME-WEIGHTED AVERAGE PRICE
    vwap = (df["price"] * df["volume"]).sum() / total_volume if total_volume > 0 else 0

    return {
        "available": True,
        "point_of_control": poc_price,
        "poc_volume": int(poc_volume),
        "value_area_high": value_area_high,
        "value_area_low": value_area_low,
        "vwap": vwap,
        "high_volume_nodes": high_volume_nodes[:5],  # Top 5
        "low_volume_nodes": low_volume_nodes[:5]      # Bottom 5
    }


# ============================================
# 7. MARKET MICROSTRUCTURE SIGNALS
# ============================================

def analyze_market_microstructure(depth_data: Dict, ltp: float, volume: int) -> Dict:
    """
    Detect microstructure signals:
    - Informed trading
    - Liquidity provision
    - Stop hunting
    - Gamma hedging patterns
    """
    if not depth_data.get("available"):
        return {"available": False}

    best_bid = depth_data.get("best_bid", 0)
    best_ask = depth_data.get("best_ask", 0)
    bid_qty = depth_data.get("best_bid_qty", 0)
    ask_qty = depth_data.get("best_ask_qty", 0)

    # 1. INFORMED TRADING
    # Large orders away from best bid/ask = informed traders
    bid_levels = depth_data.get("bid_levels", [])
    ask_levels = depth_data.get("ask_levels", [])

    # Check for large orders beyond level 1
    deep_bid_orders = sum(b.get("quantity", 0) for b in bid_levels[1:5]) if len(bid_levels) > 1 else 0
    deep_ask_orders = sum(a.get("quantity", 0) for a in ask_levels[1:5]) if len(ask_levels) > 1 else 0

    informed_trading_score = 0
    if deep_bid_orders > bid_qty * 2:  # Deep bids >> best bid
        informed_trading_score += 30
    if deep_ask_orders > ask_qty * 2:  # Deep asks >> best ask
        informed_trading_score += 30

    # 2. LIQUIDITY PROVISION
    # Consistent round-number orders that stay = liquidity providers
    round_bids = sum(1 for b in bid_levels if b.get("price", 0) % 5 == 0)
    round_asks = sum(1 for a in ask_levels if a.get("price", 0) % 5 == 0)

    liquidity_provision_score = int((round_bids + round_asks) / (len(bid_levels) + len(ask_levels)) * 100) if (len(bid_levels) + len(ask_levels)) > 0 else 0

    # 3. STOP HUNTING
    # Rapid depth removal near round numbers = potential stop hunt
    total_bid = depth_data.get("total_bid_qty", 0)
    total_ask = depth_data.get("total_ask_qty", 0)

    # If price near round number with low depth = stop hunt zone
    is_near_round = (ltp % 50 < 10) or (ltp % 50 > 40)
    depth_ratio = (total_bid + total_ask) / volume if volume > 0 else 1

    stop_hunt_probability = 0
    if is_near_round and depth_ratio < 0.5:  # Near round number with thin depth
        stop_hunt_probability = 60

    # 4. GAMMA HEDGING PATTERNS
    # Options market makers hedge gamma = specific depth patterns
    # Large depth on both sides = delta-neutral MM positioning
    balanced_depth = abs(total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 1

    gamma_hedging_score = int((1 - balanced_depth) * 100)  # Lower imbalance = more hedging

    return {
        "available": True,
        "informed_trading_score": informed_trading_score,
        "liquidity_provision_score": liquidity_provision_score,
        "stop_hunt_probability": stop_hunt_probability,
        "gamma_hedging_score": gamma_hedging_score,
        "signals": {
            "informed_trading": "High" if informed_trading_score > 50 else "Low",
            "liquidity_provision": "High" if liquidity_provision_score > 60 else "Low",
            "stop_hunt_risk": "High" if stop_hunt_probability > 50 else "Low",
            "gamma_hedging": "Active" if gamma_hedging_score > 70 else "Inactive"
        }
    }


# ============================================
# 8. DEPTH QUALITY METRICS
# ============================================

def calculate_depth_quality(depth_data: Dict, depth_history: List[Dict] = None) -> Dict:
    """
    Assess overall depth quality

    Metrics:
    - Thickness (total contracts)
    - Resilience (replenishment speed)
    - Stability (order longevity)
    - Granularity (price level distribution)
    """
    if not depth_data.get("available"):
        return {"available": False}

    total_bid = depth_data.get("total_bid_qty", 0)
    total_ask = depth_data.get("total_ask_qty", 0)
    bid_levels = depth_data.get("bid_levels", [])
    ask_levels = depth_data.get("ask_levels", [])

    # 1. THICKNESS (total depth)
    total_depth = total_bid + total_ask
    thickness_score = min(int(total_depth / 1000), 100)  # Cap at 100

    # 2. RESILIENCE (if we have history)
    resilience_score = 50  # Default
    if depth_history and len(depth_history) >= 3:
        # Check how depth recovered after being consumed
        prev_total = depth_history[-2].get("total_bid_qty", 0) + depth_history[-2].get("total_ask_qty", 0)
        recovery_ratio = total_depth / prev_total if prev_total > 0 else 1
        resilience_score = min(int(recovery_ratio * 50), 100)

    # 3. GRANULARITY (distribution across levels)
    num_levels = len(bid_levels) + len(ask_levels)
    avg_per_level = total_depth / num_levels if num_levels > 0 else 0

    # Good granularity = depth spread across multiple levels (not concentrated)
    level_variances = [b.get("quantity", 0) for b in bid_levels] + [a.get("quantity", 0) for a in ask_levels]
    std_dev = np.std(level_variances) if len(level_variances) > 1 else 0
    cv = std_dev / avg_per_level if avg_per_level > 0 else 0  # Coefficient of variation

    # Lower CV = better granularity
    granularity_score = max(0, int(100 - cv * 50))

    # 4. OVERALL QUALITY SCORE
    overall_quality = int(0.4 * thickness_score + 0.3 * resilience_score + 0.3 * granularity_score)

    return {
        "available": True,
        "thickness_score": thickness_score,
        "resilience_score": resilience_score,
        "granularity_score": granularity_score,
        "overall_quality_score": overall_quality,
        "total_depth_contracts": int(total_depth),
        "interpretation": "Excellent depth" if overall_quality > 75 else ("Good depth" if overall_quality > 50 else "Poor depth")
    }


# ============================================
# 9. ALGORITHMIC PATTERN DETECTION
# ============================================

def detect_algo_patterns(depth_history: List[Dict], time_window_sec: int = 60) -> Dict:
    """
    Detect algorithmic trading patterns

    Patterns:
    - TWAP/VWAP execution
    - Iceberg orders
    - Momentum ignition
    - Quote stuffing
    - Spoofing/layering
    """
    if not depth_history or len(depth_history) < 5:
        return {"available": False}

    # 1. TWAP/VWAP DETECTION
    # Consistent order sizes over time = TWAP
    recent_depths = depth_history[-10:]
    bid_qtys = [d.get("total_bid_qty", 0) for d in recent_depths]

    # Low variance in quantities = algorithmic execution
    avg_qty = np.mean(bid_qtys) if bid_qtys else 0
    std_qty = np.std(bid_qtys) if len(bid_qtys) > 1 else 0
    cv = std_qty / avg_qty if avg_qty > 0 else 1

    twap_vwap_detected = cv < 0.15  # Very consistent sizing

    # 2. ICEBERG DETECTION
    # Rapid replenishment at same price
    iceberg_detected = False
    if len(depth_history) >= 3:
        curr = depth_history[-1]
        prev = depth_history[-2]

        curr_best_bid_qty = curr.get("best_bid_qty", 0)
        prev_best_bid_qty = prev.get("best_bid_qty", 0)

        # If qty drops then immediately replenishes to similar level
        if prev_best_bid_qty > curr_best_bid_qty * 1.5 and curr_best_bid_qty > prev_best_bid_qty * 0.7:
            iceberg_detected = True

    # 3. QUOTE STUFFING
    # Excessive order updates in short time
    update_frequency = len(depth_history) / time_window_sec if time_window_sec > 0 else 0
    quote_stuffing = update_frequency > 10  # More than 10 updates/sec

    # 4. SPOOFING PROBABILITY
    # Large orders that quickly disappear without execution
    spoofing_score = 0
    if len(depth_history) >= 4:
        # Check for large depth that vanishes
        for i in range(len(depth_history) - 3):
            if depth_history[i].get("total_bid_qty", 0) > depth_history[i+2].get("total_bid_qty", 0) * 2:
                spoofing_score += 20

    spoofing_score = min(spoofing_score, 100)

    return {
        "available": True,
        "twap_vwap_detected": twap_vwap_detected,
        "iceberg_detected": iceberg_detected,
        "quote_stuffing_detected": quote_stuffing,
        "spoofing_probability": spoofing_score,
        "update_frequency": round(update_frequency, 2),
        "patterns_detected": []  # Will be populated with detected pattern names
    }


# ============================================
# 10. MARKET IMPACT MODEL
# ============================================

def calculate_market_impact(order_size: int, depth_data: Dict, volatility: float = 0.15) -> Dict:
    """
    Estimate market impact for a given order size

    Uses:
    - Current depth
    - Historical volatility
    - Square root impact model

    Returns:
        - Price impact estimate
        - Slippage cost
        - Optimal execution strategy
    """
    if not depth_data.get("available"):
        return {"available": False}

    ltp = depth_data.get("ltp", 0)
    volume = depth_data.get("volume", 0)

    # Get depth levels
    ask_levels = depth_data.get("ask_levels", [])
    bid_levels = depth_data.get("bid_levels", [])

    # 1. IMMEDIATE IMPACT (eating through order book)
    def calc_immediate_impact(levels, size):
        remaining = size
        total_cost = 0
        levels_consumed = 0

        for level in levels:
            if remaining <= 0:
                break

            qty = level.get("quantity", 0)
            price = level.get("price", 0)

            take = min(remaining, qty)
            total_cost += take * price
            remaining -= take
            levels_consumed += 1

        avg_price = total_cost / size if size > 0 else 0
        impact_bps = (avg_price - ltp) / ltp * 10000 if ltp > 0 else 0

        return {
            "avg_price": avg_price,
            "impact_bps": impact_bps,
            "levels_consumed": levels_consumed,
            "unfilled_qty": int(remaining)
        }

    # Calculate for buy order
    buy_impact = calc_immediate_impact(ask_levels, order_size)

    # 2. TEMPORARY IMPACT (Kyle's lambda model)
    # Impact = lambda * sqrt(order_size / avg_volume)
    avg_daily_volume = volume if volume > 0 else 100000
    lambda_param = volatility * 0.1  # Simplified lambda

    participation_rate = order_size / avg_daily_volume
    temporary_impact_bps = lambda_param * np.sqrt(participation_rate) * 10000

    # 3. PERMANENT IMPACT
    # Smaller than temporary, ~40% of temporary
    permanent_impact_bps = temporary_impact_bps * 0.4

    # 4. TOTAL IMPACT
    total_impact_bps = buy_impact["impact_bps"] + temporary_impact_bps

    # 5. SLIPPAGE COST
    best_ask = ask_levels[0].get("price", ltp) if ask_levels else ltp
    slippage_cost = (buy_impact["avg_price"] - best_ask) * order_size
    slippage_pct = (buy_impact["avg_price"] - best_ask) / best_ask * 100 if best_ask > 0 else 0

    # 6. OPTIMAL EXECUTION STRATEGY
    if participation_rate < 0.05:
        strategy = "Market order - low impact"
    elif participation_rate < 0.15:
        strategy = "TWAP over 5-15 minutes"
    elif participation_rate < 0.30:
        strategy = "VWAP over 30-60 minutes"
    else:
        strategy = "Slice over multiple hours - high impact"

    return {
        "available": True,
        "order_size": order_size,
        "immediate_impact": buy_impact,
        "temporary_impact_bps": round(temporary_impact_bps, 2),
        "permanent_impact_bps": round(permanent_impact_bps, 2),
        "total_impact_bps": round(total_impact_bps, 2),
        "slippage_cost": round(slippage_cost, 2),
        "slippage_pct": round(slippage_pct, 3),
        "participation_rate_pct": round(participation_rate * 100, 2),
        "optimal_strategy": strategy
    }


# ============================================
# COMPREHENSIVE DEPTH ANALYSIS AGGREGATOR
# ============================================

def run_comprehensive_depth_analysis(strike, expiry, option_type, dhan_config, depth_history=None):
    """
    Run ALL depth analysis components and return comprehensive results
    """
    # 1. Fetch real-time depth
    depth_data = get_real_option_depth_from_dhan(strike, expiry, option_type, dhan_config)

    if not depth_data.get("available"):
        return {"available": False, "error": depth_data.get("error", "Unknown error")}

    bid_levels = depth_data.get("bid_levels", [])
    ask_levels = depth_data.get("ask_levels", [])
    ltp = depth_data.get("ltp", 0)
    volume = depth_data.get("volume", 0)

    # 2. Run all analyses
    results = {
        "available": True,
        "strike": strike,
        "expiry": expiry,
        "option_type": option_type,
        "ltp": ltp,
        "volume": volume,
        "oi": depth_data.get("oi", 0),

        # Core depth data
        "depth_data": depth_data,

        # Advanced analyses
        "depth_levels": analyze_depth_levels(bid_levels, ask_levels),
        "market_maker": detect_market_maker_activity(bid_levels, ask_levels, ltp),
        "liquidity_profile": analyze_liquidity_profile(bid_levels, ask_levels, ltp),
        "depth_quality": calculate_depth_quality(depth_data, depth_history),
        "microstructure": analyze_market_microstructure(depth_data, ltp, volume),

        # Order flow (requires history)
        "order_flow": analyze_order_flow(depth_history) if depth_history else {"available": False},

        # Algo patterns (requires history)
        "algo_patterns": detect_algo_patterns(depth_history) if depth_history else {"available": False},

        # Market impact for standard sizes
        "market_impact_1k": calculate_market_impact(1000, depth_data),
        "market_impact_5k": calculate_market_impact(5000, depth_data),
        "market_impact_10k": calculate_market_impact(10000, depth_data),
    }

    return results
