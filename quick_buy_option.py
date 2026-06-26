"""Quick Buy Option — standalone Streamlit app.

A self-contained one-screen option order ticket for NIFTY, extracted as an
independent app. Pick strike / type / lots, see live LTP, quantity, estimated
cost and IV/Delta from the Dhan option chain, set a limit price and place the
order through the Dhan API.

Run with:  streamlit run quick_buy_option.py

Credentials are read from Streamlit secrets (same layout as the main app):
    DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN   (or [dhan] client_id / access_token)
"""
import time
from datetime import datetime

import pandas as pd
import pytz
import requests
import streamlit as st

st.set_page_config(page_title="Quick Buy Option", page_icon="⚡", layout="centered")

DHAN_BASE_URL = "https://api.dhan.co/v2"
NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DEFAULT_LOT_SIZE = 75  # NIFTY lot size (override in the form if it changes)
IST = pytz.timezone("Asia/Kolkata")


# ── Credentials ────────────────────────────────────────────────────────────
def _load_credentials():
    try:
        cid = st.secrets.get("DHAN_CLIENT_ID", "") or st.secrets.get("dhan", {}).get("client_id", "")
        tok = st.secrets.get("DHAN_ACCESS_TOKEN", "") or st.secrets.get("dhan", {}).get("access_token", "")
        return str(cid).strip(), str(tok).strip()
    except Exception:
        return "", ""


CLIENT_ID, ACCESS_TOKEN = _load_credentials()


def _headers():
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID,
    }


# ── Dhan API helpers ───────────────────────────────────────────────────────
def _dhan_post(path, payload, max_retries=3):
    if not CLIENT_ID or not ACCESS_TOKEN:
        st.error("Dhan API credentials not configured (set DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in secrets).")
        return None
    url = f"{DHAN_BASE_URL}{path}"
    delays = [2, 4, 8]
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, headers=_headers(), json=payload, timeout=15)
            if r.status_code == 401:
                st.error("🔑 Dhan token expired or invalid. Update your access token in secrets.")
                return None
            if r.status_code == 429 and attempt < max_retries:
                time.sleep(delays[min(attempt, 2)])
                continue
            if r.status_code == 200:
                return r.json()
            st.error(f"Dhan API error {r.status_code}: {r.text[:200]}")
            return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(delays[min(attempt, 2)])
                continue
            st.error(f"Request error: {e}")
            return None
    return None


@st.cache_data(ttl=300)
def get_expiry_list():
    resp = _dhan_post("/optionchain/expirylist", {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG})
    if resp and isinstance(resp.get("data"), list):
        return resp["data"]
    return []


@st.cache_data(ttl=30)
def get_option_chain(expiry):
    resp = _dhan_post("/optionchain", {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG, "Expiry": expiry})
    if not resp or "data" not in resp:
        return None
    return resp["data"]


@st.cache_data(ttl=21600)
def _load_scrip_master():
    try:
        df = pd.read_csv(SCRIP_MASTER_URL, low_memory=False)
        df.columns = [c.strip().upper() for c in df.columns]
        return df
    except Exception:
        return None


def resolve_option_security_id(expiry, strike, opt_type):
    """Resolve the Dhan NSE_FNO security id for a NIFTY option contract."""
    df = _load_scrip_master()
    if df is None:
        return None
    try:
        inst_col = next((c for c in df.columns if "INSTRUMENT" in c and "NAME" in c), None)
        secid_col = next((c for c in df.columns if "SECURITY_ID" in c), None)
        expiry_col = next((c for c in df.columns if "EXPIRY_DATE" in c), None)
        strike_col = next((c for c in df.columns if "STRIKE" in c), None)
        otype_col = next((c for c in df.columns if "OPTION_TYPE" in c), None)
        sym_col = next((c for c in df.columns if "TRADING_SYMBOL" in c), None)
        exch_col = next((c for c in df.columns if "EXCH_ID" in c), None)
        if not all([inst_col, secid_col, expiry_col, strike_col, otype_col]):
            return None
        m = df[inst_col].astype(str).str.upper().str.strip().eq("OPTIDX")
        if exch_col:
            m &= df[exch_col].astype(str).str.upper().str.strip().eq("NSE")
        if sym_col:
            m &= df[sym_col].astype(str).str.upper().str.contains("NIFTY")
            for excl in ("BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT"):
                m &= ~df[sym_col].astype(str).str.upper().str.contains(excl)
        m &= df[otype_col].astype(str).str.upper().str.strip().eq(opt_type.upper())
        sub = df[m].copy()
        if sub.empty:
            return None
        sub["_strike"] = pd.to_numeric(sub[strike_col], errors="coerce")
        sub["_exp"] = pd.to_datetime(sub[expiry_col], errors="coerce").dt.date
        target_exp = pd.to_datetime(expiry, errors="coerce").date()
        sub = sub[(sub["_strike"].round(0) == round(float(strike))) & (sub["_exp"] == target_exp)]
        if sub.empty:
            return None
        return str(int(sub.iloc[0][secid_col]))
    except Exception:
        return None


def place_buy_order(security_id, quantity, order_type, price):
    payload = {
        "dhanClientId": CLIENT_ID,
        "transactionType": "BUY",
        "exchangeSegment": "NSE_FNO",
        "productType": "INTRADAY",
        "orderType": order_type,           # "LIMIT" or "MARKET"
        "validity": "DAY",
        "securityId": str(security_id),
        "quantity": int(quantity),
        "price": float(price) if order_type == "LIMIT" else 0,
        "triggerPrice": 0,
        "disclosedQuantity": 0,
        "afterMarketOrder": False,
    }
    return _dhan_post("/orders", payload)


# ── Option-chain parsing ───────────────────────────────────────────────────
def _parse_chain(chain):
    """Return (underlying_price, sorted_strikes, oc_dict)."""
    underlying = float(chain.get("last_price") or 0)
    oc = chain.get("oc") or {}
    strikes = sorted(float(k) for k in oc.keys())
    return underlying, strikes, oc


def _leg(oc, strike, opt_type):
    key = f"{strike:.6f}"
    node = oc.get(key) or oc.get(str(strike)) or {}
    leg = node.get("ce" if opt_type == "CE" else "pe", {}) or {}
    return {
        "ltp": float(leg.get("last_price") or 0),
        "iv": float(leg.get("implied_volatility") or 0),
        "delta": float((leg.get("greeks") or {}).get("delta") or 0),
    }


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("⚡ Quick Buy Option")

if not CLIENT_ID or not ACCESS_TOKEN:
    st.warning("Dhan credentials missing. Add DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN to Streamlit secrets to enable live data and order placement.")

expiries = get_expiry_list()
if not expiries:
    st.error("Could not load expiry list from Dhan. Check credentials / network and refresh.")
    st.stop()

expiry = st.selectbox("Expiry", expiries, index=0)
chain = get_option_chain(expiry)
if not chain:
    st.error("Could not load option chain. Refresh to retry.")
    st.stop()

underlying, strikes, oc = _parse_chain(chain)
if not strikes:
    st.error("Option chain returned no strikes.")
    st.stop()

atm_strike = min(strikes, key=lambda s: abs(s - underlying))
st.caption(f"NIFTY spot ₹{underlying:,.2f} · ATM ₹{atm_strike:,.0f} · Expiry {expiry}")

col1, col2, col3 = st.columns(3)
with col1:
    _labels = [f"{s:,.0f}{' (ATM)' if s == atm_strike else ''}" for s in strikes]
    _idx = strikes.index(atm_strike)
    sel_label = st.selectbox("Strike", _labels, index=_idx)
    strike = strikes[_labels.index(sel_label)]
with col2:
    opt_type = st.selectbox("Type", ["CE", "PE"], index=0)
with col3:
    lots = st.number_input("Lots", min_value=1, max_value=100, value=1, step=1)

lot_size = st.number_input("Lot size", min_value=1, max_value=10000, value=DEFAULT_LOT_SIZE, step=1,
                           help="NIFTY lot size used to compute quantity and cost.")

leg = _leg(oc, strike, opt_type)
quantity = int(lots) * int(lot_size)
est_cost = leg["ltp"] * quantity

m1, m2, m3 = st.columns(3)
m1.metric("LTP", f"₹{leg['ltp']:,.2f}")
m2.metric("Quantity", f"{quantity:,}")
m3.metric("Est. Cost", f"₹{est_cost:,.0f}")
st.metric("IV / Delta", f"{leg['iv']:.1f}% / {leg['delta']:.2f}")

order_type = st.radio("Order type", ["LIMIT", "MARKET"], index=0, horizontal=True)
limit_price = st.number_input(
    "Limit Price", min_value=0.0, value=float(round(leg["ltp"], 2)), step=0.05,
    disabled=(order_type == "MARKET"),
)

if st.button(f"🟢 BUY {opt_type} {strike:,.0f}  ·  {lots} lot(s) / {quantity} qty", type="primary", use_container_width=True):
    with st.spinner("Resolving contract and placing order…"):
        sec_id = resolve_option_security_id(expiry, strike, opt_type)
        if not sec_id:
            st.error("Could not resolve the option's security id from the Dhan scrip master. Order not placed.")
        else:
            result = place_buy_order(sec_id, quantity, order_type, limit_price)
            if result and (result.get("orderId") or result.get("orderStatus")):
                st.success(f"✅ Order placed — Order ID: {result.get('orderId', '—')} | Status: {result.get('orderStatus', '—')}")
                st.json(result)
            elif result:
                st.warning(f"Order response: {result}")

st.caption("Orders via Dhan API. Ensure your access token is valid, funds/margin are sufficient, "
           "and the lot size is correct before placing live orders. INTRADAY product, DAY validity.")
