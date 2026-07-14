"""Auto Option Trader — standalone Streamlit app.

A one-screen option order ticket with price-trigger automation for NIFTY and
SENSEX:

  • Pick instrument (NIFTY / SENSEX), expiry, and a strike from ATM ± 5
    (live CE/PE LTP shown for each).
  • Entry as MARKET (fire now) or LIMIT (fire when a trigger is hit). The
    trigger can be based on the underlying SPOT price OR the option LTP.
  • Target and Stop-Loss, each triggerable on SPOT or LTP.
  • Once armed the app polls every few seconds: it takes the entry when the
    entry trigger is reached, then auto-exits on target or stop-loss.

Safety: "Live trading" is OFF by default (dry-run) — triggers are evaluated and
logged but no real orders are sent until you explicitly enable it.

Run:  streamlit run auto_option_trader.py

Credentials from Streamlit secrets (same layout as the main app):
    DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN   (or [dhan] client_id / access_token)
"""
import time
from datetime import datetime

import pandas as pd
import pytz
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except Exception:
    _HAS_AUTOREFRESH = False

st.set_page_config(page_title="Auto Option Trader", page_icon="⚡", layout="centered")

DHAN_BASE_URL = "https://api.dhan.co/v2"
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
IST = pytz.timezone("Asia/Kolkata")

INSTRUMENTS = {
    "NIFTY":  {"scrip": 13, "seg": "IDX_I", "fno_seg": "NSE_FNO", "exch": "NSE",
               "underlying_sym": "NIFTY", "lot": 75, "gap": 50},
    "SENSEX": {"scrip": 51, "seg": "IDX_I", "fno_seg": "BSE_FNO", "exch": "BSE",
               "underlying_sym": "SENSEX", "lot": 20, "gap": 100},
}


# ── Credentials ────────────────────────────────────────────────────────────
def _load_credentials():
    try:
        cid = st.secrets.get("DHAN_CLIENT_ID", "") or st.secrets.get("dhan", {}).get("client_id", "")
        tok = st.secrets.get("DHAN_ACCESS_TOKEN", "") or st.secrets.get("dhan", {}).get("access_token", "")
        return str(cid).strip(), str(tok).strip()
    except Exception:
        return "", ""


CLIENT_ID, ACCESS_TOKEN = _load_credentials()


def _load_telegram():
    try:
        tok = st.secrets.get("TELEGRAM_BOT_TOKEN", "") or st.secrets.get("telegram", {}).get("bot_token", "")
        chat = st.secrets.get("TELEGRAM_CHAT_ID", "") or st.secrets.get("telegram", {}).get("chat_id", "")
        if isinstance(chat, (int, float)):
            chat = str(int(chat))
        return str(tok).strip(), str(chat).strip()
    except Exception:
        return "", ""


TG_TOKEN, TG_CHAT = _load_telegram()


def notify_telegram(msg):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


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
        return None
    url = f"{DHAN_BASE_URL}{path}"
    delays = [1, 2, 4]
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, headers=_headers(), json=payload, timeout=12)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and attempt < max_retries:
                time.sleep(delays[min(attempt, 2)])
                continue
            if r.status_code == 401:
                st.session_state["_token_bad"] = True
            return {"_error": f"{r.status_code}: {r.text[:200]}"}
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(delays[min(attempt, 2)])
                continue
            return {"_error": str(e)}
    return None


def _remember(key, value):
    st.session_state.setdefault("_lastgood", {})[key] = {"v": value, "t": time.time()}


def _recall(key, max_age):
    rec = st.session_state.get("_lastgood", {}).get(key)
    if rec and (time.time() - rec["t"] < max_age):
        return rec["v"]
    return None


def get_expiry_list(scrip, seg):
    """Expiry list with session last-good caching (never caches a failure)."""
    key = f"exp_{scrip}_{seg}"
    fresh = _recall(key, 300)
    if fresh:
        return fresh
    resp = _dhan_post("/optionchain/expirylist", {"UnderlyingScrip": scrip, "UnderlyingSeg": seg}, max_retries=1)
    data = resp.get("data") if isinstance(resp, dict) else None
    if isinstance(data, list) and data:
        _remember(key, data)
        st.session_state["_last_err"] = None
        return data
    st.session_state["_last_err"] = (resp or {}).get("_error", "empty response") if isinstance(resp, dict) else "no response"
    # fall back to any previous good list (even if stale) so the app stays usable
    stale = (st.session_state.get("_lastgood", {}).get(key) or {}).get("v")
    return stale or []


CHAIN_MIN_INTERVAL = 4.0  # Dhan Option Chain API allows ~1 request / 3s — throttle hard


def get_option_chain(scrip, seg, expiry):
    """Throttled option chain: at most one Dhan call every CHAIN_MIN_INTERVAL
    seconds regardless of Streamlit reruns; reuses the last-good chain in
    between and on any rate-limit blip."""
    key = f"chain_{scrip}_{seg}_{expiry}"
    tkey = f"_t_{key}"
    cached = _recall(key, 120)
    since = time.time() - st.session_state.get(tkey, 0)
    if cached is not None and since < CHAIN_MIN_INTERVAL:
        return cached  # too soon — reuse cache, don't hit Dhan
    st.session_state[tkey] = time.time()
    # max_retries=0: never hammer the chain endpoint; the 20s loop retries later
    resp = _dhan_post("/optionchain", {"UnderlyingScrip": scrip, "UnderlyingSeg": seg, "Expiry": expiry}, max_retries=0)
    data = resp.get("data") if isinstance(resp, dict) else None
    if data and data.get("oc"):
        _remember(key, data)
        return data
    st.session_state["_last_err"] = (resp or {}).get("_error", "empty chain") if isinstance(resp, dict) else "no response"
    return cached  # fall back to last-good (already validated < 120s)


NIFTY_INDEX_SCRIP = 13
NIFTY_INDEX_SEG = "IDX_I"


def get_index_ltp(scrip=NIFTY_INDEX_SCRIP, seg=NIFTY_INDEX_SEG):
    """Throttled live LTP of an index (default NIFTY 50). Returns float or None."""
    key = f"idxltp_{scrip}_{seg}"
    tkey = f"_t_{key}"
    cached = _recall(key, 120)
    if cached is not None and (time.time() - st.session_state.get(tkey, 0)) < CHAIN_MIN_INTERVAL:
        return cached
    st.session_state[tkey] = time.time()
    resp = _dhan_post("/marketfeed/ltp", {seg: [int(scrip)]}, max_retries=0)
    if not resp or resp.get("_error"):
        return cached
    try:
        items = (resp.get("data") or {}).get(seg, {})
        node = items.get(str(scrip)) or next(iter(items.values()), None)
        val = float(node.get("last_price")) if node else None
        if val:
            _remember(key, val)
            return val
    except Exception:
        pass
    return cached


@st.cache_data(ttl=21600)
def _load_scrip_master():
    try:
        df = pd.read_csv(SCRIP_MASTER_URL, low_memory=False)
        df.columns = [c.strip().upper() for c in df.columns]
        return df
    except Exception:
        return None


def resolve_option_security_id(exch, underlying_sym, expiry, strike, opt_type):
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
            m &= df[exch_col].astype(str).str.upper().str.strip().eq(exch.upper())
        if sym_col:
            m &= df[sym_col].astype(str).str.upper().str.contains(underlying_sym.upper())
            if underlying_sym.upper() == "NIFTY":
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


def _dhan_get(path, max_retries=2):
    if not CLIENT_ID or not ACCESS_TOKEN:
        return None
    url = f"{DHAN_BASE_URL}{path}"
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers=_headers(), timeout=12)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and attempt < max_retries:
                time.sleep(2)
                continue
            return {"_error": f"{r.status_code}: {r.text[:200]}"}
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"_error": str(e)}
    return None


def get_open_positions():
    """Return open Dhan positions (netQty != 0), throttled to avoid rate limits."""
    cached = _recall("positions", 120)
    if cached is not None and (time.time() - st.session_state.get("_t_positions", 0)) < 5:
        return cached
    st.session_state["_t_positions"] = time.time()
    resp = _dhan_get("/positions", max_retries=0)
    if not resp or (isinstance(resp, dict) and resp.get("_error")):
        return cached if cached is not None else []
    data = resp if isinstance(resp, list) else (resp.get("data") if isinstance(resp, dict) else [])
    out = []
    for p in data or []:
        try:
            net = int(float(p.get("netQty", 0) or 0))
        except Exception:
            net = 0
        if net != 0:
            out.append(p)
    _remember("positions", out)
    return out


def exit_position(pos):
    """Flatten a single open position with an opposite MARKET order."""
    try:
        net = int(float(pos.get("netQty", 0) or 0))
    except Exception:
        net = 0
    if net == 0:
        return {"_error": "no open quantity"}
    txn = "SELL" if net > 0 else "BUY"
    payload = {
        "dhanClientId": CLIENT_ID,
        "transactionType": txn,
        "exchangeSegment": pos.get("exchangeSegment"),
        "productType": pos.get("productType", "INTRADAY"),
        "orderType": "MARKET",
        "validity": "DAY",
        "securityId": str(pos.get("securityId")),
        "quantity": abs(net),
        "price": 0,
        "triggerPrice": 0,
        "disclosedQuantity": 0,
        "afterMarketOrder": False,
    }
    return _dhan_post("/orders", payload)


def place_order(security_id, fno_seg, txn_type, quantity, order_type, price=0):
    payload = {
        "dhanClientId": CLIENT_ID,
        "transactionType": txn_type,        # BUY / SELL
        "exchangeSegment": fno_seg,         # NSE_FNO / BSE_FNO
        "productType": "INTRADAY",
        "orderType": order_type,            # MARKET / LIMIT
        "validity": "DAY",
        "securityId": str(security_id),
        "quantity": int(quantity),
        "price": float(price) if order_type == "LIMIT" else 0,
        "triggerPrice": 0,
        "disclosedQuantity": 0,
        "afterMarketOrder": False,
    }
    return _dhan_post("/orders", payload)


# ── Chain parsing ──────────────────────────────────────────────────────────
def _parse_chain(chain):
    underlying = float(chain.get("last_price") or 0)
    oc = chain.get("oc") or {}
    strikes = sorted(float(k) for k in oc.keys())
    return underlying, strikes, oc


def _leg(oc, strike, opt_type):
    node = oc.get(f"{strike:.6f}") or oc.get(str(strike)) or {}
    leg = node.get("ce" if opt_type == "CE" else "pe", {}) or {}
    return {
        "ltp": float(leg.get("last_price") or 0),
        "iv": float(leg.get("implied_volatility") or 0),
        "delta": float((leg.get("greeks") or {}).get("delta") or 0),
    }


def _reached(level, current, ref):
    """True when `current` has reached/crossed `level`, direction inferred from
    where the value started (`ref`)."""
    if level is None or current is None:
        return False
    if ref is None:
        ref = current
    if ref <= level:
        return current >= level
    return current <= level


def _spot_levels(center, span=200, step=5):
    """Dropdown list of NIFTY spot price levels around `center` (±span, in `step`)."""
    base = round(center / step) * step
    lo = base - span
    n = int((2 * span) / step) + 1
    return [round(lo + i * step, 2) for i in range(n)]


def _nearest_idx(levels, val):
    return min(range(len(levels)), key=lambda i: abs(levels[i] - val)) if levels else 0


# ── Supabase persistence (so an armed trade survives refresh / reconnect) ────
TRADE_KEY = CLIENT_ID or "default"


@st.cache_resource
def _get_supabase():
    try:
        from supabase import create_client
        url = st.secrets.get("supabase", {}).get("url", "")
        key = (st.secrets.get("supabase", {}).get("anon_key", "")
               or st.secrets.get("supabase", {}).get("key", ""))
        if url and key:
            return create_client(url, key)
    except Exception:
        return None
    return None


def save_trade_db(trade):
    sb = _get_supabase()
    if not sb:
        return
    try:
        sb.table("auto_option_trades").upsert(
            {"id": TRADE_KEY, "payload": trade,
             "updated_at": datetime.now(IST).isoformat()}
        ).execute()
    except Exception:
        pass


def load_trade_db():
    sb = _get_supabase()
    if not sb:
        return None
    try:
        res = sb.table("auto_option_trades").select("payload").eq("id", TRADE_KEY).limit(1).execute()
        if res.data:
            return res.data[0].get("payload")
    except Exception:
        return None
    return None


# ── Trade state machine ────────────────────────────────────────────────────
def _trade():
    return st.session_state.get("trade")


def _log(msg):
    logs = st.session_state.setdefault("trade_logs", [])
    logs.append(f"{datetime.now(IST).strftime('%H:%M:%S')} — {msg}")
    st.session_state["trade_logs"] = logs[-30:]


def _live_value(t, basis, spot, ltp):
    return spot if basis == "SPOT" else ltp


def monitor(spot, ltp):
    """Evaluate triggers for the active trade. Called every poll."""
    t = _trade()
    if not t or t["status"] in ("CLOSED",):
        return
    live = st.session_state.get("live_trading", False)

    # ENTRY
    if t["status"] == "ARMED":
        do_entry = False
        if t["order_type"] == "MARKET":
            do_entry = True
        else:  # LIMIT / trigger
            cur = _live_value(t, t["entry_basis"], spot, ltp)
            if _reached(t["entry_trigger"], cur, t["entry_ref"]):
                do_entry = True
        if do_entry:
            if live:
                res = place_order(t["sec_id"], t["fno_seg"], "BUY", t["qty"],
                                  "MARKET" if t["order_type"] == "MARKET" else "LIMIT",
                                  price=ltp)
                if res and res.get("_error"):
                    err = res["_error"]
                    _log(f"ENTRY order error: {err}")
                    # Halt on permanent errors (auth / invalid IP / input) so we
                    # don't keep firing the same rejected order every refresh.
                    if "429" not in err and "too many" not in err.lower():
                        t["status"] = "ERROR"
                        t["error"] = err
                        st.session_state["trade"] = t
                        save_trade_db(t)
                        notify_telegram(f"⚠️ <b>ENTRY FAILED — halted</b>\n{err}\nFix the issue and re-arm.")
                    return
                _log(f"ENTRY BUY placed (order {res.get('orderId','—') if res else '—'}) @ ~₹{ltp:.2f}")
            else:
                _log(f"[DRY] ENTRY BUY {t['qty']} @ ~₹{ltp:.2f}")
            t["status"] = "IN_POSITION"
            t["entry_fill_ltp"] = ltp
            t["entry_fill_spot"] = spot
            st.session_state["trade"] = t
            save_trade_db(t)
            notify_telegram(
                f"🟢 <b>TRADE ENTERED</b> {'(LIVE)' if live else '(DRY)'}\n"
                f"{t['instrument']} {t['strike']:.0f}{t['type']} · {t['lots']} lot(s)/{t['qty']} qty\n"
                f"Entry ~₹{ltp:.2f} · NIFTY spot ₹{spot:,.0f} · {datetime.now(IST).strftime('%H:%M:%S IST')}"
            )
        return

    # EXIT (target / stop-loss)
    if t["status"] == "IN_POSITION":
        # Target
        if t.get("target_price") is not None:
            cur = _live_value(t, t["target_basis"], spot, ltp)
            ref = t["entry_fill_spot"] if t["target_basis"] == "SPOT" else t["entry_fill_ltp"]
            if _reached(t["target_price"], cur, ref):
                _exit(t, "TARGET", spot, ltp, live)
                return
        # Stop-loss
        if t.get("sl_price") is not None:
            cur = _live_value(t, t["sl_basis"], spot, ltp)
            ref = t["entry_fill_spot"] if t["sl_basis"] == "SPOT" else t["entry_fill_ltp"]
            if _reached(t["sl_price"], cur, ref):
                _exit(t, "STOP-LOSS", spot, ltp, live)
                return


def _exit(t, reason, spot, ltp, live):
    if live:
        res = place_order(t["sec_id"], t["fno_seg"], "SELL", t["qty"], "MARKET")
        if res and res.get("_error"):
            _log(f"{reason} exit order error: {res['_error']}")
            return
        _log(f"{reason} EXIT SELL placed (order {res.get('orderId','—') if res else '—'}) @ ~₹{ltp:.2f}")
    else:
        _log(f"[DRY] {reason} EXIT SELL {t['qty']} @ ~₹{ltp:.2f}")
    t["status"] = "CLOSED"
    t["exit_reason"] = reason
    t["exit_ltp"] = ltp
    st.session_state["trade"] = t
    save_trade_db(t)
    _emoji = {"TARGET": "✅", "STOP-LOSS": "🛑", "MANUAL": "⏹"}.get(reason, "⚪")
    notify_telegram(
        f"{_emoji} <b>TRADE EXIT — {reason}</b> {'(LIVE)' if live else '(DRY)'}\n"
        f"{t['instrument']} {t['strike']:.0f}{t['type']} · {t['lots']} lot(s)/{t['qty']} qty\n"
        f"Exit ~₹{ltp:.2f} · NIFTY spot ₹{spot:,.0f} · {datetime.now(IST).strftime('%H:%M:%S IST')}"
    )


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("⚡ Auto Option Trader")

# Auto-refresh every 20s — placed FIRST so it keeps running (and retrying) even
# when a transient Dhan error stops the script further down.
if _HAS_AUTOREFRESH:
    st_autorefresh(interval=20000, key="trade_poll")
else:
    st.warning("streamlit-autorefresh not installed — the app will NOT auto-refresh. "
               "Add `streamlit-autorefresh` to requirements.txt.")

if not CLIENT_ID or not ACCESS_TOKEN:
    st.warning("Dhan credentials missing — add DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN to secrets for live data & orders.")

# Restore an armed/open trade from Supabase once per session (survives F5 / reconnect)
if not st.session_state.get("_db_checked"):
    st.session_state["_db_checked"] = True
    _restored = load_trade_db()
    if _restored and _restored.get("status") in ("ARMED", "IN_POSITION"):
        st.session_state["trade"] = _restored
        _log(f"Restored {_restored.get('status')} trade from Supabase.")

inst_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()), index=0)
cfg = INSTRUMENTS[inst_name]

expiries = get_expiry_list(cfg["scrip"], cfg["seg"])
if not expiries:
    _err = st.session_state.get("_last_err") or "check credentials / network"
    st.error(f"Could not load expiry list from Dhan ({_err}). Auto-retrying every 20s…")
    if st.button("🔄 Retry now"):
        st.rerun()
    st.stop()
expiry = st.selectbox("Expiry", expiries, index=0)

chain = get_option_chain(cfg["scrip"], cfg["seg"], expiry)
if not chain:
    _err = st.session_state.get("_last_err") or "rate limit / network"
    st.error(f"Could not load option chain ({_err}). Auto-retrying every 20s…")
    if st.button("🔄 Retry now"):
        st.rerun()
    st.stop()

spot, strikes, oc = _parse_chain(chain)
if not strikes:
    st.error("Option chain returned no strikes.")
    st.stop()

atm = min(strikes, key=lambda s: abs(s - spot))
atm_idx = strikes.index(atm)
window = strikes[max(0, atm_idx - 5): atm_idx + 6]  # ATM ± 5

# SPOT-basis triggers ALWAYS use the NIFTY 50 index spot — even for SENSEX options.
if cfg["underlying_sym"] == "NIFTY":
    nifty_spot = spot
else:
    nifty_spot = get_index_ltp(NIFTY_INDEX_SCRIP, NIFTY_INDEX_SEG) or spot

st.caption(f"{inst_name} spot ₹{spot:,.2f} · ATM ₹{atm:,.0f} · NIFTY spot ₹{nifty_spot:,.2f} "
           f"(used for SPOT triggers) · Expiry {expiry} · {datetime.now(IST).strftime('%H:%M:%S IST')}")

# ATM ± 5 LTP table
rows = []
for s in window:
    ce, pe = _leg(oc, s, "CE"), _leg(oc, s, "PE")
    rows.append({
        "Strike": f"{s:,.0f}" + (" (ATM)" if s == atm else ""),
        "CE LTP": round(ce["ltp"], 2),
        "PE LTP": round(pe["ltp"], 2),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("### 🎯 Order Setup")
c1, c2, c3 = st.columns(3)
with c1:
    win_labels = [f"{s:,.0f}" + (" (ATM)" if s == atm else "") for s in window]
    sel = st.selectbox("Strike", win_labels, index=win_labels.index(f"{atm:,.0f} (ATM)") if f"{atm:,.0f} (ATM)" in win_labels else 0)
    strike = window[win_labels.index(sel)]
with c2:
    opt_type = st.selectbox("Type", ["CE", "PE"], index=0)
with c3:
    lots = st.number_input("Lots", min_value=1, max_value=100, value=1, step=1)

lot_size = st.number_input("Lot size", min_value=1, max_value=10000, value=cfg["lot"], step=1)
sel_leg = _leg(oc, strike, opt_type)
qty = int(lots) * int(lot_size)
est_cost = sel_leg["ltp"] * qty
m1, m2, m3, m4 = st.columns(4)
m1.metric("Option LTP", f"₹{sel_leg['ltp']:,.2f}")
m2.metric("Quantity", f"{qty:,}")
m3.metric("Est. Cost", f"₹{est_cost:,.0f}")
m4.metric("IV / Delta", f"{sel_leg['iv']:.1f}% / {sel_leg['delta']:.2f}")
st.caption(f"Amount = LTP ₹{sel_leg['ltp']:,.2f} × Qty {qty:,} = ₹{est_cost:,.2f}")

st.markdown("#### Entry")
spot_levels = _spot_levels(nifty_spot)
order_type = st.radio("Entry order type", ["MARKET", "LIMIT"], horizontal=True,
                      help="MARKET = enter immediately. LIMIT = enter when a trigger price is reached.")
entry_basis, entry_trigger = "LTP", None
if order_type == "LIMIT":
    e1, e2 = st.columns(2)
    with e1:
        entry_basis = st.selectbox("Entry trigger on", ["SPOT", "LTP"], index=0,
                                   help="SPOT = NIFTY 50 index level (used for NIFTY and SENSEX). LTP = the option's price.")
    with e2:
        if entry_basis == "SPOT":
            entry_trigger = st.selectbox("Entry NIFTY spot level", spot_levels,
                                         index=_nearest_idx(spot_levels, nifty_spot), key="entry_spot")
        else:
            entry_trigger = st.number_input("Entry trigger price (option LTP)", min_value=0.0,
                                            value=float(round(sel_leg["ltp"], 2)), step=0.05, key="entry_ltp")

st.markdown("#### Target & Stop-Loss")
tcol, scol = st.columns(2)
with tcol:
    use_target = st.checkbox("Use Target", value=True)
    target_basis = st.selectbox("Target on", ["SPOT", "LTP"], index=0, key="tb",
                                help="SPOT = NIFTY 50 index level. LTP = the option's price.")
    if target_basis == "SPOT":
        target_price = st.selectbox("Target NIFTY spot level", spot_levels,
                                    index=_nearest_idx(spot_levels, nifty_spot), key="tps")
    else:
        target_price = st.number_input("Target option price", min_value=0.0,
                                       value=float(round(sel_leg["ltp"] * 1.3, 2)), step=0.05, key="tp")
with scol:
    use_sl = st.checkbox("Use Stop-Loss", value=True)
    sl_basis = st.selectbox("Stop-Loss on", ["SPOT", "LTP"], index=0, key="sb",
                            help="SPOT = NIFTY 50 index level. LTP = the option's price.")
    if sl_basis == "SPOT":
        sl_price = st.selectbox("Stop-Loss NIFTY spot level", spot_levels,
                                index=_nearest_idx(spot_levels, nifty_spot), key="sls")
    else:
        sl_price = st.number_input("Stop-Loss option price", min_value=0.0,
                                   value=float(round(sel_leg["ltp"] * 0.7, 2)), step=0.05, key="sp")

st.divider()
live_trading = st.checkbox("🔴 LIVE TRADING (place real Dhan orders)", value=False,
                           help="OFF = dry-run: triggers are evaluated and logged, but no orders are sent.")
st.session_state["live_trading"] = live_trading

t = _trade()
armed = t is not None and t["status"] in ("ARMED", "IN_POSITION")

ac1, ac2 = st.columns(2)
with ac1:
    if st.button("🟢 ARM TRADE", type="primary", use_container_width=True, disabled=armed):
        sec_id = resolve_option_security_id(cfg["exch"], cfg["underlying_sym"], expiry, strike, opt_type)
        if not sec_id:
            st.error("Could not resolve the option's security id from the Dhan scrip master.")
        else:
            st.session_state["trade"] = {
                "status": "ARMED",
                "instrument": inst_name, "fno_seg": cfg["fno_seg"],
                "strike": strike, "type": opt_type, "lots": int(lots), "qty": qty,
                "sec_id": sec_id, "order_type": order_type,
                "entry_basis": entry_basis, "entry_trigger": entry_trigger,
                "entry_ref": (nifty_spot if entry_basis == "SPOT" else sel_leg["ltp"]),
                "target_basis": target_basis, "target_price": (target_price if use_target else None),
                "sl_basis": sl_basis, "sl_price": (sl_price if use_sl else None),
                "armed_at": datetime.now(IST).strftime("%H:%M:%S"),
            }
            save_trade_db(st.session_state["trade"])
            _log(f"ARMED {inst_name} {strike:.0f}{opt_type} x{lots} ({'LIVE' if live_trading else 'DRY'}) — "
                 f"entry {order_type}{'' if order_type=='MARKET' else f' @ {entry_basis} {entry_trigger}'}")
            st.rerun()
with ac2:
    if st.button("⏹ DISARM / FLATTEN", use_container_width=True, disabled=not armed):
        if t and t["status"] == "IN_POSITION":
            _exit(t, "MANUAL", nifty_spot, sel_leg["ltp"], live_trading)
        else:
            if t:
                t["status"] = "CLOSED"
                st.session_state["trade"] = t
                save_trade_db(t)
            _log("Disarmed before entry.")
        st.rerun()

# Run monitor each script pass — SPOT-basis triggers use NIFTY spot for both instruments
if t and t["status"] in ("ARMED", "IN_POSITION"):
    monitor(nifty_spot, sel_leg["ltp"])
    t = _trade()

# Status panel
if t:
    badge = {"ARMED": "🟡 ARMED (waiting for entry trigger)",
             "IN_POSITION": "🟢 IN POSITION",
             "ERROR": "⛔ HALTED (order rejected — fix & re-arm)",
             "CLOSED": "⚪ CLOSED"}.get(t["status"], t["status"])
    st.markdown(f"**Status:** {badge}")
    st.caption(
        f"{t['instrument']} {t['strike']:.0f}{t['type']} · {t['lots']} lot(s)/{t['qty']} qty · "
        f"entry {t['order_type']}"
        + ("" if t["order_type"] == "MARKET" else f" @ {t['entry_basis']} {t['entry_trigger']}")
        + (f" · target {t['target_basis']} {t['target_price']}" if t.get("target_price") is not None else "")
        + (f" · SL {t['sl_basis']} {t['sl_price']}" if t.get("sl_price") is not None else "")
    )
    if t["status"] == "CLOSED" and t.get("exit_reason"):
        st.info(f"Closed via {t['exit_reason']} @ ~₹{t.get('exit_ltp', 0):.2f}")
    if t["status"] == "ERROR" and t.get("error"):
        st.error(f"Order rejected: {t['error']}")
        if "DH-905" in t["error"] or "Invalid IP" in t["error"]:
            st.warning("DH-905 Invalid IP — Dhan is blocking order placement from this server's IP. "
                       "Whitelist your IP in Dhan (My Profile → DhanHQ Trading API → IP), or run the app "
                       "from a machine whose IP is authorized. Market data works; only order placement is blocked.")

if st.session_state.get("trade_logs"):
    with st.expander("📜 Activity log", expanded=True):
        for line in reversed(st.session_state["trade_logs"]):
            st.text(line)

# ── Live Dhan positions (open trades) ───────────────────────────────────────
st.markdown("### 📂 Live Positions (Dhan)")
positions = get_open_positions()
if not positions:
    st.caption("No open positions in your Dhan account.")
else:
    if st.button("🔴 EXIT ALL POSITIONS", type="primary", use_container_width=True, key="exit_all"):
        for p in positions:
            res = exit_position(p)
            ok = isinstance(res, dict) and (res.get("orderId") or res.get("orderStatus"))
            _log(f"EXIT ALL → {p.get('tradingSymbol','?')}: "
                 + (f"order {res.get('orderId','—')}" if ok else f"error {res.get('_error', res) if isinstance(res, dict) else res}"))
        st.rerun()

    for i, p in enumerate(positions):
        try:
            net = int(float(p.get("netQty", 0) or 0))
        except Exception:
            net = 0
        avg = p.get("buyAvg") or p.get("costPrice") or 0
        try:
            pnl = float(p.get("unrealizedProfit", p.get("realizedProfit", 0)) or 0)
        except Exception:
            pnl = 0.0
        side = "LONG" if net > 0 else "SHORT"
        cols = st.columns([4, 1])
        cols[0].markdown(
            f"**{p.get('tradingSymbol', p.get('securityId', '?'))}** · {side} {abs(net)} · "
            f"avg ₹{float(avg or 0):,.2f} · P&L ₹{pnl:,.0f}"
        )
        if cols[1].button("Exit", key=f"exit_{i}_{p.get('securityId')}", use_container_width=True):
            res = exit_position(p)
            ok = isinstance(res, dict) and (res.get("orderId") or res.get("orderStatus"))
            _log(f"EXIT {p.get('tradingSymbol','?')}: "
                 + (f"order {res.get('orderId','—')}" if ok else f"error {res.get('_error', res) if isinstance(res, dict) else res}"))
            st.rerun()
    st.caption("Exit places a real opposite MARKET order on Dhan regardless of the LIVE toggle.")

# (Auto-refresh is registered at the top of the page so it runs even on errors.)

_db_on = _get_supabase() is not None
st.caption(
    "Orders via Dhan API (INTRADAY, DAY). The app refreshes every 20 seconds — triggers are checked on each "
    "refresh while a trade is armed/in-position, so keep this tab open (an internet drop pauses monitoring). "
    + ("Armed trade is saved to Supabase and restored after a refresh/reconnect. "
       if _db_on else
       "Supabase not configured — armed trade lives only in this tab's memory (lost on hard refresh). ")
    + "Verify lot size, margin and token before enabling LIVE trading."
)
if not _db_on:
    st.caption("To persist triggers, add [supabase] url + anon_key to secrets and create table: "
               "auto_option_trades(id text primary key, payload jsonb, updated_at timestamptz).")
