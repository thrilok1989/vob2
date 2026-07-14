"""Standalone NIFTY Price Alert app.

Runs separately from vob_minimal.py. Polls the NIFTY 50 spot LTP via Dhan
every few seconds; fires Telegram + Discord + browser notifications when
spot comes within a user-set proximity of any target price.

Run: streamlit run nifty_price_alert.py
"""

import json
import os
import time
from datetime import datetime, timezone

import pytz
import requests
import streamlit as st

# ── Config (reads st.secrets first, then env vars) ──────────────────────────
def _cfg(name, default=""):
    try:
        v = st.secrets.get(name, "")
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name, default)

DHAN_CLIENT_ID    = _cfg("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = _cfg("DHAN_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = _cfg("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _cfg("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK_URL = _cfg(
    "DISCORD_WEBHOOK_URL",
    # Fallback to the same webhook the main app uses
    "https://discord.com/api/webhooks/1517484830588141749/I3rR-1g1Z6QDzZztCb43l-rYx3eUhDcy13gx-t2jusdbD6BB5S60wsEQEPcouoA8mtpX",
)

NIFTY_SCRIP = "13"      # Dhan NIFTY 50 index security_id
NIFTY_SEG = "IDX_I"
IST = pytz.timezone("Asia/Kolkata")


# ── Spot fetcher ────────────────────────────────────────────────────────────
def fetch_nifty_spot():
    """Returns (ltp, error_str_or_None)."""
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        return None, "Dhan credentials missing (set DHAN_CLIENT_ID + DHAN_ACCESS_TOKEN)"
    url = "https://api.dhan.co/v2/marketfeed/ltp"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID,
    }
    payload = {NIFTY_SEG: [int(NIFTY_SCRIP)]}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=8)
        if r.status_code != 200:
            return None, f"Dhan HTTP {r.status_code}: {r.text[:120]}"
        data = r.json().get("data") or {}
        node = data.get(NIFTY_SEG) or {}
        leaf = node.get(NIFTY_SCRIP) or {}
        ltp = leaf.get("last_price")
        if ltp is None:
            return None, "no LTP in Dhan response"
        return float(ltp), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


# ── Notification senders ────────────────────────────────────────────────────
def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "telegram creds missing"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID, "text": msg[:4090], "parse_mode": "HTML",
        }, timeout=8)
        if r.status_code == 200:
            return True, "ok"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def send_discord(msg):
    if not DISCORD_WEBHOOK_URL:
        return False, "webhook missing"
    # Strip HTML for discord-style
    import re as _re
    body = _re.sub(r"<b>(.*?)</b>", r"**\1**", msg, flags=_re.DOTALL)
    body = _re.sub(r"<[^>]+>", "", body)
    body = body[:1900]
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json={
            "content": body, "username": "NIFTY Price Alert",
        }, timeout=8)
        if r.status_code in (200, 204):
            return True, "ok"
        return False, f"HTTP {r.status_code}: {r.text[:80]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NIFTY Price Alert", page_icon="🔔", layout="wide")
st.title("🔔 NIFTY Price Alert")
st.caption(
    "Standalone alert app — enter one or more target prices, set proximity, "
    "and you'll get a Telegram + Discord + browser notification when NIFTY "
    "spot comes within range."
)

# Sidebar config
with st.sidebar:
    st.header("⚙️ Settings")
    refresh_secs = st.slider("Refresh interval (sec)", 3, 30, 5, step=1)
    proximity = st.number_input("Proximity threshold (points)",
                                 min_value=1.0, max_value=200.0, value=15.0, step=1.0)
    cooldown_min = st.number_input("Per-target cooldown (minutes)",
                                    min_value=1, max_value=120, value=15, step=1)
    play_sound = st.checkbox("Play browser sound on hit", value=True)
    st.divider()
    st.markdown("**Channel status:**")
    if DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN:
        st.success(f"Dhan: ✅ ({DHAN_CLIENT_ID[:6]}…)")
    else:
        st.error("Dhan: ❌ credentials missing")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        st.success(f"Telegram: ✅ (chat {TELEGRAM_CHAT_ID[-4:]})")
    else:
        st.warning("Telegram: ⚪ creds missing")
    if DISCORD_WEBHOOK_URL:
        st.success("Discord: ✅ (webhook set)")
    else:
        st.warning("Discord: ⚪ webhook missing")
    st.divider()
    if st.button("🧪 Test notifications"):
        t_ok, t_msg = send_telegram("🧪 <b>Test NIFTY Price Alert</b>\nIf you see this, Telegram works.")
        d_ok, d_msg = send_discord("🧪 **Test NIFTY Price Alert**\nIf you see this, Discord works.")
        st.write(f"Telegram: {'✅' if t_ok else '❌'} {t_msg}")
        st.write(f"Discord: {'✅' if d_ok else '❌'} {d_msg}")

# Target manager — list in session_state
targets = st.session_state.setdefault("targets", [])
hit_history = st.session_state.setdefault("hit_history", [])  # list of {ts, target, ltp}
last_alert = st.session_state.setdefault("last_alert", {})    # {target_price: dt}

# ── Preset target slots: Supports + Resistances ─────────────────────────
st.subheader("📍 Set target prices (Supports & Resistances)")
st.caption("Enter up to 2 support levels and 2 resistance levels. Leave at 0 to skip a slot. "
           "Supports trigger when price drops to them; Resistances trigger when price rises to them. "
           "Use the custom form below for any extra targets with custom direction.")

preset = st.session_state.setdefault("preset_levels", {
    "S1": 0.0, "S2": 0.0, "R1": 0.0, "R2": 0.0,
})
pcols = st.columns(4)
preset["S1"] = pcols[0].number_input("🟢 Support 1", min_value=0.0,
                                       value=float(preset["S1"]), step=10.0, key="ps1")
preset["S2"] = pcols[1].number_input("🟢 Support 2", min_value=0.0,
                                       value=float(preset["S2"]), step=10.0, key="ps2")
preset["R1"] = pcols[2].number_input("🔴 Resistance 1", min_value=0.0,
                                       value=float(preset["R1"]), step=10.0, key="pr1")
preset["R2"] = pcols[3].number_input("🔴 Resistance 2", min_value=0.0,
                                       value=float(preset["R2"]), step=10.0, key="pr2")

# Re-build the targets list from preset + any custom additions
_custom_targets = [t for t in targets if not t.get("preset")]
_preset_targets = []
for label, side_default in [("S1", "From above (falling)"),
                              ("S2", "From above (falling)"),
                              ("R1", "From below (rising)"),
                              ("R2", "From below (rising)")]:
    price = preset[label]
    if price > 0:
        _preset_targets.append({
            "price": float(price),
            "side": side_default,
            "preset": label,
        })
st.session_state["targets"] = _preset_targets + _custom_targets
targets = st.session_state["targets"]

# Add CUSTOM target form (for anything beyond the 4 preset slots)
with st.expander("➕ Add custom target (with any direction)", expanded=False):
    with st.form("add_target", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        new_price = c1.number_input("Target price", min_value=0.0, value=0.0, step=10.0)
        side_pref = c2.selectbox("Trigger direction",
                                  ["Both (any cross)", "From below (rising)", "From above (falling)"])
        submitted = c3.form_submit_button("➕ Add")
        if submitted and new_price > 0:
            targets.append({"price": float(new_price), "side": side_pref, "preset": None})
            st.success(f"Added custom target ₹{new_price:.1f} ({side_pref})")

# Show targets table
if targets:
    st.subheader("🎯 Active targets")
    cols = st.columns([2, 3, 3, 3, 1])
    cols[0].markdown("**Slot**")
    cols[1].markdown("**Target Price**")
    cols[2].markdown("**Direction**")
    cols[3].markdown("**Last alerted**")
    cols[4].markdown("**Action**")
    for i, t in enumerate(targets):
        c = st.columns([2, 3, 3, 3, 1])
        slot = t.get("preset") or "Custom"
        slot_emoji = {"S1": "🟢", "S2": "🟢", "R1": "🔴", "R2": "🔴"}.get(slot, "⚙️")
        c[0].write(f"{slot_emoji} {slot}")
        c[1].write(f"₹{t['price']:.1f}")
        c[2].write(t["side"])
        la = last_alert.get(t["price"])
        c[3].write(la.strftime("%H:%M:%S") if la else "—")
        if not t.get("preset") and c[4].button("🗑️", key=f"del_{i}"):
            targets.pop(i)
            st.rerun()
else:
    st.info("Add at least one target price above to start monitoring.")

# Live spot fetch + check
st.divider()
st.subheader("📊 Live monitor")
ltp, err = fetch_nifty_spot()
now_ist = datetime.now(IST)

if err:
    st.error(f"Spot fetch failed: {err}")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("NIFTY Spot", f"₹{ltp:,.1f}")
    c2.metric("Updated", now_ist.strftime("%H:%M:%S IST"))
    nearest = None
    if targets:
        nearest = min(targets, key=lambda t: abs(ltp - t["price"]))
        dist = ltp - nearest["price"]
        c3.metric(
            f"Nearest target ₹{nearest['price']:.1f}",
            f"{dist:+.1f} pts",
            "WITHIN RANGE ⚠️" if abs(dist) <= proximity else f">{proximity:.0f} pts away",
        )

    # Trigger evaluation
    triggered_now = []
    prev_ltp = st.session_state.get("_prev_ltp")
    for t in targets:
        diff = ltp - t["price"]
        if abs(diff) > proximity:
            continue
        # Cooldown check
        la = last_alert.get(t["price"])
        if la and (now_ist - la).total_seconds() < cooldown_min * 60:
            continue
        # Direction check
        side = t["side"]
        if side == "From below (rising)":
            if prev_ltp is None or prev_ltp >= t["price"] or ltp < t["price"] - proximity:
                # Only fire when we cross UP toward target
                if not (prev_ltp is not None and prev_ltp < t["price"] <= ltp + proximity):
                    continue
        elif side == "From above (falling)":
            if not (prev_ltp is not None and prev_ltp > t["price"] >= ltp - proximity):
                continue
        # Fire
        triggered_now.append((t, diff))
        last_alert[t["price"]] = now_ist

    if triggered_now:
        for t, diff in triggered_now:
            direction = "above" if diff > 0 else ("below" if diff < 0 else "at")
            msg = (
                f"🔔 <b>NIFTY PRICE ALERT</b>\n"
                f"Target ₹{t['price']:.1f} reached\n"
                f"Spot ₹{ltp:,.1f} ({diff:+.1f} pts {direction})\n"
                f"Side: {t['side']}\n"
                f"{now_ist.strftime('%H:%M:%S IST')}"
            )
            t_ok, _ = send_telegram(msg)
            d_ok, _ = send_discord(msg)
            hit_history.insert(0, {
                "ts": now_ist, "target": t["price"], "ltp": ltp,
                "tg": t_ok, "dc": d_ok,
            })
            st.toast(f"🔔 NIFTY hit target ₹{t['price']:.1f}", icon="🔔")
            if play_sound:
                # Tiny embedded WAV — short beep
                st.markdown(
                    "<audio autoplay><source "
                    "src='data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQ==' "
                    "type='audio/wav'></audio>",
                    unsafe_allow_html=True,
                )
        st.success(f"🔔 Fired {len(triggered_now)} alert(s)")

    # Update prev_ltp for cross detection next cycle
    st.session_state["_prev_ltp"] = ltp

# History
if hit_history:
    st.divider()
    st.subheader("📜 Hit history (this session)")
    rows = []
    for h in hit_history[:30]:
        rows.append({
            "Time": h["ts"].strftime("%H:%M:%S"),
            "Target": f"₹{h['target']:.1f}",
            "Spot at hit": f"₹{h['ltp']:,.1f}",
            "Telegram": "✅" if h.get("tg") else "❌",
            "Discord": "✅" if h.get("dc") else "❌",
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Auto-refresh
time.sleep(refresh_secs)
st.rerun()
