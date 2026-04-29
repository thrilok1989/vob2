import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import time
import io
import pytz
import os
import datetime
from datetime import datetime, timedelta
try:
    from google import genai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False
from vob_indicators import *
from vob_data import *
from vob_analysis import *

def _get_tg_creds():
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN", "") or getattr(st.secrets, "TELEGRAM_BOT_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "") or getattr(st.secrets, "TELEGRAM_CHAT_ID", "")
        if isinstance(chat_id, (int, float)):
            chat_id = str(int(chat_id))
    except Exception:
        token = chat_id = ""
    return token, chat_id

def _get_gemini_key():
    try:
        return (st.secrets.get("GEMINI_API_KEY", "")
                or st.secrets.get("gemini", {}).get("api_key", "")
                or os.environ.get("GEMINI_API_KEY", ""))
    except Exception:
        return ""

def _strip_html_tags(text):
    """Strip HTML tags for plain-text fallback."""
    import re
    return re.sub(r'<[^>]+>', '', text)


def _tg_msg_hash(message: str) -> str:
    import hashlib
    # Hash on the first non-empty line so minor timestamp suffixes don't break dedup
    first_line = next((l.strip() for l in message.splitlines() if l.strip()), message[:120])
    return hashlib.md5(first_line.encode('utf-8', errors='ignore')).hexdigest()


def send_telegram_message_sync(message, force=False, cooldown_seconds: int = 1800):
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID = _get_tg_creds()
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    # Only send during market hours (8:30 AM - 3:45 PM IST, weekdays) unless forced
    if not force:
        _now = datetime.now(pytz.timezone('Asia/Kolkata'))
        if _now.weekday() >= 5 or not (_now.replace(hour=8, minute=30, second=0, microsecond=0) <= _now <= _now.replace(hour=15, minute=45, second=0, microsecond=0)):
            return

        # Supabase-backed cooldown (survives app restarts)
        _msg_hash = _tg_msg_hash(message)
        _db = getattr(st.session_state, 'db', None)
        if _db is not None:
            try:
                if _db.is_tg_cooldown_active(_msg_hash, cooldown_seconds):
                    return
            except Exception:
                pass
        else:
            # Fallback: in-memory cooldown
            _sent = st.session_state.setdefault('_tg_sent_log', {})
            _last = _sent.get(_msg_hash)
            _now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
            if _last and (_now_ist - _last).total_seconds() < cooldown_seconds:
                return
            _sent[_msg_hash] = _now_ist
            if len(_sent) > 100:
                for _k in sorted(_sent, key=lambda k: _sent[k])[:len(_sent) - 100]:
                    _sent.pop(_k, None)

    # Global rate limit: no two messages sent less than 2 seconds apart
    _now_tg = datetime.now(pytz.timezone('Asia/Kolkata'))
    _last_tg = getattr(st.session_state, '_last_tg_send_time', None)
    if _last_tg and (_now_tg - _last_tg).total_seconds() < 2:
        import time as _time; _time.sleep(2)
    st.session_state._last_tg_send_time = datetime.now(pytz.timezone('Asia/Kolkata'))

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram max message length is 4096 chars — truncate to be safe
    msg_text = message[:4090] if len(message) > 4090 else message

    try:
        # Try HTML mode first
        response = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg_text, "parse_mode": "HTML"}, timeout=10)
        sent_ok = response.status_code == 200
        if sent_ok:
            # Log to Supabase so cooldown persists across restarts
            if not force:
                _msg_hash = _tg_msg_hash(message)
                _db = getattr(st.session_state, 'db', None)
                if _db is not None:
                    try:
                        _db.log_tg_message(_msg_hash, preview=message[:200])
                    except Exception:
                        pass
            return response.json()
        # HTML parse error (400) — retry as plain text
        if response.status_code == 400:
            plain_text = _strip_html_tags(msg_text)
            response2 = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": plain_text}, timeout=10)
            if response2.status_code == 200:
                if not force:
                    _msg_hash = _tg_msg_hash(message)
                    _db = getattr(st.session_state, 'db', None)
                    if _db is not None:
                        try:
                            _db.log_tg_message(_msg_hash, preview=message[:200])
                        except Exception:
                            pass
                return response2.json()
            st.error(f"Telegram error (plain fallback): {response2.status_code} — {response2.text[:200]}")
        else:
            st.error(f"Telegram error: {response.status_code} — {response.text[:200]}")
    except Exception as e:
        st.error(f"Telegram notification error: {e}")



def test_telegram_connection():
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID = _get_tg_creds()
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Credentials not configured"
    try:
        test_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            return True, "Connected"
        return False, f"Error {response.status_code}"
    except Exception as e:
        return False, str(e)



def send_telegram_photo_sync(image_bytes, force=False):
    """Send a PNG image to Telegram via sendPhoto."""
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID = _get_tg_creds()
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if not force:
        _now = datetime.now(pytz.timezone('Asia/Kolkata'))
        if _now.weekday() >= 5 or not (_now.replace(hour=8, minute=30, second=0, microsecond=0) <= _now <= _now.replace(hour=15, minute=45, second=0, microsecond=0)):
            return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        response = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID},
            files={"photo": ("signal.png", image_bytes, "image/png")},
            timeout=20,
        )
        if response.status_code != 200:
            st.error(f"Telegram photo error: {response.status_code}")
    except Exception as e:
        st.error(f"Telegram photo error: {e}")


def ai_analyze_telegram_message(message, kind="master"):
    """Pass a rendered telegram message through Gemini and return its analysis.

    kind: "master" for master trading signal, "oc" for option chain deep analysis.
    Returns (analysis_text, error) — analysis_text is plain text suitable to send
    back to Telegram or display in the app.
    """
    GEMINI_API_KEY = _get_gemini_key()
    client = _get_gemini_client(GEMINI_API_KEY)
    if client is None:
        return None, "Gemini not configured (no GEMINI_API_KEY)."

    try:
        # Strip HTML tags so Gemini sees clean text
        import re
        clean = re.sub(r'<[^>]+>', '', message)

        if kind == "oc":
            task = (
                "This is an Option Chain Deep Analysis snapshot from a Nifty trading app. "
                "Interpret what the option writers/buyers are doing right now, identify the "
                "dominant side, and state whether the current structure favours a bounce, "
                "breakdown, or range. Be concise."
            )
        else:
            task = (
                "This is a Nifty options Master Trading Signal snapshot. Act as an experienced "
                "options trader and give an actionable trade plan."
            )

        prompt = f"""{task}

Respond in exactly this format (plain text, no markdown headers):

📍 Read: one-line summary of the current setup
✅ Trade: entry zone | stop loss | target 1 | target 2
⚠️ Invalidation: one line — what kills the trade
🎯 Confidence: LOW / MEDIUM / HIGH — one-line reason
💡 Key Risk: biggest thing that could go wrong in the next 15 minutes

Keep the whole reply under 170 words. No disclaimers.

SNAPSHOT:
{clean}
"""
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return (resp.text or "").strip(), None
    except Exception as e:
        return None, f"Gemini error: {str(e)[:200]}"

@st.cache_resource

@st.cache_resource
def _get_gemini_client(api_key: str = ""):
    """Initialize and cache the Gemini client (google-genai SDK).
    Key is passed as a param so cache invalidates when the key changes."""
    if not _HAS_GEMINI or not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def ai_explain_signal(master, sa_result, underlying_price, mf_data=None, unwind_summary=None):
    """Ask Gemini to explain the current master signal in plain English."""
    GEMINI_API_KEY = _get_gemini_key()
    client = _get_gemini_client(GEMINI_API_KEY)
    if client is None:
        return None, "Gemini not configured. Add GEMINI_API_KEY to Streamlit secrets."

    try:
        candle = master.get('candle', {})
        gex = master.get('gex', {})
        vidya = master.get('vidya', {})
        ltp_trap = master.get('ltp_trap', {})
        oi_trend = master.get('oi_trend', {})
        reasons = master.get('reasons', [])
        alignment = master.get('alignment', {})
        bull_align = sum(1 for v in alignment.values() if v.get('sentiment_10m') == 'Bullish')
        bear_align = sum(1 for v in alignment.values() if v.get('sentiment_10m') == 'Bearish')

        mf_poc = mf_data.get('poc_price') if mf_data else None
        mf_hi_sent_price = mf_data.get('highest_sentiment_price') if mf_data else None
        mf_hi_sent_dir = mf_data.get('highest_sentiment_direction') if mf_data else None

        oc_bias = sa_result.get('market_bias', 'N/A') if sa_result else 'N/A'
        oc_conf = sa_result.get('confidence', 'N/A') if sa_result else 'N/A'

        context = f"""
Nifty Spot: ₹{underlying_price:.2f}
Signal: {master.get('signal', 'N/A')} ({master.get('trade_type', 'N/A')})
Score: {master.get('abs_score', 0)}/10 ({master.get('strength', 'N/A')})
Confidence: {master.get('confidence', 0)}%

Candle: {candle.get('pattern', 'N/A')} ({candle.get('direction', 'N/A')})
Location: {', '.join(master.get('location', []))}
Volume: {master.get('volume', {}).get('label', 'N/A')} ({master.get('volume', {}).get('ratio', 0)}x)

Resistance: {master.get('resistance_levels', [])[:3]}
Support: {master.get('support_levels', [])[:3]}

GEX: Net={gex.get('net_gex', 0):+.0f}L | ATM={gex.get('atm_gex', 0):+.0f}L | Flip=₹{gex.get('gamma_flip', 'N/A')} | Magnet=₹{gex.get('magnet', 'N/A')} | Mode={gex.get('market_mode', 'N/A')}

OI Trend (ATM): CE={oi_trend.get('ce_activity', 'N/A')} PE={oi_trend.get('pe_activity', 'N/A')} Signal={oi_trend.get('signal', 'N/A')}

VIDYA: {vidya.get('trend', 'N/A')} (Delta {vidya.get('delta_pct', 0):+.0f}%)
VWAP: ₹{ltp_trap.get('vwap', 0):.0f} ({ltp_trap.get('price_vs_vwap', 'N/A')})
Delta Volume: {master.get('delta_trend', 'N/A')}

Alignment (10m): Bull={bull_align} Bear={bear_align}
Option Chain Bias: {oc_bias} (Confidence: {oc_conf}%)

Money Flow POC: ₹{mf_poc if mf_poc else 'N/A'}
MF Strongest Sentiment: ₹{mf_hi_sent_price if mf_hi_sent_price else 'N/A'} ({mf_hi_sent_dir if mf_hi_sent_dir else 'N/A'})

Unwinding Verdict: {unwind_summary.get('verdict') if unwind_summary else 'N/A'}

Confluence Reasons:
{chr(10).join(['- ' + r for r in reasons])}
"""

        prompt = f"""You are an experienced Nifty options trader. Analyze this live market snapshot and give a concise, actionable assessment.

{context}

Respond in exactly this format (markdown):
**📍 Read:** 1 line summarizing what the market is doing right now.
**✅ Trade Setup:** Entry zone, stop loss, target 1, target 2 (use nearest S/R levels).
**⚠️ Invalidation:** What single event would invalidate this trade.
**🎯 Confidence:** LOW / MEDIUM / HIGH with a one-line reason.
**💡 Key Risk:** The biggest thing that could go wrong in the next 15 minutes.

Keep the total under 180 words. No disclaimers."""

        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return resp.text, None
    except Exception as e:
        return None, f"Gemini error: {str(e)[:200]}"

def compute_unwinding_summary(df_atm8):

def check_pcr_sr_proximity_alert(underlying_price, proximity_pts=25):
    """Fire a Telegram alert when spot is within proximity_pts of any PCR S/R level.
    Uses the same _pcr_sr_snapshot stored by the UI — identical data to what's displayed."""
    snapshot = getattr(st.session_state, '_pcr_sr_snapshot', [])
    if not snapshot:
        return
    alerted = st.session_state.setdefault('_pcr_proximity_alerted', {})
    for _s in snapshot:
        label = _s['label']
        pcr_val = _s['pcr']
        sr_type = _s['type']
        if 'Neutral' in sr_type:
            continue
        level = _s['level']
        dist = abs(underlying_price - level)
        if dist > proximity_pts:
            # Reset alert when price moves away (>20 pts buffer)
            if dist > 20:
                alerted.pop(label, None)
            continue
        # Already alerted this level recently?
        last_alert = alerted.get(label)
        if last_alert and (datetime.now(pytz.timezone('Asia/Kolkata')) - last_alert).total_seconds() < 1800:
            continue
        sr_clean = sr_type.replace('🔴', '').replace('🟢', '').replace('⚪', '').strip()
        zone_low  = f"₹{level - proximity_pts:.0f}"
        zone_high = f"₹{level + proximity_pts:.0f}"
        if sr_clean == 'Resistance':
            msg = f"⚠️ PCR RESISTANCE near ₹{level:.0f} | Spot ₹{underlying_price:.0f} | {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M IST')}"
        else:
            msg = f"⚠️ PCR SUPPORT near ₹{level:.0f} | Spot ₹{underlying_price:.0f} | {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M IST')}"
        alerted[label] = datetime.now(pytz.timezone('Asia/Kolkata'))
        return msg
    return None



def send_candle_at_sr_alert(candle, underlying_price, pcr_sr_snapshot, support_levels, resistance_levels, proximity_pts=25):
    """Fire when a bullish candle forms at support or bearish candle forms at resistance.
    Independent of score — fires on pattern+location match alone. Cooldown 10 min per level."""
    def _e(v): return str(v).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    direction = candle.get('direction', '')
    pattern = candle.get('pattern', 'No Pattern')
    if direction not in ('Bullish', 'Bearish') or pattern in ('No Pattern', 'N/A', ''):
        return
    alerted = st.session_state.setdefault('_candle_sr_alerted', {})
    now = datetime.now(pytz.timezone('Asia/Kolkata'))

    # Build candidate levels: PCR S/R + OC levels
    candidate_supports = []
    candidate_resistances = []
    for s in (pcr_sr_snapshot or []):
        sr_clean = s['type'].replace('🔴','').replace('🟢','').replace('⚪','').strip()
        if sr_clean == 'Support':
            candidate_supports.append(('PCR', s['label'], s['level']))
        elif sr_clean == 'Resistance':
            candidate_resistances.append(('PCR', s['label'], s['level']))
    for lvl in (support_levels or []):
        candidate_supports.append(('OC', f"₹{lvl:.0f}", lvl))
    for lvl in (resistance_levels or []):
        candidate_resistances.append(('OC', f"₹{lvl:.0f}", lvl))

    time_str = now.strftime('%H:%M:%S IST')
    if direction == 'Bullish':
        for src, lbl, level in candidate_supports:
            if abs(underlying_price - level) > proximity_pts:
                continue
            key = f"candle_bull_{level:.0f}"
            last = alerted.get(key)
            if last and (now - last).total_seconds() < 1800:
                continue
            msg = f"🕯 BULLISH CANDLE at Support ₹{level:.0f} | {_e(pattern)} | Spot ₹{underlying_price:.0f} | {time_str}"
            alerted[key] = now
            return msg
    elif direction == 'Bearish':
        for src, lbl, level in candidate_resistances:
            if abs(underlying_price - level) > proximity_pts:
                continue
            key = f"candle_bear_{level:.0f}"
            last = alerted.get(key)
            if last and (now - last).total_seconds() < 1800:
                continue
            msg = f"🕯 BEARISH CANDLE at Resistance ₹{level:.0f} | {_e(pattern)} | Spot ₹{underlying_price:.0f} | {time_str}"
            alerted[key] = now
            return msg
    return None



def send_capping_at_sr_alert(sa_result, underlying_price, proximity_pts=25):
    """Fire when sudden call capping or put support is detected at S/R. Cooldown 5 min per strike."""
    def _e(v): return str(v).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    if sa_result is None:
        return
    adf = sa_result.get('analysis_df')
    if adf is None:
        return
    alerted = st.session_state.setdefault('_capping_sr_alerted', {})
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    time_str = now.strftime('%H:%M:%S IST')

    # Call capping (resistance)
    try:
        cap_rows = adf[
            adf['Call_Class'].isin(['High Conviction Resistance', 'Strong Resistance']) &
            adf['Call_Activity'].isin(['Writing (Vol Confirmed)', 'Writing (Resistance)'])
        ]
        for _, r in cap_rows.iterrows():
            strike = float(r['Strike'])
            if abs(underlying_price - strike) > proximity_pts:
                continue
            key = f"cap_call_{strike:.0f}"
            last = alerted.get(key)
            if last and (now - last).total_seconds() < 1800:
                continue
            vol_tag = "🔥 Vol Confirmed" if r.get('CE_Vol_High', False) else ""
            oi_l = float(r.get('CE_OI', 0) or 0) / 100000
            msg = f"🟥 CALL CAPPING ₹{strike:.0f} {vol_tag} | OI {oi_l:.1f}L | Spot ₹{underlying_price:.0f} | {time_str}"
            alerted[key] = now
            return msg
    except Exception:
        pass

    # Put support
    try:
        sup_rows = adf[
            adf['Put_Class'].isin(['High Conviction Support', 'Strong Support']) &
            adf['Put_Activity'].isin(['Writing (Vol Confirmed)', 'Writing (Support)'])
        ]
        for _, r in sup_rows.iterrows():
            strike = float(r['Strike'])
            if abs(underlying_price - strike) > proximity_pts:
                continue
            key = f"cap_put_{strike:.0f}"
            last = alerted.get(key)
            if last and (now - last).total_seconds() < 1800:
                continue
            vol_tag = "🔥 Vol Confirmed" if r.get('PE_Vol_High', False) else ""
            oi_l = float(r.get('PE_OI', 0) or 0) / 100000
            msg = f"🟩 PUT WRITING ₹{strike:.0f} {vol_tag} | OI {oi_l:.1f}L | Spot ₹{underlying_price:.0f} | {time_str}"
            alerted[key] = now
            return msg
    except Exception:
        pass
    return None



def send_decapping_alert(underlying_price):
    """Fire Telegram alert when any ATM±2 strike shows CE/PE OI shedding. Cooldown: 10 min per strike."""
    _decap_atm = getattr(st.session_state, '_decap_atm_data', [])
    if not _decap_atm:
        return None

    _alerted = st.session_state.setdefault('_decap_alerted', {})
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    time_str = now.strftime('%H:%M:%S IST')
    msgs = []

    for _e in _decap_atm:
        if _e.get('ce_decapping'):
            _key = f"decap_ce_{_e['strike']:.0f}"
            if not _alerted.get(_key) or (now - _alerted[_key]).total_seconds() > 600:
                msgs.append(
                    f"⚡ <b>DECAPPING {_e['label']} ₹{_e['strike']:.0f}</b>\n"
                    f"CE OI shed <b>{_e['ce_shed_pct']:.1f}%</b> "
                    f"({_e.get('prev_ce_oi_l',0):.1f}L → {_e['ce_oi_l']:.1f}L)\n"
                    f"→ Ceiling lifting → breakout risk ↑ | Spot ₹{underlying_price:.0f} | {time_str}"
                )
                _alerted[_key] = now

        if _e.get('pe_depeg'):
            _key = f"depeg_pe_{_e['strike']:.0f}"
            if not _alerted.get(_key) or (now - _alerted[_key]).total_seconds() > 600:
                msgs.append(
                    f"⚡ <b>DEPEG {_e['label']} ₹{_e['strike']:.0f}</b>\n"
                    f"PE OI shed <b>{_e['pe_shed_pct']:.1f}%</b> "
                    f"({_e.get('prev_pe_oi_l',0):.1f}L → {_e['pe_oi_l']:.1f}L)\n"
                    f"→ Floor dropping → breakdown risk ↑ | Spot ₹{underlying_price:.0f} | {time_str}"
                )
                _alerted[_key] = now

        if _e.get('ce_capping') and _e.get('ce_built_pct', 0) >= 1.0:
            _key = f"cap_ce_{_e['strike']:.0f}"
            if not _alerted.get(_key) or (now - _alerted[_key]).total_seconds() > 600:
                msgs.append(
                    f"⚡ <b>CALL CAPPING {_e['label']} ₹{_e['strike']:.0f}</b>\n"
                    f"CE OI built <b>{_e['ce_built_pct']:.1f}%</b> "
                    f"({_e.get('prev_ce_oi_l',0):.1f}L → {_e['ce_oi_l']:.1f}L)\n"
                    f"→ Ceiling forming → resistance ↑ | Spot ₹{underlying_price:.0f} | {time_str}"
                )
                _alerted[_key] = now

        if _e.get('pe_capping') and _e.get('pe_built_pct', 0) >= 1.0:
            _key = f"cap_pe_{_e['strike']:.0f}"
            if not _alerted.get(_key) or (now - _alerted[_key]).total_seconds() > 600:
                msgs.append(
                    f"⚡ <b>PUT CAPPING {_e['label']} ₹{_e['strike']:.0f}</b>\n"
                    f"PE OI built <b>{_e['pe_built_pct']:.1f}%</b> "
                    f"({_e.get('prev_pe_oi_l',0):.1f}L → {_e['pe_oi_l']:.1f}L)\n"
                    f"→ Floor forming → support ↑ | Spot ₹{underlying_price:.0f} | {time_str}"
                )
                _alerted[_key] = now

    return "\n\n".join(msgs) if msgs else None



def send_ob_zone_alert(ob_data, underlying_price, proximity_pts=30):
    """Fire a Telegram alert when price enters an active Order Block zone. Cooldown: 15 min per zone."""
    if not ob_data:
        return None
    _alerted = st.session_state.setdefault('_ob_alerted', {})
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    time_str = now.strftime('%H:%M:%S IST')
    msgs = []

    for _obs, _emoji, _label, _action in [
        (ob_data.get('bullish_obs', []), '🟩', 'BULLISH', 'BUY — demand zone holding'),
        (ob_data.get('bearish_obs', []), '🟥', 'BEARISH', 'SELL — supply zone holding'),
    ]:
        for _ob in _obs:
            _ob_lo, _ob_hi, _ob_avg = _ob['low'], _ob['high'], _ob['avg']
            # Price inside or within proximity_pts of the OB zone
            _inside = _ob_lo <= underlying_price <= _ob_hi
            _near   = abs(underlying_price - _ob_avg) <= proximity_pts
            if not (_inside or _near):
                continue
            _key = f"ob_{_label}_{_ob_lo:.0f}_{_ob_hi:.0f}"
            _last = _alerted.get(_key)
            if _last and (now - _last).total_seconds() < 900:
                continue
            _status = '⚡ Price INSIDE zone' if _inside else f'⚡ Price {abs(underlying_price - _ob_avg):.0f}pts from zone'
            msgs.append(
                f"{_emoji} <b>ORDER BLOCK DETECTED — {_label} OB</b>\n"
                f"Zone: ₹{_ob_lo:.0f} – ₹{_ob_hi:.0f} | Avg ₹{_ob_avg:.0f}\n"
                f"{_status} | Spot ₹{underlying_price:.0f}\n"
                f"Action: {_action}\n"
                f"{time_str}"
            )
            _alerted[_key] = now

    return "\n\n".join(msgs) if msgs else None



def send_rejection_alert(candle, underlying_price, df_5m, sa_result, pcr_sr_snapshot, support_levels, resistance_levels, proximity_pts=25):
    """Detect and alert price rejection at ceiling (resistance) or floor (support).

    Rejection confirmed when at least 2 of 3 signals agree:
      Chart  — bearish wick (upper wick > body) at ceiling / bullish wick at floor,
               OR a bearish/bullish candle pattern
      OC     — CE ChgOI building at resistance OR PE ChgOI building at support
      Depth  — BidAskPressure bearish at ceiling / bullish at floor

    Cooldown: 10 min per level.
    """
    def _e(v): return str(v).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    alerted = st.session_state.setdefault('_rejection_alerted', {})
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    time_str = now.strftime('%H:%M:%S IST')

    # ── Chart signal ──────────────────────────────────────────────────────────
    chart_bear = False  # rejection wick or pattern at ceiling
    chart_bull = False  # bounce wick or pattern at floor
    wick_detail = ''
    try:
        if df_5m is not None and len(df_5m) >= 1:
            last = df_5m.iloc[-1]
            o, h, l, c = float(last['open']), float(last['high']), float(last['low']), float(last['close'])
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            total_range = h - l if h > l else 1
            if upper_wick > body and upper_wick / total_range > 0.35:
                chart_bear = True
                wick_detail = f"Upper wick {upper_wick:.0f}pts ({upper_wick/total_range*100:.0f}% of range)"
            if lower_wick > body and lower_wick / total_range > 0.35:
                chart_bull = True
                wick_detail = f"Lower wick {lower_wick:.0f}pts ({lower_wick/total_range*100:.0f}% of range)"
        pat = candle.get('pattern', '')
        cdir = candle.get('direction', '')
        bearish_patterns = {'Bearish Engulfing','Shooting Star','Bearish Harami','Evening Star',
                            'Tweezer Top','Strong Red Candle','Pin Bar'}
        bullish_patterns = {'Bullish Engulfing','Hammer','Bullish Harami','Morning Star',
                            'Tweezer Bottom','Strong Green Candle','Pin Bar'}
        if pat in bearish_patterns or cdir == 'Bearish':
            chart_bear = True
        if pat in bullish_patterns or cdir == 'Bullish':
            chart_bull = True
    except Exception:
        pass

    # ── OC signal: CE/PE ChgOI at the strike ─────────────────────────────────
    oc_bear_strikes = {}   # strike → chg_oi for fresh CE writing
    oc_bull_strikes = {}   # strike → chg_oi for fresh PE writing
    try:
        adf = (sa_result or {}).get('analysis_df')
        if adf is not None:
            for _, r in adf.iterrows():
                sk = float(r['Strike'])
                chg_ce = float(r.get('changeinOpenInterest_CE', 0) or 0)
                chg_pe = float(r.get('changeinOpenInterest_PE', 0) or 0)
                if chg_ce > 5000:   # fresh call writing
                    oc_bear_strikes[sk] = chg_ce
                if chg_pe > 5000:   # fresh put writing
                    oc_bull_strikes[sk] = chg_pe
    except Exception:
        pass

    # ── Depth signal: BidAskPressure from analysis_df ─────────────────────────
    depth_bear_strikes = set()
    depth_bull_strikes = set()
    try:
        adf = (sa_result or {}).get('analysis_df')
        if adf is not None and 'BidAskPressure' in adf.columns:
            for _, r in adf.iterrows():
                sk = float(r['Strike'])
                bap = float(r.get('BidAskPressure', 0) or 0)
                if bap < -500:
                    depth_bear_strikes.add(sk)
                elif bap > 500:
                    depth_bull_strikes.add(sk)
    except Exception:
        pass

    # ── Find STRONGEST S/R only (highest OI strike near price) ───────────────
    # Only check rejection at the single strongest ceiling and strongest floor
    strongest_res = None   # (src, lbl, level, strike)
    strongest_sup = None
    try:
        adf = (sa_result or {}).get('analysis_df')
        if adf is not None:
            # Strongest resistance: highest CE OI above price within 3× proximity
            res_rows = adf[adf['Strike'] >= underlying_price - proximity_pts].copy()
            if not res_rows.empty and 'CE_OI' in res_rows.columns:
                res_rows = res_rows[res_rows['CE_OI'] == res_rows['CE_OI'].max()]
                if not res_rows.empty:
                    sk = float(res_rows.iloc[0]['Strike'])
                    oi = float(res_rows.iloc[0]['CE_OI'])
                    strongest_res = ('OC', f"₹{sk:.0f} OI:{oi/100000:.1f}L", sk, int(sk))
            # Strongest support: highest PE OI below price within 3× proximity
            sup_rows = adf[adf['Strike'] <= underlying_price + proximity_pts].copy()
            if not sup_rows.empty and 'PE_OI' in sup_rows.columns:
                sup_rows = sup_rows[sup_rows['PE_OI'] == sup_rows['PE_OI'].max()]
                if not sup_rows.empty:
                    sk = float(sup_rows.iloc[0]['Strike'])
                    oi = float(sup_rows.iloc[0]['PE_OI'])
                    strongest_sup = ('OC', f"₹{sk:.0f} OI:{oi/100000:.1f}L", sk, int(sk))
    except Exception:
        pass
    # Fallback to PCR levels if OC not available
    if strongest_res is None:
        for s in (pcr_sr_snapshot or []):
            sr_clean = s['type'].replace('🔴','').replace('🟢','').replace('⚪','').strip()
            if sr_clean == 'Resistance' and abs(underlying_price - s['level']) <= proximity_pts * 3:
                strongest_res = ('PCR', s['label'], s['level'], int(s['strike']))
                break
    if strongest_sup is None:
        for s in reversed(pcr_sr_snapshot or []):
            sr_clean = s['type'].replace('🔴','').replace('🟢','').replace('⚪','').strip()
            if sr_clean == 'Support' and abs(underlying_price - s['level']) <= proximity_pts * 3:
                strongest_sup = ('PCR', s['label'], s['level'], int(s['strike']))
                break

    candidate_res = [strongest_res] if strongest_res else []
    candidate_sup = [strongest_sup] if strongest_sup else []

    # ── Check ceiling rejection ───────────────────────────────────────────────
    for src, lbl, level, strike in candidate_res:
        if abs(underlying_price - level) > proximity_pts:
            continue
        key = f"rej_res_{level:.0f}"
        last_alerted = alerted.get(key)
        if last_alerted and (now - last_alerted).total_seconds() < 1800:
            continue
        # Count confirming signals
        signals = []
        if chart_bear:
            signals.append(f"📊 Chart: {_e(wick_detail) if wick_detail else _e(candle.get('pattern',''))}")
        oc_chg = oc_bear_strikes.get(float(strike), 0)
        if oc_chg:
            signals.append(f"📋 OC: CE ChgOI +{int(oc_chg/1000)}K at ₹{strike}")
        if float(strike) in depth_bear_strikes:
            signals.append(f"📉 Depth: Ask wall dominant at ₹{strike}")
        if len(signals) < 2:
            continue
        msg = f"🔴 REJECTION at ₹{level:.0f} ({len(signals)}/3) | Spot ₹{underlying_price:.0f} | {time_str}"
        alerted[key] = now
        return msg

    # ── Check floor bounce ────────────────────────────────────────────────────
    for src, lbl, level, strike in candidate_sup:
        if abs(underlying_price - level) > proximity_pts:
            continue
        key = f"rej_sup_{level:.0f}"
        last_alerted = alerted.get(key)
        if last_alerted and (now - last_alerted).total_seconds() < 1800:
            continue
        signals = []
        if chart_bull:
            signals.append(f"📊 Chart: {_e(wick_detail) if wick_detail else _e(candle.get('pattern',''))}")
        oc_chg = oc_bull_strikes.get(float(strike), 0)
        if oc_chg:
            signals.append(f"📋 OC: PE ChgOI +{int(oc_chg/1000)}K at ₹{strike}")
        if float(strike) in depth_bull_strikes:
            signals.append(f"📈 Depth: Bid wall dominant at ₹{strike}")
        if len(signals) < 2:
            continue
        msg = f"🟢 BOUNCE at ₹{level:.0f} ({len(signals)}/3) | Spot ₹{underlying_price:.0f} | {time_str}"
        alerted[key] = now
        return msg

    return None



def generate_ai_context_message():
    """One-time AI context/glossary — split into two messages to stay under 4096 chars."""
    part1 = """🟡 <b>NIFTY SIGNAL GUIDE (EXECUTION)</b>

<b>📋 ALERT TYPES</b>
⚠️ PCR NEAR LEVEL → price near S/R (±25 pts)
🕯 CANDLE AT LEVEL → confirmation candle
🟥 CALL CAPPING 🔥 → SELL zone (CE writers capping)
🟩 PUT SUPPORT 🔥 → BUY zone (PE writers defending)
🔴 REJECTION (CEILING) → SELL trigger
🟢 BOUNCE (FLOOR) → BUY trigger
👉 No trade without 2 confirmations minimum

<b>📊 SIGNAL SCORE</b>
Range: -5 (strong bear) → +5 (strong bull) | 🟥=Bear 🟩=Bull ⚪=Neutral
👉 Use as bias filter only — not entry trigger

<b>🌍 ALIGNMENT CODES (decode signal alignment block)</b>
N50=Nifty50 SENS=Sensex BNF=BankNifty IT=NiftyIT
REL=Reliance ICICI=ICICIBank VIX=IndiaVIX GOLD CRUDE INR
SP500=S&amp;P500 JP225=Japan225 HSI=HangSeng UK100=FTSE100
Timeframes: 10m|1h|4h|1D|4D|Pattern → 3+ same = confirmed trend
NP=NoPattern Ham=Hammer ShStar=ShootingStar
SGC=StrongGreen SRC=StrongRed BullEng/BearEng BullHar/BearHar

<b>🔬 STRIKE ANALYSIS CODES (ATM±2)</b>
PCR≤0.7=Resistance | 0.71–1.7=Neutral | ≥1.8=Support
CB=CE Bid | CA=CE Ask | PB=PE Bid | PA=PE Ask
P=Pressure (+ve=buyers -ve=sellers) | BA=Bid-Ask pressure
Depth &gt;5K=major wall | &lt;500=breakable | 🧱=Sell wall 🛡=Buy wall

<b>🧱 ENTRY LOGIC</b>
🔴 SELL (CEILING) — all must align:
• Call OI highest (wall) | Market Depth: CA &gt;5K | Delta negative | Wick rejection
👉 Entry: At ceiling stall | SL: Above ceiling

🟢 BUY (FLOOR) — all must align:
• Put OI highest (support) | Market Depth: PB &gt;5K | Delta positive | Wick bounce
👉 Entry: At floor stall | SL: Below floor

<b>🔄 OI WINDING / UNWINDING (in every signal)</b>
CE Build(resist↑) = call writers adding → ceiling stronger
CE Unwind(resist↓) = call writers exiting → ceiling may break
PE Build(supp↑) = put writers adding → floor stronger
PE Unwind(supp↓) = put writers exiting → floor may break
Parallel Bull = CE unwind + PE build → strong BUY signal
Parallel Bear = PE unwind + CE build → strong SELL signal

<b>📡 MARKET MODE (live GEX in every signal)</b>
GEX +ve(+XXL) → RANGE → sell ceiling / buy floor
GEX -ve(-XXL) → TREND → follow momentum, no counter
Confirm with VIDYA direction | 🧊 No depth wall = No trade

<b>🔄 SECTOR ROTATION (in every signal)</b>
RISK-ON 🟢 = cyclicals lead (AUTO/METAL/REALTY/BANK/ENERGY) → bullish for NIFTY
RISK-OFF 🔴 = defensives lead (PHARMA/FMCG/IT/MEDIA) → cautious/bearish for NIFTY
MIXED ⚪ = no clear rotation → wait for alignment
10m/1h = last 10/60 candle bias per sector (🟢Bullish ⚪Neutral 🔴Bearish)
👉 RISK-ON + GEX +ve → buy floor | RISK-OFF + GEX -ve → sell ceiling"""

    part2 = """🟡 <b>NIFTY SIGNAL GUIDE (REFERENCE)</b>

<b>⚠️ CRITICAL RULES</b>
1. GEX -ve + VIDYA trend → DO NOT FADE
2. Market Depth &lt;500 qty → weak level → expect break
3. DTE ≤5 → MaxPain magnet (price pulls toward max pain)
4. Straddle &gt;&gt; ATR → big move already priced in
5. PCR ≤0.7 near ATM+1 → heavy resistance above
6. Delta divergence → price ≠ delta direction → reversal warning

<b>🚫 NO TRADE ZONE</b>
Alignment mixed | GEX neutral | No depth walls | Delta ≈ 0
= Sit out — this is your edge

<b>💰 MONEY FLOW PROFILE (in every signal)</b>
POC (Point of Control) = price with most volume = magnet (price always revisits)
VAH (Value Area High) = upper boundary of 70% vol zone → acts as ceiling/resistance
VAL (Value Area Low) = lower boundary of 70% vol zone → acts as floor/support
Strongest node = price range where buyers/sellers dominated most
🟢 Bullish node = buyers controlled → bounce zone | 🔴 Bearish = sellers → rejection zone
⭐ = POC node (highest vol = strongest magnet)
👉 Price between VAL–VAH = fair value | Below VAL = undervalued | Above VAH = overvalued

<b>⚡ VOLUME DELTA</b>
Total Delta = Buy Vol − Sell Vol | +ve=buyers winning, -ve=sellers
Cum Delta = running total — rising=buyers accumulating, falling=sellers
Ratio &gt;1=buyers dominant | &lt;1=sellers dominant
Divergence = price moves up but delta -ve (or vice versa) → reversal warning
VAH/VAL/POC delta = what buyers/sellers did at Money Flow key levels (most important)
🟢 +Delta at POC/VAL → buyers defending → confirms support / magnet holding
🔴 -Delta at POC/VAH → sellers defending → confirms resistance / ceiling holding

<b>📉 MARKET DEPTH per strike</b>
CB=CE Bid (call buyers) | CA=CE Ask (call sellers)
PB=PE Bid (put buyers) | PA=PE Ask (put sellers)
🧱 Sellers wall: CA &gt; 2×CB → ceiling strong, expect rejection
🛡 Buyers wall: PB &gt; 2×PA → floor strong, expect bounce

<b>📦 OTHER TERMS</b>
GEX Flip = level where market shifts range↔trend
VIDYA -ve%=falling trend | +ve%=rising trend
VPFR: 3 timeframe POC/VAH/VAL confluence = strong zone
Triple POC P1/P2/P3 clustered = very strong magnet
VOB=Volume Order Blocks | HVP=High Volume Pivots
IVR 🔥≥70%=sell favoured | 🧊≤30%=buy favoured
Skew 🔴&gt;1.1=put fear | 🟢&lt;0.9=call greed | ATR14=SL size
Lead/Lag sectors = day%change ranked | RISK-ON=cyclicals up | RISK-OFF=defensives up"""

    return [part1, part2]


def send_master_signal_telegram(result, underlying_price, option_data=None, force=False, skip_image=False, alert_header=""):
    """Send master signal to Telegram. Pass force=True to bypass all guards."""
    if result is None:
        return
    trade_type = result.get('trade_type', '')
    abs_score = result.get('abs_score', 0)
    signal = result.get('signal', '')
    if not force:
        if 'NO TRADE' in trade_type.upper():
            return

        # Location gating: only send BUY when spot is at/near support, and SELL
        # when spot is at/near resistance. "Near" = within 30 pts AND closer to
        # the relevant level than to the opposite level.
        sup_levels = [s for s in (result.get('support_levels') or []) if s]
        res_levels = [r for r in (result.get('resistance_levels') or []) if r]
        nearest_sup = min(sup_levels, key=lambda s: abs(underlying_price - s)) if sup_levels else None
        nearest_res = min(res_levels, key=lambda r: abs(underlying_price - r)) if res_levels else None
        dist_sup = abs(underlying_price - nearest_sup) if nearest_sup is not None else float('inf')
        dist_res = abs(underlying_price - nearest_res) if nearest_res is not None else float('inf')
        PROX_PTS = 30
        is_buy = 'BUY' in trade_type.upper() or 'BREAKOUT' in signal.upper()
        is_sell = 'SELL' in trade_type.upper() or 'BREAKDOWN' in signal.upper()
        if is_buy and (dist_sup > PROX_PTS or dist_sup >= dist_res):
            return
        if is_sell and (dist_res > PROX_PTS or dist_res >= dist_sup):
            return
    time_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')
    # PCR-based S/R block — taken directly from the UI snapshot (same data shown on screen)
    pcr_sr_block = ""
    pcr_sr_levels = []
    try:
        _snapshot = getattr(st.session_state, '_pcr_sr_snapshot', [])
        if _snapshot:
            _pcr_lines = []
            for _s in _snapshot:
                _type_clean = _s['type'].replace('🔴','').replace('🟢','').replace('⚪','').strip()
                _off_txt = f"{_s['offset']:+.0f}" if _s['offset'] != 0 else "0"
                _pcr_lines.append(
                    f"  {_s['label']} ₹{_s['strike']:.0f} | PCR:{_s['pcr']:.2f} | {_type_clean} ₹{_s['level']:.0f} (offset {_off_txt})"
                )
                if _type_clean in ('Resistance', 'Support'):
                    pcr_sr_levels.append({'label': _s['label'], 'type': _type_clean, 'level': _s['level'], 'pcr': _s['pcr']})
            _res_lvls = [x['level'] for x in pcr_sr_levels if x['type'] == 'Resistance']
            _sup_lvls = [x['level'] for x in pcr_sr_levels if x['type'] == 'Support']
            _ceiling = f"₹{min(_res_lvls):.0f}" if _res_lvls else "—"
            _floor   = f"₹{max(_sup_lvls):.0f}" if _sup_lvls else "—"
            # pcr_sr_block suppressed from message — data absorbed into unified strike block
            pcr_sr_block = ""
    except Exception:
        pcr_sr_block = ""
    depth_sr_block = ""  # absorbed into unified strike block

    # VPFR block
    vpfr_block = ""
    try:
        _vpfr = result.get('vpfr', {}) or {}
        _vpfr_lines = []
        for _tf, _label, _bars in [('short', 'Short (30)', 30), ('medium', 'Medium (60)', 60), ('long', 'Long (180)', 180)]:
            _vd = _vpfr.get(_tf)
            if _vd:
                _vpfr_lines.append(
                    f"  {_label}: POC ₹{_vd['poc']:.0f} | VAH ₹{_vd['vah']:.0f} | VAL ₹{_vd['val']:.0f}"
                )
        if _vpfr_lines:
            vpfr_block = "\n<b>📊 VPFR Levels:</b>\n" + "\n".join(_vpfr_lines) + "\n"
    except Exception:
        vpfr_block = ""

    _short_names = {'NIFTY 50':'NIFTY 50','SENSEX':'SENSEX','BANKNIFTY':'BANK NIFTY','NIFTY IT':'NIFTY IT',
                    'RELIANCE':'RELIANCE','ICICIBANK':'ICICI BANK','INDIA VIX':'INDIA VIX','GOLD':'GOLD',
                    'CRUDE OIL':'CRUDE OIL','USD/INR':'USD/INR',
                    'S&P 500':'S&P 500','JAPAN 225':'JAPAN 225','HANG SENG':'HANG SENG','UK 100':'UK 100'}
    _pat_short = {
        'No Pattern':'NP','Doji':'Doji','Hammer':'Ham','Shooting Star':'ShStar',
        'Bullish Engulfing':'BullEng','Bearish Engulfing':'BearEng',
        'Bullish Harami':'BullHar','Bearish Harami':'BearHar',
        'Morning Star':'MornStar','Evening Star':'EveStar',
        'Tweezer Top':'TwTop','Tweezer Bottom':'TwBot',
        'Strong Green Candle':'SGC','Strong Red Candle':'SRC',
        'Inside Bar':'InsBar','Outside Bar':'OutBar',
        'Pin Bar':'PinBar','Marubozu':'Maru',
    }
    def _ae(s): return '🟢' if s == 'Bullish' else '🔴' if s == 'Bearish' else '⚪'
    align_parts = []
    for name in ['NIFTY 50','SENSEX','BANKNIFTY','NIFTY IT','RELIANCE','ICICIBANK','INDIA VIX','GOLD','CRUDE OIL','USD/INR','S&P 500','JAPAN 225','HANG SENG','UK 100']:
        data = result.get('alignment', {}).get(name)
        if data is None:
            continue
        e10 = _ae(data.get('sentiment_10m', ''))
        e1h = _ae(data.get('sentiment_1h', ''))
        e4h = _ae(data.get('sentiment_4h', ''))
        e1d = _ae(data.get('sentiment_1d', ''))
        e4d = _ae(data.get('sentiment_4d', ''))
        pat = data.get('candle_pattern', '') or ''
        cdir = data.get('candle_dir', '') or ''
        pat_clean = pat.strip()
        if not pat_clean or pat_clean == 'No Pattern' or pat_clean == 'N/A':
            pat_str = 'NP'
        else:
            p_short = _pat_short.get(pat_clean, pat_clean[:6])
            p_emoji = '🟢' if cdir == 'Bullish' else '🔴' if cdir == 'Bearish' else '⚪'
            pat_str = f"{p_short}{p_emoji}"
        align_parts.append(f"{_short_names.get(name,name)}:{e10}{e1h}{e4h}{e1d}{e4d}{pat_str}")
    align_text = "  " + "  ".join(align_parts) if align_parts else "  Data unavailable"

    # Location
    loc_text = ", ".join(result['location'])

    # Reasons
    reason_text = "\n".join([f"  ✔ {r}" for r in result['reasons']])

    gex = result['gex']
    vix = result['vix']

    # Signal color
    if 'BUY' in result['trade_type'] or 'BREAKOUT' in result['signal']:
        signal_emoji = "🟢"
    elif 'SELL' in result['trade_type'] or 'BREAKDOWN' in result['signal']:
        signal_emoji = "🔴"
    else:
        signal_emoji = "🟡"

    res_text = ", ".join([f"₹{r:.0f}" for r in result['resistance_levels'][:3]]) if result['resistance_levels'] else "None"
    sup_text = ", ".join([f"₹{s:.0f}" for s in result['support_levels'][:3]]) if result['support_levels'] else "None"

    # Unwinding / parallel winding summary from ATM±5 strikes
    unwind_block = ""
    try:
        df_atm8 = option_data.get('df_atm8') if option_data else None
        uw = compute_unwinding_summary(df_atm8)
        if uw:
            _uv = uw['verdict']
            _uv_e = '🔴' if 'BEAR' in _uv.upper() else '🟢' if 'BULL' in _uv.upper() else '⚪'
            unwind_block = (
                f"\n<b>🔄 OI Wind/Unwind:</b> {_uv_e} {_uv}\n"
                f"  CE: Unwind🔴{uw['ce_unwind_count']}(resist↓) | Build🟢{uw['ce_build_count']}(resist↑)\n"
                f"  PE: Unwind🔴{uw['pe_unwind_count']}(supp↓) | Build🟢{uw['pe_build_count']}(supp↑)\n"
                f"  Parallel:{uw['parallel_count']}(Bull:{uw['bull_parallel']} Bear:{uw['bear_parallel']})\n"
                f"  PE Unwind:{uw['pe_unwind_top']}\n"
                f"  CE Build:{uw['ce_build_top']}\n"
            )
    except Exception:
        unwind_block = ""

    # Market Depth block: top bid/ask walls from ATM±2 strikes
    depth_block = ""
    try:
        _df_d = option_data.get('df_summary') if option_data else None
        if _df_d is not None and not _df_d.empty:
            _depth_lines = []
            for _, _dr in _df_d.iterrows():
                try:
                    _dsk = float(_dr.get('Strike', 0) or 0)
                    if abs(_dsk - underlying_price) > 200:
                        continue
                    _cb = int(_dr.get('bidQty_CE', 0) or 0)
                    _ca = int(_dr.get('askQty_CE', 0) or 0)
                    _pb = int(_dr.get('bidQty_PE', 0) or 0)
                    _pa = int(_dr.get('askQty_PE', 0) or 0)
                    _ba = float(_dr.get('BidAskPressure', 0) or 0)
                    _side = 'Above' if _dsk >= underlying_price else 'Below'
                    _wall_tag = ''
                    if _ca > 0 and _cb > 0:
                        if _ca / max(_cb, 1) > 2: _wall_tag = ' 🧱Sell'
                        elif _cb / max(_ca, 1) > 2: _wall_tag = ' 🛡Buy'
                    _arrow = '↑' if _dsk >= underlying_price else '↓'
                    def _fk(v): return f"{v/1000:.1f}K" if abs(v) >= 1000 else str(v)
                    _depth_lines.append(
                        f"  ₹{_dsk:.0f}{_arrow} CB:{_fk(_cb)} CA:{_fk(_ca)} PB:{_fk(_pb)} PA:{_fk(_pa)} P:{_ba:+.0f}{_wall_tag}"
                    )
                except Exception:
                    pass
            if _depth_lines:
                depth_block = "\n<b>📉 MARKET DEPTH:</b>\n" + "\n".join(_depth_lines) + "\n"
    except Exception:
        depth_block = ""

    # Volume Delta block: summary + candles at VAH/VAL/POC zones
    vol_delta_block = ""
    try:
        _vd = getattr(st.session_state, '_volume_delta_data', None)
        if _vd and _vd.get('summary'):
            _vds = _vd['summary']
            _bias_e = '🟢' if _vds.get('bias') == 'Bullish' else '🔴' if _vds.get('bias') == 'Bearish' else '⚪'
            def _fmt_vol(v):
                v = int(v or 0)
                return f"{v/1000000:.1f}M" if abs(v) >= 1000000 else f"{v/1000:.0f}K"
            _tot_d  = int(_vds.get('total_delta', 0))
            _buy_v  = int(_vds.get('total_buy_volume', 0))
            _sell_v = int(_vds.get('total_sell_volume', 0))
            _d_rat  = float(_vds.get('delta_ratio', 0))
            _cum_d  = int(_vds.get('cum_delta_last', 0))
            _divg   = int(_vds.get('divergence_bars', 0))
            vol_delta_block = (
                f"\n<b>⚡ VOLUME DELTA:</b> {_bias_e} {_vds.get('bias','N/A')}\n"
                f"  Delta: {_fmt_vol(_tot_d)} | Cum Delta: {_fmt_vol(_cum_d)}\n"
                f"  Buy Vol: {_fmt_vol(_buy_v)} | Sell Vol: {_fmt_vol(_sell_v)}\n"
                f"  Ratio: {_d_rat:.2f} | Divergences: {_divg}\n"
            )
            # Delta at VAH / VAL / POC zones from Money Flow Profile
            _mf = getattr(st.session_state, '_money_flow_data', None)
            _vd_df = _vd.get('df')
            if _vd_df is not None and not _vd_df.empty and _mf:
                _poc = _mf.get('poc_price', 0)
                _vah = _mf.get('value_area_high', 0)
                _val = _mf.get('value_area_low', 0)
                _mf_levels = [(l, n) for l, n in [(_poc,'POC'), (_vah,'VAH'), (_val,'VAL')] if l]
                _zone_lines = []
                for _lvl, _lbl in _mf_levels:
                    _near = _vd_df[abs(_vd_df['close'] - _lvl) <= 25].tail(2)
                    for _, _c in _near.iterrows():
                        _cd = int(_c.get('delta', 0))
                        try:
                            _ct = pd.to_datetime(_c.get('datetime')).strftime('%H:%M')
                        except Exception:
                            _ct = str(_c.get('datetime', ''))[-13:-8]
                        _ce = '🟢' if _cd > 0 else '🔴'
                        _zone_lines.append(
                            f"  {_ce} {_lbl}₹{_lvl:.0f} @{_ct} "
                            f"Δ:{_fmt_vol(_cd)} B:{_fmt_vol(int(_c.get('buy_volume',0)))} S:{_fmt_vol(int(_c.get('sell_volume',0)))}"
                        )
                if _zone_lines:
                    vol_delta_block += "<b>  Delta at VAH/VAL/POC:</b>\n" + "\n".join(_zone_lines[:6]) + "\n"
    except Exception:
        vol_delta_block = ""

    # Market Context block: DTE, Max Pain, Straddle, IV Rank, IV Skew, ATR, OI Velocity
    market_ctx_block = ""
    try:
        _mp = (option_data or {}).get('max_pain_strike')
        _expiry_str = (option_data or {}).get('expiry', '')
        _df_ctx = (option_data or {}).get('df_summary')

        # DTE
        _dte = None
        _rollover_flag = ""
        try:
            if _expiry_str:
                _exp_str = str(_expiry_str).strip()
                # API stores expiry as YYYY-MM-DD; try both formats
                for _fmt in ('%Y-%m-%d', '%d-%b-%Y', '%Y/%m/%d'):
                    try:
                        _exp_dt = datetime.strptime(_exp_str, _fmt)
                        break
                    except ValueError:
                        continue
                else:
                    _exp_dt = None
                if _exp_dt:
                    _dte = (_exp_dt.date() - datetime.now(pytz.timezone('Asia/Kolkata')).date()).days
                if _dte <= 5:
                    _rollover_flag = " ⚠️Rollover"
        except Exception:
            pass

        # ATM Straddle + IV Skew from df_summary
        _straddle = None
        _iv_skew = None
        _atm_iv_now = None
        try:
            if _df_ctx is not None and not _df_ctx.empty and 'Strike' in _df_ctx.columns:
                _sg = config.get('strike_gap', 50)
                _atm_sk = round(underlying_price / _sg) * _sg
                _r_atm  = _df_ctx[_df_ctx['Strike'] == _atm_sk]
                _r_atm1 = _df_ctx[_df_ctx['Strike'] == _atm_sk + _sg]  # ATM+1 (CE skew ref)
                _r_atm_1 = _df_ctx[_df_ctx['Strike'] == _atm_sk - _sg] # ATM-1 (PE skew ref)
                if not _r_atm.empty:
                    _lce = float(_r_atm.iloc[0].get('lastPrice_CE', 0) or 0)
                    _lpe = float(_r_atm.iloc[0].get('lastPrice_PE', 0) or 0)
                    if _lce > 0 and _lpe > 0:
                        _straddle = _lce + _lpe
                    _iv_ce_atm = float(_r_atm.iloc[0].get('impliedVolatility_CE', 0) or 0)
                    _iv_pe_atm = float(_r_atm.iloc[0].get('impliedVolatility_PE', 0) or 0)
                    if _iv_ce_atm > 0 and _iv_pe_atm > 0:
                        _atm_iv_now = round((_iv_ce_atm + _iv_pe_atm) / 2, 1)
                # IV Skew: ATM-1 PE IV vs ATM+1 CE IV (ratio > 1 = put skew / hedging)
                if not _r_atm1.empty and not _r_atm_1.empty:
                    _iv_ce1 = float(_r_atm1.iloc[0].get('impliedVolatility_CE', 0) or 0)
                    _iv_pe1 = float(_r_atm_1.iloc[0].get('impliedVolatility_PE', 0) or 0)
                    if _iv_ce1 > 0 and _iv_pe1 > 0:
                        _skew_ratio = round(_iv_pe1 / _iv_ce1, 2)
                        _skew_emoji = '🔴' if _skew_ratio > 1.1 else ('🟢' if _skew_ratio < 0.9 else '⚪')
                        _iv_skew = f"{_skew_emoji}{_skew_ratio}(PE{_iv_pe1:.0f}/CE{_iv_ce1:.0f})"
        except Exception:
            pass

        # Session IV Rank (min-max over session history)
        _iv_rank_str = None
        try:
            _iv_hist = st.session_state.get('_iv_history', [])
            if _iv_hist and _atm_iv_now and len(_iv_hist) >= 3:
                _iv_min = min(_iv_hist)
                _iv_max = max(_iv_hist)
                if _iv_max > _iv_min:
                    _ivr = round((_atm_iv_now - _iv_min) / (_iv_max - _iv_min) * 100, 0)
                    _ivr_emoji = '🔥' if _ivr >= 70 else ('🧊' if _ivr <= 30 else '⚪')
                    _iv_rank_str = f"{_ivr_emoji}{int(_ivr)}%(IV{_atm_iv_now})"
        except Exception:
            pass

        # ATR(14)
        _atr_str = None
        try:
            _atr_val = st.session_state.get('_atr14')
            if _atr_val:
                _atr_str = f"₹{_atr_val:.1f}"
        except Exception:
            pass

        # OI Velocity (rate of total OI change per reading from oi_history)
        _oi_vel_str = None
        try:
            _oi_hist = getattr(st.session_state, 'oi_history', [])
            if _oi_hist and len(_oi_hist) >= 3:
                _oi_vals = [sum(v for v in x.values() if isinstance(v, (int, float))) if isinstance(x, dict) else x for x in _oi_hist[-6:]]
                _oi_delta = _oi_vals[-1] - _oi_vals[0] if len(_oi_vals) >= 2 else 0
                _oi_vel_emoji = '🔺' if _oi_delta > 0 else ('🔻' if _oi_delta < 0 else '➡️')
                _oi_vel_str = f"{_oi_vel_emoji}{abs(_oi_delta)/1000:.1f}K/rd"
        except Exception:
            pass

        _ctx_parts = []
        if _dte is not None:
            _ctx_parts.append(f"DTE:{_dte}{_rollover_flag}")
        if _mp:
            _ctx_parts.append(f"MaxPain:₹{_mp:.0f}")
        if _straddle:
            _ctx_parts.append(f"Straddle:₹{_straddle:.0f}")
        if _iv_rank_str:
            _ctx_parts.append(f"IVR:{_iv_rank_str}")
        if _iv_skew:
            _ctx_parts.append(f"Skew:{_iv_skew}")
        if _atr_str:
            _ctx_parts.append(f"ATR14:{_atr_str}")
        if _oi_vel_str:
            _ctx_parts.append(f"OIVel:{_oi_vel_str}")
        if _ctx_parts:
            market_ctx_block = "\n<b>🌐 Market Context:</b> " + " | ".join(_ctx_parts) + "\n"
    except Exception:
        market_ctx_block = ""

    # Future Swing block (for Part 1 — directional info needed up front)
    swing_block = ""
    # Triple POC block (for Part 2 — deep context)
    poc_block = ""
    try:
        _poc = st.session_state.get('_poc_data') or {}
        _sw  = st.session_state.get('_swing_data') or {}
        _poc_parts = []
        for _pk, _plabel in [('poc1', 'P1(10)'), ('poc2', 'P2(25)'), ('poc3', 'P3(70)')]:
            _p = _poc.get(_pk)
            if _p and _p.get('poc'):
                _poc_parts.append(f"{_plabel}₹{_p['poc']:.0f}")
        _proj = _sw.get('projection')
        _swings = _sw.get('swings', {}) or {}
        _lh = _swings.get('last_swing_high')
        _ll = _swings.get('last_swing_low')
        _sw_dir = _swings.get('direction', '')
        _sw_emoji = '🔴' if _sw_dir == 'bearish' else '🟢' if _sw_dir == 'bullish' else '⚪'
        _swing_parts = []
        if _lh:
            _swing_parts.append(f"SwH₹{_lh['value']:.0f}")
        if _ll:
            _swing_parts.append(f"SwL₹{_ll['value']:.0f}")
        if _proj:
            _swing_parts.append(
                f"→Target₹{_proj['target']:.0f}({_proj['sign']}{_proj['swing_pct']:.1f}%)"
            )
        if _swing_parts:
            swing_block = f"🌀 <b>Future Swing:</b> {_sw_emoji}{_sw_dir.capitalize()} | " + " | ".join(_swing_parts) + "\n"
        if _poc_parts:
            poc_block = "\n<b>📍 Triple POC:</b> " + " | ".join(_poc_parts) + "\n"
    except Exception:
        swing_block = ""
        poc_block = ""

    # Price Action block: LTP Trap, VOB zones, HVP, Delta Vol
    price_action_block = ""
    try:
        vidya = result.get('vidya', {}) or {}
        ltp_trap_d = result.get('ltp_trap', {}) or {}
        vob_b = result.get('vob_blocks') or {}
        hvp_d = result.get('hvp', {}) or {}
        htf_sr_d = result.get('htf_sr', {}) or {}
        delta_trend_d = result.get('delta_trend', 'N/A')

        # VIDYA detailed
        vidya_line = (
            f"  Trend: {vidya.get('trend', 'N/A')} | Smoothed: ₹{vidya.get('smoothed_last', 0):.0f} | "
            f"Delta Vol: {vidya.get('delta_pct', 0):+.0f}%"
        )
        vidya_vol = f"  Buy Vol: {int(vidya.get('buy_vol', 0)):,} | Sell Vol: {int(vidya.get('sell_vol', 0)):,}"

        # LTP Trap
        if ltp_trap_d.get('buy_trap'):
            trap_label = "🪤 Buy Trap"
        elif ltp_trap_d.get('sell_trap'):
            trap_label = "🪤 Sell Trap"
        else:
            trap_label = "No LTP Trap"
        ltp_line = (
            f"  {trap_label} | VWAP: ₹{ltp_trap_d.get('vwap', 0):.0f} "
            f"({ltp_trap_d.get('price_vs_vwap', 'N/A')})"
        )

        # VOB zones (top 3 each by volume)
        vob_lines = []
        for b in sorted((vob_b.get('bullish') or []), key=lambda x: -(x.get('volume', 0)))[:2]:
            vob_lines.append(f"  🟢 ₹{b.get('lower', 0):.0f}-{b.get('upper', 0):.0f}")
        for b in sorted((vob_b.get('bearish') or []), key=lambda x: -(x.get('volume', 0)))[:2]:
            vob_lines.append(f"  🔴 ₹{b.get('lower', 0):.0f}-{b.get('upper', 0):.0f}")
        vob_text = "\n".join(vob_lines) if vob_lines else "  None detected"

        # HVP
        hvp_lines = []
        for h in (hvp_d.get('bullish_hvp') or [])[-3:]:
            hvp_lines.append(f"  🟢 HVP Sup ₹{h.get('price', 0):.0f} | Vol: {int(h.get('volume', 0)):,}")
        for h in (hvp_d.get('bearish_hvp') or [])[-3:]:
            hvp_lines.append(f"  🔴 HVP Res ₹{h.get('price', 0):.0f} | Vol: {int(h.get('volume', 0)):,}")
        hvp_text = "\n".join(hvp_lines) if hvp_lines else "  No HVP detected"

        # VOB — top 1 bull + top 1 bear inline
        vob_parts = []
        for b in sorted((vob_b.get('bullish') or []), key=lambda x: -(x.get('volume', 0)))[:1]:
            vob_parts.append(f"🟢₹{b.get('lower', 0):.0f}-{b.get('upper', 0):.0f}")
        for b in sorted((vob_b.get('bearish') or []), key=lambda x: -(x.get('volume', 0)))[:1]:
            vob_parts.append(f"🔴₹{b.get('lower', 0):.0f}-{b.get('upper', 0):.0f}")
        vob_inline = " ".join(vob_parts) if vob_parts else "—"

        # HVP inline — top 1 each, skip if empty
        hvp_parts = []
        for h in (hvp_d.get('bullish_hvp') or [])[-1:]:
            hvp_parts.append(f"🟢₹{h.get('price', 0):.0f}")
        for h in (hvp_d.get('bearish_hvp') or [])[-1:]:
            hvp_parts.append(f"🔴₹{h.get('price', 0):.0f}")
        hvp_inline = " ".join(hvp_parts) if hvp_parts else "—"

        _vwap_val = ltp_trap_d.get('vwap', 0)
        _vwap_pos = ltp_trap_d.get('price_vs_vwap', 'N/A')
        price_action_block = (
            f"\n🔄 VWAP:₹{_vwap_val:.0f}({_vwap_pos}) | LTP:{trap_label} | "
            f"VOB:{vob_inline} | ΔVol:{delta_trend_d} | HVP:{hvp_inline}\n"
        )
    except Exception:
        price_action_block = ""

    # Money Flow Profile block — high-volume node ranges + sentiment
    mf_block = ""
    try:
        mf = getattr(st.session_state, '_money_flow_data', None)
        if mf and mf.get('rows'):
            high_nodes = [r for r in mf['rows'] if r.get('node_type') == 'High']
            high_nodes.sort(key=lambda r: -r['total_volume'])
            top_nodes = high_nodes[:5]
            hn_lines = []
            for r in top_nodes:
                sent = r.get('sentiment', 'Neutral')
                s_emoji = '🟢' if sent == 'Bullish' else '🔴' if sent == 'Bearish' else '⚪'
                poc_tag = ' ⭐POC' if r.get('is_poc') else ''
                hn_lines.append(
                    f"  {s_emoji} ₹{r['bin_low']:.0f}-₹{r['bin_high']:.0f} | "
                    f"{sent} ({r.get('sentiment_strength', 0):.0f}%) | Vol:{r['volume_pct']:.1f}%{poc_tag}"
                )
            poc_price = mf.get('poc_price', 0)
            vah = mf.get('value_area_high', 0)
            val = mf.get('value_area_low', 0)
            hi_sent_price = mf.get('highest_sentiment_price', 0)
            hi_sent_dir = mf.get('highest_sentiment_direction', 'Neutral')
            hi_sent_emoji = '🟢' if hi_sent_dir == 'Bullish' else '🔴' if hi_sent_dir == 'Bearish' else '⚪'
            # Top nodes with clear labels
            node_parts = []
            for r in top_nodes[:3]:
                s_e = '🟢' if r.get('sentiment') == 'Bullish' else '🔴' if r.get('sentiment') == 'Bearish' else '⚪'
                sent_short = 'Bull' if r.get('sentiment') == 'Bullish' else 'Bear' if r.get('sentiment') == 'Bearish' else 'Neut'
                poc_tag = ' ⭐POC' if r.get('is_poc') else ''
                node_parts.append(f"  {s_e} ₹{r['bin_low']:.0f}-{r['bin_high']:.0f} {sent_short} {r['volume_pct']:.0f}% vol{poc_tag}")
            nodes_inline = "\n".join(node_parts) if node_parts else "  —"
            mf_block = (
                f"\n💰 <b>Money Flow Profile:</b>\n"
                f"  POC ₹{poc_price:.0f} (most traded = price magnet)\n"
                f"  VAH ₹{vah:.0f} (value area ceiling) | VAL ₹{val:.0f} (value area floor)\n"
                f"  Strongest sentiment: {hi_sent_emoji} ₹{hi_sent_price:.0f} {hi_sent_dir}\n"
                f"  Top volume nodes:\n{nodes_inline}\n"
            )
    except Exception:
        mf_block = ""

    # ── Unified Strike Analysis Block (ATM±2) ──────────────────────────────────
    # Merges: PCR S/R + Depth (chart price + strike) + Capping OI + OC Bias signals
    strike_analysis_block = ""
    oc_bias_block = ""  # replaced by strike_analysis_block
    try:
        _df_sum = (option_data or {}).get('df_summary') if option_data else None
        if _df_sum is not None and not _df_sum.empty and 'Strike' in _df_sum.columns:
            _strikes_sorted = sorted(_df_sum['Strike'].unique())
            _atm_strike = None
            if 'Zone' in _df_sum.columns:
                _atm_rows = _df_sum[_df_sum['Zone'] == 'ATM']
                if not _atm_rows.empty:
                    _atm_strike = int(_atm_rows.iloc[0]['Strike'])
            if _atm_strike is None:
                _atm_strike = min(_strikes_sorted, key=lambda s: abs(s - underlying_price))
            _atm_pos = _strikes_sorted.index(_atm_strike) if _atm_strike in _strikes_sorted else -1

            # Capping lookup from sa_result
            _cap_res_map, _cap_sup_map = {}, {}
            try:
                _sa_u = getattr(st.session_state, '_sa_result', None)
                if _sa_u:
                    for _, _cr in (_sa_u.get('top_resistance') or pd.DataFrame()).iterrows():
                        _cap_res_map[float(_cr['Strike'])] = {'oi': _cr.get('CE_OI', 0), 'str': _cr.get('Call_Strength', ''), 'act': _cr.get('Call_Activity', '')}
                    for _, _cs in (_sa_u.get('top_support') or pd.DataFrame()).iterrows():
                        _cap_sup_map[float(_cs['Strike'])] = {'oi': _cs.get('PE_OI', 0), 'str': _cs.get('Put_Strength', ''), 'act': _cs.get('Put_Activity', '')}
            except Exception:
                pass

            # PCR snapshot lookup
            _pcr_snap_map = {float(s['strike']): s for s in (getattr(st.session_state, '_pcr_sr_snapshot', []) or [])}

            def _b(val):
                return '🟢' if val == 'Bullish' else '🔴' if val == 'Bearish' else '⚪'
            def _esc(v):
                return str(v).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            def _fmtk(v):
                try:
                    n = float(v) if v != 'N/A' else 0
                    if n >= 1_000_000:
                        return f"{n/1_000_000:.1f}M"
                    if n >= 1_000:
                        return f"{n/1000:.1f}K"
                    return str(int(n))
                except Exception:
                    return '—'

            _strike_blocks = []
            for _off in [2, 1, 0, -1, -2]:
                _idx = _atm_pos + _off
                if not (0 <= _idx < len(_strikes_sorted)):
                    continue
                _sk = _strikes_sorted[_idx]
                _row = _df_sum[_df_sum['Strike'] == _sk]
                if _row.empty:
                    continue
                _r = _row.iloc[0]
                _g = lambda c, _r=_r: (_r[c] if c in _r.index else 'N/A')

                _label = 'ATM  ' if _off == 0 else (f'ATM+{_off}' if _off > 0 else f'ATM{_off}')
                _pcr = _g('PCR')
                _score = _g('BiasScore')
                _verdict = str(_g('Verdict'))
                _v_emoji = '🔴' if 'Bear' in _verdict else '🟢' if 'Bull' in _verdict else '⚪'

                # PCR S/R classification (from snapshot)
                _snap = _pcr_snap_map.get(float(_sk))
                if _snap:
                    _sr_raw = _snap['type'].replace('🔴','').replace('🟢','').replace('⚪','').strip()
                    _sr_abbr = 'Res' if 'Resist' in _sr_raw else ('Sup' if 'Support' in _sr_raw else 'Neut')
                    _off_txt = f"{_snap['offset']:+.0f}" if _snap['offset'] != 0 else "0"
                    _pcr_sr_str = f"{_v_emoji}{_sr_abbr}₹{_snap['level']:.0f}(off{_off_txt})"
                else:
                    _pcr_sr_str = f"{_v_emoji}{_verdict[:4]}"

                # Depth: chart-adjusted price → raw strike + wall qty
                _dist_d = abs(float(_sk) - underlying_price)
                _off_d = min(15, int(_dist_d * 0.15))
                _ce_ask = float(_g('askQty_CE') or 0)
                _pe_ask = float(_g('askQty_PE') or 0)
                _ce_bid = float(_g('bidQty_CE') or 0)
                _pe_bid = float(_g('bidQty_PE') or 0)
                if float(_sk) >= underlying_price:
                    _eff = float(_sk) - _off_d
                    _wall = _ce_ask + _pe_bid
                    _depth_str = f"📌R₹{_eff:.0f}→₹{_sk:.0f}({_fmtk(_wall)})"
                else:
                    _eff = float(_sk) + _off_d
                    _wall = _pe_ask + _ce_bid
                    _depth_str = f"📌S₹{_eff:.0f}→₹{_sk:.0f}({_fmtk(_wall)})"

                # Capping (OI-based)
                _cap = _cap_res_map.get(float(_sk)) or _cap_sup_map.get(float(_sk))
                if _cap:
                    _cap_emoji = '🟥' if float(_sk) in _cap_res_map else '🟩'
                    _cap_oi = f"{_cap['oi']/100000:.1f}L"
                    _cap_str_s = 'HiConv' if 'High' in str(_cap['str']) else ('Mod' if 'Mod' in str(_cap['str']) else str(_cap['str'])[:5])
                    _cap_str = f"{_cap_emoji}{_cap_oi} {_cap_str_s}"
                else:
                    _cap_str = ""

                # Line 1: strike header
                _l1 = f"<b>{_label} ₹{_sk:.0f}</b> PCR:{_pcr} {_pcr_sr_str} {_cap_str} Sc:{_score}"

                # Line 2: depth + Δ/Γ/Θ + COI/V/IV + Ask/Bid + BA + entry/move
                _ba = _esc(str(_g('BidAskPressure')))
                _entry = _esc(str(_g('Operator_Entry'))).replace('Entry ','').replace('No Signal','NoSig').replace('No Entry','NoEnt')
                _move = _esc(str(_g('FakeReal'))).replace('Real ','R').replace('Fake ','Fk')
                _l2 = (f"  {_depth_str} | "
                       f"Δ{_b(_g('Delta_Bias'))}Γ{_b(_g('Gamma_Bias'))}Θ{_b(_g('Theta_Bias'))} "
                       f"COI{_b(_g('ChgOI_Bias'))}V{_b(_g('Volume_Bias'))}IV{_b(_g('IV_Bias'))} "
                       f"Ask{_b(_g('AskQty_Bias'))}Bid{_b(_g('BidQty_Bias'))} "
                       f"BA:{_ba} E:{_entry} Mv:{_move}")

                # Line 3: CE/PE raw qty + volume + OI comparison
                _ce_vol = _fmtk(_g('totalTradedVolume_CE'))
                _pe_vol = _fmtk(_g('totalTradedVolume_PE'))
                _l3 = (f"  CE B:{_fmtk(_ce_bid)} A:{_fmtk(_ce_ask)} Vol:{_ce_vol} "
                       f"| PE B:{_fmtk(_pe_bid)} A:{_fmtk(_pe_ask)} Vol:{_pe_vol} "
                       f"| COI:{_esc(str(_g('ChgOI_Cmp')))} OI:{_esc(str(_g('OI_Cmp')))}")

                _strike_blocks.append(_l1 + "\n" + _l2 + "\n" + _l3)

            if _strike_blocks:
                strike_analysis_block = "\n<b>🔬 Strike Analysis (ATM±2):</b>\n" + "\n\n".join(_strike_blocks) + "\n"
    except Exception:
        strike_analysis_block = ""

    # Option Chain Deep Analysis block (from session-state sa_result)
    oc_deep_block = ""
    try:
        sa = getattr(st.session_state, '_sa_result', None)
        if sa is not None:
            oc_bias = sa.get('market_bias', '')
            oc_conf = sa.get('confidence', 0)
            if 'Bullish' in oc_bias:
                oc_emoji, oc_cond = '🟢', 'BULLISH'
            elif 'Bearish' in oc_bias:
                oc_emoji, oc_cond = '🔴', 'BEARISH'
            else:
                oc_emoji, oc_cond = '🟡', 'SIDEWAYS'

            oc_res_lines = []
            try:
                for _, r in sa['top_resistance'].iterrows():
                    oc_res_lines.append(
                        f"  🟥 ₹{r['Strike']:.0f} | {r.get('Call_Strength', '')} | "
                        f"OI: {r.get('CE_OI', 0)/100000:.1f}L | {r.get('Call_Activity', '')}"
                    )
            except Exception:
                pass
            oc_sup_lines = []
            try:
                for _, r in sa['top_support'].iterrows():
                    oc_sup_lines.append(
                        f"  🟩 ₹{r['Strike']:.0f} | {r.get('Put_Strength', '')} | "
                        f"OI: {r.get('PE_OI', 0)/100000:.1f}L | {r.get('Put_Activity', '')}"
                    )
            except Exception:
                pass

            # Active signals
            oc_active = []
            try:
                adf = sa.get('analysis_df')
                if adf is not None:
                    cap = adf[
                        adf['Call_Class'].isin(['High Conviction Resistance', 'Strong Resistance']) &
                        adf['Call_Activity'].isin(['Writing (Vol Confirmed)', 'Writing (Resistance)'])
                    ]
                    for _, r in cap.iterrows():
                        vol_tag = "🔥" if r.get('CE_Vol_High', False) else ""
                        oc_active.append(f"🟥 CALL CAPPING {vol_tag} at ₹{r['Strike']:.0f} (OI:{r['CE_OI']/100000:.1f}L, Vol:{r['CE_Vol']/1000:.0f}K)")
                    sup = adf[
                        adf['Put_Class'].isin(['High Conviction Support', 'Strong Support']) &
                        adf['Put_Activity'].isin(['Writing (Vol Confirmed)', 'Writing (Support)'])
                    ]
                    for _, r in sup.iterrows():
                        vol_tag = "🔥" if r.get('PE_Vol_High', False) else ""
                        oc_active.append(f"🟩 PUT CAPPING {vol_tag} at ₹{r['Strike']:.0f} (OI:{r['PE_OI']/100000:.1f}L, Vol:{r['PE_Vol']/1000:.0f}K)")
            except Exception:
                pass

            # Breakout/Breakdown levels
            try:
                bout_df = sa.get('breakout_zones')
                bdn_df = sa.get('breakdown_zones')
                bout_lvl = f"₹{bout_df.iloc[0]['Strike']:.0f}" if bout_df is not None and not bout_df.empty else "None"
                bdn_lvl = f"₹{bdn_df.iloc[0]['Strike']:.0f}" if bdn_df is not None and not bdn_df.empty else "None"
            except Exception:
                bout_lvl, bdn_lvl = "None", "None"

            bias_reasoning = "\n".join([f"  • {s}" for s in sa.get('bias_signals', [])[:4]]) or "  —"
            oc_res_text = "\n".join(oc_res_lines) if oc_res_lines else "  None"
            oc_sup_text = "\n".join(oc_sup_lines) if oc_sup_lines else "  None"
            oc_active_text = "\n".join(oc_active) if oc_active else "  None"

            oc_deep_block = f"""
━━━ {oc_emoji} <b>OPTION CHAIN: {oc_cond}</b> ━━━
📈 Confidence: <b>{oc_conf}%</b>
🚀 Breakout: {bout_lvl} | 💥 Breakdown: {bdn_lvl}

<b>📋 ACTIVE CAPPING / SUPPORT:</b>
{oc_active_text}

<b>📋 BIAS REASONING:</b>
{bias_reasoning}
"""
    except Exception:
        oc_deep_block = ""

    # Multi-instrument capping bias block (compact per-instrument lines)
    _mi_short = {'SENSEX': 'SENSEX', 'BANKNIFTY': 'BANK NIFTY',
                 'RELIANCE': 'RELIANCE', 'ICICIBANK': 'ICICI BANK', 'INFOSYS': 'INFOSYS'}
    def _bias_emoji(b):
        b = str(b)
        return '🟢' if 'Bullish' in b else '🔴' if 'Bearish' in b else '⚪'

    def _mi_cap_line(sn, bias, und, cap_list, sup_list):
        """Build compact capping line: BIAS+NAME UND 🟥R₹cap(OI) 🟩S₹sup(OI) [📍 if near]"""
        be = _bias_emoji(bias)
        parts = [f"{be}{sn}"]
        if und:
            parts.append(f"₹{und:.0f}")
        prox_pct = 0.005  # 0.5% proximity threshold
        for cap in (cap_list or [])[:1]:
            try:
                csk = float(cap['strike'])
                coi = cap.get('oi_l', 0) or 0
                near = und and abs(und - csk) / und <= prox_pct
                tag = f"{'📍' if near else ''}🟥R₹{csk:.0f}({coi:.0f}L)"
                if cap.get('vol_confirmed'):
                    tag += '🔥'
                parts.append(tag)
            except Exception:
                pass
        for sup in (sup_list or [])[:1]:
            try:
                ssk = float(sup['strike'])
                soi = sup.get('oi_l', 0) or 0
                near = und and abs(und - ssk) / und <= prox_pct
                tag = f"{'📍' if near else ''}🟩S₹{ssk:.0f}({soi:.0f}L)"
                if sup.get('vol_confirmed'):
                    tag += '🔥'
                parts.append(tag)
            except Exception:
                pass
        return " ".join(parts)

    _mi_lines = []
    # NIFTY line using sa_result
    try:
        _sa_bias = st.session_state.get('_sa_result') or {}
        _sa_cap = [{'strike': r['Strike'], 'oi_l': float(r.get('openInterest_CE', 0) or 0) / 100000,
                    'vol_confirmed': bool(r.get('Call_Capping_Confirmed', False))}
                   for _, r in (_sa_bias.get('analysis_df', __import__('pandas').DataFrame())).iterrows()
                   if r.get('Call_Capping_Confirmed')] if _sa_bias.get('analysis_df') is not None else []
        _sa_sup = [{'strike': r['Strike'], 'oi_l': float(r.get('openInterest_PE', 0) or 0) / 100000,
                    'vol_confirmed': bool(r.get('Put_Support_Confirmed', False))}
                   for _, r in (_sa_bias.get('analysis_df', __import__('pandas').DataFrame())).iterrows()
                   if r.get('Put_Support_Confirmed')] if _sa_bias.get('analysis_df') is not None else []
        _sa_cap = sorted(_sa_cap, key=lambda x: x['strike'])  # lowest cap = nearest resistance
        _sa_sup = sorted(_sa_sup, key=lambda x: x['strike'], reverse=True)  # highest sup = nearest support
        _mi_lines.append(_mi_cap_line('NIFTY 50', _sa_bias.get('market_bias', ''), underlying_price, _sa_cap, _sa_sup))
    except Exception:
        _mi_lines.append(f"⚫NIFTY 50 ₹{underlying_price:.0f}")

    # Other instruments using mi_instrument_data
    try:
        _mi_state = st.session_state.get('mi_instrument_data') or {}
        for _ikey in INSTRUMENT_CONFIGS:
            _sn = _mi_short.get(_ikey, _ikey[:5])
            try:
                _res = _mi_state.get(_ikey) or {}
                if 'error' in _res:
                    _mi_lines.append(f"⚫{_sn}")
                    continue
                _deep = _res.get('deep') or {}
                _bias = _deep.get('market_bias', '') if _deep else _res.get('pcr_bias', '')
                _und  = _res.get('underlying') or 0
                _caps = _res.get('capping') or []
                _sups = _res.get('support') or []
                # Sort: nearest cap above price, nearest sup below price
                _caps = sorted([c for c in _caps if float(c.get('strike', 0)) >= _und], key=lambda x: x['strike'])
                _sups = sorted([s for s in _sups if float(s.get('strike', 0)) <= _und], key=lambda x: x['strike'], reverse=True)
                _mi_lines.append(_mi_cap_line(_sn, _bias, _und, _caps, _sups))
            except Exception:
                _mi_lines.append(f"⚫{_sn}")
    except Exception:
        pass

    _mi_bias_block = ("\n<b>📡 Index/Stock Capping:</b>\n" + "\n".join(_mi_lines) + "\n") if _mi_lines else ""

    # ── Sector Rotation Block ──
    sector_rotation_block = ""
    try:
        _sr = getattr(st.session_state, '_sector_rotation', None)
        if _sr:
            def _se(s): return '🟢' if s == 'Bullish' else '🔴' if s == 'Bearish' else '⚪'
            _leading  = _sr.get('leading', [])
            _lagging  = _sr.get('lagging', [])
            _rbias    = _sr.get('rotation_bias', '—')
            _lead_str = " ".join(
                f"{_se(r['s10'])}{r['name']}{r['day_chg_pct']:+.1f}%(10m{_se(r['s10'])} 1h{_se(r['s1h'])})"
                for r in _leading
            )
            _lag_str  = " ".join(
                f"{_se(r['s10'])}{r['name']}{r['day_chg_pct']:+.1f}%(10m{_se(r['s10'])} 1h{_se(r['s1h'])})"
                for r in _lagging
            )
            sector_rotation_block = (
                f"\n<b>🔄 SECTOR ROTATION:</b> {_rbias}\n"
                f"  Lead: {_lead_str if _lead_str else '—'}\n"
                f"  Lag:  {_lag_str if _lag_str else '—'}\n"
            )
    except Exception:
        sector_rotation_block = ""



    _oit = result.get('oi_trend', {})
    _vid = result.get('vidya', {})
    _ob = result.get('order_blocks', {})
    _net_gex = gex.get('net_gex', 0)
    _gex_action = "→sell ceiling/buy floor" if _net_gex > 0 else "→follow momentum" if _net_gex < 0 else "→wait"
    # ── Order Block block for msg_part1 ──
    _ob_lines = []
    for _bobs, _emoji, _label in [
        (_ob.get('bullish_obs', []), '🟩', 'Demand'),
        (_ob.get('bearish_obs', []), '🟥', 'Supply'),
    ]:
        for _b in _bobs[:2]:
            _dist = abs(underlying_price - _b['avg'])
            _b_time = ''
            try:
                _b_time = f" @{pd.to_datetime(_b['time']).strftime('%H:%M')}"
            except Exception:
                pass
            _ob_lines.append(
                f"{_emoji} OB {_label} ₹{_b['low']:.0f}-₹{_b['high']:.0f} "
                f"(avg ₹{_b['avg']:.0f}, {_dist:.0f}pts away{_b_time})"
            )
    ob_block = ("\n🔲 <b>ORDER BLOCKS (LuxAlgo):</b>\n" + "\n".join(_ob_lines) + "\n") if _ob_lines else ""

    # ── Decapping / Depeg block (ATM±2 per-strike) ──
    _decap_atm = getattr(st.session_state, '_decap_atm_data', [])
    _decap_lines = []
    for _e in _decap_atm:
        _ce_str = (f"CE:{_e['ce_oi_l']:.1f}L(−{_e['ce_shed_pct']:.1f}%⚡)"
                   if _e.get('ce_decapping') else f"CE:{_e['ce_oi_l']:.1f}L")
        _pe_str = (f"PE:{_e['pe_oi_l']:.1f}L(−{_e['pe_shed_pct']:.1f}%⚡)"
                   if _e.get('pe_depeg') else f"PE:{_e['pe_oi_l']:.1f}L")
        _decap_lines.append(f"  {_e['label']} ₹{_e['strike']:.0f}: {_ce_str} | {_pe_str}")
    decap_block = ("\n<b>🔓 DECAPPING/DEPEG (ATM±2):</b>\n" + "\n".join(_decap_lines) + "\n") if _decap_lines else ""

    # ── Part 1: Signal + Direction + OI Positioning ──
    # Layout: header → time/spot → candle/vol/loc → gamma/sentiment → OI ATM →
    #         future swing → OI positioning (winding + option chain verdict)
    msg_part1 = f"""{signal_emoji} <b>{result['signal']}</b> | {result['trade_type']}
🕐 {time_str} | ₹{underlying_price:.0f}

🕯 {result['candle']['pattern']} ({result['candle']['direction']}) | Vol:{result['volume']['ratio']}x | 📍{loc_text}
🔮 GEX:{gex['net_gex']:+.0f}L({gex['market_mode']} {_gex_action}) Flip:{'₹'+str(int(gex['gamma_flip'])) if gex['gamma_flip'] else '—'}
📊 PCR×GEX:{result['pcr_gex']['badge']} VIX:{float(vix.get('vix',0)):.2f}{vix.get('direction','')} VIDYA:{_vid.get('trend','N/A')}{_vid.get('delta_pct',0):+.0f}%{' ▲' if _vid.get('cross_up') else ' ▼' if _vid.get('cross_down') else ''}
📊 OI ATM {_oit.get('atm_strike','')}: CE {_oit.get('ce_activity','—')} | PE {_oit.get('pe_activity','—')} | {_oit.get('signal','—')}
{decap_block}{ob_block}
<b>📍 DIRECTION</b>
{swing_block}
<b>📉 MARKET DEPTH</b>{depth_block}
<b>🔬 STRIKE ANALYSIS (ATM±2)</b>{strike_analysis_block}
<b>🔄 OI POSITIONING</b>{unwind_block}{oc_deep_block}"""

    # ── Part 2: Deep Analysis + Indices & Stocks at bottom ──
    # Layout: header → market context → vpfr/triple POC/money flow → strike analysis →
    #         price action (vwap/vob/hvp) → indices & stocks (alignment + capping) → AI prompt
    msg_part2 = f"""{signal_emoji} <b>DETAIL (2/2)</b> | {result['signal']} | {time_str}

<b>📊 MARKET CONTEXT</b>{market_ctx_block}
<b>⚡ VOLUME DELTA</b>{vol_delta_block}
<b>📈 VPFR / POC / MONEY FLOW</b>{vpfr_block}{poc_block}{mf_block}
<b>🔄 PRICE STRUCTURE</b>{price_action_block}{sector_rotation_block}
<b>🌍 INDICES &amp; STOCKS</b>
<b>Alignment (10m|1h|4h|1D|4D|Pat):</b>
{align_text}
{_mi_bias_block}
🟡 <code>Analyze ALL data above (Part 1 + Part 2): signal/score, GEX, VIX+VIDYA, OI ATM, future swing, OI winding/positioning, option chain verdict, Market Context (DTE/MaxPain/Straddle/IVR/Skew/ATR/OIVel), VPFR, Triple POC, Money Flow (POC/VAH/VAL), Strike Analysis ATM±2 (PCR S/R + Depth + Capping + Δ/Γ/Θ + BA + CE/PE vol), LTP trap+VWAP, VOB, HVP, Volume Delta (total/cum/ratio + candle delta at VAH/VAL/POC zones), Sector Rotation (leading/lagging sectors 10m+1h bias + RISK-ON/OFF/MIXED), alignment + capping per instrument (NIFTY 50, SENSEX, BANK NIFTY, NIFTY IT, RELIANCE, ICICI BANK, INFOSYS, INDIA VIX, GOLD, CRUDE OIL, USD/INR, S&P 500 futures, JAPAN 225, HANG SENG, UK 100 — 10m|1h|4h|1D|4D). SHORT answers:
GEX RULE (use actual GEX value from data above): GEX +ve → RANGE mode → sell ceiling, buy floor | GEX -ve → TREND mode → follow momentum, no counter-trades. Confirm with VIDYA direction.
1. Market structure: bull/bear/range + reason (state GEX value and what mode it signals)
2. Strongest wall: strike + OI + market depth (bid/ask wall at strike) + VPFR confluence (POC/VAH/VAL near OI S/R strike) + Money Flow Profile POC alignment + why (this is the ceiling/floor where price stalls)
3. Index/Stocks: NIFTY 50 / SENSEX / BANK NIFTY / RELIANCE / ICICI BANK / INFOSYS / INDIA VIX / GOLD / CRUDE OIL / USD/INR / S&P 500 futures / JAPAN 225 / HANG SENG / UK 100 — bias + Cap/Sup/Range
4. Sector Rotation: which sectors leading/lagging (10m+1h bias) → is it RISK-ON (cyclicals up) or RISK-OFF (defensives up)?
5. Entry: ₹___ (at ceiling = strongest OI resistance for SELL / at floor = strongest OI support for BUY — where price won't break) | SL: ₹___ (just above ceiling for SELL / just below floor for BUY) | Target: ₹___ | BUY/SELL Auto scoring engine (like +3 SELL, -2 BUY)
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.</code>"""

    message = msg_part1  # used for Gemini analysis context

    # Send Part 1 (with optional alert header prepended) then Part 2
    _part1_out = (alert_header + "\n\n" + msg_part1) if alert_header else msg_part1
    try:
        send_telegram_message_sync(_part1_out, force=force)
    except Exception as _txt_err:
        st.warning(f"Telegram Part 1 send error: {_txt_err}")
    try:
        send_telegram_message_sync(msg_part2, force=force)
    except Exception as _txt_err:
        st.warning(f"Telegram Part 2 send error: {_txt_err}")

    # Auto-forward to Gemini and post its analysis back to Telegram + app
    try:
        _ai_text, _ai_err = ai_analyze_telegram_message(message, kind="master")
        if _ai_text:
            st.session_state._last_gemini_master = {
                'text': _ai_text,
                'time': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST'),
            }
            _ai_telegram = f"🤖 <b>GEMINI ANALYSIS — MASTER SIGNAL</b>\n\n{_ai_text}"
            try:
                send_telegram_message_sync(_ai_telegram, force=force)
            except Exception:
                pass
        # errors (e.g. no API key) are not persisted — nothing to show
    except Exception:
        pass


