"""📱 VOB Mobile — phone-friendly companion view for vob_minimal.py.

vob_minimal.py stays the single backend: every refresh cycle it exports its
computed signals to mobile_snapshot.json (see export_mobile_snapshot there).
This app only READS that file and renders it as a single-column, big-touch
mobile dashboard. It performs no API calls and no computation of its own.

Run both apps side by side (same machine / same working directory):

    streamlit run vob_minimal.py --server.port 8501   # backend (keep a tab open)
    streamlit run vob_mobile.py  --server.port 8502   # open this on your phone

To point at a snapshot elsewhere, set VOB_MOBILE_SNAPSHOT to the same path in
both processes. Note: Streamlit apps only compute while a browser session is
open, so the main app needs at least one open tab (desktop or background) for
the snapshot to keep updating.
"""

import json
import os
from datetime import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pytz

SNAPSHOT_PATH = os.environ.get(
    'VOB_MOBILE_SNAPSHOT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mobile_snapshot.json'))

IST = pytz.timezone('Asia/Kolkata')

st.set_page_config(
    page_title="VOB Mobile",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Mobile-first styling: tight padding, big type, dark cards ───────────────
st.markdown("""
<style>
  .block-container { padding: 0.8rem 0.9rem 3rem 0.9rem; max-width: 560px; }
  header[data-testid="stHeader"] { height: 0; }
  div[data-testid="stExpander"] summary p { font-size: 15px; font-weight: 600; }
  .vob-card { border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
              background: #1a2030; border: 1px solid #333; color: #eee; }
  .vob-hero { font-size: 21px; font-weight: 800; }
  .vob-sub  { color: #aab; font-size: 13px; margin-top: 2px; }
  .vob-row  { display: flex; justify-content: space-between; align-items: center;
              padding: 6px 2px; border-bottom: 1px solid #2a3145; font-size: 15px; }
  .vob-row:last-child { border-bottom: none; }
  .vob-chip { display: inline-block; border-radius: 14px; padding: 3px 11px;
              margin: 2px 3px 2px 0; font-size: 13px; font-weight: 600; }
  .vob-flex { display: flex; gap: 8px; flex-wrap: wrap; }
  .vob-mini { flex: 1 1 30%; min-width: 105px; border-radius: 10px;
              padding: 8px 10px; text-align: center; color: #eee; }
  .vob-mini .t { font-size: 12px; color: #aab; }
  .vob-mini .v { font-size: 14px; font-weight: 700; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

GREEN_BG, GREEN_BD = '#0a3d2a', '#00ff88'
RED_BG, RED_BD = '#3d0a1f', '#ff4444'
GRAY_BG, GRAY_BD = '#222a3a', '#888'


def _tone(text):
    """(bg, border) colors from a bull/bear-ish label."""
    t = (text or '').lower()
    if 'bull' in t or 'buy' in t or t == 'up':
        return GREEN_BG, GREEN_BD
    if 'bear' in t or 'sell' in t or t == 'down':
        return RED_BG, RED_BD
    return GRAY_BG, GRAY_BD


def card(html, bg=None, bd=None):
    style = f"background:{bg}; border:1px solid {bd};" if bg else ""
    st.markdown(f"<div class='vob-card' style='{style}'>{html}</div>",
                unsafe_allow_html=True)


def load_snapshot():
    try:
        with open(SNAPSHOT_PATH, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


# ── Controls (kept out of the way in the sidebar) ───────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    refresh_s = st.slider("Auto-refresh (seconds)", 5, 60, 10, 5)
    paused = st.toggle("Pause refresh", value=False)
    st.caption(f"Snapshot file:\n`{SNAPSHOT_PATH}`")

if not paused:
    st_autorefresh(interval=refresh_s * 1000, key="vob_mobile_refresh")

snap = load_snapshot()

if not snap:
    st.markdown("## 📱 VOB Mobile")
    st.warning(
        "No snapshot found yet.\n\n"
        "Start the backend first and keep one browser tab of it open:\n\n"
        "```\nstreamlit run vob_minimal.py\n```\n"
        f"It writes `{os.path.basename(SNAPSHOT_PATH)}` every cycle; "
        "this app refreshes automatically once it appears.")
    st.stop()

# ── Header: spot + freshness ────────────────────────────────────────────────
age_s = None
try:
    snap_dt = IST.localize(datetime.strptime(snap.get('ts', ''), '%Y-%m-%d %H:%M:%S'))
    age_s = (datetime.now(IST) - snap_dt).total_seconds()
except Exception:
    pass

if age_s is not None and age_s > 120:
    fresh = f"<span style='color:{RED_BD};'>⚠️ stale · {age_s/60:.0f} min old</span>"
elif age_s is not None:
    fresh = f"<span style='color:{GREEN_BD};'>● live · {age_s:.0f}s ago</span>"
else:
    fresh = ""
card(
    f"<div class='vob-hero'>NIFTY ₹{snap.get('spot', 0):,.1f}</div>"
    f"<div class='vob-sub'>{snap.get('ts', '—')} IST &nbsp; {fresh}</div>"
)
if age_s is not None and age_s > 120:
    st.error("Snapshot is stale — make sure vob_minimal.py is running "
             "with a browser tab open.", icon="⚠️")

# ── Overall NIFTY verdict (14-leg) ──────────────────────────────────────────
ov = snap.get('leg_overall') or {}
if ov:
    bg, bd = _tone(ov.get('dir'))
    by = ov.get('by_speed') or {}
    minis = ""
    for k, icon, name in (('fast', '🚀', 'Fast'), ('lag', '🐢', 'Lagging'), ('mis', '🌫️', 'Misguiding')):
        v = by.get(k) or {}
        mbg, mbd = _tone(v.get('dir'))
        minis += (f"<div class='vob-mini' style='background:{mbg}; border:1px solid {mbd};'>"
                  f"<div class='t'>{icon} {name}</div>"
                  f"<div class='v'>{v.get('em', '')} {v.get('label', '—')}</div>"
                  f"<div class='t'>net {v.get('net', 0):+d}</div></div>")
    card(
        f"<div class='vob-hero'>{ov.get('em', '')} {ov.get('label', '—')}</div>"
        f"<div class='vob-sub'>14-leg verdict · {ov.get('bull', 0)}↑ / {ov.get('bear', 0)}↓ "
        f"(net {ov.get('net', 0):+d}) · act on 🚀 Fast</div>"
        f"<div class='vob-flex' style='margin-top:8px;'>{minis}</div>",
        bg, bd)

# ── Composite bias + action ─────────────────────────────────────────────────
cb = snap.get('composite') or {}
if cb:
    bg, bd = _tone(cb.get('direction') or cb.get('label'))
    enter = ("<span class='vob-chip' style='background:#00ff88; color:#003018;'>⚡ ENTER NOW</span>"
             if cb.get('enter_now') else "")
    card(
        f"<div class='vob-hero'>{cb.get('emoji', '')} {cb.get('label', '—')} "
        f"<span style='font-size:15px;'>({cb.get('score', 0):+.1f})</span> {enter}</div>"
        f"<div class='vob-sub'>{cb.get('action', '')} · Confidence {cb.get('confidence', '—')}</div>"
        f"<div class='vob-sub'>ATM {cb.get('atm_strike', 0):.0f} · "
        f"CE ₹{cb.get('ce_ltp', 0):.2f} · PE ₹{cb.get('pe_ltp', 0):.2f}</div>",
        bg, bd)
    if cb.get('reasons') or cb.get('contradictions'):
        with st.expander("Why? (confirming vs contradicting)"):
            for r in cb.get('reasons') or []:
                st.markdown(f"✅ {r}")
            for c in cb.get('contradictions') or []:
                st.markdown(f"⚠️ {c}")

# ── VOB watch: premiums about to RISE / FALL ────────────────────────────────
vw = snap.get('vob_watch') or {}
rise, fall = vw.get('rise') or [], vw.get('fall') or []
if rise or fall:
    st.markdown("#### 🧲 VOB watch — option premium about to…")
    if rise:
        rows = "".join(
            f"<div class='vob-row'><span>📍 {r['tag']}</span>"
            f"<span>₹{r['ltp']:.2f} <span style='color:#aab;'>(mid ₹{r['mid']:.2f})</span></span></div>"
            for r in rise)
        card(f"<b style='color:{GREEN_BD};'>RISE ⬆️ — LTP at its bullish VOB (support)</b>{rows}",
             GREEN_BG, GREEN_BD)
    if fall:
        rows = "".join(
            f"<div class='vob-row'><span>📍 {r['tag']}</span>"
            f"<span>₹{r['ltp']:.2f} <span style='color:#aab;'>(mid ₹{r['mid']:.2f})</span></span></div>"
            for r in fall)
        card(f"<b style='color:{RED_BD};'>FALL ⬇️ — LTP at its bearish VOB (resistance) · avoid buy / exit</b>{rows}",
             RED_BG, RED_BD)

# ── LTP board: ATM±3 CE / PE ────────────────────────────────────────────────
legs = snap.get('legs') or []
if legs:
    st.markdown("#### 💰 LTP board — ATM±3")
    srb = snap.get('sr_behavior') or {}

    def _leg_rows(side):
        out = ""
        for l in legs:
            tag = l.get('tag', '')
            if f" {side} " not in f" {tag} ":
                continue
            mfp = (l.get('mfp') or '—')
            _, mbd = _tone(mfp)
            beh = ""
            for bt, bv in srb.items():
                if bt in tag or tag.endswith(bt) or bt == tag:
                    de = '🟢' if bv.get('direction') == 'bull' else ('🔴' if bv.get('direction') == 'bear' else '⚪')
                    beh = f" {de}{bv.get('state', '')}"
                    break
            out += (f"<div class='vob-row'><span>{tag}{beh}</span>"
                    f"<span><b>₹{l.get('ltp', 0):.2f}</b> "
                    f"<span style='color:{mbd}; font-size:12px;'>{mfp}</span></span></div>")
        return out

    ce_rows = _leg_rows('CE')
    pe_rows = _leg_rows('PE')
    if ce_rows:
        card(f"<b style='color:{GREEN_BD};'>CALLs (CE)</b>{ce_rows}")
    if pe_rows:
        card(f"<b style='color:{RED_BD};'>PUTs (PE)</b>{pe_rows}")

# ── Spot S/R zones + OI walls ───────────────────────────────────────────────
zones = snap.get('sr_zones') or {}
oi = snap.get('oi') or {}
if zones or oi:
    st.markdown("#### 📐 Levels")
    chips = ""
    for lv in zones.get('support') or []:
        chips += (f"<span class='vob-chip' style='background:{GREEN_BG}; "
                  f"border:1px solid {GREEN_BD}; color:{GREEN_BD};'>S ₹{lv:,.0f}</span>")
    for lv in zones.get('resistance') or []:
        chips += (f"<span class='vob-chip' style='background:{RED_BG}; "
                  f"border:1px solid {RED_BD}; color:{RED_BD};'>R ₹{lv:,.0f}</span>")
    oi_line = ""
    if oi:
        cw, pw = oi.get('call_wall') or {}, oi.get('put_wall') or {}
        oi_line = (f"<div class='vob-sub' style='margin-top:6px;'>🧱 CALL wall ₹{cw.get('strike', 0):,.0f} "
                   f"({cw.get('oi', 0)/100000:.1f}L) · PUT wall ₹{pw.get('strike', 0):,.0f} "
                   f"({pw.get('oi', 0)/100000:.1f}L) · PCR {oi.get('pcr', '—')}</div>")
    card((chips or "<span class='vob-sub'>No major zones yet.</span>") + oi_line)

# ── Bias engines by speed ───────────────────────────────────────────────────
engines = snap.get('engines') or []
if engines:
    st.markdown("#### 📊 All bias engines")
    for cat, icon, name in (('fast', '🚀', 'Fast'), ('lag', '🐢', 'Lagging'), ('mis', '🌫️', 'Misguiding')):
        rows = [e for e in engines if e.get('cat') == cat]
        if not rows:
            continue
        v = (snap.get('verdicts') or {}).get(cat) or {}
        title = f"{icon} {name} — {v.get('em', '')} {v.get('label', '')} (net {v.get('net', 0):+d})" \
            if v else f"{icon} {name}"
        with st.expander(title, expanded=(cat == 'fast')):
            for e in rows:
                st.markdown(f"{e.get('bias', '')} **{e.get('engine', '')}** — {e.get('detail', '')}")

# ── Recent alerts ───────────────────────────────────────────────────────────
alerts = snap.get('alerts') or []
if alerts:
    st.markdown("#### 🔔 Recent alerts")
    for a in reversed(alerts):
        head = (a.get('text') or '').split('\n', 1)[0]
        with st.expander(f"{a.get('ts', '')} · {head[:64]}"):
            st.text(a.get('text', ''))

st.caption("📱 VOB Mobile — read-only companion. All data is computed by "
           "vob_minimal.py (keep it running with one tab open); this app just "
           "renders its latest snapshot.")
