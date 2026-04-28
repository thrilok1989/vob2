import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import time
from datetime import datetime, timedelta
import json
import hashlib
import numpy as np
import math
from scipy.stats import norm
from pytz import timezone
import io
import os
from db.supabase_client import SupabaseDB
from indicators.money_flow_profile import calculate_money_flow_profile
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False
try:
    from google import genai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False
from indicators.volume_delta import calculate_volume_delta

st.set_page_config(
    page_title="Nifty Trading & Options Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Only auto-refresh during market hours (8:30 AM - 3:45 PM IST, weekdays)
_ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
_is_market_open = (
    _ist_now.weekday() < 5 and
    _ist_now.replace(hour=8, minute=30, second=0, microsecond=0) <= _ist_now <= _ist_now.replace(hour=15, minute=45, second=0, microsecond=0)
)
if _is_market_open:
    st_autorefresh(interval=30000, key="datarefresh")

st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stSelectbox > div > div > select {
        background-color:
        color: white;
    }
    .metric-container {
        background-color:
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .price-up {
        color:
    }
    .price-down {
        color:
    }
</style>
""", unsafe_allow_html=True)

try:
    DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "") or st.secrets.get("dhan", {}).get("client_id", "")
    DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "") or st.secrets.get("dhan", {}).get("access_token", "")
    supabase_url = st.secrets.get("supabase", {}).get("url", "")
    supabase_key = st.secrets.get("supabase", {}).get("anon_key", "")
    GEMINI_API_KEY = (
        st.secrets.get("GEMINI_API_KEY", "")
        or st.secrets.get("gemini", {}).get("api_key", "")
        or st.secrets.get("gemini", {}).get("GEMINI_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    )
    try:
        TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "") or getattr(st.secrets, "TELEGRAM_BOT_TOKEN", "")
        TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "") or getattr(st.secrets, "TELEGRAM_CHAT_ID", "")
        if isinstance(TELEGRAM_CHAT_ID, (int, float)):
            TELEGRAM_CHAT_ID = str(int(TELEGRAM_CHAT_ID))
    except:
        TELEGRAM_BOT_TOKEN = TELEGRAM_CHAT_ID = ""
except Exception:
    DHAN_CLIENT_ID = DHAN_ACCESS_TOKEN = supabase_url = supabase_key = TELEGRAM_BOT_TOKEN = TELEGRAM_CHAT_ID = GEMINI_API_KEY = ""

NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"

# Instrument configs for multi-instrument capping/OI/volume monitor
INSTRUMENT_CONFIGS = {
    'SENSEX':    {'scrip': 51,   'seg': 'IDX_I',  'lot': 10,  'strike_gap': 100, 'atm_strikes': 4, 'name': 'SENSEX'},
    'BANKNIFTY': {'scrip': 25,   'seg': 'IDX_I',  'lot': 15,  'strike_gap': 100, 'atm_strikes': 4, 'name': 'BANK NIFTY'},
    'RELIANCE':  {'scrip': 2885, 'seg': 'NSE_EQ', 'lot': 250, 'strike_gap': 20,  'atm_strikes': 4, 'name': 'RELIANCE'},
    'ICICIBANK': {'scrip': 4963, 'seg': 'NSE_EQ', 'lot': 700, 'strike_gap': 10,  'atm_strikes': 4, 'name': 'ICICI BANK'},
    'INFOSYS':   {'scrip': 1594, 'seg': 'NSE_EQ', 'lot': 400, 'strike_gap': 25,  'atm_strikes': 4, 'name': 'INFOSYS'},
}

@st.cache_data(ttl=300)
from vob_indicators import *
from vob_data import *
from vob_analysis import *
from vob_alerts import *

def cached_pivot_calculation(df_json, pivot_settings):
    df = pd.read_json(io.StringIO(df_json))
    return PivotIndicator.get_all_pivots(df, pivot_settings)

@st.cache_data(ttl=60)
def cached_iv_average(option_data_json):
    df = pd.read_json(io.StringIO(option_data_json))
    iv_ce_avg = df['impliedVolatility_CE'].mean()
    iv_pe_avg = df['impliedVolatility_PE'].mean()
    return iv_ce_avg, iv_pe_avg


def render_master_signal_image(result, underlying_price, option_data=None):
    """Render the master signal as a dark-themed PNG image using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import io
    plt.switch_backend('Agg')  # safe to call even after matplotlib is already imported

    BG    = '#0d1117'
    BG2   = '#161b22'
    GREEN = '#00e676'
    RED   = '#ff5252'
    YELLOW= '#ffd740'
    CYAN  = '#40c4ff'
    WHITE = '#e6edf3'
    GRAY  = '#8b949e'
    DIM   = '#30363d'

    trade_type = result.get('trade_type', 'NO TRADE')
    signal_str = result.get('signal', '⚪ NO TRADE')
    if 'BUY' in trade_type or 'BREAKOUT' in signal_str.upper():
        sig_clr, sig_bg = GREEN,  '#003a1e'
    elif 'SELL' in trade_type or 'BREAKDOWN' in signal_str.upper():
        sig_clr, sig_bg = RED,    '#3a0000'
    else:
        sig_clr, sig_bg = YELLOW, '#3a3000'

    fig = plt.figure(figsize=(10, 17), facecolor=BG, dpi=110)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    L = 0.025   # left margin
    R = 0.975   # right edge
    W = R - L

    def box(y_top, h, color=BG2, lw=0.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (L, y_top - h), W, h,
            boxstyle="round,pad=0.004",
            facecolor=color, edgecolor=DIM, linewidth=lw,
            transform=ax.transAxes, zorder=1, clip_on=False))

    def t(x, y, s, c=WHITE, sz=8, w='normal', ha='left'):
        ax.text(x, y, str(s), color=c, fontsize=sz, fontweight=w,
                ha=ha, va='top', transform=ax.transAxes,
                zorder=2, clip_on=False)

    def row(label, value, y, lc=GRAY, vc=WHITE, sz=8):
        t(L+0.01, y, label, c=lc, sz=sz, w='bold')
        t(L+0.22, y, value, c=vc, sz=sz)

    # ── HEADER ──────────────────────────────────────────────
    hh = 0.075
    box(0.995, hh, color=sig_bg)
    ist = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M IST')
    t(0.5, 0.99,  'MASTER TRADING SIGNAL',        c=sig_clr, sz=13, w='bold', ha='center')
    t(0.5, 0.965, f'{ist}  |  Spot: ₹{underlying_price:,.2f}', c=WHITE, sz=9, ha='center')
    t(0.5, 0.942, signal_str.replace('⚪','').replace('🟢','').replace('🔴','').strip(),
      c=sig_clr, sz=11, w='bold', ha='center')

    y = 0.995 - hh - 0.006

    # ── CANDLE / VIX / VIDYA ────────────────────────────────
    sh = 0.045
    box(y, sh)
    candle = result.get('candle',{}) or {}
    vol    = result.get('volume',{}) or {}
    vidya  = result.get('vidya', {}) or {}
    vix_d  = result.get('vix',   {}) or {}
    t(L+0.01, y-0.004, f"Candle: {candle.get('pattern','N/A')} ({candle.get('direction','N/A')})  Vol: {vol.get('label','N/A')} ({vol.get('ratio',0)}x)", c=GRAY, sz=8)
    t(L+0.01, y-0.020, f"VIDYA: {vidya.get('trend','N/A')} | Delta:{vidya.get('delta_pct',0):+.0f}%   VIX: {vix_d.get('vix','N/A')} ({vix_d.get('direction','N/A')})", c=WHITE, sz=8)
    loc = ', '.join(result.get('location', []) or [])
    t(L+0.01, y-0.033, f"Loc: {loc[:90]}", c=CYAN, sz=7.5)
    y -= sh + 0.005

    # ── LEVELS ──────────────────────────────────────────────
    sh = 0.055
    box(y, sh)
    t(L+0.01, y-0.004, 'KEY LEVELS', c=CYAN, sz=8.5, w='bold')
    res_lvls = result.get('resistance_levels',[]) or []
    sup_lvls = result.get('support_levels',  []) or []
    t(L+0.01, y-0.018, 'R: ' + '  '.join(f'₹{r:,.0f}' for r in res_lvls[:5]), c=RED,   sz=8.5)
    t(L+0.01, y-0.031, 'S: ' + '  '.join(f'₹{s:,.0f}' for s in sup_lvls[:5]), c=GREEN, sz=8.5)
    pcr_snap = getattr(st.session_state, '_pcr_sr_snapshot', [])
    res_p = [s['level'] for s in pcr_snap if 'Resistance' in s.get('type','')]
    sup_p = [s['level'] for s in pcr_snap if 'Support'    in s.get('type','')]
    ceil_ = f"₹{min(res_p):,.0f}" if res_p else '—'
    floor_= f"₹{max(sup_p):,.0f}" if sup_p else '—'
    t(L+0.01, y-0.044, f"PCR Ceil: {ceil_}   PCR Floor: {floor_}", c=GRAY, sz=7.5)
    y -= sh + 0.005

    # ── GEX ─────────────────────────────────────────────────
    sh = 0.042
    box(y, sh)
    gex = result.get('gex',{}) or {}
    t(L+0.01, y-0.004, 'GEX', c=CYAN, sz=8.5, w='bold')
    flip = f"₹{int(gex.get('gamma_flip',0))}" if gex.get('gamma_flip') else 'N/A'
    mag  = f"₹{int(gex.get('magnet',0))}"     if gex.get('magnet')     else 'N/A'
    rep  = f"₹{int(gex.get('repeller',0))}"   if gex.get('repeller')   else 'N/A'
    side = 'Above' if gex.get('above_flip') else 'Below' if gex.get('above_flip') is not None else 'N/A'
    t(L+0.01, y-0.017, f"Net:{gex.get('net_gex',0):+.0f}L  ATM:{gex.get('atm_gex',0):+.0f}L  Flip:{flip}({side})  Mode:{gex.get('market_mode','N/A')}", c=WHITE, sz=8)
    t(L+0.01, y-0.030, f"Magnet:{mag}  Repeller:{rep}  PCRxGEX: {result.get('pcr_gex',{}).get('badge','N/A')}", c=GRAY, sz=7.5)
    y -= sh + 0.005

    # ── VPFR ────────────────────────────────────────────────
    vpfr = result.get('vpfr',{}) or {}
    if vpfr:
        sh = 0.058
        box(y, sh)
        t(L+0.01, y-0.004, 'VPFR LEVELS (POC | VAH | VAL)', c=CYAN, sz=8.5, w='bold')
        hx = [L+0.01, L+0.18, L+0.45, L+0.65, L+0.82]
        for hv, hc in zip(['','Timeframe','POC','VAH','VAL'], [GRAY,GRAY,GRAY,GRAY,GRAY]):
            t(hx[['','Timeframe','POC','VAH','VAL'].index(hv)], y-0.017, hv, c=hc, sz=7, w='bold')
        ry = y - 0.028
        for tf, lbl in [('short','30 bars'),('medium','60 bars'),('long','180 bars')]:
            vd = vpfr.get(tf)
            if vd:
                t(hx[1], ry, lbl,                  c=GRAY,   sz=7.5)
                t(hx[2], ry, f"₹{vd['poc']:,.0f}", c=YELLOW, sz=7.5)
                t(hx[3], ry, f"₹{vd['vah']:,.0f}", c=RED,    sz=7.5)
                t(hx[4], ry, f"₹{vd['val']:,.0f}", c=GREEN,  sz=7.5)
                ry -= 0.012
        y -= sh + 0.005

    # ── PCR S/R ─────────────────────────────────────────────
    if pcr_snap:
        sh = 0.012 * len(pcr_snap) + 0.025
        box(y, sh)
        t(L+0.01, y-0.004, 'PCR S/R', c=CYAN, sz=8.5, w='bold')
        ry = y - 0.016
        for s in pcr_snap:
            tp = s.get('type','')
            tc = RED if 'Res' in tp else GREEN if 'Sup' in tp else GRAY
            off_s = f"{s['offset']:+.0f}" if s['offset'] != 0 else '0'
            t(L+0.01, ry, f"{s['label']}  ₹{s['strike']:.0f}  PCR:{s['pcr']:.2f}  {tp.replace('🔴','').replace('🟢','').replace('⚪','').strip()}  ₹{s['level']:.0f} (off:{off_s})", c=tc, sz=7.5)
            ry -= 0.011
        y -= sh + 0.005

    # ── OC BIAS TABLE ───────────────────────────────────────
    df_sum = (option_data or {}).get('df_summary') if option_data else None
    if df_sum is not None and not df_sum.empty and 'Strike' in df_sum.columns:
        srt = sorted(df_sum['Strike'].unique())
        atm = None
        if 'Zone' in df_sum.columns:
            ar = df_sum[df_sum['Zone']=='ATM']
            if not ar.empty:
                atm = int(ar.iloc[0]['Strike'])
        if atm is None:
            atm = min(srt, key=lambda s: abs(s - underlying_price))
        ai = srt.index(atm) if atm in srt else -1

        sh = 0.073
        box(y, sh)
        t(L+0.01, y-0.004, 'OC BIAS  (COI | V | D | G | T | Ask | Bid | IV | DEX | GEX)', c=CYAN, sz=8, w='bold')
        hdr_x = [L+0.01,L+0.14,L+0.28,L+0.42, L+0.50,L+0.56,L+0.61,L+0.66,L+0.71,L+0.76,L+0.81,L+0.86,L+0.91]
        for hv,hx2 in zip(['Label','Strike','Verdict','Score','COI','V','D','G','T','Ask','Bid','IV','DEX/GEX'],hdr_x):
            t(hx2, y-0.017, hv, c=GRAY, sz=6.5, w='bold')
        ry = y - 0.028
        for off,lbl in [(2,'ATM+2'),(1,'ATM+1'),(0,'ATM'),(-1,'ATM-1'),(-2,'ATM-2')]:
            idx = ai + off
            if not (0 <= idx < len(srt)):
                continue
            sk  = srt[idx]
            row_ = df_sum[df_sum['Strike']==sk]
            if row_.empty: continue
            r_ = row_.iloc[0]
            g_ = lambda c: (r_[c] if c in r_.index else 'N/A')
            verd = str(g_('Verdict'))
            sc   = g_('BiasScore')
            vc   = RED if 'Bear' in verd else GREEN if 'Bull' in verd else YELLOW
            t(hdr_x[0],  ry, lbl,           c=GRAY, sz=7)
            t(hdr_x[1],  ry, f"₹{sk:.0f}",  c=WHITE, sz=7)
            t(hdr_x[2],  ry, verd[:11],     c=vc,   sz=7)
            sc_c = RED if str(sc).startswith('-') else GREEN
            t(hdr_x[3],  ry, str(sc),       c=sc_c, sz=7)
            for i2, bf in enumerate(['ChgOI_Bias','Volume_Bias','Delta_Bias','Gamma_Bias','Theta_Bias','AskQty_Bias','BidQty_Bias','IV_Bias','DeltaExp']):
                bv = str(g_(bf))
                bc2 = GREEN if bv=='Bullish' else RED if bv=='Bearish' else GRAY
                sym = 'B' if bv=='Bullish' else 'S' if bv=='Bearish' else '-'
                t(hdr_x[4+i2], ry, sym, c=bc2, sz=7.5, w='bold')
            # DEX/GEX combined
            dex = str(g_('DeltaExp')); gex2 = str(g_('GammaExp'))
            dg_str = f"{'B' if dex=='Bullish' else 'S'}/{'B' if gex2=='Bullish' else 'S'}"
            dg_c = GREEN if dex=='Bullish' and gex2=='Bullish' else RED if dex=='Bearish' and gex2=='Bearish' else YELLOW
            t(hdr_x[12], ry, dg_str, c=dg_c, sz=7, w='bold')
            # Entry/Scalp/Move below
            entry_line = f"  Entry:{g_('Operator_Entry')}  Scalp:{g_('Scalp_Moment')}  Move:{g_('FakeReal')}  COI:{g_('ChgOI_Cmp')}  OI:{g_('OI_Cmp')}"
            ry -= 0.011
            t(L+0.01, ry, entry_line[:95], c=GRAY, sz=6.5)
            ry -= 0.012
        y -= sh + 0.005

    # ── OI TREND ────────────────────────────────────────────
    oi = result.get('oi_trend',{}) or {}
    if oi:
        sh = 0.038
        box(y, sh)
        t(L+0.01, y-0.004, 'OI TREND', c=CYAN, sz=8.5, w='bold')
        oi_sig = oi.get('signal','N/A')
        oi_c   = GREEN if 'Bull' in oi_sig else RED if 'Bear' in oi_sig else YELLOW
        t(L+0.01, y-0.017, f"CE:{oi.get('ce_activity','N/A')} (OI:{oi.get('ce_oi_pct',0):+.1f}%)  PE:{oi.get('pe_activity','N/A')} (OI:{oi.get('pe_oi_pct',0):+.1f}%)", c=WHITE, sz=8)
        t(L+0.01, y-0.029, f"Signal: {oi_sig}  Sup:{oi.get('support_status','N/A')}  Res:{oi.get('resistance_status','N/A')}", c=oi_c, sz=8)
        y -= sh + 0.005

    # ── ALIGNMENT ───────────────────────────────────────────
    alignment = result.get('alignment',{}) or {}
    if alignment:
        sh = 0.05
        box(y, sh)
        t(L+0.01, y-0.004, 'ALIGNMENT  10m | 1h | Pattern', c=CYAN, sz=8.5, w='bold')
        SN = {'NIFTY 50':'N50','SENSEX':'SENS','BANKNIFTY':'BNF','NIFTY IT':'IT',
              'RELIANCE':'REL','ICICIBANK':'ICICI','INDIA VIX':'VIX','GOLD':'GOLD',
              'CRUDE OIL':'CRUDE','USD/INR':'INR',
              'S&P 500':'SP500','JAPAN 225':'JP225','HANG SENG':'HSI','UK 100':'UK100'}
        PS = {'No Pattern':'NP','Bullish Engulfing':'BullEng','Bearish Engulfing':'BearEng',
              'Hammer':'Ham','Shooting Star':'ShStr','Tweezer Top':'TwTop',
              'Tweezer Bottom':'TwBot','Strong Green Candle':'SGC','Strong Red Candle':'SRC',
              'Doji':'Doji','Inside Bar':'InsBar','Marubozu':'Maru'}
        items = []
        for name in ['NIFTY 50','SENSEX','BANKNIFTY','NIFTY IT','RELIANCE','ICICIBANK','INDIA VIX','GOLD','CRUDE OIL','USD/INR']:
            d = alignment.get(name)
            if not d: continue
            s10  = d.get('sentiment_10m','')
            s1h  = d.get('sentiment_1h', '')
            pat  = (d.get('candle_pattern','') or '').strip()
            cdir = d.get('candle_dir','') or ''
            a10  = '+' if s10=='Bullish' else '-' if s10=='Bearish' else '.'
            a1h  = '+' if s1h=='Bullish' else '-' if s1h=='Bearish' else '.'
            c10  = GREEN if s10=='Bullish' else RED if s10=='Bearish' else GRAY
            c1h  = GREEN if s1h=='Bullish' else RED if s1h=='Bearish' else GRAY
            ps   = PS.get(pat, pat[:6]) if pat and pat not in ('No Pattern','N/A') else 'NP'
            pc   = GREEN if cdir=='Bullish' else RED if cdir=='Bearish' else GRAY
            items.append((SN.get(name,name), a10, c10, a1h, c1h, ps, pc))
        # 2 rows of 5
        for ri, chunk in enumerate([items[:5], items[5:]]):
            rx = L + 0.01
            ry = y - 0.018 - ri * 0.014
            for nm, a10, c10, a1h, c1h, ps, pc in chunk:
                t(rx, ry, nm+':', c=GRAY, sz=7); rx += len(nm)*0.008 + 0.012
                t(rx, ry, a10,    c=c10,  sz=7); rx += 0.011
                t(rx, ry, a1h,    c=c1h,  sz=7); rx += 0.011
                t(rx, ry, ps,     c=pc,   sz=6.5); rx += max(len(ps)*0.009, 0.055)
        y -= sh + 0.005

    # ── MONEY FLOW ──────────────────────────────────────────
    mf = getattr(st.session_state, '_money_flow_data', None)
    if mf and mf.get('rows'):
        sh = 0.036
        box(y, sh)
        poc_p  = mf.get('poc_price',0)
        va_h   = mf.get('value_area_high',0)
        va_l   = mf.get('value_area_low', 0)
        hi_dir = mf.get('highest_sentiment_direction','Neutral')
        hi_p   = mf.get('highest_sentiment_price',0)
        mf_c   = GREEN if hi_dir=='Bullish' else RED if hi_dir=='Bearish' else YELLOW
        t(L+0.01, y-0.004, 'MONEY FLOW', c=CYAN, sz=8.5, w='bold')
        t(L+0.01, y-0.018, f"POC:₹{poc_p:.0f}  VA:₹{va_l:.0f}-₹{va_h:.0f}  Strongest:{hi_dir} @ ₹{hi_p:.0f}", c=mf_c, sz=8)
        y -= sh + 0.005

    # ── FOOTER ──────────────────────────────────────────────
    t(0.5, max(y - 0.01, 0.01), 'Auto-generated signal. Manual verification required.',
      c=GRAY, sz=7, ha='center')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=BG,
                pad_inches=0.15)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Credentials not configured"

    try:
        test_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            return True, "✅ Telegram bot is active and connected"
        else:
            return False, f"❌ Telegram API error: {response.status_code}"

    except Exception as e:
        return False, f"❌ Telegram connection failed: {str(e)}"


class DhanAPI:

def create_candlestick_chart(df, title, interval, show_pivots=True, pivot_settings=None, vob_blocks=None, poc_data=None, swing_data=None, money_flow_data=None, volume_delta_data=None):
    if df.empty:
        return go.Figure()

    # Use 3 rows if volume delta is available (price, delta overlay, volume)
    has_delta = volume_delta_data is not None and volume_delta_data.get('df') is not None
    fig = make_subplots(
        rows=3 if has_delta else 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.2, 0.25] if has_delta else [0.7, 0.3],
        specs=[[{"secondary_y": False}]] * (3 if has_delta else 2),
        subplot_titles=(['', 'Volume Delta', 'Volume'] if has_delta else ['', 'Volume'])
    )

    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Nifty 50',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        ),
        row=1, col=1
    )

    if show_pivots and len(df) > 50:
        try:
            df_json = df.to_json()
            pivots = cached_pivot_calculation(df_json, pivot_settings or {})
            timeframes = {}
            for pivot in pivots:
                tf = pivot['timeframe']
                if tf not in timeframes:
                    timeframes[tf] = {'highs': [], 'lows': [], 'color': pivot['color']}
                if pivot['type'] == 'high':
                    timeframes[tf]['highs'].append((pivot['timestamp'], pivot['value']))
                else:
                    timeframes[tf]['lows'].append((pivot['timestamp'], pivot['value']))
            x_start = df['datetime'].min()
            x_end = df['datetime'].max()
            for tf, data in timeframes.items():
                color = data['color']
                for label, points in [('R', data['highs']), ('S', data['lows'])]:
                    for _, val in points[-3:]:
                        fig.add_shape(type="line", x0=x_start, x1=x_end, y0=val, y1=val,
                                      line=dict(color=color, width=1, dash="dash"), row=1, col=1)
                        fig.add_annotation(x=x_end, y=val, text=f"{tf} {label} {val:.1f}",
                                           showarrow=False, font=dict(color=color, size=10), xanchor="left", row=1, col=1)
        except Exception as e:
            st.warning(f"Error adding pivot levels: {str(e)}")

    try:
        vwap = ReversalDetector.calculate_vwap(df)
        if not vwap.empty:
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=vwap,
                    mode='lines',
                    name='VWAP',
                    line=dict(color='#FFD700', width=2, dash='dot'),
                    opacity=0.8
                ),
                row=1, col=1
            )
    except Exception as e:
        pass

    if vob_blocks:
        try:
            x_start = df['datetime'].min()
            x_end = df['datetime'].max()
            for btype, color, fill, arrow in [
                ('bullish', '#26ba9f', 'rgba(38, 186, 159, 0.15)', '↑'),
                ('bearish', '#6626ba', 'rgba(102, 38, 186, 0.15)', '↓')
            ]:
                for block in vob_blocks.get(btype, [])[-5:]:
                    fig.add_shape(type="rect", x0=x_start, x1=x_end, y0=block['lower'], y1=block['upper'],
                                  fillcolor=fill, line=dict(color=color, width=2), row=1, col=1)
                    fig.add_shape(type="line", x0=x_start, x1=x_end, y0=block['mid'], y1=block['mid'],
                                  line=dict(color=color, width=1, dash="dash"), row=1, col=1)
                    vol_text = VolumeOrderBlocks.format_volume(block['volume'])
                    fig.add_annotation(x=x_end, y=block['mid'], text=f"VOB{arrow} {vol_text} ({block['volume_pct']:.0f}%)",
                                       showarrow=False, font=dict(color=color, size=9), xanchor="left", row=1, col=1)
        except Exception as e:
            pass

    if poc_data:
        try:
            x_start = df['datetime'].min()
            x_end = df['datetime'].max()
            for poc_key, color in [('poc1', '#e91e63'), ('poc2', '#2196f3'), ('poc3', '#4caf50')]:
                poc = poc_data.get(poc_key)
                if poc and poc.get('poc'):
                    period = poc_data.get('periods', {}).get(poc_key, '')
                    fig.add_shape(type="line", x0=x_start, x1=x_end, y0=poc['poc'], y1=poc['poc'],
                                  line=dict(color=color, width=2), row=1, col=1)
                    fig.add_annotation(x=x_end, y=poc['poc'], text=f"POC{poc_key[-1]} ({period}): ₹{poc['poc']:.0f}",
                                       showarrow=False, font=dict(color=color, size=10), xanchor="left", row=1, col=1)
        except Exception as e:
            pass

    if swing_data and swing_data.get('swings'):
        try:
            swings = swing_data['swings']
            projection = swing_data.get('projection')
            x_end = df['datetime'].max()
            atr_est = (df['high'].max() - df['low'].min()) * 0.02
            for key, color, fill, label, y0_off, y1_off in [
                ('last_swing_high', '#eb7514', 'rgba(235, 117, 20, 0.2)', 'H', 0, atr_est),
                ('last_swing_low', '#15dd7c', 'rgba(21, 221, 124, 0.2)', 'L', -atr_est, 0)
            ]:
                pt = swings.get(key)
                if pt and pt.get('value'):
                    x0 = df['datetime'].iloc[pt['index']] if pt['index'] < len(df) else x_end
                    fig.add_shape(type="rect", x0=x0, x1=x_end, y0=pt['value']+y0_off, y1=pt['value']+y1_off,
                                  fillcolor=fill, line=dict(color=color, width=1), row=1, col=1)
                    fig.add_annotation(x=x_end, y=pt['value'], text=f"Swing {label}: ₹{pt['value']:.0f}",
                                       showarrow=False, font=dict(color=color, size=9), xanchor="left", row=1, col=1)
            if projection:
                from_value = projection['from_value']
                target = projection['target']
                direction = projection['direction']
                proj_color = "#15dd7c" if direction == 'bullish' else "#eb7514"
                fig.add_shape(
                    type="line",
                    x0=x_end,
                    x1=x_end,
                    y0=from_value,
                    y1=target,
                    line=dict(color=proj_color, width=2, dash="dash"),
                    row=1, col=1
                )
                fig.add_annotation(
                    x=x_end,
                    y=target,
                    text=f"Target: {projection['sign']}{projection['swing_pct']:.1f}% → ₹{target:.0f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=proj_color,
                    font=dict(color=proj_color, size=10),
                    xanchor="left",
                    row=1, col=1
                )
        except Exception as e:
            pass

    # ── Money Flow Profile overlay on price chart ──
    if money_flow_data and money_flow_data.get('rows'):
        try:
            mf_rows = money_flow_data['rows']
            poc_price = money_flow_data['poc_price']
            va_high = money_flow_data['value_area_high']
            va_low = money_flow_data['value_area_low']
            x_start = df['datetime'].min()
            x_end = df['datetime'].max()

            # Draw horizontal bars as shapes on the right side of the chart
            max_vol = max(r['total_volume'] for r in mf_rows) if mf_rows else 1
            for row in mf_rows:
                ratio = row['total_volume'] / max_vol if max_vol > 0 else 0
                if ratio < 0.05:
                    continue
                # Color by node type
                if row['node_type'] == 'High':
                    fill_color = 'rgba(255, 235, 59, 0.25)'
                elif row['node_type'] == 'Low':
                    fill_color = 'rgba(242, 54, 69, 0.15)'
                else:
                    fill_color = 'rgba(41, 98, 255, 0.18)'

                fig.add_shape(type="rect", x0=x_start, x1=x_end,
                              y0=row['bin_low'], y1=row['bin_high'],
                              fillcolor=fill_color, line=dict(width=0),
                              layer='below', row=1, col=1)

            # POC line
            fig.add_shape(type="line", x0=x_start, x1=x_end,
                          y0=poc_price, y1=poc_price,
                          line=dict(color='#ffeb3b', width=2, dash='solid'),
                          row=1, col=1)
            fig.add_annotation(x=x_end, y=poc_price,
                               text=f"MF-POC ₹{poc_price:.0f}",
                               showarrow=False, font=dict(color='#ffeb3b', size=10),
                               xanchor="left", row=1, col=1)

            # Value area boundaries
            fig.add_shape(type="line", x0=x_start, x1=x_end,
                          y0=va_high, y1=va_high,
                          line=dict(color='#2962ff', width=1, dash='dot'),
                          row=1, col=1)
            fig.add_shape(type="line", x0=x_start, x1=x_end,
                          y0=va_low, y1=va_low,
                          line=dict(color='#2962ff', width=1, dash='dot'),
                          row=1, col=1)
            fig.add_annotation(x=x_start, y=va_high,
                               text=f"VA-H ₹{va_high:.0f}",
                               showarrow=False, font=dict(color='#2962ff', size=9),
                               xanchor="right", row=1, col=1)
            fig.add_annotation(x=x_start, y=va_low,
                               text=f"VA-L ₹{va_low:.0f}",
                               showarrow=False, font=dict(color='#2962ff', size=9),
                               xanchor="right", row=1, col=1)
        except Exception:
            pass

    # ── Volume Delta Candles (row 2) ──
    vol_row = 3 if has_delta else 2
    if has_delta:
        try:
            delta_df = volume_delta_data['df']
            delta_colors = ['#089981' if d > 0 else '#f23645' for d in delta_df['delta'].values]
            # Divergence: bullish candle but negative delta (or vice versa) - use dimmed opposite color
            for i in range(len(delta_df)):
                row_d = delta_df.iloc[i]
                if row_d['divergence']:
                    delta_colors[i] = '#f2364580' if row_d['bar_type'] == 'bullish' else '#08998180'

            fig.add_trace(
                go.Bar(
                    x=delta_df['datetime'],
                    y=delta_df['delta'],
                    name='Volume Delta',
                    marker_color=delta_colors,
                    opacity=0.85,
                    showlegend=False,
                    hovertemplate='Delta: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
            fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, row=2, col=1)

            # Cumulative delta line overlay
            fig.add_trace(
                go.Scatter(
                    x=delta_df['datetime'],
                    y=delta_df['cum_delta'],
                    mode='lines',
                    name='Cum Delta',
                    line=dict(color='#FFD700', width=1.5),
                    opacity=0.7,
                    yaxis='y4' if has_delta else 'y3',
                    showlegend=False,
                    hovertemplate='Cum Δ: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1, secondary_y=False
            )

            # Max volume price dots on the price chart
            fig.add_trace(
                go.Scatter(
                    x=delta_df['datetime'],
                    y=delta_df['max_vol_price'],
                    mode='markers',
                    name='Max Vol Price',
                    marker=dict(size=3, color='white', opacity=0.4),
                    showlegend=False,
                    hovertemplate='Max Vol: ₹%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        except Exception:
            pass

    volume_colors = ['#00ff88' if close >= open else '#ff4444'
                    for close, open in zip(df['close'], df['open'])]

    fig.add_trace(
        go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7,
            showlegend=False
        ),
        row=vol_row, col=1
    )

    chart_height = 850 if has_delta else 700
    fig.update_layout(
        title=title,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=chart_height,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color='white'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e'
    )

    _gc = dict(showgrid=True, gridwidth=1, gridcolor='#333333')
    fig.update_xaxes(title_text="Time (IST)", type='date', row=vol_row, col=1, **_gc)
    fig.update_xaxes(type='date', row=1, col=1, **_gc)
    fig.update_yaxes(title_text="Price (₹)", side='left', row=1, col=1, **_gc)
    fig.update_yaxes(title_text="Volume", side='left', row=vol_row, col=1, **_gc)
    if has_delta:
        fig.update_xaxes(type='date', row=2, col=1, **_gc)
        fig.update_yaxes(title_text="Delta", side='left', row=2, col=1, **_gc)

    return fig

def display_metrics(ltp_data, df, db, symbol="NIFTY50"):

def _detect_candle_pattern(df, label):
    """Detect candle pattern from last candle in dataframe. Returns confirmation string."""
    if df is None or df.empty or len(df) < 2:
        return None
    try:
        c = df.iloc[-1]
        o, h, l, cl = float(c['Open']), float(c['High']), float(c['Low']), float(c['Close'])
        body = abs(cl - o)
        rng = h - l if h > l else 1
        upper_wick = h - max(o, cl)
        lower_wick = min(o, cl) - l
        is_bullish = cl > o
        is_bearish = cl < o
        if lower_wick > body * 2 and upper_wick < body:
            pat = "Hammer ✅" if is_bullish else "Hanging Man ✅"
        elif upper_wick > body * 2 and lower_wick < body:
            pat = "Shooting Star ✅" if is_bearish else "Inv Hammer ✅"
        elif body / rng > 0.6 and is_bullish:
            pat = "Strong Green ✅"
        elif body / rng > 0.6 and is_bearish:
            pat = "Strong Red ✅"
        else:
            pat = "Doji/Inside ⚪"
        return f"{label} Candle: {pat}"
    except Exception:
        return None


def _analyze_zone(spot, zone_bottom, zone_top, option_data, df_5m, df_1m=None):
    """Analyze market conditions inside a S/R zone. Returns list of confirmation signals."""
    confirmations = []

    # 1. OI check — use correct Dhan API column names
    try:
        if option_data:
            sg = option_data.get('df_summary')
            if sg is not None and not sg.empty and 'Strike' in sg.columns:
                strikes_in_zone = sg[(sg['Strike'] >= zone_bottom) & (sg['Strike'] <= zone_top)]
                if not strikes_in_zone.empty:
                    ce_col = next((c for c in ['changeinOpenInterest_CE', 'CE_ChgOI', 'chgOI_CE'] if c in strikes_in_zone.columns), None)
                    pe_col = next((c for c in ['changeinOpenInterest_PE', 'PE_ChgOI', 'chgOI_PE'] if c in strikes_in_zone.columns), None)
                    ce_chg = float(strikes_in_zone[ce_col].sum()) if ce_col else 0
                    pe_chg = float(strikes_in_zone[pe_col].sum()) if pe_col else 0
                    if pe_chg > 0 and pe_chg > ce_chg:
                        confirmations.append("OI: PE writing ↑ (floor holding) ✅")
                    elif ce_chg > 0 and ce_chg > pe_chg:
                        confirmations.append("OI: CE writing ↑ (ceiling holding) ✅")
                    else:
                        confirmations.append("OI: Mixed ⚪")
    except Exception:
        pass

    # 2. Depth check — use bidQty/askQty from df_summary for strikes in zone
    try:
        if option_data:
            sg = option_data.get('df_summary')
            if sg is not None and not sg.empty and 'Strike' in sg.columns:
                strikes_in_zone = sg[(sg['Strike'] >= zone_bottom) & (sg['Strike'] <= zone_top)]
                if not strikes_in_zone.empty:
                    bid_ce = float(strikes_in_zone.get('bidQty_CE', pd.Series([0])).sum()) if 'bidQty_CE' in strikes_in_zone.columns else 0
                    ask_ce = float(strikes_in_zone.get('askQty_CE', pd.Series([0])).sum()) if 'askQty_CE' in strikes_in_zone.columns else 0
                    bid_pe = float(strikes_in_zone.get('bidQty_PE', pd.Series([0])).sum()) if 'bidQty_PE' in strikes_in_zone.columns else 0
                    ask_pe = float(strikes_in_zone.get('askQty_PE', pd.Series([0])).sum()) if 'askQty_PE' in strikes_in_zone.columns else 0
                    # Support zone: strong bid on PE side = buyers defending
                    # Resistance zone: strong ask on CE side = sellers defending
                    if bid_pe > ask_pe * 1.2:
                        confirmations.append("Depth: PE Bid wall (support defending) ✅")
                    elif ask_ce > bid_ce * 1.2:
                        confirmations.append("Depth: CE Ask wall (resistance defending) ✅")
                    elif bid_ce > ask_ce * 1.2:
                        confirmations.append("Depth: CE Bid wall (breakout pressure) ✅")
                    else:
                        confirmations.append("Depth: Balanced ⚪")
    except Exception:
        pass

    # 3. Delta Volume check — from 5m candle (always available)
    try:
        df_vol = df_1m if (df_1m is not None and not df_1m.empty) else df_5m
        if df_vol is not None and not df_vol.empty and len(df_vol) >= 2:
            last = df_vol.iloc[-1]
            vol = float(last.get('Volume', 0) or 0)
            avg_vol = float(df_vol['Volume'].mean()) if 'Volume' in df_vol.columns else 0
            # Check last 3 candles for volume trend
            last3_vol = df_vol['Volume'].iloc[-3:].mean() if len(df_vol) >= 3 and 'Volume' in df_vol.columns else vol
            prev_avg = df_vol['Volume'].iloc[:-3].mean() if len(df_vol) > 3 and 'Volume' in df_vol.columns else avg_vol
            close_trend = df_vol['Close'].iloc[-3:].diff().sum() if len(df_vol) >= 3 and 'Close' in df_vol.columns else 0
            if last3_vol > prev_avg * 1.3 and close_trend > 0:
                confirmations.append("Delta Vol: Buying surge ✅")
            elif last3_vol > prev_avg * 1.3 and close_trend < 0:
                confirmations.append("Delta Vol: Selling surge ✅")
            elif vol > avg_vol * 1.3:
                confirmations.append("Delta Vol: Volume spike ✅")
            else:
                confirmations.append("Delta Vol: Normal ⚪")
    except Exception:
        pass

    # 4. ATM OI Activity — from latest master signal oi_trend
    try:
        master = getattr(st.session_state, '_master_signal_latest', None)
        if master:
            oit = master.get('oi_trend', {}) or {}
            atm_s = oit.get('atm_strike', '')
            ce_act = oit.get('ce_activity', 'N/A')
            pe_act = oit.get('pe_activity', 'N/A')
            ce_pct = oit.get('ce_oi_pct', 0)
            pe_pct = oit.get('pe_oi_pct', 0)
            # Bullish at support: PE Short Building + CE weakening
            # Bearish at resistance: CE Short Building + PE weakening
            bullish_oi = pe_act in ('Short Building', 'Long Building') and ce_act in ('Short Covering', 'Long Unwinding')
            bearish_oi = ce_act in ('Short Building', 'Long Building') and pe_act in ('Short Covering', 'Long Unwinding')
            if bullish_oi:
                confirmations.append(f"ATM OI ({atm_s}): PE {pe_act}({pe_pct:+.1f}%) CE {ce_act} → Bullish ✅")
            elif bearish_oi:
                confirmations.append(f"ATM OI ({atm_s}): CE {ce_act}({ce_pct:+.1f}%) PE {pe_act} → Bearish ✅")
            else:
                confirmations.append(f"ATM OI ({atm_s}): CE {ce_act}({ce_pct:+.1f}%) | PE {pe_act}({pe_pct:+.1f}%) ⚪")
    except Exception:
        pass

    # 5a. 5-min candle pattern
    pat_5m = _detect_candle_pattern(df_5m, "5m")
    if pat_5m:
        confirmations.append(pat_5m)

    # 5b. 1-min candle pattern (if available)
    if df_1m is not None and not df_1m.empty:
        pat_1m = _detect_candle_pattern(df_1m, "1m")
        if pat_1m:
            confirmations.append(pat_1m)

    return confirmations


def _get_zone_telegram_msg(spot, zone_type, zone_bottom, zone_top, confirmations, strike, call_entry, call_target, call_sl, put_entry, put_target, put_sl):
    """Build short Telegram zone alert message."""
    score = sum(1 for c in confirmations if '✅' in c)
    total = len(confirmations)
    emoji = "🟢" if zone_type == "SUPPORT" else "🔴"
    trade_type = "CALL" if zone_type == "SUPPORT" else "PUT"
    time_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M IST')
    return f"{emoji} SPOT IN {zone_type} ZONE | ₹{spot:.0f} | Zone ₹{zone_bottom:.0f}–₹{zone_top:.0f} | {score}/{total} confirm | {time_str}"


def _place_trade(api, trade_type, strike, security_id, entry_price, target, sl, lot_size, spot, confirmations, db):
    """Place order on Dhan and store in Supabase."""
    # Dhan NSE_FNO segment, 1 lot = 25 qty for Nifty (weekly)
    qty = lot_size * 25
    result = api.place_order(
        security_id=security_id,
        exchange_segment="NSE_FNO",
        transaction_type="BUY",
        quantity=qty,
        order_type="MARKET",
    )
    order_id = None
    error = result.get('error') if result else "No response"
    if result and not result.get('error'):
        order_id = result.get('orderId') or result.get('order_id') or str(result)

    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    trade = {
        'trade_type': trade_type,
        'strike': str(strike),
        'security_id': str(security_id),
        'entry_price': entry_price,
        'target': target,
        'sl': sl,
        'lot_size': lot_size,
        'status': 'OPEN',
        'order_id': order_id,
        'entry_time': now.isoformat(),
        'spot_at_entry': spot,
        'zone_confirmations': " | ".join(confirmations),
    }
    saved = db.upsert_auto_trade(trade)
    return saved, order_id, error


def _exit_trade(api, trade, exit_reason, db):
    """Place exit order on Dhan and update Supabase."""
    qty = int(trade.get('lot_size', 1)) * 25
    security_id = trade.get('security_id')
    result = api.place_order(
        security_id=security_id,
        exchange_segment="NSE_FNO",
        transaction_type="SELL",
        quantity=qty,
        order_type="MARKET",
    )
    exit_price = None
    if result and not result.get('error'):
        exit_price = result.get('averagePrice') or result.get('price')

    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    updated = {**trade, 'status': 'CLOSED', 'exit_reason': exit_reason, 'exit_time': now.isoformat()}
    if exit_price:
        updated['exit_price'] = float(exit_price)
    db.upsert_auto_trade(updated)
    return updated, exit_price


def _render_alignment_capping_top():
    """Render Alignment + Index/Stock Capping panels from session state."""
    _master = getattr(st.session_state, '_master_signal_latest', None)
    _sa = getattr(st.session_state, '_sa_result', None)

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
        'Inside Bar':'InsBar','Outside Bar':'OutBar','Pin Bar':'PinBar','Marubozu':'Maru',
    }
    def _ae(s): return '🟢' if s == 'Bullish' else '🔴' if s == 'Bearish' else '⚪'

    _ac_col1, _ac_col2 = st.columns([1, 1])

    # ── Alignment panel ──
    with _ac_col1:
        with st.expander("🌍 Alignment (10m|1h|4h|1D|4D|Pat)", expanded=True):
            if _master is None:
                st.info("Alignment data loads after first signal calculation.")
            else:
                _alignment = _master.get('alignment', {})
                if not _alignment:
                    st.info("No alignment data yet.")
                else:
                    _rows = []
                    for name in ['NIFTY 50','SENSEX','BANKNIFTY','NIFTY IT','RELIANCE','ICICIBANK','INDIA VIX','GOLD','CRUDE OIL','USD/INR','S&P 500','JAPAN 225','HANG SENG','UK 100']:
                        data = _alignment.get(name)
                        if data is None:
                            continue
                        e10 = _ae(data.get('sentiment_10m', ''))
                        e1h = _ae(data.get('sentiment_1h', ''))
                        e4h = _ae(data.get('sentiment_4h', ''))
                        e1d = _ae(data.get('sentiment_1d', ''))
                        e4d = _ae(data.get('sentiment_4d', ''))
                        pat = (data.get('candle_pattern', '') or '').strip()
                        cdir = data.get('candle_dir', '') or ''
                        if not pat or pat in ('No Pattern', 'N/A'):
                            pat_str = 'NP'
                            pat_color = '#888'
                        else:
                            p_short = _pat_short.get(pat, pat[:7])
                            p_emoji = '🟢' if cdir == 'Bullish' else '🔴' if cdir == 'Bearish' else '⚪'
                            pat_str = f"{p_short}{p_emoji}"
                            pat_color = '#00cc66' if cdir == 'Bullish' else '#ff4444' if cdir == 'Bearish' else '#aaa'
                        sn = _short_names.get(name, name)
                        _rows.append((sn, e10, e1h, e4h, e1d, e4d, pat_str, pat_color))
                    for sn, e10, e1h, e4h, e1d, e4d, pat_str, pat_color in _rows:
                        st.markdown(
                            f'<div style="font-family:monospace;font-size:13px;padding:2px 0;">'
                            f'<b style="min-width:50px;display:inline-block;">{sn}</b> '
                            f'{e10}{e1h}{e4h}{e1d}{e4d} '
                            f'<span style="color:{pat_color};font-size:12px;">{pat_str}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # ── Index/Stock Capping panel ──
    with _ac_col2:
        with st.expander("📡 Index/Stock Capping", expanded=True):
            if _sa is None:
                st.info("Capping data loads after option chain analysis.")
            else:
                try:
                    # Rebuild capping lines same as Telegram message
                    _underlying = _master.get('underlying_price', 0) if _master else 0
                    _sa_bias = _sa or {}
                    _adf = _sa_bias.get('analysis_df')

                    def _cap_badge(label, bias, spot, caps, sups):
                        b_em = '🟢' if bias == 'Bullish' else '🔴' if bias == 'Bearish' else '⚪'
                        parts = [f"{b_em}{label} ₹{spot:.0f}"]
                        for c in (caps or [])[:2]:
                            oi_l = c.get('oi_l', 0)
                            fire = '🔥' if oi_l > 0 else ''
                            parts.append(f"🟥R₹{c['strike']:.0f}({oi_l:.0f}L){fire}")
                        for s in (sups or [])[:2]:
                            oi_l = s.get('oi_l', 0)
                            fire = '🔥' if oi_l > 0 else ''
                            parts.append(f"🟩S₹{s['strike']:.0f}({oi_l:.0f}L){fire}")
                        return " ".join(parts)

                    lines = []
                    if _adf is not None and not _adf.empty and _underlying:
                        _res = _adf[_adf['Strike'] > _underlying].sort_values('Strike').head(2)
                        _sup = _adf[_adf['Strike'] < _underlying].sort_values('Strike', ascending=False).head(2)
                        _caps_n50 = [{'strike': float(r['Strike']), 'oi_l': float(r.get('CE_OI',0) or 0)/100000} for _, r in _res.iterrows()]
                        _sups_n50 = [{'strike': float(r['Strike']), 'oi_l': float(r.get('PE_OI', r.get('openInterest_PE',0)) or 0)/100000} for _, r in _sup.iterrows()]
                        lines.append(_cap_badge('N50', _sa_bias.get('market_bias',''), _underlying, _caps_n50, _sups_n50))

                    # Stock capping from alignment data
                    _stocks = [
                        ('SENX', 'SENSEX'), ('BNF', 'BANKNIFTY'), ('REL', 'RELIANCE'),
                        ('ICICI', 'ICICIBANK'), ('INFO', 'NIFTY IT'),
                    ]
                    _stock_data = _sa_bias.get('stock_data', {})
                    for sn, full_name in _stocks:
                        _sd = _stock_data.get(full_name) or _stock_data.get(sn) or {}
                        _bias = _sd.get('bias', '')
                        _und = _sd.get('ltp', 0) or 0
                        _scaps = _sd.get('caps', [])
                        _ssups = _sd.get('sups', [])
                        if _und:
                            lines.append(_cap_badge(sn, _bias, _und, _scaps, _ssups))

                    if lines:
                        for ln in lines:
                            st.markdown(f'<div style="font-family:monospace;font-size:13px;padding:2px 0;">{ln}</div>', unsafe_allow_html=True)
                    else:
                        # Fallback: show raw market bias
                        bias = _sa_bias.get('market_bias', '—')
                        b_em = '🟢' if bias == 'Bullish' else '🔴' if bias == 'Bearish' else '⚪'
                        st.markdown(f"**N50 bias:** {b_em} {bias}")
                        if _adf is not None and not _adf.empty:
                            _top_r = _adf[_adf.get('Call_Class', pd.Series(dtype=str)).isin(['High Conviction Resistance','Strong Resistance'])].head(3) if 'Call_Class' in _adf.columns else pd.DataFrame()
                            _top_s = _adf[_adf.get('Put_Class', pd.Series(dtype=str)).isin(['High Conviction Support','Strong Support'])].head(3) if 'Put_Class' in _adf.columns else pd.DataFrame()
                            if not _top_r.empty:
                                st.markdown("**🔴 Resistance:** " + " | ".join([f"₹{int(r['Strike'])}" for _, r in _top_r.iterrows()]))
                            if not _top_s.empty:
                                st.markdown("**🟢 Support:** " + " | ".join([f"₹{int(r['Strike'])}" for _, r in _top_s.iterrows()]))
                except Exception as _e:
                    st.caption(f"Capping render error: {_e}")

    # ── Sector Rotation panel (full width below alignment+capping) ──
    _sr_data = getattr(st.session_state, '_sector_rotation', None)
    with st.expander("🔄 Sector Rotation & Bias (10m | 1h)", expanded=False):
        if _sr_data is None:
            st.info("Sector rotation data loads with alignment (every 2 min).")
        else:
            _bias = _sr_data.get('rotation_bias', '—')
            st.markdown(f"**Rotation Bias:** {_bias}")
            _all = _sr_data.get('all', [])
            if _all:
                _s10_e = lambda s: '🟢' if s == 'Bullish' else '🔴' if s == 'Bearish' else '⚪'
                _rows_md = []
                for r in _all:
                    _chg = r.get('day_chg_pct', 0)
                    _chg_e = '🟢' if _chg > 0 else '🔴' if _chg < 0 else '⚪'
                    _rows_md.append(
                        f'<div style="font-family:monospace;font-size:13px;padding:1px 0;">'
                        f'<b style="display:inline-block;min-width:70px;">{r["name"]}</b> '
                        f'{_chg_e}{_chg:+.2f}% &nbsp; 10m:{_s10_e(r["s10"])} 1h:{_s10_e(r["s1h"])}'
                        f'</div>'
                    )
                st.markdown("\n".join(_rows_md), unsafe_allow_html=True)


def _render_vol_delta_chart():
    """Render Buy Volume vs Sell Volume over time, same style as per-strike OI chart."""
    import plotly.graph_objects as go
    vd = getattr(st.session_state, '_volume_delta_data', None)
    # Fallback: compute from _df_5m if session data not yet populated
    if vd is None or vd.get('df') is None:
        _fallback_df = getattr(st.session_state, '_df_5m', None)
        if _fallback_df is not None and not _fallback_df.empty:
            try:
                vd = calculate_volume_delta(_fallback_df)
                if vd:
                    st.session_state._volume_delta_data = vd
            except Exception:
                vd = None
    if vd is None or vd.get('df') is None:
        st.info("⚡ Volume Delta builds after first chart load — refresh once to populate.")
        return
    try:
        df = vd['df'].copy()
        if df.empty:
            st.info("No Volume Delta data yet.")
            return
        df['datetime'] = pd.to_datetime(df['datetime'])
        vds = vd.get('summary', {})
        bias = vds.get('bias', 'N/A')
        bias_e = '🟢' if bias == 'Bullish' else '🔴' if bias == 'Bearish' else '⚪'
        tot_d = int(vds.get('total_delta', 0))
        d_rat = float(vds.get('delta_ratio', 1))

        fig = go.Figure()
        # Buy Volume bars (green)
        fig.add_trace(go.Bar(
            x=df['datetime'], y=df['buy_volume'],
            name='Buy Volume', marker_color='#089981',
            opacity=0.85
        ))
        # Sell Volume bars (red, negative direction for mirror effect)
        fig.add_trace(go.Bar(
            x=df['datetime'], y=-df['sell_volume'],
            name='Sell Volume', marker_color='#f23645',
            opacity=0.85
        ))
        # Cumulative delta line on secondary y
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=df['cum_delta'],
            name='Cum Delta', mode='lines',
            line=dict(color='#FFD700', width=2, dash='dot'),
            yaxis='y2'
        ))
        # Divergence markers
        div_df = df[df['divergence'] == True] if 'divergence' in df.columns else pd.DataFrame()
        if not div_df.empty:
            fig.add_trace(go.Scatter(
                x=div_df['datetime'], y=div_df['delta'],
                mode='markers', name='Divergence',
                marker=dict(symbol='diamond', size=10, color='#FF6B35',
                            line=dict(color='white', width=1)),
                yaxis='y'
            ))
        fig.update_layout(
            title=f'⚡ Buy vs Sell Volume | {bias_e} {bias} | Delta: {tot_d:+,} | Ratio: {d_rat:.2f}x',
            template='plotly_dark',
            height=320,
            barmode='overlay',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=9)),
            margin=dict(l=10, r=60, t=80, b=20),
            xaxis=dict(tickformat='%H:%M', title='Time'),
            yaxis=dict(title='Volume', zeroline=True, zerolinecolor='#555'),
            yaxis2=dict(title='Cum Delta', overlaying='y', side='right',
                        showgrid=False, zeroline=True, zerolinecolor='#FFD70060'),
            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Delta", f"{tot_d:+,}", delta=bias,
                  delta_color="normal" if bias == 'Bullish' else "inverse")
        m2.metric("Buy Volume", f"{int(vds.get('total_buy_volume', 0)):,}")
        m3.metric("Sell Volume", f"{int(vds.get('total_sell_volume', 0)):,}")
        m4.metric("Delta Ratio", f"{d_rat:.2f}x")
        m5.metric("Divergences", f"{int(vds.get('divergence_bars', 0))}")
    except Exception as e:
        st.caption(f"Volume Delta chart error: {e}")


def _render_per_strike_oi_top():
    """Render Per-Strike CE vs PE OI charts + signals from session state."""
    import plotly.graph_objects as go
    oi_history = st.session_state.get('oi_history', [])
    if not oi_history:
        st.info("📊 Per-Strike OI data builds up as the app refreshes. Please wait...")
        return
    try:
        oi_history_df = pd.DataFrame(oi_history)
        oi_strikes = st.session_state.get('oi_current_strikes') or []
        if not oi_strikes:
            _lv = st.session_state.get('oi_last_valid_data')
            if _lv is not None:
                oi_strikes = [int(s) for s in _lv['Strike'].tolist()]
        oi_strikes = sorted(oi_strikes)
        if not oi_strikes:
            st.info("Waiting for strike data...")
            return
        oi_position_labels = ['ITM-2', 'ITM-1', 'ATM', 'OTM+1', 'OTM+2']
        st.markdown(f"**📈 {len(oi_history)} data points | OI values in Lakhs**")
        oi_strike_cols = st.columns(min(len(oi_strikes), 5))
        for i, strike in enumerate(oi_strikes):
            if i >= len(oi_strike_cols):
                break
            ce_col = f'{strike}_CE'
            pe_col = f'{strike}_PE'
            with oi_strike_cols[i]:
                label = oi_position_labels[i] if i < len(oi_position_labels) else f'Strike {i}'
                fig_strike = go.Figure()
                if ce_col in oi_history_df.columns:
                    fig_strike.add_trace(go.Scatter(
                        x=oi_history_df['time'], y=oi_history_df[ce_col] / 100000,
                        mode='lines+markers', name='Call OI',
                        line=dict(color='#ff4444', width=2), marker=dict(size=3),
                    ))
                if pe_col in oi_history_df.columns:
                    fig_strike.add_trace(go.Scatter(
                        x=oi_history_df['time'], y=oi_history_df[pe_col] / 100000,
                        mode='lines+markers', name='Put OI',
                        line=dict(color='#00cc66', width=2), marker=dict(size=3),
                    ))
                cur_ce = oi_history_df[ce_col].iloc[-1] / 100000 if ce_col in oi_history_df.columns and len(oi_history_df) > 0 else 0
                cur_pe = oi_history_df[pe_col].iloc[-1] / 100000 if pe_col in oi_history_df.columns and len(oi_history_df) > 0 else 0
                fig_strike.update_layout(
                    title=f'{label}<br>₹{strike}<br>CE:{cur_ce:.1f}L PE:{cur_pe:.1f}L',
                    template='plotly_dark', height=260, showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=8)),
                    margin=dict(l=5, r=5, t=80, b=20),
                    xaxis=dict(tickformat='%H:%M', title=''),
                    yaxis=dict(title='OI (L)'),
                    plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e'
                )
                st.plotly_chart(fig_strike, use_container_width=True)
                # Signal badges
                if ce_col in oi_history_df.columns and pe_col in oi_history_df.columns and len(oi_history_df) >= 3:
                    ce_s = oi_history_df[ce_col]; pe_s = oi_history_df[pe_col]
                    ce_chg = ce_s.iloc[-1] - ce_s.iloc[0]; pe_chg = pe_s.iloc[-1] - pe_s.iloc[0]
                    ce_pct = (ce_chg / ce_s.iloc[0] * 100) if ce_s.iloc[0] > 0 else 0
                    pe_pct = (pe_chg / pe_s.iloc[0] * 100) if pe_s.iloc[0] > 0 else 0
                    if pe_chg > 0: st.success(f"Support Building (PE +{pe_pct:.1f}%)")
                    elif pe_chg < 0: st.error(f"Support Weakening (PE {pe_pct:.1f}%)")
                    if ce_chg > 0: st.error(f"Resistance Building (CE +{ce_pct:.1f}%)")
                    elif ce_chg < 0: st.success(f"Resistance Weakening (CE {ce_pct:.1f}%)")
                    oi_diff_l = abs(cur_pe - cur_ce)
                    if cur_pe > cur_ce:
                        r = cur_pe / cur_ce if cur_ce > 0 else 0
                        c = '#00ff88' if r >= 2.0 else '#00cc66'; lbl = 'STRONG SUPPORT' if r >= 2.0 else 'MODERATE SUPPORT'
                        st.markdown(f'<div style="background:{c}30;padding:6px;border-radius:6px;border-left:3px solid {c};font-size:12px;"><b style="color:{c};">{lbl}</b> | PE {cur_pe:.1f}L vs CE {cur_ce:.1f}L | Diff:{oi_diff_l:.1f}L | Ratio:{r:.1f}x</div>', unsafe_allow_html=True)
                    elif cur_ce > cur_pe:
                        r = cur_ce / cur_pe if cur_pe > 0 else 0
                        c = '#ff4444' if r >= 2.0 else '#cc4444'; lbl = 'STRONG RESISTANCE' if r >= 2.0 else 'MODERATE RESISTANCE'
                        st.markdown(f'<div style="background:{c}30;padding:6px;border-radius:6px;border-left:3px solid {c};font-size:12px;"><b style="color:{c};">{lbl}</b> | CE {cur_ce:.1f}L vs PE {cur_pe:.1f}L | Diff:{oi_diff_l:.1f}L | Ratio:{r:.1f}x</div>', unsafe_allow_html=True)
                    # LTP-based long/short signals
                    ce_ltp_col = f'{strike}_CE_LTP'; pe_ltp_col = f'{strike}_PE_LTP'
                    if ce_ltp_col in oi_history_df.columns and len(oi_history_df) >= 3:
                        ce_ltp_chg = oi_history_df[ce_ltp_col].iloc[-1] - oi_history_df[ce_ltp_col].iloc[0]
                        ce_ltp_pct = (ce_ltp_chg / oi_history_df[ce_ltp_col].iloc[0] * 100) if oi_history_df[ce_ltp_col].iloc[0] > 0 else 0
                        if ce_chg < 0 and ce_ltp_chg > 0:
                            st.markdown(f'<div style="background:#00ff8830;padding:5px;border-radius:5px;border-left:3px solid #00ff88;font-size:12px;"><b style="color:#00ff88;">CE: SHORT COVERING</b> | OI {ce_pct:.1f}% | LTP +{ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                        elif ce_chg > 0 and ce_ltp_chg > 0:
                            st.markdown(f'<div style="background:#ff880030;padding:5px;border-radius:5px;border-left:3px solid #ff8800;font-size:12px;"><b style="color:#ff8800;">CE: LONG BUILDING</b> | OI +{ce_pct:.1f}% | LTP +{ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                        elif ce_chg > 0 and ce_ltp_chg <= 0:
                            st.markdown(f'<div style="background:#ff444430;padding:5px;border-radius:5px;border-left:3px solid #ff4444;font-size:12px;"><b style="color:#ff4444;">CE: SHORT BUILDING</b> | OI +{ce_pct:.1f}% | LTP {ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                        elif ce_chg < 0 and ce_ltp_chg <= 0:
                            st.markdown(f'<div style="background:#88888830;padding:5px;border-radius:5px;border-left:3px solid #888;font-size:12px;"><b style="color:#888;">CE: LONG UNWINDING</b> | OI {ce_pct:.1f}% | LTP {ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                    if pe_ltp_col in oi_history_df.columns and len(oi_history_df) >= 3:
                        pe_ltp_chg = oi_history_df[pe_ltp_col].iloc[-1] - oi_history_df[pe_ltp_col].iloc[0]
                        pe_ltp_pct = (pe_ltp_chg / oi_history_df[pe_ltp_col].iloc[0] * 100) if oi_history_df[pe_ltp_col].iloc[0] > 0 else 0
                        if pe_chg > 0 and pe_ltp_chg <= 0:
                            st.markdown(f'<div style="background:#00ff8830;padding:5px;border-radius:5px;border-left:3px solid #00ff88;font-size:12px;"><b style="color:#00ff88;">PE: SHORT BUILDING</b> | OI +{pe_pct:.1f}% | LTP {pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                        elif pe_chg < 0 and pe_ltp_chg > 0:
                            st.markdown(f'<div style="background:#00cc6630;padding:5px;border-radius:5px;border-left:3px solid #00cc66;font-size:12px;"><b style="color:#00cc66;">PE: SHORT COVERING</b> | OI {pe_pct:.1f}% | LTP +{pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                        elif pe_chg > 0 and pe_ltp_chg > 0:
                            st.markdown(f'<div style="background:#ff444430;padding:5px;border-radius:5px;border-left:3px solid #ff4444;font-size:12px;"><b style="color:#ff4444;">PE: LONG BUILDING</b> | OI +{pe_pct:.1f}% | LTP +{pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                        elif pe_chg < 0 and pe_ltp_chg <= 0:
                            st.markdown(f'<div style="background:#88888830;padding:5px;border-radius:5px;border-left:3px solid #888;font-size:12px;"><b style="color:#888;">PE: LONG UNWINDING</b> | OI {pe_pct:.1f}% | LTP {pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
    except Exception as _e:
        st.caption(f"OI chart error: {_e}")


def show_auto_trade_section(option_data, df_5m, api, db):
    """Render the full auto-trade UI section."""
    st.markdown("---")
    st.markdown("## 🎯 Zone-Based Auto Trade")

    # ── Load persisted config from Supabase ──
    if 'trade_cfg' not in st.session_state:
        cfg = db.get_trade_config() or {}
        st.session_state.trade_cfg = cfg

    cfg = st.session_state.trade_cfg

    # ── Active trade from Supabase ──
    if 'active_trade' not in st.session_state:
        st.session_state.active_trade = db.get_active_trade()

    active_trade = st.session_state.active_trade

    # ── ACTIVE TRADE STATUS ──
    if active_trade and active_trade.get('status') == 'OPEN':
        tt = active_trade.get('trade_type', '')
        color = '#00ff88' if tt == 'CALL' else '#ff4444'
        entry_p = active_trade.get('entry_price', 0)
        target_p = active_trade.get('target', 0)
        sl_p = active_trade.get('sl', 0)
        strike_s = active_trade.get('strike', '')
        etime = active_trade.get('entry_time', '')[:16]
        st.markdown(f"""
        <div style="background:{color}20;padding:15px;border-radius:10px;border:2px solid {color};margin-bottom:10px;">
        <b style="color:{color};">🟢 ACTIVE TRADE — BUY {tt}</b><br>
        Strike: {strike_s} | Entry Spot: ₹{entry_p} | Target Spot: ₹{target_p} | SL Spot: ₹{sl_p}<br>
        <span style="color:#aaa;font-size:12px;">Entered: {etime}</span>
        </div>""", unsafe_allow_html=True)

        if st.button("🔴 Manual Exit", key="manual_exit_btn", type="primary"):
            with st.spinner("Exiting trade..."):
                updated, exit_p = _exit_trade(api, active_trade, "MANUAL", db)
                st.session_state.active_trade = None
                pnl = (float(exit_p or entry_p) - float(entry_p)) * 25 * int(active_trade.get('lot_size', 1))
                tg_msg = f"🔴 <b>MANUAL EXIT</b>\n{strike_s} {tt} | Exit: ₹{exit_p or '—'} | P&L: ₹{pnl:+.0f}"
                send_telegram_message_sync(tg_msg, force=True)
                st.success(f"✅ Exited at ₹{exit_p or '—'}")
                st.rerun()
        st.markdown("---")

    # ── MANUAL INPUTS ──
    st.markdown("### ⚙️ Manual Zone & Trade Setup")
    st.caption("All values saved to Supabase — safe across refresh")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🟢 Support Zone**")
        sup_bot = st.number_input("Bottom", value=float(cfg.get('support_zone_bottom') or 0), step=25.0, key="sup_bot", format="%.0f")
        sup_top = st.number_input("Top", value=float(cfg.get('support_zone_top') or 0), step=25.0, key="sup_top", format="%.0f")
    with c2:
        st.markdown("**🔴 Resistance Zone**")
        res_bot = st.number_input("Bottom", value=float(cfg.get('resistance_zone_bottom') or 0), step=25.0, key="res_bot", format="%.0f")
        res_top = st.number_input("Top", value=float(cfg.get('resistance_zone_top') or 0), step=25.0, key="res_top", format="%.0f")

    # Strike selector ATM±5
    strike_options = []
    atm_strike = None
    try:
        if option_data:
            und = option_data.get('underlying', 0)
            sg = option_data.get('strike_gap', 50)
            atm_strike = round(und / sg) * sg
            # Prefer df_atm8 (ATM±5 strikes, 11 rows) over df_summary (ATM±2)
            df_atm_src = option_data.get('df_atm8') or option_data.get('df_summary')
            if df_atm_src is not None:
                strike_col = 'strikePrice' if 'strikePrice' in df_atm_src.columns else 'Strike'
                all_strikes = sorted(df_atm_src[strike_col].unique().tolist())
                atm_pos = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - atm_strike))
                strike_options = [int(all_strikes[i]) for i in range(max(0, atm_pos - 5), min(len(all_strikes), atm_pos + 6))]
    except Exception:
        pass

    saved_strike = int(cfg.get('selected_strike') or (strike_options[len(strike_options)//2] if strike_options else 0))
    strike_idx = strike_options.index(saved_strike) if saved_strike in strike_options else 0
    selected_strike = st.selectbox("Strike Price (ATM±5)", options=strike_options, index=strike_idx, key="strike_sel") if strike_options else None

    st.markdown("**📞 CALL Setup** (BUY at Support Bottom) — enter **spot price** levels")
    ca1, ca2, ca3 = st.columns(3)
    with ca1:
        call_entry = st.number_input("Entry Spot ₹", value=float(cfg.get('call_entry') or 0), step=25.0, key="call_entry", format="%.0f")
    with ca2:
        call_target = st.number_input("Target Spot ₹", value=float(cfg.get('call_target') or 0), step=25.0, key="call_target", format="%.0f")
    with ca3:
        call_sl = st.number_input("SL Spot ₹", value=float(cfg.get('call_sl') or 0), step=25.0, key="call_sl_inp", format="%.0f")

    st.markdown("**📉 PUT Setup** (BUY at Resistance Top) — enter **spot price** levels")
    pa1, pa2, pa3 = st.columns(3)
    with pa1:
        put_entry = st.number_input("Entry Spot ₹", value=float(cfg.get('put_entry') or 0), step=25.0, key="put_entry", format="%.0f")
    with pa2:
        put_target = st.number_input("Target Spot ₹", value=float(cfg.get('put_target') or 0), step=25.0, key="put_target", format="%.0f")
    with pa3:
        put_sl = st.number_input("SL Spot ₹", value=float(cfg.get('put_sl') or 0), step=25.0, key="put_sl_inp", format="%.0f")

    auto_enabled = st.toggle("🤖 Enable Auto Zone Watch", value=bool(cfg.get('auto_trade_enabled', False)), key="auto_trade_toggle")

    # Save config button
    if st.button("💾 Save Setup", key="save_trade_cfg"):
        new_cfg = {
            'support_zone_bottom': sup_bot,
            'support_zone_top': sup_top,
            'resistance_zone_bottom': res_bot,
            'resistance_zone_top': res_top,
            'selected_strike': selected_strike,
            'call_entry': call_entry,
            'call_target': call_target,
            'call_sl': call_sl,
            'put_entry': put_entry,
            'put_target': put_target,
            'put_sl': put_sl,
            'auto_trade_enabled': auto_enabled,
            'lot_size': 1,
        }
        db.save_trade_config(new_cfg)
        st.session_state.trade_cfg = new_cfg
        st.success("✅ Setup saved to Supabase")

    # ── ZONE WATCH (only when auto enabled and no active trade) ──
    if auto_enabled and not (active_trade and active_trade.get('status') == 'OPEN'):
        spot = option_data.get('underlying', 0) if option_data else 0
        if spot and selected_strike:
            in_support = sup_bot > 0 and sup_top > 0 and sup_bot <= spot <= sup_top
            in_resistance = res_bot > 0 and res_top > 0 and res_bot <= spot <= res_top

            if in_support or in_resistance:
                zone_type = "SUPPORT" if in_support else "RESISTANCE"
                zb = sup_bot if in_support else res_bot
                zt = sup_top if in_support else res_top
                confs = _analyze_zone(spot, zb, zt, option_data, df_1m, getattr(st.session_state, '_df_1m_trade', None))
                score = sum(1 for c in confs if '✅' in c)

                color = '#00ff88' if in_support else '#ff4444'
                trade_type = "CALL" if in_support else "PUT"
                entry_v = call_entry if in_support else put_entry
                target_v = call_target if in_support else put_target
                sl_v = call_sl if in_support else put_sl

                st.markdown(f"""
                <div style="background:{color}20;padding:15px;border-radius:10px;border:2px solid {color};">
                <b style="color:{color};">⚠️ SPOT IN {zone_type} ZONE — ₹{spot:.0f}</b><br>
                Zone: ₹{zb:.0f} – ₹{zt:.0f} | Confirmations: {score}/{len(confs)}
                </div>""", unsafe_allow_html=True)

                for c in confs:
                    st.markdown(f"  {c}")

                # Send Telegram alert once per zone entry (cooldown 5 min)
                _last_zone_alert = st.session_state.get('_last_zone_alert_time')
                _last_zone_type = st.session_state.get('_last_zone_alert_type')
                _now_z = datetime.now(pytz.timezone('Asia/Kolkata'))
                _zone_cooldown_ok = (
                    _last_zone_alert is None or
                    (_now_z - _last_zone_alert).total_seconds() > 1800 or
                    _last_zone_type != zone_type
                )
                if _zone_cooldown_ok:
                    try:
                        tg_msg = _get_zone_telegram_msg(
                            spot, zone_type, zb, zt, confs,
                            selected_strike, call_entry, call_target, call_sl,
                            put_entry, put_target, put_sl
                        )
                        send_telegram_message_sync(tg_msg, force=True)
                        st.session_state._last_zone_alert_time = _now_z
                        st.session_state._last_zone_alert_type = zone_type
                    except Exception:
                        pass

                # Enter button
                btn_col, _ = st.columns([1, 2])
                with btn_col:
                    if st.button(f"✅ Enter {trade_type}", key=f"enter_{trade_type.lower()}_btn", type="primary"):
                        if entry_v <= 0:
                            st.error(f"Set {trade_type} entry price first and save setup.")
                        else:
                            with st.spinner(f"Placing {trade_type} order..."):
                                # Resolve security_id from option chain — required for Dhan order
                                sec_id = None
                                try:
                                    if option_data and option_data.get('df_summary') is not None:
                                        df_s2 = option_data['df_summary']
                                        row = df_s2[df_s2['Strike'] == selected_strike]
                                        if not row.empty:
                                            col_name = 'CE_SecurityId' if trade_type == 'CALL' else 'PE_SecurityId'
                                            # Also try suffixed form from merge
                                            alt_col = f"securityId_CE" if trade_type == 'CALL' else f"securityId_PE"
                                            for cn in [col_name, alt_col, 'securityId']:
                                                if cn in row.columns and row.iloc[0][cn]:
                                                    sec_id = int(row.iloc[0][cn])
                                                    break
                                except Exception:
                                    pass
                                if not sec_id:
                                    st.error(f"Could not find Dhan security ID for {selected_strike} {trade_type}. Load option chain first.")
                                    st.stop()

                                saved_trade, order_id, err = _place_trade(
                                    api, trade_type, selected_strike, sec_id,
                                    entry_v, target_v, sl_v, 1, spot, confs, db
                                )
                                st.session_state.active_trade = db.get_active_trade()
                                tg_entry = (
                                    f"{'🟢' if trade_type == 'CALL' else '🔴'} <b>AUTO ENTRY — BUY {trade_type}</b>\n"
                                    f"Strike: {selected_strike}{trade_type[0]}E | Spot: ₹{spot:.0f}\n"
                                    f"Entry Spot: ₹{entry_v:.0f} | Target Spot: ₹{target_v:.0f} | SL Spot: ₹{sl_v:.0f}\n"
                                    f"Zone: {zone_type} ₹{zb:.0f}–₹{zt:.0f} | Confirmations: {score}/{len(confs)}\n"
                                    f"Order ID: {order_id or '—'}"
                                )
                                if err:
                                    tg_entry += f"\n⚠️ Order error: {err}"
                                send_telegram_message_sync(tg_entry, force=True)
                                if err:
                                    st.warning(f"Order placed with warning: {err}")
                                else:
                                    st.success(f"✅ {trade_type} order placed! Order ID: {order_id}")
                                st.rerun()

    # ── TRADE MONITOR (auto exit at target/SL using spot price) ──
    if active_trade and active_trade.get('status') == 'OPEN':
        sec_id = active_trade.get('security_id')
        target_p = float(active_trade.get('target') or 0)
        sl_p = float(active_trade.get('sl') or 0)
        entry_p = float(active_trade.get('entry_price') or 0)
        tt = active_trade.get('trade_type', '')
        current_spot = float(option_data.get('underlying', 0)) if option_data else 0
        current_ltp = None
        dhan_pnl = None

        # Get option LTP + P&L from Dhan positions (for display only)
        try:
            positions = api.get_positions() if api else None
            if positions:
                pos_list = positions if isinstance(positions, list) else positions.get('data', [])
                for pos in pos_list:
                    pos_sec = str(pos.get('securityId', '') or pos.get('security_id', ''))
                    if pos_sec == str(sec_id):
                        current_ltp = float(pos.get('lastTradedPrice', 0) or pos.get('ltp', 0) or 0)
                        dhan_pnl = float(pos.get('unrealizedProfit', 0) or pos.get('pnl', 0) or 0)
                        break
        except Exception:
            pass
        if not current_ltp:
            try:
                if api and sec_id:
                    current_ltp = api.get_option_ltp(sec_id)
            except Exception:
                pass

        # Display using spot as primary reference
        spot_disp = f"Spot ₹{current_spot:.0f}" if current_spot else "Spot —"
        ltp_disp = f" | Option LTP ₹{current_ltp:.1f}" if current_ltp else ""
        pnl_disp = f"₹{dhan_pnl:+.0f}" if dhan_pnl is not None else "—"
        st.markdown(
            f"📡 **Live:** {spot_disp}{ltp_disp} | "
            f"Entry ₹{entry_p:.0f} | Target ₹{target_p:.0f} | SL ₹{sl_p:.0f} | P&L {pnl_disp}"
        )

        # Auto-exit: compare SPOT against target/SL (all values are spot prices)
        exit_reason = None
        if current_spot > 0:
            if target_p > 0 and tt == 'CALL' and current_spot >= target_p:
                exit_reason = "TARGET"
            elif target_p > 0 and tt == 'PUT' and current_spot <= target_p:
                exit_reason = "TARGET"
            elif sl_p > 0 and tt == 'CALL' and current_spot <= sl_p:
                exit_reason = "SL"
            elif sl_p > 0 and tt == 'PUT' and current_spot >= sl_p:
                exit_reason = "SL"

        if exit_reason:
            updated, exit_p = _exit_trade(api, active_trade, exit_reason, db)
            st.session_state.active_trade = None
            pnl_str = f"₹{dhan_pnl:+.0f}" if dhan_pnl is not None else f"Spot exit ₹{current_spot:.0f}"
            emoji = "✅" if exit_reason == "TARGET" else "🛑"
            tg_exit = (
                f"{emoji} <b>AUTO EXIT — {exit_reason} HIT</b>\n"
                f"{active_trade.get('strike')} {tt} | Spot: ₹{current_spot:.0f} | P&L: {pnl_str}"
            )
            send_telegram_message_sync(tg_exit, force=True)
            st.success(f"{emoji} {exit_reason} hit! Spot ₹{current_spot:.0f} | P&L {pnl_str}")
            st.rerun()

        # Reverse signal check
        try:
            master = getattr(st.session_state, '_master_signal_latest', None)
            if master:
                trade_signal = master.get('trade_type', '')
                if tt == 'CALL' and ('SELL' in trade_signal or 'BREAKDOWN' in trade_signal.upper()):
                    st.warning("⚠️ REVERSE SIGNAL detected — consider manual exit")
                    tg_rev = f"⚠️ <b>REVERSE SIGNAL</b> — Active {tt} trade but SELL signal detected. Review manually."
                    _last_rev = st.session_state.get('_last_reverse_alert')
                    if not _last_rev or (datetime.now(pytz.timezone('Asia/Kolkata')) - _last_rev).total_seconds() > 1800:
                        send_telegram_message_sync(tg_rev, force=True)
                        st.session_state._last_reverse_alert = datetime.now(pytz.timezone('Asia/Kolkata'))
                elif tt == 'PUT' and ('BUY' in trade_signal or 'BREAKOUT' in trade_signal.upper()):
                    st.warning("⚠️ REVERSE SIGNAL detected — consider manual exit")
                    tg_rev = f"⚠️ <b>REVERSE SIGNAL</b> — Active {tt} trade but BUY signal detected. Review manually."
                    _last_rev = st.session_state.get('_last_reverse_alert')
                    if not _last_rev or (datetime.now(pytz.timezone('Asia/Kolkata')) - _last_rev).total_seconds() > 1800:
                        send_telegram_message_sync(tg_rev, force=True)
                        st.session_state._last_reverse_alert = datetime.now(pytz.timezone('Asia/Kolkata'))
        except Exception:
            pass

    # ── TRADE HISTORY ──
    with st.expander("📋 Today's Trade History", expanded=False):
        history = db.get_trade_history()
        if history:
            for t in history:
                status_color = '#00ff88' if t.get('status') == 'OPEN' else '#aaa'
                ep = t.get('exit_price', '—')
                pnl_str = ''
                if t.get('exit_price') and t.get('entry_price'):
                    pnl = (float(t['exit_price']) - float(t['entry_price'])) * 25 * int(t.get('lot_size', 1))
                    pnl_str = f" | P&L: ₹{pnl:+.0f}"
                st.markdown(
                    f"<span style='color:{status_color};'>●</span> "
                    f"**{t.get('trade_type')}** {t.get('strike')} | "
                    f"Entry:₹{t.get('entry_price')} Target:₹{t.get('target')} SL:₹{t.get('sl')} | "
                    f"Exit:₹{ep} ({t.get('exit_reason','—')}){pnl_str} | {t.get('entry_time','')[:16]}",
                    unsafe_allow_html=True
                )
        else:
            st.info("No trades today.")


def main():
    st.title("📈 Nifty Trading & Options Analyzer")

    # ── Top-of-page buttons ──
    _btn_col1, _btn_col2 = st.columns(2)
    with _btn_col1:
        _top_send_clicked = st.button(
            "📤 Send Signal to Telegram",
            key="top_send_telegram",
            help="Force-send Master Signal + Option Chain Deep Analysis to Telegram",
            use_container_width=True,
            type="primary",
        )
        if _top_send_clicked:
            st.session_state['_top_send_triggered'] = True
    with _btn_col2:
        _ctx_clicked = st.button(
            "📚 Send AI Context to Telegram",
            key="send_ai_context",
            help="Send AI glossary/guide to Telegram (once at start of day)",
            use_container_width=True,
            type="primary",
        )
        if _ctx_clicked:
            try:
                for _ctx_part in generate_ai_context_message():
                    send_telegram_message_sync(_ctx_part, force=True)
                st.success("✅ AI context guide sent to Telegram! (2 messages)")
            except Exception as _ctx_err:
                st.error(f"Failed: {_ctx_err}")

    # Placeholder — auto trade section renders here (filled after db/api init below)
    _auto_trade_container = st.container()
    # Placeholder — Per-Strike OI chart renders here (filled after auto trade)
    _per_strike_oi_container = st.container()
    # Placeholder — Alignment + Capping panel (filled after api init)
    _align_cap_container = st.container()

    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    market_open = current_time.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=15, minute=45, second=0, microsecond=0)

    is_market_hours = market_open <= current_time <= market_close
    is_weekday = current_time.weekday() < 5

    if not (is_market_hours and is_weekday):
        st.warning("⏰ **Market is Closed**")
        if not is_weekday:
            st.info(f"📅 Today is {current_time.strftime('%A')}. Markets are closed on weekends.")
        else:
            st.info(f"""
            🕐 **Current Time:** {current_time.strftime('%H:%M:%S IST')}
            📊 **Market Hours:** 8:30 AM - 3:45 PM IST (Monday to Friday)
            The app will be fully functional during market hours.
            """)
        st.caption("You can still view cached data if available.")

    for key, default in [('pcr_history', []), ('pcr_last_valid_data', None),
                          ('gex_history', []), ('gex_last_valid_data', None),
                          ('last_gex_alert', None), ('gex_current_strikes', []),
                          ('oi_history', []), ('oi_last_valid_data', None),
                          ('oi_current_strikes', []),
                          ('chgoi_history', []), ('chgoi_last_valid_data', None),
                          ('chgoi_current_strikes', []),
                          ('vol_history', []), ('vol_last_valid_data', None),
                          ('vol_current_strikes', [])]:
        if key not in st.session_state:
            st.session_state[key] = default

    try:
        if not supabase_url or not supabase_key:
            st.error("Please configure your Supabase credentials in Streamlit secrets")
            st.info("""
            Add the following to your Streamlit secrets:
            ```
            [supabase]
            url = "your_supabase_url"
            anon_key = "your_supabase_anon_key"
            ```
            """)
            return
        db = SupabaseDB(supabase_url, supabase_key)
        db.sync_pending()
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return

    # Load PCR/GEX history from Supabase if session_state is empty (e.g. after page refresh)
    if not st.session_state.pcr_history:
        try:
            pcr_db_df = db.get_pcr_history()
            if not pcr_db_df.empty and 'timestamp' in pcr_db_df.columns:
                pcr_db_df['timestamp'] = pd.to_datetime(pcr_db_df['timestamp'])
                grouped = pcr_db_df.groupby('timestamp')
                for ts, group in grouped:
                    entry = {'time': ts.to_pydatetime()}
                    for _, row in group.iterrows():
                        entry[str(int(row['strike_price']))] = float(row['pcr_value'])
                    st.session_state.pcr_history.append(entry)
                st.session_state.pcr_history = st.session_state.pcr_history[-200:]
        except Exception:
            pass

    if not st.session_state.gex_history:
        try:
            gex_db_df = db.get_gex_history()
            if not gex_db_df.empty and 'timestamp' in gex_db_df.columns:
                gex_db_df['timestamp'] = pd.to_datetime(gex_db_df['timestamp'])
                grouped = gex_db_df.groupby('timestamp')
                for ts, group in grouped:
                    entry = {'time': ts.to_pydatetime(), 'total_gex': float(group['total_gex'].iloc[0])}
                    for _, row in group.iterrows():
                        entry[str(int(row['strike_price']))] = float(row['net_gex'])
                    st.session_state.gex_history.append(entry)
                st.session_state.gex_history = st.session_state.gex_history[-200:]
        except Exception:
            pass

    try:
        if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
            st.error("Please configure your Dhan API credentials in Streamlit secrets")
            st.info("""
            Add the following to your Streamlit secrets:
            ```
            DHAN_CLIENT_ID = "your_client_id"
            DHAN_ACCESS_TOKEN = "your_access_token"
            TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
            TELEGRAM_CHAT_ID = "your_telegram_chat_id"
            ```
            """)
            return
        access_token, client_id, issues = validate_credentials(DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID)
        if issues:
            st.error("Issues found with API credentials:")
            for issue in issues:
                st.error(f"• {issue}")
            st.info("The app will try to use the cleaned values automatically.")
        st.sidebar.success("API credentials processed")
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            st.sidebar.success("Telegram notifications enabled")
        else:
            st.sidebar.warning("Telegram notifications disabled - configure bot token and chat ID")

    except Exception as e:
        st.error(f"Credential validation error: {str(e)}")
        return

    user_id = get_user_id()
    user_prefs = db.get_user_preferences(user_id)

    st.sidebar.header("Configuration")

    timeframes = {
        "1 min": "1",
        "3 min": "3",
        "5 min": "5",
        "10 min": "10",
        "15 min": "15"
    }

    default_timeframe = next((k for k, v in timeframes.items() if v == user_prefs['timeframe']), "5 min")
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        list(timeframes.keys()),
        index=list(timeframes.keys()).index(default_timeframe)
    )

    interval = timeframes[selected_timeframe]

    st.sidebar.header("📊 Pivot Indicator Settings")

    show_pivots = st.sidebar.checkbox("Show Pivot Levels", value=True, help="Display Higher Timeframe Support/Resistance levels")

    if show_pivots:
        st.sidebar.subheader("Toggle Individual Pivot Levels")
        if 'pivot_settings' not in user_prefs:
            user_prefs['pivot_settings'] = {
                'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True
            }
        show_3m = st.sidebar.checkbox("3 Minute Pivots", value=user_prefs['pivot_settings'].get('show_3m', True), help="🟢 Green lines")
        show_5m = st.sidebar.checkbox("5 Minute Pivots", value=user_prefs['pivot_settings'].get('show_5m', True), help="🟠 Orange lines")
        show_10m = st.sidebar.checkbox("10 Minute Pivots", value=user_prefs['pivot_settings'].get('show_10m', True), help="🟣 Pink lines")
        show_15m = st.sidebar.checkbox("15 Minute Pivots", value=user_prefs['pivot_settings'].get('show_15m', True), help="🔵 Blue lines")
        pivot_settings = {
            'show_3m': show_3m,
            'show_5m': show_5m,
            'show_10m': show_10m,
            'show_15m': show_15m
        }
        st.sidebar.info("""
        **Pivot Levels Legend:**
        🟢 3M (Green) - 3-minute timeframe
        🟠 5M (Orange) - 5-minute timeframe
        🟣 10M (Pink) - 10-minute timeframe
        🔵 15M (Blue) - 15-minute timeframe
        S = Support, R = Resistance
        """)
    else:
        pivot_settings = {
            'show_3m': False, 'show_5m': False, 'show_10m': False, 'show_15m': False
        }

    st.sidebar.header("🔔 Trading Signals")
    enable_signals = st.sidebar.checkbox("Enable Telegram Signals", value=True, help="Send notifications when conditions are met")

    pivot_proximity = st.sidebar.slider(
        "Pivot Proximity (± Points)",
        min_value=1,
        max_value=20,
        value=user_prefs.get('pivot_proximity', 5),
        help="Distance from pivot levels to trigger signals (both above and below)"
    )

    if enable_signals:
        st.sidebar.info(f"Signals sent when:\n• Price within ±{pivot_proximity}pts of pivot\n• All option bias aligned\n• ATM at support/resistance")

    st.sidebar.header("📅 Options Settings")

    expiry_data = get_dhan_expiry_list_cached(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
    expiry_dates = []
    if expiry_data and 'data' in expiry_data:
        expiry_dates = expiry_data['data']

    selected_expiry = None
    if expiry_dates:
        expiry_options = [f"{exp} ({'Weekly' if i < 4 else 'Monthly'})" for i, exp in enumerate(expiry_dates)]
        selected_expiry_idx = st.sidebar.selectbox(
            "Select Expiry",
            range(len(expiry_options)),
            format_func=lambda x: expiry_options[x]
        )
        selected_expiry = expiry_dates[selected_expiry_idx]

    auto_refresh = st.sidebar.checkbox("Auto Refresh (2 min)", value=user_prefs['auto_refresh'])

    days_back = st.sidebar.slider("Days of Historical Data", 1, 5, user_prefs['days_back'])

    use_cache = st.sidebar.checkbox("Use Cached Data", value=True, help="Use database cache for faster loading")

    st.sidebar.header("🗑️ Database Management")
    cleanup_days = st.sidebar.selectbox("Clear History Older Than", [7, 14, 30], index=0)

    if st.sidebar.button("🗑 Clear History"):
        deleted_count = db.clear_old_candles(cleanup_days)
        st.sidebar.success(f"Deleted {deleted_count} old records")

    st.sidebar.header("🔧 Connection Test")

    if st.sidebar.button("Test Telegram Connection"):
        success, message = test_telegram_connection()
        if success:
            st.sidebar.success(message)
            test_msg = "🔔 Nifty Analyzer - Test message successful! ✅"
            send_telegram_message_sync(test_msg)
            st.sidebar.success("Test message sent to Telegram!")
        else:
            st.sidebar.error(message)

    if st.sidebar.button("💾 Save Preferences"):
        db.save_user_preferences(user_id, interval, auto_refresh, days_back, pivot_settings, pivot_proximity)
        st.sidebar.success("Preferences saved!")

    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    show_analytics = st.sidebar.checkbox("Show Analytics Dashboard", value=False)

    st.sidebar.subheader("🔧 Debug Info")
    st.sidebar.write(f"Telegram Bot Token: {'✅ Set' if TELEGRAM_BOT_TOKEN else '❌ Missing'}")
    st.sidebar.write(f"Telegram Chat ID: {'✅ Set' if TELEGRAM_CHAT_ID else '❌ Missing'}")
    st.sidebar.write(f"Token length: {len(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else 0}")
    st.sidebar.write(f"Chat ID: {TELEGRAM_CHAT_ID}")

    # ── Token Refresh (for when Dhan token expires daily) ──
    with st.sidebar.expander("🔑 Refresh Dhan Token", expanded=bool(st.session_state.get('_dhan_token_expired'))):
        st.caption("Dhan access tokens expire daily. Paste your new token here to refresh without restarting.")
        _new_token = st.text_input(
            "New Access Token",
            type="password",
            placeholder="Paste new Dhan access token...",
            key="_new_dhan_token_input",
        )
        _col_a, _col_b = st.columns(2)
        with _col_a:
            if st.button("✅ Apply Token", key="_apply_dhan_token"):
                if _new_token and len(_new_token.strip()) > 10:
                    st.session_state['_dhan_token_override'] = _new_token.strip()
                    st.session_state['_dhan_token_expired'] = False
                    st.success("Token updated!")
                    st.rerun()
                else:
                    st.error("Token too short")
        with _col_b:
            if st.button("🗑️ Clear", key="_clear_dhan_token"):
                st.session_state.pop('_dhan_token_override', None)
                st.session_state['_dhan_token_expired'] = False
                st.rerun()
        if st.session_state.get('_dhan_token_override'):
            st.success("✅ Using overridden token")

    # Use overridden token if set
    if st.session_state.get('_dhan_token_override'):
        access_token = st.session_state['_dhan_token_override']

    api = DhanAPI(access_token, client_id)

    # Fetch 1m candle data for zone analysis (cached in session_state)
    try:
        _last_1m_fetch = st.session_state.get('_last_1m_candle_fetch')
        _now_1m = datetime.now(pytz.timezone('Asia/Kolkata'))
        if _last_1m_fetch is None or (_now_1m - _last_1m_fetch).total_seconds() > 60:
            _raw_1m = api.get_intraday_data(security_id="13", exchange_segment="IDX_I", instrument="INDEX", interval="1", days_back=1)
            if _raw_1m and 'open' in _raw_1m:
                _df_1m_new = pd.DataFrame({
                    'Open': _raw_1m['open'], 'High': _raw_1m['high'],
                    'Low': _raw_1m['low'], 'Close': _raw_1m['close'],
                    'Volume': _raw_1m.get('volume', [0]*len(_raw_1m['open'])),
                })
                st.session_state._df_1m_trade = _df_1m_new
                st.session_state._last_1m_candle_fetch = _now_1m
    except Exception:
        pass

    # Fill the auto trade container at the top of the page
    with _auto_trade_container:
        try:
            _opt_data_top = st.session_state.get('_cached_option_data')
            _df_5m_top = getattr(st.session_state, '_df_5m', None)
            show_auto_trade_section(_opt_data_top, _df_5m_top, api, db)
        except Exception as _ate_top:
            st.caption(f"Auto trade init: {_ate_top}")

    # Fill Per-Strike OI chart right below auto trade section
    with _per_strike_oi_container:
        with st.expander("⚡ Buy Volume vs Sell Volume (Delta Chart)", expanded=True):
            _render_vol_delta_chart()
        with st.expander("📊 Per-Strike Call vs Put OI", expanded=True):
            _render_per_strike_oi_top()

    # Fill Alignment + Index/Stock Capping below Per-Strike OI
    with _align_cap_container:
        _render_alignment_capping_top()

    col1, col2 = st.columns([2, 1])

    pivots = None
    vob_data = None

    with col1:
        st.header("📈 Trading Chart")
        df = pd.DataFrame()
        current_price = None
        if use_cache:
            df = db.get_candles("NIFTY50", "IDX_I", interval, hours_back=days_back*24)
        need_fetch = not use_cache or df.empty or (datetime.now(pytz.UTC) - df['datetime'].max().tz_convert(pytz.UTC)).total_seconds() > 300
        if need_fetch:
            with st.spinner("Fetching data from API..."):
                data = api.get_intraday_data(security_id="13", exchange_segment="IDX_I", instrument="INDEX", interval=interval, days_back=days_back)
                if data:
                    df = process_candle_data(data, interval)
                    db.upsert_candles("NIFTY50", "IDX_I", interval, df)
        ltp_data = api.get_ltp_data("13", "IDX_I")
        if ltp_data and 'data' in ltp_data:
            for exchange, data in ltp_data['data'].items():
                for security_id, price_data in data.items():
                    current_price = price_data.get('last_price', 0)
                    break
        if current_price is not None and current_price > 0:
            try:
                db.upsert_spot_data(current_price, security_id='13', exchange_segment='IDX_I')
            except Exception:
                pass
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        if not df.empty:
            display_metrics(ltp_data, df, db)
        vob_blocks_for_chart = None
        if not df.empty and len(df) > 30:
            try:
                vob_detector = VolumeOrderBlocks(sensitivity=5)
                _, vob_blocks_for_chart = vob_detector.get_sr_levels(df)
            except Exception:
                vob_blocks_for_chart = None
        if vob_blocks_for_chart:
            st.session_state._vob_blocks = vob_blocks_for_chart
        poc_data_for_chart = None
        if not df.empty and len(df) > 100:
            try:
                poc_calculator = TriplePOC(period1=10, period2=25, period3=70)
                poc_data_for_chart = poc_calculator.calculate_all_pocs(df)
            except Exception:
                poc_data_for_chart = None
        swing_data_for_chart = None
        if not df.empty and len(df) > 50:
            try:
                swing_calculator = FutureSwing(swing_length=30, history_samples=5, calc_type='Average')
                swing_data_for_chart = swing_calculator.analyze(df)
            except Exception:
                swing_data_for_chart = None
        # Store for Telegram message access
        st.session_state['_poc_data'] = poc_data_for_chart
        st.session_state['_swing_data'] = swing_data_for_chart
        # ATR(14) and session IV tracking for market context block
        try:
            if not df.empty and len(df) >= 14:
                _h = df['high']; _l = df['low']; _c = df['close']
                _tr = pd.concat([_h - _l, (_h - _c.shift(1)).abs(), (_l - _c.shift(1)).abs()], axis=1).max(axis=1)
                st.session_state['_atr14'] = round(_tr.rolling(14).mean().iloc[-1], 1)
        except Exception:
            pass
        # Persist detected patterns (VOB, POC, Swing) to Supabase
        try:
            patterns_to_store = []
            if vob_blocks_for_chart:
                for block_type in ['bullish', 'bearish']:
                    for block in vob_blocks_for_chart.get(block_type, []):
                        patterns_to_store.append({
                            'pattern_type': f'VOB_{block_type.upper()}',
                            'timeframe': interval,
                            'direction': block_type,
                            'price_level': block.get('mid'),
                            'upper_bound': block.get('upper'),
                            'lower_bound': block.get('lower'),
                            'score': block.get('volume_pct'),
                            'metadata': {'volume': block.get('volume', 0), 'datetime': str(block.get('datetime', ''))}
                        })
            if poc_data_for_chart:
                for poc_key in ['poc1', 'poc2', 'poc3']:
                    poc = poc_data_for_chart.get(poc_key)
                    if poc:
                        period = poc_data_for_chart.get('periods', {}).get(poc_key, '')
                        patterns_to_store.append({
                            'pattern_type': f'POC_{poc_key.upper()}',
                            'timeframe': str(period),
                            'direction': 'neutral',
                            'price_level': poc.get('poc'),
                            'upper_bound': poc.get('upper_poc'),
                            'lower_bound': poc.get('lower_poc'),
                            'score': None,
                            'metadata': {'high': poc.get('high', 0), 'low': poc.get('low', 0), 'volume': poc.get('volume', 0)}
                        })
            if swing_data_for_chart:
                swings = swing_data_for_chart.get('swings', {})
                projection = swing_data_for_chart.get('projection')
                if projection:
                    patterns_to_store.append({
                        'pattern_type': 'SWING_PROJECTION',
                        'timeframe': interval,
                        'direction': projection.get('direction', 'unknown'),
                        'price_level': projection.get('target'),
                        'upper_bound': projection.get('from_value'),
                        'lower_bound': None,
                        'score': projection.get('swing_pct'),
                        'metadata': {'volume_delta': swing_data_for_chart.get('volume', {}).get('delta', 0)}
                    })
                last_high = swings.get('last_swing_high')
                last_low = swings.get('last_swing_low')
                if last_high:
                    patterns_to_store.append({
                        'pattern_type': 'SWING_HIGH',
                        'timeframe': interval,
                        'direction': 'bearish',
                        'price_level': last_high.get('value'),
                        'upper_bound': None, 'lower_bound': None, 'score': None,
                        'metadata': {'index': last_high.get('index', 0)}
                    })
                if last_low:
                    patterns_to_store.append({
                        'pattern_type': 'SWING_LOW',
                        'timeframe': interval,
                        'direction': 'bullish',
                        'price_level': last_low.get('value'),
                        'upper_bound': None, 'lower_bound': None, 'score': None,
                        'metadata': {'index': last_low.get('index', 0)}
                    })
            if patterns_to_store:
                db.upsert_detected_patterns(patterns_to_store)
        except Exception:
            pass
        # ── Compute Money Flow Profile & Volume Delta ──
        money_flow_data = None
        volume_delta_data = None
        if not df.empty and len(df) > 5:
            try:
                money_flow_data = calculate_money_flow_profile(df, num_rows=25, source='Volume')
            except Exception as e:
                st.caption(f"⚠️ MF Profile error: {str(e)[:80]}")
                money_flow_data = None
            st.session_state._money_flow_data = money_flow_data
            try:
                volume_delta_data = calculate_volume_delta(df)
                st.session_state._volume_delta_data = volume_delta_data
            except Exception as e:
                st.caption(f"⚠️ Volume Delta error: {str(e)[:80]}")
                volume_delta_data = None
            # Store Money Flow Profile POC + Volume Delta summary as patterns
            try:
                mf_vd_patterns = []
                if money_flow_data:
                    mf_vd_patterns.append({
                        'pattern_type': 'MONEY_FLOW_POC',
                        'timeframe': interval,
                        'direction': money_flow_data.get('highest_sentiment_direction', 'neutral').lower(),
                        'price_level': money_flow_data.get('poc_price'),
                        'upper_bound': money_flow_data.get('value_area_high'),
                        'lower_bound': money_flow_data.get('value_area_low'),
                        'score': None,
                        'metadata': {
                            'total_volume': money_flow_data.get('total_volume', 0),
                            'num_rows': money_flow_data.get('num_rows', 25),
                            'sentiment_price': money_flow_data.get('highest_sentiment_price', 0),
                            'source': money_flow_data.get('source', 'Volume'),
                        }
                    })
                if volume_delta_data and volume_delta_data.get('summary'):
                    vds = volume_delta_data['summary']
                    mf_vd_patterns.append({
                        'pattern_type': 'VOLUME_DELTA',
                        'timeframe': interval,
                        'direction': vds.get('bias', 'neutral').lower(),
                        'price_level': current_price,
                        'upper_bound': None,
                        'lower_bound': None,
                        'score': vds.get('delta_ratio'),
                        'metadata': {
                            'total_delta': vds.get('total_delta', 0),
                            'buy_volume': vds.get('total_buy_volume', 0),
                            'sell_volume': vds.get('total_sell_volume', 0),
                            'divergence_bars': vds.get('divergence_bars', 0),
                            'cum_delta': vds.get('cum_delta_last', 0),
                        }
                    })
                if mf_vd_patterns:
                    db.upsert_detected_patterns(mf_vd_patterns)
            except Exception:
                pass
        if not df.empty:
            fig = create_candlestick_chart(
                df,
                f"Nifty 50 - {selected_timeframe} Chart {'with Pivot Levels' if show_pivots else ''}",
                interval,
                show_pivots=show_pivots,
                pivot_settings=pivot_settings,
                vob_blocks=vob_blocks_for_chart,
                poc_data=poc_data_for_chart,
                swing_data=swing_data_for_chart,
                money_flow_data=money_flow_data,
                volume_delta_data=volume_delta_data
            )
            st.plotly_chart(fig, use_container_width=True)
            infos = [
                f"📊 Data Points: {len(df)}",
                f"🕐 Latest: {df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S IST')}",
                f"📡 Source: {'Database Cache' if use_cache else 'Live API'}",
                f"📈 Pivots: {'✅ Enabled' if show_pivots else '❌ Disabled'}"
            ]
            for c, txt in zip(st.columns(4), infos):
                c.info(txt)
            if show_pivots and len(df) > 50:
                st.markdown("""
                **Pivot Levels Legend:**
                - 🟢 **3M Levels**: 3-minute timeframe support/resistance
                - 🟠 **5M Levels**: 5-minute timeframe swing points
                - 🟣 **10M Levels**: 10-minute support/resistance zones
                - 🔵 **15M Levels**: 15-minute major support/resistance levels
                - 🟡 **VWAP**: Volume Weighted Average Price (dotted line)
                - 🟩 **VOB↑ (Teal)**: Bullish Volume Order Blocks (Support zones)
                - 🟪 **VOB↓ (Purple)**: Bearish Volume Order Blocks (Resistance zones)
                *R = Resistance (Price ceiling), S = Support (Price floor)*
                *VOB zones show volume-backed order flow areas with % distribution*
                """)
            # ── Money Flow Profile & Volume Delta Tables ──
            st.markdown("---")
            mf_tab, vd_tab = st.tabs(["💰 Money Flow Profile", "📊 Volume Delta Candles"])
            with mf_tab:
                if money_flow_data and money_flow_data.get('rows'):
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("MF-POC", f"₹{money_flow_data['poc_price']:.0f}")
                    mc2.metric("Value Area", f"₹{money_flow_data['value_area_low']:.0f} - ₹{money_flow_data['value_area_high']:.0f}")
                    mc3.metric("Top Sentiment", f"{money_flow_data['highest_sentiment_direction']} @ ₹{money_flow_data['highest_sentiment_price']:.0f}")
                    mc4.metric("Bars", f"{money_flow_data['num_bars']}")
                    mf_df = pd.DataFrame(money_flow_data['rows'])
                    mf_display = mf_df[['price_level', 'total_volume', 'bull_volume', 'bear_volume', 'delta', 'volume_pct', 'node_type', 'sentiment', 'sentiment_strength']].copy()
                    mf_display.columns = ['Price', 'Total Vol', 'Buy Vol', 'Sell Vol', 'Delta', 'Vol %', 'Node', 'Sentiment', 'Strength %']
                    mf_display['Price'] = mf_display['Price'].apply(lambda x: f"₹{x:.0f}")
                    mf_display['Total Vol'] = mf_display['Total Vol'].apply(lambda x: f"{x:,.0f}")
                    mf_display['Buy Vol'] = mf_display['Buy Vol'].apply(lambda x: f"{x:,.0f}")
                    mf_display['Sell Vol'] = mf_display['Sell Vol'].apply(lambda x: f"{x:,.0f}")
                    mf_display['Delta'] = mf_display['Delta'].apply(lambda x: f"{x:+,.0f}")
                    mf_display = mf_display.reset_index(drop=True)
                    def _mf_row_style(row):
                        styles = [''] * len(row)
                        try:
                            node = row.get('Node', '')
                            sent = row.get('Sentiment', '')
                            node_idx = list(row.index).index('Node')
                            sent_idx = list(row.index).index('Sentiment')
                            if node == 'High': styles[node_idx] = 'background-color: #ffeb3b40; color: white'
                            elif node == 'Low': styles[node_idx] = 'background-color: #f2364540; color: white'
                            else: styles[node_idx] = 'background-color: #2962ff30; color: white'
                            if sent == 'Bullish': styles[sent_idx] = 'background-color: #26a69a40; color: white'
                            elif sent == 'Bearish': styles[sent_idx] = 'background-color: #ef535040; color: white'
                        except: pass
                        return styles
                    styled_mf = mf_display.style.apply(_mf_row_style, axis=1)
                    st.dataframe(styled_mf, use_container_width=True, hide_index=True)
                    st.markdown("""
                    **Money Flow Profile:**
                    - 🟡 **High Traded Nodes**: Consolidation/value areas - price tends to spend time here
                    - 🔵 **Average Nodes**: Normal trading activity levels
                    - 🔴 **Low Traded Nodes**: Supply/demand zones, liquidity gaps - price moves fast through these
                    - **MF-POC**: Price level with highest traded volume (Point of Control)
                    - **VA-H/VA-L**: Value Area boundaries (consolidation zone)
                    """)
                else:
                    st.info(f"Money Flow Profile: No data yet. df rows={len(df) if not df.empty else 0}, money_flow_data={'computed' if money_flow_data else 'None'}")

            with vd_tab:
                if volume_delta_data and volume_delta_data.get('summary'):
                    vd_summary = volume_delta_data['summary']
                    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
                    bias_color = "normal" if vd_summary['bias'] == 'Bullish' else "inverse"
                    vc1.metric("Total Delta", f"{vd_summary['total_delta']:+,}", delta=vd_summary['bias'], delta_color=bias_color)
                    vc2.metric("Buy Volume", f"{vd_summary['total_buy_volume']:,}")
                    vc3.metric("Sell Volume", f"{vd_summary['total_sell_volume']:,}")
                    vc4.metric("Delta Ratio", f"{vd_summary['delta_ratio']:.2f}")
                    vc5.metric("Divergences", f"{vd_summary['divergence_bars']}")
                    vd_df = volume_delta_data['df'].copy()
                    vd_display = vd_df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 'delta', 'delta_pct', 'cum_delta', 'divergence']].tail(50).copy()
                    vd_display.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Buy Vol', 'Sell Vol', 'Delta', 'Delta %', 'Cum Delta', 'Divergence']
                    vd_display['Time'] = pd.to_datetime(vd_display['Time']).dt.strftime('%H:%M')
                    vd_display = vd_display.reset_index(drop=True)
                    def _vd_style(row):
                        styles = [''] * len(row)
                        try:
                            d = float(row['Delta'])
                            color = '#08998130' if d > 0 else '#f2364530' if d < 0 else ''
                            if color:
                                for i, col in enumerate(row.index):
                                    if col in ['Delta', 'Cum Delta']:
                                        styles[i] = f'background-color: {color}; color: white'
                            if row['Divergence']:
                                div_idx = list(row.index).index('Divergence')
                                styles[div_idx] = 'background-color: #FFD70040; color: white'
                        except: pass
                        return styles
                    styled_vd = vd_display.style.apply(_vd_style, axis=1)
                    st.dataframe(styled_vd, use_container_width=True, hide_index=True)
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown(f"""
                        **Volume Delta Summary:**
                        - +Delta Streak: **{vd_summary['max_positive_streak']}** bars
                        - -Delta Streak: **{vd_summary['max_negative_streak']}** bars
                        - +Delta Bars: **{vd_summary['positive_bars']}** | -Delta Bars: **{vd_summary['negative_bars']}**
                        """)
                    with dc2:
                        st.markdown("""
                        **Interpretation:**
                        - **+Delta**: Buyers dominating (price should rise)
                        - **-Delta**: Sellers dominating (price should fall)
                        - **Divergence**: Candle direction opposes delta (potential reversal)
                        - **Cum Delta**: Running total - trend of buying/selling pressure
                        """)
                else:
                    st.info(f"Volume Delta: No data yet. df rows={len(df) if not df.empty else 0}, volume_delta_data={'computed' if volume_delta_data else 'None'}")

            st.markdown("---")
            st.markdown("## 🔄 Intraday Reversal Detector")
            try:
                pivot_lows = []
                pivot_highs = []
                if show_pivots and len(df) > 50:
                    df_json = df.to_json()
                    pivots = cached_pivot_calculation(df_json, pivot_settings or {})
                    pivot_lows = [p['value'] for p in pivots if p['type'] == 'low']
                    pivot_highs = [p['value'] for p in pivots if p['type'] == 'high']
                if len(df) > 30:
                    try:
                        vob_detector = VolumeOrderBlocks(sensitivity=5)
                        vob_sr_levels, vob_blocks = vob_detector.get_sr_levels(df)
                        vob_data = {
                            'sr_levels': vob_sr_levels,
                            'blocks': vob_blocks
                        }
                    except Exception as e:
                        st.warning(f"VOB calculation error: {str(e)}")
                        vob_data = None
                bull_score, bull_signals, bull_verdict = ReversalDetector.calculate_reversal_score(df, pivot_lows)
                bear_score, bear_signals, bear_verdict = ReversalDetector.calculate_bearish_reversal_score(df, pivot_highs)
                # Persist reversal patterns to Supabase
                try:
                    reversal_patterns = []
                    if bull_score >= 3:
                        reversal_patterns.append({
                            'pattern_type': 'REVERSAL_BULLISH',
                            'timeframe': interval,
                            'direction': 'bullish',
                            'price_level': bull_signals.get('Current_Price'),
                            'upper_bound': bull_signals.get('Day_High'),
                            'lower_bound': bull_signals.get('Day_Low'),
                            'score': bull_score,
                            'metadata': {
                                'verdict': bull_verdict,
                                'vwap': bull_signals.get('VWAP'),
                                'selling_exhausted': bull_signals.get('Selling_Exhausted', ''),
                                'higher_low': bull_signals.get('Higher_Low', ''),
                                'volume_signal': bull_signals.get('Volume_Signal', ''),
                                'support_level': bull_signals.get('Support_Level'),
                            }
                        })
                    if bear_score <= -3:
                        reversal_patterns.append({
                            'pattern_type': 'REVERSAL_BEARISH',
                            'timeframe': interval,
                            'direction': 'bearish',
                            'price_level': bear_signals.get('Current_Price'),
                            'upper_bound': bear_signals.get('Day_High'),
                            'lower_bound': bear_signals.get('Day_Low'),
                            'score': abs(bear_score),
                            'metadata': {
                                'verdict': bear_verdict,
                                'vwap': bear_signals.get('VWAP'),
                                'buying_exhausted': bear_signals.get('Buying_Exhausted', ''),
                                'lower_high': bear_signals.get('Lower_High', ''),
                                'volume_signal': bear_signals.get('Volume_Signal', ''),
                            }
                        })
                    if reversal_patterns:
                        db.upsert_detected_patterns(reversal_patterns)
                except Exception:
                    pass
                col_bull, col_bear = st.columns(2)
                for col, title, verdict, sigs, score_key, items in [
                    (col_bull, "### 🟢 Bullish Reversal", bull_verdict, bull_signals, 'Reversal_Score',
                     [('Selling Exhausted', 'Selling_Exhausted'), ('Higher Low', 'Higher_Low'),
                      ('Strong Bullish Candle', 'Strong_Bullish_Candle'), ('Volume', 'Volume_Signal'), ('Above VWAP', 'Above_VWAP')]),
                    (col_bear, "### 🔴 Bearish Reversal", bear_verdict, bear_signals, 'Bearish_Score',
                     [('Buying Exhausted', 'Buying_Exhausted'), ('Lower High', 'Lower_High'),
                      ('Strong Bearish Candle', 'Strong_Bearish_Candle'), ('Volume', 'Volume_Signal'), ('Below VWAP', 'Below_VWAP')])
                ]:
                    with col:
                        st.markdown(title)
                        if "STRONG" in verdict:
                            (st.success if "BUY" in verdict else st.error)(f"**{verdict}**")
                        elif "MODERATE" in verdict:
                            st.warning(f"**{verdict}**")
                        else:
                            st.info(f"**{verdict}**")
                        st.markdown(f"**Score: {sigs.get(score_key, 0)}/6**")
                        for label, key in items:
                            st.markdown(f"- {label}: {sigs.get(key, 'N/A')}")
                if bull_signals.get('VWAP'):
                    st.info(f"📊 **VWAP:** ₹{bull_signals.get('VWAP')} | **Day High:** ₹{bull_signals.get('Day_High', 'N/A')} | **Day Low:** ₹{bull_signals.get('Day_Low', 'N/A')}")
                with st.expander("📋 Entry Rules & Recommendations"):
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown("**🟢 Bullish Entry Rules:**")
                        entry_rules = ReversalDetector.get_entry_rules(bull_signals, bull_score)
                        for rule in entry_rules:
                            st.markdown(f"- {rule}")
                    with col_r2:
                        st.markdown("**🔴 Bearish Entry Rules:**")
                        if bear_score <= -4:
                            st.markdown("- 🎯 ENTRY: Buy PE at current level")
                            st.markdown(f"- 🛑 SL: Above recent high ({bear_signals.get('Day_High', 'N/A')})")
                            st.markdown("- 🎯 Target: Previous low / Nearest support")
                        elif bear_score <= -2.5:
                            st.markdown("- ⏳ WAIT: Confirmation pending")
                            st.markdown("- 📋 Checklist: Lower High + Strong Bearish Candle + Volume")
                        else:
                            st.markdown("- ❌ NO ENTRY: Bearish conditions not met")
                    st.markdown("---")
                    st.markdown("**🧠 Trading Psychology:**")
                    st.markdown("> *Missing a trade is 100x better than entering a wrong trade.*")
                    st.markdown("- Trade only after structure forms")
                    st.markdown("- No emotional entries")
                    st.markdown("- Fixed SL, fixed target")
                    st.markdown("- If trade missed → day closed")
            except Exception as e:
                st.warning(f"Reversal analysis unavailable: {str(e)}")
            st.markdown("---")
            st.markdown("## 📊 Triple POC + Future Swing Analysis")
            if poc_data_for_chart:
                st.markdown("### 🎯 Triple Point of Control (POC)")
                poc_table_data = []
                current_price_for_poc = df['close'].iloc[-1] if not df.empty else 0
                for poc_key, period_key in [('poc1', 'poc1'), ('poc2', 'poc2'), ('poc3', 'poc3')]:
                    poc = poc_data_for_chart.get(poc_key)
                    period = poc_data_for_chart.get('periods', {}).get(period_key, '')
                    if poc:
                        if current_price_for_poc > poc.get('upper_poc', 0):
                            position = "🟢 Above"
                            signal = "Bullish"
                        elif current_price_for_poc < poc.get('lower_poc', 0):
                            position = "🔴 Below"
                            signal = "Bearish"
                        else:
                            position = "🟡 Inside"
                            signal = "Neutral"
                        poc_table_data.append({
                            'POC': f"POC {poc_key[-1]} ({period})",
                            'Value': f"₹{poc.get('poc', 0):.2f}",
                            'Upper': f"₹{poc.get('upper_poc', 0):.2f}",
                            'Lower': f"₹{poc.get('lower_poc', 0):.2f}",
                            'Range': f"₹{poc.get('high', 0):.0f} - ₹{poc.get('low', 0):.0f}",
                            'Position': position,
                            'Signal': signal
                        })
                if poc_table_data:
                    poc_df = pd.DataFrame(poc_table_data)
                    def style_poc_signal(val):
                        if val == 'Bullish':
                            return 'background-color: #00ff8840; color: white'
                        elif val == 'Bearish':
                            return 'background-color: #ff444440; color: white'
                        else:
                            return 'background-color: #FFD70040; color: white'
                    styled_poc = poc_df.style.map(style_poc_signal, subset=['Signal'])
                    st.dataframe(styled_poc, use_container_width=True, hide_index=True)
                    st.markdown("""
                    **POC Interpretation:**
                    - **POC 1 (25)**: Short-term volume profile - intraday support/resistance
                    - **POC 2 (40)**: Medium-term volume profile - swing trading levels
                    - **POC 3 (100)**: Long-term volume profile - major support/resistance
                    - **Above POC**: Bullish bias - POC acts as support
                    - **Below POC**: Bearish bias - POC acts as resistance
                    - **Inside POC**: Neutral - price consolidating at high-volume zone
                    """)
            if swing_data_for_chart:
                st.markdown("### 🔄 Future Swing Projection")
                swings = swing_data_for_chart.get('swings', {})
                projection = swing_data_for_chart.get('projection')
                volume = swing_data_for_chart.get('volume', {})
                percentages = swing_data_for_chart.get('percentages', [])
                def _swing_card(color, title, h2, sub=""):
                    p = f'<p style="color: white; margin: 0;">{sub}</p>' if sub else ""
                    return f'<div style="background-color: {color}20; padding: 15px; border-radius: 10px; border: 2px solid {color};"><h4 style="color: {color}; margin: 0;">{title}</h4><h2 style="color: {color}; margin: 5px 0;">{h2}</h2>{p}</div>'
                direction = swings.get('direction', 'Unknown')
                dir_c = "#15dd7c" if direction == 'bullish' else "#eb7514"
                delta = volume.get('delta', 0)
                dc = "#15dd7c" if delta > 0 else "#eb7514"
                sc1, sc2, sc3 = st.columns(3)
                sc1.markdown(_swing_card(dir_c, "Current Direction", f"{'🟢' if direction=='bullish' else '🔴'} {direction.upper()}"), unsafe_allow_html=True)
                if projection:
                    tc = "#15dd7c" if projection['direction'] == 'bullish' else "#eb7514"
                    sc2.markdown(_swing_card(tc, "Projected Target", f"₹{projection['target']:.0f}", f"{projection['sign']}{projection['swing_pct']:.1f}%"), unsafe_allow_html=True)
                else:
                    sc2.info("Projection not available")
                sc3.markdown(_swing_card(dc, "Volume Delta", f"{'🟢' if delta>0 else '🔴'} {delta:+,.0f}", f"Buy: {volume.get('buy_volume',0):,.0f} | Sell: {volume.get('sell_volume',0):,.0f}"), unsafe_allow_html=True)
                if percentages:
                    st.markdown("### 📈 Historical Swing Percentages")
                    swing_pct_data = []
                    for i, pct in enumerate(percentages):
                        swing_pct_data.append({
                            'Swing': f"Swing {i+1}",
                            'Percentage': f"{pct:+.2f}%",
                            'Type': '🟢 Bullish' if pct > 0 else '🔴 Bearish'
                        })
                    avg_pct = sum(abs(p) for p in percentages) / len(percentages) if percentages else 0
                    swing_pct_data.append({
                        'Swing': '📊 Average',
                        'Percentage': f"{avg_pct:.2f}%",
                        'Type': 'Used for projection'
                    })
                    swing_pct_df = pd.DataFrame(swing_pct_data)
                    st.dataframe(swing_pct_df, use_container_width=True, hide_index=True)
                st.markdown("### 📍 Swing Levels")
                swing_levels_data = []
                last_high = swings.get('last_swing_high')
                last_low = swings.get('last_swing_low')
                if last_high:
                    swing_levels_data.append({
                        'Type': '🔴 Swing High',
                        'Value': f"₹{last_high['value']:.2f}",
                        'Index': last_high['index']
                    })
                if last_low:
                    swing_levels_data.append({
                        'Type': '🟢 Swing Low',
                        'Value': f"₹{last_low['value']:.2f}",
                        'Index': last_low['index']
                    })
                if swing_levels_data:
                    swing_levels_df = pd.DataFrame(swing_levels_data)
                    st.dataframe(swing_levels_df, use_container_width=True, hide_index=True)
                st.markdown("""
                **Swing Analysis Interpretation:**
                - **Swing High**: Resistance level where price reversed down
                - **Swing Low**: Support level where price reversed up
                - **Volume Delta**: Positive = more buying, Negative = more selling
                - **Projected Target**: Based on average of historical swing percentages
                """)
        else:
            st.error("No data available. Please check your API credentials and try again.")

    with col2:
        option_data = analyze_option_chain(selected_expiry, pivots, vob_data)
        if option_data and option_data.get('underlying'):
            st.session_state._cached_option_data = option_data
            underlying_price = option_data['underlying']
            df_summary = option_data['df_summary']
            expiry = option_data.get('expiry', selected_expiry)
            atm_strike = df_summary.loc[df_summary['Zone'] == 'ATM', 'Strike'].values[0] if df_summary is not None and 'Zone' in df_summary.columns and not df_summary[df_summary['Zone'] == 'ATM'].empty else underlying_price
            # Store option chain, ATM strike data, and orderbook to Supabase
            if df_summary is not None and len(df_summary) > 0 and expiry:
                try:
                    db.upsert_option_chain(df_summary, expiry, underlying_price, atm_strike)
                    db.upsert_atm_strike_data(df_summary, expiry, underlying_price, atm_strike)
                    # Store orderbook depth data
                    orderbook_entries = []
                    for _, row in df_summary.iterrows():
                        orderbook_entries.append({
                            'strike': float(row['Strike']),
                            'bid_qty_ce': int(row.get('bidQty_CE', 0) or 0),
                            'ask_qty_ce': int(row.get('askQty_CE', 0) or 0),
                            'bid_qty_pe': int(row.get('bidQty_PE', 0) or 0),
                            'ask_qty_pe': int(row.get('askQty_PE', 0) or 0),
                            'pressure': float(row.get('BidAskPressure', 0) or 0),
                            'bias': str(row.get('PressureBias', '') or '')
                        })
                    if orderbook_entries:
                        db.upsert_orderbook(orderbook_entries, expiry)
                except Exception:
                    pass
            if enable_signals and not df.empty and df_summary is not None and len(df_summary) > 0:
                check_trading_signals(df, pivot_settings, df_summary, underlying_price, pivot_proximity)
            if df_summary is not None and len(df_summary) > 0:
                check_atm_verdict_alert(df_summary, underlying_price)
        else:
            option_data = None

    if option_data and option_data.get('underlying'):
        st.markdown("---")
        st.header("📊 Options Chain Analysis")
        st.markdown("## Open Interest Change (in Lakhs)")
        oi_col1, oi_col2 = st.columns(2)
        with oi_col1:
            st.metric("CALL ΔOI", f"{option_data['total_ce_change']:+.1f}L", delta_color="inverse")
        with oi_col2:
            st.metric("PUT ΔOI", f"{option_data['total_pe_change']:+.1f}L", delta_color="normal")
        st.markdown("## Option Chain Bias Summary")
        if option_data.get('styled_df') is not None:
            st.dataframe(option_data['styled_df'], use_container_width=True)

        # === MONEY FLOW ANALYSIS (from Option Chain) ===
        st.markdown("---")
        st.markdown("## 💰 Money Flow Analysis (ATM ±4 Strikes)")
        st.caption("Tracks where smart money is entering/exiting using OI change + Volume + Price | ATM ±200 pts (9 strikes)")
        df_mf_source = option_data.get('df_atm4')
        underlying_price = option_data['underlying']
        atm_strike_mf = option_data.get('atm_strike', underlying_price)
        if df_mf_source is not None and len(df_mf_source) > 0:
            try:
                mf_rows = []
                total_long_buildup_ce = 0
                total_short_buildup_ce = 0
                total_long_unwind_ce = 0
                total_short_cover_ce = 0
                total_long_buildup_pe = 0
                total_short_buildup_pe = 0
                total_long_unwind_pe = 0
                total_short_cover_pe = 0
                fresh_money_in = 0
                money_exit = 0

                for _, row in df_mf_source.iterrows():
                    strike = row.get('strikePrice', 0)
                    zone = row.get('Zone', '')
                    chg_oi_ce = row.get('changeinOpenInterest_CE', 0) or 0
                    chg_oi_pe = row.get('changeinOpenInterest_PE', 0) or 0
                    ltp_ce = row.get('lastPrice_CE', 0) or 0
                    ltp_pe = row.get('lastPrice_PE', 0) or 0
                    prev_oi_ce = row.get('previousOpenInterest_CE', 0) or 0
                    prev_oi_pe = row.get('previousOpenInterest_PE', 0) or 0
                    vol_ce = row.get('totalTradedVolume_CE', 0) or 0
                    vol_pe = row.get('totalTradedVolume_PE', 0) or 0
                    iv_ce = row.get('impliedVolatility_CE', 0) or 0
                    iv_pe = row.get('impliedVolatility_PE', 0) or 0

                    # CE Money Flow: OI change + Price direction + Volume
                    # OI ↑ + Price ↑ = Long Buildup (buyers entering) → Bullish for CE
                    # OI ↑ + Price ↓ = Short Buildup (writers entering) → Bearish for market
                    # OI ↓ + Price ↓ = Long Unwinding (buyers exiting) → Bearish
                    # OI ↓ + Price ↑ = Short Covering (writers exiting) → Bullish for market
                    ce_oi_pct = (chg_oi_ce / prev_oi_ce * 100) if prev_oi_ce > 0 else 0
                    # Volume confirmation: high vol = strong conviction
                    ce_vol_strength = 'High' if vol_ce > 0 and prev_oi_ce > 0 and vol_ce > prev_oi_ce * 0.1 else 'Low'
                    pe_vol_strength = 'High' if vol_pe > 0 and prev_oi_pe > 0 and vol_pe > prev_oi_pe * 0.1 else 'Low'

                    if chg_oi_ce > 0 and iv_ce > 0:
                        # OI increasing - check IV to determine long vs short
                        # IV rising + OI up = long buildup (buyers), IV falling + OI up = short buildup (writers)
                        ce_flow = 'Short Buildup'  # Default: writers selling CE = Bearish
                        ce_impact = 'Bearish'
                        total_short_buildup_ce += abs(chg_oi_ce)
                        fresh_money_in += abs(chg_oi_ce)
                    elif chg_oi_ce < 0:
                        ce_flow = 'Short Covering'  # Writers covering CE = Bullish
                        ce_impact = 'Bullish'
                        total_short_cover_ce += abs(chg_oi_ce)
                        money_exit += abs(chg_oi_ce)
                    else:
                        ce_flow = 'No Activity'
                        ce_impact = 'Neutral'

                    # PE Money Flow (OI change + Volume)
                    pe_oi_pct = (chg_oi_pe / prev_oi_pe * 100) if prev_oi_pe > 0 else 0
                    if chg_oi_pe > 0 and iv_pe > 0:
                        pe_flow = 'Short Buildup'  # Writers selling PE = Bullish (support)
                        pe_impact = 'Bullish'
                        total_short_buildup_pe += abs(chg_oi_pe)
                        fresh_money_in += abs(chg_oi_pe)
                    elif chg_oi_pe < 0:
                        pe_flow = 'Short Covering'  # Writers covering PE = Bearish
                        pe_impact = 'Bearish'
                        total_short_cover_pe += abs(chg_oi_pe)
                        money_exit += abs(chg_oi_pe)
                    else:
                        pe_flow = 'No Activity'
                        pe_impact = 'Neutral'

                    # Volume-weighted money flow (₹ value = OI change × price, confirmed by volume)
                    ce_money = (chg_oi_ce * ltp_ce) / 100000 if ltp_ce > 0 else 0
                    pe_money = (chg_oi_pe * ltp_pe) / 100000 if ltp_pe > 0 else 0
                    net_money = ce_money + pe_money
                    # Volume-adjusted flow: scale by volume ratio for conviction
                    vol_total = vol_ce + vol_pe
                    vol_ratio = vol_ce / vol_pe if vol_pe > 0 else (99 if vol_ce > 0 else 0)

                    # Net strike impact for market
                    # CE OI up = bearish, PE OI up = bullish
                    if chg_oi_pe > chg_oi_ce and chg_oi_pe > 0:
                        strike_verdict = 'Support Building'
                    elif chg_oi_ce > chg_oi_pe and chg_oi_ce > 0:
                        strike_verdict = 'Resistance Building'
                    elif chg_oi_pe < 0 and chg_oi_ce < 0:
                        strike_verdict = 'Money Exiting'
                    elif chg_oi_pe < 0 and abs(chg_oi_pe) > abs(chg_oi_ce):
                        strike_verdict = 'Support Breaking'
                    elif chg_oi_ce < 0 and abs(chg_oi_ce) > abs(chg_oi_pe):
                        strike_verdict = 'Resistance Breaking'
                    else:
                        strike_verdict = 'Neutral'

                    mf_rows.append({
                        'Strike': int(strike),
                        'Zone': zone,
                        'CE ΔOI': f"{chg_oi_ce/1000:+.1f}K",
                        'CE Flow': ce_flow,
                        'CE Vol': f"{vol_ce/1000:.0f}K" if vol_ce > 0 else '-',
                        'PE ΔOI': f"{chg_oi_pe/1000:+.1f}K",
                        'PE Flow': pe_flow,
                        'PE Vol': f"{vol_pe/1000:.0f}K" if vol_pe > 0 else '-',
                        'Vol Ratio': f"{vol_ratio:.1f}" if vol_ratio < 99 else 'CE Only' if vol_ce > 0 else '-',
                        'Net ₹ Flow': f"₹{net_money:+.1f}L",
                        'Conviction': f"{'🔥' if ce_vol_strength == 'High' or pe_vol_strength == 'High' else '⚪'} {'High' if ce_vol_strength == 'High' or pe_vol_strength == 'High' else 'Low'}",
                        'Verdict': strike_verdict,
                        '_net_money': net_money,
                        '_chg_ce': chg_oi_ce,
                        '_chg_pe': chg_oi_pe,
                        '_vol_total': vol_total,
                    })

                if mf_rows:
                    # === Money Flow Summary Metrics ===
                    support_strikes = sum(1 for r in mf_rows if r['Verdict'] == 'Support Building')
                    resistance_strikes = sum(1 for r in mf_rows if r['Verdict'] == 'Resistance Building')
                    high_conviction = sum(1 for r in mf_rows if 'High' in r.get('Conviction', ''))
                    total_vol = sum(r['_vol_total'] for r in mf_rows)

                    mf_col1, mf_col2, mf_col3, mf_col4, mf_col5 = st.columns(5)
                    with mf_col1:
                        net_flow = sum(r['_net_money'] for r in mf_rows)
                        flow_dir = "Inflow" if net_flow > 0 else "Outflow"
                        st.metric("Net Money Flow", f"₹{abs(net_flow):.1f}L", delta=flow_dir)
                    with mf_col2:
                        st.metric("Fresh Money In", f"{fresh_money_in/1000:.0f}K OI")
                    with mf_col3:
                        st.metric("Total Volume", f"{total_vol/1000:.0f}K")
                    with mf_col4:
                        st.metric("Support Building", f"{support_strikes} strikes")
                    with mf_col5:
                        st.metric("High Conviction", f"{high_conviction} strikes")

                    # === Overall Money Flow Verdict ===
                    ce_oi_total = sum(r['_chg_ce'] for r in mf_rows)
                    pe_oi_total = sum(r['_chg_pe'] for r in mf_rows)

                    if pe_oi_total > 0 and ce_oi_total > 0:
                        if pe_oi_total > ce_oi_total:
                            st.success(f"💰 SMART MONEY BULLISH — PE writers building support (PE ΔOI: {pe_oi_total/1000:+.0f}K > CE ΔOI: {ce_oi_total/1000:+.0f}K) | More puts being written = market support")
                        else:
                            st.error(f"💰 SMART MONEY BEARISH — CE writers building resistance (CE ΔOI: {ce_oi_total/1000:+.0f}K > PE ΔOI: {pe_oi_total/1000:+.0f}K) | More calls being written = market cap")
                    elif pe_oi_total < 0 and ce_oi_total < 0:
                        st.warning(f"💰 MONEY EXITING — Both CE & PE OI declining (CE: {ce_oi_total/1000:+.0f}K, PE: {pe_oi_total/1000:+.0f}K) | Expiry unwinding or uncertainty")
                    elif pe_oi_total < 0 and ce_oi_total > 0:
                        st.error(f"💰 BEARISH SHIFT — PE support breaking + CE resistance building (CE: {ce_oi_total/1000:+.0f}K, PE: {pe_oi_total/1000:+.0f}K)")
                    elif pe_oi_total > 0 and ce_oi_total < 0:
                        st.success(f"💰 BULLISH SHIFT — PE support building + CE resistance breaking (CE: {ce_oi_total/1000:+.0f}K, PE: {pe_oi_total/1000:+.0f}K)")
                    else:
                        st.info("💰 No significant money flow detected")

                    # === Where is Money Flowing? Top strikes ===
                    mf_sorted = sorted(mf_rows, key=lambda x: abs(x['_net_money']), reverse=True)
                    top_inflow = [r for r in mf_sorted if r['_net_money'] > 0][:3]
                    top_outflow = [r for r in mf_sorted if r['_net_money'] < 0][:3]

                    flow_col1, flow_col2 = st.columns(2)
                    with flow_col1:
                        st.markdown("**🟢 Top Money Inflow Strikes**")
                        for r in top_inflow:
                            st.markdown(f"**{r['Strike']}** ({r['Zone']}) — {r['Net ₹ Flow']} | {r['Verdict']}")
                        if not top_inflow:
                            st.caption("No significant inflow")
                    with flow_col2:
                        st.markdown("**🔴 Top Money Outflow Strikes**")
                        for r in top_outflow:
                            st.markdown(f"**{r['Strike']}** ({r['Zone']}) — {r['Net ₹ Flow']} | {r['Verdict']}")
                        if not top_outflow:
                            st.caption("No significant outflow")

                    # === Full Money Flow Table ===
                    with st.expander("📋 Strike-wise Money Flow Details"):
                        mf_df = pd.DataFrame(mf_rows)
                        display_mf = mf_df.drop(columns=['_net_money', '_chg_ce', '_chg_pe', '_vol_total'])
                        def _style_mf(row):
                            v = row.get('Verdict', '')
                            if 'Support' in v and 'Breaking' not in v:
                                return ['background-color:#00ff8812;color:white'] * len(row)
                            elif 'Resistance' in v and 'Breaking' not in v:
                                return ['background-color:#ff444412;color:white'] * len(row)
                            elif 'Breaking' in v:
                                return ['background-color:#FFD70015;color:white'] * len(row)
                            elif 'Exiting' in v:
                                return ['background-color:#88888815;color:white'] * len(row)
                            return [''] * len(row)
                        st.dataframe(display_mf.style.apply(_style_mf, axis=1), use_container_width=True, hide_index=True)

            except Exception as e:
                st.caption(f"Money flow loading... ({str(e)[:60]})")

        # === OI UNWINDING & PARALLEL WINDING (ATM ±5 Strikes) ===
        st.markdown("---")
        st.markdown("## 🔄 OI Unwinding & Parallel Winding (ATM ±5 Strikes)")
        st.caption("Detects where positions are closing (unwinding) and where new positions are building simultaneously (parallel winding) | ATM ±250 pts (11 strikes)")
        df_atm8 = option_data.get('df_atm8')
        if df_atm8 is not None and len(df_atm8) > 0:
            try:
                atm_s = option_data.get('atm_strike', underlying_price)
                unwind_rows = []
                ce_unwinding = []
                pe_unwinding = []
                ce_winding = []
                pe_winding = []
                parallel_pairs = []

                for _, row in df_atm8.iterrows():
                    strike = row.get('strikePrice', 0)
                    zone = row.get('Zone', '')
                    chg_ce = row.get('changeinOpenInterest_CE', 0) or 0
                    chg_pe = row.get('changeinOpenInterest_PE', 0) or 0
                    oi_ce = row.get('openInterest_CE', 0) or 0
                    oi_pe = row.get('openInterest_PE', 0) or 0
                    prev_ce = row.get('previousOpenInterest_CE', 0) or 0
                    prev_pe = row.get('previousOpenInterest_PE', 0) or 0
                    vol_ce = row.get('totalTradedVolume_CE', 0) or 0
                    vol_pe = row.get('totalTradedVolume_PE', 0) or 0
                    ltp_ce = row.get('lastPrice_CE', 0) or 0
                    ltp_pe = row.get('lastPrice_PE', 0) or 0

                    ce_pct = (chg_ce / prev_ce * 100) if prev_ce > 0 else 0
                    pe_pct = (chg_pe / prev_pe * 100) if prev_pe > 0 else 0

                    # Classify CE activity
                    if chg_ce < 0 and abs(chg_ce) > 1000:
                        ce_activity = 'Unwinding'
                        ce_unwinding.append({'strike': strike, 'chg': chg_ce, 'pct': ce_pct, 'vol': vol_ce})
                    elif chg_ce > 0 and chg_ce > 1000:
                        ce_activity = 'Buildup'
                        ce_winding.append({'strike': strike, 'chg': chg_ce, 'pct': ce_pct, 'vol': vol_ce})
                    else:
                        ce_activity = 'Flat'

                    # Classify PE activity
                    if chg_pe < 0 and abs(chg_pe) > 1000:
                        pe_activity = 'Unwinding'
                        pe_unwinding.append({'strike': strike, 'chg': chg_pe, 'pct': pe_pct, 'vol': vol_pe})
                    elif chg_pe > 0 and chg_pe > 1000:
                        pe_activity = 'Buildup'
                        pe_winding.append({'strike': strike, 'chg': chg_pe, 'pct': pe_pct, 'vol': vol_pe})
                    else:
                        pe_activity = 'Flat'

                    # Detect parallel winding: one side unwinding + other side building at SAME strike
                    parallel = ''
                    parallel_impact = ''
                    if ce_activity == 'Unwinding' and pe_activity == 'Buildup':
                        parallel = 'CE Unwind + PE Buildup'
                        parallel_impact = 'Bullish Shift'
                        parallel_pairs.append({'strike': strike, 'type': parallel, 'impact': parallel_impact})
                    elif pe_activity == 'Unwinding' and ce_activity == 'Buildup':
                        parallel = 'PE Unwind + CE Buildup'
                        parallel_impact = 'Bearish Shift'
                        parallel_pairs.append({'strike': strike, 'type': parallel, 'impact': parallel_impact})
                    elif ce_activity == 'Unwinding' and pe_activity == 'Unwinding':
                        parallel = 'Both Unwinding'
                        parallel_impact = 'Expiry Exit'
                        parallel_pairs.append({'strike': strike, 'type': parallel, 'impact': parallel_impact})
                    elif ce_activity == 'Buildup' and pe_activity == 'Buildup':
                        parallel = 'Both Buildup'
                        parallel_impact = 'High Activity'
                        parallel_pairs.append({'strike': strike, 'type': parallel, 'impact': parallel_impact})

                    # Only show strikes with activity
                    if ce_activity != 'Flat' or pe_activity != 'Flat':
                        ce_emoji = '🔻' if ce_activity == 'Unwinding' else '🔺' if ce_activity == 'Buildup' else '➖'
                        pe_emoji = '🔻' if pe_activity == 'Unwinding' else '🔺' if pe_activity == 'Buildup' else '➖'
                        par_emoji = '⚡' if parallel else ''
                        atm_tag = ' [ATM]' if strike == atm_s else ''
                        unwind_rows.append({
                            'Strike': f"{int(strike)}{atm_tag}",
                            'CE OI': f"{oi_ce/1000:.0f}K",
                            'CE ΔOI': f"{chg_ce/1000:+.1f}K ({ce_pct:+.1f}%)",
                            'CE': f"{ce_emoji} {ce_activity}",
                            'CE Vol': f"{vol_ce/1000:.0f}K",
                            'PE OI': f"{oi_pe/1000:.0f}K",
                            'PE ΔOI': f"{chg_pe/1000:+.1f}K ({pe_pct:+.1f}%)",
                            'PE': f"{pe_emoji} {pe_activity}",
                            'PE Vol': f"{vol_pe/1000:.0f}K",
                            'Parallel': f"{par_emoji} {parallel}" if parallel else '-',
                            'Impact': parallel_impact if parallel_impact else '-',
                            '_strike': strike,
                        })

                # === Summary metrics ===
                uw_col1, uw_col2, uw_col3, uw_col4, uw_col5 = st.columns(5)
                with uw_col1:
                    st.metric("CE Unwinding", f"{len(ce_unwinding)} strikes")
                with uw_col2:
                    st.metric("PE Unwinding", f"{len(pe_unwinding)} strikes")
                with uw_col3:
                    st.metric("CE Buildup", f"{len(ce_winding)} strikes")
                with uw_col4:
                    st.metric("PE Buildup", f"{len(pe_winding)} strikes")
                with uw_col5:
                    st.metric("Parallel Activity", f"{len(parallel_pairs)} strikes")

                # === Parallel Winding Signal ===
                bull_parallel = sum(1 for p in parallel_pairs if p['impact'] == 'Bullish Shift')
                bear_parallel = sum(1 for p in parallel_pairs if p['impact'] == 'Bearish Shift')
                both_unwind = sum(1 for p in parallel_pairs if p['impact'] == 'Expiry Exit')

                if bull_parallel > bear_parallel and bull_parallel > 0:
                    par_strikes = ', '.join([str(int(p['strike'])) for p in parallel_pairs if p['impact'] == 'Bullish Shift'])
                    st.success(f"⚡ BULLISH PARALLEL WINDING — CE unwinding + PE buildup at {bull_parallel} strikes ({par_strikes}) | Writers shifting from calls to puts = expecting move UP")
                elif bear_parallel > bull_parallel and bear_parallel > 0:
                    par_strikes = ', '.join([str(int(p['strike'])) for p in parallel_pairs if p['impact'] == 'Bearish Shift'])
                    st.error(f"⚡ BEARISH PARALLEL WINDING — PE unwinding + CE buildup at {bear_parallel} strikes ({par_strikes}) | Writers shifting from puts to calls = expecting move DOWN")
                elif both_unwind > 0 and bull_parallel == 0 and bear_parallel == 0:
                    st.warning(f"⚡ MASS UNWINDING — Both CE & PE positions closing at {both_unwind} strikes | Expiry/event-driven exit, expect volatility")
                elif len(parallel_pairs) == 0 and (len(ce_unwinding) > 0 or len(pe_unwinding) > 0):
                    if len(ce_unwinding) > len(pe_unwinding):
                        uw_strikes = ', '.join([str(int(u['strike'])) for u in sorted(ce_unwinding, key=lambda x: x['chg'])[:3]])
                        st.success(f"🔄 CE UNWINDING dominant at {len(ce_unwinding)} strikes ({uw_strikes}) | Call writers exiting = resistance breaking = Bullish")
                    elif len(pe_unwinding) > len(ce_unwinding):
                        uw_strikes = ', '.join([str(int(u['strike'])) for u in sorted(pe_unwinding, key=lambda x: x['chg'])[:3]])
                        st.error(f"🔄 PE UNWINDING dominant at {len(pe_unwinding)} strikes ({uw_strikes}) | Put writers exiting = support breaking = Bearish")
                    else:
                        st.info("🔄 Balanced unwinding across CE and PE | No clear directional bias")
                else:
                    st.info("No significant unwinding or parallel winding detected")

                # === Detailed Unwinding Breakdown ===
                uw_detail_col1, uw_detail_col2 = st.columns(2)
                with uw_detail_col1:
                    st.markdown("**🔻 CE Unwinding (Resistance Breaking)**")
                    if ce_unwinding:
                        for u in sorted(ce_unwinding, key=lambda x: x['chg']):
                            st.markdown(f"**{int(u['strike'])}** — OI: {u['chg']/1000:+.1f}K ({u['pct']:+.1f}%) | Vol: {u['vol']/1000:.0f}K")
                    else:
                        st.caption("No CE unwinding")
                    st.markdown("**🔺 CE Buildup (Resistance Building)**")
                    if ce_winding:
                        for w in sorted(ce_winding, key=lambda x: -x['chg']):
                            st.markdown(f"**{int(w['strike'])}** — OI: {w['chg']/1000:+.1f}K ({w['pct']:+.1f}%) | Vol: {w['vol']/1000:.0f}K")
                    else:
                        st.caption("No CE buildup")
                with uw_detail_col2:
                    st.markdown("**🔻 PE Unwinding (Support Breaking)**")
                    if pe_unwinding:
                        for u in sorted(pe_unwinding, key=lambda x: x['chg']):
                            st.markdown(f"**{int(u['strike'])}** — OI: {u['chg']/1000:+.1f}K ({u['pct']:+.1f}%) | Vol: {u['vol']/1000:.0f}K")
                    else:
                        st.caption("No PE unwinding")
                    st.markdown("**🔺 PE Buildup (Support Building)**")
                    if pe_winding:
                        for w in sorted(pe_winding, key=lambda x: -x['chg']):
                            st.markdown(f"**{int(w['strike'])}** — OI: {w['chg']/1000:+.1f}K ({w['pct']:+.1f}%) | Vol: {w['vol']/1000:.0f}K")
                    else:
                        st.caption("No PE buildup")

                # === Full Table ===
                if unwind_rows:
                    with st.expander("📋 Full Strike-wise OI Activity (ATM ±5)"):
                        uw_df = pd.DataFrame(unwind_rows).drop(columns=['_strike'])
                        def _style_unwind(row):
                            impact = row.get('Impact', '')
                            if 'Bullish' in impact:
                                return ['background-color:#00ff8812;color:white'] * len(row)
                            elif 'Bearish' in impact:
                                return ['background-color:#ff444412;color:white'] * len(row)
                            elif 'Exit' in impact or 'Unwind' in str(row.get('CE', '')) + str(row.get('PE', '')):
                                return ['background-color:#FFD70010;color:white'] * len(row)
                            return [''] * len(row)
                        st.dataframe(uw_df.style.apply(_style_unwind, axis=1), use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"Unwinding analysis loading... ({str(e)[:60]})")

        st.markdown("---")
        st.markdown("## 📈 HTF Support & Resistance Levels")
        sr_data = option_data.get('sr_data', [])
        max_pain_strike = option_data.get('max_pain_strike')
        # Store max pain to Supabase
        if max_pain_strike and option_data.get('expiry'):
            try:
                db.upsert_max_pain(option_data['expiry'], max_pain_strike, option_data.get('underlying'))
            except Exception:
                pass
        if sr_data:
            support_data = [d for d in sr_data if '🟢' in d['Type'] or '🎯' in d['Type']]
            resistance_data = [d for d in sr_data if '🔴' in d['Type']]
            for col, title, data in zip(st.columns(2),
                ["### 🟢 SUPPORT LEVELS", "### 🔴 RESISTANCE LEVELS"],
                [support_data, resistance_data]):
                with col:
                    st.markdown(title)
                    if data:
                        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                    else:
                        st.info("No levels identified")
            if max_pain_strike:
                st.info(f"🎯 **Max Pain Level:** ₹{max_pain_strike:.0f} - Price magnet at expiry")
        st.markdown("---")
        st.markdown("## 🔍 Option Chain Deep Analysis (ATM ± 5)")
        try:
            sa_df_summary = option_data.get('df_summary')
            sa_underlying = option_data.get('underlying')
            if sa_df_summary is not None and sa_underlying:
                sa_result = analyze_strike_activity(sa_df_summary, sa_underlying)
                if sa_result:
                    # Send Telegram signal (with cooldown to avoid spam)
                    if 'last_chain_signal_time' not in st.session_state:
                        st.session_state.last_chain_signal_time = None
                    ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                    should_send = st.session_state.last_chain_signal_time is None or \
                        (ist_now - st.session_state.last_chain_signal_time).total_seconds() > 300
                    if should_send:
                        try:
                            send_option_chain_signal(sa_result, sa_underlying)
                            st.session_state.last_chain_signal_time = ist_now
                        except Exception:
                            pass
                    st.session_state._sa_result = sa_result
                    # ── Decapping / Depeg: track OI for ATM±2 strikes across snapshots ──
                    try:
                        _adf2 = sa_result.get('analysis_df')
                        if _adf2 is not None and not _adf2.empty and sa_underlying:
                            _prev_snap = dict(getattr(st.session_state, '_prev_strike_oi', {}))

                            # Determine ATM strike and gap from the analysis df
                            _all_sks = sorted(_adf2['Strike'].unique())
                            _atm_sk  = min(_all_sks, key=lambda x: abs(x - sa_underlying))
                            _gaps    = [_all_sks[i+1] - _all_sks[i] for i in range(len(_all_sks)-1)]
                            _gap     = int(min(_gaps)) if _gaps else 50

                            _decap_atm_list = []
                            _dominant_decap = None
                            _dominant_depeg = None

                            for _off in [-2, -1, 0, 1, 2]:
                                _tsk    = _atm_sk + _off * _gap
                                _closest = min(_all_sks, key=lambda x: abs(x - _tsk))
                                if abs(_closest - _tsk) > _gap * 0.6:
                                    continue
                                _sk_str  = str(int(_closest))
                                _row     = _adf2[_adf2['Strike'] == _closest]
                                if _row.empty:
                                    continue
                                _rv      = _row.iloc[0]
                                _cur_ce  = float(_rv.get('CE_OI', 0))
                                _cur_pe  = float(_rv.get('PE_OI', 0))
                                _prev_ce = _prev_snap.get(_sk_str, {}).get('ce_oi', 0)
                                _prev_pe = _prev_snap.get(_sk_str, {}).get('pe_oi', 0)
                                _lbl     = f'ATM{_off:+d}' if _off != 0 else 'ATM'

                                _entry = {
                                    'strike': float(_closest), 'label': _lbl, 'offset': _off,
                                    'ce_oi_l': _cur_ce / 100000, 'pe_oi_l': _cur_pe / 100000,
                                    'ce_decapping': False, 'pe_depeg': False,
                                    'ce_capping':   False, 'pe_pegging': False,
                                    'ce_shed_pct': 0.0, 'pe_shed_pct': 0.0,
                                    'ce_build_pct': 0.0, 'pe_build_pct': 0.0,
                                }
                                if _prev_ce > 50000:
                                    if _cur_ce < _prev_ce:          # CE OI falling → decapping
                                        _shed = (_prev_ce - _cur_ce) / _prev_ce * 100
                                        _entry.update({'ce_decapping': True, 'ce_shed_pct': _shed,
                                                       'prev_ce_oi_l': _prev_ce / 100000})
                                        if _off > 0 and (_dominant_decap is None or _shed > _dominant_decap['shed_pct']):
                                            _dominant_decap = {
                                                'strike': float(_closest), 'shed_pct': _shed,
                                                'oi_l': _cur_ce / 100000, 'prev_oi_l': _prev_ce / 100000,
                                                'activity': str(_rv.get('Call_Activity', '')),
                                            }
                                    elif _cur_ce > _prev_ce:         # CE OI rising → capping
                                        _build = (_cur_ce - _prev_ce) / _prev_ce * 100
                                        _entry.update({'ce_capping': True, 'ce_build_pct': _build,
                                                       'prev_ce_oi_l': _prev_ce / 100000})
                                if _prev_pe > 50000:
                                    if _cur_pe < _prev_pe:           # PE OI falling → depeg
                                        _shed = (_prev_pe - _cur_pe) / _prev_pe * 100
                                        _entry.update({'pe_depeg': True, 'pe_shed_pct': _shed,
                                                       'prev_pe_oi_l': _prev_pe / 100000})
                                        if _off < 0 and (_dominant_depeg is None or _shed > _dominant_depeg['shed_pct']):
                                            _dominant_depeg = {
                                                'strike': float(_closest), 'shed_pct': _shed,
                                                'oi_l': _cur_pe / 100000, 'prev_oi_l': _prev_pe / 100000,
                                                'activity': str(_rv.get('Put_Activity', '')),
                                            }
                                    elif _cur_pe > _prev_pe:          # PE OI rising → pegging
                                        _build = (_cur_pe - _prev_pe) / _prev_pe * 100
                                        _entry.update({'pe_pegging': True, 'pe_build_pct': _build,
                                                       'prev_pe_oi_l': _prev_pe / 100000})
                                _decap_atm_list.append(_entry)

                            # Update snapshot for all strikes
                            _new_snap = {}
                            for _, _row2 in _adf2.iterrows():
                                _sk2 = str(int(_row2['Strike']))
                                _new_snap[_sk2] = {
                                    'ce_oi': float(_row2.get('CE_OI', 0)),
                                    'pe_oi': float(_row2.get('PE_OI', 0)),
                                }
                            st.session_state._prev_strike_oi = _new_snap
                            st.session_state._decap_atm_data = _decap_atm_list
                            st.session_state._decapping = _dominant_decap  # backward compat
                            st.session_state._depeg     = _dominant_depeg  # backward compat
                    except Exception:
                        pass
                    analysis_df = sa_result['analysis_df']
                    # Market Bias Banner
                    bias = sa_result['market_bias']
                    conf = sa_result['confidence']
                    if 'Bullish' in bias:
                        bias_color = '#00ff88'
                    elif 'Bearish' in bias:
                        bias_color = '#ff4444'
                    else:
                        bias_color = '#FFD700'
                    st.markdown(f"""
                    <div style="background:{bias_color}20;padding:20px;border-radius:12px;border:2px solid {bias_color};text-align:center;margin-bottom:15px;">
                        <h2 style="color:{bias_color};margin:0;">Market Bias: {bias}</h2>
                        <h3 style="color:white;margin:5px 0;">Confidence: {conf}%</h3>
                        <div style="background:#33333380;border-radius:10px;height:12px;margin:8px auto;max-width:400px;">
                            <div style="background:{bias_color};border-radius:10px;height:12px;width:{conf}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Bias signals
                    for sig in sa_result['bias_signals']:
                        st.caption(f"• {sig}")
                    # Decapping / Depeg — ATM±2 per-strike table
                    _decap_atm_ui = getattr(st.session_state, '_decap_atm_data', [])
                    if _decap_atm_ui:
                        st.markdown("**🔓 Decapping / Depeg Monitor (ATM±2)**")
                        _dc_cols = st.columns(len(_decap_atm_ui))
                        for _ci, _de in enumerate(_decap_atm_ui):
                            with _dc_cols[_ci]:
                                _bg = '#ff990030' if _de.get('ce_decapping') else ('#ff444430' if _de.get('pe_depeg') else '#1e1e1e')
                                _bdr = '#ff9900' if _de.get('ce_decapping') else ('#ff4444' if _de.get('pe_depeg') else '#444')
                                _ce_txt = (f"CE: {_de['ce_oi_l']:.1f}L<br>⚡−{_de['ce_shed_pct']:.1f}%"
                                           if _de.get('ce_decapping') else f"CE: {_de['ce_oi_l']:.1f}L")
                                _pe_txt = (f"PE: {_de['pe_oi_l']:.1f}L<br>⚡−{_de['pe_shed_pct']:.1f}%"
                                           if _de.get('pe_depeg') else f"PE: {_de['pe_oi_l']:.1f}L")
                                st.markdown(
                                    f'<div style="background:{_bg};border:1.5px solid {_bdr};border-radius:6px;'
                                    f'padding:6px 8px;text-align:center;font-family:monospace;font-size:12px;">'
                                    f'<b>{_de["label"]}</b><br>₹{_de["strike"]:.0f}<br>'
                                    f'{_ce_txt}<br>{_pe_txt}</div>',
                                    unsafe_allow_html=True
                                )
                    # Top Resistance & Support
                    sa_col1, sa_col2 = st.columns(2)
                    with sa_col1:
                        st.markdown("### 🔴 Top 3 Resistance Levels (Call Side)")
                        if not sa_result['top_resistance'].empty:
                            res_display = sa_result['top_resistance'].copy()
                            res_display['Strike'] = res_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                            res_display['CE_OI'] = (res_display['CE_OI'] / 100000).round(2).astype(str) + 'L'
                            res_display['CE_ChgOI'] = (res_display['CE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            res_display.columns = ['Strike', 'Classification', 'Strength', 'OI', 'ChgOI', 'Activity']
                            def style_res(row):
                                s = row['Strength']
                                if s == 'Strong':
                                    return ['background-color:#ff444440;color:white'] * len(row)
                                elif s == 'Breaking':
                                    return ['background-color:#FFD70040;color:white'] * len(row)
                                return [''] * len(row)
                            styled_res = res_display.style.apply(style_res, axis=1)
                            st.dataframe(styled_res, use_container_width=True, hide_index=True)
                        else:
                            st.info("No resistance levels found")
                    with sa_col2:
                        st.markdown("### 🟢 Top 3 Support Levels (Put Side)")
                        if not sa_result['top_support'].empty:
                            sup_display = sa_result['top_support'].copy()
                            sup_display['Strike'] = sup_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                            sup_display['PE_OI'] = (sup_display['PE_OI'] / 100000).round(2).astype(str) + 'L'
                            sup_display['PE_ChgOI'] = (sup_display['PE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            sup_display.columns = ['Strike', 'Classification', 'Strength', 'OI', 'ChgOI', 'Activity']
                            def style_sup(row):
                                s = row['Strength']
                                if s == 'Strong':
                                    return ['background-color:#00ff8840;color:white'] * len(row)
                                elif s == 'Breaking':
                                    return ['background-color:#FFD70040;color:white'] * len(row)
                                return [''] * len(row)
                            styled_sup = sup_display.style.apply(style_sup, axis=1)
                            st.dataframe(styled_sup, use_container_width=True, hide_index=True)
                        else:
                            st.info("No support levels found")
                    # Call Capping & Put Capping Zones
                    st.markdown("### 📊 Strike-wise Call & Put Classification")
                    class_display = analysis_df[['Strike', 'Zone', 'Call_Class', 'Call_Activity', 'CE_Vol', 'Put_Class', 'Put_Activity', 'PE_Vol']].copy()
                    class_display['Strike'] = class_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                    class_display['CE_Vol'] = class_display['CE_Vol'].apply(lambda x: f"{'🔥 ' if x > 0 else ''}{x/1000:.0f}K" if x else "0K")
                    class_display['PE_Vol'] = class_display['PE_Vol'].apply(lambda x: f"{'🔥 ' if x > 0 else ''}{x/1000:.0f}K" if x else "0K")
                    # Mark high-vol rows with flame in volume column
                    if 'CE_Vol_High' in analysis_df.columns:
                        class_display.loc[analysis_df['CE_Vol_High'].values, 'CE_Vol'] = class_display.loc[analysis_df['CE_Vol_High'].values, 'CE_Vol'].apply(lambda v: f"🔥 {v.replace('🔥 ', '')}")
                    if 'PE_Vol_High' in analysis_df.columns:
                        class_display.loc[analysis_df['PE_Vol_High'].values, 'PE_Vol'] = class_display.loc[analysis_df['PE_Vol_High'].values, 'PE_Vol'].apply(lambda v: f"🔥 {v.replace('🔥 ', '')}")
                    def style_class(row):
                        styles = [''] * len(row)
                        call_idx = class_display.columns.get_loc('Call_Class')
                        put_idx = class_display.columns.get_loc('Put_Class')
                        ce_vol_idx = class_display.columns.get_loc('CE_Vol')
                        pe_vol_idx = class_display.columns.get_loc('PE_Vol')
                        if 'High Conviction' in str(row.iloc[call_idx]):
                            styles[call_idx] = 'background-color:#ff000060;color:white;font-weight:bold'
                            styles[ce_vol_idx] = 'background-color:#ff000060;color:white;font-weight:bold'
                        elif 'Strong' in str(row.iloc[call_idx]):
                            styles[call_idx] = 'background-color:#ff444440;color:white;font-weight:bold'
                        elif 'Breakout' in str(row.iloc[call_idx]):
                            styles[call_idx] = 'background-color:#FFD70040;color:white;font-weight:bold'
                        if 'High Conviction' in str(row.iloc[put_idx]):
                            styles[put_idx] = 'background-color:#00ff0060;color:white;font-weight:bold'
                            styles[pe_vol_idx] = 'background-color:#00ff0060;color:white;font-weight:bold'
                        elif 'Strong' in str(row.iloc[put_idx]):
                            styles[put_idx] = 'background-color:#00ff8840;color:white;font-weight:bold'
                        elif 'Breakdown' in str(row.iloc[put_idx]):
                            styles[put_idx] = 'background-color:#FFD70040;color:white;font-weight:bold'
                        return styles
                    styled_class = class_display.style.apply(style_class, axis=1)
                    st.dataframe(styled_class, use_container_width=True, hide_index=True)
                    # Trapped Writers & Breakout/Breakdown
                    trap_col1, trap_col2 = st.columns(2)
                    with trap_col1:
                        st.markdown("### ⚠️ Trapped Call Writers")
                        if not sa_result['trapped_call_writers'].empty:
                            tc = sa_result['trapped_call_writers'].copy()
                            tc['Strike'] = tc['Strike'].apply(lambda x: f"₹{x:.0f}")
                            tc['CE_OI'] = (tc['CE_OI'] / 100000).round(2).astype(str) + 'L'
                            tc['CE_ChgOI'] = (tc['CE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            tc.columns = ['Strike', 'OI', 'ChgOI']
                            st.dataframe(tc, use_container_width=True, hide_index=True)
                            st.caption("Price above strike + OI falling = writers buying back (bullish)")
                        else:
                            st.success("No trapped call writers")
                    with trap_col2:
                        st.markdown("### ⚠️ Trapped Put Writers")
                        if not sa_result['trapped_put_writers'].empty:
                            tp = sa_result['trapped_put_writers'].copy()
                            tp['Strike'] = tp['Strike'].apply(lambda x: f"₹{x:.0f}")
                            tp['PE_OI'] = (tp['PE_OI'] / 100000).round(2).astype(str) + 'L'
                            tp['PE_ChgOI'] = (tp['PE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            tp.columns = ['Strike', 'OI', 'ChgOI']
                            st.dataframe(tp, use_container_width=True, hide_index=True)
                            st.caption("Price below strike + OI falling = writers buying back (bearish)")
                        else:
                            st.success("No trapped put writers")
                    bk_col1, bk_col2 = st.columns(2)
                    with bk_col1:
                        st.markdown("### 🚀 Breakout Zones")
                        if not sa_result['breakout_zones'].empty:
                            bz = sa_result['breakout_zones'].copy()
                            bz['Strike'] = bz['Strike'].apply(lambda x: f"₹{x:.0f}")
                            bz['CE_OI'] = (bz['CE_OI'] / 100000).round(2).astype(str) + 'L'
                            bz['CE_ChgOI'] = (bz['CE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            bz.columns = ['Strike', 'OI', 'ChgOI']
                            st.dataframe(bz, use_container_width=True, hide_index=True)
                            st.caption("Call OI falling + price rising = resistance breaking")
                        else:
                            st.info("No breakout zones detected")
                    with bk_col2:
                        st.markdown("### 💥 Breakdown Zones")
                        if not sa_result['breakdown_zones'].empty:
                            bd = sa_result['breakdown_zones'].copy()
                            bd['Strike'] = bd['Strike'].apply(lambda x: f"₹{x:.0f}")
                            bd['PE_OI'] = (bd['PE_OI'] / 100000).round(2).astype(str) + 'L'
                            bd['PE_ChgOI'] = (bd['PE_ChgOI'] / 1000).round(1).astype(str) + 'K'
                            bd.columns = ['Strike', 'OI', 'ChgOI']
                            st.dataframe(bd, use_container_width=True, hide_index=True)
                            st.caption("Put OI falling + price falling = support breaking")
                        else:
                            st.info("No breakdown zones detected")
                else:
                    st.warning("Unable to analyze option chain. Check data.")
        except Exception as e:
            st.warning(f"Deep analysis unavailable: {str(e)}")

        # ── OC Signal History (stored in Supabase) ────────────────────────────
        st.markdown("---")
        st.markdown("## 🗂️ OC Signal History (Today)")
        try:
            _db = st.session_state.get('db')
            if _db:
                _oc_hist_df = _db.get_oc_signals()
                if not _oc_hist_df.empty:
                    _rows = []
                    for _, _hr in _oc_hist_df.iterrows():
                        _ts = str(_hr.get('timestamp', ''))[:19].replace('T', ' ')
                        _res = _hr.get('resistance_strikes', '[]')
                        _sup = _hr.get('support_strikes', '[]')
                        try:
                            _res = json.loads(_res) if isinstance(_res, str) else _res
                            _sup = json.loads(_sup) if isinstance(_sup, str) else _sup
                        except Exception:
                            _res, _sup = [], []
                        _res_txt = ', '.join([f"₹{x['strike']}({x['strength'][:3]})" for x in _res]) or '—'
                        _sup_txt = ', '.join([f"₹{x['strike']}({x['strength'][:3]})" for x in _sup]) or '—'
                        _sigs = _hr.get('active_signals', '[]')
                        try:
                            _sigs = json.loads(_sigs) if isinstance(_sigs, str) else _sigs
                        except Exception:
                            _sigs = []
                        _rows.append({
                            'Time': _ts,
                            'Spot': f"₹{float(_hr['spot_price']):.0f}",
                            'Condition': _hr.get('condition', ''),
                            'Conf%': int(_hr.get('confidence', 0)),
                            'Resistance': _res_txt,
                            'Support': _sup_txt,
                            'Breakout': f"₹{int(_hr['breakout_level'])}" if _hr.get('breakout_level') else '—',
                            'Breakdown': f"₹{int(_hr['breakdown_level'])}" if _hr.get('breakdown_level') else '—',
                            'Active Signals': ' | '.join(_sigs) if _sigs else '—',
                        })
                    _oc_disp = pd.DataFrame(_rows)
                    def _style_oc(row):
                        c = str(row['Condition'])
                        if c == 'BULLISH':
                            return ['background-color:#00ff8820'] * len(row)
                        elif c == 'BEARISH':
                            return ['background-color:#ff444420'] * len(row)
                        return [''] * len(row)
                    st.dataframe(
                        _oc_disp.style.apply(_style_oc, axis=1),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.info("No OC signals stored yet today.")
            else:
                st.info("Database not connected.")
        except Exception as _e:
            st.warning(f"OC history unavailable: {_e}")

        st.markdown("---")
        st.markdown("## 📊 PCR Analysis - Time Series (ATM ± 2)")
        def create_pcr_chart(history_df, col_name, color, title_prefix):
            """Helper to create individual PCR chart - col_name is now just strike price"""
            if col_name and col_name in history_df.columns:
                strike_val = col_name
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df['time'],
                    y=history_df[col_name],
                    mode='lines+markers',
                    name=f'₹{strike_val}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    fill='tozeroy',
                    fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}'
                ))
                fig.add_hline(y=1.0, line_dash="dash", line_color="white", line_width=1)
                fig.add_hline(y=1.2, line_dash="dot", line_color="#00ff88", line_width=1)
                fig.add_hline(y=0.7, line_dash="dot", line_color="#ff4444", line_width=1)
                current_pcr = history_df[col_name].iloc[-1] if len(history_df) > 0 else 0
                fig.update_layout(
                    title=f"{title_prefix}<br>₹{strike_val}<br>PCR: {current_pcr:.2f}",
                    template='plotly_dark',
                    height=280,
                    showlegend=False,
                    margin=dict(l=10, r=10, t=70, b=30),
                    xaxis=dict(tickformat='%H:%M', title=''),
                    yaxis=dict(title='PCR'),
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e'
                )
                return fig, current_pcr
            return None, 0
        pcr_data_available = False
        pcr_df = None
        df_summary = option_data.get('df_summary') if option_data else None
        if df_summary is not None and 'Zone' in df_summary.columns and 'PCR' in df_summary.columns:
            try:
                atm_idx = df_summary[df_summary['Zone'] == 'ATM'].index
                if len(atm_idx) > 0:
                    atm_pos = df_summary.index.get_loc(atm_idx[0])
                    start_idx = max(0, atm_pos - 2)
                    end_idx = min(len(df_summary), atm_pos + 3)
                    pcr_extract_cols = ['Strike', 'Zone', 'PCR', 'PCR_Signal',
                                                                   'openInterest_CE', 'openInterest_PE']
                    for _chg_col in ['changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                                      'lastPrice_CE', 'lastPrice_PE']:
                        if _chg_col in df_summary.columns:
                            pcr_extract_cols.append(_chg_col)
                    pcr_df = df_summary.iloc[start_idx:end_idx][pcr_extract_cols].copy()
                    if not pcr_df.empty:
                        pcr_data_available = True
                        st.session_state.pcr_last_valid_data = pcr_df.copy()
                        ist = pytz.timezone('Asia/Kolkata')
                        current_time = datetime.now(ist)
                        pcr_entry = {'time': current_time}
                        for _, row in pcr_df.iterrows():
                            strike_label = str(int(row['Strike']))
                            pcr_entry[strike_label] = row['PCR']
                        current_strikes = pcr_df['Strike'].tolist()
                        st.session_state.pcr_current_strikes = [int(s) for s in current_strikes]
                        should_add = True
                        if st.session_state.pcr_history:
                            last_entry = st.session_state.pcr_history[-1]
                            time_diff = (current_time - last_entry['time']).total_seconds()
                            if time_diff < 30:
                                should_add = False
                        if should_add:
                            st.session_state.pcr_history.append(pcr_entry)
                            if len(st.session_state.pcr_history) > 200:
                                st.session_state.pcr_history = st.session_state.pcr_history[-200:]
                            # Persist PCR to Supabase
                            try:
                                expiry_for_pcr = option_data.get('expiry', selected_expiry) if option_data else selected_expiry
                                pcr_records = []
                                for _, row in pcr_df.iterrows():
                                    pcr_records.append({
                                        'timestamp': current_time,
                                        'expiry': expiry_for_pcr,
                                        'strike': float(row['Strike']),
                                        'atm_strike': float(atm_strike) if 'atm_strike' in dir() else 0,
                                        'pcr': float(row['PCR']) if pd.notna(row['PCR']) else 0,
                                        'oi_ce': int(row.get('openInterest_CE', 0) or 0),
                                        'oi_pe': int(row.get('openInterest_PE', 0) or 0),
                                    })
                                if pcr_records:
                                    db.upsert_pcr_history(pcr_records)
                            except Exception:
                                pass
                            # Collect OI history for ATM ± 2 strikes
                            oi_entry = {'time': current_time}
                            chgoi_entry = {'time': current_time}
                            vol_entry = {'time': current_time}
                            for _, row in pcr_df.iterrows():
                                strike_label = str(int(row['Strike']))
                                oi_entry[f'{strike_label}_CE'] = int(row.get('openInterest_CE', 0) or 0)
                                oi_entry[f'{strike_label}_PE'] = int(row.get('openInterest_PE', 0) or 0)
                                oi_entry[f'{strike_label}_CE_LTP'] = float(row.get('lastPrice_CE', 0) or 0)
                                oi_entry[f'{strike_label}_PE_LTP'] = float(row.get('lastPrice_PE', 0) or 0)
                                chgoi_entry[f'{strike_label}_CE'] = int(row.get('changeinOpenInterest_CE', 0) or 0)
                                chgoi_entry[f'{strike_label}_PE'] = int(row.get('changeinOpenInterest_PE', 0) or 0)
                                vol_entry[f'{strike_label}_CE'] = int(row.get('totalTradedVolume_CE', 0) or 0)
                                vol_entry[f'{strike_label}_PE'] = int(row.get('totalTradedVolume_PE', 0) or 0)
                            st.session_state.oi_history.append(oi_entry)
                            st.session_state.oi_last_valid_data = pcr_df.copy()
                            st.session_state.oi_current_strikes = [int(s) for s in current_strikes]
                            if len(st.session_state.oi_history) > 200:
                                st.session_state.oi_history = st.session_state.oi_history[-200:]
                            # Track ATM IV for session IV rank
                            try:
                                _und_p = option_data.get('underlying', 0) if option_data else 0
                                if _und_p and df_summary is not None and not df_summary.empty:
                                    _atm_r = df_summary.iloc[(df_summary['Strike'] - _und_p).abs().argsort()].iloc[0]
                                    _ce_iv = float(_atm_r.get('impliedVolatility_CE', 0) or 0)
                                    _pe_iv = float(_atm_r.get('impliedVolatility_PE', 0) or 0)
                                    _atm_iv_val = round((_ce_iv + _pe_iv) / 2, 1) if _ce_iv and _pe_iv else round(_ce_iv or _pe_iv, 1)
                                    if _atm_iv_val > 0:
                                        _iv_hist = st.session_state.get('_iv_history', [])
                                        _iv_hist.append(_atm_iv_val)
                                        st.session_state['_iv_history'] = _iv_hist[-120:]
                            except Exception:
                                pass
                            st.session_state.chgoi_history.append(chgoi_entry)
                            st.session_state.chgoi_last_valid_data = pcr_df.copy()
                            st.session_state.chgoi_current_strikes = [int(s) for s in current_strikes]
                            if len(st.session_state.chgoi_history) > 200:
                                st.session_state.chgoi_history = st.session_state.chgoi_history[-200:]
                            st.session_state.vol_history.append(vol_entry)
                            st.session_state.vol_last_valid_data = pcr_df.copy()
                            st.session_state.vol_current_strikes = [int(s) for s in current_strikes]
                            if len(st.session_state.vol_history) > 200:
                                st.session_state.vol_history = st.session_state.vol_history[-200:]
            except Exception as e:
                st.caption(f"⚠️ Current fetch issue: {str(e)[:50]}...")
        if len(st.session_state.pcr_history) > 0:
            try:
                history_df = pd.DataFrame(st.session_state.pcr_history)
                current_strikes = getattr(st.session_state, 'pcr_current_strikes', [])
                if not current_strikes and st.session_state.pcr_last_valid_data is not None:
                    current_strikes = [int(s) for s in st.session_state.pcr_last_valid_data['Strike'].tolist()]
                current_strikes = sorted(current_strikes)
                pcr_col1, pcr_col2, pcr_col3, pcr_col4, pcr_col5 = st.columns(5)
                def display_pcr_with_signal(container, fig, pcr_val):
                    if fig:
                        container.plotly_chart(fig, use_container_width=True)
                        if pcr_val > 1.2:
                            container.success("Bullish")
                        elif pcr_val < 0.7:
                            container.error("Bearish")
                        else:
                            container.warning("Neutral")
                zone_info = {}
                zone_df = pcr_df if pcr_df is not None else st.session_state.pcr_last_valid_data
                if zone_df is not None:
                    for _, row in zone_df.iterrows():
                        zone_info[int(row['Strike'])] = row['Zone']
                position_labels = ['🟣 ITM-2', '🟣 ITM-1', '🟡 ATM', '🔵 OTM+1', '🔵 OTM+2']
                position_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']
                columns = [pcr_col1, pcr_col2, pcr_col3, pcr_col4, pcr_col5]
                for i, col in enumerate(columns):
                    with col:
                        if i < len(current_strikes):
                            strike = current_strikes[i]
                            strike_col = str(strike)
                            zone = zone_info.get(strike, position_labels[i].split()[-1])
                            if strike_col in history_df.columns:
                                fig, pcr_val = create_pcr_chart(history_df, strike_col, position_colors[i], f'{position_labels[i]}')
                                display_pcr_with_signal(st, fig, pcr_val)
                            else:
                                st.info(f"₹{strike} - Building history...")
                        else:
                            st.info(f"{position_labels[i]} N/A")
                st.markdown("### Current PCR Values")
                display_df = pcr_df if pcr_df is not None else st.session_state.pcr_last_valid_data
                if display_df is not None:
                    pcr_display = display_df[['Strike', 'Zone', 'PCR', 'PCR_Signal']].copy()
                    pcr_display['CE OI (L)'] = (display_df['openInterest_CE'] / 100000).round(2)
                    pcr_display['PE OI (L)'] = (display_df['openInterest_PE'] / 100000).round(2)
                    st.dataframe(pcr_display, use_container_width=True, hide_index=True)
                col_info1, col_info2 = st.columns([3, 1])
                with col_info1:
                    status = "🟢 Live" if pcr_data_available else "🟡 Using cached history"
                    st.caption(f"{status} | 📈 {len(st.session_state.pcr_history)} data points | History preserved on refresh failures")
                with col_info2:
                    if st.button("🗑️ Clear History"):
                        st.session_state.pcr_history = []
                        st.session_state.pcr_last_valid_data = None
                        st.rerun()
            except Exception as e:
                st.warning(f"Error displaying PCR charts: {str(e)}")
        else:
            st.info("📊 PCR history will build up as the app refreshes. Please wait for data collection...")
        st.markdown("---")
        st.markdown("## 📊 OI Analysis - Time Series (ATM ± 2)")
        if len(st.session_state.oi_history) > 0:
            try:
                oi_history_df = pd.DataFrame(st.session_state.oi_history)
                oi_strikes = st.session_state.oi_current_strikes or []
                if not oi_strikes and st.session_state.oi_last_valid_data is not None:
                    oi_strikes = [int(s) for s in st.session_state.oi_last_valid_data['Strike'].tolist()]
                oi_strikes = sorted(oi_strikes)
                zone_df_oi = pcr_df if pcr_df is not None else st.session_state.oi_last_valid_data
                oi_zone_info = {}
                if zone_df_oi is not None:
                    for _, row in zone_df_oi.iterrows():
                        oi_zone_info[int(row['Strike'])] = row['Zone']
                oi_position_labels = ['ITM-2', 'ITM-1', 'ATM', 'OTM+1', 'OTM+2']
                oi_colors_ce = ['#ff6666', '#ff9999', '#ffcc00', '#66bbff', '#3399ff']
                oi_colors_pe = ['#cc44cc', '#aa66aa', '#ff8800', '#44dd88', '#22bb66']
                if len(oi_strikes) >= 3:
                    oi_col1, oi_col2 = st.columns(2)
                    # Combined OI chart - all strikes CE vs PE
                    with oi_col1:
                        fig_ce = go.Figure()
                        for i, strike in enumerate(oi_strikes):
                            ce_col = f'{strike}_CE'
                            if ce_col in oi_history_df.columns:
                                label = oi_position_labels[i] if i < len(oi_position_labels) else f'Strike {i}'
                                fig_ce.add_trace(go.Scatter(
                                    x=oi_history_df['time'],
                                    y=oi_history_df[ce_col] / 100000,
                                    mode='lines+markers',
                                    name=f'₹{strike} ({label})',
                                    line=dict(width=2),
                                    marker=dict(size=3),
                                ))
                        fig_ce.update_layout(
                            title='Call OI (ATM ± 2)',
                            template='plotly_dark',
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=30),
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='OI (Lakhs)'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_ce, use_container_width=True)
                    with oi_col2:
                        fig_pe = go.Figure()
                        for i, strike in enumerate(oi_strikes):
                            pe_col = f'{strike}_PE'
                            if pe_col in oi_history_df.columns:
                                label = oi_position_labels[i] if i < len(oi_position_labels) else f'Strike {i}'
                                fig_pe.add_trace(go.Scatter(
                                    x=oi_history_df['time'],
                                    y=oi_history_df[pe_col] / 100000,
                                    mode='lines+markers',
                                    name=f'₹{strike} ({label})',
                                    line=dict(width=2),
                                    marker=dict(size=3),
                                ))
                        fig_pe.update_layout(
                            title='Put OI (ATM ± 2)',
                            template='plotly_dark',
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=30),
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='OI (Lakhs)'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_pe, use_container_width=True)
                    # Per-strike CE vs PE comparison charts
                    st.markdown("### Per-Strike Call vs Put OI")
                    oi_strike_cols = st.columns(min(len(oi_strikes), 5))
                    for i, strike in enumerate(oi_strikes):
                        if i >= len(oi_strike_cols):
                            break
                        ce_col = f'{strike}_CE'
                        pe_col = f'{strike}_PE'
                        with oi_strike_cols[i]:
                            label = oi_position_labels[i] if i < len(oi_position_labels) else f'Strike {i}'
                            fig_strike = go.Figure()
                            if ce_col in oi_history_df.columns:
                                fig_strike.add_trace(go.Scatter(
                                    x=oi_history_df['time'],
                                    y=oi_history_df[ce_col] / 100000,
                                    mode='lines+markers',
                                    name='Call OI',
                                    line=dict(color='#ff4444', width=2),
                                    marker=dict(size=3),
                                ))
                            if pe_col in oi_history_df.columns:
                                fig_strike.add_trace(go.Scatter(
                                    x=oi_history_df['time'],
                                    y=oi_history_df[pe_col] / 100000,
                                    mode='lines+markers',
                                    name='Put OI',
                                    line=dict(color='#00cc66', width=2),
                                    marker=dict(size=3),
                                ))
                            current_ce = oi_history_df[ce_col].iloc[-1] / 100000 if ce_col in oi_history_df.columns and len(oi_history_df) > 0 else 0
                            current_pe = oi_history_df[pe_col].iloc[-1] / 100000 if pe_col in oi_history_df.columns and len(oi_history_df) > 0 else 0
                            fig_strike.update_layout(
                                title=f'{label}<br>₹{strike}<br>CE: {current_ce:.1f}L | PE: {current_pe:.1f}L',
                                template='plotly_dark',
                                height=280,
                                showlegend=True,
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=9)),
                                margin=dict(l=10, r=10, t=80, b=30),
                                xaxis=dict(tickformat='%H:%M', title=''),
                                yaxis=dict(title='OI (L)'),
                                plot_bgcolor='#1e1e1e',
                                paper_bgcolor='#1e1e1e'
                            )
                            st.plotly_chart(fig_strike, use_container_width=True)
                            # Analyze OI trend over time for support/resistance strength
                            ce_series = oi_history_df[ce_col] if ce_col in oi_history_df.columns else pd.Series([0])
                            pe_series = oi_history_df[pe_col] if pe_col in oi_history_df.columns else pd.Series([0])
                            if len(ce_series) >= 3:
                                ce_first, ce_last = ce_series.iloc[0], ce_series.iloc[-1]
                                pe_first, pe_last = pe_series.iloc[0], pe_series.iloc[-1]
                                ce_change = ce_last - ce_first
                                pe_change = pe_last - pe_first
                                ce_pct = (ce_change / ce_first * 100) if ce_first > 0 else 0
                                pe_pct = (pe_change / pe_first * 100) if pe_first > 0 else 0
                                # Support signal (PE OI trend)
                                if pe_change > 0:
                                    st.success(f"Support Building (PE +{pe_pct:.1f}%)")
                                elif pe_change < 0:
                                    st.error(f"Support Weakening (PE {pe_pct:.1f}%)")
                                else:
                                    st.info("Support Flat")
                                # Resistance signal (CE OI trend)
                                if ce_change > 0:
                                    st.error(f"Resistance Building (CE +{ce_pct:.1f}%)")
                                elif ce_change < 0:
                                    st.success(f"Resistance Weakening (CE {ce_pct:.1f}%)")
                                else:
                                    st.info("Resistance Flat")
                                # Overall verdict: compare CE vs PE OI difference
                                oi_diff = (current_pe - current_ce) * 100000  # back to absolute
                                oi_diff_lakhs = abs(current_pe - current_ce)
                                pe_ce_ratio = current_pe / current_ce if current_ce > 0 else 0
                                if current_pe > current_ce:
                                    if pe_ce_ratio >= 2.0:
                                        st.markdown(f'<div style="background:#00ff8840;padding:8px;border-radius:8px;border-left:4px solid #00ff88;"><b style="color:#00ff88;">STRONG SUPPORT</b> | PE {current_pe:.1f}L vs CE {current_ce:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {pe_ce_ratio:.1f}x</div>', unsafe_allow_html=True)
                                    elif pe_ce_ratio >= 1.3:
                                        st.markdown(f'<div style="background:#00cc6640;padding:8px;border-radius:8px;border-left:4px solid #00cc66;"><b style="color:#00cc66;">MODERATE SUPPORT</b> | PE {current_pe:.1f}L vs CE {current_ce:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {pe_ce_ratio:.1f}x</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div style="background:#88aa4440;padding:8px;border-radius:8px;border-left:4px solid #88aa44;"><b style="color:#88aa44;">WEAK SUPPORT</b> | PE {current_pe:.1f}L vs CE {current_ce:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {pe_ce_ratio:.1f}x</div>', unsafe_allow_html=True)
                                elif current_ce > current_pe:
                                    ce_pe_ratio = current_ce / current_pe if current_pe > 0 else 0
                                    if ce_pe_ratio >= 2.0:
                                        st.markdown(f'<div style="background:#ff444440;padding:8px;border-radius:8px;border-left:4px solid #ff4444;"><b style="color:#ff4444;">STRONG RESISTANCE</b> | CE {current_ce:.1f}L vs PE {current_pe:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {ce_pe_ratio:.1f}x</div>', unsafe_allow_html=True)
                                    elif ce_pe_ratio >= 1.3:
                                        st.markdown(f'<div style="background:#cc444440;padding:8px;border-radius:8px;border-left:4px solid #cc4444;"><b style="color:#cc4444;">MODERATE RESISTANCE</b> | CE {current_ce:.1f}L vs PE {current_pe:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {ce_pe_ratio:.1f}x</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div style="background:#aa664440;padding:8px;border-radius:8px;border-left:4px solid #aa6644;"><b style="color:#aa6644;">WEAK RESISTANCE</b> | CE {current_ce:.1f}L vs PE {current_pe:.1f}L | Diff: {oi_diff_lakhs:.1f}L | Ratio: {ce_pe_ratio:.1f}x</div>', unsafe_allow_html=True)
                                else:
                                    st.warning("Balanced (CE = PE)")
                                # Long/Short Building/Covering analysis based on OI + LTP trend
                                ce_ltp_col = f'{strike}_CE_LTP'
                                pe_ltp_col = f'{strike}_PE_LTP'
                                ce_has_ltp = ce_ltp_col in oi_history_df.columns and len(oi_history_df) >= 3
                                pe_has_ltp = pe_ltp_col in oi_history_df.columns and len(oi_history_df) >= 3
                                if ce_has_ltp:
                                    ce_ltp_first = oi_history_df[ce_ltp_col].iloc[0]
                                    ce_ltp_last = oi_history_df[ce_ltp_col].iloc[-1]
                                    ce_ltp_change = ce_ltp_last - ce_ltp_first
                                    ce_ltp_pct = (ce_ltp_change / ce_ltp_first * 100) if ce_ltp_first > 0 else 0
                                    # CE: OI↑+Price↑=Long Building, OI↑+Price↓=Short Building, OI↓+Price↑=Short Covering, OI↓+Price↓=Long Unwinding
                                    if ce_change > 0 and ce_ltp_change > 0:
                                        st.markdown(f'<div style="background:#ff880040;padding:6px;border-radius:6px;border-left:3px solid #ff8800;font-size:13px;"><b style="color:#ff8800;">CE: LONG BUILDING</b> | OI +{ce_pct:.1f}% | LTP +{ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif ce_change > 0 and ce_ltp_change <= 0:
                                        st.markdown(f'<div style="background:#ff444440;padding:6px;border-radius:6px;border-left:3px solid #ff4444;font-size:13px;"><b style="color:#ff4444;">CE: SHORT BUILDING</b> | OI +{ce_pct:.1f}% | LTP {ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif ce_change < 0 and ce_ltp_change > 0:
                                        st.markdown(f'<div style="background:#00ff8840;padding:6px;border-radius:6px;border-left:3px solid #00ff88;font-size:13px;"><b style="color:#00ff88;">CE: SHORT COVERING</b> | OI {ce_pct:.1f}% | LTP +{ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif ce_change < 0 and ce_ltp_change <= 0:
                                        st.markdown(f'<div style="background:#88888840;padding:6px;border-radius:6px;border-left:3px solid #888888;font-size:13px;"><b style="color:#888888;">CE: LONG UNWINDING</b> | OI {ce_pct:.1f}% | LTP {ce_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                if pe_has_ltp:
                                    pe_ltp_first = oi_history_df[pe_ltp_col].iloc[0]
                                    pe_ltp_last = oi_history_df[pe_ltp_col].iloc[-1]
                                    pe_ltp_change = pe_ltp_last - pe_ltp_first
                                    pe_ltp_pct = (pe_ltp_change / pe_ltp_first * 100) if pe_ltp_first > 0 else 0
                                    # PE: OI↑+Price↑=Long Building, OI↑+Price↓=Short Building, OI↓+Price↓=Short Covering, OI↓+Price↑=Long Unwinding
                                    if pe_change > 0 and pe_ltp_change > 0:
                                        st.markdown(f'<div style="background:#ff444440;padding:6px;border-radius:6px;border-left:3px solid #ff4444;font-size:13px;"><b style="color:#ff4444;">PE: LONG BUILDING</b> | OI +{pe_pct:.1f}% | LTP +{pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif pe_change > 0 and pe_ltp_change <= 0:
                                        st.markdown(f'<div style="background:#00ff8840;padding:6px;border-radius:6px;border-left:3px solid #00ff88;font-size:13px;"><b style="color:#00ff88;">PE: SHORT BUILDING</b> | OI +{pe_pct:.1f}% | LTP {pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif pe_change < 0 and pe_ltp_change < 0:
                                        st.markdown(f'<div style="background:#00cc6640;padding:6px;border-radius:6px;border-left:3px solid #00cc66;font-size:13px;"><b style="color:#00cc66;">PE: SHORT COVERING</b> | OI {pe_pct:.1f}% | LTP {pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                                    elif pe_change < 0 and pe_ltp_change >= 0:
                                        st.markdown(f'<div style="background:#88888840;padding:6px;border-radius:6px;border-left:3px solid #888888;font-size:13px;"><b style="color:#888888;">PE: LONG UNWINDING</b> | OI {pe_pct:.1f}% | LTP +{pe_ltp_pct:.1f}%</div>', unsafe_allow_html=True)
                            else:
                                oi_diff_lakhs = abs(current_pe - current_ce)
                                if current_pe > current_ce:
                                    st.success(f"Support > Resistance (Diff: {oi_diff_lakhs:.1f}L)")
                                elif current_ce > current_pe:
                                    st.error(f"Resistance > Support (Diff: {oi_diff_lakhs:.1f}L)")
                                else:
                                    st.warning("Neutral")
                else:
                    st.info("Waiting for ATM ± 2 strike data...")
                oi_info1, oi_info2 = st.columns([3, 1])
                with oi_info1:
                    st.caption(f"📈 {len(st.session_state.oi_history)} data points | OI values in Lakhs")
                with oi_info2:
                    if st.button("🗑️ Clear OI History"):
                        st.session_state.oi_history = []
                        st.session_state.oi_last_valid_data = None
                        st.rerun()
            except Exception as e:
                st.warning(f"Error displaying OI charts: {str(e)}")
        else:
            st.info("📊 OI history will build up as the app refreshes. Please wait for data collection...")
        st.markdown("---")
        st.markdown("## 📊 Change in OI Analysis - Time Series (ATM ± 2)")
        if len(st.session_state.chgoi_history) > 0:
            try:
                chgoi_history_df = pd.DataFrame(st.session_state.chgoi_history)
                chgoi_strikes = st.session_state.chgoi_current_strikes or []
                if not chgoi_strikes and st.session_state.chgoi_last_valid_data is not None:
                    chgoi_strikes = [int(s) for s in st.session_state.chgoi_last_valid_data['Strike'].tolist()]
                chgoi_strikes = sorted(chgoi_strikes)
                zone_df_chgoi = pcr_df if pcr_df is not None else st.session_state.chgoi_last_valid_data
                chgoi_zone_info = {}
                if zone_df_chgoi is not None:
                    for _, row in zone_df_chgoi.iterrows():
                        chgoi_zone_info[int(row['Strike'])] = row['Zone']
                chgoi_position_labels = ['ITM-2', 'ITM-1', 'ATM', 'OTM+1', 'OTM+2']
                if len(chgoi_strikes) >= 3:
                    chgoi_col1, chgoi_col2 = st.columns(2)
                    with chgoi_col1:
                        fig_chg_ce = go.Figure()
                        for i, strike in enumerate(chgoi_strikes):
                            ce_col = f'{strike}_CE'
                            if ce_col in chgoi_history_df.columns:
                                label = chgoi_position_labels[i] if i < len(chgoi_position_labels) else f'Strike {i}'
                                fig_chg_ce.add_trace(go.Scatter(
                                    x=chgoi_history_df['time'],
                                    y=chgoi_history_df[ce_col] / 1000,
                                    mode='lines+markers',
                                    name=f'₹{strike} ({label})',
                                    line=dict(width=2),
                                    marker=dict(size=3),
                                ))
                        fig_chg_ce.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
                        fig_chg_ce.update_layout(
                            title='Change in Call OI (ATM ± 2)',
                            template='plotly_dark',
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=30),
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='Chg OI (K)'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_chg_ce, use_container_width=True)
                    with chgoi_col2:
                        fig_chg_pe = go.Figure()
                        for i, strike in enumerate(chgoi_strikes):
                            pe_col = f'{strike}_PE'
                            if pe_col in chgoi_history_df.columns:
                                label = chgoi_position_labels[i] if i < len(chgoi_position_labels) else f'Strike {i}'
                                fig_chg_pe.add_trace(go.Scatter(
                                    x=chgoi_history_df['time'],
                                    y=chgoi_history_df[pe_col] / 1000,
                                    mode='lines+markers',
                                    name=f'₹{strike} ({label})',
                                    line=dict(width=2),
                                    marker=dict(size=3),
                                ))
                        fig_chg_pe.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
                        fig_chg_pe.update_layout(
                            title='Change in Put OI (ATM ± 2)',
                            template='plotly_dark',
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=30),
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='Chg OI (K)'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_chg_pe, use_container_width=True)
                    # Per-strike Change OI comparison charts
                    st.markdown("### Per-Strike Change in Call vs Put OI")
                    chgoi_strike_cols = st.columns(min(len(chgoi_strikes), 5))
                    for i, strike in enumerate(chgoi_strikes):
                        if i >= len(chgoi_strike_cols):
                            break
                        ce_col = f'{strike}_CE'
                        pe_col = f'{strike}_PE'
                        with chgoi_strike_cols[i]:
                            label = chgoi_position_labels[i] if i < len(chgoi_position_labels) else f'Strike {i}'
                            fig_chg_strike = go.Figure()
                            if ce_col in chgoi_history_df.columns:
                                fig_chg_strike.add_trace(go.Scatter(
                                    x=chgoi_history_df['time'],
                                    y=chgoi_history_df[ce_col] / 1000,
                                    mode='lines+markers',
                                    name='Call ChgOI',
                                    line=dict(color='#ff4444', width=2),
                                    marker=dict(size=3),
                                ))
                            if pe_col in chgoi_history_df.columns:
                                fig_chg_strike.add_trace(go.Scatter(
                                    x=chgoi_history_df['time'],
                                    y=chgoi_history_df[pe_col] / 1000,
                                    mode='lines+markers',
                                    name='Put ChgOI',
                                    line=dict(color='#00cc66', width=2),
                                    marker=dict(size=3),
                                ))
                            fig_chg_strike.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.5)
                            current_chg_ce = chgoi_history_df[ce_col].iloc[-1] / 1000 if ce_col in chgoi_history_df.columns and len(chgoi_history_df) > 0 else 0
                            current_chg_pe = chgoi_history_df[pe_col].iloc[-1] / 1000 if pe_col in chgoi_history_df.columns and len(chgoi_history_df) > 0 else 0
                            fig_chg_strike.update_layout(
                                title=f'{label}<br>₹{strike}<br>CE: {current_chg_ce:+.1f}K | PE: {current_chg_pe:+.1f}K',
                                template='plotly_dark',
                                height=280,
                                showlegend=True,
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=9)),
                                margin=dict(l=10, r=10, t=80, b=30),
                                xaxis=dict(tickformat='%H:%M', title=''),
                                yaxis=dict(title='Chg OI (K)'),
                                plot_bgcolor='#1e1e1e',
                                paper_bgcolor='#1e1e1e'
                            )
                            st.plotly_chart(fig_chg_strike, use_container_width=True)
                            if current_chg_ce > 0 and current_chg_pe <= 0:
                                st.error("CE Buildup (Bearish)")
                            elif current_chg_pe > 0 and current_chg_ce <= 0:
                                st.success("PE Buildup (Bullish)")
                            elif current_chg_ce > 0 and current_chg_pe > 0:
                                st.warning("Both Building")
                            elif current_chg_ce < 0 and current_chg_pe < 0:
                                st.warning("Both Unwinding")
                            else:
                                st.info("Neutral")
                else:
                    st.info("Waiting for ATM ± 2 strike data...")
                chgoi_info1, chgoi_info2 = st.columns([3, 1])
                with chgoi_info1:
                    st.caption(f"📈 {len(st.session_state.chgoi_history)} data points | Change in OI values in Thousands")
                with chgoi_info2:
                    if st.button("🗑️ Clear ChgOI History"):
                        st.session_state.chgoi_history = []
                        st.session_state.chgoi_last_valid_data = None
                        st.rerun()
            except Exception as e:
                st.warning(f"Error displaying Change OI charts: {str(e)}")
        else:
            st.info("📊 Change in OI history will build up as the app refreshes. Please wait for data collection...")
        st.markdown("---")
        st.markdown("## 📊 Volume Analysis - Time Series (ATM ± 2)")
        if len(st.session_state.vol_history) > 0:
            try:
                vol_history_df = pd.DataFrame(st.session_state.vol_history)
                vol_strikes = st.session_state.vol_current_strikes or []
                if not vol_strikes and st.session_state.vol_last_valid_data is not None:
                    vol_strikes = [int(s) for s in st.session_state.vol_last_valid_data['Strike'].tolist()]
                vol_strikes = sorted(vol_strikes)
                vol_position_labels = ['ITM-2', 'ITM-1', 'ATM', 'OTM+1', 'OTM+2']
                if len(vol_strikes) >= 3:
                    vol_col1, vol_col2 = st.columns(2)
                    with vol_col1:
                        fig_vol_ce = go.Figure()
                        for i, strike in enumerate(vol_strikes):
                            ce_col = f'{strike}_CE'
                            if ce_col in vol_history_df.columns:
                                label = vol_position_labels[i] if i < len(vol_position_labels) else f'Strike {i}'
                                fig_vol_ce.add_trace(go.Scatter(
                                    x=vol_history_df['time'],
                                    y=vol_history_df[ce_col] / 1000,
                                    mode='lines+markers',
                                    name=f'₹{strike} ({label})',
                                    line=dict(width=2),
                                    marker=dict(size=3),
                                ))
                        fig_vol_ce.update_layout(
                            title='Call Volume (ATM ± 2)',
                            template='plotly_dark',
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=30),
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='Volume (K)'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_vol_ce, use_container_width=True)
                    with vol_col2:
                        fig_vol_pe = go.Figure()
                        for i, strike in enumerate(vol_strikes):
                            pe_col = f'{strike}_PE'
                            if pe_col in vol_history_df.columns:
                                label = vol_position_labels[i] if i < len(vol_position_labels) else f'Strike {i}'
                                fig_vol_pe.add_trace(go.Scatter(
                                    x=vol_history_df['time'],
                                    y=vol_history_df[pe_col] / 1000,
                                    mode='lines+markers',
                                    name=f'₹{strike} ({label})',
                                    line=dict(width=2),
                                    marker=dict(size=3),
                                ))
                        fig_vol_pe.update_layout(
                            title='Put Volume (ATM ± 2)',
                            template='plotly_dark',
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=30),
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='Volume (K)'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_vol_pe, use_container_width=True)
                    # Per-strike Vol CE vs Vol PE comparison charts
                    st.markdown("### Per-Strike Call vs Put Volume")
                    vol_strike_cols = st.columns(min(len(vol_strikes), 5))
                    for i, strike in enumerate(vol_strikes):
                        if i >= len(vol_strike_cols):
                            break
                        ce_col = f'{strike}_CE'
                        pe_col = f'{strike}_PE'
                        with vol_strike_cols[i]:
                            label = vol_position_labels[i] if i < len(vol_position_labels) else f'Strike {i}'
                            fig_vol_strike = go.Figure()
                            if ce_col in vol_history_df.columns:
                                fig_vol_strike.add_trace(go.Scatter(
                                    x=vol_history_df['time'],
                                    y=vol_history_df[ce_col] / 1000,
                                    mode='lines+markers',
                                    name='Call Vol',
                                    line=dict(color='#ff4444', width=2),
                                    marker=dict(size=3),
                                ))
                            if pe_col in vol_history_df.columns:
                                fig_vol_strike.add_trace(go.Scatter(
                                    x=vol_history_df['time'],
                                    y=vol_history_df[pe_col] / 1000,
                                    mode='lines+markers',
                                    name='Put Vol',
                                    line=dict(color='#00cc66', width=2),
                                    marker=dict(size=3),
                                ))
                            current_ce_vol = vol_history_df[ce_col].iloc[-1] / 1000 if ce_col in vol_history_df.columns and len(vol_history_df) > 0 else 0
                            current_pe_vol = vol_history_df[pe_col].iloc[-1] / 1000 if pe_col in vol_history_df.columns and len(vol_history_df) > 0 else 0
                            # Volume trend signal
                            ce_increasing = False
                            pe_increasing = False
                            if len(vol_history_df) >= 2:
                                prev_ce = vol_history_df[ce_col].iloc[-2] / 1000 if ce_col in vol_history_df.columns else 0
                                prev_pe = vol_history_df[pe_col].iloc[-2] / 1000 if pe_col in vol_history_df.columns else 0
                                ce_increasing = current_ce_vol > prev_ce
                                pe_increasing = current_pe_vol > prev_pe
                            ce_trend = "↑" if ce_increasing else "↓"
                            pe_trend = "↑" if pe_increasing else "↓"
                            fig_vol_strike.update_layout(
                                title=f'{label}<br>₹{strike}<br>CE: {current_ce_vol:.1f}K{ce_trend} | PE: {current_pe_vol:.1f}K{pe_trend}',
                                template='plotly_dark',
                                height=280,
                                showlegend=True,
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=9)),
                                margin=dict(l=10, r=10, t=80, b=30),
                                xaxis=dict(tickformat='%H:%M', title=''),
                                yaxis=dict(title='Volume (K)'),
                                plot_bgcolor='#1e1e1e',
                                paper_bgcolor='#1e1e1e'
                            )
                            st.plotly_chart(fig_vol_strike, use_container_width=True)
                            # Signal interpretation
                            if current_ce_vol > current_pe_vol * 1.2 and ce_increasing:
                                st.error("Resistance Active 🔴")
                            elif current_pe_vol > current_ce_vol * 1.2 and pe_increasing:
                                st.success("Support Active 🟢")
                            elif ce_increasing and not pe_increasing:
                                st.warning("CE Vol Rising")
                            elif pe_increasing and not ce_increasing:
                                st.warning("PE Vol Rising")
                            else:
                                st.info("Balanced")
                else:
                    st.info("Waiting for ATM ± 2 strike data...")
                vol_info1, vol_info2 = st.columns([3, 1])
                with vol_info1:
                    st.caption(f"📈 {len(st.session_state.vol_history)} data points | Volume values in Thousands (K)")
                with vol_info2:
                    if st.button("🗑️ Clear Vol History"):
                        st.session_state.vol_history = []
                        st.session_state.vol_last_valid_data = None
                        st.rerun()
            except Exception as e:
                st.warning(f"Error displaying Volume charts: {str(e)}")
        else:
            st.info("📊 Volume history will build up as the app refreshes. Please wait for data collection...")
        st.markdown("---")
        st.markdown("## 📊 Gamma Exposure (GEX) Analysis - Dealer Hedging Flow")
        try:
            df_summary = option_data.get('df_summary')
            underlying_price = option_data.get('underlying')
            if df_summary is not None and underlying_price:
                gex_data = calculate_dealer_gex(df_summary, underlying_price)
                if gex_data:
                    st.session_state._gex_data = gex_data
                    gex_df = gex_data['gex_df']
                    st.session_state.gex_last_valid_data = gex_data
                    check_gex_alert(gex_data, df_summary, underlying_price)
                    def _gex_card(c, title, value, sub, color=None):
                        if value is not None and color:
                            c.markdown(f'<div style="background-color: {color}20; padding: 15px; border-radius: 10px; border: 2px solid {color};"><h4 style="color: {color}; margin: 0;">{title}</h4><h2 style="color: {color}; margin: 5px 0;">{value}</h2><p style="color: white; margin: 0; font-size: 12px;">{sub}</p></div>', unsafe_allow_html=True)
                        else:
                            c.markdown(f'<div style="background-color: #33333380; padding: 15px; border-radius: 10px; border: 2px solid #666;"><h4 style="color: #999; margin: 0;">{title}</h4><h2 style="color: #999; margin: 5px 0;">N/A</h2><p style="color: #666; margin: 0; font-size: 12px;">{sub}</p></div>', unsafe_allow_html=True)
                    gc1, gc2, gc3, gc4 = st.columns(4)
                    _gex_card(gc1, "Net GEX", f"{gex_data['total_gex']:+.2f}L", gex_data['gex_signal'], gex_data['gex_color'])
                    if gex_data['gamma_flip_level']:
                        fc = "#00ff88" if underlying_price > gex_data['gamma_flip_level'] else "#ff4444"
                        _gex_card(gc2, "Gamma Flip", f"₹{gex_data['gamma_flip_level']:.0f}", gex_data['spot_vs_flip'], fc)
                    else:
                        _gex_card(gc2, "Gamma Flip", None, "No flip detected")
                    _gex_card(gc3, "GEX Magnet", f"₹{gex_data['gex_magnet']:.0f}" if gex_data['gex_magnet'] else None,
                              "Price attracted here" if gex_data['gex_magnet'] else "No magnet", "#00ff88" if gex_data['gex_magnet'] else None)
                    _gex_card(gc4, "GEX Repeller", f"₹{gex_data['gex_repeller']:.0f}" if gex_data['gex_repeller'] else None,
                              "Price accelerates here" if gex_data['gex_repeller'] else "No repeller", "#ff4444" if gex_data['gex_repeller'] else None)
                    st.markdown(f"""
                    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 4px solid {gex_data['gex_color']}; margin: 10px 0;">
                        <b style="color: {gex_data['gex_color']};">Market Regime:</b> {gex_data['gex_interpretation']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("### 🎯 PCR × GEX Confluence")
                    atm_data = df_summary[df_summary['Zone'] == 'ATM']
                    if not atm_data.empty:
                        atm_pcr = atm_data.iloc[0].get('PCR', 1.0)
                        confluence_badge, confluence_signal, confluence_strength = calculate_pcr_gex_confluence(atm_pcr, gex_data)
                        st.session_state._confluence = (confluence_badge, confluence_signal, confluence_strength)
                        conf_col1, conf_col2 = st.columns([1, 3])
                        with conf_col1:
                            if "BULL" in confluence_badge:
                                badge_color = "#00ff88"
                            elif "BEAR" in confluence_badge:
                                badge_color = "#ff4444"
                            else:
                                badge_color = "#FFD700"
                            st.markdown(f"""
                            <div style="background-color: {badge_color}30; padding: 20px; border-radius: 15px; border: 3px solid {badge_color}; text-align: center;">
                                <h2 style="color: {badge_color}; margin: 0; font-size: 24px;">{confluence_badge}</h2>
                                <p style="color: white; margin: 5px 0; font-size: 14px;">{confluence_signal}</p>
                                <p style="color: #888; margin: 0; font-size: 12px;">Strength: {'★' * confluence_strength}{'☆' * (3 - confluence_strength)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with conf_col2:
                            st.markdown("""
                            **Confluence Matrix:**
                            - 🟢🔥 **STRONG BULL**: Bullish PCR + Negative GEX = Violent upside potential
                            - 🔴🔥 **STRONG BEAR**: Bearish PCR + Positive GEX = Strong rejection/pin down
                            - 🟢📍 **BULL RANGE**: Bullish PCR + Positive GEX = Support with chop
                            - 🔴⚡ **BEAR TREND**: Bearish PCR + Negative GEX = Downside acceleration
                            """)
                    st.markdown("### 📊 Net GEX by Strike (Dealer Hedging Pressure)")
                    fig_gex = go.Figure()
                    colors = ['#00ff88' if x >= 0 else '#ff4444' for x in gex_df['Net_GEX']]
                    fig_gex.add_trace(go.Bar(
                        x=gex_df['Strike'],
                        y=gex_df['Net_GEX'],
                        marker_color=colors,
                        name='Net GEX',
                        text=[f"{x:.1f}L" for x in gex_df['Net_GEX']],
                        textposition='outside',
                        textfont=dict(size=10)
                    ))
                    fig_gex.add_hline(y=0, line_dash="dash", line_color="white", line_width=2)
                    if gex_data['gamma_flip_level']:
                        fig_gex.add_vline(
                            x=gex_data['gamma_flip_level'],
                            line_dash="dot",
                            line_color="#FFD700",
                            line_width=2,
                            annotation_text=f"Gamma Flip: ₹{gex_data['gamma_flip_level']:.0f}",
                            annotation_position="top"
                        )
                    fig_gex.add_vline(
                        x=underlying_price,
                        line_dash="solid",
                        line_color="#00aaff",
                        line_width=3,
                        annotation_text=f"Spot: ₹{underlying_price:.0f}",
                        annotation_position="bottom"
                    )
                    fig_gex.update_layout(
                        title=f"Net GEX by Strike | Total: {gex_data['total_gex']:+.2f}L",
                        template='plotly_dark',
                        height=400,
                        showlegend=False,
                        xaxis_title="Strike Price",
                        yaxis_title="Net GEX (Lakhs)",
                        plot_bgcolor='#1e1e1e',
                        paper_bgcolor='#1e1e1e',
                        margin=dict(l=50, r=50, t=60, b=50)
                    )
                    st.plotly_chart(fig_gex, use_container_width=True)
                    with st.expander("📋 GEX Breakdown by Strike"):
                        gex_display = gex_df.copy()
                        gex_display['Strike'] = gex_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                        def color_gex(val):
                            try:
                                v = float(val)
                                if v > 10:
                                    return 'background-color: #00ff8840; color: white'
                                elif v > 0:
                                    return 'background-color: #00ff8820; color: white'
                                elif v < -10:
                                    return 'background-color: #ff444440; color: white'
                                elif v < 0:
                                    return 'background-color: #ff444420; color: white'
                                else:
                                    return ''
                            except:
                                return ''
                        styled_gex = gex_display.style.map(color_gex, subset=['Call_GEX', 'Put_GEX', 'Net_GEX'])
                        st.dataframe(styled_gex, use_container_width=True, hide_index=True)
                        st.markdown("""
                        **GEX Interpretation:**
                        - **Positive Net GEX (Green)**: Dealers LONG gamma → Price tends to PIN/REVERT
                        - **Negative Net GEX (Red)**: Dealers SHORT gamma → Price tends to ACCELERATE
                        - **GEX Magnet**: Strike with highest positive GEX (price attracted)
                        - **GEX Repeller**: Strike with most negative GEX (price accelerates away)
                        - **Gamma Flip**: Level where dealers switch from long to short gamma
                        """)
                    st.markdown("### 📈 GEX Time Series (ATM ± 2 Strikes)")
                    ist = pytz.timezone('Asia/Kolkata')
                    current_time = datetime.now(ist)
                    gex_entry = {'time': current_time, 'total_gex': gex_data['total_gex']}
                    for _, row in gex_df.iterrows():
                        strike_label = str(int(row['Strike']))
                        gex_entry[strike_label] = row['Net_GEX']
                        gex_entry[f'{strike_label}_Call'] = row.get('Call_GEX', 0)
                        gex_entry[f'{strike_label}_Put'] = row.get('Put_GEX', 0)
                    current_gex_strikes = [int(row['Strike']) for _, row in gex_df.iterrows()]
                    st.session_state.gex_current_strikes = sorted(current_gex_strikes)
                    should_add_gex = True
                    if st.session_state.gex_history:
                        last_gex_entry = st.session_state.gex_history[-1]
                        time_diff = (current_time - last_gex_entry['time']).total_seconds()
                        if time_diff < 30:
                            should_add_gex = False
                    if should_add_gex:
                        st.session_state.gex_history.append(gex_entry)
                        if len(st.session_state.gex_history) > 200:
                            st.session_state.gex_history = st.session_state.gex_history[-200:]
                        # Persist GEX to Supabase
                        try:
                            expiry_for_gex = option_data.get('expiry', selected_expiry) if option_data else selected_expiry
                            gex_records = []
                            for _, row in gex_df.iterrows():
                                gex_records.append({
                                    'timestamp': current_time,
                                    'expiry': expiry_for_gex,
                                    'strike': float(row['Strike']),
                                    'atm_strike': float(atm_strike) if 'atm_strike' in dir() else 0,
                                    'total_gex': float(gex_data['total_gex']),
                                    'call_gex': float(row.get('Call_GEX', 0)),
                                    'put_gex': float(row.get('Put_GEX', 0)),
                                    'net_gex': float(row.get('Net_GEX', 0)),
                                    'gamma_flip': gex_data.get('gamma_flip_level'),
                                    'signal': gex_data.get('gex_signal', ''),
                                    'spot': float(underlying_price),
                                })
                            if gex_records:
                                db.upsert_gex_history(gex_records)
                        except Exception:
                            pass
                    def create_gex_chart(history_df, col_name, color, title_prefix):
                        """Helper to create individual GEX chart per strike"""
                        if col_name and col_name in history_df.columns:
                            strike_val = col_name
                            gex_values = history_df[col_name].dropna()
                            max_abs = 20
                            if len(gex_values) > 0:
                                max_abs = max(abs(gex_values.max()), abs(gex_values.min()), 15)
                            y_range = [-max_abs * 1.1, max_abs * 1.1]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=history_df['time'],
                                y=history_df[col_name],
                                mode='lines+markers',
                                name=f'₹{strike_val}',
                                line=dict(color=color, width=2),
                                marker=dict(size=4),
                                fill='tozeroy',
                                fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}'
                            ))
                            fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=2)
                            fig.add_hline(y=10, line_dash="dot", line_color="#00ff88", line_width=1,
                                         annotation_text="+10", annotation_position="right")
                            fig.add_hline(y=-10, line_dash="dot", line_color="#ff4444", line_width=1,
                                         annotation_text="-10", annotation_position="right")
                            current_gex = history_df[col_name].iloc[-1] if len(history_df) > 0 else 0
                            fig.update_layout(
                                title=f"{title_prefix}<br>₹{strike_val}<br>GEX: {current_gex:+.1f}L",
                                template='plotly_dark',
                                height=300,
                                showlegend=False,
                                margin=dict(l=10, r=10, t=70, b=30),
                                xaxis=dict(tickformat='%H:%M', title=''),
                                yaxis=dict(
                                    title='GEX (L)',
                                    range=y_range,
                                    zeroline=True,
                                    zerolinecolor='white',
                                    zerolinewidth=2,
                                    tickmode='array',
                                    tickvals=[-20, -10, 0, 10, 20] if max_abs <= 25 else None
                                ),
                                plot_bgcolor='#1e1e1e',
                                paper_bgcolor='#1e1e1e'
                            )
                            return fig, current_gex
                        return None, 0
                    try:
                        current_strikes = sorted([int(row['Strike']) for _, row in gex_df.iterrows()])
                        st.session_state.gex_current_strikes = current_strikes
                        current_gex_values = {}
                        for _, row in gex_df.iterrows():
                            strike = int(row['Strike'])
                            current_gex_values[strike] = row['Net_GEX']
                        gex_col1, gex_col2, gex_col3, gex_col4, gex_col5 = st.columns(5)
                        def display_gex_with_signal(container, fig, gex_val):
                            if fig:
                                container.plotly_chart(fig, use_container_width=True)
                                if gex_val > 10:
                                    container.success("📍 Pin Zone")
                                elif gex_val < -10:
                                    container.error("⚡ Accel Zone")
                                else:
                                    container.warning("➡️ Neutral")
                        position_labels = ['🟣 ITM-2', '🟣 ITM-1', '🟡 ATM', '🔵 OTM+1', '🔵 OTM+2']
                        position_colors = ['#ff44ff', '#cc44cc', '#ffaa00', '#00aaff', '#0088dd']
                        columns = [gex_col1, gex_col2, gex_col3, gex_col4, gex_col5]
                        has_history = len(st.session_state.gex_history) > 0
                        gex_history_df = pd.DataFrame(st.session_state.gex_history) if has_history else None
                        for i, col in enumerate(columns):
                            with col:
                                if i < len(current_strikes):
                                    strike = current_strikes[i]
                                    strike_col = str(strike)
                                    current_gex = current_gex_values.get(strike, 0)
                                    if has_history and gex_history_df is not None and strike_col in gex_history_df.columns:
                                        fig, gex_val = create_gex_chart(gex_history_df, strike_col, position_colors[i], f'{position_labels[i]}')
                                        display_gex_with_signal(st, fig, gex_val)
                                    else:
                                        fig = go.Figure()
                                        now = datetime.now(pytz.timezone('Asia/Kolkata'))
                                        fig.add_trace(go.Scatter(
                                            x=[now],
                                            y=[current_gex],
                                            mode='markers+text',
                                            marker=dict(size=20, color=position_colors[i], symbol='diamond'),
                                            text=[f'{current_gex:+.1f}L'],
                                            textposition='top center',
                                            textfont=dict(size=12, color='white'),
                                            name=f'₹{strike}'
                                        ))
                                        max_abs = max(abs(current_gex), 15)
                                        y_range = [-max_abs * 1.3, max_abs * 1.3]
                                        fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=2)
                                        fig.add_hline(y=10, line_dash="dot", line_color="#00ff88", line_width=1,
                                                     annotation_text="+10", annotation_position="right")
                                        fig.add_hline(y=-10, line_dash="dot", line_color="#ff4444", line_width=1,
                                                     annotation_text="-10", annotation_position="right")
                                        fig.update_layout(
                                            title=f"{position_labels[i]}<br>₹{strike}<br>GEX: {current_gex:+.1f}L",
                                            template='plotly_dark',
                                            height=300,
                                            showlegend=False,
                                            margin=dict(l=10, r=10, t=70, b=30),
                                            xaxis=dict(tickformat='%H:%M', title='', showticklabels=True),
                                            yaxis=dict(
                                                title='GEX (L)',
                                                range=y_range,
                                                zeroline=True,
                                                zerolinecolor='white',
                                                zerolinewidth=2,
                                                tickmode='array',
                                                tickvals=[-20, -10, 0, 10, 20] if max_abs <= 25 else None
                                            ),
                                            plot_bgcolor='#1e1e1e',
                                            paper_bgcolor='#1e1e1e'
                                        )
                                        display_gex_with_signal(st, fig, current_gex)
                                else:
                                    st.info(f"{position_labels[i]} N/A")
                        # Combined Call GEX vs Put GEX time series
                        st.markdown("### 📊 Call GEX vs Put GEX - Time Series (ATM ± 2)")
                        if has_history and gex_history_df is not None:
                            gex_pos_labels = ['ITM-2', 'ITM-1', 'ATM', 'OTM+1', 'OTM+2']
                            gex_ts_col1, gex_ts_col2 = st.columns(2)
                            with gex_ts_col1:
                                fig_call_gex = go.Figure()
                                for idx, strike in enumerate(current_strikes):
                                    call_col = f'{strike}_Call'
                                    if call_col in gex_history_df.columns:
                                        lbl = gex_pos_labels[idx] if idx < len(gex_pos_labels) else f'Strike {idx}'
                                        fig_call_gex.add_trace(go.Scatter(
                                            x=gex_history_df['time'],
                                            y=gex_history_df[call_col],
                                            mode='lines+markers',
                                            name=f'₹{strike} ({lbl})',
                                            line=dict(width=2),
                                            marker=dict(size=3),
                                        ))
                                fig_call_gex.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
                                fig_call_gex.update_layout(
                                    title='Call GEX by Strike (ATM ± 2)',
                                    template='plotly_dark',
                                    height=350,
                                    margin=dict(l=10, r=10, t=50, b=30),
                                    xaxis=dict(tickformat='%H:%M', title='Time'),
                                    yaxis=dict(title='Call GEX (Lakhs)'),
                                    plot_bgcolor='#1e1e1e',
                                    paper_bgcolor='#1e1e1e',
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                                )
                                st.plotly_chart(fig_call_gex, use_container_width=True)
                            with gex_ts_col2:
                                fig_put_gex = go.Figure()
                                for idx, strike in enumerate(current_strikes):
                                    put_col = f'{strike}_Put'
                                    if put_col in gex_history_df.columns:
                                        lbl = gex_pos_labels[idx] if idx < len(gex_pos_labels) else f'Strike {idx}'
                                        fig_put_gex.add_trace(go.Scatter(
                                            x=gex_history_df['time'],
                                            y=gex_history_df[put_col],
                                            mode='lines+markers',
                                            name=f'₹{strike} ({lbl})',
                                            line=dict(width=2),
                                            marker=dict(size=3),
                                        ))
                                fig_put_gex.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
                                fig_put_gex.update_layout(
                                    title='Put GEX by Strike (ATM ± 2)',
                                    template='plotly_dark',
                                    height=350,
                                    margin=dict(l=10, r=10, t=50, b=30),
                                    xaxis=dict(tickformat='%H:%M', title='Time'),
                                    yaxis=dict(title='Put GEX (Lakhs)'),
                                    plot_bgcolor='#1e1e1e',
                                    paper_bgcolor='#1e1e1e',
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                                )
                                st.plotly_chart(fig_put_gex, use_container_width=True)
                            # Per-strike Call GEX vs Put GEX comparison
                            st.markdown("### Per-Strike Call GEX vs Put GEX")
                            gex_cmp_cols = st.columns(min(len(current_strikes), 5))
                            for idx, strike in enumerate(current_strikes):
                                if idx >= len(gex_cmp_cols):
                                    break
                                call_col = f'{strike}_Call'
                                put_col = f'{strike}_Put'
                                with gex_cmp_cols[idx]:
                                    lbl = gex_pos_labels[idx] if idx < len(gex_pos_labels) else f'Strike {idx}'
                                    fig_gex_cmp = go.Figure()
                                    if call_col in gex_history_df.columns:
                                        fig_gex_cmp.add_trace(go.Scatter(
                                            x=gex_history_df['time'],
                                            y=gex_history_df[call_col],
                                            mode='lines+markers',
                                            name='Call GEX',
                                            line=dict(color='#ff4444', width=2),
                                            marker=dict(size=3),
                                        ))
                                    if put_col in gex_history_df.columns:
                                        fig_gex_cmp.add_trace(go.Scatter(
                                            x=gex_history_df['time'],
                                            y=gex_history_df[put_col],
                                            mode='lines+markers',
                                            name='Put GEX',
                                            line=dict(color='#00cc66', width=2),
                                            marker=dict(size=3),
                                        ))
                                    fig_gex_cmp.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.5)
                                    cur_call_gex = gex_history_df[call_col].iloc[-1] if call_col in gex_history_df.columns and len(gex_history_df) > 0 else 0
                                    cur_put_gex = gex_history_df[put_col].iloc[-1] if put_col in gex_history_df.columns and len(gex_history_df) > 0 else 0
                                    fig_gex_cmp.update_layout(
                                        title=f'{lbl}<br>₹{strike}<br>CE: {cur_call_gex:+.1f}L | PE: {cur_put_gex:+.1f}L',
                                        template='plotly_dark',
                                        height=280,
                                        showlegend=True,
                                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=9)),
                                        margin=dict(l=10, r=10, t=80, b=30),
                                        xaxis=dict(tickformat='%H:%M', title=''),
                                        yaxis=dict(title='GEX (L)'),
                                        plot_bgcolor='#1e1e1e',
                                        paper_bgcolor='#1e1e1e'
                                    )
                                    st.plotly_chart(fig_gex_cmp, use_container_width=True)
                                    net = cur_call_gex + cur_put_gex
                                    if net > 5:
                                        st.success("Pin Zone")
                                    elif net < -5:
                                        st.error("Accel Zone")
                                    else:
                                        st.warning("Neutral")
                        st.markdown("### Current GEX Values")
                        gex_display = gex_df[['Strike', 'Zone', 'Call_GEX', 'Put_GEX', 'Net_GEX']].copy()
                        gex_display['Strike'] = gex_display['Strike'].apply(lambda x: f"₹{x:.0f}")
                        styled_gex_table = gex_display.style.map(color_gex, subset=['Call_GEX', 'Put_GEX', 'Net_GEX'])
                        st.dataframe(styled_gex_table, use_container_width=True, hide_index=True)
                        gex_info1, gex_info2 = st.columns([3, 1])
                        with gex_info1:
                            history_status = f"📈 {len(st.session_state.gex_history)} data points" if has_history else "⏳ Building history..."
                            st.caption(f"🟢 Live | {history_status} | GEX > 10 = Pin Zone | GEX < -10 = Acceleration Zone")
                        with gex_info2:
                            if st.button("🗑️ Clear GEX History"):
                                st.session_state.gex_history = []
                                st.session_state.gex_last_valid_data = None
                                st.rerun()
                    except Exception as e:
                        st.warning(f"Error displaying GEX charts: {str(e)}")
                else:
                    st.warning("Unable to calculate GEX. Check option chain data.")
        except Exception as e:
            st.warning(f"GEX analysis unavailable: {str(e)}")
        with st.expander("📊 Detailed Greeks & Raw Values"):
            df_summary = option_data['df_summary']
            detail_cols = ['Strike', 'Zone', 'lastPrice_CE', 'lastPrice_PE',
                           'openInterest_CE', 'openInterest_PE', 'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                           'totalTradedVolume_CE', 'totalTradedVolume_PE',
                           'Delta_CE', 'Delta_PE', 'Gamma_CE', 'Gamma_PE', 'Vega_CE', 'Vega_PE', 'Theta_CE', 'Theta_PE',
                           'impliedVolatility_CE', 'impliedVolatility_PE',
                           'bidQty_CE', 'bidQty_PE', 'askQty_CE', 'askQty_PE']
            detail_cols = [col for col in detail_cols if col in df_summary.columns]
            if detail_cols:
                st.dataframe(df_summary[detail_cols].style.apply(highlight_atm_row, axis=1), use_container_width=True)
        csv_data = create_csv_download(option_data['df_summary'])
        st.download_button(
            label="📥 Download Summary as CSV",
            data=csv_data,
            file_name=f"nifty_options_summary_{option_data['expiry']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    # === MASTER TRADING SIGNAL ===
    if option_data and option_data.get('underlying') and not df.empty:
        st.markdown("---")
        _force_send_clicked = False
        hdr_col1, hdr_col2 = st.columns([3, 1])
        with hdr_col1:
            _gemini_ok = bool(GEMINI_API_KEY) and _HAS_GEMINI
            st.markdown(f"## 🎯 Master Trading Signal {'🤖✅' if _gemini_ok else '🤖❌'}")
        with hdr_col2:
            _ai_explain_clicked = st.button("🤖 AI Explain", key="ai_explain_master", help="Ask Gemini to analyze the current signal and give a trade plan")

        # Persistent Gemini analysis panel — only shown when real analysis text exists
        _last_gm = st.session_state.get('_last_gemini_master')
        _last_gm_oc = st.session_state.get('_last_gemini_oc')
        _gm_valid = _last_gm and _last_gm.get('time') and not str(_last_gm.get('text', '')).startswith('⚠️')
        _gm_oc_valid = _last_gm_oc and _last_gm_oc.get('time') and not str(_last_gm_oc.get('text', '')).startswith('⚠️')
        if _gm_valid or _gm_oc_valid:
            with st.expander("🤖 Latest Gemini Analysis (auto)", expanded=True):
                if _gm_valid:
                    st.markdown(f"**Master Signal** — {_last_gm.get('time', '')}")
                    st.markdown(_last_gm.get('text', ''))
                if _gm_oc_valid:
                    st.markdown("---")
                    st.markdown(f"**Option Chain Deep Analysis** — {_last_gm_oc.get('time', '')}")
                    st.markdown(_last_gm_oc.get('text', ''))

        try:
            _sa = getattr(st.session_state, '_sa_result', None)
            _gex = getattr(st.session_state, '_gex_data', None)
            _conf = getattr(st.session_state, '_confluence', None)
            _do_send = _force_send_clicked or st.session_state.pop('_top_send_triggered', False)
            if _do_send:
                try:
                    _master_now = generate_master_signal(df, _sa, _gex, _conf, option_data['underlying'], api)
                    if _master_now:
                        send_master_signal_telegram(_master_now, option_data['underlying'], option_data, force=True)
                    else:
                        st.warning("⚠️ Master signal could not be generated — price data may be empty.")
                    if _sa is not None:
                        send_option_chain_signal(_sa, option_data['underlying'], force=True)
                    st.success("✅ Sent Master Signal to Telegram")
                except Exception as _e:
                    st.error(f"Failed to force-send: {_e}")
            if _ai_explain_clicked:
                with st.spinner("🤖 Gemini is analyzing the current setup..."):
                    try:
                        _master_for_ai = generate_master_signal(df, _sa, _gex, _conf, option_data['underlying'], api) if _sa else None
                        if _master_for_ai is None:
                            st.warning("Master signal not ready yet — load option chain first.")
                        else:
                            _mf = st.session_state.get('_money_flow_data')
                            _uw = compute_unwinding_summary(option_data.get('df_atm8'))
                            _text, _err = ai_explain_signal(_master_for_ai, _sa, option_data['underlying'], _mf, _uw)
                            if _err:
                                st.error(_err)
                            else:
                                st.markdown("### 🤖 Gemini Analysis")
                                st.markdown(_text)
                    except Exception as _e:
                        st.error(f"AI explain failed: {_e}")
            if _sa:
                master = generate_master_signal(df, _sa, _gex, _conf, option_data['underlying'], api)
                if master:
                    st.session_state._master_signal_latest = master
                    _ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                    if False:  # auto 5-min signal disabled — alerts fire on their own
                        # Refresh PCR S/R snapshot from current OI history so the
                        # auto-send always has fresh data (the UI table sets it later
                        # in the same render, so session state may be stale on first fires)
                        try:
                            _snap_fresh = []
                            _oi_hist_s = getattr(st.session_state, 'oi_history', [])
                            _oi_str_s = getattr(st.session_state, 'oi_current_strikes', [])
                            if len(_oi_hist_s) >= 3 and _oi_str_s:
                                _sorted_ss = sorted(_oi_str_s)
                                _atm_i = len(_sorted_ss) // 2
                                for _soff, _slbl in [(-2, 'ATM-2'), (-1, 'ATM-1'), (0, 'ATM'), (1, 'ATM+1'), (2, 'ATM+2')]:
                                    _si2 = _atm_i + _soff
                                    if 0 <= _si2 < len(_sorted_ss):
                                        _sk = str(_sorted_ss[_si2])
                                        _sodf = pd.DataFrame(_oi_hist_s)
                                        _sce = _sodf[f'{_sk}_CE'].iloc[-1] if f'{_sk}_CE' in _sodf.columns else 0
                                        _spe = _sodf[f'{_sk}_PE'].iloc[-1] if f'{_sk}_PE' in _sodf.columns else 0
                                        _spcr = round(_spe / _sce, 2) if _sce > 0 else 1.0
                                        _ssr = calculate_pcr_sr_level(_spcr, int(_sk))
                                        _snap_fresh.append({'label': _slbl, 'strike': int(_sk),
                                                            'pcr': _spcr, 'type': _ssr['type'],
                                                            'level': _ssr['level'], 'offset': _ssr['offset']})
                            # Fallback: compute from df_summary (always available on first load)
                            if not _snap_fresh and option_data and option_data.get('df_summary') is not None:
                                _dfs2 = option_data['df_summary']
                                _und2 = option_data.get('underlying', 0)
                                if 'PCR' in _dfs2.columns and 'Strike' in _dfs2.columns and _und2:
                                    _slist2 = sorted(_dfs2['Strike'].unique())
                                    _ai2 = min(range(len(_slist2)), key=lambda i: abs(_slist2[i] - _und2))
                                    for _soff, _slbl in [(-2, 'ATM-2'), (-1, 'ATM-1'), (0, 'ATM'), (1, 'ATM+1'), (2, 'ATM+2')]:
                                        _si2 = _ai2 + _soff
                                        if 0 <= _si2 < len(_slist2):
                                            _row2 = _dfs2[_dfs2['Strike'] == _slist2[_si2]]
                                            if not _row2.empty:
                                                _spcr2 = float(_row2['PCR'].iloc[0])
                                                _ssr2 = calculate_pcr_sr_level(_spcr2, int(_slist2[_si2]))
                                                _snap_fresh.append({'label': _slbl, 'strike': int(_slist2[_si2]),
                                                                    'pcr': _spcr2, 'type': _ssr2['type'],
                                                                    'level': _ssr2['level'], 'offset': _ssr2['offset']})
                            if _snap_fresh:
                                st.session_state._pcr_sr_snapshot = _snap_fresh
                        except Exception:
                            pass
                        try:
                            send_master_signal_telegram(master, option_data['underlying'], option_data)
                            st.session_state.last_master_signal_time = _ist_now
                        except Exception:
                            pass
                        # Store signal in Supabase
                        try:
                            _align = master.get('alignment', {})
                            _non_vix = {k: v for k, v in _align.items() if 'VIX' not in k}
                            _bull_a = sum(1 for v in _non_vix.values() if v.get('sentiment_10m') == 'Bullish')
                            _bear_a = sum(1 for v in _non_vix.values() if v.get('sentiment_10m') == 'Bearish')
                            _align_sum = f"Bull:{_bull_a} Bear:{_bear_a}/{len(_non_vix)}"
                            db.upsert_master_signal({
                                'spot_price': option_data['underlying'],
                                'signal': master['signal'],
                                'trade_type': master['trade_type'],
                                'score': master['score'],
                                'abs_score': master['abs_score'],
                                'strength': master['strength'],
                                'confidence': master['confidence'],
                                'candle_pattern': master['candle']['pattern'],
                                'candle_direction': master['candle']['direction'],
                                'volume_label': master['volume']['label'],
                                'volume_ratio': master['volume']['ratio'],
                                'location': ', '.join(master['location']),
                                'resistance_levels': ', '.join([f"{r:.0f}" for r in master['resistance_levels'][:3]]),
                                'support_levels': ', '.join([f"{s:.0f}" for s in master['support_levels'][:3]]),
                                'net_gex': master['gex']['net_gex'],
                                'atm_gex': master['gex']['atm_gex'],
                                'gamma_flip': master['gex']['gamma_flip'],
                                'gex_mode': master['gex']['market_mode'],
                                'pcr_gex_badge': master['pcr_gex']['badge'],
                                'market_bias': master['market_bias'],
                                'vix_value': master['vix'].get('vix'),
                                'vix_direction': master['vix'].get('direction', ''),
                                'oi_trend_signal': master.get('oi_trend', {}).get('signal', ''),
                                'ce_activity': master.get('oi_trend', {}).get('ce_activity', ''),
                                'pe_activity': master.get('oi_trend', {}).get('pe_activity', ''),
                                'support_status': master.get('oi_trend', {}).get('support_status', ''),
                                'resistance_status': master.get('oi_trend', {}).get('resistance_status', ''),
                                'vidya_trend': master.get('vidya', {}).get('trend', ''),
                                'vidya_delta_pct': master.get('vidya', {}).get('delta_pct', 0),
                                'delta_vol_trend': master.get('delta_trend', ''),
                                'vwap': master.get('ltp_trap', {}).get('vwap'),
                                'price_vs_vwap': master.get('ltp_trap', {}).get('price_vs_vwap', ''),
                                'reasons': ' | '.join(master['reasons']),
                                'alignment_summary': _align_sum,
                            })
                        except Exception:
                            pass

                    # Refresh PCR S/R snapshot every cycle so proximity alerts always have fresh data
                    try:
                        _snap_cycle = []
                        _oi_hist_c = getattr(st.session_state, 'oi_history', [])
                        _oi_str_c  = getattr(st.session_state, 'oi_current_strikes', [])
                        if len(_oi_hist_c) >= 3 and _oi_str_c:
                            _sorted_c = sorted(_oi_str_c)
                            _atm_c = len(_sorted_c) // 2
                            for _soff, _slbl in [(-2,'ATM-2'),(-1,'ATM-1'),(0,'ATM'),(1,'ATM+1'),(2,'ATM+2')]:
                                _si = _atm_c + _soff
                                if 0 <= _si < len(_sorted_c):
                                    _sk = str(_sorted_c[_si])
                                    _sodf = pd.DataFrame(_oi_hist_c)
                                    _sce = _sodf[f'{_sk}_CE'].iloc[-1] if f'{_sk}_CE' in _sodf.columns else 0
                                    _spe = _sodf[f'{_sk}_PE'].iloc[-1] if f'{_sk}_PE' in _sodf.columns else 0
                                    _spcr = round(_spe / _sce, 2) if _sce > 0 else 1.0
                                    _ssr = calculate_pcr_sr_level(_spcr, int(_sk))
                                    _snap_cycle.append({'label': _slbl, 'strike': int(_sk),
                                                        'pcr': _spcr, 'type': _ssr['type'],
                                                        'level': _ssr['level'], 'offset': _ssr['offset']})
                        if not _snap_cycle and option_data and option_data.get('df_summary') is not None:
                            _dfs_c = option_data['df_summary']
                            _und_c = option_data.get('underlying', 0)
                            if 'PCR' in _dfs_c.columns and 'Strike' in _dfs_c.columns and _und_c:
                                _slist_c = sorted(_dfs_c['Strike'].unique())
                                _ai_c = min(range(len(_slist_c)), key=lambda i: abs(_slist_c[i] - _und_c))
                                for _soff, _slbl in [(-2,'ATM-2'),(-1,'ATM-1'),(0,'ATM'),(1,'ATM+1'),(2,'ATM+2')]:
                                    _si = _ai_c + _soff
                                    if 0 <= _si < len(_slist_c):
                                        _row_c = _dfs_c[_dfs_c['Strike'] == _slist_c[_si]]
                                        if not _row_c.empty:
                                            _spcr_c = float(_row_c['PCR'].iloc[0])
                                            _ssr_c = calculate_pcr_sr_level(_spcr_c, int(_slist_c[_si]))
                                            _snap_cycle.append({'label': _slbl, 'strike': int(_slist_c[_si]),
                                                                'pcr': _spcr_c, 'type': _ssr_c['type'],
                                                                'level': _ssr_c['level'], 'offset': _ssr_c['offset']})
                        if _snap_cycle:
                            st.session_state._pcr_sr_snapshot = _snap_cycle
                    except Exception:
                        pass

                    # All S/R alerts — send ONLY the short alert header (no repeated full signal)
                    _sa_c  = getattr(st.session_state, '_sa_result', None)
                    _pcr_s = getattr(st.session_state, '_pcr_sr_snapshot', [])
                    _df5m_c = getattr(st.session_state, '_df_5m', None)

                    def _send_with_header(header):
                        # Cooldown + market-hours check are now handled inside send_telegram_message_sync
                        send_telegram_message_sync(header, force=False)

                    # PCR S/R proximity alert
                    try:
                        _h = check_pcr_sr_proximity_alert(option_data['underlying'])
                        if _h: _send_with_header(_h)
                    except Exception:
                        pass

                    # Candle pattern at S/R alert
                    try:
                        _h = send_candle_at_sr_alert(
                            master['candle'], option_data['underlying'],
                            _pcr_s, master.get('support_levels', []), master.get('resistance_levels', []),
                        )
                        if _h: _send_with_header(_h)
                    except Exception:
                        pass

                    # Capping at S/R alert
                    try:
                        if _sa_c is not None:
                            _h = send_capping_at_sr_alert(_sa_c, option_data['underlying'])
                            if _h: _send_with_header(_h)
                    except Exception:
                        pass

                    # Decapping / Depeg alert
                    try:
                        _h = send_decapping_alert(option_data['underlying'])
                        if _h: _send_with_header(_h)
                    except Exception:
                        pass

                    # Order Block zone alert
                    try:
                        _ob_data = master.get('order_blocks', {})
                        _h = send_ob_zone_alert(_ob_data, option_data['underlying'])
                        if _h: _send_with_header(_h)
                    except Exception:
                        pass

                    # Rejection / bounce at strongest wall
                    try:
                        _h = send_rejection_alert(
                            master['candle'], option_data['underlying'],
                            _df5m_c, _sa_c, _pcr_s,
                            master.get('support_levels', []), master.get('resistance_levels', []),
                        )
                        if _h: _send_with_header(_h)
                    except Exception:
                        pass

                    # Signal Banner
                    if 'BUY' in master['trade_type'] or 'BREAKOUT' in master['signal']:
                        sig_color = '#00ff88'
                    elif 'SELL' in master['trade_type'] or 'BREAKDOWN' in master['signal']:
                        sig_color = '#ff4444'
                    else:
                        sig_color = '#FFD700'
                    st.markdown(f"""
                    <div style="background:{sig_color}20;padding:25px;border-radius:15px;border:3px solid {sig_color};text-align:center;margin-bottom:15px;">
                        <h1 style="color:{sig_color};margin:0;font-size:2.5em;">{master['signal']}</h1>
                        <h2 style="color:white;margin:5px 0;">{master['trade_type']}</h2>
                        <h3 style="color:{sig_color};margin:5px 0;">Confluence: {master['abs_score']}/10 ({master['strength']}) | Confidence: {master['confidence']}%</h3>
                        <div style="background:#33333380;border-radius:10px;height:14px;margin:10px auto;max-width:500px;">
                            <div style="background:{sig_color};border-radius:10px;height:14px;width:{master['confidence']}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Details in columns
                    ms_col1, ms_col2, ms_col3 = st.columns(3)
                    with ms_col1:
                        st.markdown("#### 🕯 Candle Pattern (Nifty Price Action)")
                        cd = master['candle']
                        d = cd['details']
                        pat_color = '#00ff88' if cd['direction'] == 'Bullish' else '#ff4444' if cd['direction'] == 'Bearish' else '#FFD700'
                        st.markdown(f"""
                        <div style="background:{pat_color}20;padding:10px;border-radius:8px;border-left:4px solid {pat_color};margin-bottom:8px;">
                            <b style="color:{pat_color};font-size:16px;">{cd['pattern']}</b> <span style="color:white;">({cd['direction']})</span><br>
                            <span style="color:#aaa;font-size:13px;">O: ₹{d['open']:.2f} | H: ₹{d['high']:.2f} | L: ₹{d['low']:.2f} | C: ₹{d['close']:.2f}</span><br>
                            <span style="color:#aaa;font-size:12px;">Body: {d['body_ratio']*100:.0f}% | {'🟢 Bull' if d['is_green'] else '🔴 Bear'}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        # Last 5 candles breakdown
                        st.markdown("**Last 5 Candles:**")
                        candle_rows = []
                        for ci, cn in enumerate(cd.get('candles', [])):
                            candle_rows.append({
                                '#': ci + 1,
                                'Time': cn['time'],
                                'Type': f"{'🟢' if cn['type'] == 'Bull' else '🔴'} {cn['type']}",
                                'Pattern': cn['pattern'],
                                'O': f"₹{cn['open']:.0f}",
                                'H': f"₹{cn['high']:.0f}",
                                'L': f"₹{cn['low']:.0f}",
                                'C': f"₹{cn['close']:.0f}",
                                'Vol': f"{cn['volume']:,}" if cn['volume'] else '-',
                            })
                        if candle_rows:
                            st.dataframe(pd.DataFrame(candle_rows), use_container_width=True, hide_index=True, height=210)
                        st.caption(f"Bull: {cd.get('bull_count', 0)} | Bear: {cd.get('bear_count', 0)} out of last 5")

                        st.markdown(f"**📍 Location:** {', '.join(master['location'])}")
                        vol_color = '#00ff88' if master['volume']['spike'] else '#888888'
                        st.markdown(f"**📊 Volume:** <span style='color:{vol_color}'>{master['volume']['label']} ({master['volume']['ratio']}x)</span>", unsafe_allow_html=True)
                        st.markdown("#### 🟢 Order Blocks")
                        ob = master['order_blocks']
                        if ob.get('bullish_ob'):
                            st.success(f"Bullish OB: ₹{ob['bullish_ob']['low']:.0f} - ₹{ob['bullish_ob']['high']:.0f}")
                        else:
                            st.info("No Bullish OB detected")
                        if ob.get('bearish_ob'):
                            st.error(f"Bearish OB: ₹{ob['bearish_ob']['low']:.0f} - ₹{ob['bearish_ob']['high']:.0f}")
                        else:
                            st.info("No Bearish OB detected")

                    with ms_col2:
                        st.markdown("#### 🔮 GEX (from app)")
                        gex = master['gex']
                        gex_items = [
                            ("Net GEX", f"{gex['net_gex']:+.1f}L"),
                            ("ATM GEX", f"{gex['atm_gex']:+.1f}L"),
                            ("Gamma Flip", f"₹{gex['gamma_flip']:.0f}" if gex['gamma_flip'] else "N/A"),
                            ("Magnet", f"₹{gex['magnet']:.0f}" if gex['magnet'] else "N/A"),
                            ("Repeller", f"₹{gex['repeller']:.0f}" if gex['repeller'] else "N/A"),
                            ("Market Mode", gex['market_mode']),
                        ]
                        for label, val in gex_items:
                            st.markdown(f"**{label}:** {val}")
                        if gex['above_flip'] is not None:
                            if gex['above_flip']:
                                st.success("Above Gamma Flip (Bullish)")
                            else:
                                st.error("Below Gamma Flip (Bearish)")
                        st.markdown(f"#### 📊 PCR × GEX")
                        st.markdown(f"**{master['pcr_gex']['badge']}**")

                        st.markdown("#### 🟥 Resistance / 🟩 Support")
                        for r in master['resistance_levels'][:3]:
                            st.markdown(f"🟥 ₹{r:.0f}")
                        for s in master['support_levels'][:3]:
                            st.markdown(f"🟩 ₹{s:.0f}")

                        st.markdown("#### 🔲 Order Blocks (LuxAlgo)")
                        _ob_ui = master.get('order_blocks', {})
                        _current_p = option_data.get('underlying', 0)
                        for _bobs, _em, _lbl in [
                            (_ob_ui.get('bullish_obs', []), '🟩', 'Demand'),
                            (_ob_ui.get('bearish_obs', []), '🟥', 'Supply'),
                        ]:
                            for _bb in _bobs[:2]:
                                _dist = abs(_current_p - _bb['avg'])
                                _inside = _bb['low'] <= _current_p <= _bb['high']
                                _flag = ' ⚡IN ZONE' if _inside else f" {_dist:.0f}pts"
                                st.markdown(
                                    f'{_em} **{_lbl}** ₹{_bb["low"]:.0f}–₹{_bb["high"]:.0f}'
                                    f'<span style="font-size:11px;color:#aaa;">{_flag}</span>',
                                    unsafe_allow_html=True
                                )

                    with ms_col3:
                        st.markdown("#### 📉 VIX")
                        vix = master['vix']
                        vix_val = vix.get('vix', 0)
                        vix_dir = vix.get('direction', 'Unknown')
                        if vix_val > 0:
                            vix_aligned = ('Aligned' in str(master['reasons']))
                            if vix_dir == 'Falling':
                                st.success(f"VIX: {vix_val:.2f} ({vix_dir}) {'- Aligned' if vix_aligned else '- Opposite'}")
                            elif vix_dir == 'Rising':
                                st.error(f"VIX: {vix_val:.2f} ({vix_dir}) {'- Aligned' if vix_aligned else '- Opposite'}")
                            else:
                                st.info(f"VIX: {vix_val:.2f} ({vix_dir})")
                        else:
                            st.info("VIX data loading...")

                        st.markdown("#### 📋 Confluence Factors")
                        for reason in master['reasons']:
                            st.markdown(f"✔ {reason}")

                    # === OI TREND ANALYSIS (below 3-col, above alignment) ===
                    oi_t = master.get('oi_trend', {})
                    if oi_t.get('atm_strike'):
                        st.markdown("---")
                        st.markdown(f"## 📊 OI Timeline Trend Analysis (ATM: {oi_t['atm_strike']})")
                        ot_col1, ot_col2, ot_col3 = st.columns(3)
                        with ot_col1:
                            st.markdown("#### CE Activity (Resistance)")
                            ce_act = oi_t.get('ce_activity', 'N/A')
                            ce_color = '#ff8800' if ce_act == 'Long Building' else '#ff4444' if ce_act == 'Short Building' else '#00ff88' if ce_act == 'Short Covering' else '#888888'
                            st.markdown(f'<div style="background:{ce_color}30;padding:10px;border-radius:8px;border-left:4px solid {ce_color};"><b style="color:{ce_color};font-size:16px;">CE: {ce_act}</b><br><span style="color:#aaa;">OI: {oi_t.get("ce_oi_pct", 0):+.1f}% | LTP: {oi_t.get("ce_ltp_pct", 0):+.1f}%</span><br><span style="color:#aaa;">ChgOI: {oi_t.get("ce_chgoi", 0):,} ({oi_t.get("ce_chgoi_trend", "N/A")})</span></div>', unsafe_allow_html=True)
                            res_status = oi_t.get('resistance_status', 'N/A')
                            res_clr = '#ff4444' if 'Building' in res_status else '#00ff88' if res_status in ['Breaking', 'Weakening'] else '#888'
                            st.markdown(f'<div style="margin-top:8px;padding:8px;border-radius:6px;background:{res_clr}20;border-left:3px solid {res_clr};"><b style="color:{res_clr};">Resistance: {res_status}</b></div>', unsafe_allow_html=True)
                        with ot_col2:
                            st.markdown("#### PE Activity (Support)")
                            pe_act = oi_t.get('pe_activity', 'N/A')
                            pe_color = '#ff4444' if pe_act == 'Long Building' else '#00ff88' if pe_act == 'Short Building' else '#00cc66' if pe_act == 'Short Covering' else '#888888'
                            st.markdown(f'<div style="background:{pe_color}30;padding:10px;border-radius:8px;border-left:4px solid {pe_color};"><b style="color:{pe_color};font-size:16px;">PE: {pe_act}</b><br><span style="color:#aaa;">OI: {oi_t.get("pe_oi_pct", 0):+.1f}% | LTP: {oi_t.get("pe_ltp_pct", 0):+.1f}%</span><br><span style="color:#aaa;">ChgOI: {oi_t.get("pe_chgoi", 0):,} ({oi_t.get("pe_chgoi_trend", "N/A")})</span></div>', unsafe_allow_html=True)
                            sup_status = oi_t.get('support_status', 'N/A')
                            sup_clr = '#00ff88' if 'Building' in sup_status else '#ff4444' if sup_status in ['Breaking', 'Weakening'] else '#888'
                            st.markdown(f'<div style="margin-top:8px;padding:8px;border-radius:6px;background:{sup_clr}20;border-left:3px solid {sup_clr};"><b style="color:{sup_clr};">Support: {sup_status}</b></div>', unsafe_allow_html=True)
                        with ot_col3:
                            st.markdown("#### OI Trend Signal")
                            oi_sig = oi_t.get('signal', 'Neutral')
                            if 'Bullish' in oi_sig:
                                sig_clr = '#00ff88'
                            elif 'Bearish' in oi_sig:
                                sig_clr = '#ff4444'
                            elif oi_sig == 'Range':
                                sig_clr = '#FFD700'
                            elif oi_sig == 'Volatile':
                                sig_clr = '#FFA500'
                            else:
                                sig_clr = '#888888'
                            st.markdown(f'<div style="background:{sig_clr}30;padding:15px;border-radius:10px;border:2px solid {sig_clr};text-align:center;"><b style="color:{sig_clr};font-size:20px;">{oi_sig.upper()}</b><br><span style="color:#aaa;">CE: {oi_t.get("ce_activity", "N/A")} | PE: {oi_t.get("pe_activity", "N/A")}</span></div>', unsafe_allow_html=True)
                            st.markdown("")
                            st.markdown(f"**Interpretation:**")
                            if 'Bullish' in oi_sig:
                                st.success("Support building + Resistance weakening/breaking = Upside expected")
                            elif 'Bearish' in oi_sig:
                                st.error("Resistance building + Support weakening/breaking = Downside expected")
                            elif oi_sig == 'Range':
                                st.warning("Both support & resistance building = Sideways/Range-bound")
                            elif oi_sig == 'Volatile':
                                st.warning("Both covering = High volatility expected, breakout imminent")
                            else:
                                st.info("Insufficient OI trend data")

                    # === VOB / VIDYA / HTF S&R / DELTA VOLUME ===
                    st.markdown("---")
                    st.markdown("## 📊 Price Action Analysis (VOB / VIDYA / HTF S&R)")
                    pa_col1, pa_col2, pa_col3 = st.columns(3)

                    with pa_col1:
                        st.markdown("#### 🔮 VIDYA Trend")
                        vidya = master.get('vidya', {})
                        v_trend = vidya.get('trend', 'Unknown')
                        v_color = '#00ff88' if v_trend == 'Bullish' else '#ff4444' if v_trend == 'Bearish' else '#888'
                        cross_text = ''
                        if vidya.get('cross_up'):
                            cross_text = ' | Fresh ▲ Cross'
                        elif vidya.get('cross_down'):
                            cross_text = ' | Fresh ▼ Cross'
                        st.markdown(f'<div style="background:{v_color}30;padding:12px;border-radius:8px;border-left:4px solid {v_color};"><b style="color:{v_color};font-size:18px;">VIDYA: {v_trend}{cross_text}</b><br><span style="color:#aaa;">Smoothed: ₹{vidya.get("smoothed_last", 0):.0f} | Delta Vol: {vidya.get("delta_pct", 0):+.0f}%</span><br><span style="color:#aaa;">Buy Vol: {vidya.get("buy_vol", 0):,.0f} | Sell Vol: {vidya.get("sell_vol", 0):,.0f}</span></div>', unsafe_allow_html=True)

                        st.markdown("#### 🔄 LTP Trap")
                        trap = master.get('ltp_trap', {})
                        if trap.get('buy_trap'):
                            st.success(f"LTP Trap BUY detected | VWAP: ₹{trap.get('vwap', 0):.0f}")
                        elif trap.get('sell_trap'):
                            st.error(f"LTP Trap SELL detected | VWAP: ₹{trap.get('vwap', 0):.0f}")
                        else:
                            st.info(f"No LTP Trap | VWAP: ₹{trap.get('vwap', 0):.0f} | Price {trap.get('price_vs_vwap', 'N/A')}")

                    with pa_col2:
                        st.markdown("#### 🟢🔴 VOB Zones")
                        vob_b = master.get('vob_blocks', {})
                        if vob_b:
                            for b in vob_b.get('bullish', [])[-3:]:
                                st.markdown(f'<div style="background:#00ff8820;padding:6px;border-radius:6px;border-left:3px solid #00ff88;font-size:13px;margin-bottom:4px;"><b style="color:#00ff88;">Support</b> ₹{b["lower"]:.0f} - ₹{b["upper"]:.0f} | Vol: {b.get("volume", 0):,.0f}</div>', unsafe_allow_html=True)
                            for b in vob_b.get('bearish', [])[-3:]:
                                st.markdown(f'<div style="background:#ff444420;padding:6px;border-radius:6px;border-left:3px solid #ff4444;font-size:13px;margin-bottom:4px;"><b style="color:#ff4444;">Resistance</b> ₹{b["lower"]:.0f} - ₹{b["upper"]:.0f} | Vol: {b.get("volume", 0):,.0f}</div>', unsafe_allow_html=True)
                        else:
                            st.info("VOB data loading...")

                        st.markdown("#### 🟢🔴 HVP (High Volume Pivots)")
                        hvp = master.get('hvp', {})
                        for h in hvp.get('bullish_hvp', [])[-3:]:
                            st.markdown(f'<span style="color:#00ff88;">🟢 Support ₹{h["price"]:.0f}</span>', unsafe_allow_html=True)
                        for h in hvp.get('bearish_hvp', [])[-3:]:
                            st.markdown(f'<span style="color:#ff4444;">🔴 Resistance ₹{h["price"]:.0f}</span>', unsafe_allow_html=True)
                        if not hvp.get('bullish_hvp') and not hvp.get('bearish_hvp'):
                            st.info("No HVP detected")

                    with pa_col3:
                        st.markdown("#### 📐 HTF S&R (Price Pivots)")
                        htf = master.get('htf_sr', {})
                        htf_levels = htf.get('levels', [])
                        if htf_levels:
                            current_p = option_data['underlying']
                            # Show nearest 3 support and 3 resistance
                            sup_levels = sorted([l for l in htf_levels if l['type'] == 'Support' and l['level'] < current_p], key=lambda x: -x['level'])[:3]
                            res_levels = sorted([l for l in htf_levels if l['type'] == 'Resistance' and l['level'] > current_p], key=lambda x: x['level'])[:3]
                            for r in res_levels:
                                st.markdown(f'<div style="background:#ff444418;padding:4px 8px;border-radius:4px;margin-bottom:3px;font-size:13px;"><b style="color:#ff4444;">R</b> ₹{r["level"]:.0f} <span style="color:#888;">({r["tf"]})</span></div>', unsafe_allow_html=True)
                            st.markdown(f'<div style="background:#FFD70020;padding:4px 8px;border-radius:4px;margin-bottom:3px;text-align:center;"><b style="color:#FFD700;">PRICE ₹{current_p:.0f}</b></div>', unsafe_allow_html=True)
                            for s in sup_levels:
                                st.markdown(f'<div style="background:#00ff8818;padding:4px 8px;border-radius:4px;margin-bottom:3px;font-size:13px;"><b style="color:#00ff88;">S</b> ₹{s["level"]:.0f} <span style="color:#888;">({s["tf"]})</span></div>', unsafe_allow_html=True)
                        else:
                            st.info("HTF pivot data loading...")

                        st.markdown(f"#### 📊 Delta Volume Trend")
                        d_trend = master.get('delta_trend', 'Neutral')
                        dt_clr = '#00ff88' if d_trend == 'Bullish' else '#ff4444' if d_trend == 'Bearish' else '#888'
                        st.markdown(f'<div style="background:{dt_clr}25;padding:8px;border-radius:6px;border-left:3px solid {dt_clr};"><b style="color:{dt_clr};">{d_trend}</b></div>', unsafe_allow_html=True)

                    # VPFR Table
                    _vpfr = master.get('vpfr', {}) or {}
                    _vpfr_rows = []
                    for _tf, _label, _bars in [('short', 'Short', 30), ('medium', 'Medium', 60), ('long', 'Long', 180)]:
                        _vd = _vpfr.get(_tf)
                        if _vd:
                            _spot = option_data['underlying']
                            _poc, _vah, _val = _vd['poc'], _vd['vah'], _vd['val']
                            _pos = 'Above VAH' if _spot > _vah else 'Below VAL' if _spot < _val else 'In VA'
                            _poc_dist = _spot - _poc
                            _vpfr_rows.append({
                                'Timeframe': f'{_label} ({_bars})',
                                'POC': f'₹{_poc:.0f}',
                                'VAH': f'₹{_vah:.0f}',
                                'VAL': f'₹{_val:.0f}',
                                'Spot vs VA': _pos,
                                'Dist to POC': f'{_poc_dist:+.0f}',
                            })
                    if _vpfr_rows:
                        st.markdown("#### 📊 VPFR — Volume Profile Fixed Range")
                        st.caption("POC = price magnet | VAH = resistance | VAL = support | 70% value area")
                        _vpfr_df = pd.DataFrame(_vpfr_rows)
                        def _style_vpfr(row):
                            pos = str(row.get('Spot vs VA', ''))
                            if pos == 'Above VAH':
                                return ['background-color:#00ff8815'] * len(row)
                            elif pos == 'Below VAL':
                                return ['background-color:#ff444415'] * len(row)
                            return ['background-color:#FFD70010'] * len(row)
                        st.dataframe(_vpfr_df.style.apply(_style_vpfr, axis=1), use_container_width=True, hide_index=True)

                    # Delta Volume Chart
                    delta_df = master.get('delta_vol_df')
                    if delta_df is not None and len(delta_df) > 0:
                        # Filter to today only
                        _today_ist = datetime.now(pytz.timezone('Asia/Kolkata')).date()
                        delta_df = delta_df[delta_df['datetime'].dt.date == _today_ist]
                    if delta_df is not None and len(delta_df) > 0:
                        st.markdown("### 📊 Delta Volume (Buy vs Sell)")
                        fig_delta = go.Figure()
                        colors = ['#00ff88' if d > 0 else '#ff4444' for d in delta_df['delta']]
                        fig_delta.add_trace(go.Bar(
                            x=delta_df['datetime'], y=delta_df['delta'],
                            marker_color=colors, name='Delta Volume', opacity=0.7,
                        ))
                        fig_delta.add_trace(go.Scatter(
                            x=delta_df['datetime'], y=delta_df['delta_ma'],
                            mode='lines', name='Delta MA(10)',
                            line=dict(color='#FFD700', width=2),
                        ))
                        fig_delta.add_trace(go.Scatter(
                            x=delta_df['datetime'], y=delta_df['cum_delta'],
                            mode='lines', name='Cumulative Delta',
                            line=dict(color='#00BFFF', width=2, dash='dot'),
                            yaxis='y2',
                        ))
                        fig_delta.update_layout(
                            template='plotly_dark', height=350,
                            xaxis=dict(tickformat='%H:%M', title='Time'),
                            yaxis=dict(title='Delta Volume'),
                            yaxis2=dict(title='Cumulative Delta', overlaying='y', side='right', showgrid=False),
                            plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                            hovermode='x unified', barmode='relative',
                        )
                        fig_delta.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, opacity=0.5)
                        st.plotly_chart(fig_delta, use_container_width=True)

                    # === FULL ALIGNMENT TABLE (below the 3-col section) ===
                    st.markdown("---")
                    st.markdown("## 🌍 Index & Stock Alignment - Candle Patterns & Sentiment")
                    align_data = master.get('alignment', {})
                    if align_data:
                        # Candle Pattern Table
                        st.markdown("### 🕯 Candle Patterns Across Indices")
                        pattern_rows = []
                        # Define display order
                        display_order = ['NIFTY 50', 'SENSEX', 'BANKNIFTY', 'NIFTY IT', 'RELIANCE', 'ICICIBANK', 'INDIA VIX', 'GOLD', 'CRUDE OIL', 'USD/INR', 'S&P 500', 'JAPAN 225', 'HANG SENG', 'UK 100']
                        for name in display_order:
                            ad = align_data.get(name)
                            if ad is None:
                                continue
                            pat_emoji = '🟢' if ad.get('candle_dir') == 'Bullish' else '🔴' if ad.get('candle_dir') == 'Bearish' else '🟡'
                            pattern_rows.append({
                                'Index': name,
                                'LTP': f"₹{ad['ltp']:.2f}" if ad.get('ltp', 0) > 0 else 'N/A',
                                'Candle': f"{pat_emoji} {ad.get('candle_pattern', 'N/A')}",
                                'Direction': ad.get('candle_dir', 'N/A'),
                                'Bull/Bear (5)': f"{ad.get('bull_count', '-')}/{ad.get('bear_count', '-')}",
                                'Day H': f"₹{ad['day_high']:.0f}" if ad.get('day_high') else '-',
                                'Day L': f"₹{ad['day_low']:.0f}" if ad.get('day_low') else '-',
                            })
                        if pattern_rows:
                            pat_df = pd.DataFrame(pattern_rows)
                            def _style_pat(row):
                                d = row['Direction']
                                if d == 'Bullish':
                                    return ['background-color:#00ff8815;color:white'] * len(row)
                                elif d == 'Bearish':
                                    return ['background-color:#ff444415;color:white'] * len(row)
                                return [''] * len(row)
                            st.dataframe(pat_df.style.apply(_style_pat, axis=1), use_container_width=True, hide_index=True)

                        # Multi-timeframe Sentiment Table
                        st.markdown("### 📊 Multi-Timeframe Sentiment (Price Action)")
                        sent_rows = []
                        for name in display_order:
                            ad = align_data.get(name)
                            if ad is None:
                                continue
                            def _sent_emoji(s):
                                return '🟢' if s == 'Bullish' else '🔴' if s == 'Bearish' else '🟡'
                            s10 = ad.get('sentiment_10m', 'N/A')
                            s1h = ad.get('sentiment_1h', 'N/A')
                            s4h = ad.get('sentiment_4h', 'N/A')
                            s1d = ad.get('sentiment_1d', 'N/A')
                            s4d = ad.get('sentiment_4d', 'N/A')
                            sent_rows.append({
                                'Index': name,
                                'LTP': f"₹{ad['ltp']:.2f}" if ad.get('ltp', 0) > 0 else 'N/A',
                                '10 Min': f"{_sent_emoji(s10)} {s10} ({ad.get('pct_10m', 0):+.2f}%)" if s10 != 'N/A' else 'N/A',
                                '1 Hour': f"{_sent_emoji(s1h)} {s1h} ({ad.get('pct_1h', 0):+.2f}%)" if s1h != 'N/A' else 'N/A',
                                '4 Hours': f"{_sent_emoji(s4h)} {s4h} ({ad.get('pct_4h', 0):+.2f}%)" if s4h != 'N/A' else 'N/A',
                                '1 Day': f"{_sent_emoji(s1d)} {s1d} ({ad.get('pct_1d', 0):+.2f}%)" if s1d != 'N/A' else 'N/A',
                                '4 Days': f"{_sent_emoji(s4d)} {s4d} ({ad.get('pct_4d', 0):+.2f}%)" if s4d != 'N/A' else 'N/A',
                                'Trend': ad.get('trend', 'N/A'),
                            })
                        if sent_rows:
                            sent_df = pd.DataFrame(sent_rows)
                            def _style_sent(row):
                                t = row['Trend']
                                if t == 'Bullish':
                                    return ['background-color:#00ff8815;color:white'] * len(row)
                                elif t == 'Bearish':
                                    return ['background-color:#ff444415;color:white'] * len(row)
                                return [''] * len(row)
                            st.dataframe(sent_df.style.apply(_style_sent, axis=1), use_container_width=True, hide_index=True)

                        # Summary
                        non_vix_align = {k: v for k, v in align_data.items() if 'VIX' not in k}
                        bull_10m = sum(1 for v in non_vix_align.values() if v.get('sentiment_10m') == 'Bullish')
                        bear_10m = sum(1 for v in non_vix_align.values() if v.get('sentiment_10m') == 'Bearish')
                        bull_1h = sum(1 for v in non_vix_align.values() if v.get('sentiment_1h') == 'Bullish')
                        bear_1h = sum(1 for v in non_vix_align.values() if v.get('sentiment_1h') == 'Bearish')
                        total_idx = len(non_vix_align)
                        sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)
                        with sum_col1:
                            if bull_10m > bear_10m:
                                st.success(f"10m: Bullish ({bull_10m}/{total_idx})")
                            elif bear_10m > bull_10m:
                                st.error(f"10m: Bearish ({bear_10m}/{total_idx})")
                            else:
                                st.warning(f"10m: Mixed ({bull_10m}B/{bear_10m}R)")
                        with sum_col2:
                            if bull_1h > bear_1h:
                                st.success(f"1h: Bullish ({bull_1h}/{total_idx})")
                            elif bear_1h > bull_1h:
                                st.error(f"1h: Bearish ({bear_1h}/{total_idx})")
                            else:
                                st.warning(f"1h: Mixed ({bull_1h}B/{bear_1h}R)")
                        with sum_col3:
                            bull_4h = sum(1 for v in non_vix_align.values() if v.get('sentiment_4h') == 'Bullish')
                            bear_4h = sum(1 for v in non_vix_align.values() if v.get('sentiment_4h') == 'Bearish')
                            if bull_4h > bear_4h:
                                st.success(f"4h: Bullish ({bull_4h}/{total_idx})")
                            elif bear_4h > bull_4h:
                                st.error(f"4h: Bearish ({bear_4h}/{total_idx})")
                            else:
                                st.warning(f"4h: Mixed ({bull_4h}B/{bear_4h}R)")
                        with sum_col4:
                            bull_1d = sum(1 for v in non_vix_align.values() if v.get('sentiment_1d') == 'Bullish')
                            bear_1d = sum(1 for v in non_vix_align.values() if v.get('sentiment_1d') == 'Bearish')
                            if bull_1d > bear_1d:
                                st.success(f"1D: Bullish ({bull_1d}/{total_idx})")
                            elif bear_1d > bull_1d:
                                st.error(f"1D: Bearish ({bear_1d}/{total_idx})")
                            else:
                                st.warning(f"1D: Mixed ({bull_1d}B/{bear_1d}R)")
                        with sum_col5:
                            bull_4d = sum(1 for v in non_vix_align.values() if v.get('sentiment_4d') == 'Bullish')
                            bear_4d = sum(1 for v in non_vix_align.values() if v.get('sentiment_4d') == 'Bearish')
                            if bull_4d > bear_4d:
                                st.success(f"4D: Bullish ({bull_4d}/{total_idx})")
                            elif bear_4d > bull_4d:
                                st.error(f"4D: Bearish ({bear_4d}/{total_idx})")
                            else:
                                st.warning(f"4D: Mixed ({bull_4d}B/{bear_4d}R)")
                        # === % Change from Open - Line Chart ===
                        st.markdown("### 📈 Price Action - % Change from Open (Today)")
                        fig_pct = go.Figure()
                        line_colors = {
                            'NIFTY 50': '#FFFFFF',   # white
                            'SENSEX':    '#8B00FF',  # violet
                            'BANKNIFTY': '#FF69B4',  # pink
                            'NIFTY IT':  '#1E90FF',  # blue
                            'RELIANCE':  '#00FF00',  # green
                            'ICICIBANK': '#FFFF00',  # yellow
                            'INDIA VIX': '#FFA500',  # orange
                            'USD/INR':   '#FF0000',  # red
                            'GOLD':      '#FFD700',  # gold
                            'CRUDE OIL': '#FF66CC',  # rose pink
                        }
                        for name in display_order:
                            ad = align_data.get(name)
                            if ad is None:
                                continue
                            pct_time = ad.get('pct_series_time', [])
                            pct_vals = ad.get('pct_series_vals', [])
                            if pct_time and pct_vals:
                                line_width = 3 if name == 'NIFTY 50' else 2
                                inverse_instruments = {'INDIA VIX', 'CRUDE OIL', 'USD/INR'}
                                dash = 'dot' if name == 'INDIA VIX' else 'dashdot' if name in ('GOLD', 'CRUDE OIL', 'USD/INR') else None
                                plot_vals = [-v for v in pct_vals] if name in inverse_instruments else pct_vals
                                display_name = f"{name} (inv)" if name in inverse_instruments else name
                                fig_pct.add_trace(go.Scatter(
                                    x=pct_time, y=plot_vals,
                                    mode='lines',
                                    name=display_name,
                                    line=dict(color=line_colors.get(name, '#888'), width=line_width, dash=dash),
                                ))
                        fig_pct.add_hline(y=0, line_dash="solid", line_color="white", line_width=1.5)
                        _ist = pytz.timezone('Asia/Kolkata')
                        _today = datetime.now(_ist).date()
                        _x_start = _ist.localize(datetime(_today.year, _today.month, _today.day, 8, 30))
                        _x_end = _ist.localize(datetime(_today.year, _today.month, _today.day, 15, 45))
                        fig_pct.update_layout(
                            title='All Indices & Stocks - % Change from Day Open',
                            template='plotly_dark',
                            height=450,
                            xaxis=dict(tickformat='%H:%M', title='Time', range=[_x_start, _x_end]),
                            yaxis=dict(title='% Change', zeroline=True, zerolinecolor='white', zerolinewidth=2,
                                       ticksuffix='%'),
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig_pct, use_container_width=True)

                        # === MARKET DRIVER ANALYSIS ===
                        st.markdown("### 🔥 Nifty Market Driver Analysis")
                        st.caption("Direct = moves WITH Nifty | Inverse = moves AGAINST Nifty (when inverse falls, Nifty rises)")

                        # Classification
                        direct_instruments = ['SENSEX', 'BANKNIFTY', 'NIFTY IT', 'RELIANCE', 'ICICIBANK', 'GOLD', 'S&P 500', 'JAPAN 225', 'HANG SENG', 'UK 100']
                        inverse_instruments = ['INDIA VIX', 'CRUDE OIL', 'USD/INR']

                        nifty_data = align_data.get('NIFTY 50')
                        nifty_10m = nifty_data.get('sentiment_10m', 'N/A') if nifty_data else 'N/A'
                        nifty_1h = nifty_data.get('sentiment_1h', 'N/A') if nifty_data else 'N/A'
                        nifty_pct = nifty_data.get('pct_10m', 0) if nifty_data else 0

                        # Build driver table
                        driver_rows = []
                        direct_bull_10m = 0
                        direct_bear_10m = 0
                        inverse_supporting_bull = 0  # inverse bearish = supports Nifty bull
                        inverse_supporting_bear = 0  # inverse bullish = supports Nifty bear
                        strongest_driver = None
                        strongest_pct = 0

                        for name in display_order:
                            if name == 'NIFTY 50':
                                continue
                            ad = align_data.get(name)
                            if ad is None:
                                continue
                            s10 = ad.get('sentiment_10m', 'N/A')
                            pct10 = ad.get('pct_10m', 0)
                            s1h = ad.get('sentiment_1h', 'N/A')
                            is_inverse = name in inverse_instruments
                            corr_type = 'Inverse' if is_inverse else 'Direct'

                            # Determine if this instrument supports Nifty bull or bear
                            if is_inverse:
                                if s10 == 'Bearish':
                                    nifty_impact = 'Bullish'
                                    inverse_supporting_bull += 1
                                elif s10 == 'Bullish':
                                    nifty_impact = 'Bearish'
                                    inverse_supporting_bear += 1
                                else:
                                    nifty_impact = 'Neutral'
                            else:
                                if s10 == 'Bullish':
                                    nifty_impact = 'Bullish'
                                    direct_bull_10m += 1
                                elif s10 == 'Bearish':
                                    nifty_impact = 'Bearish'
                                    direct_bear_10m += 1
                                else:
                                    nifty_impact = 'Neutral'

                            # Track strongest mover
                            if abs(pct10) > abs(strongest_pct):
                                strongest_pct = pct10
                                strongest_driver = name

                            impact_emoji = '🟢' if nifty_impact == 'Bullish' else '🔴' if nifty_impact == 'Bearish' else '⚪'
                            corr_emoji = '🔄' if is_inverse else '➡️'
                            driver_rows.append({
                                'Instrument': name,
                                'Type': f"{corr_emoji} {corr_type}",
                                '10m Move': f"{pct10:+.2f}%",
                                '10m Sentiment': f"{'🟢' if s10 == 'Bullish' else '🔴' if s10 == 'Bearish' else '🟡'} {s10}",
                                '1h Sentiment': f"{'🟢' if s1h == 'Bullish' else '🔴' if s1h == 'Bearish' else '🟡'} {s1h}",
                                'Nifty Impact': f"{impact_emoji} {nifty_impact}",
                            })

                        if driver_rows:
                            driver_df = pd.DataFrame(driver_rows)
                            def _style_driver(row):
                                impact = row.get('Nifty Impact', '')
                                if 'Bullish' in impact:
                                    return ['background-color:#00ff8812;color:white'] * len(row)
                                elif 'Bearish' in impact:
                                    return ['background-color:#ff444412;color:white'] * len(row)
                                return [''] * len(row)
                            st.dataframe(driver_df.style.apply(_style_driver, axis=1), use_container_width=True, hide_index=True)

                        # Composite Nifty Fire Signal
                        total_bull_support = direct_bull_10m + inverse_supporting_bull
                        total_bear_support = direct_bear_10m + inverse_supporting_bear
                        total_instruments = len(direct_instruments) + len(inverse_instruments)
                        available_instruments = sum(1 for n in direct_instruments + inverse_instruments if align_data.get(n) is not None)

                        st.markdown("#### 🎯 Composite Nifty Signal")
                        fire_col1, fire_col2, fire_col3, fire_col4 = st.columns(4)
                        with fire_col1:
                            st.metric("Bull Drivers", f"{total_bull_support}/{available_instruments}")
                        with fire_col2:
                            st.metric("Bear Drivers", f"{total_bear_support}/{available_instruments}")
                        with fire_col3:
                            if strongest_driver:
                                st.metric("Strongest Mover", f"{strongest_driver}", delta=f"{strongest_pct:+.2f}%")
                        with fire_col4:
                            nifty_pct_display = f"{nifty_pct:+.2f}%" if nifty_data else 'N/A'
                            st.metric("NIFTY 50", nifty_pct_display)

                        # Fire signal logic
                        bull_pct = (total_bull_support / available_instruments * 100) if available_instruments > 0 else 0
                        bear_pct = (total_bear_support / available_instruments * 100) if available_instruments > 0 else 0

                        if bull_pct >= 80:
                            st.success(f"🔥🔥🔥 NIFTY FIRE UP — {total_bull_support}/{available_instruments} instruments supporting bullish ({bull_pct:.0f}%) | Direct indices bullish + VIX/Crude/USD falling")
                        elif bull_pct >= 60:
                            st.success(f"🔥 NIFTY BULLISH — {total_bull_support}/{available_instruments} instruments supporting bullish ({bull_pct:.0f}%)")
                        elif bear_pct >= 80:
                            st.error(f"🔥🔥🔥 NIFTY FIRE DOWN — {total_bear_support}/{available_instruments} instruments supporting bearish ({bear_pct:.0f}%) | Direct indices bearish + VIX/Crude/USD rising")
                        elif bear_pct >= 60:
                            st.error(f"🔥 NIFTY BEARISH — {total_bear_support}/{available_instruments} instruments supporting bearish ({bear_pct:.0f}%)")
                        else:
                            st.warning(f"⚖️ MIXED SIGNALS — Bull: {total_bull_support} vs Bear: {total_bear_support} | No clear direction")

                        # Detailed breakdown
                        with st.expander("📋 Driver Breakdown"):
                            bd_col1, bd_col2 = st.columns(2)
                            with bd_col1:
                                st.markdown("**➡️ Direct (move WITH Nifty)**")
                                for name in direct_instruments:
                                    ad = align_data.get(name)
                                    if ad is None:
                                        continue
                                    s10 = ad.get('sentiment_10m', 'N/A')
                                    pct = ad.get('pct_10m', 0)
                                    emoji = '🟢' if s10 == 'Bullish' else '🔴' if s10 == 'Bearish' else '🟡'
                                    st.markdown(f"{emoji} **{name}**: {pct:+.2f}% ({s10})")
                            with bd_col2:
                                st.markdown("**🔄 Inverse (move AGAINST Nifty)**")
                                for name in inverse_instruments:
                                    ad = align_data.get(name)
                                    if ad is None:
                                        continue
                                    s10 = ad.get('sentiment_10m', 'N/A')
                                    pct = ad.get('pct_10m', 0)
                                    # For inverse: bearish = good for Nifty
                                    if s10 == 'Bearish':
                                        st.markdown(f"🟢 **{name}**: {pct:+.2f}% (Falling = Nifty Bullish)")
                                    elif s10 == 'Bullish':
                                        st.markdown(f"🔴 **{name}**: {pct:+.2f}% (Rising = Nifty Bearish)")
                                    else:
                                        st.markdown(f"🟡 **{name}**: {pct:+.2f}% (Neutral)")
                    else:
                        st.info("Alignment data loading... will appear on next refresh.")
                else:
                    st.info("Generating master signal... waiting for data.")
            else:
                st.info("Master signal requires deep analysis data. Please wait...")
        except Exception as e:
            st.warning(f"Master signal unavailable: {str(e)}")

    # === MULTI-INSTRUMENT CAPPING / OI / VOLUME MONITOR ===
    st.markdown("---")
    st.markdown("## 📊 Multi-Instrument Capping · OI · Volume Monitor")
    st.caption("NIFTY · SENSEX · BANK NIFTY · RELIANCE · ICICI BANK · INFOSYS — Call Capping & Put Capping with Volume Confirmation (ATM ±4 strikes)")

    _mi_refresh = st.button("🔄 Refresh Instrument Data", key="mi_refresh_btn")
    if _mi_refresh or 'mi_instrument_data' not in st.session_state:
        with st.spinner("Fetching option chains for all instruments..."):
            st.session_state.mi_instrument_data = {
                key: get_instrument_capping_analysis(cfg)
                for key, cfg in INSTRUMENT_CONFIGS.items()
            }

    _mi_data = st.session_state.get('mi_instrument_data', {})

    # --- Build NIFTY summary from already-fetched option_data ---
    _nifty_summary = None
    try:
        if option_data and option_data.get('underlying') and option_data.get('df_summary') is not None:
            _ndf = option_data['df_summary']
            _nspot = option_data['underlying']
            _natm = min(_ndf['Strike'], key=lambda x: abs(x - _nspot))
            _near = _ndf[abs(_ndf['Strike'] - _natm) <= 200].copy()
            _n_ce_oi  = _near['openInterest_CE'].sum()
            _n_pe_oi  = _near['openInterest_PE'].sum()
            _n_ce_chg = _near['changeinOpenInterest_CE'].sum()
            _n_pe_chg = _near['changeinOpenInterest_PE'].sum()
            _n_ce_vol = _near['totalTradedVolume_CE'].sum() if 'totalTradedVolume_CE' in _near.columns else 0
            _n_pe_vol = _near['totalTradedVolume_PE'].sum() if 'totalTradedVolume_PE' in _near.columns else 0
            _n_pcr = _n_pe_oi / _n_ce_oi if _n_ce_oi > 0 else 1.0

            # Capping / support from sa_result if available
            _n_capping, _n_support = [], []
            try:
                _sa = sa_result
                if _sa and _sa.get('analysis_df') is not None:
                    _adf = _sa['analysis_df']
                    for _, _r in _adf.iterrows():
                        if _r['Call_Class'] in ('High Conviction Resistance', 'Strong Resistance') and _r['Strike'] > _nspot:
                            _n_capping.append({
                                'strike': _r['Strike'], 'oi_l': _r['CE_OI'] / 100000,
                                'chgoi_k': _r['CE_ChgOI'] / 1000, 'vol_k': _r['CE_Vol'] / 1000,
                                'vol_confirmed': _r.get('CE_Vol_High', False),
                                'strength': _r['Call_Strength'],
                            })
                        if _r['Put_Class'] in ('High Conviction Support', 'Strong Support') and _r['Strike'] < _nspot:
                            _n_support.append({
                                'strike': _r['Strike'], 'oi_l': _r['PE_OI'] / 100000,
                                'chgoi_k': _r['PE_ChgOI'] / 1000, 'vol_k': _r['PE_Vol'] / 1000,
                                'vol_confirmed': _r.get('PE_Vol_High', False),
                                'strength': _r['Put_Strength'],
                            })
                    _n_capping.sort(key=lambda x: x['oi_l'], reverse=True)
                    _n_support.sort(key=lambda x: x['oi_l'], reverse=True)
            except Exception:
                pass

            _nifty_summary = {
                'underlying': _nspot, 'pcr': _n_pcr,
                'pcr_bias': 'Bullish' if _n_pcr > 1.2 else 'Bearish' if _n_pcr < 0.7 else 'Neutral',
                'total_ce_chgoi_l': _n_ce_chg / 100000, 'total_pe_chgoi_l': _n_pe_chg / 100000,
                'total_ce_vol_k': _n_ce_vol / 1000, 'total_pe_vol_k': _n_pe_vol / 1000,
                'oi_bias': 'Bullish' if _n_pe_chg > _n_ce_chg else 'Bearish',
                'capping': _n_capping[:3], 'support': _n_support[:3],
            }
    except Exception:
        pass

    # Merge NIFTY + other instruments in display order
    _all_instruments = [('NIFTY', 'NIFTY 50', _nifty_summary)] + [
        (k, cfg['name'], _mi_data.get(k)) for k, cfg in INSTRUMENT_CONFIGS.items()
    ]

    # ── Table 1: Summary ──────────────────────────────────────────────────────
    st.markdown("### 📋 Instrument Summary")
    _summary_rows = []
    for inst_key, inst_name, res in _all_instruments:
        if not res or 'error' in res:
            _summary_rows.append({
                'Instrument': inst_name, 'LTP': '—', 'PCR': '—', 'PCR Bias': '—',
                'CE ΔOI (L)': '—', 'PE ΔOI (L)': '—', 'OI Bias': '—',
                'CE Vol (K)': '—', 'PE Vol (K)': '—',
            })
            continue
        _oi_bias_tag = '🟢 PE>' if res['total_pe_chgoi_l'] > res['total_ce_chgoi_l'] else '🔴 CE>'
        _summary_rows.append({
            'Instrument': inst_name,
            'LTP': f"₹{res['underlying']:,.1f}",
            'PCR': f"{res['pcr']:.2f}",
            'PCR Bias': res['pcr_bias'],
            'CE ΔOI (L)': f"{res['total_ce_chgoi_l']:+.1f}",
            'PE ΔOI (L)': f"{res['total_pe_chgoi_l']:+.1f}",
            'OI Bias': _oi_bias_tag,
            'CE Vol (K)': f"{res['total_ce_vol_k']:,.0f}",
            'PE Vol (K)': f"{res['total_pe_vol_k']:,.0f}",
        })
    _sum_df = pd.DataFrame(_summary_rows)

    def _style_summary(row):
        bias = row.get('PCR Bias', '')
        oi   = row.get('OI Bias', '')
        styles = [''] * len(row)
        pcr_idx = _sum_df.columns.get_loc('PCR Bias')
        oi_idx  = _sum_df.columns.get_loc('OI Bias')
        if bias == 'Bullish':
            styles[pcr_idx] = 'background-color:#00ff8830;color:#00ff88;font-weight:bold'
        elif bias == 'Bearish':
            styles[pcr_idx] = 'background-color:#ff444430;color:#ff4444;font-weight:bold'
        else:
            styles[pcr_idx] = 'background-color:#FFD70030;color:#FFD700;font-weight:bold'
        if '🟢' in str(oi):
            styles[oi_idx] = 'background-color:#00ff8830;color:#00ff88;font-weight:bold'
        elif '🔴' in str(oi):
            styles[oi_idx] = 'background-color:#ff444430;color:#ff4444;font-weight:bold'
        return styles

    st.dataframe(
        _sum_df.style.apply(_style_summary, axis=1),
        use_container_width=True, hide_index=True
    )

    # ── Table 2 & 3: Capping + Support ───────────────────────────────────────
    _cap_tab, _sup_tab = st.tabs(["🟥 Call Capping (CE)", "🟢 Put Capping (PE)"])

    def _build_strike_table(all_insts, side):
        rows = []
        for inst_key, inst_name, res in all_insts:
            if not res or 'error' in res:
                continue
            strikes = res.get(side, [])
            if not strikes:
                rows.append({
                    'Instrument': inst_name, 'Strike': '—', 'Strength': '—',
                    'Vol ✓': '—', 'OI (L)': '—', 'ΔOI (K)': '—', 'Vol (K)': '—',
                })
            for s in strikes:
                rows.append({
                    'Instrument': inst_name,
                    'Strike': f"₹{s['strike']:.0f}",
                    'Strength': s['strength'],
                    'Vol ✓': '🔥 Confirmed' if s['vol_confirmed'] else '〰 OI Only',
                    'OI (L)': f"{s['oi_l']:.1f}",
                    'ΔOI (K)': f"+{s['chgoi_k']:.0f}",
                    'Vol (K)': f"{s['vol_k']:,.0f}",
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['Instrument', 'Strike', 'Strength', 'Vol ✓', 'OI (L)', 'ΔOI (K)', 'Vol (K)']
        )

    def _style_strike_table(df, is_capping):
        def _row_style(row):
            styles = [''] * len(row)
            s_idx  = df.columns.get_loc('Strength')
            v_idx  = df.columns.get_loc('Vol ✓')
            strength = str(row.get('Strength', ''))
            vol_conf = str(row.get('Vol ✓', ''))
            if 'High Conviction' in strength:
                base_bg = '#ff000050' if is_capping else '#00ff0050'
                styles[s_idx] = f'background-color:{base_bg};font-weight:bold'
            elif 'Strong' in strength:
                base_bg = '#ff444430' if is_capping else '#00ff8830'
                styles[s_idx] = f'background-color:{base_bg};font-weight:bold'
            if '🔥' in vol_conf:
                styles[v_idx] = 'background-color:#FF8C0050;color:#FF8C00;font-weight:bold'
            return styles
        return df.style.apply(_row_style, axis=1)

    with _cap_tab:
        _cap_df = _build_strike_table(_all_instruments, 'capping')
        st.dataframe(_style_strike_table(_cap_df, is_capping=True), use_container_width=True, hide_index=True)
        st.caption("High Conviction = OI > median×1.2 AND Volume > median×1.2 | 🔥 = Volume confirmed active writing above price")

    with _sup_tab:
        _sup_df = _build_strike_table(_all_instruments, 'support')
        st.dataframe(_style_strike_table(_sup_df, is_capping=False), use_container_width=True, hide_index=True)
        st.caption("High Conviction = OI > median×1.2 AND Volume > median×1.2 | 🔥 = Volume confirmed active writing below price")

    # ── Deep Analysis per Instrument (Market Bias, Resistance, Support, Classification) ──
    st.markdown("### 🔍 Option Chain Deep Analysis — ATM ±5 Strikes")

    _deep_tab_labels = ['NIFTY 50'] + [cfg['name'] for cfg in INSTRUMENT_CONFIGS.values()]
    _deep_tabs = st.tabs(_deep_tab_labels)

    def _render_deep_analysis(tab, inst_name, deep, underlying, oi_trend=None):
        with tab:
            if deep is None:
                st.warning(f"Deep analysis unavailable for {inst_name}")
                return
            if 'error' in deep:
                st.error(f"Error: {deep.get('error', '')[:60]}")
                return

            bias = deep.get('market_bias', 'N/A')
            conf = deep.get('confidence', 0)
            signals = deep.get('bias_signals', [])
            bias_color = '#00ff88' if 'Bullish' in bias else '#ff4444' if 'Bearish' in bias else '#FFD700'

            # Bias header
            bc1, bc2 = st.columns([2, 1])
            with bc1:
                st.markdown(
                    f"<span style='font-size:1.1em;font-weight:bold;color:{bias_color}'>Market Bias: {bias}</span>",
                    unsafe_allow_html=True
                )
                for sig in signals:
                    st.markdown(f"• {sig}")
            with bc2:
                st.metric("Confidence", f"{conf}%")

            # ── PCR-based S/R Table — always visible, shown before Top 3 ──
            st.markdown("📐 **PCR-based Support / Resistance Levels**")
            st.caption("PCR ≤0.7 → Resistance | 0.8–1.7 → Neutral | ≥1.8 → Support | Offset applied to reference strike")
            _pcr_rows = []
            _pcr_sr_snapshot = []
            _pcr_strike_map = {}
            if oi_trend:
                for _lbl in ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']:
                    td = oi_trend.get(_lbl)
                    if td:
                        _pcr_strike_map[_lbl] = {'strike': td['strike'], 'pcr': td.get('pcr_strike', 1.0)}
            if not _pcr_strike_map:
                # Fallback: compute PCR from analysis_df (CE_OI/PE_OI always available)
                # Also try df_summary (has PCR column) for NIFTY path
                _dfs = None
                if deep:
                    _dfs = deep.get('df_summary')
                    if _dfs is None or 'PCR' not in (_dfs.columns if _dfs is not None else []):
                        _adf = deep.get('analysis_df')
                        if _adf is not None and 'CE_OI' in _adf.columns and 'PE_OI' in _adf.columns:
                            _dfs = _adf.copy()
                            _dfs['PCR'] = (_dfs['PE_OI'] / _dfs['CE_OI'].replace(0, 1)).round(2)
                if _dfs is not None and 'PCR' in _dfs.columns and 'Strike' in _dfs.columns:
                    try:
                        _und = underlying or 0
                        if not _und and inst_name == 'NIFTY 50':
                            _und = st.session_state.get('_last_underlying') or 0
                        _atm_s = float(_dfs.iloc[(_dfs['Strike'] - _und).abs().argsort()].iloc[0]['Strike']) if _und else float(_dfs['Strike'].median())
                        _slist = sorted(_dfs['Strike'].unique())
                        _atm_i = min(range(len(_slist)), key=lambda i: abs(_slist[i] - _atm_s))
                        for _off, _lbl in [(-2, 'ATM-2'), (-1, 'ATM-1'), (0, 'ATM'), (1, 'ATM+1'), (2, 'ATM+2')]:
                            _si = _atm_i + _off
                            if 0 <= _si < len(_slist):
                                _row = _dfs[_dfs['Strike'] == _slist[_si]]
                                if not _row.empty:
                                    _pcr_strike_map[_lbl] = {
                                        'strike': float(_slist[_si]),
                                        'pcr': float(_row['PCR'].iloc[0])
                                    }
                    except Exception:
                        pass
            for _lbl in ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']:
                _sm = _pcr_strike_map.get(_lbl)
                if not _sm:
                    continue
                pcr_val = _sm['pcr']
                sr = calculate_pcr_sr_level(pcr_val, _sm['strike'])
                _pcr_rows.append({
                    'Label': _lbl,
                    'Strike': f"₹{_sm['strike']:.0f}",
                    'PCR': f"{pcr_val:.2f}",
                    'S/R Type': sr['type'],
                    'S/R Level': f"₹{sr['level']:.0f}",
                    'Offset (pts)': f"{sr['offset']:+.0f}",
                    'Interpretation': sr['interpretation'],
                })
                _pcr_sr_snapshot.append({
                    'label': _lbl, 'strike': _sm['strike'],
                    'pcr': pcr_val, 'type': sr['type'],
                    'level': sr['level'], 'offset': sr['offset'],
                })
            if inst_name == 'NIFTY 50':
                st.session_state._pcr_sr_snapshot = _pcr_sr_snapshot
            if _pcr_rows:
                _pcr_df = pd.DataFrame(_pcr_rows)
                def _style_pcr(row):
                    styles = [''] * len(row)
                    sr_idx = _pcr_df.columns.get_loc('S/R Type')
                    lv_idx = _pcr_df.columns.get_loc('S/R Level')
                    sr_type = str(row.iloc[sr_idx])
                    if 'Resistance' in sr_type:
                        styles[sr_idx] = 'background-color:#ff444450;font-weight:bold'
                        styles[lv_idx] = 'background-color:#ff444430'
                    elif 'Support' in sr_type:
                        styles[sr_idx] = 'background-color:#00ff8850;font-weight:bold'
                        styles[lv_idx] = 'background-color:#00ff8830'
                    return styles
                st.dataframe(_pcr_df.style.apply(_style_pcr, axis=1), use_container_width=True, hide_index=True)
            else:
                st.info("PCR data not yet available — load option chain data first.")

            # Top Resistance / Support side by side
            res_col, sup_col = st.columns(2)
            with res_col:
                st.markdown("🔴 **Top 3 Resistance (CE)**")
                top_res = deep.get('top_resistance')
                if top_res is not None and not top_res.empty:
                    _res_disp = top_res.copy()
                    _res_disp['Strike'] = _res_disp['Strike'].apply(lambda x: f"₹{x:.0f}")
                    _res_disp['CE_OI'] = (_res_disp['CE_OI'] / 100000).round(1).astype(str) + 'L'
                    _res_disp['CE_ChgOI'] = (_res_disp['CE_ChgOI'] / 1000).round(0).astype(str) + 'K'
                    _res_disp = _res_disp.rename(columns={
                        'Call_Strength': 'Strength', 'Call_Activity': 'Activity',
                        'CE_OI': 'OI', 'CE_ChgOI': 'ΔOI',
                    })
                    disp_cols = [c for c in ['Strike', 'Strength', 'OI', 'ΔOI', 'Activity'] if c in _res_disp.columns]

                    def _style_res(row):
                        s = [''] * len(row)
                        if 'Strength' in _res_disp.columns:
                            si = _res_disp.columns.get_loc('Strength') if 'Strength' in disp_cols else -1
                            if si >= 0 and 'High Conviction' in str(row.iloc[si]):
                                s[si] = 'background-color:#ff000050;font-weight:bold'
                            elif si >= 0 and 'Strong' in str(row.iloc[si]):
                                s[si] = 'background-color:#ff444430;font-weight:bold'
                        return s

                    st.dataframe(
                        _res_disp[disp_cols].style.apply(_style_res, axis=1),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.info("No resistance levels")

            with sup_col:
                st.markdown("🟢 **Top 3 Support (PE)**")
                top_sup = deep.get('top_support')
                if top_sup is not None and not top_sup.empty:
                    _sup_disp = top_sup.copy()
                    _sup_disp['Strike'] = _sup_disp['Strike'].apply(lambda x: f"₹{x:.0f}")
                    _sup_disp['PE_OI'] = (_sup_disp['PE_OI'] / 100000).round(1).astype(str) + 'L'
                    _sup_disp['PE_ChgOI'] = (_sup_disp['PE_ChgOI'] / 1000).round(0).astype(str) + 'K'
                    _sup_disp = _sup_disp.rename(columns={
                        'Put_Strength': 'Strength', 'Put_Activity': 'Activity',
                        'PE_OI': 'OI', 'PE_ChgOI': 'ΔOI',
                    })
                    disp_cols_s = [c for c in ['Strike', 'Strength', 'OI', 'ΔOI', 'Activity'] if c in _sup_disp.columns]

                    def _style_sup(row):
                        s = [''] * len(row)
                        if 'Strength' in _sup_disp.columns:
                            si = _sup_disp.columns.get_loc('Strength') if 'Strength' in disp_cols_s else -1
                            if si >= 0 and 'High Conviction' in str(row.iloc[si]):
                                s[si] = 'background-color:#00ff0050;font-weight:bold'
                            elif si >= 0 and 'Strong' in str(row.iloc[si]):
                                s[si] = 'background-color:#00ff8830;font-weight:bold'
                        return s

                    st.dataframe(
                        _sup_disp[disp_cols_s].style.apply(_style_sup, axis=1),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.info("No support levels")

            # Strike-wise Call & Put Classification table
            st.markdown("📊 **Strike-wise Call & Put Classification**")
            adf = deep.get('analysis_df')
            if adf is not None and not adf.empty:
                _cls_cols = ['Strike', 'Zone', 'Call_Class', 'Call_Activity', 'CE_Vol', 'Put_Class', 'Put_Activity', 'PE_Vol']
                _cls_cols = [c for c in _cls_cols if c in adf.columns]
                _cls = adf[_cls_cols].copy()
                _cls['Strike'] = _cls['Strike'].apply(lambda x: f"₹{x:.0f}")
                if 'CE_Vol' in _cls.columns:
                    _cls['CE_Vol'] = _cls['CE_Vol'].apply(lambda x: f"{x/1000:.0f}K")
                if 'PE_Vol' in _cls.columns:
                    _cls['PE_Vol'] = _cls['PE_Vol'].apply(lambda x: f"{x/1000:.0f}K")

                _ce_vol_high = adf.get('CE_Vol_High', pd.Series([False] * len(adf))).values if 'CE_Vol_High' in adf.columns else [False] * len(adf)
                _pe_vol_high = adf.get('PE_Vol_High', pd.Series([False] * len(adf))).values if 'PE_Vol_High' in adf.columns else [False] * len(adf)

                def _style_cls(row):
                    styles = [''] * len(row)
                    call_i = _cls.columns.get_loc('Call_Class') if 'Call_Class' in _cls.columns else -1
                    put_i  = _cls.columns.get_loc('Put_Class')  if 'Put_Class'  in _cls.columns else -1
                    cevol_i = _cls.columns.get_loc('CE_Vol') if 'CE_Vol' in _cls.columns else -1
                    pevol_i = _cls.columns.get_loc('PE_Vol') if 'PE_Vol' in _cls.columns else -1
                    call_cls = str(row.iloc[call_i]) if call_i >= 0 else ''
                    put_cls  = str(row.iloc[put_i])  if put_i  >= 0 else ''
                    if 'High Conviction' in call_cls:
                        if call_i >= 0: styles[call_i] = 'background-color:#ff000060;font-weight:bold'
                        if cevol_i >= 0: styles[cevol_i] = 'background-color:#ff000060;font-weight:bold'
                    elif 'Strong' in call_cls:
                        if call_i >= 0: styles[call_i] = 'background-color:#ff444440;font-weight:bold'
                    elif 'Breakout' in call_cls:
                        if call_i >= 0: styles[call_i] = 'background-color:#FFD70040;font-weight:bold'
                    if 'High Conviction' in put_cls:
                        if put_i  >= 0: styles[put_i]  = 'background-color:#00ff0060;font-weight:bold'
                        if pevol_i >= 0: styles[pevol_i] = 'background-color:#00ff0060;font-weight:bold'
                    elif 'Strong' in put_cls:
                        if put_i  >= 0: styles[put_i]  = 'background-color:#00ff8840;font-weight:bold'
                    elif 'Breakdown' in put_cls:
                        if put_i  >= 0: styles[put_i]  = 'background-color:#FFD70040;font-weight:bold'
                    return styles

                st.dataframe(_cls.style.apply(_style_cls, axis=1), use_container_width=True, hide_index=True)
            else:
                st.info("No classification data")

            # Trapped writers + Breakout/Breakdown
            tw_col, bk_col = st.columns(2)
            with tw_col:
                trapped_ce = deep.get('trapped_call_writers')
                trapped_pe = deep.get('trapped_put_writers')
                if trapped_ce is not None and not trapped_ce.empty:
                    st.warning(f"⚠️ {len(trapped_ce)} Trapped Call Writer(s)")
                    _tc = trapped_ce.copy()
                    _tc['Strike'] = _tc['Strike'].apply(lambda x: f"₹{x:.0f}")
                    _tc['CE_OI'] = (_tc['CE_OI'] / 100000).round(1).astype(str) + 'L'
                    st.dataframe(_tc, use_container_width=True, hide_index=True)
                else:
                    st.success("No trapped call writers")
            with bk_col:
                bz = deep.get('breakout_zones')
                bd = deep.get('breakdown_zones')
                if bz is not None and not bz.empty:
                    st.success(f"🚀 Breakout Zone: ₹{bz.iloc[0]['Strike']:.0f}")
                elif bd is not None and not bd.empty:
                    st.error(f"💥 Breakdown Zone: ₹{bd.iloc[0]['Strike']:.0f}")
                else:
                    st.info("No breakout/breakdown zones")

            # OI Timeline Trend Analysis — ATM ±1 (tabular)
            if oi_trend:
                st.markdown("---")
                st.markdown("📈 **OI Timeline Trend Analysis (ATM ±1)**")
                st.caption("Derived from day's changeinOpenInterest vs previousOpenInterest | OI in Lakhs · ChgOI in K contracts")

                _act_icons = {
                    'Short Building': '📈 Short Building',
                    'Short Covering': '📉 Short Covering',
                    'Long Building':  '🟢 Long Building',
                    'Long Unwinding': '🔻 Long Unwinding',
                    'Neutral': '⚪ Neutral',
                }
                _sig_colors = {
                    'BULLISH': '#00ff8850', 'MILDLY BULLISH': '#90EE9030',
                    'BEARISH': '#ff444450', 'MILDLY BEARISH': '#ff888830',
                    'RANGE': '#FFD70030', 'VOLATILE': '#FF8C0030', 'NEUTRAL': '',
                }

                # ── OI Activity Table ──────────────────────────────────────
                _oi_rows = []
                for _lbl in ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']:
                    td = oi_trend.get(_lbl)
                    if not td:
                        continue
                    _oi_rows.append({
                        'Label': _lbl,
                        'Strike': f"₹{td['strike']:.0f}",
                        'Signal': td['signal'],
                        'CE Activity': _act_icons.get(td['ce_activity'], td['ce_activity']),
                        'CE OI Δ%': f"{td['ce_oi_pct']:+.1f}%",
                        'CE ChgOI': f"{td['ce_chgoi']/1000:+,.0f}K {'↑' if td['ce_chgoi']>0 else '↓'}",
                        'CE OI': f"{td['ce_oi']/100000:.1f}L",
                        'CE Status': td['resistance_status'],
                        'PE Activity': _act_icons.get(td['pe_activity'], td['pe_activity']),
                        'PE OI Δ%': f"{td['pe_oi_pct']:+.1f}%",
                        'PE ChgOI': f"{td['pe_chgoi']/1000:+,.0f}K {'↑' if td['pe_chgoi']>0 else '↓'}",
                        'PE OI': f"{td['pe_oi']/100000:.1f}L",
                        'PE Status': td['support_status'],
                    })
                if _oi_rows:
                    _oi_df = pd.DataFrame(_oi_rows)
                    def _style_oi(row):
                        sig = str(row.get('Signal', ''))
                        bg = _sig_colors.get(sig, '')
                        base = [f'background-color:{bg}' if bg else ''] * len(row)
                        # Colour CE/PE status cells
                        for ci, col in enumerate(_oi_df.columns):
                            if 'CE Status' == col:
                                base[ci] = 'background-color:#ff444430' if 'Building' in str(row.get(col,'')) else 'background-color:#FFD70030' if 'Break' in str(row.get(col,'')) else ''
                            elif 'PE Status' == col:
                                base[ci] = 'background-color:#00ff8830' if 'Building' in str(row.get(col,'')) else 'background-color:#FFD70030' if 'Break' in str(row.get(col,'')) else ''
                            elif col == 'Signal':
                                base[ci] = f'background-color:{_sig_colors.get(sig,"")};font-weight:bold' if sig in _sig_colors else 'font-weight:bold'
                        return base
                    st.dataframe(_oi_df.style.apply(_style_oi, axis=1), use_container_width=True, hide_index=True)

    # NIFTY tab — use existing sa_result + build OI trend from session_state oi_history
    _nifty_deep = None
    _nifty_oi_trend = {}
    try:
        _nifty_deep = sa_result
        # Build NIFTY OI trend from session_state history (same as master signal logic)
        _oi_hist = getattr(st.session_state, 'oi_history', [])
        _oi_strikes = getattr(st.session_state, 'oi_current_strikes', [])
        _chgoi_hist = getattr(st.session_state, 'chgoi_history', [])
        if len(_oi_hist) >= 3 and _oi_strikes:
            _sorted_s = sorted(_oi_strikes)
            _atm_idx  = len(_sorted_s) // 2
            for _off, _lbl in [(-2, 'ATM-2'), (-1, 'ATM-1'), (0, 'ATM'), (1, 'ATM+1'), (2, 'ATM+2')]:
                _si = _atm_idx + _off
                if 0 <= _si < len(_sorted_s):
                    _s = str(_sorted_s[_si])
                    _odf = pd.DataFrame(_oi_hist)
                    _ce_c, _pe_c = f'{_s}_CE', f'{_s}_PE'
                    _ce_ltp_c, _pe_ltp_c = f'{_s}_CE_LTP', f'{_s}_PE_LTP'
                    _td = {'strike': int(_s), 'ce_oi_pct': 0, 'pe_oi_pct': 0,
                           'ce_chgoi': 0, 'pe_chgoi': 0, 'ce_oi': 0, 'pe_oi': 0,
                           'ce_ltp': 0, 'pe_ltp': 0, 'ce_activity': 'N/A', 'pe_activity': 'N/A',
                           'resistance_status': 'N/A', 'support_status': 'N/A',
                           'signal': 'NEUTRAL', 'pcr_strike': 1.0}
                    if _ce_c in _odf.columns:
                        _f, _l = _odf[_ce_c].iloc[0], _odf[_ce_c].iloc[-1]
                        _td['ce_oi_pct'] = round((_l - _f) / _f * 100, 1) if _f > 0 else 0
                        _td['ce_oi'] = _l   # raw contracts (history stores raw ints)
                        _ce_ltp_f = _odf[_ce_ltp_c].iloc[0] if _ce_ltp_c in _odf.columns else 0
                        _ce_ltp_l = _odf[_ce_ltp_c].iloc[-1] if _ce_ltp_c in _odf.columns else 0
                        _td['ce_ltp'] = _ce_ltp_l
                        _oi_up = (_l - _f) > 0; _ltp_up = (_ce_ltp_l - _ce_ltp_f) > 0
                        _td['ce_activity'] = ('Long Building' if _oi_up and _ltp_up else
                                              'Short Building' if _oi_up else
                                              'Short Covering' if not _oi_up and _ltp_up else 'Long Unwinding')
                    if _pe_c in _odf.columns:
                        _f, _l = _odf[_pe_c].iloc[0], _odf[_pe_c].iloc[-1]
                        _td['pe_oi_pct'] = round((_l - _f) / _f * 100, 1) if _f > 0 else 0
                        _td['pe_oi'] = _l   # raw contracts
                        _pe_ltp_f = _odf[_pe_ltp_c].iloc[0] if _pe_ltp_c in _odf.columns else 0
                        _pe_ltp_l = _odf[_pe_ltp_c].iloc[-1] if _pe_ltp_c in _odf.columns else 0
                        _td['pe_ltp'] = _pe_ltp_l
                        _oi_up = (_l - _f) > 0; _ltp_up = (_pe_ltp_l - _pe_ltp_f) > 0
                        _td['pe_activity'] = ('Long Building' if _oi_up and _ltp_up else
                                              'Short Building' if _oi_up else
                                              'Short Covering' if not _oi_up and _ltp_up else 'Long Unwinding')
                    if len(_chgoi_hist) >= 3:
                        _cdf = pd.DataFrame(_chgoi_hist)
                        if _ce_c in _cdf.columns: _td['ce_chgoi'] = int(_cdf[_ce_c].iloc[-1])  # raw contracts
                        if _pe_c in _cdf.columns: _td['pe_chgoi'] = int(_cdf[_pe_c].iloc[-1])  # raw contracts
                    # Resistance / Support status
                    _ce_act, _pe_act = _td['ce_activity'], _td['pe_activity']
                    _td['resistance_status'] = ('Building Strong' if _ce_act == 'Short Building' else
                                                'Breaking' if _ce_act == 'Short Covering' else
                                                'Weakening' if _ce_act == 'Long Unwinding' else 'Building (Bulls)')
                    _td['support_status'] = ('Building Strong' if _pe_act == 'Short Building' else
                                             'Breaking' if _pe_act == 'Short Covering' else
                                             'Weakening' if _pe_act == 'Long Unwinding' else 'Building (Bears)')
                    if _pe_act == 'Short Building' and _ce_act in ('Short Covering', 'Long Unwinding'):
                        _td['signal'] = 'BULLISH'
                    elif _ce_act == 'Short Building' and _pe_act in ('Short Covering', 'Long Unwinding'):
                        _td['signal'] = 'BEARISH'
                    elif _pe_act == 'Short Building' and _ce_act == 'Short Building':
                        _td['signal'] = 'RANGE'
                    elif _ce_act == 'Short Covering' and _pe_act == 'Short Covering':
                        _td['signal'] = 'VOLATILE'
                    elif _pe_act == 'Short Building':
                        _td['signal'] = 'MILDLY BULLISH'
                    elif _ce_act == 'Short Building':
                        _td['signal'] = 'MILDLY BEARISH'
                    # Per-strike PCR from last OI history snapshot
                    try:
                        _ce_last = _odf[_ce_c].iloc[-1] if _ce_c in _odf.columns else 0
                        _pe_last = _odf[_pe_c].iloc[-1] if _pe_c in _odf.columns else 0
                        _td['pcr_strike'] = round(_pe_last / _ce_last, 2) if _ce_last > 0 else 1.0
                    except Exception:
                        pass
                    _nifty_oi_trend[_lbl] = _td
    except Exception:
        pass
    # Inject df_summary + underlying into _nifty_deep so PCR fallback works
    if _nifty_deep is not None and option_data:
        _nifty_deep = dict(_nifty_deep)
        if option_data.get('df_summary') is not None:
            _nifty_deep['df_summary'] = option_data['df_summary']
    _nifty_underlying = option_data.get('underlying') if option_data else None
    if _nifty_underlying:
        st.session_state['_last_underlying'] = _nifty_underlying
    _render_deep_analysis(_deep_tabs[0], 'NIFTY 50', _nifty_deep, _nifty_underlying, _nifty_oi_trend)

    # Other instruments
    for _ti, (inst_key, cfg) in enumerate(INSTRUMENT_CONFIGS.items()):
        _res = _mi_data.get(inst_key)
        _deep = _res.get('deep') if _res and 'error' not in _res else None
        _underlying = _res.get('underlying') if _res else None
        _oi_tr = _res.get('oi_trend') if _res and 'error' not in _res else None
        _render_deep_analysis(_deep_tabs[_ti + 1], cfg['name'], _deep, _underlying, _oi_tr)

    # === CANDLE PATTERN TIMELINE ===
    if not df.empty and len(df) > 10:
        st.markdown("---")
        st.markdown("## 🕯 Candle Pattern Timeline")
        try:
            # Timeframe selector
            _pat_tf = st.radio("Timeframe", ["5 min", "1 min"], horizontal=True, key="pat_tf_radio")
            if _pat_tf == "5 min":
                df_pat = df.set_index('datetime').resample('5min').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna().reset_index()
            else:
                df_pat = df.copy()
            # Filter to latest trading day available in chart data
            if not df_pat.empty:
                _latest_trading_day = df_pat['datetime'].dt.date.max()
                df_pat = df_pat[df_pat['datetime'].dt.date == _latest_trading_day]
            if len(df_pat) >= 3:
                pattern_rows = []
                for i in range(2, len(df_pat)):
                    # Get last 3 candles up to index i for pattern detection
                    window = df_pat.iloc[max(0, i - 4):i + 1].copy().reset_index(drop=True)
                    c = window.iloc[-1]
                    c_body = abs(c['close'] - c['open'])
                    c_range = c['high'] - c['low']
                    c_body_ratio = c_body / c_range if c_range > 0 else 0
                    c_green = c['close'] > c['open']
                    c_upper = c['high'] - max(c['close'], c['open'])
                    c_lower = min(c['close'], c['open']) - c['low']
                    p = window.iloc[-2] if len(window) >= 2 else None
                    p2 = window.iloc[-3] if len(window) >= 3 else None

                    pat = 'Normal'
                    direction = 'Neutral'

                    # Check multi-candle patterns FIRST (higher significance)
                    # 3-candle patterns
                    if p is not None and p2 is not None:
                        p_body = abs(p['close'] - p['open'])
                        p_green = p['close'] > p['open']
                        p_range = p['high'] - p['low']
                        p_body_ratio = p_body / p_range if p_range > 0 else 0
                        p2_body = abs(p2['close'] - p2['open'])
                        p2_green = p2['close'] > p2['open']
                        p2_range = p2['high'] - p2['low']
                        if not p2_green and (p2_body / p2_range > 0.5 if p2_range > 0 else False) and p_body_ratio < 0.3 and c_green and c_body_ratio > 0.5 and c['close'] > (p2['open'] + p2['close']) / 2:
                            pat, direction = 'Morning Star', 'Bullish'
                        elif p2_green and (p2_body / p2_range > 0.5 if p2_range > 0 else False) and p_body_ratio < 0.3 and not c_green and c_body_ratio > 0.5 and c['close'] < (p2['open'] + p2['close']) / 2:
                            pat, direction = 'Evening Star', 'Bearish'
                        elif p2_green and p_green and c_green and p['close'] > p2['close'] and c['close'] > p['close'] and (p2_body / p2_range > 0.5 if p2_range > 0 else False) and p_body_ratio > 0.5 and c_body_ratio > 0.5:
                            pat, direction = 'Three White Soldiers', 'Bullish'
                        elif not p2_green and not p_green and not c_green and p['close'] < p2['close'] and c['close'] < p['close'] and (p2_body / p2_range > 0.5 if p2_range > 0 else False) and p_body_ratio > 0.5 and c_body_ratio > 0.5:
                            pat, direction = 'Three Black Crows', 'Bearish'

                    # 2-candle patterns
                    if pat == 'Normal' and p is not None:
                        p_body = abs(p['close'] - p['open'])
                        p_green = p['close'] > p['open']
                        p_range = p['high'] - p['low']
                        if c_green and not p_green and c_body > p_body and c['close'] > p['open'] and c['open'] < p['close']:
                            pat, direction = 'Bullish Engulfing', 'Bullish'
                        elif not c_green and p_green and c_body > p_body and c['close'] < p['open'] and c['open'] > p['close']:
                            pat, direction = 'Bearish Engulfing', 'Bearish'
                        elif c_body < p_body * 0.6 and not p_green and c_green and min(c['open'], c['close']) > min(p['open'], p['close']) and max(c['open'], c['close']) < max(p['open'], p['close']):
                            pat, direction = 'Bullish Harami', 'Bullish'
                        elif c_body < p_body * 0.6 and p_green and not c_green and min(c['open'], c['close']) > min(p['open'], p['close']) and max(c['open'], c['close']) < max(p['open'], p['close']):
                            pat, direction = 'Bearish Harami', 'Bearish'
                        elif c_green and not p_green and c['open'] < p['low'] and c['close'] > (p['open'] + p['close']) / 2 and c['close'] < p['open']:
                            pat, direction = 'Piercing Line', 'Bullish'
                        elif not c_green and p_green and c['open'] > p['high'] and c['close'] < (p['open'] + p['close']) / 2 and c['close'] > p['open']:
                            pat, direction = 'Dark Cloud Cover', 'Bearish'
                        elif c_green and not p_green and abs(c['low'] - p['low']) / max(c_range, 0.01) < 0.05:
                            pat, direction = 'Tweezer Bottom', 'Bullish'
                        elif not c_green and p_green and abs(c['high'] - p['high']) / max(p_range, 0.01) < 0.05:
                            pat, direction = 'Tweezer Top', 'Bearish'

                    # 1-candle patterns (lowest priority)
                    if pat == 'Normal':
                        if c_lower > c_body * 2 and c_upper < c_body * 0.5 and c_body_ratio < 0.4:
                            pat, direction = 'Hammer', 'Bullish'
                        elif c_upper > c_body * 2 and c_lower < c_body * 0.5 and c_body_ratio < 0.4 and c_green:
                            pat, direction = 'Inverted Hammer', 'Bullish'
                        elif c_upper > c_body * 2 and c_lower < c_body * 0.5 and c_body_ratio < 0.4 and not c_green:
                            pat, direction = 'Shooting Star', 'Bearish'
                        elif c_body_ratio >= 0.95 and c_range > 0:
                            pat = 'Bull Marubozu' if c_green else 'Bear Marubozu'
                            direction = 'Bullish' if c_green else 'Bearish'
                        elif c_body_ratio < 0.1 and c_range > 0:
                            pat, direction = 'Doji', 'Indecision'
                        elif c_body_ratio < 0.35 and c_upper > c_body and c_lower > c_body and c_range > 0:
                            pat, direction = 'Spinning Top', 'Indecision'
                    if pat != 'Normal':
                        _tfmt = '%H:%M:%S' if _pat_tf == '1 min' else '%H:%M'
                        time_str = c['datetime'].strftime(_tfmt) if hasattr(c['datetime'], 'strftime') else str(c['datetime'])
                        pat_emoji = '🟢' if direction == 'Bullish' else '🔴' if direction == 'Bearish' else '🟡'
                        pattern_rows.append({
                            'Time': time_str,
                            'Pattern': f"{pat_emoji} {pat}",
                            'Direction': direction,
                            'Open': f"₹{c['open']:.2f}",
                            'High': f"₹{c['high']:.2f}",
                            'Low': f"₹{c['low']:.2f}",
                            'Close': f"₹{c['close']:.2f}",
                            'Body%': f"{c_body_ratio * 100:.0f}%",
                            'Volume': f"{int(c['volume']):,}" if c['volume'] else '-',
                        })

                if pattern_rows:
                    pat_timeline_df = pd.DataFrame(pattern_rows)
                    # Reverse so latest is on top
                    pat_timeline_df = pat_timeline_df.iloc[::-1].reset_index(drop=True)

                    # Summary counts
                    bull_pats = sum(1 for r in pattern_rows if r['Direction'] == 'Bullish')
                    bear_pats = sum(1 for r in pattern_rows if r['Direction'] == 'Bearish')
                    neutral_pats = sum(1 for r in pattern_rows if r['Direction'] not in ['Bullish', 'Bearish'])
                    pt_col1, pt_col2, pt_col3, pt_col4 = st.columns(4)
                    with pt_col1:
                        st.metric("Total Patterns", len(pattern_rows))
                    with pt_col2:
                        st.metric("🟢 Bullish", bull_pats)
                    with pt_col3:
                        st.metric("🔴 Bearish", bear_pats)
                    with pt_col4:
                        st.metric("🟡 Indecision", neutral_pats)

                    def _style_pattern_row(row):
                        d = row.get('Direction', '')
                        if d == 'Bullish':
                            return ['background-color:#00ff8812;color:white'] * len(row)
                        elif d == 'Bearish':
                            return ['background-color:#ff444412;color:white'] * len(row)
                        return ['background-color:#FFD70008;color:white'] * len(row)

                    st.dataframe(
                        pat_timeline_df.style.apply(_style_pattern_row, axis=1),
                        use_container_width=True, hide_index=True,
                        height=min(500, 50 + len(pat_timeline_df) * 35)
                    )
                    st.caption(f"🕯 Patterns detected from {_latest_trading_day.strftime('%d-%b-%Y')} {_pat_tf} chart | {len(pattern_rows)} patterns found")
                else:
                    st.info(f"No significant candle patterns detected yet on {_pat_tf} chart.")
            else:
                st.info(f"Waiting for {_pat_tf} candle data to build up...")
        except Exception as e:
            st.caption(f"Pattern timeline loading... ({str(e)[:50]})")

    # === SIGNAL HISTORY TABLE ===
    if option_data and option_data.get('underlying'):
        st.markdown("---")
        st.markdown("## 📋 Signal History (Today)")
        try:
            sig_hist = db.get_master_signals()
            if not sig_hist.empty:
                display_cols = {
                    'timestamp': 'Time',
                    'signal': 'Signal',
                    'trade_type': 'Trade',
                    'spot_price': 'Spot',
                    'abs_score': 'Score',
                    'strength': 'Strength',
                    'confidence': 'Conf%',
                    'candle_pattern': 'Candle',
                    'candle_direction': 'Dir',
                    'volume_label': 'Volume',
                    'location': 'Location',
                    'gex_mode': 'GEX Mode',
                    'pcr_gex_badge': 'PCR×GEX',
                    'market_bias': 'Bias',
                    'vix_direction': 'VIX',
                    'oi_trend_signal': 'OI Trend',
                    'ce_activity': 'CE Activity',
                    'pe_activity': 'PE Activity',
                    'support_status': 'Support',
                    'resistance_status': 'Resistance',
                    'vidya_trend': 'VIDYA',
                    'delta_vol_trend': 'Delta Vol',
                    'price_vs_vwap': 'vs VWAP',
                    'reasons': 'Confluence Factors',
                }
                available_cols = [c for c in display_cols.keys() if c in sig_hist.columns]
                display_df = sig_hist[available_cols].copy()
                display_df.rename(columns={c: display_cols[c] for c in available_cols}, inplace=True)
                # Format time column
                if 'Time' in display_df.columns:
                    try:
                        display_df['Time'] = pd.to_datetime(display_df['Time']).dt.strftime('%H:%M:%S')
                    except Exception:
                        pass
                if 'Spot' in display_df.columns:
                    display_df['Spot'] = display_df['Spot'].apply(lambda x: f"₹{x:.0f}" if pd.notna(x) else '')
                if 'Score' in display_df.columns:
                    display_df['Score'] = display_df['Score'].apply(lambda x: f"{x}/10" if pd.notna(x) else '')
                if 'Conf%' in display_df.columns:
                    display_df['Conf%'] = display_df['Conf%'].apply(lambda x: f"{x}%" if pd.notna(x) else '')

                def _style_signal_row(row):
                    trade = row.get('Trade', '')
                    if 'BUY' in str(trade):
                        return ['background-color:#00ff8812;color:white'] * len(row)
                    elif 'SELL' in str(trade):
                        return ['background-color:#ff444412;color:white'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    display_df.style.apply(_style_signal_row, axis=1),
                    use_container_width=True, hide_index=True,
                    height=min(400, 50 + len(display_df) * 35)
                )
                st.caption(f"📊 {len(display_df)} signals today | Stored in Supabase")

                # Expandable detail view
                with st.expander("🔍 View Full Signal Details"):
                    for idx, row in sig_hist.iterrows():
                        ts = row.get('timestamp', '')
                        try:
                            ts = pd.to_datetime(ts).strftime('%H:%M:%S')
                        except Exception:
                            pass
                        sig = row.get('signal', '')
                        trade = row.get('trade_type', '')
                        sig_clr = '#00ff88' if 'BUY' in str(trade) else '#ff4444' if 'SELL' in str(trade) else '#FFD700'
                        st.markdown(f"""
                        <div style="background:{sig_clr}15;padding:12px;border-radius:8px;border-left:4px solid {sig_clr};margin-bottom:10px;">
                            <b style="color:{sig_clr};font-size:15px;">{ts} | {sig} | {trade}</b><br>
                            <span style="color:white;">Spot: ₹{row.get('spot_price', 0):.0f} | Score: {row.get('abs_score', 0)}/10 ({row.get('strength', '')}) | Conf: {row.get('confidence', 0)}%</span><br>
                            <span style="color:#aaa;">🕯 {row.get('candle_pattern', '')} ({row.get('candle_direction', '')}) | 📊 Vol: {row.get('volume_label', '')} ({row.get('volume_ratio', 0)}x)</span><br>
                            <span style="color:#aaa;">📍 {row.get('location', '')}</span><br>
                            <span style="color:#aaa;">🟥 Res: {row.get('resistance_levels', '')} | 🟩 Sup: {row.get('support_levels', '')}</span><br>
                            <span style="color:#aaa;">🔮 GEX: Net {row.get('net_gex', 0):.1f}L | ATM {row.get('atm_gex', 0):.1f}L | {row.get('gex_mode', '')} | PCR×GEX: {row.get('pcr_gex_badge', '')}</span><br>
                            <span style="color:#aaa;">📉 VIX: {row.get('vix_value', '')} ({row.get('vix_direction', '')})</span><br>
                            <span style="color:#aaa;">📊 OI: {row.get('oi_trend_signal', '')} | CE: {row.get('ce_activity', '')} | PE: {row.get('pe_activity', '')} | Sup: {row.get('support_status', '')} | Res: {row.get('resistance_status', '')}</span><br>
                            <span style="color:#aaa;">🔮 VIDYA: {row.get('vidya_trend', '')} (Δ{row.get('vidya_delta_pct', 0):+.0f}%) | Delta Vol: {row.get('delta_vol_trend', '')} | VWAP: {row.get('price_vs_vwap', '')}</span><br>
                            <span style="color:#aaa;">📋 {row.get('reasons', '')}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No signals recorded today yet. Signals are stored every 5 minutes during market hours.")
        except Exception as e:
            st.caption(f"Signal history loading... ({str(e)[:50]})")

    if show_analytics:
        st.markdown("---")
        display_analytics_dashboard(db)

    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    st.sidebar.info(f"Last Updated: {current_time}")

if __name__ == "__main__":
    main()
