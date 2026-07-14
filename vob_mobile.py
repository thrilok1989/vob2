"""📱 VOB Mobile — phone-friendly view of vob_minimal.py.

Two ways to use it — no separate backend process is required for the first:

1. **Embedded (recommended, works on Streamlit Cloud)** — vob_minimal.py
   imports this module. Open the MAIN app with `?view=mobile` on the URL
   (e.g. https://your-app.streamlit.app/?view=mobile) and the same session
   runs the full engine, hides the desktop dashboard, and renders the mobile
   tabs from data computed live in that very session.

2. **Standalone (two processes on the SAME machine)** — run this file as its
   own Streamlit app; it reads the mobile_snapshot.json that vob_minimal.py
   writes each cycle:

       streamlit run vob_minimal.py --server.port 8501   # backend, keep a tab open
       streamlit run vob_mobile.py  --server.port 8502   # open on your phone

   Both must share one disk (set VOB_MOBILE_SNAPSHOT to override the path).
   Two separate cloud deployments have separate disks — use embedded mode there.
"""

import json
import os
from datetime import datetime

import streamlit as st
import pytz
import pandas as pd
import plotly.graph_objects as go

SNAPSHOT_PATH = os.environ.get(
    'VOB_MOBILE_SNAPSHOT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mobile_snapshot.json'))

IST = pytz.timezone('Asia/Kolkata')

GREEN_BG, GREEN_BD = '#0a3d2a', '#00ff88'
RED_BG, RED_BD = '#3d0a1f', '#ff4444'
GRAY_BG, GRAY_BD = '#222a3a', '#888'
BLUE_BD = '#6aa2ff'  # neutral single-series line color

TABS = ['🏠 Home', '📈 Charts', '⛓️ Chain', '📉 Trends', '📋 Tables', '🔔 Alerts']


def inject_mobile_css():
    """Mobile-first styling: tight padding, big type, dark cards."""
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


def candle_chart(ch, sr_zones=None, height=320):
    """Phone-sized candlestick figure with the leg's VOB zones shaded
    (green = bullish/support, red = bearish/resistance) and, for the spot
    chart, dashed S/R level lines. Recessive grid, no rangeslider, native
    per-candle hover."""
    fig = go.Figure(go.Candlestick(
        x=ch['t'], open=ch['o'], high=ch['h'], low=ch['l'], close=ch['c'],
        increasing_line_color=GREEN_BD, decreasing_line_color=RED_BD,
        increasing_fillcolor=GREEN_BD, decreasing_fillcolor=RED_BD,
        line_width=1, whiskerwidth=0.4, name='',
    ))
    lo, hi = min(ch['l']), max(ch['h'])
    pad = (hi - lo) * 0.25 or 1
    for z in ch.get('vob_bull') or []:
        if z['upper'] > lo - pad and z['lower'] < hi + pad:
            fig.add_hrect(y0=z['lower'], y1=z['upper'], line_width=0,
                          fillcolor='rgba(0,255,136,0.14)')
    for z in ch.get('vob_bear') or []:
        if z['upper'] > lo - pad and z['lower'] < hi + pad:
            fig.add_hrect(y0=z['lower'], y1=z['upper'], line_width=0,
                          fillcolor='rgba(255,68,68,0.14)')
    for lv in (sr_zones or {}).get('support') or []:
        if lo - pad < lv < hi + pad:
            fig.add_hline(y=lv, line_dash='dot', line_width=1, line_color=GREEN_BD,
                          annotation_text=f"S {lv:,.0f}", annotation_font_size=10,
                          annotation_font_color=GREEN_BD)
    for lv in (sr_zones or {}).get('resistance') or []:
        if lo - pad < lv < hi + pad:
            fig.add_hline(y=lv, line_dash='dot', line_width=1, line_color=RED_BD,
                          annotation_text=f"R {lv:,.0f}", annotation_font_size=10,
                          annotation_font_color=RED_BD)
    fig.update_layout(
        height=height, margin=dict(l=0, r=4, t=8, b=0), showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8892a6', size=10),
        xaxis=dict(rangeslider_visible=False, nticks=6, showgrid=False,
                   fixedrange=True),
        yaxis=dict(side='right', gridcolor='rgba(136,146,166,0.18)',
                   fixedrange=True),
        dragmode=False, hovermode='x',
    )
    return fig


def line_chart(series, height=240):
    """Phone-sized line chart. series = [(name, t, v, color), …].
    One series → no legend (title names it); two+ → horizontal legend on top.
    Thin 2px lines, recessive grid, unified hover."""
    fig = go.Figure()
    for name, t, v, color in series:
        fig.add_trace(go.Scatter(x=t, y=v, mode='lines', name=name,
                                 line=dict(color=color, width=2)))
    fig.update_layout(
        height=height, margin=dict(l=0, r=4, t=26 if len(series) > 1 else 8, b=0),
        showlegend=len(series) > 1,
        legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0,
                    font=dict(size=11)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8892a6', size=10),
        xaxis=dict(nticks=6, showgrid=False, fixedrange=True),
        yaxis=dict(side='right', gridcolor='rgba(136,146,166,0.18)',
                   fixedrange=True),
        dragmode=False, hovermode='x unified',
    )
    return fig


def load_snapshot():
    try:
        with open(SNAPSHOT_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _age(ts_str):
    try:
        dt = IST.localize(datetime.strptime(ts_str or '', '%Y-%m-%d %H:%M:%S'))
        return (datetime.now(IST) - dt).total_seconds()
    except Exception:
        return None


# ── Tab renderers ────────────────────────────────────────────────────────────

def _render_home(snap):
    # Overall NIFTY verdict (14-leg)
    ov = snap.get('leg_overall') or {}
    if ov:
        bg, bd = _tone(ov.get('dir'))
        by = ov.get('by_speed') or {}
        minis = ""
        for k, icon, name in (('fast', '🚀', 'Fast'), ('lag', '🐢', 'Lagging'),
                              ('mis', '🌫️', 'Misguiding')):
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

    # Composite bias + action
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

    # VOB watch: premiums about to RISE / FALL
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

    # LTP board: ATM±3 CE / PE
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

    # Spot S/R zones + OI walls
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


def _render_charts(snap):
    charts = snap.get('charts') or {}
    if not charts:
        st.caption("No chart data yet — appears once the engine completes a data cycle.")
        return
    vw = snap.get('vob_watch') or {}
    _rise_tags = {r['tag'] for r in (vw.get('rise') or [])}
    _fall_tags = {r['tag'] for r in (vw.get('fall') or [])}
    tags = list(charts.keys())

    def _label(t):
        if t in _rise_tags:
            return f"{t} ⬆️ at bullish VOB"
        if t in _fall_tags:
            return f"{t} ⬇️ at bearish VOB"
        return t

    sel = st.selectbox("Chart", tags, index=0, key='vob_chart_sel',
                       format_func=_label, label_visibility='collapsed')
    ch = charts.get(sel)
    if ch and ch.get('c'):
        _chg = ch['c'][-1] - ch['o'][0]
        _cc = GREEN_BD if _chg >= 0 else RED_BD
        st.markdown(
            f"<div class='vob-sub' style='margin-bottom:2px;'>{sel} · "
            f"last <b style='color:inherit;'>₹{ch['c'][-1]:,.2f}</b> · "
            f"<span style='color:{_cc};'>{_chg:+,.2f}</span> over shown bars · "
            f"🟩 bullish VOB (support) · 🟥 bearish VOB (resistance)</div>",
            unsafe_allow_html=True)
        st.plotly_chart(
            candle_chart(ch, sr_zones=(snap.get('sr_zones') if sel == 'NIFTY Spot' else None)),
            use_container_width=True,
            config={'displayModeBar': False})

    # 🔍 Tap-to-popup: one button per chart opens a big modal dialog.
    if hasattr(st, 'dialog'):
        st.caption("🔍 Tap a leg to open its LTP chart as a popup:")

        def _short(t):
            if t == 'NIFTY Spot':
                return '📊 SPOT'
            parts = t.split()
            side_strike = ' '.join(parts[-2:]) if len(parts) >= 2 else t
            mark = '⬆️' if t in _rise_tags else ('⬇️' if t in _fall_tags else '')
            return f"{mark}{side_strike}"

        _cols = st.columns(3)
        for _i, _t in enumerate(tags):
            with _cols[_i % 3]:
                if st.button(_short(_t), key=f"pop_{_t}", use_container_width=True):
                    st.session_state['_popup_leg'] = _t
                    st.rerun()

        _open = st.session_state.get('_popup_leg')
        if _open and _open in charts:
            try:
                _dlg_deco = st.dialog(f"📈 {_label(_open)}", width="large")
            except TypeError:
                _dlg_deco = st.dialog(f"📈 {_label(_open)}")

            @_dlg_deco
            def _chart_popup():
                chp = charts[_open]
                _pchg = chp['c'][-1] - chp['o'][0]
                _pcc = GREEN_BD if _pchg >= 0 else RED_BD
                st.markdown(
                    f"<div class='vob-sub'>last <b style='color:inherit;'>₹{chp['c'][-1]:,.2f}</b> · "
                    f"<span style='color:{_pcc};'>{_pchg:+,.2f}</span> over shown bars · "
                    f"🟩 bullish VOB · 🟥 bearish VOB</div>",
                    unsafe_allow_html=True)
                st.plotly_chart(
                    candle_chart(chp,
                                 sr_zones=(snap.get('sr_zones') if _open == 'NIFTY Spot' else None),
                                 height=480),
                    use_container_width=True,
                    config={'displayModeBar': False})
                st.caption("Auto-refresh is paused while this chart is open.")
                if st.button("✖ Close chart", key="pop_close", use_container_width=True):
                    st.session_state.pop('_popup_leg', None)
                    st.rerun()

            _chart_popup()


def _render_chain(snap):
    oi = snap.get('oi') or {}
    if oi:
        cw, pw = oi.get('call_wall') or {}, oi.get('put_wall') or {}
        card(f"🧱 <b>OI walls</b>"
             f"<div class='vob-row'><span style='color:{RED_BD};'>CALL wall (resistance)</span>"
             f"<span>₹{cw.get('strike', 0):,.0f} · {cw.get('oi', 0)/100000:.1f}L</span></div>"
             f"<div class='vob-row'><span style='color:{GREEN_BD};'>PUT wall (support)</span>"
             f"<span>₹{pw.get('strike', 0):,.0f} · {pw.get('oi', 0)/100000:.1f}L</span></div>"
             f"<div class='vob-row'><span>PCR (total PE/CE OI)</span>"
             f"<span><b>{oi.get('pcr', '—')}</b></span></div>")
    chain = snap.get('chain') or {}
    rows = chain.get('rows') or []
    if not rows:
        st.caption("Option chain table appears once the chain loads.")
        return
    st.markdown(f"#### ⛓️ Option chain — ATM ₹{chain.get('atm', 0):,.0f} ±5")
    st.caption("Swipe the table sideways for all columns. OI/ChgOI/Vol in contracts.")
    df_c = pd.DataFrame(rows)
    for _col in df_c.columns:
        if any(k in _col for k in ('OI', 'Vol', 'ChgOI')) and 'IV' not in _col:
            df_c[_col] = df_c[_col].fillna(0).astype(int)
        elif 'LTP' in _col or 'IV' in _col:
            df_c[_col] = df_c[_col].astype(float).round(2)

    def _hl(row):
        try:
            if float(row['Strike']) == float(chain.get('atm', 0)):
                return ['background-color: rgba(106,162,255,0.18)'] * len(row)
        except Exception:
            pass
        return [''] * len(row)

    try:
        st.dataframe(df_c.style.apply(_hl, axis=1), use_container_width=True,
                     hide_index=True, height=420)
    except Exception:
        st.dataframe(df_c, use_container_width=True, hide_index=True, height=420)


def _render_trends(snap):
    trends = snap.get('trends') or {}
    if not trends:
        st.caption("Trend timelines build up over the session — check back a few "
                   "minutes after market open.")
        return
    pcr = trends.get('pcr')
    if pcr and pcr.get('v'):
        st.markdown("##### 📉 PCR (avg across tracked strikes)")
        st.plotly_chart(line_chart([("PCR", pcr['t'], pcr['v'], BLUE_BD)]),
                        use_container_width=True, config={'displayModeBar': False})
    oc, op = trends.get('oi_ce'), trends.get('oi_pe')
    if oc and oc.get('v'):
        st.markdown("##### 🧱 Open Interest — CE vs PE (total, tracked strikes)")
        st.plotly_chart(line_chart([("CE OI", oc['t'], oc['v'], RED_BD),
                                    ("PE OI", op['t'], op['v'], GREEN_BD)]),
                        use_container_width=True, config={'displayModeBar': False})
        st.caption("Rising CE OI = call writing (resistance) · rising PE OI = put writing (support).")
    cc, cp = trends.get('chgoi_ce'), trends.get('chgoi_pe')
    if cc and cc.get('v'):
        st.markdown("##### Δ Change in OI — CE vs PE")
        st.plotly_chart(line_chart([("CE ChgOI", cc['t'], cc['v'], RED_BD),
                                    ("PE ChgOI", cp['t'], cp['v'], GREEN_BD)]),
                        use_container_width=True, config={'displayModeBar': False})
    vc, vp = trends.get('vol_ce'), trends.get('vol_pe')
    if vc and vc.get('v'):
        st.markdown("##### 🔊 Volume — CE vs PE")
        st.plotly_chart(line_chart([("CE Vol", vc['t'], vc['v'], RED_BD),
                                    ("PE Vol", vp['t'], vp['v'], GREEN_BD)]),
                        use_container_width=True, config={'displayModeBar': False})
    iv = trends.get('iv')
    if iv and iv.get('v'):
        st.markdown("##### 🌡️ ATM IV (session)")
        st.plotly_chart(line_chart([("ATM IV", iv['t'], iv['v'], BLUE_BD)]),
                        use_container_width=True, config={'displayModeBar': False})


def _render_tables(snap):
    # Full 14-leg signal table
    leg_rows = snap.get('leg_rows') or []
    if leg_rows:
        st.markdown("#### 📋 Leg bias tables")
        st.caption("Every signal per leg — swipe the table sideways to see all columns.")
        _clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in leg_rows]
        _ce_t = [r for r in _clean if ' CE ' in f" {r.get('Leg', '')} "]
        _pe_t = [r for r in _clean if ' PE ' in f" {r.get('Leg', '')} "]
        if _ce_t:
            with st.expander(f"🟢 CALL (CE) legs — {len(_ce_t)}", expanded=True):
                st.dataframe(pd.DataFrame(_ce_t), use_container_width=True, hide_index=True)
        if _pe_t:
            with st.expander(f"🔴 PUT (PE) legs — {len(_pe_t)}", expanded=False):
                st.dataframe(pd.DataFrame(_pe_t), use_container_width=True, hide_index=True)

    # Greek absorption / capping
    ga = snap.get('greek_absorb') or {}
    if ga.get('rows'):
        _nd = ga.get('nifty', 'neutral')
        _em = '🟢' if _nd == 'bull' else ('🔴' if _nd == 'bear' else '⚪')
        st.markdown("#### 🧪 Greek Absorption — who is stopping the LTP")
        st.caption(f"{_em} Net read → NIFTY **{str(_nd).upper()}** · "
                   f"CALL CAPPING {ga.get('ce_capping', 0)} · PUT WRITING {ga.get('pe_writing', 0)}")
        with st.expander(f"Absorption rows — {len(ga['rows'])}", expanded=False):
            st.dataframe(pd.DataFrame(ga['rows']), use_container_width=True, hide_index=True)

    # Bias engines by speed
    engines = snap.get('engines') or []
    if engines:
        st.markdown("#### 📊 All bias engines")
        for cat, icon, name in (('fast', '🚀', 'Fast'), ('lag', '🐢', 'Lagging'),
                                ('mis', '🌫️', 'Misguiding')):
            rows = [e for e in engines if e.get('cat') == cat]
            if not rows:
                continue
            v = (snap.get('verdicts') or {}).get(cat) or {}
            title = f"{icon} {name} — {v.get('em', '')} {v.get('label', '')} (net {v.get('net', 0):+d})" \
                if v else f"{icon} {name}"
            with st.expander(title, expanded=(cat == 'fast')):
                for e in rows:
                    st.markdown(f"{e.get('bias', '')} **{e.get('engine', '')}** — {e.get('detail', '')}")


def _render_alerts(snap):
    alerts = snap.get('alerts') or []
    if not alerts:
        st.caption("No alerts fired yet this session.")
        return
    st.markdown("#### 🔔 Recent alerts")
    for a in reversed(alerts):
        head = (a.get('text') or '').split('\n', 1)[0]
        with st.expander(f"{a.get('ts', '')} · {head[:64]}"):
            st.text(a.get('text', ''))


def render_mobile_view(snap, embedded=False):
    """Render the tabbed mobile dashboard from a snapshot dict.

    embedded=True → called from inside vob_minimal.py with the snapshot built
    in this same session (no file / freshness diagnostics needed).
    embedded=False → standalone app reading mobile_snapshot.json from disk.
    """
    if not snap:
        st.markdown("## 📱 VOB Mobile")
        if embedded:
            st.info("⏳ First load — the engine has started its first cycle in "
                    "the background. This page refreshes itself (every 20s "
                    "during market hours, 60s when closed) and will switch to "
                    "a status card, then to live data once the option chain "
                    "loads (market hours: 8:30–15:45 IST, Mon–Fri). No action "
                    "needed — just leave it open.")
        else:
            st.warning(
                "No snapshot found yet.\n\n"
                "Start the backend first and keep one browser tab of it open:\n\n"
                "```\nstreamlit run vob_minimal.py\n```\n"
                f"It writes `{os.path.basename(SNAPSHOT_PATH)}` every cycle; "
                "this app refreshes automatically once it appears.")
            st.info(
                "**Both apps must run on the same machine / deployment** — "
                f"they share this file:\n\n`{SNAPSHOT_PATH}`\n\n"
                "Two separate cloud deployments have separate disks and cannot "
                "share it. On Streamlit Cloud, use the embedded mobile mode "
                "instead: open the MAIN app's URL with `?view=mobile`.", icon="ℹ️")
        return

    data_age = _age(snap.get('ts'))
    hb_age = _age(snap.get('hb_ts'))
    backend_alive = embedded or (hb_age is not None and hb_age <= 90)

    # Backend is up but has never completed a data cycle → explain, don't render
    if not snap.get('ts'):
        st.markdown("## 📱 VOB Mobile")
        if backend_alive:
            hb_note = "" if embedded else f" (heartbeat {hb_age:.0f}s ago)"
            st.info(f"✅ Backend is running{hb_note} — waiting for its first "
                    f"full data cycle.\n\n"
                    f"Backend status: **{snap.get('status', '—')}**\n\n"
                    "The snapshot fills up once the option chain loads and the "
                    "bias engines compute (needs market hours, 8:30–15:45 IST "
                    "Mon–Fri, and working API credentials).")
        else:
            st.warning("Backend heartbeat is old — vob_minimal.py is not "
                       "running (or has no open browser tab). Start it and "
                       "keep one tab open.")
        return

    # ── Header: spot + freshness (always visible above the tabs) ────────────
    if data_age is not None and data_age > 120:
        fresh = f"<span style='color:{RED_BD};'>⚠️ data {data_age/60:.0f} min old</span>"
    elif data_age is not None:
        fresh = f"<span style='color:{GREEN_BD};'>● live · {data_age:.0f}s ago</span>"
    else:
        fresh = ""
    card(
        f"<div class='vob-hero'>NIFTY ₹{snap.get('spot', 0):,.1f}</div>"
        f"<div class='vob-sub'>{snap.get('ts', '—')} IST &nbsp; {fresh}</div>"
    )
    if data_age is not None and data_age > 120:
        if backend_alive:
            st.info("Data hasn't updated recently — outside market hours "
                    "(8:30–15:45 IST Mon–Fri) this shows the last session's "
                    f"data. Backend status: **{snap.get('status', '—')}**.",
                    icon="🕐")
        else:
            st.error("Data is stale and the backend heartbeat is old — make "
                     "sure vob_minimal.py is running with a browser tab open.",
                     icon="⚠️")

    # ── Tab bar (selection survives auto-refresh via its session key) ───────
    try:
        tab = st.segmented_control("View", TABS, default=TABS[0],
                                   key='_mob_tab', label_visibility='collapsed')
    except Exception:
        tab = st.radio("View", TABS, horizontal=True, key='_mob_tab',
                       label_visibility='collapsed')
    tab = tab or TABS[0]

    if tab == TABS[0]:
        _render_home(snap)
    elif tab == TABS[1]:
        _render_charts(snap)
    elif tab == TABS[2]:
        _render_chain(snap)
    elif tab == TABS[3]:
        _render_trends(snap)
    elif tab == TABS[4]:
        _render_tables(snap)
    elif tab == TABS[5]:
        _render_alerts(snap)

    src = ("computed live in this session" if embedded
           else "computed by vob_minimal.py (keep it running with one tab open)")
    st.caption(f"📱 VOB Mobile — read-only companion. All data is {src}.")


def main():
    """Standalone mode: own Streamlit app reading mobile_snapshot.json."""
    from streamlit_autorefresh import st_autorefresh

    st.set_page_config(
        page_title="VOB Mobile",
        page_icon="📱",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_mobile_css()

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        refresh_s = st.slider("Auto-refresh (seconds)", 5, 60, 10, 5)
        paused = st.toggle("Pause refresh", value=False)
        st.caption(f"Snapshot file:\n`{SNAPSHOT_PATH}`")

    # Pause refresh while a popup chart is open so it doesn't dismiss itself.
    if not paused and not st.session_state.get('_popup_leg'):
        st_autorefresh(interval=refresh_s * 1000, key="vob_mobile_refresh")

    render_mobile_view(load_snapshot(), embedded=False)


if __name__ == "__main__":
    main()
