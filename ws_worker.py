"""Dhan WebSocket worker — streams sub-second LTP + cumulative delta to Supabase.

Runs as a separate always-on process (NOT inside Streamlit). The minimal app
reads from Supabase as a side-channel for fresher tick data than the 20s REST
polling can give.

Architecture:
    ┌──────────────────┐  ws stream   ┌──────────────┐
    │  Dhan WebSocket  │ ───────────▶ │ ws_worker.py │
    └──────────────────┘              └──────┬───────┘
                                             │ upsert every ~1.5s
                                             ▼
                                       ┌──────────────┐
                                       │  Supabase    │
                                       │  dhan_ticks  │
                                       │  dhan_sweeps │
                                       └──────┬───────┘
                                              │ read every 20s
                                              ▼
                                       ┌──────────────┐
                                       │ Streamlit app │
                                       └──────────────┘

Run:
    DHAN_CLIENT_ID=...  DHAN_ACCESS_TOKEN=...
    SUPABASE_URL=...    SUPABASE_KEY=...
    python ws_worker.py

Required Supabase tables (run db/schema.sql, which has these):
    dhan_ticks(id text PK, exchange_segment text, security_id int,
               ltp float, cum_delta float, volume float, last_trade_qty float,
               updated_at timestamptz)
    dhan_sweeps(id bigserial PK, exchange_segment text, security_id int,
                direction text, magnitude float, detail text, fired_at timestamptz)

This skeleton focuses on Ticker + Quote packets (enough for LTP + tick-rule
cum delta). The 20-level depth (`Full` packet) for true L2 sweep detection is
left as a TODO — the in-app cycle-based depth sweep in vob_minimal.py covers
that signal at ~20s granularity until this is wired up.
"""
import asyncio
import json
import os
import struct
import time
from datetime import datetime

import pytz

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets")

try:
    from supabase import create_client
except ImportError:
    create_client = None

CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "").strip()
ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = (os.environ.get("SUPABASE_KEY", "").strip()
                or os.environ.get("SUPABASE_ANON_KEY", "").strip())

if not CLIENT_ID or not ACCESS_TOKEN:
    raise SystemExit("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN env vars")

WS_URL = (f"wss://api-feed.dhan.co?version=2&token={ACCESS_TOKEN}"
          f"&clientId={CLIENT_ID}&authType=2")

# Instruments to subscribe. Default: NIFTY 50 index spot.
# Override with WATCH_INSTRUMENTS env var, format: "IDX_I:13,NSE_FNO:54321"
WATCH = [("IDX_I", 13)]
if os.environ.get("WATCH_INSTRUMENTS"):
    WATCH = [
        (p.split(":")[0].strip(), int(p.split(":")[1]))
        for p in os.environ["WATCH_INSTRUMENTS"].split(",") if ":" in p
    ]

IST = pytz.timezone("Asia/Kolkata")
FLUSH_INTERVAL_S = 1.5  # write to Supabase at most every 1.5s
SWEEP_COOLDOWN_S = 8.0  # don't re-fire same direction within this window
SWEEP_THIN_PCT = 0.50   # ≥50% top-5 thinning required
TICK_TOL = 0.05         # min price tick (₹) needed to confirm a sweep

# Dhan v2 request codes (subscription)
REQ_TICKER = 15
REQ_QUOTE = 17
REQ_FULL = 21
REQ_UNSUBSCRIBE = 16

# Dhan v2 response codes (incoming packets)
RESP_TICKER = 2
RESP_QUOTE = 4
RESP_OI = 5
RESP_PREV_CLOSE = 6
RESP_FULL = 8
RESP_DISCONNECT = 50

# Instruments to subscribe at Full (20-level depth + OI) for sweep detection.
# Format: "NSE_FNO:55321,NSE_FNO:55322" — typically ATM CE/PE strikes.
WATCH_DEPTH = []
if os.environ.get("WATCH_DEPTH"):
    WATCH_DEPTH = [
        (p.split(":")[0].strip(), int(p.split(":")[1]))
        for p in os.environ["WATCH_DEPTH"].split(",") if ":" in p
    ]

# Exchange-segment byte → string (per Dhan docs)
SEG_MAP = {
    0: "IDX_I", 1: "NSE_EQ", 2: "NSE_FNO", 3: "NSE_CURRENCY",
    4: "BSE_EQ", 5: "MCX_COMM", 7: "BSE_CURRENCY", 8: "BSE_FNO",
}

# Per-instrument running state
state = {}          # f"{seg}:{scrip}" → dict
last_flush = 0.0
sb = None
if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"[{datetime.now(IST):%H:%M:%S}] Supabase connected")
    except Exception as e:
        print(f"Supabase init failed: {e} — running in print-only mode")
else:
    print("WARN: Supabase not configured — running in print-only mode")


def _key(seg, scrip):
    return f"{seg}:{scrip}"


def _ensure_state(seg, scrip):
    k = _key(seg, scrip)
    if k not in state:
        state[k] = {
            "seg": seg, "scrip": int(scrip),
            "ltp": None, "prev_ltp": None,
            "cum_delta": 0.0, "volume": 0.0, "last_trade_qty": 0.0,
            "last_update_ts": None,
            # Sweep-detection state from Full packets
            "top5_bid_qty": None,
            "top5_ask_qty": None,
            "last_sweep_up_ts": 0.0,
            "last_sweep_dn_ts": 0.0,
        }
    return state[k]


def _decode_full_body(body: bytes):
    """Decode the body of a Full (response code 8) packet — Dhan v2.

    Best-effort layout (verify against the live feed; offsets can drift between
    Dhan API versions):
      base (42 B): LTP(f) LTQ(H) LTT(i) ATP(f) Volume(I) TTBQ(I) TTSQ(I)
                   Open(f) Close(f) High(f) Low(f)
      depth (400 B): 20 levels × (bid_qty(I) ask_qty(I) bid_ord(H) ask_ord(H)
                                  bid_price(f) ask_price(f) = 20 B)
      OI (4 B): uint32 — present in some payloads
    Returns dict with ltp + ltq + volume + 20-level depth arrays, or None.
    """
    if len(body) < 42 + 400:
        return None
    try:
        ltp, ltq, ltt, atp, vol, ttbq, ttsq, op, cl, hi, lo = \
            struct.unpack_from("<fHifIIIffff", body, 0)
        depth = []
        pos = 42
        for _ in range(20):
            b_qty, a_qty, b_ord, a_ord, b_px, a_px = \
                struct.unpack_from("<IIHHff", body, pos)
            depth.append({
                "bid_qty": int(b_qty), "ask_qty": int(a_qty),
                "bid_orders": int(b_ord), "ask_orders": int(a_ord),
                "bid_price": float(b_px), "ask_price": float(a_px),
            })
            pos += 20
        oi = None
        if len(body) >= pos + 4:
            try:
                oi = struct.unpack_from("<I", body, pos)[0]
            except struct.error:
                pass
        return {
            "ltp": float(ltp), "ltq": int(ltq), "ltt": int(ltt),
            "atp": float(atp), "volume": int(vol),
            "ttbq": int(ttbq), "ttsq": int(ttsq),
            "depth": depth, "oi": oi,
        }
    except struct.error:
        return None


def _detect_sweep_and_log(seg, scrip, depth, new_ltp):
    """Compare current top-5 bid/ask qty against previous snapshot. On a
    ≥SWEEP_THIN_PCT thinning + price tick in the matching direction, insert
    a row into dhan_sweeps. Throttled per direction per instrument."""
    if not depth or new_ltp is None:
        return
    s = _ensure_state(seg, scrip)
    top5_bid = sum(d["bid_qty"] for d in depth[:5])
    top5_ask = sum(d["ask_qty"] for d in depth[:5])
    prev_bid, prev_ask = s["top5_bid_qty"], s["top5_ask_qty"]
    prev_ltp = s["ltp"]
    s["top5_bid_qty"] = top5_bid
    s["top5_ask_qty"] = top5_ask
    if prev_bid is None or prev_ask is None or prev_ltp is None:
        return
    tick = new_ltp - prev_ltp
    now = time.time()

    # UP sweep — top-5 ask thinned ≥X% AND price ticked up
    if prev_ask > 0 and tick >= TICK_TOL:
        thinning = (prev_ask - top5_ask) / prev_ask
        if thinning >= SWEEP_THIN_PCT and (now - s["last_sweep_up_ts"]) >= SWEEP_COOLDOWN_S:
            s["last_sweep_up_ts"] = now
            magnitude = float(prev_ask - top5_ask)
            detail = (f"top5 ask {prev_ask:,} → {top5_ask:,} (−{thinning*100:.0f}%) "
                      f"· LTP {prev_ltp:.2f} → {new_ltp:.2f}")
            _insert_sweep(seg, scrip, "up", magnitude, detail)

    # DOWN sweep — top-5 bid thinned ≥X% AND price ticked down
    if prev_bid > 0 and tick <= -TICK_TOL:
        thinning = (prev_bid - top5_bid) / prev_bid
        if thinning >= SWEEP_THIN_PCT and (now - s["last_sweep_dn_ts"]) >= SWEEP_COOLDOWN_S:
            s["last_sweep_dn_ts"] = now
            magnitude = float(prev_bid - top5_bid)
            detail = (f"top5 bid {prev_bid:,} → {top5_bid:,} (−{thinning*100:.0f}%) "
                      f"· LTP {prev_ltp:.2f} → {new_ltp:.2f}")
            _insert_sweep(seg, scrip, "down", magnitude, detail)


def _insert_sweep(seg, scrip, direction, magnitude, detail):
    line = (f"[{datetime.now(IST):%H:%M:%S}] {direction.upper()} SWEEP "
            f"{seg}:{scrip} · {detail}")
    print(line)
    if not sb:
        return
    try:
        sb.table("dhan_sweeps").insert({
            "exchange_segment": seg,
            "security_id": int(scrip),
            "direction": direction,
            "magnitude": magnitude,
            "detail": detail,
            "fired_at": datetime.now(IST).isoformat(),
        }).execute()
    except Exception as e:
        print(f"dhan_sweeps insert error: {e}")


def _decode_packets(data: bytes):
    """Yield (resp_code, exch_seg_byte, sec_id, payload_dict) from a binary frame."""
    out = []
    pos, n = 0, len(data)
    while pos + 8 <= n:
        resp_code = data[pos]
        msg_len = struct.unpack_from("<H", data, pos + 1)[0]
        exch_seg = data[pos + 3]
        sec_id = struct.unpack_from("<I", data, pos + 4)[0]
        body_start = pos + 8
        body_end = pos + msg_len if msg_len > 8 else pos + 16
        if body_end > n:
            break
        body = data[body_start:body_end]
        payload = None

        if resp_code == RESP_TICKER and len(body) >= 8:
            ltp, ltt = struct.unpack_from("<fi", body, 0)
            payload = {"ltp": float(ltp), "ltt": int(ltt)}

        elif resp_code == RESP_QUOTE and len(body) >= 42:
            # Order per Dhan v2 docs: LTP(f), LTQ(H), LTT(i), ATP(f), Vol(I),
            #                         TTBQ(I), TTSQ(I), Open(f), Close(f),
            #                         High(f), Low(f) → 4+2+4+4+4+4+4+4+4+4+4 = 42
            try:
                ltp, ltq, ltt, atp, vol, ttbq, ttsq, op, cl, hi, lo = \
                    struct.unpack_from("<fHifIIIffff", body, 0)
                payload = {
                    "ltp": float(ltp), "ltq": int(ltq), "ltt": int(ltt),
                    "atp": float(atp), "volume": int(vol),
                    "ttbq": int(ttbq), "ttsq": int(ttsq),
                }
            except struct.error:
                pass

        elif resp_code == RESP_FULL:
            # Full = Quote + 20-level depth + OI; used for L2 sweep detection
            payload = _decode_full_body(body)

        elif resp_code == RESP_DISCONNECT:
            print(f"[{datetime.now(IST):%H:%M:%S}] Disconnect packet received")
            # Caller will see the connection drop on the next iter.

        out.append((resp_code, exch_seg, sec_id, payload))
        pos = body_end if body_end > pos else pos + 8
    return out


def _apply_tick(seg, scrip, payload):
    """Update running state with a new tick. Tick rule for buy/sell aggression:
    LTP > prev_LTP → buy; LTP < prev_LTP → sell; equal → ignore."""
    s = _ensure_state(seg, scrip)
    new_ltp = payload.get("ltp")
    if new_ltp is None:
        return
    prev = s["ltp"]
    s["prev_ltp"] = prev
    s["ltp"] = float(new_ltp)
    ltq = payload.get("ltq")
    if ltq and prev is not None:
        if s["ltp"] > prev:
            s["cum_delta"] += ltq
        elif s["ltp"] < prev:
            s["cum_delta"] -= ltq
        s["last_trade_qty"] = ltq
    if "volume" in payload:
        s["volume"] = payload["volume"]
    s["last_update_ts"] = time.time()


async def _flush_state_to_db():
    """Upsert all current per-instrument state to Supabase. Throttled."""
    global last_flush
    now = time.time()
    if now - last_flush < FLUSH_INTERVAL_S:
        return
    last_flush = now
    if not sb:
        return
    iso = datetime.now(IST).isoformat()
    rows = []
    for k, s in state.items():
        if s["ltp"] is None:
            continue
        rows.append({
            "id": k,
            "exchange_segment": s["seg"],
            "security_id": int(s["scrip"]),
            "ltp": float(s["ltp"]),
            "cum_delta": float(s["cum_delta"]),
            "volume": float(s["volume"]),
            "last_trade_qty": float(s["last_trade_qty"]),
            "updated_at": iso,
        })
    if not rows:
        return
    try:
        sb.table("dhan_ticks").upsert(rows).execute()
    except Exception as e:
        print(f"[{datetime.now(IST):%H:%M:%S}] Supabase upsert error: {e}")


async def _handle_message(msg):
    if isinstance(msg, str):
        # Server status / text frame
        print(f"[text] {msg[:200]}")
        return
    for resp_code, exch_seg_byte, sec_id, payload in _decode_packets(msg):
        if payload is None:
            continue
        seg = SEG_MAP.get(exch_seg_byte, f"SEG_{exch_seg_byte}")
        # Run sweep detection BEFORE updating LTP (we need prev LTP inside).
        if resp_code == RESP_FULL and "depth" in payload:
            _detect_sweep_and_log(seg, sec_id, payload["depth"], payload.get("ltp"))
        _apply_tick(seg, sec_id, payload)
    await _flush_state_to_db()


async def _subscribe(ws):
    # Quote = LTP + LTQ + volume → needed for tick-rule cum delta
    if WATCH:
        instruments = [
            {"ExchangeSegment": seg, "SecurityId": str(scrip)}
            for seg, scrip in WATCH
        ]
        await ws.send(json.dumps({
            "RequestCode": REQ_QUOTE,
            "InstrumentCount": len(instruments),
            "InstrumentList": instruments,
        }))
        print(f"[{datetime.now(IST):%H:%M:%S}] Subscribed Quote ({len(instruments)}): {instruments}")
    # Full = Quote + 20-level depth + OI → needed for L2 sweep detection
    if WATCH_DEPTH:
        instruments_full = [
            {"ExchangeSegment": seg, "SecurityId": str(scrip)}
            for seg, scrip in WATCH_DEPTH
        ]
        await ws.send(json.dumps({
            "RequestCode": REQ_FULL,
            "InstrumentCount": len(instruments_full),
            "InstrumentList": instruments_full,
        }))
        print(f"[{datetime.now(IST):%H:%M:%S}] Subscribed Full ({len(instruments_full)}): {instruments_full}")


async def run():
    backoff = 5
    while True:
        try:
            print(f"[{datetime.now(IST):%H:%M:%S}] Connecting to Dhan WS…")
            async with websockets.connect(
                WS_URL, ping_interval=20, ping_timeout=10, max_size=2**20
            ) as ws:
                backoff = 5
                await _subscribe(ws)
                async for raw in ws:
                    try:
                        await _handle_message(raw)
                    except Exception as e:
                        print(f"handle err: {e}")
        except Exception as e:
            print(f"[{datetime.now(IST):%H:%M:%S}] WS error: {e} — retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    print(f"[{datetime.now(IST):%H:%M:%S}] ws_worker started · "
          f"Quote: {len(WATCH)} · Full(depth+sweep): {len(WATCH_DEPTH)}")
    asyncio.run(run())
