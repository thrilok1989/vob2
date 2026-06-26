"""NSE public-data fetchers: FII/DII cash, FII derivatives stats, market breadth.

FII/DII cash and FII derivatives stats are end-of-day data (NSE publishes once
daily, ~6-7 PM IST). Market breadth is intraday-live during market hours.

NSE's public endpoints need browser-like headers and a primed cookie jar, and
can still fail or be rate-limited from cloud hosts. Every function therefore
returns None on any error and callers must handle that gracefully.
"""
import io
from datetime import datetime, timedelta

import pandas as pd
import requests

NSE_BASE = "https://www.nseindia.com"
NSE_ARCHIVES = "https://archives.nseindia.com"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_BASE + "/",
}


def _nse_session():
    """Return a requests session with NSE cookies primed."""
    s = requests.Session()
    s.headers.update(_BROWSER_HEADERS)
    try:
        s.get(NSE_BASE, timeout=10)
    except Exception:
        pass
    return s


def get_fii_dii_cash():
    """Latest FII & DII cash-segment buy/sell/net in Rs crore (end-of-day).

    Returns {'FII': {date, buy, sell, net}, 'DII': {...}} or None.
    """
    try:
        s = _nse_session()
        r = s.get(NSE_BASE + "/api/fiidiiTradeReact", timeout=12)
        if r.status_code != 200:
            return None
        rows = r.json()
        out = {}
        for row in rows:
            cat = str(row.get("category", "")).upper()
            entry = {
                "date": row.get("date", ""),
                "buy": float(row.get("buyValue") or 0),
                "sell": float(row.get("sellValue") or 0),
                "net": float(row.get("netValue") or 0),
            }
            if "DII" in cat:
                out["DII"] = entry
            elif "FII" in cat or "FPI" in cat:
                out["FII"] = entry
        return out or None
    except Exception:
        return None


def get_fii_derivatives_stats():
    """FII participant-wise OI in index futures & index options (end-of-day).

    Reads NSE's participant-wise OI archive, trying the last several trading
    days until a published file is found. Long/short are contract counts.

    Returns dict or None.
    """
    s = _nse_session()
    for back in range(0, 7):
        day = datetime.now() - timedelta(days=back)
        if day.weekday() >= 5:
            continue
        ddmmyyyy = day.strftime("%d%m%Y")
        url = f"{NSE_ARCHIVES}/content/nsccl/fao_participant_oi_{ddmmyyyy}.csv"
        try:
            r = s.get(url, timeout=12)
            if r.status_code != 200 or not r.text.strip():
                continue
            df = pd.read_csv(io.StringIO(r.text), skiprows=1)
            df.columns = [c.strip() for c in df.columns]
            ct_col = df.columns[0]
            fii = df[df[ct_col].astype(str).str.strip().str.upper() == "FII"]
            if fii.empty:
                continue
            row = fii.iloc[0]

            def _num(col):
                try:
                    return float(row.get(col, 0) or 0)
                except Exception:
                    return 0.0

            fi_long = _num("Future Index Long")
            fi_short = _num("Future Index Short")
            oi_call_long = _num("Option Index Call Long")
            oi_put_long = _num("Option Index Put Long")
            oi_call_short = _num("Option Index Call Short")
            oi_put_short = _num("Option Index Put Short")
            return {
                "date": day.strftime("%d-%b-%Y"),
                "fut_index_long": fi_long,
                "fut_index_short": fi_short,
                "fut_index_net": fi_long - fi_short,
                "opt_index_call_long": oi_call_long,
                "opt_index_put_long": oi_put_long,
                "opt_index_call_short": oi_call_short,
                "opt_index_put_short": oi_put_short,
            }
        except Exception:
            continue
    return None


def get_market_breadth():
    """NIFTY 50 advances/declines/unchanged (intraday-live during market hours).

    Returns {advances, declines, unchanged, ad_ratio} or None.
    """
    try:
        s = _nse_session()
        r = s.get(NSE_BASE + "/api/allIndices", timeout=12)
        if r.status_code != 200:
            return None
        for item in r.json().get("data", []):
            if str(item.get("index", "")).upper() == "NIFTY 50":
                adv = int(item.get("advances") or 0)
                dec = int(item.get("declines") or 0)
                unch = int(item.get("unchanged") or 0)
                return {
                    "advances": adv,
                    "declines": dec,
                    "unchanged": unch,
                    "ad_ratio": round(adv / dec, 2) if dec else float(adv),
                }
        return None
    except Exception:
        return None
