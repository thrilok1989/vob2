import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import time
import datetime
from datetime import datetime, timedelta
import pytz
from pytz import timezone
import os
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False
from vob_indicators import calculate_dealer_gex, calculate_pcr_gex_confluence

DHAN_BASE_URL = "https://api.dhan.co/v2"
INSTRUMENT_CONFIGS = {
    'SENSEX':    {'scrip': 51,   'seg': 'IDX_I',  'lot': 10,  'strike_gap': 100, 'atm_strikes': 4, 'name': 'SENSEX'},
    'BANKNIFTY': {'scrip': 25,   'seg': 'IDX_I',  'lot': 15,  'strike_gap': 100, 'atm_strikes': 4, 'name': 'BANK NIFTY'},
    'RELIANCE':  {'scrip': 2885, 'seg': 'NSE_EQ', 'lot': 250, 'strike_gap': 20,  'atm_strikes': 4, 'name': 'RELIANCE'},
    'ICICIBANK': {'scrip': 4963, 'seg': 'NSE_EQ', 'lot': 700, 'strike_gap': 10,  'atm_strikes': 4, 'name': 'ICICI BANK'},
    'INFOSYS':   {'scrip': 1594, 'seg': 'NSE_EQ', 'lot': 400, 'strike_gap': 25,  'atm_strikes': 4, 'name': 'INFOSYS'},
}

try:
    DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "") or st.secrets.get("dhan", {}).get("client_id", "")
    DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "") or st.secrets.get("dhan", {}).get("access_token", "")
except Exception:
    DHAN_CLIENT_ID = DHAN_ACCESS_TOKEN = ""

class DhanAPI:
    def __init__(self, access_token, client_id):
        self.access_token = access_token.strip() if access_token else ""
        self.client_id = client_id.strip() if client_id else ""
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id
        }

    def _handle_response(self, response, context=""):
        """Check response; flag token expiry on 401 and return None."""
        if response.status_code == 200:
            return response.json()
        if response.status_code == 401:
            st.session_state['_dhan_token_expired'] = True
            st.error("🔑 **Dhan token expired.** Open the **Refresh Dhan Token** panel in the sidebar, paste your new access token, and click Apply.")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
        return None

    def get_intraday_data(self, security_id="13", exchange_segment="IDX_I", instrument="INDEX", interval="1", days_back=1):
        url = f"{self.base_url}/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": False,
            "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    def get_ltp_data(self, security_id="13", exchange_segment="IDX_I"):
        url = f"{self.base_url}/marketfeed/ltp"
        payload = {
            exchange_segment: [int(security_id)]
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Error fetching LTP: {str(e)}")
            return None

    def place_order(self, security_id, exchange_segment, transaction_type, quantity, order_type="MARKET", price=0, product_type="INTRADAY"):
        url = f"{self.base_url}/orders"
        payload = {
            "dhanClientId": self.client_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "validity": "DAY",
            "securityId": str(security_id),
            "quantity": quantity,
            "price": price,
            "triggerPrice": 0,
            "disclosedQuantity": 0,
            "afterMarketOrder": False,
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 401:
                st.session_state['_dhan_token_expired'] = True
                return {"error": "Token expired — refresh in sidebar"}
            if response.status_code == 200:
                return response.json()
            return {"error": f"{response.status_code} — {response.text[:200]}"}
        except Exception as e:
            return {"error": str(e)}

    def get_option_ltp(self, security_id, exchange_segment="NSE_FNO"):
        url = f"{self.base_url}/marketfeed/ltp"
        payload = {exchange_segment: [int(security_id)]}
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 401:
                st.session_state['_dhan_token_expired'] = True
                return None
            if response.status_code == 200:
                data = response.json()
                items = data.get('data', {}).get(exchange_segment, [])
                if items:
                    return float(items[0].get('last_price', 0))
            return None
        except Exception:
            return None

    def get_positions(self):
        url = f"{self.base_url}/positions"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 401:
                st.session_state['_dhan_token_expired'] = True
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def get_orders(self):
        url = f"{self.base_url}/orders"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 401:
                st.session_state['_dhan_token_expired'] = True
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def get_order_by_id(self, order_id):
        url = f"{self.base_url}/orders/{order_id}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 401:
                st.session_state['_dhan_token_expired'] = True
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

@st.cache_data(ttl=300)
def get_dhan_expiry_list_cached(underlying_scrip: int, underlying_seg: str):
    return get_dhan_expiry_list(underlying_scrip, underlying_seg)

def _dhan_post(url, payload, max_retries=4):
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    delays = [2, 4, 8, 16]
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 401:
                st.session_state['_dhan_token_expired'] = True
                st.error("🔑 **Dhan token expired.** Open **Refresh Dhan Token** in the sidebar and paste your new token.")
                return None
            if response.status_code == 429:
                if attempt < max_retries:
                    d = delays[min(attempt, 3)]
                    st.warning(f"⏳ Rate limited by Dhan API. Retrying in {d}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(d)
                    continue
                st.error("❌ Rate limit exceeded after multiple retries. Please wait a moment and refresh.")
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries and "429" in str(e):
                time.sleep(delays[min(attempt, 3)])
                continue
            st.error(f"Request error: {e}")
            return None
    return None

def get_dhan_option_chain(underlying_scrip: int, underlying_seg: str, expiry: str, max_retries: int = 4):
    return _dhan_post("https://api.dhan.co/v2/optionchain",
                      {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg, "Expiry": expiry},
                      max_retries)

def get_dhan_expiry_list(underlying_scrip: int, underlying_seg: str, max_retries: int = 4):
    return _dhan_post("https://api.dhan.co/v2/optionchain/expirylist",
                      {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg},
                      max_retries)


def validate_credentials(access_token, client_id):
    issues = []

    clean_token = access_token.strip() if access_token else ""
    clean_client_id = client_id.strip() if client_id else ""

    if not clean_token:
        issues.append("Access token is empty")
    elif len(clean_token) < 50:
        issues.append("Access token seems too short")
    elif clean_token != access_token:
        issues.append("Access token has leading/trailing whitespace")

    if not clean_client_id:
        issues.append("Client ID is empty")
    elif len(clean_client_id) < 5:
        issues.append("Client ID seems too short")
    elif clean_client_id != client_id:
        issues.append("Client ID has leading/trailing whitespace")

    if any(ord(c) < 32 or ord(c) > 126 for c in clean_token):
        issues.append("Access token contains invalid characters")

    if any(ord(c) < 32 or ord(c) > 126 for c in clean_client_id):
        issues.append("Client ID contains invalid characters")

    return clean_token, clean_client_id, issues

def get_user_id():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]
    return st.session_state.user_id


def get_instrument_capping_analysis(config):
    """Fetch option chain for one instrument and return compact capping/support/OI/volume summary."""
    try:
        expiry_data = get_dhan_expiry_list(config['scrip'], config['seg'])
        if not expiry_data or 'data' not in expiry_data or not expiry_data['data']:
            return None
        expiry = expiry_data['data'][0]

        oc = get_dhan_option_chain(config['scrip'], config['seg'], expiry)
        if not oc or 'data' not in oc:
            return None

        data = oc['data']
        underlying = data['last_price']
        oc_data = data['oc']

        calls, puts = [], []
        for strike, sd in oc_data.items():
            if 'ce' in sd:
                row = sd['ce'].copy(); row['strikePrice'] = float(strike); calls.append(row)
            if 'pe' in sd:
                row = sd['pe'].copy(); row['strikePrice'] = float(strike); puts.append(row)

        if not calls or not puts:
            return None

        df = pd.merge(
            pd.DataFrame(calls), pd.DataFrame(puts),
            on='strikePrice', suffixes=('_CE', '_PE')
        ).sort_values('strikePrice')

        for old, new in {'last_price': 'lastPrice', 'oi': 'openInterest',
                         'previous_oi': 'previousOpenInterest', 'volume': 'totalTradedVolume',
                         'iv': 'impliedVolatility'}.items():
            for sfx in ('_CE', '_PE'):
                if f'{old}{sfx}' in df.columns:
                    df.rename(columns={f'{old}{sfx}': f'{new}{sfx}'}, inplace=True)

        prev_ce = df['previousOpenInterest_CE'] if 'previousOpenInterest_CE' in df.columns else df['openInterest_CE']
        prev_pe = df['previousOpenInterest_PE'] if 'previousOpenInterest_PE' in df.columns else df['openInterest_PE']
        df['changeinOpenInterest_CE'] = df['openInterest_CE'] - prev_ce
        df['changeinOpenInterest_PE'] = df['openInterest_PE'] - prev_pe

        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        atm_range = config['atm_strikes'] * config['strike_gap']
        near = df[abs(df['strikePrice'] - atm_strike) <= atm_range].copy()
        if near.empty:
            return None

        # Thresholds
        median_ce_oi  = near['openInterest_CE'].median() or 1
        median_pe_oi  = near['openInterest_PE'].median() or 1
        median_ce_vol = (near['totalTradedVolume_CE'].median() if 'totalTradedVolume_CE' in near.columns else 0) or 1
        median_pe_vol = (near['totalTradedVolume_PE'].median() if 'totalTradedVolume_PE' in near.columns else 0) or 1

        # PCR and net ΔOI
        total_ce_oi    = near['openInterest_CE'].sum()
        total_pe_oi    = near['openInterest_PE'].sum()
        total_ce_chgoi = near['changeinOpenInterest_CE'].sum()
        total_pe_chgoi = near['changeinOpenInterest_PE'].sum()
        total_ce_vol   = near['totalTradedVolume_CE'].sum() if 'totalTradedVolume_CE' in near.columns else 0
        total_pe_vol   = near['totalTradedVolume_PE'].sum() if 'totalTradedVolume_PE' in near.columns else 0
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

        capping_strikes, support_strikes = [], []
        for _, row in near.iterrows():
            strike   = row['strikePrice']
            ce_oi    = row.get('openInterest_CE', 0) or 0
            pe_oi    = row.get('openInterest_PE', 0) or 0
            ce_chgoi = row.get('changeinOpenInterest_CE', 0) or 0
            pe_chgoi = row.get('changeinOpenInterest_PE', 0) or 0
            ce_vol   = row.get('totalTradedVolume_CE', 0) or 0
            pe_vol   = row.get('totalTradedVolume_PE', 0) or 0

            # Call capping
            if ce_oi > median_ce_oi * 1.2 and ce_chgoi > 0 and underlying < strike:
                ce_high_vol = ce_vol > median_ce_vol * 1.2
                capping_strikes.append({
                    'strike': strike, 'oi_l': ce_oi / 100000, 'chgoi_k': ce_chgoi / 1000,
                    'vol_k': ce_vol / 1000, 'vol_confirmed': ce_high_vol,
                    'strength': 'High Conviction' if ce_high_vol else 'Strong',
                })

            # Put support
            if pe_oi > median_pe_oi * 1.2 and pe_chgoi > 0 and underlying > strike:
                pe_high_vol = pe_vol > median_pe_vol * 1.2
                support_strikes.append({
                    'strike': strike, 'oi_l': pe_oi / 100000, 'chgoi_k': pe_chgoi / 1000,
                    'vol_k': pe_vol / 1000, 'vol_confirmed': pe_high_vol,
                    'strength': 'High Conviction' if pe_high_vol else 'Strong',
                })

        capping_strikes.sort(key=lambda x: x['oi_l'], reverse=True)
        support_strikes.sort(key=lambda x: x['oi_l'], reverse=True)

        # --- OI Timeline Trend (ATM ±1 strikes, snapshot-based) ---
        oi_trend_data = {}
        try:
            for offset, label in [(-2*config['strike_gap'], 'ATM-2'), (-config['strike_gap'], 'ATM-1'), (0, 'ATM'), (config['strike_gap'], 'ATM+1'), (2*config['strike_gap'], 'ATM+2')]:
                target = atm_strike + offset
                closest = min(df['strikePrice'], key=lambda x: abs(x - target))
                row = df[df['strikePrice'] == closest].iloc[0]
                ce_chgoi   = row.get('changeinOpenInterest_CE', 0) or 0
                pe_chgoi   = row.get('changeinOpenInterest_PE', 0) or 0
                ce_prev_oi = row.get('previousOpenInterest_CE', 0) or 1
                pe_prev_oi = row.get('previousOpenInterest_PE', 0) or 1
                ce_oi      = row.get('openInterest_CE', 0) or 0
                pe_oi      = row.get('openInterest_PE', 0) or 0
                ce_ltp     = row.get('lastPrice_CE', 0) or 0
                pe_ltp     = row.get('lastPrice_PE', 0) or 0

                ce_oi_pct  = round(ce_chgoi / ce_prev_oi * 100, 1) if ce_prev_oi > 0 else 0
                pe_oi_pct  = round(pe_chgoi / pe_prev_oi * 100, 1) if pe_prev_oi > 0 else 0

                # Classify activity from OI direction + price-to-strike relationship
                price_below_strike = underlying < closest
                price_above_strike = underlying > closest
                ce_activity = (
                    'Short Building'  if ce_chgoi > 0 and price_below_strike else
                    'Long Building'   if ce_chgoi > 0 and not price_below_strike else
                    'Short Covering'  if ce_chgoi < 0 and not price_below_strike else
                    'Long Unwinding'  if ce_chgoi < 0 else 'Neutral'
                )
                pe_activity = (
                    'Short Building'  if pe_chgoi > 0 and price_above_strike else
                    'Long Building'   if pe_chgoi > 0 and not price_above_strike else
                    'Short Covering'  if pe_chgoi < 0 and price_below_strike else
                    'Long Unwinding'  if pe_chgoi < 0 else 'Neutral'
                )

                # Support / Resistance status
                resistance_status = (
                    'Building Strong' if ce_activity == 'Short Building' else
                    'Breaking'        if ce_activity == 'Short Covering' else
                    'Weakening'       if ce_activity == 'Long Unwinding' else
                    'Building (Bulls)' if ce_activity == 'Long Building' else 'Neutral'
                )
                support_status = (
                    'Building Strong' if pe_activity == 'Short Building' else
                    'Breaking'        if pe_activity == 'Short Covering' else
                    'Weakening'       if pe_activity == 'Long Unwinding' else
                    'Building (Bears)' if pe_activity == 'Long Building' else 'Neutral'
                )

                # Overall OI trend signal
                if pe_activity == 'Short Building' and ce_activity in ('Short Covering', 'Long Unwinding'):
                    signal = 'BULLISH'
                elif ce_activity == 'Short Building' and pe_activity in ('Short Covering', 'Long Unwinding'):
                    signal = 'BEARISH'
                elif pe_activity == 'Short Building' and ce_activity == 'Short Building':
                    signal = 'RANGE'
                elif ce_activity == 'Short Covering' and pe_activity == 'Short Covering':
                    signal = 'VOLATILE'
                elif pe_activity == 'Short Building':
                    signal = 'MILDLY BULLISH'
                elif ce_activity == 'Short Building':
                    signal = 'MILDLY BEARISH'
                else:
                    signal = 'NEUTRAL'

                pcr_strike = pe_oi / ce_oi if ce_oi > 0 else 1.0
                oi_trend_data[label] = {
                    'strike': closest, 'ce_oi_pct': ce_oi_pct, 'pe_oi_pct': pe_oi_pct,
                    'ce_chgoi': ce_chgoi, 'pe_chgoi': pe_chgoi,
                    'ce_oi': ce_oi, 'pe_oi': pe_oi, 'ce_ltp': ce_ltp, 'pe_ltp': pe_ltp,
                    'ce_activity': ce_activity, 'pe_activity': pe_activity,
                    'resistance_status': resistance_status, 'support_status': support_status,
                    'signal': signal, 'pcr_strike': pcr_strike,
                }
        except Exception:
            pass

        # --- Deep analysis (ATM ±5 strikes) via analyze_strike_activity ---
        deep_sa = None
        try:
            deep_range = 5 * config['strike_gap']
            df_deep = df[abs(df['strikePrice'] - atm_strike) <= deep_range].copy()
            df_deep = df_deep.rename(columns={'strikePrice': 'Strike'})
            df_deep['Zone'] = df_deep['Strike'].apply(
                lambda x: 'ATM' if x == atm_strike else ('ITM' if x < underlying else 'OTM')
            )
            df_deep = df_deep.sort_values('Strike', ascending=False).reset_index(drop=True)
            deep_sa = analyze_strike_activity(df_deep, underlying)
        except Exception:
            pass

        return {
            'underlying': underlying, 'expiry': expiry, 'pcr': pcr,
            'pcr_bias': 'Bullish' if pcr > 1.2 else 'Bearish' if pcr < 0.7 else 'Neutral',
            'total_ce_chgoi_l': total_ce_chgoi / 100000,
            'total_pe_chgoi_l': total_pe_chgoi / 100000,
            'total_ce_vol_k': total_ce_vol / 1000,
            'total_pe_vol_k': total_pe_vol / 1000,
            'oi_bias': 'Bullish' if total_pe_chgoi > total_ce_chgoi else 'Bearish',
            'capping': capping_strikes[:3],
            'support': support_strikes[:3],
            'deep': deep_sa,
            'oi_trend': oi_trend_data,
        }
    except Exception as e:
        return {'error': str(e)}



@st.cache_data(ttl=60, show_spinner=False)
def _fetch_yf_intraday(symbol: str, interval: str = "1m", period: str = "1d", prepost: bool = True):
    """Fetch 1-min intraday OHLC via yfinance and convert to Dhan-style dict."""
    if not _HAS_YF:
        return None
    try:
        hist = yf.Ticker(symbol).history(period=period, interval=interval, prepost=prepost)
        if hist is None or hist.empty:
            return None
        ist = pytz.timezone('Asia/Kolkata')
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize('UTC').tz_convert(ist)
        else:
            hist.index = hist.index.tz_convert(ist)
        return {
            'open': hist['Open'].tolist(),
            'high': hist['High'].tolist(),
            'low': hist['Low'].tolist(),
            'close': hist['Close'].tolist(),
            'volume': hist['Volume'].fillna(0).tolist(),
            'timestamp': [int(t.timestamp()) for t in hist.index],
        }
    except Exception:
        return None


def compute_sector_rotation():
    """Fetch NSE sector indices via yfinance, compute 10m + 1h bias and rank by performance."""
    _sectors = [
        ('AUTO',     '^CNXAUTO'),
        ('PHARMA',   '^CNXPHARMA'),
        ('FMCG',     '^CNXFMCG'),
        ('METAL',    '^CNXMETAL'),
        ('REALTY',   '^CNXREALTY'),
        ('ENERGY',   '^CNXENERGY'),
        ('PSU BANK', '^CNXPSUBANK'),
        ('INFRA',    '^CNXINFRA'),
        ('MEDIA',    '^CNXMEDIA'),
        ('IT',       '^CNXIT'),
        ('BANK',     '^NSEBANK'),
    ]
    results = []
    for sec_name, yf_sym in _sectors:
        try:
            raw = _fetch_yf_intraday(yf_sym, interval="1m", period="2d")
            if not raw or 'open' not in raw:
                continue
            df = process_candle_data(raw, "1")
            if df.empty:
                continue
            ltp = float(df.iloc[-1]['close'])
            # Day open = first candle of today
            import pytz as _ptz
            _ist = _ptz.timezone('Asia/Kolkata')
            _today = datetime.now(_ist).date()
            df_today = df[df['datetime'].dt.date == _today]
            if df_today.empty:
                df_today = df.tail(200)
            day_open = float(df_today.iloc[0]['open'])
            day_chg_pct = (ltp - day_open) / day_open * 100 if day_open else 0

            # 10m sentiment: last 10 candles
            def _sent(sub):
                if len(sub) < 2: return 'N/A'
                c0, c1 = sub.iloc[0]['close'], sub.iloc[-1]['close']
                if c1 > c0 * 1.0005: return 'Bullish'
                if c1 < c0 * 0.9995: return 'Bearish'
                return 'Neutral'
            s10 = _sent(df_today.tail(10))
            # 1h sentiment: last 60 candles
            s1h = _sent(df_today.tail(60))

            results.append({
                'name': sec_name, 'ltp': ltp,
                'day_chg_pct': day_chg_pct,
                's10': s10, 's1h': s1h,
            })
        except Exception:
            pass

    if not results:
        return None

    results.sort(key=lambda x: x['day_chg_pct'], reverse=True)
    leading  = [r for r in results if r['day_chg_pct'] > 0][:3]
    lagging  = [r for r in results if r['day_chg_pct'] < 0][-3:][::-1]
    # Rotation bias: if cyclicals (METAL/AUTO/REALTY/BANK/ENERGY) top = risk-on
    cyclicals = {'AUTO', 'METAL', 'REALTY', 'BANK', 'ENERGY', 'INFRA'}
    defensives = {'PHARMA', 'FMCG', 'IT', 'MEDIA'}
    top3_names = {r['name'] for r in leading}
    cyc_count = len(top3_names & cyclicals)
    def_count = len(top3_names & defensives)
    if cyc_count >= 2:
        rotation_bias = 'RISK-ON 🟢 (cyclicals leading → bullish for NIFTY)'
    elif def_count >= 2:
        rotation_bias = 'RISK-OFF 🔴 (defensives leading → cautious/bearish)'
    else:
        rotation_bias = 'MIXED ⚪ (no clear rotation)'

    return {
        'leading': leading,
        'lagging': lagging,
        'all': results,
        'rotation_bias': rotation_bias,
    }



def fetch_alignment_data(api):
    """Fetch candle data for SENSEX, BANKNIFTY, NIFTY IT, RELIANCE, ICICI, VIX.
    Detect candle patterns, compute 10m/1h/4h sentiment for each."""
    # Dhan security IDs + yfinance fallback symbols for instruments Dhan can't
    # resolve without monthly contract lookup (MCX/CDS futures, SENSEX on BSE).
    tickers = [
        ('NIFTY 50', '13', 'IDX_I', 'INDEX', None),
        ('SENSEX', '51', 'IDX_I', 'INDEX', None),
        ('BANKNIFTY', '25', 'IDX_I', 'INDEX', None),
        ('NIFTY IT', '30', 'IDX_I', 'INDEX', None),
        ('RELIANCE', '2885', 'NSE_EQ', 'EQUITY', None),
        ('ICICIBANK', '4963', 'NSE_EQ', 'EQUITY', None),
        ('INDIA VIX', None, None, None, '^INDIAVIX'),
        # Below: Dhan requires monthly contract IDs for MCX/CDS futures, use yfinance
        ('GOLD', None, None, None, 'GC=F'),
        ('CRUDE OIL', None, None, None, 'CL=F'),
        ('USD/INR', None, None, None, 'INR=X'),
        ('S&P 500', None, None, None, 'ES=F'),       # E-mini S&P futures (24h)
        ('JAPAN 225', None, None, None, 'NKD=F'),    # CME E-mini Nikkei futures (24h)
        ('HANG SENG', None, None, None, '^HSI'),     # Cash (best available on yfinance)
        ('UK 100', None, None, None, '^FTSE'),       # Cash (best available on yfinance)
    ]
    alignment = {}
    for name, sec_id, seg, inst, yf_symbol in tickers:
        try:
            # Fetch 5 days of 1-min data to support 1d and 4d trend computation
            if yf_symbol:
                data = _fetch_yf_intraday(yf_symbol, interval="1m", period="5d")
            else:
                data = api.get_intraday_data(security_id=sec_id, exchange_segment=seg, instrument=inst, interval="1", days_back=5)
            if data and 'open' in data:
                adf = process_candle_data(data, "1")
                if adf.empty:
                    alignment[name] = {'ltp': 0, 'trend': 'Unknown', 'candle_pattern': 'N/A',
                                       'candle_dir': 'N/A', 'candles': [],
                                       'sentiment_10m': 'N/A', 'sentiment_1h': 'N/A', 'sentiment_4h': 'N/A',
                                       'sentiment_1d': 'N/A', 'sentiment_4d': 'N/A'}
                    continue

                ltp = adf.iloc[-1]['close']
                # Candle pattern detection (5-min chart using today's data only)
                today_ist = datetime.now(pytz.timezone('Asia/Kolkata')).date()
                adf_today = adf[adf['datetime'].dt.date == today_ist]
                if adf_today.empty:
                    adf_today = adf
                try:
                    adf_5m = adf_today.set_index('datetime').resample('5min').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna().reset_index()
                except Exception:
                    adf_5m = adf_today
                cp = detect_candle_patterns(adf_5m, lookback=5)

                # Multi-timeframe sentiment
                def calc_sentiment(sub_df):
                    if sub_df is None or len(sub_df) < 2:
                        return 'N/A', 0
                    first_close = sub_df.iloc[0]['close']
                    last_close = sub_df.iloc[-1]['close']
                    pct = ((last_close - first_close) / first_close) * 100 if first_close > 0 else 0
                    green_count = sum(1 for _, r in sub_df.iterrows() if r['close'] > r['open'])
                    bear_count = len(sub_df) - green_count
                    if pct > 0.1 and green_count > bear_count:
                        return 'Bullish', round(pct, 2)
                    elif pct < -0.1 and bear_count > green_count:
                        return 'Bearish', round(pct, 2)
                    else:
                        return 'Neutral', round(pct, 2)

                # Last 10 minutes
                s_10m, pct_10m = calc_sentiment(adf.tail(10))
                # Last 60 minutes (1 hour)
                s_1h, pct_1h = calc_sentiment(adf.tail(60))
                # Last 240 minutes (4 hours)
                s_4h, pct_4h = calc_sentiment(adf.tail(240))
                # Last 1 day (~375 trading minutes)
                s_1d, pct_1d = calc_sentiment(adf.tail(375))
                # Last 4 days (~1500 trading minutes)
                s_4d, pct_4d = calc_sentiment(adf.tail(1500))

                # Overall trend from LTP
                prev_ltp = adf.iloc[-2]['close'] if len(adf) >= 2 else ltp
                trend = 'Bullish' if ltp > prev_ltp else 'Bearish' if ltp < prev_ltp else 'Flat'

                # % change series from day open for line chart (today only)
                day_open = adf_today.iloc[0]['open']
                pct_series_time = adf_today['datetime'].tolist()
                pct_series_vals = [((c - day_open) / day_open) * 100 if day_open > 0 else 0 for c in adf_today['close'].tolist()]

                alignment[name] = {
                    'ltp': ltp, 'trend': trend,
                    'candle_pattern': cp['pattern'], 'candle_dir': cp['direction'],
                    'candles': cp.get('candles', []),
                    'bull_count': cp.get('bull_count', 0), 'bear_count': cp.get('bear_count', 0),
                    'sentiment_10m': s_10m, 'pct_10m': pct_10m,
                    'sentiment_1h': s_1h, 'pct_1h': pct_1h,
                    'sentiment_4h': s_4h, 'pct_4h': pct_4h,
                    'sentiment_1d': s_1d, 'pct_1d': pct_1d,
                    'sentiment_4d': s_4d, 'pct_4d': pct_4d,
                    'day_high': adf_today['high'].max(), 'day_low': adf_today['low'].min(),
                    'open': adf_today.iloc[0]['open'],
                    'pct_series_time': pct_series_time,
                    'pct_series_vals': pct_series_vals,
                }
            else:
                alignment[name] = {'ltp': 0, 'trend': 'Unknown', 'candle_pattern': 'N/A',
                                   'candle_dir': 'N/A', 'candles': [],
                                   'sentiment_10m': 'N/A', 'sentiment_1h': 'N/A', 'sentiment_4h': 'N/A',
                                   'sentiment_1d': 'N/A', 'sentiment_4d': 'N/A'}
        except Exception:
            alignment[name] = {'ltp': 0, 'trend': 'Unknown', 'candle_pattern': 'N/A',
                               'candle_dir': 'N/A', 'candles': [],
                               'sentiment_10m': 'N/A', 'sentiment_1h': 'N/A', 'sentiment_4h': 'N/A',
                               'sentiment_1d': 'N/A', 'sentiment_4d': 'N/A'}
    return alignment


def fetch_vix_data(api):
    """Fetch India VIX data via yfinance (Dhan does not expose VIX intraday)."""
    try:
        if _HAS_YF:
            data = _fetch_yf_intraday('^INDIAVIX', interval='1m', period='1d')
            if data and data.get('close'):
                return {'vix': data['close'][-1]}
    except Exception:
        pass
    return {'vix': 0}

