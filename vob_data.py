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
from vob_indicators import calculate_dealer_gex, calculate_pcr_gex_confluence, detect_candle_patterns, process_candle_data

DHAN_BASE_URL = "https://api.dhan.co/v2"
INSTRUMENT_CONFIGS = {
    'SENSEX':    {'scrip': 51,   'seg': 'IDX_I',  'lot': 10,  'strike_gap': 100, 'atm_strikes': 4, 'name': 'SENSEX'},
    'BANKNIFTY': {'scrip': 25,   'seg': 'IDX_I',  'lot': 15,  'strike_gap': 100, 'atm_strikes': 4, 'name': 'BANK NIFTY'},
    'RELIANCE':  {'scrip': 2885, 'seg': 'NSE_EQ', 'lot': 250, 'strike_gap': 20,  'atm_strikes': 4, 'name': 'RELIANCE'},
    'ICICIBANK': {'scrip': 4963, 'seg': 'NSE_EQ', 'lot': 700, 'strike_gap': 10,  'atm_strikes': 4, 'name': 'ICICI BANK'},
    'INFOSYS':   {'scrip': 1594, 'seg': 'NSE_EQ', 'lot': 400, 'strike_gap': 25,  'atm_strikes': 4, 'name': 'INFOSYS'},
}

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

