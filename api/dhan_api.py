"""
Dhan API module – extracted from vob_minimal.py.

Provides the DhanAPI class, option-chain / expiry-list helpers,
credential validation, and a resilient POST wrapper with retry logic.
"""

import time
from datetime import datetime, timedelta

import pytz
import requests
import streamlit as st

from config import DHAN_BASE_URL

# ---------------------------------------------------------------------------
# Module-level credentials (populated at import time from Streamlit secrets)
# ---------------------------------------------------------------------------
try:
    DHAN_CLIENT_ID = (
        st.secrets.get("DHAN_CLIENT_ID", "")
        or st.secrets.get("dhan", {}).get("client_id", "")
    )
    DHAN_ACCESS_TOKEN = (
        st.secrets.get("DHAN_ACCESS_TOKEN", "")
        or st.secrets.get("dhan", {}).get("access_token", "")
    )
except Exception:
    DHAN_CLIENT_ID = DHAN_ACCESS_TOKEN = ""


# ---------------------------------------------------------------------------
# DhanAPI class
# ---------------------------------------------------------------------------
class DhanAPI:
    def __init__(self, access_token, client_id):
        self.access_token = access_token.strip() if access_token else ""
        self.client_id = client_id.strip() if client_id else ""
        self.base_url = DHAN_BASE_URL
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id
        }

    def get_intraday_data(self, security_id="13", exchange_segment="IDX_I",
                          instrument="INDEX", interval="1", days_back=1):
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
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
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
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"LTP API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching LTP: {str(e)}")
            return None


# ---------------------------------------------------------------------------
# Low-level POST helper with retry / rate-limit handling
# ---------------------------------------------------------------------------
def _dhan_post(url, payload, max_retries=4):
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json',
    }
    delays = [2, 4, 8, 16]
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 429:
                if attempt < max_retries:
                    d = delays[min(attempt, 3)]
                    st.warning(f"\u23f3 Rate limited by Dhan API. Retrying in {d}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(d)
                    continue
                st.error("\u274c Rate limit exceeded after multiple retries. Please wait a moment and refresh.")
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


# ---------------------------------------------------------------------------
# Option-chain & expiry-list helpers
# ---------------------------------------------------------------------------
def get_dhan_option_chain(underlying_scrip: int, underlying_seg: str,
                          expiry: str, max_retries: int = 4):
    return _dhan_post(
        f"{DHAN_BASE_URL}/optionchain",
        {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg, "Expiry": expiry},
        max_retries,
    )


def get_dhan_expiry_list(underlying_scrip: int, underlying_seg: str,
                         max_retries: int = 4):
    return _dhan_post(
        f"{DHAN_BASE_URL}/optionchain/expirylist",
        {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg},
        max_retries,
    )


@st.cache_data(ttl=300)
def get_dhan_expiry_list_cached(underlying_scrip: int, underlying_seg: str):
    return get_dhan_expiry_list(underlying_scrip, underlying_seg)


# ---------------------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------------------
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
