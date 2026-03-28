import streamlit as st
import pytz

IST = pytz.timezone('Asia/Kolkata')

NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"
NIFTY_SECURITY_ID = "13"
NIFTY_EXCHANGE_SEGMENT = "IDX_I"
NIFTY_SYMBOL = "NIFTY50"
CONTRACT_MULTIPLIER = 25
ATM_RANGE = 100  # ±100 from ATM strike
RISK_FREE_RATE = 0.06

DHAN_BASE_URL = "https://api.dhan.co/v2"
TELEGRAM_API_URL = "https://api.telegram.org"

MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 8, 30
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 15, 45

DEFAULT_PIVOT_SETTINGS = {'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True}

def load_secrets():
    try:
        dhan_client_id = st.secrets.get("DHAN_CLIENT_ID", "") or st.secrets.get("dhan", {}).get("client_id", "")
        dhan_access_token = st.secrets.get("DHAN_ACCESS_TOKEN", "") or st.secrets.get("dhan", {}).get("access_token", "")
        supabase_url = st.secrets.get("supabase", {}).get("url", "")
        supabase_key = st.secrets.get("supabase", {}).get("anon_key", "")
        try:
            telegram_bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "") or getattr(st.secrets, "TELEGRAM_BOT_TOKEN", "")
            telegram_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "") or getattr(st.secrets, "TELEGRAM_CHAT_ID", "")
            if isinstance(telegram_chat_id, (int, float)):
                telegram_chat_id = str(int(telegram_chat_id))
        except:
            telegram_bot_token = telegram_chat_id = ""
    except Exception:
        dhan_client_id = dhan_access_token = supabase_url = supabase_key = telegram_bot_token = telegram_chat_id = ""
    return {
        'dhan_client_id': dhan_client_id,
        'dhan_access_token': dhan_access_token,
        'supabase_url': supabase_url,
        'supabase_key': supabase_key,
        'telegram_bot_token': telegram_bot_token,
        'telegram_chat_id': telegram_chat_id,
    }
