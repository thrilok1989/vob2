import streamlit as st
import pandas as pd
import numpy as np
import json
import pytz
from datetime import datetime, timedelta
from supabase import create_client, Client
from db.cache_manager import CacheManager

IST = pytz.timezone('Asia/Kolkata')

class SupabaseDB:
    def __init__(self, url, key):
        self.client: Client = create_client(url, key)
        self.cache = CacheManager.get_instance()
        self.is_connected = True
        self._health_check()

    def _health_check(self):
        try:
            self.client.table('candles_data').select('id').limit(1).execute()
            self.is_connected = True
        except:
            self.is_connected = False

    def sync_pending(self):
        if not self.cache.has_pending():
            return
        try:
            self._health_check()
            if not self.is_connected:
                return
            for table_name, batches in list(self.cache.get_pending().items()):
                for batch in batches:
                    try:
                        conflict = batch.get('conflict_cols')
                        records = batch['records']
                        if isinstance(records, pd.DataFrame):
                            records = records.replace({np.nan: None}).to_dict('records')
                        if conflict:
                            self.client.table(table_name).upsert(records, on_conflict=conflict).execute()
                        else:
                            self.client.table(table_name).upsert(records).execute()
                    except:
                        continue
                self.cache.clear_pending(table_name)
        except:
            pass

    def _safe_upsert(self, table_name, records, conflict_cols):
        if isinstance(records, pd.DataFrame):
            records = records.replace({np.nan: None}).to_dict('records')
        if not records:
            return
        try:
            self.client.table(table_name).upsert(records, on_conflict=conflict_cols).execute()
            self.cache.update(table_name, records)
            self.is_connected = True
        except Exception:
            self.cache.queue_write(table_name, records, conflict_cols)
            self.is_connected = False

    def _safe_query(self, table_name, query_fn, filters=None):
        try:
            result = query_fn()
            if result.data:
                df = pd.DataFrame(result.data)
                self.cache.update(table_name, result.data)
                self.is_connected = True
                return df
            return pd.DataFrame()
        except Exception:
            self.is_connected = False
            return self.cache.get(table_name, filters)

    # ── Candles ──
    def upsert_candles(self, symbol, exchange, timeframe, df):
        if df.empty:
            return
        records = []
        for _, row in df.iterrows():
            dt = row['datetime']
            records.append({
                'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe,
                'timestamp': int(row['timestamp']),
                'datetime': dt.isoformat(),
                'trading_day': dt.date().isoformat() if hasattr(dt, 'date') else str(dt)[:10],
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
                'volume': int(row['volume']),
                'data_source': 'dhan_intraday',
                'update_time': datetime.now(pytz.UTC).isoformat()
            })
        self._safe_upsert('candles_data', records, 'symbol,exchange,timeframe,timestamp')

    def get_candles(self, symbol, exchange, timeframe, hours_back=24):
        cutoff = (datetime.now(pytz.UTC) - timedelta(hours=hours_back)).isoformat()
        def query():
            return self.client.table('candles_data').select('*') \
                .eq('symbol', symbol).eq('exchange', exchange).eq('timeframe', timeframe) \
                .gte('datetime', cutoff).order('timestamp', desc=False).execute()
        df = self._safe_query('candles_data', query, {'symbol': symbol, 'timeframe': timeframe})
        if not df.empty and 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    def clear_old_candles(self, days_old=7):
        try:
            cutoff = (datetime.now(pytz.UTC) - timedelta(days=days_old)).isoformat()
            result = self.client.table('candles_data').delete().lt('datetime', cutoff).execute()
            return len(result.data) if result.data else 0
        except:
            return 0

    # ── Spot Data ──
    def upsert_spot_data(self, ltp, security_id='13', exchange_segment='IDX_I'):
        now = datetime.now(IST)
        record = {
            'timestamp': now.isoformat(),
            'trading_day': now.date().isoformat(),
            'ltp': float(ltp),
            'security_id': security_id,
            'exchange_segment': exchange_segment,
            'data_source': 'dhan_ltp',
            'update_time': datetime.now(pytz.UTC).isoformat()
        }
        self._safe_upsert('nifty_spot_data', [record], 'timestamp,security_id')

    def get_latest_spot(self, security_id='13'):
        def query():
            return self.client.table('nifty_spot_data').select('*') \
                .eq('security_id', security_id).order('timestamp', desc=True).limit(1).execute()
        df = self._safe_query('nifty_spot_data', query, {'security_id': security_id})
        if not df.empty:
            return df.iloc[0].to_dict()
        return None

    def get_spot_history(self, trading_day=None, security_id='13'):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            return self.client.table('nifty_spot_data').select('*') \
                .eq('security_id', security_id).eq('trading_day', trading_day) \
                .order('timestamp', desc=False).execute()
        return self._safe_query('nifty_spot_data', query, {'security_id': security_id, 'trading_day': trading_day})

    # ── Option Chain ──
    def upsert_option_chain(self, df, expiry, underlying_price, atm_strike):
        now = datetime.now(IST)
        records = []
        for _, row in df.iterrows():
            rec = {
                'timestamp': now.isoformat(),
                'trading_day': now.date().isoformat(),
                'expiry': expiry,
                'strike_price': float(row['strikePrice']),
                'atm_strike': float(atm_strike),
                'underlying_price': float(underlying_price),
                'data_source': 'dhan_optionchain',
                'update_time': datetime.now(pytz.UTC).isoformat()
            }
            for db_col, df_col in [
                ('last_price_ce', 'lastPrice_CE'), ('open_interest_ce', 'openInterest_CE'),
                ('previous_oi_ce', 'previousOpenInterest_CE'), ('change_in_oi_ce', 'changeinOpenInterest_CE'),
                ('volume_ce', 'totalTradedVolume_CE'), ('iv_ce', 'impliedVolatility_CE'),
                ('bid_qty_ce', 'bidQty_CE'), ('ask_qty_ce', 'askQty_CE'),
                ('last_price_pe', 'lastPrice_PE'), ('open_interest_pe', 'openInterest_PE'),
                ('previous_oi_pe', 'previousOpenInterest_PE'), ('change_in_oi_pe', 'changeinOpenInterest_PE'),
                ('volume_pe', 'totalTradedVolume_PE'), ('iv_pe', 'impliedVolatility_PE'),
                ('bid_qty_pe', 'bidQty_PE'), ('ask_qty_pe', 'askQty_PE'),
                ('delta_ce', 'Delta_CE'), ('gamma_ce', 'Gamma_CE'), ('vega_ce', 'Vega_CE'),
                ('theta_ce', 'Theta_CE'), ('rho_ce', 'Rho_CE'),
                ('delta_pe', 'Delta_PE'), ('gamma_pe', 'Gamma_PE'), ('vega_pe', 'Vega_PE'),
                ('theta_pe', 'Theta_PE'), ('rho_pe', 'Rho_PE'),
            ]:
                val = row.get(df_col)
                rec[db_col] = float(val) if pd.notna(val) else None
            records.append(rec)
        self._safe_upsert('option_chain_data', records, 'timestamp,expiry,strike_price')

    def get_option_chain(self, expiry, trading_day=None):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            return self.client.table('option_chain_data').select('*') \
                .eq('expiry', expiry).eq('trading_day', trading_day) \
                .order('timestamp', desc=True).limit(20).execute()
        return self._safe_query('option_chain_data', query, {'expiry': expiry, 'trading_day': trading_day})

    def get_option_chain_history(self, expiry, strike_price, trading_day=None):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            return self.client.table('option_chain_data').select('*') \
                .eq('expiry', expiry).eq('strike_price', strike_price) \
                .eq('trading_day', trading_day).order('timestamp', desc=False).execute()
        return self._safe_query('option_chain_data', query)

    # ── ATM Strike Data ──
    def upsert_atm_strike_data(self, df_summary, expiry, underlying_price, atm_strike):
        now = datetime.now(IST)
        records = []
        col_map = {
            'Strike': 'strike_price', 'Zone': 'zone', 'Level': 'level',
            'PCR': 'pcr', 'PCR_Signal': 'pcr_signal',
            'BiasScore': 'bias_score', 'Verdict': 'verdict',
            'LTP_Bias': 'ltp_bias', 'OI_Bias': 'oi_bias', 'ChgOI_Bias': 'chg_oi_bias',
            'Volume_Bias': 'volume_bias', 'Delta_Bias': 'delta_bias', 'Gamma_Bias': 'gamma_bias',
            'Theta_Bias': 'theta_bias', 'AskQty_Bias': 'ask_qty_bias', 'BidQty_Bias': 'bid_qty_bias',
            'AskBid_Bias': 'ask_bid_bias', 'IV_Bias': 'iv_bias', 'DVP_Bias': 'dvp_bias',
            'PressureBias': 'pressure_bias', 'DeltaExp': 'delta_exp', 'GammaExp': 'gamma_exp',
            'Operator_Entry': 'operator_entry', 'Scalp_Moment': 'scalp_moment', 'FakeReal': 'fake_real',
            'BidAskPressure': 'bid_ask_pressure',
            'Gamma_SR': 'gamma_sr', 'Delta_SR': 'delta_sr', 'Depth_SR': 'depth_sr',
            'OI_Wall': 'oi_wall', 'ChgOI_Wall': 'chg_oi_wall', 'Max_Pain': 'max_pain',
        }
        for _, row in df_summary.iterrows():
            rec = {
                'timestamp': now.isoformat(),
                'trading_day': now.date().isoformat(),
                'expiry': expiry,
                'atm_strike': float(atm_strike),
                'underlying_price': float(underlying_price),
                'data_source': 'computed',
                'update_time': datetime.now(pytz.UTC).isoformat()
            }
            for src, dst in col_map.items():
                val = row.get(src)
                if pd.notna(val) if not isinstance(val, str) else True:
                    rec[dst] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else str(val)
            records.append(rec)
        self._safe_upsert('atm_strike_data', records, 'timestamp,expiry,strike_price')

    def get_atm_strike_data(self, expiry, trading_day=None):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            return self.client.table('atm_strike_data').select('*') \
                .eq('expiry', expiry).eq('trading_day', trading_day) \
                .order('timestamp', desc=True).limit(20).execute()
        return self._safe_query('atm_strike_data', query, {'expiry': expiry})

    # ── PCR History ──
    def upsert_pcr_history(self, entries):
        now = datetime.now(IST)
        records = []
        for e in entries:
            records.append({
                'timestamp': e.get('timestamp', now).isoformat() if hasattr(e.get('timestamp', now), 'isoformat') else str(e.get('timestamp', now)),
                'trading_day': now.date().isoformat(),
                'expiry': e['expiry'],
                'strike_price': float(e['strike']),
                'atm_strike': float(e.get('atm_strike', 0)),
                'pcr_value': float(e['pcr']),
                'oi_ce': int(e.get('oi_ce', 0)),
                'oi_pe': int(e.get('oi_pe', 0)),
                'data_source': 'computed',
                'update_time': datetime.now(pytz.UTC).isoformat()
            })
        self._safe_upsert('pcr_history', records, 'timestamp,expiry,strike_price')

    def get_pcr_history(self, trading_day=None, expiry=None, strikes=None):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            q = self.client.table('pcr_history').select('*').eq('trading_day', trading_day)
            if expiry:
                q = q.eq('expiry', expiry)
            return q.order('timestamp', desc=False).execute()
        df = self._safe_query('pcr_history', query, {'trading_day': trading_day})
        if not df.empty and strikes:
            df = df[df['strike_price'].isin([float(s) for s in strikes])]
        return df

    # ── GEX History ──
    def upsert_gex_history(self, entries):
        now = datetime.now(IST)
        records = []
        for e in entries:
            records.append({
                'timestamp': e.get('timestamp', now).isoformat() if hasattr(e.get('timestamp', now), 'isoformat') else str(e.get('timestamp', now)),
                'trading_day': now.date().isoformat(),
                'expiry': e['expiry'],
                'strike_price': float(e.get('strike', 0)),
                'atm_strike': float(e.get('atm_strike', 0)),
                'total_gex': float(e.get('total_gex', 0)),
                'call_gex': float(e.get('call_gex', 0)),
                'put_gex': float(e.get('put_gex', 0)),
                'net_gex': float(e.get('net_gex', 0)),
                'gamma_flip_level': float(e['gamma_flip']) if e.get('gamma_flip') else None,
                'gex_signal': e.get('signal', ''),
                'spot_price': float(e.get('spot', 0)),
                'data_source': 'computed',
                'update_time': datetime.now(pytz.UTC).isoformat()
            })
        self._safe_upsert('gex_history', records, 'timestamp,expiry,strike_price')

    def get_gex_history(self, trading_day=None, expiry=None, strikes=None):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            q = self.client.table('gex_history').select('*').eq('trading_day', trading_day)
            if expiry:
                q = q.eq('expiry', expiry)
            return q.order('timestamp', desc=False).execute()
        df = self._safe_query('gex_history', query, {'trading_day': trading_day})
        if not df.empty and strikes:
            df = df[df['strike_price'].isin([float(s) for s in strikes])]
        return df

    # ── Detected Patterns ──
    def upsert_detected_patterns(self, patterns):
        if not patterns:
            return
        now = datetime.now(IST)
        records = []
        for p in patterns:
            rec = {
                'timestamp': p.get('timestamp', now).isoformat() if hasattr(p.get('timestamp', now), 'isoformat') else str(p.get('timestamp', now)),
                'trading_day': now.date().isoformat(),
                'pattern_type': p['pattern_type'],
                'timeframe': p.get('timeframe'),
                'direction': p.get('direction'),
                'price_level': float(p['price_level']) if p.get('price_level') else None,
                'upper_bound': float(p['upper_bound']) if p.get('upper_bound') else None,
                'lower_bound': float(p['lower_bound']) if p.get('lower_bound') else None,
                'score': float(p['score']) if p.get('score') else None,
                'metadata': json.dumps(p.get('metadata', {})),
                'data_source': 'computed',
                'update_time': datetime.now(pytz.UTC).isoformat()
            }
            records.append(rec)
        self._safe_upsert('detected_patterns', records, 'timestamp,pattern_type,price_level,timeframe')

    def get_detected_patterns(self, trading_day=None, pattern_type=None):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            q = self.client.table('detected_patterns').select('*').eq('trading_day', trading_day)
            if pattern_type:
                q = q.eq('pattern_type', pattern_type)
            return q.order('timestamp', desc=False).execute()
        return self._safe_query('detected_patterns', query, {'trading_day': trading_day})

    # ── Orderbook ──
    def upsert_orderbook(self, data, expiry):
        now = datetime.now(IST)
        records = []
        for entry in data:
            records.append({
                'timestamp': now.isoformat(),
                'trading_day': now.date().isoformat(),
                'expiry': expiry,
                'strike_price': float(entry['strike']),
                'bid_qty_ce': int(entry.get('bid_qty_ce', 0)),
                'ask_qty_ce': int(entry.get('ask_qty_ce', 0)),
                'bid_qty_pe': int(entry.get('bid_qty_pe', 0)),
                'ask_qty_pe': int(entry.get('ask_qty_pe', 0)),
                'bid_ask_pressure': float(entry.get('pressure', 0)),
                'pressure_bias': entry.get('bias', ''),
                'data_source': 'dhan_optionchain',
                'update_time': datetime.now(pytz.UTC).isoformat()
            })
        self._safe_upsert('orderbook_data', records, 'timestamp,expiry,strike_price')

    def get_orderbook(self, expiry, trading_day=None):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            return self.client.table('orderbook_data').select('*') \
                .eq('expiry', expiry).eq('trading_day', trading_day) \
                .order('timestamp', desc=True).limit(20).execute()
        return self._safe_query('orderbook_data', query, {'expiry': expiry})

    # ── User Preferences (carried over) ──
    @staticmethod
    def _default_prefs():
        return {'timeframe': '5', 'auto_refresh': True, 'days_back': 1, 'pivot_proximity': 5,
                'pivot_settings': {'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True}}

    def save_user_preferences(self, user_id, timeframe, auto_refresh, days_back, pivot_settings, pivot_proximity=5):
        try:
            data = {
                'user_id': user_id, 'timeframe': timeframe, 'auto_refresh': auto_refresh,
                'days_back': days_back, 'pivot_settings': json.dumps(pivot_settings),
                'pivot_proximity': pivot_proximity, 'updated_at': datetime.now(pytz.UTC).isoformat()
            }
            self.client.table('user_preferences').upsert(data, on_conflict="user_id").execute()
        except Exception as e:
            if "23505" not in str(e) and "duplicate key" not in str(e).lower():
                st.error(f"Error saving preferences: {str(e)}")

    def get_user_preferences(self, user_id):
        try:
            result = self.client.table('user_preferences').select('*').eq('user_id', user_id).execute()
            if result.data:
                prefs = result.data[0]
                if 'pivot_settings' in prefs and prefs['pivot_settings']:
                    prefs['pivot_settings'] = json.loads(prefs['pivot_settings'])
                else:
                    prefs['pivot_settings'] = {'show_3m': True, 'show_5m': True, 'show_10m': True, 'show_15m': True}
                return prefs
            return self._default_prefs()
        except:
            return self._default_prefs()

    # ── Market Analytics (carried over) ──
    def save_market_analytics(self, symbol, analytics_data):
        try:
            data = {'symbol': symbol, 'date': datetime.now(IST).date().isoformat()}
            data.update({k: analytics_data[k] for k in ['day_high', 'day_low', 'day_open', 'day_close', 'total_volume', 'avg_price', 'price_change', 'price_change_pct']})
            self.client.table('market_analytics').upsert(data, on_conflict="symbol,date").execute()
        except Exception as e:
            if "23505" not in str(e) and "duplicate key" not in str(e).lower():
                st.error(f"Error saving analytics: {str(e)}")

    def get_market_analytics(self, symbol, days_back=30):
        try:
            cutoff = (datetime.now().date() - timedelta(days=days_back)).isoformat()
            result = self.client.table('market_analytics').select('*') \
                .eq('symbol', symbol).gte('date', cutoff).order('date', desc=False).execute()
            return pd.DataFrame(result.data) if result.data else pd.DataFrame()
        except:
            return pd.DataFrame()

    # ── Alerts History ──
    def upsert_alert(self, alert_type, direction=None, strike_price=None, underlying_price=None,
                     signal_details=None, score=None, metadata=None, sent_via='telegram'):
        now = datetime.now(IST)
        record = {
            'timestamp': now.isoformat(),
            'trading_day': now.date().isoformat(),
            'alert_type': alert_type,
            'direction': direction,
            'strike_price': float(strike_price) if strike_price else None,
            'underlying_price': float(underlying_price) if underlying_price else None,
            'signal_details': signal_details,
            'score': float(score) if score else None,
            'metadata': json.dumps(metadata) if metadata else None,
            'sent_via': sent_via,
            'data_source': 'computed',
            'update_time': datetime.now(pytz.UTC).isoformat()
        }
        self._safe_upsert('alerts_history', [record], 'timestamp,alert_type,strike_price')

    def get_alerts_history(self, trading_day=None, alert_type=None):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            q = self.client.table('alerts_history').select('*').eq('trading_day', trading_day)
            if alert_type:
                q = q.eq('alert_type', alert_type)
            return q.order('timestamp', desc=False).execute()
        return self._safe_query('alerts_history', query, {'trading_day': trading_day})

    # ── Master Signal History ──
    def upsert_master_signal(self, signal_data):
        """Store master trading signal with all details."""
        now = datetime.now(IST)
        record = {
            'timestamp': now.isoformat(),
            'trading_day': now.date().isoformat(),
            'spot_price': float(signal_data.get('spot_price', 0)),
            'signal': signal_data.get('signal', ''),
            'trade_type': signal_data.get('trade_type', ''),
            'score': int(signal_data.get('score', 0)),
            'abs_score': int(signal_data.get('abs_score', 0)),
            'strength': signal_data.get('strength', ''),
            'confidence': int(signal_data.get('confidence', 0)),
            'candle_pattern': signal_data.get('candle_pattern', ''),
            'candle_direction': signal_data.get('candle_direction', ''),
            'volume_label': signal_data.get('volume_label', ''),
            'volume_ratio': float(signal_data.get('volume_ratio', 0)),
            'location': signal_data.get('location', ''),
            'resistance_levels': signal_data.get('resistance_levels', ''),
            'support_levels': signal_data.get('support_levels', ''),
            'net_gex': float(signal_data.get('net_gex', 0)),
            'atm_gex': float(signal_data.get('atm_gex', 0)),
            'gamma_flip': float(signal_data.get('gamma_flip', 0)) if signal_data.get('gamma_flip') else None,
            'gex_mode': signal_data.get('gex_mode', ''),
            'pcr_gex_badge': signal_data.get('pcr_gex_badge', ''),
            'market_bias': signal_data.get('market_bias', ''),
            'vix_value': float(signal_data.get('vix_value', 0)) if signal_data.get('vix_value') else None,
            'vix_direction': signal_data.get('vix_direction', ''),
            'oi_trend_signal': signal_data.get('oi_trend_signal', ''),
            'ce_activity': signal_data.get('ce_activity', ''),
            'pe_activity': signal_data.get('pe_activity', ''),
            'support_status': signal_data.get('support_status', ''),
            'resistance_status': signal_data.get('resistance_status', ''),
            'vidya_trend': signal_data.get('vidya_trend', ''),
            'vidya_delta_pct': float(signal_data.get('vidya_delta_pct', 0)),
            'delta_vol_trend': signal_data.get('delta_vol_trend', ''),
            'vwap': float(signal_data.get('vwap', 0)) if signal_data.get('vwap') else None,
            'price_vs_vwap': signal_data.get('price_vs_vwap', ''),
            'reasons': signal_data.get('reasons', ''),
            'alignment_summary': signal_data.get('alignment_summary', ''),
            'data_source': 'master_signal',
            'update_time': datetime.now(pytz.UTC).isoformat()
        }
        self._safe_upsert('master_signals', [record], 'timestamp,trading_day')

    def get_master_signals(self, trading_day=None):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            return self.client.table('master_signals').select('*') \
                .eq('trading_day', trading_day) \
                .order('timestamp', desc=True).execute()
        return self._safe_query('master_signals', query, {'trading_day': trading_day})

    # ── Max Pain History ──
    def upsert_max_pain(self, expiry, max_pain_strike, underlying_price=None):
        now = datetime.now(IST)
        distance = abs(underlying_price - max_pain_strike) if underlying_price and max_pain_strike else None
        record = {
            'timestamp': now.isoformat(),
            'trading_day': now.date().isoformat(),
            'expiry': expiry,
            'max_pain_strike': float(max_pain_strike),
            'underlying_price': float(underlying_price) if underlying_price else None,
            'distance_from_spot': float(distance) if distance else None,
            'data_source': 'computed',
            'update_time': datetime.now(pytz.UTC).isoformat()
        }
        self._safe_upsert('max_pain_history', [record], 'timestamp,expiry')

    def get_max_pain_history(self, expiry=None, trading_day=None):
        if trading_day is None:
            trading_day = datetime.now(IST).date().isoformat()
        def query():
            q = self.client.table('max_pain_history').select('*').eq('trading_day', trading_day)
            if expiry:
                q = q.eq('expiry', expiry)
            return q.order('timestamp', desc=False).execute()
        return self._safe_query('max_pain_history', query, {'trading_day': trading_day})
