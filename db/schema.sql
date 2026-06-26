-- =============================================
-- NIFTY TRADING & OPTIONS ANALYZER - FULL SQL
-- Supabase (PostgreSQL)
-- Run this in Supabase SQL Editor
-- =============================================

-- 1. CANDLES DATA (OHLCV candle data)
CREATE TABLE IF NOT EXISTS candles_data (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    datetime TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    data_source TEXT DEFAULT 'dhan_intraday',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, exchange, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON candles_data(symbol, exchange, timeframe, datetime);

-- 2. NIFTY SPOT DATA (LTP spot price history)
CREATE TABLE IF NOT EXISTS nifty_spot_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    ltp DOUBLE PRECISION NOT NULL,
    security_id TEXT DEFAULT '13',
    exchange_segment TEXT DEFAULT 'IDX_I',
    data_source TEXT DEFAULT 'dhan_ltp',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, security_id)
);
CREATE INDEX IF NOT EXISTS idx_spot_security_day ON nifty_spot_data(security_id, trading_day);

-- 3. OPTION CHAIN DATA (Full option chain snapshots)
CREATE TABLE IF NOT EXISTS option_chain_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry TEXT NOT NULL,
    strike_price DOUBLE PRECISION NOT NULL,
    atm_strike DOUBLE PRECISION,
    underlying_price DOUBLE PRECISION,
    last_price_ce DOUBLE PRECISION,
    open_interest_ce DOUBLE PRECISION,
    previous_oi_ce DOUBLE PRECISION,
    change_in_oi_ce DOUBLE PRECISION,
    volume_ce DOUBLE PRECISION,
    iv_ce DOUBLE PRECISION,
    bid_qty_ce DOUBLE PRECISION,
    ask_qty_ce DOUBLE PRECISION,
    last_price_pe DOUBLE PRECISION,
    open_interest_pe DOUBLE PRECISION,
    previous_oi_pe DOUBLE PRECISION,
    change_in_oi_pe DOUBLE PRECISION,
    volume_pe DOUBLE PRECISION,
    iv_pe DOUBLE PRECISION,
    bid_qty_pe DOUBLE PRECISION,
    ask_qty_pe DOUBLE PRECISION,
    delta_ce DOUBLE PRECISION,
    gamma_ce DOUBLE PRECISION,
    vega_ce DOUBLE PRECISION,
    theta_ce DOUBLE PRECISION,
    rho_ce DOUBLE PRECISION,
    delta_pe DOUBLE PRECISION,
    gamma_pe DOUBLE PRECISION,
    vega_pe DOUBLE PRECISION,
    theta_pe DOUBLE PRECISION,
    rho_pe DOUBLE PRECISION,
    data_source TEXT DEFAULT 'dhan_optionchain',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_option_chain_expiry_day ON option_chain_data(expiry, trading_day);

-- 4. ATM STRIKE DATA (ATM strike analysis)
CREATE TABLE IF NOT EXISTS atm_strike_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry TEXT NOT NULL,
    strike_price DOUBLE PRECISION,
    atm_strike DOUBLE PRECISION,
    underlying_price DOUBLE PRECISION,
    zone TEXT,
    level TEXT,
    pcr DOUBLE PRECISION,
    pcr_signal TEXT,
    bias_score DOUBLE PRECISION,
    verdict TEXT,
    ltp_bias TEXT,
    oi_bias TEXT,
    chg_oi_bias TEXT,
    volume_bias TEXT,
    delta_bias TEXT,
    gamma_bias TEXT,
    theta_bias TEXT,
    ask_qty_bias TEXT,
    bid_qty_bias TEXT,
    ask_bid_bias TEXT,
    iv_bias TEXT,
    dvp_bias TEXT,
    pressure_bias TEXT,
    delta_exp TEXT,
    gamma_exp TEXT,
    operator_entry TEXT,
    scalp_moment TEXT,
    fake_real TEXT,
    bid_ask_pressure TEXT,
    gamma_sr TEXT,
    delta_sr TEXT,
    depth_sr TEXT,
    oi_wall TEXT,
    chg_oi_wall TEXT,
    max_pain TEXT,
    data_source TEXT DEFAULT 'computed',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_atm_strike_expiry_day ON atm_strike_data(expiry, trading_day);

-- 5. PCR HISTORY (Put-Call Ratio history)
CREATE TABLE IF NOT EXISTS pcr_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry TEXT NOT NULL,
    strike_price DOUBLE PRECISION NOT NULL,
    atm_strike DOUBLE PRECISION,
    pcr_value DOUBLE PRECISION,
    oi_ce BIGINT,
    oi_pe BIGINT,
    data_source TEXT DEFAULT 'computed',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_pcr_day ON pcr_history(trading_day);

-- 6. GEX HISTORY (Gamma Exposure history)
CREATE TABLE IF NOT EXISTS gex_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry TEXT NOT NULL,
    strike_price DOUBLE PRECISION,
    atm_strike DOUBLE PRECISION,
    total_gex DOUBLE PRECISION,
    call_gex DOUBLE PRECISION,
    put_gex DOUBLE PRECISION,
    net_gex DOUBLE PRECISION,
    gamma_flip_level DOUBLE PRECISION,
    gex_signal TEXT,
    spot_price DOUBLE PRECISION,
    data_source TEXT DEFAULT 'computed',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_gex_day ON gex_history(trading_day);

-- 6b. BID/ASK QTY HISTORY (per-strike option-chain snapshot)
CREATE TABLE IF NOT EXISTS bid_ask_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry TEXT NOT NULL,
    strike_price DOUBLE PRECISION NOT NULL,
    atm_strike DOUBLE PRECISION,
    bid_qty_ce BIGINT,
    bid_qty_pe BIGINT,
    ask_qty_ce BIGINT,
    ask_qty_pe BIGINT,
    volume_ce BIGINT,
    volume_pe BIGINT,
    oi_ce BIGINT,
    oi_pe BIGINT,
    chgoi_ce BIGINT,
    chgoi_pe BIGINT,
    ltp_ce DOUBLE PRECISION,
    ltp_pe DOUBLE PRECISION,
    data_source TEXT DEFAULT 'computed',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_bid_ask_day ON bid_ask_history(trading_day);

-- 6c. VOLUME DELTA HISTORY (index-level Buy/Sell volume time series)
CREATE TABLE IF NOT EXISTS volume_delta_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    symbol TEXT NOT NULL DEFAULT 'NIFTY50',
    buy_volume BIGINT,
    sell_volume BIGINT,
    delta BIGINT,
    cum_delta BIGINT,
    divergence BOOLEAN DEFAULT FALSE,
    bias TEXT,
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol)
);
CREATE INDEX IF NOT EXISTS idx_vol_delta_day ON volume_delta_history(trading_day);

-- 7. DETECTED PATTERNS (Chart patterns)
CREATE TABLE IF NOT EXISTS detected_patterns (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    pattern_type TEXT NOT NULL,
    timeframe TEXT,
    direction TEXT,
    price_level DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    lower_bound DOUBLE PRECISION,
    score DOUBLE PRECISION,
    metadata JSONB,
    data_source TEXT DEFAULT 'computed',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, pattern_type, price_level, timeframe)
);
CREATE INDEX IF NOT EXISTS idx_patterns_day ON detected_patterns(trading_day);

-- 8. ORDERBOOK DATA (Bid/Ask orderbook)
CREATE TABLE IF NOT EXISTS orderbook_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry TEXT NOT NULL,
    strike_price DOUBLE PRECISION NOT NULL,
    bid_qty_ce BIGINT,
    ask_qty_ce BIGINT,
    bid_qty_pe BIGINT,
    ask_qty_pe BIGINT,
    bid_ask_pressure DOUBLE PRECISION,
    pressure_bias TEXT,
    data_source TEXT DEFAULT 'dhan_optionchain',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_orderbook_expiry_day ON orderbook_data(expiry, trading_day);

-- 9. USER PREFERENCES
CREATE TABLE IF NOT EXISTS user_preferences (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL UNIQUE,
    timeframe TEXT DEFAULT '5',
    auto_refresh BOOLEAN DEFAULT TRUE,
    days_back INTEGER DEFAULT 1,
    pivot_settings JSONB,
    pivot_proximity DOUBLE PRECISION DEFAULT 5,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 10. MARKET ANALYTICS (Daily market stats)
CREATE TABLE IF NOT EXISTS market_analytics (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    day_high DOUBLE PRECISION,
    day_low DOUBLE PRECISION,
    day_open DOUBLE PRECISION,
    day_close DOUBLE PRECISION,
    total_volume BIGINT,
    avg_price DOUBLE PRECISION,
    price_change DOUBLE PRECISION,
    price_change_pct DOUBLE PRECISION,
    UNIQUE(symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_analytics_symbol ON market_analytics(symbol, date);

-- 11. ALERTS HISTORY (All alert records)
CREATE TABLE IF NOT EXISTS alerts_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    alert_type TEXT NOT NULL,
    direction TEXT,
    strike_price DOUBLE PRECISION,
    underlying_price DOUBLE PRECISION,
    signal_details TEXT,
    score DOUBLE PRECISION,
    metadata JSONB,
    sent_via TEXT DEFAULT 'telegram',
    data_source TEXT DEFAULT 'computed',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, alert_type, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_alerts_day ON alerts_history(trading_day);

-- 12. MASTER SIGNALS (Master trading signals with full data)
CREATE TABLE IF NOT EXISTS master_signals (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    spot_price DOUBLE PRECISION,
    signal TEXT,
    trade_type TEXT,
    score INTEGER,
    abs_score INTEGER,
    strength TEXT,
    confidence INTEGER,
    candle_pattern TEXT,
    candle_direction TEXT,
    volume_label TEXT,
    volume_ratio DOUBLE PRECISION,
    location TEXT,
    resistance_levels TEXT,
    support_levels TEXT,
    net_gex DOUBLE PRECISION,
    atm_gex DOUBLE PRECISION,
    gamma_flip DOUBLE PRECISION,
    gex_mode TEXT,
    pcr_gex_badge TEXT,
    market_bias TEXT,
    vix_value DOUBLE PRECISION,
    vix_direction TEXT,
    oi_trend_signal TEXT,
    ce_activity TEXT,
    pe_activity TEXT,
    support_status TEXT,
    resistance_status TEXT,
    vidya_trend TEXT,
    vidya_delta_pct DOUBLE PRECISION,
    delta_vol_trend TEXT,
    vwap DOUBLE PRECISION,
    price_vs_vwap TEXT,
    reasons TEXT,
    alignment_summary TEXT,
    data_source TEXT DEFAULT 'master_signal',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, trading_day)
);
CREATE INDEX IF NOT EXISTS idx_master_signals_day ON master_signals(trading_day);

-- 13. MAX PAIN HISTORY (Max pain strike history)
CREATE TABLE IF NOT EXISTS max_pain_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry TEXT NOT NULL,
    max_pain_strike DOUBLE PRECISION,
    underlying_price DOUBLE PRECISION,
    distance_from_spot DOUBLE PRECISION,
    data_source TEXT DEFAULT 'computed',
    update_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, expiry)
);
CREATE INDEX IF NOT EXISTS idx_max_pain_day ON max_pain_history(trading_day);

-- =============================================
-- SUMMARY: 13 TABLES
-- =============================================
-- 1.  candles_data          - OHLCV candle data
-- 2.  nifty_spot_data       - LTP spot price history
-- 3.  option_chain_data     - Full option chain snapshots
-- 4.  atm_strike_data       - ATM strike analysis data
-- 5.  pcr_history           - Put-Call Ratio history
-- 6.  gex_history           - Gamma Exposure history
-- 7.  detected_patterns     - Detected chart patterns
-- 8.  orderbook_data        - Bid/Ask orderbook data
-- 9.  user_preferences      - User settings
-- 10. market_analytics      - Daily market stats
-- 11. alerts_history        - All alert records
-- 12. master_signals        - Master trading signals
-- 13. max_pain_history      - Max pain strike history

-- =============================================
-- ADDITIONAL TABLES used by vob_minimal.py
-- =============================================

-- 14. OC SIGNAL HISTORY (option-chain analysis snapshots)
CREATE TABLE IF NOT EXISTS oc_signal_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL UNIQUE,
    trading_day DATE NOT NULL,
    spot_price DOUBLE PRECISION,
    condition TEXT,
    confidence INTEGER,
    resistance_strikes JSONB,
    support_strikes JSONB,
    active_signals JSONB,
    breakout_level DOUBLE PRECISION,
    breakdown_level DOUBLE PRECISION,
    bias_reasoning JSONB,
    update_time TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_oc_signal_day ON oc_signal_history(trading_day);

-- 15. TRADE CONFIG (single-row auto-trade zone configuration)
CREATE TABLE IF NOT EXISTS trade_config (
    id INTEGER PRIMARY KEY,
    support_zone_bottom DOUBLE PRECISION,
    support_zone_top DOUBLE PRECISION,
    resistance_zone_bottom DOUBLE PRECISION,
    resistance_zone_top DOUBLE PRECISION,
    selected_strike DOUBLE PRECISION,
    call_entry DOUBLE PRECISION,
    call_target DOUBLE PRECISION,
    call_sl DOUBLE PRECISION,
    put_entry DOUBLE PRECISION,
    put_target DOUBLE PRECISION,
    put_sl DOUBLE PRECISION,
    auto_trade_enabled BOOLEAN DEFAULT FALSE,
    lot_size INTEGER DEFAULT 1,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 16. AUTO TRADES (zone-based auto trade records)
CREATE TABLE IF NOT EXISTS auto_trades (
    id BIGSERIAL PRIMARY KEY,
    trading_day DATE NOT NULL,
    trade_type TEXT,                 -- CALL / PUT
    strike DOUBLE PRECISION,
    security_id TEXT,
    entry_price DOUBLE PRECISION,
    target DOUBLE PRECISION,
    sl DOUBLE PRECISION,
    lot_size INTEGER DEFAULT 1,
    status TEXT DEFAULT 'OPEN',      -- OPEN / CLOSED
    exit_price DOUBLE PRECISION,
    exit_reason TEXT,                -- TARGET / SL / MANUAL / REVERSE
    order_id TEXT,
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    zone_confirmations TEXT,
    spot_at_entry DOUBLE PRECISION,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_auto_trades_day ON auto_trades(trading_day);
CREATE INDEX IF NOT EXISTS idx_auto_trades_status ON auto_trades(status);

-- 17. AUTO OPTION TRADES (standalone auto_option_trader.py trigger persistence)
CREATE TABLE IF NOT EXISTS auto_option_trades (
    id TEXT PRIMARY KEY,             -- Dhan client id (one active trigger per account)
    payload JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 18. DHAN TICKS (written by ws_worker.py — sub-second LTP + tick-rule cum delta)
CREATE TABLE IF NOT EXISTS dhan_ticks (
    id TEXT PRIMARY KEY,             -- "<exchange_segment>:<security_id>"
    exchange_segment TEXT,
    security_id INTEGER,
    ltp DOUBLE PRECISION,
    cum_delta DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    last_trade_qty DOUBLE PRECISION,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dhan_ticks_updated ON dhan_ticks(updated_at);

-- 20. VOB APP STATE (persists composite scores + history across Streamlit restarts)
CREATE TABLE IF NOT EXISTS vob_app_state (
    id TEXT PRIMARY KEY,             -- Dhan client id (or 'default')
    payload JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 19. DHAN SWEEPS (written by ws_worker.py — L2 sweep events for ignition condition)
CREATE TABLE IF NOT EXISTS dhan_sweeps (
    id BIGSERIAL PRIMARY KEY,
    exchange_segment TEXT,
    security_id INTEGER,
    direction TEXT,                  -- 'up' | 'down'
    magnitude DOUBLE PRECISION,      -- size of the swept resting qty
    detail TEXT,
    fired_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dhan_sweeps_fired ON dhan_sweeps(fired_at DESC);
