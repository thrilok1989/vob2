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
