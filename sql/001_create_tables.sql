-- VOB2 Supabase Schema: All tables for Supabase-first architecture
-- Run this in Supabase SQL Editor

-- 1. Candles Data (replaces candle_data)
CREATE TABLE IF NOT EXISTS candles_data (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    datetime TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    open NUMERIC(12,2) NOT NULL,
    high NUMERIC(12,2) NOT NULL,
    low NUMERIC(12,2) NOT NULL,
    close NUMERIC(12,2) NOT NULL,
    volume BIGINT NOT NULL,
    data_source TEXT NOT NULL DEFAULT 'dhan_intraday',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(symbol, exchange, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_candles_trading_day ON candles_data(trading_day);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON candles_data(symbol, timeframe, datetime);

-- 2. Nifty Spot Data (LTP history)
CREATE TABLE IF NOT EXISTS nifty_spot_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    ltp NUMERIC(12,2) NOT NULL,
    exchange_segment TEXT NOT NULL DEFAULT 'IDX_I',
    security_id TEXT NOT NULL DEFAULT '13',
    data_source TEXT NOT NULL DEFAULT 'dhan_ltp',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(timestamp, security_id)
);
CREATE INDEX IF NOT EXISTS idx_spot_timestamp ON nifty_spot_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_spot_trading_day ON nifty_spot_data(trading_day);

-- 3. Option Chain Data (full chain per strike per timestamp)
CREATE TABLE IF NOT EXISTS option_chain_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry DATE NOT NULL,
    strike_price NUMERIC(10,2) NOT NULL,
    atm_strike NUMERIC(10,2),
    underlying_price NUMERIC(12,2),
    last_price_ce NUMERIC(12,2),
    open_interest_ce BIGINT,
    previous_oi_ce BIGINT,
    change_in_oi_ce BIGINT,
    volume_ce BIGINT,
    iv_ce NUMERIC(8,4),
    bid_qty_ce BIGINT,
    ask_qty_ce BIGINT,
    last_price_pe NUMERIC(12,2),
    open_interest_pe BIGINT,
    previous_oi_pe BIGINT,
    change_in_oi_pe BIGINT,
    volume_pe BIGINT,
    iv_pe NUMERIC(8,4),
    bid_qty_pe BIGINT,
    ask_qty_pe BIGINT,
    delta_ce NUMERIC(10,6),
    gamma_ce NUMERIC(10,6),
    vega_ce NUMERIC(10,6),
    theta_ce NUMERIC(10,6),
    rho_ce NUMERIC(10,6),
    delta_pe NUMERIC(10,6),
    gamma_pe NUMERIC(10,6),
    vega_pe NUMERIC(10,6),
    theta_pe NUMERIC(10,6),
    rho_pe NUMERIC(10,6),
    data_source TEXT NOT NULL DEFAULT 'dhan_optionchain',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_oc_timestamp ON option_chain_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_oc_strike ON option_chain_data(strike_price);
CREATE INDEX IF NOT EXISTS idx_oc_expiry ON option_chain_data(expiry);
CREATE INDEX IF NOT EXISTS idx_oc_atm ON option_chain_data(atm_strike);
CREATE INDEX IF NOT EXISTS idx_oc_trading_day ON option_chain_data(trading_day);

-- 4. ATM Strike Data (bias/verdict per strike)
CREATE TABLE IF NOT EXISTS atm_strike_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry DATE NOT NULL,
    strike_price NUMERIC(10,2) NOT NULL,
    atm_strike NUMERIC(10,2),
    underlying_price NUMERIC(12,2),
    zone TEXT,
    level TEXT,
    pcr NUMERIC(8,4),
    pcr_signal TEXT,
    bias_score INTEGER,
    verdict TEXT,
    ltp_bias TEXT, oi_bias TEXT, chg_oi_bias TEXT, volume_bias TEXT,
    delta_bias TEXT, gamma_bias TEXT, theta_bias TEXT,
    ask_qty_bias TEXT, bid_qty_bias TEXT, ask_bid_bias TEXT,
    iv_bias TEXT, dvp_bias TEXT, pressure_bias TEXT,
    delta_exp TEXT, gamma_exp TEXT,
    operator_entry TEXT,
    scalp_moment TEXT,
    fake_real TEXT,
    bid_ask_pressure NUMERIC(12,2),
    gamma_sr TEXT, delta_sr TEXT, depth_sr TEXT, oi_wall TEXT,
    chg_oi_wall TEXT, max_pain TEXT,
    data_source TEXT NOT NULL DEFAULT 'computed',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_atm_timestamp ON atm_strike_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_atm_strike ON atm_strike_data(strike_price);
CREATE INDEX IF NOT EXISTS idx_atm_expiry ON atm_strike_data(expiry);
CREATE INDEX IF NOT EXISTS idx_atm_atm_strike ON atm_strike_data(atm_strike);

-- 5. PCR History (replaces session_state.pcr_history)
CREATE TABLE IF NOT EXISTS pcr_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry DATE NOT NULL,
    strike_price NUMERIC(10,2) NOT NULL,
    atm_strike NUMERIC(10,2),
    pcr_value NUMERIC(8,4) NOT NULL,
    oi_ce BIGINT,
    oi_pe BIGINT,
    data_source TEXT NOT NULL DEFAULT 'computed',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_pcr_timestamp ON pcr_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_pcr_strike ON pcr_history(strike_price);
CREATE INDEX IF NOT EXISTS idx_pcr_expiry ON pcr_history(expiry);
CREATE INDEX IF NOT EXISTS idx_pcr_trading_day ON pcr_history(trading_day);

-- 6. GEX History (replaces session_state.gex_history)
CREATE TABLE IF NOT EXISTS gex_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry DATE NOT NULL,
    strike_price NUMERIC(10,2),
    atm_strike NUMERIC(10,2),
    total_gex NUMERIC(12,4),
    call_gex NUMERIC(12,4),
    put_gex NUMERIC(12,4),
    net_gex NUMERIC(12,4),
    gamma_flip_level NUMERIC(12,2),
    gex_signal TEXT,
    spot_price NUMERIC(12,2),
    data_source TEXT NOT NULL DEFAULT 'computed',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_gex_timestamp ON gex_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_gex_strike ON gex_history(strike_price);
CREATE INDEX IF NOT EXISTS idx_gex_expiry ON gex_history(expiry);
CREATE INDEX IF NOT EXISTS idx_gex_trading_day ON gex_history(trading_day);

-- 7. Detected Patterns (VOB, POC, Swing, Reversal)
CREATE TABLE IF NOT EXISTS detected_patterns (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    pattern_type TEXT NOT NULL,
    timeframe TEXT,
    direction TEXT,
    price_level NUMERIC(12,2),
    upper_bound NUMERIC(12,2),
    lower_bound NUMERIC(12,2),
    score NUMERIC(8,2),
    metadata JSONB,
    data_source TEXT NOT NULL DEFAULT 'computed',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(timestamp, pattern_type, price_level, COALESCE(timeframe, ''))
);
CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON detected_patterns(timestamp);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON detected_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_trading_day ON detected_patterns(trading_day);

-- 8. Orderbook Data (bid/ask depth)
CREATE TABLE IF NOT EXISTS orderbook_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry DATE NOT NULL,
    strike_price NUMERIC(10,2) NOT NULL,
    bid_qty_ce BIGINT,
    ask_qty_ce BIGINT,
    bid_qty_pe BIGINT,
    ask_qty_pe BIGINT,
    bid_ask_pressure NUMERIC(12,2),
    pressure_bias TEXT,
    data_source TEXT NOT NULL DEFAULT 'dhan_optionchain',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(timestamp, expiry, strike_price)
);
CREATE INDEX IF NOT EXISTS idx_ob_timestamp ON orderbook_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_ob_strike ON orderbook_data(strike_price);
CREATE INDEX IF NOT EXISTS idx_ob_expiry ON orderbook_data(expiry);

-- 9. Macro Data (future use)
CREATE TABLE IF NOT EXISTS macro_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    indicator_name TEXT NOT NULL,
    indicator_value NUMERIC(16,6),
    metadata JSONB,
    data_source TEXT NOT NULL DEFAULT 'manual',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(timestamp, indicator_name)
);
CREATE INDEX IF NOT EXISTS idx_macro_timestamp ON macro_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_macro_indicator ON macro_data(indicator_name);

-- Migration: copy existing candle_data to candles_data
INSERT INTO candles_data (symbol, exchange, timeframe, timestamp, datetime, trading_day, open, high, low, close, volume, data_source, update_time)
SELECT symbol, exchange, timeframe, timestamp, datetime, datetime::date, open, high, low, close, volume, 'dhan_intraday', NOW()
FROM candle_data
ON CONFLICT (symbol, exchange, timeframe, timestamp) DO NOTHING;
