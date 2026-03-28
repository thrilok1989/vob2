-- VOB2 Schema Extension: Alerts History & Max Pain History
-- Run this in Supabase SQL Editor after 001_create_tables.sql

-- 10. Alerts History (tracks all signals sent)
CREATE TABLE IF NOT EXISTS alerts_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    alert_type TEXT NOT NULL,
    direction TEXT,
    strike_price NUMERIC(10,2),
    underlying_price NUMERIC(12,2),
    signal_details TEXT,
    score NUMERIC(8,2),
    metadata JSONB,
    sent_via TEXT DEFAULT 'telegram',
    data_source TEXT NOT NULL DEFAULT 'computed',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_alerts_unique ON alerts_history(timestamp, alert_type, COALESCE(strike_price, 0));
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts_history(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_trading_day ON alerts_history(trading_day);

-- 11. Max Pain History (tracks max pain level over time)
CREATE TABLE IF NOT EXISTS max_pain_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    trading_day DATE NOT NULL,
    expiry DATE NOT NULL,
    max_pain_strike NUMERIC(10,2) NOT NULL,
    underlying_price NUMERIC(12,2),
    distance_from_spot NUMERIC(10,2),
    data_source TEXT NOT NULL DEFAULT 'computed',
    update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(timestamp, expiry)
);
CREATE INDEX IF NOT EXISTS idx_maxpain_timestamp ON max_pain_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_maxpain_expiry ON max_pain_history(expiry);
CREATE INDEX IF NOT EXISTS idx_maxpain_trading_day ON max_pain_history(trading_day);

-- 12. User Preferences (if not exists from 001)
CREATE TABLE IF NOT EXISTS user_preferences (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    timeframe TEXT DEFAULT '5',
    auto_refresh BOOLEAN DEFAULT true,
    days_back INTEGER DEFAULT 1,
    pivot_settings JSONB,
    pivot_proximity INTEGER DEFAULT 5,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id)
);

-- 13. Market Analytics (if not exists from 001)
CREATE TABLE IF NOT EXISTS market_analytics (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    day_high NUMERIC(12,2),
    day_low NUMERIC(12,2),
    day_open NUMERIC(12,2),
    day_close NUMERIC(12,2),
    total_volume BIGINT,
    avg_price NUMERIC(12,2),
    price_change NUMERIC(12,2),
    price_change_pct NUMERIC(8,4),
    UNIQUE(symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_analytics_symbol ON market_analytics(symbol, date);
