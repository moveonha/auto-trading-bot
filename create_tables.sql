
-- 아래 SQL을 Supabase Dashboard → SQL Editor에서 실행하세요!

CREATE TABLE IF NOT EXISTS crypto_ohlcv (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp BIGINT NOT NULL,
    datetime TIMESTAMPTZ NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(symbol, timeframe, timestamp)
);

CREATE TABLE IF NOT EXISTS crypto_features (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp BIGINT NOT NULL,
    datetime TIMESTAMPTZ NOT NULL,
    open DECIMAL(20,8),
    high DECIMAL(20,8),
    low DECIMAL(20,8),
    close DECIMAL(20,8),
    volume DECIMAL(20,8),
    technical_indicators JSONB,
    targets JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(symbol, timeframe, timestamp)
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_crypto_ohlcv_symbol_timeframe ON crypto_ohlcv(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_crypto_ohlcv_datetime ON crypto_ohlcv(datetime);
CREATE INDEX IF NOT EXISTS idx_crypto_features_symbol_timeframe ON crypto_features(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_crypto_features_datetime ON crypto_features(datetime);
