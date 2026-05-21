ALTER TABLE {database}.mt5_ohlc_m1_raw
ADD COLUMN IF NOT EXISTS source_origin LowCardinality(String) DEFAULT 'MT5' AFTER broker_source_id;

ALTER TABLE {database}.mt5_ohlc_m1_raw
ADD COLUMN IF NOT EXISTS volume UInt64 DEFAULT 0 AFTER close_scaled;

ALTER TABLE {database}.ohlc_m1_canonical
ADD COLUMN IF NOT EXISTS source_origin LowCardinality(String) DEFAULT 'MT5' AFTER broker_source_id;

ALTER TABLE {database}.ohlc_m1_canonical
ADD COLUMN IF NOT EXISTS volume UInt64 DEFAULT 0 AFTER close_scaled;

ALTER TABLE {database}.ohlc_m1_conflicts
ADD COLUMN IF NOT EXISTS source_origin LowCardinality(String) DEFAULT 'MT5' AFTER broker_source_id;

ALTER TABLE {database}.ohlc_m1_conflicts
ADD COLUMN IF NOT EXISTS existing_volume UInt64 DEFAULT 0 AFTER existing_close_scaled;

ALTER TABLE {database}.ohlc_m1_conflicts
ADD COLUMN IF NOT EXISTS incoming_volume UInt64 DEFAULT 0 AFTER incoming_close_scaled;

ALTER TABLE {database}.ingest_state
ADD COLUMN IF NOT EXISTS source_origin LowCardinality(String) DEFAULT 'MT5' AFTER broker_source_id;

ALTER TABLE {database}.ingest_operations
ADD COLUMN IF NOT EXISTS source_origin LowCardinality(String) DEFAULT 'MT5' AFTER broker_source_id;

ALTER TABLE {database}.ohlc_m1_verified_coverage
ADD COLUMN IF NOT EXISTS source_origin LowCardinality(String) DEFAULT 'MT5' AFTER broker_source_id;

ALTER TABLE {database}.data_certificates
ADD COLUMN IF NOT EXISTS source_origin LowCardinality(String) DEFAULT 'MT5' AFTER broker_source_id;

ALTER TABLE {database}.verification_results
ADD COLUMN IF NOT EXISTS source_origin LowCardinality(String) DEFAULT 'MT5' AFTER broker_source_id;

ALTER TABLE {database}.repair_log
ADD COLUMN IF NOT EXISTS source_origin LowCardinality(String) DEFAULT 'MT5' AFTER broker_source_id;

CREATE TABLE IF NOT EXISTS {database}.symbol_data_sources
(
    broker_source_id String,
    logical_symbol String,
    source_origin LowCardinality(String),
    source_symbol String,
    priority UInt8,
    status LowCardinality(String),
    configured_at_utc Int64
)
ENGINE = ReplacingMergeTree(configured_at_utc)
ORDER BY (broker_source_id, logical_symbol, source_origin, priority);
