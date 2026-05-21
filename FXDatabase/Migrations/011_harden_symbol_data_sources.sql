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

ALTER TABLE {database}.symbol_data_sources
ADD COLUMN IF NOT EXISTS broker_source_id String DEFAULT '' FIRST;
