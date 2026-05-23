from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_swift_market_series_pins_canonical_m1_ohlcv_contract():
    contracts = _read("FXDataEngine/Sources/FXDataEngine/MarketData/MarketData.swift")
    for token in (
        "public struct M1OHLCVBar",
        "public struct M1OHLCVSeries",
        "public let utcTimestamps: ContiguousArray<Int64>",
        "public let open: ContiguousArray<Int64>",
        "public let high: ContiguousArray<Int64>",
        "public let low: ContiguousArray<Int64>",
        "public let close: ContiguousArray<Int64>",
        "public let volume: ContiguousArray<UInt64>",
        "@inlinable public var hasVolume: Bool { volume.contains { $0 > 0 } }",
        "M1 OHLCV columns must have equal length",
    ):
        assert token in contracts
    assert "spread" not in contracts.lower()


def test_swift_gateway_preserves_volume_from_fxdatabase_response():
    gateway = _read("FXDataEngine/Sources/FXDataEngine/MarketData/MarketDataGateway.swift")

    assert "volume: outVolume" in gateway
    assert "bucketVolume = volume[index]" in gateway
    assert "bucketVolume = bucketVolume.saturatingAdd(volume[index])" in gateway
    assert "spread" not in gateway.lower()


def test_swift_offline_export_uses_ohlcv_tsv_header_without_legacy_cost_columns():
    offline_export = _read("FXDataEngine/Sources/FXDataEngine/MarketData/OfflineExport.swift")

    assert 'var lines = ["time_unix\\topen\\thigh\\tlow\\tclose\\tvolume"]' in offline_export
    assert "String(series.volume[index])" in offline_export
    assert "spread" not in offline_export.lower()
