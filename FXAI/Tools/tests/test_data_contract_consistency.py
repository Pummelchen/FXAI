from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_datacore_bundle_pins_canonical_m1_ohlc_and_spread_contract():
    contracts = _read("Engine/Core/core_pipeline_contracts.mqh")
    for token in (
        "double open_arr[];",
        "double high_arr[];",
        "double low_arr[];",
        "double close_arr[];",
        "datetime time_arr[];",
        "int spread_m1[];",
    ):
        assert token in contracts


def test_datacore_loader_validates_the_m1_series_bundle_before_feature_construction():
    data_core = _read("Engine/Core/core_data_core.mqh")
    feature_core = _read("Engine/Core/core_feature_core.mqh")

    assert "FXAI_ValidateM1SeriesBundle(bundle.time_arr," in data_core
    assert "bundle.open_arr," in data_core
    assert "bundle.high_arr," in data_core
    assert "bundle.low_arr," in data_core
    assert "bundle.close_arr," in data_core
    assert "bundle.spread_m1" in data_core
    assert "bundle.spread_m1" in feature_core


def test_audit_sample_path_consumes_same_time_ohlc_spread_contract():
    audit_samples = _read("Tests/audit_samples.mqh")
    assert "const datetime &time_arr[]" in audit_samples
    assert "const double &open_arr[]" in audit_samples
    assert "const double &high_arr[]" in audit_samples
    assert "const double &low_arr[]" in audit_samples
    assert "const double &close_arr[]" in audit_samples
    assert "const int &spread_arr[]" in audit_samples
    assert "FXAI_GetSpreadAtIndex(" in audit_samples
