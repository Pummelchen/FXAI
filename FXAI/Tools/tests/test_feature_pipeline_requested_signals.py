from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_requested_signal_math_helpers_exist():
    feature_math = _read("Engine/feature_math.mqh")
    required_helpers = [
        "double FXAI_QSDEMAAt(",
        "double FXAI_RSIAt(",
        "double FXAI_ATRAt(",
        "double FXAI_ParkinsonVolAt(",
        "double FXAI_RollingMedianAt(",
        "double FXAI_RollingMADAt(",
        "double FXAI_RogersSatchellVolAt(",
        "double FXAI_GarmanKlassVolAt(",
        "double FXAI_KalmanEstimateAt(",
        "double FXAI_EhlersSuperSmootherAt(",
    ]

    for helper_signature in required_helpers:
        assert helper_signature in feature_math


def test_requested_signals_are_registered_in_feature_schema():
    registry = _read("Engine/feature_registry.mqh")
    expected_names = {
        38: "qsdema100_edge",
        39: "qsdema200_edge",
        40: "rsi14",
        41: "atr14_unit",
        42: "natr14",
        43: "parkinson20",
        44: "rogers_satchell20",
        45: "garman_klass20",
        46: "median21_edge",
        47: "hampel21",
        48: "kalman34_edge",
        49: "supersmoother20_edge",
    }

    for feature_index, feature_name in expected_names.items():
        expected_line = f'case {feature_index}: return "{feature_name}";'
        assert expected_line in registry


def test_requested_signals_are_emitted_by_canonical_feature_builder():
    feature_build = _read("Engine/feature_build.mqh")
    expected_assignments = {
        38: r"features\[38\]\s*=\s*FXAI_MAEdgeFeature\(c,\s*qsdema_100,\s*vol_unit\);",
        39: r"features\[39\]\s*=\s*FXAI_MAEdgeFeature\(c,\s*qsdema_200,\s*vol_unit\);",
        40: r"features\[40\]\s*=\s*\(rsi14\s*-\s*50\.0\)\s*/\s*50\.0;",
        41: r"features\[41\]\s*=\s*\(c\s*>\s*0\.0\s*\?\s*\(\(atr14\s*/\s*c\)\s*/\s*vol_unit\)\s*:\s*0\.0\);",
        42: r"features\[42\]\s*=\s*natr14;",
        43: r"features\[43\]\s*=\s*\(vol_unit\s*>\s*0\.0\s*\?\s*\(parkinson20\s*/\s*vol_unit\)\s*:\s*0\.0\);",
        44: r"features\[44\]\s*=\s*\(vol_unit\s*>\s*0\.0\s*\?\s*\(rs20\s*/\s*vol_unit\)\s*:\s*0\.0\);",
        45: r"features\[45\]\s*=\s*\(vol_unit\s*>\s*0\.0\s*\?\s*\(gk20\s*/\s*vol_unit\)\s*:\s*0\.0\);",
        46: r"features\[46\]\s*=\s*FXAI_MAEdgeFeature\(c,\s*med21,\s*vol_unit\);",
        47: r"features\[47\]\s*=\s*\(c\s*-\s*med21\)\s*/\s*hampel_denom;",
        48: r"features\[48\]\s*=\s*FXAI_MAEdgeFeature\(c,\s*kalman34,\s*vol_unit\);",
        49: r"features\[49\]\s*=\s*FXAI_MAEdgeFeature\(c,\s*ss20,\s*vol_unit\);",
    }

    for feature_index, pattern in expected_assignments.items():
        assert re.search(pattern, feature_build), feature_index

    assert "double med21 = FXAI_RollingMedianAt(main_m1, i, 21);" in feature_build
    assert "double mad21 = FXAI_RollingMADAt(main_m1, i, 21, med21);" in feature_build
