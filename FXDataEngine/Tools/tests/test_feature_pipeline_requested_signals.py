from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _assert_tokens(text: str, tokens: list[str]) -> None:
    for token in tokens:
        assert token in text


def test_requested_signal_math_helpers_exist_in_swift():
    feature_math = _read("FXDataEngine/Sources/FXDataEngine/FeatureMath.swift")
    _assert_tokens(
        feature_math,
        [
            "public static func qsdemaAsSeries(",
            "public static func rsiAsSeries(",
            "public static func atrAsSeries(",
            "public static func parkinsonVolAsSeries(",
            "public static func rollingMedianAsSeries(",
            "public static func rollingMADAsSeries(",
            "public static func rogersSatchellVolAsSeries(",
            "public static func garmanKlassVolAsSeries(",
            "public static func kalmanEstimateAsSeries(",
            "public static func ehlersSuperSmootherAsSeries(",
        ],
    )


def test_requested_signals_are_registered_in_swift_feature_schema():
    registry = _read("FXDataEngine/Sources/FXDataEngine/FeatureSchema.swift")
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
        assert f'case {feature_index}: "{feature_name}"' in registry


def test_requested_signals_are_emitted_by_swift_feature_pipeline():
    feature_pipeline = _read("FXDataEngine/Sources/FXDataEngine/FeaturePipeline.swift")
    _assert_tokens(
        feature_pipeline,
        [
            "features[38] = movingAverageEdgeValue(",
            "FeatureMath.qsdemaAsSeries(close, startIndex: 0, period: 100)",
            "features[39] = movingAverageEdgeValue(",
            "FeatureMath.qsdemaAsSeries(close, startIndex: 0, period: 200)",
            "let rsi14 = FeatureMath.rsiAsSeries(volatilityClose, startIndex: 0, period: 14)",
            "features[40] = fxClampSignedUnit((rsi14 - 50.0) / 50.0)",
            "let atr14 = FeatureMath.atrAsSeries(high: high, low: low, close: volatilityClose, startIndex: 0, period: 14)",
            "features[41] = fxClamp(natr14, 0.0, 1.0)",
            "features[42] = features[41]",
            "features[43] = fxClamp(FeatureMath.parkinsonVolAsSeries(high: high, low: low, startIndex: 0, period: 20) * 100.0, 0.0, 1.0)",
            "FeatureMath.rogersSatchellVolAsSeries(open: open, high: high, low: low, close: volatilityClose, startIndex: 0, period: 20)",
            "FeatureMath.garmanKlassVolAsSeries(open: open, high: high, low: low, close: volatilityClose, startIndex: 0, period: 20)",
            "let median21 = FeatureMath.rollingMedianAsSeries(close, startIndex: 0, period: 21)",
            "let mad21 = FeatureMath.rollingMADAsSeries(close, startIndex: 0, period: 21, median: median21)",
            "features[46] = movingAverageEdgeValue(current: currentClose, average: median21)",
            "features[47] = fxClampSignedUnit((currentClose - median21) / max(1.4826 * mad21, 1.0))",
            "let kalman34 = FeatureMath.kalmanEstimateAsSeries(close, startIndex: 0, period: 34)",
            "features[48] = movingAverageEdgeValue(current: currentClose, average: kalman34)",
            "let superSmoother20 = FeatureMath.ehlersSuperSmootherAsSeries(close, startIndex: 0, period: 20)",
            "features[49] = movingAverageEdgeValue(current: currentClose, average: superSmoother20)",
        ],
    )
