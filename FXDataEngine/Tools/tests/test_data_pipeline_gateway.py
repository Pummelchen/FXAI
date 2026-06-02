from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN_MT5_MARKET_PATTERNS = (
    r"\bCopyRates\s*\(",
    r"\bCopyTicksRange\s*\(",
    r"\bCopyTicks\s*\(",
    r"\bCopyClose\s*\(",
    r"\bSymbolInfoTick\s*\(",
    r"\biBarShift\s*\(",
    r"\biTime\s*\(",
    r"\biOpen\s*\(",
    r"\biClose\s*\(",
)


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _assert_tokens(text: str, tokens: list[str]) -> None:
    for token in tokens:
        assert token in text


def _iter_swift_code_files() -> list[Path]:
    roots = [
        ROOT / "FXDataEngine/Sources/FXDataEngine",
        ROOT / "FXDataEngine/Sources/FXDataEngineCLI",
        ROOT / "FXDataEngine/Tests/FXDataEngineTests",
    ]
    paths: list[Path] = []
    for root in roots:
        paths.extend(path for path in root.rglob("*.swift") if path.is_file())
    return sorted(paths)


def test_mt5_market_data_apis_are_absent_from_swift_data_engine():
    offenders: list[str] = []
    for path in _iter_swift_code_files():
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_MT5_MARKET_PATTERNS:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: {pattern}")
    assert not offenders, "MT5 market-data API calls remain in Swift FXDataEngine:\n" + "\n".join(offenders)


def test_market_data_gateway_uses_fxdatabase_ohlcv_contract():
    gateway = _read("FXDataEngine/Sources/FXDataEngine/MarketData/MarketDataGateway.swift")
    market_data = _read("FXDataEngine/Sources/FXDataEngine/MarketData/MarketData.swift")

    _assert_tokens(
        gateway,
        [
            "import FXBacktestAPI",
            "public struct FXDatabaseMarketHistoryRequest",
            "public struct FXDatabaseMarketUniverseRequest",
            "public struct FXDatabaseMarketDataLoader",
            "public func apiRequest() -> FXBacktestM1HistoryRequest",
            ".loadM1History(request.apiRequest())",
            "return try M1OHLCVSeries(response: response)",
            "requireAlignedTimestamps: Bool = true",
        ],
    )
    _assert_tokens(
        market_data,
        [
            "public struct M1OHLCVSeries: Sendable",
            "public let volume: ContiguousArray<UInt64>",
            "@inlinable public var hasVolume: Bool { volume.contains { $0 > 0 } }",
            "FXBacktestM1HistoryResponse",
        ],
    )
    assert "spread" not in market_data.lower()


def test_prediction_path_consumers_use_core_pipeline_only():
    pipeline = _read("FXDataEngine/Sources/FXDataEngine/Core/FXDataEngine.swift")
    invocation = _read("FXDataEngine/Sources/FXDataEngine/Plugins/PluginInvocation.swift")
    runtime = _read("FXDataEngine/Sources/FXDataEngine/Features/RuntimeFeaturePipeline.swift")

    _assert_tokens(
        pipeline,
        [
            "public let dataCore: DataCore",
            "public let featureCore: FeatureCore",
            "public let normalizationCore: NormalizationCore",
            "let dataBundle = try dataCore.buildBundle(request: dataRequest, universe: universe)",
            "let featureFrame = try featureCore.buildFrame(",
            "let normalizationFrame = try normalizationCore.buildInputFrame(",
            "from: featureFrame",
            "let payloadFrame = try normalizationCore.buildPayloadFrame(",
            "dataHasVolume: featureFrame.hasVolume",
        ],
    )
    _assert_tokens(
        invocation,
        [
            "try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)",
            "try plugin.train(request, hyperParameters: hyperParameters)",
            "return try plugin.predict(request, hyperParameters: hyperParameters)",
        ],
    )
    _assert_tokens(
        runtime,
        [
            "public struct RuntimeFeaturePipelineRequirement",
            "public struct RuntimeFeaturePipelinePlan",
            "public enum RuntimeFeaturePipelineTools",
            "public static func bootstrapRequirement(",
        ],
    )


def test_normalization_pipeline_remains_causal_and_train_split_safe():
    normalization_state = _read("FXDataEngine/Sources/FXDataEngine/Normalization/NormalizationState.swift")
    warmup = _read("FXDataEngine/Sources/FXDataEngine/Lifecycle/Warmup.swift")

    _assert_tokens(
        normalization_state,
        [
            "public mutating func prepareForSample(",
            "let rewind =",
            "let configChanged =",
            "state.lastSampleTimeUTC = sampleTimeUTC",
            "public mutating func record(",
            "state.values[offset] = fxSafeFinite(value)",
            "public func recentValues(",
        ],
    )
    assert normalization_state.index("public mutating func prepareForSample(") < normalization_state.index(
        "public mutating func record("
    )
    _assert_tokens(
        warmup,
        [
            "public static func normalizationCandidateSplit(",
            "let validationEnd = endIndex",
            "let trainingEnd = validationStart - purge - 1",
            "guard trainingEnd - trainingStart >= 100 else { return nil }",
        ],
    )
