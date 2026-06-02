from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _assert_tokens(text: str, tokens: list[str]) -> None:
    for token in tokens:
        assert token in text


def test_swift_normalization_enum_exposes_exact_buffer_variants():
    constants = _read("FXDataEngine/Sources/FXDataEngine/Core/Constants.swift")
    core_types = _read("FXDataEngine/Sources/FXDataEngine/Core/CoreTypes.swift")
    normalization = _read("FXDataEngine/Sources/FXDataEngine/Normalization/Normalization.swift")

    _assert_tokens(
        constants,
        [
            "public static let normMethodCount = 17",
            "public static let unitRangeFloor = 0.0001",
            "public static let unitRangeCeil = 0.9999",
            "public static let signedUnitRangeFloor = -0.9999",
            "public static let signedUnitRangeCeil = 0.9999",
            "public func fxClampUnit(_ value: Double) -> Double",
            "public func fxClampSignedUnit(_ value: Double) -> Double",
        ],
    )
    _assert_tokens(
        core_types,
        [
            "public enum FeatureNormalizationMethod: Int, Codable, Sendable, CaseIterable",
            "case existing = 0",
            "case minMaxBuffer5",
            "case changePercent",
            "case binary01",
            "case logReturn",
            "case relativeChangePercent",
            "case candleGeometry",
            "case volatilityStdReturns",
            "case atrNatrUnit",
            "case zScore",
            "case robustMedianIQR",
            "case quantileToNormal",
            "case powerYeoJohnson",
            "case revin",
            "case dain",
            "case minMaxBuffer2",
            "case minMaxBuffer3",
        ],
    )
    _assert_tokens(
        normalization,
        [
            "case .minMaxBuffer2:",
            "case .minMaxBuffer3:",
            "case .binary01:",
            "value = hasPrevious ? (current > prior ? 1.0 : 0.0) : 0.0",
            "out[index] = fxClampSignedUnit(value)",
        ],
    )


def test_swift_normalization_implements_fitted_and_adaptive_layers():
    normalization_state = _read("FXDataEngine/Sources/FXDataEngine/Normalization/NormalizationState.swift")
    runtime_sections = _read("FXDataEngine/Sources/FXDataEngine/RuntimeArtifacts/RuntimeArtifactSections.swift")

    _assert_tokens(
        normalization_state,
        [
            "public static let candidateMax = FXDataEngineConstants.normalizationCandidateMax",
            "public static func normalizationMethodCandidates(",
            "var usesAdaptivePayloadNormalization: Bool",
            "self == .revin || self == .dain",
            "var usesFittedStats: Bool",
            "var usesRollingNormalizationHistory: Bool",
            "public mutating func applyGroupWindows(",
            "public mutating func record(",
            "public mutating func fit(",
            "public func featureStats(",
        ],
    )
    _assert_tokens(
        runtime_sections,
        [
            "public static let normalizationQuantileKnots = 17",
            "public enum RuntimeNormalizationWindowsCodec",
            "public enum RuntimeNormalizationWindowConfigCodec",
            "public enum RuntimeNormalizationHistoryCodec",
            "public enum RuntimeNormalizationFitCodec",
        ],
    )


def test_swift_feature_pipeline_uses_volume_instead_of_legacy_spread_rank():
    feature_pipeline = _read("FXDataEngine/Sources/FXDataEngine/Features/FeaturePipeline.swift")
    feature_schema = _read("FXDataEngine/Sources/FXDataEngine/Features/FeatureSchema.swift")
    build_tools = _read("FXDataEngine/Sources/FXDataEngine/Features/FeatureBuildTools.swift")

    _assert_tokens(
        feature_pipeline,
        [
            "let hasVolume = Self.hasUsableVolume(universe)",
            "fillVolumeAwareMicrostructureFeatures(&features, universe: universe, index: sampleIndex, hasVolume: hasVolume)",
            "features[6] = mean20 > 0 ? fxClamp((log1p(vol) / log1p(mean20)) - 1.0, -1.0, 1.0) : 0.0",
            "features[68] = std20 > 0 ? fxClampSignedUnit((vol - mean20) / std20 / 4.0) : 0.0",
            "features[75] = vol > 0 ? 1.0 : 0.0",
            "features[83] = volumeRank(series, index, window: 20)",
            "func volumeRank(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double",
            "func timeframeState(_ series: M1OHLCVSeries, index: Int, window: Int, hasVolume: Bool) -> [Double]",
        ],
    )
    _assert_tokens(
        feature_schema,
        [
            "case 6: \"volume_norm\"",
            "case 68: \"volume_shock\"",
            "case 74: \"volume_session_activity\"",
            "case 80: \"volume_log\"",
            "case 83: \"volume_rank_20\"",
            "case 3: \"volume_pressure\"",
        ],
    )
    assert "spread" not in feature_pipeline.lower()
    assert "spread" not in feature_schema.lower()
    assert "spread" not in build_tools.lower()


def test_swift_bounded_feature_emitters_use_open_interval_clamps():
    feature_math = _read("FXDataEngine/Sources/FXDataEngine/Features/FeatureMath.swift")
    feature_pipeline = _read("FXDataEngine/Sources/FXDataEngine/Features/FeaturePipeline.swift")
    feature_build = _read("FXDataEngine/Sources/FXDataEngine/Features/FeatureBuildTools.swift")
    event_macro = _read("FXDataEngine/Sources/FXDataEngine/Services/MacroEvents.swift")

    _assert_tokens(
        feature_math,
        [
            "self.upperWickNorm = fxClampUnit(upperWickNorm)",
            "self.lowerWickNorm = fxClampUnit(lowerWickNorm)",
        ],
    )
    _assert_tokens(
        feature_build,
        [
            "return fxClampUnit(1.0 - distance / radius)",
            "return fxClampSignedUnit(0.60 * asiaToEurope + 0.80 * europeToUS - 0.70 * usToRollover)",
            "return fxClampUnit(0.70 * overlap + 0.30 * cyclicHourPulse(hourValue: hour, centerHour: 15.0, radiusHours: 2.0))",
            "let tripleSwapBias = fxClampSignedUnit(dayBias * (0.35 + 0.65 * rollBias))",
        ],
    )
    _assert_tokens(
        feature_pipeline,
        [
            "features[12] = fxClampSignedUnit((contextAggregates.upRatio[index] - 0.5) * 2.0)",
            "features[15] = fxClampSignedUnit((Double(weekday) - 3.0) / 2.0)",
            "features[16] = fxClampSignedUnit((Double(hour) - 11.5) / 11.5)",
            "features[17] = fxClampSignedUnit((Double(minute) - 29.5) / 29.5)",
            "features[62] = fxClampSignedUnit(sharedUtility)",
            "features[63] = fxClampSignedUnit((sharedStability * 2.0) - 1.0)",
            "features[64] = fxClampSignedUnit((sharedLead * 2.0) - 1.0)",
            "features[65] = fxClampSignedUnit((sharedCoverage * 2.0) - 1.0)",
            "features[67] = fxClampSignedUnit(features[20] - features[19])",
        ],
    )
    _assert_tokens(
        event_macro,
        [
            "self.eventClassBias = fxClampSignedUnit(eventClassBias)",
            "self.ratesActivity = unitClamp(ratesActivity)",
            "let policyDivergence = fxClampSignedUnit(policyNorm - 0.35 * inflationNorm + 0.20 * growthNorm)",
            "let policyPressure = fxClampSignedUnit(0.70 * policyNorm + 0.30 * inflationNorm)",
            "let growthPressure = fxClampSignedUnit(0.78 * growthNorm + 0.22 * tradeNorm)",
            "let familyDiversity = fxClampUnit(Double(familyHits) / 5.0)",
            "let density = fxClampUnit(coverageWeight / 2.0)",
            "let stateQuality = fxClampUnit(",
            "features[FXDataEngineConstants.macroEventFeatureOffset + index] = vector[index]",
        ],
    )


def test_training_pipeline_uses_horizon_aware_fit_and_payload_transform():
    data_engine = _read("FXDataEngine/Sources/FXDataEngine/Core/FXDataEngine.swift")
    feature_pipeline = _read("FXDataEngine/Sources/FXDataEngine/Features/FeaturePipeline.swift")
    normalization_core = _read("FXDataEngine/Sources/FXDataEngine/Normalization/Normalization.swift")
    training_samples = _read("FXDataEngine/Sources/FXDataEngine/Lifecycle/TrainingSamples.swift")

    _assert_tokens(
        data_engine,
        [
            "let featureFrame = try featureCore.buildFrame(",
            "let normalizationFrame = try normalizationCore.buildInputFrame(",
            "from: featureFrame",
            "let payloadFrame = try normalizationCore.buildPayloadFrame(",
            "PreparedTrainingPayload(",
        ],
    )
    _assert_tokens(
        feature_pipeline,
        [
            "public func buildFrame(bundle: DataCoreBundle, request: FeatureCoreRequest? = nil) throws -> FeatureCoreFrame",
            "horizonMinutes: max(1, request?.horizonMinutes ?? 1)",
            "normalizationMethod: request?.normalizationMethod ?? .existing",
        ],
    )
    _assert_tokens(
        normalization_core,
        [
            "public func buildInputFrame(",
            "from featureFrame: FeatureCoreFrame",
            ") throws -> NormalizationCoreFrame",
            "public func buildPayloadFrame(_ request: NormalizationPayloadRequest) throws -> NormalizationPayloadFrame",
            "public func finalizePredictRequest(manifest: PluginManifestV4, request: PredictRequestV4) throws -> PredictRequestV4",
            "public func finalizeTrainRequest(manifest: PluginManifestV4, request: TrainRequestV4) throws -> TrainRequestV4",
        ],
    )
    _assert_tokens(
        training_samples,
        [
            "public struct PreparedSampleNormalizationCacheRequest",
            "public static func routedNormalizationSampleCacheRequests(",
            "public static func cachedPreparedSample(",
            "public static func cachedPreparedSampleWindow(",
        ],
    )


def test_warmup_normalization_fits_candidates_on_train_only_split():
    warmup = _read("FXDataEngine/Sources/FXDataEngine/Lifecycle/Warmup.swift")
    warmup_tests = _read("FXDataEngine/Tests/FXDataEngineTests/WarmupTests.swift")

    _assert_tokens(
        warmup,
        [
            "public static func normalizationCandidateSplit(",
            "let validationEnd = endIndex",
            "var purge = horizonMinutes + 240",
            "let trainingEnd = validationStart - purge - 1",
            "guard trainingEnd - trainingStart >= 100 else { return nil }",
        ],
    )
    _assert_tokens(
        warmup_tests,
        [
            "WarmupTools.normalizationCandidateSplit(horizonMinutes: 13, startIndex: 0, endIndex: 599)",
            "XCTAssertNil(WarmupTools.normalizationCandidateSplit(horizonMinutes: 13, startIndex: 0, endIndex: 239))",
        ],
    )


def test_runtime_artifacts_persist_normalization_fits():
    runtime_artifacts = _read("FXDataEngine/Sources/FXDataEngine/RuntimeArtifacts/RuntimeArtifacts.swift")
    runtime_sections = _read("FXDataEngine/Sources/FXDataEngine/RuntimeArtifacts/RuntimeArtifactSections.swift")
    runtime_tests = _read("FXDataEngine/Tests/FXDataEngineTests/RuntimeArtifactsTests.swift")
    normalization_tests = _read("FXDataEngine/Tests/FXDataEngineTests/NormalizationStateTests.swift")

    _assert_tokens(
        runtime_artifacts,
        [
            "public static let version = 15",
            "public var normalizationMethodCount: Int",
            "public var normalizationRollWindowMax: Int",
            "normalizationMethodCount: Int = FXDataEngineConstants.normMethodCount",
        ],
    )
    _assert_tokens(
        runtime_sections,
        [
            "case normalizationWindows",
            "case normalizationWindowConfig",
            "case normalizationHistory",
            "case normalizationFit",
            "public enum RuntimeNormalizationFitCodec",
            "for horizon in 0..<RuntimeArtifactConstants.maxHorizons",
            "for methodID in 0..<FXDataEngineConstants.normMethodCount",
        ],
    )
    _assert_tokens(
        runtime_tests,
        [
            "XCTAssertEqual(header.version, 15)",
            "XCTAssertEqual(header.normalizationMethodCount, 17)",
        ],
    )
    _assert_tokens(
        normalization_tests,
        [
            "func testNormalizationFitCodecRoundTripsLegacySection() throws",
            "let encoded = try RuntimeNormalizationFitCodec.encode(fit)",
            "let decoded = try RuntimeNormalizationFitCodec.decode(from: encoded)",
        ],
    )


def test_tooling_accepts_new_normalization_id_range():
    promotion = _read("FXDataEngine/Tools/OfflineLab/offline_lab/promotion.py")
    audit_run = _read("FXDataEngine/Tools/testlab/audit_run.py")
    assert '"AI_FeatureNormalization": (0, 0, 16, "N")' in promotion
    assert "||0||0||16||N" in audit_run
