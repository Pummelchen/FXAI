from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _assert_tokens(text: str, tokens: list[str]) -> None:
    for token in tokens:
        assert token in text


def test_swift_data_pipeline_exposes_dedicated_cores_and_contracts():
    pipeline = _read("FXDataEngine/Sources/FXDataEngine/Core/FXDataEngine.swift")
    _assert_tokens(
        pipeline,
        [
            "public struct FXDataEnginePipeline: Sendable",
            "public let dataCore: DataCore",
            "public let featureCore: FeatureCore",
            "public let normalizationCore: NormalizationCore",
            "public func preparePredictPayload(",
            "public func prepareTrainPayload(",
            "public func prepareTrainingDataset(",
        ],
    )


def test_swift_core_pipeline_contracts_define_stage_structures():
    data_core = _read("FXDataEngine/Sources/FXDataEngine/Core/DataCore.swift")
    feature_core = _read("FXDataEngine/Sources/FXDataEngine/Features/FeaturePipeline.swift")
    normalization_core = _read("FXDataEngine/Sources/FXDataEngine/Normalization/Normalization.swift")
    training_samples = _read("FXDataEngine/Sources/FXDataEngine/Lifecycle/TrainingSamples.swift")
    market_data = _read("FXDataEngine/Sources/FXDataEngine/MarketData/MarketData.swift")

    _assert_tokens(
        data_core,
        [
            "public struct DataCoreRequest",
            "public struct DataCoreBundle",
            "public struct DataCoreContextAggregates",
            "public struct DataCore: Sendable",
        ],
    )
    _assert_tokens(
        feature_core,
        [
            "public struct FeatureCoreRequest",
            "public struct FeatureCoreFrame",
            "public struct FeatureCore: Sendable",
        ],
    )
    _assert_tokens(
        normalization_core,
        [
            "public struct NormalizationCoreFrame",
            "public struct NormalizationPayloadRequest",
            "public struct NormalizationPayloadFrame",
            "public struct NormalizationCore: Sendable",
        ],
    )
    _assert_tokens(
        training_samples,
        [
            "public struct PreparedTrainingSample",
            "public struct PreparedTrainingPayload",
            "public struct PreparedTrainingDataset",
        ],
    )
    _assert_tokens(
        market_data,
        [
            "public struct M1OHLCVSeries: Sendable",
            "public struct MarketUniverse: Sendable",
        ],
    )


def test_swift_core_stage_files_expose_unified_api():
    data_core = _read("FXDataEngine/Sources/FXDataEngine/Core/DataCore.swift")
    feature_core = _read("FXDataEngine/Sources/FXDataEngine/Features/FeaturePipeline.swift")
    normalization_core = _read("FXDataEngine/Sources/FXDataEngine/Normalization/Normalization.swift")

    _assert_tokens(
        data_core,
        [
            "public mutating func addContextSymbol(",
            "public mutating func captureContextSymbols(",
            "public func buildBundle(request: DataCoreRequest, universe: MarketUniverse) throws -> DataCoreBundle",
            "public static func build(",
            "buildAlignedIndexMap(",
        ],
    )
    _assert_tokens(
        feature_core,
        [
            "public func buildFrame(bundle: DataCoreBundle, request: FeatureCoreRequest? = nil) throws -> FeatureCoreFrame",
            "public func buildFeatureVector(bundle: DataCoreBundle, sampleIndex: Int? = nil) -> [Double]",
            "public static func hasUsableVolume(_ universe: MarketUniverse) -> Bool",
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


def test_runtime_training_and_audit_paths_use_swift_pipeline_cores():
    pipeline = _read("FXDataEngine/Sources/FXDataEngine/Core/FXDataEngine.swift")
    audit_samples = _read("FXDataEngine/Sources/FXDataEngine/Audit/AuditSamples.swift")
    audit_runner = _read("FXDataEngine/Sources/FXDataEngine/Audit/AuditRunner.swift")
    lifecycle_bootstrap = _read("FXDataEngine/Sources/FXDataEngine/Lifecycle/LifecycleBootstrap.swift")

    _assert_tokens(
        pipeline,
        [
            "let dataBundle = try dataCore.buildBundle(request: dataRequest, universe: universe)",
            "let featureFrame = try featureCore.buildFrame(",
            "let normalizationFrame = try normalizationCore.buildInputFrame(",
            "from: featureFrame",
            "let payloadFrame = try normalizationCore.buildPayloadFrame(",
            "return PreparedTrainingPayload(",
        ],
    )
    _assert_tokens(
        audit_samples,
        [
            "public enum AuditSampleTools",
            "public static func marketUniverse(",
            "public static func buildSample(",
            "let payload = try pipeline.prepareTrainPayload(",
        ],
    )
    _assert_tokens(
        audit_runner,
        [
            "public struct AuditRunnerConfiguration",
            "public enum AuditRunnerTools",
            "AuditSampleTools.buildSample(",
        ],
    )
    _assert_tokens(
        lifecycle_bootstrap,
                [
                    "public struct LifecycleBootstrapProbe",
                    "public struct LifecycleBootstrapPlan",
                    "public static func buildProbe(",
                ],
            )


def test_mql_shortcut_paths_are_removed_from_swift_pipeline_callers():
    forbidden = [
        "FXAI_LoadSeriesOptionalCached",
        "FXAI_LoadRatesOptional",
        "FXAI_UpdateRatesRolling",
        "FXAI_ComputeFeatureVector",
        "FXAI_ApplyPayloadTransformPipelineEx",
    ]
    swift_files = list((ROOT / "FXDataEngine/Sources/FXDataEngine").rglob("*.swift"))
    for path in swift_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.relative_to(ROOT).as_posix()} contains retired MQL shortcut {token}"


def test_audit_harness_uses_native_swift_context_contracts():
    audit_context = _read("FXDataEngine/Sources/FXDataEngine/Audit/AuditContextSeries.swift")
    audit_suite = _read("FXDataEngine/Sources/FXDataEngine/Audit/AuditSuite.swift")
    audit_utilities = _read("FXDataEngine/Sources/FXDataEngine/Audit/AuditUtilities.swift")

    _assert_tokens(
        audit_context,
        [
            "public struct AuditAsSeriesOHLCV",
            "public struct AuditContextFeatureSet",
            "public enum AuditContextSeriesTools",
        ],
    )
    _assert_tokens(
        audit_suite,
        [
            "public struct AuditSuiteConfiguration",
            "public struct AuditPluginFactory",
            "public enum AuditSuiteTools",
            "public static func runSuite(",
        ],
    )
    _assert_tokens(
        audit_utilities,
        [
            "public enum AuditUtilityTools",
            "public static func sanitizeNormalizationMethod(",
            "public static func noSpreadStaticRegimeID(",
        ],
    )
