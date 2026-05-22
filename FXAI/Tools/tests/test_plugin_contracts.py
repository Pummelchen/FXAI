from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _assert_tokens(text: str, tokens: list[str]) -> None:
    for token in tokens:
        assert token in text


def test_swift_plugin_protocol_covers_manifest_lifecycle_predict_and_train():
    contracts = _read("FXDataEngine/Sources/FXDataEngine/PluginContracts.swift")
    invocation = _read("FXDataEngine/Sources/FXDataEngine/PluginInvocation.swift")

    _assert_tokens(
        contracts,
        [
            "public protocol FXAIPluginV4: Sendable",
            "var manifest: PluginManifestV4 { get }",
            "mutating func reset()",
            "func selfTest() -> Bool",
            "mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws",
            "func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4",
        ],
    )
    _assert_tokens(
        invocation,
        [
            "public static func trainViaV4<Plugin: FXAIPluginV4>(",
            "public static func predictViaV4<Plugin: FXAIPluginV4>(",
            "try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)",
            "try plugin.train(request, hyperParameters: hyperParameters)",
            "return try plugin.predict(request, hyperParameters: hyperParameters)",
        ],
    )


def test_swift_plugin_manifest_and_context_contracts_are_volume_aware():
    contracts = _read("FXDataEngine/Sources/FXDataEngine/PluginContracts.swift")
    tests = _read("FXDataEngine/Tests/FXDataEngineTests/PluginContractTests.swift")

    _assert_tokens(
        contracts,
        [
            "public struct PluginManifestV4",
            "public var requiresVolumeWhenAvailable: Bool",
            "requiresVolumeWhenAvailable: Bool = true",
            "public struct PluginContextV4",
            "public var dataHasVolume: Bool",
            "public var priceCostPoints: Double",
            "public static func validateCompatibility(manifest: PluginManifestV4, context: PluginContextV4) throws",
        ],
    )
    _assert_tokens(
        tests,
        [
            "func testManifestValidationRequiresCoherentCapabilities() throws",
            "func testPredictRequestValidationChecksWindowContract() throws",
            "func testManifestContextCompatibilityMatchesLegacyWindowRules() throws",
            "dataHasVolume: true",
        ],
    )


def test_plugin_contract_suite_exercises_lifecycle_predict_state_and_synthetic_series_paths():
    suite = _read("FXDataEngine/Sources/FXDataEngine/PluginContractSuite.swift")
    suite_tests = _read("FXDataEngine/Tests/FXDataEngineTests/PluginContractSuiteTests.swift")

    _assert_tokens(
        suite,
        [
            "public protocol FXAIPluginPersistentState: FXAIPluginV4",
            "public protocol FXAIPluginSyntheticSeriesSupport: FXAIPluginV4",
            "mutating func saveStateData() throws -> Data",
            "mutating func loadStateData(_ data: Data) throws",
            "mutating func setSyntheticSeries(_ series: M1OHLCVSeries) throws",
            "mutating func clearSyntheticSeries()",
            "PluginContractSuiteTools.runSuite(",
            "persistent_state_roundtrip",
            "synthetic_series_contract",
        ],
    )
    _assert_tokens(
        suite_tests,
        [
            "ContractSuiteGoodPlugin: FXAIPluginPersistentState, FXAIPluginSyntheticSeriesSupport",
            "func testContractSuiteRunsAllCasesForValidPlugin()",
            "func testContractSuiteReportsRegistryAndPredictFailures()",
            "func testAuditPluginContractSelfTestMirrorsLegacyWrapper()",
        ],
    )


def test_audit_runner_can_gate_on_plugin_contract_sanity():
    suite = _read("FXDataEngine/Sources/FXDataEngine/PluginContractSuite.swift")
    tests = _read("FXDataEngine/Tests/FXDataEngineTests/PluginContractSuiteTests.swift")
    _assert_tokens(
        suite,
        [
            "public struct AuditPluginContractSelfTestResult",
            "public enum AuditPluginContractTools",
            "public static func selfTest(",
            "AuditPluginContractSelfTestResult(",
        ],
    )
    _assert_tokens(
        tests,
        [
            "let validResult = AuditPluginContractTools.selfTest(factories: [valid])",
            "XCTAssertTrue(validResult.passed)",
            "let invalidResult = AuditPluginContractTools.selfTest(factories: [invalid])",
            "XCTAssertFalse(invalidResult.passed)",
        ],
    )
