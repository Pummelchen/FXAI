import XCTest
@testable import FXDataEngine

final class LifecycleBootstrapTests: XCTestCase {
    func testBootstrapProbeBuildsLegacyDummyPredictRequestAndWindow() throws {
        let manifest = PluginManifestV4(
            aiID: 4,
            aiName: "WindowProbe",
            family: .transformer,
            capabilityMask: [.selfTest, .windowContext],
            minHorizonMinutes: 3,
            maxHorizonMinutes: 20,
            minSequenceBars: 4,
            maxSequenceBars: 16
        )

        let probe = try LifecycleBootstrapTools.buildProbe(
            manifest: manifest,
            symbol: "EURUSD",
            sampleTimeUTC: 1_800_018_000,
            normalizationMethod: .existing,
            pointValue: 0.0001,
            dataHasVolume: true
        )

        XCTAssertEqual(probe.aiID, 4)
        XCTAssertTrue(probe.requiresPredict)
        XCTAssertTrue(probe.requiresSelfTest)
        XCTAssertEqual(probe.predictRequest.context.apiVersion, FXDataEngineConstants.latestPluginAPIVersion)
        XCTAssertEqual(probe.predictRequest.context.regimeID, 0)
        XCTAssertEqual(probe.predictRequest.context.sessionBucket, 3)
        XCTAssertEqual(probe.predictRequest.context.horizonMinutes, 5)
        XCTAssertEqual(probe.predictRequest.context.sequenceBars, 16)
        XCTAssertEqual(probe.predictRequest.context.featureSchema, manifest.featureSchema)
        XCTAssertEqual(probe.predictRequest.context.normalizationMethod, .existing)
        XCTAssertEqual(probe.predictRequest.context.pointValue, 0.0001, accuracy: 1e-12)
        XCTAssertEqual(probe.predictRequest.context.domainHash, PluginContractTools.symbolHash01("EURUSD"), accuracy: 1e-12)
        XCTAssertTrue(probe.predictRequest.context.dataHasVolume)

        XCTAssertEqual(probe.predictRequest.x.count, FXDataEngineConstants.aiWeights)
        XCTAssertEqual(probe.predictRequest.x[0], 1.0, accuracy: 0.0)
        XCTAssertEqual(probe.predictRequest.x.dropFirst().reduce(0.0, +), 0.0, accuracy: 0.0)
        XCTAssertEqual(probe.predictRequest.windowSize, 15)
        XCTAssertEqual(probe.predictRequest.xWindow.count, 15)
        XCTAssertEqual(probe.rawComplianceWindow[0][0], 0.92, accuracy: 1e-12)
        XCTAssertEqual(probe.rawComplianceWindow[14][0], 0.30, accuracy: 1e-12)
        XCTAssertEqual(probe.predictRequest.xWindow[0][0], 1.0, accuracy: 1e-12)
        XCTAssertNoThrow(try probe.predictRequest.validate())
    }

    func testBootstrapValidationPlanFailsFastLikeLegacyValidator() {
        let manifest0 = PluginManifestV4(aiID: 0, aiName: "First", family: .linear)
        let manifest1 = PluginManifestV4(aiID: 1, aiName: "Second", family: .tree)

        let featureRegistryFail = LifecycleBootstrapTools.buildValidationPlan(
            featureRegistrySelfTestPassed: false,
            manifests: [manifest0, manifest1],
            symbol: "EURUSD",
            sampleTimeUTC: 1_800_018_000,
            expectedPluginCount: 2
        )
        XCTAssertFalse(featureRegistryFail.valid)
        XCTAssertEqual(featureRegistryFail.reason, "feature_registry_self_test")
        XCTAssertTrue(featureRegistryFail.probes.isEmpty)

        let missing = LifecycleBootstrapTools.buildValidationPlan(
            featureRegistrySelfTestPassed: true,
            manifests: [manifest0],
            symbol: "EURUSD",
            sampleTimeUTC: 1_800_018_000,
            expectedPluginCount: 2
        )
        XCTAssertFalse(missing.valid)
        XCTAssertEqual(missing.reason, "plugin_missing:1")
        XCTAssertEqual(missing.probes.count, 1)
        XCTAssertFalse(missing.probes[0].predictRequest.context.dataHasVolume)
    }

    func testBootstrapValidationPlanBuildsOneProbePerExpectedPlugin() {
        let manifests = [
            PluginManifestV4(aiID: 0, aiName: "First", family: .linear),
            PluginManifestV4(aiID: 1, aiName: "Second", family: .tree)
        ]

        let plan = LifecycleBootstrapTools.buildValidationPlan(
            featureRegistrySelfTestPassed: true,
            manifests: manifests,
            symbol: "USDJPY",
            sampleTimeUTC: 0,
            pointValue: 0.0,
            dataHasVolume: false,
            expectedPluginCount: 2
        )

        XCTAssertTrue(plan.valid)
        XCTAssertEqual(plan.reason, "")
        XCTAssertEqual(plan.probes.map(\.aiID), [0, 1])
        XCTAssertEqual(plan.probes[0].predictRequest.context.sessionBucket, 0)
        XCTAssertEqual(plan.probes[0].predictRequest.context.pointValue, 1.0, accuracy: 0.0)
        XCTAssertFalse(plan.probes[0].predictRequest.context.dataHasVolume)
    }

    func testBootstrapExecutionRunsPredictValidationAndSelfTest() {
        let manifests = [
            PluginManifestV4(aiID: 0, aiName: "First", family: .linear),
            PluginManifestV4(aiID: 1, aiName: "Second", family: .tree)
        ]
        let plan = LifecycleBootstrapTools.buildValidationPlan(
            featureRegistrySelfTestPassed: true,
            manifests: manifests,
            symbol: "EURUSD",
            sampleTimeUTC: 1_800_018_000,
            dataHasVolume: true,
            expectedPluginCount: 2
        )

        let result = LifecycleBootstrapTools.executeValidationPlan(
            plan,
            plugins: manifests.map { manifest in
                LifecycleBootstrapPluginFactory(
                    makePlugin: { BootstrapValidatingPlugin(manifest: manifest, requireVolume: true) }
                )
            }
        )

        XCTAssertTrue(result.valid)
        XCTAssertEqual(result.reason, "")
        XCTAssertEqual(result.probes.map(\.aiID), [0, 1])
        XCTAssertTrue(result.probes.allSatisfy(\.manifestMatched))
        XCTAssertTrue(result.probes.allSatisfy(\.requestValidated))
        XCTAssertTrue(result.probes.allSatisfy(\.predictSucceeded))
        XCTAssertTrue(result.probes.allSatisfy(\.predictionValidated))
        XCTAssertTrue(result.probes.allSatisfy(\.selfTestPassed))
    }

    func testBootstrapExecutionFailsFastOnMissingPlugin() {
        let manifest = PluginManifestV4(aiID: 0, aiName: "First", family: .linear)
        let plan = LifecycleBootstrapTools.buildValidationPlan(
            featureRegistrySelfTestPassed: true,
            manifests: [manifest],
            symbol: "EURUSD",
            sampleTimeUTC: 1_800_018_000,
            expectedPluginCount: 1
        )

        let result = LifecycleBootstrapTools.executeValidationPlan(plan, plugins: [])

        XCTAssertFalse(result.valid)
        XCTAssertEqual(result.reason, "plugin_missing:0")
        XCTAssertEqual(result.probes.count, 1)
        XCTAssertFalse(result.probes[0].valid)
        XCTAssertEqual(result.probes[0].reason, "plugin_missing:0")
    }

    func testBootstrapExecutionNormalizesInvalidPlanReason() {
        let result = LifecycleBootstrapTools.executeValidationPlan(
            LifecycleBootstrapPlan(valid: false),
            plugins: []
        )

        XCTAssertFalse(result.valid)
        XCTAssertEqual(result.reason, "bootstrap_plan_invalid")
        XCTAssertTrue(result.probes.isEmpty)
    }

    func testBootstrapExecutionRejectsDuplicatePluginFactoryIDs() {
        let manifest = PluginManifestV4(aiID: 0, aiName: "First", family: .linear)
        let plan = LifecycleBootstrapTools.buildValidationPlan(
            featureRegistrySelfTestPassed: true,
            manifests: [manifest],
            symbol: "EURUSD",
            sampleTimeUTC: 1_800_018_000,
            expectedPluginCount: 1
        )
        let factory = LifecycleBootstrapPluginFactory(
            makePlugin: { BootstrapValidatingPlugin(manifest: manifest, requireVolume: false) }
        )

        let result = LifecycleBootstrapTools.executeValidationPlan(
            plan,
            plugins: [factory, factory]
        )

        XCTAssertFalse(result.valid)
        XCTAssertEqual(result.reason, "plugin_duplicate:0")
        XCTAssertTrue(result.probes.isEmpty)
    }

    func testBootstrapExecutionRejectsInvalidPrediction() {
        let manifest = PluginManifestV4(aiID: 0, aiName: "BadPrediction", family: .linear)
        let plan = LifecycleBootstrapTools.buildValidationPlan(
            featureRegistrySelfTestPassed: true,
            manifests: [manifest],
            symbol: "EURUSD",
            sampleTimeUTC: 1_800_018_000,
            expectedPluginCount: 1
        )

        let result = LifecycleBootstrapTools.executeValidationPlan(
            plan,
            plugins: [
                LifecycleBootstrapPluginFactory(
                    makePlugin: { BootstrapInvalidPredictionPlugin(manifest: manifest) }
                )
            ]
        )

        XCTAssertFalse(result.valid)
        XCTAssertTrue(result.reason.hasPrefix("prediction_invalid:0:"))
        XCTAssertEqual(result.probes.count, 1)
        XCTAssertTrue(result.probes[0].predictSucceeded)
        XCTAssertFalse(result.probes[0].predictionValidated)
        XCTAssertFalse(result.probes[0].selfTestPassed)
    }

    func testBootstrapExecutionRejectsSelfTestFailure() {
        let manifest = PluginManifestV4(aiID: 0, aiName: "SelfTestFailure", family: .linear)
        let plan = LifecycleBootstrapTools.buildValidationPlan(
            featureRegistrySelfTestPassed: true,
            manifests: [manifest],
            symbol: "EURUSD",
            sampleTimeUTC: 1_800_018_000,
            expectedPluginCount: 1
        )

        let result = LifecycleBootstrapTools.executeValidationPlan(
            plan,
            plugins: [
                LifecycleBootstrapPluginFactory(
                    makePlugin: { BootstrapSelfTestFailurePlugin(manifest: manifest) }
                )
            ]
        )

        XCTAssertFalse(result.valid)
        XCTAssertEqual(result.reason, "self_test_failed:0")
        XCTAssertEqual(result.probes.count, 1)
        XCTAssertTrue(result.probes[0].predictionValidated)
        XCTAssertFalse(result.probes[0].selfTestPassed)
    }

    func testBootstrapExecutionRejectsManifestMismatch() {
        let expected = PluginManifestV4(aiID: 0, aiName: "Expected", family: .linear)
        let actual = PluginManifestV4(aiID: 0, aiName: "Actual", family: .linear)
        let plan = LifecycleBootstrapTools.buildValidationPlan(
            featureRegistrySelfTestPassed: true,
            manifests: [expected],
            symbol: "EURUSD",
            sampleTimeUTC: 1_800_018_000,
            expectedPluginCount: 1
        )

        let result = LifecycleBootstrapTools.executeValidationPlan(
            plan,
            plugins: [
                LifecycleBootstrapPluginFactory(
                    manifest: expected,
                    makePlugin: { BootstrapValidatingPlugin(manifest: actual, requireVolume: false) }
                )
            ]
        )

        XCTAssertFalse(result.valid)
        XCTAssertEqual(result.reason, "manifest_mismatch:0")
        XCTAssertEqual(result.probes.count, 1)
        XCTAssertFalse(result.probes[0].manifestMatched)
    }
}

private struct BootstrapValidatingPlugin: FXAIPluginV4 {
    let manifest: PluginManifestV4
    let requireVolume: Bool

    mutating func reset() {}

    func selfTest() -> Bool {
        (try? manifest.validate()) != nil
    }

    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {}

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        if requireVolume, !request.context.dataHasVolume {
            throw FXDataEngineError.validation("expected volume-aware bootstrap request")
        }
        return PredictionV4(
            classProbabilities: [0.2, 0.2, 0.6],
            moveMeanPoints: 0.5,
            moveQ25Points: 0.2,
            moveQ50Points: 0.5,
            moveQ75Points: 0.8,
            mfeMeanPoints: 0.7,
            maeMeanPoints: 0.3,
            hitTimeFraction: 0.5,
            pathRisk: 0.2,
            fillRisk: request.context.dataHasVolume ? 0.1 : 0.0,
            confidence: 0.7,
            reliability: 0.8
        )
    }
}

private struct BootstrapInvalidPredictionPlugin: FXAIPluginV4 {
    let manifest: PluginManifestV4

    mutating func reset() {}

    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {}

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        return PredictionV4(classProbabilities: [0.7, 0.7, 0.0])
    }
}

private struct BootstrapSelfTestFailurePlugin: FXAIPluginV4 {
    let manifest: PluginManifestV4

    mutating func reset() {}

    func selfTest() -> Bool {
        false
    }

    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {}

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        return PredictionV4(
            classProbabilities: [0.1, 0.1, 0.8],
            moveMeanPoints: 0.2,
            moveQ25Points: 0.1,
            moveQ50Points: 0.2,
            moveQ75Points: 0.3,
            confidence: 0.5,
            reliability: 0.5
        )
    }
}
