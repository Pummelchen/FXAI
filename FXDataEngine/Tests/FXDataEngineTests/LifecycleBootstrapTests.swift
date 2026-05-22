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
        XCTAssertEqual(probe.predictRequest.context.apiVersion, FXDataEngineConstants.apiVersionV4)
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
}
