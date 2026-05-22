import XCTest
@testable import FXDataEngine

final class RuntimePerformanceTests: XCTestCase {
    func testRuntimeStageNamesMatchLegacyEnumNames() {
        XCTAssertEqual(RuntimePerformanceState.stageName(stageID: 0), "total")
        XCTAssertEqual(RuntimePerformanceState.stageName(stageID: 1), "feature_pipeline")
        XCTAssertEqual(RuntimePerformanceState.stageName(stageID: 6), "control_plane")
        XCTAssertEqual(RuntimePerformanceState.stageName(stageID: 99), "unknown")
    }

    func testRuntimePerformanceBlendAndBudgetPressureMatchLegacyRules() {
        XCTAssertEqual(RuntimePerformanceState.blend(previous: 10.0, value: 20.0, observations: 0), 20.0)
        XCTAssertEqual(RuntimePerformanceState.blend(previous: 10.0, value: 20.0, observations: 1), 11.2, accuracy: 1e-12)

        var state = RuntimePerformanceState()
        XCTAssertEqual(state.budgetPressure(budgetMS: 10.0), 0.0)
        state.recordStage(.total, elapsedMS: 15.0, sampleTimeUTC: 1_704_067_200)
        XCTAssertTrue(state.ready)
        XCTAssertEqual(state.lastTimeUTC, 1_704_067_200)
        XCTAssertEqual(state.budgetPressure(budgetMS: 10.0), 0.5, accuracy: 1e-12)

        state.recordStage(.total, elapsedMS: 5.0)
        XCTAssertEqual(state.stageMeanMS[RuntimeStage.total.rawValue], 13.8, accuracy: 1e-12)
        XCTAssertEqual(state.stageMaxMS[RuntimeStage.total.rawValue], 15.0, accuracy: 1e-12)
        XCTAssertEqual(state.budgetPressure(budgetMS: 10.0), 0.38, accuracy: 1e-12)
    }

    func testPluginPerformanceAndWorkingSetRows() {
        var state = RuntimePerformanceState()
        state.setActiveModels(3)
        state.recordStage(.router, elapsedMS: 1.25)
        state.recordPluginPredict(aiID: 4, elapsedMS: 2.0)
        state.recordPluginPredict(aiID: 4, elapsedMS: 4.0)
        state.recordPluginUpdate(aiID: 4, elapsedMS: -5.0)
        state.setPluginWorkingSetKB(aiID: 4, workingSetKB: 10.0)
        state.setPluginWorkingSetKB(aiID: 4, workingSetKB: 8.0)

        let manifest = PluginManifestV4(
            aiID: 4,
            aiName: "PerfPlugin",
            family: .linear,
            capabilityMask: [.selfTest]
        )
        let stageRows = state.stageManifestRows()
        let pluginRows = state.pluginManifestRows(manifests: [manifest])

        XCTAssertEqual(stageRows.count, 1)
        XCTAssertEqual(stageRows[0].stageName, "router")
        XCTAssertEqual(stageRows[0].activeModels, 3)
        XCTAssertEqual(pluginRows.count, 1)
        XCTAssertEqual(pluginRows[0].predictMeanMS ?? -1.0, 2.24, accuracy: 1e-12)
        XCTAssertEqual(pluginRows[0].predictMaxMS ?? -1.0, 4.0, accuracy: 1e-12)
        XCTAssertEqual(pluginRows[0].updateMeanMS ?? -1.0, 0.0, accuracy: 1e-12)
        XCTAssertEqual(pluginRows[0].workingSetKB ?? -1.0, 10.0, accuracy: 1e-12)
    }

    func testWorkingSetEstimateUsesSequenceAndCapabilities() {
        let manifest = PluginManifestV4(
            aiID: 5,
            aiName: "StatefulWindow",
            family: .transformer,
            capabilityMask: [.selfTest, .stateful, .windowContext, .multiHorizon]
        )
        let payloadBytes = Double(FXDataEngineConstants.aiWeights) * 8.0
        let expected = (payloadBytes + payloadBytes * 10.0) * 2.30 / 1024.0
        XCTAssertEqual(
            RuntimePerformanceState.estimatePluginWorkingSetKB(manifest: manifest, sequenceBars: 10),
            expected,
            accuracy: 1e-12
        )
        XCTAssertGreaterThanOrEqual(
            RuntimePerformanceState.estimatePluginWorkingSetKB(manifest: manifest, sequenceBars: 0),
            1.0
        )
    }
}
