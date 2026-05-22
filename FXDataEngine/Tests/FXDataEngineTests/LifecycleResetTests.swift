import XCTest
@testable import FXDataEngine

final class LifecycleResetTests: XCTestCase {
    func testLifecycleResetPlanMatchesLegacyFailClosedSignalDefaultsAndActions() {
        let plan = LifecycleResetTools.buildResetPlan(
            symbol: "EURUSD",
            aiWarmupEnabled: true,
            horizonListRaw: "{5;1|13,5}",
            predictionTargetMinutes: 8,
            pluginsReady: true,
            maxContextSymbols: 3,
            aiCount: 4
        )

        XCTAssertEqual(plan.symbol, "EURUSD")
        XCTAssertEqual(plan.lastSignalBarUTC, 0)
        XCTAssertEqual(plan.lastSignal, -1)
        XCTAssertEqual(plan.lastSignalKey, -1)
        XCTAssertEqual(plan.lastReason, "reset")
        XCTAssertFalse(plan.warmupDone)
        XCTAssertEqual(plan.configuredHorizons, [1, 5, 8, 13])

        XCTAssertEqual(plan.signalCache.expectedMovePoints, 0.0, accuracy: 0.0)
        XCTAssertEqual(plan.signalCache.tradeEdgePoints, 0.0, accuracy: 0.0)
        XCTAssertEqual(plan.signalCache.pathRisk, 1.0, accuracy: 0.0)
        XCTAssertEqual(plan.signalCache.fillRisk, 1.0, accuracy: 0.0)
        XCTAssertEqual(plan.signalCache.probabilityCalibrationMethod, "LOGISTIC_AFFINE")
        XCTAssertEqual(plan.signalCache.probabilityCalibrationSkipProbability, 1.0, accuracy: 0.0)
        XCTAssertEqual(plan.signalCache.executionQualityMethod, "SCORECARD_V1")
        XCTAssertEqual(plan.signalCache.executionQualityState, "UNKNOWN")

        XCTAssertEqual(plan.actions.first, .resetModelHyperParams)
        XCTAssertTrue(plan.actions.contains(.resetAllPlugins))
        XCTAssertTrue(plan.actions.contains(.loadMetaArtifacts))
        XCTAssertTrue(plan.actions.contains(.loadRuntimeArtifacts))
        XCTAssertEqual(plan.actions.filter { $0 == .resetModelAuxState }.count, 1)
        XCTAssertTrue(plan.resetPlugins)
    }

    func testLifecycleResetPlanClearsContextUtilityAndAITrainState() {
        let plan = LifecycleResetTools.buildResetPlan(
            symbol: "GBPUSD",
            aiWarmupEnabled: false,
            horizonListRaw: "",
            predictionTargetMinutes: 30,
            maxContextSymbols: 2,
            aiCount: 3
        )

        XCTAssertTrue(plan.warmupDone)
        XCTAssertEqual(plan.configuredHorizons, [30])
        XCTAssertEqual(plan.contextUtility.count, 2)
        for row in plan.contextUtility {
            XCTAssertEqual(row.utility, 0.0, accuracy: 0.0)
            XCTAssertEqual(row.stability, 0.0, accuracy: 0.0)
            XCTAssertEqual(row.lead, 0.0, accuracy: 0.0)
            XCTAssertEqual(row.coverage, 0.0, accuracy: 0.0)
            XCTAssertFalse(row.ready)
        }

        XCTAssertEqual(plan.aiStates.map(\.aiID), [0, 1, 2])
        for state in plan.aiStates {
            XCTAssertFalse(state.trained)
            XCTAssertEqual(state.lastTrainBarUTC, 0)
            XCTAssertTrue(state.auxResetRequired)
        }
    }

    func testLifecycleResetPlanResolvesNormalizationWindowsLikeLegacyReset() {
        let cold = LifecycleResetTools.buildResetPlan(
            symbol: "USDJPY",
            aiWarmupEnabled: false,
            horizonListRaw: "",
            predictionTargetMinutes: 30,
            normalizationWindowsReady: false
        )
        XCTAssertEqual(cold.normalizationDefaultWindow, 256)
        XCTAssertEqual(cold.normalizationFeatureWindows.count, FXDataEngineConstants.aiFeatures)
        XCTAssertEqual(Set(cold.normalizationFeatureWindows), [256])

        var existing = Array(repeating: 96, count: FXDataEngineConstants.aiFeatures)
        existing[0] = 4
        existing[20] = 5000
        let ready = LifecycleResetTools.buildResetPlan(
            symbol: "USDJPY",
            aiWarmupEnabled: false,
            horizonListRaw: "",
            predictionTargetMinutes: 1,
            normalizationWindowsReady: true,
            existingNormalizationFeatureWindows: existing,
            existingNormalizationDefaultWindow: 4
        )
        XCTAssertEqual(ready.normalizationDefaultWindow, NormalizationWindowTools.minimumWindow)
        XCTAssertEqual(ready.normalizationFeatureWindows[0], NormalizationWindowTools.minimumWindow)
        XCTAssertEqual(ready.normalizationFeatureWindows[1], 96)
        XCTAssertEqual(ready.normalizationFeatureWindows[20], NormalizationWindowTools.maximumWindow)
    }
}
