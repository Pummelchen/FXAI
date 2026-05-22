import XCTest
@testable import FXDataEngine

final class RuntimeCycleTests: XCTestCase {
    func testCyclePlanClampsLegacyRuntimeInputsAndContinues() {
        let plan = RuntimeCycleTools.planCycle(RuntimeCycleInput(
            symbol: "EURUSD",
            pluginsReady: true,
            predictionTargetMinutes: 0,
            aiWindow: 10,
            onlineSamples: 999,
            onlineEpochs: 0,
            trainEpochs: 99,
            aiType: 999,
            ensembleMode: false,
            ensembleAgreePercent: 120.0,
            buyThreshold: 0.40,
            sellThreshold: 0.60,
            evThresholdPoints: -5.0,
            evLookbackSamples: 500,
            lastSymbol: "GBPUSD",
            warmupEnabled: false,
            warmupDone: false,
            signalBarUTC: 123_456
        ))

        XCTAssertEqual(plan.action, .continueStages)
        XCTAssertEqual(plan.reason, "continue")
        XCTAssertTrue(plan.requiresStateReset)
        XCTAssertFalse(plan.publishIdleSnapshot)
        XCTAssertEqual(plan.signalBarUTC, 123_456)
        XCTAssertNil(plan.returnedSignal)

        XCTAssertEqual(plan.settings.baseHorizonMinutes, 1)
        XCTAssertEqual(plan.settings.windowBars, 50)
        XCTAssertEqual(plan.settings.onlineSamples, 200)
        XCTAssertEqual(plan.settings.onlineEpochs, 1)
        XCTAssertEqual(plan.settings.trainEpochs, 20)
        XCTAssertEqual(plan.settings.aiType, WarmupTools.sgdLogitAIID)
        XCTAssertFalse(plan.settings.ensembleMode)
        XCTAssertEqual(plan.settings.agreePercent, 100.0, accuracy: 0.0)
        XCTAssertEqual(plan.settings.thresholds, WarmupThresholdPair(buy: 0.50, sell: 0.49))
        XCTAssertEqual(plan.settings.evThresholdPoints, 0.0, accuracy: 0.0)
        XCTAssertEqual(plan.settings.evLookbackSamples, 400)
        XCTAssertEqual(plan.settings.decisionKey, WarmupTools.sgdLogitAIID)
    }

    func testCyclePlanFailsClosedWhenPluginsAreNotReadyBeforeResetGate() {
        let plan = RuntimeCycleTools.planCycle(RuntimeCycleInput(
            symbol: "EURUSD",
            pluginsReady: false,
            predictionTargetMinutes: 60,
            aiWindow: 100,
            onlineSamples: 20,
            onlineEpochs: 2,
            trainEpochs: 4,
            aiType: 6,
            ensembleMode: false,
            ensembleAgreePercent: 55.0,
            buyThreshold: 0.70,
            sellThreshold: 0.30,
            evThresholdPoints: 1.0,
            evLookbackSamples: 40,
            lastSymbol: "GBPUSD",
            signalBarUTC: 123_456
        ))

        XCTAssertEqual(plan.action, .publishIdle)
        XCTAssertEqual(plan.reason, "plugins_not_ready")
        XCTAssertFalse(plan.requiresStateReset)
        XCTAssertTrue(plan.publishIdleSnapshot)
        XCTAssertEqual(plan.returnedSignal, -1)
    }

    func testCyclePlanPreservesLegacyWarmupAndBarTimeGateOrder() {
        let warmup = RuntimeCycleTools.planCycle(RuntimeCycleInput(
            symbol: "EURUSD",
            predictionTargetMinutes: 60,
            aiWindow: 100,
            onlineSamples: 20,
            onlineEpochs: 2,
            trainEpochs: 4,
            aiType: 6,
            ensembleMode: false,
            ensembleAgreePercent: 55.0,
            buyThreshold: 0.70,
            sellThreshold: 0.30,
            evThresholdPoints: 1.0,
            evLookbackSamples: 40,
            warmupEnabled: true,
            warmupDone: false,
            signalBarUTC: 0
        ))
        XCTAssertEqual(warmup.action, .publishIdle)
        XCTAssertEqual(warmup.reason, "warmup_pending")
        XCTAssertTrue(warmup.requiresStateReset)
        XCTAssertEqual(warmup.returnedSignal, -1)

        let missingBar = RuntimeCycleTools.planCycle(RuntimeCycleInput(
            symbol: "EURUSD",
            predictionTargetMinutes: 60,
            aiWindow: 100,
            onlineSamples: 20,
            onlineEpochs: 2,
            trainEpochs: 4,
            aiType: 6,
            ensembleMode: false,
            ensembleAgreePercent: 55.0,
            buyThreshold: 0.70,
            sellThreshold: 0.30,
            evThresholdPoints: 1.0,
            evLookbackSamples: 40,
            lastSymbol: "EURUSD",
            warmupEnabled: false,
            warmupDone: false,
            signalBarUTC: 0
        ))
        XCTAssertEqual(missingBar.action, .publishIdle)
        XCTAssertEqual(missingBar.reason, "bar_time_failed")
        XCTAssertFalse(missingBar.requiresStateReset)
        XCTAssertEqual(missingBar.returnedSignal, -1)
    }

    func testCyclePlanRestoresCachedSignalWithLegacyDecisionKey() {
        let decisionKey = RuntimeCycleTools.decisionKey(
            ensembleMode: true,
            aiType: 6,
            agreePercent: 72.34
        )
        XCTAssertEqual(decisionKey, 100_723)
        XCTAssertEqual(
            RuntimeCycleTools.decisionKey(ensembleMode: true, aiType: 6, agreePercent: .nan),
            100_500
        )

        let plan = RuntimeCycleTools.planCycle(RuntimeCycleInput(
            symbol: "EURUSD",
            predictionTargetMinutes: 60,
            aiWindow: 100,
            onlineSamples: 20,
            onlineEpochs: 2,
            trainEpochs: 4,
            aiType: 6,
            ensembleMode: true,
            ensembleAgreePercent: 72.34,
            buyThreshold: 0.70,
            sellThreshold: 0.30,
            evThresholdPoints: 1.0,
            evLookbackSamples: 40,
            lastSymbol: "EURUSD",
            warmupEnabled: true,
            warmupDone: true,
            signalBarUTC: 123_456,
            lastSignalBarUTC: 123_456,
            lastSignalDecisionKey: decisionKey,
            lastSignal: 1
        ))

        XCTAssertEqual(plan.action, .restoreCachedSignal)
        XCTAssertEqual(plan.reason, "signal_cache_hit")
        XCTAssertFalse(plan.publishIdleSnapshot)
        XCTAssertEqual(plan.signalBarUTC, 123_456)
        XCTAssertEqual(plan.returnedSignal, 1)
        XCTAssertEqual(plan.settings.decisionKey, 100_723)
    }

    func testCyclePlanDoesNotRestoreCacheAfterSymbolReset() {
        let decisionKey = RuntimeCycleTools.decisionKey(
            ensembleMode: false,
            aiType: 6,
            agreePercent: 60.0
        )
        let plan = RuntimeCycleTools.planCycle(RuntimeCycleInput(
            symbol: "EURUSD",
            predictionTargetMinutes: 60,
            aiWindow: 100,
            onlineSamples: 20,
            onlineEpochs: 2,
            trainEpochs: 4,
            aiType: 6,
            ensembleMode: false,
            ensembleAgreePercent: 60.0,
            buyThreshold: 0.70,
            sellThreshold: 0.30,
            evThresholdPoints: 1.0,
            evLookbackSamples: 40,
            lastSymbol: "GBPUSD",
            signalBarUTC: 123_456,
            lastSignalBarUTC: 123_456,
            lastSignalDecisionKey: decisionKey,
            lastSignal: 1
        ))

        XCTAssertEqual(plan.action, .continueStages)
        XCTAssertTrue(plan.requiresStateReset)
        XCTAssertNil(plan.returnedSignal)
    }
}
