import XCTest
@testable import FXDataEngine

final class RuntimeSignalStateTests: XCTestCase {
    func testRuntimeSignalCacheResetMatchesLegacyFailClosedDefaults() {
        let cache = RuntimeSignalCache.reset

        XCTAssertEqual(cache.expectedMovePoints, 0.0)
        XCTAssertEqual(cache.tradeEdgePoints, 0.0)
        XCTAssertEqual(cache.confidence, 0.0)
        XCTAssertEqual(cache.reliability, 0.0)
        XCTAssertEqual(cache.pathRisk, 1.0)
        XCTAssertEqual(cache.fillRisk, 1.0)
        XCTAssertEqual(cache.tradeGate, 0.0)
        XCTAssertEqual(cache.minMovePoints, 0.0)

        XCTAssertEqual(cache.policyTradeProbability, 0.0)
        XCTAssertEqual(cache.policyNoTradeProbability, 1.0)
        XCTAssertEqual(cache.policySizeMultiplier, 1.0)
        XCTAssertEqual(cache.policyAction, .noTrade)

        XCTAssertFalse(cache.adaptiveRouterReady)
        XCTAssertEqual(cache.adaptiveRouterTopLabel, "UNKNOWN")
        XCTAssertEqual(cache.adaptiveRouterPosture, "NORMAL")

        XCTAssertFalse(cache.dynamicEnsembleReady)
        XCTAssertEqual(cache.dynamicEnsembleSkipProbability, 1.0)
        XCTAssertEqual(cache.dynamicEnsembleTradePosture, "NORMAL")

        XCTAssertFalse(cache.executionQualityReady)
        XCTAssertTrue(cache.executionQualityMemoryStale)
        XCTAssertTrue(cache.executionQualityDataStale)
        XCTAssertEqual(cache.executionQualityMethod, "SCORECARD_V1")
        XCTAssertEqual(cache.executionQualityTierKey, "GLOBAL|*|*|*")
        XCTAssertEqual(cache.executionQualityCautionLotScale, 1.0)

        XCTAssertFalse(cache.probabilityCalibrationReady)
        XCTAssertTrue(cache.probabilityCalibrationCalibrationStale)
        XCTAssertTrue(cache.probabilityCalibrationInputStale)
        XCTAssertEqual(cache.probabilityCalibrationMethod, "LOGISTIC_AFFINE")
        XCTAssertEqual(cache.probabilityCalibrationRawAction, "SKIP")
        XCTAssertEqual(cache.probabilityCalibrationRawSkipProbability, 1.0)
        XCTAssertEqual(cache.probabilityCalibrationSkipProbability, 1.0)
    }

    func testRuntimeSignalCacheProjectsPreparedRuntimeStates() {
        var cache = RuntimeSignalCache.reset
        cache.apply(
            signal: RiskPolicySignalState(
                confidence: 0.71,
                reliability: 0.62,
                tradeGate: 0.58,
                pathRisk: 0.33,
                fillRisk: 0.27,
                minMovePoints: 8.0,
                tradeEdgePoints: 6.0,
                expectedMovePoints: 12.0,
                hierarchyScore: 0.65,
                hierarchyConsistency: 0.66,
                hierarchyTradability: 0.67,
                hierarchyExecution: 0.68,
                contextStrength: 0.44,
                macroStateQuality: 0.55,
                policyAction: .enter,
                policyEnterProb: 0.52,
                policyNoTradeProb: 0.18,
                policyHoldQuality: 0.42,
                policySizeMultiplier: 1.25,
                policyPortfolioFit: 0.72,
                policyCapitalEfficiency: 0.74
            ),
            portfolioPressure: 0.34,
            contextQuality: 0.73,
            hierarchyHorizonFit: 0.81,
            horizonMinutes: 45,
            regimeID: 3
        )
        cache.apply(policy: MetaPolicyDecision(
            tradeProbability: 0.61,
            noTradeProbability: 0.14,
            enterProbability: 0.57,
            exitProbability: 0.08,
            directionBias: 0.22,
            sizeMultiplier: 1.35,
            holdQuality: 0.49,
            expectedUtility: 1.75,
            confidence: 0.66,
            portfolioFit: 0.77,
            capitalEfficiency: 0.79,
            addProbability: 0.13,
            reduceProbability: 0.09,
            tightenProbability: 0.07,
            timeoutProbability: 0.05,
            action: .add
        ))
        cache.apply(
            controlPlaneOverlay: RiskControlPlaneOverlayState(
                controlPlaneScore: 1.2,
                controlPlaneBuyScore: 0.9,
                controlPlaneSellScore: 0.4
            ),
            symbol: "EURUSD",
            barTimeUTC: 1_704_067_200
        )
        cache.apply(
            adaptiveRouterState: AdaptiveRegimeState(
                valid: true,
                symbol: "EURUSD",
                generatedAt: 1_704_067_210,
                topLabel: "LIQUIDITY_STRESS",
                confidence: 0.76,
                sessionLabel: "LONDON",
                priceCostRegime: "ELEVATED",
                volatilityRegime: "EXPANDING",
                newsRiskScore: 0.41,
                liquidityStress: 0.82,
                reasons: ["Liquidity stress elevated", "Session handoff burst active"]
            ),
            posture: "CAUTION",
            abstainBias: 0.18,
            routes: [
                AdaptiveRouterPluginRoute(name: "trend", eligible: true, routedWeight: 2.0, suitability: 1.1, status: .active),
                AdaptiveRouterPluginRoute(name: "mean", eligible: true, routedWeight: 1.0, suitability: 0.7, status: .downweighted),
                AdaptiveRouterPluginRoute(name: "event", eligible: false, suitability: 0.2, status: .suppressed)
            ]
        )
        cache.apply(
            dynamicEnsembleState: DynamicEnsembleRuntimeState(
                ready: true,
                generatedAt: 1_704_067_211,
                symbol: "EURUSD",
                topRegime: "LIQUIDITY_STRESS",
                sessionLabel: "LONDON",
                tradePosture: "ABSTAIN_BIAS",
                ensembleQuality: 0.64,
                abstainBias: 0.24,
                buyProbability: 0.21,
                sellProbability: 0.31,
                skipProbability: 0.48,
                reasons: ["adaptive_router_abstain_bias"]
            ),
            records: [
                DynamicEnsemblePluginRecord(ready: true, aiName: "trend", trustScore: 0.8, normalizedWeight: 0.4, status: .active),
                DynamicEnsemblePluginRecord(ready: true, aiName: "mean", trustScore: 0.6, normalizedWeight: 0.2, status: .downweighted),
                DynamicEnsemblePluginRecord(ready: true, aiName: "event", trustScore: 0.3, normalizedWeight: 0.1, status: .suppressed),
                DynamicEnsemblePluginRecord(ready: true, aiName: "idle", status: .excluded)
            ]
        )
        cache.apply(executionQualityState: ExecutionQualityPairState(
            ready: true,
            fallbackUsed: true,
            memoryStale: false,
            dataStale: false,
            supportUsable: true,
            generatedAt: 1_704_067_212,
            method: "SCORECARD_V2",
            selectedTierKind: "SYMBOL",
            selectedTierKey: "SYMBOL|EURUSD|*|*",
            selectedSupport: 42,
            selectedQuality: 0.71,
            priceCostNowPoints: 1.2,
            priceCostExpectedPoints: 1.4,
            priceCostWideningRisk: 0.25,
            expectedSlippagePoints: 0.3,
            slippageRisk: 0.2,
            fillQualityScore: 0.76,
            latencySensitivityScore: 0.22,
            liquidityFragilityScore: 0.36,
            executionQualityScore: 0.69,
            allowedDeviationPoints: 2.5,
            cautionLotScale: 0.7,
            cautionEnterProbabilityBuffer: 0.05,
            executionState: "CAUTION",
            reasons: ["fallback_tier"]
        ))
        cache.apply(probabilityCalibrationState: ProbabilityCalibrationRuntimeState(
            ready: true,
            fallbackUsed: true,
            calibrationStale: false,
            inputStale: false,
            generatedAt: 1_704_067_213,
            method: "ISOTONIC",
            sessionLabel: "LONDON",
            regimeLabel: "LIQUIDITY_STRESS",
            selectedTierKind: "SESSION",
            selectedTierKey: "SESSION|LONDON|*|*",
            selectedSupport: 64,
            selectedQuality: 0.83,
            rawBuyProbability: 0.2,
            rawSellProbability: 0.5,
            rawSkipProbability: 0.3,
            rawScore: -0.3,
            rawAction: "SELL",
            calibratedBuyProbability: 0.18,
            calibratedSellProbability: 0.47,
            calibratedSkipProbability: 0.35,
            calibratedConfidence: 0.58,
            expectedMoveMeanPoints: 11.0,
            expectedMoveQ25Points: 7.0,
            expectedMoveQ50Points: 10.0,
            expectedMoveQ75Points: 14.0,
            priceCostPoints: 1.3,
            slippageCostPoints: 0.4,
            uncertaintyScore: 0.21,
            uncertaintyPenaltyPoints: 0.9,
            riskPenaltyPoints: 1.1,
            expectedGrossEdgePoints: 6.3,
            edgeAfterCostsPoints: 3.4,
            finalAction: "SELL",
            abstain: true,
            reasons: ["edge_below_floor", "fallback_tier"]
        ))

        XCTAssertEqual(cache.expectedMovePoints, 12.0)
        XCTAssertEqual(cache.policyTradeProbability, 0.61)
        XCTAssertEqual(cache.policyAction, .add)
        XCTAssertEqual(cache.controlPlaneScore, 1.2)
        XCTAssertEqual(cache.adaptiveRouterPriceCostRegime, "ELEVATED")
        XCTAssertEqual(cache.adaptiveRouterActivePluginsCSV, "trend:0.6667:1.1000")
        XCTAssertEqual(cache.adaptiveRouterDownweightedPluginsCSV, "mean:0.3333:0.7000")
        XCTAssertEqual(cache.adaptiveRouterSuppressedPluginsCSV, "event:0.0000:0.2000")
        XCTAssertEqual(cache.dynamicEnsembleActivePluginsCSV, "trend:0.4000:0.8000")
        XCTAssertEqual(cache.dynamicEnsembleDownweightedPluginsCSV, "mean:0.2000:0.6000")
        XCTAssertEqual(cache.dynamicEnsembleSuppressedPluginsCSV, "event:0.1000:0.3000")
        XCTAssertEqual(cache.executionQualityPriceCostNowPoints, 1.2)
        XCTAssertEqual(cache.executionQualityState, "CAUTION")
        XCTAssertEqual(cache.probabilityCalibrationFinalAction, "SELL")
        XCTAssertEqual(cache.probabilityCalibrationPrimaryReason, "edge_below_floor")

        let restoredSignal = cache.riskPolicySignalState
        XCTAssertEqual(restoredSignal.confidence, 0.71)
        XCTAssertEqual(restoredSignal.policyAction, .add)
        XCTAssertEqual(restoredSignal.policyEnterProb, 0.57)
        XCTAssertEqual(restoredSignal.policyNoTradeProb, 0.14)
    }

    func testReducedSignalAndInvalidRouterProjectionClearStaleRuntimeFields() {
        var cache = RuntimeSignalCache.reset
        cache.apply(policy: MetaPolicyDecision(
            tradeProbability: 0.8,
            exitProbability: 0.7,
            directionBias: -0.4,
            expectedUtility: 2.0,
            addProbability: 0.6,
            reduceProbability: 0.5,
            tightenProbability: 0.4,
            timeoutProbability: 0.3,
            action: .exit
        ))
        cache.apply(
            adaptiveRouterState: AdaptiveRegimeState(
                valid: true,
                topLabel: "HIGH_VOL_EVENT",
                confidence: 0.8,
                sessionLabel: "NEWYORK",
                priceCostRegime: "ELEVATED",
                volatilityRegime: "EXPANDING",
                newsRiskScore: 0.6,
                liquidityStress: 0.7,
                reasons: ["event"]
            ),
            posture: "ABSTAIN_BIAS",
            abstainBias: 0.4,
            routes: [AdaptiveRouterPluginRoute(name: "event", eligible: true, routedWeight: 1.0, suitability: 0.4, status: .downweighted)]
        )

        cache.apply(signal: RiskPolicySignalState(policyAction: .noTrade, policyEnterProb: 0.2, policyNoTradeProb: 0.7))
        cache.apply(adaptiveRouterState: .reset, posture: "BLOCK", abstainBias: 0.9)

        XCTAssertEqual(cache.policyTradeProbability, 0.2)
        XCTAssertEqual(cache.policyNoTradeProbability, 0.7)
        XCTAssertEqual(cache.policyExitProbability, 0.0)
        XCTAssertEqual(cache.policyDirectionBias, 0.0)
        XCTAssertEqual(cache.policyExpectedUtility, 0.0)
        XCTAssertEqual(cache.policyAddProbability, 0.0)
        XCTAssertEqual(cache.policyReduceProbability, 0.0)
        XCTAssertEqual(cache.policyTightenProbability, 0.0)
        XCTAssertEqual(cache.policyTimeoutProbability, 0.0)
        XCTAssertEqual(cache.policyAction, .noTrade)

        XCTAssertFalse(cache.adaptiveRouterReady)
        XCTAssertEqual(cache.adaptiveRouterTopLabel, "UNKNOWN")
        XCTAssertEqual(cache.adaptiveRouterPosture, "NORMAL")
        XCTAssertEqual(cache.adaptiveRouterSession, "")
        XCTAssertEqual(cache.adaptiveRouterPriceCostRegime, "")
        XCTAssertEqual(cache.adaptiveRouterActivePluginsCSV, "")
        XCTAssertEqual(cache.adaptiveRouterDownweightedPluginsCSV, "")
    }

    func testControlPlaneSnapshotBuilderAndWriterMatchLegacyRows() throws {
        var cache = RuntimeSignalCache.reset
        cache.confidence = 0.72
        cache.reliability = 0.64
        cache.tradeGate = 0.55
        cache.hierarchyScore = 0.61
        cache.macroStateQuality = 0.47
        cache.minMovePoints = 10.0
        cache.tradeEdgePoints = 20.0
        cache.expectedMovePoints = 30.0
        cache.policyTradeProbability = 0.62
        cache.policyNoTradeProbability = 0.11
        cache.policyEnterProbability = 0.58
        cache.policyExitProbability = 0.08
        cache.policyAddProbability = 0.07
        cache.policyReduceProbability = 0.06
        cache.policyTightenProbability = 0.05
        cache.policyTimeoutProbability = 0.04
        cache.policySizeMultiplier = 1.4
        cache.policyPortfolioFit = 0.73
        cache.policyCapitalEfficiency = 0.81
        cache.policyAction = .enter
        cache.portfolioPressure = 0.33
        cache.controlPlaneBarTimeUTC = 1_704_067_199

        let snapshot = RuntimeSignalStateTools.controlPlaneSnapshot(
            symbol: "EURUSD",
            direction: -1,
            signalIntensity: 2.0,
            login: 42,
            magic: 77,
            chartID: 99,
            barTimeUTC: 1_704_067_200,
            cache: cache,
            exposure: RiskPortfolioExposureState(
                grossExposureLots: 2.0,
                correlatedExposureLots: 1.5,
                directionalClusterLots: 1.0
            ),
            capitalRiskPct: 0.4
        )

        XCTAssertTrue(snapshot.valid)
        XCTAssertEqual(snapshot.signalIntensity, 0.7, accuracy: 1e-12)
        XCTAssertEqual(snapshot.tradeEdgeNorm, 0.5, accuracy: 1e-12)
        XCTAssertEqual(snapshot.expectedMoveNorm, 1.5, accuracy: 1e-12)
        XCTAssertEqual(snapshot.policyLifecycleAction, .enter)

        let fallbackTimeSnapshot = RuntimeSignalStateTools.controlPlaneSnapshot(
            symbol: "EURUSD",
            direction: 1,
            signalIntensity: 1.0,
            login: 42,
            magic: 77,
            chartID: 99,
            barTimeUTC: 0,
            cache: cache
        )
        XCTAssertEqual(fallbackTimeSnapshot.barTimeUTC, 1_704_067_199)

        let tsv = RuntimeSignalStateTools.controlPlaneSnapshotTSV(snapshot)
        let parsed = ControlPlaneSnapshot.parse(tsv: tsv)
        XCTAssertEqual(parsed.symbol, "EURUSD")
        XCTAssertEqual(parsed.login, 42)
        XCTAssertEqual(parsed.magic, 77)
        XCTAssertEqual(parsed.chartID, 99)
        XCTAssertEqual(parsed.policyTradeProb, 0.62, accuracy: 1e-12)
        XCTAssertEqual(parsed.portfolioPressure, 0.33, accuracy: 1e-12)

        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let repository = RuntimeArtifactFileRepository(rootURL: root)
        let fileURL = try repository.writeControlPlaneLocalSnapshot(snapshot)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fileURL.path))
        let stored = try String(contentsOf: fileURL, encoding: .utf8)
        XCTAssertEqual(ControlPlaneSnapshot.parse(tsv: stored).symbol, "EURUSD")
        try? FileManager.default.removeItem(at: root)
    }
}
