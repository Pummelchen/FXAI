import XCTest
@testable import FXDataEngine

final class RuntimeFeaturePipelineTests: XCTestCase {
    func testRuntimeFeaturePipelineRequirementAndWindowsMatchLegacyBlock() {
        let plan = RuntimeFeaturePipelineTools.buildPlan(
            baseSamples: 60,
            onlineSamples: 90,
            baseHorizonMinutes: 13,
            selectedHorizonMinutes: 21,
            configuredHorizons: [1, 2, 34],
            shadowSamples: 4,
            shadowEpochs: 9,
            shadowEveryBars: 0,
            ensembleMode: true,
            shadowAllowed: true,
            signalSequence: 7
        )

        XCTAssertEqual(plan.requirement.featureLookback, 10)
        XCTAssertEqual(plan.requirement.horizonLoadMax, 34)
        XCTAssertEqual(plan.requirement.neededBars, 134)
        XCTAssertEqual(plan.requirement.alignUpToIndex, 133)
        XCTAssertEqual(plan.maxValidIndex, 123)
        XCTAssertEqual(plan.initStart, 21)
        XCTAssertEqual(plan.initEnd, 80)
        XCTAssertEqual(plan.onlineStart, 21)
        XCTAssertEqual(plan.onlineEnd, 110)
        XCTAssertEqual(plan.shadowSamples, 8)
        XCTAssertEqual(plan.shadowEpochs, 3)
        XCTAssertEqual(plan.shadowEveryBars, 1)
        XCTAssertTrue(plan.runShadow)
        XCTAssertEqual(plan.shadowStart, 21)
        XCTAssertEqual(plan.shadowEnd, 28)
        XCTAssertTrue(plan.haveInitWindow)
        XCTAssertTrue(plan.haveOnlineWindow)
        XCTAssertTrue(plan.haveShadowWindow)
        XCTAssertEqual(plan.precomputeEnd, 110)
    }

    func testRuntimeFeaturePipelineContextStateMatchesLegacyFormula() {
        let state = RuntimeFeaturePipelineTools.contextState(
            contextMean: -0.20,
            contextStandardDeviation: 0.30,
            contextUpRatio: 0.80,
            dynamicContext: RuntimeDynamicContextInputs(
                utility: 0.50,
                stability: 0.20,
                lead: -1.0,
                coverage: 1.0
            )
        )

        XCTAssertEqual(state.strength, 0.80, accuracy: 1e-12)
        XCTAssertEqual(state.quality, 0.175, accuracy: 1e-12)

        let clamped = RuntimeFeaturePipelineTools.contextState(
            contextMean: 10.0,
            contextStandardDeviation: 10.0,
            contextUpRatio: -10.0,
            dynamicContext: RuntimeDynamicContextInputs(utility: 5.0, stability: 5.0, lead: 5.0, coverage: 5.0)
        )
        XCTAssertEqual(clamped.strength, 4.0, accuracy: 0.0)
        XCTAssertEqual(clamped.quality, 2.0, accuracy: 0.0)
    }

    func testHorizonPolicyFeaturesUsePriceCostReplacementForLegacyCostSlots() {
        let features = RuntimeFeaturePipelineTools.buildHorizonPolicyFeatures(
            HorizonPolicyFeatureInputs(
                horizonMinutes: 8,
                baseHorizonMinutes: 4,
                expectedAbsMovePoints: 12.0,
                minMovePoints: 3.0,
                sampleTimeUTC: 1_704_067_200,
                currentVolPoints: 2.0,
                priceCostPoints: 1.0,
                regimeID: 3,
                aiHint: 2,
                contextStrength: 1.5,
                contextQuality: 0.4,
                modelReliabilityHint: 0.7,
                regimeEdgePoints: 1.5,
                modelEdgePoints: -0.5,
                holdPenaltyPerMinute: 0.01,
                configuredHorizons: [4, 8, 16]
            )
        )

        XCTAssertEqual(features.count, FXDataEngineConstants.horizonPolicyFeatures)
        XCTAssertEqual(features[0], 1.0, accuracy: 0.0)
        XCTAssertEqual(features[1], 0.75, accuracy: 1e-12)
        XCTAssertEqual(features[2], 1.0, accuracy: 1e-12)
        XCTAssertEqual(features[3], 1.0 / sqrt(8.0), accuracy: 1e-12)
        XCTAssertEqual(features[4], -0.08, accuracy: 1e-12)
        XCTAssertEqual(features[5], (1.5 / 3.0) / 3.0, accuracy: 1e-12)
        XCTAssertEqual(features[6], (-0.5 / 3.0) / 3.0, accuracy: 1e-12)
        XCTAssertEqual(features[7], 0.08, accuracy: 1e-12)
        XCTAssertEqual(features[8], -1.0 / 6.0, accuracy: 1e-12)
        XCTAssertEqual(features[9], -1.0, accuracy: 1e-12)
        XCTAssertEqual(features[10], -1.0, accuracy: 1e-12)
        XCTAssertEqual(features[11], 0.5, accuracy: 1e-12)
        XCTAssertEqual(features[22], -0.5, accuracy: 1e-12)
        XCTAssertEqual(features[23], (1.0 / 7.0) - 0.5, accuracy: 1e-12)
        XCTAssertEqual(features[26], 0.5, accuracy: 1e-12)
        XCTAssertEqual(features[30], -0.6, accuracy: 1e-12)
    }

    func testRoutedHorizonSelectionUsesPreparedPolicyAndPriorSignals() {
        let disabled = RuntimeFeaturePipelineTools.selectRoutedHorizon(
            candidates: [RuntimeRoutedHorizonCandidate(horizonMinutes: 13, expectedAbsMovePoints: 50.0)],
            fallbackHorizonMinutes: 5,
            multiHorizonEnabled: false,
            minMovePoints: 2.0,
            holdPenaltyPerMinute: 0.01
        )
        XCTAssertEqual(disabled.horizonMinutes, 5)
        XCTAssertTrue(disabled.usedFallback)

        let selected = RuntimeFeaturePipelineTools.selectRoutedHorizon(
            candidates: [
                RuntimeRoutedHorizonCandidate(horizonMinutes: 5, expectedAbsMovePoints: 8.0),
                RuntimeRoutedHorizonCandidate(
                    horizonMinutes: 13,
                    expectedAbsMovePoints: 15.0,
                    regimeEdgeReady: true,
                    regimeEdgePoints: 1.0,
                    regimeObservations: 3,
                    regimeTotalObservations: 10.0,
                    modelEdgeReady: true,
                    modelEdgePoints: 0.5,
                    modelObservations: 0,
                    horizonPolicyReady: true,
                    horizonPolicyValue: 0.4,
                    oofHorizonPriorScore: 0.2
                )
            ],
            fallbackHorizonMinutes: 5,
            multiHorizonEnabled: true,
            minMovePoints: 2.0,
            holdPenaltyPerMinute: 0.01
        )

        XCTAssertEqual(selected.horizonMinutes, 13)
        XCTAssertFalse(selected.usedFallback)
        XCTAssertGreaterThan(selected.score, 3.80)
    }

    func testRuntimeTransferInputBuildsBiasVectorAndLegacySharedWindow() {
        let features = [0.10, -0.20, 0.30]
        let rawX = RuntimeTransferTools.modelInputVector(features: features)
        XCTAssertEqual(rawX.count, FXDataEngineConstants.aiWeights)
        XCTAssertEqual(rawX[0], 1.0)
        XCTAssertEqual(rawX[1], 0.10, accuracy: 1e-12)
        XCTAssertEqual(rawX[2], -0.20, accuracy: 1e-12)
        XCTAssertEqual(rawX[3], 0.30, accuracy: 1e-12)

        let ignoredFirst = sample(valid: true, marker: 9.0)
        let validSecond = sample(valid: true, marker: 1.0)
        let invalidThird = sample(valid: false, marker: 2.0)
        let validFourth = sample(valid: true, marker: 3.0)
        let window = RuntimeTransferTools.currentSharedWindow(
            currentRawX: rawX,
            samples: [ignoredFirst, validSecond, invalidThird, validFourth],
            span: 4
        )

        XCTAssertEqual(window.count, 3)
        XCTAssertEqual(window[0][1], 0.10, accuracy: 1e-12)
        XCTAssertEqual(window[1][1], 1.0, accuracy: 1e-12)
        XCTAssertEqual(window[2][1], 3.0, accuracy: 1e-12)

        let transferInput = RuntimeTransferTools.currentTransferInput(
            features: features,
            samples: [ignoredFirst, validSecond],
            horizonMinutes: 13,
            symbol: "EURUSD"
        )
        XCTAssertEqual(transferInput.rawX[0], 1.0, accuracy: 0.0)
        XCTAssertEqual(transferInput.rawX[1], 0.10, accuracy: 1e-12)
        XCTAssertGreaterThanOrEqual(transferInput.windowSize, 2)

        let sharedPayload = RuntimeTransferTools.sharedTransferPayload(
            input: transferInput,
            domainHash: 0.50,
            horizonMinutes: 13
        )
        XCTAssertEqual(sharedPayload.count, FXDataEngineConstants.sharedTransferFeatures)
        XCTAssertEqual(sharedPayload[0], 1.0, accuracy: 0.0)

        let directPayload = RuntimeTransferTools.currentSharedTransferPayload(
            features: features,
            samples: [ignoredFirst, validSecond],
            horizonMinutes: 13,
            symbol: "EURUSD",
            domainHash: 0.50
        )
        XCTAssertEqual(directPayload, sharedPayload)
    }

    private func sample(valid: Bool, marker: Double) -> RuntimeArtifactPreparedSample {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[0] = 1.0
        x[1] = marker
        return RuntimeArtifactPreparedSample(valid: valid, x: x)
    }
}
