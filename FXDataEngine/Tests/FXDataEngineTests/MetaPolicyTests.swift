import XCTest
@testable import FXDataEngine

final class MetaPolicyTests: XCTestCase {
    func testPolicyFeatureBuilderMatchesLegacyVectorFormula() {
        let features = MetaPolicyTools.buildPolicyFeatures(makePolicyFeatureInputs())
        let expected = [
            1,
            0.55,
            0.25,
            0.2,
            0.125,
            0.3,
            0.72,
            0.65,
            0.71,
            0.75,
            0.69,
            0.58,
            0.62,
            0.8,
            0.85,
            0.6696000000000001,
            0.7,
            0.5,
            0.7016524999999998,
            0.33,
            0.49056,
            0.6656000000000001,
            0.728,
            0.72,
            0.5504,
            0.24639999999999998,
            0.31,
            0.68,
            0.5625,
            0.18000000000000002,
            0.557025,
            0.3
        ]

        XCTAssertEqual(features.count, FXDataEngineConstants.policyFeatures)
        XCTAssertEqual(features.count, expected.count)
        for index in 0..<expected.count {
            XCTAssertEqual(features[index], expected[index], accuracy: 1e-12, "feature \(index)")
        }
    }

    func testPolicyFeatureBuilderUsesLegacyRiskDefaultsForMissingStackFeatures() {
        let features = MetaPolicyTools.buildPolicyFeatures(
            MetaPolicyFeatureInputs(stackFeatures: [], minMovePoints: 0.0)
        )

        XCTAssertEqual(features.count, FXDataEngineConstants.policyFeatures)
        XCTAssertEqual(features[0], 1.0, accuracy: 0.0)
        XCTAssertEqual(features[13], 0.0, accuracy: 0.0)
        XCTAssertEqual(features[14], 0.0, accuracy: 0.0)
        XCTAssertEqual(features[4], -0.5, accuracy: 0.0)
    }

    func testDefaultPolicyDecisionMatchesLegacyClearState() {
        let decision = MetaPolicyDecision()

        XCTAssertEqual(decision.tradeProbability, 0.0, accuracy: 0.0)
        XCTAssertEqual(decision.noTradeProbability, 1.0, accuracy: 0.0)
        XCTAssertEqual(decision.sizeMultiplier, 1.0, accuracy: 0.0)
        XCTAssertEqual(decision.action, .noTrade)
    }

    func testPolicyHeuristicAndNetworkPredictionMatchLegacyFormula() throws {
        let inputs = makePolicyFeatureInputs()
        let features = MetaPolicyTools.buildPolicyFeatures(inputs)

        let cold = MetaPolicyTools.predictPolicy(
            MetaPolicyNetworkState(),
            features: features,
            deployment: inputs.deployment
        )
        XCTAssertNil(cold.learnedTradeProbability)
        XCTAssertEqual(cold.decision.tradeProbability, 0.8818468949999998, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.noTradeProbability, 0.26578009725, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.enterProbability, 0.6378225067552403, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.exitProbability, 0.38257638400000005, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.directionBias, 0.37060000000000004, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.sizeMultiplier, 0.9083033105685002, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.holdQuality, 0.7321728, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.expectedUtility, 0.53189628375, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.confidence, 0.7914829427499999, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.portfolioFit, 0.6963999999999999, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.capitalEfficiency, 0.5600041920000001, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.addProbability, 0.582570025952201, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.reduceProbability, 0.30505206058500006, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.tightenProbability, 0.21532370665740003, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.timeoutProbability, 0.1949770433872, accuracy: 1e-12)
        XCTAssertEqual(cold.decision.action, .enter)

        let prediction = MetaPolicyTools.predictPolicy(
            makePolicyNetworkState(),
            features: features,
            deployment: inputs.deployment
        )
        XCTAssertEqual(prediction.hidden[0], 0.5716699660851173, accuracy: 1e-12)
        XCTAssertEqual(prediction.hidden[1], -0.04203521554649179, accuracy: 1e-12)
        XCTAssertEqual(prediction.hidden[2], 0.25089110491434896, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(prediction.learnedTradeProbability), 0.536015488131259, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(prediction.learnedDirectionBias), 0.012059041424469757, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(prediction.learnedSizeMultiplier), 0.9455685646579525, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(prediction.learnedHoldQuality), 0.5230772945576461, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.tradeProbability, 0.7050419266444299, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.noTradeProbability, 0.22777328254623067, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.enterProbability, 0.5113533402115176, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.exitProbability, 0.3320017717532056, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.directionBias, 0.22718361656978794, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.sizeMultiplier, 0.902736320279239, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.holdQuality, 0.6149768235933021, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.expectedUtility, 0.4317985860151124, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.confidence, 0.6525608514354181, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.portfolioFit, 0.6728000000000001, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.capitalEfficiency, 0.5753868502105787, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.addProbability, 0.5060685338106461, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.reduceProbability, 0.278149254730763, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.tightenProbability, 0.20986865717615097, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.timeoutProbability, 0.193043816545383, accuracy: 1e-12)
        XCTAssertEqual(prediction.decision.action, .hold)
    }

    func testPolicyNetworkUpdateMatchesLegacyFormula() {
        let features = MetaPolicyTools.buildPolicyFeatures(makePolicyFeatureInputs())
        let updated = MetaPolicyTools.updatedPolicyNetwork(
            makePolicyNetworkState(),
            features: features,
            tradeTarget: 1.0,
            directionTarget: 0.8,
            sizeTarget: 1.3,
            holdTarget: 0.65,
            sampleWeight: 1.6
        )

        XCTAssertTrue(updated.ready)
        XCTAssertEqual(updated.observations, 65)
        XCTAssertEqual(updated.tradeBias, -0.021150300012280646, accuracy: 1e-12)
        XCTAssertEqual(updated.directionBias, 0.055028607449297974, accuracy: 1e-12)
        XCTAssertEqual(updated.sizeBias, 0.025007529647696522, accuracy: 1e-12)
        XCTAssertEqual(updated.holdBias, -0.057579169485053275, accuracy: 1e-12)
        XCTAssertEqual(updated.tradeWeights[0], 0.25505761759294127, accuracy: 1e-12)
        XCTAssertEqual(updated.directionWeights[1], 0.1993670771673838, accuracy: 1e-12)
        XCTAssertEqual(updated.sizeWeights[2], -0.07874317852214952, accuracy: 1e-12)
        XCTAssertEqual(updated.holdWeights[0], 0.20138272401925605, accuracy: 1e-12)
        XCTAssertEqual(updated.holdWeights[2], 0.16060641117940827, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[0], 0.051208135794964145, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[1], -0.07730410860831952, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[2], 0.032929221874570765, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[0][0], 0.40120813579496417, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[0][1], 0.2006635210239332, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[0][5], 0.3003610102435436, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[1][2], -0.24932483507295852, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[1][15], 0.15180445362839642, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[2][18], 0.22205424682172048, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[2][29], 0.4005253526108286, accuracy: 1e-12)
    }

    private func makePolicyFeatureInputs() -> MetaPolicyFeatureInputs {
        var deployment = LiveDeploymentProfile()
        deployment.analogWeight = 0.40
        deployment.regimeTransitionWeight = 0.60
        deployment.macroQualityFloor = 0.55
        deployment.teacherSignalGain = 1.20
        deployment.studentSignalGain = 1.10
        deployment.foundationQualityGain = 1.30
        deployment.macroStateGain = 1.40
        deployment.policyLifecycleGain = 1.25
        deployment.foundationWeight = 0.50
        deployment.teacherWeight = 0.65
        deployment.policyTradeFloor = 0.52
        deployment.policySizeBias = 1.10

        return MetaPolicyFeatureInputs(
            stackFeatures: StackerTools.buildStackFeatures(makeStackFeatureInputs()),
            tradeGate: 0.71,
            tradeEdgePoints: 2.4,
            expectedMovePoints: 5.0,
            minMovePoints: 2.0,
            macroQuality: 0.62,
            contextQuality: 0.40,
            contextStrength: 1.50,
            foundationTrust: 0.61,
            foundationDirectionBias: 0.33,
            studentTrust: 0.48,
            analogSimilarity: 0.64,
            analogQuality: 0.70,
            regime: RegimeGraphQuery(
                persistence: 0.72,
                transitionConfidence: 0.64,
                instability: 0.28,
                edgeBias: 0.31,
                qualityBias: 0.68,
                macroAlignment: 0.45,
                predictedRegime: 3
            ),
            deployment: deployment,
            portfolioPressureHint: 0.45
        )
    }

    private func makeStackFeatureInputs() -> StackFeatureInputs {
        StackFeatureInputs(
            buyPct: 55.0,
            sellPct: 25.0,
            skipPct: 20.0,
            avgBuyEV: 3.5,
            avgSellEV: 1.2,
            minMovePoints: 2.0,
            expectedMovePoints: 5.0,
            volProxy: 2.5,
            horizonMinutes: 60,
            maxConfiguredHorizonMinutes: 240,
            avgConfidence: 0.72,
            avgReliability: 0.65,
            moveDispersion: 1.8,
            directionalMargin: 0.4,
            activeFamilyRatio: 0.8,
            dominantFamilyRatio: 0.6,
            contextStrength: 1.3,
            contextQuality: 0.7,
            avgHitTime: 0.35,
            avgPathRisk: 0.2,
            avgFillRisk: 0.15,
            avgMFERatio: 1.4,
            avgMAERatio: 0.6,
            avgContextEdgeNorm: 0.45,
            avgContextRegret: 0.22,
            avgGlobalEdgeNorm: 0.38,
            bestCounterfactualEdgeNorm: 0.51,
            ensembleVsBestGapNorm: 0.12,
            avgPortfolioEdgeNorm: 0.44,
            avgPortfolioStability: 0.72,
            avgPortfolioCorrelationPenalty: 0.18,
            avgPortfolioDiversification: 0.66,
            bestModelShare: 0.31,
            bestBuyShare: 0.45,
            bestSellShare: 0.28,
            avgContextTrust: 0.57,
            foundationTrust: 0.61,
            foundationDirectionBias: 0.33,
            foundationMoveRatio: 1.2,
            studentTrust: 0.48,
            studentTradability: 0.53,
            analogSimilarity: 0.64,
            analogEdgeNorm: 0.41,
            analogQuality: 0.7,
            hierarchyConsistency: 0.75,
            hierarchyTradability: 0.69,
            hierarchyExecutionViability: 0.58,
            hierarchyHorizonFit: 0.62
        )
    }

    private func makePolicyNetworkState() -> MetaPolicyNetworkState {
        var inputWeights = Array(
            repeating: Array(repeating: 0.0, count: FXDataEngineConstants.policyFeatures),
            count: FXDataEngineConstants.policyHidden
        )
        inputWeights[0][0] = 0.40
        inputWeights[0][1] = 0.20
        inputWeights[0][5] = 0.30
        inputWeights[1][2] = -0.25
        inputWeights[1][15] = 0.15
        inputWeights[2][18] = 0.22
        inputWeights[2][29] = 0.40

        var hiddenBias = Array(repeating: 0.0, count: FXDataEngineConstants.policyHidden)
        hiddenBias[0] = 0.05
        hiddenBias[1] = -0.08
        hiddenBias[2] = 0.03

        var tradeWeights = Array(repeating: 0.0, count: FXDataEngineConstants.policyHidden)
        tradeWeights[0] = 0.25
        tradeWeights[1] = -0.15
        tradeWeights[2] = 0.10

        var directionWeights = Array(repeating: 0.0, count: FXDataEngineConstants.policyHidden)
        directionWeights[0] = -0.10
        directionWeights[1] = 0.20
        directionWeights[2] = 0.15

        var sizeWeights = Array(repeating: 0.0, count: FXDataEngineConstants.policyHidden)
        sizeWeights[0] = 0.12
        sizeWeights[1] = 0.18
        sizeWeights[2] = -0.08

        var holdWeights = Array(repeating: 0.0, count: FXDataEngineConstants.policyHidden)
        holdWeights[0] = 0.20
        holdWeights[1] = 0.05
        holdWeights[2] = 0.16

        return MetaPolicyNetworkState(
            inputWeights: inputWeights,
            hiddenBias: hiddenBias,
            tradeWeights: tradeWeights,
            directionWeights: directionWeights,
            sizeWeights: sizeWeights,
            holdWeights: holdWeights,
            tradeBias: -0.03,
            directionBias: 0.04,
            sizeBias: 0.02,
            holdBias: -0.06,
            ready: true,
            observations: 64
        )
    }
}
