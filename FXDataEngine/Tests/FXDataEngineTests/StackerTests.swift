import XCTest
@testable import FXDataEngine

final class StackerTests: XCTestCase {
    func testStackFeatureBuilderMatchesLegacyVectorFormula() {
        let features = StackerTools.buildStackFeatures(makeFeatureInputs())
        let expected = [
            1,
            0.10000000000000009,
            -0.5,
            -0.59999999999999998,
            0.58333333333333337,
            0.19999999999999998,
            0.30000000000000004,
            0.55000000000000004,
            0.092244340001498615,
            0.28749999999999998,
            -0.099999999999999978,
            0.625,
            0.25,
            -0.35000000000000003,
            0.30000000000000004,
            0.30000000000000004,
            0.5,
            0.4375,
            0.625,
            0.60000000000000009,
            0.71999999999999997,
            0.65000000000000002,
            0.45000000000000001,
            0.40000000000000002,
            0.80000000000000004,
            0.59999999999999998,
            0.65000000000000002,
            0.34999999999999998,
            0.80000000000000004,
            0.46799999999999997,
            0.12000000000000002,
            0.17999999999999999,
            -0.27250000000000002,
            0.39000000000000001,
            0.39166666666666666,
            0.063187372901026551,
            0.17499999999999999,
            0.069999999999999951,
            0.20499999999999996,
            -0.23999999999999999,
            0.47999999999999998,
            0.17999999999999999,
            0.43200000000000005,
            0.091999999999999998,
            0.05534660400089917,
            0.23039999999999999,
            -0.2465,
            0.33571428571428574,
            0.34999999999999998,
            0.20000000000000001,
            0.14999999999999999,
            0.19999999999999996,
            0.099999999999999978,
            0.57599999999999996,
            0.51000000000000001,
            -0.024000000000000077,
            0.45000000000000001,
            0.22,
            0.38,
            0.51000000000000001,
            0.12,
            0.44,
            0.71999999999999997,
            0.17999999999999999,
            0.66000000000000003,
            0.31,
            0.45000000000000001,
            0.28000000000000003,
            0.56999999999999995,
            0.29000000000000004,
            0.56159999999999999,
            0.42333333333333334,
            0.60999999999999999,
            0.33000000000000002,
            0.1333333333333333,
            0.47999999999999998,
            0.53000000000000003,
            0.64000000000000001,
            0.40999999999999998,
            0.69999999999999996,
            0.75,
            0.68999999999999995,
            0.57999999999999996,
            0.62
        ]

        XCTAssertEqual(features.count, FXDataEngineConstants.stackFeatures)
        XCTAssertEqual(features.count, expected.count)
        for index in 0..<expected.count {
            XCTAssertEqual(features[index], expected[index], accuracy: 1e-12, "feature \(index)")
        }
    }

    func testStackObjectivesTrustAndUtilitiesMatchLegacyFormula() {
        let features = makeFeatures()

        XCTAssertTrue(StackerTools.isModelInList(aiID: 7, modelIDs: [1, 7, 12]))
        XCTAssertFalse(StackerTools.isModelInList(aiID: 8, modelIDs: [1, 7, 12]))
        XCTAssertEqual(StackerTools.stackFeature(features, 999, default: 0.25), 0.25, accuracy: 0.0)

        XCTAssertEqual(StackerTools.stackPortfolioObjective(features: features), 0.602, accuracy: 1e-12)
        XCTAssertEqual(StackerTools.stackRoutingObjective(features: features), 0.275, accuracy: 1e-12)
        XCTAssertEqual(StackerTools.stackRouterContextTrust(features: features), 0.474, accuracy: 1e-12)

        XCTAssertEqual(
            StackerTools.stackRouterActionUtility(action: .buy, labelClass: .buy, realizedEdge: 2.5, qualityScore: 1.2),
            1.0,
            accuracy: 0.0
        )
        XCTAssertEqual(
            StackerTools.stackRouterActionUtility(action: .skip, labelClass: .skip, realizedEdge: 0.5, qualityScore: 0.8),
            0.396,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            StackerTools.stackRouterActionUtility(action: .skip, labelClass: .buy, realizedEdge: 2.0, qualityScore: 0.5),
            -0.79,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            StackerTools.stackRouterActionUtility(action: .sell, labelClass: .buy, realizedEdge: 2.0, qualityScore: 0.5),
            -0.92,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            StackerTools.stackRouterActionUtility(action: .sell, labelClass: .skip, realizedEdge: -2.0, qualityScore: 1.0),
            -0.58,
            accuracy: 1e-12
        )
    }

    func testStackHeuristicAndNetworkPredictionMatchLegacyFormula() {
        let features = StackerTools.buildStackFeatures(makeFeatureInputs())
        let heuristic = StackerTools.heuristicStackProbabilities(features: features)
        XCTAssertEqual(heuristic[LabelClass.sell.rawValue], 0.39096986223853664, accuracy: 1e-12)
        XCTAssertEqual(heuristic[LabelClass.buy.rawValue], 0.6011122055452974, accuracy: 1e-12)
        XCTAssertEqual(heuristic[LabelClass.skip.rawValue], 0.007917932216165883, accuracy: 1e-12)

        let state = makeStackNetworkState()
        let prediction = StackerTools.predictStackProbabilities(state, features: features)
        XCTAssertEqual(prediction.hidden[0], 0.5511280285381469, accuracy: 1e-12)
        XCTAssertEqual(prediction.hidden[1], 0.18130656540177173, accuracy: 1e-12)
        XCTAssertEqual(prediction.hidden[2], 0.12435300177159618, accuracy: 1e-12)
        XCTAssertEqual(prediction.rawProbabilities[LabelClass.sell.rawValue], 0.38147346863924947, accuracy: 1e-12)
        XCTAssertEqual(prediction.rawProbabilities[LabelClass.buy.rawValue], 0.28949911417535534, accuracy: 1e-12)
        XCTAssertEqual(prediction.rawProbabilities[LabelClass.skip.rawValue], 0.3290274171853952, accuracy: 1e-12)
        XCTAssertEqual(prediction.probabilities[LabelClass.sell.rawValue], 0.38282497777707236, accuracy: 1e-12)
        XCTAssertEqual(prediction.probabilities[LabelClass.buy.rawValue], 0.30909570816832066, accuracy: 1e-12)
        XCTAssertEqual(prediction.probabilities[LabelClass.skip.rawValue], 0.308079314054607, accuracy: 1e-12)
    }

    func testStackNetworkUpdateMatchesLegacyFormula() {
        let features = StackerTools.buildStackFeatures(makeFeatureInputs())
        let updated = StackerTools.updatedStackNetwork(
            makeStackNetworkState(),
            features: features,
            labelClass: .buy,
            sampleWeight: 1.4
        )

        XCTAssertTrue(updated.ready)
        XCTAssertEqual(updated.observations, 11)
        XCTAssertEqual(updated.outputBias[LabelClass.sell.rawValue], 0.03632101673362803, accuracy: 1e-12)
        XCTAssertEqual(updated.outputBias[LabelClass.buy.rawValue], 0.005477341222722736, accuracy: 1e-12)
        XCTAssertEqual(updated.outputBias[LabelClass.skip.rawValue], -0.0017983579563507629, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[LabelClass.sell.rawValue][0], 0.2924577056540137, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[LabelClass.sell.rawValue][1], -0.10247894838555274, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[LabelClass.sell.rawValue][2], 0.04829840682531248, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[LabelClass.buy.rawValue][0], -0.18595644098220424, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[LabelClass.buy.rawValue][1], 0.25461635651100717, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[LabelClass.buy.rawValue][2], 0.10316704276954333, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[LabelClass.skip.rawValue][0], 0.09349645315086758, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[LabelClass.skip.rawValue][1], 0.047860309697222644, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[LabelClass.skip.rawValue][2], -0.1514654495948558, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[0], 0.09277353997802817, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[1], -0.19308763133333767, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[2], 0.053577350813206526, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[0][0], 0.49277353997802814, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[0][1], 0.19927552825594447, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[1][2], -0.3034534457205436, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[1][4], 0.40402856357183636, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[2][6], 0.25107092306663903, accuracy: 1e-12)
    }

    func testTradeGatePredictionMatchesLegacyFormula() throws {
        let features = StackerTools.buildStackFeatures(makeFeatureInputs())
        let priorCell = makeOOFPriorCell()

        XCTAssertEqual(
            StackerTools.tradeGateHeuristic(features: features),
            0.99,
            accuracy: 0.0
        )
        XCTAssertEqual(
            StackerTools.tradeGateHeuristic(features: features, oofPriorCell: priorCell),
            0.907049,
            accuracy: 1e-12
        )

        let cold = StackerTools.predictTradeGate(
            TradeGateNetworkState(),
            features: features,
            oofPriorCell: priorCell
        )
        XCTAssertEqual(cold.probability, 0.907049, accuracy: 1e-12)
        XCTAssertNil(cold.learnedProbability)

        let prediction = StackerTools.predictTradeGate(
            makeTradeGateNetworkState(),
            features: features,
            oofPriorCell: priorCell
        )
        XCTAssertEqual(try XCTUnwrap(prediction.oofPrior), 0.61295, accuracy: 1e-12)
        XCTAssertEqual(prediction.heuristicProbability, 0.907049, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(prediction.learnedProbability), 0.512524629565967, accuracy: 1e-12)
        XCTAssertEqual(prediction.hidden[0], 0.36096932243320357, accuracy: 1e-12)
        XCTAssertEqual(prediction.hidden[1], 0.08479588154870195, accuracy: 1e-12)
        XCTAssertEqual(prediction.hidden[2], 0.08678116153147494, accuracy: 1e-12)
        XCTAssertEqual(prediction.probability, 0.7110782613149985, accuracy: 1e-12)
    }

    func testTradeGateUpdateMatchesLegacyFormula() {
        let features = StackerTools.buildStackFeatures(makeFeatureInputs())
        let updated = StackerTools.updatedTradeGateNetwork(
            makeTradeGateNetworkState(),
            features: features,
            tradeTarget: true,
            sampleWeight: 1.7
        )

        XCTAssertTrue(updated.ready)
        XCTAssertEqual(updated.observations, 73)
        XCTAssertEqual(updated.outputBias, -0.03938947966933394, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[0], 0.30382776767058545, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[1], -0.24909835102167174, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[2], 0.15091964094682914, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[0], 0.05276839451457164, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[1], -0.1026335567664257, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[2], 0.021579591923060883, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[0][7], 0.4015205683929355, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[0][20], 0.15199247582921196, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[1][49], -0.2005256870582457, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[1][80], 0.2980232959826215, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[2][53], 0.2509085645788837, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[2][57], -0.34965069726060755, accuracy: 1e-12)
    }

    func testStackRouterObservationMatchesLegacyPreparedState() {
        let features = makeFeatures()
        let probabilities = [0.2, 0.6, 0.2]

        let first = StackerTools.observedRouterCells(
            [],
            labelClass: .buy,
            realizedEdge: 2.0,
            qualityScore: 1.1,
            features: features,
            predictedProbabilities: probabilities,
            sampleWeight: 1.5
        )

        XCTAssertEqual(first.count, LabelClass.allCases.count)
        XCTAssertEqual(first[LabelClass.sell.rawValue].value, -0.992, accuracy: 1e-12)
        XCTAssertEqual(first[LabelClass.sell.rawValue].counterfactual, -1.0, accuracy: 0.0)
        XCTAssertEqual(first[LabelClass.sell.rawValue].regret, 1.0, accuracy: 0.0)
        XCTAssertEqual(first[LabelClass.sell.rawValue].observations, 1)
        XCTAssertTrue(first[LabelClass.sell.rawValue].ready)

        XCTAssertEqual(first[LabelClass.buy.rawValue].value, 1.0, accuracy: 0.0)
        XCTAssertEqual(first[LabelClass.buy.rawValue].counterfactual, 0.7684, accuracy: 1e-12)
        XCTAssertEqual(first[LabelClass.buy.rawValue].regret, 0.0, accuracy: 0.0)
        XCTAssertEqual(first[LabelClass.buy.rawValue].observations, 1)

        XCTAssertEqual(first[LabelClass.skip.rawValue].value, -0.85, accuracy: 1e-12)
        XCTAssertEqual(first[LabelClass.skip.rawValue].counterfactual, -1.0, accuracy: 0.0)
        XCTAssertEqual(first[LabelClass.skip.rawValue].regret, 1.0, accuracy: 0.0)
        XCTAssertEqual(first[LabelClass.skip.rawValue].observations, 1)

        let second = StackerTools.observedRouterCells(
            first,
            labelClass: .buy,
            realizedEdge: 2.0,
            qualityScore: 1.1,
            features: features,
            predictedProbabilities: probabilities,
            sampleWeight: 1.5
        )

        XCTAssertEqual(second[LabelClass.sell.rawValue].value, -0.992, accuracy: 1e-12)
        XCTAssertEqual(second[LabelClass.buy.rawValue].counterfactual, 0.7684, accuracy: 1e-12)
        XCTAssertEqual(second[LabelClass.skip.rawValue].regret, 1.0, accuracy: 0.0)
        XCTAssertEqual(second[LabelClass.sell.rawValue].observations, 2)
        XCTAssertEqual(second[LabelClass.buy.rawValue].observations, 2)
        XCTAssertEqual(second[LabelClass.skip.rawValue].observations, 2)
    }

    func testStackRouterBlendMatchesLegacyFormula() {
        let features = makeFeatures()
        let observed = StackerTools.observedRouterCells(
            StackerTools.observedRouterCells(
                [],
                labelClass: .buy,
                realizedEdge: 2.0,
                qualityScore: 1.1,
                features: features,
                predictedProbabilities: [0.2, 0.6, 0.2],
                sampleWeight: 1.5
            ),
            labelClass: .buy,
            realizedEdge: 2.0,
            qualityScore: 1.1,
            features: features,
            predictedProbabilities: [0.2, 0.6, 0.2],
            sampleWeight: 1.5
        )

        let blended = StackerTools.stackRouterBlend(
            probabilities: [0.25, 0.5, 0.25],
            features: features,
            actionCells: observed
        )

        XCTAssertEqual(blended[LabelClass.sell.rawValue], 0.2434787518887173, accuracy: 1e-12)
        XCTAssertEqual(blended[LabelClass.buy.rawValue], 0.5124675146925366, accuracy: 1e-12)
        XCTAssertEqual(blended[LabelClass.skip.rawValue], 0.2440537334187462, accuracy: 1e-12)
        XCTAssertEqual(blended.reduce(0.0, +), 1.0, accuracy: 1e-12)
    }

    private func makeFeatures() -> [Double] {
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.stackFeatures)
        features[6] = 0.4
        features[56] = 0.4
        features[57] = 0.2
        features[58] = 0.3
        features[59] = 0.5
        features[60] = 0.1
        features[61] = 0.6
        features[62] = 0.8
        features[63] = 0.25
        features[64] = 0.7
        features[68] = 0.55
        features[69] = 0.35
        features[70] = 0.65
        features[71] = 0.2
        return features
    }

    private func makeFeatureInputs() -> StackFeatureInputs {
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

    private func makeStackNetworkState() -> StackNetworkState {
        var inputWeights = Array(
            repeating: Array(repeating: 0.0, count: FXDataEngineConstants.stackFeatures),
            count: FXDataEngineConstants.stackHidden
        )
        inputWeights[0][0] = 0.5
        inputWeights[0][1] = 0.2
        inputWeights[1][2] = -0.3
        inputWeights[1][4] = 0.4
        inputWeights[2][6] = 0.25

        var hiddenBias = Array(repeating: 0.0, count: FXDataEngineConstants.stackHidden)
        hiddenBias[0] = 0.1
        hiddenBias[1] = -0.2
        hiddenBias[2] = 0.05

        var outputWeights = Array(
            repeating: Array(repeating: 0.0, count: FXDataEngineConstants.stackHidden),
            count: LabelClass.allCases.count
        )
        outputWeights[LabelClass.sell.rawValue][0] = 0.3
        outputWeights[LabelClass.sell.rawValue][1] = -0.1
        outputWeights[LabelClass.sell.rawValue][2] = 0.05
        outputWeights[LabelClass.buy.rawValue][0] = -0.2
        outputWeights[LabelClass.buy.rawValue][1] = 0.25
        outputWeights[LabelClass.buy.rawValue][2] = 0.1
        outputWeights[LabelClass.skip.rawValue][0] = 0.1
        outputWeights[LabelClass.skip.rawValue][1] = 0.05
        outputWeights[LabelClass.skip.rawValue][2] = -0.15

        return StackNetworkState(
            inputWeights: inputWeights,
            hiddenBias: hiddenBias,
            outputWeights: outputWeights,
            outputBias: [0.05, -0.02, 0.01],
            ready: true,
            observations: 10
        )
    }

    private func makeOOFPriorCell() -> OOFHorizonPriorCell {
        OOFHorizonPriorCell(
            scoreEMA: 3.2,
            edgeEMA: 1.4,
            qualityEMA: 0.75,
            tradeRateEMA: 0.65,
            ready: true,
            observations: 40
        )
    }

    private func makeTradeGateNetworkState() -> TradeGateNetworkState {
        var inputWeights = Array(
            repeating: Array(repeating: 0.0, count: FXDataEngineConstants.stackFeatures),
            count: FXDataEngineConstants.tradeGateHidden
        )
        inputWeights[0][7] = 0.4
        inputWeights[0][20] = 0.15
        inputWeights[1][49] = -0.2
        inputWeights[1][80] = 0.3
        inputWeights[2][53] = 0.25
        inputWeights[2][57] = -0.35

        var hiddenBias = Array(repeating: 0.0, count: FXDataEngineConstants.tradeGateHidden)
        hiddenBias[0] = 0.05
        hiddenBias[1] = -0.1
        hiddenBias[2] = 0.02

        var outputWeights = Array(repeating: 0.0, count: FXDataEngineConstants.tradeGateHidden)
        outputWeights[0] = 0.3
        outputWeights[1] = -0.25
        outputWeights[2] = 0.15

        return TradeGateNetworkState(
            inputWeights: inputWeights,
            hiddenBias: hiddenBias,
            outputWeights: outputWeights,
            outputBias: -0.05,
            ready: true,
            observations: 72
        )
    }
}
