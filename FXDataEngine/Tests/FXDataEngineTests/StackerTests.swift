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
}
