import XCTest
@testable import FXDataEngine

final class StackerTests: XCTestCase {
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
}
