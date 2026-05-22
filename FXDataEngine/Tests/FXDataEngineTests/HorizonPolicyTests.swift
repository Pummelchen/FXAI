import XCTest
@testable import FXDataEngine

final class HorizonPolicyTests: XCTestCase {
    func testHorizonPolicyNetworkPredictAndUpdateMatchLegacyFormula() {
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.horizonPolicyFeatures)
        features[0] = 1.0
        features[1] = 2.0

        var inputWeights = Array(
            repeating: Array(repeating: 0.0, count: FXDataEngineConstants.horizonPolicyFeatures),
            count: FXDataEngineConstants.horizonPolicyHidden
        )
        inputWeights[0][0] = 0.5
        inputWeights[1][1] = -0.25
        var hiddenBias = Array(repeating: 0.0, count: FXDataEngineConstants.horizonPolicyHidden)
        hiddenBias[0] = 0.2
        hiddenBias[1] = -0.1
        var outputWeights = Array(repeating: 0.0, count: FXDataEngineConstants.horizonPolicyHidden)
        outputWeights[0] = 0.4
        outputWeights[1] = -0.3

        let state = HorizonPolicyNetworkState(
            inputWeights: inputWeights,
            hiddenBias: hiddenBias,
            outputWeights: outputWeights,
            outputBias: 0.1
        )
        let prediction = HorizonPolicyTools.predictValue(state, features: features)
        XCTAssertEqual(prediction.value, 0.502861980946276, accuracy: 1e-12)
        XCTAssertEqual(prediction.hidden[0], 0.6043677771171635, accuracy: 1e-12)
        XCTAssertEqual(prediction.hidden[1], -0.5370495669980352, accuracy: 1e-12)

        let updated = HorizonPolicyTools.updatedNetwork(state, features: features, rewardScaled: 1.25)
        XCTAssertTrue(updated.ready)
        XCTAssertEqual(updated.observations, 1)
        XCTAssertEqual(updated.outputBias, 0.11494276038107448, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[0], 0.40902452287550445, accuracy: 1e-12)
        XCTAssertEqual(updated.outputWeights[1], -0.30802020299241145, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[0], 0.20379390463899574, accuracy: 1e-12)
        XCTAssertEqual(updated.hiddenBias[1], -0.1031898807996526, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[0][0], 0.5037879046389957, accuracy: 1e-12)
        XCTAssertEqual(updated.inputWeights[1][1], -0.25637676159930517, accuracy: 1e-12)

        let updatedPrediction = HorizonPolicyTools.predictValue(updated, features: features)
        XCTAssertEqual(updatedPrediction.value, 0.5375438264943644, accuracy: 1e-12)
    }

    func testLegacyTanhCapsMatchMQLHelper() {
        XCTAssertEqual(HorizonPolicyTools.legacyTanh(19.0), 1.0, accuracy: 0.0)
        XCTAssertEqual(HorizonPolicyTools.legacyTanh(-19.0), -1.0, accuracy: 0.0)
        XCTAssertEqual(HorizonPolicyTools.legacyTanh(0.0), 0.0, accuracy: 0.0)
    }

    func testOOFHorizonPriorUpdateScoreAndGateMatchLegacyFormula() {
        let first = HorizonPolicyTools.updatedOOFPriorCell(
            OOFHorizonPriorCell(),
            scoreProxy: 5.0,
            edgeRatio: 1.5,
            quality: 0.75,
            tradeTarget: true
        )

        XCTAssertTrue(first.ready)
        XCTAssertEqual(first.scoreEMA, 5.0, accuracy: 0.0)
        XCTAssertEqual(first.edgeEMA, 1.5, accuracy: 0.0)
        XCTAssertEqual(first.qualityEMA, 0.75, accuracy: 0.0)
        XCTAssertEqual(first.tradeRateEMA, 1.0, accuracy: 0.0)
        XCTAssertEqual(first.observations, 1)
        XCTAssertEqual(HorizonPolicyTools.oofHorizonPriorScore(first), 0.1045, accuracy: 1e-12)
        XCTAssertEqual(HorizonPolicyTools.oofTradeGatePrior(first), 0.542, accuracy: 1e-12)

        let second = HorizonPolicyTools.updatedOOFPriorCell(
            first,
            scoreProxy: -3.0,
            edgeRatio: -1.0,
            quality: 0.25,
            tradeTarget: false
        )
        XCTAssertEqual(second.scoreEMA, 4.04, accuracy: 1e-12)
        XCTAssertEqual(second.edgeEMA, 1.2, accuracy: 1e-12)
        XCTAssertEqual(second.qualityEMA, 0.69, accuracy: 1e-12)
        XCTAssertEqual(second.tradeRateEMA, 0.88, accuracy: 1e-12)
        XCTAssertEqual(second.observations, 2)
        XCTAssertEqual(HorizonPolicyTools.oofHorizonPriorScore(second), 0.08344, accuracy: 1e-12)
        XCTAssertEqual(HorizonPolicyTools.oofTradeGatePrior(second), 0.53396, accuracy: 1e-12)
    }

    func testOOFHorizonPriorDefaultsAndObservationCapMatchLegacyRules() {
        XCTAssertEqual(HorizonPolicyTools.oofHorizonPriorScore(OOFHorizonPriorCell()), 0.0, accuracy: 0.0)
        XCTAssertEqual(HorizonPolicyTools.oofTradeGatePrior(OOFHorizonPriorCell()), -1.0, accuracy: 0.0)

        let capped = HorizonPolicyTools.updatedOOFPriorCell(
            OOFHorizonPriorCell(
                scoreEMA: 8.0,
                edgeEMA: 4.0,
                qualityEMA: 1.0,
                tradeRateEMA: 1.0,
                ready: true,
                observations: HorizonPolicyTools.observationCap
            ),
            scoreProxy: -10.0,
            edgeRatio: -10.0,
            quality: -5.0,
            tradeTarget: false
        )

        XCTAssertEqual(capped.scoreEMA, 7.7, accuracy: 1e-12)
        XCTAssertEqual(capped.edgeEMA, 3.85, accuracy: 1e-12)
        XCTAssertEqual(capped.qualityEMA, 0.975, accuracy: 1e-12)
        XCTAssertEqual(capped.tradeRateEMA, 0.975, accuracy: 1e-12)
        XCTAssertEqual(capped.observations, HorizonPolicyTools.observationCap)
    }
}
