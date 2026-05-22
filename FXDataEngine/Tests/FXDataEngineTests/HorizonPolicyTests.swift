import XCTest
@testable import FXDataEngine

final class HorizonPolicyTests: XCTestCase {
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
