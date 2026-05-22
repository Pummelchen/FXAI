import XCTest
@testable import FXDataEngine

final class RegimeGraphTests: XCTestCase {
    func testRegimeGraphDefaultsMatchLegacyClearState() {
        let query = RegimeGraphQuery()
        XCTAssertEqual(query.persistence, 0.0)
        XCTAssertEqual(query.transitionConfidence, 0.0)
        XCTAssertEqual(query.instability, 0.0)
        XCTAssertEqual(query.edgeBias, 0.0)
        XCTAssertEqual(query.qualityBias, 0.0)
        XCTAssertEqual(query.macroAlignment, 0.0)
        XCTAssertEqual(query.predictedRegime, 0)

        let graph = RegimeGraphState(regimeCount: 4)
        XCTAssertFalse(graph.ready)
        XCTAssertEqual(graph.lastRegime, -1)
        XCTAssertEqual(graph.lastTimeUTC, 0)
        XCTAssertEqual(graph.transitionObservations.count, 16)
    }

    func testRecordStateUpdatesDwellAndTransitionMass() {
        var graph = RegimeGraphState(regimeCount: 4)
        graph.recordState(regimeID: 1, sampleTimeUTC: 1_000, macroQuality: 0.5)
        XCTAssertFalse(graph.ready)
        XCTAssertEqual(graph.query(regimeID: 1, macroQuality: 0.2).predictedRegime, 1)

        graph.recordState(regimeID: 1, sampleTimeUTC: 1_060, macroQuality: 0.8)
        var query = graph.query(regimeID: 1, macroQuality: 0.2)
        XCTAssertTrue(graph.ready)
        XCTAssertEqual(query.persistence, 1.0, accuracy: 1e-12)
        XCTAssertEqual(query.transitionConfidence, 1.0, accuracy: 1e-12)
        XCTAssertEqual(query.instability, 0.0, accuracy: 1e-12)
        XCTAssertEqual(query.predictedRegime, 1)
        XCTAssertEqual(query.macroAlignment, 0.1384, accuracy: 1e-12)

        graph.recordState(regimeID: 3, sampleTimeUTC: 1_120, macroQuality: 0.4)
        query = graph.query(regimeID: 1, macroQuality: 0.2)
        XCTAssertEqual(query.persistence, 0.5, accuracy: 1e-12)
        XCTAssertEqual(query.transitionConfidence, 0.5, accuracy: 1e-12)
        XCTAssertEqual(query.instability, 0.5, accuracy: 1e-12)
        XCTAssertEqual(query.predictedRegime, 1)
    }

    func testFeedbackUpdatesEdgeQualityAndMacroBias() {
        var graph = RegimeGraphState(regimeCount: 4)
        graph.recordState(regimeID: 1, sampleTimeUTC: 1_000, macroQuality: 0.5)
        graph.recordState(regimeID: 3, sampleTimeUTC: 1_060, macroQuality: 0.4)
        graph.updateFeedback(
            fromRegime: 1,
            toRegime: 3,
            realizedEdgePoints: -4.0,
            qualityScore: 1.5,
            macroQuality: 0.7
        )

        let query = graph.query(regimeID: 1, macroQuality: 0.2)
        XCTAssertLessThan(query.edgeBias, 0.0)
        XCTAssertGreaterThan(query.qualityBias, 0.0)
        XCTAssertGreaterThan(query.macroAlignment, 0.0)
        XCTAssertEqual(query.predictedRegime, 3)

        graph.reset()
        XCTAssertFalse(graph.ready)
        XCTAssertEqual(graph.lastRegime, -1)
        XCTAssertEqual(graph.query(regimeID: 1, macroQuality: 0.2).predictedRegime, 1)
    }
}
