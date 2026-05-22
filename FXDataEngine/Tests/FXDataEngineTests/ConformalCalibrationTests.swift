import XCTest
@testable import FXDataEngine

final class ConformalCalibrationTests: XCTestCase {
    func testConformalQuantileAndPushScoreMatchLegacyRules() {
        var state = ConformalCalibrationState()
        let slot = TrainingSampleTools.horizonSlot(horizonMinutes: 5)

        XCTAssertEqual(
            state.quantile(aiIndex: 0, regimeID: 1, horizonSlot: slot, scoreKind: .classScore, fallback: 0.35),
            0.35,
            accuracy: 1e-12
        )

        _ = state.pushScore(aiIndex: 0, regimeID: 1, horizonSlot: slot, classScore: 0.10, moveScore: 0.20, pathScore: 0.30)
        _ = state.pushScore(aiIndex: 0, regimeID: 1, horizonSlot: slot, classScore: 0.90, moveScore: 0.40, pathScore: 0.50)
        _ = state.pushScore(aiIndex: 0, regimeID: 1, horizonSlot: slot, classScore: 0.20, moveScore: 0.60, pathScore: 0.70)
        _ = state.pushScore(aiIndex: 0, regimeID: 1, horizonSlot: slot, classScore: 0.70, moveScore: 0.80, pathScore: 0.90)

        XCTAssertEqual(
            state.quantile(aiIndex: 0, regimeID: 1, horizonSlot: slot, scoreKind: .classScore, fallback: 0.35),
            0.70,
            accuracy: 1e-12
        )
        XCTAssertEqual(state.aiStates[0].counts[ConformalCalibrationAIState.cellIndex(regimeID: 1, horizonSlot: slot)], 4)
    }

    func testConformalPendingQueueReplacesLastSequenceAndKeepsRingOrder() {
        var state = ConformalCalibrationState()
        let first = PredictionV4(
            classProbabilities: [0.10, 0.20, 0.70],
            moveQ25Points: 1.0,
            moveQ50Points: 2.0,
            moveQ75Points: 3.0,
            pathRisk: 0.30
        )
        let replacement = PredictionV4(
            classProbabilities: [0.30, 0.40, 0.30],
            moveQ25Points: 4.0,
            moveQ50Points: 5.0,
            moveQ75Points: 6.0,
            pathRisk: 0.60
        )

        XCTAssertTrue(state.enqueuePending(aiIndex: 2, signalSequence: 10, regimeID: 3, horizonMinutes: 8, prediction: first))
        XCTAssertTrue(state.enqueuePending(aiIndex: 2, signalSequence: 10, regimeID: 4, horizonMinutes: 13, prediction: replacement))

        let pending = state.pendingEntries(aiIndex: 2)
        XCTAssertEqual(pending.count, 1)
        XCTAssertEqual(pending[0].regimeID, 4)
        XCTAssertEqual(pending[0].horizonMinutes, 13)
        XCTAssertEqual(pending[0].classProbabilities[1], 0.40, accuracy: 1e-12)
        XCTAssertEqual(pending[0].moveQ50Points, 5.0, accuracy: 1e-12)
        XCTAssertEqual(pending[0].pathRisk, 0.60, accuracy: 1e-12)
        XCTAssertEqual(ConformalPendingEntry.sanitizedProbabilities([0.30, 0.40]), [0.30, 0.40, 0.0])
    }

    func testConformalUpdateFromPendingUsesNoSpreadFillRisk() {
        var state = ConformalCalibrationState()
        let prediction = PredictionV4(
            classProbabilities: [0.10, 0.70, 0.20],
            moveQ25Points: 1.0,
            moveQ50Points: 2.0,
            moveQ75Points: 5.0,
            pathRisk: 0.40
        )
        XCTAssertTrue(state.enqueuePending(aiIndex: 1, signalSequence: 42, regimeID: 2, horizonMinutes: 5, prediction: prediction))
        XCTAssertTrue(state.enqueuePending(aiIndex: 1, signalSequence: 43, regimeID: 2, horizonMinutes: 5, prediction: prediction))

        let updated = state.updateFromPending(aiIndex: 1, outcome: ConformalRealizedOutcome(
            signalSequence: 42,
            labelClass: .buy,
            realizedMovePoints: 3.0,
            mfePoints: 3.0,
            maePoints: 0.5,
            timeToHitFraction: 0.25,
            minMovePoints: 1.0
        ))

        let slot = TrainingSampleTools.horizonSlot(horizonMinutes: 5)
        XCTAssertTrue(updated)
        XCTAssertEqual(
            state.quantile(aiIndex: 1, regimeID: 2, horizonSlot: slot, scoreKind: .classScore, fallback: 0.0),
            0.30,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            state.quantile(aiIndex: 1, regimeID: 2, horizonSlot: slot, scoreKind: .moveScore, fallback: 0.0),
            0.25,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            state.quantile(aiIndex: 1, regimeID: 2, horizonSlot: slot, scoreKind: .pathScore, fallback: 0.0),
            0.295,
            accuracy: 1e-12
        )
        XCTAssertEqual(state.pendingEntries(aiIndex: 1).map(\.signalSequence), [43])
    }

    func testConformalPredictionAdjustmentMatchesLegacyFormula() throws {
        var state = ConformalCalibrationState()
        let slot = TrainingSampleTools.horizonSlot(horizonMinutes: 5)
        _ = state.pushScore(aiIndex: 0, regimeID: 0, horizonSlot: slot, classScore: 0.50, moveScore: 2.0, pathScore: 0.40)
        let prediction = PredictionV4(
            classProbabilities: [0.20, 0.60, 0.20],
            moveMeanPoints: 5.0,
            moveQ25Points: 2.0,
            moveQ50Points: 4.0,
            moveQ75Points: 6.0,
            pathRisk: 0.10,
            fillRisk: 0.20,
            confidence: 0.90,
            reliability: 0.90
        )

        let adjusted = state.applyingAdjustment(
            aiIndex: 0,
            regimeID: 0,
            horizonMinutes: 5,
            minMovePoints: 1.0,
            to: prediction
        )

        XCTAssertEqual(adjusted.classProbabilities[0], 0.1568, accuracy: 1e-12)
        XCTAssertEqual(adjusted.classProbabilities[1], 0.4704, accuracy: 1e-12)
        XCTAssertEqual(adjusted.classProbabilities[2], 0.3728, accuracy: 1e-12)
        XCTAssertEqual(adjusted.moveMeanPoints, 4.70, accuracy: 1e-12)
        XCTAssertEqual(adjusted.moveQ25Points, 0.0, accuracy: 1e-12)
        XCTAssertEqual(adjusted.moveQ50Points, 4.0, accuracy: 1e-12)
        XCTAssertEqual(adjusted.moveQ75Points, 8.0, accuracy: 1e-12)
        XCTAssertEqual(adjusted.pathRisk, 0.262, accuracy: 1e-12)
        XCTAssertEqual(adjusted.fillRisk, 0.332, accuracy: 1e-12)
        XCTAssertEqual(adjusted.confidence, 0.4704, accuracy: 1e-12)
        XCTAssertEqual(adjusted.reliability, 0.7425, accuracy: 1e-12)
        try adjusted.validate()
    }

    func testConformalCalibrationCodecRoundTripsLegacySection() throws {
        var state = ConformalCalibrationState()
        let slot = TrainingSampleTools.horizonSlot(horizonMinutes: 21)
        _ = state.pushScore(aiIndex: 3, regimeID: 4, horizonSlot: slot, classScore: 0.44, moveScore: 1.25, pathScore: 0.37)
        _ = state.enqueuePending(
            aiIndex: 3,
            signalSequence: 77,
            regimeID: 4,
            horizonMinutes: 21,
            prediction: PredictionV4(
                classProbabilities: [0.25, 0.50, 0.25],
                moveQ25Points: 2.0,
                moveQ50Points: 3.0,
                moveQ75Points: 5.0,
                pathRisk: 0.33
            )
        )

        let encoded = try RuntimeConformalCalibrationCodec.encode(state)
        let decoded = try RuntimeConformalCalibrationCodec.decode(from: encoded)

        XCTAssertEqual(encoded.count, RuntimeConformalCalibrationCodec.byteCount)
        XCTAssertEqual(
            decoded.quantile(aiIndex: 3, regimeID: 4, horizonSlot: slot, scoreKind: .classScore, fallback: 0.0),
            0.44,
            accuracy: 1e-12
        )
        XCTAssertEqual(decoded.pendingEntries(aiIndex: 3).first?.signalSequence, 77)
        XCTAssertEqual(decoded.pendingEntries(aiIndex: 3).first?.classProbabilities[1] ?? 0.0, 0.50, accuracy: 1e-12)
    }
}
