import XCTest
@testable import FXDataEngine

final class ModelReliabilityTests: XCTestCase {
    func testReliabilityUpdateMatchesLegacyFormula() {
        XCTAssertEqual(
            ModelReliabilityTools.updatedReliability(
                currentReliability: 1.0,
                labelClass: .buy,
                signal: LabelClass.buy.rawValue,
                realizedMovePoints: 4.2,
                minMovePoints: 2.0,
                expectedMovePoints: 4.0,
                probabilities: [0.10, 0.72, 0.18]
            ),
            1.02265,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            ModelReliabilityTools.updatedReliability(
                currentReliability: 1.25,
                labelClass: .sell,
                signal: LabelClass.buy.rawValue,
                realizedMovePoints: -1.0,
                minMovePoints: 2.0,
                expectedMovePoints: 3.0,
                probabilities: [0.58, 0.30, 0.12]
            ),
            1.22353,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            ModelReliabilityTools.updatedReliability(
                currentReliability: 0.8,
                labelClass: .skip,
                signal: -1,
                realizedMovePoints: 0.4,
                minMovePoints: 1.5,
                expectedMovePoints: 1.5,
                probabilities: [0.20, 0.10, 0.70]
            ),
            0.8126,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            ModelReliabilityTools.updatedReliability(
                currentReliability: 1.4,
                labelClass: .buy,
                signal: -1,
                realizedMovePoints: 6.0,
                minMovePoints: 2.0,
                expectedMovePoints: 3.0,
                probabilities: [0.20, 0.60, 0.20]
            ),
            1.376,
            accuracy: 1e-12
        )
    }

    func testReliabilityClockMatchesLegacySequenceRules() {
        let initial = ReliabilityClock()
        let first = ModelReliabilityTools.advancedClock(initial, signalBarTimeUTC: 1_704_067_200)
        XCTAssertEqual(first.barTimeUTC, 1_704_067_200)
        XCTAssertEqual(first.sequence, 0)

        let unchanged = ModelReliabilityTools.advancedClock(first, signalBarTimeUTC: 1_704_067_200)
        XCTAssertEqual(unchanged, first)

        let advanced = ModelReliabilityTools.advancedClock(first, signalBarTimeUTC: 1_704_067_260)
        XCTAssertEqual(advanced.barTimeUTC, 1_704_067_260)
        XCTAssertEqual(advanced.sequence, 1)

        XCTAssertEqual(ModelReliabilityTools.advancedClock(advanced, signalBarTimeUTC: 0), advanced)
    }

    func testReliabilityPendingQueueEnqueueReplaceAndOverflowMatchLegacyRing() {
        var queue = ReliabilityPendingQueue(capacity: 3)
        queue = ModelReliabilityTools.enqueuedPending(
            queue,
            signalSequence: 10,
            signal: LabelClass.buy.rawValue,
            regimeID: 2,
            sessionBucket: 1,
            expectedMovePoints: 3.5,
            horizonMinutes: 5,
            probabilities: [0.10, 0.70, 0.20]
        )
        queue = ModelReliabilityTools.enqueuedPending(
            queue,
            signalSequence: 10,
            signal: LabelClass.sell.rawValue,
            regimeID: 3,
            sessionBucket: 2,
            expectedMovePoints: 4.5,
            horizonMinutes: 8,
            probabilities: [0.60, 0.20, 0.20]
        )

        XCTAssertEqual(queue.activeEntries().map(\.signalSequence), [10])
        XCTAssertEqual(queue.activeEntries()[0].signal, LabelClass.sell.rawValue)
        XCTAssertEqual(queue.activeEntries()[0].regimeID, 3)
        XCTAssertEqual(queue.activeEntries()[0].horizonMinutes, 8)
        XCTAssertEqual(queue.activeEntries()[0].probabilities[LabelClass.sell.rawValue], 0.60, accuracy: 1e-12)

        queue = ModelReliabilityTools.enqueuedPending(
            queue,
            signalSequence: 11,
            signal: -1,
            regimeID: 4,
            sessionBucket: 0,
            expectedMovePoints: 2.0,
            horizonMinutes: 13,
            probabilities: [0.20, 0.20, 0.60]
        )
        queue = ModelReliabilityTools.enqueuedPending(
            queue,
            signalSequence: 12,
            signal: LabelClass.buy.rawValue,
            regimeID: 5,
            sessionBucket: 0,
            expectedMovePoints: 2.0,
            horizonMinutes: 21,
            probabilities: [0.20, 0.70, 0.10]
        )

        XCTAssertEqual(queue.activeEntries().map(\.signalSequence), [11, 12])
        XCTAssertEqual(
            ModelReliabilityTools.maxReliabilityPendingHorizon(fallbackHorizonMinutes: 5, queues: [queue]),
            21
        )
    }

    func testReliabilityPendingResolutionConsumesAgedEntriesAndKeepsYoungEntries() {
        var queue = ReliabilityPendingQueue(capacity: 5)
        queue = ModelReliabilityTools.enqueuedPending(
            queue,
            signalSequence: 2,
            signal: LabelClass.buy.rawValue,
            regimeID: 1,
            sessionBucket: 2,
            expectedMovePoints: 4.0,
            horizonMinutes: 5,
            probabilities: [0.10, 0.70, 0.20]
        )
        queue = ModelReliabilityTools.enqueuedPending(
            queue,
            signalSequence: 10,
            signal: LabelClass.sell.rawValue,
            regimeID: 3,
            sessionBucket: 1,
            expectedMovePoints: 2.0,
            horizonMinutes: 5,
            probabilities: [0.70, 0.10, 0.20]
        )
        queue = ModelReliabilityTools.enqueuedPending(
            queue,
            signalSequence: 8,
            signal: -1,
            regimeID: 4,
            sessionBucket: 0,
            expectedMovePoints: 1.0,
            horizonMinutes: 3,
            probabilities: [0.20, 0.20, 0.60]
        )

        let resolution = ModelReliabilityTools.resolvedPendingOutcomes(
            queue,
            currentSignalSequence: 13,
            availableBarCount: 14
        )

        XCTAssertEqual(resolution.keptQueue.activeEntries().map(\.signalSequence), [10])
        XCTAssertEqual(resolution.outcomeActions.map { $0.entry.signalSequence }, [2, 8])
        XCTAssertEqual(resolution.outcomeActions.map(\.age), [11, 5])
        XCTAssertEqual(resolution.outcomeActions.map(\.predictionIndex), [11, 5])
        XCTAssertEqual(resolution.outcomeActions.map(\.futureIndex), [6, 2])
        XCTAssertEqual(resolution.outcomeActions.map(\.canEvaluate), [true, true])
    }

    func testReliabilityPendingResolutionDropsAgedEntriesWhenHistoryCannotEvaluate() {
        var queue = ReliabilityPendingQueue(capacity: 4)
        queue = ModelReliabilityTools.enqueuedPending(
            queue,
            signalSequence: 1,
            signal: LabelClass.buy.rawValue,
            regimeID: 1,
            sessionBucket: 2,
            expectedMovePoints: 4.0,
            horizonMinutes: 5,
            probabilities: [0.10, 0.70, 0.20]
        )

        let unchanged = ModelReliabilityTools.resolvedPendingOutcomes(
            queue,
            currentSignalSequence: -1,
            availableBarCount: 0
        )
        XCTAssertEqual(unchanged.keptQueue.activeEntries().map(\.signalSequence), [1])
        XCTAssertTrue(unchanged.outcomeActions.isEmpty)

        let resolution = ModelReliabilityTools.resolvedPendingOutcomes(
            queue,
            currentSignalSequence: 7,
            availableBarCount: 5
        )
        XCTAssertTrue(resolution.keptQueue.activeEntries().isEmpty)
        XCTAssertEqual(resolution.outcomeActions.count, 1)
        XCTAssertEqual(resolution.outcomeActions[0].predictionIndex, 6)
        XCTAssertEqual(resolution.outcomeActions[0].futureIndex, 1)
        XCTAssertFalse(resolution.outcomeActions[0].canEvaluate)
    }

    func testVoteWeightClampsLikeLegacyAccessor() {
        XCTAssertEqual(ModelReliabilityTools.voteWeight(aiIndex: -1, reliabilities: [0.1]), 1.0, accuracy: 0.0)
        XCTAssertEqual(ModelReliabilityTools.voteWeight(aiIndex: 3, reliabilities: [0.1]), 1.0, accuracy: 0.0)
        XCTAssertEqual(ModelReliabilityTools.voteWeight(aiIndex: 0, reliabilities: [0.1]), 0.20, accuracy: 0.0)
        XCTAssertEqual(ModelReliabilityTools.voteWeight(aiIndex: 1, reliabilities: [0.1, 3.5]), 3.00, accuracy: 0.0)
        XCTAssertEqual(ModelReliabilityTools.voteWeight(aiIndex: 1, reliabilities: [0.1, 1.7]), 1.70, accuracy: 0.0)
    }
}
