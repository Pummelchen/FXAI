import XCTest
@testable import FXDataEngine

final class MetaPendingReplayTests: XCTestCase {
    func testMetaPendingReplayQueueUsesKindFeatureCountsAndLegacyReplaceOverflow() {
        var queue = MetaPendingReplayQueue(kind: .policy, capacity: 3)
        queue = MetaPendingReplayTools.enqueuedPending(
            queue,
            signalSequence: 1,
            regimeID: 2,
            horizonMinutes: 5,
            minMovePoints: 1.5,
            features: [1.0, 2.0, 3.0]
        )
        queue = MetaPendingReplayTools.enqueuedPending(
            queue,
            signalSequence: 1,
            regimeID: 3,
            horizonMinutes: 8,
            minMovePoints: 2.5,
            features: [4.0]
        )

        XCTAssertEqual(queue.activeEntries().map(\.signalSequence), [1])
        XCTAssertEqual(queue.activeEntries()[0].regimeID, 3)
        XCTAssertEqual(queue.activeEntries()[0].horizonMinutes, 8)
        XCTAssertEqual(queue.activeEntries()[0].minMovePoints, 2.5, accuracy: 0.0)
        XCTAssertEqual(queue.activeEntries()[0].features.count, FXDataEngineConstants.policyFeatures)
        XCTAssertEqual(queue.activeEntries()[0].features[0], 4.0, accuracy: 0.0)
        XCTAssertEqual(queue.activeEntries()[0].features[1], 0.0, accuracy: 0.0)

        queue = MetaPendingReplayTools.enqueuedPending(
            queue,
            signalSequence: 2,
            regimeID: 4,
            horizonMinutes: 13,
            features: []
        )
        queue = MetaPendingReplayTools.enqueuedPending(
            queue,
            signalSequence: 3,
            regimeID: 5,
            horizonMinutes: 21,
            features: []
        )

        XCTAssertEqual(queue.activeEntries().map(\.signalSequence), [2, 3])
    }

    func testStackPendingReplayResolutionConsumesAgedEntriesAndKeepsYoungEntries() {
        var queue = MetaPendingReplayQueue(kind: .stack, capacity: 5)
        queue = MetaPendingReplayTools.enqueuedPending(
            queue,
            signalSequence: 2,
            signal: LabelClass.buy.rawValue,
            regimeID: 1,
            horizonMinutes: 5,
            expectedMovePoints: 4.0,
            probabilities: [0.10, 0.70, 0.20],
            features: [0.25, 0.50]
        )
        queue = MetaPendingReplayTools.enqueuedPending(
            queue,
            signalSequence: 10,
            signal: LabelClass.sell.rawValue,
            regimeID: 3,
            horizonMinutes: 5,
            expectedMovePoints: 2.0,
            probabilities: [0.70, 0.10, 0.20],
            features: []
        )
        queue = MetaPendingReplayTools.enqueuedPending(
            queue,
            signalSequence: 8,
            signal: -1,
            regimeID: 4,
            horizonMinutes: 3,
            expectedMovePoints: 1.0,
            probabilities: [0.20, 0.20, 0.60],
            features: []
        )

        let resolution = MetaPendingReplayTools.resolvedPendingOutcomes(
            queue,
            currentSignalSequence: 13,
            availableBarCount: 14
        )

        XCTAssertEqual(resolution.keptQueue.kind, .stack)
        XCTAssertEqual(resolution.keptQueue.activeEntries().map(\.signalSequence), [10])
        XCTAssertEqual(resolution.outcomeActions.map { $0.entry.signalSequence }, [2, 8])
        XCTAssertEqual(resolution.outcomeActions.map(\.age), [11, 5])
        XCTAssertEqual(resolution.outcomeActions.map(\.predictionIndex), [11, 5])
        XCTAssertEqual(resolution.outcomeActions.map(\.canEvaluate), [true, true])
        XCTAssertEqual(resolution.outcomeActions[0].entry.features.count, FXDataEngineConstants.stackFeatures)
        XCTAssertEqual(resolution.outcomeActions[0].entry.probabilities[LabelClass.buy.rawValue], 0.70, accuracy: 0.0)
    }

    func testMetaPendingReplayResolutionDropsAgedEntriesWhenHistoryCannotEvaluate() {
        var queue = MetaPendingReplayQueue(kind: .horizonPolicy, capacity: 4)
        queue = MetaPendingReplayTools.enqueuedPending(
            queue,
            signalSequence: 1,
            regimeID: 2,
            horizonMinutes: 5,
            minMovePoints: 1.5,
            features: [1.0, 2.0]
        )

        let unchanged = MetaPendingReplayTools.resolvedPendingOutcomes(
            queue,
            currentSignalSequence: -1,
            availableBarCount: 0
        )
        XCTAssertEqual(unchanged.keptQueue.activeEntries().map(\.signalSequence), [1])
        XCTAssertTrue(unchanged.outcomeActions.isEmpty)

        let resolution = MetaPendingReplayTools.resolvedPendingOutcomes(
            queue,
            currentSignalSequence: 7,
            availableBarCount: 5
        )
        XCTAssertEqual(resolution.keptQueue.kind, .horizonPolicy)
        XCTAssertTrue(resolution.keptQueue.activeEntries().isEmpty)
        XCTAssertEqual(resolution.outcomeActions.count, 1)
        XCTAssertEqual(resolution.outcomeActions[0].predictionIndex, 6)
        XCTAssertFalse(resolution.outcomeActions[0].canEvaluate)
        XCTAssertEqual(resolution.outcomeActions[0].entry.features.count, FXDataEngineConstants.horizonPolicyFeatures)
    }
}
