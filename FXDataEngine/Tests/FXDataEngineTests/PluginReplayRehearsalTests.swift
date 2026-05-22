import XCTest
@testable import FXDataEngine

final class PluginReplayRehearsalTests: XCTestCase {
    func testReplayRehearsalSelectsLegacyTopTwoScores() throws {
        let entries = [
            entry(priority: 2.00, regimeID: 3, horizonMinutes: 15, labelClass: .buy, sampleTimeUTC: 1_000),
            entry(priority: 2.50, regimeID: 1, horizonMinutes: 5, labelClass: .sell, sampleTimeUTC: 1_001),
            entry(priority: 1.00, regimeID: 3, horizonMinutes: 5, labelClass: .skip, sampleTimeUTC: 1_002),
            entry(priority: 2.00, regimeID: 2, horizonMinutes: 30, labelClass: .buy, sampleTimeUTC: 1_003)
        ]

        let selected = PluginReplayRehearsalTools.selectedCandidates(
            entries: entries,
            regimeID: 3,
            horizonMinutes: 5
        )

        XCTAssertEqual(selected.count, 2)
        XCTAssertEqual(selected[0].sourceIndex, 1)
        XCTAssertEqual(selected[0].score, 3.10, accuracy: 1e-12)
        XCTAssertEqual(selected[1].sourceIndex, 0)
        XCTAssertEqual(selected[1].score, 2.80, accuracy: 1e-12)

        let firstRequest = selected[0].trainRequest
        try firstRequest.validate()
        XCTAssertEqual(firstRequest.context.regimeID, 1)
        XCTAssertEqual(firstRequest.context.horizonMinutes, 5)
        XCTAssertEqual(firstRequest.labelClass, .sell)
        XCTAssertEqual(firstRequest.fillRisk, 0.35, accuracy: 1e-12)
        XCTAssertEqual(firstRequest.nextVolumeTarget, 4.0, accuracy: 1e-12)
        XCTAssertTrue(firstRequest.context.dataHasVolume)
    }

    func testReplayRehearsalBuildsWindowedTrainRequestsAndSkipsInvalidEntries() throws {
        var row0 = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        row0[0] = 1.0
        row0[5] = 0.25
        var row1 = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        row1[0] = 1.0
        row1[5] = 0.75

        let valid = entry(
            priority: 1.0,
            regimeID: 4,
            horizonMinutes: 60,
            labelClass: .buy,
            sampleTimeUTC: 1_100,
            sequenceBars: 3,
            xWindow: [row0, row1],
            windowSize: 2
        )
        let invalid = entry(
            valid: false,
            priority: 99.0,
            regimeID: 4,
            horizonMinutes: 60,
            labelClass: .sell,
            sampleTimeUTC: 1_101
        )

        let requests = PluginReplayRehearsalTools.trainingRequests(
            entries: [invalid, valid],
            regimeID: 4,
            horizonMinutes: 60
        )

        XCTAssertEqual(requests.count, 1)
        let request = try XCTUnwrap(requests.first)
        try request.validate()
        XCTAssertEqual(request.windowSize, 2)
        XCTAssertEqual(request.xWindow.count, 2)
        XCTAssertEqual(request.context.sequenceBars, 3)
        XCTAssertEqual(request.xWindow[1][5], 0.75, accuracy: 1e-12)
    }

    func testRuntimeSampleEntryDefaultsVolumeAwareContextWhenVolumeTargetExists() throws {
        let sample = RuntimeArtifactPreparedSample(
            valid: true,
            labelClass: .buy,
            regimeID: 2,
            horizonMinutes: 15,
            horizonSlot: TrainingSampleTools.horizonSlot(horizonMinutes: 15),
            movePoints: 9.0,
            minMovePoints: 2.0,
            costPoints: 0.4,
            sampleWeight: 1.3,
            nextVolumeTarget: 11.0,
            sampleTimeUTC: 1_200,
            x: Array(repeating: 1.0, count: FXDataEngineConstants.aiWeights)
        )

        let entry = PluginReplayRehearsalEntry(sample: sample)
        XCTAssertTrue(entry.context.dataHasVolume)
        XCTAssertEqual(entry.context.sessionBucket, PluginContractTools.deriveSessionBucket(timestampUTC: 1_200))
        XCTAssertEqual(entry.nextVolumeTarget, 11.0, accuracy: 1e-12)

        let selected = PluginReplayRehearsalTools.selectedCandidates(
            entries: [entry],
            regimeID: 2,
            horizonMinutes: 15,
            replaySteps: 99
        )
        XCTAssertEqual(selected.count, 1)
        try selected[0].trainRequest.validate()
    }

    private func entry(
        valid: Bool = true,
        priority: Double,
        regimeID: Int,
        horizonMinutes: Int,
        labelClass: LabelClass,
        sampleTimeUTC: Int64,
        sequenceBars: Int = 1,
        xWindow: [[Double]] = [],
        windowSize: Int? = nil
    ) -> PluginReplayRehearsalEntry {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[0] = 1.0
        x[7] = 0.25
        let context = PluginContextV4(
            regimeID: regimeID,
            sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: sampleTimeUTC),
            horizonMinutes: horizonMinutes,
            sequenceBars: sequenceBars,
            priceCostPoints: 0.25,
            minMovePoints: 2.0,
            pointValue: 1.5,
            domainHash: 0.42,
            sampleTimeUTC: sampleTimeUTC,
            dataHasVolume: true
        )
        return PluginReplayRehearsalEntry(
            valid: valid,
            priority: priority,
            context: context,
            labelClass: labelClass,
            movePoints: labelClass == .sell ? -8.0 : 8.0,
            sampleWeight: 1.25,
            mfePoints: 12.0,
            maePoints: 3.0,
            timeToHitFraction: 0.4,
            pathFlags: SamplePathFlags.dualHit.rawValue,
            pathRisk: 0.20,
            fillRisk: 0.35,
            maskedStepTarget: 0.10,
            nextVolumeTarget: 4.0,
            regimeShiftTarget: 0.15,
            contextLeadTarget: 0.70,
            x: x,
            xWindow: xWindow,
            windowSize: windowSize
        )
    }
}
