import XCTest
@testable import FXDataEngine

final class ExecutionReplayTests: XCTestCase {
    func testExecutionProfilePresetsAndEntryCostMatchLegacyFormula() {
        let prime = ExecutionProfile.preset(.primeECN)
        XCTAssertEqual(prime.profileID, .primeECN)
        XCTAssertEqual(prime.commissionPerLotSide, 3.5, accuracy: 0.0)
        XCTAssertEqual(prime.costBufferPoints, 1.5, accuracy: 0.0)
        XCTAssertEqual(prime.slippagePoints, 0.20, accuracy: 0.0)
        XCTAssertEqual(prime.fillPenaltyPoints, 0.15, accuracy: 0.0)
        XCTAssertEqual(
            ExecutionReplayTools.entryCostPoints(
                priceCostPoints: 0.0,
                commissionPoints: 1.2,
                baseCostBufferPoints: 0.4,
                profile: prime
            ),
            3.45,
            accuracy: 1e-12
        )

        let stress = ExecutionProfile.preset(.stress)
        XCTAssertEqual(stress.commissionPerLotSide, 5.0, accuracy: 0.0)
        XCTAssertEqual(stress.slippageStressWeight, 0.25, accuracy: 0.0)
        XCTAssertEqual(stress.allowedDeviationPoints, 8.0, accuracy: 0.0)
    }

    func testSlippageFillAndDeviationMatchLegacyFormulas() {
        let retail = ExecutionProfile.preset(.retailFX)
        XCTAssertEqual(
            ExecutionReplayTools.slippagePoints(
                profile: retail,
                roundTripCostPoints: 3.2,
                horizonMinutes: 16,
                liquidityStressPoints: 1.5,
                pathFlags: [.dualHit, .slowHit]
            ),
            1.694,
            accuracy: 1e-12
        )

        let stress = ExecutionProfile.preset(.stress)
        XCTAssertEqual(
            ExecutionReplayTools.fillPenaltyPoints(
                profile: stress,
                roundTripCostPoints: 4.0,
                liquidityStressPoints: 2.0,
                pathFlags: [.dualHit, .spreadStress]
            ),
            1.61,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            ExecutionReplayTools.allowedDeviationPoints(profile: stress, pathRisk: 0.6, fillRisk: 0.4),
            10.7,
            accuracy: 1e-12
        )
    }

    func testReplayFrameAndReplayCostsMatchLegacyScenarioFormula() {
        let profile = ExecutionProfile.preset(.defaultProfile)
        let frame = ExecutionReplayTools.buildReplayFrame(
            profile: profile,
            sampleTimeUTC: 1_704_097_800,
            horizonMinutes: 16,
            liquidityStressPoints: 0.0,
            pathFlags: [.dualHit],
            scenarioID: 11
        )

        XCTAssertEqual(frame.slippageMultiplier, 1.1920000000000002, accuracy: 1e-12)
        XCTAssertEqual(frame.fillMultiplier, 1.1115, accuracy: 1e-12)
        XCTAssertEqual(frame.latencyAddPoints, 0.0, accuracy: 0.0)
        XCTAssertEqual(frame.rejectProbability, 0.05695, accuracy: 1e-12)
        XCTAssertEqual(frame.partialFillProbability, 0.131, accuracy: 1e-12)
        XCTAssertEqual(frame.driftPenaltyPoints, 0.05500000000000001, accuracy: 1e-12)
        XCTAssertEqual(frame.eventFlags.rawValue, SamplePathFlags.dualHit.union(.slowHit).rawValue)

        XCTAssertEqual(
            ExecutionReplayTools.replaySlippagePoints(
                profile: profile,
                frame: frame,
                roundTripCostPoints: 3.0,
                horizonMinutes: 16,
                pathFlags: [.dualHit]
            ),
            0.9847600000000001,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            ExecutionReplayTools.replayFillPenaltyPoints(
                profile: profile,
                frame: frame,
                roundTripCostPoints: 3.0,
                pathFlags: [.dualHit]
            ),
            0.29426,
            accuracy: 1e-12
        )
    }
}
