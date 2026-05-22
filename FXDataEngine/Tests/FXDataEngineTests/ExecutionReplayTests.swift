import XCTest
@testable import FXDataEngine

final class ExecutionReplayTests: XCTestCase {
    private func traceSeries(volumes: [UInt64]) throws -> M1OHLCVSeries {
        let start = Int64(1_704_092_100)
        var utc: [Int64] = []
        var open: [Int64] = []
        var high: [Int64] = []
        var low: [Int64] = []
        var close: [Int64] = []
        var resolvedVolume: [UInt64] = []

        for index in 0..<14 {
            let base = Int64(1_000 + index)
            utc.append(start + Int64(index * 60))
            open.append(base)
            high.append(base + 5)
            low.append(base - 5)
            close.append(base + 1)
            resolvedVolume.append(index < volumes.count ? volumes[index] : 0)
        }

        let overrides: [Int: (Int64, Int64, Int64, Int64)] = [
            8: (1_000, 1_012, 996, 1_008),
            9: (1_008, 1_010, 996, 998),
            10: (998, 1_005, 994, 1_002),
            11: (1_002, 1_009, 1_000, 1_007),
            12: (1_007, 1_016, 1_001, 1_004),
            13: (1_004, 1_014, 1_000, 1_012)
        ]
        for (index, value) in overrides {
            open[index] = value.0
            high[index] = value.1
            low[index] = value.2
            close[index] = value.3
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "fixture",
                sourceOrigin: "unit-test",
                logicalSymbol: "EURUSD",
                timeframe: .m1,
                digits: 5
            ),
            utcTimestamps: ContiguousArray(utc),
            open: ContiguousArray(open),
            high: ContiguousArray(high),
            low: ContiguousArray(low),
            close: ContiguousArray(close),
            volume: ContiguousArray(resolvedVolume)
        )
    }

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

    func testTraceStatsUseVolumeLiquidityAndOHLCGeometry() throws {
        let series = try traceSeries(volumes: [
            300, 300, 300, 300, 300, 300, 300, 300,
            75, 200, 300, 100, 150, 300
        ])
        let trace = ExecutionReplayTools.buildTraceStats(
            series: series,
            index: 13,
            horizonMinutes: 5
        )

        XCTAssertEqual(trace.liquidityMeanRatio, 2.0833333333333335, accuracy: 1e-12)
        XCTAssertEqual(trace.liquidityPeakRatio, 4.0, accuracy: 1e-12)
        XCTAssertEqual(trace.rangeMeanRatio, 0.9404761904761905, accuracy: 1e-12)
        XCTAssertEqual(trace.bodyEfficiency, 0.48415103415103417, accuracy: 1e-12)
        XCTAssertEqual(trace.gapRatio, 0.2857142857142857, accuracy: 1e-12)
        XCTAssertEqual(trace.reversalRatio, 0.8, accuracy: 1e-12)
        XCTAssertEqual(trace.sessionTransitionExposure, 0.789, accuracy: 1e-12)
        XCTAssertEqual(trace.rolloverExposure, 0.0, accuracy: 0.0)
    }

    func testTraceStatsKeepNeutralLiquidityWhenVolumeMissing() throws {
        let series = try traceSeries(volumes: Array(repeating: 0, count: 14))
        let trace = ExecutionReplayTools.buildTraceStats(
            series: series,
            index: 13,
            horizonMinutes: 5
        )

        XCTAssertEqual(trace.liquidityMeanRatio, 1.0, accuracy: 0.0)
        XCTAssertEqual(trace.liquidityPeakRatio, 1.0, accuracy: 0.0)
        XCTAssertEqual(trace.rangeMeanRatio, 0.9404761904761905, accuracy: 1e-12)
        XCTAssertEqual(trace.reversalRatio, 0.8, accuracy: 1e-12)
    }
}
