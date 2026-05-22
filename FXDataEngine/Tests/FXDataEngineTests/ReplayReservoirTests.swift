import XCTest
@testable import FXDataEngine

final class ReplayReservoirTests: XCTestCase {
    func testLiquidityStressReplayPriorityUsesPathAndFillRisk() {
        let sample = runtimeSample(
            labelClass: .buy,
            sampleWeight: 1.4,
            qualityScore: 1.7,
            pathRisk: 0.30,
            fillRisk: 0.25,
            pathFlags: [.dualHit, .liquidityStress, .slowHit]
        )

        XCTAssertEqual(SamplePathFlags.liquidityStress.rawValue, 4)
        XCTAssertEqual(ReplaySampleFlags.liquidityStress.rawValue, 4)
        XCTAssertEqual(ReplayReservoirState.priority(for: sample), 2.65, accuracy: 1e-12)
    }

    func testReplayReservoirAddsDeduplicatesAndUpdatesAnalogMemory() {
        var reservoir = ReplayReservoirState()
        var analog = AnalogMemoryStore()
        let slot = TrainingSampleTools.horizonSlot(horizonMinutes: 15)
        let sample = runtimeSample(
            regimeID: 2,
            horizonMinutes: 15,
            horizonSlot: slot,
            sampleTimeUTC: 1_704_067_200,
            pathRisk: 0.40,
            fillRisk: 0.20
        )

        XCTAssertEqual(reservoir.add(sample, analogMemory: &analog), 0)
        XCTAssertNil(reservoir.add(sample, analogMemory: &analog))
        XCTAssertEqual(reservoir.count, 1)
        XCTAssertEqual(reservoir.bucketCounts[2][slot], 1)
        XCTAssertEqual(reservoir.lastSampleTimeUTCByHorizon[slot], 1_704_067_200)
        XCTAssertEqual(analog.size, 1)
        XCTAssertEqual(analog.entries[0].pathRisk, 0.40, accuracy: 1e-12)
        XCTAssertEqual(analog.entries[0].fillRisk, 0.20, accuracy: 1e-12)
    }

    func testReplayReservoirEvictsFromOverweightBucket() {
        var reservoir = ReplayReservoirState(capacity: 2)
        let firstSlot = TrainingSampleTools.horizonSlot(horizonMinutes: 5)
        let thirdSlot = TrainingSampleTools.horizonSlot(horizonMinutes: 15)
        let first = runtimeSample(regimeID: 1, horizonMinutes: 5, horizonSlot: firstSlot, sampleTimeUTC: 100)
        let second = runtimeSample(regimeID: 1, horizonMinutes: 5, horizonSlot: firstSlot, sampleTimeUTC: 101)
        let third = runtimeSample(regimeID: 2, horizonMinutes: 15, horizonSlot: thirdSlot, sampleTimeUTC: 102)

        XCTAssertEqual(reservoir.add(first), 0)
        XCTAssertEqual(reservoir.add(second), 1)
        let replaced = reservoir.add(third)

        XCTAssertNotNil(replaced)
        XCTAssertEqual(reservoir.count, 2)
        XCTAssertEqual(reservoir.bucketCounts[1][firstSlot], 1)
        XCTAssertEqual(reservoir.bucketCounts[2][thirdSlot], 1)
        XCTAssertTrue(reservoir.entries.contains { $0.used && $0.sample.sampleTimeUTC == 102 })
    }

    func testReplayBoostAndSelectionMatchLegacyDeterministicScoringWithLiquidityInputs() {
        var reservoir = ReplayReservoirState()
        let slot = TrainingSampleTools.horizonSlot(horizonMinutes: 5)
        let skip = runtimeSample(labelClass: .skip, regimeID: 0, horizonMinutes: 5, horizonSlot: slot, sampleTimeUTC: 200)
        let buy = runtimeSample(labelClass: .buy, regimeID: 0, horizonMinutes: 5, horizonSlot: slot, sampleTimeUTC: 201)

        _ = reservoir.add(skip)
        _ = reservoir.add(buy)
        reservoir.boostPriorityByOutcome(
            sampleTimeUTC: 200,
            horizonMinutes: 5,
            regimeID: 0,
            labelClass: .skip,
            signal: 1,
            movePoints: 12.0,
            minMovePoints: 4.0
        )

        XCTAssertTrue(reservoir.entries[0].flags.contains(.falsePositive))
        XCTAssertGreaterThan(reservoir.entries[0].priority, reservoir.entries[1].priority)

        let selected = reservoir.selectReplaySamples(regimeID: 0, horizonMinutes: 5, epochs: 1, drawsPerEpoch: 1)
        XCTAssertEqual(selected.count, 1)
        XCTAssertEqual(selected[0].sampleTimeUTC, 200)
        XCTAssertEqual(reservoir.cursor, 1)
    }

    func testReplayReservoirCodecRoundTripsLegacySection() throws {
        var reservoir = ReplayReservoirState()
        let slot = TrainingSampleTools.horizonSlot(horizonMinutes: 15)
        _ = reservoir.add(runtimeSample(regimeID: 2, horizonMinutes: 15, horizonSlot: slot, sampleTimeUTC: 300))
        reservoir.boostPriorityByOutcome(
            sampleTimeUTC: 300,
            horizonMinutes: 15,
            regimeID: 2,
            labelClass: .sell,
            signal: 1,
            movePoints: -20.0,
            minMovePoints: 5.0
        )

        let encoded = try RuntimeReplayReservoirCodec.encode(reservoir)
        let decoded = try RuntimeReplayReservoirCodec.decode(from: encoded)

        XCTAssertEqual(encoded.count, RuntimeReplayReservoirCodec.byteCount)
        XCTAssertEqual(decoded.count, 1)
        XCTAssertEqual(decoded.bucketCounts[2][slot], 1)
        XCTAssertEqual(decoded.entries[0].sample.sampleTimeUTC, 300)
        XCTAssertTrue(decoded.entries[0].flags.contains(.wrongDirection))
    }

    private func runtimeSample(
        labelClass: LabelClass = .buy,
        regimeID: Int = 0,
        horizonMinutes: Int = 5,
        horizonSlot: Int = -1,
        sampleTimeUTC: Int64 = 1,
        sampleWeight: Double = 1.0,
        qualityScore: Double = 1.0,
        pathRisk: Double = 0.0,
        fillRisk: Double = 0.0,
        pathFlags: SamplePathFlags = []
    ) -> RuntimeArtifactPreparedSample {
        let resolvedHorizonSlot = horizonSlot >= 0 ? horizonSlot : TrainingSampleTools.horizonSlot(horizonMinutes: horizonMinutes)
        let x = [1.0, 0.25, -0.5] + Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights - 3)
        return RuntimeArtifactPreparedSample(sample: PreparedTrainingSample(
            valid: true,
            labelClass: labelClass,
            regimeID: regimeID,
            horizonMinutes: horizonMinutes,
            horizonSlot: resolvedHorizonSlot,
            movePoints: 10.0,
            minMovePoints: 5.0,
            sampleWeight: sampleWeight,
            qualityScore: qualityScore,
            pathFlags: pathFlags,
            pathRisk: pathRisk,
            fillRisk: fillRisk,
            sampleTimeUTC: sampleTimeUTC,
            x: x
        ))
    }
}
