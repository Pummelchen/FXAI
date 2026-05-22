import XCTest
@testable import FXDataEngine

final class MetaPendingReplayOutcomeTests: XCTestCase {
    func testStackPendingReplayOutcomeBuildsM1TargetsAndUpdatesPreparedStates() throws {
        let series = try makeSeries(highOverrides: [3: 1_006])
        let entry = MetaPendingReplayEntry(
            kind: .stack,
            signalSequence: 2,
            signal: -1,
            regimeID: 1,
            horizonMinutes: 5,
            expectedMovePoints: 4.0,
            probabilities: [0.10, 0.70, 0.20],
            features: Array(repeating: 0.25, count: FXDataEngineConstants.stackFeatures)
        )
        let action = MetaPendingReplayOutcomeAction(
            entry: entry,
            age: 5,
            predictionIndex: 5,
            canEvaluate: true
        )

        let outcome = try XCTUnwrap(MetaPendingReplayTools.outcomeTargets(
            for: action,
            series: series,
            config: MetaPendingReplayOutcomeConfig(
                currentBarIndex: 7,
                priceCostPoints: 1.0,
                executionProfile: ExecutionProfile(costBufferPoints: 0.0)
            )
        ))

        XCTAssertEqual(outcome.predictionSeriesIndex, 2)
        XCTAssertEqual(outcome.labelClass, .buy)
        XCTAssertEqual(outcome.minMovePoints, 1.0, accuracy: 1e-12)
        XCTAssertGreaterThan(outcome.realizedEdgePoints, 0.0)
        XCTAssertGreaterThan(outcome.stack.sampleWeight, 3.0)
        XCTAssertTrue(outcome.stack.tradeTarget)
        XCTAssertEqual(outcome.stack.predictedProbabilities[LabelClass.buy.rawValue], 0.70, accuracy: 1e-12)

        let stackState = MetaPendingReplayTools.updatedStackNetwork(StackNetworkState(), with: outcome)
        XCTAssertTrue(stackState.ready)
        XCTAssertEqual(stackState.observations, 1)

        let tradeGateState = MetaPendingReplayTools.updatedTradeGateNetwork(TradeGateNetworkState(), with: outcome)
        XCTAssertTrue(tradeGateState.ready)
        XCTAssertEqual(tradeGateState.observations, 1)

        let cells = MetaPendingReplayTools.observedStackRouterCells([], with: outcome)
        XCTAssertEqual(cells[LabelClass.buy.rawValue].observations, 1)
        XCTAssertTrue(cells[LabelClass.buy.rawValue].ready)
    }

    func testPolicyAndHorizonOutcomesBuildTargetsAndApplyPureUpdates() throws {
        let series = try makeSeries(highOverrides: [3: 1_006])
        let config = MetaPendingReplayOutcomeConfig(
            currentBarIndex: 7,
            priceCostPoints: 1.0,
            executionProfile: ExecutionProfile(costBufferPoints: 0.0),
            currentRegimeID: 4,
            currentMacroQuality: 0.80
        )

        let policyEntry = MetaPendingReplayEntry(
            kind: .policy,
            signalSequence: 2,
            regimeID: 1,
            horizonMinutes: 5,
            minMovePoints: 1.0,
            features: Array(repeating: 0.10, count: FXDataEngineConstants.policyFeatures)
        )
        let policyOutcome = try XCTUnwrap(MetaPendingReplayTools.outcomeTargets(
            for: MetaPendingReplayOutcomeAction(
                entry: policyEntry,
                age: 5,
                predictionIndex: 5,
                canEvaluate: true
            ),
            series: series,
            config: config
        ))

        XCTAssertEqual(policyOutcome.labelClass, .buy)
        XCTAssertEqual(policyOutcome.policy.tradeTarget, 1.0, accuracy: 1e-12)
        XCTAssertEqual(policyOutcome.policy.directionTarget, 1.0, accuracy: 1e-12)
        XCTAssertEqual(policyOutcome.policy.sizeTarget, 1.60, accuracy: 1e-12)
        XCTAssertGreaterThan(policyOutcome.policy.holdTarget, 0.70)

        let policyState = MetaPendingReplayTools.updatedPolicyNetwork(MetaPolicyNetworkState(), with: policyOutcome)
        XCTAssertTrue(policyState.ready)
        XCTAssertEqual(policyState.observations, 1)

        let graph = MetaPendingReplayTools.updatedRegimeGraph(RegimeGraphState(), with: policyOutcome, config: config)
        XCTAssertTrue(graph.ready)

        let horizonEntry = MetaPendingReplayEntry(
            kind: .horizonPolicy,
            signalSequence: 2,
            regimeID: 1,
            horizonMinutes: 5,
            minMovePoints: 1.0,
            features: Array(repeating: 0.10, count: FXDataEngineConstants.horizonPolicyFeatures)
        )
        let horizonOutcome = try XCTUnwrap(MetaPendingReplayTools.outcomeTargets(
            for: MetaPendingReplayOutcomeAction(
                entry: horizonEntry,
                age: 5,
                predictionIndex: 5,
                canEvaluate: true
            ),
            series: series,
            config: config
        ))

        XCTAssertGreaterThan(horizonOutcome.horizonPolicyReward, 5.0)
        let horizonState = MetaPendingReplayTools.updatedHorizonPolicyNetwork(HorizonPolicyNetworkState(), with: horizonOutcome)
        XCTAssertTrue(horizonState.ready)
        XCTAssertEqual(horizonState.observations, 1)
    }

    func testPendingReplayOutcomeDoesNotReadBeyondResolvedHorizon() throws {
        let series = try makeSeries(highOverrides: [8: 1_020])
        let config = MetaPendingReplayOutcomeConfig(
            currentBarIndex: 8,
            priceCostPoints: 1.0,
            executionProfile: ExecutionProfile(costBufferPoints: 0.0)
        )
        let horizonFive = MetaPendingReplayEntry(
            kind: .stack,
            signalSequence: 2,
            regimeID: 1,
            horizonMinutes: 5,
            features: []
        )
        let shortOutcome = try XCTUnwrap(MetaPendingReplayTools.outcomeTargets(
            for: MetaPendingReplayOutcomeAction(
                entry: horizonFive,
                age: 6,
                predictionIndex: 6,
                canEvaluate: true
            ),
            series: series,
            config: config
        ))
        XCTAssertEqual(shortOutcome.labelClass, .skip)

        let horizonSix = MetaPendingReplayEntry(
            kind: .stack,
            signalSequence: 2,
            regimeID: 1,
            horizonMinutes: 6,
            features: []
        )
        let fullOutcome = try XCTUnwrap(MetaPendingReplayTools.outcomeTargets(
            for: MetaPendingReplayOutcomeAction(
                entry: horizonSix,
                age: 6,
                predictionIndex: 6,
                canEvaluate: true
            ),
            series: series,
            config: config
        ))
        XCTAssertEqual(fullOutcome.labelClass, .buy)
    }

    private func makeSeries(highOverrides: [Int: Int64]) throws -> M1OHLCVSeries {
        let closeValues: [Int64] = [1_000, 1_000, 1_000, 1_000, 1_000, 1_000, 1_000, 1_000, 1_000]
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        for index in 0..<closeValues.count {
            let price = closeValues[index]
            utc.append(1_704_067_200 + Int64(index * 60))
            open.append(price)
            close.append(price)
            high.append(highOverrides[index] ?? (price + 1))
            low.append(price - 1)
            volume.append(UInt64(100 + index))
        }
        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "test",
                sourceOrigin: "TEST",
                logicalSymbol: "EURUSD",
                providerSymbol: "EURUSD",
                digits: 5,
                firstUTC: utc.first,
                lastUTC: utc.last
            ),
            utcTimestamps: utc,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }
}
