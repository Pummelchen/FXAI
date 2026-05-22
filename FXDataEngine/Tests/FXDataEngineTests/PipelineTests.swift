import XCTest
@testable import FXDataEngine

final class PipelineTests: XCTestCase {
    func testMarketSeriesCarriesVolumeAndValidatesOHLCV() throws {
        let series = try makeSeries(volumeEnabled: true, count: 20)
        XCTAssertEqual(series.count, 20)
        XCTAssertEqual(series.volume[0], 100)
        XCTAssertTrue(series.hasVolume)
        XCTAssertEqual(series.bar(at: 3).volume, 103)
    }

    func testFeatureCoreZerosVolumeFeaturesWhenDatasetHasNoVolume() throws {
        let series = try makeSeries(volumeEnabled: false, count: 160)
        let universe = try MarketUniverse(series: [series])
        let bundle = try DataCore().buildBundle(
            request: DataCoreRequest(symbol: "EURUSD", neededBars: 120),
            universe: universe
        )
        let frame = try FeatureCore().buildFrame(bundle: bundle)

        XCTAssertFalse(frame.hasVolume)
        XCTAssertEqual(frame.raw[6], 0.0)
        XCTAssertEqual(frame.raw[68], 0.0)
        XCTAssertEqual(frame.raw[75], 0.0)
        XCTAssertEqual(frame.raw[80], 0.0)
    }

    func testFeatureCoreUsesVolumeWhenDatasetProvidesIt() throws {
        let series = try makeSeries(volumeEnabled: true, count: 160)
        let universe = try MarketUniverse(series: [series])
        let bundle = try DataCore().buildBundle(
            request: DataCoreRequest(symbol: "EURUSD", neededBars: 120),
            universe: universe
        )
        let frame = try FeatureCore().buildFrame(bundle: bundle)

        XCTAssertTrue(frame.hasVolume)
        XCTAssertNotEqual(frame.raw[6], 0.0)
        XCTAssertEqual(frame.raw[75], 1.0)
        XCTAssertGreaterThan(frame.raw[80], 0.0)
    }

    func testPipelineBuildsPluginPredictPayloadWithVolumeContext() throws {
        let series = try makeSeries(volumeEnabled: true, count: 220)
        let universe = try MarketUniverse(series: [series])
        let manifest = PluginManifestV4(
            aiID: 1,
            aiName: "VolumeAwareLinear",
            family: .linear,
            capabilityMask: [.selfTest, .windowContext],
            minSequenceBars: 2,
            maxSequenceBars: 16
        )
        let payload = try FXDataEnginePipeline().preparePredictPayload(
            universe: universe,
            request: DataCoreRequest(symbol: "EURUSD", neededBars: 120),
            manifest: manifest,
            horizonMinutes: 2
        )
        let request = payload.predictRequest

        XCTAssertTrue(request.context.dataHasVolume)
        XCTAssertEqual(request.x.count, FXDataEngineConstants.aiWeights)
        XCTAssertGreaterThan(request.windowSize, 0)
        XCTAssertLessThanOrEqual(request.windowSize, request.context.sequenceBars - 1)
        XCTAssertEqual(request.x[0], 1.0)
    }

    func testFeatureCoreUsesDataCoreContextAggregatesForContextSlots() throws {
        let primary = try makeTrendSeries(symbol: "EURUSD", drift: 2, count: 180)
        let risingContext = try makeTrendSeries(symbol: "USDJPY", drift: 5, count: 180)
        let fallingContext = try makeTrendSeries(symbol: "GBPUSD", drift: -3, count: 180)
        let universe = try MarketUniverse(
            primarySymbol: "EURUSD",
            series: [primary, risingContext, fallingContext]
        )
        let bundle = try DataCore().buildBundle(
            request: DataCoreRequest(
                symbol: "EURUSD",
                neededBars: 120,
                contextSymbols: ["USDJPY", "GBPUSD"]
            ),
            universe: universe
        )

        let frame = try FeatureCore().buildFrame(bundle: bundle)
        let index = bundle.sampleIndex
        let aggregates = bundle.contextAggregates
        let volatilityUnit = max(rawRollingAbsReturn(primary, index: index, window: 20), 1e-6)
        let previousVolatilityUnit = max(rawRollingAbsReturn(primary, index: index - 1, window: 20), 1e-6)

        XCTAssertEqual(frame.raw[10], aggregates.mean[index] / volatilityUnit, accuracy: 1e-12)
        XCTAssertEqual(frame.raw[11], aggregates.standardDeviation[index] / volatilityUnit, accuracy: 1e-12)
        XCTAssertEqual(frame.raw[12], fxClampSignedUnit((aggregates.upRatio[index] - 0.5) * 2.0), accuracy: 1e-12)
        XCTAssertEqual(
            frame.raw[50],
            aggregates.extraValue(sampleIndex: index, featureIndex: 0) / volatilityUnit,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            frame.raw[51],
            aggregates.extraValue(sampleIndex: index, featureIndex: 1) / volatilityUnit,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            frame.raw[52],
            aggregates.extraValue(sampleIndex: index, featureIndex: 2) / volatilityUnit,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            frame.raw[53],
            fxClampSignedUnit(aggregates.extraValue(sampleIndex: index, featureIndex: 3)),
            accuracy: 1e-12
        )

        let sharedOffset = FXDataEngineConstants.contextSharedOffset
        XCTAssertEqual(
            frame.raw[62],
            fxClampSignedUnit(aggregates.extraValue(sampleIndex: index, featureIndex: sharedOffset)),
            accuracy: 1e-12
        )
        XCTAssertEqual(
            frame.raw[63],
            fxClampSignedUnit(
                (aggregates.extraValue(sampleIndex: index, featureIndex: sharedOffset + 1, default: 0.5) * 2.0) - 1.0
            ),
            accuracy: 1e-12
        )
        XCTAssertEqual(
            frame.raw[64],
            fxClampSignedUnit(
                (aggregates.extraValue(sampleIndex: index, featureIndex: sharedOffset + 2, default: 0.5) * 2.0) - 1.0
            ),
            accuracy: 1e-12
        )
        XCTAssertEqual(
            frame.raw[65],
            fxClampSignedUnit(
                (aggregates.extraValue(sampleIndex: index, featureIndex: sharedOffset + 3) * 2.0) - 1.0
            ),
            accuracy: 1e-12
        )
        XCTAssertEqual(frame.previous[10], aggregates.mean[index - 1] / previousVolatilityUnit, accuracy: 1e-12)
    }

    func testTripleBarrierLabelUsesAscendingFutureBars() throws {
        let series = try makeBreakoutSeries(symbol: "EURUSD", count: 90, breakoutIndex: 52)
        let label = TrainingSampleTools.buildTripleBarrierLabel(
            series: series,
            index: 50,
            horizonMinutes: 5,
            roundTripCostPoints: 1.0,
            evThresholdPoints: 1.0
        )

        XCTAssertEqual(label.labelClass, .buy)
        XCTAssertGreaterThan(label.realizedMovePoints, 0.0)
        XCTAssertEqual(label.timeToHitFraction, 0.4, accuracy: 1e-12)
        XCTAssertGreaterThan(label.mfePoints, label.maePoints)
    }

    func testPipelineBuildsTrainPayloadWithTripleBarrierLabel() throws {
        let series = try makeBreakoutSeries(symbol: "EURUSD", count: 110, breakoutIndex: 62)
        let universe = try MarketUniverse(primarySymbol: "EURUSD", series: [series])
        let manifest = PluginManifestV4(
            aiID: 7,
            aiName: "TrainableWindow",
            family: .linear,
            capabilityMask: [.selfTest, .windowContext],
            minSequenceBars: 2,
            maxSequenceBars: 16
        )
        let payload = try FXDataEnginePipeline().prepareTrainPayload(
            universe: universe,
            request: DataCoreRequest(symbol: "EURUSD", neededBars: 50, alignUpToIndex: 60),
            manifest: manifest,
            horizonMinutes: 5,
            roundTripCostPoints: 1.0,
            evThresholdPoints: 1.0
        )
        let request = payload.trainRequest

        try request.validate()
        XCTAssertEqual(payload.sample.labelClass, .buy)
        XCTAssertEqual(payload.sample.horizonSlot, 3)
        XCTAssertEqual(request.labelClass, .buy)
        XCTAssertEqual(request.context.regimeID, payload.sample.regimeID)
        XCTAssertEqual(request.context.dataHasVolume, true)
        XCTAssertEqual(request.x.count, FXDataEngineConstants.aiWeights)
        XCTAssertGreaterThan(request.windowSize, 0)
        XCTAssertGreaterThan(request.movePoints, 0.0)
        XCTAssertGreaterThan(payload.sample.qualityScore, 0.35)
        XCTAssertEqual(request.fillRisk, 0.0)
    }

    func testPipelineBuildsTrainingDatasetAcrossIndexRange() throws {
        let series = try makeBreakoutSeries(symbol: "EURUSD", count: 130, breakoutIndex: 80)
        let universe = try MarketUniverse(primarySymbol: "EURUSD", series: [series])
        let manifest = PluginManifestV4(
            aiID: 8,
            aiName: "DatasetWindow",
            family: .linear,
            capabilityMask: [.selfTest, .windowContext],
            minSequenceBars: 2,
            maxSequenceBars: 16
        )

        let dataset = try FXDataEnginePipeline().prepareTrainingDataset(
            universe: universe,
            baseRequest: DataCoreRequest(symbol: "EURUSD", neededBars: 50),
            manifest: manifest,
            datasetRequest: TrainingDatasetRequest(
                startIndex: 60,
                endIndex: 72,
                stride: 3,
                maxSamples: 4,
                horizonMinutes: 5,
                roundTripCostPoints: 1.0,
                evThresholdPoints: 1.0
            )
        )

        XCTAssertEqual(dataset.symbol, "EURUSD")
        XCTAssertEqual(dataset.payloads.map(\.dataBundle.sampleIndex), [60, 63, 66, 69])
        XCTAssertEqual(dataset.trainRequests.count, 4)
        for request in dataset.trainRequests {
            try request.validate()
            XCTAssertEqual(request.context.horizonMinutes, 5)
            XCTAssertGreaterThan(request.windowSize, 0)
        }
    }

    private func makeSeries(volumeEnabled: Bool, count: Int) throws -> M1OHLCVSeries {
        let start = Int64(1_704_067_200)
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        var price = Int64(108_000)

        for index in 0..<count {
            let next = price + Int64((index % 9) - 3)
            utc.append(start + Int64(index * 60))
            open.append(price)
            high.append(max(price, next) + 5)
            low.append(min(price, next) - 5)
            close.append(next)
            volume.append(volumeEnabled ? UInt64(100 + (index % 17)) : 0)
            price = next
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "demo",
                sourceOrigin: "DEMO",
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

    private func makeTrendSeries(symbol: String, drift: Int64, count: Int) throws -> M1OHLCVSeries {
        let start = Int64(1_704_067_200)
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        var price = Int64(108_000)

        for index in 0..<count {
            let wave = Int64((index % 11) - 5)
            let next = max(90_000, price + drift + wave)
            utc.append(start + Int64(index * 60))
            open.append(price)
            high.append(max(price, next) + 5)
            low.append(min(price, next) - 5)
            close.append(next)
            volume.append(UInt64(100 + (index % 19)))
            price = next
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "demo",
                sourceOrigin: "DEMO",
                logicalSymbol: symbol,
                providerSymbol: symbol,
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

    private func makeBreakoutSeries(symbol: String, count: Int, breakoutIndex: Int) throws -> M1OHLCVSeries {
        let start = Int64(1_704_067_200)
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        var price = Int64(100_000)

        for index in 0..<count {
            let next: Int64
            if index < breakoutIndex - 1 {
                next = price
            } else if index == breakoutIndex - 1 {
                next = price + 1
            } else {
                next = 100_020 + Int64(index - breakoutIndex)
            }
            utc.append(start + Int64(index * 60))
            open.append(price)
            high.append(max(price, next) + 2)
            low.append(min(price, next) - 2)
            close.append(next)
            volume.append(UInt64(200 + (index % 23)))
            price = next
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "demo",
                sourceOrigin: "DEMO",
                logicalSymbol: symbol,
                providerSymbol: symbol,
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

    private func rawRollingAbsReturn(_ series: M1OHLCVSeries, index: Int, window: Int) -> Double {
        guard window >= 2, index > 0, index < series.count else { return 0.0 }
        let start = max(1, index - window + 1)
        var sum = 0.0
        var count = 0
        for row in start...index {
            sum += abs(rawReturn(series, index: row, lookback: 1))
            count += 1
        }
        return count > 0 ? sum / Double(count) : 0.0
    }

    private func rawReturn(_ series: M1OHLCVSeries, index: Int, lookback: Int) -> Double {
        let prior = index - lookback
        guard prior >= 0, index >= 0, index < series.count else { return 0.0 }
        let old = Double(series.close[prior])
        guard old > 0 else { return 0.0 }
        return (Double(series.close[index]) - old) / old
    }
}
