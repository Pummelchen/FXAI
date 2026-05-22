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
}
