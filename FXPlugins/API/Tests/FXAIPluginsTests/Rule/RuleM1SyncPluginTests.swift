import FXAIPlugins
import FXDataEngine
import XCTest

final class RuleM1SyncPluginTests: XCTestCase {
    private static let baseTimeUTC: Int64 = 1_704_153_600

    func testManifestMatchesLegacyM1SyncContract() throws {
        let plugin = RuleM1SyncPlugin()

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.m1Sync.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "rule_m1sync")
        XCTAssertEqual(plugin.manifest.family, .ruleBased)
        XCTAssertEqual(plugin.manifest.referenceTier, .ruleBaseline)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.multiHorizon))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.selfTest))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
    }

    func testPredictFailsClosedWithoutSyntheticSeries() throws {
        let plugin = RuleM1SyncPlugin()
        let prediction = try plugin.predict(Self.request(sampleTimeUTC: Self.baseTimeUTC + 180), hyperParameters: HyperParameters())

        XCTAssertEqual(prediction.classProbabilities[LabelClass.sell.rawValue], 0.02, accuracy: 1e-12)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.buy.rawValue], 0.02, accuracy: 1e-12)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.skip.rawValue], 0.96, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveMeanPoints, 0.0, accuracy: 1e-12)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testPredictDetectsUpChainFromSyntheticM1OHLCV() throws {
        var plugin = RuleM1SyncPlugin(m1SyncBars: 3)
        try plugin.setSyntheticSeries(Self.series(closes: [100, 101, 102, 103, 104], nextOpenAdjustment: 0))

        let prediction = try plugin.predict(Self.request(sampleTimeUTC: Self.baseTimeUTC + 180), hyperParameters: HyperParameters())

        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.buy.rawValue], 0.90)
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.sell.rawValue], 0.009)
        XCTAssertLessThan(prediction.classProbabilities[LabelClass.sell.rawValue], 0.011)
        XCTAssertLessThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.10)
        XCTAssertGreaterThan(prediction.moveMeanPoints, 2.0)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testPredictDetectsDownChainFromSyntheticM1OHLCV() throws {
        var plugin = RuleM1SyncPlugin(m1SyncBars: 3)
        try plugin.setSyntheticSeries(Self.series(closes: [104, 103, 102, 101, 100], nextOpenAdjustment: 0))

        let prediction = try plugin.predict(Self.request(sampleTimeUTC: Self.baseTimeUTC + 180), hyperParameters: HyperParameters())

        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.sell.rawValue], 0.90)
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.buy.rawValue], 0.009)
        XCTAssertLessThan(prediction.classProbabilities[LabelClass.buy.rawValue], 0.011)
        XCTAssertLessThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.10)
        XCTAssertGreaterThan(prediction.moveMeanPoints, 2.0)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testPredictSkipsBrokenChain() throws {
        var plugin = RuleM1SyncPlugin(m1SyncBars: 3)
        try plugin.setSyntheticSeries(Self.series(closes: [100, 102, 101, 103, 104], nextOpenAdjustment: 0))

        let prediction = try plugin.predict(Self.request(sampleTimeUTC: Self.baseTimeUTC + 180), hyperParameters: HyperParameters())

        XCTAssertEqual(prediction.classProbabilities[LabelClass.skip.rawValue], 0.96, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveMeanPoints, 0.0, accuracy: 1e-12)
    }

    func testTrainUpdatesReliabilityState() throws {
        var plugin = RuleM1SyncPlugin(m1SyncBars: 3)
        try plugin.setSyntheticSeries(Self.series(closes: [100, 101, 102, 103, 104], nextOpenAdjustment: 0))
        let before = try plugin.predict(Self.request(sampleTimeUTC: Self.baseTimeUTC + 180), hyperParameters: HyperParameters())
        let train = TrainRequestV4(
            valid: true,
            context: Self.context(sampleTimeUTC: Self.baseTimeUTC + 180),
            labelClass: .buy,
            movePoints: 3.0,
            sampleWeight: 1.0,
            nextVolumeTarget: 1.0,
            x: Self.features
        )

        try plugin.train(train, hyperParameters: HyperParameters())
        let after = try plugin.predict(Self.request(sampleTimeUTC: Self.baseTimeUTC + 180), hyperParameters: HyperParameters())

        XCTAssertGreaterThan(after.reliability, before.reliability)
    }

    func testRegistryExposesM1Sync() {
        let pluginNames = FXAIPluginRegistry.availablePlugins().map(\.manifest.aiName)
        let plan = FXAIPluginRegistry.accelerationPlans().first { $0.pluginName == "rule_m1sync" }

        XCTAssertTrue(pluginNames.contains("rule_m1sync"))
        XCTAssertEqual(plan?.primaryBackends, [.swiftScalar, .metal])
        XCTAssertEqual(plan?.candidateBackends, [.swiftSIMD])
        XCTAssertTrue(plan?.usesVolumeWhenAvailable ?? false)
    }

    func testConvertedPluginOwnsMetalKernel() throws {
        let root = Self.pluginRoot()

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("RuleM1SyncPlugin.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal/RuleM1SyncMetal.swift").path))
    }

    func testPredictUsesVolumeConfirmationWhenAvailable() throws {
        var volumePlugin = RuleM1SyncPlugin(m1SyncBars: 3)
        var noVolumePlugin = RuleM1SyncPlugin(m1SyncBars: 3)
        let volumeSeries = try Self.series(
            closes: [100, 101, 102, 103, 104],
            nextOpenAdjustment: 0,
            volumes: [10, 10, 10, 100, 100]
        )
        let noVolumeSeries = try Self.series(
            closes: [100, 101, 102, 103, 104],
            nextOpenAdjustment: 0,
            volumes: [0, 0, 0, 0, 0]
        )
        XCTAssertTrue(volumeSeries.hasVolume)
        XCTAssertFalse(noVolumeSeries.hasVolume)
        try volumePlugin.setSyntheticSeries(volumeSeries)
        try noVolumePlugin.setSyntheticSeries(noVolumeSeries)

        let request = Self.request(sampleTimeUTC: Self.baseTimeUTC + 180, dataHasVolume: true)
        let withVolume = try volumePlugin.predict(request, hyperParameters: HyperParameters())
        let withoutVolume = try noVolumePlugin.predict(request, hyperParameters: HyperParameters())

        XCTAssertGreaterThan(withVolume.reliability, withoutVolume.reliability)
        XCTAssertGreaterThan(withVolume.classProbabilities[LabelClass.buy.rawValue], withoutVolume.classProbabilities[LabelClass.buy.rawValue])
    }

    private static var features: [Double] {
        Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
    }

    private static func request(sampleTimeUTC: Int64, dataHasVolume: Bool = true) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(sampleTimeUTC: sampleTimeUTC, dataHasVolume: dataHasVolume),
            x: features
        )
    }

    private static func context(sampleTimeUTC: Int64, dataHasVolume: Bool = true) -> PluginContextV4 {
        PluginContextV4(
            sessionBucket: 2,
            horizonMinutes: 5,
            sequenceBars: 1,
            priceCostPoints: 0.20,
            minMovePoints: 0.50,
            pointValue: 1.0,
            sampleTimeUTC: sampleTimeUTC,
            dataHasVolume: dataHasVolume
        )
    }

    private static func series(
        closes: [Int64],
        nextOpenAdjustment: Int64,
        volumes: [UInt64]? = nil
    ) throws -> M1OHLCVSeries {
        var timestamps = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()

        for index in closes.indices {
            let closeValue = closes[index]
            let openValue = index == closes.count - 1 ? closeValue + nextOpenAdjustment : closeValue
            timestamps.append(baseTimeUTC + Int64(index * 60))
            open.append(openValue)
            high.append(max(openValue, closeValue) + 1)
            low.append(min(openValue, closeValue) - 1)
            close.append(closeValue)
            volume.append(volumes?[index] ?? UInt64(100 + index))
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "test",
                sourceOrigin: "TEST",
                logicalSymbol: "EURUSD",
                timeframe: .m1,
                digits: 5,
                firstUTC: timestamps.first,
                lastUTC: timestamps.last
            ),
            utcTimestamps: timestamps,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }

    private static func pluginRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("rule_m1sync")
    }
}
