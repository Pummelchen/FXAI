import XCTest
@testable import FXDataEngine

final class FeatureBuildToolsTests: XCTestCase {
    func testCyclicSessionPulsesMatchLegacyBoundsAndWraparound() {
        XCTAssertEqual(
            FeatureBuildTools.cyclicHourPulse(hourValue: 23.0, centerHour: 23.0, radiusHours: 3.0),
            FXDataEngineConstants.unitRangeCeil,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            FeatureBuildTools.cyclicHourPulse(hourValue: 20.0, centerHour: 23.0, radiusHours: 3.0),
            FXDataEngineConstants.unitRangeFloor,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            FeatureBuildTools.cyclicHourPulse(hourValue: 1.0, centerHour: 23.0, radiusHours: 3.0),
            1.0 / 3.0,
            accuracy: 1e-12
        )
    }

    func testSessionAndCarryFeaturesUsePreparedOfflineInputs() {
        let transitionTime = utc(2024, 1, 3, 13, 30)
        let pulseFloor = FXDataEngineConstants.unitRangeFloor
        XCTAssertEqual(FeatureBuildTools.mqlWeekday(sampleTimeUTC: transitionTime), 3)
        XCTAssertEqual(
            FeatureBuildTools.sessionTransition(sampleTimeUTC: transitionTime),
            (0.60 * pulseFloor) + (0.80 * 0.80) - (0.70 * pulseFloor),
            accuracy: 1e-12
        )
        XCTAssertEqual(
            FeatureBuildTools.sessionOverlap(sampleTimeUTC: transitionTime),
            (0.70 * pulseFloor) + (0.30 * 0.25),
            accuracy: 1e-12
        )

        let rolloverTime = utc(2024, 1, 3, 23, 0)
        let carry = FeatureBuildTools.carryFeatures(CarryFeatureInputs(
            sampleTimeUTC: rolloverTime,
            tripleRolloverWeekday: 3,
            swapLong: 2.0,
            swapShort: -1.0,
            momentumSignal: 1.0,
            contextSignal: 0.5
        ))
        XCTAssertEqual(carry.tripleSwapBias, FXDataEngineConstants.signedUnitRangeCeil, accuracy: 1e-12)
        XCTAssertEqual(carry.swapLongPressure, 1.0, accuracy: 1e-12)
        XCTAssertEqual(carry.swapShortPressure, -0.5, accuracy: 1e-12)
        XCTAssertEqual(carry.carryAlignment, 1.275, accuracy: 1e-12)
    }

    func testLocalFeatureFamilyDriftUsesVolumeGroupReplacementForLegacyCostFamily() {
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        let registry = FeatureRegistry()
        for index in features.indices {
            switch registry.group(for: index) {
            case .price:
                features[index] = 0.9
            case .multiTimeframe:
                features[index] = -0.3
            case .context:
                features[index] = 0.6
            case .volume:
                features[index] = -0.6
            case .microstructure:
                features[index] = 0.4
            case .filters:
                features[index] = -0.2
            default:
                features[index] = 0.0
            }
        }
        features[79] = 100.0

        let expected = (abs(0.9 - (-0.3)) + abs(0.6 - (-0.6)) + 0.5 * abs(0.4 - (-0.2))) / 3.0
        XCTAssertEqual(FeatureBuildTools.localFeatureFamilyDrift(features: features), expected, accuracy: 1e-12)
    }

    func testFeatureCoreUsesLegacySignedTimeAndSessionFeatures() throws {
        let sampleTime = utc(2024, 1, 3, 13, 30)
        let series = try makeSeriesEnding(at: sampleTime, count: 140)
        let universe = try MarketUniverse(series: [series])
        let bundle = try DataCore().buildBundle(
            request: DataCoreRequest(symbol: "EURUSD", neededBars: 120),
            universe: universe
        )
        let frame = try FeatureCore().buildFrame(bundle: bundle)

        XCTAssertEqual(frame.raw[15], 0.0, accuracy: 1e-12)
        XCTAssertEqual(frame.raw[16], (13.0 - 11.5) / 11.5, accuracy: 1e-12)
        XCTAssertEqual(frame.raw[17], (30.0 - 29.5) / 29.5, accuracy: 1e-12)
        XCTAssertEqual(frame.raw[72], FeatureBuildTools.sessionTransition(sampleTimeUTC: sampleTime), accuracy: 1e-12)
        XCTAssertEqual(frame.raw[73], FeatureBuildTools.sessionOverlap(sampleTimeUTC: sampleTime), accuracy: 1e-12)
    }

    private func utc(_ year: Int, _ month: Int, _ day: Int, _ hour: Int, _ minute: Int) -> Int64 {
        var components = DateComponents()
        components.calendar = Calendar(identifier: .gregorian)
        components.timeZone = TimeZone(secondsFromGMT: 0)
        components.year = year
        components.month = month
        components.day = day
        components.hour = hour
        components.minute = minute
        components.second = 0
        return Int64(components.date!.timeIntervalSince1970)
    }

    private func makeSeriesEnding(at sampleTimeUTC: Int64, count: Int) throws -> M1OHLCVSeries {
        var utcValues = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        var price = Int64(100_000)
        let start = sampleTimeUTC - Int64(max(count - 1, 0) * 60)

        for index in 0..<count {
            let next = price + Int64((index % 7) - 2)
            utcValues.append(start + Int64(index * 60))
            open.append(price)
            high.append(max(price, next) + 3)
            low.append(min(price, next) - 3)
            close.append(next)
            volume.append(UInt64(100 + (index % 11)))
            price = next
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "test",
                sourceOrigin: "TEST",
                logicalSymbol: "EURUSD",
                providerSymbol: "EURUSD",
                digits: 5,
                firstUTC: utcValues.first,
                lastUTC: utcValues.last
            ),
            utcTimestamps: utcValues,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }
}
