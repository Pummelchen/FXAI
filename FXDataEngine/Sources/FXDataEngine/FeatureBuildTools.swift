import Foundation

public struct CarryFeatureInputs: Codable, Hashable, Sendable {
    public var sampleTimeUTC: Int64
    public var tripleRolloverWeekday: Int
    public var swapLong: Double
    public var swapShort: Double
    public var momentumSignal: Double
    public var contextSignal: Double

    public init(
        sampleTimeUTC: Int64,
        tripleRolloverWeekday: Int = 3,
        swapLong: Double = 0.0,
        swapShort: Double = 0.0,
        momentumSignal: Double = 0.0,
        contextSignal: Double = 0.0
    ) {
        self.sampleTimeUTC = sampleTimeUTC
        self.tripleRolloverWeekday = min(max(tripleRolloverWeekday, 0), 6)
        self.swapLong = fxSafeFinite(swapLong)
        self.swapShort = fxSafeFinite(swapShort)
        self.momentumSignal = fxSafeFinite(momentumSignal)
        self.contextSignal = fxSafeFinite(contextSignal)
    }
}

public struct CarryFeatures: Codable, Hashable, Sendable {
    public var tripleSwapBias: Double
    public var swapLongPressure: Double
    public var swapShortPressure: Double
    public var carryAlignment: Double

    public init(
        tripleSwapBias: Double = 0.0,
        swapLongPressure: Double = 0.0,
        swapShortPressure: Double = 0.0,
        carryAlignment: Double = 0.0
    ) {
        self.tripleSwapBias = fxClampSignedUnit(tripleSwapBias)
        self.swapLongPressure = fxClamp(fxSafeFinite(swapLongPressure), -4.0, 4.0)
        self.swapShortPressure = fxClamp(fxSafeFinite(swapShortPressure), -4.0, 4.0)
        self.carryAlignment = fxClamp(fxSafeFinite(carryAlignment), -6.0, 6.0)
    }
}

public enum FeatureBuildTools {
    public static func cyclicHourPulse(hourValue: Double, centerHour: Double, radiusHours: Double) -> Double {
        let radius = radiusHours > 1e-6 ? radiusHours : 1.0
        var distance = abs(hourValue - centerHour)
        if distance > 12.0 {
            distance = 24.0 - distance
        }
        return fxClampUnit(1.0 - distance / radius)
    }

    public static func rolloverProximity(sampleTimeUTC: Int64) -> Double {
        cyclicHourPulse(hourValue: hourValue(sampleTimeUTC: sampleTimeUTC), centerHour: 23.0, radiusHours: 3.0)
    }

    public static func sessionTransition(sampleTimeUTC: Int64) -> Double {
        let hour = hourValue(sampleTimeUTC: sampleTimeUTC)
        let asiaToEurope = cyclicHourPulse(hourValue: hour, centerHour: 7.0, radiusHours: 2.5)
        let europeToUS = cyclicHourPulse(hourValue: hour, centerHour: 13.0, radiusHours: 2.5)
        let usToRollover = cyclicHourPulse(hourValue: hour, centerHour: 21.0, radiusHours: 2.5)
        return fxClampSignedUnit(0.60 * asiaToEurope + 0.80 * europeToUS - 0.70 * usToRollover)
    }

    public static func sessionOverlap(sampleTimeUTC: Int64) -> Double {
        let hour = hourValue(sampleTimeUTC: sampleTimeUTC)
        let londonOpen = cyclicHourPulse(hourValue: hour, centerHour: 8.0, radiusHours: 3.0)
        let newYorkOpen = cyclicHourPulse(hourValue: hour, centerHour: 13.5, radiusHours: 3.0)
        let overlap = min(londonOpen, newYorkOpen)
        return fxClampUnit(0.70 * overlap + 0.30 * cyclicHourPulse(hourValue: hour, centerHour: 15.0, radiusHours: 2.0))
    }

    public static func carryFeatures(_ inputs: CarryFeatureInputs) -> CarryFeatures {
        let rolloverWeekday = min(max(inputs.tripleRolloverWeekday, 0), 6)
        let currentWeekday = mqlWeekday(sampleTimeUTC: inputs.sampleTimeUTC)
        let dayBias = currentWeekday == rolloverWeekday ? 1.0 : -0.25
        let rollBias = rolloverProximity(sampleTimeUTC: inputs.sampleTimeUTC)
        let tripleSwapBias = fxClampSignedUnit(dayBias * (0.35 + 0.65 * rollBias))

        let scale = max(max(abs(inputs.swapLong), abs(inputs.swapShort)), 0.25)
        let longPressure = fxClamp(inputs.swapLong / scale, -4.0, 4.0)
        let shortPressure = fxClamp(inputs.swapShort / scale, -4.0, 4.0)
        let carrySkew = (inputs.swapLong - inputs.swapShort) / scale
        let directional = fxClamp(0.70 * inputs.momentumSignal + 0.30 * inputs.contextSignal, -4.0, 4.0)
        let alignment = fxClamp(carrySkew * directional, -6.0, 6.0)

        return CarryFeatures(
            tripleSwapBias: tripleSwapBias,
            swapLongPressure: longPressure,
            swapShortPressure: shortPressure,
            carryAlignment: alignment
        )
    }

    public static func localFeatureFamilyDrift(
        features: [Double],
        registry: FeatureRegistry = FeatureRegistry()
    ) -> Double {
        var familySums = Array(repeating: 0.0, count: FeatureGroup.allCases.count)
        var familyCounts = Array(repeating: 0, count: FeatureGroup.allCases.count)
        for featureIndex in 0..<min(features.count, FXDataEngineConstants.aiFeatures) where featureIndex != 79 {
            let group = registry.group(for: featureIndex)
            let groupIndex = group.rawValue
            guard groupIndex >= 0, groupIndex < familySums.count else { continue }
            familySums[groupIndex] += fxSafeFinite(features[featureIndex])
            familyCounts[groupIndex] += 1
        }

        func mean(_ group: FeatureGroup) -> Double {
            let index = group.rawValue
            guard index >= 0, index < familySums.count, familyCounts[index] > 0 else { return 0.0 }
            return familySums[index] / Double(familyCounts[index])
        }

        var drift = 0.0
        drift += abs(mean(.price) - mean(.multiTimeframe))
        drift += abs(mean(.context) - mean(.volume))
        drift += 0.50 * abs(mean(.microstructure) - mean(.filters))
        return fxClamp(drift / 3.0, 0.0, 6.0)
    }

    public static func hourValue(sampleTimeUTC: Int64) -> Double {
        let components = utcComponents(sampleTimeUTC: sampleTimeUTC)
        return Double(components.hour ?? 0) + Double(components.minute ?? 0) / 60.0
    }

    public static func mqlWeekday(sampleTimeUTC: Int64) -> Int {
        let weekday = utcComponents(sampleTimeUTC: sampleTimeUTC).weekday ?? 1
        return min(max(weekday - 1, 0), 6)
    }

    private static func utcComponents(sampleTimeUTC: Int64) -> DateComponents {
        let date = Date(timeIntervalSince1970: TimeInterval(max(sampleTimeUTC, 0)))
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        return calendar.dateComponents([.weekday, .hour, .minute], from: date)
    }
}
