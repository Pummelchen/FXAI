import Foundation

public enum HorizonTools {
    public static let defaultConfiguredHorizons = [1, 2, 3, 5, 8, 13, 21, 34]

    public static func value(in values: [Double], index: Int, default defaultValue: Double) -> Double {
        guard index >= 0, index < values.count else { return defaultValue }
        return values[index]
    }

    public static func positiveIntMean(
        _ values: [Int],
        startIndex: Int,
        width: Int,
        fallback: Double
    ) -> Double {
        guard !values.isEmpty, startIndex >= 0, startIndex < values.count, width > 0 else {
            return max(fallback, 0.10)
        }

        let end = min(values.count, startIndex + width)
        var sum = 0.0
        var used = 0
        for index in startIndex..<end {
            let value = Double(values[index])
            if value <= 0.0 { continue }
            sum += value
            used += 1
        }
        guard used > 0 else { return max(fallback, 0.10) }
        return sum / Double(used)
    }

    public static func clampHorizon(_ horizonMinutes: Int) -> Int {
        min(max(horizonMinutes, 1), 720)
    }

    public static func parseHorizonList(
        _ raw: String,
        baseHorizon: Int,
        maxHorizons: Int = RuntimeArtifactConstants.maxHorizons
    ) -> [Int] {
        let horizonLimit = max(1, maxHorizons)
        let normalized = raw
            .replacingOccurrences(of: "{", with: "")
            .replacingOccurrences(of: "}", with: "")
            .replacingOccurrences(of: ";", with: ",")
            .replacingOccurrences(of: "|", with: ",")

        var horizons: [Int] = []
        for part in normalized.split(separator: ",", omittingEmptySubsequences: false) {
            let token = part.trimmingCharacters(in: .whitespacesAndNewlines)
            if token.isEmpty { continue }

            let horizon = clampHorizon(mqlStringToInteger(token))
            if horizons.contains(horizon) { continue }
            horizons.append(horizon)
            if horizons.count >= horizonLimit {
                break
            }
        }

        let base = clampHorizon(baseHorizon)
        if !horizons.contains(base) {
            horizons.append(base)
        }

        if horizons.isEmpty {
            horizons.append(base)
        }

        return horizons.sorted()
    }

    public static func maxConfiguredHorizon(
        configuredHorizons: [Int],
        fallbackHorizon: Int
    ) -> Int {
        var maximum = clampHorizon(fallbackHorizon)
        for horizon in configuredHorizons {
            maximum = max(maximum, clampHorizon(horizon))
        }
        return maximum
    }

    public static func horizonSlot(
        horizonMinutes: Int,
        configuredHorizons: [Int] = defaultConfiguredHorizons,
        maxHorizons: Int = RuntimeArtifactConstants.maxHorizons
    ) -> Int {
        let horizonLimit = max(1, maxHorizons)
        let horizons = Array(configuredHorizons.prefix(horizonLimit)).map(clampHorizon)
        guard !horizons.isEmpty else { return 0 }

        let horizon = clampHorizon(horizonMinutes)
        var best = 0
        var bestDiff = abs(horizons[0] - horizon)
        for index in horizons.indices.dropFirst() {
            let diff = abs(horizons[index] - horizon)
            if diff < bestDiff {
                bestDiff = diff
                best = index
            }
        }
        return min(max(best, 0), horizonLimit - 1)
    }

    public static func noSpreadStaticRegimeID(
        timestampUTC: Int64,
        volatilityProxyAbs: Double,
        volatilityRef: Double,
        regimeCount: Int = FXDataEngineConstants.pluginRegimeBuckets
    ) -> Int {
        let session = sessionGroup(timestampUTC: timestampUTC)
        let volRef = max(abs(volatilityRef), 1e-6)
        let volHigh = abs(volatilityProxyAbs) > (1.15 * volRef + 0.02)
        let regime = session * 4 + (volHigh ? 2 : 0)
        return Int(fxClamp(Double(regime), 0.0, Double(max(regimeCount - 1, 0))))
    }

    public static func sessionGroup(timestampUTC: Int64) -> Int {
        let timestamp = timestampUTC > 0 ? timestampUTC : 0
        let date = Date(timeIntervalSince1970: TimeInterval(timestamp))
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let hour = calendar.component(.hour, from: date)
        if hour < 8 { return 0 }
        if hour < 16 { return 1 }
        return 2
    }

    private static func mqlStringToInteger(_ token: String) -> Int {
        let scanner = Scanner(string: token)
        scanner.charactersToBeSkipped = nil
        return scanner.scanInt() ?? 0
    }
}
