import Foundation

public enum NormalizationWindowTools {
    public static let defaultWindow = 192
    public static let minimumWindow = 16
    public static let maximumWindow = RuntimeArtifactConstants.normalizationRollWindowMax

    public static func clamp(_ window: Int) -> Int {
        min(max(window, minimumWindow), maximumWindow)
    }

    public static func defaultWindow(predictionTargetMinutes: Int) -> Int {
        var window = defaultWindow
        if predictionTargetMinutes <= 2 {
            window = 128
        } else if predictionTargetMinutes >= 30 {
            window = 256
        }
        return min(max(window, 32), maximumWindow)
    }

    public static func normalizedFeatureWindows(
        _ windows: [Int],
        defaultWindow: Int,
        featureCount: Int = FXDataEngineConstants.aiFeatures
    ) -> [Int] {
        let resolvedDefault = clamp(defaultWindow)
        var output = Array(repeating: resolvedDefault, count: featureCount)
        for index in 0..<min(windows.count, featureCount) {
            output[index] = clamp(windows[index])
        }
        return output
    }

    public static func buildGroupWindows(
        fast: Int,
        mid: Int,
        slow: Int,
        regime: Int,
        featureCount: Int = FXDataEngineConstants.aiFeatures
    ) -> [Int] {
        let fastWindow = clamp(fast)
        let midWindow = clamp(mid)
        let slowWindow = clamp(slow)
        let regimeWindow = clamp(regime)
        var output = Array(repeating: midWindow, count: featureCount)
        for feature in 0..<featureCount {
            let window: Int
            if feature <= 6 {
                window = fastWindow
            } else if feature <= 14 {
                window = midWindow
            } else if feature <= 21 {
                window = regimeWindow
            } else if feature <= 33 {
                window = slowWindow
            } else if feature <= 65 {
                window = midWindow
            } else if feature <= 71 {
                window = fastWindow
            } else if feature <= 75 {
                window = regimeWindow
            } else if feature <= 78 {
                window = slowWindow
            } else {
                window = midWindow
            }
            output[feature] = window
        }
        return output
    }
}

public enum NormalizationMetaSupportTools {
    public static let candidateMax = FXDataEngineConstants.normalizationCandidateMax

    public static func barRandom01(barTimeUTC: Int64, salt: Int) -> Double {
        let maskedTime = UInt64(bitPattern: barTimeUTC) & 0x7fff_ffff
        var value = UInt32(truncatingIfNeeded: maskedTime)
        let saltSeed = UInt32(truncatingIfNeeded: salt) &+ 1
        value ^= saltSeed &* 1_103_515_245 &+ 12_345
        value ^= value << 13
        value ^= value >> 17
        value ^= value << 5
        return Double(value % 100_000) / 100_000.0
    }

    public static func shouldSampleByPercent(barTimeUTC: Int64, salt: Int, percent: Double) -> Bool {
        let clampedPercent = fxClamp(percent, 0.0, 100.0)
        if clampedPercent <= 0.0 {
            return false
        }
        if clampedPercent >= 100.0 {
            return true
        }
        return barRandom01(barTimeUTC: barTimeUTC, salt: salt) < clampedPercent / 100.0
    }

    public static func isShadowBar(cadenceBars: Int, barSequence: Int) -> Bool {
        if cadenceBars <= 0 {
            return false
        }
        if cadenceBars == 1 {
            return true
        }
        if barSequence < 0 {
            return false
        }
        return barSequence % cadenceBars == 0
    }

    public static func sanitizeNormalizationMethod(_ methodID: Int) -> FeatureNormalizationMethod {
        FeatureNormalizationMethod(rawValue: methodID) ?? .existing
    }

    public static func normalizationMethodCandidates(
        aiID: Int,
        currentMethod: FeatureNormalizationMethod,
        maxCandidates: Int = candidateMax
    ) -> [FeatureNormalizationMethod] {
        let limit = min(max(0, maxCandidates), candidateMax)
        guard limit > 0 else { return [] }

        let deepModel = AIModelID(rawValue: aiID)?.usesDeepNormalizationCandidates ?? false
        let seedMethods: [FeatureNormalizationMethod]
        if deepModel {
            seedMethods = [
                currentMethod,
                .existing,
                .volatilityStdReturns,
                .atrNatrUnit,
                .zScore,
                .revin,
                .dain,
                .robustMedianIQR,
                .minMaxBuffer3
            ]
        } else {
            seedMethods = [
                currentMethod,
                .existing,
                .zScore,
                .robustMedianIQR,
                .quantileToNormal,
                .changePercent,
                .volatilityStdReturns,
                .atrNatrUnit,
                .powerYeoJohnson,
                .minMaxBuffer3
            ]
        }

        var methods: [FeatureNormalizationMethod] = []
        methods.reserveCapacity(min(seedMethods.count, limit))
        for method in seedMethods where !methods.contains(method) {
            methods.append(method)
            if methods.count >= limit {
                break
            }
        }
        return methods
    }
}

public extension FeatureNormalizationMethod {
    var usesAdaptivePayloadNormalization: Bool {
        self == .revin || self == .dain
    }

    var usesFittedStats: Bool {
        switch self {
        case .minMaxBuffer5, .minMaxBuffer2, .minMaxBuffer3,
             .zScore, .robustMedianIQR, .quantileToNormal,
             .powerYeoJohnson, .revin, .dain:
            true
        default:
            false
        }
    }

    var usesRollingNormalizationHistory: Bool {
        switch self {
        case .minMaxBuffer5, .minMaxBuffer2, .minMaxBuffer3,
             .zScore, .robustMedianIQR, .quantileToNormal:
            true
        default:
            false
        }
    }
}

public struct NormalizationLegacyWindowState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var defaultWindow: Int
    public var featureWindows: [Int]

    public init(
        ready: Bool = false,
        defaultWindow: Int = NormalizationWindowTools.defaultWindow,
        featureWindows: [Int] = []
    ) {
        self.ready = ready
        self.defaultWindow = NormalizationWindowTools.clamp(defaultWindow)
        self.featureWindows = NormalizationWindowTools.normalizedFeatureWindows(
            featureWindows,
            defaultWindow: self.defaultWindow
        )
    }

    public mutating func apply(featureWindows: [Int], defaultWindow: Int) {
        self.defaultWindow = NormalizationWindowTools.clamp(defaultWindow)
        self.featureWindows = NormalizationWindowTools.normalizedFeatureWindows(
            featureWindows,
            defaultWindow: self.defaultWindow
        )
        ready = true
    }
}

public struct NormalizationWindowConfigState: Codable, Hashable, Sendable {
    public var initialized: Bool
    public var defaultWindow: Int
    public var configVersion: Int
    public var featureWindows: [Int]

    public init(
        initialized: Bool = false,
        defaultWindow: Int = NormalizationWindowTools.defaultWindow,
        configVersion: Int = 0,
        featureWindows: [Int] = []
    ) {
        self.initialized = initialized
        self.defaultWindow = NormalizationWindowTools.clamp(defaultWindow)
        self.configVersion = max(0, configVersion)
        self.featureWindows = NormalizationWindowTools.normalizedFeatureWindows(
            featureWindows,
            defaultWindow: self.defaultWindow
        )
    }

    public mutating func reset(defaultWindow: Int = NormalizationWindowTools.defaultWindow) {
        self.defaultWindow = NormalizationWindowTools.clamp(defaultWindow)
        featureWindows = Array(repeating: self.defaultWindow, count: FXDataEngineConstants.aiFeatures)
        initialized = true
        configVersion += 1
    }

    public mutating func set(featureWindows windows: [Int], defaultWindow: Int = NormalizationWindowTools.defaultWindow) {
        let resolvedDefault = NormalizationWindowTools.clamp(defaultWindow)
        if !initialized {
            reset(defaultWindow: resolvedDefault)
        } else {
            self.defaultWindow = resolvedDefault
        }
        featureWindows = NormalizationWindowTools.normalizedFeatureWindows(
            windows,
            defaultWindow: resolvedDefault
        )
        configVersion += 1
    }
}

public struct NormalizationWindowRuntimeState: Codable, Hashable, Sendable {
    public var legacy: NormalizationLegacyWindowState
    public var config: NormalizationWindowConfigState

    public init(
        legacy: NormalizationLegacyWindowState = NormalizationLegacyWindowState(),
        config: NormalizationWindowConfigState = NormalizationWindowConfigState()
    ) {
        self.legacy = legacy
        self.config = config
    }

    public mutating func apply(featureWindows: [Int], defaultWindow: Int) {
        config.set(featureWindows: featureWindows, defaultWindow: defaultWindow)
        legacy.apply(featureWindows: featureWindows, defaultWindow: defaultWindow)
    }

    public mutating func applyGroupWindows(
        fast: Int,
        mid: Int,
        slow: Int,
        regime: Int,
        defaultWindow: Int
    ) {
        apply(
            featureWindows: NormalizationWindowTools.buildGroupWindows(
                fast: fast,
                mid: mid,
                slow: slow,
                regime: regime
            ),
            defaultWindow: defaultWindow
        )
    }
}

public struct NormalizationHistoryMethodState: Codable, Hashable, Sendable {
    public var lastSampleTimeUTC: Int64
    public var lastConfigVersion: Int
    public var counts: [Int]
    public var heads: [Int]
    public var values: [Double]

    public init(
        lastSampleTimeUTC: Int64 = 0,
        lastConfigVersion: Int = -1,
        counts: [Int] = [],
        heads: [Int] = [],
        values: [Double] = []
    ) {
        self.lastSampleTimeUTC = max(0, lastSampleTimeUTC)
        self.lastConfigVersion = lastConfigVersion
        self.counts = Self.normalizedIntArray(
            counts,
            count: FXDataEngineConstants.aiFeatures,
            lower: 0,
            upper: RuntimeArtifactConstants.normalizationRollWindowMax
        )
        self.heads = Self.normalizedIntArray(
            heads,
            count: FXDataEngineConstants.aiFeatures,
            lower: 0,
            upper: RuntimeArtifactConstants.normalizationRollWindowMax - 1
        )
        let valueCount = FXDataEngineConstants.aiFeatures * RuntimeArtifactConstants.normalizationRollWindowMax
        var normalizedValues = Array(repeating: 0.0, count: valueCount)
        for index in 0..<min(values.count, valueCount) {
            normalizedValues[index] = fxSafeFinite(values[index])
        }
        self.values = normalizedValues
    }

    public static func valueOffset(featureIndex: Int, windowIndex: Int) -> Int {
        featureIndex * RuntimeArtifactConstants.normalizationRollWindowMax + windowIndex
    }

    public mutating func reset() {
        lastSampleTimeUTC = 0
        lastConfigVersion = -1
        counts = Array(repeating: 0, count: FXDataEngineConstants.aiFeatures)
        heads = Array(repeating: 0, count: FXDataEngineConstants.aiFeatures)
        values = Array(
            repeating: 0.0,
            count: FXDataEngineConstants.aiFeatures * RuntimeArtifactConstants.normalizationRollWindowMax
        )
    }

    public mutating func clearCountsAndHeads() {
        counts = Array(repeating: 0, count: FXDataEngineConstants.aiFeatures)
        heads = Array(repeating: 0, count: FXDataEngineConstants.aiFeatures)
    }

    private static func normalizedIntArray(
        _ values: [Int],
        count: Int,
        lower: Int,
        upper: Int
    ) -> [Int] {
        var output = Array(repeating: lower, count: count)
        let upperBound = max(lower, upper)
        for index in 0..<min(values.count, count) {
            output[index] = min(max(values[index], lower), upperBound)
        }
        return output
    }
}

public struct NormalizationHistoryState: Codable, Hashable, Sendable {
    public var initialized: Bool
    public var methods: [NormalizationHistoryMethodState]

    public init(
        initialized: Bool = false,
        methods: [NormalizationHistoryMethodState] = []
    ) {
        self.initialized = initialized
        self.methods = Array(methods.prefix(FXDataEngineConstants.normMethodCount))
        if self.methods.count < FXDataEngineConstants.normMethodCount {
            self.methods.append(contentsOf: Array(
                repeating: NormalizationHistoryMethodState(),
                count: FXDataEngineConstants.normMethodCount - self.methods.count
            ))
        }
    }

    public mutating func reset() {
        initialized = true
        methods = Array(
            repeating: NormalizationHistoryMethodState(),
            count: FXDataEngineConstants.normMethodCount
        )
    }

    public mutating func resetMethod(_ method: FeatureNormalizationMethod) {
        ensureInitialized()
        methods[method.rawValue].reset()
    }

    @discardableResult
    public mutating func prepareForSample(
        method: FeatureNormalizationMethod,
        sampleTimeUTC: Int64,
        configVersion: Int
    ) -> Bool {
        ensureInitialized()
        guard method.usesRollingNormalizationHistory else {
            methods[method.rawValue].lastConfigVersion = configVersion
            return false
        }

        var state = methods[method.rawValue]
        let rewind = sampleTimeUTC > 0 &&
            state.lastSampleTimeUTC > 0 &&
            sampleTimeUTC <= state.lastSampleTimeUTC
        let configChanged = state.lastConfigVersion != configVersion
        if rewind || configChanged {
            state.clearCountsAndHeads()
        }
        if sampleTimeUTC > 0 {
            state.lastSampleTimeUTC = sampleTimeUTC
        }
        state.lastConfigVersion = configVersion
        methods[method.rawValue] = state
        return rewind || configChanged
    }

    public mutating func record(
        method: FeatureNormalizationMethod,
        featureIndex: Int,
        value: Double
    ) {
        ensureInitialized()
        guard method.usesRollingNormalizationHistory else { return }
        guard featureIndex >= 0, featureIndex < FXDataEngineConstants.aiFeatures else { return }

        var state = methods[method.rawValue]
        var head = state.heads[featureIndex]
        if head < 0 || head >= RuntimeArtifactConstants.normalizationRollWindowMax {
            head = 0
        }
        let offset = NormalizationHistoryMethodState.valueOffset(featureIndex: featureIndex, windowIndex: head)
        state.values[offset] = fxSafeFinite(value)
        head += 1
        if head >= RuntimeArtifactConstants.normalizationRollWindowMax {
            head = 0
        }
        state.heads[featureIndex] = head
        if state.counts[featureIndex] < RuntimeArtifactConstants.normalizationRollWindowMax {
            state.counts[featureIndex] += 1
        }
        methods[method.rawValue] = state
    }

    public func historyCount(
        method: FeatureNormalizationMethod,
        featureIndex: Int,
        window: Int
    ) -> Int {
        guard method.rawValue >= 0, method.rawValue < methods.count else { return 0 }
        guard featureIndex >= 0, featureIndex < FXDataEngineConstants.aiFeatures else { return 0 }
        let limit = NormalizationWindowTools.clamp(window)
        return min(methods[method.rawValue].counts[featureIndex], limit)
    }

    public func recentValues(
        method: FeatureNormalizationMethod,
        featureIndex: Int,
        window: Int
    ) -> [Double] {
        guard method.rawValue >= 0, method.rawValue < methods.count else { return [] }
        guard featureIndex >= 0, featureIndex < FXDataEngineConstants.aiFeatures else { return [] }
        let state = methods[method.rawValue]
        let count = min(state.counts[featureIndex], NormalizationWindowTools.clamp(window))
        guard count > 0 else { return [] }

        var output: [Double] = []
        output.reserveCapacity(count)
        for step in 0..<count {
            var index = state.heads[featureIndex] - 1 - step
            while index < 0 {
                index += RuntimeArtifactConstants.normalizationRollWindowMax
            }
            let offset = NormalizationHistoryMethodState.valueOffset(
                featureIndex: featureIndex,
                windowIndex: index
            )
            output.append(state.values[offset])
        }
        return output
    }

    private mutating func ensureInitialized() {
        if !initialized {
            reset()
        }
    }
}

public struct NormalizationFitFeatureStats: Codable, Hashable, Sendable {
    public var minimum: Double
    public var maximum: Double
    public var mean: Double
    public var standardDeviation: Double
    public var median: Double
    public var interquartileRange: Double
    public var yeoJohnsonLambda: Double
    public var yeoJohnsonMean: Double
    public var yeoJohnsonStandardDeviation: Double
    public var quantiles: [Double]

    public init(
        minimum: Double,
        maximum: Double,
        mean: Double,
        standardDeviation: Double,
        median: Double,
        interquartileRange: Double,
        yeoJohnsonLambda: Double = 1.0,
        yeoJohnsonMean: Double? = nil,
        yeoJohnsonStandardDeviation: Double? = nil,
        quantiles: [Double] = []
    ) {
        self.minimum = fxSafeFinite(minimum)
        self.maximum = fxSafeFinite(maximum)
        self.mean = fxSafeFinite(mean)
        self.standardDeviation = max(1e-6, fxSafeFinite(standardDeviation, fallback: 1.0))
        self.median = fxSafeFinite(median)
        self.interquartileRange = max(1e-6, fxSafeFinite(interquartileRange, fallback: 1.0))
        self.yeoJohnsonLambda = fxSafeFinite(yeoJohnsonLambda, fallback: 1.0)
        self.yeoJohnsonMean = fxSafeFinite(yeoJohnsonMean ?? mean)
        self.yeoJohnsonStandardDeviation = max(
            1e-6,
            fxSafeFinite(yeoJohnsonStandardDeviation ?? standardDeviation, fallback: 1.0)
        )
        var resolved = Array(quantiles.prefix(RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots))
        if resolved.count < RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots {
            let span = self.maximum - self.minimum
            let start = resolved.count
            let needed = RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots - resolved.count
            for offset in 0..<needed {
                let index = start + offset
                let p = RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots > 1
                    ? Double(index) / Double(RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots - 1)
                    : 0.0
                resolved.append(self.minimum + p * span)
            }
        }
        self.quantiles = resolved.map { fxSafeFinite($0) }
    }
}

public enum NormalizationFitTools {
    public static func fallbackStats(featureIndex: Int, registry: FeatureRegistry = FeatureRegistry()) -> NormalizationFitFeatureStats {
        let bounds = registry.clipBounds(for: featureIndex)
        let mean = 0.5 * (bounds.lower + bounds.upper)
        let standardDeviation = max((bounds.upper - bounds.lower) / 4.0, 1e-6)
        let interquartileRange = max((bounds.upper - bounds.lower) / 2.0, 1e-6)
        return NormalizationFitFeatureStats(
            minimum: bounds.lower,
            maximum: bounds.upper,
            mean: mean,
            standardDeviation: standardDeviation,
            median: mean,
            interquartileRange: interquartileRange,
            yeoJohnsonLambda: 1.0,
            yeoJohnsonMean: mean,
            yeoJohnsonStandardDeviation: standardDeviation
        )
    }

    public static func sortedQuantile(_ sortedValues: [Double], q: Double) -> Double {
        let count = sortedValues.count
        guard count > 0 else { return 0.0 }
        guard count > 1 else { return sortedValues[0] }
        let clippedQ = fxClamp(q, 0.0, 1.0)
        let position = clippedQ * Double(count - 1)
        let lowerIndex = Int(floor(position))
        let upperIndex = min(lowerIndex + 1, count - 1)
        let fraction = position - Double(lowerIndex)
        return sortedValues[lowerIndex] + fraction * (sortedValues[upperIndex] - sortedValues[lowerIndex])
    }

    public static func yeoJohnson(_ value: Double, lambda: Double) -> Double {
        let x = fxSafeFinite(value)
        let l = fxSafeFinite(lambda, fallback: 1.0)
        if x >= 0.0 {
            if abs(l) < 1e-8 {
                return log(1.0 + x)
            }
            return (pow(1.0 + x, l) - 1.0) / l
        }

        let lambda2 = 2.0 - l
        if abs(lambda2) < 1e-8 {
            return -log(1.0 - x)
        }
        return -((pow(1.0 - x, lambda2) - 1.0) / lambda2)
    }

    public static func fitYeoJohnsonStats(values: [Double]) -> (lambda: Double, mean: Double, standardDeviation: Double) {
        guard !values.isEmpty else {
            return (1.0, 0.0, 1.0)
        }
        let lambdaGrid = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        var bestLambda = 1.0
        var bestMean = 0.0
        var bestStandardDeviation = 1.0
        var bestScore = Double.greatestFiniteMagnitude

        for lambda in lambdaGrid {
            var sum = 0.0
            var sum2 = 0.0
            for value in values {
                let transformed = yeoJohnson(value, lambda: lambda)
                sum += transformed
                sum2 += transformed * transformed
            }
            let mean = sum / Double(values.count)
            let variance = max((sum2 / Double(values.count)) - (mean * mean), 1e-12)
            let standardDeviation = sqrt(variance)

            var thirdMoment = 0.0
            for value in values {
                let z = (yeoJohnson(value, lambda: lambda) - mean) / standardDeviation
                thirdMoment += z * z * z
            }
            let skew = thirdMoment / Double(values.count)
            let score = abs(skew) + 0.025 * abs(log(standardDeviation + 1e-6))
            if score < bestScore {
                bestScore = score
                bestLambda = lambda
                bestMean = mean
                bestStandardDeviation = standardDeviation
            }
        }

        if bestStandardDeviation < 1e-6 {
            bestStandardDeviation = 1.0
        }
        return (bestLambda, bestMean, bestStandardDeviation)
    }

    public static func fitFeatureStats(values rawValues: [Double], featureIndex: Int) -> NormalizationFitFeatureStats {
        let values = rawValues.map { fxSafeFinite($0) }
        guard !values.isEmpty else {
            return fallbackStats(featureIndex: featureIndex)
        }
        let count = Double(values.count)
        let sum = values.reduce(0.0, +)
        let sum2 = values.reduce(0.0) { $0 + $1 * $1 }
        let mean = sum / count
        let variance = max((sum2 / count) - (mean * mean), 1e-12)
        let standardDeviation = sqrt(variance)
        let sorted = values.sorted()
        let minimum = sorted.first ?? 0.0
        let maximum = sorted.last ?? 0.0
        let median = sortedQuantile(sorted, q: 0.50)
        let q25 = sortedQuantile(sorted, q: 0.25)
        let q75 = sortedQuantile(sorted, q: 0.75)
        let iqr = q75 - q25
        let interquartileRange = Swift.abs(iqr) < 1e-9 ? 1.0 : iqr
        let quantiles = (0..<RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots).map { index in
            let p = RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots > 1
                ? Double(index) / Double(RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots - 1)
                : 0.0
            return sortedQuantile(sorted, q: p)
        }
        let yj = fitYeoJohnsonStats(values: sorted)
        return NormalizationFitFeatureStats(
            minimum: minimum,
            maximum: maximum,
            mean: mean,
            standardDeviation: standardDeviation,
            median: median,
            interquartileRange: interquartileRange,
            yeoJohnsonLambda: yj.lambda,
            yeoJohnsonMean: yj.mean,
            yeoJohnsonStandardDeviation: yj.standardDeviation,
            quantiles: quantiles
        )
    }

    public static func inverseNormalCDF(_ probability: Double) -> Double {
        let x = fxClamp(probability, 1e-12, 1.0 - 1e-12)
        let a1 = -3.969683028665376e+01
        let a2 = 2.209460984245205e+02
        let a3 = -2.759285104469687e+02
        let a4 = 1.383577518672690e+02
        let a5 = -3.066479806614716e+01
        let a6 = 2.506628277459239e+00
        let b1 = -5.447609879822406e+01
        let b2 = 1.615858368580409e+02
        let b3 = -1.556989798598866e+02
        let b4 = 6.680131188771972e+01
        let b5 = -1.328068155288572e+01
        let c1 = -7.784894002430293e-03
        let c2 = -3.223964580411365e-01
        let c3 = -2.400758277161838e+00
        let c4 = -2.549732539343734e+00
        let c5 = 4.374664141464968e+00
        let c6 = 2.938163982698783e+00
        let d1 = 7.784695709041462e-03
        let d2 = 3.224671290700398e-01
        let d3 = 2.445134137142996e+00
        let d4 = 3.754408661907416e+00
        let pLow = 0.02425
        let pHigh = 1.0 - pLow

        if x < pLow {
            let q = sqrt(-2.0 * log(x))
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        }
        if x > pHigh {
            let q = sqrt(-2.0 * log(1.0 - x))
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        }
        let q = x - 0.5
        let r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    }
}

public struct NormalizationFitSlotState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var observations: Int
    public var minimum: [Double]
    public var maximum: [Double]
    public var mean: [Double]
    public var standardDeviation: [Double]
    public var median: [Double]
    public var interquartileRange: [Double]
    public var yeoJohnsonLambda: [Double]
    public var yeoJohnsonMean: [Double]
    public var yeoJohnsonStandardDeviation: [Double]
    public var quantiles: [Double]

    public init(
        ready: Bool = false,
        observations: Int = 0,
        minimum: [Double] = [],
        maximum: [Double] = [],
        mean: [Double] = [],
        standardDeviation: [Double] = [],
        median: [Double] = [],
        interquartileRange: [Double] = [],
        yeoJohnsonLambda: [Double] = [],
        yeoJohnsonMean: [Double] = [],
        yeoJohnsonStandardDeviation: [Double] = [],
        quantiles: [Double] = []
    ) {
        self.ready = ready
        self.observations = max(0, observations)
        let fallback = Self.fallbackArrays()
        self.minimum = Self.normalizedFeatureArray(minimum, fallback: fallback.minimum, floor: nil)
        self.maximum = Self.normalizedFeatureArray(maximum, fallback: fallback.maximum, floor: nil)
        self.mean = Self.normalizedFeatureArray(mean, fallback: fallback.mean, floor: nil)
        self.standardDeviation = Self.normalizedFeatureArray(standardDeviation, fallback: fallback.standardDeviation, floor: 1e-6)
        self.median = Self.normalizedFeatureArray(median, fallback: fallback.median, floor: nil)
        self.interquartileRange = Self.normalizedFeatureArray(interquartileRange, fallback: fallback.interquartileRange, floor: 1e-6)
        self.yeoJohnsonLambda = Self.normalizedFeatureArray(yeoJohnsonLambda, fallback: fallback.yeoJohnsonLambda, floor: nil)
        self.yeoJohnsonMean = Self.normalizedFeatureArray(yeoJohnsonMean, fallback: fallback.yeoJohnsonMean, floor: nil)
        self.yeoJohnsonStandardDeviation = Self.normalizedFeatureArray(
            yeoJohnsonStandardDeviation,
            fallback: fallback.yeoJohnsonStandardDeviation,
            floor: 1e-6
        )
        self.quantiles = Self.normalizedQuantileArray(quantiles, fallback: fallback.quantiles)
    }

    public static func quantileOffset(featureIndex: Int, knotIndex: Int) -> Int {
        featureIndex * RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots + knotIndex
    }

    public func stats(featureIndex: Int) -> NormalizationFitFeatureStats {
        guard featureIndex >= 0, featureIndex < FXDataEngineConstants.aiFeatures else {
            return NormalizationFitTools.fallbackStats(featureIndex: featureIndex)
        }
        let knots = (0..<RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots).map { knot in
            quantiles[Self.quantileOffset(featureIndex: featureIndex, knotIndex: knot)]
        }
        return NormalizationFitFeatureStats(
            minimum: minimum[featureIndex],
            maximum: maximum[featureIndex],
            mean: mean[featureIndex],
            standardDeviation: standardDeviation[featureIndex],
            median: median[featureIndex],
            interquartileRange: interquartileRange[featureIndex],
            yeoJohnsonLambda: yeoJohnsonLambda[featureIndex],
            yeoJohnsonMean: yeoJohnsonMean[featureIndex],
            yeoJohnsonStandardDeviation: yeoJohnsonStandardDeviation[featureIndex],
            quantiles: knots
        )
    }

    public mutating func setStats(_ stats: NormalizationFitFeatureStats, featureIndex: Int) {
        guard featureIndex >= 0, featureIndex < FXDataEngineConstants.aiFeatures else { return }
        minimum[featureIndex] = stats.minimum
        maximum[featureIndex] = stats.maximum
        mean[featureIndex] = stats.mean
        standardDeviation[featureIndex] = stats.standardDeviation
        median[featureIndex] = stats.median
        interquartileRange[featureIndex] = stats.interquartileRange
        yeoJohnsonLambda[featureIndex] = stats.yeoJohnsonLambda
        yeoJohnsonMean[featureIndex] = stats.yeoJohnsonMean
        yeoJohnsonStandardDeviation[featureIndex] = stats.yeoJohnsonStandardDeviation
        for knot in 0..<RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots {
            let offset = Self.quantileOffset(featureIndex: featureIndex, knotIndex: knot)
            quantiles[offset] = stats.quantiles[knot]
        }
    }

    public mutating func clear() {
        self = NormalizationFitSlotState()
    }

    private static func fallbackArrays() -> NormalizationFitSlotState {
        var minimum: [Double] = []
        var maximum: [Double] = []
        var mean: [Double] = []
        var standardDeviation: [Double] = []
        var median: [Double] = []
        var interquartileRange: [Double] = []
        var yeoJohnsonLambda: [Double] = []
        var yeoJohnsonMean: [Double] = []
        var yeoJohnsonStandardDeviation: [Double] = []
        var quantiles: [Double] = []
        minimum.reserveCapacity(FXDataEngineConstants.aiFeatures)
        maximum.reserveCapacity(FXDataEngineConstants.aiFeatures)
        mean.reserveCapacity(FXDataEngineConstants.aiFeatures)
        standardDeviation.reserveCapacity(FXDataEngineConstants.aiFeatures)
        median.reserveCapacity(FXDataEngineConstants.aiFeatures)
        interquartileRange.reserveCapacity(FXDataEngineConstants.aiFeatures)
        yeoJohnsonLambda.reserveCapacity(FXDataEngineConstants.aiFeatures)
        yeoJohnsonMean.reserveCapacity(FXDataEngineConstants.aiFeatures)
        yeoJohnsonStandardDeviation.reserveCapacity(FXDataEngineConstants.aiFeatures)
        quantiles.reserveCapacity(
            FXDataEngineConstants.aiFeatures * RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots
        )
        for feature in 0..<FXDataEngineConstants.aiFeatures {
            let stats = NormalizationFitTools.fallbackStats(featureIndex: feature)
            minimum.append(stats.minimum)
            maximum.append(stats.maximum)
            mean.append(stats.mean)
            standardDeviation.append(stats.standardDeviation)
            median.append(stats.median)
            interquartileRange.append(stats.interquartileRange)
            yeoJohnsonLambda.append(stats.yeoJohnsonLambda)
            yeoJohnsonMean.append(stats.yeoJohnsonMean)
            yeoJohnsonStandardDeviation.append(stats.yeoJohnsonStandardDeviation)
            quantiles.append(contentsOf: stats.quantiles)
        }
        return NormalizationFitSlotState(
            ready: false,
            observations: 0,
            minimum: minimum,
            maximum: maximum,
            mean: mean,
            standardDeviation: standardDeviation,
            median: median,
            interquartileRange: interquartileRange,
            yeoJohnsonLambda: yeoJohnsonLambda,
            yeoJohnsonMean: yeoJohnsonMean,
            yeoJohnsonStandardDeviation: yeoJohnsonStandardDeviation,
            quantiles: quantiles,
            skipFallback: true
        )
    }

    private init(
        ready: Bool,
        observations: Int,
        minimum: [Double],
        maximum: [Double],
        mean: [Double],
        standardDeviation: [Double],
        median: [Double],
        interquartileRange: [Double],
        yeoJohnsonLambda: [Double],
        yeoJohnsonMean: [Double],
        yeoJohnsonStandardDeviation: [Double],
        quantiles: [Double],
        skipFallback: Bool
    ) {
        self.ready = ready
        self.observations = max(0, observations)
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.standardDeviation = standardDeviation
        self.median = median
        self.interquartileRange = interquartileRange
        self.yeoJohnsonLambda = yeoJohnsonLambda
        self.yeoJohnsonMean = yeoJohnsonMean
        self.yeoJohnsonStandardDeviation = yeoJohnsonStandardDeviation
        self.quantiles = quantiles
    }

    private static func normalizedFeatureArray(_ values: [Double], fallback: [Double], floor: Double?) -> [Double] {
        var output = fallback
        for index in 0..<min(values.count, FXDataEngineConstants.aiFeatures) {
            let value = fxSafeFinite(values[index], fallback: fallback[index])
            if let floor {
                output[index] = max(floor, value)
            } else {
                output[index] = value
            }
        }
        return output
    }

    private static func normalizedQuantileArray(_ values: [Double], fallback: [Double]) -> [Double] {
        let count = FXDataEngineConstants.aiFeatures * RuntimeArtifactPayloadMaterializer.normalizationQuantileKnots
        var output = fallback
        for index in 0..<min(values.count, count) {
            output[index] = fxSafeFinite(values[index], fallback: fallback[index])
        }
        return output
    }
}

public struct NormalizationFitState: Codable, Hashable, Sendable {
    public var initialized: Bool
    public var slots: [NormalizationFitSlotState]

    public init(initialized: Bool = false, slots: [NormalizationFitSlotState] = []) {
        self.initialized = initialized
        self.slots = Array(slots.prefix(Self.slotCount))
        if self.slots.count < Self.slotCount {
            self.slots.append(contentsOf: Array(
                repeating: NormalizationFitSlotState(),
                count: Self.slotCount - self.slots.count
            ))
        }
    }

    public static var slotCount: Int {
        RuntimeArtifactConstants.maxHorizons * FXDataEngineConstants.normMethodCount
    }

    public static func slotIndex(horizonSlot: Int, method: FeatureNormalizationMethod) -> Int {
        let slot = Int(fxClamp(Double(horizonSlot), 0.0, Double(RuntimeArtifactConstants.maxHorizons - 1)))
        return slot * FXDataEngineConstants.normMethodCount + method.rawValue
    }

    public mutating func reset() {
        initialized = true
        slots = Array(repeating: NormalizationFitSlotState(), count: Self.slotCount)
    }

    public mutating func clearSlot(horizonSlot: Int, method: FeatureNormalizationMethod) {
        ensureInitialized()
        slots[Self.slotIndex(horizonSlot: horizonSlot, method: method)].clear()
    }

    @discardableResult
    public mutating func fit(
        method: FeatureNormalizationMethod,
        horizonMinutes: Int,
        rawRows: [[Double]],
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) -> Bool {
        ensureInitialized()
        let horizonSlot = TrainingSampleTools.horizonSlot(
            horizonMinutes: horizonMinutes,
            configuredHorizons: configuredHorizons
        )
        let slotIndex = Self.slotIndex(horizonSlot: horizonSlot, method: method)
        slots[slotIndex].clear()
        guard method.usesFittedStats else { return true }
        guard rawRows.count >= 8 else { return false }

        var slot = slots[slotIndex]
        slot.observations = rawRows.count
        for feature in 0..<FXDataEngineConstants.aiFeatures {
            let values = rawRows.map { row -> Double in
                guard feature >= 0, feature < row.count else { return 0.0 }
                return fxSafeFinite(row[feature])
            }
            slot.setStats(NormalizationFitTools.fitFeatureStats(values: values, featureIndex: feature), featureIndex: feature)
        }
        slot.ready = true
        slots[slotIndex] = slot
        return true
    }

    public func featureStats(
        method: FeatureNormalizationMethod,
        horizonMinutes: Int,
        featureIndex: Int,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) -> (stats: NormalizationFitFeatureStats, ready: Bool) {
        let fallback = NormalizationFitTools.fallbackStats(featureIndex: featureIndex)
        guard initialized, method.usesFittedStats else {
            return (fallback, false)
        }
        guard featureIndex >= 0, featureIndex < FXDataEngineConstants.aiFeatures else {
            return (fallback, false)
        }
        let horizonSlot = TrainingSampleTools.horizonSlot(
            horizonMinutes: horizonMinutes,
            configuredHorizons: configuredHorizons
        )
        let slot = slots[Self.slotIndex(horizonSlot: horizonSlot, method: method)]
        return (slot.stats(featureIndex: featureIndex), slot.ready)
    }

    public func quantileToNormal(
        method: FeatureNormalizationMethod,
        horizonMinutes: Int,
        featureIndex: Int,
        value: Double,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) -> Double {
        let lookup = featureStats(
            method: method,
            horizonMinutes: horizonMinutes,
            featureIndex: featureIndex,
            configuredHorizons: configuredHorizons
        )
        let knots = lookup.stats.quantiles
        guard let first = knots.first, let last = knots.last else { return 0.0 }
        let current = fxSafeFinite(value)
        if current <= first {
            return fxClamp(NormalizationFitTools.inverseNormalCDF(1e-6), -6.0, 6.0)
        }
        if current >= last {
            return fxClamp(NormalizationFitTools.inverseNormalCDF(1.0 - 1e-6), -6.0, 6.0)
        }

        var q = 0.5
        for index in 0..<(knots.count - 1) {
            let q0 = knots[index]
            let q1 = knots[index + 1]
            if current > q1 {
                continue
            }
            let p0 = Double(index) / Double(knots.count - 1)
            let p1 = Double(index + 1) / Double(knots.count - 1)
            if abs(q1 - q0) < 1e-9 {
                q = 0.5 * (p0 + p1)
            } else {
                q = p0 + (current - q0) / (q1 - q0) * (p1 - p0)
            }
            break
        }
        return fxClamp(NormalizationFitTools.inverseNormalCDF(fxClamp(q, 1e-6, 1.0 - 1e-6)), -6.0, 6.0)
    }

    private mutating func ensureInitialized() {
        if !initialized {
            reset()
        }
    }
}
