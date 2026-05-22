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
