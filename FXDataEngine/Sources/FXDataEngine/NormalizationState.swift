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
