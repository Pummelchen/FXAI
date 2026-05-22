import Foundation

public enum RuntimeCycleAction: String, Codable, Sendable {
    case continueStages
    case publishIdle
    case restoreCachedSignal
}

public struct RuntimeCycleSettings: Codable, Hashable, Sendable {
    public var baseHorizonMinutes: Int
    public var windowBars: Int
    public var onlineSamples: Int
    public var onlineEpochs: Int
    public var trainEpochs: Int
    public var aiType: Int
    public var ensembleMode: Bool
    public var agreePercent: Double
    public var thresholds: WarmupThresholdPair
    public var evThresholdPoints: Double
    public var evLookbackSamples: Int
    public var decisionKey: Int

    public init(
        baseHorizonMinutes: Int,
        windowBars: Int,
        onlineSamples: Int,
        onlineEpochs: Int,
        trainEpochs: Int,
        aiType: Int,
        ensembleMode: Bool,
        agreePercent: Double,
        thresholds: WarmupThresholdPair,
        evThresholdPoints: Double,
        evLookbackSamples: Int,
        decisionKey: Int
    ) {
        self.baseHorizonMinutes = HorizonTools.clampHorizon(baseHorizonMinutes)
        self.windowBars = min(max(windowBars, 50), 500)
        self.onlineSamples = min(max(onlineSamples, 5), 200)
        self.onlineEpochs = min(max(onlineEpochs, 1), 5)
        self.trainEpochs = min(max(trainEpochs, 1), 20)
        self.aiType = RuntimeCycleTools.validatedAIType(aiType)
        self.ensembleMode = ensembleMode
        self.agreePercent = RuntimeCycleTools.clampedAgreePercent(agreePercent)
        self.thresholds = WarmupTools.sanitizeThresholdPair(buyThreshold: thresholds.buy, sellThreshold: thresholds.sell)
        self.evThresholdPoints = fxClamp(evThresholdPoints, 0.0, 100.0)
        self.evLookbackSamples = min(max(evLookbackSamples, 20), 400)
        self.decisionKey = decisionKey
    }
}

public struct RuntimeCycleInput: Codable, Hashable, Sendable {
    public var symbol: String
    public var pluginsReady: Bool
    public var predictionTargetMinutes: Int
    public var aiWindow: Int
    public var onlineSamples: Int
    public var onlineEpochs: Int
    public var trainEpochs: Int
    public var aiType: Int
    public var ensembleMode: Bool
    public var ensembleAgreePercent: Double
    public var buyThreshold: Double
    public var sellThreshold: Double
    public var evThresholdPoints: Double
    public var evLookbackSamples: Int
    public var lastSymbol: String
    public var warmupEnabled: Bool
    public var warmupDone: Bool
    public var signalBarUTC: Int64
    public var lastSignalBarUTC: Int64
    public var lastSignalDecisionKey: Int
    public var lastSignal: Int

    public init(
        symbol: String,
        pluginsReady: Bool = true,
        predictionTargetMinutes: Int,
        aiWindow: Int,
        onlineSamples: Int,
        onlineEpochs: Int,
        trainEpochs: Int,
        aiType: Int,
        ensembleMode: Bool,
        ensembleAgreePercent: Double,
        buyThreshold: Double,
        sellThreshold: Double,
        evThresholdPoints: Double,
        evLookbackSamples: Int,
        lastSymbol: String = "",
        warmupEnabled: Bool = false,
        warmupDone: Bool = true,
        signalBarUTC: Int64,
        lastSignalBarUTC: Int64 = 0,
        lastSignalDecisionKey: Int = Int.min,
        lastSignal: Int = -1
    ) {
        self.symbol = symbol
        self.pluginsReady = pluginsReady
        self.predictionTargetMinutes = predictionTargetMinutes
        self.aiWindow = aiWindow
        self.onlineSamples = onlineSamples
        self.onlineEpochs = onlineEpochs
        self.trainEpochs = trainEpochs
        self.aiType = aiType
        self.ensembleMode = ensembleMode
        self.ensembleAgreePercent = ensembleAgreePercent
        self.buyThreshold = buyThreshold
        self.sellThreshold = sellThreshold
        self.evThresholdPoints = evThresholdPoints
        self.evLookbackSamples = evLookbackSamples
        self.lastSymbol = lastSymbol
        self.warmupEnabled = warmupEnabled
        self.warmupDone = warmupDone
        self.signalBarUTC = max(0, signalBarUTC)
        self.lastSignalBarUTC = max(0, lastSignalBarUTC)
        self.lastSignalDecisionKey = lastSignalDecisionKey
        self.lastSignal = lastSignal
    }
}

public struct RuntimeCyclePlan: Codable, Hashable, Sendable {
    public var action: RuntimeCycleAction
    public var reason: String
    public var settings: RuntimeCycleSettings
    public var requiresStateReset: Bool
    public var publishIdleSnapshot: Bool
    public var signalBarUTC: Int64
    public var returnedSignal: Int?

    public init(
        action: RuntimeCycleAction,
        reason: String,
        settings: RuntimeCycleSettings,
        requiresStateReset: Bool = false,
        publishIdleSnapshot: Bool = false,
        signalBarUTC: Int64 = 0,
        returnedSignal: Int? = nil
    ) {
        self.action = action
        self.reason = reason
        self.settings = settings
        self.requiresStateReset = requiresStateReset
        self.publishIdleSnapshot = publishIdleSnapshot
        self.signalBarUTC = max(0, signalBarUTC)
        self.returnedSignal = returnedSignal
    }
}

public enum RuntimeCycleTools {
    public static func validatedAIType(_ aiType: Int) -> Int {
        guard (0..<FXDataEngineConstants.aiCount).contains(aiType) else {
            return WarmupTools.sgdLogitAIID
        }
        return aiType
    }

    public static func decisionKey(ensembleMode: Bool, aiType: Int, agreePercent: Double) -> Int {
        if ensembleMode {
            let pctKey = Int((clampedAgreePercent(agreePercent) * 10.0).rounded())
            return 100_000 + pctKey
        }
        return validatedAIType(aiType)
    }

    public static func buildSettings(_ input: RuntimeCycleInput) -> RuntimeCycleSettings {
        let aiType = validatedAIType(input.aiType)
        let agreePercent = clampedAgreePercent(input.ensembleAgreePercent)
        return RuntimeCycleSettings(
            baseHorizonMinutes: input.predictionTargetMinutes,
            windowBars: input.aiWindow,
            onlineSamples: input.onlineSamples,
            onlineEpochs: input.onlineEpochs,
            trainEpochs: input.trainEpochs,
            aiType: aiType,
            ensembleMode: input.ensembleMode,
            agreePercent: agreePercent,
            thresholds: WarmupTools.sanitizeThresholdPair(
                buyThreshold: input.buyThreshold,
                sellThreshold: input.sellThreshold
            ),
            evThresholdPoints: input.evThresholdPoints,
            evLookbackSamples: input.evLookbackSamples,
            decisionKey: decisionKey(
                ensembleMode: input.ensembleMode,
                aiType: aiType,
                agreePercent: agreePercent
            )
        )
    }

    public static func planCycle(_ input: RuntimeCycleInput) -> RuntimeCyclePlan {
        let settings = buildSettings(input)

        guard input.pluginsReady else {
            return RuntimeCyclePlan(
                action: .publishIdle,
                reason: "plugins_not_ready",
                settings: settings,
                publishIdleSnapshot: true,
                returnedSignal: -1
            )
        }

        let requiresStateReset = input.lastSymbol != input.symbol

        if input.warmupEnabled, !input.warmupDone {
            return RuntimeCyclePlan(
                action: .publishIdle,
                reason: "warmup_pending",
                settings: settings,
                requiresStateReset: requiresStateReset,
                publishIdleSnapshot: true,
                returnedSignal: -1
            )
        }

        guard input.signalBarUTC > 0 else {
            return RuntimeCyclePlan(
                action: .publishIdle,
                reason: "bar_time_failed",
                settings: settings,
                requiresStateReset: requiresStateReset,
                publishIdleSnapshot: true,
                returnedSignal: -1
            )
        }

        if !requiresStateReset,
           input.lastSignalBarUTC == input.signalBarUTC,
           input.lastSignalDecisionKey == settings.decisionKey {
            return RuntimeCyclePlan(
                action: .restoreCachedSignal,
                reason: "signal_cache_hit",
                settings: settings,
                requiresStateReset: requiresStateReset,
                signalBarUTC: input.signalBarUTC,
                returnedSignal: normalizedSignal(input.lastSignal)
            )
        }

        return RuntimeCyclePlan(
            action: .continueStages,
            reason: "continue",
            settings: settings,
            requiresStateReset: requiresStateReset,
            signalBarUTC: input.signalBarUTC
        )
    }

    private static func normalizedSignal(_ signal: Int) -> Int {
        signal == 0 || signal == 1 ? signal : -1
    }

    static func clampedAgreePercent(_ value: Double) -> Double {
        fxClamp(value.isFinite ? value : 50.0, 50.0, 100.0)
    }
}
