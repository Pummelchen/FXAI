import Foundation

public struct RuntimeTimeContext: Codable, Hashable, Sendable {
    public var ready: Bool
    public var serverNow: Int64
    public var utcNow: Int64
    public var localNow: Int64
    public var serverUTCOffsetSeconds: Int
    public var localUTCOffsetSeconds: Int
    public var sessionBucket: Int
    public var serverDayOfWeek: Int
    public var utcDayOfWeek: Int
    public var localDayOfWeek: Int

    public init(
        ready: Bool = false,
        serverNow: Int64 = 0,
        utcNow: Int64 = 0,
        localNow: Int64 = 0,
        serverUTCOffsetSeconds: Int = 0,
        localUTCOffsetSeconds: Int = 0,
        sessionBucket: Int = 0,
        serverDayOfWeek: Int = -1,
        utcDayOfWeek: Int = -1,
        localDayOfWeek: Int = -1
    ) {
        self.ready = ready
        self.serverNow = max(0, serverNow)
        self.utcNow = max(0, utcNow)
        self.localNow = max(0, localNow)
        self.serverUTCOffsetSeconds = serverUTCOffsetSeconds
        self.localUTCOffsetSeconds = localUTCOffsetSeconds
        self.sessionBucket = Int(fxClamp(Double(sessionBucket), 0.0, Double(FXDataEngineConstants.pluginSessionBuckets - 1)))
        self.serverDayOfWeek = serverDayOfWeek
        self.utcDayOfWeek = utcDayOfWeek
        self.localDayOfWeek = localDayOfWeek
    }

    public static var reset: RuntimeTimeContext {
        RuntimeTimeContext()
    }

    public var summary: String {
        guard ready else { return "time_context_unavailable" }
        return "server=\(serverNow) utc=\(utcNow) local=\(localNow) server_offset=\(serverUTCOffsetSeconds) local_offset=\(localUTCOffsetSeconds) session=\(sessionBucket)"
    }
}

public struct RuntimeRouterCapResult: Codable, Hashable, Sendable {
    public var activeAIIDs: [Int]
    public var activeModelCount: Int
    public var runtimeModelCap: Int
    public var budgetPressure: Double

    public init(activeAIIDs: [Int], runtimeModelCap: Int, budgetPressure: Double) {
        self.activeAIIDs = activeAIIDs
        self.activeModelCount = activeAIIDs.count
        self.runtimeModelCap = runtimeModelCap
        self.budgetPressure = budgetPressure
    }
}

public struct RuntimeFinalizeInput: Codable, Hashable, Sendable {
    public var symbol: String
    public var decision: Int
    public var signalBarUTC: Int64
    public var decisionKey: Int
    public var ensembleMode: Bool
    public var aiType: Int
    public var singleNoTradeReason: String
    public var ensembleMetaTotal: Double
    public var macroProfileShortfall: Double
    public var regimeTransitionPenalty: Double
    public var tradeGate: Double
    public var policyTradeProbability: Double
    public var policyConfidence: Double
    public var policySizeMultiplier: Double
    public var probabilityCalibrationReady: Bool
    public var probabilityCalibrationPrimaryReason: String

    public init(
        symbol: String = "",
        decision: Int = -1,
        signalBarUTC: Int64 = 0,
        decisionKey: Int = 0,
        ensembleMode: Bool = false,
        aiType: Int = -1,
        singleNoTradeReason: String = "",
        ensembleMetaTotal: Double = 0.0,
        macroProfileShortfall: Double = 0.0,
        regimeTransitionPenalty: Double = 0.0,
        tradeGate: Double = 0.0,
        policyTradeProbability: Double = 0.0,
        policyConfidence: Double = 0.0,
        policySizeMultiplier: Double = 1.0,
        probabilityCalibrationReady: Bool = false,
        probabilityCalibrationPrimaryReason: String = ""
    ) {
        self.symbol = symbol
        self.decision = decision
        self.signalBarUTC = max(0, signalBarUTC)
        self.decisionKey = decisionKey
        self.ensembleMode = ensembleMode
        self.aiType = aiType
        self.singleNoTradeReason = singleNoTradeReason
        self.ensembleMetaTotal = ensembleMetaTotal
        self.macroProfileShortfall = macroProfileShortfall
        self.regimeTransitionPenalty = regimeTransitionPenalty
        self.tradeGate = tradeGate
        self.policyTradeProbability = policyTradeProbability
        self.policyConfidence = policyConfidence
        self.policySizeMultiplier = policySizeMultiplier
        self.probabilityCalibrationReady = probabilityCalibrationReady
        self.probabilityCalibrationPrimaryReason = probabilityCalibrationPrimaryReason
    }
}

public struct RuntimeFinalizedSignal: Codable, Hashable, Sendable {
    public var symbol: String
    public var decision: Int
    public var signalBarUTC: Int64
    public var decisionKey: Int
    public var reason: String
    public var signalIntensity: Double

    public init(
        symbol: String,
        decision: Int,
        signalBarUTC: Int64,
        decisionKey: Int,
        reason: String,
        signalIntensity: Double
    ) {
        self.symbol = symbol
        self.decision = decision
        self.signalBarUTC = max(0, signalBarUTC)
        self.decisionKey = decisionKey
        self.reason = reason
        self.signalIntensity = fxClamp(signalIntensity, 0.0, 4.0)
    }
}

public enum RuntimeStageTools {
    public static let m1SyncAIID = 28

    public static func buildTimeContext(
        serverNow: Int64,
        utcNow: Int64? = nil,
        localNow: Int64? = nil
    ) -> RuntimeTimeContext {
        guard serverNow > 0 else { return .reset }

        let resolvedUTC = max(utcNow ?? serverNow, 0)
        let resolvedLocal = max(localNow ?? serverNow, 0)
        let effectiveUTC = resolvedUTC > 0 ? resolvedUTC : serverNow
        let effectiveLocal = resolvedLocal > 0 ? resolvedLocal : serverNow

        return RuntimeTimeContext(
            ready: true,
            serverNow: serverNow,
            utcNow: effectiveUTC,
            localNow: effectiveLocal,
            serverUTCOffsetSeconds: Int(serverNow - effectiveUTC),
            localUTCOffsetSeconds: Int(effectiveLocal - effectiveUTC),
            sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: serverNow),
            serverDayOfWeek: dayOfWeek(timestamp: serverNow),
            utcDayOfWeek: dayOfWeek(timestamp: effectiveUTC),
            localDayOfWeek: dayOfWeek(timestamp: effectiveLocal)
        )
    }

    public static func finalizeDecision(_ input: RuntimeFinalizeInput) -> RuntimeFinalizedSignal {
        let reason: String
        if input.decision == 1 {
            reason = "buy"
        } else if input.decision == 0 {
            reason = "sell"
        } else if input.probabilityCalibrationReady, !input.probabilityCalibrationPrimaryReason.isEmpty {
            reason = input.probabilityCalibrationPrimaryReason
        } else if !input.ensembleMode, input.aiType == m1SyncAIID, !input.singleNoTradeReason.isEmpty {
            reason = input.singleNoTradeReason
        } else if input.ensembleMode, input.ensembleMetaTotal <= 0.0 {
            reason = "no_meta_weight"
        } else {
            reason = "no_consensus_or_ev"
        }

        var intensity = fxClamp(
            (0.55 * input.tradeGate +
                0.25 * input.policyTradeProbability +
                0.20 * input.policyConfidence) *
                fxClamp(input.policySizeMultiplier, 0.25, 1.60) *
                fxClamp(
                    1.0 - 0.35 * input.macroProfileShortfall - 0.20 * input.regimeTransitionPenalty,
                    0.20,
                    1.0
                ),
            0.0,
            4.0
        )
        if input.decision < 0 {
            intensity = 0.0
        }

        return RuntimeFinalizedSignal(
            symbol: input.symbol,
            decision: input.decision,
            signalBarUTC: input.signalBarUTC,
            decisionKey: input.decisionKey,
            reason: reason,
            signalIntensity: intensity
        )
    }

    public static func serverToUTC(_ serverTime: Int64, context: RuntimeTimeContext) -> Int64 {
        guard serverTime > 0 else { return 0 }
        return serverTime - Int64(context.ready ? context.serverUTCOffsetSeconds : 0)
    }

    public static func utcToServer(_ utcTime: Int64, context: RuntimeTimeContext) -> Int64 {
        guard utcTime > 0 else { return 0 }
        return utcTime + Int64(context.ready ? context.serverUTCOffsetSeconds : 0)
    }

    public static func localToServer(_ localTime: Int64, context: RuntimeTimeContext) -> Int64 {
        guard localTime > 0 else { return 0 }
        guard context.ready else { return localTime }
        return localTime - Int64(context.localUTCOffsetSeconds) + Int64(context.serverUTCOffsetSeconds)
    }

    public static func applyPerformanceModelCap(
        activeAIIDs: [Int],
        deployProfile: LiveDeploymentProfile,
        performance: RuntimePerformanceState
    ) -> RuntimeRouterCapResult {
        var runtimeModelCap = deployProfile.maxRuntimeModels
        if runtimeModelCap <= 0 {
            runtimeModelCap = FXDataEngineConstants.aiCount
        }
        runtimeModelCap = Int(fxClamp(Double(runtimeModelCap), 1.0, Double(FXDataEngineConstants.aiCount)))

        let pressure = performance.budgetPressure(budgetMS: deployProfile.performanceBudgetMS)
        if pressure > 0.05, runtimeModelCap > 1 {
            let pressureScale = fxClamp(1.0 - 0.55 * min(pressure, 1.0), 0.25, 1.0)
            runtimeModelCap = max(1, Int(floor(Double(runtimeModelCap) * pressureScale)))
        }

        let capped = activeAIIDs.count > runtimeModelCap ? Array(activeAIIDs.prefix(runtimeModelCap)) : activeAIIDs
        return RuntimeRouterCapResult(
            activeAIIDs: capped,
            runtimeModelCap: runtimeModelCap,
            budgetPressure: pressure
        )
    }

    private static func dayOfWeek(timestamp: Int64) -> Int {
        guard timestamp > 0 else { return -1 }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        return calendar.component(.weekday, from: Date(timeIntervalSince1970: TimeInterval(timestamp))) - 1
    }
}
