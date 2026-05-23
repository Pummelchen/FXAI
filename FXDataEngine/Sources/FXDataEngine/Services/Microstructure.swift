import Foundation

public enum MicrostructureConstants {
    public static let maxReasons = 6
    public static let defaultFreshnessMaxSeconds: Int64 = 45
}

public struct MicrostructurePairState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var available: Bool
    public var stale: Bool
    public var generatedAt: Int64
    public var tickImbalance30s: Double
    public var directionalEfficiency60s: Double
    public var priceCostCurrent: Double
    public var priceCostZscore60s: Double
    public var tickRate60s: Double
    public var tickRateZscore60s: Double
    public var realizedVol5m: Double
    public var volBurstScore5m: Double
    public var localExtremaBreachScore60s: Double
    public var sweepAndRejectFlag60s: Bool
    public var breakoutReversalScore60s: Double
    public var exhaustionProxy60s: Double
    public var liquidityStressScore: Double
    public var hostileExecutionScore: Double
    public var microstructureRegime: String
    public var sessionTag: String
    public var handoffFlag: Bool
    public var sessionOpenBurstScore: Double
    public var sessionPriceCostBehaviorScore: Double
    public var tradeGate: String
    public var cautionLotScale: Double
    public var cautionEnterProbabilityBuffer: Double
    public var reasons: [String]

    public init(
        ready: Bool = false,
        available: Bool = false,
        stale: Bool = true,
        generatedAt: Int64 = 0,
        tickImbalance30s: Double = 0.0,
        directionalEfficiency60s: Double = 0.0,
        priceCostCurrent: Double = 0.0,
        priceCostZscore60s: Double = 0.0,
        tickRate60s: Double = 0.0,
        tickRateZscore60s: Double = 0.0,
        realizedVol5m: Double = 0.0,
        volBurstScore5m: Double = 0.0,
        localExtremaBreachScore60s: Double = 0.0,
        sweepAndRejectFlag60s: Bool = false,
        breakoutReversalScore60s: Double = 0.0,
        exhaustionProxy60s: Double = 0.0,
        liquidityStressScore: Double = 0.0,
        hostileExecutionScore: Double = 0.0,
        microstructureRegime: String = "UNKNOWN",
        sessionTag: String = "UNKNOWN",
        handoffFlag: Bool = false,
        sessionOpenBurstScore: Double = 0.0,
        sessionPriceCostBehaviorScore: Double = 0.0,
        tradeGate: String = "UNKNOWN",
        cautionLotScale: Double = -1.0,
        cautionEnterProbabilityBuffer: Double = -1.0,
        reasons: [String] = []
    ) {
        self.ready = ready
        self.available = available
        self.stale = stale
        self.generatedAt = max(0, generatedAt)
        self.tickImbalance30s = fxClamp(tickImbalance30s, -1.0, 1.0)
        self.directionalEfficiency60s = fxClamp(directionalEfficiency60s, 0.0, 1.0)
        self.priceCostCurrent = max(0.0, priceCostCurrent)
        self.priceCostZscore60s = fxClamp(priceCostZscore60s, -8.0, 8.0)
        self.tickRate60s = max(0.0, tickRate60s)
        self.tickRateZscore60s = fxClamp(tickRateZscore60s, -8.0, 8.0)
        self.realizedVol5m = max(0.0, realizedVol5m)
        self.volBurstScore5m = fxClamp(volBurstScore5m, 0.0, 8.0)
        self.localExtremaBreachScore60s = fxClamp(localExtremaBreachScore60s, 0.0, 1.0)
        self.sweepAndRejectFlag60s = sweepAndRejectFlag60s
        self.breakoutReversalScore60s = fxClamp(breakoutReversalScore60s, 0.0, 1.0)
        self.exhaustionProxy60s = fxClamp(exhaustionProxy60s, 0.0, 1.0)
        self.liquidityStressScore = fxClamp(liquidityStressScore, 0.0, 1.0)
        self.hostileExecutionScore = fxClamp(hostileExecutionScore, 0.0, 1.0)
        self.microstructureRegime = microstructureRegime.isEmpty ? "UNKNOWN" : microstructureRegime
        self.sessionTag = sessionTag.isEmpty ? "UNKNOWN" : sessionTag
        self.handoffFlag = handoffFlag
        self.sessionOpenBurstScore = fxClamp(sessionOpenBurstScore, 0.0, 1.0)
        self.sessionPriceCostBehaviorScore = fxClamp(sessionPriceCostBehaviorScore, 0.0, 1.0)
        self.tradeGate = tradeGate.isEmpty ? "UNKNOWN" : tradeGate
        self.cautionLotScale = cautionLotScale
        self.cautionEnterProbabilityBuffer = cautionEnterProbabilityBuffer
        self.reasons = Self.uniqueReasons(reasons)
    }

    private enum CodingKeys: String, CodingKey {
        case ready
        case available
        case stale
        case generatedAt
        case tickImbalance30s
        case directionalEfficiency60s
        case priceCostCurrent
        case priceCostZscore60s
        case tickRate60s
        case tickRateZscore60s
        case realizedVol5m
        case volBurstScore5m
        case localExtremaBreachScore60s
        case sweepAndRejectFlag60s
        case breakoutReversalScore60s
        case exhaustionProxy60s
        case liquidityStressScore
        case hostileExecutionScore
        case microstructureRegime
        case sessionTag
        case handoffFlag
        case sessionOpenBurstScore
        case sessionPriceCostBehaviorScore
        case tradeGate
        case cautionLotScale
        case cautionEnterProbabilityBuffer
        case reasons
        case legacySpreadCurrent = "spreadCurrent"
        case legacySpreadZscore60s = "spreadZscore60s"
        case legacySessionSpreadBehaviorScore = "sessionSpreadBehaviorScore"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            ready: try container.decodeIfPresent(Bool.self, forKey: .ready) ?? false,
            available: try container.decodeIfPresent(Bool.self, forKey: .available) ?? false,
            stale: try container.decodeIfPresent(Bool.self, forKey: .stale) ?? true,
            generatedAt: try container.decodeIfPresent(Int64.self, forKey: .generatedAt) ?? 0,
            tickImbalance30s: try container.decodeIfPresent(Double.self, forKey: .tickImbalance30s) ?? 0.0,
            directionalEfficiency60s: try container.decodeIfPresent(Double.self, forKey: .directionalEfficiency60s) ?? 0.0,
            priceCostCurrent: try container.decodeIfPresent(Double.self, forKey: .priceCostCurrent)
                ?? container.decodeIfPresent(Double.self, forKey: .legacySpreadCurrent)
                ?? 0.0,
            priceCostZscore60s: try container.decodeIfPresent(Double.self, forKey: .priceCostZscore60s)
                ?? container.decodeIfPresent(Double.self, forKey: .legacySpreadZscore60s)
                ?? 0.0,
            tickRate60s: try container.decodeIfPresent(Double.self, forKey: .tickRate60s) ?? 0.0,
            tickRateZscore60s: try container.decodeIfPresent(Double.self, forKey: .tickRateZscore60s) ?? 0.0,
            realizedVol5m: try container.decodeIfPresent(Double.self, forKey: .realizedVol5m) ?? 0.0,
            volBurstScore5m: try container.decodeIfPresent(Double.self, forKey: .volBurstScore5m) ?? 0.0,
            localExtremaBreachScore60s: try container.decodeIfPresent(Double.self, forKey: .localExtremaBreachScore60s) ?? 0.0,
            sweepAndRejectFlag60s: try container.decodeIfPresent(Bool.self, forKey: .sweepAndRejectFlag60s) ?? false,
            breakoutReversalScore60s: try container.decodeIfPresent(Double.self, forKey: .breakoutReversalScore60s) ?? 0.0,
            exhaustionProxy60s: try container.decodeIfPresent(Double.self, forKey: .exhaustionProxy60s) ?? 0.0,
            liquidityStressScore: try container.decodeIfPresent(Double.self, forKey: .liquidityStressScore) ?? 0.0,
            hostileExecutionScore: try container.decodeIfPresent(Double.self, forKey: .hostileExecutionScore) ?? 0.0,
            microstructureRegime: try container.decodeIfPresent(String.self, forKey: .microstructureRegime) ?? "UNKNOWN",
            sessionTag: try container.decodeIfPresent(String.self, forKey: .sessionTag) ?? "UNKNOWN",
            handoffFlag: try container.decodeIfPresent(Bool.self, forKey: .handoffFlag) ?? false,
            sessionOpenBurstScore: try container.decodeIfPresent(Double.self, forKey: .sessionOpenBurstScore) ?? 0.0,
            sessionPriceCostBehaviorScore: try container.decodeIfPresent(Double.self, forKey: .sessionPriceCostBehaviorScore)
                ?? container.decodeIfPresent(Double.self, forKey: .legacySessionSpreadBehaviorScore)
                ?? 0.0,
            tradeGate: try container.decodeIfPresent(String.self, forKey: .tradeGate) ?? "UNKNOWN",
            cautionLotScale: try container.decodeIfPresent(Double.self, forKey: .cautionLotScale) ?? -1.0,
            cautionEnterProbabilityBuffer: try container.decodeIfPresent(Double.self, forKey: .cautionEnterProbabilityBuffer) ?? -1.0,
            reasons: try container.decodeIfPresent([String].self, forKey: .reasons) ?? []
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(ready, forKey: .ready)
        try container.encode(available, forKey: .available)
        try container.encode(stale, forKey: .stale)
        try container.encode(generatedAt, forKey: .generatedAt)
        try container.encode(tickImbalance30s, forKey: .tickImbalance30s)
        try container.encode(directionalEfficiency60s, forKey: .directionalEfficiency60s)
        try container.encode(priceCostCurrent, forKey: .priceCostCurrent)
        try container.encode(priceCostZscore60s, forKey: .priceCostZscore60s)
        try container.encode(tickRate60s, forKey: .tickRate60s)
        try container.encode(tickRateZscore60s, forKey: .tickRateZscore60s)
        try container.encode(realizedVol5m, forKey: .realizedVol5m)
        try container.encode(volBurstScore5m, forKey: .volBurstScore5m)
        try container.encode(localExtremaBreachScore60s, forKey: .localExtremaBreachScore60s)
        try container.encode(sweepAndRejectFlag60s, forKey: .sweepAndRejectFlag60s)
        try container.encode(breakoutReversalScore60s, forKey: .breakoutReversalScore60s)
        try container.encode(exhaustionProxy60s, forKey: .exhaustionProxy60s)
        try container.encode(liquidityStressScore, forKey: .liquidityStressScore)
        try container.encode(hostileExecutionScore, forKey: .hostileExecutionScore)
        try container.encode(microstructureRegime, forKey: .microstructureRegime)
        try container.encode(sessionTag, forKey: .sessionTag)
        try container.encode(handoffFlag, forKey: .handoffFlag)
        try container.encode(sessionOpenBurstScore, forKey: .sessionOpenBurstScore)
        try container.encode(sessionPriceCostBehaviorScore, forKey: .sessionPriceCostBehaviorScore)
        try container.encode(tradeGate, forKey: .tradeGate)
        try container.encode(cautionLotScale, forKey: .cautionLotScale)
        try container.encode(cautionEnterProbabilityBuffer, forKey: .cautionEnterProbabilityBuffer)
        try container.encode(reasons, forKey: .reasons)
    }

    public static var reset: MicrostructurePairState {
        MicrostructurePairState()
    }

    public var reasonCount: Int {
        reasons.count
    }

    public var reasonsCSV: String {
        reasons.filter { !$0.isEmpty }.joined(separator: "; ")
    }

    public mutating func appendReason(_ reason: String) {
        let value = Self.normalizedReason(reason)
        guard !value.isEmpty,
              !reasons.contains(value),
              reasons.count < MicrostructureConstants.maxReasons else {
            return
        }
        reasons.append(value)
    }

    private static func uniqueReasons(_ input: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(input.count, MicrostructureConstants.maxReasons))
        for raw in input {
            let value = normalizedReason(raw)
            guard !value.isEmpty,
                  !output.contains(value),
                  output.count < MicrostructureConstants.maxReasons else {
                continue
            }
            output.append(value)
        }
        return output
    }

    private static func normalizedReason(_ raw: String) -> String {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        switch value {
        case "wide_spread":
            return "wide_price_cost"
        case "Spread instability elevated":
            return "Price-cost instability elevated"
        default:
            return value
        }
    }

    @available(*, deprecated, renamed: "priceCostCurrent")
    public var spreadCurrent: Double {
        get { priceCostCurrent }
        set { priceCostCurrent = max(0.0, newValue) }
    }

    @available(*, deprecated, renamed: "priceCostZscore60s")
    public var spreadZscore60s: Double {
        get { priceCostZscore60s }
        set { priceCostZscore60s = fxClamp(newValue, -8.0, 8.0) }
    }

    @available(*, deprecated, renamed: "sessionPriceCostBehaviorScore")
    public var sessionSpreadBehaviorScore: Double {
        get { sessionPriceCostBehaviorScore }
        set { sessionPriceCostBehaviorScore = fxClamp(newValue, 0.0, 1.0) }
    }
}

public struct MicrostructureSymbolMapEntry: Codable, Hashable, Sendable {
    public var symbol: String
    public var pairID: String

    public init(symbol: String, pairID: String) {
        self.symbol = symbol.uppercased()
        self.pairID = pairID.uppercased()
    }
}

public enum MicrostructureTools {
    public static func parseSymbolMap(tsv: String) -> [MicrostructureSymbolMapEntry] {
        var entries: [MicrostructureSymbolMapEntry] = []
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 3, parts[0] == "symbol" else { continue }
            let symbol = String(parts[1]).uppercased()
            let pairID = String(parts[2]).uppercased()
            guard !symbol.isEmpty, pairID.count == 6 else { continue }
            entries.append(MicrostructureSymbolMapEntry(symbol: symbol, pairID: pairID))
        }
        return entries
    }

    public static func pairID(
        symbol: String,
        symbolMap: [MicrostructureSymbolMapEntry] = [],
        newsPulseSymbolMap: [NewsPulseSymbolMapEntry] = []
    ) -> String {
        let cleanSymbol = symbol.uppercased()
        if let mapped = symbolMap.first(where: { $0.symbol == cleanSymbol })?.pairID,
           mapped.count == 6 {
            return mapped
        }
        return NewsPulseTools.pairID(symbol: symbol, symbolMap: newsPulseSymbolMap)
    }

    public static func readPairState(
        symbol: String,
        snapshotTSV: String?,
        symbolMapTSV: String? = nil,
        newsPulseSymbolMapTSV: String? = nil,
        nowUTC: Int64 = 0,
        freshnessMaxSeconds: Int64 = MicrostructureConstants.defaultFreshnessMaxSeconds
    ) -> MicrostructurePairState? {
        let map = symbolMapTSV.map(parseSymbolMap(tsv:)) ?? []
        let newsMap = newsPulseSymbolMapTSV.map(NewsPulseTools.parseSymbolMap(tsv:)) ?? []
        let pairID = pairID(symbol: symbol, symbolMap: map, newsPulseSymbolMap: newsMap)
        guard pairID.count == 6, let snapshotTSV else { return nil }

        let state = normalizedAvailableState(
            parseSnapshot(tsv: snapshotTSV, pairID: pairID),
            nowUTC: nowUTC,
            freshnessMaxSeconds: freshnessMaxSeconds
        )
        return state.available ? state : nil
    }

    public static func parseSnapshot(tsv: String, pairID: String) -> MicrostructurePairState {
        var state = MicrostructurePairState.reset
        let targetPairID = pairID.uppercased()
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 4 else { continue }
            let kind = String(parts[0])
            let target = String(parts[1]).uppercased()
            let key = String(parts[2])
            let value = String(parts[3])

            if kind == "meta", target == "GLOBAL" {
                if key == "generated_at_unix" {
                    state.generatedAt = Int64(value) ?? 0
                }
                continue
            }

            if kind == "pair", target == targetPairID {
                state.available = true
                state.ready = true
                switch key {
                case "tick_imbalance_30s":
                    state.tickImbalance30s = Double(value) ?? 0.0
                case "directional_efficiency_60s":
                    state.directionalEfficiency60s = Double(value) ?? 0.0
                case "spread_current", "price_cost_current":
                    state.priceCostCurrent = Double(value) ?? 0.0
                case "spread_zscore_60s", "price_cost_zscore_60s":
                    state.priceCostZscore60s = Double(value) ?? 0.0
                case "tick_rate_60s":
                    state.tickRate60s = Double(value) ?? 0.0
                case "tick_rate_zscore_60s":
                    state.tickRateZscore60s = Double(value) ?? 0.0
                case "realized_vol_5m":
                    state.realizedVol5m = Double(value) ?? 0.0
                case "vol_burst_score_5m":
                    state.volBurstScore5m = Double(value) ?? 0.0
                case "local_extrema_breach_score_60s":
                    state.localExtremaBreachScore60s = Double(value) ?? 0.0
                case "sweep_and_reject_flag_60s":
                    state.sweepAndRejectFlag60s = (Int(value) ?? 0) != 0
                case "breakout_reversal_score_60s":
                    state.breakoutReversalScore60s = Double(value) ?? 0.0
                case "exhaustion_proxy_60s":
                    state.exhaustionProxy60s = Double(value) ?? 0.0
                case "liquidity_stress_score":
                    state.liquidityStressScore = Double(value) ?? 0.0
                case "hostile_execution_score":
                    state.hostileExecutionScore = Double(value) ?? 0.0
                case "microstructure_regime":
                    state.microstructureRegime = value
                case "session_tag":
                    state.sessionTag = value
                case "handoff_flag":
                    state.handoffFlag = (Int(value) ?? 0) != 0
                case "session_open_burst_score":
                    state.sessionOpenBurstScore = Double(value) ?? 0.0
                case "session_spread_behavior_score", "session_price_cost_behavior_score":
                    state.sessionPriceCostBehaviorScore = Double(value) ?? 0.0
                case "trade_gate":
                    state.tradeGate = value
                case "stale":
                    state.stale = (Int(value) ?? 0) != 0
                case "caution_lot_scale":
                    state.cautionLotScale = Double(value) ?? 0.0
                case "caution_enter_prob_buffer":
                    state.cautionEnterProbabilityBuffer = Double(value) ?? 0.0
                default:
                    break
                }
                continue
            }

            if kind == "pair_reason", target == targetPairID {
                state.appendReason(value)
            }
        }
        return state
    }

    private static func normalizedAvailableState(
        _ state: MicrostructurePairState,
        nowUTC: Int64,
        freshnessMaxSeconds: Int64
    ) -> MicrostructurePairState {
        var output = state
        if output.available, nowUTC > 0, output.generatedAt > 0, nowUTC - output.generatedAt > max(freshnessMaxSeconds, 10) {
            output.stale = true
        }
        output.tickImbalance30s = fxClamp(output.tickImbalance30s, -1.0, 1.0)
        output.directionalEfficiency60s = fxClamp(output.directionalEfficiency60s, 0.0, 1.0)
        output.priceCostCurrent = max(output.priceCostCurrent, 0.0)
        output.priceCostZscore60s = fxClamp(output.priceCostZscore60s, -8.0, 8.0)
        output.tickRate60s = max(output.tickRate60s, 0.0)
        output.tickRateZscore60s = fxClamp(output.tickRateZscore60s, -8.0, 8.0)
        output.realizedVol5m = max(output.realizedVol5m, 0.0)
        output.volBurstScore5m = fxClamp(output.volBurstScore5m, 0.0, 8.0)
        output.localExtremaBreachScore60s = fxClamp(output.localExtremaBreachScore60s, 0.0, 1.0)
        output.breakoutReversalScore60s = fxClamp(output.breakoutReversalScore60s, 0.0, 1.0)
        output.exhaustionProxy60s = fxClamp(output.exhaustionProxy60s, 0.0, 1.0)
        output.liquidityStressScore = fxClamp(output.liquidityStressScore, 0.0, 1.0)
        output.hostileExecutionScore = fxClamp(output.hostileExecutionScore, 0.0, 1.0)
        output.sessionOpenBurstScore = fxClamp(output.sessionOpenBurstScore, 0.0, 1.0)
        output.sessionPriceCostBehaviorScore = fxClamp(output.sessionPriceCostBehaviorScore, 0.0, 1.0)
        if output.microstructureRegime.isEmpty {
            output.microstructureRegime = "UNKNOWN"
        }
        if output.sessionTag.isEmpty {
            output.sessionTag = "UNKNOWN"
        }
        if output.tradeGate.isEmpty {
            output.tradeGate = "UNKNOWN"
        }
        return output
    }
}
