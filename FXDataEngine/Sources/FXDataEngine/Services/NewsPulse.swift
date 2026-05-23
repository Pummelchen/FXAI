import Foundation

public enum NewsPulseConstants {
    public static let maxReasons = 6
    public static let defaultFreshnessMaxSeconds: Int64 = 360
}

public struct NewsPulsePairState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var available: Bool
    public var stale: Bool
    public var generatedAt: Int64
    public var eventETAMinutes: Int
    public var newsRiskScore: Double
    public var newsPressure: Double
    public var tradeGate: String
    public var sessionProfile: String
    public var calibrationProfile: String
    public var watchlistTagsCSV: String
    public var cautionLotScale: Double
    public var cautionEnterProbabilityBuffer: Double
    public var reasons: [String]

    public init(
        ready: Bool = false,
        available: Bool = false,
        stale: Bool = true,
        generatedAt: Int64 = 0,
        eventETAMinutes: Int = -1,
        newsRiskScore: Double = 0.0,
        newsPressure: Double = 0.0,
        tradeGate: String = "UNKNOWN",
        sessionProfile: String = "default",
        calibrationProfile: String = "default",
        watchlistTagsCSV: String = "",
        cautionLotScale: Double = -1.0,
        cautionEnterProbabilityBuffer: Double = -1.0,
        reasons: [String] = []
    ) {
        self.ready = ready
        self.available = available
        self.stale = stale
        self.generatedAt = max(0, generatedAt)
        self.eventETAMinutes = eventETAMinutes
        self.newsRiskScore = fxClamp(newsRiskScore, 0.0, 1.0)
        self.newsPressure = fxClamp(newsPressure, -1.0, 1.0)
        self.tradeGate = tradeGate.isEmpty ? "UNKNOWN" : tradeGate
        self.sessionProfile = sessionProfile.isEmpty ? "default" : sessionProfile
        self.calibrationProfile = calibrationProfile.isEmpty ? self.sessionProfile : calibrationProfile
        self.watchlistTagsCSV = watchlistTagsCSV
        self.cautionLotScale = cautionLotScale >= 0.0 ? fxClamp(cautionLotScale, 0.10, 1.0) : cautionLotScale
        self.cautionEnterProbabilityBuffer = cautionEnterProbabilityBuffer >= 0.0 ?
            fxClamp(cautionEnterProbabilityBuffer, 0.0, 0.25) :
            cautionEnterProbabilityBuffer
        self.reasons = Self.uniqueReasons(reasons)
    }

    public static var reset: NewsPulsePairState {
        NewsPulsePairState()
    }

    public var reasonCount: Int {
        reasons.count
    }

    public var reasonsCSV: String {
        reasons.filter { !$0.isEmpty }.joined(separator: "; ")
    }

    public mutating func appendReason(_ reason: String) {
        let value = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty,
              !reasons.contains(value),
              reasons.count < NewsPulseConstants.maxReasons else {
            return
        }
        reasons.append(value)
    }

    private static func uniqueReasons(_ input: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(input.count, NewsPulseConstants.maxReasons))
        for raw in input {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty,
                  !output.contains(value),
                  output.count < NewsPulseConstants.maxReasons else {
                continue
            }
            output.append(value)
        }
        return output
    }
}

public struct NewsPulseSymbolMapEntry: Codable, Hashable, Sendable {
    public var symbol: String
    public var pairID: String

    public init(symbol: String, pairID: String) {
        self.symbol = symbol.uppercased()
        self.pairID = pairID.uppercased()
    }
}

public enum NewsPulseTools {
    public static func parseSymbolMap(tsv: String) -> [NewsPulseSymbolMapEntry] {
        var entries: [NewsPulseSymbolMapEntry] = []
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 3, parts[0] == "symbol" else { continue }
            let symbol = String(parts[1]).uppercased()
            let pairID = String(parts[2]).uppercased()
            guard !symbol.isEmpty, pairID.count == 6 else { continue }
            entries.append(NewsPulseSymbolMapEntry(symbol: symbol, pairID: pairID))
        }
        return entries
    }

    public static func pairID(symbol: String, symbolMap: [NewsPulseSymbolMapEntry] = []) -> String {
        let cleanSymbol = symbol.uppercased()
        if let mapped = symbolMap.first(where: { $0.symbol == cleanSymbol })?.pairID,
           mapped.count == 6 {
            return mapped
        }
        return heuristicPairID(symbol: symbol)
    }

    public static func heuristicPairID(symbol: String) -> String {
        let alpha = alphaOnly(symbol)
        guard alpha.count >= 6 else { return "" }
        let chars = Array(alpha)
        for start in 0...(chars.count - 6) {
            let candidate = String(chars[start..<(start + 6)])
            let base = String(candidate.prefix(3))
            let quote = String(candidate.suffix(3))
            if base == quote {
                continue
            }
            if isSupportedCurrency(base), isSupportedCurrency(quote) {
                return candidate
            }
        }
        return ""
    }

    public static func readPairState(
        symbol: String,
        snapshotTSV: String?,
        symbolMapTSV: String? = nil,
        calendarFallback: CalendarCachePairState? = nil,
        nowUTC: Int64 = 0,
        freshnessMaxSeconds: Int64 = NewsPulseConstants.defaultFreshnessMaxSeconds
    ) -> NewsPulsePairState? {
        let map = symbolMapTSV.map(parseSymbolMap(tsv:)) ?? []
        let pairID = pairID(symbol: symbol, symbolMap: map)
        guard pairID.count == 6 else { return nil }

        var state = NewsPulsePairState.reset
        if let snapshotTSV {
            state = parseSnapshot(tsv: snapshotTSV, pairID: pairID)
        }

        if state.available {
            state = normalizedAvailableState(state, nowUTC: nowUTC, freshnessMaxSeconds: freshnessMaxSeconds)
        }

        if (!state.available || state.stale || state.tradeGate.isEmpty || state.tradeGate == "UNKNOWN"),
           let calendarFallback,
           applyCalendarFallback(&state, calendar: calendarFallback) {
            state.available = true
            state.ready = true
        }

        return state.available ? state : nil
    }

    public static func parseSnapshot(tsv: String, pairID: String) -> NewsPulsePairState {
        var state = NewsPulsePairState.reset
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
                case "event_eta_min":
                    state.eventETAMinutes = value.isEmpty ? -1 : (Int(value) ?? 0)
                case "news_risk_score":
                    state.newsRiskScore = Double(value) ?? 0.0
                case "trade_gate":
                    state.tradeGate = value
                case "news_pressure":
                    state.newsPressure = Double(value) ?? 0.0
                case "stale":
                    state.stale = (Int(value) ?? 0) != 0
                case "session_profile":
                    state.sessionProfile = value
                case "calibration_profile":
                    state.calibrationProfile = value
                case "watchlist_tags":
                    state.watchlistTagsCSV = value
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

    public static func alphaOnly(_ symbol: String) -> String {
        String(symbol.uppercased().filter { $0 >= "A" && $0 <= "Z" })
    }

    public static func isSupportedCurrency(_ code: String) -> Bool {
        switch code {
        case "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK":
            return true
        default:
            return false
        }
    }

    private static func normalizedAvailableState(
        _ state: NewsPulsePairState,
        nowUTC: Int64,
        freshnessMaxSeconds: Int64
    ) -> NewsPulsePairState {
        var output = state
        if nowUTC > 0, output.generatedAt > 0, nowUTC - output.generatedAt > max(freshnessMaxSeconds, 60) {
            output.stale = true
        }
        if output.tradeGate.isEmpty {
            output.tradeGate = "UNKNOWN"
        }
        if output.sessionProfile.isEmpty {
            output.sessionProfile = "default"
        }
        if output.calibrationProfile.isEmpty {
            output.calibrationProfile = output.sessionProfile
        }
        output.newsRiskScore = fxClamp(output.newsRiskScore, 0.0, 1.0)
        output.newsPressure = fxClamp(output.newsPressure, -1.0, 1.0)
        if output.cautionLotScale >= 0.0 {
            output.cautionLotScale = fxClamp(output.cautionLotScale, 0.10, 1.0)
        }
        if output.cautionEnterProbabilityBuffer >= 0.0 {
            output.cautionEnterProbabilityBuffer = fxClamp(output.cautionEnterProbabilityBuffer, 0.0, 0.25)
        }
        return output
    }

    private static func applyCalendarFallback(
        _ state: inout NewsPulsePairState,
        calendar: CalendarCachePairState
    ) -> Bool {
        guard calendar.ready else { return false }
        state.available = calendar.ready
        state.ready = calendar.ready
        state.stale = calendar.stale
        state.generatedAt = calendar.generatedAt
        state.eventETAMinutes = calendar.nextEventETAMinutes
        state.newsRiskScore = fxClamp(calendar.eventRiskScore, 0.0, 1.0)
        state.newsPressure = fxClamp(0.50 * calendar.eventRiskScore, -1.0, 1.0)
        state.tradeGate = calendar.tradeGate.rawValue
        state.sessionProfile = "calendar_cache"
        state.calibrationProfile = "calendar_cache"
        state.watchlistTagsCSV = "mt5_calendar_cache"
        state.cautionLotScale = calendar.cautionLotScale
        state.cautionEnterProbabilityBuffer = calendar.cautionEnterProbabilityBuffer
        for reason in calendar.reasons {
            state.appendReason(reason)
        }
        state.appendReason("calendar_cache_fallback")
        return state.ready
    }
}
