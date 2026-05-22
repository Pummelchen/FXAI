import Foundation

public enum RatesEngineConstants {
    public static let maxReasons = 6
    public static let defaultFreshnessMaxSeconds: Int64 = 900
}

public struct RatesEnginePairState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var available: Bool
    public var stale: Bool
    public var generatedAt: Int64
    public var frontEndDiff: Double
    public var expectedPathDiff: Double
    public var curveDivergenceScore: Double
    public var policyDivergenceScore: Double
    public var ratesRiskScore: Double
    public var macroToRatesTransmissionScore: Double
    public var meetingPathRepriceNow: Bool
    public var ratesRegime: String
    public var tradeGate: String
    public var policyAlignment: String
    public var reasons: [String]

    public init(
        ready: Bool = false,
        available: Bool = false,
        stale: Bool = true,
        generatedAt: Int64 = 0,
        frontEndDiff: Double = 0.0,
        expectedPathDiff: Double = 0.0,
        curveDivergenceScore: Double = 0.0,
        policyDivergenceScore: Double = 0.0,
        ratesRiskScore: Double = 0.0,
        macroToRatesTransmissionScore: Double = 0.0,
        meetingPathRepriceNow: Bool = false,
        ratesRegime: String = "UNKNOWN",
        tradeGate: String = "UNKNOWN",
        policyAlignment: String = "balanced",
        reasons: [String] = []
    ) {
        self.ready = ready
        self.available = available
        self.stale = stale
        self.generatedAt = max(0, generatedAt)
        self.frontEndDiff = fxClamp(frontEndDiff, -10.0, 10.0)
        self.expectedPathDiff = fxClamp(expectedPathDiff, -10.0, 10.0)
        self.curveDivergenceScore = fxClamp(curveDivergenceScore, 0.0, 1.0)
        self.policyDivergenceScore = fxClamp(policyDivergenceScore, 0.0, 1.0)
        self.ratesRiskScore = fxClamp(ratesRiskScore, 0.0, 1.0)
        self.macroToRatesTransmissionScore = fxClamp(macroToRatesTransmissionScore, 0.0, 1.0)
        self.meetingPathRepriceNow = meetingPathRepriceNow
        self.ratesRegime = ratesRegime.isEmpty ? "UNKNOWN" : ratesRegime
        self.tradeGate = tradeGate.isEmpty ? "UNKNOWN" : tradeGate
        self.policyAlignment = policyAlignment.isEmpty ? "balanced" : policyAlignment
        self.reasons = Self.uniqueReasons(reasons)
    }

    public static var reset: RatesEnginePairState {
        RatesEnginePairState()
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
              reasons.count < RatesEngineConstants.maxReasons else {
            return
        }
        reasons.append(value)
    }

    private static func uniqueReasons(_ input: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(input.count, RatesEngineConstants.maxReasons))
        for raw in input {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty,
                  !output.contains(value),
                  output.count < RatesEngineConstants.maxReasons else {
                continue
            }
            output.append(value)
        }
        return output
    }
}

public struct RatesEngineSymbolMapEntry: Codable, Hashable, Sendable {
    public var symbol: String
    public var pairID: String

    public init(symbol: String, pairID: String) {
        self.symbol = symbol.uppercased()
        self.pairID = pairID.uppercased()
    }
}

public enum RatesEngineTools {
    public static func parseSymbolMap(tsv: String) -> [RatesEngineSymbolMapEntry] {
        var entries: [RatesEngineSymbolMapEntry] = []
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 3, parts[0] == "symbol" else { continue }
            let symbol = String(parts[1]).uppercased()
            let pairID = String(parts[2]).uppercased()
            guard !symbol.isEmpty, pairID.count == 6 else { continue }
            entries.append(RatesEngineSymbolMapEntry(symbol: symbol, pairID: pairID))
        }
        return entries
    }

    public static func pairID(
        symbol: String,
        symbolMap: [RatesEngineSymbolMapEntry] = [],
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
        freshnessMaxSeconds: Int64 = RatesEngineConstants.defaultFreshnessMaxSeconds
    ) -> RatesEnginePairState? {
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

    static func parseSnapshot(tsv: String, pairID: String) -> RatesEnginePairState {
        var state = RatesEnginePairState.reset
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
                case "front_end_diff":
                    state.frontEndDiff = Double(value) ?? 0.0
                case "expected_path_diff":
                    state.expectedPathDiff = Double(value) ?? 0.0
                case "curve_divergence_score":
                    state.curveDivergenceScore = Double(value) ?? 0.0
                case "policy_divergence_score":
                    state.policyDivergenceScore = Double(value) ?? 0.0
                case "rates_risk_score":
                    state.ratesRiskScore = Double(value) ?? 0.0
                case "macro_to_rates_transmission_score":
                    state.macroToRatesTransmissionScore = Double(value) ?? 0.0
                case "meeting_path_reprice_now":
                    state.meetingPathRepriceNow = (Int(value) ?? 0) != 0
                case "stale":
                    state.stale = (Int(value) ?? 0) != 0
                case "rates_regime":
                    state.ratesRegime = value
                case "trade_gate":
                    state.tradeGate = value
                case "policy_alignment":
                    state.policyAlignment = value
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
        _ state: RatesEnginePairState,
        nowUTC: Int64,
        freshnessMaxSeconds: Int64
    ) -> RatesEnginePairState {
        var output = state
        if output.available, nowUTC > 0, output.generatedAt > 0, nowUTC - output.generatedAt > max(freshnessMaxSeconds, 60) {
            output.stale = true
        }
        output.frontEndDiff = fxClamp(output.frontEndDiff, -10.0, 10.0)
        output.expectedPathDiff = fxClamp(output.expectedPathDiff, -10.0, 10.0)
        output.curveDivergenceScore = fxClamp(output.curveDivergenceScore, 0.0, 1.0)
        output.policyDivergenceScore = fxClamp(output.policyDivergenceScore, 0.0, 1.0)
        output.ratesRiskScore = fxClamp(output.ratesRiskScore, 0.0, 1.0)
        output.macroToRatesTransmissionScore = fxClamp(output.macroToRatesTransmissionScore, 0.0, 1.0)
        if output.ratesRegime.isEmpty {
            output.ratesRegime = "UNKNOWN"
        }
        if output.tradeGate.isEmpty {
            output.tradeGate = "UNKNOWN"
        }
        if output.policyAlignment.isEmpty {
            output.policyAlignment = "balanced"
        }
        return output
    }
}
