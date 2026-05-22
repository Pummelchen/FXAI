import Foundation

public enum CrossAssetConstants {
    public static let maxReasons = 8
    public static let defaultFreshnessMaxSeconds: Int64 = 900
}

public struct CrossAssetPairState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var available: Bool
    public var stale: Bool
    public var generatedAt: Int64
    public var ratesRepricingScore: Double
    public var riskOffScore: Double
    public var commodityShockScore: Double
    public var volatilityShockScore: Double
    public var usdLiquidityStressScore: Double
    public var crossAssetDislocationScore: Double
    public var pairCrossAssetRiskScore: Double
    public var pairSensitivity: Double
    public var macroState: String
    public var riskState: String
    public var liquidityState: String
    public var tradeGate: String
    public var reasons: [String]

    public init(
        ready: Bool = false,
        available: Bool = false,
        stale: Bool = true,
        generatedAt: Int64 = 0,
        ratesRepricingScore: Double = 0.0,
        riskOffScore: Double = 0.0,
        commodityShockScore: Double = 0.0,
        volatilityShockScore: Double = 0.0,
        usdLiquidityStressScore: Double = 0.0,
        crossAssetDislocationScore: Double = 0.0,
        pairCrossAssetRiskScore: Double = 0.0,
        pairSensitivity: Double = 0.0,
        macroState: String = "UNKNOWN",
        riskState: String = "UNKNOWN",
        liquidityState: String = "UNKNOWN",
        tradeGate: String = "UNKNOWN",
        reasons: [String] = []
    ) {
        self.ready = ready
        self.available = available
        self.stale = stale
        self.generatedAt = max(0, generatedAt)
        self.ratesRepricingScore = ratesRepricingScore
        self.riskOffScore = riskOffScore
        self.commodityShockScore = commodityShockScore
        self.volatilityShockScore = volatilityShockScore
        self.usdLiquidityStressScore = usdLiquidityStressScore
        self.crossAssetDislocationScore = crossAssetDislocationScore
        self.pairCrossAssetRiskScore = pairCrossAssetRiskScore
        self.pairSensitivity = pairSensitivity
        self.macroState = macroState
        self.riskState = riskState
        self.liquidityState = liquidityState
        self.tradeGate = tradeGate
        self.reasons = Self.uniqueReasons(reasons)
    }

    public static var reset: CrossAssetPairState {
        CrossAssetPairState()
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
              reasons.count < CrossAssetConstants.maxReasons else {
            return
        }
        reasons.append(value)
    }

    private static func uniqueReasons(_ input: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(input.count, CrossAssetConstants.maxReasons))
        for raw in input {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty,
                  !output.contains(value),
                  output.count < CrossAssetConstants.maxReasons else {
                continue
            }
            output.append(value)
        }
        return output
    }
}

public struct CrossAssetSymbolMapEntry: Codable, Hashable, Sendable {
    public var symbol: String
    public var pairID: String

    public init(symbol: String, pairID: String) {
        self.symbol = symbol.uppercased()
        self.pairID = pairID.uppercased()
    }
}

public enum CrossAssetTools {
    public static func parseSymbolMap(tsv: String) -> [CrossAssetSymbolMapEntry] {
        var entries: [CrossAssetSymbolMapEntry] = []
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 3, parts[0] == "symbol" else { continue }
            let symbol = String(parts[1]).uppercased()
            let pairID = String(parts[2]).uppercased()
            guard !symbol.isEmpty, pairID.count == 6 else { continue }
            entries.append(CrossAssetSymbolMapEntry(symbol: symbol, pairID: pairID))
        }
        return entries
    }

    public static func pairID(
        symbol: String,
        symbolMap: [CrossAssetSymbolMapEntry] = [],
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
        freshnessMaxSeconds: Int64 = CrossAssetConstants.defaultFreshnessMaxSeconds
    ) -> CrossAssetPairState? {
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

    public static func parseSnapshot(tsv: String, pairID: String) -> CrossAssetPairState {
        var state = CrossAssetPairState.reset
        let targetPairID = pairID.uppercased()
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 4 else { continue }
            let kind = String(parts[0])
            let target = String(parts[1]).uppercased()
            let key = String(parts[2])
            let value = String(parts[3])

            if target == "GLOBAL" {
                if kind == "meta", key == "generated_at_unix" {
                    state.generatedAt = Int64(value) ?? 0
                } else if kind == "score" {
                    switch key {
                    case "rates_repricing_score":
                        state.ratesRepricingScore = Double(value) ?? 0.0
                    case "risk_off_score":
                        state.riskOffScore = Double(value) ?? 0.0
                    case "commodity_shock_score":
                        state.commodityShockScore = Double(value) ?? 0.0
                    case "volatility_shock_score":
                        state.volatilityShockScore = Double(value) ?? 0.0
                    case "usd_liquidity_stress_score":
                        state.usdLiquidityStressScore = Double(value) ?? 0.0
                    case "cross_asset_dislocation_score":
                        state.crossAssetDislocationScore = Double(value) ?? 0.0
                    default:
                        break
                    }
                }
                continue
            }

            if kind == "pair", target == targetPairID {
                state.available = true
                state.ready = true
                switch key {
                case "pair_cross_asset_risk_score":
                    state.pairCrossAssetRiskScore = Double(value) ?? 0.0
                case "pair_sensitivity":
                    state.pairSensitivity = Double(value) ?? 0.0
                case "macro_state":
                    state.macroState = value
                case "risk_state":
                    state.riskState = value
                case "liquidity_state":
                    state.liquidityState = value
                case "trade_gate":
                    state.tradeGate = value
                case "stale":
                    state.stale = (Int(value) ?? 0) != 0
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
        _ state: CrossAssetPairState,
        nowUTC: Int64,
        freshnessMaxSeconds: Int64
    ) -> CrossAssetPairState {
        var output = state
        if output.available {
            if nowUTC > 0, output.generatedAt > 0 {
                output.stale = output.stale || (nowUTC - output.generatedAt > max(freshnessMaxSeconds, 60))
            } else {
                output.stale = true
            }
        }
        return output
    }
}
