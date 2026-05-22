import Foundation

public enum LifecycleContextSymbolCategory: Int, Codable, Sendable, CaseIterable {
    case fx = 0
    case metal = 1
    case index = 2
    case energy = 3
    case crypto = 4
    case risk = 5
    case other = 6
}

public struct LifecycleContextCandidateInputs: Codable, Hashable, Sendable {
    public var liquidityScore: Double
    public var dataHealthScore: Double
    public var sessionOverlapScore: Double

    public init(
        liquidityScore: Double = 0.0,
        dataHealthScore: Double = 0.0,
        sessionOverlapScore: Double = 0.0
    ) {
        self.liquidityScore = liquidityScore
        self.dataHealthScore = dataHealthScore
        self.sessionOverlapScore = sessionOverlapScore
    }
}

public enum LifecycleContextTools {
    public static let curatedSymbols = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURJPY", "GBPJPY", "EURGBP",
        "XAUUSD", "XAGUSD",
        "US500", "SPX500", "NAS100", "USTEC", "US30", "DE40", "GER40", "UK100", "JP225",
        "WTI", "BRENT",
        "VIX", "DXY", "USDX", "TNX", "USB10", "US10Y", "BUND"
    ]

    public static func category(symbol: String) -> LifecycleContextSymbolCategory {
        let symbol = symbol.uppercased()
        if ["XAU", "XAG", "XPT", "XPD"].contains(where: { symbol.contains($0) }) {
            return .metal
        }
        if ["US30", "DE40", "GER40", "JP225", "NAS100", "USTEC", "SPX500", "US500", "UK100", "HK50", "AUS200", "FRA40"].contains(where: { symbol.contains($0) }) {
            return .index
        }
        if ["WTI", "XTI", "BRENT", "NATGAS", "NGAS"].contains(where: { symbol.contains($0) }) {
            return .energy
        }
        if ["BTC", "ETH", "XRP", "SOL", "ADA", "LTC"].contains(where: { symbol.contains($0) }) {
            return .crypto
        }
        if ["VIX", "VOL", "DXY", "USDX", "DOLLAR", "TNX", "USB10", "US10Y", "US02Y", "BUND", "GILT", "JGB"].contains(where: { symbol.contains($0) }) {
            return .risk
        }
        if symbol.count >= 6 {
            let start = symbol.prefix(3)
            let end = symbol.dropFirst(3).prefix(3)
            if start.allSatisfy(\.isLetter), end.allSatisfy(\.isLetter) {
                return .fx
            }
        }
        return .other
    }

    public static func categoryPriority(_ category: LifecycleContextSymbolCategory) -> Double {
        switch category {
        case .fx: 1.00
        case .metal: 0.92
        case .index: 0.88
        case .energy: 0.76
        case .crypto: 0.60
        case .risk: 0.94
        case .other: 0.50
        }
    }

    public static func sharedSymbolScore(mainSymbol: String, candidateSymbol: String) -> Double {
        let main = mainSymbol.uppercased()
        let candidate = candidateSymbol.uppercased()
        guard main.count >= 6, candidate.count >= 6 else { return 0.0 }
        let mainA = String(main.prefix(3))
        let mainB = String(main.dropFirst(3).prefix(3))
        let candidateA = String(candidate.prefix(3))
        let candidateB = String(candidate.dropFirst(3).prefix(3))

        var score = 0.0
        if mainA == candidateA || mainA == candidateB {
            score += 0.35
        }
        if mainB == candidateA || mainB == candidateB {
            score += 0.35
        }
        if candidate.contains(mainA) {
            score += 0.10
        }
        if candidate.contains(mainB) {
            score += 0.10
        }
        return fxClamp(score, 0.0, 1.0)
    }

    public static func sessionOverlapScore(
        mainSymbol: String,
        candidateSymbol: String,
        sessionBucket: Int,
        hour: Int
    ) -> Double {
        let main = mainSymbol.uppercased()
        let candidate = candidateSymbol.uppercased()
        let category = category(symbol: candidate)
        var score = 0.45

        if category == .fx {
            if sessionBucket == 0, candidate.contains("JPY") || candidate.contains("AUD") || candidate.contains("NZD") {
                score += 0.35
            }
            if sessionBucket == 1, candidate.contains("EUR") || candidate.contains("GBP") || candidate.contains("CHF") {
                score += 0.35
            }
            if sessionBucket == 2, candidate.contains("USD") || candidate.contains("CAD") {
                score += 0.35
            }
        } else if category == .metal || category == .risk {
            if sessionBucket >= 1 {
                score += 0.30
            }
        } else if category == .index || category == .energy {
            if sessionBucket == 2 {
                score += 0.30
            } else if sessionBucket == 1 {
                score += 0.15
            }
        }

        if main.count >= 6 {
            let mainA = String(main.prefix(3))
            let mainB = String(main.dropFirst(3).prefix(3))
            if candidate.contains(mainA) || candidate.contains(mainB) {
                score += 0.10
            }
        }
        if hour >= 21 || hour <= 1 {
            score -= 0.08
        }
        return fxClamp(score, 0.0, 1.0)
    }

    public static func redundancyPenalty(mainSymbol: String, candidateSymbol: String, selected: [String]) -> Double {
        let candidate = candidateSymbol.uppercased()
        let candidateCategory = category(symbol: candidate)
        var penalty = 0.0
        for pickedRaw in selected {
            let picked = pickedRaw.uppercased()
            guard !picked.isEmpty else { continue }
            if picked == candidate {
                return 1.0
            }
            penalty = max(penalty, 0.55 * sharedSymbolScore(mainSymbol: candidate, candidateSymbol: picked))
            if category(symbol: picked) == candidateCategory {
                penalty = max(penalty, 0.18)
            }
        }
        penalty -= 0.10 * sharedSymbolScore(mainSymbol: mainSymbol, candidateSymbol: candidate)
        return fxClamp(penalty, 0.0, 1.0)
    }

    public static func persistenceBonus(candidateSymbol: String, selected: [String]) -> Double {
        let candidate = candidateSymbol.uppercased()
        return selected.contains { $0.uppercased() == candidate } ? 0.12 : 0.0
    }

    public static func incrementalValueScore(mainSymbol: String, candidateSymbol: String, selected: [String]) -> Double {
        let shared = sharedSymbolScore(mainSymbol: mainSymbol, candidateSymbol: candidateSymbol)
        let redundancy = redundancyPenalty(mainSymbol: mainSymbol, candidateSymbol: candidateSymbol, selected: selected)
        return fxClamp(0.70 * shared + 0.30 * (1.0 - redundancy), 0.0, 1.0)
    }

    public static func candidateScore(
        mainSymbol: String,
        candidateSymbol: String,
        selected: [String],
        inputs: LifecycleContextCandidateInputs
    ) -> Double {
        let category = category(symbol: candidateSymbol)
        var score = categoryPriority(category)
        score += 0.40 * incrementalValueScore(mainSymbol: mainSymbol, candidateSymbol: candidateSymbol, selected: selected)
        score += 0.24 * inputs.liquidityScore
        score += 0.18 * inputs.dataHealthScore
        score += 0.14 * inputs.sessionOverlapScore
        score += persistenceBonus(candidateSymbol: candidateSymbol, selected: selected)
        return score
    }

    public static func parseContextSymbols(_ raw: String, maxCount: Int = FXDataEngineConstants.maxContextSymbols) -> [String] {
        let limit = max(0, min(maxCount, FXDataEngineConstants.maxContextSymbols))
        guard limit > 0 else { return [] }
        let clean = raw
            .replacingOccurrences(of: "{", with: "")
            .replacingOccurrences(of: "}", with: "")
            .replacingOccurrences(of: ";", with: ",")
            .replacingOccurrences(of: "|", with: ",")
        var symbols: [String] = []
        symbols.reserveCapacity(limit)
        for part in clean.split(separator: ",", omittingEmptySubsequences: false) {
            let symbol = part.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
            guard !symbol.isEmpty,
                  !symbols.contains(symbol) else {
                continue
            }
            symbols.append(symbol)
            if symbols.count >= limit {
                break
            }
        }
        return symbols
    }

    public static func scoreReference(selected: [String], pending: [String]) -> [String] {
        var reference: [String] = []
        for symbol in selected + pending {
            let normalized = symbol.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
            guard !normalized.isEmpty,
                  !reference.contains(normalized) else {
                continue
            }
            reference.append(normalized)
        }
        return reference
    }

    public static func curatedContextUniverse(mainSymbol: String) -> [String] {
        let main = mainSymbol.uppercased()
        return curatedSymbols.filter { $0 != main }
    }
}
