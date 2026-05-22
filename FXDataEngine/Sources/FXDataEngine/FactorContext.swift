import Foundation

public enum FactorContextConstants {
    public static let maxCurrencies = 16
    public static let maxPairs = 32
    public static let commodityProxySymbols = ["XAUUSD", "XTIUSD", "BRENT", "XAGUSD"]
}

public struct FactorDailySeries: Codable, Hashable, Sendable {
    public var symbol: String
    public var closeByShift: [Int: Double]

    public init(symbol: String, closeByShift: [Int: Double]) {
        self.symbol = DataCoreRequest.normalizedSymbol(symbol)
        var clean: [Int: Double] = [:]
        clean.reserveCapacity(closeByShift.count)
        for (shift, close) in closeByShift where shift >= 0 && close.isFinite && close > 0 {
            clean[shift] = close
        }
        self.closeByShift = clean
    }

    public func close(shift: Int) -> Double {
        closeByShift[max(0, shift)] ?? 0.0
    }
}

public struct FactorCarrySnapshot: Codable, Hashable, Sendable {
    public var swapLong: Double
    public var swapShort: Double

    public init(swapLong: Double = 0, swapShort: Double = 0) {
        self.swapLong = fxSafeFinite(swapLong)
        self.swapShort = fxSafeFinite(swapShort)
    }
}

public struct FactorMarketSnapshot: Sendable {
    public var dailySeriesBySymbol: [String: FactorDailySeries]
    public var carryBySymbol: [String: FactorCarrySnapshot]
    public var calendarPairStateBySymbol: [String: CalendarCachePairState]
    public var generatedAt: Int64
    public var allowUSDNeutralFallback: Bool

    public init(
        dailySeriesBySymbol: [String: FactorDailySeries] = [:],
        carryBySymbol: [String: FactorCarrySnapshot] = [:],
        calendarPairStateBySymbol: [String: CalendarCachePairState] = [:],
        generatedAt: Int64 = 0,
        allowUSDNeutralFallback: Bool = true
    ) {
        self.dailySeriesBySymbol = Self.normalizedDailySeries(dailySeriesBySymbol)
        self.carryBySymbol = Self.normalizedCarry(carryBySymbol)
        self.calendarPairStateBySymbol = Self.normalizedCalendarStates(calendarPairStateBySymbol)
        self.generatedAt = max(0, generatedAt)
        self.allowUSDNeutralFallback = allowUSDNeutralFallback
    }

    public func dailySeries(symbol: String) -> FactorDailySeries? {
        dailySeriesBySymbol[DataCoreRequest.normalizedSymbol(symbol)]
    }

    public func carry(symbol: String) -> FactorCarrySnapshot {
        carryBySymbol[DataCoreRequest.normalizedSymbol(symbol)] ?? FactorCarrySnapshot()
    }

    public func calendarPairState(symbol: String) -> CalendarCachePairState? {
        calendarPairStateBySymbol[DataCoreRequest.normalizedSymbol(symbol)]
    }

    private static func normalizedDailySeries(_ input: [String: FactorDailySeries]) -> [String: FactorDailySeries] {
        var output: [String: FactorDailySeries] = [:]
        output.reserveCapacity(input.count)
        for (rawSymbol, series) in input {
            let key = DataCoreRequest.normalizedSymbol(rawSymbol.isEmpty ? series.symbol : rawSymbol)
            guard !key.isEmpty else { continue }
            output[key] = FactorDailySeries(symbol: key, closeByShift: series.closeByShift)
        }
        return output
    }

    private static func normalizedCarry(_ input: [String: FactorCarrySnapshot]) -> [String: FactorCarrySnapshot] {
        var output: [String: FactorCarrySnapshot] = [:]
        output.reserveCapacity(input.count)
        for (rawSymbol, carry) in input {
            let key = DataCoreRequest.normalizedSymbol(rawSymbol)
            guard !key.isEmpty else { continue }
            output[key] = carry
        }
        return output
    }

    private static func normalizedCalendarStates(_ input: [String: CalendarCachePairState]) -> [String: CalendarCachePairState] {
        var output: [String: CalendarCachePairState] = [:]
        output.reserveCapacity(input.count)
        for (rawSymbol, state) in input {
            let key = DataCoreRequest.normalizedSymbol(rawSymbol)
            guard !key.isEmpty else { continue }
            output[key] = state
        }
        return output
    }
}

public struct CurrencyFactorState: Codable, Hashable, Sendable {
    public var currency: String
    public var ready: Bool
    public var trendScore: Double
    public var carryScore: Double
    public var policyScore: Double
    public var valueScore: Double
    public var commodityScore: Double
    public var blendedScore: Double

    public init(
        currency: String = "",
        ready: Bool = false,
        trendScore: Double = 0,
        carryScore: Double = 0,
        policyScore: Double = 0,
        valueScore: Double = 0,
        commodityScore: Double = 0,
        blendedScore: Double = 0
    ) {
        self.currency = MacroEventTools.normalizedCurrencyToken(currency)
        self.ready = ready
        self.trendScore = fxClamp(trendScore, -1.0, 1.0)
        self.carryScore = fxClamp(carryScore, -1.0, 1.0)
        self.policyScore = fxClamp(policyScore, -1.0, 1.0)
        self.valueScore = fxClamp(valueScore, -1.0, 1.0)
        self.commodityScore = fxClamp(commodityScore, -1.0, 1.0)
        self.blendedScore = fxClamp(blendedScore, -1.0, 1.0)
    }

    public static var reset: CurrencyFactorState {
        CurrencyFactorState()
    }
}

public struct PairFactorContext: Codable, Hashable, Sendable {
    public var ready: Bool
    public var stale: Bool
    public var symbol: String
    public var trendScore: Double
    public var carryScore: Double
    public var policyScore: Double
    public var valueScore: Double
    public var commodityScore: Double
    public var blendedScore: Double
    public var biasDirection: Int
    public var alignmentScore: Double
    public var generatedAt: Int64
    public var rationale: String

    public init(
        ready: Bool = false,
        stale: Bool = true,
        symbol: String = "",
        trendScore: Double = 0,
        carryScore: Double = 0,
        policyScore: Double = 0,
        valueScore: Double = 0,
        commodityScore: Double = 0,
        blendedScore: Double = 0,
        biasDirection: Int = -1,
        alignmentScore: Double = 0,
        generatedAt: Int64 = 0,
        rationale: String = ""
    ) {
        self.ready = ready
        self.stale = stale
        self.symbol = DataCoreRequest.normalizedSymbol(symbol)
        self.trendScore = fxClamp(trendScore, -1.0, 1.0)
        self.carryScore = fxClamp(carryScore, -1.0, 1.0)
        self.policyScore = fxClamp(policyScore, -1.0, 1.0)
        self.valueScore = fxClamp(valueScore, -1.0, 1.0)
        self.commodityScore = fxClamp(commodityScore, -1.0, 1.0)
        self.blendedScore = fxClamp(blendedScore, -1.0, 1.0)
        self.biasDirection = [-1, 0, 1].contains(biasDirection) ? biasDirection : -1
        self.alignmentScore = fxClamp(alignmentScore, 0.0, 1.0)
        self.generatedAt = max(0, generatedAt)
        self.rationale = rationale
    }

    public static func reset(symbol: String = "") -> PairFactorContext {
        PairFactorContext(symbol: symbol)
    }
}

public enum FactorContextTools {
    public static func d1Return(series: FactorDailySeries?, shiftNow: Int, shiftThen: Int) -> Double {
        guard let series else { return 0.0 }
        let nowClose = series.close(shift: shiftNow)
        let thenClose = series.close(shift: shiftThen)
        guard nowClose > 0, thenClose > 0 else { return 0.0 }
        return (nowClose / thenClose) - 1.0
    }

    public static func trendScore(series: FactorDailySeries?) -> Double {
        let r21 = d1Return(series: series, shiftNow: 1, shiftThen: 21)
        let r63 = d1Return(series: series, shiftNow: 1, shiftThen: 63)
        let r126 = d1Return(series: series, shiftNow: 1, shiftThen: 126)
        let blended = 0.20 * r21 + 0.35 * r63 + 0.45 * r126
        return fxClamp(blended * 8.0, -1.0, 1.0)
    }

    public static func valueScore(series: FactorDailySeries?) -> Double {
        guard let series else { return 0.0 }
        let closeNow = series.close(shift: 1)
        let close63 = series.close(shift: 63)
        let close252 = series.close(shift: 252)
        guard closeNow > 0, close63 > 0, close252 > 0 else { return 0.0 }
        let mediumAnchor = 0.5 * (close63 + close252)
        guard mediumAnchor > 0 else { return 0.0 }
        let gap = (closeNow - mediumAnchor) / mediumAnchor
        return fxClamp(-gap * 6.0, -1.0, 1.0)
    }

    public static func carryDirectional(_ carry: FactorCarrySnapshot, direction: Int) -> Double {
        let scale = max(max(abs(carry.swapLong), abs(carry.swapShort)), 0.50)
        if direction == 1 {
            return fxClamp(carry.swapLong / scale, -1.0, 1.0)
        }
        if direction == 0 {
            return fxClamp(carry.swapShort / scale, -1.0, 1.0)
        }
        return fxClamp((carry.swapLong - carry.swapShort) / scale, -1.0, 1.0)
    }

    public static func policyPressure(currency rawCurrency: String, market: FactorMarketSnapshot) -> Double {
        let currency = MacroEventTools.normalizedCurrencyToken(rawCurrency)
        guard currency.count == 3 else { return 0.0 }
        let synthetic = currency == "USD" ? "EURUSD" : "\(currency)USD"
        guard let calendarState = market.calendarPairState(symbol: synthetic), calendarState.ready else {
            return 0.0
        }

        var score = 0.0
        let hasCentralBank = calendarState.reasons.contains { $0.contains("central_bank") }
        let hasInflation = calendarState.reasons.contains { $0.contains("inflation") }
        if calendarState.tradeGate == .block {
            score += 0.20
        }
        if hasCentralBank {
            score += 0.35
        }
        if hasInflation {
            score += 0.18
        }
        if calendarState.nextEventETAMinutes >= -120, calendarState.nextEventETAMinutes <= 240 {
            score += 0.15
        }
        return fxClamp(score, -1.0, 1.0)
    }

    public static func commodityScore(currency rawCurrency: String, market: FactorMarketSnapshot) -> Double {
        let currency = MacroEventTools.normalizedCurrencyToken(rawCurrency)
        var commodityMove = 0.0
        var used = 0
        for proxy in FactorContextConstants.commodityProxySymbols {
            guard let series = market.dailySeries(symbol: proxy) else { continue }
            let r20 = d1Return(series: series, shiftNow: 1, shiftThen: 20)
            let r60 = d1Return(series: series, shiftNow: 1, shiftThen: 60)
            commodityMove += 0.45 * r20 + 0.55 * r60
            used += 1
        }
        guard used > 0 else { return 0.0 }
        commodityMove /= Double(used)

        if currency == "AUD" || currency == "NZD" || currency == "CAD" || currency == "NOK" {
            return fxClamp(commodityMove * 6.0, -1.0, 1.0)
        }
        if currency == "CHF" || currency == "JPY" {
            return fxClamp(-commodityMove * 4.0, -1.0, 1.0)
        }
        return 0.0
    }

    public static func buildCurrencyState(currency rawCurrency: String, market: FactorMarketSnapshot) -> CurrencyFactorState {
        let currency = MacroEventTools.normalizedCurrencyToken(rawCurrency)
        guard currency.count == 3 else { return .reset }

        if currency == "USD", market.allowUSDNeutralFallback {
            let policy = policyPressure(currency: currency, market: market)
            let blended = fxClamp(0.22 * policy, -1.0, 1.0)
            return CurrencyFactorState(
                currency: currency,
                ready: true,
                policyScore: policy,
                blendedScore: blended
            )
        }

        let anchorUSD = "\(currency)USD"
        let anchorReverse = "USD\(currency)"
        let selected: (symbol: String, inverted: Bool, series: FactorDailySeries)?
        if let series = market.dailySeries(symbol: anchorUSD), series.close(shift: 1) > 0 {
            selected = (anchorUSD, false, series)
        } else if let series = market.dailySeries(symbol: anchorReverse), series.close(shift: 1) > 0 {
            selected = (anchorReverse, true, series)
        } else {
            return CurrencyFactorState(currency: currency)
        }
        guard let selected else { return CurrencyFactorState(currency: currency) }

        let carry = market.carry(symbol: selected.symbol)
        var trend = trendScore(series: selected.series)
        var value = valueScore(series: selected.series)
        var policy = policyPressure(currency: currency, market: market)
        var commodity = commodityScore(currency: currency, market: market)
        var carryScore = carryDirectional(carry, direction: selected.inverted ? 0 : 1)

        if selected.inverted {
            trend *= -1.0
            value *= -1.0
            policy *= -1.0
            commodity *= -1.0
            carryScore *= -1.0
        }

        let blended = fxClamp(
            0.32 * trend
                + 0.22 * carryScore
                + 0.22 * policy
                + 0.18 * value
                + 0.06 * commodity,
            -1.0,
            1.0
        )
        return CurrencyFactorState(
            currency: currency,
            ready: true,
            trendScore: trend,
            carryScore: carryScore,
            policyScore: policy,
            valueScore: value,
            commodityScore: commodity,
            blendedScore: blended
        )
    }

    public static func buildPairContext(symbol rawSymbol: String, market: FactorMarketSnapshot) -> PairFactorContext {
        let symbol = DataCoreRequest.normalizedSymbol(rawSymbol)
        let (base, quote) = CalendarCacheTools.symbolLegs(symbol)
        guard base.count == 3, quote.count == 3 else {
            return .reset(symbol: symbol)
        }

        let baseState = buildCurrencyState(currency: base, market: market)
        let quoteState = buildCurrencyState(currency: quote, market: market)
        guard baseState.ready, quoteState.ready else {
            return .reset(symbol: symbol)
        }

        let trend = fxClamp(baseState.trendScore - quoteState.trendScore, -1.0, 1.0)
        let carry = fxClamp(baseState.carryScore - quoteState.carryScore, -1.0, 1.0)
        let policy = fxClamp(baseState.policyScore - quoteState.policyScore, -1.0, 1.0)
        let value = fxClamp(baseState.valueScore - quoteState.valueScore, -1.0, 1.0)
        let commodity = fxClamp(baseState.commodityScore - quoteState.commodityScore, -1.0, 1.0)
        let blended = fxClamp(baseState.blendedScore - quoteState.blendedScore, -1.0, 1.0)
        let biasDirection = blended > 0.08 ? 1 : (blended < -0.08 ? 0 : -1)
        let alignment = fxClamp(0.50 + 0.50 * abs(blended), 0.0, 1.0)
        let rationale = String(
            format: "trend=%.2f carry=%.2f policy=%.2f value=%.2f commodity=%.2f",
            locale: Locale(identifier: "en_US_POSIX"),
            trend,
            carry,
            policy,
            value,
            commodity
        )
        return PairFactorContext(
            ready: true,
            stale: false,
            symbol: symbol,
            trendScore: trend,
            carryScore: carry,
            policyScore: policy,
            valueScore: value,
            commodityScore: commodity,
            blendedScore: blended,
            biasDirection: biasDirection,
            alignmentScore: alignment,
            generatedAt: market.generatedAt,
            rationale: rationale
        )
    }
}
