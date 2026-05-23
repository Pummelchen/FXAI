import Foundation

public struct ExposureSymbolLegs: Codable, Hashable, Sendable {
    public var base: String
    public var quote: String

    public init(base: String = "", quote: String = "") {
        self.base = base.uppercased()
        self.quote = quote.uppercased()
    }

    public var isValidPair: Bool {
        base.count == 3 && quote.count == 3
    }
}

public enum ExposureTools {
    public static func parseSymbolLegs(
        _ symbol: String,
        baseHint: String = "",
        quoteHint: String = ""
    ) -> ExposureSymbolLegs {
        let base = baseHint.uppercased()
        let quote = quoteHint.uppercased()
        if base.count == 3, quote.count == 3 {
            return ExposureSymbolLegs(base: base, quote: quote)
        }
        let legs = CalendarCacheTools.symbolLegs(symbol)
        return ExposureSymbolLegs(base: legs.base, quote: legs.quote)
    }

    public static func symbolsShareCurrency(_ lhs: String, _ rhs: String) -> Bool {
        let lhsLegs = parseSymbolLegs(lhs)
        let rhsLegs = parseSymbolLegs(rhs)
        guard lhsLegs.isValidPair, rhsLegs.isValidPair else { return false }
        return lhsLegs.base == rhsLegs.base ||
            lhsLegs.base == rhsLegs.quote ||
            lhsLegs.quote == rhsLegs.base ||
            lhsLegs.quote == rhsLegs.quote
    }

    public static func correlationExposureWeight(anchorSymbol: String, otherSymbol: String) -> Double {
        if anchorSymbol == otherSymbol {
            return 1.0
        }

        let anchor = parseSymbolLegs(anchorSymbol)
        let other = parseSymbolLegs(otherSymbol)
        guard anchor.isValidPair, other.isValidPair else { return 0.0 }

        if anchor.base == other.quote, anchor.quote == other.base {
            return 1.0
        }
        if anchor.base == other.base, anchor.quote == other.quote {
            return 1.0
        }
        if anchor.base == other.base || anchor.quote == other.quote {
            return 0.85
        }
        if anchor.base == other.quote || anchor.quote == other.base {
            return 0.70
        }
        if symbolsShareCurrency(anchorSymbol, otherSymbol) {
            return 0.55
        }
        return 0.0
    }

    public static func runtimeBaseCurrency(_ rawSymbol: String) -> String {
        let symbol = rawSymbol.uppercased()
        guard symbol.count >= 6 else { return "" }
        return String(symbol.prefix(3))
    }

    public static func runtimeQuoteCurrency(_ rawSymbol: String) -> String {
        let symbol = rawSymbol.uppercased()
        guard symbol.count >= 6 else { return "" }
        return String(symbol.dropFirst(3).prefix(3))
    }

    public static func directionalExposureSign(
        symbol: String,
        direction: Int,
        currency rawCurrency: String
    ) -> Int {
        let currency = rawCurrency.uppercased()
        let base = runtimeBaseCurrency(symbol)
        let quote = runtimeQuoteCurrency(symbol)
        guard direction == 0 || direction == 1 else { return 0 }
        let directionSign = direction == 1 ? 1 : -1
        if currency == base {
            return directionSign
        }
        if currency == quote {
            return -directionSign
        }
        return 0
    }

    public static func directionalClusterAlignment(
        anchorSymbol: String,
        anchorDirection: Int,
        otherSymbol: String,
        otherDirection: Int
    ) -> Double {
        let anchorBase = runtimeBaseCurrency(anchorSymbol)
        let anchorQuote = runtimeQuoteCurrency(anchorSymbol)
        let otherBase = runtimeBaseCurrency(otherSymbol)
        let otherQuote = runtimeQuoteCurrency(otherSymbol)
        guard anchorBase.count == 3,
              anchorQuote.count == 3,
              otherBase.count == 3,
              otherQuote.count == 3 else {
            return 0.0
        }

        var alignment = 0.0
        for currency in [anchorBase, anchorQuote] {
            let anchorSign = directionalExposureSign(symbol: anchorSymbol, direction: anchorDirection, currency: currency)
            let otherSign = directionalExposureSign(symbol: otherSymbol, direction: otherDirection, currency: currency)
            if anchorSign == 0 || otherSign == 0 {
                continue
            }
            if anchorSign == otherSign {
                alignment += 0.50
            } else {
                alignment -= 0.25
            }
        }

        if anchorSymbol == otherSymbol, anchorDirection == otherDirection {
            alignment = 1.0
        } else if anchorBase == otherQuote, anchorQuote == otherBase, anchorDirection != otherDirection {
            alignment = max(alignment, 0.95)
        }
        return fxClamp(alignment, 0.0, 1.0)
    }
}
