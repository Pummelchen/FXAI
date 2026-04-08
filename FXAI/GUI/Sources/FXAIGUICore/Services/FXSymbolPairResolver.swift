import Foundation

public enum FXSymbolPairResolver {
    private static let supportedCurrencies: Set<String> = [
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"
    ]

    public static func pairID(from rawSymbol: String, preferredPairs: [String] = []) -> String? {
        let upperSymbol = rawSymbol.uppercased()
        let preferred = Set(preferredPairs.map { $0.uppercased() }.filter { $0.count == 6 })
        let alphaOnly = upperSymbol.filter(\.isLetter)
        guard alphaOnly.count >= 6 else { return nil }
        let text = String(alphaOnly)

        for offset in 0...(text.count - 6) {
            let start = text.index(text.startIndex, offsetBy: offset)
            let end = text.index(start, offsetBy: 6)
            let candidate = String(text[start..<end])
            guard isSupportedPair(candidate) else { continue }
            if preferred.isEmpty || preferred.contains(candidate) {
                return candidate
            }
        }
        return nil
    }

    private static func isSupportedPair(_ pair: String) -> Bool {
        guard pair.count == 6 else { return false }
        let base = String(pair.prefix(3))
        let quote = String(pair.suffix(3))
        return base != quote && supportedCurrencies.contains(base) && supportedCurrencies.contains(quote)
    }
}
