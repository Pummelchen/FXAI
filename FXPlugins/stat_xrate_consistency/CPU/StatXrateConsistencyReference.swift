import Foundation

public enum StatXrateConsistencyReference {
    public struct Quote: Equatable, Sendable {
        public let base: String
        public let quote: String
        public let rate: Double

        public init(base: String, quote: String, rate: Double) {
            self.base = base.uppercased()
            self.quote = quote.uppercased()
            self.rate = rate
        }
    }

    public struct CycleScore: Equatable, Sendable {
        public let currencies: [String]
        public let logImbalance: Double
        public let impliedRate: Double
        public let quotedRate: Double
        public let basisPoints: Double
    }

    public static func triangularScores(quotes: [Quote]) -> [CycleScore] {
        let graph = logRateGraph(quotes: quotes)
        let currencies = Array(Set(quotes.flatMap { [$0.base, $0.quote] })).sorted()
        var scores: [CycleScore] = []
        for a in currencies {
            for b in currencies where b > a {
                for c in currencies where c > b {
                    for cycle in [[a, b, c], [a, c, b]] {
                        guard
                            let ab = graph[edge(cycle[0], cycle[1])],
                            let bc = graph[edge(cycle[1], cycle[2])],
                            let ca = graph[edge(cycle[2], cycle[0])]
                        else { continue }
                        let imbalance = ab + bc + ca
                        let implied = exp(ab + bc)
                        let quoted = exp(-ca)
                        scores.append(CycleScore(
                            currencies: cycle,
                            logImbalance: imbalance,
                            impliedRate: implied,
                            quotedRate: quoted,
                            basisPoints: (exp(imbalance) - 1.0) * 10_000.0
                        ))
                    }
                }
            }
        }
        return scores.sorted { abs($0.basisPoints) > abs($1.basisPoints) }
    }

    public static func scoreCycle(_ currencies: [String], quotes: [Quote]) -> CycleScore? {
        guard currencies.count >= 3 else { return nil }
        let normalized = currencies.map { $0.uppercased() }
        let graph = logRateGraph(quotes: quotes)
        var imbalance = 0.0
        for index in 0..<normalized.count {
            let from = normalized[index]
            let to = normalized[(index + 1) % normalized.count]
            guard let rate = graph[edge(from, to)] else { return nil }
            imbalance += rate
        }
        return CycleScore(
            currencies: normalized,
            logImbalance: imbalance,
            impliedRate: exp(imbalance),
            quotedRate: 1.0,
            basisPoints: (exp(imbalance) - 1.0) * 10_000.0
        )
    }

    public static func logRateGraph(quotes: [Quote]) -> [String: Double] {
        var graph: [String: Double] = [:]
        for quote in quotes where quote.rate > 0.0 && quote.rate.isFinite {
            let logRate = log(quote.rate)
            graph[edge(quote.base, quote.quote)] = logRate
            graph[edge(quote.quote, quote.base)] = -logRate
        }
        return graph
    }

    private static func edge(_ from: String, _ to: String) -> String {
        "\(from.uppercased())/\(to.uppercased())"
    }
}
