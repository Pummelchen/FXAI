import Foundation

public enum FactorCMVPanelReference {
    public struct PanelRow: Equatable, Sendable {
        public let symbol: String
        public let closes: [Double]
        public let volumes: [Double]

        public init(symbol: String, closes: [Double], volumes: [Double] = []) {
            self.symbol = symbol
            self.closes = closes
            self.volumes = volumes
        }
    }

    public struct Exposure: Equatable, Sendable {
        public let symbol: String
        public let momentum: Double
        public let value: Double
        public let volume: Double
        public let composite: Double
    }

    public static func exposures(rows: [PanelRow], dataHasVolume: Bool) -> [Exposure] {
        let momentums = rows.map { momentum($0.closes) }
        let lastPrices = rows.map { $0.closes.last ?? 0.0 }
        let panelMeanPrice = mean(lastPrices)
        let values = lastPrices.map { panelMeanPrice > 0.0 ? -log(max($0, 1.0e-12) / panelMeanPrice) : 0.0 }
        let volumeSignals = rows.map { row -> Double in
            guard dataHasVolume, let last = row.volumes.last, last > 0.0 else { return 0.0 }
            return log(last / max(mean(row.volumes.filter { $0 > 0.0 }), 1.0e-12))
        }
        let mz = zScores(momentums)
        let vz = zScores(values)
        let volz = dataHasVolume ? zScores(volumeSignals) : Array(repeating: 0.0, count: rows.count)
        return rows.indices.map { index in
            Exposure(
                symbol: rows[index].symbol,
                momentum: mz[index],
                value: vz[index],
                volume: volz[index],
                composite: 0.45 * mz[index] + 0.40 * vz[index] + 0.15 * volz[index]
            )
        }
    }

    private static func momentum(_ closes: [Double]) -> Double {
        guard let first = closes.first, let last = closes.last, first > 0.0, last > 0.0 else { return 0.0 }
        return log(last / first)
    }

    private static func zScores(_ values: [Double]) -> [Double] {
        guard values.count > 1 else { return Array(repeating: 0.0, count: values.count) }
        let m = mean(values)
        let sigma = sqrt(max(values.map { pow($0 - m, 2.0) }.reduce(0.0, +) / Double(values.count - 1), 1.0e-12))
        return values.map { ($0 - m) / sigma }
    }

    private static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0.0 : values.reduce(0.0, +) / Double(values.count)
    }
}
