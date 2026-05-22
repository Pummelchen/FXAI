import Foundation

public struct OOFHorizonPriorCell: Codable, Hashable, Sendable {
    public var scoreEMA: Double
    public var edgeEMA: Double
    public var qualityEMA: Double
    public var tradeRateEMA: Double
    public var ready: Bool
    public var observations: Int

    public init(
        scoreEMA: Double = 0.0,
        edgeEMA: Double = 0.0,
        qualityEMA: Double = 0.0,
        tradeRateEMA: Double = 0.0,
        ready: Bool = false,
        observations: Int = 0
    ) {
        self.scoreEMA = fxSafeFinite(scoreEMA)
        self.edgeEMA = fxSafeFinite(edgeEMA)
        self.qualityEMA = fxClamp(qualityEMA, 0.0, 1.0)
        self.tradeRateEMA = fxClamp(tradeRateEMA, 0.0, 1.0)
        self.ready = ready
        self.observations = min(max(observations, 0), HorizonPolicyTools.observationCap)
    }
}

public enum HorizonPolicyTools {
    public static let observationCap = 200_000

    public static func updatedOOFPriorCell(
        _ cell: OOFHorizonPriorCell,
        scoreProxy: Double,
        edgeRatio: Double,
        quality: Double,
        tradeTarget: Bool
    ) -> OOFHorizonPriorCell {
        let score = fxClamp(scoreProxy, -4.0, 8.0)
        let edge = fxClamp(edgeRatio, -2.0, 4.0)
        let quality = fxClamp(quality, 0.0, 1.0)
        let tradeRate = tradeTarget ? 1.0 : 0.0

        guard cell.ready else {
            return OOFHorizonPriorCell(
                scoreEMA: score,
                edgeEMA: edge,
                qualityEMA: quality,
                tradeRateEMA: tradeRate,
                ready: true,
                observations: 1
            )
        }

        let observations = max(cell.observations, 0)
        let alpha = fxClamp(0.18 / sqrt(1.0 + 0.03 * Double(observations)), 0.025, 0.12)
        return OOFHorizonPriorCell(
            scoreEMA: (1.0 - alpha) * cell.scoreEMA + alpha * score,
            edgeEMA: (1.0 - alpha) * cell.edgeEMA + alpha * edge,
            qualityEMA: (1.0 - alpha) * cell.qualityEMA + alpha * quality,
            tradeRateEMA: (1.0 - alpha) * cell.tradeRateEMA + alpha * tradeRate,
            ready: true,
            observations: min(observations + 1, observationCap)
        )
    }

    public static func oofHorizonPriorScore(_ cell: OOFHorizonPriorCell) -> Double {
        guard cell.ready else { return 0.0 }
        let trust = fxClamp(Double(cell.observations) / 48.0, 0.05, 0.35)
        let prior = 0.28 * cell.scoreEMA +
            0.22 * cell.edgeEMA +
            0.28 * (2.0 * cell.qualityEMA - 1.0) +
            0.22 * (2.0 * cell.tradeRateEMA - 1.0)
        return trust * fxClamp(prior, -3.0, 3.0)
    }

    public static func oofTradeGatePrior(_ cell: OOFHorizonPriorCell) -> Double {
        guard cell.ready else { return -1.0 }
        let trust = fxClamp(Double(cell.observations) / 64.0, 0.10, 0.45)
        let prior = 0.18 +
            0.42 * cell.tradeRateEMA +
            0.20 * cell.qualityEMA +
            0.12 * fxClamp(cell.edgeEMA, 0.0, 2.0) / 2.0 +
            0.08 * fxClamp(cell.scoreEMA, 0.0, 4.0) / 4.0
        return fxClamp((1.0 - trust) * 0.50 + trust * prior, 0.01, 0.99)
    }
}
