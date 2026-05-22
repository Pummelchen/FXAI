import Foundation

public struct StackRouterActionCell: Codable, Hashable, Sendable {
    public var value: Double
    public var regret: Double
    public var counterfactual: Double
    public var ready: Bool
    public var observations: Int

    public init(
        value: Double = 0.0,
        regret: Double = 0.0,
        counterfactual: Double = 0.0,
        ready: Bool = false,
        observations: Int = 0
    ) {
        self.value = fxSafeFinite(value)
        self.regret = fxClamp(regret, 0.0, 1.0)
        self.counterfactual = fxClamp(counterfactual, -1.0, 1.0)
        self.ready = ready
        self.observations = min(max(observations, 0), StackerTools.observationCap)
    }
}

public enum StackerTools {
    public static let observationCap = 200_000

    public static func isModelInList(aiID: Int, modelIDs: [Int]) -> Bool {
        modelIDs.contains(aiID)
    }

    public static func stackFeature(_ features: [Double], _ index: Int, default defaultValue: Double = 0.0) -> Double {
        guard index >= 0, index < features.count else { return defaultValue }
        return fxSafeFinite(features[index], fallback: defaultValue)
    }

    public static func stackPortfolioObjective(features: [Double]) -> Double {
        fxClamp(
            0.30 * stackFeature(features, 61) +
            0.28 * stackFeature(features, 62) -
            0.22 * stackFeature(features, 63) +
            0.24 * stackFeature(features, 64) +
            0.10 * stackFeature(features, 70) +
            0.10 * stackFeature(features, 71),
            -1.0,
            1.0
        )
    }

    public static func stackRoutingObjective(features: [Double]) -> Double {
        fxClamp(
            0.22 * stackFeature(features, 56) -
            0.30 * stackFeature(features, 57) +
            0.18 * stackFeature(features, 58) +
            0.20 * stackFeature(features, 59) -
            0.18 * stackFeature(features, 60) +
            0.10 * stackFeature(features, 68) +
            0.16 * stackFeature(features, 69),
            -1.0,
            1.0
        )
    }

    public static func stackRouterContextTrust(features: [Double]) -> Double {
        fxClamp(
            0.18 +
            0.22 * stackFeature(features, 68) +
            0.16 * stackFeature(features, 62) +
            0.12 * stackFeature(features, 70) +
            0.10 * stackFeature(features, 71) -
            0.14 * stackFeature(features, 57) -
            0.10 * stackFeature(features, 63),
            0.0,
            1.0
        )
    }

    public static func stackRouterActionUtility(
        action: LabelClass,
        labelClass: LabelClass,
        realizedEdge: Double,
        qualityScore: Double
    ) -> Double {
        let realizedEdge = fxSafeFinite(realizedEdge)
        let edgeNorm = fxClamp(realizedEdge / max(abs(realizedEdge), 1.0), -1.0, 1.0)
        let quality = fxClamp(qualityScore, 0.0, 2.0)

        if action == .skip {
            if labelClass == .skip {
                return fxClamp(0.28 + 0.22 * quality - 0.12 * edgeNorm, -1.0, 1.0)
            }
            return fxClamp(-0.28 - 0.46 * max(edgeNorm, 0.0) - 0.10 * quality, -1.0, 1.0)
        }

        if action == labelClass {
            return fxClamp(0.32 + 0.52 * edgeNorm + 0.18 * quality, -1.0, 1.0)
        }
        if labelClass == .skip {
            return fxClamp(-0.22 - 0.18 * quality - 0.18 * abs(edgeNorm), -1.0, 1.0)
        }
        return fxClamp(-0.34 - 0.52 * max(edgeNorm, 0.0) - 0.12 * quality, -1.0, 1.0)
    }

    public static func observedRouterCells(
        _ cells: [StackRouterActionCell],
        labelClass: LabelClass,
        realizedEdge: Double,
        qualityScore: Double,
        features: [Double],
        predictedProbabilities: [Double],
        sampleWeight: Double
    ) -> [StackRouterActionCell] {
        let currentCells = normalizedCells(cells)
        let utilities = LabelClass.allCases.map {
            stackRouterActionUtility(
                action: $0,
                labelClass: labelClass,
                realizedEdge: realizedEdge,
                qualityScore: qualityScore
            )
        }
        var baseline = 0.0
        var bestUtility = -Double.greatestFiniteMagnitude
        for label in LabelClass.allCases {
            let index = label.rawValue
            baseline += vectorValue(predictedProbabilities, index, default: 0.3333333) * utilities[index]
            bestUtility = max(bestUtility, utilities[index])
        }

        let trust = fxClamp(
            sampleWeight * (0.40 + 0.60 * stackRouterContextTrust(features: features)),
            0.10,
            6.0
        )
        return LabelClass.allCases.map { label in
            let index = label.rawValue
            let cell = currentCells[index]
            let observations = max(cell.observations, 0)
            let alpha = fxClamp(0.18 / sqrt(1.0 + 0.05 * Double(observations)), 0.02, 0.18)
            let utility = utilities[index]
            let counterfactual = clipSym(utility - baseline, limit: 1.0)
            let regret = fxClamp(bestUtility - utility, 0.0, 1.0)
            if observations <= 0 {
                return StackRouterActionCell(
                    value: utility,
                    regret: regret,
                    counterfactual: counterfactual,
                    ready: true,
                    observations: 1
                )
            }

            let blend = fxClamp(alpha * trust, 0.01, 0.25)
            return StackRouterActionCell(
                value: (1.0 - blend) * cell.value + blend * utility,
                regret: (1.0 - blend) * cell.regret + blend * regret,
                counterfactual: (1.0 - blend) * cell.counterfactual + blend * counterfactual,
                ready: true,
                observations: min(observations + 1, observationCap)
            )
        }
    }

    public static func stackRouterBlend(
        probabilities: [Double],
        features: [Double],
        actionCells: [StackRouterActionCell]
    ) -> [Double] {
        var output = [
            vectorValue(probabilities, LabelClass.sell.rawValue, default: 0.3333333),
            vectorValue(probabilities, LabelClass.buy.rawValue, default: 0.3333333),
            vectorValue(probabilities, LabelClass.skip.rawValue, default: 0.3333334)
        ]
        let contextTrust = stackRouterContextTrust(features: features)
        guard contextTrust > 1e-6 else { return output }

        let cells = normalizedCells(actionCells)
        let routingObjective = stackRoutingObjective(features: features)
        let portfolioObjective = stackPortfolioObjective(features: features)
        let directionBias = fxClamp(stackFeature(features, 6), -1.0, 1.0)
        let correlationPenalty = fxClamp(stackFeature(features, 63), 0.0, 1.0)
        for label in LabelClass.allCases {
            let index = label.rawValue
            let cell = cells[index]
            guard cell.ready, cell.observations > 0 else { continue }

            let observationTrust = fxClamp(Double(cell.observations) / 48.0, 0.0, 1.0)
            var routerScore = 0.70 * cell.value +
                0.35 * cell.counterfactual -
                0.55 * cell.regret
            if label == .buy {
                routerScore += 0.10 * routingObjective * directionBias
            } else if label == .sell {
                routerScore -= 0.10 * routingObjective * directionBias
            } else {
                routerScore += 0.06 * portfolioObjective * correlationPenalty
            }

            let multiplier = exp(clipSym(contextTrust * observationTrust * routerScore, limit: 1.2))
            output[index] = fxClamp(output[index] * multiplier, 0.0005, 0.9990)
        }

        let denominator = max(output.reduce(0.0, +), 1e-12)
        return output.map { $0 / denominator }
    }

    private static func vectorValue(_ values: [Double], _ index: Int, default defaultValue: Double) -> Double {
        guard index >= 0, index < values.count else { return defaultValue }
        return fxSafeFinite(values[index], fallback: defaultValue)
    }

    private static func normalizedCells(_ cells: [StackRouterActionCell]) -> [StackRouterActionCell] {
        var output = Array(repeating: StackRouterActionCell(), count: LabelClass.allCases.count)
        for index in 0..<min(cells.count, output.count) {
            output[index] = cells[index]
        }
        return output
    }

    private static func clipSym(_ value: Double, limit: Double) -> Double {
        let limit = max(fxSafeFinite(limit), 0.0)
        return fxClamp(value, -limit, limit)
    }
}
