import Foundation

public struct MetaCalibrationPortfolioDiagnostics: Codable, Hashable, Sendable {
    public var ready: Bool
    public var meanEdgePoints: Double
    public var stability: Double
    public var correlationPenalty: Double
    public var diversification: Double
    public var objective: Double
    public var symbolCount: Int

    public init(
        meanEdgePoints: Double = 0.0,
        stability: Double = 0.0,
        correlationPenalty: Double = 0.0,
        diversification: Double = 0.0,
        objective: Double? = nil,
        symbolCount: Int = 0
    ) {
        let resolvedSymbolCount = max(symbolCount, 0)
        self.ready = resolvedSymbolCount > 0
        self.meanEdgePoints = fxSafeFinite(meanEdgePoints)
        self.stability = fxClamp(stability, 0.0, 1.0)
        self.correlationPenalty = fxClamp(correlationPenalty, 0.0, 1.0)
        self.diversification = fxClamp(diversification, 0.0, 1.0)
        self.symbolCount = resolvedSymbolCount
        self.objective = fxClamp(
            objective ?? MetaCalibrationTools.computePortfolioObjective(
                meanEdgePoints: meanEdgePoints,
                stability: stability,
                correlationPenalty: correlationPenalty,
                diversification: diversification,
                symbolCount: symbolCount
            ),
            -1.0,
            1.0
        )
    }
}

public struct MetaCalibrationPluginRouteCell: Codable, Hashable, Sendable {
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
        self.observations = min(max(observations, 0), MetaCalibrationTools.observationCap)
    }
}

public enum MetaCalibrationTools {
    public static let observationCap = 200_000

    public static func computePortfolioObjective(
        meanEdgePoints: Double,
        stability: Double,
        correlationPenalty: Double,
        diversification: Double,
        symbolCount: Int
    ) -> Double {
        let meanEdge = fxSafeFinite(meanEdgePoints)
        let scale = max(abs(meanEdge), 0.50)
        let edgeNorm = fxClamp(meanEdge / scale, -4.0, 4.0) / 4.0
        let symbolCoverage = fxClamp(Double(max(symbolCount, 0)) / 6.0, 0.0, 1.0)
        return fxClamp(
            0.34 * edgeNorm +
            0.26 * (fxClamp(stability, 0.0, 1.0) - 0.50) -
            0.24 * fxClamp(correlationPenalty, 0.0, 1.0) +
            0.24 * (fxClamp(diversification, 0.0, 1.0) - 0.50) +
            0.18 * (symbolCoverage - 0.50),
            -1.0,
            1.0
        )
    }

    public static func portfolioEdgeNorm(
        _ diagnostics: MetaCalibrationPortfolioDiagnostics,
        minMovePoints: Double
    ) -> Double {
        guard diagnostics.ready else { return 0.0 }
        let minMove = max(fxSafeFinite(minMovePoints), 0.50)
        return fxClamp(diagnostics.meanEdgePoints / minMove, -4.0, 4.0) / 4.0
    }

    public static func portfolioFactor(
        _ diagnostics: MetaCalibrationPortfolioDiagnostics,
        minMovePoints: Double
    ) -> Double {
        guard diagnostics.ready else { return 1.0 }
        let edgeNorm = portfolioEdgeNorm(diagnostics, minMovePoints: minMovePoints)
        let objective = fxClamp(0.55 * diagnostics.objective + 0.45 * edgeNorm, -1.0, 1.0)
        return fxClamp(1.0 + 0.75 * objective, 0.45, 1.85)
    }

    public static func actionLabel(signal: Int) -> LabelClass {
        signal == 1 ? .buy : (signal == 0 ? .sell : .skip)
    }

    public static func pluginRouteActionUtility(
        action: LabelClass,
        labelClass: LabelClass,
        realizedRatio: Double
    ) -> Double {
        let ratio = fxClamp(realizedRatio, -6.0, 6.0)
        let opportunity = fxClamp(abs(ratio), 0.0, 6.0)

        switch action {
        case .buy:
            if labelClass == .buy {
                return fxClamp(0.24 + 0.60 * ratio, -1.0, 1.0)
            }
            if labelClass == .skip {
                return fxClamp(-0.22 - 0.15 * opportunity, -1.0, 1.0)
            }
            return fxClamp(-0.30 - 0.50 * opportunity, -1.0, 1.0)
        case .sell:
            if labelClass == .sell {
                return fxClamp(0.24 + 0.60 * ratio, -1.0, 1.0)
            }
            if labelClass == .skip {
                return fxClamp(-0.22 - 0.15 * opportunity, -1.0, 1.0)
            }
            return fxClamp(-0.30 - 0.50 * opportunity, -1.0, 1.0)
        case .skip:
            if labelClass == .skip {
                return fxClamp(0.20 + 0.12 * (1.0 - fxClamp(opportunity / 4.0, 0.0, 1.0)), -1.0, 1.0)
            }
            return fxClamp(-0.18 - 0.24 * opportunity, -1.0, 1.0)
        }
    }

    public static func updatedPluginRouteCell(
        _ cell: MetaCalibrationPluginRouteCell,
        labelClass: LabelClass,
        signal: Int,
        realizedNetPoints: Double,
        minMovePoints: Double,
        predictedEdgePoints: Double,
        sampleWeight: Double
    ) -> MetaCalibrationPluginRouteCell {
        let minMove = max(fxSafeFinite(minMovePoints), 0.10)
        let realizedRatio = fxClamp(fxSafeFinite(realizedNetPoints) / minMove, -6.0, 6.0)
        let action = actionLabel(signal: signal)

        var bestUtility = -Double.greatestFiniteMagnitude
        var actionUtility = 0.0
        for candidate in LabelClass.allCases {
            let utility = pluginRouteActionUtility(
                action: candidate,
                labelClass: labelClass,
                realizedRatio: realizedRatio
            )
            if candidate == action {
                actionUtility = utility
            }
            bestUtility = max(bestUtility, utility)
        }

        let baseline = fxClamp(fxSafeFinite(predictedEdgePoints) / minMove, -1.0, 1.0)
        let counterfactual = clipSym(bestUtility - baseline, limit: 1.0)
        let regret = fxClamp(bestUtility - actionUtility, 0.0, 1.0)
        let observations = max(cell.observations, 0)
        let alpha = fxClamp(0.18 / sqrt(1.0 + 0.05 * Double(observations)), 0.02, 0.18)
        let blend = fxClamp(alpha * fxClamp(sampleWeight, 0.20, 6.0), 0.01, 0.24)

        if observations <= 0 {
            return MetaCalibrationPluginRouteCell(
                value: actionUtility,
                regret: regret,
                counterfactual: counterfactual,
                ready: true,
                observations: 1
            )
        }

        return MetaCalibrationPluginRouteCell(
            value: (1.0 - blend) * cell.value + blend * actionUtility,
            regret: (1.0 - blend) * cell.regret + blend * regret,
            counterfactual: (1.0 - blend) * cell.counterfactual + blend * counterfactual,
            ready: true,
            observations: min(observations + 1, observationCap)
        )
    }

    public static func contextTrust(observations: Int, denominator: Double = 64.0) -> Double {
        fxClamp(Double(max(observations, 0)) / max(denominator, 1.0), 0.0, 1.0)
    }

    public static func pluginRouteFactor(
        _ cell: MetaCalibrationPluginRouteCell,
        contextTrust: Double,
        portfolioObjective: Double
    ) -> Double {
        guard cell.ready else { return 1.0 }
        let routeTrust = Self.contextTrust(observations: cell.observations, denominator: 64.0)
        let contextTrustValue = fxClamp(contextTrust, 0.0, 1.0)
        let score = 0.72 * cell.value +
            0.38 * cell.counterfactual -
            0.58 * cell.regret +
            0.18 * fxClamp(portfolioObjective, -1.0, 1.0)
        let scaled = (0.35 + 0.65 * contextTrustValue) * routeTrust * score
        return fxClamp(exp(clipSym(scaled, limit: 1.1)), 0.55, 1.90)
    }

    public static func isModelPruned(
        reliability: Double,
        regimeObservations: Int = 0,
        regimeEdgePoints: Double = 0.0,
        globalEdgeReady: Bool = false,
        globalEdgePoints: Double = 0.0
    ) -> Bool {
        let reliability = fxClamp(reliability, 0.0, 3.0)
        if reliability < 0.30 {
            return true
        }
        if regimeObservations >= 24, fxSafeFinite(regimeEdgePoints) < -0.35 {
            return true
        }
        if globalEdgeReady, fxSafeFinite(globalEdgePoints) < -0.45 {
            return true
        }
        return false
    }

    public static func modelMetaScore(
        reliability: Double,
        metaWeight: Double,
        regimeEdgePoints: Double,
        contextEdgePoints: Double,
        contextRegret: Double,
        contextObservations: Int,
        portfolioDiagnostics: MetaCalibrationPortfolioDiagnostics,
        routeCell: MetaCalibrationPluginRouteCell,
        minMovePoints: Double
    ) -> Double {
        let minMove = max(fxSafeFinite(minMovePoints), 0.50)
        let reliability = fxClamp(reliability, 0.20, 3.00)
        let metaWeight = fxClamp(metaWeight, 0.20, 3.00)

        var edgeScale = 1.0 + fxClamp(fxSafeFinite(regimeEdgePoints) / minMove, -0.70, 1.20)
        if edgeScale < 0.15 {
            edgeScale = 0.15
        }

        let scoreContextTrust = contextTrust(observations: contextObservations, denominator: 48.0)
        let contextScore = fxClamp(
            0.55 * (fxSafeFinite(contextEdgePoints) / minMove) -
            0.40 * fxSafeFinite(contextRegret),
            -0.80,
            1.20
        )
        let contextFactor = fxClamp(1.0 + scoreContextTrust * contextScore, 0.25, 2.20)
        let portfolioFactor = portfolioFactor(portfolioDiagnostics, minMovePoints: minMove)
        let routeFactor = pluginRouteFactor(
            routeCell,
            contextTrust: contextTrust(observations: contextObservations, denominator: 64.0),
            portfolioObjective: portfolioDiagnostics.ready ? portfolioDiagnostics.objective : 0.0
        )

        return reliability * metaWeight * edgeScale * contextFactor * portfolioFactor * routeFactor
    }

    private static func clipSym(_ value: Double, limit: Double) -> Double {
        let limit = max(fxSafeFinite(limit), 0.0)
        return fxClamp(value, -limit, limit)
    }
}
