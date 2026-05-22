import Foundation

public enum PluginPathQualityTools {
    public static func populatedOutput(
        _ output: PluginModelOutputV4,
        x: [Double],
        window: [[Double]],
        context: PluginContextV4,
        family: AIFamily,
        activityGate: Double,
        structuralQuality: Double,
        executionQuality: Double? = nil,
        qualityPriors: PluginQualityBankPriors = PluginQualityBankPriors(),
        declaredWindowSize: Int? = nil
    ) -> PluginModelOutputV4 {
        var next = output
        let priors = qualityPriors
        let active = fxClamp(activityGate, 0.0, 1.0)
        let structure = fxClamp(structuralQuality, 0.0, 1.0)
        let execution = executionQuality.map { fxClamp($0, 0.0, 1.0) } ?? structure
        let moveMean = max(0.0, fxSafeFinite(next.moveMeanPoints))
        let moveQ25 = max(0.0, fxSafeFinite(next.moveQ25Points))
        let moveQ50 = max(0.0, fxSafeFinite(next.moveQ50Points))
        let moveQ75 = max(0.0, fxSafeFinite(next.moveQ75Points))
        let moveScale = max(moveMean, moveQ50, max(context.minMovePoints, 0.10))
        let quantileSpan = max(0.0, moveQ75 - moveQ25)
        let sigma = max(0.10, 0.30 * moveScale + 0.45 * quantileSpan)
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution(next.classProbabilities)
        let directional = fxClamp(max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]), 0.0, 1.0)
        let skip = fxClamp(probabilities[LabelClass.skip.rawValue], 0.0, 1.0)
        let priceCost = PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: context.priceCostPoints)
        let costRatio = fxClamp(priceCost / max(moveScale + 0.40 * sigma, 0.25), 0.0, 1.0)

        let shape = windowShape(window, declaredWindowSize: declaredWindowSize)
        let multipliers = familyMultipliers(family)

        var mfeScale = 1.05 + 0.30 * directional + 0.18 * active + 0.16 * structure +
            0.16 * shape.trend * multipliers.trend + 0.10 * shape.fastTrend +
            0.08 * shape.contextShape * multipliers.context
        var maeScale = 0.14 + 0.24 * (1.0 - active) + 0.18 * (1.0 - structure) +
            0.16 * costRatio * multipliers.execution + 0.12 * shape.noise + 0.08 * skip

        if priors.trust > 0.0 {
            let qualityBase = max(moveScale, 0.10)
            let bankMFEScale = fxClamp(priors.mfePoints / qualityBase, 0.80, 3.40)
            let bankMAEScale = fxClamp(priors.maePoints / max(max(priors.mfePoints, qualityBase), 0.10), 0.05, 1.70)
            mfeScale = (1.0 - 0.55 * priors.trust) * mfeScale + 0.55 * priors.trust * bankMFEScale
            maeScale = (1.0 - 0.55 * priors.trust) * maeScale + 0.55 * priors.trust * bankMAEScale
        }

        next.mfeMeanPoints = max(moveQ75, moveScale * fxClamp(mfeScale, 0.80, 3.50))
        next.maeMeanPoints = max(0.0, moveScale * fxClamp(maeScale, 0.05, 1.80))

        var hitFraction = 0.70 - 0.20 * active - 0.12 * structure - 0.08 * shape.fastTrend -
            0.06 * shape.contextShape + 0.18 * shape.noise + 0.16 * costRatio + 0.10 * skip
        if priors.trust > 0.0 {
            hitFraction = (1.0 - 0.60 * priors.trust) * hitFraction + 0.60 * priors.trust * priors.hitTimeFraction
        }
        next.hitTimeFraction = fxClamp(hitFraction, 0.0, 1.0)

        var pathRisk = 0.34 * fxClamp(next.maeMeanPoints / max(next.mfeMeanPoints, moveScale), 0.0, 1.0) +
            0.22 * next.hitTimeFraction +
            0.18 * costRatio +
            0.14 * (1.0 - structure) +
            0.12 * shape.noise +
            0.08 * (1.0 - execution)
        if priors.trust > 0.0 {
            pathRisk = (1.0 - 0.60 * priors.trust) * pathRisk + 0.60 * priors.trust * priors.pathRisk
        }
        next.pathRisk = fxClamp(pathRisk, 0.0, 1.0)

        var fillRisk = 0.46 * costRatio + 0.26 * (1.0 - execution) + 0.16 * skip + 0.12 * shape.noise
        if priors.trust > 0.0 {
            fillRisk = (1.0 - 0.60 * priors.trust) * fillRisk + 0.60 * priors.trust * priors.fillRisk
        }
        next.fillRisk = fxClamp(fillRisk, 0.0, 1.0)
        next.hasPathQuality = true
        return next
    }

    private static func windowShape(
        _ window: [[Double]],
        declaredWindowSize: Int?
    ) -> (trend: Double, fastTrend: Double, noise: Double, contextShape: Double) {
        let size = PluginContextRuntimeTools.effectiveWindowSize(window, declaredSize: declaredWindowSize)
        guard size > 1 else {
            return (0.0, 0.0, 0.0, 0.0)
        }
        let slope = abs(PluginContextRuntimeTools.currentWindowFeatureSlope(window, featureIndex: 0, declaredSize: size))
        let fastSlope = abs(PluginContextRuntimeTools.currentWindowFeatureRecentDelta(
            window,
            featureIndex: 0,
            recentBars: max(size / 4, 2),
            declaredSize: size
        ))
        let standardDeviation = PluginContextRuntimeTools.currentWindowFeatureStd(window, featureIndex: 0, declaredSize: size)
        let level = abs(PluginContextRuntimeTools.currentWindowFeatureEMAMean(window, featureIndex: 0, declaredSize: size))
        let localRange = PluginContextRuntimeTools.currentWindowFeatureRange(
            window,
            featureIndex: 0,
            recentBars: max(size / 2, 2),
            declaredSize: size
        )
        let contextRecent = abs(PluginContextRuntimeTools.currentWindowFeatureRecentMean(
            window,
            featureIndex: 10,
            recentBars: max(size / 4, 1),
            declaredSize: size
        ))
        let contextSlow = abs(PluginContextRuntimeTools.currentWindowFeatureMean(window, featureIndex: 10, declaredSize: size))
        let trend = fxClamp(slope / max(standardDeviation + 0.20 * abs(level), 0.10), 0.0, 1.25)
        let fastTrend = fxClamp(fastSlope / max(localRange + 0.10, 0.10), 0.0, 1.25)
        let noise = fxClamp((0.65 * standardDeviation + 0.35 * localRange) / max(abs(level) + 0.10, 0.10), 0.0, 1.25)
        let contextShape = fxClamp((contextRecent + 0.50 * contextSlow) / max(localRange + 0.10, 0.10), 0.0, 1.25)
        return (trend, fastTrend, noise, contextShape)
    }

    private static func familyMultipliers(_ family: AIFamily) -> (trend: Double, context: Double, execution: Double) {
        switch family {
        case .recurrent, .convolutional, .transformer, .stateSpace:
            return (1.12, 1.08, 1.0)
        case .worldModel, .retrieval, .mixture:
            return (1.0, 1.15, 0.92)
        case .tree, .linear:
            return (0.92, 1.0, 1.06)
        case .ruleBased:
            return (0.80, 0.85, 1.18)
        default:
            return (1.0, 1.0, 1.0)
        }
    }
}
