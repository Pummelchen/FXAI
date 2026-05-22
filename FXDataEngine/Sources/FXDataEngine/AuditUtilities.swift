import Foundation

public enum AuditUtilityTools {
    public static func value(_ values: [Double], index: Int, default defaultValue: Double) -> Double {
        guard index >= 0, index < values.count else { return defaultValue }
        return fxSafeFinite(values[index], fallback: defaultValue)
    }

    public static func positiveIntMean(
        _ values: [Int],
        startIndex: Int,
        width: Int,
        fallback: Double
    ) -> Double {
        guard !values.isEmpty, startIndex >= 0, startIndex < values.count, width > 0 else {
            return max(fallback, 0.10)
        }

        let end = min(values.count, startIndex + width)
        var sum = 0.0
        var used = 0
        for index in startIndex..<end {
            let value = Double(values[index])
            guard value > 0.0 else { continue }
            sum += value
            used += 1
        }
        guard used > 0 else { return max(fallback, 0.10) }
        return sum / Double(used)
    }

    public static func clampHorizon(_ horizonMinutes: Int) -> Int {
        HorizonTools.clampHorizon(horizonMinutes)
    }

    public static func fixedHorizonSlot(horizonMinutes: Int) -> Int {
        let horizon = clampHorizon(horizonMinutes)
        if horizon <= 3 { return 0 }
        if horizon <= 5 { return 1 }
        if horizon <= 8 { return 2 }
        if horizon <= 13 { return 3 }
        if horizon <= 21 { return 4 }
        if horizon <= 34 { return 5 }
        if horizon <= 55 { return 6 }
        return 7
    }

    public static func sanitizeNormalizationMethod(_ methodID: Int) -> FeatureNormalizationMethod {
        FeatureNormalizationMethod(rawValue: methodID) ?? .existing
    }

    public static func noSpreadStaticRegimeID(
        timestampUTC: Int64,
        liquidityStress: Double,
        liquidityStressReference: Double,
        volatilityProxyAbs: Double,
        volatilityReference: Double,
        regimeCount: Int = FXDataEngineConstants.pluginRegimeBuckets
    ) -> Int {
        let session = HorizonTools.sessionGroup(timestampUTC: timestampUTC)
        let liquidityReference = max(abs(liquidityStressReference), 0.10)
        let volatilityRef = max(abs(volatilityReference), 1e-6)
        let liquidityHigh = abs(liquidityStress) > (1.15 * liquidityReference + 0.10)
        let volatilityHigh = abs(volatilityProxyAbs) > (1.15 * volatilityRef + 0.02)
        let regime = session * 4 + (volatilityHigh ? 2 : 0) + (liquidityHigh ? 1 : 0)
        return Int(fxClamp(Double(regime), 0.0, Double(max(regimeCount - 1, 0))))
    }

    public static func defaultHyperParameters(aiID: Int) -> AIHyperParameters {
        var parameters = AIHyperParameters(
            learningRate: 0.0100,
            l2: 0.0030,
            ftrlAlpha: 0.0800,
            ftrlBeta: 1.0000,
            ftrlL1: 0.0005,
            ftrlL2: 0.0100,
            paC: 4.0000,
            paMargin: 1.2000,
            xgbLearningRate: 0.0800,
            xgbL2: 0.0200,
            xgbSplit: 0.5000,
            mlpLearningRate: 0.0100,
            mlpL2: 0.0030,
            mlpInit: 0.1000,
            quantileLearningRate: 0.0100,
            quantileL2: 0.0030,
            enhashLearningRate: 0.0100,
            enhashL1: 0.0000,
            enhashL2: 0.0050,
            tcnLayers: 4.0000,
            tcnKernel: 3.0000,
            tcnDilationBase: 2.0000
        )

        guard let model = AIModelID(rawValue: aiID) else { return parameters }
        switch model {
        case .m1Sync, .buyOnly, .sellOnly, .randomNoSkip:
            parameters.learningRate = 0.0
            parameters.l2 = 0.0
        case .ftrlLogit:
            parameters.ftrlAlpha = 0.0800
            parameters.ftrlBeta = 1.0000
            parameters.ftrlL1 = 0.0000
            parameters.ftrlL2 = 0.0100
        case .paLinear:
            parameters.learningRate = 0.0600
            parameters.l2 = 0.0030
            parameters.paC = 4.0000
            parameters.paMargin = 1.2000
        case .tcn:
            parameters.learningRate = 0.0060
            parameters.l2 = 0.0020
            parameters.tcnLayers = 4.0000
            parameters.tcnKernel = 3.0000
            parameters.tcnDilationBase = 2.0000
        case .lstm, .lstmg, .tft, .autoformer, .patchTST, .chronos, .timesfm,
             .tst, .stmn, .s4, .geodesicAttention, .qcew, .fewc, .gha, .tesseract:
            parameters.learningRate = 0.0060
            parameters.l2 = 0.0020
        default:
            break
        }
        return parameters
    }
}
