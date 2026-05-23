import Foundation

public struct TrainingRandomGenerator: Codable, Hashable, Sendable {
    public private(set) var state: UInt64

    public init(seed: UInt64 = 1) {
        self.state = seed == 0 ? 1 : seed
    }

    public mutating func setSeed(_ seed: UInt64) {
        state = seed == 0 ? 1 : seed
    }

    public mutating func nextUnit() -> Double {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let sample = state >> 11
        return fxClamp(Double(sample) / 9_007_199_254_740_991.0, 0.0, 1.0)
    }

    public mutating func range(_ lower: Double, _ upper: Double) -> Double {
        guard upper > lower else { return lower }
        return lower + (upper - lower) * fxClamp(nextUnit(), 0.0, 1.0)
    }
}

public struct AIHyperParameterInputs: Codable, Hashable, Sendable {
    public var learningRate: Double
    public var l2: Double
    public var ftrlAlpha: Double
    public var ftrlBeta: Double
    public var ftrlL1: Double
    public var ftrlL2: Double
    public var paC: Double
    public var paMargin: Double
    public var xgbLearningRate: Double
    public var xgbL2: Double
    public var xgbSplit: Double
    public var mlpLearningRate: Double
    public var mlpL2: Double
    public var mlpInit: Double
    public var tcnLayers: Int
    public var tcnKernel: Int
    public var tcnDilationBase: Int
    public var quantileLearningRate: Double
    public var quantileL2: Double
    public var enhashLearningRate: Double
    public var enhashL1: Double
    public var enhashL2: Double

    public init(
        learningRate: Double = 0.01,
        l2: Double = 0.010,
        ftrlAlpha: Double = 0.08,
        ftrlBeta: Double = 1.00,
        ftrlL1: Double = 0.0005,
        ftrlL2: Double = 0.0100,
        paC: Double = 4.00,
        paMargin: Double = 1.20,
        xgbLearningRate: Double = 0.03,
        xgbL2: Double = 4.0,
        xgbSplit: Double = 0.0,
        mlpLearningRate: Double = 0.010,
        mlpL2: Double = 0.0005,
        mlpInit: Double = 0.10,
        tcnLayers: Int = 4,
        tcnKernel: Int = 3,
        tcnDilationBase: Int = 2,
        quantileLearningRate: Double = 0.015,
        quantileL2: Double = 0.0005,
        enhashLearningRate: Double = 0.020,
        enhashL1: Double = 0.0001,
        enhashL2: Double = 0.0020
    ) {
        self.learningRate = learningRate
        self.l2 = l2
        self.ftrlAlpha = ftrlAlpha
        self.ftrlBeta = ftrlBeta
        self.ftrlL1 = ftrlL1
        self.ftrlL2 = ftrlL2
        self.paC = paC
        self.paMargin = paMargin
        self.xgbLearningRate = xgbLearningRate
        self.xgbL2 = xgbL2
        self.xgbSplit = xgbSplit
        self.mlpLearningRate = mlpLearningRate
        self.mlpL2 = mlpL2
        self.mlpInit = mlpInit
        self.tcnLayers = tcnLayers
        self.tcnKernel = tcnKernel
        self.tcnDilationBase = tcnDilationBase
        self.quantileLearningRate = quantileLearningRate
        self.quantileL2 = quantileL2
        self.enhashLearningRate = enhashLearningRate
        self.enhashL1 = enhashL1
        self.enhashL2 = enhashL2
    }
}

public struct AIHyperParameters: Codable, Hashable, Sendable {
    public var learningRate: Double
    public var l2: Double
    public var ftrlAlpha: Double
    public var ftrlBeta: Double
    public var ftrlL1: Double
    public var ftrlL2: Double
    public var paC: Double
    public var paMargin: Double
    public var xgbLearningRate: Double
    public var xgbL2: Double
    public var xgbSplit: Double
    public var mlpLearningRate: Double
    public var mlpL2: Double
    public var mlpInit: Double
    public var quantileLearningRate: Double
    public var quantileL2: Double
    public var enhashLearningRate: Double
    public var enhashL1: Double
    public var enhashL2: Double
    public var tcnLayers: Double
    public var tcnKernel: Double
    public var tcnDilationBase: Double

    public init(
        learningRate: Double,
        l2: Double,
        ftrlAlpha: Double,
        ftrlBeta: Double,
        ftrlL1: Double,
        ftrlL2: Double,
        paC: Double,
        paMargin: Double,
        xgbLearningRate: Double,
        xgbL2: Double,
        xgbSplit: Double,
        mlpLearningRate: Double,
        mlpL2: Double,
        mlpInit: Double,
        quantileLearningRate: Double,
        quantileL2: Double,
        enhashLearningRate: Double,
        enhashL1: Double,
        enhashL2: Double,
        tcnLayers: Double,
        tcnKernel: Double,
        tcnDilationBase: Double
    ) {
        self.learningRate = fxSafeFinite(learningRate)
        self.l2 = fxSafeFinite(l2)
        self.ftrlAlpha = fxSafeFinite(ftrlAlpha)
        self.ftrlBeta = fxSafeFinite(ftrlBeta)
        self.ftrlL1 = fxSafeFinite(ftrlL1)
        self.ftrlL2 = fxSafeFinite(ftrlL2)
        self.paC = fxSafeFinite(paC)
        self.paMargin = fxSafeFinite(paMargin)
        self.xgbLearningRate = fxSafeFinite(xgbLearningRate)
        self.xgbL2 = fxSafeFinite(xgbL2)
        self.xgbSplit = fxSafeFinite(xgbSplit)
        self.mlpLearningRate = fxSafeFinite(mlpLearningRate)
        self.mlpL2 = fxSafeFinite(mlpL2)
        self.mlpInit = fxSafeFinite(mlpInit)
        self.quantileLearningRate = fxSafeFinite(quantileLearningRate)
        self.quantileL2 = fxSafeFinite(quantileL2)
        self.enhashLearningRate = fxSafeFinite(enhashLearningRate)
        self.enhashL1 = fxSafeFinite(enhashL1)
        self.enhashL2 = fxSafeFinite(enhashL2)
        self.tcnLayers = fxSafeFinite(tcnLayers)
        self.tcnKernel = fxSafeFinite(tcnKernel)
        self.tcnDilationBase = fxSafeFinite(tcnDilationBase)
    }
}

public struct AIHorizonRoutingKey: Codable, Hashable, Sendable {
    public var aiID: Int
    public var horizonSlot: Int

    public init(aiID: Int, horizonSlot: Int) {
        self.aiID = aiID
        self.horizonSlot = horizonSlot
    }
}

public struct AIRegimeRoutingKey: Codable, Hashable, Sendable {
    public var aiID: Int
    public var regimeID: Int

    public init(aiID: Int, regimeID: Int) {
        self.aiID = aiID
        self.regimeID = regimeID
    }
}

public struct AIRegimeHorizonRoutingKey: Codable, Hashable, Sendable {
    public var aiID: Int
    public var regimeID: Int
    public var horizonSlot: Int

    public init(aiID: Int, regimeID: Int, horizonSlot: Int) {
        self.aiID = aiID
        self.regimeID = regimeID
        self.horizonSlot = horizonSlot
    }
}

public struct AITrainingRoutingState: Codable, Hashable, Sendable {
    public var modelHyperParameters: [Int: AIHyperParameters]
    public var horizonHyperParameters: [AIHorizonRoutingKey: AIHyperParameters]
    public var regimeHorizonHyperParameters: [AIRegimeHorizonRoutingKey: AIHyperParameters]
    public var modelNormalizationMethods: [Int: FeatureNormalizationMethod]
    public var horizonNormalizationMethods: [AIHorizonRoutingKey: FeatureNormalizationMethod]
    public var regimeHorizonNormalizationMethods: [AIRegimeHorizonRoutingKey: FeatureNormalizationMethod]
    public var modelThresholds: [Int: WarmupThresholdPair]
    public var horizonThresholds: [AIHorizonRoutingKey: WarmupThresholdPair]
    public var regimeThresholds: [AIRegimeRoutingKey: WarmupThresholdPair]
    public var regimeHorizonThresholds: [AIRegimeHorizonRoutingKey: WarmupThresholdPair]
    public var modelGlobalEdges: [Int: Double]
    public var horizonEdges: [AIHorizonRoutingKey: Double]

    public init(
        modelHyperParameters: [Int: AIHyperParameters] = [:],
        horizonHyperParameters: [AIHorizonRoutingKey: AIHyperParameters] = [:],
        regimeHorizonHyperParameters: [AIRegimeHorizonRoutingKey: AIHyperParameters] = [:],
        modelNormalizationMethods: [Int: FeatureNormalizationMethod] = [:],
        horizonNormalizationMethods: [AIHorizonRoutingKey: FeatureNormalizationMethod] = [:],
        regimeHorizonNormalizationMethods: [AIRegimeHorizonRoutingKey: FeatureNormalizationMethod] = [:],
        modelThresholds: [Int: WarmupThresholdPair] = [:],
        horizonThresholds: [AIHorizonRoutingKey: WarmupThresholdPair] = [:],
        regimeThresholds: [AIRegimeRoutingKey: WarmupThresholdPair] = [:],
        regimeHorizonThresholds: [AIRegimeHorizonRoutingKey: WarmupThresholdPair] = [:],
        modelGlobalEdges: [Int: Double] = [:],
        horizonEdges: [AIHorizonRoutingKey: Double] = [:]
    ) {
        self.modelHyperParameters = modelHyperParameters
        self.horizonHyperParameters = horizonHyperParameters
        self.regimeHorizonHyperParameters = regimeHorizonHyperParameters
        self.modelNormalizationMethods = modelNormalizationMethods
        self.horizonNormalizationMethods = horizonNormalizationMethods
        self.regimeHorizonNormalizationMethods = regimeHorizonNormalizationMethods
        self.modelThresholds = modelThresholds
        self.horizonThresholds = horizonThresholds
        self.regimeThresholds = regimeThresholds
        self.regimeHorizonThresholds = regimeHorizonThresholds
        self.modelGlobalEdges = modelGlobalEdges.mapValues { fxSafeFinite($0) }
        self.horizonEdges = horizonEdges.mapValues { fxSafeFinite($0) }
    }
}

public enum AIHyperParameterTools {
    public static func baseParameters(inputs: AIHyperParameterInputs = AIHyperParameterInputs()) -> AIHyperParameters {
        AIHyperParameters(
            learningRate: fxClamp(inputs.learningRate, 0.001, 0.200),
            l2: fxClamp(inputs.l2, 0.0, 0.100),
            ftrlAlpha: fxClamp(inputs.ftrlAlpha, 0.001, 1.000),
            ftrlBeta: fxClamp(inputs.ftrlBeta, 0.000, 5.000),
            ftrlL1: fxClamp(inputs.ftrlL1, 0.000, 0.100),
            ftrlL2: fxClamp(inputs.ftrlL2, 0.000, 1.000),
            paC: fxClamp(inputs.paC, 0.010, 10.000),
            paMargin: fxClamp(inputs.paMargin, 0.100, 2.000),
            xgbLearningRate: fxClamp(inputs.xgbLearningRate, 0.001, 0.300),
            xgbL2: fxClamp(inputs.xgbL2, 0.000, 10.000),
            xgbSplit: fxClamp(inputs.xgbSplit, -2.000, 2.000),
            mlpLearningRate: fxClamp(inputs.mlpLearningRate, 0.0005, 0.0500),
            mlpL2: fxClamp(inputs.mlpL2, 0.0000, 0.0500),
            mlpInit: fxClamp(inputs.mlpInit, 0.0100, 0.5000),
            quantileLearningRate: fxClamp(inputs.quantileLearningRate, 0.0001, 0.1000),
            quantileL2: fxClamp(inputs.quantileL2, 0.0000, 0.1000),
            enhashLearningRate: fxClamp(inputs.enhashLearningRate, 0.0005, 0.1000),
            enhashL1: fxClamp(inputs.enhashL1, 0.0000, 0.1000),
            enhashL2: fxClamp(inputs.enhashL2, 0.0000, 0.1000),
            tcnLayers: Double(Int(fxClamp(Double(inputs.tcnLayers), 2.0, 8.0))),
            tcnKernel: Double(Int(fxClamp(Double(inputs.tcnKernel), 2.0, 5.0))),
            tcnDilationBase: Double(Int(fxClamp(Double(inputs.tcnDilationBase), 1.0, 3.0)))
        )
    }

    public static func defaultParameters(aiID: Int, inputs: AIHyperParameterInputs = AIHyperParameterInputs()) -> AIHyperParameters {
        var parameters = baseParameters(inputs: inputs)
        guard let model = AIModelID(rawValue: aiID) else { return parameters }

        switch model {
        case .geodesicAttention:
            parameters.learningRate = 0.0060
            parameters.l2 = 0.0030
        case .qcew:
            parameters.learningRate = 0.0045
            parameters.l2 = 0.0035
        case .fewc:
            parameters.learningRate = 0.0060
            parameters.l2 = 0.0020
        case .gha:
            parameters.learningRate = 0.0040
            parameters.l2 = 0.0035
        case .tesseract:
            parameters.learningRate = 0.0045
            parameters.l2 = 0.0030
        case .lstm:
            parameters.learningRate = 0.0080
            parameters.l2 = 0.0040
        case .lightgbm:
            parameters.xgbLearningRate = 0.0300
            parameters.xgbL2 = 4.0000
            parameters.xgbSplit = 0.0000
        case .paLinear:
            parameters.learningRate = 0.0600
            parameters.l2 = 0.0030
            parameters.paC = 4.0000
            parameters.paMargin = 1.2000
        case .cfxWorld:
            parameters.learningRate = 0.0100
            parameters.l2 = 0.0020
        case .loffm:
            parameters.learningRate = 0.0080
            parameters.l2 = 0.0030
        case .trr:
            parameters.learningRate = 0.0090
            parameters.l2 = 0.0025
        case .graphWM:
            parameters.learningRate = 0.0080
            parameters.l2 = 0.0020
        case .moeConformal:
            parameters.learningRate = 0.0060
            parameters.l2 = 0.0030
        case .m1Sync, .buyOnly, .sellOnly, .randomNoSkip:
            parameters.learningRate = 0.0
            parameters.l2 = 0.0
        default:
            break
        }
        return parameters
    }

    public static func routedParameters(
        aiID: Int,
        regimeID: Int,
        horizonMinutes: Int,
        state: AITrainingRoutingState,
        inputs: AIHyperParameterInputs = AIHyperParameterInputs(),
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) -> AIHyperParameters {
        var parameters = defaultParameters(aiID: aiID, inputs: inputs)
        guard isValidAIID(aiID) else { return parameters }
        if let modelParameters = state.modelHyperParameters[aiID] {
            parameters = modelParameters
        }

        let horizonSlot = TrainingSampleTools.horizonSlot(
            horizonMinutes: horizonMinutes,
            configuredHorizons: configuredHorizons
        )
        guard isValidHorizonSlot(horizonSlot) else { return parameters }

        let horizonKey = AIHorizonRoutingKey(aiID: aiID, horizonSlot: horizonSlot)
        if let horizonParameters = state.horizonHyperParameters[horizonKey] {
            parameters = horizonParameters
        }

        if isValidRegimeID(regimeID),
           let bankParameters = state.regimeHorizonHyperParameters[
            AIRegimeHorizonRoutingKey(aiID: aiID, regimeID: regimeID, horizonSlot: horizonSlot)
           ] {
            parameters = bankParameters
        }

        return parameters
    }

    public static func routedNormalizationMethod(
        aiID: Int,
        regimeID: Int,
        horizonMinutes: Int,
        state: AITrainingRoutingState,
        currentMethod: FeatureNormalizationMethod,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) -> FeatureNormalizationMethod {
        var method = currentMethod
        guard isValidAIID(aiID) else { return method }

        if let modelMethod = state.modelNormalizationMethods[aiID] {
            method = modelMethod
        }

        let horizonSlot = TrainingSampleTools.horizonSlot(
            horizonMinutes: horizonMinutes,
            configuredHorizons: configuredHorizons
        )
        guard isValidHorizonSlot(horizonSlot) else { return method }

        let horizonKey = AIHorizonRoutingKey(aiID: aiID, horizonSlot: horizonSlot)
        if let horizonMethod = state.horizonNormalizationMethods[horizonKey] {
            method = horizonMethod
        }

        if isValidRegimeID(regimeID),
           let bankMethod = state.regimeHorizonNormalizationMethods[
            AIRegimeHorizonRoutingKey(aiID: aiID, regimeID: regimeID, horizonSlot: horizonSlot)
           ] {
            method = bankMethod
        }

        return method
    }

    public static func routedThresholdPair(
        aiID: Int,
        regimeID: Int,
        horizonMinutes: Int,
        baseBuy: Double,
        baseSell: Double,
        state: AITrainingRoutingState,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) -> WarmupThresholdPair {
        var thresholds = WarmupTools.sanitizeThresholdPair(buyThreshold: baseBuy, sellThreshold: baseSell)
        guard isValidAIID(aiID) else { return thresholds }

        if let modelThresholds = state.modelThresholds[aiID] {
            thresholds = modelThresholds
        }

        let horizonSlot = TrainingSampleTools.horizonSlot(
            horizonMinutes: horizonMinutes,
            configuredHorizons: configuredHorizons
        )
        guard isValidHorizonSlot(horizonSlot) else {
            return WarmupTools.sanitizeThresholdPair(buyThreshold: thresholds.buy, sellThreshold: thresholds.sell)
        }

        let horizonKey = AIHorizonRoutingKey(aiID: aiID, horizonSlot: horizonSlot)
        if let horizonThresholds = state.horizonThresholds[horizonKey] {
            thresholds.buy = 0.55 * thresholds.buy + 0.45 * horizonThresholds.buy
            thresholds.sell = 0.55 * thresholds.sell + 0.45 * horizonThresholds.sell
        }

        if isValidRegimeID(regimeID),
           let regimeThresholds = state.regimeThresholds[AIRegimeRoutingKey(aiID: aiID, regimeID: regimeID)] {
            thresholds.buy = 0.65 * thresholds.buy + 0.35 * regimeThresholds.buy
            thresholds.sell = 0.65 * thresholds.sell + 0.35 * regimeThresholds.sell
        }

        if isValidRegimeID(regimeID),
           let bankThresholds = state.regimeHorizonThresholds[
            AIRegimeHorizonRoutingKey(aiID: aiID, regimeID: regimeID, horizonSlot: horizonSlot)
           ] {
            thresholds.buy = 0.35 * thresholds.buy + 0.65 * bankThresholds.buy
            thresholds.sell = 0.35 * thresholds.sell + 0.65 * bankThresholds.sell
        }

        if let horizonEdge = state.horizonEdges[horizonKey] {
            let denominator = max(0.50, abs(state.modelGlobalEdges[aiID] ?? 0.0) + 0.50)
            let adjustment = fxClamp(horizonEdge / denominator, -0.08, 0.08)
            thresholds.buy = fxClamp(thresholds.buy - (0.35 * adjustment), 0.50, 0.95)
            thresholds.sell = fxClamp(thresholds.sell + (0.35 * adjustment), 0.05, 0.50)
        }

        return WarmupTools.sanitizeThresholdPair(buyThreshold: thresholds.buy, sellThreshold: thresholds.sell)
    }

    public static func sampleThresholdPair(
        baseBuy: Double,
        baseSell: Double,
        rng: inout TrainingRandomGenerator
    ) -> WarmupThresholdPair {
        let sanitized = WarmupTools.sanitizeThresholdPair(buyThreshold: baseBuy, sellThreshold: baseSell)
        let buyLower = max(0.52, sanitized.buy - 0.08)
        let buyUpper = min(0.90, sanitized.buy + 0.08)
        let sellLower = max(0.08, sanitized.sell - 0.08)
        let sellUpper = min(0.48, sanitized.sell + 0.08)
        return WarmupTools.sanitizeThresholdPair(
            buyThreshold: fxClamp(rng.range(buyLower, buyUpper), 0.50, 0.95),
            sellThreshold: fxClamp(rng.range(sellLower, sellUpper), 0.05, 0.50)
        )
    }

    public static func sampleParameters(
        aiID: Int,
        base: AIHyperParameters,
        rng: inout TrainingRandomGenerator
    ) -> AIHyperParameters {
        var parameters = base
        guard let model = AIModelID(rawValue: aiID) else {
            parameters.learningRate = rng.range(0.0030, 0.0600)
            parameters.l2 = rng.range(0.0000, 0.0300)
            return parameters
        }

        switch model {
        case .sgdLogit, .lstmg, .s4, .tft, .autoformer, .stmn, .tst, .patchTST,
             .chronos, .timesfm, .cfxWorld, .loffm, .trr, .graphWM, .moeConformal,
             .retrDiff:
            parameters.learningRate = rng.range(0.0030, 0.0600)
            parameters.l2 = rng.range(0.0000, 0.0300)
        case .m1Sync, .buyOnly, .sellOnly, .randomNoSkip:
            break
        case .lstm:
            parameters.learningRate = rng.range(0.0040, 0.0200)
            parameters.l2 = rng.range(0.0010, 0.0100)
        case .geodesicAttention:
            parameters.learningRate = rng.range(0.0030, 0.0150)
            parameters.l2 = rng.range(0.0010, 0.0080)
        case .qcew, .gha, .tesseract:
            parameters.learningRate = rng.range(0.0025, 0.0140)
            parameters.l2 = rng.range(0.0010, 0.0100)
        case .fewc:
            parameters.learningRate = rng.range(0.0030, 0.0280)
            parameters.l2 = rng.range(0.0005, 0.0120)
        case .tcn:
            parameters.learningRate = rng.range(0.0030, 0.0500)
            parameters.l2 = rng.range(0.0000, 0.0200)
            parameters.tcnLayers = round(rng.range(3.0, 6.0))
            parameters.tcnKernel = round(rng.range(2.0, 4.0))
            parameters.tcnDilationBase = round(rng.range(1.0, 3.0))
        case .ftrlLogit:
            parameters.ftrlAlpha = rng.range(0.0100, 0.2500)
            parameters.ftrlBeta = rng.range(0.1000, 2.5000)
            parameters.ftrlL1 = rng.range(0.0000, 0.0100)
            parameters.ftrlL2 = rng.range(0.0000, 0.1000)
        case .paLinear:
            parameters.learningRate = rng.range(0.0200, 0.0800)
            parameters.l2 = rng.range(0.0010, 0.0100)
            parameters.paC = rng.range(0.5000, 6.0000)
            parameters.paMargin = rng.range(0.6000, 1.9000)
        case .xgbFast, .xgboost:
            parameters.xgbLearningRate = rng.range(0.0050, 0.1200)
            parameters.xgbL2 = rng.range(0.0000, 0.0300)
            parameters.xgbSplit = rng.range(-0.8000, 0.8000)
        case .lightgbm:
            parameters.xgbLearningRate = rng.range(0.0200, 0.0400)
            parameters.xgbL2 = rng.range(2.0000, 6.0000)
            parameters.xgbSplit = rng.range(-0.2000, 0.2000)
        case .catboost:
            parameters.xgbLearningRate = rng.range(0.0200, 0.0500)
            parameters.xgbL2 = rng.range(3.0000, 8.0000)
            parameters.xgbSplit = rng.range(-0.2000, 0.2000)
        case .mlpTiny:
            parameters.mlpLearningRate = rng.range(0.0010, 0.0300)
            parameters.mlpL2 = rng.range(0.0000, 0.0200)
            parameters.mlpInit = rng.range(0.0300, 0.2500)
        case .quantile:
            parameters.quantileLearningRate = rng.range(0.0010, 0.0500)
            parameters.quantileL2 = rng.range(0.0000, 0.0200)
        case .enhash:
            parameters.enhashLearningRate = rng.range(0.0020, 0.0500)
            parameters.enhashL1 = rng.range(0.0000, 0.0100)
            parameters.enhashL2 = rng.range(0.0000, 0.0200)
        default:
            parameters.learningRate = rng.range(0.0030, 0.0600)
            parameters.l2 = rng.range(0.0000, 0.0300)
        }
        return parameters
    }

    private static func isValidAIID(_ aiID: Int) -> Bool {
        AIModelID(rawValue: aiID) != nil
    }

    private static func isValidRegimeID(_ regimeID: Int) -> Bool {
        regimeID >= 0 && regimeID < FXDataEngineConstants.pluginRegimeBuckets
    }

    private static func isValidHorizonSlot(_ horizonSlot: Int) -> Bool {
        horizonSlot >= 0 && horizonSlot < RuntimeArtifactConstants.maxHorizons
    }
}
