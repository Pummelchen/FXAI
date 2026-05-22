import Foundation

public struct HyperParameters: Codable, Hashable, Sendable {
    public var learningRate: Double
    public var l2: Double
    public var ftrlAlpha: Double
    public var ftrlBeta: Double
    public var ftrlL1: Double
    public var ftrlL2: Double
    public var passiveAggressiveC: Double
    public var passiveAggressiveMargin: Double
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
        learningRate: Double = 0.01,
        l2: Double = 0.0001,
        ftrlAlpha: Double = 0.05,
        ftrlBeta: Double = 1.0,
        ftrlL1: Double = 0.0,
        ftrlL2: Double = 0.0001,
        passiveAggressiveC: Double = 0.5,
        passiveAggressiveMargin: Double = 0.05,
        xgbLearningRate: Double = 0.05,
        xgbL2: Double = 0.001,
        xgbSplit: Double = 0.0,
        mlpLearningRate: Double = 0.01,
        mlpL2: Double = 0.0001,
        mlpInit: Double = 0.05,
        quantileLearningRate: Double = 0.01,
        quantileL2: Double = 0.0001,
        enhashLearningRate: Double = 0.01,
        enhashL1: Double = 0.0,
        enhashL2: Double = 0.0001,
        tcnLayers: Double = 2,
        tcnKernel: Double = 3,
        tcnDilationBase: Double = 2
    ) {
        self.learningRate = learningRate
        self.l2 = l2
        self.ftrlAlpha = ftrlAlpha
        self.ftrlBeta = ftrlBeta
        self.ftrlL1 = ftrlL1
        self.ftrlL2 = ftrlL2
        self.passiveAggressiveC = passiveAggressiveC
        self.passiveAggressiveMargin = passiveAggressiveMargin
        self.xgbLearningRate = xgbLearningRate
        self.xgbL2 = xgbL2
        self.xgbSplit = xgbSplit
        self.mlpLearningRate = mlpLearningRate
        self.mlpL2 = mlpL2
        self.mlpInit = mlpInit
        self.quantileLearningRate = quantileLearningRate
        self.quantileL2 = quantileL2
        self.enhashLearningRate = enhashLearningRate
        self.enhashL1 = enhashL1
        self.enhashL2 = enhashL2
        self.tcnLayers = tcnLayers
        self.tcnKernel = tcnKernel
        self.tcnDilationBase = tcnDilationBase
    }
}

public struct PluginManifestV4: Codable, Hashable, Sendable {
    public var apiVersion: Int
    public var aiID: Int
    public var aiName: String
    public var family: AIFamily
    public var referenceTier: ReferenceTier
    public var capabilityMask: PluginCapability
    public var featureSchema: FeatureSchema
    public var featureGroups: FeatureGroupMask
    public var minHorizonMinutes: Int
    public var maxHorizonMinutes: Int
    public var minSequenceBars: Int
    public var maxSequenceBars: Int
    public var requiresVolumeWhenAvailable: Bool

    public init(
        apiVersion: Int = FXDataEngineConstants.apiVersionV4,
        aiID: Int,
        aiName: String,
        family: AIFamily,
        referenceTier: ReferenceTier = .fullNative,
        capabilityMask: PluginCapability = [.selfTest],
        featureSchema: FeatureSchema? = nil,
        featureGroups: FeatureGroupMask? = nil,
        minHorizonMinutes: Int = 1,
        maxHorizonMinutes: Int = 240,
        minSequenceBars: Int = 1,
        maxSequenceBars: Int = 1,
        requiresVolumeWhenAvailable: Bool = true,
        schemaPolicy: FeatureSchemaPolicy = FeatureSchemaPolicy()
    ) {
        self.apiVersion = apiVersion
        self.aiID = aiID
        self.aiName = aiName
        self.family = family
        self.referenceTier = referenceTier
        self.capabilityMask = capabilityMask
        self.featureSchema = featureSchema ?? schemaPolicy.defaultSchema(for: family)
        self.featureGroups = featureGroups ?? schemaPolicy.defaultGroups(for: family)
        self.minHorizonMinutes = max(1, minHorizonMinutes)
        self.maxHorizonMinutes = max(maxHorizonMinutes, max(1, minHorizonMinutes))
        self.minSequenceBars = max(1, minSequenceBars)
        self.maxSequenceBars = min(max(maxSequenceBars, max(1, minSequenceBars)), FXDataEngineConstants.maxSequenceBars)
        self.requiresVolumeWhenAvailable = requiresVolumeWhenAvailable
    }

    public func validate() throws {
        guard apiVersion == FXDataEngineConstants.apiVersionV4 else {
            throw FXDataEngineError.validation("manifest.apiVersion")
        }
        guard (0..<FXDataEngineConstants.aiCount).contains(aiID) else {
            throw FXDataEngineError.validation("manifest.aiID")
        }
        guard !aiName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXDataEngineError.validation("manifest.aiName")
        }
        guard !featureGroups.isEmpty else {
            throw FXDataEngineError.validation("manifest.featureGroups")
        }
        guard capabilityMask.contains(.selfTest) else {
            throw FXDataEngineError.validation("manifest.capabilityMask.selfTest")
        }
        guard minHorizonMinutes > 0, maxHorizonMinutes >= minHorizonMinutes else {
            throw FXDataEngineError.validation("manifest.horizonRange")
        }
        guard minSequenceBars > 0,
              maxSequenceBars >= minSequenceBars,
              maxSequenceBars <= FXDataEngineConstants.maxSequenceBars else {
            throw FXDataEngineError.validation("manifest.sequenceRange")
        }
        if capabilityMask.contains(.replay), !capabilityMask.contains(.onlineLearning) {
            throw FXDataEngineError.validation("manifest.replayRequiresOnlineLearning")
        }
        if capabilityMask.contains(.windowContext) || capabilityMask.contains(.stateful) {
            guard maxSequenceBars > 1 else {
                throw FXDataEngineError.validation("manifest.windowContextSequenceRange")
            }
        }
    }

    public func resolvedSequenceBars(horizonMinutes: Int) -> Int {
        guard capabilityMask.contains(.windowContext) || capabilityMask.contains(.stateful) else { return 1 }
        let sequence = max(minSequenceBars, min(maxSequenceBars, max(1, horizonMinutes) * 8))
        return min(sequence, FXDataEngineConstants.maxSequenceBars)
    }
}

public struct PluginContextV4: Codable, Hashable, Sendable {
    public var apiVersion: Int
    public var regimeID: Int
    public var sessionBucket: Int
    public var horizonMinutes: Int
    public var featureSchema: FeatureSchema
    public var normalizationMethod: FeatureNormalizationMethod
    public var sequenceBars: Int
    public var pointValue: Double
    public var domainHash: Double
    public var sampleTimeUTC: Int64
    public var dataHasVolume: Bool

    public init(
        apiVersion: Int = FXDataEngineConstants.apiVersionV4,
        regimeID: Int = 0,
        sessionBucket: Int = 0,
        horizonMinutes: Int = 1,
        featureSchema: FeatureSchema = .full,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        sequenceBars: Int = 1,
        pointValue: Double = 1.0,
        domainHash: Double = 0.0,
        sampleTimeUTC: Int64 = 0,
        dataHasVolume: Bool = false
    ) {
        self.apiVersion = apiVersion
        self.regimeID = regimeID
        self.sessionBucket = sessionBucket
        self.horizonMinutes = max(1, horizonMinutes)
        self.featureSchema = featureSchema
        self.normalizationMethod = normalizationMethod
        self.sequenceBars = min(max(1, sequenceBars), FXDataEngineConstants.maxSequenceBars)
        self.pointValue = pointValue
        self.domainHash = fxClamp(domainHash, 0.0, 1.0)
        self.sampleTimeUTC = sampleTimeUTC
        self.dataHasVolume = dataHasVolume
    }

    public func validate() throws {
        guard apiVersion == FXDataEngineConstants.apiVersionV4 else {
            throw FXDataEngineError.validation("ctx.apiVersion")
        }
        guard (0..<FXDataEngineConstants.pluginRegimeBuckets).contains(regimeID) else {
            throw FXDataEngineError.validation("ctx.regimeID")
        }
        guard (0..<FXDataEngineConstants.pluginSessionBuckets).contains(sessionBucket) else {
            throw FXDataEngineError.validation("ctx.sessionBucket")
        }
        guard horizonMinutes > 0 else {
            throw FXDataEngineError.validation("ctx.horizonMinutes")
        }
        guard (1...FXDataEngineConstants.maxSequenceBars).contains(sequenceBars) else {
            throw FXDataEngineError.validation("ctx.sequenceBars")
        }
        guard pointValue.isFinite, pointValue > 0 else {
            throw FXDataEngineError.validation("ctx.pointValue")
        }
        guard domainHash.isFinite, domainHash >= 0, domainHash <= 1 else {
            throw FXDataEngineError.validation("ctx.domainHash")
        }
    }
}

public struct PredictRequestV4: Sendable {
    public let valid: Bool
    public let context: PluginContextV4
    public let windowSize: Int
    public let x: [Double]
    public let xWindow: [[Double]]

    public init(valid: Bool, context: PluginContextV4, windowSize: Int = 0, x: [Double], xWindow: [[Double]] = []) {
        self.valid = valid
        self.context = context
        self.windowSize = min(max(0, windowSize), FXDataEngineConstants.maxSequenceBars)
        self.x = x
        self.xWindow = xWindow
    }

    public func validate() throws {
        guard valid else { throw FXDataEngineError.validation("req.valid") }
        try context.validate()
        try Self.validateInput(x)
        try Self.validateWindow(xWindow, windowSize: windowSize, context: context)
    }

    public func replacingPayload(x: [Double], xWindow: [[Double]]) -> PredictRequestV4 {
        PredictRequestV4(valid: valid, context: context, windowSize: xWindow.count, x: x, xWindow: xWindow)
    }

    static func validateInput(_ x: [Double]) throws {
        guard x.count == FXDataEngineConstants.aiWeights else {
            throw FXDataEngineError.validation("req.x length")
        }
        guard x.allSatisfy(\.isFinite) else {
            throw FXDataEngineError.validation("req.x finite")
        }
    }

    static func validateWindow(_ window: [[Double]], windowSize: Int, context: PluginContextV4) throws {
        guard windowSize >= 0, windowSize <= FXDataEngineConstants.maxSequenceBars else {
            throw FXDataEngineError.validation("req.windowSize")
        }
        guard windowSize <= max(context.sequenceBars - 1, 0) else {
            throw FXDataEngineError.validation("req.windowSizeContext")
        }
        if context.sequenceBars > 1 {
            guard windowSize > 0 else { throw FXDataEngineError.validation("req.windowPayload") }
        }
        for row in window.prefix(windowSize) {
            try validateInput(row)
        }
    }
}

public struct TrainRequestV4: Sendable {
    public let valid: Bool
    public let context: PluginContextV4
    public let labelClass: LabelClass
    public let movePoints: Double
    public let sampleWeight: Double
    public let mfePoints: Double
    public let maePoints: Double
    public let timeToHitFraction: Double
    public let pathFlags: Int
    public let pathRisk: Double
    public let fillRisk: Double
    public let maskedStepTarget: Double
    public let nextVolumeTarget: Double
    public let regimeShiftTarget: Double
    public let contextLeadTarget: Double
    public let windowSize: Int
    public let x: [Double]
    public let xWindow: [[Double]]

    public init(
        valid: Bool,
        context: PluginContextV4,
        labelClass: LabelClass,
        movePoints: Double,
        sampleWeight: Double,
        mfePoints: Double = 0,
        maePoints: Double = 0,
        timeToHitFraction: Double = 1,
        pathFlags: Int = 0,
        pathRisk: Double = 0,
        fillRisk: Double = 0,
        maskedStepTarget: Double = 0,
        nextVolumeTarget: Double = 0,
        regimeShiftTarget: Double = 0,
        contextLeadTarget: Double = 0.5,
        windowSize: Int = 0,
        x: [Double],
        xWindow: [[Double]] = []
    ) {
        self.valid = valid
        self.context = context
        self.labelClass = labelClass
        self.movePoints = movePoints
        self.sampleWeight = sampleWeight
        self.mfePoints = mfePoints
        self.maePoints = maePoints
        self.timeToHitFraction = timeToHitFraction
        self.pathFlags = pathFlags
        self.pathRisk = pathRisk
        self.fillRisk = fillRisk
        self.maskedStepTarget = maskedStepTarget
        self.nextVolumeTarget = nextVolumeTarget
        self.regimeShiftTarget = regimeShiftTarget
        self.contextLeadTarget = contextLeadTarget
        self.windowSize = min(max(0, windowSize), FXDataEngineConstants.maxSequenceBars)
        self.x = x
        self.xWindow = xWindow
    }

    public func validate() throws {
        guard valid else { throw FXDataEngineError.validation("req.valid") }
        try context.validate()
        try PredictRequestV4.validateInput(x)
        try PredictRequestV4.validateWindow(xWindow, windowSize: windowSize, context: context)
        guard movePoints.isFinite else { throw FXDataEngineError.validation("req.movePoints") }
        guard sampleWeight.isFinite, sampleWeight >= 0 else { throw FXDataEngineError.validation("req.sampleWeight") }
        guard mfePoints.isFinite, mfePoints >= 0, maePoints.isFinite, maePoints >= 0 else {
            throw FXDataEngineError.validation("req.pathExcursions")
        }
        guard timeToHitFraction.isFinite, (0...1).contains(timeToHitFraction) else {
            throw FXDataEngineError.validation("req.timeToHitFraction")
        }
        guard pathRisk.isFinite, (0...1).contains(pathRisk) else { throw FXDataEngineError.validation("req.pathRisk") }
        guard fillRisk.isFinite, (0...1).contains(fillRisk) else { throw FXDataEngineError.validation("req.fillRisk") }
        guard maskedStepTarget.isFinite else { throw FXDataEngineError.validation("req.maskedStepTarget") }
        guard nextVolumeTarget.isFinite, nextVolumeTarget >= 0 else { throw FXDataEngineError.validation("req.nextVolumeTarget") }
        guard regimeShiftTarget.isFinite, (0...1).contains(regimeShiftTarget) else {
            throw FXDataEngineError.validation("req.regimeShiftTarget")
        }
        guard contextLeadTarget.isFinite, (0...1).contains(contextLeadTarget) else {
            throw FXDataEngineError.validation("req.contextLeadTarget")
        }
    }

    public func replacingPayload(x: [Double], xWindow: [[Double]]) -> TrainRequestV4 {
        TrainRequestV4(
            valid: valid,
            context: context,
            labelClass: labelClass,
            movePoints: movePoints,
            sampleWeight: sampleWeight,
            mfePoints: mfePoints,
            maePoints: maePoints,
            timeToHitFraction: timeToHitFraction,
            pathFlags: pathFlags,
            pathRisk: pathRisk,
            fillRisk: fillRisk,
            maskedStepTarget: maskedStepTarget,
            nextVolumeTarget: nextVolumeTarget,
            regimeShiftTarget: regimeShiftTarget,
            contextLeadTarget: contextLeadTarget,
            windowSize: xWindow.count,
            x: x,
            xWindow: xWindow
        )
    }
}

public struct PredictionV4: Codable, Hashable, Sendable {
    public var classProbabilities: [Double]
    public var moveMeanPoints: Double
    public var moveQ25Points: Double
    public var moveQ50Points: Double
    public var moveQ75Points: Double
    public var mfeMeanPoints: Double
    public var maeMeanPoints: Double
    public var hitTimeFraction: Double
    public var pathRisk: Double
    public var fillRisk: Double
    public var confidence: Double
    public var reliability: Double

    public init(
        classProbabilities: [Double] = [0.1, 0.1, 0.8],
        moveMeanPoints: Double = 0,
        moveQ25Points: Double = 0,
        moveQ50Points: Double = 0,
        moveQ75Points: Double = 0,
        mfeMeanPoints: Double = 0,
        maeMeanPoints: Double = 0,
        hitTimeFraction: Double = 1,
        pathRisk: Double = 0,
        fillRisk: Double = 0,
        confidence: Double = 0,
        reliability: Double = 0
    ) {
        self.classProbabilities = classProbabilities
        self.moveMeanPoints = moveMeanPoints
        self.moveQ25Points = moveQ25Points
        self.moveQ50Points = moveQ50Points
        self.moveQ75Points = moveQ75Points
        self.mfeMeanPoints = mfeMeanPoints
        self.maeMeanPoints = maeMeanPoints
        self.hitTimeFraction = hitTimeFraction
        self.pathRisk = pathRisk
        self.fillRisk = fillRisk
        self.confidence = confidence
        self.reliability = reliability
    }

    public func validate() throws {
        guard classProbabilities.count == 3,
              classProbabilities.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw FXDataEngineError.validation("pred.classProbabilities")
        }
        let sum = classProbabilities.reduce(0, +)
        guard sum > 0, abs(sum - 1.0) <= 0.02 else {
            throw FXDataEngineError.validation("pred.classSum")
        }
        guard moveMeanPoints.isFinite, moveMeanPoints >= 0 else { throw FXDataEngineError.validation("pred.moveMean") }
        guard moveQ25Points.isFinite,
              moveQ50Points.isFinite,
              moveQ75Points.isFinite,
              moveQ25Points >= 0,
              moveQ50Points >= moveQ25Points,
              moveQ75Points >= moveQ50Points else {
            throw FXDataEngineError.validation("pred.moveQuantiles")
        }
        guard mfeMeanPoints.isFinite, mfeMeanPoints >= 0, maeMeanPoints.isFinite, maeMeanPoints >= 0 else {
            throw FXDataEngineError.validation("pred.pathExcursions")
        }
        guard hitTimeFraction.isFinite, (0...1).contains(hitTimeFraction) else {
            throw FXDataEngineError.validation("pred.hitTimeFraction")
        }
        guard pathRisk.isFinite, (0...1).contains(pathRisk) else { throw FXDataEngineError.validation("pred.pathRisk") }
        guard fillRisk.isFinite, (0...1).contains(fillRisk) else { throw FXDataEngineError.validation("pred.fillRisk") }
        guard confidence.isFinite, (0...1).contains(confidence) else { throw FXDataEngineError.validation("pred.confidence") }
        guard reliability.isFinite, (0...1).contains(reliability) else { throw FXDataEngineError.validation("pred.reliability") }
    }

    public static var skip: PredictionV4 {
        PredictionV4(classProbabilities: [0.0, 0.0, 1.0])
    }
}

public protocol FXAIPluginV4: Sendable {
    var manifest: PluginManifestV4 { get }
    mutating func reset()
    func selfTest() -> Bool
    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws
    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4
}

public extension FXAIPluginV4 {
    func selfTest() -> Bool {
        do {
            try manifest.validate()
            return true
        } catch {
            return false
        }
    }
}

public enum PluginContractTools {
    public static func symbolHash01(_ symbol: String) -> Double {
        let bytes = Array(symbol.uppercased().utf8)
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in bytes {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return Double(hash % 1_000_000) / 1_000_000.0
    }

    public static func deriveSessionBucket(timestampUTC: Int64) -> Int {
        guard timestampUTC > 0 else { return 0 }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let hour = calendar.component(.hour, from: Date(timeIntervalSince1970: TimeInterval(timestampUTC)))
        switch hour {
        case 0..<4: return 0
        case 4..<8: return 1
        case 8..<12: return 2
        case 12..<16: return 3
        case 16..<20: return 4
        default: return 5
        }
    }
}
