import Foundation

public enum PluginSupportTools {
    public static func clipSymmetric(_ value: Double, limit: Double) -> Double {
        let safeLimit = max(0.0, abs(fxSafeFinite(limit)))
        guard safeLimit > 0.0 else { return fxSafeFinite(value) }
        return fxClamp(value, -safeLimit, safeLimit)
    }

    public static func sigmoid(_ value: Double) -> Double {
        let z = fxSafeFinite(value)
        if z > 35.0 { return 1.0 }
        if z < -35.0 { return 0.0 }
        return 1.0 / (1.0 + exp(-z))
    }

    public static func logit(_ probability: Double) -> Double {
        let p = fxClamp(fxSafeFinite(probability, fallback: 0.5), 1e-6, 1.0 - 1e-6)
        return log(p / (1.0 - p))
    }

    public static func moveWeight(movePoints: Double) -> Double {
        fxClamp(1.0 + 0.05 * abs(fxSafeFinite(movePoints)), 0.80, 3.00)
    }

    public static func moveEdgeWeight(movePoints: Double, priceCostPoints: Double) -> Double {
        let move = abs(fxSafeFinite(movePoints))
        let priceCost = max(0.0, fxSafeFinite(priceCostPoints))
        let edge = move - priceCost
        let denominator = max(priceCost, 1.0)
        return fxClamp(0.50 + edge / denominator, 0.25, 4.00)
    }

    public static func scaleHyperParametersForMove(_ hyperParameters: HyperParameters, movePoints: Double) -> HyperParameters {
        var output = hyperParameters
        let weight = moveWeight(movePoints: movePoints)
        output.learningRate = fxClamp(output.learningRate * weight, 0.0001, 1.0000)
        output.ftrlAlpha = fxClamp(output.ftrlAlpha * weight, 0.0001, 5.0000)
        output.xgbLearningRate = fxClamp(output.xgbLearningRate * weight, 0.0001, 1.0000)
        output.mlpLearningRate = fxClamp(output.mlpLearningRate * weight, 0.0001, 1.0000)
        output.quantileLearningRate = fxClamp(output.quantileLearningRate * weight, 0.0001, 1.0000)
        output.enhashLearningRate = fxClamp(output.enhashLearningRate * weight, 0.0001, 1.0000)
        return output
    }

    public static func moveSampleWeight(
        x: [Double],
        movePoints: Double,
        priceCostPoints: Double? = nil,
        minMovePoints: Double = 0.0,
        qualityTargets: PluginQualityTargets? = nil
    ) -> Double {
        let priceCost = PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: priceCostPoints)
        let baseWeight = moveEdgeWeight(movePoints: movePoints, priceCostPoints: priceCost)
        guard let qualityTargets else { return baseWeight }

        let moveScale = max(abs(fxSafeFinite(movePoints)), max(fxSafeFinite(minMovePoints), 0.10))
        let mfeBonus = fxClamp(qualityTargets.mfePoints / max(moveScale, 0.10), 0.0, 2.0)
        let maePenalty = fxClamp(
            qualityTargets.maePoints / max(max(qualityTargets.mfePoints, moveScale), 0.10),
            0.0,
            1.5
        )
        let timingBonus = 1.0 - fxClamp(qualityTargets.hitTimeFraction, 0.0, 1.0)
        let executionDrag = 0.60 * fxClamp(qualityTargets.pathRisk, 0.0, 1.0) +
            0.40 * fxClamp(qualityTargets.fillRisk, 0.0, 1.0)
        let quality = 1.0 + 0.18 * mfeBonus + 0.14 * timingBonus - 0.16 * maePenalty - 0.18 * executionDrag
        return baseWeight * fxClamp(quality, 0.45, 1.85)
    }

    public static func computeReplayPriority(
        rawLabelClass: Int,
        probabilities: [Double],
        movePoints: Double,
        priceCostPoints: Double,
        minMovePoints: Double
    ) -> Double {
        let label = LabelClass(rawValue: rawLabelClass) ?? (movePoints >= 0.0 ? .buy : .sell)
        let probability = label.rawValue < probabilities.count ?
            fxSafeFinite(probabilities[label.rawValue], fallback: 0.3333333) :
            0.3333333
        let edge = max(abs(fxSafeFinite(movePoints)) - max(fxSafeFinite(priceCostPoints), 0.0), 0.0)
        let minimumMove = max(fxSafeFinite(minMovePoints), 0.10)
        var priority = 0.50 + (1.0 - fxClamp(probability, 0.0, 1.0))
        priority += 0.35 * fxClamp(edge / minimumMove, 0.0, 4.0)
        if label == .skip {
            priority += 0.15
        }
        return fxClamp(priority, 0.10, 8.00)
    }
}

public struct PluginDeterministicRNG: Codable, Hashable, Sendable {
    public private(set) var state: UInt32

    public init(aiID: Int) {
        var seed = UInt32(truncatingIfNeeded: max(0, aiID) + 1)
        seed = seed &* 747_796_405 &+ 2_891_336_453
        if seed == 0 {
            seed = 2_463_534_242
        }
        self.state = seed
    }

    public init(state: UInt32) {
        self.state = state == 0 ? 2_463_534_242 : state
    }

    public mutating func next01() -> Double {
        state = 1_664_525 &* state &+ 1_013_904_223
        return fxClamp((Double(state) + 0.5) / 4_294_967_296.0, 0.0, 1.0)
    }

    public mutating func nextIndex(_ count: Int) -> Int {
        guard count > 0 else { return -1 }
        let index = Int(floor(next01() * Double(count)))
        return min(max(0, index), count - 1)
    }
}

public struct PluginTernaryCalibrator: Codable, Hashable, Sendable {
    public static let classCount = 3
    public static let binCount = 12

    public private(set) var steps: Int
    public var weights: [[Double]]
    public var biases: [Double]
    public var isotonicPositive: [[Double]]
    public var isotonicCount: [[Double]]

    public init(
        steps: Int = 0,
        weights: [[Double]] = PluginTernaryCalibrator.identityWeights(),
        biases: [Double] = Array(repeating: 0.0, count: PluginTernaryCalibrator.classCount),
        isotonicPositive: [[Double]] = PluginTernaryCalibrator.zeroBins(),
        isotonicCount: [[Double]] = PluginTernaryCalibrator.zeroBins()
    ) {
        self.steps = max(0, steps)
        self.weights = Self.normalizedMatrix(weights, columns: Self.classCount, identityFallback: true)
        self.biases = Self.normalizedVector(biases, count: Self.classCount)
        self.isotonicPositive = Self.normalizedMatrix(isotonicPositive, columns: Self.binCount, identityFallback: false)
        self.isotonicCount = Self.normalizedMatrix(isotonicCount, columns: Self.binCount, identityFallback: false)
    }

    public mutating func reset() {
        self = PluginTernaryCalibrator()
    }

    public func calibrated(_ rawProbabilities: [Double]) -> [Double] {
        let logits = buildLogits(rawProbabilities)
        var calibrated = Self.softmax3(logits)
        guard steps >= 30 else {
            return calibrated
        }

        var isotonic = calibrated
        for classIndex in 0..<Self.classCount {
            let total = isotonicCount[classIndex].reduce(0.0, +)
            guard total >= 40.0 else {
                isotonic[classIndex] = calibrated[classIndex]
                continue
            }

            var monotonic = Array(repeating: 0.0, count: Self.binCount)
            var previous = 0.01
            for bin in 0..<Self.binCount {
                var ratio = previous
                if isotonicCount[classIndex][bin] > 1e-9 {
                    ratio = isotonicPositive[classIndex][bin] / isotonicCount[classIndex][bin]
                }
                ratio = fxClamp(ratio, 0.001, 0.999)
                if ratio < previous {
                    ratio = previous
                }
                monotonic[bin] = ratio
                previous = ratio
            }

            let bin = Self.probabilityBin(calibrated[classIndex])
            isotonic[classIndex] = monotonic[bin]
        }

        for classIndex in 0..<Self.classCount {
            calibrated[classIndex] = fxClamp(0.75 * calibrated[classIndex] + 0.25 * isotonic[classIndex], 0.0005, 0.9990)
        }
        let sum = calibrated.reduce(0.0, +)
        guard sum > 0.0 else {
            return [0.10, 0.10, 0.80]
        }
        return calibrated.map { $0 / sum }
    }

    public mutating func update(
        rawProbabilities: [Double],
        labelClass: LabelClass,
        sampleWeight: Double,
        learningRate: Double
    ) {
        let logits = buildLogits(rawProbabilities)
        let calibrated = Self.softmax3(logits)
        let logRaw = Self.logRawProbabilities(rawProbabilities)
        let weight = fxClamp(sampleWeight, 0.20, 8.00)
        let calibratorLearningRate = fxClamp(0.25 * learningRate * weight, 0.0002, 0.0200)
        let l2 = 0.0005

        for classIndex in 0..<Self.classCount {
            let target = classIndex == labelClass.rawValue ? 1.0 : 0.0
            let error = target - calibrated[classIndex]

            biases[classIndex] = Self.clipSymmetric(biases[classIndex] + calibratorLearningRate * error, limit: 4.0)
            for rawIndex in 0..<Self.classCount {
                let targetWeight = classIndex == rawIndex ? 1.0 : 0.0
                let gradient = error * logRaw[rawIndex] - l2 * (weights[classIndex][rawIndex] - targetWeight)
                weights[classIndex][rawIndex] = Self.clipSymmetric(
                    weights[classIndex][rawIndex] + calibratorLearningRate * gradient,
                    limit: 4.0
                )
            }

            let bin = Self.probabilityBin(calibrated[classIndex])
            isotonicCount[classIndex][bin] += weight
            isotonicPositive[classIndex][bin] += weight * target
        }
        steps += 1
    }

    private func buildLogits(_ rawProbabilities: [Double]) -> [Double] {
        let logRaw = Self.logRawProbabilities(rawProbabilities)
        var logits = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            var value = biases[classIndex]
            for rawIndex in 0..<Self.classCount {
                value += weights[classIndex][rawIndex] * logRaw[rawIndex]
            }
            logits[classIndex] = value
        }
        return logits
    }

    private static func logRawProbabilities(_ rawProbabilities: [Double]) -> [Double] {
        (0..<classCount).map { index in
            let raw = index < rawProbabilities.count ? rawProbabilities[index] : 0.0
            return log(fxClamp(fxSafeFinite(raw), 0.0005, 0.9990))
        }
    }

    private static func softmax3(_ logits: [Double]) -> [Double] {
        let safeLogits = (0..<classCount).map { index -> Double in
            index < logits.count ? fxSafeFinite(logits[index]) : 0.0
        }
        let maximum = safeLogits.max() ?? 0.0
        var exponentials = Array(repeating: 0.0, count: classCount)
        var denominator = 0.0
        for classIndex in 0..<classCount {
            let value = exp(clipSymmetric(safeLogits[classIndex] - maximum, limit: 30.0))
            exponentials[classIndex] = value
            denominator += value
        }
        guard denominator > 0.0 else {
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        }
        return exponentials.map { $0 / denominator }
    }

    private static func probabilityBin(_ probability: Double) -> Int {
        let raw = Int(floor(fxClamp(fxSafeFinite(probability), 0.0, 0.999999) * Double(binCount)))
        return min(max(0, raw), binCount - 1)
    }

    private static func clipSymmetric(_ value: Double, limit: Double) -> Double {
        PluginSupportTools.clipSymmetric(value, limit: limit)
    }

    public static func identityWeights() -> [[Double]] {
        (0..<classCount).map { row in
            (0..<classCount).map { column in row == column ? 1.0 : 0.0 }
        }
    }

    public static func zeroBins() -> [[Double]] {
        Array(repeating: Array(repeating: 0.0, count: binCount), count: classCount)
    }

    private static func normalizedVector(_ values: [Double], count: Int) -> [Double] {
        (0..<count).map { index in
            index < values.count ? fxSafeFinite(values[index]) : 0.0
        }
    }

    private static func normalizedMatrix(_ values: [[Double]], columns: Int, identityFallback: Bool) -> [[Double]] {
        (0..<classCount).map { row in
            (0..<columns).map { column in
                if row < values.count, column < values[row].count {
                    return fxSafeFinite(values[row][column])
                }
                return identityFallback && row == column ? 1.0 : 0.0
            }
        }
    }
}

public struct PluginBinaryCalibrator: Codable, Hashable, Sendable {
    public static let binCount = 12

    public private(set) var steps: Int
    public var scale: Double
    public var bias: Double
    public var isotonicPositive: [Double]
    public var isotonicCount: [Double]

    public var ready: Bool {
        steps >= 20
    }

    public init(
        steps: Int = 0,
        scale: Double = 1.0,
        bias: Double = 0.0,
        isotonicPositive: [Double] = Array(repeating: 0.0, count: PluginBinaryCalibrator.binCount),
        isotonicCount: [Double] = Array(repeating: 0.0, count: PluginBinaryCalibrator.binCount)
    ) {
        self.steps = max(0, steps)
        self.scale = fxClamp(fxSafeFinite(scale, fallback: 1.0), 0.20, 5.00)
        self.bias = fxClamp(fxSafeFinite(bias), -4.0, 4.0)
        self.isotonicPositive = Self.normalizedBins(isotonicPositive)
        self.isotonicCount = Self.normalizedBins(isotonicCount)
    }

    public mutating func reset() {
        self = PluginBinaryCalibrator()
    }

    public func calibrated(_ rawProbability: Double) -> Double {
        let raw = fxClamp(fxSafeFinite(rawProbability, fallback: 0.5), 0.001, 0.999)
        let platt = PluginSupportTools.sigmoid(scale * PluginSupportTools.logit(raw) + bias)
        guard ready else {
            return fxClamp(platt, 0.001, 0.999)
        }

        let total = isotonicCount.reduce(0.0, +)
        guard total >= 30.0 else {
            return fxClamp(platt, 0.001, 0.999)
        }

        var monotonic = Array(repeating: 0.0, count: Self.binCount)
        var previous = 0.5
        for bin in 0..<Self.binCount {
            var ratio = previous
            if isotonicCount[bin] > 1e-9 {
                ratio = isotonicPositive[bin] / isotonicCount[bin]
            }
            ratio = fxClamp(ratio, 0.01, 0.99)
            if bin > 0, ratio < monotonic[bin - 1] {
                ratio = monotonic[bin - 1]
            }
            monotonic[bin] = ratio
            previous = ratio
        }

        let isotonic = monotonic[Self.probabilityBin(raw)]
        return fxClamp(0.70 * platt + 0.30 * isotonic, 0.001, 0.999)
    }

    public mutating func update(rawProbability: Double, target: Bool, sampleWeight: Double = 1.0) {
        update(rawProbability: rawProbability, target: target ? 1.0 : 0.0, sampleWeight: sampleWeight)
    }

    public mutating func update(rawProbability: Double, target: Double, sampleWeight: Double = 1.0) {
        let raw = fxClamp(fxSafeFinite(rawProbability, fallback: 0.5), 0.001, 0.999)
        let z = PluginSupportTools.logit(raw)
        let predicted = PluginSupportTools.sigmoid(scale * z + bias)
        let clippedTarget = fxClamp(target, 0.0, 1.0)
        let error = clippedTarget - predicted
        let weight = fxClamp(sampleWeight, 0.25, 4.00)
        let learningRate = 0.015 * weight
        let regularization = 0.0005

        scale += learningRate * (error * z - regularization * (scale - 1.0))
        bias += learningRate * error
        scale = fxClamp(scale, 0.20, 5.00)
        bias = fxClamp(bias, -4.0, 4.0)

        let bin = Self.probabilityBin(raw)
        isotonicCount[bin] += weight
        isotonicPositive[bin] += weight * clippedTarget
        steps += 1
    }

    private static func probabilityBin(_ probability: Double) -> Int {
        let raw = Int(floor(fxClamp(fxSafeFinite(probability), 0.0, 0.999999) * Double(binCount)))
        return min(max(0, raw), binCount - 1)
    }

    private static func normalizedBins(_ values: [Double]) -> [Double] {
        (0..<binCount).map { index in
            index < values.count ? max(0.0, fxSafeFinite(values[index])) : 0.0
        }
    }
}

public struct PluginMoveHead: Codable, Hashable, Sendable {
    public private(set) var steps: Int
    public var weights: [Double]

    public var ready: Bool {
        steps >= 16
    }

    public init(steps: Int = 0, weights: [Double] = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)) {
        self.steps = max(0, steps)
        self.weights = (0..<FXDataEngineConstants.aiWeights).map { index in
            index < weights.count ? fxSafeFinite(weights[index]) : 0.0
        }
    }

    public mutating func reset() {
        self = PluginMoveHead()
    }

    public func predictRaw(_ x: [Double]) -> Double {
        max(dot(x), 0.0)
    }

    public mutating func update(
        x: [Double],
        movePoints: Double,
        hyperParameters: HyperParameters,
        sampleWeight: Double
    ) {
        let target = abs(fxSafeFinite(movePoints, fallback: .nan))
        guard target.isFinite else { return }

        let prediction = predictRaw(x)
        let error = PluginSupportTools.clipSymmetric(target - prediction, limit: 20.0)
        let learningRate = fxClamp(0.08 * hyperParameters.learningRate, 0.00005, 0.02000)
        let l2 = fxClamp(0.25 * hyperParameters.l2, 0.0000, 0.1000)
        let weight = fxClamp(sampleWeight, 0.25, 4.00)

        weights[0] += learningRate * weight * error
        for index in 1..<FXDataEngineConstants.aiWeights {
            let value = index < x.count ? fxSafeFinite(x[index]) : 0.0
            let gradient = weight * PluginSupportTools.clipSymmetric(error * value, limit: 6.0) - l2 * weights[index]
            weights[index] += learningRate * gradient
        }
        steps += 1
    }

    private func dot(_ x: [Double]) -> Double {
        var value = 0.0
        for index in 0..<FXDataEngineConstants.aiWeights {
            let input = index < x.count ? fxSafeFinite(x[index]) : 0.0
            value += weights[index] * input
        }
        return value
    }
}

public struct PluginQualityTargets: Codable, Hashable, Sendable {
    public var mfePoints: Double
    public var maePoints: Double
    public var hitTimeFraction: Double
    public var pathFlags: Int
    public var pathRisk: Double
    public var fillRisk: Double
    public var maskedStepTarget: Double
    public var nextVolumeTarget: Double
    public var regimeShiftTarget: Double
    public var contextLeadTarget: Double

    public init(
        mfePoints: Double = 0.0,
        maePoints: Double = 0.0,
        hitTimeFraction: Double = 1.0,
        pathFlags: Int = 0,
        pathRisk: Double = 0.0,
        fillRisk: Double = 0.0,
        maskedStepTarget: Double = 0.0,
        nextVolumeTarget: Double = 0.0,
        regimeShiftTarget: Double = 0.0,
        contextLeadTarget: Double = 0.5
    ) {
        self.mfePoints = max(fxSafeFinite(mfePoints), 0.0)
        self.maePoints = max(fxSafeFinite(maePoints), 0.0)
        self.hitTimeFraction = fxClamp(fxSafeFinite(hitTimeFraction, fallback: 1.0), 0.0, 1.0)
        self.pathFlags = pathFlags
        self.pathRisk = fxClamp(fxSafeFinite(pathRisk), 0.0, 1.0)
        self.fillRisk = fxClamp(fxSafeFinite(fillRisk), 0.0, 1.0)
        self.maskedStepTarget = fxSafeFinite(maskedStepTarget)
        self.nextVolumeTarget = max(fxSafeFinite(nextVolumeTarget), 0.0)
        self.regimeShiftTarget = fxClamp(fxSafeFinite(regimeShiftTarget), 0.0, 1.0)
        self.contextLeadTarget = fxClamp(fxSafeFinite(contextLeadTarget, fallback: 0.5), 0.0, 1.0)
    }

    public init(request: TrainRequestV4) {
        self.init(
            mfePoints: request.mfePoints,
            maePoints: request.maePoints,
            hitTimeFraction: request.timeToHitFraction,
            pathFlags: request.pathFlags,
            pathRisk: request.pathRisk,
            fillRisk: request.fillRisk,
            maskedStepTarget: request.maskedStepTarget,
            nextVolumeTarget: request.nextVolumeTarget,
            regimeShiftTarget: request.regimeShiftTarget,
            contextLeadTarget: request.contextLeadTarget
        )
    }
}

public struct PluginContextBucket: Codable, Hashable, Sendable {
    public var regimeID: Int
    public var sessionBucket: Int
    public var horizonBucket: Int

    public init(regimeID: Int, sessionBucket: Int, horizonBucket: Int) {
        self.regimeID = Int(fxClamp(Double(regimeID), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))
        self.sessionBucket = Int(fxClamp(Double(sessionBucket), 0.0, Double(FXDataEngineConstants.pluginSessionBuckets - 1)))
        self.horizonBucket = Int(fxClamp(Double(horizonBucket), 0.0, Double(FXDataEngineConstants.pluginHorizonBuckets - 1)))
    }

    public init(context: PluginContextV4) {
        self.init(
            regimeID: context.regimeID,
            sessionBucket: context.sessionBucket,
            horizonBucket: PluginContextRuntimeTools.contextHorizonBucket(horizonMinutes: context.horizonMinutes)
        )
    }
}

public struct PluginQualityBankPriors: Codable, Hashable, Sendable {
    public var mfePoints: Double
    public var maePoints: Double
    public var hitTimeFraction: Double
    public var pathRisk: Double
    public var fillRisk: Double
    public var trust: Double

    public init(
        mfePoints: Double = 0.0,
        maePoints: Double = 0.0,
        hitTimeFraction: Double = 1.0,
        pathRisk: Double = 0.5,
        fillRisk: Double = 0.5,
        trust: Double = 0.0
    ) {
        self.mfePoints = max(fxSafeFinite(mfePoints), 0.0)
        self.maePoints = max(fxSafeFinite(maePoints), 0.0)
        self.hitTimeFraction = fxClamp(fxSafeFinite(hitTimeFraction, fallback: 1.0), 0.0, 1.0)
        self.pathRisk = fxClamp(fxSafeFinite(pathRisk, fallback: 0.5), 0.0, 1.0)
        self.fillRisk = fxClamp(fxSafeFinite(fillRisk, fallback: 0.5), 0.0, 1.0)
        self.trust = fxClamp(fxSafeFinite(trust), 0.0, 1.0)
    }
}

public struct PluginQualityBankCell: Codable, Hashable, Sendable {
    public var ready: Bool
    public var observations: Double
    public var mfePoints: Double
    public var maePoints: Double
    public var hitTimeFraction: Double
    public var pathRisk: Double
    public var fillRisk: Double

    public init(
        ready: Bool = false,
        observations: Double = 0.0,
        mfePoints: Double = 0.0,
        maePoints: Double = 0.0,
        hitTimeFraction: Double = 1.0,
        pathRisk: Double = 0.5,
        fillRisk: Double = 0.5
    ) {
        self.ready = ready
        self.observations = max(fxSafeFinite(observations), 0.0)
        self.mfePoints = max(fxSafeFinite(mfePoints), 0.0)
        self.maePoints = max(fxSafeFinite(maePoints), 0.0)
        self.hitTimeFraction = fxClamp(fxSafeFinite(hitTimeFraction, fallback: 1.0), 0.0, 1.0)
        self.pathRisk = fxClamp(fxSafeFinite(pathRisk, fallback: 0.5), 0.0, 1.0)
        self.fillRisk = fxClamp(fxSafeFinite(fillRisk, fallback: 0.5), 0.0, 1.0)
    }
}

public struct PluginQualityBank: Codable, Hashable, Sendable {
    public var ready: Bool
    public var mfeEMA: Double
    public var maeEMA: Double
    public var hitEMA: Double
    public var pathRiskEMA: Double
    public var fillRiskEMA: Double
    public var cells: [PluginContextBucket: PluginQualityBankCell]

    public init(
        ready: Bool = false,
        mfeEMA: Double = 0.0,
        maeEMA: Double = 0.0,
        hitEMA: Double = 1.0,
        pathRiskEMA: Double = 0.5,
        fillRiskEMA: Double = 0.5,
        cells: [PluginContextBucket: PluginQualityBankCell] = [:]
    ) {
        self.ready = ready
        self.mfeEMA = max(fxSafeFinite(mfeEMA), 0.0)
        self.maeEMA = max(fxSafeFinite(maeEMA), 0.0)
        self.hitEMA = fxClamp(fxSafeFinite(hitEMA, fallback: 1.0), 0.0, 1.0)
        self.pathRiskEMA = fxClamp(fxSafeFinite(pathRiskEMA, fallback: 0.5), 0.0, 1.0)
        self.fillRiskEMA = fxClamp(fxSafeFinite(fillRiskEMA, fallback: 0.5), 0.0, 1.0)
        self.cells = cells
    }

    public mutating func reset() {
        self = PluginQualityBank()
    }

    public mutating func update(request: TrainRequestV4, sampleWeight: Double) {
        update(targets: PluginQualityTargets(request: request), context: request.context, sampleWeight: sampleWeight)
    }

    public mutating func update(targets: PluginQualityTargets, context: PluginContextV4, sampleWeight: Double) {
        let boundedWeight = fxClamp(sampleWeight, 0.25, 4.0)
        let alpha = fxClamp(0.06 * boundedWeight, 0.01, 0.20)
        if !ready {
            mfeEMA = targets.mfePoints
            maeEMA = targets.maePoints
            hitEMA = targets.hitTimeFraction
            pathRiskEMA = targets.pathRisk
            fillRiskEMA = targets.fillRisk
            ready = true
            return
        }

        mfeEMA = (1.0 - alpha) * mfeEMA + alpha * targets.mfePoints
        maeEMA = (1.0 - alpha) * maeEMA + alpha * targets.maePoints
        hitEMA = (1.0 - alpha) * hitEMA + alpha * targets.hitTimeFraction
        pathRiskEMA = (1.0 - alpha) * pathRiskEMA + alpha * targets.pathRisk
        fillRiskEMA = (1.0 - alpha) * fillRiskEMA + alpha * targets.fillRisk

        let bucket = PluginContextBucket(context: context)
        var cell = cells[bucket] ?? PluginQualityBankCell()
        let observations = cell.observations
        let bankAlpha = fxClamp(0.12 * boundedWeight / sqrt(1.0 + 0.02 * observations), 0.02, 0.25)
        if !cell.ready {
            cell.mfePoints = targets.mfePoints
            cell.maePoints = targets.maePoints
            cell.hitTimeFraction = targets.hitTimeFraction
            cell.pathRisk = targets.pathRisk
            cell.fillRisk = targets.fillRisk
            cell.ready = true
        } else {
            cell.mfePoints = (1.0 - bankAlpha) * cell.mfePoints + bankAlpha * targets.mfePoints
            cell.maePoints = (1.0 - bankAlpha) * cell.maePoints + bankAlpha * targets.maePoints
            cell.hitTimeFraction = (1.0 - bankAlpha) * cell.hitTimeFraction + bankAlpha * targets.hitTimeFraction
            cell.pathRisk = (1.0 - bankAlpha) * cell.pathRisk + bankAlpha * targets.pathRisk
            cell.fillRisk = (1.0 - bankAlpha) * cell.fillRisk + bankAlpha * targets.fillRisk
        }
        cell.observations = min(observations + boundedWeight, 50_000.0)
        cells[bucket] = cell
    }

    public func priors(context: PluginContextV4) -> PluginQualityBankPriors {
        var output = PluginQualityBankPriors(
            mfePoints: mfeEMA,
            maePoints: maeEMA,
            hitTimeFraction: hitEMA,
            pathRisk: pathRiskEMA,
            fillRisk: fillRiskEMA,
            trust: ready ? 0.35 : 0.0
        )
        let bucket = PluginContextBucket(context: context)
        guard let cell = cells[bucket], cell.ready else {
            return output
        }
        let bankTrust = fxClamp(cell.observations / 120.0, 0.10, 0.85)
        output.mfePoints = (1.0 - bankTrust) * output.mfePoints + bankTrust * cell.mfePoints
        output.maePoints = (1.0 - bankTrust) * output.maePoints + bankTrust * cell.maePoints
        output.hitTimeFraction = (1.0 - bankTrust) * output.hitTimeFraction + bankTrust * cell.hitTimeFraction
        output.pathRisk = (1.0 - bankTrust) * output.pathRisk + bankTrust * cell.pathRisk
        output.fillRisk = (1.0 - bankTrust) * output.fillRisk + bankTrust * cell.fillRisk
        output.trust = fxClamp(output.trust + 0.65 * bankTrust, 0.0, 1.0)
        return output
    }
}

public struct PluginContextCalibrationCell: Codable, Hashable, Sendable {
    public var total: Double
    public var classMass: [Double]
    public var expectedMoveScale: Double
    public var expectedMoveBias: Double
    public var scaleGradientSumSquared: Double
    public var biasGradientSumSquared: Double

    public init(
        total: Double = 0.0,
        classMass: [Double] = [1.0, 1.0, 1.2],
        expectedMoveScale: Double = 1.0,
        expectedMoveBias: Double = 0.0,
        scaleGradientSumSquared: Double = 0.0,
        biasGradientSumSquared: Double = 0.0
    ) {
        self.total = max(fxSafeFinite(total), 0.0)
        self.classMass = (0..<PluginTernaryCalibrator.classCount).map { index in
            let fallback = index == LabelClass.skip.rawValue ? 1.2 : 1.0
            return index < classMass.count ? max(0.0, fxSafeFinite(classMass[index], fallback: fallback)) : fallback
        }
        self.expectedMoveScale = fxClamp(fxSafeFinite(expectedMoveScale, fallback: 1.0), 0.40, 2.50)
        self.expectedMoveBias = fxClamp(fxSafeFinite(expectedMoveBias), -20.0, 20.0)
        self.scaleGradientSumSquared = max(0.0, fxSafeFinite(scaleGradientSumSquared))
        self.biasGradientSumSquared = max(0.0, fxSafeFinite(biasGradientSumSquared))
    }
}

public struct PluginContextCalibrationBank: Codable, Hashable, Sendable {
    public var cells: [PluginContextBucket: PluginContextCalibrationCell]

    public init(cells: [PluginContextBucket: PluginContextCalibrationCell] = [:]) {
        self.cells = cells
    }

    public mutating func reset() {
        cells.removeAll()
    }

    public func classCalibrated(probabilities: [Double], context: PluginContextV4) -> [Double] {
        let bucket = PluginContextBucket(context: context)
        guard let cell = cells[bucket], cell.total > 0.0 else {
            return PluginContextRuntimeTools.normalizeClassDistribution(probabilities)
        }
        let classMass = Self.normalizedClassMass(cell.classMass)
        let priorTotal = classMass.reduce(0.0, +)
        guard priorTotal > 1e-9 else {
            return PluginContextRuntimeTools.normalizeClassDistribution(probabilities)
        }

        let raw = PluginContextRuntimeTools.normalizeClassDistribution(probabilities)
        let prior = classMass.map { $0 / priorTotal }
        let mix = fxClamp(cell.total / 120.0, 0.05, 0.35)
        let calibrated = (0..<PluginTernaryCalibrator.classCount).map { index in
            (1.0 - mix) * raw[index] + mix * prior[index]
        }
        return PluginContextRuntimeTools.normalizeClassDistribution(calibrated)
    }

    public func expectedMoveCalibrated(_ expectedMovePoints: Double, context: PluginContextV4) -> Double {
        var expectedMove = fxSafeFinite(expectedMovePoints)
        guard expectedMove > 0.0 else { return 0.0 }
        let bucket = PluginContextBucket(context: context)
        let cell = cells[bucket] ?? PluginContextCalibrationCell()
        let scale = fxClamp(fxSafeFinite(cell.expectedMoveScale, fallback: 1.0), 0.40, 2.50)
        let bias = fxClamp(fxSafeFinite(cell.expectedMoveBias), -20.0, 20.0)
        expectedMove = expectedMove * scale + bias
        guard expectedMove.isFinite, expectedMove > 0.0 else { return 0.0 }
        return expectedMove
    }

    public mutating func update(
        labelClass: LabelClass,
        expectedMovePoints: Double,
        movePoints: Double,
        sampleWeight: Double,
        context: PluginContextV4
    ) {
        let bucket = PluginContextBucket(context: context)
        var cell = cells[bucket] ?? PluginContextCalibrationCell()
        cell.classMass = Self.normalizedClassMass(cell.classMass)
        cell.expectedMoveScale = fxClamp(fxSafeFinite(cell.expectedMoveScale, fallback: 1.0), 0.40, 2.50)
        cell.expectedMoveBias = fxClamp(fxSafeFinite(cell.expectedMoveBias), -20.0, 20.0)
        cell.scaleGradientSumSquared = max(0.0, fxSafeFinite(cell.scaleGradientSumSquared))
        cell.biasGradientSumSquared = max(0.0, fxSafeFinite(cell.biasGradientSumSquared))
        let weight = fxClamp(sampleWeight, 0.25, 4.00)

        cell.classMass[labelClass.rawValue] += weight
        cell.total += weight
        if cell.total > 30_000.0 {
            cell.classMass = cell.classMass.map { $0 * 0.5 }
            cell.total *= 0.5
        }

        let minimumMove = max(context.minMovePoints, 0.10)
        let prediction = max(fxSafeFinite(expectedMovePoints), minimumMove)
        let target = max(abs(fxSafeFinite(movePoints)), minimumMove)
        let error = PluginSupportTools.clipSymmetric(target - prediction, limit: 30.0)
        let learningRate = 0.015 * weight
        let scaleGradient = error / max(prediction, 0.25)
        let biasGradient = 0.30 * error

        cell.scaleGradientSumSquared += scaleGradient * scaleGradient
        cell.biasGradientSumSquared += biasGradient * biasGradient
        let scaleLearningRate = learningRate / sqrt(cell.scaleGradientSumSquared + 1e-8)
        let biasLearningRate = learningRate / sqrt(cell.biasGradientSumSquared + 1e-8)

        cell.expectedMoveScale = fxClamp(cell.expectedMoveScale + scaleLearningRate * scaleGradient, 0.40, 2.50)
        cell.expectedMoveBias = fxClamp(cell.expectedMoveBias + biasLearningRate * biasGradient, -20.0, 20.0)
        cells[bucket] = cell
    }

    private static func normalizedClassMass(_ classMass: [Double]) -> [Double] {
        (0..<PluginTernaryCalibrator.classCount).map { index in
            let fallback = index == LabelClass.skip.rawValue ? 1.2 : 1.0
            return index < classMass.count ? max(0.0, fxSafeFinite(classMass[index], fallback: fallback)) : fallback
        }
    }
}
