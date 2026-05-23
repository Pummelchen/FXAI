import FXDataEngine
import Foundation

public struct LinEnhashCPUModel: Sendable {
    private struct HyperState: Sendable {
        let alpha: Double
        let beta: Double
        let l1: Double
        let l2: Double
    }

    private struct EvalResult: Sendable {
        let probabilities: [Double]
        let directionRaw: Double
        let skipProbability: Double
        let interactionAmplitude: Double
        let interactionStd: Double
        let collisionMetric: Double
    }

    private static let classCount = 3
    private static let tableCount = 2
    private static let fieldCount = 5
    private static let fieldPairCount = fieldCount * fieldCount
    private static let sessionCount = 4
    private static let regimeCount = 3
    private static let hashBuckets = 128
    private static let rehashPeriod = 512
    private static let calibrationBins = 10

    private var linearZ: [[Double]]
    private var linearN: [[Double]]
    private var linearW: [[Double]]
    private var hashZ: [Double]
    private var hashN: [Double]
    private var hashW: [Double]
    private var bucketUseEMA: [Double]
    private var pairCollisionEMA: [Double]
    private var bias: [Double]
    private var biasG2: [Double]
    private var normReady: Bool
    private var normSteps: Int
    private var featureMean: [Double]
    private var featureVariance: [Double]
    private var step: Int
    private var seedA: UInt32
    private var seedB: UInt32
    private var collisionDiagnosticEMA: Double
    private var calibrationErrorEMA: Double
    private var edgeHitEMA: Double
    private var uncertaintyEMA: Double
    private var interactionCount: Int
    private var interactionMean: Double
    private var interactionM2: Double
    private var moveReady: Bool
    private var moveEMAAbs: Double
    private var calibrationTemperature: Double
    private var calibrationBias: [Double]
    private var isotonicPositive: [[Double]]
    private var isotonicCount: [[Double]]
    private var calibrationSteps: Int

    public init() {
        let featureCount = FXDataEngineConstants.aiWeights
        self.linearZ = Array(repeating: Array(repeating: 0.0, count: featureCount), count: Self.classCount)
        self.linearN = Array(repeating: Array(repeating: 0.0, count: featureCount), count: Self.classCount)
        self.linearW = Array(repeating: Array(repeating: 0.0, count: featureCount), count: Self.classCount)
        let hashSize = Self.classCount * Self.tableCount * Self.fieldPairCount * Self.hashBuckets
        self.hashZ = Array(repeating: 0.0, count: hashSize)
        self.hashN = Array(repeating: 0.0, count: hashSize)
        self.hashW = Array(repeating: 0.0, count: hashSize)
        self.bucketUseEMA = Array(repeating: 0.0, count: Self.tableCount * Self.fieldPairCount * Self.hashBuckets)
        self.pairCollisionEMA = Array(repeating: 0.0, count: Self.fieldPairCount)
        self.bias = Array(repeating: 0.0, count: Self.classCount * Self.regimeCount * Self.sessionCount)
        self.biasG2 = Array(repeating: 0.0, count: Self.classCount * Self.regimeCount * Self.sessionCount)
        self.normReady = false
        self.normSteps = 0
        self.featureMean = Array(repeating: 0.0, count: featureCount)
        self.featureVariance = Array(repeating: 1.0, count: featureCount)
        self.step = 0
        self.seedA = 2_166_136_261
        self.seedB = 3_735_928_559
        self.collisionDiagnosticEMA = 0.0
        self.calibrationErrorEMA = 0.0
        self.edgeHitEMA = 0.50
        self.uncertaintyEMA = 0.0
        self.interactionCount = 0
        self.interactionMean = 0.0
        self.interactionM2 = 0.0
        self.moveReady = false
        self.moveEMAAbs = 0.0
        self.calibrationTemperature = 1.0
        self.calibrationBias = Array(repeating: 0.0, count: Self.classCount)
        self.isotonicPositive = Array(repeating: Array(repeating: 0.0, count: Self.calibrationBins), count: Self.classCount)
        self.isotonicCount = Array(repeating: Array(repeating: 0.0, count: Self.calibrationBins), count: Self.classCount)
        self.calibrationSteps = 0
        for regime in 0..<Self.regimeCount {
            for session in 0..<Self.sessionCount {
                self.bias[Self.biasIndex(LabelClass.skip.rawValue, regime, session)] = 0.20
            }
        }
    }

    public mutating func reset() {
        self = LinEnhashCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        let raw = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let x = normalized(raw, updateStatistics: true)
        let session = Self.sessionBucket(sampleTimeUTC: request.context.sampleTimeUTC)
        let regime = Self.regimeBucket(raw, priceCostPoints: request.context.priceCostPoints)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: raw,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        let evaluation = evaluate(x, session: session, regime: regime)
        let sampleWeight = fxClamp(
            PluginSupportTools.moveSampleWeight(
                x: raw,
                movePoints: request.movePoints,
                priceCostPoints: request.context.priceCostPoints,
                minMovePoints: request.context.minMovePoints,
                qualityTargets: PluginQualityTargets(request: request)
            ) * request.sampleWeight,
            0.10,
            6.00
        )
        var hyperState = adaptedHyperParameters(hyperParameters)
        hyperState = HyperState(
            alpha: fxClamp(hyperState.alpha * fxClamp(sampleWeight, 0.25, 4.00), 0.00005, 0.1500),
            beta: hyperState.beta,
            l1: hyperState.l1,
            l2: hyperState.l2
        )

        for classIndex in 0..<Self.classCount {
            let target = classIndex == label.rawValue ? 1.0 : 0.0
            let biasGradient = PluginSupportTools.clipSymmetric(sampleWeight * (target - evaluation.probabilities[classIndex]), limit: 4.0)
            let ftrlGradient = PluginSupportTools.clipSymmetric(sampleWeight * (evaluation.probabilities[classIndex] - target), limit: 4.0)
            let biasOffset = Self.biasIndex(classIndex, regime, session)
            biasG2[biasOffset] += biasGradient * biasGradient
            let biasLR = hyperState.alpha / sqrt(biasG2[biasOffset] + 1.0e-8)
            bias[biasOffset] = PluginSupportTools.clipSymmetric(bias[biasOffset] + biasLR * biasGradient, limit: 8.0)

            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                Self.ftrlUpdate(
                    z: &linearZ[classIndex][featureIndex],
                    n: &linearN[classIndex][featureIndex],
                    w: &linearW[classIndex][featureIndex],
                    gradient: ftrlGradient * x[featureIndex],
                    state: hyperState
                )
            }
        }

        updateHashedInteractions(x: x, label: label, probabilities: evaluation.probabilities, sampleWeight: sampleWeight, state: hyperState)
        updateCalibrationAndDiagnostics(
            evaluation: evaluation,
            label: label,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints,
            sampleWeight: sampleWeight,
            learningRate: hyperState.alpha
        )
        updateMoveEMA(movePoints: request.movePoints, alpha: 0.05)
        step += 1
        if step % Self.rehashPeriod == 0 || collisionDiagnosticEMA > 0.65 {
            rotateSecondaryTable()
        }
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        let raw = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let x = normalizedForPrediction(raw)
        let session = Self.sessionBucket(sampleTimeUTC: request.context.sampleTimeUTC)
        let regime = Self.regimeBucket(raw, priceCostPoints: request.context.priceCostPoints)
        let evaluation = evaluate(x, session: session, regime: regime)
        let probabilities = calibrated(evaluation.probabilities)
        var mean = (probabilities[LabelClass.buy.rawValue] + probabilities[LabelClass.sell.rawValue]) *
            (evaluation.interactionAmplitude + 0.35 * evaluation.interactionStd)
        if interactionCount > 1 {
            let variance = interactionM2 / Double(interactionCount - 1)
            if variance > 0.0 {
                mean += 0.20 * sqrt(variance)
            }
        }
        mean = max(mean, moveReady ? moveEMAAbs : 0.0)
        let sigma = max(0.10, 0.40 * evaluation.interactionStd + 0.35 * evaluation.collisionMetric + 0.15 * uncertaintyEMA)
        let confidence = fxClamp(
            1.0 / (1.0 + 0.60 * evaluation.interactionStd + 0.50 * evaluation.collisionMetric + 0.20 * calibrationErrorEMA),
            0.0,
            1.0
        )
        let output = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: mean,
            moveQ25Points: max(0.0, mean - 0.55 * sigma),
            moveQ50Points: mean,
            moveQ75Points: max(mean, mean + 0.55 * sigma),
            mfeMeanPoints: mean,
            maeMeanPoints: max(0.0, 0.35 * mean),
            hitTimeFraction: fxClamp(0.70 - 0.20 * max(probabilities[0], probabilities[1]), 0.0, 1.0),
            pathRisk: probabilities[LabelClass.skip.rawValue],
            fillRisk: fxClamp(request.context.priceCostPoints / max(mean, request.context.minMovePoints, 0.25), 0.0, 1.0),
            confidence: confidence,
            reliability: fxClamp(0.55 + 0.20 * edgeHitEMA + 0.15 * (1.0 - fxClamp(calibrationErrorEMA, 0.0, 1.0)) + 0.10 * (1.0 - fxClamp(collisionDiagnosticEMA, 0.0, 1.0)), 0.0, 1.0),
            hasQuantiles: true,
            hasConfidence: true,
            hasPathQuality: true
        )
        return PluginContextRuntimeTools.fillPrediction(
            modelOutput: output,
            calibratedMoveMeanPoints: mean,
            context: request.context
        )
    }

    private mutating func updateHashedInteractions(
        x: [Double],
        label: LabelClass,
        probabilities: [Double],
        sampleWeight: Double,
        state: HyperState
    ) {
        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                let baseInteraction = x[i] * x[j]
                if abs(baseInteraction) < 1.0e-12 { continue }
                let fieldPair = Self.fieldPairIndex(Self.featureField(i), Self.featureField(j))
                let index0 = hashIndex(table: 0, fieldPair: fieldPair, i: i, j: j)
                let index1 = hashIndex(table: 1, fieldPair: fieldPair, i: i, j: j)
                let use0 = bucketUseEMA[Self.bucketIndex(0, fieldPair, index0)]
                let use1 = bucketUseEMA[Self.bucketIndex(1, fieldPair, index1)]
                let inverse0 = 1.0 / (0.05 + use0)
                let inverse1 = 1.0 / (0.05 + use1)
                let inverseSum = max(inverse0 + inverse1, 1.0e-12)
                let pairScale = 1.0 / (1.0 + 0.75 * (use0 + use1))
                let a0 = pairScale * (inverse0 / inverseSum)
                let a1 = pairScale * (inverse1 / inverseSum)
                let bucketOffset0 = Self.bucketIndex(0, fieldPair, index0)
                let bucketOffset1 = Self.bucketIndex(1, fieldPair, index1)
                bucketUseEMA[bucketOffset0] = 0.995 * bucketUseEMA[bucketOffset0] + 0.005
                bucketUseEMA[bucketOffset1] = 0.995 * bucketUseEMA[bucketOffset1] + 0.005
                pairCollisionEMA[fieldPair] = 0.995 * pairCollisionEMA[fieldPair] + 0.005 * 0.5 * (use0 + use1)
                let value0 = hashSign(table: 0, fieldPair: fieldPair, i: i, j: j) * baseInteraction
                let value1 = hashSign(table: 1, fieldPair: fieldPair, i: i, j: j) * baseInteraction

                for classIndex in 0..<Self.classCount {
                    let target = classIndex == label.rawValue ? 1.0 : 0.0
                    let classGradient = PluginSupportTools.clipSymmetric(sampleWeight * (probabilities[classIndex] - target), limit: 4.0)
                    let hashOffset0 = Self.hashIndex(classIndex, 0, fieldPair, index0)
                    let hashOffset1 = Self.hashIndex(classIndex, 1, fieldPair, index1)
                    Self.ftrlUpdate(
                        z: &hashZ[hashOffset0],
                        n: &hashN[hashOffset0],
                        w: &hashW[hashOffset0],
                        gradient: classGradient * a0 * value0,
                        state: state
                    )
                    Self.ftrlUpdate(
                        z: &hashZ[hashOffset1],
                        n: &hashN[hashOffset1],
                        w: &hashW[hashOffset1],
                        gradient: classGradient * a1 * value1,
                        state: state
                    )
                }
            }
        }
    }

    private mutating func updateCalibrationAndDiagnostics(
        evaluation: EvalResult,
        label: LabelClass,
        movePoints: Double,
        priceCostPoints: Double,
        sampleWeight: Double,
        learningRate: Double
    ) {
        updateCalibrator(rawProbabilities: evaluation.probabilities, label: label, sampleWeight: sampleWeight, learningRate: learningRate)
        let directionalLabel = label == .buy ? 1.0 : (label == .sell ? 0.0 : (movePoints >= 0.0 ? 1.0 : 0.0))
        let confidence = fxClamp(
            1.0 / (1.0 + 0.60 * evaluation.interactionStd + 0.50 * evaluation.collisionMetric + 0.20 * calibrationErrorEMA),
            0.25,
            1.0
        )
        let directionConfidence = 0.5 + (evaluation.directionRaw - 0.5) * confidence
        let calibrationError = abs(directionalLabel - directionConfidence)
        calibrationErrorEMA = 0.98 * calibrationErrorEMA + 0.02 * calibrationError
        collisionDiagnosticEMA = 0.98 * collisionDiagnosticEMA + 0.02 * evaluation.collisionMetric
        uncertaintyEMA = 0.98 * uncertaintyEMA + 0.02 * fxClamp(evaluation.interactionStd, 0.0, 2.0)
        let predictedClass = (0..<Self.classCount).max { evaluation.probabilities[$0] < evaluation.probabilities[$1] } ?? LabelClass.skip.rawValue
        let edge = abs(movePoints) - max(priceCostPoints, 0.0)
        let edgeHit: Double
        if predictedClass == label.rawValue {
            edgeHit = label == .skip ? (edge <= 0.0 ? 1.0 : 0.25) : (edge > 0.0 ? 1.0 : 0.0)
        } else {
            edgeHit = 0.0
        }
        edgeHitEMA = 0.98 * edgeHitEMA + 0.02 * edgeHit
        let interactionScore = evaluation.interactionAmplitude + 0.35 * evaluation.interactionStd
        interactionCount += 1
        let delta = interactionScore - interactionMean
        interactionMean += delta / Double(interactionCount)
        interactionM2 += delta * (interactionScore - interactionMean)
    }

    private func evaluate(_ x: [Double], session: Int, regime: Int) -> EvalResult {
        var logits = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            var value = bias[Self.biasIndex(classIndex, regime, session)]
            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                value += linearW[classIndex][featureIndex] * x[featureIndex]
            }
            logits[classIndex] = value
        }

        var differenceSum = 0.0
        var differenceSum2 = 0.0
        var collisionSum = 0.0
        var pairCount = 0
        var interactionAmplitude = 0.0

        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                let baseInteraction = x[i] * x[j]
                if abs(baseInteraction) < 1.0e-12 { continue }
                let fieldPair = Self.fieldPairIndex(Self.featureField(i), Self.featureField(j))
                let index0 = hashIndex(table: 0, fieldPair: fieldPair, i: i, j: j)
                let index1 = hashIndex(table: 1, fieldPair: fieldPair, i: i, j: j)
                let use0 = bucketUseEMA[Self.bucketIndex(0, fieldPair, index0)]
                let use1 = bucketUseEMA[Self.bucketIndex(1, fieldPair, index1)]
                let inverse0 = 1.0 / (0.05 + use0)
                let inverse1 = 1.0 / (0.05 + use1)
                let inverseSum = max(inverse0 + inverse1, 1.0e-12)
                let pairScale = 1.0 / (1.0 + 0.75 * (use0 + use1))
                let a0 = pairScale * (inverse0 / inverseSum)
                let a1 = pairScale * (inverse1 / inverseSum)
                let value0 = hashSign(table: 0, fieldPair: fieldPair, i: i, j: j) * baseInteraction
                let value1 = hashSign(table: 1, fieldPair: fieldPair, i: i, j: j) * baseInteraction
                var buyContribution = 0.0
                var sellContribution = 0.0

                for classIndex in 0..<Self.classCount {
                    let contribution = a0 * hashW[Self.hashIndex(classIndex, 0, fieldPair, index0)] * value0 +
                        a1 * hashW[Self.hashIndex(classIndex, 1, fieldPair, index1)] * value1
                    logits[classIndex] += contribution
                    if classIndex == LabelClass.buy.rawValue { buyContribution = contribution }
                    if classIndex == LabelClass.sell.rawValue { sellContribution = contribution }
                }

                let difference = buyContribution - sellContribution
                differenceSum += difference
                differenceSum2 += difference * difference
                interactionAmplitude += 0.5 * (abs(buyContribution) + abs(sellContribution))
                collisionSum += 0.5 * (use0 + use1)
                pairCount += 1
            }
        }

        let probabilities = Self.softmax3(logits)
        let skipProbability = probabilities[LabelClass.skip.rawValue]
        let denominator = max(probabilities[LabelClass.buy.rawValue] + probabilities[LabelClass.sell.rawValue], 1.0e-9)
        let directionRaw = probabilities[LabelClass.buy.rawValue] / denominator
        if pairCount > 0 {
            interactionAmplitude /= Double(pairCount)
            let meanDifference = differenceSum / Double(pairCount)
            let variance = max(0.0, differenceSum2 / Double(pairCount) - meanDifference * meanDifference)
            return EvalResult(
                probabilities: probabilities,
                directionRaw: directionRaw,
                skipProbability: skipProbability,
                interactionAmplitude: interactionAmplitude,
                interactionStd: sqrt(variance),
                collisionMetric: fxClamp(collisionSum / Double(pairCount), 0.0, 1.0)
            )
        }
        return EvalResult(
            probabilities: probabilities,
            directionRaw: directionRaw,
            skipProbability: skipProbability,
            interactionAmplitude: 0.0,
            interactionStd: 0.0,
            collisionMetric: 0.0
        )
    }

    private mutating func normalized(_ x: [Double], updateStatistics: Bool) -> [Double] {
        if updateStatistics {
            let alpha = normSteps < 128 ? 0.04 : 0.010
            for index in 1..<FXDataEngineConstants.aiWeights {
                let delta = x[index] - featureMean[index]
                featureMean[index] += alpha * delta
                let varianceDelta = x[index] - featureMean[index]
                featureVariance[index] = max((1.0 - alpha) * featureVariance[index] + alpha * varianceDelta * varianceDelta, 1.0e-6)
            }
            normSteps += 1
            if normSteps >= 32 {
                normReady = true
            }
        }
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        output[0] = 1.0
        for index in 1..<FXDataEngineConstants.aiWeights {
            var value = x[index]
            if normReady {
                value = (x[index] - featureMean[index]) / sqrt(featureVariance[index] + 1.0e-6)
            }
            output[index] = PluginSupportTools.clipSymmetric(value, limit: 6.0)
        }
        return output
    }

    private func normalizedForPrediction(_ x: [Double]) -> [Double] {
        var copy = self
        return copy.normalized(x, updateStatistics: false)
    }

    private func adaptedHyperParameters(_ hyperParameters: HyperParameters) -> HyperState {
        var alpha = fxClamp(hyperParameters.enhashLearningRate, 0.0002, 0.1000)
        var beta = 1.0 + 4.0 * fxClamp(hyperParameters.enhashL2, 0.0000, 0.1000)
        var l1 = fxClamp(hyperParameters.enhashL1, 0.0000, 0.1000)
        var l2 = fxClamp(hyperParameters.enhashL2, 0.0000, 0.1000)
        let penalty = 0.45 * collisionDiagnosticEMA + 0.35 * calibrationErrorEMA + 0.20 * uncertaintyEMA
        let reward = 0.30 * (edgeHitEMA - 0.50)
        alpha *= fxClamp(1.0 - penalty + reward, 0.35, 1.50)
        beta *= fxClamp(1.0 + 0.80 * collisionDiagnosticEMA, 0.80, 3.00)
        l1 *= fxClamp(1.0 + 1.50 * collisionDiagnosticEMA, 0.80, 3.00)
        l2 *= fxClamp(1.0 + calibrationErrorEMA + 0.80 * collisionDiagnosticEMA, 0.80, 3.50)
        return HyperState(
            alpha: fxClamp(alpha, 0.00005, 0.1500),
            beta: fxClamp(beta, 0.25000, 12.0000),
            l1: fxClamp(l1, 0.00000, 0.2000),
            l2: fxClamp(l2, 0.00000, 0.2000)
        )
    }

    private mutating func updateCalibrator(rawProbabilities: [Double], label: LabelClass, sampleWeight: Double, learningRate: Double) {
        let weight = fxClamp(sampleWeight, 0.25, 6.00)
        let calibrationLearningRate = fxClamp(0.18 * learningRate * weight, 0.0002, 0.0200)
        let probabilities = calibrated(rawProbabilities)
        var temperatureGradient = 0.0
        for classIndex in 0..<Self.classCount {
            let target = classIndex == label.rawValue ? 1.0 : 0.0
            let error = target - probabilities[classIndex]
            calibrationBias[classIndex] = PluginSupportTools.clipSymmetric(
                calibrationBias[classIndex] + calibrationLearningRate * error,
                limit: 4.0
            )
            temperatureGradient += error * log(fxClamp(rawProbabilities[classIndex], 0.0005, 0.9990))
            let bin = Self.isotonicBin(probabilities[classIndex])
            isotonicCount[classIndex][bin] += weight
            isotonicPositive[classIndex][bin] += weight * target
        }
        calibrationTemperature = fxClamp(calibrationTemperature - 0.02 * calibrationLearningRate * temperatureGradient, 0.50, 3.00)
        calibrationSteps += 1
    }

    private func calibrated(_ rawProbabilities: [Double]) -> [Double] {
        let inverseTemperature = 1.0 / fxClamp(calibrationTemperature, 0.50, 3.00)
        var logits = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            logits[classIndex] = log(fxClamp(rawProbabilities[classIndex], 0.0005, 0.9990)) * inverseTemperature + calibrationBias[classIndex]
        }
        var probabilities = Self.softmax3(logits)
        guard calibrationSteps >= 30 else { return probabilities }
        var isotonic = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            let total = isotonicCount[classIndex].reduce(0.0, +)
            guard total >= 30.0 else {
                isotonic[classIndex] = probabilities[classIndex]
                continue
            }
            var previous = 0.01
            var monotonic = Array(repeating: 0.0, count: Self.calibrationBins)
            for bin in 0..<Self.calibrationBins {
                var ratio = previous
                if isotonicCount[classIndex][bin] > 1.0e-9 {
                    ratio = isotonicPositive[classIndex][bin] / isotonicCount[classIndex][bin]
                }
                ratio = fxClamp(ratio, 0.001, 0.999)
                if ratio < previous { ratio = previous }
                monotonic[bin] = ratio
                previous = ratio
            }
            isotonic[classIndex] = monotonic[Self.isotonicBin(probabilities[classIndex])]
        }
        for classIndex in 0..<Self.classCount {
            probabilities[classIndex] = fxClamp(0.75 * probabilities[classIndex] + 0.25 * isotonic[classIndex], 0.0005, 0.9990)
        }
        return PluginContextRuntimeTools.normalizeClassDistribution(probabilities)
    }

    private mutating func updateMoveEMA(movePoints: Double, alpha: Double) {
        let target = abs(fxSafeFinite(movePoints))
        guard target.isFinite else { return }
        if !moveReady {
            moveEMAAbs = target
            moveReady = true
        } else {
            moveEMAAbs = (1.0 - alpha) * moveEMAAbs + alpha * target
        }
    }

    private mutating func rotateSecondaryTable() {
        seedB = seedB &* 1_664_525 &+ 1_013_904_223
        for classIndex in 0..<Self.classCount {
            for fieldPair in 0..<Self.fieldPairCount {
                for bucket in 0..<Self.hashBuckets {
                    let offset = Self.hashIndex(classIndex, 1, fieldPair, bucket)
                    hashW[offset] *= 0.75
                    hashZ[offset] *= 0.75
                    hashN[offset] *= 0.90
                    bucketUseEMA[Self.bucketIndex(1, fieldPair, bucket)] *= 0.50
                    bucketUseEMA[Self.bucketIndex(0, fieldPair, bucket)] *= 0.98
                }
            }
        }
    }

    private static func ftrlUpdate(z: inout Double, n: inout Double, w: inout Double, gradient: Double, state: HyperState) {
        var gradient = fxSafeFinite(gradient)
        gradient = fxClamp(gradient, -10.0, 10.0)
        let oldN = n
        let newN = oldN + gradient * gradient
        let sigma = (sqrt(newN) - sqrt(oldN)) / max(state.alpha, 1.0e-9)
        z += gradient - sigma * w
        n = newN
        w = PluginSupportTools.clipSymmetric(ftrlWeight(z: z, n: n, state: state), limit: 8.0)
    }

    private static func ftrlWeight(z: Double, n: Double, state: HyperState) -> Double {
        let absZ = abs(z)
        guard absZ > state.l1 else { return 0.0 }
        let denominator = (state.beta + sqrt(max(n, 0.0))) / max(state.alpha, 1.0e-9) + state.l2
        guard denominator > 1.0e-12 else { return 0.0 }
        let sign = z < 0.0 ? -1.0 : 1.0
        return -(z - sign * state.l1) / denominator
    }

    private static func preparedFeatures(_ x: [Double], dataHasVolume: Bool) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<FXDataEngineConstants.aiWeights {
            let value = index < x.count ? fxSafeFinite(x[index]) : 0.0
            output[index] = fxClamp(value, -50.0, 50.0)
        }
        if !dataHasVolume {
            for index in [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83] where index < output.count {
                output[index] = 0.0
            }
        }
        return output
    }

    private static func softmax3(_ logits: [Double]) -> [Double] {
        let maxLogit = logits.prefix(Self.classCount).map { fxSafeFinite($0) }.max() ?? 0.0
        var values = Array(repeating: 0.0, count: Self.classCount)
        var sum = 0.0
        for classIndex in 0..<Self.classCount {
            let value = exp(PluginSupportTools.clipSymmetric(fxSafeFinite(logits[classIndex]) - maxLogit, limit: 30.0))
            values[classIndex] = value
            sum += value
        }
        guard sum > 0.0 else { return [0.3333333, 0.3333333, 0.3333333] }
        return values.map { $0 / sum }
    }

    private static func sessionBucket(sampleTimeUTC: Int64) -> Int {
        let secondsPerDay: Int64 = 86_400
        let hour = Int(((sampleTimeUTC % secondsPerDay) + secondsPerDay) % secondsPerDay / 3_600)
        if hour < 6 { return 0 }
        if hour < 12 { return 1 }
        if hour < 20 { return 2 }
        return 3
    }

    private static func regimeBucket(_ x: [Double], priceCostPoints: Double) -> Int {
        let volume = x.count > 6 ? abs(x[6]) : 0.0
        let score = volume + 0.20 * max(priceCostPoints, 0.0)
        if score < 0.90 { return 0 }
        if score < 1.80 { return 1 }
        return 2
    }

    private static func featureField(_ index: Int) -> Int {
        if index <= 3 { return 0 }
        if index <= 6 { return 1 }
        if index <= 8 { return 2 }
        if index <= 12 { return 3 }
        return 4
    }

    private static func fieldPairIndex(_ a: Int, _ b: Int) -> Int {
        let first = min(a, b)
        let second = max(a, b)
        return first * Self.fieldCount + second
    }

    private func hashIndex(table: Int, fieldPair: Int, i: Int, j: Int) -> Int {
        let seed = table == 0 ? seedA : seedB
        let a = UInt32(truncatingIfNeeded: i &* 73_856_093)
        let b = UInt32(truncatingIfNeeded: j &* 19_349_663)
        let p = UInt32(truncatingIfNeeded: fieldPair &* 83_492_791)
        return Int((a ^ b ^ p ^ seed) % UInt32(Self.hashBuckets))
    }

    private func hashSign(table: Int, fieldPair: Int, i: Int, j: Int) -> Double {
        let a = UInt32(truncatingIfNeeded: i &* 31)
        let b = UInt32(truncatingIfNeeded: j &* 17)
        let p = UInt32(truncatingIfNeeded: fieldPair &* 131)
        let t = UInt32(truncatingIfNeeded: table) &* 2_654_435_761
        return ((a ^ b ^ p ^ t) & 1) == 0 ? 1.0 : -1.0
    }

    private static func hashIndex(_ classIndex: Int, _ table: Int, _ fieldPair: Int, _ bucket: Int) -> Int {
        (((classIndex * Self.tableCount + table) * Self.fieldPairCount + fieldPair) * Self.hashBuckets) + bucket
    }

    private static func bucketIndex(_ table: Int, _ fieldPair: Int, _ bucket: Int) -> Int {
        ((table * Self.fieldPairCount + fieldPair) * Self.hashBuckets) + bucket
    }

    private static func biasIndex(_ classIndex: Int, _ regime: Int, _ session: Int) -> Int {
        ((classIndex * Self.regimeCount + regime) * Self.sessionCount) + session
    }

    private static func isotonicBin(_ probability: Double) -> Int {
        min(max(Int(floor(fxClamp(probability, 0.0, 0.999999) * Double(Self.calibrationBins))), 0), Self.calibrationBins - 1)
    }
}
