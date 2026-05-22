import FXDataEngine
import Foundation

public struct LinFTRLCPUModel: Sendable {
    private struct AdaptiveParams: Sendable {
        let alpha: Double
        let beta: Double
        let l1: Double
        let l2: Double
    }

    private static let classCount = 3
    private static let hashBuckets1 = 128
    private static let hashBuckets2 = 97
    private static let calibrationBins = 10

    private var step: Int
    private var classZ: [[Double]]
    private var classN: [[Double]]
    private var hashZ1: [[Double]]
    private var hashN1: [[Double]]
    private var hashZ2: [[Double]]
    private var hashN2: [[Double]]
    private var hashLoad1: [Double]
    private var hashLoad2: [Double]
    private var hashMean1: Double
    private var hashMean2: Double
    private var moveZ: [Double]
    private var moveN: [Double]
    private var moveHashZ1: [Double]
    private var moveHashN1: [Double]
    private var moveHashZ2: [Double]
    private var moveHashN2: [Double]
    private var moveSteps: Int
    private var moveReady: Bool
    private var moveEMAAbs: Double
    private var classEMA: [Double]
    private var lossReady: Bool
    private var lossFast: Double
    private var lossSlow: Double
    private var driftCooldown: Int
    private var calibrationTemperature: Double
    private var calibrationBias: [Double]
    private var isotonicPositive: [[Double]]
    private var isotonicCount: [[Double]]
    private var calibrationSteps: Int
    private let useHash: Bool
    private let useHash2: Bool

    public init() {
        let featureCount = FXDataEngineConstants.aiWeights
        self.step = 0
        self.classZ = Array(repeating: Array(repeating: 0.0, count: featureCount), count: Self.classCount)
        self.classN = Array(repeating: Array(repeating: 0.0, count: featureCount), count: Self.classCount)
        self.hashZ1 = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets1), count: Self.classCount)
        self.hashN1 = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets1), count: Self.classCount)
        self.hashZ2 = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets2), count: Self.classCount)
        self.hashN2 = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets2), count: Self.classCount)
        self.hashLoad1 = Array(repeating: 0.0, count: Self.hashBuckets1)
        self.hashLoad2 = Array(repeating: 0.0, count: Self.hashBuckets2)
        self.hashMean1 = 0.0
        self.hashMean2 = 0.0
        self.moveZ = Array(repeating: 0.0, count: featureCount)
        self.moveN = Array(repeating: 0.0, count: featureCount)
        self.moveHashZ1 = Array(repeating: 0.0, count: Self.hashBuckets1)
        self.moveHashN1 = Array(repeating: 0.0, count: Self.hashBuckets1)
        self.moveHashZ2 = Array(repeating: 0.0, count: Self.hashBuckets2)
        self.moveHashN2 = Array(repeating: 0.0, count: Self.hashBuckets2)
        self.moveSteps = 0
        self.moveReady = false
        self.moveEMAAbs = 0.0
        self.classEMA = Array(repeating: 1.0, count: Self.classCount)
        self.lossReady = false
        self.lossFast = 0.0
        self.lossSlow = 0.0
        self.driftCooldown = 0
        self.calibrationTemperature = 1.0
        self.calibrationBias = Array(repeating: 0.0, count: Self.classCount)
        self.isotonicPositive = Array(repeating: Array(repeating: 0.0, count: Self.calibrationBins), count: Self.classCount)
        self.isotonicCount = Array(repeating: Array(repeating: 0.0, count: Self.calibrationBins), count: Self.classCount)
        self.calibrationSteps = 0
        self.useHash = true
        self.useHash2 = true
    }

    public mutating func reset() {
        self = LinFTRLCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        let hp = PluginSupportTools.scaleHyperParametersForMove(hyperParameters, movePoints: request.movePoints)
        let cost = PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: request.context.priceCostPoints)
        let edge = max(0.0, abs(fxSafeFinite(request.movePoints)) - cost)
        let denominator = max(cost, 0.50)
        var sampleWeight = fxClamp(0.35 + edge / denominator, 0.15, 6.00)
        if label == .skip {
            sampleWeight *= 0.90
        }
        sampleWeight = fxClamp(sampleWeight * request.sampleWeight, 0.10, 6.00)
        updateWeighted(label: label, x: x, hyperParameters: hp, sampleWeight: sampleWeight, movePoints: request.movePoints)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let params = adaptiveParams(hyperParameters, sampleWeight: 1.0)
        let weights = classWeights(params)
        var logits = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            logits[classIndex] = PluginSupportTools.clipSymmetric(scoreClass(x, classIndex: classIndex, weights: weights), limit: 35.0)
        }
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution(calibrated(Self.softmax3(logits)))
        let moveMean = max(expectedMovePoints(x, hyperParameters: hyperParameters), 0.0)
        let sigma = max(0.10, 0.35 * moveMean + 0.25 * (moveReady ? moveEMAAbs : 0.0))
        let output = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: moveMean,
            moveQ25Points: max(0.0, moveMean - 0.55 * sigma),
            moveQ50Points: max(0.0, moveMean),
            moveQ75Points: max(moveMean, moveMean + 0.55 * sigma),
            mfeMeanPoints: moveMean,
            maeMeanPoints: max(0.0, 0.35 * moveMean),
            hitTimeFraction: fxClamp(0.70 - 0.20 * max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]), 0.0, 1.0),
            pathRisk: probabilities[LabelClass.skip.rawValue],
            fillRisk: fxClamp(request.context.priceCostPoints / max(moveMean, request.context.minMovePoints, 0.25), 0.0, 1.0),
            confidence: fxClamp(max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]), 0.0, 1.0),
            reliability: fxClamp(0.45 + 0.25 * (moveReady ? 1.0 : 0.0) + 0.30 * min(Double(moveSteps) / 64.0, 1.0), 0.0, 1.0),
            hasQuantiles: true,
            hasConfidence: true,
            hasPathQuality: true
        )
        return PluginContextRuntimeTools.fillPrediction(
            modelOutput: output,
            calibratedMoveMeanPoints: moveMean,
            context: request.context
        )
    }

    private mutating func updateWeighted(
        label: LabelClass,
        x: [Double],
        hyperParameters: HyperParameters,
        sampleWeight: Double,
        movePoints: Double
    ) {
        step += 1
        for classIndex in 0..<Self.classCount {
            classEMA[classIndex] = 0.997 * classEMA[classIndex] + (classIndex == label.rawValue ? 0.003 : 0.0)
        }
        let meanClass = classEMA.reduce(0.0, +) / Double(Self.classCount)
        let classBalance = fxClamp(meanClass / max(classEMA[label.rawValue], 0.005), 0.60, 2.50)
        let recency = 0.85 + 0.30 * (1.0 - exp(-Double(step) / 512.0))
        let weight = fxClamp(sampleWeight * classBalance * recency, 0.10, 6.00)
        let params = adaptiveParams(hyperParameters, sampleWeight: weight)
        let weights = classWeights(params)

        var logits = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            logits[classIndex] = PluginSupportTools.clipSymmetric(scoreClass(x, classIndex: classIndex, weights: weights), limit: 35.0)
        }
        let rawProbabilities = Self.softmax3(logits)
        var logitGradient = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            let target = classIndex == label.rawValue ? 1.0 : 0.0
            logitGradient[classIndex] = PluginSupportTools.clipSymmetric((rawProbabilities[classIndex] - target) * weight, limit: 4.0)
        }

        var linearGradient = Array(repeating: Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights), count: Self.classCount)
        var hashGradient1 = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets1), count: Self.classCount)
        var hashGradient2 = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets2), count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                linearGradient[classIndex][featureIndex] = PluginSupportTools.clipSymmetric(
                    logitGradient[classIndex] * x[featureIndex],
                    limit: 8.0
                )
            }
        }

        if useHash {
            for i in 1..<FXDataEngineConstants.aiWeights {
                for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                    let hashedValue = Self.hashSign(i, j) * x[i] * x[j]
                    let h1 = Self.hashIndex(i, j)
                    hashLoad1[h1] = 0.997 * hashLoad1[h1] + 0.003
                    hashMean1 = 0.999 * hashMean1 + 0.001 * hashLoad1[h1]
                    let rb1 = collisionRebalance1(h1)
                    let h2 = Self.hashIndex2(i, j)
                    hashLoad2[h2] = 0.997 * hashLoad2[h2] + 0.003
                    hashMean2 = 0.999 * hashMean2 + 0.001 * hashLoad2[h2]
                    let rb2 = collisionRebalance2(h2)

                    for classIndex in 0..<Self.classCount {
                        let gradient = PluginSupportTools.clipSymmetric(logitGradient[classIndex] * hashedValue, limit: 8.0)
                        hashGradient1[classIndex][h1] += rb1 * gradient
                        if useHash2 {
                            hashGradient2[classIndex][h2] += 0.70 * rb2 * gradient
                        }
                    }
                }
            }
        }

        let gradientScale = gradientScale(linear: linearGradient, hash1: hashGradient1, hash2: hashGradient2, sampleWeight: weight)
        for classIndex in 0..<Self.classCount {
            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                Self.ftrlUpdate(
                    z: &classZ[classIndex][featureIndex],
                    n: &classN[classIndex][featureIndex],
                    weight: weights.linear[classIndex][featureIndex],
                    gradient: linearGradient[classIndex][featureIndex] * gradientScale,
                    alpha: params.alpha
                )
            }
            for bucket in 0..<Self.hashBuckets1 {
                Self.ftrlUpdate(
                    z: &hashZ1[classIndex][bucket],
                    n: &hashN1[classIndex][bucket],
                    weight: weights.hash1[classIndex][bucket],
                    gradient: hashGradient1[classIndex][bucket] * gradientScale,
                    alpha: params.alpha
                )
            }
            for bucket in 0..<Self.hashBuckets2 {
                Self.ftrlUpdate(
                    z: &hashZ2[classIndex][bucket],
                    n: &hashN2[classIndex][bucket],
                    weight: weights.hash2[classIndex][bucket],
                    gradient: hashGradient2[classIndex][bucket] * gradientScale,
                    alpha: params.alpha
                )
            }
        }

        let loss = -log(fxClamp(rawProbabilities[label.rawValue], 1.0e-6, 1.0))
        updateLossDrift(crossEntropyLoss: loss)
        let calibrationLearningRate = fxClamp(params.alpha * 0.15, 0.0002, 0.0200)
        updateCalibrator(rawProbabilities: rawProbabilities, label: label, sampleWeight: weight, learningRate: calibrationLearningRate)
        updateMoveHead(x: x, movePoints: movePoints, params: params, sampleWeight: weight)
    }

    private mutating func updateMoveHead(
        x: [Double],
        movePoints: Double,
        params: AdaptiveParams,
        sampleWeight: Double
    ) {
        let target = abs(fxSafeFinite(movePoints, fallback: .nan))
        guard target.isFinite else { return }
        let moveParams = AdaptiveParams(
            alpha: fxClamp(0.70 * params.alpha, 0.00005, 3.0000),
            beta: fxClamp(0.80 * params.beta + 0.05, 0.0000, 10.0000),
            l1: 0.0,
            l2: fxClamp(0.50 * params.l2 + 0.001, 0.0000, 2.0000)
        )
        let weights = moveWeights(moveParams)
        let prediction = max(scoreMove(x, weights: weights), 0.0)
        let error = prediction - target
        let huberGradient = abs(error) <= 1.0 ? error : (error > 0.0 ? 1.0 : -1.0)
        let weight = fxClamp(sampleWeight, 0.25, 6.00)

        var linearGradient = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        var hashGradient1 = Array(repeating: 0.0, count: Self.hashBuckets1)
        var hashGradient2 = Array(repeating: 0.0, count: Self.hashBuckets2)
        var normSquared = 0.0
        for featureIndex in 0..<FXDataEngineConstants.aiWeights {
            let gradient = PluginSupportTools.clipSymmetric(weight * huberGradient * x[featureIndex], limit: 6.0)
            linearGradient[featureIndex] = gradient
            normSquared += gradient * gradient
        }
        if useHash {
            for i in 1..<FXDataEngineConstants.aiWeights {
                for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                    let hashedValue = Self.hashSign(i, j) * x[i] * x[j]
                    let h1 = Self.hashIndex(i, j)
                    let gradient1 = PluginSupportTools.clipSymmetric(weight * huberGradient * hashedValue, limit: 6.0)
                    hashGradient1[h1] += gradient1
                    normSquared += gradient1 * gradient1
                    if useHash2 {
                        let h2 = Self.hashIndex2(i, j)
                        let gradient2 = PluginSupportTools.clipSymmetric(weight * huberGradient * 0.70 * hashedValue, limit: 6.0)
                        hashGradient2[h2] += gradient2
                        normSquared += gradient2 * gradient2
                    }
                }
            }
        }
        let scale = normSquared > 0.0 && sqrt(normSquared) > 6.0 ? 6.0 / sqrt(normSquared) : 1.0
        for featureIndex in 0..<FXDataEngineConstants.aiWeights {
            Self.ftrlUpdate(
                z: &moveZ[featureIndex],
                n: &moveN[featureIndex],
                weight: weights.linear[featureIndex],
                gradient: linearGradient[featureIndex] * scale,
                alpha: moveParams.alpha
            )
        }
        for bucket in 0..<Self.hashBuckets1 {
            Self.ftrlUpdate(
                z: &moveHashZ1[bucket],
                n: &moveHashN1[bucket],
                weight: weights.hash1[bucket],
                gradient: hashGradient1[bucket] * scale,
                alpha: moveParams.alpha
            )
        }
        for bucket in 0..<Self.hashBuckets2 {
            Self.ftrlUpdate(
                z: &moveHashZ2[bucket],
                n: &moveHashN2[bucket],
                weight: weights.hash2[bucket],
                gradient: hashGradient2[bucket] * scale,
                alpha: moveParams.alpha
            )
        }
        updateMoveEMA(movePoints: movePoints, alpha: 0.05)
        moveSteps += 1
    }

    private static func ftrlUpdate(z: inout Double, n: inout Double, weight: Double, gradient: Double, alpha: Double) {
        let oldN = n
        let sigma = (sqrt(oldN + gradient * gradient) - sqrt(oldN)) / max(alpha, 1.0e-9)
        z += gradient - sigma * weight
        n = oldN + gradient * gradient
    }

    private func adaptiveParams(_ hyperParameters: HyperParameters, sampleWeight: Double) -> AdaptiveParams {
        var alpha = fxClamp(hyperParameters.ftrlAlpha, 0.0001, 5.0000)
        var beta = fxClamp(hyperParameters.ftrlBeta, 0.0000, 8.0000)
        let l1 = fxClamp(hyperParameters.ftrlL1, 0.0000, 0.3000)
        var l2 = fxClamp(hyperParameters.ftrlL2, 0.0000, 2.0000)
        let time = Double(max(step, 1))
        let warmup = fxClamp(time / 128.0, 0.10, 1.00)
        let inverseSqrt = 1.0 / sqrt(1.0 + 0.0025 * max(0.0, time - 128.0))
        let sampleScale = fxClamp(0.80 + 0.20 * fxClamp(sampleWeight, 0.25, 6.00), 0.70, 1.80)
        alpha = fxClamp(alpha * warmup * inverseSqrt * sampleScale, 0.00005, 5.0000)
        beta = fxClamp(beta * (1.0 + 0.15 * (1.0 - warmup)), 0.0000, 10.0000)
        if driftCooldown > 0 {
            alpha = fxClamp(alpha * 0.70, 0.00005, 5.0000)
            beta = fxClamp(beta * 1.30, 0.0000, 10.0000)
            l2 = fxClamp(l2 * 1.20, 0.0000, 2.5000)
        }
        return AdaptiveParams(alpha: alpha, beta: beta, l1: l1, l2: l2)
    }

    private func classWeights(_ params: AdaptiveParams) -> (linear: [[Double]], hash1: [[Double]], hash2: [[Double]]) {
        (
            linear: classZ.enumerated().map { classIndex, values in
                values.enumerated().map { featureIndex, zValue in
                    Self.ftrlWeight(z: zValue, n: classN[classIndex][featureIndex], params: params)
                }
            },
            hash1: hashZ1.enumerated().map { classIndex, values in
                values.enumerated().map { bucket, zValue in
                    Self.ftrlWeight(z: zValue, n: hashN1[classIndex][bucket], params: params)
                }
            },
            hash2: hashZ2.enumerated().map { classIndex, values in
                values.enumerated().map { bucket, zValue in
                    Self.ftrlWeight(z: zValue, n: hashN2[classIndex][bucket], params: params)
                }
            }
        )
    }

    private func moveWeights(_ params: AdaptiveParams) -> (linear: [Double], hash1: [Double], hash2: [Double]) {
        (
            linear: moveZ.enumerated().map { index, zValue in Self.ftrlWeight(z: zValue, n: moveN[index], params: params) },
            hash1: moveHashZ1.enumerated().map { index, zValue in Self.ftrlWeight(z: zValue, n: moveHashN1[index], params: params) },
            hash2: moveHashZ2.enumerated().map { index, zValue in Self.ftrlWeight(z: zValue, n: moveHashN2[index], params: params) }
        )
    }

    private static func ftrlWeight(z: Double, n: Double, params: AdaptiveParams) -> Double {
        let absZ = abs(z)
        guard absZ > params.l1 else { return 0.0 }
        let sign = z < 0.0 ? -1.0 : 1.0
        let denominator = ((params.beta + sqrt(max(n, 0.0))) / max(params.alpha, 1.0e-9)) + params.l2
        return -(z - sign * params.l1) / denominator
    }

    private func scoreClass(
        _ x: [Double],
        classIndex: Int,
        weights: (linear: [[Double]], hash1: [[Double]], hash2: [[Double]])
    ) -> Double {
        var value = Self.dot(weights.linear[classIndex], x)
        guard useHash else { return value }
        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                let hashedValue = Self.hashSign(i, j) * x[i] * x[j]
                value += weights.hash1[classIndex][Self.hashIndex(i, j)] * hashedValue
                if useHash2 {
                    value += 0.70 * weights.hash2[classIndex][Self.hashIndex2(i, j)] * hashedValue
                }
            }
        }
        return value
    }

    private func scoreMove(_ x: [Double], weights: (linear: [Double], hash1: [Double], hash2: [Double])) -> Double {
        var value = Self.dot(weights.linear, x)
        guard useHash else { return value }
        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                let hashedValue = Self.hashSign(i, j) * x[i] * x[j]
                value += weights.hash1[Self.hashIndex(i, j)] * hashedValue
                if useHash2 {
                    value += 0.70 * weights.hash2[Self.hashIndex2(i, j)] * hashedValue
                }
            }
        }
        return value
    }

    private func expectedMovePoints(_ x: [Double], hyperParameters: HyperParameters) -> Double {
        let params = adaptiveParams(hyperParameters, sampleWeight: 1.0)
        let moveParams = AdaptiveParams(
            alpha: fxClamp(0.70 * params.alpha, 0.00005, 3.0000),
            beta: fxClamp(0.80 * params.beta + 0.05, 0.0000, 10.0000),
            l1: 0.0,
            l2: fxClamp(0.50 * params.l2 + 0.001, 0.0000, 2.0000)
        )
        let prediction = max(scoreMove(x, weights: moveWeights(moveParams)), 0.0)
        if moveSteps >= 24, moveReady, moveEMAAbs > 0.0 {
            return 0.65 * prediction + 0.35 * moveEMAAbs
        }
        if moveSteps >= 24 {
            return prediction
        }
        if moveReady, moveEMAAbs > 0.0 {
            return moveEMAAbs
        }
        return prediction
    }

    private func gradientScale(linear: [[Double]], hash1: [[Double]], hash2: [[Double]], sampleWeight: Double) -> Double {
        var normSquared = 0.0
        for classIndex in 0..<Self.classCount {
            for value in linear[classIndex] { normSquared += value * value }
            for value in hash1[classIndex] { normSquared += value * value }
            for value in hash2[classIndex] { normSquared += value * value }
        }
        guard normSquared > 0.0 else { return 1.0 }
        let norm = sqrt(normSquared)
        let clip = fxClamp(7.0 + sqrt(sampleWeight), 6.0, 12.0)
        return norm > clip ? clip / norm : 1.0
    }

    private mutating func updateCalibrator(
        rawProbabilities: [Double],
        label: LabelClass,
        sampleWeight: Double,
        learningRate: Double
    ) {
        let weight = fxClamp(sampleWeight, 0.25, 6.00)
        let calibrationLearningRate = fxClamp(0.18 * learningRate * weight, 0.0002, 0.0200)
        let calibrated = calibrated(rawProbabilities)
        var temperatureGradient = 0.0
        for classIndex in 0..<Self.classCount {
            let target = classIndex == label.rawValue ? 1.0 : 0.0
            let error = target - calibrated[classIndex]
            calibrationBias[classIndex] = PluginSupportTools.clipSymmetric(
                calibrationBias[classIndex] + calibrationLearningRate * error,
                limit: 4.0
            )
            temperatureGradient += error * log(fxClamp(rawProbabilities[classIndex], 0.0005, 0.9990))
            let bin = Self.isotonicBin(calibrated[classIndex])
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
            let probability = fxClamp(rawProbabilities[classIndex], 0.0005, 0.9990)
            logits[classIndex] = log(probability) * inverseTemperature + calibrationBias[classIndex]
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

    private mutating func updateLossDrift(crossEntropyLoss: Double) {
        if !lossReady {
            lossFast = crossEntropyLoss
            lossSlow = crossEntropyLoss
            lossReady = true
            return
        }
        lossFast = 0.90 * lossFast + 0.10 * crossEntropyLoss
        lossSlow = 0.995 * lossSlow + 0.005 * crossEntropyLoss
        if driftCooldown > 0 { driftCooldown -= 1 }
        guard step >= 256, driftCooldown == 0 else { return }
        if lossFast > 1.7 * max(lossSlow, 0.10) {
            driftCooldown = 96
        }
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

    private func collisionRebalance1(_ bucket: Int) -> Double {
        let overload = max(0.0, hashLoad1[bucket] - hashMean1)
        return 1.0 / sqrt(1.0 + overload)
    }

    private func collisionRebalance2(_ bucket: Int) -> Double {
        let overload = max(0.0, hashLoad2[bucket] - hashMean2)
        return 1.0 / sqrt(1.0 + overload)
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

    private static func dot(_ weights: [Double], _ x: [Double]) -> Double {
        var value = 0.0
        let count = min(weights.count, x.count)
        for index in 0..<count {
            value += weights[index] * x[index]
        }
        return value
    }

    private static func softmax3(_ logits: [Double]) -> [Double] {
        let maxLogit = logits.prefix(Self.classCount).map { fxSafeFinite($0) }.max() ?? 0.0
        var values = Array(repeating: 0.0, count: Self.classCount)
        var sum = 0.0
        for classIndex in 0..<Self.classCount {
            let value = exp(fxClamp(fxSafeFinite(logits[classIndex]) - maxLogit, -30.0, 30.0))
            values[classIndex] = value
            sum += value
        }
        guard sum > 0.0 else { return [0.3333333, 0.3333333, 0.3333333] }
        return values.map { $0 / sum }
    }

    private static func isotonicBin(_ probability: Double) -> Int {
        min(max(Int(floor(fxClamp(probability, 0.0, 0.999999) * Double(Self.calibrationBins))), 0), Self.calibrationBins - 1)
    }

    public static func hashIndex(_ i: Int, _ j: Int) -> Int {
        let a = UInt32(truncatingIfNeeded: i &* 73_856_093)
        let b = UInt32(truncatingIfNeeded: j &* 19_349_663)
        return Int((a ^ b) % UInt32(Self.hashBuckets1))
    }

    public static func hashIndex2(_ i: Int, _ j: Int) -> Int {
        let a = UInt32(truncatingIfNeeded: i &* 83_492_791)
        let b = UInt32(truncatingIfNeeded: j) &* 2_654_435_761
        return Int((a ^ b) % UInt32(Self.hashBuckets2))
    }

    public static func hashSign(_ i: Int, _ j: Int) -> Double {
        ((i &* 31 + j &* 17) & 1) == 0 ? 1.0 : -1.0
    }
}
