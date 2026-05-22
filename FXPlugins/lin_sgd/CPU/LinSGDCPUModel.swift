import FXDataEngine
import Foundation

public struct LinSGDCPUModel: Sendable {
    private static let classCount = 3
    private static let hashBuckets = 192
    private static let isotonicBins = 10

    private var step: Int
    private var classWeights: [[Double]]
    private var classM: [[Double]]
    private var classV: [[Double]]
    private var hashWeights: [[Double]]
    private var hashM: [[Double]]
    private var hashV: [[Double]]
    private var moveWeights: [Double]
    private var moveM: [Double]
    private var moveV: [Double]
    private var moveHashWeights: [Double]
    private var moveHashM: [Double]
    private var moveHashV: [Double]
    private var moveReady: Bool
    private var moveEMAAbs: Double
    private var calibrationTemperature: Double
    private var calibrationBias: [Double]
    private var isotonicPositive: [[Double]]
    private var isotonicCount: [[Double]]
    private var calibrationSteps: Int
    private var lossReady: Bool
    private var lossFast: Double
    private var lossSlow: Double
    private var driftCooldown: Int
    private var classEMA: [Double]

    public init() {
        let featureCount = FXDataEngineConstants.aiWeights
        self.step = 0
        self.classWeights = Array(repeating: Array(repeating: 0.0, count: featureCount), count: Self.classCount)
        self.classM = Array(repeating: Array(repeating: 0.0, count: featureCount), count: Self.classCount)
        self.classV = Array(repeating: Array(repeating: 0.0, count: featureCount), count: Self.classCount)
        self.hashWeights = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets), count: Self.classCount)
        self.hashM = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets), count: Self.classCount)
        self.hashV = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets), count: Self.classCount)
        self.moveWeights = Array(repeating: 0.0, count: featureCount)
        self.moveM = Array(repeating: 0.0, count: featureCount)
        self.moveV = Array(repeating: 0.0, count: featureCount)
        self.moveHashWeights = Array(repeating: 0.0, count: Self.hashBuckets)
        self.moveHashM = Array(repeating: 0.0, count: Self.hashBuckets)
        self.moveHashV = Array(repeating: 0.0, count: Self.hashBuckets)
        self.moveReady = false
        self.moveEMAAbs = 0.0
        self.calibrationTemperature = 1.0
        self.calibrationBias = Array(repeating: 0.0, count: Self.classCount)
        self.isotonicPositive = Array(repeating: Array(repeating: 0.0, count: Self.isotonicBins), count: Self.classCount)
        self.isotonicCount = Array(repeating: Array(repeating: 0.0, count: Self.isotonicBins), count: Self.classCount)
        self.calibrationSteps = 0
        self.lossReady = false
        self.lossFast = 0.0
        self.lossSlow = 0.0
        self.driftCooldown = 0
        self.classEMA = Array(repeating: 1.0, count: Self.classCount)
    }

    public mutating func reset() {
        self = LinSGDCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        var hp = PluginSupportTools.scaleHyperParametersForMove(hyperParameters, movePoints: request.movePoints)
        let priceCost = PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: request.context.priceCostPoints)
        let edgeWeight = PluginSupportTools.moveSampleWeight(
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: priceCost,
            minMovePoints: request.context.minMovePoints,
            qualityTargets: PluginQualityTargets(request: request)
        )
        let excess = max(0.0, abs(fxSafeFinite(request.movePoints)) - priceCost)
        hp.learningRate = fxClamp(hp.learningRate * (1.0 + 0.08 * excess), 0.00002, 0.25000)
        hp.l2 = fxClamp(hp.l2 * (1.0 - 0.15 * fxClamp(excess / 12.0, 0.0, 1.0)), 0.0, 0.10)
        let classWeight = label == .skip ? 0.80 : 1.0
        let sampleWeight = fxClamp(edgeWeight * classWeight * request.sampleWeight, 0.10, 6.00)
        updateWeighted(label: label, x: x, hyperParameters: hp, sampleWeight: sampleWeight, movePoints: request.movePoints)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let rawProbabilities = Self.softmax3(buildClassLogits(x))
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution(calibrated(rawProbabilities))
        var moveMean = predictMoveRaw(x)
        if moveReady, moveEMAAbs > 0.0 {
            moveMean = 0.65 * moveMean + 0.35 * moveEMAAbs
        }
        moveMean = max(moveMean, 0.0)
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
            reliability: fxClamp(0.45 + 0.25 * (moveReady ? 1.0 : 0.0) + 0.30 * min(Double(step) / 64.0, 1.0), 0.0, 1.0),
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
            classEMA[classIndex] = 0.995 * classEMA[classIndex] + (classIndex == label.rawValue ? 0.005 : 0.0)
        }
        let meanCount = classEMA.reduce(0.0, +) / Double(Self.classCount)
        let classBalance = fxClamp(meanCount / max(classEMA[label.rawValue], 0.01), 0.60, 2.20)
        let weight = fxClamp(sampleWeight * classBalance, 0.10, 6.00)
        let learningRate = scheduledLearningRate(hyperParameters, sampleWeight: weight)
        let l2 = fxClamp(hyperParameters.l2, 0.0, 0.0500)
        let rawProbabilities = Self.softmax3(buildClassLogits(x))

        var classGradient = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            let target = classIndex == label.rawValue ? 1.0 : 0.0
            classGradient[classIndex] = PluginSupportTools.clipSymmetric((target - rawProbabilities[classIndex]) * weight, limit: 4.0)
        }

        var linearGradients = Array(
            repeating: Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights),
            count: Self.classCount
        )
        var hashGradients = Array(
            repeating: Array(repeating: 0.0, count: Self.hashBuckets),
            count: Self.classCount
        )

        for classIndex in 0..<Self.classCount {
            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                linearGradients[classIndex][featureIndex] = PluginSupportTools.clipSymmetric(
                    classGradient[classIndex] * x[featureIndex],
                    limit: 8.0
                )
            }
        }
        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                let hashedValue = Self.hashSign(i, j) * x[i] * x[j]
                let bucket = Self.hashIndex(i, j)
                for classIndex in 0..<Self.classCount {
                    hashGradients[classIndex][bucket] += PluginSupportTools.clipSymmetric(
                        classGradient[classIndex] * hashedValue,
                        limit: 8.0
                    )
                }
            }
        }

        clipGradients(linear: &linearGradients, hashed: &hashGradients, sampleWeight: weight)
        adamWUpdate(linear: linearGradients, hashed: hashGradients, learningRate: learningRate, l2: l2)

        let loss = -log(fxClamp(rawProbabilities[label.rawValue], 1.0e-6, 1.0))
        applyDriftGuard(crossEntropyLoss: loss)
        updateCalibrator(rawProbabilities: rawProbabilities, label: label, sampleWeight: weight, learningRate: learningRate)
        updateMoveHead(x: x, movePoints: movePoints, hyperParameters: hyperParameters, sampleWeight: weight)
    }

    private mutating func adamWUpdate(
        linear: [[Double]],
        hashed: [[Double]],
        learningRate: Double,
        l2: Double
    ) {
        let beta1 = 0.90
        let beta2 = 0.999
        let epsilon = 1.0e-8
        let time = Double(max(step, 1))
        let biasCorrection1 = max(1.0 - pow(beta1, time), 1.0e-8)
        let biasCorrection2 = max(1.0 - pow(beta2, time), 1.0e-8)

        for classIndex in 0..<Self.classCount {
            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                if featureIndex != 0 {
                    classWeights[classIndex][featureIndex] *= (1.0 - learningRate * l2)
                }
                let gradient = linear[classIndex][featureIndex]
                classM[classIndex][featureIndex] = beta1 * classM[classIndex][featureIndex] + (1.0 - beta1) * gradient
                classV[classIndex][featureIndex] = beta2 * classV[classIndex][featureIndex] + (1.0 - beta2) * gradient * gradient
                let mhat = classM[classIndex][featureIndex] / biasCorrection1
                let vhat = classV[classIndex][featureIndex] / biasCorrection2
                classWeights[classIndex][featureIndex] = PluginSupportTools.clipSymmetric(
                    classWeights[classIndex][featureIndex] + learningRate * (mhat / (sqrt(vhat) + epsilon)),
                    limit: 25.0
                )
            }
            for bucket in 0..<Self.hashBuckets {
                hashWeights[classIndex][bucket] *= (1.0 - learningRate * l2)
                let gradient = hashed[classIndex][bucket]
                hashM[classIndex][bucket] = beta1 * hashM[classIndex][bucket] + (1.0 - beta1) * gradient
                hashV[classIndex][bucket] = beta2 * hashV[classIndex][bucket] + (1.0 - beta2) * gradient * gradient
                let mhat = hashM[classIndex][bucket] / biasCorrection1
                let vhat = hashV[classIndex][bucket] / biasCorrection2
                hashWeights[classIndex][bucket] = PluginSupportTools.clipSymmetric(
                    hashWeights[classIndex][bucket] + learningRate * (mhat / (sqrt(vhat) + epsilon)),
                    limit: 20.0
                )
            }
        }
    }

    private mutating func updateMoveHead(
        x: [Double],
        movePoints: Double,
        hyperParameters: HyperParameters,
        sampleWeight: Double
    ) {
        let target = abs(fxSafeFinite(movePoints, fallback: .nan))
        guard target.isFinite else { return }
        let prediction = predictMoveRaw(x)
        let error = target - prediction
        let huberGradient = abs(error) <= 1.0 ? error : (error > 0.0 ? 1.0 : -1.0)
        let weight = fxClamp(sampleWeight, 0.25, 4.00)
        let learningRate = scheduledMoveLearningRate(hyperParameters, sampleWeight: weight)
        let weightDecay = fxClamp(0.20 * hyperParameters.l2, 0.0, 0.0500)
        let beta1 = 0.90
        let beta2 = 0.999
        let epsilon = 1.0e-8
        let time = Double(max(step, 1))
        let biasCorrection1 = max(1.0 - pow(beta1, time), 1.0e-8)
        let biasCorrection2 = max(1.0 - pow(beta2, time), 1.0e-8)

        var linearGradients = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        var norm2 = 0.0
        for featureIndex in 0..<FXDataEngineConstants.aiWeights {
            let gradient = PluginSupportTools.clipSymmetric(weight * huberGradient * x[featureIndex], limit: 6.0)
            linearGradients[featureIndex] = gradient
            norm2 += gradient * gradient
        }
        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                let gradient = PluginSupportTools.clipSymmetric(weight * huberGradient * Self.hashSign(i, j) * x[i] * x[j], limit: 6.0)
                norm2 += gradient * gradient
            }
        }
        let scale = norm2 > 0.0 && sqrt(norm2) > 6.0 ? 6.0 / sqrt(norm2) : 1.0
        for featureIndex in 0..<FXDataEngineConstants.aiWeights {
            let gradient = linearGradients[featureIndex] * scale
            if featureIndex != 0 {
                moveWeights[featureIndex] *= (1.0 - learningRate * weightDecay)
            }
            moveM[featureIndex] = beta1 * moveM[featureIndex] + (1.0 - beta1) * gradient
            moveV[featureIndex] = beta2 * moveV[featureIndex] + (1.0 - beta2) * gradient * gradient
            moveWeights[featureIndex] = PluginSupportTools.clipSymmetric(
                moveWeights[featureIndex] + learningRate * ((moveM[featureIndex] / biasCorrection1) / (sqrt(moveV[featureIndex] / biasCorrection2) + epsilon)),
                limit: 20.0
            )
        }
        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                let bucket = Self.hashIndex(i, j)
                let gradient = PluginSupportTools.clipSymmetric(weight * huberGradient * Self.hashSign(i, j) * x[i] * x[j], limit: 6.0) * scale
                moveHashWeights[bucket] *= (1.0 - learningRate * weightDecay)
                moveHashM[bucket] = beta1 * moveHashM[bucket] + (1.0 - beta1) * gradient
                moveHashV[bucket] = beta2 * moveHashV[bucket] + (1.0 - beta2) * gradient * gradient
                moveHashWeights[bucket] = PluginSupportTools.clipSymmetric(
                    moveHashWeights[bucket] + learningRate * ((moveHashM[bucket] / biasCorrection1) / (sqrt(moveHashV[bucket] / biasCorrection2) + epsilon)),
                    limit: 15.0
                )
            }
        }
        updateMoveEMA(movePoints: movePoints, alpha: 0.05)
    }

    private mutating func updateCalibrator(
        rawProbabilities: [Double],
        label: LabelClass,
        sampleWeight: Double,
        learningRate: Double
    ) {
        let weight = fxClamp(sampleWeight, 0.25, 4.00)
        let calibrationLearningRate = fxClamp(0.20 * learningRate * weight, 0.0002, 0.0200)
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

    private mutating func applyDriftGuard(crossEntropyLoss: Double) {
        if !lossReady {
            lossFast = crossEntropyLoss
            lossSlow = crossEntropyLoss
            lossReady = true
            return
        }
        lossFast = 0.90 * lossFast + 0.10 * crossEntropyLoss
        lossSlow = 0.99 * lossSlow + 0.01 * crossEntropyLoss
        if driftCooldown > 0 { driftCooldown -= 1 }
        guard step >= 256, driftCooldown == 0 else { return }
        if lossFast > 1.8 * max(lossSlow, 0.10) {
            for classIndex in 0..<Self.classCount {
                for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                    classM[classIndex][featureIndex] *= 0.20
                    classV[classIndex][featureIndex] *= 0.20
                }
                for bucket in 0..<Self.hashBuckets {
                    hashM[classIndex][bucket] *= 0.20
                    hashV[classIndex][bucket] *= 0.20
                }
            }
            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                moveM[featureIndex] *= 0.20
                moveV[featureIndex] *= 0.20
            }
            for bucket in 0..<Self.hashBuckets {
                moveHashM[bucket] *= 0.20
                moveHashV[bucket] *= 0.20
            }
            driftCooldown = 64
        }
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
            var monotonic = Array(repeating: 0.0, count: Self.isotonicBins)
            for bin in 0..<Self.isotonicBins {
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

    private func buildClassLogits(_ x: [Double]) -> [Double] {
        var logits = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            var value = 0.0
            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                value += classWeights[classIndex][featureIndex] * x[featureIndex]
            }
            for i in 1..<FXDataEngineConstants.aiWeights {
                for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                    let bucket = Self.hashIndex(i, j)
                    value += hashWeights[classIndex][bucket] * Self.hashSign(i, j) * x[i] * x[j]
                }
            }
            logits[classIndex] = PluginSupportTools.clipSymmetric(value, limit: 35.0)
        }
        return logits
    }

    private func predictMoveRaw(_ x: [Double]) -> Double {
        var value = 0.0
        for featureIndex in 0..<FXDataEngineConstants.aiWeights {
            value += moveWeights[featureIndex] * x[featureIndex]
        }
        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                let bucket = Self.hashIndex(i, j)
                value += moveHashWeights[bucket] * Self.hashSign(i, j) * x[i] * x[j]
            }
        }
        return max(value, 0.0)
    }

    private func scheduledLearningRate(_ hyperParameters: HyperParameters, sampleWeight: Double) -> Double {
        let base = fxClamp(hyperParameters.learningRate, 0.00002, 0.20000)
        let time = Double(max(step, 1))
        let warmup = fxClamp(time / 128.0, 0.10, 1.00)
        let inverseSqrt = 1.0 / sqrt(1.0 + 0.004 * max(0.0, time - 128.0))
        let phase = time.truncatingRemainder(dividingBy: 2048.0) / 2048.0
        let cosine = 0.60 + 0.40 * (0.5 * (1.0 + cos(Double.pi * phase)))
        let sampleScale = fxClamp(0.80 + 0.20 * fxClamp(sampleWeight, 0.25, 4.00), 0.70, 1.60)
        return fxClamp(base * warmup * inverseSqrt * cosine * sampleScale, 0.00001, 0.08000)
    }

    private func scheduledMoveLearningRate(_ hyperParameters: HyperParameters, sampleWeight: Double) -> Double {
        let base = fxClamp(0.65 * hyperParameters.learningRate, 0.00001, 0.08000)
        return fxClamp(base * (0.90 + 0.10 * fxClamp(sampleWeight, 0.25, 4.00)), 0.00001, 0.05000)
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

    private func clipGradients(linear: inout [[Double]], hashed: inout [[Double]], sampleWeight: Double) {
        var normSquared = 0.0
        for classIndex in 0..<Self.classCount {
            for value in linear[classIndex] { normSquared += value * value }
            for value in hashed[classIndex] { normSquared += value * value }
        }
        guard normSquared > 0.0 else { return }
        let norm = sqrt(normSquared)
        let clip = fxClamp(8.0 + sqrt(sampleWeight), 6.0, 12.0)
        guard norm > clip else { return }
        let scale = clip / norm
        for classIndex in 0..<Self.classCount {
            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                linear[classIndex][featureIndex] *= scale
            }
            for bucket in 0..<Self.hashBuckets {
                hashed[classIndex][bucket] *= scale
            }
        }
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
            let value = exp(fxClamp(fxSafeFinite(logits[classIndex]) - maxLogit, -35.0, 35.0))
            values[classIndex] = value
            sum += value
        }
        guard sum > 0.0 else { return [0.3333333, 0.3333333, 0.3333333] }
        return values.map { $0 / sum }
    }

    private static func isotonicBin(_ probability: Double) -> Int {
        min(max(Int(floor(fxClamp(probability, 0.0, 0.999999) * Double(Self.isotonicBins))), 0), Self.isotonicBins - 1)
    }

    public static func hashIndex(_ i: Int, _ j: Int) -> Int {
        let a = UInt32(truncatingIfNeeded: i &* 73_856_093)
        let b = UInt32(truncatingIfNeeded: j &* 19_349_663)
        return Int((a ^ b) % UInt32(Self.hashBuckets))
    }

    public static func hashSign(_ i: Int, _ j: Int) -> Double {
        ((i &* 31 + j &* 17) & 1) == 0 ? 1.0 : -1.0
    }
}
