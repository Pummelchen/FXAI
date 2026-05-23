import FXDataEngine
import Foundation

public struct LinPACPUModel: Sendable {
    private struct ReplaySample: Sendable {
        let x: [Double]
        let label: Int
        let movePoints: Double
        let sampleWeight: Double
        let hardness: Double
    }

    private static let classCount = 3
    private static let hashBuckets1 = 128
    private static let hashBuckets2 = 97
    private static let calibrationBins = 10
    private static let replayCapacity = 192
    private static let topRivals = 2

    private var linearWeights: [[Double]]
    private var averagedLinearWeights: [[Double]]
    private var linearSigma: [[Double]]
    private var hashWeights1: [[Double]]
    private var averagedHashWeights1: [[Double]]
    private var hashSigma1: [[Double]]
    private var hashWeights2: [[Double]]
    private var averagedHashWeights2: [[Double]]
    private var hashSigma2: [[Double]]
    private var hashOccupancy1: [Int]
    private var hashOccupancy2: [Int]
    private var hashBandwidth1: [Double]
    private var hashBandwidth2: [Double]
    private var hash2Scale: Double
    private var hashOccupancyReady: Bool
    private var moveMuLinear: [Double]
    private var moveMuHash1: [Double]
    private var moveMuHash2: [Double]
    private var moveLogVarLinear: [Double]
    private var moveLogVarHash1: [Double]
    private var moveLogVarHash2: [Double]
    private var moveLogVarBias: Double
    private var moveSteps: Int
    private var moveReady: Bool
    private var moveEMAAbs: Double
    private var steps: Int
    private var marginReady: Bool
    private var marginEMA: Double
    private var paMode: Int
    private var confidenceRadius: Double
    private var classEMA: [Double]
    private var lossReady: Bool
    private var lossFast: Double
    private var lossSlow: Double
    private var driftCooldown: Int
    private var calibrationWeights: [[Double]]
    private var calibrationBias: [Double]
    private var isotonicPositive: [[Double]]
    private var isotonicCount: [[Double]]
    private var calibrationSteps: Int
    private var guardReady: Bool
    private var guardUseAverage: Bool
    private var guardLiveFast: Double
    private var guardLiveSlow: Double
    private var guardAverageFast: Double
    private var guardAverageSlow: Double
    private var replay: [ReplaySample]
    private var replayHead: Int
    private var rng: PluginDeterministicRNG
    private let useHash: Bool
    private let useHash2: Bool

    public init() {
        let featureCount = FXDataEngineConstants.aiWeights
        self.linearWeights = Array(repeating: Array(repeating: 0.0, count: featureCount), count: Self.classCount)
        self.averagedLinearWeights = Array(repeating: Array(repeating: 0.0, count: featureCount), count: Self.classCount)
        self.linearSigma = Array(repeating: Array(repeating: 1.0, count: featureCount), count: Self.classCount)
        self.hashWeights1 = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets1), count: Self.classCount)
        self.averagedHashWeights1 = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets1), count: Self.classCount)
        self.hashSigma1 = Array(repeating: Array(repeating: 1.0, count: Self.hashBuckets1), count: Self.classCount)
        self.hashWeights2 = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets2), count: Self.classCount)
        self.averagedHashWeights2 = Array(repeating: Array(repeating: 0.0, count: Self.hashBuckets2), count: Self.classCount)
        self.hashSigma2 = Array(repeating: Array(repeating: 1.0, count: Self.hashBuckets2), count: Self.classCount)
        self.hashOccupancy1 = Array(repeating: 0, count: Self.hashBuckets1)
        self.hashOccupancy2 = Array(repeating: 0, count: Self.hashBuckets2)
        self.hashBandwidth1 = Array(repeating: 1.0, count: Self.hashBuckets1)
        self.hashBandwidth2 = Array(repeating: 1.0, count: Self.hashBuckets2)
        self.hash2Scale = 1.0
        self.hashOccupancyReady = false
        self.moveMuLinear = Array(repeating: 0.0, count: featureCount)
        self.moveMuHash1 = Array(repeating: 0.0, count: Self.hashBuckets1)
        self.moveMuHash2 = Array(repeating: 0.0, count: Self.hashBuckets2)
        self.moveLogVarLinear = Array(repeating: 0.0, count: featureCount)
        self.moveLogVarHash1 = Array(repeating: 0.0, count: Self.hashBuckets1)
        self.moveLogVarHash2 = Array(repeating: 0.0, count: Self.hashBuckets2)
        self.moveLogVarBias = 0.0
        self.moveSteps = 0
        self.moveReady = false
        self.moveEMAAbs = 0.0
        self.steps = 0
        self.marginReady = false
        self.marginEMA = 0.0
        self.paMode = 0
        self.confidenceRadius = 1.0
        self.classEMA = Array(repeating: 1.0, count: Self.classCount)
        self.lossReady = false
        self.lossFast = 0.0
        self.lossSlow = 0.0
        self.driftCooldown = 0
        self.calibrationWeights = Self.identityCalibrationWeights()
        self.calibrationBias = Array(repeating: 0.0, count: Self.classCount)
        self.isotonicPositive = Array(repeating: Array(repeating: 0.0, count: Self.calibrationBins), count: Self.classCount)
        self.isotonicCount = Array(repeating: Array(repeating: 0.0, count: Self.calibrationBins), count: Self.classCount)
        self.calibrationSteps = 0
        self.guardReady = false
        self.guardUseAverage = false
        self.guardLiveFast = 0.0
        self.guardLiveSlow = 0.0
        self.guardAverageFast = 0.0
        self.guardAverageSlow = 0.0
        self.replay = []
        self.replayHead = 0
        self.rng = PluginDeterministicRNG(aiID: AIModelID.paLinear.rawValue)
        self.useHash = true
        self.useHash2 = true
        buildCollisionProfile()
    }

    public mutating func reset() {
        self = LinPACPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        var label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        let scaledHyperParameters = PluginSupportTools.scaleHyperParametersForMove(hyperParameters, movePoints: request.movePoints)
        let cost = PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: request.context.priceCostPoints)
        let absoluteMove = abs(fxSafeFinite(request.movePoints))
        let excess = max(0.0, absoluteMove - cost)
        var edgeRatio = excess / max(cost, 0.50)

        if label == .skip {
            let skipPrior = fxClamp(1.35 - 0.30 * edgeRatio, 0.25, 1.50)
            if edgeRatio > 1.8, absoluteMove > 1.2 * max(cost, 0.50) {
                label = request.movePoints >= 0.0 ? .buy : .sell
            } else {
                edgeRatio *= skipPrior
            }
        } else if edgeRatio < 0.05 {
            label = .skip
        }

        var eventWeight = fxClamp(0.35 + edgeRatio, 0.10, 6.00)
        if label == .skip {
            eventWeight *= fxClamp(1.20 - 0.20 * edgeRatio, 0.25, 1.40)
        } else {
            eventWeight *= fxClamp(0.65 + 0.40 * edgeRatio, 0.40, 2.50)
        }
        eventWeight = fxClamp(eventWeight * request.sampleWeight, 0.10, 6.00)

        let moveScale = fxClamp(1.0 + 0.10 * excess, 0.70, 3.50)
        var marginScale = fxClamp(moveScale * (1.0 + 0.25 * (eventWeight - 1.0)), 0.60, 4.00)
        let session = Self.sessionBucket(sampleTimeUTC: request.context.sampleTimeUTC, fallback: request.context.sessionBucket)
        let sessionScale: Double
        switch session {
        case 0: sessionScale = 1.10
        case 1: sessionScale = 0.95
        case 2: sessionScale = 0.92
        default: sessionScale = 1.05
        }
        let volatilityProxy = abs(x[safe: 1]) + 0.7 * abs(x[safe: 2]) + 0.5 * abs(x[safe: 3])
        let regimeScale: Double
        if volatilityProxy < 0.20 {
            regimeScale = 1.10
        } else if volatilityProxy > 2.00 {
            regimeScale = 0.90
        } else {
            regimeScale = 1.00
        }
        let costScale = fxClamp(1.0 + 0.08 * (cost - 1.0), 0.80, 1.30)
        marginScale = fxClamp(marginScale * sessionScale * regimeScale * costScale, 0.50, 5.00)

        updateWeighted(
            label: label.rawValue,
            x: x,
            hyperParameters: scaledHyperParameters,
            marginScale: marginScale,
            sampleWeight: eventWeight,
            movePoints: request.movePoints,
            fromReplay: false
        )

        let replayCount: Int
        if replay.count >= 96 {
            replayCount = 2
        } else if replay.count >= 24 {
            replayCount = 1
        } else {
            replayCount = 0
        }
        for _ in 0..<replayCount {
            guard let sample = pickHardReplay() else { break }
            updateWeighted(
                label: sample.label,
                x: sample.x,
                hyperParameters: scaledHyperParameters,
                marginScale: marginScale,
                sampleWeight: fxClamp(0.75 * sample.sampleWeight, 0.10, 4.00),
                movePoints: sample.movePoints,
                fromReplay: true
            )
        }
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let averaged = guardReady ? guardUseAverage : steps > 24
        let scores = computeScores(x, averaged: averaged)
        let denominator = max(marginReady ? marginEMA : fxClamp(hyperParameters.passiveAggressiveMargin, 0.25, 4.0), 0.25)
        let logits = scores.map { PluginSupportTools.clipSymmetric($0 / denominator, limit: 20.0) }
        let rawProbabilities = Self.softmax3(logits)
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution(calibrated(rawProbabilities))
        let move = predictMoveDistribution(x)
        var mean = predictExpectedMovePoints(x)
        mean = max(0.0, mean)
        let sigma = sqrt(max(exp(move.logVariance), 1.0e-6))
        let q25 = max(0.0, move.mean - 0.60 * sigma)
        let q50 = max(q25, max(0.0, move.mean))
        let q75 = max(q50, max(0.0, move.mean + 0.60 * sigma))
        let directional = max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue])
        let lossDrag = lossReady ? fxClamp(lossFast / max(lossSlow, 0.10), 0.0, 2.0) : 1.0
        let output = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: mean,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: max(q75, mean),
            mfeMeanPoints: max(mean, 1.15 * q75),
            maeMeanPoints: max(0.0, 0.35 * mean + 0.10 * request.context.priceCostPoints),
            hitTimeFraction: fxClamp(0.70 - 0.20 * directional + 0.05 * probabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            pathRisk: fxClamp(probabilities[LabelClass.skip.rawValue] + 0.08 * max(0.0, lossDrag - 1.0), 0.0, 1.0),
            fillRisk: fxClamp(request.context.priceCostPoints / max(mean, request.context.minMovePoints, 0.25), 0.0, 1.0),
            confidence: fxClamp(directional, 0.0, 1.0),
            reliability: fxClamp(0.45 + 0.25 * (moveReady ? 1.0 : 0.0) + 0.30 * min(Double(moveSteps) / 64.0, 1.0), 0.0, 1.0),
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

    private mutating func updateWeighted(
        label: Int,
        x: [Double],
        hyperParameters: HyperParameters,
        marginScale: Double,
        sampleWeight: Double,
        movePoints: Double,
        fromReplay: Bool
    ) {
        guard (0..<Self.classCount).contains(label) else { return }
        steps += 1

        for classIndex in 0..<Self.classCount {
            classEMA[classIndex] = 0.997 * classEMA[classIndex] + (classIndex == label ? 0.003 : 0.0)
        }
        let meanClass = classEMA.reduce(0.0, +) / Double(Self.classCount)
        let classBalance = fxClamp(meanClass / max(classEMA[label], 0.005), 0.60, 2.50)
        let recency = 0.85 + 0.30 * (1.0 - exp(-Double(steps) / 512.0))
        let weightedSample = fxClamp(sampleWeight * classBalance * recency, 0.10, 6.00)

        let volatilityProxy = abs(x[safe: 1]) + 0.5 * abs(x[safe: 2]) + 0.35 * abs(x[safe: 3])
        let confidenceTarget = fxClamp(0.50 + volatilityProxy, 0.25, 4.00)
        confidenceRadius = fxClamp(0.995 * confidenceRadius + 0.005 * confidenceTarget, 0.20, 5.00)

        var cValue = fxClamp(hyperParameters.passiveAggressiveC, 0.001, 100.0)
        var margin = fxClamp(fxClamp(hyperParameters.passiveAggressiveMargin, 0.05, 4.0) * marginScale, 0.05, 8.0)
        let decay = fxClamp(hyperParameters.l2, 0.0, 0.2)
        if fromReplay {
            cValue = fxClamp(cValue * 0.85, 0.001, 100.0)
            margin = fxClamp(margin * 1.05, 0.05, 8.0)
        }
        if driftCooldown > 0 {
            cValue = fxClamp(cValue * 0.70, 0.001, 100.0)
            margin = fxClamp(margin * 1.10, 0.05, 8.0)
        }

        applyDecay(decay)
        updateCollisionRebalance()

        let scores = computeScores(x, averaged: false)
        let rivals = topRivalClasses(scores: scores, label: label)
        guard !rivals.isEmpty else { return }

        var totalHardness = 0.0
        for rivalRank in 0..<min(Self.topRivals, rivals.count) {
            let rival = rivals[rivalRank]
            let loss = margin - scores[label] + scores[rival]
            guard loss > 0.0 else { continue }
            totalHardness += loss

            let excess = max(0.0, abs(movePoints) - PluginContextRuntimeTools.inputPriceCostPoints(x))
            let mode = selectPAMode(loss: loss, excessPoints: excess)
            paMode = mode
            let normSquared = computeDiffNormSquared(x, label: label, rival: rival)
            var tau = computeTau(mode: mode, loss: loss, cValue: cValue, normSquared: normSquared)
            tau *= weightedSample
            if rivalRank == 1 {
                tau *= 0.60
            }
            if driftCooldown > 0 {
                tau *= 0.80
            }
            tau = fxClamp(tau, 0.0, cValue)
            guard tau > 0.0 else { continue }

            updateLinearWeights(label: label, rival: rival, x: x, tau: tau)
            if useHash {
                updateHashedWeights(label: label, rival: rival, x: x, tau: tau)
            }
        }

        updateAveragedWeights()
        let scoresPost = computeScores(x, averaged: false)
        let scoresAverage = computeScores(x, averaged: true)
        guard let bestRival = bestRival(scores: scoresPost, label: label) else { return }
        let marginGap = scoresPost[label] - scoresPost[bestRival]
        let absoluteMargin = abs(marginGap)
        if !marginReady {
            marginEMA = max(absoluteMargin, 0.25)
            marginReady = true
        } else {
            marginEMA = max(0.95 * marginEMA + 0.05 * absoluteMargin, 0.25)
        }

        let denominator = max(marginEMA, 0.25)
        let probabilities = Self.softmax3(scoresPost.map { PluginSupportTools.clipSymmetric($0 / denominator, limit: 20.0) })
        let averageProbabilities = Self.softmax3(scoresAverage.map { PluginSupportTools.clipSymmetric($0 / denominator, limit: 20.0) })
        let ce = -log(fxClamp(probabilities[label], 1.0e-6, 1.0))
        let ceAverage = -log(fxClamp(averageProbabilities[label], 1.0e-6, 1.0))
        updateABGuard(liveCrossEntropy: ce, averagedCrossEntropy: ceAverage)
        updateLossDrift(crossEntropyLoss: ce)
        updateCalibrator(rawProbabilities: probabilities, label: label, sampleWeight: weightedSample, learningRate: fxClamp(0.01 * sqrt(weightedSample), 0.0005, 0.0300))
        updateMoveHead(x: x, movePoints: movePoints, hyperParameters: hyperParameters, sampleWeight: weightedSample)

        if !fromReplay {
            pushReplay(label: label, x: x, movePoints: movePoints, sampleWeight: weightedSample, hardness: ce + 0.30 * totalHardness)
        }
    }

    private mutating func updateLinearWeights(label: Int, rival: Int, x: [Double], tau: Double) {
        for featureIndex in 0..<FXDataEngineConstants.aiWeights {
            let value = x[featureIndex]
            let sigmaLabel = linearSigma[label][featureIndex]
            let sigmaRival = linearSigma[rival][featureIndex]
            linearWeights[label][featureIndex] += tau * sigmaLabel * value
            linearWeights[rival][featureIndex] -= tau * sigmaRival * value
            let squared = value * value
            linearSigma[label][featureIndex] = fxClamp(
                1.0 / (1.0 / max(sigmaLabel, 1.0e-6) + squared / max(confidenceRadius, 1.0e-3)),
                1.0e-5,
                10.0
            )
            linearSigma[rival][featureIndex] = fxClamp(
                1.0 / (1.0 / max(sigmaRival, 1.0e-6) + squared / max(confidenceRadius, 1.0e-3)),
                1.0e-5,
                10.0
            )
        }
    }

    private mutating func updateHashedWeights(label: Int, rival: Int, x: [Double], tau: Double) {
        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                let hashedValue = Self.hashSign(i, j) * x[i] * x[j]
                if abs(hashedValue) < 1.0e-12 { continue }
                let h1 = Self.hashIndex(i, j)
                let value1 = hashBandwidth1[h1] * hashedValue
                let sigmaLabel1 = hashSigma1[label][h1]
                let sigmaRival1 = hashSigma1[rival][h1]
                hashWeights1[label][h1] += tau * sigmaLabel1 * value1
                hashWeights1[rival][h1] -= tau * sigmaRival1 * value1
                let squared1 = value1 * value1
                hashSigma1[label][h1] = fxClamp(
                    1.0 / (1.0 / max(sigmaLabel1, 1.0e-6) + squared1 / max(confidenceRadius, 1.0e-3)),
                    1.0e-5,
                    10.0
                )
                hashSigma1[rival][h1] = fxClamp(
                    1.0 / (1.0 / max(sigmaRival1, 1.0e-6) + squared1 / max(confidenceRadius, 1.0e-3)),
                    1.0e-5,
                    10.0
                )

                if useHash2 {
                    let h2 = Self.hashIndex2(i, j)
                    let value2 = hash2Scale * hashBandwidth2[h2] * hashedValue
                    let sigmaLabel2 = hashSigma2[label][h2]
                    let sigmaRival2 = hashSigma2[rival][h2]
                    hashWeights2[label][h2] += tau * sigmaLabel2 * value2
                    hashWeights2[rival][h2] -= tau * sigmaRival2 * value2
                    let squared2 = value2 * value2
                    hashSigma2[label][h2] = fxClamp(
                        1.0 / (1.0 / max(sigmaLabel2, 1.0e-6) + squared2 / max(confidenceRadius, 1.0e-3)),
                        1.0e-5,
                        10.0
                    )
                    hashSigma2[rival][h2] = fxClamp(
                        1.0 / (1.0 / max(sigmaRival2, 1.0e-6) + squared2 / max(confidenceRadius, 1.0e-3)),
                        1.0e-5,
                        10.0
                    )
                }
            }
        }
    }

    private mutating func updateAveragedWeights() {
        let beta = 1.0 / Double(max(steps, 1))
        for classIndex in 0..<Self.classCount {
            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                averagedLinearWeights[classIndex][featureIndex] += beta * (linearWeights[classIndex][featureIndex] - averagedLinearWeights[classIndex][featureIndex])
            }
            for bucket in 0..<Self.hashBuckets1 {
                averagedHashWeights1[classIndex][bucket] += beta * (hashWeights1[classIndex][bucket] - averagedHashWeights1[classIndex][bucket])
            }
            for bucket in 0..<Self.hashBuckets2 {
                averagedHashWeights2[classIndex][bucket] += beta * (hashWeights2[classIndex][bucket] - averagedHashWeights2[classIndex][bucket])
            }
        }
    }

    private func computeDiffNormSquared(_ x: [Double], label: Int, rival: Int) -> Double {
        var normSquared = 0.0
        for featureIndex in 0..<FXDataEngineConstants.aiWeights {
            let value = x[featureIndex]
            normSquared += (linearSigma[label][featureIndex] + linearSigma[rival][featureIndex]) * value * value
        }
        if useHash {
            for i in 1..<FXDataEngineConstants.aiWeights {
                for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                    let hashedValue = Self.hashSign(i, j) * x[i] * x[j]
                    let h1 = Self.hashIndex(i, j)
                    let value1 = hashBandwidth1[h1] * hashedValue
                    normSquared += (hashSigma1[label][h1] + hashSigma1[rival][h1]) * value1 * value1
                    if useHash2 {
                        let h2 = Self.hashIndex2(i, j)
                        let value2 = hash2Scale * hashBandwidth2[h2] * hashedValue
                        normSquared += (hashSigma2[label][h2] + hashSigma2[rival][h2]) * value2 * value2
                    }
                }
            }
        }
        return max(normSquared, 1.0e-9)
    }

    private mutating func applyDecay(_ decay: Double) {
        guard decay > 0.0 else { return }
        let coefficient = max(1.0 - 0.01 * decay, 0.90)
        for classIndex in 0..<Self.classCount {
            for featureIndex in 1..<FXDataEngineConstants.aiWeights {
                linearWeights[classIndex][featureIndex] *= coefficient
                averagedLinearWeights[classIndex][featureIndex] *= coefficient
            }
            for bucket in 0..<Self.hashBuckets1 {
                hashWeights1[classIndex][bucket] *= coefficient
                averagedHashWeights1[classIndex][bucket] *= coefficient
            }
            for bucket in 0..<Self.hashBuckets2 {
                hashWeights2[classIndex][bucket] *= coefficient
                averagedHashWeights2[classIndex][bucket] *= coefficient
            }
        }
        for featureIndex in 1..<FXDataEngineConstants.aiWeights {
            moveMuLinear[featureIndex] *= coefficient
            moveLogVarLinear[featureIndex] *= coefficient
        }
        for bucket in 0..<Self.hashBuckets1 {
            moveMuHash1[bucket] *= coefficient
            moveLogVarHash1[bucket] *= coefficient
        }
        for bucket in 0..<Self.hashBuckets2 {
            moveMuHash2[bucket] *= coefficient
            moveLogVarHash2[bucket] *= coefficient
        }
    }

    private func computeScores(_ x: [Double], averaged: Bool) -> [Double] {
        (0..<Self.classCount).map { scoreClass(x, classIndex: $0, averaged: averaged) }
    }

    private func scoreClass(_ x: [Double], classIndex: Int, averaged: Bool) -> Double {
        let linear = averaged ? averagedLinearWeights[classIndex] : linearWeights[classIndex]
        var value = Self.dot(linear, x)
        guard useHash else { return value }
        let hash1 = averaged ? averagedHashWeights1[classIndex] : hashWeights1[classIndex]
        let hash2 = averaged ? averagedHashWeights2[classIndex] : hashWeights2[classIndex]
        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                let hashedValue = Self.hashSign(i, j) * x[i] * x[j]
                let h1 = Self.hashIndex(i, j)
                value += hash1[h1] * hashBandwidth1[h1] * hashedValue
                if useHash2 {
                    let h2 = Self.hashIndex2(i, j)
                    value += 0.70 * hash2[h2] * hash2Scale * hashBandwidth2[h2] * hashedValue
                }
            }
        }
        return value
    }

    private mutating func buildCollisionProfile() {
        hashOccupancy1 = Array(repeating: 0, count: Self.hashBuckets1)
        hashOccupancy2 = Array(repeating: 0, count: Self.hashBuckets2)
        hashBandwidth1 = Array(repeating: 1.0, count: Self.hashBuckets1)
        hashBandwidth2 = Array(repeating: 1.0, count: Self.hashBuckets2)
        for i in 1..<FXDataEngineConstants.aiWeights {
            for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                hashOccupancy1[Self.hashIndex(i, j)] += 1
                hashOccupancy2[Self.hashIndex2(i, j)] += 1
            }
        }
        for bucket in 0..<Self.hashBuckets1 {
            hashBandwidth1[bucket] = fxClamp(1.0 / sqrt(Double(max(1, hashOccupancy1[bucket]))), 0.20, 1.00)
        }
        for bucket in 0..<Self.hashBuckets2 {
            hashBandwidth2[bucket] = fxClamp(1.0 / sqrt(Double(max(1, hashOccupancy2[bucket]))), 0.20, 1.00)
        }
        hash2Scale = 1.0
        hashOccupancyReady = true
    }

    private mutating func updateCollisionRebalance() {
        guard hashOccupancyReady, steps % 128 == 0 else { return }
        let mean1 = hashOccupancy1.map(Double.init).reduce(0.0, +) / Double(Self.hashBuckets1)
        let mean2 = hashOccupancy2.map(Double.init).reduce(0.0, +) / Double(Self.hashBuckets2)
        let overload1 = hashOccupancy1.map { max(0.0, Double($0) - mean1) }.reduce(0.0, +)
        let overload2 = hashOccupancy2.map { max(0.0, Double($0) - mean2) }.reduce(0.0, +)
        var collisionRatio = mean1 > 1.0e-9 ? overload1 / (mean1 * Double(Self.hashBuckets1)) : 0.0
        if mean2 > 1.0e-9 {
            collisionRatio = 0.5 * collisionRatio + 0.5 * (overload2 / (mean2 * Double(Self.hashBuckets2)))
        }
        collisionRatio = fxClamp(collisionRatio, 0.0, 2.0)
        hash2Scale = fxClamp(0.70 + 0.45 * collisionRatio, 0.60, 1.40)

        for bucket in 0..<Self.hashBuckets1 {
            var target = fxClamp(1.0 / sqrt(Double(max(1, hashOccupancy1[bucket]))), 0.20, 1.00)
            let magnitude = (0..<Self.classCount).map { abs(hashWeights1[$0][bucket]) }.reduce(0.0, +)
            if magnitude > 9.0 {
                target *= 0.85
            }
            hashBandwidth1[bucket] = fxClamp(0.98 * hashBandwidth1[bucket] + 0.02 * target, 0.15, 1.10)
        }
        for bucket in 0..<Self.hashBuckets2 {
            var target = fxClamp(1.0 / sqrt(Double(max(1, hashOccupancy2[bucket]))), 0.20, 1.00)
            let magnitude = (0..<Self.classCount).map { abs(hashWeights2[$0][bucket]) }.reduce(0.0, +)
            if magnitude > 7.0 {
                target *= 0.85
            }
            hashBandwidth2[bucket] = fxClamp(0.98 * hashBandwidth2[bucket] + 0.02 * target, 0.15, 1.10)
        }
    }

    private func topRivalClasses(scores: [Double], label: Int) -> [Int] {
        (0..<Self.classCount)
            .filter { $0 != label }
            .sorted { scores[$0] > scores[$1] }
            .prefix(Self.topRivals)
            .map { $0 }
    }

    private func bestRival(scores: [Double], label: Int) -> Int? {
        (0..<Self.classCount)
            .filter { $0 != label }
            .max { scores[$0] < scores[$1] }
    }

    private func selectPAMode(loss: Double, excessPoints: Double) -> Int {
        if driftCooldown > 0 {
            return 1
        }
        if steps < 192 {
            return 0
        }
        if loss > 1.20, excessPoints > 0.0 {
            return 2
        }
        if lossReady, lossFast > 1.18 * max(0.05, lossSlow) {
            return 1
        }
        return 0
    }

    private func computeTau(mode: Int, loss: Double, cValue: Double, normSquared: Double) -> Double {
        guard loss > 0.0 else { return 0.0 }
        let safeNorm = max(normSquared, 1.0e-9)
        let tau: Double
        switch mode {
        case 1:
            tau = loss / (2.0 * safeNorm + 1.0 / (2.0 * max(cValue, 1.0e-9)))
        case 2:
            tau = min(cValue, loss / (2.0 * safeNorm))
        default:
            tau = min(cValue, loss / (2.0 * safeNorm))
        }
        return fxClamp(tau, 0.0, cValue)
    }

    private func predictMoveDistribution(_ x: [Double]) -> (mean: Double, logVariance: Double) {
        var mean = Self.dot(moveMuLinear, x)
        var logVariance = moveLogVarBias + Self.dot(moveLogVarLinear, x)
        if useHash {
            for i in 1..<FXDataEngineConstants.aiWeights {
                for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                    let hashedValue = Self.hashSign(i, j) * x[i] * x[j]
                    let h1 = Self.hashIndex(i, j)
                    let value1 = hashBandwidth1[h1] * hashedValue
                    mean += moveMuHash1[h1] * value1
                    logVariance += moveLogVarHash1[h1] * value1
                    if useHash2 {
                        let h2 = Self.hashIndex2(i, j)
                        let value2 = hash2Scale * hashBandwidth2[h2] * hashedValue
                        mean += moveMuHash2[h2] * value2
                        logVariance += moveLogVarHash2[h2] * value2
                    }
                }
            }
        }
        return (max(0.0, mean), fxClamp(logVariance, -4.0, 4.0))
    }

    private func predictMoveRaw(_ x: [Double]) -> Double {
        let move = predictMoveDistribution(x)
        return max(0.0, move.mean + 0.30 * sqrt(max(exp(move.logVariance), 1.0e-6)))
    }

    private func predictExpectedMovePoints(_ x: [Double]) -> Double {
        var head = predictMoveRaw(x)
        if moveSteps >= 24, moveReady, moveEMAAbs > 0.0 {
            head = 0.70 * head + 0.30 * moveEMAAbs
        }
        if head > 0.0 {
            return head
        }
        return moveReady ? moveEMAAbs : 0.0
    }

    private mutating func updateMoveHead(
        x: [Double],
        movePoints: Double,
        hyperParameters: HyperParameters,
        sampleWeight: Double
    ) {
        let target = abs(fxSafeFinite(movePoints, fallback: .nan))
        guard target.isFinite else { return }
        let prediction = predictMoveDistribution(x)
        let variance = max(exp(prediction.logVariance), 0.05)
        let error = prediction.mean - target
        let muGradient = PluginSupportTools.clipSymmetric(error, limit: 2.0) / max(variance, 0.25)
        let logVarianceGradient = PluginSupportTools.clipSymmetric(0.5 * (1.0 - (error * error) / max(variance, 0.25)), limit: 2.0)
        let weight = fxClamp(sampleWeight, 0.25, 6.00)
        let learningRate = fxClamp(0.02 * hyperParameters.learningRate * (0.85 + 0.15 * weight), 0.00005, 0.02000)
        let weightDecay = fxClamp(0.25 * hyperParameters.l2, 0.0, 0.0500)

        var muLinearGradient = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        var logVarLinearGradient = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        var muHashGradient1 = Array(repeating: 0.0, count: Self.hashBuckets1)
        var logVarHashGradient1 = Array(repeating: 0.0, count: Self.hashBuckets1)
        var muHashGradient2 = Array(repeating: 0.0, count: Self.hashBuckets2)
        var logVarHashGradient2 = Array(repeating: 0.0, count: Self.hashBuckets2)
        var normSquared = 0.0

        for featureIndex in 0..<FXDataEngineConstants.aiWeights {
            muLinearGradient[featureIndex] = PluginSupportTools.clipSymmetric(weight * muGradient * x[featureIndex], limit: 6.0)
            logVarLinearGradient[featureIndex] = PluginSupportTools.clipSymmetric(weight * logVarianceGradient * x[featureIndex], limit: 6.0)
            normSquared += muLinearGradient[featureIndex] * muLinearGradient[featureIndex]
            normSquared += logVarLinearGradient[featureIndex] * logVarLinearGradient[featureIndex]
        }
        if useHash {
            for i in 1..<FXDataEngineConstants.aiWeights {
                for j in (i + 1)..<FXDataEngineConstants.aiWeights {
                    let hashedValue = Self.hashSign(i, j) * x[i] * x[j]
                    let h1 = Self.hashIndex(i, j)
                    let value1 = hashBandwidth1[h1] * hashedValue
                    muHashGradient1[h1] += PluginSupportTools.clipSymmetric(weight * muGradient * value1, limit: 6.0)
                    logVarHashGradient1[h1] += PluginSupportTools.clipSymmetric(weight * logVarianceGradient * value1, limit: 6.0)
                    if useHash2 {
                        let h2 = Self.hashIndex2(i, j)
                        let value2 = hash2Scale * hashBandwidth2[h2] * hashedValue
                        muHashGradient2[h2] += PluginSupportTools.clipSymmetric(weight * muGradient * value2, limit: 6.0)
                        logVarHashGradient2[h2] += PluginSupportTools.clipSymmetric(weight * logVarianceGradient * value2, limit: 6.0)
                    }
                }
            }
        }
        for bucket in 0..<Self.hashBuckets1 {
            normSquared += muHashGradient1[bucket] * muHashGradient1[bucket]
            normSquared += logVarHashGradient1[bucket] * logVarHashGradient1[bucket]
        }
        for bucket in 0..<Self.hashBuckets2 {
            normSquared += muHashGradient2[bucket] * muHashGradient2[bucket]
            normSquared += logVarHashGradient2[bucket] * logVarHashGradient2[bucket]
        }
        let gradientScale = normSquared > 0.0 && sqrt(normSquared) > 6.0 ? 6.0 / sqrt(normSquared) : 1.0

        for featureIndex in 0..<FXDataEngineConstants.aiWeights {
            if featureIndex != 0 {
                moveMuLinear[featureIndex] *= 1.0 - learningRate * weightDecay
                moveLogVarLinear[featureIndex] *= 1.0 - learningRate * weightDecay
            }
            moveMuLinear[featureIndex] = PluginSupportTools.clipSymmetric(
                moveMuLinear[featureIndex] - learningRate * muLinearGradient[featureIndex] * gradientScale,
                limit: 20.0
            )
            moveLogVarLinear[featureIndex] = PluginSupportTools.clipSymmetric(
                moveLogVarLinear[featureIndex] - learningRate * logVarLinearGradient[featureIndex] * gradientScale,
                limit: 20.0
            )
        }
        moveLogVarBias = PluginSupportTools.clipSymmetric(moveLogVarBias - learningRate * (weight * logVarianceGradient) * gradientScale, limit: 4.0)
        for bucket in 0..<Self.hashBuckets1 {
            moveMuHash1[bucket] = PluginSupportTools.clipSymmetric(
                moveMuHash1[bucket] * (1.0 - learningRate * weightDecay) - learningRate * muHashGradient1[bucket] * gradientScale,
                limit: 15.0
            )
            moveLogVarHash1[bucket] = PluginSupportTools.clipSymmetric(
                moveLogVarHash1[bucket] * (1.0 - learningRate * weightDecay) - learningRate * logVarHashGradient1[bucket] * gradientScale,
                limit: 15.0
            )
        }
        for bucket in 0..<Self.hashBuckets2 {
            moveMuHash2[bucket] = PluginSupportTools.clipSymmetric(
                moveMuHash2[bucket] * (1.0 - learningRate * weightDecay) - learningRate * muHashGradient2[bucket] * gradientScale,
                limit: 12.0
            )
            moveLogVarHash2[bucket] = PluginSupportTools.clipSymmetric(
                moveLogVarHash2[bucket] * (1.0 - learningRate * weightDecay) - learningRate * logVarHashGradient2[bucket] * gradientScale,
                limit: 12.0
            )
        }
        updateMoveEMA(movePoints: movePoints, alpha: 0.05)
        moveSteps += 1
    }

    private func calibrated(_ rawProbabilities: [Double]) -> [Double] {
        let logits = buildCalibrationLogits(rawProbabilities)
        var probabilities = Self.softmax3(logits)
        guard calibrationSteps >= 30 else { return probabilities }
        var isotonic = probabilities
        for classIndex in 0..<Self.classCount {
            let total = isotonicCount[classIndex].reduce(0.0, +)
            guard total >= 30.0 else { continue }
            var previous = 0.01
            var monotonic = Array(repeating: 0.0, count: Self.calibrationBins)
            for bin in 0..<Self.calibrationBins {
                var ratio = previous
                if isotonicCount[classIndex][bin] > 1.0e-9 {
                    ratio = isotonicPositive[classIndex][bin] / isotonicCount[classIndex][bin]
                }
                ratio = fxClamp(ratio, 0.001, 0.999)
                if ratio < previous {
                    ratio = previous
                }
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

    private mutating func updateCalibrator(rawProbabilities: [Double], label: Int, sampleWeight: Double, learningRate: Double) {
        let calibratedLogits = buildCalibrationLogits(rawProbabilities)
        let probabilities = Self.softmax3(calibratedLogits)
        let logRaw = rawProbabilities.map { log(fxClamp($0, 0.0005, 0.9990)) }
        let weight = fxClamp(sampleWeight, 0.25, 6.00)
        let calibrationLearningRate = fxClamp(0.20 * learningRate * weight, 0.0002, 0.0200)
        let l2 = 0.0005

        for classIndex in 0..<Self.classCount {
            for bin in 0..<Self.calibrationBins {
                isotonicCount[classIndex][bin] *= 0.9995
                isotonicPositive[classIndex][bin] *= 0.9995
            }
        }

        for classIndex in 0..<Self.classCount {
            let target = classIndex == label ? 1.0 : 0.0
            let error = target - probabilities[classIndex]
            calibrationBias[classIndex] = PluginSupportTools.clipSymmetric(
                calibrationBias[classIndex] + calibrationLearningRate * error,
                limit: 4.0
            )
            for rawIndex in 0..<Self.classCount {
                let identity = classIndex == rawIndex ? 1.0 : 0.0
                let gradient = error * logRaw[rawIndex] - l2 * (calibrationWeights[classIndex][rawIndex] - identity)
                calibrationWeights[classIndex][rawIndex] = PluginSupportTools.clipSymmetric(
                    calibrationWeights[classIndex][rawIndex] + calibrationLearningRate * gradient,
                    limit: 5.0
                )
            }
            let bin = Self.isotonicBin(probabilities[classIndex])
            isotonicCount[classIndex][bin] += weight
            isotonicPositive[classIndex][bin] += weight * target
        }
        calibrationSteps += 1
    }

    private func buildCalibrationLogits(_ rawProbabilities: [Double]) -> [Double] {
        let logRaw = rawProbabilities.map { log(fxClamp($0, 0.0005, 0.9990)) }
        var logits = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            var value = calibrationBias[classIndex]
            for rawIndex in 0..<Self.classCount {
                value += calibrationWeights[classIndex][rawIndex] * logRaw[rawIndex]
            }
            logits[classIndex] = value
        }
        return logits
    }

    private mutating func updateABGuard(liveCrossEntropy: Double, averagedCrossEntropy: Double) {
        if !guardReady {
            guardLiveFast = liveCrossEntropy
            guardLiveSlow = liveCrossEntropy
            guardAverageFast = averagedCrossEntropy
            guardAverageSlow = averagedCrossEntropy
            guardUseAverage = false
            guardReady = true
            return
        }
        guardLiveFast = 0.92 * guardLiveFast + 0.08 * liveCrossEntropy
        guardLiveSlow = 0.995 * guardLiveSlow + 0.005 * liveCrossEntropy
        guardAverageFast = 0.92 * guardAverageFast + 0.08 * averagedCrossEntropy
        guardAverageSlow = 0.995 * guardAverageSlow + 0.005 * averagedCrossEntropy
        guard steps > 64 else { return }
        if guardAverageFast < 0.985 * guardLiveFast, guardAverageSlow <= 1.03 * guardLiveSlow {
            guardUseAverage = true
        } else if guardLiveFast < 0.97 * guardAverageFast {
            guardUseAverage = false
        }
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
        if driftCooldown > 0 {
            driftCooldown -= 1
        }
        guard steps >= 256, driftCooldown == 0 else { return }
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

    private mutating func pushReplay(label: Int, x: [Double], movePoints: Double, sampleWeight: Double, hardness: Double) {
        let sample = ReplaySample(
            x: x,
            label: label,
            movePoints: movePoints,
            sampleWeight: sampleWeight,
            hardness: fxClamp(hardness, 0.0, 20.0)
        )
        if replay.count < Self.replayCapacity {
            replay.append(sample)
        } else {
            replay[replayHead] = sample
        }
        replayHead = (replayHead + 1) % Self.replayCapacity
    }

    private mutating func pickHardReplay() -> ReplaySample? {
        guard !replay.isEmpty else { return nil }
        var bestIndex = rng.nextIndex(replay.count)
        var bestHardness = replay[bestIndex].hardness
        for _ in 1..<6 {
            let index = rng.nextIndex(replay.count)
            if replay[index].hardness > bestHardness {
                bestIndex = index
                bestHardness = replay[index].hardness
            }
        }
        return replay[bestIndex]
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

    private static func sessionBucket(sampleTimeUTC: Int64, fallback: Int) -> Int {
        if sampleTimeUTC > 0 {
            let seconds = Int(((sampleTimeUTC % 86_400) + 86_400) % 86_400)
            let hour = seconds / 3_600
            if (6...12).contains(hour) { return 1 }
            if (13...20).contains(hour) { return 2 }
            if hour >= 21 || hour <= 2 { return 0 }
            return 3
        }
        return min(max(fallback, 0), 3)
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
        let safe = (0..<Self.classCount).map { index -> Double in
            index < logits.count ? fxSafeFinite(logits[index]) : 0.0
        }
        let maximum = safe.max() ?? 0.0
        var exponentials = Array(repeating: 0.0, count: Self.classCount)
        var sum = 0.0
        for index in 0..<Self.classCount {
            let value = exp(fxClamp(safe[index] - maximum, -30.0, 30.0))
            exponentials[index] = value
            sum += value
        }
        guard sum > 0.0 else { return [0.3333333, 0.3333333, 0.3333333] }
        return exponentials.map { $0 / sum }
    }

    private static func isotonicBin(_ probability: Double) -> Int {
        min(max(Int(floor(fxClamp(probability, 0.0, 0.999999) * Double(Self.calibrationBins))), 0), Self.calibrationBins - 1)
    }

    private static func identityCalibrationWeights() -> [[Double]] {
        (0..<Self.classCount).map { row in
            (0..<Self.classCount).map { column in row == column ? 1.0 : 0.0 }
        }
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

private extension Array where Element == Double {
    subscript(safe index: Int) -> Double {
        indices.contains(index) ? self[index] : 0.0
    }
}
