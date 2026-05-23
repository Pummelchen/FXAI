import FXDataEngine
import Foundation

public struct DistQuantileCPUModel: Sendable {
    private static let quantileCount = 9
    private static let middleIndex = 4
    private static let sideCount = 4
    private static let sessionCount = 4
    private static let regimeCount = 3
    private static let classFeatureCount = 8
    private static let calibrationFeatureCount = 8
    private static let classCount = 3

    private var tau: [Double]
    private var medianShort: [Double]
    private var medianMedium: [Double]
    private var gapUpShort: [[Double]]
    private var gapDownShort: [[Double]]
    private var gapUpMedium: [[Double]]
    private var gapDownMedium: [[Double]]
    private var g2MedianShort: [Double]
    private var g2MedianMedium: [Double]
    private var g2GapUpShort: [[Double]]
    private var g2GapDownShort: [[Double]]
    private var g2GapUpMedium: [[Double]]
    private var g2GapDownMedium: [[Double]]
    private var sessionBias: [[Double]]
    private var g2SessionBias: [[Double]]
    private var regimeScale: [[Double]]
    private var g2RegimeScale: [[Double]]
    private var classWeights: [[Double]]
    private var classG2: [[Double]]
    private var calibrationWeights: [Double]
    private var calibrationG2: [Double]
    private var diagnosticsCount: Int
    private var pitMean: Double
    private var pitM2: Double
    private var crossingEMA: Double
    private var calibrationErrorEMA: Double
    private var reliabilityWeight: Double
    private var mediumReady: Bool
    private var mediumTargetEMA: Double
    private var moveReady: Bool
    private var moveEMAAbs: Double
    private var moveHead: PluginMoveHead
    private var qualityBank: PluginQualityBank

    public init() {
        let featureCount = FXDataEngineConstants.aiWeights
        self.tau = [0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 0.95]
        self.medianShort = Array(repeating: 0.0, count: featureCount)
        self.medianMedium = Array(repeating: 0.0, count: featureCount)
        self.gapUpShort = Self.zeroSideFeatureMatrix()
        self.gapDownShort = Self.zeroSideFeatureMatrix()
        self.gapUpMedium = Self.zeroSideFeatureMatrix()
        self.gapDownMedium = Self.zeroSideFeatureMatrix()
        self.g2MedianShort = Array(repeating: 0.0, count: featureCount)
        self.g2MedianMedium = Array(repeating: 0.0, count: featureCount)
        self.g2GapUpShort = Self.zeroSideFeatureMatrix()
        self.g2GapDownShort = Self.zeroSideFeatureMatrix()
        self.g2GapUpMedium = Self.zeroSideFeatureMatrix()
        self.g2GapDownMedium = Self.zeroSideFeatureMatrix()
        self.sessionBias = Array(
            repeating: Array(repeating: 0.0, count: Self.sessionCount),
            count: 2
        )
        self.g2SessionBias = Array(
            repeating: Array(repeating: 0.0, count: Self.sessionCount),
            count: 2
        )
        self.regimeScale = Array(
            repeating: Array(repeating: 1.0, count: Self.regimeCount),
            count: 2
        )
        self.g2RegimeScale = Array(
            repeating: Array(repeating: 0.0, count: Self.regimeCount),
            count: 2
        )
        self.classWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.classFeatureCount),
            count: Self.classCount
        )
        self.classG2 = Array(
            repeating: Array(repeating: 0.0, count: Self.classFeatureCount),
            count: Self.classCount
        )
        self.classWeights[LabelClass.skip.rawValue][0] = 0.20
        self.calibrationWeights = Array(repeating: 0.0, count: Self.calibrationFeatureCount)
        self.calibrationG2 = Array(repeating: 0.0, count: Self.calibrationFeatureCount)
        self.diagnosticsCount = 0
        self.pitMean = 0.50
        self.pitM2 = 0.0
        self.crossingEMA = 0.0
        self.calibrationErrorEMA = 0.0
        self.reliabilityWeight = 1.0
        self.mediumReady = false
        self.mediumTargetEMA = 0.0
        self.moveReady = false
        self.moveEMAAbs = 0.0
        self.moveHead = PluginMoveHead()
        self.qualityBank = PluginQualityBank()
    }

    public mutating func reset() {
        self = DistQuantileCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let cost = PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: request.context.priceCostPoints)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: cost
        )
        updateMoveEMA(movePoints: request.movePoints)
        updateMediumTarget(movePoints: request.movePoints)

        let session = Self.sessionBucket(context: request.context)
        let regime = Self.regimeBucket(x: x, context: request.context, cost: cost)
        var learningRate = fxClamp(hyperParameters.quantileLearningRate, 0.00005, 0.05000)
        let weightDecay = fxClamp(hyperParameters.quantileL2, 0.00000, 0.05000)
        if reliabilityWeight < 0.75 {
            learningRate *= 0.85
        }

        let qualityTargets = PluginQualityTargets(request: request)
        let sampleWeight = fxClamp(
            request.sampleWeight *
                PluginSupportTools.moveSampleWeight(
                    x: x,
                    movePoints: request.movePoints,
                    priceCostPoints: cost,
                    minMovePoints: request.context.minMovePoints,
                    qualityTargets: qualityTargets
                ) * reliabilityWeight,
            0.10,
            6.00
        )
        qualityBank.update(targets: qualityTargets, context: request.context, sampleWeight: sampleWeight)

        trainQuantileHead(
            head: 0,
            x: x,
            session: session,
            regime: regime,
            target: request.movePoints,
            sampleWeight: sampleWeight,
            learningRate: learningRate,
            weightDecay: weightDecay
        )
        trainQuantileHead(
            head: 1,
            x: x,
            session: session,
            regime: regime,
            target: mediumTargetEMA,
            sampleWeight: 0.60 * sampleWeight,
            learningRate: 0.80 * learningRate,
            weightDecay: 0.80 * weightDecay
        )

        let quantiles = blendedQuantiles(x: x, session: session, regime: regime)
        let classFeatures = buildClassFeatures(
            short: quantiles.short,
            medium: quantiles.medium,
            blended: quantiles.blended,
            cost: cost,
            x: x,
            dataHasVolume: request.context.dataHasVolume
        )
        let rawClassProbabilities = updateClassHead(
            label: label.rawValue,
            features: classFeatures,
            sampleWeight: sampleWeight,
            learningRate: 0.70 * learningRate,
            weightDecay: 0.50 * weightDecay
        )

        let calibrationFeatures = buildCalibrationFeatures(
            blended: quantiles.blended,
            classProbabilities: rawClassProbabilities,
            cost: cost,
            x: x,
            dataHasVolume: request.context.dataHasVolume
        )
        let directionTarget = label == .buy ? 1 : (label == .sell ? 0 : (request.movePoints >= 0.0 ? 1 : 0))
        let directionWeight = label == .skip ? 0.35 * sampleWeight : sampleWeight
        updateDirectionCalibrator(
            target: directionTarget,
            features: calibrationFeatures,
            sampleWeight: directionWeight,
            learningRate: 0.50 * learningRate,
            weightDecay: 0.30 * weightDecay
        )

        updateDiagnostics(
            movePoints: request.movePoints,
            blended: quantiles.blended,
            label: label.rawValue,
            classProbabilities: rawClassProbabilities
        )
        moveHead.update(x: x, movePoints: request.movePoints, hyperParameters: hyperParameters, sampleWeight: sampleWeight)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let cost = PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: request.context.priceCostPoints)
        let session = Self.sessionBucket(context: request.context)
        let regime = Self.regimeBucket(x: x, context: request.context, cost: cost)
        let quantiles = blendedQuantiles(x: x, session: session, regime: regime)
        let classFeatures = buildClassFeatures(
            short: quantiles.short,
            medium: quantiles.medium,
            blended: quantiles.blended,
            cost: cost,
            x: x,
            dataHasVolume: request.context.dataHasVolume
        )
        let rawClassProbabilities = predictClassProbabilities(classFeatures)
        let classProbabilities = adjustedClassProbabilities(
            blended: quantiles.blended,
            rawClassProbabilities: rawClassProbabilities,
            cost: cost,
            x: x,
            dataHasVolume: request.context.dataHasVolume
        )

        let buyTail = 0.50 * max(0.0, quantiles.blended[Self.quantileCount - 1] - cost) +
            0.50 * max(0.0, quantiles.blended[Self.quantileCount - 2] - cost)
        let sellTail = 0.50 * max(0.0, -quantiles.blended[0] - cost) +
            0.50 * max(0.0, -quantiles.blended[1] - cost)
        let expectedValue = classProbabilities[LabelClass.buy.rawValue] * buyTail +
            classProbabilities[LabelClass.sell.rawValue] * sellTail
        let amplitude = 0.50 * (abs(quantiles.blended[Self.quantileCount - 1]) + abs(quantiles.blended[0]))
        var moveMean = max(expectedValue, 0.35 * amplitude) * fxClamp(reliabilityWeight, 0.50, 1.50)
        if moveMean > 0.0, moveReady, moveEMAAbs > 0.0 {
            let learnedMove = moveHead.ready ? moveHead.predictRaw(x) : 0.0
            moveMean = 0.55 * moveMean + 0.30 * moveEMAAbs + 0.15 * max(0.0, learnedMove)
        } else if moveMean <= 0.0 {
            moveMean = moveReady ? moveEMAAbs : 0.0
        }

        let q25 = max(0.0, abs(quantiles.blended[1]))
        let q50 = max(q25, abs(quantiles.blended[Self.middleIndex]))
        let q75 = max(q50, abs(quantiles.blended[Self.quantileCount - 2]))
        let confidence = fxClamp(
            max(classProbabilities[LabelClass.buy.rawValue], classProbabilities[LabelClass.sell.rawValue]),
            0.0,
            1.0
        )
        let baseOutput = PluginModelOutputV4(
            classProbabilities: classProbabilities,
            moveMeanPoints: max(0.0, moveMean),
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: max(q75, moveMean),
            maeMeanPoints: max(0.0, 0.35 * moveMean),
            hitTimeFraction: fxClamp(0.65 - 0.22 * confidence + 0.18 * classProbabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            pathRisk: fxClamp(0.25 + 0.45 * classProbabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            fillRisk: fxClamp(cost / max(moveMean + request.context.minMovePoints, 0.25), 0.0, 1.0),
            confidence: confidence,
            reliability: fxClamp(reliabilityWeight / 1.50, 0.0, 1.0),
            hasQuantiles: true,
            hasConfidence: true,
            hasPathQuality: true
        )
        let output = PluginPathQualityTools.populatedOutput(
            baseOutput,
            x: x,
            window: window,
            context: request.context,
            family: .distributional,
            activityGate: 1.0 - classProbabilities[LabelClass.skip.rawValue],
            structuralQuality: fxClamp(reliabilityWeight / 1.50, 0.0, 1.0),
            qualityPriors: qualityBank.priors(context: request.context),
            declaredWindowSize: request.windowSize
        )
        return PluginContextRuntimeTools.fillPrediction(
            modelOutput: output,
            calibratedMoveMeanPoints: max(0.0, moveMean),
            context: request.context
        )
    }

    private func blendedQuantiles(
        x: [Double],
        session: Int,
        regime: Int
    ) -> (short: [Double], medium: [Double], blended: [Double]) {
        let short = buildHead(head: 0, x: x, session: session, regime: regime)
        let medium = buildHead(head: 1, x: x, session: session, regime: regime)
        var weightShort = 0.55
        if regime == 0 {
            weightShort = 0.45
        } else if regime == 2 {
            weightShort = 0.70
        }
        var blended = Array(repeating: 0.0, count: Self.quantileCount)
        for index in 0..<Self.quantileCount {
            blended[index] = weightShort * short[index] + (1.0 - weightShort) * medium[index]
        }
        enforceMonotonicity(&blended)
        return (short, medium, blended)
    }

    private func buildHead(head: Int, x: [Double], session: Int, regime: Int) -> [Double] {
        var quantiles = Array(repeating: 0.0, count: Self.quantileCount)
        let headIndex = head == 0 ? 0 : 1
        let medianWeights = headIndex == 0 ? medianShort : medianMedium
        let upWeights = headIndex == 0 ? gapUpShort : gapUpMedium
        let downWeights = headIndex == 0 ? gapDownShort : gapDownMedium
        let median = Self.dot(medianWeights, x) + sessionBias[headIndex][session]
        let scale = fxClamp(regimeScale[headIndex][regime], 0.40, 2.80)
        quantiles[Self.middleIndex] = median

        for side in 0..<Self.sideCount {
            let upGap = scale * Self.softplus(Self.dot(upWeights[side], x))
            let downGap = scale * Self.softplus(Self.dot(downWeights[side], x))
            quantiles[Self.middleIndex + 1 + side] = quantiles[Self.middleIndex + side] + upGap
            quantiles[Self.middleIndex - 1 - side] = quantiles[Self.middleIndex - side] - downGap
        }
        enforceMonotonicity(&quantiles)
        return quantiles
    }

    private mutating func trainQuantileHead(
        head: Int,
        x: [Double],
        session: Int,
        regime: Int,
        target: Double,
        sampleWeight: Double,
        learningRate: Double,
        weightDecay: Double
    ) {
        let headIndex = head == 0 ? 0 : 1
        let quantiles = buildHead(head: headIndex, x: x, session: session, regime: regime)
        let upWeights = headIndex == 0 ? gapUpShort : gapUpMedium
        let downWeights = headIndex == 0 ? gapDownShort : gapDownMedium
        let scale = fxClamp(regimeScale[headIndex][regime], 0.40, 2.80)

        var zUp = Array(repeating: 0.0, count: Self.sideCount)
        var zDown = Array(repeating: 0.0, count: Self.sideCount)
        var softplusUp = Array(repeating: 0.0, count: Self.sideCount)
        var softplusDown = Array(repeating: 0.0, count: Self.sideCount)
        var sigmoidUp = Array(repeating: 0.0, count: Self.sideCount)
        var sigmoidDown = Array(repeating: 0.0, count: Self.sideCount)
        for side in 0..<Self.sideCount {
            zUp[side] = Self.dot(upWeights[side], x)
            zDown[side] = Self.dot(downWeights[side], x)
            softplusUp[side] = Self.softplus(zUp[side])
            softplusDown[side] = Self.softplus(zDown[side])
            sigmoidUp[side] = PluginSupportTools.sigmoid(zUp[side])
            sigmoidDown[side] = PluginSupportTools.sigmoid(zDown[side])
        }

        var quantileGradients = Array(repeating: 0.0, count: Self.quantileCount)
        for index in 0..<Self.quantileCount {
            quantileGradients[index] = PluginSupportTools.clipSymmetric(
                sampleWeight * Self.quantileGradientSignal(
                    target: target,
                    prediction: quantiles[index],
                    tau: tau[index],
                    huberDelta: 5.0
                ),
                limit: 2.0
            )
        }

        var medianGradient = quantileGradients.reduce(0.0, +)
        var upGradients = Array(repeating: 0.0, count: Self.sideCount)
        var downGradients = Array(repeating: 0.0, count: Self.sideCount)
        var scaleGradient = 0.0
        for side in 0..<Self.sideCount {
            var sumUp = 0.0
            for index in (Self.middleIndex + 1 + side)..<Self.quantileCount {
                sumUp += quantileGradients[index]
            }

            var sumDown = 0.0
            let downUpperBound = Self.middleIndex - 1 - side
            if downUpperBound >= 0 {
                for index in 0...downUpperBound {
                    sumDown += quantileGradients[index]
                }
            }

            upGradients[side] = scale * sigmoidUp[side] * sumUp
            downGradients[side] = -scale * sigmoidDown[side] * sumDown
            scaleGradient += softplusUp[side] * sumUp - softplusDown[side] * sumDown
        }

        let xNorm2 = max(Self.dot(x, x), 1.0)
        var norm2 = medianGradient * medianGradient * xNorm2 + scaleGradient * scaleGradient
        for side in 0..<Self.sideCount {
            norm2 += upGradients[side] * upGradients[side] * xNorm2
            norm2 += downGradients[side] * downGradients[side] * xNorm2
        }
        let gradientNorm = sqrt(norm2)
        let gradientScale = gradientNorm > 8.0 && gradientNorm > 1.0e-9 ? 8.0 / gradientNorm : 1.0
        medianGradient *= gradientScale
        scaleGradient *= gradientScale
        for side in 0..<Self.sideCount {
            upGradients[side] *= gradientScale
            downGradients[side] *= gradientScale
        }

        applyMedianUpdate(head: headIndex, x: x, gradientScalar: medianGradient, learningRate: learningRate, weightDecay: weightDecay)
        for side in 0..<Self.sideCount {
            applyGapUpdate(
                head: headIndex,
                side: side,
                x: x,
                gradientScalar: upGradients[side],
                learningRate: learningRate,
                weightDecay: weightDecay,
                direction: .up
            )
            applyGapUpdate(
                head: headIndex,
                side: side,
                x: x,
                gradientScalar: downGradients[side],
                learningRate: learningRate,
                weightDecay: weightDecay,
                direction: .down
            )
        }
        Self.applyScalarUpdate(
            weight: &sessionBias[headIndex][session],
            g2: &g2SessionBias[headIndex][session],
            gradient: medianGradient,
            learningRate: 0.60 * learningRate,
            weightDecay: 0.10 * weightDecay,
            lowerBound: -8.0,
            upperBound: 8.0
        )
        Self.applyScalarUpdate(
            weight: &regimeScale[headIndex][regime],
            g2: &g2RegimeScale[headIndex][regime],
            gradient: scaleGradient,
            learningRate: 0.40 * learningRate,
            weightDecay: 0.0,
            lowerBound: 0.30,
            upperBound: 3.20
        )
    }

    private mutating func applyMedianUpdate(
        head: Int,
        x: [Double],
        gradientScalar: Double,
        learningRate: Double,
        weightDecay: Double
    ) {
        for index in 0..<FXDataEngineConstants.aiWeights {
            let gradient = gradientScalar * x[index]
            if head == 0 {
                Self.applyVectorUpdate(
                    weight: &medianShort[index],
                    g2: &g2MedianShort[index],
                    gradient: gradient,
                    learningRate: learningRate,
                    weightDecay: index > 0 ? weightDecay : 0.0
                )
            } else {
                Self.applyVectorUpdate(
                    weight: &medianMedium[index],
                    g2: &g2MedianMedium[index],
                    gradient: gradient,
                    learningRate: learningRate,
                    weightDecay: index > 0 ? weightDecay : 0.0
                )
            }
        }
    }

    private enum GapDirection {
        case up
        case down
    }

    private mutating func applyGapUpdate(
        head: Int,
        side: Int,
        x: [Double],
        gradientScalar: Double,
        learningRate: Double,
        weightDecay: Double,
        direction: GapDirection
    ) {
        for index in 0..<FXDataEngineConstants.aiWeights {
            let gradient = gradientScalar * x[index]
            let decay = index > 0 ? weightDecay : 0.0
            switch (head, direction) {
            case (0, .up):
                Self.applyVectorUpdate(
                    weight: &gapUpShort[side][index],
                    g2: &g2GapUpShort[side][index],
                    gradient: gradient,
                    learningRate: learningRate,
                    weightDecay: decay
                )
            case (0, .down):
                Self.applyVectorUpdate(
                    weight: &gapDownShort[side][index],
                    g2: &g2GapDownShort[side][index],
                    gradient: gradient,
                    learningRate: learningRate,
                    weightDecay: decay
                )
            case (_, .up):
                Self.applyVectorUpdate(
                    weight: &gapUpMedium[side][index],
                    g2: &g2GapUpMedium[side][index],
                    gradient: gradient,
                    learningRate: learningRate,
                    weightDecay: decay
                )
            case (_, .down):
                Self.applyVectorUpdate(
                    weight: &gapDownMedium[side][index],
                    g2: &g2GapDownMedium[side][index],
                    gradient: gradient,
                    learningRate: learningRate,
                    weightDecay: decay
                )
            }
        }
    }

    private mutating func updateClassHead(
        label: Int,
        features: [Double],
        sampleWeight: Double,
        learningRate: Double,
        weightDecay: Double
    ) -> [Double] {
        let probabilities = predictClassProbabilities(features)
        let featureNorm2 = max(Self.dot(features, features), 1.0)
        var classGradient = Array(repeating: 0.0, count: Self.classCount)
        var norm2 = 0.0
        for classIndex in 0..<Self.classCount {
            let target = classIndex == label ? 1.0 : 0.0
            classGradient[classIndex] = sampleWeight * (target - probabilities[classIndex])
            norm2 += classGradient[classIndex] * classGradient[classIndex] * featureNorm2
        }
        let gradientNorm = sqrt(norm2)
        let gradientScale = gradientNorm > 6.0 && gradientNorm > 1.0e-9 ? 6.0 / gradientNorm : 1.0
        for classIndex in 0..<Self.classCount {
            let scaledClassGradient = classGradient[classIndex] * gradientScale
            for featureIndex in 0..<Self.classFeatureCount {
                let gradient = scaledClassGradient * features[featureIndex]
                classG2[classIndex][featureIndex] += gradient * gradient
                let step = learningRate / sqrt(classG2[classIndex][featureIndex] + 1.0e-8)
                if featureIndex > 0, weightDecay > 0.0 {
                    classWeights[classIndex][featureIndex] *= 1.0 - step * weightDecay
                }
                classWeights[classIndex][featureIndex] = PluginSupportTools.clipSymmetric(
                    classWeights[classIndex][featureIndex] + step * gradient,
                    limit: 10.0
                )
            }
        }
        return probabilities
    }

    private func predictClassProbabilities(_ features: [Double]) -> [Double] {
        let logits = (0..<Self.classCount).map { classIndex in
            Self.dot(classWeights[classIndex], features)
        }
        return Self.softmax3(logits)
    }

    private func adjustedClassProbabilities(
        blended: [Double],
        rawClassProbabilities: [Double],
        cost: Double,
        x: [Double],
        dataHasVolume: Bool
    ) -> [Double] {
        let quantileWidth = max(blended[Self.quantileCount - 1] - blended[0], 0.05)
        let buyEV = 0.50 * (blended[Self.middleIndex] + blended[Self.quantileCount - 1]) - cost
        let sellEV = 0.50 * (-blended[Self.middleIndex] - blended[0]) - cost
        let denominator = max(rawClassProbabilities[LabelClass.buy.rawValue] + rawClassProbabilities[LabelClass.sell.rawValue], 1.0e-9)
        let directionFromClass = rawClassProbabilities[LabelClass.buy.rawValue] / denominator
        let directionFromEV = PluginSupportTools.sigmoid(PluginSupportTools.clipSymmetric((buyEV - sellEV) / (quantileWidth + 0.10), limit: 12.0))
        let calibrationFeatures = buildCalibrationFeatures(
            blended: blended,
            classProbabilities: rawClassProbabilities,
            cost: cost,
            x: x,
            dataHasVolume: dataHasVolume
        )
        let directionFromCalibration = predictDirectionCalibration(calibrationFeatures)
        let directionProbability = fxClamp(
            0.45 * directionFromClass + 0.35 * directionFromEV + 0.20 * directionFromCalibration,
            0.001,
            0.999
        )

        var skipProbability = fxClamp(rawClassProbabilities[LabelClass.skip.rawValue], 0.0, 0.98)
        if buyEV <= 0.0, sellEV <= 0.0 {
            skipProbability = max(skipProbability, 0.70)
        }
        let active = fxClamp(1.0 - skipProbability, 0.0, 1.0)
        return PluginContextRuntimeTools.normalizeClassDistribution([
            (1.0 - directionProbability) * active,
            directionProbability * active,
            1.0 - active
        ])
    }

    private func buildClassFeatures(
        short: [Double],
        medium: [Double],
        blended: [Double],
        cost: Double,
        x: [Double],
        dataHasVolume: Bool
    ) -> [Double] {
        let shortWidth = max(short[Self.quantileCount - 1] - short[0], 0.05)
        let mediumWidth = max(medium[Self.quantileCount - 1] - medium[0], 0.05)
        let blendedWidth = max(blended[Self.quantileCount - 1] - blended[0], 0.05)
        let expectedBuy = 0.50 * (blended[Self.middleIndex] + blended[Self.quantileCount - 1]) - cost
        let expectedSell = 0.50 * (-blended[Self.middleIndex] - blended[0]) - cost
        return [
            1.0,
            PluginSupportTools.clipSymmetric(short[Self.middleIndex] / shortWidth, limit: 8.0),
            PluginSupportTools.clipSymmetric(shortWidth, limit: 20.0),
            PluginSupportTools.clipSymmetric(medium[Self.middleIndex] / mediumWidth, limit: 8.0),
            PluginSupportTools.clipSymmetric(mediumWidth, limit: 20.0),
            PluginSupportTools.clipSymmetric((expectedBuy - expectedSell) / blendedWidth, limit: 8.0),
            PluginSupportTools.clipSymmetric(cost, limit: 12.0),
            PluginSupportTools.clipSymmetric(Self.volatilityAndVolumeSignal(x, dataHasVolume: dataHasVolume), limit: 8.0)
        ]
    }

    private func buildCalibrationFeatures(
        blended: [Double],
        classProbabilities: [Double],
        cost: Double,
        x: [Double],
        dataHasVolume: Bool
    ) -> [Double] {
        let quantileWidth = max(blended[Self.quantileCount - 1] - blended[0], 0.05)
        let expectedBuy = 0.50 * (blended[Self.middleIndex] + blended[Self.quantileCount - 1]) - cost
        let expectedSell = 0.50 * (-blended[Self.middleIndex] - blended[0]) - cost
        return [
            1.0,
            PluginSupportTools.clipSymmetric(blended[Self.middleIndex] / quantileWidth, limit: 8.0),
            PluginSupportTools.clipSymmetric((expectedBuy - expectedSell) / quantileWidth, limit: 8.0),
            PluginSupportTools.clipSymmetric(classProbabilities[LabelClass.buy.rawValue] - classProbabilities[LabelClass.sell.rawValue], limit: 1.0),
            PluginSupportTools.clipSymmetric(classProbabilities[LabelClass.skip.rawValue], limit: 1.0),
            PluginSupportTools.clipSymmetric(cost, limit: 12.0),
            PluginSupportTools.clipSymmetric(Self.volatilityAndVolumeSignal(x, dataHasVolume: dataHasVolume), limit: 8.0),
            PluginSupportTools.clipSymmetric(1.0 / (quantileWidth + 0.10), limit: 8.0)
        ]
    }

    private func predictDirectionCalibration(_ features: [Double]) -> Double {
        PluginSupportTools.sigmoid(Self.dot(calibrationWeights, features))
    }

    private mutating func updateDirectionCalibrator(
        target: Int,
        features: [Double],
        sampleWeight: Double,
        learningRate: Double,
        weightDecay: Double
    ) {
        let prediction = predictDirectionCalibration(features)
        let error = sampleWeight * (Double(target) - prediction)
        let featureNorm2 = max(Self.dot(features, features), 1.0)
        let gradientNorm = sqrt(error * error * featureNorm2)
        let gradientScale = gradientNorm > 4.0 && gradientNorm > 1.0e-9 ? 4.0 / gradientNorm : 1.0
        let scaledError = error * gradientScale
        for index in 0..<Self.calibrationFeatureCount {
            let gradient = scaledError * features[index]
            calibrationG2[index] += gradient * gradient
            let step = learningRate / sqrt(calibrationG2[index] + 1.0e-8)
            if index > 0, weightDecay > 0.0 {
                calibrationWeights[index] *= 1.0 - step * weightDecay
            }
            calibrationWeights[index] = PluginSupportTools.clipSymmetric(
                calibrationWeights[index] + step * gradient,
                limit: 10.0
            )
        }
    }

    private mutating func updateDiagnostics(
        movePoints: Double,
        blended: [Double],
        label: Int,
        classProbabilities: [Double]
    ) {
        let pit = approxPIT(target: movePoints, quantiles: blended)
        diagnosticsCount += 1
        let count = Double(diagnosticsCount)
        let delta = pit - pitMean
        pitMean += delta / count
        pitM2 += delta * (pit - pitMean)

        var crossing = 0.0
        for index in 1..<Self.quantileCount where blended[index] < blended[index - 1] {
            crossing = 1.0
            break
        }
        crossingEMA = 0.98 * crossingEMA + 0.02 * crossing
        let resolvedLabel = (0..<Self.classCount).contains(label) ? label : LabelClass.skip.rawValue
        let trueProbability = fxClamp(classProbabilities[resolvedLabel], 0.0, 1.0)
        calibrationErrorEMA = 0.98 * calibrationErrorEMA + 0.02 * (1.0 - trueProbability)

        if diagnosticsCount >= 25 {
            let variance = diagnosticsCount > 1 ? pitM2 / Double(diagnosticsCount - 1) : 1.0 / 12.0
            let pitPenalty = abs(pitMean - 0.5) + 6.0 * abs(variance - 1.0 / 12.0)
            let drift = pitPenalty + 0.8 * crossingEMA + 0.9 * calibrationErrorEMA
            let targetReliability = fxClamp(1.25 - drift, 0.35, 1.50)
            reliabilityWeight = fxClamp(0.97 * reliabilityWeight + 0.03 * targetReliability, 0.25, 1.75)
        }
    }

    private func approxPIT(target: Double, quantiles: [Double]) -> Double {
        if target <= quantiles[0] {
            return 0.01
        }
        if target >= quantiles[Self.quantileCount - 1] {
            return 0.99
        }
        for index in 0..<(Self.quantileCount - 1) where target >= quantiles[index] && target <= quantiles[index + 1] {
            let span = max(quantiles[index + 1] - quantiles[index], 1.0e-9)
            let fraction = (target - quantiles[index]) / span
            return fxClamp(tau[index] + (tau[index + 1] - tau[index]) * fraction, 0.001, 0.999)
        }
        return 0.50
    }

    private mutating func updateMoveEMA(movePoints: Double) {
        let absoluteMove = abs(fxSafeFinite(movePoints))
        if !moveReady {
            moveEMAAbs = absoluteMove
            moveReady = true
        } else {
            moveEMAAbs = 0.95 * moveEMAAbs + 0.05 * absoluteMove
        }
    }

    private mutating func updateMediumTarget(movePoints: Double) {
        let value = fxSafeFinite(movePoints)
        if !mediumReady {
            mediumTargetEMA = value
            mediumReady = true
        } else {
            mediumTargetEMA = 0.98 * mediumTargetEMA + 0.02 * value
        }
    }

    private func enforceMonotonicity(_ quantiles: inout [Double]) {
        for index in 1..<Self.quantileCount where quantiles[index] < quantiles[index - 1] + 1.0e-4 {
            quantiles[index] = quantiles[index - 1] + 1.0e-4
        }
    }

    private static func applyVectorUpdate(
        weight: inout Double,
        g2: inout Double,
        gradient: Double,
        learningRate: Double,
        weightDecay: Double
    ) {
        g2 += gradient * gradient
        let step = learningRate / sqrt(g2 + 1.0e-8)
        if weightDecay > 0.0 {
            weight *= 1.0 - step * weightDecay
        }
        weight = PluginSupportTools.clipSymmetric(weight + step * gradient, limit: 10.0)
    }

    private static func applyScalarUpdate(
        weight: inout Double,
        g2: inout Double,
        gradient: Double,
        learningRate: Double,
        weightDecay: Double,
        lowerBound: Double,
        upperBound: Double
    ) {
        g2 += gradient * gradient
        let step = learningRate / sqrt(g2 + 1.0e-8)
        if weightDecay > 0.0 {
            weight *= 1.0 - step * weightDecay
        }
        weight = fxClamp(weight + step * gradient, lowerBound, upperBound)
    }

    private static func quantileGradientSignal(
        target: Double,
        prediction: Double,
        tau: Double,
        huberDelta: Double
    ) -> Double {
        let error = fxSafeFinite(target) - fxSafeFinite(prediction)
        let direction = error >= 0.0 ? tau : tau - 1.0
        let absoluteError = abs(error)
        if absoluteError <= huberDelta {
            return direction
        }
        return direction * (huberDelta / max(absoluteError, 1.0e-9))
    }

    private static func sessionBucket(context: PluginContextV4) -> Int {
        if (0..<Self.sessionCount).contains(context.sessionBucket) {
            return context.sessionBucket
        }
        guard context.sampleTimeUTC > 0 else { return 0 }
        let secondsInDay = Int(context.sampleTimeUTC % 86_400)
        let hour = max(0, min(23, secondsInDay / 3_600))
        if hour < 6 {
            return 0
        }
        if hour < 12 {
            return 1
        }
        if hour < 20 {
            return 2
        }
        return 3
    }

    private static func regimeBucket(x: [Double], context: PluginContextV4, cost: Double) -> Int {
        let volatility = max(abs(x[safe: 4]), abs(x[safe: 5]), abs(x[safe: 9]))
        let volumePressure = context.dataHasVolume ? abs(x[safe: 6]) : 0.0
        let score = volatility + 0.20 * max(cost, 0.0) + 0.10 * volumePressure
        if score < 0.90 {
            return 0
        }
        if score < 1.80 {
            return 1
        }
        return 2
    }

    private static func volatilityAndVolumeSignal(_ x: [Double], dataHasVolume: Bool) -> Double {
        let volatility = max(abs(x[safe: 4]), abs(x[safe: 5]), abs(x[safe: 9]))
        let volume = dataHasVolume ? abs(x[safe: 6]) + 0.35 * abs(x[safe: 80]) + 0.25 * abs(x[safe: 81]) : 0.0
        return volatility + 0.30 * volume
    }

    private static func preparedFeatures(_ x: [Double], dataHasVolume: Bool) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<FXDataEngineConstants.aiWeights {
            output[index] = fxClamp(index < x.count ? fxSafeFinite(x[index]) : 0.0, -50.0, 50.0)
        }
        if !dataHasVolume {
            zeroVolumeFeatures(&output)
        }
        return output
    }

    private static func preparedWindow(_ window: [[Double]], dataHasVolume: Bool) -> [[Double]] {
        window.map { preparedFeatures($0, dataHasVolume: dataHasVolume) }
    }

    private static func zeroVolumeFeatures(_ features: inout [Double]) {
        for index in [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83] where index < features.count {
            features[index] = 0.0
        }
    }

    private static func softplus(_ value: Double) -> Double {
        let z = fxSafeFinite(value)
        if z > 30.0 {
            return z
        }
        if z < -30.0 {
            return exp(z)
        }
        return log(1.0 + exp(z))
    }

    private static func softmax3(_ logits: [Double]) -> [Double] {
        let safe = (0..<Self.classCount).map { index -> Double in
            index < logits.count ? fxSafeFinite(logits[index]) : 0.0
        }
        let maximum = safe.max() ?? 0.0
        var exponentials = Array(repeating: 0.0, count: Self.classCount)
        var sum = 0.0
        for index in 0..<Self.classCount {
            let value = exp(PluginSupportTools.clipSymmetric(safe[index] - maximum, limit: 30.0))
            exponentials[index] = value
            sum += value
        }
        guard sum > 0.0 else {
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        }
        return exponentials.map { $0 / sum }
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        var value = 0.0
        let count = min(lhs.count, rhs.count)
        for index in 0..<count {
            value += lhs[index] * rhs[index]
        }
        return value
    }

    private static func zeroSideFeatureMatrix() -> [[Double]] {
        Array(
            repeating: Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights),
            count: Self.sideCount
        )
    }
}

private extension Array where Element == Double {
    subscript(safe index: Int) -> Double {
        indices.contains(index) ? self[index] : 0.0
    }
}
