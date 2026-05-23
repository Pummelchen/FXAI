import FXDataEngine
import Foundation

public struct MemRetrdiffCPUModel: Sendable {
    private static let memoryCapacity = 256
    private static let embeddingDimension = 16
    private static let baseDimension = 16
    private static let topK = 5
    private static let classCount = 3

    private var steps: Int
    private var head: Int
    private var count: Int
    private var projection: [[Double]]
    private var embeddings: [[Double]]
    private var futureMove: [Double]
    private var moveVariance: [Double]
    private var futureUp: [Double]
    private var futureEvent: [Double]
    private var labelMass: [[Double]]
    private var sampleTime: [Int64]
    private var regimeVolatility: [Double]
    private var regimeDirection: [Double]
    private var prototypeWeight: [Double]
    private var qualityBank: PluginQualityBank

    public init() {
        self.steps = 0
        self.head = 0
        self.count = 0
        self.projection = Self.seededProjection()
        self.embeddings = Array(
            repeating: Array(repeating: 0.0, count: Self.embeddingDimension),
            count: Self.memoryCapacity
        )
        self.futureMove = Array(repeating: 0.0, count: Self.memoryCapacity)
        self.moveVariance = Array(repeating: 0.0, count: Self.memoryCapacity)
        self.futureUp = Array(repeating: 0.0, count: Self.memoryCapacity)
        self.futureEvent = Array(repeating: 0.0, count: Self.memoryCapacity)
        self.labelMass = Array(
            repeating: [0.0, 0.0, 1.0],
            count: Self.memoryCapacity
        )
        self.sampleTime = Array(repeating: 0, count: Self.memoryCapacity)
        self.regimeVolatility = Array(repeating: 0.0, count: Self.memoryCapacity)
        self.regimeDirection = Array(repeating: 0.0, count: Self.memoryCapacity)
        self.prototypeWeight = Array(repeating: 1.0, count: Self.memoryCapacity)
        self.qualityBank = PluginQualityBank()
    }

    public mutating func reset() {
        self = MemRetrdiffCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        let base = buildBase(
            x: x,
            window: window,
            windowSize: request.windowSize,
            context: request.context
        )
        let embedding = embed(base: base)
        let previousPrediction = predictInternal(
            x: x,
            window: window,
            windowSize: request.windowSize,
            context: request.context
        )
        learnProjection(
            base: base,
            label: label,
            movePoints: request.movePoints,
            predictedBuy: previousPrediction.output.classProbabilities[LabelClass.buy.rawValue],
            predictedEvent: 1.0 - previousPrediction.output.classProbabilities[LabelClass.skip.rawValue],
            predictedMove: previousPrediction.output.moveMeanPoints,
            minMovePoints: request.context.minMovePoints
        )

        let timestamp = request.context.sampleTimeUTC
        let regimeVol = Self.regimeVolatilitySignal(x, dataHasVolume: request.context.dataHasVolume)
        let regimeDir = x[safe: 0] - x[safe: 1]
        if !tryMergePrototype(
            embedding: embedding,
            label: label,
            movePoints: request.movePoints,
            sampleTimeUTC: timestamp,
            regimeVolatility: regimeVol,
            regimeDirection: regimeDir
        ) {
            appendMemory(
                embedding: embedding,
                label: label,
                movePoints: request.movePoints,
                sampleTimeUTC: timestamp,
                regimeVolatility: regimeVol,
                regimeDirection: regimeDir
            )
        }

        let sampleWeight = fxClamp(
            request.sampleWeight *
                PluginSupportTools.moveSampleWeight(
                    x: x,
                    movePoints: request.movePoints,
                    priceCostPoints: request.context.priceCostPoints,
                    minMovePoints: request.context.minMovePoints,
                    qualityTargets: PluginQualityTargets(request: request)
                ),
            0.20,
            4.00
        )
        qualityBank.update(request: request, sampleWeight: sampleWeight)
        steps += 1
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let prediction = predictInternal(
            x: x,
            window: window,
            windowSize: request.windowSize,
            context: request.context
        )
        return PluginContextRuntimeTools.fillPrediction(
            modelOutput: prediction.output,
            calibratedMoveMeanPoints: prediction.output.moveMeanPoints,
            context: request.context
        )
    }

    private func predictInternal(
        x: [Double],
        window: [[Double]],
        windowSize: Int,
        context: PluginContextV4
    ) -> (output: PluginModelOutputV4, nearestDistance: Double) {
        if count < 8 {
            var cold = PluginModelOutputV4(
                classProbabilities: [0.10, 0.10, 0.80],
                moveMeanPoints: 0.0,
                moveQ25Points: 0.0,
                moveQ50Points: 0.0,
                moveQ75Points: 0.0,
                mfeMeanPoints: 0.0,
                maeMeanPoints: 0.0,
                hitTimeFraction: 1.0,
                pathRisk: 0.80,
                fillRisk: 0.0,
                confidence: 0.0,
                reliability: 0.0,
                hasQuantiles: true,
                hasConfidence: true,
                hasPathQuality: true
            )
            cold = PluginPathQualityTools.populatedOutput(
                cold,
                x: x,
                window: window,
                context: context,
                family: .retrieval,
                activityGate: 0.20,
                structuralQuality: 0.0,
                qualityPriors: qualityBank.priors(context: context),
                declaredWindowSize: windowSize
            )
            return (cold, .infinity)
        }

        let base = buildBase(x: x, window: window, windowSize: windowSize, context: context)
        let embedding = embed(base: base)
        let retrieved = retrieveTopK(embedding)
        guard let nearest = retrieved.first, nearest.index >= 0 else {
            let fallback = PluginModelOutputV4(classProbabilities: [0.10, 0.10, 0.80], hasQuantiles: true, hasConfidence: true)
            return (fallback, .infinity)
        }

        let now = context.sampleTimeUTC
        let currentVol = Self.regimeVolatilitySignal(x, dataHasVolume: context.dataHasVolume)
        let currentDirection = x[safe: 0] - x[safe: 1]
        var classMass = Array(repeating: 0.0, count: Self.classCount)
        var weightSum = 0.0
        var moveMean = 0.0
        var moveSecondMoment = 0.0
        var support = 0.0

        for candidate in retrieved where candidate.index >= 0 {
            let slot = candidate.index
            var weight = 1.0 / max(0.03, candidate.distance)
            let regimeSimilarity = 1.0 / (
                1.0 +
                    abs(regimeVolatility[slot] - currentVol) +
                    0.50 * abs(regimeDirection[slot] - currentDirection)
            )
            var recency = 1.0
            if sampleTime[slot] > 0, now > sampleTime[slot] {
                let ageMinutes = Double(now - sampleTime[slot]) / 60.0
                recency = 1.0 / (1.0 + ageMinutes / 720.0)
            }
            let prototype = sqrt(max(prototypeWeight[slot], 1.0))
            weight *= regimeSimilarity * recency * prototype
            support += prototype
            for classIndex in 0..<Self.classCount {
                classMass[classIndex] += weight * labelMass[slot][classIndex]
            }
            moveMean += weight * futureMove[slot]
            moveSecondMoment += weight * (moveVariance[slot] + futureMove[slot] * futureMove[slot])
            weightSum += weight
        }

        guard weightSum > 0.0 else {
            let fallback = PluginModelOutputV4(classProbabilities: [0.10, 0.10, 0.80], hasQuantiles: true, hasConfidence: true)
            return (fallback, nearest.distance)
        }

        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution(classMass.map { $0 / weightSum })
        moveMean /= weightSum
        moveSecondMoment /= weightSum
        let variance = max(0.0, moveSecondMoment - moveMean * moveMean)
        let sigma = sqrt(variance + 1.0e-6)
        let q25 = max(0.0, moveMean - 0.674 * sigma)
        let q50 = max(q25, moveMean)
        let q75 = max(q50, moveMean + 0.674 * sigma)
        let entropy = probabilities.reduce(0.0) { partial, probability in
            let p = fxClamp(probability, 1.0e-9, 1.0)
            return partial - p * log(p)
        }
        let confidence = fxClamp(1.0 - entropy / log(3.0), 0.0, 1.0)
        let reliability = fxClamp((1.0 / (1.0 + nearest.distance)) * min(1.0, support / 8.0), 0.0, 1.0)
        let baseOutput = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: max(0.0, moveMean),
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: max(q75, moveMean),
            maeMeanPoints: max(0.0, moveMean * 0.35),
            hitTimeFraction: fxClamp(0.70 - 0.30 * reliability + 0.15 * probabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            pathRisk: fxClamp(0.35 * probabilities[LabelClass.skip.rawValue] + 0.30 * (1.0 - reliability), 0.0, 1.0),
            fillRisk: fxClamp(context.priceCostPoints / max(moveMean + context.minMovePoints, 0.25), 0.0, 1.0),
            confidence: confidence,
            reliability: reliability,
            hasQuantiles: true,
            hasConfidence: true,
            hasPathQuality: true
        )
        let output = PluginPathQualityTools.populatedOutput(
            baseOutput,
            x: x,
            window: window,
            context: context,
            family: .retrieval,
            activityGate: 1.0 - probabilities[LabelClass.skip.rawValue],
            structuralQuality: reliability,
            qualityPriors: qualityBank.priors(context: context),
            declaredWindowSize: windowSize
        )
        return (output, nearest.distance)
    }

    private mutating func learnProjection(
        base: [Double],
        label: LabelClass,
        movePoints: Double,
        predictedBuy: Double,
        predictedEvent: Double,
        predictedMove: Double,
        minMovePoints: Double
    ) {
        let directionTarget = label == .buy ? 1.0 : (label == .sell ? -1.0 : 0.0)
        let eventTarget = label == .skip ? 0.0 : 1.0
        let minimumMove = max(minMovePoints, 0.10)
        let magnitudeTarget = fxClamp(abs(movePoints) / minimumMove, 0.0, 8.0)
        let directionError = directionTarget - (2.0 * predictedBuy - 1.0)
        let eventError = eventTarget - predictedEvent
        let magnitudeError = magnitudeTarget - fxClamp(predictedMove / minimumMove, 0.0, 8.0)
        let learningRate = 0.0030
        for row in 0..<Self.embeddingDimension {
            let signal = row % 3 == 0 ? directionError : (row % 3 == 1 ? eventError : 0.50 * magnitudeError)
            for column in 0..<Self.baseDimension {
                projection[row][column] = PluginSupportTools.clipSymmetric(
                    0.999 * projection[row][column] + learningRate * signal * base[column],
                    limit: 2.0
                )
            }
        }
    }

    private mutating func tryMergePrototype(
        embedding: [Double],
        label: LabelClass,
        movePoints: Double,
        sampleTimeUTC: Int64,
        regimeVolatility: Double,
        regimeDirection: Double
    ) -> Bool {
        let retrieved = retrieveTopK(embedding)
        guard let nearest = retrieved.first, nearest.index >= 0, nearest.distance <= 0.08 else {
            return false
        }
        let slot = nearest.index
        let totalMass = max(labelMass[slot].reduce(0.0, +), 1.0e-9)
        var dominant = LabelClass.skip.rawValue
        if labelMass[slot][LabelClass.buy.rawValue] > labelMass[slot][dominant] {
            dominant = LabelClass.buy.rawValue
        }
        if labelMass[slot][LabelClass.sell.rawValue] > labelMass[slot][dominant] {
            dominant = LabelClass.sell.rawValue
        }
        if dominant != label.rawValue, labelMass[slot][dominant] / totalMass > 0.70 {
            return false
        }

        let upTarget = label == .buy ? 1.0 : 0.0
        let eventTarget = label == .skip ? 0.0 : 1.0
        let nextPrototypeWeight = min(16.0, prototypeWeight[slot] + 1.0)
        let alpha = 1.0 / nextPrototypeWeight
        for index in 0..<Self.embeddingDimension {
            embeddings[slot][index] = (1.0 - alpha) * embeddings[slot][index] + alpha * embedding[index]
        }
        futureMove[slot] = (1.0 - alpha) * futureMove[slot] + alpha * abs(movePoints)
        let moveDelta = abs(movePoints) - futureMove[slot]
        moveVariance[slot] = (1.0 - alpha) * moveVariance[slot] + alpha * moveDelta * moveDelta
        futureUp[slot] = (1.0 - alpha) * futureUp[slot] + alpha * upTarget
        futureEvent[slot] = (1.0 - alpha) * futureEvent[slot] + alpha * eventTarget
        for classIndex in 0..<Self.classCount {
            let target = classIndex == label.rawValue ? 1.0 : 0.0
            labelMass[slot][classIndex] = (1.0 - alpha) * labelMass[slot][classIndex] + alpha * target
        }
        self.regimeVolatility[slot] = (1.0 - alpha) * self.regimeVolatility[slot] + alpha * regimeVolatility
        self.regimeDirection[slot] = (1.0 - alpha) * self.regimeDirection[slot] + alpha * regimeDirection
        sampleTime[slot] = sampleTimeUTC
        prototypeWeight[slot] = nextPrototypeWeight
        return true
    }

    private mutating func appendMemory(
        embedding: [Double],
        label: LabelClass,
        movePoints: Double,
        sampleTimeUTC: Int64,
        regimeVolatility: Double,
        regimeDirection: Double
    ) {
        let slot = head
        embeddings[slot] = embedding
        futureMove[slot] = abs(movePoints)
        moveVariance[slot] = 0.25 * abs(movePoints) * abs(movePoints)
        futureUp[slot] = label == .buy ? 1.0 : 0.0
        futureEvent[slot] = label == .skip ? 0.0 : 1.0
        for classIndex in 0..<Self.classCount {
            labelMass[slot][classIndex] = classIndex == label.rawValue ? 1.0 : 0.0
        }
        sampleTime[slot] = sampleTimeUTC
        self.regimeVolatility[slot] = regimeVolatility
        self.regimeDirection[slot] = regimeDirection
        prototypeWeight[slot] = 1.0
        head = (head + 1) % Self.memoryCapacity
        count = min(count + 1, Self.memoryCapacity)
    }

    private func retrieveTopK(_ embedding: [Double]) -> [(index: Int, distance: Double)] {
        var top = Array(repeating: (index: -1, distance: Double.greatestFiniteMagnitude), count: Self.topK)
        guard count > 0 else { return top }
        for offset in 0..<count {
            let slot = (head - 1 - offset + Self.memoryCapacity) % Self.memoryCapacity
            let distance = squaredDistance(embedding, embeddings[slot])
            for rank in 0..<Self.topK where distance < top[rank].distance {
                if rank < Self.topK - 1 {
                    for shift in stride(from: Self.topK - 1, to: rank, by: -1) {
                        top[shift] = top[shift - 1]
                    }
                }
                top[rank] = (slot, distance)
                break
            }
        }
        return top
    }

    private func embed(base: [Double]) -> [Double] {
        var embedding = Array(repeating: 0.0, count: Self.embeddingDimension)
        for row in 0..<Self.embeddingDimension {
            embedding[row] = Self.dot(projection[row], base)
        }
        let norm = sqrt(max(Self.dot(embedding, embedding), 1.0e-9))
        return embedding.map { $0 / norm }
    }

    private func buildBase(
        x: [Double],
        window: [[Double]],
        windowSize: Int,
        context: PluginContextV4
    ) -> [Double] {
        var base = Array(repeating: 0.0, count: Self.baseDimension)
        for index in 0..<8 {
            base[index] = x[safe: index]
        }

        let size = PluginContextRuntimeTools.effectiveWindowSize(window, declaredSize: windowSize)
        let mean1 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureMean(window, featureIndex: 0, declaredSize: size) : 0.0
        let mean2 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureMean(window, featureIndex: 1, declaredSize: size) : 0.0
        let mean4 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureMean(window, featureIndex: 3, declaredSize: size) : 0.0
        let var1 = size > 1 ? PluginContextRuntimeTools.currentWindowFeatureStd(window, featureIndex: 0, declaredSize: size) : 0.0
        let var2 = size > 1 ? PluginContextRuntimeTools.currentWindowFeatureStd(window, featureIndex: 1, declaredSize: size) : 0.0
        let trend1 = size > 1 ? PluginContextRuntimeTools.currentWindowFeatureSlope(window, featureIndex: 0, declaredSize: size) : 0.0
        let trend2 = size > 1 ? PluginContextRuntimeTools.currentWindowFeatureSlope(window, featureIndex: 1, declaredSize: size) : 0.0

        let attn1 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureEMAMean(window, featureIndex: 0, decay: 0.62, declaredSize: size) : 0.0
        let attn2 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureEMAMean(window, featureIndex: 1, decay: 0.62, declaredSize: size) : 0.0
        let attn3 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureEMAMean(window, featureIndex: 2, decay: 0.68, declaredSize: size) : 0.0
        let attn4 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureEMAMean(window, featureIndex: 3, decay: 0.68, declaredSize: size) : 0.0
        let attn5 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureEMAMean(window, featureIndex: 4, decay: 0.72, declaredSize: size) : 0.0
        let attn6 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureEMAMean(window, featureIndex: 5, decay: 0.72, declaredSize: size) : 0.0
        let attn7 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureEMAMean(window, featureIndex: 6, decay: 0.76, declaredSize: size) : 0.0
        let convFast1 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureRecentDelta(window, featureIndex: 0, recentBars: 3, declaredSize: size) : 0.0
        let convFast2 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureRecentDelta(window, featureIndex: 1, recentBars: 3, declaredSize: size) : 0.0
        let convFast4 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureRecentDelta(window, featureIndex: 3, recentBars: 3, declaredSize: size) : 0.0
        let convSlow3 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureRecentMean(window, featureIndex: 2, recentBars: min(max(size / 2, 1), size), declaredSize: size) : 0.0
        let convSlow5 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureRecentMean(window, featureIndex: 4, recentBars: min(max(size / 2, 1), size), declaredSize: size) : 0.0
        let convSlow6 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureRecentMean(window, featureIndex: 5, recentBars: min(max(size / 2, 1), size), declaredSize: size) : 0.0
        let convSlow7 = size > 0 ? PluginContextRuntimeTools.currentWindowFeatureRecentMean(window, featureIndex: 6, recentBars: min(max(size / 2, 1), size), declaredSize: size) : 0.0
        let block1 = 0.55 * mean1 + 0.25 * trend1 + 0.20 * attn1
        let block2 = 0.55 * mean2 + 0.25 * trend2 + 0.20 * attn2
        let block3 = 0.50 * attn3 + 0.25 * convSlow3 + 0.25 * x[safe: 3]
        let block4 = 0.50 * attn4 + 0.25 * convFast4 + 0.25 * x[safe: 4]
        let block5 = 0.50 * attn5 + 0.25 * convSlow5 + 0.25 * x[safe: 5]
        let block6 = 0.50 * attn6 + 0.25 * convSlow6 + 0.25 * x[safe: 6]
        let block7 = 0.50 * attn7 + 0.25 * convSlow7 + 0.25 * x[safe: 7]

        base[8] = PluginSupportTools.clipSymmetric(0.48 * mean1 + 0.18 * trend1 + 0.16 * attn1 + 0.08 * convFast1 + 0.10 * block1, limit: 8.0)
        base[9] = PluginSupportTools.clipSymmetric(0.42 * trend1 + 0.18 * attn2 + 0.14 * convFast2 + 0.16 * block2, limit: 8.0)
        base[10] = PluginSupportTools.clipSymmetric(0.56 * var1 + 0.18 * abs(attn6) + 0.12 * abs(convSlow6) + 0.14 * abs(block6), limit: 8.0)
        base[11] = PluginSupportTools.clipSymmetric(0.48 * mean2 + 0.16 * trend2 + 0.16 * attn3 + 0.08 * convSlow3 + 0.12 * block3, limit: 8.0)
        base[12] = PluginSupportTools.clipSymmetric(0.42 * trend2 + 0.16 * attn4 + 0.14 * convFast4 + 0.16 * block4, limit: 8.0)
        base[13] = PluginSupportTools.clipSymmetric(0.56 * var2 + 0.18 * abs(attn5) + 0.12 * abs(convSlow5) + 0.14 * abs(block5), limit: 8.0)
        base[14] = PluginSupportTools.clipSymmetric(0.56 * mean4 + 0.14 * attn7 + 0.12 * convSlow7 + 0.18 * block7, limit: 8.0)
        base[15] = Double(Self.sessionBucket(context))
        return base
    }

    private func squaredDistance(_ lhs: [Double], _ rhs: [Double]) -> Double {
        var value = 0.0
        for index in 0..<Self.embeddingDimension {
            let delta = lhs[index] - rhs[index]
            value += delta * delta
        }
        return value
    }

    private static func seededProjection() -> [[Double]] {
        (0..<Self.embeddingDimension).map { row in
            (0..<Self.baseDimension).map { column in
                var hash = UInt32((row + 1) * 73_856_093) ^ UInt32((column + 3) * 19_349_663) ^ 2_654_435_769
                hash ^= hash >> 13
                hash = hash &* 1_274_126_177
                hash ^= hash >> 16
                let unit = Double(hash & 2_147_483_647) / 2_147_483_647.0
                return 2.0 * unit - 1.0
            }
        }
    }

    private static func sessionBucket(_ context: PluginContextV4) -> Int {
        if (0..<FXDataEngineConstants.pluginSessionBuckets).contains(context.sessionBucket) {
            return context.sessionBucket
        }
        return PluginContractTools.deriveSessionBucket(timestampUTC: context.sampleTimeUTC)
    }

    private static func regimeVolatilitySignal(_ x: [Double], dataHasVolume: Bool) -> Double {
        let volatility = max(abs(x[safe: 4]), abs(x[safe: 5]), abs(x[safe: 9]))
        let volume = dataHasVolume ? abs(x[safe: 6]) + 0.35 * abs(x[safe: 80]) + 0.25 * abs(x[safe: 81]) : 0.0
        return volatility + 0.20 * volume
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

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        var value = 0.0
        let count = min(lhs.count, rhs.count)
        for index in 0..<count {
            value += lhs[index] * rhs[index]
        }
        return value
    }
}

private extension Array where Element == Double {
    subscript(safe index: Int) -> Double {
        indices.contains(index) ? self[index] : 0.0
    }
}
