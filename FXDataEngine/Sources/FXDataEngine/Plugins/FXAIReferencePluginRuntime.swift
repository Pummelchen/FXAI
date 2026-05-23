import Foundation

public struct FXAIReferencePluginRuntime: Sendable {
    private static let classCount = 3
    private static let expertCount = 4
    private static let memoryCapacity = 128
    private static let stumpCount = 8

    private var steps: Int
    private var classBias: [Double]
    private var classWeights: [[Double]]
    private var moveEMA: Double
    private var moveHead: PluginMoveHead
    private var calibrator: PluginTernaryCalibrator
    private var contextCalibration: PluginContextCalibrationBank
    private var sequenceEMA: [Double]
    private var memorySlots: [FXAIMemorySlot]
    private var memoryHead: Int
    private var memoryCount: Int
    private var expertBias: [[Double]]
    private var expertWeights: [[[Double]]]
    private var expertUsage: [Double]
    private var treeLeafWeights: [[Double]]

    public init(descriptor: FXAIPluginImplementationDescriptor) {
        let featureCount = FXDataEngineConstants.aiWeights
        self.steps = 0
        self.classBias = Self.seededClassBias(descriptor: descriptor)
        self.classWeights = Self.seededClassWeights(descriptor: descriptor)
        self.moveEMA = 1.0
        self.moveHead = PluginMoveHead()
        self.calibrator = PluginTernaryCalibrator()
        self.contextCalibration = PluginContextCalibrationBank()
        self.sequenceEMA = Array(repeating: 0.0, count: featureCount)
        self.memorySlots = Array(repeating: FXAIMemorySlot(), count: Self.memoryCapacity)
        self.memoryHead = 0
        self.memoryCount = 0
        self.expertBias = Array(
            repeating: Array(repeating: 0.0, count: Self.classCount),
            count: Self.expertCount
        )
        self.expertWeights = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0, count: featureCount),
                count: Self.classCount
            ),
            count: Self.expertCount
        )
        self.expertUsage = Array(repeating: 1.0 / Double(Self.expertCount), count: Self.expertCount)
        self.treeLeafWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.classCount),
            count: Self.stumpCount * 2
        )
    }

    public mutating func train(
        _ request: TrainRequestV4,
        descriptor: FXAIPluginImplementationDescriptor,
        hyperParameters: HyperParameters
    ) {
        let features = Self.engineeredFeatures(request.x, window: request.xWindow, windowSize: request.windowSize, context: request.context)
        let rawProbabilities = rawPrediction(features: features, context: request.context, descriptor: descriptor)
        let label = request.labelClass
        let sampleWeight = Self.sampleWeight(request)
        guard sampleWeight > 0.0 else { return }

        updateOnlineClassHead(
            features: features,
            rawProbabilities: rawProbabilities,
            label: label,
            sampleWeight: sampleWeight,
            hyperParameters: hyperParameters
        )
        updateTreeStumps(features: features, label: label, rawProbabilities: rawProbabilities, sampleWeight: sampleWeight, hyperParameters: hyperParameters)
        updateExperts(features: features, label: label, sampleWeight: sampleWeight, hyperParameters: hyperParameters)
        updateSequenceState(features: features, sampleWeight: sampleWeight)
        appendMemory(features: features, label: label, movePoints: request.movePoints, sampleWeight: sampleWeight)

        moveHead.update(x: features, movePoints: request.movePoints, hyperParameters: hyperParameters, sampleWeight: sampleWeight)
        let moveTarget = max(abs(fxSafeFinite(request.movePoints)), request.context.minMovePoints, 1.0)
        let moveAlpha = fxClamp(0.02 * sampleWeight, 0.005, 0.20)
        moveEMA = (1.0 - moveAlpha) * moveEMA + moveAlpha * moveTarget
        calibrator.update(
            rawProbabilities: rawProbabilities,
            labelClass: label,
            sampleWeight: sampleWeight,
            learningRate: hyperParameters.learningRate
        )
        contextCalibration.update(
            labelClass: label,
            expectedMovePoints: estimatedMove(features: features, edge: classEdge(features), context: request.context),
            movePoints: request.movePoints,
            sampleWeight: sampleWeight,
            context: request.context
        )
        steps += 1
    }

    public func predict(
        _ request: PredictRequestV4,
        descriptor: FXAIPluginImplementationDescriptor,
        hyperParameters: HyperParameters
    ) -> PredictionV4 {
        let features = Self.engineeredFeatures(request.x, window: request.xWindow, windowSize: request.windowSize, context: request.context)
        var probabilities = rawPrediction(features: features, context: request.context, descriptor: descriptor)
        probabilities = calibrator.calibrated(probabilities)
        probabilities = contextCalibration.classCalibrated(probabilities: probabilities, context: request.context)

        let edge = classEdge(features) + descriptor.profile.edge(request)
        let moveMean = contextCalibration.expectedMoveCalibrated(
            estimatedMove(features: features, edge: edge, context: request.context),
            context: request.context
        )
        let output = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: moveMean,
            moveQ25Points: max(0.0, moveMean * 0.65),
            moveQ50Points: moveMean,
            moveQ75Points: max(moveMean, moveMean * 1.35),
            mfeMeanPoints: moveMean,
            maeMeanPoints: max(0.0, 0.35 * moveMean),
            hitTimeFraction: fxClamp(0.70 - 0.20 * max(probabilities[0], probabilities[1]), 0.0, 1.0),
            pathRisk: probabilities[LabelClass.skip.rawValue],
            fillRisk: fxClamp(request.context.priceCostPoints / max(moveMean, request.context.minMovePoints, 0.25), 0.0, 1.0),
            confidence: max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]),
            reliability: reliability(probabilities: probabilities, descriptor: descriptor),
            hasQuantiles: true,
            hasConfidence: true,
            hasPathQuality: true
        )
        let prediction = PluginContextRuntimeTools.fillPrediction(
            modelOutput: output,
            calibratedMoveMeanPoints: moveMean,
            context: request.context
        )
        return prediction
    }

    private func rawPrediction(
        features: [Double],
        context: PluginContextV4,
        descriptor: FXAIPluginImplementationDescriptor?
    ) -> [Double] {
        let profile = descriptor?.profile ?? .linear
        let familyEdge = familyEdge(features: features, context: context, profile: profile)
        let learnedEdge = classEdge(features)
        let memoryEdge = profile == .memory || profile == .world || profile == .sequence ? memoryVote(features).edge : 0.0
        let mixture = profile == .mixture ? mixtureProbabilities(features) : nil
        let expertEdge = mixture?.edge ?? 0.0
        let edge = fxClamp(familyEdge + learnedEdge + memoryEdge + expertEdge, -12.0, 12.0)
        let volumeLift = context.dataHasVolume ? 0.16 * fxClamp(abs(Self.feature(features, 6)), 0.0, 1.0) : 0.0
        let costDrag = fxClamp(context.priceCostPoints / max(context.minMovePoints, 0.25), 0.0, 2.0)
        let tradeLogit = 1.05 + profile.strengthScale * abs(edge) + volumeLift - 0.22 * costDrag
        let tradeProbability = fxClamp(PluginSupportTools.sigmoid(tradeLogit), 0.03, 0.97)
        let upProbability = fxClamp(PluginSupportTools.sigmoid(2.40 * edge), 0.02, 0.98)

        var base = [
            tradeProbability * (1.0 - upProbability),
            tradeProbability * upProbability,
            1.0 - tradeProbability
        ]
        if let mixture {
            base = (0..<Self.classCount).map { 0.65 * base[$0] + 0.35 * mixture.probabilities[$0] }
        }
        return PluginContextRuntimeTools.normalizeClassDistribution(base)
    }

    private func familyEdge(features: [Double], context: PluginContextV4, profile: FXAIReferencePluginProfile) -> Double {
        let shortReturn = Self.feature(features, 0)
        let mediumSlope = Self.feature(features, 3)
        let volatility = max(abs(Self.feature(features, 4)), abs(Self.feature(features, 5)), 1.0e-6)
        let volume = context.dataHasVolume ? Self.feature(features, 6) : 0.0
        let mtf = Self.feature(features, 7) - Self.feature(features, 8) + 0.35 * Self.feature(features, 9)
        let contextEdge = 0.40 * Self.feature(features, 10) + 0.35 * Self.feature(features, 12) + 0.25 * Self.feature(features, 64)
        let priceShape = 0.35 * Self.feature(features, 18) - 0.20 * Self.feature(features, 19) + 0.20 * Self.feature(features, 20)
        let deterministic = seededProjection(features)

        switch profile {
        case .linear:
            return 0.48 * shortReturn + 0.34 * mediumSlope + 0.14 * mtf + 0.06 * volume + deterministic
        case .tree:
            return treeVote(features) + 0.18 * contextEdge + 0.08 * volume + deterministic
        case .sequence:
            return 0.26 * shortReturn + 0.24 * mediumSlope + 0.32 * mtf + 0.12 * sequenceEdge(features) + 0.06 * volume + deterministic
        case .distribution:
            return (0.35 * shortReturn + 0.22 * mtf + 0.18 * priceShape + 0.10 * volume + deterministic) / max(1.0 + volatility, 1.0)
        case .statistical:
            return 0.26 * shortReturn + 0.18 * mtf - 0.12 * Self.feature(features, 4) + 0.18 * contextEdge + 0.10 * volume + deterministic
        case .factor:
            return 0.18 * shortReturn + 0.16 * mtf + 0.44 * contextEdge + 0.12 * volume + deterministic
        case .trend:
            return 0.26 * shortReturn + 0.36 * mediumSlope + 0.26 * mtf + 0.10 * volume + deterministic
        case .mixture:
            return 0.20 * shortReturn + 0.20 * mediumSlope + 0.22 * mtf + 0.20 * contextEdge + 0.08 * volume + deterministic
        case .memory:
            return 0.16 * shortReturn + 0.18 * mediumSlope + 0.28 * mtf + 0.16 * contextEdge + 0.10 * volume + deterministic
        case .world:
            return 0.14 * shortReturn + 0.18 * mediumSlope + 0.20 * mtf + 0.34 * contextEdge + 0.08 * volume + deterministic
        case .reinforcement:
            let riskPenalty = 0.14 * abs(Self.feature(features, 11)) + 0.10 * abs(Self.feature(features, 21))
            return 0.22 * shortReturn + 0.22 * mediumSlope + 0.22 * mtf + 0.16 * contextEdge + 0.12 * volume - riskPenalty + deterministic
        }
    }

    private func classEdge(_ features: [Double]) -> Double {
        dot(classWeights[LabelClass.buy.rawValue], features) + classBias[LabelClass.buy.rawValue] -
            dot(classWeights[LabelClass.sell.rawValue], features) - classBias[LabelClass.sell.rawValue]
    }

    private func estimatedMove(features: [Double], edge: Double, context: PluginContextV4) -> Double {
        let learned = moveHead.ready ? moveHead.predictRaw(features) : 0.0
        let volatility = max(abs(Self.feature(features, 4)), abs(Self.feature(features, 5)), 0.0)
        let edgeMove = abs(edge) * 72.0 + 8.0 * volatility
        let minimum = max(context.minMovePoints, context.priceCostPoints, 1.0)
        return max(minimum, learned, moveEMA, edgeMove)
    }

    private func reliability(probabilities: [Double], descriptor: FXAIPluginImplementationDescriptor) -> Double {
        let trained = fxClamp(Double(steps) / 64.0, 0.0, 1.0)
        let confidence = max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue])
        return fxClamp(descriptor.profile.baseReliability + 0.20 * trained + 0.12 * confidence, 0.0, 1.0)
    }

    private mutating func updateOnlineClassHead(
        features: [Double],
        rawProbabilities: [Double],
        label: LabelClass,
        sampleWeight: Double,
        hyperParameters: HyperParameters
    ) {
        let learningRate = fxClamp(hyperParameters.learningRate, 0.0001, 0.08)
        let l2 = fxClamp(hyperParameters.l2, 0.0, 0.05)
        for classIndex in 0..<Self.classCount {
            let target = classIndex == label.rawValue ? 1.0 : 0.0
            let error = PluginSupportTools.clipSymmetric(target - rawProbabilities[classIndex], limit: 2.0)
            classBias[classIndex] = fxClamp(classBias[classIndex] + learningRate * sampleWeight * error, -8.0, 8.0)
            for featureIndex in 0..<FXDataEngineConstants.aiWeights {
                let gradient = sampleWeight * error * features[featureIndex] - l2 * classWeights[classIndex][featureIndex]
                classWeights[classIndex][featureIndex] = fxClamp(
                    classWeights[classIndex][featureIndex] + learningRate * PluginSupportTools.clipSymmetric(gradient, limit: 8.0),
                    -8.0,
                    8.0
                )
            }
        }
    }

    private mutating func updateTreeStumps(
        features: [Double],
        label: LabelClass,
        rawProbabilities: [Double],
        sampleWeight: Double,
        hyperParameters: HyperParameters
    ) {
        let learningRate = fxClamp(hyperParameters.xgbLearningRate, 0.0005, 0.08)
        for stump in 0..<Self.stumpCount {
            let leaf = stumpLeaf(stump, features: features)
            for classIndex in 0..<Self.classCount {
                let target = classIndex == label.rawValue ? 1.0 : 0.0
                let error = target - rawProbabilities[classIndex]
                treeLeafWeights[leaf][classIndex] = fxClamp(
                    treeLeafWeights[leaf][classIndex] + learningRate * sampleWeight * error,
                    -4.0,
                    4.0
                )
            }
        }
    }

    private mutating func updateExperts(
        features: [Double],
        label: LabelClass,
        sampleWeight: Double,
        hyperParameters: HyperParameters
    ) {
        let gates = expertGates(features)
        let learningRate = fxClamp(hyperParameters.learningRate, 0.0001, 0.04)
        let l2 = fxClamp(hyperParameters.l2, 0.0, 0.05)
        for expert in 0..<Self.expertCount {
            let expertProbabilities = expertProbabilities(expert: expert, features: features)
            let gate = gates[expert]
            expertUsage[expert] = 0.995 * expertUsage[expert] + 0.005 * gate
            for classIndex in 0..<Self.classCount {
                let target = classIndex == label.rawValue ? 1.0 : 0.0
                let error = target - expertProbabilities[classIndex]
                expertBias[expert][classIndex] += learningRate * sampleWeight * gate * error
                for featureIndex in stride(from: expert, to: FXDataEngineConstants.aiWeights, by: Self.expertCount) {
                    let gradient = sampleWeight * gate * error * features[featureIndex] -
                        l2 * expertWeights[expert][classIndex][featureIndex]
                    expertWeights[expert][classIndex][featureIndex] = fxClamp(
                        expertWeights[expert][classIndex][featureIndex] + learningRate * gradient,
                        -5.0,
                        5.0
                    )
                }
            }
        }
    }

    private mutating func updateSequenceState(features: [Double], sampleWeight: Double) {
        let alpha = fxClamp(0.02 * sampleWeight, 0.005, 0.10)
        for index in 0..<FXDataEngineConstants.aiWeights {
            sequenceEMA[index] = (1.0 - alpha) * sequenceEMA[index] + alpha * features[index]
        }
    }

    private mutating func appendMemory(features: [Double], label: LabelClass, movePoints: Double, sampleWeight: Double) {
        memorySlots[memoryHead] = FXAIMemorySlot(
            features: features,
            label: label,
            movePoints: abs(fxSafeFinite(movePoints)),
            weight: sampleWeight
        )
        memoryHead = (memoryHead + 1) % Self.memoryCapacity
        memoryCount = min(memoryCount + 1, Self.memoryCapacity)
    }

    private func treeVote(_ features: [Double]) -> Double {
        var edge = 0.0
        for stump in 0..<Self.stumpCount {
            let leaf = stumpLeaf(stump, features: features)
            let weights = treeLeafWeights[leaf]
            edge += weights[LabelClass.buy.rawValue] - weights[LabelClass.sell.rawValue]
        }
        return fxClamp(0.08 * edge + 0.30 * Self.signFeature(features, 0) + 0.22 * Self.signFeature(features, 3), -4.0, 4.0)
    }

    private func stumpLeaf(_ stump: Int, features: [Double]) -> Int {
        let indexes = [0, 3, 4, 6, 7, 8, 12, 18]
        let thresholds = [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0]
        let index = indexes[stump % indexes.count]
        let right = Self.feature(features, index) >= thresholds[stump % thresholds.count]
        return stump * 2 + (right ? 1 : 0)
    }

    private func sequenceEdge(_ features: [Double]) -> Double {
        var edge = 0.0
        for index in [0, 1, 2, 3, 7, 8, 13, 14] {
            edge += 0.12 * (features[index] - sequenceEMA[index])
        }
        return fxClamp(edge, -3.0, 3.0)
    }

    private func memoryVote(_ features: [Double]) -> (edge: Double, move: Double) {
        guard memoryCount > 0 else { return (0.0, 0.0) }
        var candidates: [(distance: Double, slot: FXAIMemorySlot)] = []
        candidates.reserveCapacity(memoryCount)
        for offset in 0..<memoryCount {
            let index = (memoryHead - 1 - offset + Self.memoryCapacity) % Self.memoryCapacity
            let slot = memorySlots[index]
            candidates.append((distance(features, slot.features), slot))
        }
        candidates.sort { $0.distance < $1.distance }
        var weightedEdge = 0.0
        var weightedMove = 0.0
        var total = 0.0
        for candidate in candidates.prefix(5) {
            let weight = candidate.slot.weight / max(candidate.distance, 1.0e-4)
            let direction: Double
            switch candidate.slot.label {
            case .buy: direction = 1.0
            case .sell: direction = -1.0
            case .skip: direction = 0.0
            }
            weightedEdge += weight * direction
            weightedMove += weight * candidate.slot.movePoints
            total += weight
        }
        guard total > 0.0 else { return (0.0, 0.0) }
        return (fxClamp(weightedEdge / total, -2.0, 2.0), max(0.0, weightedMove / total))
    }

    private func mixtureProbabilities(_ features: [Double]) -> (probabilities: [Double], edge: Double) {
        let gates = expertGates(features)
        var combined = Array(repeating: 0.0, count: Self.classCount)
        for expert in 0..<Self.expertCount {
            let probabilities = expertProbabilities(expert: expert, features: features)
            for classIndex in 0..<Self.classCount {
                combined[classIndex] += gates[expert] * probabilities[classIndex]
            }
        }
        combined = PluginContextRuntimeTools.normalizeClassDistribution(combined)
        return (combined, combined[LabelClass.buy.rawValue] - combined[LabelClass.sell.rawValue])
    }

    private func expertGates(_ features: [Double]) -> [Double] {
        var logits = Array(repeating: 0.0, count: Self.expertCount)
        for expert in 0..<Self.expertCount {
            let sessionPull = Double(expert) == floor(fxClamp(Self.feature(features, 16) * 4.0, 0.0, 3.0)) ? 0.20 : 0.0
            logits[expert] = 0.22 * Self.feature(features, expert) +
                0.18 * Self.feature(features, 7 + expert) -
                0.35 * (expertUsage[expert] - 0.25) +
                sessionPull
        }
        return Self.softmax(logits)
    }

    private func expertProbabilities(expert: Int, features: [Double]) -> [Double] {
        var logits = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            logits[classIndex] = expertBias[expert][classIndex] + dot(expertWeights[expert][classIndex], features)
        }
        return Self.softmax(logits)
    }

    private func seededProjection(_ features: [Double]) -> Double {
        var value = 0.0
        for index in stride(from: 0, to: FXDataEngineConstants.aiWeights, by: 11) {
            let sign = ((index + steps) & 1) == 0 ? 1.0 : -1.0
            value += sign * 0.015 * features[index]
        }
        return fxClamp(value, -0.25, 0.25)
    }

    private static func engineeredFeatures(
        _ x: [Double],
        window: [[Double]],
        windowSize: Int,
        context: PluginContextV4
    ) -> [Double] {
        var output = clippedFeatures(x)
        let size = min(max(0, windowSize), window.count, FXDataEngineConstants.maxSequenceBars)
        if size > 0 {
            for index in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 18, 21] {
                let mean = windowEMAMean(window, index: index, size: size)
                output[index] = fxClamp(0.72 * output[index] + 0.28 * mean, -8.0, 8.0)
            }
            output[79] = fxClamp(windowSlope(window, index: 0, size: size), -8.0, 8.0)
        }
        if !context.dataHasVolume {
            for index in [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83] where index < output.count {
                output[index] = 0.0
            }
        }
        return output
    }

    private static func clippedFeatures(_ x: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<output.count {
            let value = index < x.count ? fxSafeFinite(x[index]) : 0.0
            output[index] = fxClamp(value, -8.0, 8.0)
        }
        return output
    }

    private static func windowEMAMean(_ window: [[Double]], index: Int, size: Int) -> Double {
        var weight = 1.0
        var weightSum = 0.0
        var sum = 0.0
        for rowIndex in 0..<size {
            let value = index < window[rowIndex].count ? fxSafeFinite(window[rowIndex][index]) : 0.0
            sum += weight * value
            weightSum += weight
            weight *= 0.72
        }
        return weightSum > 0.0 ? sum / weightSum : 0.0
    }

    private static func windowSlope(_ window: [[Double]], index: Int, size: Int) -> Double {
        guard size > 1 else { return 0.0 }
        let first = index < window[0].count ? fxSafeFinite(window[0][index]) : 0.0
        let last = index < window[size - 1].count ? fxSafeFinite(window[size - 1][index]) : 0.0
        return (first - last) / Double(size - 1)
    }

    private static func sampleWeight(_ request: TrainRequestV4) -> Double {
        let quality = PluginQualityTargets(
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
        return fxClamp(
            request.sampleWeight * PluginSupportTools.moveSampleWeight(
                x: request.x,
                movePoints: request.movePoints,
                priceCostPoints: request.context.priceCostPoints,
                minMovePoints: request.context.minMovePoints,
                qualityTargets: quality
            ),
            0.0,
            8.0
        )
    }

    private static func seededClassBias(descriptor: FXAIPluginImplementationDescriptor) -> [Double] {
        let id = Double(descriptor.aiID.rawValue + 1)
        return [
            0.02 * sin(id * 0.73),
            0.02 * cos(id * 0.61),
            0.04 + 0.01 * sin(id * 0.37)
        ]
    }

    private static func seededClassWeights(descriptor: FXAIPluginImplementationDescriptor) -> [[Double]] {
        var output = Array(
            repeating: Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights),
            count: Self.classCount
        )
        let id = Double(descriptor.aiID.rawValue + 1)
        for index in 0..<FXDataEngineConstants.aiWeights {
            let base = 0.0025 * sin(id * Double(index + 3) * 0.017)
            output[LabelClass.buy.rawValue][index] = base
            output[LabelClass.sell.rawValue][index] = -base
            output[LabelClass.skip.rawValue][index] = 0.0015 * cos(id * Double(index + 5) * 0.013)
        }
        return output
    }

    private static func softmax(_ logits: [Double]) -> [Double] {
        let maxLogit = logits.map { fxSafeFinite($0) }.max() ?? 0.0
        var expValues = Array(repeating: 0.0, count: logits.count)
        var sum = 0.0
        for index in 0..<logits.count {
            let value = exp(PluginSupportTools.clipSymmetric(fxSafeFinite(logits[index]) - maxLogit, limit: 30.0))
            expValues[index] = value
            sum += value
        }
        guard sum > 0.0 else {
            return Array(repeating: 1.0 / Double(max(logits.count, 1)), count: logits.count)
        }
        return expValues.map { $0 / sum }
    }

    private func dot(_ weights: [Double], _ features: [Double]) -> Double {
        var result = 0.0
        for index in 0..<min(weights.count, features.count) {
            result += weights[index] * features[index]
        }
        return fxClamp(result, -20.0, 20.0)
    }

    private func distance(_ lhs: [Double], _ rhs: [Double]) -> Double {
        var result = 0.0
        for index in 0..<min(lhs.count, rhs.count, 48) {
            let delta = lhs[index] - rhs[index]
            result += delta * delta
        }
        return sqrt(max(result, 0.0))
    }

    private static func feature(_ features: [Double], _ index: Int) -> Double {
        guard index >= 0, index < features.count else { return 0.0 }
        return fxSafeFinite(features[index])
    }

    private static func signFeature(_ features: [Double], _ index: Int) -> Double {
        let value = feature(features, index)
        if value > 0.0 { return 1.0 }
        if value < 0.0 { return -1.0 }
        return 0.0
    }
}

private struct FXAIMemorySlot: Sendable {
    var features: [Double]
    var label: LabelClass
    var movePoints: Double
    var weight: Double

    init(
        features: [Double] = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights),
        label: LabelClass = .skip,
        movePoints: Double = 0.0,
        weight: Double = 0.0
    ) {
        self.features = features
        self.label = label
        self.movePoints = max(0.0, fxSafeFinite(movePoints))
        self.weight = fxClamp(fxSafeFinite(weight), 0.0, 8.0)
    }
}
