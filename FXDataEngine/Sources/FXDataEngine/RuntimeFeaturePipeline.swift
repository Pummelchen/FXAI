import Foundation

public struct RuntimeFeaturePipelineRequirement: Codable, Hashable, Sendable {
    public var featureLookback: Int
    public var horizonLoadMax: Int
    public var neededBars: Int
    public var alignUpToIndex: Int

    public init(featureLookback: Int, horizonLoadMax: Int, neededBars: Int, alignUpToIndex: Int) {
        self.featureLookback = max(0, featureLookback)
        self.horizonLoadMax = HorizonTools.clampHorizon(horizonLoadMax)
        self.neededBars = max(0, neededBars)
        self.alignUpToIndex = max(0, alignUpToIndex)
    }
}

public struct RuntimeFeaturePipelinePlan: Codable, Hashable, Sendable {
    public var requirement: RuntimeFeaturePipelineRequirement
    public var selectedHorizonMinutes: Int
    public var initStart: Int
    public var initEnd: Int
    public var onlineStart: Int
    public var onlineEnd: Int
    public var shadowStart: Int
    public var shadowEnd: Int
    public var shadowSamples: Int
    public var shadowEpochs: Int
    public var shadowEveryBars: Int
    public var shadowAllowed: Bool
    public var runShadow: Bool
    public var maxValidIndex: Int
    public var precomputeEnd: Int

    public var haveInitWindow: Bool { initEnd >= initStart }
    public var haveOnlineWindow: Bool { onlineEnd >= onlineStart }
    public var haveShadowWindow: Bool { shadowEnd >= shadowStart }

    public init(
        requirement: RuntimeFeaturePipelineRequirement,
        selectedHorizonMinutes: Int,
        initStart: Int,
        initEnd: Int,
        onlineStart: Int,
        onlineEnd: Int,
        shadowStart: Int,
        shadowEnd: Int,
        shadowSamples: Int,
        shadowEpochs: Int,
        shadowEveryBars: Int,
        shadowAllowed: Bool,
        runShadow: Bool,
        maxValidIndex: Int,
        precomputeEnd: Int
    ) {
        self.requirement = requirement
        self.selectedHorizonMinutes = HorizonTools.clampHorizon(selectedHorizonMinutes)
        self.initStart = initStart
        self.initEnd = initEnd
        self.onlineStart = onlineStart
        self.onlineEnd = onlineEnd
        self.shadowStart = shadowStart
        self.shadowEnd = shadowEnd
        self.shadowSamples = fxClampedInt(shadowSamples, lower: 8, upper: 200)
        self.shadowEpochs = fxClampedInt(shadowEpochs, lower: 1, upper: 3)
        self.shadowEveryBars = max(1, shadowEveryBars)
        self.shadowAllowed = shadowAllowed
        self.runShadow = runShadow
        self.maxValidIndex = maxValidIndex
        self.precomputeEnd = precomputeEnd
    }
}

public struct RuntimeDynamicContextInputs: Codable, Hashable, Sendable {
    public var utility: Double
    public var stability: Double
    public var lead: Double
    public var coverage: Double

    public init(utility: Double = 0.0, stability: Double = 0.0, lead: Double = 0.0, coverage: Double = 0.0) {
        self.utility = fxSafeFinite(utility)
        self.stability = fxSafeFinite(stability)
        self.lead = fxSafeFinite(lead)
        self.coverage = fxSafeFinite(coverage)
    }
}

public struct RuntimeFeaturePipelineContextState: Codable, Hashable, Sendable {
    public var strength: Double
    public var quality: Double

    public init(strength: Double = 0.0, quality: Double = 0.0) {
        self.strength = fxClamp(strength, 0.0, 4.0)
        self.quality = fxClamp(quality, -1.0, 2.0)
    }
}

public struct HorizonPolicyFeatureInputs: Codable, Hashable, Sendable {
    public var horizonMinutes: Int
    public var baseHorizonMinutes: Int
    public var expectedAbsMovePoints: Double
    public var minMovePoints: Double
    public var sampleTimeUTC: Int64
    public var currentVolPoints: Double
    public var priceCostPoints: Double
    public var regimeID: Int
    public var aiHint: Int
    public var contextStrength: Double
    public var contextQuality: Double
    public var modelReliabilityHint: Double
    public var regimeEdgePoints: Double
    public var modelEdgePoints: Double
    public var holdPenaltyPerMinute: Double
    public var configuredHorizons: [Int]

    public init(
        horizonMinutes: Int,
        baseHorizonMinutes: Int,
        expectedAbsMovePoints: Double,
        minMovePoints: Double,
        sampleTimeUTC: Int64,
        currentVolPoints: Double,
        priceCostPoints: Double = 0.0,
        regimeID: Int,
        aiHint: Int = -1,
        contextStrength: Double = 0.0,
        contextQuality: Double = 0.0,
        modelReliabilityHint: Double = 0.50,
        regimeEdgePoints: Double = 0.0,
        modelEdgePoints: Double = 0.0,
        holdPenaltyPerMinute: Double = 0.0,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) {
        self.horizonMinutes = HorizonTools.clampHorizon(horizonMinutes)
        self.baseHorizonMinutes = HorizonTools.clampHorizon(baseHorizonMinutes)
        self.expectedAbsMovePoints = max(0.0, fxSafeFinite(expectedAbsMovePoints))
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.sampleTimeUTC = max(0, sampleTimeUTC)
        self.currentVolPoints = max(0.0, fxSafeFinite(currentVolPoints))
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.regimeID = Int(fxClamp(Double(regimeID), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))
        self.aiHint = aiHint
        self.contextStrength = fxClamp(contextStrength, 0.0, 4.0)
        self.contextQuality = fxClamp(contextQuality, -1.0, 2.0)
        self.modelReliabilityHint = fxClamp(modelReliabilityHint, 0.0, 1.0)
        self.regimeEdgePoints = fxSafeFinite(regimeEdgePoints)
        self.modelEdgePoints = fxSafeFinite(modelEdgePoints)
        self.holdPenaltyPerMinute = fxClamp(holdPenaltyPerMinute, 0.0, 0.02)
        self.configuredHorizons = configuredHorizons.map(HorizonTools.clampHorizon)
    }
}

public struct RuntimeRoutedHorizonCandidate: Codable, Hashable, Sendable {
    public var horizonMinutes: Int
    public var expectedAbsMovePoints: Double
    public var regimeEdgeReady: Bool
    public var regimeEdgePoints: Double
    public var regimeObservations: Int
    public var regimeTotalObservations: Double
    public var modelEdgeReady: Bool
    public var modelEdgePoints: Double
    public var modelObservations: Int
    public var horizonPolicyReady: Bool
    public var horizonPolicyValue: Double
    public var oofHorizonPriorScore: Double

    public init(
        horizonMinutes: Int,
        expectedAbsMovePoints: Double,
        regimeEdgeReady: Bool = false,
        regimeEdgePoints: Double = 0.0,
        regimeObservations: Int = 0,
        regimeTotalObservations: Double = 0.0,
        modelEdgeReady: Bool = false,
        modelEdgePoints: Double = 0.0,
        modelObservations: Int = 0,
        horizonPolicyReady: Bool = false,
        horizonPolicyValue: Double = 0.0,
        oofHorizonPriorScore: Double = 0.0
    ) {
        self.horizonMinutes = HorizonTools.clampHorizon(horizonMinutes)
        self.expectedAbsMovePoints = max(0.0, fxSafeFinite(expectedAbsMovePoints))
        self.regimeEdgeReady = regimeEdgeReady
        self.regimeEdgePoints = fxSafeFinite(regimeEdgePoints)
        self.regimeObservations = max(0, regimeObservations)
        self.regimeTotalObservations = max(0.0, fxSafeFinite(regimeTotalObservations))
        self.modelEdgeReady = modelEdgeReady
        self.modelEdgePoints = fxSafeFinite(modelEdgePoints)
        self.modelObservations = max(0, modelObservations)
        self.horizonPolicyReady = horizonPolicyReady
        self.horizonPolicyValue = fxSafeFinite(horizonPolicyValue)
        self.oofHorizonPriorScore = fxSafeFinite(oofHorizonPriorScore)
    }
}

public struct RuntimeRoutedHorizonSelection: Codable, Hashable, Sendable {
    public var horizonMinutes: Int
    public var score: Double
    public var usedFallback: Bool

    public init(horizonMinutes: Int, score: Double = 0.0, usedFallback: Bool = false) {
        self.horizonMinutes = HorizonTools.clampHorizon(horizonMinutes)
        self.score = fxSafeFinite(score)
        self.usedFallback = usedFallback
    }
}

public struct RuntimeTransferInputState: Codable, Hashable, Sendable {
    public var rawX: [Double]
    public var sharedWindow: [[Double]]

    public init(rawX: [Double], sharedWindow: [[Double]]) {
        self.rawX = TrainingSampleTools.sanitizeModelInput(rawX)
        self.sharedWindow = sharedWindow.map { TrainingSampleTools.sanitizeModelInput($0) }
    }

    public var windowSize: Int {
        sharedWindow.count
    }
}

public enum RuntimeFeaturePipelineTools {
    public static let featureLookback = 10

    public static func bootstrapRequirement(
        baseSamples: Int,
        onlineSamples: Int,
        baseHorizonMinutes: Int,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) -> RuntimeFeaturePipelineRequirement {
        let horizonLoadMax = HorizonTools.maxConfiguredHorizon(
            configuredHorizons: configuredHorizons,
            fallbackHorizon: baseHorizonMinutes
        )
        var needed = max(max(baseSamples, onlineSamples), 0) + horizonLoadMax + featureLookback
        if needed < 128 {
            needed = 128
        }
        return RuntimeFeaturePipelineRequirement(
            featureLookback: featureLookback,
            horizonLoadMax: horizonLoadMax,
            neededBars: needed,
            alignUpToIndex: max(needed - 1, 0)
        )
    }

    public static func buildPlan(
        baseSamples: Int,
        onlineSamples: Int,
        baseHorizonMinutes: Int,
        selectedHorizonMinutes: Int,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons,
        shadowSamples: Int,
        shadowEpochs: Int,
        shadowEveryBars: Int,
        ensembleMode: Bool,
        shadowAllowed: Bool,
        signalSequence: Int
    ) -> RuntimeFeaturePipelinePlan {
        let requirement = bootstrapRequirement(
            baseSamples: baseSamples,
            onlineSamples: onlineSamples,
            baseHorizonMinutes: baseHorizonMinutes,
            configuredHorizons: configuredHorizons
        )
        let horizon = HorizonTools.clampHorizon(selectedHorizonMinutes)
        let sanitizedShadowSamples = fxClampedInt(shadowSamples, lower: 8, upper: 200)
        let sanitizedShadowEpochs = fxClampedInt(shadowEpochs, lower: 1, upper: 3)
        let sanitizedShadowEvery = max(1, shadowEveryBars)
        let runShadow = ensembleMode &&
            shadowAllowed &&
            NormalizationMetaSupportTools.isShadowBar(cadenceBars: sanitizedShadowEvery, barSequence: signalSequence)

        let maxValid = requirement.neededBars - requirement.featureLookback - 1
        let initStart = horizon
        let onlineStart = horizon
        let shadowStart = horizon
        let initEnd = min(horizon + max(baseSamples, 0) - 1, maxValid)
        let onlineEnd = min(horizon + max(onlineSamples, 0) - 1, maxValid)
        let shadowEnd = min(horizon + sanitizedShadowSamples - 1, maxValid)

        var precomputeEnd = -1
        if initEnd >= initStart {
            precomputeEnd = initEnd
        }
        if onlineEnd >= onlineStart, onlineEnd > precomputeEnd {
            precomputeEnd = onlineEnd
        }
        if runShadow, shadowEnd >= shadowStart, shadowEnd > precomputeEnd {
            precomputeEnd = shadowEnd
        }

        return RuntimeFeaturePipelinePlan(
            requirement: requirement,
            selectedHorizonMinutes: horizon,
            initStart: initStart,
            initEnd: initEnd,
            onlineStart: onlineStart,
            onlineEnd: onlineEnd,
            shadowStart: shadowStart,
            shadowEnd: shadowEnd,
            shadowSamples: sanitizedShadowSamples,
            shadowEpochs: sanitizedShadowEpochs,
            shadowEveryBars: sanitizedShadowEvery,
            shadowAllowed: shadowAllowed,
            runShadow: runShadow,
            maxValidIndex: maxValid,
            precomputeEnd: precomputeEnd
        )
    }

    public static func contextState(
        contextMean: Double,
        contextStandardDeviation: Double,
        contextUpRatio: Double,
        dynamicContext: RuntimeDynamicContextInputs
    ) -> RuntimeFeaturePipelineContextState {
        let strength = fxClamp(
            abs(fxSafeFinite(contextMean)) +
                fxSafeFinite(contextStandardDeviation) +
                abs(fxSafeFinite(contextUpRatio, fallback: 0.5) - 0.5),
            0.0,
            4.0
        )
        let quality = fxClamp(
            0.45 * dynamicContext.utility +
                0.25 * dynamicContext.stability +
                0.20 * dynamicContext.lead +
                0.10 * dynamicContext.coverage,
            -1.0,
            2.0
        )
        return RuntimeFeaturePipelineContextState(strength: strength, quality: quality)
    }

    public static func selectRoutedHorizon(
        candidates: [RuntimeRoutedHorizonCandidate],
        fallbackHorizonMinutes: Int,
        multiHorizonEnabled: Bool,
        minMovePoints: Double,
        holdPenaltyPerMinute: Double
    ) -> RuntimeRoutedHorizonSelection {
        let baseHorizon = HorizonTools.clampHorizon(fallbackHorizonMinutes)
        guard multiHorizonEnabled, !candidates.isEmpty else {
            return RuntimeRoutedHorizonSelection(horizonMinutes: baseHorizon, usedFallback: true)
        }

        var bestScore = -Double.greatestFiniteMagnitude
        var bestHorizon = baseHorizon
        let holdPenalty = fxClamp(holdPenaltyPerMinute, 0.0, 0.02)
        let minimumMove = max(minMovePoints, 0.50)

        for candidate in candidates {
            let horizon = HorizonTools.clampHorizon(candidate.horizonMinutes)
            let expectedAbs = max(0.0, candidate.expectedAbsMovePoints)
            guard expectedAbs > 0.0 else { continue }

            let net = expectedAbs - max(0.0, minMovePoints)
            var score = (net / sqrt(Double(horizon))) - (holdPenalty * Double(horizon))
            if candidate.regimeEdgeReady {
                let totalObservations = max(candidate.regimeTotalObservations, 1.0)
                let ucb = candidate.regimeEdgePoints +
                    0.35 * sqrt(log(1.0 + totalObservations) / (1.0 + Double(candidate.regimeObservations)))
                score += 0.25 * (ucb / minimumMove)
            }
            if candidate.modelEdgeReady {
                let modelUCB = candidate.modelEdgePoints + 0.20 / sqrt(1.0 + Double(candidate.modelObservations))
                score += 0.15 * (modelUCB / minimumMove)
            }
            if candidate.horizonPolicyReady {
                score += 0.35 * candidate.horizonPolicyValue
            }
            score += candidate.oofHorizonPriorScore

            if score > bestScore {
                bestScore = score
                bestHorizon = horizon
            }
        }

        return RuntimeRoutedHorizonSelection(
            horizonMinutes: bestHorizon,
            score: bestScore == -Double.greatestFiniteMagnitude ? 0.0 : bestScore,
            usedFallback: bestScore == -Double.greatestFiniteMagnitude
        )
    }

    public static func buildHorizonPolicyFeatures(_ inputs: HorizonPolicyFeatureInputs) -> [Double] {
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.horizonPolicyFeatures)
        let horizon = HorizonTools.clampHorizon(inputs.horizonMinutes)
        let baseHorizon = HorizonTools.clampHorizon(inputs.baseHorizonMinutes)
        let minimumMove = max(inputs.minMovePoints, 0.50)
        let contextStrength = fxClamp(inputs.contextStrength, 0.0, 4.0)
        let contextQuality = fxClamp(inputs.contextQuality, -1.0, 2.0)
        let reliabilityHint = fxClamp(inputs.modelReliabilityHint, 0.0, 1.0)
        let sessionBucket = PluginContractTools.deriveSessionBucket(timestampUTC: inputs.sampleTimeUTC)
        let currentVolPoints = max(inputs.currentVolPoints, 0.0)
        let netEdge = inputs.expectedAbsMovePoints - inputs.minMovePoints
        let regimeEdge = inputs.regimeEdgePoints / minimumMove
        let modelEdge = inputs.aiHint >= 0 ? inputs.modelEdgePoints / minimumMove : 0.0
        let horizonSlot = HorizonTools.horizonSlot(
            horizonMinutes: horizon,
            configuredHorizons: inputs.configuredHorizons
        )
        let date = Date(timeIntervalSince1970: TimeInterval(max(0, inputs.sampleTimeUTC)))
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let hour = calendar.component(.hour, from: date)
        let minute = calendar.component(.minute, from: date)
        let dayOfWeek = calendar.component(.weekday, from: date) - 1
        let priceCostPoints = max(0.0, inputs.priceCostPoints)

        features[0] = 1.0
        features[1] = fxClamp((inputs.expectedAbsMovePoints - inputs.minMovePoints) / minimumMove, -4.0, 6.0) / 4.0
        features[2] = fxClamp(inputs.expectedAbsMovePoints / minimumMove, 0.0, 8.0) / 4.0
        features[3] = 1.0 / sqrt(Double(max(horizon, 1)))
        features[4] = -inputs.holdPenaltyPerMinute * Double(horizon)
        features[5] = fxClamp(regimeEdge, -3.0, 3.0) / 3.0
        features[6] = fxClamp(modelEdge, -3.0, 3.0) / 3.0
        features[7] = fxClamp(currentVolPoints, 0.0, 50.0) / 25.0
        features[8] = fxClamp(priceCostPoints / minimumMove, 0.0, 2.0) - 0.5
        features[9] = (Double(hour) - 11.5) / 11.5
        features[10] = (Double(minute) - 29.5) / 29.5
        features[11] = fxClamp((Double(horizon) - Double(baseHorizon)) / Double(max(baseHorizon, 1)), -2.0, 2.0) / 2.0
        features[12] = fxClamp((inputs.expectedAbsMovePoints - inputs.minMovePoints) / Double(max(horizon, 1)), -2.0, 6.0) / 4.0
        features[13] = fxClamp(inputs.expectedAbsMovePoints / sqrt(Double(max(horizon, 1))), 0.0, 20.0) / 10.0
        features[14] = fxClamp((Double(horizon) / Double(max(baseHorizon, 1))) - 1.0, -2.0, 4.0) / 2.0
        features[15] = fxClamp(Double(inputs.regimeID) / Double(max(FXDataEngineConstants.pluginRegimeBuckets - 1, 1)), 0.0, 1.0) - 0.5
        features[16] = fxClamp(contextStrength / 2.0, 0.0, 1.5) - 0.25
        features[17] = fxClamp(contextQuality, -1.0, 2.0) / 2.0
        features[18] = reliabilityHint - 0.5
        features[19] = fxClamp(((inputs.expectedAbsMovePoints - inputs.minMovePoints) / minimumMove) * (0.5 + contextQuality), -6.0, 6.0) / 6.0
        features[20] = fxClamp(currentVolPoints * (0.25 + contextStrength), 0.0, 80.0) / 40.0
        features[21] = fxClamp(priceCostPoints / max(currentVolPoints, 1.0), 0.0, 4.0) / 2.0 - 0.5
        features[22] = (Double(sessionBucket) / Double(max(FXDataEngineConstants.pluginSessionBuckets - 1, 1))) - 0.5
        features[23] = (Double(horizonSlot) / Double(max(RuntimeArtifactConstants.maxHorizons - 1, 1))) - 0.5
        features[24] = fxClamp((netEdge / minimumMove) * reliabilityHint, -6.0, 6.0) / 6.0
        features[25] = fxClamp(inputs.expectedAbsMovePoints / max(currentVolPoints * sqrt(Double(max(horizon, 1))), 1.0), 0.0, 6.0) / 3.0 - 0.5
        features[26] = fxClamp(inputs.minMovePoints / max(priceCostPoints, 0.10), 0.0, 6.0) / 3.0 - 0.5
        features[27] = fxClamp(priceCostPoints / max(inputs.expectedAbsMovePoints, minimumMove), 0.0, 2.0) - 0.5
        features[28] = fxClamp((contextStrength * (contextQuality + 1.0)) / 4.0, 0.0, 2.0) - 0.5
        features[29] = fxClamp(reliabilityHint * (1.0 + fxClamp(regimeEdge + modelEdge, -2.0, 2.0)), 0.0, 2.0) - 0.5
        features[30] = (Double(dayOfWeek) - 2.5) / 2.5
        features[31] = fxClamp((inputs.holdPenaltyPerMinute * Double(horizon)) / max(abs(netEdge / minimumMove), 0.25), 0.0, 2.0) - 0.5
        features[32] = fxClamp(netEdge / max(inputs.expectedAbsMovePoints, minimumMove), -1.0, 1.0)
        features[33] = fxClamp(contextQuality * reliabilityHint, -1.0, 2.0) / 2.0
        features[34] = fxClamp(contextStrength * reliabilityHint, 0.0, 4.0) / 2.0 - 0.5
        features[35] = fxClamp(regimeEdge * max(contextQuality, 0.0), -4.0, 4.0) / 4.0
        features[36] = fxClamp(modelEdge * max(contextQuality, 0.0), -4.0, 4.0) / 4.0
        features[37] = fxClamp((inputs.expectedAbsMovePoints / Double(max(horizon, 1))) / max(currentVolPoints, 0.50), 0.0, 6.0) / 3.0 - 0.5
        features[38] = fxClamp(Double(sessionBucket) / Double(max(FXDataEngineConstants.pluginSessionBuckets - 1, 1)), 0.0, 1.0) * reliabilityHint - 0.5
        features[39] = fxClamp((abs(netEdge) / minimumMove) * (0.50 + abs(contextQuality)), 0.0, 8.0) / 4.0 - 0.5
        features[40] = fxClamp(abs(regimeEdge), 0.0, 4.0) / 2.0 - 0.5
        features[41] = fxClamp(abs(modelEdge), 0.0, 4.0) / 2.0 - 0.5
        features[42] = fxClamp(contextStrength * max(contextQuality + 1.0, 0.0), 0.0, 8.0) / 4.0 - 0.5
        features[43] = fxClamp((netEdge / minimumMove) * max(reliabilityHint, 0.05) / sqrt(Double(max(horizon, 1))), -4.0, 4.0) / 4.0
        features[44] = fxClamp((inputs.holdPenaltyPerMinute * Double(horizon)) * max(currentVolPoints, 0.25), 0.0, 6.0) / 3.0 - 0.5
        features[45] = fxClamp((priceCostPoints / max(inputs.expectedAbsMovePoints + minimumMove, minimumMove)) * (0.50 + reliabilityHint), 0.0, 2.0) - 0.5
        features[46] = fxClamp((Double(sessionBucket + 1) / Double(max(FXDataEngineConstants.pluginSessionBuckets, 1))) * (contextStrength + 0.50), 0.0, 3.0) / 1.5 - 0.5
        features[47] = fxClamp(((Double(baseHorizon) / Double(max(horizon, 1))) - 1.0) * (0.50 + max(contextQuality, 0.0)), -3.0, 3.0) / 3.0
        return features.map { fxSafeFinite($0) }
    }
}

public enum RuntimeTransferTools {
    public static func modelInputVector(features: [Double]) -> [Double] {
        var rawX = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        rawX[0] = 1.0
        for index in 0..<min(features.count, FXDataEngineConstants.aiFeatures) {
            rawX[index + 1] = fxSafeFinite(features[index])
        }
        return rawX
    }

    public static func currentSharedWindow(
        currentRawX: [Double],
        samples: [RuntimeArtifactPreparedSample],
        span: Int
    ) -> [[Double]] {
        let sanitizedCurrent = TrainingSampleTools.sanitizeModelInput(currentRawX)
        let clampedSpan = max(1, min(span, FXDataEngineConstants.maxSequenceBars))
        var window = [sanitizedCurrent]
        guard clampedSpan > 1 else { return window }

        for sample in samples.dropFirst().prefix(clampedSpan - 1) where sample.valid {
            window.append(sample.x)
            if window.count >= clampedSpan {
                break
            }
        }
        return window
    }

    public static func currentSharedWindow(
        currentRawX: [Double],
        samples: [RuntimeArtifactPreparedSample],
        horizonMinutes: Int,
        symbol: String
    ) -> [[Double]] {
        let span = ModelContextTools.contextSequenceSpan(
            maxCap: 24,
            horizonMinutes: horizonMinutes,
            symbol: symbol,
            baseMin: 8
        )
        return currentSharedWindow(currentRawX: currentRawX, samples: samples, span: max(span, 1))
    }

    public static func currentTransferInput(
        features: [Double],
        samples: [RuntimeArtifactPreparedSample],
        horizonMinutes: Int,
        symbol: String
    ) -> RuntimeTransferInputState {
        let rawX = modelInputVector(features: features)
        return RuntimeTransferInputState(
            rawX: rawX,
            sharedWindow: currentSharedWindow(
                currentRawX: rawX,
                samples: samples,
                horizonMinutes: horizonMinutes,
                symbol: symbol
            )
        )
    }
}

private func fxClampedInt(_ value: Int, lower: Int, upper: Int) -> Int {
    Int(fxClamp(Double(value), Double(lower), Double(upper)))
}
