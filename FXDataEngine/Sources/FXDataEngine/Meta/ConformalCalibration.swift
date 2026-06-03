import Foundation

public enum ConformalScoreKind: Int, Codable, Sendable, CaseIterable {
    case classScore = 0
    case moveScore = 1
    case pathScore = 2
}

public struct ConformalPendingEntry: Codable, Hashable, Sendable {
    public var signalSequence: Int
    public var regimeID: Int
    public var horizonMinutes: Int
    public var classProbabilities: [Double]
    public var moveQ25Points: Double
    public var moveQ50Points: Double
    public var moveQ75Points: Double
    public var pathRisk: Double

    public init(
        signalSequence: Int = -1,
        regimeID: Int = 0,
        horizonMinutes: Int = 1,
        classProbabilities: [Double] = [0.0, 0.0, 1.0],
        moveQ25Points: Double = 0.0,
        moveQ50Points: Double = 0.0,
        moveQ75Points: Double = 0.0,
        pathRisk: Double = 0.5
    ) {
        self.signalSequence = signalSequence
        self.regimeID = Int(fxClamp(Double(regimeID), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))
        self.horizonMinutes = TrainingSampleTools.clampHorizon(horizonMinutes)
        self.classProbabilities = Self.sanitizedProbabilities(classProbabilities)
        self.moveQ25Points = max(0.0, fxSafeFinite(moveQ25Points))
        self.moveQ50Points = max(self.moveQ25Points, fxSafeFinite(moveQ50Points))
        self.moveQ75Points = max(self.moveQ50Points, fxSafeFinite(moveQ75Points))
        self.pathRisk = fxClamp(pathRisk, 0.0, 1.0)
    }

    public init(signalSequence: Int, regimeID: Int, horizonMinutes: Int, prediction: PredictionV4) {
        self.init(
            signalSequence: signalSequence,
            regimeID: regimeID,
            horizonMinutes: horizonMinutes,
            classProbabilities: prediction.classProbabilities,
            moveQ25Points: prediction.moveQ25Points,
            moveQ50Points: prediction.moveQ50Points,
            moveQ75Points: prediction.moveQ75Points,
            pathRisk: prediction.pathRisk
        )
    }

    public static func sanitizedProbabilities(_ probabilities: [Double]) -> [Double] {
        var output = [0.0, 0.0, 0.0]
        for index in 0..<min(probabilities.count, 3) {
            output[index] = max(0.0, fxSafeFinite(probabilities[index]))
        }
        return output.reduce(0.0, +) > 0.0 ? output : [0.0, 0.0, 1.0]
    }
}

public struct ConformalRealizedOutcome: Codable, Hashable, Sendable {
    public var signalSequence: Int
    public var labelClass: LabelClass?
    public var realizedMovePoints: Double
    public var mfePoints: Double
    public var maePoints: Double
    public var timeToHitFraction: Double
    public var pathFlags: SamplePathFlags
    public var minMovePoints: Double
    public var fillRisk: Double?

    public init(
        signalSequence: Int,
        labelClass: LabelClass? = nil,
        realizedMovePoints: Double,
        mfePoints: Double,
        maePoints: Double,
        timeToHitFraction: Double,
        pathFlags: SamplePathFlags = [],
        minMovePoints: Double,
        fillRisk: Double? = nil
    ) {
        self.signalSequence = signalSequence
        self.labelClass = labelClass
        self.realizedMovePoints = fxSafeFinite(realizedMovePoints)
        self.mfePoints = max(0.0, fxSafeFinite(mfePoints))
        self.maePoints = max(0.0, fxSafeFinite(maePoints))
        self.timeToHitFraction = fxClamp(timeToHitFraction, 0.0, 1.0)
        self.pathFlags = pathFlags
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.fillRisk = fillRisk.map { fxClamp($0, 0.0, 1.0) }
    }

    public var resolvedLabelClass: LabelClass {
        labelClass ?? (realizedMovePoints >= 0.0 ? .buy : .sell)
    }
}

public struct ConformalCalibrationPolicy: Codable, Hashable, Sendable {
    public var targetCoverage: Double
    public var minCalibrationCount: Int
    public var fallbackCutoff: Double

    public init(
        targetCoverage: Double = 0.90,
        minCalibrationCount: Int = 16,
        fallbackCutoff: Double = 0.35
    ) {
        self.targetCoverage = fxClamp(fxSafeFinite(targetCoverage, fallback: 0.90), 0.50, 0.995)
        self.minCalibrationCount = min(
            max(minCalibrationCount, 1),
            RuntimeArtifactConstants.conformalDepth
        )
        self.fallbackCutoff = fxClamp(fxSafeFinite(fallbackCutoff), 0.0, 6.0)
    }

    public var alpha: Double {
        1.0 - targetCoverage
    }
}

public struct ConformalCalibrationDiagnostics: Codable, Hashable, Sendable {
    public var scoreKind: ConformalScoreKind
    public var sampleCount: Int
    public var minCalibrationCount: Int
    public var targetCoverage: Double
    public var alpha: Double
    public var finiteSampleRank: Int
    public var cutoff: Double
    public var fallbackCutoff: Double
    public var fallbackUsed: Bool
    public var conservativeMaximumUsed: Bool

    public init(
        scoreKind: ConformalScoreKind,
        sampleCount: Int,
        minCalibrationCount: Int,
        targetCoverage: Double,
        alpha: Double,
        finiteSampleRank: Int,
        cutoff: Double,
        fallbackCutoff: Double,
        fallbackUsed: Bool,
        conservativeMaximumUsed: Bool
    ) {
        self.scoreKind = scoreKind
        self.sampleCount = max(0, sampleCount)
        self.minCalibrationCount = max(1, minCalibrationCount)
        self.targetCoverage = fxClamp(fxSafeFinite(targetCoverage, fallback: 0.90), 0.50, 0.995)
        self.alpha = fxClamp(fxSafeFinite(alpha, fallback: 0.10), 0.005, 0.50)
        self.finiteSampleRank = max(0, finiteSampleRank)
        self.cutoff = fxSafeFinite(cutoff, fallback: fallbackCutoff)
        self.fallbackCutoff = fxSafeFinite(fallbackCutoff)
        self.fallbackUsed = fallbackUsed
        self.conservativeMaximumUsed = conservativeMaximumUsed
    }

    public var sufficientSamples: Bool {
        sampleCount >= minCalibrationCount
    }
}

public struct ConformalPredictionSet: Codable, Hashable, Sendable {
    public var classes: [LabelClass]
    public var cutoff: Double
    public var diagnostics: ConformalCalibrationDiagnostics
    public var usedArgmaxFallback: Bool

    public init(
        classes: [LabelClass],
        cutoff: Double,
        diagnostics: ConformalCalibrationDiagnostics,
        usedArgmaxFallback: Bool = false
    ) {
        self.classes = classes.sorted { $0.rawValue < $1.rawValue }
        self.cutoff = fxClamp(fxSafeFinite(cutoff, fallback: diagnostics.cutoff), 0.0, 1.0)
        self.diagnostics = diagnostics
        self.usedArgmaxFallback = usedArgmaxFallback
    }

    public func contains(_ label: LabelClass) -> Bool {
        classes.contains(label)
    }
}

public struct ConformalMoveInterval: Codable, Hashable, Sendable {
    public var lowerPoints: Double
    public var medianPoints: Double
    public var upperPoints: Double
    public var cutoff: Double
    public var diagnostics: ConformalCalibrationDiagnostics

    public init(
        lowerPoints: Double,
        medianPoints: Double,
        upperPoints: Double,
        cutoff: Double,
        diagnostics: ConformalCalibrationDiagnostics
    ) {
        let lower = max(0.0, fxSafeFinite(lowerPoints))
        let median = max(lower, fxSafeFinite(medianPoints, fallback: lower))
        self.lowerPoints = lower
        self.medianPoints = median
        self.upperPoints = max(median, fxSafeFinite(upperPoints, fallback: median))
        self.cutoff = fxClamp(fxSafeFinite(cutoff, fallback: diagnostics.cutoff), 0.0, 6.0)
        self.diagnostics = diagnostics
    }

    public func containsAbsoluteMove(_ movePoints: Double) -> Bool {
        let move = abs(fxSafeFinite(movePoints))
        return move >= lowerPoints && move <= upperPoints
    }
}

public struct ConformalCalibrationAIState: Codable, Hashable, Sendable {
    public var counts: [Int]
    public var heads: [Int]
    public var classScores: [Double]
    public var moveScores: [Double]
    public var pathScores: [Double]
    public var pendingHead: Int
    public var pendingTail: Int
    public var pendingEntries: [ConformalPendingEntry]

    public init(
        counts: [Int] = [],
        heads: [Int] = [],
        classScores: [Double] = [],
        moveScores: [Double] = [],
        pathScores: [Double] = [],
        pendingHead: Int = 0,
        pendingTail: Int = 0,
        pendingEntries: [ConformalPendingEntry] = []
    ) {
        let cellCount = Self.scoreCellCount
        let scoreCount = cellCount * RuntimeArtifactConstants.conformalDepth
        self.counts = Self.normalizedIntArray(
            counts,
            count: cellCount,
            defaultValue: 0,
            lower: 0,
            upper: RuntimeArtifactConstants.conformalDepth
        )
        self.heads = Self.normalizedIntArray(
            heads,
            count: cellCount,
            defaultValue: 0,
            lower: 0,
            upper: RuntimeArtifactConstants.conformalDepth - 1
        )
        self.classScores = Self.normalizedScoreArray(classScores, count: scoreCount, defaultValue: 0.35, upper: 1.0)
        self.moveScores = Self.normalizedScoreArray(moveScores, count: scoreCount, defaultValue: 0.20, upper: 6.0)
        self.pathScores = Self.normalizedScoreArray(pathScores, count: scoreCount, defaultValue: 0.10, upper: 1.0)
        let pendingCapacity = RuntimeArtifactConstants.reliabilityPendingCapacity
        self.pendingHead = Int(fxClamp(Double(pendingHead), 0.0, Double(max(pendingCapacity - 1, 0))))
        self.pendingTail = Int(fxClamp(Double(pendingTail), 0.0, Double(max(pendingCapacity - 1, 0))))
        self.pendingEntries = Array(pendingEntries.prefix(pendingCapacity))
        if self.pendingEntries.count < pendingCapacity {
            self.pendingEntries.append(contentsOf: Array(
                repeating: ConformalPendingEntry(),
                count: pendingCapacity - self.pendingEntries.count
            ))
        }
    }

    public static var scoreCellCount: Int {
        FXDataEngineConstants.pluginRegimeBuckets * RuntimeArtifactConstants.maxHorizons
    }

    public static func cellIndex(regimeID: Int, horizonSlot: Int) -> Int {
        let regime = Int(fxClamp(Double(regimeID), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))
        let slot = Int(fxClamp(Double(horizonSlot), 0.0, Double(RuntimeArtifactConstants.maxHorizons - 1)))
        return regime * RuntimeArtifactConstants.maxHorizons + slot
    }

    public static func scoreOffset(cellIndex: Int, depthIndex: Int) -> Int {
        cellIndex * RuntimeArtifactConstants.conformalDepth + depthIndex
    }

    private static func normalizedIntArray(
        _ values: [Int],
        count: Int,
        defaultValue: Int,
        lower: Int,
        upper: Int
    ) -> [Int] {
        var output = Array(repeating: defaultValue, count: count)
        let upperBound = max(lower, upper)
        for index in 0..<min(values.count, count) {
            output[index] = min(max(values[index], lower), upperBound)
        }
        return output
    }

    private static func normalizedScoreArray(
        _ values: [Double],
        count: Int,
        defaultValue: Double,
        upper: Double
    ) -> [Double] {
        var output = Array(repeating: defaultValue, count: count)
        for index in 0..<min(values.count, count) {
            output[index] = fxClamp(values[index], 0.0, upper)
        }
        return output
    }
}

public struct ConformalCalibrationState: Codable, Hashable, Sendable {
    public var aiStates: [ConformalCalibrationAIState]

    public init(aiStates: [ConformalCalibrationAIState] = []) {
        self.aiStates = Array(aiStates.prefix(FXDataEngineConstants.aiCount))
        if self.aiStates.count < FXDataEngineConstants.aiCount {
            self.aiStates.append(contentsOf: Array(
                repeating: ConformalCalibrationAIState(),
                count: FXDataEngineConstants.aiCount - self.aiStates.count
            ))
        }
    }

    public mutating func reset() {
        aiStates = Array(repeating: ConformalCalibrationAIState(), count: FXDataEngineConstants.aiCount)
    }

    public static func finiteSampleRank(sampleCount: Int, targetCoverage: Double) -> Int {
        let count = max(0, sampleCount)
        guard count > 0 else { return 0 }
        let coverage = fxClamp(fxSafeFinite(targetCoverage, fallback: 0.90), 0.50, 0.995)
        return max(1, Int(ceil(Double(count + 1) * coverage)))
    }

    public func scoreSamples(
        aiIndex: Int,
        regimeID: Int,
        horizonSlot: Int,
        scoreKind: ConformalScoreKind
    ) -> [Double] {
        guard aiIndex >= 0, aiIndex < aiStates.count else { return [] }
        guard regimeID >= 0, regimeID < FXDataEngineConstants.pluginRegimeBuckets else { return [] }
        guard horizonSlot >= 0, horizonSlot < RuntimeArtifactConstants.maxHorizons else { return [] }

        let state = aiStates[aiIndex]
        let cell = ConformalCalibrationAIState.cellIndex(regimeID: regimeID, horizonSlot: horizonSlot)
        let count = min(max(state.counts[cell], 0), RuntimeArtifactConstants.conformalDepth)
        guard count > 0 else { return [] }

        let upperBound = Self.scoreUpperBound(scoreKind)
        let scores = scoreArray(for: state, kind: scoreKind)
        let base = ConformalCalibrationAIState.scoreOffset(cellIndex: cell, depthIndex: 0)
        return scores[base..<(base + count)].map { fxClamp($0, 0.0, upperBound) }
    }

    public func calibrationDiagnostics(
        aiIndex: Int,
        regimeID: Int,
        horizonSlot: Int,
        scoreKind: ConformalScoreKind,
        policy: ConformalCalibrationPolicy = ConformalCalibrationPolicy()
    ) -> ConformalCalibrationDiagnostics {
        let upperBound = Self.scoreUpperBound(scoreKind)
        let fallback = fxClamp(policy.fallbackCutoff, 0.0, upperBound)
        let scores = scoreSamples(
            aiIndex: aiIndex,
            regimeID: regimeID,
            horizonSlot: horizonSlot,
            scoreKind: scoreKind
        )
        let rank = Self.finiteSampleRank(sampleCount: scores.count, targetCoverage: policy.targetCoverage)
        guard scores.count >= policy.minCalibrationCount else {
            return ConformalCalibrationDiagnostics(
                scoreKind: scoreKind,
                sampleCount: scores.count,
                minCalibrationCount: policy.minCalibrationCount,
                targetCoverage: policy.targetCoverage,
                alpha: policy.alpha,
                finiteSampleRank: rank,
                cutoff: fallback,
                fallbackCutoff: fallback,
                fallbackUsed: true,
                conservativeMaximumUsed: false
            )
        }

        let cutoff: Double
        let conservativeMaximumUsed: Bool
        if rank > scores.count {
            cutoff = upperBound
            conservativeMaximumUsed = true
        } else {
            cutoff = scores.sorted()[max(rank - 1, 0)]
            conservativeMaximumUsed = false
        }

        return ConformalCalibrationDiagnostics(
            scoreKind: scoreKind,
            sampleCount: scores.count,
            minCalibrationCount: policy.minCalibrationCount,
            targetCoverage: policy.targetCoverage,
            alpha: policy.alpha,
            finiteSampleRank: rank,
            cutoff: cutoff,
            fallbackCutoff: fallback,
            fallbackUsed: false,
            conservativeMaximumUsed: conservativeMaximumUsed
        )
    }

    public func splitConformalCutoff(
        aiIndex: Int,
        regimeID: Int,
        horizonSlot: Int,
        scoreKind: ConformalScoreKind,
        policy: ConformalCalibrationPolicy = ConformalCalibrationPolicy()
    ) -> Double {
        calibrationDiagnostics(
            aiIndex: aiIndex,
            regimeID: regimeID,
            horizonSlot: horizonSlot,
            scoreKind: scoreKind,
            policy: policy
        ).cutoff
    }

    public func predictionSet(
        aiIndex: Int,
        regimeID: Int,
        horizonMinutes: Int,
        probabilities: [Double],
        policy: ConformalCalibrationPolicy = ConformalCalibrationPolicy()
    ) -> ConformalPredictionSet {
        let horizonSlot = TrainingSampleTools.horizonSlot(horizonMinutes: horizonMinutes)
        let regime = Int(fxClamp(Double(regimeID), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))
        let diagnostics = calibrationDiagnostics(
            aiIndex: aiIndex,
            regimeID: regime,
            horizonSlot: horizonSlot,
            scoreKind: .classScore,
            policy: policy
        )
        let normalized = PluginContextRuntimeTools.normalizeClassDistribution(probabilities)
        var labels = LabelClass.allCases.filter { label in
            let index = label.rawValue
            return index < normalized.count && 1.0 - normalized[index] <= diagnostics.cutoff + 1.0e-12
        }
        var usedArgmaxFallback = false
        if labels.isEmpty,
           let bestIndex = normalized.indices.max(by: { normalized[$0] < normalized[$1] }),
           let bestLabel = LabelClass(rawValue: bestIndex) {
            labels = [bestLabel]
            usedArgmaxFallback = true
        }
        return ConformalPredictionSet(
            classes: labels,
            cutoff: diagnostics.cutoff,
            diagnostics: diagnostics,
            usedArgmaxFallback: usedArgmaxFallback
        )
    }

    public func moveInterval(
        aiIndex: Int,
        regimeID: Int,
        horizonMinutes: Int,
        minMovePoints: Double,
        prediction: PredictionV4,
        policy: ConformalCalibrationPolicy = ConformalCalibrationPolicy(
            targetCoverage: 0.90,
            minCalibrationCount: 16,
            fallbackCutoff: 0.20
        )
    ) -> ConformalMoveInterval {
        let horizonSlot = TrainingSampleTools.horizonSlot(horizonMinutes: horizonMinutes)
        let regime = Int(fxClamp(Double(regimeID), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))
        let diagnostics = calibrationDiagnostics(
            aiIndex: aiIndex,
            regimeID: regime,
            horizonSlot: horizonSlot,
            scoreKind: .moveScore,
            policy: policy
        )
        let minimumMove = max(minMovePoints, 0.10)
        let rawQ25 = max(0.0, fxSafeFinite(prediction.moveQ25Points))
        let rawQ50 = max(rawQ25, fxSafeFinite(prediction.moveQ50Points, fallback: rawQ25))
        let rawQ75 = max(rawQ50, fxSafeFinite(prediction.moveQ75Points, fallback: rawQ50))
        let moveWidth = max(rawQ75 - rawQ25, max(minimumMove, 0.25))
        let padding = diagnostics.cutoff * max(0.50 * moveWidth, minimumMove)
        return ConformalMoveInterval(
            lowerPoints: max(0.0, rawQ25 - 0.50 * padding),
            medianPoints: rawQ50,
            upperPoints: rawQ75 + 0.50 * padding,
            cutoff: diagnostics.cutoff,
            diagnostics: diagnostics
        )
    }

    public func quantile(
        aiIndex: Int,
        regimeID: Int,
        horizonSlot: Int,
        scoreKind: ConformalScoreKind,
        fallback: Double
    ) -> Double {
        guard aiIndex >= 0, aiIndex < aiStates.count else { return fallback }
        guard regimeID >= 0, regimeID < FXDataEngineConstants.pluginRegimeBuckets else { return fallback }
        guard horizonSlot >= 0, horizonSlot < RuntimeArtifactConstants.maxHorizons else { return fallback }

        let state = aiStates[aiIndex]
        let cell = ConformalCalibrationAIState.cellIndex(regimeID: regimeID, horizonSlot: horizonSlot)
        let count = min(max(state.counts[cell], 0), RuntimeArtifactConstants.conformalDepth)
        guard count > 0 else { return fallback }

        let scores = scoreArray(for: state, kind: scoreKind)
        let base = ConformalCalibrationAIState.scoreOffset(cellIndex: cell, depthIndex: 0)
        let sorted = Array(scores[base..<(base + count)]).sorted()
        let quantileIndex = min(max(Int(floor(0.90 * Double(count - 1))), 0), count - 1)
        return sorted[quantileIndex]
    }

    @discardableResult
    public mutating func pushScore(
        aiIndex: Int,
        regimeID: Int,
        horizonSlot: Int,
        classScore: Double,
        moveScore: Double,
        pathScore: Double
    ) -> Bool {
        guard aiIndex >= 0, aiIndex < aiStates.count else { return false }
        guard horizonSlot >= 0, horizonSlot < RuntimeArtifactConstants.maxHorizons else { return false }
        let regime = Int(fxClamp(Double(regimeID), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))
        let cell = ConformalCalibrationAIState.cellIndex(regimeID: regime, horizonSlot: horizonSlot)
        var state = aiStates[aiIndex]
        var head = state.heads[cell]
        if head < 0 || head >= RuntimeArtifactConstants.conformalDepth {
            head = 0
        }
        let offset = ConformalCalibrationAIState.scoreOffset(cellIndex: cell, depthIndex: head)
        state.classScores[offset] = fxClamp(classScore, 0.0, 1.0)
        state.moveScores[offset] = fxClamp(moveScore, 0.0, 6.0)
        state.pathScores[offset] = fxClamp(pathScore, 0.0, 1.0)
        head += 1
        if head >= RuntimeArtifactConstants.conformalDepth {
            head = 0
        }
        state.heads[cell] = head
        if state.counts[cell] < RuntimeArtifactConstants.conformalDepth {
            state.counts[cell] += 1
        }
        aiStates[aiIndex] = state
        return true
    }

    @discardableResult
    public mutating func enqueuePending(
        aiIndex: Int,
        signalSequence: Int,
        regimeID: Int,
        horizonMinutes: Int,
        prediction: PredictionV4
    ) -> Bool {
        guard aiIndex >= 0, aiIndex < aiStates.count else { return false }
        guard signalSequence >= 0 else { return false }
        var state = aiStates[aiIndex]
        let capacity = RuntimeArtifactConstants.reliabilityPendingCapacity
        guard capacity > 0 else { return false }

        let entry = ConformalPendingEntry(
            signalSequence: signalSequence,
            regimeID: regimeID,
            horizonMinutes: horizonMinutes,
            prediction: prediction
        )

        let previous = state.pendingTail == 0 ? capacity - 1 : state.pendingTail - 1
        if state.pendingHead != state.pendingTail,
           state.pendingEntries[previous].signalSequence == signalSequence {
            state.pendingEntries[previous] = entry
            aiStates[aiIndex] = state
            return true
        }

        state.pendingEntries[state.pendingTail] = entry
        var nextTail = state.pendingTail + 1
        if nextTail >= capacity {
            nextTail = 0
        }
        if nextTail == state.pendingHead {
            state.pendingHead += 1
            if state.pendingHead >= capacity {
                state.pendingHead = 0
            }
        }
        state.pendingTail = nextTail
        aiStates[aiIndex] = state
        return true
    }

    @discardableResult
    public mutating func updateFromPending(
        aiIndex: Int,
        outcome: ConformalRealizedOutcome
    ) -> Bool {
        guard aiIndex >= 0, aiIndex < aiStates.count else { return false }
        var state = aiStates[aiIndex]
        let pending = pendingEntries(in: state)
        guard !pending.isEmpty else { return false }

        var kept: [ConformalPendingEntry] = []
        kept.reserveCapacity(pending.count)
        var matched: ConformalPendingEntry?
        for entry in pending {
            if matched == nil && entry.signalSequence == outcome.signalSequence {
                matched = entry
            } else {
                kept.append(entry)
            }
        }

        state = resetPendingQueue(for: state, keeping: kept)
        aiStates[aiIndex] = state

        guard let matched else { return false }
        let classIndex = outcome.resolvedLabelClass.rawValue
        let probabilities = ConformalPendingEntry.sanitizedProbabilities(matched.classProbabilities)
        let pTrue = probabilities[classIndex]
        let minMove = max(outcome.minMovePoints, 0.10)
        let width = max(matched.moveQ75Points - matched.moveQ25Points, max(minMove, 0.25))
        let moveScore = abs(abs(outcome.realizedMovePoints) - max(matched.moveQ50Points, 0.0)) / width
        let pathActual = TrainingSampleTools.pathRisk(
            mfePoints: outcome.mfePoints,
            maePoints: outcome.maePoints,
            minMovePoints: minMove,
            timeToHitFraction: outcome.timeToHitFraction,
            pathFlags: outcome.pathFlags
        )
        let fillActual = outcome.fillRisk ?? 0.0
        let pathScore = 0.70 * abs(pathActual - matched.pathRisk) +
            0.30 * abs(fillActual - matched.pathRisk)
        return pushScore(
            aiIndex: aiIndex,
            regimeID: matched.regimeID,
            horizonSlot: TrainingSampleTools.horizonSlot(horizonMinutes: matched.horizonMinutes),
            classScore: 1.0 - fxClamp(pTrue, 0.0, 1.0),
            moveScore: fxClamp(moveScore, 0.0, 6.0),
            pathScore: fxClamp(pathScore, 0.0, 1.0)
        )
    }

    public func applyingAdjustment(
        aiIndex: Int,
        regimeID: Int,
        horizonMinutes: Int,
        minMovePoints: Double,
        to prediction: PredictionV4
    ) -> PredictionV4 {
        let horizonSlot = TrainingSampleTools.horizonSlot(horizonMinutes: horizonMinutes)
        guard horizonSlot >= 0, horizonSlot < RuntimeArtifactConstants.maxHorizons else {
            return prediction
        }
        let regime = Int(fxClamp(Double(regimeID), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))

        let qClass = quantile(aiIndex: aiIndex, regimeID: regime, horizonSlot: horizonSlot, scoreKind: .classScore, fallback: 0.35)
        let qMove = quantile(aiIndex: aiIndex, regimeID: regime, horizonSlot: horizonSlot, scoreKind: .moveScore, fallback: 0.20)
        let qPath = quantile(aiIndex: aiIndex, regimeID: regime, horizonSlot: horizonSlot, scoreKind: .pathScore, fallback: 0.10)

        let uncertainty = fxClamp(qClass, 0.0, 0.55)
        let skipBoost = fxClamp(0.32 * uncertainty + 0.14 * qPath, 0.0, 0.45)
        let probabilities = ConformalPendingEntry.sanitizedProbabilities(prediction.classProbabilities)
        let sell = probabilities[LabelClass.sell.rawValue] * (1.0 - skipBoost)
        let buy = probabilities[LabelClass.buy.rawValue] * (1.0 - skipBoost)
        let skip = probabilities[LabelClass.skip.rawValue] +
            skipBoost * (1.0 - probabilities[LabelClass.skip.rawValue])
        let denominator = max(sell + buy + skip, 1e-12)

        let moveWidth = max(
            prediction.moveQ75Points - prediction.moveQ25Points,
            max(minMovePoints, 0.25)
        )
        let extra = fxClamp(qMove, 0.0, 3.0) * max(0.50 * moveWidth, minMovePoints)
        let adjustedQ25 = max(0.0, prediction.moveQ25Points - 0.50 * extra)
        let adjustedQ50 = max(adjustedQ25, prediction.moveQ50Points)
        let adjustedQ75 = max(adjustedQ50, prediction.moveQ75Points + 0.50 * extra)

        return PredictionV4(
            classProbabilities: [sell / denominator, buy / denominator, skip / denominator],
            moveMeanPoints: max(0.0, prediction.moveMeanPoints * (1.0 - 0.12 * uncertainty)),
            moveQ25Points: adjustedQ25,
            moveQ50Points: adjustedQ50,
            moveQ75Points: adjustedQ75,
            mfeMeanPoints: prediction.mfeMeanPoints,
            maeMeanPoints: prediction.maeMeanPoints,
            hitTimeFraction: prediction.hitTimeFraction,
            pathRisk: fxClamp(prediction.pathRisk + 0.28 * qPath + 0.10 * uncertainty, 0.0, 1.0),
            fillRisk: fxClamp(prediction.fillRisk + 0.18 * qPath + 0.12 * uncertainty, 0.0, 1.0),
            confidence: fxClamp(max(buy / denominator, sell / denominator), 0.0, 1.0),
            reliability: fxClamp(prediction.reliability * (1.0 - 0.35 * uncertainty), 0.0, 1.0)
        )
    }

    public func pendingEntries(aiIndex: Int) -> [ConformalPendingEntry] {
        guard aiIndex >= 0, aiIndex < aiStates.count else { return [] }
        return pendingEntries(in: aiStates[aiIndex])
    }

    private func scoreArray(for state: ConformalCalibrationAIState, kind: ConformalScoreKind) -> [Double] {
        switch kind {
        case .classScore:
            state.classScores
        case .moveScore:
            state.moveScores
        case .pathScore:
            state.pathScores
        }
    }

    private static func scoreUpperBound(_ kind: ConformalScoreKind) -> Double {
        switch kind {
        case .classScore, .pathScore:
            1.0
        case .moveScore:
            6.0
        }
    }

    private func pendingEntries(in state: ConformalCalibrationAIState) -> [ConformalPendingEntry] {
        let capacity = RuntimeArtifactConstants.reliabilityPendingCapacity
        guard capacity > 0, state.pendingHead != state.pendingTail else { return [] }
        var output: [ConformalPendingEntry] = []
        var index = state.pendingHead
        while index != state.pendingTail {
            let entry = state.pendingEntries[index]
            if entry.signalSequence >= 0 {
                output.append(entry)
            }
            index += 1
            if index >= capacity {
                index = 0
            }
        }
        return output
    }

    private func resetPendingQueue(
        for state: ConformalCalibrationAIState,
        keeping entries: [ConformalPendingEntry]
    ) -> ConformalCalibrationAIState {
        var next = state
        let capacity = RuntimeArtifactConstants.reliabilityPendingCapacity
        next.pendingEntries = Array(repeating: ConformalPendingEntry(), count: capacity)
        next.pendingHead = 0
        next.pendingTail = 0
        for entry in entries.prefix(max(capacity - 1, 0)) {
            next.pendingEntries[next.pendingTail] = entry
            next.pendingTail += 1
            if next.pendingTail >= capacity {
                next.pendingTail = 0
            }
        }
        return next
    }
}
