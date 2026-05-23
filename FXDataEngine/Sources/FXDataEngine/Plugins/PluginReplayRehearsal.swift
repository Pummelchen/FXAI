import Foundation

public struct PluginReplayRehearsalEntry: Codable, Hashable, Sendable {
    public var valid: Bool
    public var priority: Double
    public var context: PluginContextV4
    public var labelClass: LabelClass
    public var movePoints: Double
    public var sampleWeight: Double
    public var mfePoints: Double
    public var maePoints: Double
    public var timeToHitFraction: Double
    public var pathFlags: Int
    public var pathRisk: Double
    public var fillRisk: Double
    public var maskedStepTarget: Double
    public var nextVolumeTarget: Double
    public var regimeShiftTarget: Double
    public var contextLeadTarget: Double
    public var x: [Double]
    public var xWindow: [[Double]]
    public var windowSize: Int

    public init(
        valid: Bool = true,
        priority: Double,
        context: PluginContextV4,
        labelClass: LabelClass,
        movePoints: Double,
        sampleWeight: Double,
        mfePoints: Double = 0.0,
        maePoints: Double = 0.0,
        timeToHitFraction: Double = 1.0,
        pathFlags: Int = 0,
        pathRisk: Double = 0.0,
        fillRisk: Double = 0.0,
        maskedStepTarget: Double = 0.0,
        nextVolumeTarget: Double = 0.0,
        regimeShiftTarget: Double = 0.0,
        contextLeadTarget: Double = 0.5,
        x: [Double],
        xWindow: [[Double]] = [],
        windowSize: Int? = nil
    ) {
        let sanitizedWindow = PluginContextPayloadTools.sanitizedWindow(
            xWindow,
            declaredWindowSize: windowSize ?? xWindow.count
        )
        self.valid = valid
        self.priority = fxSafeFinite(priority, fallback: -Double.greatestFiniteMagnitude)
        self.context = PluginContextPayloadTools.sanitizedContext(context)
        self.labelClass = labelClass
        self.movePoints = fxSafeFinite(movePoints)
        self.sampleWeight = max(0.0, fxSafeFinite(sampleWeight))
        self.mfePoints = max(0.0, fxSafeFinite(mfePoints))
        self.maePoints = max(0.0, fxSafeFinite(maePoints))
        self.timeToHitFraction = fxClamp(timeToHitFraction, 0.0, 1.0)
        self.pathFlags = pathFlags
        self.pathRisk = fxClamp(pathRisk, 0.0, 1.0)
        self.fillRisk = fxClamp(fillRisk, 0.0, 1.0)
        self.maskedStepTarget = fxSafeFinite(maskedStepTarget)
        self.nextVolumeTarget = max(0.0, fxSafeFinite(nextVolumeTarget))
        self.regimeShiftTarget = fxClamp(regimeShiftTarget, 0.0, 1.0)
        self.contextLeadTarget = fxClamp(contextLeadTarget, 0.0, 1.0)
        self.x = PluginContextPayloadTools.sanitizedInputVector(x)
        self.xWindow = sanitizedWindow.window
        self.windowSize = sanitizedWindow.windowSize
    }

    public init(
        sample: RuntimeArtifactPreparedSample,
        priority: Double? = nil,
        featureSchema: FeatureSchema = .full,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        sequenceBars: Int = 1,
        dataHasVolume: Bool? = nil,
        xWindow: [[Double]] = [],
        windowSize: Int? = nil
    ) {
        let prepared = sample.preparedTrainingSample
        let context = PluginContextV4(
            regimeID: prepared.regimeID,
            sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: prepared.sampleTimeUTC),
            horizonMinutes: prepared.horizonMinutes,
            featureSchema: featureSchema,
            normalizationMethod: normalizationMethod,
            sequenceBars: sequenceBars,
            priceCostPoints: prepared.costPoints,
            minMovePoints: prepared.minMovePoints,
            pointValue: prepared.pointValue,
            domainHash: prepared.domainHash,
            sampleTimeUTC: prepared.sampleTimeUTC,
            dataHasVolume: dataHasVolume ?? (prepared.nextVolumeTarget > 0.0)
        )
        self.init(
            valid: sample.valid,
            priority: priority ?? ReplayReservoirState.priority(for: sample),
            context: context,
            labelClass: prepared.labelClass,
            movePoints: prepared.movePoints,
            sampleWeight: prepared.sampleWeight,
            mfePoints: prepared.mfePoints,
            maePoints: prepared.maePoints,
            timeToHitFraction: prepared.timeToHitFraction,
            pathFlags: prepared.pathFlags.rawValue,
            pathRisk: prepared.pathRisk,
            fillRisk: prepared.fillRisk,
            maskedStepTarget: prepared.maskedStepTarget,
            nextVolumeTarget: prepared.nextVolumeTarget,
            regimeShiftTarget: prepared.regimeShiftTarget,
            contextLeadTarget: prepared.contextLeadTarget,
            x: prepared.x,
            xWindow: xWindow,
            windowSize: windowSize
        )
    }

    public var trainRequest: TrainRequestV4 {
        TrainRequestV4(
            valid: valid,
            context: context,
            labelClass: labelClass,
            movePoints: movePoints,
            sampleWeight: sampleWeight,
            mfePoints: mfePoints,
            maePoints: maePoints,
            timeToHitFraction: timeToHitFraction,
            pathFlags: pathFlags,
            pathRisk: pathRisk,
            fillRisk: fillRisk,
            maskedStepTarget: maskedStepTarget,
            nextVolumeTarget: nextVolumeTarget,
            regimeShiftTarget: regimeShiftTarget,
            contextLeadTarget: contextLeadTarget,
            windowSize: windowSize,
            x: x,
            xWindow: xWindow
        )
    }
}

public struct PluginReplayRehearsalCandidate: Codable, Hashable, Sendable {
    public var sourceIndex: Int
    public var score: Double
    public var entry: PluginReplayRehearsalEntry

    public init(sourceIndex: Int, score: Double, entry: PluginReplayRehearsalEntry) {
        self.sourceIndex = sourceIndex
        self.score = fxSafeFinite(score, fallback: -Double.greatestFiniteMagnitude)
        self.entry = entry
    }

    public var trainRequest: TrainRequestV4 {
        entry.trainRequest
    }
}

public struct PluginReplayBufferState: Codable, Hashable, Sendable {
    public var capacity: Int
    public private(set) var head: Int
    public private(set) var size: Int
    public private(set) var entries: [PluginReplayRehearsalEntry]

    public init(
        capacity: Int = FXDataEngineConstants.pluginReplayCapacity,
        head: Int = 0,
        size: Int = 0,
        entries: [PluginReplayRehearsalEntry] = []
    ) {
        self.capacity = max(1, capacity)
        self.head = head
        self.size = size
        self.entries = entries
        normalizeShape()
    }

    public mutating func reset() {
        head = 0
        size = 0
        entries = Array(repeating: Self.emptyEntry(), count: capacity)
    }

    @discardableResult
    public mutating func store(_ request: TrainRequestV4, priority: Double) -> Int {
        normalizeShape()
        let slot = head
        entries[slot] = PluginReplayRehearsalEntry(
            valid: request.valid,
            priority: priority,
            context: request.context,
            labelClass: request.labelClass,
            movePoints: request.movePoints,
            sampleWeight: request.sampleWeight,
            mfePoints: request.mfePoints,
            maePoints: request.maePoints,
            timeToHitFraction: request.timeToHitFraction,
            pathFlags: request.pathFlags,
            pathRisk: request.pathRisk,
            fillRisk: request.fillRisk,
            maskedStepTarget: request.maskedStepTarget,
            nextVolumeTarget: request.nextVolumeTarget,
            regimeShiftTarget: request.regimeShiftTarget,
            contextLeadTarget: request.contextLeadTarget,
            x: request.x,
            xWindow: request.xWindow,
            windowSize: request.windowSize
        )
        head = (head + 1) % capacity
        if size < capacity {
            size += 1
        }
        return slot
    }

    public var activeEntries: [PluginReplayRehearsalEntry] {
        normalizeEntries(entries).prefix(size).map { $0 }
    }

    public func selectedCandidates(
        regimeID: Int,
        horizonMinutes: Int,
        replaySteps: Int = FXDataEngineConstants.pluginReplaySteps
    ) -> [PluginReplayRehearsalCandidate] {
        PluginReplayRehearsalTools.selectedCandidates(
            entries: activeEntries,
            regimeID: regimeID,
            horizonMinutes: horizonMinutes,
            replaySteps: replaySteps
        )
    }

    private mutating func normalizeShape() {
        capacity = max(1, capacity)
        if head < 0 || head >= capacity {
            head = 0
        }
        size = min(max(0, size), capacity)
        entries = normalizeEntries(entries)
    }

    private func normalizeEntries(_ source: [PluginReplayRehearsalEntry]) -> [PluginReplayRehearsalEntry] {
        if source.count == capacity {
            return source
        }
        if source.count > capacity {
            return Array(source.prefix(capacity))
        }
        return source + Array(repeating: Self.emptyEntry(), count: capacity - source.count)
    }

    private static func emptyEntry() -> PluginReplayRehearsalEntry {
        PluginReplayRehearsalEntry(
            valid: false,
            priority: 0.0,
            context: PluginContextV4(),
            labelClass: .skip,
            movePoints: 0.0,
            sampleWeight: 0.0,
            x: []
        )
    }
}

public enum PluginReplayRehearsalTools {
    public static func selectedCandidates(
        entries: [PluginReplayRehearsalEntry],
        regimeID: Int,
        horizonMinutes: Int,
        replaySteps: Int = FXDataEngineConstants.pluginReplaySteps
    ) -> [PluginReplayRehearsalCandidate] {
        let steps = min(max(0, replaySteps), entries.count)
        guard steps > 0, !entries.isEmpty else { return [] }

        var best = Array(
            repeating: PluginReplayRehearsalCandidate(
                sourceIndex: -1,
                score: -Double.greatestFiniteMagnitude,
                entry: PluginReplayRehearsalEntry(
                    valid: false,
                    priority: -Double.greatestFiniteMagnitude,
                    context: PluginContextV4(),
                    labelClass: .skip,
                    movePoints: 0.0,
                    sampleWeight: 0.0,
                    x: []
                )
            ),
            count: steps
        )

        for (index, entry) in entries.enumerated() where entry.valid {
            var score = entry.priority
            if entry.context.regimeID == regimeID {
                score += 0.80
            }
            if entry.context.horizonMinutes == horizonMinutes {
                score += 0.60
            }

            for slot in 0..<steps where score > best[slot].score {
                if slot < steps - 1 {
                    for shift in stride(from: steps - 1, to: slot, by: -1) {
                        best[shift] = best[shift - 1]
                    }
                }
                best[slot] = PluginReplayRehearsalCandidate(sourceIndex: index, score: score, entry: entry)
                break
            }
        }

        return best.filter { $0.sourceIndex >= 0 }
    }

    public static func trainingRequests(
        entries: [PluginReplayRehearsalEntry],
        regimeID: Int,
        horizonMinutes: Int,
        replaySteps: Int = FXDataEngineConstants.pluginReplaySteps
    ) -> [TrainRequestV4] {
        selectedCandidates(
            entries: entries,
            regimeID: regimeID,
            horizonMinutes: horizonMinutes,
            replaySteps: replaySteps
        ).map(\.trainRequest)
    }
}
