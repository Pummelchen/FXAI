import Foundation

public enum MetaPendingReplayKind: String, Codable, Hashable, Sendable {
    case stack
    case policy
    case horizonPolicy

    public var featureCount: Int {
        switch self {
        case .stack:
            FXDataEngineConstants.stackFeatures
        case .policy:
            FXDataEngineConstants.policyFeatures
        case .horizonPolicy:
            FXDataEngineConstants.horizonPolicyFeatures
        }
    }
}

public struct MetaPendingReplayEntry: Codable, Hashable, Sendable {
    public var kind: MetaPendingReplayKind
    public var signalSequence: Int
    public var signal: Int
    public var regimeID: Int
    public var horizonMinutes: Int
    public var minMovePoints: Double
    public var expectedMovePoints: Double
    public var probabilities: [Double]
    public var features: [Double]

    public init(
        kind: MetaPendingReplayKind,
        signalSequence: Int = -1,
        signal: Int = -1,
        regimeID: Int = 0,
        horizonMinutes: Int = 1,
        minMovePoints: Double = 0.0,
        expectedMovePoints: Double = 0.0,
        probabilities: [Double] = [],
        features: [Double] = []
    ) {
        self.kind = kind
        self.signalSequence = signalSequence
        self.signal = signal
        self.regimeID = regimeID
        self.horizonMinutes = HorizonTools.clampHorizon(horizonMinutes)
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.expectedMovePoints = max(0.0, fxSafeFinite(expectedMovePoints))
        self.probabilities = Self.normalizedProbabilities(probabilities)
        self.features = Self.normalizedFeatures(features, count: kind.featureCount)
    }

    private static func normalizedProbabilities(_ probabilities: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: LabelClass.allCases.count)
        for index in 0..<min(probabilities.count, output.count) {
            output[index] = fxClamp(probabilities[index], 0.0, 1.0)
        }
        return output
    }

    private static func normalizedFeatures(_ features: [Double], count: Int) -> [Double] {
        var output = Array(repeating: 0.0, count: count)
        for index in 0..<min(features.count, count) {
            output[index] = fxSafeFinite(features[index])
        }
        return output
    }
}

public struct MetaPendingReplayQueue: Codable, Hashable, Sendable {
    public var kind: MetaPendingReplayKind
    public var head: Int
    public var tail: Int
    public var entries: [MetaPendingReplayEntry]

    public init(
        kind: MetaPendingReplayKind,
        head: Int = 0,
        tail: Int = 0,
        entries: [MetaPendingReplayEntry] = [],
        capacity: Int = RuntimeArtifactConstants.reliabilityPendingCapacity
    ) {
        let capacity = max(1, capacity)
        self.kind = kind
        self.head = Int(fxClamp(Double(head), 0.0, Double(capacity - 1)))
        self.tail = Int(fxClamp(Double(tail), 0.0, Double(capacity - 1)))
        self.entries = entries.prefix(capacity).map { entry in
            MetaPendingReplayEntry(
                kind: kind,
                signalSequence: entry.signalSequence,
                signal: entry.signal,
                regimeID: entry.regimeID,
                horizonMinutes: entry.horizonMinutes,
                minMovePoints: entry.minMovePoints,
                expectedMovePoints: entry.expectedMovePoints,
                probabilities: entry.probabilities,
                features: entry.features
            )
        }
        if self.entries.count < capacity {
            self.entries.append(contentsOf: Array(
                repeating: MetaPendingReplayEntry(kind: kind),
                count: capacity - self.entries.count
            ))
        }
    }

    public var capacity: Int {
        entries.count
    }

    public func activeEntries() -> [MetaPendingReplayEntry] {
        guard capacity > 0, head != tail else { return [] }
        var output: [MetaPendingReplayEntry] = []
        var index = head
        while index != tail {
            let entry = entries[index]
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
}

public struct MetaPendingReplayOutcomeAction: Codable, Hashable, Sendable {
    public var entry: MetaPendingReplayEntry
    public var age: Int
    public var predictionIndex: Int
    public var canEvaluate: Bool

    public init(
        entry: MetaPendingReplayEntry,
        age: Int,
        predictionIndex: Int,
        canEvaluate: Bool
    ) {
        self.entry = entry
        self.age = max(0, age)
        self.predictionIndex = max(0, predictionIndex)
        self.canEvaluate = canEvaluate
    }
}

public struct MetaPendingReplayResolution: Codable, Hashable, Sendable {
    public var keptQueue: MetaPendingReplayQueue
    public var outcomeActions: [MetaPendingReplayOutcomeAction]

    public init(
        keptQueue: MetaPendingReplayQueue,
        outcomeActions: [MetaPendingReplayOutcomeAction] = []
    ) {
        self.keptQueue = keptQueue
        self.outcomeActions = outcomeActions
    }
}

public enum MetaPendingReplayTools {
    public static func enqueuedPending(
        _ queue: MetaPendingReplayQueue,
        signalSequence: Int,
        signal: Int = -1,
        regimeID: Int,
        horizonMinutes: Int,
        minMovePoints: Double = 0.0,
        expectedMovePoints: Double = 0.0,
        probabilities: [Double] = [],
        features: [Double] = []
    ) -> MetaPendingReplayQueue {
        guard signalSequence >= 0 else { return queue }
        var queue = MetaPendingReplayQueue(
            kind: queue.kind,
            head: queue.head,
            tail: queue.tail,
            entries: queue.entries,
            capacity: queue.capacity
        )
        let entry = MetaPendingReplayEntry(
            kind: queue.kind,
            signalSequence: signalSequence,
            signal: signal,
            regimeID: regimeID,
            horizonMinutes: horizonMinutes,
            minMovePoints: minMovePoints,
            expectedMovePoints: expectedMovePoints,
            probabilities: probabilities,
            features: features
        )

        var previous = queue.tail - 1
        if previous < 0 {
            previous += queue.capacity
        }
        if queue.head != queue.tail, queue.entries[previous].signalSequence == signalSequence {
            queue.entries[previous] = entry
            return queue
        }

        queue.entries[queue.tail] = entry
        var nextTail = queue.tail + 1
        if nextTail >= queue.capacity {
            nextTail = 0
        }
        if nextTail == queue.head {
            queue.head += 1
            if queue.head >= queue.capacity {
                queue.head = 0
            }
        }
        queue.tail = nextTail
        return queue
    }

    public static func resolvedPendingOutcomes(
        _ queue: MetaPendingReplayQueue,
        currentSignalSequence: Int,
        availableBarCount: Int
    ) -> MetaPendingReplayResolution {
        guard currentSignalSequence >= 0 else {
            return MetaPendingReplayResolution(keptQueue: queue)
        }

        let capacity = max(queue.capacity, 1)
        let keepLimit = max(capacity - 1, 0)
        var keptEntries: [MetaPendingReplayEntry] = []
        var actions: [MetaPendingReplayOutcomeAction] = []

        for entry in queue.activeEntries() {
            let age = currentSignalSequence - entry.signalSequence
            let horizon = HorizonTools.clampHorizon(entry.horizonMinutes)
            guard age >= horizon else {
                if keptEntries.count < keepLimit {
                    keptEntries.append(entry)
                }
                continue
            }

            let predictionIndex = age
            actions.append(MetaPendingReplayOutcomeAction(
                entry: entry,
                age: age,
                predictionIndex: predictionIndex,
                canEvaluate: predictionIndex >= 0 && predictionIndex < availableBarCount
            ))
        }

        return MetaPendingReplayResolution(
            keptQueue: MetaPendingReplayQueue(
                kind: queue.kind,
                head: 0,
                tail: keptEntries.count,
                entries: keptEntries,
                capacity: capacity
            ),
            outcomeActions: actions
        )
    }
}
