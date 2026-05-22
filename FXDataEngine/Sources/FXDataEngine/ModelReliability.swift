import Foundation

public struct ReliabilityClock: Codable, Hashable, Sendable {
    public var barTimeUTC: Int64
    public var sequence: Int

    public init(barTimeUTC: Int64 = 0, sequence: Int = 0) {
        self.barTimeUTC = max(0, barTimeUTC)
        self.sequence = max(0, sequence)
    }
}

public struct ReliabilityPendingEntry: Codable, Hashable, Sendable {
    public var signalSequence: Int
    public var signal: Int
    public var regimeID: Int
    public var sessionBucket: Int
    public var expectedMovePoints: Double
    public var horizonMinutes: Int
    public var probabilities: [Double]

    public init(
        signalSequence: Int = -1,
        signal: Int = -1,
        regimeID: Int = -1,
        sessionBucket: Int = 0,
        expectedMovePoints: Double = 0.0,
        horizonMinutes: Int = 1,
        probabilities: [Double] = []
    ) {
        self.signalSequence = signalSequence
        self.signal = signal
        self.regimeID = regimeID
        self.sessionBucket = max(0, sessionBucket)
        self.expectedMovePoints = max(0.0, fxSafeFinite(expectedMovePoints))
        self.horizonMinutes = TrainingSampleTools.clampHorizon(horizonMinutes)
        self.probabilities = Self.normalizedProbabilities(probabilities)
    }

    private static func normalizedProbabilities(_ probabilities: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: LabelClass.allCases.count)
        for index in 0..<min(probabilities.count, output.count) {
            output[index] = fxClamp(probabilities[index], 0.0, 1.0)
        }
        return output
    }
}

public struct ReliabilityPendingQueue: Codable, Hashable, Sendable {
    public var head: Int
    public var tail: Int
    public var entries: [ReliabilityPendingEntry]

    public init(
        head: Int = 0,
        tail: Int = 0,
        entries: [ReliabilityPendingEntry] = [],
        capacity: Int = RuntimeArtifactConstants.reliabilityPendingCapacity
    ) {
        let capacity = max(1, capacity)
        self.head = Int(fxClamp(Double(head), 0.0, Double(capacity - 1)))
        self.tail = Int(fxClamp(Double(tail), 0.0, Double(capacity - 1)))
        self.entries = Array(entries.prefix(capacity))
        if self.entries.count < capacity {
            self.entries.append(contentsOf: Array(
                repeating: ReliabilityPendingEntry(),
                count: capacity - self.entries.count
            ))
        }
    }

    public var capacity: Int {
        entries.count
    }

    public func activeEntries() -> [ReliabilityPendingEntry] {
        guard capacity > 0, head != tail else { return [] }
        var output: [ReliabilityPendingEntry] = []
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

public enum ModelReliabilityTools {
    public static func updatedReliability(
        currentReliability: Double,
        labelClass: LabelClass,
        signal: Int,
        realizedMovePoints: Double,
        minMovePoints: Double,
        expectedMovePoints: Double,
        probabilities: [Double]
    ) -> Double {
        let bestClass = bestProbabilityClass(probabilities)
        let labelIndex = labelClass.rawValue
        let pTrue = fxClamp(probability(probabilities, labelIndex), 0.0, 1.0)
        let minMove = max(fxSafeFinite(minMovePoints), 0.10)
        let realizedMove = fxSafeFinite(realizedMovePoints)
        let target: Double

        if signal == LabelClass.buy.rawValue || signal == LabelClass.sell.rawValue {
            let directionalMove = signal == LabelClass.buy.rawValue ? realizedMove : -realizedMove
            let netPoints = directionalMove - minMove
            let edgeNorm = fxClamp(netPoints / minMove, -2.5, 2.5)
            let predictedClass: LabelClass = signal == LabelClass.buy.rawValue ? .buy : .sell
            var classBonus = predictedClass == labelClass ? 0.20 : -0.20
            if labelClass == .skip {
                classBonus -= 0.15
            }
            let expectedMove = max(fxSafeFinite(expectedMovePoints), minMove)
            let expectedFit = 1.0 - fxClamp(
                abs(abs(realizedMove) - expectedMove) / max(expectedMove, 0.10),
                0.0,
                1.5
            )
            target = 1.0 +
                0.35 * edgeNorm +
                classBonus +
                0.10 * (pTrue - 0.5) * 2.0 +
                0.08 * expectedFit
        } else if labelClass == .skip {
            target = 1.10 + 0.10 * pTrue
        } else {
            let opportunity = fxClamp((abs(realizedMove) - minMove) / minMove, 0.0, 3.0)
            target = 0.95 - 0.20 * opportunity
        }

        var adjustedTarget = target
        if bestClass == labelClass {
            adjustedTarget += 0.05
        }
        adjustedTarget = fxClamp(adjustedTarget, 0.20, 2.80)
        return fxClamp(
            0.97 * fxSafeFinite(currentReliability, fallback: 1.0) + 0.03 * adjustedTarget,
            0.20,
            3.00
        )
    }

    public static func advancedClock(_ clock: ReliabilityClock, signalBarTimeUTC: Int64) -> ReliabilityClock {
        guard signalBarTimeUTC > 0 else { return clock }
        guard clock.barTimeUTC > 0 else {
            return ReliabilityClock(barTimeUTC: signalBarTimeUTC, sequence: 0)
        }
        guard signalBarTimeUTC != clock.barTimeUTC else { return clock }
        return ReliabilityClock(barTimeUTC: signalBarTimeUTC, sequence: clock.sequence + 1)
    }

    public static func enqueuedPending(
        _ queue: ReliabilityPendingQueue,
        signalSequence: Int,
        signal: Int,
        regimeID: Int,
        sessionBucket: Int,
        expectedMovePoints: Double,
        horizonMinutes: Int,
        probabilities: [Double]
    ) -> ReliabilityPendingQueue {
        guard signalSequence >= 0 else { return queue }
        var queue = ReliabilityPendingQueue(
            head: queue.head,
            tail: queue.tail,
            entries: queue.entries,
            capacity: queue.capacity
        )
        let entry = ReliabilityPendingEntry(
            signalSequence: signalSequence,
            signal: signal,
            regimeID: regimeID,
            sessionBucket: sessionBucket,
            expectedMovePoints: expectedMovePoints,
            horizonMinutes: horizonMinutes,
            probabilities: probabilities
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

    public static func maxReliabilityPendingHorizon(
        fallbackHorizonMinutes: Int,
        queues: [ReliabilityPendingQueue]
    ) -> Int {
        var maximum = TrainingSampleTools.clampHorizon(fallbackHorizonMinutes)
        for queue in queues {
            for entry in queue.activeEntries() where entry.signalSequence >= 0 {
                maximum = max(maximum, TrainingSampleTools.clampHorizon(entry.horizonMinutes))
            }
        }
        return maximum
    }

    public static func voteWeight(aiIndex: Int, reliabilities: [Double]) -> Double {
        guard aiIndex >= 0, aiIndex < reliabilities.count else { return 1.0 }
        return fxClamp(reliabilities[aiIndex], 0.20, 3.00)
    }

    private static func bestProbabilityClass(_ probabilities: [Double]) -> LabelClass {
        var best = LabelClass.sell
        for label in LabelClass.allCases.dropFirst() {
            if probability(probabilities, label.rawValue) > probability(probabilities, best.rawValue) {
                best = label
            }
        }
        return best
    }

    private static func probability(_ probabilities: [Double], _ index: Int) -> Double {
        guard index >= 0, index < probabilities.count else { return 0.0 }
        return fxSafeFinite(probabilities[index])
    }
}
