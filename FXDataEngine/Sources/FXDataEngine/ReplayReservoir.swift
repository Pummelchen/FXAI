import Foundation

public struct ReplaySampleFlags: OptionSet, Codable, Hashable, Sendable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public init(pathFlags: SamplePathFlags) {
        self.rawValue = pathFlags.rawValue
    }

    public static let dualHit = ReplaySampleFlags(rawValue: SamplePathFlags.dualHit.rawValue)
    public static let killedEarly = ReplaySampleFlags(rawValue: SamplePathFlags.killedEarly.rawValue)
    public static let legacySpreadStress = ReplaySampleFlags(rawValue: SamplePathFlags.spreadStress.rawValue)
    public static let slowHit = ReplaySampleFlags(rawValue: SamplePathFlags.slowHit.rawValue)
    public static let falsePositive = ReplaySampleFlags(rawValue: 16)
    public static let missedMove = ReplaySampleFlags(rawValue: 32)
    public static let wrongDirection = ReplaySampleFlags(rawValue: 64)
}

public struct ReplayReservoirEntry: Codable, Hashable, Sendable {
    public var used: Bool
    public var priority: Double
    public var flags: ReplaySampleFlags
    public var sample: RuntimeArtifactPreparedSample

    public init(
        used: Bool = false,
        priority: Double = 0.0,
        flags: ReplaySampleFlags = [],
        sample: RuntimeArtifactPreparedSample = RuntimeArtifactPreparedSample()
    ) {
        self.used = used
        self.priority = fxClamp(priority, 0.0, 20.0)
        self.flags = flags
        self.sample = sample
    }
}

public struct ReplayReservoirState: Codable, Sendable {
    public private(set) var capacity: Int
    public private(set) var count: Int
    public private(set) var cursor: Int
    public private(set) var lastSampleTimeUTCByHorizon: [Int64]
    public private(set) var bucketCounts: [[Int]]
    public private(set) var entries: [ReplayReservoirEntry]

    public init(
        capacity: Int = RuntimeArtifactConstants.replayCapacity,
        count: Int = 0,
        cursor: Int = 0,
        lastSampleTimeUTCByHorizon: [Int64] = [],
        bucketCounts: [[Int]] = [],
        entries: [ReplayReservoirEntry] = []
    ) {
        self.capacity = max(1, capacity)
        self.count = count
        self.cursor = cursor
        self.lastSampleTimeUTCByHorizon = lastSampleTimeUTCByHorizon
        self.bucketCounts = bucketCounts
        self.entries = entries
        normalizeShape()
    }

    public mutating func reset() {
        count = 0
        cursor = 0
        lastSampleTimeUTCByHorizon = Array(repeating: 0, count: RuntimeArtifactConstants.maxHorizons)
        bucketCounts = Array(
            repeating: Array(repeating: 0, count: RuntimeArtifactConstants.maxHorizons),
            count: FXDataEngineConstants.pluginRegimeBuckets
        )
        entries = Array(repeating: ReplayReservoirEntry(), count: capacity)
    }

    @discardableResult
    public mutating func add(_ sample: RuntimeArtifactPreparedSample) -> Int? {
        normalizeShape()
        guard sample.valid else { return nil }

        let regimeID = normalizedRegime(sample.regimeID)
        let horizonSlot = normalizedHorizonSlot(sample.horizonSlot, horizonMinutes: sample.horizonMinutes)
        guard !(sample.sampleTimeUTC > 0 && lastSampleTimeUTCByHorizon[horizonSlot] == sample.sampleTimeUTC) else {
            return nil
        }

        let slot = nextInsertionSlot(regimeID: regimeID, horizonSlot: horizonSlot)
        guard slot >= 0 else { return nil }

        if entries[slot].used {
            decrementBucket(for: entries[slot].sample)
        } else {
            count += 1
        }

        var stored = sample
        stored.regimeID = regimeID
        stored.horizonSlot = horizonSlot
        entries[slot] = ReplayReservoirEntry(
            used: true,
            priority: Self.priority(for: stored),
            flags: ReplaySampleFlags(pathFlags: stored.pathFlags),
            sample: stored
        )
        bucketCounts[regimeID][horizonSlot] += 1
        if stored.sampleTimeUTC > 0 {
            lastSampleTimeUTCByHorizon[horizonSlot] = stored.sampleTimeUTC
        }
        return slot
    }

    @discardableResult
    public mutating func add(_ sample: RuntimeArtifactPreparedSample, analogMemory: inout AnalogMemoryStore) -> Int? {
        let slot = add(sample)
        if slot != nil {
            let prepared = sample.preparedTrainingSample
            analogMemory.update(
                modelInput: prepared.x,
                regimeID: normalizedRegime(prepared.regimeID),
                sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: prepared.sampleTimeUTC),
                horizonMinutes: prepared.horizonMinutes,
                domainHash: prepared.domainHash,
                movePoints: prepared.movePoints,
                minMovePoints: prepared.minMovePoints,
                qualityScore: prepared.qualityScore,
                pathRisk: prepared.pathRisk,
                fillRisk: prepared.fillRisk,
                sampleTimeUTC: prepared.sampleTimeUTC,
                sampleWeight: prepared.sampleWeight
            )
        }
        return slot
    }

    public mutating func boostPriorityByOutcome(
        sampleTimeUTC: Int64,
        horizonMinutes: Int,
        regimeID: Int,
        labelClass: LabelClass,
        signal: Int,
        movePoints: Double,
        minMovePoints: Double
    ) {
        normalizeShape()
        guard sampleTimeUTC > 0 else { return }

        let horizonSlot = TrainingSampleTools.horizonSlot(horizonMinutes: horizonMinutes)
        let minMove = max(minMovePoints, 0.50)
        let edge = max(abs(movePoints) - minMove, 0.0)
        let edgeRatio = fxClamp(edge / minMove, 0.0, 4.0)
        let falsePositive = (signal == 1 || signal == 0) && labelClass == .skip
        let wrongDirection = (signal == 1 && labelClass == .sell) || (signal == 0 && labelClass == .buy)
        let missedMove = signal == -1 && labelClass != .skip && edge > 0.0

        var baseBoost = 0.0
        var addFlags: ReplaySampleFlags = []
        if falsePositive {
            baseBoost += 1.10 + 0.35 * edgeRatio
            addFlags.insert(.falsePositive)
        }
        if wrongDirection {
            baseBoost += 1.35 + 0.45 * edgeRatio
            addFlags.insert(.wrongDirection)
        }
        if missedMove {
            baseBoost += 1.00 + 0.50 * edgeRatio
            addFlags.insert(.missedMove)
        }
        guard baseBoost > 0.0 else { return }

        for index in 0..<capacity where entries[index].used {
            guard entries[index].sample.sampleTimeUTC == sampleTimeUTC,
                  entries[index].sample.horizonSlot == horizonSlot else {
                continue
            }
            var boost = baseBoost
            if (0..<FXDataEngineConstants.pluginRegimeBuckets).contains(regimeID),
               entries[index].sample.regimeID == regimeID {
                boost += 0.20
            }
            if entries[index].flags.contains(.dualHit) {
                boost += 0.10
            }
            entries[index].priority = fxClamp(entries[index].priority + boost, 0.25, 20.0)
            entries[index].flags.formUnion(addFlags)
            break
        }
    }

    public mutating func selectReplaySamples(
        regimeID: Int,
        horizonMinutes: Int,
        epochs: Int,
        drawsPerEpoch: Int = 12,
        probeLimit: Int = 64
    ) -> [RuntimeArtifactPreparedSample] {
        normalizeShape()
        guard epochs > 0, drawsPerEpoch > 0, count > 0 else { return [] }

        let horizonSlot = TrainingSampleTools.horizonSlot(horizonMinutes: horizonMinutes)
        let preferredRegime = normalizedRegime(regimeID)
        let resolvedProbeLimit = min(max(1, probeLimit), min(count, capacity))
        var selected: [RuntimeArtifactPreparedSample] = []
        selected.reserveCapacity(epochs * drawsPerEpoch)

        for epoch in 0..<epochs {
            for draw in 0..<drawsPerEpoch {
                var bestIndex = -1
                var bestScore = -Double.greatestFiniteMagnitude
                let start = (cursor + draw * 7 + epoch * 13) % capacity
                for probe in 0..<resolvedProbeLimit {
                    let index = (start + probe) % capacity
                    guard entries[index].used else { continue }
                    let entry = entries[index]
                    var score = entry.priority
                    if entry.sample.regimeID == preferredRegime { score += 1.00 }
                    if entry.sample.horizonSlot == horizonSlot { score += 0.75 }
                    if entry.sample.labelClass != .skip { score += 0.20 }
                    if entry.flags.contains(.dualHit) { score += 0.20 }
                    if entry.flags.contains(.falsePositive) { score += 0.35 }
                    if entry.flags.contains(.missedMove) { score += 0.45 }
                    if entry.flags.contains(.wrongDirection) { score += 0.55 }
                    score -= 0.05 * abs(Double(cursor - index))
                    if score > bestScore {
                        bestScore = score
                        bestIndex = index
                    }
                }
                guard bestIndex >= 0 else { continue }
                selected.append(entries[bestIndex].sample)
                cursor = (bestIndex + 1) % capacity
            }
        }
        return selected
    }

    public static func priority(for sample: RuntimeArtifactPreparedSample) -> Double {
        let prepared = sample.preparedTrainingSample
        var priority = sample.sampleWeight
        priority += 0.35 * sample.qualityScore
        priority += 0.10 * fxClamp(prepared.pathRisk + prepared.fillRisk, 0.0, 3.0)
        if sample.labelClass != .skip {
            priority += 0.20
        }
        let flags = ReplaySampleFlags(pathFlags: sample.pathFlags)
        if flags.contains(.dualHit) {
            priority += 0.30
        }
        if flags.contains(.slowHit) {
            priority += 0.10
        }
        return fxClamp(priority, 0.25, 12.0)
    }

    private mutating func normalizeShape() {
        capacity = max(1, capacity)
        count = max(0, min(count, capacity))
        if cursor < 0 || cursor >= capacity {
            cursor = 0
        }
        if lastSampleTimeUTCByHorizon.count < RuntimeArtifactConstants.maxHorizons {
            lastSampleTimeUTCByHorizon.append(contentsOf: Array(
                repeating: 0,
                count: RuntimeArtifactConstants.maxHorizons - lastSampleTimeUTCByHorizon.count
            ))
        } else if lastSampleTimeUTCByHorizon.count > RuntimeArtifactConstants.maxHorizons {
            lastSampleTimeUTCByHorizon.removeLast(lastSampleTimeUTCByHorizon.count - RuntimeArtifactConstants.maxHorizons)
        }

        let regimeCount = FXDataEngineConstants.pluginRegimeBuckets
        if bucketCounts.count < regimeCount {
            bucketCounts.append(contentsOf: Array(
                repeating: [],
                count: regimeCount - bucketCounts.count
            ))
        } else if bucketCounts.count > regimeCount {
            bucketCounts.removeLast(bucketCounts.count - regimeCount)
        }
        for index in bucketCounts.indices {
            if bucketCounts[index].count < RuntimeArtifactConstants.maxHorizons {
                bucketCounts[index].append(contentsOf: Array(
                    repeating: 0,
                    count: RuntimeArtifactConstants.maxHorizons - bucketCounts[index].count
                ))
            } else if bucketCounts[index].count > RuntimeArtifactConstants.maxHorizons {
                bucketCounts[index].removeLast(bucketCounts[index].count - RuntimeArtifactConstants.maxHorizons)
            }
            bucketCounts[index] = bucketCounts[index].map { max(0, $0) }
        }

        if entries.count < capacity {
            entries.append(contentsOf: Array(repeating: ReplayReservoirEntry(), count: capacity - entries.count))
        } else if entries.count > capacity {
            entries.removeLast(entries.count - capacity)
        }
        count = min(entries.filter(\.used).count, capacity)
    }

    private func normalizedRegime(_ regimeID: Int) -> Int {
        guard regimeID >= 0 && regimeID < FXDataEngineConstants.pluginRegimeBuckets else { return 0 }
        return regimeID
    }

    private func normalizedHorizonSlot(_ horizonSlot: Int, horizonMinutes: Int) -> Int {
        if horizonSlot >= 0 && horizonSlot < RuntimeArtifactConstants.maxHorizons {
            return horizonSlot
        }
        return TrainingSampleTools.horizonSlot(horizonMinutes: horizonMinutes)
    }

    private mutating func decrementBucket(for sample: RuntimeArtifactPreparedSample) {
        let regime = normalizedRegime(sample.regimeID)
        let horizon = normalizedHorizonSlot(sample.horizonSlot, horizonMinutes: sample.horizonMinutes)
        if bucketCounts[regime][horizon] > 0 {
            bucketCounts[regime][horizon] -= 1
        }
    }

    private func nextInsertionSlot(regimeID: Int, horizonSlot: Int) -> Int {
        if count < capacity,
           let freeIndex = entries.firstIndex(where: { !$0.used }) {
            return freeIndex
        }

        let newBucket = Double(bucketCounts[regimeID][horizonSlot])
        var bestIndex = -1
        var bestEvictScore = -Double.greatestFiniteMagnitude
        for index in 0..<capacity where entries[index].used {
            let entry = entries[index]
            let oldRegime = normalizedRegime(entry.sample.regimeID)
            let oldHorizon = normalizedHorizonSlot(entry.sample.horizonSlot, horizonMinutes: entry.sample.horizonMinutes)
            let oldBucket = Double(bucketCounts[oldRegime][oldHorizon])
            var evictScore = oldBucket - 0.25 * entry.priority
            if entry.sample.labelClass == .skip {
                evictScore += 0.10
            }
            if oldBucket > newBucket {
                evictScore += 0.50
            }
            if evictScore > bestEvictScore {
                bestEvictScore = evictScore
                bestIndex = index
            }
        }
        return bestIndex
    }
}
