import Foundation

public struct FoundationSignals: Codable, Hashable, Sendable {
    public var maskedStep: Double
    public var nextVolumeTarget: Double
    public var regimeTransition: Double
    public var contextAlignment: Double
    public var directionBias: Double
    public var moveRatio: Double
    public var tradability: Double
    public var trust: Double

    public init(
        maskedStep: Double = 0.0,
        nextVolumeTarget: Double = 0.0,
        regimeTransition: Double = 0.0,
        contextAlignment: Double = 0.5,
        directionBias: Double = 0.0,
        moveRatio: Double = 1.0,
        tradability: Double = 0.0,
        trust: Double = 0.0
    ) {
        self.maskedStep = maskedStep
        self.nextVolumeTarget = nextVolumeTarget
        self.regimeTransition = regimeTransition
        self.contextAlignment = contextAlignment
        self.directionBias = directionBias
        self.moveRatio = moveRatio
        self.tradability = tradability
        self.trust = trust
    }
}

public struct StudentSignals: Codable, Hashable, Sendable {
    public var classProbabilities: [Double]
    public var moveRatio: Double
    public var tradability: Double
    public var horizonFit: Double
    public var trust: Double

    public init(
        classProbabilities: [Double] = [0.3333, 0.3333, 0.3334],
        moveRatio: Double = 1.0,
        tradability: Double = 0.0,
        horizonFit: Double = 0.5,
        trust: Double = 0.0
    ) {
        var probabilities = Array(repeating: 0.0, count: 3)
        for index in 0..<min(3, classProbabilities.count) {
            probabilities[index] = fxSafeFinite(classProbabilities[index])
        }
        self.classProbabilities = probabilities
        self.moveRatio = moveRatio
        self.tradability = tradability
        self.horizonFit = horizonFit
        self.trust = trust
    }
}

public struct AnalogMemoryQuery: Codable, Hashable, Sendable {
    public var similarity: Double
    public var directionAgreement: Double
    public var edgeNorm: Double
    public var quality: Double
    public var pathSafety: Double
    public var executionSafety: Double
    public var domainAlignment: Double
    public var matches: Int

    public init(
        similarity: Double = 0.0,
        directionAgreement: Double = 0.0,
        edgeNorm: Double = 0.0,
        quality: Double = 0.0,
        pathSafety: Double = 0.5,
        executionSafety: Double = 0.5,
        domainAlignment: Double = 0.0,
        matches: Int = 0
    ) {
        self.similarity = similarity
        self.directionAgreement = directionAgreement
        self.edgeNorm = edgeNorm
        self.quality = quality
        self.pathSafety = pathSafety
        self.executionSafety = executionSafety
        self.domainAlignment = domainAlignment
        self.matches = matches
    }
}

public struct HierarchicalSignals: Codable, Hashable, Sendable {
    public var tradability: Double
    public var directionConfidence: Double
    public var moveAdequacy: Double
    public var pathQuality: Double
    public var executionViability: Double
    public var horizonFit: Double
    public var consistency: Double
    public var score: Double

    public init(
        tradability: Double = 0.0,
        directionConfidence: Double = 0.0,
        moveAdequacy: Double = 0.0,
        pathQuality: Double = 0.0,
        executionViability: Double = 0.0,
        horizonFit: Double = 0.0,
        consistency: Double = 0.0,
        score: Double = 0.0
    ) {
        self.tradability = tradability
        self.directionConfidence = directionConfidence
        self.moveAdequacy = moveAdequacy
        self.pathQuality = pathQuality
        self.executionViability = executionViability
        self.horizonFit = horizonFit
        self.consistency = consistency
        self.score = score
    }
}

public struct AnalogMemoryEntry: Codable, Hashable, Sendable {
    public var sampleTimeUTC: Int64
    public var regimeID: Int
    public var sessionBucket: Int
    public var horizonBucket: Int
    public var domainHash: Double
    public var vector: [Double]
    public var direction: Double
    public var edgeNorm: Double
    public var quality: Double
    public var pathRisk: Double
    public var fillRisk: Double
    public var weight: Double

    public init(
        sampleTimeUTC: Int64 = 0,
        regimeID: Int = 0,
        sessionBucket: Int = 0,
        horizonBucket: Int = 0,
        domainHash: Double = 0.0,
        vector: [Double] = Array(repeating: 0.0, count: FXDataEngineConstants.analogMemoryFeatures),
        direction: Double = 0.0,
        edgeNorm: Double = 0.0,
        quality: Double = 0.0,
        pathRisk: Double = 0.0,
        fillRisk: Double = 0.0,
        weight: Double = 0.0
    ) {
        self.sampleTimeUTC = sampleTimeUTC
        self.regimeID = regimeID
        self.sessionBucket = sessionBucket
        self.horizonBucket = horizonBucket
        self.domainHash = domainHash
        self.vector = AnalogMemoryTools.normalizedVector(vector)
        self.direction = direction
        self.edgeNorm = edgeNorm
        self.quality = quality
        self.pathRisk = pathRisk
        self.fillRisk = fillRisk
        self.weight = weight
    }
}

public struct AnalogMemoryStore: Codable, Sendable {
    public private(set) var capacity: Int
    public private(set) var head: Int
    public private(set) var size: Int
    public private(set) var entries: [AnalogMemoryEntry]

    public var ready: Bool {
        size >= FXDataEngineConstants.analogMemoryMinMatches
    }

    public init(capacity: Int = FXDataEngineConstants.analogMemoryCapacity) {
        let resolvedCapacity = max(1, capacity)
        self.capacity = resolvedCapacity
        self.head = 0
        self.size = 0
        self.entries = Array(repeating: AnalogMemoryEntry(), count: resolvedCapacity)
    }

    public init(
        capacity: Int,
        head: Int,
        size: Int,
        entries: [AnalogMemoryEntry]
    ) {
        let resolvedCapacity = max(1, capacity)
        self.capacity = resolvedCapacity
        self.head = head
        self.size = size
        self.entries = Array(entries.prefix(resolvedCapacity))
        if self.entries.count < resolvedCapacity {
            self.entries.append(contentsOf: Array(
                repeating: AnalogMemoryEntry(),
                count: resolvedCapacity - self.entries.count
            ))
        }
        ensureStorage()
    }

    public mutating func reset() {
        ensureStorage()
        head = 0
        size = 0
        entries = Array(repeating: AnalogMemoryEntry(), count: capacity)
    }

    public mutating func update(
        modelInput: [Double],
        regimeID: Int,
        sessionBucket: Int,
        horizonMinutes: Int,
        domainHash: Double,
        movePoints: Double,
        minMovePoints: Double,
        qualityScore: Double,
        pathRisk: Double,
        fillRisk: Double,
        sampleTimeUTC: Int64,
        sampleWeight: Double
    ) {
        ensureStorage()
        let minMove = max(minMovePoints, 0.10)
        let direction = abs(movePoints) > minMove ? fxSign(movePoints) : 0.0
        let slot = (head >= 0 && head < capacity) ? head : 0
        head = (slot + 1) % capacity
        if size < capacity {
            size += 1
        }

        entries[slot] = AnalogMemoryEntry(
            sampleTimeUTC: sampleTimeUTC,
            regimeID: regimeID >= 0 && regimeID < FXDataEngineConstants.pluginRegimeBuckets ? regimeID : 0,
            sessionBucket: sessionBucket >= 0 && sessionBucket < FXDataEngineConstants.pluginSessionBuckets ? sessionBucket : 0,
            horizonBucket: AnalogMemoryTools.horizonBucket(horizonMinutes: horizonMinutes),
            domainHash: fxClamp(domainHash, 0.0, 1.0),
            vector: AnalogMemoryTools.buildVector(modelInput: modelInput),
            direction: direction,
            edgeNorm: fxClamp(movePoints / minMove, -6.0, 6.0) / 6.0,
            quality: fxClamp(qualityScore, 0.0, 2.0) / 2.0,
            pathRisk: fxClamp(pathRisk, 0.0, 1.0),
            fillRisk: fxClamp(fillRisk, 0.0, 1.0),
            weight: fxClamp(sampleWeight, 0.10, 8.0)
        )
    }

    private mutating func ensureStorage() {
        capacity = max(1, capacity)
        if entries.count < capacity {
            entries.append(contentsOf: Array(repeating: AnalogMemoryEntry(), count: capacity - entries.count))
        } else if entries.count > capacity {
            entries.removeLast(entries.count - capacity)
        }
        size = min(max(size, 0), capacity)
        if head < 0 || head >= capacity {
            head = size % capacity
        }
    }

    public func query(
        modelInput: [Double],
        regimeID: Int,
        sessionBucket: Int,
        horizonMinutes: Int,
        domainHash: Double
    ) -> AnalogMemoryQuery {
        guard size > 0 else { return AnalogMemoryQuery() }

        let vector = AnalogMemoryTools.buildVector(modelInput: modelInput)
        let regime = regimeID >= 0 && regimeID < FXDataEngineConstants.pluginRegimeBuckets ? regimeID : 0
        let session = sessionBucket >= 0 && sessionBucket < FXDataEngineConstants.pluginSessionBuckets ? sessionBucket : 0
        let horizon = AnalogMemoryTools.horizonBucket(horizonMinutes: horizonMinutes)
        let domain = fxClamp(domainHash, 0.0, 1.0)

        var bestSimilarity = 0.0
        var sumWeight = 0.0
        var sumDirection = 0.0
        var sumEdge = 0.0
        var sumQuality = 0.0
        var sumPath = 0.0
        var sumFill = 0.0
        var sumDomain = 0.0
        var matches = 0

        for index in 0..<min(size, entries.count) {
            let entry = entries[index]
            let distance = AnalogMemoryTools.distance(vector, entry: entry)
            let similarity = exp(-0.45 * distance)
            let regimeBoost = entry.regimeID == regime ? 1.20 : 0.85
            let sessionBoost = entry.sessionBucket == session ? 1.10 : 0.92
            let horizonBoost = entry.horizonBucket == horizon ? 1.15 : 0.90
            let domainAlignment = 1.0 - min(abs(entry.domainHash - domain), 1.0)
            let weight = similarity *
                regimeBoost *
                sessionBoost *
                horizonBoost *
                (0.70 + 0.30 * domainAlignment) *
                (0.40 + 0.60 * fxClamp(entry.weight / 4.0, 0.0, 1.0))

            if weight < 0.05 { continue }

            matches += 1
            bestSimilarity = max(bestSimilarity, weight)
            sumWeight += weight
            sumDirection += weight * entry.direction
            sumEdge += weight * entry.edgeNorm
            sumQuality += weight * entry.quality
            sumPath += weight * (1.0 - entry.pathRisk)
            sumFill += weight * (1.0 - entry.fillRisk)
            sumDomain += weight * domainAlignment
        }

        guard sumWeight > 1e-9 else { return AnalogMemoryQuery() }
        return AnalogMemoryQuery(
            similarity: fxClamp(bestSimilarity, 0.0, 1.0),
            directionAgreement: fxClamp(sumDirection / sumWeight, -1.0, 1.0),
            edgeNorm: fxClamp(sumEdge / sumWeight, -1.0, 1.0),
            quality: fxClamp(sumQuality / sumWeight, 0.0, 1.0),
            pathSafety: fxClamp(sumPath / sumWeight, 0.0, 1.0),
            executionSafety: fxClamp(sumFill / sumWeight, 0.0, 1.0),
            domainAlignment: fxClamp(sumDomain / sumWeight, 0.0, 1.0),
            matches: matches
        )
    }
}

public enum AnalogMemoryTools {
    public static func inputFeature(_ modelInput: [Double], featureIndex: Int) -> Double {
        let index = featureIndex + 1
        guard index >= 0, index < modelInput.count else { return 0.0 }
        return modelInput[index]
    }

    public static func horizonBucket(
        horizonMinutes: Int,
        bucketCount: Int = FXDataEngineConstants.pluginHorizonBuckets
    ) -> Int {
        var horizon = horizonMinutes
        if horizon < 1 { horizon = 1 }
        if horizon > 720 { horizon = 720 }

        let buckets = bucketCount
        if buckets <= 1 { return 0 }

        let scaled = log(Double(horizon) + 1.0) / log(721.0)
        var bucket = Int(floor(scaled * Double(buckets)))
        if bucket < 0 { bucket = 0 }
        if bucket >= buckets { bucket = buckets - 1 }
        return bucket
    }

    public static func analogFeature(modelInput: [Double], featureIndex: Int) -> Double {
        switch featureIndex {
        case 0:
            fxClamp(inputFeature(modelInput, featureIndex: 0), -4.0, 4.0)
        case 1:
            fxClamp(inputFeature(modelInput, featureIndex: 3), -4.0, 4.0)
        case 2:
            fxClamp(inputFeature(modelInput, featureIndex: 5), 0.0, 6.0)
        case 3:
            fxClamp(inputFeature(modelInput, featureIndex: 10), -4.0, 4.0)
        case 4:
            fxClamp(inputFeature(modelInput, featureIndex: 41), 0.0, 6.0)
        case 5:
            fxClamp(inputFeature(modelInput, featureIndex: 62), -4.0, 4.0)
        case 6:
            fxClamp(inputFeature(modelInput, featureIndex: 72), -4.0, 4.0)
        case 7:
            fxClamp(inputFeature(modelInput, featureIndex: 78), -6.0, 8.0)
        case 8:
            fxClamp(inputFeature(modelInput, featureIndex: 80), -4.0, 8.0)
        case 9:
            fxClamp(inputFeature(modelInput, featureIndex: 82), 0.0, 8.0)
        case 10:
            fxClamp(
                0.70 * inputFeature(modelInput, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 2) +
                    0.30 * inputFeature(modelInput, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 19),
                0.0,
                1.0
            )
        case 11:
            fxClamp(
                0.60 * inputFeature(modelInput, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 3) +
                    0.40 * inputFeature(modelInput, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 15),
                -6.0,
                6.0
            )
        default:
            0.0
        }
    }

    public static func buildVector(modelInput: [Double]) -> [Double] {
        (0..<FXDataEngineConstants.analogMemoryFeatures).map { analogFeature(modelInput: modelInput, featureIndex: $0) }
    }

    public static func normalizedVector(_ vector: [Double]) -> [Double] {
        var normalized = Array(repeating: 0.0, count: FXDataEngineConstants.analogMemoryFeatures)
        for index in 0..<min(vector.count, FXDataEngineConstants.analogMemoryFeatures) {
            normalized[index] = fxSafeFinite(vector[index])
        }
        return normalized
    }

    public static func distance(_ vector: [Double], entry: AnalogMemoryEntry) -> Double {
        let normalized = normalizedVector(vector)
        var accumulated = 0.0
        for index in 0..<FXDataEngineConstants.analogMemoryFeatures {
            let weight: Double
            if index == 2 || index == 4 || index == 7 || index == 9 {
                weight = 1.20
            } else if index >= 10 {
                weight = 0.85
            } else {
                weight = 1.0
            }
            let entryValue = HorizonTools.value(in: entry.vector, index: index, default: 0.0)
            let delta = normalized[index] - entryValue
            accumulated += weight * delta * delta
        }
        return sqrt(max(accumulated, 0.0))
    }

    public static func buildHierarchicalSignals(
        classProbabilities: [Double],
        expectedMovePoints: Double,
        minMovePoints: Double,
        confidence: Double,
        reliability: Double,
        pathRisk: Double,
        fillRisk: Double,
        hitTimeFraction: Double,
        contextQuality: Double,
        horizonMinutes: Int,
        foundation: FoundationSignals = FoundationSignals(),
        student: StudentSignals = StudentSignals(),
        analog: AnalogMemoryQuery = AnalogMemoryQuery()
    ) -> HierarchicalSignals {
        let minMove = max(minMovePoints, 0.10)
        let sellProbability = fxClamp(HorizonTools.value(in: classProbabilities, index: LabelClass.sell.rawValue, default: 0.3333), 0.0, 1.0)
        let buyProbability = fxClamp(HorizonTools.value(in: classProbabilities, index: LabelClass.buy.rawValue, default: 0.3333), 0.0, 1.0)
        let skipProbability = fxClamp(HorizonTools.value(in: classProbabilities, index: LabelClass.skip.rawValue, default: 0.3334), 0.0, 1.0)
        let directionGap = fxClamp(abs(buyProbability - sellProbability), 0.0, 1.0)
        let directionBias = fxSign(buyProbability - sellProbability)
        let moveRatio = fxClamp(expectedMovePoints / minMove, 0.0, 6.0) / 6.0
        let studentBias = fxClamp(
            HorizonTools.value(in: student.classProbabilities, index: LabelClass.buy.rawValue, default: 0.3333) -
                HorizonTools.value(in: student.classProbabilities, index: LabelClass.sell.rawValue, default: 0.3333),
            -1.0,
            1.0
        )
        let foundationBias = fxClamp(foundation.directionBias, -1.0, 1.0)
        let analogBias = fxClamp(analog.directionAgreement, -1.0, 1.0)
        let directionalAgreement = fxClamp(
            1.0 -
                0.35 * abs(directionBias - fxSign(foundationBias)) -
                0.30 * abs(directionBias - fxSign(studentBias)) -
                0.25 * abs(directionBias - fxSign(analogBias)),
            0.0,
            1.0
        )

        let tradability = fxClamp(
            0.20 +
                0.18 * (1.0 - skipProbability) +
                0.14 * fxClamp(confidence, 0.0, 1.0) +
                0.12 * fxClamp(reliability, 0.0, 1.0) +
                0.10 * fxClamp(foundation.tradability, 0.0, 1.0) +
                0.10 * fxClamp(student.tradability, 0.0, 1.0) +
                0.08 * fxClamp(analog.quality, 0.0, 1.0) +
                0.06 * fxClamp(analog.similarity, 0.0, 1.0) -
                0.10 * fxClamp(pathRisk, 0.0, 1.0) -
                0.10 * fxClamp(fillRisk, 0.0, 1.0),
            0.0,
            1.0
        )
        let directionConfidence = fxClamp(
            0.35 * directionGap +
                0.18 * fxClamp(confidence, 0.0, 1.0) +
                0.14 * fxClamp(reliability, 0.0, 1.0) +
                0.12 * directionalAgreement +
                0.10 * abs(foundationBias) +
                0.06 * abs(studentBias) +
                0.05 * abs(analogBias),
            0.0,
            1.0
        )
        let moveAdequacy = fxClamp(
            0.34 * moveRatio +
                0.20 * fxClamp(foundation.moveRatio / 2.0, 0.0, 1.0) +
                0.18 * fxClamp(student.moveRatio / 2.0, 0.0, 1.0) +
                0.16 * fxClamp(0.5 + 0.5 * analog.edgeNorm, 0.0, 1.0) +
                0.12 * (1.0 - skipProbability),
            0.0,
            1.0
        )
        let pathQuality = fxClamp(
            0.34 * (1.0 - fxClamp(pathRisk, 0.0, 1.0)) +
                0.20 * fxClamp(analog.pathSafety, 0.0, 1.0) +
                0.18 * fxClamp(reliability, 0.0, 1.0) +
                0.14 * (1.0 - fxClamp(hitTimeFraction, 0.0, 1.0)) +
                0.14 * fxClamp(0.5 + 0.5 * contextQuality, 0.0, 1.0),
            0.0,
            1.0
        )
        let executionViability = fxClamp(
            0.40 * (1.0 - fxClamp(fillRisk, 0.0, 1.0)) +
                0.24 * fxClamp(analog.executionSafety, 0.0, 1.0) +
                0.16 * fxClamp(foundation.tradability, 0.0, 1.0) +
                0.10 * fxClamp(student.tradability, 0.0, 1.0) +
                0.10 * fxClamp(reliability, 0.0, 1.0),
            0.0,
            1.0
        )
        let horizonScale = fxClamp(Double(max(horizonMinutes, 1)) / 240.0, 0.0, 1.0)
        let horizonFit = fxClamp(
            0.34 * fxClamp(student.horizonFit, 0.0, 1.0) +
                0.18 * (1.0 - abs(fxClamp(hitTimeFraction, 0.0, 1.0) - 0.45)) +
                0.16 * (1.0 - 0.55 * fxClamp(foundation.regimeTransition, 0.0, 1.0)) +
                0.14 * fxClamp(analog.quality, 0.0, 1.0) +
                0.10 * fxClamp(1.0 - horizonScale, 0.0, 1.0) +
                0.08 * fxClamp(foundation.contextAlignment, 0.0, 1.0),
            0.0,
            1.0
        )
        let consistency = fxClamp(
            0.12 +
                0.20 * tradability +
                0.18 * directionConfidence +
                0.16 * moveAdequacy +
                0.14 * pathQuality +
                0.10 * executionViability +
                0.10 * horizonFit +
                0.10 * directionalAgreement,
            0.0,
            1.0
        )
        let score = fxClamp(
            (tradability + directionConfidence + moveAdequacy + pathQuality + executionViability + horizonFit + consistency) / 7.0,
            0.0,
            1.0
        )

        return HierarchicalSignals(
            tradability: tradability,
            directionConfidence: directionConfidence,
            moveAdequacy: moveAdequacy,
            pathQuality: pathQuality,
            executionViability: executionViability,
            horizonFit: horizonFit,
            consistency: consistency,
            score: score
        )
    }
}
