import Foundation

public struct RegimeGraphQuery: Codable, Hashable, Sendable {
    public var persistence: Double
    public var transitionConfidence: Double
    public var instability: Double
    public var edgeBias: Double
    public var qualityBias: Double
    public var macroAlignment: Double
    public var predictedRegime: Int

    public init(
        persistence: Double = 0.0,
        transitionConfidence: Double = 0.0,
        instability: Double = 0.0,
        edgeBias: Double = 0.0,
        qualityBias: Double = 0.0,
        macroAlignment: Double = 0.0,
        predictedRegime: Int = 0
    ) {
        self.persistence = persistence
        self.transitionConfidence = transitionConfidence
        self.instability = instability
        self.edgeBias = edgeBias
        self.qualityBias = qualityBias
        self.macroAlignment = macroAlignment
        self.predictedRegime = predictedRegime
    }
}

public struct RegimeGraphState: Codable, Sendable {
    public private(set) var ready: Bool
    public private(set) var lastRegime: Int
    public private(set) var lastTimeUTC: Int64
    public private(set) var regimeCount: Int
    public private(set) var dwellEMA: [Double]
    public private(set) var outboundMass: [Double]
    public private(set) var transitionObservations: [Double]
    public private(set) var transitionEdge: [Double]
    public private(set) var transitionQuality: [Double]
    public private(set) var macroAlignment: [Double]

    public init(regimeCount: Int = FXDataEngineConstants.pluginRegimeBuckets) {
        let count = max(1, regimeCount)
        self.ready = false
        self.lastRegime = -1
        self.lastTimeUTC = 0
        self.regimeCount = count
        self.dwellEMA = Array(repeating: 0.0, count: count)
        self.outboundMass = Array(repeating: 0.0, count: count)
        self.transitionObservations = Array(repeating: 0.0, count: count * count)
        self.transitionEdge = Array(repeating: 0.0, count: count * count)
        self.transitionQuality = Array(repeating: 0.0, count: count * count)
        self.macroAlignment = Array(repeating: 0.0, count: count * count)
    }

    public mutating func reset() {
        self = RegimeGraphState(regimeCount: regimeCount)
    }

    public mutating func recordState(
        regimeID: Int,
        sampleTimeUTC: Int64,
        macroQuality: Double
    ) {
        ensureStorage()
        let regime = sanitizeRegime(regimeID)
        let timestamp = max(sampleTimeUTC, 0)

        if lastRegime >= 0,
           lastRegime < regimeCount,
           lastTimeUTC > 0,
           timestamp > lastTimeUTC {
            let previous = lastRegime
            let transitionIndex = index(from: previous, to: regime)
            let observations = transitionObservations[transitionIndex]
            let alpha = fxClamp(0.14 / sqrt(1.0 + 0.02 * observations), 0.01, 0.14)
            let dwellBars = max(Int((timestamp - lastTimeUTC) / 60), 1)
            let dwellNorm = fxClamp(Double(dwellBars) / 30.0, 0.0, 1.0)

            transitionObservations[transitionIndex] += 1.0
            outboundMass[previous] += 1.0
            if previous == regime {
                dwellEMA[previous] = (1.0 - alpha) * dwellEMA[previous] + alpha * dwellNorm
            } else {
                dwellEMA[previous] = (1.0 - alpha) * dwellEMA[previous] + alpha * 0.25 * dwellNorm
            }

            let macro = fxClamp(macroQuality, 0.0, 1.0)
            macroAlignment[transitionIndex] = (1.0 - alpha) * macroAlignment[transitionIndex] + alpha * macro
            ready = true
        }

        lastRegime = regime
        lastTimeUTC = timestamp
    }

    public mutating func updateFeedback(
        fromRegime: Int,
        toRegime: Int,
        realizedEdgePoints: Double,
        qualityScore: Double,
        macroQuality: Double
    ) {
        ensureStorage()
        guard fromRegime >= 0, fromRegime < regimeCount else { return }
        let destination = toRegime >= 0 && toRegime < regimeCount ? toRegime : fromRegime
        let transitionIndex = index(from: fromRegime, to: destination)
        let observations = transitionObservations[transitionIndex]
        let alpha = fxClamp(0.16 / sqrt(1.0 + 0.03 * observations), 0.01, 0.16)
        let edgeNorm = fxClamp(realizedEdgePoints / max(abs(realizedEdgePoints), 1.0), -1.0, 1.0)
        let quality = fxClamp(qualityScore, 0.0, 2.0) / 2.0
        let macro = fxClamp(macroQuality, 0.0, 1.0)

        transitionEdge[transitionIndex] = (1.0 - alpha) * transitionEdge[transitionIndex] + alpha * edgeNorm
        transitionQuality[transitionIndex] = (1.0 - alpha) * transitionQuality[transitionIndex] + alpha * quality
        macroAlignment[transitionIndex] = (1.0 - alpha) * macroAlignment[transitionIndex] + alpha * macro
        ready = true
    }

    public func query(regimeID: Int, macroQuality: Double) -> RegimeGraphQuery {
        let regime = regimeID >= 0 && regimeID < regimeCount ? regimeID : 0
        let total = HorizonTools.value(in: outboundMass, index: regime, default: 0.0)
        if total <= 1e-6 {
            return RegimeGraphQuery(
                persistence: fxClamp(HorizonTools.value(in: dwellEMA, index: regime, default: 0.0), 0.0, 1.0),
                predictedRegime: regime
            )
        }

        var bestMass = -1.0
        var bestRegime = regime
        var edgeAccumulator = 0.0
        var qualityAccumulator = 0.0
        var macroAccumulator = 0.0

        for next in 0..<regimeCount {
            let transitionIndex = index(from: regime, to: next)
            let observations = HorizonTools.value(in: transitionObservations, index: transitionIndex, default: 0.0)
            if observations > bestMass {
                bestMass = observations
                bestRegime = next
            }
            let probability = observations / total
            edgeAccumulator += probability * HorizonTools.value(in: transitionEdge, index: transitionIndex, default: 0.0)
            qualityAccumulator += probability * HorizonTools.value(in: transitionQuality, index: transitionIndex, default: 0.0)
            macroAccumulator += probability * HorizonTools.value(in: macroAlignment, index: transitionIndex, default: 0.0)
        }

        let selfTransitionIndex = index(from: regime, to: regime)
        let persistence = fxClamp(
            HorizonTools.value(in: transitionObservations, index: selfTransitionIndex, default: 0.0) / total,
            0.0,
            1.0
        )
        return RegimeGraphQuery(
            persistence: persistence,
            transitionConfidence: fxClamp(bestMass / total, 0.0, 1.0),
            instability: fxClamp(1.0 - persistence, 0.0, 1.0),
            edgeBias: fxClamp(edgeAccumulator, -1.0, 1.0),
            qualityBias: fxClamp(qualityAccumulator, 0.0, 1.0),
            macroAlignment: fxClamp(0.70 * macroAccumulator + 0.30 * fxClamp(macroQuality, 0.0, 1.0), 0.0, 1.0),
            predictedRegime: bestRegime
        )
    }

    private mutating func ensureStorage() {
        regimeCount = max(1, regimeCount)
        ensureVector(&dwellEMA, count: regimeCount)
        ensureVector(&outboundMass, count: regimeCount)
        ensureVector(&transitionObservations, count: regimeCount * regimeCount)
        ensureVector(&transitionEdge, count: regimeCount * regimeCount)
        ensureVector(&transitionQuality, count: regimeCount * regimeCount)
        ensureVector(&macroAlignment, count: regimeCount * regimeCount)
        if lastRegime >= regimeCount {
            lastRegime = -1
        }
        if lastTimeUTC < 0 {
            lastTimeUTC = 0
        }
    }

    private func sanitizeRegime(_ regimeID: Int) -> Int {
        regimeID >= 0 && regimeID < regimeCount ? regimeID : 0
    }

    private func index(from: Int, to: Int) -> Int {
        sanitizeRegime(from) * regimeCount + sanitizeRegime(to)
    }

    private func ensureVector(_ vector: inout [Double], count: Int) {
        if vector.count < count {
            vector.append(contentsOf: Array(repeating: 0.0, count: count - vector.count))
        } else if vector.count > count {
            vector.removeLast(vector.count - count)
        }
    }
}
