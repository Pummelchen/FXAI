import Foundation

public struct AuditAdversarialConfiguration: Codable, Hashable, Sendable {
    public var bars: Int
    public var horizonMinutes: Int
    public var pointValue: Double
    public var priceCostPoints: Double
    public var evThresholdPoints: Double
    public var normalizationMethod: FeatureNormalizationMethod
    public var maxEvaluationSamples: Int
    public var minEvaluationSamples: Int?

    public init(
        bars: Int,
        horizonMinutes: Int,
        pointValue: Double = 0.0001,
        priceCostPoints: Double = 0.0,
        evThresholdPoints: Double = 0.25,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        maxEvaluationSamples: Int = Int.max,
        minEvaluationSamples: Int? = nil
    ) {
        self.bars = max(64, bars)
        self.horizonMinutes = TrainingSampleTools.clampHorizon(horizonMinutes)
        let sanitizedPoint = fxSafeFinite(pointValue)
        self.pointValue = sanitizedPoint > 0.0 ? sanitizedPoint : 0.0001
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.evThresholdPoints = max(0.0, fxSafeFinite(evThresholdPoints))
        self.normalizationMethod = normalizationMethod
        self.maxEvaluationSamples = max(0, maxEvaluationSamples)
        self.minEvaluationSamples = minEvaluationSamples.map { max(1, $0) }
    }
}

public enum AuditAdversarialTools {
    public static func generateScenario<Plugin: FXAIPluginV4>(
        plugin: inout Plugin,
        marketSeries: M1OHLCVSeries,
        configuration: AuditAdversarialConfiguration,
        hyperParameters: HyperParameters = HyperParameters()
    ) throws -> AuditGeneratedScenarioSeries? {
        let manifest = plugin.manifest
        try manifest.validate()
        guard marketSeries.count >= configuration.bars + 64,
              configuration.maxEvaluationSamples > 0 else {
            return nil
        }

        let multipliedBars = configuration.bars > Int.max / 4
            ? Int.max
            : configuration.bars * 4
        let expandedBars = configuration.bars > Int.max - 512
            ? Int.max
            : configuration.bars + 512
        let searchBars = min(marketSeries.count, max(multipliedBars, expandedBars))
        guard searchBars >= configuration.bars + 64 else { return nil }

        let searchRange = (marketSeries.count - searchBars)..<marketSeries.count
        let searchChronological = AuditScenarioTools.marketScenarioBars(
            from: marketSeries,
            range: searchRange,
            point: configuration.pointValue
        )
        guard searchChronological.count == searchBars,
              let baseGenerated = AuditScenarioTools.generatedScenarioSeries(
                chronologicalBars: searchChronological,
                point: configuration.pointValue
              ) else {
            return nil
        }

        plugin.reset()
        var weakness = Array(repeating: 0.0, count: baseGenerated.primary.count)
        let horizon = configuration.horizonMinutes
        let startIndex = horizon + 1
        var endIndex = baseGenerated.primary.count - 220
        if endIndex <= startIndex { endIndex = baseGenerated.primary.count - 32 }
        if endIndex <= startIndex { endIndex = baseGenerated.primary.count - 2 }
        guard endIndex > startIndex else { return nil }

        for index in candidateIndexes(
            start: startIndex,
            end: endIndex,
            maxSamples: configuration.maxEvaluationSamples
        ) {
            guard let sample = try? AuditSampleTools.buildSample(
                generated: baseGenerated,
                sampleIndexAsSeries: index,
                horizonMinutes: horizon,
                manifest: manifest,
                pointValue: configuration.pointValue,
                priceCostPoints: configuration.priceCostPoints,
                evThresholdPoints: configuration.evThresholdPoints,
                normalizationMethod: configuration.normalizationMethod
            ) else {
                continue
            }

            let macroActivity = macroActivity(from: sample.payload.payloadFrame.x)
            let prediction = validatedPrediction {
                let request = sample.predictRequest
                try request.validate()
                return try plugin.predict(request, hyperParameters: hyperParameters)
            }
            if let prediction {
                weakness[index] = AuditScoringTools.adversarialWeaknessScore(
                    labelClass: sample.payload.sample.labelClass,
                    movePoints: sample.payload.sample.movePoints,
                    minMovePoints: sample.payload.sample.minMovePoints,
                    mfePoints: sample.payload.sample.mfePoints,
                    maePoints: sample.payload.sample.maePoints,
                    timeToHitFraction: sample.payload.sample.timeToHitFraction,
                    pathFlags: sample.payload.sample.pathFlags,
                    fillRisk: sample.payload.sample.fillRisk,
                    macroActivity: macroActivity,
                    sampleTimeUTC: sample.payload.sample.sampleTimeUTC,
                    prediction: prediction
                )
            } else {
                weakness[index] = invalidPredictionWeakness(
                    fillRisk: sample.payload.sample.fillRisk,
                    macroActivity: macroActivity,
                    sampleTimeUTC: sample.payload.sample.sampleTimeUTC
                )
            }

            try sample.trainRequest.validate()
            try plugin.train(sample.trainRequest, hyperParameters: hyperParameters)
        }

        guard let bestStart = bestWindowStart(
            weakness: weakness,
            bars: configuration.bars,
            minEvaluationSamples: resolvedMinEvaluationSamples(configuration)
        ) else {
            return AuditScenarioTools.generateMarketScenarioSeries(
                spec: AuditScenarioTools.scenarioSpec(scenarioID: 8),
                marketSeries: marketSeries,
                bars: configuration.bars,
                point: configuration.pointValue,
                applyWorldPlan: false
            )
        }

        let selectedAsSeries = barsAsSeries(from: baseGenerated.primary, range: bestStart..<(bestStart + configuration.bars))
        let selectedChronological = Array(selectedAsSeries.reversed())
        return AuditScenarioTools.generatedScenarioSeries(
            chronologicalBars: selectedChronological,
            point: configuration.pointValue
        )
    }

    private static func candidateIndexes(start: Int, end: Int, maxSamples: Int) -> [Int] {
        let total = max(0, end - start)
        guard total > 0, maxSamples > 0 else { return [] }
        if maxSamples >= total {
            return Array(start..<end)
        }
        if maxSamples == 1 {
            return [start]
        }
        var output: [Int] = []
        output.reserveCapacity(maxSamples)
        var previous = -1
        for sample in 0..<maxSamples {
            let offset = (sample * (total - 1)) / (maxSamples - 1)
            let index = start + offset
            if index != previous {
                output.append(index)
                previous = index
            }
        }
        return output
    }

    private static func validatedPrediction(_ body: () throws -> PredictionV4) -> PredictionV4? {
        do {
            let prediction = try body()
            try prediction.validate()
            return prediction
        } catch {
            return nil
        }
    }

    private static func macroActivity(from x: [Double]) -> Double {
        let offset = FXDataEngineConstants.macroEventFeatureOffset
        return max(
            modelInputValue(x, index: offset + 2),
            max(modelInputValue(x, index: offset), modelInputValue(x, index: offset + 1))
        )
    }

    private static func modelInputValue(_ x: [Double], index: Int) -> Double {
        guard index >= 0, index < x.count else { return 0.0 }
        return fxSafeFinite(x[index])
    }

    private static func invalidPredictionWeakness(
        fillRisk: Double,
        macroActivity: Double,
        sampleTimeUTC: Int64
    ) -> Double {
        3.5 +
            0.25 * fxClamp(fillRisk, 0.0, 4.0) +
            0.15 * AuditScoringTools.sessionEdgePressure(sampleTimeUTC: sampleTimeUTC) +
            0.15 * fxClamp(macroActivity, 0.0, 1.0)
    }

    private static func resolvedMinEvaluationSamples(_ configuration: AuditAdversarialConfiguration) -> Int {
        if let minEvaluationSamples = configuration.minEvaluationSamples {
            return min(max(1, minEvaluationSamples), configuration.bars)
        }
        return min(max(96, configuration.bars / 6), configuration.bars)
    }

    private static func bestWindowStart(
        weakness: [Double],
        bars: Int,
        minEvaluationSamples: Int
    ) -> Int? {
        guard bars > 0, weakness.count >= bars else { return nil }
        var prefixWeakness = Array(repeating: 0.0, count: weakness.count + 1)
        var prefixTailWeakness = Array(repeating: 0.0, count: weakness.count + 1)
        var prefixValid = Array(repeating: 0, count: weakness.count + 1)
        var prefixTailHits = Array(repeating: 0, count: weakness.count + 1)
        for index in weakness.indices {
            let value = fxSafeFinite(weakness[index])
            let valid = value > 0.0
            let tail = value > 1.25
            prefixWeakness[index + 1] = prefixWeakness[index] + value
            prefixTailWeakness[index + 1] = prefixTailWeakness[index] + (tail ? value : 0.0)
            prefixValid[index + 1] = prefixValid[index] + (valid ? 1 : 0)
            prefixTailHits[index + 1] = prefixTailHits[index] + (tail ? 1 : 0)
        }

        var bestStart: Int?
        var bestScore = -Double.greatestFiniteMagnitude
        let maxStart = max(0, weakness.count - bars)
        for start in 0...maxStart {
            let end = start + bars
            let valid = prefixValid[end] - prefixValid[start]
            guard valid >= minEvaluationSamples else { continue }
            let sumWeakness = prefixWeakness[end] - prefixWeakness[start]
            let tailWeakness = prefixTailWeakness[end] - prefixTailWeakness[start]
            let tailHits = prefixTailHits[end] - prefixTailHits[start]
            let meanWeakness = sumWeakness / Double(max(valid, 1))
            let tailDensity = Double(tailHits) / Double(max(valid, 1))
            let tailShare = tailWeakness / max(sumWeakness, 1e-6)
            let recentBonus = 1.0 - Double(start) / Double(max(maxStart, 1))
            let score = 0.76 * meanWeakness +
                0.16 * tailDensity +
                0.08 * tailShare +
                0.03 * recentBonus
            if score > bestScore {
                bestScore = score
                bestStart = start
            }
        }
        return bestStart
    }

    private static func barsAsSeries(
        from series: AuditAsSeriesOHLCV,
        range: Range<Int>
    ) -> [AuditScenarioDoubleBar] {
        guard range.lowerBound >= 0, range.upperBound <= series.count else { return [] }
        var output: [AuditScenarioDoubleBar] = []
        output.reserveCapacity(range.count)
        for index in range {
            output.append(AuditScenarioDoubleBar(
                timestampUTC: series.timeUTC[index],
                open: series.open[index],
                high: series.high[index],
                low: series.low[index],
                close: series.close[index],
                volume: series.volume[index],
                fillRiskPoints: series.fillRiskPoints[index]
            ))
        }
        return output
    }
}
