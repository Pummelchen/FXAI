import Foundation

public struct AuditPreparedSample: Sendable {
    public var payload: PreparedTrainingPayload
    public var traceStats: ExecutionTraceStats
    public var sampleIndexAsSeries: Int
    public var sampleIndexAscending: Int

    public var predictRequest: PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: payload.context,
            windowSize: payload.payloadFrame.windowSize,
            x: payload.payloadFrame.x,
            xWindow: payload.payloadFrame.xWindow
        )
    }

    public var trainRequest: TrainRequestV4 {
        payload.trainRequest
    }

    public init(
        payload: PreparedTrainingPayload,
        traceStats: ExecutionTraceStats,
        sampleIndexAsSeries: Int,
        sampleIndexAscending: Int
    ) {
        self.payload = payload
        self.traceStats = traceStats
        self.sampleIndexAsSeries = max(0, sampleIndexAsSeries)
        self.sampleIndexAscending = max(0, sampleIndexAscending)
    }
}

public enum AuditSampleTools {
    public static func marketUniverse(
        from generated: AuditGeneratedScenarioSeries,
        symbol: String = "EURUSD",
        pointValue: Double = 0.0001
    ) throws -> MarketUniverse {
        let primarySymbol = DataCoreRequest.normalizedSymbol(symbol).isEmpty
            ? "EURUSD"
            : DataCoreRequest.normalizedSymbol(symbol)
        var series: [M1OHLCVSeries] = [
            try m1Series(
                from: generated.primary,
                symbol: primarySymbol,
                providerSymbol: primarySymbol,
                pointValue: pointValue
            )
        ]
        for (offset, context) in generated.contexts.prefix(FXDataEngineConstants.maxContextSymbols).enumerated() {
            let contextSymbol = "AUDITCTX\(offset + 1)"
            series.append(try m1Series(
                from: context,
                symbol: contextSymbol,
                providerSymbol: contextSymbol,
                pointValue: pointValue
            ))
        }
        return try MarketUniverse(primarySymbol: primarySymbol, series: series, requireAlignedTimestamps: true)
    }

    public static func buildSample(
        generated: AuditGeneratedScenarioSeries,
        sampleIndexAsSeries: Int,
        horizonMinutes: Int,
        manifest: PluginManifestV4,
        symbol: String = "EURUSD",
        pointValue: Double = 0.0001,
        priceCostPoints: Double = 0.0,
        evThresholdPoints: Double = 0.25,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        tradeKillerMinutes: Int? = nil,
        pipeline: FXDataEnginePipeline = FXDataEnginePipeline()
    ) throws -> AuditPreparedSample {
        let universe = try marketUniverse(from: generated, symbol: symbol, pointValue: pointValue)
        let count = universe.primary.count
        guard sampleIndexAsSeries >= 0, sampleIndexAsSeries < count else {
            throw FXDataEngineError.invalidRequest("audit sample index \(sampleIndexAsSeries) is outside 0..<\(count)")
        }
        let horizon = TrainingSampleTools.clampHorizon(horizonMinutes)
        guard sampleIndexAsSeries - horizon >= 0 else {
            throw FXDataEngineError.insufficientData("audit sample index \(sampleIndexAsSeries) does not leave \(horizon) future as-series bars")
        }

        let ascendingIndex = count - 1 - sampleIndexAsSeries
        let sequenceBars = manifest.resolvedSequenceBars(horizonMinutes: horizon)
        let neededBars = min(max(ascendingIndex + 1, 1), max(256, sequenceBars + 1))
        let contextSymbols = universe.symbols.filter { $0 != universe.primarySymbol }
        let request = DataCoreRequest(
            liveMode: false,
            symbol: universe.primarySymbol,
            neededBars: neededBars,
            alignUpToIndex: ascendingIndex,
            contextSymbols: contextSymbols
        )
        let payload = try pipeline.prepareTrainPayload(
            universe: universe,
            request: request,
            manifest: manifest,
            horizonMinutes: horizon,
            roundTripCostPoints: priceCostPoints,
            evThresholdPoints: evThresholdPoints,
            normalizationMethod: normalizationMethod,
            tradeKillerMinutes: tradeKillerMinutes
        )
        let traceStats = ExecutionReplayTools.buildTraceStats(
            series: universe.primary,
            index: ascendingIndex,
            horizonMinutes: horizon
        )
        return AuditPreparedSample(
            payload: payload,
            traceStats: traceStats,
            sampleIndexAsSeries: sampleIndexAsSeries,
            sampleIndexAscending: ascendingIndex
        )
    }

    private static func m1Series(
        from asSeries: AuditAsSeriesOHLCV,
        symbol: String,
        providerSymbol: String,
        pointValue: Double
    ) throws -> M1OHLCVSeries {
        guard asSeries.isConsistent, asSeries.count > 0 else {
            throw FXDataEngineError.invalidRequest("audit series for \(symbol) is empty or inconsistent")
        }
        let digits = digitsForPoint(pointValue)
        let scale = pow(10.0, Double(digits))
        let count = asSeries.count
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        utc.reserveCapacity(count)
        open.reserveCapacity(count)
        high.reserveCapacity(count)
        low.reserveCapacity(count)
        close.reserveCapacity(count)
        volume.reserveCapacity(count)

        for asIndex in stride(from: count - 1, through: 0, by: -1) {
            let scaledOpen = scaledPrice(asSeries.open[asIndex], scale: scale)
            let scaledClose = scaledPrice(asSeries.close[asIndex], scale: scale)
            let scaledHigh = max(
                scaledPrice(asSeries.high[asIndex], scale: scale),
                max(scaledOpen, scaledClose)
            )
            let scaledLow = max(
                1,
                min(scaledPrice(asSeries.low[asIndex], scale: scale), min(scaledOpen, scaledClose))
            )
            utc.append(asSeries.timeUTC[asIndex])
            open.append(scaledOpen)
            high.append(scaledHigh)
            low.append(scaledLow)
            close.append(scaledClose)
            volume.append(scaledVolume(asSeries.volume[asIndex]))
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "audit",
                sourceOrigin: "AUDIT",
                logicalSymbol: symbol,
                providerSymbol: providerSymbol,
                digits: digits,
                firstUTC: utc.first,
                lastUTC: utc.last
            ),
            utcTimestamps: utc,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }

    private static func digitsForPoint(_ pointValue: Double) -> Int {
        guard pointValue.isFinite, pointValue > 0.0 else { return 4 }
        let digits = Int((-log10(pointValue)).rounded())
        return min(max(digits, 0), 10)
    }

    private static func scaledPrice(_ value: Double, scale: Double) -> Int64 {
        guard value.isFinite, value > 0.0, scale.isFinite, scale > 0.0 else { return 1 }
        let scaled = (value * scale).rounded()
        if scaled >= Double(Int64.max) { return Int64.max }
        return max(1, Int64(scaled))
    }

    private static func scaledVolume(_ value: Double) -> UInt64 {
        guard value.isFinite, value > 0.0 else { return 0 }
        if value >= Double(UInt64.max) { return UInt64.max }
        return UInt64(value.rounded())
    }
}
