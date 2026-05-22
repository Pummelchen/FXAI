import Foundation

public struct FeatureCoreRequest: Codable, Hashable, Sendable {
    public var sampleIndex: Int
    public var horizonMinutes: Int
    public var normalizationMethod: FeatureNormalizationMethod

    public init(
        sampleIndex: Int,
        horizonMinutes: Int = 1,
        normalizationMethod: FeatureNormalizationMethod = .existing
    ) {
        self.sampleIndex = sampleIndex
        self.horizonMinutes = max(1, horizonMinutes)
        self.normalizationMethod = normalizationMethod
    }
}

public struct FeatureCoreFrame: Sendable {
    public let valid: Bool
    public let sampleIndex: Int
    public let horizonMinutes: Int
    public let normalizationMethod: FeatureNormalizationMethod
    public let sampleTimeUTC: Int64
    public let hasVolume: Bool
    public let hasPrevious: Bool
    public let raw: [Double]
    public let previous: [Double]

    public init(
        valid: Bool,
        sampleIndex: Int,
        horizonMinutes: Int,
        normalizationMethod: FeatureNormalizationMethod,
        sampleTimeUTC: Int64,
        hasVolume: Bool,
        hasPrevious: Bool,
        raw: [Double],
        previous: [Double]
    ) {
        self.valid = valid
        self.sampleIndex = sampleIndex
        self.horizonMinutes = horizonMinutes
        self.normalizationMethod = normalizationMethod
        self.sampleTimeUTC = sampleTimeUTC
        self.hasVolume = hasVolume
        self.hasPrevious = hasPrevious
        self.raw = raw
        self.previous = previous
    }
}

public struct FeatureCore: Sendable {
    private let registry: FeatureRegistry

    public init(registry: FeatureRegistry = FeatureRegistry()) {
        self.registry = registry
    }

    public func buildFrame(bundle: DataCoreBundle, request: FeatureCoreRequest? = nil) throws -> FeatureCoreFrame {
        guard bundle.ready else {
            throw FXDataEngineError.invalidRequest("data bundle is not ready")
        }

        let sampleIndex = request?.sampleIndex ?? bundle.sampleIndex
        guard sampleIndex >= 0, sampleIndex < bundle.primary.count else {
            throw FXDataEngineError.invalidRequest("sample index is outside primary series")
        }

        let raw = buildFeatureVector(bundle: bundle, sampleIndex: sampleIndex)
        let previousIndex = sampleIndex > 0 ? sampleIndex - 1 : -1
        let previous = previousIndex >= 0
            ? buildFeatureVector(bundle: bundle, sampleIndex: previousIndex)
            : Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        return FeatureCoreFrame(
            valid: true,
            sampleIndex: sampleIndex,
            horizonMinutes: max(1, request?.horizonMinutes ?? 1),
            normalizationMethod: request?.normalizationMethod ?? .existing,
            sampleTimeUTC: bundle.primary.utcTimestamps[sampleIndex],
            hasVolume: Self.hasUsableVolume(bundle.universe),
            hasPrevious: previousIndex >= 0,
            raw: raw,
            previous: previous
        )
    }

    public func buildFeatureVector(bundle: DataCoreBundle, sampleIndex: Int? = nil) -> [Double] {
        let resolvedSampleIndex = sampleIndex ?? bundle.sampleIndex
        return buildFeatureVector(
            universe: bundle.universe,
            sampleIndex: resolvedSampleIndex,
            contextAggregates: resolvedSampleIndex <= bundle.sampleIndex ? bundle.contextAggregates : nil
        )
    }

    public func buildFeatureVector(universe: MarketUniverse, sampleIndex: Int) -> [Double] {
        let contextSymbols = universe.contextSeries().map(\.metadata.logicalSymbol)
        let contextAggregates = DataCoreContextAggregator.build(
            universe: universe,
            upToIndex: sampleIndex,
            contextSymbols: contextSymbols
        )
        return buildFeatureVector(
            universe: universe,
            sampleIndex: sampleIndex,
            contextAggregates: contextAggregates
        )
    }

    private func buildFeatureVector(
        universe: MarketUniverse,
        sampleIndex: Int,
        contextAggregates: DataCoreContextAggregates?
    ) -> [Double] {
        let primary = universe.primary
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        guard sampleIndex >= 0, sampleIndex < primary.count else { return features }

        let hasVolume = Self.hasUsableVolume(universe)
        fillPriceFeatures(&features, series: primary, index: sampleIndex)
        fillContextFeatures(&features, universe: universe, index: sampleIndex, contextAggregates: contextAggregates)
        fillTimeFeatures(&features, timestamp: primary.utcTimestamps[sampleIndex])
        fillOHLCGeometryFeatures(&features, series: primary, index: sampleIndex)
        fillMovingAverageFeatures(&features, series: primary, index: sampleIndex)
        fillVolatilityFeatures(&features, series: primary, index: sampleIndex)
        fillFilterFeatures(&features, series: primary, index: sampleIndex)
        fillVolumeAwareMicrostructureFeatures(&features, universe: universe, index: sampleIndex, hasVolume: hasVolume)
        fillMainMTFFeatures(&features, series: primary, index: sampleIndex, hasVolume: hasVolume)
        fillContextMTFFeatures(&features, universe: universe, index: sampleIndex, hasVolume: hasVolume)
        fillFeatureFamilyDrift(&features)
        return features.map { fxSafeFinite($0) }
    }

    public static func hasUsableVolume(_ universe: MarketUniverse) -> Bool {
        universe.seriesBySymbol.values.contains(where: \.hasVolume)
    }

    private func fillPriceFeatures(_ features: inout [Double], series: M1OHLCVSeries, index: Int) {
        features[0] = normalizedReturn(series, index, 1)
        features[1] = normalizedReturn(series, index, 3)
        features[2] = normalizedReturn(series, index, 5)
        features[3] = closeSlope(series, index, window: 10)
        features[4] = closeZScore(series, index, window: 10)
        features[5] = returnStd(series, index, window: 10)
    }

    private func fillContextFeatures(
        _ features: inout [Double],
        universe: MarketUniverse,
        index: Int,
        contextAggregates: DataCoreContextAggregates?
    ) {
        if let contextAggregates,
           contextAggregates.hasCompleteSample(index) {
            fillAggregatedContextFeatures(
                &features,
                primary: universe.primary,
                index: index,
                contextAggregates: contextAggregates
            )
            return
        }

        let contexts = universe.contextSeries().prefix(FXDataEngineConstants.contextTopSymbols)
        var returns: [Double] = []
        var topRows: [(series: M1OHLCVSeries, ret: Double, corr: Double)] = []
        for series in contexts where index < series.count {
            let ret = normalizedReturn(series, index, 1)
            let corr = rollingCorrelation(universe.primary, series, index, window: 30)
            returns.append(ret)
            topRows.append((series, ret, corr))
        }
        features[10] = returns.mean
        features[11] = returns.standardDeviation
        features[12] = returns.isEmpty ? 0.5 : Double(returns.filter { $0 > 0 }.count) / Double(returns.count)

        topRows.sort { abs($0.ret) > abs($1.ret) }
        for slot in 0..<min(topRows.count, FXDataEngineConstants.contextTopSymbols) {
            let base = 50 + slot * 4
            let row = topRows[slot]
            features[base] = row.ret
            features[base + 1] = fxClamp(Double(slot + 1) / Double(FXDataEngineConstants.contextTopSymbols), 0.0, 1.0)
            features[base + 2] = fxClampSignedUnit(row.ret - features[0])
            features[base + 3] = row.corr
        }
        features[62] = fxClamp(abs(features[10]) + features[11], 0.0, 1.0)
        features[63] = fxClamp(1.0 - features[11], 0.0, 1.0)
        features[64] = features[10] == 0 ? 0.0 : fxClampSignedUnit(features[10] - features[0])
        features[65] = fxClamp(Double(returns.count) / Double(FXDataEngineConstants.contextTopSymbols), 0.0, 1.0)
    }

    private func fillAggregatedContextFeatures(
        _ features: inout [Double],
        primary: M1OHLCVSeries,
        index: Int,
        contextAggregates: DataCoreContextAggregates
    ) {
        let volatilityUnit = max(rollingAbsRawReturn(primary, index, window: 20), 1e-6)

        features[10] = contextAggregates.mean[index] / volatilityUnit
        features[11] = contextAggregates.standardDeviation[index] / volatilityUnit
        features[12] = fxClampSignedUnit((contextAggregates.upRatio[index] - 0.5) * 2.0)

        for slot in 0..<FXDataEngineConstants.contextTopSymbols {
            let base = 50 + slot * 4
            let extraBase = slot * FXDataEngineConstants.contextBaseSymbolFeatures
            features[base] = contextAggregates.extraValue(sampleIndex: index, featureIndex: extraBase) / volatilityUnit
            features[base + 1] = contextAggregates.extraValue(sampleIndex: index, featureIndex: extraBase + 1) / volatilityUnit
            features[base + 2] = contextAggregates.extraValue(sampleIndex: index, featureIndex: extraBase + 2) / volatilityUnit
            features[base + 3] = fxClampSignedUnit(
                contextAggregates.extraValue(sampleIndex: index, featureIndex: extraBase + 3)
            )
        }

        let sharedOffset = FXDataEngineConstants.contextSharedOffset
        let sharedUtility = contextAggregates.extraValue(sampleIndex: index, featureIndex: sharedOffset)
        let sharedStability = contextAggregates.extraValue(sampleIndex: index, featureIndex: sharedOffset + 1, default: 0.5)
        let sharedLead = contextAggregates.extraValue(sampleIndex: index, featureIndex: sharedOffset + 2, default: 0.5)
        let sharedCoverage = contextAggregates.extraValue(sampleIndex: index, featureIndex: sharedOffset + 3)
        features[62] = fxClampSignedUnit(sharedUtility)
        features[63] = fxClampSignedUnit((sharedStability * 2.0) - 1.0)
        features[64] = fxClampSignedUnit((sharedLead * 2.0) - 1.0)
        features[65] = fxClampSignedUnit((sharedCoverage * 2.0) - 1.0)
    }

    private func fillTimeFeatures(_ features: inout [Double], timestamp: Int64) {
        let date = Date(timeIntervalSince1970: TimeInterval(timestamp))
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let weekday = calendar.component(.weekday, from: date)
        let hour = calendar.component(.hour, from: date)
        let minute = calendar.component(.minute, from: date)
        features[15] = fxClamp((Double(weekday) - 1.0) / 6.0, 0.0, 1.0)
        features[16] = fxClamp(Double(hour) / 23.0, 0.0, 1.0)
        features[17] = fxClamp(Double(minute) / 59.0, 0.0, 1.0)
        features[72] = sessionTransitionScore(hour: hour, minute: minute)
        features[73] = sessionOverlapScore(hour: hour)
    }

    private func fillOHLCGeometryFeatures(_ features: inout [Double], series: M1OHLCVSeries, index: Int) {
        let open = Double(series.open[index])
        let high = Double(series.high[index])
        let low = Double(series.low[index])
        let close = Double(series.close[index])
        let range = max(high - low, 1.0)
        features[18] = fxClampSignedUnit((close - open) / range)
        features[19] = fxClamp((high - max(open, close)) / range, 0.0, 1.0)
        features[20] = fxClamp((min(open, close) - low) / range, 0.0, 1.0)
        features[21] = fxClamp(range / max(close, 1.0) * 1_000.0, 0.0, 1.0)
        features[66] = fxClamp((close - low) / range, 0.0, 1.0)
        features[67] = fxClampSignedUnit(features[20] - features[19])
    }

    private func fillMovingAverageFeatures(_ features: inout [Double], series: M1OHLCVSeries, index: Int) {
        features[7] = normalizedReturn(series, index, 5)
        features[8] = normalizedReturn(series, index, 15)
        features[9] = normalizedReturn(series, index, 60)
        features[13] = closeSlope(series, index, window: 25)
        features[14] = closeSlope(series, index, window: 120)

        let slots: [(Int, Int)] = [
            (22, 500), (23, 1_000), (24, 1_500), (25, 3_000),
            (26, 3_000), (27, 6_000), (28, 6_000), (29, 12_000),
            (30, 500), (31, 1_000), (32, 1_500), (33, 3_000),
            (34, 3_000), (35, 6_000), (36, 6_000), (37, 12_000)
        ]
        for (slot, window) in slots {
            features[slot] = movingAverageEdge(series, index, window: min(window, 500))
        }
        features[38] = movingAverageEdge(series, index, window: 100)
        features[39] = movingAverageEdge(series, index, window: 200)
    }

    private func fillVolatilityFeatures(_ features: inout [Double], series: M1OHLCVSeries, index: Int) {
        features[40] = rsi(series, index, window: 14)
        features[41] = atrUnit(series, index, window: 14)
        features[42] = features[41]
        features[43] = parkinsonVol(series, index, window: 20)
        features[44] = rogersSatchellVol(series, index, window: 20)
        features[45] = garmanKlassVol(series, index, window: 20)
    }

    private func fillFilterFeatures(_ features: inout [Double], series: M1OHLCVSeries, index: Int) {
        features[46] = movingAverageEdge(series, index, window: 21)
        features[47] = abs(closeZScore(series, index, window: 21)) > 0.75 ? 1.0 : 0.0
        features[48] = movingAverageEdge(series, index, window: 34)
        features[49] = movingAverageEdge(series, index, window: 20)
    }

    private func fillVolumeAwareMicrostructureFeatures(
        _ features: inout [Double],
        universe: MarketUniverse,
        index: Int,
        hasVolume: Bool
    ) {
        guard hasVolume else {
            for slot in [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83] {
                features[slot] = 0.0
            }
            return
        }

        let series = universe.primary
        let vol = Double(series.volume[index])
        let mean20 = volumeMean(series, index, window: 20)
        let mean50 = volumeMean(series, index, window: 50)
        let std20 = volumeStd(series, index, window: 20)
        let prevVol = index > 0 ? Double(series.volume[index - 1]) : vol
        let prevPrevVol = index > 1 ? Double(series.volume[index - 2]) : prevVol
        let range = Double(max(series.high[index] - series.low[index], 1))
        let priceRet = features[0]

        features[6] = mean20 > 0 ? fxClamp((log1p(vol) / log1p(mean20)) - 1.0, -1.0, 1.0) : 0.0
        features[68] = std20 > 0 ? fxClampSignedUnit((vol - mean20) / std20 / 4.0) : 0.0
        features[69] = fxClampSignedUnit(((vol - prevVol) - (prevVol - prevPrevVol)) / max(mean20, 1.0))
        features[70] = fxClamp(vol / max(range, 1.0) / max(mean20 / 10.0, 1.0), 0.0, 1.0)
        features[71] = volumeSlope(series, index, window: 20)
        features[74] = volumeSessionActivity(series, index: index)
        features[75] = vol > 0 ? 1.0 : 0.0
        features[76] = volumeRank(series, index, window: 50)
        features[77] = fxClampSignedUnit(priceRet * features[68])
        features[78] = mean50 > 0 ? fxClamp(mean20 / mean50, 0.0, 2.0) / 2.0 : 0.0
        features[80] = fxClamp(log1p(vol) / 12.0, 0.0, 1.0)
        features[81] = features[68]
        features[82] = mean50 > 0 ? fxClamp(mean20 / mean50, 0.0, 2.0) / 2.0 : 0.0
        features[83] = volumeRank(series, index, window: 20)
    }

    private func fillMainMTFFeatures(_ features: inout [Double], series: M1OHLCVSeries, index: Int, hasVolume: Bool) {
        let windows = [5, 15, 30, 60]
        for (slot, window) in windows.enumerated() {
            let metrics = timeframeState(series, index: index, window: window, hasVolume: hasVolume)
            let base = FXDataEngineConstants.mainMTFFeatureOffset + slot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            for metric in 0..<FXDataEngineConstants.mtfStateFeaturesPerTimeframe {
                features[base + metric] = metrics[metric]
            }
        }
    }

    private func fillContextMTFFeatures(_ features: inout [Double], universe: MarketUniverse, index: Int, hasVolume: Bool) {
        let windows = [1, 5, 15, 30, 60]
        let contexts = Array(universe.contextSeries().prefix(FXDataEngineConstants.contextTopSymbols))
        for slot in 0..<FXDataEngineConstants.contextTopSymbols {
            guard slot < contexts.count else { continue }
            let series = contexts[slot]
            guard index < series.count else { continue }
            for (tfSlot, window) in windows.enumerated() {
                let metrics = timeframeState(series, index: index, window: window, hasVolume: hasVolume)
                let base = FXDataEngineConstants.contextMTFFeatureOffset +
                    slot * FXDataEngineConstants.contextSlotMTFFeatures +
                    tfSlot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe
                for metric in 0..<FXDataEngineConstants.mtfStateFeaturesPerTimeframe {
                    features[base + metric] = metrics[metric]
                }
            }
        }
    }

    private func fillFeatureFamilyDrift(_ features: inout [Double]) {
        let groups: [FeatureGroup] = [.price, .multiTimeframe, .context, .volume, .microstructure, .filters]
        var means: [FeatureGroup: Double] = [:]
        for group in groups {
            var values: [Double] = []
            for featureIndex in 0..<FXDataEngineConstants.aiFeatures where registry.group(for: featureIndex) == group {
                values.append(abs(features[featureIndex]))
            }
            means[group] = values.mean
        }
        let priceMTFDrift = abs((means[.price] ?? 0) - (means[.multiTimeframe] ?? 0))
        let contextVolumeDrift = abs((means[.context] ?? 0) - (means[.volume] ?? 0))
        let microFilterDrift = abs((means[.microstructure] ?? 0) - (means[.filters] ?? 0))
        let drift = priceMTFDrift + contextVolumeDrift + 0.5 * microFilterDrift
        features[79] = fxClamp(drift, 0.0, 1.0)
    }
}

private extension FeatureCore {
    func rawReturn(_ series: M1OHLCVSeries, _ index: Int, _ lookback: Int) -> Double {
        let prior = index - lookback
        guard prior >= 0, index >= 0, index < series.count else { return 0.0 }
        let old = Double(series.close[prior])
        let new = Double(series.close[index])
        guard old > 0 else { return 0.0 }
        return (new - old) / old
    }

    func rollingAbsRawReturn(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        guard window >= 2, index > 0, index < series.count else { return 0.0 }
        let start = max(1, index - window + 1)
        var sum = 0.0
        var count = 0
        for row in start...index {
            sum += abs(rawReturn(series, row, 1))
            count += 1
        }
        return count > 0 ? sum / Double(count) : 0.0
    }

    func normalizedReturn(_ series: M1OHLCVSeries, _ index: Int, _ lookback: Int) -> Double {
        let prior = index - lookback
        guard prior >= 0 else { return 0.0 }
        let old = Double(series.close[prior])
        let new = Double(series.close[index])
        guard old > 0 else { return 0.0 }
        return fxClampSignedUnit((new - old) / old * 100.0)
    }

    func closeSlope(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        let start = max(0, index - max(1, window) + 1)
        guard index > start else { return 0.0 }
        let first = Double(series.close[start])
        let last = Double(series.close[index])
        guard first > 0 else { return 0.0 }
        return fxClampSignedUnit((last - first) / first * 100.0 / Double(index - start))
    }

    func closeZScore(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        let values = closeValues(series, index: index, window: window)
        guard values.count > 1 else { return 0.0 }
        let mean = values.mean
        let std = values.standardDeviation
        guard std > 0 else { return 0.0 }
        return fxClampSignedUnit((values.last! - mean) / std / 4.0)
    }

    func returnStd(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        guard index > 0 else { return 0.0 }
        let start = max(1, index - max(1, window) + 1)
        var returns: [Double] = []
        returns.reserveCapacity(max(0, index - start + 1))
        for row in start...index {
            let old = Double(series.close[row - 1])
            let new = Double(series.close[row])
            if old > 0 {
                returns.append((new - old) / old)
            }
        }
        return fxClamp(returns.standardDeviation * 100.0, 0.0, 1.0)
    }

    func movingAverageEdge(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        let values = closeValues(series, index: index, window: window)
        guard !values.isEmpty else { return 0.0 }
        let mean = values.mean
        guard mean > 0 else { return 0.0 }
        return fxClampSignedUnit((Double(series.close[index]) - mean) / mean * 100.0)
    }

    func closeValues(_ series: M1OHLCVSeries, index: Int, window: Int) -> [Double] {
        guard index >= 0, index < series.count else { return [] }
        let start = max(0, index - max(1, window) + 1)
        return (start...index).map { Double(series.close[$0]) }
    }

    func rsi(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        guard index > 0 else { return 0.5 }
        let start = max(1, index - max(1, window) + 1)
        var gain = 0.0
        var loss = 0.0
        for row in start...index {
            let delta = Double(series.close[row] - series.close[row - 1])
            if delta >= 0 { gain += delta } else { loss += abs(delta) }
        }
        guard gain + loss > 0 else { return 0.5 }
        return fxClamp(gain / (gain + loss), 0.0, 1.0)
    }

    func atrUnit(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        guard index >= 0 else { return 0.0 }
        let start = max(0, index - max(1, window) + 1)
        var ranges: [Double] = []
        ranges.reserveCapacity(index - start + 1)
        for row in start...index {
            ranges.append(Double(max(series.high[row] - series.low[row], 0)) / max(Double(series.close[row]), 1.0))
        }
        return fxClamp(ranges.mean * 1_000.0, 0.0, 1.0)
    }

    func parkinsonVol(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        let start = max(0, index - max(1, window) + 1)
        var values: [Double] = []
        for row in start...index {
            let high = Double(series.high[row])
            let low = Double(series.low[row])
            if high > 0, low > 0 {
                let logRange = log(high / low)
                values.append(logRange * logRange)
            }
        }
        return fxClamp(sqrt(values.mean / (4.0 * log(2.0))) * 100.0, 0.0, 1.0)
    }

    func rogersSatchellVol(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        let start = max(0, index - max(1, window) + 1)
        var values: [Double] = []
        for row in start...index {
            let open = Double(series.open[row])
            let high = Double(series.high[row])
            let low = Double(series.low[row])
            let close = Double(series.close[row])
            if open > 0, high > 0, low > 0, close > 0 {
                let value = log(high / close) * log(high / open) + log(low / close) * log(low / open)
                values.append(max(0.0, value))
            }
        }
        return fxClamp(sqrt(values.mean) * 100.0, 0.0, 1.0)
    }

    func garmanKlassVol(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        let start = max(0, index - max(1, window) + 1)
        var values: [Double] = []
        for row in start...index {
            let open = Double(series.open[row])
            let high = Double(series.high[row])
            let low = Double(series.low[row])
            let close = Double(series.close[row])
            if open > 0, high > 0, low > 0, close > 0 {
                let range = log(high / low)
                let body = log(close / open)
                values.append(max(0.0, 0.5 * range * range - (2.0 * log(2.0) - 1.0) * body * body))
            }
        }
        return fxClamp(sqrt(values.mean) * 100.0, 0.0, 1.0)
    }

    func volumeMean(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        volumeValues(series, index: index, window: window).mean
    }

    func volumeStd(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        volumeValues(series, index: index, window: window).standardDeviation
    }

    func volumeValues(_ series: M1OHLCVSeries, index: Int, window: Int) -> [Double] {
        guard index >= 0, index < series.count else { return [] }
        let start = max(0, index - max(1, window) + 1)
        return (start...index).map { Double(series.volume[$0]) }
    }

    func volumeSlope(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        let values = volumeValues(series, index: index, window: window)
        guard let first = values.first, let last = values.last, first > 0 else { return 0.0 }
        return fxClampSignedUnit((last - first) / first)
    }

    func volumeRank(_ series: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        let values = volumeValues(series, index: index, window: window)
        guard !values.isEmpty else { return 0.0 }
        let current = Double(series.volume[index])
        let belowOrEqual = values.filter { $0 <= current }.count
        return fxClamp(Double(belowOrEqual) / Double(values.count), 0.0, 1.0)
    }

    func volumeSessionActivity(_ series: M1OHLCVSeries, index: Int) -> Double {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let date = Date(timeIntervalSince1970: TimeInterval(series.utcTimestamps[index]))
        let hour = calendar.component(.hour, from: date)
        let sessionWindow = hour >= 7 && hour <= 20 ? 50 : 120
        let mean = volumeMean(series, index, window: sessionWindow)
        guard mean > 0 else { return 0.0 }
        return fxClamp(Double(series.volume[index]) / mean, 0.0, 2.0) / 2.0
    }

    func sessionTransitionScore(hour: Int, minute: Int) -> Double {
        let transitionHours = [0, 7, 12, 13, 16, 21]
        let distance = transitionHours.map { abs(($0 * 60) - (hour * 60 + minute)) }.min() ?? 1_440
        return fxClamp(1.0 - Double(min(distance, 120)) / 120.0, 0.0, 1.0)
    }

    func sessionOverlapScore(hour: Int) -> Double {
        if (12...16).contains(hour) { return 1.0 }
        if (7...20).contains(hour) { return 0.5 }
        return 0.0
    }

    func rollingCorrelation(_ a: M1OHLCVSeries, _ b: M1OHLCVSeries, _ index: Int, window: Int) -> Double {
        guard index > 0 else { return 0.0 }
        let start = max(1, index - max(1, window) + 1)
        guard start <= index, index < a.count, index < b.count else { return 0.0 }
        var ra: [Double] = []
        var rb: [Double] = []
        for row in start...index {
            let a0 = Double(a.close[row - 1])
            let b0 = Double(b.close[row - 1])
            if a0 > 0, b0 > 0 {
                ra.append((Double(a.close[row]) - a0) / a0)
                rb.append((Double(b.close[row]) - b0) / b0)
            }
        }
        guard ra.count == rb.count, ra.count > 1 else { return 0.0 }
        let ma = ra.mean
        let mb = rb.mean
        var cov = 0.0
        var va = 0.0
        var vb = 0.0
        for index in ra.indices {
            let da = ra[index] - ma
            let db = rb[index] - mb
            cov += da * db
            va += da * da
            vb += db * db
        }
        guard va > 0, vb > 0 else { return 0.0 }
        return fxClampSignedUnit(cov / sqrt(va * vb))
    }

    func timeframeState(_ series: M1OHLCVSeries, index: Int, window: Int, hasVolume: Bool) -> [Double] {
        let start = max(0, index - max(1, window) + 1)
        let open = Double(series.open[start])
        let high = Double((start...index).map { series.high[$0] }.max() ?? series.high[index])
        let low = Double((start...index).map { series.low[$0] }.min() ?? series.low[index])
        let close = Double(series.close[index])
        let range = max(high - low, 1.0)
        let volumePressure: Double
        if hasVolume {
            let volumeSum = (start...index).reduce(0.0) { $0 + Double(series.volume[$1]) }
            let mean = volumeMean(series, index, window: max(window * 2, 20))
            volumePressure = mean > 0 ? fxClamp(volumeSum / Double(index - start + 1) / mean, 0.0, 2.0) / 2.0 : 0.0
        } else {
            volumePressure = 0.0
        }
        return [
            fxClampSignedUnit((close - open) / range),
            fxClamp((close - low) / range, 0.0, 1.0),
            fxClamp(range / max(close, 1.0) * 1_000.0, 0.0, 1.0),
            volumePressure
        ]
    }
}

private extension DataCoreContextAggregates {
    func hasCompleteSample(_ sampleIndex: Int) -> Bool {
        guard sampleIndex >= 0,
              sampleIndex < mean.count,
              sampleIndex < standardDeviation.count,
              sampleIndex < upRatio.count else {
            return false
        }
        let requiredExtraCount = (sampleIndex + 1) * FXDataEngineConstants.contextExtraFeatures
        return requiredExtraCount <= extra.count
    }
}

private extension Array where Element == Double {
    var mean: Double {
        guard !isEmpty else { return 0.0 }
        return reduce(0.0, +) / Double(count)
    }

    var standardDeviation: Double {
        guard count > 1 else { return 0.0 }
        let mean = self.mean
        let variance = reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(count - 1)
        return sqrt(Swift.max(0.0, variance))
    }
}
