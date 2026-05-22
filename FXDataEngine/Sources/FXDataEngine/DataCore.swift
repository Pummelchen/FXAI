import Foundation

public struct DataCoreRequest: Codable, Hashable, Sendable {
    public var liveMode: Bool
    public var symbol: String
    public var signalTimestampUTC: Int64?
    public var neededBars: Int
    public var alignUpToIndex: Int?
    public var contextSymbols: [String]

    public init(
        liveMode: Bool = false,
        symbol: String,
        signalTimestampUTC: Int64? = nil,
        neededBars: Int,
        alignUpToIndex: Int? = nil,
        contextSymbols: [String] = []
    ) {
        let normalizedSymbol = Self.normalizedSymbol(symbol)
        self.liveMode = liveMode
        self.symbol = normalizedSymbol
        self.signalTimestampUTC = signalTimestampUTC
        self.neededBars = neededBars
        self.alignUpToIndex = alignUpToIndex
        self.contextSymbols = Self.normalizedContextSymbols(contextSymbols).filter { $0 != normalizedSymbol }
    }

    public static func normalizedSymbol(_ symbol: String) -> String {
        symbol.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
    }

    public static func normalizedContextSymbols(_ symbols: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(symbols.count, FXDataEngineConstants.maxContextSymbols))
        var seen: Set<String> = []
        for raw in symbols {
            let symbol = normalizedSymbol(raw)
            guard !symbol.isEmpty, seen.insert(symbol).inserted else { continue }
            output.append(symbol)
            if output.count >= FXDataEngineConstants.maxContextSymbols {
                break
            }
        }
        return output
    }

    @discardableResult
    public mutating func addContextSymbol(_ symbol: String) -> Bool {
        let normalized = Self.normalizedSymbol(symbol)
        guard !normalized.isEmpty,
              normalized != self.symbol,
              !contextSymbols.contains(normalized),
              contextSymbols.count < FXDataEngineConstants.maxContextSymbols else {
            return false
        }
        contextSymbols.append(normalized)
        return true
    }

    public mutating func captureContextSymbols(_ symbols: [String]) {
        for symbol in symbols {
            _ = addContextSymbol(symbol)
        }
    }
}

public struct DataCoreTimeframeNeeds: Codable, Hashable, Sendable {
    public var m5: Int
    public var m15: Int
    public var m30: Int
    public var h1: Int

    public static func legacy(neededBars: Int) -> DataCoreTimeframeNeeds {
        let needed = max(0, neededBars)
        return DataCoreTimeframeNeeds(
            m5: max((needed / 5) + 80, 220),
            m15: max((needed / 15) + 80, 220),
            m30: max((needed / 30) + 80, 220),
            h1: max((needed / 60) + 80, 220)
        )
    }
}

public struct DataCoreTimeframeLags: Codable, Hashable, Sendable {
    public var m5Seconds: Int
    public var m15Seconds: Int
    public var m30Seconds: Int
    public var h1Seconds: Int

    public static let legacy = DataCoreTimeframeLags(
        m5Seconds: 600,
        m15Seconds: 1_800,
        m30Seconds: 3_600,
        h1Seconds: 7_200
    )
}

public enum DataCoreAlignment {
    public static func movePoints(priceNow: Double, priceFuture: Double, point: Double) -> Double {
        guard point > 0.0 else { return 0.0 }
        return (fxSafeFinite(priceFuture) - fxSafeFinite(priceNow)) / point
    }

    public static func findAlignedIndex(
        targetTimesAscending: ContiguousArray<Int64>,
        referenceTimeUTC: Int64,
        maxLagSeconds: Int
    ) -> Int {
        guard referenceTimeUTC > 0, !targetTimesAscending.isEmpty else { return -1 }
        var low = 0
        var high = targetTimesAscending.count - 1
        var answer = -1

        while low <= high {
            let mid = (low + high) / 2
            if targetTimesAscending[mid] <= referenceTimeUTC {
                answer = mid
                low = mid + 1
            } else {
                high = mid - 1
            }
        }

        guard answer >= 0 else { return -1 }
        guard maxLagSeconds > 0 else { return answer }
        let lag = referenceTimeUTC - targetTimesAscending[answer]
        return lag >= 0 && lag <= Int64(maxLagSeconds) ? answer : -1
    }

    public static func buildAlignedIndexMap(
        referenceTimesAscending: ContiguousArray<Int64>,
        targetTimesAscending: ContiguousArray<Int64>,
        maxLagSeconds: Int,
        upToIndex: Int? = nil
    ) -> [Int] {
        let count = referenceTimesAscending.count
        var output = Array(repeating: -1, count: count)
        guard count > 0, !targetTimesAscending.isEmpty else { return output }
        let upTo = min(max(upToIndex ?? (count - 1), 0), count - 1)
        var targetIndex = 0

        for refIndex in 0...upTo {
            let referenceTime = referenceTimesAscending[refIndex]
            guard referenceTime > 0 else { continue }
            while targetIndex + 1 < targetTimesAscending.count,
                  targetTimesAscending[targetIndex + 1] <= referenceTime {
                targetIndex += 1
            }
            let lag = referenceTime - targetTimesAscending[targetIndex]
            guard lag >= 0 else { continue }
            if maxLagSeconds > 0, lag > Int64(maxLagSeconds) {
                continue
            }
            output[refIndex] = targetIndex
        }
        return output
    }

    public static func alignedFreshnessWeight(
        targetTimesAscending: ContiguousArray<Int64>,
        targetIndex: Int,
        referenceTimeUTC: Int64,
        maxLagSeconds: Int
    ) -> Double {
        guard targetIndex >= 0,
              targetIndex < targetTimesAscending.count,
              referenceTimeUTC > 0 else {
            return 0.0
        }
        guard maxLagSeconds > 0 else { return 1.0 }
        let lag = referenceTimeUTC - targetTimesAscending[targetIndex]
        guard lag >= 0, lag <= Int64(maxLagSeconds) else { return 0.0 }
        return fxClamp(1.0 - Double(lag) / Double(maxLagSeconds), 0.0, 1.0)
    }
}

public struct DataCoreContextAggregates: Sendable {
    public var mean: [Double]
    public var standardDeviation: [Double]
    public var upRatio: [Double]
    public var extra: [Double]
    public var symbolUtility: [Double]
    public var symbolStability: [Double]
    public var symbolLead: [Double]
    public var symbolCoverage: [Double]
    public var symbolReady: [Bool]

    public static func empty(count: Int) -> DataCoreContextAggregates {
        DataCoreContextAggregates(
            mean: Array(repeating: 0.0, count: count),
            standardDeviation: Array(repeating: 0.0, count: count),
            upRatio: Array(repeating: 0.5, count: count),
            extra: Array(repeating: 0.0, count: count * FXDataEngineConstants.contextExtraFeatures),
            symbolUtility: Array(repeating: 0.0, count: FXDataEngineConstants.maxContextSymbols),
            symbolStability: Array(repeating: 0.0, count: FXDataEngineConstants.maxContextSymbols),
            symbolLead: Array(repeating: 0.0, count: FXDataEngineConstants.maxContextSymbols),
            symbolCoverage: Array(repeating: 0.0, count: FXDataEngineConstants.maxContextSymbols),
            symbolReady: Array(repeating: false, count: FXDataEngineConstants.maxContextSymbols)
        )
    }

    public func extraValue(sampleIndex: Int, featureIndex: Int, default defaultValue: Double = 0.0) -> Double {
        guard sampleIndex >= 0,
              featureIndex >= 0,
              featureIndex < FXDataEngineConstants.contextExtraFeatures else {
            return defaultValue
        }
        let index = sampleIndex * FXDataEngineConstants.contextExtraFeatures + featureIndex
        guard index >= 0, index < extra.count else { return defaultValue }
        return extra[index]
    }

    mutating func setExtraValue(sampleIndex: Int, featureIndex: Int, value: Double) {
        guard sampleIndex >= 0,
              featureIndex >= 0,
              featureIndex < FXDataEngineConstants.contextExtraFeatures else {
            return
        }
        let index = sampleIndex * FXDataEngineConstants.contextExtraFeatures + featureIndex
        guard index >= 0, index < extra.count else { return }
        extra[index] = fxSafeFinite(value)
    }
}

public struct DataCoreBundle: Sendable {
    public let ready: Bool
    public let request: DataCoreRequest
    public let universe: MarketUniverse
    public let sampleIndex: Int
    public let contextAggregates: DataCoreContextAggregates

    public var primary: M1OHLCVSeries { universe.primary }
    public var sampleTimeUTC: Int64 { primary.utcTimestamps[sampleIndex] }

    public init(
        ready: Bool,
        request: DataCoreRequest,
        universe: MarketUniverse,
        sampleIndex: Int,
        contextAggregates: DataCoreContextAggregates = DataCoreContextAggregates.empty(count: 0)
    ) {
        self.ready = ready
        self.request = request
        self.universe = universe
        self.sampleIndex = sampleIndex
        self.contextAggregates = contextAggregates
    }
}

public struct DataCore: Sendable {
    public init() {}

    public func buildBundle(request: DataCoreRequest, universe: MarketUniverse) throws -> DataCoreBundle {
        guard request.neededBars > 0 else {
            throw FXDataEngineError.invalidRequest("neededBars must be positive")
        }
        guard universe.primarySymbol == request.symbol.uppercased() else {
            throw FXDataEngineError.invalidRequest("request symbol \(request.symbol) does not match primary universe symbol \(universe.primarySymbol)")
        }
        guard universe.primary.count >= request.neededBars else {
            throw FXDataEngineError.insufficientData("primary series has \(universe.primary.count) rows, need \(request.neededBars)")
        }

        let sampleIndex: Int
        if let alignUpToIndex = request.alignUpToIndex {
            sampleIndex = min(max(alignUpToIndex, 0), universe.primary.count - 1)
        } else if let timestamp = request.signalTimestampUTC {
            guard let found = universe.primary.utcTimestamps.lastIndex(where: { $0 <= timestamp }) else {
                throw FXDataEngineError.insufficientData("no primary M1 bar at or before requested signal timestamp")
            }
            sampleIndex = found
        } else {
            sampleIndex = universe.primary.count - 1
        }

        guard sampleIndex + 1 >= request.neededBars else {
            throw FXDataEngineError.insufficientData("sample index \(sampleIndex) does not leave enough lookback for \(request.neededBars) bars")
        }

        for contextSymbol in request.contextSymbols where universe[contextSymbol] == nil {
            throw FXDataEngineError.missingSeries(symbol: contextSymbol, timeframe: .m1)
        }

        let aggregateSymbols = request.contextSymbols.isEmpty
            ? universe.contextSeries().map(\.metadata.logicalSymbol)
            : request.contextSymbols
        let contextAggregates = DataCoreContextAggregator.build(
            universe: universe,
            upToIndex: sampleIndex,
            contextSymbols: aggregateSymbols
        )

        return DataCoreBundle(
            ready: true,
            request: request,
            universe: universe,
            sampleIndex: sampleIndex,
            contextAggregates: contextAggregates
        )
    }
}

public enum DataCoreContextAggregator {
    public static func build(
        universe: MarketUniverse,
        upToIndex: Int,
        contextSymbols: [String],
        maxLagSeconds: Int = 120
    ) -> DataCoreContextAggregates {
        let primary = universe.primary
        var aggregates = DataCoreContextAggregates.empty(count: primary.count)
        guard primary.count > 0 else { return aggregates }

        let symbols = DataCoreRequest.normalizedContextSymbols(contextSymbols)
            .filter { $0 != universe.primarySymbol }
        let contexts = symbols.compactMap { universe[$0] }
        guard !contexts.isEmpty else { return aggregates }

        let upTo = min(max(upToIndex, 0), primary.count - 1)
        let upToFill = upTo
        let alignedMaps = contexts.map {
            DataCoreAlignment.buildAlignedIndexMap(
                referenceTimesAscending: primary.utcTimestamps,
                targetTimesAscending: $0.utcTimestamps,
                maxLagSeconds: maxLagSeconds,
                upToIndex: upToFill
            )
        }

        for index in 0...upToFill {
            let referenceTime = primary.utcTimestamps[index]
            guard referenceTime > 0 else { continue }

            let mainReturn = safeReturn(primary.close, current: index, previous: index - 1)
            var mainVolatility = rollingAbsReturn(primary.close, current: index, width: 20)
            if mainVolatility < 1e-6 {
                mainVolatility = abs(mainReturn)
            }
            if mainVolatility < 1e-6 {
                mainVolatility = 1e-4
            }

            var weightedSum = 0.0
            var weightedSum2 = 0.0
            var weightTotal = 0.0
            var upWeight = 0.0
            var valid = 0
            var topRows: [(score: Double, ret: Double, lag: Double, relative: Double, correlation: Double)] = []

            for (contextOffset, context) in contexts.enumerated() {
                let map = alignedMaps[contextOffset]
                guard index < map.count else { continue }
                let contextIndex = map[index]
                guard contextIndex >= 0 else { continue }

                let freshness = alignedFreshnessWeight(
                    timeUTC: context.utcTimestamps[contextIndex],
                    referenceTimeUTC: referenceTime,
                    maxLagSeconds: maxLagSeconds
                )
                guard freshness > 0.0 else { continue }

                let contextReturn = safeReturn(context.close, current: contextIndex, previous: contextIndex - 1)
                let contextLag = safeReturn(context.close, current: contextIndex - 1, previous: contextIndex - 2)
                let contextRelative = contextReturn - mainReturn
                let contextCorrelation = alignedCorrelation(
                    primary: primary,
                    context: context,
                    alignedMap: map,
                    current: index,
                    window: 20
                )
                let relativeEdge = fxClamp(abs(contextRelative) / mainVolatility, 0.0, 4.0)
                let returnEdge = fxClamp(abs(contextReturn) / mainVolatility, 0.0, 4.0)
                let lagEdge = fxClamp(abs(contextLag) / mainVolatility, 0.0, 4.0)
                let correlationEdge = abs(contextCorrelation)
                let symbolScore = freshness * (
                    0.40 * correlationEdge +
                        0.30 * relativeEdge +
                        0.20 * lagEdge +
                        0.10 * returnEdge
                )
                let weight = 0.20 + symbolScore

                weightedSum += weight * contextReturn
                weightedSum2 += weight * contextReturn * contextReturn
                weightTotal += weight
                if contextReturn > 0.0 {
                    upWeight += weight
                }
                valid += 1
                topRows.append((
                    score: symbolScore,
                    ret: contextReturn * freshness,
                    lag: contextLag * freshness,
                    relative: contextRelative * freshness,
                    correlation: contextCorrelation * freshness
                ))
            }

            guard valid > 0, weightTotal > 0 else { continue }

            let mean = weightedSum / weightTotal
            let variance = max((weightedSum2 / weightTotal) - (mean * mean), 0.0)
            let coverage = fxClamp(Double(valid) / Double(contexts.count), 0.0, 1.0)
            let confidence = 0.30 + 0.70 * coverage
            let upRatio = upWeight / weightTotal

            aggregates.mean[index] = mean * coverage
            aggregates.standardDeviation[index] = sqrt(variance) * confidence
            aggregates.upRatio[index] = 0.5 + ((upRatio - 0.5) * coverage)

            topRows.sort { $0.score > $1.score }
            for slot in 0..<min(FXDataEngineConstants.contextTopSymbols, topRows.count) {
                let row = topRows[slot]
                aggregates.setExtraValue(sampleIndex: index, featureIndex: slot * 4, value: row.ret)
                aggregates.setExtraValue(sampleIndex: index, featureIndex: slot * 4 + 1, value: row.lag)
                aggregates.setExtraValue(sampleIndex: index, featureIndex: slot * 4 + 2, value: row.relative)
                aggregates.setExtraValue(sampleIndex: index, featureIndex: slot * 4 + 3, value: row.correlation)
            }

            let adapterStability = topRows.prefix(FXDataEngineConstants.contextTopSymbols).reduce(0.5) { partial, row in
                partial + (1.0 - fxClamp(abs(row.ret - row.lag) / max(mainVolatility, 1e-4), 0.0, 1.0))
            }
            let adapterLead = topRows.prefix(FXDataEngineConstants.contextTopSymbols).reduce(0.5) { partial, row in
                partial + fxClamp(abs(row.lag) / max(mainVolatility, 1e-4), 0.0, 4.0) / 4.0
            }
            let adapterCount = min(FXDataEngineConstants.contextTopSymbols, topRows.count) + 1
            aggregates.setExtraValue(
                sampleIndex: index,
                featureIndex: FXDataEngineConstants.contextSharedOffset,
                value: fxClamp(mean / max(mainVolatility, 1e-4), -1.0, 1.0)
            )
            aggregates.setExtraValue(
                sampleIndex: index,
                featureIndex: FXDataEngineConstants.contextSharedOffset + 1,
                value: fxClamp(adapterStability / Double(adapterCount), 0.0, 1.0)
            )
            aggregates.setExtraValue(
                sampleIndex: index,
                featureIndex: FXDataEngineConstants.contextSharedOffset + 2,
                value: fxClamp(adapterLead / Double(adapterCount), 0.0, 1.0)
            )
            aggregates.setExtraValue(
                sampleIndex: index,
                featureIndex: FXDataEngineConstants.contextSharedOffset + 3,
                value: coverage
            )
        }

        for (contextOffset, context) in contexts.prefix(FXDataEngineConstants.maxContextSymbols).enumerated() {
            let map = alignedMaps[contextOffset]
            let utility = contextSeriesUtility(primary: primary, context: context, alignedMap: map, current: upTo)
            guard utility > -1e8 else { continue }
            let aligned = upTo < map.count ? map[upTo] : -1
            guard aligned >= 0 else { continue }
            let mainVolatility = max(rollingAbsReturn(primary.close, current: upTo, width: 20), 1e-4)
            let contextReturn = safeReturn(context.close, current: aligned, previous: aligned - 1)
            let contextLag = safeReturn(context.close, current: aligned - 1, previous: aligned - 2)
            aggregates.symbolUtility[contextOffset] = utility
            aggregates.symbolStability[contextOffset] = 1.0 - fxClamp(abs(contextReturn - contextLag) / mainVolatility, 0.0, 1.0)
            aggregates.symbolLead[contextOffset] = fxClamp(abs(contextLag) / mainVolatility, 0.0, 4.0) / 4.0
            aggregates.symbolCoverage[contextOffset] = alignedFreshnessWeight(
                timeUTC: context.utcTimestamps[aligned],
                referenceTimeUTC: primary.utcTimestamps[upTo],
                maxLagSeconds: maxLagSeconds
            )
            aggregates.symbolReady[contextOffset] = true
        }

        return aggregates
    }

    private static func safeReturn(_ values: ContiguousArray<Int64>, current: Int, previous: Int) -> Double {
        guard current >= 0,
              previous >= 0,
              current < values.count,
              previous < values.count else {
            return 0.0
        }
        let base = Double(values[previous])
        guard base > 0 else { return 0.0 }
        return (Double(values[current]) - base) / base
    }

    private static func rollingAbsReturn(_ values: ContiguousArray<Int64>, current: Int, width: Int) -> Double {
        guard width >= 2, current > 0, current < values.count else { return 0.0 }
        let oldest = max(1, current - width + 1)
        var sum = 0.0
        var count = 0
        for index in oldest...current {
            sum += abs(safeReturn(values, current: index, previous: index - 1))
            count += 1
        }
        return count > 0 ? sum / Double(count) : 0.0
    }

    private static func alignedFreshnessWeight(timeUTC: Int64, referenceTimeUTC: Int64, maxLagSeconds: Int) -> Double {
        guard referenceTimeUTC > 0 else { return 0.0 }
        guard maxLagSeconds > 0 else { return 1.0 }
        let lag = referenceTimeUTC - timeUTC
        guard lag >= 0, lag <= Int64(maxLagSeconds) else { return 0.0 }
        return fxClamp(1.0 - Double(lag) / Double(maxLagSeconds), 0.0, 1.0)
    }

    private static func alignedCorrelation(
        primary: M1OHLCVSeries,
        context: M1OHLCVSeries,
        alignedMap: [Int],
        current: Int,
        window: Int
    ) -> Double {
        guard current > 0, current < primary.count else { return 0.0 }
        var sx = 0.0
        var sy = 0.0
        var sxx = 0.0
        var syy = 0.0
        var sxy = 0.0
        var used = 0
        let oldest = max(1, current - max(1, window) + 1)
        for mainIndex in oldest...current {
            guard mainIndex < alignedMap.count else { continue }
            let contextIndex = alignedMap[mainIndex]
            guard contextIndex > 0, contextIndex < context.count else { continue }
            let x = safeReturn(primary.close, current: mainIndex, previous: mainIndex - 1)
            let y = safeReturn(context.close, current: contextIndex, previous: contextIndex - 1)
            sx += x
            sy += y
            sxx += x * x
            syy += y * y
            sxy += x * y
            used += 1
        }
        guard used >= 4 else { return 0.0 }
        let count = Double(used)
        let covariance = sxy - (sx * sy) / count
        let vx = sxx - (sx * sx) / count
        let vy = syy - (sy * sy) / count
        guard vx > 1e-12, vy > 1e-12 else { return 0.0 }
        return fxClamp(covariance / sqrt(vx * vy), -1.0, 1.0)
    }

    private static func contextSeriesUtility(
        primary: M1OHLCVSeries,
        context: M1OHLCVSeries,
        alignedMap: [Int],
        current: Int
    ) -> Double {
        guard current > 0, current < primary.count, alignedMap.count > 4 else {
            return -1e9
        }
        var sumScore = 0.0
        var used = 0
        let oldest = max(1, current - 15)
        for mainIndex in oldest...current {
            guard mainIndex < alignedMap.count else { continue }
            let contextIndex = alignedMap[mainIndex]
            guard contextIndex >= 0, contextIndex < context.count else { continue }
            let fresh = alignedFreshnessWeight(
                timeUTC: context.utcTimestamps[contextIndex],
                referenceTimeUTC: primary.utcTimestamps[mainIndex],
                maxLagSeconds: 120
            )
            let mainReturn = safeReturn(primary.close, current: mainIndex, previous: mainIndex - 1)
            let contextReturn = safeReturn(context.close, current: contextIndex, previous: contextIndex - 1)
            let contextLag = safeReturn(context.close, current: contextIndex - 1, previous: contextIndex - 2)
            let correlation = alignedCorrelation(
                primary: primary,
                context: context,
                alignedMap: alignedMap,
                current: mainIndex,
                window: 20
            )
            var volatility = rollingAbsReturn(primary.close, current: mainIndex, width: 20)
            if volatility < 1e-6 {
                volatility = abs(mainReturn)
            }
            if volatility < 1e-6 {
                volatility = 1e-4
            }
            let relative = fxClamp(abs(contextReturn - mainReturn) / volatility, 0.0, 4.0)
            let lead = fxClamp(abs(contextLag) / volatility, 0.0, 4.0)
            let magnitude = fxClamp(abs(contextReturn) / volatility, 0.0, 4.0)
            sumScore += fresh * (
                0.45 * abs(correlation) +
                    0.30 * relative +
                    0.15 * lead +
                    0.10 * magnitude
            )
            used += 1
        }
        return used > 0 ? sumScore / Double(used) : -1e9
    }
}
