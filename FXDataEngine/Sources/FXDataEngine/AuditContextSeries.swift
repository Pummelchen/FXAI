import Foundation

public struct AuditAsSeriesOHLCV: Codable, Hashable, Sendable {
    public var timeUTC: [Int64]
    public var open: [Double]
    public var high: [Double]
    public var low: [Double]
    public var close: [Double]
    public var volume: [Double]
    public var fillRiskPoints: [Double]

    public var count: Int { close.count }
    public var isConsistent: Bool {
        timeUTC.count == count &&
            open.count == count &&
            high.count == count &&
            low.count == count &&
            volume.count == count &&
            fillRiskPoints.count == count
    }

    public init(
        timeUTC: [Int64] = [],
        open: [Double] = [],
        high: [Double] = [],
        low: [Double] = [],
        close: [Double] = [],
        volume: [Double] = [],
        fillRiskPoints: [Double] = []
    ) {
        self.timeUTC = timeUTC
        self.open = open.map { fxSafeFinite($0) }
        self.high = high.map { fxSafeFinite($0) }
        self.low = low.map { fxSafeFinite($0) }
        self.close = close.map { fxSafeFinite($0) }
        self.volume = volume.map { max(0.0, fxSafeFinite($0)) }
        self.fillRiskPoints = fillRiskPoints.map { max(0.0, fxSafeFinite($0)) }
    }
}

public struct AuditAggregatedCandleState: Codable, Hashable, Sendable {
    public var bodyBias: Double
    public var closeLocation: Double
    public var rangePressure: Double
    public var liquidityPressure: Double

    public init(
        bodyBias: Double = 0.0,
        closeLocation: Double = 0.0,
        rangePressure: Double = 0.0,
        liquidityPressure: Double = 0.0
    ) {
        self.bodyBias = fxClamp(bodyBias, -1.2, 1.2)
        self.closeLocation = fxClamp(closeLocation, -1.2, 1.2)
        self.rangePressure = fxClamp(rangePressure, -6.0, 6.0)
        self.liquidityPressure = fxClamp(liquidityPressure, -6.0, 8.0)
    }
}

public struct AuditContextFeatureSet: Codable, Hashable, Sendable {
    public var mean: [Double]
    public var standardDeviation: [Double]
    public var upRatio: [Double]
    public var extra: [Double]

    public var count: Int { mean.count }

    public init(count: Int) {
        let safeCount = max(0, count)
        self.mean = Array(repeating: 0.0, count: safeCount)
        self.standardDeviation = Array(repeating: 0.0, count: safeCount)
        self.upRatio = Array(repeating: 0.0, count: safeCount)
        self.extra = Array(repeating: 0.0, count: safeCount * FXDataEngineConstants.contextExtraFeatures)
    }

    public func extraValue(sampleIndex: Int, featureIndex: Int, default defaultValue: Double = 0.0) -> Double {
        guard let index = AuditContextSeriesTools.contextExtraIndex(
            sampleIndex: sampleIndex,
            featureIndex: featureIndex
        ),
              index < extra.count else {
            return defaultValue
        }
        return fxSafeFinite(extra[index])
    }

    public mutating func setExtraValue(sampleIndex: Int, featureIndex: Int, value: Double) {
        guard let index = AuditContextSeriesTools.contextExtraIndex(
            sampleIndex: sampleIndex,
            featureIndex: featureIndex
        ),
              index < extra.count else {
            return
        }
        extra[index] = fxSafeFinite(value)
    }
}

public enum AuditContextSeriesTools {
    public static func contextExtraIndex(sampleIndex: Int, featureIndex: Int) -> Int? {
        guard sampleIndex >= 0,
              featureIndex >= 0,
              featureIndex < FXDataEngineConstants.contextExtraFeatures else {
            return nil
        }
        return sampleIndex * FXDataEngineConstants.contextExtraFeatures + featureIndex
    }

    public static func contextMTFBarsForSlot(_ timeframeSlot: Int) -> Int {
        switch timeframeSlot {
        case 0: 1
        case 1: 5
        case 2: 15
        case 3: 30
        case 4: 60
        default: 1
        }
    }

    public static func contextSlotMTFExtraIndex(slot: Int, timeframeSlot: Int, metric: MTFStateMetric) -> Int? {
        guard slot >= 0,
              slot < FXDataEngineConstants.contextTopSymbols,
              timeframeSlot >= 0,
              timeframeSlot < FXDataEngineConstants.contextMTFTimeframeCount else {
            return nil
        }
        return FXDataEngineConstants.contextMTFOffset +
            slot * FXDataEngineConstants.contextSlotMTFFeatures +
            timeframeSlot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe +
            metric.rawValue
    }

    public static func rollingCorrelationAsSeries(
        _ a: [Double],
        _ b: [Double],
        startIndex: Int,
        width: Int
    ) -> Double {
        let count = a.count
        guard count == b.count,
              width >= 4,
              startIndex >= 0,
              startIndex + width < count else {
            return 0.0
        }

        var sumA = 0.0
        var sumB = 0.0
        var sumAA = 0.0
        var sumBB = 0.0
        var sumAB = 0.0
        var used = 0

        for offset in 0..<width {
            let returnA = FeatureMath.safeReturn(
                a,
                currentIndex: startIndex + offset,
                previousIndex: startIndex + offset + 1
            )
            let returnB = FeatureMath.safeReturn(
                b,
                currentIndex: startIndex + offset,
                previousIndex: startIndex + offset + 1
            )
            sumA += returnA
            sumB += returnB
            sumAA += returnA * returnA
            sumBB += returnB * returnB
            sumAB += returnA * returnB
            used += 1
        }

        guard used >= 4 else { return 0.0 }
        let countDouble = Double(used)
        let meanA = sumA / countDouble
        let meanB = sumB / countDouble
        let varianceA = sumAA / countDouble - meanA * meanA
        let varianceB = sumBB / countDouble - meanB * meanB
        let covariance = sumAB / countDouble - meanA * meanB
        guard varianceA > 1e-12, varianceB > 1e-12 else { return 0.0 }
        return fxClamp(covariance / sqrt(varianceA * varianceB), -1.0, 1.0)
    }

    public static func reverseChronologicalBarsToSeries(_ bars: [AuditScenarioDoubleBar]) -> AuditAsSeriesOHLCV {
        let reversed = bars.reversed()
        return AuditAsSeriesOHLCV(
            timeUTC: reversed.map(\.timestampUTC),
            open: reversed.map(\.open),
            high: reversed.map(\.high),
            low: reversed.map(\.low),
            close: reversed.map(\.close),
            volume: reversed.map(\.volume),
            fillRiskPoints: reversed.map(\.fillRiskPoints)
        )
    }

    public static func reverseChronologicalCloseSeries(
        timeUTC: [Int64],
        close: [Double]
    ) -> (timeUTC: [Int64], close: [Double]) {
        let count = min(timeUTC.count, close.count)
        guard count > 0 else { return ([], []) }
        let indexes = (0..<count).reversed()
        return (
            indexes.map { timeUTC[$0] },
            indexes.map { fxSafeFinite(close[$0]) }
        )
    }

    public static func aggregateCloseTimeframe(
        chronologicalBars: [AuditScenarioDoubleBar],
        step: Int
    ) -> (timeUTC: [Int64], close: [Double]) {
        let count = chronologicalBars.count
        let bars = step > 0 ? count / step : 0
        guard bars > 0 else { return ([], []) }

        var timeUTC: [Int64] = []
        var close: [Double] = []
        timeUTC.reserveCapacity(bars)
        close.reserveCapacity(bars)
        for bucket in 0..<bars {
            let end = min(count - 1, bucket * step + step - 1)
            timeUTC.append(chronologicalBars[end].timestampUTC)
            close.append(chronologicalBars[end].close)
        }
        return reverseChronologicalCloseSeries(timeUTC: timeUTC, close: close)
    }

    public static func deriveContextSeriesFromBase(
        point: Double,
        base: AuditAsSeriesOHLCV,
        transformID: Int
    ) -> AuditAsSeriesOHLCV {
        let count = base.count
        guard count > 0, base.isConsistent else { return AuditAsSeriesOHLCV() }

        var contextOpen = Array(repeating: 0.0, count: count)
        var contextHigh = Array(repeating: 0.0, count: count)
        var contextLow = Array(repeating: 0.0, count: count)
        var contextClose = Array(repeating: 0.0, count: count)
        var contextVolume = Array(repeating: 0.0, count: count)
        var contextFillRisk = Array(repeating: 0.0, count: count)
        let pt = point > 0.0 ? point : 1e-5

        for index in stride(from: count - 1, through: 0, by: -1) {
            let closeBase = value(base.close, index: index, default: 0.0)
            let previousBase = value(base.close, index: index + 1, default: closeBase)
            let returnBase = previousBase > 0.0 ? (closeBase - previousBase) / previousBase : 0.0
            var closeValue = closeBase
            var gapScale = 0.60
            var rangeScale = 1.0
            var fillRiskScale = 1.0
            var volumeScale = 1.0

            if transformID == 0 {
                closeValue = closeBase * (1.0 + 0.60 * returnBase)
                gapScale = 0.70
                rangeScale = 1.05
                fillRiskScale = 1.05 + 0.20 * fxClamp(abs(returnBase) * 4_000.0, 0.0, 1.0)
                volumeScale = 1.05
            } else if transformID == 1 {
                closeValue = 0.65 * closeBase + 0.35 * previousBase
                gapScale = 0.45
                rangeScale = 0.90
                fillRiskScale = 0.95 + 0.10 * fxClamp(abs(returnBase) * 3_000.0, 0.0, 1.0)
                volumeScale = 0.95
            } else {
                closeValue = closeBase * (1.0 - 0.35 * returnBase)
                gapScale = -0.25
                rangeScale = 1.08
                fillRiskScale = 1.08 + 0.15 * fxClamp(abs(returnBase) * 3_500.0, 0.0, 1.0)
                volumeScale = 1.08
            }

            if closeValue <= pt {
                closeValue = max(closeBase, pt)
            }

            let previousContext = value(contextClose, index: index + 1, default: closeValue)
            let openBase = value(base.open, index: index, default: closeBase)
            let openGap = previousBase > 0.0 ? (openBase - previousBase) / previousBase : 0.0
            var openValue = previousContext * (1.0 + gapScale * openGap)
            if openValue <= pt {
                openValue = previousContext
            }
            if openValue <= pt {
                openValue = closeValue
            }

            let highBase = value(base.high, index: index, default: max(openBase, closeBase))
            let lowBase = value(base.low, index: index, default: min(openBase, closeBase))
            let baseRange = max(highBase - lowBase, pt)
            let baseBodyHigh = max(openBase, closeBase)
            let baseBodyLow = min(openBase, closeBase)
            let upperRatio = max(highBase - baseBodyHigh, 0.0) / baseRange
            let lowerRatio = max(baseBodyLow - lowBase, 0.0) / baseRange
            let scaledRange = rangeScale * baseRange * max(closeValue / max(closeBase, pt), 0.25)
            let upperWick = max(0.25 * pt, scaledRange * upperRatio)
            let lowerWick = max(0.25 * pt, scaledRange * lowerRatio)
            let bodyHigh = max(openValue, closeValue)
            let bodyLow = min(openValue, closeValue)

            contextOpen[index] = openValue
            contextHigh[index] = bodyHigh + upperWick
            contextLow[index] = max(pt, bodyLow - lowerWick)
            contextClose[index] = closeValue
            contextVolume[index] = max(0.0, value(base.volume, index: index, default: 0.0) * volumeScale)
            contextFillRisk[index] = max(0.0, value(base.fillRiskPoints, index: index, default: 0.0) * fillRiskScale)
        }

        return AuditAsSeriesOHLCV(
            timeUTC: base.timeUTC,
            open: contextOpen,
            high: contextHigh,
            low: contextLow,
            close: contextClose,
            volume: contextVolume,
            fillRiskPoints: contextFillRisk
        )
    }

    public static func computeAggregatedCandleLiquidityState(
        index: Int,
        windowBars: Int,
        series: AuditAsSeriesOHLCV,
        point: Double
    ) -> AuditAggregatedCandleState? {
        let count = series.count
        guard series.isConsistent,
              index >= 0,
              windowBars >= 1,
              count > 0 else {
            return nil
        }
        let last = index + windowBars - 1
        guard last < count else { return nil }

        let pointValue = point > 0.0 ? point : 1.0
        let aggregateOpen = value(series.open, index: last, default: 0.0)
        let aggregateClose = value(series.close, index: index, default: 0.0)
        var aggregateHigh = value(series.high, index: index, default: max(aggregateOpen, aggregateClose))
        var aggregateLow = value(series.low, index: index, default: min(aggregateOpen, aggregateClose))
        var fillRiskSumCurrent = 0.0
        var volumeSumCurrent = 0.0
        var currentUsed = 0

        for offset in 0..<windowBars {
            let currentIndex = index + offset
            if currentIndex < 0 || currentIndex >= count { break }
            aggregateHigh = max(aggregateHigh, value(series.high, index: currentIndex, default: aggregateHigh))
            aggregateLow = min(aggregateLow, value(series.low, index: currentIndex, default: aggregateLow))
            fillRiskSumCurrent += max(0.0, value(series.fillRiskPoints, index: currentIndex, default: 0.0))
            volumeSumCurrent += max(0.0, value(series.volume, index: currentIndex, default: 0.0))
            currentUsed += 1
        }

        let range = max(aggregateHigh - aggregateLow, pointValue)
        let rangePoints = max(0.0, (aggregateHigh - aggregateLow) / pointValue)
        let fillRiskCurrent = fillRiskSumCurrent / Double(max(currentUsed, 1))
        let volumeCurrent = volumeSumCurrent / Double(max(currentUsed, 1))

        var averageRangePoints = 0.0
        var averageFillRisk = 0.0
        var averageVolume = 0.0
        var windowsUsed = 0

        for window in 0..<20 {
            let baseIndex = index + window * windowBars
            let baseLast = baseIndex + windowBars - 1
            if baseIndex < 0 || baseLast >= count { break }

            var windowHigh = value(series.high, index: baseIndex, default: 0.0)
            var windowLow = value(series.low, index: baseIndex, default: windowHigh)
            var windowFillRiskSum = 0.0
            var windowVolumeSum = 0.0
            var windowUsed = 0
            for offset in 0..<windowBars {
                let currentIndex = baseIndex + offset
                if currentIndex < 0 || currentIndex >= count { break }
                windowHigh = max(windowHigh, value(series.high, index: currentIndex, default: windowHigh))
                windowLow = min(windowLow, value(series.low, index: currentIndex, default: windowLow))
                windowFillRiskSum += max(0.0, value(series.fillRiskPoints, index: currentIndex, default: 0.0))
                windowVolumeSum += max(0.0, value(series.volume, index: currentIndex, default: 0.0))
                windowUsed += 1
            }

            averageRangePoints += max(0.0, (windowHigh - windowLow) / pointValue)
            averageFillRisk += windowFillRiskSum / Double(max(windowUsed, 1))
            averageVolume += windowVolumeSum / Double(max(windowUsed, 1))
            windowsUsed += 1
        }

        if windowsUsed <= 0 {
            averageRangePoints = max(rangePoints, 0.25)
            averageFillRisk = max(fillRiskCurrent, 0.25)
            averageVolume = max(volumeCurrent, 0.0)
        } else {
            averageRangePoints /= Double(windowsUsed)
            averageFillRisk /= Double(windowsUsed)
            averageVolume /= Double(windowsUsed)
        }

        let costPressure = (fillRiskCurrent / max(averageFillRisk, 0.25)) - 1.0
        let volumeStress: Double
        if averageVolume > 0.0, volumeCurrent > 0.0 {
            volumeStress = (averageVolume / max(volumeCurrent, 1e-9)) - 1.0
        } else {
            volumeStress = 0.0
        }
        let liquidityPressure = averageVolume > 0.0
            ? 0.65 * costPressure + 0.35 * volumeStress
            : costPressure

        return AuditAggregatedCandleState(
            bodyBias: (aggregateClose - aggregateOpen) / range,
            closeLocation: ((aggregateClose - aggregateLow) - (aggregateHigh - aggregateClose)) / range,
            rangePressure: fxClamp((rangePoints / max(averageRangePoints, 0.25)) - 1.0, -6.0, 6.0),
            liquidityPressure: fxClamp(liquidityPressure, -8.0, 8.0)
        )
    }

    public static func buildContextFeatures(
        mainClose: [Double],
        point: Double,
        contexts: [AuditAsSeriesOHLCV]
    ) -> AuditContextFeatureSet {
        let count = mainClose.count
        var output = AuditContextFeatureSet(count: count)
        let topContexts = Array(contexts.prefix(FXDataEngineConstants.contextTopSymbols))
        let consistentContextCount = topContexts.filter(\.isConsistent).count

        for index in 0..<count {
            let mainReturn = FeatureMath.safeReturn(mainClose, currentIndex: index, previousIndex: index + 1)
            var returns = Array(repeating: 0.0, count: FXDataEngineConstants.contextTopSymbols)
            var lags = Array(repeating: 0.0, count: FXDataEngineConstants.contextTopSymbols)

            for slot in 0..<FXDataEngineConstants.contextTopSymbols {
                guard slot < topContexts.count, topContexts[slot].isConsistent else { continue }
                let context = topContexts[slot]
                returns[slot] = FeatureMath.safeReturn(context.close, currentIndex: index, previousIndex: index + 1)
                lags[slot] = FeatureMath.safeReturn(context.close, currentIndex: index + 1, previousIndex: index + 2)
            }

            let sum = returns.reduce(0.0, +)
            let sumSquares = returns.reduce(0.0) { $0 + $1 * $1 }
            let mean = sum / Double(FXDataEngineConstants.contextTopSymbols)
            let variance = max(sumSquares / Double(FXDataEngineConstants.contextTopSymbols) - mean * mean, 0.0)
            let upCount = returns.reduce(0) { $0 + ($1 > 0.0 ? 1 : 0) }
            output.mean[index] = mean
            output.standardDeviation[index] = sqrt(variance)
            output.upRatio[index] = Double(upCount) / Double(FXDataEngineConstants.contextTopSymbols)

            for slot in 0..<FXDataEngineConstants.contextTopSymbols {
                let correlation = slot < topContexts.count && topContexts[slot].isConsistent
                    ? rollingCorrelationAsSeries(mainClose, topContexts[slot].close, startIndex: index, width: 16)
                    : 0.0
                output.setExtraValue(sampleIndex: index, featureIndex: slot * 4, value: returns[slot])
                output.setExtraValue(sampleIndex: index, featureIndex: slot * 4 + 1, value: lags[slot])
                output.setExtraValue(sampleIndex: index, featureIndex: slot * 4 + 2, value: returns[slot] - mainReturn)
                output.setExtraValue(sampleIndex: index, featureIndex: slot * 4 + 3, value: correlation)
            }

            var mainVolatility = FeatureMath.rollingAbsReturnAsSeries(mainClose, startIndex: index, width: 20)
            if mainVolatility < 1e-6 {
                mainVolatility = abs(mainReturn)
            }
            if mainVolatility < 1e-6 {
                mainVolatility = 1e-4
            }

            var stability = 1.0
            for slot in 0..<FXDataEngineConstants.contextTopSymbols {
                stability -= 0.20 * fxClamp(abs(returns[slot] - lags[slot]) / max(mainVolatility, 1e-4), 0.0, 1.0)
            }
            var lead = 0.0
            for slot in 0..<FXDataEngineConstants.contextTopSymbols {
                lead += fxClamp(abs(lags[slot]) / max(mainVolatility, 1e-4), 0.0, 4.0) / 4.0
            }
            lead /= Double(FXDataEngineConstants.contextTopSymbols)

            output.setExtraValue(
                sampleIndex: index,
                featureIndex: FXDataEngineConstants.contextSharedOffset,
                value: fxClamp(mean / max(mainVolatility, 1e-4), -1.0, 1.0)
            )
            output.setExtraValue(
                sampleIndex: index,
                featureIndex: FXDataEngineConstants.contextSharedOffset + 1,
                value: fxClamp(stability, 0.0, 1.0)
            )
            output.setExtraValue(
                sampleIndex: index,
                featureIndex: FXDataEngineConstants.contextSharedOffset + 2,
                value: fxClamp(lead, 0.0, 1.0)
            )
            output.setExtraValue(
                sampleIndex: index,
                featureIndex: FXDataEngineConstants.contextSharedOffset + 3,
                value: Double(min(consistentContextCount, FXDataEngineConstants.contextTopSymbols)) /
                    Double(FXDataEngineConstants.contextTopSymbols)
            )

            for slot in 0..<min(topContexts.count, FXDataEngineConstants.contextTopSymbols) {
                setContextSlotMTFExtras(
                    featureSet: &output,
                    sampleIndex: index,
                    topSlot: slot,
                    point: point,
                    series: topContexts[slot]
                )
            }
        }

        return output
    }

    public static func setContextSlotMTFExtras(
        featureSet: inout AuditContextFeatureSet,
        sampleIndex: Int,
        topSlot: Int,
        point: Double,
        series: AuditAsSeriesOHLCV
    ) {
        for timeframeSlot in 0..<FXDataEngineConstants.contextMTFTimeframeCount {
            guard let state = computeAggregatedCandleLiquidityState(
                index: sampleIndex,
                windowBars: contextMTFBarsForSlot(timeframeSlot),
                series: series,
                point: point
            ) else {
                continue
            }

            if let bodyIndex = contextSlotMTFExtraIndex(slot: topSlot, timeframeSlot: timeframeSlot, metric: .bodyBias) {
                featureSet.setExtraValue(sampleIndex: sampleIndex, featureIndex: bodyIndex, value: state.bodyBias)
            }
            if let locationIndex = contextSlotMTFExtraIndex(slot: topSlot, timeframeSlot: timeframeSlot, metric: .closeLocation) {
                featureSet.setExtraValue(sampleIndex: sampleIndex, featureIndex: locationIndex, value: state.closeLocation)
            }
            if let rangeIndex = contextSlotMTFExtraIndex(slot: topSlot, timeframeSlot: timeframeSlot, metric: .rangePressure) {
                featureSet.setExtraValue(sampleIndex: sampleIndex, featureIndex: rangeIndex, value: state.rangePressure)
            }
            if let liquidityIndex = contextSlotMTFExtraIndex(slot: topSlot, timeframeSlot: timeframeSlot, metric: .volumePressure) {
                featureSet.setExtraValue(sampleIndex: sampleIndex, featureIndex: liquidityIndex, value: state.liquidityPressure)
            }
        }
    }

    private static func value(_ values: [Double], index: Int, default defaultValue: Double) -> Double {
        guard index >= 0, index < values.count else { return defaultValue }
        return fxSafeFinite(values[index])
    }
}
