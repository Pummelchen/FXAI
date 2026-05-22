import Foundation

public enum PluginSharedTransferPayloadTools {
    public static func buildInput(
        x: [Double],
        domainHash: Double,
        horizonMinutes: Int
    ) -> [Double] {
        baseInput(x: x, domainHash: domainHash, horizonMinutes: horizonMinutes)
    }

    public static func buildInput(
        x: [Double],
        window: [[Double]],
        declaredWindowSize: Int? = nil,
        domainHash: Double,
        horizonMinutes: Int
    ) -> [Double] {
        var output = baseInput(x: x, domainHash: domainHash, horizonMinutes: horizonMinutes)
        let windowSize = effectiveWindowSize(window, declaredWindowSize: declaredWindowSize)
        guard windowSize > 0 else {
            output[20] = feature(x, 0)
            output[21] = feature(x, 3)
            output[22] = feature(x, 41)
            output[23] = 0.50 * feature(x, 68) + 0.25 * feature(x, 69) + 0.25 * feature(x, 80)
            output[24] = 0.35 * feature(x, 10) + 0.25 * feature(x, 62) + 0.20 * feature(x, 63) + 0.20 * feature(x, 64)
            output[25] = 0.25 * macroFeature(x, 0) +
                0.15 * macroFeature(x, 1) +
                0.20 * macroFeature(x, 2) +
                0.20 * macroFeature(x, 4) +
                0.20 * macroFeature(x, 5)
            output[26] = 0.20 * feature(x, 72) +
                0.15 * feature(x, 73) +
                0.20 * feature(x, 74) +
                0.10 * feature(x, 75) +
                0.15 * feature(x, 78) +
                0.10 * (feature(x, 76) - feature(x, 77)) +
                0.10 * feature(x, 79)
            output[27] = 0.30 * feature(x, 79) +
                0.25 * abs(feature(x, 63)) +
                0.20 * abs(feature(x, 68)) +
                0.15 * feature(x, 82) +
                0.10 * abs(macroFeature(x, 4))
            return output
        }

        let retFast = windowFeatureEMAMean(window, featureIndex: 0, declaredWindowSize: windowSize)
        let retMid = windowFeatureEMAMean(window, featureIndex: 1, declaredWindowSize: windowSize)
        let retLong = windowFeatureEMAMean(window, featureIndex: 2, declaredWindowSize: windowSize)
        let slopeM1 = windowFeatureSlope(window, featureIndex: 3, declaredWindowSize: windowSize)
        let volumeMean = windowFeatureMean(window, featureIndex: 5, declaredWindowSize: windowSize)
        let volumeStd = windowFeatureStd(window, featureIndex: 5, declaredWindowSize: windowSize)
        let atrMean = windowFeatureMean(window, featureIndex: 41, declaredWindowSize: windowSize)
        let contextMean = windowFeatureMean(window, featureIndex: 10, declaredWindowSize: windowSize)
        let contextStd = windowFeatureStd(window, featureIndex: 10, declaredWindowSize: windowSize)
        let sharedUtility = windowFeatureEMAMean(window, featureIndex: 62, declaredWindowSize: windowSize)
        let sharedStability = windowFeatureMean(window, featureIndex: 63, declaredWindowSize: windowSize)
        let sharedLead = windowFeatureMean(window, featureIndex: 64, declaredWindowSize: windowSize)
        let sharedCoverage = windowFeatureMean(window, featureIndex: 65, declaredWindowSize: windowSize)
        let volumeShock = windowFeatureMean(window, featureIndex: 68, declaredWindowSize: windowSize)
        let volumeAccel = windowFeatureMean(window, featureIndex: 69, declaredWindowSize: windowSize)
        let volumeTrend = windowFeatureEMAMean(window, featureIndex: 71, declaredWindowSize: windowSize)
        let sessionTransition = windowFeatureMean(window, featureIndex: 72, declaredWindowSize: windowSize)
        let sessionOverlap = windowFeatureMean(window, featureIndex: 73, declaredWindowSize: windowSize)
        let volumeSessionActivity = windowFeatureMean(window, featureIndex: 74, declaredWindowSize: windowSize)
        let volumeAvailableFlag = windowFeatureMean(window, featureIndex: 75, declaredWindowSize: windowSize)
        let volumeBias = windowFeatureMean(window, featureIndex: 76, declaredWindowSize: windowSize) -
            windowFeatureMean(window, featureIndex: 77, declaredWindowSize: windowSize)
        let volumePersistence = windowFeatureMean(window, featureIndex: 78, declaredWindowSize: windowSize)
        let familyDrift = windowFeatureMean(window, featureIndex: 79, declaredWindowSize: windowSize)
        let volumeLog = windowFeatureMean(window, featureIndex: 80, declaredWindowSize: windowSize)
        let volumeZ = windowFeatureMean(window, featureIndex: 81, declaredWindowSize: windowSize)
        let volumeVolatility = windowFeatureMean(window, featureIndex: 82, declaredWindowSize: windowSize)
        let volumeRank = windowFeatureMean(window, featureIndex: 83, declaredWindowSize: windowSize)
        let macroPre = windowFeatureMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 0, declaredWindowSize: windowSize)
        let macroPost = windowFeatureMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 1, declaredWindowSize: windowSize)
        let macroImportance = windowFeatureMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 2, declaredWindowSize: windowSize)
        let macroSurprise = windowFeatureEMAMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 3, declaredWindowSize: windowSize)
        let macroSurpriseAbs = windowFeatureMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 4, declaredWindowSize: windowSize)
        let macroClass = windowFeatureMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 5, declaredWindowSize: windowSize)
        let macroPolicyDivergence = windowFeatureMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 14, declaredWindowSize: windowSize)
        let macroPolicyPressure = windowFeatureMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 15, declaredWindowSize: windowSize)
        let macroInflationPressure = windowFeatureMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 16, declaredWindowSize: windowSize)
        let macroGrowthPressure = windowFeatureMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 18, declaredWindowSize: windowSize)
        let macroStateQuality = windowFeatureMean(window, featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 19, declaredWindowSize: windowSize)
        let retDelta = windowFeatureRecentDelta(window, featureIndex: 0, recentBars: max(windowSize / 4, 3), declaredWindowSize: windowSize)
        let volumeDelta = windowFeatureRecentDelta(window, featureIndex: 80, recentBars: max(windowSize / 4, 3), declaredWindowSize: windowSize)

        output[20] = fxClamp(0.42 * retFast + 0.33 * retMid + 0.15 * retLong + 0.10 * retDelta, -4.0, 4.0)
        output[21] = fxClamp(0.38 * slopeM1 + 0.22 * (retFast - retMid) + 0.20 * volumeTrend +
            0.20 * windowFeatureSlope(window, featureIndex: 71, declaredWindowSize: windowSize), -4.0, 4.0)
        output[22] = fxClamp(0.34 * volumeMean + 0.22 * volumeStd + 0.22 * atrMean +
            0.22 * windowFeatureMean(window, featureIndex: 43, declaredWindowSize: windowSize), 0.0, 6.0)
        output[23] = fxClamp(0.26 * windowFeatureMean(window, featureIndex: 6, declaredWindowSize: windowSize) +
            0.24 * volumeShock +
            0.18 * volumeAccel +
            0.18 * volumeLog +
            0.14 * volumeZ, -6.0, 8.0)
        output[24] = fxClamp(0.28 * contextMean +
            0.14 * contextStd +
            0.22 * sharedUtility +
            0.14 * sharedStability +
            0.12 * sharedLead +
            0.10 * sharedCoverage, -4.0, 4.0)
        output[25] = fxClamp(0.18 * macroPre +
            0.12 * macroPost +
            0.22 * macroImportance +
            0.18 * macroSurprise +
            0.16 * macroSurpriseAbs +
            0.10 * macroClass +
            0.08 * macroPolicyPressure +
            0.06 * macroStateQuality, -6.0, 6.0)
        output[26] = fxClamp(0.18 * sessionTransition +
            0.14 * sessionOverlap +
            0.20 * volumeSessionActivity +
            0.12 * volumeAvailableFlag +
            0.14 * volumePersistence +
            0.12 * volumeBias +
            0.08 * familyDrift +
            0.10 * macroPolicyDivergence +
            0.08 * macroGrowthPressure, -4.0, 4.0)
        output[27] = fxClamp(0.24 * familyDrift +
            0.18 * windowFeatureStd(window, featureIndex: 63, declaredWindowSize: windowSize) +
            0.18 * windowFeatureRange(window, featureIndex: 10, declaredWindowSize: windowSize) +
            0.14 * abs(volumeDelta) +
            0.14 * volumeVolatility +
            0.10 * abs(volumeRank) +
            0.12 * macroInflationPressure +
            0.10 * macroStateQuality, 0.0, 6.0)
        return output
    }

    public static func windowFeatureMean(
        _ window: [[Double]],
        featureIndex: Int,
        declaredWindowSize: Int? = nil
    ) -> Double {
        let size = effectiveWindowSize(window, declaredWindowSize: declaredWindowSize)
        guard isValidFeatureIndex(featureIndex), size > 0 else { return 0.0 }
        var sum = 0.0
        for barIndex in 0..<size {
            sum += windowValue(window, barIndex: barIndex, featureIndex: featureIndex)
        }
        return sum / Double(size)
    }

    public static func windowFeatureEMAMean(
        _ window: [[Double]],
        featureIndex: Int,
        decay: Double = 0.72,
        declaredWindowSize: Int? = nil
    ) -> Double {
        let size = effectiveWindowSize(window, declaredWindowSize: declaredWindowSize)
        guard isValidFeatureIndex(featureIndex), size > 0 else { return 0.0 }
        let alpha = fxClamp(decay, 0.05, 0.98)
        var weight = 1.0
        var weightSum = 0.0
        var sum = 0.0
        for barIndex in 0..<size {
            sum += weight * windowValue(window, barIndex: barIndex, featureIndex: featureIndex)
            weightSum += weight
            weight *= alpha
        }
        return weightSum > 0.0 ? sum / weightSum : 0.0
    }

    public static func windowFeatureStd(
        _ window: [[Double]],
        featureIndex: Int,
        declaredWindowSize: Int? = nil
    ) -> Double {
        let size = effectiveWindowSize(window, declaredWindowSize: declaredWindowSize)
        guard isValidFeatureIndex(featureIndex), size > 1 else { return 0.0 }
        let mean = windowFeatureMean(window, featureIndex: featureIndex, declaredWindowSize: size)
        var accumulator = 0.0
        for barIndex in 0..<size {
            let delta = windowValue(window, barIndex: barIndex, featureIndex: featureIndex) - mean
            accumulator += delta * delta
        }
        return sqrt(accumulator / Double(max(size, 1)))
    }

    public static func windowFeatureRange(
        _ window: [[Double]],
        featureIndex: Int,
        recentBars: Int = 0,
        declaredWindowSize: Int? = nil
    ) -> Double {
        let size = effectiveWindowSize(window, declaredWindowSize: declaredWindowSize)
        guard isValidFeatureIndex(featureIndex), size > 0 else { return 0.0 }
        let count = recentBars <= 0 ? size : min(max(1, recentBars), size)
        var low = windowValue(window, barIndex: 0, featureIndex: featureIndex)
        var high = low
        for barIndex in 0..<count {
            let value = windowValue(window, barIndex: barIndex, featureIndex: featureIndex)
            low = min(low, value)
            high = max(high, value)
        }
        return high - low
    }

    public static func windowFeatureSlope(
        _ window: [[Double]],
        featureIndex: Int,
        declaredWindowSize: Int? = nil
    ) -> Double {
        let size = effectiveWindowSize(window, declaredWindowSize: declaredWindowSize)
        guard isValidFeatureIndex(featureIndex), size > 1 else { return 0.0 }
        let first = windowValue(window, barIndex: 0, featureIndex: featureIndex)
        let last = windowValue(window, barIndex: size - 1, featureIndex: featureIndex)
        return (first - last) / Double(max(size - 1, 1))
    }

    public static func windowFeatureRecentDelta(
        _ window: [[Double]],
        featureIndex: Int,
        recentBars: Int,
        declaredWindowSize: Int? = nil
    ) -> Double {
        let size = effectiveWindowSize(window, declaredWindowSize: declaredWindowSize)
        guard isValidFeatureIndex(featureIndex), size > 0 else { return 0.0 }
        var count = recentBars
        if count <= 1 {
            count = max(size / 4, 2)
        }
        count = min(count, size)
        let lastIndex = max(count - 1, 0)
        return windowValue(window, barIndex: 0, featureIndex: featureIndex) -
            windowValue(window, barIndex: lastIndex, featureIndex: featureIndex)
    }

    private static func baseInput(
        x: [Double],
        domainHash: Double,
        horizonMinutes: Int
    ) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.sharedTransferFeatures)
        output[0] = 1.0
        output[1] = feature(x, 62)
        output[2] = feature(x, 63)
        output[3] = feature(x, 64)
        output[4] = fxClamp(0.5 + 0.5 * feature(x, 65), 0.0, 1.0)

        var retMix = 0.0
        var lagMix = 0.0
        var relativeMix = 0.0
        var correlationMix = 0.0
        var weightTotal = 0.0
        for slot in 0..<FXDataEngineConstants.contextTopSymbols {
            let base = 50 + slot * 4
            let contextReturn = feature(x, base)
            let contextLag = feature(x, base + 1)
            let contextRelative = feature(x, base + 2)
            let contextCorrelation = feature(x, base + 3)
            let weight = fxClamp(
                (0.30 + 0.70 * abs(contextCorrelation)) *
                    (0.35 + 0.65 * output[4]) *
                    (0.35 + 0.25 * abs(contextReturn) + 0.25 * abs(contextLag) + 0.15 * abs(contextRelative)),
                0.0,
                3.0
            )
            guard weight > 1e-6 else { continue }
            retMix += weight * contextReturn
            lagMix += weight * contextLag
            relativeMix += weight * contextRelative
            correlationMix += weight * contextCorrelation
            weightTotal += weight
        }
        if weightTotal > 1e-6 {
            output[5] = retMix / weightTotal
            output[6] = lagMix / weightTotal
            output[7] = relativeMix / weightTotal
            output[8] = correlationMix / weightTotal
        }

        let domain = fxClamp(domainHash, 0.0, 1.0)
        let horizonScale = fxClamp(log(1.0 + Double(max(horizonMinutes, 1))) / log(1.0 + 1440.0), 0.0, 1.0)
        let mainMTF = mainMTFSummary(x)
        let contextMTF = contextMTFSummary(x)
        let macroPre = macroFeature(x, 0)
        let macroPost = macroFeature(x, 1)
        let macroImportance = macroFeature(x, 2)
        let macroSurprise = macroFeature(x, 3)
        let macroSurpriseAbs = macroFeature(x, 4)
        let macroClass = macroFeature(x, 5)
        let macroSurpriseZ = macroFeature(x, 6)
        let macroRevisionAbs = macroFeature(x, 7)
        let macroCurrencyRelevance = macroFeature(x, 8)
        let macroProvenance = macroFeature(x, 9)
        let macroPolicyDivergence = macroFeature(x, 14)
        let macroPolicyPressure = macroFeature(x, 15)
        let macroInflationPressure = macroFeature(x, 16)
        let macroGrowthPressure = macroFeature(x, 18)
        let macroStateQuality = macroFeature(x, 19)

        output[9] = 2.0 * domain - 1.0
        output[10] = 2.0 * horizonScale - 1.0
        output[11] = fxClamp(0.60 * feature(x, 72) +
            0.15 * feature(x, 73) +
            0.10 * macroPre -
            0.08 * macroPost +
            0.07 * mainMTF.location +
            0.06 * contextMTF.location, -1.0, 1.0)
        output[12] = fxClamp(0.28 * feature(x, 74) +
            0.16 * feature(x, 75) +
            0.14 * feature(x, 78) +
            0.10 * macroImportance +
            0.08 * macroClass +
            0.08 * mainMTF.body +
            0.06 * contextMTF.body, -1.0, 1.0)
        output[13] = fxClamp(0.16 * feature(x, 76) -
            0.16 * feature(x, 77) -
            0.10 * feature(x, 79) +
            0.08 * feature(x, 6) +
            0.08 * feature(x, 81) +
            0.06 * macroSurpriseAbs -
            0.05 * mainMTF.volumePressure -
            0.05 * feature(x, 82) +
            0.04 * contextMTF.volumePressure, -4.0, 4.0)
        output[14] = fxClamp(0.42 * feature(x, 18) +
            0.18 * feature(x, 19) -
            0.18 * feature(x, 20) +
            0.12 * feature(x, 21) +
            0.06 * macroImportance +
            0.08 * mainMTF.body +
            0.08 * mainMTF.location, -4.0, 4.0)
        output[15] = fxClamp(0.16 * feature(x, 66) +
            0.12 * feature(x, 67) +
            0.12 * feature(x, 68) +
            0.10 * feature(x, 69) +
            0.14 * feature(x, 71) +
            0.10 * macroSurprise +
            0.08 * macroSurpriseAbs +
            0.06 * feature(x, 81) +
            0.04 * feature(x, 83) +
            0.05 * mainMTF.range +
            0.05 * contextMTF.range, -6.0, 6.0)
        output[16] = fxClamp(0.48 * feature(x, 68) +
            0.18 * feature(x, 81) +
            0.12 * feature(x, 80) +
            0.10 * macroImportance +
            0.08 * macroSurpriseAbs +
            0.08 * mainMTF.volumePressure, -4.0, 8.0)
        output[17] = fxClamp(0.55 * feature(x, 70) +
            0.18 * feature(x, 82) +
            0.10 * macroPost +
            0.08 * macroSurpriseAbs +
            0.05 * mainMTF.range +
            0.04 * abs(feature(x, 83)), 0.0, 8.0)
        output[18] = fxClamp(0.45 * macroSurprise +
            0.20 * macroClass +
            0.15 * feature(x, 78) +
            0.10 * feature(x, 72) +
            0.10 * feature(x, 73) +
            0.12 * macroSurpriseZ +
            0.10 * macroPolicyDivergence +
            0.08 * macroCurrencyRelevance, -6.0, 6.0)
        output[19] = fxClamp(0.40 * macroSurpriseAbs +
            0.22 * macroImportance +
            0.18 * macroPre +
            0.10 * macroPost +
            0.10 * feature(x, 79) +
            0.10 * macroRevisionAbs +
            0.08 * macroProvenance +
            0.08 * macroPolicyPressure +
            0.06 * macroInflationPressure +
            0.06 * macroGrowthPressure +
            0.08 * macroStateQuality, 0.0, 6.0)
        return output
    }

    private static func mainMTFSummary(_ x: [Double]) -> (body: Double, location: Double, range: Double, volumePressure: Double) {
        var body = 0.0
        var location = 0.0
        var range = 0.0
        var volumePressure = 0.0
        for slot in 0..<FXDataEngineConstants.mainMTFTimeframeCount {
            let base = FXDataEngineConstants.mainMTFFeatureOffset +
                slot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            body += feature(x, base)
            location += feature(x, base + 1)
            range += feature(x, base + 2)
            volumePressure += feature(x, base + 3)
        }
        let count = Double(max(FXDataEngineConstants.mainMTFTimeframeCount, 1))
        return (body / count, location / count, range / count, volumePressure / count)
    }

    private static func contextMTFSummary(_ x: [Double]) -> (body: Double, location: Double, range: Double, volumePressure: Double) {
        var body = 0.0
        var location = 0.0
        var range = 0.0
        var volumePressure = 0.0
        var weightTotal = 0.0
        for slot in 0..<FXDataEngineConstants.contextTopSymbols {
            let slotCorrelation = abs(feature(x, 50 + slot * 4 + 3))
            let slotWeight = 0.35 + 0.65 * slotCorrelation
            var slotBody = 0.0
            var slotLocation = 0.0
            var slotRange = 0.0
            var slotVolumePressure = 0.0
            var slotUsed = 0
            for timeframeSlot in 0..<FXDataEngineConstants.contextMTFTimeframeCount {
                let base = FXDataEngineConstants.contextMTFFeatureOffset +
                    slot * FXDataEngineConstants.contextSlotMTFFeatures +
                    timeframeSlot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe
                slotBody += feature(x, base)
                slotLocation += feature(x, base + 1)
                slotRange += feature(x, base + 2)
                slotVolumePressure += feature(x, base + 3)
                slotUsed += 1
            }
            guard slotUsed > 0 else { continue }
            let count = Double(slotUsed)
            body += slotWeight * slotBody / count
            location += slotWeight * slotLocation / count
            range += slotWeight * slotRange / count
            volumePressure += slotWeight * slotVolumePressure / count
            weightTotal += slotWeight
        }
        guard weightTotal > 1e-6 else { return (0.0, 0.0, 0.0, 0.0) }
        return (body / weightTotal, location / weightTotal, range / weightTotal, volumePressure / weightTotal)
    }

    private static func macroFeature(_ x: [Double], _ relativeIndex: Int) -> Double {
        feature(x, FXDataEngineConstants.macroEventFeatureOffset + relativeIndex)
    }

    private static func feature(_ x: [Double], _ featureIndex: Int) -> Double {
        PluginTransferSupportTools.inputFeature(x, featureIndex: featureIndex)
    }

    private static func isValidFeatureIndex(_ featureIndex: Int) -> Bool {
        let inputIndex = featureIndex + 1
        return inputIndex >= 1 && inputIndex < FXDataEngineConstants.aiWeights
    }

    private static func effectiveWindowSize(_ window: [[Double]], declaredWindowSize: Int?) -> Int {
        let requested = declaredWindowSize.map { max(0, $0) } ?? window.count
        return min(requested, window.count)
    }

    private static func windowValue(_ window: [[Double]], barIndex: Int, featureIndex: Int) -> Double {
        let inputIndex = featureIndex + 1
        guard barIndex >= 0,
              barIndex < window.count,
              inputIndex >= 0,
              inputIndex < window[barIndex].count
        else { return 0.0 }
        return fxSafeFinite(window[barIndex][inputIndex])
    }
}
