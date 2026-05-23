import XCTest
@testable import FXDataEngine

final class NormalizationStateTests: XCTestCase {
    func testNormalizationWindowClampAndTargetDefaultsMatchLegacyRules() {
        XCTAssertEqual(NormalizationWindowTools.clamp(2), 16)
        XCTAssertEqual(NormalizationWindowTools.clamp(999), RuntimeArtifactConstants.normalizationRollWindowMax)
        XCTAssertEqual(NormalizationWindowTools.defaultWindow(predictionTargetMinutes: 1), 128)
        XCTAssertEqual(NormalizationWindowTools.defaultWindow(predictionTargetMinutes: 5), 192)
        XCTAssertEqual(NormalizationWindowTools.defaultWindow(predictionTargetMinutes: 30), 256)
    }

    func testBuildGroupWindowsMatchesLegacyFeatureBands() {
        let windows = NormalizationWindowTools.buildGroupWindows(fast: 64, mid: 192, slow: 256, regime: 384)

        XCTAssertEqual(windows.count, FXDataEngineConstants.aiFeatures)
        XCTAssertEqual(windows[0], 64)
        XCTAssertEqual(windows[6], 64)
        XCTAssertEqual(windows[7], 192)
        XCTAssertEqual(windows[15], 384)
        XCTAssertEqual(windows[22], 256)
        XCTAssertEqual(windows[34], 192)
        XCTAssertEqual(windows[66], 64)
        XCTAssertEqual(windows[72], 384)
        XCTAssertEqual(windows[76], 256)
        XCTAssertEqual(windows[79], 192)
    }

    func testMetaSupportSamplingAndShadowRulesMatchLegacyHelpers() {
        XCTAssertEqual(
            NormalizationMetaSupportTools.barRandom01(barTimeUTC: 0, salt: 0),
            0.66991,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            NormalizationMetaSupportTools.barRandom01(barTimeUTC: 1_704_067_200, salt: 3),
            0.10814,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            NormalizationMetaSupportTools.barRandom01(barTimeUTC: -1, salt: 0),
            0.67312,
            accuracy: 1e-12
        )

        XCTAssertFalse(NormalizationMetaSupportTools.shouldSampleByPercent(barTimeUTC: 1_704_067_200, salt: 3, percent: 0.0))
        XCTAssertTrue(NormalizationMetaSupportTools.shouldSampleByPercent(barTimeUTC: 1_704_067_200, salt: 3, percent: 100.0))
        XCTAssertTrue(NormalizationMetaSupportTools.shouldSampleByPercent(barTimeUTC: 1_704_067_200, salt: 3, percent: 11.0))
        XCTAssertFalse(NormalizationMetaSupportTools.shouldSampleByPercent(barTimeUTC: 1_704_067_200, salt: 3, percent: 10.8))

        XCTAssertFalse(NormalizationMetaSupportTools.isShadowBar(cadenceBars: 0, barSequence: 10))
        XCTAssertTrue(NormalizationMetaSupportTools.isShadowBar(cadenceBars: 1, barSequence: -5))
        XCTAssertFalse(NormalizationMetaSupportTools.isShadowBar(cadenceBars: 4, barSequence: -1))
        XCTAssertTrue(NormalizationMetaSupportTools.isShadowBar(cadenceBars: 4, barSequence: 12))
        XCTAssertFalse(NormalizationMetaSupportTools.isShadowBar(cadenceBars: 4, barSequence: 13))
    }

    func testMetaSupportNormalizationMethodCandidatesMatchLegacyOrdering() {
        XCTAssertEqual(AIModelID.allCases.count, FXDataEngineConstants.aiCount)
        XCTAssertEqual(AIModelID.lstm.rawValue, 7)
        XCTAssertEqual(AIModelID.qcew.rawValue, 32)
        XCTAssertEqual(AIModelID.mythosRDT.rawValue, 62)
        XCTAssertEqual(AIModelID.demoMovingAverageCross.rawValue, 63)
        XCTAssertEqual(AIModelID.demoFXStupid.rawValue, 64)
        XCTAssertEqual(AIModelID.demoFX7.rawValue, 65)
        XCTAssertTrue(AIModelID.lstm.usesDeepNormalizationCandidates)
        XCTAssertFalse(AIModelID.gru.usesDeepNormalizationCandidates)

        XCTAssertEqual(NormalizationMetaSupportTools.sanitizeNormalizationMethod(-1), .existing)
        XCTAssertEqual(NormalizationMetaSupportTools.sanitizeNormalizationMethod(FeatureNormalizationMethod.dain.rawValue), .dain)
        XCTAssertEqual(NormalizationMetaSupportTools.sanitizeNormalizationMethod(99), .existing)

        XCTAssertEqual(
            NormalizationMetaSupportTools.normalizationMethodCandidates(aiID: AIModelID.lstm.rawValue, currentMethod: .dain),
            [
                .dain,
                .existing,
                .volatilityStdReturns,
                .atrNatrUnit,
                .zScore,
                .revin,
                .robustMedianIQR,
                .minMaxBuffer3
            ]
        )
        XCTAssertEqual(
            NormalizationMetaSupportTools.normalizationMethodCandidates(aiID: AIModelID.catboost.rawValue, currentMethod: .existing),
            [
                .existing,
                .zScore,
                .robustMedianIQR,
                .quantileToNormal,
                .changePercent,
                .volatilityStdReturns,
                .atrNatrUnit,
                .powerYeoJohnson,
                .minMaxBuffer3
            ]
        )
        XCTAssertEqual(
            NormalizationMetaSupportTools.normalizationMethodCandidates(aiID: AIModelID.qcew.rawValue, currentMethod: .zScore, maxCandidates: 3),
            [.zScore, .existing, .volatilityStdReturns]
        )
        XCTAssertEqual(
            NormalizationMetaSupportTools.normalizationMethodCandidates(aiID: -1, currentMethod: .revin, maxCandidates: 0),
            []
        )
    }

    func testWindowConfigResetAndSetMirrorLegacyVersionBumps() {
        var config = NormalizationWindowConfigState()
        config.reset(defaultWindow: 999)

        XCTAssertTrue(config.initialized)
        XCTAssertEqual(config.defaultWindow, 512)
        XCTAssertEqual(config.configVersion, 1)
        XCTAssertTrue(config.featureWindows.allSatisfy { $0 == 512 })

        config.set(featureWindows: [8, 64, 999], defaultWindow: 32)
        XCTAssertEqual(config.configVersion, 2)
        XCTAssertEqual(config.defaultWindow, 32)
        XCTAssertEqual(config.featureWindows[0], 16)
        XCTAssertEqual(config.featureWindows[1], 64)
        XCTAssertEqual(config.featureWindows[2], 512)
        XCTAssertEqual(config.featureWindows[3], 32)

        var fresh = NormalizationWindowConfigState()
        fresh.set(featureWindows: [64], defaultWindow: 128)
        XCTAssertTrue(fresh.initialized)
        XCTAssertEqual(fresh.configVersion, 2)
        XCTAssertEqual(fresh.featureWindows[0], 64)
        XCTAssertEqual(fresh.featureWindows[1], 128)
    }

    func testRuntimeWindowStateAppliesConfigAndLegacyMirrors() {
        var state = NormalizationWindowRuntimeState()
        state.applyGroupWindows(fast: 32, mid: 96, slow: 256, regime: 384, defaultWindow: 96)

        XCTAssertTrue(state.legacy.ready)
        XCTAssertTrue(state.config.initialized)
        XCTAssertEqual(state.legacy.defaultWindow, 96)
        XCTAssertEqual(state.config.defaultWindow, 96)
        XCTAssertEqual(state.legacy.featureWindows[0], 32)
        XCTAssertEqual(state.config.featureWindows[0], 32)
        XCTAssertEqual(state.legacy.featureWindows[22], 256)
        XCTAssertEqual(state.config.configVersion, 2)
    }

    func testNormalizationWindowCodecsRoundTripLegacySections() throws {
        var runtime = NormalizationWindowRuntimeState()
        runtime.apply(featureWindows: [8, 64, 256], defaultWindow: 96)

        let legacyEncoded = try RuntimeNormalizationWindowsCodec.encode(runtime.legacy)
        let configEncoded = try RuntimeNormalizationWindowConfigCodec.encode(runtime.config)
        let legacyDecoded = try RuntimeNormalizationWindowsCodec.decode(from: legacyEncoded)
        let configDecoded = try RuntimeNormalizationWindowConfigCodec.decode(from: configEncoded)

        XCTAssertEqual(legacyEncoded.count, RuntimeNormalizationWindowsCodec.byteCount)
        XCTAssertEqual(configEncoded.count, RuntimeNormalizationWindowConfigCodec.byteCount)
        XCTAssertTrue(legacyDecoded.ready)
        XCTAssertEqual(legacyDecoded.defaultWindow, 96)
        XCTAssertEqual(legacyDecoded.featureWindows[0], 16)
        XCTAssertEqual(legacyDecoded.featureWindows[1], 64)
        XCTAssertEqual(legacyDecoded.featureWindows[2], 256)
        XCTAssertTrue(configDecoded.initialized)
        XCTAssertEqual(configDecoded.configVersion, 2)
        XCTAssertEqual(configDecoded.featureWindows[3], 96)
    }

    func testNormalizationMethodStatePoliciesMatchLegacyGroups() {
        XCTAssertTrue(FeatureNormalizationMethod.zScore.usesRollingNormalizationHistory)
        XCTAssertTrue(FeatureNormalizationMethod.quantileToNormal.usesRollingNormalizationHistory)
        XCTAssertFalse(FeatureNormalizationMethod.powerYeoJohnson.usesRollingNormalizationHistory)
        XCTAssertTrue(FeatureNormalizationMethod.powerYeoJohnson.usesFittedStats)
        XCTAssertTrue(FeatureNormalizationMethod.revin.usesAdaptivePayloadNormalization)
        XCTAssertFalse(FeatureNormalizationMethod.existing.usesFittedStats)
    }

    func testNormalizationHistoryRecordsNewestFirstAndResetsOnRewindOrConfigChange() {
        var history = NormalizationHistoryState()
        XCTAssertTrue(history.prepareForSample(method: .zScore, sampleTimeUTC: 100, configVersion: 1))
        history.record(method: .zScore, featureIndex: 4, value: 1.0)
        history.record(method: .zScore, featureIndex: 4, value: 2.0)
        history.record(method: .zScore, featureIndex: 4, value: 3.0)

        XCTAssertEqual(history.historyCount(method: .zScore, featureIndex: 4, window: 2), 3)
        XCTAssertEqual(history.recentValues(method: .zScore, featureIndex: 4, window: 2), [3.0, 2.0, 1.0])

        XCTAssertTrue(history.prepareForSample(method: .zScore, sampleTimeUTC: 90, configVersion: 1))
        XCTAssertEqual(history.historyCount(method: .zScore, featureIndex: 4, window: 64), 0)
        history.record(method: .zScore, featureIndex: 4, value: 4.0)
        XCTAssertEqual(history.recentValues(method: .zScore, featureIndex: 4, window: 64), [4.0])

        XCTAssertTrue(history.prepareForSample(method: .zScore, sampleTimeUTC: 110, configVersion: 2))
        XCTAssertEqual(history.historyCount(method: .zScore, featureIndex: 4, window: 64), 0)

        XCTAssertFalse(history.prepareForSample(method: .existing, sampleTimeUTC: 120, configVersion: 3))
        history.record(method: .existing, featureIndex: 4, value: 5.0)
        XCTAssertEqual(history.historyCount(method: .existing, featureIndex: 4, window: 64), 0)
    }

    func testNormalizationHistoryCodecRoundTripsLegacySection() throws {
        var history = NormalizationHistoryState()
        _ = history.prepareForSample(method: .robustMedianIQR, sampleTimeUTC: 1_704_067_200, configVersion: 7)
        history.record(method: .robustMedianIQR, featureIndex: 8, value: -1.5)
        history.record(method: .robustMedianIQR, featureIndex: 8, value: 2.5)

        let encoded = try RuntimeNormalizationHistoryCodec.encode(history)
        let decoded = try RuntimeNormalizationHistoryCodec.decode(from: encoded)

        XCTAssertEqual(encoded.count, RuntimeNormalizationHistoryCodec.byteCount)
        XCTAssertTrue(decoded.initialized)
        XCTAssertEqual(decoded.methods[FeatureNormalizationMethod.robustMedianIQR.rawValue].lastSampleTimeUTC, 1_704_067_200)
        XCTAssertEqual(decoded.methods[FeatureNormalizationMethod.robustMedianIQR.rawValue].lastConfigVersion, 7)
        XCTAssertEqual(decoded.recentValues(method: .robustMedianIQR, featureIndex: 8, window: 4), [2.5, -1.5])
    }

    func testNormalizationFallbackAndFitStatsMatchLegacyMath() {
        let fallback = NormalizationFitTools.fallbackStats(featureIndex: 5)
        XCTAssertEqual(fallback.minimum, 0.0, accuracy: 1e-12)
        XCTAssertEqual(fallback.maximum, 10.0, accuracy: 1e-12)
        XCTAssertEqual(fallback.mean, 5.0, accuracy: 1e-12)
        XCTAssertEqual(fallback.standardDeviation, 2.5, accuracy: 1e-12)
        XCTAssertEqual(fallback.interquartileRange, 5.0, accuracy: 1e-12)

        let stats = NormalizationFitTools.fitFeatureStats(values: [0, 1, 2, 3, 4, 5, 6, 7], featureIndex: 0)
        XCTAssertEqual(stats.minimum, 0.0, accuracy: 1e-12)
        XCTAssertEqual(stats.maximum, 7.0, accuracy: 1e-12)
        XCTAssertEqual(stats.mean, 3.5, accuracy: 1e-12)
        XCTAssertEqual(stats.standardDeviation, sqrt(5.25), accuracy: 1e-12)
        XCTAssertEqual(stats.median, 3.5, accuracy: 1e-12)
        XCTAssertEqual(stats.interquartileRange, 3.5, accuracy: 1e-12)
        XCTAssertEqual(stats.quantiles[4], 1.75, accuracy: 1e-12)
        XCTAssertEqual(stats.quantiles[12], 5.25, accuracy: 1e-12)
    }

    func testNormalizationFitStateFitsRowsAndSkipsNonFittedMethods() {
        var fit = NormalizationFitState()
        let rows = (0..<8).map { value -> [Double] in
            var row = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
            row[0] = Double(value)
            row[5] = Double(value) * 2.0
            return row
        }

        XCTAssertTrue(fit.fit(method: .zScore, horizonMinutes: 5, rawRows: rows))
        let feature0 = fit.featureStats(method: .zScore, horizonMinutes: 5, featureIndex: 0)
        XCTAssertTrue(feature0.ready)
        XCTAssertEqual(feature0.stats.mean, 3.5, accuracy: 1e-12)
        XCTAssertEqual(feature0.stats.standardDeviation, sqrt(5.25), accuracy: 1e-12)
        XCTAssertEqual(feature0.stats.quantiles[16], 7.0, accuracy: 1e-12)

        XCTAssertFalse(fit.fit(method: .zScore, horizonMinutes: 5, rawRows: Array(rows.prefix(7))))
        XCTAssertFalse(fit.featureStats(method: .zScore, horizonMinutes: 5, featureIndex: 0).ready)

        XCTAssertTrue(fit.fit(method: .existing, horizonMinutes: 5, rawRows: rows))
        let existing = fit.featureStats(method: .existing, horizonMinutes: 5, featureIndex: 0)
        XCTAssertFalse(existing.ready)
        XCTAssertEqual(existing.stats.mean, 0.0, accuracy: 1e-12)
    }

    func testNormalizationCoreAppliesFittedZScoreStatsToModelInput() throws {
        var fit = NormalizationFitState()
        let rows = (0..<8).map { value -> [Double] in
            var row = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
            row[0] = Double(value)
            return row
        }
        XCTAssertTrue(fit.fit(method: .zScore, horizonMinutes: 5, rawRows: rows))

        var raw = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        raw[0] = 7.0
        let frame = FeatureCoreFrame(
            valid: true,
            sampleIndex: 7,
            horizonMinutes: 5,
            normalizationMethod: .zScore,
            sampleTimeUTC: 1_704_067_200,
            hasVolume: false,
            hasPrevious: true,
            raw: raw,
            previous: Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        )

        let normalized = try NormalizationCore().buildInputFrame(from: frame, fitState: fit)
        let expected = (7.0 - 3.5) / sqrt(5.25) / 4.0

        XCTAssertEqual(normalized.normalized[0], expected, accuracy: 1e-12)
        XCTAssertEqual(normalized.modelInput[0], 1.0, accuracy: 1e-12)
        XCTAssertEqual(normalized.modelInput[1], expected, accuracy: 1e-12)
    }

    func testNormalizationCoreKeepsConstantFittedMinMaxFeaturesNeutral() throws {
        var fit = NormalizationFitState()
        let rows = (0..<8).map { _ -> [Double] in
            var row = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
            row[0] = 0.25
            return row
        }
        XCTAssertTrue(fit.fit(method: .minMaxBuffer5, horizonMinutes: 5, rawRows: rows))

        var raw = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        raw[0] = 0.25
        let frame = FeatureCoreFrame(
            valid: true,
            sampleIndex: 7,
            horizonMinutes: 5,
            normalizationMethod: .minMaxBuffer5,
            sampleTimeUTC: 1_704_067_200,
            hasVolume: false,
            hasPrevious: true,
            raw: raw,
            previous: Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        )

        let normalized = try NormalizationCore().buildInputFrame(from: frame, fitState: fit)

        XCTAssertEqual(normalized.normalized[0], 0.5, accuracy: 1e-12)
        XCTAssertEqual(normalized.modelInput[1], 0.5, accuracy: 1e-12)
    }

    func testNormalizationFitCodecRoundTripsLegacySection() throws {
        var fit = NormalizationFitState()
        let rows = (0..<8).map { value -> [Double] in
            var row = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
            row[0] = Double(value)
            return row
        }
        XCTAssertTrue(fit.fit(method: .robustMedianIQR, horizonMinutes: 13, rawRows: rows))

        let encoded = try RuntimeNormalizationFitCodec.encode(fit)
        let decoded = try RuntimeNormalizationFitCodec.decode(from: encoded)
        let lookup = decoded.featureStats(method: .robustMedianIQR, horizonMinutes: 13, featureIndex: 0)

        XCTAssertEqual(encoded.count, RuntimeNormalizationFitCodec.byteCount)
        XCTAssertTrue(decoded.initialized)
        XCTAssertTrue(lookup.ready)
        XCTAssertEqual(lookup.stats.median, 3.5, accuracy: 1e-12)
        XCTAssertEqual(lookup.stats.interquartileRange, 3.5, accuracy: 1e-12)
    }
}
