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
}
