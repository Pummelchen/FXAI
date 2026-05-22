import XCTest
@testable import FXDataEngine

final class PluginSharedTransferPayloadTests: XCTestCase {
    func testBaseInputBuildsContextMTFAndMacroSignals() {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        set(&values, 6, 0.4)
        set(&values, 18, 0.2)
        set(&values, 19, 0.1)
        set(&values, 20, -0.2)
        set(&values, 21, 0.3)
        set(&values, 62, 0.25)
        set(&values, 63, -0.30)
        set(&values, 64, 0.45)
        set(&values, 65, 0.40)
        set(&values, 66, 0.20)
        set(&values, 67, -0.10)
        set(&values, 68, 0.40)
        set(&values, 69, 0.30)
        set(&values, 70, 0.50)
        set(&values, 71, 0.25)
        set(&values, 72, 0.50)
        set(&values, 73, 0.20)
        set(&values, 74, 0.40)
        set(&values, 75, 1.00)
        set(&values, 76, 0.20)
        set(&values, 77, -0.10)
        set(&values, 78, 0.30)
        set(&values, 79, 0.60)
        set(&values, 80, 0.25)
        set(&values, 81, 0.50)
        set(&values, 82, 0.70)
        set(&values, 83, 0.35)

        let contexts = [
            (0.8, -0.3, 0.2, 0.5),
            (-0.4, 0.6, -0.1, -0.8),
            (0.1, 0.2, 0.3, 0.0)
        ]
        for (slot, context) in contexts.enumerated() {
            let base = 50 + slot * 4
            set(&values, base, context.0)
            set(&values, base + 1, context.1)
            set(&values, base + 2, context.2)
            set(&values, base + 3, context.3)
        }

        for slot in 0..<FXDataEngineConstants.mainMTFTimeframeCount {
            let base = FXDataEngineConstants.mainMTFFeatureOffset +
                slot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            set(&values, base, [0.1, 0.2, -0.1, 0.0][slot])
            set(&values, base + 1, [0.3, 0.4, 0.2, 0.1][slot])
            set(&values, base + 2, [0.5, 0.6, 0.4, 0.7][slot])
            set(&values, base + 3, [0.05, 0.10, -0.05, 0.20][slot])
        }

        for slot in 0..<FXDataEngineConstants.contextTopSymbols {
            for timeframeSlot in 0..<FXDataEngineConstants.contextMTFTimeframeCount {
                let base = FXDataEngineConstants.contextMTFFeatureOffset +
                    slot * FXDataEngineConstants.contextSlotMTFFeatures +
                    timeframeSlot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe
                let slotScale = Double(slot + 1)
                let timeframeScale = Double(timeframeSlot)
                set(&values, base, 0.05 * slotScale + 0.01 * timeframeScale)
                set(&values, base + 1, 0.10 * slotScale + 0.02 * timeframeScale)
                set(&values, base + 2, 0.15 * slotScale + 0.01 * timeframeScale)
                set(&values, base + 3, 0.04 * slotScale + 0.03 * timeframeScale)
            }
        }

        setMacro(&values, 0, 0.20)
        setMacro(&values, 1, -0.10)
        setMacro(&values, 2, 0.50)
        setMacro(&values, 3, 0.40)
        setMacro(&values, 4, 0.30)
        setMacro(&values, 5, -0.20)
        setMacro(&values, 6, 0.60)
        setMacro(&values, 7, 0.10)
        setMacro(&values, 8, 0.40)
        setMacro(&values, 9, 0.70)
        setMacro(&values, 14, 0.20)
        setMacro(&values, 15, 0.30)
        setMacro(&values, 16, 0.40)
        setMacro(&values, 18, 0.50)
        setMacro(&values, 19, 0.60)

        let x = RuntimeTransferTools.modelInputVector(features: values)
        let payload = PluginSharedTransferPayloadTools.buildInput(
            x: x,
            domainHash: 0.75,
            horizonMinutes: 60
        )

        XCTAssertEqual(payload.count, FXDataEngineConstants.sharedTransferFeatures)
        XCTAssertEqual(payload[0], 1.0, accuracy: 0.0)
        XCTAssertEqual(payload[1], 0.25, accuracy: 1e-12)
        XCTAssertEqual(payload[2], -0.30, accuracy: 1e-12)
        XCTAssertEqual(payload[3], 0.45, accuracy: 1e-12)
        XCTAssertEqual(payload[4], 0.70, accuracy: 1e-12)

        let blend = expectedContextBlend(contexts, coverage: payload[4])
        XCTAssertEqual(payload[5], blend.ret, accuracy: 1e-12)
        XCTAssertEqual(payload[6], blend.lag, accuracy: 1e-12)
        XCTAssertEqual(payload[7], blend.relative, accuracy: 1e-12)
        XCTAssertEqual(payload[8], blend.correlation, accuracy: 1e-12)
        XCTAssertEqual(payload[9], 0.50, accuracy: 1e-12)
        XCTAssertEqual(payload[10], 2.0 * log(61.0) / log(1441.0) - 1.0, accuracy: 1e-12)

        let main = expectedMainMTF(values)
        let contextMTF = expectedContextMTF(values)
        XCTAssertEqual(
            payload[11],
            fxClamp(0.60 * value(values, 72) + 0.15 * value(values, 73) + 0.10 * macro(values, 0) -
                0.08 * macro(values, 1) + 0.07 * main.location + 0.06 * contextMTF.location, -1.0, 1.0),
            accuracy: 1e-12
        )
        XCTAssertEqual(
            payload[13],
            fxClamp(0.16 * value(values, 76) - 0.16 * value(values, 77) - 0.10 * value(values, 79) +
                0.08 * value(values, 6) + 0.08 * value(values, 81) + 0.06 * macro(values, 4) -
                0.05 * main.volumePressure - 0.05 * value(values, 82) + 0.04 * contextMTF.volumePressure, -4.0, 4.0),
            accuracy: 1e-12
        )
        XCTAssertEqual(
            payload[16],
            fxClamp(0.48 * value(values, 68) + 0.18 * value(values, 81) + 0.12 * value(values, 80) +
                0.10 * macro(values, 2) + 0.08 * macro(values, 4) + 0.08 * main.volumePressure, -4.0, 8.0),
            accuracy: 1e-12
        )
        XCTAssertEqual(
            payload[19],
            fxClamp(0.40 * macro(values, 4) + 0.22 * macro(values, 2) + 0.18 * macro(values, 0) +
                0.10 * macro(values, 1) + 0.10 * value(values, 79) + 0.10 * macro(values, 7) +
                0.08 * macro(values, 9) + 0.08 * macro(values, 15) + 0.06 * macro(values, 16) +
                0.06 * macro(values, 18) + 0.08 * macro(values, 19), 0.0, 6.0),
            accuracy: 1e-12
        )
    }

    func testEmptyWindowUsesCurrentFeatureFallbacks() {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        set(&values, 0, 0.10)
        set(&values, 3, -0.20)
        set(&values, 10, 0.30)
        set(&values, 41, 0.60)
        set(&values, 62, 0.20)
        set(&values, 63, -0.40)
        set(&values, 64, 0.50)
        set(&values, 68, 0.40)
        set(&values, 69, 0.20)
        set(&values, 72, 0.10)
        set(&values, 73, 0.20)
        set(&values, 74, 0.30)
        set(&values, 75, 1.00)
        set(&values, 76, 0.60)
        set(&values, 77, 0.20)
        set(&values, 78, 0.70)
        set(&values, 79, 0.80)
        set(&values, 80, 0.50)
        set(&values, 82, 0.90)
        setMacro(&values, 0, 0.50)
        setMacro(&values, 1, 0.20)
        setMacro(&values, 2, 0.30)
        setMacro(&values, 4, 0.60)
        setMacro(&values, 5, -0.20)

        let payload = PluginSharedTransferPayloadTools.buildInput(
            x: RuntimeTransferTools.modelInputVector(features: values),
            window: [],
            domainHash: 0.10,
            horizonMinutes: 5
        )

        XCTAssertEqual(payload[20], 0.10, accuracy: 1e-12)
        XCTAssertEqual(payload[21], -0.20, accuracy: 1e-12)
        XCTAssertEqual(payload[22], 0.60, accuracy: 1e-12)
        XCTAssertEqual(payload[23], 0.50 * 0.40 + 0.25 * 0.20 + 0.25 * 0.50, accuracy: 1e-12)
        XCTAssertEqual(payload[24], 0.35 * 0.30 + 0.25 * 0.20 + 0.20 * -0.40 + 0.20 * 0.50, accuracy: 1e-12)
        XCTAssertEqual(payload[25], 0.25 * 0.50 + 0.15 * 0.20 + 0.20 * 0.30 + 0.20 * 0.60 + 0.20 * -0.20, accuracy: 1e-12)
        XCTAssertEqual(payload[26], 0.20 * 0.10 + 0.15 * 0.20 + 0.20 * 0.30 + 0.10 * 1.00 +
            0.15 * 0.70 + 0.10 * (0.60 - 0.20) + 0.10 * 0.80, accuracy: 1e-12)
        XCTAssertEqual(payload[27], 0.30 * 0.80 + 0.25 * 0.40 + 0.20 * 0.40 + 0.15 * 0.90 +
            0.10 * 0.60, accuracy: 1e-12)
    }

    func testWindowInputUsesArithmeticTransferStatistics() {
        let window = [
            row([(0, 1.0), (1, 0.5), (2, 0.25), (3, 0.8), (5, 2.0), (6, 0.4), (10, 0.3),
                 (41, 1.0), (43, 0.4), (62, 0.5), (63, 0.1), (64, 0.2), (65, 0.3),
                 (68, 0.4), (69, 0.2), (71, 0.7), (72, 0.1), (73, 0.2), (74, 0.3),
                 (75, 0.4), (76, 0.5), (77, 0.1), (78, 0.6), (79, 0.2), (80, 0.4),
                 (81, 0.5), (82, 0.6), (83, 0.7), (FXDataEngineConstants.macroEventFeatureOffset + 0, 0.2),
                 (FXDataEngineConstants.macroEventFeatureOffset + 1, 0.1), (FXDataEngineConstants.macroEventFeatureOffset + 2, 0.3),
                 (FXDataEngineConstants.macroEventFeatureOffset + 3, 0.4), (FXDataEngineConstants.macroEventFeatureOffset + 4, 0.5),
                 (FXDataEngineConstants.macroEventFeatureOffset + 5, 0.2), (FXDataEngineConstants.macroEventFeatureOffset + 14, 0.3),
                 (FXDataEngineConstants.macroEventFeatureOffset + 15, 0.4), (FXDataEngineConstants.macroEventFeatureOffset + 16, 0.5),
                 (FXDataEngineConstants.macroEventFeatureOffset + 18, 0.6), (FXDataEngineConstants.macroEventFeatureOffset + 19, 0.7)]),
            row([(0, 0.6), (1, 0.3), (2, 0.15), (3, 0.5), (5, 1.5), (6, 0.2), (10, 0.1),
                 (41, 0.8), (43, 0.3), (62, 0.4), (63, 0.2), (64, 0.1), (65, 0.2),
                 (68, 0.3), (69, 0.1), (71, 0.5), (72, 0.2), (73, 0.1), (74, 0.2),
                 (75, 0.3), (76, 0.4), (77, 0.2), (78, 0.5), (79, 0.3), (80, 0.2),
                 (81, 0.4), (82, 0.5), (83, 0.6), (FXDataEngineConstants.macroEventFeatureOffset + 0, 0.1),
                 (FXDataEngineConstants.macroEventFeatureOffset + 1, 0.2), (FXDataEngineConstants.macroEventFeatureOffset + 2, 0.2),
                 (FXDataEngineConstants.macroEventFeatureOffset + 3, 0.3), (FXDataEngineConstants.macroEventFeatureOffset + 4, 0.4),
                 (FXDataEngineConstants.macroEventFeatureOffset + 5, 0.1), (FXDataEngineConstants.macroEventFeatureOffset + 14, 0.2),
                 (FXDataEngineConstants.macroEventFeatureOffset + 15, 0.3), (FXDataEngineConstants.macroEventFeatureOffset + 16, 0.4),
                 (FXDataEngineConstants.macroEventFeatureOffset + 18, 0.5), (FXDataEngineConstants.macroEventFeatureOffset + 19, 0.6)]),
            row([(0, 0.2), (1, 0.1), (2, 0.05), (3, 0.1), (5, 1.0), (6, -0.1), (10, -0.2),
                 (41, 0.6), (43, 0.2), (62, 0.2), (63, -0.1), (64, 0.0), (65, 0.1),
                 (68, 0.1), (69, -0.1), (71, 0.2), (72, 0.3), (73, -0.1), (74, 0.1),
                 (75, 0.2), (76, 0.3), (77, 0.1), (78, 0.4), (79, 0.4), (80, -0.1),
                 (81, 0.2), (82, 0.4), (83, 0.5), (FXDataEngineConstants.macroEventFeatureOffset + 0, 0.0),
                 (FXDataEngineConstants.macroEventFeatureOffset + 1, 0.3), (FXDataEngineConstants.macroEventFeatureOffset + 2, 0.1),
                 (FXDataEngineConstants.macroEventFeatureOffset + 3, 0.2), (FXDataEngineConstants.macroEventFeatureOffset + 4, 0.3),
                 (FXDataEngineConstants.macroEventFeatureOffset + 5, 0.0), (FXDataEngineConstants.macroEventFeatureOffset + 14, 0.1),
                 (FXDataEngineConstants.macroEventFeatureOffset + 15, 0.2), (FXDataEngineConstants.macroEventFeatureOffset + 16, 0.3),
                 (FXDataEngineConstants.macroEventFeatureOffset + 18, 0.4), (FXDataEngineConstants.macroEventFeatureOffset + 19, 0.5)])
        ]

        let payload = PluginSharedTransferPayloadTools.buildInput(
            x: window[0],
            window: window,
            domainHash: 0.30,
            horizonMinutes: 12
        )

        let retFast = ema(window, 0)
        let retMid = ema(window, 1)
        let retLong = ema(window, 2)
        XCTAssertEqual(PluginSharedTransferPayloadTools.windowFeatureMean(window, featureIndex: 5), mean(window, 5), accuracy: 1e-12)
        XCTAssertEqual(PluginSharedTransferPayloadTools.windowFeatureStd(window, featureIndex: 5), std(window, 5), accuracy: 1e-12)
        XCTAssertEqual(PluginSharedTransferPayloadTools.windowFeatureSlope(window, featureIndex: 3), slope(window, 3), accuracy: 1e-12)
        XCTAssertEqual(PluginSharedTransferPayloadTools.windowFeatureRecentDelta(window, featureIndex: 0, recentBars: 3), 0.8, accuracy: 1e-12)
        XCTAssertEqual(payload[20], fxClamp(0.42 * retFast + 0.33 * retMid + 0.15 * retLong + 0.10 * 0.8, -4.0, 4.0), accuracy: 1e-12)
        XCTAssertEqual(payload[22], fxClamp(0.34 * mean(window, 5) + 0.22 * std(window, 5) +
            0.22 * mean(window, 41) + 0.22 * mean(window, 43), 0.0, 6.0), accuracy: 1e-12)
        XCTAssertEqual(payload[27], fxClamp(0.24 * mean(window, 79) + 0.18 * std(window, 63) +
            0.18 * range(window, 10) + 0.14 * abs(delta(window, 80, 3)) + 0.14 * mean(window, 82) +
            0.10 * abs(mean(window, 83)) + 0.12 * mean(window, FXDataEngineConstants.macroEventFeatureOffset + 16) +
            0.10 * mean(window, FXDataEngineConstants.macroEventFeatureOffset + 19), 0.0, 6.0), accuracy: 1e-12)
    }

    func testTemporalBucketsAndBarFeatureExtraction() {
        XCTAssertEqual(PluginSharedTransferPayloadTools.domainBucket(domainHash: -0.5), 0)
        XCTAssertEqual(PluginSharedTransferPayloadTools.domainBucket(domainHash: 1.0), 7)
        XCTAssertEqual(PluginSharedTransferPayloadTools.horizonBucket(horizonMinutes: 0), 0)
        XCTAssertEqual(PluginSharedTransferPayloadTools.horizonBucket(horizonMinutes: 3), 1)
        XCTAssertEqual(PluginSharedTransferPayloadTools.horizonBucket(horizonMinutes: 31), 4)
        XCTAssertEqual(PluginSharedTransferPayloadTools.horizonBucket(horizonMinutes: 721), 7)

        let window = [
            row([
                (0, 5.0), (3, -5.0), (5, 7.0), (10, -5.0), (41, 7.0), (62, -5.0),
                (80, 9.0), (82, 9.0), (72, 5.0), (78, -5.0),
                (FXDataEngineConstants.macroEventFeatureOffset + 2, 0.8),
                (FXDataEngineConstants.macroEventFeatureOffset + 3, 2.0),
                (FXDataEngineConstants.macroEventFeatureOffset + 14, -0.5),
                (FXDataEngineConstants.macroEventFeatureOffset + 15, 1.0),
                (FXDataEngineConstants.macroEventFeatureOffset + 19, 0.4)
            ])
        ]
        let barFeatures = PluginSharedTransferPayloadTools.barFeatures(window, barIndex: 0)

        XCTAssertEqual(barFeatures.count, FXDataEngineConstants.sharedTransferBarFeatures)
        XCTAssertEqual(barFeatures[0], 4.0, accuracy: 0.0)
        XCTAssertEqual(barFeatures[1], -4.0, accuracy: 0.0)
        XCTAssertEqual(barFeatures[2], 6.0, accuracy: 0.0)
        XCTAssertEqual(barFeatures[3], -4.0, accuracy: 0.0)
        XCTAssertEqual(barFeatures[4], 6.0, accuracy: 0.0)
        XCTAssertEqual(barFeatures[5], -4.0, accuracy: 0.0)
        XCTAssertEqual(barFeatures[6], 8.0, accuracy: 0.0)
        XCTAssertEqual(barFeatures[7], 8.0, accuracy: 0.0)
        XCTAssertEqual(barFeatures[8], 4.0, accuracy: 0.0)
        XCTAssertEqual(barFeatures[9], -4.0, accuracy: 0.0)
        XCTAssertEqual(barFeatures[10], 0.70 * 0.8 + 0.30 * 0.4, accuracy: 1e-12)
        XCTAssertEqual(barFeatures[11], 0.50 * 2.0 + 0.30 * 1.0 + 0.20 * -0.5, accuracy: 1e-12)
        XCTAssertEqual(PluginSharedTransferPayloadTools.barFeatures(window, barIndex: 1), Array(repeating: 0.0, count: 12))
    }

    func testSequenceTokensUsePayloadFallbackWhenWindowIsEmpty() {
        var payload = Array(repeating: 0.0, count: FXDataEngineConstants.sharedTransferFeatures)
        for index in 0..<payload.count {
            payload[index] = Double(index) / 10.0
        }

        let tokens = PluginSharedTransferPayloadTools.sequenceTokens(payload: payload, window: [])

        XCTAssertEqual(tokens.count, FXDataEngineConstants.sharedTransferSequenceTokens)
        XCTAssertEqual(tokens[0], 2.0, accuracy: 1e-12)
        XCTAssertEqual(tokens[7], 2.7, accuracy: 1e-12)
        XCTAssertEqual(tokens[8], 0.50 * 1.1 + 0.50 * 1.2, accuracy: 1e-12)
        XCTAssertEqual(tokens[9], 0.55 * 1.3 + 0.45 * 1.4, accuracy: 1e-12)
        XCTAssertEqual(tokens[10], 0.50 * 1.5 + 0.50 * 1.6, accuracy: 1e-12)
        XCTAssertEqual(tokens[11], 0.50 * 1.7 + 0.50 * 1.8, accuracy: 1e-12)
        XCTAssertEqual(tokens[12], 0.60 * 0.6 + 0.40 * 0.7, accuracy: 1e-12)
        XCTAssertEqual(tokens[13], 0.50 * 0.4 + 0.50 * 0.8, accuracy: 1e-12)
        XCTAssertEqual(tokens[14], 1.9, accuracy: 1e-12)
        XCTAssertEqual(tokens[15], (abs(tokens[0]) + abs(tokens[3]) + abs(tokens[7])) / 3.0, accuracy: 1e-12)
    }

    func testSequenceTokensUseWindowSegmentsAndVolumePressure() {
        let window = [
            temporalRow(
                [(0, 1.0), (1, 0.4), (2, 0.5), (3, 0.4), (5, 2.0), (10, 0.3), (62, 0.5),
                 (64, 0.2), (65, 0.6), (72, 0.1), (73, 0.3), (74, 0.5), (78, 0.4),
                 (79, 0.2), (80, 0.6), (81, 0.2), (FXDataEngineConstants.macroEventFeatureOffset + 0, 0.2),
                 (FXDataEngineConstants.macroEventFeatureOffset + 1, 0.1), (FXDataEngineConstants.macroEventFeatureOffset + 2, 0.4),
                 (FXDataEngineConstants.macroEventFeatureOffset + 3, 0.5), (FXDataEngineConstants.macroEventFeatureOffset + 4, 0.3),
                 (FXDataEngineConstants.macroEventFeatureOffset + 6, 0.2), (FXDataEngineConstants.macroEventFeatureOffset + 7, 0.1),
                 (FXDataEngineConstants.macroEventFeatureOffset + 8, 0.4), (FXDataEngineConstants.macroEventFeatureOffset + 9, 0.5)],
                mainBody: 0.2,
                mainVolumePressure: 0.3,
                contextBody: 0.1,
                contextVolumePressure: 0.05
            ),
            temporalRow(
                [(0, 0.5), (1, 0.3), (2, 0.4), (3, 0.2), (5, 1.0), (10, 0.1), (62, 0.3),
                 (64, 0.4), (65, 0.2), (72, 0.2), (73, 0.1), (74, 0.3), (78, 0.6),
                 (79, 0.4), (80, 0.4), (81, 0.4), (FXDataEngineConstants.macroEventFeatureOffset + 0, 0.1),
                 (FXDataEngineConstants.macroEventFeatureOffset + 1, 0.2), (FXDataEngineConstants.macroEventFeatureOffset + 2, 0.2),
                 (FXDataEngineConstants.macroEventFeatureOffset + 3, 0.3), (FXDataEngineConstants.macroEventFeatureOffset + 4, 0.2),
                 (FXDataEngineConstants.macroEventFeatureOffset + 6, 0.4), (FXDataEngineConstants.macroEventFeatureOffset + 7, 0.3),
                 (FXDataEngineConstants.macroEventFeatureOffset + 8, 0.2), (FXDataEngineConstants.macroEventFeatureOffset + 9, 0.1)],
                mainBody: 0.2,
                mainVolumePressure: 0.3,
                contextBody: 0.1,
                contextVolumePressure: 0.05
            ),
            temporalRow([(1, 0.2), (2, 0.3), (5, 0.5), (10, -0.1), (79, 0.1), (80, 0.1)],
                        mainBody: 0.2,
                        mainVolumePressure: 0.3,
                        contextBody: 0.1,
                        contextVolumePressure: 0.05),
            temporalRow([(1, 0.1), (2, 0.1), (5, 0.25), (10, -0.3), (79, 0.0), (80, 0.0)],
                        mainBody: 0.2,
                        mainVolumePressure: 0.3,
                        contextBody: 0.1,
                        contextVolumePressure: 0.05)
        ]

        let tokens = PluginSharedTransferPayloadTools.sequenceTokens(payload: [], window: window)

        XCTAssertEqual(tokens[0], 0.75, accuracy: 1e-12)
        XCTAssertEqual(tokens[1], 0.15, accuracy: 1e-12)
        XCTAssertEqual(tokens[2], 0.20, accuracy: 1e-12)
        XCTAssertEqual(tokens[3], 0.30 + 0.35 * (0.75 - 0.20), accuracy: 1e-12)
        XCTAssertEqual(tokens[4], 0.65 * 1.5 + 0.35 * (1.5 - 0.375), accuracy: 1e-12)
        XCTAssertEqual(tokens[5], 0.60 * 0.5 + 0.25 * 0.3 + 0.15 * (0.5 - 0.05), accuracy: 1e-12)
        XCTAssertEqual(tokens[7], 0.45 * 0.4 + 0.30 * 0.3 + 0.25 * 0.4, accuracy: 1e-12)
        XCTAssertEqual(tokens[12], 0.25, accuracy: 1e-12)
        XCTAssertEqual(tokens[13], 0.55, accuracy: 1e-12)
        XCTAssertEqual(tokens[14], 0.45, accuracy: 1e-12)
        XCTAssertEqual(tokens[15], fxClamp(0.40 * std(window, 5) + 0.35 * std(window, 80) + 0.25 * std(window, 79), 0.0, 6.0), accuracy: 1e-12)
    }

    private func expectedContextBlend(
        _ contexts: [(Double, Double, Double, Double)],
        coverage: Double
    ) -> (ret: Double, lag: Double, relative: Double, correlation: Double) {
        var ret = 0.0
        var lag = 0.0
        var relative = 0.0
        var correlation = 0.0
        var total = 0.0
        for context in contexts {
            let weight = fxClamp(
                (0.30 + 0.70 * abs(context.3)) *
                    (0.35 + 0.65 * coverage) *
                    (0.35 + 0.25 * abs(context.0) + 0.25 * abs(context.1) + 0.15 * abs(context.2)),
                0.0,
                3.0
            )
            ret += weight * context.0
            lag += weight * context.1
            relative += weight * context.2
            correlation += weight * context.3
            total += weight
        }
        return (ret / total, lag / total, relative / total, correlation / total)
    }

    private func expectedMainMTF(_ values: [Double]) -> (body: Double, location: Double, range: Double, volumePressure: Double) {
        var body = 0.0
        var location = 0.0
        var range = 0.0
        var volumePressure = 0.0
        for slot in 0..<FXDataEngineConstants.mainMTFTimeframeCount {
            let base = FXDataEngineConstants.mainMTFFeatureOffset +
                slot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            body += value(values, base)
            location += value(values, base + 1)
            range += value(values, base + 2)
            volumePressure += value(values, base + 3)
        }
        let count = Double(FXDataEngineConstants.mainMTFTimeframeCount)
        return (body / count, location / count, range / count, volumePressure / count)
    }

    private func expectedContextMTF(_ values: [Double]) -> (body: Double, location: Double, range: Double, volumePressure: Double) {
        var body = 0.0
        var location = 0.0
        var range = 0.0
        var volumePressure = 0.0
        var total = 0.0
        for slot in 0..<FXDataEngineConstants.contextTopSymbols {
            let slotWeight = 0.35 + 0.65 * abs(value(values, 50 + slot * 4 + 3))
            var slotBody = 0.0
            var slotLocation = 0.0
            var slotRange = 0.0
            var slotVolumePressure = 0.0
            for timeframeSlot in 0..<FXDataEngineConstants.contextMTFTimeframeCount {
                let base = FXDataEngineConstants.contextMTFFeatureOffset +
                    slot * FXDataEngineConstants.contextSlotMTFFeatures +
                    timeframeSlot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe
                slotBody += value(values, base)
                slotLocation += value(values, base + 1)
                slotRange += value(values, base + 2)
                slotVolumePressure += value(values, base + 3)
            }
            let count = Double(FXDataEngineConstants.contextMTFTimeframeCount)
            body += slotWeight * slotBody / count
            location += slotWeight * slotLocation / count
            range += slotWeight * slotRange / count
            volumePressure += slotWeight * slotVolumePressure / count
            total += slotWeight
        }
        return (body / total, location / total, range / total, volumePressure / total)
    }

    private func row(_ pairs: [(Int, Double)]) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        for pair in pairs {
            set(&values, pair.0, pair.1)
        }
        return RuntimeTransferTools.modelInputVector(features: values)
    }

    private func temporalRow(
        _ pairs: [(Int, Double)],
        mainBody: Double,
        mainVolumePressure: Double,
        contextBody: Double,
        contextVolumePressure: Double
    ) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        for pair in pairs {
            set(&values, pair.0, pair.1)
        }
        for slot in 0..<FXDataEngineConstants.mainMTFTimeframeCount {
            let base = FXDataEngineConstants.mainMTFFeatureOffset +
                slot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            set(&values, base, mainBody)
            set(&values, base + 3, mainVolumePressure)
        }
        for slot in 0..<FXDataEngineConstants.contextTopSymbols {
            for timeframeSlot in 0..<FXDataEngineConstants.contextMTFTimeframeCount {
                let base = FXDataEngineConstants.contextMTFFeatureOffset +
                    slot * FXDataEngineConstants.contextSlotMTFFeatures +
                    timeframeSlot * FXDataEngineConstants.mtfStateFeaturesPerTimeframe
                set(&values, base, contextBody)
                set(&values, base + 3, contextVolumePressure)
            }
        }
        return RuntimeTransferTools.modelInputVector(features: values)
    }

    private func mean(_ window: [[Double]], _ featureIndex: Int) -> Double {
        window.reduce(0.0) { $0 + $1[featureIndex + 1] } / Double(window.count)
    }

    private func ema(_ window: [[Double]], _ featureIndex: Int, decay: Double = 0.72) -> Double {
        var weight = 1.0
        var total = 0.0
        var sum = 0.0
        for item in window {
            sum += weight * item[featureIndex + 1]
            total += weight
            weight *= decay
        }
        return sum / total
    }

    private func std(_ window: [[Double]], _ featureIndex: Int) -> Double {
        let average = mean(window, featureIndex)
        let sum = window.reduce(0.0) { partial, item in
            let diff = item[featureIndex + 1] - average
            return partial + diff * diff
        }
        return sqrt(sum / Double(window.count))
    }

    private func slope(_ window: [[Double]], _ featureIndex: Int) -> Double {
        (window[0][featureIndex + 1] - window[window.count - 1][featureIndex + 1]) / Double(window.count - 1)
    }

    private func range(_ window: [[Double]], _ featureIndex: Int) -> Double {
        let values = window.map { $0[featureIndex + 1] }
        return values.max()! - values.min()!
    }

    private func delta(_ window: [[Double]], _ featureIndex: Int, _ recentBars: Int) -> Double {
        window[0][featureIndex + 1] - window[min(recentBars, window.count) - 1][featureIndex + 1]
    }

    private func set(_ values: inout [Double], _ featureIndex: Int, _ featureValue: Double) {
        values[featureIndex] = featureValue
    }

    private func setMacro(_ values: inout [Double], _ relativeIndex: Int, _ featureValue: Double) {
        values[FXDataEngineConstants.macroEventFeatureOffset + relativeIndex] = featureValue
    }

    private func value(_ values: [Double], _ featureIndex: Int) -> Double {
        values[featureIndex]
    }

    private func macro(_ values: [Double], _ relativeIndex: Int) -> Double {
        value(values, FXDataEngineConstants.macroEventFeatureOffset + relativeIndex)
    }
}
