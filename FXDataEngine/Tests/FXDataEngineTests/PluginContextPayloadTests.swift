import XCTest
@testable import FXDataEngine

final class PluginContextPayloadTests: XCTestCase {
    func testSanitizedContextResolvesInvalidBucketsAndFallbacks() throws {
        var context = PluginContextV4(
            regimeID: 99,
            sessionBucket: -4,
            horizonMinutes: 0,
            sequenceBars: 999,
            priceCostPoints: .nan,
            minMovePoints: -3.0,
            pointValue: -1.0,
            domainHash: 0.25,
            sampleTimeUTC: 0,
            dataHasVolume: true
        )
        context.domainHash = .nan

        let fallbackTimeUTC: Int64 = 1_717_243_200
        let state = PluginContextPayloadState(
            context: context,
            pointValueFallback: 0.0001,
            sampleTimeFallbackUTC: fallbackTimeUTC,
            symbolFallback: "EURUSD"
        )

        XCTAssertEqual(state.context.regimeID, 0)
        XCTAssertEqual(state.context.sessionBucket, 3)
        XCTAssertEqual(state.context.horizonMinutes, 1)
        XCTAssertEqual(state.context.sequenceBars, FXDataEngineConstants.maxSequenceBars)
        XCTAssertEqual(state.context.priceCostPoints, 0.0, accuracy: 0.0)
        XCTAssertEqual(state.context.minMovePoints, 0.0, accuracy: 0.0)
        XCTAssertEqual(state.context.pointValue, 0.0001, accuracy: 0.0)
        XCTAssertEqual(state.context.sampleTimeUTC, fallbackTimeUTC)
        XCTAssertEqual(state.context.domainHash, PluginContractTools.symbolHash01("EURUSD"), accuracy: 0.0)
        XCTAssertTrue(state.context.dataHasVolume)
        try state.context.validate()

        var mutableState = PluginContextPayloadState()
        mutableState.setContext(
            PluginContextV4(sessionBucket: -1, sampleTimeUTC: 0),
            sampleTimeFallbackUTC: fallbackTimeUTC
        )
        XCTAssertEqual(mutableState.context.sessionBucket, 3)
        XCTAssertEqual(mutableState.resolveContextTimeUTC(), fallbackTimeUTC)
    }

    func testWindowPayloadSanitizesRowsAndPadsDeclaredSize() {
        let shortRow = [1.0, Double.nan]
        var longRow = Array(repeating: 2.0, count: FXDataEngineConstants.aiWeights + 3)
        longRow[10] = .infinity

        let payload = PluginContextPayloadTools.sanitizedWindow(
            [shortRow, longRow],
            declaredWindowSize: 4
        )

        XCTAssertEqual(payload.windowSize, 4)
        XCTAssertEqual(payload.window.count, 4)
        XCTAssertEqual(payload.window[0].count, FXDataEngineConstants.aiWeights)
        XCTAssertEqual(payload.window[0][0], 1.0, accuracy: 0.0)
        XCTAssertEqual(payload.window[0][1], 0.0, accuracy: 0.0)
        XCTAssertEqual(payload.window[0][2], 0.0, accuracy: 0.0)
        XCTAssertEqual(payload.window[1][0], 2.0, accuracy: 0.0)
        XCTAssertEqual(payload.window[1][10], 0.0, accuracy: 0.0)
        XCTAssertEqual(payload.window[1][FXDataEngineConstants.aiWeights - 1], 2.0, accuracy: 0.0)
        XCTAssertTrue(payload.window[2].allSatisfy { $0 == 0.0 })
        XCTAssertTrue(payload.window[3].allSatisfy { $0 == 0.0 })
    }

    func testResolveHelpersPreferContextOrDryRunInput() {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[7] = -3.5
        let state = PluginContextPayloadState(
            context: PluginContextV4(
                horizonMinutes: 5,
                sequenceBars: 1,
                priceCostPoints: 2.0,
                minMovePoints: 1.25,
                pointValue: 0.01,
                sampleTimeUTC: 123
            )
        )

        let dryRunState = PluginContextPayloadState()
        XCTAssertEqual(dryRunState.resolvePriceCostPoints(x: x), 3.5, accuracy: 0.0)
        XCTAssertEqual(dryRunState.resolveContextTimeUTC(fallback: 999), 999)

        XCTAssertTrue(state.contextPriceCostReady)
        XCTAssertTrue(state.contextTimeReady)
        XCTAssertEqual(state.resolvePriceCostPoints(x: x), 2.0, accuracy: 0.0)
        XCTAssertEqual(state.resolvePriceCostPoints(x: x, preferContext: false), 3.5, accuracy: 0.0)
        XCTAssertEqual(state.resolveMinMovePoints(), 1.25, accuracy: 0.0)
        XCTAssertEqual(state.resolvePointValue(fallback: 0.0001), 0.01, accuracy: 0.0)
        XCTAssertEqual(state.resolveContextTimeUTC(fallback: 999), 123)
    }

    func testTextEventContextCarriesTokenizerContractAndFallbacks() throws {
        let event = PluginTextEventV4(
            eventTimeUTC: 1_800_000_000,
            source: "calendar",
            headline: "USD growth beat supports risk-on flows",
            body: "Fed speakers remain hawkish before CPI.",
            importance: 0.75,
            symbols: ["EURUSD", "USDJPY"]
        )
        let context = PluginContextV4(
            horizonMinutes: 15,
            tokenizerContract: PluginTokenizerContractV4(version: "fxai-tokenizer-v1", minNGram: 1, maxNGram: 2),
            textEvents: [event]
        )

        try context.validate()
        XCTAssertEqual(context.textEvents.count, 1)
        XCTAssertTrue(context.textEvents[0].mergedText.contains("risk-on"))
        XCTAssertEqual(context.tokenizerContract.maxNGram, 2)

        let encoded = try JSONEncoder().encode(context)
        let decoded = try JSONDecoder().decode(PluginContextV4.self, from: encoded)
        XCTAssertEqual(decoded.textEvents, context.textEvents)
        XCTAssertEqual(decoded.tokenizerContract, context.tokenizerContract)
    }

    func testSharedAdapterInputDelegatesToTransferPayloadBuilder() {
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        setFeature(&features, 62, 0.25)
        setFeature(&features, 63, -0.10)
        setFeature(&features, 64, 0.40)
        setFeature(&features, 72, 0.50)
        setFeature(&features, 75, 1.0)
        setFeature(&features, 79, 0.20)
        setFeature(&features, 82, 0.60)

        let x = RuntimeTransferTools.modelInputVector(features: features)
        let context = PluginContextV4(
            horizonMinutes: 60,
            sequenceBars: 2,
            domainHash: 0.75
        )
        let state = PluginContextPayloadState(
            context: context,
            windowSize: 1,
            xWindow: [x]
        )

        let payload = state.buildSharedAdapterInput(x: x)
        let expected = PluginSharedTransferPayloadTools.buildInput(
            x: x,
            window: [x],
            declaredWindowSize: 1,
            domainHash: 0.75,
            horizonMinutes: 60
        )

        XCTAssertEqual(payload.count, FXDataEngineConstants.sharedTransferFeatures)
        for index in 0..<FXDataEngineConstants.sharedTransferFeatures {
            XCTAssertEqual(payload[index], expected[index], accuracy: 1e-12)
        }
        XCTAssertTrue(state.hasSharedAdapterSignal(payload))
        XCTAssertEqual(
            state.sharedAdapterSignalStrength(payload),
            PluginTransferSupportTools.sharedAdapterSignalStrength(payload),
            accuracy: 1e-12
        )
    }

    private func setFeature(_ features: inout [Double], _ index: Int, _ value: Double) {
        features[index] = value
    }
}
