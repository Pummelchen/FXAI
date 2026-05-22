import XCTest
@testable import FXDataEngine

final class PreparedSampleWindowTests: XCTestCase {
    func testPreparedSampleWindowUsesOlderAscendingRowsAndStopsOnInvalidSamples() {
        var samples = (0..<6).map { sample(index: $0, valid: true) }
        var window = TrainingSampleTools.preparedSampleWindow(samples: samples, anchorIndex: 4, requestedBars: 3)
        XCTAssertEqual(window.size, 3)
        XCTAssertEqual(window.rows.map { $0[1] }, [3.0, 2.0, 1.0])

        samples[2] = sample(index: 2, valid: false)
        window = TrainingSampleTools.preparedSampleWindow(samples: samples, anchorIndex: 4, requestedBars: 3)
        XCTAssertEqual(window.size, 1)
        XCTAssertEqual(window.rows.map { $0[1] }, [3.0])
    }

    func testPreparedSampleWindowClampsRequestedBarsAndHandlesBoundaryAnchors() {
        let samples = (0..<(FXDataEngineConstants.maxSequenceBars + 4)).map { sample(index: $0, valid: true) }

        let single = TrainingSampleTools.preparedSampleWindow(samples: samples, anchorIndex: 2, requestedBars: 0)
        XCTAssertEqual(single.size, 1)
        XCTAssertEqual(single.rows.map { $0[1] }, [1.0])

        let capped = TrainingSampleTools.preparedSampleWindow(
            samples: samples,
            anchorIndex: samples.count - 1,
            requestedBars: FXDataEngineConstants.maxSequenceBars + 20
        )
        XCTAssertEqual(capped.size, FXDataEngineConstants.maxSequenceBars)
        XCTAssertEqual(capped.rows.first?[1], Double(samples.count - 2))
        XCTAssertEqual(capped.rows.last?[1], Double(samples.count - 1 - FXDataEngineConstants.maxSequenceBars))

        XCTAssertEqual(
            TrainingSampleTools.preparedSampleWindow(samples: samples, anchorIndex: 0, requestedBars: 4).size,
            0
        )
        XCTAssertEqual(
            TrainingSampleTools.preparedSampleWindow(samples: samples, anchorIndex: samples.count, requestedBars: 4).size,
            0
        )
    }

    func testCachedPreparedSampleWindowUsesRoutedNormalizationCache() {
        let aiID = AIModelID.tcn.rawValue
        let configuredHorizons = [1, 13, 60]
        let horizonSlot = TrainingSampleTools.horizonSlot(
            horizonMinutes: 13,
            configuredHorizons: configuredHorizons
        )
        let routing = AITrainingRoutingState(
            modelNormalizationMethods: [aiID: .zScore],
            horizonNormalizationMethods: [AIHorizonRoutingKey(aiID: aiID, horizonSlot: horizonSlot): .robustMedianIQR],
            regimeHorizonNormalizationMethods: [
                AIRegimeHorizonRoutingKey(aiID: aiID, regimeID: 2, horizonSlot: horizonSlot): .dain
            ]
        )

        let source = (0..<6).map { sample(index: $0, valid: true, regimeID: 2, horizonMinutes: 13) }
        let cached = (0..<6).map { sample(index: 100 + $0, valid: true, regimeID: 2, horizonMinutes: 13) }
        let caches = [
            PreparedSampleNormalizationCache(
                method: .robustMedianIQR,
                horizonMinutes: 13,
                fitStartIndex: 0,
                fitEndIndex: 5,
                samples: source
            ),
            PreparedSampleNormalizationCache(
                method: .dain,
                horizonMinutes: 13,
                fitStartIndex: 0,
                fitEndIndex: 5,
                samples: cached
            )
        ]

        XCTAssertEqual(
            TrainingSampleTools.findNormalizationSampleCache(
                method: .dain,
                horizonMinutes: 13,
                fitStartIndex: 0,
                fitEndIndex: 5,
                caches: caches
            ),
            1
        )
        XCTAssertNil(
            TrainingSampleTools.findNormalizationSampleCache(
                method: .dain,
                horizonMinutes: 13,
                fitStartIndex: 1,
                fitEndIndex: 5,
                caches: caches
            )
        )

        let window = TrainingSampleTools.cachedPreparedSampleWindow(
            aiID: aiID,
            samples: source,
            anchorIndex: 4,
            caches: caches,
            requestedBars: 3,
            routingState: routing,
            currentMethod: .existing,
            configuredHorizons: configuredHorizons
        )
        XCTAssertEqual(window.size, 3)
        XCTAssertEqual(window.rows.map { $0[1] }, [103.0, 102.0, 101.0])

        let missingCacheSample = TrainingSampleTools.cachedPreparedSample(
            aiID: aiID,
            referenceSample: source[3],
            sampleIndex: 3,
            caches: [],
            routingState: routing,
            currentMethod: .existing,
            configuredHorizons: configuredHorizons
        )
        XCTAssertFalse(missingCacheSample.valid)
        XCTAssertEqual(missingCacheSample.x[1], 3.0)
    }

    func testRoutedNormalizationCacheRequestsUseLegacyUniqueMethodOrderAndDefault() {
        let aiID = AIModelID.tcn.rawValue
        let configuredHorizons = [1, 13, 60]
        let horizonSlot = TrainingSampleTools.horizonSlot(
            horizonMinutes: 13,
            configuredHorizons: configuredHorizons
        )
        let routing = AITrainingRoutingState(
            modelNormalizationMethods: [aiID: .zScore],
            horizonNormalizationMethods: [AIHorizonRoutingKey(aiID: aiID, horizonSlot: horizonSlot): .robustMedianIQR],
            regimeHorizonNormalizationMethods: [
                AIRegimeHorizonRoutingKey(aiID: aiID, regimeID: 1, horizonSlot: horizonSlot): .dain,
                AIRegimeHorizonRoutingKey(aiID: aiID, regimeID: 2, horizonSlot: horizonSlot): .quantileToNormal
            ]
        )
        var samples = [
            sample(index: 0, valid: true, regimeID: 0, horizonMinutes: 13),
            sample(index: 1, valid: true, regimeID: 1, horizonMinutes: 13),
            sample(index: 2, valid: false, regimeID: 2, horizonMinutes: 13),
            sample(index: 3, valid: true, regimeID: 2, horizonMinutes: 13),
            sample(index: 4, valid: true, regimeID: 1, horizonMinutes: 13)
        ]

        var requests = TrainingSampleTools.routedNormalizationSampleCacheRequests(
            aiID: aiID,
            startIndex: 1,
            endIndex: 4,
            horizonMinutes: 13,
            samples: samples,
            routingState: routing,
            currentMethod: .existing,
            configuredHorizons: configuredHorizons
        )
        XCTAssertEqual(requests.map(\.method), [.dain, .quantileToNormal, .robustMedianIQR])
        XCTAssertEqual(requests.map(\.horizonMinutes), [13, 13, 13])
        XCTAssertEqual(requests.map(\.sampleStartIndex), [1, 1, 1])
        XCTAssertEqual(requests.map(\.sampleEndIndex), [4, 4, 4])

        samples[3] = sample(index: 3, valid: true, regimeID: 1, horizonMinutes: 13)
        requests = TrainingSampleTools.routedNormalizationSampleCacheRequests(
            aiID: aiID,
            startIndex: -2,
            endIndex: 40,
            horizonMinutes: 13,
            samples: samples,
            routingState: routing,
            currentMethod: .existing,
            configuredHorizons: configuredHorizons
        )
        XCTAssertEqual(requests.map(\.method), [.robustMedianIQR, .dain])
        XCTAssertEqual(requests.map(\.sampleStartIndex), [-2, -2])
        XCTAssertEqual(requests.map(\.sampleEndIndex), [40, 40])

        XCTAssertTrue(
            TrainingSampleTools.routedNormalizationSampleCacheRequests(
                aiID: -1,
                startIndex: 0,
                endIndex: 4,
                horizonMinutes: 13,
                samples: samples,
                routingState: routing,
                currentMethod: .existing,
                configuredHorizons: configuredHorizons
            ).isEmpty
        )
    }

    func testCachedPreparedSampleWindowStopsOnInvalidSourceOrCacheRows() {
        let aiID = AIModelID.tcn.rawValue
        var source = (0..<6).map { sample(index: $0, valid: true, regimeID: 0, horizonMinutes: 5) }
        var cached = (0..<6).map { sample(index: 200 + $0, valid: true, regimeID: 0, horizonMinutes: 5) }
        cached[2] = sample(index: 202, valid: false, regimeID: 0, horizonMinutes: 5)
        var caches = [
            PreparedSampleNormalizationCache(
                method: .existing,
                horizonMinutes: 5,
                fitStartIndex: 0,
                fitEndIndex: 5,
                samples: cached
            )
        ]

        var window = TrainingSampleTools.cachedPreparedSampleWindow(
            aiID: aiID,
            samples: source,
            anchorIndex: 4,
            caches: caches,
            requestedBars: 3,
            routingState: AITrainingRoutingState(),
            currentMethod: .existing
        )
        XCTAssertEqual(window.size, 1)
        XCTAssertEqual(window.rows.map { $0[1] }, [203.0])

        source[3] = sample(index: 3, valid: false, regimeID: 0, horizonMinutes: 5)
        caches[0].samples = (0..<6).map { sample(index: 200 + $0, valid: true, regimeID: 0, horizonMinutes: 5) }
        window = TrainingSampleTools.cachedPreparedSampleWindow(
            aiID: aiID,
            samples: source,
            anchorIndex: 4,
            caches: caches,
            requestedBars: 3,
            routingState: AITrainingRoutingState(),
            currentMethod: .existing
        )
        XCTAssertEqual(window.size, 0)
    }

    private func sample(
        index: Int,
        valid: Bool,
        regimeID: Int = 0,
        horizonMinutes: Int = 1
    ) -> PreparedTrainingSample {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[1] = Double(index)
        x[FXDataEngineConstants.aiWeights - 1] = Double(index) * 0.5
        return PreparedTrainingSample(
            valid: valid,
            regimeID: regimeID,
            horizonMinutes: horizonMinutes,
            x: x
        )
    }
}
