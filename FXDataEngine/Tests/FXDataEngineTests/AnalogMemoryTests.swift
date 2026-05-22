import XCTest
@testable import FXDataEngine

final class AnalogMemoryTests: XCTestCase {
    func testAnalogSignalDefaultsMatchLegacyClearHelpers() {
        XCTAssertEqual(FoundationSignals().contextAlignment, 0.5)
        XCTAssertEqual(FoundationSignals().moveRatio, 1.0)
        XCTAssertEqual(StudentSignals().classProbabilities, [0.3333, 0.3333, 0.3334])
        XCTAssertEqual(StudentSignals().horizonFit, 0.5)
        XCTAssertEqual(AnalogMemoryQuery().pathSafety, 0.5)
        XCTAssertEqual(AnalogMemoryQuery().executionSafety, 0.5)
        XCTAssertEqual(HierarchicalSignals().score, 0.0)
    }

    func testAnalogVectorUsesLegacyFeatureMappingAndClamps() {
        let input = makeModelInput([
            0: 5.0,
            3: -5.0,
            5: 10.0,
            10: 2.0,
            41: 7.0,
            62: -5.0,
            72: 3.0,
            78: 9.0,
            80: -5.0,
            82: 9.0,
            FXDataEngineConstants.macroEventFeatureOffset + 2: 0.8,
            FXDataEngineConstants.macroEventFeatureOffset + 19: 0.2,
            FXDataEngineConstants.macroEventFeatureOffset + 3: 5.0,
            FXDataEngineConstants.macroEventFeatureOffset + 15: -5.0
        ])

        assertEqual(
            AnalogMemoryTools.buildVector(modelInput: input),
            [4.0, -4.0, 6.0, 2.0, 6.0, -4.0, 3.0, 8.0, -4.0, 8.0, 0.62, 1.0]
        )
        XCTAssertEqual(AnalogMemoryTools.horizonBucket(horizonMinutes: 0), 0)
        XCTAssertEqual(AnalogMemoryTools.horizonBucket(horizonMinutes: 720), 7)
    }

    func testAnalogMemoryQueryMatchesSingleEntryLegacyWeights() {
        let input = makeModelInput([0: 1.0, 3: 0.5, 5: 2.0, 10: 0.25])
        var store = AnalogMemoryStore(capacity: 4)
        store.update(
            modelInput: input,
            regimeID: 2,
            sessionBucket: 1,
            horizonMinutes: 13,
            domainHash: 0.4,
            movePoints: 10.0,
            minMovePoints: 2.0,
            qualityScore: 1.5,
            pathRisk: 0.2,
            fillRisk: 0.1,
            sampleTimeUTC: 1_704_067_200,
            sampleWeight: 4.0
        )

        let query = store.query(
            modelInput: input,
            regimeID: 2,
            sessionBucket: 1,
            horizonMinutes: 13,
            domainHash: 0.4
        )

        XCTAssertFalse(store.ready)
        XCTAssertEqual(store.size, 1)
        XCTAssertEqual(store.head, 1)
        XCTAssertEqual(query.matches, 1)
        XCTAssertEqual(query.similarity, 1.0, accuracy: 1e-12)
        XCTAssertEqual(query.directionAgreement, 1.0, accuracy: 1e-12)
        XCTAssertEqual(query.edgeNorm, 5.0 / 6.0, accuracy: 1e-12)
        XCTAssertEqual(query.quality, 0.75, accuracy: 1e-12)
        XCTAssertEqual(query.pathSafety, 0.8, accuracy: 1e-12)
        XCTAssertEqual(query.executionSafety, 0.9, accuracy: 1e-12)
        XCTAssertEqual(query.domainAlignment, 1.0, accuracy: 1e-12)
    }

    func testAnalogMemoryRingBufferReadinessAndReset() {
        let input = makeModelInput([0: 1.0])
        var store = AnalogMemoryStore(capacity: 3)
        for index in 0..<4 {
            store.update(
                modelInput: input,
                regimeID: 0,
                sessionBucket: 0,
                horizonMinutes: 5,
                domainHash: 0.2,
                movePoints: Double(index + 1),
                minMovePoints: 0.5,
                qualityScore: 1.0,
                pathRisk: 0.0,
                fillRisk: 0.0,
                sampleTimeUTC: Int64(index),
                sampleWeight: 1.0
            )
        }

        XCTAssertTrue(store.ready)
        XCTAssertEqual(store.size, 3)
        XCTAssertEqual(store.head, 1)
        XCTAssertEqual(store.entries[0].sampleTimeUTC, 3)
        store.reset()
        XCTAssertFalse(store.ready)
        XCTAssertEqual(store.size, 0)
        XCTAssertEqual(store.head, 0)
    }

    func testHierarchicalSignalsStayBoundedAndRewardAgreement() {
        let signals = AnalogMemoryTools.buildHierarchicalSignals(
            classProbabilities: [0.10, 0.80, 0.10],
            expectedMovePoints: 6.0,
            minMovePoints: 2.0,
            confidence: 0.7,
            reliability: 0.6,
            pathRisk: 0.2,
            fillRisk: 0.1,
            hitTimeFraction: 0.3,
            contextQuality: 0.4,
            horizonMinutes: 13,
            foundation: FoundationSignals(directionBias: 1.0, moveRatio: 1.5, tradability: 0.6),
            student: StudentSignals(classProbabilities: [0.15, 0.75, 0.10], moveRatio: 1.4, tradability: 0.5, horizonFit: 0.7),
            analog: AnalogMemoryQuery(
                similarity: 0.9,
                directionAgreement: 1.0,
                edgeNorm: 0.8,
                quality: 0.75,
                pathSafety: 0.8,
                executionSafety: 0.9,
                domainAlignment: 1.0,
                matches: 4
            )
        )

        XCTAssertGreaterThan(signals.tradability, 0.55)
        XCTAssertGreaterThan(signals.directionConfidence, 0.55)
        XCTAssertGreaterThan(signals.score, 0.55)
        XCTAssertLessThanOrEqual(signals.score, 1.0)
    }

    private func makeModelInput(_ features: [Int: Double]) -> [Double] {
        var input = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        input[0] = 1.0
        for (featureIndex, value) in features {
            input[featureIndex + 1] = value
        }
        return input
    }

    private func assertEqual(_ actual: [Double], _ expected: [Double], accuracy: Double = 1e-12) {
        XCTAssertEqual(actual.count, expected.count)
        for index in 0..<min(actual.count, expected.count) {
            XCTAssertEqual(actual[index], expected[index], accuracy: accuracy, "index \(index)")
        }
    }
}
