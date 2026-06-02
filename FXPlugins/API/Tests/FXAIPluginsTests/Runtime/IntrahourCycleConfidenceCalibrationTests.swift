import FXDataEngine
import XCTest
@testable import FXAIPlugins

final class IntrahourCycleConfidenceCalibrationTests: XCTestCase {
    func testPolicyProducesDeterministicCalibrationEnvelope() throws {
        let policy = FXAIIntrahourCycleCalibrationPolicy.standard
        let calibration = try XCTUnwrap(policy.calibration(
            learnedEdge: 0.90,
            directionalMass: 10.0,
            directionalObservationWeight: 144.0
        ))

        XCTAssertTrue(calibration.deterministic)
        XCTAssertEqual(calibration.confidenceFloor, policy.deterministicConfidenceFloor)
        XCTAssertEqual(calibration.readiness, 1.0, accuracy: 1.0e-12)
        XCTAssertLessThanOrEqual(calibration.activation, policy.strongActivationCeiling)
        XCTAssertGreaterThan(calibration.reliabilityFloor, policy.reliabilityBase)
    }

    func testInsufficientGlobalEvidenceLeavesPredictionUnchanged() {
        var adapter = FXAIIntrahourCycleDirectionAdapter()
        let base = Self.baselinePrediction()

        for _ in 0..<5 {
            adapter.train(Self.trainRequest(label: .buy, timestampUTC: Self.timestamp(minute: 12)))
        }

        let adjusted = adapter.adjustedPrediction(
            base,
            context: Self.pluginContext(timestampUTC: Self.timestamp(minute: 12))
        )

        XCTAssertEqual(adjusted, base)
    }

    func testStrongMinuteEvidenceAppliesConfidenceFloorAndDirectionalOverlay() throws {
        var adapter = FXAIIntrahourCycleDirectionAdapter()
        let base = Self.baselinePrediction()

        for _ in 0..<10 {
            adapter.train(Self.trainRequest(label: .buy, timestampUTC: Self.timestamp(minute: 17)))
        }

        let adjusted = adapter.adjustedPrediction(
            base,
            context: Self.pluginContext(timestampUTC: Self.timestamp(minute: 17))
        )

        try adjusted.validate()
        XCTAssertGreaterThan(adjusted.classProbabilities[LabelClass.buy.rawValue], base.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertLessThan(adjusted.classProbabilities[LabelClass.skip.rawValue], base.classProbabilities[LabelClass.skip.rawValue])
        XCTAssertGreaterThanOrEqual(adjusted.confidence, FXAIIntrahourCycleCalibrationPolicy.standard.deterministicConfidenceFloor)
        XCTAssertGreaterThan(adjusted.reliability, base.reliability)
        XCTAssertEqual(adjusted.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-12)
        XCTAssertGreaterThan(adjusted.moveMeanPoints, 0.0)
        XCTAssertGreaterThanOrEqual(adjusted.moveQ75Points, adjusted.moveQ50Points)
    }

    func testBalancedMinuteEvidenceDoesNotFabricateConfidence() {
        var adapter = FXAIIntrahourCycleDirectionAdapter()
        let base = Self.baselinePrediction()

        for _ in 0..<10 {
            adapter.train(Self.trainRequest(label: .buy, timestampUTC: Self.timestamp(minute: 24)))
            adapter.train(Self.trainRequest(label: .sell, timestampUTC: Self.timestamp(minute: 24)))
        }

        let adjusted = adapter.adjustedPrediction(
            base,
            context: Self.pluginContext(timestampUTC: Self.timestamp(minute: 24))
        )

        XCTAssertEqual(adjusted, base)
    }

    func testResetClearsLearnedCalibrationState() {
        var adapter = FXAIIntrahourCycleDirectionAdapter()
        let base = Self.baselinePrediction()

        for _ in 0..<10 {
            adapter.train(Self.trainRequest(label: .sell, timestampUTC: Self.timestamp(minute: 31)))
        }
        XCTAssertGreaterThanOrEqual(
            adapter.adjustedPrediction(
                base,
                context: Self.pluginContext(timestampUTC: Self.timestamp(minute: 31))
            ).confidence,
            FXAIIntrahourCycleCalibrationPolicy.standard.deterministicConfidenceFloor
        )

        adapter.reset()
        let adjustedAfterReset = adapter.adjustedPrediction(
            base,
            context: Self.pluginContext(timestampUTC: Self.timestamp(minute: 31))
        )

        XCTAssertEqual(adjustedAfterReset, base)
    }

    private static func baselinePrediction() -> PredictionV4 {
        PredictionV4(
            classProbabilities: [0.33, 0.34, 0.33],
            confidence: 0.12,
            reliability: 0.18
        )
    }

    private static func trainRequest(label: LabelClass, timestampUTC: Int64) -> TrainRequestV4 {
        TrainRequestV4(
            valid: true,
            context: pluginContext(timestampUTC: timestampUTC),
            labelClass: label,
            movePoints: 50_000.0,
            sampleWeight: 8.0,
            x: Self.features()
        )
    }

    private static func pluginContext(timestampUTC: Int64) -> PluginContextV4 {
        PluginContextV4(
            horizonMinutes: 1,
            priceCostPoints: 0.0,
            minMovePoints: 1.0,
            sampleTimeUTC: timestampUTC,
            dataHasVolume: true
        )
    }

    private static func timestamp(minute: Int) -> Int64 {
        Int64(minute * 60)
    }

    private static func features() -> [Double] {
        Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
    }
}
