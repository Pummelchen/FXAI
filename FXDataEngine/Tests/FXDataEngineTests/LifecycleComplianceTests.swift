import XCTest
@testable import FXDataEngine

final class LifecycleComplianceTests: XCTestCase {
    func testPredictionValidationMirrorsLegacyReasons() {
        let valid = PredictionV4(
            classProbabilities: [0.20, 0.30, 0.50],
            moveMeanPoints: 4.0,
            moveQ25Points: 2.0,
            moveQ50Points: 4.0,
            moveQ75Points: 6.0,
            mfeMeanPoints: 7.0,
            maeMeanPoints: 1.5,
            hitTimeFraction: 0.4,
            pathRisk: 0.2,
            fillRisk: 0.3,
            confidence: 0.7,
            reliability: 0.8
        )
        XCTAssertTrue(LifecycleComplianceTools.validatePredictionOutput(valid).isValid)

        let badSum = PredictionV4(classProbabilities: [0.20, 0.30, 0.70])
        XCTAssertEqual(LifecycleComplianceTools.validatePredictionOutput(badSum).reason, "class_sum_norm")

        let badRange = PredictionV4(classProbabilities: [1.01, 0.0, 0.0])
        XCTAssertEqual(LifecycleComplianceTools.validatePredictionOutput(badRange).reason, "class_probs_range")

        var badMove = valid
        badMove.moveMeanPoints = -0.1
        XCTAssertEqual(LifecycleComplianceTools.validatePredictionOutput(badMove).reason, "move_mean")
    }

    func testSequenceBarsAndHorizonClampLikeComplianceHarness() {
        let single = PluginManifestV4(aiID: 1, aiName: "Single", family: .linear)
        XCTAssertEqual(LifecycleComplianceTools.complianceSequenceBars(manifest: single), 1)

        let sequence = PluginManifestV4(
            aiID: 2,
            aiName: "Sequence",
            family: .transformer,
            minHorizonMinutes: 5,
            maxHorizonMinutes: 20,
            minSequenceBars: 4,
            maxSequenceBars: 12
        )
        XCTAssertEqual(LifecycleComplianceTools.complianceSequenceBars(manifest: sequence), 12)
        XCTAssertEqual(LifecycleComplianceTools.complianceHorizon(manifest: sequence, desiredHorizonMinutes: 2), 5)
        XCTAssertEqual(LifecycleComplianceTools.complianceHorizon(manifest: sequence, desiredHorizonMinutes: 13), 13)
        XCTAssertEqual(LifecycleComplianceTools.complianceHorizon(manifest: sequence, desiredHorizonMinutes: 34), 20)

        let long = PluginManifestV4(
            aiID: 3,
            aiName: "LongSequence",
            family: .transformer,
            minSequenceBars: 20,
            maxSequenceBars: 20
        )
        XCTAssertEqual(LifecycleComplianceTools.complianceSequenceBars(manifest: long), 16)
    }

    func testComplianceWindowAppliesLegacyDecay() {
        let features = [1.0, -2.0, 0.5]
        let window = LifecycleComplianceTools.complianceWindow(features: features, sequenceBars: 4)

        XCTAssertEqual(window.count, 3)
        XCTAssertEqual(window[0][0], 0.92, accuracy: 1e-12)
        XCTAssertEqual(window[0][1], -1.84, accuracy: 1e-12)
        XCTAssertEqual(window[1][0], 0.84, accuracy: 1e-12)
        XCTAssertEqual(window[2][2], 0.38, accuracy: 1e-12)
        XCTAssertTrue(LifecycleComplianceTools.complianceWindow(features: features, sequenceBars: 1).isEmpty)
    }

    func testPredictionDistanceMatchesLegacyWeights() {
        let first = PredictionV4(
            classProbabilities: [0.20, 0.50, 0.30],
            moveMeanPoints: 10.0,
            moveQ25Points: 8.0,
            moveQ50Points: 10.0,
            moveQ75Points: 12.0,
            mfeMeanPoints: 14.0,
            maeMeanPoints: 4.0,
            hitTimeFraction: 0.4,
            pathRisk: 0.2,
            fillRisk: 0.3,
            confidence: 0.7,
            reliability: 0.8
        )
        let second = PredictionV4(
            classProbabilities: [0.10, 0.60, 0.30],
            moveMeanPoints: 6.0,
            moveQ25Points: 4.0,
            moveQ50Points: 7.0,
            moveQ75Points: 9.0,
            mfeMeanPoints: 10.0,
            maeMeanPoints: 5.0,
            hitTimeFraction: 0.7,
            pathRisk: 0.5,
            fillRisk: 0.1,
            confidence: 0.6,
            reliability: 0.4
        )

        XCTAssertEqual(LifecycleComplianceTools.predictionDistance(first, second), 0.79, accuracy: 1e-12)
    }

    func testComplianceRandSymmetricMatchesLegacyLcg() {
        var state: UInt64 = 1
        let first = LifecycleComplianceTools.complianceRandSymmetric(state: &state)

        XCTAssertEqual(state, 1_015_568_748)
        XCTAssertEqual(first, 0.7973, accuracy: 1e-12)
    }
}
