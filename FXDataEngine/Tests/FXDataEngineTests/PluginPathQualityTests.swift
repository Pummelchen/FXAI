import XCTest
@testable import FXDataEngine

final class PluginPathQualityTests: XCTestCase {
    func testPathQualityPopulatesDeterministicNoWindowFallback() throws {
        let context = PluginContextV4(
            horizonMinutes: 5,
            sequenceBars: 1,
            priceCostPoints: 1.2,
            minMovePoints: 0.5,
            dataHasVolume: true
        )
        let output = PluginModelOutputV4(
            classProbabilities: [0.20, 0.60, 0.20],
            moveMeanPoints: 4.0,
            moveQ25Points: 2.0,
            moveQ50Points: 4.0,
            moveQ75Points: 6.0
        )
        let result = PluginPathQualityTools.populatedOutput(
            output,
            x: makeInput(costSlot: 99.0),
            window: [],
            context: context,
            family: .tree,
            activityGate: 0.50,
            structuralQuality: 0.80,
            executionQuality: 0.70
        )

        XCTAssertTrue(result.hasPathQuality)
        XCTAssertEqual(result.mfeMeanPoints, 6.0, accuracy: 1e-12)
        XCTAssertEqual(result.maeMeanPoints, 1.404553846153846, accuracy: 1e-12)
        XCTAssertEqual(result.hitTimeFraction, 0.560923076923077, accuracy: 1e-12)
        XCTAssertEqual(result.pathRisk, 0.29653292307692314, accuracy: 1e-12)
        XCTAssertEqual(result.fillRisk, 0.21615384615384617, accuracy: 1e-12)

        let prediction = PluginContextRuntimeTools.fillPrediction(
            modelOutput: result,
            calibratedMoveMeanPoints: 4.0,
            context: context
        )
        try prediction.validate()
        XCTAssertEqual(prediction.pathRisk, result.pathRisk, accuracy: 1e-12)
        XCTAssertEqual(prediction.fillRisk, result.fillRisk, accuracy: 1e-12)

        let nonFinite = PluginPathQualityTools.populatedOutput(
            PluginModelOutputV4(classProbabilities: [.nan, .infinity, 0.0], moveMeanPoints: .nan, moveQ75Points: .infinity),
            x: makeInput(costSlot: .infinity),
            window: [],
            context: context,
            family: .other,
            activityGate: .nan,
            structuralQuality: .infinity
        )
        XCTAssertTrue(nonFinite.mfeMeanPoints.isFinite)
        XCTAssertTrue(nonFinite.maeMeanPoints.isFinite)
        XCTAssertTrue(nonFinite.hitTimeFraction.isFinite)
        XCTAssertTrue(nonFinite.pathRisk.isFinite)
        XCTAssertTrue(nonFinite.fillRisk.isFinite)
    }

    func testPathQualityUsesWindowShapeFamilyMultipliersAndQualityBankPriors() {
        let context = PluginContextV4(
            horizonMinutes: 21,
            sequenceBars: 5,
            priceCostPoints: 0.6,
            minMovePoints: 0.2,
            dataHasVolume: true
        )
        let output = PluginModelOutputV4(
            classProbabilities: [0.30, 0.20, 0.50],
            moveMeanPoints: 3.0,
            moveQ25Points: 1.0,
            moveQ50Points: 2.5,
            moveQ75Points: 4.0
        )
        let priors = PluginQualityBankPriors(
            mfePoints: 9.0,
            maePoints: 2.0,
            hitTimeFraction: 0.30,
            pathRisk: 0.25,
            fillRisk: 0.35,
            trust: 0.60
        )

        let result = PluginPathQualityTools.populatedOutput(
            output,
            x: makeInput(costSlot: 0.0),
            window: makeWindow(),
            context: context,
            family: .transformer,
            activityGate: 0.90,
            structuralQuality: 0.70,
            executionQuality: 0.40,
            qualityPriors: priors,
            declaredWindowSize: 4
        )

        XCTAssertEqual(result.mfeMeanPoints, 6.439464149488645, accuracy: 1e-12)
        XCTAssertEqual(result.maeMeanPoints, 0.9809400814476178, accuracy: 1e-12)
        XCTAssertEqual(result.hitTimeFraction, 0.43795714567417793, accuracy: 1e-12)
        XCTAssertEqual(result.pathRisk, 0.32155054186480647, accuracy: 1e-12)
        XCTAssertEqual(result.fillRisk, 0.3837478407058622, accuracy: 1e-12)
    }

    private func makeInput(costSlot: Double) -> [Double] {
        var input = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        input[0] = 1.0
        input[7] = costSlot
        return input
    }

    private func makeWindow() -> [[Double]] {
        let rows: [(Double, Double)] = [
            (1.2, 0.8),
            (0.8, 0.6),
            (-0.1, 0.4),
            (-0.4, 0.2)
        ]
        return rows.map { feature0, feature10 in
            var input = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
            input[0] = 1.0
            input[1] = feature0
            input[11] = feature10
            return input
        }
    }
}
