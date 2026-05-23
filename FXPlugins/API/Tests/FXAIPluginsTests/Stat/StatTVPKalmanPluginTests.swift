import FXAIPlugins
import FXDataEngine
import XCTest

final class StatTVPKalmanPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedContract() throws {
        let plugin = StatTVPKalmanPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.statTVPKalman.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "stat_tvp_kalman")
        XCTAssertEqual(plugin.manifest.family, .stateSpace)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.nativeDistribution))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "stat_tvp_kalman")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertFalse(plan.primaryBackends.contains(.metal))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testFreshKalmanModelProducesValidDistribution() throws {
        let plugin = StatTVPKalmanPlugin()
        let prediction = try plugin.predict(Self.buyRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.10)
        XCTAssertGreaterThanOrEqual(prediction.moveQ75Points, prediction.moveQ50Points)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testTrainingUpdatesKalmanStateAndSeparatesDirections() throws {
        var plugin = StatTVPKalmanPlugin()
        let buyRequest = Self.buyRequest(dataHasVolume: true)
        let sellRequest = Self.sellRequest(dataHasVolume: true)

        for iteration in 0..<112 {
            try plugin.train(Self.trainRequest(from: buyRequest, label: .buy, movePoints: 2.5 + 0.01 * Double(iteration % 5)), hyperParameters: Self.hyperParameters())
            try plugin.train(Self.trainRequest(from: sellRequest, label: .sell, movePoints: 2.2 + 0.01 * Double(iteration % 4)), hyperParameters: Self.hyperParameters())
        }

        let buyPrediction = try plugin.predict(buyRequest, hyperParameters: Self.hyperParameters())
        let sellPrediction = try plugin.predict(sellRequest, hyperParameters: Self.hyperParameters())
        XCTAssertGreaterThan(buyPrediction.classProbabilities[LabelClass.buy.rawValue], sellPrediction.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(sellPrediction.classProbabilities[LabelClass.sell.rawValue], buyPrediction.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(buyPrediction.reliability, 0.55)
        XCTAssertNoThrow(try buyPrediction.validate())
        XCTAssertNoThrow(try sellPrediction.validate())
    }

    func testVolumeAffectsKalmanUncertaintyOnlyWhenDatasetHasVolume() throws {
        let plugin = StatTVPKalmanPlugin()
        let withVolume = try plugin.predict(Self.buyRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let withoutVolume = try plugin.predict(Self.buyRequest(dataHasVolume: false), hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(withVolume.classProbabilities, withoutVolume.classProbabilities)
        XCTAssertNoThrow(try withVolume.validate())
        XCTAssertNoThrow(try withoutVolume.validate())
    }

    func testResetClearsKalmanState() throws {
        let request = Self.buyRequest(dataHasVolume: true)
        var plugin = StatTVPKalmanPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<80 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 2.4), hyperParameters: Self.hyperParameters())
        }
        let trained = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertNotEqual(trained.classProbabilities, before.classProbabilities)
        XCTAssertNotEqual(trained.reliability, before.reliability)

        plugin.reset()
        let reset = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertEqual(reset.classProbabilities, before.classProbabilities)
        XCTAssertEqual(reset.moveMeanPoints, before.moveMeanPoints, accuracy: 1.0e-12)
    }

    func testConvertedPluginOwnsCPUCodeOnly() throws {
        let root = Self.pluginRoot()

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatTVPKalmanCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatTVPKalmanAccelerated.swift").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    private static func buyRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(direction: 1.0, volume: 1.35)
        )
    }

    private static func sellRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(direction: -1.0, volume: 1.15)
        )
    }

    private static func trainRequest(from request: PredictRequestV4, label: LabelClass, movePoints: Double) -> TrainRequestV4 {
        TrainRequestV4(
            valid: true,
            context: request.context,
            labelClass: label,
            movePoints: movePoints,
            sampleWeight: 1.0,
            mfePoints: abs(movePoints) * 1.20,
            maePoints: abs(movePoints) * 0.35,
            timeToHitFraction: 0.42,
            pathRisk: 0.18,
            fillRisk: 0.11,
            nextVolumeTarget: request.context.dataHasVolume ? 1.0 : 0.0,
            x: request.x
        )
    }

    private static func context(dataHasVolume: Bool) -> PluginContextV4 {
        PluginContextV4(
            sessionBucket: 2,
            horizonMinutes: 10,
            sequenceBars: 1,
            priceCostPoints: 0.20,
            minMovePoints: 0.50,
            pointValue: 1.0,
            sampleTimeUTC: 1_800_100_000,
            dataHasVolume: dataHasVolume
        )
    }

    private static func features(direction: Double, volume: Double) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = 0.18 * direction
        values[2] = 0.10 * direction
        values[3] = 0.12 * direction
        values[4] = 0.24
        values[6] = volume
        values[7] = 0.08 * direction
        values[12] = 0.22 * direction
        values[40] = 0.10 * volume
        values[FXDataEngineConstants.macroEventFeatureOffset + 14] = 0.06 * direction
        values[FXDataEngineConstants.macroEventFeatureOffset + 19] = 0.05
        return values
    }

    private static func hyperParameters() -> HyperParameters {
        HyperParameters(learningRate: 0.03, l2: 0.001)
    }

    private static func pluginRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("stat_tvp_kalman")
    }
}
