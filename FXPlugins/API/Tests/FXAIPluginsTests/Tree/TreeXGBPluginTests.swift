import FXAIPlugins
import FXDataEngine
import XCTest

final class TreeXGBPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedXGBContract() throws {
        let plugin = TreeXGBPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.xgboost.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "tree_xgb")
        XCTAssertEqual(plugin.manifest.family, .tree)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.replay))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "tree_xgb")
        XCTAssertTrue(plan.primaryBackends.contains(.metal))
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testFreshXGBModelProducesLegacyBalancedTreeDistribution() throws {
        let plugin = TreeXGBPlugin()
        let prediction = try plugin.predict(Self.predictRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.buy.rawValue], prediction.classProbabilities[LabelClass.sell.rawValue], accuracy: 1.0e-9)
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.20)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testTrainingBuildsTreesSeparatesDirectionsAndLearnsMoveDistribution() throws {
        var plugin = TreeXGBPlugin()
        let buyRequest = Self.predictRequest(dataHasVolume: true)
        let sellRequest = Self.sellRequest(from: buyRequest)

        for iteration in 0..<180 {
            try plugin.train(Self.trainRequest(from: buyRequest, label: .buy, movePoints: 3.8 + 0.02 * Double(iteration % 7)), hyperParameters: Self.hyperParameters())
            try plugin.train(Self.trainRequest(from: sellRequest, label: .sell, movePoints: -3.5 - 0.02 * Double(iteration % 7)), hyperParameters: Self.hyperParameters())
        }

        let afterBuy = try plugin.predict(buyRequest, hyperParameters: Self.hyperParameters())
        let afterSell = try plugin.predict(sellRequest, hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(afterBuy.classProbabilities[LabelClass.buy.rawValue], afterSell.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(afterSell.classProbabilities[LabelClass.sell.rawValue], afterBuy.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(afterBuy.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterSell.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterBuy.reliability, 0.35)
        XCTAssertNoThrow(try afterBuy.validate())
        XCTAssertNoThrow(try afterSell.validate())
    }

    func testVolumeAffectsTreeSplitsOnlyWhenDatasetHasVolume() throws {
        var plugin = TreeXGBPlugin()
        let withVolume = Self.predictRequest(dataHasVolume: true)
        let withoutVolume = Self.predictRequest(dataHasVolume: false)

        for iteration in 0..<256 {
            var features = withVolume.x
            let highVolume = iteration % 4 != 0
            features[6] = highVolume ? 1.35 : 0.15
            let request = PredictRequestV4(valid: true, context: withVolume.context, x: features)
            let label: LabelClass = highVolume ? .buy : .sell
            let move = label == .buy ? 3.2 : -3.0
            try plugin.train(Self.trainRequest(from: request, label: label, movePoints: move), hyperParameters: Self.hyperParameters())
        }

        let predictionWithVolume = try plugin.predict(withVolume, hyperParameters: Self.hyperParameters())
        let predictionWithoutVolume = try plugin.predict(withoutVolume, hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(predictionWithVolume.classProbabilities, predictionWithoutVolume.classProbabilities)
        XCTAssertNoThrow(try predictionWithVolume.validate())
        XCTAssertNoThrow(try predictionWithoutVolume.validate())
    }

    func testResetClearsTreeState() throws {
        let request = Self.predictRequest(dataHasVolume: true)
        var plugin = TreeXGBPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<160 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 3.4), hyperParameters: Self.hyperParameters())
        }
        let trained = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertNotEqual(trained.classProbabilities, before.classProbabilities)

        plugin.reset()
        let reset = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertEqual(reset.classProbabilities, before.classProbabilities)
        XCTAssertEqual(reset.moveMeanPoints, before.moveMeanPoints, accuracy: 1.0e-12)
    }

    func testConvertedPluginOwnsCPUAndMetalCodeFolders() throws {
        let root = Self.pluginRoot()

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/TreeXGBCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal/TreeXGBMetal.swift").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    private static func predictRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: PluginContextV4(
                sessionBucket: 2,
                horizonMinutes: 5,
                sequenceBars: 1,
                priceCostPoints: 0.20,
                minMovePoints: 0.50,
                pointValue: 1.0,
                sampleTimeUTC: 1_800_025_000,
                dataHasVolume: dataHasVolume
            ),
            x: features()
        )
    }

    private static func sellRequest(from request: PredictRequestV4) -> PredictRequestV4 {
        var features = request.x
        features[1] = -features[1]
        features[2] = -features[2]
        features[3] = -features[3]
        features[8] = -features[8]
        features[12] = -features[12]
        return PredictRequestV4(valid: true, context: request.context, x: features)
    }

    private static func trainRequest(from request: PredictRequestV4, label: LabelClass, movePoints: Double) -> TrainRequestV4 {
        TrainRequestV4(
            valid: true,
            context: request.context,
            labelClass: label,
            movePoints: movePoints,
            sampleWeight: 1.0,
            mfePoints: abs(movePoints) * 1.20,
            maePoints: abs(movePoints) * 0.32,
            timeToHitFraction: 0.40,
            pathRisk: 0.18,
            fillRisk: 0.10,
            nextVolumeTarget: request.context.dataHasVolume ? 1.0 : 0.0,
            x: request.x
        )
    }

    private static func features() -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = 0.14
        values[2] = 0.08
        values[3] = 0.05
        values[4] = 1.05
        values[5] = 0.20
        values[6] = 1.35
        values[7] = 0.04
        values[8] = 0.09
        values[9] = 0.02
        values[12] = 0.07
        values[80] = 0.90
        values[81] = 0.30
        return values
    }

    private static func hyperParameters() -> HyperParameters {
        HyperParameters(
            learningRate: 0.03,
            l2: 0.001,
            xgbLearningRate: 0.15,
            xgbL2: 0.02,
            xgbSplit: 0.50
        )
    }

    private static func pluginRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("tree_xgb")
    }
}
