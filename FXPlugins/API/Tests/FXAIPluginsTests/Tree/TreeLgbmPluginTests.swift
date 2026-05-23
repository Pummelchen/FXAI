import FXAIPlugins
import FXDataEngine
import XCTest

final class TreeLgbmPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedLightGBMContract() throws {
        let plugin = TreeLgbmPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.lightgbm.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "tree_lgbm")
        XCTAssertEqual(plugin.manifest.family, .tree)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.replay))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.nativeDistribution))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "tree_lgbm")
        XCTAssertTrue(plan.primaryBackends.contains(.metal))
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testFreshLightGBMModelProducesValidDistributionAndQuantiles() throws {
        let plugin = TreeLgbmPlugin()
        let prediction = try plugin.predict(Self.predictRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.20)
        XCTAssertGreaterThanOrEqual(prediction.moveQ75Points, prediction.moveQ50Points)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testTrainingBuildsHistogramTreesSeparatesDirectionsAndLearnsMoveDistribution() throws {
        var plugin = TreeLgbmPlugin()
        let buyRequest = Self.predictRequest(dataHasVolume: true)
        let sellRequest = Self.sellRequest(from: buyRequest)

        for iteration in 0..<168 {
            try plugin.train(Self.trainRequest(from: buyRequest, label: .buy, movePoints: 4.5 + 0.02 * Double(iteration % 7)), hyperParameters: Self.hyperParameters())
            try plugin.train(Self.trainRequest(from: sellRequest, label: .sell, movePoints: -4.2 - 0.02 * Double(iteration % 7)), hyperParameters: Self.hyperParameters())
        }

        let afterBuy = try plugin.predict(buyRequest, hyperParameters: Self.hyperParameters())
        let afterSell = try plugin.predict(sellRequest, hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(afterBuy.classProbabilities[LabelClass.buy.rawValue], afterSell.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(afterSell.classProbabilities[LabelClass.sell.rawValue], afterBuy.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(afterBuy.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterSell.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterBuy.reliability, 0.50)
        XCTAssertNoThrow(try afterBuy.validate())
        XCTAssertNoThrow(try afterSell.validate())
    }

    func testVolumeAffectsHistogramSplitsOnlyWhenDatasetHasVolume() throws {
        var plugin = TreeLgbmPlugin()
        let withVolume = Self.predictRequest(dataHasVolume: true)
        let withoutVolume = Self.predictRequest(dataHasVolume: false)

        for iteration in 0..<336 {
            var features = withVolume.x
            let highVolume = iteration % 4 != 0
            features[6] = highVolume ? 1.70 : 0.08
            features[80] = highVolume ? 0.92 : 0.14
            let request = PredictRequestV4(valid: true, context: withVolume.context, x: features)
            let label: LabelClass = highVolume ? .buy : .sell
            let move = label == .buy ? 4.1 : -3.8
            try plugin.train(Self.trainRequest(from: request, label: label, movePoints: move), hyperParameters: Self.hyperParameters())
        }

        let predictionWithVolume = try plugin.predict(withVolume, hyperParameters: Self.hyperParameters())
        let predictionWithoutVolume = try plugin.predict(withoutVolume, hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(predictionWithVolume.classProbabilities, predictionWithoutVolume.classProbabilities)
        XCTAssertNoThrow(try predictionWithVolume.validate())
        XCTAssertNoThrow(try predictionWithoutVolume.validate())
    }

    func testResetClearsLightGBMState() throws {
        let request = Self.predictRequest(dataHasVolume: true)
        var plugin = TreeLgbmPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<288 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 4.0), hyperParameters: Self.hyperParameters())
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

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/TreeLgbmCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal/TreeLgbmMetal.swift").path))
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
                sampleTimeUTC: 1_800_030_000,
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
            mfePoints: abs(movePoints) * 1.28,
            maePoints: abs(movePoints) * 0.31,
            timeToHitFraction: 0.36,
            pathRisk: 0.17,
            fillRisk: 0.10,
            nextVolumeTarget: request.context.dataHasVolume ? 1.0 : 0.0,
            x: request.x
        )
    }

    private static func features() -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = 0.17
        values[2] = 0.10
        values[3] = 0.07
        values[4] = 1.10
        values[5] = 0.22
        values[6] = 1.70
        values[7] = 0.05
        values[8] = 0.10
        values[9] = 0.03
        values[12] = 0.08
        values[20] = 0.04
        values[80] = 0.92
        values[81] = 0.36
        return values
    }

    private static func hyperParameters() -> HyperParameters {
        HyperParameters(
            learningRate: 0.03,
            l2: 0.001,
            xgbLearningRate: 0.08,
            xgbL2: 0.02,
            xgbSplit: 0.50
        )
    }

    private static func pluginRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("tree_lgbm")
    }
}
