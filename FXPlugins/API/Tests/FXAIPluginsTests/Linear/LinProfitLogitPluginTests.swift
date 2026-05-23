import FXAIPlugins
import FXDataEngine
import XCTest

final class LinProfitLogitPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedProfitLogitContract() throws {
        let plugin = LinProfitLogitPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.linProfitLogit.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "lin_profit_logit")
        XCTAssertEqual(plugin.manifest.family, .linear)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.replay))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "lin_profit_logit")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertTrue(plan.candidateBackends.contains(.metal))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testFreshSeededFrameworkModelProducesValidDistribution() throws {
        let plugin = LinProfitLogitPlugin()
        let prediction = try plugin.predict(Self.predictRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
        XCTAssertGreaterThanOrEqual(prediction.classProbabilities[LabelClass.skip.rawValue], 0.05)
        XCTAssertGreaterThanOrEqual(prediction.moveMeanPoints, 0.0)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testTrainingSeparatesBuyAndSellAndLearnsMoveEMA() throws {
        var plugin = LinProfitLogitPlugin()
        let buyRequest = Self.predictRequest(dataHasVolume: true)
        var sellFeatures = buyRequest.x
        sellFeatures[1] = -sellFeatures[1]
        sellFeatures[2] = -sellFeatures[2]
        sellFeatures[3] = -sellFeatures[3]
        sellFeatures[7] = -sellFeatures[7]
        sellFeatures[12] = -sellFeatures[12]
        let sellRequest = PredictRequestV4(valid: true, context: buyRequest.context, x: sellFeatures)

        for _ in 0..<64 {
            try plugin.train(Self.trainRequest(from: buyRequest, label: .buy, movePoints: 3.2), hyperParameters: Self.hyperParameters())
            try plugin.train(Self.trainRequest(from: sellRequest, label: .sell, movePoints: 2.8), hyperParameters: Self.hyperParameters())
        }

        let afterBuy = try plugin.predict(buyRequest, hyperParameters: Self.hyperParameters())
        let afterSell = try plugin.predict(sellRequest, hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(afterBuy.classProbabilities[LabelClass.buy.rawValue], afterSell.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(afterSell.classProbabilities[LabelClass.sell.rawValue], afterBuy.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(afterBuy.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterSell.moveMeanPoints, 0.0)
        XCTAssertNoThrow(try afterBuy.validate())
        XCTAssertNoThrow(try afterSell.validate())
    }

    func testVolumeAffectsModelOnlyWhenDatasetHasVolume() throws {
        var plugin = LinProfitLogitPlugin()
        let withVolume = Self.predictRequest(dataHasVolume: true)
        let withoutVolume = Self.predictRequest(dataHasVolume: false)

        for _ in 0..<32 {
            try plugin.train(Self.trainRequest(from: withVolume, label: .buy, movePoints: 3.0), hyperParameters: Self.hyperParameters())
        }

        let predictionWithVolume = try plugin.predict(withVolume, hyperParameters: Self.hyperParameters())
        let predictionWithoutVolume = try plugin.predict(withoutVolume, hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(predictionWithVolume.classProbabilities, predictionWithoutVolume.classProbabilities)
        XCTAssertNoThrow(try predictionWithVolume.validate())
        XCTAssertNoThrow(try predictionWithoutVolume.validate())
    }

    func testResetClearsOnlineState() throws {
        let request = Self.predictRequest(dataHasVolume: true)
        var plugin = LinProfitLogitPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<32 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 3.0), hyperParameters: Self.hyperParameters())
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

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/LinProfitLogitCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal/LinProfitLogitMetal.swift").path))
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
                sampleTimeUTC: 1_800_018_000,
                dataHasVolume: dataHasVolume
            ),
            x: features()
        )
    }

    private static func trainRequest(from request: PredictRequestV4, label: LabelClass, movePoints: Double) -> TrainRequestV4 {
        TrainRequestV4(
            valid: true,
            context: request.context,
            labelClass: label,
            movePoints: movePoints,
            sampleWeight: 1.0,
            mfePoints: abs(movePoints) * 1.25,
            maePoints: abs(movePoints) * 0.35,
            timeToHitFraction: 0.45,
            pathRisk: 0.20,
            fillRisk: 0.10,
            nextVolumeTarget: request.context.dataHasVolume ? 1.0 : 0.0,
            x: request.x
        )
    }

    private static func features() -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = 0.09
        values[2] = 0.03
        values[3] = 0.04
        values[4] = 0.20
        values[6] = 1.25
        values[7] = 0.08
        values[12] = 0.06
        values[40] = 0.05
        values[80] = 0.70
        values[81] = 0.30
        return values
    }

    private static func hyperParameters() -> HyperParameters {
        HyperParameters(learningRate: 0.04, l2: 0.002)
    }

    private static func pluginRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("lin_profit_logit")
    }
}
