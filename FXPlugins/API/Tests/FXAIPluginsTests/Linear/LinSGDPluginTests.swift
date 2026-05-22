import FXAIPlugins
import FXDataEngine
import XCTest

final class LinSGDPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedSGDContract() throws {
        let plugin = LinSGDPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.sgdLogit.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "lin_sgd")
        XCTAssertEqual(plugin.manifest.family, .linear)
        XCTAssertEqual(plugin.manifest.featureSchema, .sparseStat)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.replay))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "lin_sgd")
        XCTAssertTrue(plan.primaryBackends.contains(.swiftSIMD))
        XCTAssertTrue(plan.candidateBackends.contains(.metal))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
        XCTAssertTrue(plan.declaresHardwareAcceleration)
    }

    func testFreshModelMatchesMQL5ZeroStateDistribution() throws {
        let plugin = LinSGDPlugin()
        let prediction = try plugin.predict(Self.predictRequest(dataHasVolume: true), hyperParameters: HyperParameters())

        XCTAssertEqual(prediction.classProbabilities[LabelClass.sell.rawValue], 1.0 / 3.0, accuracy: 1.0e-12)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.buy.rawValue], 1.0 / 3.0, accuracy: 1.0e-12)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.skip.rawValue], 1.0 / 3.0, accuracy: 1.0e-12)
        XCTAssertEqual(prediction.moveMeanPoints, 0.0, accuracy: 1.0e-12)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testTrainingSeparatesBuyAndSellAndLearnsMoveHead() throws {
        var plugin = LinSGDPlugin()
        let buyRequest = Self.predictRequest(dataHasVolume: true)
        var sellFeatures = buyRequest.x
        sellFeatures[0] = -sellFeatures[0]
        sellFeatures[3] = -sellFeatures[3]
        sellFeatures[7] = -sellFeatures[7]
        sellFeatures[8] = abs(sellFeatures[8])
        let sellRequest = PredictRequestV4(valid: true, context: buyRequest.context, x: sellFeatures)

        for _ in 0..<32 {
            try plugin.train(Self.trainRequest(from: buyRequest, label: .buy, movePoints: 3.5), hyperParameters: HyperParameters())
            try plugin.train(Self.trainRequest(from: sellRequest, label: .sell, movePoints: 2.5), hyperParameters: HyperParameters())
        }

        let afterBuy = try plugin.predict(buyRequest, hyperParameters: HyperParameters())
        let afterSell = try plugin.predict(sellRequest, hyperParameters: HyperParameters())

        XCTAssertGreaterThan(afterBuy.classProbabilities[LabelClass.buy.rawValue], afterSell.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(afterSell.classProbabilities[LabelClass.sell.rawValue], afterBuy.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(afterBuy.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterSell.moveMeanPoints, 0.0)
        XCTAssertNoThrow(try afterBuy.validate())
        XCTAssertNoThrow(try afterSell.validate())
    }

    func testVolumeAffectsTrainedModelOnlyWhenDatasetHasVolume() throws {
        var plugin = LinSGDPlugin()
        let withVolume = Self.predictRequest(dataHasVolume: true)
        let withoutVolume = Self.predictRequest(dataHasVolume: false)

        for _ in 0..<24 {
            try plugin.train(Self.trainRequest(from: withVolume, label: .buy, movePoints: 3.0), hyperParameters: HyperParameters())
        }

        let predictionWithVolume = try plugin.predict(withVolume, hyperParameters: HyperParameters())
        let predictionWithoutVolume = try plugin.predict(withoutVolume, hyperParameters: HyperParameters())

        XCTAssertNotEqual(predictionWithVolume.classProbabilities, predictionWithoutVolume.classProbabilities)
        XCTAssertNoThrow(try predictionWithVolume.validate())
        XCTAssertNoThrow(try predictionWithoutVolume.validate())
    }

    func testResetClearsOnlineState() throws {
        let request = Self.predictRequest(dataHasVolume: true)
        var plugin = LinSGDPlugin()
        let before = try plugin.predict(request, hyperParameters: HyperParameters())

        for _ in 0..<20 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 3.0), hyperParameters: HyperParameters())
        }
        let trained = try plugin.predict(request, hyperParameters: HyperParameters())
        XCTAssertNotEqual(trained.classProbabilities, before.classProbabilities)

        plugin.reset()
        let reset = try plugin.predict(request, hyperParameters: HyperParameters())
        XCTAssertEqual(reset.classProbabilities, before.classProbabilities)
        XCTAssertEqual(reset.moveMeanPoints, before.moveMeanPoints, accuracy: 1.0e-12)
    }

    func testConvertedPluginOwnsCPUAndMetalCodeFolders() throws {
        let root = Self.pluginRoot()

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/LinSGDCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal/LinSGDMetal.swift").path))
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
        values[0] = 0.09
        values[3] = 0.04
        values[4] = 0.20
        values[6] = 0.85
        values[7] = 0.08
        values[8] = -0.03
        values[12] = 0.06
        return values
    }

    private static func pluginRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("lin_sgd")
    }
}
