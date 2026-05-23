import FXAIPlugins
import FXDataEngine
import XCTest

final class MixLoffmPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedLOFFMContract() throws {
        let plugin = MixLoffmPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.loffm.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "mix_loffm")
        XCTAssertEqual(plugin.manifest.family, .mixture)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.replay))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "mix_loffm")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertTrue(plan.candidateBackends.contains(.pyTorchMPS))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testFreshSeededExpertsProduceValidDistribution() throws {
        let plugin = MixLoffmPlugin()
        let prediction = try plugin.predict(Self.predictRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
        XCTAssertGreaterThanOrEqual(prediction.moveMeanPoints, 0.0)
        XCTAssertGreaterThanOrEqual(prediction.moveQ75Points, prediction.moveQ25Points)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testTrainingSeparatesDirectionsAndLearnsMoveHead() throws {
        var plugin = MixLoffmPlugin()
        let buyRequest = Self.predictRequest(dataHasVolume: true)
        let sellRequest = Self.sellRequest(from: buyRequest)

        for _ in 0..<80 {
            try plugin.train(Self.trainRequest(from: buyRequest, label: .buy, movePoints: 3.4), hyperParameters: Self.hyperParameters())
            try plugin.train(Self.trainRequest(from: sellRequest, label: .sell, movePoints: -3.1), hyperParameters: Self.hyperParameters())
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

    func testVolumeAffectsMixtureOnlyWhenDatasetHasVolume() throws {
        var plugin = MixLoffmPlugin()
        let withVolume = Self.predictRequest(dataHasVolume: true)
        let withoutVolume = Self.predictRequest(dataHasVolume: false)

        for _ in 0..<48 {
            try plugin.train(Self.trainRequest(from: withVolume, label: .buy, movePoints: 2.8), hyperParameters: Self.hyperParameters())
        }

        let predictionWithVolume = try plugin.predict(withVolume, hyperParameters: Self.hyperParameters())
        let predictionWithoutVolume = try plugin.predict(withoutVolume, hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(predictionWithVolume.classProbabilities, predictionWithoutVolume.classProbabilities)
        XCTAssertNoThrow(try predictionWithVolume.validate())
        XCTAssertNoThrow(try predictionWithoutVolume.validate())
    }

    func testResetClearsOnlineState() throws {
        let request = Self.predictRequest(dataHasVolume: true)
        var plugin = MixLoffmPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<40 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 3.0), hyperParameters: Self.hyperParameters())
        }
        let trained = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertNotEqual(trained.classProbabilities, before.classProbabilities)

        plugin.reset()
        let reset = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertEqual(reset.classProbabilities, before.classProbabilities)
        XCTAssertEqual(reset.moveMeanPoints, before.moveMeanPoints, accuracy: 1.0e-12)
    }

    func testConvertedPluginOwnsCPUAndPyTorchCodeFolders() throws {
        let root = Self.pluginRoot()

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/MixLoffmCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch/mix_loffm_torch.py").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal").path))
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
                priceCostPoints: 0.18,
                minMovePoints: 0.45,
                pointValue: 1.0,
                sampleTimeUTC: 1_800_022_000,
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
        features[7] = -features[7]
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
            maePoints: abs(movePoints) * 0.35,
            timeToHitFraction: 0.40,
            pathRisk: 0.20,
            fillRisk: 0.10,
            nextVolumeTarget: request.context.dataHasVolume ? 1.0 : 0.0,
            x: request.x
        )
    }

    private static func features() -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = 0.11
        values[2] = 0.07
        values[3] = 0.05
        values[4] = 0.20
        values[5] = 0.12
        values[6] = 1.20
        values[7] = 0.08
        values[8] = -0.02
        values[9] = 0.04
        values[10] = 0.03
        values[11] = -0.02
        values[12] = 0.06
        values[80] = 0.80
        values[81] = 0.35
        return values
    }

    private static func hyperParameters() -> HyperParameters {
        HyperParameters(learningRate: 0.025, l2: 0.001)
    }

    private static func pluginRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("mix_loffm")
    }
}
