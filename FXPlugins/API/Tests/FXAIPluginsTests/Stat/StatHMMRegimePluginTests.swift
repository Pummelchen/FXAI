import FXAIPlugins
import FXDataEngine
import XCTest

final class StatHMMRegimePluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedHMMContract() throws {
        let plugin = StatHMMRegimePlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.statHMMRegime.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "stat_hmm_regime")
        XCTAssertEqual(plugin.manifest.family, .stateSpace)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.nativeDistribution))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "stat_hmm_regime")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertFalse(plan.primaryBackends.contains(.metal))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testFreshHMMModelProducesValidRegimeDistribution() throws {
        let plugin = StatHMMRegimePlugin()
        let prediction = try plugin.predict(Self.buyRegimeRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.20)
        XCTAssertGreaterThanOrEqual(prediction.moveQ75Points, prediction.moveQ50Points)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testRegimeObservationSeparatesBuyAndSellDirections() throws {
        let plugin = StatHMMRegimePlugin()
        let buyPrediction = try plugin.predict(Self.buyRegimeRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let sellPrediction = try plugin.predict(Self.sellRegimeRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(buyPrediction.classProbabilities[LabelClass.buy.rawValue], sellPrediction.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(sellPrediction.classProbabilities[LabelClass.sell.rawValue], buyPrediction.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertNoThrow(try buyPrediction.validate())
        XCTAssertNoThrow(try sellPrediction.validate())
    }

    func testTrainingUpdatesRegimesAndReliability() throws {
        var plugin = StatHMMRegimePlugin()
        let buyRequest = Self.buyRegimeRequest(dataHasVolume: true)
        let sellRequest = Self.sellRegimeRequest(dataHasVolume: true)

        for iteration in 0..<120 {
            try plugin.train(Self.trainRequest(from: buyRequest, label: .buy, movePoints: 2.5 + 0.01 * Double(iteration % 5)), hyperParameters: Self.hyperParameters())
            try plugin.train(Self.trainRequest(from: sellRequest, label: .sell, movePoints: -2.4 - 0.01 * Double(iteration % 5)), hyperParameters: Self.hyperParameters())
        }

        let afterBuy = try plugin.predict(buyRequest, hyperParameters: Self.hyperParameters())
        let afterSell = try plugin.predict(sellRequest, hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(afterBuy.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterSell.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterBuy.reliability, 0.60)
        XCTAssertNoThrow(try afterBuy.validate())
        XCTAssertNoThrow(try afterSell.validate())
    }

    func testVolumeAffectsRegimeConfidenceOnlyWhenDatasetHasVolume() throws {
        let withVolume = Self.buyRegimeRequest(dataHasVolume: true)
        let withoutVolume = Self.buyRegimeRequest(dataHasVolume: false)
        let plugin = StatHMMRegimePlugin()

        let predictionWithVolume = try plugin.predict(withVolume, hyperParameters: Self.hyperParameters())
        let predictionWithoutVolume = try plugin.predict(withoutVolume, hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(predictionWithVolume.classProbabilities, predictionWithoutVolume.classProbabilities)
        XCTAssertNoThrow(try predictionWithVolume.validate())
        XCTAssertNoThrow(try predictionWithoutVolume.validate())
    }

    func testResetClearsHMMState() throws {
        let request = Self.buyRegimeRequest(dataHasVolume: true)
        var plugin = StatHMMRegimePlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<80 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 2.5), hyperParameters: Self.hyperParameters())
        }
        let trained = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertNotEqual(trained.reliability, before.reliability)

        plugin.reset()
        let reset = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertEqual(reset.classProbabilities, before.classProbabilities)
        XCTAssertEqual(reset.moveMeanPoints, before.moveMeanPoints, accuracy: 1.0e-12)
    }

    func testConvertedPluginOwnsCPUCodeOnly() throws {
        let root = Self.pluginRoot()

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatHMMRegimeCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatHMMRegimeAccelerated.swift").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    private static func buyRegimeRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(observation: 0.32, volume: 1.25)
        )
    }

    private static func sellRegimeRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(observation: -0.32, volume: 1.10)
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
            horizonMinutes: 5,
            sequenceBars: 1,
            priceCostPoints: 0.20,
            minMovePoints: 0.50,
            pointValue: 1.0,
            sampleTimeUTC: 1_800_090_000,
            dataHasVolume: dataHasVolume
        )
    }

    private static func features(observation: Double, volume: Double) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = observation
        values[2] = 0.20 * observation
        values[3] = 0.15 * observation
        values[4] = 1.02
        values[6] = volume
        values[7] = 0.25 * observation
        values[12] = 0.05 * observation
        values[40] = 0.04
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
        return url.appendingPathComponent("stat_hmm_regime")
    }
}
