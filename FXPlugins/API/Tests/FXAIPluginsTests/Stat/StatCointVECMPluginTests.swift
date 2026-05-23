import FXAIPlugins
import FXDataEngine
import XCTest

final class StatCointVECMPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedCointVECMContract() throws {
        let plugin = StatCointVECMPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.statCointVECM.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "stat_coint_vecm")
        XCTAssertEqual(plugin.manifest.family, .stateSpace)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.nativeDistribution))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "stat_coint_vecm")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertFalse(plan.primaryBackends.contains(.metal))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testFreshCointVECMModelProducesValidMeanReversionDistribution() throws {
        let plugin = StatCointVECMPlugin()
        let prediction = try plugin.predict(Self.highResidualRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.20)
        XCTAssertGreaterThanOrEqual(prediction.moveQ75Points, prediction.moveQ50Points)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testTrainingUpdatesResidualStateAndSeparatesMeanReversionDirections() throws {
        var plugin = StatCointVECMPlugin()
        let highResidual = Self.highResidualRequest(dataHasVolume: true)
        let lowResidual = Self.lowResidualRequest(dataHasVolume: true)

        for iteration in 0..<120 {
            try plugin.train(Self.trainRequest(from: highResidual, label: .sell, movePoints: -2.6 - 0.01 * Double(iteration % 4)), hyperParameters: Self.hyperParameters())
            try plugin.train(Self.trainRequest(from: lowResidual, label: .buy, movePoints: 2.4 + 0.01 * Double(iteration % 4)), hyperParameters: Self.hyperParameters())
        }

        let highPrediction = try plugin.predict(highResidual, hyperParameters: Self.hyperParameters())
        let lowPrediction = try plugin.predict(lowResidual, hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(highPrediction.classProbabilities[LabelClass.sell.rawValue], lowPrediction.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(lowPrediction.classProbabilities[LabelClass.buy.rawValue], highPrediction.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(highPrediction.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(highPrediction.reliability, 0.55)
        XCTAssertNoThrow(try highPrediction.validate())
        XCTAssertNoThrow(try lowPrediction.validate())
    }

    func testVolumeAffectsResidualConfidenceOnlyWhenDatasetHasVolume() throws {
        var plugin = StatCointVECMPlugin()
        let withVolume = Self.highResidualRequest(dataHasVolume: true)
        let withoutVolume = Self.highResidualRequest(dataHasVolume: false)

        for iteration in 0..<120 {
            var features = withVolume.x
            features[6] = iteration % 2 == 0 ? 1.40 : 0.02
            let request = PredictRequestV4(valid: true, context: withVolume.context, x: features)
            let label: LabelClass = features[6] > 1.0 ? .sell : .buy
            let move = label == .sell ? -2.5 : 2.3
            try plugin.train(Self.trainRequest(from: request, label: label, movePoints: move), hyperParameters: Self.hyperParameters())
        }

        let predictionWithVolume = try plugin.predict(withVolume, hyperParameters: Self.hyperParameters())
        let predictionWithoutVolume = try plugin.predict(withoutVolume, hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(predictionWithVolume.classProbabilities, predictionWithoutVolume.classProbabilities)
        XCTAssertNoThrow(try predictionWithVolume.validate())
        XCTAssertNoThrow(try predictionWithoutVolume.validate())
    }

    func testResetClearsCointVECMState() throws {
        let request = Self.highResidualRequest(dataHasVolume: true)
        var plugin = StatCointVECMPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<80 {
            try plugin.train(Self.trainRequest(from: request, label: .sell, movePoints: -2.5), hyperParameters: Self.hyperParameters())
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

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatCointVECMCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatCointVECMAccelerated.swift").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    private static func highResidualRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(primary: 0.88, anchor: 0.04, volume: 1.40)
        )
    }

    private static func lowResidualRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(primary: -0.82, anchor: 0.04, volume: 1.10)
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
            sampleTimeUTC: 1_800_070_000,
            dataHasVolume: dataHasVolume
        )
    }

    private static func features(primary: Double, anchor: Double, volume: Double) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = primary
        values[2] = 0.05 * primary
        values[3] = 0.04 * primary
        values[4] = 1.04
        values[6] = volume
        values[7] = 0.03 * primary
        values[12] = anchor
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
        return url.appendingPathComponent("stat_coint_vecm")
    }
}
