import FXAIPlugins
import FXDataEngine
import XCTest

final class StatXrateConsistencyPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedContract() throws {
        let plugin = StatXrateConsistencyPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.statXRateConsistency.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "stat_xrate_consistency")
        XCTAssertEqual(plugin.manifest.family, .stateSpace)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.nativeDistribution))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "stat_xrate_consistency")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertFalse(plan.primaryBackends.contains(.metal))
        XCTAssertFalse(plan.primaryBackends.contains(.pyTorchMPS))
        XCTAssertFalse(plan.primaryBackends.contains(.tensorFlowMetal))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testFreshModelProducesValidDistribution() throws {
        let plugin = StatXrateConsistencyPlugin()
        let prediction = try plugin.predict(Self.buyCorrectionRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.10)
        XCTAssertGreaterThanOrEqual(prediction.moveQ75Points, prediction.moveQ50Points)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testConsistencyDivergenceSeparatesBuyAndSellDirections() throws {
        let plugin = StatXrateConsistencyPlugin()
        let buyPrediction = try plugin.predict(Self.buyCorrectionRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let sellPrediction = try plugin.predict(Self.sellCorrectionRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(buyPrediction.classProbabilities[LabelClass.buy.rawValue], sellPrediction.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(sellPrediction.classProbabilities[LabelClass.sell.rawValue], buyPrediction.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertNoThrow(try buyPrediction.validate())
        XCTAssertNoThrow(try sellPrediction.validate())
    }

    func testTrainingUpdatesMoveAndReliabilityState() throws {
        let request = Self.buyCorrectionRequest(dataHasVolume: true)
        var plugin = StatXrateConsistencyPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for iteration in 0..<96 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 2.2 + 0.01 * Double(iteration % 4)), hyperParameters: Self.hyperParameters())
        }

        let after = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertGreaterThan(after.moveMeanPoints, before.moveMeanPoints)
        XCTAssertGreaterThan(after.reliability, before.reliability)
        XCTAssertNoThrow(try after.validate())
    }

    func testVolumeAffectsOnlyWhenDatasetHasVolume() throws {
        let plugin = StatXrateConsistencyPlugin()
        let withVolume = try plugin.predict(Self.buyCorrectionRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let withoutVolume = try plugin.predict(Self.buyCorrectionRequest(dataHasVolume: false), hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(withVolume.classProbabilities, withoutVolume.classProbabilities)
        XCTAssertNoThrow(try withVolume.validate())
        XCTAssertNoThrow(try withoutVolume.validate())
    }

    func testResetClearsOnlineState() throws {
        let request = Self.buyCorrectionRequest(dataHasVolume: true)
        var plugin = StatXrateConsistencyPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<80 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 2.4), hyperParameters: Self.hyperParameters())
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

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatXrateConsistencyCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatXrateConsistencyAccelerated.swift").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    private static func buyCorrectionRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(crossRateSignal: -0.55, macroPressure: 0.55, volume: 1.25)
        )
    }

    private static func sellCorrectionRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(crossRateSignal: 0.55, macroPressure: -0.55, volume: 1.10)
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
            horizonMinutes: 15,
            sequenceBars: 1,
            priceCostPoints: 0.20,
            minMovePoints: 0.50,
            pointValue: 1.0,
            sampleTimeUTC: 1_800_090_000,
            dataHasVolume: dataHasVolume
        )
    }

    private static func features(crossRateSignal: Double, macroPressure: Double, volume: Double) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = -0.04 * crossRateSignal
        values[4] = 0.22
        values[6] = volume
        values[7] = 0.03 * crossRateSignal
        values[12] = crossRateSignal
        values[40] = 0.10 * volume
        values[FXDataEngineConstants.macroEventFeatureOffset + 14] = macroPressure
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
        return url.appendingPathComponent("stat_xrate_consistency")
    }
}
