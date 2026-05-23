import FXAIPlugins
import FXDataEngine
import XCTest

final class StatMicroflowProxyPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedMicroflowContract() throws {
        let plugin = StatMicroflowProxyPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.statMicroflowProxy.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "stat_microflow_proxy")
        XCTAssertEqual(plugin.manifest.family, .stateSpace)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.nativeDistribution))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "stat_microflow_proxy")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertFalse(plan.primaryBackends.contains(.metal))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testFreshMicroflowModelProducesValidDistribution() throws {
        let plugin = StatMicroflowProxyPlugin()
        let prediction = try plugin.predict(Self.buyFlowRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.10)
        XCTAssertGreaterThanOrEqual(prediction.moveQ75Points, prediction.moveQ50Points)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testMicroflowFeaturesSeparateBuyAndSellDirections() throws {
        let plugin = StatMicroflowProxyPlugin()
        let buyPrediction = try plugin.predict(Self.buyFlowRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let sellPrediction = try plugin.predict(Self.sellFlowRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(buyPrediction.classProbabilities[LabelClass.buy.rawValue], sellPrediction.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(sellPrediction.classProbabilities[LabelClass.sell.rawValue], buyPrediction.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertNoThrow(try buyPrediction.validate())
        XCTAssertNoThrow(try sellPrediction.validate())
    }

    func testTrainingUpdatesMoveAndReliabilityState() throws {
        let request = Self.buyFlowRequest(dataHasVolume: true)
        var plugin = StatMicroflowProxyPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for iteration in 0..<96 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 2.3 + 0.01 * Double(iteration % 5)), hyperParameters: Self.hyperParameters())
        }

        let after = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertGreaterThan(after.moveMeanPoints, before.moveMeanPoints)
        XCTAssertGreaterThan(after.reliability, before.reliability)
        XCTAssertNoThrow(try after.validate())
    }

    func testVolumeAffectsMicroflowOnlyWhenDatasetHasVolume() throws {
        let withVolume = Self.buyFlowRequest(dataHasVolume: true)
        let withoutVolume = Self.buyFlowRequest(dataHasVolume: false)
        let plugin = StatMicroflowProxyPlugin()

        let predictionWithVolume = try plugin.predict(withVolume, hyperParameters: Self.hyperParameters())
        let predictionWithoutVolume = try plugin.predict(withoutVolume, hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(predictionWithVolume.classProbabilities, predictionWithoutVolume.classProbabilities)
        XCTAssertNoThrow(try predictionWithVolume.validate())
        XCTAssertNoThrow(try predictionWithoutVolume.validate())
    }

    func testResetClearsMicroflowState() throws {
        let request = Self.buyFlowRequest(dataHasVolume: true)
        var plugin = StatMicroflowProxyPlugin()
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

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatMicroflowProxyCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatMicroflowProxyAccelerated.swift").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    private static func buyFlowRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(flow: 0.92, fastReturn: 0.04, volume: 1.35)
        )
    }

    private static func sellFlowRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(flow: -0.88, fastReturn: -0.04, volume: 1.15)
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
            sampleTimeUTC: 1_800_080_000,
            dataHasVolume: dataHasVolume
        )
    }

    private static func features(flow: Double, fastReturn: Double, volume: Double) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = 0.07 * flow
        values[2] = 0.04 * flow
        values[3] = 0.05 * flow
        values[4] = 1.02
        values[6] = volume
        values[7] = fastReturn
        values[12] = flow
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
        return url.appendingPathComponent("stat_microflow_proxy")
    }
}
