import FXAIPlugins
import FXDataEngine
import XCTest

final class FactorCMVPanelPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedContract() throws {
        let plugin = FactorCMVPanelPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.factorCMVPanel.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "factor_cmv_panel")
        XCTAssertEqual(plugin.manifest.family, .other)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.nativeDistribution))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "factor_cmv_panel")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertFalse(plan.primaryBackends.contains(.metal))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testCMVPanelFactorSeparatesPositiveAndNegativePanelStates() throws {
        let plugin = FactorCMVPanelPlugin()
        let positive = try plugin.predict(Self.positivePanelRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let negative = try plugin.predict(Self.negativePanelRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(positive.classProbabilities[LabelClass.buy.rawValue], negative.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(negative.classProbabilities[LabelClass.sell.rawValue], positive.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertNoThrow(try positive.validate())
        XCTAssertNoThrow(try negative.validate())
    }

    func testTrainingUpdatesLinearBaseAndReliabilityState() throws {
        let request = Self.positivePanelRequest(dataHasVolume: true)
        var plugin = FactorCMVPanelPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for iteration in 0..<96 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 2.1 + 0.01 * Double(iteration % 4)), hyperParameters: Self.hyperParameters())
        }

        let after = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertGreaterThan(after.moveMeanPoints, before.moveMeanPoints)
        XCTAssertGreaterThan(after.reliability, before.reliability)
        XCTAssertNoThrow(try after.validate())
    }

    func testVolumeAffectsCMVPanelOnlyWhenDatasetHasVolume() throws {
        let plugin = FactorCMVPanelPlugin()
        let withVolume = try plugin.predict(Self.positivePanelRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let withoutVolume = try plugin.predict(Self.positivePanelRequest(dataHasVolume: false), hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(withVolume.classProbabilities, withoutVolume.classProbabilities)
        XCTAssertNoThrow(try withVolume.validate())
        XCTAssertNoThrow(try withoutVolume.validate())
    }

    func testResetClearsCMVPanelState() throws {
        let request = Self.positivePanelRequest(dataHasVolume: true)
        var plugin = FactorCMVPanelPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<80 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 2.3), hyperParameters: Self.hyperParameters())
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

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/FactorCMVPanelCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/FactorCMVPanelAccelerated.swift").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    private static func positivePanelRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(panelPressure: 0.75, valueGap: -0.35, direction: 1.0, volume: 1.20)
        )
    }

    private static func negativePanelRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            x: features(panelPressure: -0.75, valueGap: 0.35, direction: -1.0, volume: 1.10)
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
            horizonMinutes: 30,
            sequenceBars: 1,
            priceCostPoints: 0.20,
            minMovePoints: 0.50,
            pointValue: 1.0,
            sampleTimeUTC: 1_800_130_000,
            dataHasVolume: dataHasVolume
        )
    }

    private static func features(panelPressure: Double, valueGap: Double, direction: Double, volume: Double) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = 0.18 * direction
        values[2] = 0.05 * direction
        values[3] = 0.06 * direction
        values[4] = valueGap
        values[6] = volume
        values[7] = 0.04 * direction
        values[12] = 0.08 * direction
        values[40] = 0.0
        values[FXDataEngineConstants.macroEventFeatureOffset + 14] = panelPressure
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
        return url.appendingPathComponent("factor_cmv_panel")
    }
}
