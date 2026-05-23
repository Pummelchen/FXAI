import FXAIPlugins
import FXDataEngine
import XCTest

final class StatEMDHHTPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedContract() throws {
        let plugin = StatEMDHHTPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.statEMDHHT.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "stat_emd_hht")
        XCTAssertEqual(plugin.manifest.family, .stateSpace)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.windowContext))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.nativeDistribution))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertEqual(plugin.manifest.maxSequenceBars, 64)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "stat_emd_hht")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertTrue(plan.primaryBackends.contains(.swiftSIMD))
        XCTAssertFalse(plan.primaryBackends.contains(.metal))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testEMDHHTSeparatesPositiveAndNegativeModeShifts() throws {
        let plugin = StatEMDHHTPlugin()
        let positive = try plugin.predict(Self.momentumRequest(direction: 1.0, dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let negative = try plugin.predict(Self.momentumRequest(direction: -1.0, dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(positive.classProbabilities[LabelClass.buy.rawValue], negative.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(negative.classProbabilities[LabelClass.sell.rawValue], positive.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertNoThrow(try positive.validate())
        XCTAssertNoThrow(try negative.validate())
    }

    func testModeShiftChangesPredictionVersusFlatWindow() throws {
        let plugin = StatEMDHHTPlugin()
        let trending = try plugin.predict(Self.momentumRequest(direction: 1.0, dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let flat = try plugin.predict(Self.flatRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(trending.classProbabilities[LabelClass.buy.rawValue], flat.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertNoThrow(try trending.validate())
        XCTAssertNoThrow(try flat.validate())
    }

    func testTrainingUpdatesLinearBaseAndReliabilityState() throws {
        let request = Self.momentumRequest(direction: 1.0, dataHasVolume: true)
        var plugin = StatEMDHHTPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for iteration in 0..<96 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 2.1 + 0.01 * Double(iteration % 4)), hyperParameters: Self.hyperParameters())
        }

        let after = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertGreaterThan(after.moveMeanPoints, before.moveMeanPoints)
        XCTAssertGreaterThan(after.reliability, before.reliability)
        XCTAssertNoThrow(try after.validate())
    }

    func testVolumeAffectsEMDHHTOnlyWhenDatasetHasVolume() throws {
        let plugin = StatEMDHHTPlugin()
        let withVolume = try plugin.predict(Self.momentumRequest(direction: 1.0, dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let withoutVolume = try plugin.predict(Self.momentumRequest(direction: 1.0, dataHasVolume: false), hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(withVolume.classProbabilities, withoutVolume.classProbabilities)
        XCTAssertNoThrow(try withVolume.validate())
        XCTAssertNoThrow(try withoutVolume.validate())
    }

    func testResetClearsEMDHHTState() throws {
        let request = Self.momentumRequest(direction: 1.0, dataHasVolume: true)
        var plugin = StatEMDHHTPlugin()
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

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatEMDHHTCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/StatEMDHHTAccelerated.swift").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    private static func momentumRequest(direction: Double, dataHasVolume: Bool) -> PredictRequestV4 {
        let window = trendWindow(direction: direction, volume: 1.20)
        return PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            windowSize: window.count,
            x: features(direction: direction, volume: 1.20),
            xWindow: window
        )
    }

    private static func flatRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        let window = trendWindow(direction: 0.0, volume: 1.20)
        return PredictRequestV4(
            valid: true,
            context: context(dataHasVolume: dataHasVolume),
            windowSize: window.count,
            x: features(direction: 0.0, volume: 1.20),
            xWindow: window
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
            windowSize: request.windowSize,
            x: request.x,
            xWindow: request.xWindow
        )
    }

    private static func context(dataHasVolume: Bool) -> PluginContextV4 {
        PluginContextV4(
            sessionBucket: 2,
            horizonMinutes: 30,
            sequenceBars: 33,
            priceCostPoints: 0.20,
            minMovePoints: 0.50,
            pointValue: 1.0,
            sampleTimeUTC: 1_800_130_000,
            dataHasVolume: dataHasVolume
        )
    }

    private static func features(direction: Double, volume: Double) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = 0.12 * direction
        values[2] = 0.05 * direction
        values[3] = 0.08 * direction
        values[4] = 0.10
        values[6] = volume
        values[7] = 0.06 * direction
        values[12] = 0.05 * direction
        values[40] = 0.15 * direction
        return values
    }

    private static func trendWindow(direction: Double, volume: Double) -> [[Double]] {
        (0..<32).map { index in
            var row = features(direction: direction, volume: volume)
            let age = Double(index)
            row[1] = direction == 0.0 ? 0.0 : direction * (0.90 - 0.025 * age)
            row[2] = direction == 0.0 ? 0.0 : direction * (0.45 - 0.010 * age)
            row[6] = volume
            row[40] = 0.12 * direction
            return row
        }
    }

    private static func hyperParameters() -> HyperParameters {
        HyperParameters(learningRate: 0.03, l2: 0.001)
    }

    private static func pluginRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("stat_emd_hht")
    }
}
