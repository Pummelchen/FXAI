import FXAIPlugins
import FXDataEngine
import XCTest

final class RlPPOPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedContract() throws {
        let plugin = RlPPOPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.rlPPO.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "rl_ppo")
        XCTAssertEqual(plugin.manifest.family, .other)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.windowContext))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.nativeDistribution))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertEqual(plugin.manifest.maxSequenceBars, 96)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "rl_ppo")
        XCTAssertTrue(plan.primaryBackends.contains(.swiftScalar))
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertTrue(plan.primaryBackends.contains(.pyTorchMPS))
        XCTAssertFalse(plan.candidateBackends.contains(.coreMLNeuralEngine))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testWindowContextChangesPrediction() throws {
        let plugin = RlPPOPlugin()
        let flat = try plugin.predict(Self.request(direction: 1.0, shapedWindow: false, dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let shaped = try plugin.predict(Self.request(direction: 1.0, shapedWindow: true, dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(flat.classProbabilities, shaped.classProbabilities)
        XCTAssertNoThrow(try flat.validate())
        XCTAssertNoThrow(try shaped.validate())
    }

    func testPositiveAndNegativeContextsSeparate() throws {
        let plugin = RlPPOPlugin()
        let positive = try plugin.predict(Self.request(direction: 1.0, shapedWindow: true, dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let negative = try plugin.predict(Self.request(direction: -1.0, shapedWindow: true, dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(positive.classProbabilities, negative.classProbabilities)
        XCTAssertNoThrow(try positive.validate())
        XCTAssertNoThrow(try negative.validate())
    }

    func testTrainingUpdatesReliabilityAndMoveState() throws {
        let request = Self.request(direction: 1.0, shapedWindow: true, dataHasVolume: true)
        var plugin = RlPPOPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for iteration in 0..<80 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 2.2 + 0.01 * Double(iteration % 7)), hyperParameters: Self.hyperParameters())
        }

        let after = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertGreaterThan(after.reliability, before.reliability)
        XCTAssertGreaterThan(after.moveMeanPoints, before.moveMeanPoints)
        XCTAssertNoThrow(try after.validate())
    }

    func testSellTruthDecisionUsesLabelWhenMoveMagnitudeIsPositive() throws {
        let request = Self.request(direction: -1.0, shapedWindow: true, dataHasVolume: true)
        var plugin = RlPPOPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<90 {
            try plugin.train(
                Self.trainRequest(from: request, label: .sell, movePoints: 3.0),
                hyperParameters: Self.hyperParameters()
            )
        }

        let after = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertGreaterThan(after.classProbabilities[LabelClass.sell.rawValue], before.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(after.classProbabilities[LabelClass.sell.rawValue], after.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertNoThrow(try after.validate())
    }

    func testDirectionalMoveBelowCostResolvesToSkipTruthDecision() throws {
        let request = Self.request(
            direction: 1.0,
            shapedWindow: true,
            dataHasVolume: true,
            priceCostPoints: 1.25,
            minMovePoints: 0.50
        )
        var plugin = RlPPOPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<100 {
            try plugin.train(
                Self.trainRequest(from: request, label: .buy, movePoints: 0.40),
                hyperParameters: Self.hyperParameters()
            )
        }

        let after = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertGreaterThan(after.classProbabilities[LabelClass.skip.rawValue], before.classProbabilities[LabelClass.skip.rawValue])
        XCTAssertGreaterThan(after.classProbabilities[LabelClass.skip.rawValue], after.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(after.classProbabilities[LabelClass.skip.rawValue], after.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertNoThrow(try after.validate())
    }

    func testVolumeAffectsPredictionOnlyWhenDatasetHasVolume() throws {
        let plugin = RlPPOPlugin()
        let withVolume = try plugin.predict(Self.request(direction: 1.0, shapedWindow: true, dataHasVolume: true), hyperParameters: Self.hyperParameters())
        let withoutVolume = try plugin.predict(Self.request(direction: 1.0, shapedWindow: true, dataHasVolume: false), hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(withVolume.classProbabilities, withoutVolume.classProbabilities)
        XCTAssertNoThrow(try withVolume.validate())
        XCTAssertNoThrow(try withoutVolume.validate())
    }

    func testResetClearsState() throws {
        let request = Self.request(direction: 1.0, shapedWindow: true, dataHasVolume: true)
        var plugin = RlPPOPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        for _ in 0..<64 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 2.5), hyperParameters: Self.hyperParameters())
        }
        let trained = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertNotEqual(trained.reliability, before.reliability)
        plugin.reset()
        let reset = try plugin.predict(request, hyperParameters: Self.hyperParameters())
        XCTAssertEqual(reset.classProbabilities, before.classProbabilities)
        XCTAssertEqual(reset.moveMeanPoints, before.moveMeanPoints, accuracy: 1.0e-12)
    }

    func testConvertedPluginOwnsExpectedAccelerators() throws {
        let root = Self.pluginRoot()
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/RlPPOCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/RlPPOAccelerated.swift").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    private static func request(
        direction: Double,
        shapedWindow: Bool,
        dataHasVolume: Bool,
        priceCostPoints: Double = 0.20,
        minMovePoints: Double = 0.50
    ) -> PredictRequestV4 {
        let window = sequenceWindow(direction: direction, volume: 1.15, shaped: shapedWindow)
        return PredictRequestV4(
            valid: true,
            context: context(
                dataHasVolume: dataHasVolume,
                priceCostPoints: priceCostPoints,
                minMovePoints: minMovePoints
            ),
            windowSize: window.count,
            x: features(direction: direction, volume: 1.15),
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
            timeToHitFraction: 0.40,
            pathRisk: 0.20,
            fillRisk: 0.10,
            nextVolumeTarget: request.context.dataHasVolume ? 1.0 : 0.0,
            windowSize: request.windowSize,
            x: request.x,
            xWindow: request.xWindow
        )
    }

    private static func context(
        dataHasVolume: Bool,
        priceCostPoints: Double = 0.20,
        minMovePoints: Double = 0.50
    ) -> PluginContextV4 {
        PluginContextV4(
            sessionBucket: 2,
            horizonMinutes: 30,
            sequenceBars: 33,
            priceCostPoints: priceCostPoints,
            minMovePoints: minMovePoints,
            pointValue: 1.0,
            sampleTimeUTC: 1_800_230_000,
            dataHasVolume: dataHasVolume
        )
    }

    private static func features(direction: Double, volume: Double) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[1] = 0.14 * direction
        values[2] = 0.07 * direction
        values[3] = 0.04 * direction
        values[4] = 0.20
        values[6] = volume
        values[7] = 0.06 * direction
        values[12] = 0.11 * direction
        values[13] = 0.03 * direction
        values[14] = -0.02 * direction
        values[15] = 0.05 * direction
        values[40] = 0.12 * direction
        values[FXDataEngineConstants.macroEventFeatureOffset + 14] = 0.08 * direction
        values[FXDataEngineConstants.macroEventFeatureOffset + 19] = -0.04 * direction
        return values
    }

    private static func sequenceWindow(direction: Double, volume: Double, shaped: Bool) -> [[Double]] {
        (0..<32).map { index in
            var row = features(direction: direction, volume: volume)
            let age = Double(index)
            if shaped {
                row[1] = direction * (0.85 - 0.018 * age)
                row[2] = direction * (0.45 - 0.012 * age)
                row[3] = direction * (0.25 - 0.006 * age)
                row[6] = volume + 0.05 * sin(age)
            } else {
                row[1] = 0.04 * direction
                row[2] = 0.02 * direction
                row[3] = 0.01 * direction
                row[6] = volume
            }
            return row
        }
    }

    private static func hyperParameters() -> HyperParameters {
        HyperParameters(learningRate: 0.025, l2: 0.001)
    }

    private static func pluginRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { url.deleteLastPathComponent() }
        return url.appendingPathComponent("rl_ppo")
    }
}
