import FXAIPlugins
import FXDataEngine
import XCTest

final class MemRetrdiffPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedRetrdiffContract() throws {
        let plugin = MemRetrdiffPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.retrDiff.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "mem_retrdiff")
        XCTAssertEqual(plugin.manifest.family, .retrieval)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.replay))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.windowContext))
        XCTAssertEqual(plugin.manifest.minSequenceBars, 8)
        XCTAssertEqual(plugin.manifest.maxSequenceBars, 64)
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "mem_retrdiff")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertTrue(plan.candidateBackends.contains(.metal))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testColdMemoryFailsClosedUntilEnoughExamplesExist() throws {
        let plugin = MemRetrdiffPlugin()
        let prediction = try plugin.predict(Self.predictRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities[LabelClass.sell.rawValue], 0.10, accuracy: 1.0e-9)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.buy.rawValue], 0.10, accuracy: 1.0e-9)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.skip.rawValue], 0.80, accuracy: 1.0e-9)
        XCTAssertEqual(prediction.moveMeanPoints, 0.0, accuracy: 1.0e-12)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testTrainingBuildsRetrievalMemoryAndSeparatesDirections() throws {
        var plugin = MemRetrdiffPlugin()
        let buyBase = Self.predictRequest(dataHasVolume: true)
        let sellBase = Self.sellRequest(from: buyBase)

        for index in 0..<18 {
            let buy = Self.shiftedRequest(buyBase, shift: Double(index) * 0.22)
            let sell = Self.shiftedRequest(sellBase, shift: -Double(index) * 0.22)
            try plugin.train(Self.trainRequest(from: buy, label: .buy, movePoints: 3.4 + 0.03 * Double(index)), hyperParameters: Self.hyperParameters())
            try plugin.train(Self.trainRequest(from: sell, label: .sell, movePoints: -3.1 - 0.02 * Double(index)), hyperParameters: Self.hyperParameters())
        }

        let afterBuy = try plugin.predict(buyBase, hyperParameters: Self.hyperParameters())
        let afterSell = try plugin.predict(sellBase, hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(afterBuy.classProbabilities[LabelClass.buy.rawValue], afterSell.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(afterSell.classProbabilities[LabelClass.sell.rawValue], afterBuy.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(afterBuy.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterSell.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterBuy.reliability, 0.0)
        XCTAssertGreaterThan(afterSell.reliability, 0.0)
        XCTAssertNoThrow(try afterBuy.validate())
        XCTAssertNoThrow(try afterSell.validate())
    }

    func testVolumeAffectsRetrievalEmbeddingOnlyWhenDatasetHasVolume() throws {
        var plugin = MemRetrdiffPlugin()
        let withVolume = Self.predictRequest(dataHasVolume: true)
        let withoutVolume = Self.predictRequest(dataHasVolume: false)
        let lowVolumeSell = Self.sellRequest(from: withVolume)

        for index in 0..<14 {
            let shifted = Self.shiftedRequest(withVolume, shift: Double(index) * 0.20)
            try plugin.train(Self.trainRequest(from: shifted, label: .buy, movePoints: 2.8 + 0.02 * Double(index)), hyperParameters: Self.hyperParameters())
            var sellFeatures = Self.shiftedRequest(lowVolumeSell, shift: Double(index) * 0.20)
            sellFeatures = Self.withVolume(sellFeatures, volume: 0.05)
            try plugin.train(Self.trainRequest(from: sellFeatures, label: .sell, movePoints: -2.6 - 0.02 * Double(index)), hyperParameters: Self.hyperParameters())
        }

        let predictionWithVolume = try plugin.predict(withVolume, hyperParameters: Self.hyperParameters())
        let predictionWithoutVolume = try plugin.predict(withoutVolume, hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(predictionWithVolume.classProbabilities, predictionWithoutVolume.classProbabilities)
        XCTAssertNoThrow(try predictionWithVolume.validate())
        XCTAssertNoThrow(try predictionWithoutVolume.validate())
    }

    func testResetClearsRetrievalMemory() throws {
        let request = Self.predictRequest(dataHasVolume: true)
        var plugin = MemRetrdiffPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for index in 0..<16 {
            let shifted = Self.shiftedRequest(request, shift: Double(index) * 0.22)
            try plugin.train(Self.trainRequest(from: shifted, label: .buy, movePoints: 3.0), hyperParameters: Self.hyperParameters())
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

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/MemRetrdiffCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal/MemRetrdiffMetal.swift").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    private static func predictRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        let context = PluginContextV4(
            sessionBucket: 2,
            horizonMinutes: 5,
            sequenceBars: 16,
            priceCostPoints: 0.18,
            minMovePoints: 0.45,
            pointValue: 1.0,
            sampleTimeUTC: 1_800_020_000,
            dataHasVolume: dataHasVolume
        )
        let features = features()
        let window = window(seed: features)
        return PredictRequestV4(valid: true, context: context, windowSize: window.count, x: features, xWindow: window)
    }

    private static func sellRequest(from request: PredictRequestV4) -> PredictRequestV4 {
        var features = request.x
        features[0] = -features[0]
        features[1] = -features[1]
        features[2] = -features[2]
        features[3] = -features[3]
        features[7] = -features[7]
        let window = Self.window(seed: features)
        return PredictRequestV4(valid: true, context: request.context, windowSize: window.count, x: features, xWindow: window)
    }

    private static func shiftedRequest(_ request: PredictRequestV4, shift: Double) -> PredictRequestV4 {
        var features = request.x
        let phase = shift
        features[0] += 4.50 * sin(phase)
        features[1] += 3.75 * cos(0.70 * phase)
        features[2] += 3.25 * sin(1.30 * phase)
        features[3] += 2.80 * cos(1.70 * phase)
        features[4] += abs(1.40 * sin(0.50 * phase))
        features[5] += abs(1.10 * cos(0.90 * phase))
        features[7] += 2.65 * sin(2.10 * phase)
        features[12] += 2.35 * cos(1.10 * phase)
        let window = Self.window(seed: features)
        return PredictRequestV4(valid: true, context: request.context, windowSize: window.count, x: features, xWindow: window)
    }

    private static func withVolume(_ request: PredictRequestV4, volume: Double) -> PredictRequestV4 {
        var features = request.x
        features[6] = volume
        features[80] = volume
        features[81] = 0.5 * volume
        let window = Self.window(seed: features)
        return PredictRequestV4(valid: true, context: request.context, windowSize: window.count, x: features, xWindow: window)
    }

    private static func trainRequest(from request: PredictRequestV4, label: LabelClass, movePoints: Double) -> TrainRequestV4 {
        TrainRequestV4(
            valid: true,
            context: request.context,
            labelClass: label,
            movePoints: movePoints,
            sampleWeight: 1.0,
            mfePoints: abs(movePoints) * 1.20,
            maePoints: abs(movePoints) * 0.40,
            timeToHitFraction: 0.45,
            pathRisk: 0.20,
            fillRisk: 0.12,
            nextVolumeTarget: request.context.dataHasVolume ? 1.0 : 0.0,
            windowSize: request.xWindow.count,
            x: request.x,
            xWindow: request.xWindow
        )
    }

    private static func features() -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[0] = 0.08
        values[1] = 0.10
        values[2] = 0.06
        values[3] = 0.05
        values[4] = 0.22
        values[5] = 0.16
        values[6] = 1.15
        values[7] = 0.07
        values[9] = 0.10
        values[12] = 0.05
        values[80] = 0.75
        values[81] = 0.40
        return values
    }

    private static func window(seed: [Double]) -> [[Double]] {
        (0..<15).map { row in
            var values = seed
            let t = Double(row)
            values[1] = seed[1] - 0.006 * t
            values[2] = seed[2] - 0.004 * t
            values[3] = seed[3] - 0.003 * t
            values[4] = seed[4] + 0.002 * t
            values[6] = max(0.0, seed[6] - 0.010 * t)
            return values
        }
    }

    private static func hyperParameters() -> HyperParameters {
        HyperParameters(learningRate: 0.02, l2: 0.001)
    }

    private static func pluginRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("mem_retrdiff")
    }
}
