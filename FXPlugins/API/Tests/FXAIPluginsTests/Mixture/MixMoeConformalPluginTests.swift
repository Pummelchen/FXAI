import FXAIPlugins
import FXDataEngine
import XCTest

final class MixMoeConformalPluginTests: XCTestCase {
    func testManifestAndAccelerationPlanMatchConvertedMoEConformalContract() throws {
        let plugin = MixMoeConformalPlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.moeConformal.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "mix_moe_conformal")
        XCTAssertEqual(plugin.manifest.family, .mixture)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.onlineLearning))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.replay))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.pluginName, "mix_moe_conformal")
        XCTAssertTrue(plan.primaryBackends.contains(.accelerate))
        XCTAssertTrue(plan.candidateBackends.contains(.pyTorchMPS))
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
    }

    func testFreshConformalRouterProducesValidConservativeDistribution() throws {
        let plugin = MixMoeConformalPlugin()
        let prediction = try plugin.predict(Self.predictRequest(dataHasVolume: true), hyperParameters: Self.hyperParameters())

        XCTAssertEqual(prediction.classProbabilities.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.45)
        XCTAssertGreaterThanOrEqual(prediction.moveMeanPoints, 0.0)
        XCTAssertGreaterThanOrEqual(prediction.moveQ75Points, prediction.moveQ25Points)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testTrainingSeparatesBuyAndSellAndLearnsMoveHead() throws {
        var plugin = MixMoeConformalPlugin()
        let buyRequest = Self.predictRequest(dataHasVolume: true)
        let sellRequest = Self.sellRequest(from: buyRequest)

        for _ in 0..<120 {
            try plugin.train(Self.trainRequest(from: buyRequest, label: .buy, movePoints: 3.8), hyperParameters: Self.hyperParameters())
            try plugin.train(Self.trainRequest(from: sellRequest, label: .sell, movePoints: -3.6), hyperParameters: Self.hyperParameters())
        }

        let afterBuy = try plugin.predict(buyRequest, hyperParameters: Self.hyperParameters())
        let afterSell = try plugin.predict(sellRequest, hyperParameters: Self.hyperParameters())

        XCTAssertGreaterThan(afterBuy.classProbabilities[LabelClass.buy.rawValue], afterSell.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(afterSell.classProbabilities[LabelClass.sell.rawValue], afterBuy.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(afterBuy.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterSell.moveMeanPoints, 0.0)
        XCTAssertGreaterThan(afterBuy.reliability, 0.55)
        XCTAssertNoThrow(try afterBuy.validate())
        XCTAssertNoThrow(try afterSell.validate())
    }

    func testVolumeAffectsConformalRouterOnlyWhenDatasetHasVolume() throws {
        var plugin = MixMoeConformalPlugin()
        let withVolume = Self.predictRequest(dataHasVolume: true)
        let withoutVolume = Self.predictRequest(dataHasVolume: false)

        for _ in 0..<72 {
            try plugin.train(Self.trainRequest(from: withVolume, label: .buy, movePoints: 3.2), hyperParameters: Self.hyperParameters())
        }

        let predictionWithVolume = try plugin.predict(withVolume, hyperParameters: Self.hyperParameters())
        let predictionWithoutVolume = try plugin.predict(withoutVolume, hyperParameters: Self.hyperParameters())

        XCTAssertNotEqual(predictionWithVolume.classProbabilities, predictionWithoutVolume.classProbabilities)
        XCTAssertNoThrow(try predictionWithVolume.validate())
        XCTAssertNoThrow(try predictionWithoutVolume.validate())
    }

    func testResetClearsOnlineConformalState() throws {
        let request = Self.predictRequest(dataHasVolume: true)
        var plugin = MixMoeConformalPlugin()
        let before = try plugin.predict(request, hyperParameters: Self.hyperParameters())

        for _ in 0..<60 {
            try plugin.train(Self.trainRequest(from: request, label: .buy, movePoints: 3.1), hyperParameters: Self.hyperParameters())
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

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/MixMoeConformalCPUModel.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("CPU/MixMoeConformalReference.swift").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("PyTorch/mix_moe_conformal_torch.py").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Metal").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("TensorFlow").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("NLP").path))
    }

    func testReferenceConformalCutoffCoverageAndBalancedRouter() throws {
        let probabilities = [
            [0.08, 0.86, 0.06],
            [0.82, 0.10, 0.08],
            [0.15, 0.20, 0.65],
            [0.12, 0.76, 0.12]
        ]
        let labels = [1, 0, 2, 1]
        let scores = zip(probabilities, labels).map { pair in 1.0 - pair.0[pair.1] }
        let cutoff = MixMoeConformalReference.splitConformalCutoff(scores: scores, alpha: 0.10)
        let coverage = MixMoeConformalReference.empiricalCoverage(probabilities: probabilities, labels: labels, cutoff: cutoff)
        let gates = MixMoeConformalReference.routerGates(
            regime: [1.0, 0.5, -0.1],
            expertWeights: [[0.2, 0.1, 0.0], [0.2, 0.1, 0.0], [-0.1, 0.3, 0.2], [0.0, -0.2, 0.3]],
            usageEMA: [0.80, 0.10, 0.05, 0.05]
        )

        XCTAssertGreaterThanOrEqual(coverage, 0.90)
        XCTAssertEqual(cutoff, 1.0, accuracy: 1.0e-12)
        XCTAssertTrue(MixMoeConformalReference.predictionSet(probabilities: probabilities[0], cutoff: cutoff).classes.contains(1))
        XCTAssertEqual(gates.reduce(0.0, +), 1.0, accuracy: 1.0e-12)
        XCTAssertLessThan(gates[0], gates[1])
    }

    func testReferenceSplitConformalCutoffUsesFiniteSampleRank() throws {
        let scores = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        let cutoff = MixMoeConformalReference.splitConformalCutoff(scores: scores, alpha: 0.10)
        let conservativeCutoff = MixMoeConformalReference.splitConformalCutoff(
            scores: [0.10, 0.20, 0.30, 0.40],
            alpha: 0.10
        )
        let narrowSet = MixMoeConformalReference.predictionSet(probabilities: [0.08, 0.70, 0.22], cutoff: 0.35)
        let fallbackSet = MixMoeConformalReference.predictionSet(probabilities: [0.15, 0.45, 0.40], cutoff: 0.20)

        XCTAssertEqual(cutoff, 0.80, accuracy: 1.0e-12)
        XCTAssertEqual(conservativeCutoff, 1.0, accuracy: 1.0e-12)
        XCTAssertEqual(narrowSet.classes, Set([LabelClass.buy.rawValue]))
        XCTAssertEqual(fallbackSet.classes, Set([LabelClass.buy.rawValue]))
    }

    func testReferenceRouterSpecializesExpertsByRegimeAndPenalizesOveruse() throws {
        let expertWeights = [
            [0.0, 1.4, -0.2],
            [0.0, -1.4, -0.2],
            [0.0, 0.2, 1.3],
            [0.0, -0.2, -1.3]
        ]
        let uniformUsage = Array(repeating: 0.25, count: 4)
        let positiveTrend = MixMoeConformalReference.routerGates(
            regime: [1.0, 1.0, 0.0],
            expertWeights: expertWeights,
            usageEMA: uniformUsage,
            loadBalanceStrength: 0.0
        )
        let negativeTrend = MixMoeConformalReference.routerGates(
            regime: [1.0, -1.0, 0.0],
            expertWeights: expertWeights,
            usageEMA: uniformUsage,
            loadBalanceStrength: 0.0
        )
        let overusedPositive = MixMoeConformalReference.routerGates(
            regime: [1.0, 1.0, 0.0],
            expertWeights: expertWeights,
            usageEMA: [0.95, 0.02, 0.02, 0.01],
            loadBalanceStrength: 0.70
        )

        XCTAssertGreaterThan(positiveTrend[0], positiveTrend[1])
        XCTAssertGreaterThan(negativeTrend[1], negativeTrend[0])
        XCTAssertLessThan(overusedPositive[0], positiveTrend[0])
        XCTAssertEqual(positiveTrend.reduce(0.0, +), 1.0, accuracy: 1.0e-12)
        XCTAssertEqual(negativeTrend.reduce(0.0, +), 1.0, accuracy: 1.0e-12)
        XCTAssertEqual(overusedPositive.reduce(0.0, +), 1.0, accuracy: 1.0e-12)
    }

    private static func predictRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: PluginContextV4(
                sessionBucket: 2,
                horizonMinutes: 5,
                sequenceBars: 1,
                priceCostPoints: 0.22,
                minMovePoints: 0.50,
                pointValue: 1.0,
                sampleTimeUTC: 1_800_023_000,
                dataHasVolume: dataHasVolume
            ),
            x: features()
        )
    }

    private static func sellRequest(from request: PredictRequestV4) -> PredictRequestV4 {
        var features = request.x
        features[0] = -features[0]
        features[1] = -features[1]
        features[2] = -features[2]
        features[3] = -features[3]
        features[5] = -features[5]
        return PredictRequestV4(valid: true, context: request.context, x: features)
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
            timeToHitFraction: 0.42,
            pathRisk: 0.20,
            fillRisk: 0.12,
            nextVolumeTarget: request.context.dataHasVolume ? 1.0 : 0.0,
            x: request.x
        )
    }

    private static func features() -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[0] = 0.11
        values[1] = 0.08
        values[2] = 0.05
        values[3] = 0.03
        values[4] = 1.10
        values[5] = 0.10
        values[6] = 0.90
        values[7] = 0.06
        values[8] = -0.02
        values[9] = 0.04
        values[12] = 0.07
        values[80] = 0.85
        values[81] = 0.40
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
        return url.appendingPathComponent("mix_moe_conformal")
    }
}
