import FXAIPlugins
import FXDataEngine
import XCTest

final class RuleRandomPluginTests: XCTestCase {
    func testManifestMatchesLegacyRuleRandomContract() throws {
        let plugin = RuleRandomPlugin()

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.randomNoSkip.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "rule_random")
        XCTAssertEqual(plugin.manifest.family, .ruleBased)
        XCTAssertEqual(plugin.manifest.referenceTier, .ruleBaseline)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.multiHorizon))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.selfTest))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
    }

    func testPredictUsesLegacyDeterministicHashBuySide() throws {
        let plugin = RuleRandomPlugin()
        let request = Self.predictRequest(sampleTimeUTC: 1_800_018_000)

        let prediction = try plugin.predict(request, hyperParameters: HyperParameters())

        XCTAssertEqual(prediction.classProbabilities[LabelClass.sell.rawValue], 0.005, accuracy: 1e-12)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.buy.rawValue], 0.995, accuracy: 1e-12)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.skip.rawValue], 0.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveMeanPoints, 1.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ25Points, 0.8075, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ50Points, 1.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ75Points, 1.1925, accuracy: 1e-12)
        XCTAssertEqual(prediction.confidence, 0.995, accuracy: 1e-12)
        XCTAssertEqual(prediction.reliability, 0.50, accuracy: 1e-12)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testPredictUsesLegacyDeterministicHashSellSide() throws {
        let plugin = RuleRandomPlugin()
        let request = Self.predictRequest(sampleTimeUTC: 0)

        let prediction = try plugin.predict(request, hyperParameters: HyperParameters())

        XCTAssertEqual(prediction.classProbabilities[LabelClass.sell.rawValue], 0.995, accuracy: 1e-12)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.buy.rawValue], 0.005, accuracy: 1e-12)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.skip.rawValue], 0.0, accuracy: 1e-12)
    }

    func testTrainIsNoOpButStillValidatesContract() throws {
        var plugin = RuleRandomPlugin()
        let request = TrainRequestV4(
            valid: true,
            context: Self.context(sampleTimeUTC: 1_800_018_000, dataHasVolume: true),
            labelClass: .buy,
            movePoints: 1.0,
            sampleWeight: 1.0,
            nextVolumeTarget: 12.0,
            x: Self.features
        )

        XCTAssertNoThrow(try plugin.train(request, hyperParameters: HyperParameters()))
    }

    func testRegistryExposesRuleRandom() {
        let pluginNames = FXAIPluginRegistry.availablePlugins().map(\.manifest.aiName)
        let plan = FXAIPluginRegistry.accelerationPlans().first { $0.pluginName == "rule_random" }

        XCTAssertTrue(pluginNames.contains("rule_random"))
        XCTAssertEqual(plan?.primaryBackends, [.swiftScalar])
        XCTAssertFalse(plan?.declaresHardwareAcceleration ?? true)
    }

    private static var features: [Double] {
        Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
    }

    private static func predictRequest(sampleTimeUTC: Int64) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(sampleTimeUTC: sampleTimeUTC, dataHasVolume: true),
            x: features
        )
    }

    private static func context(sampleTimeUTC: Int64, dataHasVolume: Bool = false) -> PluginContextV4 {
        PluginContextV4(
            horizonMinutes: 5,
            sequenceBars: 1,
            priceCostPoints: 0.20,
            minMovePoints: 0.0,
            pointValue: 1.0,
            sampleTimeUTC: sampleTimeUTC,
            dataHasVolume: dataHasVolume
        )
    }
}
