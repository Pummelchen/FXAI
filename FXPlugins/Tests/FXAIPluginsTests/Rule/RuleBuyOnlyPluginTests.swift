import FXAIPlugins
import FXDataEngine
import XCTest

final class RuleBuyOnlyPluginTests: XCTestCase {
    func testManifestMatchesLegacyRuleBuyOnlyContract() throws {
        let plugin = RuleBuyOnlyPlugin()

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.buyOnly.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "rule_buyonly")
        XCTAssertEqual(plugin.manifest.family, .ruleBased)
        XCTAssertEqual(plugin.manifest.referenceTier, .ruleBaseline)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.multiHorizon))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.selfTest))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
    }

    func testPredictMatchesLegacyBuyOnlyDistribution() throws {
        let plugin = RuleBuyOnlyPlugin()
        let request = Self.predictRequest(priceCostPoints: 0.20, minMovePoints: 0.0, dataHasVolume: true)

        let prediction = try plugin.predict(request, hyperParameters: HyperParameters())

        XCTAssertEqual(prediction.classProbabilities[LabelClass.sell.rawValue], 0.001, accuracy: 1e-12)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.buy.rawValue], 0.999, accuracy: 1e-12)
        XCTAssertEqual(prediction.classProbabilities[LabelClass.skip.rawValue], 0.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveMeanPoints, 1.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ25Points, 0.85, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ50Points, 1.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ75Points, 1.15, accuracy: 1e-12)
        XCTAssertEqual(prediction.confidence, 0.999, accuracy: 1e-12)
        XCTAssertEqual(prediction.reliability, 0.55, accuracy: 1e-12)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testPredictUsesMinimumMoveWhenProvided() throws {
        let plugin = RuleBuyOnlyPlugin()
        let request = Self.predictRequest(priceCostPoints: 0.20, minMovePoints: 2.0)

        let prediction = try plugin.predict(request, hyperParameters: HyperParameters())

        XCTAssertEqual(prediction.moveMeanPoints, 6.25, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ25Points, 5.3125, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ50Points, 6.25, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ75Points, 7.1875, accuracy: 1e-12)
    }

    func testTrainIsNoOpButStillValidatesContract() throws {
        var plugin = RuleBuyOnlyPlugin()
        let request = TrainRequestV4(
            valid: true,
            context: Self.context(dataHasVolume: true),
            labelClass: .buy,
            movePoints: 1.0,
            sampleWeight: 1.0,
            nextVolumeTarget: 12.0,
            x: Self.features
        )

        XCTAssertNoThrow(try plugin.train(request, hyperParameters: HyperParameters()))
    }

    func testRegistryExposesRuleBuyOnlyAndAccelerationPlan() {
        let pluginNames = FXAIPluginRegistry.availablePlugins().map(\.manifest.aiName)
        let plan = FXAIPluginRegistry.accelerationPlans().first { $0.pluginName == "rule_buyonly" }

        XCTAssertTrue(pluginNames.contains("rule_buyonly"))
        XCTAssertEqual(plan?.pluginName, "rule_buyonly")
        XCTAssertEqual(plan?.primaryBackends, [.swiftScalar])
        XCTAssertFalse(plan?.declaresHardwareAcceleration ?? true)
    }

    private static var features: [Double] {
        Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
    }

    private static func predictRequest(
        priceCostPoints: Double,
        minMovePoints: Double,
        dataHasVolume: Bool = false
    ) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context(priceCostPoints: priceCostPoints, minMovePoints: minMovePoints, dataHasVolume: dataHasVolume),
            x: features
        )
    }

    private static func context(
        priceCostPoints: Double = 0.20,
        minMovePoints: Double = 0.0,
        dataHasVolume: Bool = false
    ) -> PluginContextV4 {
        PluginContextV4(
            horizonMinutes: 5,
            sequenceBars: 1,
            priceCostPoints: priceCostPoints,
            minMovePoints: minMovePoints,
            pointValue: 1.0,
            sampleTimeUTC: 1_800_018_000,
            dataHasVolume: dataHasVolume
        )
    }
}
