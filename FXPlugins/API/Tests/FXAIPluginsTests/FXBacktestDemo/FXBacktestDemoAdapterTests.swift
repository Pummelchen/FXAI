import FXAIPlugins
import FXDataEngine
import XCTest

final class FXBacktestDemoAdapterTests: XCTestCase {
    func testMovingAverageCrossManifestAndAccelerationPlan() throws {
        let plugin = MovingAverageCrossFXDataEnginePlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.demoMovingAverageCross.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "fxbacktest_moving_average_cross")
        XCTAssertEqual(plugin.manifest.family, .ruleBased)
        XCTAssertEqual(plugin.manifest.referenceTier, .ruleBaseline)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.multiHorizon))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.selfTest))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.primaryBackends, [.swiftScalar, .metal])
        XCTAssertEqual(plan.candidateBackends, [.swiftSIMD])
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
        XCTAssertTrue(plan.declaresHardwareAcceleration)
        let metalFile = Self.pluginRoot("fxbacktest_moving_average_cross")
            .appendingPathComponent("Metal/MovingAverageCrossMetal.swift")
        XCTAssertTrue(FileManager.default.fileExists(atPath: metalFile.path))
    }

    func testMovingAverageCrossPredictsBuyAndSellFromFastSlowReturns() throws {
        let plugin = MovingAverageCrossFXDataEnginePlugin()

        let buyPrediction = try plugin.predict(
            Self.request(features: Self.features(volumeSignal: 0.70, fastReturn: 0.12, slowReturn: 0.02)),
            hyperParameters: HyperParameters()
        )
        let sellPrediction = try plugin.predict(
            Self.request(features: Self.features(volumeSignal: 0.70, fastReturn: -0.03, slowReturn: 0.08)),
            hyperParameters: HyperParameters()
        )

        XCTAssertGreaterThan(buyPrediction.classProbabilities[LabelClass.buy.rawValue], buyPrediction.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(sellPrediction.classProbabilities[LabelClass.sell.rawValue], sellPrediction.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(buyPrediction.moveMeanPoints, 1.0)
        XCTAssertGreaterThan(sellPrediction.moveMeanPoints, 1.0)
        XCTAssertNoThrow(try buyPrediction.validate())
        XCTAssertNoThrow(try sellPrediction.validate())
    }

    func testMovingAverageCrossSkipsFlatFastSlowSignal() throws {
        let plugin = MovingAverageCrossFXDataEnginePlugin()

        let prediction = try plugin.predict(
            Self.request(features: Self.features(volumeSignal: 0.90, fastReturn: 0.03, slowReturn: 0.03)),
            hyperParameters: HyperParameters()
        )

        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.skip.rawValue], 0.80)
        XCTAssertEqual(prediction.moveMeanPoints, 0.0, accuracy: 1e-12)
        XCTAssertNoThrow(try prediction.validate())
    }

    func testFXStupidManifestAndAccelerationPlan() throws {
        let plugin = FXStupidFXDataEnginePlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.demoFXStupid.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "fxbacktest_fxstupid")
        XCTAssertEqual(plugin.manifest.family, .ruleBased)
        XCTAssertEqual(plugin.manifest.referenceTier, .ruleBaseline)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.multiHorizon))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.selfTest))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.primaryBackends, [.swiftScalar])
        XCTAssertTrue(plan.candidateBackends.isEmpty)
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
        XCTAssertFalse(plan.declaresHardwareAcceleration)
    }

    func testFXStupidPredictsBuySellAndSkipFromFeatureEdge() throws {
        let plugin = FXStupidFXDataEnginePlugin()

        let buyPrediction = try plugin.predict(
            Self.request(features: Self.features(shortReturn: 0.07, slope: 0.03, volumeSignal: 0.60)),
            hyperParameters: HyperParameters()
        )
        let sellPrediction = try plugin.predict(
            Self.request(features: Self.features(shortReturn: -0.07, slope: -0.03, volumeSignal: -0.60)),
            hyperParameters: HyperParameters()
        )
        let skipPrediction = try plugin.predict(
            Self.request(features: Self.features(shortReturn: 0.0, slope: 0.0, volumeSignal: 0.0)),
            hyperParameters: HyperParameters()
        )

        XCTAssertGreaterThan(buyPrediction.classProbabilities[LabelClass.buy.rawValue], buyPrediction.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(sellPrediction.classProbabilities[LabelClass.sell.rawValue], sellPrediction.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(skipPrediction.classProbabilities[LabelClass.skip.rawValue], 0.80)
        XCTAssertGreaterThan(buyPrediction.reliability, 0.52)
        XCTAssertGreaterThan(sellPrediction.reliability, 0.52)
        XCTAssertNoThrow(try buyPrediction.validate())
        XCTAssertNoThrow(try sellPrediction.validate())
        XCTAssertNoThrow(try skipPrediction.validate())
    }

    func testFX7ManifestAndAccelerationPlan() throws {
        let plugin = FX7FXDataEnginePlugin()
        let plan = plugin.accelerationPlan

        XCTAssertEqual(plugin.manifest.aiID, AIModelID.demoFX7.rawValue)
        XCTAssertEqual(plugin.manifest.aiName, "fx7")
        XCTAssertEqual(plugin.manifest.family, .ruleBased)
        XCTAssertEqual(plugin.manifest.referenceTier, .ruleBaseline)
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.windowContext))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.multiHorizon))
        XCTAssertTrue(plugin.manifest.capabilityMask.contains(.selfTest))
        XCTAssertTrue(plugin.manifest.requiresVolumeWhenAvailable)
        XCTAssertNoThrow(try plugin.manifest.validate())
        XCTAssertTrue(plugin.selfTest())
        XCTAssertEqual(plan.primaryBackends, [.swiftScalar, .metal])
        XCTAssertEqual(plan.candidateBackends, [.swiftSIMD, .accelerate])
        XCTAssertTrue(plan.usesVolumeWhenAvailable)
        XCTAssertTrue(plan.declaresHardwareAcceleration)
        let metalFile = Self.pluginRoot("fx7")
            .appendingPathComponent("Metal/FX7Metal.swift")
        XCTAssertTrue(FileManager.default.fileExists(atPath: metalFile.path))
    }

    func testFX7PredictsDirectionalAndSkipSignals() throws {
        let plugin = FX7FXDataEnginePlugin()

        let buyPrediction = try plugin.predict(
            Self.request(features: Self.features(
                shortReturn: 0.08,
                slope: 0.05,
                volumeSignal: 0.60,
                fastReturn: 0.14,
                slowReturn: 0.03
            )),
            hyperParameters: HyperParameters()
        )
        let sellPrediction = try plugin.predict(
            Self.request(features: Self.features(
                shortReturn: -0.08,
                slope: -0.05,
                volumeSignal: -0.60,
                fastReturn: -0.14,
                slowReturn: -0.03
            )),
            hyperParameters: HyperParameters()
        )
        let skipPrediction = try plugin.predict(
            Self.request(features: Self.features()),
            hyperParameters: HyperParameters()
        )

        XCTAssertGreaterThan(buyPrediction.classProbabilities[LabelClass.buy.rawValue], buyPrediction.classProbabilities[LabelClass.sell.rawValue])
        XCTAssertGreaterThan(sellPrediction.classProbabilities[LabelClass.sell.rawValue], sellPrediction.classProbabilities[LabelClass.buy.rawValue])
        XCTAssertGreaterThan(skipPrediction.classProbabilities[LabelClass.skip.rawValue], 0.80)
        XCTAssertGreaterThan(buyPrediction.moveMeanPoints, 1.0)
        XCTAssertGreaterThan(sellPrediction.moveMeanPoints, 1.0)
        XCTAssertNoThrow(try buyPrediction.validate())
        XCTAssertNoThrow(try sellPrediction.validate())
        XCTAssertNoThrow(try skipPrediction.validate())
    }

    func testRegistryExposesConvertedRuleAndFXBacktestDemoPlugins() {
        let pluginNames = Set(FXAIPluginRegistry.availablePlugins().map(\.manifest.aiName))
        let planNames = Set(FXAIPluginRegistry.accelerationPlans().map(\.pluginName))

        XCTAssertEqual(pluginNames.count, FXDataEngineConstants.aiCount)
        XCTAssertEqual(planNames.count, FXDataEngineConstants.aiCount)
        XCTAssertTrue(pluginNames.contains("rule_buyonly"))
        XCTAssertTrue(pluginNames.contains("rule_sellonly"))
        XCTAssertTrue(pluginNames.contains("rule_random"))
        XCTAssertTrue(pluginNames.contains("rule_m1sync"))
        XCTAssertTrue(pluginNames.contains("fxbacktest_moving_average_cross"))
        XCTAssertTrue(pluginNames.contains("fxbacktest_fxstupid"))
        XCTAssertTrue(pluginNames.contains("fx7"))
    }

    private static func request(features: [Double], dataHasVolume: Bool = true) -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: PluginContextV4(
                horizonMinutes: 5,
                sequenceBars: 1,
                priceCostPoints: 0.20,
                minMovePoints: 0.50,
                pointValue: 1.0,
                sampleTimeUTC: 1_800_018_000,
                dataHasVolume: dataHasVolume
            ),
            x: features
        )
    }

    private static func features(
        shortReturn: Double = 0.0,
        slope: Double = 0.0,
        volumeSignal: Double = 0.0,
        fastReturn: Double = 0.0,
        slowReturn: Double = 0.0
    ) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[0] = shortReturn
        values[3] = slope
        values[6] = volumeSignal
        values[7] = fastReturn
        values[8] = slowReturn
        return values
    }

    private static func pluginRoot(_ name: String) -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent(name)
    }
}
