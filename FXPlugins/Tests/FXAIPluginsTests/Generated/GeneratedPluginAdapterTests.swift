import FXAIPlugins
import FXDataEngine
import XCTest

final class GeneratedPluginAdapterTests: XCTestCase {
    func testGeneratedDefinitionsCoverEveryRemainingLegacyPlugin() throws {
        let definitions = FXAIGeneratedPluginDefinition.all
        let generatedIDs = Set(definitions.map(\.aiID))
        let handwrittenIDs: Set<AIModelID> = [
            .m1Sync,
            .buyOnly,
            .sellOnly,
            .randomNoSkip,
            .demoMovingAverageCross,
            .demoFXStupid
        ]

        XCTAssertEqual(definitions.count, FXDataEngineConstants.aiCount - handwrittenIDs.count)
        XCTAssertEqual(generatedIDs.count, definitions.count)
        XCTAssertTrue(generatedIDs.isDisjoint(with: handwrittenIDs))

        for modelID in AIModelID.allCases where !handwrittenIDs.contains(modelID) {
            XCTAssertTrue(generatedIDs.contains(modelID), "missing generated adapter for \(modelID)")
        }
    }

    func testRegistryExposesAllSwiftEraPluginsAndPlans() throws {
        let plugins = FXAIPluginRegistry.availablePlugins()
        let plans = FXAIPluginRegistry.accelerationPlans()
        let ids = Set(plugins.map { $0.manifest.aiID })
        let names = Set(plugins.map { $0.manifest.aiName })

        XCTAssertEqual(plugins.count, FXDataEngineConstants.aiCount)
        XCTAssertEqual(ids.count, FXDataEngineConstants.aiCount)
        XCTAssertEqual(plans.count, FXDataEngineConstants.aiCount)
        XCTAssertEqual(Set(plans.map(\.pluginName)).count, FXDataEngineConstants.aiCount)
        XCTAssertTrue(names.contains("fxbacktest_moving_average_cross"))
        XCTAssertTrue(names.contains("fxbacktest_fxstupid"))
        XCTAssertTrue(names.contains("ai_mythos_rdt"))
        XCTAssertTrue(names.contains("tree_xgb_fast"))
    }

    func testEveryGeneratedAdapterValidatesSelfTestsAndPredicts() throws {
        let request = Self.predictRequest(dataHasVolume: true)

        for adapter in FXAIGeneratedPluginAdapter.generatedPlugins() {
            XCTAssertNoThrow(try adapter.manifest.validate(), adapter.manifest.aiName)
            XCTAssertTrue(adapter.selfTest(), adapter.manifest.aiName)
            XCTAssertTrue(adapter.manifest.requiresVolumeWhenAvailable, adapter.manifest.aiName)
            XCTAssertTrue(adapter.accelerationPlan.usesVolumeWhenAvailable, adapter.manifest.aiName)
            XCTAssertFalse(adapter.accelerationPlan.primaryBackends.isEmpty, adapter.manifest.aiName)

            let prediction = try adapter.predict(request, hyperParameters: HyperParameters())
            XCTAssertNoThrow(try prediction.validate(), adapter.manifest.aiName)
        }
    }

    func testGeneratedAdaptersUseVolumeWhenAvailable() throws {
        let withVolume = Self.predictRequest(dataHasVolume: true)
        let withoutVolume = Self.predictRequest(dataHasVolume: false)

        for adapter in FXAIGeneratedPluginAdapter.generatedPlugins() {
            let predictionWithVolume = try adapter.predict(withVolume, hyperParameters: HyperParameters())
            let predictionWithoutVolume = try adapter.predict(withoutVolume, hyperParameters: HyperParameters())

            XCTAssertNotEqual(
                predictionWithVolume.classProbabilities,
                predictionWithoutVolume.classProbabilities,
                "volume did not affect \(adapter.manifest.aiName)"
            )
        }
    }

    func testAccelerationPlansClassifyAppleSiliconBackends() throws {
        let plans = Dictionary(uniqueKeysWithValues: FXAIPluginRegistry.accelerationPlans().map { ($0.pluginName, $0) })

        XCTAssertTrue(plans["ai_mythos_rdt"]?.primaryBackends.contains(.pyTorchMPS) ?? false)
        XCTAssertTrue(plans["ai_lstm"]?.primaryBackends.contains(.tensorFlowMetal) ?? false)
        XCTAssertTrue(plans["tree_xgb_fast"]?.primaryBackends.contains(.metal) ?? false)
        XCTAssertTrue(plans["stat_tvp_kalman"]?.primaryBackends.contains(.accelerate) ?? false)
        XCTAssertTrue(plans["rl_ppo"]?.candidateBackends.contains(.coreMLNeuralEngine) ?? false)
    }

    private static func predictRequest(dataHasVolume: Bool) -> PredictRequestV4 {
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        features[0] = 0.09
        features[3] = 0.04
        features[4] = 0.20
        features[6] = 0.85
        features[7] = 0.08
        features[8] = -0.03
        features[12] = 0.06
        return PredictRequestV4(
            valid: true,
            context: PluginContextV4(
                sessionBucket: 2,
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
}
