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

    func testGeneratedAdaptersLearnFromTrainingSamples() throws {
        let buyRequest = Self.predictRequest(dataHasVolume: true)
        var sellFeatures = buyRequest.x
        sellFeatures[0] = -sellFeatures[0]
        sellFeatures[3] = -sellFeatures[3]
        sellFeatures[7] = -sellFeatures[7]
        sellFeatures[8] = abs(sellFeatures[8])
        let sellRequest = PredictRequestV4(valid: true, context: buyRequest.context, x: sellFeatures)

        for definition in FXAIGeneratedPluginDefinition.all {
            var adapter = FXAIGeneratedPluginAdapter(definition: definition)
            for _ in 0..<16 {
                try adapter.train(Self.trainRequest(from: buyRequest, label: .buy, movePoints: 2.5), hyperParameters: HyperParameters())
                try adapter.train(Self.trainRequest(from: sellRequest, label: .sell, movePoints: 2.0), hyperParameters: HyperParameters())
            }
            let afterBuy = try adapter.predict(buyRequest, hyperParameters: HyperParameters())
            let afterSell = try adapter.predict(sellRequest, hyperParameters: HyperParameters())

            XCTAssertGreaterThan(
                afterBuy.classProbabilities[LabelClass.buy.rawValue],
                afterSell.classProbabilities[LabelClass.buy.rawValue],
                "training did not separate buy and sell samples for \(definition.aiName)"
            )
            XCTAssertNoThrow(try afterBuy.validate(), definition.aiName)
            XCTAssertNoThrow(try afterSell.validate(), definition.aiName)
        }
    }

    func testGeneratedAdapterResetClearsOnlineRuntimeState() throws {
        let definition = try XCTUnwrap(FXAIGeneratedPluginDefinition.all.first { $0.aiName == "lin_sgd" })
        let request = Self.predictRequest(dataHasVolume: true)
        var adapter = FXAIGeneratedPluginAdapter(definition: definition)
        let before = try adapter.predict(request, hyperParameters: HyperParameters())

        for _ in 0..<20 {
            try adapter.train(Self.trainRequest(from: request, label: .buy, movePoints: 3.0), hyperParameters: HyperParameters())
        }
        let trained = try adapter.predict(request, hyperParameters: HyperParameters())
        XCTAssertNotEqual(trained.classProbabilities, before.classProbabilities)

        adapter.reset()
        let reset = try adapter.predict(request, hyperParameters: HyperParameters())
        for index in 0..<LabelClass.allCases.count {
            XCTAssertEqual(reset.classProbabilities[index], before.classProbabilities[index], accuracy: 1.0e-12)
        }
    }

    func testSequenceFamilyUsesWindowContext() throws {
        let definition = try XCTUnwrap(FXAIGeneratedPluginDefinition.all.first { $0.aiName == "ai_lstm" })
        let flatWindow = Self.window(rows: 7) { _, _ in 0.0 }
        let trendWindow = Self.window(rows: 7) { row, feature in
            switch feature {
            case 0, 3, 7:
                return 0.12 - 0.015 * Double(row)
            case 8:
                return -0.08 + 0.010 * Double(row)
            default:
                return 0.0
            }
        }
        let flatRequest = Self.predictRequest(dataHasVolume: true, sequenceBars: 8, xWindow: flatWindow)
        let trendRequest = Self.predictRequest(dataHasVolume: true, sequenceBars: 8, xWindow: trendWindow)
        let adapter = FXAIGeneratedPluginAdapter(definition: definition)

        let flatPrediction = try adapter.predict(flatRequest, hyperParameters: HyperParameters())
        let trendPrediction = try adapter.predict(trendRequest, hyperParameters: HyperParameters())

        XCTAssertNotEqual(flatPrediction.classProbabilities, trendPrediction.classProbabilities)
        XCTAssertGreaterThan(
            trendPrediction.classProbabilities[LabelClass.buy.rawValue],
            flatPrediction.classProbabilities[LabelClass.buy.rawValue]
        )
    }

    func testAccelerationPlansClassifyAppleSiliconBackends() throws {
        let plans = Dictionary(uniqueKeysWithValues: FXAIPluginRegistry.accelerationPlans().map { ($0.pluginName, $0) })

        XCTAssertTrue(plans["ai_mythos_rdt"]?.primaryBackends.contains(.pyTorchMPS) ?? false)
        XCTAssertTrue(plans["ai_lstm"]?.primaryBackends.contains(.tensorFlowMetal) ?? false)
        XCTAssertTrue(plans["tree_xgb_fast"]?.primaryBackends.contains(.metal) ?? false)
        XCTAssertTrue(plans["stat_tvp_kalman"]?.primaryBackends.contains(.accelerate) ?? false)
        XCTAssertTrue(plans["rl_ppo"]?.candidateBackends.contains(.coreMLNeuralEngine) ?? false)
    }

    private static func predictRequest(
        dataHasVolume: Bool,
        sequenceBars: Int = 1,
        xWindow: [[Double]] = []
    ) -> PredictRequestV4 {
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
                sequenceBars: sequenceBars,
                priceCostPoints: 0.20,
                minMovePoints: 0.50,
                pointValue: 1.0,
                sampleTimeUTC: 1_800_018_000,
                dataHasVolume: dataHasVolume
            ),
            windowSize: xWindow.count,
            x: features,
            xWindow: xWindow
        )
    }

    private static func window(rows: Int, value: (Int, Int) -> Double) -> [[Double]] {
        (0..<rows).map { row in
            (0..<FXDataEngineConstants.aiWeights).map { feature in
                value(row, feature)
            }
        }
    }

    private static func trainRequest(from request: PredictRequestV4, label: LabelClass, movePoints: Double) -> TrainRequestV4 {
        TrainRequestV4(
            valid: true,
            context: request.context,
            labelClass: label,
            movePoints: movePoints,
            sampleWeight: 1.0,
            nextVolumeTarget: request.context.dataHasVolume ? 1.0 : 0.0,
            x: request.x
        )
    }
}
