import FXDataEngine
import XCTest
@testable import FXAIPlugins

final class SineTestPluginSmokeTests: XCTestCase {
    func testEveryPluginPredictsAndTrainsOnLocalSineTestSeries() async throws {
        let marketSeries = try Self.makeSineTestSeries()
        let universe = try MarketUniverse(primarySymbol: "SINETEST", series: [marketSeries])
        let featureCore = FeatureCore()
        let schemaPolicy = FeatureSchemaPolicy()
        let sampleIndex = 420
        let rawFeaturesByIndex = (0...sampleIndex).map { index in
            featureCore.buildFeatureVector(universe: universe, sampleIndex: index)
        }
        let modelInputsByIndex = rawFeaturesByIndex.map { schemaPolicy.modelInput(from: $0) }
        let plugins = FXAIPluginRegistry.availablePlugins().compactMap { $0 as? any FXAIPlannedPlugin }
        XCTAssertEqual(plugins.count, FXDataEngineConstants.aiCount)

        let noAcceleratorEnvironment = FXPluginRuntimeEnvironment(
            metalDevice: MetalAccelerationDevice(available: false, deviceName: nil, supportsUnifiedMemory: false),
            pythonExecutable: nil,
            pyTorchMPSAvailable: false,
            tensorFlowMetalAvailable: false,
            foundationNLPAvailable: false,
            coreMLNeuralEngineAvailable: false,
            onnxRuntimeAvailable: false,
            remoteInferenceAvailable: false
        )

        for var plugin in plugins {
            let request = try Self.predictRequest(
                for: plugin,
                marketSeries: marketSeries,
                sampleIndex: sampleIndex,
                modelInputsByIndex: modelInputsByIndex,
                schemaPolicy: schemaPolicy
            )
            try request.validate()
            try PluginContractTools.validateCompatibility(manifest: plugin.manifest, context: request.context)

            let prediction = try plugin.predict(request, hyperParameters: HyperParameters())
            try prediction.validate()

            try plugin.train(
                Self.trainRequest(from: request, marketSeries: marketSeries, sampleIndex: sampleIndex),
                hyperParameters: HyperParameters()
            )
            let trainedPrediction = try plugin.predict(request, hyperParameters: HyperParameters())
            try trainedPrediction.validate()

            var runtime = FXAIAcceleratedPluginRuntime(
                plugin: plugin,
                configuration: FXAIPluginRuntimeConfiguration(
                    mode: .automatic,
                    fallbackPolicy: .fallBackToCPU,
                    environment: noAcceleratorEnvironment
                )
            )
            let runtimePrediction = try runtime.predict(request, hyperParameters: HyperParameters())
            try runtimePrediction.validate()
            try runtime.train(
                Self.trainRequest(from: request, marketSeries: marketSeries, sampleIndex: sampleIndex),
                hyperParameters: HyperParameters()
            )

            for backend in plugin.accelerationPlan.declaredBackends where !backend.isCPUOnly {
                let forcedRuntime = FXAIAcceleratedPluginRuntime(
                    plugin: plugin,
                    configuration: FXAIPluginRuntimeConfiguration(
                        mode: Self.runtimeMode(for: backend),
                        fallbackPolicy: .fallBackToCPU,
                        environment: noAcceleratorEnvironment
                    )
                )
                let forcedPrediction = try forcedRuntime.predict(request, hyperParameters: HyperParameters())
                try forcedPrediction.validate()
            }
        }
    }

    private static func makeSineTestSeries() throws -> M1OHLCVSeries {
        let startUTC: Int64 = 1_704_067_200
        return try PluginSineTestSeriesFactory.makeSeries(
            brokerSourceId: "plugin-smoke",
            startUTC: startUTC,
            dayCount: 1
        )
    }

    private static func predictRequest(
        for plugin: any FXAIPlannedPlugin,
        marketSeries: M1OHLCVSeries,
        sampleIndex: Int,
        modelInputsByIndex: [[Double]],
        schemaPolicy: FeatureSchemaPolicy
    ) throws -> PredictRequestV4 {
        let horizon = min(max(15, plugin.manifest.minHorizonMinutes), plugin.manifest.maxHorizonMinutes)
        let sequenceBars = plugin.manifest.resolvedSequenceBars(horizonMinutes: horizon)
        let windowSize = max(sequenceBars - 1, 0)
        guard sampleIndex >= windowSize, sampleIndex < modelInputsByIndex.count else {
            throw FXDataEngineError.invalidRequest("SineTest sample index cannot satisfy \(plugin.manifest.aiName)")
        }
        let x = Self.applySchema(
            modelInputsByIndex[sampleIndex],
            plugin: plugin,
            schemaPolicy: schemaPolicy
        )
        let xWindow = windowSize == 0
            ? []
            : (sampleIndex - windowSize..<sampleIndex).map { index in
                Self.applySchema(modelInputsByIndex[index], plugin: plugin, schemaPolicy: schemaPolicy)
            }
        let sampleTimeUTC = marketSeries.utcTimestamps[sampleIndex]
        return PredictRequestV4(
            valid: true,
            context: PluginContextV4(
                sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: sampleTimeUTC),
                horizonMinutes: horizon,
                featureSchema: plugin.manifest.featureSchema,
                sequenceBars: sequenceBars,
                minMovePoints: 1.0,
                pointValue: 1.0,
                domainHash: PluginContractTools.symbolHash01(marketSeries.metadata.logicalSymbol),
                sampleTimeUTC: sampleTimeUTC,
                dataHasVolume: marketSeries.hasVolume
            ),
            windowSize: xWindow.count,
            x: x,
            xWindow: xWindow
        )
    }

    private static func trainRequest(
        from request: PredictRequestV4,
        marketSeries: M1OHLCVSeries,
        sampleIndex: Int
    ) -> TrainRequestV4 {
        let horizonBars = min(request.context.horizonMinutes, marketSeries.count - sampleIndex - 1)
        let currentClose = marketSeries.close[sampleIndex]
        let nextClose = marketSeries.close[sampleIndex + max(1, horizonBars)]
        let delta = nextClose - currentClose
        let label: LabelClass = delta > 0 ? .buy : (delta < 0 ? .sell : .skip)
        return TrainRequestV4(
            valid: true,
            context: request.context,
            labelClass: label,
            movePoints: abs(Double(delta)),
            sampleWeight: 1.0,
            nextVolumeTarget: request.context.dataHasVolume ? Double(marketSeries.volume[sampleIndex + 1]) : 0.0,
            windowSize: request.windowSize,
            x: request.x,
            xWindow: request.xWindow
        )
    }

    private static func applySchema(
        _ x: [Double],
        plugin: any FXAIPlannedPlugin,
        schemaPolicy: FeatureSchemaPolicy
    ) -> [Double] {
        schemaPolicy.apply(schema: plugin.manifest.featureSchema, groups: plugin.manifest.featureGroups, to: x)
    }

    private static func runtimeMode(for backend: FXPluginAccelerationBackend) -> FXPluginRuntimeMode {
        switch backend {
        case .swiftScalar:
            return .swiftScalar
        case .swiftSIMD:
            return .swiftSIMD
        case .accelerate:
            return .accelerate
        case .metal:
            return .metal
        case .pyTorchMPS:
            return .pyTorchMPS
        case .tensorFlowMetal:
            return .tensorFlowMetal
        case .foundationNLP:
            return .foundationNLP
        case .coreMLNeuralEngine:
            return .coreMLNeuralEngine
        case .onnxRuntime:
            return .onnxRuntime
        case .remoteRPC:
            return .remoteRPC
        }
    }
}
