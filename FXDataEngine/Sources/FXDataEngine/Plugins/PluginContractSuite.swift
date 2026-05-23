import Foundation

public protocol FXAIPluginPersistentState: FXAIPluginV4 {
    mutating func saveStateData() throws -> Data
    mutating func loadStateData(_ data: Data) throws
}

public protocol FXAIPluginSyntheticSeriesSupport: FXAIPluginV4 {
    mutating func setSyntheticSeries(_ series: M1OHLCVSeries) throws
    mutating func clearSyntheticSeries()
}

public struct PluginContractSuiteConfiguration: Codable, Hashable, Sendable {
    public var requireCompleteAISet: Bool
    public var sampleTimeUTC: Int64
    public var dataHasVolume: Bool

    public init(
        requireCompleteAISet: Bool = false,
        sampleTimeUTC: Int64 = PluginContractSuiteTools.defaultSampleTimeUTC,
        dataHasVolume: Bool = true
    ) {
        self.requireCompleteAISet = requireCompleteAISet
        self.sampleTimeUTC = max(0, sampleTimeUTC)
        self.dataHasVolume = dataHasVolume
    }
}

public struct PluginContractSuiteFactory: Sendable {
    public let manifest: PluginManifestV4
    private let makePluginBody: @Sendable () -> any FXAIPluginV4

    public init<Plugin: FXAIPluginV4>(makePlugin: @escaping @Sendable () -> Plugin) {
        let plugin = makePlugin()
        self.manifest = plugin.manifest
        self.makePluginBody = { makePlugin() }
    }

    public func makePlugin() -> any FXAIPluginV4 {
        makePluginBody()
    }
}

public struct AuditPluginContractSelfTestResult: Codable, Hashable, Sendable {
    public var passed: Bool
    public var reason: String
    public var suite: TestSuiteResult

    public init(suite: TestSuiteResult) {
        self.suite = suite
        self.passed = suite.passed
        self.reason = suite.legacyReason
    }
}

public enum AuditPluginContractTools {
    public static func selfTest(
        factories: [PluginContractSuiteFactory],
        configuration: PluginContractSuiteConfiguration = PluginContractSuiteConfiguration()
    ) -> AuditPluginContractSelfTestResult {
        AuditPluginContractSelfTestResult(
            suite: PluginContractSuiteTools.runSuite(
                factories: factories,
                configuration: configuration
            )
        )
    }
}

public enum PluginContractSuiteTools {
    public static let defaultSampleTimeUTC: Int64 = 1_704_153_600

    public static func defaultHyperParameters() -> HyperParameters {
        HyperParameters(
            learningRate: 0.01,
            l2: 0.0001,
            ftrlAlpha: 0.05,
            ftrlBeta: 1.0,
            ftrlL1: 0.0,
            ftrlL2: 0.0001,
            passiveAggressiveC: 0.5,
            passiveAggressiveMargin: 1.0,
            xgbLearningRate: 0.05,
            xgbL2: 0.0001,
            xgbSplit: 0.5,
            mlpLearningRate: 0.01,
            mlpL2: 0.0001,
            mlpInit: 0.05,
            quantileLearningRate: 0.01,
            quantileL2: 0.0001,
            enhashLearningRate: 0.01,
            enhashL1: 0.0,
            enhashL2: 0.0001,
            tcnLayers: 2.0,
            tcnKernel: 3.0,
            tcnDilationBase: 2.0
        )
    }

    public static func buildPredictRequest(
        manifest: PluginManifestV4,
        sampleTimeUTC: Int64 = defaultSampleTimeUTC,
        dataHasVolume: Bool = true
    ) -> PredictRequestV4 {
        let sequenceBars = max(manifest.minSequenceBars, 1)
        let context = PluginContextV4(
            regimeID: 0,
            sessionBucket: 0,
            horizonMinutes: max(manifest.minHorizonMinutes, 1),
            featureSchema: manifest.featureSchema,
            normalizationMethod: .existing,
            sequenceBars: sequenceBars,
            priceCostPoints: 0.0,
            minMovePoints: 0.0,
            pointValue: 0.0001,
            domainHash: 0.5,
            sampleTimeUTC: sampleTimeUTC,
            dataHasVolume: dataHasVolume
        )
        let windowSize = max(context.sequenceBars - 1, 0)
        return PredictRequestV4(
            valid: true,
            context: context,
            windowSize: windowSize,
            x: deterministicInputVector(),
            xWindow: deterministicInputWindow(rowCount: windowSize)
        )
    }

    public static func buildSyntheticSeries(symbol: String = "EURUSD") throws -> M1OHLCVSeries {
        let count = 32
        let baseTime = defaultSampleTimeUTC
        var timestamps = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        timestamps.reserveCapacity(count)
        open.reserveCapacity(count)
        high.reserveCapacity(count)
        low.reserveCapacity(count)
        close.reserveCapacity(count)
        volume.reserveCapacity(count)

        for index in 0..<count {
            let base = 110_000 + Int64(20 * index)
            let closeValue = base + (index % 2 == 0 ? 10 : -5)
            timestamps.append(baseTime + Int64(60 * index))
            open.append(base)
            close.append(closeValue)
            high.append(max(base, closeValue) + 15)
            low.append(min(base, closeValue) - 15)
            volume.append(UInt64(100 + index))
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "contract_suite",
                sourceOrigin: "SYNTHETIC",
                logicalSymbol: symbol,
                timeframe: .m1,
                digits: 5,
                firstUTC: timestamps.first,
                lastUTC: timestamps.last
            ),
            utcTimestamps: timestamps,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }

    public static func finitePredictionFailureReason(_ prediction: PredictionV4) -> String? {
        guard prediction.classProbabilities.count == 3 else {
            return "class_probs"
        }
        var probabilitySum = 0.0
        for probability in prediction.classProbabilities {
            guard probability.isFinite, probability >= 0.0, probability <= 1.0 else {
                return "class_probs"
            }
            probabilitySum += probability
        }
        guard abs(probabilitySum - 1.0) <= 1.0e-6 else {
            return "probability_sum"
        }

        let fields = [
            prediction.moveMeanPoints,
            prediction.moveQ25Points,
            prediction.moveQ50Points,
            prediction.moveQ75Points,
            prediction.mfeMeanPoints,
            prediction.maeMeanPoints,
            prediction.hitTimeFraction,
            prediction.pathRisk,
            prediction.fillRisk,
            prediction.confidence,
            prediction.reliability,
        ]
        return fields.allSatisfy(\.isFinite) ? nil : "prediction_fields"
    }

    public static func runSuite(
        factories: [PluginContractSuiteFactory],
        configuration: PluginContractSuiteConfiguration = PluginContractSuiteConfiguration()
    ) -> TestSuiteResult {
        var suite = TestSuiteTools.reset("plugin_contracts")
        var reason = registryLifecycleFailureReason(
            factories: factories,
            requireCompleteAISet: configuration.requireCompleteAISet
        )
        suite.addCase(name: "registry_lifecycle", passed: reason == nil, reason: reason ?? "")

        reason = manifestAndSelfTestFailureReason(factories: factories)
        suite.addCase(name: "manifest_and_selftest", passed: reason == nil, reason: reason ?? "")

        reason = predictContractFailureReason(factories: factories, configuration: configuration)
        suite.addCase(name: "predict_request_contract", passed: reason == nil, reason: reason ?? "")

        reason = persistenceRoundTripFailureReason(factories: factories)
        suite.addCase(name: "persistent_state_roundtrip", passed: reason == nil, reason: reason ?? "")

        reason = syntheticSeriesFailureReason(factories: factories)
        suite.addCase(name: "synthetic_series_contract", passed: reason == nil, reason: reason ?? "")
        return suite
    }

    private static func deterministicInputVector() -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<output.count {
            output[index] = index == 0 ? 1.0 : 0.01 * Double(index + 1)
        }
        return output
    }

    private static func deterministicInputWindow(rowCount: Int) -> [[Double]] {
        guard rowCount > 0 else { return [] }
        var rows: [[Double]] = []
        rows.reserveCapacity(rowCount)
        for row in 0..<rowCount {
            var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
            for feature in 0..<values.count {
                values[feature] = feature == 0 ? 1.0 : 0.005 * Double(row + 1) * Double(feature + 1)
            }
            rows.append(values)
        }
        return rows
    }

    private static func registryLifecycleFailureReason(
        factories: [PluginContractSuiteFactory],
        requireCompleteAISet: Bool
    ) -> String? {
        guard !factories.isEmpty else { return "registry_initialize" }
        var seen = Set<Int>()
        for factory in factories {
            let aiID = factory.manifest.aiID
            guard (0..<FXDataEngineConstants.aiCount).contains(aiID) else {
                return "registry_missing_\(aiID)"
            }
            guard seen.insert(aiID).inserted else {
                return "registry_duplicate_\(aiID)"
            }
        }
        if requireCompleteAISet {
            for aiID in 0..<FXDataEngineConstants.aiCount where !seen.contains(aiID) {
                return "registry_missing_\(aiID)"
            }
        }
        return nil
    }

    private static func manifestAndSelfTestFailureReason(factories: [PluginContractSuiteFactory]) -> String? {
        for factory in factories {
            let plugin = factory.makePlugin()
            let manifest = plugin.manifest
            do {
                try manifest.validate()
            } catch {
                return "manifest_selftest_\(manifest.aiID)"
            }
            guard manifest.aiID == factory.manifest.aiID,
                  !manifest.aiName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  plugin.selfTest() else {
                return "manifest_selftest_\(manifest.aiID)"
            }
        }
        return nil
    }

    private static func predictContractFailureReason(
        factories: [PluginContractSuiteFactory],
        configuration: PluginContractSuiteConfiguration
    ) -> String? {
        let hyperParameters = defaultHyperParameters()
        for factory in factories {
            var plugin = factory.makePlugin()
            let manifest = plugin.manifest
            let request = buildPredictRequest(
                manifest: manifest,
                sampleTimeUTC: configuration.sampleTimeUTC,
                dataHasVolume: configuration.dataHasVolume
            )
            do {
                try request.validate()
                try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
                plugin.reset()
                let prediction = try plugin.predict(request, hyperParameters: hyperParameters)
                if let predictionReason = finitePredictionFailureReason(prediction) {
                    return "predict_\(manifest.aiID)_\(predictionReason)"
                }
            } catch {
                return "predict_\(manifest.aiID)_prediction_fields"
            }
        }
        return nil
    }

    private static func persistenceRoundTripFailureReason(factories: [PluginContractSuiteFactory]) -> String? {
        for factory in factories {
            let plugin = factory.makePlugin()
            guard var persistent = plugin as? any FXAIPluginPersistentState else {
                continue
            }
            let aiID = persistent.manifest.aiID
            let data: Data
            do {
                data = try persistent.saveStateData()
            } catch {
                return "save_state_\(aiID)"
            }
            do {
                persistent.reset()
                try persistent.loadStateData(data)
            } catch {
                return "load_state_\(aiID)"
            }
        }
        return nil
    }

    private static func syntheticSeriesFailureReason(factories: [PluginContractSuiteFactory]) -> String? {
        for factory in factories {
            let plugin = factory.makePlugin()
            guard var synthetic = plugin as? any FXAIPluginSyntheticSeriesSupport else {
                continue
            }
            let aiID = synthetic.manifest.aiID
            do {
                let series = try buildSyntheticSeries()
                try synthetic.setSyntheticSeries(series)
                synthetic.clearSyntheticSeries()
            } catch {
                return "synthetic_series_\(aiID)"
            }
        }
        return nil
    }
}
