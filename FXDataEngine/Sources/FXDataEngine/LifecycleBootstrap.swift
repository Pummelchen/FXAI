import Foundation

public struct LifecycleBootstrapProbe: Sendable {
    public var aiID: Int
    public var manifest: PluginManifestV4
    public var hyperParameters: HyperParameters
    public var rawComplianceWindow: [[Double]]
    public var predictRequest: PredictRequestV4
    public var requiresPredict: Bool
    public var requiresSelfTest: Bool

    public init(
        aiID: Int,
        manifest: PluginManifestV4,
        hyperParameters: HyperParameters = HyperParameters(),
        rawComplianceWindow: [[Double]] = [],
        predictRequest: PredictRequestV4,
        requiresPredict: Bool = true,
        requiresSelfTest: Bool = true
    ) {
        self.aiID = max(0, aiID)
        self.manifest = manifest
        self.hyperParameters = hyperParameters
        self.rawComplianceWindow = rawComplianceWindow
        self.predictRequest = predictRequest
        self.requiresPredict = requiresPredict
        self.requiresSelfTest = requiresSelfTest
    }
}

public struct LifecycleBootstrapPlan: Sendable {
    public var valid: Bool
    public var reason: String
    public var probes: [LifecycleBootstrapProbe]

    public init(valid: Bool = true, reason: String = "", probes: [LifecycleBootstrapProbe] = []) {
        self.valid = valid
        self.reason = reason
        self.probes = probes
    }
}

public struct LifecycleBootstrapProbeExecution: Codable, Hashable, Sendable {
    public var aiID: Int
    public var aiName: String
    public var manifestMatched: Bool
    public var requestValidated: Bool
    public var predictRequired: Bool
    public var predictSucceeded: Bool
    public var predictionValidated: Bool
    public var selfTestRequired: Bool
    public var selfTestPassed: Bool
    public var valid: Bool
    public var reason: String

    public init(
        aiID: Int,
        aiName: String,
        manifestMatched: Bool = false,
        requestValidated: Bool = false,
        predictRequired: Bool = true,
        predictSucceeded: Bool = false,
        predictionValidated: Bool = false,
        selfTestRequired: Bool = true,
        selfTestPassed: Bool = false,
        valid: Bool = false,
        reason: String = ""
    ) {
        self.aiID = max(0, aiID)
        self.aiName = aiName
        self.manifestMatched = manifestMatched
        self.requestValidated = requestValidated
        self.predictRequired = predictRequired
        self.predictSucceeded = predictSucceeded
        self.predictionValidated = predictionValidated
        self.selfTestRequired = selfTestRequired
        self.selfTestPassed = selfTestPassed
        self.valid = valid
        self.reason = reason
    }
}

public struct LifecycleBootstrapExecutionResult: Codable, Hashable, Sendable {
    public var valid: Bool
    public var reason: String
    public var probes: [LifecycleBootstrapProbeExecution]

    public init(valid: Bool = true, reason: String = "", probes: [LifecycleBootstrapProbeExecution] = []) {
        self.valid = valid
        self.reason = reason
        self.probes = probes
    }
}

public struct LifecycleBootstrapPluginFactory: Sendable {
    public let manifest: PluginManifestV4
    private let runProbeBody: @Sendable (_ probe: LifecycleBootstrapProbe) -> LifecycleBootstrapProbeExecution

    public init<Plugin: FXAIPluginV4>(
        manifest: PluginManifestV4,
        makePlugin: @escaping @Sendable () -> Plugin
    ) {
        self.manifest = manifest
        self.runProbeBody = { probe in
            var execution = LifecycleBootstrapProbeExecution(
                aiID: probe.aiID,
                aiName: probe.manifest.aiName,
                predictRequired: probe.requiresPredict,
                selfTestRequired: probe.requiresSelfTest
            )

            let plugin = makePlugin()
            let pluginManifest = plugin.manifest
            do {
                try probe.manifest.validate()
                try pluginManifest.validate()
            } catch {
                execution.reason = "manifest_invalid:\(probe.aiID):\(error)"
                return execution
            }

            guard pluginManifest == probe.manifest else {
                execution.reason = "manifest_mismatch:\(probe.aiID)"
                return execution
            }
            execution.manifestMatched = true

            do {
                try probe.predictRequest.validate()
                execution.requestValidated = true
            } catch {
                execution.reason = "predict_request_invalid:\(probe.aiID):\(error)"
                return execution
            }

            if probe.requiresPredict {
                let prediction: PredictionV4
                do {
                    prediction = try plugin.predict(
                        probe.predictRequest,
                        hyperParameters: probe.hyperParameters
                    )
                    execution.predictSucceeded = true
                } catch {
                    execution.reason = "predict_failed:\(probe.aiID):\(error)"
                    return execution
                }

                do {
                    try prediction.validate()
                    execution.predictionValidated = true
                } catch {
                    execution.reason = "prediction_invalid:\(probe.aiID):\(error)"
                    return execution
                }
            } else {
                execution.predictSucceeded = true
                execution.predictionValidated = true
            }

            if probe.requiresSelfTest {
                guard plugin.selfTest() else {
                    execution.reason = "self_test_failed:\(probe.aiID)"
                    return execution
                }
                execution.selfTestPassed = true
            } else {
                execution.selfTestPassed = true
            }

            execution.valid = true
            return execution
        }
    }

    public init<Plugin: FXAIPluginV4>(
        makePlugin: @escaping @Sendable () -> Plugin
    ) {
        let plugin = makePlugin()
        self.init(manifest: plugin.manifest, makePlugin: makePlugin)
    }

    public func runProbe(_ probe: LifecycleBootstrapProbe) -> LifecycleBootstrapProbeExecution {
        runProbeBody(probe)
    }
}

public enum LifecycleBootstrapTools {
    public static func dummyFeatureVector() -> [Double] {
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        features[0] = 1.0
        return features
    }

    public static func buildProbe(
        manifest: PluginManifestV4,
        symbol: String,
        sampleTimeUTC: Int64,
        normalizationMethod: FeatureNormalizationMethod,
        pointValue: Double,
        dataHasVolume: Bool = false,
        normalizer: NormalizationCore = NormalizationCore(),
        hyperParameters: HyperParameters = HyperParameters()
    ) throws -> LifecycleBootstrapProbe {
        try manifest.validate()

        let horizonMinutes = LifecycleComplianceTools.complianceHorizon(
            manifest: manifest,
            desiredHorizonMinutes: 5
        )
        let sequenceBars = manifest.resolvedSequenceBars(horizonMinutes: horizonMinutes)
        let context = PluginContextV4(
            regimeID: 0,
            sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: sampleTimeUTC),
            horizonMinutes: horizonMinutes,
            featureSchema: manifest.featureSchema,
            normalizationMethod: normalizationMethod,
            sequenceBars: sequenceBars,
            pointValue: pointValue > 0.0 && pointValue.isFinite ? pointValue : 1.0,
            domainHash: PluginContractTools.symbolHash01(symbol),
            sampleTimeUTC: max(0, sampleTimeUTC),
            dataHasVolume: dataHasVolume
        )

        let features = dummyFeatureVector()
        let window = LifecycleComplianceTools.complianceWindow(features: features, sequenceBars: sequenceBars)
        let request = PredictRequestV4(
            valid: true,
            context: context,
            windowSize: window.count,
            x: features,
            xWindow: window
        )
        let finalized = try normalizer.finalizePredictRequest(manifest: manifest, request: request)
        try finalized.validate()

        return LifecycleBootstrapProbe(
            aiID: manifest.aiID,
            manifest: manifest,
            hyperParameters: hyperParameters,
            rawComplianceWindow: window,
            predictRequest: finalized
        )
    }

    public static func buildValidationPlan(
        featureRegistrySelfTestPassed: Bool,
        manifests: [PluginManifestV4?],
        symbol: String,
        sampleTimeUTC: Int64,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        pointValue: Double = 1.0,
        dataHasVolume: Bool = false,
        expectedPluginCount: Int = FXDataEngineConstants.aiCount,
        normalizer: NormalizationCore = NormalizationCore()
    ) -> LifecycleBootstrapPlan {
        guard featureRegistrySelfTestPassed else {
            return LifecycleBootstrapPlan(valid: false, reason: "feature_registry_self_test")
        }

        let pluginCount = max(0, min(expectedPluginCount, FXDataEngineConstants.aiCount))
        var probes: [LifecycleBootstrapProbe] = []
        probes.reserveCapacity(pluginCount)

        for aiID in 0..<pluginCount {
            guard aiID < manifests.count, let manifest = manifests[aiID] else {
                return LifecycleBootstrapPlan(valid: false, reason: "plugin_missing:\(aiID)", probes: probes)
            }
            do {
                let probe = try buildProbe(
                    manifest: manifest,
                    symbol: symbol,
                    sampleTimeUTC: sampleTimeUTC,
                    normalizationMethod: normalizationMethod,
                    pointValue: pointValue,
                    dataHasVolume: dataHasVolume,
                    normalizer: normalizer
                )
                probes.append(probe)
            } catch {
                return LifecycleBootstrapPlan(
                    valid: false,
                    reason: "plugin_bootstrap:\(aiID):\(error)",
                    probes: probes
                )
            }
        }

        return LifecycleBootstrapPlan(probes: probes)
    }

    public static func executeValidationPlan(
        _ plan: LifecycleBootstrapPlan,
        plugins: [LifecycleBootstrapPluginFactory]
    ) -> LifecycleBootstrapExecutionResult {
        guard plan.valid else {
            let reason = plan.reason.isEmpty ? "bootstrap_plan_invalid" : plan.reason
            return LifecycleBootstrapExecutionResult(valid: false, reason: reason)
        }

        var byAIID: [Int: LifecycleBootstrapPluginFactory] = [:]
        for plugin in plugins {
            let aiID = plugin.manifest.aiID
            do {
                try plugin.manifest.validate()
            } catch {
                return LifecycleBootstrapExecutionResult(
                    valid: false,
                    reason: "manifest_invalid:\(aiID):\(error)"
                )
            }
            guard byAIID[aiID] == nil else {
                return LifecycleBootstrapExecutionResult(
                    valid: false,
                    reason: "plugin_duplicate:\(aiID)"
                )
            }
            byAIID[aiID] = plugin
        }

        var executions: [LifecycleBootstrapProbeExecution] = []
        executions.reserveCapacity(plan.probes.count)
        for probe in plan.probes {
            guard let plugin = byAIID[probe.aiID] else {
                let execution = LifecycleBootstrapProbeExecution(
                    aiID: probe.aiID,
                    aiName: probe.manifest.aiName,
                    predictRequired: probe.requiresPredict,
                    selfTestRequired: probe.requiresSelfTest,
                    reason: "plugin_missing:\(probe.aiID)"
                )
                executions.append(execution)
                return LifecycleBootstrapExecutionResult(
                    valid: false,
                    reason: execution.reason,
                    probes: executions
                )
            }

            let execution = plugin.runProbe(probe)
            executions.append(execution)
            guard execution.valid else {
                return LifecycleBootstrapExecutionResult(
                    valid: false,
                    reason: execution.reason,
                    probes: executions
                )
            }
        }

        return LifecycleBootstrapExecutionResult(probes: executions)
    }
}
