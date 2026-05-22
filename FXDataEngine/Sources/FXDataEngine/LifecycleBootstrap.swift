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
}
