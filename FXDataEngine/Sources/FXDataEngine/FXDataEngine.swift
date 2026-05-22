import Foundation

public struct PreparedPluginPayload: Sendable {
    public let dataBundle: DataCoreBundle
    public let featureFrame: FeatureCoreFrame
    public let normalizationFrame: NormalizationCoreFrame
    public let payloadFrame: NormalizationPayloadFrame
    public let context: PluginContextV4

    public var predictRequest: PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context,
            windowSize: payloadFrame.windowSize,
            x: payloadFrame.x,
            xWindow: payloadFrame.xWindow
        )
    }
}

public struct FXDataEnginePipeline: Sendable {
    public let dataCore: DataCore
    public let featureCore: FeatureCore
    public let normalizationCore: NormalizationCore
    public let schemaPolicy: FeatureSchemaPolicy

    public init(
        dataCore: DataCore = DataCore(),
        featureCore: FeatureCore = FeatureCore(),
        normalizationCore: NormalizationCore = NormalizationCore(),
        schemaPolicy: FeatureSchemaPolicy = FeatureSchemaPolicy()
    ) {
        self.dataCore = dataCore
        self.featureCore = featureCore
        self.normalizationCore = normalizationCore
        self.schemaPolicy = schemaPolicy
    }

    public func preparePredictPayload(
        universe: MarketUniverse,
        request dataRequest: DataCoreRequest,
        manifest: PluginManifestV4,
        horizonMinutes: Int,
        normalizationMethod: FeatureNormalizationMethod = .existing
    ) throws -> PreparedPluginPayload {
        try manifest.validate()
        let dataBundle = try dataCore.buildBundle(request: dataRequest, universe: universe)
        let featureFrame = try featureCore.buildFrame(
            bundle: dataBundle,
            request: FeatureCoreRequest(
                sampleIndex: dataBundle.sampleIndex,
                horizonMinutes: horizonMinutes,
                normalizationMethod: normalizationMethod
            )
        )
        let normalizationFrame = try normalizationCore.buildInputFrame(from: featureFrame)
        let sequenceBars = manifest.resolvedSequenceBars(horizonMinutes: horizonMinutes)
        let xWindow = buildInputWindow(
            universe: universe,
            centerIndex: dataBundle.sampleIndex,
            sequenceBars: sequenceBars,
            normalizationMethod: normalizationMethod
        )
        let payloadFrame = try normalizationCore.buildPayloadFrame(NormalizationPayloadRequest(
            valid: true,
            featureSchema: manifest.featureSchema,
            featureGroups: manifest.featureGroups,
            normalizationMethod: normalizationMethod,
            horizonMinutes: horizonMinutes,
            sequenceBars: sequenceBars,
            sampleTimeUTC: featureFrame.sampleTimeUTC,
            windowSize: xWindow.count,
            x: normalizationFrame.modelInput,
            xWindow: xWindow
        ))
        let context = PluginContextV4(
            regimeID: 0,
            sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: featureFrame.sampleTimeUTC),
            horizonMinutes: horizonMinutes,
            featureSchema: manifest.featureSchema,
            normalizationMethod: normalizationMethod,
            sequenceBars: sequenceBars,
            pointValue: 1.0 / pow(10.0, Double(universe.primary.metadata.digits)),
            domainHash: PluginContractTools.symbolHash01(universe.primarySymbol),
            sampleTimeUTC: featureFrame.sampleTimeUTC,
            dataHasVolume: featureFrame.hasVolume
        )
        return PreparedPluginPayload(
            dataBundle: dataBundle,
            featureFrame: featureFrame,
            normalizationFrame: normalizationFrame,
            payloadFrame: payloadFrame,
            context: context
        )
    }

    public func buildInputWindow(
        universe: MarketUniverse,
        centerIndex: Int,
        sequenceBars: Int,
        normalizationMethod: FeatureNormalizationMethod
    ) -> [[Double]] {
        let capped = min(max(1, sequenceBars), FXDataEngineConstants.maxSequenceBars)
        guard capped > 1, centerIndex > 0 else { return [] }
        let rows = min(capped - 1, centerIndex)
        var window: [[Double]] = []
        window.reserveCapacity(rows)
        for offset in stride(from: rows, through: 1, by: -1) {
            let featureFrame = FeatureCoreFrame(
                valid: true,
                sampleIndex: centerIndex - offset,
                horizonMinutes: 1,
                normalizationMethod: normalizationMethod,
                sampleTimeUTC: universe.primary.utcTimestamps[centerIndex - offset],
                hasVolume: FeatureCore.hasUsableVolume(universe),
                hasPrevious: centerIndex - offset > 0,
                raw: featureCore.buildFeatureVector(universe: universe, sampleIndex: centerIndex - offset),
                previous: centerIndex - offset > 0
                    ? featureCore.buildFeatureVector(universe: universe, sampleIndex: centerIndex - offset - 1)
                    : Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
            )
            if let normalized = try? normalizationCore.buildInputFrame(from: featureFrame) {
                window.append(normalized.modelInput)
            }
        }
        return window
    }
}
