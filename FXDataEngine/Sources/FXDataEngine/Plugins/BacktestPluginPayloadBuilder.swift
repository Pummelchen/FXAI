import Foundation

public struct BacktestPluginPayloadBuildRequest: Sendable {
    public var universe: MarketUniverse
    public var pluginManifest: PluginManifestV4
    public var sampleIndex: Int
    public var horizonMinutes: Int
    public var normalizationMethod: FeatureNormalizationMethod
    public var priceCostPoints: Double
    public var minMovePoints: Double
    public var pointValue: Double
    public var textEvents: [PluginTextEventV4]

    public init(
        universe: MarketUniverse,
        pluginManifest: PluginManifestV4,
        sampleIndex: Int,
        horizonMinutes: Int = 1,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        priceCostPoints: Double = 0.0,
        minMovePoints: Double = 0.0,
        pointValue: Double = 1.0,
        textEvents: [PluginTextEventV4] = []
    ) {
        self.universe = universe
        self.pluginManifest = pluginManifest
        self.sampleIndex = sampleIndex
        self.horizonMinutes = max(1, horizonMinutes)
        self.normalizationMethod = normalizationMethod
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.pointValue = max(1.0e-12, fxSafeFinite(pointValue, fallback: 1.0))
        self.textEvents = Array(textEvents.prefix(PluginContextV4.maxTextEvents))
    }
}

public struct BacktestPluginPayload: Sendable {
    public var predictRequest: PredictRequestV4
    public var rawFeatures: [Double]
    public var modelInput: [Double]
    public var modelInputWindow: [[Double]]
    public var sampleIndex: Int
    public var sampleTimeUTC: Int64
    public var featureGraphHash: String
    public var normalizationStateHash: String

    public init(
        predictRequest: PredictRequestV4,
        rawFeatures: [Double],
        modelInput: [Double],
        modelInputWindow: [[Double]],
        sampleIndex: Int,
        sampleTimeUTC: Int64,
        featureGraphHash: String,
        normalizationStateHash: String
    ) {
        self.predictRequest = predictRequest
        self.rawFeatures = rawFeatures
        self.modelInput = modelInput
        self.modelInputWindow = modelInputWindow
        self.sampleIndex = sampleIndex
        self.sampleTimeUTC = sampleTimeUTC
        self.featureGraphHash = featureGraphHash
        self.normalizationStateHash = normalizationStateHash
    }
}

public struct BacktestPluginPayloadBuilder: Sendable {
    private let dataCore: DataCore
    private let featureCore: FeatureCore
    private let schemaPolicy: FeatureSchemaPolicy

    public init(
        dataCore: DataCore = DataCore(),
        featureCore: FeatureCore = FeatureCore(),
        schemaPolicy: FeatureSchemaPolicy = FeatureSchemaPolicy()
    ) {
        self.dataCore = dataCore
        self.featureCore = featureCore
        self.schemaPolicy = schemaPolicy
    }

    /// Builds a plugin prediction request from canonical M1 OHLCV data using only bars at or before `sampleIndex`.
    public func buildPredictRequest(_ request: BacktestPluginPayloadBuildRequest) throws -> BacktestPluginPayload {
        try request.pluginManifest.validate()
        let primary = request.universe.primary
        guard request.sampleIndex >= 0, request.sampleIndex < primary.count else {
            throw FXDataEngineError.invalidRequest("backtest sample index is outside primary series")
        }

        let sequenceBars = request.pluginManifest.resolvedSequenceBars(horizonMinutes: request.horizonMinutes)
        guard request.sampleIndex + 1 >= sequenceBars else {
            throw FXDataEngineError.insufficientData("sample index \(request.sampleIndex) does not leave enough history for \(sequenceBars) plugin sequence bars")
        }

        let contextSymbols = request.universe.contextSeries().map(\.metadata.logicalSymbol)
        let bundle = try dataCore.buildBundle(
            request: DataCoreRequest(
                liveMode: false,
                symbol: request.universe.primarySymbol,
                signalTimestampUTC: nil,
                neededBars: sequenceBars,
                alignUpToIndex: request.sampleIndex,
                contextSymbols: contextSymbols
            ),
            universe: request.universe
        )

        let frame = try featureCore.buildFrame(
            bundle: bundle,
            request: FeatureCoreRequest(
                sampleIndex: request.sampleIndex,
                horizonMinutes: request.horizonMinutes,
                normalizationMethod: request.normalizationMethod
            )
        )
        let modelInput = schemaPolicy.apply(
            schema: request.pluginManifest.featureSchema,
            groups: request.pluginManifest.featureGroups,
            to: schemaPolicy.modelInput(from: frame.raw)
        )

        let window = try buildWindow(
            bundle: bundle,
            sampleIndex: request.sampleIndex,
            sequenceBars: sequenceBars,
            manifest: request.pluginManifest,
            horizonMinutes: request.horizonMinutes,
            normalizationMethod: request.normalizationMethod
        )

        let context = PluginContextV4(
            regimeID: 0,
            sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: frame.sampleTimeUTC),
            horizonMinutes: request.horizonMinutes,
            featureSchema: request.pluginManifest.featureSchema,
            normalizationMethod: request.normalizationMethod,
            sequenceBars: sequenceBars,
            priceCostPoints: request.priceCostPoints,
            minMovePoints: request.minMovePoints,
            pointValue: request.pointValue,
            domainHash: PluginContractTools.symbolHash01(primary.metadata.logicalSymbol),
            sampleTimeUTC: frame.sampleTimeUTC,
            dataHasVolume: frame.hasVolume,
            tokenizerContract: PluginTokenizerContractV4(),
            textEvents: request.textEvents
        )
        try PluginContractTools.validateCompatibility(manifest: request.pluginManifest, context: context)

        let predictRequest = PredictRequestV4(
            valid: frame.valid,
            context: context,
            windowSize: window.count,
            x: modelInput,
            xWindow: window
        )
        try predictRequest.validate()

        return BacktestPluginPayload(
            predictRequest: predictRequest,
            rawFeatures: frame.raw,
            modelInput: modelInput,
            modelInputWindow: window,
            sampleIndex: request.sampleIndex,
            sampleTimeUTC: frame.sampleTimeUTC,
            featureGraphHash: Self.featureGraphHash(manifest: request.pluginManifest),
            normalizationStateHash: Self.normalizationStateHash(
                method: request.normalizationMethod,
                symbol: primary.metadata.logicalSymbol,
                sampleIndex: request.sampleIndex,
                sampleTimeUTC: frame.sampleTimeUTC
            )
        )
    }

    private func buildWindow(
        bundle: DataCoreBundle,
        sampleIndex: Int,
        sequenceBars: Int,
        manifest: PluginManifestV4,
        horizonMinutes: Int,
        normalizationMethod: FeatureNormalizationMethod
    ) throws -> [[Double]] {
        let windowSize = max(0, sequenceBars - 1)
        guard windowSize > 0 else { return [] }

        let start = sampleIndex - windowSize
        guard start >= 0 else {
            throw FXDataEngineError.insufficientData("plugin sequence window would require bars before the available dataset")
        }

        var rows: [[Double]] = []
        rows.reserveCapacity(windowSize)
        for index in start..<sampleIndex {
            let frame = try featureCore.buildFrame(
                bundle: bundle,
                request: FeatureCoreRequest(
                    sampleIndex: index,
                    horizonMinutes: horizonMinutes,
                    normalizationMethod: normalizationMethod
                )
            )
            let modelInput = schemaPolicy.apply(
                schema: manifest.featureSchema,
                groups: manifest.featureGroups,
                to: schemaPolicy.modelInput(from: frame.raw)
            )
            rows.append(modelInput)
        }
        return rows
    }

    public static func featureGraphHash(manifest: PluginManifestV4) -> String {
        stableHash([
            "fxdataengine.features",
            String(FXDataEngineConstants.latestPluginAPIVersion),
            manifest.aiName,
            String(manifest.featureSchema.rawValue),
            String(manifest.featureGroups.rawValue),
            String(manifest.minSequenceBars),
            String(manifest.maxSequenceBars)
        ])
    }

    public static func normalizationStateHash(
        method: FeatureNormalizationMethod,
        symbol: String,
        sampleIndex: Int,
        sampleTimeUTC: Int64
    ) -> String {
        stableHash([
            "fxdataengine.normalization",
            String(method.rawValue),
            symbol.uppercased(),
            String(sampleIndex),
            String(sampleTimeUTC)
        ])
    }

    private static func stableHash(_ parts: [String]) -> String {
        let input = parts.joined(separator: "\u{1F}")
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in input.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return String(format: "%016llx", hash)
    }
}
