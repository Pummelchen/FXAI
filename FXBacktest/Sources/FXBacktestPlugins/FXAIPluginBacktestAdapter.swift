import FXAIPlugins
import FXBacktestCore
import FXBacktestAPI
import FXDataEngine
import Foundation

public struct FXAIPluginBacktestAdapter: FXBacktestPluginV1 {
    public let descriptor: FXBacktestPluginDescriptor
    public let parameterDefinitions: [ParameterDefinition]
    public let metalKernel: MetalKernelV1? = nil
    public let accelerationDescriptor: PluginAccelerationDescriptor

    private let manifest: PluginManifestV4
    private let accelerationPlan: FXPluginAccelerationPlan
    private let makeRuntime: @Sendable () -> FXAIAcceleratedPluginRuntime
    private let payloadBuilder: BacktestPluginPayloadBuilder

    public init(runtime: FXAIAcceleratedPluginRuntime) throws {
        self.manifest = runtime.manifest
        self.accelerationPlan = runtime.accelerationPlan
        self.makeRuntime = {
            var passRuntime = runtime
            passRuntime.configuration = FXAIPluginRuntimeConfiguration(mode: .cpuOnly)
            return passRuntime
        }
        self.payloadBuilder = BacktestPluginPayloadBuilder()
        self.descriptor = FXBacktestPluginDescriptor(
            id: runtime.manifest.aiName,
            displayName: Self.displayName(from: runtime.manifest),
            version: "fxdataengine-api-v\(runtime.manifest.apiVersion)",
            apiVersion: .latest,
            summary: Self.summary(from: runtime.manifest, plan: runtime.accelerationPlan),
            author: "FXAI",
            supportsCPU: true,
            supportsMetal: false
        )
        self.parameterDefinitions = try Self.parameterDefinitions(for: runtime)
        self.accelerationDescriptor = PluginAccelerationDescriptor(
            pluginIdentifier: runtime.manifest.aiName,
            supportedBackends: [.swiftScalar],
            safety: .deterministicWholePass
        )
        try descriptor.validateLatestAPI()
    }

    public func runPass(
        market: OhlcDataSeries,
        parameters: ParameterVector,
        context: BacktestContext
    ) throws -> BacktestPassResult {
        try runPass(marketUniverse: market.universe, parameters: parameters, context: context)
    }

    public func runPass(
        marketUniverse: OhlcMarketUniverse,
        parameters: ParameterVector,
        context: BacktestContext
    ) throws -> BacktestPassResult {
        let universe = try Self.marketUniverse(from: marketUniverse)
        let primary = universe.primary
        let minimumBars = max(3, manifest.resolvedSequenceBars(horizonMinutes: horizonMinutes(parameters)))
        guard primary.count > minimumBars else {
            throw FXBacktestError.invalidMarketData("\(descriptor.id) needs more than \(minimumBars) M1 bars.")
        }

        var runtime = makeRuntime()
        runtime.reset()
        let hyperParameters = Self.hyperParameters(from: parameters)
        let confidenceThreshold = fxClamp(parameters["signal_confidence_threshold"] ?? 0.55, 0.0, 1.0)
        let closeOnSkip = (parameters["close_on_skip"] ?? 1.0) >= 0.5
        let horizon = horizonMinutes(parameters)
        let minMovePoints = max(0.0, parameters["min_move_points"] ?? 0.0)
        var broker = BacktestBroker(context: context)
        var flags: UInt32 = 0

        for sampleIndex in (minimumBars - 1)..<(primary.count - 1) {
            let payload = try payloadBuilder.buildPredictRequest(
                BacktestPluginPayloadBuildRequest(
                    universe: universe,
                    pluginManifest: manifest,
                    sampleIndex: sampleIndex,
                    horizonMinutes: horizon,
                    normalizationMethod: .existing,
                    minMovePoints: minMovePoints,
                    pointValue: pow(10.0, -Double(primary.metadata.digits))
                )
            )
            let prediction = try runtime.predict(payload.predictRequest, hyperParameters: hyperParameters)
            try prediction.validate()

            let executionIndex = sampleIndex + 1
            let executionPrice = primary.close[executionIndex]
            switch Self.tradeDirection(
                from: prediction,
                confidenceThreshold: confidenceThreshold,
                minMovePoints: minMovePoints
            ) {
            case .long:
                broker.openMarket(direction: .long, price: executionPrice)
            case .short:
                broker.openMarket(direction: .short, price: executionPrice)
            case nil:
                if closeOnSkip {
                    broker.closeMarket(price: executionPrice)
                } else {
                    broker.markToMarket(price: executionPrice)
                }
            }

            let futureDelta = primary.close[executionIndex] - primary.close[sampleIndex]
            let movePoints = Double(abs(futureDelta))
            let label: LabelClass = futureDelta > 0 ? .buy : (futureDelta < 0 ? .sell : .skip)
            let trainRequest = TrainRequestV4(
                valid: true,
                context: payload.predictRequest.context,
                labelClass: label,
                movePoints: movePoints,
                sampleWeight: 1.0,
                windowSize: payload.predictRequest.windowSize,
                x: payload.predictRequest.x,
                xWindow: payload.predictRequest.xWindow
            )
            try trainRequest.validate()
            try runtime.train(trainRequest, hyperParameters: hyperParameters)
            flags |= prediction.confidence >= confidenceThreshold ? 0 : 1
        }

        broker.finish(price: primary.close[primary.count - 1])
        return BacktestPassResult(
            passIndex: parameters.combinationIndex,
            pluginIdentifier: descriptor.id,
            engine: .cpu,
            parameters: parameters.snapshots,
            netProfit: broker.netProfit,
            grossProfit: broker.grossProfit,
            grossLoss: broker.grossLoss,
            maxDrawdown: broker.maxDrawdown,
            totalTrades: broker.totalTrades,
            winningTrades: broker.winningTrades,
            losingTrades: broker.losingTrades,
            winRate: broker.winRate,
            profitFactor: broker.profitFactor,
            barsProcessed: primary.count,
            flags: flags,
            errorMessage: nil
        )
    }

    private static func tradeDirection(
        from prediction: PredictionV4,
        confidenceThreshold: Double,
        minMovePoints: Double
    ) -> TradeDirection? {
        let sell = prediction.classProbabilities[LabelClass.sell.rawValue]
        let buy = prediction.classProbabilities[LabelClass.buy.rawValue]
        let skip = prediction.classProbabilities[LabelClass.skip.rawValue]
        guard max(buy, sell) >= confidenceThreshold || prediction.confidence >= confidenceThreshold else {
            return nil
        }
        guard prediction.moveMeanPoints >= minMovePoints else {
            return nil
        }
        if buy > sell, buy >= skip {
            return .long
        }
        if sell > buy, sell >= skip {
            return .short
        }
        return nil
    }

    private static func marketUniverse(from backtestUniverse: OhlcMarketUniverse) throws -> MarketUniverse {
        let series = try backtestUniverse.symbols.map { symbol in
            guard let item = backtestUniverse[symbol] else {
                throw FXBacktestError.invalidMarketData("Missing market series for \(symbol).")
            }
            return try m1Series(from: item)
        }
        return try MarketUniverse(primarySymbol: backtestUniverse.primary.metadata.logicalSymbol, series: series)
    }

    private static func m1Series(from market: OhlcDataSeries) throws -> M1OHLCVSeries {
        try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: market.metadata.brokerSourceId,
                sourceOrigin: market.metadata.sourceOrigin,
                logicalSymbol: market.metadata.logicalSymbol,
                providerSymbol: market.metadata.mt5Symbol,
                timeframe: .m1,
                digits: market.metadata.digits,
                firstUTC: market.metadata.firstUtc,
                lastUTC: market.metadata.lastUtc
            ),
            utcTimestamps: market.utcTimestamps,
            open: market.open,
            high: market.high,
            low: market.low,
            close: market.close,
            volume: market.volume
        )
    }

    private static func parameterDefinitions(for runtime: FXAIAcceleratedPluginRuntime) throws -> [ParameterDefinition] {
        let catalog = FXBacktestPluginConfigurationCatalog.pluginConfigurations(plugins: [runtime])
        let firstScope = catalog.first { $0.pluginId == runtime.manifest.aiName } ?? catalog.first
        var definitions = try (firstScope?.parameters ?? []).map(Self.parameterDefinition(from:))
        definitions.append(try ParameterDefinition(
            key: "signal_confidence_threshold",
            displayName: "Signal Confidence Threshold",
            defaultValue: 0.55,
            defaultMinimum: 0.50,
            defaultStep: 0.05,
            defaultMaximum: 0.95,
            valueKind: .decimal
        ))
        definitions.append(try ParameterDefinition(
            key: "horizon_minutes",
            displayName: "Horizon Minutes",
            defaultValue: Double(runtime.manifest.minHorizonMinutes),
            defaultMinimum: Double(runtime.manifest.minHorizonMinutes),
            defaultStep: 1,
            defaultMaximum: Double(runtime.manifest.maxHorizonMinutes),
            valueKind: .integer
        ))
        definitions.append(try ParameterDefinition(
            key: "min_move_points",
            displayName: "Minimum Move Points",
            defaultValue: 0,
            defaultMinimum: 0,
            defaultStep: 0.1,
            defaultMaximum: 100,
            valueKind: .decimal
        ))
        definitions.append(try ParameterDefinition(
            key: "close_on_skip",
            displayName: "Close On Skip",
            defaultValue: 1,
            defaultMinimum: 0,
            defaultStep: 1,
            defaultMaximum: 1,
            valueKind: .boolean
        ))

        var seen: Set<String> = []
        return definitions.filter { seen.insert($0.key).inserted }
    }

    private static func parameterDefinition(from dto: FXBacktestConfigurationParameterDTO) throws -> ParameterDefinition {
        try ParameterDefinition(
            key: dto.key,
            displayName: dto.displayName,
            defaultValue: dto.defaultValue,
            defaultMinimum: dto.minimum,
            defaultStep: dto.step,
            defaultMaximum: dto.maximum,
            valueKind: valueKind(from: dto.valueKind)
        )
    }

    private static func valueKind(from dto: FXBacktestConfigurationValueKind) -> ParameterValueKind {
        switch dto {
        case .integer:
            return .integer
        case .decimal:
            return .decimal
        case .boolean:
            return .boolean
        }
    }

    private static func hyperParameters(from parameters: ParameterVector) -> HyperParameters {
        HyperParameters(
            learningRate: parameters["learning_rate"] ?? 0.01,
            l2: parameters["l2"] ?? 0.0001,
            ftrlAlpha: parameters["ftrl_alpha"] ?? 0.05,
            ftrlBeta: parameters["ftrl_beta"] ?? 1.0,
            ftrlL1: parameters["ftrl_l1"] ?? 0.0,
            ftrlL2: parameters["ftrl_l2"] ?? 0.0001,
            passiveAggressiveC: parameters["passive_aggressive_c"] ?? 0.5,
            passiveAggressiveMargin: parameters["passive_aggressive_margin"] ?? 0.05,
            xgbLearningRate: parameters["xgb_learning_rate"] ?? 0.05,
            xgbL2: parameters["xgb_l2"] ?? 0.001,
            xgbSplit: parameters["xgb_split"] ?? 0.0,
            mlpLearningRate: parameters["mlp_learning_rate"] ?? 0.01,
            mlpL2: parameters["mlp_l2"] ?? 0.0001,
            mlpInit: parameters["mlp_init"] ?? 0.05,
            quantileLearningRate: parameters["quantile_learning_rate"] ?? 0.01,
            quantileL2: parameters["quantile_l2"] ?? 0.0001,
            enhashLearningRate: parameters["enhash_learning_rate"] ?? 0.01,
            enhashL1: parameters["enhash_l1"] ?? 0.0,
            enhashL2: parameters["enhash_l2"] ?? 0.0001,
            tcnLayers: parameters["tcn_layers"] ?? 2,
            tcnKernel: parameters["tcn_kernel"] ?? 3,
            tcnDilationBase: parameters["tcn_dilation_base"] ?? 2
        )
    }

    private func horizonMinutes(_ parameters: ParameterVector) -> Int {
        let raw = parameters["horizon_minutes"] ?? Double(manifest.minHorizonMinutes)
        return min(max(Int(raw.rounded()), manifest.minHorizonMinutes), manifest.maxHorizonMinutes)
    }

    private static func displayName(from manifest: PluginManifestV4) -> String {
        manifest.aiName
            .split(separator: "_")
            .map { $0.uppercased() }
            .joined(separator: " ")
    }

    private static func summary(from manifest: PluginManifestV4, plan: FXPluginAccelerationPlan) -> String {
        let backends = plan.declaredBackends.map(\.rawValue).joined(separator: ", ")
        return "Root FXPlugins zoo adapter for \(manifest.aiName), family \(manifest.family), declared backends: \(backends)."
    }
}
