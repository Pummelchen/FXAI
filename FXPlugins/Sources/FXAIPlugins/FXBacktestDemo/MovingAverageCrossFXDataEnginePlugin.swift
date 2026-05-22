import FXBacktestCore
import FXBacktestPlugins
import FXDataEngine
import Foundation

public struct MovingAverageCrossFXDataEnginePlugin: FXAIPlannedPlugin {
    private let backtestPlugin = MovingAverageCross()

    public var manifest: PluginManifestV4 {
        PluginManifestV4(
            aiID: AIModelID.demoMovingAverageCross.rawValue,
            aiName: "fxbacktest_moving_average_cross",
            family: .ruleBased,
            referenceTier: .ruleBaseline,
            capabilityMask: [.selfTest, .multiHorizon],
            featureGroups: [.price, .multiTimeframe, .volume],
            minHorizonMinutes: 1,
            maxHorizonMinutes: 240,
            minSequenceBars: 1,
            maxSequenceBars: 1,
            requiresVolumeWhenAvailable: true
        )
    }

    public var accelerationPlan: FXPluginAccelerationPlan {
        FXPluginAccelerationPlan(
            pluginName: "fxbacktest_moving_average_cross",
            primaryBackends: [.swiftScalar, .metal],
            candidateBackends: [.swiftSIMD],
            usesVolumeWhenAvailable: true,
            notes: "FXBacktest demo adapter. The original backtest plugin already declares a Metal kernel for parameter sweeps; FXDataEngine prediction uses price/multi-timeframe features plus volume confidence when available."
        )
    }

    public init() {}

    public mutating func reset() {}

    public func selfTest() -> Bool {
        backtestPlugin.descriptor.supportsMetal && backtestPlugin.metalKernel != nil && (try? manifest.validate()) != nil
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)

        let fastReturn = FXBacktestDemoPredictionTools.feature(request, 7)
        let slowReturn = FXBacktestDemoPredictionTools.feature(request, 8)
        let edge = fastReturn - slowReturn
        let volumeSignal = request.context.dataHasVolume ? FXBacktestDemoPredictionTools.feature(request, 6) : 0.0
        let volumeBoost = request.context.dataHasVolume ? 0.10 * fxClamp(abs(volumeSignal), 0.0, 1.0) : 0.0
        let strength = fxClamp(abs(edge) * 4.0 + volumeBoost, 0.0, 1.0)
        let moveSeed = max(request.context.minMovePoints, request.context.priceCostPoints, abs(edge) * 100.0)

        if edge > 1.0e-9 {
            return FXBacktestDemoPredictionTools.directionalPrediction(label: .buy, strength: strength, moveSeed: moveSeed, reliability: 0.60 + volumeBoost)
        }
        if edge < -1.0e-9 {
            return FXBacktestDemoPredictionTools.directionalPrediction(label: .sell, strength: strength, moveSeed: moveSeed, reliability: 0.60 + volumeBoost)
        }
        return FXBacktestDemoPredictionTools.directionalPrediction(label: .skip, strength: 0.0, moveSeed: 0.0, reliability: 0.50)
    }
}
