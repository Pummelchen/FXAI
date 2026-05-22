import FXDataEngine
import Foundation

public struct FXStupidFXDataEnginePlugin: FXAIPlannedPlugin {
    public var manifest: PluginManifestV4 {
        PluginManifestV4(
            aiID: AIModelID.demoFXStupid.rawValue,
            aiName: "fxbacktest_fxstupid",
            family: .ruleBased,
            referenceTier: .ruleBaseline,
            capabilityMask: [.selfTest, .multiHorizon],
            featureGroups: [.price, .volume, .microstructure],
            minHorizonMinutes: 1,
            maxHorizonMinutes: 240,
            minSequenceBars: 1,
            maxSequenceBars: 1,
            requiresVolumeWhenAvailable: true
        )
    }

    public var accelerationPlan: FXPluginAccelerationPlan {
        FXPluginAccelerationPlan(
            pluginName: "fxbacktest_fxstupid",
            primaryBackends: [.swiftScalar],
            usesVolumeWhenAvailable: true,
            notes: "FXBacktest demo adapter for the converted FXStupid EA. Keep scalar because the original flow is stateful order control; use volume as a confidence input when FXDataEngine has volume."
        )
    }

    public init() {}

    public mutating func reset() {}

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)

        let shortReturn = FXBacktestDemoPredictionTools.feature(request, 0)
        let slope = FXBacktestDemoPredictionTools.feature(request, 3)
        let volumeSignal = request.context.dataHasVolume ? FXBacktestDemoPredictionTools.feature(request, 6) : 0.0
        let volumeBias = request.context.dataHasVolume ? 0.10 * fxClamp(volumeSignal, -1.0, 1.0) : 0.0
        let edge = shortReturn + 0.50 * slope + volumeBias
        let strength = fxClamp(abs(edge) * 3.0 + abs(volumeBias), 0.0, 1.0)
        let moveSeed = max(request.context.minMovePoints, request.context.priceCostPoints, abs(edge) * 80.0)

        if edge > 0.01 {
            return FXBacktestDemoPredictionTools.directionalPrediction(label: .buy, strength: strength, moveSeed: moveSeed, reliability: 0.52 + abs(volumeBias))
        }
        if edge < -0.01 {
            return FXBacktestDemoPredictionTools.directionalPrediction(label: .sell, strength: strength, moveSeed: moveSeed, reliability: 0.52 + abs(volumeBias))
        }
        return FXBacktestDemoPredictionTools.directionalPrediction(label: .skip, strength: 0.0, moveSeed: 0.0, reliability: 0.48)
    }
}
