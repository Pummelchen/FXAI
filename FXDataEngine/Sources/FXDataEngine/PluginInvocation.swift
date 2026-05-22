import Foundation

public enum PluginInvocationTools {
    public static func trainViaV4<Plugin: FXAIPluginV4>(
        plugin: inout Plugin,
        request: TrainRequestV4,
        hyperParameters: HyperParameters,
        performance: inout RuntimePerformanceState,
        measuredElapsedMS: Double? = nil
    ) throws {
        let manifest = plugin.manifest
        try manifest.validate()
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)

        let sequenceBars = max(request.windowSize + 1, request.context.sequenceBars)
        performance.setPluginWorkingSetKB(
            aiID: manifest.aiID,
            workingSetKB: RuntimePerformanceState.estimatePluginWorkingSetKB(
                manifest: manifest,
                sequenceBars: sequenceBars
            )
        )

        let start = DispatchTime.now().uptimeNanoseconds
        defer {
            performance.recordPluginUpdate(
                aiID: manifest.aiID,
                elapsedMS: measuredElapsedMS ?? elapsedMilliseconds(since: start),
                sampleTimeUTC: request.context.sampleTimeUTC
            )
        }
        try plugin.train(request, hyperParameters: hyperParameters)
    }

    public static func predictViaV4<Plugin: FXAIPluginV4>(
        plugin: Plugin,
        request: PredictRequestV4,
        hyperParameters: HyperParameters,
        performance: inout RuntimePerformanceState,
        measuredElapsedMS: Double? = nil
    ) throws -> PredictionV4 {
        let manifest = plugin.manifest
        try manifest.validate()
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)

        let sequenceBars = max(request.windowSize + 1, request.context.sequenceBars)
        performance.setPluginWorkingSetKB(
            aiID: manifest.aiID,
            workingSetKB: RuntimePerformanceState.estimatePluginWorkingSetKB(
                manifest: manifest,
                sequenceBars: sequenceBars
            )
        )

        let start = DispatchTime.now().uptimeNanoseconds
        defer {
            performance.recordPluginPredict(
                aiID: manifest.aiID,
                elapsedMS: measuredElapsedMS ?? elapsedMilliseconds(since: start),
                sampleTimeUTC: request.context.sampleTimeUTC
            )
        }
        return try plugin.predict(request, hyperParameters: hyperParameters)
    }

    private static func elapsedMilliseconds(since start: UInt64) -> Double {
        let end = DispatchTime.now().uptimeNanoseconds
        guard end >= start else { return 0.0 }
        return Double(end - start) / 1_000_000.0
    }
}
