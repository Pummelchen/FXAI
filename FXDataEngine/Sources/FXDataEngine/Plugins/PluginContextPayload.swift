import Foundation

public struct PluginContextPayloadState: Codable, Hashable, Sendable {
    public var context: PluginContextV4
    public private(set) var contextTimeReady: Bool
    public private(set) var contextPriceCostReady: Bool
    public private(set) var windowSize: Int
    public private(set) var xWindow: [[Double]]

    public init(
        context: PluginContextV4? = nil,
        contextTimeReady: Bool? = nil,
        contextPriceCostReady: Bool? = nil,
        windowSize: Int? = nil,
        xWindow: [[Double]] = [],
        pointValueFallback: Double = 1.0,
        sampleTimeFallbackUTC: Int64 = 0,
        symbolFallback: String = ""
    ) {
        let rawContext = context ?? PluginContextV4()
        self.context = PluginContextPayloadTools.sanitizedContext(
            rawContext,
            pointValueFallback: pointValueFallback,
            sampleTimeFallbackUTC: sampleTimeFallbackUTC,
            symbolFallback: symbolFallback
        )
        self.contextTimeReady = contextTimeReady ?? (rawContext.sampleTimeUTC > 0 || sampleTimeFallbackUTC > 0)
        self.contextPriceCostReady = contextPriceCostReady ??
            (context != nil && rawContext.priceCostPoints.isFinite && rawContext.priceCostPoints >= 0.0)
        let payload = PluginContextPayloadTools.sanitizedWindow(
            xWindow,
            declaredWindowSize: windowSize ?? xWindow.count
        )
        self.windowSize = payload.windowSize
        self.xWindow = payload.window
    }

    public mutating func setContext(
        _ context: PluginContextV4,
        pointValueFallback: Double = 1.0,
        sampleTimeFallbackUTC: Int64 = 0,
        symbolFallback: String = ""
    ) {
        self.context = PluginContextPayloadTools.sanitizedContext(
            context,
            pointValueFallback: pointValueFallback,
            sampleTimeFallbackUTC: sampleTimeFallbackUTC,
            symbolFallback: symbolFallback
        )
        contextTimeReady = context.sampleTimeUTC > 0 || sampleTimeFallbackUTC > 0
        contextPriceCostReady = context.priceCostPoints.isFinite && context.priceCostPoints >= 0.0
    }

    public mutating func setWindowPayload(
        windowSize: Int,
        xWindow: [[Double]]
    ) {
        let payload = PluginContextPayloadTools.sanitizedWindow(
            xWindow,
            declaredWindowSize: windowSize
        )
        self.windowSize = payload.windowSize
        self.xWindow = payload.window
    }

    public mutating func clearWindowPayload() {
        windowSize = 0
        xWindow = []
    }

    public func resolvePriceCostPoints(x: [Double], preferContext: Bool = true) -> Double {
        PluginContextPayloadTools.resolvedPriceCostPoints(
            contextPriceCostPoints: preferContext && contextPriceCostReady ? context.priceCostPoints : nil,
            x: x
        )
    }

    public func resolveMinMovePoints() -> Double {
        PluginContextPayloadTools.resolvedMinMovePoints(context.minMovePoints)
    }

    public func resolvePointValue(fallback: Double = 1.0) -> Double {
        PluginContextPayloadTools.resolvedPointValue(context.pointValue, fallback: fallback)
    }

    public func resolveContextTimeUTC(fallback: Int64 = 0) -> Int64 {
        PluginContextPayloadTools.resolvedContextTimeUTC(contextTimeReady ? context.sampleTimeUTC : 0, fallback: fallback)
    }

    public func buildSharedAdapterInput(x: [Double]) -> [Double] {
        PluginSharedTransferPayloadTools.buildInput(
            x: x,
            window: xWindow,
            declaredWindowSize: windowSize,
            domainHash: context.domainHash,
            horizonMinutes: context.horizonMinutes
        )
    }

    public func hasSharedAdapterSignal(_ adapterInput: [Double]) -> Bool {
        PluginTransferSupportTools.hasSharedAdapterSignal(adapterInput)
    }

    public func sharedAdapterSignalStrength(_ adapterInput: [Double]) -> Double {
        PluginTransferSupportTools.sharedAdapterSignalStrength(adapterInput)
    }
}

public enum PluginContextPayloadTools {
    public static func sanitizedContext(
        _ context: PluginContextV4,
        pointValueFallback: Double = 1.0,
        sampleTimeFallbackUTC: Int64 = 0,
        symbolFallback: String = ""
    ) -> PluginContextV4 {
        let sampleTimeUTC = resolvedContextTimeUTC(context.sampleTimeUTC, fallback: sampleTimeFallbackUTC)
        let regimeID = (0..<FXDataEngineConstants.pluginRegimeBuckets).contains(context.regimeID) ? context.regimeID : 0
        let sessionBucket: Int
        if (0..<FXDataEngineConstants.pluginSessionBuckets).contains(context.sessionBucket) {
            sessionBucket = context.sessionBucket
        } else {
            sessionBucket = PluginContractTools.deriveSessionBucket(timestampUTC: sampleTimeUTC)
        }

        let domainHash: Double
        if context.domainHash.isFinite, (0.0...1.0).contains(context.domainHash) {
            domainHash = context.domainHash
        } else if !symbolFallback.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            domainHash = PluginContractTools.symbolHash01(symbolFallback)
        } else {
            domainHash = 0.0
        }

        return PluginContextV4(
            apiVersion: context.apiVersion,
            regimeID: regimeID,
            sessionBucket: sessionBucket,
            horizonMinutes: max(1, context.horizonMinutes),
            featureSchema: context.featureSchema,
            normalizationMethod: context.normalizationMethod,
            sequenceBars: min(max(1, context.sequenceBars), FXDataEngineConstants.maxSequenceBars),
            priceCostPoints: max(0.0, fxSafeFinite(context.priceCostPoints)),
            minMovePoints: resolvedMinMovePoints(context.minMovePoints),
            pointValue: resolvedPointValue(context.pointValue, fallback: pointValueFallback),
            domainHash: domainHash,
            sampleTimeUTC: sampleTimeUTC,
            dataHasVolume: context.dataHasVolume,
            tokenizerContract: context.tokenizerContract,
            textEvents: context.textEvents
        )
    }

    public static func sanitizedWindow(
        _ window: [[Double]],
        declaredWindowSize: Int? = nil
    ) -> (window: [[Double]], windowSize: Int) {
        let declared = declaredWindowSize ?? window.count
        let windowSize = min(max(0, declared), FXDataEngineConstants.maxSequenceBars)
        guard windowSize > 0 else {
            return ([], 0)
        }

        var sanitized: [[Double]] = []
        sanitized.reserveCapacity(windowSize)
        for index in 0..<windowSize {
            let source = index < window.count ? window[index] : []
            sanitized.append(sanitizedInputVector(source))
        }
        return (sanitized, windowSize)
    }

    public static func sanitizedInputVector(_ values: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<min(values.count, FXDataEngineConstants.aiWeights) {
            output[index] = fxSafeFinite(values[index])
        }
        return output
    }

    public static func resolvedPriceCostPoints(
        contextPriceCostPoints: Double?,
        x: [Double]
    ) -> Double {
        PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: contextPriceCostPoints)
    }

    public static func resolvedMinMovePoints(_ value: Double) -> Double {
        let safeValue = fxSafeFinite(value)
        return safeValue > 0.0 ? safeValue : 0.0
    }

    public static func resolvedPointValue(_ value: Double, fallback: Double = 1.0) -> Double {
        let safeValue = fxSafeFinite(value)
        if safeValue > 0.0 {
            return safeValue
        }
        let safeFallback = fxSafeFinite(fallback)
        return safeFallback > 0.0 ? safeFallback : 1.0
    }

    public static func resolvedContextTimeUTC(_ value: Int64, fallback: Int64 = 0) -> Int64 {
        if value > 0 {
            return value
        }
        return max(0, fallback)
    }
}
