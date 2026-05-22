import Foundation

public struct NormalizationCoreFrame: Sendable {
    public let valid: Bool
    public let horizonMinutes: Int
    public let normalizationMethod: FeatureNormalizationMethod
    public let sampleTimeUTC: Int64
    public let hasVolume: Bool
    public let normalized: [Double]
    public let modelInput: [Double]

    public init(
        valid: Bool,
        horizonMinutes: Int,
        normalizationMethod: FeatureNormalizationMethod,
        sampleTimeUTC: Int64,
        hasVolume: Bool,
        normalized: [Double],
        modelInput: [Double]
    ) {
        self.valid = valid
        self.horizonMinutes = horizonMinutes
        self.normalizationMethod = normalizationMethod
        self.sampleTimeUTC = sampleTimeUTC
        self.hasVolume = hasVolume
        self.normalized = normalized
        self.modelInput = modelInput
    }
}

public struct NormalizationPayloadRequest: Sendable {
    public var valid: Bool
    public var featureSchema: FeatureSchema
    public var featureGroups: FeatureGroupMask
    public var normalizationMethod: FeatureNormalizationMethod
    public var horizonMinutes: Int
    public var sequenceBars: Int
    public var sampleTimeUTC: Int64
    public var windowSize: Int
    public var x: [Double]
    public var xWindow: [[Double]]

    public init(
        valid: Bool,
        featureSchema: FeatureSchema = .full,
        featureGroups: FeatureGroupMask = .all,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        horizonMinutes: Int = 1,
        sequenceBars: Int = 1,
        sampleTimeUTC: Int64 = 0,
        windowSize: Int = 0,
        x: [Double],
        xWindow: [[Double]] = []
    ) {
        self.valid = valid
        self.featureSchema = featureSchema
        self.featureGroups = featureGroups
        self.normalizationMethod = normalizationMethod
        self.horizonMinutes = max(1, horizonMinutes)
        self.sequenceBars = min(max(1, sequenceBars), FXDataEngineConstants.maxSequenceBars)
        self.sampleTimeUTC = sampleTimeUTC
        self.windowSize = min(max(0, windowSize), FXDataEngineConstants.maxSequenceBars)
        self.x = x
        self.xWindow = xWindow
    }
}

public struct NormalizationPayloadFrame: Sendable {
    public let valid: Bool
    public let featureSchema: FeatureSchema
    public let featureGroups: FeatureGroupMask
    public let normalizationMethod: FeatureNormalizationMethod
    public let horizonMinutes: Int
    public let sequenceBars: Int
    public let sampleTimeUTC: Int64
    public let windowSize: Int
    public let x: [Double]
    public let xWindow: [[Double]]
}

public struct NormalizationCore: Sendable {
    private let schemaPolicy: FeatureSchemaPolicy

    public init(schemaPolicy: FeatureSchemaPolicy = FeatureSchemaPolicy()) {
        self.schemaPolicy = schemaPolicy
    }

    public func buildInputFrame(from featureFrame: FeatureCoreFrame) throws -> NormalizationCoreFrame {
        guard featureFrame.valid else {
            throw FXDataEngineError.invalidRequest("feature frame is not valid")
        }
        let normalized = applyFeatureNormalization(
            method: featureFrame.normalizationMethod,
            raw: featureFrame.raw,
            previous: featureFrame.previous,
            hasPrevious: featureFrame.hasPrevious
        )
        let input = schemaPolicy.modelInput(from: normalized)
        return NormalizationCoreFrame(
            valid: true,
            horizonMinutes: featureFrame.horizonMinutes,
            normalizationMethod: featureFrame.normalizationMethod,
            sampleTimeUTC: featureFrame.sampleTimeUTC,
            hasVolume: featureFrame.hasVolume,
            normalized: normalized,
            modelInput: input
        )
    }

    public func buildPayloadFrame(_ request: NormalizationPayloadRequest) throws -> NormalizationPayloadFrame {
        guard request.valid else {
            throw FXDataEngineError.invalidRequest("payload request is not valid")
        }

        let sanitizedX = sanitizeInputVector(request.x)
        let maskedX = schemaPolicy.apply(schema: request.featureSchema, groups: request.featureGroups, to: sanitizedX)
        let trimmedWindow = Array(request.xWindow.prefix(request.windowSize))
            .map { schemaPolicy.apply(schema: request.featureSchema, groups: request.featureGroups, to: sanitizeInputVector($0)) }
        return NormalizationPayloadFrame(
            valid: true,
            featureSchema: request.featureSchema,
            featureGroups: request.featureGroups,
            normalizationMethod: request.normalizationMethod,
            horizonMinutes: request.horizonMinutes,
            sequenceBars: request.sequenceBars,
            sampleTimeUTC: request.sampleTimeUTC,
            windowSize: trimmedWindow.count,
            x: maskedX,
            xWindow: trimmedWindow
        )
    }

    public func finalizePredictRequest(manifest: PluginManifestV4, request: PredictRequestV4) throws -> PredictRequestV4 {
        let payload = try buildPayloadFrame(NormalizationPayloadRequest(
            valid: request.valid,
            featureSchema: manifest.featureSchema,
            featureGroups: manifest.featureGroups,
            normalizationMethod: request.context.normalizationMethod,
            horizonMinutes: request.context.horizonMinutes,
            sequenceBars: request.context.sequenceBars,
            sampleTimeUTC: request.context.sampleTimeUTC,
            windowSize: request.windowSize,
            x: request.x,
            xWindow: request.xWindow
        ))
        return request.replacingPayload(x: payload.x, xWindow: payload.xWindow)
    }

    public func finalizeTrainRequest(manifest: PluginManifestV4, request: TrainRequestV4) throws -> TrainRequestV4 {
        let payload = try buildPayloadFrame(NormalizationPayloadRequest(
            valid: request.valid,
            featureSchema: manifest.featureSchema,
            featureGroups: manifest.featureGroups,
            normalizationMethod: request.context.normalizationMethod,
            horizonMinutes: request.context.horizonMinutes,
            sequenceBars: request.context.sequenceBars,
            sampleTimeUTC: request.context.sampleTimeUTC,
            windowSize: request.windowSize,
            x: request.x,
            xWindow: request.xWindow
        ))
        return request.replacingPayload(x: payload.x, xWindow: payload.xWindow)
    }

    private func applyFeatureNormalization(
        method: FeatureNormalizationMethod,
        raw: [Double],
        previous: [Double],
        hasPrevious: Bool
    ) -> [Double] {
        var out = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        for index in 0..<FXDataEngineConstants.aiFeatures {
            let current = index < raw.count ? fxSafeFinite(raw[index]) : 0.0
            let prior = index < previous.count ? fxSafeFinite(previous[index]) : 0.0
            let value: Double
            switch method {
            case .existing:
                value = current
            case .minMaxBuffer5:
                value = bufferedUnit(current, buffer: 0.05)
            case .minMaxBuffer2:
                value = bufferedUnit(current, buffer: 0.02)
            case .minMaxBuffer3:
                value = bufferedUnit(current, buffer: 0.03)
            case .changePercent:
                value = hasPrevious ? (current - prior) / max(abs(prior), 1e-9) : 0.0
            case .relativeChangePercent:
                value = hasPrevious ? (current - prior) / max((abs(current) + abs(prior)) * 0.5, 1e-9) : 0.0
            case .binary01:
                value = hasPrevious ? (current > prior ? 1.0 : 0.0) : 0.0
            case .logReturn:
                value = hasPrevious && current > -0.999 && prior > -0.999 ? log1p(current) - log1p(prior) : 0.0
            case .candleGeometry, .volatilityStdReturns, .atrNatrUnit:
                value = current
            case .zScore, .robustMedianIQR, .quantileToNormal, .powerYeoJohnson, .revin, .dain:
                value = current
            }
            out[index] = fxClampSignedUnit(value)
        }
        return out
    }

    private func bufferedUnit(_ value: Double, buffer: Double) -> Double {
        let buffered = (value + 1.0 + buffer) / (2.0 + buffer * 2.0)
        return fxClamp(buffered, 0.0, 1.0)
    }

    private func sanitizeInputVector(_ vector: [Double]) -> [Double] {
        var out = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        out[0] = 1.0
        let count = min(vector.count, FXDataEngineConstants.aiWeights)
        for index in 1..<count {
            out[index] = fxSafeFinite(vector[index])
        }
        return out
    }
}
