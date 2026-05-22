import Foundation

public enum MLFramework: String, Codable, Hashable, Sendable {
    case nativeSwift
    case metal
    case pyTorch
    case tensorFlow
}

public enum MLBackendMode: Codable, Hashable, Sendable {
    case inProcess(MLFramework)
    case externalPython(framework: MLFramework, executable: String, module: String)
}

public struct MLBackendDescriptor: Codable, Hashable, Sendable {
    public let mode: MLBackendMode
    public let modelIdentifier: String
    public let supportsTraining: Bool
    public let supportsInference: Bool
    public let usesVolumeFeatures: Bool

    public init(
        mode: MLBackendMode,
        modelIdentifier: String,
        supportsTraining: Bool = true,
        supportsInference: Bool = true,
        usesVolumeFeatures: Bool = true
    ) {
        self.mode = mode
        self.modelIdentifier = modelIdentifier
        self.supportsTraining = supportsTraining
        self.supportsInference = supportsInference
        self.usesVolumeFeatures = usesVolumeFeatures
    }
}

public struct MLInferencePayload: Codable, Hashable, Sendable {
    public let apiVersion: Int
    public let modelIdentifier: String
    public let framework: MLFramework
    public let dataHasVolume: Bool
    public let x: [Double]
    public let xWindow: [[Double]]

    public init(
        apiVersion: Int = FXDataEngineConstants.apiVersionV4,
        modelIdentifier: String,
        framework: MLFramework,
        dataHasVolume: Bool,
        x: [Double],
        xWindow: [[Double]]
    ) {
        self.apiVersion = apiVersion
        self.modelIdentifier = modelIdentifier
        self.framework = framework
        self.dataHasVolume = dataHasVolume
        self.x = x
        self.xWindow = xWindow
    }
}

public struct MLTrainingPayload: Codable, Hashable, Sendable {
    public let inference: MLInferencePayload
    public let labelClass: LabelClass
    public let movePoints: Double
    public let sampleWeight: Double
    public let nextVolumeTarget: Double

    public init(inference: MLInferencePayload, request: TrainRequestV4) {
        self.inference = inference
        self.labelClass = request.labelClass
        self.movePoints = request.movePoints
        self.sampleWeight = request.sampleWeight
        self.nextVolumeTarget = request.nextVolumeTarget
    }
}

public struct MLTensorContextDescriptor: Codable, Hashable, Sendable {
    public var modelDim: Int
    public var hiddenDim: Int
    public var headCount: Int
    public var headDim: Int
    public var sequenceCapacity: Int
    public var stride: Int
    public var patchSize: Int
    public var dilation: Int
    public var positionStepPenalty: Double

    public init(
        modelDim: Int,
        hiddenDim: Int,
        headCount: Int,
        sequenceCapacity: Int,
        stride: Int = 1,
        patchSize: Int = 1,
        dilation: Int = 1,
        positionStepPenalty: Double = 0.06
    ) {
        self.modelDim = min(max(modelDim, 1), FXDataEngineConstants.aiFeatures)
        self.hiddenDim = min(max(hiddenDim, 1), FXDataEngineConstants.aiMLPHidden)
        self.headCount = min(max(headCount, 1), FXDataEngineConstants.aiFeatures)
        self.headDim = max(self.modelDim / self.headCount, 1)
        self.sequenceCapacity = min(max(sequenceCapacity, 1), FXDataEngineConstants.maxSequenceBars)
        self.stride = max(stride, 1)
        self.patchSize = max(patchSize, 1)
        self.dilation = max(dilation, 1)
        self.positionStepPenalty = fxClamp(positionStepPenalty, 0.0, 2.0)
    }
}

public struct MLSequenceRuntimeDescriptor: Codable, Hashable, Sendable {
    public var maxSteps: Int
    public var stride: Int
    public var patchSize: Int
    public var normalize: Bool
    public var includeCurrent: Bool
    public var positionStepPenalty: Double

    public init(
        maxSteps: Int,
        stride: Int = 1,
        patchSize: Int = 1,
        normalize: Bool = true,
        includeCurrent: Bool = true,
        positionStepPenalty: Double = 0.06
    ) {
        self.maxSteps = min(max(maxSteps, 1), FXDataEngineConstants.maxSequenceBars)
        self.stride = max(stride, 1)
        self.patchSize = max(patchSize, 1)
        self.normalize = normalize
        self.includeCurrent = includeCurrent
        self.positionStepPenalty = fxClamp(positionStepPenalty, 0.0, 2.0)
    }
}

public enum MLTensorBridgeTools {
    public static func clippedCurrentInput(_ x: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<FXDataEngineConstants.aiWeights {
            let value = index < x.count ? fxSafeFinite(x[index]) : 0.0
            output[index] = index == 0 ? 1.0 : PluginSupportTools.clipSymmetric(value, limit: 8.0)
        }
        return output
    }

    public static func contextDescriptor(
        style: SequenceStyle,
        maxSteps: Int = FXDataEngineConstants.maxSequenceBars,
        horizonMinutes: Int = 1
    ) -> MLTensorContextDescriptor {
        let cap = max(min(maxSteps, FXDataEngineConstants.maxSequenceBars), 4)
        var modelDim = 16
        var heads = 2
        var stride = 1
        var patch = 1
        var dilation = 1
        var positionPenalty = 0.06

        switch style {
        case .recurrent:
            modelDim = min(18, FXDataEngineConstants.aiFeatures)
            heads = 2
        case .convolutional:
            modelDim = min(18, FXDataEngineConstants.aiFeatures)
            patch = 2
        case .transformer:
            modelDim = min(24, FXDataEngineConstants.aiFeatures)
            heads = 4
            patch = cap >= 24 ? 2 : 1
        case .stateSpace:
            modelDim = min(20, FXDataEngineConstants.aiFeatures)
            dilation = 2
        case .world:
            modelDim = min(22, FXDataEngineConstants.aiFeatures)
            heads = 4
            stride = cap >= 32 ? 2 : 1
            positionPenalty = 0.04
        case .generic:
            modelDim = min(16, FXDataEngineConstants.aiFeatures)
        }

        if horizonMinutes >= 60 {
            stride = max(stride, 2)
        }
        return MLTensorContextDescriptor(
            modelDim: modelDim,
            hiddenDim: FXDataEngineConstants.aiMLPHidden,
            headCount: heads,
            sequenceCapacity: cap,
            stride: stride,
            patchSize: patch,
            dilation: dilation,
            positionStepPenalty: positionPenalty
        )
    }

    public static func sequenceRuntimeDescriptor(
        dims: MLTensorContextDescriptor,
        normalize: Bool = true,
        includeCurrent: Bool = true
    ) -> MLSequenceRuntimeDescriptor {
        MLSequenceRuntimeDescriptor(
            maxSteps: dims.sequenceCapacity,
            stride: dims.stride,
            patchSize: dims.patchSize,
            normalize: normalize,
            includeCurrent: includeCurrent,
            positionStepPenalty: dims.positionStepPenalty
        )
    }
}

public protocol ExternalMLBackend: Sendable {
    var descriptor: MLBackendDescriptor { get }
    func predict(_ payload: MLInferencePayload) async throws -> PredictionV4
    func train(_ payload: MLTrainingPayload) async throws
}

public struct PythonMLBackendBridge: ExternalMLBackend {
    public let descriptor: MLBackendDescriptor

    public init(framework: MLFramework, executable: String = "python3", module: String, modelIdentifier: String) {
        precondition(framework == .pyTorch || framework == .tensorFlow, "Python bridge supports PyTorch or TensorFlow")
        self.descriptor = MLBackendDescriptor(
            mode: .externalPython(framework: framework, executable: executable, module: module),
            modelIdentifier: modelIdentifier
        )
    }

    public func predict(_ payload: MLInferencePayload) async throws -> PredictionV4 {
        throw FXDataEngineError.externalBackend("Python \(payload.framework.rawValue) bridge is declared but not wired to a process runner yet")
    }

    public func train(_ payload: MLTrainingPayload) async throws {
        throw FXDataEngineError.externalBackend("Python \(payload.inference.framework.rawValue) bridge is declared but not wired to a process runner yet")
    }
}

public enum MLBackendFactory {
    public static func inferencePayload(
        descriptor: MLBackendDescriptor,
        request: PredictRequestV4
    ) -> MLInferencePayload {
        let framework: MLFramework
        switch descriptor.mode {
        case .inProcess(let resolved):
            framework = resolved
        case .externalPython(let resolved, _, _):
            framework = resolved
        }
        return MLInferencePayload(
            modelIdentifier: descriptor.modelIdentifier,
            framework: framework,
            dataHasVolume: request.context.dataHasVolume,
            x: request.x,
            xWindow: request.xWindow
        )
    }
}
