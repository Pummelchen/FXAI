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
