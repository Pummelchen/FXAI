import Foundation

public enum MLFramework: String, Codable, Hashable, Sendable {
    case nativeSwift
    case metal
    case pyTorch
    case tensorFlow
    case foundationNLP
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
    public let horizonMinutes: Int
    public let sequenceBars: Int
    public let priceCostPoints: Double
    public let minMovePoints: Double
    public let x: [Double]
    public let xWindow: [[Double]]

    public init(
        apiVersion: Int = FXDataEngineConstants.apiVersionV4,
        modelIdentifier: String,
        framework: MLFramework,
        dataHasVolume: Bool,
        horizonMinutes: Int = 1,
        sequenceBars: Int = 1,
        priceCostPoints: Double = 0.0,
        minMovePoints: Double = 0.0,
        x: [Double],
        xWindow: [[Double]]
    ) {
        self.apiVersion = apiVersion
        self.modelIdentifier = modelIdentifier
        self.framework = framework
        self.dataHasVolume = dataHasVolume
        self.horizonMinutes = max(1, horizonMinutes)
        self.sequenceBars = min(max(1, sequenceBars), FXDataEngineConstants.maxSequenceBars)
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.x = x
        self.xWindow = xWindow
    }

    private enum CodingKeys: String, CodingKey {
        case apiVersion
        case modelIdentifier
        case framework
        case dataHasVolume
        case horizonMinutes
        case sequenceBars
        case priceCostPoints
        case minMovePoints
        case x
        case xWindow
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            apiVersion: try container.decodeIfPresent(Int.self, forKey: .apiVersion) ?? FXDataEngineConstants.apiVersionV4,
            modelIdentifier: try container.decode(String.self, forKey: .modelIdentifier),
            framework: try container.decode(MLFramework.self, forKey: .framework),
            dataHasVolume: try container.decodeIfPresent(Bool.self, forKey: .dataHasVolume) ?? false,
            horizonMinutes: try container.decodeIfPresent(Int.self, forKey: .horizonMinutes) ?? 1,
            sequenceBars: try container.decodeIfPresent(Int.self, forKey: .sequenceBars) ?? 1,
            priceCostPoints: try container.decodeIfPresent(Double.self, forKey: .priceCostPoints) ?? 0.0,
            minMovePoints: try container.decodeIfPresent(Double.self, forKey: .minMovePoints) ?? 0.0,
            x: try container.decode([Double].self, forKey: .x),
            xWindow: try container.decodeIfPresent([[Double]].self, forKey: .xWindow) ?? []
        )
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
    private let executable: String
    private let module: String
    private let environment: [String: String]

    public init(
        framework: MLFramework,
        executable: String = "python3",
        module: String,
        modelIdentifier: String,
        environment: [String: String] = [:]
    ) {
        precondition(
            framework == .pyTorch || framework == .tensorFlow || framework == .foundationNLP,
            "Python bridge supports PyTorch, TensorFlow, or Foundation NLP"
        )
        self.executable = executable
        self.module = module
        self.environment = environment
        self.descriptor = MLBackendDescriptor(
            mode: .externalPython(framework: framework, executable: executable, module: module),
            modelIdentifier: modelIdentifier
        )
    }

    public func predict(_ payload: MLInferencePayload) async throws -> PredictionV4 {
        let command = PythonMLBackendCommand(operation: "predict", inference: payload, training: nil)
        let response = try await run(command)
        guard response.ok else {
            throw FXDataEngineError.externalBackend(response.error ?? "Python backend returned failure")
        }
        if let error = response.error {
            throw FXDataEngineError.externalBackend(error)
        }
        guard let prediction = response.prediction else {
            throw FXDataEngineError.externalBackend("Python backend did not return a prediction")
        }
        try prediction.validate()
        return prediction
    }

    public func train(_ payload: MLTrainingPayload) async throws {
        let command = PythonMLBackendCommand(operation: "train", inference: nil, training: payload)
        let response = try await run(command)
        guard response.ok else {
            throw FXDataEngineError.externalBackend(response.error ?? "Python backend returned failure")
        }
        if let error = response.error {
            throw FXDataEngineError.externalBackend(error)
        }
    }

    public func predictSynchronously(_ payload: MLInferencePayload) throws -> PredictionV4 {
        let command = PythonMLBackendCommand(operation: "predict", inference: payload, training: nil)
        let response = try runSynchronously(command)
        guard response.ok else {
            throw FXDataEngineError.externalBackend(response.error ?? "Python backend returned failure")
        }
        if let error = response.error {
            throw FXDataEngineError.externalBackend(error)
        }
        guard let prediction = response.prediction else {
            throw FXDataEngineError.externalBackend("Python backend did not return a prediction")
        }
        try prediction.validate()
        return prediction
    }

    public func trainSynchronously(_ payload: MLTrainingPayload) throws {
        let command = PythonMLBackendCommand(operation: "train", inference: nil, training: payload)
        let response = try runSynchronously(command)
        guard response.ok else {
            throw FXDataEngineError.externalBackend(response.error ?? "Python backend returned failure")
        }
        if let error = response.error {
            throw FXDataEngineError.externalBackend(error)
        }
    }

    private func run(_ command: PythonMLBackendCommand) async throws -> PythonMLBackendResponse {
        try await Task.detached(priority: .utility) {
            let input = try JSONEncoder().encode(command)
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            if self.module.contains("/") || self.module.hasSuffix(".py") {
                process.arguments = [self.executable, self.module, command.operation]
            } else {
                process.arguments = [self.executable, "-m", self.module, command.operation]
            }
            if !self.environment.isEmpty {
                process.environment = ProcessInfo.processInfo.environment.merging(self.environment) { _, new in new }
            }

            let stdin = Pipe()
            let stdout = Pipe()
            let stderr = Pipe()
            process.standardInput = stdin
            process.standardOutput = stdout
            process.standardError = stderr

            try process.run()
            stdin.fileHandleForWriting.write(input)
            try stdin.fileHandleForWriting.close()
            process.waitUntilExit()

            let output = stdout.fileHandleForReading.readDataToEndOfFile()
            let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
            guard process.terminationStatus == 0 else {
                let message = String(data: errorData, encoding: .utf8) ?? "Python backend failed"
                throw FXDataEngineError.externalBackend(message.trimmingCharacters(in: .whitespacesAndNewlines))
            }
            return try JSONDecoder().decode(PythonMLBackendResponse.self, from: output)
        }.value
    }

    private func runSynchronously(_ command: PythonMLBackendCommand) throws -> PythonMLBackendResponse {
        let input = try JSONEncoder().encode(command)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        if module.contains("/") || module.hasSuffix(".py") {
            process.arguments = [executable, module, command.operation]
        } else {
            process.arguments = [executable, "-m", module, command.operation]
        }
        if !environment.isEmpty {
            process.environment = ProcessInfo.processInfo.environment.merging(environment) { _, new in new }
        }

        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr

        try process.run()
        stdin.fileHandleForWriting.write(input)
        try stdin.fileHandleForWriting.close()
        process.waitUntilExit()

        let output = stdout.fileHandleForReading.readDataToEndOfFile()
        let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
        guard process.terminationStatus == 0 else {
            let message = String(data: errorData, encoding: .utf8) ?? "Python backend failed"
            throw FXDataEngineError.externalBackend(message.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return try JSONDecoder().decode(PythonMLBackendResponse.self, from: output)
    }
}

private struct PythonMLBackendCommand: Codable, Sendable {
    let operation: String
    let inference: MLInferencePayload?
    let training: MLTrainingPayload?
}

private struct PythonMLBackendResponse: Codable, Sendable {
    let ok: Bool
    let prediction: PredictionV4?
    let error: String?
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
            horizonMinutes: request.context.horizonMinutes,
            sequenceBars: request.context.sequenceBars,
            priceCostPoints: request.context.priceCostPoints,
            minMovePoints: request.context.minMovePoints,
            x: request.x,
            xWindow: request.xWindow
        )
    }

    public static func trainingPayload(
        descriptor: MLBackendDescriptor,
        request: TrainRequestV4
    ) -> MLTrainingPayload {
        let inference = MLInferencePayload(
            modelIdentifier: descriptor.modelIdentifier,
            framework: framework(for: descriptor),
            dataHasVolume: request.context.dataHasVolume,
            horizonMinutes: request.context.horizonMinutes,
            sequenceBars: request.context.sequenceBars,
            priceCostPoints: request.context.priceCostPoints,
            minMovePoints: request.context.minMovePoints,
            x: request.x,
            xWindow: request.xWindow
        )
        return MLTrainingPayload(inference: inference, request: request)
    }

    private static func framework(for descriptor: MLBackendDescriptor) -> MLFramework {
        switch descriptor.mode {
        case .inProcess(let resolved):
            return resolved
        case .externalPython(let resolved, _, _):
            return resolved
        }
    }
}
