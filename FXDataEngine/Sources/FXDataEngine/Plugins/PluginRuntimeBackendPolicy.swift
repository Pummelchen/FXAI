import Foundation

public enum FXPluginRuntimeMode: String, Codable, Hashable, Sendable, CaseIterable {
    case automatic
    case cpuOnly
    case swiftScalar
    case swiftSIMD
    case accelerate
    case metal
    case pyTorchMPS
    case tensorFlowMetal
    case foundationNLP
    case coreMLNeuralEngine
    case onnxRuntime
    case remoteRPC

    public var forcedBackend: FXPluginAccelerationBackend? {
        switch self {
        case .automatic, .cpuOnly:
            return nil
        case .swiftScalar:
            return .swiftScalar
        case .swiftSIMD:
            return .swiftSIMD
        case .accelerate:
            return .accelerate
        case .metal:
            return .metal
        case .pyTorchMPS:
            return .pyTorchMPS
        case .tensorFlowMetal:
            return .tensorFlowMetal
        case .foundationNLP:
            return .foundationNLP
        case .coreMLNeuralEngine:
            return .coreMLNeuralEngine
        case .onnxRuntime:
            return .onnxRuntime
        case .remoteRPC:
            return .remoteRPC
        }
    }
}

public enum FXPluginRuntimeFallbackPolicy: String, Codable, Hashable, Sendable, CaseIterable {
    case fallBackToCPU
    case strict
}

public struct FXPluginRuntimeEnvironment: Codable, Hashable, Sendable {
    public var metalDevice: MetalAccelerationDevice
    public var pythonExecutable: String?
    public var pyTorchMPSAvailable: Bool
    public var tensorFlowMetalAvailable: Bool
    public var foundationNLPAvailable: Bool
    public var coreMLNeuralEngineAvailable: Bool
    public var onnxRuntimeAvailable: Bool
    public var remoteInferenceAvailable: Bool
    public var remoteInferenceEndpoint: String?
    public var remoteInferenceAuthToken: String?
    public var remoteInferenceTimeoutSeconds: TimeInterval

    public init(
        metalDevice: MetalAccelerationDevice = .probe(),
        pythonExecutable: String? = nil,
        pyTorchMPSAvailable: Bool = false,
        tensorFlowMetalAvailable: Bool = false,
        foundationNLPAvailable: Bool = false,
        coreMLNeuralEngineAvailable: Bool = false,
        onnxRuntimeAvailable: Bool = false,
        remoteInferenceAvailable: Bool = false,
        remoteInferenceEndpoint: String? = nil,
        remoteInferenceAuthToken: String? = nil,
        remoteInferenceTimeoutSeconds: TimeInterval = 10.0
    ) {
        self.metalDevice = metalDevice
        self.pythonExecutable = pythonExecutable?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        self.pyTorchMPSAvailable = pyTorchMPSAvailable
        self.tensorFlowMetalAvailable = tensorFlowMetalAvailable
        self.foundationNLPAvailable = foundationNLPAvailable
        self.coreMLNeuralEngineAvailable = coreMLNeuralEngineAvailable
        self.onnxRuntimeAvailable = onnxRuntimeAvailable
        self.remoteInferenceAvailable = remoteInferenceAvailable
        self.remoteInferenceEndpoint = remoteInferenceEndpoint?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        self.remoteInferenceAuthToken = remoteInferenceAuthToken?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        self.remoteInferenceTimeoutSeconds = remoteInferenceTimeoutSeconds.isFinite && remoteInferenceTimeoutSeconds > 0.0
            ? min(remoteInferenceTimeoutSeconds, 300.0)
            : 10.0
    }

    public static var local: FXPluginRuntimeEnvironment {
        let environment = ProcessInfo.processInfo.environment
        return local(environment: environment)
    }

    public static func local(environment: [String: String]) -> FXPluginRuntimeEnvironment {
        let remoteTimeout = environment["FXAI_REMOTE_INFERENCE_TIMEOUT_SECONDS"].flatMap(Double.init)
        return FXPluginRuntimeEnvironment(
            metalDevice: .probe(),
            pythonExecutable: defaultPythonExecutable(environment: environment),
            pyTorchMPSAvailable: environment["FXAI_ENABLE_PYTORCH_MPS"] == "1",
            tensorFlowMetalAvailable: environment["FXAI_ENABLE_TENSORFLOW_METAL"] == "1",
            foundationNLPAvailable: environment["FXAI_ENABLE_FOUNDATION_NLP"] == "1",
            coreMLNeuralEngineAvailable: environment["FXAI_ENABLE_COREML_NEURAL_ENGINE"] == "1",
            onnxRuntimeAvailable: environment["FXAI_ENABLE_ONNX_RUNTIME"] == "1",
            remoteInferenceAvailable: environment["FXAI_ENABLE_REMOTE_RPC"] == "1",
            remoteInferenceEndpoint: environment["FXAI_REMOTE_INFERENCE_ENDPOINT"],
            remoteInferenceAuthToken: environment["FXAI_REMOTE_INFERENCE_AUTH_TOKEN"],
            remoteInferenceTimeoutSeconds: remoteTimeout ?? 10.0
        )
    }

    public static func defaultPythonExecutable(environment: [String: String]) -> String {
        if let configured = environment["FXAI_PYTHON"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !configured.isEmpty {
            return configured
        }
        return "python3.12"
    }

    public var isFXAIAppleSiliconTarget: Bool {
        metalDevice.hardware.isM2M3OrNewer
    }

    public func supports(_ backend: FXPluginAccelerationBackend) -> Bool {
        switch backend {
        case .swiftScalar, .swiftSIMD, .accelerate:
            return true
        case .metal:
            return metalDevice.optimizedForFXAIAppleSilicon
        case .pyTorchMPS:
            return pythonExecutable != nil && pyTorchMPSAvailable && isFXAIAppleSiliconTarget
        case .tensorFlowMetal:
            return pythonExecutable != nil && tensorFlowMetalAvailable && isFXAIAppleSiliconTarget
        case .foundationNLP:
            return foundationNLPAvailable && isFXAIAppleSiliconTarget
        case .coreMLNeuralEngine:
            return coreMLNeuralEngineAvailable && isFXAIAppleSiliconTarget
        case .onnxRuntime:
            return pythonExecutable != nil && onnxRuntimeAvailable
        case .remoteRPC:
            return remoteInferenceAvailable && remoteInferenceEndpoint != nil
        }
    }

    public func remoteRPCConfiguration() -> RemoteRPCMLBackendConfiguration? {
        guard remoteInferenceAvailable, let remoteInferenceEndpoint else { return nil }
        return RemoteRPCMLBackendConfiguration(
            endpoint: remoteInferenceEndpoint,
            authToken: remoteInferenceAuthToken,
            timeoutSeconds: remoteInferenceTimeoutSeconds
        )
    }
}

public struct FXPluginRuntimeResolution: Codable, Hashable, Sendable {
    public var pluginName: String
    public var requestedMode: FXPluginRuntimeMode
    public var selectedBackend: FXPluginAccelerationBackend
    public var fallbackBackend: FXPluginAccelerationBackend?
    public var fallbackReason: String?
    public var usesVolumeWhenAvailable: Bool

    public init(
        pluginName: String,
        requestedMode: FXPluginRuntimeMode,
        selectedBackend: FXPluginAccelerationBackend,
        fallbackBackend: FXPluginAccelerationBackend? = nil,
        fallbackReason: String? = nil,
        usesVolumeWhenAvailable: Bool
    ) {
        self.pluginName = pluginName
        self.requestedMode = requestedMode
        self.selectedBackend = selectedBackend
        self.fallbackBackend = fallbackBackend
        self.fallbackReason = fallbackReason
        self.usesVolumeWhenAvailable = usesVolumeWhenAvailable
    }

    public var didFallback: Bool {
        fallbackBackend != nil
    }

    public var requiresExternalPython: Bool {
        selectedBackend.requiresExternalPython
    }

    public var mlFramework: MLFramework? {
        switch selectedBackend {
        case .swiftScalar, .swiftSIMD, .accelerate:
            return .nativeSwift
        case .metal:
            return .metal
        case .pyTorchMPS:
            return .pyTorch
        case .tensorFlowMetal:
            return .tensorFlow
        case .foundationNLP:
            return .foundationNLP
        case .coreMLNeuralEngine:
            return nil
        case .onnxRuntime:
            return .onnxRuntime
        case .remoteRPC:
            return .remoteRPC
        }
    }
}

public enum FXPluginRuntimeResolver {
    private static let automaticPreference: [FXPluginAccelerationBackend] = [
        .remoteRPC,
        .onnxRuntime,
        .pyTorchMPS,
        .tensorFlowMetal,
        .metal,
        .foundationNLP,
        .accelerate,
        .swiftSIMD,
        .swiftScalar
    ]

    public static func resolve(
        plan: FXPluginAccelerationPlan,
        mode: FXPluginRuntimeMode = .automatic,
        fallbackPolicy: FXPluginRuntimeFallbackPolicy = .fallBackToCPU,
        environment: FXPluginRuntimeEnvironment = .local
    ) throws -> FXPluginRuntimeResolution {
        let declared = plan.declaredBackends
        guard !declared.isEmpty else {
            throw FXDataEngineError.validation("runtime.\(plan.pluginName).noBackendsDeclared")
        }

        switch mode {
        case .automatic:
            if let selected = automaticPreference.first(where: { declared.contains($0) && environment.supports($0) }) {
                return FXPluginRuntimeResolution(
                    pluginName: plan.pluginName,
                    requestedMode: mode,
                    selectedBackend: selected,
                    usesVolumeWhenAvailable: plan.usesVolumeWhenAvailable
                )
            }
            return try fallbackResolution(
                plan: plan,
                mode: mode,
                reason: "no declared accelerator is available",
                fallbackPolicy: fallbackPolicy
            )

        case .cpuOnly:
            guard let selected = plan.cpuFallbackBackend else {
                throw FXDataEngineError.validation("runtime.\(plan.pluginName).cpuFallbackMissing")
            }
            return FXPluginRuntimeResolution(
                pluginName: plan.pluginName,
                requestedMode: mode,
                selectedBackend: selected,
                usesVolumeWhenAvailable: plan.usesVolumeWhenAvailable
            )

        case .swiftScalar, .swiftSIMD, .accelerate, .metal, .pyTorchMPS, .tensorFlowMetal, .foundationNLP, .coreMLNeuralEngine, .onnxRuntime, .remoteRPC:
            guard let requested = mode.forcedBackend else {
                throw FXDataEngineError.validation("runtime.\(plan.pluginName).invalidMode")
            }
            guard declared.contains(requested) else {
                throw FXDataEngineError.validation("runtime.\(plan.pluginName).undeclaredBackend.\(requested.rawValue)")
            }
            guard environment.supports(requested) else {
                return try fallbackResolution(
                    plan: plan,
                    mode: mode,
                    reason: "\(requested.rawValue) is not available in the current runtime environment",
                    fallbackPolicy: fallbackPolicy
                )
            }
            return FXPluginRuntimeResolution(
                pluginName: plan.pluginName,
                requestedMode: mode,
                selectedBackend: requested,
                usesVolumeWhenAvailable: plan.usesVolumeWhenAvailable
            )
        }
    }

    private static func fallbackResolution(
        plan: FXPluginAccelerationPlan,
        mode: FXPluginRuntimeMode,
        reason: String,
        fallbackPolicy: FXPluginRuntimeFallbackPolicy
    ) throws -> FXPluginRuntimeResolution {
        guard fallbackPolicy == .fallBackToCPU, let fallback = plan.cpuFallbackBackend else {
            throw FXDataEngineError.externalBackend("runtime.\(plan.pluginName): \(reason)")
        }
        return FXPluginRuntimeResolution(
            pluginName: plan.pluginName,
            requestedMode: mode,
            selectedBackend: fallback,
            fallbackBackend: fallback,
            fallbackReason: reason,
            usesVolumeWhenAvailable: plan.usesVolumeWhenAvailable
        )
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}
