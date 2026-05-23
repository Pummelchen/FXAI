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

    public init(
        metalDevice: MetalAccelerationDevice = .probe(),
        pythonExecutable: String? = nil,
        pyTorchMPSAvailable: Bool = false,
        tensorFlowMetalAvailable: Bool = false,
        foundationNLPAvailable: Bool = false,
        coreMLNeuralEngineAvailable: Bool = false
    ) {
        self.metalDevice = metalDevice
        self.pythonExecutable = pythonExecutable?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        self.pyTorchMPSAvailable = pyTorchMPSAvailable
        self.tensorFlowMetalAvailable = tensorFlowMetalAvailable
        self.foundationNLPAvailable = foundationNLPAvailable
        self.coreMLNeuralEngineAvailable = coreMLNeuralEngineAvailable
    }

    public static var local: FXPluginRuntimeEnvironment {
        let environment = ProcessInfo.processInfo.environment
        return FXPluginRuntimeEnvironment(
            metalDevice: .probe(),
            pythonExecutable: environment["FXAI_PYTHON"] ?? "python3",
            pyTorchMPSAvailable: environment["FXAI_ENABLE_PYTORCH_MPS"] == "1",
            tensorFlowMetalAvailable: environment["FXAI_ENABLE_TENSORFLOW_METAL"] == "1",
            foundationNLPAvailable: environment["FXAI_ENABLE_FOUNDATION_NLP"] == "1",
            coreMLNeuralEngineAvailable: environment["FXAI_ENABLE_COREML_NEURAL_ENGINE"] == "1"
        )
    }

    public func supports(_ backend: FXPluginAccelerationBackend) -> Bool {
        switch backend {
        case .swiftScalar, .swiftSIMD, .accelerate:
            return true
        case .metal:
            return metalDevice.available
        case .pyTorchMPS:
            return pythonExecutable != nil && pyTorchMPSAvailable
        case .tensorFlowMetal:
            return pythonExecutable != nil && tensorFlowMetalAvailable
        case .foundationNLP:
            return foundationNLPAvailable
        case .coreMLNeuralEngine:
            return coreMLNeuralEngineAvailable
        }
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
        }
    }
}

public enum FXPluginRuntimeResolver {
    private static let automaticPreference: [FXPluginAccelerationBackend] = [
        .pyTorchMPS,
        .tensorFlowMetal,
        .metal,
        .coreMLNeuralEngine,
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

        case .swiftScalar, .swiftSIMD, .accelerate, .metal, .pyTorchMPS, .tensorFlowMetal, .foundationNLP, .coreMLNeuralEngine:
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
