import Foundation

public enum FXPluginAccelerationBackend: String, Codable, Hashable, Sendable, CaseIterable {
    case swiftScalar
    case swiftSIMD
    case accelerate
    case metal
    case pyTorchMPS
    case tensorFlowMetal
    case foundationNLP
    case coreMLNeuralEngine

    public var isCPUOnly: Bool {
        switch self {
        case .swiftScalar, .swiftSIMD, .accelerate:
            return true
        case .metal, .pyTorchMPS, .tensorFlowMetal, .foundationNLP, .coreMLNeuralEngine:
            return false
        }
    }

    public var requiresExternalPython: Bool {
        switch self {
        case .pyTorchMPS, .tensorFlowMetal:
            return true
        case .swiftScalar, .swiftSIMD, .accelerate, .metal, .foundationNLP, .coreMLNeuralEngine:
            return false
        }
    }
}

public struct FXPluginAccelerationPlan: Codable, Hashable, Sendable {
    public let pluginName: String
    public let primaryBackends: [FXPluginAccelerationBackend]
    public let candidateBackends: [FXPluginAccelerationBackend]
    public let usesVolumeWhenAvailable: Bool
    public let notes: String

    public init(
        pluginName: String,
        primaryBackends: [FXPluginAccelerationBackend],
        candidateBackends: [FXPluginAccelerationBackend] = [],
        usesVolumeWhenAvailable: Bool = true,
        notes: String
    ) {
        self.pluginName = pluginName
        self.primaryBackends = primaryBackends
        self.candidateBackends = candidateBackends
        self.usesVolumeWhenAvailable = usesVolumeWhenAvailable
        self.notes = notes
    }

    public var declaresHardwareAcceleration: Bool {
        !Set(primaryBackends + candidateBackends).intersection([
            .swiftSIMD,
            .accelerate,
            .metal,
            .pyTorchMPS,
            .tensorFlowMetal,
            .foundationNLP,
            .coreMLNeuralEngine
        ]).isEmpty
    }

    public var declaredBackends: [FXPluginAccelerationBackend] {
        var seen = Set<FXPluginAccelerationBackend>()
        return (primaryBackends + candidateBackends).filter { seen.insert($0).inserted }
    }

    public func declares(_ backend: FXPluginAccelerationBackend) -> Bool {
        declaredBackends.contains(backend)
    }

    public var cpuFallbackBackend: FXPluginAccelerationBackend? {
        declaredBackends.first(where: \.isCPUOnly)
    }
}

public protocol FXAIPlannedPlugin: FXAIPluginV4 {
    var accelerationPlan: FXPluginAccelerationPlan { get }
}

public extension FXAIPlannedPlugin {
    func resolveRuntimeBackend(
        mode: FXPluginRuntimeMode = .automatic,
        fallbackPolicy: FXPluginRuntimeFallbackPolicy = .fallBackToCPU,
        environment: FXPluginRuntimeEnvironment = .local
    ) throws -> FXPluginRuntimeResolution {
        try FXPluginRuntimeResolver.resolve(
            plan: accelerationPlan,
            mode: mode,
            fallbackPolicy: fallbackPolicy,
            environment: environment
        )
    }
}
