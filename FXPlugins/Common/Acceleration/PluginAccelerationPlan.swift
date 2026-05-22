import FXDataEngine
import Foundation

public enum FXPluginAccelerationBackend: String, Codable, Hashable, Sendable, CaseIterable {
    case swiftScalar
    case swiftSIMD
    case accelerate
    case metal
    case pyTorchMPS
    case tensorFlowMetal
    case coreMLNeuralEngine
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
            .coreMLNeuralEngine
        ]).isEmpty
    }
}

public protocol FXAIPlannedPlugin: FXAIPluginV4 {
    var accelerationPlan: FXPluginAccelerationPlan { get }
}
