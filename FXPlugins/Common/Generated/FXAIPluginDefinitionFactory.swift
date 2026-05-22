import FXDataEngine
import Foundation

enum FXAIPluginZooDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        DistributionPluginDefinitions.all,
        FactorPluginDefinitions.all,
        LinearPluginDefinitions.all,
        MemoryPluginDefinitions.all,
        MixturePluginDefinitions.all,
        RLPluginDefinitions.all,
        SequencePluginDefinitions.all,
        StatPluginDefinitions.all,
        TreePluginDefinitions.all,
        TrendPluginDefinitions.all,
        WorldPluginDefinitions.all
    ].flatMap { $0 }.sorted { $0.aiID.rawValue < $1.aiID.rawValue }
}

enum FXAIPluginDefinitionFactory {
    static func linear(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .linear,
            featureSchema: .sparseStat,
            featureGroups: [.price, .multiTimeframe, .volatility, .volume, .filters],
            capabilities: [.selfTest, .onlineLearning, .multiHorizon],
            maxSequenceBars: 1,
            profile: .linear,
            primaryBackends: [.swiftSIMD, .accelerate],
            notes: "Native Swift adapter for online linear plugin. Uses volume-aware fallback inference until full fitted-state parity is ported."
        )
    }

    static func tree(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .tree,
            featureSchema: .tree,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume, .filters],
            capabilities: [.selfTest, .multiHorizon],
            maxSequenceBars: 1,
            profile: .tree,
            primaryBackends: [.swiftScalar, .metal],
            candidateBackends: [.swiftSIMD, .accelerate],
            notes: "Tree ensemble adapter. Metal is planned for batched scoring while CPU remains the deterministic reference."
        )
    }

    static func sequence(
        _ aiID: AIModelID,
        _ name: String,
        _ family: AIFamily,
        _ primary: [FXPluginAccelerationBackend],
        _ candidate: [FXPluginAccelerationBackend]
    ) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: family,
            featureSchema: .sequence,
            featureGroups: [.price, .multiTimeframe, .volatility, .time, .context, .volume],
            capabilities: [.selfTest, .windowContext, .multiHorizon],
            maxSequenceBars: 256,
            profile: .sequence,
            primaryBackends: primary,
            candidateBackends: candidate,
            notes: "Sequence-model adapter. Tensor training/inference is delegated to the declared backend; Swift fallback is volume-aware and deterministic."
        )
    }

    static func distribution(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .distributional,
            featureSchema: .sparseStat,
            featureGroups: [.price, .multiTimeframe, .volatility, .volume],
            capabilities: [.selfTest, .nativeDistribution, .multiHorizon],
            maxSequenceBars: 1,
            profile: .distribution,
            primaryBackends: [.accelerate],
            candidateBackends: [.metal],
            notes: "Distributional adapter using Accelerate-class statistics first, with Metal reserved for batched quantile work."
        )
    }

    static func statistical(
        _ aiID: AIModelID,
        _ name: String,
        _ primary: [FXPluginAccelerationBackend] = [.accelerate],
        _ candidate: [FXPluginAccelerationBackend] = [.swiftSIMD]
    ) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .other,
            featureSchema: .sparseStat,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume, .filters],
            capabilities: [.selfTest, .multiHorizon],
            maxSequenceBars: 1,
            profile: .statistical,
            primaryBackends: primary,
            candidateBackends: candidate,
            notes: "Statistical plugin adapter. Accelerate handles matrix/vector math; Python fitting can be added for complex estimators."
        )
    }

    static func factor(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .other,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .context, .volume],
            capabilities: [.selfTest, .multiHorizon],
            maxSequenceBars: 1,
            profile: .factor,
            primaryBackends: [.accelerate],
            candidateBackends: [.swiftSIMD],
            notes: "Factor plugin adapter using context and volume-aware scoring. Accelerate is the primary native math path."
        )
    }

    static func trend(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .other,
            featureSchema: .sparseStat,
            featureGroups: [.price, .multiTimeframe, .volatility, .volume],
            capabilities: [.selfTest, .multiHorizon],
            maxSequenceBars: 1,
            profile: .trend,
            primaryBackends: [.swiftSIMD, .accelerate],
            candidateBackends: [.metal],
            notes: "Trend plugin adapter with SIMD/Accelerate rolling-window path and optional Metal for large parameter sweeps."
        )
    }

    static func mixture(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .mixture,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume],
            capabilities: [.selfTest, .multiHorizon],
            maxSequenceBars: 1,
            profile: .mixture,
            primaryBackends: [.accelerate],
            candidateBackends: [.pyTorchMPS],
            notes: "Mixture adapter with native gating fallback and optional PyTorch MPS for learned expert routing."
        )
    }

    static func memory(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .retrieval,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume],
            capabilities: [.selfTest, .stateful, .multiHorizon],
            maxSequenceBars: 128,
            profile: .memory,
            primaryBackends: [.accelerate],
            candidateBackends: [.metal],
            notes: "Retrieval-memory adapter. Accelerate distance search is primary; Metal top-k is reserved for large replay banks."
        )
    }

    static func world(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .worldModel,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .time, .context, .volume],
            capabilities: [.selfTest, .windowContext, .multiHorizon],
            maxSequenceBars: 128,
            profile: .world,
            primaryBackends: [.accelerate],
            candidateBackends: [.pyTorchMPS, .coreMLNeuralEngine],
            notes: "World-model adapter. Native graph/context scoring remains the fallback; learned world models can move to PyTorch MPS and Core ML inference."
        )
    }

    static func reinforcement(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .other,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume, .microstructure],
            capabilities: [.selfTest, .windowContext, .multiHorizon],
            maxSequenceBars: 128,
            profile: .reinforcement,
            primaryBackends: [.pyTorchMPS],
            candidateBackends: [.coreMLNeuralEngine],
            notes: "RL policy adapter. PPO training belongs in PyTorch MPS; Core ML is an inference-only candidate after policy stabilization."
        )
    }
}
