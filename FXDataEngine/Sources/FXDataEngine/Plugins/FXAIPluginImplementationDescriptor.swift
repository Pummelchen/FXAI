import Foundation

public struct FXAIPluginImplementationDescriptor: Sendable, Hashable {
    public let aiID: AIModelID
    public let aiName: String
    public let family: AIFamily
    public let featureSchema: FeatureSchema
    public let featureGroups: FeatureGroupMask
    public let capabilities: PluginCapability
    public let maxSequenceBars: Int
    public let profile: FXAIReferencePluginProfile
    public let primaryBackends: [FXPluginAccelerationBackend]
    public let candidateBackends: [FXPluginAccelerationBackend]
    public let notes: String



    public var manifest: PluginManifestV4 {
        PluginManifestV4(
            aiID: aiID.rawValue,
            aiName: aiName,
            family: family,
            referenceTier: PluginPersistenceTools.defaultReferenceTier(aiID: aiID.rawValue),
            capabilityMask: capabilities,
            featureSchema: featureSchema,
            featureGroups: featureGroups,
            minHorizonMinutes: 1,
            maxHorizonMinutes: 240,
            minSequenceBars: 1,
            maxSequenceBars: maxSequenceBars,
            requiresVolumeWhenAvailable: true
        )
    }

    public var accelerationPlan: FXPluginAccelerationPlan {
        FXPluginAccelerationPlan(
            pluginName: aiName,
            primaryBackends: primaryBackends,
            candidateBackends: candidateBackends,
            usesVolumeWhenAvailable: true,
            notes: notes
        )
    }

    public init(
        aiID: AIModelID,
        aiName: String,
        family: AIFamily,
        featureSchema: FeatureSchema,
        featureGroups: FeatureGroupMask,
        capabilities: PluginCapability,
        maxSequenceBars: Int,
        profile: FXAIReferencePluginProfile,
        primaryBackends: [FXPluginAccelerationBackend],
        candidateBackends: [FXPluginAccelerationBackend] = [],
        notes: String
    ) {
        self.aiID = aiID
        self.aiName = aiName
        self.family = family
        self.featureSchema = featureSchema
        self.featureGroups = featureGroups
        self.capabilities = capabilities
        self.maxSequenceBars = maxSequenceBars
        self.profile = profile
        self.primaryBackends = primaryBackends
        self.candidateBackends = candidateBackends
        self.notes = notes
    }
}

public enum FXAIReferencePluginProfile: String, Sendable, Hashable {
    case linear
    case tree
    case sequence
    case distribution
    case statistical
    case factor
    case trend
    case mixture
    case memory
    case world
    case reinforcement

    public var strengthScale: Double {
        switch self {
        case .linear, .trend: 4.0
        case .tree, .factor: 3.6
        case .sequence, .world: 3.2
        case .distribution, .statistical, .mixture, .memory: 3.0
        case .reinforcement: 2.8
        }
    }

    public var skipThreshold: Double {
        switch self {
        case .linear, .trend: 0.08
        case .tree, .factor: 0.10
        case .sequence, .world, .reinforcement: 0.14
        case .distribution, .statistical, .mixture, .memory: 0.12
        }
    }

    public var moveScale: Double {
        switch self {
        case .linear, .trend, .factor: 90.0
        case .tree, .mixture, .memory: 100.0
        case .sequence, .world, .reinforcement: 110.0
        case .distribution, .statistical: 80.0
        }
    }

    public var baseReliability: Double {
        switch self {
        case .linear, .trend: 0.52
        case .tree, .factor: 0.54
        case .sequence, .world: 0.50
        case .distribution, .statistical, .mixture, .memory: 0.51
        case .reinforcement: 0.48
        }
    }

    public func edge(_ request: PredictRequestV4) -> Double {
        let shortReturn = Self.feature(request, 0)
        let mediumSlope = Self.feature(request, 3)
        let volatility = abs(Self.feature(request, 4))
        let volume = request.context.dataHasVolume ? Self.feature(request, 6) : 0.0
        let fastReturn = Self.feature(request, 7)
        let slowReturn = Self.feature(request, 8)
        let contextSignal = Self.feature(request, 12)
        let mtfEdge = fastReturn - slowReturn

        switch self {
        case .linear:
            return 0.48 * shortReturn + 0.34 * mediumSlope + 0.12 * mtfEdge + 0.06 * volume
        case .tree:
            return 0.35 * shortReturn + 0.25 * mediumSlope + 0.25 * mtfEdge + 0.10 * contextSignal + 0.05 * volume
        case .sequence:
            return 0.28 * shortReturn + 0.28 * mediumSlope + 0.30 * mtfEdge + 0.08 * contextSignal + 0.06 * volume
        case .distribution:
            return (0.40 * shortReturn + 0.25 * mediumSlope + 0.20 * mtfEdge + 0.10 * volume) / max(1.0 + volatility, 1.0)
        case .statistical:
            return 0.30 * shortReturn + 0.22 * mediumSlope + 0.22 * mtfEdge + 0.14 * contextSignal + 0.12 * volume
        case .factor:
            return 0.22 * shortReturn + 0.20 * mediumSlope + 0.18 * mtfEdge + 0.30 * contextSignal + 0.10 * volume
        case .trend:
            return 0.34 * shortReturn + 0.36 * mediumSlope + 0.20 * mtfEdge + 0.10 * volume
        case .mixture:
            return 0.25 * shortReturn + 0.25 * mediumSlope + 0.25 * mtfEdge + 0.15 * contextSignal + 0.10 * volume
        case .memory:
            return 0.20 * shortReturn + 0.25 * mediumSlope + 0.35 * mtfEdge + 0.10 * contextSignal + 0.10 * volume
        case .world:
            return 0.18 * shortReturn + 0.22 * mediumSlope + 0.24 * mtfEdge + 0.28 * contextSignal + 0.08 * volume
        case .reinforcement:
            return 0.24 * shortReturn + 0.24 * mediumSlope + 0.24 * mtfEdge + 0.16 * contextSignal + 0.12 * volume
        }
    }
}


private extension FXAIReferencePluginProfile {
    static func feature(_ request: PredictRequestV4, _ index: Int) -> Double {
        guard index >= 0, index < request.x.count else { return 0.0 }
        return fxSafeFinite(request.x[index])
    }
}

public extension FXAIPluginImplementationDescriptor {
    static func linear(_ aiID: AIModelID, _ name: String) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
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

    static func tree(_ aiID: AIModelID, _ name: String) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
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
    ) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
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

    static func distribution(_ aiID: AIModelID, _ name: String) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
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
    ) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
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

    static func factor(_ aiID: AIModelID, _ name: String) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
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

    static func trend(_ aiID: AIModelID, _ name: String) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
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

    static func mixture(_ aiID: AIModelID, _ name: String) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
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

    static func memory(_ aiID: AIModelID, _ name: String) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
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

    static func world(_ aiID: AIModelID, _ name: String) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
            aiID: aiID,
            aiName: name,
            family: .worldModel,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .time, .context, .volume],
            capabilities: [.selfTest, .windowContext, .multiHorizon],
            maxSequenceBars: 128,
            profile: .world,
            primaryBackends: [.accelerate],
            candidateBackends: [.pyTorchMPS],
            notes: "World-model adapter. Native graph/context scoring remains the fallback; learned world models can move to PyTorch MPS. Core ML is excluded until export, load, prediction, and parity tests exist."
        )
    }

    static func reinforcement(_ aiID: AIModelID, _ name: String) -> FXAIPluginImplementationDescriptor {
        FXAIPluginImplementationDescriptor(
            aiID: aiID,
            aiName: name,
            family: .other,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume, .microstructure],
            capabilities: [.selfTest, .windowContext, .multiHorizon],
            maxSequenceBars: 128,
            profile: .reinforcement,
            primaryBackends: [.pyTorchMPS],
            candidateBackends: [],
            notes: "RL policy adapter. PPO training belongs in PyTorch MPS. Core ML is excluded until export, load, prediction, and parity tests exist."
        )
    }
}
