import FXDataEngine
import Foundation

public enum StatXrateConsistencyAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_xrate_consistency",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy cross-rate consistency branch. The live path is a small online state-space correction over sparse OHLCV/context features, so Accelerate/Swift SIMD are suitable; Metal, PyTorch, TensorFlow, and NLP variants are not required."
    )
}
