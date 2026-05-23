import FXDataEngine
import Foundation

public enum StatARIMAXGARCHAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_arimax_garch",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy framework ARIMAX/GARCH branch with seeded multiclass ARIMAX policy weights, online GARCH volatility recursion, volume-aware volatility fallback, move EMA, and native distribution outputs. Accelerate/Swift SIMD are suitable for the small vector operations; Metal, PyTorch, TensorFlow, and NLP are not required for the live online model."
    )
}
