import FXDataEngine
import Foundation

public enum FactorCarryAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "factor_carry",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy carry factor branch with seeded online linear base policy, carry/context pressure, and volatility penalty. Accelerate/Swift SIMD fit the small vector operations; Metal, PyTorch, TensorFlow, and NLP variants are not required."
    )
}
