import FXDataEngine
import Foundation

public enum FactorPPPValueAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "factor_ppp_value",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy PPP valuation factor. The live model is a small sparse factor formula with online quality/move state, so Accelerate/Swift SIMD are appropriate; Metal, PyTorch, TensorFlow, and NLP variants are not required."
    )
}
