import FXDataEngine
import Foundation

public enum StatEMDHHTAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_emd_hht",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy empirical mode decomposition / Hilbert-Huang transform proxy. It uses recent intrinsic-mode deltas, mean-shift energy, rolling volatility, and online linear adaptation; Swift SIMD/Accelerate fit the small vector path, while Metal, PyTorch, TensorFlow, and NLP variants are not required."
    )
}
