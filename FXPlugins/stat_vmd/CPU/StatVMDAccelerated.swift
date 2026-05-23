import FXDataEngine
import Foundation

public enum StatVMDAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_vmd",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy variational mode decomposition proxy. It uses two EMA mode components, rolling volatility, and online linear adaptation; Swift SIMD/Accelerate fit the small vector path, while Metal, PyTorch, TensorFlow, and NLP variants are not required."
    )
}
