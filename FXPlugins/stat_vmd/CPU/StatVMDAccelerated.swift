import FXDataEngine
import Foundation

public enum StatVMDAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_vmd",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate, .metal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy variational mode decomposition proxy. It uses two EMA mode components, rolling volatility, and online linear adaptation; Metal covers batched mode-proxy scans on Apple GPUs, while PyTorch, TensorFlow, and NLP variants are not required."
    )
}
