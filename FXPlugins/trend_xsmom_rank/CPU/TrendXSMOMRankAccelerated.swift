import FXDataEngine
import Foundation

public enum TrendXSMOMRankAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "trend_xsmom_rank",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy cross-sectional momentum rank branch with seeded online linear base policy. The live path is small sparse-vector math, so Swift SIMD/Accelerate are sufficient; Metal, PyTorch, TensorFlow, and NLP variants are not required."
    )
}
