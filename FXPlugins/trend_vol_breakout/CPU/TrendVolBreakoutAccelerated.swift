import FXDataEngine
import Foundation

public enum TrendVolBreakoutAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "trend_vol_breakout",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy volatility breakout branch with non-overlapping recent-vs-baseline range expansion, rolling slopes, volatility normalization, and online linear adaptation. Swift SIMD/Accelerate fit the small vector path; Metal, PyTorch, TensorFlow, and NLP variants are not required."
    )
}
