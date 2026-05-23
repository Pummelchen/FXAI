import FXDataEngine
import Foundation

public enum TrendVolBreakoutAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "trend_vol_breakout",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate, .metal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy volatility breakout branch with non-overlapping recent-vs-baseline range expansion, rolling slopes, volatility normalization, and online linear adaptation. Metal covers batched offline breakout scans on Apple GPUs; PyTorch, TensorFlow, and NLP variants are not required."
    )
}
