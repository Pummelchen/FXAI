import FXDataEngine
import Foundation

public enum TrendTSMOMVolAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "trend_tsmom_vol",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy time-series momentum over volatility branch. It uses bounded rolling-window deltas, slopes, volatility normalization, and online linear adaptation; Swift SIMD/Accelerate fit the small vector path, while Metal, PyTorch, TensorFlow, and NLP variants are not required."
    )
}
