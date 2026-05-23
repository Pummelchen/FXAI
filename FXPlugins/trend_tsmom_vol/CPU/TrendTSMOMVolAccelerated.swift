import FXDataEngine
import Foundation

public enum TrendTSMOMVolAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "trend_tsmom_vol",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate, .metal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy time-series momentum over volatility branch. It uses bounded rolling-window deltas, slopes, volatility normalization, and online linear adaptation; Metal covers batched offline window scans on Apple GPUs, while PyTorch, TensorFlow, and NLP variants are not required."
    )
}
