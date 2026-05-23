import FXDataEngine
import Foundation

public enum AITimesfmAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_timesfm",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS],
        candidateBackends: [.coreMLNeuralEngine],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_timesfm using foundationForecaster semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, NLP. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
