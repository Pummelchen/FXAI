import FXDataEngine
import Foundation

public enum AIBiLSTMAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_bilstm",
        primaryBackends: [.swiftScalar, .accelerate, .tensorFlowMetal, .pyTorchMPS],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_bilstm using bidirectional semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, TensorFlow. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
