import FXDataEngine
import Foundation

public enum AIGRUAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_gru",
        primaryBackends: [.swiftScalar, .accelerate, .tensorFlowMetal, .pyTorchMPS],
        candidateBackends: [.coreMLNeuralEngine],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_gru using gru semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, TensorFlow. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
