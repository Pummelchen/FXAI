import FXDataEngine
import Foundation

public enum AILSTMAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_lstm",
        primaryBackends: [.swiftScalar, .accelerate, .tensorFlowMetal, .pyTorchMPS],
        candidateBackends: [.metal, .coreMLNeuralEngine],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_lstm using recurrent semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, TensorFlow, Metal. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
