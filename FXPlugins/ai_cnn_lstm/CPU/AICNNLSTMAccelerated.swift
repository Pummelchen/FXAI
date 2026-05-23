import FXDataEngine
import Foundation

public enum AICNNLSTMAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_cnn_lstm",
        primaryBackends: [.swiftScalar, .accelerate, .tensorFlowMetal, .pyTorchMPS, .metal],
        candidateBackends: [.coreMLNeuralEngine],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_cnn_lstm using cnnLSTM semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, TensorFlow, Metal. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
