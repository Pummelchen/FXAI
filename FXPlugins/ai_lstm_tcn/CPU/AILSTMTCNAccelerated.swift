import FXDataEngine
import Foundation

public enum AILSTMTCNAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_lstm_tcn",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS, .tensorFlowMetal, .metal],
        candidateBackends: [.coreMLNeuralEngine],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_lstm_tcn using lstmTCN semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, TensorFlow, Metal. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
