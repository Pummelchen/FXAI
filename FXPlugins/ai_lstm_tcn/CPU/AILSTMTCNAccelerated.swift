import FXDataEngine
import Foundation

public enum AILSTMTCNAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_lstm_tcn",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS, .tensorFlowMetal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_lstm_tcn using lstmTCN semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, TensorFlow. CPU remains the deterministic fallback; Python variants are independent implementations under this plugin folder."
    )
}
