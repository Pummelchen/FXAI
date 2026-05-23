import FXDataEngine
import Foundation

public enum AIAttnCNNBiLSTMAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_attn_cnn_bilstm",
        primaryBackends: [.swiftScalar, .accelerate, .tensorFlowMetal, .pyTorchMPS],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU implementation of ai_attn_cnn_bilstm using attentionCNNBiLSTM semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, TensorFlow. CPU remains the deterministic fallback; Python variants are independent implementations under this plugin folder."
    )
}
