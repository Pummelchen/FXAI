import FXDataEngine
import Foundation

public enum AILSTMAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_lstm",
        primaryBackends: [.swiftScalar, .accelerate, .tensorFlowMetal, .pyTorchMPS],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU implementation of ai_lstm using recurrent semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, TensorFlow. CPU remains the deterministic fallback; Python variants are independent implementations under this plugin folder."
    )
}
