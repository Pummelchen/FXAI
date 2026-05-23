import FXDataEngine
import Foundation

public enum AILSTMGAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_lstmg",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS, .tensorFlowMetal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_lstmg using gatedRecurrent semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, TensorFlow. CPU remains the deterministic fallback; Python variants are independent implementations under this plugin folder."
    )
}
