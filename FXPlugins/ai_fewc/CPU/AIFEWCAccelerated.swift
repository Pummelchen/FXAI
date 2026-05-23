import FXDataEngine
import Foundation

public enum AIFEWCAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_fewc",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.pyTorchMPS, .metal],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_fewc using fewc semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, Metal. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
