import FXDataEngine
import Foundation

public enum AIStmnAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_stmn",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS],
        candidateBackends: [.metal],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU implementation of ai_stmn using stmn semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, Metal. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
