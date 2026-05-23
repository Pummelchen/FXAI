import FXDataEngine
import Foundation

public enum AIQCEWAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_qcew",
        primaryBackends: [.swiftScalar, .accelerate, .metal],
        candidateBackends: [.pyTorchMPS],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU implementation of ai_qcew using qcew semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, Metal. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
