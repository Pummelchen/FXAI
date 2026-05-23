import FXDataEngine
import Foundation

public enum AITRRAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_trr",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.pyTorchMPS],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU implementation of ai_trr using trendReversalRecurrent semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
