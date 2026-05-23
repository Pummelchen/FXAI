import FXDataEngine
import Foundation

public enum AIGeodesicAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_geodesic",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS, .metal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU implementation of ai_geodesic using geodesicAttention semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, Metal. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
