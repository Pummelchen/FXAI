import FXDataEngine
import Foundation

public enum AITFTAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_tft",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU implementation of ai_tft using temporalFusionTransformer semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch. CPU remains the deterministic fallback; Python variants are independent implementations under this plugin folder."
    )
}
