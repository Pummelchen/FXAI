import FXDataEngine
import Foundation

public enum AITFTAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_tft",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS, .metal],
        candidateBackends: [.coreMLNeuralEngine],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_tft using temporalFusionTransformer semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, Metal. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
