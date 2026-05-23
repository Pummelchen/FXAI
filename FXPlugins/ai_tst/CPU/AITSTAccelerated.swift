import FXDataEngine
import Foundation

public enum AITSTAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_tst",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS, .metal],
        candidateBackends: [.coreMLNeuralEngine],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_tst using transformer semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, Metal. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
