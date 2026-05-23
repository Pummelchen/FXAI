import FXDataEngine
import Foundation

public enum AIMythosRDTAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_mythos_rdt",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS],
        candidateBackends: [.foundationNLP, .coreMLNeuralEngine],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 ai_mythos_rdt using mythosRDT semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, NLP. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
