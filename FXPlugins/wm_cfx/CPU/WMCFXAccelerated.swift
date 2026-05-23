import FXDataEngine
import Foundation

public enum WMCFXAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "wm_cfx",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.pyTorchMPS],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 wm_cfx using currencyFactorWorld semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
