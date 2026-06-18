import FXDataEngine
import Foundation

public enum AINhitsAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_nhits",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS],
        candidateBackends: [.foundationNLP, .onnxRuntime],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU implementation of ai_nhits using nhits hierarchical interpolation semantics, online class and move heads, window context, OHLCV volume gating, calibration, and path-quality heads. Accelerator folders: PyTorch, NLP, ONNX. CPU remains the deterministic fallback; Python/Metal/NLP variants are independent implementations under this plugin folder."
    )
}
