import FXDataEngine
import Foundation

public enum AIMLPAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_mlp",
        primaryBackends: [.swiftScalar, .accelerate, .metal, .pyTorchMPS, .tensorFlowMetal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy ai_mlp multi-layer perceptron with temporal context adaptation. Metal provides independent batched dense projection and softmax kernels; PyTorch/MPS and TensorFlow/Metal folders provide independent sequence implementations for Apple Silicon. Core ML / Neural Engine is a later inference export candidate. NLP is not required."
    )
}
