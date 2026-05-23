import FXDataEngine
import Foundation

public enum AIMLPAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_mlp",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS, .tensorFlowMetal],
        candidateBackends: [.coreMLNeuralEngine],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy ai_mlp multi-layer perceptron with temporal context adaptation. PyTorch/MPS and TensorFlow/Metal folders provide independent batched sequence implementations for Apple Silicon; Core ML / Neural Engine is a later inference export candidate. Direct Metal and NLP variants are not required."
    )
}
