import FXDataEngine
import Foundation

public enum AINhitsMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void ai_nhits_sequence_logits(
        device const float* features [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* logits [[buffer(2)]],
        constant uint& featureCount [[buffer(3)]],
        uint classIndex [[thread_position_in_grid]]
    ) {
        if (classIndex >= 3) { return; }
        float value = weights[classIndex * featureCount];
        for (uint i = 0; i < featureCount; ++i) {
            value += weights[classIndex * featureCount + i] * features[i];
        }
        logits[classIndex] = clamp(value, -35.0f, 35.0f);
    }

    kernel void ai_nhits_softmax3(
        device const float* logits [[buffer(0)]],
        device float* probabilities [[buffer(1)]],
        uint tid [[thread_position_in_grid]]
    ) {
        if (tid > 0) { return; }
        const float m = max(logits[0], max(logits[1], logits[2]));
        const float e0 = exp(clamp(logits[0] - m, -30.0f, 30.0f));
        const float e1 = exp(clamp(logits[1] - m, -30.0f, 30.0f));
        const float e2 = exp(clamp(logits[2] - m, -30.0f, 30.0f));
        const float s = max(e0 + e1 + e2, 0.000001f);
        probabilities[0] = e0 / s;
        probabilities[1] = e1 / s;
        probabilities[2] = e2 / s;
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "ai_nhits",
        primaryBackends: [.swiftScalar, .accelerate, .pyTorchMPS],
        candidateBackends: [.foundationNLP, .onnxRuntime],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Metal projection kernel source for ai_nhits. Metal kernels perform a simple linear projection and 3-class softmax for runtime compilation and parity verification. The real N-HiTS inference path uses the Swift CPU or PyTorch MPS accelerator."
    )
}
