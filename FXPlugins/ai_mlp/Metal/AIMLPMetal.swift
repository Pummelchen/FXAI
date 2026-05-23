import FXDataEngine
import Foundation

public enum AIMLPMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void ai_mlp_dense_hidden(
        device const float* features [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const float* bias [[buffer(2)]],
        device float* hidden [[buffer(3)]],
        constant uint& featureCount [[buffer(4)]],
        uint hiddenIndex [[thread_position_in_grid]]
    ) {
        float value = bias[hiddenIndex];
        for (uint i = 0; i < featureCount; ++i) {
            value += weights[hiddenIndex * featureCount + i] * features[i];
        }
        hidden[hiddenIndex] = tanh(clamp(value, -18.0f, 18.0f));
    }

    kernel void ai_mlp_class_logits(
        device const float* hidden [[buffer(0)]],
        device const float* classWeights [[buffer(1)]],
        device float* logits [[buffer(2)]],
        constant uint& hiddenCount [[buffer(3)]],
        uint classIndex [[thread_position_in_grid]]
    ) {
        if (classIndex >= 3) { return; }
        float value = classWeights[classIndex * (hiddenCount + 1)];
        for (uint i = 0; i < hiddenCount; ++i) {
            value += classWeights[classIndex * (hiddenCount + 1) + i + 1] * hidden[i];
        }
        logits[classIndex] = clamp(value, -35.0f, 35.0f);
    }

    kernel void ai_mlp_softmax3(
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
        pluginName: "ai_mlp",
        primaryBackends: [.swiftScalar, .accelerate, .metal, .pyTorchMPS, .tensorFlowMetal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Metal kernels for ai_mlp dense hidden projection, class logits, and softmax. CPU remains the deterministic reference path."
    )
}
