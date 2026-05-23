import FXDataEngine
import Foundation

public enum LinProfitLogitMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void lin_profit_logit_scores(
        device const float* state [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* logits [[buffer(2)]],
        uint classIndex [[thread_position_in_grid]]
    ) {
        if (classIndex >= 3) { return; }
        const uint stateDim = 16;
        const uint base = classIndex * stateDim;
        float sum = 0.0f;
        for (uint i = 0; i < stateDim; ++i) {
            sum += weights[base + i] * state[i];
        }
        logits[classIndex] = clamp(sum, -12.0f, 12.0f);
    }

    kernel void lin_profit_logit_update(
        device const float* state [[buffer(0)]],
        device const float* probabilities [[buffer(1)]],
        device const float* target [[buffer(2)]],
        device float* weights [[buffer(3)]],
        constant float& learningRate [[buffer(4)]],
        constant float& l1 [[buffer(5)]],
        constant float& l2 [[buffer(6)]],
        uint gid [[thread_position_in_grid]]
    ) {
        const uint stateDim = 16;
        const uint total = 3 * stateDim;
        if (gid >= total) { return; }
        const uint classIndex = gid / stateDim;
        const uint stateIndex = gid % stateDim;
        const float w = weights[gid];
        const float sign = w > 0.0f ? 1.0f : (w < 0.0f ? -1.0f : 0.0f);
        const float gradient = (target[classIndex] - probabilities[classIndex]) * state[stateIndex] - l2 * w - l1 * sign;
        weights[gid] = clamp(w + learningRate * gradient, -8.0f, 8.0f);
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "lin_profit_logit",
        primaryBackends: [.swiftSIMD, .accelerate],
        candidateBackends: [.metal],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the framework profit-logit kind. Metal folder contains fixed 16-state batch scoring and profit update kernels; PyTorch, TensorFlow, and NLP are not suitable for this online linear model."
    )
}
