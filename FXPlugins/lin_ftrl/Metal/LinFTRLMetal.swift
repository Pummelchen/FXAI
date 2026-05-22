import FXDataEngine
import Foundation

public enum LinFTRLMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void lin_ftrl_linear_logits(
        device const float* features [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* logits [[buffer(2)]],
        constant uint& featureCount [[buffer(3)]],
        uint gid [[thread_position_in_grid]]
    ) {
        if (gid >= 3) { return; }
        float sum = 0.0f;
        const uint base = gid * featureCount;
        for (uint i = 0; i < featureCount; ++i) {
            sum += weights[base + i] * features[i];
        }
        logits[gid] = clamp(sum, -35.0f, 35.0f);
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "lin_ftrl",
        primaryBackends: [.swiftSIMD, .accelerate],
        candidateBackends: [.metal],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU FTRL-prox port with two hashed interaction spaces. Metal folder contains the batch linear-logit kernel source; hash-space batch dispatch is the next accelerator step."
    )
}
