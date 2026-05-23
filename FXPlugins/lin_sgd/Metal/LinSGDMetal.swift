import FXDataEngine
import Foundation

public enum LinSGDMetal {
    public static let variantName = "lin_sgd_metal"

    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void lin_sgd_linear_logits(
        device const float *features [[buffer(0)]],
        device const float *weights [[buffer(1)]],
        device float *logits [[buffer(2)]],
        constant uint &featureCount [[buffer(3)]],
        uint classIndex [[thread_position_in_grid]]
    ) {
        if (classIndex >= 3) { return; }
        float value = 0.0;
        for (uint i = 0; i < featureCount; ++i) {
            value += weights[classIndex * featureCount + i] * features[i];
        }
        logits[classIndex] = clamp(value, -35.0, 35.0);
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "lin_sgd",
        primaryBackends: [.swiftSIMD, .accelerate],
        candidateBackends: [.metal],
        usesVolumeWhenAvailable: true,
        notes: "Full plugin-local Swift CPU port of legacy MQL5 lin_sgd. Metal folder provides the batch linear-logit kernel source; hashed interaction dispatch is added after CPU parity is locked."
    )
}
