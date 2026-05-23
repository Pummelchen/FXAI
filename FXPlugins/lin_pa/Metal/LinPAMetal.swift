import FXDataEngine
import Foundation

public enum LinPAMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void lin_pa_linear_scores(
        device const float* features [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* scores [[buffer(2)]],
        constant uint& featureCount [[buffer(3)]],
        uint classIndex [[thread_position_in_grid]]
    ) {
        if (classIndex >= 3) { return; }
        float sum = 0.0f;
        const uint base = classIndex * featureCount;
        for (uint i = 0; i < featureCount; ++i) {
            sum += weights[base + i] * features[i];
        }
        scores[classIndex] = clamp(sum, -20.0f, 20.0f);
    }

    kernel void lin_pa_pair_products(
        device const float* features [[buffer(0)]],
        device float* products [[buffer(1)]],
        constant uint& featureCount [[buffer(2)]],
        uint gid [[thread_position_in_grid]]
    ) {
        const uint total = featureCount * featureCount;
        if (gid >= total) { return; }
        const uint i = gid / featureCount;
        const uint j = gid % featureCount;
        products[gid] = (i < j && i > 0) ? features[i] * features[j] : 0.0f;
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "lin_pa",
        primaryBackends: [.swiftSIMD, .accelerate],
        candidateBackends: [.metal],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU Crammer-Singer PA port with confidence scaling, averaged weights, dual hashed interactions, calibration, move distribution, and hard replay. Metal folder contains batch scoring and pair-product kernels for the PA/hash accelerator path."
    )
}
