import FXDataEngine
import Foundation

public enum LinEnhashMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void lin_enhash_pair_products(
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
        pluginName: "lin_enhash",
        primaryBackends: [.swiftSIMD, .accelerate],
        candidateBackends: [.metal],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU ENHash/FTRL port with dual field-aware hash tables. Metal folder contains the pair-product kernel used by the future batched hash scorer."
    )
}
