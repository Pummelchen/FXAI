import FXDataEngine
import Foundation

public enum TreeRFMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void tree_rf_margin(
        device const float* state [[buffer(0)]],
        device const int* splitFeatures [[buffer(1)]],
        device const float* splitThresholds [[buffer(2)]],
        device const float* leafMass [[buffer(3)]],
        device float* out [[buffer(4)]],
        uint tid [[thread_position_in_grid]]
    ) {
        if (tid > 0) { return; }
        float vote0 = 0.001f;
        float vote1 = 0.001f;
        float vote2 = 0.001f;
        for (uint tree = 0; tree < 13; ++tree) {
            uint leaf = 0;
            for (uint depth = 0; depth < 3; ++depth) {
                const int feature = clamp(splitFeatures[tree * 3 + depth], 0, 15);
                const float threshold = splitThresholds[tree * 3 + depth];
                if (state[feature] > threshold) {
                    leaf |= (1u << depth);
                }
            }
            leaf = min(leaf, 7u);
            const uint offset = (tree * 8u + leaf) * 3u;
            const float m0 = leafMass[offset + 0u];
            const float m1 = leafMass[offset + 1u];
            const float m2 = leafMass[offset + 2u];
            const float total = max(m0 + m1 + m2, 1.0f);
            vote0 += m0 / total;
            vote1 += m1 / total;
            vote2 += m2 / total;
        }
        const float denominator = max(vote0 + vote1 + vote2, 1.0f);
        out[0] = clamp((vote1 - vote0) / denominator * 5.0f, -8.0f, 8.0f);
        out[1] = clamp(max(vote0, vote1) / denominator, 0.0f, 1.0f);
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "tree_rf",
        primaryBackends: [.swiftScalar, .accelerate, .metal],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy framework Random Forest branch with 13 seeded depth-3 trees, online leaf-class mass updates, threshold drift, move EMA, and native distribution outputs. Metal provides independent batched forest margin/confidence scoring; PyTorch, TensorFlow, and NLP are not suitable for this small non-neural tree ensemble."
    )
}
