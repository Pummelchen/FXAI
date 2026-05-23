import FXDataEngine
import Foundation

public enum TreeCatboostMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct CatLevelSplit {
        int feature;
        float threshold;
        int defaultLeft;
    };

    struct CatTreeRef {
        uint splitOffset;
        uint leafOffset;
        uint depth;
    };

    kernel void tree_catboost_margins(
        device const float* features [[buffer(0)]],
        device const CatLevelSplit* splits [[buffer(1)]],
        device const CatTreeRef* trees [[buffer(2)]],
        device const float* leafValues [[buffer(3)]],
        device const float* bias [[buffer(4)]],
        device float* margins [[buffer(5)]],
        constant uint& treeCount [[buffer(6)]],
        uint classIndex [[thread_position_in_grid]]
    ) {
        if (classIndex >= 3) { return; }
        float margin = bias[classIndex];
        for (uint treeIndex = 0; treeIndex < treeCount; ++treeIndex) {
            const CatTreeRef tree = trees[treeIndex];
            uint leaf = 0;
            for (uint depth = 0; depth < tree.depth && depth < 6; ++depth) {
                const CatLevelSplit split = splits[tree.splitOffset + depth];
                const float value = features[split.feature];
                const bool goLeft = isfinite(value) ? (value <= split.threshold) : (split.defaultLeft != 0);
                leaf = (leaf << 1) | (goLeft ? 0u : 1u);
            }
            margin += leafValues[(tree.leafOffset + leaf) * 3u + classIndex];
        }
        margins[classIndex] = clamp(margin, -35.0f, 35.0f);
    }

    kernel void tree_catboost_softmax3(
        device const float* margins [[buffer(0)]],
        device float* probabilities [[buffer(1)]],
        uint tid [[thread_position_in_grid]]
    ) {
        if (tid > 0) { return; }
        const float m = max(margins[0], max(margins[1], margins[2]));
        const float e0 = exp(clamp(margins[0] - m, -30.0f, 30.0f));
        const float e1 = exp(clamp(margins[1] - m, -30.0f, 30.0f));
        const float e2 = exp(clamp(margins[2] - m, -30.0f, 30.0f));
        const float s = max(e0 + e1 + e2, 0.000001f);
        probabilities[0] = e0 / s;
        probabilities[1] = e1 / s;
        probabilities[2] = e2 / s;
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "tree_catboost",
        primaryBackends: [.swiftScalar, .accelerate, .metal],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 tree_catboost with ordered CTR features, symmetric multiclass trees, calibration, drift-aware rebuild cadence, move estimates, and quality heads. Metal kernels provide independent symmetric-tree margin and softmax scoring; PyTorch, TensorFlow, and NLP are not suitable for this non-neural tree ensemble."
    )
}
