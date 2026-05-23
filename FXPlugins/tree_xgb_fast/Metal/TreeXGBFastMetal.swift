import FXDataEngine
import Foundation

public enum TreeXGBFastMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct XGBFastNode {
        int isLeaf;
        int feature;
        float threshold;
        int defaultLeft;
        int left;
        int right;
        float leafValue;
    };

    struct XGBFastTreeRef {
        uint nodeOffset;
        uint nodeCount;
        float weight;
        uint classIndex;
    };

    kernel void tree_xgb_fast_margins(
        device const float* features [[buffer(0)]],
        device const XGBFastNode* nodes [[buffer(1)]],
        device const XGBFastTreeRef* trees [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        device float* margins [[buffer(4)]],
        constant uint& treeCount [[buffer(5)]],
        uint classIndex [[thread_position_in_grid]]
    ) {
        if (classIndex >= 3) { return; }
        float margin = bias[classIndex];
        for (uint treeIndex = 0; treeIndex < treeCount; ++treeIndex) {
            const XGBFastTreeRef tree = trees[treeIndex];
            if (tree.classIndex != classIndex || tree.nodeCount == 0) { continue; }
            uint nodeIndex = 0;
            for (uint guard = 0; guard < 127; ++guard) {
                const XGBFastNode node = nodes[tree.nodeOffset + nodeIndex];
                if (node.isLeaf != 0) {
                    margin += tree.weight * node.leafValue;
                    break;
                }
                const float value = features[node.feature];
                const bool goLeft = isfinite(value) ? (value <= node.threshold) : (node.defaultLeft != 0);
                const int nextNode = goLeft ? node.left : node.right;
                if (nextNode < 0 || nextNode >= int(tree.nodeCount)) { break; }
                nodeIndex = uint(nextNode);
            }
        }
        margins[classIndex] = clamp(margin, -35.0f, 35.0f);
    }

    kernel void tree_xgb_fast_softmax3(
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
        pluginName: "tree_xgb_fast",
        primaryBackends: [.swiftScalar, .accelerate, .metal],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 tree_xgb_fast with online OvR gradient trees, ring-buffer training, binary calibration, and move/quality heads. Metal kernels provide independent batched tree-margin and softmax scoring; PyTorch, TensorFlow, and NLP are not suitable for this non-neural tree ensemble."
    )
}
