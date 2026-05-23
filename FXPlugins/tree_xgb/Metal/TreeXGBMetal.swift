import FXDataEngine
import Foundation

public enum TreeXGBMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct XGBNode {
        int isLeaf;
        int feature;
        float threshold;
        int defaultLeft;
        int left;
        int right;
        float leafValue;
    };

    struct XGBTreeRef {
        uint nodeOffset;
        uint nodeCount;
    };

    kernel void tree_xgb_margin(
        device const float* features [[buffer(0)]],
        device const XGBNode* nodes [[buffer(1)]],
        device const XGBTreeRef* trees [[buffer(2)]],
        constant float& bias [[buffer(3)]],
        device float* marginOut [[buffer(4)]],
        constant uint& treeCount [[buffer(5)]],
        uint tid [[thread_position_in_grid]]
    ) {
        if (tid > 0) { return; }
        float margin = bias;
        for (uint treeIndex = 0; treeIndex < treeCount; ++treeIndex) {
            const XGBTreeRef tree = trees[treeIndex];
            if (tree.nodeCount == 0) { continue; }
            uint nodeIndex = 0;
            for (uint guard = 0; guard < 31; ++guard) {
                const XGBNode node = nodes[tree.nodeOffset + nodeIndex];
                if (node.isLeaf != 0) {
                    margin += node.leafValue;
                    break;
                }
                const float value = features[node.feature];
                const bool goLeft = isfinite(value) ? (value <= node.threshold) : (node.defaultLeft != 0);
                const int nextNode = goLeft ? node.left : node.right;
                if (nextNode < 0 || nextNode >= int(tree.nodeCount)) { break; }
                nodeIndex = uint(nextNode);
            }
        }
        marginOut[0] = clamp(margin, -35.0f, 35.0f);
    }

    kernel void tree_xgb_sigmoid(
        device const float* margin [[buffer(0)]],
        device float* probability [[buffer(1)]],
        uint tid [[thread_position_in_grid]]
    ) {
        if (tid > 0) { return; }
        const float z = clamp(margin[0], -35.0f, 35.0f);
        probability[0] = 1.0f / (1.0f + exp(-z));
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "tree_xgb",
        primaryBackends: [.swiftScalar, .accelerate, .metal],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 tree_xgb with binary XGBoost-style online trees, leaf class-mass blending, calibration, move statistics, and quality heads. Metal kernels provide independent tree-margin and sigmoid scoring; PyTorch, TensorFlow, and NLP are not suitable for this non-neural tree ensemble."
    )
}
