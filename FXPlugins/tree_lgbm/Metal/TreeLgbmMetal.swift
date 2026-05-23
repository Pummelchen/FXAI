import FXDataEngine
import Foundation

public enum TreeLgbmMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct LGBNode {
        int isLeaf;
        int feature;
        float threshold;
        int defaultLeft;
        int left;
        int right;
        float leafValue;
    };

    struct LGBTreeRef {
        uint nodeOffset;
        uint nodeCount;
        uint classIndex;
    };

    kernel void tree_lgbm_margins(
        device const float* features [[buffer(0)]],
        device const LGBNode* nodes [[buffer(1)]],
        device const LGBTreeRef* trees [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        device float* margins [[buffer(4)]],
        constant uint& treeCount [[buffer(5)]],
        uint classIndex [[thread_position_in_grid]]
    ) {
        if (classIndex >= 3) { return; }
        float margin = bias[classIndex];
        for (uint treeIndex = 0; treeIndex < treeCount; ++treeIndex) {
            const LGBTreeRef tree = trees[treeIndex];
            if (tree.classIndex != classIndex || tree.nodeCount == 0) { continue; }
            uint nodeIndex = 0;
            for (uint guard = 0; guard < 125; ++guard) {
                const LGBNode node = nodes[tree.nodeOffset + nodeIndex];
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
        margins[classIndex] = clamp(margin, -35.0f, 35.0f);
    }

    kernel void tree_lgbm_softmax3(
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
        pluginName: "tree_lgbm",
        primaryBackends: [.swiftScalar, .accelerate, .metal],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of legacy MQL5 tree_lgbm with per-class histogram leaves, GOSS-style sample selection, DART-compatible class ensembles, ternary calibration, move quantiles, validation gating, and quality heads. Metal kernels provide independent multiclass tree-margin and softmax scoring; PyTorch, TensorFlow, and NLP are not suitable for this non-neural tree ensemble."
    )
}
