import FXDataEngine
import Foundation

public enum MemRetrdiffMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void mem_retrdiff_distances(
        device const float* queryEmbedding [[buffer(0)]],
        device const float* memoryEmbeddings [[buffer(1)]],
        device float* distances [[buffer(2)]],
        constant uint& embeddingDimension [[buffer(3)]],
        uint memoryIndex [[thread_position_in_grid]]
    ) {
        float sum = 0.0f;
        const uint base = memoryIndex * embeddingDimension;
        for (uint i = 0; i < embeddingDimension; ++i) {
            const float d = queryEmbedding[i] - memoryEmbeddings[base + i];
            sum += d * d;
        }
        distances[memoryIndex] = sum;
    }

    kernel void mem_retrdiff_weighted_vote(
        device const float* distances [[buffer(0)]],
        device const float* labelMass [[buffer(1)]],
        device const float* futureMove [[buffer(2)]],
        device const float* moveVariance [[buffer(3)]],
        device float* output [[buffer(4)]],
        constant uint& memoryCount [[buffer(5)]],
        uint tid [[thread_position_in_grid]]
    ) {
        if (tid > 0) { return; }
        float cls0 = 0.0f;
        float cls1 = 0.0f;
        float cls2 = 0.0f;
        float sw = 0.0f;
        float mv = 0.0f;
        float mv2 = 0.0f;
        for (uint i = 0; i < memoryCount; ++i) {
            const float w = 1.0f / max(0.03f, distances[i]);
            const uint cbase = i * 3;
            cls0 += w * labelMass[cbase + 0];
            cls1 += w * labelMass[cbase + 1];
            cls2 += w * labelMass[cbase + 2];
            mv += w * futureMove[i];
            mv2 += w * (moveVariance[i] + futureMove[i] * futureMove[i]);
            sw += w;
        }
        if (sw <= 0.0f) {
            output[0] = 0.10f;
            output[1] = 0.10f;
            output[2] = 0.80f;
            output[3] = 0.0f;
            output[4] = 0.0f;
            return;
        }
        output[0] = cls0 / sw;
        output[1] = cls1 / sw;
        output[2] = cls2 / sw;
        const float mean = mv / sw;
        const float second = mv2 / sw;
        output[3] = max(0.0f, mean);
        output[4] = sqrt(max(0.0f, second - mean * mean) + 0.000001f);
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "mem_retrdiff",
        primaryBackends: [.accelerate],
        candidateBackends: [.metal],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the MQL5 retrieval-difference memory. Metal kernels cover batched embedding distance scans and weighted memory voting; PyTorch, TensorFlow, and NLP are not suitable for this non-neural memory model."
    )
}
