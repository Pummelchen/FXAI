import FXDataEngine
import Foundation

public enum DistQuantileMetal {
    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    inline float fxai_softplus(float z) {
        if (z > 30.0f) { return z; }
        if (z < -30.0f) { return exp(z); }
        return log(1.0f + exp(z));
    }

    kernel void dist_quantile_head_score(
        device const float* features [[buffer(0)]],
        device const float* medianWeights [[buffer(1)]],
        device const float* upWeights [[buffer(2)]],
        device const float* downWeights [[buffer(3)]],
        device const float* sessionBias [[buffer(4)]],
        device const float* regimeScale [[buffer(5)]],
        device float* quantiles [[buffer(6)]],
        constant uint& featureCount [[buffer(7)]],
        constant uint& sessionIndex [[buffer(8)]],
        constant uint& regimeIndex [[buffer(9)]],
        uint row [[thread_position_in_grid]]
    ) {
        const uint qCount = 9;
        const uint mid = 4;
        const uint sideCount = 4;
        const uint baseFeature = row * featureCount;
        const uint baseQuantile = row * qCount;
        float median = sessionBias[sessionIndex];
        for (uint i = 0; i < featureCount; ++i) {
            median += medianWeights[i] * features[baseFeature + i];
        }
        float scale = clamp(regimeScale[regimeIndex], 0.40f, 2.80f);
        quantiles[baseQuantile + mid] = median;

        for (uint side = 0; side < sideCount; ++side) {
            float zu = 0.0f;
            float zd = 0.0f;
            const uint sideBase = side * featureCount;
            for (uint i = 0; i < featureCount; ++i) {
                const float x = features[baseFeature + i];
                zu += upWeights[sideBase + i] * x;
                zd += downWeights[sideBase + i] * x;
            }
            quantiles[baseQuantile + mid + 1 + side] =
                quantiles[baseQuantile + mid + side] + scale * fxai_softplus(zu);
            quantiles[baseQuantile + mid - 1 - side] =
                quantiles[baseQuantile + mid - side] - scale * fxai_softplus(zd);
        }

        for (uint k = 1; k < qCount; ++k) {
            const float prev = quantiles[baseQuantile + k - 1] + 0.0001f;
            if (quantiles[baseQuantile + k] < prev) {
                quantiles[baseQuantile + k] = prev;
            }
        }
    }

    kernel void dist_quantile_class_score(
        device const float* classFeatures [[buffer(0)]],
        device const float* classWeights [[buffer(1)]],
        device float* logits [[buffer(2)]],
        uint gid [[thread_position_in_grid]]
    ) {
        const uint zfCount = 8;
        const uint classCount = 3;
        const uint row = gid / classCount;
        const uint classIndex = gid % classCount;
        float sum = 0.0f;
        const uint featureBase = row * zfCount;
        const uint weightBase = classIndex * zfCount;
        for (uint i = 0; i < zfCount; ++i) {
            sum += classWeights[weightBase + i] * classFeatures[featureBase + i];
        }
        logits[gid] = clamp(sum, -30.0f, 30.0f);
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "dist_quantile",
        primaryBackends: [.swiftSIMD, .accelerate],
        candidateBackends: [.metal],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU implementation of the native quantile-head distribution model. Metal kernels cover batched quantile-head scoring and class-feature scoring; PyTorch, TensorFlow, and NLP are not suitable for this non-neural online quantile learner."
    )
}
