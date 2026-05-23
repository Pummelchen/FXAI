import FXDataEngine
import Foundation

public enum StatEMDHHTMetal {
    public static let variantName = "stat_emd_hht_metal"

    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    inline float fxai_read_feature(device const float* values, uint count, uint index) {
        if (index >= count) { return 0.0f; }
        return clamp(values[index], -50.0f, 50.0f);
    }

    inline float fxai_window_feature(device const float* window, uint base, uint featureCount, uint row, uint feature) {
        return fxai_read_feature(window + base + row * featureCount, featureCount, feature);
    }

    kernel void stat_emd_hht_mode_proxy(
        device const float* features [[buffer(0)]],
        device const float* window [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& sampleCount [[buffer(3)]],
        constant uint& featureCount [[buffer(4)]],
        constant uint& windowLength [[buffer(5)]],
        constant bool& hasVolume [[buffer(6)]],
        uint sampleIndex [[thread_position_in_grid]]
    ) {
        if (sampleIndex >= sampleCount) { return; }

        const uint featureBase = sampleIndex * featureCount;
        const uint windowBase = sampleIndex * windowLength * featureCount;
        const uint recentCount = min(windowLength, 8u);
        float modeMean = 0.0f;
        float recentMean = 0.0f;
        float variance = 0.0f;
        for (uint i = 0; i < windowLength; ++i) {
            modeMean += fxai_window_feature(window, windowBase, featureCount, i, 1u);
        }
        modeMean = windowLength > 0u ? modeMean / float(windowLength) : fxai_read_feature(features + featureBase, featureCount, 1u);
        for (uint i = 0; i < recentCount; ++i) {
            recentMean += fxai_window_feature(window, windowBase, featureCount, i, 1u);
        }
        recentMean = recentCount > 0u ? recentMean / float(recentCount) : modeMean;
        for (uint i = 0; i < windowLength; ++i) {
            const float value = fxai_window_feature(window, windowBase, featureCount, i, 1u);
            const float d = value - modeMean;
            variance += d * d;
        }
        const float volatility = max(sqrt(variance / float(max(windowLength, 1u))), 0.01f);
        const float recent = recentCount > 0u ? fxai_window_feature(window, windowBase, featureCount, 0u, 1u) : modeMean;
        const float older = recentCount > 1u ? fxai_window_feature(window, windowBase, featureCount, recentCount - 1u, 1u) : recent;
        const float recentDelta = recent - older;
        const float meanShift = recentMean - modeMean;
        const float volumeSignal = hasVolume
            ? clamp(0.65f * fxai_read_feature(features + featureBase, featureCount, 40u) + 0.35f * fxai_read_feature(features + featureBase, featureCount, 6u), -8.0f, 8.0f)
            : 0.0f;
        const float margin = clamp(0.35f * recentDelta + 0.20f * meanShift + 0.45f * fxai_read_feature(features + featureBase, featureCount, 1u), -8.0f, 8.0f);
        const float strength = min((abs(recentDelta) + abs(meanShift)) / volatility, 2.0f) * 0.50f;
        const float confidence = clamp(0.36f + 0.36f * strength + 0.16f / (1.0f + volatility) + 0.06f * abs(volumeSignal), 0.0f, 1.0f);

        const uint outBase = sampleIndex * 4u;
        output[outBase + 0u] = margin;
        output[outBase + 1u] = confidence;
        output[outBase + 2u] = volatility;
        output[outBase + 3u] = volumeSignal;
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_emd_hht",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate, .metal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Metal implementation for batched empirical-mode/Hilbert-Huang proxy scans, including recent intrinsic-mode delta, mean-shift energy, volatility, and optional volume confirmation."
    )
}
