import FXDataEngine
import Foundation

public enum TrendTSMOMVolMetal {
    public static let variantName = "trend_tsmom_vol_metal"

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

    kernel void trend_tsmom_vol_window_score(
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
        const uint count16 = min(windowLength, 16u);
        const uint count8 = min(windowLength, 8u);
        float mean = 0.0f;
        float variance = 0.0f;
        for (uint i = 0; i < windowLength; ++i) {
            mean += fxai_window_feature(window, windowBase, featureCount, i, 1u);
        }
        mean = windowLength > 0u ? mean / float(windowLength) : fxai_read_feature(features + featureBase, featureCount, 1u);
        for (uint i = 0; i < windowLength; ++i) {
            const float value = fxai_window_feature(window, windowBase, featureCount, i, 1u);
            const float d = value - mean;
            variance += d * d;
        }
        const float volatility = max(sqrt(variance / float(max(windowLength, 1u))), 0.01f);
        const float recent = windowLength > 0u ? fxai_window_feature(window, windowBase, featureCount, 0u, 1u) : fxai_read_feature(features + featureBase, featureCount, 1u);
        const float older = count16 > 1u ? fxai_window_feature(window, windowBase, featureCount, count16 - 1u, 1u) : recent;
        const float delta16 = recent - older;
        const float older8 = count8 > 1u ? fxai_window_feature(window, windowBase, featureCount, count8 - 1u, 1u) : recent;
        const float delta8 = recent - older8;

        float slope = 0.0f;
        float weightSum = 0.0f;
        for (uint i = 0; i < windowLength; ++i) {
            const float weight = float(windowLength - i);
            slope += weight * fxai_window_feature(window, windowBase, featureCount, i, 1u);
            weightSum += weight;
        }
        slope = weightSum > 0.0f ? slope / weightSum - mean : fxai_read_feature(features + featureBase, featureCount, 1u);

        const float volumeSignal = hasVolume
            ? clamp(0.65f * fxai_read_feature(features + featureBase, featureCount, 40u) + 0.35f * fxai_read_feature(features + featureBase, featureCount, 6u), -8.0f, 8.0f)
            : 0.0f;
        const float margin = clamp((0.45f * delta16 + 0.35f * slope + 0.20f * delta8) / volatility, -8.0f, 8.0f);
        const float confidence = clamp(0.38f + 0.17f * min(abs(delta16) / volatility, 2.0f) + 0.17f * min(abs(slope) / volatility, 2.0f) + 0.16f / (1.0f + volatility) + 0.06f * abs(volumeSignal), 0.0f, 1.0f);

        const uint outBase = sampleIndex * 4u;
        output[outBase + 0u] = margin;
        output[outBase + 1u] = confidence;
        output[outBase + 2u] = volatility;
        output[outBase + 3u] = volumeSignal;
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "trend_tsmom_vol",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate, .metal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Metal implementation for batched time-series-momentum-over-volatility window scans. It mirrors the CPU feature path and applies volume only when the dataset has positive FXDatabase volume."
    )
}
