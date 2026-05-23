import FXDataEngine
import Foundation

public enum TrendVolBreakoutMetal {
    public static let variantName = "trend_vol_breakout_metal"

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

    inline float fxai_window_range(device const float* window, uint base, uint featureCount, uint start, uint count, uint feature, uint windowLength) {
        if (start >= windowLength || count == 0u) { return 0.0f; }
        const uint end = min(windowLength, start + count);
        float low = 3.402823466e+38f;
        float high = -3.402823466e+38f;
        for (uint i = start; i < end; ++i) {
            const float value = fxai_window_feature(window, base, featureCount, i, feature);
            low = min(low, value);
            high = max(high, value);
        }
        return max(high - low, 0.0f);
    }

    kernel void trend_vol_breakout_window_score(
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
        const float recentRange = fxai_window_range(window, windowBase, featureCount, 0u, 8u, 1u, windowLength);
        const float baselineRange = fxai_window_range(window, windowBase, featureCount, 8u, 24u, 1u, windowLength);
        const float expansion = recentRange - baselineRange;

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
        const float older = windowLength > 1u ? fxai_window_feature(window, windowBase, featureCount, min(windowLength - 1u, 15u), 1u) : recent;
        const float slope = recent - older;
        const float breakoutDirection = expansion >= 0.0f ? 1.0f : -1.0f;
        const float volumeSignal = hasVolume
            ? clamp(0.65f * fxai_read_feature(features + featureBase, featureCount, 40u) + 0.35f * fxai_read_feature(features + featureBase, featureCount, 6u), -8.0f, 8.0f)
            : 0.0f;
        const float margin = clamp(breakoutDirection * slope / volatility, -8.0f, 8.0f);
        const float confidence = clamp(0.35f + 0.38f * min(abs(expansion) / volatility, 1.0f) + 0.15f * abs(slope) + 0.12f / (1.0f + volatility) + 0.06f * abs(volumeSignal), 0.0f, 1.0f);

        const uint outBase = sampleIndex * 4u;
        output[outBase + 0u] = margin;
        output[outBase + 1u] = confidence;
        output[outBase + 2u] = volatility;
        output[outBase + 3u] = volumeSignal;
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "trend_vol_breakout",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate, .metal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Metal implementation for batched volatility-breakout window scans using recent-versus-baseline range expansion, rolling slope, volatility, and optional FXDatabase volume."
    )
}
