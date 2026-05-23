import FXDataEngine
import Foundation

public enum StatVMDMetal {
    public static let variantName = "stat_vmd_metal"

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

    kernel void stat_vmd_mode_proxy(
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
        float fastMode = windowLength > 0u ? fxai_window_feature(window, windowBase, featureCount, windowLength - 1u, 1u) : fxai_read_feature(features + featureBase, featureCount, 1u);
        float slowMode = windowLength > 0u ? fxai_window_feature(window, windowBase, featureCount, windowLength - 1u, 2u) : fxai_read_feature(features + featureBase, featureCount, 2u);
        float mean = 0.0f;
        float variance = 0.0f;
        for (int i = int(windowLength) - 1; i >= 0; --i) {
            const uint row = uint(i);
            fastMode = 0.55f * fxai_window_feature(window, windowBase, featureCount, row, 1u) + 0.45f * fastMode;
            slowMode = 0.85f * fxai_window_feature(window, windowBase, featureCount, row, 2u) + 0.15f * slowMode;
            mean += fxai_window_feature(window, windowBase, featureCount, row, 1u);
        }
        mean = windowLength > 0u ? mean / float(windowLength) : fxai_read_feature(features + featureBase, featureCount, 1u);
        for (uint i = 0; i < windowLength; ++i) {
            const float value = fxai_window_feature(window, windowBase, featureCount, i, 1u);
            const float d = value - mean;
            variance += d * d;
        }
        const float volatility = max(sqrt(variance / float(max(windowLength, 1u))), 0.01f);
        const float slope = windowLength > 1u
            ? fxai_window_feature(window, windowBase, featureCount, 0u, 1u) - fxai_window_feature(window, windowBase, featureCount, min(windowLength - 1u, 15u), 1u)
            : fxai_read_feature(features + featureBase, featureCount, 1u);
        const float volumeSignal = hasVolume
            ? clamp(0.65f * fxai_read_feature(features + featureBase, featureCount, 40u) + 0.35f * fxai_read_feature(features + featureBase, featureCount, 6u), -8.0f, 8.0f)
            : 0.0f;
        const float margin = clamp(0.30f * fastMode + 0.30f * slowMode + 0.40f * slope, -8.0f, 8.0f);
        const float decompositionStrength = min((abs(fastMode) + abs(slowMode) + 0.50f * abs(slope)) / volatility, 2.0f) * 0.50f;
        const float confidence = clamp(0.36f + 0.36f * decompositionStrength + 0.16f / (1.0f + volatility) + 0.06f * abs(volumeSignal), 0.0f, 1.0f);

        const uint outBase = sampleIndex * 4u;
        output[outBase + 0u] = margin;
        output[outBase + 1u] = confidence;
        output[outBase + 2u] = volatility;
        output[outBase + 3u] = volumeSignal;
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_vmd",
        primaryBackends: [.swiftScalar, .swiftSIMD, .accelerate, .metal],
        candidateBackends: [],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Metal implementation for batched VMD proxy scans, computing fast/slow mode estimates, rolling variance, slope, and optional FXDatabase volume confirmation."
    )
}
