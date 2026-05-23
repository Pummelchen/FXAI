import FXDataEngine
import Foundation

public enum MovingAverageCrossMetal {
    public static let variantName = "fxbacktest_moving_average_cross_metal"

    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    inline float fxai_volume_boost(float value, bool hasVolume) {
        if (!hasVolume) { return 0.0f; }
        return 0.10f * clamp(abs(value), 0.0f, 1.0f);
    }

    kernel void moving_average_cross_signal_grid(
        device const float* close [[buffer(0)]],
        device const float* volumeSignal [[buffer(1)]],
        device const uint* fastPeriods [[buffer(2)]],
        device const uint* slowPeriods [[buffer(3)]],
        device float* scores [[buffer(4)]],
        device float* directions [[buffer(5)]],
        constant uint& barCount [[buffer(6)]],
        constant uint& gridCount [[buffer(7)]],
        constant bool& hasVolume [[buffer(8)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        const uint bar = gid.x;
        const uint grid = gid.y;
        if (bar >= barCount || grid >= gridCount) { return; }

        const uint fast = max(fastPeriods[grid], 1u);
        const uint slow = max(slowPeriods[grid], fast + 1u);
        const uint outIndex = grid * barCount + bar;
        if (bar + 1u < slow) {
            scores[outIndex] = 0.0f;
            directions[outIndex] = 0.0f;
            return;
        }

        float fastSum = 0.0f;
        float slowSum = 0.0f;
        for (uint i = 0; i < fast; ++i) {
            fastSum += close[bar - i];
        }
        for (uint i = 0; i < slow; ++i) {
            slowSum += close[bar - i];
        }

        const float fastMA = fastSum / float(fast);
        const float slowMA = slowSum / float(slow);
        const float edge = fastMA - slowMA;
        const float priceScale = max(abs(slowMA), 0.000001f);
        const float normalizedEdge = edge / priceScale;
        const float volumeValue = hasVolume ? volumeSignal[bar] : 0.0f;
        const float strength = clamp(abs(normalizedEdge) * 32.0f + fxai_volume_boost(volumeValue, hasVolume), 0.0f, 1.0f);
        scores[outIndex] = strength;
        directions[outIndex] = edge > 0.0f ? 1.0f : (edge < 0.0f ? -1.0f : 0.0f);
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "fxbacktest_moving_average_cross",
        primaryBackends: [.swiftScalar, .metal],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Metal implementation for offline moving-average parameter sweeps. The kernel scores fast/slow MA cross strength per bar and applies volume confidence only when FXDatabase volume is present."
    )
}
