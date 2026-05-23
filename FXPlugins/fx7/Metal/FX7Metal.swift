import FXDataEngine
import Foundation

public enum FX7Metal {
    public static let variantName = "fx7_metal"

    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    inline float fx7_clamp(float value, float lower, float upper) {
        if (!isfinite(value)) { return 0.0f; }
        return min(max(value, lower), upper);
    }

    inline float fx7_read(device const float* values, uint count, uint index) {
        if (index >= count) { return 0.0f; }
        return fx7_clamp(values[index], -50.0f, 50.0f);
    }

    inline float fx7_sign(float value) {
        return value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f);
    }

    kernel void fx7_core_signal_score(
        device const float* features [[buffer(0)]],
        device const float* window [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& featureCount [[buffer(3)]],
        constant uint& windowLength [[buffer(4)]],
        constant bool& hasVolume [[buffer(5)]],
        constant float& costScale [[buffer(6)]],
        uint gid [[thread_position_in_grid]]
    ) {
        if (gid > 0) { return; }

        const float shortReturn = fx7_read(features, featureCount, 0);
        const float mediumSlope = fx7_read(features, featureCount, 3);
        const float fastReturn = fx7_read(features, featureCount, 7);
        const float slowReturn = fx7_read(features, featureCount, 8);
        const float volatility = max(abs(fx7_read(features, featureCount, 4)) + 0.50f * abs(fx7_read(features, featureCount, 5)), 0.01f);
        const float contextSignal = fx7_read(features, featureCount, 12);
        const float volumeSignal = hasVolume
            ? fx7_clamp(0.65f * fx7_read(features, featureCount, 40) + 0.35f * fx7_read(features, featureCount, 6), -1.0f, 1.0f)
            : 0.0f;

        float weighted = 0.0f;
        float weightSum = 0.0f;
        const uint rowCount = min(windowLength, 16u);
        for (uint row = 0; row < rowCount; ++row) {
            const uint base = row * featureCount;
            const float weight = 1.0f / float(row + 1u);
            weighted += weight * (0.55f * fx7_read(window + base, featureCount, 7) + 0.45f * fx7_read(window + base, featureCount, 3));
            weightSum += weight;
        }
        const float windowMomentum = weightSum > 0.0f ? fx7_clamp(weighted / weightSum, -1.0f, 1.0f) : 0.0f;

        const float mtfEdge = fastReturn - slowReturn;
        const float trendInput = 0.38f * fastReturn +
            0.30f * slowReturn +
            0.18f * mediumSlope +
            0.09f * shortReturn +
            0.05f * windowMomentum;
        const float trend = tanh(2.4f * trendInput);
        const float breakout = tanh(4.0f * mtfEdge);
        const float signs[5] = {
            fx7_sign(shortReturn),
            fx7_sign(mediumSlope),
            fx7_sign(fastReturn),
            fx7_sign(slowReturn),
            fx7_sign(windowMomentum)
        };
        float signSum = 0.0f;
        float signCount = 0.0f;
        for (uint i = 0; i < 5u; ++i) {
            if (signs[i] != 0.0f) {
                signSum += signs[i];
                signCount += 1.0f;
            }
        }
        const float alignment = signCount > 0.0f ? fx7_clamp(signSum / signCount, -1.0f, 1.0f) : 0.0f;
        const float efficiency = fx7_clamp(abs(mtfEdge) / (volatility + abs(shortReturn) + 0.01f), 0.0f, 2.0f);
        const float volGate = 1.0f / (1.0f + max(volatility - 1.50f, 0.0f) * 0.60f);
        const float panicPenalty = fx7_clamp(max(volatility - 1.80f, 0.0f) * 0.25f, 0.0f, 0.45f);
        const float reversalPenalty = fx7_clamp(max(0.0f, -trend * shortReturn) * 0.35f, 0.0f, 0.35f);
        const float rawEdge = 0.56f * trend +
            0.20f * breakout +
            0.08f * alignment +
            0.06f * contextSignal +
            (hasVolume ? 0.08f * volumeSignal : 0.0f);
        const float boundedCostScale = max(costScale, 0.001f);
        const float costGate = 1.0f / (1.0f + exp(-8.0f * (abs(rawEdge) - boundedCostScale)));
        const float edge = fx7_clamp(rawEdge * volGate * costGate - fx7_sign(rawEdge) * (panicPenalty + reversalPenalty), -1.0f, 1.0f);
        const float confidence = fx7_clamp(
            0.25f +
                0.30f * abs(edge) +
                0.18f * min(efficiency, 1.0f) +
                0.12f * abs(alignment) +
                0.08f * abs(volumeSignal) +
                0.07f * costGate,
            0.0f,
            1.0f
        );

        output[0] = edge;
        output[1] = confidence;
        output[2] = volatility;
        output[3] = volumeSignal;
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "fx7",
        primaryBackends: [.swiftScalar, .metal],
        candidateBackends: [.swiftSIMD, .accelerate],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Metal implementation of FX7 signal scoring. The kernel scores volume-aware multi-timeframe momentum, alignment, volatility gating, and execution-cost acceptance for live buffer parity tests."
    )
}
