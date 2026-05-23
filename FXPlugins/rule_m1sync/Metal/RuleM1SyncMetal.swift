import FXDataEngine
import Foundation

public enum RuleM1SyncMetal {
    public static let variantName = "rule_m1sync_metal"

    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    inline float fxai_sigmoidf(float z) {
        return 1.0f / (1.0f + exp(-clamp(z, -32.0f, 32.0f)));
    }

    kernel void rule_m1sync_chain_scan(
        device const long* utcTimestamps [[buffer(0)]],
        device const float* open [[buffer(1)]],
        device const float* close [[buffer(2)]],
        device const float* volume [[buffer(3)]],
        device const long* sampleTimes [[buffer(4)]],
        device float* labels [[buffer(5)]],
        device float* expectedMovePoints [[buffer(6)]],
        device float* strength [[buffer(7)]],
        device float* volumeConfirmation [[buffer(8)]],
        constant uint& barCount [[buffer(9)]],
        constant uint& sampleCount [[buffer(10)]],
        constant uint& requestedSyncBars [[buffer(11)]],
        constant float& priceCostPoints [[buffer(12)]],
        constant float& minMovePoints [[buffer(13)]],
        constant float& pointValue [[buffer(14)]],
        constant bool& hasVolume [[buffer(15)]],
        uint sampleIndex [[thread_position_in_grid]]
    ) {
        if (sampleIndex >= sampleCount) { return; }

        const uint syncBars = clamp(requestedSyncBars, 2u, 12u);
        if (barCount < syncBars + 1u) {
            labels[sampleIndex] = 0.0f;
            expectedMovePoints[sampleIndex] = 0.0f;
            strength[sampleIndex] = 0.0f;
            volumeConfirmation[sampleIndex] = 0.0f;
            return;
        }

        const long sampleTime = sampleTimes[sampleIndex];
        int closedIndex = -1;
        for (uint i = 0; i < barCount; ++i) {
            if (utcTimestamps[i] <= sampleTime) {
                closedIndex = int(i);
            }
        }
        if (closedIndex < 0) {
            closedIndex = int(max(barCount, 1u)) - 2;
        }
        closedIndex = min(closedIndex, int(barCount) - 1);
        if (closedIndex < int(syncBars) - 1) {
            labels[sampleIndex] = 0.0f;
            expectedMovePoints[sampleIndex] = 0.0f;
            strength[sampleIndex] = 0.0f;
            volumeConfirmation[sampleIndex] = 0.0f;
            return;
        }

        const uint start = uint(closedIndex) - syncBars + 1u;
        const float point = max(pointValue, 0.000001f);
        const float cost = max(priceCostPoints, 0.0f);
        const float minimumMove = max(minMovePoints, max(0.10f, cost));
        const float epsilon = max(0.10f * point, 0.02f * cost * point);

        bool upChain = true;
        bool downChain = true;
        float minStepPoints = 3.402823466e+38f;
        float previous = close[start];
        if (previous <= 0.0f) {
            upChain = false;
            downChain = false;
        }
        for (uint index = start + 1u; index <= uint(closedIndex); ++index) {
            const float current = close[index];
            const float step = current - previous;
            if (current <= 0.0f) {
                upChain = false;
                downChain = false;
            }
            if (step <= epsilon) { upChain = false; }
            if (step >= -epsilon) { downChain = false; }
            minStepPoints = min(minStepPoints, abs(step) / point);
            previous = current;
        }

        float nowPrice = close[uint(closedIndex)];
        if (uint(closedIndex) + 1u < barCount && open[uint(closedIndex) + 1u] > 0.0f) {
            nowPrice = open[uint(closedIndex) + 1u];
        }
        const float finalStep = nowPrice - close[uint(closedIndex)];
        if (finalStep <= epsilon) { upChain = false; }
        if (finalStep >= -epsilon) { downChain = false; }
        minStepPoints = min(minStepPoints, abs(finalStep) / point);

        if (!upChain && !downChain) {
            labels[sampleIndex] = 0.0f;
            expectedMovePoints[sampleIndex] = 0.0f;
            strength[sampleIndex] = 0.0f;
            volumeConfirmation[sampleIndex] = 0.0f;
            return;
        }

        float volumeScore = 0.0f;
        if (hasVolume) {
            float sum = 0.0f;
            uint count = 0u;
            for (uint index = start; index <= uint(closedIndex); ++index) {
                if (volume[index] > 0.0f) {
                    sum += volume[index];
                    count += 1u;
                }
            }
            if (count > 0u && sum > 0.0f) {
                const float average = sum / float(count);
                volumeScore = clamp((volume[uint(closedIndex)] / average - 0.70f) / 0.80f, 0.0f, 1.0f);
                volumeScore = max(volumeScore, 0.35f);
            }
        }

        const float totalPoints = abs(nowPrice - close[start]) / point;
        const float edgePoints = totalPoints - cost;
        const float totalScore = fxai_sigmoidf(edgePoints / max(minimumMove, 0.10f));
        const float stepScore = fxai_sigmoidf((minStepPoints / max(minimumMove, 0.10f)) - 0.15f);
        labels[sampleIndex] = upChain ? 1.0f : -1.0f;
        expectedMovePoints[sampleIndex] = max(totalPoints, 0.0f);
        strength[sampleIndex] = clamp(0.60f * totalScore + 0.40f * stepScore + 0.04f * volumeScore, 0.0f, 1.0f);
        volumeConfirmation[sampleIndex] = volumeScore;
    }
    """

    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "rule_m1sync",
        primaryBackends: [.swiftScalar, .metal],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Metal implementation for batched M1 synchronization-chain scans over FXDatabase M1 OHLCV arrays with volume confirmation when volume is present."
    )
}
