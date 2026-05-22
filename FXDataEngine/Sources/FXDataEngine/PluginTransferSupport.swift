import Foundation

public struct PluginTransferSlotState: Codable, Hashable, Sendable {
    public var observations: Double
    public var alignment: Double
    public var lead: Double
    public var moveScale: Double

    public init(
        observations: Double = 0.0,
        alignment: Double = 0.0,
        lead: Double = 0.5,
        moveScale: Double = 1.0
    ) {
        self.observations = max(0.0, fxSafeFinite(observations))
        self.alignment = fxClamp(fxSafeFinite(alignment), -1.0, 1.0)
        self.lead = fxClamp(fxSafeFinite(lead, fallback: 0.5), 0.0, 1.0)
        self.moveScale = fxClamp(fxSafeFinite(moveScale, fallback: 1.0), 0.50, 2.50)
    }
}

public struct PluginTransferSlotPriors: Codable, Hashable, Sendable {
    public var classProbabilities: [Double]
    public var moveScaleMultiplier: Double
    public var reliabilityBoost: Double
    public var domainWeight: Double

    public init(
        classProbabilities: [Double] = [0.0, 0.0, 0.0],
        moveScaleMultiplier: Double = 1.0,
        reliabilityBoost: Double = 0.0,
        domainWeight: Double = 0.0
    ) {
        self.classProbabilities = (0..<PluginTernaryCalibrator.classCount).map { index in
            index < classProbabilities.count ? max(0.0, fxSafeFinite(classProbabilities[index])) : 0.0
        }
        self.moveScaleMultiplier = fxClamp(fxSafeFinite(moveScaleMultiplier, fallback: 1.0), 0.70, 1.50)
        self.reliabilityBoost = fxClamp(fxSafeFinite(reliabilityBoost), -0.15, 0.20)
        self.domainWeight = max(0.0, fxSafeFinite(domainWeight))
    }
}

public enum PluginTransferSupportTools {
    public static func inputFeature(_ x: [Double], featureIndex: Int) -> Double {
        let inputIndex = featureIndex + 1
        guard inputIndex >= 0, inputIndex < x.count else { return 0.0 }
        return fxSafeFinite(x[inputIndex])
    }

    public static func transferSlotSignal(x: [Double], slot: Int) -> Double {
        guard slot >= 0, slot < FXDataEngineConstants.contextTopSymbols else { return 0.0 }
        let base = 50 + slot * 4
        let contextReturn = inputFeature(x, featureIndex: base)
        let contextLag = inputFeature(x, featureIndex: base + 1)
        let contextRelative = inputFeature(x, featureIndex: base + 2)
        let signal = 0.30 * contextReturn + 0.50 * contextLag + 0.20 * contextRelative
        return PluginSupportTools.clipSymmetric(signal, limit: 4.0)
    }

    public static func hasSharedAdapterSignal(_ adapterInput: [Double]) -> Bool {
        guard adapterInput.count >= FXDataEngineConstants.sharedTransferFeatures else { return false }
        var magnitude = 0.0
        for index in 1..<FXDataEngineConstants.sharedTransferFeatures {
            magnitude += abs(fxSafeFinite(adapterInput[index]))
        }
        return magnitude > 1e-6
    }

    public static func sharedAdapterSignalStrength(_ adapterInput: [Double]) -> Double {
        guard adapterInput.count >= FXDataEngineConstants.sharedTransferFeatures else { return 0.0 }
        return fxClamp(
            0.14 +
                0.07 * value(adapterInput, 4) +
                0.04 * abs(value(adapterInput, 8)) +
                0.04 * abs(value(adapterInput, 11)) +
                0.04 * abs(value(adapterInput, 12)) +
                0.04 * abs(value(adapterInput, 15)) +
                0.03 * abs(value(adapterInput, 18)) -
                0.02 * abs(value(adapterInput, 19)) +
                0.05 * abs(value(adapterInput, 20)) +
                0.05 * abs(value(adapterInput, 21)) +
                0.04 * value(adapterInput, 22) +
                0.04 * abs(value(adapterInput, 23)) +
                0.04 * abs(value(adapterInput, 24)) +
                0.03 * abs(value(adapterInput, 25)) +
                0.03 * abs(value(adapterInput, 26)) -
                0.03 * value(adapterInput, 27),
            0.0,
            0.62
        )
    }

    public static func blendTransferSlotPriors(
        x: [Double],
        slots: [PluginTransferSlotState]
    ) -> PluginTransferSlotPriors {
        let coverage = fxClamp(0.5 + 0.5 * inputFeature(x, featureIndex: 65), 0.0, 1.0)
        var domainBuy = 0.0
        var domainSell = 0.0
        var domainSkip = 0.0
        var domainMove = 0.0
        var domainReliability = 0.0
        var domainWeight = 0.0

        for slot in 0..<FXDataEngineConstants.contextTopSymbols {
            let state = slot < slots.count ? slots[slot] : PluginTransferSlotState()
            guard state.observations > 1e-6 else { continue }

            let base = 50 + slot * 4
            let contextCorrelation = inputFeature(x, featureIndex: base + 3)
            let signal = transferSlotSignal(x: x, slot: slot)
            let observationTrust = fxClamp(state.observations / 24.0, 0.0, 1.0)
            let weight = fxClamp(
                observationTrust *
                    (0.25 + 0.75 * abs(contextCorrelation)) *
                    (0.20 + 0.80 * coverage) *
                    (0.20 + 0.80 * abs(signal)),
                0.0,
                2.0
            )
            guard weight > 1e-6 else { continue }

            let alignment = fxClamp(state.alignment, -1.0, 1.0)
            let lead = fxClamp(state.lead, 0.0, 1.0)
            let moveScale = fxClamp(state.moveScale, 0.50, 2.50)
            var buyPrior = 0.10
            var sellPrior = 0.10
            let skipPrior = fxClamp(0.55 - 0.10 * abs(signal), 0.05, 0.80)
            if signal > 0.0 {
                buyPrior = fxClamp(0.45 + 0.22 * alignment + 0.15 * lead + 0.08 * abs(signal), 0.05, 0.95)
                sellPrior = fxClamp(0.20 - 0.12 * alignment, 0.02, 0.60)
            } else if signal < 0.0 {
                sellPrior = fxClamp(0.45 - 0.22 * alignment + 0.15 * lead + 0.08 * abs(signal), 0.05, 0.95)
                buyPrior = fxClamp(0.20 + 0.12 * alignment, 0.02, 0.60)
            }

            let priorSum = max(buyPrior + sellPrior + skipPrior, 1e-12)
            buyPrior /= priorSum
            sellPrior /= priorSum
            let normalizedSkipPrior = skipPrior / priorSum

            domainBuy += weight * buyPrior
            domainSell += weight * sellPrior
            domainSkip += weight * normalizedSkipPrior
            domainMove += weight * moveScale
            domainReliability += weight * (0.50 + 0.25 * abs(alignment) + 0.25 * lead)
            domainWeight += weight
        }

        guard domainWeight > 1e-6 else {
            return PluginTransferSlotPriors()
        }

        return PluginTransferSlotPriors(
            classProbabilities: [
                domainSell / domainWeight,
                domainBuy / domainWeight,
                domainSkip / domainWeight
            ],
            moveScaleMultiplier: fxClamp(domainMove / domainWeight, 0.70, 1.50),
            reliabilityBoost: fxClamp((domainReliability / domainWeight) - 0.50, -0.15, 0.20),
            domainWeight: domainWeight
        )
    }

    public static func updatedCrossSymbolTransferSlots(
        x: [Double],
        currentSlots: [PluginTransferSlotState],
        movePoints: Double,
        sampleWeight: Double,
        minMovePoints: Double
    ) -> [PluginTransferSlotState] {
        var slots = normalizedSlots(currentSlots)
        updateCrossSymbolTransferSlots(
            x: x,
            slots: &slots,
            movePoints: movePoints,
            sampleWeight: sampleWeight,
            minMovePoints: minMovePoints
        )
        return slots
    }

    public static func updateCrossSymbolTransferSlots(
        x: [Double],
        slots: inout [PluginTransferSlotState],
        movePoints: Double,
        sampleWeight: Double,
        minMovePoints: Double
    ) {
        slots = normalizedSlots(slots)
        let moveSign = fxSign(fxSafeFinite(movePoints))
        guard abs(moveSign) > 1e-9 else { return }

        let coverage = fxClamp(0.5 + 0.5 * inputFeature(x, featureIndex: 65), 0.0, 1.0)
        let moveScale = max(abs(fxSafeFinite(movePoints)), max(fxSafeFinite(minMovePoints), 0.10))
        let boundedSampleWeight = fxClamp(fxSafeFinite(sampleWeight, fallback: 1.0), 0.25, 4.0)
        for slot in 0..<FXDataEngineConstants.contextTopSymbols {
            let base = 50 + slot * 4
            let contextCorrelation = inputFeature(x, featureIndex: base + 3)
            let signal = transferSlotSignal(x: x, slot: slot)
            guard abs(signal) > 1e-6 else { continue }

            let trust = fxClamp(
                (0.30 + 0.70 * abs(contextCorrelation)) *
                    (0.20 + 0.80 * coverage) *
                    boundedSampleWeight,
                0.02,
                2.50
            )
            let observations = slots[slot].observations
            let alpha = fxClamp(0.05 * trust / sqrt(1.0 + 0.02 * observations), 0.01, 0.20)
            let alignmentTarget = fxClamp(fxSign(signal) * moveSign, -1.0, 1.0)
            let leadTarget = fxClamp(0.5 + 0.5 * fxSign(inputFeature(x, featureIndex: base + 1)) * moveSign, 0.0, 1.0)
            let moveTarget = fxClamp(moveScale / max(abs(signal), 0.10), 0.50, 2.50)

            if observations <= 1e-6 {
                slots[slot].alignment = alignmentTarget
                slots[slot].lead = leadTarget
                slots[slot].moveScale = moveTarget
            } else {
                slots[slot].alignment = fxClamp(
                    (1.0 - alpha) * slots[slot].alignment + alpha * alignmentTarget,
                    -1.0,
                    1.0
                )
                slots[slot].lead = fxClamp((1.0 - alpha) * slots[slot].lead + alpha * leadTarget, 0.0, 1.0)
                slots[slot].moveScale = fxClamp(
                    (1.0 - alpha) * slots[slot].moveScale + alpha * moveTarget,
                    0.50,
                    2.50
                )
            }
            slots[slot].observations = min(observations + trust, 5_000.0)
        }
    }

    private static func normalizedSlots(_ slots: [PluginTransferSlotState]) -> [PluginTransferSlotState] {
        (0..<FXDataEngineConstants.contextTopSymbols).map { index in
            index < slots.count ? PluginTransferSlotState(
                observations: slots[index].observations,
                alignment: slots[index].alignment,
                lead: slots[index].lead,
                moveScale: slots[index].moveScale
            ) : PluginTransferSlotState()
        }
    }

    private static func value(_ values: [Double], _ index: Int) -> Double {
        guard index >= 0, index < values.count else { return 0.0 }
        return fxSafeFinite(values[index])
    }
}
