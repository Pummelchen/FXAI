import Foundation

public struct LifecycleComplianceValidationResult: Codable, Hashable, Sendable {
    public var isValid: Bool
    public var reason: String

    public init(isValid: Bool, reason: String = "") {
        self.isValid = isValid
        self.reason = reason
    }

    public static let valid = LifecycleComplianceValidationResult(isValid: true)
}

public enum LifecycleComplianceTools {
    public static func validatePrediction(_ prediction: PredictionV4) -> LifecycleComplianceValidationResult {
        guard prediction.classProbabilities.count == 3 else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "class_probs")
        }

        var probabilitySum = 0.0
        for probability in prediction.classProbabilities {
            guard probability.isFinite, probability >= 0.0 else {
                return LifecycleComplianceValidationResult(isValid: false, reason: "class_probs")
            }
            probabilitySum += probability
        }

        guard probabilitySum.isFinite, probabilitySum > 0.0 else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "class_sum")
        }
        guard abs(probabilitySum - 1.0) <= 0.02 else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "class_sum_norm")
        }
        guard prediction.moveMeanPoints.isFinite, prediction.moveMeanPoints >= 0.0 else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "move_mean")
        }
        guard prediction.moveQ25Points.isFinite,
              prediction.moveQ50Points.isFinite,
              prediction.moveQ75Points.isFinite,
              prediction.moveQ25Points >= 0.0,
              prediction.moveQ50Points >= prediction.moveQ25Points,
              prediction.moveQ75Points >= prediction.moveQ50Points else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "move_quantiles")
        }
        guard prediction.mfeMeanPoints.isFinite,
              prediction.mfeMeanPoints >= 0.0,
              prediction.maeMeanPoints.isFinite,
              prediction.maeMeanPoints >= 0.0 else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "path_excursions")
        }
        guard prediction.hitTimeFraction.isFinite,
              (0.0...1.0).contains(prediction.hitTimeFraction) else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "hit_time_frac")
        }
        guard prediction.pathRisk.isFinite,
              (0.0...1.0).contains(prediction.pathRisk) else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "path_risk")
        }
        guard prediction.fillRisk.isFinite,
              (0.0...1.0).contains(prediction.fillRisk) else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "fill_risk")
        }
        guard prediction.confidence.isFinite,
              (0.0...1.0).contains(prediction.confidence) else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "confidence")
        }
        guard prediction.reliability.isFinite,
              (0.0...1.0).contains(prediction.reliability) else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "reliability")
        }

        return .valid
    }

    public static func validatePredictionOutput(_ prediction: PredictionV4) -> LifecycleComplianceValidationResult {
        let base = validatePrediction(prediction)
        guard base.isValid else { return base }

        for probability in prediction.classProbabilities {
            guard probability <= 1.0 else {
                return LifecycleComplianceValidationResult(isValid: false, reason: "class_probs_range")
            }
        }
        guard prediction.moveMeanPoints.isFinite, prediction.moveMeanPoints >= 0.0 else {
            return LifecycleComplianceValidationResult(isValid: false, reason: "move_mean")
        }

        return .valid
    }

    public static func complianceSequenceBars(manifest: PluginManifestV4) -> Int {
        if manifest.maxSequenceBars <= 1 {
            return 1
        }

        var sequenceBars = manifest.maxSequenceBars
        if sequenceBars < manifest.minSequenceBars {
            sequenceBars = manifest.minSequenceBars
        }
        if sequenceBars < 2 {
            sequenceBars = 2
        }
        if sequenceBars > 16 {
            sequenceBars = 16
        }
        return sequenceBars
    }

    public static func complianceHorizon(manifest: PluginManifestV4, desiredHorizonMinutes: Int) -> Int {
        var horizon = desiredHorizonMinutes
        if horizon < manifest.minHorizonMinutes {
            horizon = manifest.minHorizonMinutes
        }
        if horizon > manifest.maxHorizonMinutes {
            horizon = manifest.maxHorizonMinutes
        }
        if horizon < 1 {
            horizon = 1
        }
        return horizon
    }

    public static func complianceWindow(features: [Double], sequenceBars: Int) -> [[Double]] {
        let sequence = min(max(sequenceBars, 1), FXDataEngineConstants.maxSequenceBars)
        guard sequence > 1 else { return [] }

        return (0..<(sequence - 1)).map { barIndex in
            let decay = max(1.0 - 0.08 * Double(barIndex + 1), 0.30)
            return features.map { $0 * decay }
        }
    }

    public static func predictionDistance(_ first: PredictionV4, _ second: PredictionV4) -> Double {
        var distance = 0.0
        for index in 0..<3 {
            let firstProbability = index < first.classProbabilities.count ? first.classProbabilities[index] : 0.0
            let secondProbability = index < second.classProbabilities.count ? second.classProbabilities[index] : 0.0
            distance += abs(firstProbability - secondProbability)
        }
        distance += 0.05 * abs(first.moveMeanPoints - second.moveMeanPoints)
        distance += 0.02 * abs(first.moveQ25Points - second.moveQ25Points)
        distance += 0.02 * abs(first.moveQ50Points - second.moveQ50Points)
        distance += 0.02 * abs(first.moveQ75Points - second.moveQ75Points)
        distance += 0.02 * abs(first.mfeMeanPoints - second.mfeMeanPoints)
        distance += 0.02 * abs(first.maeMeanPoints - second.maeMeanPoints)
        distance += 0.05 * abs(first.hitTimeFraction - second.hitTimeFraction)
        distance += 0.05 * abs(first.pathRisk - second.pathRisk)
        distance += 0.05 * abs(first.fillRisk - second.fillRisk)
        distance += 0.10 * abs(first.confidence - second.confidence)
        distance += 0.10 * abs(first.reliability - second.reliability)
        return distance
    }

    public static func complianceRandSymmetric(state: inout UInt64) -> Double {
        state = state &* 1_664_525 &+ 1_013_904_223
        let bucket = state % 20_001
        return (Double(bucket) / 10_000.0) - 1.0
    }
}
