import FXDataEngine
import Foundation

enum FXBacktestDemoPredictionTools {
    static func feature(_ request: PredictRequestV4, _ index: Int) -> Double {
        guard index >= 0, index < request.x.count else { return 0.0 }
        return fxSafeFinite(request.x[index])
    }

    static func directionalPrediction(
        label: LabelClass,
        strength: Double,
        moveSeed: Double,
        reliability: Double
    ) -> PredictionV4 {
        let clampedStrength = fxClamp(strength, 0.0, 1.0)
        let directional = fxClamp(0.55 + 0.40 * clampedStrength, 0.55, 0.95)
        let opposite = 0.05
        let skip = max(0.02, 1.0 - directional - opposite)
        let probabilities: [Double]
        switch label {
        case .buy:
            probabilities = normalized([opposite, directional, skip])
        case .sell:
            probabilities = normalized([directional, opposite, skip])
        case .skip:
            probabilities = [0.08, 0.08, 0.84]
        }

        let meanMove = label == .skip ? 0.0 : max(1.0, moveSeed)
        let sigma = max(0.10, 0.30 * meanMove)
        let q25 = max(0.0, meanMove - 0.50 * sigma)
        let q50 = max(q25, meanMove)
        let q75 = max(q50, meanMove + 0.50 * sigma)
        return PredictionV4(
            classProbabilities: probabilities,
            moveMeanPoints: meanMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: meanMove,
            maeMeanPoints: max(0.0, 0.35 * meanMove),
            hitTimeFraction: 1.0,
            pathRisk: probabilities[LabelClass.skip.rawValue],
            fillRisk: 0.0,
            confidence: max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]),
            reliability: fxClamp(reliability, 0.0, 1.0)
        )
    }

    private static func normalized(_ probabilities: [Double]) -> [Double] {
        let sum = probabilities.reduce(0.0, +)
        guard sum > 0.0 else { return [0.08, 0.08, 0.84] }
        return probabilities.map { $0 / sum }
    }
}
