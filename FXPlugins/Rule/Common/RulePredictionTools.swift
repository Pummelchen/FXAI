import FXDataEngine
import Foundation

enum RulePredictionTools {
    static func fixedDirectionalPrediction(
        request: PredictRequestV4,
        buyProbability: Double,
        sellProbability: Double,
        confidence: Double,
        sigmaScale: Double = 0.30,
        quantileScale: Double = 0.50,
        reliability: Double = 0.55
    ) -> PredictionV4 {
        let minimumMove = request.context.minMovePoints > 0.0
            ? request.context.minMovePoints
            : max(request.context.priceCostPoints, 0.10)
        let meanMove = max(1.0, 3.0 * minimumMove + 0.25)
        let sigma = max(0.10, sigmaScale * meanMove)
        let q25 = max(0.0, meanMove - quantileScale * sigma)
        let q50 = max(q25, meanMove)
        let q75 = max(q50, meanMove + quantileScale * sigma)

        return PredictionV4(
            classProbabilities: [sellProbability, buyProbability, 0.0],
            moveMeanPoints: meanMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: meanMove,
            maeMeanPoints: max(0.0, 0.35 * meanMove),
            hitTimeFraction: 1.0,
            pathRisk: 0.0,
            fillRisk: 0.0,
            confidence: confidence,
            reliability: reliability
        )
    }

    static func deterministicRandomBuySide(request: PredictRequestV4) -> Bool {
        var timestamp = request.context.sampleTimeUTC
        if timestamp < 0 {
            timestamp = -timestamp
        }
        var accumulator = Double(timestamp)
        let limit = min(request.x.count, 8)
        if limit > 0 {
            for index in 0..<limit {
                accumulator = accumulator * 1.618_033_988_75 + abs(request.x[index]) * 1_000.0 + 17.0 * Double(index + 1)
                if accumulator > 2_147_483_000.0 {
                    accumulator -= 2_147_483_000.0 * floor(accumulator / 2_147_483_000.0)
                }
            }
        }
        let hash = UInt64(abs(Int64(accumulator.rounded())))
        return hash % 2 == 0
    }
}
