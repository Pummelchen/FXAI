import Foundation

public enum DistQuantileReference {
    public static func pinballLoss(prediction: Double, target: Double, quantile: Double) -> Double {
        let q = min(max(quantile, 0.0), 1.0)
        let error = target - prediction
        return max(q * error, (q - 1.0) * error)
    }

    public static func coverage(predictions: [Double], targets: [Double]) -> Double {
        let count = min(predictions.count, targets.count)
        guard count > 0 else { return 0.0 }
        let hits = zip(predictions.prefix(count), targets.prefix(count)).filter { prediction, target in target <= prediction }.count
        return Double(hits) / Double(count)
    }

    public static func monotonicProjection(_ quantiles: [Double]) -> [Double] {
        guard !quantiles.isEmpty else { return [] }
        var output = quantiles
        for index in 1..<output.count where output[index] < output[index - 1] {
            output[index] = output[index - 1]
        }
        return output
    }
}
