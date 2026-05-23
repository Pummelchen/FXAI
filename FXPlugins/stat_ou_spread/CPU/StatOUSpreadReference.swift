import Foundation

public enum StatOUSpreadReference {
    public struct OUEstimate: Equatable, Sendable {
        public let mean: Double
        public let theta: Double
        public let sigma: Double
        public let halfLife: Double
        public let arIntercept: Double
        public let arSlope: Double
        public let residualVariance: Double
        public let isStationary: Bool
    }

    public static func estimate(spread: [Double], deltaTime: Double = 1.0) -> OUEstimate {
        guard spread.count > 2, deltaTime > 0 else {
            let level = mean(spread)
            return OUEstimate(mean: level, theta: 0.0, sigma: 0.0, halfLife: .infinity, arIntercept: level, arSlope: 1.0, residualVariance: 0.0, isStationary: false)
        }
        let lagged = Array(spread.dropLast())
        let current = Array(spread.dropFirst())
        let xMean = mean(lagged)
        let yMean = mean(current)
        let denominator = lagged.map { ($0 - xMean) * ($0 - xMean) }.reduce(0.0, +)
        let slope = denominator <= 1.0e-12 ? 1.0 : zip(lagged, current).map { ($0 - xMean) * ($1 - yMean) }.reduce(0.0, +) / denominator
        let intercept = yMean - slope * xMean
        let clampedSlope = min(max(slope, 1.0e-8), 0.999999)
        let theta = max(-log(clampedSlope) / deltaTime, 0.0)
        let meanReversionLevel = abs(1.0 - slope) <= 1.0e-12 ? yMean : intercept / (1.0 - slope)
        let residuals = zip(lagged, current).map { previous, value in value - (intercept + slope * previous) }
        let residualVariance = variance(residuals)
        let sigma = theta <= 1.0e-12
            ? sqrt(max(residualVariance, 0.0))
            : sqrt(max(residualVariance * 2.0 * theta / max(1.0 - slope * slope, 1.0e-12), 0.0))
        let halfLife = theta <= 1.0e-12 ? Double.infinity : log(2.0) / theta
        return OUEstimate(
            mean: meanReversionLevel,
            theta: theta,
            sigma: sigma,
            halfLife: halfLife,
            arIntercept: intercept,
            arSlope: slope,
            residualVariance: residualVariance,
            isStationary: abs(slope) < 1.0
        )
    }

    public static func zScore(value: Double, estimate: OUEstimate) -> Double {
        let scale = max(estimate.sigma / sqrt(max(2.0 * estimate.theta, 1.0e-12)), sqrt(max(estimate.residualVariance, 1.0e-12)))
        return (value - estimate.mean) / max(scale, 1.0e-12)
    }

    private static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0.0 : values.reduce(0.0, +) / Double(values.count)
    }

    private static func variance(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0.0 }
        let m = mean(values)
        return values.map { ($0 - m) * ($0 - m) }.reduce(0.0, +) / Double(values.count - 1)
    }
}
