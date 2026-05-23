import Foundation

public enum StatTVPKalmanReference {
    public struct Model: Equatable, Sendable {
        public var transition: [[Double]]
        public var observation: [Double]
        public var processCovariance: [[Double]]
        public var observationVariance: Double
        public var state: [Double]
        public var covariance: [[Double]]

        public init(
            transition: [[Double]],
            observation: [Double],
            processCovariance: [[Double]],
            observationVariance: Double,
            state: [Double],
            covariance: [[Double]]
        ) {
            self.transition = transition
            self.observation = observation
            self.processCovariance = processCovariance
            self.observationVariance = observationVariance
            self.state = state
            self.covariance = covariance
        }
    }

    public struct FilterStep: Equatable, Sendable {
        public let predictedState: [Double]
        public let filteredState: [Double]
        public let innovation: Double?
        public let innovationVariance: Double?
    }

    public static func filter(observations: [Double?], model initialModel: Model) -> [FilterStep] {
        var model = initialModel
        var output: [FilterStep] = []
        output.reserveCapacity(observations.count)
        for observationValue in observations {
            let predictedState = matVec(model.transition, model.state)
            let predictedCovariance = add(matMul(matMul(model.transition, model.covariance), transpose(model.transition)), model.processCovariance)
            guard let y = observationValue, y.isFinite else {
                model.state = predictedState
                model.covariance = predictedCovariance
                output.append(FilterStep(predictedState: predictedState, filteredState: predictedState, innovation: nil, innovationVariance: nil))
                continue
            }
            let hx = dot(model.observation, predictedState)
            let innovation = y - hx
            let ph = matVec(predictedCovariance, model.observation)
            let innovationVariance = max(dot(model.observation, ph) + model.observationVariance, 1.0e-12)
            let gain = ph.map { $0 / innovationVariance }
            model.state = zip(predictedState, gain).map { $0 + $1 * innovation }
            let kh = outer(gain, model.observation)
            let identityMinusKH = subtract(identity(model.state.count), kh)
            model.covariance = matMul(identityMinusKH, predictedCovariance)
            output.append(FilterStep(predictedState: predictedState, filteredState: model.state, innovation: innovation, innovationVariance: innovationVariance))
        }
        return output
    }

    private static func identity(_ n: Int) -> [[Double]] {
        (0..<n).map { row in (0..<n).map { row == $0 ? 1.0 : 0.0 } }
    }

    private static func transpose(_ matrix: [[Double]]) -> [[Double]] {
        guard let width = matrix.first?.count else { return [] }
        return (0..<width).map { column in matrix.map { $0[column] } }
    }

    private static func matVec(_ matrix: [[Double]], _ vector: [Double]) -> [Double] {
        matrix.map { dot($0, vector) }
    }

    private static func matMul(_ lhs: [[Double]], _ rhs: [[Double]]) -> [[Double]] {
        let rhsT = transpose(rhs)
        return lhs.map { row in rhsT.map { dot(row, $0) } }
    }

    private static func add(_ lhs: [[Double]], _ rhs: [[Double]]) -> [[Double]] {
        zip(lhs, rhs).map { zip($0, $1).map(+) }
    }

    private static func subtract(_ lhs: [[Double]], _ rhs: [[Double]]) -> [[Double]] {
        zip(lhs, rhs).map { zip($0, $1).map(-) }
    }

    private static func outer(_ lhs: [Double], _ rhs: [Double]) -> [[Double]] {
        lhs.map { l in rhs.map { l * $0 } }
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).map(*).reduce(0.0, +)
    }
}
