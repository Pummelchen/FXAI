import Foundation

public struct PluginTernaryCalibrator: Codable, Hashable, Sendable {
    public static let classCount = 3
    public static let binCount = 12

    public private(set) var steps: Int
    public var weights: [[Double]]
    public var biases: [Double]
    public var isotonicPositive: [[Double]]
    public var isotonicCount: [[Double]]

    public init(
        steps: Int = 0,
        weights: [[Double]] = PluginTernaryCalibrator.identityWeights(),
        biases: [Double] = Array(repeating: 0.0, count: PluginTernaryCalibrator.classCount),
        isotonicPositive: [[Double]] = PluginTernaryCalibrator.zeroBins(),
        isotonicCount: [[Double]] = PluginTernaryCalibrator.zeroBins()
    ) {
        self.steps = max(0, steps)
        self.weights = Self.normalizedMatrix(weights, columns: Self.classCount, identityFallback: true)
        self.biases = Self.normalizedVector(biases, count: Self.classCount)
        self.isotonicPositive = Self.normalizedMatrix(isotonicPositive, columns: Self.binCount, identityFallback: false)
        self.isotonicCount = Self.normalizedMatrix(isotonicCount, columns: Self.binCount, identityFallback: false)
    }

    public mutating func reset() {
        self = PluginTernaryCalibrator()
    }

    public func calibrated(_ rawProbabilities: [Double]) -> [Double] {
        let logits = buildLogits(rawProbabilities)
        var calibrated = Self.softmax3(logits)
        guard steps >= 30 else {
            return calibrated
        }

        var isotonic = calibrated
        for classIndex in 0..<Self.classCount {
            let total = isotonicCount[classIndex].reduce(0.0, +)
            guard total >= 40.0 else {
                isotonic[classIndex] = calibrated[classIndex]
                continue
            }

            var monotonic = Array(repeating: 0.0, count: Self.binCount)
            var previous = 0.01
            for bin in 0..<Self.binCount {
                var ratio = previous
                if isotonicCount[classIndex][bin] > 1e-9 {
                    ratio = isotonicPositive[classIndex][bin] / isotonicCount[classIndex][bin]
                }
                ratio = fxClamp(ratio, 0.001, 0.999)
                if ratio < previous {
                    ratio = previous
                }
                monotonic[bin] = ratio
                previous = ratio
            }

            let bin = Self.probabilityBin(calibrated[classIndex])
            isotonic[classIndex] = monotonic[bin]
        }

        for classIndex in 0..<Self.classCount {
            calibrated[classIndex] = fxClamp(0.75 * calibrated[classIndex] + 0.25 * isotonic[classIndex], 0.0005, 0.9990)
        }
        let sum = calibrated.reduce(0.0, +)
        guard sum > 0.0 else {
            return [0.10, 0.10, 0.80]
        }
        return calibrated.map { $0 / sum }
    }

    public mutating func update(
        rawProbabilities: [Double],
        labelClass: LabelClass,
        sampleWeight: Double,
        learningRate: Double
    ) {
        let logits = buildLogits(rawProbabilities)
        let calibrated = Self.softmax3(logits)
        let logRaw = Self.logRawProbabilities(rawProbabilities)
        let weight = fxClamp(sampleWeight, 0.20, 8.00)
        let calibratorLearningRate = fxClamp(0.25 * learningRate * weight, 0.0002, 0.0200)
        let l2 = 0.0005

        for classIndex in 0..<Self.classCount {
            let target = classIndex == labelClass.rawValue ? 1.0 : 0.0
            let error = target - calibrated[classIndex]

            biases[classIndex] = Self.clipSymmetric(biases[classIndex] + calibratorLearningRate * error, limit: 4.0)
            for rawIndex in 0..<Self.classCount {
                let targetWeight = classIndex == rawIndex ? 1.0 : 0.0
                let gradient = error * logRaw[rawIndex] - l2 * (weights[classIndex][rawIndex] - targetWeight)
                weights[classIndex][rawIndex] = Self.clipSymmetric(
                    weights[classIndex][rawIndex] + calibratorLearningRate * gradient,
                    limit: 4.0
                )
            }

            let bin = Self.probabilityBin(calibrated[classIndex])
            isotonicCount[classIndex][bin] += weight
            isotonicPositive[classIndex][bin] += weight * target
        }
        steps += 1
    }

    private func buildLogits(_ rawProbabilities: [Double]) -> [Double] {
        let logRaw = Self.logRawProbabilities(rawProbabilities)
        var logits = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            var value = biases[classIndex]
            for rawIndex in 0..<Self.classCount {
                value += weights[classIndex][rawIndex] * logRaw[rawIndex]
            }
            logits[classIndex] = value
        }
        return logits
    }

    private static func logRawProbabilities(_ rawProbabilities: [Double]) -> [Double] {
        (0..<classCount).map { index in
            let raw = index < rawProbabilities.count ? rawProbabilities[index] : 0.0
            return log(fxClamp(fxSafeFinite(raw), 0.0005, 0.9990))
        }
    }

    private static func softmax3(_ logits: [Double]) -> [Double] {
        let safeLogits = (0..<classCount).map { index -> Double in
            index < logits.count ? fxSafeFinite(logits[index]) : 0.0
        }
        let maximum = safeLogits.max() ?? 0.0
        var exponentials = Array(repeating: 0.0, count: classCount)
        var denominator = 0.0
        for classIndex in 0..<classCount {
            let value = exp(clipSymmetric(safeLogits[classIndex] - maximum, limit: 30.0))
            exponentials[classIndex] = value
            denominator += value
        }
        guard denominator > 0.0 else {
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        }
        return exponentials.map { $0 / denominator }
    }

    private static func probabilityBin(_ probability: Double) -> Int {
        let raw = Int(floor(fxClamp(fxSafeFinite(probability), 0.0, 0.999999) * Double(binCount)))
        return min(max(0, raw), binCount - 1)
    }

    private static func clipSymmetric(_ value: Double, limit: Double) -> Double {
        fxClamp(value, -abs(limit), abs(limit))
    }

    public static func identityWeights() -> [[Double]] {
        (0..<classCount).map { row in
            (0..<classCount).map { column in row == column ? 1.0 : 0.0 }
        }
    }

    public static func zeroBins() -> [[Double]] {
        Array(repeating: Array(repeating: 0.0, count: binCount), count: classCount)
    }

    private static func normalizedVector(_ values: [Double], count: Int) -> [Double] {
        (0..<count).map { index in
            index < values.count ? fxSafeFinite(values[index]) : 0.0
        }
    }

    private static func normalizedMatrix(_ values: [[Double]], columns: Int, identityFallback: Bool) -> [[Double]] {
        (0..<classCount).map { row in
            (0..<columns).map { column in
                if row < values.count, column < values[row].count {
                    return fxSafeFinite(values[row][column])
                }
                return identityFallback && row == column ? 1.0 : 0.0
            }
        }
    }
}
