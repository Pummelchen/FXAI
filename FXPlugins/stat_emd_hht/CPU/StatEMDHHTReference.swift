import Foundation

public enum StatEMDHHTReference {
    public struct Decomposition: Equatable, Sendable {
        public let imfs: [[Double]]
        public let residual: [Double]
        public let siftCounts: [Int]
        public let instantaneousAmplitude: [[Double]]
        public let instantaneousFrequency: [[Double]]
    }

    public static func decompose(
        signal: [Double],
        maxIMFs: Int = 4,
        maxSiftings: Int = 24,
        tolerance: Double = 1.0e-4
    ) -> Decomposition {
        guard signal.count >= 4, maxIMFs > 0 else {
            return Decomposition(imfs: [], residual: signal, siftCounts: [], instantaneousAmplitude: [], instantaneousFrequency: [])
        }

        var residual = signal
        var imfs: [[Double]] = []
        var siftCounts: [Int] = []
        for _ in 0..<maxIMFs {
            let extrema = localExtrema(residual)
            guard extrema.maxima.count + extrema.minima.count >= 3 else { break }
            let extracted = extractIMF(from: residual, maxSiftings: maxSiftings, tolerance: tolerance)
            guard energy(extracted.imf) > 1.0e-12 else { break }
            imfs.append(extracted.imf)
            siftCounts.append(extracted.siftCount)
            residual = zip(residual, extracted.imf).map { $0 - $1 }
            let remainingExtrema = localExtrema(residual)
            if remainingExtrema.maxima.count + remainingExtrema.minima.count < 3 {
                break
            }
        }
        let hilbert = imfs.map { analyticSummary($0) }
        return Decomposition(
            imfs: imfs,
            residual: residual,
            siftCounts: siftCounts,
            instantaneousAmplitude: hilbert.map(\.amplitude),
            instantaneousFrequency: hilbert.map(\.frequency)
        )
    }

    private static func extractIMF(from values: [Double], maxSiftings: Int, tolerance: Double) -> (imf: [Double], siftCount: Int) {
        var candidate = values
        var count = 0
        for iteration in 0..<maxSiftings {
            let extrema = localExtrema(candidate)
            guard extrema.maxima.count + extrema.minima.count >= 3 else { break }
            let upper = envelope(length: candidate.count, anchors: extrema.maxima.map { ($0, candidate[$0]) })
            let lower = envelope(length: candidate.count, anchors: extrema.minima.map { ($0, candidate[$0]) })
            let meanEnvelope = zip(upper, lower).map { 0.5 * ($0 + $1) }
            let next = zip(candidate, meanEnvelope).map { $0 - $1 }
            count = iteration + 1
            let ratio = energy(meanEnvelope) / max(energy(candidate), 1.0e-12)
            candidate = next
            let zeroCrossingDifference = abs(zeroCrossings(candidate) - (extrema.maxima.count + extrema.minima.count))
            if ratio < tolerance, zeroCrossingDifference <= 1 {
                break
            }
        }
        return (candidate, count)
    }

    private static func localExtrema(_ values: [Double]) -> (maxima: [Int], minima: [Int]) {
        guard values.count >= 3 else { return ([], []) }
        var maxima = [0]
        var minima = [0]
        for index in 1..<(values.count - 1) {
            let left = values[index] - values[index - 1]
            let right = values[index + 1] - values[index]
            if left >= 0.0, right < 0.0 {
                maxima.append(index)
            }
            if left <= 0.0, right > 0.0 {
                minima.append(index)
            }
        }
        maxima.append(values.count - 1)
        minima.append(values.count - 1)
        return (maxima, minima)
    }

    private static func envelope(length: Int, anchors: [(Int, Double)]) -> [Double] {
        guard length > 0 else { return [] }
        let sorted = anchors.sorted { $0.0 < $1.0 }
        guard !sorted.isEmpty else { return Array(repeating: 0.0, count: length) }
        var output = Array(repeating: sorted[0].1, count: length)
        for pairIndex in 0..<(sorted.count - 1) {
            let left = sorted[pairIndex]
            let right = sorted[pairIndex + 1]
            let span = max(right.0 - left.0, 1)
            for index in max(left.0, 0)...min(right.0, length - 1) {
                let t = Double(index - left.0) / Double(span)
                output[index] = left.1 + t * (right.1 - left.1)
            }
        }
        if let last = sorted.last, last.0 < length {
            for index in last.0..<length {
                output[index] = last.1
            }
        }
        return output
    }

    private static func analyticSummary(_ values: [Double]) -> (amplitude: [Double], frequency: [Double]) {
        let spectrum = dft(values.map { Complex(real: $0, imag: 0.0) }, inverse: false)
        let n = spectrum.count
        guard n > 0 else { return ([], []) }
        var analyticSpectrum = spectrum
        for index in 0..<n {
            if index == 0 || (n.isMultiple(of: 2) && index == n / 2) {
                continue
            } else if index < (n + 1) / 2 {
                analyticSpectrum[index] = analyticSpectrum[index] * 2.0
            } else {
                analyticSpectrum[index] = .zero
            }
        }
        let analytic = dft(analyticSpectrum, inverse: true)
        let amplitude = analytic.map { sqrt($0.real * $0.real + $0.imag * $0.imag) }
        let phase = unwrap(analytic.map { atan2($0.imag, $0.real) })
        var frequency = Array(repeating: 0.0, count: n)
        if n > 1 {
            for index in 1..<n {
                frequency[index] = max((phase[index] - phase[index - 1]) / (2.0 * Double.pi), 0.0)
            }
            frequency[0] = frequency[1]
        }
        return (amplitude, frequency)
    }

    private static func dft(_ input: [Complex], inverse: Bool) -> [Complex] {
        let n = input.count
        guard n > 0 else { return [] }
        let sign = inverse ? 1.0 : -1.0
        let scale = inverse ? 1.0 / Double(n) : 1.0
        return (0..<n).map { k in
            var sum = Complex.zero
            for t in 0..<n {
                let angle = sign * 2.0 * Double.pi * Double(k * t) / Double(n)
                sum = sum + input[t] * Complex(real: cos(angle), imag: sin(angle))
            }
            return sum * scale
        }
    }

    private static func unwrap(_ phase: [Double]) -> [Double] {
        guard !phase.isEmpty else { return [] }
        var output = phase
        var offset = 0.0
        for index in 1..<phase.count {
            let delta = phase[index] - phase[index - 1]
            if delta > Double.pi {
                offset -= 2.0 * Double.pi
            } else if delta < -Double.pi {
                offset += 2.0 * Double.pi
            }
            output[index] += offset
        }
        return output
    }

    private static func zeroCrossings(_ values: [Double]) -> Int {
        guard values.count > 1 else { return 0 }
        var count = 0
        for index in 1..<values.count where values[index - 1] * values[index] < 0.0 {
            count += 1
        }
        return count
    }

    private static func energy(_ values: [Double]) -> Double {
        values.map { $0 * $0 }.reduce(0.0, +)
    }

    private struct Complex: Equatable {
        var real: Double
        var imag: Double

        static let zero = Complex(real: 0.0, imag: 0.0)

        static func + (lhs: Complex, rhs: Complex) -> Complex {
            Complex(real: lhs.real + rhs.real, imag: lhs.imag + rhs.imag)
        }

        static func * (lhs: Complex, rhs: Complex) -> Complex {
            Complex(real: lhs.real * rhs.real - lhs.imag * rhs.imag, imag: lhs.real * rhs.imag + lhs.imag * rhs.real)
        }

        static func * (lhs: Complex, rhs: Double) -> Complex {
            Complex(real: lhs.real * rhs, imag: lhs.imag * rhs)
        }
    }
}
