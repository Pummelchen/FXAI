import Foundation

public enum StatVMDReference {
    public struct Decomposition: Equatable, Sendable {
        public let modes: [[Double]]
        public let centerFrequencies: [Double]
        public let reconstruction: [Double]
        public let residual: [Double]
        public let residualNorms: [Double]
    }

    public static func decompose(
        signal: [Double],
        modeCount: Int,
        alpha: Double = 800.0,
        iterations: Int = 48
    ) -> Decomposition {
        guard signal.count >= 4, modeCount > 0 else {
            return Decomposition(modes: [], centerFrequencies: [], reconstruction: [], residual: signal, residualNorms: [])
        }

        let demeanedMean = mean(signal)
        let centered = signal.map { $0 - demeanedMean }
        let spectrum = dft(centered.map { Complex(real: $0, imag: 0.0) }, inverse: false)
        let n = spectrum.count
        let frequencies = (0..<n).map { index -> Double in
            if index <= n / 2 {
                return Double(index) / Double(n)
            }
            return -Double(n - index) / Double(n)
        }
        var centers = (0..<modeCount).map { Double($0 + 1) / Double(2 * modeCount + 1) }
        var modeSpectra = Array(repeating: Array(repeating: Complex.zero, count: n), count: modeCount)
        var residualNorms: [Double] = []

        for _ in 0..<max(iterations, 1) {
            for mode in 0..<modeCount {
                let otherSum = sumSpectra(modeSpectra, excluding: mode)
                for bin in 0..<n {
                    let residual = spectrum[bin] - otherSum[bin]
                    let bandwidth = 1.0 + alpha * pow(abs(frequencies[bin]) - centers[mode], 2.0)
                    modeSpectra[mode][bin] = residual / bandwidth
                }
                centers[mode] = spectralCenter(modeSpectra[mode], frequencies: frequencies, previous: centers[mode])
            }
            let reconstructedSpectrum = sumSpectra(modeSpectra, excluding: nil)
            let spectralResidual = zip(spectrum, reconstructedSpectrum).map { $0 - $1 }
            residualNorms.append(sqrt(spectralResidual.map { $0.magnitudeSquared }.reduce(0.0, +) / Double(max(n, 1))))
        }

        let modes = modeSpectra.map { spec in
            dft(spec, inverse: true).map(\.real)
        }
        let reconstruction = (0..<signal.count).map { index in
            demeanedMean + modes.map { $0[index] }.reduce(0.0, +)
        }
        let residual = zip(signal, reconstruction).map { $0 - $1 }
        return Decomposition(
            modes: modes,
            centerFrequencies: centers,
            reconstruction: reconstruction,
            residual: residual,
            residualNorms: residualNorms
        )
    }

    private static func sumSpectra(_ spectra: [[Complex]], excluding excluded: Int?) -> [Complex] {
        guard let width = spectra.first?.count else { return [] }
        var output = Array(repeating: Complex.zero, count: width)
        for (index, spectrum) in spectra.enumerated() where excluded == nil || excluded != index {
            for bin in 0..<width {
                output[bin] = output[bin] + spectrum[bin]
            }
        }
        return output
    }

    private static func spectralCenter(_ spectrum: [Complex], frequencies: [Double], previous: Double) -> Double {
        var numerator = 0.0
        var denominator = 0.0
        for (value, frequency) in zip(spectrum, frequencies) where frequency > 0.0 {
            let power = value.magnitudeSquared
            numerator += abs(frequency) * power
            denominator += power
        }
        guard denominator > 1.0e-12 else { return previous }
        return numerator / denominator
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

    private static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0.0 : values.reduce(0.0, +) / Double(values.count)
    }

    private struct Complex: Equatable {
        var real: Double
        var imag: Double

        var magnitudeSquared: Double { real * real + imag * imag }

        static let zero = Complex(real: 0.0, imag: 0.0)

        static func + (lhs: Complex, rhs: Complex) -> Complex {
            Complex(real: lhs.real + rhs.real, imag: lhs.imag + rhs.imag)
        }

        static func - (lhs: Complex, rhs: Complex) -> Complex {
            Complex(real: lhs.real - rhs.real, imag: lhs.imag - rhs.imag)
        }

        static func * (lhs: Complex, rhs: Complex) -> Complex {
            Complex(real: lhs.real * rhs.real - lhs.imag * rhs.imag, imag: lhs.real * rhs.imag + lhs.imag * rhs.real)
        }

        static func * (lhs: Complex, rhs: Double) -> Complex {
            Complex(real: lhs.real * rhs, imag: lhs.imag * rhs)
        }

        static func / (lhs: Complex, rhs: Double) -> Complex {
            Complex(real: lhs.real / rhs, imag: lhs.imag / rhs)
        }
    }
}
