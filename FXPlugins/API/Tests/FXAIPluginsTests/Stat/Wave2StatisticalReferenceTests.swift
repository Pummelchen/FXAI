import FXAIPlugins
import XCTest

final class Wave2StatisticalReferenceTests: XCTestCase {
    func testARIMAXAndGARCHRecoverSyntheticStructure() {
        var series = [0.4, 0.55]
        var exogenous = Array(repeating: [0.0], count: 80)
        for index in 2..<80 {
            exogenous[index][0] = sin(Double(index) * 0.17)
            let noise = 0.015 * sin(Double(index) * 0.61)
            series.append(0.20 + 0.72 * series[index - 1] + 0.31 * exogenous[index][0] + noise)
        }

        let arimax = StatARIMAXGARCHReference.fitARIMAX(series: series, exogenous: exogenous, lagOrder: 1)
        let garch = StatARIMAXGARCHReference.estimateGARCH(residuals: arimax.residuals)

        XCTAssertEqual(arimax.coefficients[1], 0.72, accuracy: 0.08)
        XCTAssertEqual(arimax.coefficients[2], 0.31, accuracy: 0.08)
        XCTAssertEqual(garch.variances.count, arimax.residuals.count)
        XCTAssertTrue(garch.variances.allSatisfy { $0 > 0.0 && $0.isFinite })
        XCTAssertTrue(garch.logLikelihood.isFinite)
    }

    func testOUReferenceEstimatesMeanReversionAndHalfLife() {
        let spread = (0..<80).map { index in
            1.25 + 0.80 * pow(0.91, Double(index)) + 0.015 * sin(Double(index) * 0.41)
        }

        let estimate = StatOUSpreadReference.estimate(spread: spread)
        let zScore = StatOUSpreadReference.zScore(value: spread.last ?? 0.0, estimate: estimate)

        XCTAssertGreaterThan(estimate.theta, 0.0)
        XCTAssertGreaterThan(estimate.halfLife, 0.0)
        XCTAssertTrue(estimate.halfLife.isFinite)
        XCTAssertLessThan(abs(zScore), 3.0)
    }

    func testKalmanReferenceHandlesPredictUpdateAndMissingObservations() {
        let model = StatTVPKalmanReference.Model(
            transition: [[1.0]],
            observation: [1.0],
            processCovariance: [[0.02]],
            observationVariance: 0.10,
            state: [0.0],
            covariance: [[1.0]]
        )
        let steps = StatTVPKalmanReference.filter(observations: [0.2, 0.4, nil, 0.8, 1.0], model: model)

        XCTAssertEqual(steps.count, 5)
        XCTAssertNil(steps[2].innovation)
        XCTAssertEqual(steps[2].predictedState, steps[2].filteredState)
        XCTAssertGreaterThan(steps.last?.filteredState.first ?? 0.0, steps.first?.filteredState.first ?? 0.0)
        XCTAssertTrue(steps.compactMap(\.innovationVariance).allSatisfy { $0 > 0.0 && $0.isFinite })
    }

    func testHMMReferenceRunsLogSpaceEMAndViterbi() {
        let observations = [-1.20, -1.08, -0.96, -1.15, -0.88, 1.05, 1.20, 0.95, 1.12, 0.90, -1.05, -0.92]
        let fit = StatHMMRegimeReference.baumWelch(observations: observations, regimes: 2, iterations: 8)
        let path = StatHMMRegimeReference.viterbi(observations: observations, fit: fit)

        XCTAssertEqual(fit.transition.count, 2)
        XCTAssertEqual(fit.posteriors.count, observations.count)
        XCTAssertGreaterThanOrEqual(fit.logLikelihoods.last ?? -Double.infinity, (fit.logLikelihoods.first ?? -Double.infinity) - 1.0e-6)
        XCTAssertEqual(path.count, observations.count)
        XCTAssertGreaterThan(Set(path).count, 1)
        XCTAssertTrue(fit.posteriors.allSatisfy { abs($0.reduce(0.0, +) - 1.0) < 1.0e-8 })
    }

    func testMSGARCHReferenceFiltersVolatilityRegimes() {
        let quiet = (0..<40).map { 0.008 * sin(Double($0) * 0.43) }
        let volatile = (0..<40).map { 0.090 * sin(Double($0) * 0.71) }
        let result = StatMSGARCHReference.filter(
            returns: quiet + volatile,
            transition: [[0.94, 0.06], [0.08, 0.92]],
            omega: [0.00002, 0.0012],
            alpha: [0.05, 0.18],
            beta: [0.90, 0.74]
        )
        let firstHalfHigh = result.probabilities.prefix(40).map { $0[1] }.reduce(0.0, +) / 40.0
        let secondHalfHigh = result.probabilities.suffix(40).map { $0[1] }.reduce(0.0, +) / 40.0

        XCTAssertEqual(result.probabilities.count, 80)
        XCTAssertTrue(result.probabilities.allSatisfy { abs($0.reduce(0.0, +) - 1.0) < 1.0e-8 })
        XCTAssertGreaterThan(secondHalfHigh, firstHalfHigh)
        XCTAssertTrue(result.variances.flatMap { $0 }.allSatisfy { $0 > 0.0 && $0.isFinite })
        XCTAssertTrue(result.logLikelihood.isFinite)
    }

    func testVECMReferenceRecoversCointegratingVector() {
        var x = [1.0]
        var residual = [0.15]
        var y = [1.0 + 2.0 * x[0] + residual[0]]
        for index in 1..<96 {
            x.append(x[index - 1] + 0.04 * sin(Double(index) * 0.31))
            residual.append(0.35 * residual[index - 1] + 0.01 * sin(Double(index) * 0.79))
            y.append(1.0 + 2.0 * x[index] + residual[index])
        }

        let fit = StatCointVECMReference.estimatePair(y: y, x: x, lagOrder: 1)
        let forecast = StatCointVECMReference.forecastNext(yLast: y.last ?? 0.0, xLast: x.last ?? 0.0, fit: fit)
        let zScores = StatCointVECMReference.spreadZScores(fit: fit)

        XCTAssertEqual(fit.beta, 2.0, accuracy: 0.12)
        XCTAssertLessThan(fit.residualVariance, 0.02)
        XCTAssertLessThan(fit.adfTStatistic, -2.0)
        XCTAssertTrue(forecast.residual.isFinite)
        XCTAssertEqual(zScores.count, y.count)
    }

    func testEMDHHTReferenceExtractsIMFsAndHilbertFrequency() {
        let signal = (0..<64).map { index in
            sin(2.0 * Double.pi * Double(index) / 16.0) +
                0.35 * sin(2.0 * Double.pi * Double(index) / 5.0)
        }
        let result = StatEMDHHTReference.decompose(signal: signal, maxIMFs: 3)

        XCTAssertGreaterThan(result.imfs.count, 0)
        XCTAssertEqual(result.residual.count, signal.count)
        XCTAssertEqual(result.instantaneousAmplitude.count, result.imfs.count)
        XCTAssertEqual(result.instantaneousFrequency.count, result.imfs.count)
        XCTAssertTrue(result.instantaneousFrequency.flatMap { $0 }.allSatisfy { $0.isFinite && $0 >= 0.0 })
    }

    func testVMDReferenceSeparatesModesAndReconstructsSignal() {
        let signal = (0..<48).map { index in
            0.7 * sin(2.0 * Double.pi * Double(index) / 12.0) +
                0.3 * sin(2.0 * Double.pi * Double(index) / 4.0)
        }
        let result = StatVMDReference.decompose(signal: signal, modeCount: 2, alpha: 250.0, iterations: 18)
        let reconstructionError = mse(signal, result.reconstruction)
        let signalPower = signal.map { $0 * $0 }.reduce(0.0, +) / Double(signal.count)

        XCTAssertEqual(result.modes.count, 2)
        XCTAssertEqual(result.centerFrequencies.count, 2)
        XCTAssertEqual(result.reconstruction.count, signal.count)
        XCTAssertLessThan(reconstructionError, signalPower)
        XCTAssertTrue(result.centerFrequencies.allSatisfy { $0.isFinite && $0 > 0.0 })
        XCTAssertTrue(result.residualNorms.allSatisfy(\.isFinite))
    }

    func testXRateReferenceScoresTriangularConsistency() {
        let consistent = [
            StatXrateConsistencyReference.Quote(base: "EUR", quote: "USD", rate: 1.10),
            StatXrateConsistencyReference.Quote(base: "USD", quote: "JPY", rate: 150.0),
            StatXrateConsistencyReference.Quote(base: "EUR", quote: "JPY", rate: 165.0)
        ]
        let inconsistent = [
            StatXrateConsistencyReference.Quote(base: "EUR", quote: "USD", rate: 1.10),
            StatXrateConsistencyReference.Quote(base: "USD", quote: "JPY", rate: 150.0),
            StatXrateConsistencyReference.Quote(base: "EUR", quote: "JPY", rate: 164.0)
        ]

        let consistentScore = StatXrateConsistencyReference.scoreCycle(["EUR", "USD", "JPY"], quotes: consistent)
        let inconsistentScores = StatXrateConsistencyReference.triangularScores(quotes: inconsistent)

        XCTAssertNotNil(consistentScore)
        XCTAssertEqual(consistentScore?.basisPoints ?? 1.0, 0.0, accuracy: 1.0e-9)
        XCTAssertGreaterThan(abs(inconsistentScores.first?.basisPoints ?? 0.0), 50.0)
    }

    private func mse(_ lhs: [Double], _ rhs: [Double]) -> Double {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0.0 }
        return zip(lhs.prefix(count), rhs.prefix(count)).map { pow($0 - $1, 2.0) }.reduce(0.0, +) / Double(count)
    }
}
