import XCTest
@testable import FXDataEngine

final class TrainingHyperParametersTests: XCTestCase {
    func testRandomGeneratorMatchesLegacyLCGAndRangeRules() {
        var rng = TrainingRandomGenerator(seed: 0)
        var expectedState: UInt64 = 1
        expectedState = expectedState &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let expectedUnit = Double(expectedState >> 11) / 9_007_199_254_740_991.0
        XCTAssertEqual(rng.nextUnit(), expectedUnit, accuracy: 1e-15)
        XCTAssertEqual(rng.state, expectedState)

        let stateBeforeClosedRange = rng.state
        XCTAssertEqual(rng.range(5.0, 5.0), 5.0, accuracy: 0.0)
        XCTAssertEqual(rng.state, stateBeforeClosedRange)
    }

    func testBaseParametersClampLegacyInputRanges() {
        let parameters = AIHyperParameterTools.baseParameters(inputs: AIHyperParameterInputs(
            learningRate: -1.0,
            l2: 9.0,
            ftrlAlpha: 0.0,
            ftrlBeta: 99.0,
            ftrlL1: -2.0,
            ftrlL2: 99.0,
            paC: 0.0,
            paMargin: 9.0,
            xgbLearningRate: 9.0,
            xgbL2: -1.0,
            xgbSplit: 4.0,
            mlpLearningRate: 0.0,
            mlpL2: 9.0,
            mlpInit: 9.0,
            tcnLayers: 99,
            tcnKernel: -9,
            tcnDilationBase: 99,
            quantileLearningRate: 0.0,
            quantileL2: 9.0,
            enhashLearningRate: 0.0,
            enhashL1: 9.0,
            enhashL2: 9.0
        ))

        XCTAssertEqual(parameters.learningRate, 0.001, accuracy: 0.0)
        XCTAssertEqual(parameters.l2, 0.100, accuracy: 0.0)
        XCTAssertEqual(parameters.ftrlAlpha, 0.001, accuracy: 0.0)
        XCTAssertEqual(parameters.ftrlBeta, 5.000, accuracy: 0.0)
        XCTAssertEqual(parameters.paC, 0.010, accuracy: 0.0)
        XCTAssertEqual(parameters.paMargin, 2.000, accuracy: 0.0)
        XCTAssertEqual(parameters.xgbLearningRate, 0.300, accuracy: 0.0)
        XCTAssertEqual(parameters.xgbL2, 0.000, accuracy: 0.0)
        XCTAssertEqual(parameters.xgbSplit, 2.000, accuracy: 0.0)
        XCTAssertEqual(parameters.mlpLearningRate, 0.0005, accuracy: 0.0)
        XCTAssertEqual(parameters.mlpL2, 0.0500, accuracy: 0.0)
        XCTAssertEqual(parameters.mlpInit, 0.5000, accuracy: 0.0)
        XCTAssertEqual(parameters.tcnLayers, 8.0, accuracy: 0.0)
        XCTAssertEqual(parameters.tcnKernel, 2.0, accuracy: 0.0)
        XCTAssertEqual(parameters.tcnDilationBase, 3.0, accuracy: 0.0)
        XCTAssertEqual(parameters.quantileLearningRate, 0.0001, accuracy: 0.0)
        XCTAssertEqual(parameters.quantileL2, 0.1000, accuracy: 0.0)
        XCTAssertEqual(parameters.enhashLearningRate, 0.0005, accuracy: 0.0)
        XCTAssertEqual(parameters.enhashL1, 0.1000, accuracy: 0.0)
        XCTAssertEqual(parameters.enhashL2, 0.1000, accuracy: 0.0)
    }

    func testDefaultParametersApplyModelSpecificStartingValues() {
        let lstm = AIHyperParameterTools.defaultParameters(aiID: AIModelID.lstm.rawValue)
        XCTAssertEqual(lstm.learningRate, 0.0080, accuracy: 0.0)
        XCTAssertEqual(lstm.l2, 0.0040, accuracy: 0.0)

        let lightgbm = AIHyperParameterTools.defaultParameters(aiID: AIModelID.lightgbm.rawValue)
        XCTAssertEqual(lightgbm.xgbLearningRate, 0.0300, accuracy: 0.0)
        XCTAssertEqual(lightgbm.xgbL2, 4.0000, accuracy: 0.0)
        XCTAssertEqual(lightgbm.xgbSplit, 0.0000, accuracy: 0.0)

        let m1sync = AIHyperParameterTools.defaultParameters(aiID: AIModelID.m1Sync.rawValue)
        XCTAssertEqual(m1sync.learningRate, 0.0, accuracy: 0.0)
        XCTAssertEqual(m1sync.l2, 0.0, accuracy: 0.0)
    }

    func testThresholdSamplingUsesSanitizedRangesAndLCGOrder() {
        var rng = TrainingRandomGenerator(seed: 7)
        var expectedRNG = rng
        let first = expectedRNG.nextUnit()
        let second = expectedRNG.nextUnit()
        let expectedBuy = 0.62 + (0.78 - 0.62) * first
        let expectedSell = 0.22 + (0.38 - 0.22) * second

        let sampled = AIHyperParameterTools.sampleThresholdPair(baseBuy: 0.70, baseSell: 0.30, rng: &rng)
        XCTAssertEqual(sampled.buy, expectedBuy, accuracy: 1e-12)
        XCTAssertEqual(sampled.sell, expectedSell, accuracy: 1e-12)
        XCTAssertEqual(rng.state, expectedRNG.state)
    }

    func testSampleParametersUseModelSpecificRangesAndLeaveRuleBaselinesUntouched() {
        let base = AIHyperParameterTools.baseParameters()
        var baselineRNG = TrainingRandomGenerator(seed: 99)
        let buyOnly = AIHyperParameterTools.sampleParameters(aiID: AIModelID.buyOnly.rawValue, base: base, rng: &baselineRNG)
        XCTAssertEqual(buyOnly, base)
        XCTAssertEqual(baselineRNG.state, 99)

        var tcnRNG = TrainingRandomGenerator(seed: 99)
        let tcn = AIHyperParameterTools.sampleParameters(aiID: AIModelID.tcn.rawValue, base: base, rng: &tcnRNG)
        XCTAssertGreaterThanOrEqual(tcn.learningRate, 0.0030)
        XCTAssertLessThanOrEqual(tcn.learningRate, 0.0500)
        XCTAssertGreaterThanOrEqual(tcn.l2, 0.0000)
        XCTAssertLessThanOrEqual(tcn.l2, 0.0200)
        XCTAssertGreaterThanOrEqual(tcn.tcnLayers, 3.0)
        XCTAssertLessThanOrEqual(tcn.tcnLayers, 6.0)
        XCTAssertEqual(tcn.tcnLayers.rounded(), tcn.tcnLayers)
        XCTAssertNotEqual(tcnRNG.state, 99)

        var treeRNG = TrainingRandomGenerator(seed: 123)
        let catboost = AIHyperParameterTools.sampleParameters(aiID: AIModelID.catboost.rawValue, base: base, rng: &treeRNG)
        XCTAssertGreaterThanOrEqual(catboost.xgbLearningRate, 0.0200)
        XCTAssertLessThanOrEqual(catboost.xgbLearningRate, 0.0500)
        XCTAssertGreaterThanOrEqual(catboost.xgbL2, 3.0000)
        XCTAssertLessThanOrEqual(catboost.xgbL2, 8.0000)
        XCTAssertGreaterThanOrEqual(catboost.xgbSplit, -0.2000)
        XCTAssertLessThanOrEqual(catboost.xgbSplit, 0.2000)
    }
}
