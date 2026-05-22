import Foundation
import XCTest
@testable import FXDataEngine

final class PluginContractSuiteTests: XCTestCase {
    func testContractSuiteDefaultHyperParametersMatchLegacyValues() {
        let hp = PluginContractSuiteTools.defaultHyperParameters()

        XCTAssertEqual(hp.learningRate, 0.01, accuracy: 0.0)
        XCTAssertEqual(hp.l2, 0.0001, accuracy: 0.0)
        XCTAssertEqual(hp.passiveAggressiveMargin, 1.0, accuracy: 0.0)
        XCTAssertEqual(hp.xgbL2, 0.0001, accuracy: 0.0)
        XCTAssertEqual(hp.xgbSplit, 0.5, accuracy: 0.0)
        XCTAssertEqual(hp.tcnLayers, 2.0, accuracy: 0.0)
        XCTAssertEqual(hp.tcnKernel, 3.0, accuracy: 0.0)
        XCTAssertEqual(hp.tcnDilationBase, 2.0, accuracy: 0.0)
    }

    func testContractSuiteBuildsDeterministicPredictRequestAndSeries() throws {
        let manifest = PluginManifestV4(
            aiID: 1,
            aiName: "Windowed",
            family: .transformer,
            capabilityMask: [.selfTest, .windowContext],
            minSequenceBars: 4,
            maxSequenceBars: 8
        )
        let request = PluginContractSuiteTools.buildPredictRequest(manifest: manifest)

        try request.validate()
        XCTAssertEqual(request.context.horizonMinutes, manifest.minHorizonMinutes)
        XCTAssertEqual(request.context.sequenceBars, 4)
        XCTAssertEqual(request.windowSize, 3)
        XCTAssertEqual(request.context.pointValue, 0.0001, accuracy: 0.0)
        XCTAssertEqual(request.context.domainHash, 0.5, accuracy: 0.0)
        XCTAssertTrue(request.context.dataHasVolume)
        XCTAssertEqual(request.x[0], 1.0, accuracy: 0.0)
        XCTAssertEqual(request.x[1], 0.02, accuracy: 0.0)
        XCTAssertEqual(request.xWindow[0][0], 1.0, accuracy: 0.0)
        XCTAssertEqual(request.xWindow[0][1], 0.01, accuracy: 0.0)
        XCTAssertEqual(request.xWindow[2][1], 0.03, accuracy: 0.0)

        let series = try PluginContractSuiteTools.buildSyntheticSeries()
        XCTAssertEqual(series.count, 32)
        XCTAssertEqual(series.utcTimestamps[0], PluginContractSuiteTools.defaultSampleTimeUTC)
        XCTAssertEqual(series.open[0], 110_000)
        XCTAssertEqual(series.close[0], 110_010)
        XCTAssertEqual(series.close[1], 110_015)
        XCTAssertTrue(series.hasVolume)
    }

    func testContractSuiteFinitePredictionReasonMatchesLegacyChecks() {
        XCTAssertNil(PluginContractSuiteTools.finitePredictionFailureReason(PredictionV4(
            classProbabilities: [0.2, 0.3, 0.5],
            moveMeanPoints: 1.0,
            moveQ25Points: 0.5,
            moveQ50Points: 1.0,
            moveQ75Points: 1.5,
            confidence: 0.5,
            reliability: 0.8
        )))
        XCTAssertEqual(
            PluginContractSuiteTools.finitePredictionFailureReason(PredictionV4(classProbabilities: [0.2, -0.1, 0.9])),
            "class_probs"
        )
        XCTAssertEqual(
            PluginContractSuiteTools.finitePredictionFailureReason(PredictionV4(classProbabilities: [0.2, 0.2, 0.2])),
            "probability_sum"
        )
        XCTAssertEqual(
            PluginContractSuiteTools.finitePredictionFailureReason(PredictionV4(
                classProbabilities: [0.2, 0.3, 0.5],
                moveMeanPoints: .infinity
            )),
            "prediction_fields"
        )
    }

    func testContractSuiteRunsAllCasesForValidPlugin() {
        let factory = PluginContractSuiteFactory {
            ContractSuiteGoodPlugin(manifest: Self.windowedManifest(aiID: 2))
        }

        let suite = PluginContractSuiteTools.runSuite(factories: [factory])

        XCTAssertTrue(suite.passed)
        XCTAssertEqual(suite.total, 5)
        XCTAssertEqual(suite.failed, 0)
        XCTAssertEqual(suite.legacyReason, "")
    }

    func testContractSuiteReportsRegistryAndPredictFailures() {
        let duplicateA = PluginContractSuiteFactory {
            ContractSuiteGoodPlugin(manifest: Self.windowedManifest(aiID: 3, name: "A"))
        }
        let duplicateB = PluginContractSuiteFactory {
            ContractSuiteGoodPlugin(manifest: Self.windowedManifest(aiID: 3, name: "B"))
        }
        let duplicateSuite = PluginContractSuiteTools.runSuite(factories: [duplicateA, duplicateB])
        XCTAssertEqual(duplicateSuite.cases[0].reason, "registry_duplicate_3")

        let badPrediction = PluginContractSuiteFactory {
            ContractSuiteBadPredictionPlugin(manifest: Self.windowedManifest(aiID: 4))
        }
        let predictionSuite = PluginContractSuiteTools.runSuite(factories: [badPrediction])
        XCTAssertEqual(predictionSuite.legacyReason, "predict_request_contract:predict_4_probability_sum")
    }

    private static func windowedManifest(aiID: Int, name: String = "ContractPlugin") -> PluginManifestV4 {
        PluginManifestV4(
            aiID: aiID,
            aiName: name,
            family: .transformer,
            capabilityMask: [.selfTest, .onlineLearning, .windowContext],
            minSequenceBars: 2,
            maxSequenceBars: 8
        )
    }
}

private struct ContractSuiteGoodPlugin: FXAIPluginPersistentState, FXAIPluginSyntheticSeriesSupport {
    var manifest: PluginManifestV4
    var loadedState = Data()
    var syntheticCount = 0

    mutating func reset() {}
    func selfTest() -> Bool { true }
    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {}
    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        PredictionV4(
            classProbabilities: [0.2, 0.3, 0.5],
            moveMeanPoints: 1.0,
            moveQ25Points: 0.5,
            moveQ50Points: 1.0,
            moveQ75Points: 1.5,
            mfeMeanPoints: 1.4,
            maeMeanPoints: 0.3,
            hitTimeFraction: 0.4,
            pathRisk: 0.2,
            fillRisk: 0.1,
            confidence: 0.5,
            reliability: 0.8
        )
    }
    mutating func saveStateData() throws -> Data {
        Data([1, 2, 3])
    }
    mutating func loadStateData(_ data: Data) throws {
        guard !data.isEmpty else { throw FXDataEngineError.validation("state") }
        loadedState = data
    }
    mutating func setSyntheticSeries(_ series: M1OHLCVSeries) throws {
        guard series.count == 32, series.hasVolume else { throw FXDataEngineError.validation("series") }
        syntheticCount = series.count
    }
    mutating func clearSyntheticSeries() {
        syntheticCount = 0
    }
}

private struct ContractSuiteBadPredictionPlugin: FXAIPluginV4 {
    var manifest: PluginManifestV4

    mutating func reset() {}
    func selfTest() -> Bool { true }
    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {}
    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        PredictionV4(classProbabilities: [0.2, 0.2, 0.2])
    }
}
