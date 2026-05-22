import XCTest
@testable import FXDataEngine

final class AuditRunnerTests: XCTestCase {
    func testAuditRunnerConfigurationSanitizesNumericInputs() {
        let configuration = AuditRunnerConfiguration(
            horizonMinutes: 0,
            pointValue: .infinity,
            priceCostPoints: .nan,
            evThresholdPoints: -4.0,
            maxSamples: -3
        )

        XCTAssertEqual(configuration.pointValue, 0.0001)
        XCTAssertEqual(configuration.priceCostPoints, 0.0)
        XCTAssertEqual(configuration.evThresholdPoints, 0.0)
        XCTAssertEqual(configuration.maxSamples, 0)
    }

    func testAuditRunnerEvaluatesAndTrainsProtocolPlugin() throws {
        let spec = AuditScenarioTools.scenarioSpec(scenarioID: 5)
        let generated = try makeGeneratedSeries(spec: spec)
        var plugin = AlwaysBuyAuditPlugin()

        let metrics = try AuditRunnerTools.runScenario(
            plugin: &plugin,
            generated: generated,
            spec: spec,
            configuration: AuditRunnerConfiguration(
                horizonMinutes: 8,
                pointValue: 0.0001,
                priceCostPoints: 1.0,
                maxSamples: 12
            )
        )

        XCTAssertEqual(metrics.samplesTotal, 12)
        XCTAssertEqual(metrics.validPredictions, 12)
        XCTAssertEqual(metrics.invalidPredictions, 0)
        XCTAssertEqual(metrics.buyCount, 12)
        XCTAssertEqual(plugin.trainCount, 12)
        XCTAssertGreaterThan(metrics.trueBuyCount, 0)
        XCTAssertGreaterThan(metrics.score, 0.0)
        XCTAssertGreaterThanOrEqual(metrics.resetDelta, 0.0)
        XCTAssertGreaterThanOrEqual(metrics.sequenceDelta, 0.0)
    }

    func testAuditRunnerRecordsInvalidPredictionsAndStillTrains() throws {
        let spec = AuditScenarioTools.scenarioSpec(scenarioID: 0)
        let generated = try makeGeneratedSeries(spec: spec)
        var plugin = InvalidAuditPlugin()

        let metrics = try AuditRunnerTools.runScenario(
            plugin: &plugin,
            generated: generated,
            spec: spec,
            configuration: AuditRunnerConfiguration(
                horizonMinutes: 8,
                pointValue: 0.0001,
                maxSamples: 5
            )
        )

        XCTAssertEqual(metrics.samplesTotal, 5)
        XCTAssertEqual(metrics.validPredictions, 0)
        XCTAssertEqual(metrics.invalidPredictions, 5)
        XCTAssertEqual(plugin.trainCount, 5)
        XCTAssertTrue(metrics.issueFlags.contains(.invalidPrediction))
    }

    private func makeGeneratedSeries(spec: AuditScenarioSpec) throws -> AuditGeneratedScenarioSeries {
        try XCTUnwrap(AuditScenarioTools.generateSyntheticScenarioSeries(
            spec: spec,
            bars: 512,
            seed: 11,
            point: 0.0001
        ))
    }
}

private struct AlwaysBuyAuditPlugin: FXAIPluginV4 {
    var trainCount = 0
    let manifest = PluginManifestV4(
        aiID: 0,
        aiName: "AlwaysBuy",
        family: .linear,
        capabilityMask: [.selfTest, .windowContext],
        minSequenceBars: 1,
        maxSequenceBars: 4
    )

    mutating func reset() {}

    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        trainCount += 1
    }

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        return PredictionV4(
            classProbabilities: [0.08, 0.82, 0.10],
            moveMeanPoints: 8.0,
            moveQ25Points: 4.0,
            moveQ50Points: 8.0,
            moveQ75Points: 12.0,
            mfeMeanPoints: 10.0,
            maeMeanPoints: 3.0,
            hitTimeFraction: 0.35,
            pathRisk: 0.20,
            fillRisk: 0.10,
            confidence: 0.82,
            reliability: 0.70
        )
    }
}

private struct InvalidAuditPlugin: FXAIPluginV4 {
    var trainCount = 0
    let manifest = PluginManifestV4(
        aiID: 1,
        aiName: "Invalid",
        family: .linear,
        capabilityMask: [.selfTest]
    )

    mutating func reset() {}

    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        trainCount += 1
    }

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        return PredictionV4(classProbabilities: [0.0, 0.0, 0.0])
    }
}
