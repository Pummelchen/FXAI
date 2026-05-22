import XCTest
@testable import FXDataEngine

final class AuditSamplesTests: XCTestCase {
    func testAuditGeneratedSeriesConvertsToAlignedMarketUniverse() throws {
        let generated = try makeGeneratedSeries()
        let universe = try AuditSampleTools.marketUniverse(
            from: generated,
            symbol: "eurusd",
            pointValue: 0.0001
        )

        XCTAssertEqual(universe.primarySymbol, "EURUSD")
        XCTAssertEqual(universe.symbols, ["AUDITCTX1", "AUDITCTX2", "AUDITCTX3", "EURUSD"])
        XCTAssertEqual(universe.primary.count, generated.primary.count)
        XCTAssertEqual(universe.primary.utcTimestamps[0], generated.primary.timeUTC.last)
        XCTAssertEqual(universe.primary.utcTimestamps[universe.primary.count - 1], generated.primary.timeUTC[0])
        XCTAssertTrue(universe.primary.hasVolume)
        XCTAssertEqual(universe.primary.metadata.digits, 4)
    }

    func testAuditBuildSampleCreatesPredictAndTrainPayloadsWithAsSeriesFutureRules() throws {
        let generated = try makeGeneratedSeries()
        let manifest = PluginManifestV4(
            aiID: 0,
            aiName: "AuditLinear",
            family: .linear,
            capabilityMask: [.selfTest, .windowContext],
            minSequenceBars: 1,
            maxSequenceBars: 4
        )
        let sample = try AuditSampleTools.buildSample(
            generated: generated,
            sampleIndexAsSeries: 64,
            horizonMinutes: 8,
            manifest: manifest,
            symbol: "EURUSD",
            pointValue: 0.0001,
            priceCostPoints: 1.0,
            evThresholdPoints: 0.25
        )

        XCTAssertEqual(sample.sampleIndexAsSeries, 64)
        XCTAssertEqual(sample.sampleIndexAscending, generated.primary.count - 1 - 64)
        XCTAssertEqual(sample.payload.context.sampleTimeUTC, generated.primary.timeUTC[64])
        XCTAssertEqual(sample.payload.context.sequenceBars, 4)
        XCTAssertEqual(sample.predictRequest.windowSize, 3)
        XCTAssertEqual(sample.predictRequest.x.count, FXDataEngineConstants.aiWeights)
        XCTAssertEqual(sample.trainRequest.x.count, FXDataEngineConstants.aiWeights)
        XCTAssertTrue(sample.trainRequest.valid)
        XCTAssertTrue(sample.payload.context.dataHasVolume)
        XCTAssertGreaterThan(sample.payload.sample.sampleWeight, 0.0)
        XCTAssertGreaterThanOrEqual(sample.traceStats.liquidityMeanRatio, 0.5)
        try sample.predictRequest.validate()
        try sample.trainRequest.validate()
    }

    func testAuditBuildSampleRejectsInvalidAsSeriesIndexes() throws {
        let generated = try makeGeneratedSeries()
        let manifest = PluginManifestV4(
            aiID: 0,
            aiName: "AuditLinear",
            family: .linear,
            capabilityMask: [.selfTest]
        )

        XCTAssertThrowsError(try AuditSampleTools.buildSample(
            generated: generated,
            sampleIndexAsSeries: -1,
            horizonMinutes: 8,
            manifest: manifest
        ))
        XCTAssertThrowsError(try AuditSampleTools.buildSample(
            generated: generated,
            sampleIndexAsSeries: 4,
            horizonMinutes: 8,
            manifest: manifest
        ))
    }

    private func makeGeneratedSeries() throws -> AuditGeneratedScenarioSeries {
        try XCTUnwrap(AuditScenarioTools.generateSyntheticScenarioSeries(
            spec: AuditScenarioTools.scenarioSpec(scenarioID: 5),
            bars: 512,
            seed: 7,
            point: 0.0001
        ))
    }
}
