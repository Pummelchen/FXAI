import XCTest
@testable import FXDataEngine

final class AuditSuiteTests: XCTestCase {
    func testAuditSuiteParsesScenarioListWithAliasesAndFallback() {
        XCTAssertEqual(
            AuditSuiteTools.parseScenarioList("{drift_up|market_liquidity_shock;drift_up, market_macro_event}"),
            [1, 12, 14]
        )
        XCTAssertEqual(
            AuditSuiteTools.parseScenarioList("{unknown, }"),
            AuditSuiteTools.fallbackScenarioIDs
        )
    }

    func testAuditSuiteResolvesPluginsByAllNameIDAndFallback() {
        let first = AuditPluginFactory(makePlugin: { SuiteAlwaysBuyPlugin(aiID: 3, name: "Alpha") })
        let second = AuditPluginFactory(makePlugin: { SuiteAlwaysBuyPlugin(aiID: 7, name: "Beta") })
        let plugins = [first, second]

        XCTAssertEqual(
            AuditSuiteTools.resolvePlugins(plugins, allPlugins: true, pluginList: "{}").map(\.manifest.aiID),
            [3, 7]
        )
        XCTAssertEqual(
            AuditSuiteTools.resolvePlugins(plugins, allPlugins: false, pluginList: "{beta|3}").map(\.manifest.aiID),
            [3, 7]
        )
        XCTAssertEqual(
            AuditSuiteTools.resolvePlugins(plugins, allPlugins: false, pluginList: "{missing}", fallbackAIID: 7)
                .map(\.manifest.aiID),
            [7]
        )
    }

    func testAuditSuiteRunsSyntheticScenariosAndBuildsReport() throws {
        let plugin = AuditPluginFactory(makePlugin: { SuiteAlwaysBuyPlugin(aiID: 5, name: "SuiteBuy") })
        let result = try AuditSuiteTools.runSuite(
            plugins: [plugin],
            configuration: AuditSuiteConfiguration(
                bars: 2_048,
                scenarioList: "{monotonic_up, drift_up}",
                seed: 17,
                runner: AuditRunnerConfiguration(
                    horizonMinutes: 8,
                    pointValue: 0.0001,
                    priceCostPoints: 1.0,
                    maxSamples: 4
                )
            )
        )

        XCTAssertEqual(result.attemptedRuns, 2)
        XCTAssertEqual(result.skippedRuns, 0)
        XCTAssertEqual(result.metrics.count, 2)
        XCTAssertEqual(result.metrics.map(\.aiID), [5, 5])
        XCTAssertEqual(result.metrics.map(\.samplesTotal), [4, 4])
        XCTAssertTrue(result.reportDocument.hasPrefix(AuditReportTools.headerLine))
        XCTAssertEqual(nonEmptyReportLines(result.reportDocument).count, 3)
    }

    func testAuditSuiteSkipsMarketScenariosWithoutMarketSeries() throws {
        let plugin = AuditPluginFactory(makePlugin: { SuiteAlwaysBuyPlugin(aiID: 5, name: "SuiteBuy") })
        let result = try AuditSuiteTools.runSuite(
            plugins: [plugin],
            configuration: AuditSuiteConfiguration(
                bars: 2_048,
                scenarioList: "{market_recent}",
                runner: AuditRunnerConfiguration(horizonMinutes: 8, maxSamples: 4)
            )
        )

        XCTAssertEqual(result.attemptedRuns, 1)
        XCTAssertEqual(result.skippedRuns, 1)
        XCTAssertEqual(result.metrics.count, 0)
        XCTAssertEqual(nonEmptyReportLines(result.reportDocument).count, 1)
    }

    func testAuditSuiteSkipsAdversarialScenarioUntilDedicatedGeneratorExists() throws {
        let plugin = AuditPluginFactory(makePlugin: { SuiteAlwaysBuyPlugin(aiID: 5, name: "SuiteBuy") })
        let result = try AuditSuiteTools.runSuite(
            plugins: [plugin],
            configuration: AuditSuiteConfiguration(
                bars: 2_048,
                scenarioList: "{market_adversarial}",
                runner: AuditRunnerConfiguration(horizonMinutes: 8, maxSamples: 4)
            ),
            marketSeries: try makeMarketSeries(count: 64)
        )

        XCTAssertEqual(result.attemptedRuns, 1)
        XCTAssertEqual(result.skippedRuns, 1)
        XCTAssertEqual(result.metrics.count, 0)
    }

    func testAuditSuiteReportWriterResetsAndAppendsRowsWithoutDuplicateHeader() throws {
        let plugin = AuditPluginFactory(makePlugin: { SuiteAlwaysBuyPlugin(aiID: 5, name: "SuiteBuy") })
        let result = try AuditSuiteTools.runSuite(
            plugins: [plugin],
            configuration: AuditSuiteConfiguration(
                bars: 2_048,
                scenarioList: "{monotonic_up, drift_up}",
                seed: 17,
                runner: AuditRunnerConfiguration(horizonMinutes: 8, maxSamples: 2)
            )
        )
        let fileManager = FileManager.default
        let directory = fileManager.temporaryDirectory
            .appendingPathComponent("FXDataEngineAuditSuiteTests-\(UUID().uuidString)")
        let reportURL = directory.appendingPathComponent("audit.tsv")
        defer { try? fileManager.removeItem(at: directory) }

        try AuditSuiteTools.writeReport(metrics: [result.metrics[0]], to: reportURL, resetOutput: true)
        try AuditSuiteTools.writeReport(metrics: [result.metrics[1]], to: reportURL, resetOutput: false)

        let text = try String(contentsOf: reportURL, encoding: .utf8)
        let lines = nonEmptyReportLines(text)
        XCTAssertEqual(lines.count, 3)
        XCTAssertEqual(lines.filter { $0 == AuditReportTools.headerLine }.count, 1)

        let emptyReportURL = directory.appendingPathComponent("empty-append.tsv")
        try Data().write(to: emptyReportURL)
        try AuditSuiteTools.writeReport(metrics: [result.metrics[0]], to: emptyReportURL, resetOutput: false)
        let emptyAppendText = try String(contentsOf: emptyReportURL, encoding: .utf8)
        XCTAssertEqual(nonEmptyReportLines(emptyAppendText).first, AuditReportTools.headerLine)
    }

    private func nonEmptyReportLines(_ text: String) -> [String] {
        text.components(separatedBy: "\r\n").filter { !$0.isEmpty }
    }

    private func makeMarketSeries(count: Int) throws -> M1OHLCVSeries {
        let start = Int64(1_704_067_200)
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        for index in 0..<count {
            let base = Int64(110_000 + index)
            utc.append(start + Int64(index * 60))
            open.append(base)
            high.append(base + 8)
            low.append(base - 8)
            close.append(base + 2)
            volume.append(UInt64(100 + index))
        }
        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "demo",
                sourceOrigin: "DUKASCOPY",
                logicalSymbol: "EURUSD",
                providerSymbol: "EUR/USD",
                digits: 5,
                firstUTC: utc.first,
                lastUTC: utc.last
            ),
            utcTimestamps: utc,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }
}

private struct SuiteAlwaysBuyPlugin: FXAIPluginV4 {
    let manifest: PluginManifestV4

    init(aiID: Int, name: String) {
        self.manifest = PluginManifestV4(
            aiID: aiID,
            aiName: name,
            family: .linear,
            capabilityMask: [.selfTest, .windowContext],
            minSequenceBars: 1,
            maxSequenceBars: 3
        )
    }

    mutating func reset() {}

    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
    }

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        return PredictionV4(
            classProbabilities: [0.10, 0.80, 0.10],
            moveMeanPoints: 7.0,
            moveQ25Points: 3.0,
            moveQ50Points: 7.0,
            moveQ75Points: 10.0,
            mfeMeanPoints: 8.0,
            maeMeanPoints: 3.0,
            hitTimeFraction: 0.40,
            pathRisk: 0.20,
            fillRisk: 0.10,
            confidence: 0.80,
            reliability: 0.70
        )
    }
}
