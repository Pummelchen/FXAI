import FXBacktestCore
import FXBacktestAPI
import FXBacktestPlugins
import XCTest

final class OperationalAgentsTests: XCTestCase {
    func testFXDatabaseConnectivityAcceptsMatchingAPIVersion() async throws {
        let agent = FXDatabaseConnectivityAgent(statusLoader: { _ in
            FXBacktestAPIStatusResponse(apiVersion: FXBacktestAPIV1.latestVersion, service: "FXDatabase", status: "ok")
        })

        let outcome = try await agent.check(connection: FXDatabaseConnectionSettings())

        XCTAssertEqual(outcome.status, .ok)
    }

    func testFXDatabaseConnectivityRejectsWrongAPIVersion() async throws {
        let agent = FXDatabaseConnectivityAgent(statusLoader: { _ in
            FXBacktestAPIStatusResponse(apiVersion: "wrong.version", service: "FXDatabase", status: "ok")
        })

        let outcome = try await agent.check(connection: FXDatabaseConnectionSettings())

        XCTAssertEqual(outcome.status, .failed)
        XCTAssertTrue(outcome.message.contains("version mismatch"))
    }

    func testMarketReadinessRejectsMixedDemoAndFXDatabaseData() throws {
        let demo = try market(symbol: "EURUSD", brokerSourceId: "demo")
        let databaseBacked = try market(symbol: "USDJPY", brokerSourceId: "icmarkets-sc-mt5-4", mt5Symbol: "USDJPY", digits: 3)
        let universe = try OhlcMarketUniverse(primarySymbol: "EURUSD", series: [demo, databaseBacked])

        let outcome = MarketReadinessAgent().evaluate(universe: universe)

        XCTAssertEqual(outcome.status, .failed)
        XCTAssertTrue(outcome.message.contains("Cannot mix demo and FXDatabase"))
    }

    func testPluginValidationAcceptsBundledPlugins() {
        let agent = PluginValidationAgent()

        for plugin in FXBacktestPluginRegistry.availablePlugins {
            XCTAssertEqual(plugin.descriptor.apiVersion, .latest)
            let outcome = agent.validate(plugin: plugin)
            XCTAssertEqual(outcome.status, .ok, outcome.message)
        }
    }

    func testPluginValidationRejectsInvalidLatestAPIDescriptor() {
        let outcome = PluginValidationAgent().validate(plugin: AnyFXBacktestPlugin(InvalidDescriptorTestPlugin()))

        XCTAssertEqual(outcome.status, .failed)
        XCTAssertTrue(outcome.message.contains("must support at least one execution backend"))
    }

    func testOptimizationCoordinatorRejectsMetalForCPUOnlyPlugin() throws {
        let plugin = AnyFXBacktestPlugin(CPUOnlyTestPlugin())
        let market = try market(symbol: "EURUSD", brokerSourceId: "demo")
        let sweep = try ParameterSweep.singlePass(definitions: plugin.parameterDefinitions)

        XCTAssertThrowsError(
            try OptimizationRunCoordinatorAgent().prepare(
                plugin: plugin,
                marketUniverse: market.universe,
                sweep: sweep,
                target: .metal,
                maxWorkers: 1,
                chunkSize: 1,
                initialDeposit: 10_000,
                contractSize: 100_000,
                lotSize: 0.01
            )
        )
    }

    func testResourceHealthAcceptsCPUExecution() {
        let outcome = ResourceHealthAgent().evaluate(target: .cpu, maxWorkers: 1, chunkSize: 1)

        XCTAssertNotEqual(outcome.status, .failed)
    }

    func testResultPersistenceAgentSavesAndPurgesThroughStoreAPI() async throws {
        let client = RecordingFXDatabaseResultClient()
        let store = FXDatabaseBacktestResultStore(client: client)
        let plugin = FX7()
        let sweep = try ParameterSweep.singlePass(definitions: plugin.parameterDefinitions)
        let run = BacktestStoredRun(
            runID: "agent-run",
            pluginIdentifier: plugin.descriptor.id,
            engine: .cpu,
            brokerSourceId: "demo",
            primarySymbol: "EURUSD",
            symbols: ["EURUSD"],
            settings: BacktestRunSettings(),
            sweep: sweep,
            note: "agent-test"
        )
        let result = BacktestPassResult(
            passIndex: 0,
            pluginIdentifier: plugin.descriptor.id,
            engine: .cpu,
            parameters: [BacktestParameterValue(key: "signal_stride_bars", value: 15)],
            netProfit: 1,
            grossProfit: 1,
            grossLoss: 0,
            maxDrawdown: 0,
            totalTrades: 1,
            winningTrades: 1,
            losingTrades: 0,
            winRate: 1,
            profitFactor: 1,
            barsProcessed: 10
        )

        let saveOutcome = try await ResultPersistenceAgent.saveSnapshot(
            store: store,
            run: run,
            results: [result],
            progress: BacktestProgress(completedPasses: 1, totalPasses: 1, elapsedSeconds: 0.1),
            status: "completed"
        )
        let purge = try await ResultPersistenceAgent.purgeAll(store: store)

        let operations = await client.operations()
        XCTAssertEqual(saveOutcome.status, .ok)
        XCTAssertEqual(purge.outcome.status, .ok)
        XCTAssertEqual(operations, [
            "start:agent-run:com.fxbacktest.plugins.fx7.v1:1",
            "append:agent-run:1",
            "complete:agent-run:1:completed",
            "schema",
            "purge:all"
        ])
    }

    private func market(
        symbol: String,
        brokerSourceId: String,
        mt5Symbol: String? = nil,
        digits: Int = 5
    ) throws -> OhlcDataSeries {
        let start = Int64(1_704_067_200)
        let utc = ContiguousArray((0..<10).map { start + Int64($0 * 60) })
        let close = ContiguousArray((0..<10).map { Int64(100_000 + $0 * 10) })
        return try OhlcDataSeries(
            metadata: FXBacktestMarketMetadata(
                brokerSourceId: brokerSourceId,
                logicalSymbol: symbol,
                mt5Symbol: mt5Symbol,
                digits: digits,
                firstUtc: utc.first,
                lastUtc: utc.last
            ),
            utcTimestamps: utc,
            open: close,
            high: ContiguousArray(close.map { $0 + 5 }),
            low: ContiguousArray(close.map { $0 - 5 }),
            close: close
        )
    }

}

private struct CPUOnlyTestPlugin: FXBacktestPluginV1 {
    let descriptor = FXBacktestPluginDescriptor(
        id: "com.fxbacktest.tests.cpu-only.v1",
        displayName: "CPU Only Test",
        version: "1.0.0",
        summary: "CPU-only test plugin for operational-agent target validation.",
        author: "FXBacktestTests",
        supportsCPU: true,
        supportsMetal: false
    )

    let parameterDefinitions: [ParameterDefinition] = [
        try! ParameterDefinition(
            key: "dummy",
            displayName: "Dummy",
            defaultValue: 1,
            defaultMinimum: 1,
            defaultStep: 1,
            defaultMaximum: 1,
            valueKind: .integer
        )
    ]

    func runPass(
        market: OhlcDataSeries,
        parameters: ParameterVector,
        context: BacktestContext
    ) throws -> BacktestPassResult {
        BacktestPassResult(
            passIndex: parameters.combinationIndex,
            pluginIdentifier: descriptor.id,
            engine: context.settings.target,
            parameters: parameters.snapshots,
            netProfit: 0,
            grossProfit: 0,
            grossLoss: 0,
            maxDrawdown: 0,
            totalTrades: 0,
            winningTrades: 0,
            losingTrades: 0,
            winRate: 0,
            profitFactor: 0,
            barsProcessed: market.count
        )
    }
}

private struct InvalidDescriptorTestPlugin: FXBacktestPluginV1 {
    let descriptor = FXBacktestPluginDescriptor(
        id: "com.fxbacktest.tests.invalid.v1",
        displayName: "Invalid Test",
        version: "1.0.0",
        summary: "Invalid test plugin.",
        author: "FXBacktestTests",
        supportsCPU: false,
        supportsMetal: false
    )

    let parameterDefinitions: [ParameterDefinition] = []

    func runPass(
        market: OhlcDataSeries,
        parameters: ParameterVector,
        context: BacktestContext
    ) throws -> BacktestPassResult {
        throw FXBacktestError.invalidParameter("not executable")
    }
}
