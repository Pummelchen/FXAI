import FXBacktestAPI
import FXBacktestCore
import FXBacktestPlugins
import XCTest

final class BacktestResultStoreTests: XCTestCase {
    func testFXDatabaseResultStoreUsesFXDatabaseBacktestAPI() async throws {
        let client = RecordingFXDatabaseResultClient()
        let store = FXDatabaseBacktestResultStore(client: client)
        let plugin = FX7()
        let sweep = try ParameterSweep.singlePass(definitions: plugin.parameterDefinitions)
        let run = BacktestStoredRun(
            runID: "test-run",
            pluginIdentifier: plugin.descriptor.id,
            engine: .cpu,
            brokerSourceId: "demo",
            primarySymbol: "EURUSD",
            symbols: ["EURUSD"],
            settings: BacktestRunSettings(),
            sweep: sweep,
            note: "unit"
        )
        let result = BacktestPassResult(
            passIndex: 0,
            pluginIdentifier: plugin.descriptor.id,
            engine: .cpu,
            parameters: [BacktestParameterValue(key: "signal_stride_bars", value: 15)],
            netProfit: 12,
            grossProfit: 20,
            grossLoss: -8,
            maxDrawdown: 3,
            totalTrades: 2,
            winningTrades: 1,
            losingTrades: 1,
            winRate: 0.5,
            profitFactor: 2.5,
            barsProcessed: 100
        )

        try await store.ensureSchema()
        try await store.startRun(run)
        try await store.appendResults([result], runID: run.runID)
        try await store.completeRun(
            runID: run.runID,
            progress: BacktestProgress(completedPasses: 1, totalPasses: 1, elapsedSeconds: 0.1),
            status: "completed"
        )
        _ = try await store.purge(olderThanDays: 30)

        let operations = await client.operations()
        XCTAssertEqual(operations, [
            "schema",
            "start:test-run:com.fxbacktest.plugins.fx7.v1:1",
            "append:test-run:1",
            "complete:test-run:1:completed",
            "purge:older_than_days:30"
        ])
    }

    func testFXDatabaseConfigurationStoreRegistersSharedAndPluginAcceleratorParametersThroughAPI() async throws {
        let client = RecordingFXDatabaseConfigurationClient()
        let store = FXDatabaseBacktestConfigurationStore(client: client)
        let plugin = AnyFXBacktestPlugin(FX7())
        let response = try await store.register(plugins: [plugin])

        XCTAssertEqual(response.affectedRows, BacktestCommonConfigurationDefaults.parameters().count + plugin.parameterDefinitions.count * 3)
        let snapshot = try await store.fetch(pluginIds: [plugin.descriptor.id])
        XCTAssertEqual(snapshot.sharedParameters.first { $0.key == "initial_deposit_usd" }?.defaultValue, 1_000)
        XCTAssertEqual(snapshot.sharedParameters.first { $0.key == "lot_size_lots" }?.defaultValue, 0.01)
        XCTAssertEqual(Set(snapshot.pluginConfigurations.map(\.acceleratorId)), ["swiftScalar", "swiftSIMD", "metal"])
        XCTAssertTrue(snapshot.pluginConfigurations.allSatisfy { $0.parameters.contains { $0.key == "signal_stride_bars" } })

        let operations = await client.operations()
        XCTAssertEqual(operations, [
            "config-schema",
            "config-register:5:3",
            "config-get:com.fxbacktest.plugins.fx7.v1"
        ])
    }
}

actor RecordingFXDatabaseResultClient: FXDatabaseBacktestResultAPIClient {
    private var recordedOperations: [String] = []

    func ensureBacktestResultSchema() async throws -> FXBacktestResultMutationResponse {
        recordedOperations.append("schema")
        return FXBacktestResultMutationResponse(sqlStatements: 2)
    }

    func startBacktestRun(_ run: FXBacktestResultRunStartRequest) async throws -> FXBacktestResultMutationResponse {
        recordedOperations.append("start:\(run.runId):\(run.pluginId):\(run.totalPasses)")
        return FXBacktestResultMutationResponse(runId: run.runId, affectedRows: 1, sqlStatements: 3)
    }

    func appendBacktestResults(_ results: FXBacktestResultPassAppendRequest) async throws -> FXBacktestResultMutationResponse {
        recordedOperations.append("append:\(results.runId):\(results.results.count)")
        return FXBacktestResultMutationResponse(runId: results.runId, affectedRows: results.results.count, sqlStatements: 1)
    }

    func completeBacktestRun(_ completion: FXBacktestResultRunCompleteRequest) async throws -> FXBacktestResultMutationResponse {
        recordedOperations.append("complete:\(completion.runId):\(completion.completedPasses):\(completion.status)")
        return FXBacktestResultMutationResponse(runId: completion.runId, affectedRows: 1, sqlStatements: 1)
    }

    func purgeBacktestResults(_ purge: FXBacktestResultPurgeRequest) async throws -> FXBacktestResultPurgeResponse {
        let scope = purge.all ? "all" : "older_than_days:\(purge.olderThanDays ?? 0)"
        recordedOperations.append("purge:\(scope)")
        return FXBacktestResultPurgeResponse(report: FXBacktestResultPurgeReport(scope: scope, sqlStatements: 2))
    }

    func getBacktestRun(_ run: FXBacktestResultRunGetRequest) async throws -> FXBacktestResultRunGetResponse {
        recordedOperations.append("get-run:\(run.runId)")
        return FXBacktestResultRunGetResponse(run: nil)
    }

    func getBacktestPasses(_ passes: FXBacktestResultPassesGetRequest) async throws -> FXBacktestResultPassesGetResponse {
        recordedOperations.append("get-passes:\(passes.runId):\(passes.offset):\(passes.limit)")
        return FXBacktestResultPassesGetResponse(runId: passes.runId, offset: passes.offset, limit: passes.limit, results: [])
    }

    func operations() -> [String] {
        recordedOperations
    }
}

actor RecordingFXDatabaseConfigurationClient: FXDatabaseBacktestConfigurationAPIClient {
    private var recordedOperations: [String] = []
    private var snapshot = FXBacktestConfigurationSnapshotResponse(sharedParameters: [], pluginConfigurations: [])

    func ensureBacktestConfigurationSchema() async throws -> FXBacktestResultMutationResponse {
        recordedOperations.append("config-schema")
        return FXBacktestResultMutationResponse(affectedRows: 0, sqlStatements: 3)
    }

    func registerBacktestConfiguration(_ registration: FXBacktestConfigurationRegistrationRequest) async throws -> FXBacktestResultMutationResponse {
        recordedOperations.append("config-register:\(registration.sharedParameters.count):\(registration.pluginConfigurations.count)")
        snapshot = FXBacktestConfigurationSnapshotResponse(
            sharedParameters: registration.sharedParameters,
            pluginConfigurations: registration.pluginConfigurations
        )
        let rows = registration.sharedParameters.count + registration.pluginConfigurations.reduce(0) { $0 + $1.parameters.count }
        return FXBacktestResultMutationResponse(affectedRows: rows, sqlStatements: 5)
    }

    func getBacktestConfiguration(_ request: FXBacktestConfigurationGetRequest) async throws -> FXBacktestConfigurationSnapshotResponse {
        recordedOperations.append("config-get:\(request.pluginIds?.joined(separator: ",") ?? "all")")
        guard let pluginIds = request.pluginIds else { return snapshot }
        return FXBacktestConfigurationSnapshotResponse(
            sharedParameters: snapshot.sharedParameters,
            pluginConfigurations: snapshot.pluginConfigurations.filter { pluginIds.contains($0.pluginId) }
        )
    }

    func operations() -> [String] {
        recordedOperations
    }
}
