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
