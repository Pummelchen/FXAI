import ClickHouse
import Config
import Domain
import FXBacktestAPI
import FXBacktestAPIServer
import XCTest

final class FXBacktestAPITests: XCTestCase {
    func testHistoryRequestRequiresV1VersionAndMinuteAlignedRange() throws {
        XCTAssertThrowsError(try FXBacktestM1HistoryRequest(
            apiVersion: "old",
            brokerSourceId: "demo",
            logicalSymbol: "EURUSD",
            utcStartInclusive: 1_704_067_200,
            utcEndExclusive: 1_704_067_260
        ).validate()) { error in
            XCTAssertEqual(error as? FXBacktestAPIValidationError, .unsupportedVersion("old"))
        }

        XCTAssertThrowsError(try FXBacktestM1HistoryRequest(
            brokerSourceId: "demo",
            logicalSymbol: "EURUSD",
            utcStartInclusive: 1_704_067_201,
            utcEndExclusive: 1_704_067_260
        ).validate())

        XCTAssertThrowsError(try FXBacktestM1HistoryRequest(
            brokerSourceId: "demo",
            logicalSymbol: "EURUSD",
            utcStartInclusive: 1_704_067_200,
            utcEndExclusive: 1_704_067_260,
            maximumRows: FXBacktestAPIV1.maximumRowsLimit + 1
        ).validate()) { error in
            guard case .invalidField(let message) = error as? FXBacktestAPIValidationError else {
                XCTFail("Expected invalidField, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("maximum_rows"))
        }
    }

    func testHTTPHandlerServesStatusAndHistoryWithV1Envelope() async throws {
        let handler = FXBacktestAPIHTTPHandler(historyProvider: MockHistoryProvider())

        let statusResponse = await handler.handle(method: "GET", path: FXBacktestAPIV1.statusPath, body: Data())
        XCTAssertEqual(statusResponse.statusCode, 200)
        let status = try JSONDecoder().decode(FXBacktestAPIStatusResponse.self, from: statusResponse.body)
        XCTAssertEqual(status.apiVersion, FXBacktestAPIV1.version)

        let request = FXBacktestM1HistoryRequest(
            brokerSourceId: "demo",
            logicalSymbol: "EURUSD",
            utcStartInclusive: 1_704_067_200,
            utcEndExclusive: 1_704_067_320,
            expectedMT5Symbol: "EURUSD",
            expectedDigits: 5,
            maximumRows: 2
        )
        let body = try JSONEncoder().encode(request)
        let historyResponse = await handler.handle(method: "POST", path: FXBacktestAPIV1.m1HistoryPath, body: body)
        XCTAssertEqual(historyResponse.statusCode, 200)
        let history = try JSONDecoder().decode(FXBacktestM1HistoryResponse.self, from: historyResponse.body)
        XCTAssertEqual(history.apiVersion, FXBacktestAPIV1.version)
        XCTAssertEqual(history.metadata.rowCount, 2)
        XCTAssertEqual(history.utcTimestamps, [1_704_067_200, 1_704_067_260])
    }

    func testHTTPHandlerRejectsUnsupportedAPIVersion() async throws {
        let handler = FXBacktestAPIHTTPHandler(historyProvider: MockHistoryProvider())
        let request = FXBacktestM1HistoryRequest(
            apiVersion: "v0",
            brokerSourceId: "demo",
            logicalSymbol: "EURUSD",
            utcStartInclusive: 1_704_067_200,
            utcEndExclusive: 1_704_067_260
        )
        let response = await handler.handle(
            method: "POST",
            path: FXBacktestAPIV1.m1HistoryPath,
            body: try JSONEncoder().encode(request)
        )

        XCTAssertEqual(response.statusCode, 400)
        let error = try JSONDecoder().decode(FXBacktestAPIErrorResponse.self, from: response.body)
        XCTAssertEqual(error.apiVersion, FXBacktestAPIV1.version)
        XCTAssertEqual(error.error.code, "invalid_request")
    }

    func testHistoryResponseValidationRejectsMetadataDrift() throws {
        let response = FXBacktestM1HistoryResponse(
            metadata: FXBacktestM1HistoryMetadata(
                brokerSourceId: "demo",
                logicalSymbol: "EURUSD",
                mt5Symbol: "EURUSD",
                digits: 5,
                requestedUtcStart: 1_704_067_200,
                requestedUtcEndExclusive: 1_704_067_260,
                firstUtc: 1_704_067_260,
                lastUtc: 1_704_067_260,
                rowCount: 1
            ),
            utcTimestamps: [1_704_067_200],
            open: [108_000],
            high: [108_020],
            low: [107_990],
            close: [108_010],
            volume: [0]
        )

        XCTAssertThrowsError(try response.validate())
    }

    func testHTTPHandlerMapsProviderInvalidRequestToV1Error() async throws {
        let handler = FXBacktestAPIHTTPHandler(historyProvider: InvalidRequestProvider())
        let request = FXBacktestM1HistoryRequest(
            brokerSourceId: "demo",
            logicalSymbol: "EURUSD",
            utcStartInclusive: 1_704_067_200,
            utcEndExclusive: 1_704_067_260
        )
        let response = await handler.handle(
            method: "POST",
            path: FXBacktestAPIV1.m1HistoryPath,
            body: try JSONEncoder().encode(request)
        )

        XCTAssertEqual(response.statusCode, 400)
        let error = try JSONDecoder().decode(FXBacktestAPIErrorResponse.self, from: response.body)
        XCTAssertEqual(error.apiVersion, FXBacktestAPIV1.version)
        XCTAssertEqual(error.error.code, "invalid_request")
    }

    func testHTTPHandlerRoutesBacktestResultRequestsThroughResultProvider() async throws {
        let resultProvider = MockResultProvider()
        let handler = FXBacktestAPIHTTPHandler(historyProvider: MockHistoryProvider(), resultProvider: resultProvider)

        let schemaResponse = await handler.handle(
            method: "POST",
            path: FXBacktestAPIV1.resultSchemaPath,
            body: try JSONEncoder().encode(FXBacktestResultSchemaRequest())
        )
        XCTAssertEqual(schemaResponse.statusCode, 200)

        let startRequest = FXBacktestResultRunStartRequest(
            runId: "run-1",
            pluginId: "com.fxbacktest.tests.plugin.v1",
            engine: "cpu",
            brokerSourceId: "demo",
            primarySymbol: "EURUSD",
            symbols: ["EURUSD"],
            settingsJSON: "{}",
            parameterSpaceJSON: "{}",
            totalPasses: 1
        )
        let startResponse = await handler.handle(
            method: "POST",
            path: FXBacktestAPIV1.resultRunStartPath,
            body: try JSONEncoder().encode(startRequest)
        )
        XCTAssertEqual(startResponse.statusCode, 200)
        let start = try JSONDecoder().decode(FXBacktestResultMutationResponse.self, from: startResponse.body)
        XCTAssertEqual(start.runId, "run-1")

        let appendRequest = FXBacktestResultPassAppendRequest(
            runId: "run-1",
            results: [FXBacktestResultPassDTO(
                passIndex: 0,
                pluginId: "com.fxbacktest.tests.plugin.v1",
                engine: "cpu",
                netProfit: 1,
                grossProfit: 1,
                grossLoss: 0,
                maxDrawdown: 0,
                totalTrades: 1,
                winningTrades: 1,
                losingTrades: 0,
                winRate: 1,
                profitFactor: 1,
                barsProcessed: 10,
                parametersJSON: "[]"
            )]
        )
        let appendResponse = await handler.handle(
            method: "POST",
            path: FXBacktestAPIV1.resultPassAppendPath,
            body: try JSONEncoder().encode(appendRequest)
        )
        XCTAssertEqual(appendResponse.statusCode, 200)

        let operations = await resultProvider.operations()
        XCTAssertEqual(operations, [
            "schema",
            "start:run-1",
            "append:run-1:1"
        ])
    }

    func testHTTPHandlerRequiresConfiguredResultProviderForResultEndpoints() async throws {
        let handler = FXBacktestAPIHTTPHandler(historyProvider: MockHistoryProvider())

        let response = await handler.handle(
            method: "POST",
            path: FXBacktestAPIV1.resultSchemaPath,
            body: try JSONEncoder().encode(FXBacktestResultSchemaRequest())
        )

        XCTAssertEqual(response.statusCode, 503)
        let error = try JSONDecoder().decode(FXBacktestAPIErrorResponse.self, from: response.body)
        XCTAssertEqual(error.apiVersion, FXBacktestAPIV1.version)
        XCTAssertEqual(error.error.code, "result_store_unavailable")
    }

    func testResultServiceOwnsClickHouseSchemaAndMutationsBehindAPI() async throws {
        let clickHouse = RecordingResultClickHouse()
        let service = FXDatabaseBacktestResultService(clickHouse: clickHouse, database: "fxdatabase_test")
        let start = FXBacktestResultRunStartRequest(
            runId: "run-service",
            pluginId: "com.fxbacktest.tests.plugin.v1",
            engine: "cpu",
            brokerSourceId: "demo",
            primarySymbol: "EURUSD",
            symbols: ["EURUSD"],
            settingsJSON: "{}",
            parameterSpaceJSON: "{}",
            totalPasses: 1
        )
        let pass = FXBacktestResultPassDTO(
            passIndex: 0,
            pluginId: "com.fxbacktest.tests.plugin.v1",
            engine: "cpu",
            netProfit: 1,
            grossProfit: 1,
            grossLoss: 0,
            maxDrawdown: 0,
            totalTrades: 1,
            winningTrades: 1,
            losingTrades: 0,
            winRate: 1,
            profitFactor: 1,
            barsProcessed: 10,
            parametersJSON: "[]"
        )

        _ = try await service.ensureResultSchema(FXBacktestResultSchemaRequest())
        _ = try await service.startRun(start)
        _ = try await service.appendPassResults(FXBacktestResultPassAppendRequest(runId: "run-service", results: [pass]))
        _ = try await service.completeRun(FXBacktestResultRunCompleteRequest(
            runId: "run-service",
            completedPasses: 1,
            elapsedSeconds: 0.1,
            status: "completed"
        ))
        _ = try await service.purgeResults(FXBacktestResultPurgeRequest(all: true))

        let sql = await clickHouse.sql()
        XCTAssertTrue(sql.contains { $0.contains("CREATE DATABASE IF NOT EXISTS `fxdatabase_test`") })
        XCTAssertTrue(sql.contains { $0.contains("CREATE TABLE IF NOT EXISTS `fxdatabase_test`.`fxbacktest_runs`") })
        XCTAssertTrue(sql.contains { $0.contains("CREATE TABLE IF NOT EXISTS `fxdatabase_test`.`fxbacktest_pass_results`") })
        XCTAssertTrue(sql.contains { $0.contains("INSERT INTO `fxdatabase_test`.`fxbacktest_runs`") })
        XCTAssertTrue(sql.contains { $0.contains("INSERT INTO `fxdatabase_test`.`fxbacktest_pass_results` FORMAT JSONEachRow") })
        XCTAssertTrue(sql.contains { $0.contains("ALTER TABLE `fxdatabase_test`.`fxbacktest_runs`") && $0.contains("status = 'completed'") })
        XCTAssertTrue(sql.contains { $0.contains("ALTER TABLE `fxdatabase_test`.`fxbacktest_pass_results` DELETE WHERE 1") })
    }

    func testHistoryServiceServesSineTestAsVirtualSecurityWithoutClickHouseReadiness() async throws {
        let clickHouse = NeverCalledClickHouse()
        let service = FXDatabaseBacktestHistoryService(config: try Self.makeConfig(), clickHouse: clickHouse)
        let start = 1_704_067_200
        let response = try await service.loadM1History(FXBacktestM1HistoryRequest(
            brokerSourceId: "any-virtual-broker",
            logicalSymbol: "SINETEST",
            utcStartInclusive: Int64(start),
            utcEndExclusive: Int64(start + 61 * 60),
            expectedMT5Symbol: "SineTest",
            expectedDigits: 6
        ))

        try response.validate()
        XCTAssertEqual(response.metadata.sourceOrigin, "SYNTHETIC")
        XCTAssertEqual(response.metadata.logicalSymbol, "SINETEST")
        XCTAssertEqual(response.metadata.mt5Symbol, "SineTest")
        XCTAssertEqual(response.metadata.rowCount, 61)
        XCTAssertEqual(response.open[0], 1_000_000)
        XCTAssertEqual(response.open[30], 1)
        XCTAssertEqual(response.open[60], 1_000_000)
        XCTAssertTrue(response.volume.allSatisfy { $0 > 0 })
        let executeCount = await clickHouse.executeCount()
        XCTAssertEqual(executeCount, 0)
    }

    func testHTTPHandlerDoesNotExposeExecutionSpecEndpoint() async throws {
        let handler = FXBacktestAPIHTTPHandler(historyProvider: MockHistoryProvider())

        let response = await handler.handle(
            method: "POST",
            path: "/v1/execution/spec",
            body: Data("{}".utf8)
        )

        XCTAssertEqual(response.statusCode, 404)
        let error = try JSONDecoder().decode(FXBacktestAPIErrorResponse.self, from: response.body)
        XCTAssertEqual(error.error.code, "not_found")
    }

    private static func makeConfig() throws -> ConfigBundle {
        let appData = """
        {
          "chunk_size": 50000,
          "live_scan_interval_seconds": 10,
          "log_level": "normal",
          "strict_symbol_failures": false,
          "verifier_random_ranges": 0
        }
        """.data(using: .utf8)!
        return ConfigBundle(
            app: try JSONDecoder().decode(AppConfigFile.self, from: appData),
            clickHouse: ClickHouseConfig(
                url: URL(string: "http://localhost:8123")!,
                database: "fxdatabase_test",
                username: nil,
                password: nil,
                requestTimeoutSeconds: 10,
                retryCount: 0
            ),
            mt5Bridge: MT5BridgeConfig(
                mode: .listen,
                host: "127.0.0.1",
                port: 5055,
                connectTimeoutSeconds: 10,
                requestTimeoutSeconds: 10
            ),
            brokerTime: BrokerTimeConfig(
                brokerSourceId: try BrokerSourceId("configured-broker"),
                offsetSegments: []
            ),
            symbols: SymbolConfig(symbols: [])
        )
    }
}

private struct MockHistoryProvider: FXBacktestHistoryProviding {
    func loadM1History(_ request: FXBacktestM1HistoryRequest) async throws -> FXBacktestM1HistoryResponse {
        FXBacktestM1HistoryResponse(
            metadata: FXBacktestM1HistoryMetadata(
                brokerSourceId: request.brokerSourceId,
                sourceOrigin: request.sourceOrigin,
                logicalSymbol: request.logicalSymbol,
                mt5Symbol: request.expectedMT5Symbol ?? request.logicalSymbol,
                digits: request.expectedDigits ?? 5,
                requestedUtcStart: request.utcStartInclusive,
                requestedUtcEndExclusive: request.utcEndExclusive,
                firstUtc: 1_704_067_200,
                lastUtc: 1_704_067_260,
                rowCount: 2
            ),
            utcTimestamps: [1_704_067_200, 1_704_067_260],
            open: [108_000, 108_010],
            high: [108_020, 108_030],
            low: [107_990, 108_000],
            close: [108_010, 108_020],
            volume: [0, 0]
        )
    }
}

private struct InvalidRequestProvider: FXBacktestHistoryProviding {
    func loadM1History(_ request: FXBacktestM1HistoryRequest) async throws -> FXBacktestM1HistoryResponse {
        throw FXBacktestAPIServiceError.invalidRequest("Invalid logical symbol.")
    }
}

private actor MockResultProvider: FXBacktestResultProviding {
    private var recordedOperations: [String] = []

    func ensureResultSchema(_ request: FXBacktestResultSchemaRequest) async throws -> FXBacktestResultMutationResponse {
        recordedOperations.append("schema")
        return FXBacktestResultMutationResponse(sqlStatements: 2)
    }

    func startRun(_ request: FXBacktestResultRunStartRequest) async throws -> FXBacktestResultMutationResponse {
        recordedOperations.append("start:\(request.runId)")
        return FXBacktestResultMutationResponse(runId: request.runId, affectedRows: 1, sqlStatements: 1)
    }

    func appendPassResults(_ request: FXBacktestResultPassAppendRequest) async throws -> FXBacktestResultMutationResponse {
        recordedOperations.append("append:\(request.runId):\(request.results.count)")
        return FXBacktestResultMutationResponse(runId: request.runId, affectedRows: request.results.count, sqlStatements: 1)
    }

    func completeRun(_ request: FXBacktestResultRunCompleteRequest) async throws -> FXBacktestResultMutationResponse {
        recordedOperations.append("complete:\(request.runId):\(request.completedPasses)")
        return FXBacktestResultMutationResponse(runId: request.runId, affectedRows: 1, sqlStatements: 1)
    }

    func purgeResults(_ request: FXBacktestResultPurgeRequest) async throws -> FXBacktestResultPurgeResponse {
        let scope = request.all ? "all" : "older_than_days:\(request.olderThanDays ?? 0)"
        recordedOperations.append("purge:\(scope)")
        return FXBacktestResultPurgeResponse(report: FXBacktestResultPurgeReport(scope: scope, sqlStatements: 2))
    }

    func getRun(_ request: FXBacktestResultRunGetRequest) async throws -> FXBacktestResultRunGetResponse {
        recordedOperations.append("get-run:\(request.runId)")
        return FXBacktestResultRunGetResponse(run: nil)
    }

    func getPasses(_ request: FXBacktestResultPassesGetRequest) async throws -> FXBacktestResultPassesGetResponse {
        recordedOperations.append("get-passes:\(request.runId):\(request.offset):\(request.limit)")
        return FXBacktestResultPassesGetResponse(runId: request.runId, offset: request.offset, limit: request.limit, results: [])
    }

    func operations() -> [String] {
        recordedOperations
    }
}

private actor RecordingResultClickHouse: ClickHouseClientProtocol {
    private var recordedSQL: [String] = []

    func execute(_ query: ClickHouseQuery) async throws -> String {
        recordedSQL.append(query.sql)
        return ""
    }

    func sql() -> [String] {
        recordedSQL
    }
}

private actor NeverCalledClickHouse: ClickHouseClientProtocol {
    private var count = 0

    func execute(_: ClickHouseQuery) async throws -> String {
        count += 1
        return ""
    }

    func executeCount() -> Int {
        count
    }
}
