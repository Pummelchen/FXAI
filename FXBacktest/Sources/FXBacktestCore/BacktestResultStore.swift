import Foundation
import FXBacktestAPI

public struct BacktestStoredRun: Codable, Hashable, Sendable {
    public let runID: String
    public let pluginIdentifier: String
    public let engine: BacktestExecutionTarget
    public let brokerSourceId: String
    public let primarySymbol: String
    public let symbols: [String]
    public let settings: BacktestRunSettings
    public let sweep: ParameterSweep
    public let note: String?

    public init(
        runID: String = UUID().uuidString,
        pluginIdentifier: String,
        engine: BacktestExecutionTarget,
        brokerSourceId: String,
        primarySymbol: String,
        symbols: [String],
        settings: BacktestRunSettings,
        sweep: ParameterSweep,
        note: String? = nil
    ) {
        self.runID = runID
        self.pluginIdentifier = pluginIdentifier
        self.engine = engine
        self.brokerSourceId = brokerSourceId
        self.primarySymbol = primarySymbol.uppercased()
        self.symbols = symbols.map { $0.uppercased() }.sorted()
        self.settings = settings
        self.sweep = sweep
        self.note = note
    }
}

public struct BacktestResultPurgeReport: Codable, Hashable, Sendable {
    public let scope: String
    public let sqlStatements: Int

    public init(scope: String, sqlStatements: Int) {
        self.scope = scope
        self.sqlStatements = sqlStatements
    }
}

public protocol BacktestResultStore: Sendable {
    func ensureSchema() async throws
    func startRun(_ run: BacktestStoredRun) async throws
    func appendResults(_ results: [BacktestPassResult], runID: String) async throws
    func completeRun(runID: String, progress: BacktestProgress, status: String) async throws
    func purgeAll() async throws -> BacktestResultPurgeReport
    func purge(olderThanDays days: Int) async throws -> BacktestResultPurgeReport
}

public protocol FXDatabaseBacktestResultAPIClient: Sendable {
    func ensureBacktestResultSchema() async throws -> FXBacktestResultMutationResponse
    func startBacktestRun(_ run: FXBacktestResultRunStartRequest) async throws -> FXBacktestResultMutationResponse
    func appendBacktestResults(_ results: FXBacktestResultPassAppendRequest) async throws -> FXBacktestResultMutationResponse
    func completeBacktestRun(_ completion: FXBacktestResultRunCompleteRequest) async throws -> FXBacktestResultMutationResponse
    func purgeBacktestResults(_ purge: FXBacktestResultPurgeRequest) async throws -> FXBacktestResultPurgeResponse
    func getBacktestRun(_ run: FXBacktestResultRunGetRequest) async throws -> FXBacktestResultRunGetResponse
    func getBacktestPasses(_ passes: FXBacktestResultPassesGetRequest) async throws -> FXBacktestResultPassesGetResponse
}

extension FXBacktestAPIClient: FXDatabaseBacktestResultAPIClient {}

public struct FXDatabaseBacktestResultStore: BacktestResultStore {
    public let connection: FXDatabaseConnectionSettings
    private let client: any FXDatabaseBacktestResultAPIClient

    public init(
        connection: FXDatabaseConnectionSettings = FXDatabaseConnectionSettings(),
        client: (any FXDatabaseBacktestResultAPIClient)? = nil
    ) {
        self.connection = connection
        self.client = client ?? FXBacktestAPIClient(
            baseURL: connection.apiBaseURL,
            requestTimeoutSeconds: connection.requestTimeoutSeconds
        )
    }

    public func ensureSchema() async throws {
        _ = try await client.ensureBacktestResultSchema()
    }

    public func startRun(_ run: BacktestStoredRun) async throws {
        _ = try await client.startBacktestRun(FXBacktestResultRunStartRequest(
            runId: run.runID,
            pluginId: run.pluginIdentifier,
            engine: run.engine.rawValue,
            brokerSourceId: run.brokerSourceId,
            primarySymbol: run.primarySymbol,
            symbols: run.symbols,
            settingsJSON: try Self.jsonString(run.settings),
            parameterSpaceJSON: try Self.jsonString(run.sweep),
            totalPasses: run.sweep.combinationCount,
            note: run.note
        ))
    }

    public func appendResults(_ results: [BacktestPassResult], runID: String) async throws {
        guard !results.isEmpty else { return }
        let rows = try results.map { result in
            FXBacktestResultPassDTO(
                passIndex: result.passIndex,
                pluginId: result.pluginIdentifier,
                engine: result.engine.rawValue,
                netProfit: result.netProfit,
                grossProfit: result.grossProfit,
                grossLoss: result.grossLoss,
                maxDrawdown: result.maxDrawdown,
                totalTrades: Self.uint32Clamped(result.totalTrades),
                winningTrades: Self.uint32Clamped(result.winningTrades),
                losingTrades: Self.uint32Clamped(result.losingTrades),
                winRate: result.winRate,
                profitFactor: result.profitFactor.isFinite ? result.profitFactor : 0,
                barsProcessed: Self.uint32Clamped(result.barsProcessed),
                flags: result.flags,
                errorMessage: result.errorMessage,
                parametersJSON: try Self.jsonString(result.parameters)
            )
        }
        _ = try await client.appendBacktestResults(FXBacktestResultPassAppendRequest(runId: runID, results: rows))
    }

    public func completeRun(runID: String, progress: BacktestProgress, status: String = "completed") async throws {
        let safeStatus = status.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "completed" : status
        _ = try await client.completeBacktestRun(FXBacktestResultRunCompleteRequest(
            runId: runID,
            completedPasses: progress.completedPasses,
            elapsedSeconds: progress.elapsedSeconds,
            status: safeStatus
        ))
    }

    public func purgeAll() async throws -> BacktestResultPurgeReport {
        let response = try await client.purgeBacktestResults(FXBacktestResultPurgeRequest(all: true))
        return BacktestResultPurgeReport(scope: response.report.scope, sqlStatements: response.report.sqlStatements)
    }

    public func purge(olderThanDays days: Int) async throws -> BacktestResultPurgeReport {
        guard days > 0 else {
            throw FXBacktestError.invalidParameter("Purge age must be > 0 days.")
        }
        let response = try await client.purgeBacktestResults(FXBacktestResultPurgeRequest(olderThanDays: days))
        return BacktestResultPurgeReport(scope: response.report.scope, sqlStatements: response.report.sqlStatements)
    }

    public func fetchRun(runID: String) async throws -> FXBacktestResultRunRecord? {
        try await client.getBacktestRun(FXBacktestResultRunGetRequest(runId: runID)).run
    }

    public func fetchResults(runID: String, offset: Int = 0, limit: Int = 1_000) async throws -> [FXBacktestResultPassDTO] {
        try await client.getBacktestPasses(FXBacktestResultPassesGetRequest(runId: runID, offset: offset, limit: limit)).results
    }

    private static func jsonString<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(data: try encoder.encode(value), encoding: .utf8) ?? "{}"
    }

    private static func uint32Clamped(_ value: Int) -> UInt32 {
        guard value > 0 else { return 0 }
        if value >= Int(UInt32.max) {
            return UInt32.max
        }
        return UInt32(value)
    }
}
