import ClickHouse
import Foundation
import FXBacktestAPI

public struct FXDatabaseBacktestResultService: FXBacktestResultProviding {
    private let clickHouse: ClickHouseClientProtocol
    private let database: String

    public init(clickHouse: ClickHouseClientProtocol, database: String) {
        self.clickHouse = clickHouse
        self.database = database
    }

    public func ensureResultSchema(_ request: FXBacktestResultSchemaRequest) async throws -> FXBacktestResultMutationResponse {
        try request.validate()
        try await ensureSchema()
        return FXBacktestResultMutationResponse(sqlStatements: 3)
    }

    public func startRun(_ request: FXBacktestResultRunStartRequest) async throws -> FXBacktestResultMutationResponse {
        try request.validate()
        try await ensureSchema()
        let symbolsSQL = request.symbols.map(Self.sqlString).joined(separator: ",")
        _ = try await clickHouse.execute(.mutation("""
        INSERT INTO \(table("fxbacktest_runs"))
        (run_id, created_at, completed_at, plugin_id, engine, broker_source_id, primary_symbol, symbols, api_version, settings_json, parameter_space_json, status, completed_passes, total_passes, note)
        VALUES (
          \(Self.sqlString(request.runId)),
          now64(3),
          NULL,
          \(Self.sqlString(request.pluginId)),
          \(Self.sqlString(request.engine)),
          \(Self.sqlString(request.brokerSourceId)),
          \(Self.sqlString(request.primarySymbol.uppercased())),
          [\(symbolsSQL)],
          \(Self.sqlString(FXBacktestAPIV1.latestVersion)),
          \(Self.sqlString(request.settingsJSON)),
          \(Self.sqlString(request.parameterSpaceJSON)),
          'running',
          0,
          \(request.totalPasses),
          \(Self.sqlString(request.note ?? ""))
        )
        """, idempotent: false))
        return FXBacktestResultMutationResponse(runId: request.runId, affectedRows: 1, sqlStatements: 4)
    }

    public func appendPassResults(_ request: FXBacktestResultPassAppendRequest) async throws -> FXBacktestResultMutationResponse {
        try request.validate()
        guard !request.results.isEmpty else {
            return FXBacktestResultMutationResponse(runId: request.runId, affectedRows: 0, sqlStatements: 0)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let rows = request.results.map { result in
            PassResultInsertRow(
                run_id: request.runId,
                pass_index: result.passIndex,
                plugin_id: result.pluginId,
                engine: result.engine,
                net_profit: result.netProfit,
                gross_profit: result.grossProfit,
                gross_loss: result.grossLoss,
                max_drawdown: result.maxDrawdown,
                total_trades: result.totalTrades,
                winning_trades: result.winningTrades,
                losing_trades: result.losingTrades,
                win_rate: result.winRate,
                profit_factor: result.profitFactor,
                bars_processed: result.barsProcessed,
                flags: result.flags,
                error_message: result.errorMessage ?? "",
                parameters_json: result.parametersJSON
            )
        }
        let payload = try rows.map { row in
            guard let encoded = String(data: try encoder.encode(row), encoding: .utf8) else {
                throw FXBacktestAPIServiceError.invalidRequest("Could not encode result row as JSONEachRow.")
            }
            return encoded
        }.joined(separator: "\n")
        _ = try await clickHouse.execute(.mutation("""
        INSERT INTO \(table("fxbacktest_pass_results")) FORMAT JSONEachRow
        \(payload)
        """, idempotent: false))
        return FXBacktestResultMutationResponse(runId: request.runId, affectedRows: request.results.count, sqlStatements: 1)
    }

    public func completeRun(_ request: FXBacktestResultRunCompleteRequest) async throws -> FXBacktestResultMutationResponse {
        try request.validate()
        try await ensureSchema()
        _ = try await clickHouse.execute(.mutation("""
        ALTER TABLE \(table("fxbacktest_runs"))
        UPDATE
          completed_at = now64(3),
          status = \(Self.sqlString(request.status)),
          completed_passes = \(request.completedPasses)
        WHERE run_id = \(Self.sqlString(request.runId))
        """, idempotent: false))
        return FXBacktestResultMutationResponse(runId: request.runId, affectedRows: 1, sqlStatements: 4)
    }

    public func purgeResults(_ request: FXBacktestResultPurgeRequest) async throws -> FXBacktestResultPurgeResponse {
        try request.validate()
        try await ensureSchema()
        if request.all {
            _ = try await clickHouse.execute(.mutation("ALTER TABLE \(table("fxbacktest_pass_results")) DELETE WHERE 1", idempotent: false))
            _ = try await clickHouse.execute(.mutation("ALTER TABLE \(table("fxbacktest_runs")) DELETE WHERE 1", idempotent: false))
            return FXBacktestResultPurgeResponse(report: FXBacktestResultPurgeReport(scope: "all", sqlStatements: 5))
        }
        guard let days = request.olderThanDays, days > 0 else {
            throw FXBacktestAPIServiceError.invalidRequest("older_than_days must be positive unless all is true.")
        }
        _ = try await clickHouse.execute(.mutation("""
        ALTER TABLE \(table("fxbacktest_pass_results"))
        DELETE WHERE inserted_at < now() - INTERVAL \(days) DAY
        """, idempotent: false))
        _ = try await clickHouse.execute(.mutation("""
        ALTER TABLE \(table("fxbacktest_runs"))
        DELETE WHERE created_at < now() - INTERVAL \(days) DAY
        """, idempotent: false))
        return FXBacktestResultPurgeResponse(report: FXBacktestResultPurgeReport(scope: "older-than-\(days)-days", sqlStatements: 5))
    }

    public func getRun(_ request: FXBacktestResultRunGetRequest) async throws -> FXBacktestResultRunGetResponse {
        try request.validate()
        try await ensureSchema()
        let body = try await clickHouse.execute(.select("""
        SELECT
          run_id,
          toString(toUnixTimestamp64Milli(created_at)),
          if(completed_at IS NULL, '', toString(toUnixTimestamp64Milli(assumeNotNull(completed_at)))),
          plugin_id,
          engine,
          broker_source_id,
          primary_symbol,
          arrayStringConcat(symbols, ','),
          api_version,
          settings_json,
          parameter_space_json,
          status,
          toString(completed_passes),
          toString(total_passes),
          note
        FROM \(table("fxbacktest_runs"))
        WHERE run_id = \(Self.sqlString(request.runId))
        ORDER BY created_at DESC
        LIMIT 1
        FORMAT TabSeparated
        """))
        guard let line = body.split(separator: "\n", omittingEmptySubsequences: true).first else {
            return FXBacktestResultRunGetResponse(run: nil)
        }
        return try FXBacktestResultRunGetResponse(run: Self.parseRunRecord(String(line)))
    }

    public func getPasses(_ request: FXBacktestResultPassesGetRequest) async throws -> FXBacktestResultPassesGetResponse {
        try request.validate()
        try await ensureSchema()
        let body = try await clickHouse.execute(.select("""
        SELECT
          toString(pass_index),
          plugin_id,
          engine,
          toString(net_profit),
          toString(gross_profit),
          toString(gross_loss),
          toString(max_drawdown),
          toString(total_trades),
          toString(winning_trades),
          toString(losing_trades),
          toString(win_rate),
          toString(profit_factor),
          toString(bars_processed),
          toString(flags),
          error_message,
          parameters_json
        FROM \(table("fxbacktest_pass_results"))
        WHERE run_id = \(Self.sqlString(request.runId))
        ORDER BY pass_index ASC
        LIMIT \(request.limit) OFFSET \(request.offset)
        FORMAT TabSeparated
        """))
        let results = try body
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { try Self.parsePassDTO(String($0)) }
        return FXBacktestResultPassesGetResponse(
            runId: request.runId,
            offset: request.offset,
            limit: request.limit,
            results: results
        )
    }

    private func ensureSchema() async throws {
        _ = try await clickHouse.execute(.mutation("CREATE DATABASE IF NOT EXISTS \(Self.identifier(database))", idempotent: true))
        _ = try await clickHouse.execute(.mutation(Self.runsTableSQL(database: database), idempotent: true))
        _ = try await clickHouse.execute(.mutation(Self.passResultsTableSQL(database: database), idempotent: true))
    }

    private func table(_ name: String) -> String {
        "\(Self.identifier(database)).\(Self.identifier(name))"
    }

    private static func runsTableSQL(database: String) -> String {
        """
        CREATE TABLE IF NOT EXISTS \(identifier(database)).\(identifier("fxbacktest_runs"))
        (
          run_id String,
          created_at DateTime64(3, 'UTC'),
          completed_at Nullable(DateTime64(3, 'UTC')),
          plugin_id LowCardinality(String),
          engine LowCardinality(String),
          broker_source_id String,
          primary_symbol String,
          symbols Array(String),
          api_version String,
          settings_json String,
          parameter_space_json String,
          status LowCardinality(String),
          completed_passes UInt64,
          total_passes UInt64,
          note String
        )
        ENGINE = MergeTree
        ORDER BY (created_at, run_id)
        """
    }

    private static func passResultsTableSQL(database: String) -> String {
        """
        CREATE TABLE IF NOT EXISTS \(identifier(database)).\(identifier("fxbacktest_pass_results"))
        (
          run_id String,
          pass_index UInt64,
          plugin_id LowCardinality(String),
          engine LowCardinality(String),
          net_profit Float64,
          gross_profit Float64,
          gross_loss Float64,
          max_drawdown Float64,
          total_trades UInt32,
          winning_trades UInt32,
          losing_trades UInt32,
          win_rate Float64,
          profit_factor Float64,
          bars_processed UInt32,
          flags UInt32,
          error_message String,
          parameters_json String,
          inserted_at DateTime64(3, 'UTC') DEFAULT now64(3)
        )
        ENGINE = MergeTree
        ORDER BY (run_id, pass_index)
        """
    }

    private static func identifier(_ value: String) -> String {
        "`\(value.replacingOccurrences(of: "`", with: "``"))`"
    }

    private static func sqlString(_ value: String) -> String {
        var escaped = ""
        escaped.reserveCapacity(value.count)
        for character in value {
            switch character {
            case "\\":
                escaped.append("\\\\")
            case "'":
                escaped.append("\\'")
            case "\n":
                escaped.append("\\n")
            case "\r":
                escaped.append("\\r")
            case "\t":
                escaped.append("\\t")
            case "\0":
                escaped.append("\\0")
            default:
                escaped.append(character)
            }
        }
        return "'\(escaped)'"
    }

    private static func tsvFields(_ line: String) -> [String] {
        line.split(separator: "\t", omittingEmptySubsequences: false).map { unescapeTSV(String($0)) }
    }

    private static func unescapeTSV(_ value: String) -> String {
        var output = ""
        var escaping = false
        for character in value {
            if escaping {
                switch character {
                case "t": output.append("\t")
                case "n": output.append("\n")
                case "r": output.append("\r")
                case "\\": output.append("\\")
                default: output.append(character)
                }
                escaping = false
            } else if character == "\\" {
                escaping = true
            } else {
                output.append(character)
            }
        }
        if escaping {
            output.append("\\")
        }
        return output
    }

    private static func parseRunRecord(_ line: String) throws -> FXBacktestResultRunRecord {
        let fields = tsvFields(line)
        guard fields.count == 15 else {
            throw FXBacktestAPIServiceError.resultStoreUnavailable("Invalid backtest run row with \(fields.count) fields.")
        }
        return FXBacktestResultRunRecord(
            runId: fields[0],
            createdAtUnixMilliseconds: try int64(fields[1], field: "created_at"),
            completedAtUnixMilliseconds: fields[2].isEmpty ? nil : try int64(fields[2], field: "completed_at"),
            pluginId: fields[3],
            engine: fields[4],
            brokerSourceId: fields[5],
            primarySymbol: fields[6],
            symbols: fields[7].isEmpty ? [] : fields[7].split(separator: ",").map(String.init),
            apiVersion: fields[8],
            settingsJSON: fields[9],
            parameterSpaceJSON: fields[10],
            status: fields[11],
            completedPasses: try uint64(fields[12], field: "completed_passes"),
            totalPasses: try uint64(fields[13], field: "total_passes"),
            note: fields[14]
        )
    }

    private static func parsePassDTO(_ line: String) throws -> FXBacktestResultPassDTO {
        let fields = tsvFields(line)
        guard fields.count == 16 else {
            throw FXBacktestAPIServiceError.resultStoreUnavailable("Invalid backtest pass row with \(fields.count) fields.")
        }
        return try FXBacktestResultPassDTO(
            passIndex: uint64(fields[0], field: "pass_index"),
            pluginId: fields[1],
            engine: fields[2],
            netProfit: double(fields[3], field: "net_profit"),
            grossProfit: double(fields[4], field: "gross_profit"),
            grossLoss: double(fields[5], field: "gross_loss"),
            maxDrawdown: double(fields[6], field: "max_drawdown"),
            totalTrades: uint32(fields[7], field: "total_trades"),
            winningTrades: uint32(fields[8], field: "winning_trades"),
            losingTrades: uint32(fields[9], field: "losing_trades"),
            winRate: double(fields[10], field: "win_rate"),
            profitFactor: double(fields[11], field: "profit_factor"),
            barsProcessed: uint32(fields[12], field: "bars_processed"),
            flags: uint32(fields[13], field: "flags"),
            errorMessage: fields[14].isEmpty ? nil : fields[14],
            parametersJSON: fields[15]
        )
    }

    private static func int64(_ value: String, field: String) throws -> Int64 {
        guard let parsed = Int64(value) else {
            throw FXBacktestAPIServiceError.resultStoreUnavailable("Invalid \(field) value \(value).")
        }
        return parsed
    }

    private static func uint64(_ value: String, field: String) throws -> UInt64 {
        guard let parsed = UInt64(value) else {
            throw FXBacktestAPIServiceError.resultStoreUnavailable("Invalid \(field) value \(value).")
        }
        return parsed
    }

    private static func uint32(_ value: String, field: String) throws -> UInt32 {
        guard let parsed = UInt32(value) else {
            throw FXBacktestAPIServiceError.resultStoreUnavailable("Invalid \(field) value \(value).")
        }
        return parsed
    }

    private static func double(_ value: String, field: String) throws -> Double {
        guard let parsed = Double(value), parsed.isFinite else {
            throw FXBacktestAPIServiceError.resultStoreUnavailable("Invalid \(field) value \(value).")
        }
        return parsed
    }
}

private struct PassResultInsertRow: Encodable {
    let run_id: String
    let pass_index: UInt64
    let plugin_id: String
    let engine: String
    let net_profit: Double
    let gross_profit: Double
    let gross_loss: Double
    let max_drawdown: Double
    let total_trades: UInt32
    let winning_trades: UInt32
    let losing_trades: UInt32
    let win_rate: Double
    let profit_factor: Double
    let bars_processed: UInt32
    let flags: UInt32
    let error_message: String
    let parameters_json: String
}
