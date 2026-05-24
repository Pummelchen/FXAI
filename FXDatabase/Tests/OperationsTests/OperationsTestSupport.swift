import AppCore
import BacktestCore
import ClickHouse
import Config
import Domain
import TimeMapping
@testable import Operations
import XCTest

struct StubAgent: ProductionAgent {
    let descriptor: AgentDescriptor

    func run(context: AgentRuntimeContext, startedAt: Date) async throws -> AgentOutcome {
        AgentOutcomeFactory(kind: descriptor.kind, startedAt: startedAt).ok("ok")
    }
}

actor RecordingClickHouse: ClickHouseClientProtocol {
    private var selectBodies: [String]
    private(set) var queries: [ClickHouseQuery] = []

    init(selectBodies: [String]) {
        self.selectBodies = selectBodies
    }

    func execute(_ query: ClickHouseQuery) async throws -> String {
        queries.append(query)
        if query.sql.contains("SELECT last_ok_at_utc") {
            return selectBodies.isEmpty ? "" : selectBodies.removeFirst()
        }
        return ""
    }
}

actor FixedClickHouse: ClickHouseClientProtocol {
    private let body: String

    init(body: String) {
        self.body = body
    }

    func execute(_ query: ClickHouseQuery) async throws -> String {
        body
    }
}

actor HealthRecordingClickHouse: ClickHouseClientProtocol {
    private(set) var queries: [String] = []

    func execute(_ query: ClickHouseQuery) async throws -> String {
        queries.append(query.sql)
        let sql = query.sql
        if sql.trimmingCharacters(in: .whitespacesAndNewlines) == "SELECT 1" {
            return "1\n"
        }
        if sql.contains("FROM db.broker_sources") {
            return "1\n"
        }
        if sql.contains("FROM db.ohlc_m1_canonical") && sql.contains("max(ts_utc)") {
            return "180\n"
        }
        if sql.contains("FROM db.ohlc_m1_canonical") {
            return "10\n"
        }
        if sql.contains("FROM db.ingest_operations") {
            return "0\n"
        }
        if sql.contains("FROM db.runtime_agent_state") {
            return "0\n"
        }
        if sql.contains("FROM db.data_certificates") {
            return "2\n"
        }
        return "0\n"
    }
}

actor VerificationPlannerClickHouse: ClickHouseClientProtocol {
    func execute(_ query: ClickHouseQuery) async throws -> String {
        let sql = query.sql
        if sql.contains("ohlc_m1_canonical") {
            return "1\t60\t120\n"
        }
        if sql.contains("ohlc_m1_verified_coverage") {
            return sql.contains("logical_symbol = 'EURUSD'") ? "60\t120\n" : ""
        }
        if sql.contains("verification_results") {
            return sql.contains("logical_symbol = 'EURUSD'") ? "1\n" : "0\n"
        }
        return "0\n"
    }
}

actor OverlappingVerificationPlannerClickHouse: ClickHouseClientProtocol {
    func execute(_ query: ClickHouseQuery) async throws -> String {
        let sql = query.sql
        if sql.contains("ohlc_m1_canonical") {
            return "2\t60\t120\n"
        }
        if sql.contains("ohlc_m1_verified_coverage") {
            return "60\t120\n60\t180\n"
        }
        if sql.contains("verification_results") {
            return "1\n"
        }
        return "0\n"
    }
}

enum BackupRestoreMode {
    case clean
    case unfinished
}

actor BackupRestoreClickHouse: ClickHouseClientProtocol {
    private let mode: BackupRestoreMode

    init(mode: BackupRestoreMode) {
        self.mode = mode
    }

    func execute(_ query: ClickHouseQuery) async throws -> String {
        let sql = query.sql
        if sql.contains("FROM db.data_certificates") && sql.contains("sum(coverage_source_bar_count)") {
            return "1\t10\t10\t60\t120\n"
        }
        if sql.contains("FROM db.data_certificates") {
            return "0\n"
        }
        if sql.contains("FROM db.ingest_operations") {
            return mode == .unfinished ? "1\n" : "0\n"
        }
        return "0\n"
    }
}

enum BacktestGateMode {
    case clean
    case interruptedBackfill
    case failedAgentState
    case missingRequiredAgentState
    case unfinishedIngestOperation
    case missingVerifiedCoverage
    case missingDataCertificate
}

enum CheckpointAuditMode {
    case missingUSDJPYCheckpoint
    case interruptedEURUSD
}

actor BacktestGateClickHouse: ClickHouseClientProtocol {
    private let mode: BacktestGateMode

    init(mode: BacktestGateMode) {
        self.mode = mode
    }

    func execute(_ query: ClickHouseQuery) async throws -> String {
        let sql = query.sql
        if sql.contains("FROM db.ingest_state") {
            let status = mode == .interruptedBackfill && sql.contains("logical_symbol = 'EURUSD'") ? "backfilling" : "live"
            if sql.contains("logical_symbol = 'EURUSD'") {
                return "demo\tMT5\tEURUSD\tEURUSD\t0\t180\t180\t\(status)\tbatch\t200\n"
            }
            if sql.contains("logical_symbol = 'USDJPY'") {
                return "demo\tMT5\tUSDJPY\tUSDJPY\t0\t180\t180\tlive\tbatch\t200\n"
            }
            return ""
        }
        if sql.contains("runtime_agent_state") {
            if sql.contains("status IN") {
                return mode == .failedAgentState ? "database_verifier_repairer\tfailed\tverification mismatch\n" : ""
            }
            guard mode != .missingRequiredAgentState else { return "" }
            let now = Int64(Date().timeIntervalSince1970)
            return """
            schema_drift_guard\t\(now)
            bridge_version_guard\t\(now)
            utc_time_authority\t\(now)
            symbol_metadata_drift\t\(now)
            source_history_drift\t\(now)
            live_m1_updater\t\(now)
            database_verifier_repairer\t\(now)
            verification_coverage_planner\t\(now)
            checkpoint_gap_auditor\t\(now)
            data_certification\t\(now)

            """
        }
        if sql.contains("FROM db.ingest_operations") {
            return mode == .unfinishedIngestOperation ? "1\n" : "0\n"
        }
        if sql.contains("FROM db.ohlc_m1_verified_coverage") {
            return mode == .missingVerifiedCoverage ? "" : "60\t120\n"
        }
        if sql.contains("FROM db.data_certificates") {
            return mode == .missingDataCertificate ? "" : "60\t120\n"
        }
        if sql.contains("ohlc_m1_canonical") && sql.contains("ts_utc >= 60") && sql.contains("ts_utc < 120") {
            return "10\n"
        }
        return "0\n"
    }
}

actor CheckpointAuditClickHouse: ClickHouseClientProtocol {
    private let mode: CheckpointAuditMode

    init(mode: CheckpointAuditMode) {
        self.mode = mode
    }

    func execute(_ query: ClickHouseQuery) async throws -> String {
        let sql = query.sql
        if sql.contains("FROM db.ingest_state") {
            if sql.contains("logical_symbol = 'EURUSD'") {
                let status = mode == .interruptedEURUSD ? "backfilling" : "live"
                return "demo\tMT5\tEURUSD\tEURUSD\t0\t180\t180\t\(status)\tbatch\t200\n"
            }
            if sql.contains("logical_symbol = 'USDJPY'") {
                guard mode != .missingUSDJPYCheckpoint else { return "" }
                return "demo\tMT5\tUSDJPY\tUSDJPY\t0\t180\t180\tlive\tbatch\t200\n"
            }
            return ""
        }
        if sql.contains("ohlc_m1_canonical") {
            return "1\n"
        }
        if sql.contains("ohlc_m1_verified_coverage") {
            return "1\n"
        }
        return "0\n"
    }
}

actor BackupReadinessClickHouse: ClickHouseClientProtocol {
    private(set) var queries: [ClickHouseQuery] = []

    func execute(_ query: ClickHouseQuery) async throws -> String {
        queries.append(query)
        return "1\t60\t120\n"
    }
}

actor DataCertificateClickHouse: ClickHouseClientProtocol {
    private let rows: [String]
    private(set) var queries: [ClickHouseQuery] = []

    init(rows: [String]) {
        self.rows = rows
    }

    func execute(_ query: ClickHouseQuery) async throws -> String {
        queries.append(query)
        if query.sql.contains("FROM db.ohlc_m1_verified_coverage") {
            return rows.joined(separator: "\n") + "\n"
        }
        return ""
    }
}

func coverageRow(
    utcStart: Int64,
    utcEnd: Int64,
    sourceCount: UInt32 = 1,
    canonicalCount: UInt32 = 1
) -> String {
    let hashA = String(repeating: "a", count: 64)
    let hashB = String(repeating: "b", count: 64)
    let hashC = String(repeating: "c", count: 64)
    return [
        "EURUSD",
        "M1",
        String(utcStart),
        String(utcEnd),
        String(utcStart),
        String(utcEnd),
        String(sourceCount),
        String(canonicalCount),
        hashA,
        hashB,
        hashC,
        "batch-\(utcStart)",
        "1000",
        "MT5"
    ].joined(separator: "\t")
}

actor AlertingClickHouse: ClickHouseClientProtocol {
    private let now: Int64

    init(now: Int64) {
        self.now = now
    }

    func execute(_ query: ClickHouseQuery) async throws -> String {
        let sql = query.sql
        if sql.contains("runtime_agent_events") {
            return "health_monitor\twarning\tClickHouse healthy, MT5 bridge is not connected\t\(now - 30)\n"
        }
        if sql.contains("runtime_agent_state FINAL") {
            return """
            utc_time_authority\tfailed\tBroker UTC offset mismatch\t0\t\(now - 300)\t\(now - 300)
            symbol_metadata_drift\tok\tok\t\(now)\t0\t\(now)
            live_m1_updater\tok\tok\t\(now)\t0\t\(now)
            database_verifier_repairer\tok\tok\t\(now)\t0\t\(now)
            checkpoint_gap_auditor\tok\tok\t\(now)\t0\t\(now)

            """
        }
        if sql.contains("system.disks") {
            return "default\t/var/lib/clickhouse\t0\t100\n"
        }
        if sql.contains("verification_results") {
            return "2\n"
        }
        return "0\n"
    }
}

actor StartupClickHouse: ClickHouseClientProtocol {
    private var failuresBeforeSuccess: Int

    init(failuresBeforeSuccess: Int) {
        self.failuresBeforeSuccess = failuresBeforeSuccess
    }

    func execute(_ query: ClickHouseQuery) async throws -> String {
        if failuresBeforeSuccess > 0 {
            failuresBeforeSuccess -= 1
            throw ClickHouseError.transport("connection refused")
        }
        return "1\n"
    }
}

struct StoredSineCanonicalRow: Sendable, Equatable {
    let sourceOrigin: String
    let mt5Symbol: String
    let timeframe: String
    let mt5: Int64
    let utc: Int64
    let offset: Int64
    let offsetSource: String
    let offsetConfidence: String
    let open: Int64
    let high: Int64
    let low: Int64
    let close: Int64
    let volume: UInt64
    let digits: Int
    let hash: String
}

actor SineSyncClickHouse: ClickHouseClientProtocol {
    private var rows: [StoredSineCanonicalRow] = []
    private var coverageRows: [[String]] = []
    private var canonicalInserts = 0
    private var canonicalDeletes = 0
    private var stateInserts = 0

    func execute(_ query: ClickHouseQuery) async throws -> String {
        let sql = query.sql
        if sql.contains("SELECT utc_range_start, utc_range_end_exclusive"),
           sql.contains("FROM db.ohlc_m1_verified_coverage") {
            return coverageRows
                .map { "\($0[7])\t\($0[8])" }
                .joined(separator: "\n")
        }
        if sql.contains("SELECT 1"),
           sql.contains("FROM db.ohlc_m1_canonical") {
            return rowsByUTC(in: sql).isEmpty ? "" : "1\n"
        }
        if sql.contains("ALTER TABLE db.ohlc_m1_canonical DELETE") {
            let start = Self.integer(after: "mt5_server_ts_raw >= ", in: sql) ?? Int64.min
            let end = Self.integer(after: "mt5_server_ts_raw < ", in: sql) ?? Int64.max
            rows.removeAll { $0.mt5 >= start && $0.mt5 < end }
            canonicalDeletes += 1
            return ""
        }
        if sql.contains("INSERT INTO db.ohlc_m1_canonical") {
            rows.append(contentsOf: try Self.parseCanonicalRows(sql))
            rows.sort { $0.mt5 < $1.mt5 }
            canonicalInserts += 1
            return ""
        }
        if sql.contains("uniqExact(ts_utc)"),
           sql.contains("ts_utc IN"),
           sql.contains("countIf("),
           sql.contains("source_status"),
           sql.contains("FROM db.ohlc_m1_canonical") {
            let selected = rowsByBoundary(in: sql)
            let values: [Int64] = [
                Int64(selected.count),
                Int64(Set(selected.map(\.utc)).count),
                Int64(selected.filter { row in
                    row.mt5Symbol != SineTestSecurity.providerSymbol.rawValue ||
                        row.timeframe != Timeframe.m1.rawValue ||
                        row.mt5 != row.utc ||
                        row.offset != 0 ||
                        row.offsetSource != OffsetSource.configured.rawValue ||
                        row.offsetConfidence != OffsetConfidence.verified.rawValue ||
                        row.digits != SineTestSecurity.digits.rawValue ||
                        row.open <= 0 ||
                        row.high <= 0 ||
                        row.low <= 0 ||
                        row.close <= 0 ||
                        row.volume == 0
                }.count)
            ]
            return values.map { String($0) }.joined(separator: "\t") + "\n"
        }
        if sql.contains("uniqExact(ts_utc)"),
           sql.contains("countIf("),
           sql.contains("source_status"),
           sql.contains("FROM db.ohlc_m1_canonical") {
            let selected = rowsByUTC(in: sql)
            let values: [Int64] = [
                Int64(selected.count),
                Int64(Set(selected.map(\.utc)).count),
                selected.map(\.utc).min() ?? 0,
                selected.map(\.utc).max() ?? 0,
                Int64(selected.filter { row in
                    row.mt5Symbol != SineTestSecurity.providerSymbol.rawValue ||
                        row.timeframe != Timeframe.m1.rawValue ||
                        row.mt5 != row.utc ||
                        row.offset != 0 ||
                        row.offsetSource != OffsetSource.configured.rawValue ||
                        row.offsetConfidence != OffsetConfidence.verified.rawValue ||
                        row.digits != SineTestSecurity.digits.rawValue ||
                        row.open <= 0 ||
                        row.high <= 0 ||
                        row.low <= 0 ||
                        row.close <= 0 ||
                        row.volume == 0
                }.count)
            ]
            return values.map { String($0) }.joined(separator: "\t") + "\n"
        }
        if sql.contains("uniqExact(mt5_server_ts_raw)"),
           sql.contains("countIf(offset_confidence"),
           sql.contains("FROM db.ohlc_m1_canonical") {
            let selected = rows(in: sql)
            let values: [Int] = [
                selected.count,
                Set(selected.map(\.mt5)).count,
                Set(selected.map(\.utc)).count,
                Set(selected.map(\.sourceOrigin)).count,
                Set(selected.map(\.mt5Symbol)).count,
                Set(selected.map(\.timeframe)).count,
                Set(selected.map(\.digits)).count,
                selected.filter { $0.offsetConfidence != OffsetConfidence.verified.rawValue }.count
            ]
            return values.map { String($0) }.joined(separator: "\t") + "\n"
        }
        if sql.contains("SELECT source_origin, mt5_symbol, timeframe, mt5_server_ts_raw, ts_utc"),
           sql.contains("server_utc_offset_seconds"),
           sql.contains("FROM db.ohlc_m1_canonical") {
            return rows(in: sql).map { row in
                [
                    row.sourceOrigin,
                    row.mt5Symbol,
                    row.timeframe,
                    String(row.mt5),
                    String(row.utc),
                    String(row.offset),
                    row.offsetSource,
                    row.offsetConfidence,
                    String(row.open),
                    String(row.high),
                    String(row.low),
                    String(row.close),
                    String(row.volume),
                    String(row.digits),
                    row.hash
                ].joined(separator: "\t")
            }.joined(separator: "\n")
        }
        if sql.contains("INSERT INTO db.ohlc_m1_verified_coverage") {
            coverageRows.append(contentsOf: Self.payloadRows(sql).map {
                $0.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            })
            return ""
        }
        if sql.contains("INSERT INTO db.ingest_state") {
            stateInserts += Self.payloadRows(sql).count
            return ""
        }
        return ""
    }

    func canonicalRows() -> [StoredSineCanonicalRow] {
        rows
    }

    func canonicalInsertCount() -> Int {
        canonicalInserts
    }

    func canonicalDeleteCount() -> Int {
        canonicalDeletes
    }

    func coverageIntervalCount() -> Int {
        coverageRows.count
    }

    func clearCoverageIntervals() {
        coverageRows.removeAll()
    }

    func ingestStateInsertCount() -> Int {
        stateInserts
    }

    private func rows(in sql: String) -> [StoredSineCanonicalRow] {
        let start = Self.integer(after: "mt5_server_ts_raw >= ", in: sql) ?? Int64.min
        let end = Self.integer(after: "mt5_server_ts_raw < ", in: sql) ?? Int64.max
        return rows.filter { $0.mt5 >= start && $0.mt5 < end }.sorted { $0.mt5 < $1.mt5 }
    }

    private func rowsByUTC(in sql: String) -> [StoredSineCanonicalRow] {
        let start = Self.integer(after: "ts_utc >= ", in: sql) ?? Int64.min
        let end = Self.integer(after: "ts_utc < ", in: sql) ?? Int64.max
        return rows.filter { $0.utc >= start && $0.utc < end }.sorted { $0.utc < $1.utc }
    }

    private func rowsByBoundary(in sql: String) -> [StoredSineCanonicalRow] {
        guard let range = sql.range(of: "ts_utc IN (") else { return [] }
        let suffix = sql[range.upperBound...]
        let rawValues = suffix.prefix { $0 != ")" }
        let boundaries = Set(rawValues
            .split(separator: ",", omittingEmptySubsequences: true)
            .compactMap { Int64($0.trimmingCharacters(in: .whitespacesAndNewlines)) })
        return rows.filter { boundaries.contains($0.utc) }.sorted { $0.utc < $1.utc }
    }

    private static func parseCanonicalRows(_ sql: String) throws -> [StoredSineCanonicalRow] {
        try payloadRows(sql).map { row in
            let fields = row.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard fields.count == 20,
                  let mt5 = Int64(fields[5]),
                  let utc = Int64(fields[6]),
                  let offset = Int64(fields[7]),
                  let open = Int64(fields[10]),
                  let high = Int64(fields[11]),
                  let low = Int64(fields[12]),
                  let close = Int64(fields[13]),
                  let volume = UInt64(fields[14]),
                  let digits = Int(fields[15]) else {
                throw ProductionAgentError.invariant("invalid test canonical row: \(row)")
            }
            return StoredSineCanonicalRow(
                sourceOrigin: fields[1],
                mt5Symbol: fields[3],
                timeframe: fields[4],
                mt5: mt5,
                utc: utc,
                offset: offset,
                offsetSource: fields[8],
                offsetConfidence: fields[9],
                open: open,
                high: high,
                low: low,
                close: close,
                volume: volume,
                digits: digits,
                hash: fields[17]
            )
        }
    }

    private static func payloadRows(_ sql: String) -> [String] {
        guard let range = sql.range(of: "FORMAT TabSeparated") else { return [] }
        let payload = sql[range.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !payload.isEmpty else { return [] }
        return payload.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
    }

    private static func integer(after marker: String, in sql: String) -> Int64? {
        guard let range = sql.range(of: marker) else { return nil }
        let suffix = sql[range.upperBound...].drop { $0 == " " || $0 == "\n" || $0 == "\t" }
        let digits = suffix.prefix { $0 == "-" || $0.isNumber }
        return Int64(String(digits))
    }
}

actor RecordingCommandRunner: SystemCommandRunning {
    private var commands: [String] = []
    private let resultExitCode: Int32

    init(resultExitCode: Int32) {
        self.resultExitCode = resultExitCode
    }

    func run(_ request: SystemCommandRequest) async throws -> SystemCommandResult {
        commands.append(request.display)
        return SystemCommandResult(
            request: request,
            exitCode: resultExitCode,
            stdout: "",
            stderr: ""
        )
    }

    func executedCommands() -> [String] {
        commands
    }
}

func makeConfig(
    minimumFreeDiskBytes: Int64 = SupervisorConfig.default.minimumFreeDiskBytes,
    clickHouseDiskFreeAlertBytes: Int64 = SupervisorConfig.default.clickHouseDiskFreeAlertBytes,
    symbols: [String: UInt8] = ["EURUSD": 5, "USDJPY": 3]
) throws -> ConfigBundle {
    let appData = """
    {
      "chunk_size": 50000,
      "live_scan_interval_seconds": 10,
      "log_level": "normal",
      "strict_symbol_failures": false,
      "verifier_random_ranges": 0,
      "supervisor": {
        "minimum_free_disk_bytes": \(minimumFreeDiskBytes),
        "clickhouse_disk_free_alert_bytes": \(clickHouseDiskFreeAlertBytes)
      }
    }
    """.data(using: .utf8)!
    return ConfigBundle(
        app: try JSONDecoder().decode(AppConfigFile.self, from: appData),
        clickHouse: ClickHouseConfig(
            url: URL(string: "http://localhost:8123")!,
            database: "db",
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
            brokerSourceId: try BrokerSourceId("demo"),
            offsetSegments: []
        ),
        symbols: SymbolConfig(symbols: try symbols.keys.sorted().map { symbol in
            SymbolMapping(
                logicalSymbol: try LogicalSymbol(symbol),
                mt5Symbol: try MT5Symbol(symbol),
                digits: try Digits(Int(symbols[symbol] ?? 5))
            )
        })
    )
}

func writeConfig(_ text: String, name: String, directory: URL) throws {
    let data = try XCTUnwrap(text.data(using: .utf8))
    try data.write(to: directory.appendingPathComponent(name))
}

func writeMinimalConfigFiles(directory: URL, clickHouseJSON: String) throws {
    try writeConfig("""
    {
      "chunk_size": 50000,
      "live_scan_interval_seconds": 10,
      "log_level": "normal",
      "strict_symbol_failures": false,
      "verifier_random_ranges": 0
    }
    """, name: "app.json", directory: directory)
    try writeConfig(clickHouseJSON, name: "clickhouse.json", directory: directory)
    try writeConfig("""
    {
      "mode": "listen",
      "host": "127.0.0.1",
      "port": 5055,
      "connectTimeoutSeconds": 10,
      "requestTimeoutSeconds": 10
    }
    """, name: "mt5_bridge.json", directory: directory)
    try writeConfig("""
    {
      "symbols": [
        { "logical_symbol": "EURUSD", "mt5_symbol": "EURUSD", "digits": 5 }
      ]
    }
    """, name: "symbols.json", directory: directory)
}

func XCTAssertThrowsErrorAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    file: StaticString = #filePath,
    line: UInt = #line,
    _ errorHandler: (Error) -> Void = { _ in }
) async {
    do {
        _ = try await expression()
        XCTFail("Expected async expression to throw", file: file, line: line)
    } catch {
        errorHandler(error)
    }
}
