import AppCore
import ClickHouse
import Config
import Domain
import TimeMapping
@testable import Operations
import XCTest

final class OperationsConfigTests: XCTestCase {
    func testSupervisorConfigDefaultsWhenOmitted() throws {
        let data = """
        {
          "chunk_size": 50000,
          "live_scan_interval_seconds": 10,
          "log_level": "normal",
          "strict_symbol_failures": false,
          "verifier_random_ranges": 3
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(AppConfigFile.self, from: data)

        XCTAssertEqual(config.supervisor, .default)
        XCTAssertEqual(config.logging, .default)
    }

    func testSupervisorConfigDefaultsNewAlertThresholdsWhenObjectIsPartial() throws {
        let data = """
        {
          "chunk_size": 50000,
          "live_scan_interval_seconds": 10,
          "log_level": "normal",
          "strict_symbol_failures": false,
          "verifier_random_ranges": 3,
          "supervisor": {
            "cycle_seconds": 15
          }
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(AppConfigFile.self, from: data)

        XCTAssertEqual(config.supervisor.cycleSeconds, 15)
        XCTAssertEqual(config.supervisor.mt5BridgeDownAlertSeconds, SupervisorConfig.default.mt5BridgeDownAlertSeconds)
        XCTAssertEqual(config.supervisor.minimumFreeDiskBytes, SupervisorConfig.default.minimumFreeDiskBytes)
        XCTAssertEqual(config.supervisor.clickHouseDiskFreeAlertBytes, SupervisorConfig.default.clickHouseDiskFreeAlertBytes)
    }

    func testConfigLoaderAllowsEmptyLogPathsWhenFileLoggingIsDisabled() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FXDatabase-config-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeConfig("""
        {
          "chunk_size": 50000,
          "live_scan_interval_seconds": 10,
          "log_level": "normal",
          "strict_symbol_failures": false,
          "verifier_random_ranges": 0,
          "logging": {
            "file_logging_enabled": false,
            "log_file_path": "",
            "alert_file_path": "",
            "max_file_bytes": 0,
            "max_rotated_files": 0
          }
        }
        """, name: "app.json", directory: directory)
        try writeConfig("""
        {
          "url": "http://localhost:8123",
          "database": "db",
          "username": null,
          "password": null,
          "requestTimeoutSeconds": 10,
          "retryCount": 0
        }
        """, name: "clickhouse.json", directory: directory)
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

        let bundle = try ConfigLoader().loadBundle(configDirectory: directory)

        XCTAssertFalse(bundle.app.logging.fileLoggingEnabled)
        XCTAssertTrue(bundle.brokerTime.isAutomatic)
        XCTAssertEqual(bundle.brokerTime.brokerSourceId.rawValue, "auto")
    }

    func testConfigLoaderRejectsInsecureRemoteClickHouseHTTPByDefault() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FXDatabase-remote-http-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeMinimalConfigFiles(
            directory: directory,
            clickHouseJSON: """
            {
              "url": "http://clickhouse.example.com:8123",
              "database": "db",
              "username": "default",
              "password": null,
              "requestTimeoutSeconds": 10,
              "retryCount": 0
            }
            """
        )

        XCTAssertThrowsError(try ConfigLoader().loadBundle(configDirectory: directory)) { error in
            XCTAssertTrue(String(describing: error).contains("Remote ClickHouse endpoints must use https"))
        }
    }

    func testConfigLoaderAllowsExplicitPrivateTunnelHTTPAndDefaultsRemoteSafetyFields() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FXDatabase-private-http-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeMinimalConfigFiles(
            directory: directory,
            clickHouseJSON: """
            {
              "url": "http://clickhouse.example.com:8123",
              "database": "db",
              "username": "default",
              "password": null,
              "requestTimeoutSeconds": 10,
              "retryCount": 0,
              "allowInsecureRemoteHTTP": true
            }
            """
        )

        let bundle = try ConfigLoader().loadBundle(configDirectory: directory)
        XCTAssertTrue(bundle.clickHouse.allowInsecureRemoteHTTP)
        XCTAssertTrue(bundle.clickHouse.waitEndOfQuery)
        XCTAssertEqual(bundle.clickHouse.queryIdPrefix, "fxdatabase")
    }

    func testConfigLoaderRejectsCredentialsEmbeddedInClickHouseURL() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FXDatabase-url-credential-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeMinimalConfigFiles(
            directory: directory,
            clickHouseJSON: """
            {
              "url": "http://default:secret@localhost:8123",
              "database": "db",
              "username": null,
              "password": null,
              "requestTimeoutSeconds": 10,
              "retryCount": 0
            }
            """
        )

        XCTAssertThrowsError(try ConfigLoader().loadBundle(configDirectory: directory)) { error in
            XCTAssertTrue(String(describing: error).contains("credentials must be configured with username/password fields"))
        }
    }

    func testConfigLoaderRejectsNonASCIIClickHouseQueryIdPrefix() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FXDatabase-query-id-prefix-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeMinimalConfigFiles(
            directory: directory,
            clickHouseJSON: """
            {
              "url": "http://localhost:8123",
              "database": "db",
              "username": "default",
              "password": null,
              "requestTimeoutSeconds": 10,
              "retryCount": 0,
              "queryIdPrefix": "fxdatabaseå"
            }
            """
        )

        XCTAssertThrowsError(try ConfigLoader().loadBundle(configDirectory: directory)) { error in
            XCTAssertTrue(String(describing: error).contains("queryIdPrefix may contain only ASCII"))
        }
    }

    func testOperationalHealthServiceScopesCountersToConfiguredBroker() async throws {
        let clickHouse = HealthRecordingClickHouse()
        let config = try makeConfig()
        let service = OperationalHealthService(config: config, clickHouse: clickHouse)

        let snapshot = await service.snapshot()
        let queries = await clickHouse.queries

        XCTAssertTrue(snapshot.clickHouseOk)
        XCTAssertEqual(snapshot.brokerSourceCount, 1)
        XCTAssertEqual(snapshot.canonicalRows, 10)
        XCTAssertEqual(snapshot.latestCanonicalUtc, 180)
        XCTAssertEqual(snapshot.unfinishedIngestOperations, 0)
        XCTAssertEqual(snapshot.warningAgentCount, 0)
        XCTAssertEqual(snapshot.failedAgentCount, 0)
        XCTAssertEqual(snapshot.validDataCertificateCount, 2)

        let brokerScopedTables = [
            "broker_sources",
            "ohlc_m1_canonical",
            "ingest_operations",
            "runtime_agent_state",
            "data_certificates"
        ]
        for table in brokerScopedTables {
            let tableQueries = queries.filter { $0.contains("FROM db.\(table)") }
            XCTAssertFalse(tableQueries.isEmpty, "Expected health query for \(table)")
            for query in tableQueries {
                XCTAssertTrue(query.contains("broker_source_id = 'demo'"), query)
            }
        }
    }

    func testOperationalHealthServiceClassifiesInvalidScalarAsSchemaError() async throws {
        let service = try OperationalHealthService(
            config: makeConfig(),
            clickHouse: HealthInvalidScalarClickHouse()
        )

        let snapshot = await service.snapshot()

        XCTAssertFalse(snapshot.clickHouseOk)
        XCTAssertEqual(snapshot.status, "clickhouse_schema_error")
        XCTAssertEqual(snapshot.canonicalRows, 0)
        XCTAssertNil(snapshot.latestCanonicalUtc)
    }

    func testOperationalHealthServiceClassifiesTransportFailureAsUnavailable() async throws {
        let service = try OperationalHealthService(
            config: makeConfig(),
            clickHouse: HealthTransportFailureClickHouse()
        )

        let snapshot = await service.snapshot()

        XCTAssertFalse(snapshot.clickHouseOk)
        XCTAssertEqual(snapshot.status, "clickhouse_unavailable")
    }

    func testOperationalHealthServiceRejectsUnsafeDatabaseIdentifier() async throws {
        let service = try OperationalHealthService(
            config: makeConfig(database: "db;DROP"),
            clickHouse: HealthRecordingClickHouse()
        )

        let snapshot = await service.snapshot()

        XCTAssertFalse(snapshot.clickHouseOk)
        XCTAssertEqual(snapshot.status, "clickhouse_schema_error")
    }

    func testPersistentLogSinkWritesJSONAndRotates() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FXDatabase-log-test-\(UUID().uuidString)", isDirectory: true)
        let url = directory.appendingPathComponent("FXDatabase.log")
        let sink = try PersistentLogSink(fileURL: url, maxFileBytes: 120, maxRotatedFiles: 1)
        let logger = Logger(level: .normal, persistentLogSink: sink)

        logger.info("persistent log smoke test")
        let first = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(first.contains("\"level\":\"info\""))
        XCTAssertTrue(first.contains("persistent log smoke test"))

        for index in 0..<20 {
            sink.write(level: "debug", component: "test", message: "rotation payload \(index)")
        }

        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: "\(url.path).1"))
    }

    func testTerminalColorPolicyUsesBlackBackgroundWhenEnabled() {
        let policy = TerminalColorPolicy(environment: [:], stdoutIsTTY: true)
        let colored = policy.colorize("line", as: .cyan)

        XCTAssertTrue(colored.hasPrefix("\u{001B}[40m\u{001B}[36m"))
        XCTAssertTrue(colored.hasSuffix("\u{001B}[39m"))
    }

    func testTerminalColorPolicyRespectsNoColor() {
        let policy = TerminalColorPolicy(environment: ["NO_COLOR": "1"], stdoutIsTTY: true)

        XCTAssertEqual(policy.colorize("line", as: .cyan), "line")
    }

    func testCommandLineTokenizerHandlesQuotedInteractiveCommands() throws {
        let tokens = try CommandLineTokenizer().tokenize("repair --symbol EURUSD --from '2020-01-01' --to \"2020-02-01\"")
        XCTAssertEqual(tokens, ["repair", "--symbol", "EURUSD", "--from", "2020-01-01", "--to", "2020-02-01"])
    }

    func testCommandLineTokenizerRejectsUnterminatedQuote() {
        XCTAssertThrowsError(try CommandLineTokenizer().tokenize("backfill --symbols 'EURUSD,USDJPY")) { error in
            XCTAssertEqual(error as? CommandLineTokenizerError, .unterminatedQuote("'"))
        }
    }

    func testProductionAgentsHaveNonRedTerminalStatusColorsAndDisplayNames() {
        for kind in ProductionAgentKind.allCases {
            XCTAssertFalse(kind.displayName.isEmpty)
            XCTAssertNotEqual(kind.terminalColor, .red)
        }
        XCTAssertEqual(ProductionAgentKind.liveM1Updater.displayName, "M1 Updater")
        XCTAssertEqual(ProductionAgentKind.databaseVerifierRepairer.displayName, "Database Cleaner")
        XCTAssertEqual(ProductionAgentKind.utcTimeAuthority.startMessage, "Checking broker UTC offset authority")
    }

    func testOperatorStatusTextFormatsHumanMonthRanges() {
        XCTAssertEqual(
            OperatorStatusText.monthRangeLabel(
                startEpochSeconds: 1_330_560_000,
                endExclusiveEpochSeconds: 1_333_238_400
            ),
            "March 2012"
        )
        XCTAssertEqual(
            OperatorStatusText.monthRangeLabel(
                startEpochSeconds: 1_333_238_400,
                endExclusiveEpochSeconds: 1_338_508_800
            ),
            "April 2012-May 2012"
        )
    }
}
