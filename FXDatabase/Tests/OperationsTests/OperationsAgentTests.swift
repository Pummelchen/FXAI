import AppCore
import BacktestCore
import ClickHouse
import Config
import Domain
import TimeMapping
@testable import Operations
import XCTest

final class OperationsAgentTests: XCTestCase {
    func testAlertingAgentReportsSafetyBlocksAndDiskPressure() async throws {
        let config = try makeConfig(minimumFreeDiskBytes: 1, clickHouseDiskFreeAlertBytes: 1)
        let now = Int64(Date().timeIntervalSince1970)
        let clickHouse = AlertingClickHouse(now: now)
        let agent = AlertingAgent(intervalSeconds: 30)
        let context = AgentRuntimeContext(
            config: config,
            clickHouse: clickHouse,
            bridge: nil,
            eventStore: InMemoryAgentEventStore(),
            logger: Logger(level: .quiet),
            supervisorStartedAtUtc: UtcSecond(rawValue: now - 60),
            repairOnVerifierMismatch: false
        )

        let outcome = try await agent.run(context: context, startedAt: Date(timeIntervalSince1970: TimeInterval(now)))

        XCTAssertEqual(outcome.status, .warning)
        XCTAssertTrue(outcome.details.contains("utc_time_authority"))
        XCTAssertTrue(outcome.details.contains("unresolved verification mismatches=2"))
        XCTAssertTrue(outcome.details.contains("ClickHouse disk pressure"))
    }

    func testAgentSchedulerRunsStartupAgentsAndHonorsRunOnlyOnce() throws {
        let importer = StubAgent(
            descriptor: AgentDescriptor(
                kind: .historyImporter,
                intervalSeconds: 60,
                requiresMT5Bridge: true,
                runOnStart: true,
                runOnlyOnce: true
            )
        )
        let live = StubAgent(
            descriptor: AgentDescriptor(
                kind: .liveM1Updater,
                intervalSeconds: 10,
                requiresMT5Bridge: true,
                runOnStart: true
            )
        )
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        var scheduler = AgentScheduler()

        let firstDue = scheduler.dueAgents(from: [importer, live], now: now).map(\.descriptor.kind)
        XCTAssertEqual(firstDue, [.historyImporter, .liveM1Updater])

        scheduler.markFinished(.historyImporter, at: now, runOnlyOnce: true)
        scheduler.markFinished(.liveM1Updater, at: now, runOnlyOnce: false)

        let afterFiveSeconds = now.addingTimeInterval(5)
        XCTAssertTrue(scheduler.dueAgents(from: [importer, live], now: afterFiveSeconds).isEmpty)

        let afterElevenSeconds = now.addingTimeInterval(11)
        let laterDue = scheduler.dueAgents(from: [importer, live], now: afterElevenSeconds).map(\.descriptor.kind)
        XCTAssertEqual(laterDue, [.liveM1Updater])
    }

    func testAgentSchedulerSortsByPriority() {
        let backup = StubAgent(descriptor: AgentDescriptor(kind: .backupReadiness, intervalSeconds: 60, requiresMT5Bridge: false))
        let schema = StubAgent(descriptor: AgentDescriptor(kind: .schemaDriftGuard, intervalSeconds: 60, requiresMT5Bridge: false))
        let bridge = StubAgent(descriptor: AgentDescriptor(kind: .bridgeVersionGuard, intervalSeconds: 60, requiresMT5Bridge: true))
        let health = StubAgent(descriptor: AgentDescriptor(kind: .healthMonitor, intervalSeconds: 60, requiresMT5Bridge: false))
        let utc = StubAgent(descriptor: AgentDescriptor(kind: .utcTimeAuthority, intervalSeconds: 60, requiresMT5Bridge: true))
        var scheduler = AgentScheduler()

        let due = scheduler.dueAgents(
            from: [backup, utc, bridge, health, schema],
            now: Date(timeIntervalSince1970: 1_700_000_000)
        ).map(\.descriptor.kind)

        XCTAssertEqual(due, [.healthMonitor, .schemaDriftGuard, .bridgeVersionGuard, .utcTimeAuthority, .backupReadiness])
    }

    func testAgentSchedulerDeferredAgentRetriesAfterShortDelay() {
        let verifier = StubAgent(descriptor: AgentDescriptor(kind: .databaseVerifierRepairer, intervalSeconds: 3600, requiresMT5Bridge: false))
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        var scheduler = AgentScheduler()

        scheduler.markDeferred(.databaseVerifierRepairer, at: now, retryAfterSeconds: 10)

        XCTAssertTrue(scheduler.dueAgents(from: [verifier], now: now.addingTimeInterval(9)).isEmpty)
        XCTAssertEqual(
            scheduler.dueAgents(from: [verifier], now: now.addingTimeInterval(11)).map(\.descriptor.kind),
            [.databaseVerifierRepairer]
        )
    }

    func testDisabledHistoryImporterDoesNotSupersedeLiveStartup() throws {
        let config = try makeConfig()
        let agents = ProductionAgentFactory().makeAgents(config: config, runBackfillOnStart: false)
        var scheduler = AgentScheduler()

        let due = scheduler.dueAgents(
            from: agents,
            now: Date(timeIntervalSince1970: 1_700_000_000)
        ).map(\.descriptor.kind)

        XCTAssertFalse(due.contains(.historyImporter))
        XCTAssertTrue(due.contains(.liveM1Updater))
    }

    func testProductionAgentFactoryIncludesGuardAgentsInPriorityOrder() throws {
        let config = try makeConfig()
        let agents = ProductionAgentFactory().makeAgents(config: config, runBackfillOnStart: false)
        let kinds = agents.map(\.descriptor.kind)

        XCTAssertEqual(
            kinds,
            [
                .supervisorCoordinator,
                .healthMonitor,
                .schemaDriftGuard,
                .bridgeVersionGuard,
                .utcTimeAuthority,
                .symbolMetadataDrift,
                .sourceHistoryDrift,
                .sineTestSynchronizer,
                .historyImporter,
                .liveM1Updater,
                .databaseVerifierRepairer,
                .verificationCoveragePlanner,
                .checkpointGapAuditor,
                .dataCertification,
                .backupReadiness,
                .backupRestoreVerifier,
                .alerting
            ]
        )
    }

    func testEnabledHistoryImporterUsesCheckpointAuditRetryInterval() throws {
        let config = try makeConfig()
        let agents = ProductionAgentFactory().makeAgents(config: config, runBackfillOnStart: true)
        let importer = try XCTUnwrap(agents.first { $0.descriptor.kind == .historyImporter })

        XCTAssertEqual(importer.descriptor.intervalSeconds, config.app.supervisor.checkpointAuditIntervalSeconds)
    }

    func testSineWaveDatabaseSyncFillsMissingChunksAndSkipsCoveredRows() async throws {
        let clickHouse = SineSyncClickHouse()
        let sync = try SineWaveDatabaseSyncAgent(
            clickHouse: clickHouse,
            database: "db",
            chunkRows: 2,
            certifyInsertedRanges: false
        )
        let end = UtcSecond(rawValue: SineTestSecurity.genesisUtc.rawValue + 5 * 60)

        let first = try await sync.sync(utcEndExclusive: end)
        let second = try await sync.sync(utcEndExclusive: end)

        XCTAssertEqual(first.chunksInserted, 3)
        XCTAssertEqual(first.rowsInserted, 5)
        XCTAssertFalse(first.alreadyCurrent)
        XCTAssertTrue(second.alreadyCurrent)
        XCTAssertEqual(second.chunksInserted, 0)
        let rows = await clickHouse.canonicalRows()
        XCTAssertEqual(rows.count, 5)
        XCTAssertEqual(rows.first?.open, 1_000_000)
        XCTAssertEqual(rows.map(\.utc), [
            SineTestSecurity.genesisUtc.rawValue,
            SineTestSecurity.genesisUtc.rawValue + 60,
            SineTestSecurity.genesisUtc.rawValue + 120,
            SineTestSecurity.genesisUtc.rawValue + 180,
            SineTestSecurity.genesisUtc.rawValue + 240
        ])
        let canonicalInsertCount = await clickHouse.canonicalInsertCount()
        let coverageIntervalCount = await clickHouse.coverageIntervalCount()
        let ingestStateInsertCount = await clickHouse.ingestStateInsertCount()
        XCTAssertEqual(canonicalInsertCount, 3)
        XCTAssertEqual(coverageIntervalCount, 3)
        XCTAssertEqual(ingestStateInsertCount, 3)
    }

    func testClickHouseStartupManagerRunsStartCommandForLocalTransportFailure() async throws {
        let config = try makeConfig()
        let client = StartupClickHouse(failuresBeforeSuccess: 1)
        let command = SystemCommandRequest(
            executable: URL(fileURLWithPath: "/bin/echo"),
            arguments: ["start-clickhouse"],
            timeoutSeconds: 1
        )
        let runner = RecordingCommandRunner(resultExitCode: 0)
        let manager = ClickHouseStartupManager(
            config: config.clickHouse,
            client: client,
            logger: Logger(level: .quiet),
            commandRunner: runner,
            startCommands: [command],
            startupWaitSeconds: 0.1,
            pollIntervalNanoseconds: 1_000_000
        )

        try await manager.ensureReady()

        let commands = await runner.executedCommands()
        XCTAssertEqual(commands, ["/bin/echo start-clickhouse"])
    }

    func testClickHouseStartupManagerDoesNotStartRemoteEndpoint() async throws {
        var config = try makeConfig()
        config = ConfigBundle(
            app: config.app,
            clickHouse: ClickHouseConfig(
                url: try XCTUnwrap(URL(string: "http://db.example.com:8123")),
                database: config.clickHouse.database,
                username: nil,
                password: nil,
                requestTimeoutSeconds: 1,
                retryCount: 0
            ),
            mt5Bridge: config.mt5Bridge,
            brokerTime: config.brokerTime,
            symbols: config.symbols
        )
        let client = StartupClickHouse(failuresBeforeSuccess: Int.max)
        let runner = RecordingCommandRunner(resultExitCode: 0)
        let manager = ClickHouseStartupManager(
            config: config.clickHouse,
            client: client,
            logger: Logger(level: .quiet),
            commandRunner: runner,
            startCommands: [SystemCommandRequest(executable: URL(fileURLWithPath: "/bin/echo"), arguments: ["start"], timeoutSeconds: 1)]
        )

        await XCTAssertThrowsErrorAsync(try await manager.ensureReady()) { error in
            guard case ClickHouseStartupError.notAutoStartable = error else {
                XCTFail("Expected notAutoStartable, got \(error)")
                return
            }
        }
        let commands = await runner.executedCommands()
        XCTAssertTrue(commands.isEmpty)
    }

    func testOperationalFailureGuideCatalogCoversCoreDataSafetyFailures() {
        let text = OperationalFailureGuide.catalogText()

        XCTAssertTrue(text.contains("ClickHouse HTTP endpoint is down"))
        XCTAssertTrue(text.contains("MT5 bridge disconnects during live run"))
        XCTAssertTrue(text.contains("MetaEditor EA compile or toolchain failure"))
        XCTAssertTrue(text.contains("Missing verified broker UTC offsets"))
        XCTAssertTrue(text.contains("Canonical insert readback verification failed"))
        XCTAssertTrue(text.contains("Backtest data readiness blocked"))
        XCTAssertTrue(text.contains("Persistent logging unavailable"))
        XCTAssertTrue(text.contains("Disk full or ClickHouse storage pressure"))
        XCTAssertTrue(text.contains("Computer sleep, shutdown, or process interruption"))
    }

    func testOperationalFailureGuideMapsBacktestBlockToStopAdvice() {
        let advice = OperationalFailureGuide.advice(for: BacktestReadinessError.duplicateCanonicalKeys(2))

        XCTAssertEqual(advice.code, "BACKTEST-001")
        XCTAssertEqual(advice.severity, RecoverySeverity.stop)
        XCTAssertTrue(advice.dataSafety.contains("Research never runs"))
    }

    func testAgentExecutionPolicySupersedesConflictingAgents() {
        let policy = AgentExecutionPolicy()
        let staticBlocked = policy.staticSupersedence(for: [.historyImporter, .liveM1Updater, .databaseVerifierRepairer, .backupReadiness, .backupRestoreVerifier])
        XCTAssertEqual(staticBlocked[.liveM1Updater], "history_importer owns first-run/resume canonical writes this cycle")
        XCTAssertEqual(staticBlocked[.sourceHistoryDrift], "history_importer owns first-run/resume canonical writes this cycle")
        XCTAssertEqual(staticBlocked[.databaseVerifierRepairer], "history_importer owns first-run/resume canonical writes this cycle")
        XCTAssertEqual(staticBlocked[.backupReadiness], "history_importer owns first-run/resume canonical writes this cycle")
        XCTAssertEqual(staticBlocked[.backupRestoreVerifier], "history_importer owns first-run/resume canonical writes this cycle")

        let outcome = AgentOutcome(
            agent: .utcTimeAuthority,
            status: .failed,
            severity: .error,
            message: "offset mismatch",
            startedAtUtc: UtcSecond(rawValue: 1),
            finishedAtUtc: UtcSecond(rawValue: 2),
            durationMilliseconds: 1
        )
        let dynamicBlocked = policy.dynamicSupersedence(after: outcome)
        XCTAssertTrue(dynamicBlocked.keys.contains(.historyImporter))
        XCTAssertTrue(dynamicBlocked.keys.contains(.liveM1Updater))
        XCTAssertTrue(dynamicBlocked.keys.contains(.databaseVerifierRepairer))

        let schemaOutcome = AgentOutcome(
            agent: .schemaDriftGuard,
            status: .failed,
            severity: .error,
            message: "schema drift",
            startedAtUtc: UtcSecond(rawValue: 1),
            finishedAtUtc: UtcSecond(rawValue: 2),
            durationMilliseconds: 1
        )
        let schemaBlocked = policy.dynamicSupersedence(after: schemaOutcome)
        XCTAssertTrue(schemaBlocked.keys.contains(.bridgeVersionGuard))
        XCTAssertTrue(schemaBlocked.keys.contains(.historyImporter))
        XCTAssertTrue(schemaBlocked.keys.contains(.backupRestoreVerifier))
    }
}
