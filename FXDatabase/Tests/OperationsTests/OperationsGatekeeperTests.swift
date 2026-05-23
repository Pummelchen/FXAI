import AppCore
import ClickHouse
import Config
import Domain
import TimeMapping
@testable import Operations
import XCTest

final class OperationsGatekeeperTests: XCTestCase {
    func testSchemaDriftGuardFailsWhenRequiredTablesAreMissing() async throws {
        let config = try makeConfig()
        let agent = SchemaDriftGuardAgent(intervalSeconds: 60)
        let context = AgentRuntimeContext(
            config: config,
            clickHouse: FixedClickHouse(body: ""),
            bridge: nil,
            eventStore: InMemoryAgentEventStore(),
            logger: Logger(level: .quiet),
            supervisorStartedAtUtc: UtcSecond(rawValue: 1),
            repairOnVerifierMismatch: false
        )

        let outcome = try await agent.run(context: context, startedAt: Date(timeIntervalSince1970: 1))

        XCTAssertEqual(outcome.status, .failed)
        XCTAssertTrue(outcome.message.contains("missing required tables"))
        XCTAssertTrue(outcome.details.contains("mt5_ohlc_m1_raw"))
    }

    func testVerificationCoveragePlannerWarnsOnMissingCoverageAndCleanVerification() async throws {
        let config = try makeConfig()
        let clickHouse = VerificationPlannerClickHouse()
        let agent = VerificationCoveragePlannerAgent(intervalSeconds: 3600)
        let context = AgentRuntimeContext(
            config: config,
            clickHouse: clickHouse,
            bridge: nil,
            eventStore: InMemoryAgentEventStore(),
            logger: Logger(level: .quiet),
            supervisorStartedAtUtc: UtcSecond(rawValue: 1),
            repairOnVerifierMismatch: false
        )

        let outcome = try await agent.run(context: context, startedAt: Date(timeIntervalSince1970: 1))

        XCTAssertEqual(outcome.status, .warning)
        XCTAssertTrue(outcome.details.contains("missing_coverage=USDJPY"))
        XCTAssertTrue(outcome.details.contains("missing_clean_verification=USDJPY"))
    }

    func testVerificationCoveragePlannerAcceptsOverlappingCoverageLedgerRows() async throws {
        let config = try makeConfig(symbols: ["EURUSD": 5])
        let clickHouse = OverlappingVerificationPlannerClickHouse()
        let agent = VerificationCoveragePlannerAgent(intervalSeconds: 3600)
        let context = AgentRuntimeContext(
            config: config,
            clickHouse: clickHouse,
            bridge: nil,
            eventStore: InMemoryAgentEventStore(),
            logger: Logger(level: .quiet),
            supervisorStartedAtUtc: UtcSecond(rawValue: 1),
            repairOnVerifierMismatch: false
        )

        let outcome = try await agent.run(context: context, startedAt: Date(timeIntervalSince1970: 1))

        XCTAssertEqual(outcome.status, .ok)
        XCTAssertFalse(outcome.details.contains("coverage_span_gaps"))
    }

    func testBackupRestoreVerifierBlocksOnUnfinishedOperations() async throws {
        let config = try makeConfig()
        let clickHouse = BackupRestoreClickHouse(mode: .unfinished)
        let agent = BackupRestoreVerifierAgent(intervalSeconds: 3600)
        let context = AgentRuntimeContext(
            config: config,
            clickHouse: clickHouse,
            bridge: nil,
            eventStore: InMemoryAgentEventStore(),
            logger: Logger(level: .quiet),
            supervisorStartedAtUtc: UtcSecond(rawValue: 1),
            repairOnVerifierMismatch: false
        )

        let outcome = try await agent.run(context: context, startedAt: Date(timeIntervalSince1970: 1))

        XCTAssertEqual(outcome.status, .warning)
        XCTAssertTrue(outcome.message.contains("unfinished ingest or repair batches"))
    }

    func testInMemoryAgentEventStoreRecordsOutcomes() async throws {
        let store = InMemoryAgentEventStore()
        let broker = try BrokerSourceId("unit-test")
        let outcome = AgentOutcome(
            agent: .healthMonitor,
            status: .ok,
            severity: .info,
            message: "ok",
            startedAtUtc: UtcSecond(rawValue: 1),
            finishedAtUtc: UtcSecond(rawValue: 2),
            durationMilliseconds: 1
        )

        try await store.record(outcome, brokerSourceId: broker)

        let outcomes = await store.outcomes
        XCTAssertEqual(outcomes, [outcome])
    }

    func testClickHouseAgentStatePreservesPreviousOkTimestampOnWarning() async throws {
        let clickHouse = RecordingClickHouse(selectBodies: ["", "10\t0\n"])
        let store = ClickHouseAgentEventStore(clickHouse: clickHouse, database: "db")
        let broker = try BrokerSourceId("unit-test")
        let ok = AgentOutcome(
            agent: .healthMonitor,
            status: .ok,
            severity: .info,
            message: "ok",
            startedAtUtc: UtcSecond(rawValue: 9),
            finishedAtUtc: UtcSecond(rawValue: 10),
            durationMilliseconds: 1
        )
        let warning = AgentOutcome(
            agent: .healthMonitor,
            status: .warning,
            severity: .warning,
            message: "warn",
            startedAtUtc: UtcSecond(rawValue: 19),
            finishedAtUtc: UtcSecond(rawValue: 20),
            durationMilliseconds: 1
        )

        try await store.record(ok, brokerSourceId: broker)
        try await store.record(warning, brokerSourceId: broker)

        let stateInserts = await clickHouse.queries
            .map(\.sql)
            .filter { $0.contains("INSERT INTO db.runtime_agent_state") }
        XCTAssertEqual(stateInserts.count, 2)
        XCTAssertTrue(stateInserts[0].contains("\thealth_monitor\tok\tok\t10\t0\t10"))
        XCTAssertTrue(stateInserts[1].contains("\thealth_monitor\twarning\twarn\t10\t0\t20"))
    }

    func testMetaEditorWinePathArgumentUsesZDriveAbsolutePath() {
        let url = URL(fileURLWithPath: "/tmp/Project/FXImporter/Connectors/MetaTrader5/EA/FXDatabase.mq5")
        XCTAssertEqual(
            MetaEditorToolchain.winePathArgument(url),
            "Z:\\tmp\\Project\\FXImporter\\Connectors\\MetaTrader5\\EA\\FXDatabase.mq5"
        )
    }

    func testStartCheckDetectsOffsetCoverageGaps() throws {
        let broker = try BrokerSourceId("demo")
        let identity = try BrokerServerIdentity(company: "Broker Ltd", server: "Broker-Server", accountLogin: 1)
        let map = try BrokerOffsetMap(
            brokerSourceId: broker,
            terminalIdentity: identity,
            segments: [
                BrokerOffsetSegment(
                    brokerSourceId: broker,
                    terminalIdentity: identity,
                    validFrom: MT5ServerSecond(rawValue: 0),
                    validTo: MT5ServerSecond(rawValue: 120),
                    offset: OffsetSeconds(rawValue: 7200),
                    source: .manual,
                    confidence: .verified
                ),
                BrokerOffsetSegment(
                    brokerSourceId: broker,
                    terminalIdentity: identity,
                    validFrom: MT5ServerSecond(rawValue: 240),
                    validTo: MT5ServerSecond(rawValue: 360),
                    offset: OffsetSeconds(rawValue: 7200),
                    source: .manual,
                    confidence: .verified
                )
            ]
        )

        let gaps = StartCheckRunner.coverageGaps(
            in: map,
            from: MT5ServerSecond(rawValue: 60),
            toExclusive: MT5ServerSecond(rawValue: 300)
        )

        XCTAssertEqual(gaps, ["120..<240"])
    }

    func testBacktestReadinessGatePassesWhenDataAndAgentsAreClean() async throws {
        let config = try makeConfig()
        let clickHouse = BacktestGateClickHouse(mode: .clean)
        let gate = BacktestReadinessGate(config: config, clickHouse: clickHouse)

        try await gate.assertReady(BacktestReadinessRequest(
            brokerSourceId: config.brokerTime.brokerSourceId,
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 120)
        ))
    }

    func testBacktestReadinessGateBlocksInterruptedBackfill() async throws {
        let config = try makeConfig()
        let clickHouse = BacktestGateClickHouse(mode: .interruptedBackfill)
        let gate = BacktestReadinessGate(config: config, clickHouse: clickHouse)

        await XCTAssertThrowsErrorAsync(try await gate.assertReady(BacktestReadinessRequest(
            brokerSourceId: config.brokerTime.brokerSourceId,
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 120)
        ))) { error in
            guard case BacktestReadinessError.incompleteIngest = error else {
                XCTFail("Expected incompleteIngest, got \(error)")
                return
            }
        }
    }

    func testBacktestReadinessGateBlocksFailedAgentState() async throws {
        let config = try makeConfig()
        let clickHouse = BacktestGateClickHouse(mode: .failedAgentState)
        let gate = BacktestReadinessGate(config: config, clickHouse: clickHouse)

        await XCTAssertThrowsErrorAsync(try await gate.assertReady(BacktestReadinessRequest(
            brokerSourceId: config.brokerTime.brokerSourceId,
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 120)
        ))) { error in
            guard case BacktestReadinessError.blockingAgentState = error else {
                XCTFail("Expected blockingAgentState, got \(error)")
                return
            }
        }
    }

    func testBacktestReadinessGateBlocksMissingRequiredAgentState() async throws {
        let config = try makeConfig()
        let clickHouse = BacktestGateClickHouse(mode: .missingRequiredAgentState)
        let gate = BacktestReadinessGate(config: config, clickHouse: clickHouse)

        await XCTAssertThrowsErrorAsync(try await gate.assertReady(BacktestReadinessRequest(
            brokerSourceId: config.brokerTime.brokerSourceId,
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 120)
        ))) { error in
            guard case BacktestReadinessError.missingRequiredAgentState = error else {
                XCTFail("Expected missingRequiredAgentState, got \(error)")
                return
            }
        }
    }

    func testBacktestReadinessGateBlocksUnfinishedIngestOperation() async throws {
        let config = try makeConfig()
        let clickHouse = BacktestGateClickHouse(mode: .unfinishedIngestOperation)
        let gate = BacktestReadinessGate(config: config, clickHouse: clickHouse)

        await XCTAssertThrowsErrorAsync(try await gate.assertReady(BacktestReadinessRequest(
            brokerSourceId: config.brokerTime.brokerSourceId,
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 120)
        ))) { error in
            guard case BacktestReadinessError.unfinishedIngestOperations = error else {
                XCTFail("Expected unfinishedIngestOperations, got \(error)")
                return
            }
        }
    }

    func testBacktestReadinessGateBlocksMissingVerifiedCoverage() async throws {
        let config = try makeConfig()
        let clickHouse = BacktestGateClickHouse(mode: .missingVerifiedCoverage)
        let gate = BacktestReadinessGate(config: config, clickHouse: clickHouse)

        await XCTAssertThrowsErrorAsync(try await gate.assertReady(BacktestReadinessRequest(
            brokerSourceId: config.brokerTime.brokerSourceId,
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 120)
        ))) { error in
            guard case BacktestReadinessError.missingVerifiedCoverage = error else {
                XCTFail("Expected missingVerifiedCoverage, got \(error)")
                return
            }
        }
    }

    func testBacktestReadinessGateBlocksMissingDataCertificate() async throws {
        let config = try makeConfig()
        let clickHouse = BacktestGateClickHouse(mode: .missingDataCertificate)
        let gate = BacktestReadinessGate(config: config, clickHouse: clickHouse)

        await XCTAssertThrowsErrorAsync(try await gate.assertReady(BacktestReadinessRequest(
            brokerSourceId: config.brokerTime.brokerSourceId,
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 120)
        ))) { error in
            guard case BacktestReadinessError.missingDataCertificate = error else {
                XCTFail("Expected missingDataCertificate, got \(error)")
                return
            }
        }
    }

    func testCheckpointGapAuditWarnsOnMissingConfiguredCheckpoints() async throws {
        let config = try makeConfig()
        let clickHouse = CheckpointAuditClickHouse(mode: .missingUSDJPYCheckpoint)
        let agent = CheckpointGapAuditAgent(intervalSeconds: 300)
        let context = AgentRuntimeContext(
            config: config,
            clickHouse: clickHouse,
            bridge: nil,
            eventStore: InMemoryAgentEventStore(),
            logger: Logger(level: .quiet),
            supervisorStartedAtUtc: UtcSecond(rawValue: 1),
            repairOnVerifierMismatch: false
        )

        let outcome = try await agent.run(context: context, startedAt: Date(timeIntervalSince1970: 1))

        XCTAssertEqual(outcome.status, .warning)
        XCTAssertTrue(outcome.message.contains("missing ingest checkpoints"))
        XCTAssertTrue(outcome.details.contains("missing_checkpoints=USDJPY"))
    }

    func testCheckpointGapAuditWarnsOnInterruptedBackfillState() async throws {
        let config = try makeConfig()
        let clickHouse = CheckpointAuditClickHouse(mode: .interruptedEURUSD)
        let agent = CheckpointGapAuditAgent(intervalSeconds: 300)
        let context = AgentRuntimeContext(
            config: config,
            clickHouse: clickHouse,
            bridge: nil,
            eventStore: InMemoryAgentEventStore(),
            logger: Logger(level: .quiet),
            supervisorStartedAtUtc: UtcSecond(rawValue: 1),
            repairOnVerifierMismatch: false
        )

        let outcome = try await agent.run(context: context, startedAt: Date(timeIntervalSince1970: 1))

        XCTAssertEqual(outcome.status, .warning)
        XCTAssertTrue(outcome.message.contains("not live"))
        XCTAssertTrue(outcome.details.contains("EURUSD:status=backfilling"))
    }

    func testBackupReadinessFiltersCanonicalRowsByConfiguredBroker() async throws {
        let config = try makeConfig()
        let clickHouse = BackupReadinessClickHouse()
        let agent = BackupReadinessAgent(intervalSeconds: 3600)
        let context = AgentRuntimeContext(
            config: config,
            clickHouse: clickHouse,
            bridge: nil,
            eventStore: InMemoryAgentEventStore(),
            logger: Logger(level: .quiet),
            supervisorStartedAtUtc: UtcSecond(rawValue: 1),
            repairOnVerifierMismatch: false
        )

        _ = try await agent.run(context: context, startedAt: Date(timeIntervalSince1970: 1))

        let queries = await clickHouse.queries
        XCTAssertEqual(queries.count, 2)
        XCTAssertTrue(queries[0].sql.contains("WHERE broker_source_id = 'demo'"))
        XCTAssertTrue(queries[1].sql.contains("FROM db.data_certificates"))
        XCTAssertTrue(queries[1].sql.contains("WHERE broker_source_id = 'demo'"))
    }

    func testDataCertificateRejectsPartialVerifiedCoverage() async throws {
        let clickHouse = DataCertificateClickHouse(rows: [
            coverageRow(utcStart: 60, utcEnd: 120)
        ])
        let store = DataCertificateStore(clickHouse: clickHouse, database: "db")

        await XCTAssertThrowsErrorAsync(try await store.certify(
            brokerSourceId: try BrokerSourceId("demo"),
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 180)
        )) { error in
            guard case DataCertificateError.incompleteVerifiedCoverage = error else {
                XCTFail("Expected incompleteVerifiedCoverage, got \(error)")
                return
            }
        }
    }

    func testDataCertificateRequiresMatchingSourceAndCanonicalCounts() async throws {
        let clickHouse = DataCertificateClickHouse(rows: [
            coverageRow(utcStart: 60, utcEnd: 120, sourceCount: 1, canonicalCount: 0)
        ])
        let store = DataCertificateStore(clickHouse: clickHouse, database: "db")

        await XCTAssertThrowsErrorAsync(try await store.certify(
            brokerSourceId: try BrokerSourceId("demo"),
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 120)
        )) { error in
            guard case DataCertificateError.inconsistentCoverageCounts = error else {
                XCTFail("Expected inconsistentCoverageCounts, got \(error)")
                return
            }
        }
    }

    func testDataCertificateCreatesForContiguousVerifiedCoverage() async throws {
        let clickHouse = DataCertificateClickHouse(rows: [
            coverageRow(utcStart: 60, utcEnd: 120),
            coverageRow(utcStart: 120, utcEnd: 180)
        ])
        let store = DataCertificateStore(clickHouse: clickHouse, database: "db")

        let certificate = try await store.certify(
            brokerSourceId: try BrokerSourceId("demo"),
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 180),
            createdAtUtc: UtcSecond(rawValue: 1_000)
        )

        XCTAssertEqual(certificate.coverageRowCount, 2)
        XCTAssertEqual(certificate.coverageSourceBarCount, 2)
        XCTAssertEqual(certificate.coverageCanonicalRowCount, 2)
        let queries = await clickHouse.queries
        XCTAssertEqual(queries.count, 2)
        XCTAssertTrue(queries[1].sql.contains("INSERT INTO db.data_certificates"))
    }

    func testDataCertificateAllowsVerifiedEmptySourceGap() async throws {
        let clickHouse = DataCertificateClickHouse(rows: [
            coverageRow(utcStart: 60, utcEnd: 120, sourceCount: 0, canonicalCount: 0)
        ])
        let store = DataCertificateStore(clickHouse: clickHouse, database: "db")

        let certificate = try await store.certify(
            brokerSourceId: try BrokerSourceId("demo"),
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 120),
            createdAtUtc: UtcSecond(rawValue: 1_000)
        )

        XCTAssertEqual(certificate.coverageRowCount, 1)
        XCTAssertEqual(certificate.coverageSourceBarCount, 0)
        XCTAssertEqual(certificate.coverageCanonicalRowCount, 0)
        let queries = await clickHouse.queries
        XCTAssertTrue(queries[1].sql.contains("INSERT INTO db.data_certificates"))
    }

    func testDataCertificateUsesMinimalCoveringRowsWhenCoverageLedgerOverlaps() async throws {
        let clickHouse = DataCertificateClickHouse(rows: [
            coverageRow(utcStart: 60, utcEnd: 120, sourceCount: 1, canonicalCount: 1),
            coverageRow(utcStart: 60, utcEnd: 180, sourceCount: 2, canonicalCount: 2)
        ])
        let store = DataCertificateStore(clickHouse: clickHouse, database: "db")

        let certificate = try await store.certify(
            brokerSourceId: try BrokerSourceId("demo"),
            logicalSymbol: try LogicalSymbol("EURUSD"),
            utcStart: UtcSecond(rawValue: 60),
            utcEndExclusive: UtcSecond(rawValue: 180),
            createdAtUtc: UtcSecond(rawValue: 1_000)
        )

        XCTAssertEqual(certificate.coverageRowCount, 1)
        XCTAssertEqual(certificate.coverageSourceBarCount, 2)
        XCTAssertEqual(certificate.coverageCanonicalRowCount, 2)
    }
}
