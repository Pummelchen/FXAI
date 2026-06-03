import AppCore
import FXDatabaseHistoryCore
import ClickHouse
import Config
import Domain
import Foundation
import FXBacktestAPIServer
import Ingestion
import FXDatabaseHistoryMetal
import MT5Bridge
import Operations
import TimeMapping
import Verification

extension FXDatabaseCLI {
    static func connectBridge(config: ConfigBundle, logger: Logger) throws -> MT5BridgeClient {
        do {
            switch config.mt5Bridge.mode {
            case .listen:
                logger.db("Waiting for MT5 EA bridge at \(config.mt5Bridge.host):\(config.mt5Bridge.port)")
                return try MT5BridgeClient.listen(
                    host: config.mt5Bridge.host,
                    port: config.mt5Bridge.port,
                    connectTimeoutSeconds: config.mt5Bridge.connectTimeoutSeconds,
                    requestTimeoutSeconds: config.mt5Bridge.requestTimeoutSeconds
                )
            case .connect:
                logger.db("Connecting to MT5 bridge at \(config.mt5Bridge.host):\(config.mt5Bridge.port)")
                return try MT5BridgeClient.connect(
                    host: config.mt5Bridge.host,
                    port: config.mt5Bridge.port,
                    connectTimeoutSeconds: config.mt5Bridge.connectTimeoutSeconds,
                    requestTimeoutSeconds: config.mt5Bridge.requestTimeoutSeconds
                )
            }
        } catch let error as MT5BridgeError {
            throw MT5BridgeStartupError(error: error, config: config.mt5Bridge)
        }
    }

    static func makeLogger(config: ConfigBundle, options: CLIOptions) -> Logger {
        let level = options.overrideLogLevel ?? config.app.logLevel
        guard config.app.logging.fileLoggingEnabled else {
            return Logger(level: level)
        }

        let baseDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        var startupWarnings: [String] = []
        let logSink = makePersistentSink(
            path: config.app.logging.logFilePath,
            baseDirectory: baseDirectory,
            maxFileBytes: config.app.logging.maxFileBytes,
            maxRotatedFiles: config.app.logging.maxRotatedFiles,
            warnings: &startupWarnings
        )
        let alertSink = makePersistentSink(
            path: config.app.logging.alertFilePath,
            baseDirectory: baseDirectory,
            maxFileBytes: config.app.logging.maxFileBytes,
            maxRotatedFiles: config.app.logging.maxRotatedFiles,
            warnings: &startupWarnings
        )
        let logger = Logger(level: level, persistentLogSink: logSink, alertSink: alertSink)
        if logSink != nil {
            logger.info("Persistent log file enabled: \(config.app.logging.logFilePath)")
        }
        if alertSink != nil {
            logger.info("Persistent alert file enabled: \(config.app.logging.alertFilePath)")
        }
        for warning in startupWarnings {
            logger.warn(warning)
        }
        return logger
    }

    static func makePersistentSink(
        path: String,
        baseDirectory: URL,
        maxFileBytes: UInt64,
        maxRotatedFiles: Int,
        warnings: inout [String]
    ) -> PersistentLogSink? {
        do {
            let url = try PersistentLogSink.resolvedURL(path: path, baseDirectory: baseDirectory)
            return try PersistentLogSink(
                fileURL: url,
                maxFileBytes: maxFileBytes,
                maxRotatedFiles: maxRotatedFiles
            )
        } catch {
            warnings.append("Persistent logging path '\(path)' is disabled: \(error)")
            return nil
        }
    }

    static func selectedSymbols(from argument: String?) throws -> [LogicalSymbol]? {
        guard let argument, argument.lowercased() != "all" else { return nil }
        return try argument.split(separator: ",").map { try LogicalSymbol(String($0)) }
    }

    static func loadHistoryDataConfig(_ url: URL) throws -> HistoryDataConfigFile {
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(HistoryDataConfigFile.self, from: data)
        } catch {
            throw ConfigError.invalidFile(url, error.localizedDescription)
        }
    }

    static func makeHistoryDataRequest(
        _ historyConfig: HistoryDataConfigFile,
        config: ConfigBundle
    ) throws -> HistoricalOhlcRequest {
        guard let mapping = config.symbols.mapping(for: historyConfig.logicalSymbol, sourceOrigin: historyConfig.sourceOrigin) else {
            throw CLIError.invalidValue("logical_symbol")
        }
        return try HistoricalOhlcRequest(
            brokerSourceId: historyConfig.brokerSourceId,
            sourceOrigin: historyConfig.sourceOrigin,
            logicalSymbol: historyConfig.logicalSymbol,
            utcStartInclusive: historyConfig.fromUtc,
            utcEndExclusive: historyConfig.toUtc,
            expectedMT5Symbol: mapping.mt5Symbol,
            expectedDigits: mapping.digits
        )
    }

    static func runLiveWithRecovery(
        config: ConfigBundle,
        clickHouse: ClickHouseClientProtocol,
        logger: Logger
    ) async throws {
        let checkpointStore = ClickHouseCheckpointStore(
            client: clickHouse,
            insertBuilder: ClickHouseInsertBuilder(database: config.clickHouse.database),
            database: config.clickHouse.database
        )
        let offsetStore = ClickHouseBrokerOffsetStore(client: clickHouse, database: config.clickHouse.database)
        let scanNanoseconds = UInt64(config.app.liveScanIntervalSeconds) * 1_000_000_000
        var bridgeBackoffSeconds: UInt64 = 5
        var clickHouseBackoffSeconds = UInt64(max(10, config.app.liveScanIntervalSeconds))

        logger.info("Starting resilient live updater; scan interval \(config.app.liveScanIntervalSeconds)s")
        while !Task.isCancelled {
            do {
                let bridge = try connectBridge(config: config, logger: logger)
                bridgeBackoffSeconds = 5
                clickHouseBackoffSeconds = UInt64(max(10, config.app.liveScanIntervalSeconds))
                let agent = LiveUpdateAgent(
                    config: config,
                    bridge: bridge,
                    clickHouse: clickHouse,
                    checkpointStore: checkpointStore,
                    offsetStore: offsetStore,
                    logger: logger
                )
                while !Task.isCancelled {
                    do {
                        try await agent.runOnce()
                        try await Task.sleep(nanoseconds: scanNanoseconds)
                    } catch let error as MT5BridgeError {
                        logger.alert("MT5 bridge disconnected; live updater will reconnect", details: error.description)
                        break
                    } catch let error as ProtocolError {
                        logger.alert("MT5 bridge protocol failed; live updater will reconnect after EA check", details: error.description)
                        logger.verbose(OperationalFailureGuide.advice(for: error).formatted)
                        break
                    } catch let error as ClickHouseError {
                        logger.alert("ClickHouse failed during live update; attempting local recovery", details: error.description)
                        logger.verbose(OperationalFailureGuide.advice(for: error).formatted)
                        if await recoverClickHouse(config: config, clickHouse: clickHouse, logger: logger) {
                            clickHouseBackoffSeconds = UInt64(max(10, config.app.liveScanIntervalSeconds))
                            try await Task.sleep(nanoseconds: scanNanoseconds)
                        } else {
                            logger.alert("ClickHouse is still unavailable; retrying database recovery in \(clickHouseBackoffSeconds)s")
                            try await Task.sleep(nanoseconds: clickHouseBackoffSeconds * 1_000_000_000)
                            clickHouseBackoffSeconds = min(300, clickHouseBackoffSeconds * 2)
                        }
                    } catch {
                        logger.warn("Live update cycle failed safely; checkpoint was not advanced unless readback verification already passed")
                        logger.verbose(OperationalFailureGuide.advice(for: error).formatted)
                        try await Task.sleep(nanoseconds: scanNanoseconds)
                    }
                }
            } catch let error as MT5BridgeStartupError {
                logger.alert("MT5 bridge unavailable; retrying in \(bridgeBackoffSeconds)s", details: error.description)
                logger.verbose(error.description)
            } catch let error as MT5BridgeError {
                logger.alert("MT5 bridge unavailable; retrying in \(bridgeBackoffSeconds)s", details: error.description)
                logger.verbose(OperationalFailureGuide.advice(for: error).formatted)
            }

            try await Task.sleep(nanoseconds: bridgeBackoffSeconds * 1_000_000_000)
            bridgeBackoffSeconds = min(300, bridgeBackoffSeconds * 2)
        }
    }

    static func runSineTestSync(
        config: ConfigBundle,
        clickHouse: ClickHouseClientProtocol,
        logger: Logger,
        watch: Bool
    ) async throws {
        let sync = try SineWaveDatabaseSyncAgent(
            clickHouse: clickHouse,
            database: config.clickHouse.database,
            chunkRows: config.app.chunkSize
        )
        let sleepNanoseconds = UInt64(SineTestSecurity.syncIntervalSeconds) * 1_000_000_000
        repeat {
            let result = try await sync.syncThroughRuntimeNow()
            if result.alreadyCurrent {
                logger.ok("SineTest data is current through UTC \(result.targetEndExclusiveUtc.rawValue)")
            } else {
                logger.ok("SineTest sync inserted \(result.rowsInserted) row(s) across \(result.chunksInserted) chunk(s) through UTC \(result.targetEndExclusiveUtc.rawValue)")
            }
            if watch {
                try await Task.sleep(nanoseconds: sleepNanoseconds)
            }
        } while watch && !Task.isCancelled
    }

    static func recoverClickHouse(
        config: ConfigBundle,
        clickHouse: ClickHouseClientProtocol,
        logger: Logger
    ) async -> Bool {
        do {
            try await ClickHouseStartupManager(
                config: config.clickHouse,
                client: clickHouse,
                logger: logger
            ).ensureReady()
            return true
        } catch {
            logger.warn("ClickHouse automatic recovery did not complete")
            logger.verbose(OperationalFailureGuide.advice(for: error).formatted)
            return false
        }
    }

    static func verifyLiveBrokerOffset(
        bridge: MT5BridgeClient,
        clickHouse: ClickHouseClientProtocol,
        config: ConfigBundle,
        terminal: TerminalInfoDTO,
        logger: Logger
    ) async throws -> BrokerOffsetMap {
        let terminalIdentity = try TerminalIdentityPolicy().resolve(
            actual: terminal,
            brokerSourceId: config.brokerTime.brokerSourceId,
            expected: config.brokerTime.expectedTerminalIdentity,
            logger: logger
        )
        let liveSnapshot = try bridge.serverTimeSnapshot()
        try await BrokerOffsetAutoAuthority(
            clickHouse: clickHouse,
            database: config.clickHouse.database,
            logger: logger
        ).ensureLiveSegmentIfMissing(
            brokerSourceId: config.brokerTime.brokerSourceId,
            terminalIdentity: terminalIdentity,
            snapshot: liveSnapshot
        )
        let offsetMap = try await ClickHouseBrokerOffsetStore(
            client: clickHouse,
            database: config.clickHouse.database
        ).loadVerifiedOffsetMap(
            brokerSourceId: config.brokerTime.brokerSourceId,
            terminalIdentity: terminalIdentity
        )
        logger.ok("Loaded \(offsetMap.segments.count) verified broker UTC offset segment(s) from ClickHouse for \(terminalIdentity)")
        try BrokerOffsetRuntimeVerifier().verify(
            snapshot: liveSnapshot,
            offsetMap: offsetMap,
            acceptedLiveOffsetSeconds: config.brokerTime.acceptedLiveOffsetSeconds,
            logger: logger
        )
        return offsetMap
    }

    static func resolveBrokerConfigurationIfNeeded(
        config: ConfigBundle,
        clickHouse: ClickHouseClientProtocol,
        terminal: TerminalInfoDTO,
        logger: Logger
    ) async throws -> ConfigBundle {
        guard config.brokerTime.isAutomatic else {
            return config
        }
        let resolution = try await BrokerSourceRegistry(
            client: clickHouse,
            database: config.clickHouse.database
        ).resolve(terminalInfo: terminal)
        let action = resolution.wasCreated ? "auto-discovered" : "auto-resolved"
        logger.ok("Broker source \(action): \(resolution.brokerSourceId.rawValue) for \(resolution.terminalIdentity)")
        return config.resolvingBrokerSourceId(resolution.brokerSourceId)
    }

    static func acquireInitialRuntimeLock(config: ConfigBundle, owner: String) throws -> SupervisorLock {
        let lockId = config.brokerTime.isAutomatic ? "auto-resolution" : config.brokerTime.brokerSourceId.rawValue
        return try SupervisorLock.acquireRuntime(brokerSourceId: lockId, owner: "\(owner)-resolution")
    }

    static func acquireResolvedRuntimeLockIfNeeded(
        initialLock: SupervisorLock,
        originalConfig: ConfigBundle,
        resolvedConfig: ConfigBundle,
        owner: String
    ) throws -> SupervisorLock {
        guard originalConfig.brokerTime.isAutomatic else {
            return initialLock
        }
        return try SupervisorLock.acquireRuntime(
            brokerSourceId: resolvedConfig.brokerTime.brokerSourceId.rawValue,
            owner: owner
        )
    }

    static func runRepair(
        options: CLIOptions,
        config: ConfigBundle,
        clickHouse: ClickHouseClientProtocol,
        logger: Logger
    ) async throws {
        guard let symbol = options.repairSymbol else {
            throw CLIError.missingValue("--symbol")
        }
        guard let from = options.fromUtcDay else {
            throw CLIError.missingValue("--from")
        }
        guard let to = options.toUtcDay else {
            throw CLIError.missingValue("--to")
        }
        guard config.symbols.mapping(for: symbol, sourceOrigin: .mt5) != nil else {
            throw CLIError.invalidValue("--symbol")
        }

        let originalConfig = config
        let initialLock = try acquireInitialRuntimeLock(config: config, owner: "repair")
        logger.ok("Initial broker runtime lock acquired: \(initialLock.path)")
        let bridge = try connectBridge(config: config, logger: logger)
        let terminal = try bridge.terminalInfo()
        let config = try await resolveBrokerConfigurationIfNeeded(
            config: config,
            clickHouse: clickHouse,
            terminal: terminal,
            logger: logger
        )
        let lock = try acquireResolvedRuntimeLockIfNeeded(
            initialLock: initialLock,
            originalConfig: originalConfig,
            resolvedConfig: config,
            owner: "repair"
        )
        logger.ok("Broker runtime lock acquired: \(lock.path)")
        let offsetMap = try await verifyLiveBrokerOffset(
            bridge: bridge,
            clickHouse: clickHouse,
            config: config,
            terminal: terminal,
            logger: logger
        )
        let ranges = try RepairRangePlanner().mt5Ranges(
            brokerSourceId: config.brokerTime.brokerSourceId,
            logicalSymbol: symbol,
            utcStart: from,
            utcEndExclusive: to,
            offsetMap: offsetMap
        )
        let verifier = HistoricalRangeVerifier(
            config: config,
            bridge: bridge,
            clickHouse: clickHouse,
            offsetMap: offsetMap,
            logger: logger
        )
        let repairAgent = RepairAgent(
            clickHouse: clickHouse,
            database: config.clickHouse.database,
            logger: logger
        )
        let policy = RepairPolicy()
        for range in ranges {
            let outcome = try await verifier.verify(range: range)
            let decision = policy.decide(
                verification: outcome.result,
                mt5Available: !outcome.mt5Bars.isEmpty,
                sourceComplete: outcome.sourceComplete,
                utcMappingAmbiguous: false
            )
            try await repairAgent.repairCanonicalRange(
                range: range,
                replacementBars: outcome.mt5Bars,
                decision: decision,
                sourceComplete: outcome.sourceComplete,
                verifiedCoverage: outcome.verifiedCoverage
            )
            if case .noRepairNeeded = decision {
                continue
            } else {
                let recheck = try await verifier.verify(range: range)
                guard recheck.result.isClean else {
                    throw RepairError.refused("post-repair verification still reports \(recheck.result.mismatches.count) mismatch(es)")
                }
            }
        }
        logger.ok("\(symbol.rawValue): repair command completed for UTC range \(from.rawValue)..<\(to.rawValue)")
        _ = initialLock
        _ = lock
    }

    static func parseUtcDay(_ value: String) throws -> UtcSecond {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd"
        guard let date = formatter.date(from: value) else {
            throw CLIError.invalidValue(value)
        }
        return UtcSecond(rawValue: Int64(date.timeIntervalSince1970))
    }

    static func printUsage() {
        print("""
        FXDatabase

        Start FXDatabase without launch-time input, then type commands at the `>` prompt.

        Interactive commands:
          migrate
          bridge-check
          symbol-check
          backfill --symbols all
          backfill --symbols EURUSD,USDJPY
          live
          supervise [--with-backfill] [--supervisor-cycles N]
          startcheck
          failure-guide
          verify
          verify --random-ranges 20
          repair --symbol EURUSD --from 2020-01-01 --to 2020-02-01
          data-check --config Config/history_data.json
          sinetest-sync [--watch]
          fxbacktest-api [--api-host 127.0.0.1] [--api-port 5066]
          health-api [--api-host 127.0.0.1] [--api-port 5067]

        Command options:
          --config-dir Config
          --migrations-dir Migrations
          --config Config/history_data.json   # data-check only
          --api-host 127.0.0.1                # fxbacktest-api / health-api
          --api-port 5066                     # fxbacktest-api / health-api
          --watch                             # sinetest-sync only; repeats every 10s
          --verbose
          --debug

        Shell control commands:
          status
          stop
          wait
          help
          exit
        """)
    }
}
