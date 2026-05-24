import AppCore
import BacktestCore
import ClickHouse
import Config
import Domain
import Foundation
import FXBacktestAPIServer
import Ingestion
import MetalAccel
import MT5Bridge
import Operations
import TimeMapping
import Verification

@main
struct FXDatabaseCLI {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let result = await InteractiveCommandSession(ignoredLaunchArguments: arguments).run()
        Darwin.exit(result.rawValue)
    }

    static func run(arguments: [String]) async -> ExitCode {
        var activeLogger: Logger?
        do {
            let options = try CLIOptions(arguments: arguments)
            if options.command == .help {
                printUsage()
                return .success
            }
            if options.command == .failureGuide {
                print(OperationalFailureGuide.catalogText())
                return .success
            }
            if let reason = options.command.unavailableReason {
                throw CLIError.commandUnavailable(reason)
            }

            let loader = ConfigLoader()
            let config = try loader.loadBundle(configDirectory: options.configDirectory)
            let logger = makeLogger(config: config, options: options)
            activeLogger = logger
            let clickHouse = ClickHouseHTTPClient(config: config.clickHouse, logger: logger)
            if options.command.requiresClickHouseStartupCheck {
                try await ClickHouseStartupManager(
                    config: config.clickHouse,
                    client: clickHouse,
                    logger: logger
                ).ensureReady()
            }

            switch options.command {
            case .failureGuide:
                print(OperationalFailureGuide.catalogText())
                return .success

            case .migrate:
                logger.db("Connecting to ClickHouse at \(config.clickHouse.url.absoluteString)")
                _ = try await clickHouse.execute(.select("SELECT 1", databaseOverride: "default"))
                logger.ok("ClickHouse connection verified")
                try await ClickHouseMigrator(client: clickHouse, config: config.clickHouse, logger: logger)
                    .migrate(migrationsDirectory: options.migrationsDirectory)
                return .success

            case .bridgeCheck:
                let bridge = try connectBridge(config: config, logger: logger)
                let hello = try bridge.hello()
                logger.ok("MT5 bridge connected: \(hello.bridgeName) \(hello.bridgeVersion)")
                let terminal = try bridge.terminalInfo()
                let config = try await resolveBrokerConfigurationIfNeeded(
                    config: config,
                    clickHouse: clickHouse,
                    terminal: terminal,
                    logger: logger
                )
                logger.ok("MT5 terminal: \(terminal.terminalName), server \(terminal.server), account \(terminal.accountLogin)")
                _ = try await verifyLiveBrokerOffset(
                    bridge: bridge,
                    clickHouse: clickHouse,
                    config: config,
                    terminal: terminal,
                    logger: logger
                )
                return .success

            case .symbolCheck:
                let bridge = try connectBridge(config: config, logger: logger)
                var failureCount = 0
                for mapping in config.symbols.symbols {
                    do {
                        let info = try bridge.prepareSymbol(mapping.mt5Symbol)
                        if info.selected && info.digits == mapping.digits.rawValue {
                            logger.ok("\(mapping.logicalSymbol.rawValue): \(mapping.mt5Symbol.rawValue) selected, digits \(info.digits)")
                        } else if info.selected {
                            failureCount += 1
                            logger.warn("\(mapping.logicalSymbol.rawValue): digits mismatch, config \(mapping.digits.rawValue), MT5 \(info.digits)")
                        } else {
                            failureCount += 1
                            logger.error("\(mapping.logicalSymbol.rawValue): symbol \(mapping.mt5Symbol.rawValue) not selected in MT5")
                        }
                    } catch {
                        failureCount += 1
                        logger.error("\(mapping.logicalSymbol.rawValue): \(error)")
                    }
                }
                return failureCount == 0 ? .success : .validation

            case .backfill:
                let originalConfig = config
                let initialLock = try acquireInitialRuntimeLock(config: config, owner: "backfill")
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
                    owner: "backfill"
                )
                logger.ok("Broker runtime lock acquired: \(lock.path)")
                let checkpointStore = ClickHouseCheckpointStore(
                    client: clickHouse,
                    insertBuilder: ClickHouseInsertBuilder(database: config.clickHouse.database),
                    database: config.clickHouse.database
                )
                let offsetStore = ClickHouseBrokerOffsetStore(client: clickHouse, database: config.clickHouse.database)
                let agent = BackfillAgent(
                    config: config,
                    bridge: bridge,
                    clickHouse: clickHouse,
                    checkpointStore: checkpointStore,
                    offsetStore: offsetStore,
                    logger: logger
                )
                let symbols = try selectedSymbols(from: options.symbolsArgument)
                try await agent.run(selectedSymbols: symbols)
                _ = initialLock
                _ = lock
                return .success

            case .live:
                let originalConfig = config
                let initialLock = try acquireInitialRuntimeLock(config: config, owner: "live")
                logger.ok("Initial broker runtime lock acquired: \(initialLock.path)")
                let bridge = try connectBridge(config: config, logger: logger)
                let terminal = try bridge.terminalInfo()
                let config = try await resolveBrokerConfigurationIfNeeded(
                    config: config,
                    clickHouse: clickHouse,
                    terminal: terminal,
                    logger: logger
                )
                bridge.close()
                let lock = try acquireResolvedRuntimeLockIfNeeded(
                    initialLock: initialLock,
                    originalConfig: originalConfig,
                    resolvedConfig: config,
                    owner: "live"
                )
                logger.ok("Broker runtime lock acquired: \(lock.path)")
                try await runLiveWithRecovery(
                    config: config,
                    clickHouse: clickHouse,
                    logger: logger
                )
                _ = initialLock
                _ = lock
                return .success

            case .supervise:
                let originalConfig = config
                let initialLock = try acquireInitialRuntimeLock(config: config, owner: "supervise")
                logger.ok("Initial broker runtime lock acquired: \(initialLock.path)")
                let bridge = try connectBridge(config: config, logger: logger)
                let terminal = try bridge.terminalInfo()
                let config = try await resolveBrokerConfigurationIfNeeded(
                    config: config,
                    clickHouse: clickHouse,
                    terminal: terminal,
                    logger: logger
                )
                bridge.close()
                let lock = try acquireResolvedRuntimeLockIfNeeded(
                    initialLock: initialLock,
                    originalConfig: originalConfig,
                    resolvedConfig: config,
                    owner: "supervise"
                )
                logger.ok("Broker runtime lock acquired: \(lock.path)")
                let eventStore = ClickHouseAgentEventStore(
                    clickHouse: clickHouse,
                    database: config.clickHouse.database
                )
                let supervisor = ProductionSupervisor(
                    config: config,
                    clickHouse: clickHouse,
                    eventStore: eventStore,
                    logger: logger,
                    bridgeConnector: {
                        try connectBridge(config: config, logger: logger)
                    },
                    runBackfillOnStart: options.runBackfillOnStart ?? config.app.supervisor.runBackfillOnStart
                )
                try await supervisor.run(maxCycles: options.supervisorCycles)
                _ = initialLock
                _ = lock
                return .success

            case .startcheck:
                let runner = StartCheckRunner(
                    config: config,
                    clickHouse: clickHouse,
                    logger: logger,
                    bridgeConnector: {
                        try connectBridge(config: config, logger: logger)
                    },
                    options: StartCheckOptions(
                        migrationsDirectory: options.migrationsDirectory,
                        workingDirectory: URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
                        compileEA: options.compileEA,
                        bridgeChecks: options.bridgeChecks,
                        compileTimeoutSeconds: options.compileTimeoutSeconds
                    )
                )
                return await runner.run() ? .success : .verification

            case .verify:
                let randomRangeCount = options.randomRanges ?? config.app.verifierRandomRanges
                let bridge: MT5BridgeClient?
                var verifyConfig = config
                if options.shouldConnectBridgeForVerify(randomRangeCount: randomRangeCount) {
                    let connectedBridge = try connectBridge(config: config, logger: logger)
                    let terminal = try connectedBridge.terminalInfo()
                    verifyConfig = try await resolveBrokerConfigurationIfNeeded(
                        config: config,
                        clickHouse: clickHouse,
                        terminal: terminal,
                        logger: logger
                    )
                    bridge = connectedBridge
                } else {
                    bridge = nil
                }
                try await VerificationAgent(config: verifyConfig, bridge: bridge, clickHouse: clickHouse, logger: logger)
                    .startupChecks(randomRanges: randomRangeCount)
                return .success

            case .repair:
                try await runRepair(options: options, config: config, clickHouse: clickHouse, logger: logger)
                return .success

            case .exportCache:
                throw CLIError.commandUnavailable(options.command.unavailableReason ?? "Command unavailable.")

            case .dataCheck:
                guard let commandConfigPath = options.commandConfigPath else {
                    throw CLIError.missingValue("--config")
                }
                guard FileManager.default.fileExists(atPath: commandConfigPath.path) else {
                    throw CLIError.invalidValue("--config")
                }
                let historyConfig = try loadHistoryDataConfig(commandConfigPath)
                try await BacktestReadinessGate(config: config, clickHouse: clickHouse)
                    .assertReady(BacktestReadinessRequest(config: historyConfig))
                logger.ok("History data-readiness gate passed for \(historyConfig.logicalSymbol.rawValue)")
                let request = try makeHistoryDataRequest(historyConfig, config: config)
                let series = try await ClickHouseHistoricalOhlcDataProvider(
                    client: clickHouse,
                    database: config.clickHouse.database
                ).loadM1Ohlc(request)
                let first = series.metadata.firstUtc?.rawValue.description ?? "n/a"
                let last = series.metadata.lastUtc?.rawValue.description ?? "n/a"
                logger.ok("\(historyConfig.logicalSymbol.rawValue): loaded \(series.count) verified canonical M1 bars from ClickHouse")
                logger.info("\(historyConfig.logicalSymbol.rawValue): UTC range loaded first=\(first), last=\(last), digits=\(series.metadata.digits.rawValue)")
                if historyConfig.useMetal {
                    #if canImport(Metal)
                    let availability = MetalAvailability()
                    guard availability.isAvailable else {
                        logger.warn("Metal data buffers requested, but Metal is unavailable on this machine")
                        return .success
                    }
                    let buffers = try MetalBufferManager().makeReadOnlyBuffers(series: series)
                    logger.ok("Metal read-only OHLC buffers prepared on \(buffers.deviceName) with \(buffers.count) rows")
                    #else
                    logger.warn("Metal data buffers requested, but this Swift toolchain cannot import Metal")
                    #endif
                }
                return .success

            case .sineTestSync:
                try await runSineTestSync(
                    config: config,
                    clickHouse: clickHouse,
                    logger: logger,
                    watch: options.watchSineTestSync
                )
                return .success

            case .fxBacktestAPI:
                let historyService = FXDatabaseBacktestHistoryService(config: config, clickHouse: clickHouse)
                let resultService = FXDatabaseBacktestResultService(clickHouse: clickHouse, database: config.clickHouse.database)
                let handler = FXBacktestAPIHTTPHandler(historyProvider: historyService, resultProvider: resultService)
                try await FXBacktestAPIServer(
                    host: options.apiHost,
                    port: options.apiPort,
                    handler: handler,
                    logger: logger
                ).run()
                return .success

            case .healthAPI:
                let service = OperationalHealthService(config: config, clickHouse: clickHouse)
                try await OperationalHealthServer(
                    host: options.apiHost,
                    port: options.apiPort,
                    service: service,
                    logger: logger
                ).run()
                return .success

            case .backtest:
                throw CLIError.commandUnavailable(options.command.unavailableReason ?? "Command unavailable.")

            case .optimize:
                throw CLIError.commandUnavailable(options.command.unavailableReason ?? "Command unavailable.")

            case .help:
                printUsage()
                return .success

            case .interactive:
                print("FXDatabase is already running in the interactive command shell.")
                return .success
            }
        } catch let error as CLIError {
            print("[ERROR] \(error.description)")
            printUsage()
            return .usage
        } catch let error as ConfigError {
            print("[ERROR] Configuration problem")
            print(OperationalFailureGuide.advice(for: error).formatted)
            return .configuration
        } catch let error as TerminalIdentityPolicyError {
            print("[ERROR] MT5 terminal identity problem")
            print(OperationalFailureGuide.advice(for: error).formatted)
            activeLogger?.alert("MT5 terminal identity problem", details: error.description)
            return .configuration
        } catch let error as BrokerOffsetRuntimeError {
            print("[ERROR] Broker UTC offset problem")
            print(OperationalFailureGuide.advice(for: error).formatted)
            activeLogger?.alert("Broker UTC offset problem", details: error.description)
            return .configuration
        } catch let error as ClickHouseStartupError {
            print("[ERROR] ClickHouse startup check failed")
            print(OperationalFailureGuide.advice(for: error).formatted)
            activeLogger?.alert("ClickHouse startup check failed", details: error.description)
            return .clickHouse
        } catch let error as SupervisorError {
            print("[ERROR] Runtime lock: \(error.description)")
            print(OperationalFailureGuide.advice(for: error).formatted)
            activeLogger?.alert("Runtime lock unavailable", details: error.description)
            return .configuration
        } catch let error as BacktestReadinessError {
            print("[ERROR] History data readiness blocked")
            print(OperationalFailureGuide.advice(for: error).formatted)
            activeLogger?.alert("History data readiness blocked", details: error.description)
            return .backtest
        } catch let error as HistoryDataError {
            print("[ERROR] History data load failed")
            print(error.description)
            activeLogger?.alert("History data load failed", details: error.description)
            return .backtest
        } catch let error as TimeMappingError {
            print("[ERROR] Broker UTC offset problem")
            print(OperationalFailureGuide.advice(for: error).formatted)
            activeLogger?.alert("Broker UTC offset problem", details: error.description)
            return .configuration
        } catch let error as VerificationError {
            print("[ERROR] Verification failed")
            print(OperationalFailureGuide.advice(for: error).formatted)
            activeLogger?.alert("Verification failed", details: error.description)
            return .verification
        } catch let error as MT5BridgeStartupError {
            print("[ERROR] MT5 bridge startup check failed")
            print(error.description)
            activeLogger?.alert("MT5 bridge startup check failed", details: error.description)
            return .mt5Bridge
        } catch let error as ProtocolError {
            print("[ERROR] MT5 bridge protocol check failed")
            print(OperationalFailureGuide.advice(for: error).formatted)
            activeLogger?.alert("MT5 bridge protocol check failed", details: error.description)
            return .mt5Bridge
        } catch let error as MT5BridgeError {
            print("[ERROR] MT5 bridge is not ready")
            print(OperationalFailureGuide.advice(for: error).formatted)
            activeLogger?.alert("MT5 bridge is not ready", details: error.description)
            return .mt5Bridge
        } catch let error as ClickHouseError {
            print("[ERROR] ClickHouse operation failed")
            print(OperationalFailureGuide.advice(for: error).formatted)
            activeLogger?.alert("ClickHouse operation failed", details: error.description)
            return .clickHouse
        } catch is CancellationError {
            activeLogger?.info("Command cancelled gracefully")
            return .success
        } catch {
            print("[ERROR] \(error)")
            activeLogger?.alert("Unexpected command failure", details: String(describing: error))
            return .unknown
        }
    }
}
