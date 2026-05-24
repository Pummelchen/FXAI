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

struct CLIOptions {
    let command: Command
    let configDirectory: URL
    let migrationsDirectory: URL
    let overrideLogLevel: LogLevel?
    let symbolsArgument: String?
    let randomRanges: Int?
    let repairSymbol: LogicalSymbol?
    let fromUtcDay: UtcSecond?
    let toUtcDay: UtcSecond?
    let noBridgeRequested: Bool
    let runBackfillOnStart: Bool?
    let supervisorCycles: Int?
    let compileEA: Bool
    let bridgeChecks: Bool
    let compileTimeoutSeconds: TimeInterval
    let commandConfigPath: URL?
    let apiHost: String
    let apiPort: UInt16
    let watchSineTestSync: Bool

    init(arguments: [String]) throws {
        guard let first = arguments.first else {
            self.command = .help
            self.configDirectory = URL(fileURLWithPath: "Config")
            self.migrationsDirectory = URL(fileURLWithPath: "Migrations")
            self.overrideLogLevel = nil
            self.symbolsArgument = nil
            self.randomRanges = nil
            self.repairSymbol = nil
            self.fromUtcDay = nil
            self.toUtcDay = nil
            self.noBridgeRequested = false
            self.runBackfillOnStart = nil
            self.supervisorCycles = nil
            self.compileEA = true
            self.bridgeChecks = true
            self.compileTimeoutSeconds = 120
            self.commandConfigPath = nil
            self.apiHost = "127.0.0.1"
            self.apiPort = 5066
            self.watchSineTestSync = false
            return
        }

        self.command = try Self.parseCommand(first)
        var configDirectory = URL(fileURLWithPath: "Config")
        var migrationsDirectory = URL(fileURLWithPath: "Migrations")
        var overrideLogLevel: LogLevel?
        var symbolsArgument: String?
        var randomRanges: Int?
        var repairSymbol: LogicalSymbol?
        var fromUtcDay: UtcSecond?
        var toUtcDay: UtcSecond?
        var noBridgeRequested = false
        var runBackfillOnStart: Bool?
        var supervisorCycles: Int?
        var compileEA = true
        var bridgeChecks = true
        var compileTimeoutSeconds: TimeInterval = 120
        var commandConfigPath: URL?
        var apiHost = "127.0.0.1"
        var apiPort: UInt16 = 5066
        var watchSineTestSync = false

        var index = 1
        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--config-dir":
                index += 1
                guard index < arguments.count else { throw CLIError.missingValue(arg) }
                configDirectory = URL(fileURLWithPath: arguments[index])
            case "--migrations-dir":
                index += 1
                guard index < arguments.count else { throw CLIError.missingValue(arg) }
                migrationsDirectory = URL(fileURLWithPath: arguments[index])
            case "--symbols":
                index += 1
                guard index < arguments.count else { throw CLIError.missingValue(arg) }
                symbolsArgument = arguments[index]
            case "--random-ranges":
                index += 1
                guard index < arguments.count, let value = Int(arguments[index]), value >= 0 else {
                    throw CLIError.invalidValue(arg)
                }
                randomRanges = value
            case "--symbol":
                index += 1
                guard index < arguments.count else { throw CLIError.missingValue(arg) }
                repairSymbol = try LogicalSymbol(arguments[index])
            case "--from":
                index += 1
                guard index < arguments.count else { throw CLIError.missingValue(arg) }
                fromUtcDay = try FXDatabaseCLI.parseUtcDay(arguments[index])
            case "--to":
                index += 1
                guard index < arguments.count else { throw CLIError.missingValue(arg) }
                toUtcDay = try FXDatabaseCLI.parseUtcDay(arguments[index])
            case "--no-bridge":
                noBridgeRequested = true
            case "--with-backfill":
                runBackfillOnStart = true
            case "--without-backfill":
                runBackfillOnStart = false
            case "--supervisor-cycles":
                index += 1
                guard index < arguments.count, let value = Int(arguments[index]), value > 0 else {
                    throw CLIError.invalidValue(arg)
                }
                supervisorCycles = value
            case "--skip-ea-compile":
                compileEA = false
            case "--skip-bridge":
                bridgeChecks = false
            case "--compile-timeout-seconds":
                index += 1
                guard index < arguments.count, let value = TimeInterval(arguments[index]), value > 0, value <= 1800 else {
                    throw CLIError.invalidValue(arg)
                }
                compileTimeoutSeconds = value
            case "--verbose":
                overrideLogLevel = .verbose
            case "--debug":
                overrideLogLevel = .debug
            case "--config":
                index += 1
                guard index < arguments.count else { throw CLIError.missingValue(arg) }
                commandConfigPath = URL(fileURLWithPath: arguments[index])
            case "--api-host":
                index += 1
                guard index < arguments.count else { throw CLIError.missingValue(arg) }
                apiHost = arguments[index]
            case "--api-port":
                index += 1
                guard index < arguments.count, let value = UInt16(arguments[index]), value > 0 else {
                    throw CLIError.invalidValue(arg)
                }
                apiPort = value
            case "--watch":
                watchSineTestSync = true
            default:
                throw CLIError.unknownOption(arg)
            }
            index += 1
        }

        if command == .repair, let fromUtcDay, let toUtcDay, fromUtcDay.rawValue >= toUtcDay.rawValue {
            throw CLIError.invalidValue("--from/--to")
        }
        if commandConfigPath != nil && command != .dataCheck && command != .backtest && command != .optimize {
            throw CLIError.invalidValue("--config")
        }
        if (apiHost != "127.0.0.1" || apiPort != 5066) && command != .fxBacktestAPI && command != .healthAPI {
            throw CLIError.invalidValue("--api-host/--api-port")
        }
        if watchSineTestSync && command != .sineTestSync {
            throw CLIError.invalidValue("--watch")
        }

        self.configDirectory = configDirectory
        self.migrationsDirectory = migrationsDirectory
        self.overrideLogLevel = overrideLogLevel
        self.symbolsArgument = symbolsArgument
        self.randomRanges = randomRanges
        self.repairSymbol = repairSymbol
        self.fromUtcDay = fromUtcDay
        self.toUtcDay = toUtcDay
        self.noBridgeRequested = noBridgeRequested
        self.runBackfillOnStart = runBackfillOnStart
        self.supervisorCycles = supervisorCycles
        self.compileEA = compileEA
        self.bridgeChecks = bridgeChecks
        self.compileTimeoutSeconds = compileTimeoutSeconds
        self.commandConfigPath = commandConfigPath
        self.apiHost = apiHost
        self.apiPort = apiPort
        self.watchSineTestSync = watchSineTestSync
    }

    private static func parseCommand(_ value: String) throws -> Command {
        switch value {
        case "shell", "interactive", "console": return .interactive
        case "migrate": return .migrate
        case "bridge-check": return .bridgeCheck
        case "symbol-check": return .symbolCheck
        case "backfill": return .backfill
        case "live": return .live
        case "supervise": return .supervise
        case "startcheck", "-startcheck", "--startcheck": return .startcheck
        case "failure-guide": return .failureGuide
        case "verify": return .verify
        case "repair": return .repair
        case "export-cache": return .exportCache
        case "data-check": return .dataCheck
        case "sinetest-sync", "sine-sync": return .sineTestSync
        case "fxbacktest-api": return .fxBacktestAPI
        case "health-api": return .healthAPI
        case "backtest": return .backtest
        case "optimize": return .optimize
        case "help", "--help", "-h": return .help
        default: throw CLIError.unknownCommand(value)
        }
    }

    func shouldConnectBridgeForVerify(randomRangeCount: Int) -> Bool {
        !noBridgeRequested && randomRangeCount > 0
    }
}
