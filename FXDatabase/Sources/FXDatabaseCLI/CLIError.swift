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

enum CLIError: Error, CustomStringConvertible {
    case unknownCommand(String)
    case unknownOption(String)
    case missingValue(String)
    case invalidValue(String)
    case commandUnavailable(String)

    var description: String {
        switch self {
        case .unknownCommand(let value):
            return "Unknown command '\(value)'."
        case .unknownOption(let value):
            return "Unknown option '\(value)'."
        case .missingValue(let option):
            return "Missing value for \(option)."
        case .invalidValue(let option):
            return "Invalid value for \(option)."
        case .commandUnavailable(let reason):
            return reason
        }
    }
}
