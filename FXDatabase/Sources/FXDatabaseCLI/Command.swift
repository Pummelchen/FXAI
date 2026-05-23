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

enum Command: Equatable {
    case interactive
    case migrate
    case bridgeCheck
    case symbolCheck
    case backfill
    case live
    case supervise
    case startcheck
    case failureGuide
    case verify
    case repair
    case exportCache
    case dataCheck
    case fxBacktestAPI
    case healthAPI
    case backtest
    case optimize
    case help
}

extension Command {
    var requiresClickHouseStartupCheck: Bool {
        switch self {
        case .help, .interactive, .failureGuide, .symbolCheck, .exportCache, .backtest, .optimize:
            return false
        case .migrate, .bridgeCheck, .backfill, .live, .supervise, .startcheck, .verify, .repair, .dataCheck, .fxBacktestAPI, .healthAPI:
            return true
        }
    }

    var unavailableReason: String? {
        switch self {
        case .exportCache:
            return "export-cache is intentionally disabled. FXDatabase reads verified canonical bars directly from ClickHouse so stale local caches cannot survive verifier repairs."
        case .backtest:
            return "backtest has been removed from FXDatabase. Use fxbacktest-api to serve verified history, then run strategies in FXBacktest or another external Swift app through FXBacktest API v1."
        case .optimize:
            return "optimize has been removed from FXDatabase. Long-running optimization belongs in an external Swift app with its own durable job model; FXDatabase only serves verified OHLC data."
        case .interactive, .migrate, .bridgeCheck, .symbolCheck, .backfill, .live, .supervise, .startcheck, .failureGuide, .verify, .repair, .dataCheck, .fxBacktestAPI, .healthAPI, .help:
            return nil
        }
    }
}
