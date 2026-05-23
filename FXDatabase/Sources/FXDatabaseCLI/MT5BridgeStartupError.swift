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

struct MT5BridgeStartupError: Error, CustomStringConvertible {
    let error: MT5BridgeError
    let config: MT5BridgeConfig

    var description: String {
        switch error {
        case .bindFailed:
            return """
            Swift could not open the MT5 bridge listener on \(config.host):\(config.port).
            Reason: \(error.description)
            Next steps:
              1. Check what owns the port: lsof -nP -iTCP:\(config.port) -sTCP:LISTEN
              2. Stop the other FXDatabase process, or change Config/mt5_bridge.json to another free port.
              3. Reattach FXDatabase with the same SwiftHost/SwiftPort values.
              4. At the FXDatabase prompt run: startcheck --config-dir Config --migrations-dir Migrations
            """
        case .acceptTimedOut:
            return listenModeGuidance(reason: error.description)
        case .connectFailed:
            return connectModeGuidance(reason: error.description)
        case .invalidHost:
            return """
            The MT5 bridge host in Config/mt5_bridge.json is invalid.
            Reason: \(error.description)
            Next steps:
              1. For local MT5/Wine, set host to 127.0.0.1.
              2. Keep the EA input SwiftHost exactly the same.
              3. At the FXDatabase prompt run: startcheck --config-dir Config --migrations-dir Migrations
            """
        default:
            switch config.mode {
            case .listen:
                return listenModeGuidance(reason: error.description)
            case .connect:
                return connectModeGuidance(reason: error.description)
            }
        }
    }

    private func listenModeGuidance(reason: String) -> String {
        """
        MT5 did not connect to the Swift listener at \(config.host):\(config.port).
        Reason: \(reason)
        Next steps:
          1. Start MetaTrader 5 under Wine.
          2. Attach the compiled FXDatabase EA to any chart.
          3. In the EA inputs set SwiftHost=\(config.host) and SwiftPort=\(config.port).
          4. Enable Algo Trading and allow localhost/socket access in MT5/Wine when prompted.
          5. Leave this FXDatabase session running while the EA connects, or run at the prompt: startcheck --config-dir Config --migrations-dir Migrations
        """
    }

    private func connectModeGuidance(reason: String) -> String {
        """
        Swift could not connect to the MT5 bridge at \(config.host):\(config.port).
        Reason: \(reason)
        Next steps:
          1. Confirm the MT5 EA bridge is already listening on \(config.host):\(config.port), or switch Config/mt5_bridge.json mode to "listen".
          2. Check the port: lsof -nP -iTCP:\(config.port)
          3. Confirm macOS/Wine firewall prompts are allowed.
          4. At the FXDatabase prompt run: startcheck --config-dir Config --migrations-dir Migrations
        """
    }
}
