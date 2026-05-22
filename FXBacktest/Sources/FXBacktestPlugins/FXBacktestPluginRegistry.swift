import FXBacktestCore
import Foundation

public enum FXBacktestPluginRegistry {
    public static let availablePlugins: [AnyFXBacktestPlugin] = [
        AnyFXBacktestPlugin(FX7())
    ]
}
