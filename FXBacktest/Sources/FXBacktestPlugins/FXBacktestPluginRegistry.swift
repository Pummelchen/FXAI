import FXBacktestCore
import FXAIPlugins
import Foundation

public enum FXBacktestPluginRegistry {
    public static let availablePlugins: [AnyFXBacktestPlugin] = rootPluginZoo()

    private static func rootPluginZoo() -> [AnyFXBacktestPlugin] {
        let runtimes = FXAIPluginRegistry.acceleratedRuntimes(
            configuration: FXAIPluginRuntimeConfiguration(mode: .cpuOnly)
        )
        let rootPlugins = runtimes.compactMap { runtime -> AnyFXBacktestPlugin? in
            guard let adapter = try? FXAIPluginBacktestAdapter(runtime: runtime) else {
                return nil
            }
            return AnyFXBacktestPlugin(adapter)
        }
        guard !rootPlugins.isEmpty else {
            return [AnyFXBacktestPlugin(FX7())]
        }
        return rootPlugins
    }
}
