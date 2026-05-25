import FXAIPlugins
import FXBacktestAPI
import FXBacktestPlugins
import XCTest

final class PluginBacktestConfigurationCatalogTests: XCTestCase {
    func testConfigurationCatalogRegistersEveryPluginAcceleratorScope() throws {
        let runtimes = FXAIPluginRegistry.acceleratedRuntimes()
        let configurations = FXBacktestPluginConfigurationCatalog.pluginConfigurations(plugins: runtimes)
        let expectedScopes = runtimes.flatMap { runtime in
            runtime.accelerationPlan.declaredBackends.map { "\(runtime.manifest.aiName):\($0.rawValue)" }
        }
        let actualScopes = configurations.map { "\($0.pluginId):\($0.acceleratorId)" }

        XCTAssertEqual(Set(actualScopes), Set(expectedScopes))
        XCTAssertEqual(actualScopes.count, expectedScopes.count)
        XCTAssertFalse(configurations.isEmpty)

        for configuration in configurations {
            try configuration.validate()
            XCTAssertFalse(configuration.parameters.isEmpty)
            XCTAssertTrue(configuration.parameters.allSatisfy { $0.defaultValue >= $0.minimum && $0.defaultValue <= $0.maximum })
        }
    }
}
