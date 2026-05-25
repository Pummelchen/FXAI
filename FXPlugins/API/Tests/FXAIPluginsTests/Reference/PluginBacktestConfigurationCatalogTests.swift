import FXAIPlugins
import FXBacktestAPI
import FXDataEngine
import XCTest

final class PluginBacktestConfigurationCatalogTests: XCTestCase {
    func testConfigurationCatalogRegistersEveryPluginAcceleratorScope() throws {
        let plugins = FXAIPluginRegistry.availablePlugins()
        let configurations = FXAIPluginBacktestConfigurationCatalog.pluginConfigurations(plugins: plugins)
        let expectedScopes = plugins.compactMap { $0 as? any FXAIPlannedPlugin }.flatMap { plugin in
            plugin.accelerationPlan.declaredBackends.map { "\(plugin.manifest.aiName):\($0.rawValue)" }
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

    func testDemoPluginTemplateIsConfigurationCompleteButNotInRuntimeRegistry() throws {
        let template = DemoPluginTemplate(aiID: 0)
        XCTAssertTrue(template.configurationRows().count >= 5)
        XCTAssertFalse(FXAIPluginRegistry.availablePlugins().contains { $0.manifest.aiName == "demo_plugin_template" })
        for row in template.configurationRows() {
            try row.validate()
            XCTAssertEqual(row.parameters.map(\.key), ["lookback_bars", "confidence_floor", "use_volume_when_available"])
        }
    }
}
