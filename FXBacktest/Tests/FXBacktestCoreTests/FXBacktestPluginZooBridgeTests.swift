import FXBacktestCore
import FXBacktestPlugins
import Testing

@Suite("FXBacktest root plugin zoo bridge")
struct FXBacktestPluginZooBridgeTests {
    @Test func registryExposesRootPluginZoo() throws {
        let plugins = FXBacktestPluginRegistry.availablePlugins
        let ids = Set(plugins.map(\.descriptor.id))

        #expect(plugins.count >= 60)
        #expect(ids.contains("rule_buyonly"))
        #expect(ids.contains("fxbacktest_moving_average_cross"))
        #expect(ids.contains("fx7"))
    }

    @Test func rootPluginRunsThroughBacktestAdapter() throws {
        let plugin = try #require(FXBacktestPluginRegistry.availablePlugins.first { $0.descriptor.id == "rule_buyonly" })
        let market = try OhlcDataSeries.demoEURUSD(barCount: 320)
        let sweep = try ParameterSweep.singlePass(definitions: plugin.parameterDefinitions)
        let parameters = try sweep.parameterVector(at: 0)
        let context = BacktestContext(settings: BacktestRunSettings(target: .cpu), digits: market.metadata.digits)

        let result = try plugin.runPass(market: market, parameters: parameters, context: context)

        #expect(result.pluginIdentifier == "rule_buyonly")
        #expect(result.barsProcessed == market.count)
        #expect(result.totalTrades >= 0)
        #expect(result.errorMessage == nil)
    }
}
