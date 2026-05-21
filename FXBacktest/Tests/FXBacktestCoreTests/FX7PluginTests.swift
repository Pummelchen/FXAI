import FXBacktestCore
import FXBacktestPlugins
import XCTest

#if canImport(Metal)
import Metal
#endif

final class FX7PluginTests: XCTestCase {
    func testFX7PluginIsRegisteredAndMetalCompatible() throws {
        let plugin = FX7()

        XCTAssertEqual(plugin.descriptor.displayName, "FX7")
        XCTAssertTrue(plugin.descriptor.supportsCPU)
        XCTAssertTrue(plugin.descriptor.supportsMetal)
        XCTAssertNotNil(plugin.metalKernel)
        XCTAssertTrue(FXBacktestPluginRegistry.availablePlugins.contains { $0.descriptor.id == plugin.descriptor.id })
        XCTAssertNoThrow(try PluginAccelerationPipeline().validate(plugin.accelerationDescriptor))
    }

    func testFX7RunsClosedBarSingleSymbolTrend() throws {
        let plugin = FX7()
        let market = try trendingMarket(symbol: "EURUSD", slope: 18)
        let parameters = try defaultParameters(plugin)
        let context = BacktestContext(settings: BacktestRunSettings(lotSize: 0.01), digits: market.metadata.digits)

        let result = try plugin.runPass(market: market, parameters: parameters, context: context)

        XCTAssertEqual(result.pluginIdentifier, plugin.descriptor.id)
        XCTAssertEqual(result.barsProcessed, market.count)
        XCTAssertGreaterThan(result.totalTrades, 0)
        XCTAssertGreaterThan(result.netProfit, 0)
    }

    func testFX7RunsMultiSymbolUniverse() throws {
        let plugin = FX7()
        let eurusd = try trendingMarket(symbol: "EURUSD", slope: 16)
        let usdjpy = try trendingMarket(symbol: "USDJPY", slope: -12, digits: 3, basePrice: 150_000)
        let universe = try OhlcMarketUniverse(primarySymbol: "EURUSD", series: [eurusd, usdjpy])
        let parameters = try defaultParameters(plugin)
        let context = BacktestContext(settings: BacktestRunSettings(lotSize: 0.01), digits: eurusd.metadata.digits)

        let result = try plugin.runPass(marketUniverse: universe, parameters: parameters, context: context)

        XCTAssertEqual(result.barsProcessed, universe.count)
        XCTAssertGreaterThanOrEqual(result.totalTrades, 2)
    }

    func testFX7UsesSignalTimeframeBarsForWarmup() throws {
        let plugin = FX7()
        let market = try trendingMarket(symbol: "EURUSD", slope: 20, count: 1_200)
        let context = BacktestContext(settings: BacktestRunSettings(lotSize: 0.01), digits: market.metadata.digits)

        let m15Result = try plugin.runPass(market: market, parameters: try defaultParameters(plugin), context: context)
        let m1Result = try plugin.runPass(
            market: market,
            parameters: try defaultParameters(plugin, overrides: ["signal_stride_bars": 1]),
            context: context
        )

        XCTAssertEqual(m15Result.totalTrades, 0)
        XCTAssertGreaterThan(m1Result.totalTrades, 0)
    }

    func testFX7HonorsLongShortDirectionInputs() throws {
        let plugin = FX7()
        let market = try trendingMarket(symbol: "EURUSD", slope: 20)
        let context = BacktestContext(settings: BacktestRunSettings(lotSize: 0.01), digits: market.metadata.digits)

        let longOnly = try plugin.runPass(
            market: market,
            parameters: try defaultParameters(plugin, overrides: ["allow_short": 0]),
            context: context
        )
        let noLongs = try plugin.runPass(
            market: market,
            parameters: try defaultParameters(plugin, overrides: ["allow_long": 0]),
            context: context
        )

        XCTAssertGreaterThan(longOnly.totalTrades, 0)
        XCTAssertEqual(noLongs.totalTrades, 0)
    }

    func testFX7MetalKernelRunsSingleSymbolPass() async throws {
        #if canImport(Metal)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is unavailable on this machine.")
        }

        let market = try trendingMarket(symbol: "EURUSD", slope: 18)
        let plugin = AnyFXBacktestPlugin(FX7())
        let sweep = try ParameterSweep.singlePass(definitions: plugin.parameterDefinitions)
        let settings = BacktestRunSettings(target: .metal, maxWorkers: 1, chunkSize: 1, lotSize: 0.01)

        var passResults: [BacktestPassResult] = []
        for try await event in MetalBacktestExecutor().run(plugin: plugin, market: market, sweep: sweep, settings: settings) {
            if case .passCompleted(let result, _) = event {
                passResults.append(result)
            }
        }

        XCTAssertEqual(passResults.count, 1)
        XCTAssertEqual(passResults.first?.engine, .metal)
        XCTAssertEqual(passResults.first?.barsProcessed, market.count)
        XCTAssertGreaterThan(passResults.first?.totalTrades ?? 0, 0)
        XCTAssertGreaterThan(passResults.first?.netProfit ?? 0, 0)
        #else
        throw XCTSkip("This toolchain cannot import Metal.")
        #endif
    }

    private func defaultParameters(_ plugin: FX7, overrides: [String: Double] = [:]) throws -> ParameterVector {
        let names = plugin.parameterDefinitions.map(\.key)
        let values = plugin.parameterDefinitions.map { overrides[$0.key] ?? $0.defaultValue }
        return ParameterVector(combinationIndex: 0, names: names, values: ContiguousArray(values))
    }

    private func trendingMarket(symbol: String, slope: Int, digits: Int = 5, basePrice: Int = 108_000, count: Int = 6_000) throws -> OhlcDataSeries {
        let start = Int64(1_704_067_200)
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var price = basePrice

        for index in 0..<count {
            let wave = (index % 17) - 8
            let next = max(10_000, price + slope + wave)
            let barHigh = max(price, next) + 5
            let barLow = min(price, next) - 5
            utc.append(start + Int64(index * 60))
            open.append(Int64(price))
            high.append(Int64(barHigh))
            low.append(Int64(barLow))
            close.append(Int64(next))
            price = next
        }

        return try OhlcDataSeries(
            metadata: FXBacktestMarketMetadata(
                brokerSourceId: "demo",
                logicalSymbol: symbol,
                digits: digits,
                firstUtc: utc.first,
                lastUtc: utc.last
            ),
            utcTimestamps: utc,
            open: open,
            high: high,
            low: low,
            close: close
        )
    }
}
