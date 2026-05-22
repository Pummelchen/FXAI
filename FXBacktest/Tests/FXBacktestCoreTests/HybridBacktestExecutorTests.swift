import FXBacktestCore
import FXBacktestPlugins
import XCTest

#if canImport(Metal)
import Metal
#endif

final class HybridBacktestExecutorTests: XCTestCase {
    func testHybridExecutorRunsEachPassOnceAcrossCPUAndMetal() async throws {
        #if canImport(Metal)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is unavailable on this machine.")
        }

        let market = try OhlcDataSeries.demoEURUSD(barCount: 500)
        let plugin = AnyFXBacktestPlugin(FX7())
        let sweep = try ParameterSweep(dimensions: plugin.parameterDefinitions.enumerated().map { index, definition in
            let maximum = index == 0 ? definition.defaultValue + definition.defaultStep : definition.defaultValue
            return try ParameterSweepDimension(
                definition: definition,
                input: definition.defaultValue,
                minimum: definition.defaultValue,
                step: max(definition.defaultStep, 1),
                maximum: maximum
            )
        })
        let settings = BacktestRunSettings(target: .both, maxWorkers: 2, chunkSize: 8)

        var passResults: [BacktestPassResult] = []
        var completed: BacktestProgress?
        for try await event in HybridBacktestExecutor().run(plugin: plugin, marketUniverse: market.universe, sweep: sweep, settings: settings) {
            switch event {
            case .started:
                break
            case .passCompleted(let result, _):
                passResults.append(result)
            case .completed(let progress):
                completed = progress
            }
        }

        XCTAssertEqual(passResults.count, Int(sweep.combinationCount))
        XCTAssertEqual(Set(passResults.map(\.passIndex)).count, Int(sweep.combinationCount))
        XCTAssertEqual(completed?.completedPasses, sweep.combinationCount)
        XCTAssertTrue(passResults.contains { $0.engine == .cpu })
        XCTAssertTrue(passResults.contains { $0.engine == .metal })
        XCTAssertFalse(passResults.contains { $0.engine == .both })
        #else
        throw XCTSkip("This toolchain cannot import Metal.")
        #endif
    }
}
