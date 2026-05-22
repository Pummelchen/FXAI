import FXBacktestCore
import FXBacktestPlugins
import XCTest

final class CPUBacktestExecutorTests: XCTestCase {
    func testCPUExecutorRunsWholePassChunksAndEmitsResults() async throws {
        let market = try OhlcDataSeries.demoEURUSD(barCount: 500)
        let plugin = AnyFXBacktestPlugin(FX7())
        let sweep = try ParameterSweep.singlePass(definitions: plugin.parameterDefinitions)
        let settings = BacktestRunSettings(target: .cpu, maxWorkers: 2, chunkSize: 2)

        var passResults: [BacktestPassResult] = []
        var completed: BacktestProgress?
        for try await event in CPUBacktestExecutor().run(plugin: plugin, market: market, sweep: sweep, settings: settings) {
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
        XCTAssertEqual(completed?.completedPasses, sweep.combinationCount)
        XCTAssertTrue(passResults.allSatisfy { $0.barsProcessed == market.count })
        XCTAssertEqual(Set(passResults.map(\.passIndex)).count, Int(sweep.combinationCount))
    }
}
