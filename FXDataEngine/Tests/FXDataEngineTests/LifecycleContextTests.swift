import XCTest
@testable import FXDataEngine

final class LifecycleContextTests: XCTestCase {
    func testContextSymbolCategoriesAndPrioritiesMatchLegacyRules() {
        XCTAssertEqual(LifecycleContextTools.category(symbol: "XAUUSD"), .metal)
        XCTAssertEqual(LifecycleContextTools.category(symbol: "NAS100"), .index)
        XCTAssertEqual(LifecycleContextTools.category(symbol: "BRENT"), .energy)
        XCTAssertEqual(LifecycleContextTools.category(symbol: "BTCUSD"), .crypto)
        XCTAssertEqual(LifecycleContextTools.category(symbol: "DXY"), .risk)
        XCTAssertEqual(LifecycleContextTools.category(symbol: "EURUSD"), .fx)
        XCTAssertEqual(LifecycleContextTools.category(symbol: "ABC12"), .other)
        XCTAssertEqual(LifecycleContextTools.categoryPriority(.risk), 0.94, accuracy: 1e-12)
    }

    func testContextSharedSessionAndRedundancyScoresMatchLegacyMath() {
        XCTAssertEqual(
            LifecycleContextTools.sharedSymbolScore(mainSymbol: "EURUSD", candidateSymbol: "EURJPY"),
            0.45,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            LifecycleContextTools.sessionOverlapScore(mainSymbol: "EURUSD", candidateSymbol: "USDJPY", sessionBucket: 0, hour: 22),
            0.82,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            LifecycleContextTools.redundancyPenalty(mainSymbol: "EURUSD", candidateSymbol: "GBPUSD", selected: ["USDJPY"]),
            0.2025,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            LifecycleContextTools.incrementalValueScore(mainSymbol: "EURUSD", candidateSymbol: "GBPUSD", selected: ["USDJPY"]),
            0.55425,
            accuracy: 1e-12
        )
    }

    func testContextCandidateScoreUsesPreparedProviderInputs() {
        let score = LifecycleContextTools.candidateScore(
            mainSymbol: "EURUSD",
            candidateSymbol: "GBPUSD",
            selected: ["USDJPY"],
            inputs: LifecycleContextCandidateInputs(
                liquidityScore: 0.80,
                dataHealthScore: 0.90,
                sessionOverlapScore: 0.75
            )
        )

        XCTAssertEqual(score, 1.6807, accuracy: 1e-12)
    }

    func testContextParsingReferenceAndCuratedUniverseAreDeterministic() {
        XCTAssertEqual(
            LifecycleContextTools.parseContextSymbols("{ eurusd; USDJPY|xauusd, eurusd }"),
            ["EURUSD", "USDJPY", "XAUUSD"]
        )
        XCTAssertEqual(
            LifecycleContextTools.scoreReference(selected: ["eurusd", "USDJPY"], pending: ["EURUSD", "XAUUSD"]),
            ["EURUSD", "USDJPY", "XAUUSD"]
        )
        XCTAssertFalse(LifecycleContextTools.curatedContextUniverse(mainSymbol: "EURUSD").contains("EURUSD"))
        XCTAssertTrue(LifecycleContextTools.curatedContextUniverse(mainSymbol: "EURUSD").contains("GBPUSD"))
    }
}
