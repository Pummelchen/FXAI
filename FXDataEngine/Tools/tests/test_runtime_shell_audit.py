from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def _assert_tokens(text: str, tokens: list[str]) -> None:
    for token in tokens:
        assert token in text


def test_cycle_recovery_stays_symbol_scoped_in_swift_runtime():
    runtime_cycle = _read("FXDataEngine/Sources/FXDataEngine/Runtime/RuntimeCycle.swift")
    lifecycle_reset = _read("FXDataEngine/Sources/FXDataEngine/Lifecycle/LifecycleReset.swift")
    runtime_tests = _read("FXDataEngine/Tests/FXDataEngineTests/RuntimeCycleTests.swift")

    _assert_tokens(
        runtime_cycle,
        [
            "public var symbol: String",
            "public var lastSymbol: String",
            "let requiresStateReset = input.lastSymbol != input.symbol",
            "reason: \"signal_cache_hit\"",
            "reason: \"continue\"",
        ],
    )
    _assert_tokens(
        lifecycle_reset,
        [
            "public var symbol: String",
            "public static func buildResetPlan(",
            "symbol: symbol",
            "signalCache: RuntimeSignalCache.reset",
            "warmupDone: !aiWarmupEnabled",
        ],
    )
    _assert_tokens(
        runtime_tests,
        [
            "lastSymbol: \"GBPUSD\"",
            "XCTAssertTrue(plan.requiresStateReset)",
            "XCTAssertEqual(plan.reason, \"signal_cache_hit\")",
        ],
    )


def test_order_request_lifecycle_is_explicit_in_swift_trade_plan():
    execution_plan = _read("FXDataEngine/Sources/FXDataEngine/Runtime/TradeExecutionPlan.swift")
    system_health = _read("FXDataEngine/Sources/FXDataEngine/Runtime/SystemHealth.swift")
    execution_tests = _read("FXDataEngine/Tests/FXDataEngineTests/TradeExecutionPlanTests.swift")

    _assert_tokens(
        execution_plan,
        [
            "public var previousOrderRequestPending: Bool",
            "previousOrderRequestPending: Bool = false",
            "self.previousOrderRequestPending = previousOrderRequestPending",
            "previousOrderRequestPending: Bool",
            "return previousOrderRequestPending",
            "plan.usePending = true",
            "plan.usePending = false",
            "plan.expiryTimeUTC = input.market.generatedAtUTC > 0 ? input.market.generatedAtUTC + 20 * 60 : 0",
        ],
    )
    _assert_tokens(
        system_health,
        [
            "public struct SystemHealthState",
            "public enum SystemHealthPosture",
            "public static func refresh(",
        ],
    )
    _assert_tokens(
        execution_tests,
        [
            "newsPulseTradeGate: \"CAUTION\"",
            "XCTAssertTrue(plan.usePending)",
            "XCTAssertEqual(plan.mode, \"SELL_STOP\")",
        ],
    )
