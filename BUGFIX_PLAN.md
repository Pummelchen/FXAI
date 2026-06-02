# FXAI Bugfix Plan

Created: 2026-06-02

Source report: `/Users/andreborchert/.codex/attachments/52495dc1-cf79-43e6-8c89-9fc1991de9a1/pasted-text.txt`

This plan includes only findings that still match the current codebase after validation. Line numbers in the source report were treated as hints and rechecked by symbol/function name.

## Validation Summary

- Still valid or valid hardening item: 14
- Rejected as not justified by current code semantics: 2
- Code changes made in this pass: implemented and covered by focused regression tests.

## P0 / Correctness

### 1. Break-even trades are counted as wins

Status: fixed.

Current code increments `winningTrades` when `pnl >= 0` in both broker implementations:

- `FXBacktest/Sources/FXBacktestCore/BacktestBroker.swift`
- `FXBacktest/Sources/FXBacktestCore/ExecutionModel.swift`

Impact: break-even closes inflate `winRate`, which can distort optimizer result ranking and operator interpretation.

Fix plan:

1. Change win/loss classification to `pnl > 0` for wins and `pnl < 0` for losses.
2. Leave break-even trades counted in `totalTrades`.
3. Add tests for positive, negative, and zero-PnL closes in both broker implementations.
4. Review any UI text that assumes `winningTrades + losingTrades == totalTrades`.

## P1 / Backtest And Operator Metrics

### 2. `netProfit` semantics differ between broker implementations

Status: fixed. The realized/equity distinction is now explicit through `netProfit` and `equityNetProfit`.

Current code:

- `BacktestBroker.netProfit` returns `balance - initialDeposit`.
- `BacktestBrokerV2.netProfit` returns `equity - initialDeposit`.

Impact: realized-only and floating-PnL-inclusive values can diverge during a run, making metrics incomparable if both broker types feed results or UI state.

Fix plan:

1. Define explicit semantics in `BacktestResult` and broker APIs.
2. Prefer `netProfit` for finalized realized result after forced close.
3. Add `equityNetProfit` or `unrealizedNetProfit` if live iteration needs floating PnL.
4. Add tests proving both brokers converge after `finish`/`closeAll` and expose distinct in-run values only through explicit names.

### 3. Sharpe ratio column is a stub

Status: fixed by replacing the stubbed Sharpe column with the existing real `profitFactor` metric.

Current `FXBacktest/Sources/FXBacktest/ContentView.swift` returns `0` for every `sharpeRatio`.

Impact: the UI presents a risk-adjusted metric that is not calculated.

Fix plan:

1. Either remove/rename the column to avoid implying real Sharpe, or add enough result data to compute it.
2. If implemented, add per-pass return series or ledger-derived returns to `BacktestPassResult`.
3. Add tests for finite Sharpe, zero-variance returns, no-trade results, and loss-only results.
4. Update `FXBacktest/README.md` if the table semantics change.

### 4. Drawdown percent is initial-deposit-relative, not peak-relative

Status: fixed by renaming the UI/docs to `DD / deposit %`, which matches the currently available result denominator.

Current `drawdownPercent` divides `maxDrawdown` by `model.resultInitialDeposit`. `BacktestPassResult` does not carry `equityPeak`, so standard peak-relative max drawdown cannot be displayed.

Impact: strategies with large equity growth can show misleading drawdown percentages.

Fix plan:

1. Decide whether the UI should display initial-deposit-relative drawdown or standard peak-relative drawdown.
2. If standard, add `equityPeak` or equivalent denominator to `BacktestPassResult`, persistence DTOs, and result stores.
3. If initial-deposit-relative is intentional, rename the column and docs accordingly.

### 5. Live promotion evidence has no freshness policy

Status: fixed with a default 30-day promotion evidence age limit, validated against workload issue time.

Current `FXLivePromotionEvidence.validate()` checks only positive `approvedAtUTC`, not age relative to workload issue time.

Impact: stale promotion evidence can satisfy the live workload contract unless an outer system rejects it.

Fix plan:

1. Add a configurable maximum promotion age, or encode `expiresAtUTC` in promotion evidence.
2. Validate against `FXLiveAgentWorkloadRequest.issuedAtUTC`.
3. Add tests for fresh, stale, future, and missing/invalid approval timestamps.

## P2 / Data Integrity, Robustness, And Diagnostics

### 6. `MT5ImporterConnector` treats missing synchronization as complete

Status: fixed.

Current code uses `response.seriesSynchronized ?? true`.

Impact: if a bridge response omits `series_synchronized`, source completeness fails open. The current EA appears to send the field for range responses, but the Swift connector should still fail closed for version drift or malformed payloads.

Fix plan:

1. Change default to `false`.
2. Add connector tests for missing, true, and false `series_synchronized`.
3. Confirm backfill and verification callers handle `sourceComplete == false` correctly.

### 7. MT5 socket timeout configuration failures are reported as read/write failures

Status: fixed.

Current `setsockopt` failures for receive/send timeouts throw `readFailed` and `writeFailed`.

Impact: setup/configuration failures are indistinguishable from runtime I/O failures.

Fix plan:

1. Add `MT5BridgeError.socketConfigurationFailed(option:errno:)`.
2. Throw it from `configureTimeouts`.
3. Add description text and tests if existing MT5 bridge error tests cover diagnostics.

### 8. FXAICertify evidence post timeout is ignored

Status: fixed.

Current code discards the result of `semaphore.wait(timeout:)`.

Impact: when evidence posting is configured, timeout can be silent.

Fix plan:

1. Check the wait result.
2. Print a warning to stderr on timeout.
3. Consider including failed evidence-post status in certification output when an evidence URL is configured.

### 9. `install_fxai.sh` repeats `brew --prefix python` and does not validate the prefix

Status: fixed.

Current code constructs `$(brew --prefix python 2>/dev/null)/bin/python3` inline.

Impact: brew-prefix failure is opaque and can produce misleading path checks.

Fix plan:

1. Capture `brew_python_prefix`.
2. Check it is non-empty before testing `$brew_python_prefix/bin/python3`.
3. Avoid invoking `brew --prefix python` twice.

### 10. PSI bucket edge adjustment can exceed the reference max

Status: fixed.

Current `population_stability_index` increments non-monotonic edges by a fixed epsilon. If repeated quantile edges are close to the final max, the final edge can be pushed beyond the actual reference distribution.

Impact: histogram buckets can become artificial for clustered FX features.

Fix plan:

1. Build PSI edges from unique quantile values where possible.
2. Reduce bucket count when unique values are insufficient.
3. Add tests for repeated values, all-equal values, narrow-range values, and normal continuous values.

### 11. Standard deviation formulas are inconsistent

Status: fixed by using sample standard deviation in `common_stats.mean_std` to match `testlab.shared.mean_std_ci`.

Current code:

- `common_stats.mean_std` uses population variance.
- `testlab.shared.mean_std_ci` uses sample variance.

Impact: promotion stability and testlab CI can use different volatility estimates.

Fix plan:

1. Decide whether `mean_std` is descriptive population statistics or sample-estimator statistics.
2. Rename functions or add a `sample` flag if both are needed.
3. Add tests for two-value and small-sample inputs.

### 12. Parameter sweep count uses fixed absolute epsilon

Status: fixed.

Current code uses `floor((span / step) + 1.0e-9) + 1`.

Impact: extreme step/span magnitudes can still produce off-by-one counts or clamped duplicate endpoint values.

Fix plan:

1. Add focused tests for decimal steps that nearly divide the range and values that barely overshoot.
2. Replace fixed epsilon with relative tolerance or integer/decimal stepping policy.
3. Ensure `value(at:)` cannot emit unintended duplicate terminal values unless explicitly desired.

### 13. Execution-quality partial-fill risk combines probability and fill ratio with `max`

Status: fixed by blending explicit partial-fill probability and fill-ratio shortfall instead of taking their maximum.

Current `partial_norm` takes the maximum of explicit partial-fill probability and `1 - fill_ratio_mean`.

Impact: explicit probability and fill completeness are different quantities; taking max may overstate or double-count partial-fill risk depending on input semantics.

Fix plan:

1. Define the intended meaning of `broker_partial_fill_prob` and `broker_fill_ratio_mean`.
2. Prefer explicit probability when available, or combine probability and severity as separate features.
3. Add tests for high fill ratio with high partial probability, low fill ratio with low partial probability, and missing/default values.

### 14. `row_float` catches broad exceptions

Status: fixed.

Current `row_float` catches `Exception` in both key lookup and float conversion.

Impact: broad catches can mask unexpected mapping/type bugs.

Fix plan:

1. Narrow lookup handling to `KeyError`, `TypeError`, and `AttributeError` as needed.
2. Narrow conversion handling to `ValueError` and `TypeError`.
3. Add tests for missing keys, invalid strings, non-mapping rows, and `None`.

## Rejected Findings

### A. ATR guard is not an off-by-one bug

Source report item: 1.

Reason: `atrAsSeries` reads `close[index + 1]` for every true-range calculation. With `last = startIndex + period - 1`, the maximum previous-close index is `startIndex + period`, so `startIndex + period < count` is required for memory safety. Functions that do not read a previous close can use `<=`; ATR cannot without changing the formula.

### B. Negative slope sign is consistent with as-series indexing

Source report item: 9.

Reason: FXAI series functions treat `startIndex` as the newer/current side and larger offsets as older bars. A normal upward time trend has a negative regression slope when x increases from newer to older bars. Returning `-slope / reference` makes upward time trends positive. This is not a directional inversion bug.

Optional cleanup: add a short comment explaining the sign convention.
