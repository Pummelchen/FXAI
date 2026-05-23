# FXBacktest

FXBacktest is a native Swift macOS backtesting application for running converted MQL5 Expert Advisors as high-performance Swift plugins. It lives as ordinary tracked source in the `FXBacktest/` folder inside [FXAI](https://github.com/Pummelchen/FXAI). It works with the repo-root `FXDatabase/` Swift package as the historical Forex data provider: FXDatabase ingests and verifies M1 OHLCV data from MetaTrader 5, and FXBacktest consumes that verified data for optimization runs.

The goal is similar to the MT5 Strategy Tester optimization view: define a matrix of input/min/step/max parameters, run many complete backtest passes on CPU or Metal, and watch the live table of pass results.

## Current Capabilities

- Native SwiftPM macOS app with SwiftUI interface.
- Plugin API v1 for converted MQL5 EAs stored as optimized single-file Swift plugins.
- Pure M1 OHLC close-price broker model with position lifecycle and trade ledger types.
- Multi-symbol market universe support for plugins that need more than one loaded Forex pair.
- CPU optimizer that splits work by complete backtest pass across workers.
- Optional GPU execution through Metal for plugins that provide a matching compute kernel.
- Hybrid CPU+GPU execution that shares the pass matrix across CPU workers and Metal for maximum throughput.
- Plugin acceleration descriptor/IR scaffold for future generated Swift SIMD and Metal kernels.
- Six operational agents for FXDatabase connectivity, market readiness, run coordination, result persistence, plugin validation, and resource health.
- Read-only FXDatabase data loading through the dedicated FXBacktest API v1.
- FXDatabase-gated result persistence through the dedicated FXBacktest API v1, including explicit purge commands.
- Live pass table with MT5-style result, profit, trade count, drawdown, recovery, Sharpe placeholder, and tested input columns.
- Resident terminal command shell with `>` prompt for loading data, changing settings, starting runs, stopping active work, and checking status without relaunching.
- Demo data mode for UI and engine testing when the FXDatabase API is not running.

## Repository Layout

```text
Package.swift
Sources/
  FXBacktest/                 SwiftUI macOS app
  FXBacktestCore/             engine, data model, FXDatabase loader, plugin API
  FXBacktestPlugins/          converted EA plugins
    FX7.swift                 source link to FXPlugins/fx7/Backtest/FX7.swift
Tests/
  FXBacktestCoreTests/        engine, sweep, and Metal smoke tests
```

Important files:

- `Sources/FXBacktestCore/PluginAPI.swift`: Plugin API v1.
- `Sources/FXBacktestCore/AgentTypes.swift`: shared operational-agent status/outcome types.
- `Sources/FXBacktestCore/OperationalAgents.swift`: six production agents used by the resident app lifecycle.
- `Sources/FXBacktestCore/CPUBacktestExecutor.swift`: whole-pass CPU optimization.
- `Sources/FXBacktestCore/MetalBacktestExecutor.swift`: plugin-provided Metal kernel runner.
- `Sources/FXBacktestCore/HybridBacktestExecutor.swift`: shared CPU+Metal pass scheduling.
- `Sources/FXBacktestCore/ExecutionModel.swift`: pure M1 OHLC broker model and deterministic ledger simulator.
- `Sources/FXBacktestCore/OhlcMarketUniverse.swift`: aligned multi-symbol OHLC universe.
- `Sources/FXBacktestCore/FXDatabaseHistoryLoader.swift`: FXDatabase FXBacktest API v1 client bridge.
- `Sources/FXBacktestCore/BacktestResultStore.swift`: FXDatabase result API bridge and purge support.
- `Sources/FXBacktestCore/PluginAcceleration.swift`: plugin acceleration descriptor and IR v1.
- `Sources/FXBacktestPlugins/FX7.swift`: source link to `../FXPlugins/fx7/Backtest/FX7.swift`, the plugin-owned FX7 OHLC-only backtest implementation with closed signal-bar feature timing, CPU universe support, and a Metal kernel for single-symbol sweeps.

## Requirements

- macOS 26 or newer.
- Apple Silicon Mac recommended, especially M2/M3 or newer for the intended performance target.
- Swift 6.3 toolchain from Xcode 26.5 or newer.
- FXAI checked out with FXBacktest and FXDatabase inside the same repository:

```text
FX/
  FXAI/
    FXBacktest/
    FXDatabase/
```

The Swift package dependency is local:

```swift
.package(path: "../FXDatabase")
```

## Quickstart

### 1. Clone FXAI

```bash
git clone https://github.com/Pummelchen/FXAI.git
cd FXAI/FXBacktest
```

### 2. Build And Test

```bash
swift test
swift build -c release
```

Release builds are the relevant performance baseline because SwiftPM uses whole-module optimization in release mode.

### 3. Run The App From Source

```bash
swift run FXBacktest
```

Do not pass launch-time parameters. FXBacktest starts the SwiftUI backtester and a resident terminal prompt:

```text
>
```

Paste commands into that prompt while the app keeps running. Status messages continue to print to the terminal, and the SwiftUI live table updates at the same time.
The `status` command includes an agent count summary; `agents` prints the latest outcome for each operational agent.

FXBacktest has no supported executable flags. Extra text after `swift run FXBacktest` is ignored and cannot change app settings, data selection, plugin selection, run settings, or parameter ranges. All operator input belongs in commands typed after the resident `>` prompt.

The first screen is the working backtester, not a setup wizard. It includes:

- EA plugin picker.
- Data controls.
- CPU, GPU (Metal), and Both engine selector.
- Parameter matrix editor.
- Run/Stop buttons.
- Live MT5-style optimization table.

### 4. Run A Demo Backtest

Use this when the FXDatabase API is not running.

1. Launch FXBacktest.
2. Select `Moving Average Cross`.
3. Click `Demo`.
4. Keep `CPU` selected.
5. Adjust parameter ranges if needed.
6. Click `Run`.

The same flow from the terminal prompt is:

```text
> load-demo
> run cpu
```

The pass table updates live and sorts the best results by net profit.

### 5. Run A Backtest With FXDatabase Data

FXDatabase is responsible for historical data ingestion, broker UTC mapping, verification, repair, and internal storage. FXBacktest only reads verified data through FXDatabase's dedicated FXBacktest API v1. FXBacktest must not connect to ClickHouse directly.

In FXDatabase, prepare the data first:

```bash
cd ../FXDatabase
swift build -c release
.build/release/FXDatabase
```

At the FXDatabase `>` prompt, run:

```text
> startcheck --config-dir Config --migrations-dir Migrations
> backfill --config-dir Config --symbols all
> verify --config-dir Config --random-ranges 20
> fxbacktest-api --config-dir Config --api-host 127.0.0.1 --api-port 5066
```

Leave FXDatabase running while FXBacktest loads data. FXBacktest uses FXDatabase only as a historical M1 OHLCV provider; backtests receive open/high/low/close/volume bars and do not request spread, swap, commission, margin, bid/ask, or tick data. Plugins should use volume whenever the loaded dataset has nonzero volume.

Then in FXBacktest:

1. Set FXDatabase API URL, usually `http://127.0.0.1:5066`.
2. Set broker source id, for example `icmarkets-sc-mt5-4`.
3. Set logical symbol, for example `EURUSD`.
4. Set expected MT5 symbol and digits.
5. Set UTC start/end epoch seconds, minute-aligned.
6. Click `Load FXDatabase`.
7. Select CPU, GPU (Metal), or Both.
8. Edit the parameter matrix.
9. Click `Run`.

Before the first optimization pass starts, FXBacktest validates the loaded market universe and then uses the M1 OHLC close price as the execution price for the simplified broker model.

The same flow from the FXBacktest terminal prompt is:

```text
> load-fxdatabase --api-url http://127.0.0.1:5066 --broker icmarkets-sc-mt5-4 --symbol EURUSD --mt5-symbol EURUSD --digits 5 --from 1704067200 --to 1707177600 --max-rows 5000000
> set-param signal_stride_bars --input 15 --min 1 --step 1 --max 60
> set-param base_entry_threshold --input 0.02 --min 0.005 --step 0.005 --max 0.12
> run both --workers 8 --chunk 128
```

For multi-symbol EAs such as FX7, load an aligned market universe in one command. FXBacktest requests each symbol from FXDatabase API v1 and rejects the universe if timestamps do not line up exactly:

```text
> load-fxdatabase --api-url http://127.0.0.1:5066 --broker icmarkets-sc-mt5-4 --symbols EURUSD,USDJPY,EURGBP --from 1704067200 --to 1707177600 --max-rows 5000000
> plugin FX7
> set-param signal_stride_bars --input 15 --min 15 --step 15 --max 15
> run cpu --workers 8 --chunk 128
```

Single-symbol `--symbol`, `--mt5-symbol`, and `--digits` validation remains available for strict one-pair loads. For multi-symbol loads, FXBacktest stores the MT5 symbol and digits returned by FXDatabase for metadata validation and price scaling only.

If FXDatabase reports missing verified coverage, bad hashes, mixed digits, duplicate timestamps, invalid OHLC rows, or unsafe ingestion state, FXBacktest fails closed instead of running against questionable data.

## Terminal Command Shell

FXBacktest is intended to stay open. If no backtest is active, it waits at the `>` prompt for the next command. State-changing commands gracefully stop active data loads or optimization runs before changing the app state.

The `--api-url`, `--workers`, `--input`, and similar `--...` tokens below are command options typed inside the running app. They are not launch-time parameters. Options may be entered as `--key value` or `--key=value`.
The FXDatabase API URL must be an absolute `http` or `https` URL. CPU-only plugins reject `gpu`, `metal`, and `both` targets; select a Metal-capable plugin before choosing those paths.

Useful commands:

```text
status
agents
config
plugins
plugin <plugin-id-or-display-name>
params
set <field> <value>
set --api-url http://127.0.0.1:5066 --target both --workers 8
set --api-url http://127.0.0.1:5066 --persist-results true
set-param <key> --input 12 --min 6 --step 2 --max 40
load-demo
load-fxdatabase [--api-url URL] [--broker ID] [--symbol EURUSD] [--symbols EURUSD,USDJPY] [--mt5-symbol EURUSD] [--digits 5] [--from UTC] [--to UTC] [--max-rows N]
run [cpu|gpu|metal|both] [--workers N] [--chunk N] [--initial-deposit N] [--contract-size N] [--lot N]
save-results [--run-id ID] [--note TEXT]
clean-backtest-data --older-than-days 30
clean-backtest-data --all true
stop
reset-params
help
exit
```

`set --persist-results true` streams future optimization rows into FXDatabase through `BacktestResultStore`. FXDatabase owns the actual ClickHouse writes behind its API. `save-results` persists a point-in-time snapshot of the currently retained in-memory result rows. `clean-backtest-data` is the purge command for old or unwanted optimization result data.

## Operational Agents

FXBacktest now runs six small production agents at app boundaries where correctness matters. They are preflight and lifecycle checks, not per-pass hot-loop work:

- `FXDatabase Connectivity`: verifies `GET /v1/status` and the API version before FXDatabase-backed loads.
- `Market Readiness`: validates aligned, non-empty M1 OHLC universes and rejects demo/FXDatabase mixes.
- `Optimization Run Coordinator`: validates the target, sweep, workers, chunk size, deposit, lot size, and immutable run settings.
- `Result Persistence`: owns FXDatabase result run start, buffered writes, finalization, snapshot saves, and purge commands.
- `Plugin Validation`: validates Plugin API v1 descriptors, parameters, acceleration descriptors, and Metal declarations.
- `Resource Health`: checks CPU worker pressure, Metal availability, thermal state, memory, and disk headroom.

Use `agents` in the resident prompt to inspect the latest outcome:

```text
> agents
```

## FXDatabase Data Contract

FXBacktest consumes FXDatabase only through the dedicated FXBacktest API v1:

- API version: `fxdatabase.fxbacktest.v1`
- Status endpoint: `GET /v1/status`
- M1 history endpoint: `POST /v1/history/m1`
- Result schema endpoint: `POST /v1/backtest/results/schema`
- Result run start endpoint: `POST /v1/backtest/results/runs/start`
- Result pass append endpoint: `POST /v1/backtest/results/passes/append`
- Result run completion endpoint: `POST /v1/backtest/results/runs/complete`
- Result purge endpoint: `POST /v1/backtest/results/purge`
- Result read endpoints: `POST /v1/backtest/results/runs/get`, `POST /v1/backtest/results/passes/get`

FXBacktest imports the small `FXDatabaseFXBacktestAPI` SwiftPM product for v1 DTOs and the HTTP client. That module does not expose ClickHouse, FXDatabase internals, or the old direct history provider.

The data path is:

```text
MT5 + FXDatabase EA
  -> FXDatabase Swift ingestion
  -> FXDatabase internal canonical M1 OHLCV storage
  -> FXDatabase FXBacktest API v1
  -> FXBacktest OhlcDataSeries
  -> CPU or Metal optimization
```

FXBacktest expects:

- M1 closed bars only.
- UTC timestamps, not MT5 server timestamps.
- Scaled integer OHLC prices.
- Unsigned M1 volume values; MT5 rows currently use `0`, while other providers can supply real volume.
- Strictly increasing timestamps.
- Complete verified coverage for the requested UTC range.
- Matching broker source, logical symbol, MT5 symbol, and digits.

FXBacktest does not call FXDatabase for execution side data. Spread, swap, commission, margin, bid/ask quotes, and tick data are intentionally outside the current backtest model. Volume is market data and is passed through to plugins when available.

Direct ClickHouse access is forbidden in FXBacktest for both historical OHLCV data and optimization results. FXBacktest can start runs, append pass rows, complete runs, read stored rows, and purge old data only through FXDatabase API v1. FXDatabase remains the gatekeeper for the underlying result tables:

- `fxbacktest_runs`
- `fxbacktest_pass_results`

## Pure OHLC Execution Model

`BacktestBrokerV2` is the deterministic broker surface for converted plugins that need position lifecycle and a closed-trade ledger while staying on the simplified data model. It models:

- Per-symbol digits and scaled integer prices.
- Open and close execution directly at the M1 OHLC close price supplied by the plugin.
- Positive lot sizes from the run settings or plugin logic.
- Position lifecycle and closed-trade ledger.

There is no execution profile in `BacktestRunSettings`. PnL uses close-price delta, symbol digits, configured contract size, and lots. The older `BacktestBroker` remains available for simple single-position plugins.

## Multi-Symbol Backtests

`OhlcMarketUniverse` holds multiple `OhlcDataSeries` instances keyed by logical symbol. FXBacktest validates that all series have identical timestamps before a multi-symbol run starts. This keeps each pass deterministic and avoids plugins silently reading mismatched bars.

Plugins can implement:

```swift
func runPass(
    marketUniverse: OhlcMarketUniverse,
    parameters: ParameterVector,
    context: BacktestContext
) throws -> BacktestPassResult
```

Existing single-symbol plugins still compile because the default implementation runs against the universe primary series.

## Result Store And Purge

Optimization results can be persisted through FXDatabase's FXBacktest result API:

```text
> set --api-url http://127.0.0.1:5066 --persist-results true
> run cpu
```

For a manual snapshot of retained rows:

```text
> save-results --note current-best-window
```

To clean old result data:

```text
> clean-backtest-data --older-than-days 30
```

To remove all stored optimization result data:

```text
> clean-backtest-data --all true
```

## Plugin API v1

Converted EAs implement `FXBacktestPluginV1`:

```swift
public protocol FXBacktestPluginV1: Sendable {
    var descriptor: FXBacktestPluginDescriptor { get }
    var parameterDefinitions: [ParameterDefinition] { get }
    var metalKernel: MetalKernelV1? { get }
    var accelerationDescriptor: PluginAccelerationDescriptor { get }

    func runPass(
        market: OhlcDataSeries,
        parameters: ParameterVector,
        context: BacktestContext
    ) throws -> BacktestPassResult

    func runPass(
        marketUniverse: OhlcMarketUniverse,
        parameters: ParameterVector,
        context: BacktestContext
    ) throws -> BacktestPassResult
}
```

Plugin rules:

- Keep all mutable EA state local to `runPass`.
- Do not share mutable globals across passes.
- Treat OHLC arrays as read-only.
- Return aggregate metrics for each pass.
- Store only backtest-native plugins under `Sources/FXBacktestPlugins/`. FXAI prediction plugins live in the repo-root `FXPlugins/` package.
- Register plugins in `FXBacktestPluginRegistry`.

Single-pass reports are intentionally not implemented yet. The current product focus is maximum optimizer throughput and a live pass table.

The live optimization table follows the MT5 tester shape: fixed metric columns first (`Pass`, `Result`, `Profit`, `Total trades`, `Drawdown %`, `Recovery factor`, `Sharpe ratio`), followed by one column per tested plugin input parameter.

### Retired Demo Plugins

The former FXBacktest-local `MovingAverageCross` and `FXStupid` demo plugins have been moved into the repo-root `FXPlugins` package as FXDataEngine adapters. FX7 now also has source ownership in `FXPlugins/fx7`; FXBacktest keeps a source link so its native plugin target and tests continue to build without creating a SwiftPM package cycle.

### FX7

`FX7` is the OHLCV conversion of the FX7 MQL5 EA core. The CPU path keeps the EA's closed-bar feature flow where possible while intentionally omitting non-OHLCV dependencies such as carry, value, macro data, spread, swap, commission, margin, bid/ask, and tick data.

Key conversion details:

- `signal_stride_bars` is the signal timeframe in M1 bars. The default `15` matches the EA's `PERIOD_M15` signal timeframe.
- M1 bars are aggregated into fixed UTC signal buckets before FX7 features are calculated.
- Features use the last fully closed signal bar, and trades execute at the next M1 close.
- Signal warmup follows the EA's `SignalBarsNeeded()` logic, including the extra 100 signal bars used by the MQL5 version.
- MQL5 trend weights and windows are exposed as plugin inputs: `trend_weight_1`, `trend_weight_2`, `trend_weight_3`, `er_window`, `breakout_window`, and `short_reversal_window`.
- `allow_long` and `allow_short` mirror the EA direction gates.
- CPU runs support aligned multi-symbol universes for panic, correlation, novelty, crowding, and portfolio target selection.
- Metal runs are single-symbol only because the current Metal executor ABI passes one OHLC series. The FX7 Metal kernel uses the same signal-timeframe OHLC timing and core trend/regime/risk gates, but it is not a full multi-symbol correlation/novelty replacement for the CPU path.

## Plugin Acceleration API

`PluginAccelerationDescriptor` and `PluginAccelerationIR` define the v1 scaffold for converting suitable plugins into generated Swift SIMD or Metal kernels while keeping the hand-converted Swift plugin as the fidelity reference. FX7 provides both a CPU reference path and a single-symbol Metal kernel.

## CPU, GPU, And Hybrid Execution Model

CPU optimization splits the parameter matrix into chunks. Each worker receives complete passes and each pass owns its strategy state. FXBacktest does not split one pass across multiple threads because that risks state corruption in converted EA logic.

GPU optimization is available through Metal only for plugins that provide `MetalKernelV1`. Swift plugin code does not automatically run on the GPU. A Metal plugin kernel receives immutable OHLC buffers, a flattened parameter buffer, job records, and a result buffer. Each GPU thread owns one complete parameter combination and writes one result row.

`Both` is the hybrid mode. It requires a Metal-capable plugin and runs CPU workers plus a Metal command-buffer loop at the same time. A shared allocator hands out disjoint pass ranges, so each parameter combination is executed exactly once by either CPU or GPU. Result rows still record the engine that produced the pass as `cpu` or `metal`; the stored run target is `both`.

## Metal Kernel ABI v1

FXBacktest binds Metal buffers as:

| Index | Type | Meaning |
| --- | --- | --- |
| 0 | `const device long *` | UTC epoch seconds |
| 1 | `const device long *` | Open prices, scaled integers |
| 2 | `const device long *` | High prices, scaled integers |
| 3 | `const device long *` | Low prices, scaled integers |
| 4 | `const device long *` | Close prices, scaled integers |
| 5 | `constant uint &` | Bar count |
| 6 | `const device FXBTMetalJob *` | Jobs, one per parameter combination |
| 7 | `const device float *` | Flattened parameter values |
| 8 | `device FXBTMetalResult *` | Output rows, one per job |
| 9 | `constant FXBTMetalRunConfig &` | Initial deposit, contract-lot value, price scale, digits |

The kernel must assign exactly one independent pass to each `thread_position_in_grid` and write only `results[id]`.

## Testing

```bash
swift test
swift test -c release
swift build -c release
```

The test suite includes:

- Lazy parameter-matrix indexing.
- CPU whole-pass chunk execution.
- Metal kernel compile and dispatch smoke test when Metal is available.
- FX7 signal-timeframe warmup, direction gates, CPU flow, multi-symbol flow, and single-symbol Metal kernel smoke tests.
- Hybrid CPU+Metal scheduling without duplicate pass indexes.
- Pure OHLC broker and ledger behavior.
- Multi-symbol universe alignment validation.
- FXDatabase result API bridge and purge behavior with a mock client.
- Plugin acceleration descriptor validation.
- Operational-agent validation for FXDatabase connectivity, market readiness, plugin metadata, run coordination, resource health, and result persistence.

## Documentation

FXBacktest documentation now lives in this README and the root-level FXAI project documentation. The former standalone FXBacktest GitHub Wiki should be treated as historical once the standalone repository is retired.

## Status

FXBacktest is in the first functional engine/app stage. It can load demo data, load single-symbol or aligned multi-symbol verified FXDatabase data, supervise runs through six operational agents, run CPU, Metal, or hybrid CPU+Metal optimizations for plugins that provide a kernel, and persist optimization results through FXDatabase API v1. Future work should add more converted EA plugins, fuller generated-kernel acceleration, and optional single-pass reporting.
