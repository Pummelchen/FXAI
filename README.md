# FXAI

> **AI agents: start here**
>
> Before making changes, read [`AI_INDEX.md`](./AI_INDEX.md), then [`AGENTS.md`](./AGENTS.md).
> The generic first-session prompt is in [`.ai/START_HERE.md`](./.ai/START_HERE.md).
>
> These files summarize the repository architecture, commands, conventions, risks, and recommended reading order for a fresh AI session.
> They are vendor-neutral and intended for any high-end AI coding agent.

FXAI is a pure Swift, Metal, PyTorch, TensorFlow, ONNX Runtime, and optional remote inference research/execution stack for FX data, backtesting, plugin evaluation, demo trading, and live trading. MT5 can remain a data or execution endpoint where needed, but the source of truth is the Swift project structure in this repository.

FXAI is organized around one strict rule: only FXDatabase may touch ClickHouse directly. Every other project talks to FXDatabase through an API.

**Production Status:** ✅ Production Ready (June 2026 audit, 757+ tests passing) — [View Details](https://github.com/Pummelchen/FXAI/wiki/Production-Readiness-Audit)

**Documentation:** [Wiki](https://github.com/Pummelchen/FXAI/wiki) · [Roadmap](https://github.com/Pummelchen/FXAI/wiki/Roadmap) · [Governance](GOVERNANCE.md) · [Architecture](https://github.com/Pummelchen/FXAI/wiki/Architecture)

## Quick Start

### Installation
```bash
./install_fxai.sh
```

Dry run to see what will be installed:
```bash
DRY_RUN=1 ./install_fxai.sh
```

### Verification
```bash
# Build all packages
./fxai certify --build-only

# Run test suites
swift test --package-path FXDatabase
swift test --package-path FXDataEngine
FXAI_PYTHON=/opt/homebrew/bin/python3.12 swift test --package-path FXPlugins
swift test --package-path FXBacktest

# Run full certification
./fxai certify --all
```

### First Backtest
```bash
# Run a sample audit with walk-forward validation
python3.12 FXDataEngine/Tools/fxai_testlab.py run-audit \
  --scenario-list "{market_walkforward, market_trend, market_chop}" \
  --wf-train-years 1 --wf-test-years 0.25 --wf-window-mode rolling
```

For detailed setup instructions, see [Installation Guide](https://github.com/Pummelchen/FXAI/wiki/Installation).

## Project Map

```text
External sources
  MT5, IBKR/TWS, Yahoo Finance, TradingView, broker files, future feeds
      |
      v
FXImporter
  Connects to external data providers and normalizes incoming history/live data.
      |
      v
FXDatabase
  Owns ClickHouse access, validation, storage, deletion, SineTest data, and all database APIs.
      |
      +--> FXBacktest
      |      Requests historical and backtest data through FXDatabase APIs only.
      |      Calls the root FXPlugins zoo through FXDataEngine plugin payloads.
      |
      +--> FXDataEngine
      |      Post-processes M1 OHLCV data, builds features, contexts, labels, audits, and plugin payloads.
      |
      +--> FXGUI
             Operator UI for runtime state, reports, promotion review, and project workflows.

FXPlugins
  Flat plugin zoo. Each plugin owns its CPU code and optional Metal, PyTorch, TensorFlow,
  NLP, ONNX Runtime, or remote RPC inference code/configuration.

Support contracts and tools
  FXExecutionContracts: shared demo/live account, risk, kill-switch, and audit contracts.
  FXTools: root certification command support, including ./fxai certify.

Agents
  FXBacktestAgent: versioned remote Mac worker protocol and fail-closed worker executable.
  FXDemoAgent: versioned demo execution boundary for approved backtest-derived workloads.
  FXLiveAgent: versioned live execution boundary with promotion, risk, kill-switch, and human-release gates.
```

## Core Boundaries

- `FXImporter` pulls or receives external market data and hands normalized data to FXDatabase.
- `FXMT5Bridge` owns the standalone MT5 socket protocol used by FXImporter connectors and FXDatabase ingestion.
- `FXDatabase` is the only database authority. It owns ClickHouse, validation, migrations, ingestion, storage, deletion, and access APIs.
- `FXBacktest` never reads ClickHouse directly. It asks FXDatabase for history, registers shared/plugin/accelerator configuration through FXDatabase, stores backtest results through FXDatabase, and now exposes the root FXPlugins zoo through a FXDataEngine-backed adapter.
- `FXDataEngine` turns raw M1 OHLCV into features, labels, context payloads, audit data, and plugin-ready requests.
- `FXPlugins` is a flat plugin zoo. Plugins do not import FXDatabase, ClickHouse, or FXBacktest database APIs. They consume FXDataEngine plugin contracts for data payloads, while FXBacktest owns workload scheduling plus shared type 1 and plugin/accelerator type 2 parameter delivery. Optional external inference paths use the same FXDataEngine payload contract as local plugin code.
- `FXBacktestAgent`, `FXDemoAgent`, and `FXLiveAgent` are distributed or execution runtime projects, not data owners. Agent and execution safety contracts fail closed until certification, SineTest, lineage, account scope, risk limits, and kill-switch checks pass.

## API Version Policy

Every public project boundary carries an explicit latest API version. FXAI does not keep older API versions active. If an API advances from v1 to v2, every caller must be updated to v2 before it can use that API again.

This is an intentional fail-closed policy, not an accidental compatibility gap. N-1 rolling compatibility may be added only through a dedicated design ticket that names the supported old/new version pair, compatibility window, downgrade behavior, telemetry, and tests proving mixed-version callers fail or interoperate exactly as specified.

Current latest versions:

| Boundary | Latest version |
| --- | --- |
| FXImporter connector API | `fximporter.connector.v1` |
| FXDatabase FXBacktest API | `fxdatabase.fxbacktest.v1` |
| FXBacktest plugin API | `fxbacktest.plugin-api.v1` |
| FXBacktest plugin acceleration API | `fxbacktest.plugin-acceleration.v1` |
| FXBacktest plugin IR | `fxbacktest.plugin-ir.v1` |
| FXBacktestAgent TCP protocol | `fxbacktest.agent.tcp.v1` |
| FXExecution contracts | `fxexecution.contracts.v1` |
| FXDataEngine / FXPlugins API | `4` |
| FXDataEngine tokenizer contract | `fxai-tokenizer-v1` |

Swift descriptors, request DTOs, plugin manifests, contexts, predictions, and Python accelerator bridge payloads reject non-latest versions during validation.

Backtest configuration, lineage, and certification evidence are also part of `fxdatabase.fxbacktest.v1`. Shared run settings such as `initial_deposit_usd` defaulting to `1000` and `lot_size_lots` defaulting to `0.01`, plus each plugin/accelerator parameter set, are registered in ClickHouse only through FXDatabase API endpoints. FXBacktest does not write configuration or result files to disk.

## Data Contract
