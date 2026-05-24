# FXAI

FXAI is a pure Swift, Metal, PyTorch, and TensorFlow research and execution stack for FX data, backtesting, plugin evaluation, demo trading, and live trading. MT5 can remain a data or execution endpoint where needed, but the source of truth is the Swift project structure in this repository.

FXAI is organized around one strict rule: only FXDatabase may touch ClickHouse directly. Every other project talks to FXDatabase through an API.

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
      |      Calls FXDataEngine and FXPlugins to run CPU/Metal/PyTorch/TensorFlow/NLP backtests.
      |
      +--> FXDataEngine
      |      Post-processes M1 OHLCV data, builds features, contexts, labels, audits, and plugin payloads.
      |
      +--> FXGUI
             Operator UI for runtime state, reports, promotion review, and project workflows.

FXPlugins
  Flat plugin zoo. Each plugin owns its CPU code and optional Metal, PyTorch, TensorFlow, or NLP code.

Future agents
  FXBacktestAgent: remote Mac worker that pulls backtest batches over TCP and reports results.
  FXDemoAgent: applies selected backtest parameters to demo accounts across supported terminals/brokers.
  FXLiveAgent: applies approved parameters to live accounts across supported terminals/brokers.
```

## Core Boundaries

- `FXImporter` pulls or receives external market data and hands normalized data to FXDatabase.
- `FXDatabase` is the only database authority. It owns ClickHouse, validation, migrations, ingestion, storage, deletion, and access APIs.
- `FXBacktest` never reads ClickHouse directly. It asks FXDatabase for history, stores backtest results through FXDatabase, and calls plugins through the FXAI contracts.
- `FXDataEngine` turns raw M1 OHLCV into features, labels, context payloads, audit data, and plugin-ready requests.
- `FXPlugins` is a flat plugin zoo. Each plugin folder owns its own implementation and accelerator folders.
- `FXBacktestAgent`, `FXDemoAgent`, and `FXLiveAgent` are future distributed runtime projects, not data owners.

## API Version Policy

Every public project boundary carries an explicit latest API version. FXAI does not keep older API versions active. If an API advances from v1 to v2, every caller must be updated to v2 before it can use that API again.

Current latest versions:

| Boundary | Latest version |
| --- | --- |
| FXImporter connector API | `fximporter.connector.v1` |
| FXDatabase FXBacktest API | `fxdatabase.fxbacktest.v1` |
| FXBacktest plugin API | `fxbacktest.plugin-api.v1` |
| FXBacktest plugin acceleration API | `fxbacktest.plugin-acceleration.v1` |
| FXBacktest plugin IR | `fxbacktest.plugin-ir.v1` |
| FXDataEngine / FXPlugins API | `4` |
| FXDataEngine tokenizer contract | `fxai-tokenizer-v1` |

Swift descriptors, request DTOs, plugin manifests, contexts, predictions, and Python accelerator bridge payloads reject non-latest versions during validation.

## Data Contract

The core historical data contract is `M1 OHLCV`.

- M1 open, high, low, close are mandatory.
- Volume is optional by provider, but when `volume > 0` exists, FXDataEngine and plugins must use it.
- Spread is no longer part of the offline backtest contract.
- SineTest is a virtual FXDatabase security used by the test suite to prove plugins and accelerators can handle a simple predictable series without crashing. FXDatabase also persists it as canonical synthetic data from `2000-01-01` through runtime-now, with a full-hour peak of `1.000000`, half-hour trough of `0.001000`, and a 10-second sync agent.

## User Benefits

| User type | What FXAI gives them |
| --- | --- |
| Backtest researcher | A Swift/Metal offline backtest stack that can run many plugin families without depending on MT5 Strategy Tester. |
| Plugin developer | A clear plugin API, flat plugin folders, CPU references, and optional Metal/PyTorch/TensorFlow/NLP acceleration paths inside each plugin. |
| Data operator | One ingestion path into FXDatabase, one ClickHouse authority, and no hidden direct database access from backtests or agents. |
| Demo trader | A future FXDemoAgent path for applying proven backtest parameters to demo accounts before capital is at risk. |
| Live trader | A future FXLiveAgent path where live execution is separated from research and gated by FXAI data, plugin, and risk contracts. |
| Fleet operator | A future FXBacktestAgent model where other Macs can pull TCP batch work like MT5 backtest agents and return results to FXBacktest. |
| System architect | A repo layout where data import, database authority, feature engineering, plugins, backtesting, UI, and agents have separate ownership. |

## Main Projects

| Folder | Role |
| --- | --- |
| `FXImporter/` | External data source connectors. Current sources include MT5 bridge and Yahoo Finance history; future sources include IBKR/TWS, TradingView, broker files, and other feeds. |
| `FXDatabase/` | ClickHouse gatekeeper, database configuration, ingestion, verification, SineTest data, and FXBacktest database APIs. |
| `FXDataEngine/` | Data post-processing, feature and label contracts, audit tools, runtime artifacts, and plugin payload preparation. |
| `FXPlugins/` | Converted plugin zoo. Plugins stay flat at the root of this folder, with accelerator code under each plugin folder. |
| `FXBacktest/` | Swift/Metal offline backtest framework that uses FXDatabase APIs and calls plugins through FXDataEngine contracts. |
| `FXGUI/` | macOS SwiftUI operator interface for dashboards, reports, promotion review, and workflow access. |
| `FXBacktestAgent/` | Future distributed backtest worker for remote Macs over TCP. |
| `FXDemoAgent/` | Future demo-account execution agent for MT5, IBKR, TradingView, and other account types. |
| `FXLiveAgent/` | Future live-account execution agent with the same broker/terminal abstraction as demo, but stricter approval and safety gates. |

## Plugin Zoo

`FXPlugins/` is intentionally flat. Every plugin has its own folder and owns its own implementation details:

```text
FXPlugins/plugin_name/
  CPU/
  Metal/
  PyTorch/
  TensorFlow/
  NLP/
  PluginNamePlugin.swift
```

Only shared API and registry code belongs under `FXPlugins/API/`. Plugin-specific Metal kernels, Python models, tokenizers, or NLP logic stay inside the plugin's own folder.

## Current Runtime Standard

- macOS deployment floor: macOS 26.
- Swift tools: current repo standard is Swift tools 6.3 with Swift language mode 6.
- Hardware target: Apple Silicon M2/M3-class Macs, including newer Apple Silicon generations. M1 and Intel x86 are unsupported runtime targets.
- Metal: Apple GPU acceleration is validated through runtime compilation and plugin-local buffer parity tests, and only counts as available on unified-memory M2/M3-or-newer hosts.
- PyTorch: plugin backends require Apple Silicon MPS for accelerator runtime paths unless a test explicitly opts into CPU fallback.
- TensorFlow: plugin backends require a TensorFlow Metal GPU device for accelerator runtime paths unless a test explicitly opts into CPU fallback.
- CoreML/Neural Engine: not declared by plugins until real export, load, prediction, and parity tests exist.

## Install

Run the macOS installer from the repo root:

```bash
./install_fxai.sh
```

The installer is Bash 3 compatible for macOS. It rejects Intel x86 and Apple M1 hosts, installs Homebrew dependencies, checks Xcode/Command Line Tools for Swift and Metal, scans this repo for Python imports, and installs matching Python packages with no hard version pins.

Use a dry run to see what it would do:

```bash
DRY_RUN=1 ./install_fxai.sh
```

## Verification

Useful checks after setup:

```bash
swift test --package-path FXDatabase
swift test --package-path FXDataEngine
swift test --package-path FXPlugins
swift test --package-path FXBacktest
swift test --package-path FXGUI
swift build -c release --package-path FXDatabase
swift build -c release --package-path FXDataEngine
swift build -c release --package-path FXPlugins
swift build -c release --package-path FXBacktest
swift build -c release --package-path FXGUI
```

The strongest plugin certification check is inside the FXPlugins suite. It verifies registry coverage, volume contracts, SineTest runtime behavior, SineTest prediction sync and 95%+ prediction confidence for every plugin and declared accelerator backend, CPU/reference evidence, FXDatabase-only data access, Metal compile/runtime parity, PyTorch/TensorFlow live train-predict-persistence-load, NLP text/no-text behavior, and CoreML exclusion.

## Documentation

User-focused docs are mirrored in `Wiki/` for the GitHub wiki:

- [Wiki Home](Wiki/Home.md)
- [Architecture](Wiki/Architecture.md)
- [User Roles](Wiki/User-Roles.md)
- [Installation](Wiki/Installation.md)
- [Project Map](Wiki/Project-Map.md)

Project-local docs remain next to the code they describe:

- [FXImporter](FXImporter/README.md)
- [FXDatabase](FXDatabase/README.md)
- [FXDataEngine](FXDataEngine/README.md)
- [FXPlugins](FXPlugins/README.md)
- [FXBacktest](FXBacktest/README.md)
- [FXGUI](FXGUI/README.md)

## Operating Principle

FXAI is a decision framework, not a profit guarantee. A plugin score is not a trade by itself. FXAI is designed to control data quality, feature preparation, backtest evidence, model selection, execution routing, and operational safety before demo or live trading uses any result.
