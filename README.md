# FXAI

FXAI is a pure Swift, Metal, PyTorch, TensorFlow, ONNX Runtime, and optional remote inference research/execution stack for FX data, backtesting, plugin evaluation, demo trading, and live trading. MT5 can remain a data or execution endpoint where needed, but the source of truth is the Swift project structure in this repository.

FXAI is organized around one strict rule: only FXDatabase may touch ClickHouse directly. Every other project talks to FXDatabase through an API.

Roadmap: [FXAI Roadmap](https://github.com/Pummelchen/FXAI/wiki/Roadmap)
Governance: [FXAI Governance](GOVERNANCE.md)

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

The core historical data contract is `M1 OHLCV`.

- M1 open, high, low, close are mandatory.
- Volume is optional by provider, but when `volume > 0` exists, FXDataEngine and plugins must use it.
- Spread is no longer part of the offline backtest contract.
- SineTest is a virtual FXDatabase security used by the test suite to prove plugins and accelerators can handle a simple predictable series without crashing. FXDatabase also persists it as canonical synthetic data from `2000-01-01` through runtime-now, with a full-hour peak of `1.000000`, half-hour trough of `0.001000`, and a 10-second sync agent. API requests must pair logical symbol `SINETEST` with source origin `SYNTHETIC`; `SYNTHETIC` is rejected for any other logical symbol.

## Walk-Forward Optimization

FXAI audit runs support rolling and anchored walk-forward validation for
time-series cross-validation without random train/test leakage. The user-facing
policy can be expressed in years and days, then TestLab resolves it to the
existing bar-based audit contract so the runtime stays deterministic.

Examples:

```bash
python3.12 FXDataEngine/Tools/fxai_testlab.py run-audit \
  --scenario-list "{market_walkforward, market_trend, market_chop}" \
  --wf-train-years 1 --wf-test-years 0.25 --wf-window-mode rolling

python3.12 FXDataEngine/Tools/fxai_testlab.py walkforward-analyze \
  --output FXDataEngine/Tools/latest_walkforward_analysis.md
```

The default optimization campaign requests `1,2,3,5,10,15,20,25` training-year
windows. Year windows are enabled per security only when the available history
has enough bars for the full train/test/purge/embargo/fold policy; insufficient
years are recorded as disabled instead of being scheduled. Use `--wf-test-years`
for the out-of-sample horizon, and `--wf-purge-days` / `--wf-embargo-days` to
keep leakage buffers in calendar terms. Resolved policies are written into audit
manifests with the full window plan and minimum bar requirement. When the audit
runner emits `*.walkforward.json` fold diagnostics, `walkforward-analyze` uses
those per-window OOS scores to flag edge degradation over time; otherwise it
falls back to the aggregate `market_walkforward` TSV row.

## User Benefits

| User type | What FXAI gives them |
| --- | --- |
| Backtest researcher | A Swift/Metal offline backtest stack that can run many plugin families without depending on MT5 Strategy Tester. |
| Plugin developer | A clear plugin API, flat plugin folders, CPU references, and optional Metal/PyTorch/TensorFlow/NLP/ONNX/remote RPC acceleration paths inside each plugin. |
| Data operator | One ingestion path into FXDatabase, one ClickHouse authority, and no hidden direct database access from backtests or agents. |
| Demo trader | A versioned FXDemoAgent contract for applying proven backtest parameters to demo accounts before capital is at risk. |
| Live trader | A versioned FXLiveAgent contract where live execution is separated from research and gated by FXAI data, plugin, promotion, and risk contracts. |
| Fleet operator | A versioned FXBacktestAgent protocol where other Macs can pull TCP batch work like MT5 backtest agents and return certified results to FXBacktest. |
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
| `FXTools/` | Root operational tools. Current executable: `FXAICertify`, launched through `./fxai certify`. |
| `FXExecutionContracts/` | Shared versioned account, risk, kill-switch, and order-intent safety contracts for demo/live agents. |
| `FXBacktestAgent/` | Distributed backtest worker package and TCP protocol foundation for remote Macs over TCP. |
| `FXDemoAgent/` | Demo-account execution agent contract with dry-run-first planning, account scoping, risk limits, and kill-switch validation. |
| `FXLiveAgent/` | Live-account execution agent contract with promotion evidence, human-release planning, account scoping, risk limits, and kill-switch validation. |

## Plugin Zoo

`FXPlugins/` is intentionally flat. Every plugin has its own folder and owns its own implementation details:

```text
FXPlugins/plugin_name/
  CPU/
  Metal/
  PyTorch/
  TensorFlow/
  NLP/
  ONNX/
  PluginNamePlugin.swift
```

Only shared API and registry code belongs under `FXPlugins/API/`. Plugin-specific Metal kernels, Python models, tokenizers, or NLP logic stay inside the plugin's own folder.

FXBacktest is the only project that adapts the root plugin zoo into backtest workloads. It pulls market history through FXDatabase APIs, asks FXDataEngine to build plugin-ready requests, converts ClickHouse-stored type 1 and type 2 configuration into plugin hyperparameters, then calls the plugin runtime. Plugins stay unaware of ClickHouse storage and backtest result persistence.

`FXPlugins/demo_plugin_template/` is the compile-checked top-level template for
future plugins. It contains no trading strategy, is intentionally not in the
runtime plugin registry, and includes authoring surfaces for Swift CPU/reference,
Metal, PyTorch MPS, TensorFlow Metal, Foundation NLP, ONNX Runtime, and Remote
RPC.

## Current Runtime Standard

- macOS deployment floor: macOS 26.
- Swift tools: current repo standard is Swift tools 6.3 with Swift language mode 6.
- Hardware target: Apple Silicon M2/M3-class Macs, including newer Apple Silicon generations. M1 and Intel x86 are unsupported runtime targets.
- Metal: Apple GPU acceleration is validated through runtime compilation and plugin-local buffer parity tests, and only counts as available on unified-memory M2/M3-or-newer hosts.
- PyTorch: plugin backends require Apple Silicon MPS for accelerator runtime paths unless a test explicitly opts into CPU fallback.
- TensorFlow: plugin backends require a TensorFlow Metal GPU device for accelerator runtime paths unless a test explicitly opts into CPU fallback.
- Python bridge command: FXAI uses `FXAI_PYTHON` when set, otherwise resolves a Python 3.12 executable such as Homebrew `python3.12`. It must not silently fall back to generic `python3` because Homebrew's default Python can advance beyond TensorFlow Metal support.
- Python package baseline: `requirements/fxai-py312.lock` pins the backend/test stack for Python 3.12, including `tensorflow==2.18.1`, `tensorflow-metal==1.2.0`, `torch`, `onnxruntime`, `pytest`, `libsql`, and `certifi`.
- TensorFlow Metal stack: use `tensorflow==2.18.1` with `tensorflow-metal==1.2.0` on Python 3.12. Newer default Python versions, such as Python 3.14, are not a compatible target for this stack.
- ONNX Runtime: plugin backends may declare `onnxRuntime` for inference-only exported models under `FXPlugins/<plugin>/ONNX/<plugin>.onnx`. Enable with `FXAI_ENABLE_ONNX_RUNTIME=1`; the Python bridge uses `FXAI_ONNX_MODEL_PATH` and `FXAI_ONNX_MANIFEST_PATH` overrides when set.
- Remote RPC: plugin backends may declare `remoteRPC` for inference-only GPU-backed servers. Enable with `FXAI_ENABLE_REMOTE_RPC=1`, configure `FXAI_REMOTE_INFERENCE_ENDPOINT`, and optionally set `FXAI_REMOTE_INFERENCE_AUTH_TOKEN` plus `FXAI_REMOTE_INFERENCE_TIMEOUT_SECONDS`. The first transport is JSON over HTTP POST behind a Swift transport protocol so a gRPC adapter can be added later without changing plugin semantics.
- CoreML/Neural Engine: not declared by plugins until real export, load, prediction, and parity tests exist.

## External Inference Bridge

The optional external inference bridge adds two opt-in inference-only backends:

- `onnxRuntime`: uses the existing Python bridge and `onnxruntime.InferenceSession` to load plugin-local or explicitly configured `.onnx` models.
- `remoteRPC`: uses a Swift `RemoteRPCMLBackendBridge` and `RemoteRPCMLBackendTransport` to POST the latest FXDataEngine inference payload to an external inference server and validate the returned `PredictionV4`.

Both paths remain inert until a plugin declares the backend and the runtime environment explicitly enables it. Training remains local for this slice, and external inference failures use the existing CPU fallback diagnostics path.

## Install

Run the macOS installer from the repo root:

```bash
./install_fxai.sh
```

The installer is Bash 3 compatible for macOS. It rejects Intel x86 and Apple M1 hosts, installs Homebrew dependencies, checks Xcode/Command Line Tools for Swift and Metal, requires Python 3.12, and installs Python packages from `requirements/fxai-py312.lock`. It no longer scans this repo for imports or installs dynamically discovered unpinned packages.

Verify TensorFlow Metal on the same command used by FXAI:

```bash
${FXAI_PYTHON:-python3.12} -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Use a dry run to see what it would do:

```bash
DRY_RUN=1 ./install_fxai.sh
```

## Verification

Useful checks after setup:

```bash
./fxai certify --build-only
./fxai certify --all
swift test --package-path FXDatabase
swift test --package-path FXDataEngine
swift test --package-path FXPlugins
swift test --package-path FXBacktest
swift test --package-path FXGUI
swift test --package-path FXBacktestAgent
swift test --package-path FXDemoAgent
swift test --package-path FXLiveAgent
swift test --package-path FXExecutionContracts
swift build -c release --package-path FXDatabase
swift build -c release --package-path FXDataEngine
swift build -c release --package-path FXPlugins
swift build -c release --package-path FXBacktest
swift build -c release --package-path FXGUI
swift build -c release --package-path FXTools
swift build -c release --package-path FXBacktestAgent
swift build -c release --package-path FXDemoAgent
swift build -c release --package-path FXLiveAgent
swift build -c release --package-path FXExecutionContracts
```

The root certification entrypoint is `./fxai certify --all`. Build-only mode checks Swift/package health without requiring optional Python accelerator imports. Full mode runs package tests and treats accelerator environment probes as required. The strongest plugin-specific certification check remains inside the FXPlugins suite. It verifies registry coverage, volume contracts, SineTest runtime behavior, SineTest prediction sync and 95%+ prediction confidence for every plugin and declared accelerator backend, CPU/reference evidence, FXDatabase-only data access, Metal compile/runtime parity, PyTorch/TensorFlow live train-predict-persistence-load, NLP text/no-text behavior, and CoreML exclusion.

## Governance

FXAI governance is documented in [GOVERNANCE.md](GOVERNANCE.md). That contract defines the authoritative owners for data, feature contracts, plugin behavior, Offline Lab promotion state, runtime deployment artifacts, demo/live execution boundaries, documentation, and release evidence. Use it to choose the required verification gate for documentation-only, package-local, plugin, accelerator, data authority, research promotion, execution, and live-release hardening changes.

## Documentation

User-focused docs live in the GitHub wiki:

- [Wiki Home](https://github.com/Pummelchen/FXAI/wiki)
- [Roadmap](https://github.com/Pummelchen/FXAI/wiki/Roadmap)
- [Architecture](https://github.com/Pummelchen/FXAI/wiki/Architecture)
- [Governance](https://github.com/Pummelchen/FXAI/wiki/Governance)
- [User Roles](https://github.com/Pummelchen/FXAI/wiki/User-Roles)
- [Installation](https://github.com/Pummelchen/FXAI/wiki/Installation)
- [Project Map](https://github.com/Pummelchen/FXAI/wiki/Project-Map)

Project-local docs remain next to the code they describe:

- [Governance](GOVERNANCE.md)
- [Configuration Semantics](CONFIGURATION_SEMANTICS.md)
- [FXImporter](FXImporter/README.md)
- [FXDatabase](FXDatabase/README.md)
- [FXDataEngine](FXDataEngine/README.md)
- [FXPlugins](FXPlugins/README.md)
- [FXBacktest](FXBacktest/README.md)
- [FXGUI](FXGUI/README.md)

## Operating Principle

FXAI is a decision framework, not a profit guarantee. A plugin score is not a trade by itself. FXAI is designed to control data quality, feature preparation, backtest evidence, model selection, execution routing, and operational safety before demo or live trading uses any result.
