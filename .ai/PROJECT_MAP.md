<!--
AI onboarding file.
Mode: bootstrap
Indexed commit: 91a97e92e5622fae867a490dcd917e6c32811955
Last generated: 2026-06-25T14:01:03Z
Generator: generic high-end AI coding agent
Purpose: Help future AI sessions understand this repository quickly.
Audience: Any high-capability AI coding agent, regardless of vendor or model family.
Human edits are allowed. Future refreshes should preserve valid human edits.
-->
# PROJECT_MAP.md — FXAI source map

## Top-level map

| Path | Kind | Responsibility | Evidence |
| --- | --- | --- | --- |
| `README.md` | root doc | Human-facing overview, quickstart, boundaries, API policy. | `README.md` |
| `GOVERNANCE.md` | root doc | Canonical governance and release/change-class gates. | `GOVERNANCE.md` |
| `CONFIGURATION_SEMANTICS.md` | root doc | `.env`/`fxai.toml` precedence and supported path/profile keys. | `CONFIGURATION_SEMANTICS.md` |
| `install_fxai.sh` | shell | macOS setup and environment verifier. | `install_fxai.sh` |
| `fxai` | shell | Wrapper for `swift run --package-path FXTools FXAICertify`. | `fxai` |
| `requirements/fxai-py312.lock` | Python lock | Python 3.12 baseline for TestLab/Offline Lab/plugin backend checks. | `requirements/fxai-py312.lock` |
| `FXImporter/` | SwiftPM package | External data connectors and importer API. | `FXImporter/README.md`, `FXImporter/Package.swift` |
| `FXMT5Bridge/` | SwiftPM package | MT5 bridge protocol product `MT5Bridge`. | `FXMT5Bridge/Package.swift` |
| `FXDatabase/` | SwiftPM package | ClickHouse authority, canonical M1 OHLCV storage, API server, live operations. | `FXDatabase/README.md`, `FXDatabase/Package.swift` |
| `FXDataEngine/` | SwiftPM + Python | Feature/label/context contracts and Lab tooling. | `FXDataEngine/README.md`, `FXDataEngine/Package.swift` |
| `FXPlugins/` | SwiftPM package | Flat plugin zoo and backend runtime surfaces. | `FXPlugins/README.md`, `FXPlugins/Package.swift` |
| `FXBacktest/` | SwiftPM app/package | SwiftUI backtester, engine, result persistence through FXDatabase API. | `FXBacktest/README.md`, `FXBacktest/Package.swift` |
| `FXGUI/` | SwiftPM app/package | Operator GUI, command security, visual/report/runtime surfaces. | `FXGUI/README.md`, `FXGUI/Package.swift` |
| `FXBacktestAgent/` | SwiftPM package | Future distributed backtest worker boundary. | `FXBacktestAgent/README.md`, `FXBacktestAgent/Package.swift` |
| `FXDemoAgent/` | SwiftPM package | Demo-account execution boundary. | `FXDemoAgent/README.md`, `FXDemoAgent/Package.swift` |
| `FXLiveAgent/` | SwiftPM package | Live-account execution boundary. | `FXLiveAgent/README.md`, `FXLiveAgent/Package.swift` |
| `FXExecutionContracts/` | SwiftPM package | Shared demo/live risk, account, kill-switch, audit contracts. | `FXExecutionContracts/Package.swift` |
| `FXTools/` | SwiftPM package | Certification runner behind the root `./fxai` wrapper. | `FXTools/Package.swift` |

## SwiftPM package map

| Package | Products | Local dependencies | Test targets |
| --- | --- | --- | --- |
| `FXImporter` | `FXImporter`, `FXImporterAPI` | `FXMT5Bridge` | `FXImporterAPITests`, `MT5BridgeTests` |
| `FXMT5Bridge` | `MT5Bridge` | none verified | none verified |
| `FXDatabase` | `FXDatabaseHistoryCore`, `FXDatabaseFXBacktestAPI`, executable `FXDatabase` | `FXMT5Bridge` | Domain, validation, time mapping, ClickHouse, ingestion, verification, operations, backtest/API tests |
| `FXDataEngine` | `FXDataEngine`, executable `FXDataEngineCLI` | `FXDatabase` product `FXDatabaseFXBacktestAPI` | `FXDataEngineTests` |
| `FXPlugins` | `FXAIPlugins` | `FXDataEngine` | `FXAIPluginsTests` |
| `FXBacktest` | executable `FXBacktest`, `FXBacktestCore`, `FXBacktestPlugins` | `FXDatabase`, `FXDataEngine`, `FXPlugins` | `FXBacktestCoreTests` |
| `FXGUI` | executable `FXGUI`, `FXGUICore` | none verified | `FXGUICoreTests`, `FXGUIAppTests` |
| `FXBacktestAgent` | executable `FXBacktestAgent`, `FXBacktestAgentCore` | none verified | `FXBacktestAgentCoreTests` |
| `FXDemoAgent` | executable `FXDemoAgent`, `FXDemoAgentCore` | `FXExecutionContracts` | `FXDemoAgentCoreTests` |
| `FXLiveAgent` | executable `FXLiveAgent`, `FXLiveAgentCore` | `FXExecutionContracts` | `FXLiveAgentCoreTests` |
| `FXExecutionContracts` | `FXExecutionContracts` | none verified | `FXExecutionContractsTests` |
| `FXTools` | executable `FXAICertify` | `FXDatabase` product `FXDatabaseFXBacktestAPI` | `FXToolsTests` |

Evidence: every package row is grounded in that package's `Package.swift`.

## Internal dependency flow

```text
External data providers / MT5 EA / Yahoo D1
  -> FXImporter and FXMT5Bridge
  -> FXDatabase
      -> FXBacktest API v1
          -> FXBacktest
              -> FXDataEngine contracts
                  -> FXPlugins runtime/backends
  -> FXGUI reads/previews project/runtime/research workflows
  -> FXDemoAgent / FXLiveAgent consume approved workload and execution contracts
```

Important directionality:

- `FXDatabase` depends on `FXMT5Bridge`; it does not depend on `FXBacktest` or `FXPlugins` for strategy execution.
- `FXDataEngine` depends on `FXDatabase` only through the FXBacktest API DTO product.
- `FXPlugins` depends on `FXDataEngine` and explicitly avoids FXDatabase/ClickHouse.
- `FXBacktest` depends on FXDatabase, FXDataEngine, and FXPlugins.
- Demo/live agents depend on `FXExecutionContracts` and must not own data storage.

## External dependencies and services

| Dependency | Used for | Evidence |
| --- | --- | --- |
| ClickHouse | Canonical storage and API backing for FXDatabase. | `FXDatabase/README.md`, `install_fxai.sh` |
| MetaTrader 5 under Wine | MT5 historical/live data source through EA bridge. | `FXDatabase/README.md`, `FXImporter/README.md` |
| Homebrew | Installs git, Python 3.12, ClickHouse, cmake, pkg-config, libomp, ripgrep. | `install_fxai.sh` |
| Python 3.12 | TestLab, Offline Lab, plugin backend checks. | `install_fxai.sh`, `requirements/fxai-py312.lock` |
| PyTorch MPS | Python backend acceleration baseline. | `install_fxai.sh`, `FXPlugins/README.md` |
| TensorFlow Metal | TensorFlow backend baseline. | `install_fxai.sh`, `requirements/fxai-py312.lock` |
| ONNX Runtime | Optional inference-only plugin backend. | `FXPlugins/README.md`, `requirements/fxai-py312.lock` |
| Turso/libSQL | Offline Lab research/promotion state. | `FXDataEngine/Tools/OfflineLab/README.md` |
| Remote RPC endpoint | Optional inference-only plugin backend. | `FXPlugins/README.md` |

## Important config files

- `FXDatabase/ConfigSamples/*.json` — sample app, ClickHouse, MT5 bridge, symbols/history configs.
- `FXDatabase/Config/` — local ignored runtime config; may contain credentials.
- `FXDataEngine/fxai.toml` — default project/toolchain/GUI configuration.
- `.env` / `FXAI_ENV_FILE` / `FXAI_CONFIG` — supported by configuration semantics.
- `requirements/fxai-py312.lock` — pinned Python dependency baseline.
- `.gitignore` — generated and local runtime output exclusions.

## Unknowns from bootstrap scan

- Complete branch inventory was not available through the connector search used here.
- Complete recursive tree, migration list, and plugin folder inventory were not exhaustively enumerated.
- CI workflows and Docker/container setup were not verified; searches did not surface `.github/workflows`, `Dockerfile`, or `docker-compose.yml`.
