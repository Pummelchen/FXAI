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
# COMPONENTS.md — FXAI component cards

## FXImporter

| Field | Details |
| --- | --- |
| Responsibility | Provider-neutral market-data connector API and current external connectors. |
| Key files | `FXImporter/README.md`, `FXImporter/Package.swift`, `FXImporter/Sources/FXImporterAPI`, `FXImporter/Sources/FXImporter` |
| Public interfaces | SwiftPM products `FXImporterAPI` and `FXImporter`. |
| Internal dependencies | `FXMT5Bridge` product `MT5Bridge`. |
| External dependencies | MT5 EA bridge path and Yahoo Finance D1 connector are documented. |
| Invariants | FXDatabase remains responsible for canonical storage, UTC authority, validation, and ClickHouse. |
| Tests | `swift test --package-path FXImporter`. |
| Risks | Provider availability/terms and timestamp semantics. |

## FXMT5Bridge

| Field | Details |
| --- | --- |
| Responsibility | Standalone Swift bridge protocol package for MT5 EA communication. |
| Key files | `FXMT5Bridge/Package.swift`, `FXMT5Bridge/Sources/MT5Bridge` |
| Public interfaces | SwiftPM product `MT5Bridge`. |
| Internal dependencies | None verified. |
| Tests | No test target verified in `Package.swift`; MT5 bridge tests are declared in `FXImporter`. |
| Risks | Protocol changes can break FXDatabase ingestion and importer connectors. |

## FXDatabase

| Field | Details |
| --- | --- |
| Responsibility | Sole ClickHouse authority; ingestion, validation, canonical storage, migrations, repair, SineTest, API serving, database operations. |
| Key files | `FXDatabase/README.md`, `FXDatabase/Package.swift`, `FXDatabase/Sources/`, `FXDatabase/Migrations/`, `FXDatabase/ConfigSamples/` |
| Public interfaces | Executable `FXDatabase`; libraries `FXDatabaseHistoryCore` and `FXDatabaseFXBacktestAPI`. |
| Internal dependencies | `FXMT5Bridge`. |
| External dependencies | ClickHouse HTTP, MT5 EA through Wine/localhost bridge. |
| Invariants | Only this package touches ClickHouse; prices are scaled integers; raw MT5 timestamps are preserved; UTC conversion requires verified broker offsets; open M1 bar is not ingested. |
| Tests | `swift test --package-path FXDatabase`. |
| Risks | Schema/migration idempotency, canonical rewrite safety, credential handling, broker time authority. |

## FXDataEngine

| Field | Details |
| --- | --- |
| Responsibility | Deterministic feature construction, labels, runtime context, plugin payload preparation, Offline Lab/TestLab tooling. |
| Key files | `FXDataEngine/README.md`, `FXDataEngine/Package.swift`, `FXDataEngine/Sources/FXDataEngine`, `FXDataEngine/Tools/` |
| Public interfaces | Library `FXDataEngine`, executable `FXDataEngineCLI`, Python tool entrypoints. |
| Internal dependencies | `FXDatabaseFXBacktestAPI` product from `FXDatabase`. |
| External dependencies | Python 3.12 stack, Turso/libSQL for Offline Lab. |
| Invariants | Canonical input is M1 OHLCV from FXDatabase; learned model execution belongs to FXPlugins. |
| Tests | `swift test --package-path FXDataEngine`; Python TestLab checks when tool behavior changes. |
| Risks | Feature shape/version compatibility, leakage-safe audit controls, generated artifact state. |

## FXPlugins

| Field | Details |
| --- | --- |
| Responsibility | Flat plugin zoo with Swift CPU/reference code and optional accelerator/external backend paths. |
| Key files | `FXPlugins/README.md`, `FXPlugins/Package.swift`, `FXPlugins/API/`, plugin folders directly under `FXPlugins/` |
| Public interfaces | SwiftPM product `FXAIPlugins`, plugin manifests and runtime registry. |
| Internal dependencies | `FXDataEngine`. |
| External dependencies | Metal, Python 3.12, PyTorch MPS, TensorFlow Metal, ONNX Runtime, optional remote inference endpoint. |
| Invariants | No FXDatabase/ClickHouse dependency; backend assets are plugin-local; latest runtime API only. |
| Tests | `swift test --package-path FXPlugins`; focused backend tests with `FXAI_PYTHON` where needed. |
| Risks | Fallback behavior, checkpoint manifests, backend source hashes, accelerator evidence. |

## FXBacktest

| Field | Details |
| --- | --- |
| Responsibility | SwiftUI backtesting app, workload scheduling, simplified broker model, plugin bridge, API-backed result persistence. |
| Key files | `FXBacktest/README.md`, `FXBacktest/Package.swift`, `FXBacktest/Sources/FXBacktest`, `FXBacktest/Sources/FXBacktestCore` |
| Public interfaces | Executable `FXBacktest`, libraries `FXBacktestCore` and `FXBacktestPlugins`. |
| Internal dependencies | `FXDatabase`, `FXDataEngine`, `FXPlugins`. |
| Invariants | Reads history and writes results through FXDatabase API; never reads ClickHouse directly. |
| Tests | `swift test --package-path FXBacktest`. |
| Risks | API compatibility, plugin parameter schemas, result persistence, Metal scheduling. |

## FXGUI

| Field | Details |
| --- | --- |
| Responsibility | Optional operator GUI, dashboards, report/runtime views, project connection, command previews, packaging. |
| Key files | `FXGUI/README.md`, `FXGUI/Package.swift`, `FXGUI/Sources/FXGUIApp`, `FXGUI/Sources/FXGUICore`, `FXGUI/Docs/FXGUI_RELEASE_CHECKLIST.md` |
| Public interfaces | Executable `FXGUI`, library `FXGUICore`. |
| Internal dependencies | None verified in `Package.swift`. |
| Invariants | Terminal-first; command cards are curated by `FXAICommandSecurityPolicy`. |
| Tests | `swift test --package-path FXGUI`; `FXGUI/Tools/run_gui_validation_suite.sh` for layout snapshots. |
| Risks | Command policy bypass, stale artifact rendering, layout regressions. |

## Agent and execution packages

| Package | Responsibility | Key validation | Source references |
| --- | --- | --- | --- |
| `FXBacktestAgent` | Distributed backtest worker boundary and capability/self-check envelope. | `swift test --package-path FXBacktestAgent`; `swift run --package-path FXBacktestAgent FXBacktestAgent --self-check`. | `FXBacktestAgent/README.md`, `FXBacktestAgent/Sources/FXBacktestAgent/main.swift` |
| `FXDemoAgent` | Demo execution boundary with dry-run-first capability. | `swift test --package-path FXDemoAgent`. | `FXDemoAgent/README.md`, `FXDemoAgent/Sources/FXDemoAgent/main.swift` |
| `FXLiveAgent` | Production execution boundary with human-release gating. | `swift test --package-path FXLiveAgent`. | `FXLiveAgent/README.md`, `FXLiveAgent/Sources/FXLiveAgent/main.swift` |
| `FXExecutionContracts` | Shared account, risk, kill-switch, and audit contracts. | `swift test --package-path FXExecutionContracts`. | `FXExecutionContracts/Package.swift`, `GOVERNANCE.md` |

## FXTools

| Field | Details |
| --- | --- |
| Responsibility | Root certification runner used by `./fxai`. |
| Key files | `fxai`, `FXTools/Package.swift`, `FXTools/Sources/FXAICertify/main.swift` |
| Public interfaces | Executable `FXAICertify`; wrapper `./fxai`. |
| Behavior | Captures environment evidence, builds/tests Swift packages, and runs a ClickHouse-boundary scan. |
| Validation | `./fxai certify --build-only`; `./fxai certify --all`. |
| Risks | Certification results depend on local macOS/Xcode/Python/accelerator environment. |
