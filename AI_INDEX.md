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
# AI_INDEX.md — FXAI repository index

## Snapshot

| Field | Value | Status | Evidence |
| --- | --- | --- | --- |
| Repository | `Pummelchen/FXAI` | verified | GitHub repository metadata |
| Indexed commit | `91a97e92e5622fae867a490dcd917e6c32811955` | verified | `main` compare metadata |
| Operation mode | `bootstrap` | verified | No prior `AI_INDEX.md`, `AGENTS.md`, or `.ai/MANIFEST.json` found on `main` |
| Primary languages | Swift, Python, shell, MQL5, SQL, JSON/TOML, Metal | verified | `*/Package.swift`, `FXDataEngine/Tools/*.py`, `install_fxai.sh`, `FXImporter/Connectors/MetaTrader5/EA/FXDatabase.mq5`, `FXDatabase/Migrations`, `FXPlugins/*/Metal` |
| Host/tooling baseline | macOS 26, Swift tools 6.3, Swift language mode 6, Apple Silicon M2/M3-or-newer target, Python 3.12 | verified | `*/Package.swift`, `FXDatabase/README.md`, `install_fxai.sh`, `requirements/fxai-py312.lock` |

## Repository purpose

FXAI is a Swift-centered FX research and execution stack. It covers provider import, canonical ClickHouse storage through FXDatabase, deterministic data-engine contracts, a flat model/plugin zoo, backtesting, operator GUI workflows, and fail-closed demo/live execution boundaries.

Evidence:
- `README.md`
- `GOVERNANCE.md`
- `FXDatabase/README.md`
- `FXDataEngine/README.md`
- `FXPlugins/README.md`
- `FXBacktest/README.md`
- `FXGUI/README.md`

## Recommended first-read order

1. `AI_INDEX.md`
2. `AGENTS.md`
3. `.ai/START_HERE.md`
4. `README.md`
5. `GOVERNANCE.md`
6. `.ai/PROJECT_MAP.md`
7. `.ai/ARCHITECTURE.md`
8. `.ai/COMPONENTS.md`
9. `.ai/COMMANDS.md`
10. `.ai/TESTING.md`
11. `.ai/SECURITY.md`
12. `.ai/PLAYBOOKS.md`
13. Current source files nearest the requested change

## Architecture summary

- `FXImporter` owns provider-neutral connector contracts and external data connectors. Current verified connectors include the MT5 EA path and Yahoo Finance D1 history connector.
- `FXMT5Bridge` is a standalone Swift TCP/framed JSON bridge boundary for the MT5 EA.
- `FXDatabase` is the only component allowed to access ClickHouse directly. It owns canonical M1 OHLCV storage, migrations, validation, repair, live/supervised operations, SineTest, and FXBacktest API v1.
- `FXDataEngine` owns deterministic feature/label/context construction and plugin payload DTOs. It also hosts Python Offline Lab and TestLab tooling.
- `FXPlugins` is a flat plugin zoo with Swift CPU/reference code and optional Metal, PyTorch, TensorFlow, NLP, ONNX Runtime, and remote-RPC backends. It depends on `FXDataEngine`, not `FXDatabase`.
- `FXBacktest` is a SwiftUI backtesting app and engine that reads verified data and persists results through FXDatabase API v1.
- `FXGUI` is an optional operator GUI that remains terminal-first and enforces a curated command handoff policy.
- `FXBacktestAgent`, `FXDemoAgent`, and `FXLiveAgent` are worker/execution boundaries; demo and live execution must stay separated and fail closed.
- `FXTools` provides the root `./fxai certify` wrapper and certification evidence runner.

Evidence:
- `GOVERNANCE.md`
- `FXDatabase/Package.swift`
- `FXDataEngine/Package.swift`
- `FXPlugins/Package.swift`
- `FXBacktest/Package.swift`
- `FXGUI/Package.swift`
- `FXTools/Sources/FXAICertify/main.swift`

## Directory map

| Path | Responsibility | Status | Evidence |
| --- | --- | --- | --- |
| `FXImporter/` | Provider connectors and importer DTOs. | verified | `FXImporter/README.md`, `FXImporter/Package.swift` |
| `FXMT5Bridge/` | MT5 socket protocol library. | verified | `FXMT5Bridge/Package.swift`, `FXDatabase/README.md` |
| `FXDatabase/` | ClickHouse authority, ingestion, migrations, verification, repair, API server, live/supervisor. | verified | `FXDatabase/README.md`, `FXDatabase/Package.swift`, `FXDatabase/Sources/FXDatabaseCLI/Command.swift` |
| `FXDataEngine/` | Feature contracts, plugin payloads, Offline Lab/TestLab Python tooling. | verified | `FXDataEngine/README.md`, `FXDataEngine/Package.swift` |
| `FXPlugins/` | Flat plugin zoo and backend runtime surfaces. | verified | `FXPlugins/README.md`, `FXPlugins/Package.swift` |
| `FXBacktest/` | Backtesting app, core engine, FXDatabase API clients, plugin bridge. | verified | `FXBacktest/README.md`, `FXBacktest/Package.swift` |
| `FXGUI/` | Operator GUI, command security, dashboards, packaging. | verified | `FXGUI/README.md`, `FXGUI/Package.swift` |
| `FXBacktestAgent/` | Future distributed backtest worker boundary. | verified | `FXBacktestAgent/README.md`, `FXBacktestAgent/Package.swift` |
| `FXDemoAgent/` | Demo execution boundary. | verified | `FXDemoAgent/README.md`, `FXDemoAgent/Package.swift` |
| `FXLiveAgent/` | Live execution boundary. | verified | `FXLiveAgent/README.md`, `FXLiveAgent/Package.swift` |
| `FXExecutionContracts/` | Shared demo/live risk, account, kill-switch, audit contracts. | verified | `FXExecutionContracts/Package.swift`, `GOVERNANCE.md` |
| `FXTools/` | Certification runner behind `./fxai`. | verified | `FXTools/Package.swift`, `fxai` |
| `requirements/` | Python 3.12 package lock. | verified | `requirements/fxai-py312.lock` |
| `.ai/` | Generated vendor-neutral AI-onboarding files. | generated | this bootstrap run |

## Main entrypoints

| Entry | Purpose | Evidence |
| --- | --- | --- |
| `./install_fxai.sh` | macOS/Apple Silicon setup and environment verification. | `install_fxai.sh` |
| `./fxai certify --build-only` | Build certification through `FXTools`. | `fxai`, `FXTools/Sources/FXAICertify/main.swift` |
| `./fxai certify --all` | Full package build/test certification plus environment checks. | `FXTools/Sources/FXAICertify/main.swift` |
| `swift run --package-path FXDatabase FXDatabase ...` | Database CLI/resident shell. | `FXDatabase/Sources/FXDatabaseCLI/CLIOptions.swift` |
| `swift run --package-path FXBacktest FXBacktest` | SwiftUI backtesting app and resident prompt. | `FXBacktest/Sources/FXBacktest/FXBacktestApp.swift`, `FXBacktest/README.md` |
| `swift run --package-path FXGUI FXGUI` or `cd FXGUI && ./start.sh` | Operator GUI. | `FXGUI/Sources/FXGUIApp/FXGUIApp.swift`, `FXGUI/README.md` |
| `python3.12 FXDataEngine/Tools/fxai_testlab.py ...` | TestLab compile, audit, baseline, release-gate workflows. | `FXDataEngine/Tools/testlab/cli.py` |
| `python3.12 FXDataEngine/Tools/fxai_offline_lab.py ...` | Offline Lab control-loop, promotion, recovery, and governance artifacts. | `FXDataEngine/Tools/OfflineLab/README.md` |
| `swift run --package-path FXBacktestAgent FXBacktestAgent --self-check` | Backtest worker capability/self-check. | `FXBacktestAgent/Sources/FXBacktestAgent/main.swift` |

## High-signal commands

```bash
./install_fxai.sh
DRY_RUN=1 ./install_fxai.sh
./fxai certify --build-only
./fxai certify --all
swift test --package-path FXDatabase
swift test --package-path FXDataEngine
FXAI_PYTHON=/opt/homebrew/bin/python3.12 swift test --package-path FXPlugins
swift test --package-path FXBacktest
swift test --package-path FXGUI
python3.12 FXDataEngine/Tools/fxai_testlab.py verify-all
```

See `.ai/COMMANDS.md` and `.ai/TESTING.md` for task-specific commands. No repo-level formatter, linter, or typecheck-only command was verified; use SwiftPM build/test and `git diff --check` as the source-grounded baseline.

## Important conventions

- Only `FXDatabase` may touch ClickHouse directly. All other packages must use FXDatabase APIs.
- FXDatabase canonical prices are scaled integers and raw MT5 timestamps are preserved.
- FXDatabase must not run strategies or optimizations internally; those belong to `FXBacktest` or other external clients through API contracts.
- FXDataEngine owns deterministic data preparation; model execution belongs to `FXPlugins`.
- Plugin assets stay plugin-local; update plugin manifests and certification evidence with behavior changes.
- Generated Offline Lab and runtime artifacts must be rebuilt from Turso/libSQL or runtime state, not hand-edited.
- GUI command handoff is security-filtered; do not make FXGUI a generic shell launcher.
- Demo and live execution remain isolated, with live gated by promotion evidence, account scope, risk, stale-data, and kill-switch checks.

## Common task map

| Task | Start here | Validate with |
| --- | --- | --- |
| FXDatabase API or schema | `FXDatabase/README.md`, `FXDatabase/Sources/`, `FXDatabase/Migrations/` | `swift test --package-path FXDatabase` |
| Feature/payload change | `FXDataEngine/Sources/FXDataEngine/` | `swift test --package-path FXDataEngine` |
| Plugin behavior/backend | `FXPlugins/<plugin_id>/`, `FXPlugins/API/` | `swift test --package-path FXPlugins` |
| Backtest UX/engine | `FXBacktest/Sources/` | `swift test --package-path FXBacktest` |
| GUI workflow | `FXGUI/Sources/` | `swift test --package-path FXGUI` |
| Offline Lab/TestLab | `FXDataEngine/Tools/` | `python3.12 FXDataEngine/Tools/fxai_testlab.py verify-all` |
| Demo/live execution | `FXDemoAgent/`, `FXLiveAgent/`, `FXExecutionContracts/` | package-specific `swift test --package-path ...` |
| Release claim | root and all packages | `./fxai certify --all` |

## Security-sensitive areas

- `FXDatabase/Config/`, `FXDatabase/ConfigSamples/`, and ClickHouse credentials.
- `FXDatabase/Migrations/`, `FXDatabase/Sources/ClickHouse/`, canonical insert/repair paths, and broker UTC offsets.
- `FXImporter/Connectors/MetaTrader5/EA/FXDatabase.mq5` and `FXMT5Bridge/` protocol changes.
- `FXPlugins` external backends, checkpoint manifests, ONNX, and remote-RPC settings.
- `FXDataEngine/Tools/OfflineLab/` Turso/libSQL state and generated artifacts.
- `FXDemoAgent/`, `FXLiveAgent/`, and `FXExecutionContracts/` risk/kill-switch/account boundaries.
- `FXGUI/Sources/FXGUICore/Services/FXAICommandSecurityPolicy.swift`.

See `.ai/SECURITY.md` for the detailed AI-agent safety checklist.

## Generated files / do-not-edit zones

Generated or local runtime outputs are ignored or documented as non-source artifacts:

- SwiftPM build state: `*/.build/`, `*/.swiftpm/`, `*/Package.resolved`.
- FXDatabase local state: `FXDatabase/Config/`, `FXDatabase/Logs/`.
- Offline Lab state/reports/artifacts: many paths under `FXDataEngine/Tools/OfflineLab/`, especially `Profiles/`, `ResearchOS/`, `Distillation/`, `Runs/`, and subsystem state/report folders.
- GUI release/snapshot output: `FXGUI/Artifacts/`.
- Plugin runtime state defaults outside the repo unless overridden by environment.

Evidence:
- `.gitignore`
- `FXDataEngine/Tools/OfflineLab/README.md`
- `FXPlugins/README.md`

## Verified, inferred, unknown, conflicting

- verified: package names, products, dependencies, tools version, macOS floor, entrypoints, governance rules, key commands, and documented runtime boundaries.
- inferred: monorepo-style structure; each Swift package is independent rather than a single root workspace because there is no root `Package.swift` and each top-level component has its own `Package.swift`.
- unknown: complete CI/CD inventory, full branch inventory, complete SQL migration list, complete plugin folder inventory, and Docker/container setup.
- conflicting: no code/doc conflicts were confirmed during this bootstrap scan.
- needs_human_review: production-readiness claim in `README.md` was preserved as existing documentation; this bootstrap did not rerun the referenced audit or tests.

## Bootstrap notes

No prior generic AI-onboarding files or model-specific generated AI files were verified during this run. The new onboarding system is intentionally vendor-neutral and source-grounded. Future refreshes should compare `.ai/MANIFEST.json` to the then-current `main` commit and update only stale sections.
