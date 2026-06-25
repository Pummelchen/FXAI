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
# ARCHITECTURE.md — FXAI architecture notes

## High-level architecture

FXAI is organized as independent SwiftPM packages plus Python tooling. The central source-grounded boundary is that `FXDatabase` owns ClickHouse and canonical market data; other packages use versioned API contracts.

```text
Provider connectors / MT5 EA / Yahoo D1
        |
        v
FXImporter + FXMT5Bridge
        |
        v
FXDatabase --------------> ClickHouse
        |
        | FXBacktest API v1
        v
FXBacktest -----> FXDataEngine -----> FXPlugins
        |              |                  |
        |              |                  +-- CPU / Metal / Python / ONNX / remote inference backends
        |              +-- TestLab and Offline Lab tooling
        v
FXGUI operator workflows

FXExecutionContracts -> FXDemoAgent / FXLiveAgent
FXBacktestAgent      -> distributed backtest worker boundary
FXTools              -> root certification runner
```

Evidence:
- `README.md`
- `GOVERNANCE.md`
- `FXDatabase/README.md`
- `FXBacktest/README.md`
- `FXDataEngine/README.md`
- `FXPlugins/README.md`

## Runtime flows

### Data import and storage

1. `FXImporter` owns provider-neutral connector contracts.
2. The MT5 EA path is `FXImporter/Connectors/MetaTrader5/EA/FXDatabase.mq5`.
3. `FXMT5Bridge` owns the socket protocol and bridge DTOs.
4. `FXDatabase` validates data, handles verified broker time conversion, writes to ClickHouse, and serves FXBacktest API v1.

Evidence:
- `FXImporter/README.md`
- `FXDatabase/README.md`
- `FXDatabase/Sources/FXDatabaseCLI/FXDatabaseMain.swift`

### Backtesting and plugins

1. `FXBacktest` loads verified M1 OHLCV through FXDatabase API v1.
2. `FXDataEngine` builds feature/context/label/plugin payload contracts.
3. `FXPlugins` executes plugin-owned CPU/reference code or optional accelerator/backend code.
4. `FXBacktest` persists run configuration and results through FXDatabase API endpoints.

Evidence:
- `FXBacktest/README.md`
- `FXDataEngine/README.md`
- `FXPlugins/README.md`

### Offline Lab and GUI

- Offline Lab state is Turso/libSQL-backed and generated artifacts are rebuilt from that state.
- `FXGUI` is an optional operator surface with curated command handoff through `FXAICommandSecurityPolicy`.

Evidence:
- `FXDataEngine/Tools/OfflineLab/README.md`
- `FXGUI/README.md`
- `FXGUI/Sources/FXGUICore/Services/FXAICommandSecurityPolicy.swift`

## Component roles

| Component | Role | Source references |
| --- | --- | --- |
| Import boundary | Provider-specific data to provider-neutral DTOs. | `FXImporter/README.md` |
| MT5 bridge | Swift protocol for MT5 EA communication. | `FXMT5Bridge/Sources/MT5Bridge` |
| Database authority | ClickHouse, migrations, validation, repair, API serving. | `FXDatabase/Sources/`, `FXDatabase/Migrations/` |
| Data engine | Deterministic OHLCV feature and payload contracts. | `FXDataEngine/Sources/FXDataEngine` |
| Plugin zoo | Model/plugin implementations and backend selection. | `FXPlugins/`, `FXPlugins/API/` |
| Backtester | Workload scheduling, broker model, API-backed data/results. | `FXBacktest/Sources/` |
| Operator GUI | Command previews, dashboards, reports, runtime views. | `FXGUI/Sources/` |
| Agent packages | Worker, demo, and production execution boundaries. | `FXBacktestAgent/`, `FXDemoAgent/`, `FXLiveAgent/` |
| Certification | Build/test/evidence runner and boundary scan. | `FXTools/Sources/FXAICertify/main.swift` |

## Trust boundaries

| Boundary | Rule | Evidence |
| --- | --- | --- |
| ClickHouse | Direct access is allowed only in `FXDatabase`. | `GOVERNANCE.md`, `FXTools/Sources/FXAICertify/main.swift` |
| Broker time | Canonical UTC conversion requires verified broker offset authority. | `FXDatabase/README.md` |
| API versioning | Older public API versions are rejected unless compatibility is explicitly designed. | `README.md`, `GOVERNANCE.md` |
| Plugin backend | Declared backends require implementation and certification evidence. | `FXPlugins/README.md`, `GOVERNANCE.md` |
| Offline Lab state | Turso/libSQL is authoritative; generated outputs are rebuilt. | `FXDataEngine/Tools/OfflineLab/README.md` |
| GUI command handoff | Only approved project-local commands may be handed off. | `FXGUI/README.md`, `FXAICommandSecurityPolicy.swift` |

## Unknowns

- CI/CD workflows were not verified.
- Complete plugin and migration inventories were not exhaustively enumerated.
- Docker/container setup was not detected by search and remains unknown.
