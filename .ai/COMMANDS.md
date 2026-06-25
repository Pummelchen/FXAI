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
# COMMANDS.md — FXAI commands

## Environment setup

```bash
./install_fxai.sh
DRY_RUN=1 ./install_fxai.sh
SKIP_BREW=1 ./install_fxai.sh
SKIP_PYTHON=1 ./install_fxai.sh
FXAI_PYTHON=/opt/homebrew/bin/python3.12 ./install_fxai.sh
```

Verified behavior:
- macOS-only installer.
- Apple Silicon M2/M3-or-newer target; M1 is rejected by the installer.
- Homebrew formulas include git, Python 3.12, ClickHouse, cmake, pkg-config, libomp, ripgrep, and possibly Swift.
- Python requirements default to `requirements/fxai-py312.lock`.
- Verifies PyTorch MPS and TensorFlow Metal baseline.

Evidence:
- `install_fxai.sh`
- `requirements/fxai-py312.lock`

## Root certification

```bash
./fxai certify --build-only
./fxai certify --all
```

`./fxai` runs `swift run --package-path FXTools FXAICertify ...`. `FXAICertify` builds every package in its package list and, with tests enabled, runs package tests. It also performs a ClickHouse-boundary scan outside `FXDatabase`.

Evidence:
- `fxai`
- `FXTools/Sources/FXAICertify/main.swift`
- `GOVERNANCE.md`

## Swift package build/test matrix

Run from repo root unless the package README says otherwise.

| Package | Build | Test |
| --- | --- | --- |
| `FXImporter` | `swift build --package-path FXImporter` | `swift test --package-path FXImporter` |
| `FXMT5Bridge` | `swift build --package-path FXMT5Bridge` | no standalone test target verified |
| `FXDatabase` | `swift build --package-path FXDatabase` | `swift test --package-path FXDatabase` |
| `FXDataEngine` | `swift build --package-path FXDataEngine` | `swift test --package-path FXDataEngine` |
| `FXPlugins` | `swift build --package-path FXPlugins` | `swift test --package-path FXPlugins` |
| `FXBacktest` | `swift build --package-path FXBacktest` | `swift test --package-path FXBacktest` |
| `FXGUI` | `swift build --package-path FXGUI` | `swift test --package-path FXGUI` |
| `FXBacktestAgent` | `swift build --package-path FXBacktestAgent` | `swift test --package-path FXBacktestAgent` |
| `FXDemoAgent` | `swift build --package-path FXDemoAgent` | `swift test --package-path FXDemoAgent` |
| `FXLiveAgent` | `swift build --package-path FXLiveAgent` | `swift test --package-path FXLiveAgent` |
| `FXExecutionContracts` | `swift build --package-path FXExecutionContracts` | `swift test --package-path FXExecutionContracts` |
| `FXTools` | `swift build --package-path FXTools` | `swift test --package-path FXTools` |

Evidence: each package `Package.swift`; `GOVERNANCE.md`.

## FXDatabase commands

Build and launch:

```bash
swift build --package-path FXDatabase -c release
swift run --package-path FXDatabase FXDatabase
swift run --package-path FXDatabase FXDatabase help
```

Common command forms:

```bash
swift run --package-path FXDatabase FXDatabase startcheck --config-dir Config --migrations-dir Migrations --skip-bridge
swift run --package-path FXDatabase FXDatabase startcheck --config-dir Config --migrations-dir Migrations
swift run --package-path FXDatabase FXDatabase symbol-check --config-dir Config
swift run --package-path FXDatabase FXDatabase backfill --config-dir Config --symbols all
swift run --package-path FXDatabase FXDatabase verify --config-dir Config --random-ranges 0
swift run --package-path FXDatabase FXDatabase verify --config-dir Config --random-ranges 20
swift run --package-path FXDatabase FXDatabase fxbacktest-api --config-dir Config --api-host 127.0.0.1 --api-port 5066
swift run --package-path FXDatabase FXDatabase live --config-dir Config
swift run --package-path FXDatabase FXDatabase supervise --config-dir Config
swift run --package-path FXDatabase FXDatabase sinetest-sync
swift run --package-path FXDatabase FXDatabase sinetest-sync --watch
```

Important: `backtest`, `optimize`, and `export-cache` are intentionally unavailable in FXDatabase. Use `fxbacktest-api` and run strategy work in an external app.

Evidence:
- `FXDatabase/README.md`
- `FXDatabase/Sources/FXDatabaseCLI/CLIOptions.swift`
- `FXDatabase/Sources/FXDatabaseCLI/Command.swift`

## FXDatabase config/bootstrap

Copy local configs before non-demo FXDatabase runs:

```bash
mkdir -p FXDatabase/Config
cp FXDatabase/ConfigSamples/app.sample.json FXDatabase/Config/app.json
cp FXDatabase/ConfigSamples/clickhouse.sample.json FXDatabase/Config/clickhouse.json
cp FXDatabase/ConfigSamples/mt5_bridge.sample.json FXDatabase/Config/mt5_bridge.json
cp FXDatabase/ConfigSamples/symbols.sample.json FXDatabase/Config/symbols.json
cp FXDatabase/ConfigSamples/history_data.sample.json FXDatabase/Config/history_data.json
```

`FXDatabase/Config/` is ignored by Git and may contain local secrets. Remote ClickHouse passwords should use `passwordEnvironmentVariable` where possible.

Evidence:
- `FXDatabase/README.md`
- `FXDatabase/ConfigSamples/clickhouse.sample.json`
- `.gitignore`

## FXBacktest commands

```bash
swift test --package-path FXBacktest
swift build --package-path FXBacktest -c release
swift run --package-path FXBacktest FXBacktest
```

Inside the resident `>` prompt, documented commands include:

```text
status
agents
config
plugins
plugin <plugin-id-or-display-name>
params
set --api-url http://127.0.0.1:5066 --target both --workers 8
set-param <key> --input 12 --min 6 --step 2 --max 40
load-demo
load-fxdatabase [--api-url URL] [--broker ID] [--symbol EURUSD] [--symbols EURUSD,USDJPY] [--from UTC] [--to UTC]
run [cpu|gpu|metal|both] [--workers N] [--chunk N]
save-results [--run-id ID] [--note TEXT]
clean-backtest-data --older-than-days 30
stop
reset-params
help
exit
```

Evidence:
- `FXBacktest/README.md`
- `FXBacktest/Sources/FXBacktest/FXBacktestApp.swift`

## FXDataEngine and TestLab

```bash
swift test --package-path FXDataEngine
swift build --package-path FXDataEngine -c release
swift run --package-path FXDataEngine FXDataEngineCLI
python3.12 FXDataEngine/Tools/fxai_testlab.py compile-audit
python3.12 FXDataEngine/Tools/fxai_testlab.py compile-plugins
python3.12 FXDataEngine/Tools/fxai_testlab.py compile-main
python3.12 FXDataEngine/Tools/fxai_testlab.py doctor
python3.12 FXDataEngine/Tools/fxai_testlab.py verify-all
python3.12 FXDataEngine/Tools/fxai_testlab.py run-audit --scenario-list "{market_walkforward, market_trend, market_chop}" --wf-train-years 1 --wf-test-years 0.25 --wf-window-mode rolling
python3.12 FXDataEngine/Tools/fxai_testlab.py walkforward-analyze
```

Evidence:
- `FXDataEngine/README.md`
- `FXDataEngine/Sources/FXDataEngineCLI/main.swift`
- `FXDataEngine/Tools/testlab/cli.py`

## Offline Lab

Common documented commands:

```bash
python3 FXDataEngine/Tools/fxai_offline_lab.py validate-env
python3 FXDataEngine/Tools/fxai_offline_lab.py bootstrap --seed-demo
python3 FXDataEngine/Tools/fxai_offline_lab.py init-db
python3 FXDataEngine/Tools/fxai_offline_lab.py export-dataset --symbol-pack majors --months-list 3,6,12
python3 FXDataEngine/Tools/fxai_offline_lab.py tune-zoo --profile continuous --auto-export --symbol-pack majors --months-list 3,6,12
python3 FXDataEngine/Tools/fxai_offline_lab.py best-params --profile continuous
python3 FXDataEngine/Tools/fxai_offline_lab.py deploy-profiles --profile continuous
python3 FXDataEngine/Tools/fxai_offline_lab.py supervisor-sync --profile continuous
python3 FXDataEngine/Tools/fxai_offline_lab.py lineage-report --profile continuous
python3 FXDataEngine/Tools/fxai_offline_lab.py minimal-bundle --profile continuous
python3 FXDataEngine/Tools/fxai_offline_lab.py recover-artifacts --profile continuous
```

Do not run destructive or long-running research commands without a task-specific plan and environment confirmation.

Evidence:
- `FXDataEngine/Tools/OfflineLab/README.md`

## FXPlugins backend-focused tests

```bash
swift test --package-path FXPlugins
FXAI_PYTHON=/opt/homebrew/opt/python@3.12/libexec/bin/python3 swift test --package-path FXPlugins --filter PluginRuntimeIntegrationTests
FXAI_PYTHON=/opt/homebrew/opt/python@3.12/libexec/bin/python3 swift test --package-path FXPlugins --filter PluginExternalBackendRuntimeTests
```

Evidence:
- `FXPlugins/README.md`

## FXGUI commands

```bash
cd FXGUI
./start.sh
swift test
./Tools/run_gui_validation_suite.sh
./Tools/package_gui_release.sh
```

From repo root equivalents:

```bash
swift test --package-path FXGUI
swift build --package-path FXGUI
```

Evidence:
- `FXGUI/README.md`
- `FXGUI/Docs/FXGUI_RELEASE_CHECKLIST.md`

## Agent commands

```bash
swift run --package-path FXBacktestAgent FXBacktestAgent --self-check
swift test --package-path FXBacktestAgent
swift test --package-path FXDemoAgent
swift test --package-path FXLiveAgent
swift test --package-path FXExecutionContracts
```

Evidence:
- `FXBacktestAgent/Sources/FXBacktestAgent/main.swift`
- `FXDemoAgent/README.md`
- `FXLiveAgent/README.md`
- `GOVERNANCE.md`

## Lint/format/typecheck status

- verified lint-like command: `git diff --check` from `GOVERNANCE.md`.
- no repo-level formatter command was verified.
- no typecheck-only command was verified; use `swift build --package-path <Package>` for Swift compile/typecheck coverage.
- no Docker/local-service command was verified beyond ClickHouse Homebrew start attempts in `install_fxai.sh` and `FXDatabase/README.md`.
