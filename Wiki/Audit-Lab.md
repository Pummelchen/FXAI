# Audit Lab

## Purpose
The Audit Lab is FXAI's plugin inspection and certification system. It exists to test whether a plugin behaves correctly, stays numerically stable, reacts to market structure in a sensible way, and deserves trust before expensive backtests or releases.

## What It Checks
- prediction validity
- behavioral discipline
- state integrity
- market realism
- regression quality
- optimization guidance

## Core Components
### MT5 Audit Runner
File: `FXAI/Tests/FXAI_AuditRunner.mq5`

Runs inside MetaTrader 5 and uses the real FXAI API, plugins, and feature pipeline.

### Audit Core
File: `FXAI/Tests/audit_core.mqh`

Generates synthetic scenarios and market replay samples, then builds the raw audit report.

### Drill-Sergeant Tool
File: `FXAI/Tools/fxai_testlab.py`

Compiles, launches, analyzes, compares baselines, and enforces release gates.

### Plugin Oracle File
File: `FXAI/Tools/plugin_oracles.json`

Defines what each plugin family and plugin is supposed to be good at so the analyzer judges it by the right standard.

## Scenarios
### Synthetic
- random walk
- monotonic trend
- mean reversion
- volatility clusters
- regime shifts

### Market Replay
- `market_recent`
- `market_trend`
- `market_chop`
- `market_session_edges`
- `market_spread_shock`
- `market_walkforward`

## Typical Commands
Compile the main EA:
`python3 "FXAI/Tools/fxai_testlab.py" compile-main`

Compile the audit runner:
`python3 "FXAI/Tools/fxai_testlab.py" compile-audit`

Run the audit suite:
`python3 "FXAI/Tools/fxai_testlab.py" run-audit`

Analyze the latest report:
`python3 "FXAI/Tools/fxai_testlab.py" analyze`

Create an optimization campaign:
`python3 "FXAI/Tools/fxai_testlab.py" optimize-audit`

Save a baseline:
`python3 "FXAI/Tools/fxai_testlab.py" baseline-save --name nightly_a`

Compare against a baseline:
`python3 "FXAI/Tools/fxai_testlab.py" baseline-compare --baseline nightly_a`

Enforce a release gate:
`python3 "FXAI/Tools/fxai_testlab.py" release-gate --baseline nightly_a`

## Unattended MT5 Launch
The Audit Lab supports:
- `--login`
- `--server`
- `--password`

Or these environment variables:
- `FXAI_MT5_LOGIN`
- `FXAI_MT5_SERVER`
- `FXAI_MT5_PASSWORD`

## Outputs
Raw report:
`FILE_COMMON/FXAI/Audit/fxai_audit_report.tsv`

Drill-sergeant report:
`FXAI/Tools/latest_drill_report.md`

Baselines:
`FXAI/Tools/Baselines/`

## Best Practice
1. Compile first.
2. Run focused audits while developing.
3. Run full audits before expensive MT5 cloud work.
4. Save a baseline when behavior is stable.
5. Use the release gate before major commits and releases.
