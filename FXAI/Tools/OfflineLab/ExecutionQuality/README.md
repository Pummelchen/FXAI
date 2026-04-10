# Execution-Quality Forecaster

The Execution-Quality Forecaster is FXAI's shared phase-1 execution-intelligence layer.

It does not try to create alpha. It answers a different question:

`If FXAI wants to trade now, how likely is the market to punish the entry with spread widening, slippage, latency damage, or fragile liquidity?`

## What it predicts

Per symbol, the runtime now forecasts:

- expected spread for the execution horizon
- spread-widening risk
- expected slippage points
- slippage risk
- fill quality score
- latency sensitivity score
- liquidity fragility score
- composite execution quality score
- execution state: `NORMAL | CAUTION | STRESSED | BLOCKED`

## What it consumes

The phase-1 implementation reuses existing FXAI infrastructure instead of duplicating it:

- broker execution trace and EMA stress from `core_broker_execution`
- NewsPulse risk and event posture
- rates-engine repricing posture
- microstructure hostile-execution and liquidity-stress signals
- adaptive-router and dynamic-ensemble context labels

## Runtime artifacts

Runtime files written under `FILE_COMMON/FXAI/Runtime`:

- `execution_quality_config.tsv`
- `execution_quality_memory.tsv`
- `fxai_execution_quality_<SYMBOL>.tsv`
- `fxai_execution_quality_history_<SYMBOL>.ndjson`

OfflineLab files under `Tools/OfflineLab/ExecutionQuality`:

- `execution_quality_config.json`
- `execution_quality_memory.json`
- `Reports/execution_quality_replay_report.json`

## Commands

Validate config and tier memory:

```bash
python3 Tools/fxai_offline_lab.py execution-quality-validate
```

Rebuild replay summary from append-only runtime history:

```bash
python3 Tools/fxai_offline_lab.py execution-quality-replay-report --symbol EURUSD --hours-back 72
```

## Runtime behavior

The forecaster feeds two existing consumers directly:

1. Probabilistic Calibration
   - uses forecasted spread and slippage instead of simple static execution assumptions
   - adds explicit execution-quality reasons into abstention logic

2. Trade Risk and Order Send
   - blocks entries in `BLOCKED`
   - requires stronger enter probability in `CAUTION` or `STRESSED`
   - scales size down in degraded execution states
   - adjusts allowed deviation conservatively from the execution-quality forecast

## Phase-1 design notes

Phase 1 is intentionally deterministic and auditable:

- no exchange-level order book assumptions
- no deep-learning forecaster
- no hidden live online learning
- no duplicate execution daemon

The repo already records the broker-side execution evidence inside MT5, so the best phase-1 design is a runtime scorecard forecaster with tiered OfflineLab priors and replayable history.
