# Execution-Quality Forecaster

## Repo Mapping

This subsystem is integrated as a live FXDatabase runtime stage because FXAI already records broker execution telemetry inside the EA and already routes shared decision-quality overlays through `FXDataEngine/Engine/Runtime`.

### Runtime boundary reused

- live execution and broker telemetry:
- shared context sources consumed, not duplicated:
- downstream decision-quality and risk consumers:

### Repo-specific deviation from the brief

The repo already maintains execution traces and execution EMAs directly inside the FXDatabase runtime. Because of that, phase 1 does **not** add a separate external execution daemon. The forecaster lives in the EA runtime so it can:

- see the latest broker execution trace immediately
- avoid duplicate transport or freshness lag
- reuse the existing runtime artifact pattern already used by Dynamic Ensemble and Probabilistic Calibration

OfflineLab still owns:

- config
- tier memory
- replay reports
- validation and operator docs

### Chosen insertion point


1. adaptive router posture
2. dynamic ensemble posture
3. current shared context load

and before:

1. probabilistic calibration and abstention
2. trade-risk lot admission
3. final market-order deviation selection

That placement lets the subsystem enrich the existing cost-aware final decision path without changing plugin APIs.

### Artifact pattern reused

Artifacts added:

- `FILE_COMMON/FXAI/Runtime/execution_quality_config.tsv`
- `FILE_COMMON/FXAI/Runtime/execution_quality_memory.tsv`
- `FILE_COMMON/FXAI/Runtime/fxai_execution_quality_<SYMBOL>.tsv`
- `FILE_COMMON/FXAI/Runtime/fxai_execution_quality_history_<SYMBOL>.ndjson`
- `Tools/OfflineLab/ExecutionQuality/execution_quality_config.json`
- `Tools/OfflineLab/ExecutionQuality/execution_quality_memory.json`
- `Tools/OfflineLab/ExecutionQuality/Reports/execution_quality_replay_report.json`

### Forecast approach in phase 1

Phase 1 is a deterministic scorecard forecaster, not a black-box learned model.

It combines:

- price-cost and broker execution trace stress
- NewsPulse event and risk windows
- rates repricing and policy-stress context
- microstructure hostile-execution and liquidity-stress context
- session and handoff thinness
- tiered priors exported from OfflineLab

Outputs:

- expected price cost
- price-cost widening risk
- expected slippage
- slippage risk
- fill quality score
- latency sensitivity score
- liquidity fragility score
- composite execution quality state
- machine-readable reason codes

### Phase-1 scope

Phase 1 ships:

- deterministic runtime scorecard forecaster
- tiered prior selection with fallback and support flags
- replayable runtime history
- calibration-layer integration
- trade-risk and order-send integration
- GUI surface and replay report

Deferred:

- offline supervised fitting from richer realized execution labels
- champion/challenger execution-quality models
- explicit order-type routing beyond allowed-deviation control
