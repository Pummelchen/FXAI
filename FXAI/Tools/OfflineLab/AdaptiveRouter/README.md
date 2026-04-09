# Adaptive Router

Adaptive Router is the regime-aware plugin-orchestration layer for FXAI.

It sits above the plugin zoo and below the final execution decision. The goal is not to replace plugins. The goal is to decide:

- what regime the symbol is in right now
- which plugins fit that regime best
- which plugins should be downweighted or suppressed
- whether the live posture should stay normal, move to caution, bias toward abstention, or block

## What Phase 1 Does

- classifies a compact live regime state per symbol
- consumes NewsPulse as shared context instead of duplicating news logic
- writes promoted adaptive-router profiles from Offline Lab research artifacts
- applies regime-aware plugin weighting and suppression in the live EA
- writes live runtime state plus append-only history
- exposes replay summaries and current routing state in the macOS GUI

Phase 1 is routing-first and audit-first. It does not force retraining of the plugin zoo.

## Main Files

- `Tools/offline_lab/adaptive_router_contracts.py`
- `Tools/offline_lab/adaptive_router_config.py`
- `Tools/offline_lab/adaptive_router.py`
- `Tools/offline_lab/adaptive_router_replay.py`
- `Engine/Runtime/runtime_adaptive_router_stage.mqh`
- `Engine/Runtime/runtime_model_stage_block.mqh`
- `Engine/Runtime/runtime_policy_stage_block.mqh`

## Operator Commands

Validate config:

```bash
python3 FXAI/Tools/fxai_offline_lab.py adaptive-router-validate
```

Generate promoted profiles:

```bash
python3 FXAI/Tools/fxai_offline_lab.py adaptive-router-profiles --profile continuous
```

Build replay summary:

```bash
python3 FXAI/Tools/fxai_offline_lab.py adaptive-router-replay-report --symbol EURUSD --hours-back 72
```

The full release-gate path still remains:

```bash
python3 FXAI/Tools/fxai_testlab.py verify-all
```

## Runtime Artifacts

Promoted profile:
- `FILE_COMMON/FXAI/Offline/Promotions/fxai_adaptive_router_<symbol>.tsv`

Live runtime state:
- `FILE_COMMON/FXAI/Runtime/fxai_regime_router_<symbol>.tsv`

Append-only live history:
- `FILE_COMMON/FXAI/Runtime/fxai_regime_router_history_<symbol>.ndjson`

Research-side JSON profile:
- `Tools/OfflineLab/ResearchOS/<profile>/adaptive_router_<symbol>.json`

Replay report:
- `Tools/OfflineLab/AdaptiveRouter/Reports/adaptive_router_replay_report.json`

## Disable / Fallback

Live fallback is explicit:

- set `AdaptiveRouterEnabled = false` in `FXAI.mq5` to recover the old behavior
- or leave promoted profiles absent and the runtime falls back cleanly
- profile config can also enforce `fallback_to_student_router_only`

## Why It Exists

FX plugins do not perform equally well in all conditions.

Examples:
- trend-friendly plugins are stronger in persistent directional phases
- range-oriented plugins are stronger in mean-reverting phases
- macro-sensitive plugins matter more in event and risk-repricing regimes
- fragile plugins should be trusted less when spread or liquidity conditions degrade

Adaptive Router gives FXAI one consistent, auditable place to make those decisions.
