# FXAI MT5 Project Tree

This directory is the MT5 project subtree that gets synchronized into:

`MQL5/Experts/FXAI`

It is kept self-describing so the live MT5 tree can be inspected without opening the full git repo root.

## Main Entry Points

- `FXAI.mq5`
  Main Expert Advisor entry point for live trading and Strategy Tester runs.
- `Tests/FXAI_AuditRunner.mq5`
  MT5-side Audit Lab runner.
- `Tools/fxai_testlab.py`
  External compile, audit, baseline, and release-gate tool.

## Key Runtime Areas

- `API/`
  Plugin contracts, runtime context helpers, and TensorCore bridges.
- `Engine/`
  Data loading, feature building, normalization, warmup, runtime orchestration, persistence, and meta layers.
- `TensorCore/`
  Internal neural runtime shared by the stronger sequence and tensor-heavy plugins.
- `Plugins/`
  Model families and plugin implementations.
- `Tests/`
  Audit Lab scenarios, scoring, reports, and TensorCore sanity checks.

## Operating Notes

- Canonical research data is `M1 OHLC + spread`.
- Stateful plugins now require `native_model` checkpoint coverage for live promotion.
- Audit Lab now exercises scoped runtime-artifact persistence and conformal calibration state instead of stub no-op hooks.
- Audit scenarios build coherent `OHLC + spread` context bars rather than reconstructing them from close-only shortcuts.

## Source Of Truth

- Runtime source of truth: the live MT5 Experts tree
- Versioned mirror: the git repo copy that is synchronized into the MT5 tree after clean verification

For broader framework usage and workflow details, see the repo-root README and the GitHub wiki.
