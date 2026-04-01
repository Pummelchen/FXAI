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
- `Tools/fxai_offline_lab.py`
  SQLite-backed offline export, tuning, promotion, and control-loop tool.

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
- The shared TensorCore path now includes a self-supervised foundation encoder, teacher-student transfer heads, hierarchical trade-quality signals, and persistent analog regime memory.
- Stateful plugins now persist deterministic replay-backed checkpoint metadata under the `native_model` promotion contract, including replay and hyperparameter integrity checks.
- Stateful runtime manifests now expose checkpoint depth, native snapshot coverage, and deterministic replay flags for release gating.
- Shared transfer warmup now uses a deeper temporal backbone over the rolling window instead of only static current-bar summary features, and runtime artifacts persist that backbone state.
- Broker execution replay now persists raw symbol, side, order-type, reject, partial-fill, latency, fill-ratio, and event-mass libraries for later audit and runtime reuse.
- Ensemble routing now uses persisted regime, session, and horizon-specific value, regret, and counterfactual state, while warmup stores cross-symbol portfolio objectives for promotion-time weighting.
- Audit Lab now includes adversarial market certification on mined hostile `M1 OHLC + spread` windows, in addition to the standard market replay, walk-forward, and macro-event packs.
- Audit Lab now exercises scoped runtime-artifact persistence and conformal calibration state instead of stub no-op hooks.
- Audit scenarios build coherent `OHLC + spread` context bars rather than reconstructing them from close-only shortcuts.
- Macro-event datasets now run under schema version 2 with revision-chain, source-trust, and currency-relevance manifests that feed release-gate checks.
- Offline Lab now exports exact-window `M1 OHLC + spread` datasets into SQLite, stores full tuning ledgers and scenario metrics, and promotes ready-to-use MT5 `.set` files under `Tools/OfflineLab/Profiles/`, `MQL5/Profiles/Tester/`, and `FILE_COMMON/FXAI/Offline/Promotions/`.

## Source Of Truth

- Runtime source of truth: the live MT5 Experts tree
- Versioned mirror: the git repo copy that is synchronized into the MT5 tree after clean verification

For broader framework usage and workflow details, see the repo-root README and the GitHub wiki.
