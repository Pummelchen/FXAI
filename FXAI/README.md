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
  Stable CLI wrapper for the SQLite-backed offline export, tuning, promotion, and control-loop tool.
- `Tools/offline_lab/`
  Internal Python package for Offline Lab database, export, campaign, promotion, shadow-fleet ingest, foundation and student bundling, supervisor-service generation, teacher-factory modules, world simulation, and autonomous governance.

## Key Runtime Areas

- `API/`
  Plugin contracts, runtime context helpers, and TensorCore bridges.
- `API/Context/`
  Split plugin-context internals for state, payload, transfer, quality, replay, and runtime projection helpers.
- `Engine/`
  Data loading, feature building, normalization, warmup, runtime orchestration, persistence, and meta layers.
- `Engine/Core/`
  Shared core helpers split out from `Engine/core.mqh` for analog memory, broker replay, model context, schema handling, and request validation.
- `Engine/Lifecycle/`
  Split lifecycle internals for context-symbol discovery, reset and recovery, bootstrap, and compliance or promotion-readiness checks.
- `Engine/Warmup/`
  Warmup modules split by normalization search, scoring, transfer, portfolio diagnostics, and entrypoint orchestration.
- `Engine/Runtime/`
  Extracted live-trading helpers used by the EA entrypoint, including the live control-plane snapshot and peer-pressure logic.
- `TensorCore/`
  Internal neural runtime shared by the stronger sequence and tensor-heavy plugins.
- `Plugins/`
  Model families and plugin implementations.
- `Plugins/Sequence/ai_*/`
  Internal split state and public sections for the largest sequence-model plugins, including `ai_tcn/`, `ai_s4/`, and `ai_stmn/`.
- `Plugins/Tree/tree_catboost/`, `Plugins/Tree/tree_lgbm/`
  Internal split class sections for the largest tree-model plugins.
- `Tests/`
  Audit Lab scenarios, scoring, reports, and TensorCore sanity checks.
- `Tests/Scoring/`
  Split Audit Lab scoring internals for core helpers, adversarial packs, metrics, and scenario execution.
- `Tools/testlab/`
  Internal Python package behind `fxai_testlab.py`, split into compile, audit-run, reporting, baseline, optimization, release-gate, and CLI modules.

## Operating Notes

- Canonical research data is `M1 OHLC + spread`.
- The shared TensorCore path now includes a self-supervised foundation encoder, teacher-student transfer heads, hierarchical trade-quality signals, and persistent analog regime memory.
- The live EA now uses portfolio-native sizing and gating with directional-cluster pressure, hierarchy floors, and macro-state quality controls instead of only scalar conviction scaling.
- The live runtime now emits per-instance control-plane snapshots and consumes promoted symbol deployment profiles so research-side promotion decisions can steer trade floors, sizing bias, and peer-pressure handling.
- The live runtime now also consumes promoted supervisor-service artifacts and policy lifecycle thresholds so add, reduce, tighten, timeout, and exit behavior can be governed by the research OS rather than static EA-only logic.
- Stateful plugins now persist deterministic replay-backed checkpoint metadata under the `native_model` promotion contract, including replay and hyperparameter integrity checks.
- Stateful runtime manifests now expose checkpoint depth, native snapshot coverage, and deterministic replay flags for release gating.
- Shared transfer warmup now uses a deeper temporal backbone over the rolling window instead of only static current-bar summary features, and runtime artifacts persist that backbone state.
- Broker execution replay now persists raw symbol, side, order-type, reject, partial-fill, latency, fill-ratio, and event-mass libraries for later audit and runtime reuse.
- Ensemble routing now uses persisted regime, session, and horizon-specific value, regret, and counterfactual state, while warmup stores cross-symbol portfolio objectives for promotion-time weighting.
- Audit Lab now includes adversarial market certification on mined hostile `M1 OHLC + spread` windows, in addition to the standard market replay, walk-forward, and macro-event packs.
- Audit Lab now exercises scoped runtime-artifact persistence and conformal calibration state instead of stub no-op hooks.
- Audit scenarios build coherent `OHLC + spread` context bars rather than reconstructing them from close-only shortcuts.
- Macro-event datasets now run under schema version 2 with revision-chain, source-trust, and currency-relevance manifests, and the feature pipeline derives a higher-level macro state for policy, inflation, labor, growth, carry, decay, and quality.
- Offline Lab now exports exact-window `M1 OHLC + spread` datasets into SQLite, stores full tuning ledgers and scenario metrics, and promotes ready-to-use MT5 `.set` files under `Tools/OfflineLab/Profiles/`, `MQL5/Profiles/Tester/`, and `FILE_COMMON/FXAI/Offline/Promotions/`.
- Offline Lab also maintains champion/challenger governance, parameter lineage, family scorecards, distillation artifacts, teacher-factory payloads, live deployment profiles, shadow-fleet telemetry, and learned red-team plans under `Tools/OfflineLab/ResearchOS/` and `Tools/OfflineLab/Distillation/`.
- Offline Lab now emits foundation-teacher artifacts, portfolio-supervisor profiles, and per-symbol world-simulator plans that are consumed by MT5 runtime control-plane logic and Audit Lab adversarial generation.
- Offline Lab now also emits student deployment bundles and per-symbol or global supervisor-service artifacts so the promoted runtime can blend peer pressure, capital budget, add or reduce bias, and lifecycle floors without manual operator edits.
- Offline Lab promotion is profile-wide by default in `best-params`, unless a symbol filter is passed, and exact-window audit runs now follow the effective exported first/last bar range.

## Source Of Truth

- Runtime source of truth: the live MT5 Experts tree
- Versioned mirror: the git repo copy that is synchronized into the MT5 tree after clean verification

For broader framework usage and workflow details, see the repo-root README and the GitHub wiki.
