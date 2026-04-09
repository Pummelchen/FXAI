# FXAI MT5 Project Tree

This directory is the MT5 project subtree that gets synchronized into:

`MQL5/Experts/FXAI`

It is kept self-describing so the live MT5 tree can be inspected without opening the full git repo root.

If you are approaching FXAI as an operator rather than as a framework engineer, start with the role-based quick start in the wiki:
- `Quick Start By Role`
- then `Getting Started`
- then the subsystem page you actually need first, such as `Audit Lab`, `Offline Lab`, `NewsPulse`, or `GUI`

## Main Entry Points

- `FXAI.mq5`
  Main Expert Advisor entry point for live trading and Strategy Tester runs.
- `Tests/FXAI_AuditRunner.mq5`
  MT5-side Audit Lab runner.
- `Tools/fxai_testlab.py`
  External compile, audit, baseline, and release-gate tool.
- `Tools/fxai_offline_lab.py`
  Stable CLI wrapper for the Turso/libSQL-backed offline export, tuning, promotion, and control-loop tool.
- `Services/FXAI_NewsPulseCalendar.mq5`
  MT5 Service that exports Economic Calendar state for the NewsPulse shared news-risk subsystem.
- `Tools/offline_lab/`
  Internal Python package for Offline Lab database, export, campaign, promotion, shadow-fleet ingest, foundation and student bundling, supervisor-service generation, teacher-factory modules, world simulation, and autonomous governance.
- `Tools/OfflineLab/NewsPulse/`
  NewsPulse operator config, local status mirrors, and subsystem documentation.
- `Tools/OfflineLab/AdaptiveRouter/`
  Adaptive Regime Classifier + Plugin Router docs, configs, replay outputs, and implementation notes for the regime-aware plugin-orchestration layer.
- `Tools/OfflineLab/RatesEngine/`
  Rates / term-structure / policy-path docs, configs, replay outputs, and status artifacts for the shared macro rates subsystem.
- `GUI/`
  Optional macOS 26 SwiftUI operator app for role-based dashboards, plugin-zoo browsing, report exploration, run builders for Audit/Offline/backtest workflows, runtime inspection, promotion review, Research OS control, advanced Metal-backed visual analysis, saved workspace views, onboarding, incident recovery, detached startup, soft reconnect, terminal-first command guidance, and the shared FXAI operator theme system.

## Key Runtime Areas

- `API/`
  Plugin contracts, runtime context helpers, and TensorCore bridges.
- `API/Contract/`
  Split plugin-contract internals for support types, public API surface, and persistence wiring behind `plugin_contract.mqh`.
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
- `Engine/Runtime/Trade/runtime_trade_newspulse.mqh`
  NewsPulse runtime adapter that consumes the merged flat snapshot and applies pair-level news gating without changing the canonical model input.
- `Engine/Runtime/Trade/runtime_trade_rates_engine.mqh`
  Rates Engine runtime adapter that consumes the pair-level flat snapshot and applies rates-aware caution or block posture without changing the canonical model input.
- `Engine/Runtime/runtime_adaptive_router_stage.mqh`
  Adaptive regime classification, plugin suitability scoring, routed posture logic, and append-only runtime history for the state-aware plugin-zoo control plane.
- `Engine/Runtime/runtime_*_block.mqh`
  Split feature, transfer, model, and policy stages behind `engine_runtime.mqh` so the main runtime entrypoint stays readable.
- `TensorCore/`
  Internal neural runtime shared by the stronger sequence and tensor-heavy plugins.
- `TensorCore/Transfer/`
  Split shared transfer-backbone internals for globals, temporal pooling, feature encoding, and model heads behind `tensor_transfer.mqh`.
- `Plugins/`
  Model families and plugin implementations.
- `Plugins/Sequence/ai_*/`
  Internal split state and public sections for the largest sequence-model plugins, including `ai_tcn/`, `ai_s4/`, and `ai_stmn/`. The sequence zoo also now includes the new native research plugins `ai_qcew`, `ai_fewc`, `ai_gha`, and `ai_tesseract`.
- `Plugins/Sequence/ai_tft/Forward/`
  Split TFT forward-pass helpers into utility, sequence, and head-specific modules.
- `Plugins/Sequence/ai_patchtst/`
  Split PatchTST private, public, and training sections behind `ai_patchtst.mqh`.
- `Plugins/Linear/lin_pa/`
  Split passive-aggressive plugin internals into private, public, and training sections.
- `Plugins/Tree/tree_catboost/`, `Plugins/Tree/tree_lgbm/`
  Internal split class sections for the largest tree-model plugins.
- `Tests/`
  Audit Lab scenarios, scoring, reports, and TensorCore sanity checks.
- `Tests/Scoring/`
  Split Audit Lab scoring internals for core helpers, adversarial packs, metrics, and scenario execution.
- `Tools/testlab/`
  Internal Python package behind `fxai_testlab.py`, split into compile, audit-run, reporting, baseline, optimization, release-gate, and CLI modules.
- `Tools/offline_lab/cli_*.py`
  Split Offline Lab command routing into campaign, command, and parser modules behind `cli.py`.
- `Tools/offline_lab/common_*.py`
  Split shared Offline Lab helpers into schema, utilities, DB, statistics, and path modules behind `common.py`.
- `Tools/offline_lab/market_universe.py`
  Offline Lab market-universe config stored in Turso/libSQL metadata, including the FX-only tradable universe and indicator-only MT5 context symbols.
- `Tools/offline_lab/newspulse_*.py`
  NewsPulse contracts, config, operator policy, MT5 calendar parsing, GDELT polling, official-feed ingestion, fusion, replay helpers, daemon loop, and service-install helpers.
- `Tools/offline_lab/adaptive_router*.py`
  Adaptive Router contracts, config, regime and plugin-prior generation, profile writing, replay reporting, and CLI validation helpers for the regime-aware plugin-routing layer.
- `Tools/offline_lab/rates_engine*.py`
  Rates Engine contracts, config, operator numeric inputs, policy-path proxy generation, NewsPulse enrichment, daemon loop, replay reporting, and runtime artifact writers.
- `GUI/Sources/FXAIGUICore`, `GUI/Sources/FXAIGUIApp`
  Swift package targets for the GUI’s project scanner, runtime and Research OS artifact readers, advanced visualization builders, saved-workspace persistence, onboarding guides, incident builders, design system, navigation shell, operator-theme token/layout/rendering stack, reference-asset parsing, adaptive dashboard components, Phase 2 run builders, Phase 3 runtime/promotion views, Phase 4 Turso/Research OS control surfaces, Phase 5 Metal-backed visualization surfaces, and Phase 6 operator-polish features.
  The GUI also includes an integrated NewsPulse surface for source health, currency heatmap, pair risk, and recent tape visibility.
  It now also includes a Rates Engine surface for provider health, currency policy state, pair divergence, policy tape, and rates-aware trade gates.
  It now also includes an Adaptive Router surface for live regime state, plugin weights, suppression reasons, replay counts, and routing transitions by symbol.

## Operating Notes

- Canonical research data is `M1 OHLC + spread`.
- The preferred platform verification path is `python3 FXAI/Tools/fxai_testlab.py verify-all`.
- The preferred GUI verification path is `cd FXAI/GUI && swift test && swift build`.
- GUI release packaging is `cd FXAI/GUI && ./Tools/package_gui_release.sh`.
- The preferred Offline Lab bootstrap path is `python3 FXAI/Tools/fxai_offline_lab.py bootstrap --seed-demo`.
- The preferred market-universe inspection path is `python3 FXAI/Tools/fxai_offline_lab.py market-universe-show`.
- The preferred market-universe export path is `python3 FXAI/Tools/fxai_offline_lab.py market-universe-export`.
- The preferred NewsPulse service install path is `python3 FXAI/Tools/fxai_offline_lab.py newspulse-install-service`.
- The preferred NewsPulse smoke path is `python3 FXAI/Tools/fxai_offline_lab.py newspulse-once`.
- The preferred NewsPulse health path is `python3 FXAI/Tools/fxai_offline_lab.py newspulse-health`.
- The preferred Adaptive Router validation path is `python3 FXAI/Tools/fxai_offline_lab.py adaptive-router-validate`.
- The preferred Adaptive Router promotion path is `python3 FXAI/Tools/fxai_offline_lab.py adaptive-router-profiles --profile continuous`.
- The preferred Adaptive Router replay path is `python3 FXAI/Tools/fxai_offline_lab.py adaptive-router-replay-report --symbol EURUSD --hours-back 72`.
- The preferred Rates Engine validation path is `python3 FXAI/Tools/fxai_offline_lab.py rates-engine-validate`.
- The preferred Rates Engine smoke path is `python3 FXAI/Tools/fxai_offline_lab.py rates-engine-once`.
- The preferred Rates Engine health path is `python3 FXAI/Tools/fxai_offline_lab.py rates-engine-health`.
- The preferred Rates Engine replay path is `python3 FXAI/Tools/fxai_offline_lab.py rates-engine-replay-report --symbol EURUSD --hours-back 72`.
- The shared TensorCore path now includes a self-supervised foundation encoder, teacher-student transfer heads, hierarchical trade-quality signals, and persistent analog regime memory.
- The live EA now uses portfolio-native sizing and gating with directional-cluster pressure, hierarchy floors, and macro-state quality controls instead of only scalar conviction scaling.
- The live runtime now emits per-instance control-plane snapshots and consumes promoted symbol deployment profiles so research-side promotion decisions can steer trade floors, sizing bias, and peer-pressure handling.
- The live runtime now also consumes promoted supervisor-service artifacts and policy lifecycle thresholds so add, reduce, tighten, timeout, and exit behavior can be governed by the research OS rather than static EA-only logic.
- Live deployment profiles now carry fitted teacher, student, foundation, macro, and lifecycle gains so runtime trust and adaptation can be steered from shadow telemetry instead of only static profile weights.
- Student-router promotion now includes plugin-level weights and hard pruning, not only family-level breadth limits.
- Supervisor-service and supervisor-command artifacts now include freshness windows, directional entry-budget multipliers, and pressure velocity so stale or one-sided supervisor state is less likely to leak into live admission logic.
- Stateful plugins now persist deterministic replay-backed checkpoint metadata under the `native_model` promotion contract, including replay and hyperparameter integrity checks.
- Stateful runtime manifests now expose checkpoint depth, native snapshot coverage, and deterministic replay flags for release gating.
- Shared transfer warmup now uses a deeper temporal backbone over the rolling window instead of only static current-bar summary features, and runtime artifacts persist that backbone state.
- Broker execution replay now persists raw symbol, side, order-type, reject, partial-fill, latency, fill-ratio, and event-mass libraries for later audit and runtime reuse.
- Ensemble routing now uses persisted regime, session, and horizon-specific value, regret, and counterfactual state, while warmup stores cross-symbol portfolio objectives for promotion-time weighting.
- Audit Lab now includes adversarial market certification on mined hostile `M1 OHLC + spread` windows, in addition to the standard market replay, walk-forward, and macro-event packs.
- Audit Lab now exercises scoped runtime-artifact persistence and conformal calibration state instead of stub no-op hooks.
- Audit scenarios build coherent `OHLC + spread` context bars rather than reconstructing them from close-only shortcuts.
- Macro-event datasets now run under schema version 2 with revision-chain, source-trust, and currency-relevance manifests, and the feature pipeline derives a higher-level macro state for policy, inflation, labor, growth, carry, decay, and quality.
- Offline Lab now exports exact-window `M1 OHLC + spread` datasets into Turso/libSQL, stores full tuning ledgers and scenario metrics, and promotes ready-to-use MT5 `.set` files under `Tools/OfflineLab/Profiles/`, `MQL5/Profiles/Tester/`, and `FILE_COMMON/FXAI/Offline/Promotions/`.
- Offline Lab also maintains champion/challenger governance, parameter lineage, family scorecards, distillation artifacts, teacher-factory payloads, live deployment profiles, shadow-fleet telemetry, and learned red-team plans under `Tools/OfflineLab/ResearchOS/` and `Tools/OfflineLab/Distillation/`.
- Offline Lab now emits foundation-teacher artifacts, portfolio-supervisor profiles, and per-symbol world-simulator plans that are consumed by MT5 runtime control-plane logic and Audit Lab adversarial generation.
- World-simulator plans now include transition entropy, shock decay, and session-specific sigma/spread scaling learned from exported market windows instead of only coarse global stress factors.
- Offline Lab now also emits student deployment bundles and per-symbol or global supervisor-service artifacts so the promoted runtime can blend peer pressure, capital budget, add or reduce bias, and lifecycle floors without manual operator edits.
- Offline Lab promotion is profile-wide by default in `best-params`, unless a symbol filter is passed, and exact-window audit runs now follow the effective exported first/last bar range.
- Offline Lab now also emits operator dashboards, lineage reports, deterministic fixture artifacts, and minimal live bundles so operators can inspect or recover the promoted state without reading Turso/libSQL directly.
- Offline Lab can run local-only through libSQL or as a Turso embedded replica when `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` are configured.
- Offline Lab now also supports local Turso file encryption, bounded sync intervals for embedded replicas, branch and point-in-time restore env artifacts, Turso audit-log ingestion, and native vector-backed analog-state retrieval in the research OS.
- Runtime profiles now support explicit `research` and `production` modes so the same framework can run either as the full research OS or as a leaner live deployment surface.
- The default market-universe policy is `FX_ONLY`: FX pairs are tradable, while non-FX MT5 symbols are stored as indicator-only context instruments in the Offline Lab database configuration.
- NewsPulse is phase-1 safe by design: it adds shared news-risk gating, optional official-feed monitoring, replay timelines, operator-editable pair policy, and GUI drill-down visibility without forcing model retraining or changing the canonical model-input contract by default.
- The Rates Engine is phase-1 safe by design: it adds shared rates-aware macro gating, NewsPulse enrichment, GUI visibility, and replayable policy-path state without forcing immediate model retraining. Phase 1 supports true-market numeric inputs when operators have them and otherwise falls back to a clearly labeled NewsPulse-driven policy proxy.
- The Adaptive Router is also phase-1 safe by design: it layers regime classification, plugin trust weighting, suppression, and abstention posture above the current zoo without breaking canonical plugin inputs or forcing immediate retraining.

## Source Of Truth

- Runtime source of truth: the live MT5 Experts tree
- Versioned mirror: the git repo copy that is synchronized into the MT5 tree after clean verification
- Research source of truth: the Offline Lab Turso/libSQL database
- MT5 runtime artifact source of truth: `FILE_COMMON/FXAI/Offline/Promotions/`
- GUI source of truth: `FXAI/GUI/` inside the versioned repo and synced MT5 subtree

For broader framework usage and workflow details, see the repo-root README and the GitHub wiki.
