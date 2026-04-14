# FXAI MT5 Project Tree

This directory is the MT5 project subtree that gets synchronized into:

`MQL5/Experts/FXAI`

It is kept self-describing so the live MT5 tree can be inspected without opening the full git repo root.

If you are approaching FXAI as an operator rather than as a framework engineer, start with the versioned handbook in `Wiki/`:
- [`Home`](Wiki/Home.md)
- [`Quick Start By Role`](Wiki/Quick%20Start%20By%20Role.md)
- [`Getting Started`](Wiki/Getting%20Started.md)
- then the subsystem page you actually need first, such as [`Audit Lab`](Wiki/Audit%20Lab.md), [`Offline Lab`](Wiki/Offline%20Lab.md), [`NewsPulse`](Wiki/NewsPulse.md), [`Runtime Control Plane`](Wiki/Runtime%20Control%20Plane.md), or [`GUI`](Wiki/GUI.md)

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
- `Services/FXAI_MicrostructureProbe.mq5`
  MT5 Service that exports live tick-flow, spread, liquidity-stress, stop-run proxy, and session-handoff state for the shared microstructure subsystem.
- `Services/FXAI_CrossAssetProbe.mq5`
  MT5 Service that exports configured indicator-only MT5 symbols for the shared cross-asset macro/liquidity subsystem.
- `Tools/offline_lab/`
  Internal Python package for Offline Lab database, export, campaign, promotion, shadow-fleet ingest, foundation and student bundling, supervisor-service generation, teacher-factory modules, world simulation, and autonomous governance.
- `Tools/OfflineLab/NewsPulse/`
  NewsPulse operator config, local status mirrors, and subsystem documentation.
- `Tools/OfflineLab/AdaptiveRouter/`
  Adaptive Regime Classifier + Plugin Router docs, configs, replay outputs, and implementation notes for the regime-aware plugin-orchestration layer.
- `Tools/OfflineLab/DynamicEnsemble/`
  Dynamic Ensemble / Meta-Learner docs, config, replay outputs, and implementation notes for the post-inference plugin-weighting layer that combines live plugin outputs with shared FXAI context.
- `Tools/OfflineLab/ProbabilisticCalibration/`
  Probabilistic Calibration + Abstention docs, config, tier memory, replay outputs, and runtime decision artifacts for the cost-aware final trade-quality layer.
- `Tools/OfflineLab/ExecutionQuality/`
  Execution-Quality Forecaster docs, config, tier memory, replay outputs, and runtime forecast artifacts for the execution-condition scoring layer that feeds abstention, trade-risk, and order-send controls.
- `Tools/OfflineLab/RatesEngine/`
  Rates / term-structure / policy-path docs, configs, replay outputs, and status artifacts for the shared macro rates subsystem.
- `Tools/OfflineLab/CrossAsset/`
  Cross-asset macro/liquidity docs, configs, replay outputs, and MT5 probe status for the shared global-context subsystem.
- `Tools/OfflineLab/Microstructure/`
  Microstructure subsystem docs, local status mirrors, replay outputs, and MT5 service config for the shared short-horizon execution-state layer.
- `Tools/OfflineLab/LabelEngine/`
  Multi-Horizon Label Engine + Meta-Labeling docs, config, reports, and artifact outputs for shared target construction, cost-aware tradeability labels, and signal-level meta-label research.
- `Tools/OfflineLab/DriftGovernance/`
  Online Drift Detector + Champion/Challenger Governance docs, config, reports, and audit history for plugin-health monitoring, conservative demotion, and promotion-review workflows.
- `Tools/OfflineLab/PairNetwork/`
  Pair-Network / Factor Graph + Portfolio Conflict Resolver docs, config, graph reports, and runtime decision history for portfolio-level exposure coordination and conflict resolution.
- `GUI/`
  Optional macOS 26 SwiftUI operator app for role-based dashboards, plugin-zoo browsing, report exploration, run builders for Audit/Offline/backtest workflows, runtime inspection, promotion review, Research OS control, advanced Metal-backed visual analysis, saved workspace views, onboarding, incident recovery, detached startup, soft reconnect, terminal-first command guidance, and the shared FXAI operator theme system.
- `Wiki/`
  Versioned operator handbook with role-based quick starts, getting-started flows, subsystem guides, and scenario-driven examples.

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
- `Engine/Runtime/Trade/runtime_trade_microstructure.mqh`
  Microstructure runtime adapter that consumes per-pair MT5 tick-flow and liquidity-stress proxy state and applies execution-aware caution or block posture without changing the canonical model input.
- `Engine/Runtime/Trade/runtime_trade_cross_asset_state.mqh`
  Cross-asset runtime adapter that consumes shared pair-level macro/liquidity posture and applies deterministic caution or block state without changing canonical model inputs.
- `Engine/Runtime/Trade/runtime_trade_pair_network.mqh`
  Pair-network runtime adapter that decomposes current and candidate pair exposure, scores redundancy or contradiction, and suppresses or resizes trades before final order approval.
- `Engine/Runtime/runtime_adaptive_router_stage.mqh`
  Adaptive regime classification, plugin suitability scoring, routed posture logic, and append-only runtime history for the state-aware plugin-zoo control plane.
- `Engine/Runtime/runtime_dynamic_ensemble_stage.mqh`
  Dynamic Ensemble runtime stage that scores plugin trust after inference, normalizes participation weights, escalates abstention posture, and writes replayable runtime artifacts for the meta-learner layer.
- `Engine/Runtime/runtime_prob_calibration_stage.mqh`
  Probabilistic calibration runtime stage that maps ensemble output into calibrated probabilities, expected-move estimates, edge after costs, explicit abstention reasons, and replayable decision artifacts.
- `Engine/Runtime/runtime_execution_quality_stage.mqh`
  Execution-Quality runtime stage that forecasts spread widening, slippage, fill quality, latency sensitivity, liquidity fragility, and replayable execution-state artifacts per symbol.
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
- `Tools/offline_lab/dynamic_ensemble_*.py`
  Dynamic Ensemble contracts, config, replay reporting, and CLI validation helpers for the context-aware post-inference plugin-weighting layer.
- `Tools/offline_lab/prob_calibration_*.py`
  Probabilistic calibration contracts, config, tier-memory export, replay reporting, and deterministic reference math for the final cost-aware decision layer.
- `Tools/offline_lab/execution_quality_*.py`
  Execution-Quality contracts, config, tier-memory export, replay reporting, and deterministic reference math for the runtime execution-condition forecaster.
- `Tools/offline_lab/rates_engine*.py`
  Rates Engine contracts, config, operator numeric inputs, policy-path proxy generation, NewsPulse enrichment, daemon loop, replay reporting, and runtime artifact writers.
- `Tools/offline_lab/cross_asset_*.py`
  Cross-asset contracts, config, replay reporting, MT5 probe install helpers, and shared macro/liquidity snapshot generation on top of rates and indicator-proxy inputs.
- `Tools/offline_lab/microstructure_*.py`
  Microstructure contracts, config, MT5 service install helpers, replay reporting, deterministic reference math, and CLI health or validation helpers for the short-horizon execution-state subsystem.
- `Tools/offline_lab/label_engine*.py`
  Multi-horizon label contracts, config, deterministic label math, artifact builders, and CLI helpers for direction, magnitude, time-to-move, tradeability, and signal meta-label generation.
- `Tools/offline_lab/drift_governance*.py`
  Drift-governance contracts, config, deterministic drift math, DB persistence, report builders, challenger evaluation, and CLI helpers for plugin-health monitoring and champion/challenger policy.
- `Tools/offline_lab/pair_network*.py`
  Pair-network contracts, config, deterministic dependency math, structural exposure decomposition, report builders, and CLI helpers for portfolio conflict resolution.
- `GUI/Sources/FXAIGUICore`, `GUI/Sources/FXAIGUIApp`
  Swift package targets for the GUI’s project scanner, runtime and Research OS artifact readers, advanced visualization builders, saved-workspace persistence, onboarding guides, incident builders, design system, navigation shell, operator-theme token/layout/rendering stack, reference-asset parsing, adaptive dashboard components, Phase 2 run builders, Phase 3 runtime/promotion views, Phase 4 Turso/Research OS control surfaces, Phase 5 Metal-backed visualization surfaces, and Phase 6 operator-polish features.
  The GUI also includes an integrated NewsPulse surface for source health, currency heatmap, pair risk, and recent tape visibility.
  It now also includes a Rates Engine surface for provider health, currency policy state, pair divergence, policy tape, and rates-aware trade gates.
  It now also includes a Cross Asset surface for shared macro/liquidity state, pair impact, proxy selection, source health, and transition visibility.
  It now also includes a Microstructure surface for live per-symbol regime, liquidity stress, hostile execution, stop-run proxy flags, session handoff state, and runtime gating reasons.
  It now also includes an Adaptive Router surface for live regime state, plugin weights, suppression reasons, replay counts, and routing transitions by symbol.
  It now also includes a Dynamic Ensemble surface for live post-inference posture, participation weights, suppression state, replay drift, and final-action reasoning by symbol.
  It now also includes a Probabilistic Calibration surface for calibrated probabilities, expected move quantiles, edge-after-costs, selected tier support, and abstention reasons by symbol.
  It now also includes an Execution Quality surface for expected spread, slippage stress, fill quality, latency sensitivity, liquidity fragility, and current execution-state reasons by symbol.
  It now also includes a Label Engine surface for offline artifact review, multi-horizon tradeability rates, meta-label acceptance, top failure reasons, and per-dataset label quality diagnostics.
  It now also includes a Drift Governance surface for plugin-health states, applied or recommended actions, challenger eligibility, reason codes, and per-symbol governance context.
  It now also includes a Pair Network surface for portfolio conflict decisions, currency and factor exposure, top dependency edges, and preferred-expression selections by symbol.

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
- The preferred Dynamic Ensemble validation path is `python3 FXAI/Tools/fxai_offline_lab.py dynamic-ensemble-validate`.
- The preferred Dynamic Ensemble replay path is `python3 FXAI/Tools/fxai_offline_lab.py dynamic-ensemble-replay-report --symbol EURUSD --hours-back 72`.
- The preferred Probabilistic Calibration validation path is `python3 FXAI/Tools/fxai_offline_lab.py prob-calibration-validate`.
- The preferred Probabilistic Calibration replay path is `python3 FXAI/Tools/fxai_offline_lab.py prob-calibration-replay-report --symbol EURUSD --hours-back 72`.
- The preferred Execution-Quality validation path is `python3 FXAI/Tools/fxai_offline_lab.py execution-quality-validate`.
- The preferred Execution-Quality replay path is `python3 FXAI/Tools/fxai_offline_lab.py execution-quality-replay-report --symbol EURUSD --hours-back 72`.
- The preferred Rates Engine validation path is `python3 FXAI/Tools/fxai_offline_lab.py rates-engine-validate`.
- The preferred Rates Engine smoke path is `python3 FXAI/Tools/fxai_offline_lab.py rates-engine-once`.
- The preferred Rates Engine health path is `python3 FXAI/Tools/fxai_offline_lab.py rates-engine-health`.
- The preferred Rates Engine replay path is `python3 FXAI/Tools/fxai_offline_lab.py rates-engine-replay-report --symbol EURUSD --hours-back 72`.
- The preferred Cross Asset validation path is `python3 FXAI/Tools/fxai_offline_lab.py cross-asset-validate`.
- The preferred Cross Asset service install path is `python3 FXAI/Tools/fxai_offline_lab.py cross-asset-install-service`.
- The preferred Cross Asset smoke path is `python3 FXAI/Tools/fxai_offline_lab.py cross-asset-once`.
- The preferred Cross Asset health path is `python3 FXAI/Tools/fxai_offline_lab.py cross-asset-health`.
- The preferred Cross Asset replay path is `python3 FXAI/Tools/fxai_offline_lab.py cross-asset-replay-report --symbol EURUSD --hours-back 72`.
- The preferred Microstructure validation path is `python3 FXAI/Tools/fxai_offline_lab.py microstructure-validate`.
- The preferred Microstructure service install path is `python3 FXAI/Tools/fxai_offline_lab.py microstructure-install-service`.
- The preferred Microstructure health path is `python3 FXAI/Tools/fxai_offline_lab.py microstructure-health`.
- The preferred Microstructure replay path is `python3 FXAI/Tools/fxai_offline_lab.py microstructure-replay-report --symbol EURUSD --hours-back 72`.
- The preferred Label Engine validation path is `python3 FXAI/Tools/fxai_offline_lab.py label-engine-validate`.
- The preferred Label Engine build path is `python3 FXAI/Tools/fxai_offline_lab.py label-engine-build --profile continuous --limit-datasets 1`.
- The preferred Label Engine report path is `python3 FXAI/Tools/fxai_offline_lab.py label-engine-report --profile continuous`.
- The preferred Drift Governance validation path is `python3 FXAI/Tools/fxai_offline_lab.py drift-governance-validate`.
- The preferred Drift Governance cycle path is `python3 FXAI/Tools/fxai_offline_lab.py drift-governance-run --profile continuous`.
- The preferred Drift Governance report path is `python3 FXAI/Tools/fxai_offline_lab.py drift-governance-report --profile continuous`.
- The preferred Pair Network validation path is `python3 FXAI/Tools/fxai_offline_lab.py pair-network-validate`.
- The preferred Pair Network build path is `python3 FXAI/Tools/fxai_offline_lab.py pair-network-build --profile continuous`.
- The preferred Pair Network report path is `python3 FXAI/Tools/fxai_offline_lab.py pair-network-report --profile continuous`.
- The shared TensorCore path now includes a self-supervised foundation encoder, teacher-student transfer heads, hierarchical trade-quality signals, and persistent analog regime memory.
- Feature normalization now uses train-fit artifacts for min/max buffer, z-score, robust, quantile-to-normal, and Yeo-Johnson transforms, while RevIN/DAIN run as sequence-aware payload normalization over the current input plus rolling window; these fitted stats are persisted in runtime artifacts so warmup and live reuse the same scaling contract.
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
- The canonical feature vector already includes robust derived filters and volatility estimators such as quadruple-smoothed DEMA, RSI, ATR/NATR, Parkinson, Rogers-Satchell, Garman-Klass, rolling median plus Hampel, Kalman, and a 2-pole Ehlers Super Smoother.
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
- The Cross-Asset engine is phase-1 safe by design: it adds shared macro/liquidity state, pair-level cross-market gating, GUI visibility, and replayable global-context history without forcing immediate model retraining or duplicating the Rates Engine.
- The Microstructure layer is phase-1 safe by design: it adds MT5-observable tick-flow, spread, stop-run proxy, session-handoff, and hostile-execution gating without pretending to measure a true institutional order book and without forcing model retraining.
- The Adaptive Router is also phase-1 safe by design: it layers regime classification, plugin trust weighting, suppression, and abstention posture above the current zoo without breaking canonical plugin inputs or forcing immediate retraining.
- The Dynamic Ensemble is also phase-1 safe by design: it layers a deterministic meta-learner above the plugin outputs and above the Adaptive Router, reweights participation using calibration and shared context, and can be disabled without breaking canonical plugin contracts or forcing retraining.
- The Probabilistic Calibration layer is also phase-1 safe by design: it sits after the Dynamic Ensemble, calibrates only the final decision path in phase 1, prices spread, slippage, and uncertainty explicitly, and prefers `SKIP` when calibrated edge does not clear the configured safety floor.
- The Execution-Quality Forecaster is also phase-1 safe by design: it turns broker-visible spread, slippage, liquidity, and latency conditions into deterministic execution-state forecasts that tighten abstention, trade-risk, and allowed-deviation logic without changing canonical model inputs.
- The Multi-Horizon Label Engine + Meta-Labeling subsystem is also phase-1 safe by design: it upgrades target construction for training and evaluation with reproducible direction, magnitude, timing, tradeability, and signal-filter labels without forcing a full plugin-zoo architecture rewrite.
- The Online Drift Detector + Champion/Challenger Governance subsystem is also phase-1 safe by design: it monitors live plugin-health decay, writes deterministic governance state, downweights or restricts degraded plugins conservatively, and keeps challenger promotion behind support-aware review rather than unsafe autonomous replacement.
- The Pair-Network / Factor Graph + Portfolio Conflict Resolver subsystem is also phase-1 safe by design: it resolves redundant, contradictory, or over-concentrated pair candidates after calibration and execution-quality scoring using deterministic currency and factor exposure rules rather than opaque portfolio optimization.

## Source Of Truth

- Runtime source of truth: the live MT5 Experts tree
- Versioned mirror: the git repo copy that is synchronized into the MT5 tree after clean verification
- Research source of truth: the Offline Lab Turso/libSQL database
- MT5 runtime artifact source of truth: `FILE_COMMON/FXAI/Offline/Promotions/`
- GUI source of truth: `FXAI/GUI/` inside the versioned repo and synced MT5 subtree

For broader framework usage and workflow details, see the repo-root README and the GitHub wiki.
