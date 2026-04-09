# Dynamic Ensemble / Meta-Learner Implementation Plan

## Goal
Build a deterministic, auditable, post-inference dynamic ensemble layer for the FXAI plugin zoo that reweights, downweights, or suppresses plugins based on current context and recent plugin quality, then raises SKIP / abstain bias when the ensemble quality is weak.

This plan is phase-1 production scope. It intentionally reuses the existing FXAI control-plane pieces instead of duplicating them.

## Repo Mapping

### Existing runtime seams
- `Engine/Runtime/runtime_feature_pipeline_block.mqh`
  - already assembles `student_router`, `adaptive_router_profile`, `adaptive_regime_state`, `adaptive_news_state`, `adaptive_micro_state`, and the core feature/cost context.
- `Engine/Runtime/runtime_model_stage_block.mqh`
  - already selects active models, runs plugin inference, applies adaptive router priors, and builds the current weighted ensemble statistics.
  - this is the correct place to collect plugin outputs and compute dynamic per-plugin trust.
- `Engine/Runtime/runtime_policy_stage_block.mqh`
  - already converts aggregate ensemble state into a final `BUY / SELL / SKIP` decision and applies adaptive-router abstain posture.
  - this is the correct place to apply dynamic-ensemble quality and abstain bias to the final decision.
- `Engine/Runtime/Trade/runtime_trade_risk.mqh`
  - remains the final hard execution-risk layer and must not be replaced.

### Existing plugin-quality memory already available
FXAI already persists rolling per-plugin diagnostics in the MQL meta layer:
- `g_model_reliability`
- regime edge EMA
- context edge EMA
- context regret EMA
- context observation counts
- portfolio objective/stability/correlation/diversification
- route-factor diagnostics

Relevant helpers already exist in:
- `Engine/meta_calibration.mqh`
- `Engine/meta_reliability.mqh`
- `Engine/runtime_artifacts.mqh`

This means phase 1 does not need a separate live online-learning daemon. It can reuse these existing per-plugin memory signals directly.

### Existing shared context to reuse
- Adaptive Router
- NewsPulse
- Rates Engine
- Microstructure

The Dynamic Ensemble must consume these outputs, not rebuild them.

### Existing Offline Lab / GUI patterns to reuse
- contracts/config/replay helpers under `Tools/offline_lab/*`
- local docs/config under `Tools/OfflineLab/<Subsystem>/`
- GUI reader in `GUI/Sources/FXAIGUICore/Services/*ArtifactReader.swift`
- GUI models in `GUI/Sources/FXAIGUICore/Models/*`
- GUI view in `GUI/Sources/FXAIGUIApp/Features/*`

## Phase-1 Scope

### Included
1. Runtime dynamic ensemble stage after plugin inference and before final decision.
2. Config-driven trust-scoring and suppression thresholds.
3. Participation map with `ACTIVE`, `DOWNWEIGHTED`, `SUPPRESSED`, `EXCLUDED`.
4. Ensemble-quality score and abstain bias.
5. Runtime TSV + NDJSON history artifacts.
6. Offline Lab validation + replay-report helpers.
7. macOS GUI reader and operator surface.
8. Deterministic tests for config/replay/GUI parsing.

### Explicitly deferred
1. Learned meta-models.
2. Reinforcement learning / contextual bandits.
3. New model-input feature-vector expansion.
4. Separate daemon/service layer.
5. Portfolio-level multi-symbol meta-ensemble coupling beyond existing FXAI portfolio diagnostics.

## Runtime Design

### A. New runtime stage file
Add:
- `Engine/Runtime/runtime_dynamic_ensemble_stage.mqh`

Responsibilities:
- define dynamic-ensemble constants and structs
- compute per-plugin trust components
- compute disagreement/instability penalties
- normalize weights
- classify participation state
- compute ensemble quality and abstain bias
- write per-symbol runtime state TSV
- append NDJSON history

### B. New runtime contracts
Per-plugin runtime record should include at least:
- `ai_idx`
- `ai_name`
- `family_id`
- `signal`
- `buy_prob`, `sell_prob`, `skip_prob`
- `expected_move`
- `confidence`
- `reliability`
- `buy_ev`, `sell_ev`
- `base_meta_weight`
- `adaptive_suitability`
- `context_edge_norm`
- `context_regret`
- `portfolio_stability`
- `portfolio_corr`
- `portfolio_div`
- `ctx_trust`
- `calibration_shrink`
- `trust_score`
- `normalized_weight`
- `status`
- `reasons[]`

Ensemble runtime state should include at least:
- `symbol`
- `generated_at`
- `mode`
- `top_regime`
- `session_label`
- `trade_posture`
- `ensemble_quality`
- `abstain_bias`
- `agreement_score`
- `dominant_plugin_share`
- `participating_count`
- `downweighted_count`
- `suppressed_count`
- `buy_support`
- `sell_support`
- `skip_support`
- `final_score`
- `weights_csv`
- `active_plugins_csv`
- `downweighted_plugins_csv`
- `suppressed_plugins_csv`
- `reasons_csv`

### C. Runtime integration points
1. `runtime_feature_pipeline_block.mqh`
   - read current Rates Engine state alongside existing NewsPulse and Microstructure context so the model stage can use it.
2. `runtime_model_stage_block.mqh`
   - collect plugin outputs into dynamic-ensemble candidate records.
   - call the new dynamic-ensemble evaluator.
   - use normalized dynamic weights instead of the current static routed weights when accumulating ensemble support.
3. `runtime_policy_stage_block.mqh`
   - use dynamic-ensemble quality and abstain bias to tighten final decision logic.
   - write runtime artifacts after final posture is known.
4. `FXAI.mq5`
   - add feature flags and thresholds for Dynamic Ensemble.
5. fallback rule
   - if the dynamic-ensemble stage is disabled, fails, or yields no valid active plugins, the runtime must fall back to the current routed ensemble behavior deterministically.

## Trust Scoring Model
Use bounded rule-based trust components.

### Trust inputs
For each plugin:
- base routed meta weight
- pair/session/regime priors from Adaptive Router state and config
- recent empirical quality from existing FXAI meta memory
- calibration shrink based on confidence vs reliability gap and context regret
- disagreement penalty relative to current ensemble center
- cost stress penalty from spread/min-move conditions
- NewsPulse compatibility penalty
- Rates Engine compatibility penalty
- Microstructure compatibility penalty
- stale/health penalty

### Trust formula
Implementation target:

`trust = base_meta_weight * prior_mult * empirical_mult * calibration_mult * stability_mult * context_mult * risk_mult`

Where:
- `prior_mult` comes from config and adaptive-router status
- `empirical_mult` comes from reliability, context edge, context regret, portfolio objective
- `calibration_mult` shrinks raw confidence when confidence materially exceeds reliability or regret is elevated
- `stability_mult` penalizes flip-prone / disagreement-outlier plugins
- `context_mult` rewards pair/session/regime fit
- `risk_mult` penalizes incompatibility with news/rates/microstructure stress

Then:
- clamp trust to configured min/max
- suppress under hard threshold
- downweight under soft threshold
- normalize surviving weights to sum to 1
- cap dominant plugin share

## Ensemble Quality Model
The phase-1 ensemble-quality score should combine:
- trust-weighted agreement
- average participating trust
- current context-fit strength
- execution safety under news/rates/microstructure/cost stress
- concentration penalty if one plugin dominates
- uncertainty penalty when disagreement is high or active set is too thin

Use quality to drive:
- normal mode
- caution mode
- abstain bias
- full SKIP fallback when quality is too weak

## Offline Lab Scope
Add:
- `Tools/offline_lab/dynamic_ensemble_contracts.py`
- `Tools/offline_lab/dynamic_ensemble_config.py`
- `Tools/offline_lab/dynamic_ensemble_replay.py`

Add CLI commands:
- `dynamic-ensemble-validate`
- `dynamic-ensemble-replay-report`

Add local operator docs/config:
- `Tools/OfflineLab/DynamicEnsemble/README.md`
- `Tools/OfflineLab/DynamicEnsemble/IMPLEMENTATION_NOTE.md`
- `Tools/OfflineLab/DynamicEnsemble/dynamic_ensemble_config.json`

Replay report should summarize:
- quality/posture transitions
- participation counts
- dominant plugin frequency
- top suppression reasons
- top active plugins by cumulative weight share
- symbol-level action distribution

## GUI Scope
Add:
- `DynamicEnsembleModels.swift`
- `DynamicEnsembleArtifactReader.swift`
- `DynamicEnsembleView.swift`

Integrate into:
- `SidebarDestination.swift`
- `FXAIGUIModel.swift`
- `FXAIRootView.swift`
- saved workspace selection state
- validation fixtures

GUI should show:
- current ensemble quality
- posture / abstain bias
- buy/sell/skip support
- active/downweighted/suppressed plugins
- per-plugin weights and reasons
- top runtime reasons
- replay summary metrics

## Tests

### Python
- config validation
- trust-threshold sanity
- replay-report summarization on synthetic runtime history

### Swift
- artifact reader parsing for runtime TSV and replay report JSON
- missing-artifact fallback behavior

### MQL/runtime safety by compile + verify-all
- clean compile through `verify-all`
- no regression to single-model mode or current fallback behavior

## Docs and Wiki
Update:
- repo `README.md`
- `FXAI/README.md`
- `FXAI/GUI/README.md`
- `FXAI/Tools/OfflineLab/README.md`
- wiki page `Dynamic-Ensemble.md`
- `Home.md`, `GUI.md`, `Offline-Lab.md`, `Project-Structure.md`, `FXAI-Framework.md`, `_Sidebar.md`

## Step-by-Step Execution Plan
1. Add the implementation-plan and implementation-note docs directory.
2. Add Python contracts/config/replay helpers and CLI hooks.
3. Add the runtime dynamic-ensemble stage file with structs, scoring, normalization, artifact writing, and history.
4. Add runtime feature-pipeline support for current Rates Engine context in the model stage path.
5. Patch `runtime_model_stage_block.mqh` to collect plugin outputs, run the dynamic ensemble stage, and aggregate with normalized dynamic weights.
6. Patch `runtime_policy_stage_block.mqh` to apply quality-based abstain bias and final artifact writing.
7. Add `FXAI.mq5` inputs for enable/disable, thresholds, and dominant-weight caps.
8. Add GUI models/reader/view and wire them into the app shell.
9. Add tests and fixtures.
10. Update docs and wiki.
11. Review new code for logic correctness.
12. Deep bug check API usage, data flow, fallback behavior, and integration.
13. Run Python tests, Swift tests/build, and MT5 `verify-all`.
14. Sync mirror if needed, commit, and push.

## Plan Review Checklist
- Does the plan reuse the existing Adaptive Router rather than duplicating it? Yes.
- Does the plan sit after plugin inference instead of replacing pre-inference routing? Yes.
- Does it remain phase-1 safe without forcing retraining? Yes.
- Does it provide offline config/replay plus GUI visibility? Yes.
- Does it preserve a deterministic fallback path? Yes.
- Does it integrate with existing meta memory instead of inventing a second online scoring universe? Yes.
- Does it avoid inventing a new daemon or service where the runtime already has the necessary inputs? Yes.
