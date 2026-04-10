## Cross-Asset Macro/Liquidity State Engine: Repo Mapping

### Relevant Existing Paths

- `Tools/offline_lab/rates_engine*.py`
  Existing rates / term-structure subsystem. This is reused as the authoritative FXAI-native rates context instead of rebuilding front-end or curve logic.
- `Tools/offline_lab/microstructure_*.py` and `Services/FXAI_MicrostructureProbe.mq5`
  Existing shared-subsystem pattern for MT5 service export, FILE_COMMON artifacts, local status mirroring, replay, and service installation.
- `Engine/Runtime/Trade/runtime_trade_rates_engine.mqh`
  Existing shared flat-snapshot reader pattern for pair-level runtime state.
- `Engine/Runtime/runtime_adaptive_router_stage.mqh`
  Existing regime/router layer that already computes `macro_pressure` and `liquidity_stress`. Cross-asset state is integrated here as a higher-quality shared macro/liquidity input.
- `Engine/Runtime/runtime_dynamic_ensemble_stage.mqh`
  Existing post-inference meta-layer. Cross-asset state is added as additional context for trust penalties and posture escalation.
- `Engine/Runtime/runtime_prob_calibration_stage.mqh`
  Existing calibrated final-decision layer. Cross-asset state is added as uncertainty/risk context, not as fresh alpha logic.
- `Engine/Runtime/runtime_execution_quality_stage.mqh`
  Existing execution-condition forecaster. Cross-asset stress is added as a global execution-context modifier.
- `GUI/Sources/FXAIGUICore/Services/*ArtifactReader.swift`
  Existing operator-artifact ingestion pattern.
- `GUI/Sources/FXAIGUIApp/Features/*`
  Existing GUI surface pattern for subsystem-specific operator views.
- `Tools/offline_lab/market_universe.py`
  Existing FX-only tradable universe plus indicator-only context symbol inventory. This is reused to seed the cross-asset proxy universe.
- `Engine/Lifecycle/lifecycle_context_symbols.mqh`
  Existing broader MT5 context-symbol conventions. The new subsystem borrows those proxy expectations for DXY, yields, volatility, and equity/commodity proxies.

### Implementation Decisions

1. The subsystem is implemented as a hybrid:
   - MT5 service:
     `Services/FXAI_CrossAssetProbe.mq5`
     exports live context-symbol proxy state into `FILE_COMMON/FXAI/Runtime`.
   - Python engine:
     `Tools/offline_lab/cross_asset_*.py`
     consumes the probe snapshot plus the existing rates-engine snapshot, computes normalized feature blocks and pair-level state, and writes canonical cross-asset artifacts.

2. The rates engine remains the source of truth for:
   - front-end rate divergence
   - curve state
   - policy repricing
   - policy uncertainty

3. The new subsystem focuses on:
   - equity risk proxies
   - commodity shock proxies
   - volatility stress proxies
   - USD liquidity/dollar-pressure proxies
   - cross-asset dislocation
   - aggregated macro/liquidity state labels
   - pair-level cross-asset risk/gating state

4. Runtime integration is phase-1 safe:
   - no canonical model-input rewrite
   - no plugin API break
   - shared pair-level context is read through a dedicated runtime adapter
   - adaptive router, dynamic ensemble, probabilistic calibration, execution-quality, and trade-risk layers consume the new pair state as additional context

5. GUI integration follows the existing subsystem pattern:
   - new artifact reader
   - new models
   - new sidebar destination
   - dedicated operator surface for source health, proxy selection, feature block, macro/liquidity scores, pair-level gates, and recent transitions

### Repo-Specific Deviations From The Prompt

- Instead of depending on external paid feeds or a separate non-MT5 market-data connector, phase 1 uses:
  - existing rates-engine outputs
  - MT5-accessible indicator/context symbols
  - explicit fallback proxy mappings in config

- Instead of attempting a separate macro narrative layer, phase 1 produces:
  - deterministic normalized features
  - scorecard-style state scores
  - compact labels and reasons
  - replayable runtime artifacts

- Instead of duplicating existing rates logic, rates features are read from `rates_snapshot.json` and merged into the cross-asset state engine as an upstream dependency.

### Files Added / Changed

- Added:
  - `Tools/offline_lab/cross_asset_contracts.py`
  - `Tools/offline_lab/cross_asset_config.py`
  - `Tools/offline_lab/cross_asset_math.py`
  - `Tools/offline_lab/cross_asset_engine.py`
  - `Tools/offline_lab/cross_asset_service.py`
  - `Tools/offline_lab/cross_asset_replay.py`
  - `Services/FXAI_CrossAssetProbe.mq5`
  - `Engine/Runtime/Trade/runtime_trade_cross_asset_state.mqh`
  - `GUI/Sources/FXAIGUICore/Models/CrossAssetModels.swift`
  - `GUI/Sources/FXAIGUICore/Services/CrossAssetArtifactReader.swift`
  - `GUI/Sources/FXAIGUIApp/Features/CrossAsset/CrossAssetView.swift`
  - `Tools/OfflineLab/CrossAsset/README.md`
  - `Tools/tests/test_cross_asset.py`

- Updated:
  - CLI parser / command wiring
  - test fixture path patching
  - runtime include and stage wiring
  - adaptive router / dynamic ensemble / probabilistic calibration / execution-quality / trade-risk integrations
  - GUI model, sidebar, root view, saved-workspace persistence, and validation fixtures
  - README
