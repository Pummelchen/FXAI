# FXAI Framework

## Executive Summary
FXAI is a native MetaTrader 5 framework for building, testing, and operating cost-aware FX prediction models inside the MT5 and MQL5 ecosystem. Its purpose is practical: produce smarter `BUY / SELL / SKIP` decisions that survive realistic spreads, commissions, and changing intraday market regimes.

## Why FXAI Exists
Most MT5 model projects fail in one of two ways:
- one hard-coded strategy with poor extensibility
- many disconnected experiments that cannot be compared fairly

FXAI standardizes feature generation, normalization, regime handling, warmup, online training, calibration, ensemble logic, and compliance behind one framework.

## Core Capabilities
- Shared M1-centric market dataset with multi-timeframe and cross-symbol context
- Engineered features with configurable normalization and schema-specific projection
- Cost-aware 3-class labels: `SELL`, `BUY`, `SKIP`
- Plugin manifests, prediction contracts, and family-aware feature views
- Online updates, replay, calibration, and reliability handling
- Meta-policy routing, ensemble blending, and execution filtering
- Equity-level protection and realistic execution gating

## Why It Is Useful
- One framework can benchmark many model families under the same rules.
- Centralized data and labels reduce hidden leakage and inconsistent experiments.
- MT5-native execution keeps research, backtesting, and deployment in the same runtime.
- Cost-aware skip logic is part of the prediction target, not an afterthought.
- The project stays pure MQL5 without external DLLs or inference services.

## Plugin Naming Convention
- `ai_*` neural and sequence-oriented models
- `lin_*` online linear and sparse learners
- `tree_*` tree and boosting models
- `dist_*` distributional and probabilistic models
- `mix_*` mixture and uncertainty-routing models
- `wm_*` world-model and structural market models
- `mem_*` retrieval and memory-based models
- `rule_*` deterministic control filters

## Model Families In Scope
### Neural / Sequence
`ai_mlp`, `ai_lstm`, `ai_lstmg`, `ai_s4`, `ai_tcn`, `ai_tft`, `ai_autoformer`, `ai_patchtst`, `ai_tst`, `ai_stmn`, `ai_geodesic`, `ai_trr`, `ai_chronos`, `ai_timesfm`

### Linear / Sparse
`lin_sgd`, `lin_ftrl`, `lin_pa`, `lin_enhash`

### Tree / Boosting
`tree_xgb_fast`, `tree_xgb`, `tree_lgbm`, `tree_catboost`

### Distributional
`dist_quantile`

### Mixture
`mix_loffm`, `mix_moe_conformal`

### World / Structural
`wm_cfx`, `wm_graph`

### Memory / Retrieval
`mem_retrdiff`

### Rule-Based
`rule_m1sync`, `rule_buyonly`, `rule_sellonly`, `rule_random`

## Data Pipeline Summary
FXAI builds features from the current trading symbol and aligned context series using past-only data. The pipeline covers short-horizon returns, candle geometry, trend filters, multi-timeframe structure, volatility and range measures, cross-symbol context, and time-of-day execution context.

Plugin manifests shape how this data is consumed through:
- feature schemas
- feature-group masks
- family-specific projectors

## Operational Model
The MT5 Experts folder is the source of truth for runnable code. GitHub is the synchronized versioned copy and should only be updated after the active MT5 project compiles cleanly.

## Next Reading
- [Project Structure](Project-Structure)
- [Audit Lab](Audit-Lab)
