# Dynamic Ensemble Implementation Note

## Repo mapping

- Runtime aggregation seam: `Engine/Runtime/runtime_model_stage_block.mqh`
- Final policy seam: `Engine/Runtime/runtime_policy_stage_block.mqh`
- Context priors: Adaptive Router, NewsPulse, Rates Engine, Microstructure
- Offline Lab config/report pattern: `Tools/offline_lab/*`
- GUI artifact-reader pattern: `GUI/Sources/FXAIGUICore/Services/*ArtifactReader.swift`

## Phase-1 design

Phase 1 is deterministic and rule-based. It does not learn live weights inside the decision cycle.

It uses:

- current plugin outputs
- existing FXAI meta-memory diagnostics
- current regime/news/rates/microstructure context
- config-driven family compatibility priors

Then it publishes:

- per-plugin trust, status, and normalized weights
- ensemble quality and abstain bias
- final ensemble support snapshot and replay history

## Deferred

- learned meta-models
- contextual bandits / RL
- canonical feature-vector expansion
- portfolio-level cross-symbol ensemble coupling
