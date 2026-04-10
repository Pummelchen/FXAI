# Probabilistic Calibration + Abstention Layer

This subsystem is the final decision-quality layer for FXAI.

It does not replace plugins, the Adaptive Router, or the Dynamic Ensemble. It consumes the routed ensemble output, applies a deterministic calibration tier, estimates tradeable move and edge after costs, and can force `SKIP` when the evidence is not strong enough after spread, slippage, uncertainty, and current risk context.

## Phase 1 Scope

- ensemble-level probability calibration
- hierarchical tier selection with pair, session, regime, and global fallback
- expected-move scaling with uncertainty-aware quantile shrinkage
- edge-after-costs calculation using spread, slippage, uncertainty, and execution-risk penalties
- explicit abstention reason codes
- append-only runtime history for replay and audit
- GUI visibility for calibrated probabilities, move distribution, tier support, and final abstention reasons

## Main Artifacts

- `prob_calibration_config.json`
  Operator-facing config and thresholds.
- `prob_calibration_memory.json`
  Tier memory with support, calibration quality, and move-scale metadata.
- `Reports/prob_calibration_replay_report.json`
  Replay summary built from runtime history.
- `FILE_COMMON/FXAI/Runtime/prob_calibration_config.tsv`
  Runtime-exported config for MT5.
- `FILE_COMMON/FXAI/Runtime/prob_calibration_memory.tsv`
  Runtime-exported tier memory for MT5.
- `FILE_COMMON/FXAI/Runtime/fxai_prob_calibration_<SYMBOL>.tsv`
  Latest live calibrated decision state per symbol.
- `FILE_COMMON/FXAI/Runtime/fxai_prob_calibration_history_<SYMBOL>.ndjson`
  Append-only runtime history used for replay and audit.

## Preferred Commands

```bash
python3 FXAI/Tools/fxai_offline_lab.py prob-calibration-validate
python3 FXAI/Tools/fxai_offline_lab.py prob-calibration-replay-report --symbol EURUSD --hours-back 72
python3 FXAI/Tools/fxai_testlab.py compile-main
cd FXAI/GUI && swift test && swift build
```

## Runtime Position

Phase 1 places the subsystem here:

`plugin outputs -> Dynamic Ensemble -> Probabilistic Calibration -> expected edge -> abstention / SKIP -> final execution`

The layer is intentionally conservative:

- it does not invent directional alpha
- it only suppresses weak decisions
- stale or weak calibration support is treated as uncertainty, not as permission to trade

## Tuning Focus

- raise `trade_edge_floor_points` for stricter live discipline
- raise `support_soft_floor` to require more evidence before a tier is treated as fully trusted
- raise `min_calibration_quality` to penalize weak tiers faster
- raise `risk_penalties.micro_block_mult` when hostile execution should dominate
- raise `uncertainty_penalties.disagreement` when ensemble conflict should produce more SKIPs
- lower `soft_fallback.confidence_cap` if fallback tiers are still too aggressive
