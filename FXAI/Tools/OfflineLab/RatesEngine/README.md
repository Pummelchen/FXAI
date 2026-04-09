# Rates / Term-Structure / Policy-Path Engine

The Rates Engine is FXAI's shared macro rates layer.

It is not a trading model. It converts short-end rates, curve shape, policy-path repricing, and event-linked surprise into a compact FXAI-native state that runtime, NewsPulse, the Adaptive Router, and the GUI can consume consistently.

## Phase 1 Scope

Phase 1 is designed to be useful immediately without forcing model retraining.

It provides:
- currency-level rates and policy-path state
- pair-level rates differentials, divergence, risk, and trade gates
- NewsPulse enrichment with rates-aware confirmation and policy relevance
- runtime caution and block overlays
- GUI visibility, health, and replay
- append-only history for audit and later backtest alignment

It does not yet force expansion of the canonical model feature vector.

## Data Model

Phase 1 supports two input modes:

1. `manual_market_input`
- operator-supplied numeric short-end, expected-path, and curve values
- best available path when you already have trusted rates inputs

2. `policy_proxy_index`
- default mode
- derives policy-path and front-end proxies from NewsPulse event timing, official central-bank coverage, scheduled surprise proxies, and policy-topic classification
- robust enough for live gating and macro context, but still a proxy, not full OIS bootstrapping

Curve shape is only true-market in phase 1 when manual numeric curve inputs are supplied.

## Main Files

- `Tools/offline_lab/rates_engine_contracts.py`
- `Tools/offline_lab/rates_engine_config.py`
- `Tools/offline_lab/rates_engine_inputs.py`
- `Tools/offline_lab/rates_engine.py`
- `Tools/offline_lab/rates_engine_daemon.py`
- `Tools/offline_lab/rates_engine_newspulse.py`
- `Tools/offline_lab/rates_engine_replay.py`
- `Engine/Runtime/Trade/runtime_trade_rates_engine.mqh`
- `GUI/Sources/FXAIGUICore/Services/RatesEngineArtifactReader.swift`
- `GUI/Sources/FXAIGUIApp/Features/RatesEngine/RatesEngineView.swift`

## Artifacts

Local operator/config state:
- `Tools/OfflineLab/RatesEngine/rates_engine_config.json`
- `Tools/OfflineLab/RatesEngine/rates_provider_inputs.json`
- `Tools/OfflineLab/RatesEngine/rates_engine_status.json`
- `Tools/OfflineLab/RatesEngine/rates_history.ndjson`
- `Tools/OfflineLab/RatesEngine/Reports/rates_replay_report.json`

Shared runtime artifacts:
- `FILE_COMMON/FXAI/Runtime/rates_snapshot.json`
- `FILE_COMMON/FXAI/Runtime/rates_snapshot_flat.tsv`
- `FILE_COMMON/FXAI/Runtime/rates_symbol_map.tsv`
- `FILE_COMMON/FXAI/Runtime/rates_history.ndjson`

## Main Commands

Validate config and inputs:

```bash
python3 FXAI/Tools/fxai_offline_lab.py rates-engine-validate
```

Run one bounded cycle:

```bash
python3 FXAI/Tools/fxai_offline_lab.py rates-engine-once
```

Run continuously:

```bash
python3 FXAI/Tools/fxai_offline_lab.py rates-engine-daemon --interval-seconds 120
```

Inspect health:

```bash
python3 FXAI/Tools/fxai_offline_lab.py rates-engine-health
```

Build a replay report:

```bash
python3 FXAI/Tools/fxai_offline_lab.py rates-engine-replay-report --symbol EURUSD --hours-back 72
```

## Runtime Integration

Phase 1 uses a gating-first integration.

The runtime adapter:
- reads `rates_snapshot_flat.tsv`
- maps broker symbols back to canonical FX pairs
- applies `ALLOW | CAUTION | BLOCK`
- can block if the rates state is stale or explicitly blocked
- can scale lot size and tighten entry floors during caution regimes

This is controlled from `FXAI.mq5` with:
- `RatesEngineEnabled`
- `RatesEngineBlockOnUnknown`
- `RatesEngineFreshnessMaxSec`
- `RatesEngineCautionLotScale`
- `RatesEngineCautionEnterProbBuffer`

## GUI Integration

The macOS GUI includes a dedicated Rates Engine page showing:
- source health
- currency policy-state heatmap
- pair divergence and trade-gate detail
- recent policy tape
- health and artifact paths

## Limitations

Phase 1 does not do:
- full OIS curve bootstrapping
- institutional fixed-income modeling
- directional trading from rates alone
- forced model-vector expansion across the plugin zoo

Those are future upgrades. Phase 1 focuses on safe, auditable macro context and execution filtering.
