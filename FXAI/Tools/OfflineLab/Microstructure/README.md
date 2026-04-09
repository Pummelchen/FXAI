# Market Microstructure and Order-Flow Proxy Layer

The Market Microstructure and Order-Flow Proxy Layer is a shared FXAI subsystem for **short-horizon execution and flow state**.

It measures what the broker-visible market is doing **right now** using MT5 tick and quote proxies:

- tick imbalance
- directional efficiency
- spread stress and instability
- quote activity bursts
- realized volatility bursts
- session handoff behavior
- stop-run / sweep proxies
- hostile execution conditions

It is **not** true institutional order-book data.
It is an auditable MT5-side proxy layer built from observable broker ticks and quotes.

## What It Produces

Shared runtime artifacts in `FILE_COMMON/FXAI/Runtime`:

- `microstructure_snapshot.json`
- `microstructure_snapshot_flat.tsv`
- `microstructure_status.json`
- `microstructure_history.ndjson`
- `microstructure_symbol_map.tsv`
- `microstructure_service_config.tsv`

Offline Lab artifacts under `Tools/OfflineLab/Microstructure`:

- `microstructure_config.json`
- `microstructure_status.json`
- `Reports/microstructure_replay_report.json`

## Runtime Use

Phase 1 uses microstructure in a gating-first way:

- caution or block entries under hostile execution conditions
- raise SKIP / abstention bias under thin, wide, or stop-run-like regimes
- expose explainable reasons to the runtime and GUI
- enrich the adaptive router with live liquidity and sweep context

It does **not** force immediate retraining or feature-schema changes.

## Setup

1. Validate and write the service config:

```bash
python3 Tools/fxai_offline_lab.py microstructure-validate
```

2. Install the MT5 service:

```bash
python3 Tools/fxai_offline_lab.py microstructure-install-service
```

3. Start `FXAI_MicrostructureProbe` from MT5 `Services`.

4. Optional replay summary:

```bash
python3 Tools/fxai_offline_lab.py microstructure-replay-report --symbol EURUSD --hours-back 24
```

5. Optional health/status check:

```bash
python3 Tools/fxai_offline_lab.py microstructure-health
```

## Key Outputs

Per tradable FX symbol, the layer exposes:

- tick imbalance over `10s`, `30s`, `60s`, `5m`, `15m`
- spread mean/std/z-score
- tick-rate bursts
- realized-volatility bursts
- sweep / rejection proxies
- `liquidity_stress_score`
- `hostile_execution_score`
- `microstructure_regime`
- `trade_gate`
- explainable `reasons[]`

## What Phase 1 Does Not Claim

- no centralized spot-FX tape
- no true dealer inventory
- no true interdealer signed flow
- no reconstructed full order book

All directional and stress measures are explicitly **proxies** built from MT5-observable quote/tick behavior.
