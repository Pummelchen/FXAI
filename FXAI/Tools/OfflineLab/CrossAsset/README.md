# Cross-Asset Macro/Liquidity State Engine

The Cross-Asset Macro/Liquidity State Engine is FXAI's shared cross-market context layer.

It does not trade by itself. It converts rates, equity-risk, commodity, volatility, and dollar-liquidity context into a compact shared state that runtime, routing, calibration, execution-quality logic, and the macOS GUI can consume consistently.

## What it produces

Shared runtime artifacts under `FILE_COMMON/FXAI/Runtime`:

- `cross_asset_snapshot.json`
- `cross_asset_snapshot_flat.tsv`
- `cross_asset_status.json`
- `cross_asset_history.ndjson`
- `cross_asset_symbol_map.tsv`
- `cross_asset_probe_config.tsv`
- `cross_asset_probe_snapshot.json`
- `cross_asset_probe_status.json`
- `cross_asset_probe_history.ndjson`

Offline Lab artifacts under `Tools/OfflineLab/CrossAsset`:

- `cross_asset_config.json`
- `cross_asset_status.json`
- `State/cross_asset_state.json`
- `Reports/cross_asset_replay_report.json`

## Phase-1 design

Phase 1 is intentionally practical and auditable:

- reuses the Rates Engine instead of rebuilding rates logic
- uses an MT5 service to probe indicator-only MT5 symbols already configured in the market universe
- normalizes cross-asset raw moves into z-style features and deterministic state scores
- maps global state into pair-level `ALLOW | CAUTION | BLOCK` posture without changing canonical model inputs
- degrades safely when rates or probe data are stale

It does not claim to be a full global-macro research platform.

## What it scores

The engine exposes:

- normalized feature block:
  - front-end rate divergence
  - equity risk state
  - oil / gold / metals shock
  - volatility stress
  - USD liquidity stress
  - cross-asset dislocation
  - global macro stress
- higher-level state scores:
  - `rates_repricing_score`
  - `risk_off_score`
  - `commodity_shock_score`
  - `volatility_shock_score`
  - `usd_liquidity_stress_score`
  - `cross_asset_dislocation_score`
- interpretable labels:
  - `macro_state`
  - `risk_state`
  - `liquidity_state`
- pair-level posture:
  - `pair_cross_asset_risk_score`
  - `pair_sensitivity`
  - `trade_gate`
  - explainable `reasons[]`

## Setup

1. Validate and write the default config plus MT5 probe config:

```bash
python3 Tools/fxai_offline_lab.py cross-asset-validate
```

2. Install the MT5 probe service:

```bash
python3 Tools/fxai_offline_lab.py cross-asset-install-service
```

3. Start `FXAI_CrossAssetProbe` from MT5 `Services`.

4. Build one shared snapshot:

```bash
python3 Tools/fxai_offline_lab.py cross-asset-once
```

5. Or keep it refreshed continuously:

```bash
python3 Tools/fxai_offline_lab.py cross-asset-daemon --interval-seconds 120
```

6. Inspect health or replay:

```bash
python3 Tools/fxai_offline_lab.py cross-asset-health
python3 Tools/fxai_offline_lab.py cross-asset-replay-report --symbol EURUSD --hours-back 72
```

## Runtime use

Phase 1 uses the shared cross-asset state in a gating-first way:

- Adaptive Router absorbs macro/liquidity stress into regime reasons
- Dynamic Ensemble downweights participation when cross-asset stress is stale or blocking
- Probabilistic Calibration adds cross-asset uncertainty and risk penalties
- Execution Quality adds cross-asset stress into spread/slippage/latency scoring
- Trade Risk can block or caution entries using the pair-level cross-asset gate
- the macOS GUI exposes a dedicated Cross Asset surface for operator visibility
