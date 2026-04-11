# Pair-Network / Factor Graph + Portfolio Conflict Resolver

The Pair Network subsystem is FXAI's shared portfolio-coordination layer.

It does not create alpha on its own. It takes already-formed candidate trades, decomposes them into currency and factor exposures, scores overlap and contradiction across the active book, and then decides whether the new trade should be allowed, reduced, suppressed, blocked, or replaced by a better expression.

## What it produces

Runtime artifacts under `FILE_COMMON/FXAI/Runtime`:

- `pair_network_config.tsv`
- `pair_network_status.tsv`
- `fxai_pair_network_<SYMBOL>.tsv`
- `fxai_pair_network_history_<SYMBOL>.ndjson`

Offline Lab artifacts under `Tools/OfflineLab/PairNetwork`:

- `pair_network_config.json`
- `pair_network_status.json`
- `pair_network_history.ndjson`
- `Reports/pair_network_report.json`

## Phase-1 design

Phase 1 is intentionally structural and auditable:

- decomposes every pair into signed base and quote currency exposure
- projects that exposure into a small factor map:
  - `usd_bloc`
  - `eur_rates`
  - `safe_haven`
  - `commodity_fx`
  - `risk_on`
  - `liquidity_stress`
  - `macro_shock`
- augments the structural graph with empirical co-movement only when sufficient dataset support exists
- resolves conflicts using deterministic score thresholds instead of opaque optimization
- degrades safely into structural-only mode when empirical dependency support is weak

It does not attempt a full portfolio optimizer or opaque graph-learning layer in phase 1.

## Runtime behavior

The runtime resolver runs after candidate quality is already known from the existing stack:

1. Dynamic Ensemble produces pair-level candidates.
2. Probabilistic Calibration and Execution Quality price edge, uncertainty, and execution conditions.
3. Pair Network decomposes current positions, pending orders, and same-cycle peer candidates into shared exposure.
4. The resolver applies one of:
   - `ALLOW`
   - `ALLOW_REDUCED`
   - `SUPPRESS_REDUNDANT`
   - `BLOCK_CONTRADICTORY`
   - `BLOCK_CONCENTRATION`
   - `PREFER_ALTERNATIVE_EXPRESSION`

The live trade-risk layer then enforces the result before order send.

## Commands

Validate config and write runtime config/status mirrors:

```bash
python3 Tools/fxai_offline_lab.py pair-network-validate
```

Build the dependency graph and status report:

```bash
python3 Tools/fxai_offline_lab.py pair-network-build --profile continuous
```

Reprint the latest report:

```bash
python3 Tools/fxai_offline_lab.py pair-network-report --profile continuous
```

## Operator surfaces

Phase 1 exposes the subsystem in:

- MT5 runtime TSV + NDJSON artifacts per symbol
- Offline Lab report and append-only graph history
- the macOS GUI Pair Network surface for:
  - symbol-level conflict decisions
  - currency and factor exposure summaries
  - top dependency edges
  - preferred-expression selections
  - graph freshness and fallback flags
