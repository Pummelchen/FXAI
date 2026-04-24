# Data Policy

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

This page explains what data FXAI trusts and where data is allowed to enter the framework.

## Benefits By Role

- Live Trader: know whether a live decision is built from approved, fresh inputs.
- Demo Trader: learn why missing or stale context should fail closed instead of inventing confidence.
- Backtester: compare runs that use the same canonical market contract.
- EA Researcher: build features and labels without lookahead or hidden side paths.
- System Architect: audit the pipeline boundary and stop machine-specific shortcuts from creeping in.

## Canonical Contract

FXAI's core market-training contract is:

- `M1 OHLC + spread`
- past-only transformations
- no future labels in feature or normalization windows
- no direct plugin access to MT5 market APIs

Tick, DOM, and centralized order-book data are not core model-training inputs. MT5-visible microstructure proxies are allowed as runtime context and gates, but they must not be described as true interdealer order flow or institutional book depth.

## Pipeline Boundary

- DataCore is the only place that should request raw market bundles.
- FeatureCore is the only place that should build raw feature vectors.
- NormalizationCore is the only place that should normalize features or shape final model payloads.
- Plugins consume prepared context and payloads; they must not call MT5 market data functions directly.

## How To Check The Data Pipeline

```bash
cd /path/to/FXAI
python3 Tools/fxai_testlab.py verify-all
python3 -m pytest -q Tools/tests/test_data_pipeline_gateway.py Tools/tests/test_pipeline_cores.py Tools/tests/test_normalization_pipeline.py
```

## Example Case Scenarios

### Scenario: A plugin needs one more feature

What to do:
1. Add the feature through FeatureCore.
2. Normalize it through NormalizationCore.
3. Pass it through the plugin context or payload contract.
4. Add a leakage-safe test.

Do not pull MT5 bars from the plugin.

### Scenario: A runtime service is stale

What to do:
1. Treat the context as degraded or unknown.
2. Let the runtime caution or block if configured.
3. Repair the service path instead of relaxing safety defaults.

## Next Pages

- [FXAI Framework](FXAI%20Framework.md)
- [Runtime Control Plane](Runtime%20Control%20Plane.md)
- [Microstructure](Microstructure.md)
- [Project Structure](Project%20Structure.md)
