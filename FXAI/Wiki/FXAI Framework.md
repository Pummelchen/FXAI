# FXAI Framework

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

FXAI is a framework for turning market data, model outputs, runtime context, and audit evidence into a governed MT5 trading posture.

## Benefits By Role

- Live Trader: use FXAI to understand whether the current market is tradable, not only whether a model is directional.
- Demo Trader: use FXAI to learn how abstention, caution, and block decisions behave before going live.
- Backtester: use FXAI to compare candidates under shared data, cost, scenario, and release-gate assumptions.
- EA Researcher: use FXAI to improve the full stack: labels, features, normalization, models, routing, calibration, and promotion.
- System Architect: use FXAI to keep services, artifacts, releases, and recovery workflows auditable.

## Framework Layers

1. DataCore pulls the approved MT5 market data contract.
2. FeatureCore builds past-only feature views.
3. NormalizationCore fits and applies approved scaling without leakage.
4. Plugins produce probability, confidence, and abstention-compatible outputs.
5. Runtime layers adjust the raw model view for news, rates, cross-asset context, microstructure, routing, calibration, execution, drift, and portfolio conflict.
6. Audit Lab and Offline Lab validate, promote, and publish evidence-backed artifacts.
7. GUI and terminal workflows expose the same source of truth.

## How To Use The Framework

```bash
cd /path/to/FXAI
python3 Tools/fxai_testlab.py doctor
python3 Tools/fxai_testlab.py verify-all
python3 Tools/fxai_offline_lab.py live-state --symbol EURUSD
```

Use the GUI when you want a role-based surface. Use the terminal when you need repeatable command history.

## Example Case Scenarios

### Scenario: A raw model is bullish but FXAI blocks the trade

Interpretation:
- the framework is doing its job if costs, news, liquidity, drift, or portfolio conflict make the trade unsafe.

What to do:
1. Inspect Runtime Control Plane reasons.
2. Check subsystem freshness.
3. Do not bypass the block unless the underlying artifact is confirmed wrong and repaired.

### Scenario: A researcher wants better live behavior

What to do:
1. Find the weak layer in audit reports.
2. Improve that layer directly instead of adding another model by default.
3. Regenerate artifacts and rerun release gates before promotion.

## Next Pages

- [Data Policy](Data%20Policy.md)
- [Runtime Control Plane](Runtime%20Control%20Plane.md)
- [Offline Lab](Offline%20Lab.md)
- [GUI](GUI.md)
