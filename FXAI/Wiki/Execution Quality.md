# Execution Quality

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

Execution Quality forecasts whether a theoretically good signal can be traded under current broker and session conditions.

## Benefits By Role

- Live Trader: avoid sending trades into bad spread, slippage, fill, or liquidity conditions.
- Demo Trader: learn that execution can invalidate an otherwise correct prediction.
- Backtester: compare results with realistic execution assumptions.
- EA Researcher: separate signal quality from order-quality failure.
- System Architect: verify broker replay and execution artifacts stay fresh.

## What It Does

- estimates spread widening
- estimates slippage and fill quality stress
- flags latency sensitivity and liquidity fragility
- contributes caution, block, or reduced-risk posture

## How To Use It

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py execution-quality-validate
python3 Tools/fxai_offline_lab.py execution-quality-replay-report --symbol EURUSD --hours-back 72
```

## Example Case Scenarios

### Scenario: A signal is strong near rollover

What to do:
1. Check expected spread and slippage.
2. Check Microstructure hostile-execution state.
3. Accept abstention if execution quality would likely erase the edge.

### Scenario: Backtest looks better than live

What to do:
1. Review broker replay and execution-quality assumptions.
2. Compare live slippage and fill history.
3. Do not blame the model until execution conditions are separated.

## Next Pages

- [Microstructure](Microstructure.md)
- [Probabilistic Calibration](Probabilistic%20Calibration.md)
- [Runtime Control Plane](Runtime%20Control%20Plane.md)
