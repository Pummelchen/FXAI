# Dynamic Ensemble

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

Dynamic Ensemble combines plugin outputs with context-aware weights instead of treating every plugin as an equal vote.

## Benefits By Role

- Live Trader: understand which model families actually influenced the final posture.
- Demo Trader: see why similar raw scores can produce different final actions.
- Backtester: evaluate whether weighting improves scenario robustness.
- EA Researcher: tune weighting using calibration, costs, regime, pair, and session evidence.
- System Architect: verify ensemble artifacts and replay history before promotion.

## What It Does

- reads plugin probabilities and confidence
- applies suitability, calibration, cost, and context weighting
- suppresses unstable or degraded contributors
- writes replayable final participation weights

## How To Use It

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py dynamic-ensemble-validate
python3 Tools/fxai_offline_lab.py dynamic-ensemble-replay-report --symbol EURUSD --hours-back 72
```

## Example Case Scenarios

### Scenario: One plugin is strongly bullish but the ensemble is neutral

What to do:
1. Check the plugin's recent calibration and regime fit.
2. Check whether other families disagree with higher reliability.
3. Use the ensemble explanation before assuming the strong plugin is right.

### Scenario: A new plugin improves raw accuracy but receives low weight

What to do:
1. Inspect calibration and post-cost behavior.
2. Check drift governance state.
3. Promote weight only after walk-forward evidence supports it.

## Next Pages

- [Adaptive Router](Adaptive%20Router.md)
- [Probabilistic Calibration](Probabilistic%20Calibration.md)
- [Model Zoo](Model%20Zoo.md)
