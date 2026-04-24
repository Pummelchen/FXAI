# Adaptive Router

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

Adaptive Router classifies the current regime and routes trust toward plugin families that are better suited to that state.

## Benefits By Role

- Live Trader: see whether the active regime makes the system cautious, selective, or permissive.
- Demo Trader: learn why different plugins matter in trend, range, shock, or liquidity-stress regimes.
- Backtester: evaluate candidates by scenario and regime instead of one aggregate score.
- EA Researcher: improve routing priors and suppression logic without rewriting the whole model zoo.
- System Architect: validate router artifacts before promotion.

## What It Does

- classifies trend, range, policy-divergence, risk, event-shock, and liquidity-stress states
- scores plugin suitability by regime, pair, session, and recent quality
- suppresses plugins that are unsafe for the current context
- writes replayable router history for audit and research

## How To Use It

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py adaptive-router-validate
python3 Tools/fxai_offline_lab.py adaptive-router-profiles --profile continuous
python3 Tools/fxai_offline_lab.py adaptive-router-replay-report --symbol EURUSD --hours-back 72
```

## Example Case Scenarios

### Scenario: Trend models are suppressed in a range

What to do:
1. Check router regime confidence.
2. Verify whether range or mean-reversion families received more weight.
3. Compare the behavior in Audit Lab before changing router priors.

### Scenario: Router confidence is weak

What to do:
1. Expect caution or abstention to increase.
2. Check whether upstream context is stale or contradictory.
3. Do not force equal plugin voting just to increase trade count.

## Next Pages

- [Dynamic Ensemble](Dynamic%20Ensemble.md)
- [Runtime Control Plane](Runtime%20Control%20Plane.md)
- [Model Zoo](Model%20Zoo.md)
