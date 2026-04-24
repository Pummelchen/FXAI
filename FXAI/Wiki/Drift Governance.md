# Drift Governance

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

Drift Governance watches whether features, regimes, calibration, and plugin behavior are decaying after promotion.

## Benefits By Role

- Live Trader: know when a previously trusted profile is degraded.
- Demo Trader: see how structural change can make old behavior unreliable.
- Backtester: compare candidate behavior against drift windows.
- EA Researcher: demote weak champions and promote challengers only after evidence.
- System Architect: keep governance actions conservative, logged, and reproducible.

## What It Does

- monitors feature, regime, calibration, and pair-specific drift
- marks plugins as healthy, caution, degraded, restricted, or review-required
- writes governance reports and history
- keeps challenger promotion behind support-aware validation

## How To Use It

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py drift-governance-validate
python3 Tools/fxai_offline_lab.py drift-governance-run --profile continuous
python3 Tools/fxai_offline_lab.py drift-governance-report --profile continuous
```

## Example Case Scenarios

### Scenario: A once-good plugin starts failing after a regime shift

What to do:
1. Check drift report state and reason codes.
2. Let routing or ensemble layers downweight the plugin.
3. Promote a challenger only after walk-forward validation.

### Scenario: A live trader sees lower trade frequency

What to do:
1. Check whether drift governance restricted degraded contributors.
2. Compare with recent audit and calibration reports.
3. Treat lower frequency as protective if quality evidence degraded.

## Next Pages

- [Dynamic Ensemble](Dynamic%20Ensemble.md)
- [Promotion Criteria](Promotion%20Criteria.md)
- [Offline Lab](Offline%20Lab.md)
