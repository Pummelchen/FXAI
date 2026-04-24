# Probabilistic Calibration

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

Probabilistic Calibration converts model and ensemble output into calibrated probability, expected move, and edge-after-cost decisions.

## Benefits By Role

- Live Trader: avoid trades where direction exists but expected edge does not clear costs and uncertainty.
- Demo Trader: learn why no-trade can be the correct answer in noisy regimes.
- Backtester: compare candidates by calibrated, post-cost decision quality.
- EA Researcher: improve thresholds and calibration tiers rather than chasing raw direction only.
- System Architect: ensure abstention policy is tied to generated artifacts and release gates.

## What It Does

- maps raw ensemble output into calibrated probability
- estimates move size and uncertainty where available
- subtracts spread, slippage, and safety buffers
- returns explicit abstention reasons when edge is not strong enough

## How To Use It

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py prob-calibration-validate
python3 Tools/fxai_offline_lab.py prob-calibration-replay-report --symbol EURUSD --hours-back 72
```

## Example Case Scenarios

### Scenario: Direction is correct often, but live profit is poor

What to do:
1. Check expected edge after spread and slippage.
2. Check calibration tier support.
3. Raise abstention discipline before adding risk.

### Scenario: Random-walk windows still produce trades

What to do:
1. Inspect active ratio and abstention reasons.
2. Tighten calibration or cost floors if evidence supports it.
3. Re-run Audit Lab random-walk scenarios.

## Next Pages

- [Execution Quality](Execution%20Quality.md)
- [Audit Lab](Audit%20Lab.md)
- [Promotion Criteria](Promotion%20Criteria.md)
