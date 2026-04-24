# Label Engine

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

Label Engine creates reproducible training and evaluation targets for direction, move size, timing, and tradability after costs.

## Benefits By Role

- Live Trader: trust that promoted models were trained against practical tradeability, not only raw direction.
- Demo Trader: understand why some apparent wins are ignored when costs or timing are poor.
- Backtester: compare results against labels that match the intended horizon.
- EA Researcher: improve model quality by improving targets before adding complexity.
- System Architect: keep label definitions versioned and reproducible.

## What It Does

- builds multi-horizon direction labels
- measures move magnitude and time-to-move
- builds tradeability labels after spread and cost assumptions
- produces meta-labels that answer whether a raw signal should be traded

## How To Use It

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py label-engine-validate
python3 Tools/fxai_offline_lab.py label-engine-build --profile continuous --limit-datasets 1
python3 Tools/fxai_offline_lab.py label-engine-report --profile continuous
```

## Example Case Scenarios

### Scenario: A model predicts direction but enters too late

What to do:
1. Inspect time-to-move labels.
2. Check whether the signal clears tradeability after costs.
3. Adjust labels or thresholds before blaming only the architecture.

### Scenario: Random-walk labels show too many active trades

What to do:
1. Review no-trade and meta-label thresholds.
2. Tighten tradeability definitions if the evidence supports it.
3. Rebuild labels and rerun affected training or audit workflows.

## Next Pages

- [Offline Lab](Offline%20Lab.md)
- [Probabilistic Calibration](Probabilistic%20Calibration.md)
- [Audit Lab](Audit%20Lab.md)
