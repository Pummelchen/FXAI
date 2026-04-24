# Cross Asset

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

Cross Asset adds shared macro and liquidity context from configured indicator-only MT5 symbols.

## Benefits By Role

- Live Trader: detect when an FX signal conflicts with broader risk, commodity, volatility, or dollar-liquidity state.
- Demo Trader: learn how global context changes pair behavior.
- Backtester: compare local pair signals against global shock windows.
- EA Researcher: add context to scenario analysis without changing the canonical FX training contract.
- System Architect: keep non-FX symbols as indicator-only context, not tradable FXAI instruments.

## What It Does

- reads configured indicator-only symbols through approved runtime artifacts
- summarizes risk-on/risk-off, commodity, volatility, and liquidity pressure
- maps shared context back to affected FX pairs
- applies caution or block posture when context is hostile or stale

## How To Use It

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py cross-asset-validate
python3 Tools/fxai_offline_lab.py cross-asset-health
python3 Tools/fxai_offline_lab.py cross-asset-replay-report --symbol AUDUSD --hours-back 72
```

## Example Case Scenarios

### Scenario: AUDUSD is bullish locally but commodities and risk are deteriorating

What to do:
1. Check Cross Asset pair impact.
2. Check whether the runtime is cautioning due to broad risk state.
3. Use Audit Lab to compare similar global-context windows before relaxing controls.

### Scenario: Probe inputs are missing

What to do:
1. Treat the cross-asset layer as stale.
2. Restore the probe or config.
3. Do not claim the system has global context until health is clean.

## Next Pages

- [Runtime Control Plane](Runtime%20Control%20Plane.md)
- [Rates Engine](Rates%20Engine.md)
- [Data Policy](Data%20Policy.md)
