# Microstructure

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

Microstructure provides MT5-visible short-horizon execution and liquidity proxies. It is not true centralized FX order flow.

## Benefits By Role

- Live Trader: avoid good-looking signals during hostile spread, slippage, or liquidity conditions.
- Demo Trader: see how session handoff and spread instability affect trade eligibility.
- Backtester: understand why execution context must be considered separately from direction.
- EA Researcher: study proxy-microstructure effects without violating the core data policy.
- System Architect: keep stale or missing probe data fail-safe.

## What It Measures

- tick-pressure proxy from broker-visible changes
- spread instability
- quote or tick intensity proxy
- realized volatility burst
- sweep or reject flags
- directional efficiency
- liquidity stress
- hostile execution score

These are broker-visible proxies. They must not be marketed as institutional order book, signed interdealer flow, or centralized FX tape.

## How To Use It

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py microstructure-validate
python3 Tools/fxai_offline_lab.py microstructure-health
python3 Tools/fxai_offline_lab.py microstructure-replay-report --symbol EURUSD --hours-back 72
```

## Example Case Scenarios

### Scenario: A trade is blocked during spread widening

What to do:
1. Check Microstructure hostile-execution score.
2. Check Execution Quality expected spread and slippage.
3. Treat the block as protective if both layers agree.

### Scenario: The microstructure snapshot is stale

What to do:
1. Restart or repair the MT5 probe.
2. Keep block-on-stale protection active.
3. Confirm health before interpreting the layer again.

## Next Pages

- [Execution Quality](Execution%20Quality.md)
- [Runtime Control Plane](Runtime%20Control%20Plane.md)
- [Data Policy](Data%20Policy.md)
