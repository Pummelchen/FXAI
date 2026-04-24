# Pair Network

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

Pair Network resolves redundant, contradictory, or over-concentrated FX exposures across pairs and currencies.

## Benefits By Role

- Live Trader: avoid stacking several trades that express the same USD, EUR, JPY, or commodity-currency bet.
- Demo Trader: learn why the best single chart signal may be rejected at portfolio level.
- Backtester: evaluate portfolio conflict instead of isolated pair behavior only.
- EA Researcher: test factor and currency exposure logic with repeatable graph artifacts.
- System Architect: keep portfolio conflict decisions deterministic and auditable.

## What It Does

- builds a currency and pair dependency graph
- decomposes candidate trades into currency exposures
- identifies redundant, contradictory, or concentrated bets
- recommends preferred expressions when multiple pairs show the same macro idea

## How To Use It

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py pair-network-validate
python3 Tools/fxai_offline_lab.py pair-network-build --profile continuous
python3 Tools/fxai_offline_lab.py pair-network-report --profile continuous
```

## Example Case Scenarios

### Scenario: EURUSD and GBPUSD both trigger long while USD risk is already high

What to do:
1. Check pair-network redundancy and concentration reasons.
2. Let FXAI choose the cleaner expression or reduce size.
3. Avoid manually stacking correlated exposure unless the portfolio plan explicitly allows it.

### Scenario: A candidate contradicts an open JPY safe-haven position

What to do:
1. Inspect currency exposure decomposition.
2. Check whether the contradiction is intentional hedge or accidental conflict.
3. Prefer conflict resolution over isolated pair conviction.

## Next Pages

- [Runtime Control Plane](Runtime%20Control%20Plane.md)
- [Cross Asset](Cross%20Asset.md)
- [Audit Lab](Audit%20Lab.md)
