# Rates Engine

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

Rates Engine adds policy-path and rates-aware context to FXAI's runtime control plane.

## Benefits By Role

- Live Trader: understand when a pair is unsafe because policy divergence or rates context is stale.
- Demo Trader: observe how central-bank and rate-shock context changes trade posture.
- Backtester: evaluate whether rate-sensitive windows behave differently from ordinary windows.
- EA Researcher: use rates context to improve filtering, labels, and scenario interpretation.
- System Architect: monitor whether rates inputs are configured, fresh, and safely degraded.

## What It Does

- tracks front-end yield or OIS-style inputs when configured
- falls back to clearly labeled policy proxies when true market inputs are missing
- writes pair-level policy divergence and caution state
- enriches NewsPulse and runtime trade filters
- fails safe when inputs are stale or incomplete

## How To Use It

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py rates-engine-validate
python3 Tools/fxai_offline_lab.py rates-engine-health
python3 Tools/fxai_offline_lab.py rates-engine-replay-report --symbol EURUSD --hours-back 72
```

In the GUI, open `Rates Engine` or inspect `Live Overview` reasons when a pair is blocked or cautioned.

## Example Case Scenarios

### Scenario: EURUSD has a strong model score before an ECB/Fed event

What to do:
1. Check NewsPulse event risk.
2. Check Rates Engine policy-path state.
3. If rates context is stale or event-shock posture is active, treat a block as protective, not as a model failure.

### Scenario: A researcher wants to test policy-divergence sensitivity

What to do:
1. Build or replay a rates-aware window.
2. Compare candidate behavior with and without rates caution.
3. Promote only if post-cost and event-window behavior improves.

## Next Pages

- [NewsPulse](NewsPulse.md)
- [Runtime Control Plane](Runtime%20Control%20Plane.md)
- [Audit Lab](Audit%20Lab.md)
