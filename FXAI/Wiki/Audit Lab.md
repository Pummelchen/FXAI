# Audit Lab

Audit Lab is where FXAI proves whether a candidate or runtime assumption survives normal, hostile, and event-driven conditions.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Why This Page Matters

- Live Trader: Audit Lab tells you what kinds of conditions the current logic was designed to survive.
- Demo Trader: it gives you a benchmark for interpreting live-demo behavior.
- Backtester: it is your main disciplined evaluation surface.
- EA Researcher: it prevents overpromoting candidates that only look good in one window.
- System Architect: it helps separate platform failures from genuine model weakness.

## What Audit Lab Does

- Runs scenario-based evaluations instead of only one aggregate backtest.
- Exercises hostile windows, macro-event packs, and walk-forward behavior.
- Produces reports that explain where a candidate succeeds and where it fails.
- Helps verify that runtime layers are integrated logically, not only syntactically.

## How To Run It

Compile and run the standard release gate:

```bash
cd /Users/andreborchert/FXAI-main2/FXAI
python3 Tools/fxai_testlab.py verify-all
```

If you want focused MT5-side auditing, use the Audit Runner path documented in the project tree and GUI builder.

## What To Look For

- Trade count under stress, not only net return.
- Drawdown and failure clusters during event windows.
- Whether abstention and runtime filters reduce damage in hostile windows.
- Whether improvements are broad or only concentrated in one symbol or regime.

## Example Case Scenarios

### Scenario: A challenger wins on quiet data but loses during event windows

Interpretation:
- the candidate may be overfit to stable conditions and unsafe for live promotion.

What to do:
1. Compare event-pack results with the champion.
2. Inspect NewsPulse, Rates Engine, and Execution Quality interactions.
3. Reject the promotion unless the failure mode is understood and fixed.

### Scenario: A trader says the live system "missed too many trades"

Interpretation:
- missing trades may be correct if the hostile-window or post-cost audit supports abstention.

What to do:
1. Compare the live date range with recent audit scenarios.
2. Check whether runtime layers were protecting against the exact failure mode that the user wants to override.
3. If the audit agrees with the runtime, keep the control.

### Scenario: A backtester wants faster confidence on a symbol-specific change

What to do:
1. Narrow the symbol and date range.
2. Use the builder or terminal recipe to run a focused evaluation.
3. Compare against the last promoted profile instead of an arbitrary older baseline.

## Good Operator Habits

- Treat Audit Lab as a reality check, not a paperwork step.
- When a candidate improves one metric, ask what got worse.
- Use audit evidence before changing runtime protection logic.

## Next Pages

- [Offline Lab](Offline%20Lab.md) for promotion and research workflows.
- [Runtime Control Plane](Runtime%20Control%20Plane.md) for understanding the live layers that Audit Lab is trying to validate.
