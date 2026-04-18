# NewsPulse

NewsPulse is FXAI's shared news-risk layer. It turns macro-event timing and source health into something the runtime, research tools, and operators can all read consistently.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Why This Page Matters

- Live Trader: NewsPulse explains when the market is unsafe because of event timing or stale news inputs.
- Demo Trader: it helps you see why the system behaves differently around scheduled events.
- Backtester: it adds context for event-window comparisons.
- EA Researcher: it is a critical part of any event-sensitive evaluation.
- System Architect: it tells you whether the event data path is healthy.

## What NewsPulse Does

- merges MT5 calendar, official-feed, and GDELT-style event context
- assigns pair-level news posture and watchlist context
- exposes source freshness and source failure clearly
- enriches other layers such as Rates Engine with policy-aware context
- provides an MT5-calendar-cache fallback inside the EA runtime when the flattened pair snapshot is missing or stale

## Core Commands

```bash
cd /Users/andreborchert/FXAI-main2/FXAI
python3 Tools/fxai_offline_lab.py newspulse-health
python3 Tools/fxai_offline_lab.py newspulse-once
python3 Tools/fxai_offline_lab.py newspulse-install-service
```

## How To Use It

1. Install the MT5 calendar service when needed.
2. Refresh NewsPulse or run the daemon.
3. Check source freshness before trusting pair-level posture.
4. Read the pair reasons, not only the top-level gate.
5. If the flat pair snapshot is stale, expect the EA to fall back to MT5 calendar-cache posture rather than treating the pair as silently safe.

## Example Case Scenarios

### Scenario: Nonfarm Payrolls day

What you will likely see:
- higher news-risk score
- more pairs in `CAUTION` or `BLOCK`
- stronger interaction with rates-aware context

How to respond:
1. Confirm the sources are fresh.
2. Compare `EURUSD`, `USDJPY`, and `GBPUSD`.
3. Expect the system to be more selective even if the directional model is confident.

### Scenario: NewsPulse says everything is stale

What to do:
1. Run `newspulse-health`.
2. Identify whether the problem is calendar export, official feed, or GDELT backoff.
3. Repair the data path.
4. Do not loosen the unknown-state block just to force trading.

### Scenario: A trader says the system ignored a seemingly obvious setup

What to do:
1. Check the pair's NewsPulse reasons.
2. Determine whether an upcoming event or stale source forced caution or block.
3. Use that explanation to decide whether the system prevented a low-quality trade.

## What Good NewsPulse Use Looks Like

- healthy sources
- clear awareness of upcoming policy events
- explicit respect for stale-source blocks
- replayable event context for later audit and research
- understanding that the EA can now degrade from full NewsPulse state to MT5 calendar-cache state and still remain event-aware

## Next Pages

- [Runtime Control Plane](Runtime%20Control%20Plane.md) for how NewsPulse feeds the live stack.
- [GUI](GUI.md) for the operator surface that visualizes source health and pair drill-down.
