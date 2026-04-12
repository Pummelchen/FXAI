# Runtime Control Plane

This page explains how FXAI turns a raw model opinion into a final live-trading posture.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Why This Page Matters

- Live Trader: this is the core explanation for why a trade did or did not happen.
- Demo Trader: this is the best way to learn that FXAI is a governed stack, not a one-line signal.
- Backtester: this explains which layers should be reflected in scenario-aware evaluation.
- EA Researcher: this tells you where to improve the system without confusing one layer for another.
- System Architect: this shows which services and artifacts must stay healthy for live safety.

## Decision Layers

### NewsPulse

Purpose:
- protect the system around event risk and stale news context.

Typical effect:
- caution or block before high-risk macro events.

### Rates Engine

Purpose:
- interpret policy divergence, path repricing, and rates instability.

Typical effect:
- block when rates context is stale or when policy-shock conditions make direction unstable.

### Cross Asset

Purpose:
- inject global macro and liquidity context such as risk-off, commodities, volatility, and dollar stress.

Typical effect:
- suppress trades that look good locally but conflict with broader market state.

### Microstructure

Purpose:
- detect spread widening, liquidity stress, quote imbalance, and short-horizon hostile conditions.

Typical effect:
- abstain or scale down when execution quality is likely to destroy the theoretical edge.

### Adaptive Router

Purpose:
- identify the current market regime and favor the plugin families that historically behave better in that state.

Typical effect:
- different plugin emphasis in trend, range, event-shock, or liquidity-stress regimes.

### Dynamic Ensemble

Purpose:
- weight plugins by recent quality, pair, session, regime, and costs instead of treating them as equal voters.

Typical effect:
- less trust in unstable or miscalibrated plugins.

### Probabilistic Calibration And Abstention

Purpose:
- convert direction into calibrated confidence, expected move, and expected edge after costs.

Typical effect:
- no trade when the statistical edge is weaker than spread, slippage, and uncertainty.

### Execution Quality

Purpose:
- forecast whether the trade can actually be executed well enough to deserve sending.

Typical effect:
- block or caution when spread, slippage, or latency risk is too poor.

### Pair Network

Purpose:
- stop contradictory or redundant portfolio expressions.

Typical effect:
- block a new trade because it duplicates an open idea or conflicts with the current currency-factor book.

## How To Read A Live Decision

1. Read the final posture.
2. Read the strongest reasons.
3. Check freshness and health of the layers that influenced the decision.
4. Decide whether the block or caution is logical given current market conditions.

## Example Case Scenarios

### Scenario: Strong direction, no trade

Possible explanation:
- calibration says expected edge after costs is not good enough
- execution quality says spreads are too poor
- pair network says the trade duplicates existing exposure

What to do:
- inspect all three layers before calling it a missed opportunity.

### Scenario: Good research result, weak live behavior

Possible explanation:
- the research candidate works directionally, but the live environment is dominated by event risk or hostile execution.

What to do:
- compare audit windows to the live session conditions and inspect NewsPulse, Microstructure, and Execution Quality together.

### Scenario: Everything suddenly blocks after a restart

Possible explanation:
- one or more runtime layers are missing or stale, and the system is correctly failing closed.

What to do:
- restore the data path rather than downgrading the safety policy.

## Good Operator Interpretation

- A block is not automatically a bug.
- A cautious smaller trade can be healthier than a full-size trade with fragile execution.
- A good model with bad context is still a bad live trade.

## Next Pages

- [NewsPulse](NewsPulse.md)
- [Offline Lab](Offline%20Lab.md)
- [GUI](GUI.md)
