# Quick Start By Role

This page is the fastest way to choose the right FXAI workflow for the job you are doing today.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## What FXAI Gives Each User

- Live Trader: a live decision stack you can interrogate before trusting a trade.
- Demo Trader: a safe place to learn the system's behavior under real gating and abstention logic.
- Backtester: repeatable experiments with clearer setup and comparison than hand-run tester sessions.
- EA Researcher: governed improvement loops with reports, promotion, and lineage.
- System Architect: recoverable control-plane operations instead of opaque runtime drift.

## Live Trader

### Benefits

- You can see why a trade is allowed, cautioned, blocked, or abstained.
- You can catch stale data or missing services before the market punishes you.
- You can avoid duplicated or contradictory exposure when multiple USD or JPY expressions compete.

### How To Use FXAI

1. Start with the latest runtime monitor or GUI live overview.
2. Check artifact freshness for NewsPulse, Rates Engine, Cross Asset, and Microstructure.
3. Inspect the final trade posture, confidence, expected edge after costs, and pair-network decision.
4. If the runtime is healthy, align your discretionary action with the system's posture instead of overriding it blindly.

### Example Scenario

You see a strong `EURUSD` directional score during a central-bank day.

What to do:
- Read the NewsPulse and Rates Engine posture first.
- If the pair shows `BLOCK` because policy-path state is stale or event risk is active, do not treat the raw model direction as actionable.
- If the pair shows `CAUTION`, read the lot scaling and extra probability buffer before deciding whether the reduced trade still fits your plan.

## Demo Trader

### Benefits

- You can learn what the system does under stress without putting live capital at risk.
- You can compare the runtime story with the audit story and see where they diverge.
- You can build trust in the abstention logic instead of judging the system only by "did it fire a trade."

### How To Use FXAI

1. Run the same control-plane services you would use live.
2. Watch how the system behaves in a known event window or thin-liquidity session.
3. Compare what the runtime blocked with what Audit Lab says would have happened.
4. Keep notes on why trades were avoided, not only on which ones won.

### Example Scenario

You want to understand what happens during Asia session liquidity gaps.

What to do:
- Let Microstructure and Execution Quality run.
- Watch whether spread widening and liquidity fragility force abstention.
- Compare that to a quiet London overlap period to see how the same pair behaves differently.

## Backtester

### Benefits

- You can launch focused evaluations faster.
- You can compare runs with scenario awareness instead of treating all periods as equally informative.
- You can use audit outputs to understand why a candidate passes or fails.

### How To Use FXAI

1. Define the symbol, date range, and scenario you care about.
2. Use the builder or terminal command to launch the run.
3. Review metrics together with scenario labels, not in isolation.
4. Compare against baselines and recent promoted state.

### Example Scenario

You want to compare `USDJPY` behavior in macro-event weeks versus quiet weeks.

What to do:
- Run two backtests or audits with clear windows.
- Compare not just return, but drawdown, trade count, abstention rate, and hostile-window behavior.
- Use the result to decide whether a candidate is robust or only lucky in one environment.

## EA Researcher

### Benefits

- You can improve the whole decision stack: labels, routing, calibration, execution, and governance.
- You can promote with lineage instead of manual file shuffling.
- You can detect when a challenger is better for real reasons rather than noise.

### How To Use FXAI

1. Start in Offline Lab and inspect the current champion profile.
2. Review family scorecards, replay reports, and drift/governance state.
3. Improve the weak layer first, not only the headline model.
4. Rebuild artifacts and verify the promoted runtime view before trusting the candidate.

### Example Scenario

A new challenger improves raw direction but worsens post-cost tradability.

What to do:
- Check probabilistic calibration and execution-quality effects.
- If expected edge after costs degrades, do not promote just because accuracy improved.
- Use label-engine and calibration outputs to see whether the challenger is overtrading noise.

## System Architect

### Benefits

- You can inspect service health, artifact freshness, and recovery posture from one coherent operational frame.
- You can recover missing runtime artifacts without guessing what to rebuild.
- You can operate the research OS and promotion surfaces with fewer hidden steps.

### How To Use FXAI

1. Check the operator dashboard or platform-control surface first.
2. Confirm the data path from services to runtime artifacts is healthy.
3. If artifacts are stale or missing, use the documented recovery or rebuild commands.
4. Re-run verification before handing the system back to traders or researchers.

### Example Scenario

The live runtime starts blocking everything after a terminal restart.

What to do:
- Check which upstream service or artifact is stale.
- Rebuild or restart the affected layer instead of loosening safety settings.
- Verify that the system returns to a healthy `ALLOW` or `CAUTION` posture for the right reasons, not because controls were bypassed.

## Next Pages

- Read [Getting Started](Getting%20Started.md) if you need a clean first-run workflow.
- Read [Audit Lab](Audit%20Lab.md) for evaluation and hostile-window testing.
- Read [Offline Lab](Offline%20Lab.md) for research, promotion, and artifact recovery.
- Read [Runtime Control Plane](Runtime%20Control%20Plane.md) if you want to understand how live decisions are filtered.
