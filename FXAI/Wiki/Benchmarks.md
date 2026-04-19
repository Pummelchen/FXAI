# Benchmarks

This page explains the public benchmark surface for FXAI.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Why This Page Matters

- Live Trader: benchmarks show whether the promoted configuration has evidence behind it.
- Demo Trader: benchmarks make it easier to compare demo behavior with a known-good audit context.
- Backtester: benchmarks create a stable comparison surface across symbol packs, broker assumptions, and horizons.
- EA Researcher: benchmarks turn promotion claims into reproducible tables and linked artifacts.
- System Architect: benchmarks show whether a release changed model quality, not only code shape.

## What FXAI Publishes

- a public benchmark matrix in `Tools/Benchmarks/benchmark_matrix.md`
- a machine-readable benchmark matrix in `Tools/Benchmarks/benchmark_matrix.json`
- a reference audit bundle in `Tools/Benchmarks/ReferenceAudit/`
- exported promotion thresholds in `Tools/Benchmarks/promotion_criteria.md`
- release-note deltas in `Tools/Benchmarks/ReleaseNotes/`

## How To Regenerate The Benchmark Surface

```bash
cd /path/to/FXAI
python3 Tools/fxai_testlab.py publish-benchmarks --profile bestparams
```

That command writes:
- the benchmark matrix
- the sample audit TSV or JSON reference bundle
- the promotion-criteria export
- the release-note delta file

## How To Read The Benchmark Matrix

Each row tells you:
- which symbol pack or single-symbol context the row represents
- which broker profile and execution profile were assumed
- which target horizon was evaluated
- which strategy profile and plugin produced the result
- which linked artifact explains the row in more detail

## Example Case Scenarios

### Scenario: A live trader wants to know whether a promoted EURUSD profile has real evidence

1. Open `Tools/Benchmarks/benchmark_matrix.md`.
2. Find the `EURUSD` row for the matching broker and execution assumptions.
3. Open the linked strategy-profile or sample audit artifact.
4. Compare the row with the promotion criteria page before trusting the posture.

### Scenario: A researcher changed a strategy profile and wants to show the impact

1. Regenerate the benchmark surface.
2. Open `Tools/Benchmarks/ReleaseNotes/reference_release_notes.md`.
3. Verify that the model or profile change is paired with visible audit-score and scenario deltas.
4. If there is no meaningful delta, do not market the change as an upgrade.

### Scenario: A backtester wants a clean baseline for a horizon-sensitive change

1. Regenerate the benchmark surface after the run.
2. Compare the old and new rows for the same symbol, broker profile, execution profile, and horizon.
3. Only compare rows that share the same evaluation context.

## Good Habits

- Do not quote a benchmark number without its context row.
- Do not compare a stressed execution profile with a default one as if they were interchangeable.
- Treat the benchmark matrix as evidence, not as marketing copy.

## Next Pages

- [Promotion Criteria](Promotion%20Criteria.md)
- [Release Notes](Release%20Notes.md)
- [Audit Lab](Audit%20Lab.md)
- [Offline Lab](Offline%20Lab.md)
