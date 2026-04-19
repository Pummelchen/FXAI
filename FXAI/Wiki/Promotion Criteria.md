# Promotion Criteria

This page explains the exact public release thresholds that back FXAI promotions.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Why This Page Matters

- Live Trader: it tells you what a profile had to survive before it was promoted.
- Demo Trader: it explains why some candidates never become live profiles even when one chart looks good.
- Backtester: it shows which metrics must clear the bar for a candidate to count as release-ready.
- EA Researcher: it removes guesswork from promotion reviews.
- System Architect: it keeps release discipline anchored to code-defined gates instead of memory or chat history.

## Source Of Truth

The canonical exported artifact is:

- `Tools/Benchmarks/promotion_criteria.md`
- `Tools/Benchmarks/promotion_criteria.json`

Those files are generated from `Tools/testlab/release_gate.py`, so the published criteria and the actual release gate stay aligned.

## How To Regenerate

```bash
cd /path/to/FXAI
python3 Tools/fxai_testlab.py publish-benchmarks --profile bestparams
```

## What The Criteria Cover

- minimum audit score
- minimum cross-symbol stability
- walkforward thresholds
- adversarial thresholds
- macro-dataset thresholds
- runtime performance thresholds
- artifact size budgets

## Example Case Scenarios

### Scenario: A candidate improved headline score but still should not ship

What to check:
1. Open `Tools/Benchmarks/promotion_criteria.md`.
2. Compare the candidate against walkforward, adversarial, and macro thresholds.
3. If one hard gate fails, treat the candidate as non-promotable regardless of a prettier average score.

### Scenario: An operator wants to understand why a release gate failed

What to do:
1. Read the failed metric in the release output.
2. Find the matching threshold in `promotion_criteria.md`.
3. Fix the real weakness instead of lowering the threshold to make the failure disappear.

### Scenario: A researcher wants to justify a threshold change

What to do:
1. Change the threshold in `Tools/testlab/release_gate.py`.
2. Regenerate the benchmark suite.
3. Review the benchmark and release-note deltas.
4. Document why the new threshold improves real release safety.

## Good Habits

- Keep the public criteria generated from code.
- Treat any undocumented threshold drift as a bug.
- Use the criteria page together with the benchmark matrix, not in isolation.

## Next Pages

- [Benchmarks](Benchmarks.md)
- [Release Notes](Release%20Notes.md)
- [Audit Lab](Audit%20Lab.md)
