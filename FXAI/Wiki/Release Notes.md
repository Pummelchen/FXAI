# Release Notes

This page explains how FXAI release notes are tied to benchmark deltas.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Why This Page Matters

- Live Trader: release notes should tell you what actually improved, not only that a release exists.
- Demo Trader: release notes help explain why demo behavior changed after a new promotion or runtime profile.
- Backtester: release notes show which benchmark context changed and by how much.
- EA Researcher: release notes create a discipline of linking profile or model changes to measured deltas.
- System Architect: release notes make it easier to separate code churn from evidence-backed upgrades.

## Published Release-Delta Artifacts

The generated reference files live under:

- `Tools/Benchmarks/ReleaseNotes/reference_release_notes.md`
- `Tools/Benchmarks/ReleaseNotes/reference_release_notes.json`

They are built from the benchmark publisher and compare the current benchmark snapshot with the reference audit context.

## How To Regenerate

```bash
cd /path/to/FXAI
python3 Tools/fxai_testlab.py publish-benchmarks --profile bestparams --release-tag reference
```

## What A Good FXAI Release Note Includes

- the benchmark context that changed
- the model or plugin change
- the strategy-profile change
- the audit-score delta
- any visible walkforward or adversarial delta
- a link back to the benchmark matrix

## Example Case Scenarios

### Scenario: A researcher changed the promoted model

What to expect:
1. The release note should explicitly name the old model and the new model.
2. It should show the benchmark delta for the affected context.
3. If there is no measurable delta, the change should not be described as a clear upgrade.

### Scenario: A strategy profile changed but the model did not

What to expect:
1. The release note should still mention the strategy-profile change.
2. It should show whether the profile change improved score, walkforward behavior, or adversarial resilience.
3. It should link back to the benchmark row and promoted strategy manifest.

### Scenario: A user reads a release claim with no benchmark delta

Interpretation:
- the note is incomplete and should not be treated as strong evidence.

What to do:
1. Open the benchmark matrix.
2. Open the release note.
3. Confirm the context, thresholds, and delta all line up.

## Good Habits

- Treat release notes as evidence summaries, not hype.
- Always link model or profile changes to benchmark deltas.
- Keep release-note generation tied to the same benchmark publisher used for the public matrix.

## Next Pages

- [Benchmarks](Benchmarks.md)
- [Promotion Criteria](Promotion%20Criteria.md)
- [Getting Started](Getting%20Started.md)
