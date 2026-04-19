# Offline Lab

Offline Lab is the research, promotion, and artifact-recovery engine behind FXAI.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Why This Page Matters

- Live Trader: Offline Lab is where the promoted profile you trust was produced and validated.
- Demo Trader: it explains why demo behavior changes after a promotion.
- Backtester: it gives you the research context behind the runs you compare.
- EA Researcher: it is your main workspace.
- System Architect: it is where you rebuild artifacts and recover clean state.

## What Offline Lab Can Do

- bootstrap and seed demo research state
- export datasets and build evaluation inputs
- run tuning campaigns and compare families
- maintain champion and challenger governance
- emit promoted runtime bundles, operator dashboards, lineage, and minimal bundles
- recover artifacts when runtime or research outputs go stale

## Strategy Profiles

- promoted audit and EA presets are compiled from `Tools/OfflineLab/Profiles/strategy_profiles.json`
- the compiler layers `strategy -> symbol -> broker -> runtime` before applying run-specific overrides
- each promoted preset now carries a sibling `__strategy_profile.json` manifest so operators can see exactly which profile version and inheritance chain produced the artifact
- if you need a new deployment posture, change the catalog and regenerate artifacts instead of hand-editing dozens of MT5 input fields

## Core Commands

```bash
cd /path/to/FXAI
python3 Tools/fxai_offline_lab.py doctor
python3 Tools/fxai_offline_lab.py bootstrap --seed-demo
python3 Tools/fxai_offline_lab.py verify-deterministic
python3 Tools/fxai_offline_lab.py recover-artifacts
python3 Tools/fxai_offline_lab.py pair-network-build --profile continuous
python3 Tools/fxai_offline_lab.py dashboard --profile continuous
python3 Tools/fxai_testlab.py publish-benchmarks --profile bestparams
```

## Practical Workflows

### Promote A Better Candidate

1. Review current champion outputs.
2. Compare the challenger on audit quality, calibration, execution quality, and drift posture.
3. Promote only if the change improves real tradability, not only one headline metric.
4. Rebuild runtime artifacts and inspect the dashboard after promotion.

### Recover A Broken Runtime Bundle

1. Confirm which artifact is stale or missing.
2. Run `recover-artifacts`.
3. Rebuild any subsystem-specific artifacts such as pair network if needed.
4. Re-check live health before reopening trust in the runtime.

### Rebuild The Operator View

1. Refresh the promoted profile or research branch state.
2. Run the dashboard command.
3. Verify that the GUI and terminal now show the rebuilt state.

## Example Case Scenarios

### Scenario: Pair-network decisions look empty after a rebuild

What to do:
1. Run `pair-network-build --profile continuous`.
2. Re-run the relevant validation command.
3. Confirm the runtime status reports a built graph instead of `UNBUILT`.

### Scenario: A researcher wants to ship a new ensemble weighting scheme

What to do:
1. Validate the new subsystem outputs.
2. Compare them against current champions in Offline Lab.
3. Check that the promoted bundle produces coherent runtime artifacts before release.

### Scenario: A system architect needs to recover after accidental artifact deletion

What to do:
1. Rebuild from Offline Lab state, not by editing generated files.
2. Verify deterministic outputs.
3. Re-run release verification before handing the environment back to operators.

## What Good Offline Lab Work Looks Like

- the candidate improves post-cost decision quality
- lineage is clear
- dashboards and runtime artifacts are regenerated after changes
- promotion is conservative when drift or calibration quality is unclear

## Next Pages

- [Audit Lab](Audit%20Lab.md)
- [Runtime Control Plane](Runtime%20Control%20Plane.md)
- [GUI](GUI.md)
