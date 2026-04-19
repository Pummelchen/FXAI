# Getting Started

This page is the practical first-run guide for new FXAI users.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Why This Page Matters

- Live Trader: this page shows how to confirm the system is healthy before trusting live posture.
- Demo Trader: this page shows how to stand up the same safety layers without live risk.
- Backtester: this page shows the shortest path to a clean validation run.
- EA Researcher: this page shows the base verification commands before deeper research work.
- System Architect: this page shows the minimum operational checklist before platform changes.

## Basic Setup

1. Change into the FXAI project directory.
2. Confirm the repo and MT5 tree are aligned before you trust local edits.
3. Run the main verification gate before any release or promotion-sensitive work.

Example:

```bash
cd /path/to/FXAI
python3 Tools/fxai_testlab.py doctor
python3 Tools/fxai_testlab.py verify-all
```

## First Commands To Know

```bash
python3 Tools/fxai_testlab.py doctor
python3 Tools/fxai_testlab.py compile-main
python3 Tools/fxai_testlab.py verify-all
python3 Tools/fxai_testlab.py publish-benchmarks --profile bestparams
python3 Tools/fxai_offline_lab.py newspulse-health
python3 Tools/fxai_offline_lab.py rates-engine-health
python3 Tools/fxai_offline_lab.py cross-asset-health
python3 Tools/fxai_offline_lab.py microstructure-health
python3 Tools/fxai_offline_lab.py live-state --symbol EURUSD
```

What they do:
- `doctor`: verifies the active toolchain profile, path overrides from `fxai.toml` or `.env`, and whether MT5 compile or terminal launch prerequisites are actually present.
- `compile-main`: checks that the EA still compiles cleanly.
- `verify-all`: runs the broader release gate.
- `publish-benchmarks`: writes the public benchmark matrix, reference audit bundle, promotion criteria, and release-note delta artifacts.
- `*-health`: tells you whether the shared runtime layers are fresh enough to trust.
- `live-state --symbol ...`: shows what the live decision stack currently believes for one pair.

## Toolchain Profiles

FXAI no longer assumes one operator machine layout. The Python toolchain auto-detects one of these profiles and can be overridden in `fxai.toml` or `.env`:

- `macos_wine`: MT5 under Wine on macOS.
- `windows_native`: MT5 and MetaEditor running directly on Windows.
- `headless_ci`: no local MT5 install required, with runtime/common-file outputs rooted under `FILE_COMMON`.

Useful overrides include:
- `FXAI_MT5_ROOT`
- `FXAI_METAEDITOR`
- `FXAI_TERMINAL`
- `FXAI_COMMON_FILES`
- `FXAI_RUNTIME_DIR`
- `FXAI_DEFAULT_DB`

## Choose The Right First Workflow

### If You Are Trading

1. Check live health.
2. Check the per-symbol live state.
3. Only then interpret the pair signal.

### If You Are Learning

1. Run the health checks.
2. Observe one known pair through a quiet session and an event session.
3. Compare the differences in gating, scaling, and abstention.

### If You Are Testing

1. Compile first.
2. Run `verify-all`.
3. Launch the specific Audit Lab or backtest workflow you need.

### If You Are Researching

1. Validate the environment.
2. Review Offline Lab outputs and current promoted artifacts.
3. Make controlled changes and rerun the relevant validation command.

### If You Are Operating The Platform

1. Verify health and artifact freshness.
2. Check operator dashboard or Research OS outputs.
3. Rebuild missing artifacts before you relax any protection.

## Example Case Scenarios

### Scenario: The EA compiles but live trading looks frozen

Likely cause:
- one of the control-plane layers is stale and the runtime is correctly failing closed.

What to do:
1. Run the four health commands.
2. Identify the stale layer.
3. Refresh or restart that layer.
4. Re-check `live-state --symbol EURUSD`.

### Scenario: A new user wants to see value in 15 minutes

What to do:
1. Open the GUI or run `live-state --symbol EURUSD`.
2. Read the final posture and reasons.
3. Run one `newspulse-health` command.
4. Compare a healthy pair with a blocked pair.

Why this works:
- it demonstrates that FXAI is a decision-quality framework, not only a directional model.

### Scenario: You need a clean pre-release confidence check

What to do:
1. Run `python3 Tools/fxai_testlab.py verify-all`.
2. If you changed the GUI, run `cd GUI && swift test`.
3. If you changed an Offline Lab subsystem, run its dedicated `*-validate` command too.
4. Build MT5 release binaries with `python3 Tools/fxai_testlab.py package-mt5-release --version <tag>`.
5. Upload the generated `.ex5` files, manifest, and SHA-256 checksum files to GitHub Releases; do not commit compiled MT5 binaries.

## Common Mistakes

- Trusting the model direction without checking whether the shared runtime layers are fresh.
- Fixing a stale-service problem by loosening block-on-unknown settings instead of repairing the data path.
- Comparing research candidates only by directional metrics and ignoring calibration, cost, or execution-quality changes.

## Next Pages

- [Quick Start By Role](Quick%20Start%20By%20Role.md)
- [Benchmarks](Benchmarks.md)
- [Promotion Criteria](Promotion%20Criteria.md)
- [Release Notes](Release%20Notes.md)
- [Audit Lab](Audit%20Lab.md)
- [Offline Lab](Offline%20Lab.md)
- [NewsPulse](NewsPulse.md)
- [Runtime Control Plane](Runtime%20Control%20Plane.md)
- [GUI](GUI.md)
