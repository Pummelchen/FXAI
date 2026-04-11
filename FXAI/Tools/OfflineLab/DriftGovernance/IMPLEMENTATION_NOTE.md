# Drift Governance Implementation Note

## Repo Mapping

- Drift metrics, policy config, math, DB persistence, report building, and runtime summaries live in:
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/drift_governance.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/drift_governance_config.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/drift_governance_math.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/drift_governance_contracts.py`
- Schema support is in `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/common_schema.py`.
- CLI entry points are wired through:
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/cli_parser.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/cli_commands.py`
- Runtime integration is non-invasive and currently happens through:
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/student_router.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/adaptive_router.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/governance.py`
- GUI ingestion and display are in:
  - `/Users/andreborchert/FXAI-main2/FXAI/GUI/Sources/FXAIGUICore/Services/DriftGovernanceArtifactReader.swift`
  - `/Users/andreborchert/FXAI-main2/FXAI/GUI/Sources/FXAIGUIApp/Features/DriftGovernance/DriftGovernanceView.swift`

## Phase-1 Scope

- Computes seven drift families using transparent score rules.
- Uses support-aware reference windows and fallback scope reduction.
- Persists per-plugin governance state plus append-only action history.
- Feeds applied states back into student-router and adaptive-router weights or restrictions.
- Tracks challenger eligibility but leaves promotion conservative by default.
- Publishes a report/status contract for the GUI and operator dashboards.

## Repo-Specific Deviations

- The live runtime does not yet consume a separate MT5 governance block. Instead, the subsystem influences runtime participation through regenerated router artifacts, which is the cleanest existing control surface in this repo.
- Challenger evaluation reuses existing `tuning_runs`, `run_scenarios`, `best_configs`, `champion_registry`, and `shadow_fleet_observations` tables instead of introducing a new standalone experiment store.
- GUI symbol selection reuses the existing runtime-symbol selection state to avoid a saved-workspace contract change.

## Verification Path

- `python3 Tools/fxai_offline_lab.py drift-governance-validate`
- `python3 Tools/fxai_offline_lab.py drift-governance-run`
- `python3 Tools/fxai_offline_lab.py drift-governance-report`
- `python3 -m pytest FXAI/Tools/tests/test_drift_governance.py FXAI/Tools/tests/test_dashboard.py FXAI/Tools/tests/test_adaptive_router.py FXAI/Tools/tests/test_cli_smoke.py FXAI/Tools/tests/test_offline_fixture_golden.py -q`
- `cd FXAI/GUI && swift test && swift build`
