# Drift Governance

The Online Drift Detector + Champion/Challenger Governance subsystem monitors live plugin health, scores multiple forms of drift, and writes deterministic governance state that the runtime and operator GUI can consume.

Phase 1 covers:

- feature drift
- regime drift
- calibration drift
- pair-specific decay
- post-event behavior drift
- execution-related drift
- performance / utility drift
- conservative demotion, restriction, shadow-only, and disable recommendations
- challenger promotion eligibility tracking with support gates

## Commands

Validate config:

```bash
python3 Tools/fxai_offline_lab.py drift-governance-validate
```

Run one governance cycle for the active profile:

```bash
python3 Tools/fxai_offline_lab.py drift-governance-run --profile continuous
```

Rebuild the aggregated report without rerunning the full cycle:

```bash
python3 Tools/fxai_offline_lab.py drift-governance-report --profile continuous
```

## Artifacts

- Config: `/Users/andreborchert/FXAI-main2/FXAI/Tools/OfflineLab/DriftGovernance/drift_governance_config.json`
- Status: `/Users/andreborchert/FXAI-main2/FXAI/Tools/OfflineLab/DriftGovernance/drift_governance_status.json`
- Report: `/Users/andreborchert/FXAI-main2/FXAI/Tools/OfflineLab/DriftGovernance/Reports/drift_governance_report.json`
- History: `/Users/andreborchert/FXAI-main2/FXAI/Tools/OfflineLab/DriftGovernance/drift_governance_history.ndjson`
- Runtime summary: `/Users/andreborchert/FXAI-main2/FXAI/Tools/Runtime/drift_governance_summary.json`

## Notes

- Governance is separate from detection: drift metrics score change, then policy maps that evidence into recommended or applied actions.
- Promotions remain conservative. Phase 1 does not auto-promote challengers in normal operation.
- Empty output is valid when no shadow-fleet or registry data is present for the chosen profile. The subsystem keeps writing deterministic empty-state artifacts instead of failing.
- The GUI drift-governance surface reads only the aggregated report and status artifacts. It does not open the database directly.
