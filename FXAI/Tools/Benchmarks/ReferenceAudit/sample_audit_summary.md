# FXAI Audit Summary

execution_profile: default
manifest: <FXAI_ROOT>/Tools/Benchmarks/ReferenceAudit/sample_audit.manifest.json

## rule_m1sync | 73.5/100 | Grade D
Issues: missing required scenario: drift_up; missing required scenario: drift_down; missing required scenario: vol_cluster; missing required scenario: monotonic_down
Scenarios: monotonic_up=98.0, random_walk=97.0
