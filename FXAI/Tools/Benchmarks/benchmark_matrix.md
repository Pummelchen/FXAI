# FXAI Public Benchmark Matrix

This matrix is generated from committed FXAI audit and promotion artifacts. It is intended to show benchmark context, not only top-line claims.

| Source | Symbol Pack | Symbol | Broker | Execution | Horizon | Strategy Profile | Plugin | Audit | Ranking | Walkforward | Adversarial | Grade | Reference |
|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| reference_audit | single:EURUSD | EURUSD | default | default | 5 | reference/legacy-audit@v0 | rule_m1sync | 43.5 | n/a | n/a | n/a | F | [artifact](ReferenceAudit/sample_audit_summary.md) |
| promoted_profile | single:EURUSD | EURUSD | default | default | 5 | strategy/default@v1 | ai_mlp | 83.5 | 82.71 | 79.5 | 77.5 | C | [artifact](../OfflineLab/Profiles/bestparams/EURUSD/ai_mlp__strategy_profile.json) |
