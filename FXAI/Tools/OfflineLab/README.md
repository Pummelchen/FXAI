# FXAI Offline Lab

`fxai_offline_lab.py` is the Turso-backed offline control loop for FXAI.

It does not replace the MT5 model engine. MT5 and MQL5 still execute the real plugins. The offline lab adds:
- exact-window `M1 OHLC + spread` export from MT5
- Turso storage for exported bars, tuning runs, scenario metrics, promoted configs, branch/PITR metadata, audit-log events, and native research vectors
- repeated model-zoo tuning on 3/6/12-month windows
- automatic promotion of best parameter packs per symbol and plugin
- champion/challenger governance, parameter lineage, and family scorecards
- attribution and pruning profiles for feature families, plugin families, and live router policy
- student-router profiles that bound live model breadth and family weighting per symbol
- supervisor-service and supervisor-command artifacts for live portfolio-pressure and lifecycle control
- distillation artifacts for lighter student targets and learned red-team plans for future hostile-market runs
- NewsPulse shared-news infrastructure for MT5 calendar export, GDELT fusion, runtime gating, and operator visibility
- Rates Engine shared macro infrastructure for front-end differentials, policy-path proxies, curve state, NewsPulse enrichment, and runtime rates-aware gating
- Cross-Asset shared macro/liquidity infrastructure for rates-aware global context, equity-risk, commodity shock, volatility stress, dollar-liquidity stress, pair-level gating, and GUI visibility
- Microstructure shared execution-state infrastructure for MT5 tick-flow, spread-dynamics, liquidity-stress, stop-run proxy detection, session handoff, runtime gating, and replay visibility
- Probabilistic Calibration shared decision-quality infrastructure for calibrated ensemble probabilities, cost-aware edge filtering, abstention reasons, and replay visibility
- Execution-Quality shared execution-intelligence infrastructure for forecasted spread widening, slippage stress, fill quality, latency sensitivity, liquidity fragility, and runtime execution-state controls
- ready-to-use MT5 `.set` files so no parameter copy/paste is needed

Main commands from the repo root:

```bash
python3 FXAI/Tools/fxai_offline_lab.py validate-env
python3 FXAI/Tools/fxai_offline_lab.py bootstrap --seed-demo
python3 FXAI/Tools/fxai_offline_lab.py init-db
python3 FXAI/Tools/fxai_offline_lab.py export-dataset --symbol-pack majors --months-list 3,6,12
python3 FXAI/Tools/fxai_offline_lab.py tune-zoo --profile continuous --auto-export --symbol-pack majors --months-list 3,6,12
python3 FXAI/Tools/fxai_offline_lab.py best-params --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py turso-branch-create --profile continuous --source-database fxai-main
python3 FXAI/Tools/fxai_offline_lab.py turso-pitr-restore --profile continuous --source-database fxai-main --timestamp 2026-04-04T00:00:00Z
python3 FXAI/Tools/fxai_offline_lab.py turso-audit-sync --limit 50 --pages 1
python3 FXAI/Tools/fxai_offline_lab.py attribution-prune --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py turso-vector-reindex --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py turso-vector-neighbors --profile continuous --symbol EURUSD
python3 FXAI/Tools/fxai_offline_lab.py newspulse-validate
python3 FXAI/Tools/fxai_offline_lab.py newspulse-install-service
python3 FXAI/Tools/fxai_offline_lab.py newspulse-once
python3 FXAI/Tools/fxai_offline_lab.py newspulse-daemon --interval-seconds 60
python3 FXAI/Tools/fxai_offline_lab.py rates-engine-validate
python3 FXAI/Tools/fxai_offline_lab.py rates-engine-once
python3 FXAI/Tools/fxai_offline_lab.py rates-engine-daemon --interval-seconds 120
python3 FXAI/Tools/fxai_offline_lab.py rates-engine-replay-report --symbol EURUSD --hours-back 72
python3 FXAI/Tools/fxai_offline_lab.py cross-asset-validate
python3 FXAI/Tools/fxai_offline_lab.py cross-asset-install-service
python3 FXAI/Tools/fxai_offline_lab.py cross-asset-once
python3 FXAI/Tools/fxai_offline_lab.py cross-asset-daemon --interval-seconds 120
python3 FXAI/Tools/fxai_offline_lab.py cross-asset-health
python3 FXAI/Tools/fxai_offline_lab.py cross-asset-replay-report --symbol EURUSD --hours-back 72
python3 FXAI/Tools/fxai_offline_lab.py microstructure-validate
python3 FXAI/Tools/fxai_offline_lab.py microstructure-install-service
python3 FXAI/Tools/fxai_offline_lab.py microstructure-health
python3 FXAI/Tools/fxai_offline_lab.py microstructure-replay-report --symbol EURUSD --hours-back 72
python3 FXAI/Tools/fxai_offline_lab.py prob-calibration-validate
python3 FXAI/Tools/fxai_offline_lab.py prob-calibration-replay-report --symbol EURUSD --hours-back 72
python3 FXAI/Tools/fxai_offline_lab.py execution-quality-validate
python3 FXAI/Tools/fxai_offline_lab.py execution-quality-replay-report --symbol EURUSD --hours-back 72
python3 FXAI/Tools/fxai_offline_lab.py deploy-profiles --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py supervisor-sync --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py autonomous-governance --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py lineage-report --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py minimal-bundle --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py recover-artifacts --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py verify-deterministic
python3 FXAI/Tools/fxai_offline_lab.py control-loop --profile continuous --symbol-pack majors --months-list 3,6,12 --cycles 0 --sleep-seconds 1800
python3 FXAI/Tools/fxai_testlab.py verify-all
```

Notes:
- `validate-env` checks Python, pytest, libSQL, MT5 path assumptions, `FILE_COMMON`, and local writeability before a long run starts. The Turso CLI is reported when present, but it is not required for local-only or already-credentialed runs.
- Set `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` together to run the lab as a Turso embedded replica; without them the lab runs local-only via libSQL against the same on-disk lab file.
- Set `TURSO_ENCRYPTION_KEY` if you want the local `.turso.db` file encrypted at rest.
- Set `TURSO_SYNC_INTERVAL_SECONDS` if you want the embedded replica to auto-sync periodically in addition to explicit commit-time sync.
- Set `TURSO_DATABASE_NAME`, `TURSO_ORGANIZATION`, and `TURSO_API_TOKEN` when you want branch inventory, point-in-time restore branches, or audit-log ingestion.
- A partial Turso environment is treated as invalid on purpose. If only one of `TURSO_DATABASE_URL` or `TURSO_AUTH_TOKEN` is set, `validate-env` will fail and the lab will refuse to open until the configuration is complete or cleared.
- A partial Turso Platform API environment is also treated as invalid on purpose. If only one of `TURSO_ORGANIZATION` or `TURSO_API_TOKEN` is set, the lab will refuse to open until the configuration is completed or cleared.
- `bootstrap --seed-demo` creates the full directory and Turso/libSQL layout, validates the environment, and emits a deterministic smoke profile with dashboard, lineage, supervisor, world-plan, and minimal-bundle artifacts.
- `seed-demo` can rebuild the smoke profile later without recreating the whole lab.
- `best-params` promotes all symbols under the selected profile by default; use `--symbol`, `--symbol-list`, or `--symbol-pack` only when you want to narrow the scope.
- `turso-branch-create` creates an isolated Turso branch, mints a database token, and writes a branch env artifact for safe research isolation.
- `turso-pitr-restore` creates a new branch from an RFC3339 timestamp so governance or promotion state can be rolled back without mutating the current primary database.
- `turso-audit-sync` ingests Turso organization audit-log events into the research store so dashboards and incident review can include platform-side actions.
- `attribution-prune` builds `FILE_COMMON` attribution and student-router artifacts without needing a full deployment refresh.
- `turso-vector-reindex` builds native Turso vector embeddings for shadow observations and family scorecards.
- `turso-vector-neighbors` inspects nearest analog-state neighbors from the Turso vector index.
- `deploy-profiles` emits the live deployment TSVs consumed by the MT5 runtime.
- `deploy-profiles` now also emits fitted deployment gains derived from shadow telemetry so live teacher, student, macro, foundation, and lifecycle trust can be tuned per symbol.
- `deploy-profiles`, `autonomous-governance`, and `recover-artifacts` also emit operator dashboard, lineage, performance, and minimal live-bundle artifacts.
- `supervisor-sync` refreshes central supervisor-service and supervisor-command artifacts from live control-plane snapshots.
- `supervisor-sync` now emits freshness-bounded supervisor artifacts with directional entry-budget multipliers and pressure velocity, so stale or asymmetric portfolio pressure can be rejected by MT5 runtime loaders.
- `autonomous-governance` rebuilds portfolio supervisor and world-plan outputs from current research telemetry.
- `autonomous-governance` now also rebuilds causal attribution, plugin-pruning, and richer world plans with transition entropy, shock decay, and session-level stress scales.
- `lineage-report` explains why each symbol is running its current promoted plugin and how the champion and challenger chain was formed.
- `minimal-bundle` emits a deployment-only artifact set for a profile so operators can stage a lean live runtime without carrying the full research output tree.
- `recover-artifacts` rebuilds generated runtime artifacts from Turso/libSQL state if `FILE_COMMON` or Offline Lab outputs go stale or are deleted.
- `verify-deterministic` refreshes or checks the golden fixture outputs for the research OS artifact contract.
- `newspulse-install-service` copies and compiles the MT5 Economic Calendar service into `MQL5/Services/`.
- `newspulse-once` runs one bounded GDELT+calendar fusion cycle and writes the shared snapshot into `FILE_COMMON/FXAI/Runtime/`.
- `newspulse-daemon` keeps that snapshot refreshed continuously for runtime and GUI consumers.
- `rates-engine-validate` writes default config and numeric-input templates, validates thresholds, and confirms the rates subsystem can boot cleanly.
- `rates-engine-once` builds one shared rates snapshot from current operator numeric inputs and NewsPulse context.
- `rates-engine-daemon` keeps the rates snapshot refreshed continuously for runtime, NewsPulse enrichment, Adaptive Router context, and GUI consumers.
- `rates-engine-replay-report` summarizes recent pair gates, regime transitions, and policy-path behavior from append-only rates history.
- `cross-asset-validate` writes the default cross-asset config plus the MT5 probe config and confirms the cross-market context layer can boot cleanly.
- `cross-asset-install-service` installs and compiles the MT5 probe used to publish indicator-only context symbols into the shared runtime directory.
- `cross-asset-once` builds one shared cross-asset macro/liquidity snapshot from the probe service and the Rates Engine snapshot.
- `cross-asset-daemon` keeps the cross-asset snapshot refreshed continuously for routing, calibration, execution-quality, runtime risk, and GUI consumers.
- `cross-asset-health` shows current shared-state health for the probe service, snapshot, and replay layer.
- `cross-asset-replay-report` summarizes recent pair gates, macro-state transitions, and top cross-asset reasons from append-only shared history.
- `microstructure-validate` writes the default service config, validates thresholds and required rolling windows, and confirms the microstructure subsystem can boot cleanly.
- `microstructure-install-service` installs and compiles the MT5 tick-probe service used by the microstructure subsystem.
- `microstructure-health` shows current service, artifact, and replay status for the shared microstructure layer.
- `microstructure-replay-report` summarizes recent per-symbol regime shifts, hostile-execution transitions, and stop-run proxy events from append-only microstructure history.
- `prob-calibration-validate` writes the default calibration config and tier-memory exports used by the MT5 runtime, validates thresholds, and confirms the calibration layer can boot cleanly.
- `prob-calibration-replay-report` summarizes recent calibrated final actions, tier usage, abstention counts, edge-after-costs ranges, and top abstention reasons from append-only runtime history.
- `execution-quality-validate` writes the default execution-quality config and tier-memory exports used by the MT5 runtime, validates thresholds, and confirms the execution forecaster can boot cleanly.
- `execution-quality-replay-report` summarizes recent execution-state transitions, spread or slippage stress, liquidity-fragility changes, and execution-quality reasons from append-only runtime history.
- `fxai_testlab.py verify-all` is the one-command platform verification path: Python tests, deterministic fixture checks, and clean MT5 compiles.
- Exact-window datasets store the effective exported first and last bar range, so later tuning and promotion stay aligned to the data that was actually ingested.
- Turso access uses bounded open retry so overlapping admin and control-loop calls fail cleanly instead of drifting silently.

Runtime modes:
- `research`
  Full telemetry, shadow support, wider model breadth, and the richer operator artifact set.
- `production`
  Leaner telemetry, lower runtime budget, fewer active models, and artifacts aimed at the minimal live bundle.

Source of truth and recovery:
- Turso/libSQL is the authoritative research and promotion state.
- Turso branches are the preferred isolation surface for risky governance rehearsals or destructive what-if runs.
- `FILE_COMMON/FXAI/Offline/Promotions/` is the authoritative MT5 runtime consumption layer.
- Generated Offline Lab outputs must be rebuilt from Turso/libSQL or runtime artifacts, never edited by hand.
- If runtime artifacts drift or disappear, run `recover-artifacts`, then `deploy-profiles`, then `supervisor-sync`.

Operator incident workflow:
1. Run `validate-env` to rule out path, permission, or MT5-environment problems.
2. Run `lineage-report --profile <name>` to see what should be live.
3. Run `minimal-bundle --profile <name>` if you need a lean deployment pack for comparison.
4. Run `recover-artifacts --profile <name>` if `FILE_COMMON` artifacts or research outputs are stale or missing.
5. Run `fxai_testlab.py verify-all` before promoting or pushing changes.

Generated promotion artifacts land in:
- `FXAI/Tools/OfflineLab/Profiles/`
- `MQL5/Profiles/Tester/`
- `FILE_COMMON/FXAI/Offline/Promotions/`

Research and governance artifacts land in:
- `FXAI/Tools/OfflineLab/ResearchOS/`
- `FXAI/Tools/OfflineLab/Distillation/`

These Offline Lab runtime outputs are generated artifacts and should not be versioned in git.
