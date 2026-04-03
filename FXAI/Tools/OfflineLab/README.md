# FXAI Offline Lab

`fxai_offline_lab.py` is the Turso/libSQL-backed offline control loop for FXAI.

It does not replace the MT5 model engine. MT5 and MQL5 still execute the real plugins. The offline lab adds:
- exact-window `M1 OHLC + spread` export from MT5
- Turso/libSQL storage for exported bars, tuning runs, scenario metrics, and promoted configs
- repeated model-zoo tuning on 3/6/12-month windows
- automatic promotion of best parameter packs per symbol and plugin
- champion/challenger governance, parameter lineage, and family scorecards
- attribution and pruning profiles for feature families, plugin families, and live router policy
- student-router profiles that bound live model breadth and family weighting per symbol
- supervisor-service and supervisor-command artifacts for live portfolio-pressure and lifecycle control
- distillation artifacts for lighter student targets and learned red-team plans for future hostile-market runs
- ready-to-use MT5 `.set` files so no parameter copy/paste is needed

Main commands from the repo root:

```bash
python3 FXAI/Tools/fxai_offline_lab.py validate-env
python3 FXAI/Tools/fxai_offline_lab.py bootstrap --seed-demo
python3 FXAI/Tools/fxai_offline_lab.py init-db
python3 FXAI/Tools/fxai_offline_lab.py export-dataset --symbol-pack majors --months-list 3,6,12
python3 FXAI/Tools/fxai_offline_lab.py tune-zoo --profile continuous --auto-export --symbol-pack majors --months-list 3,6,12
python3 FXAI/Tools/fxai_offline_lab.py best-params --profile continuous
python3 FXAI/Tools/fxai_offline_lab.py attribution-prune --profile continuous
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
- A partial Turso environment is treated as invalid on purpose. If only one of `TURSO_DATABASE_URL` or `TURSO_AUTH_TOKEN` is set, `validate-env` will fail and the lab will refuse to open until the configuration is complete or cleared.
- `bootstrap --seed-demo` creates the full directory and Turso/libSQL layout, validates the environment, and emits a deterministic smoke profile with dashboard, lineage, supervisor, world-plan, and minimal-bundle artifacts.
- `seed-demo` can rebuild the smoke profile later without recreating the whole lab.
- `best-params` promotes all symbols under the selected profile by default; use `--symbol`, `--symbol-list`, or `--symbol-pack` only when you want to narrow the scope.
- `attribution-prune` builds `FILE_COMMON` attribution and student-router artifacts without needing a full deployment refresh.
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
- `fxai_testlab.py verify-all` is the one-command platform verification path: Python tests, deterministic fixture checks, and clean MT5 compiles.
- Exact-window datasets store the effective exported first and last bar range, so later tuning and promotion stay aligned to the data that was actually ingested.
- Turso/libSQL access is opened with a bounded retry and busy timeout so overlapping admin and control-loop calls do not fail on transient lock contention.

Runtime modes:
- `research`
  Full telemetry, shadow support, wider model breadth, and the richer operator artifact set.
- `production`
  Leaner telemetry, lower runtime budget, fewer active models, and artifacts aimed at the minimal live bundle.

Source of truth and recovery:
- Turso/libSQL is the authoritative research and promotion state.
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
