# FXAI Governance

This document is the canonical governance contract for FXAI code, data,
research promotion, plugin certification, demo execution, live execution, and
operator documentation. It exists to make release and promotion decisions
repeatable, auditable, and fail-closed.

## Scope

Governance applies to:

- Swift packages under `FXImporter`, `FXDatabase`, `FXDataEngine`, `FXPlugins`,
  `FXBacktest`, `FXGUI`, `FXTools`, and the agent packages.
- Plugin CPU, Metal, PyTorch, TensorFlow, and NLP implementations.
- FXDatabase ingestion, canonical storage, migrations, and API contracts.
- Offline Lab research state, tuning output, drift governance, lineage, and
  generated runtime promotion artifacts.
- Demo and live execution boundaries, including account scope, risk limits,
  kill switches, human release plans, and immutable audit events.
- Repository docs and GitHub wiki pages that describe operator-facing workflows.

## Authorities

| Area | Source of truth | Governance rule |
| --- | --- | --- |
| Market data and ClickHouse | `FXDatabase` APIs | No other package may connect to ClickHouse directly or mutate canonical data. |
| Features and plugin payloads | `FXDataEngine` contracts | Feature, label, context, volume, and cost semantics are changed only with tests and docs. |
| Plugin behavior | `FXPlugins` registry, manifests, tests, and plugin-local source | Every registered plugin and declared accelerator must stay covered by executable certification evidence. |
| Research and promotion state | Offline Lab Turso/libSQL state | Generated Offline Lab outputs are rebuilt from state and must not be edited by hand. |
| Runtime consumption artifacts | `FILE_COMMON/FXAI/Offline/Promotions/` | Runtime artifacts are deployment outputs; regenerate through Offline Lab commands when stale. |
| Demo/live execution | `FXExecutionContracts`, `FXDemoAgent`, and `FXLiveAgent` | Execution stays fail-closed until evidence, account scope, risk, stale-data, and kill-switch gates pass. |
| Operator docs | `README.md`, package READMEs, this file, and the GitHub wiki | User-facing docs must be updated in the same change as governance-relevant behavior. |

## Change Classes

Use the smallest class that describes the behavioral risk. Higher classes
inherit the lower-class checks.

| Class | Examples | Required evidence before push or release |
| --- | --- | --- |
| Documentation only | README, wiki, comments, operator wording | `git diff --check`; docs links point to existing files or pages; no secrets or credentials. |
| Package-local Swift behavior | Validation, DTOs, GUI services, package internals | Relevant `swift test --package-path <Package>` and `swift build --package-path <Package>` when build coverage is not already included by the tests. |
| Plugin CPU/reference behavior | Plugin manifests, CPU models, registry, reference equations | `swift test --package-path FXPlugins`; focused tests for changed plugin families; SineTest certification remains green. |
| Accelerator behavior | Metal kernels, PyTorch/TensorFlow/NLP bridge code, checkpoint policy | `swift test --package-path FXPlugins` with the Apple Silicon accelerator environment available; TensorFlow paths must use the Python 3.12 `tensorflow==2.18.1` and `tensorflow-metal==1.2.0` baseline unless the project baseline changes. |
| Data authority behavior | FXDatabase ingestion, ClickHouse schema, canonical rewrite, FXBacktest APIs | `swift test --package-path FXDatabase`; affected importer/backtest/data-engine tests; migration and data repair paths must preserve audit history. |
| Research or promotion behavior | Offline Lab tuning, drift governance, deployment profiles, lineage, calibration | `python3 FXDataEngine/Tools/fxai_testlab.py verify-all` when Python lab behavior changes; package tests for Swift consumers; generated artifacts must be reproducible from Turso/libSQL or runtime state. |
| Demo/live execution behavior | Account scoping, order intent, risk, kill switch, human release, broker boundary | `swift test --package-path FXExecutionContracts`, `FXDemoAgent`, and/or `FXLiveAgent`; no live order path may bypass promotion evidence and safety validation. |
| Release or live hardening | Anything intended to support production/live use | `./fxai certify --all` plus package-specific checks above; unresolved required gates block release. |

## Promotion Lifecycle

1. Research runs produce candidate parameters and telemetry in Offline Lab state.
2. `best-params`, `autonomous-governance`, `deploy-profiles`,
   `supervisor-sync`, `lineage-report`, and `minimal-bundle` derive promotion
   evidence and runtime artifacts from that state.
3. Champion/challenger governance can restrict, demote, shadow, or disable a
   plugin based on drift, calibration, execution quality, and performance
   evidence. Normal operation does not auto-promote challengers to live capital.
4. FXDatabase runtime loaders consume generated `FILE_COMMON` artifacts only
   when freshness, lineage, and format checks pass.
5. Demo and live agents accept work only after certification, SineTest,
   promotion lineage, account scope, risk limits, stale-data checks, and
   kill-switch checks pass.

Generated promotion artifacts are not hand-editable configuration. If they drift
or disappear, run `recover-artifacts`, `deploy-profiles`, and `supervisor-sync`
for the active profile instead of editing outputs directly.

## Release Gates

Before a live-release hardening change is considered complete:

- The repository must remain on the intended branch and the working tree must be
  clean after commit.
- `git diff --check` must pass.
- Relevant package tests must pass.
- `./fxai certify --all` must pass for release or live-readiness claims.
- FXPlugins full certification must not skip declared Apple Silicon accelerator
  checks when those checks are part of the claim.
- Documentation must name any new operator command, environment variable,
  generated artifact, failure mode, or gate that operators must understand.
- Known residual risks must be recorded in the relevant README, audit record, or
  issue tracker before release.

## Documentation Governance

Local repository docs are the implementation-adjacent source. The GitHub wiki is
the user-facing source. Keep both aligned when behavior affects installation,
architecture, operator workflows, release gates, or live/demo trading safety.

Documentation changes must not include:

- broker, database, API, Turso, or GitHub credentials
- private account identifiers beyond sanitized examples
- commands that bypass FXDatabase, promotion evidence, risk validation, or kill
  switches
- generated Offline Lab artifacts as if they were hand-maintained config

When a governance-relevant code change lands, update the closest package README
and either this document or the wiki when the operator-facing contract changes.

## Incident Workflow

Use this order for production, demo, live, data, or governance incidents:

1. Stop or keep stopped any affected execution path through the kill switch or
   fail-closed agent gate.
2. Preserve logs, audit rows, promotion lineage, generated artifacts, and source
   evidence before repairing anything.
3. Validate environment and data ownership boundaries with the package-specific
   commands in the relevant README.
4. Rebuild generated Offline Lab or runtime artifacts from source state instead
   of editing them by hand.
5. Run the smallest focused reproduction test, then the required class-level
   verification gate.
6. Record the root cause, repair, verification evidence, and residual risk in
   the relevant audit record or operator-facing doc.

## Baseline Commands

Use these as the default governance verification set:

```bash
git diff --check
./fxai certify --build-only
./fxai certify --all
swift test --package-path FXDatabase
swift test --package-path FXDataEngine
swift test --package-path FXPlugins
swift test --package-path FXBacktest
swift test --package-path FXExecutionContracts
swift test --package-path FXDemoAgent
swift test --package-path FXLiveAgent
python3 FXDataEngine/Tools/fxai_testlab.py verify-all
```

Run only the checks that match the change class during normal development. Run
the full set before release or live-readiness claims.
