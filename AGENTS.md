<!--
AI onboarding file.
Mode: bootstrap
Indexed commit: 91a97e92e5622fae867a490dcd917e6c32811955
Last generated: 2026-06-25T14:01:03Z
Generator: generic high-end AI coding agent
Purpose: Help future AI sessions understand this repository quickly.
Audience: Any high-capability AI coding agent, regardless of vendor or model family.
Human edits are allowed. Future refreshes should preserve valid human edits.
-->
# AGENTS.md — generic AI coding-agent instructions for FXAI

## Start every session this way

1. Read `AI_INDEX.md` first.
2. Read this file next.
3. Read `.ai/START_HERE.md` for the session protocol.
4. Read `GOVERNANCE.md`, `README.md`, and the closest package README for the requested change.
5. Inspect current source/config/tests before editing. Treat generated onboarding docs as guidance, not source of truth.

## Source-of-truth hierarchy

Use this order when facts conflict:

1. Current source code.
2. Build/test/deployment configuration.
3. CI workflows, if present.
4. Lockfiles and package metadata.
5. Tests.
6. Current README and docs.
7. Older docs/comments/historical notes.
8. Inference.

Always separate verified facts, assumptions, inferences, unknowns, and conflicts in analysis and user-facing summaries.

## Project-specific hard boundaries

- Only `FXDatabase` may access ClickHouse directly. Do not add ClickHouse clients/imports to `FXBacktest`, `FXDataEngine`, `FXPlugins`, `FXImporter`, `FXGUI`, agent packages, or execution contracts.
- Do not move strategy execution or optimization into `FXDatabase`; serve verified data through FXDatabase APIs and run strategies externally.
- Preserve FXDatabase timestamp and price invariants: distinct MT5 server time vs UTC types, raw MT5 timestamps preserved, scaled integer canonical prices, verified broker UTC offsets required, and no current-open-M1 ingestion.
- Keep learned model execution in `FXPlugins`; keep deterministic feature and payload contracts in `FXDataEngine`.
- Keep plugin backend assets plugin-local unless an existing documented shared runtime exception applies.
- Keep demo and live execution isolated. Live paths require promotion evidence, account scope, risk validation, stale-data checks, and kill-switch controls.
- Do not hand-edit generated Offline Lab or runtime promotion artifacts. Rebuild them from Turso/libSQL or approved runtime state.
- Do not weaken `FXGUI` command security policy into a generic shell launcher.

Evidence:
- `GOVERNANCE.md`
- `FXDatabase/README.md`
- `FXPlugins/README.md`
- `FXDataEngine/README.md`
- `FXDemoAgent/README.md`
- `FXLiveAgent/README.md`
- `FXGUI/README.md`

## Planning changes

Before editing:

1. Identify the package(s), API boundary, and governance change class.
2. List source files to inspect before editing.
3. Identify tests or certification commands required by `GOVERNANCE.md`.
4. State risks and unknowns.
5. Prefer the smallest source change that preserves existing contracts.

Do not assume old docs are correct. Confirm with source, package manifests, tests, and config.

## Coding conventions inferred from the repository

- Swift packages use Swift tools 6.3 and `swiftLanguageModes: [.v6]`.
- The repo uses independent top-level SwiftPM packages rather than a root package.
- Package dependencies are local path dependencies between top-level packages.
- Versioned API DTOs and fail-closed validation are core conventions.
- Python tools are under `FXDataEngine/Tools/`; package baseline is in `requirements/fxai-py312.lock`.
- Local runtime state and generated artifacts are intentionally ignored by `.gitignore`.

Evidence:
- `*/Package.swift`
- `CONFIGURATION_SEMANTICS.md`
- `requirements/fxai-py312.lock`
- `.gitignore`

## Validation expectations

Use the smallest validation set that matches the change, then state exactly what ran.

| Change area | Minimum source-grounded validation |
| --- | --- |
| Docs only | `git diff --check`; link and secret checks. |
| Package-local Swift | `swift test --package-path <Package>`; add `swift build --package-path <Package>` if tests do not build all touched targets. |
| FXDatabase data/API/schema | `swift test --package-path FXDatabase`; inspect migration idempotency and API boundary tests. |
| FXDataEngine payload/features | `swift test --package-path FXDataEngine`; affected Python TestLab checks when tool behavior changes. |
| FXPlugins behavior/backend | `swift test --package-path FXPlugins`; set `FXAI_PYTHON` for Python backend paths when required. |
| FXBacktest | `swift test --package-path FXBacktest`. |
| FXGUI | `swift test --package-path FXGUI`; GUI validation suite for layout/snapshot work. |
| Demo/live execution | Package-specific tests plus governance review of risk/kill-switch/account scope. |
| Release/live-readiness claim | `./fxai certify --all`. |

Never claim a check passed unless it actually ran.

## Commit and PR expectations

- Keep docs/onboarding refreshes separate from source-code changes unless explicitly requested.
- Preserve human-authored documentation and conventions.
- Update the nearest README and `GOVERNANCE.md` when operator-facing or governance-relevant behavior changes.
- In PR summaries, include changed files, validation performed, skipped validation with reasons, and residual risks.

## Safety rules for AI agents

- Do not store, print, commit, or echo secrets or private credentials.
- Do not create model/vendor-specific instruction files. Keep AI onboarding generic.
- Do not run destructive commands, production migrations, or deployment commands without explicit instruction and an environment-specific plan.
- Do not invent repository facts. Mark uncertainty and inspect current files.
- Before committing, verify changed files are expected and docs/source boundaries are respected.

## Refresh policy

Refresh `AI_INDEX.md`, this file, and `.ai/*` after meaningful changes to architecture, package manifests, commands, tests, deployment, security, database schema, API boundaries, plugin contracts, or generated-artifact rules. Trust current source over stale onboarding files and repair `.ai/MANIFEST.json` if it becomes invalid.
