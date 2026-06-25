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
# PLAYBOOKS.md — FXAI task playbooks

## Before any source change

1. Read `AI_INDEX.md`, `AGENTS.md`, `GOVERNANCE.md`, and the nearest package README.
2. Inspect the current files involved; do not rely only on generated docs.
3. Identify the package boundary and governance change class.
4. State verified facts, assumptions, inferences, unknowns, and validation plan.
5. Make the smallest source-grounded change.
6. Run the relevant validation from `.ai/TESTING.md`.

## Change FXDatabase API or schema

1. Read `FXDatabase/README.md`, `FXDatabase/Package.swift`, and the relevant sources under `FXDatabase/Sources/`.
2. For schema changes, inspect `FXDatabase/Migrations/` and `FXDatabase/Sources/ClickHouse/ClickHouseMigrator.swift`.
3. Preserve idempotent migration behavior and the `001_` default database override rule.
4. Preserve direct-ClickHouse access inside `FXDatabase` only.
5. Update DTO/API tests and docs when the public contract changes.
6. Validate with `swift test --package-path FXDatabase`.

Evidence:
- `FXDatabase/README.md`
- `FXDatabase/Sources/ClickHouse/ClickHouseMigrator.swift`
- `GOVERNANCE.md`

## Change feature or plugin payload contracts

1. Start in `FXDataEngine/Sources/FXDataEngine/` and `FXDataEngine/README.md`.
2. Check consumers in `FXPlugins/` and `FXBacktest/` before changing shape/version semantics.
3. Preserve the documented M1 OHLCV and volume-aware contract.
4. Update tests in `FXDataEngine/Tests/` and any affected plugin/backtest tests.
5. Validate with `swift test --package-path FXDataEngine`; add `swift test --package-path FXPlugins` if plugin payloads change.

Evidence:
- `FXDataEngine/README.md`
- `FXPlugins/README.md`

## Add or modify a plugin/backend

1. Read `FXPlugins/README.md` and the target plugin folder.
2. Use `demo_plugin_template/` only as documented; do not register a backend before implementation and evidence exist.
3. Keep plugin-specific Swift, Metal, Python, NLP, ONNX, model assets, and state adapters inside the plugin folder.
4. Update manifests, registry/certification evidence, and tests.
5. Preserve CPU fallback and latest API-version validation.
6. Validate with `swift test --package-path FXPlugins`; set `FXAI_PYTHON` for Python-backed paths when needed.

Evidence:
- `FXPlugins/README.md`
- `FXPlugins/Package.swift`

## Change FXBacktest behavior

1. Read `FXBacktest/README.md`, `FXBacktest/Package.swift`, and relevant `FXBacktest/Sources/` files.
2. Preserve FXDatabase API-only access to history and result persistence.
3. Keep launch-time behavior and resident prompt semantics aligned with the README.
4. For plugin parameters or result persistence, inspect `FXBacktestCore` stores and DTOs.
5. Validate with `swift test --package-path FXBacktest`.

Evidence:
- `FXBacktest/README.md`
- `FXBacktest/Sources/FXBacktest/FXBacktestApp.swift`

## Change GUI workflows

1. Read `FXGUI/README.md`, `FXGUI/Package.swift`, and relevant app/core sources.
2. Keep terminal-first behavior and command preview security.
3. Inspect `FXGUI/Sources/FXGUICore/Services/FXAICommandSecurityPolicy.swift` before changing command handoff.
4. Update GUI docs/checklists when operator workflows change.
5. Validate with `swift test --package-path FXGUI`; run the GUI validation suite for layout/rendering changes.

Evidence:
- `FXGUI/README.md`
- `FXGUI/Docs/FXGUI_RELEASE_CHECKLIST.md`
- `FXGUI/Sources/FXGUICore/Services/FXAICommandSecurityPolicy.swift`

## Change Offline Lab/TestLab tooling

1. Read `FXDataEngine/Tools/OfflineLab/README.md` and relevant Python modules.
2. Preserve Turso/libSQL as authoritative research/promotion state.
3. Do not hand-edit generated artifacts under ignored output directories.
4. For audit/release-gate changes, inspect `FXDataEngine/Tools/testlab/cli.py` and related modules.
5. Validate with `python3.12 FXDataEngine/Tools/fxai_testlab.py verify-all` when environment supports it.

Evidence:
- `FXDataEngine/Tools/OfflineLab/README.md`
- `FXDataEngine/Tools/testlab/cli.py`
- `.gitignore`

## Change demo, production, or execution contracts

1. Read `GOVERNANCE.md`, `FXExecutionContracts/Package.swift`, and the relevant agent README.
2. Preserve package separation and fail-closed validation.
3. Do not add direct ClickHouse access.
4. Update audit/risk/account/kill-switch tests when changing contracts.
5. Validate with the relevant package tests and `swift test --package-path FXExecutionContracts` when contracts change.

Evidence:
- `FXDemoAgent/README.md`
- `FXLiveAgent/README.md`
- `FXExecutionContracts/Package.swift`

## Refresh AI-onboarding docs

1. Read `.ai/MANIFEST.json` and `.ai/CHANGELOG.md`.
2. Compare previous indexed commit to current `main`.
3. Inspect high-impact changes in package manifests, source, tests, docs, security, deployment, and database paths.
4. Preserve correct human edits.
5. Update stale sections only; remove obsolete generated claims.
6. Validate manifest JSON, README block uniqueness, relative links, docs-only diff, and no model-specific AI files.

Evidence:
- `.ai/MANIFEST.json`
- `.ai/CHANGELOG.md`
