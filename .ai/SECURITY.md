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
# SECURITY.md — FXAI AI-agent safety guide

## Summary

FXAI has sensitive boundaries around database access, local configuration, broker/terminal identity, external model backends, generated research artifacts, GUI command handoff, and demo/production execution packages. AI agents must preserve fail-closed behavior and avoid writing private values to tracked files.

Evidence:
- `GOVERNANCE.md`
- `FXDatabase/README.md`
- `FXPlugins/README.md`
- `FXDataEngine/Tools/OfflineLab/README.md`
- `FXGUI/README.md`

## Sensitive paths

| Path / area | Why it matters | Agent rule |
| --- | --- | --- |
| `FXDatabase/Config/` | Local runtime config; ignored by Git. | Do not commit. Use `ConfigSamples/` for examples. |
| `FXDatabase/ConfigSamples/` | Sample database and bridge configuration. | Preserve safe defaults; never add private values. |
| `FXDatabase/Sources/ClickHouse/` | Database authority. | Keep direct database access inside `FXDatabase`. |
| `FXDatabase/Migrations/` | Schema authority. | Keep changes idempotent and auditable. |
| `FXImporter/Connectors/MetaTrader5/EA/FXDatabase.mq5` | External terminal bridge. | Treat account/terminal details as sensitive. |
| `FXMT5Bridge/` | Socket protocol boundary. | Keep DTO/protocol changes explicit and tested. |
| `FXPlugins/` backend folders | External model/backend runtime surfaces. | Validate manifests, hashes, fallback behavior, and runtime gates. |
| `FXDataEngine/Tools/OfflineLab/` | Research/promotion state and generated outputs. | Rebuild generated outputs; do not hand-edit ignored artifacts. |
| `FXGUI/Sources/FXGUICore/Services/FXAICommandSecurityPolicy.swift` | GUI command handoff guard. | Do not broaden to arbitrary shell execution. |
| `FXDemoAgent/`, `FXLiveAgent/`, `FXExecutionContracts/` | Execution and risk boundaries. | Preserve separation, auditability, validation, and kill-switch behavior. |

## Configuration names to recognize

Documented configuration names include `FXAI_PYTHON`, `FXAI_PLUGIN_STATE_DIR`, ONNX/runtime backend configuration variables, Turso/libSQL variables, `FXAI_ENV_FILE`, `FXAI_CONFIG`, `FXAI_TOOLCHAIN_PROFILE`, and ClickHouse password-environment settings. Treat values for these settings as private unless the repository already tracks a sanitized sample.

Evidence:
- `FXPlugins/README.md`
- `FXDataEngine/Tools/OfflineLab/README.md`
- `CONFIGURATION_SEMANTICS.md`
- `FXDatabase/ConfigSamples/clickhouse.sample.json`

## Safety invariants

- Only `FXDatabase` may connect to ClickHouse directly.
- External clients use FXDatabase API contracts.
- Canonical ingestion requires verified broker UTC offset authority.
- The current open M1 bar must not be ingested.
- Demo and production execution paths stay separated.
- Runtime artifacts are rebuilt from authoritative state, not hand-edited.
- GUI command handoff stays curated and project-local.

## Pre-commit checklist

```bash
git status --short
git diff --check
python -m json.tool .ai/MANIFEST.json >/dev/null
find . -path './.git' -prune -o -type f | grep -E '(CLAUDE\.md|copilot-instructions\.md|\.ai/.*(GPT|QWEN|GEMINI|GLM|DEEPSEEK))' || true
```

Also perform a local private-value scan without printing matched values. Confirm generated docs do not contain private credentials, account-specific values, or machine-local runtime data.

## AI-agent rules

- Do not add direct database access outside `FXDatabase`.
- Do not weaken broker time, canonical repair, runtime backend, command handoff, or execution boundary validation.
- Do not commit ignored local runtime or generated output directories.
- Do not replace sanitized samples with real local configuration.
- Do not state that security validation passed unless it was actually performed.

## Unknowns

- Full-repository local secret scanning was unavailable because a local clone could not be created in this environment.
- CI/CD secret handling was not verified during bootstrap.
