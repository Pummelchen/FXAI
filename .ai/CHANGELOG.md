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
# CHANGELOG.md — AI-onboarding changelog

## 2026-06-25 — bootstrap AI-onboarding generation

Indexed commit: `91a97e92e5622fae867a490dcd917e6c32811955`

Operation mode: `bootstrap`

### Added

- Root `AI_INDEX.md` with repository snapshot, architecture summary, directory map, entrypoints, commands, conventions, risks, unknowns, and recommended reading order.
- Root `AGENTS.md` with generic vendor-neutral operating rules for high-capability AI coding agents.
- `.ai/START_HERE.md` with a pasteable first-session prompt.
- `.ai/PROJECT_MAP.md` with top-level directory, package, dependency, and configuration maps.
- `.ai/ARCHITECTURE.md` with runtime flows and trust boundaries.
- `.ai/COMPONENTS.md` with component cards for major packages.
- `.ai/COMMANDS.md` with install, build, run, test, lab, GUI, and agent command references.
- `.ai/TESTING.md` with validation expectations by change type.
- `.ai/SECURITY.md` with security-sensitive paths and AI-agent safety rules.
- `.ai/PLAYBOOKS.md` with task-specific change workflows.
- `.ai/KNOWN_UNKNOWNS.md` with scan limitations and human-review triggers.
- `.ai/MANIFEST.json` with machine-readable index metadata and refresh policy.
- README AI-onboarding block linking to `AI_INDEX.md`, `AGENTS.md`, and `.ai/START_HERE.md`.

### Source areas used

- Root docs and scripts: `README.md`, `GOVERNANCE.md`, `CONFIGURATION_SEMANTICS.md`, `.gitignore`, `install_fxai.sh`, `fxai`, `requirements/fxai-py312.lock`.
- Package manifests and docs for `FXImporter`, `FXMT5Bridge`, `FXDatabase`, `FXDataEngine`, `FXPlugins`, `FXBacktest`, `FXGUI`, `FXBacktestAgent`, `FXDemoAgent`, `FXLiveAgent`, `FXExecutionContracts`, and `FXTools`.
- Selected entrypoints and command/security sources: `FXDatabase/Sources/FXDatabaseCLI/*`, `FXDataEngine/Tools/*`, `FXBacktest/Sources/FXBacktest/FXBacktestApp.swift`, `FXGUI/Sources/FXGUIApp/FXGUIApp.swift`, `FXGUI/Sources/FXGUICore/Services/FXAICommandSecurityPolicy.swift`, `FXTools/Sources/FXAICertify/main.swift`.

### Migration

- No prior generic AI-onboarding manifest/index was found.
- No existing model-specific AI instruction file was verified.
- No generated model-specific file was created.
- No old file was removed or deprecated.

### Validation notes

- Local Git clone was unavailable because the container could not resolve `github.com`; repository writes used the GitHub connector.
- Generated docs were validated locally for metadata headers, valid manifest JSON, README block uniqueness, generated README links, and absence of obvious GitHub token-value prefixes.
- Full product tests were not run because this is a docs-only bootstrap and no local clone/build environment was available.
