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
# START_HERE.md — first-session prompt for FXAI

Use this prompt at the beginning of a fresh AI coding session in this repository.

```text
You are working in the FXAI repository. Start by reading, in order:

1. AI_INDEX.md
2. AGENTS.md
3. README.md
4. GOVERNANCE.md
5. .ai/PROJECT_MAP.md
6. .ai/ARCHITECTURE.md
7. .ai/COMMANDS.md and .ai/TESTING.md
8. .ai/SECURITY.md
9. The README, Package.swift, source files, and tests nearest the requested change

Before editing, summarize your understanding of:
- the package or component being changed
- the relevant data/API/security boundary
- the source files you inspected
- verified facts, assumptions, inferences, unknowns, and conflicts
- the smallest safe implementation plan
- the validation commands you expect to run

Rules:
- Treat current source/config/tests as authoritative over generated onboarding docs.
- Do not modify source code until you have inspected the current files involved.
- Do not invent repository facts. Mark unknowns explicitly.
- Preserve the rule that only FXDatabase touches ClickHouse directly.
- Keep FXDatabase as a data/API authority, not a strategy or optimization runner.
- Keep FXDataEngine deterministic and keep learned model execution in FXPlugins.
- Keep demo/live execution separated and fail-closed.
- Do not hand-edit generated Offline Lab/runtime artifacts.
- Do not create model-specific AI instruction files. Keep AI guidance vendor-neutral.
- Do not store or print credentials, tokens, account identifiers, or private secrets.

When implementing:
- make the smallest source-grounded change that satisfies the task
- update nearby docs when operator-facing or governance behavior changes
- add or update tests where behavior changes
- run the narrowest meaningful validation command first
- report changed files and exact validation results
- state skipped checks and why they were skipped

Avoid overloading context. Read broad onboarding first, then focus on the packages and source files directly relevant to the task.
```

## Quick local anchors

- Repo map: `AI_INDEX.md`
- Agent rules: `AGENTS.md`
- Architecture: `.ai/ARCHITECTURE.md`
- Component cards: `.ai/COMPONENTS.md`
- Commands: `.ai/COMMANDS.md`
- Testing: `.ai/TESTING.md`
- Security: `.ai/SECURITY.md`
- Known unknowns: `.ai/KNOWN_UNKNOWNS.md`
- Machine-readable manifest: `.ai/MANIFEST.json`

## Evidence

- `README.md`
- `GOVERNANCE.md`
- `FXDatabase/README.md`
- `FXPlugins/README.md`
- `FXGUI/README.md`
