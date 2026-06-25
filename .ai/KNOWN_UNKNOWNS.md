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
# KNOWN_UNKNOWNS.md — FXAI onboarding uncertainties

## Bootstrap scan limitations

| Item | Status | Why it matters | Recommended follow-up |
| --- | --- | --- | --- |
| Local clone unavailable | known limitation | The container could not resolve `github.com`, so full local `git status`, recursive tree inspection, and repo-wide scans could not be run locally. | Run local validation after checkout. |
| Full branch inventory | unknown | Existing branch names beyond connector-visible checks were not exhaustively listed. | Run `git branch -r` or GitHub branch list locally. |
| Complete CI/CD inventory | unknown | No workflows were verified during bootstrap search. | Inspect `.github/workflows/` if present after local checkout. |
| Docker/container setup | unknown/not detected | Search did not surface Dockerfile or compose files, but full tree enumeration was unavailable. | Run `find . -maxdepth 3 -iname '*docker*' -o -name 'docker-compose.yml'`. |
| Complete SQL migration list | unknown | `FXDatabase/Migrations/` was referenced, but not fully enumerated. | Inspect migration files before schema work. |
| Complete plugin folder inventory | unknown | `FXPlugins` contains many plugin folders and excluded backend paths; not exhaustively enumerated. | Inspect target plugin folders and registry/tests before plugin work. |
| Production-readiness claim | needs_human_review | README claim was preserved but certification/tests were not rerun here. | Run `./fxai certify --all` in the intended macOS environment. |
| Full secret scan | degraded | Generated docs were locally scanned for obvious token-value prefixes; full repo scan was unavailable without clone. | Run an organization-approved secret scanner locally before merging. |

## No confirmed conflicts

No code/documentation conflicts were confirmed during this bootstrap scan. If future work finds a conflict, trust current source/config/tests over generated onboarding files and record the issue here during refresh.

## Model-specific AI files

No existing generated AI-onboarding structure or model-specific instruction file was verified during bootstrap. Known checked paths/searches included `AI_INDEX.md`, `AGENTS.md`, `.ai/MANIFEST.json`, root model-specific instruction paths, and keyword search for model/vendor-specific filenames.

If future runs find model-specific files, migrate useful vendor-neutral content into the generic onboarding set, preserve human-authored content for review, and do not create new model/vendor-specific files.

## Areas where an AI should ask a human before editing

- Any change that enables production execution, changes account scope, changes kill-switch behavior, or changes risk limits.
- Any ClickHouse schema or canonical repair change where data deletion/rewrite safety is unclear.
- Any broker time offset rule that is not already code-owned and verified.
- Any generated Offline Lab/runtime artifact that appears stale but is not clearly rebuildable from documented commands.
- Any private endpoint, account, or credential configuration.
- Any compatibility decision that would keep older API versions active.

## Stale facts removed during bootstrap

None. This was the initial generic AI-onboarding generation.
