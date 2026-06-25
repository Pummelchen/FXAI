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
# TESTING.md — FXAI testing guide

## Test frameworks and structure

| Area | Framework / runner | Structure | Evidence |
| --- | --- | --- | --- |
| Swift packages | SwiftPM/XCTest-style package tests | Each top-level package has its own `Package.swift`; most declare `testTarget`s. | `*/Package.swift` |
| Python tooling | pytest and custom Python CLIs | `FXDataEngine/Tools/tests/`, `FXDataEngine/Tools/testlab/` | `requirements/fxai-py312.lock`, `FXDataEngine/Tools/testlab/cli.py` |
| Root certification | Swift executable `FXAICertify` | Builds/tests configured package list and records evidence. | `FXTools/Sources/FXAICertify/main.swift` |
| GUI validation | Swift tests plus GUI validation script | GUI snapshots/artifacts under ignored `FXGUI/Artifacts/`. | `FXGUI/README.md`, `FXGUI/Docs/FXGUI_RELEASE_CHECKLIST.md` |
| Governance gates | Change-class-specific commands | Required evidence listed by change class. | `GOVERNANCE.md` |

## Run all major checks

```bash
./fxai certify --all
```

This is the documented root release/live-readiness gate. It builds and tests the package list in `FXTools/Sources/FXAICertify/main.swift` and captures environment evidence.

For build-only certification:

```bash
./fxai certify --build-only
```

## Package tests

```bash
swift test --package-path FXImporter
swift test --package-path FXDatabase
swift test --package-path FXDataEngine
swift test --package-path FXPlugins
swift test --package-path FXBacktest
swift test --package-path FXGUI
swift test --package-path FXBacktestAgent
swift test --package-path FXDemoAgent
swift test --package-path FXLiveAgent
swift test --package-path FXExecutionContracts
swift test --package-path FXTools
```

`FXMT5Bridge` did not have a standalone test target in its `Package.swift` during bootstrap; related bridge tests are declared in `FXImporter`.

## Focused Swift tests

SwiftPM supports filtered tests. Verified repo examples include:

```bash
swift test --package-path FXPlugins --filter PluginRuntimeIntegrationTests
swift test --package-path FXPlugins --filter PluginExternalBackendRuntimeTests
swift test --package-path FXBacktest --filter FXBacktestPluginZooBridgeTests
```

Use the package-specific test target nearest your change. Do not invent test names; inspect `Tests/` first.

## Python/TestLab checks

```bash
python3.12 FXDataEngine/Tools/fxai_testlab.py doctor
python3.12 FXDataEngine/Tools/fxai_testlab.py verify-all
python3.12 FXDataEngine/Tools/fxai_testlab.py run-audit --scenario-list "{market_walkforward, market_trend, market_chop}" --wf-train-years 1 --wf-test-years 0.25 --wf-window-mode rolling
python3.12 FXDataEngine/Tools/fxai_testlab.py walkforward-analyze
```

`verify-all` is documented as a one-command platform verification path for Python tests, deterministic fixture checks, and clean Swift builds.

Evidence:
- `FXDataEngine/README.md`
- `FXDataEngine/Tools/OfflineLab/README.md`
- `FXDataEngine/Tools/testlab/cli.py`

## GUI validation

```bash
cd FXGUI
swift test
./Tools/run_gui_validation_suite.sh
```

The GUI validation suite exports screenshots to ignored paths under `FXGUI/Artifacts/`. Use this for layout, rendering, or operator surface changes.

Evidence:
- `FXGUI/README.md`
- `FXGUI/Docs/FXGUI_RELEASE_CHECKLIST.md`
- `.gitignore`

## Minimum validation by change type

| Change type | Minimum expected validation | Notes |
| --- | --- | --- |
| Documentation only | `git diff --check`; link/secret review. | Do not claim product tests ran unless they did. |
| FXDatabase source/schema/API | `swift test --package-path FXDatabase`. | Also inspect migrations and API DTO tests when touched. |
| FXDataEngine features/payloads | `swift test --package-path FXDataEngine`. | Add TestLab validation for Python/lab impact. |
| FXPlugins plugin/backend | `swift test --package-path FXPlugins`. | Set `FXAI_PYTHON` for Python backend coverage where relevant. |
| FXBacktest | `swift test --package-path FXBacktest`. | Include API/result persistence tests when touching storage paths. |
| FXGUI | `swift test --package-path FXGUI`; GUI validation suite for layout work. | Do not commit generated screenshots/artifacts. |
| Agent/execution contracts | Relevant package tests plus `swift test --package-path FXExecutionContracts` when contracts change. | Preserve fail-closed behavior. |
| Release claim | `./fxai certify --all`. | Required by governance for release/live-readiness claims. |

## Fixtures, generated outputs, and flakes

Verified generated/local output exclusions:

- SwiftPM state: `*/.build/`, `*/.swiftpm/`, `*/Package.resolved`.
- Python caches: `__pycache__/`, `*.pyc`, pytest caches.
- Offline Lab DB/state/report/output folders under `FXDataEngine/Tools/OfflineLab/`.
- GUI artifacts and screenshots under `FXGUI/Artifacts/`.
- FXDatabase local config/logs under `FXDatabase/Config/` and `FXDatabase/Logs/`.

No specific flaky tests were verified during bootstrap. If a test is slow/flaky, record exact command, environment, failure mode, and evidence in the relevant README or `.ai/KNOWN_UNKNOWNS.md` during refresh.

## Reporting validation

When reporting results, use this format:

```text
Validation run:
- <command>: passed/failed/skipped
- <command>: passed/failed/skipped

Skipped:
- <command>: skipped because <reason>
```

Never write “tests pass” unless the tests actually ran in the current environment.
