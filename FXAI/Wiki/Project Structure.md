# Project Structure

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

This page maps common user goals to the repository areas that matter.

## Benefits By Role

- Live Trader: know which artifacts explain current live posture.
- Demo Trader: find health and replay outputs without reading the whole codebase.
- Backtester: find runners, reports, and benchmark outputs.
- EA Researcher: find plugins, Offline Lab tools, labels, profiles, and promotion artifacts.
- System Architect: know what is source, what is generated, and what must not be committed.

## Key Areas

| Area | Purpose |
|---|---|
| `FXAI.mq5` | Main MT5 Expert Advisor entry point. |
| `API/` | Plugin contracts, context helpers, and TensorCore bridge surface. |
| `Engine/Core/` | DataCore, FeatureCore, NormalizationCore, and market-data gateway boundaries. |
| `Engine/Runtime/` | Live control-plane stages and trade gating. |
| `Plugins/` | Model and framework plugin implementations. |
| `TensorCore/` | MT5-native neural runtime support. |
| `Tests/` | MT5-side audit and core-runtime runners. |
| `Tools/fxai_testlab.py` | Compile, audit, benchmark, package, and release-gate CLI. |
| `Tools/fxai_offline_lab.py` | Offline Lab, subsystem validation, promotion, and recovery CLI. |
| `Tools/OfflineLab/` | Generated or configured research, report, and subsystem artifacts. |
| `Tools/Benchmarks/` | Published benchmark and promotion evidence. |
| `GUI/` | Optional SwiftUI operator app. |
| `Wiki/` | Versioned user handbook that mirrors the GitHub wiki. |

## Source And Generated Boundaries

- Track source code, manifests, configs, docs, and benchmark reference artifacts.
- Do not commit compiled MT5 `.ex5` binaries.
- Do not commit local `.env`, SQLite databases, cache folders, or machine-specific runtime outputs.
- Publish release binaries through GitHub Releases with SHA-256 and metadata.

## Example Case Scenarios

### Scenario: A live trader wants to know why EURUSD is blocked

Start with:
1. GUI `Live Overview`.
2. Runtime status artifacts.
3. [Runtime Control Plane](Runtime%20Control%20Plane.md).

### Scenario: A researcher wants to add a model

Start with:
1. `API/plugin_contract.mqh`
2. `Plugins/`
3. `Tools/plugin_oracles.json`
4. relevant `Tools/tests/`

### Scenario: A system architect wants to release

Start with:
1. `python3 Tools/fxai_testlab.py verify-all`
2. `python3 Tools/fxai_testlab.py publish-benchmarks --profile bestparams`
3. `python3 Tools/fxai_testlab.py package-mt5-release --version <tag>`

## Next Pages

- [Data Policy](Data%20Policy.md)
- [Benchmarks](Benchmarks.md)
- [Promotion Criteria](Promotion%20Criteria.md)
