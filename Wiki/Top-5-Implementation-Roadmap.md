# FXAI Top 5 Implementation Roadmap

This roadmap turns the five highest-impact project improvements into executable engineering work. It is cross-project by design: no single subproject can finish this alone.

## Goal

Move FXAI from a set of strong Swift packages and plugin infrastructure into one auditable, distributed, Apple Silicon-optimized research and execution system.

The target end state is:

1. `FXBacktest` runs the full root `FXPlugins` zoo through `FXDataEngine` and persists all data, configuration, results, and lineage through `FXDatabase`.
2. A root certification command proves every project, plugin, and accelerator with stored evidence.
3. Every backtest result is reproducible from immutable data and runtime lineage.
4. Mac Minis on the LAN can act as distributed `FXBacktestAgent` workers over TCP.
5. `FXGUI` becomes the practical management surface for the whole FXAI stack: terminal replacement, realtime monitor, run-state monitor, log reporter, plugin tester, certification viewer, and operator control plane.

## Current Gaps Folded Into This Plan

- `FXBacktest` is not yet truly wired into the full root `FXPlugins` zoo. It still has its own backtest plugin layer and only exposes a narrow plugin set there.
- There is no root-level CI/certification command that proves the whole system builds, tests, and certifies all packages/plugins/accelerators together.
- `FXGUI` is not just a side dashboard. It sits above the whole FXAI project and must mirror current terminal features and commands into a productive GUI management tool.
- FXAI needs one unified runtime path from `FXBacktest` through `FXDataEngine` into `FXPlugins` and back into `FXDatabase`.
- FXAI needs a hard certification system that produces auditable evidence, not only passing local tests.
- FXAI needs immutable data lineage for every backtest: source data snapshot, feature graph, normalization state, plugin version, accelerator version, hardware, config, and toolchain.
- `FXBacktestAgent` must become a real distributed worker implementation for Mac Minis in the LAN over TCP.
- Demo/live trading needs a safety architecture: account isolation, risk limits, kill switches, broker adapters, and audit trail.
- FXAI needs performance benchmarks on M2/M3 for CPU, Metal, PyTorch MPS, TensorFlow Metal, and NLP paths.

## Implementation Order

The order matters. The runtime kernel and lineage model must exist before GUI controls, distributed workers, or demo/live safety can be trustworthy.

1. Runtime Kernel
2. Certification and CI Evidence
3. Immutable Lineage
4. Distributed Agents and Trading Safety
5. FXGUI Control Plane and Performance Workbench

## Related Detailed Plans

- Plugin implementation and accelerator certification details live in `FXPlugins/API/Docs/PLUGIN_100_LIVE_RUNTIME_COMPLETION_PLAN.md`.
- GUI implementation details live in `FXGUI/Docs/FXGUI_IMPLEMENTATION_PLAN.md`.
- This roadmap is the cross-project parent plan. The subproject plans should be revised when this roadmap changes public workflows, runtime contracts, or certification gates.

## Track 1: Unified FXAI Runtime Kernel

### Purpose

Create one runtime path used by `FXBacktest`, future agents, tests, and `FXGUI`:

```text
FXBacktest request
  -> FXDatabase historical-data API
  -> FXDataEngine feature/context/label pipeline
  -> FXPlugins registry and backend resolver
  -> CPU / Metal / PyTorch / TensorFlow / NLP backend
  -> FXDatabase result/config/lineage API
```

This removes the current split where `FXBacktest` has its own plugin layer while the real plugin zoo lives under `FXPlugins`.

### Code Targets

- `FXBacktest/Package.swift`
  - Add local dependencies on `FXDataEngine` and `FXPlugins` where SwiftPM cycles allow it.
  - If direct dependency creates a cycle, move shared DTOs into a small dependency-free package target under `FXPlugins/API` or `FXDataEngine/Sources/FXDataEngineContracts`.
- `FXBacktest/Sources/FXBacktestCore/`
  - Add `FXAIRuntimeKernel.swift`.
  - Add `FXAIBacktestPluginAdapter.swift`.
  - Add `FXAIPluginZooRuntime.swift`.
  - Deprecate direct use of `FXBacktestPlugins` except compatibility adapters needed during migration.
- `FXDataEngine/Sources/FXDataEngine/`
  - Add a stable `BacktestFeatureRequest` and `PluginPayloadBuilder` facade.
  - Ensure generated plugin payloads carry API version, symbol, timeframe, OHLCV, feature metadata, normalization state ID, and text/event context.
- `FXPlugins/API/`
  - Expose a stable runtime entrypoint that `FXBacktest` can call without reaching into plugin internals.
  - Return prediction, confidence, decision metadata, backend used, and diagnostics.
- `FXDatabase/Sources/FXDatabaseFXBacktestAPI/`
  - Add runtime endpoints or DTOs for loading canonical historical slices and storing run status/result batches through FXDatabase only.

### Implementation Steps

1. Inventory current `FXBacktest` plugin API and all places where `FXBacktestPlugins` is used.
2. Define the unified runtime request:
   - run ID
   - dataset selector
   - symbol set
   - time range
   - plugin ID
   - requested backend policy
   - shared backtest configuration ID
   - plugin parameter set ID
   - feature graph version
   - lineage manifest ID
3. Define the unified runtime response:
   - prediction direction/value
   - confidence percent
   - plugin diagnostics
   - backend diagnostics
   - trade decision
   - result rows
   - lineage hash
4. Build an adapter from `FXBacktest` OHLC universe types into `FXDataEngine` M1 OHLCV payloads.
5. Build an adapter from `FXPlugins` predictions into `FXBacktest` trade decisions.
6. Move FX7 and the two demo plugins fully behind the same root `FXPlugins` registry path.
7. Remove runtime selection logic that can only see the narrow `FXBacktestPlugins` registry.
8. Add fail-closed behavior when:
   - plugin API version is not latest
   - plugin parameter schema is not registered in FXDatabase
   - requested accelerator is unavailable
   - lineage cannot be produced
   - FXDatabase API is unavailable outside explicit demo mode

### Acceptance Gates

- `FXBacktest` can list every runtime plugin from `FXPlugins`, excluding only the demo template.
- `FXBacktest` can run SineTest through every plugin via `FXDataEngine` payloads.
- `FXBacktestPlugins` is either removed or reduced to compatibility shims with no independent strategy ownership.
- All backtest market data still enters through `FXDatabase` APIs only.
- All plugin configuration and result persistence still goes through `FXDatabase`.
- Direct ClickHouse scanner tests remain green.

## Track 2: Certification And CI Evidence System

### Purpose

Replace informal claims with a hard, repeatable root command that certifies the whole system and stores evidence.

Target command:

```bash
./fxai certify --all
```

or, if implemented as SwiftPM:

```bash
swift run --package-path FXTools FXAICertify --all
```

The command may print terminal output, but the authoritative evidence must be stored through `FXDatabase`.

### Code Targets

- Root `fxai` command or `FXTools/` Swift package.
- `FXDatabase` certification evidence API.
- `FXPlugins/API/Registry/FXAIPluginCertificationRegistry.swift`
  - Extend from plugin-only certification into a component of the root certification run.
- `.github/workflows/`
  - Add build/test/certification workflows for macOS runners where possible.
  - Local Apple Silicon certification remains the source of truth for Metal/MPS/TensorFlow Metal.
- `FXGUI`
  - Later consumes certification evidence from FXDatabase instead of parsing ad hoc files.

### Certification Scope

The root command must run:

1. Swift build and test for:
   - `FXImporter`
   - `FXDatabase`
   - `FXDataEngine`
   - `FXPlugins`
   - `FXBacktest`
   - `FXGUI`
   - future agent packages once implemented
2. API boundary checks:
   - no direct ClickHouse access outside `FXDatabase`
   - latest API versions only
   - no unsupported old DTO versions
3. Data pipeline checks:
   - no feature leakage
   - stable normalization windows
   - volume usage when `volume > 0`
   - no spread dependency in offline backtest contracts
4. Plugin checks:
   - registry coverage
   - SineTest prediction
   - confidence threshold
   - CPU/reference parity
   - plugin parameter schema registration
5. Accelerator checks:
   - Metal compile and live buffer parity
   - PyTorch train/predict/persist/load with MPS preference
   - TensorFlow train/predict/persist/load with TensorFlow Metal preference
   - NLP tokenizer/event/no-text fallback behavior
6. FXBacktest checks:
   - full zoo runtime integration
   - result persistence through FXDatabase
   - no result/config files written to disk for authoritative storage
7. Performance checks:
   - M2/M3 benchmark suite
   - backend timing
   - memory pressure
   - throughput per plugin/backend

### Evidence Model

Each certification run stores:

- certification run ID
- git commit
- working tree cleanliness
- host hardware class
- macOS version
- Xcode version
- Swift version
- Metal device name
- Python version
- PyTorch version and MPS availability
- TensorFlow version and Metal device availability
- package build/test results
- plugin/backend results
- SineTest results
- benchmark summaries
- failure details
- evidence hash

### Acceptance Gates

- One root command can run the whole certification suite.
- The command exits nonzero on any failed required gate.
- Evidence is persisted through FXDatabase.
- The root README and wiki point to the certification command.
- CI runs the build/test subset that can run on hosted macOS.
- Local M2/M3 certification produces the authoritative accelerator evidence.

## Track 3: Immutable Data And Runtime Lineage

### Purpose

Every backtest result must be reproducible. A result without lineage is not a trusted result.

### Code Targets

- `FXDatabase/Migrations/`
  - Add lineage tables.
- `FXDatabase/Sources/FXDatabaseBacktestCore/`
  - Add lineage domain models.
- `FXDatabase/Sources/FXDatabaseFXBacktestAPI/`
  - Add APIs for lineage creation, lookup, and attachment to backtest result batches.
- `FXDataEngine`
  - Emit feature graph hashes, normalization-state hashes, label-policy hashes, and leakage-audit hashes.
- `FXPlugins`
  - Emit plugin manifest hash and accelerator manifest hash.
- `FXBacktest`
  - Require lineage ID before persisted run execution.

### Required Lineage Fields

For every persisted backtest:

- dataset ID
- source provider ID and connector API version
- symbol and timeframe
- data start/end
- source data snapshot hash
- FXDatabase validation status
- SineTest sync status at run time
- FXDataEngine API version
- feature graph hash
- normalization state hash
- label policy hash
- leakage audit hash
- plugin ID
- plugin API version
- plugin code hash
- accelerator backend
- accelerator code hash
- plugin parameter set hash
- shared configuration hash
- FXBacktest runtime kernel version
- Swift version
- Xcode version
- macOS version
- hardware class
- Metal device ID where used
- Python package manifest where used
- run command or GUI action ID
- operator ID when available

### Implementation Steps

1. Define `FXLineageManifest`.
2. Add deterministic hash utilities shared through stable contracts.
3. Add FXDatabase tables:
   - `fxai_lineage_manifest`
   - `fxai_certification_run`
   - `fxai_certification_plugin_result`
   - `fxai_backtest_run_manifest`
   - `fxai_runtime_environment`
4. Add FXDatabase API endpoints:
   - create lineage manifest
   - attach lineage to run
   - fetch lineage by run ID
   - compare two lineage manifests
5. Make `FXBacktest` fail closed when a persisted run cannot create lineage.
6. Add tests proving identical inputs produce identical lineage hashes.
7. Add tests proving changed plugin code, feature graph, config, or accelerator changes the lineage hash.

### Acceptance Gates

- Every persisted backtest result has a lineage ID.
- Backtest result insert without lineage is rejected by FXDatabase.
- Two identical runs produce the same deterministic lineage hash, excluding wall-clock metadata.
- Any meaningful data, config, plugin, accelerator, feature, or toolchain change produces a new hash.
- `FXGUI` can display lineage for a selected run.

## Track 4: Distributed Agents And Trading Safety

### Purpose

Build the future runtime architecture without weakening the database and safety boundaries.

This track has two connected parts:

1. `FXBacktestAgent`: distributed Mac Mini backtest workers over LAN TCP.
2. `FXDemoAgent` and `FXLiveAgent`: safety-gated execution agents for broker/terminal accounts.

### FXBacktestAgent Design

```text
FXBacktest server
  -> job lease over TCP
  -> FXBacktestAgent on Mac Mini
  -> local runtime certification self-check
  -> batch execution
  -> result summary and evidence
  -> FXBacktest server
  -> FXDatabase persistence
```

### Code Targets

- `FXBacktestAgent/Package.swift`
- `FXBacktestAgent/Sources/FXBacktestAgent/`
  - TCP client
  - capability probe
  - certification self-check
  - job lease client
  - heartbeat reporter
  - local runtime executor
- `FXBacktest/Sources/FXBacktestCore/`
  - agent scheduler
  - job leasing
  - batch partitioning
  - result merge
- `FXDatabase`
  - distributed job/result tables and APIs.

### Agent Protocol v1

All messages carry an explicit latest version, for example `fxbacktest.agent.tcp.v1`.

Required messages:

- `hello`
- `capabilities`
- `certificationStatus`
- `leaseRequest`
- `leaseGrant`
- `leaseAck`
- `batchProgress`
- `batchResult`
- `batchFailure`
- `heartbeat`
- `shutdown`

Capability fields:

- CPU core count
- memory
- hardware generation
- Metal device
- PyTorch MPS available
- TensorFlow Metal available
- supported plugin IDs
- supported accelerator backends
- local certification run ID
- current load

### Distributed Safety Rules

- Agents never touch ClickHouse directly.
- Agents never write authoritative result files.
- Agents only receive bounded job leases.
- Expired leases are requeued.
- Duplicate result submission is idempotent.
- Results include lineage and agent certification evidence.
- A worker must pass local SineTest before it receives production work.

### Demo/Live Trading Safety Architecture

`FXDemoAgent` and `FXLiveAgent` must share broker abstractions but have different safety gates.

Required components:

- broker adapter API
- account registry
- account isolation
- execution permission model
- per-account risk limits
- per-symbol risk limits
- max daily loss
- max position size
- max open trades
- max order rate
- kill switch
- dry-run mode
- audit trail
- broker reconnect policy
- order reconciliation
- slippage and rejection logging
- promotion gate from backtest to demo
- promotion gate from demo to live

### Code Targets

- `FXDemoAgent/Package.swift`
- `FXLiveAgent/Package.swift`
- shared safety contracts in a dependency-free target, for example `FXExecutionContracts`
- broker adapters under each agent or a future `FXBrokerAdapters/` package if reuse becomes large enough
- FXDatabase APIs for account/risk/audit state

### Acceptance Gates

- `FXBacktestAgent` can run SineTest jobs from a server over TCP.
- Agent results are persisted only by FXBacktest/FXDatabase, not by the worker directly.
- Job leasing survives worker disconnect and duplicate result submission.
- Demo/live agents cannot place any order without account scope, risk scope, and kill-switch state.
- `FXLiveAgent` has stricter gates than `FXDemoAgent`.
- Every account action has an FXDatabase audit record.

## Track 5: FXGUI Control Plane And Performance Workbench

### Purpose

Make `FXGUI` the useful daily management surface for the entire FXAI project, not a passive artifact browser.

It should mirror the current terminal features and commands while adding live visibility, safer workflows, and fast inspection.

### Required FXGUI Roles

- terminal replacement
- realtime system monitor
- run-state monitor
- log reporter
- plugin tester
- certification viewer
- data health viewer
- FXDatabase API viewer
- backtest builder
- distributed agent monitor
- demo/live safety monitor
- performance benchmark workbench

### Code Targets

- `FXGUI/Sources/FXGUICore/`
  - `CommandCatalog`
  - `CommandExecutionService`
  - `FXDatabaseAPIClient`
  - `FXBacktestAPIClient`
  - `CertificationEvidenceClient`
  - `LineageBrowserService`
  - `AgentFleetService`
  - `BenchmarkResultService`
- `FXGUI/Sources/FXGUIApp/Features/`
  - `CommandCenter`
  - `Certification`
  - `Lineage`
  - `PluginTester`
  - `BacktestRuns`
  - `AgentFleet`
  - `DataHealth`
  - `RiskAndSafety`
  - `Benchmarks`

### Terminal Mirroring

Each terminal command exposed by `FXDatabase`, `FXBacktest`, `FXDataEngine`, `FXPlugins`, and future agents needs a GUI command definition:

- command ID
- owner project
- API version
- display label
- parameters
- validation rules
- generated terminal equivalent
- API execution path when available
- permission level
- destructive action flag
- expected result type
- log stream key

Commands should prefer versioned APIs over shell execution. Shell execution remains a fallback for developer-only workflows until API coverage exists.

### Realtime Monitoring

FXGUI should display:

- FXDatabase API status
- ClickHouse health as reported by FXDatabase
- SineTest sync status
- importer status
- data validation status
- current backtest runs
- plugin certification status
- accelerator availability
- agent fleet status
- demo/live kill switch state
- latest logs
- current failures and blocked gates

### Plugin Tester

The GUI plugin tester must:

- list all registered plugins
- show CPU/Metal/PyTorch/TensorFlow/NLP availability
- run SineTest against selected plugin/backend
- run confidence test
- run benchmark test
- show last certification evidence
- show plugin parameter schema from FXDatabase
- reject execution if API versions are stale

### Performance Workbench

Benchmarks must be designed for Apple Silicon M2/M3 and newer only.

Metrics:

- CPU scalar throughput
- CPU SIMD/Accelerate throughput where applicable
- Metal kernel compile time
- Metal runtime throughput
- PyTorch MPS train/predict time
- TensorFlow Metal train/predict time
- NLP tokenizer/event runtime
- memory use
- warm/cold latency
- batch-size scaling
- parity delta vs CPU/reference

Benchmark result storage:

- authoritative results through FXDatabase
- GUI views through FXDatabase APIs
- optional exported reports generated from stored evidence, never used as source of truth

### Acceptance Gates

- Every important terminal workflow has a GUI equivalent or a documented blocked reason.
- GUI actions show the exact terminal/API equivalent.
- GUI uses versioned APIs where APIs exist.
- GUI can run plugin SineTest and display evidence.
- GUI can inspect lineage for any backtest result.
- GUI can monitor distributed agents once Track 4 exists.
- GUI can display M2/M3 performance benchmark comparisons by plugin/backend.

## Cross-Track Quality Gates

These gates apply to every track.

1. API versions must be explicit and latest-only.
2. No project outside `FXDatabase` may touch ClickHouse directly.
3. No authoritative backtest configuration or result may be written to disk.
4. Every persisted backtest must carry lineage.
5. Every plugin and accelerator must pass SineTest before it can be used in distributed or demo/live workflows.
6. Every accelerator must declare whether it is CPU fallback, Metal, PyTorch MPS, TensorFlow Metal, or NLP runtime.
7. Every failure must be observable in terminal output, certification evidence, and FXGUI.
8. Documentation must be updated in the root README and wiki when public workflows change.

## Milestone Plan

### Milestone 0: Planning Baseline

Deliverables:

- This roadmap exists and is linked from README and wiki.
- Current package graph and runtime gaps are documented.
- Existing plugin completion plan remains referenced for plugin-specific work.

Exit criteria:

- No duplicated or conflicting top-level architecture plan.

### Milestone 1: Runtime Kernel Prototype

Deliverables:

- `FXBacktest` can discover root `FXPlugins`.
- `FXBacktest` can build a `FXDataEngine` payload for SineTest.
- One plugin runs end to end through the new kernel.

Exit criteria:

- No direct ClickHouse access.
- Result is persisted through FXDatabase API.

### Milestone 2: Full Plugin Zoo Runtime

Deliverables:

- Every non-template plugin is visible in `FXBacktest`.
- Every plugin can run CPU SineTest through the unified path.
- Accelerator selection uses the shared backend resolver.

Exit criteria:

- Old narrow `FXBacktestPlugins` ownership is removed or shim-only.

### Milestone 3: Certification Command

Deliverables:

- Root command runs package builds/tests and plugin certification.
- Evidence persisted through FXDatabase.
- CI executes hosted-compatible subset.

Exit criteria:

- Certification failure blocks release.

### Milestone 4: Lineage Enforcement

Deliverables:

- Lineage tables and APIs exist.
- FXBacktest cannot persist results without lineage.
- GUI can display run lineage.

Exit criteria:

- Reproducibility checks pass.

### Milestone 5: Distributed BacktestAgent

Deliverables:

- TCP protocol v1.
- Mac Mini worker can register, self-certify, lease work, run SineTest batch, and return results.
- Scheduler handles disconnect/retry/idempotency.

Exit criteria:

- Two Macs can participate in one certified backtest run without database boundary violations.

### Milestone 6: Demo/Live Safety Foundation

Deliverables:

- Shared execution safety contracts.
- Demo account adapter skeleton.
- Live account safety gate skeleton.
- Kill switch and audit trail in FXDatabase.

Exit criteria:

- No order-capable workflow can bypass account/risk/kill-switch checks.

### Milestone 7: FXGUI Control Plane

Deliverables:

- Command catalog mirrors terminal workflows.
- Certification, lineage, plugin tester, data health, benchmark, agent, and risk views exist.
- GUI actions use APIs first and shell fallback only when marked developer-only.

Exit criteria:

- A normal operator can manage daily FXAI workflows from FXGUI without manually typing common terminal commands.

### Milestone 8: M2/M3 Benchmark And Optimization Pass

Deliverables:

- Benchmark suite for CPU, Metal, PyTorch MPS, TensorFlow Metal, and NLP.
- Results stored in FXDatabase.
- FXGUI benchmark dashboard.
- Optimization tickets generated for slow or low-confidence backends.

Exit criteria:

- Every plugin/backend has benchmark evidence or a documented reason why the backend is not applicable.

## Final Definition Of Done

The Top 5 improvements are complete only when:

1. `FXBacktest` uses the real root `FXPlugins` zoo through `FXDataEngine`.
2. The root certification command passes and stores evidence through `FXDatabase`.
3. Every persisted run has immutable lineage.
4. `FXBacktestAgent` can distribute certified backtest work to LAN Mac Minis.
5. `FXDemoAgent` and `FXLiveAgent` have account isolation, risk limits, kill switches, broker adapters, and audit trails before any order-capable workflow.
6. `FXGUI` mirrors terminal workflows and shows live state, logs, certification, plugin tests, lineage, agents, and benchmarks.
7. M2/M3 performance evidence exists for CPU, Metal, PyTorch MPS, TensorFlow Metal, and NLP paths.
