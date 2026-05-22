# MQL5 FXDataEngine To Swift Port Plan

This plan tracks the remaining non-tensor MQL5 FXDataEngine surface that must move into the repository-root Swift `FXDataEngine` package before the legacy `FXAI/FXDataEngine` tree can be removed.

## Verified Baseline

- Legacy non-tensor surface checked: `FXAI/FXDataEngine`, excluding `TensorCore` and `Tests/TensorCore`.
- Remaining non-tensor MQL5 files: 120 files, about 53k lines.
- Current Swift package surface: OHLCV market contracts, volume-aware feature contracts, normalization payloads, plugin request DTOs, ML backend descriptors, and Metal capability descriptors.
- Explicit exclusion: old MQL5 `TensorCore` is not part of this Swift data-engine parity gate. Tensor training and inference move to FXPlugins through PyTorch or TensorFlow backends.

## Migration Boundaries

- `FXDataEngine` owns canonical M1 OHLCV input, feature construction, normalization, plugin payload contracts, offline context state, runtime policy metadata, and non-trading governance artifacts.
- `FXPlugins` owns model execution, plugin-specific training/inference, PyTorch/TensorFlow process bridges, and plugin state checkpoints.
- `FXBacktest` owns simulated broker execution, parameter sweeps, CPU/Metal pass scheduling, ledger output, and backtest result persistence.
- Live MT5 order APIs are not copied into Swift. Any old trade-execution logic that remains relevant must become deterministic offline policy/risk inputs for FXBacktest or an explicit future live-agent contract.

## Port Workstreams

| Phase | Legacy MQL5 Area | Swift Target | Scope | Exit Gate |
| --- | --- | --- | --- | --- |
| 1 | `Engine/Core`, `feature_registry`, `feature_math`, `feature_build`, `feature_norm` | `FXDataEngine` | Complete OHLCV feature math, MTF alignment helpers, volume replacements for old spread/cost slots, fitted/rolling normalization helpers. | Golden tests on feature indexes, MTF windows, volume on/off behavior, and normalization methods. |
| 2 | `API`, `API/Context`, `API/Contract` | `FXDataEngine` and `FXPlugins` | Plugin manifest/context/replay/quality/persistence DTOs; keep tensor bridge as PyTorch/TensorFlow descriptor only. | Plugin contract validation tests and fixture parity against MQL defaults. |
| 3 | `Runtime/ControlPlane`, `runtime_artifacts` | `FXDataEngine` | Safe token/path helpers, TSV profile parsing, deployment/supervisor/router profiles, snapshot/aggregate DTOs, artifact manifests. | TSV parser tests, clamping/default tests, manifest path tests. |
| 4 | `market_data_gateway`, `data_io`, `data_align`, `data_pipeline`, `engine_samples` | `FXDataEngine` with `FXDatabase` | FXDatabase-backed market loading, M1-only OHLCV assumptions, MTF resampling/alignment, sample windows, labels, context symbols. | No spread/tick dependency; fixture windows match expected labels and volume flags. |
| 5 | `event_macro`, `runtime_calendar_cache`, factor/context state | `FXDataEngine` | Offline macro/news/rates/context state readers and feature adapters. | Stale/unavailable states fail closed and produce deterministic feature payloads. |
| 6 | `Services` probes | `FXDatabase` or `FXDataEngine` readers | Convert MT5 service collectors into offline provider inputs or read-only context-state parsers. | No MT5 runtime dependency; service data can be replayed from files/API fixtures. |
| 7 | `Runtime` model/router/policy stages | `FXDataEngine` | Prediction orchestration, adaptive router, dynamic ensemble, probabilistic calibration, execution-quality policy stage, signal finalize. | Stage-by-stage tests with mock plugins and deterministic policy outputs. |
| 8 | `Runtime/Trade`, `Core/core_broker_execution` | `FXBacktest` plus `FXDataEngine` policy DTOs | Convert exposure, pair-network, risk, execution-quality and lifecycle logic into offline simulation inputs. | FXBacktest integration tests verify ledger/risk behavior without live order APIs. |
| 9 | `Lifecycle`, `Warmup`, `engine_lifecycle`, `engine_warmup` | `FXDataEngine` and agents | Bootstrap, compliance, context symbol selection, reset/recovery, warmup scoring, portfolio warmup. | Lifecycle tests cover symbol/session/regime reset and warmup readiness. |
| 10 | `Tests`, audit/scoring/scenario harness | `FXDataEngine/Tests`, `FXBacktest/Tests` | Rebuild MQL audit suites as Swift fixtures and regression tests. | Every retained legacy behavior has a Swift test or an explicit retirement note. |

## Current Gaps By Module

- `Engine/Core`: partially ported. Core constants, feature groups, plugin families, data request basics, context symbol filtering, timeframe need/lag helpers, aligned-index maps, context aggregate arrays, context aggregate feature-slot adapters, feature schema, normalization concepts, model-context capacity/span helpers, analog-memory/hierarchical signal scoring, regime graph transition scoring, and runtime performance tracking exist in Swift; broker execution remains for FXBacktest migration.
- `Engine` root: partially ported. Feature registry/build/norm exist only as a first slice; runtime artifact text manifests, binary runtime artifact header/envelope persistence, and typed v15 payload section ownership/layout are ported; M1 OHLCV training-sample preparation now covers ascending triple-barrier labels, plugin train payloads, deterministic range-based training datasets, and deterministic horizon configuration helpers. Event macro, broader FXDatabase dataset loading, and meta calibration/reliability/policy/stacker/horizon learned state remain.
- `Runtime/ControlPlane`: partially ported. Swift now has profile/snapshot DTOs, safe path helpers, TSV parsing, adaptive/student router weight helpers, portfolio supervisor defaults, supervisor service/command state parsing, freshness checks, command direction/budget helpers, file-backed profile loaders, peer aggregate scoring, and stale snapshot pruning.
- `Runtime/Trade`: not ported. Must be split between FXBacktest simulation and FXDataEngine policy/risk DTOs.
- `Runtime` stages: not ported. Must be rebuilt with plugin mockability before real PyTorch/TensorFlow backends.
- `Lifecycle` and `Warmup`: not ported. Must be moved after data/sample contracts stabilize.
- `Services`: not ported. MT5 service collectors should become offline data-provider inputs or deterministic readers.
- `Tests`: not ported, except current Swift tests for the first OHLCV/volume contract slice.

## Port Progress

- Done: control-plane foundations: deployment, portfolio supervisor, student router, adaptive router, supervisor service, supervisor command, snapshot DTOs, MQL-style TSV parsing, clamping/defaults, safe paths, and direction/budget helper behavior.
- Done: runtime artifact foundations: runtime artifact paths, version/header contract, runtime stage names, persistence coverage manifest row logic, feature registry manifest rows, feature clip bounds, and stateful checkpoint promotion readiness logic.
- Done: file-backed control-plane repositories and snapshot aggregation with MQL-style stale snapshot pruning.
- Done: runtime artifact text manifest writers for persistence coverage, feature registry, macro dataset stats, runtime performance rows, and shadow fleet ledger rows.
- Done: DataCore request/context helper slice: context symbol normalization/de-duplication, legacy higher-timeframe need/lag calculations, ascending M1 aligned-index maps, and deterministic context aggregate arrays.
- Done: FeatureCore now consumes DataCore context aggregates for feature slots 10-12 and 50-65, including MQL-style raw volatility scaling, signed up-ratio mapping, shared adapter mapping, and previous-frame payload parity.
- Done: binary runtime artifact envelope slice: MQL-order header encode/decode, payload preservation, repository read/write/file-size helpers, and the legacy dirty-save throttle.
- Done: first sample-preparation slice: no-spread M1 OHLCV triple-barrier labels on ascending Swift data, sample quality/weight/path-risk targets, and `TrainRequestV4` payload preparation.
- Done: deterministic training dataset range builder for repeated offline `TrainRequestV4` payload generation from one `MarketUniverse`.
- Done: model-context capacity helpers: horizon clamping, symbol/horizon model scale, sequence/batch/tree span defaults, move EMA state, and three-way split branching.
- Done: deterministic horizon configuration helpers: MQL-style horizon list parsing, configured-horizon slot lookup, max horizon resolution, safe array reads, positive int means, and no-spread static regime bucketing.
- Done: analog-memory core: foundation/student/analog/hierarchical signal DTOs, legacy analog feature vector mapping, no-spread ring-buffer memory updates, weighted analog queries, horizon buckets, and hierarchical score blending.
- Done: regime graph core: transition/dwell state, macro alignment EMA, feedback edge/quality EMA, predicted regime selection, and query score outputs.
- Done: runtime performance core: stage/plugin latency EMA, max/observation tracking, plugin working-set estimates, budget pressure, and performance manifest row generation.
- Done: typed runtime artifact section materialization: v15 binary payload section offsets/byte counts, owner mapping across FXDataEngine/FXPlugins/FXBacktest, prepared replay sample codec, feature-drift codec, and analog-memory codec.
- Remaining in phase 3: complete non-tensor normalization/replay/conformal state behavior; tensor/plugin model state remains owned by FXPlugins, and broker execution state moves to FXBacktest.

## Start Order

1. Port `Runtime/ControlPlane` DTOs, safe-token/path helpers, CSV weight utilities, snapshot parsing, and profile defaults.
2. Port `runtime_artifacts` manifest path helpers and feature/persistence manifest DTOs.
3. Finish `Engine/Core` parity for data requests, context-series aggregation, MTF needs/lags, and feature schema helpers.
4. Finish feature math/build/norm parity using M1 OHLCV and volume, with no spread/tick paths.
5. Add FXDatabase-backed sample/window builders and labels.
6. Add runtime stage DTOs and mock-plugin orchestration.
7. Move trade/risk behavior into FXBacktest integration tests.
8. Convert lifecycle/warmup and audit suites.

## Deletion Gate

The MQL5 `FXAI/FXDataEngine` folder can be removed only when each non-tensor file is mapped to one of these outcomes:

- ported to Swift and covered by tests,
- moved to `FXPlugins` as plugin-owned behavior,
- moved to `FXBacktest` as simulation behavior,
- moved to `FXDatabase` as data-provider behavior,
- retired with an explicit rationale in this document.
