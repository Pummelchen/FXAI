# FXPlugins

Repository-root plugin zoo shared across FXAI runtimes.

AI plugins own model execution. When a Swift plugin needs tensor training or inference, it should use a plugin-local PyTorch or TensorFlow backend. Optional inference-only ONNX Runtime and remote RPC backends are also available for exported or externally hosted models. FXDataEngine remains responsible for deterministic feature and payload contracts.

Converted plugins should consume the Swift FXDataEngine OHLCV contracts and use volume-derived features whenever the loaded dataset has nonzero volume.

Plugin certification and accelerator release evidence are part of the root
[FXAI Governance](../GOVERNANCE.md) contract. Changes to plugin behavior,
accelerator runtime paths, checkpoint policy, or SineTest evidence must satisfy
the governance gate for their change class.

`FXPlugins` does not import FXDatabase, ClickHouse, or FXBacktest database APIs. FXBacktest owns workload scheduling and shared/plugin parameter delivery; plugins receive those values only through the runtime call path after FXDataEngine has built plugin-ready requests.

The shared FXDataEngine/FXPlugins runtime API latest version is `4`, and the tokenizer contract latest version is `fxai-tokenizer-v1`. Plugin manifests, contexts, predictions, and Python accelerator bridge payloads must carry those latest versions. Older versions are rejected instead of being compatibility-shimmed.

## Layout

- `API/`: non-plugin package surface only: registry, tests, docs, and backend process hooks. Broad shared implementation primitives live in `FXDataEngine`; plugin folders should not depend on ad hoc shared plugin-zoo helpers.
- `<plugin_id>/`: every plugin lives directly under `FXPlugins` in its own folder named after its manifest `aiName`, for example `lin_sgd/`, `ai_lstm/`, `tree_xgb_fast/`, `rule_m1sync/`, `fxbacktest_moving_average_cross/`, and `fx7/`. All plugin-specific Swift, Metal, Python, model assets, and state adapters belong inside that plugin folder.
- `Package.swift`: SwiftPM boundary for the zoo. It uses static excludes plus SwiftPM's normal source discovery, not manifest-time filesystem scanning. There is no longer a root `Sources/`, `Tests/`, or `Python/` staging layout.

## Plugin Template

Use `demo_plugin_template/` as the top-level starting point for new plugins.
The template is compile-checked, intentionally excluded from
`FXAIPluginRegistry.availablePlugins()`, and covers the full current runtime
surface: Swift CPU/reference, Metal, PyTorch MPS, TensorFlow Metal, Foundation
NLP, ONNX Runtime, and Remote RPC.

The template README documents the copy/rename workflow, required
`PluginManifestV4` fields, CPU fallback expectations, plugin-local Python file
names, ONNX manifest layout, Remote RPC response contract, Python 3.12 runtime
baseline, and verification commands. New production plugins should copy that
folder, keep only the runtimes they actually implement, and add a backend to
`accelerationPlan` only after the implementation and certification evidence
exist.

Runtime entry points are pinned by `PluginBoundaryTests`: PyTorch and
TensorFlow templates expose `predict_batch` and `train_step`; the Foundation NLP
template exposes `merge_into_numeric_features` plus bridge-compatible helper
entry points; ONNX and Remote RPC templates carry schema examples instead of
live model/server code.

## Current Swift Coverage

Root `FXPlugins` now exposes all 66 FXAI model IDs through Swift `FXAIPluginV4`
contracts:

- 4 hand-ported rule plugins.
- 2 former FXBacktest demo adapters: `fxbacktest_moving_average_cross` and `fxbacktest_fxstupid`.
- 1 FXBacktest-native FX7 adapter with plugin-owned backtest source plus Swift/Metal FXDataEngine scoring.
- 59 plugin-owned native conversions with Swift CPU code under each plugin's `CPU/`
  folder and accelerator sources under plugin-owned `Metal/`, `PyTorch/`, `TensorFlow/`,
  or `NLP/` folders where suitable.
- 7 Swift rule/demo/backtest adapters. `rule_m1sync`,
  `fxbacktest_moving_average_cross`, and `fx7` now include plugin-local Metal batch kernels;
  `rule_buyonly`, `rule_sellonly`, `rule_random`, and `fxbacktest_fxstupid`
  remain scalar-only by design.
- No plugin in `FXPlugins` delegates to `FXAIReferencePluginRuntime`; the transitional wrapper layer
  has been removed from the plugin zoo.
- The 23 sequence-architecture CPU adapters that previously duplicated the same
  architecture-switch model body now delegate to
  `ai_autoformer/CPU/FXAISequenceArchitectureCPUModel.swift`; their
  plugin-owned CPU files keep only the per-plugin architecture identity, mode,
  hidden width, and family metadata. `PluginZooLayoutTests` pins this shared
  runtime exception so duplicate architecture-switch bodies are not
  reintroduced.
- Python-backed plugins can use `PythonMLBackendBridge` from `FXDataEngine`.
  The bridge sends OHLCV feature vectors, sequence windows, volume availability,
  horizon, min-move, price-cost context, and financial training targets to
  plugin-local PyTorch/TensorFlow code. The included backend has a pure-Python fallback for contract tests and
  requires PyTorch MPS or TensorFlow Metal acceleration for live accelerator
  runtime paths on Apple Silicon M2/M3-class hosts. Training calls persist lightweight online state under
  `FXAI_PLUGIN_STATE_DIR`, or `~/.fxai/plugins/state` when the environment
  variable is not set. The backend follows the FXDataEngine volume contract:
  volume-derived features are used only when `dataHasVolume` is true.
  Backends that declare `financial_targets` and `financial_loss_config` in
  `train_step` receive `MLFinancialTrainingTargets` and `MLFinancialLossSpec`.
  The shared `fxai_financial_loss.py` helpers implement the preferred hybrid
  utility objective: class CE, move Huber, quantile pinball, adverse-tail
  penalty, cost/path/fill-risk penalty, activity discipline, and downside
  utility. Walk-forward Sharpe/Sortino metrics remain validation and promotion
  gates, not the primary training loss.
  External backend checkpoints are written atomically and paired with
  `fxai_backend_checkpoint_v1` manifest sidecars that record plugin, framework,
  model identifier, stable deterministic seed, backend source hash, state size,
  and state SHA-256. Manifested checkpoints are validated before load; legacy
  checkpoints remain readable unless `FXAI_REQUIRE_CHECKPOINT_MANIFEST=1` is set,
  and `FXAI_REQUIRE_BACKEND_SOURCE_MATCH=1` can require backend source-hash
  parity for stricter replay audits.
  FXAI invokes these plugin-local Python backends with `FXAI_PYTHON` when set,
  otherwise it resolves a Python 3.12 executable. The backend-test baseline is
  pinned by `../requirements/fxai-py312.lock`; TensorFlow Metal requires
  `tensorflow==2.18.1`, `tensorflow-metal==1.2.0`, and at least one TensorFlow
  GPU device reported by `tf.config.list_physical_devices("GPU")`. Tests that
  exercise Python backends resolve that Python 3.12 stack directly and fail with
  diagnostics when it is unavailable.
- ONNX-backed plugins can declare `onnxRuntime` for inference-only exported
  models. The default layout is:

  ```text
  FXPlugins/<plugin_id>/ONNX/<plugin_id>.onnx
  FXPlugins/<plugin_id>/ONNX/<plugin_id>.manifest.json
  ```

  The manifest is optional but, when present, may pin `pluginName`,
  `modelIdentifier`, `modelSha256`, `inputName`, output names, and preferred
  ONNX providers. Runtime overrides are `FXAI_ONNX_MODEL_PATH`,
  `FXAI_ONNX_MANIFEST_PATH`, and `FXAI_ONNX_PROVIDERS`. Enable the backend with
  `FXAI_ENABLE_ONNX_RUNTIME=1` and a configured `FXAI_PYTHON`.
- Remote RPC-backed plugins can declare `remoteRPC` for inference-only external
  inference servers. Enable with `FXAI_ENABLE_REMOTE_RPC=1` and configure
  `FXAI_REMOTE_INFERENCE_ENDPOINT`; optionally set
  `FXAI_REMOTE_INFERENCE_AUTH_TOKEN` and
  `FXAI_REMOTE_INFERENCE_TIMEOUT_SECONDS`. The Swift bridge sends a JSON POST
  request containing the latest `MLInferencePayload` and expects a latest-version
  response with `ok`, optional `error`, and a valid `PredictionV4`. The default
  transport is JSON over HTTP behind `RemoteRPCMLBackendTransport`; gRPC can be
  added later as another transport without changing plugin declarations.
- `FXPluginRuntimeResolver` selects CPU, Metal, PyTorch, TensorFlow, Foundation NLP,
  ONNX Runtime, remote RPC, or CoreML/Neural Engine candidates from each plugin acceleration plan. The
  plugin-local dispatcher `API/Backends/Python/fxai_plugin_module_backend.py`
  loads a plugin's own `PyTorch/`, `TensorFlow/`, `NLP/`, or `ONNX/` implementation through
  the Swift bridge. `FXAIAcceleratedPluginRuntime` can wrap any planned plugin and
  route train/predict through the selected external backend while retaining explicit
  CPU fallback. Runtime wrappers expose a bounded `fallbackDiagnostics` store with
  resolver-level and external-backend fallback events, including operation,
  requested mode, selected backend, CPU fallback backend, policy, sanitized error
  summary, horizon, sequence length, volume availability, and sample timestamp.
  Its shared intrahour direction adapter uses
  `FXAIIntrahourCycleCalibrationPolicy` for confidence and reliability floors so
  minute-of-hour calibration gates are explicit and test-covered. Test runs can
  set `FXAI_FORCE_PYTORCH_CPU=1` or
  `FXAI_ALLOW_CPU_TENSOR_FALLBACK=1` for deterministic smoke tests; production
  accelerator paths require M2/M3-or-newer Apple Silicon GPU support.
- The runtime test suite uses a local FXDataEngine `SINETEST` fixture that mirrors
  FXDatabase's canonical virtual sine-wave security without importing FXDatabase
  into the plugin package. It checks every plugin on deterministic M1 OHLCV
  sine-wave data, including accelerator runtime selection on M2/M3-or-newer
  Apple Silicon and explicit CPU fallback behavior for contract tests.
- `FXAIPluginCertificationRegistry` is the strict 100 percent certification gate.
  It covers every registered plugin and declared accelerator, records satisfied
  runtime evidence, and fails closed until missing gates such as live Metal buffer
  parity, per-plugin PyTorch/TensorFlow/NLP persistence checks, FXDatabase-only
  data-path evidence, full verification output, and golden or standard
  reference parity are proven.
- Reference-grade Swift fixtures now live inside the owning plugin folders for
  statistical models, factor/trend panel contracts, linear learners, tree
  learners, distribution models, and memory retrieval. The focused suites
  `Wave2StatisticalReferenceTests`, `Wave3FactorTrendReferenceTests`, and
  `Wave4ReferenceParityTests` pin the reference equations and volume-gating
  behavior.
- `mix_moe_conformal` uses the shared `ConformalCalibrationPolicy` split-conformal
  contract for calibration diagnostics. The engine exposes finite-sample cutoff,
  prediction-set, and move-interval APIs with sample-count/fallback flags, while
  the plugin CPU, Swift reference, and PyTorch helper use the same conservative
  split-calibration rank rule.
- Current per-plugin reference implementation percentages are tracked in
  `API/Docs/PLUGIN_REFERENCE_IMPLEMENTATION_SCORECARD.md`; the scorecard covers
  every registered plugin and is checked by `ReferenceScorecardTests`.

Run the local verification gate with:

```bash
swift test
```

For focused backend environment verification on Apple Silicon, run:

```bash
FXAI_PYTHON=/opt/homebrew/opt/python@3.12/libexec/bin/python3 swift test --filter PluginRuntimeIntegrationTests
FXAI_PYTHON=/opt/homebrew/opt/python@3.12/libexec/bin/python3 swift test --filter PluginExternalBackendRuntimeTests
```

The current source of truth is this plugin-owned Swift zoo, the reference-grade implementation audit in `API/Docs/PLUGIN_REFERENCE_IMPLEMENTATION_AUDIT.md`, the per-plugin scorecard in `API/Docs/PLUGIN_REFERENCE_IMPLEMENTATION_SCORECARD.md`, and the executable certification gates in the Swift test suite.
