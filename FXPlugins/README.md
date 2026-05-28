# FXPlugins

Repository-root plugin zoo shared across FXAI runtimes.

AI plugins own model execution. When a Swift plugin needs tensor training or inference, it should use a plugin-local PyTorch or TensorFlow backend. FXDataEngine remains responsible for deterministic feature and payload contracts.

Converted plugins should consume the Swift FXDataEngine OHLCV contracts and use volume-derived features whenever the loaded dataset has nonzero volume.

`FXPlugins` does not import FXDatabase, ClickHouse, or FXBacktest database APIs. FXBacktest owns workload scheduling and shared/plugin parameter delivery; plugins receive those values only through the runtime call path after FXDataEngine has built plugin-ready requests.

The shared FXDataEngine/FXPlugins runtime API latest version is `4`, and the tokenizer contract latest version is `fxai-tokenizer-v1`. Plugin manifests, contexts, predictions, and Python accelerator bridge payloads must carry those latest versions. Older versions are rejected instead of being compatibility-shimmed.

## Layout

- `API/`: non-plugin package surface only: registry, tests, docs, and backend process hooks. Shared implementation primitives live in `FXDataEngine`; plugin folders should not depend on shared plugin-zoo helpers.
- `<plugin_id>/`: every plugin lives directly under `FXPlugins` in its own folder named after its manifest `aiName`, for example `lin_sgd/`, `ai_lstm/`, `tree_xgb_fast/`, `rule_m1sync/`, `fxbacktest_moving_average_cross/`, and `fx7/`. All plugin-specific Swift, Metal, Python, model assets, and state adapters belong inside that plugin folder.
- `Package.swift`: SwiftPM boundary for the zoo. There is no longer a root `Sources/`, `Tests/`, or `Python/` staging layout.

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
- Python-backed plugins can use `PythonMLBackendBridge` from `FXDataEngine`.
  The bridge sends OHLCV feature vectors, sequence windows, volume availability,
  horizon, min-move, and price-cost context to plugin-local PyTorch/TensorFlow
  code. The included backend has a pure-Python fallback for contract tests and
  requires PyTorch MPS or TensorFlow Metal acceleration for live accelerator
  runtime paths on Apple Silicon M2/M3-class hosts. Training calls persist lightweight online state under
  `FXAI_PLUGIN_STATE_DIR`, or `~/.fxai/plugins/state` when the environment
  variable is not set. The backend follows the FXDataEngine volume contract:
  volume-derived features are used only when `dataHasVolume` is true.
  FXAI invokes these plugin-local Python backends with `python3`; that command
  must resolve to the Python environment where TensorFlow reports a Metal GPU.
- `FXPluginRuntimeResolver` selects CPU, Metal, PyTorch, TensorFlow, Foundation NLP,
  or CoreML/Neural Engine candidates from each plugin acceleration plan. The
  plugin-local dispatcher `API/Backends/Python/fxai_plugin_module_backend.py`
  loads a plugin's own `PyTorch/`, `TensorFlow/`, or `NLP/` implementation through
  the Swift bridge. `FXAIAcceleratedPluginRuntime` can wrap any planned plugin and
  route train/predict through the selected external backend while retaining explicit
  CPU fallback. Test runs can set `FXAI_FORCE_PYTORCH_CPU=1` or
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
- Current per-plugin reference implementation percentages are tracked in
  `API/Docs/PLUGIN_REFERENCE_IMPLEMENTATION_SCORECARD.md`; the scorecard covers
  every registered plugin and is checked by `ReferenceScorecardTests`.

Run the local verification gate with:

```bash
swift test
```

The current source of truth is this plugin-owned Swift zoo, the reference-grade implementation audit in `API/Docs/PLUGIN_REFERENCE_IMPLEMENTATION_AUDIT.md`, the per-plugin scorecard in `API/Docs/PLUGIN_REFERENCE_IMPLEMENTATION_SCORECARD.md`, and the executable certification gates in the Swift test suite.
