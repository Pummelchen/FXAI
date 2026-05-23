# FXPlugins

Repository-root plugin zoo shared across FXAI runtimes.

AI plugins own model execution. When a converted plugin needs tensor training or inference, it should use a plugin-local PyTorch or TensorFlow backend rather than re-creating the old MQL5 `TensorCore` inside Swift FXDataEngine.

Converted plugins should consume the Swift FXDataEngine OHLCV contracts and use volume-derived features whenever the loaded dataset has nonzero volume.

## Layout

- `API/`: non-plugin package surface only: registry, tests, docs, and backend process hooks. Shared implementation primitives live in `FXDataEngine`; plugin folders should not depend on shared plugin-zoo helpers.
- `<plugin_id>/`: every plugin lives directly under `FXPlugins` in its own folder named after its manifest `aiName`, for example `lin_sgd/`, `ai_lstm/`, `tree_xgb_fast/`, `rule_m1sync/`, and `fxbacktest_moving_average_cross/`. All plugin-specific Swift, Metal, Python, model assets, and state adapters belong inside that plugin folder.
- `Package.swift`: SwiftPM boundary for the zoo. There is no longer a root `Sources/`, `Tests/`, or `Python/` staging layout.

## Current Swift Coverage

Root `FXPlugins` now exposes all 65 FXAI model IDs through Swift `FXAIPluginV4`
contracts:

- 4 hand-ported legacy rule plugins.
- 2 former FXBacktest demo adapters: `fxbacktest_moving_average_cross` and `fxbacktest_fxstupid`.
- 59 plugin-owned native conversions with Swift CPU code under each plugin's `CPU/`
  folder and accelerator sources under plugin-owned `Metal/`, `PyTorch/`, `TensorFlow/`,
  or `NLP/` folders where suitable.
- 6 Swift rule/demo adapters. `rule_m1sync` and
  `fxbacktest_moving_average_cross` now include plugin-local Metal batch kernels;
  `rule_buyonly`, `rule_sellonly`, `rule_random`, and `fxbacktest_fxstupid`
  remain scalar-only by design.
- No plugin in `FXPlugins` delegates to `FXAIReferencePluginRuntime`; the old wrapper layer
  has been removed from the plugin zoo.
- Python-backed plugins can use `PythonMLBackendBridge` from `FXDataEngine`.
  The bridge sends OHLCV feature vectors, sequence windows, volume availability,
  horizon, min-move, and price-cost context to plugin-local PyTorch/TensorFlow
  code. The included backend has a pure-Python fallback for contract tests and
  selects PyTorch MPS or TensorFlow Metal acceleration when those frameworks are
  installed. Training calls persist lightweight online state under
  `FXAI_PLUGIN_STATE_DIR`, or `~/.fxai/plugins/state` when the environment
  variable is not set. The backend follows the FXDataEngine volume contract:
  volume-derived features are used only when `dataHasVolume` is true.
- Live accelerator routing is tracked in `API/Docs/PLUGIN_100_LIVE_RUNTIME_COMPLETION_PLAN.md`.
  `FXPluginRuntimeResolver` selects CPU, Metal, PyTorch, TensorFlow, Foundation NLP,
  or CoreML/Neural Engine candidates from each plugin acceleration plan. The
  plugin-local dispatcher `API/Backends/Python/fxai_plugin_module_backend.py`
  loads a plugin's own `PyTorch/`, `TensorFlow/`, or `NLP/` implementation through
  the Swift bridge. `FXAIAcceleratedPluginRuntime` can wrap any planned plugin and
  route train/predict through the selected external backend while retaining explicit
  CPU fallback. Test runs can set `FXAI_FORCE_PYTORCH_CPU=1` for deterministic
  Python smoke tests when Apple MPS is not stable for a specific model.
- The runtime test suite consumes FXDatabase's virtual `SINETEST` security from
  `SineWaveAgent` and checks every plugin on deterministic M1 OHLCV sine-wave data,
  including accelerator runtime selection with CPU fallback for unavailable local
  Metal/PyTorch/TensorFlow/NLP backends.
- Reference-grade Swift fixtures now live inside the owning plugin folders for
  statistical models, factor/trend panel contracts, linear learners, tree
  learners, distribution models, and memory retrieval. The focused suites
  `Wave2StatisticalReferenceTests`, `Wave3FactorTrendReferenceTests`, and
  `Wave4ReferenceParityTests` pin the reference equations and volume-gating
  behavior described in `API/Docs/PLUGIN_99_REFERENCE_IMPLEMENTATION_PLAN.md`.

Run the local verification gate with:

```bash
swift test
```

The legacy MQL5 plugin reference files have been removed from the repository. The current source of truth is this plugin-owned Swift zoo, the full conversion plan in `API/Docs/FULL_PLUGIN_CONVERSION_PLAN.md`, the reference-grade implementation audit in `API/Docs/PLUGIN_REFERENCE_IMPLEMENTATION_AUDIT.md`, the 99 percent reference implementation plan in `API/Docs/PLUGIN_99_REFERENCE_IMPLEMENTATION_PLAN.md`, and the 100 percent live runtime completion plan in `API/Docs/PLUGIN_100_LIVE_RUNTIME_COMPLETION_PLAN.md`.
