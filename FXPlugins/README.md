# FXPlugins

Repository-root plugin zoo shared across FXAI runtimes.

AI plugins own model execution. When a converted plugin needs tensor training or inference, it should use a plugin-local PyTorch or TensorFlow backend rather than re-creating the old MQL5 `TensorCore` inside Swift FXDataEngine.

Converted plugins should consume the Swift FXDataEngine OHLCV contracts and use volume-derived features whenever the loaded dataset has nonzero volume.

## Layout

- `Rule/`, `Linear/`, `Tree/`, `Sequence/`, `Distribution/`, `Stat/`, `Factor/`, `Trend/`, `Mixture/`, `Memory/`, `World/`, `RL/`: converted plugin zoo families matching the former MQL5 plugin inventory.
- `Demo/`: the two FXBacktest demo/reference adapters connected to the same FXDataEngine plugin contract.
- `Backends/Python/fxai_plugin_backend.py`: generic PyTorch/TensorFlow process backend entrypoint used by Swift adapters until plugin-specific training services are added.
- `Common/`: shared registry, acceleration metadata, generated adapter runtime, tests, and archived conversion plan.
- `Package.swift`: SwiftPM boundary for the zoo. There is no longer a root `Sources/`, `Tests/`, or `Python/` staging layout.

## Current Swift Coverage

Root `FXPlugins` now exposes all 65 FXAI model IDs through Swift `FXAIPluginV4`
contracts:

- 4 hand-ported legacy rule plugins.
- 2 former FXBacktest demo adapters: `fxbacktest_moving_average_cross` and `fxbacktest_fxstupid`.
- 59 generated Swift adapters for the remaining legacy plugins, each with
  volume-aware online centroid learning, deterministic fallback prediction, and explicit Apple Silicon
  backend metadata for Swift SIMD, Accelerate, Metal, PyTorch MPS,
  TensorFlow Metal, or Core ML / Neural Engine candidates.
- Python-backed plugins can use `PythonMLBackendBridge` from `FXDataEngine`.
  The bridge sends OHLCV feature vectors, sequence windows, volume availability,
  horizon, min-move, and price-cost context to plugin-local PyTorch/TensorFlow
  code. The included backend has a pure-Python fallback for contract tests and
  selects PyTorch MPS or TensorFlow Metal acceleration when those frameworks are
  installed. Training calls persist lightweight online state under
  `FXAI_PLUGIN_STATE_DIR`, or `~/.fxai/plugins/state` when the environment
  variable is not set. The backend follows the FXDataEngine volume contract:
  volume-derived features are used only when `dataHasVolume` is true.

Run the local verification gate with:

```bash
swift test
```

The legacy MQL5 plugin reference files have been removed from the repository. The current source of truth is this family-first Swift plugin zoo plus the conversion plan in `Common/Docs/PLUGIN_CONVERSION_PLAN.md`.
