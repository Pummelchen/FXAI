# FXPlugins

Repository-root plugin source area shared across FXAI runtimes.

AI plugins own model execution. When a converted plugin needs tensor training or inference, it should use a plugin-local PyTorch or TensorFlow backend rather than re-creating the old MQL5 `TensorCore` inside Swift FXDataEngine.

Converted plugins should consume the Swift FXDataEngine OHLCV contracts and use volume-derived features whenever the loaded dataset has nonzero volume.

## Layout

- `Package.swift`: SwiftPM package for converted Swift-era plugins.
- `Sources/FXAIPlugins/`: Swift plugin implementations and adapters that conform to `FXAIPluginV4`.
- `Tests/FXAIPluginsTests/`: focused contract, parity, and acceleration-plan tests.
- `Python/fxai_plugin_backend.py`: generic PyTorch/TensorFlow process backend entrypoint used by Swift adapters until plugin-specific training services are added.
- family folders such as `Rule/`, `Sequence/`, `Tree/`, and `Stat/`: copied MQL5 reference plugins, kept temporarily for porting parity.
- `PLUGIN_CONVERSION_PLAN.md`: per-plugin Swift/Metal/PyTorch/TensorFlow/Core ML conversion plan and reviewed migration order.

## Current Swift Coverage

Root `FXPlugins` now exposes all 65 FXAI model IDs through Swift `FXAIPluginV4`
contracts:

- 4 hand-ported legacy rule plugins.
- 2 FXBacktest demo adapters: `fxbacktest_moving_average_cross` and `fxbacktest_fxstupid`.
- 59 generated Swift adapters for the remaining legacy plugins, each with
  volume-aware online centroid learning, deterministic fallback prediction, and explicit Apple Silicon
  backend metadata for Swift SIMD, Accelerate, Metal, PyTorch MPS,
  TensorFlow Metal, or Core ML / Neural Engine candidates.
- Python-backed plugins can use `PythonMLBackendBridge` from `FXDataEngine`.
  The bridge sends OHLCV feature vectors, sequence windows, volume availability,
  horizon, min-move, and price-cost context to plugin-local PyTorch/TensorFlow
  code. The included backend has a pure-Python fallback for contract tests and
  selects PyTorch MPS or TensorFlow Metal acceleration when those frameworks are
  installed.

Run the local verification gate with:

```bash
swift test
```

The old nested `FXAI/FXPlugins/` tree is still present as a legacy reference until every plugin has either a passing Swift implementation, a backend-owned PyTorch/TensorFlow implementation plan, or an explicit retirement note.
