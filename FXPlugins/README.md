# FXPlugins

Repository-root plugin source area shared across FXAI runtimes.

AI plugins own model execution. When a converted plugin needs tensor training or inference, it should use a plugin-local PyTorch or TensorFlow backend rather than re-creating the old MQL5 `TensorCore` inside Swift FXDataEngine.

Converted plugins should consume the Swift FXDataEngine OHLCV contracts and use volume-derived features whenever the loaded dataset has nonzero volume.

## Layout

- `Package.swift`: SwiftPM package for converted Swift-era plugins.
- `Sources/FXAIPlugins/`: Swift plugin implementations and adapters that conform to `FXAIPluginV4`.
- `Tests/FXAIPluginsTests/`: focused contract, parity, and acceleration-plan tests.
- family folders such as `Rule/`, `Sequence/`, `Tree/`, and `Stat/`: copied MQL5 reference plugins, kept temporarily for porting parity.
- `PLUGIN_CONVERSION_PLAN.md`: per-plugin Swift/Metal/PyTorch/TensorFlow/Core ML conversion plan and reviewed migration order.

## Current Swift Wave

The first verified wave includes:

- `rule_buyonly`
- `rule_sellonly`
- `rule_random`
- `rule_m1sync`
- `fxbacktest_moving_average_cross`
- `fxbacktest_fxstupid`

Run the local verification gate with:

```bash
swift test
```

The old nested `FXAI/FXPlugins/` tree is still present as a legacy reference until every plugin has either a passing Swift implementation, a backend-owned PyTorch/TensorFlow implementation plan, or an explicit retirement note.
